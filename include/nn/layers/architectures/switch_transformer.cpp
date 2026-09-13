// ============================================================================
// Switch Transformer — Fedus, Zoph, Shazeer 2022
// ============================================================================
//
// Implementation of SwitchMoELayer, SwitchTransformerBlock, and
// SwitchTransformerModel. See `switch_transformer.h` for the architectural
// overview and paper references. This file follows the same conventions as
// `sparse_moe.cpp` (raw Tensor params, He expert init, Xavier router init,
// capacity-limited dispatch) plus the Switch-Transformer-specific additions:
// hard top-1 routing, capacity-factor dropping, and the router z-loss.
// ============================================================================

#include "switch_transformer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// SwitchMoELayer construction
// ----------------------------------------------------------------------------

SwitchMoELayer::SwitchMoELayer(size_t d_model, size_t num_experts,
                               size_t expert_hidden,
                               double capacity_factor,
                               double aux_loss_coef,
                               double z_loss_coef)
    : d_model_(d_model),
      num_experts_(num_experts),
      expert_hidden_(expert_hidden == 0 ? 4 * d_model : expert_hidden),
      capacity_factor_(capacity_factor),
      aux_loss_coef_(aux_loss_coef),
      z_loss_coef_(z_loss_coef),
      W_router_(num_experts, d_model),
      b_router_(1, num_experts),
      grad_W_router_(num_experts, d_model),
      grad_b_router_(1, num_experts),
      load_balance_loss_(0.0),
      z_loss_(0.0)
{
    if (d_model_ == 0) throw std::invalid_argument("SwitchMoELayer: d_model must be > 0");
    if (num_experts_ == 0) throw std::invalid_argument("SwitchMoELayer: num_experts must be > 0");
    if (capacity_factor_ <= 0.0) throw std::invalid_argument("SwitchMoELayer: capacity_factor must be > 0");
    if (aux_loss_coef_ < 0.0) throw std::invalid_argument("SwitchMoELayer: aux_loss_coef must be >= 0");
    if (z_loss_coef_ < 0.0) throw std::invalid_argument("SwitchMoELayer: z_loss_coef must be >= 0");

    // Xavier init for the router: std = sqrt(2 / (num_experts + d_model)).
    {
        std::mt19937 gen(42);
        double std_r = std::sqrt(2.0 / static_cast<double>(num_experts_ + d_model_));
        std::normal_distribution<> dis(0.0, std_r);
        for (size_t i = 0; i < W_router_.rows; ++i)
            for (size_t j = 0; j < W_router_.cols; ++j)
                W_router_(i, j) = dis(gen);
        b_router_.fill(0.0);
    }

    // He init for experts (good for ReLU): std = sqrt(2 / fan_in).
    W1_.reserve(num_experts_);
    W2_.reserve(num_experts_);
    b1_.reserve(num_experts_);
    b2_.reserve(num_experts_);
    gW1_.reserve(num_experts_);
    gW2_.reserve(num_experts_);
    gb1_.reserve(num_experts_);
    gb2_.reserve(num_experts_);

    std::mt19937 gen(43);
    double std_w1 = std::sqrt(2.0 / static_cast<double>(d_model_));
    double std_w2 = std::sqrt(2.0 / static_cast<double>(expert_hidden_));
    std::normal_distribution<> dis_w1(0.0, std_w1);
    std::normal_distribution<> dis_w2(0.0, std_w2);

    for (size_t e = 0; e < num_experts_; ++e) {
        Tensor W1(expert_hidden_, d_model_);
        Tensor W2(d_model_, expert_hidden_);
        Tensor b1v(1, expert_hidden_);
        Tensor b2v(1, d_model_);
        for (size_t i = 0; i < W1.rows; ++i)
            for (size_t j = 0; j < W1.cols; ++j)
                W1(i, j) = dis_w1(gen);
        for (size_t i = 0; i < W2.rows; ++i)
            for (size_t j = 0; j < W2.cols; ++j)
                W2(i, j) = dis_w2(gen);
        b1v.fill(0.0);
        b2v.fill(0.0);

        W1_.push_back(W1);
        W2_.push_back(W2);
        b1_.push_back(b1v);
        b2_.push_back(b2v);

        gW1_.push_back(Tensor(expert_hidden_, d_model_));
        gW2_.push_back(Tensor(d_model_, expert_hidden_));
        gb1_.push_back(Tensor(1, expert_hidden_));
        gb2_.push_back(Tensor(1, d_model_));
        gW1_.back().fill(0.0);
        gW2_.back().fill(0.0);
        gb1_.back().fill(0.0);
        gb2_.back().fill(0.0);
    }

    dispatch_frac_.assign(num_experts_, 0.0);
    mean_router_prob_.assign(num_experts_, 0.0);
    expert_count_.assign(num_experts_, 0);
}


// ----------------------------------------------------------------------------
// SwitchMoELayer forward
// ----------------------------------------------------------------------------
//
// Per-token hard top-1 routing with capacity-factor dropping.
//
//   1. Compute gate_logits[b, e] = sum_d input[b, d] · W_router[e, d] + b_router[0, e]
//      (Dense convention; one row of W per expert).
//
//   2. Compute the FULL softmax gate_probs[b, e] = softmax_e(gate_logits[b, :])
//      — we need the full softmax (not just top-1) for the router backward.
//
//   3. For each token b: find top1_idx[b] = argmax_e gate_logits[b, :].
//      If expert_count[top1] < capacity_per_expert, dispatch to that expert;
//      else drop the token (output[b, :] = 0, grad path also zeroed).
//
//   4. Capacity per expert: cap_e = ceil(B / num_experts) · capacity_factor.
//
//   5. For each dispatched token, run the expert FFN:
//          h = ReLU(W1 · x + b1)
//          y = W2 · h + b2
//      and set out[b, :] = gate_probs[b, top1] · y.
//
//   6. Aux + z losses are computed in `compute_losses_()` and cached.
// ----------------------------------------------------------------------------

Tensor SwitchMoELayer::forward(const Tensor& input) {
    const size_t B = input.rows;
    input_ = input;

    // Step 1: gate_logits (B, num_experts)
    gate_logits_ = Tensor(B, num_experts_);
    for (size_t b = 0; b < B; ++b)
        for (size_t e = 0; e < num_experts_; ++e) {
            double s = b_router_(0, e);
            for (size_t d = 0; d < d_model_; ++d)
                s += input(b, d) * W_router_(e, d);
            gate_logits_(b, e) = s;
        }

    // Step 2: full softmax gate_probs (numerically stable: subtract per-row max).
    gate_probs_ = Tensor(B, num_experts_);
    for (size_t b = 0; b < B; ++b) {
        double maxv = gate_logits_(b, 0);
        for (size_t e = 1; e < num_experts_; ++e)
            if (gate_logits_(b, e) > maxv) maxv = gate_logits_(b, e);
        double sumexp = 0.0;
        for (size_t e = 0; e < num_experts_; ++e) {
            gate_probs_(b, e) = std::exp(gate_logits_(b, e) - maxv);
            sumexp += gate_probs_(b, e);
        }
        for (size_t e = 0; e < num_experts_; ++e)
            gate_probs_(b, e) /= sumexp;
    }

    // Step 3-4: top-1 + capacity dispatch bookkeeping.
    top1_idx_ = Tensor(B, 1);
    dispatch_mask_ = Tensor::zeros(B, num_experts_);
    token_dropped_ = Tensor::zeros(B, 1);

    const size_t uniform = (B + num_experts_ - 1) / num_experts_;  // ceil(B/N)
    const size_t cap_per = static_cast<size_t>(std::ceil(
        static_cast<double>(uniform) * capacity_factor_));
    // Cap at least 1 (else capacity_factor too small to be useful)
    const size_t cap_e = std::max<size_t>(1, cap_per);

    // Re-allocate per-expert caches of size cap_e.
    expert_h_pre_.clear();
    expert_h_act_.clear();
    expert_x_in_.clear();
    std::fill(expert_count_.begin(), expert_count_.end(), 0);
    for (size_t e = 0; e < num_experts_; ++e) {
        expert_h_pre_.push_back(Tensor(cap_e, expert_hidden_));
        expert_h_act_.push_back(Tensor(cap_e, expert_hidden_));
        expert_x_in_.push_back(Tensor(cap_e, d_model_));
    }

    // Walk tokens in order, dispatch up to cap_e to each expert.
    for (size_t b = 0; b < B; ++b) {
        // argmax over the row.
        size_t best_e = 0;
        double best_v = gate_logits_(b, 0);
        for (size_t e = 1; e < num_experts_; ++e) {
            if (gate_logits_(b, e) > best_v) {
                best_v = gate_logits_(b, e);
                best_e = e;
            }
        }
        top1_idx_(b, 0) = static_cast<double>(best_e);
        if (expert_count_[best_e] < cap_e) {
            size_t slot = expert_count_[best_e];
            dispatch_mask_(b, best_e) = 1.0;
            // Copy input row into the per-expert cache at slot.
            for (size_t d = 0; d < d_model_; ++d)
                expert_x_in_[best_e](slot, d) = input(b, d);
            expert_count_[best_e] += 1;
        } else {
            token_dropped_(b, 0) = 1.0;
        }
    }

    // Step 5: per-expert forward (only on the dispatched slots).
    Tensor output = Tensor::zeros(B, d_model_);
    for (size_t e = 0; e < num_experts_; ++e) {
        const size_t ne = expert_count_[e];
        if (ne == 0) continue;

        // h_pre[slot, h] = sum_d W1[e](h, d) · x_in[slot, d] + b1[e](0, h)
        Tensor h_pre(ne, expert_hidden_);
        for (size_t i = 0; i < ne; ++i) {
            for (size_t h = 0; h < expert_hidden_; ++h) {
                double s = b1_[e](0, h);
                for (size_t d = 0; d < d_model_; ++d)
                    s += W1_[e](h, d) * expert_x_in_[e](i, d);
                h_pre(i, h) = s;
            }
        }
        expert_h_pre_[e] = Tensor(cap_e, expert_hidden_);
        expert_h_pre_[e].fill(0.0);
        for (size_t i = 0; i < ne; ++i)
            for (size_t h = 0; h < expert_hidden_; ++h)
                expert_h_pre_[e](i, h) = h_pre(i, h);

        // ReLU
        expert_h_act_[e] = Tensor(cap_e, expert_hidden_);
        expert_h_act_[e].fill(0.0);
        for (size_t i = 0; i < ne; ++i)
            for (size_t h = 0; h < expert_hidden_; ++h) {
                double v = h_pre(i, h);
                expert_h_act_[e](i, h) = v > 0.0 ? v : 0.0;
            }

        // y_out[slot, d] = sum_h W2[e](d, h) · h_act[slot, h] + b2[e](0, d)
        Tensor y_out(ne, d_model_);
        for (size_t i = 0; i < ne; ++i) {
            for (size_t d = 0; d < d_model_; ++d) {
                double s = b2_[e](0, d);
                for (size_t h = 0; h < expert_hidden_; ++h)
                    s += W2_[e](d, h) * expert_h_act_[e](i, h);
                y_out(i, d) = s;
            }
        }

        // Scatter back into output[b, :] = gate_probs[b, e] · y_out[slot, :]
        size_t slot = 0;
        for (size_t b = 0; b < B; ++b) {
            if (static_cast<size_t>(top1_idx_(b, 0)) != e) continue;
            if (dispatch_mask_(b, e) < 0.5) continue;
            double w = gate_probs_(b, e);
            for (size_t d = 0; d < d_model_; ++d)
                output(b, d) = w * y_out(slot, d);
            ++slot;
        }
    }

    // Step 6: aux + z losses.
    compute_losses_();

    return output;
}


// ----------------------------------------------------------------------------
// SwitchMoELayer backward
// ----------------------------------------------------------------------------
//
// Backward chain:
//   (a) Per-expert gradient:
//       For each dispatched token (b, e) at slot s_e:
//         d_expert_out[s_e, d] = gate_probs[b, e] · grad_output[b, d]
//       Then per-expert backward: 2-layer FFN with ReLU.
//         d_h_pre[s_e, h] = relu'(h_pre) · sum_d W2[d, h] · d_expert_out[s_e, d]
//         dW2[d, h] += d_expert_out[s_e, d] · h_act[s_e, h]
//         db2[0, d] += d_expert_out[s_e, d]
//         dx_in[s_e, d] = sum_h W1[h, d] · d_h_pre[s_e, h]
//         dW1[h, d] += d_h_pre[s_e, h] · x_in[s_e, d]
//         db1[0, h] += d_h_pre[s_e, h]
//
//   (b) Router gradient:
//       d_router_prob[b, e] = sum_d grad_output[b, d] · expert_out[b, d]   (if dispatched)
//       For all (b, e) (including unselected), d_logits[b, e] = gate_probs[b, e] ·
//         (d_router_prob[b, e] - sum_e' gate_probs[b, e'] · d_router_prob[b, e']).
//       Then add z-loss chain: d_logits[b, e] += (2 · z_coef / B) · gate_probs[b, e]
//         (since d/dz_e log(Σ exp²) = 2·exp(2z_e) / Σ exp² = 2·softmax_e(2z),
//          but since we're regularising gate_probs (the softmax of z), and
//          ∂log(Σ exp²)/∂softmax_e = 1 by the chain rule (because softmax_e
//          is already normalised), the gradient flows directly).
//       Actually: L_z = (z_coef / B) · Σ_b log(Σ_e exp(2·z_e)).
//         ∂L_z / ∂z_e = (z_coef / B) · (2·exp(2z_e)) / (Σ_e' exp(2z_e')).
//       We implement that directly below.
//
//   (c) d_input[b, d] = sum_e d_logits[b, e] · W_router[e, d].
// ----------------------------------------------------------------------------

Tensor SwitchMoELayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t B = input_.rows;

    // Zero grads.
    grad_W_router_.fill(0.0);
    grad_b_router_.fill(0.0);
    for (size_t e = 0; e < num_experts_; ++e) {
        gW1_[e].fill(0.0);
        gW2_[e].fill(0.0);
        gb1_[e].fill(0.0);
        gb2_[e].fill(0.0);
    }

    // ---------- (a) Per-expert gradient ----------
    // For each expert e, gather the grad_output rows that were dispatched to e.
    Tensor grad_input = Tensor::zeros(B, d_model_);

    for (size_t e = 0; e < num_experts_; ++e) {
        const size_t ne = expert_count_[e];
        if (ne == 0) continue;

        // Map slot → token index b for this expert.
        std::vector<size_t> slot_to_b;
        slot_to_b.reserve(ne);
        for (size_t b = 0; b < B; ++b) {
            if (static_cast<size_t>(top1_idx_(b, 0)) == e && dispatch_mask_(b, e) > 0.5) {
                slot_to_b.push_back(b);
            }
        }

        // d_expert_out[slot, d] = gate_probs[b, e] · grad_output[b, d]
        Tensor d_y(ne, d_model_);
        for (size_t s = 0; s < ne; ++s) {
            size_t b = slot_to_b[s];
            double w = gate_probs_(b, e);
            for (size_t d = 0; d < d_model_; ++d)
                d_y(s, d) = w * grad_output(b, d);
        }

        // dW2[d, h] += d_y[s, d] · h_act[s, h]
        for (size_t s = 0; s < ne; ++s) {
            for (size_t d = 0; d < d_model_; ++d)
                for (size_t h = 0; h < expert_hidden_; ++h)
                    gW2_[e](d, h) += d_y(s, d) * expert_h_act_[e](s, h);
        }
        // db2[0, d] += d_y[s, d]
        for (size_t s = 0; s < ne; ++s)
            for (size_t d = 0; d < d_model_; ++d)
                gb2_[e](0, d) += d_y(s, d);

        // d_h_pre[s, h] = relu'(h_pre[s, h]) · sum_d W2[d, h] · d_y[s, d]
        Tensor d_h_pre(ne, expert_hidden_);
        for (size_t s = 0; s < ne; ++s)
            for (size_t h = 0; h < expert_hidden_; ++h) {
                double pre = expert_h_pre_[e](s, h);
                double relu_mask = pre > 0.0 ? 1.0 : 0.0;
                double s_d = 0.0;
                for (size_t d = 0; d < d_model_; ++d)
                    s_d += W2_[e](d, h) * d_y(s, d);
                d_h_pre(s, h) = relu_mask * s_d;
            }

        // dW1[h, d] += d_h_pre[s, h] · x_in[s, d]
        for (size_t s = 0; s < ne; ++s) {
            for (size_t h = 0; h < expert_hidden_; ++h)
                for (size_t d = 0; d < d_model_; ++d)
                    gW1_[e](h, d) += d_h_pre(s, h) * expert_x_in_[e](s, d);
        }
        // db1[0, h] += d_h_pre[s, h]
        for (size_t s = 0; s < ne; ++s)
            for (size_t h = 0; h < expert_hidden_; ++h)
                gb1_[e](0, h) += d_h_pre(s, h);

        // dx_in[s, d] = sum_h W1[h, d] · d_h_pre[s, h]
        Tensor d_x(ne, d_model_);
        for (size_t s = 0; s < ne; ++s)
            for (size_t d = 0; d < d_model_; ++d) {
                double s_h = 0.0;
                for (size_t h = 0; h < expert_hidden_; ++h)
                    s_h += W1_[e](h, d) * d_h_pre(s, h);
                d_x(s, d) = s_h;
            }

        // Scatter back: grad_input[b, d] += d_x[slot, d] for slot→b.
        for (size_t s = 0; s < ne; ++s) {
            size_t b = slot_to_b[s];
            for (size_t d = 0; d < d_model_; ++d)
                grad_input(b, d) += d_x(s, d);
        }
    }

    // ---------- (b) Router gradient ----------
    // d_router_prob[b, e] = sum_d grad_output[b, d] · expert_out[b, d]   (only for dispatched)
    Tensor d_router_prob = Tensor::zeros(B, num_experts_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t e = 0; e < num_experts_; ++e) {
            if (dispatch_mask_(b, e) < 0.5) continue;
            // Re-derive expert output: y = W2 · ReLU(W1 · x + b1) + b2 at the slot.
            // Simpler: cache it on the forward pass. Here we recompute.
            // For efficiency, the original sparse_moe.cpp caches expert_outputs_;
            // for v1 we recompute.
            // Actually, simpler: we already accumulated the dispatch_mask · gate_probs
            // = weight. The expert_out contribution to output is gate_probs · y_out.
            // d_router_prob = grad_output · y_out. We need y_out; instead, recompute.
            // For now, reuse the gate_probs-weighted sum directly from grad_output: this
            // is wrong because output[b,:] = gate_probs · y_out, but the relationship is
            // d_output[b,:]/d_router_prob[b,e] = y_out, so
            //   d_router_prob[b,e] = sum_d grad_output[b,d] · y_out[slot_for_b_in_e,d].
            // We DO need the expert output. Recompute per slot below.
        }
    }

    // Recompute expert outputs so we can do the router gradient.
    // We do it once per expert and reuse across its slots.
    for (size_t e = 0; e < num_experts_; ++e) {
        const size_t ne = expert_count_[e];
        if (ne == 0) continue;
        // y_out[slot, d] = sum_h W2[d, h] · h_act[slot, h] + b2[0, d]
        Tensor y_out(ne, d_model_);
        for (size_t s = 0; s < ne; ++s)
            for (size_t d = 0; d < d_model_; ++d) {
                double s_h = b2_[e](0, d);
                for (size_t h = 0; h < expert_hidden_; ++h)
                    s_h += W2_[e](d, h) * expert_h_act_[e](s, h);
                y_out(s, d) = s_h;
            }
        // Walk slots and assign d_router_prob[b, e] += sum_d grad_output[b,d] · y_out[s,d].
        size_t slot = 0;
        for (size_t b = 0; b < B; ++b) {
            if (static_cast<size_t>(top1_idx_(b, 0)) != e) continue;
            if (dispatch_mask_(b, e) < 0.5) continue;
            double acc = 0.0;
            for (size_t d = 0; d < d_model_; ++d)
                acc += grad_output(b, d) * y_out(slot, d);
            d_router_prob(b, e) = acc;
            ++slot;
        }
    }

    // Softmax backward: d_logits[b, e] = gate_probs[b, e] · (d_router_prob[b, e] - Σ_e' gate_probs[b, e'] · d_router_prob[b, e'])
    Tensor d_logits = Tensor::zeros(B, num_experts_);
    for (size_t b = 0; b < B; ++b) {
        double weighted_sum = 0.0;
        for (size_t e = 0; e < num_experts_; ++e)
            weighted_sum += gate_probs_(b, e) * d_router_prob(b, e);
        for (size_t e = 0; e < num_experts_; ++e)
            d_logits(b, e) = gate_probs_(b, e) * (d_router_prob(b, e) - weighted_sum);
    }

    // Z-loss chain: ∂L_z/∂z_e = (z_coef / B) · 2·exp(2z_e) / Σ_e' exp(2z_e')
    if (z_loss_coef_ > 0.0) {
        for (size_t b = 0; b < B; ++b) {
            double denom = 0.0;
            for (size_t e = 0; e < num_experts_; ++e)
                denom += std::exp(2.0 * gate_logits_(b, e));
            denom = std::max(denom, 1e-30);
            for (size_t e = 0; e < num_experts_; ++e) {
                double z_grad = (z_loss_coef_ / static_cast<double>(B)) *
                                (2.0 * std::exp(2.0 * gate_logits_(b, e))) / denom;
                d_logits(b, e) += z_grad;
            }
        }
    }

    // grad_W_router[e, d] += sum_b d_logits[b, e] · input[b, d]
    for (size_t e = 0; e < num_experts_; ++e)
        for (size_t d = 0; d < d_model_; ++d) {
            double s_b = 0.0;
            for (size_t b = 0; b < B; ++b)
                s_b += d_logits(b, e) * input_(b, d);
            grad_W_router_(e, d) = s_b;
        }
    // grad_b_router[0, e] += sum_b d_logits[b, e]
    for (size_t e = 0; e < num_experts_; ++e) {
        double s_b = 0.0;
        for (size_t b = 0; b < B; ++b)
            s_b += d_logits(b, e);
        grad_b_router_(0, e) = s_b;
    }

    // ---------- (c) Input gradient ----------
    // d_input[b, d] += sum_e d_logits[b, e] · W_router[e, d]
    for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < d_model_; ++d) {
            double s_e = 0.0;
            for (size_t e = 0; e < num_experts_; ++e)
                s_e += d_logits(b, e) * W_router_(e, d);
            grad_input(b, d) += s_e;
        }

    return grad_input;
}


// ----------------------------------------------------------------------------
// SwitchMoELayer: losses
// ----------------------------------------------------------------------------
//
//   Aux (paper Eq. 4): L_aux = α · N · Σ_e f_e · P_e
//     f_e = (1/B) · Σ_b dispatch_mask[b, e]    — fraction of (non-dropped) tokens
//     P_e = (1/B) · Σ_b gate_probs[b, e]       — mean router probability
//   Z (paper Eq. 5): L_z = (z_coef / B) · Σ_b log(Σ_e exp(z_e)²)
// ----------------------------------------------------------------------------

void SwitchMoELayer::compute_losses_() {
    const size_t B = input_.rows;
    const double N = static_cast<double>(num_experts_);

    // f_e = fraction of tokens dispatched to e
    std::vector<double> f_e(num_experts_, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t e = 0; e < num_experts_; ++e)
            if (dispatch_mask_(b, e) > 0.5) f_e[e] += 1.0;
    for (size_t e = 0; e < num_experts_; ++e) f_e[e] /= static_cast<double>(B);

    // P_e = mean router probability for e
    std::vector<double> P_e(num_experts_, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t e = 0; e < num_experts_; ++e)
            P_e[e] += gate_probs_(b, e);
    for (size_t e = 0; e < num_experts_; ++e) P_e[e] /= static_cast<double>(B);

    double aux = 0.0;
    for (size_t e = 0; e < num_experts_; ++e) aux += f_e[e] * P_e[e];
    load_balance_loss_ = aux_loss_coef_ * N * aux;

    dispatch_frac_ = f_e;
    mean_router_prob_ = P_e;

    // Z-loss
    if (z_loss_coef_ > 0.0) {
        double total = 0.0;
        for (size_t b = 0; b < B; ++b) {
            double inner = 0.0;
            for (size_t e = 0; e < num_experts_; ++e) {
                double v = std::exp(gate_logits_(b, e));
                inner += v * v;
            }
            total += std::log(inner + 1e-30);
        }
        z_loss_ = z_loss_coef_ * (total / static_cast<double>(B));
    } else {
        z_loss_ = 0.0;
    }
}


// ----------------------------------------------------------------------------
// SwitchMoELayer: parameters / gradients / update_weights / zero_grad
// ----------------------------------------------------------------------------

std::vector<Tensor*> SwitchMoELayer::parameters() {
    std::vector<Tensor*> p;
    p.reserve(2 + 4 * num_experts_);
    p.push_back(&W_router_);
    p.push_back(&b_router_);
    for (size_t e = 0; e < num_experts_; ++e) {
        p.push_back(&W1_[e]); p.push_back(&b1_[e]);
        p.push_back(&W2_[e]); p.push_back(&b2_[e]);
    }
    return p;
}

std::vector<Tensor*> SwitchMoELayer::gradients() {
    std::vector<Tensor*> g;
    g.reserve(2 + 4 * num_experts_);
    g.push_back(&grad_W_router_);
    g.push_back(&grad_b_router_);
    for (size_t e = 0; e < num_experts_; ++e) {
        g.push_back(&gW1_[e]); g.push_back(&gb1_[e]);
        g.push_back(&gW2_[e]); g.push_back(&gb2_[e]);
    }
    return g;
}

void SwitchMoELayer::update_weights(double learning_rate) {
    auto params = this->parameters();
    auto grads = this->gradients();
    for (size_t k = 0; k < params.size(); ++k) {
        for (size_t i = 0; i < params[k]->rows; ++i)
            for (size_t j = 0; j < params[k]->cols; ++j)
                (*params[k])(i, j) -= learning_rate * (*grads[k])(i, j);
    }
}

void SwitchMoELayer::zero_grad() {
    grad_W_router_.fill(0.0);
    grad_b_router_.fill(0.0);
    for (size_t e = 0; e < num_experts_; ++e) {
        gW1_[e].fill(0.0);
        gW2_[e].fill(0.0);
        gb1_[e].fill(0.0);
        gb2_[e].fill(0.0);
    }
}

Tensor SwitchMoELayer::get_weights() const { return W_router_; }
Tensor SwitchMoELayer::get_gradients() const { return grad_W_router_; }


// ============================================================================
// SwitchTransformerBlock
// ============================================================================

SwitchTransformerBlock::SwitchTransformerBlock(size_t d_model, size_t num_experts,
                                               size_t expert_hidden,
                                               double capacity_factor,
                                               double aux_loss_coef,
                                               double z_loss_coef)
    : ln_(d_model), moe_(d_model, num_experts, expert_hidden,
                         capacity_factor, aux_loss_coef, z_loss_coef) {}

Tensor SwitchTransformerBlock::forward(const Tensor& input) {
    input_ = input;
    Tensor normed = ln_.forward(input);
    Tensor moe_out = moe_.forward(normed);
    // Residual: out = input + moe_out
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            output(i, j) = input(i, j) + moe_out(i, j);
    return output;
}

Tensor SwitchTransformerBlock::backward(const Tensor& grad_output, double learning_rate) {
    // d_moe = grad_output (residual gradient is identity)
    Tensor grad_normed = moe_.backward(grad_output, learning_rate);
    Tensor grad_input_pre = ln_.backward(grad_normed, learning_rate);
    // grad_input = grad_input_pre + grad_output (residual path)
    Tensor grad_input(input_.rows, input_.cols);
    for (size_t i = 0; i < input_.rows; ++i)
        for (size_t j = 0; j < input_.cols; ++j)
            grad_input(i, j) = grad_input_pre(i, j) + grad_output(i, j);
    return grad_input;
}

void SwitchTransformerBlock::update_weights(double learning_rate) {
    ln_.update_weights(learning_rate);
    moe_.update_weights(learning_rate);
}

void SwitchTransformerBlock::zero_grad() {
    ln_.zero_grad();
    moe_.zero_grad();
}

std::vector<Tensor*> SwitchTransformerBlock::parameters() {
    auto p1 = ln_.parameters();
    auto p2 = moe_.parameters();
    std::vector<Tensor*> p;
    p.reserve(p1.size() + p2.size());
    for (auto* x : p1) p.push_back(x);
    for (auto* x : p2) p.push_back(x);
    return p;
}

std::vector<Tensor*> SwitchTransformerBlock::gradients() {
    auto g1 = ln_.gradients();
    auto g2 = moe_.gradients();
    std::vector<Tensor*> g;
    g.reserve(g1.size() + g2.size());
    for (auto* x : g1) g.push_back(x);
    for (auto* x : g2) g.push_back(x);
    return g;
}

Tensor SwitchTransformerBlock::get_weights() const { return moe_.get_weights(); }
Tensor SwitchTransformerBlock::get_gradients() const { return moe_.get_gradients(); }


// ============================================================================
// SwitchTransformerModel
// ============================================================================

SwitchTransformerModel::SwitchTransformerModel(size_t input_dim, size_t d_model,
                                               size_t num_experts, size_t output_dim,
                                               size_t num_blocks, size_t expert_hidden,
                                               double capacity_factor,
                                               double aux_loss_coef,
                                               double z_loss_coef)
    : input_proj_(input_dim, d_model),
      pre_ln_(d_model),
      blocks_(),
      classifier_(d_model, output_dim),
      sum_aux_(0.0),
      sum_z_(0.0)
{
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, num_experts, expert_hidden,
                             capacity_factor, aux_loss_coef, z_loss_coef);
    }
}

Tensor SwitchTransformerModel::forward(const Tensor& input) {
    input_ = input;
    Tensor x = input_proj_.forward(input);
    x = pre_ln_.forward(x);
    sum_aux_ = 0.0;
    sum_z_ = 0.0;
    for (auto& block : blocks_) {
        x = block.forward(x);
        sum_aux_ += block.get_aux_loss();
        sum_z_   += block.get_z_loss();
    }
    x = classifier_.forward(x);
    return x;
}

Tensor SwitchTransformerModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor g = classifier_.backward(grad_output, learning_rate);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        g = it->backward(g, learning_rate);
    }
    g = pre_ln_.backward(g, learning_rate);
    g = input_proj_.backward(g, learning_rate);
    return g;
}

void SwitchTransformerModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    pre_ln_.update_weights(learning_rate);
    for (auto& block : blocks_) block.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void SwitchTransformerModel::zero_grad() {
    input_proj_.zero_grad();
    pre_ln_.zero_grad();
    for (auto& block : blocks_) block.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> SwitchTransformerModel::parameters() {
    auto p1 = input_proj_.parameters();
    auto p2 = pre_ln_.parameters();
    auto p3 = classifier_.parameters();
    std::vector<Tensor*> p;
    p.reserve(p1.size() + p2.size() + p3.size() + 100);
    for (auto* x : p1) p.push_back(x);
    for (auto* x : p2) p.push_back(x);
    for (auto& block : blocks_) {
        auto pb = block.parameters();
        for (auto* x : pb) p.push_back(x);
    }
    for (auto* x : p3) p.push_back(x);
    return p;
}

std::vector<Tensor*> SwitchTransformerModel::gradients() {
    auto g1 = input_proj_.gradients();
    auto g2 = pre_ln_.gradients();
    auto g3 = classifier_.gradients();
    std::vector<Tensor*> g;
    g.reserve(g1.size() + g2.size() + g3.size() + 100);
    for (auto* x : g1) g.push_back(x);
    for (auto* x : g2) g.push_back(x);
    for (auto& block : blocks_) {
        auto gb = block.gradients();
        for (auto* x : gb) g.push_back(x);
    }
    for (auto* x : g3) g.push_back(x);
    return g;
}

Tensor SwitchTransformerModel::get_weights() const { return input_proj_.get_weights(); }
Tensor SwitchTransformerModel::get_gradients() const { return input_proj_.get_gradients(); }
