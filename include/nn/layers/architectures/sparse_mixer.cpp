#include "sparse_mixer.h"
#include "../normalization/layer_norm.h"
#include <cmath>
#include <algorithm>
#include <random>

// ============================================================================
// LinearMixingSublayer
// ============================================================================
//
// Math:
//   Input x: (B, S*D)  — flattened (batch, token, channel).
//   Reshape to (B, S, D).
//   1) T_seq mixes over the sequence dim
//      z_seq[b, s, d] = sum_{s'} x[b, s', d] * T_seq[s, s']
//   2) T_h mixes over the channel dim + b_h bias
//      y[b, s, d] = sum_{d'} z_seq[b, s, d'] * T_h[d, d'] + b_h[d]
// We use the convention that T_seq and T_h are (out, in), so the matmul
// is x @ T^T for each "row" — matches Dense.
// ============================================================================

LinearMixingSublayer::LinearMixingSublayer(size_t d_model, size_t seq_len)
    : d_model_(d_model), seq_len_(seq_len),
      W_h(d_model, d_model),
      W_seq(seq_len, seq_len),
      b_h(1, d_model),
      grad_W_h(d_model, d_model),
      grad_W_seq(seq_len, seq_len),
      grad_b_h(1, d_model),
      input_(0, 0),
      last_z_seq_(0, 0),
      last_output_(0, 0),
      last_d_input_(0, 0) {
    if (d_model == 0) throw std::invalid_argument("LinearMixingSublayer: d_model must be > 0");
    if (seq_len == 0) throw std::invalid_argument("LinearMixingSublayer: seq_len must be > 0");
    std::mt19937 gen(42);
    double std_h = std::sqrt(1.0 / (double)d_model);
    double std_seq = std::sqrt(1.0 / (double)seq_len);
    std::normal_distribution<> dh(0.0, std_h);
    std::normal_distribution<> ds(0.0, std_seq);
    for (size_t i = 0; i < W_h.rows; ++i)
        for (size_t j = 0; j < W_h.cols; ++j) W_h[i][j] = dh(gen);
    for (size_t i = 0; i < W_seq.rows; ++i)
        for (size_t j = 0; j < W_seq.cols; ++j) W_seq[i][j] = ds(gen);
    b_h.fill(0.0);
    grad_W_h.fill(0.0);
    grad_W_seq.fill(0.0);
    grad_b_h.fill(0.0);
}

Tensor LinearMixingSublayer::forward(const Tensor& input) {
    input_ = input.clone();
    size_t B = input.rows;
    size_t D = d_model_;
    size_t S = seq_len_;

    Tensor z_seq(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t sp = 0; sp < S; ++sp) {
                    acc += W_seq(s, sp) * input(b, sp * D + d);
                }
                z_seq(b, s * D + d) = acc;
            }
        }
    }
    last_z_seq_ = z_seq.clone();

    Tensor output(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = b_h(0, d);
                for (size_t dp = 0; dp < D; ++dp) {
                    acc += W_h(d, dp) * z_seq(b, s * D + dp);
                }
                output(b, s * D + d) = acc;
            }
        }
    }
    last_output_ = output.clone();
    return output;
}

Tensor LinearMixingSublayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = input_.rows;
    size_t D = d_model_;
    size_t S = seq_len_;

    grad_W_h.fill(0.0);
    grad_W_seq.fill(0.0);
    grad_b_h.fill(0.0);

    Tensor d_z_seq(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double go = grad_output(b, s * D + d);
                grad_b_h(0, d) += go;
                for (size_t dp = 0; dp < D; ++dp) {
                    grad_W_h(d, dp) += go * last_z_seq_(b, s * D + dp);
                    d_z_seq(b, s * D + dp) += go * W_h(d, dp);
                }
            }
        }
    }

    Tensor d_input(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double dzs = d_z_seq(b, s * D + d);
                for (size_t sp = 0; sp < S; ++sp) {
                    grad_W_seq(s, sp) += dzs * input_(b, sp * D + d);
                    d_input(b, sp * D + d) += dzs * W_seq(s, sp);
                }
            }
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

void LinearMixingSublayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < W_h.rows; ++i)
        for (size_t j = 0; j < W_h.cols; ++j)
            W_h(i, j) -= learning_rate * grad_W_h(i, j);
    for (size_t i = 0; i < W_seq.rows; ++i)
        for (size_t j = 0; j < W_seq.cols; ++j)
            W_seq(i, j) -= learning_rate * grad_W_seq(i, j);
    for (size_t j = 0; j < b_h.cols; ++j)
        b_h(0, j) -= learning_rate * grad_b_h(0, j);
}

void LinearMixingSublayer::zero_grad() {
    grad_W_h.fill(0.0);
    grad_W_seq.fill(0.0);
    grad_b_h.fill(0.0);
    last_d_input_ = Tensor(0, 0);
}

std::vector<Tensor*> LinearMixingSublayer::parameters() {
    return {&W_h, &W_seq, &b_h};
}
std::vector<Tensor*> LinearMixingSublayer::gradients() {
    return {&grad_W_h, &grad_W_seq, &grad_b_h};
}

// ============================================================================
// ExpertsChoiceMoE
// ============================================================================

ExpertsChoiceMoE::ExpertsChoiceMoE(size_t d_model, size_t num_experts,
                                   size_t expert_hidden, double capacity_factor)
    : d_model_(d_model), num_experts_(num_experts),
      expert_hidden_(expert_hidden == 0 ? 4 * d_model : expert_hidden),
      capacity_factor_(capacity_factor),
      W_router(num_experts, d_model),
      b_router(1, num_experts),
      grad_W_router(num_experts, d_model),
      grad_b_router(1, num_experts),
      input_(0, 0),
      gate_logits_(0, 0),
      dispatch_mask_(0, 0),
      last_d_input_(0, 0) {
    if (d_model == 0) throw std::invalid_argument("ExpertsChoiceMoE: d_model must be > 0");
    if (num_experts == 0) throw std::invalid_argument("ExpertsChoiceMoE: num_experts must be > 0");
    if (capacity_factor <= 0.0) throw std::invalid_argument("ExpertsChoiceMoE: capacity_factor must be > 0");

    std::mt19937 gen(137);
    double std_r = std::sqrt(1.0 / (double)d_model);
    std::normal_distribution<> dr(0.0, std_r);
    for (size_t i = 0; i < W_router.rows; ++i)
        for (size_t j = 0; j < W_router.cols; ++j) W_router(i, j) = dr(gen);
    b_router.fill(0.0);
    grad_W_router.fill(0.0);
    grad_b_router.fill(0.0);

    size_t H = expert_hidden_;
    W1_.resize(num_experts);
    W2_.resize(num_experts);
    b1_.resize(num_experts);
    b2_.resize(num_experts);
    gW1_.resize(num_experts);
    gW2_.resize(num_experts);
    gb1_.resize(num_experts);
    gb2_.resize(num_experts);
    expert_h_pre_.resize(num_experts);
    expert_h_act_.resize(num_experts);
    expert_inputs_.resize(num_experts);
    expert_outputs_.resize(num_experts);
    assign_.resize(num_experts);
    for (size_t e = 0; e < num_experts; ++e) {
        W1_[e] = Tensor(H, d_model);
        W2_[e] = Tensor(d_model, H);
        b1_[e] = Tensor(1, H);
        b2_[e] = Tensor(1, d_model);
        gW1_[e] = Tensor(H, d_model);
        gW2_[e] = Tensor(d_model, H);
        gb1_[e] = Tensor(1, H);
        gb2_[e] = Tensor(1, d_model);
        double std_e = std::sqrt(2.0 / (double)(d_model + H));
        std::normal_distribution<> de(0.0, std_e);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < d_model; ++j) W1_[e](i, j) = de(gen);
            b1_[e](0, i) = 0.0;
        }
        for (size_t i = 0; i < d_model; ++i) {
            for (size_t j = 0; j < H; ++j) W2_[e](i, j) = de(gen);
            b2_[e](0, i) = 0.0;
        }
        gW1_[e].fill(0.0); gW2_[e].fill(0.0);
        gb1_[e].fill(0.0); gb2_[e].fill(0.0);
    }
}

Tensor ExpertsChoiceMoE::forward(const Tensor& input) {
    input_ = input.clone();
    size_t B = input.rows;
    size_t E = num_experts_;
    size_t D = d_model_;
    size_t H = expert_hidden_;

    // Router logits: (B, E)
    Tensor logits = input * W_router.transpose();
    for (size_t b = 0; b < B; ++b)
        for (size_t e = 0; e < E; ++e)
            logits(b, e) += b_router(0, e);
    gate_logits_ = logits.clone();

    size_t cap = (size_t)std::ceil((double)B / (double)E * capacity_factor_);
    if (cap > B) cap = B;

    dispatch_mask_ = Tensor(B, E);
    for (size_t b = 0; b < B; ++b)
        for (size_t e = 0; e < E; ++e)
            dispatch_mask_(b, e) = 0.0;

    Tensor output(B, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < D; ++d)
            output(b, d) = 0.0;

    // For each expert: pick top-cap tokens, run FFN, scatter-mean into output.
    for (size_t e = 0; e < E; ++e) {
        std::vector<std::pair<double, size_t>> sorted;
        sorted.reserve(B);
        for (size_t b = 0; b < B; ++b) sorted.push_back({logits(b, e), b});
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& a2) { return a.first > a2.first; });

        assign_[e].clear();
        Tensor& exp_in = expert_inputs_[e];
        exp_in = Tensor(cap, D);
        for (size_t i = 0; i < cap; ++i) {
            size_t idx = sorted[i].second;
            assign_[e].push_back(idx);
            dispatch_mask_(idx, e) = 1.0;
            for (size_t d = 0; d < D; ++d) exp_in(i, d) = input(idx, d);
        }

        // h_pre = exp_in @ W1^T + b1  -> (cap, H)
        Tensor h_pre = exp_in * W1_[e].transpose();
        for (size_t i = 0; i < cap; ++i)
            for (size_t h = 0; h < H; ++h)
                h_pre(i, h) += b1_[e](0, h);
        expert_h_pre_[e] = h_pre.clone();

        // h_act = ReLU(h_pre)
        Tensor h_act(cap, H);
        for (size_t i = 0; i < cap; ++i)
            for (size_t h = 0; h < H; ++h) {
                double v = h_pre(i, h);
                h_act(i, h) = v > 0.0 ? v : 0.0;
            }
        expert_h_act_[e] = h_act.clone();

        // exp_out = h_act @ W2^T + b2  -> (cap, D)
        Tensor exp_out = h_act * W2_[e].transpose();
        for (size_t i = 0; i < cap; ++i)
            for (size_t d = 0; d < D; ++d)
                exp_out(i, d) += b2_[e](0, d);
        expert_outputs_[e] = exp_out.clone();

        // Scatter-mean: for each token b, average the expert outputs over all
        // experts that selected b. If no expert selected b, output stays 0
        // (the residual in the surrounding block carries the input through).
        for (size_t i = 0; i < cap; ++i) {
            size_t b = assign_[e][i];
            for (size_t d = 0; d < D; ++d) {
                output(b, d) += exp_out(i, d);
            }
        }
    }

    // Divide by per-token count (mean over selected experts).
    // A token selected by count_e experts contributes count_e * exp_out summed;
    // we want the mean, so divide by count_e.
    for (size_t b = 0; b < B; ++b) {
        size_t cnt = 0;
        for (size_t e = 0; e < E; ++e) cnt += (size_t)dispatch_mask_(b, e);
        if (cnt > 0) {
            for (size_t d = 0; d < D; ++d) output(b, d) /= (double)cnt;
        }
    }

    return output;
}

Tensor ExpertsChoiceMoE::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = input_.rows;
    size_t E = num_experts_;
    size_t D = d_model_;
    size_t H = expert_hidden_;

    grad_W_router.fill(0.0);
    grad_b_router.fill(0.0);
    for (size_t e = 0; e < E; ++e) {
        gW1_[e].fill(0.0); gW2_[e].fill(0.0);
        gb1_[e].fill(0.0); gb2_[e].fill(0.0);
    }

    // 1) Scale grad_output by 1/count_e per token and scatter into per-expert
    //    grads (each expert gets grad_out scaled by 1/count_e for its
    //    dispatched positions).
    // 2) For each expert, run FFN backward on the dispatched (cap, D) batch
    //    to compute gW1/gW2/gb1/gb2 and the per-expert d_input (cap, D).
    // 3) Sum per-expert d_input back into per-token d_input.
    // 4) Compute router gradient: d_logits[b, e] = softmax_w * sum_i grad
    //    over the experts that selected b, where softmax_w weights the
    //    contribution of each expert by its gate logit magnitude (paper
    //    style). Concretely: d_logits[b, e] = dispatch_mask[b, e] *
    //    mean_d(grad_output[b, d]) * (1 — uniform scaling).
    //
    //    For simplicity, we use the standard linear-routing gradient:
    //    d_logits[b, e] is the dot product of grad_output[b, :] with
    //    expert_e_output[b, :], weighted by the dispatch mask. (This is
    //    the "softmax-cross-entropy" style: d_logits[b,e] = M[b,e] *
    //    sum_d grad_output[b,d] * expert_e_out[b,d] / count_b, then router
    //    gets dW += d_logits^T @ input.)

    Tensor d_input(B, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < D; ++d)
            d_input(b, d) = 0.0;

    // Per-expert d_logits accumulator (B, E) — router chain via expert outputs.
    // For Experts-Choice, the routing decision (which tokens expert e selects)
    // is a HARD piecewise-constant function of W_router — only the sort order
    // matters. The gradient of the main loss through the router W is therefore
    // ZERO except at sort-thresholds. We keep this accumulator to derive
    // gradients via the auxiliary load-balancing loss (not implemented in
    // v1) but we do NOT use it to backprop into d_input (the chain through
    // the hard assignment is non-existent at generic W values).
    Tensor d_logits(num_experts_, B);  // (E, B) — kept for future aux-loss chain
    for (size_t e = 0; e < E; ++e)
        for (size_t b = 0; b < B; ++b)
            d_logits(e, b) = 0.0;

    (void)d_logits;

    for (size_t e = 0; e < E; ++e) {
        size_t cap = assign_[e].size();
        if (cap == 0) continue;

        // Scale grad_output by 1/count_b per token to get this expert's
        // contribution gradient.
        Tensor exp_grad_out(cap, D);
        for (size_t i = 0; i < cap; ++i) {
            size_t b = assign_[e][i];
            size_t cnt = 0;
            for (size_t ee = 0; ee < E; ++ee) cnt += (size_t)dispatch_mask_(b, ee);
            double inv = cnt > 0 ? 1.0 / (double)cnt : 0.0;
            for (size_t d = 0; d < D; ++d) {
                exp_grad_out(i, d) = grad_output(b, d) * inv;
            }
        }

        // FFN backward (standard 2-layer ReLU)
        // exp_grad_h2 = exp_grad_out @ W2  -> (cap, H)
        Tensor exp_grad_h2(cap, H);
        for (size_t i = 0; i < cap; ++i)
            for (size_t h = 0; h < H; ++h) {
                double acc = 0.0;
                for (size_t d = 0; d < D; ++d) acc += exp_grad_out(i, d) * W2_[e](d, h);
                exp_grad_h2(i, h) = acc;
            }
        // ReLU' mask
        for (size_t i = 0; i < cap; ++i)
            for (size_t h = 0; h < H; ++h) {
                double v = expert_h_pre_[e](i, h);
                exp_grad_h2(i, h) *= (v > 0.0) ? 1.0 : 0.0;
            }
        // gW2 += exp_grad_out^T @ h_act, gb2 += sum exp_grad_out
        for (size_t d = 0; d < D; ++d)
            for (size_t h = 0; h < H; ++h) {
                double acc = 0.0;
                for (size_t i = 0; i < cap; ++i) acc += exp_grad_out(i, d) * expert_h_act_[e](i, h);
                gW2_[e](d, h) += acc;
            }
        for (size_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (size_t i = 0; i < cap; ++i) acc += exp_grad_out(i, d);
            gb2_[e](0, d) += acc;
        }
        // gW1 += exp_grad_h2^T @ exp_in, gb1 += sum exp_grad_h2
        for (size_t h = 0; h < H; ++h)
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < cap; ++i) acc += exp_grad_h2(i, h) * expert_inputs_[e](i, d);
                gW1_[e](h, d) += acc;
            }
        for (size_t h = 0; h < H; ++h) {
            double acc = 0.0;
            for (size_t i = 0; i < cap; ++i) acc += exp_grad_h2(i, h);
            gb1_[e](0, h) += acc;
        }
        // d_input for this expert: d_exp_in = exp_grad_h2 @ W1  -> (cap, D)
        Tensor exp_d_in(cap, D);
        for (size_t i = 0; i < cap; ++i)
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t h = 0; h < H; ++h) acc += exp_grad_h2(i, h) * W1_[e](h, d);
                exp_d_in(i, d) = acc;
            }
        // Scatter back into d_input
        for (size_t i = 0; i < cap; ++i) {
            size_t b = assign_[e][i];
            for (size_t d = 0; d < D; ++d) d_input(b, d) += exp_d_in(i, d);
        }

        // Router gradient via the main loss is zero — see comment below in the
        // router backward section. We do not accumulate d_logits here.
    }

    // Router gradient via the main loss is zero (hard selection is piecewise
    // constant in W_router). grad_W_router and grad_b_router stay at zero
    // from the zero_grad() at the top. A load-balancing aux loss (not
    // implemented in v1) is the standard way to provide gradient to the
    // router for Experts-Choice routing.

    last_d_input_ = d_input;
    return d_input;
}

void ExpertsChoiceMoE::update_weights(double learning_rate) {
    for (size_t i = 0; i < W_router.rows; ++i)
        for (size_t j = 0; j < W_router.cols; ++j)
            W_router(i, j) -= learning_rate * grad_W_router(i, j);
    for (size_t j = 0; j < b_router.cols; ++j)
        b_router(0, j) -= learning_rate * grad_b_router(0, j);
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t i = 0; i < W1_[e].rows; ++i)
            for (size_t j = 0; j < W1_[e].cols; ++j)
                W1_[e](i, j) -= learning_rate * gW1_[e](i, j);
        for (size_t j = 0; j < b1_[e].cols; ++j)
            b1_[e](0, j) -= learning_rate * gb1_[e](0, j);
        for (size_t i = 0; i < W2_[e].rows; ++i)
            for (size_t j = 0; j < W2_[e].cols; ++j)
                W2_[e](i, j) -= learning_rate * gW2_[e](i, j);
        for (size_t j = 0; j < b2_[e].cols; ++j)
            b2_[e](0, j) -= learning_rate * gb2_[e](0, j);
    }
}

void ExpertsChoiceMoE::zero_grad() {
    grad_W_router.fill(0.0);
    grad_b_router.fill(0.0);
    for (size_t e = 0; e < num_experts_; ++e) {
        gW1_[e].fill(0.0); gW2_[e].fill(0.0);
        gb1_[e].fill(0.0); gb2_[e].fill(0.0);
    }
    last_d_input_ = Tensor(0, 0);
}

std::vector<Tensor*> ExpertsChoiceMoE::parameters() {
    std::vector<Tensor*> out = {&W_router, &b_router};
    for (size_t e = 0; e < num_experts_; ++e) {
        out.push_back(&W1_[e]);
        out.push_back(&W2_[e]);
        out.push_back(&b1_[e]);
        out.push_back(&b2_[e]);
    }
    return out;
}
std::vector<Tensor*> ExpertsChoiceMoE::gradients() {
    std::vector<Tensor*> out = {&grad_W_router, &grad_b_router};
    for (size_t e = 0; e < num_experts_; ++e) {
        out.push_back(&gW1_[e]);
        out.push_back(&gW2_[e]);
        out.push_back(&gb1_[e]);
        out.push_back(&gb2_[e]);
    }
    return out;
}

// ============================================================================
// SparseMixerBlock
// ============================================================================

SparseMixerBlock::SparseMixerBlock(Mode mode, size_t d_model, size_t seq_len,
                                   size_t num_experts, size_t expert_hidden,
                                   double capacity_factor, size_t ffn_mult)
    : mode_(mode), d_model_(d_model), seq_len_(seq_len),
      has_moe_(num_experts > 0),
      ln1_(d_model), ln2_(d_model),
      W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model), b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model),
      grad_b_v(1, d_model), grad_b_o(1, d_model),
      last_z1_(0, 0),
      last_q_(0, 0), last_k_(0, 0), last_v_(0, 0),
      last_attn_(0, 0),
      last_attn_out_(0, 0),
      last_d_z1_(0, 0),
      last_d_input_(0, 0),
      last_z2_(0, 0),
      last_ffn_h_(0, 0),
      last_ffn_h_act_(0, 0),
      last_ffn_out_(0, 0),
      last_d_z2_(0, 0) {
    if (d_model == 0) throw std::invalid_argument("SparseMixerBlock: d_model must be > 0");
    if (seq_len == 0) throw std::invalid_argument("SparseMixerBlock: seq_len must be > 0");

    if (mode == LINEAR_MIXING) {
        mixer_ = std::make_unique<LinearMixingSublayer>(d_model, seq_len);
    } else {
        // SELF_ATTENTION — single-head Q/K/V/O + biases
        std::mt19937 gen(42);
        double std = std::sqrt(1.0 / (double)d_model);
        std::normal_distribution<> d(0.0, std);
        for (size_t i = 0; i < d_model; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
                W_q(i, j) = d(gen);
                W_k(i, j) = d(gen);
                W_v(i, j) = d(gen);
                W_o(i, j) = d(gen);
            }
        }
        b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);
    }

    if (has_moe_) {
        size_t ehidden = expert_hidden == 0 ? 4 * d_model : expert_hidden;
        ffn_ = std::make_unique<ExpertsChoiceMoE>(d_model, num_experts, ehidden, capacity_factor);
        // ffn_dense1/2 unused but kept default (1, 1) so they exist as members
        ffn_dense1_ = Dense(1, 1);
        ffn_dense2_ = Dense(1, 1);
    } else {
        size_t ffn_dim = ffn_mult == 0 ? 4 * d_model : ffn_mult * d_model;
        if (ffn_mult == 0) ffn_dim = 4 * d_model;
        ffn_dense1_ = Dense(d_model, ffn_dim);
        ffn_dense2_ = Dense(ffn_dim, d_model);
    }
}

Tensor SparseMixerBlock::forward(const Tensor& input) {
    size_t B = input.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // =========== Sublayer 1: pre-norm + mixing/attention + residual ===========
    // z1 = ln1(input). Per-token pre-norm: reshape (B, S*D) → (B*S, D) and
    // LN-normalize each row (one row = one token). Returns (B*S, D); flatten
    // back to (B, S*D) for the mixing/attention sublayer.
    Tensor z1_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z1_flat(b * S + s, d) = input(b, s * D + d);
    Tensor z1_norm = ln1_.forward(z1_flat);
    Tensor z1(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z1(b, s * D + d) = z1_norm(b * S + s, d);
    last_z1_ = z1.clone();

    Tensor mix_out(B, S * D);
    if (mode_ == LINEAR_MIXING) {
        mix_out = mixer_->forward(z1);
    } else {
        // SELF_ATTENTION — single-head (B, S, D)
        // Q, K, V are per-token projections. Apply Dense on (B*S, D) → (B*S, D).
        Tensor z1_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    z1_flat(b * S + s, d) = z1(b, s * D + d);
        Tensor Q_flat = z1_flat * W_q.transpose();  // (B*S, D)
        Tensor K_flat = z1_flat * W_k.transpose();
        Tensor V_flat = z1_flat * W_v.transpose();
        // Add biases and reshape back to (B, S, D) flat.
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t d = 0; d < D; ++d) {
                    Q_flat(b * S + s, d) += b_q(0, d);
                    K_flat(b * S + s, d) += b_k(0, d);
                    V_flat(b * S + s, d) += b_v(0, d);
                }
            }
        }
        Tensor Q(B, S * D), K(B, S * D), V(B, S * D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t d = 0; d < D; ++d) {
                    Q(b, s * D + d) = Q_flat(b * S + s, d);
                    K(b, s * D + d) = K_flat(b * S + s, d);
                    V(b, s * D + d) = V_flat(b * S + s, d);
                }
            }
        }
        last_q_ = Q.clone();
        last_k_ = K.clone();
        last_v_ = V.clone();

        // attention scores A[b, i, j] = Q[b, i, :] · K[b, j, :] / sqrt(D)
        double scale = 1.0 / std::sqrt((double)D);
        Tensor scores(B, S * S);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    double acc = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        acc += Q(b, i * D + d) * K(b, j * D + d);
                    }
                    scores(b, i * S + j) = acc * scale;
                }
            }
        }
        // softmax per row
        Tensor attn(B, S * S);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                double mx = scores(b, i * S);
                for (size_t j = 1; j < S; ++j) mx = std::max(mx, scores(b, i * S + j));
                double sum = 0.0;
                for (size_t j = 0; j < S; ++j) {
                    double v = std::exp(scores(b, i * S + j) - mx);
                    attn(b, i * S + j) = v;
                    sum += v;
                }
                for (size_t j = 0; j < S; ++j) attn(b, i * S + j) /= sum;
            }
        }
        last_attn_ = attn.clone();

        // attn_out[b, i, d] = sum_j attn[b, i, j] * V[b, j, d]
        Tensor attn_out(B, S * D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < S; ++j) {
                        acc += attn(b, i * S + j) * V(b, j * D + d);
                    }
                    attn_out(b, i * D + d) = acc;
                }
            }
        }
        last_attn_out_ = attn_out.clone();

        // project + bias (per-token)
        Tensor attn_out_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    attn_out_flat(b * S + s, d) = attn_out(b, s * D + d);
        Tensor mix_out_flat = attn_out_flat * W_o.transpose();
        for (size_t i = 0; i < B * S; ++i)
            for (size_t d = 0; d < D; ++d)
                mix_out_flat(i, d) += b_o(0, d);
        mix_out = Tensor(B, S * D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    mix_out(b, s * D + d) = mix_out_flat(b * S + s, d);
    }

    // Residual
    Tensor r1(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < S * D; ++i)
            r1.data[b * (S*D) + i] = input.data[b * (S*D) + i] + mix_out.data[b * (S*D) + i];

    // =========== Sublayer 2: pre-norm + MoE/dense FFN + residual ===========
    Tensor z2_flat_in(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z2_flat_in(b * S + s, d) = r1(b, s * D + d);
    Tensor z2_norm = ln2_.forward(z2_flat_in);
    Tensor z2(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                z2(b, s * D + d) = z2_norm(b * S + s, d);
    last_z2_ = z2.clone();

    Tensor ffn_out(B, S * D);
    if (has_moe_) {
        // Apply MoE per token. The MoE layer takes (B, D) and returns (B, D).
        // We need to reshape z2 (B, S*D) → (B, D) per token, run MoE, concat back.
        // For a uniform MoE (no per-token specialization in the routing), we
        // can run MoE on (B*S, D) — but the dispatcher sees all tokens flat.
        // Run MoE on the full flattened (B*S, D) — this means routing runs
        // over B*S tokens (matches paper §4.3 default).
        Tensor z2_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    z2_flat(b * S + s, d) = z2(b, s * D + d);
        Tensor ffn_flat = ffn_->forward(z2_flat);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    ffn_out(b, s * D + d) = ffn_flat(b * S + s, d);
    } else {
        // Dense 2-layer GELU FFN, applied per token (residual FFN on (B*S, D)).
        Tensor z2_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    z2_flat(b * S + s, d) = z2(b, s * D + d);
        last_ffn_h_ = ffn_dense1_.forward(z2_flat);
        // GELU
        last_ffn_h_act_ = Tensor(last_ffn_h_.rows, last_ffn_h_.cols);
        for (size_t i = 0; i < last_ffn_h_.data.size(); ++i) {
            double x = last_ffn_h_.data[i];
            // GELU tanh approx
            double xc = std::max(-4.0, std::min(4.0, x));
            double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc)));
            last_ffn_h_act_.data[i] = x * cdf;
        }
        Tensor ffn_flat = ffn_dense2_.forward(last_ffn_h_act_);
        // zero ffn_dense1/2 grads (Dense::backward accumulates)
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    ffn_out(b, s * D + d) = ffn_flat(b * S + s, d);
    }

    Tensor r2(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < S * D; ++i)
            r2.data[b * (S*D) + i] = r1.data[b * (S*D) + i] + ffn_out.data[b * (S*D) + i];

    return r2;
}

Tensor SparseMixerBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = grad_output.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // =========== Sublayer 2 backward ===========
    // d_r1 = grad_output (residual bypass)
    // d_ffn_out = grad_output
    // d_r2_in: residual — d_r1 += grad_output; d_ffn_out += grad_output.
    Tensor d_r1(B, S * D);
    Tensor d_ffn_out(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < S * D; ++i) {
            double g = grad_output(b, i);
            d_r1(b, i) = g;
            d_ffn_out(b, i) = g;
        }

    // FFN backward
    Tensor d_z2(B, S * D);
    if (has_moe_) {
        Tensor d_ffn_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_ffn_flat(b * S + s, d) = d_ffn_out(b, s * D + d);
        Tensor d_z2_flat = ffn_->backward(d_ffn_flat, 0.0);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_z2(b, s * D + d) = d_z2_flat(b * S + s, d);
    } else {
        // Dense 2-layer GELU FFN backward
        // ffn_dense2_.backward(d_ffn_out_flat) -> d_h_act + gW2/gb2
        Tensor d_ffn_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_ffn_flat(b * S + s, d) = d_ffn_out(b, s * D + d);
        // zero dense grads before accumulating
        ffn_dense1_.zero_grad();
        ffn_dense2_.zero_grad();
        Tensor d_h_act = ffn_dense2_.backward(d_ffn_flat, 0.0);
        // GELU' applied to last_ffn_h_ (pre-activation). Closed form:
        //   d/dx [x * cdf(x)] = cdf(x) + x * phi(x)
        // where cdf and phi are the standard normal cdf and pdf evaluated
        // at x via the tanh approx. dcdf is d/dx [cdf(x)].
        Tensor d_h_pre(B * S, last_ffn_h_.cols);
        for (size_t i = 0; i < last_ffn_h_.data.size(); ++i) {
            double x = last_ffn_h_.data[i];
            double xc = std::max(-4.0, std::min(4.0, x));
            double k0 = std::sqrt(2.0 / M_PI);
            double inner = k0 * (xc + 0.044715 * xc * xc * xc);
            double th = std::tanh(inner);
            double dcdf = 0.5 * (1.0 - th * th) * k0 * (1.0 + 3.0 * 0.044715 * xc * xc);
            double cdf_v = 0.5 * (1.0 + th);
            double dgelu = cdf_v + x * dcdf;
            d_h_pre.data[i] = d_h_act.data[i] * dgelu;
        }
        Tensor d_z2_flat = ffn_dense1_.backward(d_h_pre, 0.0);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_z2(b, s * D + d) = d_z2_flat(b * S + s, d);
    }
    last_d_z2_ = d_z2.clone();

    // Backward through ln2 + residual. d_r1 += ln2_.backward(d_z2).
    // LN was called on (B*S, D); pass the same shape back.
    Tensor d_z2_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_z2_flat(b * S + s, d) = d_z2(b, s * D + d);
    Tensor d_ln2_flat = ln2_.backward(d_z2_flat, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_r1(b, s * D + d) += d_ln2_flat(b * S + s, d);

    // =========== Sublayer 1 backward ===========
    // d_mix_out = d_r1; d_input += d_r1 (residual bypass).
    Tensor d_mix_out(B, S * D);
    Tensor d_input(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < S * D; ++i) {
            d_mix_out(b, i) = d_r1(b, i);
            d_input(b, i) = d_r1(b, i);
        }

    Tensor d_z1(B, S * D);
    if (mode_ == LINEAR_MIXING) {
        d_z1 = mixer_->backward(d_mix_out, 0.0);
    } else {
        // SELF_ATTENTION backward
        // Forward was: Q,K,V = z1 @ W^T + b; scores = QK^T/sqrt(D); attn = softmax(scores);
        //              attn_out = attn @ V; mix_out = attn_out @ W_o^T + b_o.
        //
        // d_attn_out[b, s, d] = sum_d' d_mix_out[b, s, d'] * W_o[d', d]
        // Flatten first.
        Tensor d_mix_out_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_mix_out_flat(b * S + s, d) = d_mix_out(b, s * D + d);
        Tensor d_attn_out_flat(B * S, D);
        for (size_t i = 0; i < B * S; ++i) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t dp = 0; dp < D; ++dp) acc += d_mix_out_flat(i, dp) * W_o(dp, d);
                d_attn_out_flat(i, d) = acc;
            }
        }
        Tensor d_attn_out(B, S * D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_attn_out(b, s * D + d) = d_attn_out_flat(b * S + s, d);
        // gW_o += d_mix_out^T @ attn_out; gb_o += sum d_mix_out per channel
        for (size_t dp = 0; dp < D; ++dp) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t b = 0; b < B; ++b)
                    for (size_t s = 0; s < S; ++s)
                        acc += d_mix_out(b, s * D + dp) * last_attn_out_(b, s * D + d);
                grad_W_o(dp, d) += acc;
            }
        }
        for (size_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (size_t b = 0; b < B; ++b)
                for (size_t s = 0; s < S; ++s)
                    acc += d_mix_out(b, s * D + d);
            grad_b_o(0, d) += acc;
        }

        // dV[b, j, d] = sum_i d_attn_out[b, i, d] * attn[b, i, j] (computed in the
        // per-token-flatten block below)
        (void)0;

        // d_attn[b, i, j] = sum_d d_attn_out[b, i, d] * V[b, j, d]
        Tensor d_attn(B, S * S);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t j = 0; j < S; ++j) {
                    double acc = 0.0;
                    for (size_t d = 0; d < D; ++d) acc += d_attn_out(b, i * D + d) * last_v_(b, j * D + d);
                    d_attn(b, i * S + j) = acc;
                }
            }
        }

        // Softmax backward: d_scores[b, i, j] = attn * (d_attn - sum_k attn[b, i, k] * d_attn[b, i, k])
        Tensor d_scores(B, S * S);
        double scale = 1.0 / std::sqrt((double)D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                double sum_ = 0.0;
                for (size_t j = 0; j < S; ++j) sum_ += last_attn_(b, i * S + j) * d_attn(b, i * S + j);
                for (size_t j = 0; j < S; ++j) {
                    d_scores(b, i * S + j) = last_attn_(b, i * S + j) * (d_attn(b, i * S + j) - sum_);
                }
            }
        }

        // dQ[b, i, d] = scale * sum_j d_scores[b, i, j] * K[b, j, d]
        // dK[b, j, d] = scale * sum_i d_scores[b, i, j] * Q[b, i, d]
        // Per-token flatten: dQ_flat[i, d] = scale * sum_j d_scores[b, i, j] * K_flat[j, d]
        Tensor last_k_flat(B * S, D), last_q_flat(B * S, D), last_v_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d) {
                    last_k_flat(b * S + s, d) = last_k_(b, s * D + d);
                    last_q_flat(b * S + s, d) = last_q_(b, s * D + d);
                    last_v_flat(b * S + s, d) = last_v_(b, s * D + d);
                }
        Tensor dQ_flat(B * S, D), dK_flat(B * S, D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < S; ++i) {
                for (size_t d = 0; d < D; ++d) {
                    double accQ = 0.0;
                    for (size_t j = 0; j < S; ++j) accQ += d_scores(b, i * S + j) * last_k_flat(b * S + j, d);
                    dQ_flat(b * S + i, d) = scale * accQ;
                }
            }
            for (size_t j = 0; j < S; ++j) {
                for (size_t d = 0; d < D; ++d) {
                    double accK = 0.0;
                    for (size_t i = 0; i < S; ++i) accK += d_scores(b, i * S + j) * last_q_flat(b * S + i, d);
                    dK_flat(b * S + j, d) = scale * accK;
                }
            }
        }
        // dV[b, j, d] = sum_i d_attn_out[b, i, d] * attn[b, i, j]
        // (computed in the per-token-flatten block below)
        Tensor dV_flat(B * S, D);
        for (size_t b = 0; b < B; ++b)
            for (size_t j = 0; j < S; ++j)
                for (size_t d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (size_t i = 0; i < S; ++i) acc += d_attn_out_flat(b * S + i, d) * last_attn_(b, i * S + j);
                    dV_flat(b * S + j, d) = acc;
                }
        // Back to flat shape (B, S*D)
        Tensor dQ(B, S * D), dK(B, S * D), dV(B, S * D);
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d) {
                    dQ(b, s * D + d) = dQ_flat(b * S + s, d);
                    dK(b, s * D + d) = dK_flat(b * S + s, d);
                    dV(b, s * D + d) = dV_flat(b * S + s, d);
                }

        // gW_q += dQ^T @ z1, gb_q += sum dQ per channel
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                double acc = 0.0;
                for (size_t b = 0; b < B; ++b)
                    for (size_t s = 0; s < S; ++s)
                        acc += dQ(b, s * D + i) * last_z1_(b, s * D + j);
                grad_W_q(i, j) += acc;
            }
        }
        for (size_t i = 0; i < D; ++i) {
            double acc = 0.0;
            for (size_t b = 0; b < B; ++b)
                for (size_t s = 0; s < S; ++s)
                    acc += dQ(b, s * D + i);
            grad_b_q(0, i) += acc;
        }
        // gW_k += dK^T @ z1, gb_k += sum dK
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j < D; ++j) {
                double acc = 0.0;
                for (size_t b = 0; b < B; ++b)
                    for (size_t s = 0; s < S; ++s)
                        acc += dK(b, s * D + i) * last_z1_(b, s * D + j);
                grad_W_k(i, j) += acc;
            }
        for (size_t i = 0; i < D; ++i) {
            double acc = 0.0;
            for (size_t b = 0; b < B; ++b)
                for (size_t s = 0; s < S; ++s)
                    acc += dK(b, s * D + i);
            grad_b_k(0, i) += acc;
        }
        // gW_v += dV^T @ z1, gb_v += sum dV
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j < D; ++j) {
                double acc = 0.0;
                for (size_t b = 0; b < B; ++b)
                    for (size_t s = 0; s < S; ++s)
                        acc += dV(b, s * D + i) * last_z1_(b, s * D + j);
                grad_W_v(i, j) += acc;
            }
        for (size_t i = 0; i < D; ++i) {
            double acc = 0.0;
            for (size_t b = 0; b < B; ++b)
                for (size_t s = 0; s < S; ++s)
                    acc += dV(b, s * D + i);
            grad_b_v(0, i) += acc;
        }

        // d_z1 = dQ @ W_q + dK @ W_k + dV @ W_v (sum across Q/K/V paths).
        // Apply per-token: d_z1_flat[i, d] = sum_{dp} dX[i, dp] * W_x[dp, d]
        // for X in {Q, K, V}.
        Tensor d_z1_flat(B * S, D);
        for (size_t i = 0; i < B * S; ++i) {
            for (size_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (size_t dp = 0; dp < D; ++dp) {
                    acc += dQ_flat(i, dp) * W_q(dp, d);
                    acc += dK_flat(i, dp) * W_k(dp, d);
                    acc += dV_flat(i, dp) * W_v(dp, d);
                }
                d_z1_flat(i, d) = acc;
            }
        }
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < S; ++s)
                for (size_t d = 0; d < D; ++d)
                    d_z1(b, s * D + d) = d_z1_flat(b * S + s, d);
    }
    last_d_z1_ = d_z1.clone();

    // Backward through ln1: d_input += ln1_.backward(d_z1)
    // LN was called on (B*S, D); pass the same shape back.
    Tensor d_z1_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_z1_flat(b * S + s, d) = d_z1(b, s * D + d);
    Tensor d_ln1_flat = ln1_.backward(d_z1_flat, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_input(b, s * D + d) += d_ln1_flat(b * S + s, d);

    last_d_input_ = d_input;
    return d_input;
}

void SparseMixerBlock::update_weights(double learning_rate) {
    if (mode_ == LINEAR_MIXING) {
        mixer_->update_weights(learning_rate);
    } else {
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                W_q(i, j) -= learning_rate * grad_W_q(i, j);
                W_k(i, j) -= learning_rate * grad_W_k(i, j);
                W_v(i, j) -= learning_rate * grad_W_v(i, j);
                W_o(i, j) -= learning_rate * grad_W_o(i, j);
            }
            b_q(0, i) -= learning_rate * grad_b_q(0, i);
            b_k(0, i) -= learning_rate * grad_b_k(0, i);
            b_v(0, i) -= learning_rate * grad_b_v(0, i);
            b_o(0, i) -= learning_rate * grad_b_o(0, i);
        }
    }
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    if (has_moe_) {
        ffn_->update_weights(learning_rate);
    } else {
        ffn_dense1_.update_weights(learning_rate);
        ffn_dense2_.update_weights(learning_rate);
    }
}

void SparseMixerBlock::zero_grad() {
    if (mode_ == LINEAR_MIXING) {
        mixer_->zero_grad();
    } else {
        grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
        grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);
    }
    ln1_.zero_grad();
    ln2_.zero_grad();
    if (has_moe_) {
        ffn_->zero_grad();
    } else {
        ffn_dense1_.zero_grad();
        ffn_dense2_.zero_grad();
    }
    last_d_input_ = Tensor(0, 0);
}

std::vector<Tensor*> SparseMixerBlock::parameters() {
    std::vector<Tensor*> out;
    if (mode_ == LINEAR_MIXING) {
        for (Tensor* p : mixer_->parameters()) out.push_back(p);
    } else {
        out.push_back(&W_q); out.push_back(&W_k); out.push_back(&W_v); out.push_back(&W_o);
        out.push_back(&b_q); out.push_back(&b_k); out.push_back(&b_v); out.push_back(&b_o);
    }
    for (Tensor* p : ln1_.parameters()) out.push_back(p);
    for (Tensor* p : ln2_.parameters()) out.push_back(p);
    if (has_moe_) {
        for (Tensor* p : ffn_->parameters()) out.push_back(p);
    } else {
        for (Tensor* p : ffn_dense1_.parameters()) out.push_back(p);
        for (Tensor* p : ffn_dense2_.parameters()) out.push_back(p);
    }
    return out;
}

std::vector<Tensor*> SparseMixerBlock::gradients() {
    std::vector<Tensor*> out;
    if (mode_ == LINEAR_MIXING) {
        for (Tensor* g : mixer_->gradients()) out.push_back(g);
    } else {
        out.push_back(&grad_W_q); out.push_back(&grad_W_k); out.push_back(&grad_W_v); out.push_back(&grad_W_o);
        out.push_back(&grad_b_q); out.push_back(&grad_b_k); out.push_back(&grad_b_v); out.push_back(&grad_b_o);
    }
    for (Tensor* g : ln1_.gradients()) out.push_back(g);
    for (Tensor* g : ln2_.gradients()) out.push_back(g);
    if (has_moe_) {
        for (Tensor* g : ffn_->gradients()) out.push_back(g);
    } else {
        for (Tensor* g : ffn_dense1_.gradients()) out.push_back(g);
        for (Tensor* g : ffn_dense2_.gradients()) out.push_back(g);
    }
    return out;
}

Tensor SparseMixerBlock::get_weights() const {
    if (mode_ == LINEAR_MIXING) return mixer_->get_weights();
    return W_q;
}

Tensor SparseMixerBlock::get_gradients() const {
    if (mode_ == LINEAR_MIXING) return mixer_->get_gradients();
    return grad_W_q;
}

// (Inline GELU' computed where needed; no separate helper.)

// ============================================================================
// SparseMixerModel
// ============================================================================

SparseMixerModel::SparseMixerModel(size_t input_dim, size_t d_model, size_t output_dim,
                                   size_t seq_len, size_t num_layers,
                                   size_t num_attention_layers, size_t num_experts,
                                   size_t expert_hidden, double capacity_factor,
                                   size_t ffn_mult)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      seq_len_(seq_len), num_layers_(num_layers),
      num_attention_layers_(num_attention_layers),
      input_proj_(input_dim, d_model),
      head_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0) throw std::invalid_argument("SparseMixerModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("SparseMixerModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("SparseMixerModel: output_dim must be > 0");
    if (seq_len == 0) throw std::invalid_argument("SparseMixerModel: seq_len must be > 0");
    if (num_attention_layers > num_layers)
        throw std::invalid_argument("SparseMixerModel: num_attention_layers must be <= num_layers");

    for (size_t i = 0; i < num_layers; ++i) {
        size_t layers_from_top = num_layers - 1 - i;
        SparseMixerBlock::Mode m = (layers_from_top < num_attention_layers)
                                       ? SparseMixerBlock::SELF_ATTENTION
                                       : SparseMixerBlock::LINEAR_MIXING;
        blocks_.emplace_back(m, d_model, seq_len, num_experts, expert_hidden,
                              capacity_factor, ffn_mult);
    }
}

Tensor SparseMixerModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    size_t B = input.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // input: (B, input_dim * S)  — flatten (B, S, input_dim) → (B, S*input_dim)
    // Reshape to (B, S, input_dim) implicitly via index arithmetic.
    // Input projection: per-token Dense(input_dim → d_model).
    // z_proj[b, s, d] = sum_j input[b, s, j] * W[d, j] + b[d]
    Tensor z_proj(B, S * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double acc = input_proj_.bias(0, d);
                for (size_t j = 0; j < input_dim_; ++j) {
                    acc += input(b, s * input_dim_ + j) * input_proj_.weights(d, j);
                }
                z_proj(b, s * D + d) = acc;
            }
        }
    }
    last_z_proj_ = z_proj.clone();

    // Stack of blocks
    Tensor cur = z_proj;
    for (size_t i = 0; i < num_layers_; ++i) {
        cur = blocks_[i].forward(cur);
    }
    last_blocks_out_ = cur.clone();

    // Pre-head LayerNorm: per-token LN, reshape (B, S*D) → (B*S, D), normalize,
    // keep flat shape. (B, S*D) → (B*S, D) → ln.forward → (B*S, D) → (B, S*D).
    Tensor head_ln_in(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                head_ln_in(b * S + s, d) = cur(b, s * D + d);
    Tensor head_ln_out = head_ln_.forward(head_ln_in);
    Tensor h(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                h(b, s * D + d) = head_ln_out(b * S + s, d);
    last_head_ln_ = h.clone();

    // Mean-pool over S → (B, D)
    Tensor pooled(B, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (size_t s = 0; s < S; ++s) acc += h(b, s * D + d);
            pooled(b, d) = acc / (double)S;
        }
    }
    last_pooled_ = pooled.clone();

    // Classifier
    Tensor logits = pooled * classifier_.weights.transpose();
    for (size_t b = 0; b < B; ++b)
        for (size_t o = 0; o < output_dim_; ++o)
            logits(b, o) += classifier_.bias(0, o);
    last_logits_ = logits.clone();
    return logits;
}

Tensor SparseMixerModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t B = grad_output.rows;
    size_t S = seq_len_;
    size_t D = d_model_;

    // Classifier backward
    input_proj_.zero_grad();
    head_ln_.zero_grad();
    classifier_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();

    Tensor d_pooled = grad_output * classifier_.weights;
    // gW_classifier += grad_output^T @ pooled; gb += sum grad_output
    for (size_t o = 0; o < output_dim_; ++o) {
        for (size_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (size_t b = 0; b < B; ++b) acc += grad_output(b, o) * last_pooled_(b, d);
            classifier_.grad_weights(o, d) += acc;
        }
        double acc = 0.0;
        for (size_t b = 0; b < B; ++b) acc += grad_output(b, o);
        classifier_.grad_bias(0, o) += acc;
    }

    // d_head_ln: each row of head_ln output received d_pooled / S (mean pool gradient).
    Tensor d_head_ln(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < S * D; ++i)
            d_head_ln(b, i) = 0.0;
    for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < D; ++d) {
            double v = d_pooled(b, d) / (double)S;
            for (size_t s = 0; s < S; ++s) d_head_ln(b, s * D + d) += v;
        }
    // Reshape for head_ln_.backward (was called on (B*S, D))
    Tensor d_head_ln_flat(B * S, D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_head_ln_flat(b * S + s, d) = d_head_ln(b, s * D + d);
    Tensor d_blocks_out_flat = head_ln_.backward(d_head_ln_flat, 0.0);
    Tensor d_blocks_out(B, S * D);
    for (size_t b = 0; b < B; ++b)
        for (size_t s = 0; s < S; ++s)
            for (size_t d = 0; d < D; ++d)
                d_blocks_out(b, s * D + d) = d_blocks_out_flat(b * S + s, d);

    // Stack of blocks backward (reverse order)
    for (size_t i = num_layers_; i > 0; --i) {
        d_blocks_out = blocks_[i - 1].backward(d_blocks_out, 0.0);
    }

    // Input projection backward
    // d_z_proj[b, s, d] = d_blocks_out[b, s, d]
    // gW_proj[d, j] += sum_{b, s} d_z_proj[b, s, d] * input[b, s, j]
    // gb_proj[d] += sum_{b, s} d_z_proj[b, s, d]
    // d_input[b, s, j] = sum_d d_z_proj[b, s, d] * W[d, j]
    Tensor d_input(B, input_dim_ * S);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < input_dim_ * S; ++i)
            d_input(b, i) = 0.0;
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < S; ++s) {
            for (size_t d = 0; d < D; ++d) {
                double g = d_blocks_out(b, s * D + d);
                for (size_t j = 0; j < input_dim_; ++j) {
                    input_proj_.grad_weights(d, j) += g * last_input_(b, s * input_dim_ + j);
                    d_input(b, s * input_dim_ + j) += g * input_proj_.weights(d, j);
                }
                input_proj_.grad_bias(0, d) += g;
            }
        }
    }

    last_d_input_ = d_input;
    return d_input;
}

void SparseMixerModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    head_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
    for (auto& b : blocks_) b.update_weights(learning_rate);
}

void SparseMixerModel::zero_grad() {
    input_proj_.zero_grad();
    head_ln_.zero_grad();
    classifier_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
    last_d_input_ = Tensor(0, 0);
}

std::vector<Tensor*> SparseMixerModel::parameters() {
    std::vector<Tensor*> out;
    for (Tensor* p : input_proj_.parameters()) out.push_back(p);
    for (Tensor* p : head_ln_.parameters()) out.push_back(p);
    for (Tensor* p : classifier_.parameters()) out.push_back(p);
    for (auto& b : blocks_) for (Tensor* p : b.parameters()) out.push_back(p);
    return out;
}

std::vector<Tensor*> SparseMixerModel::gradients() {
    std::vector<Tensor*> out;
    for (Tensor* g : input_proj_.gradients()) out.push_back(g);
    for (Tensor* g : head_ln_.gradients()) out.push_back(g);
    for (Tensor* g : classifier_.gradients()) out.push_back(g);
    for (auto& b : blocks_) for (Tensor* g : b.gradients()) out.push_back(g);
    return out;
}