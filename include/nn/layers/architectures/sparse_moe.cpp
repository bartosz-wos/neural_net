#include "sparse_moe.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

// ============================================================================
// Construction & initialization
// ============================================================================

SparseMoELayer::SparseMoELayer(size_t d_model, size_t num_experts, size_t k,
                                size_t expert_hidden, double aux_loss_coef)
    : d_model_(d_model), num_experts_(num_experts), k_(k),
      expert_hidden_(expert_hidden == 0 ? 4 * d_model : expert_hidden),
      aux_loss_coef_(aux_loss_coef),
      W_router_(num_experts, d_model),
      b_router_(1, num_experts),
      grad_W_router_(num_experts, d_model),
      grad_b_router_(1, num_experts),
      load_balance_loss_(0.0)
{
    if (k_ > num_experts_) k_ = num_experts_;
    if (k_ < 1) k_ = 1;

    // Xavier init for the router: std = sqrt(2 / (num_experts + d_model))
    {
        std::mt19937 gen(42);
        double std_r = std::sqrt(2.0 / static_cast<double>(num_experts_ + d_model_));
        std::normal_distribution<> dis(0.0, std_r);
        for (size_t i = 0; i < W_router_.rows; ++i)
            for (size_t j = 0; j < W_router_.cols; ++j)
                W_router_(i, j) = dis(gen);
        b_router_.fill(0.0);
    }

    // Initialize experts (2-layer FFN, ReLU hidden).
    // He init (good for ReLU): std = sqrt(2 / fan_in)
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
    mean_gate_prob_.assign(num_experts_, 0.0);
}

// ============================================================================
// softmax_topk: forward pass
//   - logits: (batch, num_experts)
//   - probs : (batch, num_experts) — softmax over the top-k logits per row,
//             all other entries are 0
//   - topk  : (batch, k)            — indices of the top-k experts per row
//
//   Stability: subtract per-row max of the *top-k* logits (not the full row)
//   so the softmax is numerically stable AND respects the top-k mask.
// ============================================================================

void SparseMoELayer::softmax_topk(const Tensor& logits, Tensor& probs,
                                   std::vector<std::vector<size_t>>& topk) {
    size_t batch = logits.rows;
    probs = Tensor::zeros(batch, num_experts_);
    topk.assign(batch, std::vector<size_t>(k_, 0));

    for (size_t b = 0; b < batch; ++b) {
        // 1) Find top-k indices and their values
        std::vector<std::pair<double, size_t>> scored(num_experts_);
        for (size_t e = 0; e < num_experts_; ++e)
            scored[e] = {logits(b, e), e};
        // Partial sort: bring top-k to the front (descending)
        std::partial_sort(scored.begin(), scored.begin() + k_, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // 2) Compute max of top-k for stability
        double maxv = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < k_; ++i)
            if (scored[i].first > maxv) maxv = scored[i].first;

        // 3) exp + sum
        std::vector<double> exps(k_);
        double sumexp = 0.0;
        for (size_t i = 0; i < k_; ++i) {
            exps[i] = std::exp(scored[i].first - maxv);
            sumexp += exps[i];
        }
        if (sumexp < 1e-30) sumexp = 1e-30;

        // 4) Write probs and topk
        for (size_t i = 0; i < k_; ++i) {
            size_t e = scored[i].second;
            probs(b, e) = exps[i] / sumexp;
            topk[b][i] = e;
        }
    }
}

// ============================================================================
// forward
// ============================================================================

Tensor SparseMoELayer::forward(const Tensor& input) {
    size_t batch = input.rows;
    input_ = input.clone();

    // 1) Router: gate_logits = input * W_router^T + b_router   (Dense convention)
    //    gate_logits is (batch, num_experts).
    gate_logits_ = input * W_router_.transpose();
    for (size_t e = 0; e < num_experts_; ++e) {
        double be = b_router_(0, e);
        for (size_t b = 0; b < batch; ++b) {
            gate_logits_(b, e) += be;
        }
    }

    // 2) Softmax-topk → gate_probs (sparse)
    softmax_topk(gate_logits_, gate_probs_, topk_idx_);

    // 3) Run each expert on the FULL batch (in practice one could dispatch
    //    by selected tokens, but for BPTT clarity and to avoid scatter/gather
    //    we run on the whole batch and weight by the gate probs).
    expert_h_pre_.assign(num_experts_, Tensor(batch, expert_hidden_));
    expert_h_act_.assign(num_experts_, Tensor(batch, expert_hidden_));
    expert_outputs_.assign(num_experts_, Tensor(batch, d_model_));

    for (size_t e = 0; e < num_experts_; ++e) {
        // hidden_pre = input @ W1^T + b1  -> (batch, expert_hidden)
        Tensor h_pre = input * W1_[e].transpose();
        for (size_t j = 0; j < expert_hidden_; ++j) {
            double bj = b1_[e](0, j);
            for (size_t b = 0; b < batch; ++b) h_pre(b, j) += bj;
        }
        // ReLU
        Tensor h_act(batch, expert_hidden_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < expert_hidden_; ++j) {
                double v = h_pre(b, j);
                h_act(b, j) = v > 0.0 ? v : 0.0;
            }
        }
        expert_h_pre_[e] = h_pre;
        expert_h_act_[e]  = h_act;
        // out = h_act @ W2^T + b2  -> (batch, d_model)
        Tensor out = h_act * W2_[e].transpose();
        for (size_t j = 0; j < d_model_; ++j) {
            double bj = b2_[e](0, j);
            for (size_t b = 0; b < batch; ++b) out(b, j) += bj;
        }
        expert_outputs_[e] = out;
    }

    // 4) Combine: out[b, :] = sum_e gate_probs[b, e] * expert_outputs_[e][b, :]
    Tensor out(batch, d_model_);
    out.fill(0.0);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t e = 0; e < num_experts_; ++e) {
            double g = gate_probs_(b, e);
            if (g == 0.0) continue;
            for (size_t j = 0; j < d_model_; ++j) {
                out(b, j) += g * expert_outputs_[e](b, j);
            }
        }
    }

    // 5) Load-balance aux loss (Shazeer form, NOT scaled — user multiplies by alpha).
    //    f_e = (# tokens for which e is in top-k) / batch       (dispatch fraction)
    //    p_e = (1/batch) * sum_b gate_probs[b, e]              (mean gate prob)
    //    L_aux_unscaled = num_experts * sum_e f_e * p_e
    //    We return the alpha-scaled form below.
    std::fill(dispatch_frac_.begin(), dispatch_frac_.end(), 0.0);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < k_; ++i) {
            size_t e = topk_idx_[b][i];
            dispatch_frac_[e] += 1.0;
        }
    }
    if (batch > 0) {
        for (size_t e = 0; e < num_experts_; ++e) dispatch_frac_[e] /= static_cast<double>(batch);
    }
    std::fill(mean_gate_prob_.begin(), mean_gate_prob_.end(), 0.0);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t e = 0; e < num_experts_; ++e) {
            mean_gate_prob_[e] += gate_probs_(b, e);
        }
    }
    if (batch > 0) {
        for (size_t e = 0; e < num_experts_; ++e) {
            mean_gate_prob_[e] /= static_cast<double>(batch);
        }
    }
    double unscaled = 0.0;
    for (size_t e = 0; e < num_experts_; ++e) {
        unscaled += dispatch_frac_[e] * mean_gate_prob_[e];
    }
    load_balance_loss_ = aux_loss_coef_ * static_cast<double>(num_experts_) * unscaled;

    return out;
}

// ============================================================================
// backward
// ============================================================================
//
// grad_output: (batch, d_model) — gradient of the model loss w.r.t. the MoE output.
//
// Returns: d_input (batch, d_model) — gradient w.r.t. the MoE input.
// Side-effect: accumulates grad_W_router, grad_b_router, grad_W1/W2/b1/b2.
// ============================================================================

Tensor SparseMoELayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t batch = grad_output.rows;

    // 1) grad w.r.t. gate_probs and per-expert outputs.
    //    out[b, j] = sum_e g[b, e] * exp_e[b, j]
    //    => d g[b, e]            = sum_j grad_output[b, j] * exp_e[b, j]    (for ALL e, even unselected; selected g>0 anyway)
    //    => d exp_e[b, j]        = g[b, e] * grad_output[b, j]              (for ALL e; g=0 for unselected → 0)
    Tensor d_gate_probs = Tensor::zeros(batch, num_experts_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t e = 0; e < num_experts_; ++e) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                s += grad_output(b, j) * expert_outputs_[e](b, j);
            }
            d_gate_probs(b, e) = s;
        }
    }

    // grad into each expert's output
    std::vector<Tensor> d_exp_out(num_experts_, Tensor(batch, d_model_));
    for (size_t b = 0; b < batch; ++b) {
        for (size_t e = 0; e < num_experts_; ++e) {
            double g = gate_probs_(b, e);
            for (size_t j = 0; j < d_model_; ++j) {
                d_exp_out[e](b, j) = g * grad_output(b, j);
            }
        }
    }

    // 2) Backprop each expert (W1/ReLU/W2) with grad w.r.t. its output.
    //    For each expert e:
    //      a) d_h_act = d_exp_out @ W2                    (batch, expert_hidden)
    //      b) d_b2   += sum_b d_exp_out[b, :]            (1, d_model)
    //      c) d_W2   += d_exp_out^T @ h_act              (d_model, expert_hidden)
    //      d) d_h_pre = d_h_act * (h_pre > 0)            (ReLU mask)
    //      e) d_b1   += sum_b d_h_pre[b, :]              (1, expert_hidden)
    //      f) d_W1   += d_h_pre^T @ input                (expert_hidden, d_model)
    //      g) d_input (from this expert) = d_h_pre @ W1  (batch, d_model)
    //
    //    The aux-loss contribution to the experts is: alpha * num_experts * f_e
    //    multiplied by the per-token gate prob chain. We compute it via:
    //      d_logits[b, e] += alpha * num_experts * f_e   (constant w.r.t. logits for selected e)
    //    but since f_e is computed FROM the topk (which is a non-differentiable
    //    argmax), the aux loss is a STOP-GRADIENT on the routing decision and
    //    propagates only through the per-token gate probs. We add this term
    //    to d_gate_probs and use a uniform 1/batch * d_gate_probs[b, e] path
    //    through the softmax-backward. (See Switch Transformer reference.)
    //
    //    For our test/scope: we include the aux-loss term in d_gate_probs:
    //      d_gate_probs[b, e] += alpha * num_experts * dispatch_frac_[e]
    //    (this is the standard Shazeer form, gradient is broadcast to all
    //    tokens equally for expert e, regardless of whether the token picked e).
    //
    //    Actually more correctly: the loss L_aux = alpha * E * sum_e f_e * p_e
    //      = alpha * E * sum_e f_e * (1/N) sum_b g[b, e]
    //    dL/d g[b, e] = alpha * E * f_e / N       (uniform across all b for fixed e)
    //    dL/d f_e      = alpha * E * p_e          (f_e is non-diff through top-k; skip)
    //    So we add to d_gate_probs[b, e]: alpha * E * f_e / N.
    if (batch > 0) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t e = 0; e < num_experts_; ++e) {
                d_gate_probs(b, e) += aux_loss_coef_ * static_cast<double>(num_experts_) *
                                      dispatch_frac_[e] / static_cast<double>(batch);
            }
        }
    }

    Tensor d_input = Tensor::zeros(batch, d_model_);

    for (size_t e = 0; e < num_experts_; ++e) {
        // a) d_h_act = d_exp_out[e] @ W2[e]   (Dense backward: grad_input = grad_output * W)
        //    W2[e] is (d_model, expert_hidden), d_exp_out is (batch, d_model)
        //    d_h_act is (batch, expert_hidden)
        Tensor d_h_act(batch, expert_hidden_);
        d_h_act.fill(0.0);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < expert_hidden_; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < d_model_; ++k) {
                    s += d_exp_out[e](b, k) * W2_[e](k, j);
                }
                d_h_act(b, j) = s;
            }
        }

        // b) d_b2 += sum_b d_exp_out[b, :]
        for (size_t j = 0; j < d_model_; ++j) {
            double s = gb2_[e](0, j);
            for (size_t b = 0; b < batch; ++b) s += d_exp_out[e](b, j);
            gb2_[e](0, j) = s;
        }

        // c) d_W2 += d_exp_out^T @ h_act   (d_model, expert_hidden)
        //    h_act is (batch, expert_hidden), d_exp_out is (batch, d_model).
        //    grad_W2[i, j] = sum_b d_exp_out[b, i] * h_act[b, j]
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < expert_hidden_; ++j) {
                double s = gW2_[e](i, j);
                for (size_t b = 0; b < batch; ++b) {
                    s += d_exp_out[e](b, i) * expert_h_act_[e](b, j);
                }
                gW2_[e](i, j) = s;
            }
        }

        // d) d_h_pre = d_h_act * (h_pre > 0)   (ReLU mask, applied on the pre-activations)
        Tensor d_h_pre(batch, expert_hidden_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < expert_hidden_; ++j) {
                double v = expert_h_pre_[e](b, j);
                d_h_pre(b, j) = d_h_act(b, j) * (v > 0.0 ? 1.0 : 0.0);
            }
        }

        // e) d_b1 += sum_b d_h_pre[b, :]
        for (size_t j = 0; j < expert_hidden_; ++j) {
            double s = gb1_[e](0, j);
            for (size_t b = 0; b < batch; ++b) s += d_h_pre(b, j);
            gb1_[e](0, j) = s;
        }

        // f) d_W1 += d_h_pre^T @ input   (expert_hidden, d_model)
        //    grad_W1[i, j] = sum_b d_h_pre[b, i] * input[b, j]
        for (size_t i = 0; i < expert_hidden_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = gW1_[e](i, j);
                for (size_t b = 0; b < batch; ++b) {
                    s += d_h_pre(b, i) * input_(b, j);
                }
                gW1_[e](i, j) = s;
            }
        }

        // g) d_input (from this expert) += d_h_pre @ W1   (Dense backward)
        //    W1[e] is (expert_hidden, d_model), d_h_pre is (batch, expert_hidden)
        //    d_input_from_e[b, j] = sum_k d_h_pre[b, k] * W1[e](k, j)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < d_model_; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < expert_hidden_; ++k) {
                    s += d_h_pre(b, k) * W1_[e](k, j);
                }
                d_input(b, j) += s;
            }
        }
    }

    // 3) Softmax-topk backward into d_gate_logits.
    //    For each row b:
    //      selected indices S = topk_idx_[b] (size k_)
    //      For e in S:  d_logits[b, e] = g[b,e] * (d_gate_probs[b, e] - sum_{e' in S} g[b, e'] * d_gate_probs[b, e'])
    //      For e not in S: d_logits[b, e] = 0  (the top-k mask is treated as a
    //        hard non-differentiable gate on the routing; the gradient only flows
    //        through the selected indices, which is the standard "hard routing"
    //        treatment. For a softer treatment one could add a per-index softmax
    //        of (1 - one_hot_topk), but that's a different paper.)
    Tensor d_gate_logits = Tensor::zeros(batch, num_experts_);
    for (size_t b = 0; b < batch; ++b) {
        // Compute scalar: sum_{e' in S} g[b, e'] * d_gate_probs[b, e']
        double s = 0.0;
        for (size_t i = 0; i < k_; ++i) {
            size_t e = topk_idx_[b][i];
            s += gate_probs_(b, e) * d_gate_probs(b, e);
        }
        for (size_t i = 0; i < k_; ++i) {
            size_t e = topk_idx_[b][i];
            d_gate_logits(b, e) = gate_probs_(b, e) * (d_gate_probs(b, e) - s);
        }
    }

    // 4) Router Dense backward: gate_logits = input * W_router^T + b_router
    //    a) d_W_router += d_gate_logits^T @ input  (num_experts, d_model)
    //       grad_W_router[e, j] = sum_b d_gate_logits[b, e] * input[b, j]
    //    b) d_b_router += sum_b d_gate_logits[b, :]
    //    c) d_input += d_gate_logits @ W_router   (Dense backward)
    //       d_input_from_router[b, j] = sum_e d_gate_logits[b, e] * W_router[e, j]
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = grad_W_router_(e, j);
            for (size_t b = 0; b < batch; ++b) {
                s += d_gate_logits(b, e) * input_(b, j);
            }
            grad_W_router_(e, j) = s;
        }
    }
    for (size_t e = 0; e < num_experts_; ++e) {
        double s = grad_b_router_(0, e);
        for (size_t b = 0; b < batch; ++b) s += d_gate_logits(b, e);
        grad_b_router_(0, e) = s;
    }
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t e = 0; e < num_experts_; ++e) {
                s += d_gate_logits(b, e) * W_router_(e, j);
            }
            d_input(b, j) += s;
        }
    }

    return d_input;
}

// ============================================================================
// update_weights / zero_grad
// ============================================================================

void SparseMoELayer::update_weights(double learning_rate) {
    // Router
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_router_(e, j) -= learning_rate * grad_W_router_(e, j);
        }
        b_router_(0, e) -= learning_rate * grad_b_router_(0, e);
    }
    // Experts
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t i = 0; i < expert_hidden_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                W1_[e](i, j) -= learning_rate * gW1_[e](i, j);
            }
            b1_[e](0, i) -= learning_rate * gb1_[e](0, i);
        }
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < expert_hidden_; ++j) {
                W2_[e](i, j) -= learning_rate * gW2_[e](i, j);
            }
            b2_[e](0, i) -= learning_rate * gb2_[e](0, i);
        }
    }
}

void SparseMoELayer::zero_grad() {
    grad_W_router_.fill(0.0);
    grad_b_router_.fill(0.0);
    for (size_t e = 0; e < num_experts_; ++e) {
        gW1_[e].fill(0.0);
        gW2_[e].fill(0.0);
        gb1_[e].fill(0.0);
        gb2_[e].fill(0.0);
    }
}

// ============================================================================
// parameters / gradients
// ============================================================================
//
// Order (consistent with parameters() / gradients()):
//   0..num_experts-1:    expert W1, b1
//   num_experts..2*E-1:  expert W2, b2
//   2*E:                 W_router
//   2*E+1:               b_router
// To make tests easier we emit in the order: W_router, b_router, then per expert
// (W1, b1, W2, b2). This matches the parameter layout used by other layers in
// the project (W/b together per logical block).

std::vector<Tensor*> SparseMoELayer::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_router_);
    p.push_back(&b_router_);
    for (size_t e = 0; e < num_experts_; ++e) {
        p.push_back(&W1_[e]);
        p.push_back(&b1_[e]);
        p.push_back(&W2_[e]);
        p.push_back(&b2_[e]);
    }
    return p;
}

std::vector<Tensor*> SparseMoELayer::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_W_router_);
    g.push_back(&grad_b_router_);
    for (size_t e = 0; e < num_experts_; ++e) {
        g.push_back(&gW1_[e]);
        g.push_back(&gb1_[e]);
        g.push_back(&gW2_[e]);
        g.push_back(&gb2_[e]);
    }
    return g;
}

Tensor SparseMoELayer::get_weights() const {
    // Return a (num_experts * (1 + d_model + expert_hidden_), 1) concat for
    // inspection. For simplicity, just return W_router_ flattened.
    return W_router_;
}

Tensor SparseMoELayer::get_gradients() const {
    return grad_W_router_;
}
