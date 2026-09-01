#include "ttt_linear.h"
#include "../../activations/activations.h"
#include <random>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================================
// TTT-Linear implementation
// ============================================================================
//
// Forward (per token t, all stored as flat 1D tensors):
//
//   z_t   = W_in · input_t + b_in                 ∈ R^d_inner
//   err_t = W_{t-1} · z_t  -  z_t                ∈ R^d_inner   (the "innovation")
//   W_t   = W_{t-1} - (η/d) · err_t ⊗ z_t        where d = ||z_t||² + λ
//   o_t   = W_t · z_t + b                        ∈ R^d_inner
//   y_t   = W_out · o_t + b_out                   ∈ R^d_model
//
// Backward (reverse-time, full BPTT through the per-token update):
//
//   dz_t = grad_o_pre[t] · W_t^T                                 (direct path)
//        + sum_{i, j} dL/dW_t[i, j] · ∂W_t[i, j]/∂z_t[k]        (recurrence via update rule)
//
//   dL/dW_t = grad_o_pre[t] ⊗ z_t                               (direct at step t)
//           + A_{t+1}^T · dL/dW_{t+1}                            (chain from W_{t+1})
//
//   where A_t = I - (η/d) · z_t z_t^T (Jacobian ∂W_t/∂W_{t-1}).
//
//   dz_t contribution from dL/dW_t:
//     ∂W_t[i,j]/∂z_t[k] = -(η/d)·[(W_{t-1}[i,k]-δ_{ik})·z_t[j] + err_t[i]·δ_{jk}]
//                       + η · err_t[i] · z_t[j] · 2z_t[k] / d²
//
//   Then z_total = W_in^T · dz_t via Dense::backward.

void TTTLinear::initialize_state() {
    std::mt19937 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(static_cast<double>(d_inner_)));
    W_state_ = Tensor(d_inner_, d_inner_);
    for (size_t i = 0; i < d_inner_; ++i) {
        for (size_t j = 0; j < d_inner_; ++j) {
            W_state_(i, j) = nd(rng);
        }
    }
}

TTTLinear::TTTLinear(size_t d_model, size_t d_inner,
                     double eta, double lambda_reg)
    : d_model_(d_model),
      d_inner_(d_inner == 0 ? d_model : d_inner),
      eta_(eta),
      lambda_reg_(lambda_reg),
      W_in_(d_model, d_inner),
      W_out_(d_inner, d_model),
      bias_(1, d_inner) {
    if (d_model == 0) {
        throw std::invalid_argument("TTTLinear: d_model must be > 0");
    }
    if (d_inner == 0) {
        throw std::invalid_argument("TTTLinear: d_inner must be > 0 (or 0 to default to d_model)");
    }
    if (eta_ <= 0.0) {
        throw std::invalid_argument("TTTLinear: eta must be > 0");
    }
    if (lambda_reg_ < 0.0) {
        throw std::invalid_argument("TTTLinear: lambda_reg must be >= 0");
    }

    grad_W_in_w_ = Tensor(d_inner, d_model);
    grad_W_in_b_ = Tensor(1, d_inner);
    grad_W_out_w_ = Tensor(d_model, d_inner);
    grad_W_out_b_ = Tensor(1, d_model);
    grad_bias_ = Tensor(1, d_inner);

    std::fill(bias_.data.begin(), bias_.data.end(), 0.0);

    initialize_state();
    zero_grad();
}

void TTTLinear::reset_state() {
    initialize_state();
}

Tensor TTTLinear::apply_step(const Tensor& W_prev, const Tensor& z_row,
                              double eta, double lambda_reg) {
    const size_t d = W_prev.rows;
    Tensor W_new = W_prev.clone();

    Tensor err(1, d);
    double z_norm_sq = 0.0;
    for (size_t k = 0; k < d; ++k) z_norm_sq += z_row(0, k) * z_row(0, k);
    const double denom = z_norm_sq + lambda_reg;

    for (size_t i = 0; i < d; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < d; ++j) {
            acc += W_prev(i, j) * z_row(0, j);
        }
        err(0, i) = acc - z_row(0, i);
    }

    const double scale = eta / denom;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            W_new(i, j) -= scale * err(0, i) * z_row(0, j);
        }
    }
    return W_new;
}

Tensor TTTLinear::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::runtime_error("TTTLinear::forward: input cols mismatch");
    }
    const size_t T = input.rows;
    last_T_ = T;
    last_input_ = input.clone();

    Tensor z = W_in_.forward(input);
    last_z_ = z.clone();

    const size_t d = d_inner_;
    last_W_t_ = Tensor(1, (T + 1) * d * d);
    last_o_pre_ = Tensor(T, d);
    last_resid_ = Tensor(T, d);

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            last_W_t_(0, i * d + j) = W_state_(i, j);
        }
    }

    for (size_t t = 0; t < T; ++t) {
        Tensor z_row(1, d);
        for (size_t k = 0; k < d; ++k) z_row(0, k) = z(t, k);

        Tensor W_prev(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W_prev(i, j) = last_W_t_(0, t * d * d + i * d + j);
            }
        }

        // err_t = W_prev · z_t - z_t
        Tensor err(1, d);
        double z_norm_sq = 0.0;
        for (size_t k = 0; k < d; ++k) z_norm_sq += z_row(0, k) * z_row(0, k);
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += W_prev(i, j) * z_row(0, j);
            }
            err(0, i) = acc - z_row(0, i);
            last_resid_(t, i) = err(0, i);
        }

        // W_t = W_prev - η · err ⊗ z / denom
        const double denom = z_norm_sq + lambda_reg_;
        const double scale = eta_ / denom;
        Tensor W_new = W_prev.clone();
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                W_new(i, j) -= scale * err(0, i) * z_row(0, j);
            }
        }

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                last_W_t_(0, (t + 1) * d * d + i * d + j) = W_new(i, j);
            }
        }

        // o_pre_t = W_t · z_t
        for (size_t i = 0; i < d; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j) {
                acc += W_new(i, j) * z_row(0, j);
            }
            last_o_pre_(t, i) = acc;
        }
    }

    Tensor o_with_bias = last_o_pre_.clone();
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            o_with_bias(t, i) += bias_(0, i);
        }
    }

    Tensor output = W_out_.forward(o_with_bias);
    return output;
}

Tensor TTTLinear::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_T_ || grad_output.cols != d_model_) {
        throw std::runtime_error("TTTLinear::backward: grad_output shape mismatch");
    }
    const size_t T = last_T_;
    const size_t d = d_inner_;

    // Step 1: backward through W_out (Dense::backward already accumulates grad_weights and grad_bias)
    Tensor grad_o = W_out_.backward(grad_output, 0.0);  // (T, d_inner) = grad_o_with_bias

    // Step 2: backward through bias add
    // grad_o_pre[t, i] = grad_o[t, i]
    Tensor grad_o_pre = grad_o.clone();
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            grad_bias_(0, i) += grad_o(t, i);
        }
    }

    // Step 3: per-token dW_step[t] = grad_o_pre[t] ⊗ z_t (direct gradient on W_t from o_pre[t])
    Tensor dW_step(1, T * d * d);
    dW_step.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                dW_step(0, t * d * d + i * d + j) = grad_o_pre(t, i) * last_z_(t, j);
            }
        }
    }

    // Step 4: dz_t = direct + recurrence contributions, accumulated through reverse-time loop.
    Tensor dz_total(T, d);
    dz_total.fill(0.0);

    // Direct dz path: o_pre[t, i] = sum_j W_t[i, j] · z_t[j] → dz_t[k] += grad_o_pre[t, i] · W_t[i, k]
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d; ++k) {
            for (size_t i = 0; i < d; ++i) {
                const double W_ki = last_W_t_(0, (t + 1) * d * d + i * d + k);
                dz_total(t, k) += grad_o_pre(t, i) * W_ki;
            }
        }
    }

    // Reverse-time BPTT through the recurrence:
    //   dL/dW_t = dW_step[t] + A_{t+1}^T · dL/dW_{t+1}
    //   dL/dW_{t-1} = A_t^T · dL/dW_t
    //
    // dW_total entering iter t = A_{t+1}^T · dL/dW_{t+1} (chain from future)
    //   → dL/dW_t = dW_step[t] + dW_total
    //   → dL/dW_{t-1} = A_t^T · dL/dW_t (becomes dW_total for next iter)
    //
    // At each iter, also compute dz contribution from dL/dW_t via the update rule.

    Tensor dW_total(d, d);  // represents "A_{t+1}^T · dL/dW_{t+1}" entering iter t
    dW_total.fill(0.0);

    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 0; --t_signed) {
        const size_t t = static_cast<size_t>(t_signed);

        // Compute z_t's stats
        double z_norm_sq = 0.0;
        for (size_t k = 0; k < d; ++k) z_norm_sq += last_z_(t, k) * last_z_(t, k);
        const double denom = z_norm_sq + lambda_reg_;
        const double a_scale = eta_ / denom;

        // dL/dW_t = dW_step[t] + dW_total
        Tensor dL_dW_t(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                dL_dW_t(i, j) = dW_step(0, t * d * d + i * d + j) + dW_total(i, j);
            }
        }

        // dz_t contribution from dL/dW_t (via ∂W_t/∂z_t)
        //   ∂W_t[i,j]/∂z_t[k] = -(η/d)·[(W_{t-1}[i,k]-δ_{ik})·z_t[j] + err_t[i]·δ_{jk}]
        //                       + η · err_t[i] · z_t[j] · 2z_t[k] / d²
        for (size_t k = 0; k < d; ++k) {
            double dz_acc = 0.0;
            for (size_t i = 0; i < d; ++i) {
                const double err_i = last_resid_(t, i);
                const double Wprev_ik = last_W_t_(0, t * d * d + i * d + k);
                const double Wik_minus_delta = Wprev_ik - (i == k ? 1.0 : 0.0);
                for (size_t j = 0; j < d; ++j) {
                    const double W_t_ij = dL_dW_t(i, j);
                    const double z_t_j = last_z_(t, j);
                    const double z_t_k = last_z_(t, k);
                    double contrib = -a_scale * Wik_minus_delta * z_t_j;
                    if (j == k) contrib -= a_scale * err_i;
                    contrib += eta_ * err_i * z_t_j * 2.0 * z_t_k / (denom * denom);
                    dz_acc += W_t_ij * contrib;
                }
            }
            dz_total(t, k) += dz_acc;
        }

        // dL/dW_{t-1} = A_t^T · dL_dW_t
        // A_t[k, i] = δ_{ki} - a_scale · z_t[k] · z_t[i]
        // A_t^T[i, k] = A_t[k, i] = δ_{ki} - a_scale · z_t[k] · z_t[i]
        Tensor dW_new(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double acc = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    const double A_ki = (k == i ? 1.0 : 0.0) - a_scale * last_z_(t, k) * last_z_(t, i);
                    acc += A_ki * dL_dW_t(k, j);
                }
                dW_new(i, j) = acc;
            }
        }
        dW_total = dW_new;
    }

    // Step 5: backward through W_in (Dense::backward handles accumulation)
    Tensor grad_input = W_in_.backward(dz_total, 0.0);

    return grad_input;
}

void TTTLinear::update_weights(double learning_rate) {
    W_in_.update_weights(learning_rate);
    W_out_.update_weights(learning_rate);
    for (size_t i = 0; i < d_inner_; ++i) {
        bias_(0, i) -= learning_rate * grad_bias_(0, i);
    }
}

void TTTLinear::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    grad_bias_.fill(0.0);
}

std::vector<Tensor*> TTTLinear::parameters() {
    return { &W_in_.weights, &W_in_.bias, &W_out_.weights, &W_out_.bias, &bias_ };
}

std::vector<Tensor*> TTTLinear::gradients() {
    return { &W_in_.grad_weights, &W_in_.grad_bias, &W_out_.grad_weights, &W_out_.grad_bias, &grad_bias_ };
}

Tensor TTTLinear::get_weights() const {
    return W_state_.clone();
}

Tensor TTTLinear::get_gradients() const {
    return W_state_.clone();
}

// ============================================================================
// TTTLinearModel — convenience wrapper
// ============================================================================

TTTLinearModel::TTTLinearModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
                               double eta, double lambda_reg)
    : layer1_(hidden_dim, hidden_dim, eta, lambda_reg),
      layer2_(hidden_dim, hidden_dim, eta, lambda_reg),
      proj_in_(input_dim, hidden_dim),
      proj_out_(hidden_dim, output_dim) {}

Tensor TTTLinearModel::forward(const Tensor& input) {
    Tensor h = proj_in_.forward(input);
    h = layer1_.forward(h);
    h = layer2_.forward(h);
    return proj_out_.forward(h);
}

Tensor TTTLinearModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad_h = proj_out_.backward(grad_output, learning_rate);
    grad_h = layer2_.backward(grad_h, learning_rate);
    grad_h = layer1_.backward(grad_h, learning_rate);
    return proj_in_.backward(grad_h, learning_rate);
}

void TTTLinearModel::update_weights(double learning_rate) {
    proj_out_.update_weights(learning_rate);
    layer2_.update_weights(learning_rate);
    layer1_.update_weights(learning_rate);
    proj_in_.update_weights(learning_rate);
}

void TTTLinearModel::zero_grad() {
    proj_out_.zero_grad();
    layer2_.zero_grad();
    layer1_.zero_grad();
    proj_in_.zero_grad();
}

std::vector<Tensor*> TTTLinearModel::parameters() {
    auto p = proj_in_.parameters();
    auto p2 = layer1_.parameters();
    auto p3 = layer2_.parameters();
    auto p4 = proj_out_.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    p.insert(p.end(), p4.begin(), p4.end());
    return p;
}

std::vector<Tensor*> TTTLinearModel::gradients() {
    auto g = proj_in_.gradients();
    auto g2 = layer1_.gradients();
    auto g3 = layer2_.gradients();
    auto g4 = proj_out_.gradients();
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), g3.begin(), g3.end());
    g.insert(g.end(), g4.begin(), g4.end());
    return g;
}