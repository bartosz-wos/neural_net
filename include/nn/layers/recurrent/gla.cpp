#include "gla.h"
#include "../../core/tensor.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>

// Initialize projections to a small random scale — small enough that the
// recurrence stays numerically stable for forward checks, large enough
// that gradient checks are non-vacuous.
static Tensor gla_rand_init(size_t rows, size_t cols) {
    Tensor t(rows, cols);
    double s = 0.5 / std::sqrt(static_cast<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t[i][j] = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0 * s;
        }
    }
    return t;
}

GatedLinearAttention::GatedLinearAttention(size_t d_model, size_t n_heads, size_t head_dim)
    : W_Q_(d_model, 0), W_K_(d_model, 0), W_V_(d_model, 0),
      W_O_(0, 0), W_gate_(d_model, 0),
      d_model_(d_model),
      n_heads_(n_heads),
      head_dim_(head_dim == 0 ? d_model / n_heads : head_dim),
      d_inner_(n_heads * (head_dim == 0 ? d_model / n_heads : head_dim))
{
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model and n_heads must be > 0");
    }
    if (head_dim_ == 0 || d_model_ % head_dim_ != 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model must divide evenly by head_dim");
    }
    // Standard convention: d_inner = d_model (so head_dim = d_model / n_heads).
    d_inner_ = d_model;

    // Build Dense projections (x · W^T + b convention)
    W_Q_   = Dense(d_model, d_inner_);
    W_K_   = Dense(d_model, d_inner_);
    W_V_   = Dense(d_model, d_inner_);
    W_O_   = Dense(d_inner_, d_model);
    W_gate_ = Dense(d_model, n_heads_);  // per-head α_t = sigmoid(W_gate · x_t)

    // Override Dense default init with a Xavier-ish small scale
    W_Q_.weights    = gla_rand_init(d_inner_, d_model_);
    W_K_.weights    = gla_rand_init(d_inner_, d_model_);
    W_V_.weights    = gla_rand_init(d_inner_, d_model_);
    W_O_.weights    = gla_rand_init(d_model_, d_inner_);
    W_gate_.weights = gla_rand_init(n_heads_, d_model_);
    // Biases default to zero in Dense
}

// Forward pass (Task 3): per-head gated linear-attention recurrence
Tensor GatedLinearAttention::forward(const Tensor& input) {
    size_t T = input.rows;
    if (T == 0) return Tensor(0, d_model_);
    if (input.cols != d_model_) {
        throw std::invalid_argument("GatedLinearAttention::forward: input.cols must equal d_model");
    }

    cache_x_ = input.clone();

    // Project Q, K, V, gate (gate pre-sigmoid)
    cache_q_ = W_Q_.forward(input);     // (T, d_inner)
    cache_k_ = W_K_.forward(input);     // (T, d_inner)
    cache_v_ = W_V_.forward(input);     // (T, d_inner)
    Tensor gate_raw = W_gate_.forward(input);  // (T, n_heads)
    cache_gate_pre_ = Tensor(T, n_heads_);
    cache_gate_ = Tensor(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double g_raw = gate_raw[t][h];
            cache_gate_pre_[t][h] = g_raw;
            // Stable sigmoid
            if (g_raw >= 0.0) {
                cache_gate_[t][h] = 1.0 / (1.0 + std::exp(-g_raw));
            } else {
                double ez = std::exp(g_raw);
                cache_gate_[t][h] = ez / (1.0 + ez);
            }
        }
    }

    // Per-head gated linear-attention recurrence
    cache_S_.clear();
    cache_S_.resize(T);
    cache_S_[0] = Tensor(n_heads_, head_dim_ * head_dim_);  // zero state for t=0

    Tensor out_concat(T, d_inner_);
    Tensor current_state(n_heads_, head_dim_ * head_dim_);  // S_0 = 0

    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double alpha = cache_gate_[t][h];

            // Apply gate: current_state[h] *= alpha
            for (size_t i = 0; i < head_dim_ * head_dim_; ++i) {
                current_state[h][i] *= alpha;
            }

            // Add outer product: current_state[h] += outer(k_t[h], v_t[h])
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k = cache_k_[t][h * head_dim_ + i];
                    double v = cache_v_[t][h * head_dim_ + j];
                    current_state[h][i * head_dim_ + j] += k * v;
                }
            }
        }
        cache_S_[t] = current_state.clone();

        // Output: o_t[h] = S_t[h] · q_t[h]
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += current_state[h][i * head_dim_ + j] *
                           cache_q_[t][h * head_dim_ + j];
                }
                out_concat[t][h * head_dim_ + i] = sum;
            }
        }
    }

    cache_concat_o_ = out_concat.clone();
    Tensor out = W_O_.forward(cache_concat_o_);  // (T, d_model)
    return out;
}

Tensor GatedLinearAttention::last_state() const {
    if (cache_S_.empty()) return Tensor(0, 0);
    return cache_S_.back();  // (n_heads, head_dim * head_dim)
}

// Backward pass (Task 4)
Tensor GatedLinearAttention::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("GatedLinearAttention::backward: grad_output.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    // Step 1: backward through W_O to get grad_concat_o (gradient flowing back
    // to the concatenated per-head outputs).
    Tensor grad_concat_o = W_O_.backward(grad_output, learning_rate);  // (T, d_inner)

    // Allocate gradient buffers
    grad_q_ = Tensor(T, d_inner_);
    grad_k_ = Tensor(T, d_inner_);
    grad_v_ = Tensor(T, d_inner_);
    for (size_t i = 0; i < T * d_inner_; ++i) {
        grad_q_.data[i] = 0.0;
        grad_k_.data[i] = 0.0;
        grad_v_.data[i] = 0.0;
    }

    // grad_S[t][h, i, j] accumulates gradient w.r.t. S_t[h]'s (i, j) entry
    // coming from grad_concat_o and the future carrier.
    Tensor grad_S(T, n_heads_ * head_dim_ * head_dim_);
    for (size_t i = 0; i < T * n_heads_ * head_dim_ * head_dim_; ++i) {
        grad_S.data[i] = 0.0;
    }

    // First pass: grad_S from grad_o_t contribution only (output side).
    //   o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]
    //   => grad_S_t[h, i, j] += grad_o_t[h, i] · q_t[h, j]
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double g_oh_i = grad_concat_o[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    grad_S.data[t * (n_heads_ * head_dim_ * head_dim_)
                                + (h * head_dim_ + i) * head_dim_ + j]
                        += g_oh_i * cache_q_[t][h * head_dim_ + j];
                }
            }
        }
    }

    // Also accumulate grad_q_t from grad_concat_o (the gradient of L w.r.t. q_t)
    //   grad_q_t[h, j] += sum_i S_t[h, i, j] · grad_o_t[h, i]
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    sum += cache_S_[t].data[h * head_dim_ * head_dim_ + i * head_dim_ + j]
                         * grad_concat_o[t][h * head_dim_ + i];
                }
                grad_q_.data[t * d_inner_ + h * head_dim_ + j] += sum;
            }
        }
    }

    // Backward-time recurrence for grad_S.
    // The recurrence S_t = α_t · S_{t-1} + outer(k_t, v_t) gives:
    //   - Carrier: grad_S_{t-1} += α_t · grad_S_t
    //     (because S_t depends linearly on S_{t-1} via α_t multiplication;
    //      so the gradient flowing back to S_{t-1} is α_t · grad_S_t.)
    //   - grad_k_t[h, i] += sum_j grad_S_t[h, i, j] · v_t[h, j]
    //   - grad_v_t[h, j] += sum_i grad_S_t[h, i, j] · k_t[h, i]
    //   - grad_α_t[h] += sum_{i,j} grad_S_t[h, i, j] · S_{t-1}[h, i, j]
    Tensor grad_gate_pre(T, n_heads_);
    for (size_t i = 0; i < T * n_heads_; ++i) grad_gate_pre.data[i] = 0.0;

    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 1; --t_signed) {
        size_t t = static_cast<size_t>(t_signed);
        size_t t_prev = t - 1;
        for (size_t h = 0; h < n_heads_; ++h) {
            double alpha = cache_gate_[t][h];
            const Tensor& Sp = cache_S_[t_prev];

            // grad_α_t[h] += sum_{i,j} grad_S_t[h, i, j] · S_{t-1}[h, i, j]
            double g_alpha = 0.0;
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    g_alpha += grad_S.data[t * (n_heads_ * head_dim_ * head_dim_) + flat]
                             * Sp.data[flat];
                }
            }
            // Chain through sigmoid: ∂α/∂(gate_pre) = α · (1 - α)
            grad_gate_pre.data[t * n_heads_ + h] += g_alpha * alpha * (1.0 - alpha);

            // grad_k_t[h, i] += sum_j grad_S_t[h, i, j] · v_t[h, j]
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S.data[t * (n_heads_ * head_dim_ * head_dim_) + flat]
                         * cache_v_.data[t * d_inner_ + h * head_dim_ + j];
                }
                grad_k_.data[t * d_inner_ + h * head_dim_ + i] += sum;
            }

            // grad_v_t[h, j] += sum_i grad_S_t[h, i, j] · k_t[h, i]
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S.data[t * (n_heads_ * head_dim_ * head_dim_) + flat]
                         * cache_k_.data[t * d_inner_ + h * head_dim_ + i];
                }
                grad_v_.data[t * d_inner_ + h * head_dim_ + j] += sum;
            }

            // Carrier: grad_S_{t-1}[h] += α_t · grad_S_t[h]
            for (size_t i = 0; i < head_dim_ * head_dim_; ++i) {
                grad_S.data[t_prev * (n_heads_ * head_dim_ * head_dim_)
                            + h * head_dim_ * head_dim_ + i]
                    += alpha * grad_S.data[t * (n_heads_ * head_dim_ * head_dim_)
                                           + h * head_dim_ * head_dim_ + i];
            }
        }
    }

    // t=0: S_{-1} = 0, so no contribution from "α_0 · S_{-1}" path.
    // But k_0, v_0 still have gradients from grad_S[0].
    if (T > 0) {
        size_t t = 0;
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S.data[t * (n_heads_ * head_dim_ * head_dim_) + flat]
                         * cache_v_.data[t * d_inner_ + h * head_dim_ + j];
                }
                grad_k_.data[t * d_inner_ + h * head_dim_ + i] += sum;
            }
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S.data[t * (n_heads_ * head_dim_ * head_dim_) + flat]
                         * cache_k_.data[t * d_inner_ + h * head_dim_ + i];
                }
                grad_v_.data[t * d_inner_ + h * head_dim_ + j] += sum;
            }
            // grad_α_0 = 0 (because S_{-1} = 0).
        }
    }

    // Now backprop through Q, K, V, gate Dense projections. Each Dense.backward
    // accumulates grad_weights / grad_bias in the Dense itself and returns
    // grad_input w.r.t. x.
    Tensor grad_x_q = W_Q_.backward(grad_q_, learning_rate);
    Tensor grad_x_k = W_K_.backward(grad_k_, learning_rate);
    Tensor grad_x_v = W_V_.backward(grad_v_, learning_rate);
    Tensor grad_x_g = W_gate_.backward(grad_gate_pre, learning_rate);

    // Sum input gradients from all four projections (Q, K, V, gate)
    Tensor grad_x(T, d_model_);
    for (size_t i = 0; i < T * d_model_; ++i) {
        grad_x.data[i] = grad_x_q.data[i] + grad_x_k.data[i]
                        + grad_x_v.data[i] + grad_x_g.data[i];
    }

    grad_x_ = grad_x.clone();
    return grad_x;
}

void GatedLinearAttention::update_weights(double learning_rate) {
    W_Q_.update_weights(learning_rate);
    W_K_.update_weights(learning_rate);
    W_V_.update_weights(learning_rate);
    W_O_.update_weights(learning_rate);
    W_gate_.update_weights(learning_rate);
}

void GatedLinearAttention::zero_grad() {
    W_Q_.zero_grad();
    W_K_.zero_grad();
    W_V_.zero_grad();
    W_O_.zero_grad();
    W_gate_.zero_grad();
    grad_q_ = Tensor();
    grad_k_ = Tensor();
    grad_v_ = Tensor();
    grad_x_ = Tensor();
}

std::vector<Tensor*> GatedLinearAttention::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_Q_.parameters())    result.push_back(p);
    for (Tensor* p : W_K_.parameters())    result.push_back(p);
    for (Tensor* p : W_V_.parameters())    result.push_back(p);
    for (Tensor* p : W_O_.parameters())    result.push_back(p);
    for (Tensor* p : W_gate_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GatedLinearAttention::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_Q_.gradients())    result.push_back(g);
    for (Tensor* g : W_K_.gradients())    result.push_back(g);
    for (Tensor* g : W_V_.gradients())    result.push_back(g);
    for (Tensor* g : W_O_.gradients())    result.push_back(g);
    for (Tensor* g : W_gate_.gradients()) result.push_back(g);
    return result;
}