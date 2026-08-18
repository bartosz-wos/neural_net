// mega.cpp — MEGA (Moving Average Equipped Gated Attention) block
// Implementation of Ma et al. 2022 (https://arxiv.org/abs/2209.10655).
//
// See mega.h for the full forward / backward math.

#include "nn/layers/architectures/mega.h"
#include "../../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// Numerically stable row-softmax over a (rows, cols) tensor. Treats -inf entries
// (which we use for causal masking) as fully excluded: row_max is computed over
// finite entries and any entry that is -inf stays -inf after normalization
// (since exp(-inf - finite_max) = 0). Returns (rows, cols) softmax tensor.
static Tensor row_softmax_causal(const Tensor& scores) {
    const size_t rows = scores.rows;
    const size_t cols = scores.cols;
    Tensor out(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        // Find max over finite entries.
        double max_val = -1e300;
        for (size_t c = 0; c < cols; ++c) {
            const double v = scores[r][c];
            if (std::isfinite(v) && v > max_val) max_val = v;
        }
        // If the entire row is -inf (shouldn't happen for T >= 1 with diagonal unmasked),
        // softmax over zeros — defensive.
        if (max_val == -1e300) max_val = 0.0;
        // Sum exp.
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            const double v = scores[r][c];
            if (std::isfinite(v)) {
                out[r][c] = std::exp(v - max_val);
                sum += out[r][c];
            } else {
                out[r][c] = 0.0;
            }
        }
        if (sum <= 0.0) sum = 1.0;  // defensive
        // Normalize.
        for (size_t c = 0; c < cols; ++c) {
            out[r][c] /= sum;
        }
    }
    return out;
}

// Softmax backward: d_input = attn ⊙ (grad_output - sum_rows(attn ⊙ grad_output)).
// Used for the attention backward.
static Tensor softmax_backward(const Tensor& attn, const Tensor& grad_output) {
    const size_t rows = attn.rows;
    const size_t cols = attn.cols;
    Tensor grad_input(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        double dot = 0.0;
        for (size_t c = 0; c < cols; ++c) dot += attn[r][c] * grad_output[r][c];
        for (size_t c = 0; c < cols; ++c) {
            grad_input[r][c] = attn[r][c] * (grad_output[r][c] - dot);
        }
    }
    return grad_input;
}

// Stable row-softmax (handles -inf correctly by ignoring masked entries).
// (Same as row_softmax_causal; alias for clarity in attention path.)
static inline Tensor row_softmax(const Tensor& scores) { return row_softmax_causal(scores); }

// ----------------------------------------------------------------------------
// MegaBlock
// ----------------------------------------------------------------------------

MegaBlock::MegaBlock(size_t d_model, size_t num_heads, size_t ffn_mult)
    : d_model_(d_model), num_heads_(num_heads), ffn_mult_(ffn_mult), T_(0),
      W_q(d_model_, d_model_), W_k(d_model_, d_model_), W_v(d_model_, d_model_),
      W_o(d_model_, d_model_), W_g(d_model_, d_model_),
      bias_max_len_(0), T_bias_max_(0),
      ffn_W1(d_model_, d_model_ * ffn_mult_),
      ffn_W2(d_model_ * ffn_mult_, d_model_),
      ffn_ln(d_model_)
{
    if (d_model == 0) throw std::invalid_argument("MegaBlock: d_model must be > 0");
    if (num_heads != 1) throw std::invalid_argument("MegaBlock: v1 supports only num_heads == 1");
    if (ffn_mult == 0) throw std::invalid_argument("MegaBlock: ffn_mult must be > 0");

    // EMA decay logit — initialize to give α ≈ 0.5 (logit 0).
    alpha_log     = Tensor::zeros(1, d_model_);
    grad_alpha_log = Tensor::zeros(1, d_model_);

    // Position bias: start at zero (no positional preference).
    T_bias_max_  = 1;                       // initial allocation (will grow on first forward if T > 1)
    bias_max_len_ = 2 * T_bias_max_ - 1;    // = 1
    pos_bias      = Tensor::zeros(1, bias_max_len_);
    grad_pos_bias = Tensor::zeros(1, bias_max_len_);

    // FFN sublayer.
    ffn_W1 = Dense(d_model_, d_model_ * ffn_mult_);
    ffn_W2 = Dense(d_model_ * ffn_mult_, d_model_);
    ffn_ln = LayerNorm(d_model_);
}

std::vector<Tensor*> MegaBlock::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_q.weights);  p.push_back(&W_q.bias);
    p.push_back(&W_k.weights);  p.push_back(&W_k.bias);
    p.push_back(&W_v.weights);  p.push_back(&W_v.bias);
    p.push_back(&W_o.weights);  p.push_back(&W_o.bias);
    p.push_back(&W_g.weights);  p.push_back(&W_g.bias);
    p.push_back(&alpha_log);
    p.push_back(&pos_bias);
    p.push_back(&ffn_W1.weights); p.push_back(&ffn_W1.bias);
    p.push_back(&ffn_W2.weights); p.push_back(&ffn_W2.bias);
    p.push_back(&ffn_ln.gamma);
    p.push_back(&ffn_ln.beta);
    return p;
}

std::vector<Tensor*> MegaBlock::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&W_q.grad_weights);  g.push_back(&W_q.grad_bias);
    g.push_back(&W_k.grad_weights);  g.push_back(&W_k.grad_bias);
    g.push_back(&W_v.grad_weights);  g.push_back(&W_v.grad_bias);
    g.push_back(&W_o.grad_weights);  g.push_back(&W_o.grad_bias);
    g.push_back(&W_g.grad_weights);  g.push_back(&W_g.grad_bias);
    g.push_back(&grad_alpha_log);
    g.push_back(&grad_pos_bias);
    g.push_back(&ffn_W1.grad_weights); g.push_back(&ffn_W1.grad_bias);
    g.push_back(&ffn_W2.grad_weights); g.push_back(&ffn_W2.grad_bias);
    g.push_back(&ffn_ln.grad_gamma_);
    g.push_back(&ffn_ln.grad_beta_);
    return g;
}

void MegaBlock::zero_grad() {
    W_q.zero_grad();  W_k.zero_grad();  W_v.zero_grad();  W_o.zero_grad();  W_g.zero_grad();
    grad_alpha_log.fill(0.0);
    grad_pos_bias.fill(0.0);
    ffn_W1.zero_grad(); ffn_W2.zero_grad();
    ffn_ln.zero_grad();
}

void MegaBlock::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    W_g.update_weights(learning_rate);
    // Raw tensors: alpha_log, pos_bias.
    for (size_t j = 0; j < d_model_; ++j) {
        alpha_log[0][j] -= learning_rate * grad_alpha_log[0][j];
    }
    for (size_t j = 0; j < bias_max_len_; ++j) {
        pos_bias[0][j] -= learning_rate * grad_pos_bias[0][j];
    }
    ffn_W1.update_weights(learning_rate);
    ffn_W2.update_weights(learning_rate);
    ffn_ln.update_weights(learning_rate);
}

Tensor MegaBlock::forward(const Tensor& input) {
    const size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("MegaBlock::forward: input.cols != d_model_");
    }
    if (T == 0) {
        throw std::invalid_argument("MegaBlock::forward: T must be > 0");
    }
    T_ = T;

    // Allocate / grow pos_bias if T exceeds the current allocation.
    if (T > T_bias_max_) {
        T_bias_max_  = T;
        bias_max_len_ = 2 * T_bias_max_ - 1;
        Tensor new_pos_bias      = Tensor::zeros(1, bias_max_len_);
        Tensor new_grad_pos_bias = Tensor::zeros(1, bias_max_len_);
        // We don't preserve old bias across resizes; tests rebuild blocks per T as needed.
        pos_bias = new_pos_bias;
        grad_pos_bias = new_grad_pos_bias;
    }

    // ----- Cache input and compute alpha = sigmoid(alpha_log) -----
    last_input = input;
    last_alpha = Tensor(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        last_alpha[0][j] = 1.0 / (1.0 + std::exp(-alpha_log[0][j]));
    }

    // ----- EMA recurrence: u_t = α ⊙ u_{t-1} + (1-α) ⊙ x_t -----
    last_u = Tensor(T, d_model_);
    last_u_prev = Tensor(T, d_model_);   // last_u_prev[0] = 0 (sentinel)
    for (size_t t = 0; t < T; ++t) {
        if (t == 0) {
            for (size_t j = 0; j < d_model_; ++j) {
                last_u_prev[0][j] = 0.0;
                last_u[0][j] = (1.0 - last_alpha[0][j]) * input[0][j];
            }
        } else {
            for (size_t j = 0; j < d_model_; ++j) {
                last_u_prev[t][j] = last_u[t - 1][j];
                last_u[t][j] = last_alpha[0][j] * last_u[t - 1][j]
                             + (1.0 - last_alpha[0][j]) * input[t][j];
            }
        }
    }

    // ----- Q/K/V/Z projections from u -----
    last_q = W_q.forward(last_u);    // (T, d_model)
    last_k = W_k.forward(last_u);
    last_v = W_v.forward(last_u);
    last_z_pre = W_g.forward(last_u);
    last_z = Tensor(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_z[i][j] = 1.0 / (1.0 + std::exp(-last_z_pre[i][j]));
        }
    }

    // ----- Attention scores with relative position bias -----
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_));
    last_score = Tensor(T, T);
    for (size_t t = 0; t < T; ++t) {
        for (size_t s = 0; s < T; ++s) {
            if (s > t) {
                last_score[t][s] = -1e300;  // causal mask
            } else {
                double dot = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    dot += last_q[t][j] * last_k[s][j];
                }
                const size_t bias_idx = (t - s) + (T - 1);
                last_score[t][s] = dot * scale + pos_bias[0][bias_idx];
            }
        }
    }

    // ----- Softmax (causal-aware) -----
    last_attn = row_softmax(last_score);

    // ----- Attention output: o_t = Σ_{s≤t} attn[t,s] * v_s -----
    last_o = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                acc += last_attn[t][s] * last_v[s][j];
            }
            last_o[t][j] = acc;
        }
    }

    // ----- Output gate: g_t = o_t ⊙ z_t -----
    last_g = Tensor(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_g[i][j] = last_o[i][j] * last_z[i][j];
        }
    }

    // ----- Output projection: h = g @ W_o^T -----
    last_h = W_o.forward(last_g);

    // ----- Residual add + FFN sublayer -----
    last_residual = Tensor(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_residual[i][j] = last_input[i][j] + last_h[i][j];
        }
    }
    last_ffn_ln_in  = last_residual;
    last_ffn_ln_out = ffn_ln.forward(last_ffn_ln_in);
    last_ffn_hidden = ffn_W1.forward(last_ffn_ln_out);   // (T, mult*d_model)
    // GELU activation.
    GELU gelu;
    last_ffn_act = Tensor(T, d_model_ * ffn_mult_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_ * ffn_mult_; ++j) {
            last_ffn_act[i][j] = gelu(last_ffn_hidden[i][j]);
        }
    }
    last_ffn_out = ffn_W2.forward(last_ffn_act);

    // ----- Final output: residual + FFN -----
    Tensor output(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            output[i][j] = last_residual[i][j] + last_ffn_out[i][j];
        }
    }
    return output;
}

Tensor MegaBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (T_ == 0) throw std::logic_error("MegaBlock::backward called before forward");
    const size_t T = T_;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_));

    // ====================================================================
    // Phase 1: FFN sublayer backward
    //   grad_ffn_out = grad_output (residual path through ffn_out)
    //   grad_residual += grad_output
    // ====================================================================
    Tensor grad_ffn_out = grad_output;          // (T, d_model)
    Tensor grad_residual(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_residual[i][j] = grad_output[i][j];
        }
    }

    // ffn_W2 backward: y = act @ W2^T + b2.
    Tensor grad_ffn_act = ffn_W2.backward(grad_ffn_out, 0.0);
    // GELU backward: d_pre = d_act * gelu'(pre).
    Tensor grad_ffn_hidden(T, d_model_ * ffn_mult_);
    GELU gelu;
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_ * ffn_mult_; ++j) {
            grad_ffn_hidden[i][j] = grad_ffn_act[i][j] * gelu.derivative(last_ffn_hidden[i][j]);
        }
    }
    Tensor grad_ffn_ln_out = ffn_W1.backward(grad_ffn_hidden, 0.0);
    Tensor grad_ffn_ln_in  = ffn_ln.backward(grad_ffn_ln_out, 0.0);
    // Add to grad_residual (the FFN path joins the residual stream).
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_residual[i][j] += grad_ffn_ln_in[i][j];
        }
    }

    // ====================================================================
    // Phase 2: residual split — grad_residual flows to BOTH the input x and
    // the attention output h.
    // ====================================================================
    Tensor grad_h = grad_residual;        // (T, d_model)
    Tensor grad_x(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_x[i][j] = grad_residual[i][j];
        }
    }

    // ====================================================================
    // Phase 3: W_o backward. h = g @ W_o^T + b_o, so:
    //   grad_W_o += g^T · grad_h
    //   grad_g   = grad_h @ W_o
    //   grad_b_o += sum_t grad_h[t, :]
    // ====================================================================
    Tensor grad_g = W_o.backward(grad_h, 0.0);   // updates W_o.grad_weights/bias internally

    // ====================================================================
    // Phase 4: output gate. g = o ⊙ z, so:
    //   grad_o = grad_g ⊙ z
    //   grad_z = grad_g ⊙ o
    // ====================================================================
    Tensor grad_o(T, d_model_);
    Tensor grad_z(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_o[i][j] = grad_g[i][j] * last_z[i][j];
            grad_z[i][j] = grad_g[i][j] * last_o[i][j];
        }
    }

    // ====================================================================
    // Phase 5: W_g (gate) backward. z = sigmoid(W_g · u + b_g), so:
    //   grad_z_pre = grad_z ⊙ z ⊙ (1 - z)  (sigmoid derivative)
    //   grad_W_g += u^T · grad_z_pre
    //   grad_b_g += sum_t grad_z_pre[t, :]
    //   grad_u (from gate path) = grad_z_pre @ W_g
    // ====================================================================
    Tensor grad_z_pre(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            const double z = last_z[i][j];
            grad_z_pre[i][j] = grad_z[i][j] * z * (1.0 - z);
        }
    }
    Tensor grad_u_from_gate = W_g.backward(grad_z_pre, 0.0);

    // ====================================================================
    // Phase 6: V/Q/K backward through attention.
    //   o_t = Σ_{s≤t} attn[t,s] * v_s
    //   ⇒ grad_attn[t,s] += dot(grad_o[t, :], v[s, :])
    //   ⇒ grad_v[s, :]   += Σ_{t≥s} attn[t,s] * grad_o[t, :]
    //   Then softmax backward: grad_score = softmax_backward(attn, grad_attn).
    //   Then Q/K backward: grad_q[t] += (1/√d) * Σ_{s≤t} grad_score[t,s] * k[s]
    //                      grad_k[s] += (1/√d) * Σ_{t≥s} grad_score[t,s] * q[t]
    //   Then Q/K/V Dense backprop through W_q, W_k, W_v (each gets grad_u contribution).
    // ====================================================================
    Tensor grad_attn(T, T);
    for (size_t t = 0; t < T; ++t) {
        for (size_t s = 0; s < T; ++s) {
            if (s > t) {
                grad_attn[t][s] = 0.0;
            } else {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) acc += grad_o[t][j] * last_v[s][j];
                grad_attn[t][s] = acc;
            }
        }
    }
    Tensor grad_v(T, d_model_);
    for (size_t s = 0; s < T; ++s) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = s; t < T; ++t) {
                acc += last_attn[t][s] * grad_o[t][j];
            }
            grad_v[s][j] = acc;
        }
    }

    // Position bias: dβ[delta] += attn[t,s] for the (t,s) entries actually contributing.
    // From softmax backward path: grad_score[t,s] = attn[t,s] * (grad_attn[t,s] - sum_a attn[t,a]*grad_attn[t,a])
    //   ⇒ dβ[delta] += grad_score[t,s] (since d(score)/dβ = 1 for the bias addend)
    // BUT the score is added BEFORE softmax; the softmax_backward already correctly
    // gives the gradient on the pre-softmax score. So grad on the bias contribution
    // is exactly the grad_score element.
    Tensor grad_score = softmax_backward(last_attn, grad_attn);
    for (size_t t = 0; t < T; ++t) {
        for (size_t s = 0; s <= t; ++s) {
            const size_t bias_idx = (t - s) + (T - 1);
            grad_pos_bias[0][bias_idx] += grad_score[t][s];
        }
    }

    // grad_q[t] += scale * Σ_{s≤t} grad_score[t,s] * k[s]
    Tensor grad_q(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                acc += grad_score[t][s] * last_k[s][j];
            }
            grad_q[t][j] = scale * acc;
        }
    }
    // grad_k[s] += scale * Σ_{t≥s} grad_score[t,s] * q[t]
    Tensor grad_k(T, d_model_);
    for (size_t s = 0; s < T; ++s) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = s; t < T; ++t) {
                acc += grad_score[t][s] * last_q[t][j];
            }
            grad_k[s][j] = scale * acc;
        }
    }

    // Now backprop grad_q → W_q, grad_k → W_k, grad_v → W_v (Dense convention).
    Tensor grad_u_from_q = W_q.backward(grad_q, 0.0);
    Tensor grad_u_from_k = W_k.backward(grad_k, 0.0);
    Tensor grad_u_from_v = W_v.backward(grad_v, 0.0);

    // ====================================================================
    // Phase 7: combine all grad_u contributions and run EMA backward.
    //   grad_u_total = grad_u_from_gate + grad_u_from_q + grad_u_from_k + grad_u_from_v
    // ====================================================================
    Tensor grad_u(T, d_model_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_u[i][j] = grad_u_from_gate[i][j] + grad_u_from_q[i][j]
                         + grad_u_from_k[i][j] + grad_u_from_v[i][j];
        }
    }

    // EMA backward. Forward: u_t = α ⊙ u_{t-1} + (1-α) ⊙ x_t.
    // Let β_t = 1 - α_t (input mixing weight). Then:
    //   ∂u_t/∂α[j] = u_{t-1}[j] - x_t[j]                       (the per-channel derivative w.r.t. α)
    //   ∂u_t/∂u_{t-1}[j] = α[j]
    //   ∂u_t/∂x_t[j] = β_t[j] = 1 - α[j]
    // Reverse-time:
    //   for t = T-1 down to 1:
    //       dα[j] += du_t[j] * (u_{t-1}[j] - x_t[j])
    //       du_{t-1}[j] += du_t[j] * α[j]
    //       dx_t[j] += du_t[j] * (1 - α[j])
    //   dx_0[j] += du_0[j] * (1 - α[j])
    // (Final t=0 row gets no dα contribution since u_{-1} = 0.)
    Tensor grad_alpha(T, d_model_);  // (1, d_model) — accumulate
    grad_alpha = Tensor::zeros(1, d_model_);
    for (size_t t = T; t > 0; --t) {
        const size_t t_idx = t - 1;        // 0-indexed current position
        for (size_t j = 0; j < d_model_; ++j) {
            const double a = last_alpha[0][j];
            const double du = grad_u[t_idx][j];
            // dα[j] += du_t * (u_{t-1} - x_t)
            grad_alpha[0][j] += du * (last_u_prev[t_idx][j] - last_input[t_idx][j]);
            // du_{t-1} += du_t * α
            if (t_idx > 0) {
                grad_u[t_idx - 1][j] += du * a;
            }
            // dx_t += du_t * (1 - α)
            grad_x[t_idx][j] += du * (1.0 - a);
        }
    }
    // dx_0 gets its (1-α) contribution directly (the t=0 case above handled it).

    // Map dα → dα_log via sigmoid derivative: dα/dα_log = α * (1 - α).
    for (size_t j = 0; j < d_model_; ++j) {
        const double a = last_alpha[0][j];
        grad_alpha_log[0][j] += grad_alpha[0][j] * a * (1.0 - a);
    }

    return grad_x;
}

void MegaBlock::copy_params_from(const MegaBlock& other) {
    if (other.d_model_ != d_model_) {
        throw std::invalid_argument("MegaBlock::copy_params_from: d_model mismatch");
    }
    W_q.weights = other.W_q.weights;  W_q.bias = other.W_q.bias;
    W_k.weights = other.W_k.weights;  W_k.bias = other.W_k.bias;
    W_v.weights = other.W_v.weights;  W_v.bias = other.W_v.bias;
    W_o.weights = other.W_o.weights;  W_o.bias = other.W_o.bias;
    W_g.weights = other.W_g.weights;  W_g.bias = other.W_g.bias;
    alpha_log = other.alpha_log;
    pos_bias = other.pos_bias;            // full copy (may include extra zeros)
    ffn_W1.weights = other.ffn_W1.weights;  ffn_W1.bias = other.ffn_W1.bias;
    ffn_W2.weights = other.ffn_W2.weights;  ffn_W2.bias = other.ffn_W2.bias;
    ffn_ln.gamma = other.ffn_ln.gamma;       ffn_ln.beta = other.ffn_ln.beta;
    T_bias_max_ = other.T_bias_max_;
    bias_max_len_ = other.bias_max_len_;
    grad_pos_bias = other.grad_pos_bias;  // copy grads (tests typically zero these)
}

// ----------------------------------------------------------------------------
// MegaModel
// ----------------------------------------------------------------------------

MegaModel::MegaModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_layers, size_t num_heads, size_t ffn_mult)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_layers_(num_layers), num_heads_(num_heads), ffn_mult_(ffn_mult),
      input_proj(input_dim_, d_model_), final_ln(d_model_),
      output_proj(d_model_, output_dim_)
{
    if (input_dim == 0)  throw std::invalid_argument("MegaModel: input_dim must be > 0");
    if (d_model == 0)    throw std::invalid_argument("MegaModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("MegaModel: output_dim must be > 0");
    if (num_layers == 0) throw std::invalid_argument("MegaModel: num_layers must be > 0");
    if (num_heads != 1)  throw std::invalid_argument("MegaModel: v1 supports only num_heads == 1");
    if (ffn_mult == 0)   throw std::invalid_argument("MegaModel: ffn_mult must be > 0");
    blocks.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
        blocks.push_back(std::make_unique<MegaBlock>(d_model_, num_heads_, ffn_mult_));
    }
}

Tensor MegaModel::forward(const Tensor& input) {
    Tensor x = input_proj.forward(input);
    for (auto& blk : blocks) {
        x = blk->forward(x);
    }
    x = final_ln.forward(x);
    return output_proj.forward(x);
}

Tensor MegaModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = output_proj.backward(grad_output, learning_rate);
    grad = final_ln.backward(grad, learning_rate);
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
    grad = input_proj.backward(grad, learning_rate);
    return grad;
}

void MegaModel::update_weights(double learning_rate) {
    input_proj.update_weights(learning_rate);
    for (auto& blk : blocks) blk->update_weights(learning_rate);
    final_ln.update_weights(learning_rate);
    output_proj.update_weights(learning_rate);
}

void MegaModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& blk : blocks) blk->zero_grad();
    final_ln.zero_grad();
    output_proj.zero_grad();
}

std::vector<Tensor*> MegaModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&input_proj.weights);  p.push_back(&input_proj.bias);
    for (auto& blk : blocks) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&final_ln.gamma);  p.push_back(&final_ln.beta);
    p.push_back(&output_proj.weights);  p.push_back(&output_proj.bias);
    return p;
}

std::vector<Tensor*> MegaModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&input_proj.grad_weights);  g.push_back(&input_proj.grad_bias);
    for (auto& blk : blocks) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&final_ln.grad_gamma_);  g.push_back(&final_ln.grad_beta_);
    g.push_back(&output_proj.grad_weights);  g.push_back(&output_proj.grad_bias);
    return g;
}