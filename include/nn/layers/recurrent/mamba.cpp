#include "mamba.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// MambaBlock implementation
// ----------------------------------------------------------------------------
//
// This is a "v1" simplified Mamba layer that captures the core selective
// state-space scan plus the gating/output projection. We omit the
// depthwise 1D conv that the canonical Mamba block has (it's an
// architectural optimization for training stability, not part of the
// selective scan math), and the SiLU-before-SSM is applied to the
// post-projection ssm input (this is faithful to the Mamba paper).
//
// See the header for the full math description.
// ----------------------------------------------------------------------------

// ---------- helper: numerically stable elementwise functions ----------

static inline double mamba_softplus(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log(1.0 + std::exp(x));
}
static inline double mamba_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}
static inline double mamba_silu(double x) {
    return x * mamba_sigmoid(x);
}

// ---------- constructor ----------

MambaBlock::MambaBlock(size_t d_model, size_t d_state, size_t d_inner)
    : in_proj(d_model, 0),         // placeholder, fixed below
      out_proj(0, d_model),        // (d_inner, d_model)
      dt_proj(d_model, 0),         // placeholder
      B_proj(d_model, d_state),
      C_proj(d_model, d_state),
      A_log(0, 0),                 // placeholder, sized in body
      D_skip(1, 0),                // placeholder
      d_model_(d_model), d_state_(d_state)
{
    if (d_inner == 0) d_inner = 2 * d_model;
    if (d_inner == 0) throw std::invalid_argument("MambaBlock: d_model must be > 0");
    d_inner_ = d_inner;

    // Reinitialize projections with correct output dims
    // (Dense constructor takes (in_features, out_features) per the convention
    //  we follow in this codebase — see Linformer for the same correction.)
    in_proj  = Dense(d_model, 2 * d_inner);
    dt_proj  = Dense(d_model, d_inner);
    out_proj = Dense(d_inner, d_model);

    // A_log: unconstrained (d_inner, d_state). A = -exp(A_log) keeps A < 0.
    A_log = Tensor(d_inner, d_state);
    std::mt19937 gen(42);
    std::normal_distribution<> A_dis(0.0, 1.0);  // log-scale init, will become A ~ -exp(N(0,1))
    for (size_t i = 0; i < d_inner; ++i)
        for (size_t d = 0; d < d_state; ++d)
            A_log(i, d) = A_dis(gen) * 0.5;  // small init so |A| ~ e^0.5 ~ 1.6

    // D_skip: init small positive to start (Mamba paper uses 1.0)
    D_skip = Tensor(1, d_inner);
    for (size_t i = 0; i < d_inner; ++i) D_skip(0, i) = 1.0;

    // Initialize hidden grad buffers to correct shape so that
    // `gradients()` returns well-shaped tensors even before backward() is called.
    grad_A_log_  = Tensor(d_inner, d_state);
    grad_D_skip_ = Tensor(1, d_inner);

    // Biases: zero everywhere (default of Dense is small random — we explicitly zero them)
    in_proj.bias.fill(0.0);
    dt_proj.bias.fill(0.0);
    B_proj.bias.fill(0.0);
    C_proj.bias.fill(0.0);
    out_proj.bias.fill(0.0);

    // B_proj and C_proj use default Dense init (small xavier) — fine for
    // input-dependent parameters; we don't need special HIPPO-style init
    // because B and C are not state matrices in Mamba.
}

// ---------- forward ----------

Tensor MambaBlock::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("MambaBlock: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("MambaBlock: input must have at least one token");
    }
    last_input_ = input.clone();

    // Step 1: in_proj — (T, d_model) -> (T, 2*d_inner)
    last_p_ = in_proj.forward(input);
    // Step 2: split into ssm path and gate
    Tensor x_pre(T, d_inner_);
    Tensor gate(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            x_pre(t, i) = last_p_(t, i);
            gate(t, i)  = last_p_(t, d_inner_ + i);
        }
    }
    // silu on ssm path input
    last_ssm_in_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            last_ssm_in_(t, i) = mamba_silu(x_pre(t, i));
    last_gate_ = gate;  // raw, before silu — silu applied in the gating step below

    // Step 3: input-dependent SSM parameters
    last_Delta_pre_ = dt_proj.forward(input);    // (T, d_inner)
    last_Delta_     = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            last_Delta_(t, i) = mamba_softplus(last_Delta_pre_(t, i));

    last_B_t_ = B_proj.forward(input);           // (T, d_state)
    last_C_t_ = C_proj.forward(input);           // (T, d_state)

    // Step 4: Selective state-space recurrence.
    // Pre-compute A (negative) once: A[i][d] = -exp(A_log[i][d]).
    Tensor A(d_inner_, d_state_);
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t d = 0; d < d_state_; ++d)
            A(i, d) = -std::exp(A_log(i, d));

    // Tensor is 2D; we store (T, d_inner, d_state) tensors as (T*d_inner, d_state)
    // with the convention: index (t, i, d) lives at row = t * d_inner + i, col = d.
    size_t TI = T * d_inner_;
    last_A_bar_ = Tensor(TI, d_state_);
    last_B_bar_ = Tensor(TI, d_state_);
    last_h_     = Tensor((T + 1) * d_inner_, d_state_);
    // h_0 = 0
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t d = 0; d < d_state_; ++d)
            last_h_(0 * d_inner_ + i, d) = 0.0;

    last_y_ = Tensor(T, d_inner_);

    for (size_t t = 0; t < T; ++t) {
        // Compute Ā_t, B̄_t, h_t, y_t
        for (size_t i = 0; i < d_inner_; ++i) {
            double dT = last_Delta_(t, i);
            size_t row = t * d_inner_ + i;
            for (size_t d = 0; d < d_state_; ++d) {
                double a = A(i, d);
                last_A_bar_(row, d) = std::exp(dT * a);
                last_B_bar_(row, d) = dT * last_B_t_(t, d);
            }
        }
        // h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x_tilde_t
        // Storage: h_{t-1}[i][d] is at row (t-1)*d_inner + i, col d
        //          h_t[i][d]    is at row t*d_inner + i, col d
        for (size_t i = 0; i < d_inner_; ++i) {
            size_t row = t * d_inner_ + i;
            for (size_t d = 0; d < d_state_; ++d) {
                double h_prev = last_h_(row, d);  // t==0 reads h_0 which is 0
                double x_til  = last_ssm_in_(t, i);
                last_h_(row + d_inner_, d) = last_A_bar_(row, d) * h_prev
                                           + last_B_bar_(row, d) * x_til;
            }
        }
        // y_t = C_t · h_t + D_skip ⊙ x_tilde_t
        for (size_t i = 0; i < d_inner_; ++i) {
            size_t row = t * d_inner_ + i;
            double acc = 0.0;
            for (size_t d = 0; d < d_state_; ++d) {
                acc += last_C_t_(t, d) * last_h_(row + d_inner_, d);
            }
            last_y_(t, i) = acc + D_skip(0, i) * last_ssm_in_(t, i);
        }
    }

    // Step 5: silu(gate) ⊙ y, then out_proj
    Tensor gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double g_act = mamba_silu(last_gate_(t, i));
            gated(t, i) = g_act * last_y_(t, i);
        }
    }
    last_gated_ = gated;
    Tensor output = out_proj.forward(gated);
    return output;
}

// ---------- backward ----------

Tensor MambaBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = last_input_.rows;
    if (grad_output.rows != T || grad_output.cols != d_model_) {
        throw std::invalid_argument("MambaBlock: grad_output shape mismatch");
    }

    // We'll accumulate parameter gradients on Dense::grad_weights / grad_bias
    // for the 5 projections, and on A_log / D_skip directly.

    // Zero out param grads first (callers usually call zero_grad, but be safe)
    in_proj.zero_grad();
    out_proj.zero_grad();
    dt_proj.zero_grad();
    B_proj.zero_grad();
    C_proj.zero_grad();

    // Gradient w.r.t. gated (input to out_proj)
    // Forward: out_t = gated_t @ W_out^T + b_out  (Dense convention: y = x W^T + b)
    //        W_out is (d_model, d_inner)
    //        dL/dgated[t][i] = sum_j dL/dout[t][j] * W_out[j][i]
    Tensor grad_gated(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                acc += grad_output(t, j) * out_proj.weights(j, i);
            }
            grad_gated(t, i) = acc;
        }
    }

    // dL/dW_out[j][i] = sum_t dL/dout[t][j] * gated[t][i]
    for (size_t j = 0; j < d_model_; ++j)
        for (size_t i = 0; i < d_inner_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < T; ++t)
                acc += grad_output(t, j) * last_gated_(t, i);
            out_proj.grad_weights(j, i) += acc;
        }
    // dL/db_out[j] = sum_t dL/dout[t][j]
    for (size_t j = 0; j < d_model_; ++j) {
        double acc = 0.0;
        for (size_t t = 0; t < T; ++t) acc += grad_output(t, j);
        out_proj.grad_bias(0, j) += acc;
    }

    // Now split: gated = silu(gate) * y
    //   dL/dgate[t][i] = dL/dgated[t][i] * silu'(gate[t][i]) * y[t][i]
    //   dL/dy[t][i]    = dL/dgated[t][i] * silu(gate[t][i])
    Tensor grad_gate(T, d_inner_);
    Tensor grad_y(T, d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double g = last_gate_(t, i);
            double silu_g = mamba_silu(g);
            // silu'(x) = sigmoid(x) + x * sigmoid'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            double sig = mamba_sigmoid(g);
            double silu_prime = sig * (1.0 + g * (1.0 - sig));
            grad_gate(t, i) = grad_gated(t, i) * silu_prime * last_y_(t, i);
            grad_y(t, i)    = grad_gated(t, i) * silu_g;
        }
    }

    // Split grad_y into two parts:
    //   y_t[i] = sum_d C_t[d] * h_t[i][d] + D_skip[i] * ssm_in_t[i]
    //
    //   dL/dh_t[i][d] (from y) = grad_y[t][i] * C_t[d]
    //   dL/dC_t[d]             = sum_i grad_y[t][i] * h_t[i][d]   (per token)
    //   dL/dD_skip[i]          = sum_t grad_y[t][i] * ssm_in_t[i]
    //   dL/dssm_in_t[i]       += grad_y[t][i] * D_skip[i]   (then merge with recurrence grad)
    //
    //   Then, gradient from h_t (recurrence): for t = T-1, 0:
    //     dL/dh_t[i][d] = grad_y[t][i] * C_t[d]   (T-1: no recurrence backward)
    //     for t < T-1:  dL/dh_t[i][d] += dL/dh_{t+1}[i][d] * Ā_{t+1}[i][d]
    // Gradient w.r.t. h_t — same flattening as last_h_: (T, d_inner, d_state)
    // becomes (T * d_inner, d_state). Index (t, i, d) at row t*d_inner + i.
    Tensor g_h(T * d_inner_, d_state_);
    g_h.fill(0.0);

    // First, the direct y_t -> C_t and y_t -> D_skip contributions.
    // State tensor layout: last_h_(T+1)*d_inner rows, d_state cols. h_t[i][d] at
    //   row t*d_inner + i, col d. So h_{t+1} is at row (t+1)*d_inner + i, col d.
    Tensor grad_C_t(T, d_state_);
    grad_C_t.fill(0.0);
    Tensor grad_ssm_in(T, d_inner_);
    grad_ssm_in.fill(0.0);
    Tensor grad_D_skip(1, d_inner_);
    grad_D_skip.fill(0.0);

    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            size_t h_row = t * d_inner_ + i;  // h_t row
            for (size_t d = 0; d < d_state_; ++d) {
                g_h(h_row, d) = grad_y(t, i) * last_C_t_(t, d);
            }
            // D_skip path
            grad_ssm_in(t, i) = grad_y(t, i) * D_skip(0, i);
            grad_D_skip(0, i) += grad_y(t, i) * last_ssm_in_(t, i);
        }
        // dL/dC_t (per token; rolled up into C_proj backward)
        for (size_t d = 0; d < d_state_; ++d) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                size_t row = (t + 1) * d_inner_ + i;  // h_t is stored at index t+1 in our layout
                acc += grad_y(t, i) * last_h_(row, d);
            }
            grad_C_t(t, d) = acc;
        }
    }

    // Now propagate g_h backward through the recurrence.
    // For t from T-1 down to 0, the contribution to dL/dh_{t-1} from h_t
    // is g_h[t][i][d] * Ā_t[i][d]  (since h_t = Ā_t ⊙ h_{t-1} + ...)
    Tensor g_A_bar(T * d_inner_, d_state_);
    Tensor g_B_bar(T * d_inner_, d_state_);
    Tensor g_x_tilde(T, d_inner_);   // contribution to grad_ssm_in from recurrence
    g_A_bar.fill(0.0);
    g_B_bar.fill(0.0);
    g_x_tilde.fill(0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        size_t row = t * d_inner_;        // Ā_t, B̄_t row (and h_{t-1} row when t>0)
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t d = 0; d < d_state_; ++d) {
                double gh = g_h(row + i, d);
                double h_prev = last_h_(row + i, d);
                double x_til  = last_ssm_in_(t, i);
                // dL/dĀ_t[i][d] += gh * h_prev
                g_A_bar(row + i, d) += gh * h_prev;
                // dL/dB̄_t[i][d] += gh * x_tilde
                g_B_bar(row + i, d) += gh * x_til;
                // dL/dx_tilde_t[i] += gh * B̄_t[i][d]
                g_x_tilde(t, i) += gh * last_B_bar_(row + i, d);
                // dL/dh_{t-1}[i][d] += gh * Ā_t[i][d]   (for t > 0)
                if (t > 0) {
                    g_h(row + i - d_inner_, d) += gh * last_A_bar_(row + i, d);
                }
            }
        }
    }

    // Now, gradients w.r.t. A, B, Δ from the per-step accumulators.
    //   A[i][d] = -exp(A_log[i][d])   (A is reused across tokens)
    //   dL/dA[i][d] = sum_t g_Ā_t[i][d] * Ā_t[i][d] * Δ_t[i]
    //   dL/dA_log[i][d] = dL/dA[i][d] * dA/dA_log = dL/dA[i][d] * A[i][d]
    //
    //   B̄_t[i][d] = Δ_t[i] * B_t[d]
    //   dL/dΔ_t[i] (from B̄) = sum_d g_B̄_t[i][d] * B_t[d]
    //   dL/dB_t[d]            = sum_i g_B̄_t[i][d] * Δ_t[i]
    //
    //   Ā_t[i][d] = exp(Δ_t[i] * A[i][d])
    //   dL/dΔ_t[i] (from Ā) = sum_d g_Ā_t[i][d] * Ā_t[i][d] * A[i][d]
    Tensor g_A_log(d_inner_, d_state_);
    g_A_log.fill(0.0);
    // We also need g_Δ_t (T, d_inner) and g_B_t (T, d_state) for projection backward.
    Tensor g_Delta(T, d_inner_);
    Tensor g_B_t(T, d_state_);
    g_Delta.fill(0.0);
    g_B_t.fill(0.0);

    // Pre-compute A from A_log for the chain rule
    Tensor A(d_inner_, d_state_);
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t d = 0; d < d_state_; ++d)
            A(i, d) = -std::exp(A_log(i, d));

    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double dT = last_Delta_(t, i);
            size_t row = t * d_inner_ + i;  // row in g_A_bar, g_B_bar, last_A_bar_, last_B_bar_
            for (size_t d = 0; d < d_state_; ++d) {
                double a = A(i, d);
                double a_bar = last_A_bar_(row, d);
                // g_Δ from Ā path: g_Ā_t * Ā_t * A
                g_Delta(t, i) += g_A_bar(row, d) * a_bar * a;
                // g_A from Ā path: g_Ā_t * Ā_t * Δ
                g_A_log(i, d) += g_A_bar(row, d) * a_bar * dT * a;  // *A: dA/dA_log
                // g_Δ from B̄ path: g_B̄_t * B_t
                g_Delta(t, i) += g_B_bar(row, d) * last_B_t_(t, d);
                // g_B_t from B̄ path: g_B̄_t * Δ
                g_B_t(t, d) += g_B_bar(row, d) * dT;
            }
        }
    }

    // Now we have g_Delta, g_B_t, g_C_t (= grad_C_t). We need to chain
    // through softplus for dt_proj:
    //   Δ_t = softplus(dt_pre_t[i])  =>  dL/ddt_pre_t[i] = g_Delta[t][i] * sigmoid(dt_pre_t[i])
    Tensor g_dt_pre(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_inner_; ++i)
            g_dt_pre(t, i) = g_Delta(t, i) * mamba_sigmoid(last_Delta_pre_(t, i));

    // Now backprop through the 5 projections (Dense layers).
    // Each projection is forward: y = x @ W^T + b  (Dense convention).
    // For a projection that produces (T, out_dim) from (T, d_model) input,
    // backward: g_W[i][k] = sum_t g_y[t][i] * x[t][k]  and g_b[i] = sum_t g_y[t][i].
    // g_x[t][k] = sum_i g_y[t][i] * W[i][k].

    // dt_proj: (T, d_inner) from (T, d_model)
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = g_dt_pre;
        for (size_t i = 0; i < d_inner_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                dt_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            dt_proj.grad_bias(0, i) += b_acc;
        }
    }

    // B_proj: (T, d_state) from (T, d_model)
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = g_B_t;
        for (size_t i = 0; i < d_state_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                B_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            B_proj.grad_bias(0, i) += b_acc;
        }
    }

    // C_proj: (T, d_state) from (T, d_model)
    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_C_t;
        for (size_t i = 0; i < d_state_; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                C_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            C_proj.grad_bias(0, i) += b_acc;
        }
    }

    // in_proj: (T, 2*d_inner) from (T, d_model). Split into ssm path grad and gate grad.
    //   grad_ssm_in = g_x_tilde (from recurrence) + grad_ssm_in (from D_skip path) + dL/dx_pre
    //   But x_pre = silu_inv(x_tilde)... no, we forward x_tilde = silu(x_pre).
    //   So dL/dx_pre[t][i] = dL/dx_tilde[t][i] * silu'(x_pre[t][i])
    //   And dL/dgate[t][i] is grad_gate (already computed).
    Tensor grad_ssm_pre(T, d_inner_);  // dL/dx_pre
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            double total_ssm = g_x_tilde(t, i) + grad_ssm_in(t, i);
            // dL/dx_pre: chain through silu(x_pre)
            double x_pre_val = last_p_(t, i);  // x_pre = p[:, :d_inner]
            double sig = mamba_sigmoid(x_pre_val);
            double silu_prime = sig * (1.0 + x_pre_val * (1.0 - sig));
            grad_ssm_pre(t, i) = total_ssm * silu_prime;
        }
    }

    // Stack grad_p = [grad_ssm_pre (d_inner), grad_gate (d_inner)] (T, 2*d_inner)
    Tensor grad_p(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_inner_; ++i) {
            grad_p(t, i)               = grad_ssm_pre(t, i);
            grad_p(t, d_inner_ + i)    = grad_gate(t, i);
        }
    }

    {
        const Tensor& x = last_input_;
        const Tensor& g_y = grad_p;
        size_t out_dim = 2 * d_inner_;
        for (size_t i = 0; i < out_dim; ++i) {
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t t = 0; t < T; ++t) acc += g_y(t, i) * x(t, k);
                in_proj.grad_weights(i, k) += acc;
            }
            double b_acc = 0.0;
            for (size_t t = 0; t < T; ++t) b_acc += g_y(t, i);
            in_proj.grad_bias(0, i) += b_acc;
        }
    }

    // Input gradient: g_input[t][k] = sum over all projections of g_x[t][k]
    Tensor grad_input(T, d_model_);
    grad_input.fill(0.0);

    // From in_proj: g_x[t][k] = sum_i g_p[t][i] * W_in[i][k]
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < 2 * d_inner_; ++i) {
                acc += grad_p(t, i) * in_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From dt_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_inner_; ++i) {
                acc += g_dt_pre(t, i) * dt_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From B_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_state_; ++i) {
                acc += g_B_t(t, i) * B_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }
    // From C_proj
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < d_state_; ++i) {
                acc += grad_C_t(t, i) * C_proj.weights(i, k);
            }
            grad_input(t, k) += acc;
        }
    }

    // Update A_log and D_skip "gradient" tensors (we keep them as members for
    // a clean test API; we apply the gradient inside update_weights).
    // g_A_log and grad_D_skip are NOT in Dense's API; we use a small struct.

    // Stash non-Dense grads as Dense-shaped members in a "params/grads"
    // list?  We have to be careful: the API contract is `parameters()` and
    // `gradients()` returning pointers. A_log and D_skip aren't Dense, so we
    // just need to make them visible — store them as members and update in
    // update_weights.

    // For simplicity, store the per-step grads in last_Delta_ and last_h_ buffers
    // (we no longer need them after this point — they're already used). We'll
    // move A_log/D_skip gradient into member tensors via a small trick: keep
    // === Store g_A_log and grad_D_skip into the member gradient buffers ===
    // These are separate from Dense's grad_weights/grad_bias; update_weights
    // applies them as a separate step. Sizes are pre-set by zero_grad().
    if (grad_A_log_.rows != d_inner_ || grad_A_log_.cols != d_state_) {
        grad_A_log_ = Tensor(d_inner_, d_state_);
        grad_D_skip_ = Tensor(1, d_inner_);
        grad_A_log_.fill(0.0);
        grad_D_skip_.fill(0.0);
    }
    for (size_t i = 0; i < d_inner_; ++i)
        for (size_t d = 0; d < d_state_; ++d)
            grad_A_log_(i, d) = g_A_log(i, d);
    for (size_t i = 0; i < d_inner_; ++i)
        grad_D_skip_(0, i) = grad_D_skip(0, i);

    return grad_input;
}

// ---------- update_weights ----------

void MambaBlock::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    dt_proj.update_weights(learning_rate);
    B_proj.update_weights(learning_rate);
    C_proj.update_weights(learning_rate);
    if (grad_A_log_.rows == d_inner_ && grad_A_log_.cols == d_state_) {
        for (size_t i = 0; i < d_inner_; ++i)
            for (size_t d = 0; d < d_state_; ++d)
                A_log(i, d) -= learning_rate * grad_A_log_(i, d);
    }
    if (grad_D_skip_.cols == d_inner_) {
        for (size_t i = 0; i < d_inner_; ++i)
            D_skip(0, i) -= learning_rate * grad_D_skip_(0, i);
    }
}

// ---------- zero_grad ----------

void MambaBlock::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    dt_proj.zero_grad();
    B_proj.zero_grad();
    C_proj.zero_grad();
    // Reset hidden grad buffers to correct shape (filled with zero).
    grad_A_log_ = Tensor(d_inner_, d_state_);
    grad_D_skip_ = Tensor(1, d_inner_);
}

// ---------- parameters / gradients ----------

std::vector<Tensor*> MambaBlock::parameters() {
    std::vector<Tensor*> p;
    auto append_dense = [&](Dense& d) {
        p.push_back(&d.weights);
        p.push_back(&d.bias);
    };
    append_dense(in_proj);
    append_dense(out_proj);
    append_dense(dt_proj);
    append_dense(B_proj);
    append_dense(C_proj);
    p.push_back(&A_log);
    p.push_back(&D_skip);
    return p;
}

std::vector<Tensor*> MambaBlock::gradients() {
    std::vector<Tensor*> g;
    auto append_dense = [&](Dense& d) {
        g.push_back(&d.grad_weights);
        g.push_back(&d.grad_bias);
    };
    append_dense(in_proj);
    append_dense(out_proj);
    append_dense(dt_proj);
    append_dense(B_proj);
    append_dense(C_proj);
    g.push_back(&grad_A_log_);
    g.push_back(&grad_D_skip_);
    return g;
}
