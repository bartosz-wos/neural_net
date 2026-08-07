#include "deltanet.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// DeltaNet implementation
// ============================================================================
//
// Math reference (per head h, per time t):
//
//   Let k_t = cache_k_scaled_[t][h * head_dim_ + d] (after β/|k| scaling).
//   Let v_t = cache_v_[t][h * head_dim_ + d].
//   Let q_t = cache_q_[t][h * head_dim_ + d].
//   S_t[h] is stored as cache_S_[t][h] (head_dim × head_dim in row-major flat).
//
//   Sk = S_{t-1} · k_t            (vector of length head_dim)
//   r_t = v_t - Sk                (vector of length head_dim)
//   k_dot_Sk = k_t · Sk            (scalar)
//   α_t = 1 / (1 + k_dot_Sk)       (scalar)
//   S_t[h] = S_{t-1}[h] + α_t · outer(k_t, r_t)
//   o_t[h] = S_t[h] · q_t
//
// ---------- Backward derivation ----------
//
// 1) Output side: o_t[h] = S_t[h] · q_t[h] → gS_t[h] += outer(g_o_t[h], q_t[h])
//    grad_q_t[h, i] = sum_j gS_t[h, i, j] · g_o_t[h, j]  (i.e. S_t[h] · g_o_t[h])
//
//    Note: gS_t includes the downstream carrier from S_{t+1} as well.
//
// 2) Update side: S_t[h] = S_{t-1}[h] + α_t · outer(k_t, r_t)
//    where r_t = v_t - S_{t-1}·k_t.
//
//    (a) Direct contribution to gS_{t-1} from the "S_{t-1} appears linearly" path:
//        gS_{t-1}[h] += gS_t[h]
//
//    (b) Indirect path via r_t = v_t - S_{t-1}·k_t:
//        r_t only depends on S_{t-1} through S_{t-1}·k_t. So
//        grad contribution to S_{t-1} from r_t = - outer(g_r_t, k_t)
//        where g_r_t = contribution of r_t to S_t[h] ∂S_t[h]/∂r_t = α_t · k_t
//        (as a column vector — but we treat it as row-dim below).
//
//        g_r_t[h, j] = sum_i gS_t[h, i, j] · α_t · k_t[h, i]
//        g_v_t[h, j] = g_r_t[h, j]   (since r_t = v_t - S_{t-1}·k_t, ∂r_t/∂v_t = I)
//
//        g_k_t[h, i] += -sum_j g_r_t[h, j] · S_{t-1}[h, i, j]
//
//    (c) Indirect path via α_t = 1 / (1 + k_t · S_{t-1} · k_t):
//        g_α_t = sum_{i,j} gS_t[h, i, j] · k_t[h, i] · r_t[h, j]
//        (this is the inner product of gS_t with the per-head k * r^T)
//
//        g_α_t · ∂α_t/∂S_{t-1}[h, i, j] = -g_α_t · α_t² · k_t[h, i] · k_t[h, j]
//        → gS_{t-1}[h, i, j] += -g_α_t · α_t² · k_t[h, i] · k_t[h, j]
//
//        g_α_t · ∂α_t/∂k_t[h, i] = -2 · g_α_t · α_t² · (S_{t-1}·k_t)[h, i]
//                                 = -2 · g_α_t · α_t² · Sk[h, i]
//        → g_k_t[h, i] += -2 · g_α_t · α_t² · Sk[h, i]
//
//    (d) Direct contribution from the outer-product (k_t · r_t^T) to g_k_t:
//        ∂S_t[h, i, j] / ∂k_t[h, i] = α_t · r_t[h, j]
//        → g_k_t[h, i] += α_t · sum_j gS_t[h, i, j] · r_t[h, j]
//
// 3) Undo β/|k| scaling: k_scaled = (β / |k_raw|) · k_raw
//    Jacobian:
//      ∂k_scaled[d] / ∂k_raw[d2] = (β / |k_raw|) · (δ_{d,d2} - k_raw[d]·k_raw[d2] / |k_raw|²)
//    Inverse (for backprop):
//      g_k_raw[d] = (β / |k_raw|) · g_k_scaled[d] - k_scaled[d] · dot(g_k_scaled, k_raw) / (β · |k_raw|)
//    And ∂k_scaled[d] / ∂β = k_raw[d] / |k_raw| = k_scaled[d] / β
//      g_β = sum_d g_k_scaled[d] · k_scaled[d] / β
//
// 4) Undo sigmoid: β = sigmoid(β_pre) → dβ/dβ_pre = β · (1 - β)
//    g_β_pre = g_β · β · (1 - β)
//
// ============================================================================

// ---------- helpers ----------

static double dn_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ---------- constructor ----------

DeltaNet::DeltaNet(size_t d_model, size_t n_heads, size_t head_dim)
    : W_Q_(d_model, 0), W_K_(d_model, 0), W_V_(d_model, 0),
      W_O_(0, 0), W_beta_(d_model, 0),
      d_model_(d_model),
      n_heads_(n_heads),
      head_dim_(0),  // set below
      d_inner_(0)  // set below
{
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument("DeltaNet: d_model and n_heads must be > 0");
    }
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("DeltaNet: d_model must divide evenly by n_heads");
    }
    head_dim_ = (head_dim == 0) ? d_model_ / n_heads_ : head_dim;
    if (head_dim != 0 && head_dim != head_dim_) {
        throw std::invalid_argument("DeltaNet: head_dim must equal d_model/n_heads (default)");
    }

    d_inner_ = d_model_;  // default: d_inner = d_model

    // Reinitialize projections with correct output dims (Dense default-ctor
    // sizes to 0/0, can't change size after construction).
    W_Q_   = Dense(d_model_, d_inner_);
    W_K_   = Dense(d_model_, d_inner_);
    W_V_   = Dense(d_model_, d_inner_);
    W_O_   = Dense(d_inner_, d_model_);
    W_beta_ = Dense(d_model_, n_heads_);

    // Smaller init for all weights — the recurrence is sensitive to scale.
    // We use uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)] for stability.
    W_Q_.init_weights("uniform");
    W_K_.init_weights("uniform");
    W_V_.init_weights("uniform");
    W_O_.init_weights("uniform");
    W_beta_.init_weights("uniform");

    // Biases always zero
    W_Q_.bias.fill(0.0);
    W_K_.bias.fill(0.0);
    W_V_.bias.fill(0.0);
    W_O_.bias.fill(0.0);
    W_beta_.bias.fill(0.0);
}

// ---------- forward ----------

Tensor DeltaNet::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("DeltaNet::forward: input.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    cache_x_ = input.clone();

    // Project
    cache_q_ = W_Q_.forward(input);           // (T, d_inner)
    cache_k_ = W_K_.forward(input);           // (T, d_inner)
    cache_v_ = W_V_.forward(input);           // (T, d_inner)
    Tensor beta_pre = W_beta_.forward(input); // (T, n_heads)

    // Resize cache tensors to match T (in case T changes between calls)
    cache_beta_pre_ = Tensor(T, n_heads_);
    cache_beta_ = Tensor(T, n_heads_);
    cache_alpha_ = Tensor(T, n_heads_);
    cache_k_scaled_ = Tensor(T, d_inner_);

    // Sigmoid for beta
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            cache_beta_pre_[t][h] = beta_pre[t][h];
            cache_beta_[t][h] = dn_sigmoid(beta_pre[t][h]);
        }
    }

    // Per-head k-magnitude normalization: k_t[h] *= (β_t[h] / |k_t[h]|)
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double k_norm_sq = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                double k = cache_k_[t][h * head_dim_ + d];
                k_norm_sq += k * k;
            }
            double k_norm = std::sqrt(k_norm_sq + 1e-12);
            double scale = cache_beta_[t][h] / k_norm;
            for (size_t d = 0; d < head_dim_; ++d) {
                cache_k_scaled_[t][h * head_dim_ + d] =
                    cache_k_[t][h * head_dim_ + d] * scale;
            }
        }
    }

    // Per-head delta-rule recurrence
    cache_S_.clear();
    cache_S_.resize(T);

    // State tensor: (n_heads, head_dim * head_dim) — flat row-major per head.
    Tensor current_state(n_heads_, head_dim_ * head_dim_);  // zero-initialized

    Tensor output_concat(T, d_inner_);

    for (size_t t = 0; t < T; ++t) {
        cache_S_[t] = current_state.clone();

        for (size_t h = 0; h < n_heads_; ++h) {
            // Sk = S_{t-1} · k_t  (vector of length head_dim)
            double Sk[64];  // head_dim_ <= 64
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += current_state[h][i * head_dim_ + j] *
                           cache_k_scaled_[t][h * head_dim_ + j];
                }
                Sk[i] = sum;
            }

            // k_dot_Sk = k_t · Sk
            double k_dot_Sk = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                k_dot_Sk += cache_k_scaled_[t][h * head_dim_ + d] * Sk[d];
            }
            double alpha = 1.0 / (1.0 + k_dot_Sk);
            cache_alpha_[t][h] = alpha;

            // r_t = v_t - Sk
            double r_t[64];
            for (size_t d = 0; d < head_dim_; ++d) {
                r_t[d] = cache_v_[t][h * head_dim_ + d] - Sk[d];
            }

            // Update S_t[h] = S_{t-1}[h] + α · outer(k_t, r_t)
            for (size_t i = 0; i < head_dim_; ++i) {
                double k = cache_k_scaled_[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    current_state[h][i * head_dim_ + j] += alpha * k * r_t[j];
                }
            }

            // Output o_t[h] = S_t[h] · q_t[h]
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += current_state[h][i * head_dim_ + j] *
                           cache_q_[t][h * head_dim_ + j];
                }
                output_concat[t][h * head_dim_ + i] = sum;
            }
        }
    }

    cache_concat_o_ = output_concat.clone();

    // Output projection
    Tensor out = W_O_.forward(cache_concat_o_);  // (T, d_model)
    return out;
}

// ---------- backward ----------

Tensor DeltaNet::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("DeltaNet::backward: grad_output.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    // Step 1: backward through W_O. This returns grad_concat_o and accumulates
    // gradients on W_O_'s weights and bias.
    Tensor grad_concat_o = W_O_.backward(grad_output, 0.0);  // (T, d_inner)

    // Step 2: gradient buffers for the recurrence
    grad_q_ = Tensor(T, d_inner_);
    grad_k_ = Tensor(T, d_inner_);
    grad_v_ = Tensor(T, d_inner_);
    grad_k_scaled_ = Tensor(T, d_inner_);

    // Step 3: backward recurrence through the per-head state
    //
    // Working layout (all per-head, flat (head_dim * head_dim)):
    //   gS_t       = dL/dS_t (the "downstream" grad of S_t coming back from the future)
    //   gS_prev    = dL/dS_{t-1} (the accumulator we propagate backward)
    //
    // At t = T-1: gS_t = 0 (no future). We add the output-side contribution.
    // At t < T-1: gS_t starts with the propagated gS_prev from t+1, then add output-side.

    Tensor gS_prev(n_heads_, head_dim_ * head_dim_);  // zero-initialized

    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 0; --t_signed) {
        size_t t = static_cast<size_t>(t_signed);

        // gS_t = gS_prev (the future-side carrier)
        Tensor gS_t = gS_prev.clone();

        // Add output-side contribution: gS_t[h, i, j] += g_o_t[h, i] · q_t[h, j]
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double go = grad_concat_o[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    gS_t[h][i * head_dim_ + j] += go * cache_q_[t][h * head_dim_ + j];
                }
            }
        }

        // grad_q_t[h, i] = sum_j S_t[h, j, i] · g_o_t[h, j]
        //   (because o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j], so
        //    dL/dq_t[h, i] = sum_j (dL/do_t[h, j]) · S_t[h, j, i] = sum_j g_o_t[h, j] · S_t[h, j, i])
        // We use the actual S_t value (the state AFTER update at time t).
        // cache_S_[t] stores S_{t-1} (state BEFORE update). We need S_t here.
        // We do a forward pass over the recurrence to compute S_t from cache_S_[t] (using cached k/v/alpha).
        // The recurrence is:
        //   Sk = S_{t-1} · k_t
        //   r  = v_t - Sk
        //   S_t = S_{t-1} + alpha · outer(k_t, r)
        // We compute S_t inline using current_state_via_recompute.
        Tensor current_state_q(cache_S_[t].rows, cache_S_[t].cols);
        current_state_q = cache_S_[t].clone();  // start from S_{t-1} = cache_S_[t]
        for (size_t h = 0; h < n_heads_; ++h) {
            // Compute alpha, k · S_{t-1} · k
            double k_norm_sq = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                double k = cache_k_scaled_[t][h * head_dim_ + d];
                k_norm_sq += k * k;
            }
            // Sk = S_{t-1}[h] · k_t[h]
            double Sk[64];
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += cache_S_[t][h][i * head_dim_ + j] *
                           cache_k_scaled_[t][h * head_dim_ + j];
                }
                Sk[i] = sum;
            }
            double k_dot_Sk = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                k_dot_Sk += cache_k_scaled_[t][h * head_dim_ + d] * Sk[d];
            }
            double alpha = 1.0 / (1.0 + k_dot_Sk);
            // r_t = v_t - Sk
            double r_t[64];
            for (size_t d = 0; d < head_dim_; ++d) {
                r_t[d] = cache_v_[t][h * head_dim_ + d] - Sk[d];
            }
            // S_t[h, i, j] = S_{t-1}[h, i, j] + alpha · k_t[h, i] · r_t[h, j]
            for (size_t i = 0; i < head_dim_; ++i) {
                double k = cache_k_scaled_[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    current_state_q[h][i * head_dim_ + j] += alpha * k * r_t[j];
                }
            }
        }
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    // S_t[h, j, i] is at current_state_q[h][j*head_dim + i]
                    sum += current_state_q[h][j * head_dim_ + i] * grad_concat_o[t][h * head_dim_ + j];
                }
                grad_q_[t][h * head_dim_ + i] = sum;
            }
        }

        // Re-compute Sk, r_t, alpha for this head (we cached S_{t-1} in cache_S_[t])
        Tensor gS_prev_new(n_heads_, head_dim_ * head_dim_);

        double Sk[64], r_t[64];
        double g_r[64];

        for (size_t h = 0; h < n_heads_; ++h) {
            // Re-compute Sk = S_{t-1} · k_t
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += cache_S_[t][h][i * head_dim_ + j] *
                           cache_k_scaled_[t][h * head_dim_ + j];
                }
                Sk[i] = sum;
            }

            // r_t = v_t - Sk
            for (size_t d = 0; d < head_dim_; ++d) {
                r_t[d] = cache_v_[t][h * head_dim_ + d] - Sk[d];
            }

            double alpha = cache_alpha_[t][h];

            // g_r[h, j] = sum_i gS_t[h, i, j] · α · k_t[h, i]
            //            (∂S_t[h, i, j] / ∂r_t[h, j] = α · k_t[h, i])
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    sum += gS_t[h][i * head_dim_ + j] * alpha *
                           cache_k_scaled_[t][h * head_dim_ + i];
                }
                g_r[j] = sum;
            }

            // g_v_t[h, j] = g_r[h, j]
            for (size_t j = 0; j < head_dim_; ++j) {
                grad_v_[t][h * head_dim_ + j] = g_r[j];
            }

            // g_α[h] = sum_{i,j} gS_t[h, i, j] · k_t[h, i] · r_t[h, j]
            //        (∂S_t[h, i, j] / ∂α = k_t[h, i] · r_t[h, j])
            double g_alpha = 0.0;
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    g_alpha += gS_t[h][i * head_dim_ + j] *
                               cache_k_scaled_[t][h * head_dim_ + i] *
                               r_t[j];
                }
            }

            // g_k_t[h, i] from four sources:
            //   (a) Direct outer-product: α · sum_j gS_t[h, i, j] · r_t[h, j]
            //   (b) r_t path: -sum_j g_r[h, j] · S_{t-1}[h, i, j]
            //   (c) α_t path via k_t: -2 · g_α · α² · Sk[h, i]
            //   (d) The "future" path through S_{t-1} carrier is handled in gS_prev_new.
            double g_k_tot[64];
            for (size_t i = 0; i < head_dim_; ++i) {
                double a_sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    a_sum += gS_t[h][i * head_dim_ + j] * r_t[j];
                }
                double a = alpha * a_sum;

                // g_k_b[i] = -sum_j g_r[j] · S_{t-1}[h, j, i]  (NOT [i, j] — see derivation)
                double b_sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    b_sum += g_r[j] * cache_S_[t][h][j * head_dim_ + i];
                }
                double b = -b_sum;

                // g_k_c[i] = -alpha^2 * ((S_{t-1} + S_{t-1}^T) * k_t)[i] * g_alpha
                // (S_{t-1}^T * k_t)[i] = sum_j S_{t-1}[h, j, i] * k_t[h, j]
                double SkT_k = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    SkT_k += cache_S_[t][h][j * head_dim_ + i] *
                             cache_k_scaled_[t][h * head_dim_ + j];
                }
                double c = -alpha * alpha * (Sk[i] + SkT_k) * g_alpha;

                g_k_tot[i] = a + b + c;
            }
            for (size_t i = 0; i < head_dim_; ++i) {
                grad_k_scaled_[t][h * head_dim_ + i] = g_k_tot[i];
            }


            // Compute gS_prev for head h:
            //   gS_{t-1}[h, i, j] = gS_t[h, i, j]                          (direct)
            //                     - g_r[h, j] · k_t[h, i]                  (residual path)
            //                     - g_α · α² · k_t[h, i] · k_t[h, j]      (α path)
            for (size_t i = 0; i < head_dim_; ++i) {
                double k_i = cache_k_scaled_[t][h * head_dim_ + i];
                // gS_dot_k[i] = sum_l gS_t[h, l, i] * k_t[h, l]   (the i-th component of gS_t^T * k_t)
                double gS_dot_k = 0.0;
                for (size_t l = 0; l < head_dim_; ++l) {
                    gS_dot_k += gS_t[h][l * head_dim_ + i] *
                                cache_k_scaled_[t][h * head_dim_ + l];
                }
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_j = cache_k_scaled_[t][h * head_dim_ + j];
                    double direct = gS_t[h][i * head_dim_ + j];
                    double via_r = -alpha * k_j * gS_dot_k;
                    double via_alpha = -alpha * alpha * k_i * k_j * g_alpha;
                    gS_prev_new[h][i * head_dim_ + j] = direct + via_r + via_alpha;
                }
            }
        }

        gS_prev = gS_prev_new;
    }

    // Step 4: Undo the β/|k| scaling
    // k_scaled[d] = (β / |k_raw|) · k_raw[d]
    // Jacobian: dk_scaled[d]/dk_raw[d2] = (β / |k_raw|) · (δ_{d,d2} - k_raw[d]·k_raw[d2] / |k_raw|²)
    // Inverse: g_k_raw[d] = sum_j g_k_scaled[j] · dk_scaled[j]/dk_raw[d]
    //                   = (β / |k_raw|) · g_k_scaled[d] - (β / |k_raw|³) · k_raw[d] · sum_j g_k_scaled[j] · k_raw[j]
    //                   = (β / |k_raw|) · g_k_scaled[d] - k_scaled[d] · (|k_raw| / β) · (sum_j g_k_scaled[j] · k_raw[j]) / |k_raw|²
    //   Let me simplify using ≤2: ks[d] = (β/|k_raw|) · k_raw[d], so k_raw[d] = (|k_raw|/β) · ks[d]
    //   sum_j g_ks[j] · k_raw[j] = (|k_raw|/β) · sum_j g_ks[j] · ks[j]
    //   g_k_raw[d] = (β / |k_raw|) · g_ks[d] - k_raw[d] · (|k_raw| / β) · (sum_j g_ks[j] · ks[j]) / |k_raw|²
    //             = (β / |k_raw|) · g_ks[d] - k_raw[d] · dot(g_ks, ks) / (β · |k_raw|)
    //             = (1 / |k_raw|) · (β · g_ks[d] - k_raw[d] · dot(g_ks, ks) / β)

    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double k_norm_sq = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                double k = cache_k_[t][h * head_dim_ + d];
                k_norm_sq += k * k;
            }
            double k_norm = std::sqrt(k_norm_sq + 1e-12);
            double beta = std::max(cache_beta_[t][h], 1e-12);

            double dot_gks_ks = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                double ks = cache_k_scaled_[t][h * head_dim_ + d];
                double gks = grad_k_scaled_[t][h * head_dim_ + d];
                dot_gks_ks += gks * ks;
            }

            for (size_t d = 0; d < head_dim_; ++d) {
                double gks = grad_k_scaled_[t][h * head_dim_ + d];
                double k_raw = cache_k_[t][h * head_dim_ + d];
                // g_k_raw[d] = (1/|k|) · (β · g_ks[d] - k_raw[d] · dot(g_ks, ks) / β)
                double gk = (beta * gks - k_raw * dot_gks_ks / k_norm) / k_norm;
                grad_k_[t][h * head_dim_ + d] = gk;
            }
        }
    }

    // Step 5: gradient of β from the k-scaling path
    //   g_β[h] = sum_d g_ks[d] · ∂k_scaled[d] / ∂β
    //          = sum_d g_ks[d] · k_raw[d] / |k_raw|
    //          = sum_d g_ks[d] · ks[d] / β
    // Then gradient of β_pre: g_β_pre = g_β · β · (1 - β)
    Tensor grad_beta_pre(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double beta = std::max(cache_beta_[t][h], 1e-12);
            double sum = 0.0;
            for (size_t d = 0; d < head_dim_; ++d) {
                sum += grad_k_scaled_[t][h * head_dim_ + d] *
                       cache_k_scaled_[t][h * head_dim_ + d] / beta;
            }
            double b = cache_beta_[t][h];
            grad_beta_pre[t][h] = sum * b * (1.0 - b);
        }
    }

    // Step 6: backward through the Dense projections
    Tensor gx_q = W_Q_.backward(grad_q_, 0.0);
    Tensor gx_k = W_K_.backward(grad_k_, 0.0);
    Tensor gx_v = W_V_.backward(grad_v_, 0.0);
    Tensor gx_beta = W_beta_.backward(grad_beta_pre, 0.0);

    // Sum contributions to grad_x
    grad_x_ = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_x_[t][d] = gx_q[t][d] + gx_k[t][d] + gx_v[t][d] + gx_beta[t][d];
        }
    }

    return grad_x_;
}

// ---------- update / zero_grad / accessors ----------

void DeltaNet::update_weights(double learning_rate) {
    W_Q_.update_weights(learning_rate);
    W_K_.update_weights(learning_rate);
    W_V_.update_weights(learning_rate);
    W_O_.update_weights(learning_rate);
    W_beta_.update_weights(learning_rate);
}

void DeltaNet::zero_grad() {
    W_Q_.zero_grad();
    W_K_.zero_grad();
    W_V_.zero_grad();
    W_O_.zero_grad();
    W_beta_.zero_grad();
}

std::vector<Tensor*> DeltaNet::parameters() {
    std::vector<Tensor*> params;
    for (Tensor* p : W_Q_.parameters()) params.push_back(p);
    for (Tensor* p : W_K_.parameters()) params.push_back(p);
    for (Tensor* p : W_V_.parameters()) params.push_back(p);
    for (Tensor* p : W_O_.parameters()) params.push_back(p);
    for (Tensor* p : W_beta_.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> DeltaNet::gradients() {
    std::vector<Tensor*> grads;
    for (Tensor* g : W_Q_.gradients()) grads.push_back(g);
    for (Tensor* g : W_K_.gradients()) grads.push_back(g);
    for (Tensor* g : W_V_.gradients()) grads.push_back(g);
    for (Tensor* g : W_O_.gradients()) grads.push_back(g);
    for (Tensor* g : W_beta_.gradients()) grads.push_back(g);
    return grads;
}

Tensor DeltaNet::last_state() const {
    if (cache_S_.empty()) return Tensor(0, 0);
    return cache_S_.back().clone();
}
