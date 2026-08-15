#include "gated_deltanet.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Gated DeltaNet implementation
// ============================================================================
//
// Math reference (per head h, per time t):
//
//   Let k_t   = cache_k_scaled_[t][h * head_dim_ + d]  (after β/|k| scaling).
//   Let v_t   = cache_v_[t][h * head_dim_ + d].
//   Let q_t   = cache_q_[t][h * head_dim_ + d].
//   Let β_t   = cache_beta_[t][h]      ∈ (0, 1)
//   Let g_t   = cache_gate_[t][h]      ∈ (0, 1)  (the Mamba2-style decay gate)
//   S_t[h] is stored as cache_S_[t][h] (head_dim × head_dim in row-major flat).
//
//   G_t[h] = I − β_t[h] · outer(k_t[h], k_t[h])        (head_dim × head_dim)
//
//   S_t[h] = g_t[h] · S_{t-1}[h] · G_t[h] + β_t[h] · outer(k_t[h], v_t[h])
//   o_t[h] = S_t[h] · q_t[h]
//
// ---------- Backward derivation (BPTT, per head h, per time t) ----------
//
// We propagate gS_t = dL/dS_t backward from t = T-1 to t = 0.
// At t = T-1, gS_t = 0 (no future). We then add the output-side contribution.
//
// Step 1 — Output chain:
//   o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]
//   ⇒ gS_t[h, i, j] += grad_concat_o[t, h*head_dim + i] · q_t[h, j]
//   ⇒ grad_q_t[h, j] += sum_i S_t[h, i, j] · grad_concat_o[t, h*head_dim + i]
//
// Step 2 — Update chain (S_t = g_t · S_{t-1} · G_t + β_t · outer(k_t, v_t)):
//
//   (a) Linear in S_{t-1}: ∂S_t / ∂S_{t-1} = g_t · G_t^T (acting on the LEFT of S_{t-1})
//       Wait — careful with matrix contraction:
//         S_t[i, j] = g · sum_p S_{t-1}[i, p] · G[p, j] + β · k[i] · v[j]
//       So dL/dS_{t-1}[i, p] += g · sum_j (dL/dS_t[i, j]) · G[p, j]
//                            = g · (gS_t · G^T)[i, p]
//
//   (b) From v_t: g_v_t[h, j] = sum_i gS_t[h, i, j] · β_t · k_t[h, i]
//
//   (c) From β_t (scalar): β_t appears in two places:
//       (i)  β · outer(k, v)  ⇒  d/dβ = outer(k, v)
//       (ii) G = I − β · k k^T  ⇒  dG/dβ = −outer(k, k)
//       For (i): g_beta += <gS_t, outer(k, v)> = sum_{i,j} gS_t[h,i,j] · k[h,i] · v[h,j]
//       For (ii): g_beta += g · <gS_t · S_{t-1}, −outer(k, k)>
//              = −g · sum_{i,j} (gS_t · S_{t-1})[i, j] · k[h,i] · k[h,j]
//                 (here (gS_t · S_{t-1})[i,j] = sum_p gS_t[i,p] · S_{t-1}[p,j])
//
//   (d) From k_t: k_t appears in two places:
//       (i)  β · outer(k, v)  ⇒  ∂/∂k[i] = β · v (column) ⇒ g_k[i] += β · sum_j gS_t[i,j] · v[j]
//       (ii) G = I − β · k k^T  ⇒  dG/dk[i] = −β · (e_i · k^T + k · e_i^T)
//         For (ii), applied to S_{t-1}·G contract: dL/dS_{t-1} already captured the G^T path
//         in (a) and the β-scaling in (c). But for g_k, we need the contribution of k to G:
//           contribution = g · S_{t-1} · dG/dk[i]
//         For the input term dL/dG = g · gS_t · S_{t-1}^T (per the contraction in (a), the
//         chain rule on G is dL/dG = g · gS_t^T · S_{t-1})... wait, let me redo:
//
//         S_t[i, j] = g · sum_p S_{t-1}[i, p] · G[p, j] + ...
//         dL/dG[p, j] += g · sum_i gS_t[i, j] · S_{t-1}[i, p]
//
//       So dL/dG[p, j] = g · (gS_t^T · S_{t-1})[p, j]. Call this Ggrad.
//
//       Then for k_t[h, i]:
//         dL/dk[h, i] += sum_{p, j} Ggrad[p, j] · dG[p, j]/dk[h, i]
//                     = sum_{p, j} Ggrad[p, j] · (−β) · (δ_{p,i} · k[j] + k[p] · δ_{j,i})
//                     = −β · ( sum_j Ggrad[i, j] · k[j] + sum_p Ggrad[p, i] · k[p] )
//                     = −β · ( (Ggrad · k)[i] + (Ggrad^T · k)[i] )
//                     = −β · ((Ggrad + Ggrad^T) · k)[i]
//
//       Plus the (i) path:
//         g_k[i] += β · sum_j gS_t[i, j] · v[j]
//
//   (e) From g_t (scalar): g_t appears as multiplier on S_{t-1}·G
//         g_gate += sum_{i, j} gS_t[h, i, j] · (S_{t-1} · G)[i, j]
//                = <gS_t[h], S_{t-1}[h] · G[h]>
//
//   (f) gate (g_t) path is independent of S_{t-1} structure (no S_{t-1} derivative through g_t),
//       but it doesn't affect the gS_{t-1} carrier directly.
//
// Step 3 — Undo β/|k| scaling on k_t (same as DeltaNet):
//   k_scaled[d] = (β / ||k_raw||) · k_raw[d]
//   Inverse: g_k_raw[d] = (β / ||k_raw||) · g_k_scaled[d] - k_raw[d] · <g_ks, ks> / (β · ||k_raw||)
//
// Step 4 — Undo sigmoid for β_pre and gate_pre:
//   β_pre  ← σ → β : dβ/dβ_pre = β · (1 − β)
//   gate_pre ← σ → g : dg/dgate_pre = g · (1 − g)
// ============================================================================

// ---------- helpers ----------

static double gdn_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// ---------- constructor ----------

GatedDeltaNet::GatedDeltaNet(size_t d_model, size_t n_heads, size_t head_dim)
    : W_Q_(d_model, 0), W_K_(d_model, 0), W_V_(d_model, 0),
      W_O_(0, 0), W_beta_(d_model, 0), W_gate_(d_model, 0),
      d_model_(d_model),
      n_heads_(n_heads),
      head_dim_(0),  // set below
      d_inner_(0)  // set below
{
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument("GatedDeltaNet: d_model and n_heads must be > 0");
    }
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("GatedDeltaNet: d_model must divide evenly by n_heads");
    }
    head_dim_ = (head_dim == 0) ? d_model_ / n_heads_ : head_dim;
    if (head_dim != 0 && head_dim != head_dim_) {
        throw std::invalid_argument("GatedDeltaNet: head_dim must equal d_model/n_heads (default)");
    }

    d_inner_ = d_model_;  // default: d_inner = d_model

    // Reinitialize projections with correct output dims
    W_Q_     = Dense(d_model_, d_inner_);
    W_K_     = Dense(d_model_, d_inner_);
    W_V_     = Dense(d_model_, d_inner_);
    W_O_     = Dense(d_inner_, d_model_);
    W_beta_  = Dense(d_model_, n_heads_);
    W_gate_  = Dense(d_model_, n_heads_);

    // Smaller init for all weights — the recurrence is sensitive to scale.
    W_Q_.init_weights("uniform");
    W_K_.init_weights("uniform");
    W_V_.init_weights("uniform");
    W_O_.init_weights("uniform");
    W_beta_.init_weights("uniform");
    W_gate_.init_weights("uniform");

    // Biases always zero
    W_Q_.bias.fill(0.0);
    W_K_.bias.fill(0.0);
    W_V_.bias.fill(0.0);
    W_O_.bias.fill(0.0);
    W_beta_.bias.fill(0.0);
    W_gate_.bias.fill(0.0);
}

// ---------- forward ----------

Tensor GatedDeltaNet::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("GatedDeltaNet::forward: input.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    cache_x_ = input.clone();

    // Project
    cache_q_ = W_Q_.forward(input);                 // (T, d_inner)
    cache_k_ = W_K_.forward(input);                 // (T, d_inner)
    cache_v_ = W_V_.forward(input);                 // (T, d_inner)
    Tensor beta_pre = W_beta_.forward(input);       // (T, n_heads)
    Tensor gate_pre = W_gate_.forward(input);       // (T, n_heads)

    // Resize cache tensors to match T
    cache_beta_pre_ = Tensor(T, n_heads_);
    cache_beta_     = Tensor(T, n_heads_);
    cache_gate_pre_ = Tensor(T, n_heads_);
    cache_gate_     = Tensor(T, n_heads_);
    cache_k_scaled_ = Tensor(T, d_inner_);

    // Sigmoid
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            cache_beta_pre_[t][h] = beta_pre[t][h];
            cache_beta_[t][h]     = gdn_sigmoid(beta_pre[t][h]);
            cache_gate_pre_[t][h] = gate_pre[t][h];
            cache_gate_[t][h]     = gdn_sigmoid(gate_pre[t][h]);
        }
    }

    // Per-head k-magnitude normalization: k_t[h] *= (β_t[h] / ||k_t[h]||)
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

    // Per-head gated-delta recurrence
    cache_S_.clear();
    cache_S_.resize(T);

    // State tensor: (n_heads, head_dim * head_dim) — flat row-major per head.
    Tensor current_state(n_heads_, head_dim_ * head_dim_);  // zero-initialized

    Tensor output_concat(T, d_inner_);

    for (size_t t = 0; t < T; ++t) {
        // Cache state BEFORE update at time t
        cache_S_[t] = current_state.clone();

        for (size_t h = 0; h < n_heads_; ++h) {
            double gate = cache_gate_[t][h];
            double beta = cache_beta_[t][h];

            // Step 1: S_t[h] = gate · S_{t-1}[h] · G[h] + beta · outer(k, v)
            // where G[h] = I − beta · outer(k, k)
            //
            // Compute (S_{t-1}[h] · G[h]) into a temp, then scale by gate, then
            // add the beta · outer(k, v) term. We do this per-element.

            // First, compute S_{t-1} · (I − β · outer(k, k)) = S_{t-1} − β · S_{t-1} · kk^T
            // The (i, j) entry: S[i, j] − β · sum_p S[i, p] · k[p] · k[j]
            // = S[i, j] − β · k[j] · (S · k)[i]
            // Define Sk[i] = sum_p S[i, p] · k[p]. Then:
            //   (S · G)[i, j] = S[i, j] − β · Sk[i] · k[j]

            double Sk[64];
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t p = 0; p < head_dim_; ++p) {
                    sum += current_state[h][i * head_dim_ + p] *
                           cache_k_scaled_[t][h * head_dim_ + p];
                }
                Sk[i] = sum;
            }

            // Apply the update: S_new[i, j] = gate · (S[i, j] − β · Sk[i] · k[j]) + β · k[i] · v[j]
            for (size_t i = 0; i < head_dim_; ++i) {
                double k_i = cache_k_scaled_[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_j = cache_k_scaled_[t][h * head_dim_ + j];
                    double v_j = cache_v_[t][h * head_dim_ + j];
                    double old_s = current_state[h][i * head_dim_ + j];
                    double s_after_decay = old_s - beta * Sk[i] * k_j;
                    double new_s = gate * s_after_decay + beta * k_i * v_j;
                    current_state[h][i * head_dim_ + j] = new_s;
                }
            }

            // Output: o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]
            // (where S_t is the post-update state, now in current_state)
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

Tensor GatedDeltaNet::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("GatedDeltaNet::backward: grad_output.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    // Step 1: backward through W_O
    Tensor grad_concat_o = W_O_.backward(grad_output, 0.0);  // (T, d_inner)

    // Step 2: gradient buffers for the recurrence
    grad_q_        = Tensor(T, d_inner_);
    grad_k_        = Tensor(T, d_inner_);
    grad_v_        = Tensor(T, d_inner_);
    grad_k_scaled_ = Tensor(T, d_inner_);

    // Gradient buffers for the gate and beta paths (per-head, pre-sigmoid)
    Tensor grad_beta_pre(T, n_heads_);
    Tensor grad_gate_pre(T, n_heads_);

    // Step 3: backward recurrence through the per-head state
    // At each t, we need the FULL post-update S_t (not just S_{t-1}).
    // We recompute S_t from cache_S_[t] (= S_{t-1}) using cached k/v/beta/gate.
    Tensor gS_prev(n_heads_, head_dim_ * head_dim_);  // zero-initialized

    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 0; --t_signed) {
        size_t t = static_cast<size_t>(t_signed);

        // Re-compute S_t from S_{t-1} = cache_S_[t] for use in the output-side grad_q calc.
        Tensor S_t_cache(cache_S_[t].rows, cache_S_[t].cols);
        S_t_cache = cache_S_[t].clone();

        for (size_t h = 0; h < n_heads_; ++h) {
            double gate = cache_gate_[t][h];
            double beta = cache_beta_[t][h];

            double Sk[64];
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t p = 0; p < head_dim_; ++p) {
                    sum += S_t_cache[h][i * head_dim_ + p] *
                           cache_k_scaled_[t][h * head_dim_ + p];
                }
                Sk[i] = sum;
            }
            for (size_t i = 0; i < head_dim_; ++i) {
                double k_i = cache_k_scaled_[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_j = cache_k_scaled_[t][h * head_dim_ + j];
                    double v_j = cache_v_[t][h * head_dim_ + j];
                    double old_s = S_t_cache[h][i * head_dim_ + j];
                    double s_after_decay = old_s - beta * Sk[i] * k_j;
                    double new_s = gate * s_after_decay + beta * k_i * v_j;
                    S_t_cache[h][i * head_dim_ + j] = new_s;
                }
            }
        }

        // gS_t = gS_prev (the future-side carrier)
        Tensor gS_t = gS_prev.clone();

        // Add output-side contribution:
        //   gS_t[h, i, j] += grad_concat_o[t, h*head_dim + i] · q_t[h, j]
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double go = grad_concat_o[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    gS_t[h][i * head_dim_ + j] += go * cache_q_[t][h * head_dim_ + j];
                }
            }
        }

        // Output-side grad_q contribution:
        //   grad_q_t[h, i] += sum_j S_t[h, j, i] · grad_concat_o[t, h*head_dim + j]
        // where S_t is the POST-UPDATE state (stored in S_t_cache).
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += S_t_cache[h][j * head_dim_ + i] *
                           grad_concat_o[t][h * head_dim_ + j];
                }
                grad_q_[t][h * head_dim_ + i] += sum;
            }
        }

        // Now: backward through the gated-delta update.
        Tensor gS_prev_new(n_heads_, head_dim_ * head_dim_);

        for (size_t h = 0; h < n_heads_; ++h) {
            double gate = cache_gate_[t][h];
            double beta = cache_beta_[t][h];

            // Recompute Sk = S_{t-1} · k_t (the pre-update state acts on k_t)
            double Sk[64];
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t p = 0; p < head_dim_; ++p) {
                    sum += cache_S_[t][h][i * head_dim_ + p] *
                           cache_k_scaled_[t][h * head_dim_ + p];
                }
                Sk[i] = sum;
            }

            // grad_v_t[h, j] = sum_i gS_t[h, i, j] · β · k_t[h, i]
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    sum += gS_t[h][i * head_dim_ + j] * beta *
                           cache_k_scaled_[t][h * head_dim_ + i];
                }
                grad_v_[t][h * head_dim_ + j] = sum;
            }

            // Ggrad = g · gS_t^T · S_{t-1}   (the chain rule on G for the gated path)
            // Ggrad[p, j] = g · sum_i gS_t[i, j] · S_{t-1}[i, p]
            double Ggrad[64][64];
            for (size_t p = 0; p < head_dim_; ++p) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double sum = 0.0;
                    for (size_t i = 0; i < head_dim_; ++i) {
                        sum += gS_t[h][i * head_dim_ + j] *
                               cache_S_[t][h][i * head_dim_ + p];
                    }
                    Ggrad[p][j] = gate * sum;
                }
            }

            // g_beta[h] (scalar per head): three contributions.
//   The forward recurrence is S_t = gate · S_{t-1} · (I − β · outer(k_t, k_t))
//                                 + β · outer(k_t, v_t)
//   where k_t = (β/||k_raw||) · k_raw. So β appears in three places:
//
//   (i)  β · outer(k_t, v_t) contributes ∂(β · k_t[i] · v_t[j]) / ∂β = 2 · outer(k_t, v_t)[i, j]
//        (chain rule on β inside k_t)
//   (ii) −β · outer(k_t, k_t) inside G. For the same reason,
//        ∂(−β · k_t[p] · k_t[j]) / ∂β = −3 · outer(k_t, k_t)[p, j]
//   (iii) k-scaling itself (β / ||k_raw||) — handled separately at the end
//         via the chain rule on the cached grad_k_scaled_.
//
// The full contributions (i)+(ii):
//   g_beta (from (i)) += 2 · <gS_t, outer(k_t, v_t)>
//   g_beta (from (ii)) += −3 · <gate · gS_t^T · S_{t-1}, outer(k_t, k_t)>
double g_beta = 0.0;
for (size_t i = 0; i < head_dim_; ++i) {
    double k_i = cache_k_scaled_[t][h * head_dim_ + i];
    for (size_t j = 0; j < head_dim_; ++j) {
        double v_j = cache_v_[t][h * head_dim_ + j];
        g_beta += 2.0 * gS_t[h][i * head_dim_ + j] * k_i * v_j;
    }
}
for (size_t p = 0; p < head_dim_; ++p) {
    double k_p = cache_k_scaled_[t][h * head_dim_ + p];
    for (size_t j = 0; j < head_dim_; ++j) {
        double k_j = cache_k_scaled_[t][h * head_dim_ + j];
        g_beta += -3.0 * Ggrad[p][j] * k_p * k_j;
    }
}

            // g_gate[h] (scalar per head):
            //   g_gate += sum_{i, j} gS_t[h, i, j] · (S_{t-1} · G)[i, j]
            //   where (S_{t-1} · G)[i, j] = S_{t-1}[i, j] − β · Sk[i] · k[j]
            double g_gate = 0.0;
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k_j = cache_k_scaled_[t][h * head_dim_ + j];
                    double sGt_ij = cache_S_[t][h][i * head_dim_ + j] - beta * Sk[i] * k_j;
                    g_gate += gS_t[h][i * head_dim_ + j] * sGt_ij;
                }
            }

            // g_k_t[h, i] (per-element, into grad_k_scaled_):
            //   (i) from β · outer(k, v): g_k[i] += β · sum_j gS_t[i, j] · v[j]
            //   (ii) from G = I − β · outer(k, k), via Ggrad:
            //        g_k[i] += sum_{p, j} Ggrad[p, j] · dG[p, j]/dk[i]
            //                = sum_{p, j} Ggrad[p, j] · (−β) · (δ_{p,i} · k[j] + k[p] · δ_{j,i})
            //                = −β · (sum_j Ggrad[i, j] · k[j]  +  sum_p Ggrad[p, i] · k[p])
            //                = −β · ((Ggrad · k)[i] + (Ggrad^T · k)[i])
            //                = −β · ((Ggrad + Ggrad^T) · k)[i]
            for (size_t i = 0; i < head_dim_; ++i) {
                double a = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    a += gS_t[h][i * head_dim_ + j] * cache_v_[t][h * head_dim_ + j];
                }
                double contrib_v = beta * a;

                double Gk_i = 0.0;
                for (size_t p = 0; p < head_dim_; ++p) {
                    Gk_i += Ggrad[i][p] * cache_k_scaled_[t][h * head_dim_ + p];
                }
                double GTk_i = 0.0;
                for (size_t p = 0; p < head_dim_; ++p) {
                    GTk_i += Ggrad[p][i] * cache_k_scaled_[t][h * head_dim_ + p];
                }
                double contrib_G = -beta * (Gk_i + GTk_i);

                grad_k_scaled_[t][h * head_dim_ + i] = contrib_v + contrib_G;
            }

            // gS_{t-1}[h, i, p] = gate · (gS_t · G^T)[i, p]
            // = gate · sum_j gS_t[h, i, j] · G[p, j]
            // where G[p, j] = δ_{p, j} − β · k[p] · k[j]
            // = gate · ( gS_t[h, i, p]  −  β · k[p] · sum_j gS_t[h, i, j] · k[j] )
            //
            // Define gS_dot_k[i] = sum_j gS_t[h, i, j] · k[j]  (the i-th row of gS_t dotted with k).
            for (size_t i = 0; i < head_dim_; ++i) {
                double gS_dot_k = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    gS_dot_k += gS_t[h][i * head_dim_ + j] *
                                cache_k_scaled_[t][h * head_dim_ + j];
                }
                for (size_t p = 0; p < head_dim_; ++p) {
                    double k_p = cache_k_scaled_[t][h * head_dim_ + p];
                    double direct = gS_t[h][i * head_dim_ + p];
                    double via_G = -beta * k_p * gS_dot_k;
                    gS_prev_new[h][i * head_dim_ + p] = gate * (direct + via_G);
                }
            }

            // Convert scalar per-head grads to pre-sigmoid grads
            // β_pre: dβ/dβ_pre = β · (1 − β)
            // gate_pre: dg/dgate_pre = gate · (1 − gate)
            grad_beta_pre[t][h] = g_beta * cache_beta_[t][h] * (1.0 - cache_beta_[t][h]);
            grad_gate_pre[t][h] = g_gate * cache_gate_[t][h] * (1.0 - cache_gate_[t][h]);
        }

        gS_prev = gS_prev_new;
    }

    // Step 4: Undo the β/|k| scaling on k_t → k_raw
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
                // g_k_raw[d] = (1/|k|) · (β · g_ks[d] − k_raw[d] · dot(g_ks, ks) / β)
                double gk = (beta * gks - k_raw * dot_gks_ks / k_norm) / k_norm;
                grad_k_[t][h * head_dim_ + d] = gk;
            }
        }
    }

    // Step 5: backward through the Dense projections
    Tensor gx_q     = W_Q_.backward(grad_q_, 0.0);
    Tensor gx_k     = W_K_.backward(grad_k_, 0.0);
    Tensor gx_v     = W_V_.backward(grad_v_, 0.0);
    Tensor gx_beta  = W_beta_.backward(grad_beta_pre, 0.0);
    Tensor gx_gate  = W_gate_.backward(grad_gate_pre, 0.0);

    // Sum contributions to grad_x
    grad_x_ = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_model_; ++d) {
            grad_x_[t][d] = gx_q[t][d] + gx_k[t][d] + gx_v[t][d] +
                            gx_beta[t][d] + gx_gate[t][d];
        }
    }

    return grad_x_;
}

// ---------- update / zero_grad / accessors ----------

void GatedDeltaNet::update_weights(double learning_rate) {
    W_Q_.update_weights(learning_rate);
    W_K_.update_weights(learning_rate);
    W_V_.update_weights(learning_rate);
    W_O_.update_weights(learning_rate);
    W_beta_.update_weights(learning_rate);
    W_gate_.update_weights(learning_rate);
}

void GatedDeltaNet::zero_grad() {
    W_Q_.zero_grad();
    W_K_.zero_grad();
    W_V_.zero_grad();
    W_O_.zero_grad();
    W_beta_.zero_grad();
    W_gate_.zero_grad();
}

std::vector<Tensor*> GatedDeltaNet::parameters() {
    std::vector<Tensor*> params;
    for (Tensor* p : W_Q_.parameters())     params.push_back(p);
    for (Tensor* p : W_K_.parameters())     params.push_back(p);
    for (Tensor* p : W_V_.parameters())     params.push_back(p);
    for (Tensor* p : W_O_.parameters())     params.push_back(p);
    for (Tensor* p : W_beta_.parameters())  params.push_back(p);
    for (Tensor* p : W_gate_.parameters())  params.push_back(p);
    return params;
}

std::vector<Tensor*> GatedDeltaNet::gradients() {
    std::vector<Tensor*> grads;
    for (Tensor* g : W_Q_.gradients())     grads.push_back(g);
    for (Tensor* g : W_K_.gradients())     grads.push_back(g);
    for (Tensor* g : W_V_.gradients())     grads.push_back(g);
    for (Tensor* g : W_O_.gradients())     grads.push_back(g);
    for (Tensor* g : W_beta_.gradients())  grads.push_back(g);
    for (Tensor* g : W_gate_.gradients())  grads.push_back(g);
    return grads;
}

Tensor GatedDeltaNet::last_state() const {
    if (cache_S_.empty()) return Tensor(0, 0);
    return cache_S_.back().clone();
}