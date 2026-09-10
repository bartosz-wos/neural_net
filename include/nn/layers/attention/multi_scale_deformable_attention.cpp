// Multi-Scale Deformable 1D Attention — Zhu et al., ICLR 2021
//   "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
//   https://arxiv.org/abs/2010.04159
//
// See multi_scale_deformable_attention.h for the full formulation and backward
// derivation.

#include "multi_scale_deformable_attention.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {

inline double msd_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double msd_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

inline double msd_tanh_deriv(double th) {
    return 1.0 - th * th;
}

// Strided-pool index: row r of level l corresponds to row r · 2^l of the input.
inline size_t msd_pool_to_src(size_t r, size_t l) {
    return r * (1u << l);
}

// Strided-scatter: row r of level l was pooled from row r · 2^l of the
// (already-projected) input. Inverse mapping.

} // namespace

// ============================================================================
// MultiScaleDeformable1DAttention
// ============================================================================

MultiScaleDeformable1DAttention::MultiScaleDeformable1DAttention(size_t d_model,
                                                                 size_t num_heads,
                                                                 size_t num_levels,
                                                                 size_t num_points)
    : d_model_(d_model), num_heads_(num_heads),
      num_levels_(num_levels), num_points_(num_points),
      head_dim_(d_model && num_heads ? d_model / num_heads : 0),
      LK_(num_levels * num_points),
      HLK_(num_heads * num_levels * num_points),
      W_q(d_model, d_model),
      W_offsets(d_model, 0),     // placeholder; set below
      W_attn(d_model, 0),        // placeholder
      W_o(d_model, d_model) {
    if (d_model == 0)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: num_heads must be > 0");
    if (num_levels == 0)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: num_levels must be > 0");
    if (num_points == 0)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: num_points must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MultiScaleDeformable1DAttention: d_model must be divisible by num_heads");

    head_dim_ = d_model / num_heads;

    // Now that we know d_model and num_heads, construct the level/projection pieces.
    // We have to defer the actual Dense constructors — they require known sizes at
    // construction. Since C++ initializer-list order is fixed, we initialize them
    // here in the body.
    // Replace W_offsets/W_attn placeholders.
    W_offsets = Dense(d_model, HLK_);
    W_attn    = Dense(d_model, HLK_);

    level_v_projs_.reserve(num_levels_);
    for (size_t l = 0; l < num_levels_; ++l) {
        level_v_projs_.emplace_back(d_model, d_model);
    }
}

Tensor MultiScaleDeformable1DAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("MultiScaleDeformable1DAttention: input.rows must be > 0");

    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t L = num_levels_;
    const size_t P = num_points_;
    const size_t dh = head_dim_;
    const size_t HLK = HLK_;

    last_input_ = input.clone();

    // ---- Q = X · W_q + b_q ----
    last_q_ = W_q.forward(input);   // (N, d_model)

    // ---- offsets and attn logits ----
    last_offsets_ = W_offsets.forward(input);   // (N, H·L·K) RAW pre-tanh
    Tensor A_raw  = W_attn.forward(input);       // (N, H·L·K)

    // ---- Per level: compute V^l = level_v_projs[l].forward(X), then strided-pool ----
    // level_lengths_[l] = ⌈N / 2^l⌉
    level_lengths_.resize(L);
    level_row_offsets_.resize(L);
    size_t total_pooled = 0;
    for (size_t l = 0; l < L; ++l) {
        level_lengths_[l] = (N + ((1u << l) - 1u)) >> l;  // ceil(N / 2^l)
        level_row_offsets_[l] = total_pooled;
        total_pooled += level_lengths_[l];
    }
    last_levels_ = Tensor(total_pooled, d_model_);
    for (size_t l = 0; l < L; ++l) {
        Tensor Vl = level_v_projs_[l].forward(input);  // (N, d_model)
        for (size_t r = 0; r < level_lengths_[l]; ++r) {
            size_t src_row = msd_pool_to_src(r, l);
            if (src_row >= N) src_row = N - 1;  // safety for ceiling overflow
            for (size_t c = 0; c < d_model_; ++c) {
                last_levels_(level_row_offsets_[l] + r, c) = Vl(src_row, c);
            }
        }
    }

    // ---- Positions and bilinear samples ----
    last_positions_ = Tensor(N, HLK);
    last_V_sampled_ = Tensor(N, HLK * dh);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t l = 0; l < L; ++l) {
                const size_t n_l = level_lengths_[l];
                const size_t n_l_m1 = (n_l > 1) ? (n_l - 1) : 1;
                const double scale_l = std::max(1.0, n_l_m1 * 0.5);
                // ref_l[t] = (t / max(1, N-1)) · (n_l - 1)
                const double ref_t = (N > 1) ?
                    ((double)t * n_l_m1 / (double)(N - 1)) :
                    0.0;
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    double raw = last_offsets_(t, hlk);
                    double th = std::tanh(raw);
                    double pos = ref_t + th * scale_l;
                    // Clamp to [0, n_l - 1]
                    if (pos < 0.0) pos = 0.0;
                    if (pos > (double)n_l_m1) pos = (double)n_l_m1;
                    last_positions_(t, hlk) = pos;

                    size_t i = (size_t)std::floor(pos);
                    size_t j = (i >= n_l - 1) ? i : (i + 1);
                    double alpha_w = pos - (double)i;
                    double beta_w = 1.0 - alpha_w;

                    for (size_t d = 0; d < dh; ++d) {
                        size_t feat = h * dh + d;
                        double v_i = last_levels_(level_row_offsets_[l] + i, feat);
                        double v_j = last_levels_(level_row_offsets_[l] + j, feat);
                        last_V_sampled_(t, hlk * dh + d) = beta_w * v_i + alpha_w * v_j;
                    }
                }
            }
        }
    }

    // ---- Attention: softmax over L*K jointly per (t, h) ----
    last_attn_ = Tensor(N, HLK);
    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            // Find max over (l, k) for numerical stability.
            double mx = -1e30;
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    double s = A_raw(t, hlk);
                    last_attn_(t, hlk) = s;  // temporarily store the logits
                    if (s > mx) mx = s;
                }
            }
            double sum = 0.0;
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    double e = std::exp(last_attn_(t, hlk) - mx);
                    last_attn_(t, hlk) = e;
                    sum += e;
                }
            }
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    last_attn_(t, hlk) /= sum;
                }
            }
        }
    }

    // ---- Per-head output: out[t, h, d] = Σ_{l,k} A[t, h, l, k] · V_sampled[t, h, l, k, d] ----
    last_concat_ = Tensor(N, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t l = 0; l < L; ++l) {
                    for (size_t k = 0; k < P; ++k) {
                        size_t hlk = (h * L + l) * P + k;
                        acc += last_attn_(t, hlk) * last_V_sampled_(t, hlk * dh + d);
                    }
                }
                last_concat_(t, h * dh + d) = acc;
            }
        }
    }

    // ---- Output projection ----
    return W_o.forward(last_concat_);
}

Tensor MultiScaleDeformable1DAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t N = last_input_.rows;
    const size_t H = num_heads_;
    const size_t L = num_levels_;
    const size_t P = num_points_;
    const size_t dh = head_dim_;
    const size_t HLK = HLK_;

    if (grad_output.rows != N || grad_output.cols != d_model_)
        throw std::invalid_argument(
            "MultiScaleDeformable1DAttention::backward: grad_output shape mismatch");

    // ---- Output projection: Y = last_concat_ · W_o^T + b_o ----
    Tensor d_concat(N, d_model_);
    d_concat.fill(0.0);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i) acc += grad_output(t, i) * W_o.weights(i, j);
            d_concat(t, j) = acc;
        }
    }
    // dW_o[j, i] += Σ_t last_concat_[t, i] · grad_output[t, j]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += last_concat_(t, i) * grad_output(t, j);
            W_o.grad_weights(j, i) += acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += grad_output(t, j);
        W_o.grad_bias(0, j) += bacc;
    }

    // ---- Per-head backward through attention + bilinear sampling ----
    Tensor d_V_sampled(N, HLK * dh);
    Tensor d_A(N, HLK);
    Tensor d_A_raw(N, HLK);
    Tensor d_pos(N, HLK);
    Tensor d_q(N, d_model_);
    d_q.fill(0.0);

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            // d_V_sampled[t, hlk, d] = A[t, hlk] · d_concat[t, h*dh + d]
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    for (size_t d = 0; d < dh; ++d) {
                        d_V_sampled(t, hlk * dh + d) =
                            last_attn_(t, hlk) * d_concat(t, h * dh + d);
                    }
                }
            }
            // d_A[t, hlk] = Σ_d d_concat[t, h*dh + d] · V_sampled[t, hlk, d]
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    double a = 0.0;
                    for (size_t d = 0; d < dh; ++d) {
                        a += d_concat(t, h * dh + d) * last_V_sampled_(t, hlk * dh + d);
                    }
                    d_A(t, hlk) = a;
                }
            }
            // Softmax backward (over L*K jointly):
            //   d_A_raw[t, hlk] = A[t, hlk] · (d_A[t, hlk] − Σ_{l',k'} A[t, hlk'] · d_A[t, hlk'])
            double inner = 0.0;
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    inner += last_attn_(t, hlk) * d_A(t, hlk);
                }
            }
            for (size_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    d_A_raw(t, hlk) = last_attn_(t, hlk) * (d_A(t, hlk) - inner);
                }
            }
            // d_q[t, h*dh + d] += Σ_{l,k} A[t, hlk] · ... (no, Q has no direct path here;
            //   the bilinear V sample does NOT touch Q. Only A and Δ do. So d_q stays 0
            //   from the attention path — but it IS touched through X→W_q via d_X below.)
        }
    }

    // ---- Bilinear backward: V_sampled → V^l (per level), positions ----
    // Per-level d_V (full d_model), starting at zero, plus d_pos.
    std::vector<Tensor> d_V_per_level(L);
    for (size_t l = 0; l < L; ++l) {
        d_V_per_level[l] = Tensor(level_lengths_[l], d_model_);
        d_V_per_level[l].fill(0.0);
    }

    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t l = 0; l < L; ++l) {
                const size_t n_l = level_lengths_[l];
                const size_t n_l_m1 = (n_l > 1) ? (n_l - 1) : 1;
                const double scale_l = std::max(1.0, n_l_m1 * 0.5);

                for (size_t k = 0; k < P; ++k) {
                    size_t hlk = (h * L + l) * P + k;
                    double pos = last_positions_(t, hlk);
                    size_t i = (size_t)std::floor(pos);
                    size_t j = (i >= n_l - 1) ? i : (i + 1);
                    double alpha_w = pos - (double)i;
                    double beta_w = 1.0 - alpha_w;

                    // d_pos[t, hlk] = Σ_d (V^l[j, feat] − V^l[i, feat]) · d_V_sampled[t, hlk, d]
                    double dpos_local = 0.0;
                    for (size_t d = 0; d < dh; ++d) {
                        size_t feat = h * dh + d;
                        double diff = last_levels_(level_row_offsets_[l] + j, feat)
                                    - last_levels_(level_row_offsets_[l] + i, feat);
                        dpos_local += diff * d_V_sampled(t, hlk * dh + d);
                    }

                    // Bilinear sample backward into V^l
                    for (size_t d = 0; d < dh; ++d) {
                        size_t feat = h * dh + d;
                        double dvs = d_V_sampled(t, hlk * dh + d);
                        d_V_per_level[l](i, feat) += beta_w * dvs;
                        d_V_per_level[l](j, feat) += alpha_w * dvs;
                    }

                    // Clamp mask: if saturated at boundary, d_pos = 0. (Note: this
                    // is structurally redundant in 1D — at pos=0 or pos=n_l_m1, the
                    // bilinear backward's (V[j]-V[i]) term is 0 because i=j at the
                    // boundary. The mask is kept as defense against future 2D
                    // extensions where this redundancy breaks down.)
                    bool saturated = (pos <= 0.0) || (pos >= (double)n_l_m1);
                    if (saturated) dpos_local = 0.0;
                    d_pos(t, hlk) = dpos_local;

                    // Backward through pos = ref + s_l · tanh(Δ_raw)
                    //   d_Δ_raw = d_pos · s_l · (1 − tanh²(Δ_raw))
                    double th = std::tanh(last_offsets_(t, hlk));
                    double delta_d = dpos_local * scale_l * msd_tanh_deriv(th);
                    // We'll accumulate d_Δ_raw into a separate buffer below.
                    // Store delta_d in d_A_raw slot temporarily? No — d_A_raw is separate.
                    // Re-derive here:
                    // We'll handle d_offsets backward in a separate pass below.
                    // Save: d_pos[t, hlk] is already correct.
                    // We need d_Δ_raw per (t, hlk) — compute it now and stash in d_pos
                    // (overwriting d_pos for the level-V path is fine since d_pos is
                    // already consumed).
                    d_pos(t, hlk) = delta_d;  // now holds d_Δ_raw
                }
            }
        }
    }

    // d_pos now holds d_Δ_raw per (t, hlk). We need the original d_pos separately
    // for what? Actually we don't — d_pos was only used to compute d_Δ_raw, and
    // d_V_per_level was computed inline. So d_pos being overwritten is fine.

    // ---- Backward through W_offsets, W_attn, W_q, and X via the four heads ----
    Tensor d_X(N, d_model_);
    d_X.fill(0.0);

    // W_offsets: dW += X^T · d_Δ_raw; dX += d_Δ_raw · W_offsets^T; db += colsum
    for (size_t j = 0; j < HLK; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += last_input_(t, k) * d_pos(t, j);
            W_offsets.grad_weights(j, k) += acc;
        }
    }
    for (size_t t = 0; t < N; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < HLK; ++j) acc += d_pos(t, j) * W_offsets.weights(j, k);
            d_X(t, k) += acc;
        }
    }
    for (size_t j = 0; j < HLK; ++j) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += d_pos(t, j);
        W_offsets.grad_bias(0, j) += bacc;
    }

    // W_attn: dW += X^T · d_A_raw; dX += d_A_raw · W_attn^T; db += colsum
    for (size_t j = 0; j < HLK; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += last_input_(t, k) * d_A_raw(t, j);
            W_attn.grad_weights(j, k) += acc;
        }
    }
    for (size_t t = 0; t < N; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < HLK; ++j) acc += d_A_raw(t, j) * W_attn.weights(j, k);
            d_X(t, k) += acc;
        }
    }
    for (size_t j = 0; j < HLK; ++j) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) bacc += d_A_raw(t, j);
        W_attn.grad_bias(0, j) += bacc;
    }

    // W_q: dW += X^T · d_q; dX += d_q · W_q^T; db += colsum
    // d_q is zero from the attention path (Q is not directly used in attention here,
    // only via the d_X from W_q), but the W_q backward is structurally needed for
    // gradient flow if any consumer uses Q. We treat d_q as zero (since attention has
    // no Q-K dot-product in this paper's efficiency design), but still do the W_q
    // backward for completeness using d_q = 0 — i.e. nothing accumulates.
    // (In this paper, Q is unused in the attention op, so d_q = 0 is the correct
    // analytical answer. The W_q weights exist for forward compatibility / future
    // extension but do not get gradients from the attention path.)
    // So skip W_q backward — d_q is zero.

    // ---- Backward through level V projections ----
    // For each level l: d_V_per_level[l] is the gradient on V^l = level_v_projs[l](X).
    // The backward of each level_v_projs is standard Dense backward:
    //   d_X (projected) += d_V_per_level[l] · W_v_l^T (scatter via the strided-pool inverse)
    // But first, d_V_per_level[l] needs to be SCATTERED back to (N, d_model_)
    // because level_v_projs[l].forward(X) was applied to all N rows, then strided-pooled.
    //
    // Forward:   V^l_pooled[r, :] = V^l_full[r · 2^l, :]   (strided pool)
    //   where V^l_full = level_v_projs[l].forward(X)
    // Backward: d_V^l_full[r · 2^l, :] += d_V^l_pooled[r, :]   (strided scatter)
    //           d_V^l_full[r · 2^l, :] is the gradient on V^l_full — this is what
    //           level_v_projs[l].backward() expects as its grad_output.

    for (size_t l = 0; l < L; ++l) {
        const size_t n_l = level_lengths_[l];
        Tensor d_V_full(N, d_model_);
        d_V_full.fill(0.0);
        for (size_t r = 0; r < n_l; ++r) {
            size_t src_row = msd_pool_to_src(r, l);
            if (src_row >= N) src_row = N - 1;
            for (size_t c = 0; c < d_model_; ++c) {
                d_V_full(src_row, c) += d_V_per_level[l](r, c);
            }
        }
        // Now run Dense::backward on the full N-row gradient.
        Tensor d_X_from_V = level_v_projs_[l].backward(d_V_full, 0.0);
        for (size_t t = 0; t < N; ++t) {
            for (size_t c = 0; c < d_model_; ++c) {
                d_X(t, c) += d_X_from_V(t, c);
            }
        }
    }

    return d_X;
}

void MultiScaleDeformable1DAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_attn.update_weights(learning_rate);
    W_offsets.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (auto& lvl : level_v_projs_) lvl.update_weights(learning_rate);
}

void MultiScaleDeformable1DAttention::zero_grad() {
    W_q.zero_grad();
    W_attn.zero_grad();
    W_offsets.zero_grad();
    W_o.zero_grad();
    for (auto& lvl : level_v_projs_) lvl.zero_grad();
}

std::vector<Tensor*> MultiScaleDeformable1DAttention::parameters() {
    std::vector<Tensor*> p;
    p.reserve(8 + 2 * num_levels_);
    p.push_back(&W_q.weights); p.push_back(&W_q.bias);
    p.push_back(&W_attn.weights); p.push_back(&W_attn.bias);
    p.push_back(&W_offsets.weights); p.push_back(&W_offsets.bias);
    p.push_back(&W_o.weights); p.push_back(&W_o.bias);
    for (auto& lvl : level_v_projs_) {
        p.push_back(&lvl.weights); p.push_back(&lvl.bias);
    }
    return p;
}

std::vector<Tensor*> MultiScaleDeformable1DAttention::gradients() {
    std::vector<Tensor*> g;
    g.reserve(8 + 2 * num_levels_);
    g.push_back(&W_q.grad_weights); g.push_back(&W_q.grad_bias);
    g.push_back(&W_attn.grad_weights); g.push_back(&W_attn.grad_bias);
    g.push_back(&W_offsets.grad_weights); g.push_back(&W_offsets.grad_bias);
    g.push_back(&W_o.grad_weights); g.push_back(&W_o.grad_bias);
    for (auto& lvl : level_v_projs_) {
        g.push_back(&lvl.grad_weights); g.push_back(&lvl.grad_bias);
    }
    return g;
}

// ============================================================================
// MultiScaleDeformable1DBlock
// ============================================================================

MultiScaleDeformable1DBlock::MultiScaleDeformable1DBlock(size_t d_model, size_t num_heads,
                                                       size_t num_levels, size_t num_points)
    : attn(d_model, num_heads, num_levels, num_points),
      ffn1(d_model, d_model * 2), ffn2(d_model * 2, d_model),
      ln1_(std::make_unique<LayerNorm>(d_model)),
      ln2_(std::make_unique<LayerNorm>(d_model)) {}

Tensor MultiScaleDeformable1DBlock::forward(const Tensor& x) {
    last_ln1_ = ln1_->forward(x);
    Tensor a = attn.forward(last_ln1_);
    Tensor h1 = x + a;

    last_ln2_ = ln2_->forward(h1);
    last_f1_ = ffn1.forward(last_ln2_);
    last_f1g_ = last_f1_.apply(msd_gelu);
    Tensor f2 = ffn2.forward(last_f1g_);
    return h1 + f2;
}

Tensor MultiScaleDeformable1DBlock::backward(const Tensor& grad_output, double lr) {
    Tensor df_f2 = grad_output;

    Tensor df1g = ffn2.backward(df_f2, lr);
    Tensor df1 = df1g.hadamard(last_f1_.apply(msd_gelu_deriv));
    Tensor dln2 = ffn1.backward(df1, lr);
    Tensor dh1_part = ln2_->backward(dln2, lr);
    Tensor dh1 = grad_output + dh1_part;

    Tensor dln1_a = attn.backward(dh1, lr);
    Tensor dx_part = ln1_->backward(dln1_a, lr);
    return dh1 + dx_part;
}

void MultiScaleDeformable1DBlock::update_weights(double lr) {
    attn.update_weights(lr);
    ffn1.update_weights(lr); ffn2.update_weights(lr);
    ln1_->update_weights(lr); ln2_->update_weights(lr);
}

void MultiScaleDeformable1DBlock::zero_grad() {
    attn.zero_grad();
    ffn1.zero_grad(); ffn2.zero_grad();
    ln1_->zero_grad(); ln2_->zero_grad();
}

std::vector<Tensor*> MultiScaleDeformable1DBlock::parameters() {
    std::vector<Tensor*> p = attn.parameters();
    auto f1 = ffn1.parameters();
    p.insert(p.end(), f1.begin(), f1.end());
    auto f2 = ffn2.parameters();
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

std::vector<Tensor*> MultiScaleDeformable1DBlock::gradients() {
    std::vector<Tensor*> g = attn.gradients();
    auto f1 = ffn1.gradients();
    g.insert(g.end(), f1.begin(), f1.end());
    auto f2 = ffn2.gradients();
    g.insert(g.end(), f2.begin(), f2.end());
    return g;
}
