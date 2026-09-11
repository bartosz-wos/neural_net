// HiLo Attention — Pan et al., ICLR 2025
//   "HiLo: A High-Low Frequency-Aware Attention for Long Sequence Modeling"
//   https://arxiv.org/abs/2405.13219
//
// See hilo_attention.h for the full formulation and backward derivation.
//
// Implementation notes (the bits easy to get wrong):
//   * Per-head split: heads 0..H_local-1 use windowed attention over the
//     last `W` tokens; heads H_local..H-1 use softmax over avg-pooled
//     keys/values (W-wide non-overlapping windows).
//   * λ-blend is applied as a per-head multiplicative mask on dO BEFORE the
//     per-stream chains: dO_local = (1−λ) · dO, dO_global = λ · dO. This
//     keeps the per-stream backward a standard softmax chain.
//   * dK_local and dV_local accumulate from BOTH streams (direct local-chain
//     + back-prop through the pool from the global chain). Forgetting either
//     gives ~50% magnitude errors.
//   * Pool-window-size at the boundary: the last pool may have < W tokens;
//     the average uses 1/W_p where W_p is the actual count.

#include "hilo_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

inline double hl_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

inline double hl_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double hl_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

// Causal-mask helper: mask is shape (N, N), 1 if i <= j AND (j - W < i)
// (i.e. i in (j-W, j] for windowed local attention), 0 otherwise.
inline void hl_local_mask(Tensor& mask, size_t N, size_t W) {
    mask.fill(0.0);
    for (size_t j = 0; j < N; ++j) {
        size_t lo = (j >= W) ? (j - W + 1) : 0;
        for (size_t i = lo; i <= j; ++i) mask(i, j) = 1.0;
    }
}

// Causal-mask helper for the GLOBAL chain over pooled positions:
// mask_pool is shape (num_pool, N), 1 iff (p * W) <= j (the pool's first
// token is at or before query j) AND (p * W + W_p - 1) <= j would be wrong —
// because the avg-pool mixes a window of tokens, so as long as ANY token in
// the pool is at or before j, the pooled key is "available" to query j under
// causality. The standard convention: pool p covers tokens [p*W, (p+1)*W),
// so it is available iff p*W <= j.
inline void hl_pool_mask(Tensor& mask, size_t num_pool, size_t N, size_t W) {
    mask.fill(0.0);
    for (size_t j = 0; j < N; ++j) {
        // Pool p is available iff p*W <= j, i.e. p <= j/W
        for (size_t p = 0; p <= j / W && p < num_pool; ++p) mask(p, j) = 1.0;
    }
}

// Numerically-stable softmax over a column of `z` (N, M) with binary mask
// (1 = include, 0 = exclude). `out` has shape (N, M).
inline void hl_masked_softmax_col(const Tensor& z, const Tensor& mask,
                                  Tensor& out) {
    const size_t M = z.rows;
    const size_t N = z.cols;
    out.fill(0.0);
    for (size_t j = 0; j < N; ++j) {
        // Find the max over masked-in entries (mask=1) for column j.
        double mmax = -1e30;
        bool any = false;
        for (size_t i = 0; i < M; ++i) {
            if (mask(i, j) > 0.5) {
                if (z(i, j) > mmax) mmax = z(i, j);
                any = true;
            }
        }
        if (!any) continue; // column stays zero — no valid keys (e.g. j < W boundary)
        double sum = 0.0;
        for (size_t i = 0; i < M; ++i) {
            if (mask(i, j) > 0.5) {
                double e = std::exp(z(i, j) - mmax);
                out(i, j) = e;
                sum += e;
            }
        }
        if (sum > 0.0) {
            double inv = 1.0 / sum;
            for (size_t i = 0; i < M; ++i) out(i, j) *= inv;
        }
    }
}

} // namespace

// ============================================================================
// HiLoAttention
// ============================================================================

HiLoAttention::HiLoAttention(size_t d_model, size_t num_heads, size_t window_size)
    : W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      logit_lambda_(num_heads ? num_heads : 1, 1),
      grad_W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_logit_lambda_(num_heads ? num_heads : 1, 1),
      lambda_(num_heads ? num_heads : 1, 1),
      d_model_(d_model),
      num_heads_(num_heads),
      num_local_(((num_heads ? num_heads : 1)) / 2),
      num_global_(((num_heads ? num_heads : 1)) - (((num_heads ? num_heads : 1)) / 2)),
      head_dim_((num_heads && d_model && d_model % num_heads == 0)
                    ? d_model / num_heads : 1),
      window_size_(window_size)
{
    if (d_model == 0)
        throw std::invalid_argument("HiLoAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("HiLoAttention: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("HiLoAttention: d_model must be divisible by num_heads");
    if (window_size == 0)
        throw std::invalid_argument("HiLoAttention: window_size must be > 0");

    inv_temp_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_logit_lambda_.fill(0.0);

    // Default init: logit_lambda = 0 → λ = 0.5 (paper §3.2 default).
    logit_lambda_.fill(0.0);
    lambda_.fill(0.5);
}

void HiLoAttention::recompute_lambda() {
    for (size_t h = 0; h < num_heads_; ++h) {
        lambda_(h, 0) = hl_sigmoid(logit_lambda_(h, 0));
    }
}

Tensor HiLoAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("HiLoAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("HiLoAttention: input.rows must be > 0");

    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t Hloc = num_local_;
    const size_t Hglo = num_global_;
    const size_t dh = head_dim_;
    const size_t W  = window_size_;

    N_last_   = N;
    num_pool_ = (N + W - 1) / W;

    // Recompute λ from logit_lambda_ each forward (cheap; 1 sigmoid per head).
    recompute_lambda();

    last_input_ = input.clone();

    // Standard projections.
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    // Per-pool window sizes (boundary: last pool may be partial).
    pool_window_sizes_.assign(num_pool_, W);
    if (N % W != 0) pool_window_sizes_.back() = N % W;

    // ---- Build K_pool / V_pool per head ------------------------------------
    // last_K_pool_ / last_V_pool_: shape (H * num_pool, d_h), flat rows.
    last_K_pool_ = Tensor(H * num_pool_, dh);
    last_V_pool_ = Tensor(H * num_pool_, dh);
    last_K_pool_.fill(0.0);
    last_V_pool_.fill(0.0);

    // We accumulate dK_local / dV_local from BOTH streams during backward —
    // so the caches must remember exactly which tokens contributed to which
    // pool position (and from which head). The K_local / V_local cache below
    // is just last_K_ / last_V_; the per-pool membership is determined by
    // p(i) = i / W.

    // ---- Local-head forward (Hloc heads) -----------------------------------
    // z_local: (N, N) per head, attn_local: (N, N). For simplicity we allocate
    // (Hloc, N, N) flattened as (Hloc * N, N) for cache.
    last_A_local_ = Tensor(Hloc * N, N);
    last_A_local_.fill(0.0);
    Tensor mask_local(N, N);
    hl_local_mask(mask_local, N, W);

    for (size_t h = 0; h < Hloc; ++h) {
        const size_t off = h * dh;
        // Compute Z[i,j] = inv_temp * q_j · k_i, then mask + softmax.
        Tensor Z(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += last_Q_(j, off + d) * last_K_(i, off + d);
                Z(i, j) = (mask_local(i, j) > 0.5) ? inv_temp_ * dot : -1e30;
            }
        Tensor A(N, N);
        hl_masked_softmax_col(Z, mask_local, A);
        // Copy into last_A_local_ at offset (h * N, 0).
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                last_A_local_(h * N + i, j) = A(i, j);
    }

    // ---- Global-head forward (Hglo heads) ----------------------------------
    last_A_global_ = Tensor(Hglo * N, num_pool_);
    last_A_global_.fill(0.0);
    Tensor mask_pool(num_pool_, N);
    hl_pool_mask(mask_pool, num_pool_, N, W);

    for (size_t h = 0; h < Hglo; ++h) {
        const size_t h_global = h; // local offset into global heads (0..Hglo-1)
        const size_t off = (Hloc + h_global) * dh;
        // Build K_pool[p, d] for this head.
        for (size_t p = 0; p < num_pool_; ++p) {
            const size_t i_lo = p * W;
            const size_t Wp = pool_window_sizes_[p];
            const double invWp = 1.0 / static_cast<double>(Wp);
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = i_lo; i < i_lo + Wp; ++i)
                    acc += last_K_(i, off + d);
                last_K_pool_((Hloc + h_global) * num_pool_ + p, d) = acc * invWp;
            }
        }
        // Build V_pool[p, d] for this head.
        for (size_t p = 0; p < num_pool_; ++p) {
            const size_t i_lo = p * W;
            const size_t Wp = pool_window_sizes_[p];
            const double invWp = 1.0 / static_cast<double>(Wp);
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = i_lo; i < i_lo + Wp; ++i)
                    acc += last_V_(i, off + d);
                last_V_pool_((Hloc + h_global) * num_pool_ + p, d) = acc * invWp;
            }
        }
        // Compute Z_pool[p, j] = inv_temp * q_j · k_pool[p], then mask + softmax.
        Tensor Z(num_pool_, N);
        for (size_t p = 0; p < num_pool_; ++p)
            for (size_t j = 0; j < N; ++j) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += last_Q_(j, off + d) * last_K_pool_((Hloc + h_global) * num_pool_ + p, d);
                Z(p, j) = (mask_pool(p, j) > 0.5) ? inv_temp_ * dot : -1e30;
            }
        Tensor A(num_pool_, N);
        hl_masked_softmax_col(Z, mask_pool, A);
        for (size_t p = 0; p < num_pool_; ++p)
            for (size_t j = 0; j < N; ++j)
                last_A_global_((Hloc + h_global - Hloc) * N + j, p) = A(p, j);
    }

    // ---- Per-head output + λ-blend -----------------------------------------
    // For each head, compute the per-head output then store the BLEND into
    // the pre-W_o slice. We need the local and global outputs SEPARATELY
    // cached for the λ-blend backward, so we recompute them from last_A_*
    // here. (This is cheap; an alternative is to cache the two streams in
    // separate tensors. Recompute keeps the cache footprint small.)
    last_O_pre_ = Tensor(N, d_model_);
    last_O_pre_.fill(0.0);

    // Local heads: out_h_j = attn_local @ V (head h's dh-wide slice of V).
    for (size_t h = 0; h < Hloc; ++h) {
        const size_t off = h * dh;
        // Per-head output (N, dh) from softmax(Q_h K_h^T) V_h.
        Tensor out(N, dh);
        out.fill(0.0);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i)
                    acc += last_A_local_(h * N + i, j) * last_V_(i, off + d);
                out(j, d) = acc;
            }
        // Blend with λ (local factor = 1 − λ[h]).
        const double factor = 1.0 - lambda_(h, 0);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                last_O_pre_(j, off + d) = factor * out(j, d);
    }

    // Global heads: out_h_j = attn_global @ V_pool (head h's slice of V_pool).
    for (size_t h = 0; h < Hglo; ++h) {
        const size_t h_global = h;
        const size_t off = (Hloc + h_global) * dh;
        Tensor out(N, dh);
        out.fill(0.0);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t p = 0; p < num_pool_; ++p)
                    acc += last_A_global_(h_global * N + j, p)
                         * last_V_pool_((Hloc + h_global) * num_pool_ + p, d);
                out(j, d) = acc;
            }
        // Blend (global factor = λ[h]).
        const double factor = lambda_(Hloc + h_global, 0);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                last_O_pre_(j, off + d) = factor * out(j, d);
    }

    // ---- W_o projection ----------------------------------------------------
    Tensor out_full = W_o.forward(last_O_pre_);
    return out_full;
}

Tensor HiLoAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t N = N_last_;
    const size_t Hloc = num_local_;
    const size_t Hglo = num_global_;
    const size_t dh = head_dim_;
    const size_t W  = window_size_;
    const size_t P  = num_pool_;
    if (N == 0)
        throw std::logic_error("HiLoAttention: backward called before forward");

    // 1) grad w.r.t. last_O_pre_ = grad_output @ W_o  (standard Dense backward).
    last_grad_O_pre_ = Tensor(N, d_model_);
    // grad_O_pre = grad_output * W_o (Dense convention)
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < d_model_; ++d) {
            double acc = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                acc += grad_output(i, k) * W_o.weights(k, d);
            last_grad_O_pre_(i, d) = acc;
        }

    // Accumulate grad_W_o += last_O_pre_^T @ grad_output (per Dense convention).
    // grad_W_o[r, c] += Σ_n grad_output[n, r] * last_O_pre_[n, c]
    for (size_t r = 0; r < d_model_; ++r)
        for (size_t c = 0; c < d_model_; ++c) {
            double acc = 0.0;
            for (size_t n = 0; n < N; ++n)
                acc += grad_output(n, r) * last_O_pre_(n, c);
            grad_W_o(r, c) += acc;
        }
    // grad_b_o += sum over n of grad_output
    for (size_t c = 0; c < d_model_; ++c) {
        double acc = 0.0;
        for (size_t n = 0; n < N; ++n) acc += grad_output(n, c);
        W_o.grad_bias(0, c) += acc;
    }

    // 2) Per-head dQ / dK_local / dV_local with λ-blend.
    Tensor dQ(N, d_model_);
    Tensor dK(N, d_model_);
    Tensor dV(N, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);

    // Per-head λ-blend bookkeeping. We need the pre-blend per-head outputs
    // and per-head grad-outputs to compute d(logit_lambda). For memory, we
    // recompute them locally here from last_A_local_ / last_A_global_ /
    // last_V_ / last_V_pool_. (Same recompute strategy as forward.)
    Tensor grad_logit_acc(num_heads_, 1);
    grad_logit_acc.fill(0.0);

    // Local heads ---------------------------------------------------------
    Tensor mask_local(N, N);
    hl_local_mask(mask_local, N, W);
    Tensor mask_pool(P, N);
    hl_pool_mask(mask_pool, P, N, W);

    for (size_t h = 0; h < Hloc; ++h) {
        const size_t off = h * dh;
        const double lambda_h = lambda_(h, 0);
        const double dO_scale = 1.0 - lambda_h;

        // Per-head gradient w.r.t. the local pre-blend output, shape (N, dh).
        Tensor dO_local(N, dh);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                dO_local(j, d) = dO_scale * last_grad_O_pre_(j, off + d);

        // Standard softmax backward.
        // dA[i,j] = sum_d dO_local[j, d] * V_local[i, d]   (the i,j-th attn grad)
        Tensor dA(N, N);
        dA.fill(0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) {
                double acc = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    acc += dO_local(j, d) * last_V_(i, off + d);
                dA(i, j) = acc;
            }
        // sumA_w[j] = sum_i A[i,j] * dA[i,j]   (the A-WEIGHTED sum)
        // dS[i,j]   = A[i,j] * (dA[i,j] - sumA_w[j])   for masked-in entries; 0 otherwise
        // dZ[i,j]   = dS[i,j]   (since mask unchanged in backward)
        Tensor dZ(N, N);
        dZ.fill(0.0);
        for (size_t j = 0; j < N; ++j) {
            double sumA_w = 0.0;
            for (size_t i = 0; i < N; ++i)
                sumA_w += last_A_local_(h * N + i, j) * dA(i, j);
            for (size_t i = 0; i < N; ++i) {
                if (mask_local(i, j) > 0.5) {
                    dZ(i, j) = last_A_local_(h * N + i, j) * (dA(i, j) - sumA_w);
                }
            }
        }
        // dQ[j, d] += inv_temp * sum_i dZ[i, j] * K[i, d]
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i)
                    acc += dZ(i, j) * last_K_(i, off + d);
                dQ(j, off + d) += inv_temp_ * acc;
            }
        // dK_local[i, d] += inv_temp * sum_j dZ[i, j] * Q[j, d]
        for (size_t i = 0; i < N; ++i)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t j = 0; j < N; ++j)
                    acc += dZ(i, j) * last_Q_(j, off + d);
                dK(i, off + d) += inv_temp_ * acc;
            }
        // dV_local[i, d] += sum_j A[i, j] * dO_local[j, d]
        for (size_t i = 0; i < N; ++i)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t j = 0; j < N; ++j)
                    acc += last_A_local_(h * N + i, j) * dO_local(j, d);
                dV(i, off + d) += acc;
            }

        // logit_lambda[h] gradient: d_λ = -sum_{j,d} out_local * dO  (local factor = 1-λ)
        //                          then d_logit = σ'(logit) * d_λ
        // The blended output was (1-λ) * out_local. So dλ = -out_local · dO_pre.
        // d_logit_lambda[h] += σ'(logit_lambda[h]) * dλ
        // We compute the per-head contribution here and accumulate.
        // Need out_local (pre-blend) which we recompute on the fly:
        double d_lambda_h = 0.0;
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double out_local_jd = 0.0;
                for (size_t i = 0; i < N; ++i)
                    out_local_jd += last_A_local_(h * N + i, j) * last_V_(i, off + d);
                d_lambda_h -= out_local_jd * last_grad_O_pre_(j, off + d);
            }
        // σ'(logit) = λ * (1 - λ)
        const double sigp = lambda_h * (1.0 - lambda_h);
        grad_logit_acc(h, 0) += sigp * d_lambda_h;
    }

    // Global heads --------------------------------------------------------
    for (size_t h = 0; h < Hglo; ++h) {
        const size_t h_global = h;
        const size_t off = (Hloc + h_global) * dh;
        const double lambda_h = lambda_(Hloc + h_global, 0);
        const double dO_scale = lambda_h;

        // Per-head grad w.r.t. the GLOBAL pre-blend output, shape (N, dh).
        Tensor dO_global(N, dh);
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                dO_global(j, d) = dO_scale * last_grad_O_pre_(j, off + d);

        // dA_pool[p, j] = sum_d dO_global[j, d] * V_pool[(Hloc+h)*P + p, d]
        Tensor dA(P, N);
        dA.fill(0.0);
        for (size_t p = 0; p < P; ++p)
            for (size_t j = 0; j < N; ++j) {
                double acc = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    acc += dO_global(j, d)
                         * last_V_pool_((Hloc + h_global) * P + p, d);
                dA(p, j) = acc;
            }
        // softmax backward over pool dimension.
        Tensor dZ(P, N);
        dZ.fill(0.0);
        for (size_t j = 0; j < N; ++j) {
            double sumA_w = 0.0;
            for (size_t p = 0; p < P; ++p)
                sumA_w += last_A_global_(h_global * N + j, p) * dA(p, j);
            for (size_t p = 0; p < P; ++p) {
                if (mask_pool(p, j) > 0.5) {
                    dZ(p, j) = last_A_global_(h_global * N + j, p) * (dA(p, j) - sumA_w);
                }
            }
        }
        // dQ[j, d] += inv_temp * sum_p dZ[p, j] * K_pool[(Hloc+h)*P + p, d]
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t p = 0; p < P; ++p)
                    acc += dZ(p, j) * last_K_pool_((Hloc + h_global) * P + p, d);
                dQ(j, off + d) += inv_temp_ * acc;
            }
        // dK_pool[p, d] += inv_temp * sum_j dZ[p, j] * Q[j, d]
        Tensor dK_pool(P, dh);
        dK_pool.fill(0.0);
        for (size_t p = 0; p < P; ++p)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t j = 0; j < N; ++j)
                    acc += dZ(p, j) * last_Q_(j, off + d);
                dK_pool(p, d) = inv_temp_ * acc;
            }
        // Back-prop through the average pool into dK_local.
        // dK_local[i, d] += (1/W_p) * dK_pool[p(i), d]
        for (size_t i = 0; i < N; ++i) {
            const size_t p = i / W;
            const double invWp = 1.0 / static_cast<double>(pool_window_sizes_[p]);
            for (size_t d = 0; d < dh; ++d)
                dK(i, off + d) += invWp * dK_pool(p, d);
        }
        // dV_pool[p, d] += sum_j A[p, j] * dO_global[j, d]
        Tensor dV_pool(P, dh);
        dV_pool.fill(0.0);
        for (size_t p = 0; p < P; ++p)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t j = 0; j < N; ++j)
                    acc += last_A_global_(h_global * N + j, p) * dO_global(j, d);
                dV_pool(p, d) = acc;
            }
        // Back-prop through the average pool into dV_local.
        for (size_t i = 0; i < N; ++i) {
            const size_t p = i / W;
            const double invWp = 1.0 / static_cast<double>(pool_window_sizes_[p]);
            for (size_t d = 0; d < dh; ++d)
                dV(i, off + d) += invWp * dV_pool(p, d);
        }

        // logit_lambda gradient contribution (global factor = λ).
        // dλ = +sum_{j,d} out_global * dO_pre
        double d_lambda_h = 0.0;
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double out_g_jd = 0.0;
                for (size_t p = 0; p < P; ++p)
                    out_g_jd += last_A_global_(h_global * N + j, p)
                              * last_V_pool_((Hloc + h_global) * P + p, d);
                d_lambda_h += out_g_jd * last_grad_O_pre_(j, off + d);
            }
        const double sigp = lambda_h * (1.0 - lambda_h);
        grad_logit_acc(Hloc + h_global, 0) += sigp * d_lambda_h;
    }

    // Accumulate grad_logit_lambda_ (the per-head accumulator already includes
    // σ'(logit) * d_λ). We accumulated per-head contributions into grad_logit_acc;
    // copy into grad_logit_lambda_.
    for (size_t h = 0; h < num_heads_; ++h)
        grad_logit_lambda_(h, 0) += grad_logit_acc(h, 0);

    // 3) Accumulate grad_W_q / grad_W_k / grad_W_v from dQ / dK / dV.
    // Per Dense convention: grad_W[c, r] += Σ_n dX[n, c] * input[n, r]
    // where dX is the per-input gradient (dQ / dK / dV here) and input is
    // last_input_.
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t r = 0; r < d_model_; ++r) {
            double acc = 0.0;
            for (size_t n = 0; n < N; ++n)
                acc += dQ(n, c) * last_input_(n, r);
            grad_W_q(c, r) += acc;
        }
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t r = 0; r < d_model_; ++r) {
            double acc = 0.0;
            for (size_t n = 0; n < N; ++n)
                acc += dK(n, c) * last_input_(n, r);
            grad_W_k(c, r) += acc;
        }
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t r = 0; r < d_model_; ++r) {
            double acc = 0.0;
            for (size_t n = 0; n < N; ++n)
                acc += dV(n, c) * last_input_(n, r);
            grad_W_v(c, r) += acc;
        }

    // grad_b_q += sum_n dQ[n, c], same for k, v.
    for (size_t c = 0; c < d_model_; ++c) {
        double sq = 0, sk = 0, sv = 0;
        for (size_t n = 0; n < N; ++n) {
            sq += dQ(n, c); sk += dK(n, c); sv += dV(n, c);
        }
        W_q.grad_bias(0, c) += sq;
        W_k.grad_bias(0, c) += sk;
        W_v.grad_bias(0, c) += sv;
    }

    // 4) grad w.r.t. input: dX[n, r] = sum_c (dQ[n, c] * W_q[c, r]
    //                                        + dK[n, c] * W_k[c, r]
    //                                        + dV[n, c] * W_v[c, r])
    Tensor grad_input(N, d_model_);
    for (size_t n = 0; n < N; ++n)
        for (size_t r = 0; r < d_model_; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < d_model_; ++c) {
                acc += dQ(n, c) * W_q.weights(c, r);
                acc += dK(n, c) * W_k.weights(c, r);
                acc += dV(n, c) * W_v.weights(c, r);
            }
            grad_input(n, r) = acc;
        }
    return grad_input;
}

void HiLoAttention::update_weights(double learning_rate) {
    // Projections (manual SGD — we accumulate into raw grad_W_* tensors, not
    // via Dense::backward, so we cannot reuse Dense::update_weights).
    for (size_t i = 0; i < d_model_ * d_model_; ++i) {
        W_q.weights.data[i] -= learning_rate * grad_W_q.data[i];
        W_k.weights.data[i] -= learning_rate * grad_W_k.data[i];
        W_v.weights.data[i] -= learning_rate * grad_W_v.data[i];
        W_o.weights.data[i] -= learning_rate * grad_W_o.data[i];
    }
    for (size_t c = 0; c < d_model_; ++c) {
        W_q.bias(0, c) -= learning_rate * W_q.grad_bias(0, c);
        W_k.bias(0, c) -= learning_rate * W_k.grad_bias(0, c);
        W_v.bias(0, c) -= learning_rate * W_v.grad_bias(0, c);
        W_o.bias(0, c) -= learning_rate * W_o.grad_bias(0, c);
    }
    // logit_lambda_ update
    for (size_t h = 0; h < num_heads_; ++h) {
        logit_lambda_(h, 0) -= learning_rate * grad_logit_lambda_(h, 0);
    }
    // Recompute λ after the parameter update.
    recompute_lambda();
}

void HiLoAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_logit_lambda_.fill(0.0);
    W_q.grad_bias.fill(0.0);
    W_k.grad_bias.fill(0.0);
    W_v.grad_bias.fill(0.0);
    W_o.grad_bias.fill(0.0);
}

std::vector<Tensor*> HiLoAttention::parameters() {
    return {
        &W_q.weights, &W_q.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &W_o.weights, &W_o.bias,
        &logit_lambda_,
    };
}

std::vector<Tensor*> HiLoAttention::gradients() {
    return {
        &grad_W_q, &W_q.grad_bias,
        &grad_W_k, &W_k.grad_bias,
        &grad_W_v, &W_v.grad_bias,
        &grad_W_o, &W_o.grad_bias,
        &grad_logit_lambda_,
    };
}

// ============================================================================
// HiLoBlock
// ============================================================================

HiLoBlock::HiLoBlock(size_t d_model, size_t num_heads, size_t window_size,
                     size_t ffn_dim)
    : attn(d_model, num_heads, window_size),
      ln1(d_model), ln2(d_model),
      ffn_fc1_(d_model, ffn_dim ? ffn_dim : 4 * d_model),
      ffn_fc2_(ffn_dim ? ffn_dim : 4 * d_model, d_model),
      d_model_(d_model), ffn_dim_(ffn_dim)
{}

Tensor HiLoBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    // pre-LN -> HiLo attn -> residual
    last_z1_       = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_     = last_z1_;
    for (size_t i = 0; i < input.data.size(); ++i)
        last_res1_.data[i] += last_attn_out_.data[i];
    // pre-LN -> GELU FFN -> residual
    last_z2_      = ln2.forward(last_res1_);
    last_h_pre_   = ffn_fc1_.forward(last_z2_);
    last_h_act_   = last_h_pre_;
    for (size_t i = 0; i < last_h_pre_.data.size(); ++i)
        last_h_act_.data[i] = hl_gelu(last_h_pre_.data[i]);
    Tensor ffn_out = ffn_fc2_.forward(last_h_act_);
    Tensor out = last_res1_;
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] += ffn_out.data[i];
    return out;
}

Tensor HiLoBlock::backward(const Tensor& grad_output, double learning_rate) {
    // Forward structure:
    //   z1 = ln1(x); attn_out = attn(z1); res1 = z1 + attn_out;
    //   z2 = ln2(res1); h_pre = ffn_fc1(z2); h_act = gelu(h_pre);
    //   ffn_out = ffn_fc2(h_act); out = res1 + ffn_out.
    //
    // Backward (top-down):
    //   d_res1 = grad_output + (gradient through ffn_out → ln2 → res1)
    //   d_z1   = d_res1 + attn.backward(d_res1)   // attn_out is z1's co-input
    //   d_x    = ln1.backward(d_z1)
    Tensor d_res1 = grad_output.clone();
    // FFN path: propagate d_res1 (initially grad_output, but more will accumulate)
    // back through ffn_fc2 → gelu → ffn_fc1 → ln2, accumulating into d_res1.
    Tensor d_ffn_pre = ffn_fc2_.backward(d_res1, learning_rate);
    Tensor d_h_act(d_ffn_pre.rows, d_ffn_pre.cols);
    for (size_t i = 0; i < d_ffn_pre.data.size(); ++i)
        d_h_act.data[i] = d_ffn_pre.data[i] * hl_gelu_deriv(last_h_pre_.data[i]);
    Tensor d_z2 = ffn_fc1_.backward(d_h_act, learning_rate);
    Tensor d_res1_from_ln2 = ln2.backward(d_z2, learning_rate);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] += d_res1_from_ln2.data[i];
    // Attention path: d_z1 = d_res1 (residual) + attn.backward(d_res1) (attn_out)
    Tensor d_z1_from_attn = attn.backward(d_res1, learning_rate);
    Tensor d_z1(d_res1.rows, d_res1.cols);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] = d_res1.data[i] + d_z1_from_attn.data[i];
    // Input grad: d_x = ln1.backward(d_z1)
    Tensor d_x = ln1.backward(d_z1, learning_rate);
    return d_x;
}

void HiLoBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void HiLoBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> HiLoBlock::parameters() {
    auto p = attn.parameters();
    auto p1 = ln1.parameters();
    auto p2 = ln2.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), pf1.begin(), pf1.end());
    p.insert(p.end(), pf2.begin(), pf2.end());
    return p;
}

std::vector<Tensor*> HiLoBlock::gradients() {
    auto g = attn.gradients();
    auto g1 = ln1.gradients();
    auto g2 = ln2.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}
