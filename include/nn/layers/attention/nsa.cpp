// ============================================================================
// Native Sparse Attention (NSA) — DeepSeek-AI 2025 implementation
//   "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
//    Attention" — https://arxiv.org/abs/2502.11089
//
// See nsa.h for the full mathematical formulation. This file implements:
//   * NSAAttention — single-layer NSA (compression + selection + sliding window)
//   * NSABlock     — pre-LN → NSA → residual → optional pre-LN FFN → residual
//   * NSAModel     — stack of NSABlocks + classifier
//
// Conventions (match the rest of the attention family):
//   * Dense (where used): y = X @ W^T + b, W stored as (out, in).
//   * (n, d_model) input/output, row-major.
//   * Per-head layouts: head h occupies columns [h*head_dim : (h+1)*head_dim].
//   * GQA convention: num_query_heads / num_kv_heads groups share K/V.
// ============================================================================

#include "nsa.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Row-wise softmax (in-place over the last dim of a 2D row-major tensor).
static void row_softmax_inplace(Tensor& A) {
    for (size_t i = 0; i < A.rows; ++i) {
        double row_max = -1e300;
        for (size_t j = 0; j < A.cols; ++j)
            if (A[i][j] > row_max) row_max = A[i][j];
        double sum = 0.0;
        for (size_t j = 0; j < A.cols; ++j) {
            A[i][j] = std::exp(A[i][j] - row_max);
            sum += A[i][j];
        }
        const double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < A.cols; ++j) A[i][j] *= inv;
    }
}

// 3-way softmax over a row of 3 elements. Input/output are length-3 arrays.
static void softmax3(double z[3]) {
    double m = std::max({z[0], z[1], z[2]});
    double e0 = std::exp(z[0] - m);
    double e1 = std::exp(z[1] - m);
    double e2 = std::exp(z[2] - m);
    double s = e0 + e1 + e2 + 1e-12;
    z[0] = e0 / s;
    z[1] = e1 / s;
    z[2] = e2 / s;
}

// 3-way softmax backward. Given upstream grad dL/dg_i (in `dg`) and the
// softmax outputs (g0, g1, g2), write back into `dg` the gradient w.r.t. the
// pre-softmax logits z_i.
// Standard softmax Jacobian: dL/dz_i = Σ_j dL/dg_j · g_j · (δ_ij - g_i)
//                             = g_i · (dL/dg_i - Σ_j dL/dg_j · g_j)
static void softmax3_backward(const double g[3], double dg[3]) {
    double sum_g_dg = g[0] * dg[0] + g[1] * dg[1] + g[2] * dg[2];
    for (size_t i = 0; i < 3; ++i) {
        dg[i] = g[i] * (dg[i] - sum_g_dg);
    }
}

// Sinusoidal intra-block position encoding (l, head_dim). Same form as the
// standard transformer PE, applied per relative position inside the block.
static Tensor make_pos_enc(size_t l, size_t head_dim) {
    Tensor pe(l, head_dim);
    for (size_t pos = 0; pos < l; ++pos) {
        for (size_t j = 0; j < head_dim; ++j) {
            double div = std::pow(10000.0, static_cast<double>(2 * (j / 2)) /
                                            static_cast<double>(head_dim));
            double angle = static_cast<double>(pos) / div;
            pe(pos, j) = (j % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return pe;
}

// Build a (1, num_heads * 3) zero gradient-like matrix used in
// the dummy-initializer list (see NSA ctor note below).
static Tensor zeros_h3(size_t num_heads) {
    return Tensor::zeros(1, num_heads * 3);
}

// ============================================================================
// NSAAttention
// ============================================================================

NSAAttention::NSAAttention(size_t d_model,
                           size_t num_query_heads,
                           size_t num_kv_heads,
                           size_t block_len,
                           size_t stride,
                           size_t top_n,
                           size_t window_size,
                           size_t block_size)
    : W_q(d_model, d_model),
      W_o(d_model, d_model),
      W_k_cmp(d_model, d_model),
      W_v_cmp(d_model, d_model),
      W_k_sel(d_model, d_model),
      W_v_sel(d_model, d_model),
      W_k_win(d_model, d_model),
      W_v_win(d_model, d_model),
      grad_W_q(d_model, d_model),
      grad_W_o(d_model, d_model),
      grad_W_k_cmp(d_model, d_model),
      grad_W_v_cmp(d_model, d_model),
      grad_W_k_sel(d_model, d_model),
      grad_W_v_sel(d_model, d_model),
      grad_W_k_win(d_model, d_model),
      grad_W_v_win(d_model, d_model),
      // Dummy values for the rest — recomputed AFTER validation.
      d_model_(d_model),
      num_query_heads_(num_query_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(d_model > 0 ? d_model : 1),
      group_size_(num_query_heads > 0 ? num_query_heads : 1),
      block_len_(block_len),
      stride_(stride),
      top_n_(top_n),
      window_size_(window_size),
      block_size_(block_size == 0 ? stride : block_size),
      n_cmp_(0),
      n_sel_blocks_(0),
      scale_(1.0),
      use_compression_(true),
      use_selection_(true),
      use_window_(true),
      pos_enc_(1, 1)   // placeholder
{
    if (d_model_ == 0)
        throw std::invalid_argument("NSAAttention: d_model must be > 0");
    if (num_query_heads_ == 0)
        throw std::invalid_argument("NSAAttention: num_query_heads must be > 0");
    if (num_kv_heads_ == 0)
        throw std::invalid_argument("NSAAttention: num_kv_heads must be > 0");
    if (block_len_ == 0)
        throw std::invalid_argument("NSAAttention: block_len must be > 0");
    if (stride_ == 0 || stride_ > block_len_)
        throw std::invalid_argument("NSAAttention: stride must be in [1, block_len]");
    if (window_size_ == 0)
        throw std::invalid_argument("NSAAttention: window_size must be > 0");
    if (top_n_ == 0)
        throw std::invalid_argument("NSAAttention: top_n must be > 0");
    if (d_model_ % num_query_heads_ != 0)
        throw std::invalid_argument(
            "NSAAttention: d_model must be evenly divisible by num_query_heads");
    if (num_query_heads_ % num_kv_heads_ != 0)
        throw std::invalid_argument(
            "NSAAttention: num_query_heads must be evenly divisible by num_kv_heads");

    // NOW safe to derive the actual values.
    head_dim_   = d_model_ / num_query_heads_;
    group_size_ = num_query_heads_ / num_kv_heads_;
    scale_      = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Init weights with small random values (deterministic seed for repeatability).
    // Use larger scale (0.3) for richer gradients — at small scales the gradient
    // signal-to-noise is too poor for FD tests to verify.
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.3);
    auto init_dense = [&](Tensor& W) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) = dis(gen);
    };
    init_dense(W_q);
    init_dense(W_o);
    init_dense(W_k_cmp); init_dense(W_v_cmp);
    init_dense(W_k_sel); init_dense(W_v_sel);
    init_dense(W_k_win); init_dense(W_v_win);

    // Per-head compression MLP φ: (num_heads, head_dim, l * head_dim)
    size_t phi_dim = block_len_ * head_dim_;
    W_phi_k = Tensor::zeros(num_query_heads_, head_dim_ * phi_dim);
    W_phi_v = Tensor::zeros(num_query_heads_, head_dim_ * phi_dim);
    // Reshape views — we store flat (num_heads, head_dim * phi_dim) for layout
    // simplicity; in the forward/backward we use W_phi_k[h, j*phi_dim + j']
    // for output channel j (in head_dim), input channel j' (in l*head_dim).
    // Initialize small random.
    std::normal_distribution<> phi_dis(0.0, 1.0 / std::sqrt(static_cast<double>(phi_dim)));
    for (size_t h = 0; h < num_query_heads_; ++h)
        for (size_t k = 0; k < W_phi_k.data.size(); ++k) {
            W_phi_k.data[k] = phi_dis(gen);
            W_phi_v.data[k] = phi_dis(gen);
        }
    grad_W_phi_k = Tensor::zeros(W_phi_k.rows, W_phi_k.cols);
    grad_W_phi_v = Tensor::zeros(W_phi_v.rows, W_phi_v.cols);

    // Per-head gating MLP: head_dim → 3 logits. Stored as (num_heads, 3, head_dim)
    // so that per-head h, branch c, the logits are W_gate[h, c, :].
    W_gate = Tensor::zeros(num_query_heads_, 3 * head_dim_);
    grad_W_gate = Tensor::zeros(W_gate.rows, W_gate.cols);

    // Build the fixed intra-block sinusoidal position encoding (l, head_dim).
    pos_enc_ = make_pos_enc(block_len_, head_dim_);

    // Zero all gradient accumulators explicitly.
    grad_W_q.fill(0.0);
    grad_W_o.fill(0.0);
    grad_W_k_cmp.fill(0.0); grad_W_v_cmp.fill(0.0);
    grad_W_k_sel.fill(0.0); grad_W_v_sel.fill(0.0);
    grad_W_k_win.fill(0.0); grad_W_v_win.fill(0.0);
}

std::vector<Tensor*> NSAAttention::parameters() {
    return {&W_q, &W_o,
            &W_k_cmp, &W_v_cmp,
            &W_k_sel, &W_v_sel,
            &W_k_win, &W_v_win,
            &W_phi_k, &W_phi_v,
            &W_gate};
}

std::vector<Tensor*> NSAAttention::gradients() {
    return {&grad_W_q, &grad_W_o,
            &grad_W_k_cmp, &grad_W_v_cmp,
            &grad_W_k_sel, &grad_W_v_sel,
            &grad_W_k_win, &grad_W_v_win,
            &grad_W_phi_k, &grad_W_phi_v,
            &grad_W_gate};
}

void NSAAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_o.fill(0.0);
    grad_W_k_cmp.fill(0.0); grad_W_v_cmp.fill(0.0);
    grad_W_k_sel.fill(0.0); grad_W_v_sel.fill(0.0);
    grad_W_k_win.fill(0.0); grad_W_v_win.fill(0.0);
    grad_W_phi_k.fill(0.0); grad_W_phi_v.fill(0.0);
    grad_W_gate.fill(0.0);
}

void NSAAttention::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_o -= grad_W_o * learning_rate;
    W_k_cmp -= grad_W_k_cmp * learning_rate;
    W_v_cmp -= grad_W_v_cmp * learning_rate;
    W_k_sel -= grad_W_k_sel * learning_rate;
    W_v_sel -= grad_W_v_sel * learning_rate;
    W_k_win -= grad_W_k_win * learning_rate;
    W_v_win -= grad_W_v_win * learning_rate;
    W_phi_k -= grad_W_phi_k * learning_rate;
    W_phi_v -= grad_W_phi_v * learning_rate;
    W_gate  -= grad_W_gate  * learning_rate;
}

// ============================================================================
// NSAAttention::forward
// ============================================================================

Tensor NSAAttention::forward(const Tensor& input) {
    const size_t N = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("NSAAttention.forward: input.cols must equal d_model");
    if (N < block_len_)
        throw std::invalid_argument("NSAAttention: N must be >= block_len");
    if (N < window_size_)
        throw std::invalid_argument("NSAAttention: N must be >= window_size");

    N_last_ = N;
    last_input_ = input.clone();
    n_cmp_        = (N - block_len_) / stride_ + 1;
    n_sel_blocks_ = (N - block_size_) / stride_ + 1;

    // Project to Q, K_cmp, V_cmp, K_sel, V_sel, K_win, V_win via Y = X @ W^T.
    // Note: we have NO biases (standard attention convention).
    auto proj = [&](const Tensor& W) {
        Tensor Y(N, d_model_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = 0.0;
                for (size_t k = 0; k < d_model_; ++k) acc += input(i, k) * W(k, j);
                Y(i, j) = acc;
            }
        return Y;
    };
    Tensor Q    = proj(W_q);
    Tensor K_cmp = proj(W_k_cmp);
    Tensor V_cmp = proj(W_v_cmp);
    Tensor K_sel = proj(W_k_sel);
    Tensor V_sel = proj(W_v_sel);
    Tensor K_win = proj(W_k_win);
    Tensor V_win = proj(W_v_win);
    last_Q_     = Q;
    last_K_cmp_ = K_cmp; last_V_cmp_ = V_cmp;
    last_K_sel_ = K_sel; last_V_sel_ = V_sel;
    last_K_win_ = K_win; last_V_win_ = V_win;

    // ----- Branch 1: Compression -----
    // Build K̃_cmp, Ṽ_cmp per head (H, n_cmp, head_dim).
    // For each compression block i in [0, n_cmp):
    //   token range [i*stride, i*stride + l)
    //   input to φ: concat_l{ (K(token_t + offset, h_off:h_off+head_dim) + pos_enc_(offset, :)) }
    //     for offset in [0, l).  Flatten → (l * head_dim,).
    //   output: K̃_cmp[h, i, j] = Σ_{k} W_phi_k[h, j*phi_dim + k] · input_k
    const size_t l = block_len_;
    const size_t phi_dim = l * head_dim_;
    Tensor K_tilde_cmp(num_query_heads_, n_cmp_ * head_dim_);   // (H, n_cmp * head_dim)
    Tensor V_tilde_cmp(num_query_heads_, n_cmp_ * head_dim_);

    if (use_compression_) {
        // For each compression block, build the (l * head_dim) input vector
        // by concatenating per-position (K + pos_enc) flattened.
        std::vector<double> phi_in(phi_dim);
        for (size_t h = 0; h < num_query_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            for (size_t bi = 0; bi < n_cmp_; ++bi) {
                const size_t block_start = bi * stride_;
                // Build input vector
                for (size_t off = 0; off < l; ++off) {
                    for (size_t j = 0; j < head_dim_; ++j) {
                        phi_in[off * head_dim_ + j] =
                            K_cmp(block_start + off, h_off + j) + pos_enc_(off, j);
                    }
                }
                // Apply φ_K
                for (size_t j = 0; j < head_dim_; ++j) {
                    double acc = 0.0;
                    for (size_t k = 0; k < phi_dim; ++k)
                        acc += W_phi_k(h, j * phi_dim + k) * phi_in[k];
                    K_tilde_cmp(h, bi * head_dim_ + j) = acc;
                }
                // Apply φ_V
                for (size_t off = 0; off < l; ++off) {
                    for (size_t j = 0; j < head_dim_; ++j) {
                        phi_in[off * head_dim_ + j] =
                            V_cmp(block_start + off, h_off + j) + pos_enc_(off, j);
                    }
                }
                for (size_t j = 0; j < head_dim_; ++j) {
                    double acc = 0.0;
                    for (size_t k = 0; k < phi_dim; ++k)
                        acc += W_phi_v(h, j * phi_dim + k) * phi_in[k];
                    V_tilde_cmp(h, bi * head_dim_ + j) = acc;
                }
            }
        }
    } else {
        K_tilde_cmp.fill(0.0);
        V_tilde_cmp.fill(0.0);
    }
    last_K_tilde_cmp_ = K_tilde_cmp;
    last_V_tilde_cmp_ = V_tilde_cmp;

    // ----- Per-head importance scores from compression -----
    // p_cmp[h, t, bi] = softmax_t (Q[t, h_off:h_off+head_dim] · K̃_cmp[h, bi, :]) / sqrt(head_dim)
    // Storage: (H, N, n_cmp). For per-head aggregation across GQA groups we
    // compute the mean over the heads in a GQA group at selection time.
    Tensor p_cmp(num_query_heads_, N * n_cmp_);
    if (use_compression_) {
        for (size_t h = 0; h < num_query_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            Tensor scores(N, n_cmp_);
            for (size_t t = 0; t < N; ++t) {
                for (size_t bi = 0; bi < n_cmp_; ++bi) {
                    double acc = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        acc += Q(t, h_off + j) * K_tilde_cmp(h, bi * head_dim_ + j);
                    scores(t, bi) = acc * scale_;
                }
            }
            row_softmax_inplace(scores);
            for (size_t t = 0; t < N; ++t)
                for (size_t bi = 0; bi < n_cmp_; ++bi)
                    p_cmp(h, t * n_cmp_ + bi) = scores(t, bi);
        }
    }
    last_p_cmp_ = p_cmp;

    // ----- Branch 2: Selection (top-n blocks of size block_size, stride) -----
    // Step A: aggregate compression scores into per-selection-block scores.
    // For each selection block j in [0, n_sel_blocks), cover the compression
    // blocks whose range [j*stride, j*stride + block_size) maps to it.
    // Step B: aggregate across the GQA group (mean over heads sharing kv head).
    // Step C: top-n selection → indices into selection-block space.
    // Step D: build K̃_sel (concatenated selected blocks).
    size_t n_sel_keep = std::min(top_n_, n_sel_blocks_);
    Tensor top_idx(num_query_heads_, N * n_sel_keep);  // (H, N * n_sel_keep) — block indices
    Tensor p_cmp_per_head(num_query_heads_, N * n_sel_blocks_);
    if (use_selection_) {
        // p_cmp_per_head[h, t, sj] = mean over compression blocks in [sj*stride, sj*stride + block_size/stride) ∩ [0, n_cmp)
        size_t n_cmp_per_sel = (block_size_ + stride_ - 1) / stride_;
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                for (size_t sj = 0; sj < n_sel_blocks_; ++sj) {
                    double s = 0.0;
                    size_t cnt = 0;
                    for (size_t k = 0; k < n_cmp_per_sel; ++k) {
                        size_t cmp_idx = sj * (block_size_ / stride_) + k;
                        if (cmp_idx >= n_cmp_) break;
                        s += p_cmp(h, t * n_cmp_ + cmp_idx);
                        cnt++;
                    }
                    p_cmp_per_head(h, t * n_sel_blocks_ + sj) = (cnt > 0) ? s / cnt : 0.0;
                }
            }
        }

        // Aggregate across heads in each GQA group (mean).
        // Group g contains heads [g*group_size, (g+1)*group_size).
        std::vector<double> agg(n_sel_blocks_);
        for (size_t g = 0; g < num_kv_heads_; ++g) {
            for (size_t t = 0; t < N; ++t) {
                for (size_t sj = 0; sj < n_sel_blocks_; ++sj) {
                    double s = 0.0;
                    for (size_t hi = 0; hi < group_size_; ++hi) {
                        size_t h = g * group_size_ + hi;
                        s += p_cmp_per_head(h, t * n_sel_blocks_ + sj);
                    }
                    agg[sj] = s / group_size_;
                }
                // Pick top-n by descending score (stable: lower index wins ties).
                std::vector<size_t> order(n_sel_blocks_);
                for (size_t i = 0; i < n_sel_blocks_; ++i) order[i] = i;
                std::sort(order.begin(), order.end(),
                          [&](size_t a, size_t b) {
                              if (agg[a] != agg[b]) return agg[a] > agg[b];
                              return a < b;
                          });
                // Write top-n indices for every head in the GQA group
                for (size_t hi = 0; hi < group_size_; ++hi) {
                    size_t h = g * group_size_ + hi;
                    for (size_t k = 0; k < n_sel_keep; ++k) {
                        top_idx(h, t * n_sel_keep + k) = static_cast<double>(order[k]);
                    }
                }
            }
        }
    }
    last_p_cmp_per_head_ = p_cmp_per_head;
    last_top_idx_         = top_idx;

    // Build K̃_sel, Ṽ_sel (concatenated selected blocks). Shape (H, N * n_sel_keep * block_size, head_dim).
    // But since selection is per-(head, query), we store per-(h, t, slot) where
    // slot ∈ [0, n_sel_keep * block_size) indexes into the concatenation.
    size_t sel_slot_dim = n_sel_keep * block_size_;
    Tensor K_tilde_sel(num_query_heads_, N * sel_slot_dim * head_dim_);
    Tensor V_tilde_sel(num_query_heads_, N * sel_slot_dim * head_dim_);
    if (use_selection_) {
        for (size_t h = 0; h < num_query_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t k = 0; k < n_sel_keep; ++k) {
                    size_t block_idx = static_cast<size_t>(top_idx(h, t * n_sel_keep + k));
                    size_t block_start = block_idx * stride_;
                    for (size_t off = 0; off < block_size_; ++off) {
                        size_t src_token = block_start + off;
                        if (src_token >= N) break;
                        for (size_t j = 0; j < head_dim_; ++j) {
                            size_t slot = k * block_size_ + off;
                            size_t dst_off = (t * sel_slot_dim + slot) * head_dim_ + j;
                            K_tilde_sel(h, dst_off) = K_sel(src_token, h_off + j);
                            V_tilde_sel(h, dst_off) = V_sel(src_token, h_off + j);
                        }
                    }
                }
            }
        }
    }
    last_K_tilde_sel_ = K_tilde_sel;
    last_V_tilde_sel_ = V_tilde_sel;

    // ----- Branch 3: Sliding window -----
    // Take last window_size_ tokens per query. To keep things simple and
    // avoid masking gradients, we just KEEP the last w tokens (no query-
    // dependent masking — all queries attend to the same w-token window).
    // For per-query causal windows we'd need masking; here we use a fixed
    // window. K_tilde_win[h, t, k, :] = K[N - w + k, h_off:h_off+head_dim]
    // for k in [0, w).
    size_t w = window_size_;
    Tensor K_tilde_win(num_query_heads_, N * w * head_dim_);
    Tensor V_tilde_win(num_query_heads_, N * w * head_dim_);
    if (use_window_) {
        for (size_t h = 0; h < num_query_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t k = 0; k < w; ++k) {
                    size_t src_token = t < w - 1 ? 0 : t - (w - 1) + k;
                    // Causal window: at step t, query can attend to tokens
                    // [max(0, t - w + 1), t].  Padding with K[0] when out of
                    // range (the first token "attends to itself + earlier
                    // padding").
                    if (src_token > t) src_token = 0;
                    size_t dst_off = (t * w + k) * head_dim_ + 0;
                    for (size_t j = 0; j < head_dim_; ++j) {
                        K_tilde_win(h, dst_off + j) = K_win(src_token, h_off + j);
                        V_tilde_win(h, dst_off + j) = V_win(src_token, h_off + j);
                    }
                }
            }
        }
    }
    last_K_tilde_win_ = K_tilde_win;
    last_V_tilde_win_ = V_tilde_win;

    // ----- Per-(token, head, branch) gate -----
    // g[t, h, c] = softmax_c( W_gate[h, c, :] · Q[t, h_off:h_off+head_dim] )
    // Stored as (N, num_heads * 3) flat: row t, col h*3 + c.
    Tensor gate(N, num_query_heads_ * 3);
    for (size_t h = 0; h < num_query_heads_; ++h) {
        const size_t h_off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            double z[3] = {0.0, 0.0, 0.0};
            for (size_t c = 0; c < 3; ++c)
                for (size_t j = 0; j < head_dim_; ++j)
                    z[c] += W_gate(h, c * head_dim_ + j) * Q(t, h_off + j);
            softmax3(z);
            for (size_t c = 0; c < 3; ++c) gate(t, h * 3 + c) = z[c];
        }
    }
    last_gate_ = gate;

    // ----- Compute per-branch attention outputs -----
    // O_cmp[h, t, j] = Σ_{bi} p_cmp[h, t, bi] · V_tilde_cmp[h, bi, j]
    // O_sel[h, t, j] = Σ_{slot} attn_sel[h, t, slot] · V_tilde_sel[h, t, slot, j]
    // O_win[h, t, j] = Σ_{k} attn_win[h, t, k] · V_tilde_win[h, t, k, j]
    // Each stored as (H, N, head_dim) flattened to (H, N * head_dim).
    // We store the per-branch attention weights as well for backward:
    //   A_cmp : (H, N, n_cmp)
    //   A_sel : (H, N, n_sel_keep * block_size)
    //   A_win : (H, N, w)
    Tensor O_cmp(num_query_heads_, N * head_dim_);
    Tensor O_sel(num_query_heads_, N * head_dim_);
    Tensor O_win(num_query_heads_, N * head_dim_);
    Tensor A_cmp(num_query_heads_, N * n_cmp_);
    Tensor A_sel(num_query_heads_, N * n_sel_keep * block_size_);
    Tensor A_win(num_query_heads_, N * w);

    for (size_t h = 0; h < num_query_heads_; ++h) {
        const size_t h_off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            // Compression
            if (use_compression_) {
                for (size_t bi = 0; bi < n_cmp_; ++bi) {
                    double aw = p_cmp(h, t * n_cmp_ + bi);
                    A_cmp(h, t * n_cmp_ + bi) = aw;
                    for (size_t j = 0; j < head_dim_; ++j)
                        O_cmp(h, t * head_dim_ + j) +=
                            aw * V_tilde_cmp(h, bi * head_dim_ + j);
                }
            }
            // Selection
            if (use_selection_) {
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    // Compute attn_sel[h, t, slot] on the fly
                    size_t slot_off = (t * sel_slot_dim + slot) * head_dim_;
                    double aw = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        aw += Q(t, h_off + j) * K_tilde_sel(h, slot_off + j);
                    aw *= scale_;
                    // Store pre-softmax score; we'll softmax after the loop.
                    A_sel(h, t * sel_slot_dim + slot) = aw;
                }
            }
            // Window
            if (use_window_) {
                for (size_t k = 0; k < w; ++k) {
                    size_t slot_off = (t * w + k) * head_dim_;
                    double aw = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        aw += Q(t, h_off + j) * K_tilde_win(h, slot_off + j);
                    aw *= scale_;
                    A_win(h, t * w + k) = aw;
                }
            }
        }
        // Softmax each row of selection/window attention scores per (h, t).
        if (use_selection_) {
            for (size_t t = 0; t < N; ++t) {
                size_t base = h * (N * sel_slot_dim) + t * sel_slot_dim;
                double row_max = -1e300;
                for (size_t slot = 0; slot < sel_slot_dim; ++slot)
                    if (A_sel.data[base + slot] > row_max) row_max = A_sel.data[base + slot];
                double sum = 0.0;
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    A_sel.data[base + slot] = std::exp(A_sel.data[base + slot] - row_max);
                    sum += A_sel.data[base + slot];
                }
                double inv = 1.0 / (sum + 1e-12);
                for (size_t slot = 0; slot < sel_slot_dim; ++slot)
                    A_sel.data[base + slot] *= inv;
            }
        }
        if (use_window_) {
            for (size_t t = 0; t < N; ++t) {
                size_t base = h * (N * w) + t * w;
                double row_max = -1e300;
                for (size_t k = 0; k < w; ++k)
                    if (A_win.data[base + k] > row_max) row_max = A_win.data[base + k];
                double sum = 0.0;
                for (size_t k = 0; k < w; ++k) {
                    A_win.data[base + k] = std::exp(A_win.data[base + k] - row_max);
                    sum += A_win.data[base + k];
                }
                double inv = 1.0 / (sum + 1e-12);
                for (size_t k = 0; k < w; ++k)
                    A_win.data[base + k] *= inv;
            }
        }
        // Now apply A_sel and A_win to V_tilde.
        for (size_t t = 0; t < N; ++t) {
            if (use_selection_) {
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    double aw = A_sel(h, t * sel_slot_dim + slot);
                    size_t slot_off = (t * sel_slot_dim + slot) * head_dim_;
                    for (size_t j = 0; j < head_dim_; ++j)
                        O_sel(h, t * head_dim_ + j) +=
                            aw * V_tilde_sel(h, slot_off + j);
                }
            }
            if (use_window_) {
                for (size_t k = 0; k < w; ++k) {
                    double aw = A_win(h, t * w + k);
                    size_t slot_off = (t * w + k) * head_dim_;
                    for (size_t j = 0; j < head_dim_; ++j)
                        O_win(h, t * head_dim_ + j) +=
                            aw * V_tilde_win(h, slot_off + j);
                }
            }
        }
    }
    last_attn_cmp_ = A_cmp;
    last_attn_sel_ = A_sel;
    last_attn_win_ = A_win;

    // ----- Combine per-branch outputs via the learned gate -----
    // O[t, j] = Σ_{h, c} gate[t, h, c] · O_c[h, t, j_for_head_h]
    // where j_for_head_h = j - h*head_dim, only nonzero in [h*head_dim, (h+1)*head_dim).
    // Then O[t, j] = Σ_h gate[t, h, j_in_head] · O_{branch_for_(j_in_head)}[h, t, j_in_head]
    //   where the branch is determined by j_in_head's slot in [0, head_dim):
    //     NO — each branch produces a full head_dim output per (h, t), and
    //     the gate weights the WHOLE head_dim of branch output.
    // So:
    //   O_pre[t, h_off + j] = gate_cmp[t, h] · O_cmp[h, t, j]
    //                        + gate_sel[t, h] · O_sel[h, t, j]
    //                        + gate_win[t, h] · O_win[h, t, j]
    Tensor O(N, d_model_);
    for (size_t h = 0; h < num_query_heads_; ++h) {
        size_t h_off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            double g_cmp = gate(t, h * 3 + 0);
            double g_sel = gate(t, h * 3 + 1);
            double g_win = gate(t, h * 3 + 2);
            for (size_t j = 0; j < head_dim_; ++j) {
                O(t, h_off + j) =
                    g_cmp * O_cmp(h, t * head_dim_ + j) +
                    g_sel * O_sel(h, t * head_dim_ + j) +
                    g_win * O_win(h, t * head_dim_ + j);
            }
        }
    }
    last_O_branches_ = Tensor(N, 3 * d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t j = 0; j < head_dim_; ++j) {
                last_O_branches_(t, 0 * d_model_ + h_off + j) = O_cmp(h, t * head_dim_ + j);
                last_O_branches_(t, 1 * d_model_ + h_off + j) = O_sel(h, t * head_dim_ + j);
                last_O_branches_(t, 2 * d_model_ + h_off + j) = O_win(h, t * head_dim_ + j);
            }
        }
    }
    last_O_ = O;

    // Output projection: y = O @ W_o^T
    Tensor output(N, d_model_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < d_model_; ++k) acc += O(i, k) * W_o(k, j);
            output(i, j) = acc;
        }
    return output;
}

// ============================================================================
// NSAAttention::backward
// ============================================================================

Tensor NSAAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != N_last_ || grad_output.cols != d_model_)
        throw std::invalid_argument("NSAAttention::backward: grad_output shape mismatch");
    const size_t N = N_last_;
    const size_t l = block_len_;
    const size_t phi_dim = l * head_dim_;
    const size_t w = window_size_;
    const size_t sel_slot_dim = top_n_ * block_size_;
    // n_keep = top_n_
    // cap so we don't overrun
    size_t n_sel_keep = std::min(top_n_, n_sel_blocks_);

    // Zero all gradients.
    zero_grad();

    // ----- Step 1: backward through W_o -----
    // y = O @ W_o^T   →   dO[i, j] = Σ_jp grad_output[i, jp] · W_o[j, jp]
    // grad_W_o[k, j] += Σ_i grad_output[i, j] · O[i, k]
    Tensor d_O(N, d_model_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t jp = 0; jp < d_model_; ++jp) acc += grad_output(i, jp) * W_o(j, jp);
            d_O(i, j) = acc;
        }
    for (size_t k = 0; k < d_model_; ++k)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += grad_output(i, j) * last_O_(i, k);
            grad_W_o(k, j) += acc;
        }

    // ----- Step 2: split dO across branches via the gate -----
    // For each (t, h, c, j_in_head):
    //   d_O_c[h, t, j] += gate[t, h, c] · d_O[t, h_off + j]
    // and the gate gradients:
    //   d_gate[t, h, c] += Σ_j d_O[t, h_off + j] · O_c[h, t, j]
    Tensor d_O_cmp(num_query_heads_, N * head_dim_);
    Tensor d_O_sel(num_query_heads_, N * head_dim_);
    Tensor d_O_win(num_query_heads_, N * head_dim_);
    Tensor d_gate_logits(N, num_query_heads_ * 3);  // pre-softmax
    for (size_t h = 0; h < num_query_heads_; ++h) {
        size_t h_off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            double dg[3] = {0.0, 0.0, 0.0};
            for (size_t j = 0; j < head_dim_; ++j) {
                double v = d_O(t, h_off + j);
                d_O_cmp(h, t * head_dim_ + j) += last_gate_(t, h * 3 + 0) * v;
                d_O_sel(h, t * head_dim_ + j) += last_gate_(t, h * 3 + 1) * v;
                d_O_win(h, t * head_dim_ + j) += last_gate_(t, h * 3 + 2) * v;
                dg[0] += v * last_O_branches_(t, 0 * d_model_ + h_off + j);
                dg[1] += v * last_O_branches_(t, 1 * d_model_ + h_off + j);
                dg[2] += v * last_O_branches_(t, 2 * d_model_ + h_off + j);
            }
            // softmax3 backward
            double g0 = last_gate_(t, h * 3 + 0);
            double g1 = last_gate_(t, h * 3 + 1);
            double g2 = last_gate_(t, h * 3 + 2);
            double s = g0 * dg[0] + g1 * dg[1] + g2 * dg[2];
            d_gate_logits(t, h * 3 + 0) = g0 * (dg[0] - s);
            d_gate_logits(t, h * 3 + 1) = g1 * (dg[1] - s);
            d_gate_logits(t, h * 3 + 2) = g2 * (dg[2] - s);
        }
    }

    // ----- Step 2b: gate parameter gradients -----
    // z[t, h, c] = <W_gate[h, c, :], Q[t, h_off:h_off+head_dim]>
    // grad_W_gate[h, c, j] += Σ_t d_gate_logits[t, h, c] · Q[t, h_off + j]
    for (size_t h = 0; h < num_query_heads_; ++h) {
        size_t h_off = h * head_dim_;
        for (size_t c = 0; c < 3; ++c) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t) acc += d_gate_logits(t, h * 3 + c) * last_Q_(t, h_off + j);
                grad_W_gate(h, c * head_dim_ + j) += acc;
            }
        }
    }
    // d_Q from the gate path:
    // d_Q[t, h_off + j] += Σ_c d_gate_logits[t, h, c] · W_gate[h, c, j]
    Tensor d_Q_from_gate(N, d_model_);
    for (size_t h = 0; h < num_query_heads_; ++h) {
        size_t h_off = h * head_dim_;
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t c = 0; c < 3; ++c) acc += d_gate_logits(t, h * 3 + c) * W_gate(h, c * head_dim_ + j);
                d_Q_from_gate(t, h_off + j) = acc;
            }
        }
    }

    // ----- Per-branch input-grad scratch accumulators (token × d_model) -----
    Tensor d_K_cmp_acc(N, d_model_); d_K_cmp_acc.fill(0.0);
    Tensor d_V_cmp_acc(N, d_model_); d_V_cmp_acc.fill(0.0);
    Tensor d_K_sel_acc(N, d_model_); d_K_sel_acc.fill(0.0);
    Tensor d_V_sel_acc(N, d_model_); d_V_sel_acc.fill(0.0);
    Tensor d_K_win_acc(N, d_model_); d_K_win_acc.fill(0.0);
    Tensor d_V_win_acc(N, d_model_); d_V_win_acc.fill(0.0);

    // ----- Step 3: branch 1 — compression backward -----
    // Forward:
    //   O_cmp[h, t, j] = Σ_{bi} a_cmp[h, t, bi] · V_tilde_cmp[h, bi, j]
    //   a_cmp[h, t, bi] = softmax_t (Q[t, h_off:h_off+head_dim] · K_tilde_cmp[h, bi, :]) / sqrt(head_dim)
    //   K_tilde_cmp[h, bi, j] = Σ_{k} W_phi_k[h, j*phi_dim + k] · phi_in_K[bi, k]
    //   phi_in_K[bi, k] = K_cmp[block_start + off, h_off + j_in] + pos_enc_(off, j_in)
    //                     where k = off*head_dim + j_in
    //
    // Backward chain:
    //   d_a_cmp[h, t, bi]   = Σ_j d_O_cmp[h, t, j] · V_tilde_cmp[h, bi, j]
    //   d_V_tilde_cmp[h, bi, j] += Σ_t d_a_cmp[h, t, bi] · a_cmp[h, t, bi]
    //                             (handled via a[h,t,bi] × d_O_cmp[h,t,j] → V_tilde)
    //   d_V_tilde -> d_phi_in_V -> d_V_cmp
    //   d_K_tilde -> d_phi_in_K -> d_K_cmp, d_W_phi_k
    //   d_a_cmp -> d_Q via softmax (attribution back to Q scores)

    Tensor d_K_tilde_cmp(num_query_heads_, n_cmp_ * head_dim_);
    Tensor d_V_tilde_cmp(num_query_heads_, n_cmp_ * head_dim_);
    Tensor d_a_cmp(num_query_heads_, N * n_cmp_);  // derivative through softmax (pre-softmax)
    Tensor d_Q_from_cmp(N, d_model_);
    d_Q_from_cmp.fill(0.0);
    if (use_compression_) {
        // d_V_tilde_cmp[h, bi, j] = Σ_t d_O_cmp[h, t, j] · a_cmp[h, t, bi]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t bi = 0; bi < n_cmp_; ++bi) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double acc = 0.0;
                    for (size_t t = 0; t < N; ++t) acc += d_O_cmp(h, t * head_dim_ + j) * last_attn_cmp_(h, t * n_cmp_ + bi);
                    d_V_tilde_cmp(h, bi * head_dim_ + j) = acc;
                }
            }
        }
        // d_a_cmp[h, t, bi] (pre-softmax) = Σ_j d_O_cmp[h, t, j] · V_tilde_cmp[h, bi, j]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                for (size_t bi = 0; bi < n_cmp_; ++bi) {
                    double acc = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        acc += d_O_cmp(h, t * head_dim_ + j) * last_V_tilde_cmp_(h, bi * head_dim_ + j);
                    d_a_cmp(h, t * n_cmp_ + bi) = acc;
                }
            }
        }
        // Softmax backward (row-wise over n_cmp_): dS_pre[t, bi] = a · (dA - Σ dA · a)
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                double sum_da_a = 0.0;
                for (size_t bi = 0; bi < n_cmp_; ++bi)
                    sum_da_a += d_a_cmp(h, t * n_cmp_ + bi) * last_attn_cmp_(h, t * n_cmp_ + bi);
                for (size_t bi = 0; bi < n_cmp_; ++bi)
                    d_a_cmp(h, t * n_cmp_ + bi) =
                        last_attn_cmp_(h, t * n_cmp_ + bi) *
                        (d_a_cmp(h, t * n_cmp_ + bi) - sum_da_a);
            }
        }
        // d_K_tilde_cmp[h, bi, j] += Σ_t d_a_cmp[h, t, bi] · scale_ · Q[t, h_off + j]
        // d_Q from cmp path += Σ_{bi} d_a_cmp[h, t, bi] · scale_ · K_tilde_cmp[h, bi, j]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            for (size_t bi = 0; bi < n_cmp_; ++bi) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double acc = 0.0;
                    for (size_t t = 0; t < N; ++t) acc += d_a_cmp(h, t * n_cmp_ + bi) * scale_ * last_Q_(t, h_off + j);
                    d_K_tilde_cmp(h, bi * head_dim_ + j) = acc;
                }
            }
        }
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double acc = 0.0;
                    for (size_t bi = 0; bi < n_cmp_; ++bi)
                        acc += d_a_cmp(h, t * n_cmp_ + bi) * scale_ * last_K_tilde_cmp_(h, bi * head_dim_ + j);
                    d_Q_from_cmp(t, h_off + j) += acc;
                }
            }
        }

        // Now backprop K_tilde_cmp → W_phi_k and K_cmp, and V_tilde_cmp → W_phi_v and V_cmp.
        // d_phi_in_K[bi, off*head_dim + j_in] += Σ_{j_out} d_K_tilde_cmp[h, bi, j_out] · W_phi_k[h, j_out*phi_dim + k]
        // grad_W_phi_k[h, j_out*phi_dim + k] += Σ_{bi} d_K_tilde_cmp[h, bi, j_out] · phi_in_K[bi, k]
        // d_K_cmp[block_start + off, h_off + j_in] = Σ_{bi containing (off, j_in)} d_phi_in_K[bi, off*head_dim + j_in]
        // We iterate over (h, bi, off, j_in, j_out).
        std::vector<double> d_phi_in_K(phi_dim, 0.0);
        std::vector<double> d_phi_in_V(phi_dim, 0.0);
        std::vector<double> phi_in_K(phi_dim, 0.0);
        std::vector<double> phi_in_V(phi_dim, 0.0);
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t bi = 0; bi < n_cmp_; ++bi) {
                size_t block_start = bi * stride_;
                // Recompute phi_in_K, phi_in_V (deterministic, no need to cache)
                for (size_t off = 0; off < l; ++off) {
                    for (size_t j = 0; j < head_dim_; ++j) {
                        phi_in_K[off * head_dim_ + j] =
                            last_K_cmp_(block_start + off, h_off + j) + pos_enc_(off, j);
                        phi_in_V[off * head_dim_ + j] =
                            last_V_cmp_(block_start + off, h_off + j) + pos_enc_(off, j);
                    }
                }
                std::fill(d_phi_in_K.begin(), d_phi_in_K.end(), 0.0);
                std::fill(d_phi_in_V.begin(), d_phi_in_V.end(), 0.0);
                // Backprop through φ
                for (size_t j_out = 0; j_out < head_dim_; ++j_out) {
                    double d_K = d_K_tilde_cmp(h, bi * head_dim_ + j_out);
                    double d_V = d_V_tilde_cmp(h, bi * head_dim_ + j_out);
                    for (size_t k = 0; k < phi_dim; ++k) {
                        d_phi_in_K[k] += d_K * W_phi_k(h, j_out * phi_dim + k);
                        d_phi_in_V[k] += d_V * W_phi_v(h, j_out * phi_dim + k);
                        grad_W_phi_k(h, j_out * phi_dim + k) += d_K * phi_in_K[k];
                        grad_W_phi_v(h, j_out * phi_dim + k) += d_V * phi_in_V[k];
                    }
                }
                // STASH d_K_cmp and d_V_cmp for later grad_W_x and d_input computation.
                // We store in d_K_tilde_cmp / d_V_tilde_cmp repurposed as d_K_cmp / d_V_cmp
                // scratch buffers (they're not needed beyond this point in the cmp branch).
                for (size_t off = 0; off < l; ++off) {
                    for (size_t j = 0; j < head_dim_; ++j) {
                        double dK = d_phi_in_K[off * head_dim_ + j];
                        double dV = d_phi_in_V[off * head_dim_ + j];
                        // d_K_cmp_acc and d_V_cmp_acc track d_K/d_V for d_input later.
                        d_K_cmp_acc(block_start + off, h_off + j) += dK;
                        d_V_cmp_acc(block_start + off, h_off + j) += dV;
                        // Accumulate grad_W_k_cmp[k, j] += input[token, k] · dK
                        for (size_t k_in = 0; k_in < d_model_; ++k_in) {
                            grad_W_k_cmp(k_in, h_off + j) +=
                                dK * last_input_(block_start + off, k_in);
                            grad_W_v_cmp(k_in, h_off + j) +=
                                dV * last_input_(block_start + off, k_in);
                        }
                    }
                }
            }
        }
        // (No accumulation into last_d_input_ here — d_Q_from_cmp is summed
        //  into d_Q later in step 6, then propagates to last_d_input_ via
        //  the d_input computation at the end.)
    }

    // ----- Step 4: branch 2 — selection backward -----
    // Forward:
    //   attn_sel[h, t, slot] = softmax_slot (Q[t, h_off:h_off+head_dim] · K_tilde_sel[h, t, slot, :]) · scale_
    //   O_sel[h, t, j] = Σ_{slot} attn_sel[h, t, slot] · V_tilde_sel[h, t, slot, j]
    //   K_tilde_sel[h, t, slot, j] = K_sel[src_token(t, slot), h_off + j]
    //   V_tilde_sel[h, t, slot, j] = V_sel[src_token(t, slot), h_off + j]
    Tensor d_Q_from_sel(N, d_model_);
    if (use_selection_) {
        // d_V_tilde_sel[h, t, slot, j] = d_O_sel[h, t, j] · a_sel[h, t, slot]
        // d_a_sel[h, t, slot] (pre-softmax) = Σ_j d_O_sel[h, t, j] · V_tilde_sel[h, t, slot, j]
        Tensor d_a_sel(num_query_heads_, N * sel_slot_dim);
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    double acc = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        acc += d_O_sel(h, t * head_dim_ + j) *
                               last_V_tilde_sel_(h, (t * sel_slot_dim + slot) * head_dim_ + j);
                    d_a_sel(h, t * sel_slot_dim + slot) = acc;
                }
            }
        }
        // Softmax backward over slots.
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                double sum_da_a = 0.0;
                for (size_t slot = 0; slot < sel_slot_dim; ++slot)
                    sum_da_a += d_a_sel(h, t * sel_slot_dim + slot) * last_attn_sel_(h, t * sel_slot_dim + slot);
                for (size_t slot = 0; slot < sel_slot_dim; ++slot)
                    d_a_sel(h, t * sel_slot_dim + slot) =
                        last_attn_sel_(h, t * sel_slot_dim + slot) *
                        (d_a_sel(h, t * sel_slot_dim + slot) - sum_da_a);
            }
        }
        // d_K_tilde_sel[h, t, slot, j] = Σ_t' d_a_sel[h, t', slot] · scale_ · Q[t', h_off + j]  (only when t'=t)
        // Wait — attn_sel[h, t, slot] only depends on Q[t, ...] (no cross-query).
        // So d_Q[t, h_off + j] += Σ_{slot} d_a_sel[h, t, slot] · scale_ · K_tilde_sel[h, t, slot, j]
        // grad_W_k_sel[k, j] += input[src_token, k] · dK_sel[src_token, j]
        //                        where dK_sel[src_token, j] = Σ_{uses} d_a_sel[h, t, slot] · scale_ · Q[t, h_off + j]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    double das = d_a_sel(h, t * sel_slot_dim + slot) * scale_;
                    size_t k_idx = static_cast<size_t>(last_top_idx_(h, t * n_sel_keep + (slot / block_size_)));
                    size_t off_in_block = slot % block_size_;
                    size_t src_token = k_idx * stride_ + off_in_block;
                    if (src_token >= N) continue;
                    for (size_t j = 0; j < head_dim_; ++j) {
                        // grad_W_k_sel[k, h_off + j] += input[src_token, k] · das · Q[t, h_off + j]
                        for (size_t k_in = 0; k_in < d_model_; ++k_in) {
                            grad_W_k_sel(k_in, h_off + j) +=
                                das * last_Q_(t, h_off + j) * last_input_(src_token, k_in);
                        }
                        // d_K_sel_acc accumulates dK = das · Q for d_input later.
                        d_K_sel_acc(src_token, h_off + j) += das * last_Q_(t, h_off + j);
                        // d_Q_from_sel[t, h_off + j] += das · K_tilde_sel[h, t, slot, j]
                        d_Q_from_sel(t, h_off + j) += das *
                            last_K_tilde_sel_(h, (t * sel_slot_dim + slot) * head_dim_ + j);
                    }
                }
            }
        }
        // grad_W_v_sel[k, j] += input[src_token, k] · dV_sel[src_token, j]
        //                        where dV_sel[src_token, j] = Σ_{uses} d_O_sel[h, t, j] · a_sel[h, t, slot]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t slot = 0; slot < sel_slot_dim; ++slot) {
                    double aw = last_attn_sel_(h, t * sel_slot_dim + slot);
                    size_t k_idx = static_cast<size_t>(last_top_idx_(h, t * n_sel_keep + (slot / block_size_)));
                    size_t off_in_block = slot % block_size_;
                    size_t src_token = k_idx * stride_ + off_in_block;
                    if (src_token >= N) continue;
                    for (size_t j = 0; j < head_dim_; ++j) {
                        double dV = d_O_sel(h, t * head_dim_ + j) * aw;
                        d_V_sel_acc(src_token, h_off + j) += dV;
                        for (size_t k_in = 0; k_in < d_model_; ++k_in) {
                            grad_W_v_sel(k_in, h_off + j) +=
                                dV * last_input_(src_token, k_in);
                        }
                    }
                }
            }
        }
    }

    // ----- Step 5: branch 3 — sliding window backward -----
    // Same structure as selection but with a fixed (non-learnable) attention
    // pattern: K_tilde_win[h, t, k, j] = K_win[src(t, k), h_off + j].
    Tensor d_Q_from_win(N, d_model_);
    if (use_window_) {
        // d_a_win[h, t, k] (pre-softmax) = Σ_j d_O_win[h, t, j] · V_tilde_win[h, t, k, j]
        Tensor d_a_win(num_query_heads_, N * w);
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                for (size_t k = 0; k < w; ++k) {
                    double acc = 0.0;
                    for (size_t j = 0; j < head_dim_; ++j)
                        acc += d_O_win(h, t * head_dim_ + j) *
                               last_V_tilde_win_(h, (t * w + k) * head_dim_ + j);
                    d_a_win(h, t * w + k) = acc;
                }
            }
        }
        // Softmax backward
        for (size_t h = 0; h < num_query_heads_; ++h) {
            for (size_t t = 0; t < N; ++t) {
                double sum_da_a = 0.0;
                for (size_t k = 0; k < w; ++k)
                    sum_da_a += d_a_win(h, t * w + k) * last_attn_win_(h, t * w + k);
                for (size_t k = 0; k < w; ++k)
                    d_a_win(h, t * w + k) =
                        last_attn_win_(h, t * w + k) *
                        (d_a_win(h, t * w + k) - sum_da_a);
            }
        }
        // grad_W_k_win[k, j] += input[src, k] · dK_win[src, j]
        //                        where dK_win[src, j] = Σ_{uses} d_a_win[h, t, k] · scale_ · Q[t, h_off + j]
        // d_Q_from_win[t, h_off + j] += d_a_win[h, t, k] · scale_ · K_tilde_win[h, t, k, j]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t k = 0; k < w; ++k) {
                    size_t src_token = (t < w - 1) ? 0 : t - (w - 1) + k;
                    if (src_token > t) src_token = 0;
                    double daw = d_a_win(h, t * w + k) * scale_;
                    for (size_t j = 0; j < head_dim_; ++j) {
                        for (size_t k_in = 0; k_in < d_model_; ++k_in) {
                            grad_W_k_win(k_in, h_off + j) +=
                                daw * last_Q_(t, h_off + j) * last_input_(src_token, k_in);
                        }
                        d_K_win_acc(src_token, h_off + j) += daw * last_Q_(t, h_off + j);
                        d_Q_from_win(t, h_off + j) += daw *
                            last_K_tilde_win_(h, (t * w + k) * head_dim_ + j);
                    }
                }
            }
        }
        // grad_W_v_win[k, j] += input[src, k] · dV_win[src, j]
        //                        where dV_win[src, j] = Σ_{uses} d_O_win[h, t, j] · a_win[h, t, k]
        for (size_t h = 0; h < num_query_heads_; ++h) {
            size_t h_off = h * head_dim_;
            for (size_t t = 0; t < N; ++t) {
                for (size_t k = 0; k < w; ++k) {
                    size_t src_token = (t < w - 1) ? 0 : t - (w - 1) + k;
                    if (src_token > t) src_token = 0;
                    for (size_t j = 0; j < head_dim_; ++j) {
                        double dV = d_O_win(h, t * head_dim_ + j) * last_attn_win_(h, t * w + k);
                        d_V_win_acc(src_token, h_off + j) += dV;
                        for (size_t k_in = 0; k_in < d_model_; ++k_in) {
                            grad_W_v_win(k_in, h_off + j) +=
                                dV * last_input_(src_token, k_in);
                        }
                    }
                }
            }
        }
    }

    // ----- Step 6: combine d_Q contributions from all paths + grad_W_q -----
    // d_Q = d_Q_from_cmp + d_Q_from_sel + d_Q_from_win + d_Q_from_gate
    // grad_W_q[i, j] += Σ_t d_Q[t, j] · input[t, i]   (since Q[t, j] = Σ_k input[t, k] · W_q[k, j])
    // d_input[t, i] = Σ_j d_Q[t, j] · W_q[i, j]      (since input → Q)
    Tensor d_Q(N, d_model_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = d_Q_from_gate(i, j);
            // Note: if use_compression_ is false, d_Q_from_cmp is uninitialized (zero by Tensor ctor)
            // — same for sel/win. To be safe, always accumulate conditionally.
            if (use_compression_) v += (i < d_Q_from_cmp.rows && j < d_Q_from_cmp.cols) ? d_Q_from_cmp(i, j) : 0.0;
            if (use_selection_)   v += d_Q_from_sel(i, j);
            if (use_window_)      v += d_Q_from_win(i, j);
            d_Q(i, j) = v;
        }
    }
    // grad_W_q
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t) acc += d_Q(t, j) * last_input_(t, i);
            grad_W_q(i, j) += acc;
        }
    // d_input
    Tensor d_input(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += d_Q(t, j) * W_q(i, j);
            d_input(t, i) = acc;
        }
    // Add contributions to d_input from each K/V projection:
    //   d_input[t, i] += Σ_j d_K_branch[t, j] · W_branch[i, j]
    //   d_input[t, i] += Σ_j d_V_branch[t, j] · W_branch[i, j]
    auto add_input_grad_from_KV = [&](const Tensor& d_K_acc, const Tensor& d_V_acc,
                                      const Tensor& W_k, const Tensor& W_v) {
        for (size_t t = 0; t < N; ++t)
            for (size_t i = 0; i < d_model_; ++i) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    acc += d_K_acc(t, j) * W_k(i, j);
                    acc += d_V_acc(t, j) * W_v(i, j);
                }
                d_input(t, i) += acc;
            }
    };
    add_input_grad_from_KV(d_K_cmp_acc, d_V_cmp_acc, W_k_cmp, W_v_cmp);
    add_input_grad_from_KV(d_K_sel_acc, d_V_sel_acc, W_k_sel, W_v_sel);
    add_input_grad_from_KV(d_K_win_acc, d_V_win_acc, W_k_win, W_v_win);

    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// NSABlock
// ============================================================================

NSABlock::NSABlock(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
                   size_t block_len, size_t stride, size_t top_n,
                   size_t window_size, size_t block_size, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim),
      nsa_(d_model, num_query_heads, num_kv_heads,
           block_len, stride, top_n, window_size, block_size),
      ln1_(d_model),
      ln2_(d_model)
{
    if (ffn_dim_ > 0) {
        W1_ = Tensor::random(d_model_, ffn_dim_, 0.05);
        b1_ = Tensor::zeros(1, ffn_dim_);
        W2_ = Tensor::random(ffn_dim_, d_model_, 0.05);
        b2_ = Tensor::zeros(1, d_model_);
        grad_W1_ = Tensor::zeros(d_model_, ffn_dim_);
        grad_b1_ = Tensor::zeros(1, ffn_dim_);
        grad_W2_ = Tensor::zeros(ffn_dim_, d_model_);
        grad_b2_ = Tensor::zeros(1, d_model_);
    } else {
        W1_ = Tensor(0, 0); b1_ = Tensor(0, 0);
        W2_ = Tensor(0, 0); b2_ = Tensor(0, 0);
        grad_W1_ = Tensor(0, 0); grad_b1_ = Tensor(0, 0);
        grad_W2_ = Tensor(0, 0); grad_b2_ = Tensor(0, 0);
    }
}

std::vector<Tensor*> NSABlock::parameters() {
    std::vector<Tensor*> p = nsa_.parameters();
    if (ffn_dim_ > 0) {
        p.push_back(&W1_); p.push_back(&b1_);
        p.push_back(&W2_); p.push_back(&b2_);
    }
    return p;
}

std::vector<Tensor*> NSABlock::gradients() {
    std::vector<Tensor*> g = nsa_.gradients();
    if (ffn_dim_ > 0) {
        g.push_back(&grad_W1_); g.push_back(&grad_b1_);
        g.push_back(&grad_W2_); g.push_back(&grad_b2_);
    }
    return g;
}

void NSABlock::zero_grad() {
    nsa_.zero_grad();
    ln1_.zero_grad(); ln2_.zero_grad();
    if (ffn_dim_ > 0) {
        grad_W1_.fill(0.0); grad_b1_.fill(0.0);
        grad_W2_.fill(0.0); grad_b2_.fill(0.0);
    }
}

void NSABlock::update_weights(double learning_rate) {
    nsa_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    if (ffn_dim_ > 0) {
        W1_ -= grad_W1_ * learning_rate;
        b1_ -= grad_b1_ * learning_rate;
        W2_ -= grad_W2_ * learning_rate;
        b2_ -= grad_b2_ * learning_rate;
    }
}

Tensor NSABlock::forward(const Tensor& input) {
    last_input_ = input.clone();
    last_ln1_ = ln1_.forward(input);
    last_attn_out_ = nsa_.forward(last_ln1_);
    last_res1_ = input + last_attn_out_;
    if (ffn_dim_ > 0) {
        last_ln2_ = ln2_.forward(last_res1_);
        // FFN: pregelu = ln2 @ W1^T + b1  ;  h = gelu(pregelu)  ;  out = h @ W2^T + b2
        const size_t N = input.rows;
        last_ffn_pregelu_ = Tensor(N, ffn_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double acc = b1_(0, j);
                for (size_t k = 0; k < d_model_; ++k) acc += last_ln2_(i, k) * W1_(k, j);
                last_ffn_pregelu_(i, j) = acc;
            }
        Tensor h(N, ffn_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double x = last_ffn_pregelu_(i, j);
                double xc = std::max(-4.0, std::min(4.0, x));
                double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
                double th = std::tanh(u);
                h(i, j) = 0.5 * x * (1.0 + th);
            }
        last_ffn_out_ = Tensor(N, d_model_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = b2_(0, j);
                for (size_t k = 0; k < ffn_dim_; ++k) acc += h(i, k) * W2_(k, j);
                last_ffn_out_(i, j) = acc;
            }
        return last_res1_ + last_ffn_out_;
    } else {
        return last_res1_;
    }
}

Tensor NSABlock::backward(const Tensor& grad_output, double learning_rate) {
    const size_t N = grad_output.rows;
    // d_res1 = gradient w.r.t. the output of the attention sub-layer (before
    // the residual add). When FFN is present, includes both FFN path's d_ln2
    // and the bypass through grad_output. When ffn_dim=0, just = grad_output.
    Tensor d_res1(N, d_model_);
    if (ffn_dim_ > 0) {
        // d_ffn_out = grad_output, then d_h, d_W2, d_b2, d_pregelu, d_W1, d_b1, d_ln2
        Tensor d_h(N, ffn_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < ffn_dim_; ++k) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j) acc += grad_output(i, j) * W2_(k, j);
                d_h(i, k) = acc;
            }
        // d_W2[k, j] += Σ_i grad_output[i, j] · h[i, k];  we need h.
        // Reconstruct h = gelu(last_ffn_pregelu_)
        Tensor h(N, ffn_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double x = last_ffn_pregelu_(i, j);
                double xc = std::max(-4.0, std::min(4.0, x));
                double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
                double th = std::tanh(u);
                h(i, j) = 0.5 * x * (1.0 + th);
            }
        for (size_t k = 0; k < ffn_dim_; ++k)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i) acc += grad_output(i, j) * h(i, k);
                grad_W2_(k, j) += acc;
            }
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += grad_output(i, j);
            grad_b2_(0, j) += acc;
        }
        // d_pregelu[i, j] = d_h[i, j] · gelu'(pregelu[i, j])
        Tensor d_pregelu(N, ffn_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double x = last_ffn_pregelu_(i, j);
                double xc = std::max(-4.0, std::min(4.0, x));
                double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
                double th = std::tanh(u);
                double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
                double gelu_p = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
                d_pregelu(i, j) = d_h(i, j) * gelu_p;
            }
        // grad_W1[k, j] += Σ_i d_pregelu[i, j] · last_ln2_[i, k]
        for (size_t k = 0; k < d_model_; ++k)
            for (size_t j = 0; j < ffn_dim_; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i) acc += d_pregelu(i, j) * last_ln2_(i, k);
                grad_W1_(k, j) += acc;
            }
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += d_pregelu(i, j);
            grad_b1_(0, j) += acc;
        }
        // d_ln2[i, k] = Σ_j d_pregelu[i, j] · W1_[k, j]
        Tensor d_ln2(N, d_model_);
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < d_model_; ++k) {
                double acc = 0.0;
                for (size_t j = 0; j < ffn_dim_; ++j) acc += d_pregelu(i, j) * W1_(k, j);
                d_ln2(i, k) = acc;
            }
        // Backward through LayerNorm 2
        Tensor d_res1_from_ln2 = ln2_.backward(d_ln2, learning_rate);
        // d_res1 = d_res1_from_ln2 + grad_output (residual bypass through out = res1 + ffn_out)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < d_model_; ++j) d_res1(i, j) = d_res1_from_ln2(i, j) + grad_output(i, j);
    } else {
        // No FFN: out = input + nsa(ln1(input)), so d_res1 = d_nsa_out = grad_output
        d_res1 = grad_output.clone();
    }
    // d_attn_out = d_res1 (the NSA sees this gradient w.r.t. its output)
    Tensor d_attn_out = d_res1.clone();
    // NSA.backward returns d_(input to NSA) = d_ln1_input.
    Tensor d_ln1 = nsa_.backward(d_attn_out, learning_rate);
    // d_input from LN path: backward through ln1
    Tensor d_input_from_ln = ln1_.backward(d_ln1, learning_rate);
    // d_input from residual bypass: out = input + nsa(ln1(input)) — bypass adds grad_output
    // (the d_out at this block's output) directly to d_input.
    Tensor d_input(N, d_model_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_input(i, j) = d_input_from_ln(i, j) + grad_output(i, j);
    last_d_input_ = d_input;
    return d_input;
}

// ============================================================================
// NSAModel
// ============================================================================

NSAModel::NSAModel(size_t input_dim, size_t d_model, size_t output_dim,
                   size_t num_blocks, size_t num_query_heads, size_t num_kv_heads,
                   size_t block_len, size_t stride, size_t top_n,
                   size_t window_size, size_t block_size, size_t ffn_dim)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim)
{
    W_in_  = Tensor::random(input_dim_, d_model_, 0.05);
    b_in_  = Tensor::zeros(1, d_model_);
    W_out_ = Tensor::random(d_model_, output_dim_, 0.05);
    b_out_ = Tensor::zeros(1, output_dim_);
    grad_W_in_  = Tensor::zeros(input_dim_, d_model_);
    grad_b_in_  = Tensor::zeros(1, d_model_);
    grad_W_out_ = Tensor::zeros(d_model_, output_dim_);
    grad_b_out_ = Tensor::zeros(1, output_dim_);
    for (size_t i = 0; i < num_blocks; ++i)
        blocks_.emplace_back(d_model, num_query_heads, num_kv_heads,
                             block_len, stride, top_n, window_size, block_size, ffn_dim);
}

std::vector<Tensor*> NSAModel::parameters() {
    std::vector<Tensor*> p = {&W_in_, &b_in_, &W_out_, &b_out_};
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        for (auto* x : bp) p.push_back(x);
    }
    return p;
}

std::vector<Tensor*> NSAModel::gradients() {
    std::vector<Tensor*> g = {&grad_W_in_, &grad_b_in_, &grad_W_out_, &grad_b_out_};
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        for (auto* x : bg) g.push_back(x);
    }
    return g;
}

void NSAModel::zero_grad() {
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
}

void NSAModel::update_weights(double learning_rate) {
    W_in_  -= grad_W_in_  * learning_rate;
    b_in_  -= grad_b_in_  * learning_rate;
    W_out_ -= grad_W_out_ * learning_rate;
    b_out_ -= grad_b_out_ * learning_rate;
    for (auto& b : blocks_) b.update_weights(learning_rate);
}

Tensor NSAModel::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("NSAModel.forward: input cols mismatch");
    last_input_ = input.clone();
    const size_t N = input.rows;
    // in_proj: input @ W_in^T + b_in
    Tensor proj(N, d_model_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = b_in_(0, j);
            for (size_t k = 0; k < input_dim_; ++k) acc += input(i, k) * W_in_(k, j);
            proj(i, j) = acc;
        }
    last_proj_ = proj;
    Tensor cur = proj;
    for (auto& b : blocks_) cur = b.forward(cur);
    last_block_out_ = cur;
    // out_proj
    Tensor out(N, output_dim_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < output_dim_; ++j) {
            double acc = b_out_(0, j);
            for (size_t k = 0; k < d_model_; ++k) acc += cur(i, k) * W_out_(k, j);
            out(i, j) = acc;
        }
    return out;
}

Tensor NSAModel::backward(const Tensor& grad_output, double learning_rate) {
    const size_t N = grad_output.rows;
    if (grad_output.cols != output_dim_)
        throw std::invalid_argument("NSAModel.backward: grad_output cols mismatch");
    // d_cur (output of last block)
    Tensor d_cur(N, d_model_);
    for (size_t i = 0; i < N; ++i)
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < output_dim_; ++j) acc += grad_output(i, j) * W_out_(k, j);
            d_cur(i, k) = acc;
        }
    // grad_W_out, grad_b_out
    for (size_t k = 0; k < d_model_; ++k)
        for (size_t j = 0; j < output_dim_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += grad_output(i, j) * last_block_out_(i, k);
            grad_W_out_(k, j) += acc;
        }
    for (size_t j = 0; j < output_dim_; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < N; ++i) acc += grad_output(i, j);
        grad_b_out_(0, j) += acc;
    }
    // Backprop through blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it)
        d_cur = it->backward(d_cur, learning_rate);
    // d_proj = d_cur; backprop through in_proj
    for (size_t k = 0; k < input_dim_; ++k)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += d_cur(i, j) * last_input_(i, k);
            grad_W_in_(k, j) += acc;
        }
    for (size_t j = 0; j < d_model_; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < N; ++i) acc += d_cur(i, j);
        grad_b_in_(0, j) += acc;
    }
    Tensor d_input(N, input_dim_);
    for (size_t i = 0; i < N; ++i)
        for (size_t k = 0; k < input_dim_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += d_cur(i, j) * W_in_(k, j);
            d_input(i, k) = acc;
        }
    return d_input;
}
