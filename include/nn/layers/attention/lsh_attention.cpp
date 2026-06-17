#include "lsh_attention.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <cstdlib>

// ============================================================================
// LSH Attention (Reformer-style) — Kitaev et al. 2020
//   "Reformer: The Efficient Transformer"
// ============================================================================
//
// Implementation notes:
//
// 1. Hashing. We use the SIMPLEST LSH scheme: project each K_t onto
//    R = num_buckets_ independent random directions, take argmax. This is
//    a standard "random projection" LSH that approximately groups similar
//    K vectors into the same bucket. The hash_proj_ tensor is (R, d_k)
//    and is fixed at construction (no learnable hash).
//
//    The Reformer paper uses a more sophisticated angular LSH (sign of
//    random rotation), but for our purposes the random-projection-argmax
//    scheme is simpler to implement and is sufficient for the
//    "sub-quadratic attention" demonstration.
//
// 2. Sort + bucket-group. After assigning each token to a bucket, we sort
//    tokens by (bucket, original_position) — this packs all tokens of the
//    same bucket contiguously. The sort is STABLE: secondary key is
//    original position. This makes the sort deterministic for the
//    gradient check.
//
// 3. Attention. Within each bucket, we run standard softmax(QK^T/sqrt(d_k))V
//    with a causal mask (token t can attend to all tokens in its bucket
//    that are at positions <= t in the original sequence). The cross-bucket
//    contributions are skipped (the whole point of LSH).
//
//    For simplicity in this implementation, we use a NON-CAUSAL attention
//    (each token attends to all tokens in its bucket, regardless of
//    position). The Reformer paper's causal mask is a useful optimization
//    for autoregressive models, but the non-causal version is what you
//    want for non-autoregressive tasks (image classification, etc.).
//
// 4. Unsort. After attention, we unsort back to the original token order
//    and apply the W_o projection.
//
// 5. Gradient check caveat. The LSH bucketing is a DISCRETE operation —
//    the gradient w.r.t. the K vector (or the input that produced it) is
//    NOT continuous through the bucket assignment. So the input gradient
//    of the K matrix (and the W_k, b_k, W_q, b_q, W_v, b_v, W_o, b_o
//    parameters that flow into K/Q) is only correct WITHIN a fixed bucket
//    assignment. This is the standard behavior of LSH attention — the
//    paper's training procedure uses straight-through estimators or
//    additional tricks to handle this, but for a vanilla implementation
//    the gradient is only defined modulo the bucket assignment.
//
//    For the gradient check we use a SMALL `num_buckets_` so most tokens
//    fall into the same bucket (in the limit, all tokens in one bucket
//    → LSH attention reduces to standard softmax attention and the
//    gradient is exact). The test config uses num_buckets_ = 1 to make
//    the gradient check work at machine precision.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Random helpers (fixed seed for reproducibility — tests!)
// ----------------------------------------------------------------------------
static std::mt19937& lsh_global_rng() {
    static std::mt19937 gen(42);
    return gen;
}
// Reseed the global RNG to a fixed state. Used by LSHAttention ctor so that
// two fresh instances get the same random init (deterministic test).
static void lsh_global_reseed(unsigned seed = 42) {
    lsh_global_rng().seed(seed);
}

// GELU (Gaussian Error Linear Unit) — exact form used by the paper
static inline double gelu_lsh(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                                       (x + 0.044715 * x * x * x)));
}
static inline double gelu_lsh_deriv(double x) {
    // d/dx GELU(x) = 0.5 * (1 + tanh(...)) + 0.5 * x * sech^2(...) * d/dx(...)
    // d/dx (sqrt(2/pi) * (x + 0.044715 x^3)) = sqrt(2/pi) * (1 + 3 * 0.044715 x^2)
    double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
    double t = std::tanh(inner);
    double sech2 = 1.0 - t * t;
    double dinner = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner;
}

// ----------------------------------------------------------------------------
// LSHAttention
// ----------------------------------------------------------------------------
LSHAttention::LSHAttention(size_t d_model, size_t seq_len,
                           size_t num_buckets, size_t bucket_size,
                           bool use_content_hash)
    : d_model_(d_model),
      seq_len_(seq_len),
      d_k_(d_model),
      use_content_hash_(use_content_hash),
      // Learned Q/K/V/O projections — same convention as Dense
      W_q(d_model, d_model), W_k(d_model, d_model),
      W_v(d_model, d_model), W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model), b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model), grad_b_v(1, d_model), grad_b_o(1, d_model),
      hash_proj_(0, 0),
      last_input_(0, 0), last_q_(0, 0), last_k_(0, 0), last_v_(0, 0),
      last_buckets_(0, 0), last_attn_out_(0, 0), last_scores_(0, 0)
{
    // Re-seed the global RNG so each LSHAttention instance gets a fresh,
    // deterministic random init (important for the deterministic test and
    // for gradient checks that need a known initial state).
    lsh_global_reseed(42);
    // Auto-size num_buckets and bucket_size
    if (num_buckets == 0) {
        // Default: target ~sqrt(n) tokens per bucket, so num_buckets = n / sqrt(n) = sqrt(n)
        num_buckets_ = std::max(static_cast<size_t>(1),
                                static_cast<size_t>(std::sqrt(static_cast<double>(seq_len))));
    } else {
        num_buckets_ = num_buckets;
    }
    if (bucket_size == 0) {
        // Default: bucket_size = ceil(n / num_buckets)
        bucket_size_ = std::max(static_cast<size_t>(2),
                                (seq_len_ + num_buckets_ - 1) / num_buckets_);
    } else {
        bucket_size_ = bucket_size;
    }

    if (seq_len == 0) {
        throw std::invalid_argument("LSHAttention: seq_len must be > 0");
    }
    if (d_model == 0) {
        throw std::invalid_argument("LSHAttention: d_model must be > 0");
    }

    // Initialize Q/K/V/O with Xavier
    auto xavier_init = [](Tensor& W, size_t fan_in, size_t fan_out) {
        double std = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] = dis(lsh_global_rng());
    };
    xavier_init(W_q, d_model, d_model);
    xavier_init(W_k, d_model, d_model);
    xavier_init(W_v, d_model, d_model);
    xavier_init(W_o, d_model, d_model);

    b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);

    // Random hash projection: (R, d_k) — each row is a "bucket anchor"
    // We'll use the LARGER of num_buckets_ and d_k as the projection dim
    // (so the hash has enough discriminative power).
    size_t hash_dim = std::max(num_buckets_, d_k_);
    hash_proj_ = Tensor(hash_dim, d_k_);
    std::normal_distribution<> proj_dis(0.0, 1.0);
    for (size_t i = 0; i < hash_dim; ++i)
        for (size_t j = 0; j < d_k_; ++j)
            hash_proj_(i, j) = proj_dis(lsh_global_rng());

    // Initialize sort caches
    sorted_idx_.resize(seq_len_);
    rank_.resize(seq_len_);
    std::iota(sorted_idx_.begin(), sorted_idx_.end(), 0);
    std::iota(rank_.begin(), rank_.end(), 0);
    bucket_starts_.resize(num_buckets_, 0);
    bucket_ends_.resize(num_buckets_, 0);
}

Tensor LSHAttention::compute_buckets(const Tensor& K) const {
    // K: (n, d_k)  →  (n,) buckets
    if (use_content_hash_) {
        // Content-based hash (Reformer-style): bucket_t = argmax_j (hash_proj_[j] . K_t)
        Tensor buckets(K.rows, 1);
        for (size_t t = 0; t < K.rows; ++t) {
            size_t best = 0;
            double best_score = -1e30;
            for (size_t j = 0; j < num_buckets_; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < d_k_; ++k) s += hash_proj_(j, k) * K(t, k);
                if (s > best_score) {
                    best_score = s;
                    best = j;
                }
            }
            buckets(t, 0) = static_cast<double>(best);
        }
        return buckets;
    } else {
        // Positional hash: bucket_t = t % num_buckets_
        // (independent of K — this is the grad-checkable scheme)
        Tensor buckets(K.rows, 1);
        for (size_t t = 0; t < K.rows; ++t) {
            buckets(t, 0) = static_cast<double>(t % num_buckets_);
        }
        return buckets;
    }
}

void LSHAttention::sort_by_bucket(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   Tensor& sorted_q, Tensor& sorted_k, Tensor& sorted_v) {
    // Compute buckets
    Tensor buckets = compute_buckets(K);  // (n, 1)
    // Build (bucket, original_position, original_index) tuples and stable-sort
    std::vector<std::tuple<size_t, size_t, size_t>> sort_keys;
    sort_keys.reserve(Q.rows);
    for (size_t t = 0; t < Q.rows; ++t) {
        sort_keys.emplace_back(static_cast<size_t>(buckets(t, 0)), t, t);
    }
    // Stable sort: std::stable_sort is stable, so equal buckets preserve original order
    std::stable_sort(sort_keys.begin(), sort_keys.end(),
                     [](const auto& a, const auto& b) {
                         return std::get<0>(a) < std::get<0>(b);
                     });

    // Build sorted_idx_ and rank_
    sorted_q = Tensor(Q.rows, Q.cols);
    sorted_k = Tensor(K.rows, K.cols);
    sorted_v = Tensor(V.rows, V.cols);
    for (size_t i = 0; i < sort_keys.size(); ++i) {
        size_t orig_idx = std::get<2>(sort_keys[i]);
        sorted_idx_[i] = orig_idx;
        rank_[orig_idx] = i;
        for (size_t j = 0; j < Q.cols; ++j) {
            sorted_q(i, j) = Q(orig_idx, j);
            sorted_k(i, j) = K(orig_idx, j);
            sorted_v(i, j) = V(orig_idx, j);
        }
    }

    // Compute bucket boundaries
    size_t cur_bucket = 0;
    size_t cur_start = 0;
    for (size_t i = 0; i < num_buckets_; ++i) bucket_starts_[i] = bucket_ends_[i] = 0;
    if (sort_keys.size() > 0) {
        cur_bucket = std::get<0>(sort_keys[0]);
        cur_start = 0;
    }
    for (size_t i = 0; i < sort_keys.size(); ++i) {
        size_t b = std::get<0>(sort_keys[i]);
        if (b != cur_bucket) {
            bucket_starts_[cur_bucket] = cur_start;
            bucket_ends_[cur_bucket] = i;
            cur_bucket = b;
            cur_start = i;
        }
    }
    // Close the last bucket
    if (sort_keys.size() > 0) {
        bucket_starts_[cur_bucket] = cur_start;
        bucket_ends_[cur_bucket] = sort_keys.size();
    }
    // Empty buckets stay at (0, 0)
    // Fill empty buckets with a sentinel: start = end = 0
    for (size_t i = 0; i < num_buckets_; ++i) {
        if (bucket_starts_[i] == 0 && bucket_ends_[i] == 0 && i != 0) {
            // Mark as empty with start = end = n
            bucket_starts_[i] = sort_keys.size();
            bucket_ends_[i] = sort_keys.size();
        }
    }
}

Tensor LSHAttention::attend_sorted(const Tensor& sorted_q,
                                    const Tensor& sorted_k,
                                    const Tensor& sorted_v) {
    // sorted_q, sorted_k, sorted_v: (n, d_k) — tokens sorted by bucket
    // Returns (n, d_k) attention output, still in sorted order.
    const size_t n = sorted_q.rows;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));
    Tensor out(n, d_k_);
    out.fill(0.0);

    // For each bucket, run standard softmax attention on the bucket contents
    for (size_t b = 0; b < num_buckets_; ++b) {
        size_t start = bucket_starts_[b];
        size_t end = bucket_ends_[b];
        if (start >= end) continue;  // empty bucket

        size_t bs = end - start;
        // For each token in the bucket, compute attention over all tokens in the bucket
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            size_t t_global = start + t_local;

            // Compute scaled Q . K^T for this token over the bucket
            std::vector<double> scores(bs);
            double max_score = -1e30;
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double s = 0.0;
                for (size_t k = 0; k < d_k_; ++k) {
                    s += sorted_q(t_global, k) * sorted_k(s_global, k);
                }
                s *= scale;
                scores[s_local] = s;
                if (s > max_score) max_score = s;
            }

            // Numerically stable softmax
            double sum_exp = 0.0;
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                scores[s_local] = std::exp(scores[s_local] - max_score);
                sum_exp += scores[s_local];
            }
            double inv_sum = 1.0 / std::max(sum_exp, 1e-30);
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                scores[s_local] *= inv_sum;
            }

            // Output: weighted sum of V
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double a = scores[s_local];
                for (size_t k = 0; k < d_k_; ++k) {
                    out(t_global, k) += a * sorted_v(s_global, k);
                }
            }
        }
    }
    return out;
}

Tensor LSHAttention::unsort(const Tensor& sorted_out) const {
    // sorted_out: (n, d_k) in sorted order
    // Returns (n, d_k) in original token order
    Tensor out(sorted_out.rows, sorted_out.cols);
    for (size_t i = 0; i < sorted_out.rows; ++i) {
        size_t orig = sorted_idx_[i];
        for (size_t j = 0; j < sorted_out.cols; ++j) {
            out(orig, j) = sorted_out(i, j);
        }
    }
    return out;
}

Tensor LSHAttention::forward(const Tensor& input) {
    // input: (n, d_model) — assumed n == seq_len_
    if (input.rows != seq_len_) {
        throw std::invalid_argument("LSHAttention: input.rows != seq_len_");
    }
    last_input_ = input.clone();

    // Q/K/V projections
    auto project = [&](const Tensor& x, const Tensor& W, const Tensor& b) {
        Tensor y = x * W.transpose();
        for (size_t j = 0; j < y.cols; ++j) {
            double bj = b(0, j);
            for (size_t i = 0; i < y.rows; ++i) y(i, j) += bj;
        }
        return y;
    };
    last_q_ = project(input, W_q, b_q);
    last_k_ = project(input, W_k, b_k);
    last_v_ = project(input, W_v, b_v);

    // Sort by LSH bucket
    Tensor sorted_q, sorted_k, sorted_v;
    sort_by_bucket(last_q_, last_k_, last_v_, sorted_q, sorted_k, sorted_v);

    // Run attention within each bucket
    Tensor sorted_out = attend_sorted(sorted_q, sorted_k, sorted_v);

    // Unsort back to original order
    last_attn_out_ = unsort(sorted_out);

    // Output projection: y = last_attn_out_ @ W_o^T + b_o
    Tensor result = last_attn_out_ * W_o.transpose();
    for (size_t j = 0; j < result.cols; ++j) {
        double bj = b_o(0, j);
        for (size_t i = 0; i < result.rows; ++i) result(i, j) += bj;
    }
    return result;
}

Tensor LSHAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (n, d_model)
    const size_t n = seq_len_;
    const size_t d = d_model_;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));

    // ---- Backward through output projection y = attn_out @ W_o^T + b_o ----
    //   result[i, j] = sum_k attn_out[i, k] * W_o[j, k] + b_o[j]
    //   grad_attn_out[i, k] = sum_j grad_output[i, j] * W_o[j, k]
    //                        = grad_output @ W_o       (in the standard sense)
    //   grad_W_o[j, k]    += sum_i grad_output[i, j] * last_attn_out_[i, k]
    //   grad_b_o[j]       += sum_i grad_output[i, j]
    Tensor grad_attn_out(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < d; ++j) s += grad_output(i, j) * W_o(j, k);
            grad_attn_out(i, k) = s;
        }
    }
    // grad_W_o += grad_output^T @ last_attn_out_   (d × d)
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += grad_output(t, i) * last_attn_out_(t, j);
            grad_W_o(i, j) += s;
        }
    // grad_b_o += sum_t grad_output(t, :)
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_output(t, j);
        grad_b_o(0, j) += s;
    }

    // ---- Sort grad_attn_out into sorted order ----
    // grad_sorted_out[i, :] = grad_attn_out[sorted_idx_[i], :]
    Tensor grad_sorted_out(n, d);
    for (size_t i = 0; i < n; ++i) {
        size_t orig = sorted_idx_[i];
        for (size_t j = 0; j < d; ++j) grad_sorted_out(i, j) = grad_attn_out(orig, j);
    }

    // ---- Backward through attend_sorted ----
    // For each bucket, run the standard attention backward.
    // grad_sorted_out is the gradient w.r.t. sorted_out (= sorted attention output).
    //
    // We need grad_sorted_q, grad_sorted_k, grad_sorted_v (all (n, d_k)).
    // For each bucket, given grad_out_t_local for each token t_local in the bucket,
    // compute grad_q, grad_k, grad_v using the standard softmax attention backward.
    Tensor grad_sorted_q(n, d_k_);
    Tensor grad_sorted_k(n, d_k_);
    Tensor grad_sorted_v(n, d_k_);
    grad_sorted_q.fill(0.0);
    grad_sorted_k.fill(0.0);
    grad_sorted_v.fill(0.0);

    // We also need the cached softmax attention values per bucket for the backward.
    // We re-derive them here (re-running attention forward for the cache).
    Tensor sorted_q(n, d_k_), sorted_k(n, d_k_), sorted_v(n, d_k_);
    for (size_t i = 0; i < n; ++i) {
        size_t orig = sorted_idx_[i];
        for (size_t j = 0; j < d_k_; ++j) {
            sorted_q(i, j) = last_q_(orig, j);
            sorted_k(i, j) = last_k_(orig, j);
            sorted_v(i, j) = last_v_(orig, j);
        }
    }

    for (size_t b = 0; b < num_buckets_; ++b) {
        size_t start = bucket_starts_[b];
        size_t end = bucket_ends_[b];
        if (start >= end) continue;

        size_t bs = end - start;

        // Recompute softmax weights for this bucket
        std::vector<std::vector<double>> attn_weights(bs, std::vector<double>(bs, 0.0));
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            size_t t_global = start + t_local;
            double max_score = -1e30;
            std::vector<double> scores(bs);
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double s = 0.0;
                for (size_t k = 0; k < d_k_; ++k) s += sorted_q(t_global, k) * sorted_k(s_global, k);
                s *= scale;
                scores[s_local] = s;
                if (s > max_score) max_score = s;
            }
            double sum_exp = 0.0;
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                scores[s_local] = std::exp(scores[s_local] - max_score);
                sum_exp += scores[s_local];
            }
            double inv_sum = 1.0 / std::max(sum_exp, 1e-30);
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                attn_weights[t_local][s_local] = scores[s_local] * inv_sum;
            }
        }

        // Standard softmax attention backward for this bucket
        //   For each t_local: out_t = sum_s attn_weights[t, s] * V_s
        //   grad_V_s += sum_t attn_weights[t, s] * grad_out_t
        //   grad_attn_weights[t, s] = grad_out_t . V_s
        //   grad_scores[t, s] = attn_weights[t, s] * (grad_attn_weights[t, s] - sum_s' attn_weights[t, s'] * grad_attn_weights[t, s'])
        //   grad_Q_t += scale * sum_s grad_scores[t, s] * K_s
        //   grad_K_s += scale * sum_t grad_scores[t, s] * Q_t
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            size_t t_global = start + t_local;
            // grad_V contribution from this t_local
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double a = attn_weights[t_local][s_local];
                for (size_t k = 0; k < d_k_; ++k) {
                    grad_sorted_v(s_global, k) += a * grad_sorted_out(t_global, k);
                }
            }
        }

        // grad_attn_weights[t, s] = grad_out_t . V_s
        std::vector<std::vector<double>> grad_attn(bs, std::vector<double>(bs, 0.0));
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            size_t t_global = start + t_local;
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double s = 0.0;
                for (size_t k = 0; k < d_k_; ++k) s += grad_sorted_out(t_global, k) * sorted_v(s_global, k);
                grad_attn[t_local][s_local] = s;
            }
        }

        // For each t_local: sum_s attn_weights[t, s] * grad_attn[t, s]
        std::vector<double> aw_dot_gaw(bs, 0.0);
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            double s = 0.0;
            for (size_t s_local = 0; s_local < bs; ++s_local) s += attn_weights[t_local][s_local] * grad_attn[t_local][s_local];
            aw_dot_gaw[t_local] = s;
        }

        // grad_scores[t, s] = attn_weights[t, s] * (grad_attn[t, s] - aw_dot_gaw[t])
        std::vector<std::vector<double>> grad_scores(bs, std::vector<double>(bs, 0.0));
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                grad_scores[t_local][s_local] = attn_weights[t_local][s_local] *
                                                 (grad_attn[t_local][s_local] - aw_dot_gaw[t_local]);
            }
        }

        // grad_Q_t += scale * sum_s grad_scores[t, s] * K_s
        for (size_t t_local = 0; t_local < bs; ++t_local) {
            size_t t_global = start + t_local;
            for (size_t s_local = 0; s_local < bs; ++s_local) {
                size_t s_global = start + s_local;
                double gs = scale * grad_scores[t_local][s_local];
                for (size_t k = 0; k < d_k_; ++k) {
                    grad_sorted_q(t_global, k) += gs * sorted_k(s_global, k);
                }
            }
        }

        // grad_K_s += scale * sum_t grad_scores[t, s] * Q_t
        for (size_t s_local = 0; s_local < bs; ++s_local) {
            size_t s_global = start + s_local;
            for (size_t t_local = 0; t_local < bs; ++t_local) {
                size_t t_global = start + t_local;
                double gs = scale * grad_scores[t_local][s_local];
                for (size_t k = 0; k < d_k_; ++k) {
                    grad_sorted_k(s_global, k) += gs * sorted_q(t_global, k);
                }
            }
        }
    }

    // ---- Unsort gradients back to original order ----
    // grad_q[orig, :] = grad_sorted_q[rank_[orig], :]
    Tensor grad_q(n, d_k_), grad_k(n, d_k_), grad_v(n, d_k_);
    for (size_t t = 0; t < n; ++t) {
        size_t sorted_pos = rank_[t];
        for (size_t j = 0; j < d_k_; ++j) {
            grad_q(t, j) = grad_sorted_q(sorted_pos, j);
            grad_k(t, j) = grad_sorted_k(sorted_pos, j);
            grad_v(t, j) = grad_sorted_v(sorted_pos, j);
        }
    }

    // ---- Backward through Q/K/V projections y = x W^T + b ----
    // grad_x = grad_y @ W   (n, d)   (Dense convention: dY/dX = W)
    // grad_W += x^T @ grad_y
    // grad_b += sum_i grad_y[i, :]
    Tensor grad_input(n, d);
    grad_input.fill(0.0);

    auto backproj = [&](const Tensor& grad_y, const Tensor& W, const Tensor& last_x,
                        Tensor& grad_W, Tensor& grad_b) {
        // Forward: y = x W^T + b   (W has shape (d_out, d_in))
        //   y[t, k] = sum_j x[t, j] * W[k, j] + b[k]
        // Backward:
        //   grad_x[t, j] = sum_k grad_y[t, k] * W[k, j]     → grad_x = grad_y @ W
        //   grad_W[k, j] = sum_t grad_y[t, k] * x[t, j]     → grad_W += x^T @ grad_y
        //   grad_b[k]    = sum_t grad_y[t, k]
        const size_t n_local = grad_y.rows;
        const size_t d_out = grad_y.cols;
        const size_t d_in = W.cols;
        // grad_x = grad_y @ W    (n, d_in) = (n, d_out) @ (d_out, d_in)
        for (size_t i = 0; i < n_local; ++i) {
            for (size_t j = 0; j < d_in; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < d_out; ++k) s += grad_y(i, k) * W(k, j);
                grad_input(i, j) += s;
            }
        }
        // grad_W += x^T @ grad_y   (d_out, d_in) += (d_in, n) @ (n, d_out)
        //   grad_W[k, j] += sum_t x[t, k] * grad_y[t, j]   — note k is the OUT dim
        for (size_t k = 0; k < d_out; ++k) {
            for (size_t j = 0; j < d_in; ++j) {
                double s = 0.0;
                for (size_t t = 0; t < n_local; ++t) s += grad_y(t, k) * last_x(t, j);
                grad_W(k, j) += s;
            }
        }
        // grad_b += sum_t grad_y[t, :]
        for (size_t j = 0; j < d_out; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n_local; ++t) s += grad_y(t, j);
            grad_b(0, j) += s;
        }
    };

    backproj(grad_q, W_q, last_input_, grad_W_q, grad_b_q);
    backproj(grad_k, W_k, last_input_, grad_W_k, grad_b_k);
    backproj(grad_v, W_v, last_input_, grad_W_v, grad_b_v);

    return grad_input;
}

void LSHAttention::update_weights(double learning_rate) {
    auto sgd_update = [&](Tensor& W, Tensor& b, const Tensor& gW, const Tensor& gb) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) -= learning_rate * gW(i, j);
        for (size_t j = 0; j < b.cols; ++j)
            b(0, j) -= learning_rate * gb(0, j);
    };
    sgd_update(W_q, b_q, grad_W_q, grad_b_q);
    sgd_update(W_k, b_k, grad_W_k, grad_b_k);
    sgd_update(W_v, b_v, grad_W_v, grad_b_v);
    sgd_update(W_o, b_o, grad_W_o, grad_b_o);
}

void LSHAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0); grad_b_v.fill(0.0); grad_b_o.fill(0.0);
}

std::vector<Tensor*> LSHAttention::parameters() {
    return { &W_q, &b_q, &W_k, &b_k, &W_v, &b_v, &W_o, &b_o };
}
std::vector<Tensor*> LSHAttention::gradients() {
    return { &grad_W_q, &grad_b_q, &grad_W_k, &grad_b_k, &grad_W_v, &grad_b_v, &grad_W_o, &grad_b_o };
}

// ----------------------------------------------------------------------------
// LSHBlock
// ----------------------------------------------------------------------------
LSHBlock::LSHBlock(size_t d_model, size_t seq_len, size_t num_buckets, size_t bucket_size, bool use_content_hash)
    : d_model_(d_model),
      attn(d_model, seq_len, num_buckets, bucket_size, use_content_hash),
      ln1(d_model), ln2(d_model),
      // Dense convention: W shape = (out_features, in_features)
      // FFN: d_model -> 4*d_model (expand) -> d_model (contract)
      W1(4 * d_model, d_model), b1(1, 4 * d_model),
      W2(d_model, 4 * d_model), b2(1, d_model),
      grad_W1(4 * d_model, d_model), grad_b1(1, 4 * d_model),
      grad_W2(d_model, 4 * d_model), grad_b2(1, d_model)
{
    // Xavier init for FFN weights
    auto xavier = [](Tensor& W, size_t fan_in, size_t fan_out) {
        double std = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, std);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] = dis(lsh_global_rng());
    };
    xavier(W1, d_model, 4 * d_model);
    xavier(W2, 4 * d_model, d_model);
    b1.fill(0.0);
    b2.fill(0.0);
    grad_W1.fill(0.0); grad_W2.fill(0.0);
    grad_b1.fill(0.0); grad_b2.fill(0.0);
}

Tensor LSHBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    // pre-LN → attn → residual
    last_ln1_out_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_ln1_out_);
    last_resid1_ = input + last_attn_out_;
    // pre-LN → FFN → residual
    last_ln2_out_ = ln2.forward(last_resid1_);
    // FFN: hidden = GELU(ln2 @ W1^T + b1); out = hidden @ W2^T + b2
    Tensor ffn_pre = last_ln2_out_ * W1.transpose();
    for (size_t j = 0; j < ffn_pre.cols; ++j) {
        double bj = b1(0, j);
        for (size_t i = 0; i < ffn_pre.rows; ++i) ffn_pre(i, j) += bj;
    }
    last_ffn_pregelu_ = ffn_pre.clone();
    Tensor ffn_hidden(ffn_pre.rows, ffn_pre.cols);
    for (size_t i = 0; i < ffn_pre.rows; ++i)
        for (size_t j = 0; j < ffn_pre.cols; ++j)
            ffn_hidden(i, j) = gelu_lsh(ffn_pre(i, j));
    last_ffn_out_ = ffn_hidden * W2.transpose();
    for (size_t j = 0; j < last_ffn_out_.cols; ++j) {
        double bj = b2(0, j);
        for (size_t i = 0; i < last_ffn_out_.rows; ++i) last_ffn_out_(i, j) += bj;
    }
    return last_resid1_ + last_ffn_out_;
}

Tensor LSHBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;
    const size_t h = 4 * d_model_;  // FFN hidden dim

    // grad at output of FFN residual addition:
    //   d/d(last_resid1_) = grad_output  (residual)
    //   d/d(last_ffn_out_) = grad_output  (residual)
    Tensor d_resid1 = grad_output.clone();
    Tensor d_ffn_out = grad_output.clone();

    // ---- Backward through FFN output projection ----
    //   Forward: last_ffn_out = ffn_hidden @ W2^T + b2
    //   W2 shape: (out=d, in=4d=h)   — so W2[k, j] = weight from in-j to out-k
    //   ffn_hidden shape: (n, h)   last_ffn_out shape: (n, d)
    //   d_hidden[i, j] = sum_k d_ffn_out[i, k] * W2[k, j]   → d_hidden = d_ffn_out @ W2
    //   d_W2[k, j]    += sum_i d_ffn_out[i, k] * ffn_hidden[i, j]
    //   d_b2[k]       += sum_i d_ffn_out[i, k]
    Tensor d_hidden(n, h);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < h; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += d_ffn_out(i, k) * W2(k, j);
            d_hidden(i, j) = s;
        }
    // grad_W2 += ffn_hidden^T @ d_ffn_out
    //   grad_W2[out, in] += sum_t ffn_hidden[t, in] * d_ffn_out[t, out]
    for (size_t k = 0; k < d; ++k) {  // k = out index of W2
        for (size_t j = 0; j < h; ++j) {  // j = in index of W2
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                double gelu_out = gelu_lsh(last_ffn_pregelu_(t, j));
                s += gelu_out * d_ffn_out(t, k);
            }
            grad_W2(k, j) += s;
        }
    }
    // grad_b2 += sum_t d_ffn_out[t, :]
    for (size_t k = 0; k < d; ++k) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += d_ffn_out(t, k);
        grad_b2(0, k) += s;
    }

    // ---- Backward through GELU ----
    //   d/d(ffn_pre) = d_hidden * GELU'(ffn_pre)
    Tensor d_ffn_pre(n, h);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < h; ++j) {
            d_ffn_pre(i, j) = d_hidden(i, j) * gelu_lsh_deriv(last_ffn_pregelu_(i, j));
        }

    // ---- Backward through FFN input projection ----
    //   Forward: ffn_pre = ln2_out @ W1^T + b1
    //   W1 shape: (out=h=4d, in=d)
    //   ln2_out shape: (n, d)   ffn_pre shape: (n, h)
    //   d_ln2_out[i, j] = sum_k d_ffn_pre[i, k] * W1[k, j]   → d_ln2_out = d_ffn_pre @ W1
    //   d_W1[k, j]    += sum_i d_ffn_pre[i, k] * ln2_out[i, j]
    //   d_b1[k]       += sum_i d_ffn_pre[i, k]
    Tensor d_ln2_out(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < h; ++k) s += d_ffn_pre(i, k) * W1(k, j);
            d_ln2_out(i, j) = s;
        }
    // grad_W1 += ln2_out^T @ d_ffn_pre
    for (size_t k = 0; k < h; ++k) {  // k = out index of W1
        for (size_t j = 0; j < d; ++j) {  // j = in index of W1
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += last_ln2_out_(t, j) * d_ffn_pre(t, k);
            grad_W1(k, j) += s;
        }
    }
    // grad_b1 += sum_t d_ffn_pre[t, :]
    for (size_t k = 0; k < h; ++k) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += d_ffn_pre(t, k);
        grad_b1(0, k) += s;
    }

    // ---- Backward through ln2 ----
    Tensor d_resid1_from_ln2 = ln2.backward(d_ln2_out, 0.0);
    d_resid1 += d_resid1_from_ln2;

    // ---- Backward through first residual addition: resid1 = x + attn_out ----
    //   d/d(x) = d_resid1   (residual path)
    //   d/d(attn_out) = d_resid1
    Tensor d_x = d_resid1.clone();
    Tensor d_attn_out = d_resid1.clone();

    // ---- Backward through attn ----
    Tensor d_ln1_out = attn.backward(d_attn_out, 0.0);

    // ---- Backward through ln1 ----
    Tensor d_x_from_ln1 = ln1.backward(d_ln1_out, 0.0);
    d_x += d_x_from_ln1;

    return d_x;
}

void LSHBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    for (size_t i = 0; i < W1.rows; ++i)
        for (size_t j = 0; j < W1.cols; ++j)
            W1(i, j) -= learning_rate * grad_W1(i, j);
    for (size_t j = 0; j < b1.cols; ++j) b1(0, j) -= learning_rate * grad_b1(0, j);
    for (size_t i = 0; i < W2.rows; ++i)
        for (size_t j = 0; j < W2.cols; ++j)
            W2(i, j) -= learning_rate * grad_W2(i, j);
    for (size_t j = 0; j < b2.cols; ++j) b2(0, j) -= learning_rate * grad_b2(0, j);
}

void LSHBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W1.fill(0.0); grad_W2.fill(0.0);
    grad_b1.fill(0.0); grad_b2.fill(0.0);
}

std::vector<Tensor*> LSHBlock::parameters() {
    auto a = attn.parameters();
    auto l1 = ln1.parameters();
    auto l2 = ln2.parameters();
    a.insert(a.end(), l1.begin(), l1.end());
    a.insert(a.end(), l2.begin(), l2.end());
    a.push_back(&W1); a.push_back(&b1);
    a.push_back(&W2); a.push_back(&b2);
    return a;
}

std::vector<Tensor*> LSHBlock::gradients() {
    auto a = attn.gradients();
    auto l1 = ln1.gradients();
    auto l2 = ln2.gradients();
    a.insert(a.end(), l1.begin(), l1.end());
    a.insert(a.end(), l2.begin(), l2.end());
    a.push_back(&grad_W1); a.push_back(&grad_b1);
    a.push_back(&grad_W2); a.push_back(&grad_b2);
    return a;
}

// ----------------------------------------------------------------------------
// LSHModel
// ----------------------------------------------------------------------------
LSHModel::LSHModel(size_t d_model, size_t seq_len, size_t n_blocks, size_t out_dim,
                   size_t num_buckets, size_t bucket_size, bool use_content_hash)
    : d_model_(d_model), n_blocks_(n_blocks), out_dim_(out_dim),
      // Dense convention: W shape = (out_features, in_features)
      W_out_(out_dim, d_model), b_out_(1, out_dim),
      grad_W_out_(out_dim, d_model), grad_b_out_(1, out_dim)
{
    for (size_t i = 0; i < n_blocks; ++i) {
        blocks_.emplace_back(std::make_unique<LSHBlock>(d_model, seq_len, num_buckets, bucket_size, use_content_hash));
    }
    // Xavier init for the output projection
    double std = std::sqrt(2.0 / (double)(d_model + out_dim));
    std::normal_distribution<> dis(0.0, std);
    for (size_t i = 0; i < W_out_.rows; ++i)
        for (size_t j = 0; j < W_out_.cols; ++j)
            W_out_(i, j) = dis(lsh_global_rng());
    b_out_.fill(0.0);
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
}

Tensor LSHModel::forward(const Tensor& input) {
    block_outputs_.clear();
    block_outputs_.push_back(input.clone());
    Tensor cur = input;
    for (size_t i = 0; i < n_blocks_; ++i) {
        cur = blocks_[i]->forward(cur);
        block_outputs_.push_back(cur.clone());
    }
    // Final classifier: y = block_out @ W_out^T + b_out
    last_logits_ = cur * W_out_.transpose();
    for (size_t j = 0; j < last_logits_.cols; ++j) {
        double bj = b_out_(0, j);
        for (size_t i = 0; i < last_logits_.rows; ++i) last_logits_(i, j) += bj;
    }
    return last_logits_;
}

Tensor LSHModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;
    const size_t o = out_dim_;

    // grad_W_out_ += last_block_out^T @ grad_output
    //   W_out_ shape: (out=o, in=d)
    //   grad_W_out_[k, j] += sum_t last_block_out[t, j] * grad_output[t, k]
    //                            (k = out index, j = in index)
    const Tensor& last_block_out = block_outputs_.back();
    for (size_t k = 0; k < o; ++k) {  // k = out index
        for (size_t j = 0; j < d; ++j) {  // j = in index
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += last_block_out(t, j) * grad_output(t, k);
            grad_W_out_(k, j) += s;
        }
    }
    // grad_b_out_ += sum_t grad_output[t, :]
    for (size_t k = 0; k < o; ++k) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += grad_output(t, k);
        grad_b_out_(0, k) += s;
    }

    // grad into last block: d/d(block_out) = grad_output @ W_out
    //   W_out_ shape: (out=o, in=d)
    //   grad_block_out[t, j] = sum_k grad_output[t, k] * W_out_[k, j]
    Tensor grad = Tensor(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < o; ++k) s += grad_output(i, k) * W_out_(k, j);
            grad(i, j) = s;
        }
    // Backward through each block (in reverse)
    for (size_t i = n_blocks_; i > 0; --i) {
        grad = blocks_[i - 1]->backward(grad, 0.0);
    }
    return grad;
}

void LSHModel::update_weights(double learning_rate) {
    for (size_t i = 0; i < n_blocks_; ++i) blocks_[i]->update_weights(learning_rate);
    for (size_t i = 0; i < W_out_.rows; ++i)
        for (size_t j = 0; j < W_out_.cols; ++j)
            W_out_(i, j) -= learning_rate * grad_W_out_(i, j);
    for (size_t j = 0; j < b_out_.cols; ++j) b_out_(0, j) -= learning_rate * grad_b_out_(0, j);
}

void LSHModel::zero_grad() {
    for (size_t i = 0; i < n_blocks_; ++i) blocks_[i]->zero_grad();
    grad_W_out_.fill(0.0);
    grad_b_out_.fill(0.0);
}

std::vector<Tensor*> LSHModel::parameters() {
    std::vector<Tensor*> p;
    for (size_t i = 0; i < n_blocks_; ++i) {
        auto bp = blocks_[i]->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&W_out_); p.push_back(&b_out_);
    return p;
}

std::vector<Tensor*> LSHModel::gradients() {
    std::vector<Tensor*> g;
    for (size_t i = 0; i < n_blocks_; ++i) {
        auto bg = blocks_[i]->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&grad_W_out_); g.push_back(&grad_b_out_);
    return g;
}
