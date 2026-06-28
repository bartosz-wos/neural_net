// ============================================================================
// BigBird Sparse Attention — Zaheer et al. 2020
//   "Big Bird: Transformers for Longer Sequences" (arXiv:2007.14062)
// ============================================================================
//
// Three components of the BigBird attention pattern, per query t:
//   W (window):  attend to the 2*window_size+1 tokens around t (clipped to [0, n))
//   R (random):  attend to `num_random` randomly-chosen distinct positions
//   G (global):  the first `num_global` tokens attend to/from ALL tokens
//
// The mask M[t, s] = 1 iff s ∈ A(t) is FIXED at construction (seed-based).
// This makes the gradient check exact: when we perturb an input or weight
// by ±ε, the same mask is used in both branches (mask doesn't depend on
// the perturbed values), so the analytical gradient through M is well-defined.
//
// Forward (single-head):
//
//   Q = input @ W_q^T + b_q         (n, d)
//   K = input @ W_k^T + b_k         (n, d)
//   V = input @ W_v^T + b_v         (n, d)
//   scores[t, s] = Q_t . K_s / sqrt(d)            (n, n)
//   masked[t, s] = (M[t, s] == 0) ? -1e9 : scores[t, s]
//   attn[t, s]   = softmax_t(masked)              (n, n)
//   head_out     = attn @ V                        (n, d)
//   output       = head_out @ W_o^T + b_o          (n, d)
//
// Backward (reverse):
//
//   1) d_head_out from d_output via W_o (Dense backward)
//   2) d_attn, d_V from d_head_out (d_attn = d_head_out @ V^T, d_V = attn^T @ d_head_out)
//   3) d_masked = softmax_backward(d_attn, attn)   (per-row)
//   4) d_scores[t, s] = (M[t, s] == 0) ? 0 : d_masked[t, s]
//   5) dQ = d_scores @ K / sqrt(d),  dK = d_scores^T @ Q / sqrt(d),  dV unchanged
//   6) d_input from dQ/dK/dV via W_q/W_k/W_v^T  (Dense backward chain)
//   7) d_W_q, d_W_k, d_W_v, d_W_o, d_b_q, d_b_k, d_b_v, d_b_o
//
// All standard Dense-style gradient chain (y = x W^T + b):
//   dW[k, j] += sum_t d_y[t, k] * x[t, j]   (k = OUT dim, j = IN dim)
//   db[k]    += sum_t d_y[t, k]
//   dx[t, j] = sum_k d_y[t, k] * W[k, j]    (= d_y @ W)
// ============================================================================

#include "bigbird.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

// ----------------------------------------------------------------------------
// Per-instance RNG (deterministic per-seed so two fresh instances with the
// same seed produce the same mask)
// ----------------------------------------------------------------------------
static std::mt19937& bigbird_global_rng() {
    static std::mt19937 gen(0);
    return gen;
}

// ----------------------------------------------------------------------------
// Helper: row-wise softmax of a (rows, cols) Tensor. Each row's softmax is
// computed with the max-subtraction trick for numerical stability. Masked-out
// entries (those equal to -1e9) are still included in the per-row max — this
// is fine because if -1e9 is the row max, then the exp values are tiny and
// the row softmax is concentrated on the unmasked positions.
// ----------------------------------------------------------------------------
static Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        // Row max
        double row_max = x(i, 0);
        for (size_t j = 1; j < x.cols; ++j) {
            if (x(i, j) > row_max) row_max = x(i, j);
        }
        // Compute exp(x - max)
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x(i, j) - row_max);
            result(i, j) = e;
            sum += e;
        }
        // Normalize (with safety floor for the degenerate all-masked case)
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j) {
            result(i, j) *= inv;
        }
    }
    return result;
}

// ============================================================================
// BigBirdAttention constructor
// ============================================================================
BigBirdAttention::BigBirdAttention(size_t d_model, size_t seq_len,
                                   size_t window_size, size_t num_random,
                                   size_t num_global, unsigned seed)
    : d_model_(d_model),
      seq_len_(seq_len),
      window_size_(window_size),
      num_random_(num_random),
      num_global_(num_global),
      d_k_(d_model),
      // Learned projections (Dense convention)
      W_q(d_model, d_model), W_k(d_model, d_model),
      W_v(d_model, d_model), W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model), b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model),
      grad_b_v(1, d_model), grad_b_o(1, d_model),
      random_idx_(seq_len, num_random == 0 ? 1 : num_random),
      last_input_(0, 0), last_q_(0, 0), last_k_(0, 0), last_v_(0, 0),
      last_attn_(0, 0), last_head_out_(0, 0), last_output_(0, 0)
{
    if (seq_len == 0) {
        throw std::invalid_argument("BigBirdAttention: seq_len must be > 0");
    }
    if (d_model == 0) {
        throw std::invalid_argument("BigBirdAttention: d_model must be > 0");
    }
    if (num_global > seq_len) {
        throw std::invalid_argument("BigBirdAttention: num_global must be <= seq_len");
    }

    // Initialize weights with Xavier (small init for stability with softmax)
    bigbird_global_rng().seed(seed);
    std::normal_distribution<> dis(0.0, std::sqrt(2.0 / (2.0 * d_model)));
    for (auto* W : {&W_q, &W_k, &W_v, &W_o}) {
        for (size_t i = 0; i < W->rows; ++i)
            for (size_t j = 0; j < W->cols; ++j)
                (*W)(i, j) = dis(bigbird_global_rng());
    }
    b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0);
    grad_b_v.fill(0.0); grad_b_o.fill(0.0);

    // Build the random indices. For each query t, sample `num_random_`
    // distinct positions from [0, n) (excluding t itself and excluding the
    // global tokens, to avoid double-counting).
    //
    // Each query's random pattern is fixed at construction (per-t seed) so
    // gradient checks are exact.
    random_idx_.fill(0.0);
    if (num_random_ > 0) {
        for (size_t t = 0; t < seq_len_; ++t) {
            // Build the pool of candidate positions (exclude t itself and
            // the global tokens)
            std::vector<size_t> pool;
            for (size_t s = 0; s < seq_len_; ++s) {
                if (s == t) continue;       // skip self
                if (s < num_global_) continue;  // skip global tokens
                pool.push_back(s);
            }
            // Shuffle deterministically using a per-t seed
            std::mt19937 local_rng(static_cast<unsigned>(seed) ^ static_cast<unsigned>(t) * 2654435761u);
            std::shuffle(pool.begin(), pool.end(), local_rng);
            // Take the first num_random_ positions
            size_t take = std::min(num_random_, pool.size());
            for (size_t r = 0; r < take; ++r) {
                random_idx_(t, r) = static_cast<double>(pool[r]);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// BigBirdAttention::attention_mask: returns (n, n) mask of 1.0 / 0.0
// ----------------------------------------------------------------------------
Tensor BigBirdAttention::attention_mask() const {
    Tensor mask(seq_len_, seq_len_);
    mask.fill(0.0);

    for (size_t t = 0; t < seq_len_; ++t) {
        if (t < num_global_) {
            // Global token: attends to everyone, and is attended by everyone
            for (size_t s = 0; s < seq_len_; ++s) {
                mask(t, s) = 1.0;
            }
        } else {
            // Non-global token: attends to its window + random positions +
            // the global tokens
            long t_lo = (long)t - (long)window_size_;
            long t_hi = (long)t + (long)window_size_;
            if (t_lo < 0) t_lo = 0;
            if (t_hi >= (long)seq_len_) t_hi = (long)seq_len_ - 1;
            for (long s = t_lo; s <= t_hi; ++s) {
                mask(t, (size_t)s) = 1.0;
            }
            // Random positions
            for (size_t r = 0; r < num_random_; ++r) {
                size_t s = (size_t)random_idx_(t, r);
                if (s < seq_len_) mask(t, s) = 1.0;
            }
            // Global positions: every non-global token also attends to all
            // global tokens
            for (size_t s = 0; s < num_global_; ++s) {
                mask(t, s) = 1.0;
            }
        }
    }
    return mask;
}

// ----------------------------------------------------------------------------
// BigBirdAttention::compute_attention: forward pass through the attention
// ----------------------------------------------------------------------------
Tensor BigBirdAttention::compute_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    const double scale = 1.0 / std::sqrt((double)d_k_);
    // scores = Q @ K^T * scale   (n, n)
    Tensor scores = (Q * K.transpose()) * scale;

    // Apply mask: positions NOT in M get -1e9 so softmax gives 0 there
    Tensor mask = attention_mask();
    const double NEG_BIG = -1e9;
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t s = 0; s < seq_len_; ++s) {
            if (mask(t, s) < 0.5) {
                scores(t, s) = NEG_BIG;
            }
        }
    }

    // Row softmax
    Tensor attn = row_softmax(scores);

    // head_out = attn @ V  (n, d)
    Tensor head_out = attn * V;
    return head_out;
}

// ----------------------------------------------------------------------------
// BigBirdAttention::forward
// ----------------------------------------------------------------------------
Tensor BigBirdAttention::forward(const Tensor& input) {
    if (input.rows != seq_len_ || input.cols != d_model_) {
        throw std::invalid_argument("BigBirdAttention::forward: input shape mismatch");
    }
    last_input_ = input;

    // Q = input @ W_q^T + b_q   (n, d)
    last_q_ = input * W_q.transpose();
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_q_(i, j) += b_q(0, j);

    last_k_ = input * W_k.transpose();
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_k_(i, j) += b_k(0, j);

    last_v_ = input * W_v.transpose();
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_v_(i, j) += b_v(0, j);

    // Sparse attention
    last_head_out_ = compute_attention(last_q_, last_k_, last_v_);

    // Re-derive last_attn_ from last_q_ and last_k_ for the backward.
    // (We could cache it in compute_attention, but doing it here keeps the
    // helper clean.)
    {
        const double scale = 1.0 / std::sqrt((double)d_k_);
        Tensor scores = (last_q_ * last_k_.transpose()) * scale;
        Tensor mask = attention_mask();
        const double NEG_BIG = -1e9;
        for (size_t t = 0; t < seq_len_; ++t)
            for (size_t s = 0; s < seq_len_; ++s)
                if (mask(t, s) < 0.5) scores(t, s) = NEG_BIG;
        last_attn_ = row_softmax(scores);
    }

    // Output projection: output = head_out @ W_o^T + b_o   (n, d)
    last_output_ = last_head_out_ * W_o.transpose();
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_output_(i, j) += b_o(0, j);

    return last_output_;
}

// ----------------------------------------------------------------------------
// BigBirdAttention::backward
//   d_output (n, d) is the upstream gradient
//   Returns d_input (n, d)
// ----------------------------------------------------------------------------
Tensor BigBirdAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    const double scale = 1.0 / std::sqrt((double)d_k_);

    // --- Step 1: d_head_out via W_o ---
    //   output = head_out @ W_o^T + b_o
    //   d_head_out[t, j] = sum_k grad_output[t, k] * W_o[k, j]
    //   dW_o[k, j] += sum_t grad_output[t, k] * head_out[t, j]
    //   db_o[k]    += sum_t grad_output[t, k]
    Tensor d_head_out(seq_len_, d_model_);
    d_head_out.fill(0.0);
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = grad_output(t, k);
            grad_b_o(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                d_head_out(t, j) += g * W_o(k, j);
                grad_W_o(k, j) += g * last_head_out_(t, j);
            }
        }
    }

    // --- Step 2: head_out = attn @ V ---
    //   d_attn[t, s] = sum_k d_head_out[t, k] * V[s, k]
    //   d_V[s, k]    = sum_t attn[t, s] * d_head_out[t, k]
    Tensor d_attn(seq_len_, seq_len_);
    d_attn.fill(0.0);
    Tensor d_V(seq_len_, d_model_);
    d_V.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t s = 0; s < seq_len_; ++s) {
            double a = last_attn_(t, s);
            double da = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                da += d_head_out(t, k) * last_v_(s, k);
                d_V(s, k) += a * d_head_out(t, k);
            }
            d_attn(t, s) = da;
        }
    }

    // --- Step 3: softmax backward (per-row) ---
    //   For each row t: d_masked[t, s] = attn[t, s] * (d_attn[t, s] - sum_s' attn[t, s'] * d_attn[t, s'])
    Tensor d_masked(seq_len_, seq_len_);
    d_masked.fill(0.0);
    Tensor mask = attention_mask();
    for (size_t t = 0; t < seq_len_; ++t) {
        double row_sum = 0.0;
        for (size_t s = 0; s < seq_len_; ++s) {
            row_sum += last_attn_(t, s) * d_attn(t, s);
        }
        for (size_t s = 0; s < seq_len_; ++s) {
            d_masked(t, s) = last_attn_(t, s) * (d_attn(t, s) - row_sum);
            // Mask out positions not in A(t) — the masked-out scores were
            // forced to -1e9 (a constant) so they don't carry gradient
            // information. Setting their gradient to 0 is consistent with
            // the fact that a perturbation in the score at those positions
            // doesn't change the forward output (their softmax is exactly 0).
            if (mask(t, s) < 0.5) {
                d_masked(t, s) = 0.0;
            }
        }
    }

    // --- Step 4: scores = Q @ K^T / sqrt(d) ---
    //   dQ[t, j] = sum_s d_masked[t, s] * K[s, j] / sqrt(d)
    //   dK[s, j] = sum_t d_masked[t, s] * Q[t, j] / sqrt(d)
    Tensor dQ(seq_len_, d_model_);
    Tensor dK(seq_len_, d_model_);
    dQ.fill(0.0);
    dK.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t s = 0; s < seq_len_; ++s) {
            double g = d_masked(t, s) * scale;
            for (size_t j = 0; j < d_model_; ++j) {
                dQ(t, j) += g * last_k_(s, j);
                dK(s, j) += g * last_q_(t, j);
            }
        }
    }

    // --- Step 5: gradients for W_q, W_k, W_v, b_q, b_k, b_v ---
    //   Q = input @ W_q^T + b_q:
    //     dW_q[k, j] += sum_t dQ[t, k] * input[t, j]
    //     db_q[k]    += sum_t dQ[t, k]
    //     d_input_from_q[t, j] = sum_k dQ[t, k] * W_q[k, j]
    grad_W_q.fill(0.0); grad_b_q.fill(0.0);
    Tensor d_input_from_q(seq_len_, d_model_);
    d_input_from_q.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = dQ(t, k);
            grad_b_q(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_q(k, j) += g * last_input_(t, j);
                d_input_from_q(t, j) += g * W_q(k, j);
            }
        }
    }

    grad_W_k.fill(0.0); grad_b_k.fill(0.0);
    Tensor d_input_from_k(seq_len_, d_model_);
    d_input_from_k.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = dK(t, k);
            grad_b_k(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_k(k, j) += g * last_input_(t, j);
                d_input_from_k(t, j) += g * W_k(k, j);
            }
        }
    }

    grad_W_v.fill(0.0); grad_b_v.fill(0.0);
    Tensor d_input_from_v(seq_len_, d_model_);
    d_input_from_v.fill(0.0);
    for (size_t t = 0; t < seq_len_; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = d_V(t, k);
            grad_b_v(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_v(k, j) += g * last_input_(t, j);
                d_input_from_v(t, j) += g * W_v(k, j);
            }
        }
    }

    // --- Step 6: combine the three input-gradient contributions ---
    Tensor d_input(seq_len_, d_model_);
    d_input.fill(0.0);
    for (size_t i = 0; i < d_input.data.size(); ++i) {
        d_input.data[i] = d_input_from_q.data[i]
                        + d_input_from_k.data[i]
                        + d_input_from_v.data[i];
    }

    return d_input;
}

void BigBirdAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.data.size(); ++i)
            W.data[i] -= learning_rate * gW.data[i];
    };
    auto sgd_b = [&](Tensor& b, const Tensor& gb) {
        for (size_t i = 0; i < b.data.size(); ++i)
            b.data[i] -= learning_rate * gb.data[i];
    };
    sgd(W_q, grad_W_q);    sgd_b(b_q, grad_b_q);
    sgd(W_k, grad_W_k);    sgd_b(b_k, grad_b_k);
    sgd(W_v, grad_W_v);    sgd_b(b_v, grad_b_v);
    sgd(W_o, grad_W_o);    sgd_b(b_o, grad_b_o);
}

void BigBirdAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0);
    grad_b_v.fill(0.0); grad_b_o.fill(0.0);
}

std::vector<Tensor*> BigBirdAttention::parameters() {
    return {&W_q, &b_q, &W_k, &b_k, &W_v, &b_v, &W_o, &b_o};
}
std::vector<Tensor*> BigBirdAttention::gradients() {
    return {&grad_W_q, &grad_b_q, &grad_W_k, &grad_b_k,
            &grad_W_v, &grad_b_v, &grad_W_o, &grad_b_o};
}

// ============================================================================
// BigBirdBlock
// ============================================================================
BigBirdBlock::BigBirdBlock(size_t d_model, size_t seq_len,
                           size_t window_size, size_t num_random,
                           size_t num_global, unsigned seed)
    : d_model_(d_model),
      attn(d_model, seq_len, window_size, num_random, num_global, seed),
      ln1(d_model), ln2(d_model),
      // FFN: hidden = 4 * d_model
      W1(2 * d_model, d_model), b1(1, 2 * d_model),
      W2(d_model, 2 * d_model), b2(1, d_model),
      grad_W1(2 * d_model, d_model), grad_b1(1, 2 * d_model),
      grad_W2(d_model, 2 * d_model), grad_b2(1, d_model),
      last_x_(0, 0), last_ln1_out_(0, 0), last_attn_out_(0, 0),
      last_resid1_(0, 0), last_ln2_out_(0, 0),
      last_ffn_pregelu_(0, 0), last_ffn_out_(0, 0)
{
    // Xavier init for FFN
    bigbird_global_rng().seed(seed ^ 0xDEADBEEFu);
    std::normal_distribution<> dis1(0.0, std::sqrt(2.0 / (d_model + 2 * d_model)));
    for (size_t i = 0; i < W1.rows; ++i)
        for (size_t j = 0; j < W1.cols; ++j)
            W1(i, j) = dis1(bigbird_global_rng());
    std::normal_distribution<> dis2(0.0, std::sqrt(2.0 / (2 * d_model + d_model)));
    for (size_t i = 0; i < W2.rows; ++i)
        for (size_t j = 0; j < W2.cols; ++j)
            W2(i, j) = dis2(bigbird_global_rng());
    b1.fill(0.0); b2.fill(0.0);
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

Tensor BigBirdBlock::forward(const Tensor& input) {
    last_x_ = input;

    // pre-LN → attention → residual
    last_ln1_out_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_ln1_out_);
    last_resid1_ = input + last_attn_out_;  // element-wise residual

    // pre-LN → FFN → residual
    last_ln2_out_ = ln2.forward(last_resid1_);
    last_ffn_pregelu_ = last_ln2_out_ * W1.transpose();
    for (size_t i = 0; i < last_ffn_pregelu_.rows; ++i)
        for (size_t j = 0; j < last_ffn_pregelu_.cols; ++j)
            last_ffn_pregelu_(i, j) += b1(0, j);
    // Apply GELU element-wise
    Tensor ffn_hidden = last_ffn_pregelu_.apply([](double x) {
        return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                                          (x + 0.044715 * x * x * x)));
    });
    last_ffn_out_ = ffn_hidden * W2.transpose();
    for (size_t i = 0; i < last_ffn_out_.rows; ++i)
        for (size_t j = 0; j < last_ffn_out_.cols; ++j)
            last_ffn_out_(i, j) += b2(0, j);

    return last_resid1_ + last_ffn_out_;
}

Tensor BigBirdBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Residual at the end: y = last_resid1_ + last_ffn_out_
    // Both branches contribute to d_last_resid1_ and d_last_ffn_out_
    Tensor d_ffn_out = grad_output.clone();
    Tensor d_resid1(grad_output.rows, grad_output.cols);
    d_resid1.fill(0.0);
    for (size_t i = 0; i < d_resid1.data.size(); ++i) {
        d_resid1.data[i] = grad_output.data[i];  // residual path
    }
    // --- FFN backward ---
    // last_ffn_out = ffn_hidden @ W2^T + b2
    // where ffn_hidden = GELU(last_ffn_pregelu_) and last_ffn_pregelu_ = last_ln2_out_ @ W1^T + b1
    Tensor d_ffn_hidden(last_ffn_pregelu_.rows, last_ffn_pregelu_.cols);
    d_ffn_hidden.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
    Tensor ffn_hidden(last_ffn_pregelu_.rows, last_ffn_pregelu_.cols);
    for (size_t t = 0; t < last_ffn_out_.rows; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = d_ffn_out(t, k);
            grad_b2(0, k) += g;
            for (size_t j = 0; j < 2 * d_model_; ++j) {
                double gh = 0.5 * last_ffn_pregelu_(t, j) *
                            (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                                              (last_ffn_pregelu_(t, j) +
                                               0.044715 * last_ffn_pregelu_(t, j) * last_ffn_pregelu_(t, j) * last_ffn_pregelu_(t, j))));
                ffn_hidden(t, j) = gh;
                d_ffn_hidden(t, j) += g * W2(k, j);
                grad_W2(k, j) += g * gh;
            }
        }
    }

    // d_ffn_hidden → d_ffn_pregelu via GELU derivative
    Tensor d_ffn_pregelu(last_ffn_pregelu_.rows, last_ffn_pregelu_.cols);
    for (size_t i = 0; i < last_ffn_pregelu_.rows; ++i) {
        for (size_t j = 0; j < last_ffn_pregelu_.cols; ++j) {
            double x = last_ffn_pregelu_(i, j);
            double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
            double t = std::tanh(inner);
            double sech2 = 1.0 - t * t;
            double dinner = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
            double gelu_deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner;
            d_ffn_pregelu(i, j) = d_ffn_hidden(i, j) * gelu_deriv;
        }
    }

    // ffn_pregelu = last_ln2_out_ @ W1^T + b1
    // dW1[k, j] += sum_t d_ffn_pregelu[t, k] * last_ln2_out_[t, j]
    // db1[k]    += sum_t d_ffn_pregelu[t, k]
    // d_ln2_out[t, j] = sum_k d_ffn_pregelu[t, k] * W1[k, j]
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    Tensor d_ln2_out(d_resid1.rows, d_resid1.cols);
    d_ln2_out.fill(0.0);
    for (size_t t = 0; t < last_ffn_pregelu_.rows; ++t) {
        for (size_t k = 0; k < 2 * d_model_; ++k) {
            double g = d_ffn_pregelu(t, k);
            grad_b1(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W1(k, j) += g * last_ln2_out_(t, j);
                d_ln2_out(t, j) += g * W1(k, j);
            }
        }
    }

    // d_ln2_out is the gradient w.r.t. last_ln2_out_ (the OUTPUT of ln2).
    // To get the contribution to d_(last_resid1_) (the INPUT of ln2), we
    // chain through ln2.backward.
    Tensor d_resid1_from_ffn = ln2.backward(d_ln2_out, 0.0);

    // The d_resid1_from_ffn adds to d_resid1 (the FFN contributes to last_resid1_)
    for (size_t i = 0; i < d_resid1.data.size(); ++i) {
        d_resid1.data[i] += d_resid1_from_ffn.data[i];
    }

    // --- Attention backward ---
    // d_ln2_out flows into ln2.backward → d_resid1
    // d_resid1 is split: residual path → d_x_from_resid1 = d_resid1
    //                    attention path → d_attn_out = d_resid1 (the attention branch)
    Tensor d_attn_out = d_resid1.clone();
    Tensor d_x_from_resid = d_resid1.clone();
    Tensor d_ln1_out = attn.backward(d_attn_out, 0.0);
    Tensor d_x_from_ln1 = ln1.backward(d_ln1_out, 0.0);

    // Combine: d_x = d_x_from_resid + d_x_from_ln1 (both paths contribute)
    Tensor d_x(last_x_.rows, last_x_.cols);
    d_x.fill(0.0);
    for (size_t i = 0; i < d_x.data.size(); ++i) {
        d_x.data[i] = d_x_from_resid.data[i] + d_x_from_ln1.data[i];
    }

    return d_x;
}

void BigBirdBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);

    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.data.size(); ++i)
            W.data[i] -= learning_rate * gW.data[i];
    };
    sgd(W1, grad_W1);
    for (size_t i = 0; i < b1.data.size(); ++i) b1.data[i] -= learning_rate * grad_b1.data[i];
    sgd(W2, grad_W2);
    for (size_t i = 0; i < b2.data.size(); ++i) b2.data[i] -= learning_rate * grad_b2.data[i];
}

void BigBirdBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

std::vector<Tensor*> BigBirdBlock::parameters() {
    auto p = attn.parameters();
    auto l1 = ln1.parameters();
    auto l2 = ln2.parameters();
    for (auto* x : l1) p.push_back(x);
    for (auto* x : l2) p.push_back(x);
    p.push_back(&W1); p.push_back(&b1);
    p.push_back(&W2); p.push_back(&b2);
    return p;
}
std::vector<Tensor*> BigBirdBlock::gradients() {
    auto g = attn.gradients();
    auto l1 = ln1.gradients();
    auto l2 = ln2.gradients();
    for (auto* x : l1) g.push_back(x);
    for (auto* x : l2) g.push_back(x);
    g.push_back(&grad_W1); g.push_back(&grad_b1);
    g.push_back(&grad_W2); g.push_back(&grad_b2);
    return g;
}

// ============================================================================
// BigBirdModel
// ============================================================================
BigBirdModel::BigBirdModel(size_t d_model, size_t seq_len, size_t n_blocks, size_t out_dim,
                           size_t window_size, size_t num_random, size_t num_global,
                           unsigned seed)
    : d_model_(d_model),
      n_blocks_(n_blocks),
      out_dim_(out_dim),
      W_out_(out_dim, d_model), b_out_(1, out_dim),
      grad_W_out_(out_dim, d_model), grad_b_out_(1, out_dim)
{
    // Build the blocks
    for (size_t i = 0; i < n_blocks_; ++i) {
        // Each block gets a unique seed so the masks differ between blocks
        unsigned blk_seed = seed ^ (static_cast<unsigned>(i) * 0x9E3779B1u);
        blocks_.emplace_back(std::make_unique<BigBirdBlock>(
            d_model, seq_len, window_size, num_random, num_global, blk_seed));
    }
    // Classifier head init
    bigbird_global_rng().seed(seed ^ 0xBADC0DEu);
    std::normal_distribution<> dis(0.0, std::sqrt(2.0 / (d_model + out_dim)));
    for (size_t i = 0; i < W_out_.rows; ++i)
        for (size_t j = 0; j < W_out_.cols; ++j)
            W_out_(i, j) = dis(bigbird_global_rng());
    b_out_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
    block_outputs_.reserve(n_blocks_ + 1);
}

Tensor BigBirdModel::forward(const Tensor& input) {
    block_outputs_.clear();
    block_outputs_.push_back(input);

    Tensor cur = input;
    for (auto& blk : blocks_) {
        cur = blk->forward(cur);
        block_outputs_.push_back(cur);
    }

    // Per-token classifier: logits = cur @ W_out^T + b_out
    last_logits_ = cur * W_out_.transpose();
    for (size_t i = 0; i < last_logits_.rows; ++i)
        for (size_t j = 0; j < out_dim_; ++j)
            last_logits_(i, j) += b_out_(0, j);
    return last_logits_;
}

Tensor BigBirdModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    // d_logits = grad_output (n, out_dim)
    // cur = block_outputs_.back()  (n, d_model)
    // d_cur_from_head[t, j] = sum_k grad_output[t, k] * W_out_[k, j]
    // dW_out_[k, j] += sum_t grad_output[t, k] * cur[t, j]
    // db_out_[k]    += sum_t grad_output[t, k]
    const Tensor& cur = block_outputs_.back();
    Tensor d_cur(grad_output.rows, grad_output.cols);  // placeholder
    d_cur = Tensor(grad_output.rows, d_model_);
    d_cur.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
    for (size_t t = 0; t < grad_output.rows; ++t) {
        for (size_t k = 0; k < out_dim_; ++k) {
            double g = grad_output(t, k);
            grad_b_out_(0, k) += g;
            for (size_t j = 0; j < d_model_; ++j) {
                d_cur(t, j) += g * W_out_(k, j);
                grad_W_out_(k, j) += g * cur(t, j);
            }
        }
    }

    // Backward through the blocks (reverse order)
    Tensor d_through = d_cur;
    for (size_t i = n_blocks_; i > 0; --i) {
        d_through = blocks_[i - 1]->backward(d_through, 0.0);
    }
    return d_through;
}

void BigBirdModel::update_weights(double learning_rate) {
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    for (size_t i = 0; i < W_out_.data.size(); ++i)
        W_out_.data[i] -= learning_rate * grad_W_out_.data[i];
    for (size_t i = 0; i < b_out_.data.size(); ++i)
        b_out_.data[i] -= learning_rate * grad_b_out_.data[i];
}

void BigBirdModel::zero_grad() {
    for (auto& blk : blocks_) blk->zero_grad();
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
}

std::vector<Tensor*> BigBirdModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        for (auto* x : bp) p.push_back(x);
    }
    p.push_back(&W_out_);
    p.push_back(&b_out_);
    return p;
}
std::vector<Tensor*> BigBirdModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        for (auto* x : bg) g.push_back(x);
    }
    g.push_back(&grad_W_out_);
    g.push_back(&grad_b_out_);
    return g;
}
