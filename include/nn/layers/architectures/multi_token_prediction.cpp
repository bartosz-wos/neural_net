#include "multi_token_prediction.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// ----------------------------------------------------------------------------
// MultiTokenPredictionHead
// ----------------------------------------------------------------------------

MultiTokenPredictionHead::MultiTokenPredictionHead(size_t d_model, size_t vocab_size,
                                                   size_t n_future)
    : d_model_(d_model), vocab_(vocab_size), n_future_(n_future)
{
    if (d_model == 0)   throw std::invalid_argument("MultiTokenPredictionHead: d_model must be > 0");
    if (vocab_size == 0) throw std::invalid_argument("MultiTokenPredictionHead: vocab_size must be > 0");
    if (n_future == 0)   throw std::invalid_argument("MultiTokenPredictionHead: n_future must be > 0");

    W_.resize(n_future_);
    b_.resize(n_future_);
    grad_W_.resize(n_future_);
    grad_b_.resize(n_future_);
    last_logits_.resize(n_future_);
    last_p_.resize(n_future_);
    last_N_valid_.resize(n_future_, 0);

    // Small random init for W (Xavier-ish), zero bias. n_heads * (vocab, d_model) is a lot
    // of params; the small init keeps logits close to zero, softmax close to uniform,
    // and CE close to log(vocab) — a sensible starting point.
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(d_model)));
    for (size_t k = 0; k < n_future_; ++k) {
        W_[k]       = Tensor::zeros(vocab_size, d_model);
        b_[k]       = Tensor::zeros(1, vocab_size);
        grad_W_[k]  = Tensor::zeros(vocab_size, d_model);
        grad_b_[k]  = Tensor::zeros(1, vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
            for (size_t j = 0; j < d_model; ++j)
                W_[k](i, j) = dist(rng);
    }
}

double MultiTokenPredictionHead::forward(const Tensor& h, const Tensor& targets) {
    const size_t N = h.rows;
    if (h.cols != d_model_)
        throw std::invalid_argument("MultiTokenPredictionHead::forward: h.cols != d_model");
    if (targets.rows != N || targets.cols != n_future_)
        throw std::invalid_argument("MultiTokenPredictionHead::forward: targets shape "
                                     "must be (N, n_future)");

    last_h_ = h.clone();              // (N, d_model)
    last_targets_ = targets.clone();  // (N, n_future)

    double L = 0.0;
    for (size_t k = 0; k < n_future_; ++k) {
        // 1) logits = h @ W_k^T + b_k -> (N, vocab)
        Tensor logits(N, vocab_);
        for (size_t t = 0; t < N; ++t)
            for (size_t v = 0; v < vocab_; ++v) {
                double s = 0.0;
                for (size_t c = 0; c < d_model_; ++c) s += h(t, c) * W_[k](v, c);
                logits(t, v) = s + b_[k](0, v);
            }
        last_logits_[k] = logits;

        // 2) softmax over vocab axis, then masked CE.
        // mask_k[t] = 1 if t + k < N (we have a target at position t + k), else 0.
        const size_t target_offset = k;  // matches the queue entry's "predict token t+1..t+n"
        Tensor p(N, vocab_);
        size_t N_valid = 0;
        double L_k = 0.0;
        for (size_t t = 0; t < N; ++t) {
            const size_t target_pos = t + target_offset;
            if (target_pos >= N) {
                // No valid target — emit uniform probs (won't contribute to loss; the
                // mask will be applied in backward too). For numerical cleanliness
                // we still run softmax.
                double m = logits(t, 0);
                for (size_t v = 1; v < vocab_; ++v) m = std::max(m, logits(t, v));
                double z = 0.0;
                for (size_t v = 0; v < vocab_; ++v) { p(t, v) = std::exp(logits(t, v) - m); z += p(t, v); }
                for (size_t v = 0; v < vocab_; ++v) p(t, v) /= z;
                continue;
            }
            ++N_valid;
            // Numerically stable softmax
            double m = logits(t, 0);
            for (size_t v = 1; v < vocab_; ++v) m = std::max(m, logits(t, v));
            double z = 0.0;
            for (size_t v = 0; v < vocab_; ++v) { p(t, v) = std::exp(logits(t, v) - m); z += p(t, v); }
            for (size_t v = 0; v < vocab_; ++v) p(t, v) /= z;

            const double target_idx_d = targets(t, k);
            const long target_idx = static_cast<long>(target_idx_d);
            if (target_idx < 0 || static_cast<size_t>(target_idx) >= vocab_)
                throw std::invalid_argument("MultiTokenPredictionHead::forward: target index "
                                             "out of range for vocab_size");
            const double logp_t = std::log(p(t, static_cast<size_t>(target_idx)));
            L_k += -logp_t;
        }
        last_p_[k] = p;
        last_N_valid_[k] = N_valid;

        if (N_valid > 0) L_k /= static_cast<double>(N_valid);
        L += L_k;
    }
    return L;
}

Tensor MultiTokenPredictionHead::backward(double /*learning_rate*/) {
    if (last_logits_.empty())
        throw std::logic_error("MultiTokenPredictionHead::backward called before forward");

    const size_t N = last_h_.rows;

    // 1) Per-head d_logits_k: (N, vocab) with mask and (1 / N_valid_k) scale.
    std::vector<Tensor> d_logits(n_future_);
    for (size_t k = 0; k < n_future_; ++k) {
        const size_t target_offset = k;
        const double N_valid_d = static_cast<double>(last_N_valid_[k]);
        Tensor dl(N, vocab_);
        const double scale = (last_N_valid_[k] > 0) ? (1.0 / N_valid_d) : 0.0;
        for (size_t t = 0; t < N; ++t) {
            const size_t target_pos = t + target_offset;
            if (target_pos >= N) {
                // masked
                for (size_t v = 0; v < vocab_; ++v) dl(t, v) = 0.0;
                continue;
            }
            const double target_idx_d = last_targets_(t, k);
            const long target_idx = static_cast<long>(target_idx_d);
            for (size_t v = 0; v < vocab_; ++v) {
                double d = scale * last_p_[k](t, v);
                if (static_cast<size_t>(target_idx) == v) d -= scale;
                dl(t, v) = d;
            }
        }
        d_logits[k] = dl;
    }

    // 2) Per-head grad_W_k (vocab, d_model) and grad_b_k (1, vocab).
    for (size_t k = 0; k < n_future_; ++k) {
        grad_W_[k].fill(0.0);
        grad_b_[k].fill(0.0);
        for (size_t v = 0; v < vocab_; ++v) {
            for (size_t c = 0; c < d_model_; ++c) {
                double g = 0.0;
                for (size_t t = 0; t < N; ++t) g += d_logits[k](t, v) * last_h_(t, c);
                grad_W_[k](v, c) = g;
            }
            double gb = 0.0;
            for (size_t t = 0; t < N; ++t) gb += d_logits[k](t, v);
            grad_b_[k](0, v) = gb;
        }
    }

    // 3) d_trunk: SUM of per-head contributions.
    //   d_trunk[t, c] = sum_k sum_v d_logits_k[t, v] * W_k[v, c]
    Tensor d_trunk(N, d_model_);
    d_trunk.fill(0.0);
    for (size_t k = 0; k < n_future_; ++k) {
        for (size_t t = 0; t < N; ++t) {
            for (size_t c = 0; c < d_model_; ++c) {
                double s = 0.0;
                for (size_t v = 0; v < vocab_; ++v) s += d_logits[k](t, v) * W_[k](v, c);
                d_trunk(t, c) += s;
            }
        }
    }
    return d_trunk;
}

void MultiTokenPredictionHead::update_weights(double learning_rate) {
    for (size_t k = 0; k < n_future_; ++k) {
        for (size_t i = 0; i < W_[k].rows; ++i)
            for (size_t j = 0; j < W_[k].cols; ++j)
                W_[k](i, j) -= learning_rate * grad_W_[k](i, j);
        for (size_t j = 0; j < b_[k].cols; ++j)
            b_[k](0, j) -= learning_rate * grad_b_[k](0, j);
    }
}

void MultiTokenPredictionHead::zero_grad() {
    for (size_t k = 0; k < n_future_; ++k) {
        grad_W_[k].fill(0.0);
        grad_b_[k].fill(0.0);
    }
}

std::vector<Tensor*> MultiTokenPredictionHead::parameters() {
    std::vector<Tensor*> out;
    out.reserve(2 * n_future_);
    for (size_t k = 0; k < n_future_; ++k) {
        out.push_back(&W_[k]);
        out.push_back(&b_[k]);
    }
    return out;
}

std::vector<Tensor*> MultiTokenPredictionHead::gradients() {
    std::vector<Tensor*> out;
    out.reserve(2 * n_future_);
    for (size_t k = 0; k < n_future_; ++k) {
        out.push_back(&grad_W_[k]);
        out.push_back(&grad_b_[k]);
    }
    return out;
}

// ----------------------------------------------------------------------------
// MultiTokenPredictionModel
// ----------------------------------------------------------------------------

MultiTokenPredictionModel::MultiTokenPredictionModel(size_t input_dim, size_t d_model,
                                                     size_t vocab_size, size_t n_future)
    : input_dim_(input_dim), d_model_(d_model), vocab_(vocab_size),
      n_future_(n_future),
      input_proj_(input_dim, d_model),
      head_(d_model, vocab_size, n_future)
{
    if (input_dim == 0) throw std::invalid_argument("MultiTokenPredictionModel: input_dim must be > 0");
}

Tensor MultiTokenPredictionModel::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("MultiTokenPredictionModel::forward: input.cols != input_dim");

    last_trunk_ = input_proj_.forward(input);     // (N, d_model)
    // Run the head with DUMMY targets to populate last_logits_; the real loss
    // happens in the test (matches the NGPTModel convention: Layer.forward
    // returns logits, the test computes CE against shifted targets externally).
    Tensor dummy_targets(input.rows, n_future_);
    dummy_targets.fill(0.0);
    (void)head_.forward(last_trunk_, dummy_targets);

    last_head_logits_ = head_.last_logits()[n_future_ - 1].clone();  // deepest head's logits
    return last_head_logits_;                                         // (N, vocab)
}

Tensor MultiTokenPredictionModel::backward(const Tensor& grad_output, double learning_rate) {
    // gradient flows through the trunk
    Tensor d_trunk(head_.backward(learning_rate));   // (N, d_model)
    // d_trunk carries the summed CE gradient; we ignore grad_output for the logits
    // (the model doesn't actually use them in this variant — they exist only so the
    //  Layer contract holds).
    (void)grad_output;
    return input_proj_.backward(d_trunk, learning_rate);
}

void MultiTokenPredictionModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    head_.update_weights(learning_rate);
}

void MultiTokenPredictionModel::zero_grad() {
    input_proj_.zero_grad();
    head_.zero_grad();
}

std::vector<Tensor*> MultiTokenPredictionModel::parameters() {
    auto out = head_.parameters();
    auto ip = input_proj_.parameters();
    out.insert(out.end(), ip.begin(), ip.end());
    return out;
}

std::vector<Tensor*> MultiTokenPredictionModel::gradients() {
    auto out = head_.gradients();
    auto ip = input_proj_.gradients();
    out.insert(out.end(), ip.begin(), ip.end());
    return out;
}
