#include "griffin.h"
#include "../normalization/layer_norm.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// Griffin Hybrid Block implementation
// ----------------------------------------------------------------------------
//
// Forward:
//   x_ln = LN(x)                       (T, d) — shared pre-norm
//   hawk_out = Hawk(x_ln)              (T, d) — gated linear recurrence
//   attn_out = LocalAttn(x_ln)         (T, d) — local windowed attention
//   mlp_out  = GriffinMLP(x_ln)        (T, d) — GELU FFN
//   out = x + hawk_out + attn_out + mlp_out
//
// Backward:
//   In parallel composition, the three branches each receive grad_out as
//   their output gradient:
//     grad_hawk_out = grad_attn_out = grad_mlp_out = grad_out
//   Each branch returns grad_x_ln_for_branch via its backward. The
//   LayerNorm receives the SUM of those three gradients and returns the
//   gradient w.r.t. x via LN's own backward. The residual term contributes
//   +grad_out to the input gradient.
//
// All math done in (T, d) layout. The Hawk branch requires (T, d) layout
// directly (matches HawkBlock). The local-attention branch and MLP branch
// also operate in (T, d) layout.
// ============================================================================

// ----------------------------------------------------------------------------
// LocalSlidingWindowAttention
// ----------------------------------------------------------------------------
//
// Single-head causal attention with a sliding window of size `window_size`.
// For each time step t, the query at t attends to keys at
//   s ∈ [max(0, t - window_size + 1), t]
// via softmax. Output:
//   Q = X @ W_q + b_q          (T, d)
//   K = X @ W_k + b_k          (T, d)
//   V = X @ W_v + b_v          (T, d)
//   for each t:
//     window = [max(0, t - w + 1), t]
//     scores[t, s] = Q[t] · K[s] / sqrt(d)         for s in window
//     weights[t, s] = softmax(scores[t, :])
//     attn_out[t] = sum_s weights[t, s] * V[s]
//   out = attn_out @ W_o + b_o

LocalSlidingWindowAttention::LocalSlidingWindowAttention(size_t d, size_t window_size)
    : d_(d), window_size_(window_size),
      W_q(d, d), W_k(d, d), W_v(d, d), W_o(d, d) {
    if (d == 0) throw std::invalid_argument("LocalSlidingWindowAttention: d must be > 0");
    if (window_size == 0) throw std::invalid_argument("LocalSlidingWindowAttention: window_size must be > 0");

    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
}

Tensor LocalSlidingWindowAttention::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("LocalSlidingWindowAttention::forward: input cols must be d");
    }
    const size_t T = input.rows;

    // Project Q, K, V, then output projection is done at the end of forward.
    last_q = W_q.forward(input);  // (T, d)
    last_k = W_k.forward(input);  // (T, d)
    last_v = W_v.forward(input);  // (T, d)

    // Cache attention weights for backward (T rows × up to w_max cols; padded).
    last_attn = Tensor(T, window_size_);

    const double scale = 1.0 / std::sqrt(static_cast<double>(d_));
    Tensor attn_out(T, d_);
    for (size_t t = 0; t < T; ++t) {
        const size_t s_start = (t + 1 >= window_size_) ? (t + 1 - window_size_) : 0;
        const size_t s_end = t;  // inclusive
        const size_t wlen = s_end - s_start + 1;

        // Compute scores[s] = Q[t] · K[s] for s in [s_start, s_end]
        // and apply scaled softmax.
        Tensor scores(1, wlen);
        for (size_t k = 0; k < wlen; ++k) {
            const size_t s = s_start + k;
            double dot = 0.0;
            for (size_t i = 0; i < d_; ++i) dot += last_q[t][i] * last_k[s][i];
            scores[0][k] = dot * scale;
        }
        // softmax
        double max_s = scores[0][0];
        for (size_t k = 1; k < wlen; ++k)
            if (scores[0][k] > max_s) max_s = scores[0][k];
        double sum = 0.0;
        for (size_t k = 0; k < wlen; ++k) {
            scores[0][k] = std::exp(scores[0][k] - max_s);
            sum += scores[0][k];
        }
        for (size_t k = 0; k < wlen; ++k) {
            scores[0][k] /= sum;
            // Cache into last_attn (padded with 0 beyond wlen)
            last_attn[t][k] = scores[0][k];
        }
        for (size_t k = wlen; k < window_size_; ++k) {
            last_attn[t][k] = 0.0;
        }

        // Compute attn_out[t] = sum_k scores[k] * V[s_start + k]
        for (size_t i = 0; i < d_; ++i) {
            double s_v = 0.0;
            for (size_t k = 0; k < wlen; ++k) {
                s_v += scores[0][k] * last_v[s_start + k][i];
            }
            attn_out[t][i] = s_v;
        }
    }

    // Output projection: out = attn_out @ W_o + b_o
    return W_o.forward(attn_out);
}

Tensor LocalSlidingWindowAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("LocalSlidingWindowAttention::backward: grad cols must be d");
    }
    const size_t T = grad_output.rows;

    // Step 1: backward through W_o output projection: grad_attn_pre (T, d)
    Tensor grad_attn_pre = W_o.backward(grad_output, 0.0);

    // Step 2: backward through the attention itself.
    // For each time t:
    //   grad_V[s] for s in window += grad_attn_pre[t] * weights[t, s-t_start]
    //   grad_weights[t, k] = sum_i grad_attn_pre[t][i] * V[s_t+k][i]
    //   For softmax backward with window-local scores s_k = Q[t]·K[s_t+k]/sqrt(d):
    //     grad_scores[k] = sum_j weights[k]*(δ_{kj} - weights[j]) * grad_weights[k]
    //                  = weights[k] * (grad_weights[k] - sum_j weights[j]*grad_weights[j])
    //   Then grad_Q[t][i] += sum_k grad_scores[k] * K[s_t+k][i] / sqrt(d)
    //        grad_K[s][i] += grad_scores[k] * Q[t][i] / sqrt(d)    (accumulated for each t where s is in t's window)
    Tensor grad_q(T, d_);
    Tensor grad_k(T, d_);
    Tensor grad_v(T, d_);

    const double scale = 1.0 / std::sqrt(static_cast<double>(d_));

    for (size_t t = 0; t < T; ++t) {
        const size_t s_start = (t + 1 >= window_size_) ? (t + 1 - window_size_) : 0;
        const size_t s_end = t;
        const size_t wlen = s_end - s_start + 1;

        // Recompute weights from last_attn (already normalized).
        // Re-read (don't recompute via softmax to avoid float drift; last_attn
        // is exactly what forward produced).
        Tensor weights(1, wlen);
        for (size_t k = 0; k < wlen; ++k) weights[0][k] = last_attn[t][k];

        // grad_weights[k] = sum_i grad_attn_pre[t][i] * V[s_start+k][i]
        Tensor grad_weights(1, wlen);
        for (size_t k = 0; k < wlen; ++k) {
            double g = 0.0;
            const size_t s = s_start + k;
            for (size_t i = 0; i < d_; ++i)
                g += grad_attn_pre[t][i] * last_v[s][i];
            grad_weights[0][k] = g;
            // grad_V[s] += weights[k] * grad_attn_pre[t]
            for (size_t i = 0; i < d_; ++i)
                grad_v[s][i] += weights[0][k] * grad_attn_pre[t][i];
        }

        // Softmax backward: grad_scores[k] = weights[k] * (grad_weights[k] - Σ_j weights[j] * grad_weights[j])
        double dot_sum = 0.0;
        for (size_t k = 0; k < wlen; ++k)
            dot_sum += weights[0][k] * grad_weights[0][k];

        Tensor grad_scores(1, wlen);
        for (size_t k = 0; k < wlen; ++k)
            grad_scores[0][k] = weights[0][k] * (grad_weights[0][k] - dot_sum);

        // grad_Q[t][i] += sum_k grad_scores[k] * K[s_start+k][i] * scale
        for (size_t k = 0; k < wlen; ++k) {
            const size_t s = s_start + k;
            const double gsk = grad_scores[0][k] * scale;
            for (size_t i = 0; i < d_; ++i)
                grad_q[t][i] += gsk * last_k[s][i];
            // grad_K[s][i] += grad_scores[k] * Q[t][i] * scale
            for (size_t i = 0; i < d_; ++i)
                grad_k[s][i] += grad_scores[0][k] * last_q[t][i] * scale;
        }
    }

    // Step 3: backward through the Q/K/V projections. Each is a Dense on the
    // original input `last_input` of Dense (we didn't cache it explicitly for
    // each Dense, but Dense's own backward uses its internal cache).
    Tensor grad_via_q = W_q.backward(grad_q, 0.0);
    Tensor grad_via_k = W_k.backward(grad_k, 0.0);
    Tensor grad_via_v = W_v.backward(grad_v, 0.0);

    // Sum the three projections' gradients (each branch projects from input X
    // through its own W, but they share the same input).
    Tensor grad_input(T, d_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_; ++i)
            grad_input[t][i] = grad_via_q[t][i] + grad_via_k[t][i] + grad_via_v[t][i];

    return grad_input;
}

void LocalSlidingWindowAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
}

void LocalSlidingWindowAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
}

std::vector<Tensor*> LocalSlidingWindowAttention::parameters() {
    return {&W_q.weights, &W_q.bias, &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias, &W_o.weights, &W_o.bias};
}

std::vector<Tensor*> LocalSlidingWindowAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias, &W_k.grad_weights, &W_k.grad_bias,
            &W_v.grad_weights, &W_v.grad_bias, &W_o.grad_weights, &W_o.grad_bias};
}

// ----------------------------------------------------------------------------
// GriffinMLP — 2-layer GELU FFN
// ----------------------------------------------------------------------------

GriffinMLP::GriffinMLP(size_t d, size_t mult)
    : d_(d), mult_(mult), hidden_(mult * d),
      W1(d, hidden_), W2(hidden_, d) {
    if (d == 0) throw std::invalid_argument("GriffinMLP: d must be > 0");
    if (mult == 0) throw std::invalid_argument("GriffinMLP: mult must be > 0");

    W1.zero_grad();
    W2.zero_grad();
}

static inline double gelu(double x) {
    // Exact GELU using erf: 0.5*x*(1 + erf(x/sqrt(2)))
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static inline double gelu_deriv(double x) {
    // d/dx GELU(x) = 0.5*(1 + erf(x/sqrt(2))) + x * phi(x)
    // where phi(x) = (1/sqrt(2π)) exp(-x²/2)
    constexpr double INV_SQRT_2PI = 0.3989422804014327;  // 1/sqrt(2π)
    const double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    const double pdf = INV_SQRT_2PI * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

Tensor GriffinMLP::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("GriffinMLP::forward: input cols must be d");
    }
    const size_t T = input.rows;
    last_input = input;
    last_hidden = W1.forward(input);  // (T, hidden)
    last_act = Tensor(T, hidden_);
    for (size_t t = 0; t < T; ++t)
        for (size_t h = 0; h < hidden_; ++h)
            last_act[t][h] = gelu(last_hidden[t][h]);
    return W2.forward(last_act);  // (T, d)
}

Tensor GriffinMLP::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("GriffinMLP::backward: grad cols must be d");
    }
    const size_t T = grad_output.rows;

    // Step 1: backward through W2: grad_act (T, hidden)
    Tensor grad_act = W2.backward(grad_output, 0.0);

    // Step 2: backward through GELU: grad_hidden (T, hidden)
    Tensor grad_hidden(T, hidden_);
    for (size_t t = 0; t < T; ++t)
        for (size_t h = 0; h < hidden_; ++h)
            grad_hidden[t][h] = grad_act[t][h] * gelu_deriv(last_hidden[t][h]);

    // Step 3: backward through W1: grad_input (T, d)
    return W1.backward(grad_hidden, 0.0);
}

void GriffinMLP::update_weights(double learning_rate) {
    W1.update_weights(learning_rate);
    W2.update_weights(learning_rate);
}

void GriffinMLP::zero_grad() {
    W1.zero_grad();
    W2.zero_grad();
}

std::vector<Tensor*> GriffinMLP::parameters() {
    return {&W1.weights, &W1.bias, &W2.weights, &W2.bias};
}

std::vector<Tensor*> GriffinMLP::gradients() {
    return {&W1.grad_weights, &W1.grad_bias, &W2.grad_weights, &W2.grad_bias};
}

// ----------------------------------------------------------------------------
// GriffinBlock — the full hybrid
// ----------------------------------------------------------------------------

GriffinBlock::GriffinBlock(size_t d_model, size_t num_heads, size_t window_size, size_t ffn_mult)
    : d_(d_model), num_heads_(num_heads),
      window_size_(window_size), ffn_mult_(ffn_mult),
      ln(d_model), hawk(d_model), attn(d_model, window_size), mlp(d_model, ffn_mult) {
    if (d_model == 0) throw std::invalid_argument("GriffinBlock: d_model must be > 0");
    if (num_heads == 0) throw std::invalid_argument("GriffinBlock: num_heads must be > 0");
    if (d_model % num_heads != 0) throw std::invalid_argument("GriffinBlock: d_model % num_heads must be 0");
    if (window_size == 0) throw std::invalid_argument("GriffinBlock: window_size must be > 0");
    if (ffn_mult == 0) throw std::invalid_argument("GriffinBlock: ffn_mult must be > 0");
}

Tensor GriffinBlock::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("GriffinBlock::forward: input cols must be d_model");
    }

    last_ln_input = input;
    last_ln_out = ln.forward(input);  // (T, d) — shared pre-norm

    // Three parallel branches on the same LN'd input
    last_hawk_out = hawk.forward(last_ln_out);
    last_attn_out = attn.forward(last_ln_out);
    last_mlp_out  = mlp.forward(last_ln_out);

    // Residual sum
    Tensor output(input.rows, d_);
    for (size_t t = 0; t < input.rows; ++t)
        for (size_t i = 0; i < d_; ++i)
            output[t][i] = input[t][i] + last_hawk_out[t][i] + last_attn_out[t][i] + last_mlp_out[t][i];

    return output;
}

Tensor GriffinBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("GriffinBlock::backward: grad cols must be d_model");
    }
    const size_t T = grad_output.rows;

    // Each branch receives grad_output as its output gradient (parallel).
    Tensor grad_hawk_branch = hawk.backward(grad_output, 0.0);  // (T, d)
    Tensor grad_attn_branch = attn.backward(grad_output, 0.0);  // (T, d)
    Tensor grad_mlp_branch  = mlp.backward(grad_output, 0.0);   // (T, d)

    // Sum into the LayerNorm input gradient
    Tensor grad_ln_in(T, d_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_; ++i)
            grad_ln_in[t][i] = grad_hawk_branch[t][i] + grad_attn_branch[t][i] + grad_mlp_branch[t][i];

    // LN backward: returns grad wrx x = grad_ln_input
    Tensor grad_x_via_ln = ln.backward(grad_ln_in, 0.0);

    // Add residual term
    Tensor grad_input(T, d_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < d_; ++i)
            grad_input[t][i] = grad_output[t][i] + grad_x_via_ln[t][i];

    return grad_input;
}

void GriffinBlock::update_weights(double learning_rate) {
    ln.update_weights(learning_rate);
    hawk.update_weights(learning_rate);
    attn.update_weights(learning_rate);
    mlp.update_weights(learning_rate);
}

void GriffinBlock::zero_grad() {
    ln.zero_grad();
    hawk.zero_grad();
    attn.zero_grad();
    mlp.zero_grad();
}

std::vector<Tensor*> GriffinBlock::parameters() {
    std::vector<Tensor*> p;
    auto ln_p = ln.parameters();
    auto hawk_p = hawk.parameters();
    auto attn_p = attn.parameters();
    auto mlp_p = mlp.parameters();
    p.insert(p.end(), ln_p.begin(), ln_p.end());
    p.insert(p.end(), hawk_p.begin(), hawk_p.end());
    p.insert(p.end(), attn_p.begin(), attn_p.end());
    p.insert(p.end(), mlp_p.begin(), mlp_p.end());
    return p;
}

std::vector<Tensor*> GriffinBlock::gradients() {
    std::vector<Tensor*> g;
    auto ln_g = ln.gradients();
    auto hawk_g = hawk.gradients();
    auto attn_g = attn.gradients();
    auto mlp_g = mlp.gradients();
    g.insert(g.end(), ln_g.begin(), ln_g.end());
    g.insert(g.end(), hawk_g.begin(), hawk_g.end());
    g.insert(g.end(), attn_g.begin(), attn_g.end());
    g.insert(g.end(), mlp_g.begin(), mlp_g.end());
    return g;
}

void GriffinBlock::copy_params_from(const GriffinBlock& other) {
    if (other.d_ != d_) throw std::invalid_argument("GriffinBlock::copy_params_from: d_model mismatch");
    ln.gamma = other.ln.gamma;
    ln.beta = other.ln.beta;
    hawk.W_x.weights = other.hawk.W_x.weights;
    hawk.W_x.bias = other.hawk.W_x.bias;
    hawk.log_a_raw = other.hawk.log_a_raw;
    hawk.W_o.weights = other.hawk.W_o.weights;
    hawk.W_o.bias = other.hawk.W_o.bias;
    attn.W_q.weights = other.attn.W_q.weights;
    attn.W_q.bias = other.attn.W_q.bias;
    attn.W_k.weights = other.attn.W_k.weights;
    attn.W_k.bias = other.attn.W_k.bias;
    attn.W_v.weights = other.attn.W_v.weights;
    attn.W_v.bias = other.attn.W_v.bias;
    attn.W_o.weights = other.attn.W_o.weights;
    attn.W_o.bias = other.attn.W_o.bias;
    mlp.W1.weights = other.mlp.W1.weights;
    mlp.W1.bias = other.mlp.W1.bias;
    mlp.W2.weights = other.mlp.W2.weights;
    mlp.W2.bias = other.mlp.W2.bias;
}

// ----------------------------------------------------------------------------
// GriffinModel — multi-block stack
// ----------------------------------------------------------------------------

GriffinModel::GriffinModel(size_t input_dim, size_t d_model, size_t output_dim,
                            size_t num_layers, size_t num_heads,
                            size_t window_size, size_t ffn_mult)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_layers_(num_layers), num_heads_(num_heads),
      window_size_(window_size), ffn_mult_(ffn_mult),
      embed(input_dim, d_model), final_ln(d_model), classifier(d_model, output_dim) {
    if (input_dim == 0 || d_model == 0 || output_dim == 0)
        throw std::invalid_argument("GriffinModel: input/d_model/output_dim must be > 0");
    if (num_layers == 0) throw std::invalid_argument("GriffinModel: num_layers must be > 0");
    if (num_heads == 0) throw std::invalid_argument("GriffinModel: num_heads must be > 0");
    if (d_model % num_heads != 0) throw std::invalid_argument("GriffinModel: d_model % num_heads must be 0");

    blocks.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks.push_back(std::make_unique<GriffinBlock>(
            d_model, num_heads, window_size, ffn_mult));
    }
}

Tensor GriffinModel::forward(const Tensor& input) {
    // Per-token embed then stack
    Tensor h = embed.forward(input);  // (T, d_model)
    for (auto& blk : blocks) {
        h = blk->forward(h);
    }
    h = final_ln.forward(h);
    return classifier.forward(h);
}

Tensor GriffinModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor g = classifier.backward(grad_output, 0.0);
    g = final_ln.backward(g, 0.0);
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
        g = blocks[i]->backward(g, 0.0);
    }
    return embed.backward(g, 0.0);
}

void GriffinModel::update_weights(double learning_rate) {
    embed.update_weights(learning_rate);
    for (auto& blk : blocks) blk->update_weights(learning_rate);
    final_ln.update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}

void GriffinModel::zero_grad() {
    embed.zero_grad();
    for (auto& blk : blocks) blk->zero_grad();
    final_ln.zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> GriffinModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&embed.weights);
    p.push_back(&embed.bias);
    for (auto& blk : blocks) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&final_ln.gamma);
    p.push_back(&final_ln.beta);
    p.push_back(&classifier.weights);
    p.push_back(&classifier.bias);
    return p;
}

std::vector<Tensor*> GriffinModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&embed.grad_weights);
    g.push_back(&embed.grad_bias);
    for (auto& blk : blocks) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&final_ln.grad_gamma_);
    g.push_back(&final_ln.grad_beta_);
    g.push_back(&classifier.grad_weights);
    g.push_back(&classifier.grad_bias);
    return g;
}