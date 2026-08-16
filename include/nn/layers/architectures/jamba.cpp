#include "jamba.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Kaiming-He style initialization for Dense weights. We assign a fresh
// weights / grad_weights / bias / grad_bias into the existing Dense
// instances that were constructed in the JambaBlock initializer list.
void init_dense(Dense& d, size_t in_f, size_t out_f, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / static_cast<double>(in_f)));
    std::vector<double> w(in_f * out_f);
    for (auto& v : w) v = nd(rng);
    d.weights = Tensor(out_f, in_f, w.data());
    d.bias = Tensor(1, out_f);
    std::fill(d.bias.data.begin(), d.bias.data.end(), 0.0);
    d.grad_weights = Tensor(out_f, in_f);
    std::fill(d.grad_weights.data.begin(), d.grad_weights.data.end(), 0.0);
    d.grad_bias = Tensor(1, out_f);
    std::fill(d.grad_bias.data.begin(), d.grad_bias.data.end(), 0.0);
}

// GELU element-wise (closed-form using erf).
double gelu(double x) {
    return x * 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// GELU derivative for backward pass.
double gelu_deriv(double x) {
    double s = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    return s + x * pdf;
}

}  // namespace

// =====================================================================
// JambaBlock
// =====================================================================
JambaBlock::JambaBlock(size_t d_model, size_t num_heads,
                       size_t num_experts, size_t top_k,
                       size_t d_state, size_t moe_every_n, bool use_moe)
    : d_model_(d_model),
      num_heads_(num_heads),
      num_experts_(num_experts),
      top_k_(top_k),
      d_state_(d_state == 0 ? 2 * d_model : d_state),
      moe_every_n_(moe_every_n),
      use_moe_(use_moe),
      ln1_(d_model), ln2_(d_model), ln3_(d_model),
      mamba_(std::make_unique<Mamba2Block>(d_model, num_heads, d_state_)),
      attn_(std::make_unique<MultiHeadAttention>(d_model, num_heads)),
      w1_(d_model, 4 * d_model), b1_(d_model, 4 * d_model),
      w2_(4 * d_model, d_model), b2_(4 * d_model, d_model) {
    if (d_model == 0)
        throw std::invalid_argument("JambaBlock: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("JambaBlock: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("JambaBlock: d_model must be divisible by num_heads");
    if (moe_every_n != 1 && moe_every_n != 2)
        throw std::invalid_argument("JambaBlock: moe_every_n must be 1 or 2");
    if (top_k == 0)
        throw std::invalid_argument("JambaBlock: top_k must be > 0");
    if (num_experts > 0 && top_k > num_experts)
        throw std::invalid_argument("JambaBlock: top_k cannot exceed num_experts");

    // Initialize the dense FFN weights (only used when MoE is disabled).
    init_dense(w1_, d_model, 4 * d_model, 0xC0FFEE01);
    init_dense(w2_, 4 * d_model, d_model, 0xC0FFEE02);

    // Construct MoE if requested.
    if (use_moe && num_experts > 0) {
        moe_ffn_ = std::make_unique<MoELayer>(d_model, num_experts, top_k);
    } else {
        moe_ffn_.reset();
    }
}

Tensor JambaBlock::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("JambaBlock: input feature dim mismatch");
    last_x_ = input;

    // Step 1: pre-norm + Mamba-2 + residual
    last_ln1_ = ln1_.forward(input);
    last_mamba_out_ = mamba_->forward(last_ln1_);
    last_h1_ = input + last_mamba_out_;

    // Step 2: pre-norm + Attention + residual.
    // MultiHeadAttention expects (d_model, seq_len) input. Mamba-2 (and our
    // residual stream) carry (T, d_model). Transpose both directions.
    last_ln2_ = ln2_.forward(last_h1_);
    Tensor attn_in = last_ln2_.transpose();  // (T, d_model) -> (d_model, T)
    Tensor attn_out_attn_layout = attn_->forward(attn_in);  // (d_model, T)
    last_attn_out_ = attn_out_attn_layout.transpose();  // back to (T, d_model)
    last_h2_ = last_h1_ + last_attn_out_;

    // Step 3: pre-norm + FFN + residual
    last_ln3_ = ln3_.forward(last_h2_);
    if (uses_moe()) {
        last_ffn_out_ = moe_ffn_->forward(last_ln3_);
    } else {
        // Dense FFN: GELU(w1·x + b1) then w2·... + b2
        last_ffn_inner_ = w1_.forward(last_ln3_);
        for (size_t i = 0; i < last_ffn_inner_.rows; ++i)
            for (size_t j = 0; j < last_ffn_inner_.cols; ++j)
                last_ffn_inner_[i][j] = gelu(last_ffn_inner_[i][j]);
        last_ffn_out_ = w2_.forward(last_ffn_inner_);
    }
    return last_h2_ + last_ffn_out_;
}

Tensor JambaBlock::backward(const Tensor& grad_output, double learning_rate) {
    // Step 3 backward: residual splits grad_output into
    //   grad_h2 (passed through) + grad_ffn_out (residual).
    Tensor grad_h2 = grad_output;
    Tensor grad_ln3;
    if (uses_moe()) {
        grad_ln3 = moe_ffn_->backward(grad_output, learning_rate);
    } else {
        // Dense FFN backward: through w2 then w1 then GELU
        Tensor grad_inner = w2_.backward(grad_output, learning_rate);
        for (size_t i = 0; i < grad_inner.rows; ++i)
            for (size_t j = 0; j < grad_inner.cols; ++j)
                grad_inner[i][j] *= gelu_deriv(last_ffn_inner_[i][j]);
        grad_ln3 = w1_.backward(grad_inner, learning_rate);
    }
    Tensor grad_ln3_via_norm = ln3_.backward(grad_ln3, learning_rate);

    // Step 2 backward: attention. The Mamba-2 chain sees (T, d_model); the
    // attention sublayer expects (d_model, T). Transpose in both directions.
    Tensor grad_h1 = grad_h2 + grad_ln3_via_norm;
    Tensor grad_attn_in = grad_h1.transpose();  // (T, d_model) -> (d_model, T)
    Tensor grad_ln2_attn = attn_->backward(grad_attn_in, learning_rate);  // (d_model, T)
    Tensor grad_ln2 = grad_ln2_attn.transpose();  // back to (T, d_model)
    Tensor grad_ln2_via_norm = ln2_.backward(grad_ln2, learning_rate);

    // Step 1 backward: Mamba
    Tensor grad_input = grad_h1 + grad_ln2_via_norm;
    Tensor grad_ln1 = mamba_->backward(grad_input, learning_rate);
    Tensor grad_ln1_via_norm = ln1_.backward(grad_ln1, learning_rate);

    return grad_input + grad_ln1_via_norm;
}

void JambaBlock::update_weights(double learning_rate) {
    mamba_->update_weights(learning_rate);
    attn_->update_weights(learning_rate);
    if (uses_moe()) {
        moe_ffn_->update_weights(learning_rate);
    } else {
        w1_.update_weights(learning_rate);
        w2_.update_weights(learning_rate);
    }
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ln3_.update_weights(learning_rate);
}

void JambaBlock::zero_grad() {
    mamba_->zero_grad();
    attn_->zero_grad();
    if (uses_moe()) {
        moe_ffn_->zero_grad();
    } else {
        w1_.zero_grad();
        w2_.zero_grad();
    }
    ln1_.zero_grad();
    ln2_.zero_grad();
    ln3_.zero_grad();
}

std::vector<Tensor*> JambaBlock::parameters() {
    std::vector<Tensor*> p;
    for (auto* t : mamba_->parameters()) p.push_back(t);
    for (auto* t : attn_->parameters()) p.push_back(t);
    if (uses_moe()) {
        for (auto* t : moe_ffn_->parameters()) p.push_back(t);
    } else {
        p.push_back(&w1_.weights); p.push_back(&w1_.bias);
        p.push_back(&w2_.weights); p.push_back(&w2_.bias);
    }
    for (auto* t : ln1_.parameters()) p.push_back(t);
    for (auto* t : ln2_.parameters()) p.push_back(t);
    for (auto* t : ln3_.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> JambaBlock::gradients() {
    std::vector<Tensor*> g;
    for (auto* t : mamba_->gradients()) g.push_back(t);
    for (auto* t : attn_->gradients()) g.push_back(t);
    if (uses_moe()) {
        for (auto* t : moe_ffn_->gradients()) g.push_back(t);
    } else {
        g.push_back(&w1_.grad_weights); g.push_back(&w1_.grad_bias);
        g.push_back(&w2_.grad_weights); g.push_back(&w2_.grad_bias);
    }
    for (auto* t : ln1_.gradients()) g.push_back(t);
    for (auto* t : ln2_.gradients()) g.push_back(t);
    for (auto* t : ln3_.gradients()) g.push_back(t);
    return g;
}

Tensor JambaBlock::get_weights() const { return mamba_->get_weights(); }
Tensor JambaBlock::get_gradients() const { return mamba_->get_gradients(); }

// =====================================================================
// JambaStack
// =====================================================================
JambaStack::JambaStack(size_t d_model, size_t num_heads, size_t num_layers,
                       size_t num_experts, size_t top_k,
                       size_t d_state, size_t moe_every_n) {
    if (num_layers == 0)
        throw std::invalid_argument("JambaStack: num_layers must be > 0");
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        // Per the Jamba paper: MoE in every other block when moe_every_n=2.
        bool use_moe = (num_experts > 0) &&
                       ((moe_every_n == 1) || (i % moe_every_n == 0));
        blocks_.push_back(std::make_unique<JambaBlock>(
            d_model, num_heads, num_experts, top_k, d_state, moe_every_n, use_moe));
    }
}

Tensor JambaStack::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& b : blocks_) x = b->forward(x);
    return x;
}

Tensor JambaStack::backward(const Tensor& grad_output, double learning_rate) {
    Tensor g = grad_output;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it)
        g = (*it)->backward(g, learning_rate);
    return g;
}

void JambaStack::update_weights(double lr) {
    for (auto& b : blocks_) b->update_weights(lr);
}

void JambaStack::zero_grad() {
    for (auto& b : blocks_) b->zero_grad();
}

std::vector<Tensor*> JambaStack::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_)
        for (auto* t : b->parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> JambaStack::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_)
        for (auto* t : b->gradients()) g.push_back(t);
    return g;
}

Tensor JambaStack::get_weights() const { return blocks_[0]->get_weights(); }
Tensor JambaStack::get_gradients() const { return blocks_[0]->get_gradients(); }
