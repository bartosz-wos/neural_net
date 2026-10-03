#include "striped_hyena.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// StripedHyena — see the header for the paper specification and citations.
// ============================================================================

// ----------------------------------------------------------------------------
// build_stripe_schedule
//
//   pattern  = [ATTENTION, H, H, ... H]      (1 + hyena_ratio entries)
//   schedule = pattern repeated, truncated to n_layers
//
// arXiv:2403.17844 Appendix C.2: "In instances where the number of layers is
// not a multiple of the schedule, the ratio is repeated until the target depth
// is reached." Repeating a fixed pattern and truncating IS that rule; the
// pattern starts with attention, so a partial final pattern still contributes
// its leading attention block.
// ----------------------------------------------------------------------------
std::vector<bool> build_stripe_schedule(size_t n_layers, size_t hyena_ratio) {
    if (n_layers == 0) throw std::invalid_argument("build_stripe_schedule: n_layers must be > 0");
    if (hyena_ratio == 0) throw std::invalid_argument("build_stripe_schedule: hyena_ratio must be > 0");
    std::vector<bool> s;
    s.reserve(n_layers);
    const size_t period = 1 + hyena_ratio;
    for (size_t i = 0; i < n_layers; ++i) {
        s.push_back((i % period) == 0);   // attention at the head of each run
    }
    return s;
}

// ----------------------------------------------------------------------------
// Transposition adapter.
//
// MultiHeadAttention in this repo uses the TRANSPOSED sequence layout
// (d_model, seq_len) — see transformer.cpp:30-35 — while HyenaOperator,
// RMSNorm, SwiGLU and the rest of the block use the channels-LAST layout
// (seq_len, d_model). The striped block holds both, so every call into the
// attention mixer has to cross this boundary. It is the single easiest place
// for an off-by-transpose bug, and a transposed gradient silently produces
// plausible-looking numbers, so it is centralised here rather than inlined.
// ----------------------------------------------------------------------------
static Tensor transpose_to_mha(const Tensor& x) {
    Tensor t(x.cols, x.rows);   // (d_model, seq_len)
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            t[j][i] = x[i][j];
    return t;
}

static Tensor transpose_from_mha(const Tensor& t) {
    Tensor x(t.cols, t.rows);   // (seq_len, d_model)
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            x[j][i] = t[i][j];
    return x;
}

// ----------------------------------------------------------------------------
// StripedHyenaBlock
// ----------------------------------------------------------------------------
StripedHyenaBlock::StripedHyenaBlock(size_t d_model, size_t seq_len,
                                     bool is_attention, size_t ffn_mult,
                                     size_t num_heads, size_t hyena_order)
    : d_model_(d_model), seq_len_(seq_len), ffn_mult_(ffn_mult),
      is_attention_(is_attention), num_heads_(num_heads), hyena_order_(hyena_order)
{
    if (d_model == 0) throw std::invalid_argument("StripedHyenaBlock: d_model must be > 0");
    if (seq_len == 0) throw std::invalid_argument("StripedHyenaBlock: seq_len must be > 0");
    if (num_heads == 0) throw std::invalid_argument("StripedHyenaBlock: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("StripedHyenaBlock: d_model must be divisible by num_heads");
    if (hyena_order < 1) throw std::invalid_argument("StripedHyenaBlock: hyena_order must be >= 1");

    // Two RMSNorms regardless of mixer kind — the block shape is uniform so a
    // striping schedule can mix the two block kinds freely.
    ln1_ = std::unique_ptr<RMSNorm>(new RMSNorm(d_model));
    ln2_ = std::unique_ptr<RMSNorm>(new RMSNorm(d_model));

    if (is_attention_) {
        attn_ = std::unique_ptr<MultiHeadAttention>(new MultiHeadAttention(d_model, num_heads));
    } else {
        hyena_ = std::unique_ptr<HyenaOperator>(
            new HyenaOperator(d_model, seq_len, hyena_order));
    }

    if (ffn_mult_ > 0) {
        size_t ffn_hidden = d_model_ * ffn_mult_;
        ffn_ = std::unique_ptr<SwiGLU<>>(new SwiGLU<>(d_model_, ffn_hidden));
        ffn_down_ = std::unique_ptr<Dense>(new Dense(ffn_hidden, d_model_));
    }
}

Tensor StripedHyenaBlock::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("StripedHyenaBlock: input.cols != d_model_");

    last_input_ = input;

    // u = x + Mixer(RMSNorm(x))
    Tensor n1 = ln1_->forward(input);
    last_n1_ = n1;
    Tensor mix;
    if (is_attention_) {
        Tensor mha_out = attn_->forward(transpose_to_mha(n1));   // (d_model_, L)
        mix = transpose_from_mha(mha_out);                        // (L, d_model_)
    } else {
        mix = hyena_->forward(n1);                                // (L, d_model_)
    }
    last_mix_ = mix;
    Tensor u(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            u[i][j] = input[i][j] + mix[i][j];
    last_u_ = u;

    // y = u + ffn_down(SwiGLU(RMSNorm(u)))   (skipped when ffn_mult_ == 0)
    if (!ffn_) return u;

    Tensor n2 = ln2_->forward(u);
    last_n2_ = n2;
    Tensor ch = ffn_down_->forward(ffn_->forward(n2));   // (L, d_model_)
    Tensor y(u.rows, d_model_);
    for (size_t i = 0; i < u.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            y[i][j] = u[i][j] + ch[i][j];
    return y;
}

Tensor StripedHyenaBlock::backward(const Tensor& grad_output, double lr) {
    if (grad_output.rows != last_u_.rows || grad_output.cols != d_model_)
        throw std::invalid_argument("StripedHyenaBlock: grad_output shape mismatch");

    Tensor d_u = grad_output.clone();

    // Channel-mixer residual: y = u + ch where ch = ffn_down(ffn(ln2(u))).
    // The residual is a sum, so dL/dch is dL/dy unchanged; ch is d_model wide
    // because of the down-projection, so grad_output already has the right
    // shape to feed the chain.
    if (ffn_) {
        Tensor d_ch = ffn_down_->backward(grad_output, lr);   // (L, ffn_hidden)
        Tensor d_ffn = ffn_->backward(d_ch, lr);             // (L, d_model)
        Tensor d_ln2_x = ln2_->backward(d_ffn, lr);          // ln2 param grads + dL/dn2
        for (size_t i = 0; i < d_u.rows; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                d_u[i][j] += d_ln2_x[i][j];
    }

    // u = x + mix, so d_mix = d_u and d_x gets d_u as the identity residual.
    Tensor d_mix = d_u.clone();
    Tensor d_x = d_u.clone();

    // Back through the mixer, then through ln1 into x.
    Tensor d_n1;
    if (is_attention_) {
        Tensor d_mha = attn_->backward(transpose_to_mha(d_mix), lr);   // (d_model_, L)
        d_n1 = transpose_from_mha(d_mha);                             // (L, d_model_)
    } else {
        d_n1 = hyena_->backward(d_mix, lr);                           // (L, d_model_)
    }

    // The residual bypasses ln1 entirely, so the ln1 input-gradient has to be
    // added to the identity path explicitly — RMSNorm.backward returns it.
    Tensor d_ln1_x = ln1_->backward(d_n1, lr);
    for (size_t i = 0; i < d_x.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_x[i][j] += d_ln1_x[i][j];

    return d_x;
}

void StripedHyenaBlock::update_weights(double lr) {
    ln1_->update_weights(lr);
    if (attn_) attn_->update_weights(lr);
    if (hyena_) hyena_->update_weights(lr);
    if (ffn_) ffn_->update_weights(lr);
    if (ffn_down_) ffn_down_->update_weights(lr);
    if (ln2_) ln2_->update_weights(lr);
}

Tensor StripedHyenaBlock::get_weights() const {
    return is_attention_ ? attn_->W_q : hyena_->in_proj.weights;
}

Tensor StripedHyenaBlock::get_gradients() const {
    return is_attention_ ? attn_->grad_W_q : hyena_->in_proj.grad_weights;
}

std::vector<Tensor*> StripedHyenaBlock::parameters() {
    std::vector<Tensor*> out;
    for (auto* p : ln1_->parameters()) out.push_back(p);
    if (attn_) for (auto* p : attn_->parameters()) out.push_back(p);
    if (hyena_) for (auto* p : hyena_->parameters()) out.push_back(p);
    if (ffn_) for (auto* p : ffn_->parameters()) out.push_back(p);
    if (ffn_down_) for (auto* p : ffn_down_->parameters()) out.push_back(p);
    for (auto* p : ln2_->parameters()) out.push_back(p);
    return out;
}

std::vector<Tensor*> StripedHyenaBlock::gradients() {
    std::vector<Tensor*> out;
    for (auto* p : ln1_->gradients()) out.push_back(p);
    if (attn_) for (auto* p : attn_->gradients()) out.push_back(p);
    if (hyena_) for (auto* p : hyena_->gradients()) out.push_back(p);
    if (ffn_) for (auto* p : ffn_->gradients()) out.push_back(p);
    if (ffn_down_) for (auto* p : ffn_down_->gradients()) out.push_back(p);
    for (auto* p : ln2_->gradients()) out.push_back(p);
    return out;
}

void StripedHyenaBlock::zero_grad() {
    ln1_->zero_grad();
    if (attn_) attn_->zero_grad();
    if (hyena_) hyena_->zero_grad();
    if (ffn_) ffn_->zero_grad();
    if (ffn_down_) ffn_down_->zero_grad();
    ln2_->zero_grad();
}

// ----------------------------------------------------------------------------
// StripedHyenaModel
// ----------------------------------------------------------------------------
StripedHyenaModel::StripedHyenaModel(size_t input_dim, size_t d_model,
                                     size_t num_layers, size_t output_dim,
                                     size_t seq_len, size_t num_heads,
                                     size_t hyena_ratio, size_t ffn_mult)
    : in_proj_(input_dim, d_model),
      classifier_(d_model, output_dim),
      d_model_(d_model), num_layers_(num_layers), output_dim_(output_dim),
      seq_len_(seq_len), num_heads_(num_heads), hyena_ratio_(hyena_ratio)
{
    if (input_dim == 0) throw std::invalid_argument("StripedHyenaModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("StripedHyenaModel: d_model must be > 0");
    if (num_layers == 0) throw std::invalid_argument("StripedHyenaModel: num_layers must be > 0");
    if (output_dim == 0) throw std::invalid_argument("StripedHyenaModel: output_dim must be > 0");
    if (seq_len == 0) throw std::invalid_argument("StripedHyenaModel: seq_len must be > 0");
    if (num_heads == 0) throw std::invalid_argument("StripedHyenaModel: num_heads must be > 0");
    if (hyena_ratio == 0) throw std::invalid_argument("StripedHyenaModel: hyena_ratio must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("StripedHyenaModel: d_model must be divisible by num_heads");

    // Validate before allocating: the block ctor throws on a bad
    // num_heads/d_model pairing too, and we want that to surface before any
    // partially-populated `blocks` vector exists (the check above already
    // guarantees it, so no second check is needed here).

    schedule_ = build_stripe_schedule(num_layers, hyena_ratio);

    blocks.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks.emplace_back(d_model, seq_len, schedule_[i], ffn_mult, num_heads);
    }

    final_norm_ = std::unique_ptr<RMSNorm>(new RMSNorm(d_model));
}

size_t StripedHyenaModel::num_attention_blocks() const {
    size_t n = 0;
    for (bool b : schedule_) if (b) ++n;
    return n;
}

size_t StripedHyenaModel::num_hyena_blocks() const {
    return num_layers_ - num_attention_blocks();
}

Tensor StripedHyenaModel::forward(const Tensor& input) {
    if (input.cols != in_proj_.weights.cols)
        throw std::invalid_argument("StripedHyenaModel: input width mismatch");

    last_input_ = input;
    Tensor h = in_proj_.forward(input);
    last_proj_ = h;

    for (size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i].forward(h);
    }

    h = final_norm_->forward(h);

    // Mean-pool over the sequence dimension -> (1, d_model).
    Tensor pooled(1, d_model_);
    for (size_t i = 0; i < h.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            pooled[0][j] += h[i][j] / static_cast<double>(h.rows);
    last_pooled_ = pooled;

    return classifier_.forward(pooled);
}

Tensor StripedHyenaModel::backward(const Tensor& grad_output, double lr) {
    if (grad_output.rows != 1 || grad_output.cols != output_dim_)
        throw std::invalid_argument("StripedHyenaModel: grad_output shape mismatch");

    // classifier backward -> d_pooled
    Tensor d_pooled = classifier_.backward(grad_output, lr);

    // Mean-pool backward: every token receives d_pooled / L.
    const double inv_L = 1.0 / static_cast<double>(last_proj_.rows);
    Tensor d_h(last_proj_.rows, d_model_);
    for (size_t i = 0; i < last_proj_.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_h[i][j] = d_pooled[0][j] * inv_L;

    // RMSNorm.backward returns d_h w.r.t. its own input, which is the gradient
    // flowing back into the block stack.
    d_h = final_norm_->backward(d_h, lr);

    for (size_t i = blocks.size(); i-- > 0; ) {
        d_h = blocks[i].backward(d_h, lr);
    }

    // in_proj_.backward returns the gradient w.r.t. the model's INPUT (it
    // computes grad_output * weights, i.e. (L, d_model) @ (d_model, input_dim)
    // = (L, input_dim)). That value is what callers compare against finite
    // differences.
    //
    // Returning `d_h` instead is a real bug, not a stylistic choice: d_h is the
    // gradient at the projection OUTPUT, which is a different tensor with a
    // different (d_model) shape. It only goes unnoticed when input_dim ==
    // d_model, where the shapes coincide and the values look plausible — the
    // FD input-gradient check disagreed by rel_err ~1.85 in exactly that
    // configuration.
    return in_proj_.backward(d_h, lr);
}

void StripedHyenaModel::update_weights(double lr) {
    in_proj_.update_weights(lr);
    for (auto& b : blocks) b.update_weights(lr);
    final_norm_->update_weights(lr);
    classifier_.update_weights(lr);
}

std::vector<Tensor*> StripedHyenaModel::parameters() {
    std::vector<Tensor*> out;
    for (auto* p : in_proj_.parameters()) out.push_back(p);
    for (auto& b : blocks) for (auto* p : b.parameters()) out.push_back(p);
    for (auto* p : final_norm_->parameters()) out.push_back(p);
    for (auto* p : classifier_.parameters()) out.push_back(p);
    return out;
}

std::vector<Tensor*> StripedHyenaModel::gradients() {
    std::vector<Tensor*> out;
    for (auto* p : in_proj_.gradients()) out.push_back(p);
    for (auto& b : blocks) for (auto* p : b.gradients()) out.push_back(p);
    for (auto* p : final_norm_->gradients()) out.push_back(p);
    for (auto* p : classifier_.gradients()) out.push_back(p);
    return out;
}

void StripedHyenaModel::zero_grad() {
    in_proj_.zero_grad();
    for (auto& b : blocks) b.zero_grad();
    final_norm_->zero_grad();
    classifier_.zero_grad();
}
