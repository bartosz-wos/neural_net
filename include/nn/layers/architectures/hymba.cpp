// ============================================================================
// Hymba implementation — see hymba.h for architecture notes
// ============================================================================

#include "hymba.h"
#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------------
// HymbaBlock
// ---------------------------------------------------------------------------

HymbaBlock::HymbaBlock(size_t d_model, size_t d_state, size_t num_heads,
                       size_t num_kv_heads, size_t ffn_mult)
    : d_model_(d_model),
      d_state_(d_state),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      ffn_dim_((ffn_mult == 0 ? 0 : ffn_mult * d_model)),
      ln_(d_model > 0 ? d_model : 1),
      mamba_(d_model > 0 ? d_model : 1, d_state > 0 ? d_state : 1),     // dummy if invalid
      attn_(d_model > 0 ? d_model : 1, num_heads > 0 ? num_heads : 1,
            (num_heads > 0 && num_kv_heads > 0 && num_kv_heads <= num_heads) ? num_kv_heads : 1),
      mix_proj_(2 * (d_model > 0 ? d_model : 1), 2 * (d_model > 0 ? d_model : 1)),
      ln_ffn_(d_model > 0 ? d_model : 1),
      ffn1_(d_model > 0 ? d_model : 1, ffn_mult > 0 ? ffn_mult * (d_model > 0 ? d_model : 1) : 1),
      ffn2_(ffn_mult > 0 ? ffn_mult * (d_model > 0 ? d_model : 1) : 1, d_model > 0 ? d_model : 1)
{
    // Validate inputs after constructing members with safe defaults to avoid SIGFPE
    // from integer division-by-zero in sub-constructors (e.g., GQA's group_size_ = num_q/num_kv).
    if (d_model == 0)   throw std::invalid_argument("HymbaBlock: d_model must be > 0");
    if (d_state == 0)   throw std::invalid_argument("HymbaBlock: d_state must be > 0");
    if (num_heads == 0) throw std::invalid_argument("HymbaBlock: num_heads must be > 0");
    if (num_kv_heads == 0) throw std::invalid_argument("HymbaBlock: num_kv_heads must be > 0");
    if (num_kv_heads > num_heads)
        throw std::invalid_argument("HymbaBlock: num_kv_heads must be <= num_heads");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("HymbaBlock: d_model must be divisible by num_heads");
    if (ffn_mult == 0)
        throw std::invalid_argument("HymbaBlock: ffn_mult must be > 0");

    // Initialize mix_proj to bias the softmax toward Mamba at init.
    // W small random (the standard Dense init), bias [+1, -1].
    // This makes initial gate ≈ [0.731, 0.269] favoring Mamba.
    for (size_t i = 0; i < 2 * d_model; ++i) {
        for (size_t j = 0; j < 2 * d_model; ++j) {
            // Xavier-like small init
            double scale = std::sqrt(2.0 / (2.0 * d_model));
            mix_proj_.weights(i, j) = scale * (0.5 * std::sin(0.13 * i + 0.27 * j));
        }
        // Bias: +1 for Mamba channel (j < d_model), -1 for attn channel
        if (i < d_model) {
            mix_proj_.bias(0, i) = +1.0;
        } else {
            mix_proj_.bias(0, i) = -1.0;
        }
    }
}

Tensor HymbaBlock::forward(const Tensor& input) {
    size_t T = input.rows;
    size_t d = d_model_;

    last_input_ = input.clone();

    // 1. Shared pre-norm
    last_ln_x_ = ln_.forward(input);

    // 2. Two parallel paths
    last_mamba_out_ = mamba_.forward(last_ln_x_);   // (T, d)
    last_attn_out_  = attn_.forward(last_ln_x_);     // (T, d)  Q=K=V=ln_x

    // 3. Concat for mix_proj: (T, 2*d)
    last_concat_ = Tensor(T, 2 * d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            last_concat_(t, 0 * d + j) = last_mamba_out_(t, j);
            last_concat_(t, 1 * d + j) = last_attn_out_(t, j);
        }
    }

    // 4. Mix projection: (T, 2*d)
    Tensor logits = mix_proj_.forward(last_concat_);

    // 5. Per-(token, channel) 2-way softmax over the channel-pair dim
    // logits[t, 0*d+j] and logits[t, 1*d+j] are the two logits for (t, j)
    last_gate_ = Tensor(T, 2 * d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double l0 = logits(t, 0 * d + j);
            double l1 = logits(t, 1 * d + j);
            double m = std::max(l0, l1);
            double e0 = std::exp(l0 - m);
            double e1 = std::exp(l1 - m);
            double sum = e0 + e1;
            last_gate_(t, 0 * d + j) = e0 / sum;
            last_gate_(t, 1 * d + j) = e1 / sum;
        }
    }

    // 6. Mix
    last_mixed_ = Tensor(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            last_mixed_(t, j) = last_gate_(t, 0 * d + j) * last_mamba_out_(t, j)
                              + last_gate_(t, 1 * d + j) * last_attn_out_(t, j);
        }
    }

    // 7. FFN sub-block (pre-norm residual)
    last_ln_mixed_ = ln_ffn_.forward(last_mixed_);      // (T, d)
    last_ffn_hidden_ = ffn1_.forward(last_ln_mixed_);   // (T, ffn_dim)
    GELU gelu;
    last_ffn_act_ = Tensor(T, ffn_dim_);
    for (size_t i = 0; i < last_ffn_act_.data.size(); ++i)
        last_ffn_act_(i / last_ffn_act_.cols, i % last_ffn_act_.cols) =
            gelu(last_ffn_hidden_.data[i]);
    Tensor ffn_out = ffn2_.forward(last_ffn_act_);      // (T, d)

    // 8. Residual
    Tensor out = Tensor(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            out(t, j) = last_mixed_(t, j) + ffn_out(t, j);
        }
    }
    return out;
}

Tensor HymbaBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = grad_output.rows;
    size_t d = d_model_;
    size_t ffn_dim = ffn_dim_;

    // 8. Residual backward
    // d_mixed = grad_output (from residual identity path)
    // d_ffn_out = grad_output (from residual identity path)
    Tensor d_mixed  = grad_output.clone();
    Tensor d_ffn_out = grad_output.clone();

    // 7b. FFN backward
    // d_ffn_act = ffn2_.backward(d_ffn_out, lr) returns d_ffn_hidden
    Tensor d_ffn_hidden = ffn2_.backward(d_ffn_out, 0.0);
    // Apply GELU derivative
    GELU gelu;
    Tensor d_ffn_pre = Tensor(T, ffn_dim);
    for (size_t i = 0; i < d_ffn_pre.data.size(); ++i) {
        double gprime = gelu.derivative(last_ffn_hidden_.data[i]);
        d_ffn_pre.data[i] = d_ffn_hidden.data[i] * gprime;
    }
    // d_ln_mixed += ffn1_.backward(d_ffn_pre, lr)
    Tensor d_ln_mixed_from_ffn = ffn1_.backward(d_ffn_pre, 0.0);
    // d_mixed += ln_ffn_.backward(d_ln_mixed_from_ffn, lr)
    Tensor d_mixed_from_ffn = ln_ffn_.backward(d_ln_mixed_from_ffn, 0.0);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d; ++j)
            d_mixed(t, j) += d_mixed_from_ffn(t, j);

    // 6. Mix backward
    // d_mamba_out[t, j] = gate[t, 0, j] * d_mixed[t, j]
    // d_attn_out[t, j]  = gate[t, 1, j] * d_mixed[t, j]
    // d_gate[t, 0, j]   = mamba_out[t, j] * d_mixed[t, j]
    // d_gate[t, 1, j]   = attn_out[t, j]  * d_mixed[t, j]
    Tensor d_gate = Tensor(T, 2 * d);
    Tensor d_mamba_out = Tensor(T, d);
    Tensor d_attn_out  = Tensor(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double g0 = last_gate_(t, 0 * d + j);
            double g1 = last_gate_(t, 1 * d + j);
            double dm = d_mixed(t, j);
            d_mamba_out(t, j) = g0 * dm;
            d_attn_out(t, j)  = g1 * dm;
            d_gate(t, 0 * d + j) = last_mamba_out_(t, j) * dm;
            d_gate(t, 1 * d + j) = last_attn_out_(t, j)  * dm;
        }
    }

    // 5b. Softmax-over-2 backward
    // For each (t, j): let s = softmax(l0, l1), d_l_k = s_k * (d_gate_k - sum_m s_m * d_gate_m)
    Tensor d_logits = Tensor(T, 2 * d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            double s0 = last_gate_(t, 0 * d + j);
            double s1 = last_gate_(t, 1 * d + j);
            double dg0 = d_gate(t, 0 * d + j);
            double dg1 = d_gate(t, 1 * d + j);
            double dot = s0 * dg0 + s1 * dg1;
            d_logits(t, 0 * d + j) = s0 * (dg0 - dot);
            d_logits(t, 1 * d + j) = s1 * (dg1 - dot);
        }
    }

    // 4b. mix_proj backward
    // d_concat = mix_proj_.backward(d_logits, lr) returns d_concat
    Tensor d_concat = mix_proj_.backward(d_logits, 0.0);
    // Split d_concat back into d_mamba_out (contribution from concat path) and d_attn_out
    // The mix_proj.input was [mamba_out ‖ attn_out] (T, 2d)
    // d_mamba_out += d_concat[:, 0:d]
    // d_attn_out  += d_concat[:, d:2d]
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            d_mamba_out(t, j) += d_concat(t, 0 * d + j);
            d_attn_out(t, j)  += d_concat(t, 1 * d + j);
        }
    }

    // Cache for tests
    last_d_mamba_out_ = d_mamba_out.clone();
    last_d_attn_out_  = d_attn_out.clone();
    last_d_gate_      = d_gate.clone();

    // 2b. Mamba backward
    Tensor d_ln_x_from_mamba = mamba_.backward(d_mamba_out, 0.0);

    // 2b. GQA backward
    Tensor d_ln_x_from_attn = attn_.backward(d_attn_out, 0.0);

    // 1b. Sum two paths through shared pre-norm
    Tensor d_ln_x = Tensor(T, d);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d; ++j) {
            d_ln_x(t, j) = d_ln_x_from_mamba(t, j) + d_ln_x_from_attn(t, j);
        }
    }
    last_d_ln_x_ = d_ln_x.clone();

    // LayerNorm backward
    Tensor d_input = ln_.backward(d_ln_x, 0.0);
    return d_input;
}

void HymbaBlock::update_weights(double learning_rate) {
    ln_.update_weights(learning_rate);
    mamba_.update_weights(learning_rate);
    attn_.update_weights(learning_rate);
    mix_proj_.update_weights(learning_rate);
    ln_ffn_.update_weights(learning_rate);
    ffn1_.update_weights(learning_rate);
    ffn2_.update_weights(learning_rate);
}

std::vector<Tensor*> HymbaBlock::parameters() {
    std::vector<Tensor*> p;
    auto ln_p = ln_.parameters();       for (auto* x : ln_p) p.push_back(x);
    auto m_p = mamba_.parameters();    for (auto* x : m_p) p.push_back(x);
    auto a_p = attn_.parameters();     for (auto* x : a_p) p.push_back(x);
    auto x_p = mix_proj_.parameters(); for (auto* x : x_p) p.push_back(x);
    auto f_p = ln_ffn_.parameters();   for (auto* x : f_p) p.push_back(x);
    auto h1 = ffn1_.parameters();      for (auto* x : h1) p.push_back(x);
    auto h2 = ffn2_.parameters();      for (auto* x : h2) p.push_back(x);
    return p;
}

std::vector<Tensor*> HymbaBlock::gradients() {
    std::vector<Tensor*> g;
    auto ln_g = ln_.gradients();       for (auto* x : ln_g) g.push_back(x);
    auto m_g = mamba_.gradients();     for (auto* x : m_g) g.push_back(x);
    auto a_g = attn_.gradients();      for (auto* x : a_g) g.push_back(x);
    auto x_g = mix_proj_.gradients();  for (auto* x : x_g) g.push_back(x);
    auto f_g = ln_ffn_.gradients();    for (auto* x : f_g) g.push_back(x);
    auto h1 = ffn1_.gradients();       for (auto* x : h1) g.push_back(x);
    auto h2 = ffn2_.gradients();       for (auto* x : h2) g.push_back(x);
    return g;
}

void HymbaBlock::zero_grad() {
    ln_.zero_grad();
    mamba_.zero_grad();
    attn_.zero_grad();
    mix_proj_.zero_grad();
    ln_ffn_.zero_grad();
    ffn1_.zero_grad();
    ffn2_.zero_grad();
}

// ---------------------------------------------------------------------------
// HymbaModel
// ---------------------------------------------------------------------------

HymbaModel::HymbaModel(size_t input_dim, size_t d_model, size_t output_dim,
                       size_t num_layers, size_t d_state,
                       size_t num_heads, size_t num_kv_heads, size_t ffn_mult)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      d_state_(d_state),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      ffn_mult_(ffn_mult),
      input_proj_(input_dim, d_model),
      classifier_(d_model, output_dim)
{
    if (num_layers == 0) throw std::invalid_argument("HymbaModel: num_layers must be > 0");
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(std::make_unique<HymbaBlock>(d_model, d_state, num_heads, num_kv_heads, ffn_mult));
    }
    block_outputs_.resize(num_layers + 2);
}

Tensor HymbaModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor cur = input_proj_.forward(input);   // (T, d_model)
    block_outputs_[0] = cur.clone();
    for (size_t i = 0; i < num_layers_; ++i) {
        cur = blocks_[i]->forward(cur);
        block_outputs_[i + 1] = cur.clone();
    }
    Tensor logits = classifier_.forward(cur);   // (T, output_dim)
    block_outputs_[num_layers_ + 1] = logits.clone();
    return logits;
}

Tensor HymbaModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // classifier backward
    Tensor d_block_out = classifier_.backward(grad_output, 0.0);
    // blocks in reverse
    for (int i = (int)num_layers_ - 1; i >= 0; --i) {
        d_block_out = blocks_[i]->backward(d_block_out, 0.0);
    }
    // input projection backward
    Tensor d_input = input_proj_.backward(d_block_out, 0.0);
    return d_input;
}

void HymbaModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

std::vector<Tensor*> HymbaModel::parameters() {
    std::vector<Tensor*> p;
    auto ip = input_proj_.parameters();   for (auto* x : ip) p.push_back(x);
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        for (auto* x : bp) p.push_back(x);
    }
    auto cp = classifier_.parameters();   for (auto* x : cp) p.push_back(x);
    return p;
}

std::vector<Tensor*> HymbaModel::gradients() {
    std::vector<Tensor*> g;
    auto ig = input_proj_.gradients();    for (auto* x : ig) g.push_back(x);
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        for (auto* x : bg) g.push_back(x);
    }
    auto cg = classifier_.gradients();    for (auto* x : cg) g.push_back(x);
    return g;
}

void HymbaModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    classifier_.zero_grad();
}
