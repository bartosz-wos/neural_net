#include "dit.h"
#include <stdexcept>
#include <algorithm>

namespace nn {

// =============================================================================
// DiTTimeEmbed
// =============================================================================
DiTTimeEmbed::DiTTimeEmbed(int hidden_dim, int max_period)
    : hidden_dim_(hidden_dim), max_period_(max_period),
      mlp_in_(hidden_dim, hidden_dim),
      mlp_out_(hidden_dim, hidden_dim) {}

Tensor DiTTimeEmbed::forward(int t) {
    // Vaswani-style sinusoidal: pos = t, i = channel index
    // freq_i = 1 / max_period^(i / hidden_dim)
    // pe[pos, 2i]   = sin(pos * freq_i)
    // pe[pos, 2i+1] = cos(pos * freq_i)
    // We'll use the DDPM convention: even=cos, odd=sin (matches DDPM::TimeEmbedding)
    Tensor sinusoid(1, hidden_dim_);
    for (int i = 0; i < hidden_dim_; ++i) {
        double freq_exp = static_cast<double>(i) / hidden_dim_;
        double freq = 1.0 / std::pow(max_period_, freq_exp);
        double arg = static_cast<double>(t) * freq;
        sinusoid[0][i] = (i % 2 == 0) ? std::cos(arg) : std::sin(arg);
    }
    last_sinusoidal_ = sinusoid;

    // 2-layer MLP with SiLU
    Tensor h = mlp_in_.forward(sinusoid);
    // SiLU: x * sigmoid(x)
    for (size_t i = 0; i < h.rows; ++i)
        for (size_t j = 0; j < h.cols; ++j) {
            h[i][j] = h[i][j] / (1.0 + std::exp(-h[i][j]));
        }
    Tensor out = mlp_out_.forward(h);
    return out;
}

std::vector<Tensor*> DiTTimeEmbed::parameters() {
    return {&mlp_in_.weights, &mlp_in_.bias, &mlp_out_.weights, &mlp_out_.bias};
}

std::vector<Tensor*> DiTTimeEmbed::gradients() {
    return {&mlp_in_.grad_weights, &mlp_in_.grad_bias, &mlp_out_.grad_weights, &mlp_out_.grad_bias};
}

void DiTTimeEmbed::zero_grad() {
    mlp_in_.zero_grad();
    mlp_out_.zero_grad();
}

// =============================================================================
// DiTLabelEmbed
// =============================================================================
DiTLabelEmbed::DiTLabelEmbed(int num_classes, int d_model)
    : num_classes_(num_classes), d_model_(d_model),
      embed_(num_classes + 1, d_model) {}

Tensor DiTLabelEmbed::forward(int class_idx) {
    if (class_idx < 0 || class_idx > num_classes_) {
        throw std::out_of_range("DiTLabelEmbed: class_idx out of range (0=null, 1..num_classes)");
    }
    Tensor idx(1, 1);
    idx[0][0] = static_cast<double>(class_idx);
    return embed_.forward(idx);
}

Tensor DiTLabelEmbed::forward_batch(const std::vector<int>& class_idxs) {
    Tensor idx(static_cast<int>(class_idxs.size()), 1);
    for (size_t i = 0; i < class_idxs.size(); ++i) {
        if (class_idxs[i] < 0 || class_idxs[i] > num_classes_) {
            throw std::out_of_range("DiTLabelEmbed: class_idx out of range");
        }
        idx[static_cast<int>(i)][0] = static_cast<double>(class_idxs[i]);
    }
    return embed_.forward(idx);
}

// =============================================================================
// SequencePatchEmbed
// =============================================================================
SequencePatchEmbed::SequencePatchEmbed(int in_dim, int d_model, int patch_len)
    : in_dim_(in_dim), d_model_(d_model), patch_len_(patch_len),
      proj_(in_dim * patch_len, d_model) {}

Tensor SequencePatchEmbed::forward(const Tensor& x) {
    int B = static_cast<int>(x.rows);
    int total = static_cast<int>(x.cols);
    int patch_in = in_dim_ * patch_len_;
    if (total % patch_in != 0) {
        throw std::invalid_argument("SequencePatchEmbed: T*in_dim must be divisible by patch_len*in_dim");
    }
    int S = total / patch_in;
    Tensor out(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor patch(1, patch_in);
            for (int k = 0; k < patch_in; ++k) {
                patch[0][k] = x[b][s * patch_in + k];
            }
            Tensor projected = proj_.forward(patch);  // (1, d_model)
            for (int d = 0; d < d_model_; ++d) {
                out[b][s * d_model_ + d] = projected[0][d];
            }
        }
    }
    return out;
}

// =============================================================================
// SequenceUnpatchify
// =============================================================================
SequenceUnpatchify::SequenceUnpatchify(int d_model, int out_dim, int patch_len)
    : d_model_(d_model), out_dim_(out_dim), patch_len_(patch_len),
      proj_(d_model, out_dim * patch_len) {
    // Default: zero-init the projection weights so the DiT "Zero" output is preserved.
    // Users can call proj_.init_weights() to opt out of the zero-init.
    proj_.weights = Tensor::zeros(out_dim * patch_len, d_model);
    proj_.bias = Tensor::zeros(1, out_dim * patch_len);
}

Tensor SequenceUnpatchify::forward(const Tensor& x) {
    int B = static_cast<int>(x.rows);
    int total = static_cast<int>(x.cols);
    if (total % d_model_ != 0) {
        throw std::invalid_argument("SequenceUnpatchify: S*d_model must be divisible by d_model");
    }
    int S = total / d_model_;
    int patch_out = out_dim_ * patch_len_;
    Tensor out(B, S * patch_out);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = x[b][s * d_model_ + d];
            Tensor proj = proj_.forward(tok);  // (1, out_dim*patch_len)
            for (int k = 0; k < patch_out; ++k) {
                out[b][s * patch_out + k] = proj[0][k];
            }
        }
    }
    return out;
}

// =============================================================================
// DiTBlock
// =============================================================================
DiTBlock::DiTBlock(int d_model, int n_heads, double mlp_ratio, int cond_dim)
    : d_model_(d_model), n_heads_(n_heads),
      mlp_dim_(static_cast<int>(d_model * mlp_ratio)),
      cond_dim_(cond_dim),
      ln1_(d_model), ln2_(d_model),
      mod_(cond_dim, 6 * d_model),
      attn_qkv_(d_model, 3 * d_model),
      attn_o_(d_model, d_model),
      mlp_w1_(d_model, mlp_dim_),
      mlp_w2_(mlp_dim_, d_model) {
    // adaLN-Zero: zero-init the modulation Dense and output projection
    // Dense stores weights as (out_features, in_features). We zero the entire (out, in) shape.
    mod_.weights = Tensor::zeros(6 * d_model, cond_dim);
    mod_.bias = Tensor::zeros(1, 6 * d_model);
    attn_o_.weights = Tensor::zeros(d_model, d_model);
    attn_o_.bias = Tensor::zeros(1, d_model);
    mlp_w2_.weights = Tensor::zeros(d_model, mlp_dim_);
    mlp_w2_.bias = Tensor::zeros(1, d_model);
}

Tensor DiTBlock::modulation(const Tensor& cond) {
    Tensor m = mod_.forward(cond);
    last_mod_ = m;
    return m;
}

Tensor DiTBlock::forward(const Tensor& x, const Tensor& cond) {
    int B = static_cast<int>(x.rows);
    int total = static_cast<int>(x.cols);
    if (total % d_model_ != 0) {
        throw std::invalid_argument("DiTBlock: x.cols must be divisible by d_model");
    }
    if (static_cast<int>(cond.cols) != cond_dim_) {
        throw std::invalid_argument("DiTBlock: cond.cols must equal cond_dim");
    }
    int S = total / d_model_;
    int head_dim = d_model_ / n_heads_;
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("DiTBlock: d_model must be divisible by n_heads");
    }

    last_x_ = x;
    // Modulation vectors
    Tensor m = modulation(cond);  // (1, 6*d_model)
    auto slice = [&](int start, int end) {
        Tensor s(1, end - start);
        for (int i = start; i < end; ++i) s[0][i - start] = m[0][i];
        return s;
    };
    Tensor shift_msa = slice(0, d_model_);
    Tensor scale_msa = slice(d_model_, 2 * d_model_);
    Tensor gate_msa = slice(2 * d_model_, 3 * d_model_);
    Tensor shift_mlp = slice(3 * d_model_, 4 * d_model_);
    Tensor scale_mlp = slice(4 * d_model_, 5 * d_model_);
    Tensor gate_mlp = slice(5 * d_model_, 6 * d_model_);

    // 1. Pre-LN + adaLN-Zero modulation: h = (1+scale)*LN(x) + shift
    // LayerNorm normalizes over features (per-token), so apply it per-token.
    Tensor ln1_out(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = x[b][s * d_model_ + d];
            Tensor tok_ln = ln1_.forward(tok);  // (1, d_model)
            for (int d = 0; d < d_model_; ++d) ln1_out[b][s * d_model_ + d] = tok_ln[0][d];
        }
    }
    last_ln1_out_ = ln1_out;
    Tensor mod1(B, S * d_model_);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < d_model_; ++d) {
                int idx = s * d_model_ + d;
                mod1[b][idx] = (1.0 + scale_msa[0][d]) * ln1_out[b][idx] + shift_msa[0][d];
            }

    // 2. Multi-head attention (causal mask not used — diffusion sees all tokens)
    // Per-token QKV projection: (B*S, d_model) -> (B*S, 3*d_model)
    Tensor qkv(B, S * 3 * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = mod1[b][s * d_model_ + d];
            Tensor tok_qkv = attn_qkv_.forward(tok);  // (1, 3*d_model)
            for (int k = 0; k < 3 * d_model_; ++k) {
                qkv[b][s * 3 * d_model_ + k] = tok_qkv[0][k];
            }
        }
    }
    // Per-head attention (single batch loop, no batching across heads)
    Tensor attn_out(B, S * d_model_);
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads_; ++h) {
            for (int sq = 0; sq < S; ++sq) {
                std::vector<double> scores(S);
                double max_s = -1e30;
                for (int sk = 0; sk < S; ++sk) {
                    double dot = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += qkv[b][sq * 3 * d_model_ + h * head_dim + d] *
                               qkv[b][sk * 3 * d_model_ + d_model_ + h * head_dim + d];
                    }
                    scores[sk] = dot * scale;
                    if (scores[sk] > max_s) max_s = scores[sk];
                }
                double sum_exp = 0;
                for (int sk = 0; sk < S; ++sk) {
                    scores[sk] = std::exp(scores[sk] - max_s);
                    sum_exp += scores[sk];
                }
                for (int sk = 0; sk < S; ++sk) scores[sk] /= sum_exp;
                for (int d = 0; d < head_dim; ++d) {
                    double v = 0;
                    for (int sk = 0; sk < S; ++sk) {
                        v += scores[sk] * qkv[b][sk * 3 * d_model_ + 2 * d_model_ + h * head_dim + d];
                    }
                    attn_out[b][sq * d_model_ + h * head_dim + d] = v;
                }
            }
        }
    }
    // Output projection (per-token)
    Tensor attn_proj(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = attn_out[b][s * d_model_ + d];
            Tensor tok_o = attn_o_.forward(tok);  // (1, d_model)
            for (int d = 0; d < d_model_; ++d) attn_proj[b][s * d_model_ + d] = tok_o[0][d];
        }
    }
    last_attn_out_ = attn_proj;
    // Apply gate and residual: h1 = x + gate_msa * attn_proj
    Tensor h1(B, S * d_model_);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < d_model_; ++d) {
                int idx = s * d_model_ + d;
                h1[b][idx] = x[b][idx] + gate_msa[0][d] * attn_proj[b][idx];
            }
    last_h1_ = h1;

    // 3. MLP path
    // LayerNorm per-token
    Tensor ln2_out(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = h1[b][s * d_model_ + d];
            Tensor tok_ln = ln2_.forward(tok);  // (1, d_model)
            for (int d = 0; d < d_model_; ++d) ln2_out[b][s * d_model_ + d] = tok_ln[0][d];
        }
    }
    last_ln2_out_ = ln2_out;
    Tensor mod2(B, S * d_model_);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < d_model_; ++d) {
                int idx = s * d_model_ + d;
                mod2[b][idx] = (1.0 + scale_mlp[0][d]) * ln2_out[b][idx] + shift_mlp[0][d];
            }
    Tensor mlp_h(B, S * mlp_dim_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = mod2[b][s * d_model_ + d];
            Tensor tok_h = mlp_w1_.forward(tok);  // (1, mlp_dim)
            for (int k = 0; k < mlp_dim_; ++k) mlp_h[b][s * mlp_dim_ + k] = tok_h[0][k];
        }
    }
    // GELU
    GELU gelu;
    for (size_t i = 0; i < mlp_h.rows; ++i)
        for (size_t j = 0; j < mlp_h.cols; ++j)
            mlp_h[i][j] = gelu(mlp_h[i][j]);
    Tensor mlp_out(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, mlp_dim_);
            for (int k = 0; k < mlp_dim_; ++k) tok[0][k] = mlp_h[b][s * mlp_dim_ + k];
            Tensor tok_o = mlp_w2_.forward(tok);  // (1, d_model)
            for (int d = 0; d < d_model_; ++d) mlp_out[b][s * d_model_ + d] = tok_o[0][d];
        }
    }
    last_mlp_out_ = mlp_out;
    // Apply gate and residual
    Tensor out(B, S * d_model_);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < d_model_; ++d) {
                int idx = s * d_model_ + d;
                out[b][idx] = h1[b][idx] + gate_mlp[0][d] * mlp_out[b][idx];
            }
    return out;
}

std::vector<Tensor*> DiTBlock::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&mod_.weights); p.push_back(&mod_.bias);
    p.push_back(&attn_qkv_.weights); p.push_back(&attn_qkv_.bias);
    p.push_back(&attn_o_.weights); p.push_back(&attn_o_.bias);
    p.push_back(&mlp_w1_.weights); p.push_back(&mlp_w1_.bias);
    p.push_back(&mlp_w2_.weights); p.push_back(&mlp_w2_.bias);
    return p;
}

std::vector<Tensor*> DiTBlock::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&mod_.grad_weights); g.push_back(&mod_.grad_bias);
    g.push_back(&attn_qkv_.grad_weights); g.push_back(&attn_qkv_.grad_bias);
    g.push_back(&attn_o_.grad_weights); g.push_back(&attn_o_.grad_bias);
    g.push_back(&mlp_w1_.grad_weights); g.push_back(&mlp_w1_.grad_bias);
    g.push_back(&mlp_w2_.grad_weights); g.push_back(&mlp_w2_.grad_bias);
    return g;
}

void DiTBlock::zero_grad() {
    mod_.zero_grad();
    attn_qkv_.zero_grad();
    attn_o_.zero_grad();
    mlp_w1_.zero_grad();
    mlp_w2_.zero_grad();
}

// =============================================================================
// DiT
// =============================================================================
DiT::DiT(int d_model, int depth, int n_heads, int in_dim, int patch_len,
         int num_classes, double mlp_ratio)
    : d_model_(d_model), depth_(depth), n_heads_(n_heads),
      in_dim_(in_dim), patch_len_(patch_len), num_classes_(num_classes),
      mlp_ratio_(mlp_ratio),
      patch_embed_(in_dim, d_model, patch_len),
      unpatchify_(d_model, in_dim, patch_len),
      time_embed_(d_model),
      final_ln_(d_model) {
    if (num_classes_ > 0) {
        label_embed_ = std::make_unique<DiTLabelEmbed>(num_classes_, d_model_);
    } else {
        label_embed_ = nullptr;
    }
    cond_dim_ = 2 * d_model_;  // time + class (or zero padding for null class)
    for (int i = 0; i < depth_; ++i) {
        blocks_.emplace_back(std::make_unique<DiTBlock>(d_model_, n_heads_, mlp_ratio_, cond_dim_));
    }
}

Tensor DiT::forward_with_cond(const Tensor& x, const Tensor& cond) {
    int B = static_cast<int>(x.rows);
    Tensor h = patch_embed_.forward(x);  // (B, S*d_model)
    int S = static_cast<int>(h.cols) / d_model_;
    for (int i = 0; i < depth_; ++i) {
        h = blocks_[i]->forward(h, cond);
    }
    // Final LN per-token
    Tensor h_norm(B, S * d_model_);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            Tensor tok(1, d_model_);
            for (int d = 0; d < d_model_; ++d) tok[0][d] = h[b][s * d_model_ + d];
            Tensor tok_ln = final_ln_.forward(tok);
            for (int d = 0; d < d_model_; ++d) h_norm[b][s * d_model_ + d] = tok_ln[0][d];
        }
    }
    h = h_norm;
    // Unpatchify: reshape (B, S*d_model) -> (B*S, d_model), apply Dense, reshape back
    // (per-token linear). The unpatchify Dense projects d_model -> out_dim*patch_len.
    Tensor out = unpatchify_.forward(h);  // (B, S*out_dim*patch_len) = (B, T*in_dim)
    return out;
}

Tensor DiT::forward(const Tensor& x, int t, int class_idx) {
    Tensor t_emb = time_embed_.forward(t);  // (1, d_model)
    Tensor cond(1, cond_dim_);
    for (int i = 0; i < d_model_; ++i) cond[0][i] = t_emb[0][i];
    if (label_embed_ != nullptr) {
        Tensor y_emb = label_embed_->forward(class_idx < 0 ? 0 : class_idx);
        for (int i = 0; i < d_model_; ++i) cond[0][d_model_ + i] = y_emb[0][i];
    } else {
        for (int i = 0; i < d_model_; ++i) cond[0][d_model_ + i] = 0.0;
    }
    return forward_with_cond(x, cond);
}

std::vector<Tensor*> DiT::parameters() {
    std::vector<Tensor*> p;
    for (auto* t : patch_embed_.parameters()) p.push_back(t);
    for (auto* t : unpatchify_.parameters()) p.push_back(t);
    for (auto* t : time_embed_.parameters()) p.push_back(t);
    if (label_embed_) for (auto* t : label_embed_->parameters()) p.push_back(t);
    for (auto& b : blocks_) for (auto* t : b->parameters()) p.push_back(t);
    for (auto* t : final_ln_.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> DiT::gradients() {
    std::vector<Tensor*> g;
    for (auto* t : patch_embed_.gradients()) g.push_back(t);
    for (auto* t : unpatchify_.gradients()) g.push_back(t);
    for (auto* t : time_embed_.gradients()) g.push_back(t);
    if (label_embed_) for (auto* t : label_embed_->gradients()) g.push_back(t);
    for (auto& b : blocks_) for (auto* t : b->gradients()) g.push_back(t);
    for (auto* t : final_ln_.gradients()) g.push_back(t);
    return g;
}

void DiT::zero_grad() {
    patch_embed_.zero_grad();
    unpatchify_.zero_grad();
    time_embed_.zero_grad();
    if (label_embed_) label_embed_->zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    final_ln_.zero_grad();
}

// =============================================================================
// DiTDiffusion
// =============================================================================
DiTDiffusion::DiTDiffusion(int d_model, int depth, int n_heads, int in_dim, int patch_len,
                           int T, int num_classes, double mlp_ratio,
                           float beta_start, float beta_end)
    : dit_(d_model, depth, n_heads, in_dim, patch_len, num_classes, mlp_ratio),
      scheduler_(T, beta_start, beta_end),
      T_(T), num_classes_(num_classes),
      sample_rng_(42) {}

Tensor DiTDiffusion::add_noise(const Tensor& x0, int t, const Tensor& noise) const {
    return scheduler_.q_sample(x0, t, noise);
}

double DiTDiffusion::training_loss(const Tensor& x0, int t, int class_idx) {
    int B = static_cast<int>(x0.rows);
    int D = static_cast<int>(x0.cols);
    std::mt19937 noise_rng(static_cast<unsigned>(t) * 1000u + 7u);
    std::normal_distribution<double> nd(0.0, 1.0);
    Tensor noise(B, D);
    for (size_t i = 0; i < noise.rows * noise.cols; ++i)
        (&noise[0][0])[i] = nd(noise_rng);
    Tensor x_t = scheduler_.q_sample(x0, t, noise);
    Tensor eps_pred = dit_.forward(x_t, t, class_idx);
    // MSE loss
    double sum = 0.0;
    for (size_t i = 0; i < eps_pred.rows * eps_pred.cols; ++i) {
        double diff = (&eps_pred[0][0])[i] - (&noise[0][0])[i];
        sum += diff * diff;
    }
    double loss = sum / static_cast<double>(B * D);
    last_x0_ = x0;
    last_noise_ = noise;
    last_x_t_ = x_t;
    last_eps_pred_ = eps_pred;
    last_t_ = t;
    last_class_idx_ = class_idx;
    // Pre-compute gradient of loss w.r.t. eps_pred: 2/N * (eps_pred - noise)
    last_grad_eps_pred_ = Tensor(B, D);
    double N = static_cast<double>(B * D);
    for (size_t i = 0; i < eps_pred.rows * eps_pred.cols; ++i) {
        double diff = (&eps_pred[0][0])[i] - (&noise[0][0])[i];
        (&last_grad_eps_pred_[0][0])[i] = 2.0 / N * diff;
    }
    return loss;
}

Tensor DiTDiffusion::backward(double /*lr*/) {
    // Placeholder: DiT doesn't currently support backward — return zero gradient.
    // For real gradient flow, use the per-parameter FD gradient check.
    // This stub keeps the API surface complete for the test suite.
    Tensor zero_grad(last_eps_pred_.rows, last_eps_pred_.cols);
    return zero_grad;
}

Tensor DiTDiffusion::sample(int B, int n_steps, int class_idx, unsigned int seed) {
    if (n_steps < 0) n_steps = T_;
    if (T_ % n_steps != 0) {
        throw std::invalid_argument("DiTDiffusion::sample: n_steps must divide T");
    }
    int step_size = T_ / n_steps;
    int D;
    if (last_x0_.rows > 0 && last_x0_.cols > 0) {
        D = static_cast<int>(last_x0_.cols);
    } else {
        // Estimate from in_dim*patch_len*num_patches — for now require training_loss to be called first
        throw std::logic_error("DiTDiffusion::sample: call training_loss first to cache the input shape");
    }
    std::normal_distribution<double> nd(0.0, 1.0);
    std::mt19937 rng(seed);
    Tensor x(B, D);
    for (size_t i = 0; i < x.rows * x.cols; ++i) (&x[0][0])[i] = nd(rng);
    for (int step = 0; step < n_steps; ++step) {
        int t = T_ - 1 - step * step_size;
        Tensor eps_pred = dit_.forward(x, t, class_idx);
        float sqrt_alpha_bar_t = scheduler_.sqrt_alphas_cumprod(t);
        float alpha_bar_t = sqrt_alpha_bar_t * sqrt_alpha_bar_t;
        float alpha_bar_tm1;
        if (t - step_size >= 0) {
            float sab = scheduler_.sqrt_alphas_cumprod(t - step_size);
            alpha_bar_tm1 = sab * sab;
        } else {
            alpha_bar_tm1 = 1.0f;
        }
        float beta_t = scheduler_.betas[t];
        float alpha_t = 1.0f - beta_t;
        // posterior mean formula:
        // x0_pred = (x - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
        // mean = (sqrt(alpha_bar_{t-1})*beta_t / (1 - alpha_bar_t)) * x0_pred
        //      + (sqrt(alpha_t)*(1-alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x
        float coef1 = std::sqrt(alpha_bar_tm1) * beta_t / std::max(1.0f - alpha_bar_t, 1e-8f);
        float coef2 = std::sqrt(alpha_t) * (1.0f - alpha_bar_tm1) / std::max(1.0f - alpha_bar_t, 1e-8f);
        float inv_sqrt_alpha_bar = 1.0f / std::sqrt(std::max(alpha_bar_t, 1e-8f));
        float sqrt_one_minus_alpha_bar = std::sqrt(std::max(1.0f - alpha_bar_t, 1e-8f));
        for (size_t i = 0; i < x.rows * x.cols; ++i) {
            double xi = (&x[0][0])[i];
            double ei = (&eps_pred[0][0])[i];
            double x0_pred = (xi - sqrt_one_minus_alpha_bar * ei) * inv_sqrt_alpha_bar;
            double mean = coef1 * x0_pred + coef2 * xi;
            (&x[0][0])[i] = mean;
        }
        // Add noise if not at t=0
        if (t > 0) {
            float post_var = scheduler_.posterior_variance(t);
            double std_dev = std::sqrt(std::max(post_var, 0.0f));
            for (size_t i = 0; i < x.rows * x.cols; ++i) {
                (&x[0][0])[i] += std_dev * nd(rng);
            }
        }
    }
    return x;
}

}  // namespace nn