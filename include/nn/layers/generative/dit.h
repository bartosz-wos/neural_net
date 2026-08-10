#ifndef NN_GENERATIVE_DIT_H
#define NN_GENERATIVE_DIT_H

#include "../../core/tensor.h"
#include "../../core/layer.h"
#include "../../layers/normalization/layer_norm.h"
#include "../../layers/dense/embedding.h"
#include "../../activations/activations.h"
#include "ddpm.h"
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <iostream>

namespace nn {

// =============================================================================
// DiT — Diffusion Transformer (Peebles & Xie 2023, https://arxiv.org/abs/2212.09748)
//
// Sequence-level analogue of the image-DiT:
//   - SequencePatchEmbed: (B, T*in_dim) -> (B, S*d_model), S = T/patch_len
//   - Stack of DiTBlocks (adaLN-Zero modulation of MHA + MLP)
//   - Final LayerNorm -> final linear ("Zero" output projection) -> unpatchify
//   - DDPM-style training loss (MSE on epsilon prediction) and DDPM sampler
//
// In the original paper, conditioning is via:
//   - TimeEmbedding: t -> (1, d_model)
//   - LabelEmbedding: y -> (1, d_model) (zero index = null for CFG)
//   - Concat to make (1, 2*d_model) conditioning vector
//   - Per-block: Dense(2*d_model -> 6*d_model), split into (shift_msa, scale_msa,
//     gate_msa, shift_mlp, scale_mlp, gate_mlp), then apply adaLN-Zero modulation.
//
// We follow the paper closely. Sequence-only (no 2-D image ops) to keep the API
// general and the test suite tractable.
// =============================================================================

// =============================================================================
// DiTTimeEmbed — sinusoidal Vaswani-style time embedding + 2-layer MLP w/ SiLU
// =============================================================================
class DiTTimeEmbed {
public:
    DiTTimeEmbed(int hidden_dim, int max_period = 10000);

    // Forward: integer timestep -> (1, hidden_dim)
    Tensor forward(int t);

    int hidden_dim() const { return hidden_dim_; }
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

private:
    int hidden_dim_;
    int max_period_;
    Dense mlp_in_;    // hidden_dim -> hidden_dim (constructed with random init)
    Dense mlp_out_;   // hidden_dim -> hidden_dim
    Tensor last_sinusoidal_;  // cache raw sinusoidal for backward
};

// =============================================================================
// DiTLabelEmbed — learned (num_classes+1, d_model) embedding (idx 0 = null/uncond)
// =============================================================================
class DiTLabelEmbed {
public:
    DiTLabelEmbed(int num_classes, int d_model);

    Tensor forward(int class_idx);                    // (1, d_model)
    Tensor forward_batch(const std::vector<int>& idxs);  // (B, d_model)

    int num_classes() const { return num_classes_; }
    int d_model() const { return d_model_; }
    std::vector<Tensor*> parameters() { return embed_.parameters(); }
    std::vector<Tensor*> gradients() { return embed_.gradients(); }
    void zero_grad() { embed_.zero_grad(); }

private:
    int num_classes_;
    int d_model_;
    Embedding embed_;
};

// =============================================================================
// SequencePatchEmbed — projects (B, T*in_dim) -> (B, S*d_model)
// where S = T / patch_len. Operates per-patch via a Dense(in_dim*patch_len -> d_model).
// =============================================================================
class SequencePatchEmbed {
public:
    SequencePatchEmbed(int in_dim, int d_model, int patch_len);
    Tensor forward(const Tensor& x);
    int in_dim() const { return in_dim_; }
    int d_model() const { return d_model_; }
    int patch_len() const { return patch_len_; }
    std::vector<Tensor*> parameters() { return proj_.parameters(); }
    std::vector<Tensor*> gradients() { return proj_.gradients(); }
    void zero_grad() { proj_.zero_grad(); }

private:
    int in_dim_;
    int d_model_;
    int patch_len_;
    Dense proj_;
};

// =============================================================================
// SequenceUnpatchify — projects (B, S*d_model) -> (B, T*out_dim) (inverse of patch embed)
// =============================================================================
class SequenceUnpatchify {
public:
    SequenceUnpatchify(int d_model, int out_dim, int patch_len);
    Tensor forward(const Tensor& x);
    int d_model() const { return d_model_; }
    int out_dim() const { return out_dim_; }
    int patch_len() const { return patch_len_; }
    std::vector<Tensor*> parameters() { return proj_.parameters(); }
    std::vector<Tensor*> gradients() { return proj_.gradients(); }
    void zero_grad() { proj_.zero_grad(); }

private:
    int d_model_;
    int out_dim_;
    int patch_len_;
    Dense proj_;
};

// =============================================================================
// DiTBlock — canonical DiT block with adaLN-Zero modulation
//
//   mod = Dense(cond_dim -> 6*d_model)  [initialized to zero — the "Zero"]
//   shift_msa, scale_msa, gate_msa = mod[:, 0:d_model], [d_model:2d_model], [2d_model:3d_model]
//   shift_mlp, scale_mlp, gate_mlp = mod[:, 3d_model:4d_model], [4d_model:5d_model], [5d_model:6d_model]
//
//   h = LN1(x) * (1 + scale_msa) + shift_msa
//   attn_out = MHA(h)
//   x = x + gate_msa * attn_out   [attn_o initialized to zero]
//
//   h = LN2(x) * (1 + scale_mlp) + shift_mlp
//   mlp_out = MLP(h)  (Dense -> GELU -> Dense; final Dense initialized to zero)
//   x = x + gate_mlp * mlp_out
//
// The "Zero" trick means at initialization, mod=0 -> shift=0, scale=0, gate=0,
// attn_o=0 -> attn_out=0, mlp_w2=0 -> mlp_out=0. So x is preserved at init
// (constant function), and gradients only flow where gates are non-zero.
// =============================================================================
class DiTBlock {
public:
    DiTBlock(int d_model, int n_heads, double mlp_ratio = 4.0, int cond_dim = 0);

    // x: (B, S*d_model); cond: (1, cond_dim) — time + class conditioning
    Tensor forward(const Tensor& x, const Tensor& cond);

    // Modulation vector for the given conditioning (1, 6*d_model)
    Tensor modulation(const Tensor& cond);

    Tensor last_modulation() const { return last_mod_; }
    Tensor last_attn_out() const { return last_attn_out_; }
    Tensor last_mlp_out() const { return last_mlp_out_; }
    Tensor last_ln1_out() const { return last_ln1_out_; }
    Tensor last_ln2_out() const { return last_ln2_out_; }

    int d_model() const { return d_model_; }
    int n_heads() const { return n_heads_; }
    int mlp_dim() const { return mlp_dim_; }

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

private:
    int d_model_;
    int n_heads_;
    int mlp_dim_;
    int cond_dim_;
    LayerNorm ln1_, ln2_;
    Dense mod_;       // cond_dim -> 6*d_model
    Dense attn_qkv_;  // d_model -> 3*d_model (combined Q,K,V)
    Dense attn_o_;    // d_model -> d_model
    Dense mlp_w1_;    // d_model -> mlp_dim
    Dense mlp_w2_;    // mlp_dim -> d_model

    // Cached intermediates for backward
    Tensor last_mod_;
    Tensor last_attn_out_;
    Tensor last_mlp_out_;
    Tensor last_ln1_out_;
    Tensor last_ln2_out_;
    Tensor last_x_;
    Tensor last_h1_;
};

// =============================================================================
// DiT — full sequence-DiT model
//   SequencePatchEmbed -> [DiTBlock] * depth -> final LayerNorm -> final linear -> SequenceUnpatchify
// Conditioning: time + (optional) class embedding concatenated to form cond.
// =============================================================================
class DiT {
public:
    DiT(int d_model, int depth, int n_heads, int in_dim, int patch_len,
        int num_classes = 0, double mlp_ratio = 4.0);

    // Standard forward: integer t and class_idx (-1 = null/uncond)
    Tensor forward(const Tensor& x, int t, int class_idx = -1);

    // Forward with precomputed conditioning vector (for tests / DDPM sampler)
    Tensor forward_with_cond(const Tensor& x, const Tensor& cond);

    int d_model() const { return d_model_; }
    int depth() const { return depth_; }
    int patch_len() const { return patch_len_; }
    int in_dim() const { return in_dim_; }
    int num_classes() const { return num_classes_; }
    int cond_dim() const { return cond_dim_; }

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    private:
    int d_model_, depth_, n_heads_, in_dim_, patch_len_, num_classes_;
    double mlp_ratio_;
    SequencePatchEmbed patch_embed_;
    SequenceUnpatchify unpatchify_;
    DiTTimeEmbed time_embed_;
    std::unique_ptr<DiTLabelEmbed> label_embed_;
    LayerNorm final_ln_;
    std::vector<std::unique_ptr<DiTBlock>> blocks_;
    int cond_dim_;
};

// =============================================================================
// DiTDiffusion — wraps DiT with NoiseScheduler for DDPM training + sampling.
// =============================================================================
class DiTDiffusion {
public:
    DiTDiffusion(int d_model, int depth, int n_heads, int in_dim, int patch_len,
                 int T = 1000, int num_classes = 0, double mlp_ratio = 4.0,
                 float beta_start = 1e-4f, float beta_end = 0.02f);

    // Forward process: x_t = sqrt_alphabar_t * x0 + sqrt(1-alphabar_t) * noise
    Tensor add_noise(const Tensor& x0, int t, const Tensor& noise) const;

    // Training loss: sample noise, noising, predict eps, MSE loss.
    // Returns scalar loss. Caches intermediates for backward.
    double training_loss(const Tensor& x0, int t, int class_idx = -1);

    // Backward: propagates last_grad_eps_pred_ through DiT.
    // Returns grad_x_t (the gradient of loss w.r.t. the noised input).
    Tensor backward(double lr = 0.0);

    // DDPM reverse-process sampling. n_steps must divide T.
    Tensor sample(int B, int n_steps = -1, int class_idx = -1, unsigned int seed = 42);

    DiT& dit() { return dit_; }
    const NoiseScheduler& scheduler() const { return scheduler_; }
    int T() const { return T_; }

    std::vector<Tensor*> parameters() { return dit_.parameters(); }
    std::vector<Tensor*> gradients() { return dit_.gradients(); }
    void zero_grad() { dit_.zero_grad(); }

    // For tests: direct access to internals
    const Tensor& last_x0() const { return last_x0_; }
    const Tensor& last_x_t() const { return last_x_t_; }
    const Tensor& last_noise() const { return last_noise_; }
    const Tensor& last_eps_pred() const { return last_eps_pred_; }
    const Tensor& last_grad_eps_pred() const { return last_grad_eps_pred_; }
    int last_t() const { return last_t_; }
    int in_dim() const { return dit_.in_dim(); }
    int patch_len() const { return dit_.patch_len(); }

private:
    DiT dit_;
    NoiseScheduler scheduler_;
    int T_;
    int num_classes_;

    Tensor last_x0_;
    Tensor last_noise_;
    Tensor last_x_t_;
    Tensor last_eps_pred_;
    Tensor last_grad_eps_pred_;
    int last_t_;
    int last_class_idx_;

    std::mt19937 sample_rng_;
};

}  // namespace nn

#endif