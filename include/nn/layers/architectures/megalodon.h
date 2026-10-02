#ifndef MEGALODON_H
#define MEGALODON_H

#include "../../core/layer.h"
#include "../normalization/timestep_norm.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/complex_ema.h"
#include "../utility/swiglu.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// Megalodon — MEGALODON §3.3 (normalized attention) + §3.4 (two-hop residual)
//   Ma, Yang, Xiong, Chen, Yu, Zhang, Ma, Zhou, 2024.
//   "MEGALODON: Efficient LLM Pretraining and Inference with Unlimited
//   Context Length", https://arxiv.org/abs/2404.08801
//
// A Megalodon block is a pre-norm residual sequence mixer. For input
// x ∈ R^(n × d_model):
//
//   tsn  = TimestepNorm(x)                         §3.2  (causal, cumulative)
//   x'   = ComplexEMA(tsn)                         §3.1  (complex EMA kernel)
//   z    = W_z · x' + b_z                          ∈ R^(n × z_dim)
//   z'   = per-head RMS normalization of z          §3.3
//   Q    = kappa_q ⊙ z' + mu_q                    ∈ R^(n × z_dim)
//   K    = kappa_k ⊙ z' + mu_k                    ∈ R^(n × z_dim)
//   V    = SiLU(W_v · tsn + b_v)                  ∈ R^(n × v_dim)
//   A    = softmax_masked(Q K^T / sqrt(d_h))
//   H    = concat_h (Σ_s A[t,s] · V[s,h]) · W_o + b_o
//   Ŷ    = H + x                                   attention residual
//   Y    = FFN(LayerNorm(Ŷ)) + x                   TWO-HOP residual
//
// Two design points worth calling out:
//
// (1) The `‖Z‖` division in Eq. 5 is what lets the paper drop the separate
//     τ(X) scaling term AND the SiLU on Z — the norm supplies the
//     nonlinearity. The released implementation applies the normalization as
//     an RMSNorm over each head's slice, which is the same operation up to
//     the constant factor sqrt(d_h) (RMSNorm divides by rms = ‖z‖/sqrt(d_h));
//     the paper divides by ‖z‖). We follow the reference's per-head RMSNorm
//     because it is better conditioned and it composes with the learned
//     kappa (q_gamma/k_gamma) exactly as the reference's `gamma` does.
//
// (2) The FFN residual reuses the block input x, NOT the attention output Ŷ.
//     This is the paper's §3.4 fix: it removes Mega's update gate entirely
//     and stops the variance growth of deep pre-norm stacks. It is also the
//     single easiest thing to get wrong in the backward pass — `grad_x` must
//     accumulate from BOTH the attention residual and the FFN residual.
//     tests/test_megalodon.cpp Test 4 pins it directly.
//
// TimestepNorm is applied only before the attention sublayer; a plain
// LayerNorm sits before the FFN. The paper's reasoning (§3.4) is that
// LayerNorm is cheaper there and the attention output is already a mixture
// over timesteps, so it does not need the causal cumulative form.
//
// V is projected from the timestep-normalized input rather than from the CEMA
// output — matching the reference implementation, which computes
// `v = F.silu(self.wv(out_tsn))` from the pre-CEMA normalized stream.
//
// Documented v1 simplifications (deliberate, not silent omissions):
//   - No RoPE. The reference applies rotary embeddings inside the inner
//     attention, but the paper lists RoPE as an inherited Llama-2
//     hyperparameter rather than a Megalodon contribution; adding it would
//     need a separate FD-checked backward chain.
//   - No chunk-parallel / distributed plumbing (§3.5) — inference-only.
//   - No KV cache or incremental decoding.
//   - No dropout (repo-wide convention for sequence layers).
// ============================================================================

class MegalodonBlock : public Layer {
public:
    // d_model:   model width
    // num_heads: attention heads; must divide d_model
    // ffn_mult:  FFN hidden width = ffn_mult * d_model
    // num_groups: TimestepNorm groups (0 = single group over all features)
    // cema_ndim: complex EMA hidden states per channel (h in the paper)
    // chunk_size: 0 = full causal; > 0 = Megalodon-chunk, attention
    //             restricted to (same chunk) ∧ (s <= t)
    MegalodonBlock(size_t d_model, size_t num_heads = 1, size_t ffn_mult = 4,
                   size_t num_groups = 0, size_t cema_ndim = 16,
                   size_t chunk_size = 0);
    ~MegalodonBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_z_.weights; }
    Tensor get_gradients() const override { return W_z_.grad_weights; }
    std::string name() const override { return "MegalodonBlock"; }

    // Copy every learnable parameter (all sublayers) from a block with an
    // identical configuration. Throws on a configuration mismatch rather
    // than silently copying a prefix of a differently-shaped tensor.
    void copy_params_from(const MegalodonBlock& other);

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t z_dim()     const { return z_dim_; }
    size_t v_dim()     const { return v_dim_; }
    size_t ffn_hidden() const { return ffn_hidden_; }
    size_t num_groups() const { return num_groups_; }
    size_t cema_ndim() const { return cema_ndim_; }
    size_t chunk_size() const { return chunk_size_; }

    // --- Sublayers (public for direct inspection in tests, same convention as
    // --- ComplexEMA's public parameters). ---
    TimestepNorm tsn_;      // TimestepNorm(d_model, num_groups)
    ComplexEMA   cema_;     // ComplexEMA(d_model, cema_ndim)
    Dense W_z_;             // Dense(d_model, z_dim)
    Dense W_v_;             // Dense(d_model, v_dim)
    Dense W_o_;             // Dense(v_dim, d_model)
    LayerNorm ffn_ln_;      // LayerNorm(d_model)
    SwiGLU<Swish> ffn_;     // SwiGLU(d_model, ffn_hidden)
    Dense ffn_down_;        // Dense(ffn_hidden, d_model)

    // Per-element affine on the normalized z stream for Q and K. The
    // reference stores these as a single (2, z_dim) tensor `gamma`/`beta`.
    Tensor q_gamma_, q_beta_;  // (1, z_dim)
    Tensor k_gamma_, k_beta_;  // (1, z_dim)
    Tensor grad_q_gamma_, grad_q_beta_;
    Tensor grad_k_gamma_, grad_k_beta_;

    // --- Forward caches (exposed for tests and for FD checks) ---
    Tensor last_input;       // (n, d_model)  x
    Tensor last_tsn;         // (n, d_model)  TimestepNorm(x)
    Tensor last_cema;        // (n, d_model)  ComplexEMA(tsn)
    Tensor last_z_pre;       // (n, z_dim)    W_z · cema + b_z
    Tensor last_z_norm;      // (n, z_dim)    per-head RMS-normalized z
    Tensor last_rms;         // (n, num_heads) per-head rms (1/sqrt(mean+eps))
    Tensor last_q, last_k;   // (n, z_dim)
    Tensor last_v_pre;       // (n, v_dim)    W_v · tsn + b_v (pre-SiLU)
    Tensor last_v;           // (n, v_dim)    SiLU(v_pre)
    // Per-head attention scores, (n, num_heads*n): last_score_[t, h*n + s].
    // Kept per-head on purpose — see the note in forward().
    Tensor last_score_;
    // Mean over heads of the softmax attention, (n, n). Exposed for tests
    // and debugging only; the backward re-derives each head's distribution
    // from last_score_ and never reads this.
    Tensor last_attn;        // (n, n)
    Tensor last_h;           // (n, v_dim)    concat_h Σ_s A·V
    Tensor last_attn_out;    // (n, d_model)  W_o · h + b_o
    Tensor last_y_hat;       // (n, d_model)  x + attn_out
    Tensor last_ln_out;      // (n, d_model)  LayerNorm(y_hat)
    Tensor last_ffn_out;     // (n, d_model)  ffn_down(ffn(ln_out))
    Tensor grad_x_;          // (n, d_model)  gradient w.r.t. the block input

private:
    size_t d_model_, num_heads_, head_dim_;
    size_t z_dim_, v_dim_, ffn_mult_, ffn_hidden_;
    size_t num_groups_, cema_ndim_, chunk_size_;
    size_t T_ = 0;           // sequence length of the most recent forward
    double rms_eps_ = 1e-7;  // epsilon inside the per-head RMS normalization

    // True when position s is visible to query t. Full causal when
    // chunk_size_ == 0; Megalodon-chunk otherwise.
    bool visible(size_t t, size_t s) const {
        if (s > t) return false;
        if (chunk_size_ == 0) return true;
        return t / chunk_size_ == s / chunk_size_;
    }
};

// Stack of `num_layers` MegalodonBlock + input projection + final
// TimestepNorm + classifier.
class MegalodonModel : public Layer {
public:
    size_t input_dim_, d_model_, output_dim_, num_layers_;
    size_t num_heads_, ffn_mult_, num_groups_, cema_ndim_, chunk_size_;

    Dense input_proj;  // Dense(input_dim, d_model)
    std::vector<std::unique_ptr<MegalodonBlock>> blocks;
    TimestepNorm final_tsn;  // TimestepNorm(d_model, num_groups)
    Dense output_proj;       // Dense(d_model, output_dim)

    MegalodonModel(size_t input_dim, size_t d_model, size_t output_dim,
                   size_t num_layers, size_t num_heads = 1, size_t ffn_mult = 4,
                   size_t num_groups = 0, size_t cema_ndim = 16,
                   size_t chunk_size = 0);
    ~MegalodonModel() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj.weights; }
    Tensor get_gradients() const override { return input_proj.grad_weights; }
    std::string name() const override { return "MegalodonModel"; }
};

#endif  // MEGALODON_H
