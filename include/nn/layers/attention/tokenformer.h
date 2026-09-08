#ifndef TOKENFORMER_H
#define TOKENFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>

// ============================================================================
// Tokenformer / Pattention — "Tokenized-Parameter Attention Projection"
//   Wang et al., 2024, "Tokenformer: Rethinking Transformer Scaling with
//   Tokenized Model Parameters", https://arxiv.org/abs/2410.23168
//
// Standard Dense projection:
//     out = X · W^T + b                  W ∈ R^{d_out × d_in}    (matrix of params)
// Pattention replaces the parameter MATRIX with two parameter TOKEN SETS:
//     K_p ∈ R^{P × d_in}                 (parameter keys)
//     V_p ∈ R^{P × d_out}                (parameter values)
//     g   ∈ R^{P}                        (per-token gate, in (0, 1))
//     b   ∈ R^{1 × d_out}                (optional bias)
//
//   S     = X · K_p^T                    in R^{n × P}      (input × param-keys)
//   α_p   = exp(S_p) / Σ_{p'} exp(S_{p'})            (row softmax over param tokens)
//   α_g   = g · α / Σ_{p'} g_{p'} α_{p'}              (gated & re-normalized to sum to 1)
//   out   = α_g · V_p + b                in R^{n × d_out}
//
// The defining property (paper §3): P is INDEPENDENT of d_out. You can grow P
// by APPENDING zero-valued parameter tokens (K_p=0, V_p=0, g=0); the output is
// bit-exact unchanged, because:
//   - the new K_p rows contribute 0 to S (= X·0),
//   - the new g = 0 zeros out their softmax weights before re-normalization,
//   - those zero-gated rows then contribute exactly zero to the output sum.
// Test 5 (token-append invariance) pins this as a bit-exact invariant.
//
// When P = d_out and V_p = I_d_out, K_p = W, b = 0, Pattention collapses to a
// standard Dense projection — Test 6 (reduction-to-Dense sanity) pins this.
//
// Backward (hand-derived; see task plan for full derivation):
//   For each row t:
//     dV_p[p, :] += α_g[t, p] · dO[t, :]
//     db[:]      += dO[t, :]
//     dα_g[t, p] = dO[t, :] · V_p[p, :]^T
//     dα[t, p]   = g[p] · ( dα_g[t, p] − Σ_{p'} g[p'] · α[t, p'] · dα_g[t, p'] ) / Z_g
//                 (where Z_g = Σ g·α; if all g=1, this is the standard softmax backward)
//     dS[t, p]   = α[t, p] · ( dα[t, p] − Σ_{p'} α[t, p'] · dα[t, p'] )
//     dK_p[p,:] += dS[t, p] · X[t, :]
//     dX[t, :]  += Σ_p dS[t, p] · K_p[p, :]
//     dg[p]     += Σ_t ( α[t, p]/Z_g · ( dα_g[t, p] − Σ_{p'} g[p'] · α[t, p'] · dα_g[t, p'] / Z_g )
//                     − Σ_{p'} ...   ...  )   (closed-form from gating)
//
// Conventions:
//   * Input  (n, d_in),  output  (n, d_out)
//   * Multi-head is NOT supported here — Pattention is an FFN slot by design.
//   * Random init: K_p, V_p ~ Tensor::random(0.1), b = 0, g = sigmoid(logit_g)
//     with logit_g ~ Tensor::random(0.5) so g starts ~uniform(0.4, 0.6) and is
//     learnable per-token (a "soft mask" that learns to prune parameter tokens).
// ============================================================================

class Pattention : public Layer {
public:
    Pattention(size_t d_in, size_t d_out, size_t num_param_tokens);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return K_p; }
    Tensor get_gradients() const override { return grad_K_p; }
    std::string name() const override { return "Pattention"; }

    // Accessors
    size_t d_in()  const { return d_in_; }
    size_t d_out() const { return d_out_; }
    size_t num_param_tokens() const { return P_; }

    const Tensor& last_input()  const { return last_input_; }
    const Tensor& last_S()      const { return last_S_; }
    const Tensor& last_alpha()  const { return last_alpha_; }
    const Tensor& last_alpha_g() const { return last_alpha_g_; }

    // Public parameters so tests can perturb them directly for FD checks.
    Tensor K_p;        // (P, d_in)    — parameter keys
    Tensor V_p;        // (P, d_out)   — parameter values
    Tensor b;          // (1, d_out)   — bias
    Tensor logit_g;    // (P, 1)       — pre-sigmoid gate logits; g = sigmoid(logit_g)
    Tensor grad_K_p;   // (P, d_in)
    Tensor grad_V_p;   // (P, d_out)
    Tensor grad_b;     // (1, d_out)
    Tensor grad_logit_g;  // (P, 1)

private:
    size_t d_in_;
    size_t d_out_;
    size_t P_;
    Tensor last_input_;     // (n, d_in)
    Tensor last_S_;         // (n, P) — pre-softmax scores
    Tensor last_alpha_;     // (n, P) — softmax mixing weights (before gating)
    Tensor last_alpha_g_;   // (n, P) — gated, re-normalized mixing weights
    Tensor last_g_;         // (P)    — cached gate values
    Tensor last_Z_g_;       // (n)    — cached Σ g·α per row
};

// ----------------------------------------------------------------------------
// TokenformerBlock — pre-LN causal self-attn -> residual -> pre-LN
// Pattention FFN -> residual. Mirrors the convention of StickBreakingBlock /
// SHLA / based: Dense W_q/W_k/W_v/W_o for the self-attention slot, then a
// single Pattention layer in the FFN slot.
//
// Constructor: TokenformerBlock(d_model, num_heads, num_param_tokens).
// ----------------------------------------------------------------------------
class TokenformerBlock : public Layer {
public:
    TokenformerBlock(size_t d_model, size_t num_heads, size_t num_param_tokens);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn_W_q.weights; }
    Tensor get_gradients() const override { return attn_W_q.grad_weights; }
    std::string name() const override { return "TokenformerBlock"; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_ = 0;     // set after validation
    Tensor last_input_;     // (n, d_model)
    Tensor last_q_; Tensor last_k_; Tensor last_v_;
    std::vector<Tensor> last_scores_;   // per-head (n, n) softmax probs from forward

public:
    // Public so tests can poke them. (Order matches the constructor initializer list.)
    Dense attn_W_q, attn_W_k, attn_W_v, attn_W_o;
    std::unique_ptr<LayerNorm> ln1_;       // pre-attn
    std::unique_ptr<LayerNorm> ln2_;       // pre-FFN
    std::unique_ptr<Pattention> ffn_;      // Pattention FFN
};

// ----------------------------------------------------------------------------
// TokenformerModel — input projection -> N blocks -> final LN -> classifier.
// ----------------------------------------------------------------------------
class TokenformerModel : public Layer {
public:
    TokenformerModel(size_t d_input, size_t d_model, size_t d_output,
                     size_t num_blocks, size_t num_heads, size_t num_param_tokens);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj.weights; }
    Tensor get_gradients() const override { return input_proj.grad_weights; }
    std::string name() const override { return "TokenformerModel"; }

private:
    size_t d_input_;
    size_t d_model_;
    size_t d_output_;

public:
    Dense input_proj;              // (d_input, d_model)
    Dense classifier;              // (d_model, d_output)
    std::vector<std::unique_ptr<TokenformerBlock>> blocks_;
    std::unique_ptr<LayerNorm> final_ln_;
};

#endif
