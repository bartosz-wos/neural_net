#ifndef LINEAR_ATTENTION_H
#define LINEAR_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <string>

// ============================================================================
// Linear Attention — Katharopoulos, Vyas, Pappas, Fleuret, Hovsepian, Puri
// "Transformers are RNNs: Fast Autoregressive Transformers with Linear
//  Attention" (ICML 2020, https://arxiv.org/abs/2006.16236).
//
// The canonical kernel-feature-map linear attention baseline: replace the
// softmax(QK^T) V path with
//
//   O = φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))
//
// where φ(x) = elu(x) + 1 is the feature map proposed in §3.1 of the paper.
// This brings the per-step compute from O(n^2 d) to O(n d^2) (i.e. linear in
// the sequence length n), and crucially makes the *recurrent* and *parallel*
// views of the attention computation formally identical.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (per single-head layer, input x ∈ R^{n × d_model}):
//
//   1. Q = x W_q^T + b_q  ∈ R^{n × d_k}     (d_k = d_model)
//   2. K = x W_k^T + b_k  ∈ R^{n × d_k}
//   3. V = x W_v^T + b_v  ∈ R^{n × d_k}
//   4. φ(Q) = elu(Q) + 1   ∈ R^{n × d_k}   (row-wise, same map per paper §3.1)
//   5. φ(K) = elu(K) + 1
//   6. KV = φ(K)^T V       ∈ R^{d_k × d_k}  -- the linear-time trick
//   7. Z  = φ(K)^T 1_n     ∈ R^{d_k}        -- normaliser
//   8. numerator = φ(Q) @ KV              ∈ R^{n × d_k}
//   9. denominator = φ(Q) @ Z + eps_safe  ∈ R^{n}      (eps_safe = 1e-6)
//   10. context = numerator / denominator (per-row division)  ∈ R^{n × d_k}
//   11. out = context W_o^T + b_o         ∈ R^{n × d_model}
//
// Step 6 is the entire point: O(n d_k^2) instead of O(n^2 d_k). Everything
// else is identical to standard multi-head attention in cost.
//
// ----------------------------------------------------------------------------
// Backward
//
// The forward is a composition of: y = (A B) / (A c + e) where A = φ(Q) (n×d),
// B = φ(K)^T V (d×d), c = φ(K)^T 1 (d,), e = eps. With g = grad_context (the
// upstream gradient w.r.t. the per-row-divide output):
//
//   Let dDen = φ(Q) @ Z,  u = dDen + e,  s = g .* (φ(Q) B).div(u)
//   dNumer = g / u
//   dA_num = dNumer @ B^T                (back through numerator)
//   dA_den = -s ⊙ dNumer                (back through divide)
//   dZ     = A^T @ dA_den
//   dB     = A^T @ dNumer
//   dφK.V  = dZ (then split into dV += φ(K) @ dZ^T and dφK += V @ dZ as the
//              φ(K)^T V product)
//   dφK    = +dφK from V branch AND dZ from the Z branch
//   dφK_b  = dφK (this is the φ(K) gradient for the Z=φ(K)^T 1 reduction:
//                d(φ(K)^T 1)/dφK = 1_n 1_d)
//   dφK += dφK_b (1_n outer-product reduction along sequence)
//
// Concretely for our parametrisation:
//   dφK_T_V_branch: dV[i, j]   += Σ_n φ(K)[n, i] * dZ[n, j]
//                   dφK[n, i]  += Σ_j V[n, j] * dZ[n, j]     (per-row chain)
//                                  → phiK_grad += V . dZ   (element-wise, n×d)
//   dφK_Z_branch:    phiK_grad[n, i] += dZ_sum[i]            where dZ_sum = A^T @ dA_den
//   dφQ:   phiQ_grad = (dNumer @ B^T) + ((-s) ⊙ dNumer) @ 1_d   … NO:
//          phiQ gradient has TWO contributions:
//          (a) numerator chain: dA = dNumer @ B^T  (n×d)
//          (b) denominator chain: dA += -dA_den_per_row  where
//              dA_den_per_row[n, i] = (g[n] .* (-(φ(Q) B)[n] / u[n]^2)) .* c[i]
//                                   = -s[n] * c[i]
//              i.e. dA_den[n, i] = -s[n] * c[i] (broadcast across feature dim)
//          (b total): dφQ = dNumer @ B^T + outer(s, c) * (-1)
//
// All in O(n d^2) — no n^2 terms anywhere.
//
// ----------------------------------------------------------------------------
// Numerical stability
//
//   * The elu(x)+1 feature map is non-negative: elu(x)+1 ≥ 0, with
//     elu(0)+1 = 1, elu(-∞)+1 = 0, elu(∞)+1 = ∞. For inputs in a
//     reasonable range (the W_q/W_k projections are Xavier-init) the
//     values stay bounded.
//   * The denominator φ(Q) Z + eps_safe is floored at 1e-6 per row to
//     avoid 0/0 when all keys are "far" from a query. Per-row division.
//   * We do not normalise Q/K explicitly (the paper does not, either).
//
// ----------------------------------------------------------------------------
// API conventions (matching Performer / Linformer in this repo)
//
//   * (n, d_model) input/output — row-major.
//   * Single-head: for multi-head, call multiple LinearAttentions and
//     concatenate. (Same convention as Linformer/Performer.)
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual).
//
// LinearAttention   — the attention layer itself
// LinearAttentionBlock     — pre-LN transformer block wrapper
// LinearAttentionModel     — stack of blocks + classifier
// ============================================================================

class LinearAttention : public Layer {
public:
    // d_model: input/output feature dim. Single-head, so d_k = d_model.
    LinearAttention(size_t d_model);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "LinearAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    const Tensor& W_q_tensor() const { return W_q; }
    const Tensor& W_k_tensor() const { return W_k; }
    const Tensor& W_v_tensor() const { return W_v; }
    const Tensor& W_o_tensor() const { return W_o; }

private:
    size_t d_model_;
    double eps_safe_;  // 1e-6 — denominator floor

    // Learned Q/K/V/O projections — Dense convention: y = x W^T + b
    // W shape: (d_model, d_model) so that y = x @ W^T (matching layer.h Dense)
    Tensor W_q, W_k, W_v, W_o;
    Tensor b_q, b_k, b_v, b_o;  // (1, d_model) biases

    // Gradients
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    // Caches for backward (set on forward, used on backward)
    bool has_forward_cache_;
    Tensor last_input_;      // (n, d_model) — input to forward
    Tensor last_Q_;          // (n, d_model)
    Tensor last_K_;          // (n, d_model)
    Tensor last_V_;          // (n, d_model)
    Tensor last_phiQ_;       // (n, d_model)
    Tensor last_phiK_;       // (n, d_model)
    Tensor last_KV_;         // (d_model, d_model) = φ(K)^T V
    Tensor last_Z_;          // (d_model,)        = φ(K)^T 1
    Tensor last_context_;    // (n, d_model)      = (φ(Q) KV) / (φ(Q) Z + eps)
};

// ----------------------------------------------------------------------------
// LinearAttentionBlock — pre-LN residual transformer block with linear
// attention. Two pre-LayerNorms (one for the attention path, one for the FFN),
// an attention layer, a 2-layer FFN, and residual connections. Both Q and KV
// of the attention layer see the same normalised tensor; FFN is GELU-hidden.
// ----------------------------------------------------------------------------
class LinearAttentionBlock : public Layer {
public:
    LinearAttentionBlock(size_t d_model, size_t ffn_hidden = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "LinearAttentionBlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_hidden() const { return ffn_hidden_; }
    const LinearAttention& attention() const { return attn_; }

    // Public accessor for the last input tensor's row count — used by the
    // outer model to recover the pre-pool sequence length for the mean-pool
    // backward pass.
    size_t last_input_rows() const { return last_input_.rows; }
    size_t last_input_cols() const { return last_input_.cols; }

private:
    size_t d_model_;
    size_t ffn_hidden_;

    LayerNorm ln1_;
    LayerNorm ln2_;
    LinearAttention attn_;
    Dense ffn1_;       // d_model -> ffn_hidden (or d_model if ffn_hidden==0)
    Dense ffn2_;       // ffn_hidden -> d_model (or d_model if ffn_hidden==0)

    Tensor last_input_;
    Tensor last_normed1_;
    Tensor last_attn_out_;
    Tensor last_residual1_;
    Tensor last_normed2_;
    Tensor last_ffn1_pre_;   // pre-GELU activation (cached for backward)
    Tensor last_ffn1_out_;   // post-GELU activation
    Tensor last_ffn2_out_;
};

// ----------------------------------------------------------------------------
// LinearAttentionModel — stack of N LinearAttentionBlocks + final classifier.
// Same convention as PerformerModel / LinformerModel in this repo.
// ----------------------------------------------------------------------------
class LinearAttentionModel : public Layer {
public:
    LinearAttentionModel(size_t d_input, size_t d_model, size_t d_output,
                         size_t num_blocks = 2, size_t num_heads = 1,
                         size_t ffn_hidden = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "LinearAttentionModel"; }

    size_t d_model() const { return d_model_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t num_heads() const { return num_heads_; }
    const LinearAttentionBlock& block(size_t i) const { return blocks_[i]; }

private:
    size_t d_input_;
    size_t d_model_;
    size_t d_output_;
    size_t num_blocks_;
    size_t num_heads_;

    Dense input_proj_;
    std::vector<LinearAttentionBlock> blocks_;
    Dense classifier_;
};

#endif // LINEAR_ATTENTION_H
