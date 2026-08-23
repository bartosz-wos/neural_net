#ifndef COSFORMER_H
#define COSFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// cosFormer — Qin, Sun, Deng, Li, Wei, Lv, Yan, Kong, Zhong 2022
//   "cosFormer: Rethinking Softmax in Attention" (ICLR 2022,
//    https://arxiv.org/abs/2202.08791)
//
// A linear-time attention mechanism that replaces softmax attention with a
// kernel that is (i) non-negative and (ii) enforces a recency bias via a
// cosine distance re-weighting. Both properties are identified in the paper
// as the two crucial characteristics of softmax attention that prior kernel-
// method approximations (Linear Transformer / Performer) failed to capture.
//
// The cosine kernel:   s(Q_i, K_j) = ReLU(Q_i)^T ReLU(K_j) * cos(π/2 * (i-j)/M)
//
// Decomposition by Ptolemy's theorem:
//   cos(π/2 * (i-j)/M)
//     = cos(π i / 2M) cos(π j / 2M) + sin(π i / 2M) sin(π j / 2M)
//
// so the kernel factors as
//   s(Q_i, K_j) = (Q_i^cos)^T (K_j^cos) + (Q_i^sin)^T (K_j^sin)
//
// where  Q_i^cos = ReLU(Q_i) ⊙ cos(π i / 2M)        (per-row scaling vector)
//        Q_i^sin = ReLU(Q_i) ⊙ sin(π i / 2M)
//        K_j^cos = ReLU(K_j) ⊙ cos(π j / 2M)
//        K_j^sin = ReLU(K_j) ⊙ sin(π j / 2M)
//
// The attention output for token i is
//   O_i = ( Σ_j s(Q_i, K_j) V_j ) / ( Σ_j s(Q_i, K_j) )
//       = ( Q_i^cos (K_cos^T V) + Q_i^sin (K_sin^T V) )
//       / ( Q_i^cos ( Σ_j K_j^cos ) + Q_i^sin ( Σ_j K_j^sin ) )
//
// Forward is O(N · d²) — TWO (d × d) accumulators (KV_cos, KV_sin) and TWO
// (d,) sum-rows (Ksum_cos, Ksum_sin), reused across all N queries. Compare
// to standard softmax attention O(N² · d) — for N=1024, d=64, this is
// 16384× cheaper. Note the original paper states M ≥ N (a free parameter
// controlling how quickly the cosine falls off with |i-j|); we default to
// M=N for "natural" recency bias at the diagonal.
//
// The cosFormer paper applies this to both self- and cross-attention. The
// current implementation is non-causal (bidirectional), matching the
// convention used by Linformer/Performer in this repo. Causal masking is
// straightforward to add as a wrapper layer.
//
// ----------------------------------------------------------------------------
// What cosFormer captures vs. softmax:
//   * Non-negativity:        enforced by ReLU on Q, K (kernel feature maps
//                            stay ≥ 0 before the cos/sin scaling).
//   * Recency bias / locality: cos(π(i-j)/2M) peaks at i=j and falls off
//                               linearly with |i-j| (until 2M which is
//                               beyond the sequence length when M ≥ N).
//   * Identical pairwise sums (i ↔ j): cos is symmetric, so the attention
//                                      matrix is exactly symmetric.
//
// Numerical details:
//   * sin/cos evaluated with std::sin / std::cos on the integer-indexed
//     position vector c[i] = π · i / (2 M), so the matrices (cos_,
//     sin_) are baked at first forward and reused.
//   * eps=1e-6 floor on the denominator to prevent divide-by-zero if the
//     whole kernel collapses to zero (would only happen with all-zero Q or
//     K, which ReLU preserves as zero).
//   * For ReLU backward, the convention is: d/dx ReLU(x) = 1 if x>0 else 0.
//     Gradient is propagated only through positive Q/K positions.
//
// Parameter conventions (matches Linformer/Performer/GQA in this repo):
//   * (n, d_model) input/output — row-major
//   * Single-head: for multi-head, call multiple CosFormerAttentions and
//     concatenate. (Same convention as LinformerAttention.)
//   * Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual).
//   * No bias on Q/K/V/O projections (matches paper convention).
//
// cosFormerAttention — the attention layer itself
// cosFormerBlock      — pre-LN transformer block wrapper
// cosFormerModel      — stack of blocks + classifier
// ============================================================================

class CosFormerAttention : public Layer {
public:
    // d_model:  input/output feature dim
    // seq_len:  sequence length n (FIXED at construction — the cos/sin
    //           position vectors are length-n, baked at first forward).
    CosFormerAttention(size_t d_model, size_t seq_len, size_t M = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "CosFormerAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t M() const { return M_; }

    // Position vectors (length seq_len), computed on first forward()
    const Tensor& cos_pos() const { return cos_pos_; }
    const Tensor& sin_pos() const { return sin_pos_; }

    // Test helper: zero out position vectors to verify non-vacuousness
    void zero_pos_for_test() { cos_pos_.fill(0.0); sin_pos_.fill(0.0); }

    // Public for test access (matches GQA convention — params are public so
    // tests can directly check W_q, W_k, W_v, W_o projections).
    Tensor W_q, W_k, W_v, W_o;  // (d_model, d_model)  learned projections
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

private:
    void ensure_position_vectors();  // lazily fill cos_pos_ / sin_pos_ on first forward

    size_t d_model_;
    size_t seq_len_;
    size_t M_;                  // cosine period (default seq_len)

    Tensor cos_pos_;            // (seq_len,) — cos(π i / (2M))
    Tensor sin_pos_;            // (seq_len,) — sin(π i / (2M))

    // BPTT cache
    Tensor last_input_;         // (n, d_model)     cloned
    Tensor last_v_;             // (n, d_model)     V projection (needed in backward)
    Tensor last_q_relu_;        // (n, d_model)     ReLU(Q)
    Tensor last_k_relu_;        // (n, d_model)     ReLU(K)
    Tensor last_q_cos_;         // (n, d_model)     ReLU(Q) ⊙ cos_pos_
    Tensor last_q_sin_;         // (n, d_model)     ReLU(Q) ⊙ sin_pos_
    Tensor last_k_cos_;         // (n, d_model)     ReLU(K) ⊙ cos_pos_
    Tensor last_k_sin_;         // (n, d_model)     ReLU(K) ⊙ sin_pos_
    Tensor last_KV_cos_;        // (d_model, d_model) K_cos^T @ V
    Tensor last_KV_sin_;        // (d_model, d_model) K_sin^T @ V
    Tensor last_Ksum_cos_;      // (d_model,)       Σ_j K_cos[j]
    Tensor last_Ksum_sin_;      // (d_model,)       Σ_j K_sin[j]
    Tensor last_den_;           // (n,)             Q_cos @ Ksum_cos + Q_sin @ Ksum_sin
    Tensor last_out_;           // (n, d_model)     pre-W_o attention output
};

// ----------------------------------------------------------------------------
// cosFormerBlock: pre-LN → CosFormerAttn → residual → pre-LN → FFN → residual
// ----------------------------------------------------------------------------

class CosFormerBlock : public Layer {
public:
    CosFormerBlock(size_t d_model, size_t seq_len, size_t ffn_dim = 0, size_t M = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "CosFormerBlock"; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;             // (d_model)  pre-attn
    CosFormerAttention attn_;
    LayerNorm ln2_;             // (d_model)  pre-FFN
    Dense ffn_fc1_;             // (ffn_dim, d_model)
    Dense ffn_fc2_;             // (d_model, ffn_dim)

    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_pre_gelu_;   // (n, ffn_dim)  pre-GELU activation
    Tensor last_ffn_hidden_;     // (n, ffn_dim)  post-GELU activation (input to ffn_fc2_)
    Tensor last_ffn_out_;
};

// ----------------------------------------------------------------------------
// cosFormerModel: stack of CosFormerBlocks + classifier
// ----------------------------------------------------------------------------

class CosFormerModel : public Layer {
public:
    CosFormerModel(size_t d_model, size_t seq_len, size_t out_features,
                   size_t num_blocks = 1, size_t ffn_dim = 0, size_t M = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "CosFormerModel"; }

private:
    size_t d_model_;
    size_t out_features_;
    std::vector<CosFormerBlock> blocks_;
    Dense classifier_;

    Tensor last_input_;
};

#endif