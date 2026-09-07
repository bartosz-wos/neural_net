#ifndef STICK_BREAKING_H
#define STICK_BREAKING_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// Stick-Breaking Attention
//   Tan, Yang, Courville, Panda, Shen — ICLR 2025
//   "Scaling Stick-Breaking Attention: An Efficient Implementation and
//    In-depth Study"
//   https://arxiv.org/abs/2410.17980
//   Reference impl: https://github.com/shawntan/stickbreaking-attention
// ============================================================================
//
// Softmax attention is permutation-invariant, so it needs positional
// embeddings (RoPE / ALiBi / learned biases) bolted on — and those generalise
// poorly past the trained context length. Stick-breaking attention instead
// derives the ordering from the *allocation process itself*, so it needs NO
// position embeddings at all.
//
// The idea (a residual allocation / GEM process): walk backwards from the
// current query position j. At each earlier token i we decide what fraction
// β[i,j] of the *remaining stick* to allocate to token i. Whatever is left
// after token i passes to the tokens further back. Formally (paper Eq. 1),
// per head with d_h = d_model / num_heads:
//
//   z[i,j] = (q_j · k_i) * inv_temp,        inv_temp = 1 / sqrt(d_h)
//   β[i,j] = σ(z[i,j])
//   A[i,j] = β[i,j] * Π_{i<k<j} (1 - β[k,j])
//   o_j    = Σ_{i<j} A[i,j] · v_i
//
// Properties that fall out of this and NOT out of softmax:
//   * Recency bias for free: for |j-i| < |j-i'| with equal logits,
//     A[i,j] >= A[i',j]. Linguistically motivated (grammar parsing attends
//     to the most adjacent non-terminal).
//   * Σ_{i<j} A[i,j] <= 1 — NOT normalized to 1. The mechanism can attend to
//     *nothing* when all β → 0, instead of softmax's forced redistribution
//     onto low-information tokens ("attention sinks").
//   * A high score far back does not "distract" from a recent high score:
//     A[i,j] depends only on z[k,j] for i <= k < j.
//
// We use STRICT causality (i < j): a query does not attend to itself. The
// reference impl exposes `attend_current` to include i == j; we implement the
// paper's default (false), so query j=0 attends to nothing at all.
//
// ----------------------------------------------------------------------------
// Numerically stable log-space form (paper Eq. 11-13) — what we implement.
//
// Since β = σ(z):  log β      = z - softplus(z)
//                  log(1 - β) = -softplus(z)
// therefore
//   A[i,j] = exp( z[i,j] - Σ_{k=i}^{j-1} softplus(z[k,j]) )
//
// with softplus(x) = log(1+exp(x)) evaluated stably as
//   max(x,0) + log1p(exp(-|x|)).
// Computing the product in log-space avoids the underflow that a direct
// Π (1-β) over long contexts would hit.
//
// ----------------------------------------------------------------------------
// Remainder bias (paper Appendix C, Eq. 14) — ON by default.
//
// Because the weights sum to <= 1, there is leftover stick. Rather than
// discard it (which makes ||o_j|| vary wildly with position), assign it to a
// learnable per-head embedding r — the paper's "remainder bias", morally an
// attention sink that is learned rather than fixed at zero:
//
//   rem_j = 1 - Σ_{i<j} A[i,j]                 (== 1 for j = 0)
//   o_j   = Σ_{i<j} A[i,j] · v_i + rem_j · r_head
//
// The paper reports consistent gains from this, so it is the default here.
// Pass use_remainder=false for the bare Eq. 1 form.
//
// ----------------------------------------------------------------------------
// Backward derivation (the crux — see docs/plans/2026-09-07-stick-breaking-attention.md)
//
// With S[i,j] = z[i,j] - Σ_{k=i}^{j-1} softplus(z[k,j])  and  A = exp(S):
//
//   dA[i,j] = dO_j · v_i - dO_j · r_head      (2nd term via rem_j = 1 - ΣA)
//   dV[i]  += A[i,j] * dO_j
//   dr     += rem_j * dO_j
//   dS[i,j] = dA[i,j] * A[i,j]
//
// z[m,j] enters S two ways: as the leading term when m == i, and inside the
// softplus sum for every i <= m. Hence
//
//   dz[m,j] = dS[m,j] - σ(z[m,j]) * Σ_{i=0}^{m} dS[i,j]
//
// That Σ_{i<=m} dS[i,j] is a PREFIX SUM over i, accumulated as m increases —
// giving an O(L²) backward rather than the naive O(L³).
//
// Then the standard QK chain:
//   dq_j += inv_temp * Σ_{m<j} dz[m,j] * k_m
//   dk_m += inv_temp * Σ_{j>m} dz[m,j] * q_j
//
// ----------------------------------------------------------------------------
// Conventions (match attention/diff_transformer.h):
//   * Input / output: (N, d_model). N tokens, no explicit batch dim.
//   * W_q, W_k, W_v, W_o are `Dense` (weights shaped (out, in)); heads are
//     contiguous column slices of the flat (N, d_model) projections.
//   * Parameter gradients for the projections are accumulated into raw
//     grad_W_* tensors (NOT via Dense::backward) because the per-head
//     stick-breaking chain needs manual control.
//   * Block: pre-LN -> SB attn -> residual -> pre-LN -> GELU FFN -> residual
//   * Model: input Dense -> num_blocks x Block -> final LN -> classifier
// ----------------------------------------------------------------------------

class StickBreakingAttention : public Layer {
public:
    // d_model:       input/output feature dim; must be divisible by num_heads
    // num_heads:     number of attention heads (default 1)
    // use_remainder: enable the learnable remainder-bias embedding (Eq. 14)
    StickBreakingAttention(size_t d_model, size_t num_heads = 1,
                           bool use_remainder = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "StickBreakingAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    bool   use_remainder() const { return use_remainder_; }
    double inv_temp()  const { return inv_temp_; }

    // Attention map from the last forward, shape (num_heads * N, N):
    // row (h*N + j), col i  ==  A[i,j] for head h. Zero for i >= j.
    const Tensor& last_A() const { return last_A_; }
    // Leftover stick per (head, query), shape (num_heads, N): 1 - Σ_i A[i,j].
    const Tensor& last_rem() const { return last_rem_; }
    const Tensor& last_input() const { return last_input_; }

    // Parameters (public so tests can perturb them directly for FD checks)
    Dense W_q, W_k, W_v, W_o;   // each (d_model, d_model)
    Tensor remainder_;          // (num_heads, head_dim) — learnable sink
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_remainder_;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    bool   use_remainder_;
    double inv_temp_;

    // Forward caches
    Tensor last_input_;   // (N, d_model)
    Tensor last_Q_;       // (N, d_model)
    Tensor last_K_;       // (N, d_model)
    Tensor last_V_;       // (N, d_model)
    Tensor last_Z_;       // (num_heads * N, N) — logits z[i,j] at row h*N+j
    Tensor last_A_;       // (num_heads * N, N) — attention weights
    Tensor last_rem_;     // (num_heads, N)
    Tensor last_O_;       // (N, d_model) — pre-W_o output
    size_t N_last_ = 0;
};

// ----------------------------------------------------------------------------
// StickBreakingBlock — pre-LN -> SB attn -> residual -> pre-LN -> FFN -> residual
// ----------------------------------------------------------------------------
class StickBreakingBlock : public Layer {
public:
    StickBreakingBlock(size_t d_model, size_t num_heads = 1,
                       size_t ffn_dim = 0, bool use_remainder = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.grad_W_q; }
    std::string name() const override { return "StickBreakingBlock"; }

    StickBreakingAttention attn;
    LayerNorm ln1, ln2;
    Dense ffn_fc1_;   // (ffn_dim, d_model)
    Dense ffn_fc2_;   // (d_model, ffn_dim)

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;
    Tensor last_z1_;        // ln1(x)
    Tensor last_attn_out_;
    Tensor last_res1_;      // z1 + attn_out
    Tensor last_z2_;        // ln2(res1)
    Tensor last_h_pre_;     // ffn_fc1(z2)
    Tensor last_h_act_;     // GELU(h_pre)
};

// ----------------------------------------------------------------------------
// StickBreakingModel — input proj -> blocks -> final LN -> classifier
// ----------------------------------------------------------------------------
class StickBreakingModel : public Layer {
public:
    StickBreakingModel(size_t input_dim, size_t d_model, size_t output_dim,
                       size_t num_blocks, size_t num_heads = 1,
                       bool use_remainder = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return classifier.weights; }
    Tensor get_gradients() const override { return classifier.grad_weights; }
    std::string name() const override { return "StickBreakingModel"; }

    Dense input_proj;
    std::vector<std::unique_ptr<StickBreakingBlock>> blocks;
    LayerNorm final_ln;
    Dense classifier;

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_blocks_;
};

#endif // STICK_BREAKING_H
