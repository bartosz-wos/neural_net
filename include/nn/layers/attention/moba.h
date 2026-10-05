#ifndef MOBA_H
#define MOBA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// MoBA — Mixture of Block Attention
//   Lu, Jiang, Liu, Du, Jiang, Hong, Liu, He, Yuan, Wang, Huang, Yuan, Xu,
//   Lai, Chen, Hong, Liu, He, Yuan, Zhou, Zhang, Zhou, Yang, Zhou, Zhang,
//   Zhang, Yang, Sun, Zhang, Qiu — Moonshot AI / Tsinghua / Zhejiang Lab
//   "MoBA: Mixture of Block Attention for Long-Context LLMs"
//   https://arxiv.org/abs/2502.13189  (18 Feb 2025)
// ============================================================================
//
// Attention where each query token attends to a *subset* of the context, with
// the subset chosen per-query by a top-k gate over contiguous blocks of keys.
// This applies MoE routing (pick the k best of n experts) to the attention
// *mechanism* along the context-length dimension instead of the FFN dimension,
// so the model learns where to attend rather than having it hard-coded as a
// fixed sink or sliding window (paper §2.2).
//
// Paper notation → this implementation:
//   N        total tokens (input.rows)
//   B        block_size_
//   n = N/B  number of blocks
//   h        num_heads_
//   d_h      head_dim_ = d_model_ / num_heads_
//
// ----------------------------------------------------------------------------
// Eq. 2   MoBA(q,K,V) = Softmax(q K[I]^T) V[I],  I subset of [N]
// Eq. 3   block i covers I_i = [(i-1)B+1, iB]
// Eq. 5   g_i = 1 iff s_i in Topk({s_j | j in [n]}, k), else 0
// Eq. 6   s_i = <q, mean_pool(K[I_i])>
//
// THREE THINGS THAT ARE EASY TO GET WRONG (all pinned by tests):
//
//  1. The gate score is UNSCALED. Eq. 6 and Algorithm 1 line 5 (`S = Q K^T`)
//     contain NO 1/sqrt(d_h) factor. The `inv_temp_` scale belongs to the
//     *attention logits* only. Applying it to both makes every score 2x off
//     at head_dim=4. Test 3 asserts the exact hand-computed score.
//
//  2. mean_pool, not sum. Sum and mean only rescale the top-k ORDER (so
//     selection is unaffected), but they differ by a factor of block_size in
//     the reported score and in the K gradient.
//
//  3. `top_k_` counts HISTORICAL blocks only. The block containing the query
//     is always force-included (the paper's "current block is a shared expert,
//     always on"), so |I_t| <= top_k_ + 1. Blocks strictly after the query's
//     own block are future and are NEVER selected (causality).
//
//     Two readings exist and differ:
//       (a) top-k over all visible blocks 0..cur, then OR in current
//       (b) top-k over historical 0..cur-1, then add current   <-- implemented
//     Under (a) a query with top_k=1 that ranks its own block last attends to
//     1 block; under (b) it attends to 2. (b) matches the paper's reported
//     config (top_k=3, B=2048, N=32K => 3 historical + 1 current).
//
// ----------------------------------------------------------------------------
// Causality (paper §2.2):
//   * No attention to future blocks: block b is selectable by query t only if
//     b * B <= t, i.e. b <= t / B.
//   * The current block is always attended to, but with a CAUSAL MASK
//     (keys 0..t only). The paper is explicit that mean-pooling the current
//     block would leak information from subsequent tokens.
//
// ----------------------------------------------------------------------------
// Combining the two branches. The paper's Algorithm 1 lines 13-16 run the
// current block (causal) and the selected blocks (non-causal) as two attention
// calls and combine with ONLINE SOFTMAX. That is not an average of the two
// branch outputs — averaging would weight each branch equally regardless of
// how many keys it contributed. We instead take ONE softmax over the union
// I_t, which is mathematically identical to the tiled combine (a tiled
// softmax is a partitioned max-subtract softmax) and removes an entire class
// of online-softmax merge bugs. FlashAttention varlen / gather-scatter /
// tiling are out of scope for v1.
//
// ----------------------------------------------------------------------------
// Backward. The hard top-k is argmax-based, hence piecewise constant in the
// parameters: NO gradient flows through s_i into Q or K. This matches every
// other hard-routing layer in this repo (tokenformer.h, tome.h). So Q/K
// gradients come only from the attention-logit path — the same term set as
// standard MHA, but summed over SELECTED keys only. Unselected keys receive
// exactly zero K gradient (Test 8 asserts this exactly).
//
//   dA[i,t] = sum_d dO[t,d] * v_i,d
//   dV[i,d] += A[i,t] * dO[t,d]
//   dS[i,t] = A[i,t] * (dA[i,t] - sum_{i' in I_t} A[i',t] dA[i',t])
//   dz[i,t] = dS[i,t] * inv_temp_
//   dq_t,c += sum_{i in I_t} dz[i,t] k_i,c
//   dk_i,c += dz[i,t] q_t,c                  // ONLY i in I_t
//
// ----------------------------------------------------------------------------
// Conventions (match attention/stick_breaking.h):
//   * Input / output: (N, d_model). N tokens, no explicit batch dim.
//   * W_q, W_k, W_v, W_o are `Dense` (weights shaped (d_model, d_model),
//     y = x W^T + b); heads are contiguous column slices of the flat
//     (N, d_model) projections. Head h, channel c of token t is
//     X(t, h*head_dim + c).
//   * The repo's Tensor is 2-D ONLY. The head axis is flattened into ROWS:
//     last_scores_ and last_gate_ are (num_heads * N, n_blk) with row
//     h*N + t; last_A_ is (num_heads * N, N). A `h, t` vs `h*N+t` slip
//     silently corrupts multi-head routing while still passing at
//     num_heads == 1, hence the 2-head test config.
//   * Parameter gradients accumulate into raw grad_W_* / grad_b_* tensors
//     (NOT via Dense::backward) because the per-head restricted-attention
//     chain needs manual control.
//   * Block: pre-LN -> MoBA attn -> residual -> pre-LN -> GELU FFN -> residual
//   * Model: input Dense -> num_blocks x Block -> final LN -> classifier
// ============================================================================

class MoBAAttention : public Layer {
public:
    // d_model:   input/output feature dim; must be divisible by num_heads
    // num_heads: attention heads
    // block_size: B, the KV block size. Requires N % B == 0.
    // top_k:      number of HISTORICAL blocks selected per query. The
    //             current block is always added on top, so |I_t| <= top_k+1.
    // causal:     apply the causal mask inside the current block.
    MoBAAttention(size_t d_model, size_t num_heads = 1,
                  size_t block_size = 32, size_t top_k = 1,
                  bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "MoBAAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t block_size() const { return block_size_; }
    size_t top_k()     const { return top_k_; }
    bool   causal()    const { return causal_; }
    double inv_temp()  const { return inv_temp_; }

    // Raw block affinity scores from the last forward, (num_heads*N, n_blk),
    // row (h*N + t), col b == <q_t, mean_pool(K[block b])>. UNSCALED.
    const Tensor& last_scores() const { return last_scores_; }
    // Hard 0/1 gate from the last forward, (num_heads*N, n_blk).
    const Tensor& last_gate() const { return last_gate_; }
    // Attention weights, (num_heads*N, N), row (h*N + t). Zero on keys
    // outside I_t (i.e. outside the selected blocks, and masked out by
    // causality inside the current block).
    const Tensor& last_A() const { return last_A_; }
    const Tensor& last_O() const { return last_O_; }
    const Tensor& last_input() const { return last_input_; }
    // Per-token K gradient from the last backward, (N, d_model). Diagnostic:
    // unselected keys are exactly zero here. Not a parameter.
    const Tensor& last_dK() const { return last_dK_; }

    // Parameters (public so tests can perturb them directly for FD checks).
    // Bias gradients live in the Dense members' own grad_bias (repo convention,
    // as in stick_breaking) rather than in duplicated tensors here, so there is
    // exactly one gradient buffer per parameter.
    Dense W_q, W_k, W_v, W_o;   // each (d_model, d_model)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t block_size_;
    size_t top_k_;
    bool   causal_;
    double inv_temp_;

    // Forward caches
    Tensor last_input_;
    Tensor last_Q_;
    Tensor last_K_;
    Tensor last_V_;
    Tensor last_A_;       // (num_heads * N, N)
    Tensor last_O_;       // (N, d_model) — pre-W_o output
    Tensor last_scores_;  // (num_heads * N, n_blk)
    Tensor last_gate_;    // (num_heads * N, n_blk)
    size_t N_last_ = 0;
    size_t n_blk_last_ = 0;
    // Per-token K gradient from the last backward, (N, d_model). Diagnostic
    // only (tests assert unselected keys get exactly zero here); not a
    // parameter.
    Tensor last_dK_;
    // Selected block indices per (head, token): selected_[h][t] is the sorted
    // list of blocks in I_t. This is the "gather" of Algorithm 1 line 11,
    // stored explicitly so backward never re-derives the routing.
    std::vector<std::vector<std::vector<size_t>>> selected_;
};

// ----------------------------------------------------------------------------
// MoBABlock — pre-LN -> MoBA attn -> residual -> pre-LN -> GELU FFN -> residual
// ----------------------------------------------------------------------------
class MoBABlock : public Layer {
public:
    MoBABlock(size_t d_model, size_t num_heads = 1, size_t block_size = 32,
              size_t top_k = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.grad_W_q; }
    std::string name() const override { return "MoBABlock"; }

    MoBAAttention attn;
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
// MoBAModel — input proj -> blocks -> final LN -> classifier
// ----------------------------------------------------------------------------
class MoBAModel : public Layer {
public:
    MoBAModel(size_t input_dim, size_t d_model, size_t output_dim,
              size_t num_blocks, size_t block_size = 32, size_t top_k = 1,
              size_t num_heads = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return classifier.weights; }
    Tensor get_gradients() const override { return classifier.grad_weights; }
    std::string name() const override { return "MoBAModel"; }

    Dense input_proj;
    std::vector<std::unique_ptr<MoBABlock>> blocks;
    LayerNorm final_ln;
    Dense classifier;

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_blocks_;
};

#endif // MOBA_H