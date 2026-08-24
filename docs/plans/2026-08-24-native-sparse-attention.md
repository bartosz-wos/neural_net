# Native Sparse Attention (NSA) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement DeepSeek-AI 2025's NSA — a natively trainable sparse attention with three parallel branches (compression, top-n selection, sliding window) gated by a learned 3-way softmax.

**Architecture:** Three independent K/V projections (one per branch, per paper §3.3.3 "we provide independent keys and values for three branches") share a single Q projection. Compression uses blockwise MLP aggregation with intra-block positional encoding (paper Eq. 7). Selection uses blockwise importance scores induced from compression attention scores (Eq. 8-10), keeps top-n blocks (Eq. 11). Sliding window keeps last `w` tokens. A learned per-(token, head, branch) softmax gate combines the three branches. Multi-head with GQA-style K/V sharing (`num_query_heads`, `num_kv_heads`).

**Tech Stack:** Hand-rolled C++ (matches rest of repo). Pure CPU tensor ops (`Tensor::matmul`, `Softmax` row-wise). LayerNorm for the block's pre-LN. TDD throughout. Each parameter has finite-difference gradient check against the analytical backward.

---

## Reference

- **Paper**: Yuan, Gao, Dai et al. 2025, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" — https://arxiv.org/abs/2502.11089
- **Key equations** (Eq. 5, 7, 8, 11 from paper):
  - Output: `o*_t = Σ_{c∈{cmp, slc, win}} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)` (Eq. 5)
  - Compression: `K̃^c_t = {φ(k_{id+1:id+l}) | 0 ≤ i ≤ ⌊(t-l)/d⌋}` (Eq. 7). Same for V.
  - Importance scores: `p^c_t = Softmax(q_t^T · K̃^c_t)` (Eq. 8). Blockwise aggregation → top-n (Eq. 9-11).
- **Three branches have INDEPENDENT K, V projections** (§3.3.3 explicit) — preventing gradient interference between local and long-range branches. This is what we'll implement.

## Why a fresh design (vs. blind copy of paper)

The paper's compression MLP `φ` takes `(l, d_head)` → `d_head` with intra-block position encoding. For the C++ ML library (TDD on small T), we instantiate `φ` as a per-head learnable linear layer (head_dim × (l * head_dim)) with intra-block sinusoidal position encoding concatenated as an additional input. This keeps the math tractable for finite-difference grad checks at small N while preserving all four paper invariants: (i) per-block aggregation, (ii) learnable φ, (iii) intra-block position awareness, (iv) stride `d < l` overlap.

---

## Files

- **Create**: `include/nn/layers/attention/nsa.h`
- **Create**: `include/nn/layers/attention/nsa.cpp`
- **Create**: `tests/test_nsa.cpp`
- **Modify**: `include/nn/nn.h` (register header after `sliding_window.h`)
- **Modify**: `Makefile` (build rule + deps entry + run_tests echo)

---

## Task 1: Write the failing header + first test (constructor validation)

**Step 1:** Create `include/nn/layers/attention/nsa.h` with class declarations.

```cpp
#ifndef NSA_H
#define NSA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Native Sparse Attention (NSA) — DeepSeek-AI 2025
//   "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
//   https://arxiv.org/abs/2502.11089
//
// Three parallel branches:
//   1. Compression: keys/values are blockwise-aggregated via learnable MLP φ
//      with intra-block position encoding, producing n_cmp = ⌊(N-l)/stride⌋+1
//      compressed tokens per head.
//   2. Selection: top-n selection blocks, ranked by importance scores induced
//      from compression attention scores (paper Eq. 8-11).
//   3. Sliding window: last `w` tokens.
//
// Each branch has INDEPENDENT K, V projections (paper §3.3.3).
//
// A learned per-(token, head, branch) softmax gate combines the three branches:
//   out_t = Σ_c g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)         (Eq. 5)
//
// Multi-head with K/V sharing via GQA convention (num_query_heads,
// num_kv_heads), mirroring the GQA layer in this repo.
//
// Conventions:
//   * Input: (n, d_model) — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model divisible by num_query_heads (= head_dim)
//   * num_query_heads divisible by num_kv_heads
//   * l > 0, stride in [1, l], window w > 0, top_n > 0
//   * n must be >= l so compression produces at least one block
//
// NSAAttention    — the attention layer itself
// NSABlock        — pre-LN → NSA → residual → optional pre-LN FFN → residual
// NSAModel        — in_proj → N blocks → classifier
// ============================================================================

class NSAAttention : public Layer {
public:
    NSAAttention(size_t d_model,
                 size_t num_query_heads,
                 size_t num_kv_heads,
                 size_t block_len,       // l: compression block length
                 size_t stride,          // d: sliding stride between compression blocks
                 size_t top_n,           // n: top-n selection blocks
                 size_t window_size,     // w: sliding window size
                 size_t block_size = 0); // l': selection block size; defaults to stride

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "NSAAttention"; }

    // Accessors
    size_t d_model()        const { return d_model_; }
    size_t num_heads()      const { return num_query_heads_; }
    size_t num_kv_heads()   const { return num_kv_heads_; }
    size_t head_dim()       const { return head_dim_; }
    size_t block_len()      const { return block_len_; }
    size_t stride()         const { return stride_; }
    size_t top_n()          const { return top_n_; }
    size_t window_size()    const { return window_size_; }
    size_t block_size()     const { return block_size_; }
    size_t n_cmp()          const { return n_cmp_; }   // # of compressed blocks
    size_t n_sel_blocks()   const { return n_sel_blocks_; } // # of selection blocks

    // Test helper: returns the per-head gate from the last forward (n, 3, d_model),
    // softmax-normalized across branches (last 3 d_model-sized rows? actually
    // we store as (n, num_heads, 3) — see impl).
    const Tensor& last_gate() const { return last_gate_; }

    // Test helper: returns the per-branch attention maps from the last forward
    // (one per branch per head; layout depends on branch — see impl).
    const Tensor& last_attn_cmp() const { return last_attn_cmp_; }   // (H, N, n_cmp)
    const Tensor& last_attn_sel() const { return last_attn_sel_; }   // (H, N, n_sel*n_blk)
    const Tensor& last_attn_win() const { return last_attn_win_; }   // (H, N, w)

    // Test helpers (cached input grads from last backward)
    Tensor grad_input() const { return last_d_input_; }

    // Public for test access (matches GQA / cosformer convention)
    Dense W_q, W_k_cmp, W_v_cmp, W_k_sel, W_v_sel, W_k_win, W_v_win, W_o;
    Tensor grad_W_q, grad_W_k_cmp, grad_W_v_cmp,
           grad_W_k_sel, grad_W_v_sel, grad_W_k_win, grad_W_v_win,
           grad_W_o;

    // Per-head compress MLP φ: (head_dim, l * head_dim) — paper Eq. 7
    // Stored as (num_heads, head_dim, l * head_dim)
    Tensor W_phi_k, W_phi_v;
    Tensor grad_W_phi_k, grad_W_phi_v;

    // Intra-block position encoding: (l, head_dim) — sinusoidal, fixed
    Tensor pos_enc_;

    // Per-head, per-branch gating MLP: 3-way softmax over a small linear layer
    // Stored as (num_heads * 3, head_dim) — projects head_dim → 3 logits per head
    Dense W_gate;
    Tensor grad_W_gate;

    // Mutation hooks (for tests)
    void set_use_compression(bool b) { use_compression_ = b; }
    void set_use_selection(bool b)   { use_selection_ = b; }
    void set_use_window(bool b)      { use_window_ = b; }
    bool use_compression() const { return use_compression_; }
    bool use_selection() const   { return use_selection_; }
    bool use_window() const      { return use_window_; }

private:
    size_t d_model_;
    size_t num_query_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t group_size_;
    size_t block_len_;      // l
    size_t stride_;         // d
    size_t top_n_;          // n
    size_t window_size_;    // w
    size_t block_size_;     // l'
    size_t n_cmp_;          // (N - l) / d + 1, computed at forward
    size_t n_sel_blocks_;   // (N - l') / d + 1, computed at forward
    double scale_;          // 1 / sqrt(head_dim)

    bool use_compression_ = true;
    bool use_selection_   = true;
    bool use_window_      = true;

    // BPTT cache (filled in forward, consumed in backward)
    Tensor last_input_;
    Tensor last_Q_, last_K_cmp_, last_V_cmp_,
           last_K_sel_, last_V_sel_,
           last_K_win_, last_V_win_;
    Tensor last_K_tilde_cmp_, last_V_tilde_cmp_;   // (H, n_cmp, head_dim)
    Tensor last_K_tilde_sel_, last_V_tilde_sel_;   // (H, n_sel*n_blk, head_dim)
    Tensor last_K_tilde_win_, last_V_tilde_win_;   // (H, w, head_dim)
    Tensor last_p_cmp_;                             // (H, N, n_cmp) — importance scores
    Tensor last_top_idx_;                           // (H, N, top_n) — selected block indices
    Tensor last_attn_cmp_, last_attn_sel_, last_attn_win_;
    Tensor last_gate_;                              // (N, num_heads, 3) softmax
    Tensor last_O_branches_;                        // (N, 3, d_model) per-branch outputs pre-gate
    Tensor last_O_;                                 // (N, d_model) post-gate pre-W_o
    Tensor last_d_input_;
    size_t N_last_ = 0;
};

// ----------------------------------------------------------------------------
// NSABlock: pre-LN → NSA → residual → optional pre-LN FFN → residual
// ----------------------------------------------------------------------------
class NSABlock : public Layer {
public:
    NSABlock(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
             size_t block_len, size_t stride, size_t top_n,
             size_t window_size, size_t block_size = 0, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return nsa_.W_q.weights; }
    Tensor get_gradients() const override { return nsa_.grad_W_q; }
    std::string name() const override { return "NSABlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_, ffn_dim_;
    NSAAttention nsa_;
    LayerNorm ln1_, ln2_;
    Tensor W1_, b1_, W2_, b2_;
    Tensor grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    Tensor last_input_, last_ln1_, last_attn_out_, last_res1_;
    Tensor last_ln2_, last_ffn_pregelu_, last_ffn_out_;
    Tensor last_d_input_;
};

// ----------------------------------------------------------------------------
// NSAModel: in_proj → N blocks → classifier
// ----------------------------------------------------------------------------
class NSAModel : public Layer {
public:
    NSAModel(size_t input_dim, size_t d_model, size_t output_dim,
             size_t num_blocks, size_t num_query_heads, size_t num_kv_heads,
             size_t block_len, size_t stride, size_t top_n,
             size_t window_size, size_t block_size = 0, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "NSAModel"; }

private:
    size_t input_dim_, d_model_, output_dim_;
    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;
    std::vector<NSABlock> blocks_;
    Tensor last_input_, last_proj_, last_block_out_;
};

#endif
```

**Step 2:** Create `tests/test_nsa.cpp` skeleton with constructor-validation test only.

**Step 3:** Register in Makefile + `nn.h`.

**Step 4:** Run `make test_nsa`, watch it FAIL with "function not defined" / link errors.

## Task 2: Implement `NSAAttention::NSAAttention` constructor + zero-grad + parameters + update_weights

Implement only enough to make the constructor-validation test pass.

## Task 3: Implement `NSAAttention::forward` (no backward yet)

## Task 4: Forward-shape and forward-finite tests

## Task 5: Implement `NSAAttention::backward` — Q/K/V_cmp paths

## Task 6: Backward — K/V_sel, K/V_win, φ, gate, W_o paths

## Task 7: Input-gradient FD check

## Task 8: Parameter-gradient FD checks (W_q, W_k_cmp, W_v_cmp, W_phi_k, W_gate, W_o)

## Task 9: NSABlock — forward, backward, training reduces loss

## Task 10: NSAModel — forward, backward, training reduces loss

## Task 11: Mutation tests (each branch independently)

## Task 12: Register in `nn.h` umbrella and Makefile run_tests target

## Task 13: Update EXPANSION_QUEUE.md (move NSA to Done)
