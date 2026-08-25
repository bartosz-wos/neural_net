# Block-Sparse Flash Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement a flash-attention-style block-sparse attention layer that integrates a 2D block mask pattern `(n_q_blocks, n_k_blocks)` directly into the FA-2 online-softmax recurrence. Masked-out blocks contribute neither to the forward output nor to the backward gradients, and the layer generalizes dense / causal / sliding-window / strided / BigBird-style patterns from a single mask tensor.

**Architecture:** Reuse the FlashAttentionV2 online-softmax tile structure (Q-block-outer work partition) but add a per-block-mask check at the top of each (q-block, k-block) tile pair. If `mask[i_block, j_block] == 0`, skip that K/V tile entirely. Multi-head with GQA-style K/V sharing optional. For v1 we keep dense QKV (per-head K/V, like SHLA/FA-2) since the focus is on the mask routing, not MLA-style latent compression. LayerNorm-pre block + residual.

**Tech Stack:** Hand-rolled C++ (matches rest of repo). Pure CPU tensor ops, online-softmax flash recurrence from `flash_attention_v2.cpp`. TDD throughout.

---

## Reference

- **Block-sparse pattern in FlashAttention**: The original FlashAttention paper §4 ("Block-Sparse FlashAttention") describes how the same kernel can be made block-sparse by passing a 2D block mask that gates which (Q-block, K-block) pairs are processed. Each entry of the mask is a single bit (or byte) per tile pair, so the I/O and compute savings scale with the sparsity of the mask.
- **Existing FA-2 in this repo**: `include/nn/layers/attention/flash_attention_v2.{h,cpp}`. It implements the online-softmax recurrence with configurable `query_block_size` and `key_block_size`. We replicate the recurrence and add a mask check.
- **Generalizations the mask enables**:
  - Dense:        mask[i,j] = 1 ∀ i,j
  - Causal:       mask[i,j] = 1 if j ≤ i (block-causal)
  - Sliding win:  mask[i,j] = 1 if (j ∈ [i - W, i]) (Longformer/Mistral convention)
  - Strided:      mask[i,j] = 1 if (i - j) ≡ 0 (mod s) (Longformer random/strided)
  - BigBird:      union of window + random + global blocks

## Files

- **Create**: `include/nn/layers/attention/block_sparse_flash.h`
- **Create**: `include/nn/layers/attention/block_sparse_flash.cpp`
- **Create**: `tests/test_block_sparse_flash.cpp`
- **Modify**: `include/nn/nn.h` (register header after `flash_attention_v2.h`)
- **Modify**: `Makefile` (build rule + deps entry + `run_tests` echo)

## Conventions

- Input/Output: `(n, d_model)` (token-major, matches SHLA / NSA / sliding_window convention)
- `d_model` must be evenly divisible by `num_heads` → `head_dim = d_model / num_heads`
- Multi-head with optional GQA-style K/V sharing (`num_kv_heads`; if 0 → defaults to `num_heads`)
- Causal is REPLACED by an explicit user-supplied mask. We provide a helper that BUILDS the standard block-causal / block-sliding-window masks for convenience.
- The mask is a `Tensor` of shape `(n_q_blocks, n_k_blocks)` with values in {0, 1} interpreted as bool. 1 = attend, 0 = skip.
- `query_block_size` and `key_block_size` are constructor-time knobs (default 64 each). The mask dimensions must match the resulting `(n_q_blocks, n_k_blocks)` exactly. Validation throws otherwise.

## Math

```
Forward per head h, given 2D block mask M ∈ {0,1}^{n_q_blocks × n_k_blocks}:
  For each query block i in [0, n_q_blocks):
    Initialize (o_i, l_i, m_i) ← (0, 0, -∞)
    For each key block j in [0, n_k_blocks):
      if M[i, j] == 0: continue   # skip tile entirely
      K_j, V_j ← K[h, j*B_kv : (j+1)*B_kv, :], V[h, j*B_kv : (j+1)*B_kv, :]
      Q_i      ← Q[h, i*B_q  : (i+1)*B_q,  :]
      S        ← Q_i @ K_j^T / sqrt(d_k)   # (B_q, B_kv) per row
      apply causal / row-bounds to S (set positions beyond seq_len to -inf)
      m_new_i  ← max(m_i, rowmax(S))
      if m_i ≠ -∞: o_i *= exp(m_i - m_new_i);  l_i *= exp(m_i - m_new_i)
      P        ← exp(S - m_new_i)
      o_i      ← o_i + P @ V_j
      l_i      ← l_i + rowsum(P)
      m_i      ← m_new_i
    o_i     ← o_i / l_i
    context_h[i*B_q : (i+1)*B_q, :] ← o_i

Backward mirrors forward: for each (i, j) block pair, recompute S_ij and P_ij only when M[i,j] == 1.
```

## Param count

Identical to FlashAttentionV2: `4 * d_model^2` (W_q, W_k, W_v, W_o). Mask is NOT a parameter.

---

## Task 1: Header + first test (constructor validation)

Create `include/nn/layers/attention/block_sparse_flash.h`:

```cpp
#ifndef BLOCK_SPARSE_FLASH_H
#define BLOCK_SPARSE_FLASH_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cstdint>

// ============================================================================
// Block-Sparse Flash Attention
//
// Reference: Dao et al. 2022 "FlashAttention: Fast and Memory-Efficient Exact
// Attention with IO-Awareness" §4 (Block-Sparse FlashAttention).
//   https://arxiv.org/abs/2205.14135
//
// Same online-softmax flash recurrence as FlashAttentionV2, but with a 2D
// block mask M ∈ {0,1}^{n_q_blocks × n_k_blocks} that gates which (Q-block,
// K-block) pairs participate. Masked-out blocks contribute nothing to forward
// output AND no gradient flows through them in backward.
//
// The mask generalizes:
//   - dense (all 1s)
//   - causal (lower-triangular blocks)
//   - sliding window (lower-triangular + diagonal band)
//   - strided (Longformer-style)
//   - BigBird (window + random + global)
// ============================================================================

class BlockSparseFlashAttention : public Layer {
public:
    BlockSparseFlashAttention(size_t d_model,
                              size_t num_heads,
                              size_t num_kv_heads = 0,    // 0 → num_heads (MHA)
                              size_t query_block_size = 4,
                              size_t key_block_size = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "BlockSparseFlashAttention"; }

    // Configuration accessors
    size_t d_model()           const { return d_model_; }
    size_t num_heads()         const { return num_heads_; }
    size_t num_kv_heads()      const { return num_kv_heads_; }
    size_t head_dim()          const { return head_dim_; }
    size_t query_block_size()  const { return query_block_size_; }
    size_t key_block_size()    const { return key_block_size_; }

    // Test accessors
    size_t n_q_blocks()        const { return n_q_blocks_; }
    size_t n_k_blocks()        const { return n_k_blocks_; }
    const Tensor& last_mask()  const { return last_mask_; }
    const Tensor& last_input() const { return last_input_; }

    // Public params (matches FA-2 convention)
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // Helpers for building standard masks (return shape (n_q_blocks, n_k_blocks))
    static Tensor build_dense_mask(size_t n_q_blocks, size_t n_k_blocks);
    static Tensor build_causal_mask(size_t n_q_blocks, size_t n_k_blocks);
    static Tensor build_sliding_window_mask(size_t n_q_blocks, size_t n_k_blocks,
                                            size_t window_n_blocks);
    static Tensor build_strided_mask(size_t n_q_blocks, size_t n_k_blocks,
                                     size_t stride);
    static Tensor build_bigbird_mask(size_t n_q_blocks, size_t n_k_blocks,
                                     size_t window_n_blocks, size_t n_global_blocks,
                                     uint32_t seed);

    // Forward with an explicit mask. Required because the base Layer interface
    // doesn't carry a mask parameter. The mask is (n_q_blocks, n_k_blocks) in
    // {0,1}; mask[i,j] = 1 means Q-block i attends to K-block j.
    Tensor forward_with_mask(const Tensor& input, const Tensor& mask);

private:
    size_t d_model_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t query_block_size_;
    size_t key_block_size_;
    double scale_;
    size_t n_q_blocks_;
    size_t n_k_blocks_;
    bool has_forward_cache_;

    Tensor last_input_;       // (n, d_model)
    Tensor last_query_;       // (n, d_model)
    Tensor last_key_;         // (n, d_model)
    Tensor last_value_;       // (n, d_model)
    Tensor last_context_;     // (n, d_model)  pre-projection attention output
    Tensor last_mask_;        // (n_q_blocks, n_k_blocks)
    size_t last_n_;           // sequence length at last forward (for backward shape checks)
};

#endif
```

`tests/test_block_sparse_flash.cpp` skeleton with Task 1 test (5-case constructor validation):

```cpp
// --- Test 1: Constructor validation ---
// Cases: d_model=0, num_heads=0, num_heads>num_kv_heads, key_block_size=0,
// query_block_size=0, d_model%num_heads!=0, all-valid
// 4 invalid should throw, 1 valid should not throw
```

**Step 2**: Run test, verify 5 cases pass (4 throw + 1 succeeds). Expected: FAIL — file doesn't exist yet.

**Step 3**: Implement minimal constructor (defer real forward to Task 2) that validates inputs and initializes W_q/W_k/W_v/W_o as `Tensor::random(d_model, d_model, 0.3)`.

**Step 4**: Verify pass.

**Step 5**: Commit.

## Task 2: Forward shape with dense mask

Extend test file with Test 2 (forward shape with dense mask) and Test 3 (forward shape with causal mask). Implement `forward_with_mask` and the `flash_attn_block_sparse_tile` helper that mirrors FA-2 but skips masked blocks.

Verify: with a `(n, d)` input, dense mask `(n_q_blocks, n_k_blocks)` all-ones, output shape is `(n, d)` and finite.

**Step 2**: Run, verify both tests pass.

**Step 5**: Commit.

## Task 3: Forward equivalence with FA-2 (dense case)

When the mask is all-ones, the output of BlockSparseFlashAttention must match FlashAttentionV2's output for the same input/weights (up to the same tolerance as within-class). Add a test that copies the same W_q/W_k/W_v/W_o into both layers, runs forward on the same input, and asserts `max_diff < 1e-5`.

This is the strongest correctness signal: a mask-routing bug in the recurrence (e.g. wrong scale, double-counting, missing row bounds) will diverge from FA-2 in the dense case.

**Step 5**: Commit.

## Task 4: Mask gating — causal mask blocks future positions

Build a block-causal mask `M[i,j] = 1 if j ≤ i, else 0`. Run forward on a 2-head 8-token case where a future token has a strongly distinguishing K-vector. Assert that the attention map (per head, per query block) is exactly 0 at positions j > i (we expose `last_per_head_attn_weights_` or similar accessor for verification). Equivalent: rebuild the per-row attention probabilities from the Q,K cache and verify they vanish for j > i.

**Step 5**: Commit.

## Task 5: Mask gating — sliding-window mask

Build a sliding-window mask with `window_n_blocks = 2` (attend to self and previous block). Verify: row i attends only to blocks j ∈ [i-1, i] (after rounding by key_block_size).

**Step 5**: Commit.

## Task 6: Mask gating — strided mask

Build a strided mask with `stride = 2` (only blocks j ≡ i mod 2). Verify the sparsity of the per-row attention map: row i has nonzero entries only at the chosen blocks.

**Step 5**: Commit.

## Task 7: Mask gating — BigBird-style (window + global + random)

Combine a sliding window of W=2 blocks, W global blocks at the start, and 2 random blocks per row (seeded for determinism). Verify:
- Row 0 has nonzero entries at the global block positions.
- Row i (i ≥ 2) has nonzero entries at blocks {i-1, i, global, random}.
- Total nonzero block count matches the constructed pattern.

**Step 5**: Commit.

## Task 8: Input gradient FD check (dense mask)

Standard `check_input_gradient` with all-ones mask. Assert `rel_err < 1e-4`. This exercises the full backward through the masked recurrence.

**Step 5**: Commit.

## Task 9: W_q, W_k, W_v, W_o gradient FD checks

Each parameter gradient against FD. `rel_err < 1e-4` for all four.

**Step 5**: Commit.

## Task 10: Backward routing — masked-out blocks have ZERO gradient contribution

This is the central test for block-sparsity correctness. Build a causal mask (lower-triangular). Perturb one of the unmasked rows of K by 1e-3; the corresponding W_k gradient must change. Perturb one of the masked-out rows of K (j > i for some i) by 1e-3; verify the W_k gradient contribution from that row is exactly 0 (compare against a dense-mask baseline).

Concrete test: build mask, forward, compute W_k gradient. Then build a second configuration where the SAME K perturbation is applied to a MASKED-OUT row only. Assert W_k gradient identical to the unperturbed case (within FP tolerance).

**Step 5**: Commit.

## Task 11: Mask validation in forward

Calling `forward_with_mask` with a mask whose shape doesn't match `(n_q_blocks, n_k_blocks)` must throw. Test 4 invalid shapes: wrong rows, wrong cols, non-{0,1} values, all-zeros (degenerate).

**Step 5**: Commit.

## Task 12: BlockSparseFlashBlock (residual + LayerNorm + FFN)

Mirror SHLABlock's structure: pre-LN → BlockSparseFlash → residual → optional pre-LN FFN → residual. Verify forward shape and input gradient FD.

**Step 5**: Commit.

## Task 13: BlockSparseFlashModel (stack)

Mirror SHLAModel: in_proj → N blocks → classifier. Verify training reduces loss.

**Step 5**: Commit.

## Task 14: Final integration

- Register in `include/nn/nn.h` umbrella header (after `flash_attention_v2.h`).
- Add to Makefile: `build/test_block_sparse_flash` rule, deps entry in `tests:` target, `=== Running Block-Sparse Flash Attention Tests ===` echo in `run_tests`.
- Run full suite (`make tests`) — no regressions.
- Update `EXPANSION_QUEUE.md`: move the Block-Sparse Flash Attention entry to `## Done` with a one-line summary.

**Step 5**: Commit.

## Pitfalls

- **Validation order**: the scale_ computation needs `head_dim_ > 0`, which requires both `d_model_ > 0` and `num_heads_ > 0`. Use the dummy-init-list trick (initialize with `1`/`0` and recompute after validation) — same pattern as SHLA/NSA/sliding_window.
- **Block sizing**: the mask shape `(n_q_blocks, n_k_blocks)` must match `(ceil(n / query_block_size), ceil(n / key_block_size))`. If n doesn't divide evenly, the LAST block is partial. Handle by bounds-checking every (i, j) pair against `last_n_`.
- **Online softmax row-bounds**: when key_block j is partial (last k-block), some rows of S have `S[i, c] = -1e9` for `c ≥ last_n_`. These positions contribute zero to P and to the V-aggregation. Handle explicitly so backward doesn't propagate non-zero gradient through them.
- **Mask == 0 for the only block a row can attend to**: if a Q-block has NO unmasked K-block, the row's output is `0/0 = NaN`. Detect and short-circuit by setting o_i = 0, l_i = 1, m_i = 0 (uniform-zero output; gradient through that row is zero by construction).
- **Multi-head K/V with GQA**: with `num_kv_heads < num_heads`, the K/V heads are shared across multiple Q heads. Apply the same mask to the K/V heads (one mask per K/V head = `num_kv_heads` masks total, or — more naturally — one mask shared by all K/V heads since the mask is head-independent in v1).
- **FD tolerance**: for the W_k gradient through masked-out K-blocks, the tolerance must be at the FP noise floor (`~1e-15`), NOT 1e-4 — we want bit-exactness with the unperturbed baseline.

## Verification

```bash
# Run focused suite
make build/test_block_sparse_flash && ./build/test_block_sparse_flash

# Full suite — no regressions
make tests
```