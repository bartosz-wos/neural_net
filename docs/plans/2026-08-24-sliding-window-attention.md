# Sliding Window Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add Mistral-style Sliding Window Attention (SWA) with optional global+local pattern, supporting multi-head Q with shared K/V via GQA-style grouping, fixed window size per layer.

**Architecture:** Softmax attention where each query at position `t` attends only to keys `j ∈ [t - W + 1, t]` (or to the full sequence for "global" tokens). The window is applied as an additive causal mask `M[i,j] = -∞` if `j < i - W + 1` else `0`. Multi-head via per-head Q/K/V projections; optional K/V sharing via the existing GQA convention (num_query_heads, num_kv_heads) so it composes cleanly with the recent GQA work.

**Tech Stack:** C++17, plain `Tensor` matmul (`@` is overloaded), softmax via existing helpers in `core/tensor.h` (we'll need to add a softmax helper if absent; or implement inline as in cosFormer/gqa).

---

## Design

Sliding window attention (SWA) was popularized by Mistral 7B (Jiang et al. 2023, https://arxiv.org/abs/2310.06825). The key idea: standard softmax attention has O(N²) cost, which becomes prohibitive for long sequences. By restricting each query to attend only to the last W tokens (window size), we get O(N · W · d) cost and a fixed-size KV-cache memory footprint during inference.

The math is identical to softmax attention — only the mask differs:

```
attn[i, j] = 0                              if j < i - W + 1 (out of window)
attn[i, j] = 1                              if i < num_global (global token)
attn[i, j] = softmax(Q[i] · K[j] / sqrt(d) + M[i,j])[j]  otherwise
out[i] = sum_j attn[i, j] · V[j]
```

The mask has two parts:
1. **Sliding window**: `M[i, j] = -∞` when `j < i - W + 1` (with `i, j ∈ [0, n)`).
2. **Optional global tokens**: the first `num_global` tokens attend to and are attended by the full sequence (Longformer-style "global attention"). When `num_global = 0`, SWA degenerates to plain causal sliding window.

For non-causal use (bidirectional), set `causal = false` and the window becomes symmetric: `M[i, j] = -∞` if `|i - j| > W/2`.

**Why this layer (and not, say, NSA):** SWA is the canonical long-context attention primitive used by Mistral 7B, Mixtral, and the entire Mistral-family ecosystem. It's small enough to implement cleanly (no projection-split tricks), well-isolated from existing layers, and the gradient-check math is tractable. It also pairs naturally with the recent GQA, RoPE, and agent-attention work — a complete long-context LLM stack.

## Files

- Create: `include/nn/layers/attention/sliding_window.h`
- Create: `include/nn/layers/attention/sliding_window.cpp`
- Create: `tests/test_sliding_window.cpp`
- Modify: `include/nn/nn.h` (add `#include "layers/attention/sliding_window.h"`)
- Modify: `Makefile` (add `build/test_sliding_window` rule, deps, echo line)

## Implementation sketch

```cpp
class SlidingWindowAttention : public Layer {
public:
    size_t d_model_, num_heads_, num_kv_heads_, head_dim_, group_size_, window_size_, num_global_;
    bool causal_;

    Tensor W_q, W_k, W_v, W_o;            // (d_model, d_model) or split
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    Tensor last_q_, last_k_, last_v_;     // post-projection, per-head
    Tensor last_scores_, last_attn_;      // softmax output (cached)
    Tensor last_input_, last_out_;

    SlidingWindowAttention(size_t d_model, size_t num_heads,
                           size_t num_kv_heads, size_t window_size,
                           size_t num_global = 0, bool causal = false);
    // ...standard Layer API...
};
```

**Forward:**
1. Compute Q, K, V projections (GQA-style head splitting).
2. Build per-head scores `S_h[i, j] = Q_h[i] · K_h[j] / sqrt(head_dim)`.
3. Apply window mask: `S[i, j] += (-inf) if j < i - W + 1` (causal) or `|i-j| > W/2` (non-causal).
4. Apply global mask: rows/cols for global tokens = 0 in mask (i.e., they ignore the window).
5. `attn = softmax(S, axis=-1); out = attn @ V`.
6. Concat heads, project through W_o.

**Backward:** standard attention backward, but masked positions have `dS = 0` (since softmax output at masked positions is 0). For stability we set `S[i,j] = -1e9` instead of `-inf`, which avoids NaN in softmax. The gradient w.r.t. positions outside the window is exactly 0 — this is what makes SWA a "true" sparse attention.

**Block + Model:**
- `SlidingWindowBlock`: pre-LN → SWA → residual → pre-LN → FFN → residual
- `SlidingWindowModel`: stack of blocks + classifier

## TDD tasks

### Task 1: Constructor + forward shape
- Constructor validates: d_model > 0, num_heads > 0, head_dim = d_model / num_heads, num_kv_heads | num_heads, window_size > 0, num_global >= 0.
- Forward `(n=6, d=8)` returns `(6, 8)`.

### Task 2: Forward finiteness + mask behavior
- Output is finite, nonzero.
- When `num_global=0`, comparing attention weights with `window=10` vs `window=2`: weights outside the larger window for the smaller case are ~0.
- Mask correctness: row `i` of `last_attn_` has nonzero only within `[i - W + 1, i]` (causal) or `[max(0, i-W/2), min(n, i+W/2)]` (non-causal).

### Task 3: Input gradient FD check
- Numerical gradient vs analytical rel_err < 1e-6.

### Task 4: W_q/W_k/W_v/W_o gradient FD checks
- All four projection gradients at rel_err < 1e-6.

### Task 5: Multi-head GQA mode
- num_heads=4, num_kv_heads=2 → forward shape correct, gradients still match FD.

### Task 6: Global tokens
- num_global=2 → first two rows/cols of attention map are nonzero everywhere (full attention from/to global tokens).

### Task 7: Non-causal mode
- `causal=false, window=4` → row i has nonzero attention in [max(0,i-2), min(n,i+2)] (symmetric window).

### Task 8: Block + Model
- `SlidingWindowBlock` forward shape, training reduces loss.
- `SlidingWindowModel` (2 blocks) forward shape, training reduces loss.

### Task 9: Mutation test
- Zero out the window mask → output changes (proves the mask path is exercised).

## Verification

- `make build/test_sliding_window && ./build/test_sliding_window` — all 9 tasks' checks pass.
- `make tests && make run_tests` — no regressions in adjacent suites.

