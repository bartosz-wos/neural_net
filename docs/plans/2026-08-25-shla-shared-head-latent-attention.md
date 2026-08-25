# SHLA (Shared-Head Latent Attention) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement a DeepSeek-MLA-flavored attention variant in which the KV up-projection matrices are SHARED across all heads (not per-head as in vanilla MLA). The compressed latent c_KV remains shared as before, but now a single `W_uk : d_c → head_dim` and `W_uv : d_c → head_dim` (instead of stacked-per-head) are applied identically to every head. Q keeps per-head `W_uq[h]`. This is the natural next step after MLA — even fewer parameters than MLA at iso-rank while keeping the O(n·d_c) KV cache advantage.

**Architecture:** Single shared Q-down `W_dq : d_model → d_c`, single shared KV-down `W_dkv : d_model → d_c`. Per-head Q-up `W_uq[h] : d_c → head_dim`. Single shared K-up `W_uk_shared : d_c → head_dim` and shared V-up `W_uv_shared : d_c → head_dim` (both applied identically across heads). Standard softmax attention per head. Output projection `W_o : d_model → d_model`. Backward accumulates d_c_KV from BOTH the K and V branches (shared-latent coupling) and d_c_KV_shared from the shared K/V up-projections; d_c_Q from Q branch only.

**Tech Stack:** Hand-rolled C++ (matches rest of repo). Pure CPU tensor ops. LayerNorm for the block's pre-LN. TDD throughout.

---

## Reference

- **Parent layer**: `include/nn/layers/attention/mla.{h,cpp}` — MLA stores the latent c_KV as `(n, d_c)` and decompresses K/V via per-head `W_uk[h]`, `W_uv[h] : d_c → head_dim`.
- **The variant**: SHLA replaces the per-head K/V up-projections with a SINGLE shared projection. So `K_h = c_KV @ W_uk_shared` and `V_h = c_KV @ W_uv_shared` for ALL heads h. Q remains per-head.
- **Param count (vs MLA vs MHA)**, with d = d_model, c = d_c, h = head_dim, H = num_heads:
  - SHLA:  `d·c + H·d·c + d·c + d·c + d·c + d² = (4 + H)·d·c + d²`
  - MLA:    `d·c + H·d·c + d·c + H·d·c + H·d·c + d² = (1 + 3H)·d·c + d²`
  - MHA:    `4·d²`
  - SHLA beats MLA when `(4 + H) < (1 + 3H)` → `3 < 2H` → `H ≥ 2`. Always for H ≥ 2.

## Files

- **Create**: `include/nn/layers/attention/shla.h`
- **Create**: `include/nn/layers/attention/shla.cpp`
- **Create**: `tests/test_shla.cpp`
- **Modify**: `include/nn/nn.h` (register header after `nsa.h`)
- **Modify**: `Makefile` (build rule + deps entry + run_tests echo)

## Task 1: Write failing header + first test (constructor validation)

Create `include/nn/layers/attention/shla.h` with class declarations. Initial sketch (skeleton — implementations filled in subsequent tasks):

```cpp
#ifndef SHLA_H
#define SHLA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// SHLA (Shared-Head Latent Attention) — variant of DeepSeek-MLA with SHARED
// K and V up-projection matrices across heads.
//
// Math (multi-head; per head h, head_dim = d_model / num_heads):
//
//   c_Q  = X @ W_dq                (n, d_c)       # shared Q down
//   Q_h  = c_Q @ W_uq[h]           (n, head_dim)  # PER-HEAD Q up
//   c_KV = X @ W_dkv               (n, d_c)       # shared KV down (one latent)
//   K_h  = c_KV @ W_uk_shared      (n, head_dim)  # SHARED K up across all heads
//   V_h  = c_KV @ W_uv_shared      (n, head_dim)  # SHARED V up across all heads
//   attn_h   = softmax(Q_h @ K_h^T / sqrt(head_dim))
//   head_out = attn_h @ V_h
//   out      = concat_h head_out @ W_o
//
// Compared to vanilla MLA, only the K and V up-projections are SHARED across
// heads (one pair of matrices applied to all heads identically), so Q stays
// per-head. The SHLA forward computes K and V just once (not per-head) and
// reuses them across heads — a measurable speed and memory win.
// ============================================================================

class SHLAAttention : public Layer {
public:
    SHLAAttention(size_t d_model, size_t num_heads, size_t d_c);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_dq; }
    Tensor get_gradients() const override { return grad_W_dq; }
    std::string name() const override { return "SHLAAttention"; }

    // Accessors for tests
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t d_c()       const { return d_c_; }
    Tensor grad_input() const { return last_d_input_; }

    // Public params/grads (test access)
    Tensor W_dq, W_dkv;
    Tensor W_uq;          // per-head, stacked (d_model, d_c)
    Tensor W_uk_shared, W_uv_shared;   // single (head_dim, d_c) shared
    Tensor W_o;           // (d_model, d_model)
    Tensor grad_W_dq, grad_W_dkv;
    Tensor grad_W_uq;
    Tensor grad_W_uk_shared, grad_W_uv_shared;
    Tensor grad_W_o;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t d_c_;
    double scale_;

    // BPTT cache
    Tensor last_input_;
    Tensor last_c_q_;          // (n, d_c)
    Tensor last_c_kv_;         // (n, d_c)
    Tensor last_q_;            // (n, d_model)
    Tensor last_k_;            // (n, head_dim) — SAME for all heads (shared K)
    Tensor last_v_;            // (n, head_dim) — SAME for all heads (shared V)
    Tensor last_attn_;         // (num_heads * n, n)
    Tensor last_head_out_;
    Tensor last_d_input_;
};

class SHLABlock : public Layer {
public:
    SHLABlock(size_t d_model, size_t num_heads, size_t d_c, size_t ffn_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "SHLABlock"; }
    Tensor grad_input() const { return last_d_input_; }
private:
    size_t d_model_, ffn_dim_;
    LayerNorm ln1_;
    SHLAAttention attn_;
    LayerNorm ln2_;
    Dense ffn_fc1_;
    Dense ffn_fc2_;
    Tensor last_input_, last_z1_, last_attn_out_, last_res1_;
    Tensor last_z2_, last_ffn_hidden_, last_ffn_out_, last_d_input_;
};

class SHLAModel : public Layer {
public:
    SHLAModel(size_t input_dim, size_t d_model, size_t output_dim, size_t num_blocks,
              size_t num_heads, size_t d_c, size_t ffn_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "SHLAModel"; }
private:
    size_t input_dim_, d_model_, output_dim_;
    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;
    std::vector<SHLABlock> blocks_;
    Tensor last_input_;
};

#endif
```

**Step 2:** Create `tests/test_shla.cpp` with **constructor validation tests only**:

- `d_model = 0` → throws
- `num_heads = 0` → throws
- `d_c = 0` → throws
- `d_model % num_heads != 0` → throws
- valid construction does not throw

**Step 3:** Register in Makefile + `nn.h` (header only at first).

**Step 4:** Run `make test_shla`, expect build error (link errors — function not defined).

## Task 2: Implement SHLAAttention ctor + zero_grad + parameters + update_weights + empty forward

Just enough for construction to succeed and a placeholder forward returning a zero tensor.

## Task 3: Implement SHLAAttention::forward (full math)

```cpp
Tensor SHLAAttention::forward(const Tensor& input) {
    // c_Q = X @ W_dq : (n, d_c)
    // c_KV = X @ W_dkv: (n, d_c)
    // Q = c_Q @ W_uq : (n, d_model)    (per-head stack)
    // K = c_KV @ W_uk_shared : (n, head_dim)   -- SHARED (computed once)
    // V = c_KV @ W_uv_shared : (n, head_dim)   -- SHARED (computed once)
    // For each head h: scores[t][s] = (Q[t, h*hd:(h+1)*hd] . K[s, :]) * scale
    //                  A[t][s] = softmax_row(scores[t])
    //                  head_out[t, h*hd:(h+1)*hd] = A @ V
    // out = head_out @ W_o : (n, d_model)
}
```

Tests: forward shape, forward finite.

## Task 4: Implement SHLAAttention::backward (full math)

Critical chains:
- `dW_o` + `d_head_out_cat`
- Per-head attention backward (dV/h, dA, softmax backward → dS, dQ, dK) where **K/V are SHARED so their gradients are accumulated across heads**.
- Per-head `grad_W_uq[h_off + j, c] += sum_i c_Q[i, c] * dQ[i][j]`
- **SHARED** `grad_W_uk_shared[j, c] += sum_h sum_i c_KV[i, c] * dK_h[i][j]` (sum over heads since K is shared)
- **SHARED** `grad_W_uv_shared[j, c] += sum_h sum_i c_KV[i, c] * dV_h[i][j]` (sum over heads since V is shared)
- `d_c_Q[i, c] += sum_j dQ[i][j] * W_uq[h_off + j, c]` (per-head up)
- `d_c_KV[i, c] += sum_h_j dK_h[i, j] * W_uk_shared[j, c] + sum_h_j dV_h[i, j] * W_uv_shared[j, c]`
- `dW_dq[k, c] += sum_i X[i, k] * d_c_Q[i, c]`
- `dW_dkv[k, c] += sum_i X[i, k] * d_c_KV[i, c]`
- `dX[i, k] = sum_c d_c_Q[i, c] * W_dq[k, c] + sum_c d_c_KV[i, c] * W_dkv[k, c]`

Tests: input gradient FD (rel_err < 1e-5), each param gradient FD.

## Task 5: Param count test (param-count formula sanity)

Verify SHLA param count exactly matches the formula `(4 + H) * d * c + d * d`.

## Task 6: SHLABlock — forward, backward, input-gradient FD, training reduces loss

## Task 7: SHLAModel — forward, backward, training reduces loss

## Task 8: K/V shared-across-heads verification

A separate test that constructs two `SHLAAttention` instances with the same params and confirms that the K (resp. V) tensor from one and from the other are EXACTLY equal at forward time (because K and V are computed from the same c_KV and the SAME shared W_uk_shared/W_uv_shared).

## Task 9: Mutation test (shared-K coupling)

Stub out the `+ sum_h_j dV_h[i, j] * W_uv_shared[j, c]` line in `d_c_KV` accumulation. The d_c_KV gradient should change meaningfully.

## Task 10: Register in nn.h umbrella + Makefile run_tests echo

## Task 11: Update EXPANSION_QUEUE.md (move SHLA to Done)
