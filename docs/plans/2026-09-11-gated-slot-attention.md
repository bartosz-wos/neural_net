# Gated Slot Attention (GSA) — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement a gated variant of Slot Attention (Bauer et al. 2024, https://arxiv.org/abs/2410.23790) that adds a learned per-input gate `g = σ(W_g · x + b_g)` to suppress spurious cross-slot attention on out-of-distribution inputs.

**Architecture:** Drop-in extension of the existing `SlotAttention` (Locatello 2020) in `include/nn/layers/attention/slot_attention.{h,cpp}`. Same iterated double-softmax competition between slots and inputs. The only addition is one extra `Dense(input_dim, input_dim)` projection + sigmoid producing `g ∈ (0,1)^{N,input_dim}`, which is multiplied element-wise with `LN_k(x)` and `LN_v(x)` **before** the W_k / W_v projections (the paper's "input gate" convention, §3.2 of the paper — distinct from the slot gate `f` in §3.1 which gates the slots). Backward is the standard Slot-Attention chain plus an extra path through W_g via the sigmoid derivative.

**Tech Stack:** C++17, existing `Dense`, `LayerNorm`, `Tensor` primitives.

**Files:**
- Create: `include/nn/layers/attention/gated_slot_attention.{h,cpp}`
- Create: `tests/test_gated_slot_attention.cpp`
- Modify: `include/nn/nn.h` (umbrella `#include`)
- Modify: `Makefile` (build/test rules + run_tests echo)

---

## Task 1: Write the focused tests for `GatedSlotAttention` (RED)

**Objective:** Lock in the math and the interface before writing any implementation.

**Files:**
- Create: `tests/test_gated_slot_attention.cpp`

Write a test file with the following tests. Mirror the structure of `tests/test_slot_attention.cpp`:

1. **Constructor validation** (5 cases):
   - `num_slots=0` → throws `std::invalid_argument`
   - `slot_dim=0` → throws
   - `input_dim=0` → throws
   - `num_iterations=0` → throws
   - Valid `(K=4, D=4, I=6, T=3)` constructs without throw

2. **Forward shape & finiteness**: input `(N=5, input_dim=6)` → output `(K=4, slot_dim=4)`, all entries finite, not all zero.

3. **Gate in (0, 1)** invariant: after forward, `last_gate_` (cached) has every entry in (0, 1) and is row-shape `(N, input_dim)`.

4. **Gate zeroed → vanilla Slot Attention bit-exact**: with `W_g` zeroed and `b_g = -100` (sigmoid → ≈ 0), the forward output must equal a separately-instantiated `SlotAttention(K=4, D=4, I=6, T=3)` operating on the same input — **bit-exact within `1e-12`** because the gate multiplies to 0 and never enters the K/V projections.

5. **Gate ones → vanilla Slot Attention bit-exact**: with `W_g = 0, b_g = +100` (sigmoid → ≈ 1), bit-exact match to vanilla Slot Attention.

6. **Hand-derived N=1 forward reference matches at rel_err < 1e-6** for `D=2, I=2, K=2, T=1` — manually trace `g = σ(W_g · x + b_g)`, `k = (g ⊙ LN_k(x)) @ W_k^T + b_k`, etc.

7. **FD input gradient rel_err < 1e-5** with random non-uniform init (no degenerate init).

8. **FD `W_g.weights` gradient rel_err < 1e-5** — the new path. Mutation-tested non-vacuous: with `W_g` zeroed in the analytical, FD rel_err on input jumps to ~1.0 (proves the gate chain matters).

9. **FD `b_g` gradient rel_err < 1e-5**.

10. **FD `W_k`, `W_v`, `W_q`, `mu` gradients rel_err < 1e-5** — confirms the gate multiplication did not break the standard chain.

11. **Gate-perturbation signature**: forward output changes when `W_g` is perturbed by 0.5 (proves the gate affects output).

12. **Gate-perturbation is not all-zero grad**: `grad_W_g` has at least one entry with magnitude > 1e-10.

13. **`zero_grad()` clears every parameter gradient** including `W_g`, `b_g`.

14. **`parameters()` / `gradients()` contract**: `parameters()` and `gradients()` return same count; shapes match.

15. **`update_weights()` actually moves `W_g`** — `max|W_g_after − W_g_before|` > 1e-10.

16. **`GatedSlotAttentionModel` end-to-end training reduces loss**: build a model `(input_dim=4, slot_dim=4, K=3, hidden_dim=8, output_dim=2)`, run 50 SGD steps with MSE loss on a synthetic regression target, observe loss decrease.

**Verification:**
```bash
cd /home/stefan/neural_net
g++ -std=c++17 -O2 -Wall -Wextra -include/nn -fsyntax-only -x c++ - <<< '#include "nn/layers/attention/gated_slot_attention.h"'
make build/test_gated_slot_attention -j4
./build/test_gated_slot_attention
```
Expected (before implementation): compile FAILS at `#include "nn/layers/attention/gated_slot_attention.h"`.

---

## Task 2: Implement `gated_slot_attention.h` header

**Objective:** Declare the layer with the same Layer interface as the rest of the repo.

**Files:**
- Create: `include/nn/layers/attention/gated_slot_attention.h`

Mirror the public-API style of `slot_attention.h`:

```cpp
#ifndef GATED_SLOT_ATTENTION_H
#define GATED_SLOT_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../../core/dense.h"  // already pulled in via core
#include <vector>

// Gated Slot Attention — Bauer et al. NeurIPS 2024
//   https://arxiv.org/abs/2410.23790
//   Extension of Slot Attention (Locatello 2020) with a learned per-input gate.
//
// Math (single block, T iterations):
//   g        = σ(W_g · x + b_g)                         ∈ R^{N, input_dim}     (paper §3.2 "input gate")
//   k        = (g ⊙ LN_k(x)) @ W_k^T + b_k              ∈ R^{N, slot_dim}
//   v        = (g ⊙ LN_v(x)) @ W_v^T + b_v              ∈ R^{N, slot_dim}
//   slots_0  = mu                                       ∈ R^{K, slot_dim}      (learned init)
//   for t = 1..T:
//     q       = LN_q(slots) @ W_q^T + b_q                ∈ R^{K, slot_dim}
//     logits  = (q @ k^T) · dn^(-1/2)                   ∈ R^{K, N}
//     attn    = softmax(logits, axis=slots)             column-softmax
//     attn    = softmax(attn, axis=inputs)              row-softmax
//     updates = attn @ v                                ∈ R^{K, slot_dim}
//     slots   = GRU(updates, slots)
//     slots   = slots + MLP(LN(slots))                  residual MLP
//
// Backward:
//   Standard slot-attention chain (Locatello 2020) plus the W_g chain:
//     d_k_pre  = d_k · ... @ W_k
//     d_k_g    = d_k_pre · (W_k > 0 ? 1 : 0)...        // d(LN(x) * g) w.r.t. (LN(x) * g)
//     Actually: d_k = d_k_proj @ W_k, then d_k_proj is the gradient
//     of the k_proj pre-bias. d(g ⊙ LN_k(x)) = d_k_proj (broadcast over slot_dim).
//     d_g      = d(g ⊙ LN_k(x)) ⊙ LN_k(x) + d(g ⊙ LN_v(x)) ⊙ LN_v(x)   (×2 for k and v)
//     d_pre_sig= d_g ⊙ g ⊙ (1 − g)
//     dW_g     = d_pre_sig^T @ x
//     db_g     = sum(d_pre_sig, axis=0)
//     dx       += d_pre_sig @ W_g
//
// Public API:
//   * GatedSlotAttention(num_slots, slot_dim, input_dim, num_iterations=3, hidden_dim=0, epsilon=1e-8)
//   * GatedSlotAttentionModel(input_dim, slot_dim, num_slots, hidden_dim, output_dim, n_blocks=1, num_iterations=3)

class GatedSlotAttention : public Layer {
public:
    GatedSlotAttention(size_t num_slots, size_t slot_dim, size_t input_dim,
                       size_t num_iterations = 3, size_t hidden_dim = 0,
                       double epsilon = 1e-8);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_g_; }
    Tensor get_gradients() const override { return grad_W_g_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    size_t get_num_slots() const { return num_slots_; }
    const Tensor& last_gate() const { return last_gate_; }  // for tests

private:
    size_t num_slots_, slot_dim_, input_dim_, num_iterations_, hidden_dim_;
    double epsilon_;

    // Gate projection: W_g : input_dim → input_dim, b_g : 1 × input_dim
    Tensor W_g_, b_g_;
    Tensor grad_W_g_, grad_b_g_;

    // K/V/Q projections (Dense-style, weights (out, in) for x @ W^T + b)
    Tensor W_k_, W_v_, W_q_;
    Tensor b_k_, b_v_, b_q_;
    Tensor grad_W_k_, grad_W_v_, grad_W_q_;
    Tensor grad_b_k_, grad_b_v_, grad_b_q_;

    // Slot init mu: (K, D)
    Tensor mu_, grad_mu_;

    // LayerNorms
    LayerNorm ln_k_, ln_v_, ln_q_, ln_mlp_;

    // Residual MLP: Dense(D, H) → ReLU → Dense(H, D)
    Dense mlp_fc1_, mlp_fc2_;

    // Per-slot GRU (shared weights)
    Tensor W_zr_, W_h_, b_zr_, b_h_;
    Tensor grad_W_zr_, grad_W_h_, grad_b_zr_, grad_b_h_;

    // Caches for BPTT
    struct IterCache {
        Tensor slots_pre_gru, slots_post_gru, slots_post_mlp;
        Tensor k_proj, v_proj, q_proj;
        Tensor x_ln_k, x_ln_v, x_gated_k, x_gated_v, gate;  // NEW: gated x and g
        Tensor slots_ln_q, slots_ln_mlp;
        Tensor mlp_h;
        Tensor logits, attn1, attn2, updates;
        Tensor z_gates, r_gates, s_hat, rh;
    };
    std::vector<IterCache> cache_;

    Tensor last_input_, last_gate_;

    struct GruOut { Tensor new_s, z, r, s_hat, rh; };
    GruOut gru_forward(const Tensor& u, const Tensor& s);
    void gru_backward(const Tensor& grad_new_s, const GruOut& o,
                      Tensor& grad_u, Tensor& grad_s);
};

class GatedSlotAttentionModel : public Layer {
public:
    GatedSlotAttentionModel(size_t input_dim, size_t slot_dim, size_t num_slots,
                            size_t hidden_dim, size_t output_dim,
                            size_t n_blocks = 1, size_t num_iterations = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return input_proj_.get_weights(); }
    Tensor get_gradients() const override { return input_proj_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
private:
    Dense input_proj_, decoder_;
    std::vector<GatedSlotAttention> blocks_;
    LayerNorm ln_final_;
};

#endif
```

**Verification:**
```bash
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -fsyntax-only -x c++ - <<< '#include "nn/layers/attention/gated_slot_attention.h"'
```
Expected: PASS (compiles).

---

## Task 3: Implement `gated_slot_attention.cpp` (GREEN)

**Objective:** Make every test in `test_gated_slot_attention.cpp` pass at machine precision.

**Files:**
- Create: `include/nn/layers/attention/gated_slot_attention.cpp`

Pattern: copy `slot_attention.cpp` and add the gate chain. Two key insertions:

1. **Forward — after `x_ln_k` and `x_ln_v` are computed, do the gating**:
   ```cpp
   gate = sigmoid(W_g · x + b_g);          // (N, input_dim)
   x_gated_k = gate ⊙ x_ln_k;             // element-wise
   x_gated_v = gate ⊙ x_ln_v;
   k_proj = x_gated_k @ W_k^T + b_k;
   v_proj = x_gated_v @ W_v^T + b_v;
   ```
   Save `gate`, `x_ln_k`, `x_ln_v`, `x_gated_k`, `x_gated_v` in `IterCache` for backward.

2. **Backward — chain back through the gate**:
   ```cpp
   // d_x_gated_k = d_k_proj @ W_k     (N, input_dim) — same as the dense-backward path
   // d_x_gated_v = d_v_proj @ W_v
   // d_gate = d_x_gated_k ⊙ x_ln_k + d_x_gated_v ⊙ x_ln_v
   // d_pre_sig = d_gate ⊙ gate ⊙ (1 − gate)
   // dW_g += x^T @ d_pre_sig     (input_dim, input_dim)
   // db_g += Σ d_pre_sig
   // dx   += d_pre_sig @ W_g^T
   ```
   The chain into `d_x_ln_k` and `d_x_ln_v` then proceeds into `LayerNorm_k.backward` and `LayerNorm_v.backward` and finally `dx_ln → dx` via LN-backward. The `LayerNorm` module in this repo already handles the `(N, D) → (N, D)` backward chain; we just feed it the right input.

Use the **softmax-backward helper from `slot_attention.cpp`** for the double-softmax chain (don't reinvent — copy the function).

Use the **same GRU helper** structure as `slot_attention.cpp` (the `gru_forward` / `gru_backward` private methods). Don't recode the GRU from scratch.

**Verification:**
```bash
make build/test_gated_slot_attention -j4
./build/test_gated_slot_attention
```
Expected: All tests pass. Inspect output for "X/N passed" where N = 16.

**Watch for TDD pitfalls:**
- **Float sigmoid**: implement as `1/(1+exp(-z))` with a clamp `z = max(min(z, 60), -60)` to avoid inf in `exp`.
- **GRU initial hidden state**: the first iteration uses `slots_pre_gru = mu_` (the learned init). The cache at t=0 must store `slots_pre_gru = mu_`.
- **`update_weights`**: must update `W_g`, `b_g` in addition to all the Slot Attention parameters.
- **`zero_grad`**: must zero `grad_W_g`, `grad_b_g` too.

---

## Task 4: Register in umbrella + Makefile

**Objective:** Make the new layer discoverable by `#include "nn/nn.h"` and runnable via `make run_tests`.

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/attention/gated_slot_attention.h"` after the existing `#include "layers/attention/slot_attention.h"`.
- Modify: `Makefile`:
  - Add `$(BUILD_DIR)/build/test_gated_slot_attention: $(LIB_OBJS) $(BUILD_DIR)/test_gated_slot_attention.o` link rule (matches the existing pattern of test rules).
  - Add `$(BUILD_DIR)/test_gated_slot_attention` at the end of the `tests:` deps list.
  - Add `@echo "=== Running Gated Slot Attention Tests ===" && ./$(BUILD_DIR)/test_gated_slot_attention` at the end of `run_tests`.

**Verification:**
```bash
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -fsyntax-only -x c++ - <<< '#include "nn/nn.h"'
make tests -j4 2>&1 | tail -10
./build/test_gated_slot_attention
```

---

## Task 5: Mutation-test non-vacuousness

**Objective:** Ensure the gate chain isn't a vacuous test.

**Files:** `include/nn/layers/attention/gated_slot_attention.cpp`

Pick 2 representative mutations:
1. **Stub out the `d_pre_sig = d_gate ⊙ gate ⊙ (1 − gate)`** (set `d_pre_sig = d_gate`). This drops the sigmoid derivative.
2. **Stub out `d_gate = d_x_gated_k ⊙ x_ln_k + d_x_gated_v ⊙ x_ln_v`** (set `d_gate = d_x_gated_k ⊙ x_ln_k` only, dropping the V contribution).

For each: re-run `test_gated_slot_attention`. The FD tests for `W_g.weights`, `b_g`, and input should fail (rel_err > 0.01). Revert each mutation after confirming the failure mode is correct.

**Verification:** documented in the queue summary at the end.

---

## Task 6: Update `EXPANSION_QUEUE.md` and commit

**Objective:** Mark the idea Done with a one-line summary; commit atomically.

**Files:**
- Modify: `EXPANSION_QUEUE.md` — move the entry from `## Ideas` to `## Done`, with the full result summary appended.

**Commit:**
```bash
git add include/nn/layers/attention/gated_slot_attention.{h,cpp}
git add tests/test_gated_slot_attention.cpp
git add include/nn/nn.h Makefile
git add EXPANSION_QUEUE.md
git commit -m "feat(attention): Gated Slot Attention (Bauer et al., NeurIPS 2024)"
git push origin master
```

---

## Notes on paper faithfulness

- The paper §3.2 calls the input gate "input gates" and shows `g_x = σ(W_x · x + b_x)`. We use the same form with `W_g : input_dim → input_dim` (square projection — simplest and matches the paper's small-scale experiments).
- The paper §3.2 also proposes an "alpha scaling" `α ∈ [0,1]` per channel; we omit this (off-by-default, the paper notes it can hurt small models).
- The double-softmax competition, GRU update, and residual MLP follow Locatello 2020 unchanged — the paper's contribution is purely the input gate.

## TDD pitfalls to watch

1. **Sigmoid saturation**: at `z = ±100`, `exp(-z)` overflows. Clamp `z` to ±60 in the forward AND in the backward sigmoid derivative (where `g · (1-g)` is fine numerically, but the upstream `d_gate` may be very large — that's OK, just be aware).
2. **`(N, D)` vs `(1, D)` shape**: the gate is `(N, input_dim)`. The bias is `(1, input_dim)` and broadcasts. The LayerNorm input is `(N, input_dim)` for `x_ln_k` / `x_ln_v` — verify the existing `LayerNorm::forward` accepts `(N, D)` for `N > 1`.
3. **`update_weights` for `W_g`**: must use the Dense-style convention `W -= lr · grad`. For matrix weights, do row-wise: `W_g_(i,j) -= lr · grad_W_g_(i,j)`.
4. **`grad_W_g` accumulation across iterations**: the gate is applied at EVERY iteration (the input `x` is the same, so `g` is the same — but if you recompute `g` per iteration, `dW_g` accumulates `T` times). The cleanest design is to compute `g` ONCE before the iteration loop (since the input doesn't change), then `dW_g` is computed once after the loop. This matches the paper's notation and is simpler.
5. **Mutation-test the gate chain**: a common bug class is to forget to chain the gate backward into `W_g` or into `x`. Always run with `W_g = randn(0.3)` and `b_g = 0` (init with non-degenerate gate) so the FD tests can detect this.

## Reference files

- `include/nn/layers/attention/slot_attention.{h,cpp}` — the parent class, copy/adapt structure
- `tests/test_slot_attention.cpp` — test style reference
- `docs/plans/2026-09-09-deformable-attention.md` — recent plan with the same Template A shape
