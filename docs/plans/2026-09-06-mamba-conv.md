# MambaConv Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a hand-rolled `MambaConvBlock` to `include/nn/layers/recurrent/mamba_conv.{h,cpp}` that implements the *canonical* Mamba-1 block with the depthwise causal 1D convolution BEFORE the selective SSM (the design feature that the existing `MambaBlock` class deliberately omits — see `include/nn/layers/recurrent/mamba.h` comment "We do NOT include the depthwise 1D conv..."). Match the existing `MambaBlock` API so FD tests, training loops, and serialisation can drop in cleanly.

**Architecture:**
- `MambaConvBlock(d_model, d_state, d_inner=0, conv_kernel=4)` — full Mamba-1 design: input projection → depthwise causal 1D conv on the SSM path (kernel size `conv_kernel`, default 4 per the paper) → SiLU → selective SSM → gated output projection. Skip the conv when `conv_kernel == 1`.
- Full analytical backward covering the conv chain (causal left-padded depthwise shift), the SiLU chain, the selective SSM BPTT (Ā/B̄ chains + recurrence carriers), the SiLU-gate chain, and the projection chains through all 5 Dense layers + A_log + D_skip.

**Tech Stack:** Existing C++17 / hand-rolled tensor library; convention matches the rest of `recurrent/`.

---

## Reference: the math

Per sequence position t with input x_t ∈ R^{d_model}:

1. **Joint input projection**: `p_t = in_proj(x_t) ∈ R^{2·d_inner}`, split into `[x_pre_t; gate_t]`.
2. **(NEW vs `MambaBlock`) Depthwise causal 1D conv on SSM path**: `x_pre_t'[i] = bias_c[i] + Σ_{j=0..k-1} weight_c[i, j] · x_pre_{t-j}[i]` with left-padding (j > t contributes zero). One filter per channel (depthwise = `d_inner` filters of length `k`, weight shape `(d_inner, k)`).
3. **SiLU**: `x_ssm_t[i] = silu(x_pre_t'[i])`.
4. **Input-dependent SSM params**: Δ_t = softplus(dt_proj(x_t)), B_t = B_proj(x_t), C_t = C_proj(x_t).
5. **ZOH discretisation**: A = -exp(A_log) (negative), `Ā_t = exp(Δ_t ⊗ A)`, `B̄_t = Δ_t ⊗ B_t`.
6. **Selective scan**: `h_0 = 0`, `h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x_ssm_t`, `y_t = C_t · h_t`.
7. **Gating + output**: `gated_t = silu(gate_t) ⊙ y_t`, `out_t = out_proj(gated_t) ⊕ D_skip ⊙ x_ssm_t` (D_skip applied to x_ssm BEFORE the gating so the skip path carries the conv+SiLU output; alternatively we follow the existing `MambaBlock` convention of `D_skip ⊙ y` — see existing mamba.cpp §Step 7 for the canonical Mamba-1 skip convention).

### Causal left-padding convention

A convolution with kernel size k applied to a length-T sequence requires zero-padding on the left of length (k-1). Positions t = 0..k-2 thus "see" fewer than k inputs (some are zero). This is the Mamba-1 paper's exact convention.

---

## Why this layer is interesting for the repo

- The existing `MambaBlock` is widely used but explicitly *not* the canonical Mamba — its header note calls out the omission. Adding `MambaConvBlock` closes that gap with a single self-contained layer.
- It is a faithful reproduction of the design that gets quoted everywhere ("Mamba = SSM + 1D conv"), but **with a hand-rolled analytical backward** rather than `torch.compile` tricks. The depthwise conv's backward is just a causal accumulation; no im2col required.
- Provides a clean comparison point: `MambaConvBlock(d_model, d_state)` vs `MambaBlock(d_model, d_state)` — the conv adds local-mixing context before the SSM, helping with token-locality tasks.

---

## Tasks

### Task 1: Header skeleton

**Files:**
- Create: `include/nn/layers/recurrent/mamba_conv.h`

**Step 1.1:** Define the class with the canonical math comment header (mirror `mamba.h`). Public interface mirrors `MambaBlock`:
```cpp
class MambaConvBlock : public Layer {
public:
    MambaConvBlock(size_t d_model, size_t d_state, size_t d_inner = 0, size_t conv_kernel = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "MambaConvBlock"; }

    size_t d_model()    const { return d_model_; }
    size_t d_state()    const { return d_state_; }
    size_t d_inner()    const { return d_inner_; }
    size_t conv_kernel() const { return conv_kernel_; }

    // Public parameter tensors
    Dense in_proj;          // d_model -> 2*d_inner
    Dense out_proj;         // d_inner -> d_model
    Dense dt_proj;          // d_model -> d_inner
    Dense B_proj;           // d_model -> d_state
    Dense C_proj;           // d_model -> d_state
    Tensor A_log;           // (d_inner, d_state)
    Tensor D_skip;          // (1, d_inner)
    Tensor conv_weight;     // (d_inner, conv_kernel)  ONE filter per channel (depthwise)
    Tensor conv_bias;       // (1, d_inner)

    // Hidden gradient buffers
    Tensor grad_A_log_;
    Tensor grad_D_skip_;
    Tensor grad_conv_weight_;
    Tensor grad_conv_bias_;

private:
    size_t d_model_;
    size_t d_state_;
    size_t d_inner_;
    size_t conv_kernel_;

    // Forward caches
    Tensor last_input_;
    Tensor last_p_;
    Tensor last_x_pre_;     // raw post-in_proj (T, d_inner)
    Tensor last_x_conv_;    // post-conv + bias  (T, d_inner)
    Tensor last_ssm_in_;    // post-SiLU        (T, d_inner)
    Tensor last_gate_;
    Tensor last_Delta_;
    Tensor last_Delta_pre_;
    Tensor last_B_t_;
    Tensor last_C_t_;
    Tensor last_A_bar_;
    Tensor last_B_bar_;
    Tensor last_h_;
    Tensor last_y_;
    Tensor last_gated_;

    // Helpers
    static double softplus(double x);
    static double sigmoid(double x);
    static double silu(double x);
};
```

**Step 1.2:** Verify header compiles standalone (`#include "../../core/layer.h"` + `#include "../normalization/layer_norm.h"` not needed yet).

---

### Task 2: Constructor + zero_grad + accessors

**Step 2.1:** Implement the constructor following the `MambaBlock` pattern:
- If `d_inner == 0`, set `d_inner = 2 * d_model` (paper convention).
- If `conv_kernel == 0`, throw.
- Reconstruct `in_proj = Dense(d_model, 2*d_inner)`, `dt_proj = Dense(d_model, d_inner)`, `out_proj = Dense(d_inner, d_model)`.
- Initialise `conv_weight` with Xavier-uniform per-channel (scale `1/sqrt(conv_kernel)`), `conv_bias = 0`.
- Initialise `A_log` small (so `|A| = exp(A_log) ~ 1`), `D_skip = 1.0` (Mamba paper).
- Zero all projection biases (matches `MambaBlock`).

**Step 2.2:** Implement `zero_grad()`: zero all 5 Dense gradients + `grad_A_log_`, `grad_D_skip_`, `grad_conv_weight_`, `grad_conv_bias_`.

**Step 2.3:** Implement `parameters()` returning `{&in_proj.weights, &in_proj.bias, &dt_proj.weights, &dt_proj.bias, &out_proj.weights, &out_proj.bias, &B_proj.weights, &B_proj.bias, &C_proj.weights, &C_proj.bias, &A_log, &D_skip, &conv_weight, &conv_bias}` (14 tensors).

**Step 2.4:** Implement `gradients()` matching.

**Step 2.5:** Implement `update_weights()` — mirror `MambaBlock`'s update_weights: 5 Dense::update_weights + manual updates for `A_log`, `D_skip`, `conv_weight`, `conv_bias` (`param -= lr * grad`).

---

### Task 3: Write the failing test — constructor validation + forward shape

**File:** `tests/test_mamba_conv.cpp`

```cpp
#include <gtest/gtest.h>
#include "nn/nn.h"

TEST(MambaConvBlock, ConstructorValidation) {
    // d_model = 0 throws
    EXPECT_THROW(MambaConvBlock(0, 4, 0, 4), std::invalid_argument);
    // conv_kernel = 0 throws
    EXPECT_THROW(MambaConvBlock(4, 4, 0, 0), std::invalid_argument);
    // valid constructs
    EXPECT_NO_THROW(MambaConvBlock(4, 4, 0, 4));
    EXPECT_NO_THROW(MambaConvBlock(4, 4, 8, 1));   // conv_kernel=1 = no conv mixing
    EXPECT_NO_THROW(MambaConvBlock(4, 4, 8, 2));
}

TEST(MambaConvBlock, ForwardShape) {
    MambaConvBlock b(4, 4, 8, 4);
    Tensor x(3, 4);
    x.random(0.1);
    Tensor y = b.forward(x);
    EXPECT_EQ(y.rows, 3u);
    EXPECT_EQ(y.cols, 4u);
    EXPECT_TRUE(y.all_finite());
    EXPECT_GT(y.abs().max(), 0.0);
}
```

Run: `make build/test_mamba_conv && ./build/test_mamba_conv`
Expected: 2 tests pass after implementation; RED initially (function-not-defined).

---

### Task 4: Implement forward

**Step 4.1:** `forward(input)`:
- Validate `input.cols == d_model_`, `input.rows >= 1`.
- `last_input_ = input.clone()`.
- `last_p_ = in_proj.forward(input)`. Split into `last_x_pre_` and `last_gate_`.
- **Apply depthwise causal 1D conv** to `last_x_pre_`:
  ```cpp
  last_x_conv_ = Tensor(T, d_inner_);
  for (size_t i = 0; i < d_inner_; ++i) {
      for (size_t t = 0; t < T; ++t) {
          double acc = conv_bias(0, i);
          for (size_t j = 0; j < conv_kernel_; ++j) {
              if (t >= j) acc += conv_weight(i, j) * last_x_pre_(t - j, i);
          }
          last_x_conv_(t, i) = acc;
      }
  }
  ```
- SiLU: `last_ssm_in_(t, i) = silu(last_x_conv_(t, i))`.
- `dt_proj.forward(input)` → `last_Delta_pre_` → softplus → `last_Delta_`.
- `B_proj.forward(input)` → `last_B_t_`.
- `C_proj.forward(input)` → `last_C_t_`.
- Build `last_A_bar_(t, i, d) = exp(last_Delta_(t, i) * (-exp(A_log(i, d))))`.
- Build `last_B_bar_(t, i, d) = last_Delta_(t, i) * last_B_t_(t, d)`.
- Recurrence: `last_h_(0, ., .) = 0`, `last_h_(t, i, d) = last_A_bar_(t, i, d) * last_h_(t-1, i, d) + last_B_bar_(t, i, d) * last_ssm_in_(t, i)`.
- `last_y_(t, i) = Σ_d last_C_t_(t, d) * last_h_(t, i, d)`.
- SiLU gate: `last_gated_(t, i) = silu(last_gate_(t, i)) * (last_y_(t, i) + D_skip(0, i) * last_ssm_in_(t, i))`. (D_skip applied to x_ssm, then summed with y, then gated — this is the "selective skip" Mamba-1 convention. We follow the existing `MambaBlock`'s `D_skip ⊙ y` convention for consistency: `last_gated_(t, i) = silu(last_gate_(t, i)) * (last_y_(t, i) + D_skip(0, i) * last_ssm_in_(t, i))`.)
- `out_proj.forward(last_gated_)` → return.

---

### Task 5: Write the failing test — forward against a known hand-derived reference

For T=1, d_state=1, d_inner=1, conv_kernel=2, we can compute the full forward output by hand given a small input. Hand-derive:
- x = [[x0]] (T=1, d_model=1)
- p = in_proj(x) = [[x0 * W_in1 + b_in1, x0 * W_in2 + b_in2]]
  → x_pre = [x0 * W_in1 + b_in1]
  → gate = [x0 * W_in2 + b_in2]
- conv (kernel=2, depthwise): for t=0, only j=0 contributes (j=1 needs t-j=−1, excluded).
  → x_conv[0] = bias_c[0] + conv_w[0] * x_pre[0]
- silu(x_conv[0])
- Δ[0] = softplus(x0 * W_dt + b_dt)
- B[0] = x0 * W_B + b_B
- C[0] = x0 * W_C + b_C
- A[0,0] = -exp(A_log[0,0])
- Ā[0,0,0] = exp(Δ[0] * A[0,0])
- B̄[0,0,0] = Δ[0] * B[0]
- h[0,0,0] = 0 * Ā + B̄ * x_ssm = B̄ * x_ssm
- y[0] = C[0] * h[0]
- gated[0] = silu(gate[0]) * (y[0] + D_skip[0] * x_ssm[0])
- out[0] = gated[0] * W_out + b_out

For T=2 the math is closed-form too. Test asserts `rel_err < 1e-13` between hand-derived and impl.

---

### Task 6: Write the failing test — input gradient FD check

Standard centered FD with ε=1e-5 on a small config (d_model=4, d_state=2, d_inner=4, conv_kernel=2) for T=3 and T=6. Assert `rel_err < 1e-3`.

---

### Task 7: Implement backward

The backward pass, in reverse order:

1. **Output projection chain**: `d_last_gated_ = out_proj.backward(d_out, lr)`.
2. **Gated chain**: `d_last_y_[t, i] = d_last_gated_[t, i] * silu(last_gate_(t, i))`.
   - `d_D_skip[0, i] += Σ_t d_last_gated_[t, i] * last_ssm_in_[t, i] * silu(last_gate_(t, i))` (no — `last_gated = silu(gate) * (y + D_skip * x_ssm)`, so `d(D_skip[0, i]) = Σ_t d_last_gated[t, i] * silu(gate[t, i]) * x_ssm[t, i]`).
   - `d_last_ssm_in_[t, i] += d_last_gated_[t, i] * silu(last_gate_(t, i)) * D_skip(0, i)` (the skip path through D_skip).
   - `d_last_gate_[t, i] = d_last_gated_[t, i] * (y[t, i] + D_skip[0, i] * x_ssm[t, i]) * silu'(last_gate_(t, i))`.
3. **SSM backward** (same as `MambaBlock`'s proven chain):
   - `d_C_t[t, d] = Σ_i d_last_y_[t, i] * last_h_(t, i, d)` → `grad_C_proj`.
   - `d_h_(T, ., .) = 0`, walk t = T-1..0:
     - `d_h_(t, i, d) += d_C_t[t, d] * last_y_(t, i)` — wait, recompute: `last_y_(t, i) = Σ_d last_C_t_(t, d) * last_h_(t, i, d)`, so `d last_h_(t, i, d)|_{C path} = d_last_y_(t, i) * last_C_t_(t, d)`.
     - Plus the chain from `h_(t+1)`: `d_h_(t, i, d) += Ā_(t+1, i, d) * d_h_(t+1, i, d)` (only Ā drives propagation; B̄ contributes to `d_last_ssm_in_`).
   - `d_ssm_in_chain[t, i] = Σ_d d_h_(t, i, d) * last_B_bar_(t, i, d)`.
   - `d_A_log_(i, d) = Σ_t -exp(A_log(i, d)) * last_Delta_(t, i) * d_h_(t, i, d) * (something from the chain)` — the full chain follows `MambaBlock`'s `grad_A_log_` formula exactly: `grad_A_log_[i, d] = Σ_t -exp(A_log(i, d)) · last_Delta_(t, i) · (last_h_(t-1, i, d) · d_h_(t, i, d))`. Use the existing proven formula.
   - `d_Delta_(t, i)` from SSM: `d_Delta_(t, i) = Σ_d [last_h_(t-1, i, d) * d_h_(t, i, d) * exp(Δ_t[i] * A[i,d]) * A[i, d] + last_B_t_(t, d) * d_h_(t, i, d) * last_ssm_in_(t, i)]`.
4. **dt_proj / B_proj / C_proj backward**: standard Dense backward with the per-element contributions as `MambaBlock` does.
5. **SiLU backward on x_ssm**: `d_last_x_conv_[t, i] = d_last_ssm_in_[t, i] * silu'(last_x_conv_(t, i))`.
6. **Depthwise causal conv backward** (NEW chain):
   - `d_conv_bias_[0, i] = Σ_t d_last_x_conv_[t, i]`.
   - `d_conv_weight_[i, j] = Σ_t d_last_x_conv_[t, i] * last_x_pre_(t - j, i)` for `t >= j`, else 0.
   - `d_last_x_pre_[t, i] = Σ_{j=0..k-1} conv_weight_(i, j) * d_last_x_conv_(t + j, i)` for `t + j < T`, else 0.
7. **In-proj backward**: `d_p_ = in_proj.backward(concat(d_last_x_pre_, d_last_gate_), lr)`. Then `d_last_input_ = d_p_ + d_dt_proj.contribution + d_B_proj + d_C_proj`. Sum all four contributions to the input.

This is the exact chain — mirror `MambaBlock`'s code, only adding the depthwise conv stages in the right places.

---

### Task 8: Write the failing test — param gradient FD checks

For each of the 14 parameter tensors, do a centered FD check with ε=1e-5 on a small config. Assert `rel_err < 1e-3`:
- `in_proj.weights`, `in_proj.bias`
- `dt_proj.weights`, `dt_proj.bias`
- `B_proj.weights`, `B_proj.bias`
- `C_proj.weights`, `C_proj.bias`
- `out_proj.weights`, `out_proj.bias`
- `A_log`, `D_skip`
- `conv_weight`, `conv_bias` (new!)

---

### Task 9: Write the failing test — training reduces loss

Standard 50-step SGD on a small regression task. Assert `L_final < L_initial * 0.5`.

---

### Task 10: Write the failing test — contract tests

- `parameters()` and `gradients()` return 14 matching-shape tensors.
- `zero_grad()` clears all 14 gradients (norms == 0).
- `update_weights(lr)` moves all 14 parameters.
- `conv_kernel == 1` reduces to `MambaBlock` behavior (mathematically not bit-exact because of the extra conv_bias and conv_weight on a 1-tap kernel, but conceptually equivalent).

---

### Task 11: Register in `include/nn/nn.h` and Makefile

- Add `#include "layers/recurrent/mamba_conv.h"` to the umbrella (right after `mamba3.h`).
- Add `$(BUILD_DIR)/test_mamba_conv: $(LIB_OBJS) $(BUILD_DIR)/test_mamba_conv.o` rule.
- Add `$(BUILD_DIR)/test_mamba_conv` to the `tests:` deps line.
- Add `@echo "=== Running MambaConv Tests ===" && ./$(BUILD_DIR)/test_mamba_conv` to `run_tests`.

---

### Task 12: Commit

```bash
git add include/nn/layers/recurrent/mamba_conv.h \
        include/nn/layers/recurrent/mamba_conv.cpp \
        tests/test_mamba_conv.cpp \
        include/nn/nn.h \
        Makefile \
        EXPANSION_QUEUE.md \
        docs/plans/2026-09-06-mamba-conv.md
git commit -m "feat(recurrent): MambaConv (Gu & Dao 2023) — canonical Mamba-1 with depthwise causal 1D conv"
git push origin master
```

Move the queue entry from `## Ideas` to `## Done` with a one-line summary + test counts.

---

## Bug catalogue (from prior Mamba sessions, watch for these)

1. **`MambaBlock`'s convention is `D_skip ⊙ y`, not `D_skip ⊙ x_ssm`** — keep this consistent. Use `gated = silu(gate) * (y + D_skip * x_ssm)`.
2. **Softplus** must use the branched form (`x > 30`, `x < -30`) to avoid NaN on large dt_proj outputs during FD.
3. **dt_proj input gradient** — `d_last_input_` has FOUR contributions (one each from in_proj, dt_proj, B_proj, C_proj). Don't forget any.
4. **`d_last_h_` walk** — start at `T` (the `h_T` "future" slot has zero incoming gradient), walk backward.
5. **Conv backward indexing** — `d_x_pre_(t, i) = Σ_j conv_w(i, j) * d_x_conv_(t+j, i)` for `t+j < T`. With conv_kernel=4 and T=3, only j=0 contributes at t=2. Test this.
6. **Conv weight init** — must NOT be all zeros (forward would be constant = bias only). Xavier scale `1/sqrt(k)`.
7. **Conv gradient of bias** — sum over T, not over channels.
8. **Sequence length T < conv_kernel** — must still work (left-pad). For T=1, conv_kernel=4, all j > 0 are zero, so output = `bias + w[0] * x_pre[0]`.

## Estimated test count: 18-25 focused checks (constructor × 4, forward shape, finite, nonzero, hand-derived T=1, hand-derived T=2, input grad FD T=3, T=6, 14 param grad FD checks, training reduces loss, parameters/gradients contract, zero_grad, update_weights, conv_kernel=1 sanity).
