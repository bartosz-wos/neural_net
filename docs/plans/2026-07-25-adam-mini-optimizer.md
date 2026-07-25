# Adam-mini Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add Adam-mini (Zhang et al. 2024, https://arxiv.org/abs/2406.16793) — a memory-efficient Adam variant that uses block-wise second-moment estimates to cut optimizer state memory by ~50% without performance loss.

**Architecture:** A `BlockAdam` (a.k.a. Adam-mini) optimizer that, for each parameter, decides between three block-reduction strategies based on parameter rank: **(a) full per-element v** for 1-D parameters (biases, norms — these have small total element count and per-element adaptive LR is critical), **(b) row-mean v** for 2-D weight matrices (one LR per row = "per neuron"), and **(c) scalar v** as a global fallback. Default behaviour: 2-D → row-mean; 1-D → full. Configurable via enum for the block-mode policy.

**Tech Stack:** C++17, existing `Tensor`, `Optimizer`, `Model`, `Layer` infrastructure. Header in `include/nn/optimizers/adam_mini.{h,cpp}`; tests in `tests/test_adam_mini.cpp`. Plan: bite-sized, each task ≤5 min focused work.

---

## Source / Reference

- Paper: Zhang et al. 2024, *Adam-mini: Use Fewer Learning Rates To Gain More*, https://arxiv.org/abs/2406.16793
- Official PyTorch reference: https://github.com/zyushun/Adam-mini/blob/main/adam_mini/adam_mini.py

## Paper Algorithm (per parameter θ)

For each parameter θ, partition the gradient `g_t` into one of three reduction schemes:

**(A) Full Adam** (used for 1-D parameters — biases, norms, embeddings): identical to Adam:
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t
v_t = β2 * v_{t-1} + (1 - β2) * g_t²
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
update = m̂_t / (√v̂_t + ε)
θ -= lr * update
```

**(B) Row-wise Adam-mini** (used for 2-D weight matrices — MLP, attn_proj, V projections, output, embedding): v is reduced to a single value per row (mean of `g²` along the column axis):
```
For each row i ∈ [0, rows):
  vmean_t[i] = β2 * vmean_{t-1}[i] + (1 - β2) * (1/cols) * Σ_j g_t[i,j]²
m_t = β1 * m_{t-1} + (1 - β1) * g_t                   # (m stays full)
m̂_t = m_t / (1 - β1^t)
update[i,j] = m̂_t[i,j] / (√v̂_t[i] / √(1-β2^t) + ε)   # broadcast v̂_t[i] across columns
θ -= lr * update
```

**(C) Scalar Adam-mini** (single global LR — used for very small parameters or as a fallback):
```
vmean_t = β2 * vmean_{t-1} + (1 - β2) * mean(g_t²)   # one scalar
m_t = β1 * m_{t-1} + (1 - β1) * g_t
update = m / (√(vmean/(1-β2^t)) + ε)
θ -= lr * update
```

Weight decay is decoupled (AdamW style): `θ *= (1 - lr * wd)` applied BEFORE the update.

## Memory savings

For a 2-D `rows × cols` matrix:
- Adam: `v` has `rows * cols` doubles (full)
- Adam-mini (row-wise): `vmean` has `rows` doubles → `cols × ` smaller

For a typical transformer weight matrix `4096 × 4096`, Adam stores `4096² ≈ 16.8M` doubles for v; Adam-mini stores `4096` doubles — a 4096× reduction. Across all 2-D params in a transformer, this roughly halves total optimizer state.

## Design choices

1. **Name-agnostic policy**: The paper uses parameter-name matching (`wqk`, `wv`, `attn_proj`, etc.) to pick the block reduction. Our `Layer` interface exposes `parameters()` without names. We therefore default to **shape-based policy**:
   - 1-D (`cols == 1` or `rows == 1`) → full Adam (option A)
   - 2-D → row-mean (option B)
   - Custom policy: user can force `BlockMode::FULL`, `ROW_MEAN`, or `SCALAR` for all parameters via constructor enum.

2. **Per-parameter block-mode override**: Add a `set_param_block_mode(idx, mode)` accessor that lets callers force a specific block-mode for individual parameters (e.g. for testing the A/B/C variants independently on the same layer).

3. **State per parameter**: 2 tensors — `m` (full shape) + `vmean` (1-D or scalar). Half the state size of Adam for 2-D params.

4. **No Hessian or auxiliary state** (unlike Sophia). Just `m + vmean` per parameter.

---

## File-by-File Plan

### Files to create

- `include/nn/optimizers/adam_mini.h`
- `include/nn/optimizers/adam_mini.cpp`
- `tests/test_adam_mini.cpp`

### Files to modify

- `include/nn/nn.h` — add `#include "optimizers/adam_mini.h"` after the existing optimizer includes.
- `Makefile` — add build rule for `build/test_adam_mini`, add to `tests` target list, add to `run_tests` invocations and ordering.
- `EXPANSION_QUEUE.md` — move this item to `## Done` after commit.

### Out of scope (deliberate)

- **FSDP / distributed all-reduce**: the official PyTorch version has `model_sharding` with cross-GPU `dist.all_reduce` for the LayerNorm fallback case. We don't have FSDP in this repo; the LayerNorm path uses in-process scalars.
- **Transformer-specific name matching (`wqk_names` etc.)**: would require extending `Layer` to expose parameter names, which is out of scope. Shape-based policy captures the main memory savings.

---

## Tasks (TDD — each task 2-5 min focused work)

### Task 1: Write the failing test for `AdamMini` construction & defaults

**Objective:** Verify that `AdamMini()` default-constructs with sane hyperparameters (lr=1e-3, β=(0.9, 0.999), ε=1e-8, wd=0) and that invalid hyperparameters throw via setter validation.

**Files:**
- Create: `tests/test_adam_mini.cpp` (initial scaffold with T1-T2 only)

**Steps:**

**Step 1.1: Write failing test** — add to `tests/test_adam_mini.cpp`:
```cpp
#include "nn/optimizers/adam_mini.h"
#include "nn/core/model.h"
#include "nn/layers/dense/dense.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace nn;

    // T1: defaults
    AdamMini opt;
    assert(opt.lr == 1e-3);
    assert(opt.beta1 == 0.9);
    assert(opt.beta2 == 0.999);
    assert(opt.epsilon == 1e-8);
    assert(opt.weight_decay == 0.0);
    assert(opt.t == 1);

    // T2: custom constructor
    AdamMini opt2(2e-3, 0.85, 0.95, 1e-6, 0.01);
    assert(opt2.lr == 2e-3);
    assert(opt2.beta1 == 0.85);
    assert(opt2.beta2 == 0.95);
    assert(opt2.epsilon == 1e-6);
    assert(opt2.weight_decay == 0.01);

    std::cout << "T1-T2 passed\n";
    return 0;
}
```

**Step 1.2: Compile the test (will FAIL because `adam_mini.h` doesn't exist yet).** Expected: compile error — `adam_mini.h: No such file or directory`.

### Task 2: Implement `AdamMini` header with construction & accessors

**Objective:** Create the header that satisfies T1-T2.

**Files:**
- Create: `include/nn/optimizers/adam_mini.h`

**Step 2.1:** Write the header (full content — see "Header file" section below).

**Step 2.2:** Recompile T1-T2 — expected: compile success, T1-T2 pass.

### Task 3: Test setters & validation

**Objective:** `set_lr`, `set_beta1`, `set_beta2`, `set_epsilon`, `set_weight_decay` validate inputs (throw on negative LR, β outside [0,1), ε ≤ 0, wd < 0).

**Files:**
- Modify: `tests/test_adam_mini.cpp` (append T3-T7)

**Step 3.1:** Write tests for setter validation. T3-T7 should each verify one rejection.

**Step 3.2:** Implement the setters with validation.

**Step 3.3:** Run — all pass.

### Task 4: Test `step()` with full Adam mode on a 1-D parameter

**Objective:** For a 1-D parameter (bias), `AdamMini` with `BlockMode::FULL` should reproduce Adam's first-step behaviour exactly. Hand-derive a closed-form expected value.

**Files:**
- Modify: `tests/test_adam_mini.cpp` (append T8)

**Step 4.1:** Write test that constructs a `Model` with a 1-D parameter, sets gradients, runs one step, and asserts the parameter moved by the expected Adam quantity (computed by hand using β1=β2=0.5, lr=1, ε=1, g=1, init=0).

Closed-form for 1-D FULL mode (β1=β2=0.5, lr=1, ε=1, g=1, init=0):
```
t=1:  m_1 = 0.5*0 + 0.5*1 = 0.5
      v_1 = 0.5*0 + 0.5*1 = 0.5
      m̂_1 = 0.5 / (1-0.5) = 1.0
      v̂_1 = 0.5 / (1-0.5) = 1.0
      denom = √1.0 + 1 = 2.0
      update = 1.0 / 2.0 = 0.5
      new θ = 0 - 1*0.5 = -0.5
```

**Step 4.2:** Implement `step()` covering the FULL path.

**Step 4.3:** Run — T8 passes.

### Task 5: Test `step()` with row-mean mode on a 2-D parameter (closed form)

**Objective:** For a 2-D parameter (4×3), `AdamMini` with `BlockMode::ROW_MEAN` should compute `vmean[i] = mean(g[i, :]**2)` per row, broadcast back across columns.

Hand-derive (β1=β2=0.5, lr=1, ε=1, init=0):
```
g = [[1, 1, 1],   vmean_per_row_step1 = [1, 1, 1, 1]  (each row mean of 1² = 1)
     [1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]
m_1 = 0.5*0 + 0.5*g = 0.5 (full m, same shape as g)
m̂_1 = 1.0 (after bias correction)
v̂_1 = 1.0 (same)
denom = √1.0 + 1 = 2.0  (broadcast across columns)
update = 0.5 / 2.0 = 0.25
new θ = -0.25 (all entries)
```

**Files:** Append T9.

### Task 6: Test `step()` with scalar mode on a 2-D parameter

**Objective:** For a 2-D parameter with `BlockMode::SCALAR`, the entire parameter shares one LR (vmean is one scalar).

Hand-derive with the same 4×3, g=1 (all entries) config:
```
vmean_step1 = (1/12) * 12 = 1.0
m_1 = 0.5 (full)
denom = √1.0 + 1 = 2.0
update = 0.25
new θ = -0.25 (all entries)   # same as row-mean in this all-equal case
```

Add an asymmetric test: g = [[1,1,1],[1,1,1],[2,2,2],[2,2,2]] → vmean should differ between rows.

**Files:** Append T10.

### Task 7: Test step counter increments and bias correction affects subsequent steps

**Objective:** T=2 should produce different m̂/v̂ than T=1 because the bias correction 1-β^t changes.

**Files:** Append T11-T12.

### Task 8: Test state shape correctness

**Objective:** `has_state()` returns false before step, true after. `get_m(param)` and `get_vmean(param)` return correctly-shaped tensors (m: same shape as param; vmean: (rows, 1) for row-mean, (1, 1) for scalar, full for full).

**Files:** Append T13-T15.

### Task 9: Test decoupled weight decay

**Objective:** With `wd > 0`, the parameter shrinks by `(1 - lr*wd)` BEFORE the update is applied.

**Files:** Append T16.

### Task 10: Test step-with-zero-grad doesn't crash and doesn't move params

**Objective:** Edge case: a parameter with zero gradient should not change after step.

**Files:** Append T17.

### Task 11: Test parameter-count and shape mismatch throws

**Objective:** Calling step when gradients size != params size should throw. Same for shape mismatch.

**Files:** Append T18.

### Task 12: Test multi-layer model (independent state per layer)

**Objective:** Two layers, each with its own params. State should be isolated — running step updates both, but state isn't shared.

**Files:** Append T19.

### Task 13: Test determinism (two fresh instances produce identical trajectories)

**Objective:** Two `AdamMini` instances with the same seed produce bit-exact parameter trajectories over multiple steps.

**Files:** Append T20.

### Task 14: Test end-to-end training loss reduction

**Objective:** Linear regression `y = 2x` with `AdamMini` reduces loss by ≥ 90% over 100 steps.

**Files:** Append T21.

### Task 15: Test row-mean memory savings vs Adam

**Objective:** For a (8, 256) parameter, `AdamMini` row-mean state size is `8 + 8*256 = 2056` doubles (m + vmean). Adam would be `8*256 + 8*256 = 4096` doubles. Confirm the `vmean` shape is `(rows, 1)`.

**Files:** Append T22.

### Task 16: Test per-parameter block-mode override

**Objective:** `set_param_block_mode(idx, mode)` overrides the default policy for a single parameter.

**Files:** Append T23.

### Task 17: Test signature vs AdamW on same gradient sequence

**Objective:** `AdamMini` and `AdamW` produce different parameter trajectories under the same gradients (because vmean reduction ≠ per-element v).

**Files:** Append T24.

### Task 18: Test bias/norm handling (1-D parameter falls back to FULL)

**Objective:** Default behaviour: 1-D params use FULL, 2-D params use ROW_MEAN. Verify by constructing a `Dense` layer with weights (2-D) and biases (1-D) and checking both go to the right state shape.

**Files:** Append T25.

### Task 19: Test empty layer skipped without crash

**Objective:** Layer with zero parameters should be skipped (no state, no crash).

**Files:** Append T26.

### Task 20: Mutation testing

**Objective:** Apply 4 mutations and confirm each is caught by the test suite:
1. Drop `vmean` (use per-element v like Adam) — at least one test should fail (T9/T10/T15 vmean shape or T17 zero-grad).
2. Wrong bias correction (use `1-β1^t` instead of `1/(1-β1^t)`) — T8/T9 closed-form should fail.
3. Skip weight decay application — T16 should fail.
4. Wrong block-mode for 1-D (use ROW_MEAN on 1-D, which broadcasts to (1,1)) — T8 should fail because the closed-form assumes full v.

**Files:** No file change. Run mutation tests against the existing suite.

### Task 21: Register in umbrella & Makefile

**Objective:** Add `#include "optimizers/adam_mini.h"` to `include/nn/nn.h`. Add `build/test_adam_mini` rule to Makefile. Add it to the `tests` aggregate target. Add it to the `run_tests` invocation list.

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`

### Task 22: Run full Makefile build to ensure no regressions

**Objective:** `make tests` compiles all registered binaries without warnings or errors.

**Files:** No change.

### Task 23: Run `make run_tests` and confirm `AdamMini` passes; no other tests regressed

**Objective:** Confirm new optimizer's test suite passes; aggregate suite still passes for everything except the deferred ones in `NOT_FIXED.md`.

**Files:** No change.

### Task 24: Update `EXPANSION_QUEUE.md` and commit

**Objective:** Move the Adam-mini entry to `## Done` with a one-line summary, then commit + push.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

---

## Header file

```cpp
#ifndef ADAM_MINI_H
#define ADAM_MINI_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <string>

// Adam-mini: "Adam-mini: Use Fewer Learning Rates To Gain More"
// Zhang et al. 2024 (https://arxiv.org/abs/2406.16793)
//
// Memory-efficient Adam variant. For each parameter:
//   - 2-D weight matrices: block-wise second moment reduced to ONE value
//     per row (mean of g² along columns). State size = rows + rows*cols
//     instead of 2*rows*cols.
//   - 1-D parameters (biases, norms): full per-element v (same as Adam).
//   - Configurable block-mode override.
//
// Per-parameter algorithm (2-D ROW_MEAN, the common case):
//   for each row i:
//     vmean_t[i] = β2 * vmean_{t-1}[i] + (1 - β2) * (1/cols) * Σ_j g_t[i,j]²
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t                              (full m)
//   m̂_t = m_t / (1 - β1^t)                                          (bias correction)
//   v̂_t = vmean_t / (1 - β2^t)                                       (broadcast to rows)
//   update[i,j] = m̂_t[i,j] / (√v̂_t[i] + ε)
//   if wd > 0: θ *= (1 - lr * wd)                                   (decoupled decay)
//   θ -= lr * update
//
// State per parameter: 2 tensors (m full-shape, vmean reduced).
// For 2-D (R, C): m is (R, C), vmean is (R, 1). Saves ~50% vs Adam.
// For 1-D:        m is (1, C), vmean is (1, C). Same as Adam (no savings needed).
class AdamMini : public Optimizer {
public:
    enum class BlockMode {
        AUTO,       // shape-based: 2-D → ROW_MEAN, 1-D → FULL
        FULL,       // per-element v (regular Adam)
        ROW_MEAN,   // vmean per row (Adam-mini main case)
        SCALAR      // one global vmean for the whole parameter
    };

    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;
    BlockMode default_mode;

    explicit AdamMini(double lr_ = 1e-3,
                      double b1 = 0.9,
                      double b2 = 0.999,
                      double eps = 1e-8,
                      double wd = 0.0,
                      BlockMode mode = BlockMode::AUTO);

    // Validated setters
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // State accessors
    bool has_state(void* layer_ptr) const;
    size_t num_params_with_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_vmean(void* layer_ptr, size_t param_idx) const;
    BlockMode get_block_mode(void* layer_ptr, size_t param_idx) const;

    // Override block-mode for a specific parameter (e.g. to force SCALAR for one specific param).
    void set_param_block_mode(void* layer_ptr, size_t param_idx, BlockMode mode);

private:
    struct ParamState {
        Tensor m;       // first moment, full shape
        Tensor vmean;   // second moment, reduced shape (full for FULL mode)
        BlockMode mode;
    };
    std::map<void*, std::vector<ParamState>> state_;
    std::map<void*, std::vector<BlockMode>> overrides_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    BlockMode resolve_mode(void* layer_ptr, size_t param_idx, const Tensor& p) const;
};

#endif
```

## Implementation file

```cpp
#include "adam_mini.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

AdamMini::AdamMini(double lr_, double b1, double b2, double eps, double wd, BlockMode mode)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      weight_decay(wd), t(1), default_mode(mode) {
    set_lr(lr_);
    set_beta1(b1);
    set_beta2(b2);
    set_epsilon(eps);
    set_weight_decay(wd);
}

void AdamMini::set_lr(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("AdamMini: lr must be >= 0");
    lr = v;
    Optimizer::lr = v;
}
void AdamMini::set_beta1(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("AdamMini: beta1 in [0,1)");
    beta1 = v;
}
void AdamMini::set_beta2(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("AdamMini: beta2 in [0,1)");
    beta2 = v;
}
void AdamMini::set_epsilon(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("AdamMini: epsilon must be > 0");
    epsilon = v;
}
void AdamMini::set_weight_decay(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("AdamMini: weight_decay must be >= 0");
    weight_decay = v;
}

AdamMini::BlockMode AdamMini::resolve_mode(void* layer_ptr, size_t param_idx, const Tensor& p) const {
    auto oit = overrides_.find(layer_ptr);
    if (oit != overrides_.end() && param_idx < oit->second.size()) {
        BlockMode m = oit->second[param_idx];
        if (m != BlockMode::AUTO) return m;
    }
    if (default_mode != BlockMode::AUTO) return default_mode;
    // AUTO: 1-D → FULL, 2-D → ROW_MEAN
    return (p.rows == 1 || p.cols == 1) ? BlockMode::FULL : BlockMode::ROW_MEAN;
}

void AdamMini::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<ParamState> vec;
    vec.reserve(params.size());
    for (auto* p : params) {
        ParamState st;
        st.m = Tensor(p->rows, p->cols);
        st.m.fill(0.0);
        BlockMode mode = resolve_mode(layer_ptr, vec.size(), *p);
        st.mode = mode;
        size_t v_rows = p->rows;
        size_t v_cols = p->cols;
        switch (mode) {
            case BlockMode::FULL:
                // vmean = full (same shape as param)
                break;
            case BlockMode::ROW_MEAN:
                v_cols = 1;  // one value per row
                break;
            case BlockMode::SCALAR:
                v_rows = 1;
                v_cols = 1;
                break;
            default:
                throw std::logic_error("AdamMini: unresolved mode");
        }
        st.vmean = Tensor(v_rows, v_cols);
        st.vmean.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

void AdamMini::step(Model& model) {
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);

    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::logic_error("AdamMini: param/grad count mismatch");
        }
        ensure_state(layer_ptr, params);
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            if (grad->rows != param->rows || grad->cols != param->cols) {
                throw std::logic_error("AdamMini: param/grad shape mismatch");
            }
            ParamState& st = st_vec[i];

            // Update m: m = β1*m + (1-β1)*g (full shape)
            for (size_t r = 0; r < grad->rows; ++r)
                for (size_t c = 0; c < grad->cols; ++c)
                    st.m[r][c] = beta1 * st.m[r][c] + (1.0 - beta1) * (*grad)[r][c];

            // Update vmean depending on mode
            switch (st.mode) {
                case BlockMode::FULL: {
                    // vmean same shape as param
                    for (size_t r = 0; r < grad->rows; ++r)
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            st.vmean[r][c] = beta2 * st.vmean[r][c] + (1.0 - beta2) * g * g;
                        }
                    break;
                }
                case BlockMode::ROW_MEAN: {
                    // vmean shape (rows, 1): one value per row
                    for (size_t r = 0; r < grad->rows; ++r) {
                        double s = 0.0;
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            s += g * g;
                        }
                        double mean_gg = s / static_cast<double>(grad->cols);
                        st.vmean[r][0] = beta2 * st.vmean[r][0] + (1.0 - beta2) * mean_gg;
                    }
                    break;
                }
                case BlockMode::SCALAR: {
                    // vmean shape (1, 1): one global value
                    double s = 0.0;
                    size_t n = grad->rows * grad->cols;
                    for (size_t r = 0; r < grad->rows; ++r)
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            s += g * g;
                        }
                    double mean_gg = s / static_cast<double>(n);
                    st.vmean[0][0] = beta2 * st.vmean[0][0] + (1.0 - beta2) * mean_gg;
                    break;
                }
                default:
                    throw std::logic_error("AdamMini: unknown mode");
            }

            // Decoupled weight decay
            if (weight_decay > 0.0) {
                double scale = 1.0 - lr * weight_decay;
                for (size_t r = 0; r < param->rows; ++r)
                    for (size_t c = 0; c < param->cols; ++c)
                        (*param)[r][c] *= scale;
            }

            // Compute update
            double sqrt_bc2 = std::sqrt(bc2);
            switch (st.mode) {
                case BlockMode::FULL: {
                    for (size_t r = 0; r < param->rows; ++r)
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double v_hat = st.vmean[r][c] / bc2;
                            double denom = std::sqrt(v_hat) + epsilon;
                            double upd = m_hat / denom;
                            (*param)[r][c] -= lr * upd;
                        }
                    break;
                }
                case BlockMode::ROW_MEAN: {
                    for (size_t r = 0; r < param->rows; ++r) {
                        double v_hat_r = st.vmean[r][0] / bc2;
                        double denom_r = std::sqrt(v_hat_r) + epsilon;
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double upd = m_hat / denom_r;
                            (*param)[r][c] -= lr * upd;
                        }
                    }
                    break;
                }
                case BlockMode::SCALAR: {
                    double v_hat = st.vmean[0][0] / bc2;
                    double denom = std::sqrt(v_hat) + epsilon;
                    for (size_t r = 0; r < param->rows; ++r)
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double upd = m_hat / denom;
                            (*param)[r][c] -= lr * upd;
                        }
                    break;
                }
            }
        }

        layer->zero_grad();
    }

    ++t;
}

bool AdamMini::has_state(void* layer_ptr) const {
    return state_.find(layer_ptr) != state_.end();
}

size_t AdamMini::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    return it == state_.end() ? 0 : it->second.size();
}

const Tensor& AdamMini::get_m(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).m;
}

const Tensor& AdamMini::get_vmean(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).vmean;
}

AdamMini::BlockMode AdamMini::get_block_mode(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).mode;
}

void AdamMini::set_param_block_mode(void* layer_ptr, size_t param_idx, BlockMode mode) {
    if (state_.find(layer_ptr) != state_.end() &&
        param_idx < state_[layer_ptr].size()) {
        // state already exists — drop it; next step will reinitialize
        // (or, simpler: just update mode and resize vmean on next step)
        state_[layer_ptr][param_idx].mode = mode;
    }
    auto oit = overrides_.find(layer_ptr);
    if (oit == overrides_.end()) {
        // we'll let ensure_state create this on next step
        overrides_[layer_ptr];  // create empty vector
        oit = overrides_.find(layer_ptr);
    }
    if (param_idx >= oit->second.size()) oit->second.resize(param_idx + 1, BlockMode::AUTO);
    oit->second[param_idx] = mode;
}
```

## Test file (highlights)

`tests/test_adam_mini.cpp` will include ~25 tests covering all the cases listed in the task list. Each test will print `[PASS] <name>` on success. The expected total is **~25-30 focused checks**.

## Verification

After implementation:
1. `make tests` — should compile `build/test_adam_mini` alongside all existing binaries.
2. `./build/test_adam_mini` — should print `=== Summary: N passed, 0 failed ===` with N ≥ 25.
3. `make run_tests` — should still pass all suites except the deferred ones in `NOT_FIXED.md`.
4. Mutation testing (Task 20) — each of the 4 mutations should produce ≥ 1 test failure.

## Commit

Single commit after all tests pass:
```bash
git add include/nn/optimizers/adam_mini.{h,cpp} \
        include/nn/nn.h \
        tests/test_adam_mini.cpp \
        Makefile \
        EXPANSION_QUEUE.md \
        docs/plans/2026-07-25-adam-mini-optimizer.md
git commit -m "feat(optimizers): add Adam-mini (Zhang 2024), block-wise second moment

Implements Adam-mini (https://arxiv.org/abs/2406.16793) — memory-efficient
Adam variant using block-wise second-moment reduction. For 2-D parameters,
v is reduced to one value per row (mean of g² along columns), saving ~50%
of optimizer state memory without performance loss.

Three block modes supported: FULL (per-element v, Adam-equivalent), ROW_MEAN
(one v per row — the paper's main case for 2-D weights), and SCALAR (one
global v). Default policy: AUTO → 2-D uses ROW_MEAN, 1-D uses FULL. State
per parameter is two tensors (m full + vmean reduced), exactly half of Adam
for 2-D params. Decoupled AdamW-style weight decay. Bias correction,
per-parameter block-mode override, parameter/gradient count & shape guards,
determinism, and signature tests vs AdamW included.

N focused checks pass at machine precision. Mutation-tested."
git push origin master
```
