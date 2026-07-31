# SOAP Optimizer — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a `SOAP` (ShampoO with Adam in the Preconditioner's eigenbasis) optimizer following Vyas et al. 2024, "SOAP: Improving and Stabilizing Shampoo using Adam" (https://arxiv.org/abs/2409.11321, NeurIPS 2024). SOAP applies Adam's diagonal preconditioner in the eigenbasis of Shampoo's left/right Kronecker-factored preconditioners, giving the convergence benefits of Shampoo with the stability of Adam. Used in Llama 3 pretraining recipes.

**Architecture:** Per-parameter state consisting of two symmetric "covariance" matrices `L ∈ R^{m×m}` and `R ∈ R^{n×n}` (where the gradient has shape `(m, n)`), plus two Adam-style first/second moment tensors `M` and `V` in the eigenbasis. Every `precondition_frequency` steps (default 10), eigendecompose `L = Q_L · diag(λ_L) · Q_L^T` and `R = Q_R · diag(λ_R) · Q_R^T` via the Jacobi method. Between eigendecompositions, the algorithm:
1. Rotate the gradient into the eigenbasis: `g_rot = Q_L^T · g · Q_R`
2. Update Adam moments in the rotated space: `M = β1·M + (1-β1)·g_rot`, `V = β2·V + (1-β2)·g_rot²`
3. Apply bias-corrected Adam update in rotated space: `update_rot = lr · (M/(1-β1^t)) / (sqrt(V/(1-β2^t)) + ε)`
4. Rotate the update back: `update = Q_L · update_rot · Q_R^T`
5. `param -= update` (with optional decoupled weight decay)

**Tech Stack:** C++17, the existing `Optimizer` / `Model` / `Tensor` / `Layer` core. New: a private Jacobi-rotation symmetric eigendecomposition helper (used internally by SOAP — not part of the public API).

---

## Reference math (verified against Vyas et al. 2024 §3, Algorithm 1)

For a weight matrix `W ∈ R^{m×n}` (gradient `G ∈ R^{m×n}`):

```
Per step t:
    if t mod precondition_frequency == 1 (or t==1):
        L_t = β2 · L_{t-1} + (1-β2) · G · G^T         # (m × m) symmetric
        R_t = β2 · R_{t-1} + (1-β2) · G^T · G         # (n × n) symmetric
        Q_L, λ_L = eigh(L_t)
        Q_R, λ_R = eigh(R_t)
    else:
        Q_L, λ_L, Q_R, λ_R cached from last preconditioner step

    G_rot = Q_L^T · G · Q_R                            # rotate to eigenbasis
    M_t   = β1 · M_{t-1} + (1-β1) · G_rot
    V_t   = β2 · V_{t-1} + (1-β2) · G_rot ⊙ G_rot
    M̂_t  = M_t / (1 - β1^t)
    V̂_t  = V_t / (1 - β2^t)
    update_rot = lr · M̂_t / (sqrt(V̂_t) + ε)            # Adam-like in rotated basis
    update     = Q_L · update_rot · Q_R^T               # rotate back
    W_t       := W_t - update
```

Paper defaults:
- `lr = 3e-3` (paper §5.2, used in Llama-3-scale experiments)
- `β1 = 0.95` (paper)
- `β2 = 0.95` (paper) — NOTE: β2 controls BOTH second-moment decay AND the L/R exponential moving averages (paper §3.3 — same β2 used for both)
- `ε = 1e-8`
- `precondition_frequency = 10` (paper §3)
- `weight_decay = 0.0`

Notes:
- The L/R matrices use the SAME β2 as the Adam second moment (V).
- The first preconditioner update happens at t=1 (with the unrotated basis Q_L = I, Q_R = I and L = G·G^T, R = G^T·G).
- For 1-D parameters (shapes `(1, c)` or `(r, 1)`), one of the preconditioner factors is 1×1 and we fall back to per-element second-moment Adam (no eigendecomp needed for the 1-dim factor).
- For scalar (1,1) parameters, we use plain Adam.

State per parameter:
- `L ∈ R^{m×m}` (symmetric, `L_0 = 0`)
- `R ∈ R^{n×n}` (symmetric, `R_0 = 0`)
- `Q_L ∈ R^{m×m}` (orthogonal eigenbasis of L — cached)
- `Q_R ∈ R^{n×n}` (orthogonal eigenbasis of R — cached)
- `M ∈ R^{m×n}` (Adam first moment in rotated space)
- `V ∈ R^{m×n}` (Adam second moment in rotated space)
- `t` (step counter, layer-shared)

Edge cases:
- `m=1` or `n=1`: skip eigendecomposition for the 1-D side, use identity.
- `m=1 AND n=1`: scalar — plain Adam.
- Negative lr / β1 / β2 ∉ [0,1) / ε ≤ 0 / precondition_frequency < 1 / weight_decay < 0 → throw `std::invalid_argument`.

---

## Files to touch

| Operation | Path |
|-----------|------|
| Create | `include/nn/optimizers/soap.h` |
| Create | `include/nn/optimizers/soap.cpp` |
| Modify | `include/nn/nn.h` (add umbrella include) |
| Modify | `Makefile` (build rule, tests dep, run_tests echo) |
| Create | `tests/test_soap.cpp` |
| Create | `docs/plans/2026-07-31-soap-optimizer.md` (this file) |

---

## Task 1: Write the failing test scaffold

**Objective:** Compile a test file that exercises the SOAP optimizer and proves the implementation is missing.

**Files:**
- Create: `tests/test_soap.cpp`

**Step 1:** Write the test file outline that:
- Includes `nn/optimizers/soap.h`
- Constructs a `SOAP` with default hyperparameters
- Asserts defaults (`lr=3e-3, β1=0.95, β2=0.95, ε=1e-8, precondition_frequency=10, weight_decay=0, t=1, handles_weight_decay=true`)
- Asserts non-default constructor + validated setters throw on bad inputs (negative lr, β1/β2 ∉ [0,1), ε ≤ 0, precondition_frequency < 1, weight_decay < 0)
- Asserts step counter increments
- Asserts lazy state initialization (has_state before/after step)
- Asserts state shapes for L, R, M, V match param shape (L: m×m, R: n×n, M, V: m×n)
- Asserts preconditioner update happens at t=1 and at t=11 (every 10 steps)

The test file should fail to compile (no `nn/optimizers/soap.h` exists yet).

**Step 2:** Run `make build/test_soap` — expected: header not found.

---

## Task 2: Create the SOAP header

**Objective:** Declare the `SOAP` class.

**Files:**
- Create: `include/nn/optimizers/soap.h`

**Step 1:** Declare the class with:
- Constructor `SOAP(double lr=3e-3, double beta1=0.95, double beta2=0.95, double epsilon=1e-8, int precondition_frequency=10, double weight_decay=0.0)`
- Validated setters (`set_lr`, `set_beta1`, `set_beta2`, `set_epsilon`, `set_precondition_frequency`, `set_weight_decay`)
- Accessors (`get_lr`, `get_beta1`, `get_beta2`, `get_epsilon`, `get_precondition_frequency`, `get_weight_decay`, `get_t`, `has_state`, `get_L`, `get_R`, `get_Q_L`, `get_Q_R`, `get_M`, `get_V`)
- `step(Model&)` override
- `handles_weight_decay() returns true`
- Private state: `std::map<void*, std::vector<ParameterState>>`
- Private helper: `jacobi_eigendecompose(Tensor& A, Tensor& Q, Tensor& eigvals)` — symmetric eigendecomp via Jacobi rotations

**Step 2:** Compile the header in isolation:
```bash
g++ -std=c++17 -Iinclude -fsyntax-only -x c++ - <<< '#include "nn/optimizers/soap.h"'
```
Expected: clean compile.

---

## Task 3: Implement the SOAP class — step 1 (state init + preconditioner rotation)

**Objective:** Write the algorithm body for one step, focusing on getting state init + eigendecomposition correct.

**Files:**
- Modify: `include/nn/optimizers/soap.cpp`

**Step 1:** Implement:
- `ensure_state()`: lazy-init L, R (zero matrices of shape m×m, n×n), Q_L (identity m×m), Q_R (identity n×n), M, V (zero matrices of shape m×n).
- `should_precondition(t)`: returns `true` on the first step (t==1) AND every `precondition_frequency` steps thereafter.
- The Jacobi eigendecomposition helper (Jacobi rotations for symmetric matrices, max 100 iterations, eps=1e-10).
- The preconditioner update step (when `should_precondition(t)`): accumulate L, R; eigendecompose; cache Q_L, Q_R.
- The gradient rotation `G_rot = Q_L^T · G · Q_R` — three matmuls.

**Step 2:** Verify the test file compiles:
```bash
make build/test_soap
```
Expected: success (or expected to fail on missing assertion in test).

---

## Task 4: Implement the SOAP class — step 2 (Adam update + rotation back)

**Objective:** Complete the algorithm with the Adam update in the rotated space + rotation back to parameter space.

**Files:**
- Modify: `include/nn/optimizers/soap.cpp`

**Step 1:** Implement:
- Adam moments in rotated space: `M = β1·M + (1-β1)·G_rot`, `V = β2·V + (1-β2)·G_rot²`
- Bias correction: `M̂ = M/(1-β1^t)`, `V̂ = V/(1-β2^t)`
- `update_rot = lr · M̂ / (sqrt(V̂) + ε)`
- Rotation back: `update = Q_L · update_rot · Q_R^T` (note: `Q_R^T` not `Q_R` because the rotation was `Q_R^T · G` on the right)
- Weight decay (decoupled): `param *= (1 - lr·wd)`
- `param -= update`

**Step 2:** Verify rotation-back matches by checking that for orthogonal Q_L and Q_R, `(Q_L · update_rot · Q_R^T)^T = Q_R · update_rot^T · Q_L^T` and the squared norm is preserved (`||update|| = ||update_rot||` for orthogonal Q).

---

## Task 5: Handle edge cases (1-D and scalar parameters)

**Objective:** Make SOAP work on bias (1, C) and scalar (1, 1) parameters without crashing.

**Files:**
- Modify: `include/nn/optimizers/soap.cpp`

**Step 1:** Implement:
- If `m == 1` AND `n == 1`: plain Adam (skip preconditioner).
- If `m == 1`: skip L eigendecomp (L is 1×1), use Q_L = [[1]]; identity rotation.
- If `n == 1`: skip R eigendecomp, use Q_R = [[1]].
- Validate precondition_frequency >= 1 (else throw).

**Step 2:** Add a test that runs SOAP on a model with bias parameters and verifies the bias is updated correctly.

---

## Task 6: Write the focused test suite

**Objective:** Verify all SOAP behaviors.

**Files:**
- Modify: `tests/test_soap.cpp`

**Step 1:** Add tests covering:
1. Defaults round-trip (lr=3e-3, β1=0.95, β2=0.95, ε=1e-8, precondition_frequency=10, weight_decay=0)
2. Non-default constructor
3. Validated setters throw on negative lr, β1/β2 ∉ [0,1), ε ≤ 0, precondition_frequency < 1, weight_decay < 0
4. Mutator validation
5. Lazy state init (has_state before/after step)
6. State shape correctness for L (m×m), R (n×n), M (m×n), V (m×n) on Dense(3,4) → weights (4,3)
7. State shape for bias (1, C) → L (1,1), R (C, C), M (1, C), V (1, C)
8. Step counter increments 1 → 2 → 3 → ...
9. **Closed-form first step** on Dense(2,2) with zero-init + grad=G: L = β2·0 + (1-β2)·G·G^T = (1-β2)·G·G^T (where G·G^T is symmetric 2×2); eigendecompose analytically; rotate; apply Adam in rotated space; rotate back; param_1 = -lr · ... — verify the closed form matches.
10. **Preconditioner frequency**: with precondition_frequency=3, the Q_L/Q_R matrices are recomputed at t=1, t=4, t=7 (and stay cached between).
11. **Gradient rotation preserves orthogonality**: for orthogonal Q_L, Q_R, the update_rot norm == update norm (Frobenius).
12. **Decoupled weight decay** shrinks param at zero grad.
13. **Training reduces loss** on a simple regression task (≥30% loss reduction in 60 steps).
14. **Determinism** (two fresh SOAP instances produce bit-identical params over 5 steps).
15. **Scalar parameter (1, 1)** uses plain Adam path.

**Step 2:** Run `make build/test_soap && ./build/test_soap` — expected: all tests pass.

---

## Task 7: Wire SOAP into the build and umbrella header

**Objective:** Make `SOAP` available via `nn/nn.h` and runnable via `make tests` / `make run_tests`.

**Files:**
- Modify: `include/nn/nn.h` — add `#include "nn/optimizers/soap.h"`
- Modify: `Makefile`:
  - Add `$(BUILD_DIR)/test_soap: $(LIB_OBJS) $(BUILD_DIR)/test_soap.o`
  - Add `$(BUILD_DIR)/test_soap.o: tests/test_soap.cpp include/nn/optimizers/soap.h`
  - Add `test_soap` to the `tests:` deps line
  - Add `=== Running SOAP Tests ===` echo in `run_tests`

**Step 1:** Verify `make tests` compiles all targets.

**Step 2:** Verify `make run_tests` runs SOAP.

---

## Task 8: Verify full suite

**Objective:** Confirm no regressions.

**Step 1:** Run `make run_tests` — expected: SOAP passes, no regressions.

**Step 2:** If any test fails, fix the SOAP implementation (NOT the other tests).

---

## Risk: complexity of Jacobi eigendecomposition

The Jacobi method is iterative and simple to implement correctly. Worst case for a (m, m) symmetric matrix is ~50 Jacobi sweeps × ~m²(m²-1)/4 rotations = ~50·m⁴/4. For typical Dense layers (m ≤ 1024), this is ~6.5e9 rotations per preconditioner step — expensive. SOAP triggers this every 10 steps, so the amortized cost is ~6.5e8 rotations per normal step.

This is fine for our scope — modern Llama-3-scale SOAP implementations use the same Jacobi method on the CPU. For the test suite, we'll use small matrices (m, n ≤ 8) where the cost is negligible.

---

## Verification

After implementation, `make run_tests` should show:
- All existing tests still pass (no regressions)
- New SOAP test suite passes all checks

---

## Estimated scope

- ~250-350 lines of implementation (incl. the Jacobi eigendecomposition helper, ~80 lines)
- ~200-300 lines of test code
- 1 file pair (header + impl), 1 test file
- No external dependencies (Jacobi method is self-contained)