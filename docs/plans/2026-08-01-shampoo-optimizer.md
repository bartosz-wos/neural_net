# Shampoo Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a Shampoo (Gupta, Koren, Singer 2018, https://arxiv.org/abs/1802.03668) optimizer to the repo — the foundational Kronecker-factored preconditioned SGD that SOAP and PSGD build on.

**Architecture:** Preconditioned SGD that maintains two symmetric covariance matrices per 2-D parameter, L_t = β·L_{t-1} + (1-β)·G·G^T (m×m) and R_t = β·R_{t-1} + (1-β)·G^T·G (n×n), and applies the preconditioner in the update as `W := W - lr · L^{-1/4} · G · R^{-1/4}`. Falls back to plain SGD for 1-D parameters (only one of L/R is defined) and pure scalar (m=n=1).

**Tech Stack:** C++17, Tensor class (already in `core/tensor.h`), reuse the cyclic Jacobi eigendecomposition pattern from `soap.cpp` for `L^{-1/4}` and `R^{-1/4}` computation (apply as diagonal of eigenvalues to the 1/4 power after rotating).

**Location:** `include/nn/optimizers/shampoo.{h,cpp}` + `tests/test_shampoo.cpp` + Makefile + umbrella header.

---

## Algorithm reference (Gupta et al. 2018, §3, Algorithm 1)

Per-parameter update for W ∈ R^{m×n}, gradient G:
1. L_t = β · L_{t-1} + (1-β) · G · G^T        (m×m symmetric)
2. R_t = β · R_{t-1} + (1-β) · G^T · G        (n×n symmetric)
3. U_L, λ_L = eigh(L_t)        (orthogonal U_L, eigenvalues λ_L ≥ 0)
4. U_R, λ_R = eigh(R_t)
5. Compute L^{-1/4} = U_L · diag(λ_L^{-1/4}) · U_L^T
   Compute R^{-1/4} = U_R · diag(λ_R^{-1/4}) · U_R^T
6. update = LR^(1/4) preconditioner applied to G:  update = L^{-1/4} · G · R^{-1/4}
7. W := W - lr · update                  (with optional decoupled weight decay)

**Edge cases:**
- m == 1 OR n == 1: only one of L/R is defined; skip its eigendecomp (use identity rotation).
- m == 1 AND n == 1: scalar parameter, plain SGD.
- λ_L[k] = 0 or λ_R[k] → 0: clip to ε when computing λ^{-1/4} (avoids division by zero).

**State per parameter:**
- L (m×m), R (n×n) — symmetric covariance EMA
- U_L (m×m), U_R (n×n) — cached eigenbases (lazy after first step)
- (no moments — Shampoo is NOT an Adam-style optimizer)

**Defaults (Gupta et al. 2018 §5.1):**
- lr = 1e-3, β = 0.9, weight_decay = 0, eps = 1e-12

---

## Task 1: Header file `include/nn/optimizers/shampoo.h`

Objective: Declare the Shampoo optimizer class with all accessors, type, and helper signatures.

Files:
- Create: `include/nn/optimizers/shampoo.h`

Steps:
1. Write the header file.
2. Verify it compiles with `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/optimizers/shampoo.h"'`

---

## Task 2: First failing test (TDD RED)

Objective: Write tests that fail because the implementation doesn't exist yet.

Files:
- Create: `tests/test_shampoo.cpp`

Tests to write:
1. Default constructor stores expected defaults (lr=1e-3, beta=0.9, weight_decay=0, eps=1e-12, t=1)
2. Non-default constructor stores the values
3. Validated setters throw on bad inputs (lr<0, beta ∉ [0,1), eps≤0, weight_decay<0)
4. Constructor-time validation throws on bad inputs
5. State is lazy (no state before step)
6. State shape correctness (param (m,n) → L (m,m), R (n,n), U_L (m,m), U_R (n,n))
7. State doesn't accumulate across layers
8. `handles_weight_decay() == true`
9. Scalar parameter (1,1) reduces to plain SGD (no preconditioner)
10. 1-D parameter (1, n) reduces to preconditioned GD with only R
11. 1-D parameter (m, 1) reduces to preconditioned GD with only L
12. First-step closed-form on a small 2-D problem (verified analytically)
13. Rotation preserves Frobenius norm of the gradient
14. Eigendecomp on identity matrix returns identity (Q=I, λ=1)
15. Weight decay shrinks params at zero gradient
16. Training reduces loss on y=2x regression (5+ epochs)
17. Determinism (two fresh instances produce bit-identical updates over 5 steps)
18. Default vs SOAP signature (Shampoo update differs from SOAP update because no Adam)
19. Equality of L^{-1/4} · G · R^{-1/4} construction vs full eigh approach

---

## Task 3: Implementation (TDD GREEN)

Objective: Write the implementation to make all tests pass.

Files:
- Create: `include/nn/optimizers/shampoo.cpp`

Steps:
1. Implement `validate()` matching SOAP's pattern
2. Implement constructor + setters
3. Implement `step()` iterating over layers (mirror SOAP)
4. Implement `ensure_state()` (lazy init of L, R, U_L, U_R to zero; U_L into identity)
5. Implement `update_param()`:
   - Scalar (1,1): plain SGD (with weight decay)
   - 1-D (1, n): only R, update = lr · G · R^{-1/4}
   - 1-D (m, 1): only L, update = lr · L^{-1/4} · G
   - 2-D (m, n): update L, R, eigh both, form L^{-1/4} G R^{-1/4}, apply
   - Optional decoupled weight decay
6. Implement Jacobi eigendecomposition helper (copy from SOAP)
7. Implement `L^(-1/4)` / `R^(-1/4)` via diagonal of eigenvalues ^ (-1/4) in rotated basis

---

## Task 4: Makefile + umbrella header

Objective: Wire up the test target.

Files:
- Modify: `Makefile` — add `build/test_shampoo` rule, add to `tests:` deps, add `echo && ./build/test_shampoo` line
- Modify: `include/nn/nn.h` — add `#include "nn/optimizers/shampoo.h"`

---

## Task 5: Final verification

1. Run `make tests` to verify all registered tests still compile.
2. Run `make run_tests` to verify all registered tests pass.
3. Verify the new test file alone passes.
4. Commit and push.

---

## Out of scope (not in this iteration)

- Block-diagonal Shampoo (multiple smaller preconditioners per layer)
- Shampoo with momentum on the preconditioned gradient (with separate momentum V)
- Distributed/multi-GPU Shampoo
- Inverse-root via Newton iteration / series (we use the eigh approach to keep the code modular with the existing Jacobi eigendecomposition)
- Adaptive root / non-1/4 exponent (the "Generalized Shampoo" generalization)
