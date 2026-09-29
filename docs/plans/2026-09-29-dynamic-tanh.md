# Dynamic Tanh (DyT) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a `DynamicTanh` (DyT) normalization-replacement layer (Zhu et al., CVPR 2025, arXiv:2503.10622) to `include/nn/layers/normalization/`. The layer is `out = γ * tanh(α * x) + β` with three learnable parameters — a global scalar `α` and per-channel vectors `γ`, `β`.

**Architecture:** Drop-in replacement for `LayerNorm` / `RMSNorm` with the canonical Layer interface (`forward` / `backward` / `update_weights` / `parameters` / `gradients` / `zero_grad`). Element-wise forward (no batch-statistic computation). Full analytical backward including the parameter gradients for all three learnables (α, γ, β), verified against finite differences at machine precision.

**Tech Stack:** C++17, header-only style with `dynamic_tanh.h` / `dynamic_tanh.cpp` matching the existing normalization/ files. Uses the `Tensor` API (`fill`, `zeros`, indexing).

---

## Paper reference

Zhu, Chen, He — "Transformers without Normalization" (CVPR 2025; arXiv:2503.10622). Algorithm 1 (PyTorch-like pseudocode in §4):

```python
def __init__(self, C, init_α):
    self.α = Parameter(ones(1) * init_α)
    self.γ = ones(C)
    self.β = zeros(C)
def forward(x):
    x = tanh(self.α * x)
    return self.γ * x + self.β
```

Initialisation per §7.1 (non-LLM): `α₀ = 0.5` default; `γ = 1`; `β = 0`. These match LayerNorm / RMSNorm defaults and ship as the constructor defaults.

## Design choices

- **Shape**: `(rows = batch_or_N, cols = features)` to match `LayerNorm` / `RMSNorm` in this repo. The paper uses `[B, T, C]` but flattens to `[N, C]` for the per-channel affine.
- **α storage**: `Tensor alpha(1, 1)` — a single learnable scalar parameter. The paper uses `Parameter(ones(1) * init_α)` which is shape `[1]`; we treat it as `(1, 1)` to match the `(1, features)`-style per-channel convention used for `gamma` / `beta`.
- **Caching**: `last_input` is the only cache needed (the post-tanh tensor is `tanh(alpha * input)` and can be reconstructed cheaply, but caching `last_tanh` lets us avoid recomputing `tanh` in `backward`).
- **Accumulation**: `grad_alpha_`, `grad_gamma_`, `grad_beta_` accumulate across backward calls until `zero_grad()` — matches the `LayerNorm` convention exactly.
- **No statistics**: unlike `LayerNorm`/`RMSNorm`, no `eps`, no `training` flag, no batch reductions. The interface still exposes `set_training(bool)` for parity with the rest of the family but it's a no-op.

## Tests planned (test_dynamic_tanh.cpp)

1. **Constructor validation** — `features = 0` throws; valid constructions don't throw; α defaults to 0.5; γ is all-ones; β is all-zeros.
2. **Accessors** — `name()`, `features()` (new accessor since `RMSNorm` has none).
3. **Forward shape + finiteness** — `(4, 8)` input produces `(4, 8)` output, all finite.
4. **Forward init-state** — with `α=0.5, γ=1, β=0` and a large positive input (e.g. `x=10`), `out ≈ tanh(5) ≈ 1.0` (squashing extreme values, the DyT headline).
5. **Forward squashing** — `x=100` produces `out ≈ γ` (saturates at ±γ since `tanh(50) ≈ 1`).
6. **Forward γ = 1, β = 0** simplifies to `tanh(α * x)` exactly (verified by comparing against the closed-form `tanh(0.5 * x)`).
7. **Forward γ scales, β shifts** — verify with `α=1.0`, `γ=2.0`, `β=0.5`: `out = 2 * tanh(x) + 0.5`.
8. **Forward determinism** — two consecutive calls bit-exact (no RNG in forward).
9. **Forward preserves per-row structure** — different `α` values change output magnitudes (verifies α is wired).
10. **parameters()/gradients() contract** — three tensors of shapes `(1,1)`, `(1, features)`, `(1, features)`.
12. **grad_x FD-check** — `rel_err < 1e-5` for random init (4, 8) input.
13. **grad_gamma FD-check** — `rel_err < 1e-5`.
14. **grad_beta FD-check** — `rel_err < 1e-5` and equals `Σ_b grad_output[b][f]`.
15. **grad_alpha FD-check** — `rel_err < 1e-5` (the headline: the scalar learnable parameter has a correct gradient).
16. **zero_grad()** — clears all three parameter gradients.
17. **update_weights()** — moves α, γ, β by `-lr * grad`.
18. **End-to-end training reduces loss** — small MLP with DyT between Dense layers on y=2x regression, loss falls >50% in 30 SGD steps (proves the gradient chain through DyT into a downstream loss works end-to-end).
19. **Mutation test 1: drop γ from forward** (`out = tanh(αx) + β`) → grad_gamma FD test fails (analytical = sum_b(grad_output·tanh), numerical = 0).
20. **Mutation test 2: drop α from tanh argument** (`tanh(x)` instead of `tanh(α·x)`) → grad_x FD test fails and grad_alpha test fails (analytical gradient ≠ numerical).
21. **Mutation test 3: missing `(1 - t²)` factor** → grad_x grad always wrong sign for large inputs.

## Tasks

### Task 1: Header + minimal constructor

Create `include/nn/layers/normalization/dynamic_tanh.h` with the full `DynamicTanh` class declaration matching `LayerNorm` / `RMSNorm` style.

### Task 2: TDD RED — Constructor + sanity tests

Add `tests/test_dynamic_tanh.cpp` with Tests 1–3 only. Run `make build/test_dynamic_tanh` — expect link failure (no `.cpp` yet).

### Task 3: Minimal impl

Create `include/nn/layers/normalization/dynamic_tanh.cpp` with empty-stub `forward`/`backward`/`update_weights` that satisfy the test stubs (constructor + forward finiteness). Run the test, expect Tests 1-3 pass.

### Task 4: Forward math

Add the full `out = γ * tanh(α * x) + β` implementation. Add Tests 4–9. Verify each test individually as it's added.

### Task 5: Backward + parameter gradients

Implement full analytical backward (grad_x, grad_gamma, grad_beta, grad_alpha). Add Tests 10–17. Mutation-test.

### Task 6: End-to-end training test

Add Test 18.

### Task 7: Register in umbrella + Makefile

Add `#include "layers/normalization/dynamic_tanh.h"` to `include/nn/nn.h` (alphabetical, between `coord_attention.h` and `eca.h`); add build rule + test echo + test deps line to `Makefile`.

### Task 8: Final sweep

Run the focused suite 3× deterministically. Mutation-test with all three mutations. No regressions in `test_rmsnorm`, `test_layer_norm`, or any other normalization test.

---

## Files to touch

- Create: `include/nn/layers/normalization/dynamic_tanh.h`
- Create: `include/nn/layers/normalization/dynamic_tanh.cpp`
- Create: `tests/test_dynamic_tanh.cpp`
- Modify: `include/nn/nn.h` (1 line)
- Modify: `Makefile` (3 lines: build rule, tests: deps, run_tests echo)
- Create: `docs/plans/2026-09-29-dynamic-tanh.md` (this file)