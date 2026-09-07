# nGPT — Normalized Transformer on the Hypersphere: Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `HypersphereLinear` (row-L2-normalized weight matrix with the exact
normalization Jacobian in the backward) and `NGPTBlock` (normalized-LERP residual
updates with learnable per-dimension *eigen learning rates*), plus `NGPTModel`.

**Paper:** Loshchilov, Hsieh, Sun, Ginsburg (NVIDIA, 2024),
"nGPT: Normalized Transformer with Representation Learning on the Hypersphere",
https://arxiv.org/abs/2410.01131

**Architecture:** single file pair `include/nn/layers/architectures/ngpt.{h,cpp}`.
Follows the repo's `fox.{h,cpp}` / `stick_breaking.{h,cpp}` conventions: raw
`Tensor` parameters with manually-accumulated `grad_*`, `parameters()`/`gradients()`
pairs shape-matched, `(N, d_model)` row-major token layout, no explicit batch dim.

**Tech Stack:** C++17, repo's `Tensor`, `Dense`. **No `LayerNorm` anywhere** — that is
the point of the paper.

---

## Verified formulation (from the paper — do NOT deviate)

### 1. Everything lives on the unit hypersphere

Let `Norm(v) = v / ‖v‖₂` (row-wise for a matrix). The paper constrains:

* every embedding row → unit norm,
* every **row** of every projection matrix → unit norm,
* the hidden state `h` after every residual update → unit norm.

Because all vectors are unit-norm, a matmul `h · Wᵀ` is a vector of cosine
similarities in `[-1, 1]`, which is why the usual `1/√d` variance scaling is
replaced by explicit learnable scalars.

### 2. HypersphereLinear

Parameter `W ∈ R^(out, in)` (same layout as `Dense::weights`), stored
*unnormalized*; the forward normalizes each row on the fly:

```
Ŵ[o, :] = W[o, :] / ‖W[o, :]‖₂            (‖·‖ over the `in` axis)
y[t, o] = s[o] · Σ_c x[t, c] · Ŵ[o, c]
```

`s ∈ R^(1, out)` is the paper's learnable per-output scaling (§2.3, the `s_qk` /
`s_u` / `s_v` family). Default init `s = 1`.

**Backward — the part that is easy to get wrong.** For a single row `w` with
`ŵ = w/‖w‖`,

```
∂ŵ_i/∂w_j = (δ_ij − ŵ_i ŵ_j) / ‖w‖
```

i.e. `dw = (I − ŵŵᵀ) · dŵ / ‖w‖ = (dŵ − ŵ (ŵ · dŵ)) / ‖w‖`. The projector
`(I − ŵŵᵀ)` kills the radial component: **a gradient step never changes ‖w‖ to
first order**, which is exactly why the paper can re-normalize after each step
without fighting the optimizer. Concretely, per output row `o`:

```
dŴ[o, c] = Σ_t dy[t, o] · s[o] · x[t, c]
n_o      = ‖W[o, :]‖
r_o      = Σ_c Ŵ[o, c] · dŴ[o, c]                  (radial component)
dW[o, c] = (dŴ[o, c] − Ŵ[o, c] · r_o) / n_o
ds[o]    = Σ_t dy[t, o] · (Σ_c x[t, c] Ŵ[o, c])
dx[t, c] = Σ_o dy[t, o] · s[o] · Ŵ[o, c]
```

`update_weights(lr)` does the SGD step **and then re-normalizes every row** —
re-normalization is part of the step, per the paper (§2.6: "we normalize the
matrices after each step"). A test asserts the invariant after `update_weights`,
not just after construction.

### 3. Normalized-LERP residual with eigen learning rates

Standard pre-LN residual `h ← h + f(LN(h))` is replaced by (paper Eq. 10/11):

```
h ← Norm( h + α ⊙ (Norm(f(h)) − h) )        α ∈ R^(1, d_model), learnable
```

`α` are the **eigen learning rates**: the diagonal of a learnable variable-metric
matrix in the paper's Riemannian-optimization reading. Note `α = 1` recovers
`h ← Norm(f(h))` (full replacement) and `α = 0` recovers `h ← Norm(h) = h`.
Paper init: `α ≈ 0.05` scaled by `1/√d_model`; we use a plain `0.05` for a small
`d_model` and note it in a comment.

Since the update is a LERP toward a unit vector followed by a projection back onto
the sphere, the state can never blow up — the paper's stability claim.

**Backward.** With `u = Norm(f(h))`, `m = h + α ⊙ (u − h)`, `h_out = Norm(m)`:

```
dm      = (dh_out − h_out (h_out · dh_out)) / ‖m‖        (Norm Jacobian again)
dα      = dm ⊙ (u − h)
du      = dm ⊙ α
dh     += dm ⊙ (1 − α)                                  (the skip path)
df      = (du − u (u · du)) / ‖f(h)‖                     (Norm Jacobian again)
dh     += <chain df through f>
```

Three separate applications of the same `(I − v̂v̂ᵀ)/‖v‖` Jacobian. Implement it
**once** as a helper `norm_backward(v_hat, norm, dv_hat)` and reuse — DRY, and it
means one bug class instead of three.

### 4. QK scaling replaces the `1/√d` softmax temperature

Because `q` and `k` rows are unit-norm after `HypersphereLinear`, `q·k ∈ [-1, 1]`
and the softmax would be far too cold. The paper introduces learnable `s_qk` and
uses `scores = (q̂ · k̂) · s_qk · √d_head` (§2.3). We store `s_qk` as a scalar
`Tensor(1,1)`, init `1.0`, with `sqrt_dh` folded in as a constant.

### 5. Attention / MLP sublayers

`NGPTBlock` = attention sublayer + MLP sublayer, both wrapped in the normalized-LERP
residual, both built from `HypersphereLinear`:

* attention: `HypersphereLinear` W_q/W_k/W_v (each `(d_model, d_model)`), causal
  single-head-per-`num_heads` slicing identical to `fox.cpp`, then W_o.
* MLP: the paper's SwiGLU variant. To keep the first version's surface small we use
  a 2-layer GELU MLP (`HypersphereLinear` fc1 → GELU → `HypersphereLinear` fc2),
  matching every other block in this repo. Noted as an explicit deviation below.

### Deviations from the paper (recorded, deliberate)

| # | Paper | Here | Why |
|---|-------|------|-----|
| 1 | SwiGLU MLP | GELU 2-layer MLP | every block in this repo uses GELU; SwiGLU is orthogonal to the hypersphere claim being tested |
| 2 | `α` init `0.05/√d` | `α` init `0.05` | `d_model` in tests is 4–8; the `1/√d` factor would put α at the FD noise floor |
| 3 | Embeddings normalized | `NGPTModel` input proj is a `HypersphereLinear`, so its rows are normalized | equivalent for a projection-based (non-lookup) input |

---

## Tasks

### Task 1: RED — constructor + shape + unit-norm invariant for `HypersphereLinear`

**Files:**
- Create: `include/nn/layers/architectures/ngpt.h`
- Create: `include/nn/layers/architectures/ngpt.cpp`
- Create: `tests/test_ngpt.cpp`
- Modify: `include/nn/nn.h` (umbrella include)
- Modify: `Makefile` (link rule, `tests:` deps, `run_tests` echo)

**Step 1:** write `tests/test_ngpt.cpp` with tests 1–3:
1. `HypersphereLinear(0, 4)` and `HypersphereLinear(4, 0)` throw; `(4, 3)` does not.
2. forward `(5, 4) → (5, 3)`, finite, nonzero.
3. every row of `normalized_weights()` has `‖row‖ = 1` to `1e-12`.

**Step 2:** `make build/test_ngpt` → expect compile failure (no header).

**Step 3:** implement the class with forward only.

**Step 4:** `./build/test_ngpt` → tests 1–3 PASS.

**Step 5:** commit `feat(architectures): HypersphereLinear forward + unit-norm invariant`.

### Task 2: normalization-Jacobian FD check (the crux)

**Step 1:** RED — add test 4: `check_param_grad` on `W` of a
`HypersphereLinear(3, 2)` under an L2 loss, centered FD, `eps=1e-6`, assert
`normalized_fd_error < 1e-4`. Use **random** non-uniform init (per TDD skill:
uniform init hides row-vs-column bugs) and a **non-square** `(3, 2)` shape.

**Step 2:** confirm it FAILS with a deliberately naive backward
(`dW = dŴ`, no projector) — record the observed rel_err in the commit message.
This is the mutation test for the Jacobian: without the `− ŵ r` term the error is
O(1), not O(eps).

**Step 3:** implement `norm_backward` and use it. Re-run → PASS.

**Step 4:** add test 5 — hand-derived 1-row case. For `W = [[3, 4]]` (so `‖w‖ = 5`,
`ŵ = [0.6, 0.8]`), `x = [[1, 0]]`, `s = 1`: `y = 0.6`. With `dy = 1`:
`dŴ = [[1, 0]]`, `r = 0.6`, `dW = ([1,0] − [0.6,0.8]·0.6)/5 = [0.128, −0.096]`.
Assert both entries to `1e-12`. (Derived and cross-checked against the FD value
printed by the test — per TDD skill, never trust a hand-derived constant alone.)

**Step 5:** add test 6 — `ds` FD check, test 7 — input-grad FD check.

**Step 6:** commit.

### Task 3: unit-norm invariant survives `update_weights`

**Step 1:** RED — test 8: forward, backward with a nonzero grad, `update_weights(0.1)`,
then assert every row of `normalized_weights()` still has `‖row‖ = 1` **and** that
`W` itself changed (so the test is not vacuous — assert `max|W − W_before| > 1e-9`).

**Step 2:** implement re-normalization inside `update_weights`. PASS.

**Step 3:** test 9 — `zero_grad` clears both grads; test 10 —
`parameters()`/`gradients()` return 2 shape-matched tensors.

**Step 4:** commit.

### Task 4: `NGPTBlock` — normalized LERP residual

**Step 1:** RED — test 11: constructor validation (`d_model=0`, `num_heads=0`,
`d_model % num_heads != 0` throw). Test 12: forward `(4, 8) → (4, 8)`, finite.
Test 13: **every output row is unit-norm** (the hypersphere invariant — the single
most important structural property of the whole layer).

**Step 2:** implement forward. PASS.

**Step 3:** test 14 — `α = 0` (force `alpha_attn_`/`alpha_mlp_` to zero) makes the
block an identity on a unit-norm input, bit-exact to `1e-12`. This is the
signature test that distinguishes the LERP residual from a plain `h + f(h)`
residual: a plain residual with a zeroed sublayer output would also be identity,
so **also** assert that with `α = 1` the output equals `Norm(f(h))`, i.e. is
*independent of `h` except through `f`* — check that perturbing the skip path
cannot be what produces the output.

**Step 4:** commit.

### Task 5: `NGPTBlock` gradients

**Step 1:** RED — test 15: input-grad FD. Test 16: `alpha_attn_` grad FD (the
eigen-LR gradient — the paper-specific parameter). Test 17: `alpha_mlp_` grad FD.
Test 18: `s_qk` grad FD. Tests 19–22: W_q/W_k/W_v/W_o grad FD. Tests 23–24: MLP
fc1/fc2 grad FD.

**Caution (FD on a normalized output).** The block output is unit-norm, so the L2
loss against a fixed target is a function on the sphere; FD is fine but the
gradient magnitudes are small. Use `scale = 0.6` inputs, `eps = 1e-6`, and the
`normalized_fd_error` helper (worst-abs / tensor scale) that `test_fox.cpp` uses,
with a `1e-4` threshold — **not** a per-element relative error, which is
meaningless at the noise floor.

**Step 2:** implement `NGPTBlock::backward` using the shared `norm_backward`
helper. Iterate to PASS.

**Step 3:** commit.

### Task 6: `NGPTModel` + training

**Step 1:** RED — test 25: forward `(4, 3) → (4, 2)`. Test 26: training reduces
loss over 60 SGD steps. Test 27: parameters()/gradients() count contract.
Test 28: determinism — copied params → bit-exact forward.

**Note on the training test:** the model output comes off the sphere via a final
unnormalized `Dense` classifier, so ordinary L2 regression works. If it did not,
the loss would be bounded below by the sphere geometry and "training reduces loss"
would be a weak assertion.

**Step 2:** implement. PASS.

**Step 3:** test 29 — mutation test: stub the `− ŵ r` projector term in
`norm_backward`, confirm ≥3 tests fail, restore. Record which ones in the queue
entry.

**Step 4:** commit; register in `nn.h` + Makefile; move the queue entry to `## Done`.

---

## Verification gate

```bash
make build/test_ngpt && ./build/test_ngpt          # expect all checks PASS
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'   # umbrella
make tests 2>&1 | grep -iE '^(g\+\+.*error|.*Error)' ; echo "build clean"
```

Then spot-run 3–4 neighbouring suites (`test_fox`, `test_stick_breaking`,
`test_lambda_layer`) to confirm no regressions from the umbrella-header change.
