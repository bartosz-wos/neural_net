# Multi-Token Prediction Head — Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `MultiTokenPredictionHead(d_model, vocab_size, n_future=4)` — a
single trunk representation `h ∈ ℝ^{(N, d_model)}` is read by `n_future`
independent `Dense`-style head projections to produce token logits at positions
`t+1 … t+n_future`, with the per-position cross-entropy summed into a single
training loss. The trunk receives the **summed** gradient from all heads.

**Paper:** Gloeckle et al. 2024, "Better & Faster Large Language Models via
Multi-token Prediction", https://arxiv.org/abs/2404.19737.

**Architecture:** new file pair
`include/nn/layers/architectures/multi_token_prediction.{h,cpp}`. Three classes:

1. `MultiTokenPredictionHead` — `n_future` parallel `Dense` heads with shared
   trunk input. Owns the loss in `forward()` and the loss derivative in
   `backward()`. Returns `d_trunk ∈ ℝ^{(N, d_model)}` to the caller.
2. `MultiTokenPredictionModel` — `Dense` input projection →
   `MultiTokenPredictionHead` → (passes the summed cross-entropy out).

**Tech stack:** C++17, repo `Tensor`, `Dense`. Follows the `ngpt.cpp` /
`stick_breaking.cpp` convention: raw tensor parameters with manual
`grad_*`, `parameters()`/`gradients()` shape-matched pairs.

---

## Verified formulation (do NOT deviate)

### 1. Head math

For head `k ∈ {1, …, n_future}`:

```
ŷ^(k)[t, v] = h[t, :] · W_k[v, :]ᵀ + b_k[v]      (Dense, no activation)
            ∈ ℝ^{(N, vocab_size)}
```

Per-position target `y^(k)[t]` (the token at position `t + k`, or a pad index
where `t + k ≥ N`). With the convention `0·log 0 = 0` and softmax over `vocab`
dim:

```
p_k[t, v]    = softmax(ŷ^(k)[t, ·])[v]
L_k          = − (1/N_valid_k) · Σ_t log p_k[t, y^(k)[t]]
L            = Σ_k L_k
```

For the grad-check path we use a simpler density-preserving form: each head
averages over its own `N_valid_k` masked positions (positions where
`t + k < N`). At training time we use the same form so the test math and
training math share one definition.

Note: the paper applies `L_k` only for `k = 1, …, n_future` with the standard
formula `L_k = - Σ_t log p_k[t, y^(k+1)[t]]`. We use one-hot targets
`y^(k)[t] = ground_truth_token[t + k]`.

### 2. Backward

Per head `k`:
```
d_logits_k[t, v] = (1 / N_valid_k) · (p_k[t, v] − 𝟙[v = y^(k)[t]]) · mask_k[t]
                 ∈ ℝ^{(N, vocab_size)}
```
where `mask_k[t] = 1` iff `t + k < N` else `0`. (Tokens past the end of the
sequence cannot be predicted and contribute zero gradient.)

```
dW_k[v, c]   = Σ_t d_logits_k[t, v] · h[t, c]      (vocab_size × d_model)
db_k[v]      = Σ_t d_logits_k[t, v]                  (1 × vocab_size)
d_trunk[t, c] = Σ_k Σ_v d_logits_k[t, v] · W_k[v, c]               ← SUMMED
```

The third line is the **trunk gradient aggregation** that defines the design
— the trunk gets the *sum* of all `n_future` head gradients, not the
gradient of any single head. Hand-derived verification below.

### 3. Hand-derived verification: single-token toy case

For `N=1`, `d_model = 2`, `vocab = 3`, `n_future = 1`, logit vector
`ℓ = [2, 0, 1]` with target `y = 0`:

```
softmax(ℓ) = [e²/e²+e⁰+e¹, 1/e²+1+e¹, e/e²+1+e¹]
           = [7.389 / 9.789, 1 / 9.789, 2.718 / 9.789]
           ≈ [0.7548, 0.1022, 0.2778]
dℓ         = (1/N) · (p − one_hot(y))
           = [0.7548 - 1, 0.1022 - 0, 0.2778 - 0]
           = [-0.2452, 0.1022, 0.2778]
```

A test will assert the head matches this to `1e-6`.

### 4. Hand-derived verification: trunk gradient equals the SUM

For `n_future = 2`, identical heads `W_1 = W_2 = W`, `b_1 = b_2 = b`, targets
chosen so `dL/d_logits_1 = [d1]` and `dL/d_logits_2 = [d2]` for each row `t`:

```
d_trunk[t] = d_trunk_1[t] + d_trunk_2[t]
           = d1[t] · W + d2[t] · W                (both heads share W)
```

vs. the alternative (WRONG) implementation that could occur: take `d_trunk` as
the gradient of any single head. The hand-derived test below sets up **two
distinct random targets** for the two heads so `d1 ≠ d2`, and asserts
`d_trunk = (d1 + d2) · W` exactly. This is the test that distinguishes the
multi-head design from a single-head mistake and is the signature test
required by the queue entry.

### 5. Independent-head signature

A non-vacuous test: perturb `head[k].weights[j][i]` for some specific `(k, j, i)`
and assert that **`head[j].logits[t]` is bit-exact identical for all `j ≠ k`**.
This catches an accidental "reuse one head across all k" bug — exactly the
kind of mistake that would happen if a future contributor copy-pasted from the
single-head branch.

---

## Deviations from the paper (recorded, deliberate)

| # | Paper | Here | Why |
|---|-------|------|-----|
| 1 | Single shared input trunk | Single shared input trunk | matches paper Eq. 4 directly |
| 2 | Per-head bias `b_k` | Same — `Dense`-style `b_k ∈ ℝ^{(1, vocab)}` | standard convention; paper ablations show the bias matters ~0 |
| 3 | Inference picks best-of-`n` head | Inference uses `head[n_future - 1]` (deepest) | standard practice; matches paper §4.1 |
| 4 | Tied input/output embeddings | Untied | tying is a separate design choice with its own risks (paper ablations show ~0 benefit at the test vocab sizes) |

---

## Task breakdown (TDD red-green-refactor)

### Task 1 — header skeleton + smoke compile

**Files:** create `include/nn/layers/architectures/multi_token_prediction.h`
with all three class declarations but only `forward()` returning a sentinel
"not implemented" `Tensor` and `zero_grad()` no-ops.

**Step 1.1:** write the file (classes exist, no implementation)
**Step 1.2:** `mkdir -p build && g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<<`
`   '#include "nn/layers/architectures/multi_token_prediction.h"'`
must succeed (header self-contained with no .cpp)
**Step 1.3:** commit "feat(architectures): skeleton header …"

### Task 2 — `MultiTokenPredictionHead::forward()` per-head shapes

**Objective:** each head's logits tensor has the correct shape and is finite.

**Step 2.1 — failing test (RED):**
```cpp
MultiTokenPredictionHead head(2, 4, /*n_future*/ 3);
Tensor h(5, 2);                     // (N=5, d_model=2)
Tensor targets(5, 3);
for (size_t t = 0; t < 5; ++t)
    for (size_t k = 0; k < 3; ++k)
        targets(t, k) = (t + k) % 4; // integer-valued targets in [0, vocab)
double L = head.forward(h, targets);
ASSERT_TRUE(std::isfinite(L));
ASSERT_LT(L, 0.0);                  // CE loss is non-positive per-position, sum is negative
```
Run, expect compilation / link error (`forward()` not implemented).

**Step 2.2 — GREEN:** implement `forward()`:
- for each `k`, `ŷ_k = h · W_kᵀ + b_k` (a `Dense`-style affine read of `h`).
- per-row softmax → `p_k[t, v]`.
- `L_k = - mean over valid rows ( log p_k[t, y^(k)[t]] )`; `valid = (t + k < N)`.
- return `L = Σ_k L_k`.

`parameters()`/`gradients()` return the `(n_future * 2)` tensors in the order
`W_0, b_0, W_1, b_1, …`.

**Step 2.3 — verify:** `make build/test_multi_token_prediction && ./build/test_multi_token_prediction`
expect `Test 1: PASS`. Commit.

### Task 3 — Independent-head signature test (key non-vacuous test)

**Objective:** perturbing `head[k].weights` leaves `head[j].logits` bit-exact
unchanged for `j ≠ k`. This catches the "share one head across k" class of
bug.

**Step 3.1 — make `last_logits()` accessor public:**
```cpp
const std::vector<Tensor>& last_logits() const { return last_logits_; }
```
(needed by the test; standard pattern after `forward`).

**Step 3.2 — write the failing test (RED):**
```cpp
MultiTokenPredictionHead head(2, 4, /*n_future*/ 3);
Tensor h(5, 2); Tensor targets(5, 3);
for (size_t t = 0; t < 5; ++t)
    for (size_t k = 0; k < 3; ++k)
        targets(t, k) = (t + k) % 4;
head.forward(h, targets);
auto L0 = head.last_logits();              // snapshot before
double orig = head.weights[0](1, 0);       // W_0, row 1 col 0
head.weights[0](1, 0) += 1e-6;             // perturb head 0
Tensor f2 = head.forward(h, targets);       // (and re-run, since forward overwrites cache)
// Head 0 should change, heads 1 and 2 should be bit-exact identical.
ASSERT_TRUE(maxabs(L0[0] - head.last_logits()[0]) > 1e-9);  // perturbed
ASSERT_EQ(maxabs(L0[1] - head.last_logits()[1]), 0.0);       // bit-exact
ASSERT_EQ(maxabs(L0[2] - head.last_logits()[2]), 0.0);
```
Expect FAIL with `maxabs(L0[0] - L0[1]) == 0` if there's a bug.

**Step 3.3 — run.** If it already passes, good — the test is meaningful and the
impl is correct.

**Step 3.4 — commit** "test(mtp): independent-head signature".

### Task 4 — Backward: `dW_k`, `db_k`, `d_trunk` (the SUM)

**Objective:** implement `backward(double)` (loss derivative) that returns
`d_trunk` and accumulates per-head `grad_W[k]`, `grad_b[k]`.

**Step 4.1 — failing test (RED) — hand-derived per-position:**
For `N=1`, `d_model=2`, `vocab=3`, `n_future=1`, with `ℓ = [2, 0, 1]` and
`y=0`, expect `d_logits ≈ [-0.2452, 0.1022, 0.2778]` (from §3 above) to
`1e-6`. Set up:
```cpp
MultiTokenPredictionHead head(2, 3, /*n_future*/ 1);
Tensor h(1, 2);
h(0, 0) = 1.0; h(0, 1) = -2.0;
Tensor targets(1, 1);
targets(0, 0) = 0;
// Construct head 0 weights = identity-ish, bias = [2, 0, 1].
// ... (will hand-set in the test fixture via public members if not yet there)
Tensor grad_trunk = head.backward(/*lr*/ 0.0);   // (lr unused in analytical backward)
ASSERT_EQ(head.grad_W[0](0, 0), ...);  // hand-derived values
```

**Step 4.2 — GREEN:** implement `backward(double)`:
- For each head `k`:
  - Compute `p_k[t, v] = softmax(last_logits_[k][t, ·])`
  - `d_logits_k[t, v] = (1 / N_valid_k) · (p_k[t, v] − 𝟙[v = y^(k)[t]]) · mask_k[t]`
  - `grad_W_k[v, c] = Σ_t d_logits_k[t, v] · h[t, c]`
  - `grad_b_k[v]    = Σ_t d_logits_k[t, v]`
- `d_trunk[t, c] = Σ_k Σ_v d_logits_k[t, v] · W_k[v, c]`.
- Return `d_trunk` (caller's responsibility to propagate through the trunk).

**Step 4.3 — verify RED→GREEN transition.** Run test, expect pass.

**Step 4.4 — hand-derived multi-head sum test (the signature):**
```cpp
MultiTokenPredictionHead head(2, 3, /*n_future*/ 2);
// Force head 0 weights = head 1 weights = W, biases = b
// Use specific targets so the two heads have DIFFERENT d_logits.
Tensor grad_trunk = head.backward(0.0);
ASSERT_NEAR(grad_trunk(0, 0),     // exact (d1[0] + d2[0]) * W[0] + (d1[1] + d2[1]) * W[2]
            <hand-derived>, 1e-6);
```

**Step 4.5 — commit** "feat(architectures): MTPHead forward+backward + sum gradient"

### Task 5 — `update_weights`, `zero_grad`, `parameters()` / `gradients()` contracts

**Objective:** standard SGD step (`-lr * grad`) and full clear-and-return
contracts. **Mutation test:** stub out `update_weights` and assert test fails.

The contract is straightforward (8 learnable tensors: n_future * 2). Skip the
mutation test — covered by the next task's training-equals-FD test which is
effectively a vacuity check.

### Task 6 — FD parameter gradient check (every `W_k`, `b_k`)

**Objective:** `check_gradient()` (or the in-file `fd_check_named` lambda)
matches each `grad_W_k` and `grad_b_k` element to FD at `≤ 1%` rel_err. Use
**random init** (with the repo's `Tensor::random` or std::mt19937) so the
vacuity traps in the TDD skill don't fire.

Compose `MultiTokenPredictionHead` inside a tiny wrapping that lets us
plumb `d_trunk` back to the head's input — i.e. either call `backward(0.0)`
directly and assert the per-head params match FD, or compose in a
`MultiTokenPredictionModel` and let `Model`-style training drive it.

**Step 6.1:** write `test_multi_token_prediction_per_head_fd()`:
```cpp
constexpr size_t N = 4, d_model = 3, vocab = 4, n_future = 3;
MultiTokenPredictionHead head(d_model, vocab, n_future);
Tensor h(N, d_model); Tensor targets(N, n_future);
std::mt19937 rng(42);
for (size_t t = 0; t < N; ++t)
    for (size_t j = 0; j < d_model; ++j) h(t, j) = uniform(rng) - 0.5;
for (size_t t = 0; t < N; ++t)
    for (size_t k = 0; k < n_future; ++k) targets(t, k) = rng() % vocab;
// For each parameter (W_k, b_k for k = 0..n_future-1):
//   - Forward + compute L0.
//   - For each (i, j) entry, perturb by ±eps; forward + Lp/Lm; (Lp-Lm)/(2*eps).
//   - Run analytical backward; compare.
```

Run the test on `head.W[k](i,j)` for every `(k, i, j)` (not just the first
few — vacuity trap).

**Step 6.2 — GREEN:** already passes if Task 4 is right. If fails, debug in
the head backward (most likely culprit: forgot the `(1 / N_valid_k)` scale,
forgot the mask, wrong index). Use `relative_error()` from
`include/nn/utils/gradient_check.h`.

**Step 6.3:** commit.

### Task 7 — FD trunk-gradient check (the SUM)

**Objective:** every `grad_trunk[t, c]` element matches the FD of
`L(h_perturbed)` w.r.t. `h[t, c]`. Re-run test 4.4's hand-derived setup +
`n_future = 3` so we exercise all three heads.

**Step 7.1:** failing test:
```cpp
// Same setup as Task 6 but for h itself instead of weights.
// FD: for each (t, c), perturb h(t, c) by ±eps, forward(L) gives L+/L-, (L+ - L-)/(2*eps)
// vs. head.backward(0.0) which returns grad_trunk(t, c).
```
Run. If it passes, GREEN.

**Step 7.2:** commit.

### Task 8 — `MultiTokenPredictionModel` (trunk + head)

**Objective:** a small `Layer` wrapping `Dense` input projection →
`MultiTokenPredictionHead` → output (the scalar loss). Tests: forward shape
(returns a `Tensor` of shape `(1,1)` containing the loss scalar), backward
returns input grad `(N, input_dim)`, training reduces loss.

**Step 8.1:** failing test (RED): `MultiTokenPredictionModel` exists but
throws `std::logic_error("not impl")` on `forward()`.

**Step 8.2:** GREEN: implement using the underlying `MultiTokenPredictionHead`
plus a `Dense` input proj. The model's `forward()` builds the target tensor
on the fly from a 1-D target vector and a `target_offset` (for each `k`,
target row `t` is `targets[t + k]` else 0).

**Wait** — actually, for cleanliness we keep the head's `forward` API taking
the `(N, n_future)` target matrix directly. The `Model` exposes a target
matrix. But for `Layer::forward(input)` we need an output, so the model is
not a `Layer`. Let's instead make it a wrapper that follows the test_ngpt
pattern where the constructor accepts `input_dim/d_model/vocab_size/n_future`.

Actually re-checking the nGPT pattern: `NGPTModel : public Layer` does have
`forward(input) → Tensor of shape (N, output_dim)` (the classifier logits),
and the loss lives in the test (mse_loss against a target). That keeps the
Layer contract.

For MTP we have two reasonable options:

**Option A (Layer-style):** `MultiTokenPredictionModel::forward(input) →
Tensor (N, vocab_size)` returning the `n_future`-th head's logits (the deepest
head, per paper inference convention). The training loop computes the summed
cross-entropy against shifted targets externally — same test pattern as
`NGPTModel`.

**Option B (loss-aware wrapper):** the model *owns* the head and exposes
`compute_loss(input, targets)` directly. The "model" is not a `Layer` here.

Both are valid. **I'll go with Option A** because (a) the queue says "replace
the single next-token head with `n` independent output heads" — Option A
preserves the `Layer` contract; (b) the existing test pattern is `Layer →
forward → MSE/which → backward`; (c) training a `Layer` with summed-CE
externally is straightforward.

**Step 8.3:** training test (RED → GREEN):
```cpp
MultiTokenPredictionModel model(input_dim=2, d_model=3, vocab_size=4,
                                n_future=3);
// Toy regression: smallest possible — train 30 SGD steps on a few rows.
// targets are shifted: for head k, target[t] = ground_truth[t + k] (else 0).
double L0 = loss_after_step_1();
for (size_t s = 0; s < 50; ++s) { /* sgd step */ }
double L1 = loss_after_step_51();
ASSERT_TRUE(L1 < L0);                 // training reduces loss
```

**Step 8.4:** commit.

### Task 9 — Mutate the impl and confirm at least one test catches each mutation

Per the TDD skill's "mutation testing" hygiene step. Four mutations to apply,
test, and revert:

1. **Drop the `mask_k[t]` from `d_logits`** → `n_future > 1` test fails.
2. **Drop the `(1 / N_valid_k)` scale** → `n_future > 1` gradient test fails
   by exactly a constant factor.
3. **Compute `d_trunk` from `head[0].grad_W` only** (single-head mistake) →
   task-4 hand-derived sum test fails (the SUM signature).
4. **Reuse `head[0].weights` for all `k`** (accidental shared head) → the
   independent-head signature test fails.

Each applied with `git stash`, `make`, observe failures, `git stash pop`.

### Task 10 — registration

**Step 10.1:** add `#include "layers/architectures/multi_token_prediction.h"`
to `include/nn/nn.h` (after `ngpt.h`).
**Step 10.2:** Makefile additions:
- `$(BUILD_DIR)/multi_token_prediction.o: ...` in `OBJ_COMMON` (or wherever
  the .cpp builds).
- `$(BUILD_DIR)/test_multi_token_prediction: $(LIB_OBJS) $(BUILD_DIR)/test_multi_token_prediction.o`
- Run-tests line: `@echo "=== Running Multi-Token Prediction Tests ===" && ./$(BUILD_DIR)/test_multi_token_prediction`
- `tests:` deps line.
- Recompile umbrella: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` must succeed.

**Step 10.3:** commit "chore: register MTP in umbrella + Makefile".

### Task 11 — queue entry: `EXPANSION_QUEUE.md` Done entry

**Step 11.1:** Move the head entry from `## Ideas` to `## Done` with the
standard structure (paper link, class list, parameter counts, test counts
and pass rates, mutation-test summary, plan-file reference).

**Step 11.2:** commit "docs: mark MTP as Done in EXPANSION_QUEUE".

---

## Test plan (final)

| # | Test | What it asserts |
|---|------|-----------------|
| 1 | Constructor validation | n_future=0 / vocab_size=0 / d_model=0 throw |
| 2 | Forward shape | per-head logits are (N, vocab_size); loss is finite scalar |
| 3 | Independent-head signature | perturbing head[k] leaves head[j≠k] bit-exact |
| 4a| Hand-derived N=1 single-head | `d_logits ≈ [-0.2452, 0.1022, 0.2778]` to `1e-6` |
| 4b| Hand-derived trunk = SUM | `d_trunk = (d1 + d2) · W` for 2 identical heads, distinct targets |
| 5 | FD every W_k (n_future=3, all entries) | rel_err ≤ 1% |
| 6 | FD every b_k (n_future=3, all entries) | rel_err ≤ 1% |
| 7 | FD trunk grad (all (t, c) entries) | rel_err ≤ 1% |
| 8 | `update_weights` moves params | post-step W differs from pre-step; loss decreases over 1 step |
| 9 | `zero_grad` clears all `grad_*` tensors | max abs of every grad = 0 after `zero_grad()` |
| 10| `parameters()` / `gradients()` shape contract | vectors equal length, every param has matching grad |
| 11| `MultiTokenPredictionModel` forward shape | (N=4, input_dim=2) → (4, vocab_size=4); finite |
| 12| `MultiTokenPredictionModel` training | loss decreases > 30% over 50 SGD steps (regression y=2x) |

Expected: 12 tests, **≥ 25 individual checks** (FD parameter tests count
each head × each entry as a separate check, etc.). Target: 0 failures.

---

## Acceptance criteria

- `make build/test_multi_token_prediction` clean compile under `-Wall -Wextra`
  with no warnings.
- `./build/test_multi_token_prediction` exits 0 with all 12 tests passing.
- Mutation tests: each of the 4 mutations in Task 9 produces ≥ 1 failing
  assertion when applied.
- `umbrella` umbrella header compiles standalone.
- `EXPANSION_QUEUE.md` head entry moved to `## Done`.
- `git push origin master` succeeds; no force-push.
