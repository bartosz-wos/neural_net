# Stick-Breaking Attention Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `StickBreakingAttention` — a softmax-free, position-embedding-free causal
attention mechanism where attention weights come from a stick-breaking (residual
allocation) process — plus a `StickBreakingBlock` and `StickBreakingModel`.

**Paper:** Tan et al., ICLR 2025, "Scaling Stick-Breaking Attention: An Efficient
Implementation and In-depth Study", https://arxiv.org/abs/2410.17980
Reference impl: https://github.com/shawntan/stickbreaking-attention

**Architecture:** Multi-head, single file pair
`include/nn/layers/attention/stick_breaking.{h,cpp}`, matching the `DiffAttention`
convention (Dense W_q/W_k/W_v/W_o, per-head slices of a flat `(N, d_model)` tensor,
raw `grad_W_*` tensors accumulated manually, `parameters()`/`gradients()` pairs).

**Tech Stack:** C++17, repo's `Tensor` class, `LayerNorm`, `Dense`. No new deps.

---

## Verified formulation (from the paper — do NOT deviate)

Per head, with `d_h = d_model / num_heads`, query index `j`, key index `i`:

```
z[i,j] = (q_j · k_i) * inv_temp          inv_temp = 1/sqrt(d_h)      (Eq. 1)
β[i,j] = σ(z[i,j])
A[i,j] = β[i,j] * Π_{i<k<j} (1 - β[k,j])                            (Eq. 1)
o_j    = Σ_{i<j} A[i,j] · v_i                                       (Eq. 1)
```

Strictly-causal: `i < j` (a query does NOT attend to itself). The reference impl
exposes `attend_current` to include `i == j`; **we implement `attend_current=false`
(the paper's default)** so row `j=0` has an all-zero attention row.

Numerically stable log-space form (Eq. 11–13) — **this is what we implement**:

```
log(1 - β[k,j]) = -softplus(z[k,j])
A[i,j] = exp( z[i,j] - Σ_{k=i}^{j-1} softplus(z[k,j]) )
       = exp( logβ[i,j] + Σ_{k=i+1}^{j-1} log(1-β[k,j]) )
```
where `logβ[i,j] = z[i,j] - softplus(z[i,j])` and `softplus(x) = log(1+exp(x))`
computed stably as `max(x,0) + log1p(exp(-|x|))`.

Remainder bias (Appendix C, Eq. 14), **enabled by default** since the paper reports
improvements and it keeps the output scale sane:

```
rem_j = 1 - Σ_{i<j} A[i,j]        (∈ [0,1], = 1 for j=0)
o_j   = Σ_{i<j} A[i,j] v_i + rem_j · r_head          r ∈ R^{num_heads × d_h}
```

### Backward (hand-derived, the part that is easy to get wrong)

Given `dO[j, :]` (per head, `d_h`-dim) for the pre-W_o output:

```
dA[i,j] = dO_j · v_i - dO_j · r_head              # 2nd term via rem_j = 1 - ΣA
dV[i]  += A[i,j] * dO_j
dr     += rem_j * dO_j
```

`A[i,j] = exp(S[i,j])` with `S[i,j] = z[i,j] - Σ_{k=i}^{j-1} softplus(z[k,j])`, so
`dS[i,j] = dA[i,j] * A[i,j]`. Then, since `z[m,j]` appears (a) as the leading term
only when `m == i`, and (b) inside the `softplus` sum for every `i <= m <= j-1`:

```
dz[m,j] = dS[m,j] - σ(z[m,j]) * Σ_{i=0}^{m} dS[i,j]        for m < j
```
The `Σ_{i<=m} dS[i,j]` is a **prefix sum over i** — accumulate it as `m` increases.
This single line is the crux of the whole backward; a naive O(L³) double loop is
also correct and is what the test's independent reference will use.

Then the standard QK chain:
```
dq_j += inv_temp * Σ_{m<j} dz[m,j] * k_m
dk_m += inv_temp * Σ_{j>m} dz[m,j] * q_j
```

---

## Task 1: header + skeleton, constructor validation (RED → GREEN)

**Files:**
- Create: `include/nn/layers/attention/stick_breaking.h`
- Create: `include/nn/layers/attention/stick_breaking.cpp`
- Create: `tests/test_stick_breaking.cpp`
- Modify: `Makefile` (build rule, `tests:` deps, `run_tests` echo)
- Modify: `include/nn/nn.h` (umbrella include after `shla.h`)

**Step 1: failing test** — `StickBreakingAttention(0,1)` throws, `(4,0)` throws,
`(4,3)` throws (d_model not divisible by num_heads), `(4,2)` constructs.
Use the placeholder-then-validate constructor pattern (`d_model ? d_model : 1` in
the init list, `throw` in the body) so zero-size Tensors are never allocated.

**Step 2:** Run `make build/test_stick_breaking && ./build/test_stick_breaking`.
Expected: compile error / FAIL.

**Step 3:** Implement ctor + accessors + `parameters()`/`gradients()`/`zero_grad()`.

**Step 4:** Re-run. Expected PASS.

## Task 2: forward — shape, finiteness, causality

Tests: output shape `(N, d_model)`; all-finite; **row 0 attends to nothing** →
with remainder bias, `o_0 == r_head` exactly; strict causality signature — perturbing
`v_i` for `i >= j` leaves `o_j` bit-exact unchanged.

## Task 3: stick-breaking invariant

`Σ_{i<j} A[i,j] + rem_j == 1` to machine precision for every `j`, and
`0 <= A[i,j] <= 1`. Expose `last_A()` (shape `(num_heads*N, N)`) so tests can read
the attention map. This is the test that distinguishes stick-breaking from softmax:
**the sum is ≤ 1 by construction, not normalized to 1**, and the recency bias means
for equal logits `A[i,j]` decreases as `j - i` grows — assert that too.

## Task 4: hand-derived N=2 forward reference

`d_model=2, num_heads=1`, forced weights, hand-compute `z[0,1] = (q_1·k_0)/√2`,
`β = σ(z)`, `A[0,1] = β`, `o_1 = β v_0 + (1-β) r`. Assert rel_err < 1e-12.
Per the TDD skill: compute the expected value from the *formula in the test*, not
from a hand-typed decimal.

## Task 5: independent O(L³) reference for A

Test-local naive implementation using the **direct product form** `β Π(1-β)` (not
the log-space form) must match the production log-space `last_A()` bit-close
(rel_err < 1e-12). This catches log-space algebra errors.

## Task 6–10: FD gradient checks

Central-difference vs analytical, `eps=1e-5`, `rel_err < 1e-4`, on:
input, `W_q.weights`, `W_k.weights`, `W_v.weights`, `W_o.weights`, `remainder_`.
Match grads to params by **`Tensor*` pointer identity**, never by shape (the
MambaConv session lost time to exactly that bug).

Config must be **non-degenerate**: random init (`scale ≈ 0.5`), `N=4`, `num_heads=2`,
`d_model=4`, random target. Also run a `N=6` deeper check.

## Task 11: multi-head correctness

`num_heads=2` output must equal the concatenation of two independent
`num_heads=1` layers fed the same input with the corresponding weight slices.
This is the test that catches per-head slicing/offset bugs.

## Task 12–13: zero_grad / update_weights move all params

## Task 14–16: Block + Model

`StickBreakingBlock` = pre-LN → SB attn → residual → pre-LN → GELU FFN → residual.
**Chain-rule trap (Lambda Layer session):** `res1 = z1 + attn_out` where `z1 = ln1(x)`,
so the backward must call `ln1.backward(d_z1)` and add *that* to `d_x` — not add
`d_z1` directly. Test: Block input-grad FD.
`StickBreakingModel` = input Dense → N blocks → final LN → classifier. Test:
training reduces loss over 30 SGD steps.

## Task 17: mutation testing (mandatory before commit)

Confirm each mutation is caught by ≥1 test:
1. Drop the `- σ(z) * prefix_sum` term in `dz` → input/W_q/W_k FD must fail.
2. Drop the `- dO·r` term in `dA` → FD must fail (proves the remainder path).
3. Change prefix sum from `Σ_{i<=m}` to `Σ_{i<m}` (off-by-one) → FD must fail.
4. `+=` → `=` on `dk` accumulation → FD must fail (needs N ≥ 3 so the loop runs
   more than once — per the TDD skill's single-iteration-vacuity caveat).

Restore all mutations afterwards and re-run to full green.

## Task 18: verify umbrella compiles standalone

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

## Task 19: no regressions + commit

Run neighbouring suites (`test_gau`, `test_diff_transformer`, `test_based`,
`test_lambda_layer`). Then commit:
`feat(attention): stick-breaking attention (Tan et al. ICLR 2025)`
and move the queue entry to `## Done`.
