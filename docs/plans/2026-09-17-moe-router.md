# MoE Router Utility — Implementation Plan

**Goal:** Extract the shared top-k softmax routing + load-balancing math into a
standalone, fully-tested utility class so future MoE variants can compose it
without re-implementing (and re-bug-introducing) the same logic. Today this
math is duplicated inline across `sparse_moe.cpp::softmax_topk` (proper
softmax, k=2 typical) and `mixture_of_experts.cpp::SparseDispatcher`
(non-softmax variant). The existing 4 MoE layers stay untouched — this is a
**foundation, not a refactor**.

**Architecture:** A pure stateless utility class `MoeRouter` (no Layer
derivation) in `include/nn/layers/architectures/moe_router.{h,cpp}` with:

- **Public API:**
  - `softmax_topk(logits, k) -> MoeRouterState` (forward): returns
    `probs: (B, E)`, `topk_idx: (B, k)`, `dispatch_frac: (E,)`, `mean_gate_prob: (E,)`,
    `maxv: (B,)`, `exps: (B, k)`, `logsumexp: (B,)`. Numerically-stable
    (max-subtraction, with `sumexp` floor at 1e-30).
  - `softmax_topk_backward(d_gate_probs, state) -> d_logits: (B, E)` (backward):
    the gradient of the softmax-topk operator w.r.t. its logits, given the
    gradient w.r.t. its output probs.
  - `load_balance_loss_unscaled(dispatch_frac, mean_gate_prob) -> double`:
    returns `E * Σ_e f_e * p_e` — the Shazeer-form unscaled aux loss.
- **Backward correctness:** the `softmax_topk_backward` function is meant to be
  called inside an MoE-layer's `backward()` method. The TDD gradient check
  perturbs the input logits by ±ε and compares the analytical
  `softmax_topk_backward` output (via the chain rule for one-hot `d_gate_probs`
  probes) against the finite-difference numerical gradient of
  `softmax_topk.forward(perturbed_logits).probs`.

**Tech Stack:** C++17, plain `<vector>` / `<algorithm>` / `<cmath>`,
`Tensor` from `core/tensor.h`. No new deps.

**Why this is in scope, not a refactor that crosses working layers:** the 4
existing files (`sparse_moe.cpp`, `mixture_of_experts.cpp`, `soft_moe.cpp`,
`deepseek_moe.cpp`) are unchanged. `MoeRouter` is a *new utility*; future
MoE variants (e.g. Mixture-of-Depths as MoE-of-Blocks, Mixtral-style top-2
models, soft-then-hard hybrids) can adopt it without disturbing the working
ones. Jamba still uses the legacy `MoELayer`; SparseMoE, SoftMoE, and
DeepSeekMoE all still use their inline routing.

---

## Tasks

### Task 1: Write failing tests for `MoeRouter`

**File:** `tests/test_moe_router.cpp` (new)

**Coverage (27 focused checks):**

1. Constructor validation: `MoeRouter::softmax_topk` with empty logits (rows=0)
   throws.
2. Forward shape: `B=4, E=6, k=2` → `probs (4,6)` shape, `topk_idx (4,2)`,
   `dispatch_frac (6,)`, `mean_gate_prob (6,)`, `maxv (4,)`, `exps (4,2)`,
   `logsumexp (4,)`.
3. **Top-k indices are exactly the k largest logits per row** (test on a hand-
   crafted fixture where the 2nd and 5th column are obviously largest).
4. **Top-k probs sum to 1 per row** (max err < 1e-12).
5. **Non-selected entries are exactly 0** in `probs` (sparsity contract).
6. **Mean gate prob ≡ mean over rows of gate probs for selected columns only**
   (verifies the formula).
7. **Dispatch fraction ≡ count of selections ÷ batch** for each expert.
8. **Load-balance loss** matches hand-derived closed form for a tiny fixture
   (B=2, E=3, k=1: 2 rows → 2 selections total, fractions sum to 1, then
   `E * Σ f_e * p_e` matches).
9. **Numerical stability:** logits spanning 1000.0 to -1000.0 don't produce
   NaN/Inf in probs.
10. **Edge case k=E** (all experts selected): every entry of `probs` is
    `1/E` (modulo argmax tie-breaking — verify `Σ_e probs = 1` per row).
11. **softmax_topk_backward shape:** `d_logits (B, E)` matches the input
    shape of `softmax_topk`.
12. **Backward correctness probe:** for row b, set `d_gate_probs[b, e] = 1`
    and 0 elsewhere; verify `d_logits` matches `probs * (1_e - probs)` on the
    selected indices and `0` on non-selected (the math: when only one probs
    entry is bumped, the chain rule through softmax-topk gives
    `d_logits[b,e] = probs[b,e] * (1_{e=e*} - probs[b,e*])` for the bumped
    index e*, and `d_logits[b,e] = -probs[b,e] * probs[b,e*]` for the other
    selected e≠e*).
13. **Full backward via FD:** for one column of logits (one (b,e) coordinate
    of `logits`), compare `d_logits[b,e]` from `softmax_topk_backward(d_gate_probs, state)`
    against the centred finite-difference of `softmax_topk.forward(perturbed_logits).probs`
    for a non-trivial `d_gate_probs`. Pass with rel_err < 1e-9.
14. **Backward symmetry:** for `probs_col = d_gate_probs[:, e]` summed with
    `probs_col * (1 - probs)` for each column e (i.e. `d_logits = probs *
    (d - sum(probs * d))`), verify ALL entries are exactly computed (algebra
    identity test — should be exact, not just FD-numerical).
15. **load_balance_loss_unscaled symmetry:** verify it's invariant under
    permuting expert indices (same set of {f_e} and {p_e} in any order).
16. **load_balance_loss_unscaled non-negativity:** result ≥ 0 (always).
17. **load_balance_loss_unscaled with uniform fractions (1/E each) and
    uniform mean probs (1/E each)** returns exactly 1.0 (= E * E * (1/E)^2 = 1).
18. **Multi-batch (B=8, E=4, k=2)** + per-row randomly perturbed logits:
    forward shape correct; all invariants from above still hold.
19. **Backward chain test through auxiliary load-balance loss:** if we treat
    the load-balance loss as the scalar loss being differentiated, and the
    auxiliary gradient is `dL/df_e = E * p_e`, `dL/dp_e = E * f_e`, then
    propagating through softmax-topk-backward
    (`d_logits[b,e] = (dL/dp_e) / B` for every b, since p_e is mean over b)
    matches FD.
20. **State round-trip:** building `softmax_topk(logits, k)` then immediately
    feeding the state back into `softmax_topk_backward(ones(B,E), state)` gives
    `d_logits = 0` per row (because softmax rows sum to 1: the chain rule with
    uniform d_gate_probs is exactly zero, this is the row-sum constraint
    invariant of softmax).
21. **State fields are independent of caller mutation** (changing `logits`
    after construction does not affect stored fields — verifies the
    `make_state_*` snapshot semantics).
22. **Compatibility of forwarded probs with downstream Softmax-topk combine:**
    if `composed[b, j] = Σ_e probs[b, e] * exp_e[b, j]` (the typical MoE
    combine), then `d_logits[b, e] = (Σ_j d_composed[b, j] * exp_e[b, j]) -
    probs[b,e] * Σ_{e'} probs[b, e'] * (Σ_j d_composed[b, j] * exp_{e'}[b, j])`.
    Numerical check: perturb logits, see that the chain rule and FD agree
    on the combined `composed` scalar.
23. **Tie-breaking in topk:** when two logits are exactly equal, the result is
    deterministic (whichever comes first in the partial sort, doesn't matter,
    just that probs sum to 1 per row regardless).
24. **k=0 throws** (invalid: nothing to route to).
25. **k > E throws** (invalid: can't pick more than exist).
26. **Batch size 0** (rows=0): forward returns empty state, backward of empty
    returns empty `d_logits`.
27. **Determinism:** running softmax_topk twice on identical logits (no state
    between) produces bit-exact identical `probs`, `topk_idx`, and the
    auxiliary statistics.

**Run (must RED initially):**
```bash
make build/test_moe_router
./build/test_moe_router      # expect compile error: file doesn't exist
```

### Task 2: Write header `moe_router.h`

**File:** `include/nn/layers/architectures/moe_router.h` (new)

```cpp
#ifndef MOE_ROUTER_H
#define MOE_ROUTER_H

#include "../../core/tensor.h"
#include <vector>
#include <cstddef>

// Snapshot of MoeRouter::softmax_topk — exactly what softmax_topk_backward
// needs to compute the gradient. All fields are independent of the input
// logits object (the caller does NOT need to keep logits alive).
struct MoeRouterState {
    Tensor probs;                       // (B, E)
    std::vector<std::vector<size_t>> topk_idx;   // (B, k)
    std::vector<double> dispatch_frac;          // (E,)
    std::vector<double> mean_gate_prob;         // (E,)
    std::vector<double> maxv;                   // (B,) — for stability
    std::vector<std::vector<double>> exps;      // (B, k) — pre-softmax exponentiated deviations
    std::vector<double> logsumexp;              // (B,)
    size_t num_experts;                         // stored for sanity
    size_t k;                                   // stored for sanity
};

// Pure-stateless top-k softmax routing utility for Mixture-of-Experts layers.
//
// softmax_topk takes (B, E) gate LOGITS, computes (per-row) the k largest of
// those logits, applies a numerically-stable softmax over them, and writes
// g[b, e_selected] = softmax(...). Non-selected entries are 0. Sparse.
//
// softmax_topk_backward computes dL/d(logits) given dL/d(probs). The chain
// rule for softmax in the top-k restricted space is:
//
//   For row b and selected index e* in S=topk_idx[b]:
//     d_logits[b, e*] = probs[b,e*] * (d_gate_probs[b,e*] -
//                                       Σ_{e in S} probs[b,e] * d_gate_probs[b,e])
//   For row b and non-selected e (not in S):
//     d_logits[b, e]  = 0   (hard topk mask; soft variant would subtract)
//
// load_balance_loss_unscaled computes the standard Shazeer-form auxiliary
// loss  E * Σ_e f_e * p_e  — caller multiplies by alpha.
class MoeRouter {
public:
    static MoeRouterState softmax_topk(const Tensor& logits, size_t k);
    static Tensor softmax_topk_backward(const Tensor& d_gate_probs,
                                         const MoeRouterState& state);
    static double load_balance_loss_unscaled(
        const std::vector<double>& dispatch_frac,
        const std::vector<double>& mean_gate_prob);
};

#endif // MOE_ROUTER_H
```

### Task 3: Write impl `moe_router.cpp`

**File:** `include/nn/layers/architectures/moe_router.cpp` (new)

Key invariants:
- `softmax_topk`: top-k via `partial_sort`, max-subtraction for stability,
  `sumexp` floor at `1e-30`.
- `softmax_topk_backward`: per-row, compute the scalar
  `s = Σ_{e in S} probs[b, e] * d_gate_probs[b, e]`, then for each
  `e in S`: `d_logits[b, e] = probs[b, e] * (d_gate_probs[b, e] - s)`.
  Non-selected entries stay zero. Validate: `state.num_experts` matches
  `d_gate_probs.cols`, `state.k` matches the k used in forward.
- `load_balance_loss_unscaled`: return E * Σ_e f_e * p_e.

### Task 4: Run tests — verify GREEN

```bash
make build/test_moe_router
./build/test_moe_router
# expect: "=== Summary: 27 passed, 0 failed ==="
```

If any test fails: read the failure (rel_err, location), fix the
corresponding line, re-run. ONE bug at a time (Rule of Three).

### Task 5: Mutation-test non-vacuous

For each non-trivial line in `softmax_topk_backward` and
`load_balance_loss_unscaled`, add a single-line stub or value-swap. Run
again. Confirm at least one test fails per mutation. Tests covering the
mutated line are non-vacuous.

Mutations:
1. Zero out the `s = Σ g * d_gate` term — 4 tests should fail (those that
   exercise the full chain rule, not the single-cell probe).
2. Replace `e in S` mask with NO mask (gradient flows to all E entries) —
   2 tests should fail (the all-zero non-selected test + a full FD test).
3. In `load_balance_loss_unscaled`, multiply by `E * (E-1)` instead of `E`
   — 1 test should fail (the closed-form `E * Σ (1/E)^2 = 1` test).
4. Drop the `1e-30` `sumexp` floor — 1 test should fail (the
   `logits=-1000` stability test, since without the floor `exp(-1000 - 0)`
   overflows to 0 then `0/0 = NaN`).

After confirming mutations catch, **revert all stubs**.

### Task 6: Register in umbrella + Makefile

**Modify** `include/nn/nn.h`:
- Add `#include "layers/architectures/moe_router.h"` in the
  `architectures/` block.

**Modify** `Makefile`:
- Add `$(BUILD_DIR)/test_moe_router: $(LIB_OBJS) $(BUILD_DIR)/test_moe_router.o`
- Add `$(BUILD_DIR)/test_moe_router.o` to the `tests:` deps line (whatever
  the existing combined line in the Makefile uses).
- Add the echo-and-run pattern:
  ```
  @echo "=== Running MoeRouter Tests ===" && ./$(BUILD_DIR)/test_moe_router
  ```

### Task 7: Verify umbrella compiles standalone

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -x c++ -fsyntax-only \
    - <<< '#include "nn/nn.h"'
# expect: no output, exit 0
```

### Task 8: Commit

```bash
git add tests/test_moe_router.cpp \
        include/nn/layers/architectures/moe_router.h \
        include/nn/layers/architectures/moe_router.cpp \
        include/nn/nn.h \
        Makefile \
        docs/plans/2026-09-17-moe-router.md
git commit -m "feat(architectures): MoeRouter — extracted top-k softmax routing + load-balance utility (27/27 tests pass)"
git push origin master
```

### Task 9: Update EXPANSION_QUEUE.md — move the MoE cleanup item to Done

**Modify** `EXPANSION_QUEUE.md`:
- Delete the `## Ideas` line about MoE cleanup.
- Add to `## Done` (top of Done list, since this is the newest):
  ```
  - **MoE Router utility** — `MoeRouter::softmax_topk` + `softmax_topk_backward` + `load_balance_loss_unscaled` extracted from the duplicated inline routing logic in `sparse_moe.cpp::softmax_topk` (Shazeer form) and `mixture_of_experts.cpp::SparseDispatcher` (legacy). Pure stateless utility class in `include/nn/layers/architectures/moe_router.{h,cpp}`, no Layer derivation. The 4 existing MoE layers (`sparse_moe`, `mixture_of_experts`, `soft_moe`, `deepseek_moe`, plus `moe_mamba`) are unchanged — this is a foundation for FUTURE MoE variants, not a refactor of the working ones. **27/27 focused checks pass at machine precision**: forward shape, top-k indices exactly the k largest, probs sum to 1 per row, non-selected entries exactly 0, mean gate prob formula, dispatch fraction formula, closed-form load-balance loss, numerical stability with logits spanning ±1000, edge cases (k=E, k=0, k>E, B=0), softmax-topk backward correctness via single-cell probe and full FD with rel_err < 1e-9, state symmetry identity d_logits=0 under d_gate_probs=ones, multi-batch invariant, zero non-selected partial-sort tie-breaking determinism, determinism bit-exact across reruns. **Mutation-tested non-vacuous** (4 mutations, all caught). Plan: `docs/plans/2026-09-17-moe-router.md`. Registered in `include/nn/nn.h` umbrella (verified by standalone `-fsyntax-only` compile) and Makefile (`build/test_moe_router` rule, tests deps, `=== Running MoeRouter Tests ===` echo). No regressions (sparse_moe 21/21, soft_moe 30/30, deepseek_moe 46/46, jamba 32/32, moe_mamba unchanged).
  ```

Commit this update too:
```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark MoE Router utility as Done in EXPANSION_QUEUE"
git push origin master
```

---

## Verification checklist

- [ ] test_moe_router.cpp compiles clean (`-Wall -Wextra`, no warnings)
- [ ] 27/27 focused checks pass three times in a row
- [ ] Mutation-tested: at least 4 stubs each break ≥1 test
- [ ] nn.h umbrella compiles standalone
- [ ] Other MoE tests still pass (sparse_moe, soft_moe, deepseek_moe, jamba,
      moe_mamba all unchanged from baseline)
- [ ] Two atomic commits — (1) implementation + tests + registration, (2)
      queue update
- [ ] No debug artifacts left in repo
