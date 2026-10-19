# Hutchinson Hessian Trace Estimation — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `include/nn/utils/hutchinson.{h,cpp}` — Rademacher/Gaussian Hessian-vector products (forward-over-reverse) and the stochastic trace estimator that the repo's shipped `Sophia` optimizer documents as a requirement but which nothing in the repo computes.

**Architecture:** Model-free by construction. The core is a set of free functions over a **gradient oracle** (`std::function<Tensor(const Tensor&)>` — the idiom already established in `utils/adversarial.h:128`) plus plain `double`/`Tensor`, so the whole file is testable with no `Model`, no optimizer and no network. This is the same shape `utils/mup.h`, `utils/ptq.h` and `utils/curriculum.h` were built with. A thin `Model`-level adapter walks `model.layers` and supplies the oracle, matching `utils/lr_multiplier.cpp`.

**Tech Stack:** C++17, `Tensor`, `Model`, `Layer`, hand-rolled `check()` harness + mutation harness (repo convention).

---

## THE ABSENCE THAT JUSTIFIES THIS, MEASURED NOT ASSUMED

`grep -rinE 'hutchinson|second.order|hessian|newton_step|trust.region|natural.grad' include/` → the ONLY hits are `optimizers/sophia.h` and its `.cpp`. `grep -rinE 'hutchinson|gauss.newton|directional.deriv|second.deriv|jvp|vjp' include/` → **nothing at all**.

The shipped `Sophia` header says, in the repo's own words:

```
//   - h_diag_t is the per-coordinate diagonal of the loss Hessian
//   - For testability this implementation supports TWO sources of h_diag:
//       (1) External: the user calls `set_hessian_estimates(layer_ptr, h_diag)`
//           before step() -- typically with a Hutchinson trace estimate.
//   - h_diag_t = g_t .* g_t -- the empirical Fisher.   <- the FALLBACK
```

And `grep -rn 'set_hessian_estimates' --include=*.cpp --include=*.h .` → **only `sophia.{h,cpp}` itself and `tests/test_sophia.cpp`**, which feeds it hand-made tensors (`opt.set_hessian_estimates(layer, h_diag)`). So the higher-order path in the repo's only second-order optimizer is reachable **only from a hand-built constant**, and every real run silently falls back to the empirical Fisher `g⊙g`.

**A file that documents a limitation IS a specification.** `Sophia` names the estimator. This is the strongest absence-measurement in the queue's vocabulary: not "a paper would be nice" but "a shipped optimizer documents the missing call and no code provides it."

This is also exactly the gap the 2026-10-19 `lr_multiplier` session found from the *same* header (`mup.h:239`), which is why that session recorded the docs-gap grep as a first-class source of picks. It pays again here.

## CITATION — VERIFIED, AND THE COMMON YEAR IS WRONG

Two sources, both pre-arXiv-era, so **there is no arXiv ID to verify and none may be invented** (this queue has produced NINE wrong arXiv IDs):

- Hutchinson, M. F. (1989/1990). "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." *Communications in Statistics - Simulation and Computation* **18**(3):1059-1076. (tandfonline + ResearchGate both give vol 18 no 3, pp. 1059-1076; a secondary index quotes it as vol 19(2).)
- Pearlmutter, B. A. (1995). "Gradient calculations for dynamic recurrent neural networks: a survey." *IEEE Transactions on Neural Networks* **6**(5):1212-1228.

**The Pearlmutter paper is very widely cited as "1994" — including by PyTorch, whose `torch.autograd.functional.hvp` docstring says `d. Pearlmutter (1994)`. dblp, the authority for bibliographic metadata, gives `journals/tnn/Pearlmutter95`, IEEE Trans. Neural Networks 6(5):1212-1228 (1995).** The header records the dblp record and notes the 1994 discrepancy rather than silently picking one. Do not write "1994" as the year of record.

## THE ORACLE ALREADY CAUGHT ONE OF ITS OWN BUGS (record this)

`/home/stefan/.hermes/cache/scratch/hvp_oracle.py` self-checks before emitting any number. Its first draft's `1/sqrt(k)` variance check **drew TWO independent probes**, i.e. it measured `E[z1' H z2]` with `z1 != z2`, whose expectation is **0** for symmetric `H` — and the numbers showed means hovering around `0.0` while the true trace is `0.442`. The fingerprint was *shape-independent means near zero with a large, structured error*, which says "my estimator is wrong", not "the theory is wrong". The estimator is `z' H z` with **ONE** `z`. Fixed; `1/sqrt(k)` now holds to within 1.3% over `k = 1..1024`.

This is systematic-debugging §4c: in a differential test a disagreement is evidence that one of two pieces of code written this session is wrong, and the oracle written minutes earlier is the more likely one.

## NUMBERS THE ORACLE FIXED, WHICH BECOME THE SEMANTICS

Measured in float64 numpy (self-checking, `/home/stefan/.hermes/cache/scratch/hvp_oracle.py`):

| Fact | Measured | Becomes |
|---|---|---|
| `E[z'Hz] == tr(H)`, Rademacher | z = 0.06 sigma | **D1** |
| `E[z'Hz] == tr(H)`, Gaussian | z = 0.60 sigma | **D2** |
| sd of `k=1` estimate is **3004% of `\|tr\|`** | 13.28 vs 0.442 | **D3** (k=1 is useless) |
| sd falls as `1/sqrt(k)`, ratio 0.98–1.01 | over k=1..1024 | **D4** |
| FOM error falls ~10x per 10x `eps` → **O(eps), first order** | 1.2e-14 → 1.6e-8 over `1e-2`→`1e-8` | **D5** |
| `eps = 1e-16` gives `rel_err = 0.985` — garbage | best over sweep is `1e-3` | **D6** (floor) |

---

### Task 1: Write the failing test — trace estimator core

**Objective:** RED for `hutch_trace_*` with no header present.

**Files:**
- Create: `tests/test_hutchinson.cpp`
- Modify: `Makefile` (link rule ~line 263; `tests:` prereq line 871; `run_tests` line ~1030)

**Step 1: Write failing test**

`tests/test_hutchinson.cpp` skeleton (full body per Task 3's list of semantics):

```cpp
// =============================================================================
// tests/test_hutchinson.cpp -- Hutchinson stochastic trace estimation and
// forward-over-reverse Hessian-vector products.
//
// NO FINITE-DIFFERENCE GRADIENT TEST FOR THE ESTIMATOR, and there cannot be
// one: the forward is stochastic and the "function" is a distribution, so a
// centred difference compares two uncorrelated noise draws (this repo's
// systematic-debugging 5c-bis). Verification is by VALUE against a numpy
// oracle plus mutation testing.
// =============================================================================
#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", name.c_str()); }
}
```

**Step 2: Run test to verify failure**

Run: `make $(echo build)/test_hutchinson 2>&1 | tail -20`
Expected: **FAIL** — `fatal error: nn/nn.h: No such file` is WRONG; expected is a
compile error naming the missing `utils/hutchinson.h` symbols. A RED that fails
for the wrong reason proves nothing.

**Step 3–4:** deferred to Task 2 (no implementation written in this task).

**Step 5: Commit** — nothing yet; RED is not committed alone.

---

### Task 2: Header + minimal core implementation

**Objective:** `hutch_trace` / `hutch_trace_rademacher` / `hutch_trace_gaussian` green.

**Files:**
- Create: `include/nn/utils/hutchinson.h`
- Create: `include/nn/utils/hutchinson.cpp`
- Modify: `include/nn/nn.h` (add `#include "utils/hutchinson.h"` after line 87, keeping the block sorted)

```cpp
// Header contents -- the trace core only. D-series docs are mandatory; see
// the "SEMANTICS PINNED" section below for the full list.
#ifndef NN_UTILS_HUTCHINSON_H
#define NN_UTILS_HUTCHINSON_H

#include "../core/tensor.h"
#include <cstddef>
#include <functional>
#include <vector>

// The gradient oracle, identical in shape to utils/adversarial.h:128.
// grad_fn(x) must return dL/dx for the SAME L the estimator is tracing.
using HessianGradFn = std::function<Tensor(const Tensor&)>;

// Hutchinson (1989): tr(H) = E_z[ z' H z ] for E[z z'] = I.
// `probes` Rademacher probes are drawn (mean/variance matched). seed=0 -> the
// deterministic default.
double hutch_trace_gaussian(size_t dim, const HessianGradFn& grad, const Tensor& x,
                            int probes = 1, unsigned seed = 0);
double hutch_trace_rademacher(size_t dim, const HessianGradFn& grad, const Tensor& x,
                              int probes = 1, unsigned seed = 0);
#endif
```

**Run:** `make build/test_hutchinson && ./build/test_hutchinson`
Expected: the D1/D2/D3 assertions pass; the file's own `=== Summary: N passed, M failed ===` line is the count the harness parses (see Task 5 — the harness has been burned by this format before).

**Commit**
```bash
git add include/nn/utils/hutchinson.h include/nn/utils/hutchinson.cpp \
        tests/test_hutchinson.cpp include/nn/nn.h Makefile
git commit -m "feat(utils): Hutchinson stochastic Hessian trace estimation (arXiv-free; Hutchinson 1989)"
```

---

### Task 3: The remaining semantics, one RED each

Per the repo's discipline, each of these gets its own RED → GREEN. The list IS the spec; do not write them all at once.

| # | Semantics | Why it is a natural mistake |
|---|---|---|
| **D1** | `E[z'Hz] == tr(H)` for Rademacher probes | — |
| **D2** | same for Gaussian probes | — |
| **D3** | `probes == 1` is legal but a *terrible* estimator; the test asserts the measured dispersion, not that it equals the trace | an implementation that ignored `probes` and always used k=1 would pass a weak test |
| **D4** | sd falls as `1/sqrt(k)` — assert the **ratio**, not absolute values | a test that only asserts "k=16 is closer than k=1" passes even if the estimator is `k^1.5`-wrong |
| **D5** | FOM is **first order** in `eps` — assert the error *ratio* between `eps` and `eps/10` is ~10, not ~100 | a symmetric-difference implementation also *converges*; only the ORDER distinguishes them |
| **D6** | a floor: `eps` below ~`1e-12` is unusable; the entry point **validates and throws** rather than returning garbage | returning `0.98` relative error silently is the failure mode |
| **D7** | the same seed reproduces **bit-identically**; different seeds differ | a global-RNG implementation is not reproducible |
| **D8** | the returned tensor is **fresh** — mutating the input after the call does not change it | aliasing returns a caller-visible corruption |
| **D9** | `dim == 0` or `probes < 1` **throws** `std::invalid_argument` | — |
| **D10** | a **quadratic fixture**: `f(x)=½x'Hx+c'x` has exact trace; assert against the closed form | — |
| **D11** | the trace is a **sum over parameters** of per-tensor diagonal estimates, and `tr(H) = Σ_i z_i' H_ii z_i` | off-by-one in the parameter walk |
| **D12** | **exactness at `k` large on a rank-1 H**: `tr(H) = λ`, and `z'Hz` for Rademacher is bounded — assert `|estimate − λ| < λ·small` | — |

**Explicit non-feature:** no Newton step, no trust region, no L-BFGS. The gap measured in this repo is *an estimator nobody computes*, not an optimizer.

---

### Task 4: The `Model`-level adapter — the part that closes the actual gap

**Objective:** make `set_hessian_estimates` reachable from real training.

**Files:**
- Modify: `include/nn/utils/hutchinson.h`, `.cpp`

```cpp
// FOM Hessian-vector product over a Model's parameters, for ONE probe v.
// Walks model.layers -> parameters() in the same order Sophia's step() does.
std::vector<Tensor> hutch_hvp_model(Model& model, const Tensor& x, const Tensor& dL_dy,
                                    const std::vector<std::vector<Tensor>>& probes,
                                    size_t layer_idx, size_t param_idx);

// Per-tensor diagonal estimates, ready to hand to Sophia::set_hessian_estimates.
std::vector<std::vector<Tensor>> hutch_trace_diagonal(Model& model, const Tensor& x,
                                                      const Tensor& dL_dy,
                                                      int probes = 1, unsigned seed = 0);
```

`hutch_hvp_model` must **snapshot and restore** the parameters around the probe
(the FOM perturbs `θ ← θ + eps·v` and the caller must be left untouched) — this is
the same snapshot discipline `utils/lr_multiplier.cpp` already implements, and
the same reason: a helper that mutates the caller's model and silently restores
it is a landmine.

**Verification:** assert that after `hutch_hvp_model` returns, every parameter is
**bit-identical** to before (measure this, do not assume it).

---

### Task 5: Mutation harness — `tools/hutchinson_mutation.sh`

**Objective:** prove the suite is not vacuous.

Model it on `tools/curriculum_mutation.sh`, which is the most recent working
example, and apply EVERY defence that file records (each was learned the hard
way):

1. Link set from `make print-lib-objs` (**not** a regex over the Makefile —
   `LIB_OBJS` is a *recursive* make variable and a regex yields ZERO objects,
   turning every mutation into a false verdict). Assert count `>= 10`.
2. Refuse to run if `build/utils/hutchinson.o` is in the link set.
3. `rm -f build/test_hutchinson.o build/utils/hutchinson.o` before every build.
4. Verify each edit applied by **checksum**, else `MUT-NOOP`.
5. Snapshot **and** restore **both** the `.cpp` and the header, atomically
   (`cp base tmp && mv -f tmp src`).
6. Parse the summary line with the **`N passed, M failed`** format and guard
   against the recorded `NF="1 ==="` bash-syntax-error inversion.
7. Do not run `make` concurrently with the harness (it races on `build/*.o`).
8. Rebuild and re-verify the real binary **on exit** (stale-executable trap).
9. Mutations must compile **warning-free**; `-Wunused-variable` ⇒ `MUT-BROKEN`,
   which is a *build* verdict, not a mutation verdict.
10. `0x1f` as the spec delimiter — `|` collides with `||` in C++ guards.

Minimum mutation set (8): drop `1/probes`; `+` → `-` in the accumulate; use
`z` instead of `z²`; Rademacher draw `0` instead of `±1`; omit the parameter
restore; drop the `dim==0` guard; `eps` validation inverted; restore the wrong
index.

---

### Task 6: Aggregate run, ASan lane, warnings, umbrella compile

```bash
make build/test_hutchinson -j8
./build/test_hutchinson                       # expect 0 failures
bash tools/hutchinson_mutation.sh             # expect all caught, 0 broken
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'   # umbrella alone
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude ...           # ASan lane
make run_tests                                # aggregate: no NEW failures
```

The aggregate currently has exactly ONE known failure (`test_neural_ode` /
ODERNN, already deferred in `NOT_FIXED.md`). Anything beyond that is a
regression from this change.

---

## SELF-CONTRADICTION SCAN (required before implementing)

- Task 2 says the trace core lands first; Task 4 says the Model adapter needs the
  core. Consistent — the adapter is strictly additive.
- D5 asserts *first order*; the oracle measured a ~10x-per-decade fall, which is
  first order. Consistent.
- Task 3's D6 says the entry point **throws** on tiny `eps`; Task 2's signature
  shows no `eps` parameter (it belongs to the FOM entry point in Task 4). Recorded
  as an open point — resolve by giving the FOM entry point its own validated
  `eps` in Task 4 and keeping the trace core `eps`-free.

## OPEN DECISIONS

- Whether `hutch_hvp_model` should also accept an already-cached gradient (it is
  one gradient call cheaper when the caller already has `g`). Default: **yes**,
  an overload taking `(x, g0, ...)` — but only after the oracle-backed core is
  green, and only if it does not change the core's behaviour.