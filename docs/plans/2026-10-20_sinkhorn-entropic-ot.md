# Entropic Optimal Transport (Sinkhorn) — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/sinkhorn.{h,cpp}` — a log-domain entropic-OT solver, the debiased Sinkhorn divergence with its point-space adjoint, and the barycentric-projection helpers — closing a gap that was measured absent in this repo and verified non-redundant.

**Architecture:** Model-free by construction, matching `utils/mup.h`, `utils/ptq.h`, `utils/hutchinson.h` and every other recently shipped utility. Every core is a free function over `Tensor` / `std::vector<double>`, so the whole file is testable with no `Model`, no optimizer and no network. Layered:

- **Core (free functions, no Model):** `sinkhorn_log` (log-domain plan), `sinkhorn_plan`, `ot_eps`, `sinkhorn_divergence`, `sinkhorn_point_gradient`, `sinkhorn_barycentric`.
- **Model-level adapter:** `SinkhornLoss` — computes the divergence over a `Model`'s output against a reference batch, and exposes the point-space gradient as `backward` input.

**Tech Stack:** C++17, `Tensor` (`include/nn/core/tensor.h`), repo `Makefile` (`-O2 -march=native -Wall -Wextra`).

---

## Citations — verified 2026-10-20, none typed from memory

This queue has produced **nine wrong arXiv IDs** before. Every ID below was verified by a **`ti:` TITLE query against the arXiv API** (`search_query=ti:"<full title>"` and `id_list=` confirmation), not recalled.

| ID | Title as returned by the API | Role |
|---|---|---|
| **arXiv:1306.0895** | "Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances" — Marco Cuturi, 2013-06-04, NIPS 2013 | the log-domain iteration and the entropic objective |
| **arXiv:1810.08278** | "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences" — Feydy, Séjourné, Vialard, Amari, Trouvé, Peyré | the **debiased** divergence (the −½ self-terms) |
| **arXiv:1910.12958** | "Sinkhorn Divergences for Unbalanced Optimal Transport" | the unbalanced/`tau` variant — **out of scope for v1**, see §Out of scope |
| **arXiv:1706.00292** | "Learning Generative Models with Sinkhorn Divergences" — Genevay et al. | the **adjoint of the plan** w.r.t. point coordinates |

**None of these is "POT" (Flamary et al.).** Do not add a `pot` citation; it is not what is being implemented.

### The measured gap (re-measured this session, not trusted from the queue)

```
grep -rinE 'sinkhorn|optimal.transport|ot_solver|unbalanced_sinkhorn|brenier|entropic' include/   -> 0 hits
grep -rinE 'transport.plan|coupling|barycentric|assignment.problem|hungarian' include/
    -> only RealNVP `affine_coupling.h`/`coupling_layer.h` and schedule-free's "sequence coupling";
       NOTHING that computes a transport plan or a barycentric projection
```

**Why it is not redundant**, applying the queue's standing rule (*an absence proves a feature is missing, not that it belongs*): the repo's only transportation work is `layers/generative/wgan_gp.h`, a **critic gradient penalty** — an adversarial distance estimated by a learned network on the critic's *input*. OT is a *model-free* distance between two empirical measures that needs no critic, returns the **plan**, and composes with the shipped `utils/mmd_loss.{h,cpp}` (which is a kernel distance with no plan and no geometry). The two are different objects.

---

## THE SEMANTICS — all measured by `tests/sinkhorn_oracle.py`, not recalled

The oracle is written and **runs green (7/7 sections, exit 0)**. It is the specification. The C++ suite pins these numbers.

**S1 — the log-domain update has the sign that ACCUMULATES.**
With `M_ij = (-C_ij + f_i + g_j)/eps` and `lse_i = logsumexp_j M_ij`, the row marginal is `exp(lse_i)`, so the update is
`f_i += eps * (log a_i - lse_i)` (and likewise for `g`). The oracle's **first draft wrote the NEGATED step** (`f = -eps*(la - lse)`), which gives `lse^new = 2*lse - la` — it doubles the error each sweep and diverges to `NaN` in ~40 iterations. *Fingerprint: large, shape-independent error ⇒ my code, not the theory.*

**S2 — `d OT_eps / d C_ij == P_ij` (the envelope claim), verified by FD.**
Not textbook: `d/dP[<P,C> - eps H(P)] = C_ij + eps log P_ij ≠ 0`. Measured ratios `0.99999995, 0.99999996, 1.00000005, 0.99999`. Entries whose `P_ij` sits at the FD noise floor (~1e-8) are asserted on an **absolute** scale — dividing FD noise by a near-zero reference gave `−1.7e16` on the first run and read as "the envelope theorem is false".

**S3 — the debiasing is worth 5 orders of magnitude.** With generalized entropy `H(P) = −Σ P(log P − 1)`:
`plain OT_eps(a,a) = −0.147295507452766` (the bias; **negative**, not positive), and for `Y = X + 1e-3·1` in 3-D: plain `−0.147292507453` vs **debiased `3.000000000003e-06`** — a 49000× separation, and `3.0e-06` is exactly `3·d²·(1e-3)²`. `S_eps(a,a) == 0` exactly.

**S4 — `S_eps ≥ 0`** (0 negatives over 200 random pairs) and `S_eps → 0` as the sets coincide.

**S5 — BOTH limits, measured, and BOTH first-draft expectations were WRONG:**
- `eps → 0` with a **squared-Euclidean** cost gives **`W_2²`**, not `W_1²`. Measured `S/W_2² = 0.99875` at `eps=1e-3`. The first draft asserted the W1 limit and got a clean `3.24×` discrepancy.
- `eps → ∞` gives the **independent-coupling COST** `⟨ab,C⟩ − ½⟨aa,C⟩ − ½⟨bb,C⟩`, **NOT `MMD²`**. Measured at `eps=1e4`: `S − cost = 2.9e-06`, while the eps-matched Gaussian `MMD²` is `2.4e-05` — three orders below. The first draft *predicted* a linear divergence to `−inf` (`−8.0e+04` at `eps=1e4`) and that prediction is **falsified**: the eps-coefficient `−(H(a)+H(b)+1) + ½(2H(a)+1) + ½(2H(b)+1)` is **exactly 0**. **Do not assert an MMD limit in the C++ suite.**

**S6 — the point-space adjoint's SELF-TERM COEFFICIENT IS `-2`, NOT `-1`.** This is the file's load-bearing constant.
`S' = −½ Σ_{p,q} P_xx[p,q] C_xx[p,q]` with `C_xx[p,q] = Σ_c (X[p,c]−X[q,c])²`. The coordinate `X[i,k]` appears **twice**: as the first endpoint of `C_xx[i,q]` and as the *second* endpoint of `C_xx[p,i]` with the opposite sign. With `G = −½P_xx` symmetric, the two halves are equal, giving `4 Σ_{q≠i} G[i,q](X[i,k]−X[q,k])`, i.e. coefficient `−2`. Measured on a 3-point fixture: `−1` → `3.6e-02` (eps=0.3) and `4.7e-01` (eps=1.0); `−2` → **`7.3e-10`**. *A clean factor of two on one term, with everything else at 1e-10, is this repo's standing signature of a missing symmetric half.*

**S7 — FD on an ITERATIVELY-COMPUTED value is contaminated.** `S(x)` is produced by an iterative solver to a fixed tolerance, so it is a staircase; a central difference straddling a change in the **iteration count** measures that, not the derivative. On the 6×5 fixture this made **all three candidate chains — including the correct one — "fail" at ~0.9**. A 3-point fixture converges in a few dozen iterations and the FD is meaningful. *A large shape-independent error is the fingerprint of a broken oracle, not a broken implementation.*

**S8 — a correct plan may contain exact ZEROS.** At `eps=1e-3` on `C = [[0,1],[1,0]]` the solver converges in **1 iteration, residual exactly 0**, plan `[0.5 0; 0 0.5]`, and the **duals stay finite**. `exp(−1/1e-3) ≈ 1e-434`. So `P.min() > 0` **fails a correct implementation**. Assert instead: marginals recovered to tolerance, and `f,g` finite.

**S9 — different weights over the SAME support give `S > 0`**: `S([.5,.5],[.9,.1]) = 0.378288334245` vs `S(a,a) = 0` exactly. `S` is symmetric in `(a,b)` to `6.9e-15`.

---

## Out of scope for v1 (stated so it cannot creep in)

- **Unbalanced / `tau`-relaxed OT** (arXiv:1910.12958) — a different problem.
- **GPU / block tiling / log-GEMM.** `moba.h` already says "FlashAttention varlen / gather-scatter / tiling are out of scope for v1"; the same note applies.
- **Nyström / low-rank approximation** (arXiv:1812.05189 appeared in the title search and is a *different* paper).
- **Log-domain / stabilized Kantorovich potentials for the `inf`-norm** — different regularizer.
- Any change to `utils/mmd_loss.h`. It is a **different object** (kernel two-sample statistic, no plan, no geometry) and is deliberately not extended or re-exported.

---

## Task 1: `sinkhorn_log` — the plan, and nothing else

**Objective:** Log-domain Sinkhorn-Knopp returning the coupling `P`, the duals `(f, g)`, the iteration count and the marginal residual.

**Files:**
- Create: `include/nn/utils/sinkhorn.h`
- Create: `include/nn/utils/sinkhorn.cpp`

**Step 1: Write the failing test** — `tests/test_sinkhorn.cpp`

Per S1/S8, assert the *marginals*, *dual finiteness* and the *diagonal-limit plan*, never `min(P) > 0`:

```cpp
#include "nn/nn.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

int main() {
    // 7-point x 5-point, uniform weights, eps = 0.05 (oracle section 1).
    // The oracle's own seed and RNG are reproduced here so the C++ and the
    // python are computing the same problem.
    ...
    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
```

**Step 2: Run it and watch it fail**

Run: `make $(grep -o 'build/test_sinkhorn' <<< '') 2>/dev/null; g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -c tests/test_sinkhorn.cpp -o /tmp/ts.o`
Expected: compile error — `sinkhorn_log` does not exist.

**Step 3: Minimal implementation.** Signature:

```cpp
// PlanReport carries what the solver observed, so a caller can tell
// convergence from a stall without re-running anything.
struct SinkhornReport {
    int iterations = 0;
    double marginal_residual = 0.0;   // max |row - a|, |col - b|
    bool converged = false;
};

// Cost matrix C is (n, m); a is (n,), b is (m,). Weights are supplied as a
// Tensor of shape (k, 1) so they ride the same 2-D-only Tensor class the rest
// of the repo uses -- there is no 1-D Tensor here.
SinkhornReport sinkhorn_log(const Tensor& C, const Tensor& a, const Tensor& b,
                            double eps, int max_iter = 10000,
                            double tol = 1e-12, Tensor* plan_out = nullptr,
                            Tensor* f_out = nullptr, Tensor* g_out = nullptr);
```

Implementation notes, each one a place the repo has already been bitten:
- `Tensor::operator()` is `data[r*cols + c]` with **no bounds check**. Validate `C.cols == b.rows`, `C.rows == a.rows`, and `a.cols == 1 && b.cols == 1` **before** any indexing.
- `eps <= 0`, `eps` non-finite, `C` empty, `a` or `b` empty → `throw std::invalid_argument`, **discriminated by message** (`"sinkhorn: eps must be finite and > 0"`, `"sinkhorn: C is empty"`, …). See the Hutchinson session: a substring appearing in two messages discriminates nothing.
- Weights are normalised defensively: a `a` or `b` that does not sum to 1 within `1e-9` is renormalised, and a **non-positive** entry is rejected rather than silently `log`-ed (`log(0)` = `-inf` is the repo's recurring `1/0` trap).
- Per-iteration `logsumexp` must max-subtract **per row** and **per column** separately; a single global max underflows on the other axis.

**Step 4: Run and verify GREEN.** Then **mutation-test** the one thing the oracle proved matters: inverting the update sign (S1) and dropping the max-subtraction (S8).

**Step 5: Commit** `feat(utils): log-domain Sinkhorn-Knopp plan solver`.

---

## Task 2: `ot_eps` and the debiased `sinkhorn_divergence`

**Objective:** The entropic objective and the debiased divergence, with the self-terms that make it a *divergence* rather than a cost.

**Files:** Modify `include/nn/utils/sinkhorn.{h,cpp}`, `tests/test_sinkhorn.cpp`

**Tests first** (values pinned from oracle S3/S4/S5/S9):
- `S(a,a) == 0` **exactly** (`==`, not a tolerance).
- `plain ot_eps(a,a) == -0.147295507452766` at `eps=0.05` on the oracle's 7×5 fixture — a **sign** assertion, because the bias is negative and a suite that only checks `|bias| > 0` would pass a solver with the wrong entropy offset.
- Debiased vs plain on the tiny-displacement fixture: `3.000000000003e-06` vs `-0.147292507453`, i.e. assert `|S| < 1e-2 · |plain|`. This is the assertion that **distinguishes** a debiased divergence from a plain OT — a solver returning `ot_eps(a,b)` directly fails it.
- `S ≥ 0` over 200 random pairs.
- Symmetry `S(a,b) == S(b,a)` to `< 1e-12`.
- Different weights, same support → `S > 0`.
- `eps → 0 → W_2²` and `eps → ∞ → independent-coupling cost`. **No MMD assertion** (S5 — measured false).

`double sinkhorn_divergence(const Tensor& X, const Tensor& Y, const Tensor& a,
                            const Tensor& b, const Tensor& cxy, const Tensor& cxx,
                            const Tensor& cyy, double eps, int max_iter = 10000,
                            double tol = 1e-12);`

Commit `feat(utils): debiased Sinkhorn divergence (arXiv:1810.08278)`.

---

## Task 3: `sinkhorn_point_gradient` — the adjoint, with the `-2` pinned

**Objective:** `dS/dX` for squared-Euclidean cost. **This is the task where S6 lives and where the plan's own first draft was wrong.**

**Files:** Modify `include/nn/utils/sinkhorn.{h,cpp}`, `tests/test_sinkhorn.cpp`

**The formula, with the constant that is easy to get wrong:**

```
grad_X[i,k] =  2 · Σ_j  P_xy[i,j] · (X[i,k] − Y[j,k])                 // cross, coeff 2
             + (−2)· Σ_{q≠i} P_xx[i,q] · (X[i,k] − X[q,k])            // self,  coeff −2
```

The `-0.5` debias weight and the two-endpoint symmetry of `C_xx` compose to `-2`. Expose the self-coefficient through an explicitly-documented parameter **or** keep it fixed and pin it with a dedicated mutation; do **not** leave it as an unnamed literal — the oracle found this exact constant wrong once already.

**Tests first** — and per S7, on a **3-point fixture**, not a large one:
- all 6 entries at `eps = 0.3` match FD to `< 1e-8`; measured `7.3e-10`.
- **non-vacuity:** a `-1.0` coefficient must FAIL (measured `3.6e-02`). A gradient test that passes with the wrong constant is worse than none.
- `grad_X` and `grad_Y` come from one shared routine — `grad_Y` is the same expression with `X↔Y` and `a↔b`, and the two agree under swapping the arguments to `1e-12`.
- no returned tensor aliases an input (assert by mutating the input afterwards, with a companion proving the mutation was real).
- empty inputs throw; a shape mismatch throws **discriminated by message**.

Commit `feat(utils): point-space adjoint of the Sinkhorn divergence`.

---

## Task 4: `SinkhornLoss` — the Model-level adapter

**Objective:** The thing a caller actually reaches for: match a `Model`'s output distribution to a reference batch.

**Files:** Modify `include/nn/utils/sinkhorn.{h,cpp}`, `tests/test_sinkhorn.cpp`, `include/nn/nn.h`

```cpp
class SinkhornLoss {
public:
    SinkhornLoss(double eps = 0.1, int max_iter = 10000, double tol = 1e-12);
    // out: Tensor (N, D) model output treated as N weighted points in R^D.
    // ref: Tensor (M, D) reference points. Both weights uniform unless given.
    double forward(const Tensor& out, const Tensor& ref);
    // grad wrt `out`, same shape as `out`.
    Tensor backward(const Tensor& out, const Tensor& ref);
    const Tensor& last_plan() const;   // (N, M): the transport plan
    int last_iterations() const;
};
```

Tests: `forward` matches `sinkhorn_divergence` on the same inputs (the adapter adds nothing); `backward` matches `sinkhorn_point_gradient`; a short training loop reduces the loss **and** liveness is asserted (the gradient is non-zero and every parameter receives a gradient of its own shape) — the `advRob = 0.5000` lesson: *a training loop that reduces loss can be green on a completely dead chain.* And **measure** the loss at `lambda`/`eps` scale before asserting convergence: if the gradient is orders of magnitude below the task gradient, the fixture is degenerate, not slow-converging.

Commit `feat(utils): Model-level SinkhornLoss adapter`.

---

## Task 5: wire it up

**Step 1: umbrella header.** Add `#include "utils/sinkhorn.h"` to `include/nn/nn.h` next to the other utils.

**Step 2: Makefile.** Add, matching the existing convention exactly:
- `$(BUILD_DIR)/test_sinkhorn: $(LIB_OBJS) $(BUILD_DIR)/test_sinkhorn.o`
- `$(BUILD_DIR)/test_sinkhorn` into the `tests:` prerequisite list
- a `run_tests` line: `@echo "=== Running test_sinkhorn ==="; if ./$(BUILD_DIR)/test_sinkhorn; then :; else echo "test_sinkhorn" >> .run_tests_failed; fi`

**Step 3: verify standalone compile** (a focused test including only the new header cannot catch a cross-header collision):
```
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

**Step 4: zero warnings at the project's real flags** (`-O2 -march=native -Wall -Wextra`). A hand-rolled `g++ -std=c++17` is `-O0` and is *not* the build the project ships.

**Step 5: full aggregate run.** `make run_tests`. Expected: no new failures; the single failure is the ODERNN entry already deferred in `NOT_FIXED.md`.

Commit `chore(utils): wire sinkhorn into the umbrella header, Makefile and run_tests`.

---

## Task 6: mutation harness + sanitizer lane

**Objective:** Prove the suite catches the bugs the oracle's derivations actually invited. Model it on `tests/mutate_lr_multiplier.sh`, and carry forward **every** defence that session needed, because each was a real defect:

- `restore()` **must never delete the backup** — it is called once *per mutation*, so a self-cleaning restore works exactly once and silently leaks a mutant into the tree (the hutchinson session: `23 passed / 17 failed` against a tree nobody edited, 7 mutants repaired by hand). Cleanup happens once, on `EXIT`.
- Link set from **`make print-lib-objs`**, asserting the object count `>= 10` — never regex-parse `LIB_OBJS`, which is a *recursive* make variable whose literal text contains no filenames. Either extraction bug would score every mutation `MUT-BROKEN` and certify a suite that catches nothing.
- Exclude the mutated object from the link set; **a link error is not a test failure.**
- Snapshot and restore **both** the `.cpp` and the header atomically (`cp base tmp && mv -f tmp src`), or a header mutation leaks into the tree.
- `rm -f` the intermediate `.o` before every build (stale objects report phantom build failures).
- **Checksum the tree before and after each edit**; a `MUT-NOOP` is a harness defect, not a verdict. Use a Python literal `str.replace` — `perl s///` breaks on a `/` in the replacement and `\n` inside a sed *pattern* does not match.
- Wrap every throwing call in the statistical tests in `call_guarded`, or a detected mutation **aborts the binary** and is reported `MUT-CRASH` instead of a countable `CAUGHT`.
- `MUT-CRASH` in its own column; a missing sanitizer binary is `UNAVAILABLE`, never a verdict.
- **ASan+UBSan lane with an INVERTED verdict:** it fails if the *pristine* build is not ASan-clean. Needed because a mutation that overflows a buffer returns identical output at `-O2` — no value test can see it.
- **Rebuild-and-verify the real binary on EXIT** before reporting any verdict, and report the tree clean.

**Mutations that MUST be caught** (each is a mistake the oracle's own derivations invited):

| ID | Mutation | Caught by |
|---|---|---|
| M01 | `f += eps*(la-lse)` → `f = -eps*(la-lse)` (S1) | marginal assertions fail |
| M02 | drop the per-row max-subtraction in `logsumexp` | underflow → wrong plan |
| M03 | self-term coefficient `-2` → `-1` (S6) | gradient FD test |
| M04 | self-term coefficient `-2` → `-4` | gradient FD test |
| M05 | delete the `-0.5` debias on `cxx` | `S(a,a)==0` test |
| M06 | delete the `-0.5` debias on `cyy` | `S(a,a)==0` test |
| M07 | `-0.5` → `-1.0` on both self-terms | `S(a,a)==0` + tiny-displacement |
| M08 | `ot_eps` entropy `H(P)` → `-Σ P log P` (drop the `-1`) | pinned bias **sign** |
| M09 | `ot_eps` sign `-eps*H` → `+eps*H` | pinned bias **sign** |
| M10 | swap `a` and `b` in one solve | asymmetry on unequal `n`,`m` |
| M11 | `P_xx` reused as `P_yy` | 7d mixed-shape fixture |
| M12 | remove the `q != i` guard in the self-term | `P_xx[i,i]=0` hides it — **needs a fixture with a non-zero diagonal** |
| M13 | remove the empty-tensor guard | the throw test |
| M14 | remove the shape guard on `b.rows != C.cols` | the throw test (see the lr_multiplier M05 heap overflow — **an ASan finding, invisible to any value test**) |
| M15 | `sqrt`-style `2.0` → `1.0` on the cross term | gradient FD test |
| M16 | `<=` → `<` in the weight positivity guard | zero-weight fixture |

**M12 is the one to think about:** `P_xx[i,i]` is `1/n` for the diagonal cost `C_xx[i,i]=0`, which is **non-zero**, so the guard *is* exercised by a uniform fixture — but the contribution is `P_xx[i,i]·(X[i,k]−X[i,k]) = 0`, so it cancels regardless. A mutation removing the guard therefore **survives by algebra, not by accident** — measure it with a probe before writing it off (see §Provably-inert mutations).

**Provably inert — do not file as gaps, and do not silently drop:**
- Reordering `m`-major vs `n`-major iteration: bit-identical.
- Changing the iteration cap with the tolerance already met: bit-identical.
- `ot_eps`'s `sum P = 1` term: algebraically absorbed, see S5.

Give each inert mutation its own `run_mutation_expect_inert` that **inverts the verdict** — passes when the suite passes, **fails** if the mutation ever starts to matter — so a wrong "inert" claim cannot survive a future edit.

Commit `test(utils): N-mutation harness for sinkhorn + an ASan lane`.

---

## Verification checklist (do not declare done without these)

- [ ] `make run_tests` — no new failures beyond the deferred ODERNN entry
- [ ] suite deterministic across 3 reruns
- [ ] clean under ASan+UBSan
- [ ] zero warnings at `-O2 -march=native -Wall -Wextra`
- [ ] umbrella header compiles standalone
- [ ] `make print-lib-objs | wc -l >= 10`
- [ ] harness reports 0 survived / 0 broken / 0 crash / 0 noop, and the tree verifies clean afterwards
- [ ] `git status --porcelain` empty of stray `debug_*`, `*.bak`, `ref_*`

## Then

Move the item to `EXPANSION_QUEUE.md` → `## Done` with the measured results, and record in `NOT_FIXED.md` only if something genuinely remains broken.
