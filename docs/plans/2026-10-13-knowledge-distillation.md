# Knowledge Distillation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship Hinton/Vinyals/Dean knowledge distillation (arXiv:1503.02531) as a loss over teacher/student logits: softened-KL at temperature T with the paper's T² gradient rescaling, plus the hard-label/soft-target weighted combination.

**Architecture:** A new `include/nn/utils/knowledge_distillation.{h,cpp}` holding a model-free `DistillationLoss` class plus free helper functions. Model-free on purpose: it consumes `(N,K)` teacher and student logit matrices, so it composes with any classifier in the repo and can be unit-tested without building a `Model`. The `T²` convention reuses the already-shipped `temperature_scaling` semantics rather than re-deriving them.

**Tech Stack:** C++17, existing `Tensor`, existing `log_softmax` from `utils/losses.h`, existing `Reduction` enum from `utils/losses.h`.

---

## Semantics pinned from primary sources (do NOT re-derive from memory)

Read before implementing. Each is a natural mistake that a plausible-looking
reimplementation gets wrong.

1. **Softmax DIVIDE by T, not multiply.** Hinton Eq. 1: `q_i = exp(z_i/T) / Σ_j exp(z_j/T)`. Both teacher and student use the *same* T at training time; the student reverts to `T = 1` at inference.

2. **KL direction is `KL(teacher ‖ student)`.** Oracle 2: the two directions differ by 2.8e-06 on the fixture — KL is asymmetric, so a transposed argument order is a real bug that a "loss(teacher,teacher) == 0" self-test cannot detect.

3. **`log_target` form.** PyTorch `Loss.cpp:264`: `output = xlogy(target, target) - target * input` with `input` = student **log**-probs and `target` = teacher **probs**. So the loss is built from `log_softmax(student/T)` and `softmax(teacher/T)` — two separate normalisations, not a re-use of one.

4. **`xlogy` zero rule.** `x * log(y)` with `x == 0` is **0** even when `y == 0` (`0 * log 0`). A naive `p * log(p/q)` yields `NaN` when a teacher probability underflows to 0. Teacher probs are a softmax, so underflow is reachable at large T.

5. **PyTorch `Reduction::Mean` is mean over ALL ELEMENTS**, `Loss.cpp:66-68` (`unreduced.mean()`), which divides by `rows*cols`. It is **not** `batchmean`. Oracle 7: `mean/batchmean = 1/K = 0.333333` exactly. A reimplementation that divides the per-row KL by `rows` silently reproduces `batchmean` and is off by a factor of `K`. This repo's `Reduction` enum already has this split documented for BCE (D7).

6. **The T² rescaling is on the LOSS, applied by Keras** (`LogitsDistillation.compute_loss`: `loss * (self.temperature**2)`), matching Hinton's "it is important to multiply them by T² when using both hard and soft targets". Oracle 4: the gradient norm ratio is **exactly 9.0** per row at T=3.

7. **The hard/soft combination is a WEIGHTED SUM OF TWO OBJECTIVES**, not a mixture of the targets. Hinton: "the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1." So the hard term is at T=1 and the soft term at T.

8. **Hinton's Eq. 3→4 limit requires zero-meaned logits per case** (`Σ_j z_j = Σ_j v_j = 0`). Oracle measured this directly: on un-zero-meaned rows the limit does not converge (rel err stuck at ~0.28 and *growing* with T); on zero-meaned rows it converges to `(z-v)/(K·N_rows)` (rel err 9.95e-04 at T=1e3, degrading after as float32 cancellation takes over). Documented as a precondition, not asserted as an invariant.

9. **Identical models give ~1e-8, NOT bitwise zero.** Oracle 5: torch forms `xlogy(p,p) - p*s` with `log(softmax)` rounded separately, so the self-KL residue is 1.43e-07. Any test asserting `== 0.0` would fail against a correct implementation.

---

## Task 1: RED — header + failing test suite

**Objective:** Create the header (declarations only) and a test file that fails to compile against the missing implementation.

**Files:**
- Create: `include/nn/utils/knowledge_distillation.h`
- Create: `tests/test_knowledge_distillation.cpp`

**Step 1: Write the header with declarations only, no definitions.**

Include guard `NN_UTILS_KNOWLEDGE_DISTILLATION_H`, include `"../core/tensor.h"` and
`"losses.h"` (for `log_softmax` and `Reduction`). Declare:

- `class DistillationLoss` with constructor `(double temperature = 3.0, double alpha = 1.0, Reduction reduction = Reduction::Mean)`, where `alpha` is Hinton's weight on the **hard**-label cross-entropy, and methods:
  - `Tensor forward(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor& labels) const;`
  - `Tensor backward(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor& labels, const Tensor& grad_output) const;`
  - accessors `temperature()`, `alpha()`, `reduction()`
- free functions, so the pure-soft-target case is reachable without an object:
  - `double distillation_kl(const Tensor& student_logits, const Tensor& teacher_logits, double T);`
  - `Tensor distillation_kl_backward(const Tensor& student_logits, const Tensor& teacher_logits, double T);`

**Step 2: Write the test file.**

Structure it exactly like `tests/test_temperature_scaling.cpp`: a `check(name, pass)`
helper taking `(name, result)` order, a `close()` and a `rel_close()`, `static int
passed/failed`, and a final `=== Summary: N passed, M failed ===`.

Fixtures must be **oracle-derived** — record the measured numbers from
`/home/stefan/.hermes/cache/scratch/kd_oracle.py` in comments.

At minimum cover: constructor validation (T<=0, alpha<0, alpha>1 throw); forward
shape `(1,1)` and finiteness; forward vs oracle at T=1/3/7 for `Reduction::Mean`
and `Reduction::Sum`; the `mean != batchmean` distinction pinned by asserting
both the element-mean and the row-mean values differ by exactly K; zero-KL when
student==teacher (tolerance, not `== 0.0`); FD check of the backward for the soft
term with **random** non-uniform logits; the T² factor pinned by asserting the
gradient norm ratio equals T² exactly.

**Step 3: Run to verify it fails.**

```bash
make build/test_knowledge_distillation 2>&1 | tail -20
```
Expected: link failure / undefined reference (no `.cpp` yet) or compile failure.
**This is the RED state — confirm it fails for the right reason.**

---

## Task 2: GREEN — implement the forward

**Objective:** Make the forward-value tests pass.

**Files:**
- Create: `include/nn/utils/knowledge_distillation.cpp`

**Step 1: Implement `distillation_kl`.**

```
p_t = softmax(teacher/T)            // row-wise, max-subtracted
lp_s = log_softmax(student/T)       // shipped helper
per element: p_t * (log(p_t) - lp_s), with the xlogy zero rule:
   if p_t == 0 -> contribution is 0 (NOT p*log(p/q), which is NaN)
reduce: Mean -> divide by (rows*cols); Sum -> no division
multiply by T*T
```

Compute `p_t` and `log(p_t)` from the *same* shifted exponentials so the two
cannot disagree, and clamp `log(p_t)` only where `p_t > 0` so the zero rule above
is applied at the point of use.

**Step 2: Implement `DistillationLoss::forward`.**

`alpha * hard_CE(student_logits, labels)` at **T = 1** plus `(1 - alpha) * distillation_kl(...)`, each reduced with the class's own `Reduction`. `labels` is an `(N,1)` integer column; validate `rows == student_logits.rows`, indices in `[0, cols)`, integral. Reject `teacher_logits` whose shape differs from `student_logits`.

**Step 3: Build and run.**

```bash
make build/test_knowledge_distillation 2>&1 | tail -30
./build/test_knowledge_distillation
```
Expected: forward tests pass, backward tests still fail (not yet implemented).

---

## Task 3: GREEN — implement the backward

**Objective:** Make the FD tests pass.

**Files:**
- Modify: `include/nn/utils/knowledge_distillation.cpp`

**Step 1: Implement `distillation_kl_backward`.**

With `p_t = softmax(teacher/T)` and `q_s = softmax(student/T)`, and `dL/dlp_s` the
adjoint w.r.t. the student's log-probs:

```
dL/dlp_s[i][j] = (reduction_scale) * p_t[i][j]
```

(`Reduction::Mean` → `1/(rows*cols)`). Then push through the shipped helper:

```
dL/d(student/T)[i][j] = dL/dlp_s[i][j] - q_s[i][j] * Σ_k dL/dlp_s[i][k]
```

and finally chain the `1/T`:

```
dL/d(student)[i][j] = T^2 * (1/T) * (the above)
```

The `T²` (from the loss) and `1/T` (from the softmax argument) **must both** be
present; oracle 3 confirms the product is `T²(q−p)/(T·N)` to 1.4e-16 against
torch autograd in float64. Route the last step through `log_softmax_backward`
rather than hand-deriving the softmax Jacobian, so it cannot drift from the
shipped `CrossEntropyLoss`.

**Step 2: Implement `DistillationLoss::backward`.** Combine the soft adjoint with the hard-CE adjoint, scaled by `alpha` and `1-alpha` respectively, honouring `Reduction::None` (per-row output `(rows,1)`).

**Step 3: Build, run the focused suite, and confirm a real FD tolerance.**

```bash
make build/test_knowledge_distillation && ./build/test_knowledge_distillation
```
Expected: all pass. Use `eps = 1e-4` for FD — oracle measurement showed `1e-6` sits
below float32/float64 noise for this loss and produces spurious rel_err ~ 1.9.

---

## Task 4: Mutation testing

**Objective:** Prove the suite is not vacuous.

**Files:**
- Create: `tests/mutate_knowledge_distillation.sh`

Copy the structure of `tests/mutate_temperature_scaling.sh`, which already carries
every defence this repo has learned:

1. Take the baseline ONCE, before any mutation, and **assert it is pristine** by grepping lines that exist only in the correct implementation.
2. Use a **relative** `make` target (`make "build/test_knowledge_distillation"`), never an absolute one — an absolute target reports "up to date" and never rebuilds, which previously produced 7 false `MUT-SURVIVED`.
3. Verify each `sed` applied with `diff -q`; report `MUT-NOOP` separately from `MUT-SURVIVED`.
4. Report compile failure as `MUT-BROKEN`, **never** `MUT-CAUGHT`.
5. Check the binary mtime is newer than the mutated source.
6. **Never** run `make` against the same tree from two places at once.

Mutations to include, one per line, all line-anchored:
- drop the `T*T` factor → must be caught
- drop the `1/T` chain factor → must be caught
- transpose the KL direction → must be caught
- replace the `p_t == 0` zero-rule with the naive `p*log(p/q)` → must be caught
- divide by `rows` instead of `rows*cols` under `Mean` → must be caught
- drop the `xlogy` guard entirely → must be caught

**Run it and record the exact numbers.** If a mutation survives, that is a
question about the test (add a fixture), not an excuse — see the TDD skill's
inert-mutation rule: only mutations that *provably* cannot change anything may be
declared inert, and inertness must be **measured**.

---

## Task 5: Wire into the build

**Files:**
- Modify: `Makefile` (link rule near line 233, `tests:` prerequisite list near line 821/826, `run_tests` line near line 973)
- Modify: `include/nn/nn.h` (add `#include "utils/knowledge_distillation.h"` after the `utils/temperature_scaling.h` line)

**Step 1:** Add the link rule, the `tests:` prerequisite, and the `run_tests` line. **All three** — a missing one is exactly the bug recorded in `NOT_FIXED.md` for `test_mixture_of_softmaxes`, where a suite had a rule and a `run_tests` line but was never built.

**Step 2:** Verify the umbrella header compiles standalone (the repo's recorded check for cross-header collisions):

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

**Step 3:** Warning-clean at the project's real flags:
```bash
g++ -std=c++17 -O2 -march=native -Wall -Wextra -Iinclude -fsyntax-only include/nn/utils/knowledge_distillation.cpp
```

---

## Task 6: Full-suite regression run + docs

**Files:**
- Modify: `EXPANSION_QUEUE.md` (move the item to `## Done` with the findings)
- Modify: `NOT_FIXED.md` only if this run newly broke or newly fixed something

**Step 1:** `make run_tests` and confirm **no new failures** against the
deferred-entry baseline: `test_neural_ode` (ODERNN grad check), `test_wgan_gp`,
`test_vit`, `test_coord_network` are the four known-failing suites and all four
are already documented as deferred. Anything else failing is a regression I
introduced.

**Step 2:** Run the focused suite 3x to confirm determinism.

**Step 3:** Commit atomically, matching the repo's convention
(`feat(scope):`, `test(scope):`, `docs:`), and push to master.

---

## Self-contradiction scan

- Task 1 declares `alpha` as the weight on the **hard** term and the forward sums
  `alpha * hard + (1 - alpha) * soft`. Task 3's backward uses the same two
  coefficients in the same order. Consistent — do not flip the convention.
- Task 4 says T² must be dropped to test the "drop the T*T factor" mutation, while
  Task 3 says T² *and* 1/T must both be present. Consistent: T² is in the forward,
  and the backward's T² is the chain of that same loss factor.
- Semantics 5 (Mean divides by `rows*cols`) and the Task 3 backward
  `dL/dlp_s = scale * p_t` agree: the same `1/(rows*cols)` appears once in the
  forward and once in the backward, and they must not both apply it.