# Fix the Jacobi eigendecomposition in Shampoo / SOAP / GaLore — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** The three matrix-preconditioner optimizers compute an eigendecomposition that is not an eigendecomposition — the returned eigenvectors satisfy `A·Q ≠ Q·Λ` with an O(1) residual, and the eigenvalues are wrong by up to 2.95. Fix the rotation angle and pin the invariant with a test that cannot pass vacuously.

**Architecture:** `jacobi_eigendecompose` is a `static` private member duplicated verbatim in three optimizers. The defect is one expression in one line of each copy: the general-branch rotation angle has a flipped sign. Fix all three copies in place (they are private statics; a shared helper would be a repo-wide refactor this task is not scoped to make). Add a dedicated test suite that exercises the **public** preconditioner state (`get_U_L`, `get_Q_L`, `get_P`) rather than the private function, so the test pins observable behaviour.

**Tech Stack:** C++17, the repo's `Tensor`, `make tests` / `make run_tests`.

---

## Background: the measured defect

All three optimizers build a Kronecker-factored preconditioner from the eigenvalues of `L = G·Gᵀ` and `R = Gᵀ·G` (Shampoo, SOAP) or of `G_EMAᵀ·G_EMA` (GaLore). Each needs a **symmetric eigendecomposition**, and each carries its own private copy of a cyclic Jacobi rotation routine.

The rotation angle is computed as:

```cpp
theta = 0.5 * std::atan2(2.0 * apq, aqq - app);   // shampoo.cpp:166, soap.cpp:185, galore.cpp:196
```

The update formulas in the same function are:

```
new_app = c²·app + 2cs·apq + s²·aqq
new_aqq = s²·app − 2cs·apq + c²·aqq
```

These are the formulas for `A ← JᵀAJ` with `J` the rotation that **annihilates** `apq`. The angle that annihilates it is `θ = ½·atan2(2·apq, app − aqq)`. The repo's expression has `aqq − app`, i.e. the **sign of the second argument is flipped**, which makes the rotation *double* the off-diagonal rather than zero it.

### The decisive measurement (not a reading of the code)

The repo's own function contains a **degenerate branch** for `app == aqq` that computes `θ = ±π/4` — and that branch is CORRECT. So the function is internally inconsistent: correct on the measure-zero branch, wrong on the general branch. Measured with a probe that computes the post-rotation off-diagonal directly:

```
DEGENERATE app==aqq     repo: offdiag_after=+1.554e-16   fix: offdiag_after=+1.554e-16
GENERAL  app!=aqq       repo: offdiag_after=-1.269e+00   fix: offdiag_after=+1.110e-16
```

The repo leaves an off-diagonal of **−1.269** where a Jacobi rotation must leave zero. That is not a rounding error; it is the wrong branch of `atan2`.

### The convention-free ground truth

"Are the columns of `Q` actually eigenvectors of `A` with the returned eigenvalues?" is the definition of an eigendecomposition and carries no convention about which formula a reference implementation uses. Measured on the repo's **real** source (extracted from the `.cpp` files, not hand-transcribed) against `numpy.linalg.eigvalsh`:

| variant | max eigen error | `‖A·Q − Q·Λ‖` | off-diag after |
|---|---|---|---|
| repo as-is | **2.946** | **O(1)** | 0 |
| one-character θ fix | 5.3e-15 | **2.2e-16** | 0 |

The repo **does** diagonalize `A` (off-diagonal → 1e-16, trace preserved to 8.9e-16) — it converges to *some* orthogonal similarity, but the eigenvalues it reports are not the eigenvalues of `A`, and `Q`'s columns are not eigenvectors. That is why the defect is invisible to every existing test: they check orthonormality of `Q` (which holds, because `Q` is a product of rotations) and Frobenius-norm preservation (which also holds), but never the one thing that matters.

---

## Task 1: RED — a test that cannot pass vacuously

**Objective:** Write the failing test first. It must assert the *eigenvector* property, not merely orthonormality, and it must exercise the **public** API so it stays valid if the implementation is refactored.

**Files:**
- Create: `tests/test_jacobi_eig.cpp`

**Step 1: Write the failing test**

The test builds a `Dense` layer with a **non-diagonal** gradient so that `L = G·Gᵀ` is a genuinely non-diagonal symmetric matrix (a diagonal `G` gives a diagonal `L`, whose eigendecomposition is the identity and cannot discriminate anything), steps the optimizer once, then checks:

1. **`A·Q = Q·Λ`** for the returned eigenbasis — the eigenvector property. This is the assertion that fails today.
2. **Eigenvalue correctness** against an independent closed-form reference computed in the test (for a 2×2 symmetric matrix the eigenvalues are the roots of `λ² − tr·λ + det`, computed with `std::sqrt` — no library eigensolver needed).
3. **The degenerate branch still works**: `app == aqq` with `apq ≠ 0` must still annihilate the off-diagonal (this pins the branch that was already correct, so a future "simplification" of the angle cannot break it).
4. **Non-vacuity**: assert the fixture's `L` is genuinely non-diagonal (`L(0,1) ≠ 0`) — otherwise a broken implementation would pass by accident.
5. **GaLore**: the projection matrix `P` must consist of actual eigenvectors of `G_EMAᵀ·G_EMA`, i.e. `P` spans the dominant eigenspace — checked by `Pᵀ·(A·P) == Λ·(Pᵀ·P)`.

Write a self-contained 2×2 symmetric eigendecomposition reference *inside the test file* (closed form) so the test does not depend on the implementation being tested.

**Step 2: Verify RED**

Run: `make build/test_jacobi_eig && ./build/test_jacobi_eig`
Expected: **FAIL** on the eigenvector-property assertions, with a residual of order 1 (not a rounding error). The degenerate-branch assertions should pass — they are already correct.

**Step 3: Commit**

```bash
git add tests/test_jacobi_eig.cpp Makefile
git commit -m "test(optimizers): RED test for the Jacobi eigendecomposition invariant"
```

---

## Task 2: GREEN — fix the rotation angle in all three copies

**Objective:** Change the general-branch rotation angle from `atan2(2·apq, aqq − app)` to `atan2(2·apq, app − aqq)` in each of the three private copies.

**Files:**
- Modify: `include/nn/optimizers/shampoo.cpp:166`
- Modify: `include/nn/optimizers/soap.cpp:185`
- Modify: `include/nn/optimizers/galore.cpp:196`

**Step 1: The fix**

In each file, change exactly one line:

```cpp
// BEFORE
theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
// AFTER
theta = 0.5 * std::atan2(2.0 * apq, app - aqq);
```

Also update the comment above it (in `shampoo.cpp` the comment says
`// Standard Jacobi formula: theta = (aqq - app) / (2 * apq).` — which is
itself the wrong formula; the standard formula is `(app - aqq) / (2·apq)`).

**Step 2: Verify GREEN**

Run: `make build/test_jacobi_eig && ./build/test_jacobi_eig`
Expected: **PASS**, with the eigenvector residual at machine precision (≤ 1e-12 relative).

**Step 3: Run the affected optimizers' own suites**

```bash
make build/test_shampoo build/test_soap build/test_galore
./build/test_shampoo && ./build/test_soap && ./build/test_galore
```

Expected: all three still pass. **If any of them fails, that is the finding** — it means an existing test pinned the wrong behaviour, and the test must be read before changing it (see Task 4).

**Step 4: Commit**

```bash
git add include/nn/optimizers/shampoo.cpp include/nn/optimizers/soap.cpp include/nn/optimizers/galore.cpp
git commit -m "fix(optimizers): correct the Jacobi rotation angle in Shampoo, SOAP and GaLore"
```

---

## Task 3: Full-suite regression

**Objective:** Confirm the fix changes nothing else.

**Step 1: Run the full suite**

```bash
make run_tests 2>&1 | tail -40
```

Expected: exactly the same failures as the pre-change baseline — **only `test_neural_ode`**, which is the entry already deferred in `NOT_FIXED.md` and was measured identically on a clean clone of the pre-change commit. Any NEW failure is a regression and must be investigated before committing further.

**Step 2: Sanitizers**

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude -o /tmp/test_jacobi_eig_asan \
    tests/test_jacobi_eig.cpp build/**/*.o 2>/dev/null || \
  make CXXFLAGS="-std=c++17 -O1 -g -fsanitize=address,undefined -Wall -Wextra -march=native -MMD -MP" build/test_jacobi_eig
```

Run it; expect clean output.

**Step 3: Commit** (if the Makefile needed a rule)

```bash
git add Makefile
git commit -m "chore: wire the Jacobi eigendecomposition test into the build"
```

---

## Task 4: Record the finding

**Objective:** The repo's `NOT_FIXED.md` and `EXPANSION_QUEUE.md` are the institutional memory of this project — a bug found and fixed must be recorded there, or the next session will re-derive it.

**Files:**
- Modify: `NOT_FIXED.md` — the entry is removed if it was listed (it was not; this is a NEW finding), so instead record the *fixed* state in `EXPANSION_QUEUE.md` under `## Done`.
- Modify: `EXPANSION_QUEUE.md` — add a `## Done` entry describing the defect, the measurement, and the fix.

Record specifically:
- The defect: the sign flip in the general branch of the rotation angle, in three duplicated private copies.
- The measurement that found it: the convention-free `A·Q = Q·Λ` residual, and the internal-inconsistency probe (degenerate branch correct, general branch wrong).
- Why every existing test missed it: they asserted orthonormality and norm preservation, both of which hold for a product of rotations that is not an eigenbasis.
- The verified-absent finding that motivated the session: **15 generative models ship and there is no sample-quality metric** (`fid|inception|frechet` → 0 hits tree-wide).

---

## Notes for the implementer

- **Do not "fix" the tests.** If an existing test in `test_shampoo`/`test_soap`/`test_galore` fails after the change, read it first. The likely candidate is a test that pinned a value derived from the broken eigenvalues. Per the repo's own rule: a failing comparison is evidence about one of two pieces of code, not proof the implementation is wrong.
- **The degenerate branch is correct.** Do not "simplify" `theta` to a single formula — the `app == aqq` case needs `±π/4` to avoid a 0/0 in `atan2`. The new test pins that branch separately.
- **Three copies, three edits.** `grep -rn "atan2(2.0 \* apq" include/nn/optimizers/` must return exactly three hits afterwards, all with `app - aqq`.
- **Do not refactor the three copies into one shared helper.** It would be the DRY choice, but it touches three optimizers' private state layout and is a separate change from the bug fix. Note it as future work instead.
