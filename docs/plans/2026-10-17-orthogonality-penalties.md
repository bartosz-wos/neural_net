# Orthogonality penalties — implementation plan

> **For Hermes:** use subagent-driven-development to implement this task-by-task.

**Goal:** ship `include/nn/utils/ortho_penalty.{h,cpp}` — the repo's first
regulariser that acts on the **weights** rather than the loss surface.

**Architecture:** model-free free functions over `Tensor`, so the whole file is
testable with no `Model` and no optimizer (same shape as `utils/vat.h`,
`utils/vqcodebook.h`). Plus one opt-in `OrthoPenaltyLoss` that walks a `Model`'s
parameters.

**Tech Stack:** C++17, `core/tensor.h`, `core/model.h`, `core/layer.h`.

---

## The gap, measured

```
grep -rilE 'ortho' include/nn/layers/generative/   -> 0 hits
grep -rniE 'orthogonality_pen|ortho_pen|off_diagonal|gram' include/ -> no penalty
```

Every existing "orthogonal" hit is a paper name (FAVOR+), an *orthogonalizing
projection* (Muon/Newton-Schulz, `normalization/spectral_norm.h`) or a comment.
The repo has ~250 layers, 42 utils files and **nothing that adds a loss term for
non-orthogonality**. `layers/generative/wgan_gp.h` has a *gradient* penalty —
that is a different quantity and is deliberately **not** duplicated here.

## Citation (verified, not guessed)

Karras, Laine, Aittala, Laine — "Analyzing and Improving the Image Quality of
StyleGAN", arXiv:**1912.04958** (SIGGRAPH 2020, NVIDIA). Verified 2026-10-17 by
scraping `https://arxiv.org/abs/1912.04958` for `citation_title` /
`citation_author` — the arXiv **API was rate-limited** ("Rate exceeded.", 14
bytes) so the API is not available as evidence.

Three guessed IDs were **wrong** and must never be reintroduced:
`1810.01242` (Seyfert 1 WPVS48), `1809.10620` (Operations on Partial Orders),
`1802.01886` (Texygen), `1912.04948` (Drag Force in the Vacuum of Confining
Gauge Theories), `1706.07057` (Collider constraints).

§3.1 (lazy regularisation) is the weight-orthogonality term:

```
L = s * ( mean(W W^T - I)^2 + mean(W^T W - I)^2 ),   s = 1/(r*c)
```

## The oracle caught THREE bugs in my own derivation before any C++ existed

Recorded because the *fingerprint* is what localised each one:

1. **`‖W‖²_F` is wrong.** `mean` of an `r×r` matrix is over all `r²` entries, and
   `Σ_ij (WWᵀ)_ij = ‖Wᵀ1‖²` — the squared **column sums**, not the Frobenius
   norm. Signature: `rel_err ≈ 1–3`, *inconsistent across shapes*. An
   inconsistent error means a wrong **derivation**; a clean constant ratio means
   a harness differentiating the wrong function.
2. **`mean(I_r) = 1/r`, not 1.** So `m1 = (‖u‖² − r)/r²`, not `(‖u‖² − r²)/r²`.
   **Signature that localised it: the `1×N` and `N×1` shapes were the only ones
   that passed** (`rel_err` 1.8e-03 / 2.9e-02) — for a single row the `r` vs
   `r²` error nearly vanishes. A bug that only shows up on non-square weights
   needs non-square fixtures; every "natural" fixture is square.
3. **WGAN-GP's adjoint was missing the `1/N`** from the batch mean — `rel_err`
   0.667 / 0.750 / 0.500 at N = 3 / 4 / 2, i.e. exactly `1 − 1/N`. A clean ratio,
   pinned instantly. (Not shipped here; kept as an oracle-side control.)

After the fix FD agrees to **2.5e-10** worst case over ten shapes including
`(1,5)`, `(5,1)` and `(1,1)`.

## The adjoint (hand-derived, FD-verified)

```
u_k = Σ_i W[i][k]  (column sums, length c)     m1 = (‖u‖² − r)/r²
t_i = Σ_k W[i][k]  (row sums, length r)        m2 = (‖t‖² − c)/c²
dL/dW[i][k] = 4s * ( m1·u_k/r²  +  m2·t_i/c² )
```

The gradient is a sum of two rank-1 terms: a **column** term with every row
equal, and a **row** term with every column equal. Verified structure:
`max |g[i][k] − (a_i + b_k)| ≈ 1e-17`.

## THREE predictions the oracle FALSIFIED — record, do not assert

| Prediction | Measured |
|---|---|
| `g == gᵀ` for square `W` | **FALSE**, `max\|g − gᵀ\| ≈ 1e-1` |
| `L(2W) = 4·L(W)` | **FALSE**, ratio → **16** (`s` is *not* symmetric: `m1` and `m2` use different `r`/`c`) |
| `grad ≠ 0` at orthogonality | **FALSE**, **exactly 0** — orthogonality is a genuine *stationary point*, the global minimum |

The third is the most load-bearing property in the file and gets its own test.

## Semantics to pin (D1–D12), each with a named test

- **D1** `s = 1/(r·c)` (mean over **all** elements), not `1/r²` per term.
- **D2** The penalty is **0 exactly** and the gradient is **exactly 0** at an
  orthogonal `W` (e.g. `I`, a QR factor, a permutation).
- **D3** The mean is over **modes**: for rank > 2 the matrix is
  `(shape[-2], shape[-1])` per leading index. Folding the channel axis into the
  rows mixes modes and is the single easiest mistake — pinned by a fixture where
  the two disagree.
- **D4** The scale `s` is applied **once**, to the sum.
- **D5** `m1` uses `r` (rows), `m2` uses `c` (cols) — not interchangeable on a
  non-square `W`.
- **D6** Both terms are **squared means**, not means of squares: `mean(A)²` vs
  `mean(A²)` differ for `A` with mixed signs.
- **D7** Rank-1 (`1×C`, `R×1`) weights are **legal** and must not throw — they
  are exactly the shapes that exposed bug 2.
- **D8** Empty tensors (`0×C`) **throw**.
- **D9** `ortho_penalty_gradient` returns the same shape as its input and is a
  fresh tensor (never aliases).
- **D10** `λ` scaling: `λ·L` and `λ·grad`, `λ=0` → both exactly 0.
- **D11** The **one-sided** residual `|sv₁(W) − 1|`, not two-sided. Power
  iteration is a **fixed number of steps** (deterministic), and `1/sv₁` throws
  on `sv₁ == 0` rather than returning `inf`.
- **D12** Per-**slice** mode (each 2-D slice penalised separately, mean over
  slices) is a **separate function** from per-**mode** — not a flag, so a
  caller cannot get it by accident.

## Tasks

### Task 1: header + skeleton (RED first)
Write `tests/test_ortho_penalty.cpp` with test 1 (penalty value on a
hand-checkable fixture), build, **watch it fail** with undefined references.

### Task 2: `ortho_penalty` (GREEN)
2-D value only. Exact oracle values for a 4×6 non-square fixture.

### Task 3: `ortho_penalty_gradient` (GREEN)
Hand-derived adjoint, asserted elementwise against the oracle's 17-digit output
**and** against centred FD computed in the test itself.

### Task 4: rank > 2 (`ortho_penalty_modes`, `ortho_penalty_slices`)
D3, D12.

### Task 5: `ortho_penalty_lam`, `ortho_penalty_one_sided`, `ortho_spectral_norm`
D10, D11.

### Task 6: `OrthoPenaltyLoss` over a `Model` + end-to-end training test
With **liveness assertions**: the penalty must be non-zero at init and must
**decrease**, while the task loss also decreases. Per the adversarial-session
lesson, assert the orthogonality residual actually improves — a comparison that
fails identically at every hyperparameter means the fixture is degenerate.

### Task 7: mutation harness
12–14 mutations, with all of this repo's defences: `print-lib-objs` (never parse
`LIB_OBJS` with a regex — it is a *recursive* make variable), `grep -qF` for
anchors, checksum before/after to detect `MUT-NOOP`, atomic `cp && mv` restore,
`MUT-BROKEN` never `MUT-CAUGHT`, `MUT-CRASH` its own column, verdicts counted
from `[FAIL]` lines, and a re-verify of the real binary on exit.

### Task 8: wire up + full suite + docs
`nn.h` include, Makefile link rule + `tests:` prerequisite + `run_tests` line,
`docs/plans/` commit, `EXPANSION_QUEUE.md` → `## Done`.