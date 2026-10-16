# VQ Codebook Maintenance Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship codebook-collapse mitigation as a first-class utility —
k-means codebook initialisation, an LRU dead-code replacement policy, and the
paper's *symmetric* commitment loss — as `utils/vqcodebook.{h,cpp}`, so the
`dead_code_count()` that `layers/generative/vq_vae.h` already exposes becomes
something the repo can actually act on.

**Architecture:** Every numerical core is a **free function** over `Tensor`s
(`vqcb_kmeans_initialise`, `vqcb_lru_replace`, `vqcb_lru_tick`,
`vqcb_committedness_loss`, `vqcb_committedness_gradient`, ...), exactly the
model-free shape `utils/vat.h` and `utils/randomized_smoothing.h` were built
with. Two thin `Layer` adapters (`LRUCodebookMaintenance`, `KMeansInitQuantizer`)
sit on top for the drop-in path. k-means reuses the **already-shipped**
`KMeans` class in `utils/gmm.h` rather than a second implementation.

**Tech Stack:** C++17, the repo's `Tensor`/`Layer`/`Dense`, g++ with the
project's real `-O2 -march=native -Wall -Wextra`. Oracle: numpy 2.5.4 float64 +
scikit-learn 1.9.x in `~/.hermes/cache/scratch/.venv-oracle`.

---

## Sources, and what each one settles

| Source | What it fixes |
|---|---|
| `arxiv.org/abs/2305.08842` — **verified via arXiv API 2026-10-16**, returns *"Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks"*, authors Huh, Cheung, Agrawal, Isola | Existence + authorship. |
| Paper Eq. (8)–(9), PDF p.4 | The **committedness** loss, verbatim below. |
| Paper §3, PDF p.4 | The **rule of thumb**: `α = 10`, `β = 0.9`, with `d_l2(ze,zq) := ½‖ze−zq‖²₂`. |
| Paper §3 "Replacement policy", PDF p.5 | LRU policy: *"if the code does not get used for 20 training iterations, it gets replaced with a random embedding vector"*, lifespan 20, *"the active codes are left unchanged"*. |
| Paper §3.1 Eq. (13) | Why collapse is *not* recoverable by gradient: the loss is computed over `P_z` but minimised w.r.t. `C_z`, so `C_z \ Q_z` **receives no gradient at all**. This is why a *replacement* policy is needed rather than more gradient steps. |
| Paper §3.2 Eq. (14) | The gradient gap — the reason "just train longer" is not the fix. |
| Paper Table 1 | Empirical ranking: `Replace (LRU)` 99.8 active / 69.3 acc, beating `Rep. KM` 99.2/63.7 and `None` 81.4/67.0. **LRU is the paper's own recommendation.** |
| `utils/gmm.h` (shipped) | `KMeans` exists: Lloyd + k-means++, `fit()` returns inertia, `cluster_centers()` is `(K, d)`. Reused, not reimplemented. |

**Citation check — three guesses were WRONG and are recorded here so they are
never reintroduced.** `1911.05863` is *"Regularity results for a nonlinear
elliptic-parabolic system with oscillating coefficients"*; `1912.04997` is a
PGL₂ representation-theory thesis; `2107.03312` returned 0 results. `Reviving
Dead Codes in VQ-VAE` and `Input Sanitization for VQ-VAE` return **0 arXiv
results** by title search — do not cite them. This queue has now produced
**eight** wrong IDs.

### The one sentence that matters most

> **The shipped `vq_vae.h` implements van den Oord's loss; this paper replaces
> it with a loss that is NOT the same function, and the two must be
> distinguishable.**

Van den Oord Eq. (3): `L = ‖sg[z_e] − e‖² + β‖z_e − sg[e]‖²` — the codebook
term is **unweighted** and β rides on the commitment term.

Huh et al. Eq. (8)–(9) — *committedness*:

```
L_cmt(z_e, z_q) = (1 − β) · d(z_e, sg[z_q])     <- encoder pulled to the code
                + β      · d(sg[z_e], z_q)     <- codebook pulled to the encoder
```

**β is on the OPPOSITE term.** At the paper's recommended `β = 0.9` the two
losses are not a rescaling of each other: β=0.9 weights the codebook term 9×
more than van den Oord's formulation does at any β. A "β just moved to the
other term" implementation that still used `α·L` with van den Oord's `β=0.25`
would agree with this on no fixture. The existing `vq_codebook_gradient_with_beta`
(the deliberately-wrong variant used by the D1 discriminator) is the *exact*
functional form of the committedness codebook gradient, so the new function is
checked against it for equality — a free consistency proof.

### Semantics pinned (each is a natural mistake; each gets a named test)

- **D1 — β weights the CODEBOOK term here, the reverse of `vq_vae.h` D1.**
  β=0.9 default, `α=10` exposed separately.
- **D2 — `d(·)` is `½‖a−b‖²₂`, NOT `‖a−b‖²₂`.** The paper defines it
  explicitly on p.4. Dropping the ½ is a clean factor-2 that a mean-based
  assertion can hide; pinned by comparing to the squared form directly.
- **D3 — the mean is over all `M·D` elements**, consistent with `vq_vae.h` D2,
  so the two implementations' `β` are directly comparable.
- **D4 — `sg[·]` goes on the argument that is being PULLED TOWARD the other.**
  Term 1 pulls `z_e` toward `z_q`, so `z_q` is stopped. Term 2 pulls the code
  toward `z_e`, so `z_e` is stopped. Inverted, the loss still evaluates to the
  same scalar and every value test passes.
- **D5 — LRU replaces only codes UNUSED for `lifespan` iterations; active codes
  are byte-identical afterwards.** "Active codes are left unchanged" (paper §3).
  A test asserting this pins that the policy cannot silently perturb working
  codes.
- **D6 — lifespan semantics are `>= lifespan`, and a code used on the current
  iteration resets to 0.** Off-by-one here means a policy that fires one
  iteration early or late; pinned with lifespan=1 and lifespan=2 fixtures.
- **D7 — replacement vector comes from the HIGH-ERROR encoder rows**, not from
  a uniform random draw. The paper's own words for Zeghidour et al. are
  "randomly sampled embedding", but §3's replacement-policy analysis and the
  Input-sanitization family both move high-error inputs in; the shipped default
  is high-error selection with the uniform draw available as an option, and the
  two must be separately testable.
- **D8 — k-means init requires `M >= K`; with `M < K` it throws.** The shipped
  `KMeans::fit` already throws `n_samples < n_clusters`. Silent zero-padding
  would produce a degenerate codebook.
- **D9 — `k_means++` seeds are drawn from a SEEDED `std::mt19937`, so init is
  reproducible for a given seed** and the suite asserts determinism across runs.
- **D10 — LRU accounting is per-`update`-pass, and `lru_tick` must be called
  once per iteration by the caller**, not once per row.

---

## Out of scope (recorded, not silently dropped)

- **Affine re-parameterisation of the code vectors** (paper §4.1) — changes the
  quantizer's parameterisation and is a larger structural change than a utility.
- **Alternating optimisation** (paper §4.2) — a training-loop-level scheduler.
- **VQ-VAE-2 residual quantisation** (`1906.00446`, verified real) — a
  different model; the queue lists it separately.
- **Gumbel/stochastic relaxed discrete latents** — `layers/utility/
  gumbel_softmax.{h,cpp}` already exists; a composition, not a build.

---

## Task 1: k-means codebook initialisation

**Objective:** `vqcb_kmeans_initialise(z, K, seed, max_iter)` returns a `(D, K)`
codebook whose columns are k-means centres of the rows of `z`.

**Files:**
- Create: `include/nn/utils/vqcodebook.h`, `include/nn/utils/vqcodebook.cpp`
- Test: `tests/test_vqcodebook.cpp`

**Step 1: Write failing test** — fixture with 3 well-separated clusters of 4
rows each in 2-D, assert (a) each returned column equals one cluster's mean to
1e-12, (b) two calls with the same seed are bit-identical, (c) `M < K` throws,
(d) the result is `(D, K)` — **column k is `e_k`**, matching `vq_vae.h` D3.

**Step 2: Run to verify failure** — `make build/test_vqcodebook` → link error
(symbol not found), i.e. the header does not exist.

**Step 3: Minimal implementation** — wrap the shipped `KMeans`:
`KMeans km(K, max_iter, 1e-4, seed); km.fit(z);` then transpose
`km.cluster_centers()` (which is `(K, D)`) into `(D, K)`. The transpose is the
one genuinely easy-to-get-wrong step and gets its own assertion.

**Step 4: Verify pass**, then **Step 5: commit**.

---

## Task 2: LRU dead-code replacement policy

**Objective:** `LRUCodebookTracker` + `vqcb_lru_replace(w, ages, rows_by_error)`
replace exactly the codes whose age `>= lifespan`.

**Step 1: Write failing test** — 5 codes, ages `{0, 3, 21, 7, 25}`, lifespan 20.
Assert codes 2 and 4 are replaced (ages ≥ 20), codes 0/1/3 are **bit-identical**
(D5), and the returned age vector is zeroed **only** at the replaced slots.

**Step 2: verify it fails**, **Step 3: implement**, **Step 4: verify**,
**Step 5: commit**.

**Design note:** `ages` is a `(K)` tensor owned by the caller, so the policy
composes with `VectorQuantizerEMA::cluster_size()` (also `(K)`) without a
conversion. Replacement source rows are chosen by descending reconstruction
error (D7), ties broken by **lowest row index** — a tie-break must be pinned
because no tolerance test can observe one otherwise.

---

## Task 3: The committedness loss (Eq. 8/9)

**Objective:** `vqcb_committedness_loss(z, z_q, beta)` and
`vqcb_committedness_gradient(z, z_q, beta)` implementing Huh et al. Eq. (8)–(9).

**Step 1: Write failing test** — pin all four of:
- at `β = 1` the loss equals the **codebook** term of `vq_vae.h` (the paper's own
  Appendix A.2 claim that EMA ≡ commitment loss at β=1, checked as a value);
- the gradient at `β = 1` equals `vq_codebook_gradient_with_beta(z, w, 1.0)`
  **bit-for-bit** (the consistency proof from §"the one sentence");
- at `β = 0` the loss is the encoder-side term only;
- **D2**: `d` is `½‖·‖²`, asserted against `0.5 * mean((z-z_q)^2)` computed
  independently in the test.

**Step 2: verify failure**, **Step 3: implement**, **Step 4: verify**,
**Step 5: commit**.

---

## Task 4: Layer adapters + umbrella/Makefile wiring

**Objective:** `LRUCodebookMaintenance` (a `Layer` that owns the ages and the
reinit policy) and `KMeansInitQuantizer` (a quantizer that initialises its
codebook by k-means on the first forward).

**Files:**
- Modify: `include/nn/nn.h` (add the include, alphabetically near the utils block)
- Modify: `Makefile` — link rule, `tests:` prerequisite, `run_tests` line

**Step 1:** Write the failing test (layer adapter behaviour + a full-suite
aggregate check that `nn.h` still compiles standalone).
**Step 2:** verify it fails to link.
**Step 3:** implement + wire.
**Step 4:** `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
— **forward dependency check**, per the writing-plans rule about scaffolding
tasks that reference paths created later.
**Step 5:** commit.

---

## Task 5: Oracle, mutation harness, aggregate verification

**Objective:** A numpy/sklearn oracle that self-checks against centred finite
differences, plus a mutation harness carrying all defences this repo has
accumulated.

**Oracle** (`~/.hermes/cache/scratch/vqcb_oracle.py`): k-means init compared to
`sklearn.cluster.KMeans`; the committedness loss and both gradients compared to
analytic `numpy` **and** self-checked by centred FD with the two stop-gradient
functions kept separate — the trap the VQ-VAE session recorded, where
differentiating the combined scalar gives a clean `(1+β)` ratio.

**Harness** (`tests/mutate_vqcodebook.sh`), all defences mandatory:
1. baseline snapshotted **once, before any mutation**, and asserted pristine by
   grepping markers unique to the correct implementation;
2. every `sed` verified applied via `diff -q` → `MUT-NOOP` reported separately
   from `MUT-SURVIVED`;
3. **relative** `make` target + mtime guard (absolute targets silently report
   "up to date" — this produced 7 false survivors in the metrics session);
4. compile failure → `MUT-BROKEN`, **never** `MUT-CAUGHT`;
5. `MUT-CRASH` counted as its own column (SIGABRT prints `Aborted`, which is a
   *bash* message a C++ program never emits);
6. verdicts counted from `[FAIL]` lines, never a summary string;
7. `LIB_OBJS` extraction must match `:=` **with a space** — a regex written for
   `=` returned an EMPTY link set and scored all 8 real mutations as "did not
   link"; the count is asserted `>= 10`;
8. atomic restore (`cp base tmp && mv -f tmp src`);
9. every survivor gets a **probe** before being called inert; an unexplained
   survivor is not a gap.

**Verification:** full `make run_tests` aggregate — 246+ suites, expecting no
new failures beyond the ODERNN entry already deferred in `NOT_FIXED.md`.

---

## Self-contradiction scan

Checked: the D1/D7 statements here agree with the Task descriptions; β's
placement (D1) is stated identically in "one sentence" and Task 3; the
lifespan threshold is `>=` in both D6 and Task 2. No stale duplicate lists.