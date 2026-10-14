# Randomized Smoothing / Certified Robustness Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/randomized_smoothing.{h,cpp}` — the paper's
`SAMPLEUNDERNOISE`, `PREDICT`, `CERTIFY`, Theorem 1's certified radius, the
Clopper–Pearson and binomial-test statistics they rest on, Theorem 6's certified
accuracy bound, and the Gaussian-noise training objective.

**Architecture:** Model-free core. Every estimator is a pure function over
`(Tensor, seed)` or over statistics; the only callback is a
`std::function<int(const Tensor&)>` base-classifier oracle, exactly the shape
`adversarial.h` already uses for `GradientFn`. The `Model`-driven path is a thin
adapter at the bottom of the file. This makes the statistics unit-testable with no
network and makes the layer composable with any classifier in the repo.

**Tech Stack:** C++17, `std::erfc`, continued-fraction incomplete beta, bisection
inverses, `std::mt19937` + explicit Box–Muller. No new dependencies.

---

## Verified sources (read this session, not from memory)

- **arXiv ID corrected.** `EXPANSION_QUEUE.md` proposed **1612.00483**. The
  arXiv API returns *"A Large-Scale Spectroscopic Survey of Methanol and OH
  Line Emission from the Galactic Center"* (Cotton & Yusef-Zadeh, astro-ph.GA)
  for that ID — an astrophysics paper. The real ID is **1902.02918**, verified via
  `id_list=1902.02918` against the arXiv API on 2026-10-14:
  Cohen, Rosenfeld, Kolter, *Certified Adversarial Robustness via Randomized
  Smoothing*, **ICML 2019** (`<arxiv:comment>ICML 2019`).
- Full PDF fetched and text-extracted (36 pages, PyMuPDF) to
  `~/.hermes/cache/scratch/rs.txt`. Section 3.2 pseudocode (p. 6), Appendix B
  (Prop. 3–5), Appendix C (Prop. 1–2), Appendix D (Theorem 6) all read directly.
- The official code (`github.com/locuslab/smoothing`, cited p. 33) is **404** —
  both `YangSong14/Randomized-Smoothing` and the `locuslab` org are gone. The
  math is therefore taken from the paper text, and the statistics are validated
  against **scipy 1.18.1 / statsmodels 0.15.0** instead.

## Semantics pinned (each is a natural mistake; each gets a named test)

| # | Decision | Evidence |
|---|----------|----------|
| D1 | Noise is **added** to `x` and is `N(0, sigma^2 I)`; `sigma` is the **standard deviation**, not the variance | Eq. (1), p. 4: `eps ~ N(0, sigma^2 I)` |
| D2 | Radius is **`sigma * Phi^-1(p_lower)`** and requires **`p_lower > 1/2`**, else abstain | §3.1 Thm 1 + §3.2 pseudocode `if pA > 1/2` |
| D3 | `PREDICT`'s hypothesis test uses **only the top two counts**, with `n = nA + nB` (NOT the full `n`) | Pseudocode `BINOMPVALUE(nA, nA+nB, 0.5)` |
| D4 | Abstain iff `p_value > alpha`; the test is **two-sided** at `p = 0.5` | Pseudocode + Appendix C |
| D5 | `CERTIFY` uses **two disjoint sampling rounds**: `n0` to guess the class (those samples are **discarded**), then `n` for the count | Appendix C.2: "first, use `n0` samples ... to take a guess"; the bound is formed from the second round only |
| D6 | The Clopper–Pearson lower bound is **`Beta^-1(alpha; k, n-k+1)`** | see the trap below |
| D7 | At **`k = 0`** the bound is exactly **`0.0`** (scipy returns `nan`: `Beta(0,·)` is degenerate) | measured |
| D8 | Thm 1's general form is `sigma/2 * (Phi^-1(pA) - Phi^-1(pB))`; it collapses to `sigma*Phi^-1(pA)` **only because `pB = 1-pA`** is assumed | §3.1 Thm 1 vs App. C.2 derivation |
| D9 | Identical `seed` ⇒ bit-identical noise ⇒ bit-identical counts | the suite asserts it |

### D6 is a trap in the paper's own appendix

Appendix C.2 tells the reader to use

```python
proportion_confint(k, n, alpha=2*alpha, method="beta")[0]
```

Reading statsmodels 0.15.0 `proportion.py:208-210` shows that with the default
`alternative="two-sided"` it **halves alpha internally** (`alpha = alpha / 2.0`),
and the `"beta"` branch is literally `stats.beta.ppf(alpha, k, n-k+1)`
(`:271`). So `2*alpha` cancels the internal halving and the real formula is
`Beta^-1(alpha; k, n-k+1)` — the correct one-sided `(1-alpha)` Clopper–Pearson
bound. Verified by its defining property: at that point
`P(X >= k) == alpha` exactly (measured `0.0500000000` for every fixture at
`alpha=0.05`; the `2*alpha` form gives `0.1000000000`). Ship the formula, and
record the trap in the header.

## API (public surface)

```cpp
// --- statistics (model-free) ---
double normal_cdf(double x);
double normal_quantile(double p);                       // Phi^-1, bisection on erfc
double clopper_pearson_lower(int k, int n, double alpha);
double binom_pvalue_half(int n_a, int n_b);             // two-sided at p=0.5
double certified_radius_from_probs(double sigma, double p_a, double p_b);
double certified_radius(double sigma, double p_lower); // symmetric, D8
double certified_accuracy_lower_bound(int y, int m, double alpha, double rho); // Thm 6

// --- the paper's Monte-Carlo routines ---
using BaseClassifierFn = std::function<int(const Tensor&)>;   // returns class index
std::vector<int> sample_under_noise(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples, unsigned seed = 0);

struct SmoothedPrediction { bool abstained; int class_index; int n_a; int n_b;
                            double p_value; };
SmoothedPrediction smoothed_predict(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples, double alpha,
                                    unsigned seed = 0);

struct SmoothingConfig { double sigma = 0.25; int num_guess_samples = 100;
                         int num_samples = 1000; double alpha = 0.001;
                         unsigned seed = 0; };
struct CertifiedPrediction { bool abstained; int class_index; int count;
                             double p_lower; double radius; };
CertifiedPrediction certify(const BaseClassifierFn& f, const Tensor& x,
                            const SmoothingConfig& cfg);

// --- the Model-driven adapter (the outer minimisation, §4 / Appendix J.2) ---
void smoothing_train(Model& model, const Tensor& X, const Tensor& y,
                     const SmoothingConfig& cfg, double learning_rate, int epochs);
```

Defaults are the paper's own CIFAR-10 setting: `sigma = 0.25`, `alpha = 0.001`
(p. 30, §4.2), `n = 1000` (Fig. 8 right, n up to 10,000), `n0 = 100` (App. J.2).

## Oracle provenance

Every expected value in the test file is transcribed from
`~/.hermes/cache/scratch/rs_oracle_final.py` (kept in-repo as
`tests/randomized_smoothing_oracle.py`) and was verified against scipy:

- `Phi` / `Phi^-1` — my C++ kernels were ported and checked against scipy to
  `rel <= 4.5e-16` across `p ∈ [1e-12, 1-1e-12]`. **Decision:** invert `Phi` by
  **bisection on `std::erfc`**, not by porting Wichura's AS241 — the AS241
  rational fit loses ~1e-6 relative accuracy in the tail, and bisection costs
  ~60 evaluations of `erfc`.
- CP lower bound — my continued-fraction `I_x(a,b)` + bisection inverse matched
  scipy `beta.ppf` to **1.8e-13 worst-case relative** over all 12 fixtures.
- Two-sided binom p-value — `min(1, 2*min(P(X<=k), P(X>=k)))` matched
  `scipy.stats.binomtest` **exactly** (`diff = 0.000e+00`) on all 10 fixtures.
- Theorem 6 — the PDF's fraction layout is mangled; the **algebra came from its
  proof** (p. 24), not from the displayed equation.
- Appendix B Prop. 3/4 — gives a **closed-form ground truth**:
  `g == f` and the certified radius converges to `|w.x + b| / ||w||`. This is the
  headline test because it needs no oracle at all.

## Tasks

### Task 1: RED suite
`tests/test_randomized_smoothing.cpp` covering D1–D9 + Theorem 6 + Prop. 3/4 +
the vacuity guards. Must compile and fail (no header yet).

### Task 2–6: implementation, in order
`normal_cdf`/`normal_quantile` → `clopper_pearson_lower` (regularized incomplete
beta + bisection) → `binom_pvalue_half` → `sample_under_noise` (mt19937 +
**explicit Box–Muller**, so the stream is reproducible across libstdc++ versions)
→ `smoothed_predict`/`certify` → `certified_accuracy_lower_bound` →
`smoothing_train`.

### Task 7: mutation harness
`tests/mutate_randomized_smoothing.sh`, carrying the six defences this repo has
had to learn the hard way (pristine-baseline assertion, relative make targets +
mtime guard, `diff -q` no-op check, `MUT-BROKEN` ≠ `MUT-CAUGHT`, per-mutation
build from explicit sources, and an *inverted-verdict* probe for any mutation
claimed inert).

### Task 8: wiring
`Makefile` (link rule + `tests:` prereq + `run_tests` line) and `include/nn/nn.h`.

## Non-vacuity guards (learned the hard way in this repo)

- A fixture must be able to FAIL. Each "these two must disagree" assertion is
  kept deliberately.
- `N >= 2` anywhere a reduction divides by a count.
- No `std::uniform_real_distribution`/`std::normal_distribution` — the
  distribution objects are not portable across standard libraries, so
  reproducibility would not survive a toolchain change.