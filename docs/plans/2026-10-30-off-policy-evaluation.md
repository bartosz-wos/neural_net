# Off-Policy Evaluation — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship a model-free off-policy evaluation utility (`include/nn/utils/off_policy_eval.{h,cpp}`) that estimates a target policy's value from logged bandit data, with the importance-weighting, direct-method and doubly-robust estimators, plus the effective-sample-size diagnostic that decides whether the weighting is usable at all.

**Architecture:** free functions over `Tensor` in `nn::utils`, matching the shape of `utils/sinkhorn.h`, `utils/crf.h`, `utils/conformal.h` and `utils/gaussian_process.h` — no classes, no environment, no trajectory state machine. The caller supplies the logged samples (rewards, importance weights) and, for the model-based estimators, the reward-model outputs it already has. The module never touches a policy, a simulator or a `Layer`; it is arithmetic over logged data, which is exactly what makes it testable against a closed-form oracle.

**Tech Stack:** C++17, the repo's `Tensor` (row-major, `AlignedAllocator<double>`), the project's own `-O2 -march=native -Wall -Wextra` build, a float64 numpy oracle in `tests/`, and a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

```
grep -rinE '\bsnr\b|off.?policy|behavior.polic|importance.?sampl|self.normaliz' include/ tests/
-> 0 hits
```

The repo has ~250 layers and 60+ utility files and **no off-policy machinery of any kind**. The nearest shipped things are all different objects:

- `utils/mmd_loss.h` — a *kernel* two-sample statistic. No importance weights, no policy, no value.
- `utils/conformal.h` — calibrates a prediction that was already made. Not an estimator of a policy's value.
- `utils/cross_validation.h` — resamples the *same* data. Off-policy evaluation is the opposite: it combines data collected under one policy with a *different* policy's actions.
- `utils/sinkhorn.h` — an entropic transport plan. A plan, not a value estimate.

**There is no `reward` concept anywhere in `include/nn` either** (`grep -rinE '\breward' include/nn --include=*.h` → 0 hits), so this module must be *model-free over logged data*, not a layer and not an environment.

## Scope boundary (stated so it cannot creep in)

**v1 is the ONE-STEP (contextual bandit) case.** Thomas & Brunskill's framing (arXiv:1604.00923, verified) is exactly this: logged `(x, a, r)` triples, a behaviour policy `pi_0`, a target policy `pi_1`.

**OUT OF SCOPE FOR v1:**
- Sequential / multi-step estimators (Jiang & Li's per-decision DR with `gamma`-discounting, arXiv:1511.03722). They need trajectory storage, discounting and per-step cumulative weights — a different data layout, not a small extension of this one.
- Any environment, simulator, replay buffer or `Layer` integration.
- Behaviour-policy *estimation* (propensity-model fitting). The caller supplies the propensities.

## The mathematics, and the three places it is easy to get wrong

Logged data: `N` samples `(s_i, a_i, r_i)` with `s_i ~ d`, `a_i ~ pi_0(.|s_i)`, `r_i ~ P(r|s_i,a_i)`.
Target policy `pi_1`. Importance weight `rho_i = pi_1(a_i|s_i) / pi_0(a_i|s_i)`.

1. **IPS** (Horvitz–Thompson): `V_IPS = (1/N) sum_i rho_i r_i`. Unbiased when `rho` is exact. High variance.
2. **SNIS / WIS** (self-normalised): `V_SNIS = (sum_i rho_i r_i) / (sum_i rho_i)`. Biased, lower variance.
   **The denominator is `sum rho_i`, NOT `N`.** Using `N` silently returns IPS.
3. **Direct method:** `V_DM = (1/N) sum_i Vhat(s_i)` where `Vhat(s) = sum_a pi_1(a|s) qhat(s,a)`.
4. **Doubly robust:** `V_DR = (1/N) sum_i [ Vhat(s_i) + rho_i (r_i - qhat(s_i,a_i)) ]`.

**The two algebraic facts the suite pins:**

- **IPS normaliser:** `V_IPS = (1/N) sum rho_i r_i`. **`V_SNIS = V_IPS / mean(rho)`.** A fixture where `mean(rho) != 1` separates them.
- **Double robustness** (the defining property, and the thing that is easy to break):

  `E_a~pi_0 [ Vhat(s) + rho(a) (r(a) - qhat(a)) ] = Vhat(s) + sum_a pi_1(a) r(a) - sum_a pi_1(a) qhat(a) = E_a~pi_1 [r(a)]`

  using `Vhat(s) = sum_a pi_1(a|s) qhat(s,a)` and `sum_a pi_0(a) rho(a) X(a) = sum_a pi_1(a) X(a)`. The `Vhat` term and the `sum_a pi_1(a) qhat(a)` term **cancel exactly**, so:

  - with **exact propensities** and a **wrong** reward model the estimator is still unbiased, and
  - with an **exact reward model** and **wrong** propensities it is still unbiased.

  Both are checked by **exact enumeration over a finite action support**, not by Monte Carlo — so the test is deterministic and exact.

  **A DR implementation that gets the model term wrong breaks one of the two arms and the test catches it.** (Replacing `r - qhat(s,a)` with `r - Vhat(s)`, or dropping the `Vhat(s)` term, or pairing the wrong `qhat` with the wrong sample, all break it.)

- **Effective sample size** (Kish): `ESS = (sum_i rho_i)^2 / sum_i rho_i^2`. `ESS <= N`, with equality iff all `rho_i` are equal. This is the diagnostic that says whether the importance weights are usable; a single dominant weight drives it to 1.

## Citations (all scraped, none recalled)

Verified 2026-10-30 by arXiv **title search** and then by scraping each abs page's `citation_title` / `citation_author` meta tags:

- `1604.00923` — "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning" — Thomas, Philip S.; Brunskill, Emma. One-step OPE; the direct-method and doubly-robust baselines.
- `1511.03722` — "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning" — Jiang, Nan; Li, Lihong. **The bandit DR estimator was transcribed from this paper's own LaTeX source** (`eq:dr`): `DR_I = V(s_I) + rho_I (r_I - R(s_I,a_I))`, `rho = pi_1(a|s)/pi_0(a|s)`, `V(s) = sum_a pi_1(a|s) R(s,a)`.
- `2006.10460` — "Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting" — Kuzborskij, Ilja; Vernade, Claire; György, András; Szepesvári, Csaba. The self-normalised estimator.
- `1910.12809` — "Minimax Weight and Q-Function Learning for Off-Policy Evaluation" — Uehara, Masatoshi; Huang, Jiawei; Jiang, Nan. Why the effective sample size is the diagnostic that matters.
- `2104.12820` — "Universal Off-Policy Evaluation" — Chandak, Yash; Niekum, Scott; da Silva, Bruno Castro; Learned-Miller, Erik; Brunskill, Emma; Thomas, Philip S.

**NO arXiv ID IS ASSERTED FOR THE SNR METRIC.** The queue's grep included `\bsnr\b`, but an arXiv title search for "sample noise ratio" and a PMLR v162 title scan (1233 titles) found **no** such paper, so no identifier and no definition is claimed for it. The queue entry is corrected rather than papered over.

## Public API

```cpp
namespace nn::utils {

struct OPEReport {
    double ips = 0.0;
    double snis = 0.0;
    double dm = 0.0;
    double dr = 0.0;
    double ess = 0.0;
};

double ope_importance_weight(double target_prob, double behavior_prob);
Tensor ope_importance_weights(const Tensor& target_probs, const Tensor& behavior_probs);

double ope_ips_estimate(const Tensor& rewards, const Tensor& weights);
double ope_self_normalized_ips(const Tensor& rewards, const Tensor& weights);
double ope_direct_method_estimate(const Tensor& model_target_values);
double ope_doubly_robust_estimate(const Tensor& rewards, const Tensor& weights,
                                  const Tensor& model_target_values,
                                  const Tensor& logged_action_values);
double ope_effective_sample_size(const Tensor& weights);

OPEReport ope_evaluate(const Tensor& rewards, const Tensor& weights,
                        const Tensor& model_target_values,
                        const Tensor& logged_action_values);
}
```

**Shape convention:** every input is a flat sample vector; the *number of elements* is the sample count and the shape is irrelevant (a `(N,1)` and a `(1,N)` tensor with the same values are the same data). All vectors passed to one estimator must have the **same element count**.

**Guards, each with a DISTINCT message** (the discriminator is the message, not the throw — this repo's recorded trap):
empty input; mismatched element counts; non-positive behaviour probability; non-positive target probability; non-positive weight; zero total weight (SNIS's denominator).

## Task list

### Task 1 — the float64 oracle (`tests/off_policy_eval_oracle.py`)

Write the numpy specification FIRST and run it. It must contain the **exact enumeration** over a finite action support that the C++ test will reuse, plus the pinned values the C++ test will assert.

**Step 1:** write `tests/off_policy_eval_oracle.py`.
**Step 2:** run `~/.venv/bin/python tests/off_policy_eval_oracle.py` — must print `=== SUMMARY: N passed, 0 failed ===`.
**Step 3:** commit `test(utils): float64 oracle for the off-policy evaluation estimators`.

### Task 2 — the RED C++ test (`tests/test_off_policy_eval.cpp`)

**Step 1:** write the test, including an **independent reference implementation** of every estimator (arithmetic written out in the test file, not calling the library).
**Step 2:** add the Makefile rules; `make build/test_off_policy_eval` must FAIL (the header does not exist yet).
**Step 3:** commit `test(utils): RED test for the off-policy evaluation estimators`.

### Task 3 — the implementation (`include/nn/utils/off_policy_eval.{h,cpp}`)

**Step 1:** write the header with the mathematics, the scope boundary and the citations.
**Step 2:** write the implementation, minimal.
**Step 3:** `make build/test_off_policy_eval` then run it — must PASS.
**Step 4:** commit `feat(utils): the off-policy evaluation estimators and the effective sample size`.

### Task 4 — wire it into the public API and the suite

**Step 1:** add `#include "utils/off_policy_eval.h"` to `include/nn/nn.h`.
**Step 2:** verify the umbrella header compiles standalone: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`.
**Step 3:** add the Makefile build rule, the `tests:` prerequisite and the `run_tests` line.
**Step 4:** commit `feat(utils): expose the off-policy evaluation estimators in the public umbrella`.

### Task 5 — the mutation harness (`tests/mutate_off_policy_eval.sh`)

Copy the shape of `tests/mutate_sliced_wasserstein.sh` (its twelve rules are listed in that file's header) and mutate the decisions the suite claims to cover. Every survivor is a real coverage gap until MEASURED inert.

**Step 1:** write the harness; `bash tests/mutate_off_policy_eval.sh` must run and print a verdict table.
**Step 2:** commit `test(utils): mutation harness for the off-policy evaluation estimators`.

### Task 6 — the full-suite gate

**Step 1:** `make run_tests` — the only failure must be the `test_neural_ode` entry already deferred in `NOT_FIXED.md`.
**Step 2:** ASan+UBSan build of the new suite: `make BUILD_DIR=build_asan build_asan/test_off_policy_eval` and run it.
**Step 3:** commit `test(utils): the off-policy evaluation suite is ASan+UBSan clean`.

### Task 7 — the queue

Move the item to `## Done` in `EXPANSION_QUEUE.md` with a one-line summary and the test results, and refresh the still-unbuilt list with the absences **re-measured this session**.
