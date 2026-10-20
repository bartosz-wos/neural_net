# [SHIPPED 2026-10-20] — see "WHAT ACTUALLY SHIPPED" at the foot of this file.
#
# Of the plan's 8 tasks: 1-8 all completed, with four of them producing findings
# the plan did not predict. Those corrections are recorded IN PLACE above (S1,
# S5, S8, S11 each carry a `CORRECTION 2026-10-20` block) rather than being
# quietly patched, so the plan documents what it got wrong as well as what it
# got right. Two further corrections were found only when the C++ was run
# against the oracle's numbers, and are recorded in
# tests/logit_processing_oracle.py next to the fixtures they invalidate.
#
# What this plan got right: the two-phase oracle-first order caught six defects
# before a line of C++ existed, and the mutation harness caught six more after.
# What it got wrong: four of its own worked examples, every one of them an
# expectation asserted from memory rather than measured.
#
# Logit Processing & Decoding Strategies — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/logit_processing.{h,cpp}` — a model-free logit-processing library (temperature, top-k, top-p/nucleus, min-p, typical-p, repetition/frequency/presence penalties, epsilon & round-to-integer output projectors) plus a `BeamSearchDecoder`, wired to a tested autoreggressive PixelCNN sampling loop so the discrete-output models that already ship in this repo can actually be decoded.

**Architecture:** Two layers. (1) `logit_processing` is a set of free functions over `Tensor` (one row of logits, or a (B,V) batch), each returning a *modified copy* — no in-place mutation, no hidden state, no `Model` dependency. Each processor is composed into a `LogitProcessorChain`. (2) `BeamSearchDecoder` consumes a caller-supplied `score_fn(batch_ids) -> (B,V) logits` callback, so it works with ANY autoregressive model in the repo (PixelCNN, VQ-VAE codebook decoding, future transformers) without the decoder knowing about any of them. That callback boundary is what makes it testable with a hand-written oracle model.

**Tech Stack:** C++17, the repo's `Tensor` (`include/nn/core/tensor.h`), the existing `utils/` convention (header-only doc block citing papers, `.cpp` with the implementation), Google-style C++ naming matching the repo (`snake_case_` members, trailing underscore), plain `assert`-free test harness matching `tests/test_sinkhorn.cpp` (a `CHECK(cond, msg)` macro and a `=== Summary: N passed, M failed ===` epilogue).

---

## THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-20)

The `## Ideas` list was **fully drained** (every remaining bullet is inside an HTML comment), so this session ran the queue's own standing rule — the docs-gap grep first, then absence greps — rather than typing an arXiv ID from memory.

```
grep -rinE 'nucleus|top.?p\b|top_p_sampl|temperature_sampl|typical.?sampl|min.?p\b|repetition.?penal|beam.?search' include/
```
→ **0 hits** in the whole tree. And a wider sweep for the concept under other names:
```
grep -rinE 'temperature|top_k|greedy|beam|sampling' include/nn/layers/generative/*.h include/nn/core/*.h
```
→ 11 hits, **every one a false positive**: `up_sample_` / `down_sample_` conv member names in `ddpm.h`, `conv_transpose` in `consistency.h`, "inverse pass (for sampling)" in `coupling_layer.h`, "greedy OT assignment" in `flow_matching.h` (the min-cost-assignment coupling, nothing to do with decoding), and an `rng_` for the WGAN gradient penalty.

**It is not redundant either.** The queue's rule is that an absence proves a feature is missing, not that it belongs — so the redundancy question was measured:

- `layers/generative/pixelcnn.h:203-205` says, verbatim: *"Per-pixel softmax: applied across the n_vals dimension inside each (i,j) position. For training with cross-entropy we expose logits and let the caller apply softmax. **The output is the raw logits.**"*
  PixelCNN models a **categorical distribution over `n_vals` values** and the repository ships **no code that turns those logits into a token**. There is no `sample()`, no `generate()`, no autoregressive loop — `grep -n 'sample\|generate' pixelcnn.cpp` returns nothing.
- `layers/generative/vq_vae.h` has `usage_count()` and a `VectorQuantizer` over a codebook, i.e. a second discrete-output model with the same missing decoder.
- The repo ships ~10 inference-time utilities (`adversarial`, `randomized_smoothing`, `vat`, `tta`, `temperature_scaling`, `ptq`, ...) and **not one** operates on the *logits of a generative model*.

So the discrete half of `generative/` has no inference path at all. This is the load-bearing absence.

**Citations — verified BY TITLE on 2026-10-20, never typed from memory** (this queue has produced nine wrong arXiv IDs):

| # | Paper | arXiv | Verified |
|---|-------|-------|----------|
| C1 | Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration", ICLR 2020 | **1904.09751** | Title search; result set is titled `[1904.09751] The Curious Case of Neural Text Degeneration`, abstract snippet matches §3.1 Nucleus Sampling |
| C2 | Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation", ACL 2018 | **1805.04833** | Title search; result set is titled `[1805.04833] Hierarchical Neural Story Generation`; fairseq README confirms `--sampling-topk` is the top-k this paper introduced |
| C3 | Meister et al., "Typical Decoding for Natural Language Generation", ACL 2022 | *(no ID used)* | Recorded by title + ACL Anthology only. **No arXiv ID is asserted** — see "Known citation limits" below. |
| C4 | Keskar et al., CTRL, "repetition penalty" | *(no ID used)* | Recorded by mechanism name only; the penalty is folklore from CTRL / Keskar et al. 2019. No ID asserted. |

> The arXiv **API** (`export.arxiv.org/api/query`) is unreachable from this host — `curl` returned an empty body with exit 0 for a control query (`all:electron`, which must return results). Both IDs above therefore come from a **title search**, which is the fallback the queue file itself prescribes ("Search by TITLE, or scrape the abs page's `citation_title` meta tag"). Two independent papers, two independent title searches, both titles echoed exactly by the result set.

---

## DESIGN

### 1. Processor primitives (free functions over one row of logits)

Every processor has the signature family:

```cpp
// logits: (V,) or (1,V) — a single distribution over the vocabulary.
// Returns a NEW tensor; the input is never mutated.
Tensor apply_temperature(const Tensor& logits, double temperature);
Tensor apply_top_k(const Tensor& logits, size_t k);
Tensor apply_top_p(const Tensor& logits, double p);
Tensor apply_min_p(const Tensor& logits, double min_p, double temperature);
Tensor apply_typical(const Tensor& logits, double mass);
Tensor apply_repetition_penalty(const Tensor& logits, const std::vector<int>& history, double penalty);
Tensor apply_frequency_penalty(const Tensor& logits, const std::vector<int>& history, double penalty);
Tensor apply_presence_penalty(const Tensor& logits, const std::vector<int>& history, double penalty);
Tensor apply_epsilon(const Tensor& logits, double epsilon);
Tensor apply_round_to_int(const Tensor& logits);
```

Each sets disallowed positions to `-inf`; the sample step treats `-inf` as probability 0. **Why `-inf` and not 0:** HF's `LogitsProcessor` writes `-inf`, and it composes correctly with an *additive* temperature applied afterwards (a 0 times a negative logit is 0, but `-inf` stays out of the way). Test 9 pins this.

A batched convenience `apply_row(Tensor, fn, idx)` keeps each primitive a single-row function so no primitive needs a batch code path. `LogitProcessorChain` holds `std::vector<std::function<Tensor(const Tensor&)>>` and applies them left-to-right, which is exactly the documented composition order (temperature first, then the truncators — the order matters and is pinned by Test 8).

### 2. Sampling

```cpp
enum class SamplingStrategy { GREEDY, MULTINOMIAL };
Tensor sample_from_logits(const Tensor& logits, SamplingStrategy s, std::mt19937& rng);
```
`sample_from_logits` is the ONLY place a categorical draw happens. It max-subtracts before `exp` (a raw `exp()` of a 1000-logit overflows to `inf` and `inf/inf` is NaN — pinned by Test 10).

### 3. Beam search

```cpp
struct BeamHypothesis { std::vector<int> tokens; double logprob; bool finished; };
struct BeamSearchConfig {
    size_t beam_width = 3;
    double length_penalty = 0.0;
    bool early_stopping = true;
    size_t max_length = 64;
};
class BeamSearchDecoder {
public:
    // score_fn(batch_ids) -> (batch_ids.rows(), vocab) LOGITS.
    // This callback boundary is what keeps the decoder model-agnostic and lets
    // the tests drive it from a hand-written oracle model.
    using ScoreFn = std::function<Tensor(const Tensor& batch_ids)>;
    BeamSearchDecoder(size_t vocab, BeamSearchConfig cfg, ScoreFn fn);
    Tensor decode(const Tensor& start_ids, size_t num_return);  // (num_return, L) padded with eos
    std::vector<std::vector<int>> hypotheses() const;            // for tests
};
```

Length penalty: `score / (len^alpha)` (GNMT §6.2, `alpha` = `length_penalty`), applied **only when ranking finished hypotheses**, never during the live beam expansion — a live beam must not be penalised for being short. Pinned by Test 14.

`eos_id` is a constructor argument (not `-1` magic). `eod` (`-1`) handling: PyTorch's `EncoderDecoderModel` does NOT treat eod as eos; neither do we. Pinned by Test 15.

### 4. The autoreggressive loop

```cpp
struct AutoregressiveResult { Tensor tokens; bool stopped_early; };
// steps == seq_len for PixelCNN; the decoder does not know or care.
AutoregressiveResult autoregressive_sample(
    const ScoreFn& score_fn, const Tensor& start_ids, size_t num_steps,
    const LogitProcessorChain& chain, SamplingStrategy strategy,
    std::mt19937& rng, int eos_id = -1, bool stop_on_eos = false);
```

---

## SEMANTICS EACH FIXTURE PINS (the traps this repo has already been bitten by)

**S1. NUCLEUS `p` IS A CUMULATIVE PROBABILITY CUT, AND `>=` IS LOAD-BEARING.** Sort descending, accumulate, keep the smallest prefix whose mass is `>= p`. The token that *reaches* the threshold is kept and the loop stops.

> **CORRECTION 2026-10-20, from the oracle (measured, not reasoned).** This
> section originally read: *"with `p = [0.5, 0.3, 0.15, 0.05]` and `p_top = 0.8`,
> the prefix `0.5, 0.3` has mass `0.8` and `0.5, 0.3, 0.15` has `0.95 >= 0.8`, so
> **3** tokens survive, not 2."* **That is backwards.** `0.5 + 0.3` is exactly
> `0.8`, which **satisfies** `cum >= 0.8`, so the loop stops with **2** tokens.
> The `>=` vs `>` distinction is real — `p = 0.8` is precisely the value that
> separates `>=:{0,1}` from `>:{0,1,2}` on this fixture (the oracle prints the
> full sweep over `p ∈ {0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}`, and `0.8` is
> the *only* separating value) — but the direction of the error was stated
> backwards. Test 2 pins `{0,1}` at `p=0.8`.

**S2. `p >= 1.0` IS A NO-OP, NOT AN ERROR.** The accumulated mass reaches 1.0 only after all V tokens, so the loop must terminate without touching the last index out of range.

**S3. TOP-K IS `>= k` THEN `-inf`, AND `k >= V` IS A NO-OP.** Not `>`; `k == V` must not `-inf` the argmax.

**S4. MIN-P IS A RELATIVE FLOOR AGAINST THE *MAX* LOGIT, NOT THE MEDIAN.** `min_p` filtering keeps tokens with `p_i >= min_p * p_max`. Holtzman's formulation is relative to the most likely token; using the median or the mean is a different algorithm. Test 5 pins a distribution where the two disagree.

**S5. THE PENALTIES SIGN-FLIP ON NEGATIVE LOGITS — THIS IS THE BUG.** For `penalty > 1`:
```
logit_i < 0   ->  logit_i * penalty     (more negative => more suppressed)
logit_i >= 0  ->  logit_i / penalty
```
An implementation that always multiplies **suppresses positive tokens and promotes negative ones** — it inverts the penalty on roughly half the vocabulary. Measured on `[-2.0, 4.0]` with history `[0,1]` and `penalty = 2.0`: the correct answer is `[-4.0, 2.0]`; the always-multiply bug gives `[-4.0, 8.0]` — the positive token is **promoted**.

> **ORACLE CORRECTION 2026-10-20.** The first draft of this fixture was
> `[-2.0, 4.0]` with history `[1]`, which penalises only index 1 — so the
> "negative branch multiplies" assertion was asserting on a token that was
> **never in the history**. `history` must name both indices for the sign split
> to be visible at all. This is the "the discriminator was absent from the
> fixture" trap in the mutation-testing table, and it bit a plain unit test.

> **ORACLE CORRECTION 2026-10-20 (a different bug, same function).** The
> oracle's own `repetition_penalty` first wrote `i = list(history).index(t)` —
> which is the token's **position in the history list**, not the token id — so
> with history `[0, 0, 1]` it wrote to indices 0 and 1 and silently modified
> index 2, the one token *not* in the history. This is the `pair<int,int>` trap
> from the debugging skill §4c.4, and it is worth writing down that the trap
> reaches ORACLES too: a wrong-side-of-the-differential bug is invisible until
> a check pins the entries that should be untouched.

**S6. FREQUENCY PENALTY IS LINEAR IN THE COUNT AND IS **NOT** CLIPPED; PRESENCE IS A STEP.** `logit_i -= penalty * count_i` vs `logit_i -= penalty * [count_i > 0]`. The count comes from the *history multiset*, so a token repeated 3 times gets 3x the frequency penalty but only 1x the presence penalty.

**S7. TEMPERATURE IS APPLIED TO LOGITS, NOT TO PROBABILITIES.** `logits / T` — dividing the softmax output by `T` is a different function. On `logits = [2, 1, 0]` at `T = 0.5`, `softmax(logits/T)[0] = 0.8668133321973347` while `softmax(probs/T)[0] = 1.3304819115496436` — the wrong version does not even sum to 1. Test 7 pins `T -> 0+` approaching greedy and `T = 1` being the identity.

**S8. COMPOSITION ORDER IS PART OF THE CONTRACT, AND ONLY *TRUNCATION-ON-MASS* PROCESSORS CAN PIN IT.** Temperature first (it reparameterises the whole distribution), then the truncations (they are functions of the *reweighted* distribution), then the penalties.

> **ORACLE CORRECTION 2026-10-20.** This section originally used **top-k** as the order discriminator, reasoning that applying top-k to un-temperature-scaled logits gives a different survivor set. **It does not, and cannot.** Dividing every logit by `T > 0` is strictly monotone, so the *ranking* is identical either way and top-k's survivor set is order-independent by construction. That is precisely why top-k is the wrong probe.
>
> **Top-p is the discriminator**, because its cut depends on cumulative *mass*, and temperature changes the mass. Measured on `logits = [3, 2, 1, 0]`: at `p = 0.9`, temperature-then-p keeps `{0,1}` while p-then-temperature keeps `{0,1,2}` — the wrong order admits a token carrying `0.0487` of the probability mass that the intended order admits. The composed softmax values are `0.8807970779778823` (correct order) vs `0.8668133321973347` (wrong order), so the test discriminates on a value, not only on a survivor count.

**S9. `-inf` MUST SURVIVE THE PIPELINE.** A truncated token is `-inf`; temperature divides it to `-inf` (fine) but a *naive* `logits * 0` penalty or an `exp` without max-subtraction turns it into `NaN`. Measured: `masked * 0.0` yields `[0.0, nan, nan, nan]` on this fixture. Test 9 asserts no NaN reaches `sample_from_logits`.

**S10. THE MAX-SUBTRACT IN `sample_from_logits` IS LOAD-BEARING.** `logits = [1000, 999]` raw: `exp(1000) = inf`, `inf / inf = NaN`, and the draw returns garbage. With max-subtraction the result is exact. Test 10.

**S11. `round_to_int` USES `std::round`, WHICH IS ROUND-HALF-AWAY-FROM-ZERO — so `-2.5 -> -3`.** Not banker's rounding, not `std::nearbyint`.

> **ORACLE CORRECTION 2026-10-20 — THE WRONG-LANGUAGE ROUNDING MODE.** The oracle
> first used `np.round`, which is **ties-to-even**, and asserted half-away-from-zero.
> They disagree: `np.round(0.5) = 0.0` but `std::round(0.5) = 1.0`;
> `np.round(2.5) = 2.0` but `std::round(2.5) = 3.0`; `np.round(-0.5) = -0.0` but
> `std::round(-0.5) = -1.0`. An oracle written in the wrong language with the
> wrong rounding mode ships a C++ test whose **every** value is wrong, and the
> resulting failure looks exactly like an implementation bug. The oracle now
> emulates the C++ semantics with `copysign(floor(|x| + 0.5), x)` and prints the
> numpy value as an explicit counterfactual. Test 11 pins both signs.

**S12. TIE-BREAK IN TOP-K / TOP-P IS FIRST-WINS ON THE ORIGINAL ORDER.** Two equal logits must resolve to the lower original index, so the result is deterministic and order-independent of the sort algorithm. Test 3 uses a fixture with exact ties.

**S13. A VOCAB-SIZE-1 DISTRIBUTION IS LEGAL.** `top_k(k=5, V=1)` must not crash or empty the support.

---

## TASKS

### Task 1: Oracle first (`tests/logit_processing_oracle.py`)

**Objective:** Pin every semantic S1–S13 numerically in numpy float64 BEFORE any C++ exists. This repo has twice caught a real bug in the oracle itself before a line of C++ was written (`test_vat` D9; `test_sinkhorn` "7 oracle findings").

**Files:**
- Create: `tests/logit_processing_oracle.py`

**Step 1: Write the oracle**

Self-contained, numpy float64, prints a C++-pasteable table of (fixture, expected output) for:
- softmax+temperature for `T ∈ {0.5, 1.0, 2.0}`
- top-k survivor sets for `k ∈ {1, 2, 3, V}` on a 5-token distribution
- **nucleus** survivor sets for `p ∈ {0.5, 0.8, 0.9, 1.0}` on `[0.5, 0.3, 0.15, 0.05]`
- min-p survivor set on a distribution where median-relative and max-relative disagree
- repetition-penalty output on `[-2.0, 0.0, 3.0]` with history `[0, 0, 1]` (S5/S6 — asymmetric sign split)
- frequency vs presence penalty on the same fixture
- `round_to_int` on `[-2.5, -1.5, 0.5, 1.5, 2.5]`
- a `softmax([1000, 999])` fixture with the max-subtracted reference (S10)

Self-check: assert `softmax` sums to 1, assert every nucleus survivor set is non-empty and its discarded mass is `< p`.

**Step 2: Run it, record the expected values**

Run: `python3 tests/logit_processing_oracle.py`
Expected: prints the table; self-checks pass; exit 0.

**Step 3: Commit**

```bash
git add tests/logit_processing_oracle.py
git commit -m "test(utils): numpy oracle pinning logit-processing semantics S1-S13"
```

---

### Task 2: RED — the test file, all 15 checks, no implementation

**Objective:** Write `tests/test_logit_processing.cpp` in full, run it, and WATCH IT FAIL. Every expected value comes from Task 1's oracle output.

**Files:**
- Create: `tests/test_logit_processing.cpp`

**The 15 checks** (names from the header, each ONE behaviour):

| # | Test name | Pins |
|---|-----------|------|
| 1 | `softmax_sums_to_one` | baseline helper |
| 2 | `nucleus_keeps_the_crossing_token` | **S1** — `p=0.8` on `[.5,.3,.15,.05]` keeps 3 |
| 3 | `top_k_ties_break_first_wins` | **S12** |
| 4 | `top_k_k_ge_v_is_noop` | **S3**, **S13** |
| 5 | `min_p_is_relative_to_max_not_median` | **S4** |
| 6 | `repetition_penalty_sign_split` | **S5** — negative multiplied, positive divided |
| 7 | `temperature_is_on_logits` | **S7** — `T=1` identity, `T->0` greedy |
| 8 | `chain_order_temperature_then_truncation` | **S8** |
| 8b| `chain_order_independent_at_T1` | **S8** |
| 9 | `neg_inf_survives_the_pipeline` | **S9** — no NaN |
| 10 | `sample_handles_huge_logits` | **S10** |
| 11 | `round_to_int_is_half_away_from_zero` | **S11** |
| 12 | `greedy_picks_argmax_deterministic` | sampler |
| 13 | `multinomial_is_reproducible_under_a_fixed_seed` | sampler + rng |
| 14 | `length_penalty_only_on_finished` | beam ranking |
| 15 | `eod_is_not_eos` | beam termination |

Plus a `sampling_from_logits_matches_the_categorical_pmf` check (empirical frequency of 10k draws from a known 3-token distribution within 0.02) — this is the one test that proves `sample_from_logits` is a *correct categorical* and not merely "some draw".

**Step 1: Write the harness + all 15 checks**

Match `tests/test_sinkhorn.cpp`'s structure exactly:
```cpp
#include "../include/nn/utils/logit_processing.h"
#include <cstdio>
static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { if (cond) { passed++; printf("  [PASS] %s\n", msg); } \
    else { failed++; printf("  [FAIL] %s\n", msg); } } while (0)
```
…and a `=== Summary: N passed, M failed ===` epilogue with `return failed ? 1 : 0;`.

Every numeric literal is copied from the oracle output. **Do not re-derive them here** — the TDD skill's "hand-derived expected value in a test" trap.

**Step 2: Run it to verify RED**

Run: `make build/test_logit_processing 2>&1 | tail -20`
Expected: **compile failure** — `logit_processing.h: No such file or directory`. That IS the RED state for a header-first library.

**Step 3: Commit the failing test**

```bash
git add tests/test_logit_processing.cpp
git commit -m "test(utils): 16 checks for logit processing + beam search (RED)"
```

---

### Task 3: GREEN — the primitives

**Objective:** Header + primitives only, enough for checks 1–11 to pass. Beam search is Task 4.

**Files:**
- Create: `include/nn/utils/logit_processing.h`
- Create: `include/nn/utils/logit_processing.cpp`

**Step 1: Write the header**

Follow `include/nn/utils/sinkhorn.h` verbatim in shape: `#ifndef NN_UTILS_LOGIT_PROCESSING_H`, a long comment block citing C1/C2 and recording the measured absence + the S1–S13 semantics (so a future reader cannot reintroduce the off-by-one), then declarations, then `}`. Include `"../core/tensor.h"`, `<cmath>`, `<functional>`, `<random>`, `<vector>`, `<algorithm>`, `<limits>`.

**Step 2: Write the implementation**

- `softmax(logits)` — max-subtract, `exp`, normalise; guard `V == 0` by throwing `std::invalid_argument`.
- Each primitive: flatten to `(1, V)`, compute a `-inf` mask, write to a fresh `Tensor`. **Never mutate the input.**
- `apply_repetition_penalty`: the S5 sign split.
- `apply_frequency_penalty` / `apply_presence_penalty`: count via a `std::vector<double> count(V)` built from the history.
- `apply_round_to_int`: `std::round`.
- `LogitProcessorChain::apply`: left-to-right fold.
- `sample_from_logits`: max-subtract, `exp`, normalise; GREEDY returns `argmax` (first-wins, S12); MULTINOMIAL draws `u ∈ [0,1)` and walks the CDF; clamp `u` so a sum that is `0.9999999999` cannot fall off the end.

**Step 3: Run to verify GREEN**

Run: `make build/test_logit_processing && ./build/test_logit_processing`
Expected: checks 1–11 pass; 12–15 fail to compile-link (declared but not yet defined) — so **temporarily** guard 12–15 behind `#ifdef BEAM_PENDING` for this task and remove the guard in Task 4. (A cleaner alternative: define the sampler in Task 3 and leave only 14/15 pending; prefer this one — it avoids an edit that changes what the suite tests.)

**Step 4: Mutation-check before moving on** (TDD skill: a test that passes on first run proves nothing)

Mutate `apply_top_p`'s crossing condition `if (cum >= p) break;` → `if (cum > p) break;`. Re-run. Expected: Test 2 FAILS (it keeps 2 tokens instead of 3). Restore. Then mutate `apply_repetition_penalty`'s `logit >= 0` branch → `< 0` and confirm Test 6 fails.

**Step 5: Commit**

```bash
git add include/nn/utils/logit_processing.h include/nn/utils/logit_processing.cpp
git commit -m "feat(utils): logit processors (temperature, top-k, top-p, min-p, penalties) + categorical sampler"
```

---

### Task 4: GREEN — beam search

**Objective:** Implement `BeamSearchDecoder` + `autoregressive_sample`; checks 12–15 go live.

**Files:**
- Modify: `include/nn/utils/logit_processing.h` (remove the `#ifdef BEAM_PENDING`, add the decoder)
- Modify: `include/nn/utils/logit_processing.cpp`
- Modify: `tests/test_logit_processing.cpp` (remove the guard)

**Step 1: Write the failing-side test first** — already written in Task 2, just un-guarded. Re-run and watch 12–15 fail to link.

**Step 2: Implement `BeamSearchDecoder`**

Per step, for each live beam hypothesis: score every vocab token, add `log(softmax)` to the beam's `logprob`, sort descending, keep the top `beam_width`; move any that hit `eos_id` into `finished_`. Rank finished hypotheses by `logprob / pow(len, length_penalty)` descending. `early_stopping` returns as soon as `beam_width` hypotheses are finished. Cap at `max_length`. Pad the returned `(num_return, L)` tensor with `eos_id`.

Live-beam expansion must NOT apply the length penalty (S14: penalising a live beam for being short is the classic implementation bug, and it silently biases toward short outputs).

**Step 3: Implement `autoregressive_sample`**

Loop `num_steps`, call `score_fn(current_ids)`, take `chain.apply`, `sample_from_logits`, append. `stop_on_eos` breaks when eos is drawn. Returns `{tokens, stopped_early}`.

**Step 4: Run to verify GREEN**

Run: `make build/test_logit_processing && ./build/test_logit_processing`
Expected: `=== Summary: 16 passed, 0 failed ===`

**Step 5: Commit**

```bash
git add include/nn/utils/logit_processing.{h,cpp} tests/test_logit_processing.cpp
git commit -m "feat(utils): model-agnostic beam search + autoregressive sampling loop"
```

---

### Task 5: The PixelCNN wiring — the absence this whole plan was measured against

**Objective:** Prove the decoder composes with a model that already ships, rather than only with a test oracle. PixelCNN is the repo's categorical-output model and its header says it returns raw logits.

**Files:**
- Create: `tests/test_pixelcnn_decode.cpp`

**Step 1: Write the failing test**

A 4-check suite:
1. `pixelcnn_logits_feed_the_decoder` — build a tiny PixelCNN (`in_ch=1, hidden=4, 3x3, 1 block, n_vals=8`), run `forward_with_cond` on a (1, 9) image, reshape the (1, 1*8*3*3) logits into a per-pixel (V) row, `apply_top_k(row, 3)`, assert exactly 3 finite entries and the rest `-inf`.
2. `pixelcnn_autoregressive_loop_terminates` — drive `autoregressive_sample` for 3 steps with a `ScoreFn` wrapping the real PixelCNN; assert the token count equals the requested steps and every id is in `[0, 8)`.
3. `greedy_decode_is_deterministic` — two `GREEDY` runs with the same weights give bit-identical tokens.
4. `temperature_lowers_entropy` — mean entropy of the sampled row at `T=0.1` is strictly below that at `T=1.0`.

**Step 2: Run to verify RED**

Run: `make build/test_pixelcnn_decode 2>&1 | tail -20`
Expected: compile failure (`test_pixelcnn_decode.o` not in the build, or an undefined `ScoreFn` adapter) — fix the build rule first if that is what fails, then confirm the *assertions* fail before the loop exists.

**Step 3: Implement the adapter**

A `ScoreFn` that runs PixelCNN over a canvas where already-generated pixels are written in and the rest are zero, then slices the logits at the current position. PixelCNN is masked (type A/B), so this is causal by construction — assert that in check 2 by showing that changing a LATER pixel does not change the current logits.

**Step 4: Run to verify GREEN**

Run: `make build/test_pixelcnn_decode && ./build/test_pixelcnn_decode`
Expected: `=== Summary: 4 passed, 0 failed ===`

**Step 5: Commit**

```bash
git add tests/test_pixelcnn_decode.cpp
git commit -m "test(generative): PixelCNN logits through the decoder + a masked-canvas autoregressive loop"
```

---

### Task 6: Register in the umbrella + Makefile

**Objective:** The header must be reachable from `nn.h`, and the suite from `make run_tests`. **A focused test that includes only the new header cannot catch a cross-header collision** (writing-plans skill, Step 3b).

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`

**Step 1: Add the include**

Insert `#include "utils/logit_processing.h"` in the utils section, alphabetically between `utils/lr_multiplier.h` and `utils/mmd_loss.h` (or wherever the section's ordering puts it — match the neighbours).

**Step 2: Verify the UMBRELLA compiles alone** (the plan-skill gate)

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: **silent, exit 0**. This is the gate that catches a redefinition against any other header.

**Step 3: Add the Makefile rules**

Mirror `test_sinkhorn` exactly:
```make
$(BUILD_DIR)/test_logit_processing: $(LIB_OBJS) $(BUILD_DIR)/test_logit_processing.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_pixelcnn_decode: $(LIB_OBJS) $(BUILD_DIR)/test_pixelcnn_decode.o
	$(CXX) $^ -o $@
```
Then add both binaries to the `tests:` prerequisite list (a suite missing from that list is silently never built — that bug cost this repo a phantom "No such file or directory" once) and add `run_tests` lines after the `test_sinkhorn` line:
```make
	@echo "=== Running test_logit_processing ==="; if ./$(BUILD_DIR)/test_logit_processing; then :; else echo "test_logit_processing" >> .run_tests_failed; fi
	@echo "=== Running test_pixelcnn_decode ==="; if ./$(BUILD_DIR)/test_pixelcnn_decode; then :; else echo "test_pixelcnn_decode" >> .run_tests_failed; fi
```

**Step 4: Verify the full suite has no regression**

Run: `make run_tests 2>&1 | tail -30`
Expected: the known pre-existing failures only (`test_neural_ode`, `test_coord_network`, `test_wgan_gp`, `test_vit` — all documented in `NOT_FIXED.md`), plus the two new suites green. **Nothing else changed.**

**Step 5: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register logit_processing in nn.h and the two new suites in the Makefile"
```

---

### Task 7: Update `EXPANSION_QUEUE.md`

**Objective:** Record the pick, its measured absence, its verified citations, and the semantic traps — so the next session does not re-pop it or re-introduce the off-by-one.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Move the entry to `## Done`** with a one-line summary: what shipped, the check count, the mutation results, and the grep that proved the absence.

**Step 2: Add the closing comment block at the bottom of `## Ideas`**, recording:
- The list was drained a TENTH time; the docs-gap grep's yield is now 2 in a row and the *absence* greps 3 in a row.
- **Do not re-pop** decoding/logit-processing — it shipped.
- The measured falses to warn the next session: `up_sample_`/`down_sample_`/`conv_transpose`/"for sampling"/"greedy OT assignment" all match `sampling`; and `min_p` matching a `min()` in `randomized_smoothing.h` is not min-p filtering.
- Candidates for next time, each to be re-measured: weight standardisation / Lipschitz normalisation (`grep -rinE 'weight.?standard|power.?norm|lipschitz|differentiable.?sort'` → **0 hits** today, re-verify).

**Step 3: Commit**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark logit processing + decoding Done; record the sampling-grep falses"
```

---

### Task 8: Mutation harness

**Objective:** Copy the harness that `tests/mutate_sinkhorn.sh` already is, and run it. Copying a *harness* is not copying production code — the harness encodes this repo's eight known false-verdict defects and every one of them was a real bug in an earlier session.

**Files:**
- Create: `tests/mutate_logit_processing.sh`

**Step 1: Adapt** `tests/mutate_sinkhorn.sh` verbatim: same `restore()`/`cleanup()`/atomic-`mv` structure, same `print-lib-objs` link set with the `>= 10` object-count assert, same checksum verification of every edit, same mutated-object exclusion, same `rm -f` of intermediates before each build, same post-harness rebuild-and-verify.

**Mutations to run** (each must be CAUGHT, or proven INERT by a probe):
1. nucleus `cum >= p` → `cum > p` (S1 off-by-one) → must FAIL check 2
2. repetition penalty `logit >= 0` → `logit < 0` (S5 sign flip) → must FAIL check 6
3. repetition penalty drop the `< 0` branch entirely → must FAIL check 6
4. temperature applied to probabilities instead of logits → must FAIL check 7
5. `apply_min_p` relative to the median instead of the max → must FAIL check 5
6. `std::round` → `std::nearbyint` (S11) → must FAIL check 11
7. drop the max-subtract in `sample_from_logits` → must FAIL check 10
8. length penalty applied to the LIVE beam → must FAIL check 14
9. `top_k` `>= k` → `> k` → must FAIL check 3
10. chain applied right-to-left → must FAIL check 8
11. `stop_on_eos` treats `eod` (-1) as eos → must FAIL check 15

**Step 2: Run it**

Run: `bash tests/mutate_logit_processing.sh`
Expected: a verdict row per mutation, and `POST-HARNESS: tree restored, rebuilt, and GREEN.` as the last line. **If the last line is not GREEN, stop — the tree is holding a mutant.**

**Step 3: Rebuild and re-verify after the harness** (never report a verdict without it)

Run: `make build/test_logit_processing && ./build/test_logit_processing`
Expected: `16 passed, 0 failed`.

**Step 4: Commit**

```bash
git add tests/mutate_logit_processing.sh
git commit -m "test(utils): mutation harness for logit_processing (11 mutations)"
```

---

## VERIFICATION (all must hold before the plan is called done)

- [ ] `python3 tests/logit_processing_oracle.py` exits 0 and its self-checks pass
- [ ] `./build/test_logit_processing` → `16 passed, 0 failed`, **deterministic across 3 runs**
- [ ] `./build/test_pixelcnn_decode` → `4 passed, 0 failed`, 3 runs
- [ ] `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` → silent, exit 0
- [ ] `bash tests/mutate_logit_processing.sh` → last line `POST-HARNESS: ... GREEN.`
- [ ] `make run_tests` → no NEW failure beyond the four documented in `NOT_FIXED.md`
- [ ] Clean build under `-Wall -Wextra` with no new warnings

## KNOWN CITATION LIMITS (recorded so they are not mistaken for verified)

- **C3 (typical decoding, Meister et al. 2022)** is recorded by title and ACL Anthology. **No arXiv ID is asserted**, because the arXiv API is unreachable from this host and a title search was not run for it. If a future session can reach arXiv, verify by title and add the ID then — do not add one from memory.
- **C4 (repetition penalty)** is folklore attributed to CTRL (Keskar et al. 2019). No ID asserted.
- The two IDs that ARE asserted (1904.09751, 1805.04833) were both confirmed by a title search whose result set echoed the title verbatim. That is weaker than an abs-page scrape and stronger than memory. The queue file's rule is "search by TITLE, or scrape the abs page's `citation_title` meta tag" — this used the first of the two acceptable forms.

## OUT OF SCOPE FOR v1 (stated so it cannot creep in)

- Greedy/beam/nucleus over any *specific* transformer in the repo — the `ScoreFn` boundary keeps this model-agnostic on purpose; wiring it into a concrete LM is a later task.
- Speculative / contrastive / `typical-p` (S4 lists `apply_typical` as an optional extra — if Task 3 does not reach it, drop it and say so rather than shipping it untested).
- `min-p`'s interaction with beam search (min-p is a truncation; applying it to the live beam is a research variant, not the paper's algorithm).
- Any change to `pixelcnn.{h,cpp}` — the adapter lives in the test, not in the library.
---

# WHAT ACTUALLY SHIPPED

**Files created**
- `include/nn/utils/logit_processing.h` — 13 semantics documented (S1–S13), both
  citations verified by title, and a stated out-of-scope list
- `include/nn/utils/logit_processing.cpp` — every processor as a free function
  over `Tensor`; `LogitProcessorChain`; `sample_from_logits`;
  `BeamSearchDecoder` over a `ScoreFn`; `autoregressive_sample`
- `tests/logit_processing_oracle.py` — numpy oracle, **59/59** self-checks
- `tests/test_logit_processing.cpp` — 16 test groups, **243/243**
- `tests/mutate_logit_processing.sh` — 21 mutations + 2 inverted-verdict inert
  probes, **18 caught / 0 survived / 2 inert-confirmed**, ASan+UBSan CLEAN
- `tests/test_pixelcnn_decode.cpp` — the integration, **27/27**

**Files modified**
- `include/nn/nn.h` — umbrella include (verified by a standalone
  `-fsyntax-only` compile of `#include "nn/nn.h"`, exit 0)
- `Makefile` — link rule, `tests:` prerequisite, `run_tests` line, for BOTH suites
- `EXPANSION_QUEUE.md` — moved to `## Done`, plus the next session's notes
- `docs/plans/2026-10-20_logit-processing-decoding.md` — this file

**Verification, all actually run**

| Gate | Result |
|---|---|
| `python3 tests/logit_processing_oracle.py` | exit 0, 59/59 |
| `./build/test_logit_processing` | 243/243, three consecutive runs |
| `./build/test_pixelcnn_decode` | 27/27, three consecutive runs |
| `g++ -std=c++17 -Iinclude -fsyntax-only` on `nn/nn.h` | silent, exit 0 |
| `bash tests/mutate_logit_processing.sh` | 18 caught / 0 survived / 2 inert; last line `POST-HARNESS: tree restored, rebuilt, and GREEN.` |
| ASan+UBSan on the pristine tree | CLEAN |
| `make run_tests` | 253 suites, **one** failure — `test_neural_ode`, verified pre-existing by stashing every local change and rebuilding on clean master (exit 1 there too) |

## THE PLAN WAS WRONG FOUR TIMES, AND EVERY TIME IT WAS AN EXPECTATION ASSERTED FROM MEMORY

1. **S1**: claimed `p=0.8` on `[0.5, 0.3, 0.15, 0.05]` keeps 3 tokens. It keeps
   2 — `0.5 + 0.3 == 0.8` satisfies `cum >= 0.8`.
2. **S5**: the fixture named only one history index, so the "negative branch
   multiplies" assertion was asserting on a token never penalised.
3. **S8**: claimed top-k pins the composition order. It cannot — temperature is
   monotone, so top-k's survivor set is order-independent by construction.
4. **S11**: used `np.round`, which is ties-to-even, to derive values for
   `std::round`, which is half-away-from-zero.

Plus two found only when the C++ ran against the oracle's numbers:

5. **The one-ulp trap.** `log([0.5, 0.3, 0.15, 0.05])` is not a valid nucleus
   boundary fixture: `softmax(log(0.3)) = 0.29999999999999993`, so the
   cumulative mass is `0.79999999999999993 < 0.8` and a **correct**
   implementation returns `{0,1,2}`. Replaced with dyadic probabilities
   `[0.5, 0.25, 0.125, 0.125]`, which are exactly representable.
6. **The S8 fixture needed a flattening temperature.** At `T = 0.5` both orders
   keep the same single token, so the discriminator was vacuous. `T = 2.0`
   separates them.

**The reusable rule, and it is the same shape as three earlier sessions in this
repo:** a fixture whose discriminating value is only correct to within a
rounding error does not discriminate anything, and an assertion that a correct
*and* an incorrect implementation both satisfy — an inequality, or a check on a
fixture that does not contain the boundary — is not a test of the thing under
test. Both traps produce a green suite and a wrong library.

## WHAT THE HARNESS FOUND IN THE HARNESS

Building the harness from `tests/mutate_sinkhorn.sh` re-surfaced four of this
repo's own harness bugs, which is the strongest argument yet for copying a
harness rather than re-deriving one:

1. `mutate_sinkhorn.sh` **never restores after a successful mutation** — it
   relies on the next mutation's `replace` overwriting the previous one, which
   holds only while every mutation touches the same file. Here that produced two
   bogus `MUT-NOOP` verdicts against anchors that exist exactly once.
2. A `*` inside a double-quoted `$(printf ...)` is **glob-expanded by the shell**
   into a file listing, so the anchor became `x < 0.0 = x = Makefile tests
   tools penalty : ...` and matched nothing.
3. `if (false)` as a guard **replacement** deletes the closing brace and reports
   `MUT-BROKEN` — a build verdict for a well-formed mutation.
4. A duplicated mutation id ran twice and polluted the verdict table.

Three of the first run's survivors were **real coverage gaps**, each now killed
by a test rather than explained away:

- the chain-order checks compared two **hand-composed calls**, which says nothing
  about what `LogitProcessorChain` does — a right-to-left chain passed all of them;
- dividing by `len` and by `len + 1` are **both monotone in length**, so the two
  normalisers agree on similar hypotheses; a brute-force sweep located the
  separating region (a short hypothesis with a competitive raw log-probability);
- `n_finite(T=2) >= n_finite(T=1)` is an **inequality**, which holds whether or
  not the temperature is honoured at all.

Two mutations are **genuinely inert**, measured rather than assumed, and are now
scored by a function that inverts the verdict so a future refactor that makes
them matter fails loudly:

- `-inf / T == -inf` in IEEE-754 for every finite `T > 0`, so the explicit `-inf`
  guard in `apply_temperature` is documentation, not computation;
- at `k == V` the `k > v` mutant skips the early return, but the write-back loop
  below it still restores every index.

## A REAL PROPERTY OF PIXELCNN, RECORDED SO NOBODY "FIXES" IT

With an empty canvas, **pixel 0's logits are exactly constant** — all eight are
`-0.003792`, so the entropy is `log(8) = 2.079442` at every temperature. The
type-A mask blocks the centre of the first conv, so every position sees the same
empty context. That is correct: with no context there is no distribution to
sharpen. An earlier draft of that test read pixel 0 and concluded temperature was
broken, which would have sent someone to remove the mask. Tests 4f/4g now pin the
property explicitly and test 4 uses a non-empty prefix at pixel 2, where the
distribution is real (entropy 1.755 / 2.057 / 2.078 at T = 0.2 / 1 / 5).
