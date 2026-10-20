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

**S1. NUCLEUS `p` IS A CUMULATIVE PROBABILITY CUT, NOT A TOKEN COUNT.** Sort descending, accumulate, and keep the smallest prefix whose mass is `>= p`. Crucially the token that *crosses* the threshold is **kept**: with `p = [0.5, 0.3, 0.15, 0.05]` and `p_top = 0.8`, the prefix `0.5, 0.3` has mass `0.8` and `0.5, 0.3, 0.15` has `0.95 >= 0.8`, so **3** tokens survive, not 2. The HuggingFace reference has this exact off-by-one and it is the single most common implementation bug. Test 2 pins the exact survivor set.

**S2. `p >= 1.0` IS A NO-OP, NOT AN ERROR.** The accumulated mass reaches 1.0 only after all V tokens, so the loop must terminate without touching the last index out of range.

**S3. TOP-K IS `>= k` THEN `-inf`, AND `k >= V` IS A NO-OP.** Not `>`; `k == V` must not `-inf` the argmax.

**S4. MIN-P IS A RELATIVE FLOOR AGAINST THE *MAX* LOGIT, NOT THE MEDIAN.** `min_p` filtering keeps tokens with `p_i >= min_p * p_max`. Holtzman's formulation is relative to the most likely token; using the median or the mean is a different algorithm. Test 5 pins a distribution where the two disagree.

**S5. THE PENALTIES SIGN-FLIP ON NEGATIVE LOGITS — THIS IS THE BUG.** For `penalty > 1`:
```
logit_i < 0  ->  logit_i * penalty     (more negative => more suppressed)
logit_i >= 0  ->  logit_i / penalty
```
An implementation that always multiplies **suppresses positive tokens and promotes negative ones** — it inverts the penalty on roughly half the vocabulary. Both branches are pinned, on a fixture where the sign split is asymmetric, by Test 6. (HF applies this to `scores` and is careful about the sign; this is the reference behaviour.)

**S6. FREQUENCY PENALTY IS LINEAR IN THE COUNT AND IS **NOT** CLIPPED; PRESENCE IS A STEP.** `logit_i -= penalty * count_i` vs `logit_i -= penalty * [count_i > 0]`. The count comes from the *history multiset*, so a token repeated 3 times gets 3x the frequency penalty but only 1x the presence penalty.

**S7. TEMPERATURE IS APPLIED TO LOGITS, NOT TO PROBABILITIES.** `logits / T` — dividing the softmax output by `T` is a different function. Test 7 pins `T -> 0+` approaching greedy and `T = 1` being the identity, which `exp(p)/T` fails.

**S8. COMPOSITION ORDER IS PART OF THE CONTRACT.** Temperature first (it is a reparameterisation of the whole distribution), then the truncations (they are functions of the *reweighted* distribution), then the penalties. Applying top-k to un-temperature-scaled logits gives a different survivor set at `T != 1`. Test 8 pins the composed result against a hand-computed value, and Test 8b pins that `T=1` makes the chain order-independent (which is the only reason the order is a convention and not an identity).

**S9. `-inf` MUST SURVIVE THE PIPELINE.** A truncated token is `-inf`; temperature divides it to `-inf` (fine) but a *naive* `logits * 0` penalty or an `exp` without max-subtraction turns it into `NaN`. Test 9 asserts no NaN reaches `sample_from_logits`.

**S10. THE MAX-SUBTRACT IN `sample_from_logits` IS LOAD-BEARING.** `logits = [1000, 999]` raw: `exp(1000) = inf`, `inf / inf = NaN`, and the draw returns garbage. With max-subtraction the result is exact. Test 10.

**S11. `round_to_int` USES `std::round`, WHICH IS ROUND-HALF-AWAY-FROM-ZERO — so `-2.5 -> -3`.** Not banker's rounding, not `std::nearbyint`. Test 11 pins both halves, because the repo's own TTA/metrics tests have been bitten by a wrong-rounding-mode expectation before.

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