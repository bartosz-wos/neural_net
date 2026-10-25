# Speculative Decoding Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add speculative decoding — the draft/target rejection-sampling scheme of Leviathan, Kalman & Matias — to `include/nn/utils/logit_processing.{h,cpp}`, closing the one out-of-scope item that the file's own header names.

**Architecture:** A new free function `speculative_decode_step` in `nn::utils`, driven by two **model callbacks** rather than by any `Layer`: a *draft* model `q` and a *target* model `p`, each of which maps a token-id prefix to a row of logits. The step draws `gamma` guesses from `q` autoregressively, scores `gamma + 1` positions with `p`, accepts a prefix of the guesses by the paper's acceptance rule `r_i <= p_i(x)/q_i(x)`, and — when a guess is rejected — emits one correction token drawn from the residual distribution `norm(max(0, p_{n+1} - q_{n+1}))`. The distribution of the emitted tokens is **provably identical** to sampling from `p` alone, which is the paper's entire claim and the property the test suite pins.

**Tech Stack:** C++17, the repo's `Tensor` class, the existing `logit_processing.cpp` helpers (`softmax`, `probabilities`), a numpy float64 oracle, and the repo's mutation harness.

> **PLAN REVISION (2026-10-25, after the oracle ran).** The oracle
> (`tests/speculative_decoding_oracle.py`, 3074 self-checks, green) corrected
> **four** things in this plan before a line of C++ was written. Two of them
> change the API and the test matrix, so they are recorded here rather than
> silently dropped:
>
> 1. **SP3's CLAMP IS MEASURED INERT.** The plan asserted that clamping the
>    acceptance ratio at 1 is "required by the maths" and built a foil to
>    prove it. The foil was **VACUOUS**: the clamped and unclamped rules
>    produced identical output on 3000/3000 trials. The reason is a fact about
>    the *rule's shape*: when the acceptance is written as a threshold
>    comparison `r <= p/q` and `p > q`, the ratio exceeds 1 and every
>    `r in [0,1)` satisfies it automatically — which is exactly the correct
>    behaviour. **The clamp is DEFENSIVE, not load-bearing**, and the C++
>    harness must score a drop-the-clamp mutation with an **inverted** verdict
>    (it fails if the mutation ever starts to matter), not as a coverage gap.
> 2. **THE `<=` vs `<` BOUNDARY IS ALSO UNOBSERVABLE BY A RANDOM FIXTURE.**
>    The two differ only when the uniform draw lands *exactly* on the
>    acceptance probability — a measure-zero event for a continuous draw. The
>    C++ test must therefore **supply** the draw at exactly the boundary, on a
>    **dyadic** acceptance probability of 0.5, or the convention is untestable.
>    (The repo's top-p fixture already learned the general lesson: a fixture
>    whose discriminating value is only correct to within a rounding error
>    does not discriminate anything. Here the value must be exactly
>    representable *and* supplied.)
> 3. **THE FIXTURE MUST VARY BY POSITION.** The plan's first fixture used the
>    same distribution at every position, which made the SP6 wrong-index foil
>    **VACUOUS** (`p_n == p_{n+1}` for every `n`, so the wrong index computes
>    the same distribution). The shipped fixture gives every position a
>    *different* pair of distributions.
> 4. **`max(0, ·)` IS MATHEMATICALLY LOAD-BEARING, NOT MERELY A DIFFERENT
>    NUMBER.** `p` and `q` are both distributions, so `sum(p - q) == 0`
>    exactly. Dropping the componentwise clamp leaves a zero-sum vector, and
>    the normalisation **divides by zero** — the residual is not merely
>    different, it is *undefined*.

---

## 0. The gap, measured (not assumed)

The queue's `## Ideas` list was drained a **fifteenth** time. Its standing **docs-gap grep** ran first:

```
grep -rinE 'not entitled|out of scope|deliberately not|is not exposed|for now we do not|this repo does not|no hook' include/nn --include=*.h
-> 19 hits, all already accounted for in shipped headers
```

**Zero new hits — the fifth consecutive run.** The queue itself predicts this decay: "each of the previous three was a file THIS QUEUE had just written. Its yield decays as the file it reads gets fixed."

The fallback was the last entry's own **VERIFIED-ABSENT** list, and the strongest item on it is named *inside a shipped header as unimplemented*:

```
include/nn/utils/logit_processing.h:167
// OUT OF SCOPE FOR v1, stated so it cannot creep in:
//   - Speculative / lookahead / contrastive decoding.
```

And by content, not by filename:

```
grep -rinE 'speculat|draft.model|draft_model|acceptance|rejection.sampl|target.model|lookahead.decod' include/ tests/
-> include/nn/utils/logit_processing.h:168   (the out-of-scope line itself)
   include/nn/interpretability/partial_dependence.cpp:36  ("via rejection sampling" — unrelated)
   include/nn/utils/... (nothing else)
```

**This is the strongest possible shape of gap**: the repo's own newest utility file states in its own words that this one algorithm is missing, and it is the *last* item in that file's out-of-scope list.

### Not redundant

The repo ships `autoregressive_sample` (one token per target-model call) and `BeamSearchDecoder` (a *search* over the target model). Neither is speculative decoding:

- `autoregressive_sample` calls the model **once per token**, serially. Speculative decoding exists to reduce that call count.
- `BeamSearchDecoder` explores multiple hypotheses and returns the **best** by score — it is not a sampler at all, and its output distribution is not the target model's.

Speculative decoding's claim is *distribution-preserving*: the emitted token sequence has **exactly** the target model's distribution. That is a different, stronger, and testable property than either shipped mechanism has.

### Citation — verified by TITLE, then by the abs page's meta tags

This queue has produced **twelve wrong arXiv IDs**. The rule is: search by title, then confirm the abs page's `citation_title` / `citation_author` meta tags.

- **Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding"**, ICML 2023, PMLR v202:19274-19286; **arXiv:2211.17192**.

Verified two ways on 2026-10-25:

1. **Title search** returned `[2211.17192] Fast Inference from Transformers via Speculative Decoding` and the PMLR proceedings page.
2. **The abs page's own meta tags**, scraped directly:
   ```
   <meta name="citation_title" content="Fast Inference from Transformers via Speculative Decoding">
   <meta name="citation_author" content="Leviathan, Yaniv">
   <meta name="citation_author" content="Kalman, Matan">
   <meta name="citation_author" content="Matias, Yossi">
   ```
   All three authors match in order.

**The algorithm was read out of the paper's own HTML** (`https://arxiv.org/html/2211.17192`, 312 KB, HTTP 200), not from memory — Algorithm 1 `SpeculativeDecodingStep` was transcribed verbatim into §1 below. This matters: the acceptance rule's index convention (`n <- min({i-1 | ...} ∪ {gamma})`) and the residual distribution's index (`p_{n+1}`, `q_{n+1}`) are exactly the places a memory-typed implementation goes wrong, and the paper's own text is right there.

---

## 1. The algorithm, as the paper's Algorithm 1 states it

Transcribed from `https://arxiv.org/html/2211.17192`:

```
Algorithm 1  SpeculativeDecodingStep
  Inputs: M_p, M_q, prefix
  ▷ Sample γ guesses x_1, ..., x_γ from M_q autoregressively.
    for i = 1 to γ do
        q_i(x)  <- M_q(prefix + [x_1, ..., x_{i-1}])
        x_i     ~ q_i(x)
    end for
  ▷ Run M_p in parallel.
    p_1(x), ..., p_{γ+1}(x)  <-  M_p(prefix), ..., M_p(prefix + [x_1, ..., x_γ])
  ▷ Determine the number of accepted guesses n.
    r_1 ~ U(0,1), ..., r_γ ~ U(0,1)
    n   <- min( { i-1 | 1 <= i <= γ, r_i > p_i(x)/q_i(x) } ∪ {γ} )
  ▷ Adjust the distribution from M_p if needed.
    p'(x) <- p_{n+1}(x)
    if n < γ then
        p'(x) <- norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
    end if
  ▷ Return one token from M_p, and n tokens from M_q.
    t ~ p'(x)
    return prefix + [x_1, ..., x_n, t]
```

### The semantics this plan pins

**SP1 — `q_i` IS THE DRAFT MODEL'S DISTRIBUTION AT THE PREFIX ENDING IN `x_{i-1}`.**
`q_1` is the draft's distribution after `prefix`; `q_i` is the draft's distribution after `prefix + [x_1..x_{i-1}]`. So `q_i` is exactly the distribution that **produced** `x_i`. `q_{γ+1}` is the draft's distribution after `prefix + [x_1..x_γ]` — the one needed by the residual distribution.

**SP2 — `p_i` IS THE TARGET MODEL'S DISTRIBUTION AT THE SAME PREFIX.**
`p_1` is the target's distribution after `prefix`; `p_i` is the target's distribution after `prefix + [x_1..x_{i-1}]`. `p_{γ+1}` is the target's distribution after the full guessed sequence. **The target's `p_i` and the draft's `q_i` are distributions over the SAME vocabulary at the SAME position.** This is what makes `p_i(x)/q_i(x)` a ratio of probabilities for the same token.

**SP3 — THE ACCEPTANCE RULE IS `r_i <= p_i(x_i) / q_i(x_i)`, WITH `r_i ~ U(0,1)`.**
A guess is accepted when the uniform draw is at most the ratio. **The ratio is CLAMPED AT 1** — this is not in the paper's Algorithm 1 but is required by the maths: when `q_i(x_i) > p_i(x_i)` the ratio exceeds 1 and every `r_i ∈ [0,1]` accepts, which is the correct behaviour (the ratio is a probability, and a probability above 1 means "always accept"). Without the clamp, `r_i > p_i/q_i` is *never* true for `p_i > q_i`, and the acceptance set is empty — a silent, catastrophic bug that still produces plausible output.

**SP4 — `n` IS THE NUMBER OF ACCEPTED GUESSES, AND THE FIRST REJECTION ENDS THE RUN.**
`n <- min({i-1 | 1 <= i <= γ, r_i > p_i(x_i)/q_i(x_i)} ∪ {γ})`. If guess `i` is the first rejection, then `n = i-1`: the guesses `x_1..x_{i-1}` are accepted and `x_i` is discarded. If **no** guess is rejected, `n = γ`.

**SP5 — THE FIRST REJECTED GUESS IS NOT EMITTED; A CORRECTION TOKEN IS.**
The output is `prefix + [x_1..x_n, t]`. When `n < γ` the guess `x_{n+1}` is **thrown away** and `t` replaces it. This is the step that makes the output distribution equal to `p`'s.

**SP6 — THE RESIDUAL DISTRIBUTION IS `norm(max(0, p_{n+1} - q_{n+1}))`.**
The `max(0, ·)` is componentwise. `norm` is normalisation to sum to 1. The residual is over the **FULL vocabulary**.

**SP7 — WHEN `n == γ` (ALL GUESSES ACCEPTED), THE EMITTED TOKEN IS DRAWN FROM `p_{γ+1}` — UNMODIFIED.**
This is the `if n < γ` guard in the paper. The residual `norm(max(0, p_{γ+1} - q_{γ+1}))` is **not** used in this case. A residual-only implementation drops the all-accepted bonus token, and then every speculative step emits exactly `n` tokens instead of `n+1` — a `γ`-fold loss of the entire speedup.

**SP8 — THE DRAFT MODEL IS CALLED `γ` TIMES, THE TARGET MODEL `γ+1` TIMES.**
Per step, the target model is evaluated at `γ+1` positions (all in parallel in the paper) and the draft at `γ`.

**SP9 — THE OUTPUT IS EXACTLY `n+1` TOKENS: THE `n` ACCEPTED GUESSES PLUS ONE.**
Never `n`, never `n+2`. The paper's final line is `return prefix + [x_1, ..., x_n, t]`.

**SP10 — WHEN THE DRAFT AND TARGET ARE IDENTICAL, EVERY GUESS IS ACCEPTED AND `γ+1` TOKENS COME OUT.**
`p_i == q_i` gives ratio 1, `r_i <= 1` always true, so `n = γ` and `γ+1` tokens are emitted. This is the strongest single property of the implementation and it is directly testable: with identical models the output must be *exactly* the draft's `γ+1` tokens.

**SP11 — THE RESIDUAL DISTRIBUTION IS NEVER ALL-ZERO WHEN A REJECTION HAPPENS.**
If `n < γ`, guess `x_{n+1}` was rejected, so `q_{n+1}(x_{n+1}) > p_{n+1}(x_{n+1})` — hence `p_{n+1} - q_{n+1}` is **strictly positive** on at least one coordinate and the residual has non-zero mass. A `norm` that divides by a zero sum is NaN, and a NaN token would silently corrupt the sequence.

**SP12 — THE ACCEPTANCE RATIO IS COMPUTED FROM PROBABILITIES, NOT LOGITS.**
`p_i(x_i)/q_i(x_i)` is a ratio of *probabilities*. An implementation that compares *logits* (or log-probabilities) computes a different quantity and is wrong.

---

## 2. What is model-free, and why that matters

Every existing function in `logit_processing.h` is a free function over `Tensor`. This one needs **two models**, and the file's existing answer to that problem is already there: `BeamSearchDecoder::ScoreFn`, a caller-supplied `std::function<Tensor(const Tensor&)>`. The new code reuses that exact boundary.

The draft and target models are supplied as:

```cpp
using DraftFn = std::function<Tensor(const Tensor&)>;  // prefix -> (1, V) LOGITS
```

where the argument is a `(1, L)` tensor of token ids and the return is a `(1, V)` row of logits. **The library never learns what generated them.** That is what lets the tests drive it from hand-written oracle models *and* from a real PixelCNN with no change to the decoder.

This is the same model-agnostic shape as `BeamSearchDecoder`, and it is the reason the whole algorithm is testable with no `Model`, no optimizer and no network.

---

## 3. Semantics, in one table (the test matrix)

| # | Property | Why it is pinned |
|---|----------|-----------------|
| SP1 | `q_i` is the draft's distribution at `prefix + [x_1..x_{i-1}]` | the ratio compares the same position's two models |
| SP2 | `p_i` is the target's distribution at the same prefix | ditto |
| SP3 | accept iff `r_i <= min(1, p_i(x_i)/q_i(x_i))` | the clamp is **not** in the paper's text but is required by the maths |
| SP4 | `n` = accepted count; the first rejection ends the run | `n = min({i-1 : rejected} ∪ {γ})` |
| SP5 | a rejected guess is DISCARDED and replaced by the correction token | the `t` in `prefix + [x_1..x_n, t]` |
| SP6 | residual = `norm(max(0, p_{n+1} - q_{n+1}))`, componentwise, full vocab | the paper's `if n < γ` branch |
| SP7 | when `n == γ`, the token is drawn from `p_{γ+1}` **unmodified** | dropping this loses the whole speedup |
| SP8 | draft called `γ` times, target called `γ+1` times per step | the speedup claim itself |
| SP9 | output is exactly `n+1` tokens | the paper's return line |
| SP10 | identical models ⇒ `γ+1` tokens, all guesses accepted | strongest single property |
| SP11 | the residual is never all-zero when `n < γ` | a zero-sum `norm` is NaN |
| SP12 | the ratio uses probabilities, not logits | a logit comparison is a different quantity |

---

## 4. Tasks

### Task 1: Write the numpy oracle (RED for the whole feature)

**Objective:** Pin every numeric expectation and every semantic above, in Python, before a line of C++ exists.

**Files:**
- Create: `tests/speculative_decoding_oracle.py`

**Step 1: Write the oracle.**

The oracle implements:
- `softmax(logits)`
- `speculative_step(p_dists, q_dists, rng)` — the paper's Algorithm 1, transcribed independently of the C++.
- A **second, independent formulation** of the acceptance rule (a direct simulation of "sample from q, keep with probability min(1, p/q)") and a cross-check that the two agree. **One formulation agreeing with itself proves nothing** — the repo's own queue records this lesson from the typical-p work.
- The **distributional** property: for many seeds, the empirical distribution of the emitted token must match the target's distribution to within a Monte-Carlo tolerance. This is the paper's actual claim, and it is the check a hand-written example cannot replace.

The oracle must also include the **foils**: the wrong variants (no clamp on the ratio; residual used even when `n == γ`; residual computed at index `n` instead of `n+1`; guesses emitted past the first rejection) each asserted to **differ** from the correct behaviour on the fixture. *A grad-check that passes three of four candidates is still a useful test, but only if you have measured that the fourth candidate fails.*

**Step 2: Run it.**

```bash
/home/stefan/.venv/bin/python tests/speculative_decoding_oracle.py
```
Expected: `ALL ORACLE SELF-CHECKS PASSED`, exit 0.

**Step 3: Commit.**

```bash
git add tests/speculative_decoding_oracle.py
git commit -m "test(utils): numpy oracle for speculative decoding"
```

---

### Task 2: Write the C++ test suite (RED)

**Objective:** A C++ suite that fails to compile/link against the not-yet-existing API.

**Files:**
- Create: `tests/test_speculative_decoding.cpp`

**Step 1: Write the test file.**

It must:
- Drive the algorithm with a **hand-written oracle draft and target** (a `std::function` returning a fixed row per prefix), so the expected tokens are computed by hand from the algorithm, not from the implementation.
- Pin **every** row of the §3 table, including the two that matter most: SP10 (identical models ⇒ `γ+1` tokens) and SP7 (`n == γ` ⇒ the bonus token comes from `p_{γ+1}`, unmodified).
- Assert the **call counts** (SP8) by counting invocations of the draft and target callbacks.
- Assert the **output length** is exactly `n+1` (SP9) on a fixture where a rejection happens at a known position.
- Assert the **residual** is used: on a fixture where the first guess is rejected, the emitted correction token must be **absent from the draft's argmax** in a way the unmodified `p` distribution could not produce. (The fixture must be built so the two differ — a fixture that cannot separate them is a fixture that cannot test them.)
- Include a **degenerate/distribution** check: the emitted token's empirical distribution matches the target's, over many seeds, within a Monte-Carlo tolerance. This is the paper's actual claim.

**Step 2: Run it and watch it fail.**

```bash
make build/test_speculative_decoding
```
Expected: **compile error** — `speculative_decode_step` is not declared. This is the RED state and it is mandatory to observe it.

**Step 3: Commit.**

```bash
git add tests/test_speculative_decoding.cpp
git commit -m "test(utils): speculative decoding tests (RED)"
```

---

### Task 3: Implement the header declarations

**Objective:** The public API surface, documented.

**Files:**
- Modify: `include/nn/utils/logit_processing.h`

**Step 1: Add the declarations.**

```cpp
// ---------------------------------------------------------------------------
// SPECULATIVE DECODING
//
// Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative
// Decoding", ICML 2023, arXiv:2211.17192. Read from the paper's own HTML.
//
// The DRAFT and TARGET models are supplied as callbacks, exactly like
// `BeamSearchDecoder::ScoreFn`, so this file stays model-agnostic: it never
// learns what generated the logits, which is what lets the tests drive it
// from a hand-written oracle AND from a real PixelCNN.
//
// One step emits BETWEEN 1 AND gamma+1 tokens. The output distribution is
// IDENTICAL to sampling from the target model alone -- that is the paper's
// entire claim, and it is what the suite pins.
// ---------------------------------------------------------------------------

// A draft or target model: token-id prefix (1, L) -> (1, V) LOGITS.
using DraftModel = std::function<Tensor(const Tensor&)>;

struct SpeculativeConfig {
    // Number of guesses the draft model proposes per step. gamma >= 1.
    size_t gamma = 4;
};

struct SpeculativeResult {
    // The tokens emitted by THIS step: the accepted guesses plus one
    // correction token. Length is between 1 and gamma+1.
    Tensor tokens;              // (1, k)
    size_t num_accepted = 0;   // n -- how many guesses were accepted
    bool rejected = false;      // true iff a guess was rejected (n < gamma)
};

// One speculative decoding step.
//
// `draft` proposes `gamma` tokens autoregressively; `target` scores
// `gamma+1` positions. The accepted prefix and the correction token follow
// Algorithm 1 exactly (see the header's SP1..SP12 comment block).
//
// Throws std::invalid_argument on gamma == 0, a null callback, or a callback
// that returns a row whose width is not the vocabulary, and
// std::runtime_error when the draft and target disagree about the vocabulary
// width.
SpeculativeResult speculative_decode_step(const DraftModel& draft,
                                           const DraftModel& target,
                                           const Tensor& prefix,
                                           size_t gamma,
                                           std::mt19937& rng);
```

Also remove the out-of-scope line for speculative decoding from the header's
`OUT OF SCOPE FOR v1` block, and record the new section in the header's
top-of-file history comment.

**Step 2: Compile the umbrella header alone.**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: no output, exit 0. (This is the cross-header redefinition check the
repo's planning skill requires; a focused test that includes only the new
header cannot catch a collision.)

**Step 3: Commit.**

```bash
git add include/nn/utils/logit_processing.h
git commit -m "feat(utils): speculative decoding API declarations"
```

---

### Task 4: Implement the step

**Objective:** The algorithm, transcribed from the paper.

**Files:**
- Modify: `include/nn/utils/logit_processing.cpp`

**Step 1: Implement `speculative_decode_step`.**

The implementation, in order:

1. Validate `gamma >= 1`, both callbacks non-null, and `prefix.rows == 1`.
2. Determine the vocabulary width `V` from the **draft's first** call; the
   target's width must match (else `std::runtime_error`).
3. Draw the `γ` guesses autoregressively from the draft, recording each
   guess's distribution `q_i` (the draft row that produced it) and each
   `q_i(x_i)`.
   - `q_i` for `i = 1..γ` is the draft row at `prefix + [x_1..x_{i-1}]`.
   - `q_{γ+1}` is the draft row at `prefix + [x_1..x_γ]`.
4. Score the target at `γ+1` positions, recording `p_1..p_{γ+1}`.
   - `p_i` is the target row at `prefix + [x_1..x_{i-1}]`.
5. For each guess `i`, compute the acceptance ratio and draw `r_i ~ U(0,1)`.
   - **SP3: the ratio is `min(1, p_i(x_i) / q_i(x_i))`.** Clamp at 1.
   - A `q_i(x_i) == 0` cannot happen (the guess was drawn from `q_i`, so it has
     positive mass) but must be guarded rather than dividing by zero.
6. Find `n` = the number of leading accepted guesses (SP4).
7. Build the emission distribution (SP6, SP7):
   - If `n < γ`: `p' = norm(max(0, p_{n+1} - q_{n+1}))`, componentwise, over
     the full vocabulary. **Guard the zero sum** (SP11): if the sum is 0 (or
     non-finite), fall back to `p_{n+1}` itself rather than dividing by zero.
   - If `n == γ`: `p' = p_{γ+1}`, **unmodified**.
8. Draw `t ~ p'` and emit `[x_1..x_n, t]` (SP9).
9. Return the `SpeculativeResult`.

**Reuse, do not reimplement:** `softmax` and the file-private `probabilities()`
helper already exist and already do the max-subtraction (S10). Do not write a
second softmax — and do not mutate the caller's logits.

**Step 2: Build and run the suite.**

```bash
make build/test_speculative_decoding && ./build/test_speculative_decoding
```
Expected: all checks pass.

**Step 3: Commit.**

```bash
git add include/nn/utils/logit_processing.cpp
git commit -m "feat(utils): speculative decoding step"
```

---

### Task 5: Register the test in the build

**Objective:** The suite runs in CI's aggregate target.

**Files:**
- Modify: `Makefile`

**Step 1: Add the link rule** next to the other `test_*` rules:

```make
$(BUILD_DIR)/test_speculative_decoding: $(LIB_OBJS) $(BUILD_DIR)/test_speculative_decoding.o
	$(CXX) $^ -o $@
```

**Step 2: Add it to the `tests:` prerequisite list** — otherwise the binary is
never built and `run_tests` reports "No such file or directory". (This exact
bug is recorded in `NOT_FIXED.md` for `test_mixture_of_softmaxes`.)

**Step 3: Add the `run_tests` line**:

```make
	@echo "=== Running test_speculative_decoding ==="; if ./$(BUILD_DIR)/test_speculative_decoding; then :; else echo "test_speculative_decoding" >> .run_tests_failed; fi
```

**Step 4: Verify.**

```bash
make -j$(nproc) tests && ./build/test_speculative_decoding
```

**Step 5: Commit.**

```bash
git add Makefile
git commit -m "chore: register the speculative decoding suite in the build"
```

---

### Task 6: Mutation-test the implementation

**Objective:** Prove the suite actually catches bugs in the new code.

**Files:**
- Create: `tests/mutate_speculative_decoding.sh`
- Modify: `include/nn/utils/logit_processing.cpp` (mutated by the harness, restored after)

**Step 1: Copy `tests/mutate_weight_std.sh`'s structure** — that harness
already carries the repo's twenty-plus recorded harness-defect fixes
(restore must not delete its backup; the link set comes from `make
print-lib-objs` with a count assertion; the mutated TU is compiled
separately and swapped into the link set; `MUT-BROKEN` means the build
failed and is never a verdict; a mutation that did not apply is a
HARNESS-DEFECT, not a verdict; `rm -f` the intermediates before every build;
throwing calls are wrapped so a guard mutation reports a verdict instead of
aborting).

**Critical:** the harness mutates `include/nn/utils/logit_processing.cpp`, and
that file is **also** the mutation target of `tests/mutate_logit_processing.sh`.
Both harnesses must be run **separately, never concurrently** — the repo has
already recorded a build race on shared `build/*.o` that produced two false
`MUT-BROKEN` verdicts.

**Step 2: Choose the mutations.** Each is a natural mistake, and each must be
**caught**:

- `M01` — drop the `min(1, ...)` clamp on the acceptance ratio (SP3).
- `M02` — use the residual `norm(max(0, p-q))` even when `n == γ` (SP7).
- `M03` — use `p_n`/`q_n` instead of `p_{n+1}`/`q_{n+1}` in the residual (SP6).
- `M04` — emit the rejected guess as well as the correction token (SP5/SP9).
- `M05` — compare logits instead of probabilities in the ratio (SP12).
- `M06` — drop the `max(0, ·)` in the residual (SP6).
- `M07` — accept on `r_i < p_i/q_i` (strict) instead of `<=` (SP3).
- `M08` — drop the `q_{γ+1}` draft call, so the residual is built from the
  wrong draft distribution (SP1).
- `M09` — normalise the residual by the **max** instead of the sum (SP6).
- `M10` — emit `n` tokens instead of `n+1` (SP9).

**Step 3: Run it.**

```bash
bash tests/mutate_speculative_decoding.sh
```
Expected: every mutation CAUGHT, 0 SURVIVED, 0 HARNESS-DEFECT.

**A survivor is NOT automatically a gap.** For each survivor, run a probe that
prints the affected quantity with and without the mutation; if the outputs are
bit-identical, the mutation is genuinely inert and gets its own harness
function that **inverts** the verdict (so a future refactor that makes it
matter fails loudly). Do not silently drop a survivor.

**Step 4: Commit.**

```bash
git add tests/mutate_speculative_decoding.sh
git commit -m "test(utils): mutation coverage for speculative decoding"
```

---

### Task 7: Sanitizers, full suite, and the umbrella header

**Objective:** Prove nothing is broken and nothing is undefined.

**Step 1: ASan + UBSan.**

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
    include/nn/utils/logit_processing.cpp tests/test_speculative_decoding.cpp \
    -o /tmp/test_spec_asan && /tmp/test_spec_asan
```
Expected: clean, no sanitizer output.

**Step 2: The umbrella header compiles standalone.**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: exit 0.

**Step 3: Zero warnings at the project's real flags.**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c include/nn/utils/logit_processing.cpp -o /tmp/lp.o
```
Expected: no warnings.

**Step 4: The full suite.**

```bash
make -j$(nproc) tests && make run_tests
```
Expected: every suite green **except** `test_neural_ode`, which is the
pre-existing ODERNN failure already recorded in `NOT_FIXED.md` and which was
measured failing identically on a clean tree **before** this change (baseline:
`test_neural_ode` FAIL, everything else pass).

**Step 5: Commit.**

```bash
git add -u
git commit -m "test(utils): close the speculative decoding coverage gaps"
```

---

### Task 8: Close out the queue

**Objective:** Record what shipped, so the next session does not re-pop it.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Move speculative decoding to `## Done`** with a one-line summary and
the test results, following the format of the existing entries.

**Step 2: Add the next session's VERIFIED-ABSENT list.** The remaining
candidates from the last entry, re-measured by content before being claimed:

- `grep -rinE '\bkv.?cache|paged.attn|prefix.cache|past_key' include/` — a KV
  cache. **Verify before taking**: many attention layers cache for BACKWARD
  (`last_k_`, `last_v_`), which is NOT an inference-time KV cache.
- `grep -rinE '\bcrf\b|conditional.random.field|linear.chain' include/` — a CRF
  layer. Needs a forward-backward over a chain, a different shape from
  everything else in the repo.
- `grep -rinE 'elbo|variational.inference' include/` — a standalone ELBO/VI
  utility; the repo's VAE computes its KL inline.

**Step 3: Commit.**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: close out the speculative decoding plan; move it to Done"
```

---

## 5. Risks, and what to do about each

| Risk | Detection | Response |
|------|-----------|----------|
| The `min(1, ·)` clamp is missing | `M01` must be CAUGHT | add a fixture where `p_i < q_i` at the rejected position — without it the clamp is untestable |
| The residual is used when `n == γ` | `M02` must be CAUGHT | the identical-models fixture (SP10) is the discriminator |
| The emitted count is wrong | `M04`, `M10` | assert the length is exactly `n+1` |
| The ratio uses logits | `M05` | a fixture where the logit order and the probability order differ |
| A zero-sum residual | SP11 | guard it; assert the residual is never all-zero on the rejection fixture |
| The oracle is wrong, not the C++ | oracle self-checks | the oracle has its own foils, and the two independent formulations of the acceptance rule must agree |
| The harness lies | the harness's own guards | `MUT-BROKEN` ≠ verdict; checksum before/after every edit; never run two harnesses concurrently |
| A regression elsewhere | `make run_tests` | the baseline is already measured: only `test_neural_ode` fails |

---

## 6. References

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative
  Decoding", ICML 2023, PMLR v202:19274-19286, **arXiv:2211.17192**.
  Verified by title search and by the abs page's `citation_title` /
  `citation_author` meta tags on 2026-10-25. Algorithm 1 was transcribed from
  the paper's own HTML at `https://arxiv.org/html/2211.17192`.
- The queue's own VERIFIED-ABSENT list, `EXPANSION_QUEUE.md`, 2026-10-23 entry.
- The repo's `docs/plans/2026-10-23_typical-p-locally-typical-sampling.md`,
  for the oracle-first and mutation-harness conventions this plan follows.
- `tests/mutate_weight_std.sh`, for the harness-defect list this plan inherits.
