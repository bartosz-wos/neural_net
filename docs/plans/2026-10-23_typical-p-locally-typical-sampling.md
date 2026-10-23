# Typical-p (Locally Typical) Sampling Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `apply_typical_p` — the locally typical sampling truncation of Meister et al. — to `include/nn/utils/logit_processing.{h,cpp}`, closing the one out-of-scope item that the file's own header names.

**Architecture:** A new free function `apply_typical_p(const Tensor& logits, double tau, double temperature = 1.0)` in `nn::utils`, following the exact shape of the existing `apply_min_p`: it computes the conditional entropy of the (temperature-adjusted) distribution, scores every token by its absolute distance from that entropy in log-probability space, sorts ascending by that distance, and greedily accumulates probability mass until it reaches `tau`. It returns a NEW tensor with the non-surviving entries set to `-inf`, exactly like every other truncating processor, so it drops into `LogitProcessorChain` unchanged.

**Tech Stack:** C++17, the repo's `Tensor` class, the existing `logit_processing.cpp` helpers, a numpy float64 oracle, and the repo's mutation harness.

---

## 0. The gap, measured (not assumed)

The pick came from the queue's standing **docs-gap grep**, which the file itself records as its highest-yield pick source (it found `lr_multiplier` via `mup.h:239` and `hutchinson` via `sophia.h:25`):

```
grep -rinE 'not entitled|out of scope|deliberately not|is not exposed|for now we do not|this repo does not|no hook' include/nn --include=*.h
-> 19 hits, all already accounted for in shipped headers
```

None of those 19 was new — so the grep's yield fell to zero for the fourth consecutive run, exactly as it did on 2026-10-22. The fallback was the file's own **VERIFIED-ABSENT** list plus a broader concept sweep. The concept sweep returned several absent families (CRF, VAE-ELBO, KV-cache, speculative decoding, typical-p), and one of them is *named inside a shipped header as unimplemented*:

```
grep -rn 'typical' include/nn/utils/logit_processing.h
-> include/nn/utils/logit_processing.h:149:
   //   - Typical-p sampling (Meister et al.). The signature is reserved in the
   //     header so it is not silently reinvented later; it is NOT implemented,
   //     because it is not yet oracle-pinned.
```

**This is the strongest possible shape of gap**: the repo's own newest utility file (committed `2e2b490`, 2026-10-20) states in its own words that this one algorithm is missing and *why* it was not yet implemented — "it is not yet oracle-pinned". This plan supplies the oracle, which is precisely the stated blocker.

A wider sweep confirms the concept is absent everywhere, not just in this file:

```
grep -rinE 'typical.?sampl|typical.?p|locally.?typical|typicality' include/ tests/ --include=*.h --include=*.cpp
-> 0 content hits (the only two matches are the grep pattern quoted inside
   logit_processing.h's own comment block)
```

### Not redundant

The repo's other truncating processors cut on **probability mass** (top-k, top-p, min-p, epsilon). Typical-p cuts on **information content relative to the model's own conditional entropy** — a different axis, and the one the paper's central claim rests on ("Algorithm 5.3 does not imply that high-probability words should not be chosen"). A distribution where the argmax is NOT in the typical set is reachable, and the suite asserts that it is reachable — a test that would be vacuous if typical-p were just another mass cut.

### Citation — verified by TITLE, never typed from memory

This queue has produced **twelve wrong arXiv IDs**, every one a plausible-looking neighbour of the right paper, so the rule is: search by title, then confirm the abs page's `citation_title` meta tag.

- **Meister, Pimentel, Wiher, Cotterell, "Locally Typical Sampling"**, *Transactions of the Association for Computational Linguistics* **11**:102-121, 2023; ACL Anthology `2023.tacl-1.7`; **arXiv:2202.00666** (v6, 5 Jun 2025).

Verified two ways on 2026-10-23:
1. Title search: `arxiv.org/abs/2202.00666` → "Locally Typical Sampling", authors "Clara Meister, Tiago Pimentel, Gian Wiher, Ryan Cotterell" in order.
2. The paper's own HTML (`https://arxiv.org/html/2202.00666v6`, 524 KB, HTTP 200) was fetched and the algorithm read from it — **Algorithm 5.3, Definition 3.5 / Eq. 7, and the erratum attached to Algorithm 5.3**.

The algorithm was read from the paper's HTML rather than from memory, and it carries a **published erratum** that the implementation must respect (see Task 3).

---

## 1. The algorithm, as the paper states it

**Eq. 7 / Definition 3.5 (locally typical set).** A token `y` is locally typical at step `t` when

```
| log p(y | y_<t) + H(Y_t | Y_<t) |  <  ε
```

i.e. its **negative log-probability** is within `ε` of the model's **conditional entropy**. The quantity `log p + H` is the *deviation* of the token's information content from the expected information content.

**Algorithm 5.3 (the truncation set actually used for sampling).** This is the part that is easy to get wrong, and the paper's own erratum is about it:

- The paper's *stated* optimization problem (minimize the sum of `|log p(y) + H|` subject to mass `≥ τ`) is **not** what implementations do, and the paper says so: *"This definition of the optimization problem ... allows for solutions in which continuations whose log-probabilities lie closest to the conditional entropy are excluded from the set. The authors are working on a new formulation..."*
- The paper's **Computational Complexity** paragraph states the procedure that is actually implemented, and it is the one the maintained reference implementation follows:

  > "First, we compute the conditional entropy, which is an O(|V|) operation. **Second, we sort words by their absolute distance from H(p̂(·|Y_<t=y_<t))**, which can be done in O(|V| log |V|) time with standard sorting algorithms. **Finally, we greedily take words from this list until their cumulative probability exceeds the threshold τ**."

So the algorithm is:

```
1. p      = softmax(logits)                       # after any temperature
2. H      = -Σ p_i log p_i                          # conditional entropy, nats
3. d_i    = | -log p_i  -  H |                      # |log p_i + H|  (Eq. 7)
4. order  = argsort(d_i) ASCENDING                 # closest to the entropy first
5. keep   = greedy prefix of `order` until cumulative mass >= τ
6. return logits with non-kept entries set to -inf
```

**Cross-checked against a maintained reference implementation** (HuggingFace `transformers` 5.14.1, `generation/logits_process.py::TypicalLogitsWarper`), which does exactly steps 1-5:

```python
normalized = torch.nn.functional.log_softmax(scores, dim=-1)
p = torch.exp(normalized)
ent = -(normalized * p).nansum(-1, keepdim=True)

shifted_scores = torch.abs((-normalized) - ent)          # d_i
sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
sorted_logits = scores.gather(-1, sorted_indices)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

last_ind = (cumulative_probs < self.mass).sum(dim=1)
last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
```

Two details this pins that the prose alone does not:

- **The sort is ASCENDING by distance** (closest-to-entropy first), not descending.
- **The cut is `cumulative_probs < mass`, i.e. keep the prefix whose mass first reaches `τ`** — the same `>=`-style "keep the crossing token" convention as the repo's nucleus (S1).

---

## 2. Semantics to pin (S15..S21)

Every label below is a natural mistake, and each gets its own named test. The oracle must show that the *wrong* variant differs on the fixture before the test can mean anything.

| Label | Semantics | The natural mistake it forecloses |
|---|---|---|
| **S15** | The ordering key is `abs(-log p_i - H)`, i.e. the distance from the entropy in nats. | Using `abs(log p_i + H)` is the *same* number algebraically — but using `abs(log p_i) - H` (distance of the raw log-prob, un-negated) is NOT, and flips the ranking. |
| **S16** | The sort is **ascending** by that distance: the token CLOSEST to the entropy is kept first. | Sorting descending keeps the LEAST typical tokens first — the exact opposite of the algorithm, and it produces a plausible-looking non-empty set, so it survives a naive test. |
| **S17** | The cut is the smallest **ascending** prefix whose cumulative mass reaches `tau` (`>=`). | A `>` here admits one token too many — the repo's existing S1 off-by-one, in a new place. |
| **S18** | The entropy is computed on the **probability** distribution, in nats, with `0 log 0 = 0` (a `-inf` logit contributes 0, not NaN). | Summing over the raw logits, or letting `0 * -inf` produce NaN, makes the whole output NaN. |
| **S19** | `tau` is the probability mass, in `[0, 1]`. `tau >= 1` keeps everything (a no-op); `tau <= 0` is a no-op. | Rejecting `tau == 1` as out of range, or treating `tau` as a *distance* threshold. |
| **S20** | The result masks by **survival**, and a token with probability 0 is never kept (its log-probability is `-inf`, its distance is `inf`). | Keeping a zero-probability token because it happens to be "closest" to the entropy in a degenerate sense. |
| **S21** | The input is **never mutated**; the shape is preserved; `(1,V)` and `(V,1)` both work. | Mutating in place breaks `LogitProcessorChain` reentrancy (the repo's S8 dependency). |

**S15/S16 are the two that need the most care**, and they interact: they are the reason a naive "typical-p is just top-p with a weird order" implementation passes a survivor-count test while being wrong. The suite must assert the **exact survivor set**, not a count.

### The fixture that makes S16 non-vacuous

S16 needs a fixture where the ascending and descending orderings give **different survivor sets**. Constructed and MEASURED in the oracle, not hand-guessed: a distribution with a moderate entropy, so the token closest to the entropy is *neither* the argmax *nor* the rarest token. A skewed distribution (`[0.7, 0.15, 0.10, 0.05]`) is entropy-low, so its typical set is close to its argmax and the two orderings coincide — **that fixture would be vacuous for S16**, and the oracle must say so.

---

## 3. Tasks

### Task 1: Write the oracle (RED for the C++)

**Objective:** A numpy float64 oracle that pins every number the C++ test will assert.

**Files:**
- Create: `tests/typical_p_oracle.py`

**Step 1: Write the oracle**

It must implement the algorithm from §1 and, for each pinned case, print a C++-pasteable table. It must also implement the **wrong** variants (descending order, `>` cut, un-negated log-prob) and **assert that they differ** on the pinned fixture — a test that cannot fail is worse than no test, and this repo has hit that trap repeatedly (the nucleus key-shape bug, the Witten-Bell OOV class, the `n_finite` inequality).

Structure it exactly like `tests/logit_processing_oracle.py` (the sibling file it extends): a `check(cond, label)` closure, a `checks`/`fails` tally, self-checks on every claim, exit 0 iff every self-check passes.

Required content:

1. `typical_p_survivors(probs, tau)` — the correct algorithm.
2. `typical_p_survivors_WRONG_descending(...)` — S16's foil.
3. `typical_p_survivors_WRONG_strict(...)` — S17's foil (`>` instead of `>=`).
4. `typical_p_survivors_WRONG_unnegated(...)` — S15's foil.
5. The **entropy** value and the **per-token distances** for the pinned fixture, printed with `repr()` so the C++ can copy exact literals.
6. A **round-trip-safety check**: the same lesson as the nucleus fixture — the probability distribution must be recoverable from `softmax(logits)` exactly enough to discriminate. This is why the fixture is built from **logits** whose softmax is measured, not from probabilities assumed to survive a round trip.

**Step 2: Run the oracle**

Run: `/home/stefan/.venv/bin/python tests/typical_p_oracle.py`
Expected: exit 0, a printed table, and every self-check passing.

**Step 3: Commit**

```bash
git add tests/typical_p_oracle.py
git commit -m "test(utils): numpy oracle for typical-p (locally typical sampling)"
```

---

### Task 2: Write the failing C++ test (RED)

**Objective:** A test that fails because `apply_typical_p` does not exist.

**Files:**
- Modify: `tests/test_logit_processing.cpp` (add a `test_typical_p_*` function and its call in `main`)

**Step 1: Write the test**

The test must:
- Use the **exact literals printed by the oracle** — never a hand-derived number. This repo has hit the "hand-arithmetic slip with the right shape" class repeatedly (the `sqrt(0.4)` vs `-sqrt(3/2)` slip, the `6/25` vs `2/25` slip).
- Assert the **exact survivor set** (not a count) for the pinned fixture, at several `tau` values.
- Assert the entropy value and at least one per-token distance.
- Assert the S15/S16/S17 foils actually differ (by comparing against the oracle's wrong-variant sets) — so the test is known to discriminate.
- Assert S18: a row containing `-inf` (a masked token) does not produce NaN.
- Assert S19: `tau = 1.0` is a no-op; `tau <= 0` is a no-op; `tau` out of range throws.
- Assert S20: a zero-probability token is never kept.
- Assert S21: the input is unmutated and the shape is preserved, for both `(1,V)` and `(V,1)`.
- Assert the interaction with `LogitProcessorChain`: a chain containing `add_typical_p` works, and the **chain order** matters (S8's existing contract) — a typical-p then top-p chain differs from top-p then typical-p on a fixture where the two cuts disagree.

**Step 2: Run the test to verify it FAILS**

Run: `make build/test_logit_processing && ./build/test_logit_processing`
Expected: **COMPILE ERROR** — `apply_typical_p` is not declared. That is the correct RED for a new API: the feature is missing.

**Step 3: Commit**

```bash
git add tests/test_logit_processing.cpp
git commit -m "test(utils): typical-p tests (RED)"
```

---

### Task 3: Implement `apply_typical_p` (GREEN)

**Objective:** Minimal implementation that passes the Task 2 test.

**Files:**
- Modify: `include/nn/utils/logit_processing.h` (declaration + semantics block)
- Modify: `include/nn/utils/logit_processing.cpp` (definition)

**Step 1: Declare in the header**

Add to `include/nn/utils/logit_processing.h`, in the truncating-processors section (after `apply_min_p`):

```cpp
// Typical-p / locally typical sampling (S15..S21). Keeps the tokens whose
// NEGATIVE LOG-PROBABILITY is closest to the model's own CONDITIONAL ENTROPY,
// greedily until their cumulative probability mass reaches `tau`.
//
// The ordering key is `| -log p_i - H |` in nats, sorted ASCENDING, and the cut
// is the smallest prefix whose mass reaches `tau` (S17). This is NOT a mass
// cut like top-p: it ranks tokens by how close their information content is to
// the EXPECTED information content, so the argmax is not privileged.
//
// `temperature` is applied to the LOGITS FIRST, so the entropy is that of the
// temperature-adjusted distribution.
//
// tau >= 1 is a no-op (everything is kept) and tau <= 0 is a no-op.
Tensor apply_typical_p(const Tensor& logits, double tau, double temperature = 1.0);
```

Also add `LogitProcessorChain& add_typical_p(double tau, double temperature = 1.0);` to the chain class, and update the "OUT OF SCOPE FOR v1" comment block to remove the typical-p line (keeping the speculative-decoding line, which is NOT implemented by this plan).

**Step 2: Implement in the .cpp**

Reuse the file's existing private helpers (`vocab_of`, `get_at`, `set_at`, `like`, `probabilities`) — do NOT re-implement them, and do NOT add a new anonymous-namespace helper that duplicates one.

The implementation, following §1 exactly:

```cpp
Tensor apply_typical_p(const Tensor& logits, double tau, double temperature) {
    if (tau < 0.0 || tau > 1.0) {
        throw std::invalid_argument("apply_typical_p: tau must be in [0, 1]");
    }
    if (!(temperature > 0.0) || !std::isfinite(temperature)) {
        throw std::invalid_argument(
            "apply_typical_p: temperature must be finite and > 0");
    }
    std::vector<double> lg = to_vec(logits);
    for (size_t j = 0; j < lg.size(); ++j) {
        if (!std::isinf(lg[j])) lg[j] /= temperature;
    }
    std::vector<double> probs = probabilities(lg);

    // S18: the entropy is over the PROBABILITY distribution, in nats. A zero
    // probability contributes 0 (0 * log 0 == 0), never NaN.
    double entropy = 0.0;
    for (size_t j = 0; j < probs.size(); ++j) {
        if (probs[j] > 0.0) entropy -= probs[j] * std::log(probs[j]);
    }

    // S15: the ordering key is the distance from the entropy in nats.
    // S20: a zero-probability token has an INFINITE distance and is never
    // kept -- sorting it first would let a token with no mass at all survive.
    std::vector<double> dist(probs.size());
    for (size_t j = 0; j < probs.size(); ++j) {
        dist[j] = (probs[j] > 0.0) ? std::fabs(-std::log(probs[j]) - entropy)
                                   : std::numeric_limits<double>::infinity();
    }
    // S16: ASCENDING by distance -- the token CLOSEST to the entropy first.
    std::vector<size_t> order(probs.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&dist](size_t a, size_t b) { return dist[a] < dist[b]; });

    Tensor out = like(logits);
    for (size_t j = 0; j < probs.size(); ++j) set_at(out, j, kNegInf);

    if (tau <= 0.0) {   // S19: no-op -- but keep the ORIGINAL logits
        for (size_t j = 0; j < probs.size(); ++j)
            set_at(out, j, get_at(logits, j));
        return out;
    }

    double cum = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        size_t idx = order[i];
        set_at(out, idx, get_at(logits, idx));
        cum += probs[idx];
        // S17: `>=`, not `>`. The token that REACHES the threshold is kept.
        if (cum >= tau) break;
    }
    return out;
}
```

Notes on the above, each of which is a real decision:
- `tau <= 0` is a no-op that must return the ORIGINAL logits, not an all-`-inf` row. An all-`-inf` row makes the downstream sampler return index 0 silently — a total failure that looks like a working row.
- `tau >= 1` falls out of the loop naturally: the full mass reaches `>= 1` at the last token, so everything is kept. The oracle must confirm this holds for a distribution whose mass sums to exactly 1.0 in float64.
- The `-inf` guard on the temperature division mirrors `apply_min_p`'s, so a truncated token survives the processor (the repo's S9).
- `std::stable_sort` with a strict `<` comparator keeps ties first-wins on the original index, consistent with the file's S12 convention.

**Step 3: Add the chain method**

```cpp
LogitProcessorChain& LogitProcessorChain::add_typical_p(double tau, double temperature) {
    processors_.push_back(
        [tau, temperature](const Tensor& x) { return apply_typical_p(x, tau, temperature); });
    return *this;
}
```

**Step 4: Run the test to verify it PASSES**

Run: `make build/test_logit_processing && ./build/test_logit_processing`
Expected: `=== Summary: N passed, 0 failed ===` with N = the previous 210 + the new checks.

**Step 5: Commit**

```bash
git add include/nn/utils/logit_processing.h include/nn/utils/logit_processing.cpp
git commit -m "feat(utils): typical-p (locally typical sampling) truncation"
```

---

### Task 4: Mutation-test the new code (the coverage proof)

**Objective:** Prove the new suite actually exercises the new code, by mutating it and requiring the suite to catch every mutation.

**Files:**
- Modify: `tests/mutate_logit_processing.sh` (add mutations)

**Step 1: Add mutations to the harness**

The harness is the repo's accumulated knowledge about how a harness lies; copy its structure, do not re-derive it. Add at minimum:

| ID | Mutation | What it proves |
|---|---|---|
| **M22** | `dist[j] = ... std::fabs(-std::log(probs[j]) - entropy)` → `std::fabs(std::log(probs[j]) - entropy)` | S15 — the log-probability is NEGATED before the distance. |
| **M23** | `return dist[a] < dist[b];` → `return dist[a] > dist[b];` | S16 — the sort is ASCENDING, not descending. |
| **M24** | `if (cum >= tau) break;` → `if (cum > tau) break;` | S17 — the crossing token is kept. |
| **M25** | `entropy -= probs[j] * std::log(probs[j]);` → `entropy += probs[j] * std::log(probs[j]);` | S18 — the entropy's SIGN. |
| **M26** | the `probs[j] > 0.0` guard in the entropy loop → removed | S18 — a zero probability must contribute 0, not NaN. |
| **M27** | `dist[j] = ... ? ... : infinity()` → `dist[j] = 0.0` for the zero-probability case | S20 — a zero-probability token is never kept. |
| **M28** | `if (tau <= 0.0)` guard → `if (tau < 0.0)` | S19 — `tau == 0` is a no-op, not an all-`-inf` row. |

**M23 is the important one.** It is the mutation a "survivor count" test cannot see, and the repo's own history says so: the nucleus S8 check "compared two hand-composed calls" and a right-to-left chain passed it; the min-p check "asserted an inequality, which holds whether or not the temperature is honoured at all". If M23 survives, the suite is weak and the test must be strengthened before the feature is called done.

**Step 2: Run the harness**

Run: `bash tests/mutate_logit_processing.sh`
Expected: `MUTATION SUMMARY: N caught / 0 survived / 0 broken / 0 crash / 0 noop / 0 inert-confirmed`, and `ASAN pristine build: CLEAN`.

**If any mutation SURVIVES**, do not explain it away. Per the repo's own recorded rule: *"Do not silently drop a survivor — an unexplained survivor is indistinguishable from a real gap."* Either strengthen the test so it is caught, or **measure** that the mutation is genuinely inert and score it with the harness's `run_mutation_expect_inert` (the inverted verdict) — with the measurement recorded, never the assumption.

**Step 3: Commit**

```bash
git add tests/mutate_logit_processing.sh
git commit -m "test(utils): mutation coverage for typical-p"
```

---

### Task 5: Clean-build verification and the full suite

**Objective:** No regressions anywhere, and no sanitizer findings.

**Step 1: Umbrella header compiles standalone**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: silent success. (A focused test that includes only the new header cannot catch a cross-header redefinition — the queue records this as a required verification step.)

**Step 2: Zero warnings at the project's real optimization level**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c include/nn/utils/logit_processing.cpp -o /tmp/lp.o
```
Expected: no warnings. (`-O2 -march=native` is what the project actually ships; a hand-rolled `-O0` build is not the build users run.)

**Step 3: ASan + UBSan on the pristine tree**

The harness already runs this lane, but run it standalone to see the result:

```bash
bash tests/mutate_logit_processing.sh 2>&1 | grep -A3 'ASAN'
```
Expected: `ASan+UBSan on the PRISTINE tree: CLEAN`.

**Step 4: The full suite**

```bash
make run_tests
```
Expected: no NEW failures. The known pre-existing failures are recorded in `NOT_FIXED.md` and must not change: `test_neural_ode` (ODERNN grad check), `test_coord_network` (SIREN), `test_wgan_gp` (test 8, intermittent), `test_vit` (double-linked list corruption). Any NEW failure is a regression from this change and must be fixed before shipping.

**Step 5: Commit**

```bash
git add tests/typical_p_oracle.py tests/test_logit_processing.cpp
git commit -m "test(utils): typical-p oracle + suite verified against the reference implementation"
```

---

### Task 6: Update the queue and the docs

**Files:**
- Modify: `EXPANSION_QUEUE.md` (move the item to `## Done` with a one-line summary and test results)
- Modify: `include/nn/utils/logit_processing.h` (the header's own "the gap this fills" comment block gains the typical-p measurement)

**Step 1: Update the queue**

Add a `## Done` entry recording: what shipped, the file, the oracle, the test counts, the mutation counts, the sanitizer result, and — the part the queue says is worth keeping — **what the oracle caught in itself**, and the corrected citation.

**Step 2: Update the header's provenance comment**

The header's "THE GAP THIS FILLS, MEASURED NOT ASSUMED" block should gain a line recording that typical-p was added on 2026-10-23, and that it was the file's own out-of-scope note that named it.

**Step 3: Commit**

```bash
git add EXPANSION_QUEUE.md include/nn/utils/logit_processing.h
git commit -m "docs: close out the typical-p plan; move typical-p to Done in the queue"
```

---

## 4. Definition of done

- [ ] `apply_typical_p` and `LogitProcessorChain::add_typical_p` exist, are declared in the header, and are documented with the S15..S21 semantics.
- [ ] The oracle exists, exits 0, and every C++ expectation is copied from its output.
- [ ] The test suite passes with 0 failures.
- [ ] Every mutation M22..M28 is CAUGHT, or measured-inert with the inverted verdict and the measurement recorded.
- [ ] ASan+UBSan CLEAN on the pristine tree.
- [ ] Zero warnings at `-O2 -march=native -Wall -Wextra`.
- [ ] The umbrella header compiles standalone.
- [ ] `make run_tests` shows no new failures.
- [ ] The queue's `## Done` records the feature, and the header's provenance comment records the measurement.
- [ ] Committed atomically and pushed to `origin master`.

## 5. Reference files

- `include/nn/utils/logit_processing.h` / `.cpp` — the file being extended; conventions, S-labels, helper reuse.
- `tests/logit_processing_oracle.py` — the oracle to extend; the `check(cond, label)` pattern and the round-trip-safety lesson.
- `tests/test_logit_processing.cpp` — the test to extend; the `CHECK` macro, the `row()` helper, the `survivors()` helper.
- `tests/mutate_logit_processing.sh` — the harness to extend; the mutation spec format, the inverted-verdict function, the post-harness rebuild defence.
- `docs/plans/2026-10-20_logit-processing-decoding.md` — the plan whose S-labels this one continues.
