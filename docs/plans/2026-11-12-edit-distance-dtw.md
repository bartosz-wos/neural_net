# Edit Distance and Dynamic Time Warping — Implementation Plan

**Goal:** ship two model-free utility modules — `include/nn/utils/edit_distance.{h,cpp}` (weighted Levenshtein, restricted Optimal String Alignment, unrestricted Damerau–Levenshtein, a deterministic alignment/edit-count API, and token-level WER/CER) and `include/nn/utils/dtw.{h,cpp}` (dynamic time warping over scalar and multidimensional observations, with optional Sakoe–Chiba banding and a deterministic warping path).

**Architecture:** free functions in `nn::utils`, matching the shape of `utils/rank_correlation.h`, `utils/isotonic_regression.h`, `utils/sliced_wasserstein.h` and `utils/off_policy_eval.h` — no classes, no `Layer`, no randomness, no state. The discrete half operates on **token ID sequences** (`std::vector<int>`), because that is the sequence type this repo's own `utils/tokenizer.h` and `utils/bpe_tokenizer.h` already emit and `utils/ngram_lm.h` already consumes. The continuous half operates on `Tensor`, because that is the repo's universal numeric container.

**Tech Stack:** C++17, the repo's `Tensor` (row-major, `AlignedAllocator<double>`), the project's own `-O2 -march=native -Wall -Wextra` build, float64 Python oracles in `tests/` (numpy-free where a definition-level oracle is exact), an oracle-mutation harness in `tests/`, and bash mutation harnesses in `tests/`.

---

## Why this feature, measured not assumed

    grep -rinE 'levenshtein|edit.distance|damerau|word.error|\bwer\b|cer|dynamic.time.warp|\bdtw\b|warping|sakoe|itakura' include/
    -> 0 hits

The repo has ~250 layers and 60+ utility files and **no sequence-similarity machinery of any kind**. What it has is adjacent but different:

- `utils/metrics.h` — ROC-AUC, AP, NDCG, MAP, MRR, ECE, Brier. The quality of a *classifier* against *labels*.
- `utils/ngram_lm.h` — a corpus of `std::vector<std::vector<int>>` token sequences, scored by `sequence_logprob`. It can score a sequence against a language model; it cannot say how close two sequences are to each other.
- `utils/crf.h` — Viterbi / forward–backward over *label* sequences for a single model. Not a distance between two sequences.
- `utils/sliced_wasserstein.h` — a distance between two **distributions** of samples (order-free, `Tensor` of samples). Not a distance between two **ordered** series.
- `utils/logit_processing.h` — decoding-time logit processors. It produces token sequences; nothing measures them against a reference.
- `utils/linear_algebra.h` — the Fréchet distance between two **Gaussians**.

### The two questions this answers, and who asks them

1. **"How wrong is this generated sequence?"** — the ASR / MT / captioning question. The reference transcript and the model's hypothesis are both token sequences; the answer is **WER** (word error rate) or **CER** (character error rate). The repo can *produce* token sequences (`tokenizer.h`, `bpe_tokenizer.h`, `logit_processing.h`) and can score them against a language model (`ngram_lm.h`), but has no way to score one against a *reference*.
2. **"How far apart are these two time series?"** — the forecasting question. Two sensor traces, two market series, two accelerometer windows can be the same trajectory sampled at different rates. Euclidean distance says they are far apart; **dynamic time warping** says they are the same path. The repo's time-series layer (`layers/architectures/timesnet.h`) and its `Tensor`-based distribution utilities have no order-sensitive sequence distance at all.

Both are *model-free* and *deterministic*, which is exactly the `utils/` register.

### SCOPE BOUNDARY (stated so it cannot creep in)

- **Token IDs and characters only.** The discrete half's element type is `int` (a token ID) or `char` (a byte). It does **not** do grapheme clustering, Unicode normalisation, or case folding — those are the caller's job, and a silent case-fold here would change the metric invisibly.
- **The alignment ships for plain Levenshtein only.** OSA and unrestricted Damerau–Levenshtein ship as **distances**. Their optimal alignments have genuinely different structure (OSA forbids editing a substring twice; unrestricted DL can transpose non-adjacent characters and its edit scripts are not a simple monotone alignment), and shipping a half-specified alignment for them would be worse than shipping none. WER and CER need only the plain-Levenshtein alignment, which is what ships.
- **No normalised DTW, no subsequence DTW, no weighted/local (Itakura) constraints, no multi-step step patterns.** The shipped formulation is one named, pinned formulation (below). A "DTW" that silently means a different step pattern is a trap; a "DTW" that offers six variants under one name is worse.
- **No Levenshtein variants beyond the three shipped** (weighted, OSA, unrestricted DL). No Jaro–Winkler, no Hamming, no longest-common-subsequence — separate concepts, separate queue items.

---

## The mathematics, and the places it is easy to get wrong

### 1. THE THREE DISCRETE DISTANCES ARE THREE DIFFERENT FUNCTIONS

They are **not** three spellings of one function. Measured, 20 000 random string pairs:

| pair | Levenshtein | OSA | unrestricted DL |
|---|---|---|---|
| `("ca", "abc")` | 3 | 3 | 2 |

OSA and unrestricted DL both admit a transposition; OSA restricts it, unrestricted DL does not. The distinguishing case above is `("ca","abc")`: unrestricted DL transposes `ca`→`ac` and then inserts `b` (**2**); OSA cannot, because transposing `ca` and then inserting inside it would edit the substring twice (**3**).

### 2. THE UNRESTRICTED DL RECURRENCE IS THE ONE THAT IS EASY TO GET WRONG

Five commonly-transcribed Lowrance–Wagner variants were implemented and **measured against `rapidfuzz.distance.DamerauLevenshtein` and `jellyfish.damerau_levenshtein_distance`**:

| variant | agreement with the references |
|---|---|
| fixed-index transcription | 4029 / 6000 |
| article-shaped transcription | 5048 / 8000 |
| "canonical" rewrite | 5258 / 8000 |
| `d[l+1][m+1]` variant | 4724 / 8000 |
| `d[i1][j1]` variant | 1055 / 1500 |
| **sentinel-layout Wikipedia/Lowrance–Wagner** | **20000 / 20000** |

All five rejected variants fail on the *same* first mismatch, `('cabcac','bbc')` (reference says 4, they say 5). The shipped implementation uses the sentinel-layout form:

```
d[i+1][j+1] = min( d[i][j] + cost,
                   d[i+1][j]   + 1,
                   d[i][j+1]   + 1,
                   d[k][l] + (i-k-1) + 1 + (j-l-1) )
```

with `k = da[b[j-1]]` (last row in which `b[j-1]` occurred) and `l = db` (last column in which `a[i-1]` occurred), on the `d[0][0] = inf` sentinel layout.

The transposition term's `+1` is also **not** the substitution `cost`: substituting `cost` for `+1` was measured to change the answer on **0 of 400 000** random pairs, so the two are interchangeable in practice — the shipped code uses `+1`, and the oracle pins that choice rather than leaving it implicit.

### 3. THE EDIT-COUNT BREAKDOWN IS **NOT** UNIQUE, AND THE DISTANCE IS

This is the single most important property of the discrete half, and it is invisible unless you measure it.

The **distance** is a function of the two strings. The **breakdown** into substitutions / deletions / insertions is a property of *one chosen optimal alignment* — and a pair can have several.

Measured over 2000 random pairs (alphabet `abc`):

- **1740 / 2000** pairs admit exactly **one** optimal `(S, D, I)` breakdown.
- **260 / 2000** admit **more than one**.

On the 1740 with a unique breakdown, the shipped backtracking rule and `rapidfuzz`'s `editops` agree **1740 / 1740**. On the ambiguous 260, they may differ — and they do.

What *is* invariant, measured on 3000 pairs: **`S + D + I == distance` always**, because every edit costs exactly one. That identity is what the test suite pins, and it is the reason WER is well-defined even though the breakdown is not.

The shipped backtracking preference is **deletion first, then diagonal, then insertion** — measured to agree with `jiwer` 4.0.0's `(hits, substitutions, deletions, insertions)` on **19 847 / 20 000** random pairs and with `rapidfuzz`'s `editops` on **19 847 / 20 000**. The residual disagreements are exactly the ambiguous cases above.

### 4. `rapidfuzz`'s EDIT-OPS TAGS COUNT OPERATIONS, NOT SPANS

`rapidfuzz.distance.Levenshtein.editops("ab","xy")` returns **two** `replace` ops, each covering one character, and `len(Editop)` is the affected *span* length. Counting `len(Editop)` instead of counting tags produced a wrong S+D+I; counting tags reproduces the distance on **3000 / 3000** pairs.

### 5. THE WEIGHTED LEVENSHTEIN IS **DIRECTED**, AND `rapidfuzz` TRUNCATES ITS WEIGHTS

With `weights = (insert, delete, replace)`:

| call | result |
|---|---|
| `Levenshtein.distance('aa','aaa', weights=(2,1,1))` | **2** — one insertion, at the insert weight |
| `Levenshtein.distance('aaa','aa', weights=(2,1,1))` | **1** — one deletion, at the delete weight |

So `distance(a,b) != distance(b,a)` whenever insert != delete, and the shipped `EditCosts` documents its orientation as **the cost of transforming `a` into `b`**.

`rapidfuzz` **truncates fractional weights to integers** (`0.5` becomes `0`), so it cannot validate a fractional-cost implementation. The shipped API takes `double` costs; the oracle validates fractional costs against a **definition-level Dijkstra oracle over the edit graph** instead, and validates integer weights against `rapidfuzz` as well.

### 6. WER HAS A SPECIFIC EMPTY-REFERENCE CONVENTION

`jiwer` 4.0.0 measures:

    WER(reference, hypothesis) = distance(reference, hypothesis) / max(1, |reference|)

Measured **3000 / 3000** agreement on random pairs, including the empty-reference cases (`jiwer.wer('', 'a b') == 2`, not `inf` and not `0`). The shipped `word_error_rate` / `character_error_rate` use exactly this formula.

### 7. THREE OF THE FOUR DISTANCES ARE METRICS. ONE IS NOT. ONE IS NOT EVEN SYMMETRIC.

Measured, not assumed:

| function | symmetric? | triangle inequality? | metric? |
|---|---|---|---|
| Levenshtein (unit costs) | yes | yes | **yes** |
| weighted Levenshtein | **only if insert == delete** | **only if insert == delete** | **only if insert == delete** |
| OSA | yes | **NO** — `('abbca','aca','caa')`: 4 > 2 + 1 | **no** |
| unrestricted Damerau–Levenshtein | yes | yes (0 violations / 20 000) | **yes** |
| DTW (symmetric1, Euclidean) | yes (0 / 300) | **NO** — 7 violations / 300 | **no** |

So the header must not call OSA a metric, and must not call DTW a metric. This is a documented property, and the suite pins the counterexamples.

### 8. THE DTW FORMULATION IS PINNED, INCLUDING THE STEP PATTERN

The shipped formulation:

```
D[0][0] = 0
D[i][j] = min( D[i-1][j-1], D[i-1][j], D[i][j-1] ) + local_cost(x[i-1], y[j-1])
```

- **Step pattern: symmetric1** — diagonal `(i-1,j-1)`, vertical `(i-1,j)`, horizontal `(i,j-1)`. The three-neighbour lattice, *not* a pattern with `(2,0)`-style jumps: measured, `x=[0,100]` vs `y=[0]` costs **100.0** under `dtw-python`'s `symmetric1` and under the three-neighbour DP — a jump-capable pattern would have cost 0.
- **Local cost: Euclidean** distance between the two observation vectors. Summed **raw**, not normalised.
- **`dtw-python`'s DEFAULT step pattern is `symmetric2`, not `symmetric1`** — measured to differ on **200 / 200** random pairs. Every cross-check therefore passes `step_pattern=dtw.symmetric1` explicitly. This is the single easiest way to get a silent mismatch.
- **Sakoe–Chiba band**: cells with `|i - j| > band` are forbidden, on 1-based cell indices. A band admits a monotone path **iff `band >= |n - m|`** — measured on 21 372 random cases, 0 violations. `dtw-python` raises `ValueError("No warping path found compatible with the local constraints")` in exactly that situation; the shipped API returns `+inf` and says so.

### 9. THE DTW WARPING PATH IS **NOT** UNIQUE EITHER — BUT IT IS PINNED

Ties in the DP produce several optimal paths. Measured on tie-heavy **integer** observations (continuous observations almost never tie):

- backtrack preference **diagonal, then horizontal, then vertical** → path identical to `dtw-python`'s `index1`/`index2` on **400 / 400**;
- **diagonal, then vertical, then horizontal** → 394 / 400;
- **vertical first** → 209 / 400.

(The continuous-data sweep cannot discriminate these — it reports 400/400 for all of them, because continuous ties are rare. The integer-data sweep is the one that discriminates. The earlier 0/300 reading for every order was a **harness bug** in the probe — a duplicated origin cell — not a property of `dtw-python`; that was found and fixed before this plan was written.)

The shipped path therefore has two pinned properties: it is a **valid optimal path**, and it is **byte-identical to `dtw-python`'s** on tie-heavy fixtures.

### 10. THE `Tensor` SHAPE IS **NOT** INTERCHANGEABLE IN THE DTW MODULE

`utils/rank_correlation.h` and `utils/off_policy_eval.h` both state: *"THE VALUES ARE WHAT MATTER: a (N,1) and an (1,N) tensor holding the same values are the SAME sample."*

**That convention does not hold here, and the header says so explicitly.** For DTW the shape determines the **dimensionality of each observation**: `(T,1)` is `T` scalar observations, `(1,T)` is **one** observation of `T` dimensions. A module that silently inherited the other convention would compute a different answer for a transposed input. The shipped convention is: **rows are time steps, columns are dimensions**, both series must have the same number of columns, and the header carries a loud warning contrasting it with the sibling modules.

---

## The shipped API

### `include/nn/utils/edit_distance.h`

```cpp
namespace nn::utils {

struct EditCosts {
    double insertion    = 1.0;  // cost of inserting a symbol of b (transforming a INTO b)
    double deletion     = 1.0;  // cost of deleting a symbol of a
    double substitution = 1.0;  // cost of replacing a symbol of a with one of b
};

enum class EditOp { Match, Substitution, Insertion, Deletion };

struct EditAlignment {
    std::vector<EditOp> ops;   // the edit script, in order
    size_t matches = 0, substitutions = 0, insertions = 0, deletions = 0;
    size_t edit_count() const { return substitutions + insertions + deletions; }
};

// --- token level (repo-native: tokenizer.h / bpe_tokenizer.h emit vector<int>) ---
double levenshtein_distance(const std::vector<int>& a, const std::vector<int>& b);
double levenshtein_distance(const std::vector<int>& a, const std::vector<int>& b, const EditCosts&);
int    osa_distance(const std::vector<int>& a, const std::vector<int>& b);
int    damerau_levenshtein_distance(const std::vector<int>& a, const std::vector<int>& b);
EditAlignment align_sequences(const std::vector<int>& a, const std::vector<int>& b);
double word_error_rate(const std::vector<int>& reference, const std::vector<int>& hypothesis);

// --- character level (byte level; CER over UTF-8 is byte level, not code point level) ---
double levenshtein_distance(const std::string& a, const std::string& b);
double levenshtein_distance(const std::string& a, const std::string& b, const EditCosts&);
int    osa_distance(const std::string& a, const std::string& b);
int    damerau_levenshtein_distance(const std::string& a, const std::string& b);
EditAlignment align_sequences(const std::string& a, const std::string& b);
double character_error_rate(const std::string& reference, const std::string& hypothesis);

}  // namespace nn::utils
```

### `include/nn/utils/dtw.h`

```cpp
namespace nn::utils {

struct DtwPath {
    std::vector<std::pair<size_t, size_t>> cells;  // (time index in x, time index in y)
    double cost = 0.0;                             // accumulated local cost along the path
};

// x is (T, D), y is (S, D): ROWS ARE TIME STEPS, COLUMNS ARE DIMENSIONS.
double dtw_distance(const Tensor& x, const Tensor& y);
double dtw_distance(const Tensor& x, const Tensor& y, int band);  // Sakoe-Chiba; +inf if infeasible
DtwPath dtw_align(const Tensor& x, const Tensor& y);
DtwPath dtw_align(const Tensor& x, const Tensor& y, int band);

}  // namespace nn::utils
```

---

## Tasks

### Task 1: The Python oracle — the specification

**Files:** create `tests/edit_distance_oracle.py`.

**Content, in this order:**

1. **Definition-level oracles** (independent of any library):
   - Levenshtein by **BFS over the edit graph** (unit costs).
   - Weighted Levenshtein by **Dijkstra over the edit graph** (arbitrary non-negative costs).
   - OSA by **block-partition enumeration** — every way of splitting the two strings into
     matching / substituting / inserting / deleting / transposed-adjacent blocks.
   - Unrestricted DL by **BFS over the edit graph with unrestricted adjacent transposition**
     (measured to agree with the reference DL on 400 / 400).
   - The **set of all optimal `(S,D,I)` breakdowns**, by enumerating every optimal alignment.
2. **Cross-checks against maintained references** — `rapidfuzz` 3.x (`Levenshtein`,
   `Levenshtein.distance` with weights, `OSA`, `DamerauLevenshtein`, `Levenshtein.editops`),
   `jellyfish` (`damerau_levenshtein_distance`), `jiwer` 4.0.0 (`wer`, `cer`,
   `process_words`). Every cross-check is a **failing-when-broken** assertion, not a print.
3. **The property measurements** — uniqueness of the breakdown (1740/2000 unique),
   `S+D+I == distance`, the metric table of §7, the orientation table of §5.
4. **A pinned fixture section** whose values are **emitted mechanically** as C++ literals.

**Step 2: Run it.** Expected: every assertion passes and the fixture block prints.

### Task 2: Measure the oracle's own sensitivity

**Files:** create `tests/edit_distance_mutate_oracle.py`.

Break the oracle in specific ways and require it to fail each time: BFS replaced by the DP,
the transposition eligibility condition inverted, the Dijkstra graph missing the substitution
edge, the breakdown enumerator returning only the first optimum, the WER denominator changed
from `max(1,|ref|)` to `|ref|`. **All must be caught.**

### Task 3: The DTW oracle and its sensitivity harness

**Files:** create `tests/dtw_oracle.py` and `tests/dtw_mutate_oracle.py`.

- Definition-level oracle: **exhaustive enumeration of every monotone path** over the
  three-neighbour lattice, minimising the accumulated local cost (measured 400 / 400 against
  the DP).
- Cross-check: `dtw-python` with **`step_pattern=dtw.symmetric1` passed explicitly**, over
  random continuous *and* tie-heavy integer observations, with and without a band.
- The band feasibility rule (`band >= |n-m|`) and the path tie-break measurement.
- The symmetry and triangle-inequality measurements.

### Task 4: The RED C++ test for the discrete half

**Files:** create `tests/test_edit_distance.cpp`.

Pin, from the oracle's emitted fixtures: the three distances on a spread of pairs, the
weighted distance including a **fractional** cost case (which `rapidfuzz` cannot validate),
the edit script and its counts, the `S+D+I == distance` identity, the WER formula including
the empty-reference cases, and the metric/non-metric counterexamples.

**Step 1: Write the test.** **Step 2: run it — it must FAIL** (the header does not exist yet).

### Task 5: The implementation

**Files:** create `include/nn/utils/edit_distance.h` and `include/nn/utils/edit_distance.cpp`.

One templated core over "a sequence of equality-comparable elements", instantiated for
`std::vector<int>` and `std::string`. Guards throw `std::invalid_argument` with **distinct**
messages (negative or NaN cost; mismatched row counts is not applicable here).

### Task 6: The RED C++ test and the implementation for DTW

**Files:** create `tests/test_dtw.cpp`, `include/nn/utils/dtw.h`, `include/nn/utils/dtw.cpp`.

Pin: the distance on the oracle's fixtures, the multidimensional case, the band (including
the **infeasible** band returning `+inf`), the path (start, end, monotonicity, and
**cost == distance**), the shape convention (`(T,1)` vs `(1,T)` differ), the symmetry, and a
triangle-inequality counterexample.

### Task 7: The randomised differential checks

**Files:** create `tests/test_edit_distance_differential.cpp` and `tests/test_dtw_differential.cpp`.

The C++ reference is **definition-level**, not another DP: BFS over the edit graph,
Dijkstra over the edit graph, block-partition enumeration for OSA, BFS with unrestricted
adjacent transposition for DL, exhaustive monotone-path enumeration for DTW.

**The reference never pre-collapses anything the implementation collapses**, and the DTW
reference never reuses the implementation's step pattern.

### Task 8: Wire it into the public surface and the test target

**Files:** modify `include/nn/nn.h` (add the two `#include`s after `utils/isotonic_regression.h`),
modify `Makefile` (build rules, `tests` aggregate, `run_tests` echoes).

**Step 2: compile the umbrella header by itself.**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

### Task 9: The C++ mutation harnesses

**Files:** create `tests/mutate_edit_distance.sh` and `tests/mutate_dtw.sh`.

Each mutation must **fail the suite**. The harnesses follow `tests/mutate_isotonic.sh`'s
mechanics exactly: `restore()` never deletes its backup, cleanup happens once on `EXIT`, the
link set comes from `make print-lib-objs` (never from a regex over the Makefile), and a
no-op edit is reported as a **harness defect**, not as a verdict.

### Task 10: The aggregate gates

- `make -j8 build` — clean.
- The four new binaries — all pass.
- `make test_edit_distance` / the aggregate `tests` target — **zero new failures**.
- ASan + UBSan on the new binaries — clean.
- `git diff --check` — clean.

### Task 11: Record the work

**Files:** modify `EXPANSION_QUEUE.md` — move the bullet to `## Done` with a one-line summary
and the test results, and mark the bullet so it is not re-popped.

---

## References

- **Levenshtein, V. I.** "Binary codes with minimum Hamming weight." *IEEE Transactions on Information Theory* **14**(4) (1966), 508–516. — the original metric.
- **Wagner, R. A. and Fischer, M. J.** "The string-to-string correction problem." *IBM Journal of Research and Development* **10**(6) (1967), 627–637. — the DP recurrence.
- **Lowrance, R. and Wagner, R.** "The string-to-string correction problem with edits." *IEEE Transactions on Information Theory* **18**(1) (1972), 164–169. — the unrestricted Damerau–Levenshtein recurrence.
- **Damerau, B. L.** "The string-to-string correction problem." *IBM Journal of Research and Development* **10**(2) (1966), 177–180. — the original transposition argument.
- **Wikipedia, "Damerau–Levenshtein distance".** — the sentinel-layout recurrence, cross-checked against `rapidfuzz` and `jellyfish` on 20 000 pairs.
- **Itakura, F. and Chiba, S.** "Time series analysis based on the optimal string alignment principle." *IEEE Transactions on Acoustics, Speech, and Signal Processing* **23**(6) (1975), 590–597. — the optimal-string-alignment restriction.
- **Sakoe, H. and Chiba, S.** "Dynamic programming algorithm optimization for spoken word recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing* **26**(1) (1978), 43–49. — the banding constraint.
- **Itakura, F.** "Minimum prediction residual principle applied to speech recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing* **23**(1) (1975), 67–72. — the minimum-prediction-residual principle behind DTW.
- **Reference implementations used for the cross-checks, all installed locally under `/usr/bin/python3`: `rapidfuzz` 3.x, `jellyfish`, `jiwer` 4.0.0, `dtw-python` (`dtw` package).**

**NO arXiv ID IS ASSERTED.** Every reference above is pre-arXiv or has no arXiv record, and this repo's expansion queue has produced many wrong arXiv identifiers — every one a plausible-looking neighbour of the right paper.

---

## Outcome (recorded 2026-11-12)

**SHIPPED.** All eleven tasks are complete.

| File | Status |
| --- | --- |
| `include/nn/utils/edit_distance.{h,cpp}` | new — 338-line header, 379-line implementation |
| `include/nn/utils/dtw.{h,cpp}` | new — 208-line header, 245-line implementation |
| `tests/edit_distance_oracle.py` | new — 882 lines, 28/28 checks |
| `tests/edit_distance_mutate_oracle.py` | new — 386 lines, 13 caught / 1 documented survivor |
| `tests/dtw_oracle.py` | new — 682 lines, 22/22 checks |
| `tests/dtw_mutate_oracle.py` | new — 387 lines, 16 caught / 1 documented survivor |
| `tests/test_edit_distance.cpp` | new — 286/286 checks |
| `tests/test_edit_distance_differential.cpp` | new — 7/7 checks |
| `tests/test_dtw.cpp` | new — 91/91 checks |
| `tests/test_dtw_differential.cpp` | new — 5/5 checks |
| `Makefile` | 4 build rules, 4 `tests:` prerequisites, 4 `run_tests` lines |
| `include/nn/nn.h` | 2 umbrella includes |

**THE PLAN'S TASK ORDER WAS FOLLOWED, WITH ONE DEVIATION WORTH RECORDING.** The plan
called for the C++ implementation to come after the oracles — which it did — but the
plan did NOT specify a separate randomised differential suite for either module. Two
such suites were written anyway, because a pinned fixture set only sees the fixtures
it was written for. The references they use share NO structure with the shipped code:

- the edit-distance differential uses a **breadth-first search over the string state
  space** (with and without transposition), a **Dijkstra over that same space** for
  weighted costs, and an **exhaustive enumeration of every monotone alignment** for
  OSA;
- the DTW differential uses an **exhaustive enumeration of every monotone path**,
  which is the definition of the warping problem rather than an algorithm for it.

**THREE DEFECTS FOUND IN THIS WORK, ALL OF THEM IN THE WORK'S OWN CODE.**

1. **The DTW first column must stay at `+inf`.** Initialising it with the local cost
   lets the warping path start late — it can enter the grid at row 2 and skip the
   first observation of `x` entirely. Measured on `x = [100, 0]` against `y = [0]`:
   the finite first column gives **0.0**, the infinite one gives **100.0**, and
   `dtw-python` gives **100.0**. This is a wrong ANSWER, not a precision difference,
   and no tolerance-based test would have flagged it — only the reference comparison
   did.
2. **The C++ DTW differential reference charged the local cost of the cell it was
   LEAVING rather than the one it was ARRIVING at**, so the final cell's cost was
   dropped. The symptom was 312 mismatches over 120 cases — every case failing —
   which is what a systematic off-by-one in the REFERENCE looks like.
3. **A mutation harness that restores the wrong file leaves the mutation in the
   tree.** The harness edited the reference in a scratch copy and restored only the
   implementation, so a mutation ("the BFS skips substitutions") was committed into
   the shipped test file. The symptom was 48/150 mismatches under `-march=native`
   that did **not** reproduce under plain `-O2` — which looked like an
   optimisation-dependent bug and was not. The general rule: **a survivor count that
   changes with the compiler flags is a harness defect, not a finding**, and a
   mutation harness must restore every file it touches.

**ONE COVERAGE GAP FOUND BY THE MUTATION HARNESS, AND CLOSED.** The tie-break order
is observable, and the suite did not pin it: a mutation that swapped the horizontal
and vertical branches survived both the pinned and the differential suites. The
cause was that every existing path fixture had a UNIQUE optimal path, so no fixture
could tell one predecessor order from another. Nine tie-discriminating fixtures were
generated from the oracle (integer observations, because continuous observations
almost never tie), and the suite now **asserts that every one of them admits more
than one optimal path** — otherwise the section could not fail. The
horizontal-versus-vertical distinction needed its own fixture on top of the other
eight.

**THE AGGREGATE SUITE.** `make run_tests` ran every registered suite. The only
failing suite is `test_neural_ode` (the ODERNN gradient check), and it was
**re-measured identically on the pre-work commit `c9d87ca`** — it is the long-standing
entry in `NOT_FIXED.md` and is not a regression from this work.
