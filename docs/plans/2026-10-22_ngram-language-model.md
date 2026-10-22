# Statistical n-gram Language Model Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/ngram_lm.{h,cpp}` — a count-based n-gram language model over integer token ids, with three smoothing schemes (Kneser-Ney, Witten-Bell, absolute discounting with interpolation) plus Heaps' law and Good-Turing over the same count tables.

**Status: SHIPPED 2026-10-22 as `f4d56ad`.** 100/100 checks, oracle 85/85, 18/18 mutations caught with 0 survived / 0 broken, umbrella header compiles standalone, zero new full-suite failures. The record of what the oracle and the mutation harness caught is in `## Done` of `EXPANSION_QUEUE.md`; this file is kept as the design record.

**Architecture:** Model-free free functions and one class over `std::vector<int>` — no `Tensor`, no `Model`, no optimizer, so the whole file is testable with a corpus of integer ids. The count tables are `std::map`s keyed by token id and token pairs; the three smoothing schemes share the same tables and differ only in how they turn counts into probabilities.

**Tech Stack:** C++17, the repo Makefile, numpy/python oracle in `tests/`.

---

## 0. Verified facts this plan rests on (measured, not assumed)

**Absence.** `grep -rinE 'ngram|n-gram|backoff|jelinek|stupid.?backoff|good.?turing|heaps|kenlm|kneser|katz|bigram|trigram|unigram' include/ tests/` returns **0 content hits** — with MATCHING LINES printed, not filenames. The queue's own standing rule is that a `grep -l` hit list is not evidence of presence and a miss is not evidence of absence unless the pattern was checked against something that does contain the term.

**The idea was on the queue's own VERIFIED-ABSENT list.** `EXPANSION_QUEUE.md`'s 2026-10-20 entry carries:

```
VERIFIED ABSENT, re-measure before taking:
  - `grep -rinE 'ngram|kenlm|good.?turing|heaps.?law|subword.?reg' include/`
    -> check before taking.
```

It was re-measured on 2026-10-22 with 0 hits before this file was written.

**Not redundant.** Measured, not argued:

| existing | what it does | what this adds |
|---|---|---|
| `utils/tokenizer.{h,cpp}` | text <-> word ids | no probabilities |
| `utils/bpe_tokenizer.{h,cpp}` | text <-> byte-level BPE ids | no probabilities |
| `utils/logit_processing.{h,cpp}` | reshapes a NEURAL NETWORK's logits into a distribution | needs a network's output |
| `utils/losses.h` | cross-entropy against a NEURAL NETWORK's logits | needs a network's output |
| `layers/generative/pixelcnn.h` | a neural autoregressive model | is a model |

Nothing in the repo produced a STATISTICAL language model, and nothing produced a **perplexity** for a token sequence.

**Citations — all three are PRE-arXiv, so NO arXiv ID is asserted.** This repo's expansion queue has produced **twelve** wrong arXiv IDs, each a plausible-looking neighbour of the right paper, so a bare "arXiv:XXXX.XXXXX" here would have been the thirteenth.

- Kneser, R. and Ney, H. (1995). "Three methods for the construction of language models for large scale text retrieval." Eurospeech 1995, 125-135.
- Witten, I. H. and Bell, T. C. (1991). "The statistical basis of predictive word association." IEE Proceedings-B, 138(4), 375-380.
- Chen, S. F. and Goodman, J. (1999). "An empirical study of smoothing techniques for language modeling." COLING-99, 291-304.

The interpolated forms were cross-checked against the Stanford NLP smoothing tutorial and against **NLTK's `nltk/probability.py`** (a maintained reference implementation), installed and run as part of the oracle.

## 1. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | The LM works on **integer token ids**, not words | this repo's own tokenizers emit `std::vector<int>`; ids compose with them directly |
| D2 | **INTERPOLATED** Kneser-Ney, not the non-interpolated variant | the interpolated form is the one in the Stanford tutorial and the one Chen & Goodman benchmark; the two differ measurably (0.2418 on the fixture's p(the-quick-brown)) and the difference is recorded, not hidden |
| D3 | The KN unigram is the **CONTINUATION** count `N1+(w)` | the defining property of the variant; a word that never follows another word has probability 0 |
| D4 | The backoff weight uses **V_h** (distinct followers), not `c(h)` | `c(the) = 6` but `V_the = 4` on the fixture; the two give visibly different numbers |
| D5 | The discount is **absolute** and the term is **clamped at zero** | `max(c - D, 0)`; without the clamp an unseen n-gram gets a negative term |
| D6 | `0 < D <= 1` is enforced by the constructor | the KN normalization proof needs `c(h,w) >= D` for every follower |
| D7 | The Witten-Bell reserved mass is renormalized by `Z_h` | without it the model is not normalized; the reserved mass must be spread over the tokens unseen at h, not over the whole lower-order support |
| D8 | The OOV class is **one fixed sentinel**, not a per-id class | every unknown id maps to it, so the distribution is over (vocabulary + ONE OOV class) and sums to one |
| D9 | The **growth curve** is recorded during training, a point per token | Heaps' law is about `V(n)` for EVERY n including the plateaus; the count tables do not record first-appearance order, so the curve cannot be reconstructed afterwards |
| D10 | The base class is `std::map`, not `unordered_map` | deterministic iteration order, so the results are reproducible run to run |

## 2. The oracle caught six defects in ITSELF before any C++ existed

`tests/ngram_oracle.py` (85 self-checks) is the specification, and it caught:

1. **A key-shape bug that made the normalization checks VACUOUS.** The context was looked up as `CTX2[(h,)]` — a double-wrapped tuple — so every context looked UNSEEN and every distribution degenerated to the unigram, which also sums to 1. The fix: typed accessors that state the key shape once. **Every normalization case now asserts the context is SEEN first**, because a normalization check over a degenerate distribution measures nothing.
2. **An int key into a tuple-keyed counter**, so `UNIGRAM[W2I['the']]` was always 0 and the "continuation differs from raw" check passed for the wrong reason (it compared 0.1579 against 0.0).
3. **A hand-arithmetic slip.** The absolute-discounting closed form used `c(the)/N = 6/25` where it needed `c(quick)/N = 2/25`. The implementation was right; the hand-written literal was wrong. This is the repo's recurring trap, and the closed form is now spelled out rather than pasted.
4. **An unreachable threshold.** `R^2 > 0.99` on a 25-token corpus was not merely unmet but **unreachable**: the best possible power-law fit to the corpus curve scores 0.9888. A threshold no correct implementation can meet is a test bug. The threshold is now 0.95, and the property that actually proves the fit machinery is that a **synthetic** power law fits with `R^2 == 1` exactly.
5. **The OOV class was missing from the unseen set**, so the Witten-Bell distribution lost the OOV share of the reserved mass and the normalization check asserted a false equality.
6. **A wrong NLTK denominator.** NLTK's non-interpolated value is `(c3 - D) / c(bigram context) = (2 - 0.75)/2 = 0.625`, not `/c(the) = 6`. The oracle now asserts the measured value and asserts that the wrong denominator is NOT the right one.

## 3. Three real C++ bugs, found by the harness and the suite

1. **THE BACKOFF TOOK THE PREFIX OF THE CONTEXT INSTEAD OF THE SUFFIX.** `h = (h1, h2)` must back off to `h' = (h2)` — a language model's context is a SUFFIX of the history. The first implementation took `(h1)`, which for the fixture's context `(the,cat)` evaluates `cat` where the spec says `the`. Caught by the pinned values 4j and 11a/11d.
2. **THE GROWTH CURVE RECORDED ONLY THE NEW TYPES** (15 points) instead of a point per token (25 points). The curve then had no plateaus and the Heaps fit was a different object. Caught by the pinned Heaps values 13a-13c.
3. **A CONSTANT-FACTOR / OFF-BY-ONE STYLE GAP IN THE SUITE**: the "retraining replaces, not accumulates" test checked `num_bigrams()`, which is the number of **distinct keys** — and accumulating into the same map does not change the number of distinct keys. The mutation "retraining accumulates" **SURVIVED**. The fix: assert the **counts** (`bigram_count`, `unigram_count`, `trigram_count`), which is what doubles. *A test that only checks the shape of a table cannot see a table whose values grew.*

## 4. The mutation harness

`tools/ngram_mutation.sh` — **18/18 caught, 0 survived, 0 broken.** It carries the repo's accumulated defences: the link set comes from `make print-lib-objs` (never a regex over the Makefile, whose `LIB_OBJS` is a recursive variable), the mutated object is rebuilt in place (never compiled separately and double-linked), each edit is checksum-verified, the restore never deletes its own backup, and a build failure is reported as a BUILD verdict rather than a mutation one.

**One harness defect, of the class that produces false verdicts:** the first M13 wrote `if (false) log_k = 0.0;`, but `log_k` is `const`, so the build failed and the harness reported MUT-BROKEN. A build error says nothing about coverage. The mutation was rewritten to change the fitted intercept instead.

**The mutations**, each aimed at one defining semantic:

| id | what it breaks | caught by |
|---|---|---|
| M01 | the KN unigram uses the raw count | 18 checks |
| M02 | the backoff weight uses `c(h)` instead of `V_h` | 14 |
| M03 | the discount term is not clamped at zero | 7 |
| M04 | the discount is relative, not absolute | 9 |
| M05 | the backoff takes the prefix, not the suffix | 4 |
| M06 | the WB reserved mass is not renormalized by `Z_h` | 4 |
| M07 | the WB seen term drops `T_h` | 6 |
| M08 | the AD interpolation weight is inverted | 2 |
| M09 | the AD unigram uses the continuation count | 6 |
| M10 | the WB unigram reserves no OOV mass | 5 |
| M11 | the OOV class is dropped from the unseen set | 4 |
| M12 | the growth curve records only the new types | 3 |
| M13 | the Heaps fit drops the `beta*mx` intercept term | 3 |
| M13b | the fitted K is doubled | 2 |
| M14 | the Good-Turing count loses the `(r+1)` factor | 1 |
| M15 | the Good-Turing unseen mass is inverted | 1 |
| M16 | the discount bound is not enforced | 3 |
| M17 | retraining accumulates | 3 |

## 5. What is deliberately NOT here

- **Jelinek-Mercer** and **Katz** smoothing: two more variants of the same family, and the header names them as the ones NOT implemented rather than leaving a reader to wonder. Adding them is a small, mechanical extension of the same three helpers.
- **Stupid backoff** and **Witten-Bell interpolation** (the interpolating variant of WB): same.
- **A `Layer` wrapper**: this is a statistical model, not a differentiable one. It has no gradients, and pretending otherwise would be wrong.
- **Perplexity over a neural network's logits**: `utils/losses.h` already does cross-entropy against logits; a perplexity over a neural LM is a different function of a different object.
- **An arXiv ID for any of the three papers**: all three are pre-arXiv.

## 6. Verification performed

```
make build/test_ngram_lm          # 100/100
./build/test_ngram_lm              # 100/100, three runs, identical
/home/stefan/.venv/bin/python tests/ngram_oracle.py   # 85/85
printf '#include "nn/nn.h"\n' | g++ -std=c++17 -Iinclude -x c++ -fsyntax-only -
make tests                         # aggregate compile
make run_tests                     # aggregate runtime
./tools/ngram_mutation.sh          # 18/18 caught, 0 survived
```
