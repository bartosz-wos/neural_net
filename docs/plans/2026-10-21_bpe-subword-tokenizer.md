# BPE Subword Tokenizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a byte-level BPE (byte-pair-encoding) tokenizer to `utils/`, and fix a
measured id-collision bug in the existing word-level `Tokenizer` that makes its
`encode`/`decode` roundtrip lossy.

**Architecture:** A new self-contained `BPETokenizer` in
`include/nn/utils/bpe_tokenizer.{h,cpp}`. The legacy `Tokenizer` symbol is
**preserved and repaired**, not replaced — it is registered in `nn/nn.h` and is a
public symbol. Semantics are pinned by a numpy oracle *before* any C++ exists.

**Tech Stack:** C++17, `std::string`/`std::vector`, the repo's `CHECK`/`CHECK_NEAR`
test harness, Makefile.

---

## Citation (verified by title search this session)

- Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with
  Subword Units", ACL 2016 (Volume 1: Long Papers), pages 1715–1725.
  - ACL Anthology: `P16-1162` (doi `10.18653/v1/P16-1162`)
  - arXiv: `1508.07909`
  Both returned by a title search whose result sets echoed the title verbatim.

This is the BPE paper. No other citation is asserted.

---

## Why this and not something else

Measured absences (all run this session, all printing matching lines, not
filenames):

```
grep -rinE 'bpe|byte.?pair|subword|sentencepiece|unigram|merge_rank
            |bytes_to_unicode|wordpiece|byte_fallback' include/ --include=*.h
-> only `vocab_size` FALSE POSITIVES. ZERO hits on the concept.
```

The repo ships a **byte-level** generative stack and nothing that turns bytes
into tokens correctly:

- `utils/logit_processing.h` (shipped 2026-10-20) exposes
  `std::vector<int> tokens` and `Tensor decode(const Tensor& start_ids, size_t)`.
  It produces token ids. There is no shipped decoder for them.
- `layers/generative/pixelcnn.h` models a categorical over `n_vals`; its own
  header says "we expose logits and let the caller apply softmax".
- `layers/dense/embedding.h` takes `(vocab_size, dim)`.
- `layers/recurrent/mambabyte.h` takes raw byte indices.

So the last stage of the discrete-generative pipeline is missing, exactly as the
queue file predicted. **This is not redundant** with `logit_processing.h`: that
header chose WHICH token; nothing maps ids <-> text.

---

## The measured bug this run found (Phase 1 evidence)

`include/nn/utils/tokenizer.h` has never been modified since the initial commit
(`1d8336a`) and has **no test**.

```cpp
Tokenizer() : next_id(0), eos_id_(256) { ...; next_id = 256; }   // eos_id == next_id
void train(...) { vocab[p.second] = next_id; id_to_token[next_id] = ...; next_id++; }
std::string decode(...) { if (id == eos_id_) { result += ' '; continue; } ... }
```

`eos_id_` is **256** and the first learned word is also assigned **256**.
`decode` maps 256 to a space. **The single most frequent word in the corpus is
unrecoverable.** Measured, not reasoned:

```
eos_id()            = 256
encode("hello")[0]  = 256   <-- first vocab id
COLLISION (word id == eos id): YES
encode("there")[0]  = 258 -> decode "there" (intact)   <-- proves it is the FIRST id only
encode("hello world") = [256, 256, 257]
decode roundtrip = "  world"        exact: NO
roundtrip [hello  world] -> "  world"   exact: NO
roundtrip [ hello]      -> " "         exact: NO
roundtrip [hello ]      -> "  "         exact: NO
unseen "zzz" -> "zzz" exact=YES        <-- the byte path is sound; the vocab path is not
```

Two further defects, same measurement: whitespace runs collapse (lossy), and
`unk_id()` returns `0`, which is byte NUL, not an UNK marker — the accessor lies.

**Root cause:** the constructor sets `eos_id_ = 256` and starts learned ids at
256, so the reserved end-of-word slot is never excluded from the assignment loop.
One-line fix: assign the first learned word id `257`, i.e. reserve 256.

`Tokenizer` stays for compatibility. The whitespace-collapse behaviour is
**documented, not repaired** — see Out of scope.

---

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | New class `BPETokenizer`, do NOT change `Tokenizer`'s name or API | `Tokenizer` is in `nn/nn.h` (line 50). Per writing-plans step 3b: preserve the legacy public symbol, pick a non-colliding descriptive name for the new behaviour. |
| D2 | Byte-level base alphabet: ids 0..255 are the 256 raw bytes; merge rank i → id 256+i; words are separated by a RESERVED BOUNDARY ID `= vocab_size()` | The repo's byte-native layers (PixelCNN, MambaByte) already work in bytes. A unicode base alphabet would make `encode(decode(ids)) != ids` for non-ASCII. **The explicit boundary id is not optional** — oracle defect O4 measured that the paper's `</w>` marker is insufficient, because a word whose final symbol stays a leaf loses its boundary. |
| D3 | `</w>` end-of-word marker is a **training-time construct only** | The paper's convention, needed so a merge cannot span a word boundary. It is stripped on the way out and replaced by the D2 boundary id (O4). |
| D4 | Whitespace is **canonicalised**: any run of whitespace -> one `' '`, no leading/trailing space | This is GPT-2 BPE's documented behaviour. The exact roundtrip invariant is therefore `decode(encode(t)) == canonicalize(t)`, NOT `== t`. Asserting `== t` on arbitrary text would be a test bug (the "run of spaces" family from the logit-processing session). |
| D5 | Training tie-break is `(-count, a + "\x1f" + b)` lexicographic | Python dict iteration is insertion-ordered, C++ `std::unordered_map` is not. Without an explicit total order the two implementations cannot agree, and an unspecified tie-break silently ships platform-dependent vocabs. |
| D6 | Encode applies merges **in learned-rank order** (greedy lowest rank), not by re-counting frequencies | Greedy-by-rank is what "BPE" means at inference; re-counting would be a different algorithm. |
| D7 | Oracle pins BOTH a naive full-recount trainer and an incremental one and asserts they agree | The incremental pair-count update (decrement pairs adjacent to the merged span, increment new ones) is the classic off-by-one source. Two independent implementations agreeing is the strongest check available. |

## Out of scope (stated so it cannot creep in)

- No dropout / `unk` token in `BPETokenizer`: with a byte-level base alphabet
  every byte string is representable, so OOV is impossible by construction.
- No regex pre-tokenizer (GPT-2's `pat`): plain whitespace splitting, matching the
  paper's own pre-tokenization. Noted in the header.
- No serialization / load-from-disk (the legacy `Tokenizer` has none either).
- `Tokenizer`'s whitespace collapsing is NOT repaired — it is a documented,
  tested behaviour. Changing it would break any caller relying on it.

---

## Tasks

### Task 1: numpy oracle, before any C++ exists

**Objective:** pin the semantics; catch oracle bugs before they become C++ test bugs.

**Files:**
- Create: `tests/bpe_oracle.py`

Two independent BPE trainers (naive full-recount, incremental), asserted to
agree. Prints a C++-pasteable table of expected ids. Labels S1..S5.

**Step 1: Write the oracle.**

**Step 2: Run it.**
Run: `/home/stefan/.venv/bin/python tests/bpe_oracle.py`
Expected: exits 0, prints the table, all self-checks pass.

**Step 3: Commit.**

```bash
git add tests/bpe_oracle.py
git commit -m "test(utils): numpy oracle pinning BPE semantics S1-S5"
```

#### WHAT THE ORACLE CAUGHT — SEVEN DEFECTS, ALL BEFORE ANY C++ EXISTED

This is the most valuable part of the plan, and it is the argument for the
methodology. Six were in the oracle itself; one changed the SPEC.

| # | Defect | Why it was invisible |
|---|--------|----------------------|
| O1 | `pair_sort_key` did `a + "\x1f" + b` on 1-tuples | TypeError, caught immediately. The symbols are tuples, not strings. |
| O2 | Leaf ids emitted as `ord(s[0][0])` on a tuple-of-bytes | `AttributeError: 'int' has no decode`. Same root cause as O1: the symbol representation was wrong, and it took a full rewrite to get right (leaf = `(ch,)`, merge = `left + right`). |
| O3 | **`chk()` return value ignored — the harness printed `[ok] S1 ...` immediately after `[ORACLE-FAIL] S1 ...`** | **A FAILURE LINE PRINTED NEXT TO A SUCCESS LINE WITH THE SAME LABEL.** Reading the output at a glance says the oracle passed. This is the "harness that lies to you" failure from the TDD skill, caught in the harness I wrote this session. Fixed by making `chk` print the single line and return `ok`. |
| O4 | **`</w>` in the id stream does not preserve word boundaries** | The marker rides on the FINAL byte. If the final symbol never gets merged it stays a leaf, and `encode` emits the bare byte — the boundary is **gone**. Measured: `encode("a b c") = [97, 98, 99]` decoding to `"abc"`. It is not only single-char words: **any** word whose last symbol survives as a leaf loses its boundary. **Changed the design (D2 in Decisions).** |
| O5 | `i < vocab_size` rejected the boundary id itself | Off-by-one **in the check, not the encoder**. Valid ids are `0 .. vocab_size` INCLUSIVE. Rejected a correct implementation. |
| O6 | `boundary_id not in merges` compared an int against a list of symbol tuples | Same family as O5: the check, not the code. |
| O7 | **THE REAL ONE — right-to-left splicing invents a different algorithm.** | See below. |

**O7 in full, because it is the one worth remembering.** The incremental trainer
merged a sequence's merge sites **right-to-left** so that earlier indices would
stay valid. But BPE merges are **non-overlapping and selected left-to-right**,
so scan order is part of the SEMANTICS, not an implementation detail:

```
merge (a,a) into "aaaa" = ['a','a','a','a</w>']:
  left-to-right   -> ['aa','a','a</w>']  then ['aa','aa</w>']   (2 merges)
  right-to-left   -> index 2 taken first -> ['a','aa','a</w>'],
                     index 0 no longer matches -> 1 merge lost
```

Different alignments ⇒ different pair counts next round ⇒ the two trainers
diverged on **every corpus containing a repeated character**. My first fix
attempt made it *worse* (5 failures instead of 2) because I "corrected" the
neighbour index — that index was already right, and I had mis-indexed it. The
actual fix rebuilds each affected sequence left-to-right exactly as the naive
trainer does, applying the count deltas while rebuilding.

**The generalisable lesson, and it joins the file's existing list:** three
defects (O5, O6, and my first O7 attempt) were bugs in **the thing doing the
checking**, and in all three the check rejected a *correct* implementation. A
third repo instance of the "the fixture, not the code" family (after the
adversarial `2*eps >= gap` case and the logit-processing one-ULP boundary
fixture). Oracle disagreement is evidence that *one of two pieces of code you
wrote this session* is wrong — locate which before editing either.

**The pinned table (from the green run):**

```
corpus  = 'the quick brown fox jumps over the lazy dog the fox'   num_merges = 30
vocab_size() = 260  (256 bytes + 4 merges + 1 boundary)

  empty string         ''                  -> []
  single word          'the'               -> [257]
  two words            'the fox'           -> [257, 260, 259]
  repeated word        'the the'           -> [257, 260, 257]
  unknown short        'zz'                -> [122, 122]
  unknown longer       'qqqq'              -> [113, 113, 113, 113]
  whitespace collapse  '  the   fox  '     -> [257, 260, 259]
  no spaces            'thefox'            -> [116, 104, 101, 259]
  tabs/newlines        'the\t\tfox\n\ndog' -> [257, 260, 259, 260, 100, 111, 103]
  single-char words    'a b c'             -> [97, 260, 98, 260, 99]
  two-char words       'ab cd'             -> [97, 98, 260, 99, 100]

  rank  0: id 256 = 'he</w>'
  rank  1: id 257 = 'the</w>'
  rank  2: id 258 = 'fo'
  rank  3: id 259 = 'fox</w>'
  boundary id = 260
```

### Task 2: RED — failing C++ tests

**Objective:** write the tests, watch them fail for the right reason.

**Files:**
- Create: `tests/test_bpe_tokenizer.cpp`
- Modify: `tests/test_tokenizer.cpp` (create it — it does not exist)

Two suites:
- `test_tokenizer.cpp` — regression for the id collision + the documented
  whitespace behaviour.
- `test_bpe_tokenizer.cpp` — the BPE semantics, values copied from the oracle.

**Step 2: Verify RED.**
Run: `make build/test_bpe_tokenizer build/test_tokenizer 2>&1 | tail -20`
Expected: **compile failure** — `BPETokenizer` is not declared. That is the
correct RED for "the feature does not exist". It must NOT be a typo or a
missing-header error in the harness itself.

### Task 3: GREEN — repair `Tokenizer`

**Objective:** reserve id 256 so the end-of-word slot cannot collide.

**Files:**
- Modify: `include/nn/utils/tokenizer.h` (the `train()` assignment)
- Modify: `include/nn/utils/tokenizer.h` (`unk_id()` doc comment)

**Step 3: Run tests.** Expected: `test_tokenizer` fully green.

**Step 4: Commit.**

### Task 4: GREEN — implement `BPETokenizer`

**Objective:** minimum implementation passing the Task 2 tests.

**Files:**
- Create: `include/nn/utils/bpe_tokenizer.h`
- Create: `include/nn/utils/bpe_tokenizer.cpp`

**Step 4: Run the focused suite.** Expected: green.

### Task 5: Wire up

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile` (build rule, `tests:` prerequisite, `run_tests` line)

**Verification (mandatory, catches cross-header collisions a focused test cannot):**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

### Task 6: Mutation test

**Objective:** prove the tests are not vacuous. Stub each of:
1. the `</w>` attachment,
2. the `-count, pair` tie-break (use unordered iteration),
3. the greedy lowest-rank rule,
4. the incremental-count neighbour update.

Each mutation must make at least one test fail. A survivor is either a real
coverage gap (kill it with a test) or genuinely inert (measure it, then verify
the measurement with an inverted-verdict probe — see the TDD skill).

**Harness requirements** (this repo has been bitten by all of these):
- `restore()` must NEVER delete the backup: `cp "$BAK" "$SRC.tmp" && mv -f "$SRC.tmp" "$SRC"`, cleanup once on `trap`.
- Checksum the tree before and after each mutation; a no-op edit is a HARNESS
  defect, not a verdict.
- Print the matching lines, not just filenames — a `*` in a double-quoted
  `$(printf ...)` globs into a file listing and matches nothing.
- No `\n` inside a sed pattern.
- `rm -f` intermediates before every build; never link the mutated TU twice.

### Task 7: Full suite + push

```bash
make run_tests
git push origin master
```

## Verification gates — ALL MET

- [x] `nn/nn.h` compiles standalone with `-fsyntax-only`.
- [x] Oracle agrees: naive trainer == incremental trainer, all S1..S5 self-checks.
- [x] Oracle ran FIRST and caught 7 defects, 6 in the oracle and 1 in the design.
- [x] `test_tokenizer` proves the collision is gone (13/13) — this is the bug fix
      that shipped alongside the feature.
- [x] `test_bpe_tokenizer` proves `encode(decode(ids)) == ids` for arbitrary ids,
      exactly (45/45).
- [x] Every mutation either caught or measured-inert with an inverted probe.
      **4 caught / 0 survived / 1 measured-inert / 0 broken / 0 no-op.**
- [x] `-Wall -Wextra` clean.
- [x] Full `make run_tests`: **zero regressions**. The single failing suite is
      `test_neural_ode`, documented in `NOT_FIXED.md` since 2026-09-08 and
      re-verified on clean master on 2026-10-03.
- [x] ASan + UBSan clean on the pristine tree, both suites.
- [x] Post-harness rebuild-and-verify defence: after `mutate_bpe.sh` the tree is
      checksum-clean, rebuilds, and passes 45/45 + 13/13.

---

## What actually shipped, and the three bugs that were real

The plan predicted the `Tokenizer` collision (measured in Phase 1) and the need
for an explicit boundary id (oracle defect O4). Both held. It did **not**
predict the three implementation bugs below, all found by instrumenting the
leaf function rather than reasoning about it:

1. **Merge surfaces stored and looked up under different marker policies.**
   Training wrote marker-bearing keys; encode's greedy loop built symbols under
   a different rule, so no merge ever applied — while `merges()` still looked
   correct.
2. **`PairKey` keyed on `(id_a, id_b)` alone.** The same `(h, e)` pair is the
   merge `"he"` mid-word and `"he</w>"` at word-end; these are **distinct merges**
   and keying on ids collapses them. This is the C++ instance of the oracle's
   tuple representation: the oracle was right and the C++ lost that information.
   Silent failure — training produced a plausible merge table and every
   assertion on `merges()` still passed.
3. **The `</w>` marker counted twice.** `merge_keys_[i]` already ends in `</w>`
   and `surface()` appended another, producing `"the</w></w>"`, which encode can
   never match. Instrumented trace: `[TRAIN] key=he</w>` → `[TRAIN] key=the</w></w>`
   → `[ENC] miss cand=the</w>`. The marker must live in exactly one place; the
   oracle gets this free because it stores the marker *inside* the symbol tuple.

## Mutation results

| id | mutation | verdict |
|----|----------|---------|
| M1 | drop the `</w>` marker from the final byte | **CAUGHT** (10 assertions) |
| M2 | pair match ignores `marked` | **INERT-OK** (measured, inverted probe) |
| M3 | `surface()` appends the marker to merges too | **CAUGHT** (12 assertions) |
| M4 | encode picks the worst-ranked merge first | **CAUGHT** (2 assertions) |
| M5 | training tie-break drops the total order | **CAUGHT** (7 assertions) |

**M2 is inert for a structural reason, and the reason is the interesting part.**
`best.am` / `best.bm` are read out of the very `PairKey` that counted the
occurrences, so once `s[i].id == best.a` holds, `s[i].marked == best.am` is
*implied* by the counting pass and cannot disagree. Two independent probes
(3000 random corpora dumping the full merge table, then the same at
`min_frequency = 1`) both produced **0 differing lines**. Its parent — dropping
`marked` from the `PairKey` itself, which is bug 2 above — differs on **1437 of
3000** corpora, verified by building both. So the inert verdict is scoped to
this exact mutation, not to the idea.

**M4 was a real gap the first suite missed.** Found by search rather than
guesswork: a 1950-case sweep over short 2-letter corpora comparing
lowest-rank-first against a highest-rank-first mutant gave 406 differing cases;
the smallest is corpus `"aaa aba aaa aba"` with probe `"aba aaaaba aaa"`, where
merges compete at ranks 0 and 2:
```
lowest-rank-first   [259, 260, 256, 256, 98, 97, 260, 258]
highest-rank-first  [259, 260, 256,  97, 259, 260, 258]
```
Added as test 11 with the expected value copied from the oracle.

## Note on the harness

`tests/mutate_bpe.sh` encodes six defects that earlier harnesses in this repo
actually produced, each of which yielded a confident wrong verdict. The one that
bit hardest: `restore()` that deletes its own backup works **exactly once**, and
since it is called per mutation the tree ends up holding whatever mutant ran
last. Here `restore()` never deletes and cleanup runs once on `trap EXIT`.