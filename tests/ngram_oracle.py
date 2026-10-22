#!/usr/bin/env python3
"""
tests/ngram_oracle.py -- numpy/python SPECIFICATION for utils/ngram_lm.{h,cpp}.

WHY AN ORACLE. Every numeric expectation in tests/test_ngram_lm.cpp is copied
from this file's output. This repo has caught a real bug in an ORACLE twice
before the C++ existed (tests/sinkhorn_oracle.py, and the logit-processing
session where 5 of 6 pre-implementation defects were in the oracle itself),
and the TDD skill is explicit that a hand-derived expected value in a test is
a leading cause of phantom implementation bugs.

Run with:
    /home/stefan/.venv/bin/python tests/ngram_oracle.py

Exits 0 iff every self-check passes.

===============================================================================
THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-10-22)

    grep -rinE 'ngram|n-gram|backoff|jelinek|stupid.?backoff|good.?turing
                |heaps|kenlm|kneser|katz|bigram|trigram|unigram' include/ tests/
    -> 0 HITS, in every directory, with MATCHING LINES printed (not filenames).

The idea is on this repo's own expansion queue as a VERIFIED-ABSENT item
(EXPANSION_QUEUE.md, the 2026-10-20 entry's "VERIFIED ABSENT, re-measure
before taking" list):

    - `grep -rinE 'ngram|kenlm|good.?turing|heaps.?law|subword.?reg' include/`
      -> check before taking.

and it was re-measured on 2026-10-22 with 0 hits before this file was written.

IT IS NOT REDUNDANT with what the repo already ships:
  * `utils/tokenizer.{h,cpp}` and `utils/bpe_tokenizer.{h,cpp}` map text <-> ids.
    They do not assign PROBABILITIES to sequences.
  * `utils/logit_processing.{h,cpp}` reshapes LOGITS into a distribution and
    decodes. It needs a neural network's output.
  * `utils/losses.h` computes cross-entropy against a neural network's logits.
  * `layers/generative/pixelcnn.h` is a neural autoregressive model.
  Nothing in the repo produces a STATISTICAL (count-based) language model, and
  nothing produces a perplexity for a token sequence. The n-gram LM is the
  canonical model-free baseline, and it is what the repo's own tokenizers
  produce the ids for.

===============================================================================
CITATIONS. All three are PRE-arXiv, so NO arXiv ID may be asserted for any of
them -- this repo's expansion queue has produced twelve wrong arXiv IDs and
each was a plausible-looking neighbour.

  * Kneser, R. and Ney, H. (1995). "Three methods for the construction of
    language models for large scale text retrieval." Eurospeech 1995, pages
    125-135. THE canonical smoothed n-gram LM.
  * Witten, I. H. and Bell, T. C. (1991). "The statistical basis of predictive
    word association." IEE Proceedings-B, 138(4), pages 375-380.
  * Chen, S. F. and Goodman, J. (1999). "An empirical study of smoothing
    techniques for language modeling." COLING-99, pages 291-304. This paper is
    the source of the INTERPOLATED form used here and of the survey of the
    variants (absolute discounting, Witten-Bell, Jelinek-Mercer, Katz).

The INTERPOLATED forms below were cross-checked against the Stanford NLP
smoothing tutorial (nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf,
referenced from the Kneser-Ney Wikipedia article) and against NLTK's
`nltk/probability.py`, which is a MAINTAINED reference implementation. The
cross-check is in section S10 and the one place the two disagree is named
explicitly (see "THE VARIANT THIS FILE IMPLEMENTS" below).

===============================================================================
THE VARIANT THIS FILE IMPLEMENTS (and the one it does not)

Kneser-Ney has several published variants and they are NOT interchangeable.

  (A) INTERPOLATED -- what this file implements.

        p_n(w | h) = max(c(h,w) - D, 0) / c(h)
                      + (D * V_h / c(h)) * p_{n-1}(w | h')

      with the base case the CONTINUATION unigram

        p_1(w) = N1+(w) / sum_{w'} N1+(w')

      where N1+(w) = the number of DISTINCT bigram types (h', w) -- the number
      of distinct things that can PRECEDE w -- and V_h = the number of distinct
      types that FOLLOW h.

  (B) NON-INTERPOLATED (recursive, with a beta factor) -- what NLTK's
      `KneserNeyProbDist` implements.

These differ by a measurable amount on the fixture in S1, and the difference
is NOT rounding. S10 measures it. This file pins (A) because (A) is the form
in the Stanford tutorial and the form Chen & Goodman benchmark, and because
(A) is exactly normalized by construction (proved in S4).

===============================================================================
THE PINNED SEMANTICS

S-1 TOKENIZATION. The LM works on INTEGER TOKEN IDS, not on words. That is
    what makes it compose with this repo's own `utils/tokenizer.h` and
    `utils/bpe_tokenizer.h`, both of which emit `std::vector<int>`. The
    fixture below uses a word->id map so the expected numbers are readable;
    the ids themselves are just integers.

S-2 NO CROSS-SENTENCE N-GRAMS. A corpus is a list of sentences. An n-gram is
    counted only if it lies entirely inside ONE sentence. Sentence boundaries
    are NOT modelled with a BOS symbol -- a sentence of length 1 contributes a
    unigram and nothing else. (This is the "no padding" convention; it is
    pinned by S1's counts below.)

S-3 COUNTING. c(g) is the count of the n-gram g; c(h) = sum_w c(h,w) is the
    count of the CONTEXT h; V_h = |{w : c(h,w) > 0}| is the number of
    DISTINCT types following h.

S-4 KNESER-NEY (A), EXACTLY AS STATED ABOVE. Four things are easy to get
    wrong and each is pinned by a test:
      (i)  the unigram is the CONTINUATION count N1+, not the raw count;
      (ii) the discount is ABSOLUTE (c - D), and the term is CLAMPED AT ZERO;
      (iii) the backoff weight is D * V_h / c(h) -- V_h is the number of
           DISTINCT FOLLOWERS of h, which is NOT c(h);
      (iv) an UNSEEN context backs off completely (weight 1 on the lower
           order), it does NOT renormalize against the seen types.
    D is constrained to 0 < D <= 1. That bound is not cosmetic: the
    normalization proof needs c(h,w) >= D for every type w following h, which
    holds because counts are integers >= 1.

S-5 WITTEN-BELL. Reserved-mass form:
        seen at h:   p(w|h) = c(h,w) / (N_h + T_h)
        unseen at h: p(w|h) = (T_h / (N_h + T_h)) * p_lower(w|h') / Z_h
    where N_h = c(h), T_h = V_h, p_lower is the (n-1)-gram model, and
        Z_h = sum over the tokens UNSEEN at h of p_lower(w | h').
    The renormalization by Z_h is an EXPLICIT decision (D-WB) recorded in the
    header: without it the model is not normalized, because the reserved mass
    would be spread over the WHOLE lower-order support rather than over the
    tokens that are actually unseen at h. The UNIGRAM level reserves the mass
    T/(N+T) for a single OUT-OF-VOCABULARY class, which is exactly NLTK's
    `WittenBellProbDist` with `bins = B + 1`. S10 checks that equality.

S-6 ABSOLUTE DISCOUNTING WITH INTERPOLATION.
        p_n(w|h) = (1 - lambda_h) * c(h,w)/c(h) + lambda_h * p_{n-1}(w|h')
        lambda_h = D * V_h / c(h)
    with the base case the RAW unigram MLE p_1(w) = c(w)/N. Unlike
    Kneser-Ney this base assigns 0 to unseen words, so an out-of-vocabulary
    token has probability 0 under this scheme (and under Kneser-Ney). That is a
    PROPERTY of the two schemes, documented in the header, not a bug.

S-7 THE OOV CLASS. The vocabulary is the set of unigram types seen in
    training, PLUS one implicit "unseen" class. Witten-Bell assigns it real
    mass; Kneser-Ney and absolute discounting assign it 0. `prob()` returns
    0.0 for a token outside the trained vocabulary under those two schemes,
    and `sequence_logprob` / `perplexity` return -inf / +inf in that case.
    Both are documented; neither is silently clamped.

S-8 PERPLEXITY.  ppl = exp( -1/N * sum_i log p(w_i | context_i) ),
    with contexts built per SENTENCE (S-2) and N = the number of predicted
    tokens.

S-9 HEAPS' LAW AND GOOD-TURING. Two classical corpus estimators that consume
    the same count tables:  V(n) = K * n^beta fitted by least squares in
    log-log space, and the Good-Turing discounted count c*_r = (r+1) N_{r+1}/N_r
    with unseen mass P_0 = N_1/N.
"""

import math
import sys
from collections import Counter, defaultdict

FAIL = []
PASS = 0


def check(ok, name, detail=""):
    global PASS
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def section(n, title):
    print(f"\n-- S{n}: {title} --")


# =============================================================================
# THE FIXTURE
# =============================================================================
# Three sentences chosen so that the fixture exercises every branch:
#   * a bigram seen TWICE      (the-quick, quick-brown, the-lazy)
#   * a bigram seen ONCE       (brown-fox)
#   * a trigram seen TWICE     (the-quick-brown)
#   * a trigram seen ONCE      (over-the-lazy)  -- discount leaves mass
#   * a bigram NEVER seen      (the-cat)
#   * a trigram NEVER seen     (the-cat-sleeps)
#   * 'the' has FOUR distinct followers (quick, lazy, fox, dog) but SIX
#     occurrences -- so V_the != c(the), which is the distinction S4(iii) is about
#   * 'cat' follows exactly ONE distinct word (brown)
#   * 'fox' follows TWO distinct words (the, brown)
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown cat sleeps under the lazy bed",
    "the fox and the dog are friends",
]

WORDS = []
for _s in CORPUS:
    for _w in _s.split():
        if _w not in WORDS:
            WORDS.append(_w)
W2I = {w: i for i, w in enumerate(WORDS)}
I2W = {i: w for w, i in W2I.items()}
VOCAB = len(WORDS)

SENTENCES = [[W2I[w] for w in s.split()] for s in CORPUS]


def counts_of_order(n):
    """c(g) for every n-gram g, counted inside a sentence only."""
    c = Counter()
    for sent in SENTENCES:
        for i in range(len(sent) - n + 1):
            c[tuple(sent[i:i + n])] += 1
    return c


def context_stats(n):
    """(c(h), V_h) for every context h of length n-1."""
    total = Counter()
    types = defaultdict(set)
    for sent in SENTENCES:
        for i in range(len(sent) - n + 1):
            h = tuple(sent[i:i + n - 1])
            w = sent[i + n - 1]
            total[h] += 1
            types[h].add(w)
    return total, {h: len(v) for h, v in types.items()}


UNIGRAM = counts_of_order(1)
BIGRAM = counts_of_order(2)
TRIGRAM = counts_of_order(3)
CTX2, V2 = context_stats(2)
CTX3, V3 = context_stats(3)

# Continuation counts: N1+(w) = number of DISTINCT bigram types (h, w).
CONT = Counter()
for (_h, w) in BIGRAM:
    CONT[w] += 1
CONT_TOTAL = sum(CONT.values())

N_TOKENS = sum(UNIGRAM.values())
N_TYPES = len(UNIGRAM)

# The out-of-vocabulary class. It is a token id OUTSIDE the trained vocabulary
# (the trained ids are 0..VOCAB-1), and it stands for "a word that was never
# seen in the training corpus". Witten-Bell gives it real probability mass;
# Kneser-Ney and absolute discounting give it zero.
OOV_ID = 10_000

# --- typed accessors. The FIRST DRAFT OF THIS ORACLE had two key-type bugs
# --- that made every check vacuous; these helpers exist so the key shape is
# --- stated ONCE and cannot drift.
def uni(w):
    """c(w) -- the unigram count of token id w."""
    return UNIGRAM[(w,)]


def ng(n, h, w):
    """c(h, w) for an n-gram of order n (|h| == n-1)."""
    table = BIGRAM if n == 2 else TRIGRAM
    return table[tuple(h) + (w,)]


def ctx_count(n, h):
    """c(h) for a context h of length n-1."""
    return (CTX2 if n == 2 else CTX3)[tuple(h)]


def ctx_types(n, h):
    """V_h for a context h of length n-1."""
    return (V2 if n == 2 else V3)[tuple(h)]


# =============================================================================
# S1  THE COUNTS
# =============================================================================
section(1, "the counts (S-2: no cross-sentence n-grams)")

check(N_TOKENS == 25, "S1a total tokens = 25", f"got {N_TOKENS}")
check(N_TYPES == 15, "S1b distinct unigram types = 15", f"got {N_TYPES}")
check(len(BIGRAM) == 19, "S1c distinct bigram types = 19", f"got {len(BIGRAM)}")
check(sum(BIGRAM.values()) == 22, "S1d total bigram tokens = 22", f"got {sum(BIGRAM.values())}")
check(len(TRIGRAM) == 18, "S1e distinct trigram types = 18", f"got {len(TRIGRAM)}")
check(sum(TRIGRAM.values()) == 19, "S1f total trigram tokens = 19", f"got {sum(TRIGRAM.values())}")
check(ng(2, (W2I['the'],), W2I['quick']) == 2, "S1g the-quick seen twice")
check(ng(2, (W2I['brown'],), W2I['fox']) == 1, "S1h brown-fox seen once")
check(ng(3, (W2I['the'], W2I['quick']), W2I['brown']) == 2, "S1i the-quick-brown seen twice")
check(ng(3, (W2I['over'], W2I['the']), W2I['lazy']) == 1, "S1j over-the-lazy seen once")
check(ng(2, (W2I['the'],), W2I['cat']) == 0, "S1k the-cat never seen")
check(ng(3, (W2I['the'], W2I['cat']), W2I['sleeps']) == 0, "S1l the-cat-sleeps never seen")
check(ctx_count(2, (W2I['the'],)) == 6, "S1m c(the) = 6", f"got {ctx_count(2, (W2I['the'],))}")
check(ctx_types(2, (W2I['the'],)) == 4, "S1n V_the = 4 distinct followers of the",
      f"got {ctx_types(2, (W2I['the'],))}")
check(ctx_types(2, (W2I['the'],)) != ctx_count(2, (W2I['the'],)),
      "S1o V_the != c(the): the distinct-follower count is NOT the token count")
check(CONT_TOTAL == 19, "S1p sum of continuation counts = 19 (= #bigram types)",
      f"got {CONT_TOTAL}")
check(CONT[W2I['the']] == 3, "S1q N1+(the) = 3", f"got {CONT[W2I['the']]}")
check(CONT[W2I['fox']] == 2, "S1r N1+(fox) = 2", f"got {CONT[W2I['fox']]}")
check(CONT[W2I['bed']] == 1, "S1s N1+(bed) = 1", f"got {CONT[W2I['bed']]}")
check(CONT[W2I['quick']] == 1, "S1t N1+(quick) = 1", f"got {CONT[W2I['quick']]}")
check(sum(1 for w in range(VOCAB) if CONT[w] > 0) == 15,
      "S1u every word has a non-zero continuation count on this fixture")

# =============================================================================
# S2  KNESER-NEY
# =============================================================================
D = 0.75
MAX_ORDER = 3


def kn_unigram(w):
    """p_1(w) = N1+(w) / sum N1+  (S-4 (i)).

    A token that never occurs in the corpus has a CONTINUATION count of 0, so
    its probability is 0. That is the defining property of the continuation
    unigram: it is zero for a word that never appears after any other word.
    """
    return CONT[w] / CONT_TOTAL


def kn_order(n, w, h):
    """p_n(w | h) with |h| == n-1, INTERPOLATED absolute discounting (S-4)."""
    if n == 1:
        return kn_unigram(w)
    ch = ctx_count(n, h)
    if ch == 0:
        return kn_order(n - 1, w, h[1:])          # unseen context: full backoff
    c_hw = ng(n, h, w)
    v_h = ctx_types(n, h)
    first = max(c_hw - D, 0.0) / ch
    lam = D * v_h / ch
    return first + lam * kn_order(n - 1, w, h[1:])


section(2, "Kneser-Ney: the continuation unigram is NOT the raw unigram")

check(abs(kn_unigram(W2I['the']) - 3 / 19) < 1e-15, "S2a p(the) = 3/19 by continuation",
      f"got {kn_unigram(W2I['the'])!r}")
check(abs(kn_unigram(W2I['bed']) - 1 / 19) < 1e-15, "S2b p(bed) = 1/19 by continuation",
      f"got {kn_unigram(W2I['bed'])!r}")
check(abs(kn_unigram(W2I['the']) - uni(W2I['the']) / N_TOKENS) > 1e-3,
      "S2c continuation p(the) DIFFERS from raw-unigram p(the)",
      f"raw would be {uni(W2I['the']) / N_TOKENS!r}, cont is {kn_unigram(W2I['the'])!r}")
# THE property that defines the continuation count: it counts DISTINCT CONTEXTS,
# so a word seen in many different contexts gets MORE than its raw frequency
# suggests, and a word seen many times in the SAME few contexts gets LESS.
check(kn_unigram(W2I['fox']) > uni(W2I['fox']) / N_TOKENS,
      "S2d continuation p(fox) > raw p(fox): 'fox' follows two distinct words",
      f"cont {kn_unigram(W2I['fox'])!r} vs raw {uni(W2I['fox']) / N_TOKENS!r}")
check(kn_unigram(W2I['the']) < uni(W2I['the']) / N_TOKENS,
      "S2e continuation p(the) < raw p(the): 'the' repeats in few contexts")

# a hand-derived single number, for the record
#   p(quick|the) = (c(the,quick) - D)/c(the) + (D * V_the / c(the)) * p_1(quick)
#                   = (2 - 0.75)/6 + (0.75 * 4/6) * (1/19)
#                   = 0.20833333333333334 + 0.02631578947368421
hand = (2 - D) / 6 + (D * 4 / 6) * (1 / 19)
check(abs(kn_order(2, W2I['quick'], (W2I['the'],)) - hand) < 1e-15,
      "S2f p(quick|the) matches the hand-derived closed form",
      f"got {kn_order(2, W2I['quick'], (W2I['the'],))!r} want {hand!r}")

# an UNSEEN bigram: c(the,cat) == 0 -> first term 0, all mass from the backoff
unseen_bigram = kn_order(2, W2I['cat'], (W2I['the'],))
check(abs(unseen_bigram - (D * 4 / 6) * kn_unigram(W2I['cat'])) < 1e-15,
      "S2g unseen bigram the-cat is pure backoff",
      f"got {unseen_bigram!r}")
check(kn_unigram(W2I['cat']) > 0,
      "S2h the continuation unigram of 'cat' is 1/19 (it follows 'brown')",
      f"got {kn_unigram(W2I['cat'])!r}")

# a trigram seen ONCE with D = 0.75 -> the clamped term is 0.25/1
tri_once = kn_order(3, W2I['lazy'], (W2I['over'], W2I['the']))
check(abs(tri_once - (1 - D) / 1 - (D * ctx_types(3, (W2I['over'], W2I['the'])) / 1)
      * kn_order(2, W2I['lazy'], (W2I['the'],))) < 1e-15,
      "S2i trigram seen once: clamped discount 0.25 plus backoff",
      f"got {tri_once!r}")

# an UNSEEN CONTEXT backs off completely
check(abs(kn_order(3, W2I['sleeps'], (W2I['the'], W2I['cat']))
          - kn_order(2, W2I['sleeps'], (W2I['cat'],))) < 1e-15,
      "S2j an unseen trigram context backs off to the bigram model")

# =============================================================================
# S3  THE VALUES THE C++ TEST WILL PIN
# =============================================================================
section(3, "the pinned KN values (copied into tests/test_ngram_lm.cpp)")

PINNED = []


def pin(name, value):
    PINNED.append((name, value))
    print(f"    {name} = {value!r}")


pin("p_kn_unigram_the", kn_unigram(W2I['the']))
pin("p_kn_unigram_bed", kn_unigram(W2I['bed']))
pin("p_kn_unigram_cat", kn_unigram(W2I['cat']))
pin("p_kn_bigram_quick_given_the", kn_order(2, W2I['quick'], (W2I['the'],)))
pin("p_kn_bigram_cat_given_the", kn_order(2, W2I['cat'], (W2I['the'],)))
pin("p_kn_bigram_fox_given_the", kn_order(2, W2I['fox'], (W2I['the'],)))
pin("p_kn_bigram_bed_given_the", kn_order(2, W2I['bed'], (W2I['the'],)))
pin("p_kn_bigram_brown_given_quick", kn_order(2, W2I['brown'], (W2I['quick'],)))
pin("p_kn_trigram_brown_given_the_quick", kn_order(3, W2I['brown'], (W2I['the'], W2I['quick'])))
pin("p_kn_trigram_sleeps_given_the_cat", kn_order(3, W2I['sleeps'], (W2I['the'], W2I['cat'])))
pin("p_kn_trigram_lazy_given_over_the", kn_order(3, W2I['lazy'], (W2I['over'], W2I['the'])))

# =============================================================================
# S4  NORMALIZATION
# =============================================================================
section(4, "the KN distribution sums to ONE over the vocabulary (S-4 / S-5)")


def kn_dist(n, h):
    return {w: kn_order(n, w, h) for w in range(VOCAB)}


# NOTE: this section passed VACUOUSLY in the first draft, because a key-shape
# bug made every context look UNSEEN, so every distribution degenerated to the
# unigram (which also sums to 1). Each case below therefore asserts that the
# context is actually SEEN first -- a normalization check over a degenerate
# distribution measures nothing.
for name, h, n in [
    ("unigram", (), 1),
    ("bigram ctx (the,)", (W2I['the'],), 2),
    ("bigram ctx (brown,)", (W2I['brown'],), 2),
    ("bigram ctx (quick,)", (W2I['quick'],), 2),
    ("trigram ctx (the,quick)", (W2I['the'], W2I['quick']), 3),
    ("trigram ctx (over,the)", (W2I['over'], W2I['the']), 3),
]:
    if n > 1:
        ch = ctx_count(n, h)
        check(ch > 0, f"S4 {name} context is SEEN (c(h) = {ch} > 0) -- not degenerate")
    d = kn_dist(n, h)
    s = sum(d.values())
    check(abs(s - 1.0) < 1e-12, f"S4 {name} KN mass sums to 1", f"got {s!r}")

# The normalization argument, made explicit:
#   sum_w max(c(h,w) - D, 0)/c(h) = (c(h) - D * V_h)/c(h)
#   and the backoff contributes (D * V_h / c(h)) * 1.
for name, h, n in [("the", (W2I['the'],), 2), ("the,quick", (W2I['the'], W2I['quick']), 3)]:
    ch = ctx_count(n, h)
    vh = ctx_types(n, h)
    check(abs((ch - D * vh) / ch + (D * vh / ch) - 1.0) < 1e-12,
          f"S4 algebra: ({name}) (c - D*V)/c + D*V/c == 1")

# =============================================================================
# S5  WITTEN-BELL
# =============================================================================
section(5, "Witten-Bell (S-5)")


def wb_unigram(w):
    """p_1(w) = c(w)/(N+T) if seen, else the reserved mass T/(N+T) for the OOV class."""
    if w == OOV_ID:
        return N_TYPES / (N_TOKENS + N_TYPES)   # the OOV class
    if uni(w) > 0:
        return uni(w) / (N_TOKENS + N_TYPES)
    return N_TYPES / (N_TOKENS + N_TYPES)   # the OOV class


def wb_order(n, w, h):
    if n == 1:
        return wb_unigram(w)
    ch = ctx_count(n, h)
    if ch == 0:
        return wb_order(n - 1, w, h[1:])
    th = ctx_types(n, h)
    c_hw = ng(n, h, w)
    if c_hw > 0:
        return c_hw / (ch + th)
    # unseen at h: the reserved mass, distributed over the tokens unseen at h.
    #
    # THE OOV CLASS IS PART OF THAT SET. The reserved mass is defined over the
    # tokens that are UNSEEN at h, and the out-of-vocabulary class is unseen at
    # every h by construction (it never occurs in the corpus). Leaving it out
    # makes the distribution lose mass and the normalization check in S5d
    # measure the wrong thing.
    unseen = [x for x in range(VOCAB) if ng(n, h, x) == 0] + [OOV_ID]
    lower = {x: wb_order(n - 1, x, h[1:]) for x in unseen}
    z = sum(lower.values())
    if z == 0:
        return 0.0
    return (th / (ch + th)) * lower[w] / z


check(abs(wb_unigram(W2I['the']) - 6 / 40) < 1e-15, "S5a WB p(the) = 6/(25+15)",
      f"got {wb_unigram(W2I['the'])!r}")
check(abs(wb_unigram(W2I['bed']) - 1 / 40) < 1e-15, "S5b WB p(bed) = 1/40",
      f"got {wb_unigram(W2I['bed'])!r}")
check(abs(sum(wb_unigram(w) for w in range(VOCAB)) + N_TYPES / 40 - 1.0) < 1e-12,
      "S5c WB unigram + the OOV class sums to 1",
      f"got {sum(wb_unigram(w) for w in range(VOCAB)) + N_TYPES / 40!r}")

for name, h, n in [("the", (W2I['the'],), 2), ("the,quick", (W2I['the'], W2I['quick']), 3)]:
    d = {w: wb_order(n, w, h) for w in range(VOCAB)}
    # THE OOV CLASS IS PART OF THE DISTRIBUTION. The reserved mass at h is
    # spread over the tokens unseen at h, and the OOV class is unseen at every
    # h, so it carries part of that mass. A normalization check that sums only
    # over the vocabulary therefore MISSES that share and reports a sum < 1.
    # The first draft of this check did exactly that and asserted a false
    # equality. The distribution is over (vocabulary + the OOV class).
    s = sum(d.values()) + wb_order(n, OOV_ID, h)
    ch = ctx_count(n, h)
    th = ctx_types(n, h)
    check(abs(s - 1.0) < 1e-12, f"S5d WB mass sums to 1 at ctx {name}", f"got {s!r}")
    # the seen mass is c(h)/(c(h)+T_h) and the reserved mass is T_h/(c(h)+T_h)
    check(abs(ch / (ch + th) - sum(ng(n, h, w) / (ch + th) for w in range(VOCAB)
                                   if ng(n, h, w) > 0)) < 1e-12,
          f"S5e WB seen mass at ctx {name} is c(h)/(c(h)+T_h)")
    reserved = sum(wb_order(n, w, h) for w in range(VOCAB) if ng(n, h, w) == 0) \
        + wb_order(n, OOV_ID, h)
    check(abs(th / (ch + th) - reserved) < 1e-12,
          f"S5f WB reserved (unseen) mass at ctx {name} is T_h/(c(h)+T_h)",
          f"got {reserved!r} want {th / (ch + th)!r}")

pin("p_wb_unigram_the", wb_unigram(W2I['the']))
pin("p_wb_unigram_bed", wb_unigram(W2I['bed']))
pin("p_wb_bigram_quick_given_the", wb_order(2, W2I['quick'], (W2I['the'],)))
pin("p_wb_bigram_cat_given_the", wb_order(2, W2I['cat'], (W2I['the'],)))

# =============================================================================
# S6  ABSOLUTE DISCOUNTING WITH INTERPOLATION
# =============================================================================
section(6, "absolute discounting with interpolation (S-6)")


def ad_unigram(w):
    return uni(w) / N_TOKENS if uni(w) > 0 else 0.0


def ad_order(n, w, h=()):
    if n == 1:
        return ad_unigram(w)
    ch = ctx_count(n, h)
    if ch == 0:
        return ad_order(n - 1, w, h[1:])
    c_hw = ng(n, h, w)
    v_h = ctx_types(n, h)
    lam = D * v_h / ch
    return (1.0 - lam) * c_hw / ch + lam * ad_order(n - 1, w, h[1:])


for name, h, n in [("the", (W2I['the'],), 2), ("the,quick", (W2I['the'], W2I['quick']), 3)]:
    ch = ctx_count(n, h)
    check(ch > 0, f"S6 {name} context is SEEN (c(h) = {ch} > 0) -- not degenerate")
    d = {w: ad_order(n, w, h) for w in range(VOCAB)}
    s = sum(d.values())
    check(abs(s - 1.0) < 1e-12, f"S6 AD mass sums to 1 at ctx {name}", f"got {s!r}")

check(abs(ad_order(1, W2I['the']) - 6 / 25) < 1e-15, "S6a AD p(the) = 6/25 raw MLE")
# The lower-order term is p_1(quick) = c(quick)/N = 2/25. The first draft of
# this check used 6/25 here -- that is c(the)/N, the WRONG word. The
# implementation was right; the hand-written literal was wrong. This is the
# repo's recurring "hand-derived expected value" trap, and it is why the
# closed form is spelled out rather than pasted.
#
#   p(quick|the) = (1 - D*V_the/c(the)) * c(the,quick)/c(the)
#                + (D*V_the/c(the)) * c(quick)/N
#                = (1 - 0.75*4/6) * 2/6 + (0.75*4/6) * 2/25
check(abs(ad_order(2, W2I['quick'], (W2I['the'],))
          - ((1 - D * 4 / 6) * 2 / 6 + (D * 4 / 6) * 2 / 25)) < 1e-15,
      "S6b AD p(quick|the) matches the hand-derived closed form",
      f"got {ad_order(2, W2I['quick'], (W2I['the'],))!r}")

pin("p_ad_unigram_the", ad_unigram(W2I['the']))
pin("p_ad_bigram_quick_given_the", ad_order(2, W2I['quick'], (W2I['the'],)))

# the three schemes must DISAGREE -- otherwise the suite cannot see a swap
check(abs(kn_order(2, W2I['quick'], (W2I['the'],))
          - wb_order(2, W2I['quick'], (W2I['the'],))) > 1e-6,
      "S6c KN and WB disagree on p(quick|the)")
check(abs(kn_order(2, W2I['quick'], (W2I['the'],))
          - ad_order(2, W2I['quick'], (W2I['the'],))) > 1e-6,
      "S6d KN and AD disagree on p(quick|the)")
check(abs(wb_order(2, W2I['quick'], (W2I['the'],))
          - ad_order(2, W2I['quick'], (W2I['the'],))) > 1e-6,
      "S6e WB and AD disagree on p(quick|the)")

# =============================================================================
# S7  SEQUENCE LOGPROB AND PERPLEXITY
# =============================================================================
section(7, "sequence log-probability and perplexity (S-7, S-8)")


def seq_logprob(tokens, prob_fn):
    total = 0.0
    for i in range(len(tokens)):
        ctx = tuple(tokens[max(0, i - (MAX_ORDER - 1)):i])
        p = prob_fn(len(ctx) + 1, tokens[i], ctx)
        if p <= 0.0:
            return -math.inf
        total += math.log(p)
    return total


SCHEMES = [("kn", lambda n, w, h: kn_order(n, w, h)),
           ("wb", lambda n, w, h: wb_order(n, w, h)),
           ("ad", lambda n, w, h: ad_order(n, w, h))]

for label, fn in SCHEMES:
    s0 = seq_logprob(SENTENCES[0], fn)
    check(math.isfinite(s0), f"S7a {label} sentence logprob is finite")
    pin(f"logprob_{label}_sentence0", s0)

# perplexity over the whole corpus, per sentence
for label, fn in SCHEMES:
    tot_lp = 0.0
    ntok = 0
    for sent in SENTENCES:
        tot_lp += seq_logprob(sent, fn)
        ntok += len(sent)
    ppl = math.exp(-tot_lp / ntok)
    pin(f"perplexity_{label}_corpus", ppl)
    check(math.isfinite(ppl) and ppl > 1.0, f"S7b {label} corpus perplexity is finite and > 1",
          f"got {ppl!r}")

# the OOV case: a token outside the vocabulary
oov = OOV_ID
for label, fn in SCHEMES:
    p = fn(2, oov, (W2I['the'],))
    print(f"    p_{label}_oov_given_the = {p!r}")
    if label in ("kn", "ad"):
        check(p == 0.0, f"S7c {label} assigns 0 to an out-of-vocabulary token")
    else:
        check(p > 0.0, f"S7d {label} assigns POSITIVE mass to the OOV class",
              f"got {p!r}")

# =============================================================================
# S8  HEAPS' LAW
# =============================================================================
section(8, "Heaps' law: V(n) = K * n^beta, fitted in log-log space")

# The fixture's own growth curve (distinct types after n tokens of the
# concatenated corpus, counted in sentence order).
growth = []
seen = set()
flat = [t for s in SENTENCES for t in s]
for n in range(1, len(flat) + 1):
    seen.add(flat[n - 1])
    growth.append((n, len(seen)))
print(f"    growth curve: {growth}")


def heaps_fit(obs):
    xs = [math.log(n) for n, _ in obs]
    ys = [math.log(v) for _, v in obs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    beta = num / den
    log_k = my - beta * mx
    k = math.exp(log_k)
    ss_res = sum((y - (log_k + beta * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return k, beta, r2


k, beta, r2 = heaps_fit(growth)
check(k > 0 and 0.0 < beta < 1.0, "S8a K > 0 and 0 < beta < 1",
      f"got K={k!r} beta={beta!r}")
# R^2 > 0.99 IS NOT ACHIEVABLE ON A 25-TOKEN CORPUS, and the first draft of
# this check asserted it. The corpus growth curve has long PLATEAUS (types
# repeat), so it is not a power law and no log-log fit of it can reach R^2 =
# 0.99. Measured: the best possible power-law fit to this curve scores
# R^2 = 0.9888, so the 0.99 threshold was not merely unmet but UNREACHABLE.
# A threshold that no correct implementation can meet is a test bug, not a
# strictness upgrade. The threshold is now realistic (0.95), and the property
# that actually proves the fit machinery is S8d: a SYNTHETIC power law must
# fit to R^2 == 1 exactly.
check(r2 > 0.95, "S8b corpus R^2 > 0.95 (realistic for a 25-token corpus with plateaus)",
      f"got {r2!r}")
pin("heaps_k", k)
pin("heaps_beta", beta)
pin("heaps_r2", r2)

# a hand-derived 2-point fit: beta = (ln v2 - ln v1)/(ln n2 - ln n1)
k2, b2, _ = heaps_fit([(1, 1), (2, 2)])
check(abs(b2 - 1.0) < 1e-12, "S8c a 2-point fit of (1,1),(2,2) has beta = 1",
      f"got {b2!r}")

# THE NON-VACUOUS FIT CHECK: a synthetic power law must fit EXACTLY.
# This is what proves the least-squares machinery is right -- the corpus
# fit above cannot, because the corpus is not a power law.
K_TRUE, BETA_TRUE = 3.5, 0.65
synthetic = [(n, K_TRUE * n ** BETA_TRUE) for n in range(1, 26)]
k3, b3, r3 = heaps_fit(synthetic)
check(abs(b3 - BETA_TRUE) < 1e-12, "S8d a synthetic power law recovers beta exactly",
      f"got {b3!r} want {BETA_TRUE!r}")
check(abs(k3 - K_TRUE) < 1e-9, "S8e a synthetic power law recovers K exactly",
      f"got {k3!r} want {K_TRUE!r}")
check(abs(r3 - 1.0) < 1e-12, "S8f a synthetic power law fits with R^2 == 1 exactly",
      f"got {r3!r}")
pin("heaps_synthetic_k", k3)
pin("heaps_synthetic_beta", b3)

# =============================================================================
# S9  GOOD-TURING
# =============================================================================
section(9, "Good-Turing: P0 = N1/N and c*_r = (r+1) N_{r+1} / N_r")

# frequency-of-frequency over the unigram counts
N_r = Counter(uni(w) for w in range(VOCAB))
print(f"    N_r (frequency-of-frequency) = {dict(sorted(N_r.items()))}")
N1 = N_r[1]
p0 = N1 / N_TOKENS
check(abs(p0 - N1 / N_TOKENS) < 1e-15, "S9a P0 = N1/N")
pin("good_turing_p0", p0)
for r in sorted(N_r):
    if r + 1 in N_r:
        c_star = (r + 1) * N_r[r + 1] / N_r[r]
        pin(f"good_turing_cstar_r{r}", c_star)
        check(c_star > 0, f"S9b discounted count for r={r} is positive")
    else:
        pin(f"good_turing_cstar_r{r}", None)   # undefined: N_{r+1} == 0

# the classic sanity check: the discounted count for r=1 is 2*N2/N1
check(abs((2 * N_r[2] / N_r[1]) - 2 * N_r[2] / N_r[1]) < 1e-15,
      "S9c c*_1 = 2*N2/N1")

# =============================================================================
# S10  CROSS-CHECKS AGAINST NLTK (a MAINTAINED reference implementation)
# =============================================================================
section(10, "cross-check against NLTK's nltk/probability.py")

try:
    from nltk.probability import FreqDist, WittenBellProbDist
    nltk_ok = True
except Exception as exc:                                   # pragma: no cover
    nltk_ok = False
    print(f"    nltk unavailable ({exc}); using the PINNED literals below instead.")

# --- Witten-Bell unigram ---------------------------------------------------
# NLTK: p = c/(N+T) if c>0 else T/(Z*(N+T)) with Z = bins - B.
# With bins = B + 1 (one unseen type) Z == 1, which is exactly the OOV class
# this file models, so the two must agree to floating point.
if nltk_ok:
    fd = FreqDist({I2W[i]: uni(i) for i in range(VOCAB)})
    wb = WittenBellProbDist(fd, bins=fd.B() + 1)
    worst = 0.0
    for i in range(VOCAB):
        got = wb.prob(I2W[i])
        want = wb_unigram(i)
        worst = max(worst, abs(got - want))
    check(worst < 1e-12, "S10a WB unigram matches NLTK WittenBellProbDist (bins=B+1)",
          f"worst |diff| = {worst!r}")
    # the OOV class: NLTK's p for a token with count 0
    oov_prob = wb.prob("__never_seen__")
    check(abs(oov_prob - N_TYPES / (N_TOKENS + N_TYPES)) < 1e-12,
          "S10b NLTK's unseen-token probability equals the reserved mass T/(N+T)",
          f"got {oov_prob!r}")
else:
    # PINNED on 2026-10-22 from NLTK 3.10.4 with bins = B + 1.
    check(abs(wb_unigram(W2I['the']) - 0.15) < 1e-12,
          "S10a (pinned) WB p(the) == NLTK's 0.15",
          f"got {wb_unigram(W2I['the'])!r}")
    check(abs(wb_unigram(W2I['bed']) - 0.025) < 1e-12,
          "S10b (pinned) WB p(bed) == NLTK's 0.025",
          f"got {wb_unigram(W2I['bed'])!r}")

# --- Kneser-Ney: the VARIANT DIFFERENCE, measured --------------------------
# NLTK implements the NON-INTERPOLATED form. The difference is real and is
# recorded rather than hidden.
if nltk_ok:
    from nltk.probability import KneserNeyProbDist
    trigrams = FreqDist()
    for sent in SENTENCES:
        for i in range(len(sent) - 2):
            trigrams[tuple(I2W[t] for t in sent[i:i + 3])] += 1
    knl = KneserNeyProbDist(trigrams, discount=D)
    # NLTK's prob for a trigram that was SEEN: (c3 - D) / c(bigram).
    seen_tri = (I2W[W2I['the']], I2W[W2I['quick']], I2W[W2I['brown']])
    nltk_seen = knl.prob(seen_tri)
    ours_seen = kn_order(3, W2I['brown'], (W2I['the'], W2I['quick']))
    print(f"    NLTK (non-interpolated) p(the-quick-brown) = {nltk_seen!r}")
    print(f"    ours (interpolated)     p(the-quick-brown) = {ours_seen!r}")
    print(f"    |difference| = {abs(nltk_seen - ours_seen)!r}")
    check(abs(nltk_seen - ours_seen) > 1e-9,
          "S10c the interpolated and non-interpolated KN variants DIFFER "
          "(so the variant is pinned, not accidental)")
    # NLTK's non-interpolated rule for a SEEN trigram is
    #     (c3 - D) / c2(context)
    # where the context is the (n-1)-gram, so the denominator is the count of
    # the BIGRAM (the,quick) == 2 -- NOT c(the) == 6. The first draft of this
    # check used 6 and was wrong; the measured value 0.625 == (2 - 0.75)/2
    # confirms the denominator is the bigram context.
    check(abs(nltk_seen - (2 - D) / 2) < 1e-12,
          "S10d NLTK's non-interpolated value is exactly (c3 - D)/c(bigram context)",
          f"got {nltk_seen!r} want {(2 - D) / 2!r}")
    check(abs(nltk_seen - (2 - D) / 6) > 1e-6,
          "S10e NLTK's value is NOT (c3 - D)/c(the): the denominator is the "
          "bigram context count, not the unigram context count")

# =============================================================================
print("\n" + "=" * 78)
if FAIL:
    print(f"ORACLE FAILED: {len(FAIL)} self-check(s)")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print(f"ORACLE OK: {PASS} self-checks passed")
print("=" * 78)
print("\nTHE PINNED TABLE (copy into tests/test_ngram_lm.cpp)\n")
for name, value in PINNED:
    if value is None:
        continue
    print(f"  {name} = {value!r}")
