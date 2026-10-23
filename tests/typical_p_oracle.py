#!/usr/bin/env python3
"""
tests/typical_p_oracle.py -- numpy float64 oracle for TYPICAL-P (locally
typical sampling), the truncation added to
include/nn/utils/logit_processing.{h,cpp}.

WHY AN ORACLE AND NOT HAND-DERIVATION. Every numeric expectation in
tests/test_logit_processing.cpp is copied from this file's output. Two earlier
sessions in this repo caught a real bug in the ORACLE itself before a line of
C++ existed (tests/test_vat.cpp D9; tests/sinkhorn_oracle.py, "7 oracle
findings"), and the TDD skill is explicit that a hand-derived expected value in
a test is a leading cause of phantom implementation bugs -- a single-digit typo
ships a test that fails for the wrong reason.

THE ALGORITHM, AS THE PAPER STATES IT. Meister, Pimentel, Wiher, Cotterell,
"Locally Typical Sampling", TACL 11:102-121, 2023, arXiv:2202.00666 (v6).
Read from the paper's own HTML (https://arxiv.org/html/2202.00666v6):

  Eq. 7 / Definition 3.5 -- a token y is LOCALLY TYPICAL at step t when
      | log p(y | y_<t) + H(Y_t | Y_<t) |  <  eps
  i.e. its NEGATIVE LOG-PROBABILITY is within eps of the model's own
  CONDITIONAL ENTROPY.

  Algorithm 5.3, Computational Complexity paragraph -- the procedure that is
  actually implemented (and the one the maintained reference implementation
  follows):
      1. compute the conditional entropy                     O(|V|)
      2. SORT WORDS BY THEIR ABSOLUTE DISTANCE FROM H       O(|V| log |V|)
      3. greedily take words from that list until their cumulative probability
         exceeds the threshold tau                          O(|V|)

  THE PAPER'S ERRATUM (attached to Algorithm 5.3) says its *stated* optimization
  problem admits solutions that EXCLUDE the continuations closest to the
  entropy. The greedy procedure above is the correct reading, and it is what
  HuggingFace's `TypicalLogitsWarper` does. Two INDEPENDENT formulations of
  that procedure are implemented below and cross-checked against each other.

THE SEMANTICS PINNED HERE ARE S15..S21 to match the header's comment block
and docs/plans/2026-10-23_typical-p-locally-typical-sampling.md.

Run with:

    /home/stefan/.venv/bin/python tests/typical_p_oracle.py

Prints a C++-pasteable table and exits 0 iff every self-check passes.
"""
import math
import sys

import numpy as np

NEG_INF = float("-inf")
INF = float("inf")


# ---------------------------------------------------------------------------
# The primitive. Max-subtracting softmax, exactly as the C++ file does it.
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# S18: the conditional entropy, in NATS, over the PROBABILITY distribution.
#
# 0 * log 0 == 0, so a zero-probability token contributes NOTHING. A naive
# `-(p * log(p)).sum()` turns a 0 into 0 * -inf == NaN and poisons the whole
# row -- the classic failure this label exists to foreclose.
# ---------------------------------------------------------------------------
def entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    live = p[p > 0.0]
    if live.size == 0:
        return 0.0
    return float(-np.sum(live * np.log(live)))


# ---------------------------------------------------------------------------
# S15: the ordering key. | -log p_i - H |, i.e. the distance between the
# token's NEGATIVE LOG-PROBABILITY and the model's conditional entropy, in nats.
#
# S20: a zero-probability token has an INFINITE distance. It is never kept --
# sorting it first would let a token with no mass at all survive the cut.
# ---------------------------------------------------------------------------
def typical_distances(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    h = entropy(p)
    out = np.empty(p.size, dtype=np.float64)
    for j in range(p.size):
        out[j] = abs(-math.log(p[j]) - h) if p[j] > 0.0 else INF
    return out


# ---------------------------------------------------------------------------
# THE CORRECT ALGORITHM -- the paper's greedy wording.
#
# S16: the sort is ASCENDING by distance: the token CLOSEST to the entropy is
# kept FIRST. Sorting descending keeps the LEAST typical tokens first, which is
# the exact opposite of the algorithm -- and it still produces a plausible
# non-empty set, so a survivor-COUNT test cannot see it.
#
# S17: the cut is the smallest prefix whose cumulative mass REACHES tau (`>=`),
# so the token that crosses the threshold is kept. A `>` here admits one token
# too many.
# ---------------------------------------------------------------------------
def typical_survivors(probs: np.ndarray, tau: float) -> list:
    p = np.asarray(probs, dtype=np.float64)
    d = typical_distances(p)
    order = np.argsort(d, kind="stable")          # ascending, ties first-wins
    live = [False] * p.size
    cum = 0.0
    for i in order:
        live[i] = True
        cum += p[i]
        if cum >= tau:                            # S17: `>=`, not `>`
            break
    return live


# ---------------------------------------------------------------------------
# THE FOILS. Each is a plausible slip that MUST differ from the correct
# algorithm on the pinned fixture, or the corresponding test is vacuous.
#
# A test that cannot fail is worse than no test: this repo has hit that class
# repeatedly (the nucleus key-shape bug made every context look unseen, the
# Witten-Bell OOV class was missing from the unseen set, and the min-p
# `n_finite(T=2) >= n_finite(T=1)` check was an INEQUALITY that held whether or
# not the temperature was honoured at all).
# ---------------------------------------------------------------------------
def typical_survivors_WRONG_descending(probs: np.ndarray, tau: float) -> list:
    """S16's foil: sort DESCENDING by distance -- the LEAST typical first."""
    p = np.asarray(probs, dtype=np.float64)
    d = typical_distances(p)
    order = np.argsort(-d, kind="stable")
    live = [False] * p.size
    cum = 0.0
    for i in order:
        live[i] = True
        cum += p[i]
        if cum >= tau:
            break
    return live


def typical_survivors_WRONG_strict(probs: np.ndarray, tau: float) -> list:
    """S17's foil: the cut is `cum > tau` -- the crossing token is DROPPED."""
    p = np.asarray(probs, dtype=np.float64)
    d = typical_distances(p)
    order = np.argsort(d, kind="stable")
    live = [False] * p.size
    cum = 0.0
    for i in order:
        live[i] = True
        cum += p[i]
        if cum > tau:
            break
    return live


def typical_survivors_WRONG_unnegated(probs: np.ndarray, tau: float) -> list:
    """S15's foil: the distance uses the log-probability WITHOUT negating it.

    `|log p - H|` instead of `|-log p - H|`. Algebraically `|log p + H| ==
    |-log p - H|` is the SAME number, so this particular slip is invisible if
    it is written that way -- the slip that actually happens is DROPPING THE
    NEGATION on the log-probability while keeping the `+ H` form, which is
    `|log p - H|` and is a DIFFERENT number. That is the foil here.
    """
    p = np.asarray(probs, dtype=np.float64)
    h = entropy(p)
    d = np.array([abs(math.log(p[j]) - h) if p[j] > 0.0 else INF
                  for j in range(p.size)], dtype=np.float64)
    order = np.argsort(d, kind="stable")
    live = [False] * p.size
    cum = 0.0
    for i in order:
        live[i] = True
        cum += p[i]
        if cum >= tau:
            break
    return live


# ---------------------------------------------------------------------------
# THE SECOND, INDEPENDENT FORMULATION. This is HuggingFace's
# `TypicalLogitsWarper` transcribed literally, from transformers 5.14.1,
# generation/logits_process.py:
#
#     normalized = torch.nn.functional.log_softmax(scores, dim=-1)
#     p = torch.exp(normalized)
#     ent = -(normalized * p).nansum(-1, keepdim=True)
#
#     shifted_scores = torch.abs((-normalized) - ent)
#     sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
#     sorted_logits = scores.gather(-1, sorted_indices)
#     cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
#
#     last_ind = (cumulative_probs < self.mass).sum(dim=1)
#     last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
#
# The INDEX FORMULATION and the GREEDY FORMULATION are different code paths to
# the same object, and they are cross-checked against each other below. A
# single formulation agreeing with itself proves nothing.
# ---------------------------------------------------------------------------
def typical_survivors_HF_INDEX_FORM(probs: np.ndarray, tau: float) -> list:
    p = np.asarray(probs, dtype=np.float64)
    d = typical_distances(p)
    sorted_idx = np.argsort(d, kind="stable")
    sp = p[sorted_idx]
    cum = np.cumsum(sp)
    last_ind = int((cum < tau).sum())       # HF: (cumulative_probs < mass).sum()
    last_ind = min(last_ind, sp.size - 1)   # HF: clamp_(max=shape-1)
    return [int(i) for i in sorted_idx[:last_ind + 1]]


def survivors_str(live) -> str:
    kept = [i for i, ok in enumerate(live) if ok]
    return "{" + ", ".join(str(i) for i in kept) + "}" if kept else "{} (EMPTY)"


# ---------------------------------------------------------------------------
# THE PINNED FIXTURES.
#
# FIXTURE A -- logits [0.0, -1.0, 0.5, -2.0].
#
#   p = [0.31726532567666477, 0.11671539071300707, 0.5230820908987207,
#        0.0429371927116076]
#   H = 1.0890655963558222
#   d = [0.05895126931313799, 1.058951269313138, 0.441048730686862,
#        2.058951269313138]
#
#   The ARGMAX is token 2, but the token CLOSEST TO THE ENTROPY is token 0.
#   That is the whole point of the algorithm and it is what makes the fixture
#   non-vacuous: a mass cut (top-p) keeps token 2, and the typical cut keeps
#   token 0. A test that only checked a survivor COUNT would pass an
#   implementation that simply did top-p.
#
# FIXTURE B -- logits [0.0, 0.0], i.e. a uniform two-token distribution.
#
#   p = [0.5, 0.5] EXACTLY (0.5 is dyadic), H = log 2 EXACTLY, d = [0, 0].
#
#   This is the S17 boundary fixture. At tau = 0.5 the cumulative mass reaches
#   EXACTLY 0.5, so `>=` keeps ONE token and `>` keeps TWO. The value is
#   exactly representable in binary floating point, which is the lesson the
#   nucleus fixture already learned the hard way: "a fixture whose
#   discriminating value is only correct to within a rounding error does not
#   discriminate anything".
# ---------------------------------------------------------------------------
FIXTURE_A_LOGITS = [0.0, -1.0, 0.5, -2.0]
FIXTURE_B_LOGITS = [0.0, 0.0]


def main() -> int:
    checks = 0
    fails = []

    def check(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(label)
            print(f"  !! FAILED: {label}")

    print("=" * 78)
    print("TYPICAL-P (LOCALLY TYPICAL SAMPLING) ORACLE (numpy float64)")
    print("=" * 78)

    # ---- 1. FIXTURE A: the pinned numbers ---------------------------------
    print("\n--- 1. FIXTURE A: logits [0.0, -1.0, 0.5, -2.0] ---")
    pA = softmax(np.array(FIXTURE_A_LOGITS, dtype=np.float64))
    hA = entropy(pA)
    dA = typical_distances(pA)
    print(f"  p = {[repr(float(x)) for x in pA]}")
    print(f"  H = {repr(hA)}")
    print(f"  d = {[repr(float(x)) for x in dA]}")
    print(f"  argmax = {int(np.argmax(pA))}   "
          f"closest-to-entropy = {int(np.argsort(dA, kind='stable')[0])}")
    check(abs(hA - 1.0890655963558222) < 1e-15, "A: the entropy matches the pinned value")
    check(int(np.argmax(pA)) == 2, "A: the argmax is token 2")
    check(int(np.argsort(dA, kind="stable")[0]) == 0,
          "A: the token closest to the entropy is token 0, NOT the argmax")

    # The distances must be well separated, or a tie-break would decide the
    # result and the fixture would be testing the tie-break, not the algorithm.
    gaps = np.diff(np.sort(dA))
    print(f"  distance gaps = {[repr(float(x)) for x in gaps]}")
    check(float(np.min(gaps)) > 1e-6,
          "A: the distances are separated by >> 1 ULP, so no tie-break decides it")

    # ---- 2. THE SURVIVOR TABLE --------------------------------------------
    print("\n--- 2. SURVIVOR TABLE (fixture A) ---")
    table = {}
    for tau in (0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        live = typical_survivors(pA, tau)
        kept = [i for i, ok in enumerate(live) if ok]
        mass = float(sum(pA[i] for i in kept))
        table[tau] = (live, kept, mass)
        print(f"  tau={tau:<5} -> {survivors_str(live):<14} n={len(kept)}  "
              f"kept_mass={mass!r}")
        check(len(kept) >= 1, f"A: tau={tau} keeps at least one token")
        check(mass >= tau - 1e-12, f"A: tau={tau} keeps at least tau of mass")
        # THE CROSS-CHECK: the greedy formulation and HF's index formulation
        # are different code paths to the same object.
        hf = typical_survivors_HF_INDEX_FORM(pA, tau)
        check(sorted(kept) == sorted(hf),
              f"A: tau={tau} greedy and HF-index formulations AGREE")

    # ---- 3. THE FOILS MUST DIFFER (else the test is vacuous) --------------
    print("\n--- 3. THE FOILS (each MUST differ on fixture A) ---")
    for tau in (0.2, 0.3, 0.5):
        good = table[tau][0]
        desc = typical_survivors_WRONG_descending(pA, tau)
        strict = typical_survivors_WRONG_strict(pA, tau)
        unneg = typical_survivors_WRONG_unnegated(pA, tau)
        print(f"  tau={tau}:")
        print(f"      correct      = {survivors_str(good)}")
        print(f"      S16 foil (descending) = {survivors_str(desc)}")
        print(f"      S17 foil (strict >)   = {survivors_str(strict)}")
        print(f"      S15 foil (un-negated) = {survivors_str(unneg)}")
        check(good != desc, f"S16: tau={tau} the DESCENDING order differs")
        check(good != unneg, f"S15: tau={tau} the UN-NEGATED distance differs")

    # S17's foil is checked on FIXTURE B, where the boundary is EXACT.
    # On fixture A no cumulative mass lands exactly on tau, so `>` and `>=`
    # agree there -- a fixture that cannot separate them is a fixture that
    # cannot test them.
    print("\n  S17's foil needs an EXACT boundary -- FIXTURE B (dyadic):")
    pB = softmax(np.array(FIXTURE_B_LOGITS, dtype=np.float64))
    hB = entropy(pB)
    print(f"      p = {[repr(float(x)) for x in pB]}")
    print(f"      H = {repr(hB)}")
    check(pB[0] == 0.5 and pB[1] == 0.5,
          "B: the two-token uniform distribution is EXACTLY [0.5, 0.5]")
    check(hB == 0.6931471805599453, "B: the entropy is EXACTLY log 2")
    goodB = typical_survivors(pB, 0.5)
    strictB = typical_survivors_WRONG_strict(pB, 0.5)
    print(f"      tau=0.5 correct (>=) = {survivors_str(goodB)}")
    print(f"      tau=0.5 foil   (>)   = {survivors_str(strictB)}")
    check(goodB != strictB,
          "S17: at tau=0.5 on the dyadic fixture, `>=` and `>` DISAGREE")
    check([i for i, ok in enumerate(goodB) if ok] == [0],
          "S17: `>=` keeps exactly ONE token at the exact boundary")
    check([i for i, ok in enumerate(strictB) if ok] == [0, 1],
          "S17: the `>` foil keeps TWO -- it drops the crossing token")
    # And the two formulations must agree on B too.
    check(sorted([i for i, ok in enumerate(goodB) if ok])
          == sorted(typical_survivors_HF_INDEX_FORM(pB, 0.5)),
          "B: greedy and HF-index formulations AGREE at the exact boundary")

    # ---- 4. IT IS NOT A MASS CUT ------------------------------------------
    # The load-bearing property: on fixture A the typical set and the top-p
    # set DISAGREE. If they agreed everywhere, typical-p would be top-p.
    print("\n--- 4. TYPICAL-P IS NOT TOP-P (the load-bearing property) ---")
    for tau in (0.2, 0.3, 0.5):
        op = np.argsort(-pA, kind="stable")
        lv = [False] * pA.size
        cum = 0.0
        for i in op:
            lv[i] = True
            cum += pA[i]
            if cum >= tau:
                break
        topp = [i for i, ok in enumerate(lv) if ok]
        typ = table[tau][1]
        print(f"  tau={tau}: typical={typ}  top-p={topp}  "
              f"{'DIFFERENT' if typ != topp else '*** SAME -- VACUOUS ***'}")
        check(typ != topp, f"tau={tau}: the typical set differs from the top-p set")

    # ---- 5. S19: THE TAU BOUNDARIES ---------------------------------------
    # MEASURED, and the plan's first draft was WRONG here: `tau = 0` keeps
    # exactly ONE token (the closest to the entropy), NOT "everything" and NOT
    # "nothing". The greedy loop always takes at least one token before it
    # tests the threshold, and HF's index formulation agrees.
    print("\n--- 5. S19: THE TAU BOUNDARIES ---")
    for tau in (0.0, 1.0):
        live = typical_survivors(pA, tau)
        kept = [i for i, ok in enumerate(live) if ok]
        mass = float(sum(pA[i] for i in kept))
        print(f"  tau={tau}: kept={kept}  n={len(kept)}  kept_mass={mass!r}")
        check(len(kept) >= 1, f"S19: tau={tau} keeps at least one token")
    check(len([i for i, ok in enumerate(typical_survivors(pA, 0.0)) if ok]) == 1,
          "S19: tau=0 keeps EXACTLY ONE token (the closest to the entropy)")
    check(len([i for i, ok in enumerate(typical_survivors(pA, 1.0)) if ok]) == 4,
          "S19: tau=1 keeps EVERY token")
    # HF only applies the warper when typical_p < 1.0, so tau == 1 is a no-op
    # in the reference implementation too.
    check(len([i for i, ok in enumerate(typical_survivors(pA, 1.0)) if ok])
          == pA.size,
          "S19: tau=1 is a no-op, consistent with HF skipping typical_p == 1.0")

    # ---- 6. S20: A ZERO-PROBABILITY TOKEN IS NEVER KEPT --------------------
    print("\n--- 6. S20: A ZERO-PROBABILITY TOKEN (logit -inf) ---")
    pC = softmax(np.array([0.0, NEG_INF, -1.0], dtype=np.float64))
    hC = entropy(pC)
    dC = typical_distances(pC)
    print(f"  p = {[repr(float(x)) for x in pC]}")
    print(f"  H = {repr(hC)}")
    print(f"  d = {[repr(float(x)) for x in dC]}")
    check(pC[1] == 0.0, "S20: the -inf logit has probability exactly 0")
    check(dC[1] == INF, "S20: the zero-probability token has an INFINITE distance")
    for tau in (0.5, 0.9, 1.0):
        kept = [i for i, ok in enumerate(typical_survivors(pC, tau)) if ok]
        print(f"  tau={tau}: kept={kept}  (token 1 must be absent)")
        check(1 not in kept, f"S20: tau={tau} the zero-probability token is NOT kept")
    # S18: the entropy must be FINITE even with a zero-probability token.
    check(math.isfinite(hC), "S18: the entropy is FINITE with a zero-probability token")
    check(not math.isnan(hC), "S18: the entropy is NOT NaN with a zero-probability token")

    # ---- 7. S18: THE ENTROPY'S SIGN ---------------------------------------
    # The entropy is POSITIVE. A sign flip is a plausible slip (the S25
    # mutation) and a magnitude-only check cannot see it, so the SIGN is
    # asserted.
    print("\n--- 7. S18: THE ENTROPY'S SIGN ---")
    check(hA > 0.0, "S18: the entropy is POSITIVE")
    check(abs(hB - math.log(2.0)) < 1e-15,
          "S18: the uniform two-token entropy is log 2")
    # And the entropy of a POINT MASS is 0 -- the lower bound, and the reason
    # the "distance from the entropy" is a distance from the information
    # content of a distribution that knows exactly one thing.
    pD = softmax(np.array([0.0, NEG_INF, NEG_INF], dtype=np.float64))
    print(f"  point mass p = {[repr(float(x)) for x in pD]}  H = {repr(entropy(pD))}")
    check(entropy(pD) == 0.0, "S18: a point-mass distribution has entropy exactly 0")
    dD = typical_distances(pD)
    print(f"  point mass d = {[repr(float(x)) for x in dD]}")
    check(dD[0] == 0.0, "S18: the only possible token has distance 0 from H=0")

    # ---- 8. THE TEMPERATURE INTERACTION -----------------------------------
    print("\n--- 8. THE TEMPERATURE INTERACTION (entropy of the ADJUSTED dist) ---")
    for T in (0.5, 1.0, 2.0, 4.0):
        pT = softmax(np.array(FIXTURE_A_LOGITS, dtype=np.float64) / T)
        hT = entropy(pT)
        live = typical_survivors(pT, 0.5)
        kept = [i for i, ok in enumerate(live) if ok]
        print(f"  T={T:<4} H={hT!r:<20} survivors={kept}")
        check(math.isfinite(hT), f"T={T}: the adjusted entropy is finite")
    check(len([i for i, ok in enumerate(typical_survivors(
        softmax(np.array(FIXTURE_A_LOGITS, dtype=np.float64) / 4.0), 0.5)) if ok]) == 2,
        "temperature: T=4 on fixture A keeps TWO tokens at tau=0.5")

    # ---- 9. THE CHAIN ORDER MATTERS ---------------------------------------
    # A chain folding RIGHT TO LEFT would apply the two cuts in the other
    # order, and on this fixture the two orders give DIFFERENT sets. This is
    # the repo's S8 contract, extended.
    print("\n--- 9. THE CHAIN ORDER (S8 extended) ---")
    for tau, pthresh in ((0.3, 0.5), (0.5, 0.7)):
        typ = [i for i, ok in enumerate(typical_survivors(pA, tau)) if ok]
        op = np.argsort(-pA, kind="stable")
        lv = [False] * pA.size
        cum = 0.0
        for i in op:
            lv[i] = True
            cum += pA[i]
            if cum >= pthresh:
                break
        topp = [i for i, ok in enumerate(lv) if ok]

        masked_after_typ = [FIXTURE_A_LOGITS[j] if j in typ else NEG_INF
                            for j in range(len(FIXTURE_A_LOGITS))]
        after_typ = [i for i, ok in enumerate(typical_survivors(
            softmax(np.array(masked_after_typ, dtype=np.float64)), pthresh)) if ok]

        masked_after_topp = [FIXTURE_A_LOGITS[j] if j in topp else NEG_INF
                             for j in range(len(FIXTURE_A_LOGITS))]
        after_topp = [i for i, ok in enumerate(typical_survivors(
            softmax(np.array(masked_after_topp, dtype=np.float64)), tau)) if ok]

        print(f"  tau={tau} p={pthresh}: typical={typ} top-p={topp}")
        print(f"      typical->top-p = {after_typ}")
        print(f"      top-p->typical = {after_topp}")
        check(after_typ != after_topp,
              f"chain order MATTERS at tau={tau} p={pthresh} (the two differ)")

    # ORACLE DEFECT FOUND BY RUNNING IT (the first draft of this section also
    # asserted "the raw cuts differ" at EVERY (tau, p) pair, and it FAILED at
    # tau=0.5 p=0.7 -- because the typical set and the top-p set COINCIDE
    # there: both are {0, 2}. The assertion was WRONG, not the algorithm.
    #
    # The reason it is worth recording: the two orders of the chain still
    # DIFFER at that point ({0,2} vs {2}), because masking changes the
    # distribution whose ENTROPY the second cut is measured against. So the
    # chain-order property does NOT require the raw cuts to differ -- asking
    # for that was conflating two different claims. The "typical is not top-p"
    # property is asserted separately, in section 4, on the taus where the
    # two genuinely disagree.
    print("\n  The chain order can differ EVEN WHERE THE RAW CUTS COINCIDE:")
    tau, pthresh = 0.5, 0.7
    typ = [i for i, ok in enumerate(typical_survivors(pA, tau)) if ok]
    op = np.argsort(-pA, kind="stable")
    lv = [False] * pA.size
    cum = 0.0
    for i in op:
        lv[i] = True
        cum += pA[i]
        if cum >= pthresh:
            break
    topp = [i for i, ok in enumerate(lv) if ok]
    print(f"      tau={tau} p={pthresh}: typical={typ} top-p={topp}  "
          f"{'COINCIDE' if typ == topp else 'differ'}")
    check(typ == topp,
          "at tau=0.5 p=0.7 the raw cuts COINCIDE -- so the chain-order check "
          "must not depend on them differing")

    # ---- 10. THE TIES -----------------------------------------------------
    # Ties in the DISTANCE are broken first-wins on the original index, the
    # repo's S12 convention. Fixture B is a total tie (both distances are 0).
    print("\n--- 10. THE TIES (S12 convention) ---")
    goodB = typical_survivors(pB, 0.5)
    check([i for i, ok in enumerate(goodB) if ok] == [0],
          "ties: on a total tie the LOWER original index wins")
    pTie = softmax(np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float64))
    print(f"  p = {[repr(float(x)) for x in pTie]}  H = {repr(entropy(pTie))}")
    dTie = typical_distances(pTie)
    print(f"  d = {[repr(float(x)) for x in dTie]}")
    tie_kept = [i for i, ok in enumerate(typical_survivors(pTie, 0.3)) if ok]
    print(f"  tau=0.3 -> kept={tie_kept}")
    check(tie_kept == [0], "ties: the lower original index wins among tied tokens")

    # ---- SUMMARY ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"ORACLE SELF-CHECKS: {checks} run, {len(fails)} failed")
    for f in fails:
        print(f"  FAILED: {f}")
    print("=" * 78)

    if fails:
        return 1
    print("ALL ORACLE SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
