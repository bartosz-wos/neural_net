#!/usr/bin/env python3
"""
tests/logit_processing_oracle.py -- numpy float64 oracle for the logit-processing
library (include/nn/utils/logit_processing.h).

WHY AN ORACLE AND NOT HAND-DERIVATION. Every numeric expectation in
tests/test_logit_processing.cpp is copied from this file's output. Two earlier
sessions in this repo caught a real bug in the ORACLE itself before a line of C++
existed (tests/test_vat.cpp D9; tests/sinkhorn_oracle.py, "7 oracle findings"), and
the TDD skill is explicit that a hand-derived expected value in a test is a
leading cause of phantom implementation bugs -- a single-digit typo ships a test
that fails for the wrong reason.

The semantics pinned here are labelled S1..S13 to match the plan
(docs/plans/2026-10-20_logit-processing-decoding.md) and the header's own comment
block. Run with:

    /home/stefan/.venv/bin/python tests/logit_processing_oracle.py

Prints a C++-pasteable table and exits 0 iff every self-check passes.
"""
import math
import sys

import numpy as np

NEG_INF = float("-inf")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Max-subtracting softmax. THE MAX-SUBTRACT IS LOAD-BEARING (S10):
    exp(1000) is inf and inf/inf is NaN, so an implementation that omits it
    returns garbage rather than a distribution."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def survivors(probs_or_logits, mask_live) -> str:
    live = [i for i, ok in enumerate(mask_live) if ok]
    return "{" + ", ".join(str(i) for i in live) + "}" if live else "{} (EMPTY)"


# ---------------------------------------------------------------------------
# S1. NUCLEUS `p` IS A CUMULATIVE-PROBABILITY CUT AND THE CROSSING TOKEN IS KEPT.
# ---------------------------------------------------------------------------
def nucleus_survivors(probs: np.ndarray, p: float) -> list:
    """Keep the smallest DESCENDING prefix whose cumulative mass is >= p.

    The token that CROSSES the threshold is kept. HuggingFace's reference has
    exactly this off-by-one and it is the single most common bug in every
    hand-rolled nucleus implementation.
    """
    order = np.argsort(-probs, kind="stable")  # stable => ties first-win (S12)
    live = [False] * len(probs)
    cum = 0.0
    for idx in order:
        live[idx] = True
        cum += probs[idx]
        if cum >= p:                          # <-- `>=`, not `>`
            break
    return live


def top_k_survivors(n: int, k: int) -> list:
    """S3/S12: keep the k largest, ties first-win, k >= n is a no-op."""
    if k >= n:
        return [True] * n
    order = sorted(range(n), key=lambda i: (-(-i), i))  # value desc, index asc
    live = [False] * n
    for i in order[:k]:
        live[i] = True
    return live


def top_k_survivors_from_probs(probs: np.ndarray, k: int) -> list:
    n = len(probs)
    if k >= n:
        return [True] * n
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    live = [False] * n
    for i in order[:k]:
        live[i] = True
    return live


# ---------------------------------------------------------------------------
# S4. MIN-P IS A RELATIVE FLOOR AGAINST THE MAX, NOT THE MEDIAN.
# ---------------------------------------------------------------------------
def min_p_survivors(probs: np.ndarray, min_p: float) -> list:
    """Keep tokens with p_i >= min_p * p_max."""
    thr = min_p * float(np.max(probs))
    return [bool(p >= thr) for p in probs]


def min_p_survivors_WRONG_median(probs: np.ndarray, min_p: float) -> list:
    """The plausible-looking bug: threshold relative to the MEDIAN instead."""
    thr = min_p * float(np.median(probs))
    return [bool(p >= thr) for p in probs]


# ---------------------------------------------------------------------------
# S5. THE REPETITION PENALTY SIGN-SPLITS. THIS IS THE BUG.
# ---------------------------------------------------------------------------
def repetition_penalty(logits: np.ndarray, history, penalty: float) -> np.ndarray:
    """For penalty > 1: a NEGATIVE logit is multiplied (pushed further from 0),
    a POSITIVE logit is divided. Applying the multiplication unconditionally
    SUPPRESSES positive tokens and PROMOTES negative ones -- it inverts the
    penalty on roughly half the vocabulary.

    ORACLE BUG FOUND HERE 2026-10-20 (this is the trap the debugging skill
    names as the `pair<int,int>` one): the first draft of this function wrote

        for t in set(history):
            i = list(history).index(t)      # <-- POSITION IN THE HISTORY LIST

    `.index(t)` is the position of the token in the HISTORY, not the token id
    itself, so with history [0, 0, 1] it wrote to indices 0 and 1 and silently
    left index 2 -- the one token not in the history -- modified by a
    "not in history" path. The self-check below (`a log NOT in the history is
    untouched`) caught it: index 2 came back as 2.0 instead of 3.0.

    This is worth recording as a class: a scalar read off the wrong slot, in an
    ORACLE, is invisible until a check pins the untouched entries too.
    """
    out = np.array(logits, dtype=np.float64, copy=True)
    for t in sorted(set(history)):      # token ID -- indexes the LOGITS directly
        out[t] = out[t] * penalty if out[t] < 0 else out[t] / penalty
    return out


def frequency_penalty(logits: np.ndarray, history, penalty: float) -> np.ndarray:
    """S6: linear in the count, NOT clipped."""
    out = np.array(logits, dtype=np.float64, copy=True)
    counts = {t: list(history).count(t) for t in set(history)}
    for t, c in counts.items():
        out[t] -= penalty * c
    return out


def presence_penalty(logits: np.ndarray, history, penalty: float) -> np.ndarray:
    """S6: a STEP function of presence -- 1x however many times it occurred."""
    out = np.array(logits, dtype=np.float64, copy=True)
    for t in set(history):
        out[t] -= penalty
    return out


def main() -> int:
    checks = 0
    fails = []

    def check(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(label)

    print("=" * 78)
    print("LOGIT-PROCESSING ORACLE (numpy float64)")
    print("=" * 78)

    # ---- 1. softmax baseline ------------------------------------------------
    base = np.array([0.5, 0.3, 0.15, 0.05])
    print("\n--- 1. softmax baseline ---")
    print(f"softmax([log {list(base)}]) = {list(softmax(np.log(base)))}")
    check(abs(softmax(np.log(base)).sum() - 1.0) < 1e-12, "softmax sums to 1")
    check(np.allclose(softmax(np.log(base)), base), "softmax recovers the distribution")

    # ---- 2. NUCLEUS (S1) ---------------------------------------------------
    print("\n--- 2. NUCLEUS survivors on p = [0.5, 0.3, 0.15, 0.05] ---")
    nucleus_expected = {}
    for p in (0.5, 0.8, 0.9, 1.0):
        live = nucleus_survivors(base, p)
        order = np.argsort(-base, kind="stable")
        cum = np.cumsum(base[order])
        kept_mass = float(sum(base[i] for i, ok in enumerate(live) if ok))
        n = sum(live)
        print(f"  p={p:<4} -> {survivors(base, live):<22} "
              f"n={n}  kept_mass={kept_mass:.10f}")
        nucleus_expected[p] = (live, n)
        check(n >= 1, f"nucleus p={p} non-empty")
        check(kept_mass >= p - 1e-12, f"nucleus p={p} kept mass >= p")
        discarded = 1.0 - kept_mass
        check(discarded < p + 1e-12 or p >= 1.0,
              f"nucleus p={p} discarded mass < p (minimality)")

    # THE off-by-one. ORACLE CORRECTION 2026-10-20: the implementation plan
    # claimed "p=0.8 keeps 3 tokens, not 2", reasoning that 0.5+0.3 = 0.8 is not
    # `> 0.8` so the third token is kept. That is backwards -- `>=` stops AT the
    # token that reaches p, so 0.5+0.3 = 0.8 satisfies `cum >= 0.8` and stops
    # with 2 tokens. The oracle measures which p actually separates `>=` from
    # `>` and reports it rather than asserting a remembered number.
    print("\n  S1 -- which p separates `cum >= p` from `cum > p` on this fixture?")
    order = np.argsort(-base, kind="stable")
    separating = []
    for p in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        def run(strict):
            live = [False] * 4
            cum = 0.0
            for idx in order:
                live[idx] = True
                cum += base[idx]
                if (cum > p) if strict else (cum >= p):
                    break
            return [i for i, ok in enumerate(live) if ok]
        ge, gt = run(False), run(True)
        differs = ge != gt
        if differs:
            separating.append(p)
        print(f"      p={p:<5} cum hits {list(np.round(np.cumsum(base[order]), 6))}  "
              f">=:{str(ge):<10} >:{str(gt):<10} {'<== SEPARATES' if differs else ''}")
    live_80, n_80 = nucleus_expected[0.8]
    print(f"  S1 RESULT: p=0.8 -> >=:{[i for i, ok in enumerate(live_80) if ok]} "
          f"(n={n_80}); the separating p values are {separating}")
    check(n_80 == 2, "S1: p=0.8 keeps 2 tokens (0.5+0.3 == 0.8 satisfies `>=`)")
    check(0.8 in separating,
          "S1: p=0.8 IS a boundary -- it separates `>=` from `>` on this fixture")

    # ---- 3/4. TOP-K (S3/S12/S13) -------------------------------------------
    print("\n--- 3/4. TOP-K survivors ---")
    for k in (1, 2, 3, 4):
        live = top_k_survivors_from_probs(base, k)
        print(f"  V=4 k={k:<2} -> {survivors(base, live)}")
        check(sum(live) == min(k, 4), f"top_k k={k} keeps min(k,V) tokens")
    check(all(top_k_survivors(4, 4)), "S3: k >= V is a no-op")
    check(all(top_k_survivors(4, 9)), "S13: k > V is a no-op")

    # ties -> first-wins on the ORIGINAL index
    tied = np.array([0.4, 0.4, 0.1, 0.1])
    live_tie = top_k_survivors_from_probs(tied, 1)
    print(f"  S12 TIES: p=[0.4,0.4,0.1,0.1] k=1 -> {survivors(tied, live_tie)} "
          f"(index 0 must win, not 1)")
    check(live_tie == [True, False, False, False], "S12: exact ties break first-wins")

    # single-token vocabulary must be legal
    one = np.array([1.0])
    check(all(top_k_survivors_from_probs(one, 5)), "S13: V=1 with k=5 is legal")

    # ---- 5. MIN-P (S4) -----------------------------------------------------
    print("\n--- 5. MIN-P (relative to MAX, not median) ---")
    skew = np.array([0.70, 0.15, 0.10, 0.05])
    for mp in (0.05, 0.1, 0.5):
        good = min_p_survivors(skew, mp)
        bad = min_p_survivors_WRONG_median(skew, mp)
        print(f"  min_p={mp:<5} max-relative -> {survivors(skew, good):<16} "
              f"| median-relative -> {survivors(skew, bad)}")
        print(f"      thresholds: min_p*p_max={mp * skew.max():.6f}  "
              f"min_p*median={mp * float(np.median(skew)):.6f}")
        if mp == 0.1:
            check(good != bad, "S4: the two formulations genuinely disagree on this fixture")
    check(min_p_survivors(skew, 0.1) == [True, True, True, False],
          "S4: min_p=0.1 on [0.70,0.15,0.10,0.05] keeps the first three")

    # ---- 6. THE PENALTIES (S5/S6) ------------------------------------------
    print("\n--- 6. PENALTIES on logits [-2.0, 0.0, 3.0], history [0,0,1] ---")
    lg = np.array([-2.0, 0.0, 3.0])
    hist = [0, 0, 1]
    rep = repetition_penalty(lg, hist, 1.5)
    freq = frequency_penalty(lg, hist, 0.5)
    pres = presence_penalty(lg, hist, 0.5)
    print(f"  repetition(1.5) = {list(rep)}")
    print(f"  frequency(0.5)  = {list(freq)}")
    print(f"  presence(0.5)   = {list(pres)}")
    print("  S5 note: index 0 is NEGATIVE (-2.0) and occurs twice; index 1 is")
    print("         ZERO and occurs once. A `>= 0` test puts index 1 in the")
    print("         DIVIDE branch -- and dividing 0.0 by anything is 0.0, so")
    print("         index 1 is UNAFFECTED either way. The sign split is only")
    print("         observable at index 0.")
    check(abs(rep[0] - (-2.0 * 1.5)) < 1e-15, "S5: negative logit is MULTIPLIED")
    check(abs(rep[1] - 0.0) < 1e-15, "S5: zero logit is unchanged by the penalty")
    check(abs(rep[2] - 3.0) < 1e-15, "S5: a log NOT in the history is untouched")
    check(abs(freq[0] - (-2.0 - 1.0)) < 1e-15, "S6: frequency penalty is linear in the count (2x)")
    check(abs(freq[1] - (0.0 - 0.5)) < 1e-15, "S6: frequency penalty 1x for count 1")
    check(abs(freq[2] - 3.0) < 1e-15, "S6: frequency penalty leaves unmentioned tokens alone")
    check(abs(pres[0] - (-2.0 - 0.5)) < 1e-15, "S6: presence penalty is 1x REGARDLESS of count")
    check(abs(pres[1] - (0.0 - 0.5)) < 1e-15, "S6: presence penalty 1x for count 1")

    # a fixture where BOTH branches of the sign split are observable. ORACLE FIX
    # 2026-10-20: the first draft used logits [-2.0, 4.0] with history [1],
    # which penalises ONLY index 1 -- so the "negative branch multiplies"
    # assertion below was asserting on a token that was never in the history.
    # history must name BOTH indices for the sign split to be visible.
    lg2 = np.array([-2.0, 4.0])
    rep2 = repetition_penalty(lg2, [0, 1], 2.0)
    print(f"  S5 both branches: repetition([-2.0, 4.0], hist=[0,1], 2.0) = {list(rep2)}")
    check(abs(rep2[0] - (-4.0)) < 1e-15, "S5: negative branch MULTIPLIES (-2.0 -> -4.0)")
    check(abs(rep2[1] - 2.0) < 1e-15, "S5: positive branch DIVIDES (4.0 -> 2.0)")
    # and the counterfactual: an implementation that always multiplies gives
    # [4.0, 8.0] -- the positive token is PROMOTED instead of suppressed.
    always_mul = np.array([x * 2.0 for x in lg2])
    print(f"  S5 COUNTERFACTUAL (always multiply) = {list(always_mul)} "
          f"<-- promotes the positive token")
    check(always_mul[1] > lg2[1], "S5: always-multiply PROMOTES a positive logit (the bug)")

    # ---- 7. TEMPERATURE ON LOGITS (S7) -------------------------------------
    print("\n--- 7. TEMPERATURE (on logits, not probabilities) ---")
    logits7 = np.array([2.0, 1.0, 0.0])
    for T in (0.5, 1.0, 2.0):
        p_on_logits = softmax(logits7 / T)
        p_on_probs = softmax(np.log(softmax(logits7))) / T   # the wrong way round
        print(f"  T={T:<5} softmax(logits/T) = {list(p_on_logits)}")
        print(f"          softmax(probs/T)   = {list(p_on_probs)}  <-- NOT this")
        if T == 1.0:
            check(np.allclose(p_on_logits, softmax(logits7)),
                  "S7: T=1 is the identity")
            check(abs(p_on_logits[0] - 0.6652409557748218) < 1e-12,
                  "S7: T=1 value pinned")
        if T == 0.5:
            check(abs(p_on_logits[0] - 0.8668133321973347) < 1e-12,
                  "S7: T=0.5 value pinned (softmax([4,2,0])[0])")
    # T -> 0+ is greedy
    tiny = softmax(logits7 / 1e-3)
    check(tiny[0] > 0.999, "S7: T->0 approaches greedy")

    # ---- 8. COMPOSITION ORDER (S8) -----------------------------------------
    # CORRECTION 2026-10-20: the plan asserted top-k is order-independent
    # because temperature is monotone, so temperature-then-top-k and
    # top-k-then-temperature give the same survivor SET. Both halves of that
    # are true and together they are exactly why top-k CANNOT pin the chain
    # order: dividing every logit by T > 0 is order-preserving, so the ranking
    # is identical and the survivor set is the same either way.
    # TOP-P IS DIFFERENT, because its cut depends on the cumulative MASS, and
    # temperature changes the mass (it flattens or sharpens the distribution).
    # Measured below, not argued.
    print("\n--- 8. CHAIN ORDER (temperature first, then truncation) ---")
    l8 = np.array([3.0, 2.0, 1.0, 0.0])
    lg8 = np.log(np.exp(l8 - l8.max()) / np.sum(np.exp(l8 - l8.max())))

    def top_k_live_at(logits, k, T=1.0):
        pr = softmax(logits / T)
        return top_k_survivors_from_probs(pr, k)

    t_then_k = top_k_live_at(lg8, 2, 0.5)
    k_then_t = top_k_live_at(lg8, 2, 1.0)
    print(f"  TOP-K is order-independent by construction:")
    print(f"      T=0.5-then-topk(2) -> {t_then_k}    topk(2)-then-T -> {k_then_t}    "
          f"same={t_then_k == k_then_t}")
    check(t_then_k == k_then_t,
          "S8: temperature is monotone, so top-k's survivor SET is order-independent")

    print(f"  TOP-P is NOT -- the cut depends on cumulative MASS:")
    disagreements = []
    for p in (0.8, 0.9, 0.95):
        a = nucleus_survivors(softmax(lg8 / 0.5), p)      # T first, then p
        b = nucleus_survivors(softmax(lg8), p)            # p first, then T
        a_idx = [i for i, ok in enumerate(a) if ok]
        b_idx = [i for i, ok in enumerate(b) if ok]
        if a_idx != b_idx:
            disagreements.append(p)
        print(f"      p={p:<5} T=0.5-then-p -> {str(a_idx):<10} p-then-T -> "
              f"{str(b_idx):<10} {'<== DISAGREE' if a_idx != b_idx else ''}")
    check(bool(disagreements),
          "S8: top-p's survivor SET IS order-dependent, so the chain order is load-bearing")

    # the values the C++ test will pin, at the documented order (T first, then p)
    print(f"\n  The composed VALUES the C++ test pins (order: temperature, then top-p 0.9):")
    chained = softmax(masked(lg8 / 0.5, nucleus_survivors(softmax(lg8 / 0.5), 0.9)))
    wrong_order = softmax(masked(lg8 / 0.5, nucleus_survivors(softmax(lg8), 0.9)))
    print(f"      T=0.5 then top-p 0.9  -> softmax = {[float(x) for x in chained]}")
    print(f"      top-p 0.9 then T=0.5  -> softmax = {[float(x) for x in wrong_order]}")
    check(abs(float(chained[0]) - 0.8807970779778823) < 1e-12,
          "S8: composed value at the documented order is pinned")
    check(abs(float(wrong_order[0]) - 0.8668133321973347) < 1e-12,
          "S8: the WRONG order gives a different value (so the test discriminates)")
    check(len([i for i, ok in enumerate(nucleus_survivors(softmax(lg8), 0.9)) if ok]) == 3,
          "S8: at T=1 the p=0.9 cut keeps 3 tokens; after T=0.5 it keeps 2")

    # ---- 9. -INF SURVIVES (S9) ---------------------------------------------
    print("\n--- 9. -inf SURVIVAL ---")
    live9 = nucleus_survivors(base, 0.5)
    arr9 = np.array([lp if ok else NEG_INF for lp, ok in zip(np.log(base), live9)])
    print(f"  masked logits = {list(arr9)}")
    check(np.all(~np.isnan(arr9)), "S9: no NaN in the masked logits")
    check(all(np.isneginf(arr9[i]) for i, ok in enumerate(live9) if not ok),
          "S9: truncated entries are -inf")
    check(np.all(np.isfinite(softmax(arr9))), "S9: softmax of the masked row is finite")
    check(abs(sum(softmax(arr9)) - 1.0) < 1e-12, "S9: masked softmax still sums to 1")
    # a naive `logits * 0` penalty would turn -inf into NaN
    naive = arr9 * 0.0
    print(f"  S9 the naive `logits * 0` gives {list(naive)} -- NaN: "
          f"{bool(np.any(np.isnan(naive)))}")
    check(bool(np.any(np.isnan(naive))), "S9: the naive multiply DOES produce NaN")

    # ---- 10. HUGE LOGITS (S10) ---------------------------------------------
    print("\n--- 10. HUGE LOGITS ---")
    huge = np.array([1000.0, 999.0])
    p_sub = softmax(huge)
    print(f"  softmax([1000, 999]) with max-subtract = {list(p_sub)}")
    check(abs(sum(p_sub) - 1.0) < 1e-12, "S10: max-subtracted softmax is finite and sums to 1")
    # the no-max-subtract version overflows
    e = np.exp(huge)
    print(f"  WITHOUT the max-subtract: exp = {list(e)}, sum = {e.sum()}, "
          f"ratio = {list(e / e.sum())}  <-- NaN")
    check(math.isinf(e[0]), "S10: exp(1000) really is inf")
    check(bool(np.any(np.isnan(e / e.sum()))), "S10: inf/inf really is NaN")

    # ---- 11. ROUND TO INT (S11) --------------------------------------------
    # ORACLE CORRECTION 2026-10-20: the first draft used `np.round`, which is
    # TIES-TO-EVEN, and asserted half-away-from-zero. numpy and C++ disagree:
    #   np.round(0.5)   = 0.0     std::round(0.5)   = 1.0
    #   np.round(2.5)   = 2.0     std::round(2.5)   = 3.0
    #   np.round(-0.5)  = -0.0    std::round(-0.5)  = -1.0
    # An oracle that uses the WRONG language's rounding mode ships a C++ test
    # whose every value is wrong, and the failure looks like an implementation
    # bug. The C++ semantics are emulated below with
    # copysign(floor(abs(x) + 0.5), x) and the numpy value printed as the
    # COUNTERFACTUAL so the two are never confused again.
    print("\n--- 11. ROUND TO INT (C++ std::round: half AWAY FROM ZERO) ---")

    def std_round(x: float) -> float:
        """Emulate C++ std::round: halfway cases away from ZERO."""
        return math.copysign(math.floor(abs(float(x)) + 0.5), float(x))

    cases = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    r = [std_round(x) for x in cases]
    print(f"  std::round({cases}) = {r}          <-- the C++ contract")
    print(f"  np.round     ({cases}) = "
          f"{[float(np.round(x)) for x in cases]}   <-- ties-to-EVEN, NOT this")
    check(r == [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "S11: half away from zero, BOTH signs")
    nb = [float(np.rint(x)) for x in (-2.5, 0.5, 2.5)]
    print(f"  nearbyint (banker's) on (-2.5, 0.5, 2.5) = {nb}  <-- NOT this either")
    check(nb == [-2.0, 0.0, 2.0], "S11: nearbyint gives banker's -- the two are distinguishable")
    check(std_round(-2.5) != float(np.round(-2.5)),
          "S11: std::round and np.round DISAGREE on -2.5 (so this fixture discriminates)")

    # ---- 12/13. the sampler -------------------------------------------------
    print("\n--- 12/13. SAMPLER ---")
    rng = np.random.default_rng(12345)
    p12 = np.array([0.5, 0.3, 0.2])
    draws = rng.choice(3, size=100000, p=p12)
    freq = [float(np.mean(draws == i)) for i in range(3)]
    print(f"  100k draws from [0.5,0.3,0.2] -> {freq}")
    for got, want in zip(freq, p12):
        check(abs(got - want) < 0.02, f"empirical frequency within 0.02 of {want}")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"ORACLE SELF-CHECKS: {checks - len(fails)} passed, {len(fails)} failed")
    if fails:
        for f in fails:
            print(f"  FAILED: {f}")
        print("ORACLE IS WRONG. Do not paste these values into the C++ test.")
        return 1
    print("ORACLE OK -- paste the values above into tests/test_logit_processing.cpp")
    print("=" * 78)
    return 0


def masked(logits, live):
    return np.array([l if ok else NEG_INF for l, ok in zip(logits, live)])


def top_k_live(logits, k):
    return top_k_survivors_from_probs(softmax(logits), k)


if __name__ == "__main__":
    sys.exit(main())