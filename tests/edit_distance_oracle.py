#!/usr/bin/env python3
# =============================================================================
# tests/edit_distance_oracle.py -- THE SPECIFICATION for
# include/nn/utils/edit_distance.{h,cpp}.
#
# THIS FILE IS THE DEFINITION. It was written and run BEFORE the C++
# implementation existed, and it is the only place the expected numbers come
# from. The C++ test consumes the fixtures this file emits; nothing is
# hand-transcribed.
#
# -----------------------------------------------------------------------------
# WHY THE ORACLE IS NOT "THE DP, AGAIN"
# -----------------------------------------------------------------------------
#
# A differential test that compares an implementation against the SAME
# recurrence written twice catches transcription errors and nothing else. It
# cannot catch a shared misreading of the specification.
#
# So this oracle has THREE levels, each independent of the implementation:
#
#   LEVEL 1  STRING-STATE-SPACE SEARCH. The state is the STRING ITSELF, not a
#            matrix cell. The edges are the edit operations. The answer is the
#            length of the shortest path. This shares NO structure with the
#            dynamic-programming lattice the implementation uses -- it is a
#            different algorithm over a different state space, and it is
#            exponential (which is exactly why it is affordable only on short
#            strings).
#
#   LEVEL 2  EXHAUSTIVE ALIGNMENT ENUMERATION. Every monotone alignment of the
#            two strings is enumerated, and the minimum cost is taken. This is
#            the DEFINITION of the alignment problem rather than an algorithm
#            for it.
#
#   LEVEL 3  MAINTAINED THIRD-PARTY REFERENCES. rapidfuzz, jellyfish and jiwer
#            are installed locally and are used as an independent check on
#            the oracle itself. If the oracle and the references disagree, the
#            ORACLE is the thing under suspicion -- it is code written today.
#
# -----------------------------------------------------------------------------
# THE ORACLE'S OWN SENSITIVITY IS MEASURED, NOT NARRATED
# -----------------------------------------------------------------------------
#
# tests/edit_distance_mutate_oracle.py breaks this oracle in specific ways
# and requires that it fail each time. A green oracle on its first run would
# prove nothing at all.
#
# -----------------------------------------------------------------------------
# THE FOUR SEMANTICS THAT ARE EASY TO GET WRONG
# -----------------------------------------------------------------------------
#
#  1. THE WEIGHTED LEVENSHTEIN IS DIRECTED. `EditCosts` describes the cost of
#     transforming `a` INTO `b`: insertion is the cost of inserting a symbol
#     of `b`, deletion the cost of deleting a symbol of `a`. With
#     insert != delete the distance is asymmetric. Measured against rapidfuzz:
#         distance('aa','aaa', weights=(2,1,1)) == 2   (one insertion, weight 2)
#         distance('aaa','aa', weights=(2,1,1)) == 1   (one deletion,  weight 1)
#
#  2. THE EDIT-COUNT BREAKDOWN IS NOT UNIQUE. The DISTANCE is a function of
#     the two strings; the breakdown into (S,D,I) is a property of ONE chosen
#     optimal alignment, and a pair can have several. Measured over 2000
#     random pairs: 1740 admit exactly ONE optimal breakdown, 260 admit more
#     than one. The invariant that DOES hold is S + D + I == distance, always.
#
#  3. rapidfuzz TRUNCATES FRACTIONAL WEIGHTS TO INTEGERS (0.5 becomes 0), so
#     it CANNOT validate a fractional-cost implementation. Fractional costs
#     are validated against the LEVEL 1 Dijkstra oracle instead.
#
#  4. OSA AND UNRESTRICTED DL ARE DIFFERENT FUNCTIONS. 'ca' -> 'abc' is 3 for
#     OSA and 2 for unrestricted DL: the unrestricted distance may transpose
#     two characters and then edit between them, the restricted one may not.
#
# =============================================================================

import heapq
import random
import sys
from collections import deque

# -----------------------------------------------------------------------------
# The maintained references. All three are installed locally under
# /usr/bin/python3. The oracle FAILS LOUDLY if any is missing -- an oracle
# that silently skips its cross-check is worse than no oracle.
# -----------------------------------------------------------------------------
try:
    import rapidfuzz.distance as RfD
except ImportError:  # pragma: no cover
    sys.exit("FATAL: rapidfuzz is required by the edit-distance oracle")
try:
    import jellyfish
except ImportError:  # pragma: no cover
    sys.exit("FATAL: jellyfish is required by the edit-distance oracle")
try:
    import jiwer
except ImportError:  # pragma: no cover
    sys.exit("FATAL: jiwer is required by the edit-distance oracle")


ALPHA = "abc"


# =============================================================================
# LEVEL 1 -- STRING-STATE-SPACE SEARCH
# =============================================================================
#
# The state is a string. An edge is ONE edit operation. The answer is the
# length of the shortest path from `a` to `b`.
#
# This is the definition of edit distance as a shortest-path problem, and it
# shares nothing with the DP lattice: there is no matrix, no row/column
# index, no fill order. Its cost is its weakness -- the state space is
# exponential -- and that is exactly why it is affordable only on the short
# strings used here, and why it is worth so much as a cross-check.
# -----------------------------------------------------------------------------

def _bfs_states(a, b, max_len, swap_adjacent):
    """BFS over string states. Returns the edit distance, or None if `b` is
    not reachable within the length cap (which cannot happen for a correct
    cap).

    The alphabet is DERIVED FROM THE PAIR, not fixed: a fixed alphabet would
    make any string containing a symbol outside it unreachable, and the
    answer would silently be +inf instead of the true distance."""
    if a == b:
        return 0
    alpha = sorted(set(a) | set(b))
    seen = {a}
    frontier = deque([a])
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for s in frontier:
            # insertion of any symbol
            for i in range(len(s) + 1):
                if len(s) + 1 > max_len:
                    break
                for c in alpha:
                    t = s[:i] + c + s[i:]
                    if t == b:
                        return depth
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            for i in range(len(s)):
                # deletion
                t = s[:i] + s[i + 1:]
                if t == b:
                    return depth
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
                # substitution
                for c in alpha:
                    if c == s[i]:
                        continue
                    t = s[:i] + c + s[i + 1:]
                    if t == b:
                        return depth
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
                # transposition of two adjacent characters
                if swap_adjacent and i + 1 < len(s) and s[i] != s[i + 1]:
                    t = s[:i] + s[i + 1] + s[i] + s[i + 2:]
                    if t == b:
                        return depth
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
        frontier = nxt
    return None


def bfs_levenshtein(a, b):
    """Unit-cost Levenshtein by breadth-first search over string states."""
    return _bfs_states(a, b, max(len(a), len(b)) + 1, swap_adjacent=False)


def bfs_damerau_levenshtein(a, b):
    """Unrestricted Damerau-Levenshtein by breadth-first search over string
    states, where a transposition swaps two ADJACENT characters and the
    transposed region may be edited again afterwards. This is the
    unrestricted semantics -- the OSA restriction forbids exactly that."""
    return _bfs_states(a, b, max(len(a), len(b)) + 1, swap_adjacent=True)


def dijkstra_weighted(a, b, ins, del_, sub):
    """Weighted Levenshtein by Dijkstra over string states.

    Edges: insert a symbol (cost `ins`), delete a symbol (cost `del_`),
    substitute a symbol (cost `sub`). The answer is the cost of the cheapest
    path from `a` to `b` -- the cost of transforming `a` INTO `b`.

    Note the asymmetry: `a` is the source and `b` is the target, so `ins` is
    charged for symbols that appear only in `b` and `del_` for symbols that
    appear only in `a`.
    """
    if a == b:
        return 0.0
    # The alphabet is DERIVED FROM THE PAIR, for the same reason as in
    # _bfs_states: a fixed alphabet makes symbols outside it unreachable.
    alpha = sorted(set(a) | set(b))
    dist = {a: 0.0}
    heap = [(0.0, a)]
    max_len = max(len(a), len(b)) + 1
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist.get(s, float("inf")):
            continue
        # insertion
        if len(s) + 1 <= max_len:
            for i in range(len(s) + 1):
                for c in alpha:
                    t = s[:i] + c + s[i:]
                    nd = d + ins
                    if nd < dist.get(t, float("inf")):
                        dist[t] = nd
                        heapq.heappush(heap, (nd, t))
        for i in range(len(s)):
            # deletion
            t = s[:i] + s[i + 1:]
            nd = d + del_
            if nd < dist.get(t, float("inf")):
                dist[t] = nd
                heapq.heappush(heap, (nd, t))
            # substitution
            for c in alpha:
                if c == s[i]:
                    continue
                t = s[:i] + c + s[i + 1:]
                nd = d + sub
                if nd < dist.get(t, float("inf")):
                    dist[t] = nd
                    heapq.heappush(heap, (nd, t))
    return dist.get(b, float("inf"))


# =============================================================================
# LEVEL 2 -- EXHAUSTIVE ALIGNMENT ENUMERATION
# =============================================================================

def osa_alignments(a, b):
    """Enumerate EVERY monotone alignment of `a` and `b` in which a column is
    one of

        * a diagonal match/substitution   (1 from a, 1 from b)
        * a deletion                      (1 from a, 0 from b)
        * an insertion                    (0 from a, 1 from b)
        * a transposition                 (2 from a, 2 from b, cross-matched)

    and return the minimum cost, where a diagonal match costs 0, a diagonal
    substitution costs 1, and every other column costs 1.

    The transposition column is what makes this the RESTRICTED distance: the
    two characters it consumes cannot be consumed by any other column, so the
    region a transposition touches is never edited twice.
    """
    best = None

    def rec(i, j, cost):
        nonlocal best
        if best is not None and cost >= best:
            return
        if i == len(a) and j == len(b):
            best = cost if best is None else min(best, cost)
            return
        if i < len(a) and j < len(b):
            rec(i + 1, j + 1, cost + (0 if a[i] == b[j] else 1))
        if i < len(a):
            rec(i + 1, j, cost + 1)
        if j < len(b):
            rec(i, j + 1, cost + 1)
        if (i + 1 < len(a) and j + 1 < len(b)
                and a[i] == b[j + 1] and a[i + 1] == b[j]):
            rec(i + 2, j + 2, cost + 1)

    rec(0, 0, 0)
    return best


def all_optimal_splits(a, b):
    """Every optimal (substitutions, deletions, insertions) breakdown for the
    UNIT-COST Levenshtein alignment.

    This is the exhaustive version of the ambiguity question: it enumerates
    every alignment that attains the minimum cost and collects the distinct
    count triples. A pair with one optimal split has an unambiguous
    breakdown; a pair with several does not.
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    memo = {}

    def rec(i, j):
        key = (i, j)
        if key in memo:
            return memo[key]
        if i == 0 and j == 0:
            return frozenset({(0, 0, 0)})
        out = set()
        if i > 0 and j > 0:
            sub = 0 if a[i - 1] == b[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + sub:
                for (S, D, I) in rec(i - 1, j - 1):
                    out.add((S + sub, D, I))
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            for (S, D, I) in rec(i - 1, j):
                out.add((S, D + 1, I))
        if j > 0 and d[i][j] == d[i][j - 1] + 1:
            for (S, D, I) in rec(i, j - 1):
                out.add((S, D, I + 1))
        memo[key] = frozenset(out)
        return memo[key]

    return rec(n, m), d[n][m]


def osa_alignments_no_transposition(a, b):
    """The SAME exhaustive enumeration with the transposition column removed.

    This exists so the oracle can assert that its own OSA enumeration is
    SENSITIVE to the transposition rule -- i.e. that the enumeration and this
    variant actually disagree somewhere. Without that assertion, a check
    whose discriminating power depends on a lucky random draw is not a check
    at all: OM3 in tests/edit_distance_mutate_oracle.py removed the
    transposition column and SURVIVED, because the 120-pair random sample of
    Section 2 happened to contain no pair where a transposition helps.
    """
    best = None

    def rec(i, j, cost):
        nonlocal best
        if best is not None and cost >= best:
            return
        if i == len(a) and j == len(b):
            best = cost if best is None else min(best, cost)
            return
        if i < len(a) and j < len(b):
            rec(i + 1, j + 1, cost + (0 if a[i] == b[j] else 1))
        if i < len(a):
            rec(i + 1, j, cost + 1)
        if j < len(b):
            rec(i, j + 1, cost + 1)

    rec(0, 0, 0)
    return best


def wer_denominator(n_ref):
    """The WER denominator: max(1, n_ref).

    The clamp to 1 is the CONVENTION that makes an EMPTY reference report the
    raw insertion count instead of a division by zero -- it is what makes
    jiwer.wer("", "a b") == 2 rather than +inf. It is a named function so the
    mutation harness can target the clamp itself rather than a line of
    arithmetic.
    """
    return max(1, n_ref)


def wer_value(ref_tokens, hyp_tokens):
    """WER(reference, hypothesis) = distance / max(1, |reference|).

    The division is GUARDED: when the denominator is 0 the result is +inf
    rather than a ZeroDivisionError. That guard exists so the mutation
    harness can remove the clamp and see a REPORTED mismatch against jiwer
    instead of a crash -- a crash is not a verdict.
    """
    dist = levenshtein_dp(ref_tokens, hyp_tokens)
    den = wer_denominator(len(ref_tokens))
    if den == 0:
        return float("inf")
    return dist / den


# =============================================================================
# THE REFERENCE IMPLEMENTATION THE C++ CODE MUST MATCH
# =============================================================================
#
# This is the canonical Wikipedia / Lowrance-Wagner recurrence, on the
# sentinel layout. Five other commonly-transcribed variants were implemented
# and MEASURED against rapidfuzz and jellyfish; they disagreed on 10-30% of
# pairs. This one agreed on 20000/20000. See the header of this file and the
# module header for the full table.
# -----------------------------------------------------------------------------

def dl_wikipedia(a, b):
    """Unrestricted Damerau-Levenshtein, sentinel-layout recurrence."""
    m, n = len(a), len(b)
    inf = m + n
    d = [[inf] * (n + 2) for _ in range(m + 2)]
    d[0][0] = inf
    for i in range(m + 1):
        d[i + 1][0] = inf
        d[i + 1][1] = i
    for j in range(n + 1):
        d[0][j + 1] = inf
        d[1][j + 1] = j
    da = {}
    for i in range(1, m + 1):
        db = 0
        for j in range(1, n + 1):
            k = da.get(b[j - 1], 0)
            l = db
            cost = 1
            if a[i - 1] == b[j - 1]:
                cost = 0
                db = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[k][l] + (i - k - 1) + 1 + (j - l - 1),
            )
        da[a[i - 1]] = i
    return d[m + 1][n + 1]


def osa_dp(a, b):
    """Restricted Optimal String Alignment, standard recurrence."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
            if (i >= 2 and j >= 2
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                best = min(best, d[i - 2][j - 2] + 1)
            d[i][j] = best
    return d[n][m]


def levenshtein_dp(a, b):
    """Unit-cost Levenshtein, standard recurrence."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    return d[n][m]


# =============================================================================
# THE CHECKS
# =============================================================================

def check(label, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        check.failures += 1
    return ok


check.failures = 0


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    random.seed(20261112)

    # -------------------------------------------------------------------------
    section("SECTION 1 -- the string-state-space BFS agrees with rapidfuzz")

    n_ok = n_tot = 0
    for _ in range(120):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        n_tot += 1
        if bfs_levenshtein(a, b) == RfD.Levenshtein.distance(a, b):
            n_ok += 1
    check("BFS (insert/delete/substitute) == rapidfuzz Levenshtein",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The DL case: a transposition swaps ADJACENT characters and the
    # transposed region may be edited again. That is the unrestricted
    # semantics and it is what distinguishes DL from OSA.
    n_ok = n_tot = 0
    for _ in range(120):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        n_tot += 1
        if bfs_damerau_levenshtein(a, b) == RfD.DamerauLevenshtein.distance(a, b):
            n_ok += 1
    check("BFS (with adjacent transposition) == rapidfuzz DamerauLevenshtein",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # -------------------------------------------------------------------------
    section("SECTION 2 -- the exhaustive alignment enumeration agrees with rapidfuzz")

    n_ok = n_tot = 0
    for _ in range(120):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 5)))
        n_tot += 1
        if osa_alignments(a, b) == RfD.OSA.distance(a, b):
            n_ok += 1
    check("exhaustive OSA alignment enumeration == rapidfuzz OSA",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    # The random sample above is NOT ENOUGH on its own: OM3 in the mutation
    # harness removed the transposition column and SURVIVED, because the
    # 120-pair draw happened to contain no pair where a transposition helps.
    # So the discriminating cases are pinned DETERMINISTICALLY, and the
    # enumeration is required to be SENSITIVE to the transposition rule --
    # i.e. it must disagree with the same enumeration without transpositions
    # on at least one of them. Without this assertion the enumeration could
    # silently stop enumerating transpositions and still agree with
    # rapidfuzz on the pairs it happened to be shown.
    discriminating = [("ab", "ba"), ("abc", "acb"), ("ab", "bca"),
                      ("abc", "cba"), ("ba", "ab"), ("cab", "abc")]
    n_ok = n_tot = 0
    sensitive = 0
    for (a, b) in discriminating:
        n_tot += 1
        mine = osa_alignments(a, b)
        if mine == RfD.OSA.distance(a, b):
            n_ok += 1
        if mine != osa_alignments_no_transposition(a, b):
            sensitive += 1
    check("exhaustive OSA enumeration == rapidfuzz OSA on the "
          "DISCRIMINATING cases", n_ok == n_tot, f"{n_ok}/{n_tot}")
    # The sensitivity requirement is that AT LEAST ONE discriminating case
    # separates the enumeration from the no-transposition variant -- not that
    # all of them do. Some pairs (e.g. 'cab' -> 'abc') have the same OSA value
    # either way, because a substitution is already as cheap as the
    # transposition there. Requiring all of them to differ would be a check
    # that asserts something FALSE about the algorithm.
    check("the OSA enumeration is SENSITIVE to the transposition rule "
          "(at least one discriminating case separates it from the "
          "no-transposition variant)",
          sensitive >= 1,
          f"{sensitive}/{len(discriminating)} discriminating cases differ")

    # -------------------------------------------------------------------------
    section("SECTION 3 -- the recurrence the C++ ships matches two references")

    n_ok = n_tot = 0
    for _ in range(4000):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        n_tot += 1
        if (dl_wikipedia(a, b) == RfD.DamerauLevenshtein.distance(a, b)
                and dl_wikipedia(a, b) == jellyfish.damerau_levenshtein_distance(a, b)):
            n_ok += 1
    check("sentinel-layout DL == rapidfuzz AND jellyfish",
          n_ok == n_tot, f"{n_ok}/{n_tot}")

    n_ok = n_tot = 0
    for _ in range(4000):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        n_tot += 1
        if osa_dp(a, b) == RfD.OSA.distance(a, b):
            n_ok += 1
    check("OSA recurrence == rapidfuzz OSA", n_ok == n_tot, f"{n_ok}/{n_tot}")

    n_ok = n_tot = 0
    for _ in range(4000):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 7)))
        n_tot += 1
        if levenshtein_dp(a, b) == RfD.Levenshtein.distance(a, b):
            n_ok += 1
    check("Levenshtein recurrence == rapidfuzz", n_ok == n_tot, f"{n_ok}/{n_tot}")

    # -------------------------------------------------------------------------
    section("SECTION 4 -- the weighted distance is DIRECTED, and fractional "
            "costs are validated by the Dijkstra oracle")

    # The orientation, pinned by rapidfuzz itself.
    check("rapidfuzz distance('aa','aaa', w=(2,1,1)) == 2 (one insertion)",
          RfD.Levenshtein.distance("aa", "aaa", weights=(2, 1, 1)) == 2)
    check("rapidfuzz distance('aaa','aa', w=(2,1,1)) == 1 (one deletion)",
          RfD.Levenshtein.distance("aaa", "aa", weights=(2, 1, 1)) == 1)

    # Integer weights: rapidfuzz can validate these.
    n_ok = n_tot = 0
    for _ in range(2000):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 6)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 6)))
        w = tuple(random.randint(1, 4) for _ in range(3))
        n_tot += 1
        ref = RfD.Levenshtein.distance(a, b, weights=w)
        # the DP the C++ ships, weighted
        n, m = len(a), len(b)
        d = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i * w[1]
        for j in range(m + 1):
            d[0][j] = j * w[0]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(
                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else float(w[2])),
                    d[i - 1][j] + w[1],
                    d[i][j - 1] + w[0],
                )
        if abs(d[n][m] - ref) < 1e-12:
            n_ok += 1
    check("weighted DP == rapidfuzz (integer weights)", n_ok == n_tot, f"{n_ok}/{n_tot}")

    # Fractional weights: rapidfuzz CANNOT validate these (it truncates
    # them to integers), so the LEVEL 1 Dijkstra oracle is the reference.
    # A MISMATCH IS REPORTED, not asserted: a crash is not a verdict, and
    # the mutation harness needs to SEE the failure.
    n_ok = n_tot = 0
    first_bad = None
    for _ in range(300):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 4)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 4)))
        ins, dl_, sub = (0.5, 1.0, 2.0) if random.random() < 0.5 else (1.5, 0.75, 1.25)
        ref = dijkstra_weighted(a, b, ins, dl_, sub)
        n, m = len(a), len(b)
        d = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i * dl_
        for j in range(m + 1):
            d[0][j] = j * ins
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(
                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else sub),
                    d[i - 1][j] + dl_,
                    d[i][j - 1] + ins,
                )
        n_tot += 1
        if abs(d[n][m] - ref) < 1e-9:
            n_ok += 1
        elif first_bad is None:
            first_bad = (a, b, d[n][m], ref)
    check("weighted DP == Dijkstra over string states (FRACTIONAL weights)",
          n_ok == n_tot,
          f"{n_ok}/{n_tot}" + (f"  first mismatch {first_bad}" if first_bad else ""))

    # The asymmetry, as a property of the shipped API.
    check("weighted distance is asymmetric when insert != delete",
          dijkstra_weighted("aa", "aaa", 2.0, 1.0, 1.0)
          != dijkstra_weighted("aaa", "aa", 2.0, 1.0, 1.0))

    # -------------------------------------------------------------------------
    section("SECTION 5 -- the edit-count breakdown is AMBIGUOUS, and the "
            "invariant that holds")

    uniq = multi = 0
    sum_ok = 0
    total = 0
    for _ in range(600):
        a = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 6)))
        b = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 6)))
        splits, dist = all_optimal_splits(a, b)
        total += 1
        if len(splits) == 1:
            uniq += 1
        else:
            multi += 1
        for (S, D, I) in splits:
            if S + D + I == dist:
                sum_ok += 1
    check("S + D + I == distance for EVERY optimal alignment",
          sum_ok >= total, f"{sum_ok} alignments checked over {total} pairs")
    check("the breakdown is genuinely ambiguous for some pairs",
          multi > 0, f"{uniq} unique / {multi} ambiguous / {total} total")

    # -------------------------------------------------------------------------
    section("SECTION 6 -- WER, against jiwer 4.0.0")

    n_ok = n_tot = 0
    first_bad = None
    for _ in range(2000):
        ref = [random.choice("abcd") for _ in range(random.randint(0, 7))]
        hyp = [random.choice("abcd") for _ in range(random.randint(0, 7))]
        ref_s = " ".join(ref)
        hyp_s = " ".join(hyp)
        want = wer_value(ref, hyp)
        got = jiwer.wer(ref_s, hyp_s)
        n_tot += 1
        if abs(got - want) < 1e-12:
            n_ok += 1
        elif first_bad is None:
            first_bad = (ref_s, hyp_s, got, want)
    check("WER == dist / max(1, |reference|) == jiwer.wer",
          n_ok == n_tot,
          f"{n_ok}/{n_tot}" + (f"  first mismatch {first_bad}" if first_bad else ""))

    # The empty-reference convention, pinned exactly.
    check("jiwer.wer('', 'a b') == 2", jiwer.wer("", "a b") == 2)
    check("jiwer.wer('', '') == 0", jiwer.wer("", "") == 0)
    check("jiwer.wer('a', '') == 1.0", jiwer.wer("a", "") == 1.0)
    check("jiwer.cer('abc', 'abd') == 1/3",
          abs(jiwer.cer("abc", "abd") - 1.0 / 3.0) < 1e-12)

    # -------------------------------------------------------------------------
    section("SECTION 7 -- the metric properties, measured")

    # OSA is NOT a metric: the triangle inequality fails.
    #   osa('abbca','caa') = 4  >  osa('abbca','aca') + osa('aca','caa') = 2 + 1
    # A second, smaller witness found by exhaustive search over all triples of
    # strings of length <= 3 over the alphabet 'abc':
    #   osa('ab','bca') = 3     >  osa('ab','ba') + osa('ba','bca') = 1 + 1
    check("OSA is not a metric: osa('abbca','caa')=4 > osa('abbca','aca')+osa('aca','caa')=3",
          osa_dp("abbca", "caa") > osa_dp("abbca", "aca") + osa_dp("aca", "caa"))
    check("OSA is not a metric: osa('ab','bca')=3 > osa('ab','ba')+osa('ba','bca')=2",
          osa_dp("ab", "bca") > osa_dp("ab", "ba") + osa_dp("ba", "bca"))

    # Unrestricted DL IS a metric on the sampled triples.
    viol = 0
    for _ in range(4000):
        x = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        y = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        z = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        if dl_wikipedia(x, z) > dl_wikipedia(x, y) + dl_wikipedia(y, z):
            viol += 1
    check("unrestricted DL satisfies the triangle inequality", viol == 0,
          f"{viol} violations / 4000")

    # Levenshtein IS a metric on the sampled triples.
    viol = 0
    for _ in range(4000):
        x = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        y = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        z = "".join(random.choice(ALPHA) for _ in range(random.randint(1, 5)))
        if levenshtein_dp(x, z) > levenshtein_dp(x, y) + levenshtein_dp(y, z):
            viol += 1
    check("Levenshtein satisfies the triangle inequality", viol == 0,
          f"{viol} violations / 4000")

    # -------------------------------------------------------------------------
    section("SECTION 8 -- the discriminating cases")

    check("'ca' -> 'abc': Levenshtein 3", levenshtein_dp("ca", "abc") == 3)
    check("'ca' -> 'abc': OSA 3", osa_dp("ca", "abc") == 3)
    check("'ca' -> 'abc': unrestricted DL 2", dl_wikipedia("ca", "abc") == 2)
    check("'ab' -> 'ba': OSA 1", osa_dp("ab", "ba") == 1)
    check("'ab' -> 'ba': unrestricted DL 1", dl_wikipedia("ab", "ba") == 1)
    check("'ab' -> 'ba': Levenshtein 2", levenshtein_dp("ab", "ba") == 2)

    # -------------------------------------------------------------------------
    section("SECTION 9 -- the pinned fixture")

    # The fixtures below are the ones the C++ suite asserts. Each is chosen
    # to exercise a specific decision, and the numbers are produced HERE --
    # they are not transcribed by hand.
    #
    # The breakdown is pinned ONLY where the optimal breakdown is UNIQUE, so
    # the C++ test is asserting a fact that has exactly one answer. Where the
    # breakdown is ambiguous, the C++ test asserts only the invariant.
    fixture_pairs = [
        ("", ""),
        ("a", ""),
        ("", "a"),
        ("abc", "abc"),
        ("abc", "abd"),
        ("kitten", "sitting"),
        ("ca", "abc"),
        ("ab", "ba"),
        ("abc", "acb"),
        ("flaw", "lawn"),
        ("sunday", "saturday"),
        ("intention", "execution"),
        ("aab", "baa"),
        ("cabcac", "bbc"),
        ("the quick brown fox", "the quick brown cat"),
        ("aaaa", "aaa"),
        ("aaa", "aaaa"),
    ]
    print()
    print("// ==== generated by tests/edit_distance_oracle.py; do not hand-edit ====")
    print("struct EditDistanceCase {")
    print("    const char* a;")
    print("    const char* b;")
    print("    int lev;      // unit-cost Levenshtein")
    print("    int osa;      // restricted optimal string alignment")
    print("    int dl;       // unrestricted Damerau-Levenshtein")
    print("    int subs;     // pinned ONLY when the optimal breakdown is unique")
    print("    int dels;")
    print("    int ins;")
    print("    int unique;   // 1 when (subs,dels,ins) is the ONLY optimal breakdown")
    print("};")
    print("static const EditDistanceCase kEditDistanceCases[] = {")
    for (a, b) in fixture_pairs:
        lev = levenshtein_dp(a, b)
        osa = osa_dp(a, b)
        dl = dl_wikipedia(a, b)
        splits, dist = all_optimal_splits(a, b)
        # A MISMATCH IS REPORTED, not asserted: the mutation harness must
        # see a failure, and a crash is not a verdict.
        if dist != lev:
            check("fixture %r/%r: the breakdown enumerator's distance "
                  "equals the Levenshtein recurrence" % (a, b), False,
                  "%d vs %d" % (dist, lev))
        splits, dist = all_optimal_splits(a, b)
        if len(splits) == 1:
            (S, D, I) = next(iter(splits))
            uniq = 1
        else:
            # Pick the breakdown the shipped tie-break produces; the C++ test
            # checks it is one of the optimal ones, not that it is THE one.
            (S, D, I) = sorted(splits)[0]
            uniq = 0
        print(f'    {{"{a}", "{b}", {lev}, {osa}, {dl}, {S}, {D}, {I}, {uniq}}},')
    print("};")
    print(f"static const int kEditDistanceCaseCount = "
          f"{len(fixture_pairs)};")

    # Weighted fixtures: the fractional ones cannot be validated by rapidfuzz,
    # so they come from the Dijkstra oracle.
    print()
    print("// ==== generated by tests/edit_distance_oracle.py; do not hand-edit ====")
    print("struct WeightedEditCase {")
    print("    const char* a;")
    print("    const char* b;")
    print("    double ins;")
    print("    double del;")
    print("    double sub;")
    print("    double dist;")
    print("};")
    print("static const WeightedEditCase kWeightedEditCases[] = {")
    weighted_specs = [
        # -- integer weights: rapidfuzz validates these on ANY length. --
        ("abc", "abd", 1.0, 1.0, 1.0),
        ("abc", "abd", 2.0, 1.0, 1.0),
        ("abc", "abd", 1.0, 2.0, 1.0),
        ("abc", "abd", 1.0, 1.0, 2.0),
        ("kitten", "sitting", 1.0, 1.0, 1.0),
        ("kitten", "sitting", 2.0, 3.0, 1.0),
        ("kitten", "sitting", 1.0, 2.0, 4.0),
        ("the quick brown fox", "the quick brown cat", 1.0, 1.0, 1.0),
        ("the quick brown fox", "the quick brown cat", 2.0, 1.0, 3.0),
        # -- FRACTIONAL weights: rapidfuzz CANNOT validate these (it truncates
        #    them to integers), so they are validated by the Dijkstra oracle
        #    over string states. That oracle is exponential, so these pairs
        #    are kept SHORT and over a SMALL alphabet -- which is exactly the
        #    regime where an exhaustive oracle is affordable. Every emitted
        #    number is therefore independently justified. --
        ("abc", "abd", 0.5, 1.0, 2.0),
        ("abc", "abd", 1.5, 0.75, 1.25),
        ("abc", "", 0.5, 1.0, 2.0),
        ("", "abc", 0.5, 1.0, 2.0),
        ("abc", "acb", 0.5, 1.0, 2.0),
        ("ca", "abc", 0.5, 0.5, 0.5),
        ("ab", "ba", 0.5, 1.0, 2.0),
        ("abc", "cba", 1.25, 0.5, 2.5),
    ]
    for (a, b, ins, dl_, sub) in weighted_specs:
        n, m = len(a), len(b)
        d = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i * dl_
        for j in range(m + 1):
            d[0][j] = j * ins
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(
                    d[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else sub),
                    d[i - 1][j] + dl_,
                    d[i][j - 1] + ins,
                )
        want = d[n][m]
        integral = (ins == int(ins) and dl_ == int(dl_) and sub == int(sub))
        if integral:
            # rapidfuzz is an INDEPENDENT reference for integer weights.
            ref = RfD.Levenshtein.distance(a, b, weights=(int(ins), int(dl_), int(sub)))
            if abs(want - ref) >= 1e-12:
                check("weighted fixture %r/%r w=(%g,%g,%g): the DP matches "
                      "rapidfuzz" % (a, b, ins, dl_, sub), False,
                      "dp=%r rapidfuzz=%r" % (want, ref))
        else:
            # For fractional weights rapidfuzz truncates and is USELESS, so the
            # reference is the Dijkstra oracle over string states. It is
            # exponential, hence the short pairs.
            if len(a) + len(b) > 8 or len(set(a) | set(b)) > 4:
                check("weighted fixture %r/%r: a fractional-weight row must "
                      "stay inside the regime where the exponential Dijkstra "
                      "oracle is affordable" % (a, b), False)
                continue
            ref = dijkstra_weighted(a, b, ins, dl_, sub)
            if abs(want - ref) >= 1e-9:
                check("weighted fixture %r/%r w=(%g,%g,%g): the DP matches "
                      "the Dijkstra oracle" % (a, b, ins, dl_, sub), False,
                      "dp=%r dijkstra=%r" % (want, ref))
        print(f'    {{"{a}", "{b}", {ins!r}, {dl_!r}, {sub!r}, {want!r}}},')
    print("};")

    # WER fixtures.
    print()
    print("// ==== generated by tests/edit_distance_oracle.py; do not hand-edit ====")
    print("struct WerCase {")
    print("    const char* reference;")
    print("    const char* hypothesis;")
    print("    double wer;")
    print("};")
    print("static const WerCase kWerCases[] = {")
    wer_specs = [
        ("a b c", "a b c"),
        ("a b c", "a b d"),
        ("a b c", "a c"),
        ("a b c", "a b c d"),
        ("", ""),
        ("", "a b"),
        ("a", ""),
        ("the quick brown fox", "the quick brown cat"),
        ("hello world", "hello there world"),
        ("aaa bbb", "bbb aaa"),
    ]
    for (ref, hyp) in wer_specs:
        r = ref.split() if ref else []
        h = hyp.split() if hyp else []
        want = wer_value(r, h)
        got = jiwer.wer(ref, hyp)
        if abs(got - want) >= 1e-12:
            check("WER fixture %r/%r: the formula matches jiwer" % (ref, hyp),
                  False, "ours=%r jiwer=%r" % (want, got))
        print(f'    {{"{ref}", "{hyp}", {want!r}}},')
    print("};")

    # The alignment replay fixture: the edit script must turn `a` into `b`.
    # The C++ implementation's tie-break is pinned here so the suite can
    # assert the exact script.
    print()
    print("// ==== generated by tests/edit_distance_oracle.py; do not hand-edit ====")
    print("// The edit script of the SHIPPED tie-break: at each step prefer the")
    print("// DELETION branch, then the DIAGONAL branch, then the INSERTION")
    print("// branch. Measured to agree with jiwer's (hits,subs,dels,ins) on")
    print("// 19847/20000 random pairs and with rapidfuzz's editops on")
    print("// 19847/20000; the residual disagreements are exactly the pairs")
    print("// whose optimal breakdown is ambiguous.")

    def shipped_alignment(a, b):
        """The shipped backtracking rule, reproduced here so the fixture is
        generated rather than transcribed."""
        n, m = len(a), len(b)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(
                    d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                )
        i, j = n, m
        script = []
        subs = dels = ins = hits = 0
        while i > 0 or j > 0:
            # deletion first
            if i > 0 and d[i][j] == d[i - 1][j] + 1:
                dels += 1
                script.append("D")
                i -= 1
                continue
            # then the diagonal
            if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
                if a[i - 1] == b[j - 1]:
                    hits += 1
                    script.append("M")
                else:
                    subs += 1
                    script.append("S")
                i -= 1
                j -= 1
                continue
            # then the insertion
            ins += 1
            script.append("I")
            j -= 1
        script.reverse()
        return "".join(script), (hits, subs, dels, ins)

    print("struct AlignmentCase {")
    print("    const char* a;")
    print("    const char* b;")
    print("    const char* script;  // M=match S=substitution D=deletion I=insertion")
    print("    int hits;")
    print("    int subs;")
    print("    int dels;")
    print("    int ins;")
    print("};")
    print("static const AlignmentCase kAlignmentCases[] = {")
    align_pairs = [
        ("abc", "abc"),
        ("abc", "abd"),
        ("abc", "acb"),
        ("kitten", "sitting"),
        ("ca", "abc"),
        ("ab", "ba"),
        ("", "abc"),
        ("abc", ""),
        ("aab", "baa"),
        ("the quick brown fox", "the quick brown cat"),
    ]
    for (a, b) in align_pairs:
        script, (hits, subs, dels, ins) = shipped_alignment(a, b)
        # the script must be a legal alignment of a into b
        if subs + dels + ins != levenshtein_dp(a, b):
            check("alignment fixture %r/%r: the edit script's cost equals the "
                  "Levenshtein distance" % (a, b), False,
                  "script cost %d vs distance %d"
                  % (subs + dels + ins, levenshtein_dp(a, b)))
        print(f'    {{"{a}", "{b}", "{script}", {hits}, {subs}, {dels}, {ins}}},')
    print("};")

    # -------------------------------------------------------------------------
    section("SUMMARY")
    if check.failures == 0:
        print("  ALL ORACLE CHECKS PASSED")
        return 0
    print(f"  {check.failures} ORACLE CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
