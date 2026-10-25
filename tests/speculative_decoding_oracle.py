#!/usr/bin/env python3
"""
tests/speculative_decoding_oracle.py -- numpy float64 oracle for SPECULATIVE
DECODING, the draft/target rejection-sampling scheme added to
include/nn/utils/logit_processing.{h,cpp}.

RUN THIS BEFORE WRITING ANY C++. Every numeric expectation in
tests/test_speculative_decoding_oracle.py's C++ sibling
(tests/test_speculative_decoding.cpp) is copied from this file's output.

THE ALGORITHM, AS THE PAPER'S ALGORITHM 1 STATES IT. Leviathan, Kalman,
Matias, "Fast Inference from Transformers via Speculative Decoding", ICML
2023, arXiv:2211.17192. Transcribed VERBATIM from the paper's own HTML
(https://arxiv.org/html/2211.17192), not from memory:

    Algorithm 1  SpeculativeDecodingStep
      Inputs: M_p, M_q, prefix
      > Sample gamma guesses x_1, ..., x_gamma from M_q autoregressively.
        for i = 1 to gamma do
            q_i(x)  <- M_q(prefix + [x_1, ..., x_{i-1}])
            x_i     ~ q_i(x)
        end for
      > Run M_p in parallel.
        p_1(x), ..., p_{gamma+1}(x)  <-  M_p(prefix), ..., M_p(prefix + [x_1,...,x_gamma])
      > Determine the number of accepted guesses n.
        r_1 ~ U(0,1), ..., r_gamma ~ U(0,1)
        n   <- min( { i-1 | 1 <= i <= gamma, r_i > p_i(x)/q_i(x) } union {gamma} )
      > Adjust the distribution from M_p if needed.
        p'(x) <- p_{n+1}(x)
        if n < gamma then
            p'(x) <- norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
        end if
      > Return one token from M_p, and n tokens from M_q.
        t ~ p'(x)
        return prefix + [x_1, ..., x_n, t]

THE SEMANTICS PINNED HERE ARE SP1..SP12, to match the header's comment block
and docs/plans/2026-10-25_speculative-decoding.md.

===============================================================================
WHAT THIS ORACLE CAUGHT IN ITS OWN FIRST DRAFT (2026-10-25)
===============================================================================
The first draft of this file asserted a CLAMP on the acceptance ratio -- that
`min(1, p_i(x_i)/q_i(x_i))` is REQUIRED by the mathematics -- and built a
"no-clamp foil" to prove it. THE FOIL WAS VACUOUS: it produced IDENTICAL output
to the correct implementation on every trial.

The reason is a fact about the ALGORITHM, not about the implementation:

    accept iff  r <= p/q ,   r ~ U[0,1)

When p > q the ratio EXCEEDS 1, and then `r <= ratio` is satisfied by every
r in [0,1) AUTOMATICALLY -- which is precisely the correct behaviour ("the
target likes this token at least as much as the draft did, so accept it").
So the clamp is INERT when the acceptance rule is written as a THRESHOLD
comparison, and no fixture can ever discriminate it.

This is the same shape as the repo's recorded "measured-inert mutation" cases
(see the TDD skill's "an unexplained survivor is NOT a gap until a probe has
measured whether it can change anything"): the clamp is DEFENSIVE, it
documents that the ratio is a probability, and it must be MEASURED as inert
rather than asserted as load-bearing. The C++ harness therefore scores the
drop-the-clamp mutation with an INVERTED verdict -- it fails if the mutation
ever starts to matter.

WHAT IS ACTUALLY LOAD-BEARING AT THE ACCEPTANCE STEP, and is measured here:

  (a) THE RATIO ITSELF must be `p/q` of PROBABILITIES. A rule that accepts
      DETERMINISTICALLY on `p >= q` is a different algorithm, and it is the
      natural mistake -- see SP12 below.
  (b) THE POSITION INDEX: `p_i` and `q_i` are the two models' distributions
      at the SAME position.
  (c) THE RESIDUAL INDEX: `p_{n+1}`/`q_{n+1}`, not `p_n`/`q_n`.

===============================================================================
TWO INDEPENDENT FORMULATIONS OF THE STEP, CROSS-CHECKED TOKEN-FOR-TOKEN
===============================================================================
One formulation agreeing with itself proves nothing. The step is implemented
here TWICE, in structurally different code paths, and they are required to
produce IDENTICAL token sequences given the same guesses and the same uniform
draws:

  * `step_A` -- the paper's Algorithm 1, transcribed literally: compute the
    acceptance FLAGS for all gamma guesses, derive the index n, then build
    ONE distribution and draw the correction token from it.
  * `step_B` -- the DEFINITIONAL form, from the paper's correctness proof
    (Section 2.3, Appendix A.1): "at each position, draw from q, accept with
    probability min(1, p/q), and on a rejection draw the correction from the
    residual and STOP". It never computes an index n.

They share no code and no control flow, so their agreement is evidence.

The uniform draws are passed in EXPLICITLY (as `us`) rather than being drawn
inside, precisely so the two formulations can be compared token-for-token
instead of only in distribution.

Run with:

    /home/stefan/.venv/bin/python tests/speculative_decoding_oracle.py

Prints a C++-pasteable table and exits 0 iff every self-check passes.
"""
import math
import sys

import numpy as np

NEG_INF = float("-inf")


# ===========================================================================
# The primitives
# ===========================================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """Max-subtracting softmax, exactly as the C++ file does it."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def norm(v: np.ndarray) -> np.ndarray:
    """Normalise to sum to 1. A zero sum is a caller error, not a silent NaN."""
    s = float(np.sum(v))
    if not (s > 0.0) or not math.isfinite(s):
        raise ValueError(f"norm(): non-positive sum {s!r}")
    return np.asarray(v, dtype=np.float64) / s


def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """SP6: norm(max(0, p - q)), componentwise, over the FULL vocabulary."""
    return norm(np.maximum(0.0, np.asarray(p, dtype=np.float64)
                            - np.asarray(q, dtype=np.float64)))


def acceptance_prob(p_at: float, q_at: float) -> float:
    """The acceptance probability min(1, p/q). See the header note: the clamp
    is DEFENSIVE -- it is inert when the rule is written as a threshold
    comparison, and it is kept so the code says what it means."""
    if q_at <= 0.0:
        # The guess was DRAWN from q, so q(guess) > 0 by construction. This
        # branch is unreachable in the real algorithm; it exists so that a
        # caller-supplied guess cannot produce a division by zero.
        return 1.0
    return min(1.0, p_at / q_at)


def draw_from(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF draw from a probability row, given u in [0,1)."""
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if u < acc:
            return i
    return len(probs) - 1


# ===========================================================================
# FORMULATION A -- the paper's Algorithm 1, transcribed literally.
# ===========================================================================
def step_A(p_rows, q_rows, guesses, us):
    """
    p_rows : list of length gamma+1 -- the TARGET model's probability row at
             each of the gamma+1 positions.
    q_rows : list of length gamma+1 -- the DRAFT model's probability row at
             each of the gamma+1 positions.
    guesses : the gamma tokens the draft proposed (drawn from q_1..q_gamma).
    us      : gamma+1 uniform draws. us[0..gamma-1] are the acceptance draws
              r_i; us[gamma] is the draw for the final token.

    Returns (tokens, n, rejected, accepted_flags).
    """
    gamma = len(q_rows) - 1
    assert len(p_rows) == gamma + 1
    assert len(guesses) == gamma
    assert len(us) == gamma + 1

    # --- Determine the number of accepted guesses n. ------------------------
    # The paper: n <- min({i-1 | 1<=i<=gamma, r_i > p_i(x)/q_i(x)} union {gamma})
    accepted_flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        accepted_flags.append(us[i] <= acceptance_prob(pi, qi))

    # SP4: n is the number of LEADING accepted guesses. The first rejection
    # ends the run -- everything after it is discarded, because the correction
    # token replaces the position where the rejection happened.
    n = gamma
    first_rejection = -1
    for i in range(gamma):
        if not accepted_flags[i]:
            n = i            # 0-based: the guess at index i is rejected
            first_rejection = i
            break

    rejected = (n < gamma)

    # --- Adjust the distribution from M_p if needed. ------------------------
    if rejected:
        # SP6: the residual at index n+1 (1-based) == index n (0-based).
        p_prime = residual(p_rows[n], q_rows[n])
    else:
        # SP7: when every guess is accepted, the token is drawn from
        # p_{gamma+1} UNMODIFIED. Using the residual here would lose the
        # bonus token and cost the entire speedup -- and for identical models
        # the residual is ALL-ZERO, so it would also be a division by zero.
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)

    # --- Return one token from M_p, and n tokens from M_q. ------------------
    t = draw_from(p_prime, us[gamma])

    # SP9: exactly n+1 tokens -- the n accepted guesses plus one.
    tokens = [guesses[i] for i in range(n)] + [t]
    return tokens, n, rejected, accepted_flags


# ===========================================================================
# FORMULATION B -- the definitional form, from the correctness proof.
#
# "at each position, draw from q; accept the draw with probability
#  min(1, p/q); on a rejection, draw the correction from the residual and
#  STOP".
#
# It never computes an index n and never computes a threshold list.
# ===========================================================================
def step_B(p_rows, q_rows, guesses, us):
    """
    Same inputs as step_A. Returns (tokens, n, rejected, None).
    """
    gamma = len(q_rows) - 1
    tokens = []
    for i in range(gamma):
        pi_row = np.asarray(p_rows[i], dtype=np.float64)
        qi_row = np.asarray(q_rows[i], dtype=np.float64)
        x = guesses[i]
        ratio = acceptance_prob(float(pi_row[x]), float(qi_row[x]))
        if us[i] <= ratio:
            tokens.append(x)
            continue
        # Rejected: the correction is drawn from the residual AT THIS
        # POSITION, and the run stops here.
        r = residual(pi_row, qi_row)
        tokens.append(draw_from(r, us[gamma]))
        return tokens, len(tokens) - 1, True, None
    # Every guess was accepted: the bonus token comes from p_{gamma+1}.
    bonus = np.asarray(p_rows[gamma], dtype=np.float64)
    tokens.append(draw_from(bonus, us[gamma]))
    return tokens, gamma, False, None


# ===========================================================================
# THE FOILS -- each is a natural mistake. Each must DIFFER from the correct
# behaviour on a fixture that can separate them.
# ===========================================================================
def foil_RESIDUAL_ALWAYS(p_rows, q_rows, guesses, us):
    """SP7 foil: the residual is used even when n == gamma."""
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(us[i] <= acceptance_prob(pi, qi))
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    # <-- residual ALWAYS, even when nothing was rejected.
    p_prime = residual(p_rows[n], q_rows[n])
    t = draw_from(p_prime, us[gamma])
    return [guesses[i] for i in range(n)] + [t], n, (n < gamma)


def foil_WRONG_INDEX(p_rows, q_rows, guesses, us):
    """SP6 foil: the residual is built at index n-1 instead of n."""
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(us[i] <= acceptance_prob(pi, qi))
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    if n < gamma:
        idx = max(0, n - 1)                        # <-- WRONG index
        p_prime = residual(p_rows[idx], q_rows[idx])
    else:
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
    t = draw_from(p_prime, us[gamma])
    return [guesses[i] for i in range(n)] + [t], n, (n < gamma)


def foil_EMIT_REJECTED(p_rows, q_rows, guesses, us):
    """SP5/SP9 foil: the rejected guess is emitted as well as the correction."""
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(us[i] <= acceptance_prob(pi, qi))
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    if n < gamma:
        p_prime = residual(p_rows[n], q_rows[n])
    else:
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
    t = draw_from(p_prime, us[gamma])
    # <-- the rejected guess x_n is ALSO emitted. (Only meaningful when there
    # IS a rejection; when n == gamma there is no x_{gamma+1} to emit.)
    emitted = [guesses[i] for i in range(n + 1)] if n < gamma \
        else [guesses[i] for i in range(n)]
    return emitted + [t], n, (n < gamma)


def foil_NO_MAX_ZERO(p_rows, q_rows, guesses, us):
    """SP6 foil: the componentwise max(0, .) is dropped from the residual."""
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(us[i] <= acceptance_prob(pi, qi))
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    if n < gamma:
        # <-- NO max(0, .)
        p_prime = norm(np.asarray(p_rows[n], dtype=np.float64)
                       - np.asarray(q_rows[n], dtype=np.float64))
    else:
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
    t = draw_from(p_prime, us[gamma])
    return [guesses[i] for i in range(n)] + [t], n, (n < gamma)


def foil_STRICT_ACCEPT(p_rows, q_rows, guesses, us):
    """SP3 foil: accept iff r_i < ratio (strict), so r_i == ratio rejects."""
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(us[i] < acceptance_prob(pi, qi))
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    if n < gamma:
        p_prime = residual(p_rows[n], q_rows[n])
    else:
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
    t = draw_from(p_prime, us[gamma])
    return [guesses[i] for i in range(n)] + [t], n, (n < gamma)


def foil_DETERMINISTIC_ACCEPT(p_rows, q_rows, guesses, us):
    """SP12 foil: accept DETERMINISTICALLY on p >= q, ignoring the uniform.

    This is the natural mistake: "keep the guess if the target model likes it
    at least as much as the draft did". It ignores the acceptance PROBABILITY
    entirely, so its acceptance RATE is 0 or 1 per token instead of p/q.
    """
    gamma = len(q_rows) - 1
    flags = []
    for i in range(gamma):
        pi = float(p_rows[i][guesses[i]])
        qi = float(q_rows[i][guesses[i]])
        flags.append(pi >= qi)
    n = gamma
    for i in range(gamma):
        if not flags[i]:
            n = i
            break
    if n < gamma:
        p_prime = residual(p_rows[n], q_rows[n])
    else:
        p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
    t = draw_from(p_prime, us[gamma])
    return [guesses[i] for i in range(n)] + [t], n, (n < gamma)


def foil_WRONG_P_ROW(p_rows, q_rows, guesses, us):
    """SP2 foil: the target's row is taken at index i+1 instead of i."""
    shifted_p = [np.asarray(p_rows[min(i + 1, len(p_rows) - 1)], dtype=np.float64)
                 for i in range(len(p_rows))]
    return step_A(shifted_p, q_rows, guesses, us)


# ===========================================================================
# FIXTURES
# ===========================================================================
# FIXTURE A -- a POSITION-VARYING pair of models.
#
# THE FIRST DRAFT OF THIS ORACLE USED THE SAME DISTRIBUTION AT EVERY
# POSITION, and that made the SP6 wrong-index foil VACUOUS: with p_n == p_{n+1}
# for every n, an implementation that reads the residual from the wrong index
# computes the SAME distribution and the test cannot see it. A fixture whose
# discriminating value is absent from every position cannot test anything.
#
# So every position now carries a DIFFERENT distribution, and the acceptance
# probabilities differ from position to position as well.
#
# Position 0: p = [0.60, 0.20, 0.15, 0.05]   q = [0.25, 0.25, 0.25, 0.25]
#             acceptance probabilities: [1.0, 0.8, 0.6, 0.2]
#             -- token 0 has p > q (ratio 2.4 -> clamped to 1) and token 3 has
#                p < q (ratio 0.2), so BOTH the accept-always case and the
#                genuinely fractional case are present at position 0.
# Position 1: p = [0.10, 0.50, 0.30, 0.10]   q = [0.40, 0.20, 0.20, 0.20]
#             acceptance probabilities: [0.25, 1.0, 1.0, 0.5]
# Position 2: p = [0.25, 0.25, 0.25, 0.25]   q = [0.10, 0.40, 0.30, 0.20]
#             acceptance probabilities: [1.0, 0.625, 0.8333..., 1.0]
# Position 3: p = [0.05, 0.15, 0.20, 0.60]   q = [0.20, 0.20, 0.30, 0.30]
#             acceptance probabilities: [0.25, 0.75, 0.6666..., 1.0]
FIXTURE_A_VOCAB = 4
FIXTURE_A_P = [
    [0.60, 0.20, 0.15, 0.05],
    [0.10, 0.50, 0.30, 0.10],
    [0.25, 0.25, 0.25, 0.25],
    [0.05, 0.15, 0.20, 0.60],
]
FIXTURE_A_Q = [
    [0.25, 0.25, 0.25, 0.25],
    [0.40, 0.20, 0.20, 0.20],
    [0.10, 0.40, 0.30, 0.20],
    [0.20, 0.20, 0.30, 0.30],
]


# FIXTURE B -- IDENTICAL draft and target.
#
# The strongest single property (SP10): with p == q the ratio is 1
# everywhere, every guess is accepted, and the step emits EXACTLY gamma+1
# tokens. It is also the discriminator for SP7: the residual of identical
# distributions is ALL-ZERO, so an implementation that always uses the
# residual either divides by zero or emits a wrong token.
FIXTURE_B_VOCAB = 3
FIXTURE_B_ROWS = [[0.5, 0.3, 0.2]] * 4


# FIXTURE C -- the SP12 discriminator.
#
# The natural mistake at the acceptance step is to accept DETERMINISTICALLY
# on `p >= q`, ignoring the acceptance PROBABILITY. The correct rule accepts
# a guess with probability p/q, which for p < q is strictly between 0 and 1.
#
# The measurement: over many uniform draws, the correct rule accepts a guess
# with p < q with rate p/q, while the deterministic rule accepts it with
# rate 0 (or 1, for p > q). The two rates are FAR apart, so the property is
# measured rather than argued.
FIXTURE_C_VOCAB = 2
FIXTURE_C_P = [[0.8, 0.2]]
FIXTURE_C_Q = [[0.5, 0.5]]


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
    print("SPECULATIVE DECODING ORACLE (numpy float64)")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # 1. FIXTURE A: the pinned numbers
    # -----------------------------------------------------------------------
    print("\n--- 1. FIXTURE A: position-varying models ---")
    for pos in range(4):
        p = np.array(FIXTURE_A_P[pos], dtype=np.float64)
        q = np.array(FIXTURE_A_Q[pos], dtype=np.float64)
        acc = [acceptance_prob(float(p[t]), float(q[t]))
               for t in range(FIXTURE_A_VOCAB)]
        print(f"  pos {pos}: p = {[repr(float(x)) for x in p]}")
        print(f"          q = {[repr(float(x)) for x in q]}")
        print(f"          acceptance probs = {[repr(float(x)) for x in acc]}")
        check(abs(float(p.sum()) - 1.0) < 1e-12, f"A pos {pos}: p sums to 1")
        check(abs(float(q.sum()) - 1.0) < 1e-12, f"A pos {pos}: q sums to 1")
        for t in range(FIXTURE_A_VOCAB):
            check(acc[t] <= 1.0 + 1e-15,
                  f"A pos {pos} token {t}: the acceptance probability is <= 1")

    p0 = np.array(FIXTURE_A_P[0], dtype=np.float64)
    q0 = np.array(FIXTURE_A_Q[0], dtype=np.float64)
    check(p0[0] > q0[0], "A pos 0: token 0 has p > q (the ratio exceeds 1)")
    check(p0[3] < q0[3], "A pos 0: token 3 has p < q (a fractional ratio)")
    acc0 = [acceptance_prob(float(p0[t]), float(q0[t]))
            for t in range(FIXTURE_A_VOCAB)]
    check([round(float(x), 12) for x in acc0] == [1.0, 0.8, 0.6, 0.2],
          "A pos 0: the acceptance probabilities are exactly [1.0, 0.8, 0.6, 0.2]")
    # The unclamped ratio at token 0 is 2.4 -- the clamp is what turns it into
    # 1.0. This is the value the C++ test pins.
    print(f"  unclamped ratio at pos 0 token 0 = {p0[0] / q0[0]!r}")

    # -----------------------------------------------------------------------
    # 2. THE RESIDUAL DISTRIBUTION (SP6)
    # -----------------------------------------------------------------------
    print("\n--- 2. THE RESIDUAL DISTRIBUTION (SP6) ---")
    res0 = residual(p0, q0)
    print(f"  residual(p0,q0) = {[repr(float(x)) for x in res0]}")
    check(abs(float(res0.sum()) - 1.0) < 1e-12, "SP6: the residual sums to 1")
    # p0 - q0 = [0.35, -0.05, -0.10, -0.20] -> max(0,.) = [0.35, 0, 0, 0]
    check(float(res0[0]) == 1.0,
          "SP6: the residual is the POINT MASS on token 0 (p0 - q0 is "
          "positive only there)")
    for t in (1, 2, 3):
        check(float(res0[t]) == 0.0,
              f"SP6: the residual is ZERO at token {t} (p < q there)")

    # A residual with TWO surviving coordinates, so the SUM divisor matters.
    p2 = np.array([0.60, 0.40, 0.0, 0.0], dtype=np.float64)
    q2 = np.array([0.30, 0.10, 0.30, 0.30], dtype=np.float64)
    res2 = residual(p2, q2)
    print(f"  residual2 = {[repr(float(x)) for x in res2]}")
    check(abs(float(res2[0]) - 0.5) < 1e-12 and abs(float(res2[1]) - 0.5) < 1e-12,
          "SP6: residual2 is exactly [0.5, 0.5, 0, 0] -- the SUM divisor matters")
    check(abs(float(res2[0]) - 1.0) > 1e-12,
          "SP6: residual2 is NOT normalised by the MAX (which would give 1.0)")

    # The residual at position 1 -- a DIFFERENT distribution from position 0,
    # so the wrong-index foil has something to separate.
    res1 = residual(np.array(FIXTURE_A_P[1]), np.array(FIXTURE_A_Q[1]))
    print(f"  residual(pos1) = {[repr(float(x)) for x in res1]}")
    check(res1.tolist() != res0.tolist(),
          "SP6: the residual at position 1 DIFFERS from the one at position 0 "
          "-- so the index in the residual is observable")

    # -----------------------------------------------------------------------
    # 3. IDENTICAL MODELS (SP10, SP7)
    # -----------------------------------------------------------------------
    print("\n--- 3. IDENTICAL MODELS (SP10) ---")
    pB = np.array(FIXTURE_B_ROWS[0], dtype=np.float64)
    qB = np.array(FIXTURE_B_ROWS[0], dtype=np.float64)
    check(np.array_equal(pB, qB), "B: the draft and target are IDENTICAL")
    for trial in range(20):
        rng = np.random.default_rng(20261025 + trial)
        gamma = 3
        guesses = [int(rng.choice(FIXTURE_B_VOCAB, p=qB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toks, n, rej, flags = step_A(FIXTURE_B_ROWS, FIXTURE_B_ROWS, guesses, us)
        check(n == gamma, f"B trial {trial}: ALL gamma guesses are accepted")
        check(not rej, f"B trial {trial}: nothing was rejected")
        check(len(toks) == gamma + 1, f"B trial {trial}: EXACTLY gamma+1 tokens")
        check(toks[:gamma] == guesses,
              f"B trial {trial}: the emitted tokens ARE the guesses")
    print(f"  20 trials: every one emitted exactly 4 tokens")

    resB = np.maximum(0.0, pB - qB)
    print(f"  max(0, p - q) for identical models = {[repr(float(x)) for x in resB]}")
    check(float(resB.sum()) == 0.0,
          "SP7: the residual of IDENTICAL models is ALL-ZERO -- the residual "
          "must NOT be used when n == gamma")

    # -----------------------------------------------------------------------
    # 4. THE REJECTION PATH (SP4, SP5, SP9)
    # -----------------------------------------------------------------------
    print("\n--- 4. THE REJECTION PATH ---")
    gamma = 3
    seen_n = {}
    for seed in range(500):
        rng = np.random.default_rng(seed)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toks, n, rej, flags = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        check(len(toks) == n + 1,
              f"seed {seed}: the emitted count is EXACTLY n+1 ({len(toks)} vs {n + 1})")
        if rej:
            seen_n[n] = seen_n.get(n, 0) + 1
            # SP4: n is the index of the FIRST rejection, so the guesses
            # before it are the accepted ones.
            check(flags[:n] == [True] * n,
                  f"seed {seed}: the first n guesses are the ACCEPTED ones")
            check(flags[n] is False,
                  f"seed {seed}: the guess at index n is the FIRST rejection")
            # SP5: the rejected guess is NOT emitted. The emitted tokens are
            # the accepted guesses plus ONE correction.
            check(len(toks) == n + 1,
                  f"seed {seed}: only the correction replaces the rejected guess")
            for t in toks:
                check(0 <= t < FIXTURE_A_VOCAB,
                      f"seed {seed}: every emitted token is in the vocabulary")
    print(f"  rejection counts by n: {seen_n}")
    check(0 in seen_n, "a rejection at position 1 (n == 0) is REACHABLE")
    check(len(seen_n) >= 2,
          "rejections at MORE THAN ONE position are reachable -- so the "
          "residual's INDEX is observable")

    # The first rejection really is where the run stops: a rejection at index
    # 1 must discard guess 2 as well.
    for seed in range(500):
        rng = np.random.default_rng(seed)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toks, n, rej, flags = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        if rej and n == 0:
            check(len(toks) == 1,
                  f"SP5 seed {seed}: a rejection at position 1 emits exactly "
                  f"ONE token (the correction), not the guess as well")
            break

    # -----------------------------------------------------------------------
    # 5. THE TWO INDEPENDENT FORMULATIONS AGREE, TOKEN FOR TOKEN
    # -----------------------------------------------------------------------
    print("\n--- 5. THE TWO INDEPENDENT FORMULATIONS AGREE TOKEN-FOR-TOKEN ---")
    disagreements = 0
    total = 0
    for seed in range(2000):
        rng = np.random.default_rng(seed)
        gamma = 3
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toksA, nA, rejA, _ = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        toksB, nB, rejB, _ = step_B(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        total += 1
        if toksA != toksB or nA != nB or rejA != rejB:
            disagreements += 1
    print(f"  {total} trials, {disagreements} disagreements")
    check(disagreements == 0,
          "the paper's Algorithm 1 and the definitional formulation produce "
          "IDENTICAL tokens on every trial")

    # -----------------------------------------------------------------------
    # 6. THE DISTRIBUTIONAL PROPERTY (the paper's ACTUAL claim)
    # -----------------------------------------------------------------------
    print("\n--- 6. THE DISTRIBUTION OF THE EMITTED TOKENS MATCHES p ---")
    # No hand-written example can replace this: over many seeds the emitted
    # token's empirical distribution must match the TARGET's distribution.
    V = 3
    p_one = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    q_one = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    trials = 60000
    for name, fn in (("A", step_A), ("B", step_B)):
        rng = np.random.default_rng(4242)
        counts = np.zeros(V, dtype=np.int64)
        for _ in range(trials):
            gamma = 1
            x = int(rng.choice(V, p=q_one))          # the draft's guess
            us = [float(rng.random()) for _ in range(gamma + 1)]
            toks, n, rej, _ = fn([p_one, p_one], [q_one, q_one], [x], us)
            counts[toks[0]] += 1
        emp = counts / trials
        print(f"  {name}: empirical = {[repr(float(x)) for x in emp]}")
        print(f"     target   = {[repr(float(x)) for x in p_one]}")
        max_dev = float(np.max(np.abs(emp - p_one)))
        sigma = math.sqrt(max(p_one) * (1 - max(p_one)) / trials)
        print(f"     max |empirical - target| = {max_dev:.6f}  "
              f"(3.5 sigma = {3.5 * sigma:.6f})")
        check(max_dev < 3.5 * sigma,
              f"{name}: the emitted distribution matches the TARGET's")
        for t in range(V):
            check(abs(float(emp[t]) - float(p_one[t])) < 0.02,
                  f"{name}: token {t}'s emission rate matches p")

    # -----------------------------------------------------------------------
    # 7. SP12: THE ACCEPTANCE IS PROBABILISTIC, NOT DETERMINISTIC
    # -----------------------------------------------------------------------
    print("\n--- 7. SP12: THE ACCEPTANCE IS PROBABILISTIC ---")
    pC = np.array(FIXTURE_C_P[0], dtype=np.float64)
    qC = np.array(FIXTURE_C_Q[0], dtype=np.float64)
    ratio1 = acceptance_prob(float(pC[1]), float(qC[1]))
    print(f"  p = {[repr(float(x)) for x in pC]}")
    print(f"  q = {[repr(float(x)) for x in qC]}")
    print(f"  ratio at token 1 = {ratio1!r}")
    check(abs(ratio1 - 0.4) < 1e-12, "SP12: the ratio at token 1 is exactly 0.4")
    check(pC[1] < qC[1], "SP12: token 1 has p < q")

    # The CORRECT rate over many uniform draws.
    rng = np.random.default_rng(31337)
    trials = 200000
    us = rng.random(trials)
    correct_rate = float(np.mean(us <= ratio1))
    print(f"  measured acceptance rate at token 1 = {correct_rate:.6f}")
    check(abs(correct_rate - 0.4) < 0.005,
          "SP12: the measured acceptance rate equals p/q (0.4)")
    # The DETERMINISTIC rule's rate: p >= q is false here, so it rejects
    # ALWAYS -- rate 0. The two rates are 0.4 apart, which is unmissable.
    det_rate = 1.0 if float(pC[1]) >= float(qC[1]) else 0.0
    print(f"  deterministic (p >= q) acceptance rate = {det_rate:.6f}")
    check(abs(correct_rate - det_rate) > 0.3,
          "SP12: the probabilistic rate and the deterministic rate are FAR apart")

    # -----------------------------------------------------------------------
    # 8. THE FOILS MUST DIFFER (else the test is vacuous)
    # -----------------------------------------------------------------------
    print("\n--- 8. THE FOILS (each MUST differ from the correct behaviour) ---")
    gamma = 3

    def run_many(fn, trials=3000):
        """
        Collect the set of observable outcomes. A foil that RAISES is a
        DIFFERENT observable from one that returns tokens -- an exception is
        as distinguishable as a wrong token -- so the exception is recorded as
        a distinct outcome rather than crashing the oracle.
        """
        outs = set()
        ns = set()
        for s in range(trials):
            rng = np.random.default_rng(s)
            guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
            us = [float(rng.random()) for _ in range(gamma + 1)]
            try:
                toks, n, rej = fn(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
                outs.add(tuple(toks))
                ns.add(n)
            except Exception as exc:          # a crash IS a different outcome
                outs.add(("RAISED", type(exc).__name__))
                ns.add(-1)
        return outs, ns

    good_outs, good_ns = run_many(lambda p, q, g, u: step_A(p, q, g, u)[:3])
    foil_defs = [
        ("SP7 residual-always", foil_RESIDUAL_ALWAYS),
        ("SP6 wrong-index", foil_WRONG_INDEX),
        ("SP5 emit-rejected", foil_EMIT_REJECTED),
        ("SP6 no-max-zero", foil_NO_MAX_ZERO),
        ("SP12 deterministic-accept", foil_DETERMINISTIC_ACCEPT),
        ("SP2 wrong-p-row", foil_WRONG_P_ROW),
    ]
    for name, fn in foil_defs:
        f_outs, f_ns = run_many(fn)
        differs = (f_outs != good_outs) or (f_ns != good_ns)
        print(f"  {name:26s}: {len(f_outs):4d} distinct outputs, "
              f"{'DIFFERS' if differs else '*** SAME -- VACUOUS ***'}")
        check(differs, f"the {name} foil DIFFERS from the correct behaviour")

    # SP6's max(0, .) IS MATHEMATICALLY LOAD-BEARING, and the reason is worth
    # recording because it is not obvious:
    #
    #   p and q are both DISTRIBUTIONS, so sum(p) == sum(q) == 1, and therefore
    #       sum(p - q) == 0.
    #
    # Dropping the componentwise max(0, .) leaves a vector whose SUM IS ZERO,
    # so the normalisation divides by zero. The residual is not merely
    # "different" without the clamp -- it is UNDEFINED. The foil therefore
    # raises on every rejection, which the oracle records as a distinct
    # outcome rather than a crash.
    print("\n  Why max(0, .) is load-bearing (not merely a different number):")
    for pos in range(4):
        p_row = np.array(FIXTURE_A_P[pos], dtype=np.float64)
        q_row = np.array(FIXTURE_A_Q[pos], dtype=np.float64)
        raw_sum = float(np.sum(p_row - q_row))
        print(f"    pos {pos}: sum(p - q) = {raw_sum!r}")
        check(abs(raw_sum) < 1e-12,
              f"pos {pos}: sum(p - q) is ZERO -- without the componentwise "
              f"max(0, .) the normalisation divides by zero")
    clamped_sum = float(np.sum(np.maximum(0.0, p0 - q0)))
    print(f"    sum(max(0, p - q)) at pos 0 = {clamped_sum!r}")
    check(clamped_sum > 0.0,
          "pos 0: the CLAMPED residual has positive mass, so it normalises")

    # -----------------------------------------------------------------------
    # 9. THE TWO BOUNDARY CONVENTIONS ARE MEASURED INERT
    # -----------------------------------------------------------------------
    print("\n--- 9. THE BOUNDARY CONVENTIONS (measured inert) ---")

    # (a) THE CLAMP on the acceptance probability.
    #
    # Dropping `min(1, .)` changes NOTHING when the rule is written as a
    # threshold comparison, because r in [0,1) satisfies `r <= ratio`
    # automatically whenever ratio > 1. This is MEASURED here over many
    # trials, not argued.
    def step_A_no_clamp(p_rows, q_rows, guesses, us):
        gamma = len(q_rows) - 1
        flags = []
        for i in range(gamma):
            pi = float(p_rows[i][guesses[i]])
            qi = float(q_rows[i][guesses[i]])
            ratio = 1.0 if qi <= 0.0 else pi / qi      # <-- NO CLAMP
            flags.append(us[i] <= ratio)
        n = gamma
        for i in range(gamma):
            if not flags[i]:
                n = i
                break
        if n < gamma:
            p_prime = residual(p_rows[n], q_rows[n])
        else:
            p_prime = np.asarray(p_rows[gamma], dtype=np.float64)
        t = draw_from(p_prime, us[gamma])
        return [guesses[i] for i in range(n)] + [t], n, (n < gamma)

    diff = 0
    for s in range(3000):
        rng = np.random.default_rng(s)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        a = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)[:3]
        b = step_A_no_clamp(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        if a != b:
            diff += 1
    print(f"  (a) the clamped and unclamped rules differ on {diff}/3000 trials")
    check(diff == 0,
          "(a) the clamp is MEASURED INERT as a threshold comparison -- the C++ "
          "harness must score a drop-the-clamp mutation as inert, not as a gap")

    # The clamp IS load-bearing in the PROBABILITY form, which is why the C++
    # keeps it explicitly rather than relying on IEEE luck.
    prob_form_correct = min(1.0, float(pC[0]) / float(qC[0]))
    prob_form_unclamped = float(pC[0]) / float(qC[0])
    print(f"      in the probability form: correct {prob_form_correct!r} "
          f"vs unclamped {prob_form_unclamped!r}")
    check(prob_form_correct != prob_form_unclamped,
          "(a) the clamp IS load-bearing when the acceptance probability is "
          "computed explicitly -- which is why the C++ keeps it")

    # (b) `<=` VERSUS `<` AT THE ACCEPTANCE BOUNDARY.
    #
    # The two differ ONLY when r_i EXACTLY equals the acceptance probability.
    # The uniform draw is continuous, so that event has probability ZERO and a
    # RANDOM fixture can never observe it. The only way to discriminate is to
    # SUPPLY a uniform draw that lands exactly on the boundary -- which is why
    # the C++ test pins the boundary convention on a fixture with an
    # EXACTLY-DYADIC acceptance probability (0.5) and a draw of exactly 0.5.
    #
    # This is the same lesson the repo's top-p fixture learned: "a fixture
    # whose discriminating value is only correct to within a rounding error
    # does not discriminate anything". Here the discriminating value must be
    # EXACTLY representable AND must be supplied rather than drawn.
    print("\n  (b) `<` versus `<=` at the acceptance boundary:")
    # A fixture with a DYADIC acceptance probability: p = [0.5, 0.5],
    # q = [0.25, 0.25] -> the ratio at token 0 is 0.5/0.25 = 2.0 -> clamped 1.
    # Use a token whose ratio is exactly 0.5: p = [0.25, 0.75], q = [0.5, 0.5]
    # -> ratio at token 0 = 0.25/0.5 = 0.5 EXACTLY (dyadic).
    pD = np.array([0.25, 0.75], dtype=np.float64)
    qD = np.array([0.5, 0.5], dtype=np.float64)
    ratioD = acceptance_prob(float(pD[0]), float(qD[0]))
    print(f"      p = {[repr(float(x)) for x in pD]}  "
          f"q = {[repr(float(x)) for x in qD]}")
    print(f"      ratio at token 0 = {ratioD!r}  (exactly representable)")
    check(ratioD == 0.5, "(b) the boundary ratio is EXACTLY 0.5")
    # With r == 0.5 exactly, `<=` ACCEPTS and `<` REJECTS.
    accepts_le = (0.5 <= ratioD)
    accepts_lt = (0.5 < ratioD)
    print(f"      with u = 0.5 exactly: `<=` accepts = {accepts_le}, "
          f"`<` accepts = {accepts_lt}")
    check(accepts_le and not accepts_lt,
          "(b) the boundary IS discriminable when the draw is SUPPLIED at "
          "exactly 0.5 -- the C++ test must do this, and a random fixture "
          "cannot")
    # And the random-draw version cannot see it:
    random_disc = 0
    for s in range(3000):
        rng = np.random.default_rng(s)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        a = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)[:3]
        b = foil_STRICT_ACCEPT(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        if a != b:
            random_disc += 1
    print(f"      a RANDOM fixture discriminates `<` from `<=` on "
          f"{random_disc}/3000 trials")
    check(random_disc == 0,
          "(b) a random fixture CANNOT discriminate the boundary -- the "
          "C++ test must supply the draw at exactly the boundary")

    # -----------------------------------------------------------------------
    # 10. SP11: THE RESIDUAL IS NEVER ALL-ZERO WHEN A REJECTION HAPPENS
    # -----------------------------------------------------------------------
    print("\n--- 10. SP11: THE RESIDUAL IS NEVER ALL-ZERO ON A REJECTION ---")
    ok = True
    for seed in range(500):
        rng = np.random.default_rng(seed)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toks, n, rej, flags = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        if rej:
            r = np.maximum(0.0, np.asarray(FIXTURE_A_P[n], dtype=np.float64)
                           - np.asarray(FIXTURE_A_Q[n], dtype=np.float64))
            if not (float(r.sum()) > 0.0):
                ok = False
                break
    check(ok, "SP11: on every rejection, the residual has STRICTLY positive mass")

    # -----------------------------------------------------------------------
    # 11. SP9: THE EMITTED COUNT IS n+1
    # -----------------------------------------------------------------------
    print("\n--- 11. SP9: THE EMITTED COUNT IS EXACTLY n+1 ---")
    counts_by_n = {}
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        guesses = [int(rng.integers(0, FIXTURE_A_VOCAB)) for _ in range(gamma)]
        us = [float(rng.random()) for _ in range(gamma + 1)]
        toks, n, rej, flags = step_A(FIXTURE_A_P, FIXTURE_A_Q, guesses, us)
        check(len(toks) == n + 1,
              f"seed {seed}: the emitted count is EXACTLY n+1")
        counts_by_n[n] = counts_by_n.get(n, 0) + 1
    print(f"  emitted-count histogram by n: {counts_by_n}")

    # -----------------------------------------------------------------------
    # 12. THE C++-PASTEABLE TABLE
    # -----------------------------------------------------------------------
    print("\n--- 12. THE C++-PASTEABLE TABLE ---")
    print("  // [oracle] FIXTURE A -- position-varying draft/target")
    for pos in range(4):
        print(f"  //   pos {pos}: p = {FIXTURE_A_P[pos]!r}")
        print(f"  //           q = {FIXTURE_A_Q[pos]!r}")
    acc0_list = [acceptance_prob(float(p0[t]), float(q0[t]))
                 for t in range(FIXTURE_A_VOCAB)]
    print(f"  // pos 0 acceptance probs = {[repr(float(x)) for x in acc0_list]}")
    print(f"  // pos 0 residual         = {[repr(float(x)) for x in res0]}")
    print(f"  // pos 1 residual         = {[repr(float(x)) for x in res1]}")
    print(f"  // pos 0 unclamped ratio token 0 = {p0[0] / q0[0]!r}")
    print("  // [oracle] FIXTURE B -- identical models")
    print(f"  //   rows = {FIXTURE_B_ROWS[0]!r}")
    print("  // [oracle] FIXTURE C -- the SP12 discriminator")
    print(f"  //   p = {FIXTURE_C_P[0]!r}   q = {FIXTURE_C_Q[0]!r}")
    print(f"  //   ratio at token 1 = {ratio1!r}   measured rate = {correct_rate:.6f}")

    # A DETERMINISTIC worked example, for the C++ test to copy verbatim.
    print("\n  // [oracle] a deterministic worked example (gamma = 3)")
    work_guesses = [3, 1, 2]
    work_us = [0.9, 0.1, 0.4, 0.25]
    wt, wn, wrej, wflags = step_A(FIXTURE_A_P, FIXTURE_A_Q, work_guesses, work_us)
    print(f"  //   guesses = {work_guesses}   us = {work_us}")
    print(f"  //   -> tokens = {wt}   n = {wn}   rejected = {wrej}   "
          f"flags = {wflags}")
    wtB, wnB, wrejB, _ = step_B(FIXTURE_A_P, FIXTURE_A_Q, work_guesses, work_us)
    check(wt == wtB and wn == wnB and wrej == wrejB,
          "the worked example agrees between the two formulations")
    check(len(wt) == wn + 1, "the worked example emits exactly n+1 tokens")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"ORACLE SELF-CHECKS: {checks} run, {len(fails)} failed")
    for f in fails:
        if f:
            print(f"  FAILED: {f}")
    print("=" * 78)

    if fails:
        return 1
    print("ALL ORACLE SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
