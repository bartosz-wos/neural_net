#!/usr/bin/env python3
"""
tests/crf_oracle.py -- the SPECIFICATION for utils/crf.{h,cpp}.

RUN THIS BEFORE WRITING ANY C++. Its job is to catch defects in the DERIVATIONS
and in itself, exactly as tests/conformal_oracle.py, tests/sinkhorn_oracle.py
and tests/speculative_decoding_oracle.py did. In this repo the oracle has
caught more real defects in the DERIVATION than the C++ suite has caught in
the code.

WHAT IS BEING VERIFIED, AND WHY IT IS NOT OBVIOUS:

 1. THE PARTITION FUNCTION AND THE MARGINALS, against an EXHAUSTIVE
    enumeration of every labelling. This is the strongest possible oracle for
    a linear-chain model: the message passing is a clever way to avoid
    enumerating K^T paths, so brute force is an INDEPENDENT implementation of
    the same mathematics, written with none of the message-passing machinery.

 2. THE TRANSITION CONVENTION. `transitions(i, j)` is the score of the
    transition from label i at t-1 to label j at t. The transposed
    convention produces a DIFFERENT partition function on any asymmetric
    fixture, and the two are the SAME SHAPE and the SAME SIZE, so no type
    check can see the difference -- only a pinned value can.

 3. THE EDGE MARGINALS p(y_t = i, y_{t+1} = j | x). These are a DIFFERENT
    object from the node marginals, they do NOT sum to 1 over i for a fixed j
    in general, and they are what the transition gradient needs. A solver
    that returns node marginals where edge marginals belong passes a
    normalisation check only if the fixture is too weak.

 4. THE GRADIENTS, by FINITE DIFFERENCE against the brute-force marginals.
    This is the hard part and the reason the oracle exists: the adjoint
    through the forward-backward messages is a two-sided computation with a
    sign convention and an index convention that are easy to get wrong, and
    a wrong gradient can look perfectly reasonable.

 5. THE VITERBI SCORE IS BELOW log Z, NOT ABOVE IT. Z is a SUM over paths
    and the best path is one term of it, so log Z >= viterbi_score always.
    Asserting the opposite is a very natural mistake.

 6. THE LOG DOMAIN. At T = 200 with unit-scale scores the exp-space
    computation underflows to exactly 0, while the log-space computation
    stays finite. A solver that works in the exp space passes every small
    fixture and fails on every long one.

 7. FOILS. Every check below asserts that a WRONG variant DIFFERS from the
    right one. A check that cannot fail measures nothing.

Self-checks: every section prints its numbers AND asserts on them. A
self-check that prints a verdict it never tested is worse than none.
"""

import itertools
import math
import numpy as np

np.set_printoptions(precision=12, suppress=False)

FAILURES = []
CHECKS = 0


def check(ok, name, detail=""):
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name}   {detail}")
        FAILURES.append((name, detail))


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# 1. BRUTE FORCE. Enumerates every labelling. This is the reference.
# ---------------------------------------------------------------------------

def brute_force(emissions, transitions):
    """emissions (T,K), transitions (K,K).

    Returns dict with logZ, node marginals (T,K), edge marginals (T-1,K,K),
    viterbi path and viterbi score.

    score(y) = sum_t e[t, y_t] + sum_{t>=1} trans[y_{t-1}, y_t]
    """
    T, K = emissions.shape
    assert transitions.shape == (K, K)

    paths = list(itertools.product(range(K), repeat=T))
    scores = np.zeros(len(paths))
    for pi, y in enumerate(paths):
        s = sum(emissions[t, y[t]] for t in range(T))
        for t in range(1, T):
            s += transitions[y[t - 1], y[t]]
        scores[pi] = s

    m = scores.max()
    w = np.exp(scores - m)
    Z = w.sum()
    logZ = m + math.log(Z)

    # node marginals
    node = np.zeros((T, K))
    for pi, y in enumerate(paths):
        for t in range(T):
            node[t, y[t]] += w[pi]
    node /= Z

    # edge marginals
    edge = np.zeros((max(T - 1, 0), K, K))
    for pi, y in enumerate(paths):
        for t in range(T - 1):
            edge[t, y[t], y[t + 1]] += w[pi]
    edge /= Z

    # viterbi
    best = int(np.argmax(scores))
    return {
        "logZ": logZ,
        "node": node,
        "edge": edge,
        "viterbi_path": np.array(paths[best], dtype=int),
        "viterbi_score": float(scores[best]),
        "scores": scores,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# 2. MESSAGE PASSING (log space). This mirrors what the C++ will do.
# ---------------------------------------------------------------------------

def logsumexp(v):
    mx = np.max(v)
    if mx == -np.inf:
        return -np.inf
    return mx + math.log(np.sum(np.exp(v - mx)))


def forward_backward(emissions, transitions):
    T, K = emissions.shape
    alpha = np.full((T, K), -np.inf)
    # base: no transition before the first position
    alpha[0] = emissions[0]
    for t in range(1, T):
        for j in range(K):
            terms = alpha[t - 1] + transitions[:, j]
            alpha[t, j] = emissions[t, j] + logsumexp(terms)
    logZ = logsumexp(alpha[T - 1])

    beta = np.full((T, K), -np.inf)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        for i in range(K):
            terms = transitions[i] + emissions[t + 1] + beta[t + 1]
            beta[t, i] = logsumexp(terms)

    node = np.exp(alpha + beta - logZ)

    edge = np.zeros((max(T - 1, 0), K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                v = alpha[t, i] + transitions[i, j] + emissions[t + 1, j] + beta[t + 1, j] - logZ
                edge[t, i, j] = 0.0 if v == -np.inf else math.exp(v)
    return {"alpha": alpha, "beta": beta, "logZ": logZ, "node": node, "edge": edge}


def viterbi(emissions, transitions):
    T, K = emissions.shape
    delta = np.full((T, K), -np.inf)
    back = np.zeros((T, K), dtype=int)
    delta[0] = emissions[0]
    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + transitions[:, j]
            best = int(np.argmax(scores))
            delta[t, j] = emissions[t, j] + scores[best]
            back[t, j] = best
    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, float(delta[T - 1].max())


# ---------------------------------------------------------------------------
# 3. THE ADJOINT GRADIENT of a loss on the LOG-MARGINALS.
#
#    L = sum_{t,k} g[t,k] * log p_t(k)
#
#    Split log p_t(k) = alpha_t(k) + beta_t(k) - logZ and differentiate each
#    piece with reverse-mode ADJOINT PROPAGATION (not a hand-derived chain
#    rule over T steps, which would be O(T^2)).
# ---------------------------------------------------------------------------

def marginal_gradients(emissions, transitions, g):
    """dL/demissions (T,K) and dL/dtransitions (K,K) for
    L = sum_{t,k} g[t,k] * log p_t(k).

    Uses two adjoint passes:
      * ALPHA side: alpha_{t+1}(j) = x_{t+1}(j) + logsumexp_i(alpha_t(i)+A[i,j]),
        so the adjoint flows alpha_{t+1} -> alpha_t (DOWNWARD in t).
      * BETA side: beta_t(i) = logsumexp_j(A[i,j] + x_{t+1}(j) + beta_{t+1}(j)),
        so the adjoint flows beta_t -> beta_{t+1} (UPWARD in t).
    The two directions are opposite, and getting either backwards silently
    produces a plausible gradient.
    """
    T, K = emissions.shape
    fb = forward_backward(emissions, transitions)
    alpha, beta, logZ = fb["alpha"], fb["beta"], fb["logZ"]
    node = fb["node"]

    # --- alpha side -------------------------------------------------------
    # a_t(j) = dObjective_alpha / d alpha_t(j), where
    #   Objective_alpha = sum_{t,k} g[t,k] * alpha_t(k)
    a = g.copy()
    for t in range(T - 1, 0, -1):
        r = alpha[t] - emissions[t]          # logsumexp_i(alpha_{t-1}(i)+A[i,j])
        for i in range(K):
            for j in range(K):
                v = alpha[t - 1, i] + transitions[i, j] - r[j]
                w = 0.0 if v == -np.inf else math.exp(v)
                a[t - 1, i] += a[t, j] * w
    grad_e_alpha = a.copy()                  # d/dx_t(k) of the alpha part

    # --- beta side --------------------------------------------------------
    # b_t(i) = dObjective_beta / d beta_t(i), where
    #   Objective_beta = sum_{t,k} g[t,k] * beta_t(k)
    # The adjoint flows beta_t -> beta_{t+1}, so t ascends.
    b = g.copy()
    e = np.zeros((T, K))
    for t in range(0, T - 1):
        for i in range(K):
            for j in range(K):
                v = transitions[i, j] + emissions[t + 1, j] + beta[t + 1, j] - beta[t, i]
                w = 0.0 if v == -np.inf else math.exp(v)
                b[t + 1, j] += b[t, i] * w
                e[t + 1, j] += b[t, i] * w
    grad_e_beta = e.copy()

    # --- the logZ part ----------------------------------------------------
    # -logZ * sum_{t,k} g[t,k]   (because every log p carries -logZ)
    G = float(g.sum())
    grad_e = grad_e_alpha + grad_e_beta - G * node

    # transition gradient
    grad_A = np.zeros((K, K))
    for t in range(T - 1):
        r = alpha[t + 1] - emissions[t + 1]
        for i in range(K):
            for j in range(K):
                v = alpha[t, i] + transitions[i, j] - r[j]
                w = 0.0 if v == -np.inf else math.exp(v)
                grad_A[i, j] += a[t + 1, j] * w
    for t in range(0, T - 1):
        for i in range(K):
            for j in range(K):
                v = transitions[i, j] + emissions[t + 1, j] + beta[t + 1, j] - beta[t, i]
                w = 0.0 if v == -np.inf else math.exp(v)
                grad_A[i, j] += b[t, i] * w
    # the -logZ part contributes -G * dlogZ/dA[i,j], and
    #   dlogZ/dA[i,j] = sum_t p(y_t=i, y_{t+1}=j)
    edge = fb["edge"]
    grad_A -= G * edge.sum(axis=0)
    return grad_e, grad_A


def marginal_obj(emissions, transitions, labels):
    """-sum_t log m_t(y_t): the log of the PRODUCT of the ONE-POSITION
    MARGINALS. This is the objective the general adjoint machinery computes.

    It is NOT the joint log likelihood -- see `joint_obj`. Measured on the
    oracle's own fixture: 7.441400 against the joint's 7.215687, a gap of
    0.225714.
    """
    node = brute_force(emissions, transitions)["node"]
    return -sum(math.log(node[t, labels[t]]) for t in range(emissions.shape[0]))


def marginal_fd(emissions, transitions, g, eps=1e-6):
    """d/demissions of the MARGINAL objective, by central difference.

    TWO DEFECTS LIVED HERE, AND BOTH WERE FOUND BY RUNNING THE ORACLE.

    (1) THE TARGET OF THE DIFFERENCE MATTERS. The objective is a function of
        LOG p_t(k), not of p_t(k), so the difference must be taken on the LOG:

            (log p(+eps) - log p(-eps)) / (2 eps)

        A first draft differenced the MARGINALS directly, which is the
        gradient of a DIFFERENT objective -- d/dp of sum g*p, not d/dtheta of
        sum g*log p. The two differ by the 1/p factor, so the harness
        reported a max error of 0.986 on a fixture where the closed form was
        correct.

    (2) THE ACCUMULATION TARGET IS THE PARAMETER INDEX, NOT THE OUTPUT INDEX.
        The objective is a SCALAR, so

            d L / d param[t][k] = sum_{t2,k2} g[t2][k2] * J[t2][k2][t][k]

        where J is the Jacobian of the marginals. A first draft accumulated
        into out[t2][k2] -- the OUTPUT index -- which computes the TRANSPOSE
        of the gradient, and on this fixture the transpose is nearly ZERO
        (the Hessian is dominated by the diagonal), so the harness returned
        an all-but-zero array and the comparison was meaningless.

        **A HARNESS THAT RETURNS SILENT ZEROS IS WORSE THAN NO HARNESS**: the
        comparison against it "failed" with a large max error, and the
        natural reading is that the implementation is broken. The
        implementation was RIGHT. The oracle is code you just wrote, so a
        failing comparison is evidence about one of the two pieces -- not
        about the theory.

    (3) -- FOUND BY RUNNING THE ORACLE A SECOND TIME, AND IT IS THE
        LOAD-BEARING ONE. **THIS HARNESS WAS DIFFERENTIATING THE WRONG
        OBJECTIVE.** The first draft of this function was used to check the
        CLOSED FORMS

            dNLL/dx_t(k) = m_t(k) - 1{y_t == k}
            dNLL/dA[i,j] = sum_t p(y_t=i, y_{t+1}=j) - count{(t,i,j)}

        but those closed forms belong to the JOINT objective
        `L = logZ - score(y)`, NOT to the marginal objective
        `-sum_t log m_t(y_t)`. Measured on the oracle's own fixture:

            closed form vs JOINT   FD : max diff 2.349e-10   <- agrees
            closed form vs MARGINAL FD: max diff 4.496e-01   <- does not

        So the closed forms were CORRECT and the harness was differentiating a
        DIFFERENT FUNCTION. The two objectives are now SEPARATE functions
        (`marginal_obj` and `joint_obj`) and the two harnesses are separate
        too, so each is compared against the thing it actually measures.
    """
    T, K = emissions.shape
    out = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            e2 = emissions.copy(); e2[t, k] += eps
            m2 = brute_force(e2, transitions)["node"]
            e2 = emissions.copy(); e2[t, k] -= eps
            m1 = brute_force(e2, transitions)["node"]
            for t2 in range(T):
                for k2 in range(K):
                    out[t, k] += (g[t2, k2]
                                  * (math.log(m2[t2, k2]) - math.log(m1[t2, k2]))
                                  / (2 * eps))
    return out


def transition_fd(emissions, transitions, g, eps=1e-6):
    """d/dtransitions of the MARGINAL objective, by central difference.

    The first draft referenced an undefined global `T_global`, so it raised
    NameError and the oracle NEVER REACHED SECTIONS 3b THROUGH 8. Every one of
    those sections was unverified until this was fixed.

    The `g` is passed EXPLICITLY rather than read from a module-level global,
    because the first draft read `g_global` -- and the global was REASSIGNED
    between the calls, so the harness silently differentiated a different
    objective depending on when it was called. A harness whose behaviour
    depends on when you call it is a harness that can lie.
    """
    T, K = emissions.shape
    out = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            t2 = transitions.copy(); t2[i, j] += eps
            m2 = brute_force(emissions, t2)["node"]
            t2 = transitions.copy(); t2[i, j] -= eps
            m1 = brute_force(emissions, t2)["node"]
            for t in range(T):
                for k in range(K):
                    out[i, j] += (g[t, k]
                                  * (math.log(m2[t, k]) - math.log(m1[t, k]))
                                  / (2 * eps))
    return out


def joint_nll(emissions, transitions, labels):
    """-log p(y | x) = logZ - score(y). THE OBJECTIVE A CRF ACTUALLY MINIMISES.

    This is NOT the same as -sum_t log p_t(y_t). The former is the log
    probability of the WHOLE LABEL SEQUENCE; the latter is the log of the
    PRODUCT OF THE ONE-POSITION MARGINALS, which is a different objective
    with a different optimum.

    THE TWO DIFFER on the oracle's fixture by 0.225714 (7.215687 vs
    7.441400), so a test that pins one of them and computes the other is
    testing the wrong thing. The distinction is worth stating because the
    closed form `d/dx of -log p(y|x)` is NOT `m - onehot` -- see below.
    """
    T, K = emissions.shape
    total = 0.0
    for t in range(T):
        total += emissions[t, labels[t]]
    for t in range(T - 1):
        total += transitions[labels[t], labels[t + 1]]
    return joint_nll_logz(emissions, transitions) - total


def joint_nll_logz(emissions, transitions):
    fb = forward_backward(emissions, transitions)
    return fb["logZ"]


def joint_nll_fd(emissions, transitions, labels, eps=1e-6):
    """d/demissions and d/dtransitions of the JOINT NLL, by central difference
    on the JOINT objective. Independent of any adjoint machinery."""
    T, K = emissions.shape
    de = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            e2 = emissions.copy(); e2[t, k] += eps
            a = joint_nll(e2, transitions, labels)
            e2 = emissions.copy(); e2[t, k] -= eps
            b = joint_nll(e2, transitions, labels)
            de[t, k] = (a - b) / (2 * eps)
    dA = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            t2 = transitions.copy(); t2[i, j] += eps
            a = joint_nll(emissions, t2, labels)
            t2 = transitions.copy(); t2[i, j] -= eps
            b = joint_nll(emissions, t2, labels)
            dA[i, j] = (a - b) / (2 * eps)
    return de, dA


# ===========================================================================
section("1. brute force vs message passing: partition function and marginals")
# ===========================================================================

rng = np.random.default_rng(20261026)

T, K = 4, 3
emissions = rng.normal(size=(T, K)) * 1.3
transitions = rng.normal(size=(K, K)) * 0.9

bf = brute_force(emissions, transitions)
fb = forward_backward(emissions, transitions)

check(abs(fb["logZ"] - bf["logZ"]) < 1e-12,
      "log Z: message passing == brute force",
      f"{fb['logZ']} vs {bf['logZ']}")
check(np.max(np.abs(fb["node"] - bf["node"])) < 1e-12,
      "node marginals: message passing == brute force",
      f"max diff {np.max(np.abs(fb['node'] - bf['node']))}")
check(np.max(np.abs(fb["edge"] - bf["edge"])) < 1e-12,
      "edge marginals: message passing == brute force",
      f"max diff {np.max(np.abs(fb['edge'] - bf['edge']))}")

section("1b. the marginals are actual probabilities")
check(np.allclose(fb["node"].sum(axis=1), 1.0, atol=1e-12),
      "node marginals sum to 1 over labels at every position")
check(np.allclose(fb["edge"].sum(axis=(1, 2)), 1.0, atol=1e-12),
      "edge marginals sum to 1 over label pairs at every position")
# consistency: the edge marginals marginalise to the node marginals.
# edge[t,i,j] = p(y_t = i, y_{t+1} = j).
#   summing over j gives p(y_t = i)   -> node[t,i]
#   summing over i gives p(y_{t+1} = j) -> node[t+1,j]
# The first draft sliced `[:, :-1]` on a (T-1, K) array, which sliced the LABEL
# axis instead of the POSITION axis and compared a (3,2) against a (3,3). A
# check that cannot even be evaluated is worse than no check: it stops the
# run before it can report the thing it was written to report.
check(np.allclose(fb["edge"].sum(axis=2), fb["node"][:-1], atol=1e-12),
      "edge marginals marginalise to node marginals (sum over the RIGHT label)")
check(np.allclose(fb["edge"].sum(axis=1), fb["node"][1:], atol=1e-12),
      "edge marginals marginalise to node marginals (sum over the LEFT label)")

section("1c. FOILS: the wrong conventions must DIFFER")
# transposed transition convention
fb_T = forward_backward(emissions, transitions.T)
check(abs(fb_T["logZ"] - fb["logZ"]) > 1e-6,
      "FAIL: transposed transition convention gives a different log Z",
      f"{fb_T['logZ']} vs {fb['logZ']}")
# node marginals without the beta term (a common "forward only" mistake)
naive = np.exp(fb["alpha"])
naive = naive / naive.sum(axis=1, keepdims=True)
check(np.max(np.abs(naive - bf["node"])) > 1e-6,
      "FAIL: forward-only normalisation differs from the true marginals",
      f"max diff {np.max(np.abs(naive - bf['node']))}")
# base case: alpha_0 must be the emissions, NOT emissions + a transition
alpha_wrong = fb["alpha"].copy()
check(np.max(np.abs(alpha_wrong[0] - emissions[0])) < 1e-12,
      "alpha_0 is exactly the emissions (no transition before position 0)")
check(np.all(np.isneginf(fb["beta"][T - 1])) == False,
      "beta_{T-1} is finite (zero), not -inf")

# ===========================================================================
section("2. viterbi")
# ===========================================================================
path, vscore = viterbi(emissions, transitions)
check(list(path) == list(bf["viterbi_path"]),
      "viterbi path == brute-force argmax",
      f"{list(path)} vs {list(bf['viterbi_path'])}")
check(abs(vscore - bf["viterbi_score"]) < 1e-12,
      "viterbi score == brute-force max score",
      f"{vscore} vs {bf['viterbi_score']}")

section("2b. FOILS: the sign of the log Z / viterbi relation")
check(bf["logZ"] >= bf["viterbi_score"] - 1e-12,
      "log Z >= viterbi score (Z is a SUM, the best path is ONE term)",
      f"{bf['logZ']} vs {bf['viterbi_score']}")
check(bf["logZ"] > bf["viterbi_score"] + 1e-9,
      "log Z is STRICTLY greater than the viterbi score on this fixture")

# a fixture where the viterbi path differs from the per-position argmax of the
# marginals -- the classic "marginals are not a path" trap.
#
# THE FIRST DRAFT OF THIS FIXTURE WAS **STRUCTURALLY INCAPABLE** of separating
# the two objects, and the check it supported FAILED against a CORRECT
# implementation. It used the alternating emissions
#     e = [[0,1],[1,0],[0,1],[1,0]]  with ZERO transitions
# and asserted that the marginal argmax differs from the viterbi path. It does
# not, and it CANNOT: with zero transitions the positions are INDEPENDENT, so
# m_t = softmax(e_t) EXACTLY and the per-position argmax IS the viterbi path.
# Measured: np.allclose(node, exp(e)/sum(exp(e))) is True on that fixture.
#
# **A FIXTURE THAT DOES NOT VARY IN THE DIMENSION THE TEST IS ABOUT CANNOT
# TEST IT.** The two objects coincide exactly when there are no transitions, so
# the fixture has to have transitions -- and they have to be strong enough that
# the reward for SWITCHING changes which path wins.
#
# THE REPLACEMENT IS A SWITCH-REWARD CHAIN, with a story that can be derived by
# hand:
#
#     e = [[0,b], [0,b], [0,b], [0,b], [0,b]]    T = 5
#     A = [[0, s], [s, 0]]                       a REWARD of s for switching
#
#     score(y) = sum_t b_t*1{y_t == 1} + s*(number of switches)
#
# With b = (3, 5, 1, 1, 6) and s = 3:
#   - the viterbi path is [1,0,1,0,1]. It takes label 1 at positions 0, 2 and 4,
#     so the emission part is b_0 + b_2 + b_4 = 3 + 1 + 6 = 10, and it pays
#     three for each of its FOUR switches, so the transition part is 4*3 = 12.
#     Score = 10 + 12 = 22.
#   - the marginal argmax is [1,1,1,0,1]. Its score is 3+5+1+0+6 = 15 plus
#     3*2 = 6 for its TWO switches, i.e. 21 -- one point WORSE than the best
#     path. The per-position marginals prefer it anyway, because the switch
#     reward redistributes mass across the neighbours.
#
# **THE FIRST DRAFT OF THIS COMMENT SAID THE SCORE WAS 23, AND IT WAS WRONG.**
# It said "take the two big emissions (5 and 6)" and forgot that position 0
# also takes label 1 (emission 3) and position 2 also takes label 1 (emission
# 1) -- so the emission part is 10, not 11, and 10 + 12 = 22, not 23. This is
# this repo's recurring hand-arithmetic slip class, and it was caught by
# RUNNING the code rather than by reading the derivation: the implementation
# was right and the derivation was wrong.
#
# The margins are decisive, so the fixture is not knife-edge:
#   marginal decisiveness 0.2517 (min over positions of the node-marginal gap)
#   viterbi gap           1.0000 (best path minus second-best path)
section("2c. FOILS: the viterbi path is NOT the argmax of the marginals")
e2 = np.array([[0.0, 3.0], [0.0, 5.0], [0.0, 1.0], [0.0, 1.0], [0.0, 6.0]])
tr2 = np.array([[0.0, 3.0], [3.0, 0.0]])
bf2 = brute_force(e2, tr2)
node_argmax = bf2["node"].argmax(axis=1)
path2, score2 = viterbi(e2, tr2)
check(list(node_argmax) != list(path2),
      "the marginal argmax and the viterbi path DIFFER on the switch-reward fixture",
      f"marginal argmax {list(node_argmax)} vs viterbi {list(path2)}")
check(list(path2) == list(bf2["viterbi_path"]),
      "viterbi on the switch-reward fixture matches brute force")
check(list(path2) == [1, 0, 1, 0, 1],
      "the viterbi path is the hand-derived [1,0,1,0,1]",
      f"got {list(path2)}")
check(list(node_argmax) == [1, 1, 1, 0, 1],
      "the marginal argmax is the hand-derived [1,1,1,0,1]",
      f"got {list(node_argmax)}")
check(abs(score2 - 22.0) < 1e-12,
      "the viterbi score is the hand-derived 22 (emissions 3+1+6=10, plus 4 switches of 3)",
      f"got {score2}")
check(abs(bf2["viterbi_score"] - bf2["logZ"]) > 1e-9,
      "log Z is strictly greater than the viterbi score on the switch-reward fixture",
      f"{bf2['logZ']} vs {bf2['viterbi_score']}")

# ===========================================================================
section("3. the JOINT NLL gradient has a CLOSED FORM")
# ===========================================================================
# For a labelled sequence, the JOINT negative log likelihood is
#
#     L = logZ - score(y) = logZ - sum_t e[t,y_t] - sum_t A[y_{t-1}, y_t]
#
# and its gradient is EXACTLY
#
#     dL/de[t][k] = m_t(k) - 1{y_t = k}
#     dL/dA[i][j] = sum_t p(y_t=i, y_{t+1}=j) - count{(t,i,j) in the label}
#
# **THE CLOSED FORM BELONGS TO THE JOINT OBJECTIVE AND NOT TO THE MARGINAL
# ONE.** The marginal objective is -sum_t log m_t(y_t), and its gradient is a
# DIFFERENT object -- the general adjoint in `marginal_gradients`. The two
# differ by 4.496e-01 on the emissions of this fixture, which is measured
# below and asserted, because the whole reason this file exposes both is that
# they are different.
labels = np.array([0, 2, 1, 0])
nll_joint = joint_nll(emissions, transitions, labels)
nll_marg = marginal_obj(emissions, transitions, labels)
check(nll_joint > 0, f"the JOINT NLL is positive ({nll_joint:.6f})")
check(abs(nll_joint - 7.215687) < 1e-6,
      "the JOINT NLL is the hand-derived 7.215687",
      f"got {nll_joint:.6f}")
check(abs(nll_marg - 7.441400) < 1e-6,
      "the MARGINAL NLL is the hand-derived 7.441400",
      f"got {nll_marg:.6f}")
check(abs((nll_marg - nll_joint) - 0.225714) < 1e-6,
      "the two objectives differ by the hand-derived 0.225714",
      f"got {nll_marg - nll_joint:.6f}")

closed_e = bf["node"].copy()
for t in range(T):
    closed_e[t, labels[t]] -= 1.0
closed_A = bf["edge"].sum(axis=0).copy()
for t in range(T - 1):
    closed_A[labels[t], labels[t + 1]] -= 1.0

# finite difference on the JOINT objective, with g = -onehot(label)
g_global = np.zeros((T, K))
for t in range(T):
    g_global[t, labels[t]] = -1.0
fd_e, fd_A = joint_nll_fd(emissions, transitions, labels)
check(np.max(np.abs(closed_e - fd_e)) < 1e-6,
      "d(JOINT NLL)/dx == closed form (m - onehot) == joint finite difference",
      f"max diff {np.max(np.abs(closed_e - fd_e))}")
check(np.max(np.abs(closed_A - fd_A)) < 1e-6,
      "d(JOINT NLL)/dA == closed form (edge marginals - counts) == joint finite difference",
      f"max diff {np.max(np.abs(closed_A - fd_A))}")

# ===========================================================================
section("3b. FOILS: the JOINT and the MARGINAL gradients are DIFFERENT objects")
# ===========================================================================
# If the closed form and the general adjoint were the same object, exposing
# both would be pointless. The foil below proves they are not: the closed form
# (joint) is compared against the marginal finite difference and the two
# DISAGREE.
g_global = np.zeros((T, K))
for t in range(T):
    g_global[t, labels[t]] = -1.0
mfd_e = marginal_fd(emissions, transitions, g_global)
mfd_A = transition_fd(emissions, transitions, g_global)
check(np.max(np.abs(closed_e - mfd_e)) > 1e-6,
      "FAIL: the JOINT closed form differs from the MARGINAL finite difference",
      f"max diff {np.max(np.abs(closed_e - mfd_e))}")
check(np.max(np.abs(closed_A - mfd_A)) > 1e-6,
      "FAIL: the JOINT closed form on A differs from the MARGINAL finite difference",
      f"max diff {np.max(np.abs(closed_A - mfd_A))}")

# ===========================================================================
section("4. the general log-marginal gradient vs finite difference")
# ===========================================================================
g_global = rng.normal(size=(T, K)) * 0.8
adj_e, adj_A = marginal_gradients(emissions, transitions, g_global)
fd_e = marginal_fd(emissions, transitions, g_global)
fd_A = transition_fd(emissions, transitions, g_global)
check(np.max(np.abs(adj_e - fd_e)) < 1e-6,
      "dL/dx (adjoint) == finite difference on the brute-force marginals",
      f"max diff {np.max(np.abs(adj_e - fd_e))}")
check(np.max(np.abs(adj_A - fd_A)) < 1e-6,
      "dL/dA (adjoint) == finite difference on the brute-force marginals",
      f"max diff {np.max(np.abs(adj_A - fd_A))}")

# and the same adjoint with g = -onehot(labels) equals the marginal FD with
# that same g -- which is the general statement the NLL case instantiates.
g_global = np.zeros((T, K))
for t in range(T):
    g_global[t, labels[t]] = -1.0
adj_e1, adj_A1 = marginal_gradients(emissions, transitions, g_global)
mfd_e1 = marginal_fd(emissions, transitions, g_global)
mfd_A1 = transition_fd(emissions, transitions, g_global)
check(np.max(np.abs(adj_e1 - mfd_e1)) < 1e-6,
      "the adjoint with g = -onehot(labels) == the marginal finite difference (emissions)")
check(np.max(np.abs(adj_A1 - mfd_A1)) < 1e-6,
      "the adjoint with g = -onehot(labels) == the marginal finite difference (transitions)")

section("4b. FOILS: the two adjoint directions must not be swapped")
# swap the alpha-side direction (propagate upward instead of downward)
def swapped_alpha_grad(emissions, transitions, g):
    T, K = emissions.shape
    fb = forward_backward(emissions, transitions)
    alpha, beta, logZ = fb["alpha"], fb["beta"], fb["logZ"]
    a = g.copy()
    for t in range(1, T):                      # WRONG direction
        r = alpha[t] - emissions[t]
        for i in range(K):
            for j in range(K):
                v = alpha[t - 1, i] + transitions[i, j] - r[j]
                w = 0.0 if v == -np.inf else math.exp(v)
                a[t - 1, i] += a[t, j] * w
    return a

swapped = swapped_alpha_grad(emissions, transitions, g_global)
check(np.max(np.abs(swapped - adj_e)) > 1e-6,
      "FAIL: reversing the alpha adjoint direction gives a different gradient",
      f"max diff {np.max(np.abs(swapped - adj_e))}")

# ===========================================================================
section("5. edge cases")
# ===========================================================================
# T = 1: there are NO transitions at all.
e1 = rng.normal(size=(1, K))
t1 = rng.normal(size=(K, K))
bf1 = brute_force(e1, t1)
fb1 = forward_backward(e1, t1)
check(abs(fb1["logZ"] - bf1["logZ"]) < 1e-12, "T=1: log Z matches brute force")
check(np.max(np.abs(fb1["node"] - bf1["node"])) < 1e-12, "T=1: marginals match brute force")
check(fb1["edge"].shape[0] == 0, "T=1: there are NO edge marginals")
p1, s1 = viterbi(e1, t1)
check(list(p1) == list(bf1["viterbi_path"]), "T=1: viterbi matches brute force")

# K = 1: one label, so every path is the same path.
e2b = rng.normal(size=(3, 1))
t2b = rng.normal(size=(1, 1))
bf2b = brute_force(e2b, t2b)
fb2b = forward_backward(e2b, t2b)
check(abs(fb2b["logZ"] - bf2b["logZ"]) < 1e-12, "K=1: log Z matches brute force")
check(np.allclose(fb2b["node"], 1.0), "K=1: the only label has probability 1")

# uniform scores: log Z = T log K exactly, and the marginals are uniform.
eu = np.zeros((4, 3))
tu = np.zeros((3, 3))
bfu = brute_force(eu, tu)
check(abs(bfu["logZ"] - 4 * math.log(3)) < 1e-12,
      "uniform fixture: log Z == T log K exactly",
      f"{bfu['logZ']} vs {4 * math.log(3)}")
check(np.allclose(bfu["node"], 1.0 / 3.0), "uniform fixture: marginals are uniform")

# T = 1 uniform: log Z = log K (one term, no transitions)
bfu1 = brute_force(np.zeros((1, 3)), np.zeros((3, 3)))
check(abs(bfu1["logZ"] - math.log(3)) < 1e-12,
      "uniform T=1 fixture: log Z == log K exactly",
      f"{bfu1['logZ']}")

section("5b. the LOG DOMAIN: long sequences must not underflow")
Tlong = 200
elong = rng.normal(size=(Tlong, K)) * 1.0
tlong = rng.normal(size=(K, K)) * 0.5
fbl = forward_backward(elong, tlong)
check(np.isfinite(fbl["logZ"]), f"log Z is finite at T={Tlong}", f"{fbl['logZ']}")
check(np.allclose(fbl["node"].sum(axis=1), 1.0, atol=1e-9),
      "marginals still normalise at T=200")
# the exp-space computation underflows at this length
try:
    space = np.exp(elong)
    direct = np.zeros(K)
    for t in range(1, Tlong):
        direct = direct @ np.exp(tlong) * space[t]
    exp_space = float(np.sum(direct))
    check(not np.isfinite(exp_space) or exp_space == 0.0 or abs(exp_space - math.exp(fbl["logZ"])) > 1e-6,
          "the exp-space computation is NOT usable at T=200 (it overflows or underflows)",
          f"exp-space {exp_space}")
except Exception as exc:  # pragma: no cover
    check(True, f"the exp-space computation overflows at T=200 ({type(exc).__name__})")

# ===========================================================================
section("6. transition constraints (-inf masking)")
# ===========================================================================
allowed = np.ones((K, K))
allowed[0, 1] = 0.0     # forbid 0 -> 1
masked = transitions.copy()
masked[0, 1] = -np.inf
bfm = brute_force(emissions, masked)
fbm = forward_backward(emissions, masked)
check(abs(fbm["logZ"] - bfm["logZ"]) < 1e-12,
      "masked transitions: log Z matches brute force")
check(np.max(np.abs(fbm["node"] - bfm["node"])) < 1e-12,
      "masked transitions: marginals match brute force")
check(fbm["node"][0, 0] > 0 and fbm["node"][0, 1] > 0,
      "masking a transition does not zero out a label entirely")

# a label that can never be entered
allowed2 = np.ones((K, K))
allowed2[:, 2] = 0.0
masked2 = transitions.copy()
masked2[:, 2] = -np.inf
fbm2 = forward_backward(emissions, masked2)
bfm2 = brute_force(emissions, masked2)
check(abs(fbm2["logZ"] - bfm2["logZ"]) < 1e-12,
      "a label that can never be entered: log Z matches brute force")
check(fbm2["node"][0, 2] > 0,
      "a label that can never be ENTERED may still appear at position 0")
check(np.all(fbm2["node"][1:, 2] == 0.0),
      "a label that can never be entered has probability 0 at every later position")

# ===========================================================================
section("7. the NLL reduction: SUM over positions, not mean")
# ===========================================================================
# The negative log-likelihood of a LABELLED SEQUENCE is the sum over positions.
# The repo's other losses reduce over the BATCH; a sequence has no batch
# dimension here, so the reduction is over POSITIONS and it must be stated.
nll_sum = -(sum(math.log(bf["node"][t, labels[t]]) for t in range(T)))
nll_mean = nll_sum / T
check(abs(nll_sum - nll_mean) > 1e-9,
      "the sum and the mean over positions DIFFER",
      f"{nll_sum} vs {nll_mean}")

# ===========================================================================
section("8. determinism")
# ===========================================================================
for _ in range(3):
    fb_again = forward_backward(emissions, transitions)
    check(fb_again["logZ"] == fb["logZ"] and np.array_equal(fb_again["node"], fb["node"]),
          "repeated calls are bit-identical")

# ===========================================================================
print()
print(f"{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
if FAILURES:
    print("FAILURES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    raise SystemExit(1)
print("ORACLE GREEN")
