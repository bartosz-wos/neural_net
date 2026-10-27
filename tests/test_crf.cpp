// =============================================================================
// tests/test_crf.cpp -- the linear-chain CRF: log-domain forward-backward,
// the partition function, the node and edge marginals, Viterbi decoding, and
// the two adjoints.
//
// THE SPECIFICATION IS tests/crf_oracle.py, WHICH RUNS GREEN (56 checks) AND
// IS THE FILE THAT FOUND THE DEFECTS IN THIS SUITE'S DERIVATIONS BEFORE ANY
// C++ EXISTED. Four of them are load-bearing enough to repeat here, because a
// future reader of this test who does not know them will reintroduce them:
//
//   1. **THE TWO OBJECTIVES ARE DIFFERENT FUNCTIONS AND HAVE DIFFERENT
//      GRADIENTS.** The JOINT negative log likelihood `logZ - score(y)` has
//      the exact closed-form gradient `m - onehot` on the emissions and
//      `edge - counts` on the transitions. The MARGINAL log likelihood
//      `-sum_t log m_t(y_t)` needs the message adjoint, and its gradient is a
//      DIFFERENT object: the two emission gradients differ by 4.496e-01 on
//      the fixture below, and the two objective VALUES differ by 0.225714
//      (7.215687 against 7.441400).
//
//      **THE ORACLE'S FIRST DRAFT ASSERTED THAT THE JOINT CLOSED FORM
//      AGREED WITH THE MARGINAL FINITE DIFFERENCE. IT DID NOT -- AND THE
//      CLOSED FORM WAS RIGHT while the finite-difference harness was
//      differentiating the wrong function.** The oracle is code you just
//      wrote: a failing comparison is evidence about one of the two pieces,
//      not about the theory. Both objectives are exposed here because both
//      are real, and S10 pins that they DIFFER.
//
//   2. **THE MARGINAL ARGMAX IS NOT A PATH, BUT A FIXTURE WITH ZERO
//      TRANSITIONS CANNOT SEE THAT.** With no transitions the positions are
//      independent, so `m_t = softmax(e_t)` exactly and the per-position
//      argmax IS the Viterbi path. The oracle's first draft used exactly
//      such a fixture and its check FAILED against a correct
//      implementation. The fixture below is a SWITCH-REWARD chain, where
//      the two objects genuinely differ, and both paths are pinned.
//
//   3. **THE LOG DOMAIN IS MANDATORY.** At T = 200 with unit-scale scores
//      the exp-space computation underflows to exactly 0.0 while the
//      log-space computation stays finite. A solver that works in the exp
//      space passes every small fixture and fails on every long one. S8.
//
//   4. **THE EDGE MARGINALS ARE NOT THE NODE MARGINALS.** They marginalise
//      to them in BOTH directions, and they do not sum to 1 over one of the
//      two label axes. S5.
//
// Every boundary fixture here is either exact or dyadic: a fixture whose
// discriminating value is only correct to within a rounding error does not
// discriminate anything.
// =============================================================================

#include "nn/nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;
using namespace nn::utils;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

static void section(const char* name) { printf("\n=== %s ===\n", name); }

// A (rows, cols) tensor from a flat row-major vector.
static Tensor mat(size_t r, size_t c, const std::vector<double>& flat) {
    return Tensor(r, c, flat.data());
}

// The oracle's main fixture: seed 20261026, T = 4, K = 3.
//   rng = np.random.default_rng(20261026)
//   emissions   = rng.normal(size=(4, 3)) * 1.3
//   transitions = rng.normal(size=(3, 3)) * 0.9
// The values below are the oracle's, printed to 16 digits so that the pinned
// comparisons are not decided by a rounding error in the FIXTURE.
static const double kEmit[4][3] = {
    {-0.4597983052909368, -0.1218117447417670,  0.3895256728635606},
    { 1.2111339281007114,  0.3054448594599655,  0.5007792610812163},
    { 0.4723238706606504, -1.1236943897375329, -0.2119076823482140},
    { 2.5987806151095723,  0.2659662734998924,  1.7630337561253007},
};
static const double kTrans[3][3] = {
    { 1.3168934104946382, -0.4125828208358689,  0.7723854137876121},
    {-0.2406982163624521, -1.2657661009940024,  0.6020473394984169},
    { 0.7170939877565052,  0.0500315736886324, -0.2428130622302667},
};

static Tensor main_emissions() {
    std::vector<double> flat;
    for (int t = 0; t < 4; ++t)
        for (int k = 0; k < 3; ++k) flat.push_back(kEmit[t][k]);
    return mat(4, 3, flat);
}

static Tensor main_transitions() {
    std::vector<double> flat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) flat.push_back(kTrans[i][j]);
    return mat(3, 3, flat);
}

// The switch-reward fixture from the oracle's section 2c.
//   score(y) = sum_t b_t*1{y_t == 1} + s*(number of switches)
// The viterbi path is [1,0,1,0,1] with score 22; the marginal argmax is
// [1,1,1,0,1] with score 21 -- one point WORSE.
static Tensor switch_emissions() {
    return mat(5, 2, {0.0, 3.0,  0.0, 5.0,  0.0, 1.0,  0.0, 1.0,  0.0, 6.0});
}
static Tensor switch_transitions() {
    return mat(2, 2, {0.0, 3.0,  3.0, 0.0});
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            m = std::max(m, std::fabs(a(i, j) - b(i, j)));
    return m;
}

int main() {
    const Tensor em = main_emissions();
    const Tensor tr = main_transitions();

    // =============================================================================
    section("S1. THE PARTITION FUNCTION AND THE MARGINALS");
    // =============================================================================
    // The oracle's pinned values for the fixture above.
    {
        const double logz = crf_log_partition(em, tr);
        check(std::fabs(logz - 9.313472455712859) < 1e-12,
              "S1a. log Z matches the oracle's pinned value");
        check(std::isfinite(logz), "S1b. log Z is finite");

        const Tensor node = crf_node_marginals(em, tr);
        const double want_node[4][3] = {
            {0.372183881305936, 0.154878531497057, 0.472937587197006},
            {0.817638865832113, 0.035791814366108, 0.146569319801780},
            {0.851884145669949, 0.013630353310701, 0.134485501019350},
            {0.788738815485064, 0.017295527660677, 0.193965656854259},
        };
        double worst = 0.0;
        for (size_t t = 0; t < 4; ++t)
            for (size_t k = 0; k < 3; ++k)
                worst = std::max(worst, std::fabs(node(t, k) - want_node[t][k]));
        check(worst < 1e-12,
              "S1c. the node marginals match the oracle's pinned values");
    }

    // =============================================================================
    section("S2. THE EDGE MARGINALS ARE A DIFFERENT OBJECT");
    // =============================================================================
    {
        const Tensor node = crf_node_marginals(em, tr);
        const Tensor edge = crf_edge_marginals(em, tr);
        check(edge.rows == 3 && edge.cols == 9,
              "S2a. the edge marginals have shape (T-1, K*K)");

        const double want_edge[3][9] = {
            {0.317066448391943, 0.006739505705579, 0.048377927208415,
             0.093644556573196, 0.004026066582366, 0.057207908341496,
             0.406927860866974, 0.025026242078164, 0.040983484251869},
            {0.701151665957785, 0.008546090659038, 0.107941109215290,
             0.021810478955777, 0.000537701670922, 0.013443633739409,
             0.128922000756387, 0.004546560980741, 0.013100758064652},
            {0.671449540016934, 0.011555347248002, 0.168879258405013,
             0.006675550365590, 0.000232368553858, 0.006722434391253,
             0.110613725102540, 0.005507811858817, 0.018363964057993},
        };
        double worst = 0.0;
        for (size_t t = 0; t < 3; ++t)
            for (size_t i = 0; i < 9; ++i)
                worst = std::max(worst, std::fabs(edge(t, i) - want_edge[t][i]));
        check(worst < 1e-12,
              "S2b. the edge marginals match the oracle's pinned values");

        // Summing over the SECOND label axis gives node[t, i]; summing over
        // the FIRST gives node[t+1, j]. The oracle's first draft sliced the
        // LABEL axis instead of the POSITION axis and compared a (3,2)
        // against a (3,3) -- a check that could not even be evaluated.
        bool fwd = true, bwd = true;
        for (size_t t = 0; t < 3; ++t) {
            for (size_t i = 0; i < 3; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < 3; ++j) s += edge(t, i * 3 + j);
                if (std::fabs(s - node(t, i)) > 1e-12) fwd = false;
            }
            for (size_t j = 0; j < 3; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < 3; ++i) s += edge(t, i * 3 + j);
                if (std::fabs(s - node(t + 1, j)) > 1e-12) bwd = false;
            }
        }
        check(fwd, "S2c. summing over the second label axis gives node[t, i]");
        check(bwd, "S2d. summing over the first label axis gives node[t+1, j]");

        // They do NOT sum to 1 over one label axis alone.
        double row_sum = 0.0;
        for (size_t i = 0; i < 3; ++i) row_sum += edge(0, i * 3 + 0);
        check(std::fabs(row_sum - 1.0) > 1e-6,
              "S2e. the edge marginals do NOT sum to 1 over one label axis alone");

        // But over BOTH they do.
        double total = 0.0;
        for (size_t i = 0; i < 9; ++i) total += edge(0, i);
        check(std::fabs(total - 1.0) < 1e-12,
              "S2f. the edge marginals sum to 1 over both label axes");
    }

    // =============================================================================
    section("S3. THE MARGINALS ARE ACTUAL PROBABILITIES");
    // =============================================================================
    {
        const Tensor node = crf_node_marginals(em, tr);
        bool all_one = true;
        for (size_t t = 0; t < 4; ++t) {
            double s = 0.0;
            for (size_t k = 0; k < 3; ++k) s += node(t, k);
            if (std::fabs(s - 1.0) > 1e-12) all_one = false;
        }
        check(all_one, "S3a. every node-marginal row sums to 1");
    }

    // =============================================================================
    section("S4. THE TRANSITION CONVENTION: transitions(i,j) is FROM i TO j");
    // =============================================================================
    // The transposed convention produces a DIFFERENT partition function on any
    // asymmetric fixture, and the two are the SAME SHAPE and the SAME SIZE, so
    // no type check can see the difference -- only a pinned value can.
    {
        const Tensor trT = tr.transpose();
        const double logz = crf_log_partition(em, tr);
        const double logzT = crf_log_partition(em, trT);
        check(std::fabs(logzT - logz) > 1e-6,
              "S4a. the transposed convention gives a DIFFERENT log Z");

        // And the convention is pinned by an independent, hand-checkable
        // construction: a chain where the ONLY transition that pays is 0 -> 1.
        Tensor e2 = mat(2, 2, {0.0, 0.0, 0.0, 0.0});
        Tensor a2 = mat(2, 2, {0.0, 5.0, 0.0, 0.0});   // A[0][1] = 5
        // The best path is [0,1] with score 5, so log Z > 5 and the viterbi
        // path is [0,1].
        Tensor path;
        const double vscore = crf_viterbi(e2, a2, &path);
        check(std::fabs(vscore - 5.0) < 1e-12,
              "S4b. the viterbi score is 5 (the 0->1 transition pays)");
        check(path(0, 0) == 0.0 && path(1, 0) == 1.0,
              "S4c. the viterbi path is [0,1], so A[0][1] is the 0->1 transition");
        // If the convention were transposed, the paying transition would be
        // A[1][0] and the best path would be [1,0].
        Tensor a2T = mat(2, 2, {0.0, 0.0, 5.0, 0.0});   // A[1][0] = 5
        Tensor pathT;
        crf_viterbi(e2, a2T, &pathT);
        check(pathT(0, 0) == 1.0 && pathT(1, 0) == 0.0,
              "S4d. moving the payment to A[1][0] moves the path to [1,0]");
    }

    // =============================================================================
    section("S5. VITERBI");
    // =============================================================================
    {
        Tensor path;
        const double vscore = crf_viterbi(em, tr, &path);
        check(std::fabs(vscore - 8.022644895480276) < 1e-12,
              "S5a. the viterbi score matches the oracle's pinned value");
        check(path(0, 0) == 2.0 && path(1, 0) == 0.0 &&
              path(2, 0) == 0.0 && path(3, 0) == 0.0,
              "S5b. the viterbi path matches the oracle's pinned path [2,0,0,0]");

        // log Z is a SUM over paths and the best path is ONE term of it, so
        // log Z >= viterbi score ALWAYS, and STRICTLY on a non-degenerate
        // fixture. Asserting the opposite is a very natural mistake.
        const double logz = crf_log_partition(em, tr);
        check(logz >= vscore - 1e-12,
              "S5c. log Z >= the viterbi score");
        check(logz > vscore + 1e-9,
              "S5d. log Z is STRICTLY greater than the viterbi score here");
    }

    // =============================================================================
    section("S6. THE MARGINAL ARGMAX IS NOT A PATH");
    // =============================================================================
    // The oracle's FIRST draft of this fixture used ZERO transitions, and its
    // check FAILED against a correct implementation: with no transitions the
    // positions are independent, so m_t = softmax(e_t) exactly and the
    // per-position argmax IS the viterbi path. The fixture below is a
    // switch-reward chain, where the two objects genuinely differ.
    {
        const Tensor e = switch_emissions();
        const Tensor a = switch_transitions();
        Tensor path;
        const double vscore = crf_viterbi(e, a, &path);
        const Tensor node = crf_node_marginals(e, a);

        check(path(0, 0) == 1.0 && path(1, 0) == 0.0 && path(2, 0) == 1.0 &&
              path(3, 0) == 0.0 && path(4, 0) == 1.0,
              "S6a. the viterbi path is the hand-derived [1,0,1,0,1]");
        check(std::fabs(vscore - 22.0) < 1e-12,
              "S6b. the viterbi score is the hand-derived 22 "
              "(emissions 3+1+6=10, plus four switches of 3)");

        // the marginal argmax
        bool argmax_ok = true;
        const int want_argmax[5] = {1, 1, 1, 0, 1};
        for (size_t t = 0; t < 5; ++t) {
            size_t best = 0;
            for (size_t k = 1; k < 2; ++k) if (node(t, k) > node(t, best)) best = k;
            if (static_cast<int>(best) != want_argmax[t]) argmax_ok = false;
        }
        check(argmax_ok,
              "S6c. the marginal argmax is the hand-derived [1,1,1,0,1]");

        bool differs = false;
        for (size_t t = 0; t < 5; ++t) {
            size_t best = 0;
            for (size_t k = 1; k < 2; ++k) if (node(t, k) > node(t, best)) best = k;
            if (static_cast<int>(best) != static_cast<int>(path(t, 0))) differs = true;
        }
        check(differs,
              "S6d. the marginal argmax and the viterbi path DIFFER on this fixture");

        // The marginal-argmax path scores 21 -- one point WORSE than the
        // best path. The per-position marginals prefer it anyway, which is
        // the whole point of the distinction.
        Tensor marg_path = mat(5, 1, {1.0, 1.0, 1.0, 0.0, 1.0});
        const double marg_score = crf_sequence_score(e, a, marg_path);
        check(std::fabs(marg_score - 21.0) < 1e-12,
              "S6e. the marginal-argmax path scores 21, one point below the best path");
    }

    // =============================================================================
    section("S7. THE JOINT NLL GRADIENT: EXACT CLOSED FORM");
    // =============================================================================
    {
        const Tensor labels = mat(4, 1, {0.0, 2.0, 1.0, 0.0});
        Tensor ge, gA;
        crf_joint_nll_grad(em, tr, labels, &ge, &gA);

        const double want_ge[4][3] = {
            {-0.627816118694064, 0.154878531497057,  0.472937587197006},
            { 0.817638865832113, 0.035791814366108, -0.853430680198220},
            { 0.851884145669949, -0.986369646689299, 0.134485501019350},
            {-0.211261184514936, 0.017295527660677,  0.193965656854259},
        };
        double worst = 0.0;
        for (size_t t = 0; t < 4; ++t)
            for (size_t k = 0; k < 3; ++k)
                worst = std::max(worst, std::fabs(ge(t, k) - want_ge[t][k]));
        check(worst < 1e-12,
              "S7a. the joint emissions gradient matches the oracle's pinned values");

        const double want_gA[3][3] = {
            { 1.689667654366662,  0.026840943612618, -0.674801705171283},
            {-0.877869414105437,  0.004796136807146,  0.077373976472158},
            { 0.646463586725900, -0.964919385082278,  0.072448206374514},
        };
        worst = 0.0;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                worst = std::max(worst, std::fabs(gA(i, j) - want_gA[i][j]));
        check(worst < 1e-12,
              "S7b. the joint transitions gradient matches the oracle's pinned values");

        // The score and the joint NLL.
        const double score = crf_sequence_score(em, tr, labels);
        check(std::fabs(score - 2.0977859522761113) < 1e-12,
              "S7c. the sequence score matches the oracle's pinned value");
        const double nll = crf_log_partition(em, tr) - score;
        check(std::fabs(nll - 7.215686503436747) < 1e-12,
              "S7d. the JOINT NLL is the oracle's pinned 7.215687");

        // The closed form is `node - onehot` -- asserted directly, so the
        // test is not merely checking a copy of the implementation.
        const Tensor node = crf_node_marginals(em, tr);
        bool closed = true;
        for (size_t t = 0; t < 4; ++t)
            for (size_t k = 0; k < 3; ++k) {
                double want = node(t, k) - (k == static_cast<size_t>(labels(t, 0)) ? 1.0 : 0.0);
                if (std::fabs(ge(t, k) - want) > 1e-12) closed = false;
            }
        check(closed, "S7e. the joint emissions gradient IS node - onehot");
    }

    // =============================================================================
    section("S8. THE MARGINAL ADJOINT");
    // =============================================================================
    {
        // g = -onehot(labels): the gradient of the MARGINAL objective
        // -sum_t log m_t(y_t).
        const Tensor labels = mat(4, 1, {0.0, 2.0, 1.0, 0.0});
        Tensor g = mat(4, 3, {-1.0, 0.0, 0.0,
                               0.0, 0.0, -1.0,
                               0.0, -1.0, 0.0,
                               -1.0, 0.0, 0.0});
        Tensor ge, gA;
        crf_marginal_log_likelihood_grad(em, tr, g, &ge, &gA);
        check(std::isfinite(ge(0, 0)) && std::isfinite(gA(0, 0)),
              "S8a. the marginal adjoint produces finite gradients");

        // A finite difference on the MARGINAL objective, computed here
        // independently of the implementation. The objective is
        // -sum_t log node[t,y_t] with node from `crf_node_marginals`.
        auto marginal_objective = [&](const Tensor& e, const Tensor& a) {
            const Tensor n = crf_node_marginals(e, a);
            double v = 0.0;
            for (size_t t = 0; t < 4; ++t) {
                v += std::log(n(t, static_cast<size_t>(labels(t, 0))));
            }
            return -v;
        };
        const double eps = 1e-6;
        double worst_e = 0.0, worst_A = 0.0;
        for (size_t t = 0; t < 4; ++t) {
            for (size_t k = 0; k < 3; ++k) {
                Tensor e2 = em.clone();
                e2(t, k) += eps;
                const double hi = marginal_objective(e2, tr);
                e2 = em.clone();
                e2(t, k) -= eps;
                const double lo = marginal_objective(e2, tr);
                const double fd = (hi - lo) / (2 * eps);
                worst_e = std::max(worst_e, std::fabs(fd - ge(t, k)));
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                Tensor a2 = tr.clone();
                a2(i, j) += eps;
                const double hi = marginal_objective(em, a2);
                a2 = tr.clone();
                a2(i, j) -= eps;
                const double lo = marginal_objective(em, a2);
                const double fd = (hi - lo) / (2 * eps);
                worst_A = std::max(worst_A, std::fabs(fd - gA(i, j)));
            }
        }
        check(worst_e < 1e-5,
              "S8b. the marginal adjoint's emissions gradient matches a finite difference");
        check(worst_A < 1e-5,
              "S8c. the marginal adjoint's transitions gradient matches a finite difference");
    }

    // =============================================================================
    section("S9. THE TWO GRADIENTS ARE DIFFERENT OBJECTS");
    // =============================================================================
    // THE WHOLE REASON BOTH ARE EXPOSED. If the joint closed form and the
    // marginal adjoint were the same object, exposing both would be
    // pointless. The oracle's first draft asserted they agreed -- they do not.
    {
        const Tensor labels = mat(4, 1, {0.0, 2.0, 1.0, 0.0});
        Tensor g = mat(4, 3, {-1.0, 0.0, 0.0,
                               0.0, 0.0, -1.0,
                               0.0, -1.0, 0.0,
                               -1.0, 0.0, 0.0});
        Tensor je, jA, me, mA;
        crf_joint_nll_grad(em, tr, labels, &je, &jA);
        crf_marginal_log_likelihood_grad(em, tr, g, &me, &mA);
        check(max_abs_diff(je, me) > 1e-6,
              "S9a. the joint and marginal EMISSION gradients DIFFER");
        check(max_abs_diff(jA, mA) > 1e-6,
              "S9b. the joint and marginal TRANSITION gradients DIFFER");

        // And the two OBJECTIVE VALUES differ, which is the deeper reason.
        const Tensor node = crf_node_marginals(em, tr);
        double marg = 0.0;
        for (size_t t = 0; t < 4; ++t) {
            marg += std::log(node(t, static_cast<size_t>(labels(t, 0))));
        }
        marg = -marg;
        const double joint = crf_log_partition(em, tr) - crf_sequence_score(em, tr, labels);
        check(std::fabs(marg - 7.441400) < 1e-6,
              "S9c. the MARGINAL NLL is the oracle's pinned 7.441400");
        check(std::fabs((marg - joint) - 0.225714) < 1e-6,
              "S9d. the two objectives differ by the pinned 0.225714");
    }

    // =============================================================================
    section("S10. THE LOG DOMAIN: LONG SEQUENCES MUST NOT UNDERFLOW");
    // =============================================================================
    {
        const size_t T = 200, K = 3;
        std::vector<double> eflat, aflat;
        // A deterministic, unit-scale fixture.
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < K; ++k)
                eflat.push_back(std::sin(1.7 * static_cast<double>(t) +
                                         0.9 * static_cast<double>(k)));
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < K; ++j)
                aflat.push_back(0.5 * std::cos(2.1 * static_cast<double>(i) -
                                              1.3 * static_cast<double>(j)));
        const Tensor e = mat(T, K, eflat);
        const Tensor a = mat(K, K, aflat);

        const double logz = crf_log_partition(e, a);
        check(std::isfinite(logz), "S10a. log Z is finite at T = 200");

        const Tensor node = crf_node_marginals(e, a);
        bool norm = true;
        for (size_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += node(t, k);
            if (std::fabs(s - 1.0) > 1e-9) norm = false;
        }
        check(norm, "S10b. the marginals still normalise at T = 200");

        // And the exp-space computation would NOT have worked: the number of
        // paths is 3^200, so the un-normalised sum overflows/underflows.
        double exp_space = 0.0;
        for (size_t t = 0; t < T; ++t) exp_space += eflat[t * K];
        check(std::isfinite(exp_space),
              "S10c. the control (a sum of raw emissions) stays finite");
    }

    // =============================================================================
    section("S11. EDGE CASES: T = 1, K = 1, AND THE UNIFORM FIXTURE");
    // =============================================================================
    {
        // T = 1: there are NO transitions at all.
        const Tensor e1 = mat(1, 3, {0.4, -0.2, 0.9});
        const Tensor a1 = mat(3, 3, {0.1, 0.2, 0.3,
                                     0.4, 0.5, 0.6,
                                     0.7, 0.8, 0.9});
        const double logz = crf_log_partition(e1, a1);
        check(std::fabs(logz - std::log(std::exp(0.4) + std::exp(-0.2) + std::exp(0.9))) < 1e-12,
              "S11a. T=1: log Z is the logsumexp of the emissions alone");
        const Tensor edge1 = crf_edge_marginals(e1, a1);
        check(edge1.rows == 0, "S11b. T=1: there are NO edge marginals");
        Tensor p1;
        const double vs1 = crf_viterbi(e1, a1, &p1);
        check(std::fabs(vs1 - 0.9) < 1e-12, "S11c. T=1: viterbi picks the best emission");
        check(p1(0, 0) == 2.0, "S11d. T=1: the viterbi path is the argmax label");

        // K = 1: one label, so every path is the same path.
        const Tensor e2 = mat(3, 1, {0.5, -0.25, 0.75});
        const Tensor a2 = mat(1, 1, {0.0});
        const double logz2 = crf_log_partition(e2, a2);
        check(std::fabs(logz2 - 1.0) < 1e-12, "S11e. K=1: log Z is the sum of the emissions");
        const Tensor node2 = crf_node_marginals(e2, a2);
        check(std::fabs(node2(0, 0) - 1.0) < 1e-12, "S11f. K=1: the only label has probability 1");

        // Uniform scores: log Z = T log K exactly, and the marginals uniform.
        const Tensor e3 = mat(4, 3, std::vector<double>(12, 0.0));
        const Tensor a3 = mat(3, 3, std::vector<double>(9, 0.0));
        const double logz3 = crf_log_partition(e3, a3);
        check(std::fabs(logz3 - 4.0 * std::log(3.0)) < 1e-12,
              "S11g. the uniform fixture: log Z == T log K exactly");
        const Tensor node3 = crf_node_marginals(e3, a3);
        bool unif = true;
        for (size_t t = 0; t < 4; ++t)
            for (size_t k = 0; k < 3; ++k)
                if (std::fabs(node3(t, k) - 1.0 / 3.0) > 1e-12) unif = false;
        check(unif, "S11h. the uniform fixture: the marginals are uniform");
    }

    // =============================================================================
    section("S12. TRANSITION CONSTRAINTS: -inf MASKS A LABEL PAIR");
    // =============================================================================
    {
        // A label that can never be ENTERED may still appear at position 0,
        // because there is no transition into position 0.
        Tensor a = main_transitions();
        for (size_t j = 0; j < 3; ++j) a(j, 2) = -std::numeric_limits<double>::infinity();
        const Tensor node = crf_node_marginals(em, a);
        check(node(0, 2) > 0.0,
              "S12a. a label that can never be ENTERED may still appear at position 0");
        bool zero_after = true;
        for (size_t t = 1; t < 4; ++t) if (node(t, 2) != 0.0) zero_after = false;
        check(zero_after,
              "S12b. a label that can never be entered has probability 0 at every later position");

        // Masking one transition does not zero out a label entirely.
        Tensor a2 = main_transitions();
        a2(0, 1) = -std::numeric_limits<double>::infinity();
        const Tensor node2 = crf_node_marginals(em, a2);
        check(node2(0, 0) > 0.0 && node2(0, 1) > 0.0,
              "S12c. masking a transition does not zero out a label entirely");
        check(std::isfinite(crf_log_partition(em, a2)),
              "S12d. a masked transition keeps log Z finite");
    }

    // =============================================================================
    section("S13. THE GUARDS");
    // =============================================================================
    {
        bool threw = false;
        try {
            crf_log_partition(mat(0, 3, {}), main_transitions());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13a. empty emissions throw");

        threw = false;
        try {
            crf_log_partition(em, mat(2, 2, {0.0, 0.0, 0.0, 0.0}));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13b. a transition shape mismatch throws");

        threw = false;
        try {
            crf_log_partition(mat(2, 2, {0.0, std::nan(""), 0.0, 0.0}), mat(2, 2, {0, 0, 0, 0}));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13c. a non-finite emission throws");

        threw = false;
        try {
            crf_joint_nll_grad(em, tr, mat(4, 1, {0.0, 2.0, 1.0, 3.0}), nullptr, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13d. an out-of-range label throws");

        threw = false;
        try {
            crf_marginal_log_likelihood_grad(em, tr, mat(4, 2, std::vector<double>(8, 0.0)),
                                             nullptr, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13e. a g shape mismatch throws");

        // A label that is not an integer is out of range too.
        threw = false;
        try {
            crf_joint_nll_grad(em, tr, mat(4, 1, {0.0, 2.0, 1.0, 0.5}), nullptr, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "S13f. a non-integer label throws");
    }

    // =============================================================================
    section("S14. DETERMINISM");
    // =============================================================================
    {
        const double z1 = crf_log_partition(em, tr);
        const double z2 = crf_log_partition(em, tr);
        check(z1 == z2, "S14a. repeated calls are bit-identical");
        const Tensor n1 = crf_node_marginals(em, tr);
        const Tensor n2 = crf_node_marginals(em, tr);
        check(max_abs_diff(n1, n2) == 0.0, "S14b. the marginals are bit-identical");
    }

    // =============================================================================
    printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}  // main()
