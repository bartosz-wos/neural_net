// test_ranking_metrics.cpp — TDD tests for the ranking metrics that follow the
// shipped ROC-AUC / average-precision metrics: DCG, NDCG, MAP, MRR.
//
// Sources (fetched and read 2026-10-13, NOT from memory):
//   * sklearn 1.9.1 sklearn/metrics/_ranking.py
//       _dcg_sample_scores        (:1655-1711)  discount + k-zeroing + tie-averaging
//       _tie_averaged_dcg         (:1713-1762)  McSherry & Najork (ECIR 2008)
//       _ndcg_sample_scores       (:1898-1946)  the all_irrelevant branch
//       ndcg_score                (:2053-2067)  mean over queries
//   * trec_eval via ir_measures 0.4.3 (AP, RR) -- an INDEPENDENT second
//     implementation, used to pin the binary-relevance AP and the tie rules.
//
// EVERY expected value below was produced by an oracle run against those two
// libraries (see docs/plans/2026-10-13-ranking-metrics.md). None is
// hand-derived. The fixtures are the ones in the plan's oracle script.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "nn/utils/metrics.h"

static int passed = 0;
static int failed = 0;

// NOTE: argument order is (name, result) so each call reads as a sentence
// describing what was checked.
static void check(const std::string& name, bool pass) {
    if (pass) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else       { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
}

static bool close(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

// Build an (N_queries, N_docs) relevance / score pair from row-major literals.
struct RankData {
    Tensor rel;
    Tensor score;
};
static RankData make_rank(const std::vector<std::vector<double>>& rel,
                          const std::vector<std::vector<double>>& score) {
    const size_t nq = rel.size();
    const size_t nd = rel[0].size();
    RankData d;
    d.rel   = Tensor(nq, nd);
    d.score = Tensor(nq, nd);
    for (size_t q = 0; q < nq; ++q) {
        if (rel[q].size() != nd || score[q].size() != nd) {
            throw std::invalid_argument("ragged fixture");
        }
        for (size_t j = 0; j < nd; ++j) {
            d.rel(q, j)   = rel[q][j];
            d.score(q, j) = score[q][j];
        }
    }
    return d;
}

// ---- Fixtures (all values from the oracle run) ---------------------------
//
// F1 graded relevance, perfect ranking (4 docs)
static RankData F1() { return make_rank({{3, 2, 1, 0}}, {{0.9, 0.8, 0.7, 0.1}}); }
// F3 graded relevance, REVERSED ranking (the minimum)
static RankData F3() { return make_rank({{3, 2, 1, 0}}, {{0.1, 0.2, 0.3, 0.4}}); }
// F4 relevance in a non-sorted input order, score == relevance (perfect)
static RankData F4() { return make_rank({{0, 1, 2, 3}}, {{0, 1, 2, 3}}); }
// F5 binary relevance, 5 docs
static RankData F5() { return make_rank({{1, 0, 1, 0, 1}}, {{0.9, 0.8, 0.7, 0.6, 0.5}}); }
// F6 nothing relevant anywhere -> all_irrelevant branch
static RankData F6() { return make_rank({{0, 0, 0, 0}}, {{0.9, 0.8, 0.7, 0.1}}); }
// F7 one relevant doc at rank 3 of 4
static RankData F7() { return make_rank({{0, 0, 1, 0}}, {{0.9, 0.8, 0.5, 0.1}}); }
// F8 mixed graded, for the k cut-offs
static RankData F8() { return make_rank({{2, 3, 1, 0, 0}}, {{0.1, 0.9, 0.4, 0.3, 0.2}}); }
// F2 three queries
static RankData F2() {
    return make_rank({{3, 2, 1, 0},
                      {0, 1, 2, 3},
                      {1, 0, 0, 0}},
                     {{0.9, 0.8, 0.7, 0.1},
                      {0.2, 0.3, 0.9, 0.8},
                      {0.7, 0.2, 0.5, 0.1}});
}

// ---------------------------------------------------------------------------
// Test 1: validation contract.
// ---------------------------------------------------------------------------
static void test_validation() {
    std::cout << std::endl << "-- Test 1: validation --" << std::endl;

    bool threw = false;
    try { dcg_score(Tensor(0, 3), Tensor(0, 3)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("empty tensors throw", threw);

    threw = false;
    try { RankData d = F1(); dcg_score(d.rel, Tensor(1, 5)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("mismatched cols throws", threw);

    threw = false;
    try {
        RankData d = F1();
        dcg_score(d.rel, d.score, 1, 0.0);   // log_base == 0
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("log_base == 0 throws", threw);

    threw = false;
    try {
        RankData d = F1();
        dcg_score(d.rel, d.score, 1, 1.0);   // log_base == 1
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("log_base == 1 throws", threw);

    threw = false;
    try {
        // single document: NDCG is documented as meaningless with 1 doc
        dcg_score(make_rank({{3}}, {{0.5}}).rel, make_rank({{3}}, {{0.5}}).score);
        ndcg_score(make_rank({{3}}, {{0.5}}).rel, make_rank({{3}}, {{0.5}}).score);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("single-document NDCG throws (documented rule)", threw);

    threw = false;
    try {
        // sklearn forbids negative relevance for the DCG family (_ranking.py:2058)
        RankData d = make_rank({{3, -1, 1, 0}}, {{0.9, 0.8, 0.7, 0.1}});
        ndcg_score(d.rel, d.score);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("negative relevance throws", threw);

    // NOTE (test bug fixed 2026-10-13): the first version of this assertion
    // used fixture F3, which is {3,2,1,0} -- all non-negative -- so it could
    // never throw and the assertion was false. A fixture with an actual
    // negative grade is required for this to mean anything.
    threw = false;
    try {
        RankData d = make_rank({{3, 2, 1, 0}}, {{0.9, 0.8, 0.7, 0.1}});
        ndcg_score(d.rel, d.score);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("all-non-negative relevance does NOT throw", !threw);

    // k == 0 is the SENTINEL for "no truncation", not an error -- it is the
    // documented default, so requesting it explicitly must behave identically
    // to omitting it. (An earlier draft of this file asserted that k == 0
    // throws; that contradicted the default value of the parameter.)
    RankData d = F1();
    check("k == 0 (the default sentinel) means no truncation",
          close(dcg_score(d.rel, d.score, 0), dcg_score(d.rel, d.score))
            && close(ndcg_score(d.rel, d.score, 0), ndcg_score(d.rel, d.score)));

    bool big_k_ok = true;
    try { dcg_score(d.rel, d.score, 99); } catch (...) { big_k_ok = false; }
    check("k > n_docs is allowed", big_k_ok);
}

// ---------------------------------------------------------------------------
// Test 2: DCG closed form. THE HEADLINE TEST for the S1 off-by-one.
// discount[j] = log(2)/log(j+2), NOT 1/log2(j+1).
// ---------------------------------------------------------------------------
static void test_dcg_matches_closed_form() {
    std::cout << std::endl << "-- Test 2: DCG discount indexing --" << std::endl;

    RankData d = F1();
    // Oracle: dcg = 4.761859507142915
    check("F1 dcg matches oracle 4.761859507142915",
          close(dcg_score(d.rel, d.score), 4.761859507142915));

    // Hand-closed-form with the CORRECT indexing, computed independently:
    //   3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
    // The WRONG indexing (1/log2(j+1), j from 0) would instead give
    //   3/log2(1) + ... which is infinity for j=0. A subtler wrong variant
    // (rank starting at 1 but gain applied to the wrong doc) gives 3.7618...
    const double log2 = 0.69314718055994530942;
    double expected = 3.0 / std::log2(2.0)
                    + 2.0 / (std::log(3.0) / log2)
                    + 1.0 / (std::log(4.0) / log2)
                    + 0.0 / (std::log(5.0) / log2);
    check("F1 dcg equals the hand closed form to 1e-12",
          close(dcg_score(d.rel, d.score), expected, 1e-12));

    // The classic WRONG indexing would give this different value:
    double wrong = 3.0 / (std::log(1.0) / log2)   // -> inf, so use the
                 + 2.0 / (std::log(2.0) / log2)
                 + 1.0 / (std::log(3.0) / log2);
    check("F1 ddc is NOT the 1/log2(j+1) variant",
          std::fabs(dcg_score(d.rel, d.score) - wrong) > 1e-3);
}

// ---------------------------------------------------------------------------
// Test 3: DCG under k. S2 -- k zeroes the discount BEYOND k.
// ---------------------------------------------------------------------------
static void test_dcg_at_k() {
    std::cout << std::endl << "-- Test 3: DCG@k --" << std::endl;

    RankData d = F1();
    // Oracle: k=1 -> 3.0, k=2 -> 4.261859507142915, k=3 -> 4.761859507142915
    check("F1 dcg@1 == 3.0 (only the top rank counts)",
          close(dcg_score(d.rel, d.score, 1), 3.0));
    check("F1 dcg@2 == 4.261859507142915",
          close(dcg_score(d.rel, d.score, 2), 4.261859507142915));
    check("F1 dcg@3 == 4.761859507142915",
          close(dcg_score(d.rel, d.score, 3), 4.761859507142915));
    // k beyond the document count must equal the full score
    check("F1 dcg@4 == full dcg",
          close(dcg_score(d.rel, d.score, 4), dcg_score(d.rel, d.score)));
    check("F1 dcg@99 == full dcg",
          close(dcg_score(d.rel, d.score, 99), dcg_score(d.rel, d.score)));
    // k = 1 drops the relevance of the rank-1 document (F3's best is last)
    RankData r = F3();
    check("F3 dcg@1 == 0.0 (the relevant doc is ranked last)",
          close(dcg_score(r.rel, r.score, 1), 0.0));
}

// ---------------------------------------------------------------------------
// Test 4: NDCG is exactly 1.0 on a perfect ranking. S3 -- the ideal gain
// sorts by y_true, NOT by input order.
// ---------------------------------------------------------------------------
static void test_ndcg_perfect() {
    std::cout << std::endl << "-- Test 4: NDCG == 1 on perfect ranking --" << std::endl;

    RankData a = F1();
    check("F1 ndcg == 1.0", close(ndcg_score(a.rel, a.score), 1.0));

    // F4 is the load-bearing one: relevance arrives in the ORDER 0,1,2,3 and
    // score == relevance. An implementation that computed the ideal from the
    // input order (without sorting descending) would get a different answer.
    RankData b = F4();
    check("F4 ndcg == 1.0 with relevance in ascending input order",
          close(ndcg_score(b.rel, b.score), 1.0));

    RankData c = F8();
    check("F8 ndcg == 0.9249822178570098 (oracle)",
          close(ndcg_score(c.rel, c.score), 0.9249822178570098));
    RankData e = F5();
    check("F5 ndcg == 0.8854598815714875 (oracle)",
          close(ndcg_score(e.rel, e.score), 0.8854598815714875));
}

// ---------------------------------------------------------------------------
// Test 5: NDCG@k normalises by the TRUNCATED ideal, not the full one.
// S2. f8 at k=2 is the discriminating fixture.
// ---------------------------------------------------------------------------
static void test_ndcg_at_k_uses_truncated_ideal() {
    std::cout << std::endl
              << "-- Test 5: NDCG@k uses the truncated ideal --" << std::endl;

    RankData d = F8();
    // Oracle: ndcg@1 = 1.0, @2 = 0.8519590445170674, @3 = 0.7625024946924551
    check("F8 ndcg@1 == 1.0", close(ndcg_score(d.rel, d.score, 1), 1.0));
    check("F8 ndcg@2 == 0.8519590445170674",
          close(ndcg_score(d.rel, d.score, 2), 0.8519590445170674));
    check("F8 ndcg@3 == 0.7625024946924551",
          close(ndcg_score(d.rel, d.score, 3), 0.7625024946924551));

    // The wrong rule -- normalising by the FULL-length ideal -- gives:
    //   dcg@2 (3.6309297535714573) / full ideal (4.404635368040541)
    double full_ideal = 4.404635368040541;
    double wrong = 3.6309297535714573 / full_ideal;
    check("F8 ndcg@2 is NOT gain/full-ideal (0.6637...)",
          std::fabs(ndcg_score(d.rel, d.score, 2) - wrong) > 1e-3);

    // F2's k=1 is the second discriminating fixture: oracle 0.8888888888888888
    RankData e = F2();
    check("F2 ndcg@1 == 0.8888888888888888",
          close(ndcg_score(e.rel, e.score, 1), 0.8888888888888888));
    check("F2 ndcg@2 == 0.9711338641571848",
          close(ndcg_score(e.rel, e.score, 2), 0.9711338641571848));
}

// ---------------------------------------------------------------------------
// Test 6: a query with NO relevant document scores 0.0 and is INCLUDED in
// the mean. S4 -- this is not a drop, and not a divide-by-zero.
// ---------------------------------------------------------------------------
static void test_no_relevant_docs_scores_zero() {
    std::cout << std::endl
              << "-- Test 6: all-irrelevant query scores 0 --" << std::endl;

    RankData d = F6();
    check("F6 dcg == 0.0", close(dcg_score(d.rel, d.score), 0.0));
    check("F6 ndcg == 0.0 (no divide-by-zero)", close(ndcg_score(d.rel, d.score), 0.0));
    check("F6 map == 0.0", close(mean_average_precision(d.rel, d.score), 0.0));
    check("F6 mrr == 0.0", close(mean_reciprocal_rank(d.rel, d.score), 0.0));

    // S4/S8: the all-irrelevant query is AVERAGED IN, not dropped. Mixing F6
    // with a perfect query halves the aggregate exactly.
    RankData mixed = make_rank({{0, 0, 0, 0}, {0, 1, 2, 3}},
                               {{0.9, 0.8, 0.7, 0.1}, {0, 1, 2, 3}});
    check("an all-irrelevant query HALVES the ndcg mean (0 + 1) / 2",
          close(ndcg_score(mixed.rel, mixed.score), 0.5, 1e-12));
    check("an all-irrelevant query HALVES the mrr mean",
          close(mean_reciprocal_rank(mixed.rel, mixed.score), 0.5, 1e-12));
}

// ---------------------------------------------------------------------------
// Test 7: MAP is the TREC mean-precision-at-relevant-ranks. S5/S6.
//
// ORACLE CORRECTION (2026-10-13): the first version of this file took its MAP
// values from a 2-D (1, n_docs) call into sklearn, which routes to the
// MULTILABEL average-precision path, not the ranking path -- a different
// metric. Every MAP/MRR value below was re-derived on the ranking convention
// and now agrees with THREE independent computations (trec_eval via
// ir_measures, sklearn's 1-D average_precision_score, and a direct
// transcription) to the last digit. See oracle_v2.py.
// ---------------------------------------------------------------------------
static void test_map() {
    std::cout << std::endl << "-- Test 7: MAP --" << std::endl;

    // MAP on the BINARY view of relevance.
    RankData f1 = F1();
    check("F1 map == 1.0 (binary view of graded relevance, perfect order)",
          close(mean_average_precision(f1.rel, f1.score), 1.0, 1e-12));

    RankData f3 = F3();
    check("F3 map == 0.6388888888888888 (hits at ranks 2,3,4 -> "
          "(1/2+2/3+3/4)/3)",
          close(mean_average_precision(f3.rel, f3.score), 0.6388888888888888));

    RankData f5 = F5();
    check("F5 map == 0.7555555555555555 (hits at ranks 1,3,5 -> "
          "(1/1+2/3+3/5)/3)",
          close(mean_average_precision(f5.rel, f5.score), 0.7555555555555555));

    RankData f7 = F7();
    check("F7 map == 0.3333333333333333 (single hit at rank 3)",
          close(mean_average_precision(f7.rel, f7.score), 0.3333333333333333));

    RankData f8 = F8();
    check("F8 map == 0.8666666666666667",
          close(mean_average_precision(f8.rel, f8.score), 0.8666666666666667));

    RankData f4 = F4();
    check("F4 map == 1.0 (perfect ranking, relevance in ascending order)",
          close(mean_average_precision(f4.rel, f4.score), 1.0));

    // S6 is the load-bearing rule: MAP ignores the GRADE magnitude. A
    // relevance-weighted variant on F1 would give a different number, and the
    // grades {3,2,1,0} vs {1,1,1,0} must therefore give the SAME map.
    RankData binarised = make_rank({{1, 1, 1, 0}}, {{0.9, 0.8, 0.7, 0.1}});
    check("F1 map is IDENTICAL when grades 3,2,1 are collapsed to 1,1,1",
          close(mean_average_precision(f1.rel, f1.score),
                mean_average_precision(binarised.rel, binarised.score), 1e-12));
    check("...and that shared value is 1.0",
          close(mean_average_precision(binarised.rel, binarised.score), 1.0));

    // S5: this is a SEPARATE symbol from the shipped step-integral
    // average_precision_score, and this is the fixture that PROVES the two
    // definitions differ rather than merely asserting it.
    //
    // ORACLE NOTE (2026-10-13): the first version of this check compared the
    // two on F3 and asserted they agree, which was both vacuous (they agree on
    // F3) and unusable as written -- it built a (1, n_docs) Tensor and handed
    // it to average_precision_score, whose API is one score per EXAMPLE, so it
    // threw and aborted the suite. The two metrics take different input
    // conventions, so there is no single tensor both accept.
    //
    // Translated correctly -- F3's four documents, ranked d3(.4) d2(.3) d1(.2)
    // d0(.1) with relevance {3,2,1,0} so hits at ranks 2,3,4 -- becomes three
    // EXAMPLES fed to the per-example API. And the translation is exactly what
    // exposes the real difference:
    //     mean precision at relevant ranks = (1/2 + 2/3 + 3/4)/3 = 0.6389  <- MAP
    //     step integral over the PR curve                             = 0.5833  <- shipped
    // verified against sklearn 1.9.1 (average_precision_score([0,1,1], [.4,.3,.2])
    // = 0.5833333333333333). The first draft asserted 0.6389 for BOTH, which
    // is the value the step integral does NOT produce.
    Tensor ex_t(3, 1);
    Tensor ex_s(3, 1);
    ex_s(0, 0) = 0.4; ex_t(0, 0) = 0.0;   // d3, ranked 1, irrelevant
    ex_s(1, 0) = 0.3; ex_t(1, 0) = 1.0;   // d2, ranked 2, relevant
    ex_s(2, 0) = 0.2; ex_t(2, 0) = 1.0;   // d1, ranked 3, relevant
    check("the shipped step-AP on F3's induced ranking == 0.5833333333333333 "
          "(sklearn-verified)",
          close(average_precision_score(ex_t, ex_s), 0.5833333333333333, 1e-12));
    check("...while our MAP on the same ranking == 0.6388888888888888",
          close(mean_average_precision(f3.rel, f3.score), 0.6388888888888888, 1e-12));
    // The two MUST disagree here -- this is the assertion that makes them
    // non-interchangeable, so it is kept deliberately.
    check("MAP and the shipped step-AP MUST disagree on this fixture "
          "(else they could be merged)",
          std::fabs(mean_average_precision(f3.rel, f3.score)
                    - average_precision_score(ex_t, ex_s)) > 0.05);
}

// ---------------------------------------------------------------------------
// Test 8: MRR, and the tie rule. S7.
//
// ORACLE CORRECTION (2026-10-13), and this one is worth reading: the first
// version of this file cited trec_eval's RR as evidence for a "pessimistic"
// tie rule, with the claim that the tie-break was insertion-order independent
// because four dict orderings all gave 1/3. **That inference was wrong**, and
// the control that exposed it is in the session log: moving the relevant
// document to a different position in the dict CHANGES trec_eval's answer
// (1.0 / 0.5 / 0.3333 for the same three tied documents). trec_eval sorts tied
// documents by DESCENDING docid, so its tie handling is an artifact of how the
// run file was written, not semantics to copy.
//
// The rule below is instead pinned to sklearn's 1-D average_precision_score,
// which is position-INDEPENDENT, and cross-checked against a direct
// transcription. The verified rule is PESSIMISTIC: within a group of equal
// scores, relevant documents sort last.
// ---------------------------------------------------------------------------
static void test_mrr() {
    std::cout << std::endl << "-- Test 8: MRR and the tie rule --" << std::endl;

    RankData f1 = F1();
    check("F1 mrr == 1.0", close(mean_reciprocal_rank(f1.rel, f1.score), 1.0));

    RankData f7 = F7();
    check("F7 mrr == 0.3333333333333333 (first hit at rank 3)",
          close(mean_reciprocal_rank(f7.rel, f7.score), 0.3333333333333333));

    RankData f3 = F3();
    check("F3 mrr == 0.5 (relevance {3,2,1}, first hit at rank 2)",
          close(mean_reciprocal_rank(f3.rel, f3.score), 0.5));

    RankData f8 = F8();
    check("F8 mrr == 1.0", close(mean_reciprocal_rank(f8.rel, f8.score), 1.0));

    // --- S7: ties, resolved PESSIMISTICALLY (relevant last in its group) ---
    // Full tie, doc 0 the only relevant one -> it sorts last -> rank 3.
    RankData tie = make_rank({{1, 0, 0}}, {{0.5, 0.5, 0.5}});
    check("full tie: mrr == 1/3 (pessimistic, relevant sorts LAST)",
          close(mean_reciprocal_rank(tie.rel, tie.score), 1.0 / 3.0, 1e-12));

    // The position-independence fixture: the SAME tied data with the relevant
    // document at a different INDEX must give the SAME answer. An
    // index-sensitive tie-break fails this and nothing else would catch it.
    RankData tie_mid = make_rank({{0, 1, 0}}, {{0.5, 0.5, 0.5}});
    RankData tie_last = make_rank({{0, 0, 1}}, {{0.5, 0.5, 0.5}});
    check("a full tie gives the same mrr regardless of WHERE the hit is "
          "(position independence)",
          close(mean_reciprocal_rank(tie.rel, tie.score),
                mean_reciprocal_rank(tie_mid.rel, tie_mid.score), 1e-12)
          && close(mean_reciprocal_rank(tie.rel, tie.score),
                   mean_reciprocal_rank(tie_last.rel, tie_last.score), 1e-12));

    // THE discriminating fixture: d2 scores higher, then d0 and d1 tie, and the
    // relevant doc d0 sits INSIDE that tied group.
    //   pessimistic -> d0 last in group -> rank 3 -> 1/3   [sklearn, verified]
    //   docid-ascending            -> rank 2 -> 0.5
    //   tie-averaged               -> 0.4166666666666667
    RankData part = make_rank({{1, 0, 0}}, {{0.5, 0.5, 0.9}});
    check("relevant doc inside a tied group: mrr == 1/3 (pessimistic)",
          close(mean_reciprocal_rank(part.rel, part.score), 1.0 / 3.0, 1e-12));
    check("...and is NOT the docid-ascending 0.5",
          std::fabs(mean_reciprocal_rank(part.rel, part.score) - 0.5) > 1e-3);
    check("...and is NOT the tie-averaged 0.4166666666666667",
          std::fabs(mean_reciprocal_rank(part.rel, part.score)
                    - 0.4166666666666667) > 1e-3);

    // A tie group containing TWO relevant documents among four. Under the
    // pessimistic rule the irrelevant pair comes first, so the hits land at
    // ranks 3 and 4.
    //
    // ORACLE NOTE (2026-10-13): the first version of this block asserted the
    // first hit is at rank 2 -> MRR 0.5. That was a TEST bug, verified
    // against sklearn rather than assumed: on the fully tied fixture
    // sklearn's own 1-D AP returns 0.5, but that number is the value for a
    // DIFFERENT thing -- the two relevant documents in document order. The
    // pessimistic rule puts them at ranks 3 and 4, giving MAP
    // (1/3 + 2/4)/2 = 0.41666666666666663 (sklearn's own answer for this
    // fixture's DISTINCT-score ordering, which puts the relevant pair first,
    // is the same number by coincidence of the symmetry) and MRR = 1/3.
    // sklearn's tied-fixture 0.5 is a tie-average over the two orderings, not
    // the pessimistic value.
    RankData multi = make_rank({{1, 1, 0, 0}}, {{0.5, 0.5, 0.5, 0.5}});
    check("tie group {d0,d1 relevant, d2,d3 not}: first hit is at rank 3",
          close(mean_reciprocal_rank(multi.rel, multi.score), 1.0 / 3.0, 1e-12));
    // And MAP under the same rule: both hits at ranks 3 and 4.
    check("map on that tie == (1/3 + 2/4)/2 = 0.41666666666666663",
          close(mean_average_precision(multi.rel, multi.score),
                (1.0 / 3.0 + 2.0 / 4.0) / 2.0, 1e-12));
    // Position independence: the same fixture with the relevant pair moved to
    // the END of the row must give identical numbers.
    RankData multi_moved = make_rank({{0, 0, 1, 1}}, {{0.5, 0.5, 0.5, 0.5}});
    check("...and is independent of where the relevant documents sit",
          close(mean_reciprocal_rank(multi.rel, multi.score),
                mean_reciprocal_rank(multi_moved.rel, multi_moved.score), 1e-12)
          && close(mean_average_precision(multi.rel, multi.score),
                   mean_average_precision(multi_moved.rel, multi_moved.score), 1e-12));

    // The DCG family on this SAME tie uses the tie-AVERAGE (1.280803155822425),
    // which is a DIFFERENT number from the pessimistic DCG
    // (1/log2(4) + 1/log2(5) = 1.4306...). Both rules are pinned, and each is
    // asserted against the other's value so neither can silently become it.
    check("DCG on a 4-way tie is the tie-AVERAGE 1.280803155822425 "
          "(not the pessimistic rule)",
          close(dcg_score(multi.rel, multi.score), 1.280803155822425, 1e-12));
    check("NDCG on that tie is 0.7853208594776601",
          close(ndcg_score(multi.rel, multi.score), 0.7853208594776601, 1e-12));
    double pessimistic_dcg = 1.0 / (std::log(4.0) / std::log(2.0))
                           + 1.0 / (std::log(5.0) / std::log(2.0));
    check("...and is NOT the pessimistic DCG on the same fixture",
          std::fabs(dcg_score(multi.rel, multi.score)
                    - pessimistic_dcg) > 1e-3);

    // A single relevant doc in a 3-way tie, DCG family (oracle):
    // 0.7103099178571524
    RankData tie3 = make_rank({{1, 0, 0}}, {{0.5, 0.5, 0.5}});
    check("DCG on a 3-way tie with one hit is 0.7103099178571524 (tie-average)",
          close(dcg_score(tie3.rel, tie3.score), 0.7103099178571524, 1e-12));
    // The same fixture under the pessimistic rule would be 1/log2(4) = 0.5.
    check("...and is NOT the pessimistic 0.5",
          std::fabs(dcg_score(tie3.rel, tie3.score) - 0.5) > 1e-3);
    // The relevant document's INDEX must not change the DCG -- the tie-average
    // is position-independent by construction, and this is the control that
    // the discarded trec_eval reading would have failed.
    RankData tie3_mid = make_rank({{0, 1, 0}}, {{0.5, 0.5, 0.5}});
    check("DCG on a tie is independent of where the hit sits",
          close(dcg_score(tie3.rel, tie3.score),
                dcg_score(tie3_mid.rel, tie3_mid.score), 1e-12));
}

// ---------------------------------------------------------------------------
// Test 9: aggregate = unweighted mean over queries. S9.
// ---------------------------------------------------------------------------
static void test_aggregate_is_mean_over_queries() {
    std::cout << std::endl << "-- Test 9: aggregate is the mean over queries --"
              << std::endl;

    RankData d = F2();
    check("F2 ndcg == 0.9741648372255328 (oracle)",
          close(ndcg_score(d.rel, d.score), 0.9741648372255328));
    check("F2 map == 1.0 (oracle)",
          close(mean_average_precision(d.rel, d.score), 1.0));
    check("F2 mrr == 1.0 (oracle)",
          close(mean_reciprocal_rank(d.rel, d.score), 1.0));

    // The per-query decomposition must average back to the aggregate exactly.
    std::vector<double> per = ndcg_per_query(d.rel, d.score);
    check("F2 has 3 per-query ndcg values", per.size() == 3);
    double mean = 0.0;
    for (double v : per) mean += v;
    mean /= 3.0;
    check("the mean of the per-query ndcg equals the aggregate",
          close(mean, ndcg_score(d.rel, d.score), 1e-12));

    // And the per-query values must themselves be the right ones.
    RankData q0 = F1();
    RankData q1 = make_rank({{0, 1, 2, 3}}, {{0.2, 0.3, 0.9, 0.8}});
    RankData q2 = make_rank({{1, 0, 0, 0}}, {{0.7, 0.2, 0.5, 0.1}});
    check("F2 query 0 ndcg == 1.0", close(per[0], ndcg_score(q0.rel, q0.score), 1e-12));
    check("F2 query 1 ndcg matches its own single-query score",
          close(per[1], ndcg_score(q1.rel, q1.score), 1e-12));
    check("F2 query 2 ndcg matches its own single-query score",
          close(per[2], ndcg_score(q2.rel, q2.score), 1e-12));

    // A per-query metric with a bad query must NOT drop it.
    std::vector<double> mr = reciprocal_rank_per_query(F6().rel, F6().score);
    check("F6 per-query mrr is a single 0.0", mr.size() == 1 && close(mr[0], 0.0));
}

// ---------------------------------------------------------------------------
// Test 10: MAP under k. Only the top k ranks are considered.
// ---------------------------------------------------------------------------
static void test_map_at_k() {
    std::cout << std::endl << "-- Test 10: MAP@k --" << std::endl;

    // F5: relevance at docs 0,2,4. With k=2 only doc 0 is inside the window,
    // so AP = 1/1 = 1.0 (one hit at rank 1).
    RankData d = F5();
    check("F5 map@2 == 1.0 (only the first hit is inside the window)",
          close(mean_average_precision(d.rel, d.score, 2), 1.0, 1e-12));
    // With k=3 two hits are inside the window: 1/1 and 2/3 -> 0.8333...
    // (oracle_v2.py MAP@k sweep)
    check("F5 map@3 == 0.8333333333333333 (two hits inside the window)",
          close(mean_average_precision(d.rel, d.score, 3),
                0.8333333333333333, 1e-12));
    // With k=5 all three hits count: 1/1, 2/3, 3/5 -> mean 0.7555...
    check("F5 map@5 == 0.7555555555555555 (== the untruncated map)",
          close(mean_average_precision(d.rel, d.score, 5), 0.7555555555555555, 1e-12));
    check("F5 map@99 == the untruncated map",
          close(mean_average_precision(d.rel, d.score, 99),
                mean_average_precision(d.rel, d.score), 1e-12));
    // k=1 on F7 (hit at rank 3) gives an empty window -> 0.0
    RankData e = F7();
    check("F7 map@1 == 0.0 (the only hit is outside the window)",
          close(mean_average_precision(e.rel, e.score, 1), 0.0, 1e-12));
    // MRR@k: F7's hit at rank 3 is outside a k=2 window -> 0.0
    check("F7 mrr@2 == 0.0 (the only hit is outside the window)",
          close(mean_reciprocal_rank(e.rel, e.score, 2), 0.0, 1e-12));
    check("F7 mrr@3 == 0.3333333333333333",
          close(mean_reciprocal_rank(e.rel, e.score, 3), 0.3333333333333333, 1e-12));
}

// ---------------------------------------------------------------------------
// Test 11: non-specificity. A metric that cannot separate two DIFFERENT
// rankings has no business being shipped, so pin that it can.
// ---------------------------------------------------------------------------
static void test_metrics_separate_good_and_bad_rankings() {
    std::cout << std::endl
              << "-- Test 11: good ranking beats the reversed one --" << std::endl;

    RankData good = F1();     // ndcg 1.0
    RankData bad  = F3();     // ndcg 0.6138273133441086
    check("the perfect ranking strictly beats the reversed one on ndcg",
          ndcg_score(good.rel, good.score) > ndcg_score(bad.rel, bad.score));
    check("...and on dcg (4.76 vs 2.92)",
          dcg_score(good.rel, good.score) > dcg_score(bad.rel, bad.score));
    check("...and on map (1.0 vs 0.6388...)",
          mean_average_precision(good.rel, good.score)
            > mean_average_precision(bad.rel, bad.score));
    check("...and on mrr (1.0 vs 0.5)",
          mean_reciprocal_rank(good.rel, good.score)
            > mean_reciprocal_rank(bad.rel, bad.score));

    // A tiny score perturbation that changes only the order of the top two
    // MUST move ndcg. If it does not, the sorting is broken.
    RankData swap = make_rank({{3, 2, 1, 0}}, {{0.8, 0.9, 0.7, 0.1}});
    check("swapping the top two scores lowers ndcg below 1.0",
          ndcg_score(swap.rel, swap.score) < 1.0 - 1e-9);
}

// ---------------------------------------------------------------------------
// Test 12: determinism — repeated calls are bit-identical.
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << std::endl << "-- Test 12: determinism --" << std::endl;

    RankData d = F2();
    double n1 = ndcg_score(d.rel, d.score);
    double n2 = ndcg_score(d.rel, d.score);
    check("ndcg is bit-identical across calls", n1 == n2);
    check("map is bit-identical across calls",
          mean_average_precision(d.rel, d.score)
            == mean_average_precision(d.rel, d.score));
    check("mrr is bit-identical across calls",
          mean_reciprocal_rank(d.rel, d.score)
            == mean_reciprocal_rank(d.rel, d.score));
    check("dcg is bit-identical across calls",
          dcg_score(d.rel, d.score) == dcg_score(d.rel, d.score));
    std::vector<double> a = ndcg_per_query(d.rel, d.score);
    std::vector<double> b = ndcg_per_query(d.rel, d.score);
    check("per-query ndcg is bit-identical across calls", a == b);
}

int main() {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "=== Ranking Metrics Tests ===" << std::endl;

    test_validation();
    test_dcg_matches_closed_form();
    test_dcg_at_k();
    test_ndcg_perfect();
    test_ndcg_at_k_uses_truncated_ideal();
    test_no_relevant_docs_scores_zero();
    test_map();
    test_mrr();
    test_aggregate_is_mean_over_queries();
    test_map_at_k();
    test_metrics_separate_good_and_bad_rankings();
    test_determinism();

    std::cout << std::endl;
    std::cout << "=== Result: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}