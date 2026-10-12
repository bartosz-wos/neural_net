// test_calibration_diagnostics.cpp — TDD tests for the calibration
// diagnostics that follow the shipped expected_calibration_error.
//
// Sources (fetched and read 2026-10-12, NOT from memory):
//   * adaptive calibration error / thresholded ACE
//       Nixon, Dusenberry, Zhang, Jerfel, Tran — "Measuring Calibration in
//       Deep Learning", arXiv:1904.01685, Sec. 5.2-5.3.
//   * classwise (multiclass) calibration error
//       Kull, Perello-Nieto, Kangsepp, Silva Filho, Song, Flach — "Beyond
//       temperature scaling: ... Dirichlet calibration", NeurIPS 2019,
//       arXiv:1910.12656, Eq. (4).

#include <iostream>
#include <iomanip>
#include <cmath>
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

// Build (N,2) binary tensors: probs[i] is the predicted P(y=1) for row i,
// labels[i] is the true label (1.0 / 0.0).
struct Bin {
    Tensor prob;
    Tensor lab;
};
static Bin make_binary(const std::vector<double>& probs,
                       const std::vector<double>& labels) {
    Bin b;
    b.prob = Tensor(static_cast<size_t>(probs.size()), 1);
    b.lab  = Tensor(static_cast<size_t>(labels.size()), 1);
    for (size_t i = 0; i < probs.size(); ++i) {
        b.prob(i, 0) = probs[i];
        b.lab(i, 0) = labels[i];
    }
    return b;
}

// ---------------------------------------------------------------------------
// Test 1: constructor/validation contract.
// ---------------------------------------------------------------------------
static void test_validation() {
    std::cout << std::endl << "-- Test 1: validation --" << std::endl;

    bool threw = false;
    try { adaptive_calibration_error(Tensor(0, 1), Tensor(0, 1), 1.0, 5, 1.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check("empty tensors throw", threw);

    threw = false;
    try {
        Bin b = make_binary({0.5}, {1.0});
        adaptive_calibration_error(b.lab, b.prob, 1.0, 0, 1.0);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("n_bins == 0 throws", threw);

    threw = false;
    try {
        Bin b = make_binary({0.5}, {1.0});
        adaptive_calibration_error(b.lab, b.prob, 1.0, 5, -1.0);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("threshold < 0 throws", threw);

    threw = false;
    try {
        Bin b = make_binary({0.5}, {1.0});
        thresholded_calibration_error(b.lab, b.prob, 1.0, 5, 1.5);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("threshold > 1 throws", threw);

    threw = false;
    try {
        Bin b = make_binary({0.5, 0.6}, {1.0});
        adaptive_calibration_error(b.lab, b.prob, 1.0, 5, 0.0);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("row count mismatch throws", threw);

    // A single-class (no positive) fixture is NOT an error, unlike ROC-AUC.
    // Every bin has accuracy 0, so the metric reduces to the mean confidence
    // -- a well-defined and useful measure of overconfidence. Test 2 pins the
    // value; here we only assert that it does not throw.
    threw = false;
    try {
        Bin b = make_binary({0.5, 0.6}, {0.0, 0.0});
        double v = adaptive_calibration_error(b.lab, b.prob, 1.0, 5, 0.0);
        std::cout << "    no-positive ACE = " << v << " (mean confidence)" << std::endl;
    }
    catch (const std::invalid_argument&) { threw = true; }
    check("single-class input does not throw", !threw);
}

// ---------------------------------------------------------------------------
// Test 2: perfectly calibrated data gives exactly 0.
// Every prediction equals the observed frequency of its bin.
// ---------------------------------------------------------------------------
static void test_zero_when_calibrated() {
    std::cout << std::endl << "-- Test 2: calibrated data -> 0 --" << std::endl;

    // 4 predictions at 0.5, 2 of which are positive => freq = 0.5 = pred.
    Bin b = make_binary({0.5, 0.5, 0.5, 0.5}, {1.0, 1.0, 0.0, 0.0});
    double ace = adaptive_calibration_error(b.lab, b.prob, 1.0, 1, 0.0);
    check("adaptive ECE == 0 when calibrated", close(ace, 0.0, 1e-12));

    double tace = thresholded_calibration_error(b.lab, b.prob, 1.0, 1, 0.0);
    check("thresholded ACE == 0 when calibrated (threshold 0 == ACE)",
          close(tace, ace, 1e-12));

    // Always-wrong predictor: pred 1.0 on every row, none positive.
    Bin w = make_binary({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    double ace2 = adaptive_calibration_error(w.lab, w.prob, 1.0, 1, 0.0);
    check("always-wrong predictor has ACE == 1.0", close(ace2, 1.0, 1e-12));

    // A single-class fixture is well defined, NOT an error (unlike ROC-AUC):
    // every bin has accuracy 0, so ACE reduces to the mean confidence.
    // 3 rows at 0.2/0.5/0.8, no positives, one bin -> mean conf = 0.5.
    Bin s = make_binary({0.2, 0.5, 0.8}, {0.0, 0.0, 0.0});
    double ace3 = adaptive_calibration_error(s.lab, s.prob, 1.0, 1, 0.0);
    std::cout << "    no-positive ACE = " << ace3 << " (expected 0.5)" << std::endl;
    check("no-positive fixture gives ACE == mean confidence", close(ace3, 0.5, 1e-12));
}

// ---------------------------------------------------------------------------
// Test 3: ACE uses EQUAL-COUNT bins, and weights each bin UNIFORMLY.
//
// Nixon Sec. 5.2: "We chose an adaptive scheme which spaces the bin
// intervals so that each contains an equal number of predictions." Sec. 5.3
// gives TACE = (1/(K*R)) * sum over classes and ranges of |acc - conf|,
// i.e. a UNIFORM average over the reported ranges -- NOT the sample-count
// weighting that expected_calibration_error uses. The two differ whenever
// bins are unequal in size, which for equal-count bins they are not.
//
// This fixture pins that difference: equal-count bins make the weights 1/R
// regardless of how clustered the probabilities are.
// ---------------------------------------------------------------------------
static void test_equal_count_bins_uniform_weights() {
    std::cout << std::endl << "-- Test 3: ACE uses equal-count bins, uniform 1/R weights --" << std::endl;

    // 8 rows, 2 bins of 4. Low bin (probs <= 0.5) is badly over-confident;
    // high bin is well calibrated. Uniform weighting lets the badly-wrong
    // small-probability bin contribute HALF the error.
    //   bin0: pred 0.5 each, 0/4 positive  -> |0.0 - 0.5| = 0.5
    //   bin1: pred 1.0 each, 4/4 positive  -> |1.0 - 1.0| = 0.0
    //   ACE = (0.5 + 0.0)/2 = 0.25
    Bin b = make_binary({0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0},
                        {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0});
    double ace = adaptive_calibration_error(b.lab, b.prob, 1.0, 2, 0.0);
    std::cout << "    ACE = " << ace << " (expected 0.25)" << std::endl;
    check("ACE == 0.25 on the two-bin fixture", close(ace, 0.25, 1e-12));

    // The SAMPLED expected_calibration_error on the SAME data is 0.5*4/8 = 0.25
    // too -- because with equal-count bins the weights coincide. So this
    // fixture alone cannot separate the two rules; the separating fixture is
    // Test 4, where bin sizes are UNEVEN.
}

// ---------------------------------------------------------------------------
// Test 4: the case that separates ACE from a count-weighted ECE.
//
// With n_bins > n/2 the equal-count scheme must produce UNEQUAL bin sizes
// (Nixon's scheme: "Calibration range r defined by the floor(N/R)th index of
// the sorted and thresholded predictions"). A count-weighted metric then
// cannot agree with a uniform one.
//
// N=3, R=2. Ranges are as equal as integer division allows: ceil(3/2)=2, so
// {0.2, 0.4} and {0.9} -- sizes 2 and 1, matching Sec. 5.2's "each contains
// an equal number of predictions".
//   bin0 {0.2,0.4}: acc 0/2 = 0.0, conf 0.3      -> |0.0-0.3| = 0.3
//   bin1 {0.9}    : acc 1/1 = 1.0, conf 0.9      -> |1.0-0.9| = 0.1
//   ACE (uniform)  = (0.3 + 0.1)/2 = 0.2
//   count-weighted = (2/3)*0.3 + (1/3)*0.1 = 0.23333...
// They differ, so exactly one of the two can be right. Note a literal
// `floor(3/2)=1` reading would give sizes 1 and 2 -- assigning the same two
// gaps to the opposite ends of the confidence range, for ACE 0.175. That is
// the only place this implementation departs from the paper's literal
// wording, in order to satisfy the property the same section states; the
// assertion below pins the ceil reading.
// ---------------------------------------------------------------------------
static void test_ace_differs_from_count_weighted() {
    std::cout << std::endl << "-- Test 4: ACE != count-weighted ECE (uneven bins) --" << std::endl;

    Bin b = make_binary({0.2, 0.4, 0.9}, {0.0, 0.0, 1.0});
    double ace = adaptive_calibration_error(b.lab, b.prob, 1.0, 2, 0.0);
    std::cout << "    ACE = " << ace << " (expected 0.2)" << std::endl;
    check("ACE == 0.2 (uniform over 2 unequal ranges)", close(ace, 0.2, 1e-12));

    double ece_uniform = expected_calibration_error(b.lab, b.prob, 2, 1.0,
                                                    CalibrationStrategy::Uniform);
    std::cout << "    uniform-binned ECE = " << ece_uniform << std::endl;
    check("uniform-binned ECE differs from ACE on this fixture",
          std::fabs(ece_uniform - ace) > 1e-6);
}

// ---------------------------------------------------------------------------
// Test 5: thresholding (Nixon Sec. 5.3) drops sub-threshold predictions.
//
// Only predictions STRICTLY above the threshold are evaluated. With
// threshold = 0.6, the 0.2/0.4 rows are excluded entirely, so the metric is
// computed on the 0.9 row alone.
//   pred 0.9, label 1 -> |1.0 - 0.9| = 0.1  => TACE = 0.1
// ---------------------------------------------------------------------------
static void test_thresholding_excludes_small_predictions() {
    std::cout << std::endl << "-- Test 5: TACE threshold excludes small predictions --" << std::endl;

    Bin b = make_binary({0.2, 0.4, 0.9}, {0.0, 0.0, 1.0});
    double tace = thresholded_calibration_error(b.lab, b.prob, 1.0, 1, 0.6);
    std::cout << "    TACE(0.6) = " << tace << " (expected 0.1)" << std::endl;
    check("TACE(0.6) == 0.1", close(tace, 0.1, 1e-12));

    // Threshold 0 must reduce exactly to ACE (Nixon: "ACE can be implemented
    // as TACE with the threshold set to 0"). Pinned as a non-vacuity assertion.
    double ace1 = adaptive_calibration_error(b.lab, b.prob, 1.0, 1, 0.0);
    double tace0 = thresholded_calibration_error(b.lab, b.prob, 1.0, 1, 0.0);
    check("TACE(threshold=0) == ACE", close(tace0, ace1, 1e-12));

    // Raising the threshold from 0.6 to 0.95 must exclude everything, and an
    // empty evaluation set is defined as 0.0 rather than NaN.
    double none = thresholded_calibration_error(b.lab, b.prob, 1.0, 1, 0.95);
    check("TACE with all rows filtered out == 0.0", close(none, 0.0, 1e-12));

    // BOUNDARY: the comparison is `>=`, not `>`. A prediction of exactly the
    // threshold is KEPT, so TACE(0.9) evaluates the single 0.9 row -- the
    // same 0.1 as TACE(0.6). With a strict `>` this would drop it and
    // return 0.0, i.e. "perfectly calibrated" for a model that just made a
    // 10%-wrong confident prediction.
    double bnd = thresholded_calibration_error(b.lab, b.prob, 1.0, 1, 0.9);
    std::cout << "    TACE(0.9) = " << bnd << " (expected 0.1 -- boundary is inclusive)" << std::endl;
    check("a prediction exactly AT the threshold is kept", close(bnd, 0.1, 1e-12));
}

// ---------------------------------------------------------------------------
// Test 6: classwise-ECE (Kull Eq. 4), per class and averaged.
//
// cwECE = (1/k) * sum_j sum_i (|B_{i,j}| / n) * |y_j(B_{i,j}) - p_j(B_{i,j})|
//
// The (1/k) is the average over classes; each class's own contribution is
// its OWN bin-size-weighted ECE (NOT the uniform 1/R weighting ACE uses).
// That asymmetry -- classwise weights by bin size, ACE does not -- is the
// whole point of the two being different metrics, so it is pinned here.
// ---------------------------------------------------------------------------
static void test_classwise_ece_binary() {
    std::cout << std::endl << "-- Test 6: classwise-ECE on binary data --" << std::endl;

    // Binary data expressed in the 2-COLUMN form the multiclass API requires:
    // p = [1-p1, p1]. 4 rows, the model says 0.25 everywhere, and the true
    // y=1 rate is 1/4 -> every class is calibrated -> classwise-ECE == 0.
    Tensor proba(4, 2);
    Tensor lab(4, 1);
    for (size_t i = 0; i < 4; ++i) {
        proba(i, 0) = 0.75;
        proba(i, 1) = 0.25;
    }
    const double y[4] = {0.0, 0.0, 0.0, 1.0};
    for (size_t i = 0; i < 4; ++i) lab(i, 0) = y[i];

    double cw = classwise_ece(lab, proba, 10);
    std::cout << "    cwECE = " << cw << " (expected 0.0)" << std::endl;
    check("classwise-ECE == 0 for the calibrated constant predictor",
          close(cw, 0.0, 1e-12));
}

// Test 6b: the classwise per-class decomposition must be exposed and must
// sum to cwECE.
//
// EXPECTED VALUES ARE NOT HAND-DERIVED. They were computed by an independent
// Python transcription of the equal-width binning (each score falls in bin
// floor(p*10), then sum (|B|/n)*|acc-conf|), cross-checked against the
// implementation. My first hand-derivation of this fixture was WRONG (it
// predicted 0.675/0.425; the correct values are 0.625/0.600/0.125) and the
// C++ was right -- the same "the test is not automatically right" trap the
// repo's own metrics session recorded, reached again from a different
// direction. The values below come from the oracle, not from my head.
static void test_classwise_decomposition() {
    std::cout << std::endl << "-- Test 6b: per-class decomposition sums to cwECE --" << std::endl;

    // 3 classes, 4 rows, supports (2, 1, 1) -- deliberately unequal so the
    // Macro and Weighted averages must differ.
    //   row: p0,  p1,  p2,  y
    //   0 : 0.10, 0.80, 0.10, 0
    //   1 : 0.10, 0.80, 0.10, 0
    //   2 : 0.80, 0.10, 0.10, 1
    //   3 : 0.10, 0.10, 0.80, 2
    Tensor proba(4, 3);
    Tensor lab(4, 1);
    const double p[4][3] = {{0.10, 0.80, 0.10},
                            {0.10, 0.80, 0.10},
                            {0.80, 0.10, 0.10},
                            {0.10, 0.10, 0.80}};
    const double y[4] = {0.0, 0.0, 1.0, 2.0};
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) proba(i, j) = p[i][j];
        lab(i, 0) = y[i];
    }

    std::vector<double> per = classwise_ece_per_class(lab, proba, 10);
    check("per-class decomposition has k entries", per.size() == 3);

    // Independent Python values on this fixture.
    const double expect0 = 0.625, expect1 = 0.600, expect2 = 0.125;
    std::cout << "    per = " << per[0] << ", " << per[1] << ", " << per[2] << std::endl;
    std::cout << "    expect " << expect0 << ", " << expect1 << ", " << expect2 << std::endl;
    check("class-0 ECE matches the independent oracle", close(per[0], expect0, 1e-9));
    check("class-1 ECE matches the independent oracle", close(per[1], expect1, 1e-9));
    check("class-2 ECE matches the independent oracle", close(per[2], expect2, 1e-9));
    check("the three classes genuinely differ (fixture is not degenerate)",
          std::fabs(per[0] - per[2]) > 1e-6);

    const double macro = (expect0 + expect1 + expect2) / 3.0;  // 1/k, k = 3
    double cw = classwise_ece(lab, proba, 10);
    std::cout << "    cwECE = " << cw << " (expected " << macro << ")" << std::endl;
    check("cwECE == mean of per-class ECEs", close(cw, macro, 1e-9));

    // MACRO and WEIGHTED must DIFFER here (supports are 2, 1, 1), which is
    // what makes the averaging-mode argument observable at all.
    double m = classwise_ece(lab, proba, 10, MulticlassAveraging::Macro);
    double w = classwise_ece(lab, proba, 10, MulticlassAveraging::Weighted);
    check("macro == plain cwECE", close(m, cw, 1e-9));
    const double wtd_expect = (2.0 * expect0 + 1.0 * expect1 + 1.0 * expect2) / 4.0;
    std::cout << "    weighted = " << w << " (expected " << wtd_expect << ")" << std::endl;
    check("weighted cwECE weights by class support", close(w, wtd_expect, 1e-9));
    check("macro and weighted genuinely differ", std::fabs(m - w) > 1e-6);
}

// Test 6c: the KEY property separating classwise-ECE from top-label ECE.
// A model can be perfectly confidence-calibrated (its top probability always
// matches its accuracy) while being badly miscalibrated on the non-top
// classes. Confidence-ECE cannot see that; classwise-ECE must.
// ---------------------------------------------------------------------------
static void test_classwise_sees_what_confidence_ece_cannot() {
    std::cout << std::endl
              << "-- Test 6c: classwise-ECE detects non-top-class miscalibration --" << std::endl;

    // 3 classes. The model is ALWAYS wrong, but always assigns 0.9 to the
    // (non-true) class it picks and splits the rest 0.05/0.05.
    //   row: p, label
    //   0 : [0.05, 0.90, 0.05], y=0   (predicts class 1 -- wrong)
    //   1 : [0.05, 0.90, 0.05], y=1   (predicts class 1 -- right)
    //   2 : [0.90, 0.05, 0.05], y=2   (predicts class 0 -- wrong)
    // Top-label confidence: 0.9 with accuracy 1/3 -> gap 0.5667.
    // Class 0: pred 0.05,0.05,0.90 ; true 1,0,0 -> bins:
    //    [0.05,0.1) has 2 rows, acc 1/2, conf 0.05 -> 0.45 * 2/4
    //    [0.9,1.0) has 1 row,  acc 0/1, conf 0.90 -> 0.90 * 1/4
    //    class0 ECE = 0.25*(0.45) + 0.25*(0.90) = 0.3375
    // Class 1: pred 0.90,0.90,0.05 ; true 0,1,0
    //    [0.9,1.0) has 2 rows, acc 1/2, conf 0.90 -> 0.40 * 2/4
    //    [0.05,0.1) has 1 row, acc 0/1, conf 0.05 -> 0.05 * 1/4
    //    class1 ECE = 0.5*0.40 + 0.25*0.05 = 0.2125
    // Class 2: pred 0.05,0.05,0.05 ; true 0,0,1
    //    all in [0.05,0.1): acc 1/3, conf 0.05 -> 0.28333 * 3/4
    //    class2 ECE = 0.75 * (1/3 - 0.05) = 0.2125
    Tensor proba(3, 3);
    Tensor lab(3, 1);
    const double p[3][3] = {{0.05, 0.90, 0.05}, {0.05, 0.90, 0.05}, {0.90, 0.05, 0.05}};
    const double y[3] = {0.0, 1.0, 2.0};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) proba(i, j) = p[i][j];
        lab(i, 0) = y[i];
    }

    // Values below come from the independent Python oracle, not from my head.
    // My first hand-derivation said 0.3375/0.2125/0.2125; the correct values
    // are 0.6/0.2833/0.2833 and the implementation matched the oracle exactly.
    std::vector<double> per = classwise_ece_per_class(lab, proba, 10);
    check("three classes reported", per.size() == 3);
    const double e0 = 0.6, e1 = 1.0 / 3.0 - 0.05, e2 = e1;
    std::cout << "    per = " << per[0] << ", " << per[1] << ", " << per[2] << std::endl;
    std::cout << "    expect " << e0 << ", " << e1 << ", " << e2 << std::endl;
    check("class-0 ECE matches the independent oracle", close(per[0], e0, 1e-9));
    check("class-1 ECE matches the independent oracle", close(per[1], e1, 1e-9));
    check("class-2 ECE matches the independent oracle", close(per[2], e2, 1e-9));
    check("at least one class is badly miscalibrated (fixture is non-degenerate)",
          std::max(std::max(per[0], per[1]), per[2]) > 0.3);
}

// ---------------------------------------------------------------------------
// Test 7: determinism -- the metrics are pure functions of the inputs.
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::cout << std::endl << "-- Test 7: determinism --" << std::endl;

    std::vector<double> probs = {0.9, 0.1, 0.7, 0.3, 0.55, 0.45, 0.2, 0.8};
    std::vector<double> labels = {1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Bin b = make_binary(probs, labels);

    double a1 = adaptive_calibration_error(b.lab, b.prob, 1.0, 4, 0.0);
    double a2 = adaptive_calibration_error(b.lab, b.prob, 1.0, 4, 0.0);
    check("adaptive ECE is bit-identical across calls", a1 == a2);

    double t1 = thresholded_calibration_error(b.lab, b.prob, 1.0, 4, 0.05);
    double t2 = thresholded_calibration_error(b.lab, b.prob, 1.0, 4, 0.05);
    check("thresholded ACE is bit-identical across calls", t1 == t2);

    // classwise-ECE takes the (N,C) probability form, so the binary fixture
    // is expanded to [1-p, p] rather than passed as an (N,1) column.
    Tensor proba(probs.size(), 2);
    Tensor lab(static_cast<size_t>(labels.size()), 1);
    for (size_t i = 0; i < probs.size(); ++i) {
        proba(i, 0) = 1.0 - probs[i];
        proba(i, 1) = probs[i];
        lab(i, 0) = labels[i];
    }
    double c1 = classwise_ece(lab, proba, 10);
    double c2 = classwise_ece(lab, proba, 10);
    check("classwise ECE is bit-identical across calls", c1 == c2);
}

// ---------------------------------------------------------------------------
// Test 8: bin-count monotonicity of ACE on a clustered fixture.
//
// ACE partitions the SORTED predictions into equal-count ranges, so changing
// R changes which predictions share a bin. On this fixture the finer split
// can only separate a well-calibrated group from a mis-calibrated one, so
// the metric is well-defined and reproducible at each R.
// ---------------------------------------------------------------------------
static void test_ace_varies_with_bin_count() {
    std::cout << std::endl << "-- Test 8: ACE responds to the bin count --" << std::endl;

    // Half the rows are perfectly predicted, half are perfectly wrong, and
    // all predictions sit at two distinct confidence levels.
    std::vector<double> probs = {0.99, 0.99, 0.99, 0.99, 0.51, 0.51, 0.51, 0.51};
    std::vector<double> labels = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    Bin b = make_binary(probs, labels);

    double r1 = adaptive_calibration_error(b.lab, b.prob, 1.0, 1, 0.0);
    double r2 = adaptive_calibration_error(b.lab, b.prob, 1.0, 2, 0.0);
    std::cout << "    R=1 -> " << r1 << " ; R=2 -> " << r2 << std::endl;
    // R=1 pools everything: acc 0.5, conf 0.75 -> gap 0.25.
    check("R=1 gives 0.25", close(r1, 0.25, 1e-12));
    // R=2 splits the 8 sorted rows into two ranges of 4 (ceil(8/2)=4).
    //   bin0: pred 0.99, all 4 correct -> |1.00 - 0.99| = 0.01
    //   bin1: pred 0.51, all 4 wrong   -> |0.00 - 0.51| = 0.51
    //   ACE = (0.01 + 0.51)/2 = 0.26
    check("R=2 gives 0.26", close(r2, 0.26, 1e-12));
    check("the two bin counts genuinely differ", std::fabs(r1 - r2) > 1e-6);
}

// Test 9: the equal-COUNT ranges are formed by SORTING ON CONFIDENCE, and the
// input rows are deliberately NOT already sorted.
//
// This fixture exists because the mutation harness found a real gap. Six
// mutations were caught, but "don't sort at all" SURVIVED -- and the reason
// was the fixtures, not the mutation: every fixture above uses at most two
// distinct probability levels, so the sorted and the input-order partitions
// contain exactly the same samples and agree to the last bit. Every one of
// them was measuring the binning rule and none of them was measuring the
// SORT.
//
// The rows below interleave six distinct confidences in a scrambled order, so
// confidence-ordering and input-order grouping genuinely differ. Independent
// Python gives 0.28125 sorted and 0.00625 unsorted -- a 45x gap, so a test
// pinned at the sorted value cannot pass by accident.
// ---------------------------------------------------------------------------
static void test_ace_bins_by_confidence_not_input_order() {
    std::cout << std::endl
              << "-- Test 9: ACE ranges are formed by sorting on confidence --" << std::endl;

    const std::vector<double> probs = {0.95, 0.05, 0.50, 0.45, 0.60, 0.40, 0.80, 0.20};
    const std::vector<double> labels = {1.0,  0.0,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0};
    Bin b = make_binary(probs, labels);

    // Sorted ascending: {0.05,0.20,0.40,0.45} (labels 0,0,0,0) then
    // {0.50,0.60,0.80,0.95} (labels 1,1,1,1).
    //   bin0: 0/4 correct, mean conf 0.275  -> |0.0   - 0.275  | = 0.275
    //   bin1: 4/4 correct, mean conf 0.7125 -> |1.0   - 0.7125 | = 0.2875
    //   ACE = (0.275 + 0.2875)/2 = 0.28125
    // The labels are perfectly anti-correlated with confidence, so the
    // sorted partition is exactly the worst case for an unsorted one.
    const double ace = adaptive_calibration_error(b.lab, b.prob, 1.0, 2, 0.0);
    std::cout << "    ACE = " << ace << " (expected 0.28125)" << std::endl;
    check("scrambled input yields the confidence-sorted ACE",
          close(ace, 0.28125, 1e-9));

    // Non-vacuity: the same data in DESCENDING confidence order must give the
    // SAME answer, because sorting makes the result permutation-invariant.
    const std::vector<double> rev_probs(probs.rbegin(), probs.rend());
    const std::vector<double> rev_labels(labels.rbegin(), labels.rend());
    Bin r = make_binary(rev_probs, rev_labels);
    double ace_rev = adaptive_calibration_error(r.lab, r.prob, 1.0, 2, 0.0);
    std::cout << "    ACE (reversed) = " << ace_rev << std::endl;
    check("permuting the rows does not change ACE", close(ace_rev, ace, 1e-12));

    // And the input-order (unsorted) grouping is a genuinely different number,
    // which is why M4 must be caught: unsorted bins would give 0.00625.
    check("the value is NOT the input-order grouping",
          std::fabs(ace - 0.00625) > 1e-3);
}

int main() {
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "=== Calibration Diagnostics Tests ===" << std::endl;

    test_validation();
    test_zero_when_calibrated();
    test_equal_count_bins_uniform_weights();
    test_ace_differs_from_count_weighted();
    test_thresholding_excludes_small_predictions();
    test_classwise_ece_binary();
    test_classwise_decomposition();
    test_classwise_sees_what_confidence_ece_cannot();
    test_determinism();
    test_ace_varies_with_bin_count();
    test_ace_bins_by_confidence_not_input_order();

    std::cout << std::endl;
    std::cout << "=== Result: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}