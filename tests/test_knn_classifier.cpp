// test_knn_classifier.cpp — Tests for the k-Nearest-Neighbors classifier layer.
//
// Reference: Cover & Hart 1967 "Nearest Neighbor Pattern Classification"
//   and the distance-weighted soft-vote variants.
//
// What we test:
//   1.  Empty-support fallback: forward returns uniform probabilities,
//       backward returns zero gradients.
//   2.  Forward shape: (batch, feature_dim) -> (batch, num_classes).
//   3.  Output rows sum to 1 (proper probability simplex).
//   4.  Output is finite (no NaN/Inf).
//   5.  Output is non-trivial (some mass on the right class).
//   6.  k=1, query == one of the support points => output is one-hot on that label.
//   7.  Nearest-neighbor sanity: when class clusters are well-separated,
//       the predicted label matches the ground truth.
//   8.  Cosine-cluster ordering: queries from a known cluster point mostly
//       to that cluster's class.
//   9.  Input gradient check (L2 loss, 2 support points per class, k=2).
//  10.  Input gradient check with k > support-count clamps correctly.
//  11.  Input gradient check with k=1 (sharp voting).
//  12.  Parameter / gradient accessors return empty (no learnable params).
//  13.  Determinism: two fresh instances with the same fit data produce
//       identical outputs.
//  14.  Training-step sanity: shifting the query towards an alternative
//       class flips the predicted class.
//  15.  Larger setup (multi-class blobs, k=5, batch=4) — gradient check
//       and forward/backward timings.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include "nn/layers/architectures/knn_classifier.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_err(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
}

// L2 loss and gradient helpers (0.5 * sum (y - target)^2).
static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

// Numerical gradient via centered finite differences. eps = 1e-5.
static Tensor numerical_grad_input(
    KNNClassifier& layer,
    Tensor& input,
    const Tensor& target,
    double eps = 1e-5)
{
    Tensor ng(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = layer.forward(input);
            double lp = l2_loss_value(out_p, target);

            input(i, j) = orig - eps;
            Tensor out_m = layer.forward(input);
            double lm = l2_loss_value(out_m, target);

            input(i, j) = orig;
            ng(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return ng;
}

// Build a 2-D point cloud with two well-separated clusters per class.
// Class 0 lives around (-3, 0); class 1 around (+3, 0); class 2 around (0, +3).
static void make_support_3class(size_t per_class, Tensor& sup, vector<int>& lab,
                                 size_t feature_dim = 2, unsigned seed = 42) {
    const size_t num_classes = 3;
    sup = Tensor(per_class * num_classes, feature_dim);
    lab.assign(per_class * num_classes, 0);
    mt19937 gen(seed);
    normal_distribution<> nd(0.0, 0.2);
    for (size_t c = 0; c < num_classes; ++c) {
        double cx = (c == 0) ? -3.0 : (c == 1 ? +3.0 : 0.0);
        double cy = (c == 2) ? +3.0 : 0.0;
        for (size_t i = 0; i < per_class; ++i) {
            size_t row = c * per_class + i;
            lab[row] = static_cast<int>(c);
            sup(row, 0) = cx + nd(gen);
            if (feature_dim > 1) sup(row, 1) = cy + nd(gen);
            for (size_t d = 2; d < feature_dim; ++d) sup(row, d) = nd(gen);
        }
    }
}

// =====================================================================
// Test 1: empty-support fallback
// =====================================================================
static void test_empty_support() {
    cout << endl << "--- Test 1: empty-support fallback ---" << endl;
    KNNClassifier layer(3, 4, /*k=*/2);
    check("not fitted at construction", !layer.is_fitted());

    Tensor query(2, 3);
    for (size_t i = 0; i < query.data.size(); ++i) query.data[i] = 0.1 * i;

    Tensor out = layer.forward(query);
    check("output shape (2, 4)", out.rows == 2 && out.cols == 4);

    // Each row must equal uniform 1/num_classes.
    bool uniform = true;
    double expected = 0.25;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (fabs(out(i, j) - expected) > 1e-9) uniform = false;
    check("rows equal uniform 1/4", uniform);

    // backward should return zero.
    Tensor grad_out(2, 4);
    grad_out.fill(0.0);
    Tensor gi = layer.backward(grad_out, 0.0);
    bool zero = true;
    for (size_t i = 0; i < gi.data.size(); ++i)
        if (fabs(gi.data[i]) > 0.0) zero = false;
    check("backward returns zero gradient", zero);
}

// =====================================================================
// Test 2: forward shape with simple support set
// =====================================================================
static void test_forward_shape() {
    cout << endl << "--- Test 2: forward shape ---" << endl;
    KNNClassifier layer(2, 3, /*k=*/2);
    Tensor sup(6, 2);
    vector<int> lab = {0, 0, 1, 1, 2, 2};
    for (size_t i = 0; i < 6; ++i) sup(i, 0) = (double)i;
    layer.fit(sup, lab);

    Tensor query(4, 2);
    for (size_t i = 0; i < 4; ++i) query(i, 0) = (double)i + 0.5;

    Tensor out = layer.forward(query);
    check("output (4, 3)", out.rows == 4 && out.cols == 3);
}

// =====================================================================
// Test 3: probability simplex
// =====================================================================
static void test_probability_simplex() {
    cout << endl << "--- Test 3: output rows sum to 1 ---" << endl;
    KNNClassifier layer(2, 3, /*k=*/3);
    Tensor sup(6, 2);
    vector<int> lab = {0, 0, 1, 1, 2, 2};
    for (size_t i = 0; i < 6; ++i) sup(i, 0) = (double)i;
    layer.fit(sup, lab);

    Tensor query(5, 2);
    for (size_t i = 0; i < 5; ++i) {
        query(i, 0) = (double)i;
        query(i, 1) = 0.5;
    }
    Tensor out = layer.forward(query);
    bool simplex = true;
    for (size_t i = 0; i < out.rows; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < out.cols; ++j) s += out(i, j);
        if (fabs(s - 1.0) > 1e-9) simplex = false;
    }
    check("every row sums to 1", simplex);
}

// =====================================================================
// Test 4: output is finite
// =====================================================================
static void test_output_finite() {
    cout << endl << "--- Test 4: output is finite ---" << endl;
    KNNClassifier layer(3, 5, /*k=*/3);
    Tensor sup(10, 3);
    vector<int> lab = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    mt19937 gen(1);
    normal_distribution<> nd(0.0, 1.0);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t d = 0; d < 3; ++d) sup(i, d) = nd(gen);
    }
    layer.fit(sup, lab);

    Tensor query(7, 3);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t d = 0; d < 3; ++d) query(i, d) = nd(gen);
    }
    Tensor out = layer.forward(query);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("all output entries finite", finite);
}

// =====================================================================
// Test 5: non-trivial output
//
// Even when some classes legitimately receive zero mass (the k=3 vote
// might not pull from every class for a given query), the output must
// still be a valid probability distribution and have nonzero total mass.
// =====================================================================
static void test_output_nontrivial() {
    cout << endl << "--- Test 5: output is a non-trivial probability distribution ---" << endl;
    KNNClassifier layer(2, 3, /*k=*/3);
    Tensor sup(6, 2);
    vector<int> lab = {0, 0, 1, 1, 2, 2};
    sup(0, 0) = 0.0; sup(0, 1) = 0.0;
    sup(1, 0) = 0.5; sup(1, 1) = 0.0;
    sup(2, 0) = 5.0; sup(2, 1) = 5.0;
    sup(3, 0) = 5.5; sup(3, 1) = 5.0;
    sup(4, 0) = 5.0; sup(4, 1) = 5.5;
    sup(5, 0) = 5.5; sup(5, 1) = 5.5;
    layer.fit(sup, lab);

    // Place query equidistant from clusters 0 and 1 so the top-3 will pull
    // from both (and possibly from cluster 2 if the margin is right).
    Tensor query(2, 2);
    query(0, 0) = 2.5; query(0, 1) = 2.5;
    query(1, 0) = 0.3; query(1, 1) = 0.0;   // near cluster 0
    Tensor out = layer.forward(query);

    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("all entries finite", finite);

    // Each row must sum to 1.
    bool simplex = true;
    for (size_t b = 0; b < out.rows; ++b) {
        double s = 0.0;
        for (size_t c = 0; c < out.cols; ++c) s += out(b, c);
        if (fabs(s - 1.0) > 1e-9) simplex = false;
    }
    check("each row sums to 1", simplex);

    // Total mass > 0.
    double total = out.sum();
    check("total mass > 0", total > 0.0);
}

// =====================================================================
// Test 6: k=1, query == support point -> one-hot
// =====================================================================
static void test_k1_one_hot() {
    cout << endl << "--- Test 6: k=1, query identical to support point -> one-hot ---" << endl;
    KNNClassifier layer(2, 3, /*k=*/1);
    Tensor sup(3, 2);
    vector<int> lab = {0, 1, 2};
    sup(0, 0) = 0.0; sup(0, 1) = 0.0;
    sup(1, 0) = 1.0; sup(1, 1) = 0.0;
    sup(2, 0) = 0.0; sup(2, 1) = 1.0;
    layer.fit(sup, lab);

    Tensor query(3, 2);
    query(0, 0) = 0.0; query(0, 1) = 0.0;   // == sup 0  -> class 0
    query(1, 0) = 1.0; query(1, 1) = 0.0;   // == sup 1  -> class 1
    query(2, 0) = 0.0; query(2, 1) = 1.0;   // == sup 2  -> class 2
    Tensor out = layer.forward(query);
    bool onehot = true;
    for (size_t b = 0; b < 3; ++b) {
        for (size_t c = 0; c < 3; ++c) {
            double expected = (c == static_cast<size_t>(lab[b])) ? 1.0 : 0.0;
            if (fabs(out(b, c) - expected) > 1e-9) onehot = false;
        }
    }
    check("each row is one-hot at the support's label", onehot);
}

// =====================================================================
// Test 7: nearest-neighbor classification correctness on well-separated data
// =====================================================================
static void test_classification_correctness() {
    cout << endl << "--- Test 7: well-separated clusters -> perfect classification ---" << endl;
    KNNClassifier layer(2, 3, /*k=*/1, /*temperature=*/1.0);
    Tensor sup;
    vector<int> lab;
    make_support_3class(/*per_class=*/4, sup, lab, /*feature_dim=*/2, /*seed=*/7);
    layer.fit(sup, lab);

    // Query each cluster's center -> should classify correctly.
    Tensor query(3, 2);
    query(0, 0) = -3.0; query(0, 1) =  0.0;
    query(1, 0) = +3.0; query(1, 1) =  0.0;
    query(2, 0) =  0.0; query(2, 1) = +3.0;
    Tensor out = layer.forward(query);
    size_t pred[3];
    for (size_t b = 0; b < 3; ++b) {
        size_t best = 0; double bestv = out(b, 0);
        for (size_t c = 1; c < 3; ++c) {
            if (out(b, c) > bestv) { bestv = out(b, c); best = c; }
        }
        pred[b] = best;
    }
    check("query near (-3, 0)  -> class 0", pred[0] == 0);
    check("query near (+3, 0)  -> class 1", pred[1] == 1);
    check("query near (0, +3)  -> class 2", pred[2] == 2);
}

// =====================================================================
// Test 8: cosine-cluster ordering with distance-weighted voting
// =====================================================================
static void test_distance_weighted_voting() {
    cout << endl << "--- Test 8: distance-weighted vote favors closer neighbors ---" << endl;
    KNNClassifier layer(2, 2, /*k=*/3, /*temperature=*/0.5);
    // Class 0 has two far-away supports at (-5, 0) and (-5.5, 0); class 1
    // has one close support at (+0.1, 0).
    Tensor sup(3, 2);
    vector<int> lab = {0, 0, 1};
    sup(0, 0) = -5.0;  sup(0, 1) = 0.0;
    sup(1, 0) = -5.5;  sup(1, 1) = 0.0;
    sup(2, 0) = +0.1;  sup(2, 1) = 0.0;
    layer.fit(sup, lab);

    Tensor query(1, 2);
    query(0, 0) = 0.0; query(0, 1) = 0.0;   // closer to class 1 support
    Tensor out = layer.forward(query);
    check("output finite", std::isfinite(out(0, 0)) && std::isfinite(out(0, 1)));
    check("class 1 mass > class 0 mass for query near +0.1", out(0, 1) > out(0, 0));
    check("rows sum to 1", fabs(out(0, 0) + out(0, 1) - 1.0) < 1e-9);
}

// =====================================================================
// Test 9: input gradient check (basic)
//
// Uses clearly-separated support points and queries so the top-k selection
// is stable under the eps=1e-5 numerical perturbation (no ties at the
// kth-distance boundary).
// =====================================================================
static void test_input_gradient_basic() {
    cout << endl << "--- Test 9: input gradient check (basic) ---" << endl;
    KNNClassifier layer(2, 2, /*k=*/2, /*temperature=*/1.0);
    Tensor sup(4, 2);
    vector<int> lab = {0, 0, 1, 1};
    // Two well-separated 2-point clusters per class.
    sup(0, 0) =  0.0; sup(0, 1) =  0.0;
    sup(1, 0) =  0.5; sup(1, 1) =  0.0;
    sup(2, 0) =  5.0; sup(2, 1) =  5.0;
    sup(3, 0) =  5.5; sup(3, 1) =  5.0;
    layer.fit(sup, lab);

    // Place queries strictly inside one cluster each, with margin to the
    // boundary (so the top-k is unambiguous under +/- 1e-5 perturbation).
    Tensor query(3, 2);
    query(0, 0) =  0.2;  query(0, 1) =  0.0;   // inside class 0
    query(1, 0) =  0.4;  query(1, 1) =  0.0;   // inside class 0
    query(2, 0) =  5.3;  query(2, 1) =  5.0;   // inside class 1
    Tensor target(3, 2);
    target(0, 0) = 1.0; target(0, 1) = 0.0;
    target(1, 0) = 0.7; target(1, 1) = 0.3;
    target(2, 0) = 0.0; target(2, 1) = 1.0;

    Tensor out = layer.forward(query);
    Tensor go = l2_loss_grad(out, target);
    Tensor ag = layer.backward(go, 0.0);
    Tensor ng = numerical_grad_input(layer, query, target);

    double max_err = 0.0;
    for (size_t i = 0; i < ag.data.size(); ++i) {
        max_err = max(max_err, rel_err(ag.data[i], ng.data[i]));
    }
    cout << "  input grad max rel err = " << max_err << endl;
    check("input gradient check (rel_err < 1e-5)", max_err < 1e-5);
}

// =====================================================================
// Test 10: gradient check with k clamped to support size
// =====================================================================
static void test_input_gradient_k_larger_than_support() {
    cout << endl << "--- Test 10: gradient check (k > support_count) ---" << endl;
    // 2 supports, k=5 (must be clamped to 2).
    KNNClassifier layer(2, 2, /*k=*/5, /*temperature=*/1.0);
    Tensor sup(2, 2);
    vector<int> lab = {0, 1};
    sup(0, 0) = 0.0; sup(0, 1) = 0.0;
    sup(1, 0) = 1.0; sup(1, 1) = 1.0;
    layer.fit(sup, lab);

    Tensor query(2, 2);
    query(0, 0) = 0.1; query(0, 1) = 0.2;
    query(1, 0) = 0.7; query(1, 1) = 0.8;
    Tensor target(2, 2);
    target(0, 0) = 1.0; target(0, 1) = 0.0;
    target(1, 0) = 0.0; target(1, 1) = 1.0;

    Tensor out = layer.forward(query);
    Tensor go = l2_loss_grad(out, target);
    Tensor ag = layer.backward(go, 0.0);
    Tensor ng = numerical_grad_input(layer, query, target);

    double max_err = 0.0;
    for (size_t i = 0; i < ag.data.size(); ++i) {
        max_err = max(max_err, rel_err(ag.data[i], ng.data[i]));
    }
    cout << "  input grad max rel err = " << max_err << endl;
    check("input gradient check (rel_err < 1e-5)", max_err < 1e-5);
}

// =====================================================================
// Test 11: gradient check, k=1 (sharp voting)
//
// Supports are well-separated along each axis so any query placed near the
// origin has exactly one unambiguously-nearest support.
// =====================================================================
static void test_input_gradient_k1() {
    cout << endl << "--- Test 11: input gradient check (k=1) ---" << endl;
    KNNClassifier layer(3, 4, /*k=*/1, /*temperature=*/1.0);
    Tensor sup(4, 3);
    vector<int> lab = {0, 1, 2, 3};
    // Place each support at a corner of a cube so each query has a unique
    // nearest support under eps perturbation.
    sup(0, 0) = 0.0; sup(0, 1) = 0.0; sup(0, 2) = 0.0;
    sup(1, 0) = 5.0; sup(1, 1) = 0.0; sup(1, 2) = 0.0;
    sup(2, 0) = 0.0; sup(2, 1) = 5.0; sup(2, 2) = 0.0;
    sup(3, 0) = 0.0; sup(3, 1) = 0.0; sup(3, 2) = 5.0;
    layer.fit(sup, lab);

    // Place queries close to one support each, with enough margin that the
    // nearest-support selection is stable under 1e-5 perturbation.
    Tensor query(2, 3);
    query(0, 0) = 0.1; query(0, 1) = 0.1; query(0, 2) = 0.1;   // near sup 0
    query(1, 0) = 4.9; query(1, 1) = 0.1; query(1, 2) = 0.1;   // near sup 1
    Tensor target(2, 4);
    target(0, 0) = 0.8; target(0, 1) = 0.1; target(0, 2) = 0.05; target(0, 3) = 0.05;
    target(1, 0) = 0.1; target(1, 1) = 0.7; target(1, 2) = 0.1;  target(1, 3) = 0.1;

    Tensor out = layer.forward(query);
    Tensor go = l2_loss_grad(out, target);
    Tensor ag = layer.backward(go, 0.0);
    Tensor ng = numerical_grad_input(layer, query, target);

    double max_err = 0.0;
    for (size_t i = 0; i < ag.data.size(); ++i) {
        max_err = max(max_err, rel_err(ag.data[i], ng.data[i]));
    }
    cout << "  input grad max rel err = " << max_err << endl;
    check("input gradient check (rel_err < 1e-5)", max_err < 1e-5);
}

// =====================================================================
// Test 12: parameters() and gradients() are empty
// =====================================================================
static void test_no_parameters() {
    cout << endl << "--- Test 12: no learnable parameters ---" << endl;
    KNNClassifier layer(4, 5, /*k=*/3);
    check("parameters() is empty", layer.parameters().empty());
    check("gradients() is empty", layer.gradients().empty());
    // update_weights is a no-op; should not crash.
    layer.update_weights(0.01);
    check("update_weights() is safe", true);
    layer.zero_grad();
    check("zero_grad() is safe", true);
}

// =====================================================================
// Test 13: determinism
// =====================================================================
static void test_determinism() {
    cout << endl << "--- Test 13: determinism ---" << endl;
    Tensor sup;
    vector<int> lab;
    make_support_3class(5, sup, lab, /*feature_dim=*/3, /*seed=*/123);

    KNNClassifier a(3, 3, /*k=*/3, /*temperature=*/1.0);
    KNNClassifier b(3, 3, /*k=*/3, /*temperature=*/1.0);
    a.fit(sup, lab);
    b.fit(sup, lab);

    Tensor query(4, 3);
    mt19937 gen(99);
    normal_distribution<> nd(0.0, 1.0);
    for (size_t i = 0; i < query.data.size(); ++i) query.data[i] = nd(gen);

    Tensor oa = a.forward(query);
    Tensor ob = b.forward(query);
    double max_diff = 0.0;
    for (size_t i = 0; i < oa.data.size(); ++i)
        max_diff = max(max_diff, fabs(oa.data[i] - ob.data[i]));
    cout << "  max |o_a - o_b| = " << max_diff << endl;
    check("two fresh instances produce identical outputs", max_diff == 0.0);
}

// =====================================================================
// Test 14: shift-query-towards-class-flips-prediction
// =====================================================================
static void test_query_shift_changes_prediction() {
    cout << endl << "--- Test 14: shifting the query changes the prediction ---" << endl;
    KNNClassifier layer(2, 2, /*k=*/3, /*temperature=*/0.5);
    Tensor sup(4, 2);
    vector<int> lab = {0, 0, 1, 1};
    sup(0, 0) = -1.0; sup(0, 1) =  0.0;
    sup(1, 0) = -0.8; sup(1, 1) =  0.1;
    sup(2, 0) =  1.0; sup(2, 1) =  0.0;
    sup(3, 0) =  0.9; sup(3, 1) = -0.1;
    layer.fit(sup, lab);

    Tensor query(2, 2);
    query(0, 0) = -0.5; query(0, 1) = 0.0;   // near class 0
    query(1, 0) =  0.5; query(1, 1) = 0.0;   // near class 1
    Tensor out = layer.forward(query);
    check("query[0] predicts class 0", out(0, 0) > out(0, 1));
    check("query[1] predicts class 1", out(1, 1) > out(1, 0));
}

// =====================================================================
// Test 15: larger setup with gradient check + timing
// =====================================================================
static void test_large_gradient_and_timing() {
    cout << endl << "--- Test 15: larger setup (3 classes, k=5, batch=4) ---" << endl;
    KNNClassifier layer(4, 3, /*k=*/5, /*temperature=*/1.0);
    Tensor sup;
    vector<int> lab;
    make_support_3class(/*per_class=*/8, sup, lab, /*feature_dim=*/4, /*seed=*/2025);
    layer.fit(sup, lab);

    Tensor query(4, 4);
    mt19937 gen(11);
    normal_distribution<> nd(0.0, 0.6);
    for (size_t i = 0; i < query.data.size(); ++i) query.data[i] = nd(gen);

    Tensor target(4, 3);
    for (size_t i = 0; i < target.rows; ++i) {
        // Pick the class whose cluster center is closest to the query as the
        // "desired" target (one-hot).
        double cx[3] = {-3.0, +3.0, 0.0};
        double cy[3] = { 0.0,  0.0, 3.0};
        size_t best = 0; double bd = 1e18;
        for (size_t c = 0; c < 3; ++c) {
            double dx = query(i, 0) - cx[c];
            double dy = query(i, 1) - cy[c];
            double d2 = dx*dx + dy*dy;
            if (d2 < bd) { bd = d2; best = c; }
        }
        for (size_t c = 0; c < 3; ++c) target(i, c) = (c == best) ? 1.0 : 0.0;
    }

    Tensor out = layer.forward(query);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("output finite", finite);

    Tensor go = l2_loss_grad(out, target);
    Tensor ag = layer.backward(go, 0.0);
    Tensor ng = numerical_grad_input(layer, query, target);

    double max_err = 0.0;
    for (size_t i = 0; i < ag.data.size(); ++i) {
        max_err = max(max_err, rel_err(ag.data[i], ng.data[i]));
    }
    cout << "  input grad max rel err = " << max_err << endl;
    check("input gradient check (rel_err < 1e-5)", max_err < 1e-5);

    // Timing sanity (forward + backward under 100ms at this size).
    auto t0 = chrono::steady_clock::now();
    for (int rep = 0; rep < 20; ++rep) {
        Tensor o = layer.forward(query);
        Tensor g = layer.backward(l2_loss_grad(o, target), 0.0);
        (void)o; (void)g;
    }
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count() / 20.0;
    cout << "  fwd+bwd per call: " << ms << " ms" << endl;
    check("fwd+bwd under 100 ms", ms < 100.0);
}

int main() {
    cout << "=== KNN Classifier Tests ===" << endl;
    test_empty_support();
    test_forward_shape();
    test_probability_simplex();
    test_output_finite();
    test_output_nontrivial();
    test_k1_one_hot();
    test_classification_correctness();
    test_distance_weighted_voting();
    test_input_gradient_basic();
    test_input_gradient_k_larger_than_support();
    test_input_gradient_k1();
    test_no_parameters();
    test_determinism();
    test_query_shift_changes_prediction();
    test_large_gradient_and_timing();

    cout << endl;
    cout << "=== Summary: " << passed << " passed, " << failed << " failed (of "
         << (passed + failed) << ") ===" << endl;
    return failed == 0 ? 0 : 1;
}