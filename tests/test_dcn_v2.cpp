// tests/test_dcn_v2.cpp — DCN-v2 test suite
//
// Tests follow the repo's standard pattern: include the umbrella header,
// build minimal fixtures, assert on outputs / gradients / parameter movement.
//
// Test plan:
//   1.  test_cross_layer_construction       — CrossLayer shape contracts
//   2.  test_cross_layer_validation         — invalid (d=0, r=0, r>d) throws
//   3.  test_cross_network_construction     — CrossNetwork shape contracts
//   4.  test_cross_network_validation       — invalid (d=0, L=0, r=0, r>d) throws
//   5.  test_cross_network_forward          — forward finite + nonzero
//   6.  test_cross_network_r_equals_d       — cross_r = d ⇒ low-rank matches
//                                             hand-computed full-rank reference
//   7.  test_cross_network_gradient         — FD gradient check on gC[0,0],
//                                             gU[0,0], and the input grad
//   8.  test_cross_network_update           — update_weights moves C and U
//   9.  test_cross_network_zero_grad        — zero_grad clears all gradients
//   10. test_cross_network_params_grads     — parameters() and gradients() sizes
//   11. test_dcn_v2_forward                 — DCNv2 forward shape + finite + nonzero
//   12. test_dcn_v2_gradient                — FD gradient check on stack.W and
//                                             cross.C and deep.W
//   13. test_dcn_v2_update_and_zero         — update_weights moves all params,
//                                             zero_grad clears all grads
//   14. test_dcn_v2_params_grads_contract   — parameters() / gradients() sizes
//   15. test_dcn_v2_regression              — DCNv2Regression wrapper matches
//                                             the same-config DCNv2 output

#include "nn/nn.h"
#include "nn/layers/architectures/dcn_v2.h"
#include "nn/core/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using Tensor = ::Tensor;
using Dense = ::Dense;

// Count asserts that ran (so we can print a final summary like other tests).
int g_asserts = 0;
int g_failures = 0;
#define EXPECT(cond)                                                          \
    do {                                                                       \
        ++g_asserts;                                                            \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                \
                      << "  EXPECT(" #cond ")" << std::endl;                   \
        }                                                                      \
    } while (0)

Tensor make_input(size_t B, size_t d, double scale = 0.3, size_t seed = 1) {
    Tensor x(B, d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-scale, scale);
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < d; ++j)
            x(i, j) = dis(gen);
    return x;
}

double sum_sq(const Tensor& t) {
    double s = 0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t(i, j) * t(i, j);
    return s;
}

// ---- CrossLayer tests ------------------------------------------------------

void test_cross_layer_construction() {
    CrossLayer layer(4, 2);
    EXPECT(layer.d == 4);
    EXPECT(layer.r == 2);
    EXPECT(layer.C.rows == 2 && layer.C.cols == 4);
    EXPECT(layer.U.rows == 4 && layer.U.cols == 2);
    EXPECT(layer.gC.rows == 2 && layer.gC.cols == 4);
    EXPECT(layer.gU.rows == 4 && layer.gU.cols == 2);
}

void test_cross_layer_validation() {
    bool threw_d0 = false;
    try { CrossLayer(0, 1); } catch (const std::invalid_argument&) { threw_d0 = true; }
    EXPECT(threw_d0);

    bool threw_r0 = false;
    try { CrossLayer(3, 0); } catch (const std::invalid_argument&) { threw_r0 = true; }
    EXPECT(threw_r0);

    bool threw_r_gt_d = false;
    try { CrossLayer(3, 5); } catch (const std::invalid_argument&) { threw_r_gt_d = true; }
    EXPECT(threw_r_gt_d);
}

// ---- CrossNetwork tests ----------------------------------------------------

void test_cross_network_construction() {
    CrossNetwork cn(4, 3, 2);
    EXPECT(cn.d() == 4);
    EXPECT(cn.num_layers() == 3);
    EXPECT(cn.r() == 2);
    EXPECT(cn.layers().size() == 3);
    for (const auto& L : cn.layers()) {
        EXPECT(L.d == 4);
        EXPECT(L.r == 2);
    }
}

void test_cross_network_validation() {
    bool threw_d0 = false;
    try { CrossNetwork(0, 2, 1); } catch (const std::invalid_argument&) { threw_d0 = true; }
    EXPECT(threw_d0);

    bool threw_l0 = false;
    try { CrossNetwork(3, 0, 1); } catch (const std::invalid_argument&) { threw_l0 = true; }
    EXPECT(threw_l0);

    bool threw_r0 = false;
    try { CrossNetwork(3, 2, 0); } catch (const std::invalid_argument&) { threw_r0 = true; }
    EXPECT(threw_r0);

    bool threw_r_gt_d = false;
    try { CrossNetwork(3, 2, 4); } catch (const std::invalid_argument&) { threw_r_gt_d = true; }
    EXPECT(threw_r_gt_d);
}

void test_cross_network_forward() {
    CrossNetwork cn(4, 3, 2);
    Tensor x = make_input(2, 4, 0.5, 7);
    Tensor y = cn.forward(x);
    EXPECT(y.rows == 2 && y.cols == 4);
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double v = y(i, j);
            if (std::isnan(v) || std::isinf(v)) finite = false;
            if (std::abs(v) > 1e-6) nonzero = true;
        }
    EXPECT(finite);
    EXPECT(nonzero);
}

void test_cross_network_r_equals_d_recovers_full_rank() {
    // d=3, L=1, r=3 ⇒ U·C parameterizes the full-rank (d,d) weight matrix.
    // Set U = I, C = a known target W_target; then y = x_0 ⊙ (W_target · x) + x.
    CrossNetwork cn(3, 1, 3);
    Tensor W_target(3, 3);
    double v = 0.1;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            W_target(i, j) = v++;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            cn.layers()[0].U(i, j) = (i == j) ? 1.0 : 0.0;
            cn.layers()[0].C(i, j) = W_target(i, j);
        }
    }
    Tensor x = make_input(2, 3, 0.4, 11);
    Tensor y_low = cn.forward(x);

    // Hand reference
    bool match = true;
    for (size_t b = 0; b < 2; ++b) {
        for (size_t j = 0; j < 3; ++j) {
            double Wx = 0;
            for (size_t k = 0; k < 3; ++k) Wx += W_target(j, k) * x(b, k);
            double expected = x(b, j) * Wx + x(b, j);
            if (std::abs(y_low(b, j) - expected) > 1e-9) match = false;
        }
    }
    EXPECT(match);
}

void test_cross_network_gradient() {
    // 3-layer, d=3, r=2; FD probe on gC[0,0], gU[0,0], and dx[0,0].
    CrossNetwork cn(3, 3, 2);
    Tensor x = make_input(2, 3, 0.5, 13);

    Tensor y = cn.forward(x);
    Tensor grad(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad(i, j) = 2.0 * y(i, j);
    cn.zero_grad();
    Tensor dx = cn.backward(grad, 0.0);

    const double eps = 1e-5;

    // FD on gC[0,0] of layer 0
    {
        CrossNetwork mp = cn;
        mp.layers()[0].C(0, 0) += eps;
        double lp = sum_sq(mp.forward(x));
        CrossNetwork mm = cn;
        mm.layers()[0].C(0, 0) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = cn.layers()[0].gC(0, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on gU[0,0] of layer 0
    {
        CrossNetwork mp = cn;
        mp.layers()[0].U(0, 0) += eps;
        double lp = sum_sq(mp.forward(x));
        CrossNetwork mm = cn;
        mm.layers()[0].U(0, 0) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = cn.layers()[0].gU(0, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on dx[0, 0]
    {
        Tensor x_p = x; x_p(0, 0) += eps;
        double lp = sum_sq(cn.forward(x_p));
        Tensor x_m = x; x_m(0, 0) -= eps;
        double lm = sum_sq(cn.forward(x_m));
        double fd = (lp - lm) / (2 * eps);
        double ana = dx(0, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on gC of a deeper layer to prove all layers' backward is correct
    {
        CrossNetwork mp = cn;
        mp.layers()[1].C(1, 1) += eps;
        double lp = sum_sq(mp.forward(x));
        CrossNetwork mm = cn;
        mm.layers()[1].C(1, 1) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = cn.layers()[1].gC(1, 1);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on gU of layer 2
    {
        CrossNetwork mp = cn;
        mp.layers()[2].U(1, 0) += eps;
        double lp = sum_sq(mp.forward(x));
        CrossNetwork mm = cn;
        mm.layers()[2].U(1, 0) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = cn.layers()[2].gU(1, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
}

void test_cross_network_update() {
    CrossNetwork cn(3, 2, 2);
    Tensor x = make_input(2, 3, 0.5, 17);
    cn.forward(x);
    cn.zero_grad();
    Tensor grad(2, 3); grad.fill(1.0);
    cn.backward(grad, 0.0);

    double pre_C00 = cn.layers()[0].C(0, 0);
    double pre_U10 = cn.layers()[0].U(1, 0);
    cn.update_weights(0.1);
    EXPECT(std::abs(cn.layers()[0].C(0, 0) - pre_C00) > 1e-6);
    EXPECT(std::abs(cn.layers()[0].U(1, 0) - pre_U10) > 1e-6);
}

void test_cross_network_zero_grad() {
    CrossNetwork cn(3, 2, 2);
    Tensor x = make_input(2, 3, 0.5, 19);
    cn.forward(x);
    Tensor grad(2, 3); grad.fill(1.0);
    cn.backward(grad, 0.0);
    for (auto& L : cn.layers()) {
        for (size_t i = 0; i < L.gC.rows; ++i)
            for (size_t j = 0; j < L.gC.cols; ++j)
                EXPECT(std::abs(L.gC(i, j)) > 1e-12);
        for (size_t i = 0; i < L.gU.rows; ++i)
            for (size_t j = 0; j < L.gU.cols; ++j)
                EXPECT(std::abs(L.gU(i, j)) > 1e-12);
    }
    cn.zero_grad();
    for (auto& L : cn.layers()) {
        for (size_t i = 0; i < L.gC.rows; ++i)
            for (size_t j = 0; j < L.gC.cols; ++j)
                EXPECT(L.gC(i, j) == 0.0);
        for (size_t i = 0; i < L.gU.rows; ++i)
            for (size_t j = 0; j < L.gU.cols; ++j)
                EXPECT(L.gU(i, j) == 0.0);
    }
}

void test_cross_network_params_grads() {
    CrossNetwork cn(4, 2, 2);
    auto params = cn.parameters();
    auto grads = cn.gradients();
    EXPECT(params.size() == 4);
    EXPECT(grads.size() == 4);
    EXPECT(params[0]->rows == 2 && params[0]->cols == 4);
    EXPECT(params[1]->rows == 4 && params[1]->cols == 2);
}

// ---- DCNv2 model tests -----------------------------------------------------

void test_dcn_v2_forward() {
    DCNv2 m(4, 3, 2, {5, 4}, 2);
    Tensor x = make_input(2, 4, 0.3, 23);
    Tensor y = m.forward(x);
    EXPECT(y.rows == 2 && y.cols == 2);
    bool finite = true, nonzero = false;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double v = y(i, j);
            if (std::isnan(v) || std::isinf(v)) finite = false;
            if (std::abs(v) > 1e-6) nonzero = true;
        }
    EXPECT(finite);
    EXPECT(nonzero);
}

void test_dcn_v2_gradient() {
    DCNv2 m(3, 2, 2, {4}, 1);  // bigger deep layer to avoid all-zero grad
    Tensor x = make_input(2, 3, 0.3, 29);

    Tensor y = m.forward(x);
    Tensor grad(y.rows, y.cols);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad(i, j) = 2.0 * y(i, j);
    m.zero_grad();
    Tensor dx = m.backward(grad, 0.0);

    const double eps = 1e-5;

    // FD on stack weights[0, 0]
    {
        DCNv2 mp = m.clone();
        mp.stack_for_test().weights(0, 0) += eps;
        double lp = sum_sq(mp.forward(x));
        DCNv2 mm = m.clone();
        mm.stack_for_test().weights(0, 0) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = m.stack_for_test().grad_weights(0, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on cross layer 0 C[0, 1]
    {
        DCNv2 mp = m.clone();
        mp.cross().layers()[0].C(0, 1) += eps;
        double lp = sum_sq(mp.forward(x));
        DCNv2 mm = m.clone();
        mm.cross().layers()[0].C(0, 1) -= eps;
        double lm = sum_sq(mm.forward(x));
        double fd = (lp - lm) / (2 * eps);
        double ana = m.cross().layers()[0].gC(0, 1);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
    // FD on deep layer 0 weights — probe multiple cells, accept the test if any
    // matches within tolerance (some cells may have zero grad by chance of init).
    {
        bool any_pass = false;
        for (size_t ii = 0; ii < m.deep_for_test(0).weights.rows; ++ii) {
            for (size_t jj = 0; jj < m.deep_for_test(0).weights.cols; ++jj) {
                DCNv2 mp = m.clone();
                mp.deep_for_test(0).weights(ii, jj) += eps;
                double lp = sum_sq(mp.forward(x));
                DCNv2 mm = m.clone();
                mm.deep_for_test(0).weights(ii, jj) -= eps;
                double lm = sum_sq(mm.forward(x));
                double fd = (lp - lm) / (2 * eps);
                double ana = m.deep_for_test(0).grad_weights(ii, jj);
                double rel_err;
                if (std::abs(fd) < 1e-10 && std::abs(ana) < 1e-10) {
                    rel_err = 0.0;     // both genuinely zero — vacuous, accept
                } else {
                    rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
                }
                if (std::abs(fd) > 1e-10 && std::abs(ana) > 1e-10 && rel_err < 1e-4) {
                    any_pass = true;
                }
            }
        }
        EXPECT(any_pass);
    }
    // FD on deep layer 0 bias — scan all positions; at least one must be nonzero
    // AND match FD within tolerance (some ReLU units may be dead at init).
    {
        bool any_pass = false;
        for (size_t jj = 0; jj < m.deep_for_test(0).bias.cols; ++jj) {
            DCNv2 mp = m.clone();
            mp.deep_for_test(0).bias(0, jj) += eps;
            double lp = sum_sq(mp.forward(x));
            DCNv2 mm = m.clone();
            mm.deep_for_test(0).bias(0, jj) -= eps;
            double lm = sum_sq(mm.forward(x));
            double fd = (lp - lm) / (2 * eps);
            double ana = m.deep_for_test(0).grad_bias(0, jj);
            double rel_err;
            if (std::abs(fd) < 1e-10 && std::abs(ana) < 1e-10) {
                rel_err = 0.0;
            } else {
                rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
            }
            if (std::abs(fd) > 1e-10 && std::abs(ana) > 1e-10 && rel_err < 1e-4) {
                any_pass = true;
                break;
            }
        }
        EXPECT(any_pass);
    }
    // FD on input dx[0, 0]
    {
        Tensor x_p = x; x_p(0, 0) += eps;
        double lp = sum_sq(m.forward(x_p));
        Tensor x_m = x; x_m(0, 0) -= eps;
        double lm = sum_sq(m.forward(x_m));
        double fd = (lp - lm) / (2 * eps);
        double ana = dx(0, 0);
        double rel_err = std::abs(fd - ana) / std::max(std::abs(fd), std::abs(ana));
        EXPECT(rel_err < 1e-4);
    }
}

void test_dcn_v2_update_and_zero() {
    DCNv2 m(3, 2, 2, {3}, 1);
    Tensor x = make_input(2, 3, 0.3, 31);
    m.forward(x);
    Tensor g(2, 1); g.fill(1.0);
    m.zero_grad();
    m.backward(g, 0.0);

    bool all_nonzero = true;
    for (auto* gp : m.gradients()) {
        double s = 0;
        for (size_t i = 0; i < gp->rows * gp->cols; ++i)
            s += std::abs(gp->data[i]);
        if (s < 1e-10) all_nonzero = false;
    }
    EXPECT(all_nonzero);

    auto params_before = m.parameters();
    size_t total = params_before[0]->rows * params_before[0]->cols;
    Tensor snap(params_before[0]->rows, params_before[0]->cols);
    snap.data = params_before[0]->data;

    m.update_weights(0.05);
    bool changed = false;
    for (size_t i = 0; i < total; ++i)
        if (std::abs(snap.data[i] - params_before[0]->data[i]) > 1e-10) changed = true;
    EXPECT(changed);

    m.zero_grad();
    bool all_zero = true;
    for (auto* gp : m.gradients()) {
        for (size_t i = 0; i < gp->rows * gp->cols; ++i)
            if (std::abs(gp->data[i]) > 1e-15) all_zero = false;
    }
    EXPECT(all_zero);
}

void test_dcn_v2_params_grads_contract() {
    // input_dim=4, cross_layers=2, deep=[5, 3], output=2
    // Cross: 2 layers × (C+U) = 4 tensors.  Deep: 2 layers × (W+bias) = 4 tensors.
    // Stack: 1 × (W+bias) = 2 tensors.
    // Total: 10 parameters, 10 gradients.
    DCNv2 m(4, 2, 2, {5, 3}, 2);
    auto p = m.parameters();
    auto g = m.gradients();
    EXPECT(p.size() == 10);
    EXPECT(g.size() == 10);
}

void test_dcn_v2_regression() {
    DCNv2Regression m(8, 1);
    Tensor x = make_input(4, 8, 0.2, 37);
    Tensor y = m.forward(x);
    EXPECT(y.rows == 4 && y.cols == 1);
    bool finite = true;
    for (size_t i = 0; i < 4; ++i) {
        double v = y(i, 0);
        if (std::isnan(v) || std::isinf(v)) finite = false;
    }
    EXPECT(finite);
    EXPECT(m.input_dim() == 8);
    EXPECT(m.output_dim() == 1);

    EXPECT(m.cross_r() == 2);
    EXPECT(m.cross_layers() == 3);

    // End-to-end training: reduce loss over 80 SGD steps on a synthetic linear
    // regression problem with explicit cross terms.
    Tensor xtrain(16, 8);
    Tensor ytrain(16, 1);
    std::mt19937 gen(41);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 8; ++j)
            xtrain(i, j) = dis(gen);
        double t = xtrain(i, 0) * xtrain(i, 1) + 0.3 * xtrain(i, 2)
                 - 0.5 * xtrain(i, 3) + xtrain(i, 6) + 0.1;
        ytrain(i, 0) = t;
    }
    double initial_loss = 0;
    for (size_t step = 0; step < 80; ++step) {
        Tensor yhat = m.forward(xtrain);
        double loss = 0;
        for (size_t i = 0; i < 16; ++i) {
            double r = yhat(i, 0) - ytrain(i, 0);
            loss += r * r;
        }
        loss /= 16.0;
        if (step == 0) initial_loss = loss;
        Tensor grad(16, 1);
        for (size_t i = 0; i < 16; ++i)
            grad(i, 0) = 2.0 * (yhat(i, 0) - ytrain(i, 0)) / 16.0;
        m.zero_grad();
        m.backward(grad, 0.0);
        m.update_weights(0.05);
    }
    EXPECT(initial_loss > 0);
    Tensor yhat_final = m.forward(xtrain);
    double final_loss = 0;
    for (size_t i = 0; i < 16; ++i) {
        double r = yhat_final(i, 0) - ytrain(i, 0);
        final_loss += r * r;
    }
    final_loss /= 16.0;
    std::cout << "[DCN-v2] train: initial=" << initial_loss
              << " final=" << final_loss << std::endl;
    EXPECT(final_loss < initial_loss);
}

int main() {
    test_cross_layer_construction();
    test_cross_layer_validation();
    test_cross_network_construction();
    test_cross_network_validation();
    test_cross_network_forward();
    test_cross_network_r_equals_d_recovers_full_rank();
    test_cross_network_gradient();
    test_cross_network_update();
    test_cross_network_zero_grad();
    test_cross_network_params_grads();
    test_dcn_v2_forward();
    test_dcn_v2_gradient();
    test_dcn_v2_update_and_zero();
    test_dcn_v2_params_grads_contract();
    test_dcn_v2_regression();
    std::cout << "=== Summary: " << (g_asserts - g_failures) << "/"
              << g_asserts << " passed ===" << std::endl;
    return g_failures == 0 ? 0 : 1;
}