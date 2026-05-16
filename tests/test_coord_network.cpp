#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) {
        cout << "  [PASS] " << name << endl;
        ++passed;
    } else {
        cout << "  [FAIL] " << name << endl;
        ++failed;
    }
    return pass;
}

static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows * t.cols; ++i)
        s += t.data[i] * t.data[i];
    return std::sqrt(s);
}

static double sdf_circle(double x, double y) {
    return std::sqrt(x * x + y * y) - 1.0;
}

// =====================================================================
// Test 1: SDF learning with CoordinateNetwork (Fourier Features + MLP)
// =====================================================================
static void test_sdf_learning() {
    cout << endl << "-- Test 1: CoordinateNetwork SDF learning --" << endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_coord(-0.99, 0.99);

    size_t N = 500;
    Tensor coords(N, 2);
    Tensor targets(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double x = dist_coord(rng);
        double y = dist_coord(rng);
        coords[i][0] = x;
        coords[i][1] = y;
        targets[i][0] = sdf_circle(x, y);
    }

    CoordinateNetwork con(2, 256, {128, 64}, 1, false);

    double initial_loss = 0.0;
    {
        Tensor pred = con.forward(coords);
        for (size_t i = 0; i < N; ++i) {
            double err = pred[i][0] - targets[i][0];
            initial_loss += err * err;
        }
        initial_loss /= N;
    }

    int epochs = 300;
    double lr = 0.005;
    for (int ep = 0; ep < epochs; ++ep) {
        Tensor pred = con.forward(coords);
        Tensor loss_grad = pred - targets;
        con.zero_grad();
        Tensor grad_x = con.backward(loss_grad, lr);
        con.update_weights(lr);
        (void)grad_x;
    }

    Tensor pred = con.forward(coords);
    double final_loss = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double err = pred[i][0] - targets[i][0];
        final_loss += err * err;
    }
    final_loss /= N;

    check("CoN SDF: loss decreased after training", final_loss < initial_loss);
    check("CoN SDF: final loss < 0.1", final_loss < 0.1);
    check("CoN SDF: final loss < 0.05", final_loss < 0.05);
    check("CoN SDF: gradients are non-zero", tensor_l2norm(con.get_gradients()) > 1e-8);

    double test_x = 0.5, test_y = 0.5;
    Tensor test_pt(1, 2);
    test_pt[0][0] = test_x; test_pt[0][1] = test_y;
    Tensor test_out = con.forward(test_pt);
    double abs_err = std::abs(test_out[0][0] - sdf_circle(test_x, test_y));
    check("CoN SDF: prediction on test point (0.5, 0.5)", abs_err < 0.2);
}

// =====================================================================
// Test 2: High-frequency function — naive MLP fails, Fourier features succeed
// =====================================================================
static void test_frequency_encoding() {
    cout << endl << "-- Test 2: Fourier Features enable high-frequency learning --" << endl;

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto target_fn = [](double x) {
        return std::sin(2.0 * M_PI * 5.0 * x);
    };

    size_t N = 300;
    Tensor coords(N, 1);
    Tensor targets(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        coords[i][0] = x;
        targets[i][0] = target_fn(x);
    }

    Model naive_mlp;
    naive_mlp.add_layer(new Dense(1, 64));
    naive_mlp.add_layer(new Activation<ReLU>(ReLU()));
    naive_mlp.add_layer(new Dense(64, 64));
    naive_mlp.add_layer(new Activation<ReLU>(ReLU()));
    naive_mlp.add_layer(new Dense(64, 1));

    CoordinateNetwork fourier_con(1, 128, {64}, 1, false);

    SGD opt_naive(0.01);
    for (int ep = 0; ep < 500; ++ep) {
        naive_mlp.train(coords, targets, opt_naive, 1);
    }

    double lr = 0.005;
    for (int ep = 0; ep < 300; ++ep) {
        Tensor pred = fourier_con.forward(coords);
        Tensor loss_grad = pred - targets;
        fourier_con.zero_grad();
        fourier_con.backward(loss_grad, lr);
        fourier_con.update_weights(lr);
    }

    const size_t TEST = 50;
    double max_naive_err = 0.0, max_fourier_err = 0.0;
    double naive_loss = 0.0, fourier_loss = 0.0;

    for (double x = 0.01; x < 1.0; x += 1.0 / TEST) {
        Tensor test_pt(1, 1);
        test_pt[0][0] = x;
        double y_true = target_fn(x);

        Tensor naive_out = naive_mlp.forward(test_pt);
        double naive_err = std::abs(naive_out[0][0] - y_true);
        naive_loss += naive_err * naive_err;
        max_naive_err = std::max(max_naive_err, naive_err);

        Tensor fourier_out = fourier_con.forward(test_pt);
        double fourier_err = std::abs(fourier_out[0][0] - y_true);
        fourier_loss += fourier_err * fourier_err;
        max_fourier_err = std::max(max_fourier_err, fourier_err);
    }
    naive_loss /= TEST;
    fourier_loss /= TEST;

    check("High-freq: Fourier CoN max error < naive MLP max error",
          max_fourier_err < max_naive_err);
    check("High-freq: Fourier CoN loss < 0.1", fourier_loss < 0.1);
    check("High-freq: Fourier CoN beats naive on high freq",
          max_fourier_err < 0.3);
}

// =====================================================================
// Test 3: SIREN can learn SDF well
// =====================================================================
static void test_siren() {
    cout << endl << "-- Test 3: SIREN SDF learning --" << endl;

    std::mt19937 rng(999);
    std::uniform_real_distribution<double> dist_coord(-0.9, 0.9);

    size_t N = 400;
    Tensor coords(N, 2);
    Tensor targets(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double x = dist_coord(rng);
        double y = dist_coord(rng);
        coords[i][0] = x;
        coords[i][1] = y;
        targets[i][0] = sdf_circle(x, y);
    }

    SIREN siren(2, {128, 128}, 1, 30.0);

    double lr = 0.002;
    double initial_loss = 0.0;
    {
        Tensor pred = siren.forward(coords);
        for (size_t i = 0; i < N; ++i) {
            double err = pred[i][0] - targets[i][0];
            initial_loss += err * err;
        }
        initial_loss /= N;
    }

    for (int ep = 0; ep < 400; ++ep) {
        Tensor pred = siren.forward(coords);
        Tensor loss_grad = pred - targets;
        siren.zero_grad();
        siren.backward(loss_grad, lr);
        siren.update_weights(lr);
    }

    Tensor pred = siren.forward(coords);
    double final_loss = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double err = pred[i][0] - targets[i][0];
        final_loss += err * err;
    }
    final_loss /= N;

    check("SIREN: loss decreased after training", final_loss < initial_loss);
    check("SIREN: final loss < 0.1", final_loss < 0.1);
    check("SIREN: gradients are non-zero",
          tensor_l2norm(siren.get_gradients()) > 1e-8);

    Tensor test_pt(1, 2);
    test_pt[0][0] = 0.5; test_pt[0][1] = 0.5;
    Tensor test_out = siren.forward(test_pt);
    double abs_err = std::abs(test_out[0][0] - sdf_circle(0.5, 0.5));
    check("SIREN: SDF prediction accurate on test point", abs_err < 0.2);
}

// =====================================================================
// Test 4: FourierFeatures forward/backward shape and gradient
// =====================================================================
static void test_fourier_gradient() {
    cout << endl << "-- Test 4: FourierFeatures gradient correctness --" << endl;

    std::mt19937 rng(777);
    std::normal_distribution<double> dist(0.0, 1.0);

    GaussianFourierFeatures ff(2, 8, 1.0);

    Tensor coords(3, 2);
    for (size_t i = 0; i < 3; ++i) {
        coords[i][0] = dist(rng);
        coords[i][1] = dist(rng);
    }

    Tensor out = ff.forward(coords);
    check("FourierFeatures output shape (N, 2*F)", out.cols == 16 && out.rows == 3);

    Tensor grad_out(3, 16);
    grad_out.fill(1.0);
    ff.zero_grad();
    Tensor grad_x = ff.backward(grad_out, 0.0);

    check("FourierFeatures: gradient w.r.t. input is non-zero",
          tensor_l2norm(grad_x) > 1e-10);
    check("FourierFeatures: gradient w.r.t. frequencies is non-zero",
          tensor_l2norm(ff.get_gradients()) > 1e-10);

    // Verify that output values are in [-1, 1] (cos/sin range)
    double out_max = out.max();
    check("FourierFeatures: output values in valid range",
          out_max <= 1.0 + 1e-10 && out_max >= -1.0 - 1e-10);
}

// =====================================================================
// Test 5: LearnedFourierFeatures is trainable
// =====================================================================
static void test_learned_fourier() {
    cout << endl << "-- Test 5: LearnedFourierFeatures is trainable --" << endl;

    LearnedFourierFeatures lff(2, 64);

    std::mt19937 rng(888);
    std::uniform_real_distribution<double> dist(-0.8, 0.8);
    size_t N = 100;
    Tensor coords(N, 2);
    Tensor targets(N, 1);
    for (size_t i = 0; i < N; ++i) {
        coords[i][0] = dist(rng);
        coords[i][1] = dist(rng);
        targets[i][0] = coords[i][0] * coords[i][0] + coords[i][1] * coords[i][1];
    }

    Dense head(128, 32);
    head.init_weights("xavier");
    Dense head2(32, 1);
    head2.init_weights("xavier");

    double initial_loss = 0.0;
    {
        Tensor feat = lff.forward(coords);
        Tensor h = head.forward(feat);
        h = h.apply(ReLU());
        Tensor pred = head2.forward(h);
        for (size_t i = 0; i < N; ++i) {
            double err = pred[i][0] - targets[i][0];
            initial_loss += err * err;
        }
        initial_loss /= N;
    }

    double lr = 0.01;
    for (int ep = 0; ep < 200; ++ep) {
        lff.zero_grad();
        head.zero_grad();
        head2.zero_grad();

        Tensor feat = lff.forward(coords);
        Tensor h = head.forward(feat);
        h = h.apply(ReLU());
        Tensor pred = head2.forward(h);

        Tensor loss_grad = pred - targets;
        Tensor grad_h = head2.backward(loss_grad, 0.0);
        grad_h = grad_h.hadamard(h.apply([](double x){ return x > 0 ? 1.0 : 0.0; }));
        head.backward(grad_h, 0.0);
        lff.backward(head.get_gradients(), 0.0);

        head2.update_weights(lr);
        head.update_weights(lr);
    }

    Tensor feat_out = lff.forward(coords);
    check("LearnedFourierFeatures: produces (N, 2F) output",
          feat_out.cols == 128 && feat_out.rows == N);
    check("LearnedFourierFeatures: has non-zero gradients",
          tensor_l2norm(lff.get_gradients()) > 1e-10);
}

// =====================================================================
// Test 6: CoordinateNetwork vs SIREN on smooth function
// =====================================================================
static void test_coord_network_siren() {
    cout << endl << "-- Test 6: CoordinateNetwork with SIREN variant --" << endl;

    auto target_fn = [](double x) { return std::sin(2.0 * M_PI * 3.0 * x); };

    std::mt19937 rng(333);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    size_t N = 200;
    Tensor coords(N, 1);
    Tensor targets(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        coords[i][0] = x;
        targets[i][0] = target_fn(x);
    }

    CoordinateNetwork con(1, 64, {64}, 1, false);
    SIREN siren(1, {64, 64}, 1, 30.0);

    double lr = 0.01;

    for (int ep = 0; ep < 200; ++ep) {
        Tensor pred = con.forward(coords);
        Tensor loss_grad = pred - targets;
        con.zero_grad();
        con.backward(loss_grad, lr);
        con.update_weights(lr);

        pred = siren.forward(coords);
        loss_grad = pred - targets;
        siren.zero_grad();
        siren.backward(loss_grad, lr);
        siren.update_weights(lr);
    }

    double con_loss = 0.0, siren_loss = 0.0;
    for (double x = 0.01; x < 1.0; x += 0.02) {
        Tensor pt(1, 1); pt[0][0] = x;
        double y = target_fn(x);

        Tensor con_out = con.forward(pt);
        con_loss += (con_out[0][0] - y) * (con_out[0][0] - y);

        Tensor siren_out = siren.forward(pt);
        siren_loss += (siren_out[0][0] - y) * (siren_out[0][0] - y);
    }

    check("CoN + SIREN test: CoN learns sine wave", con_loss < 1.0);
    check("CoN + SIREN test: SIREN learns sine wave", siren_loss < 0.5);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== Coordinate Network Tests ===" << endl;
    cout << setprecision(8);

    test_sdf_learning();
    test_frequency_encoding();
    test_siren();
    test_fourier_gradient();
    test_learned_fourier();
    test_coord_network_siren();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;

    return (failed > 0) ? 1 : 0;
}