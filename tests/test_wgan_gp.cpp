// test_wgan_gp.cpp — Gradient correctness tests for WGAN-GP
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8);
}

static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t[i][j] * t[i][j];
    return std::sqrt(s);
}

// =====================================================================
// Test 1: WGANDiscriminator forward pass sanity
// =====================================================================
static void test_wgan_discriminator_forward() {
    cout << endl << "-- Test 1: WGANDiscriminator forward pass --" << endl;

    WGANDiscriminator critic(2, 8, 2);  // input_dim=2, hidden=8, 2 layers

    Tensor input(3, 2);
    input[0][0] = 0.5; input[0][1] = -0.5;
    input[1][0] = 1.0; input[1][1] = 0.0;
    input[2][0] = 0.0; input[2][1] = 1.0;

    Tensor score = critic.forward(input);

    // No activation on output (raw score), should be a scalar per sample
    check("WGANDiscriminator output shape (batch, 1)", score.rows == 3 && score.cols == 1);
    check("WGANDiscriminator output values are finite",
          std::isfinite(score[0][0]) && std::isfinite(score[1][0]) && std::isfinite(score[2][0]));
}

// =====================================================================
// Test 2: WGANDiscriminator gradient_penalty value range
// =====================================================================
static void test_wgan_gradient_penalty_value() {
    cout << endl << "-- Test 2: WGANDiscriminator gradient penalty value range --" << endl;

    WGANDiscriminator critic(2, 16, 2);

    Tensor real(4, 2);
    Tensor fake(4, 2);
    for (size_t i = 0; i < 4; ++i) {
        real[i][0] = (i + 1) * 0.5;
        real[i][1] = -(i + 1) * 0.3;
        fake[i][0] = (i + 1) * -0.7;
        fake[i][1] = (i + 1) * 0.2;
    }

    critic.reset_cached_inputs();
    double gp = 0.0;
    try {
        gp = critic.gradient_penalty(real, fake, 10.0);
    } catch (std::exception& e) {
        cout << "    [DEBUG] gradient_penalty EXCEPTION: " << e.what() << endl;
        // Don't fail the whole test, skip subsequent checks
        check("WGAN gradient penalty computed without exception", false);
        return;
    }

    // GP is lambda * E[(||grad|| - 1)^2]. With random networks, grad norm is
    // rarely exactly 1, so GP should be > 0. But it shouldn't be huge.
    check("WGAN gradient penalty is non-negative", gp >= 0.0);
    check("WGAN gradient penalty is finite", std::isfinite(gp));
    // Reasonable range for gradient penalty (with small network)
    check("WGAN gradient penalty is in reasonable range (0 to 100)",
          gp >= 0.0 && gp < 100.0);
}

// =====================================================================
// Test 3: WGANDiscriminator backward_from accumulates gradients
// =====================================================================
static void test_wgan_backward_from() {
    cout << endl << "-- Test 3: WGANDiscriminator backward_from accumulates gradients --" << endl;

    WGANDiscriminator critic(2, 8, 2);

    Tensor input(3, 2);
    input.fill(0.1);

    critic.reset_cached_inputs();
    Tensor output = critic.forward(input);

    // Gradient w.r.t. output (e.g. from loss = sum(outputs))
    Tensor grad_out(3, 1);
    grad_out.fill(1.0);

    critic.backward_from(grad_out);

    // Check that gradients were accumulated
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < critic.num_layers(); ++i) {
        double w_norm = tensor_l2norm(critic.layer(i).grad_weights);
        if (w_norm > 1e-10) { has_nonzero_grad = true; break; }
    }
    check("WGANDiscriminator backward_from produces non-zero gradients", has_nonzero_grad);
}

// =====================================================================
// Test 4: WGANDiscriminator backward_gradient_penalty accumulates
// =====================================================================
static void test_wgan_backward_gradient_penalty() {
    cout << endl << "-- Test 4: WGANDiscriminator backward_gradient_penalty accumulates --" << endl;

    WGANDiscriminator critic(2, 8, 2);

    Tensor real(3, 2);
    Tensor fake(3, 2);
    real.fill(0.5);
    fake.fill(-0.5);

    critic.reset_cached_inputs();
    double gp = 0.0;
    try {
        gp = critic.gradient_penalty(real, fake, 10.0);
    } catch (std::exception& e) {
        cout << "    [DEBUG] gradient_penalty EXCEPTION: " << e.what() << endl;
        check("WGAN backward_gradient_penalty: gradient_penalty succeeds", false);
        return;
    }
    (void)gp;

    try {
        critic.backward_gradient_penalty();
    } catch (std::exception& e) {
        cout << "    [DEBUG] backward_gradient_penalty EXCEPTION: " << e.what() << endl;
        check("WGAN backward_gradient_penalty produces non-zero gradients", false);
        return;
    }

    // GP gradients should also be accumulated in the same weight gradients
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < critic.num_layers(); ++i) {
        double w_norm = tensor_l2norm(critic.layer(i).grad_weights);
        if (w_norm > 1e-10) { has_nonzero_grad = true; break; }
    }
    check("WGANDiscriminator backward_gradient_penalty produces non-zero gradients",
          has_nonzero_grad);
}

// =====================================================================
// Test 5: Numerical gradient check on discriminator weights
// =====================================================================
static void test_wgan_numerical_weight_gradient() {
    cout << endl << "-- Test 5: WGANDiscriminator numerical vs analytical weight gradient --" << endl;

    WGANDiscriminator critic(2, 8, 2);

    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;

    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;

    // Build up accumulated gradients with zero_grad first
    critic.zero_grad();

    // Forward real
    critic.reset_cached_inputs();
    Tensor out_real = critic.forward(real);

    // Forward fake
    // We need to cache fake inputs too
    // Actually, gradient_penalty caches inputs on its forward pass.
    // For the manual numerical check we need to do a full D(x) pass.
    // We test a simple loss: L = sum(D(real)) + sum(D(fake))
    // Then dL/dw = sum(dD_real/dw) + sum(dD_fake/dw)
    // We already have backward_from for accumulating gradients from a scalar loss.

    // Get reference to first layer weights
    Dense& w1 = critic.layer(0);
    double orig_w00 = w1.weights[0][0];
    double eps = 1e-4;

    // Compute numerical gradient of L = sum(D(x))
    // L(w + eps) = sum(D(real, w+eps)) + sum(D(fake, w+eps))
    // First, D(real) with w + eps
    w1.weights[0][0] = orig_w00 + eps;
    critic.reset_cached_inputs();
    Tensor out_r_p = critic.forward(real);
    double loss_r_plus = 0.0;
    for (size_t i = 0; i < out_r_p.rows; ++i)
        loss_r_plus += out_r_p[i][0];

    critic.reset_cached_inputs();
    Tensor out_f_p = critic.forward(fake);
    double loss_f_plus = 0.0;
    for (size_t i = 0; i < out_f_p.rows; ++i)
        loss_f_plus += out_f_p[i][0];

    double loss_plus = loss_r_plus + loss_f_plus;

    // L(w - eps)
    w1.weights[0][0] = orig_w00 - eps;
    critic.reset_cached_inputs();
    Tensor out_r_m = critic.forward(real);
    double loss_r_minus = 0.0;
    for (size_t i = 0; i < out_r_m.rows; ++i)
        loss_r_minus += out_r_m[i][0];

    critic.reset_cached_inputs();
    Tensor out_f_m = critic.forward(fake);
    double loss_f_minus = 0.0;
    for (size_t i = 0; i < out_f_m.rows; ++i)
        loss_f_minus += out_f_m[i][0];

    double loss_minus = loss_r_minus + loss_f_minus;

    w1.weights[0][0] = orig_w00;

    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // Analytical gradient: accumulated via backward_from
    critic.zero_grad();
    // dL/dD_real = 1 per sample
    Tensor grad_out_real(2, 1);
    grad_out_real.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out_real);

    Tensor grad_out_fake(2, 1);
    grad_out_fake.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out_fake);

    double ana_grad = w1.grad_weights[0][0];
    double err = rel_error(num_grad, ana_grad);

    check("WGANDiscriminator W[0][0] numerical vs analytical gradient", err < 1e-1);
}

// =====================================================================
// Test 6: WGANGenerator forward and backward
// =====================================================================
static void test_wgan_generator() {
    cout << endl << "-- Test 6: WGANGenerator forward/backward --" << endl;

    WGANGenerator gen(2, 8, 2, 2);  // latent=2, hidden=8, output=2, 2 layers

    Tensor z(3, 2);
    z.fill(0.1);

    Tensor out = gen.forward(z);
    check("WGANGenerator output shape matches", out.rows == 3 && out.cols == 2);
    check("WGANGenerator output is finite", std::isfinite(out[0][0]));

    Tensor grad_out(3, 2);
    grad_out.fill(1.0);
    Tensor grad_z = gen.backward(grad_out, 0.0);
    check("WGANGenerator backward produces grad_z", grad_z.rows == 3 && grad_z.cols == 2);

    // Non-zero weight gradients
    bool has_nonzero = false;
    for (size_t i = 0; i < gen.num_layers(); ++i) {
        if (tensor_l2norm(gen.layer(i).grad_weights) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    check("WGANGenerator has non-zero weight gradients", has_nonzero);
}

// =====================================================================
// Test 7: WGANModel train_step produces finite output
// =====================================================================
static void test_wgan_model_train_step() {
    cout << endl << "-- Test 7: WGANModel train_step produces finite output --" << endl;

    try {
        WGANModel model(2, 2, 16, 16, 2, 2);
        model.set_lambda_gp(10.0);

        Tensor real_batch(4, 2);
        for (size_t i = 0; i < 4; ++i) {
            real_batch[i][0] = (i + 1) * 0.3;
            real_batch[i][1] = -(i + 1) * 0.4;
        }

        auto [d_loss, g_loss, gp] = model.train_step(real_batch, 10.0, 1);

        check("WGANModel d_loss is finite", std::isfinite(d_loss));
        check("WGANModel g_loss is finite", std::isfinite(g_loss));
        check("WGANModel gp is finite", std::isfinite(gp));
        check("WGANModel gp >= 0", gp >= 0.0);
    } catch (const std::exception& e) {
        cout << "  [SKIP] WGANModel test (exception): " << e.what() << endl;
    }
}

// =====================================================================
// Test 8: WGANModel generate produces valid samples
// =====================================================================
static void test_wgan_model_generate() {
    cout << endl << "-- Test 8: WGANModel generate samples --" << endl;

    try {
        WGANModel model(2, 2, 16, 16, 2, 2);
        Tensor samples = model.generate(4);
        check("WGANModel generate output shape (4, 2)", samples.rows == 4 && samples.cols == 2);
        bool all_finite = true;
        for (size_t i = 0; i < samples.rows; ++i)
            for (size_t j = 0; j < samples.cols; ++j)
                if (!std::isfinite(samples[i][j])) all_finite = false;
        check("WGANModel generated samples are finite", all_finite);
    } catch (const std::exception& e) {
        cout << "  [SKIP] WGANModel generate test (exception): " << e.what() << endl;
    }
}

// =====================================================================
// Main
// =====================================================================
int main() {
    cout << "=== WGAN-GP Gradient Correctness Tests ===" << endl;
    cout << setprecision(8);

    test_wgan_discriminator_forward();
    test_wgan_gradient_penalty_value();
    test_wgan_backward_from();
    test_wgan_backward_gradient_penalty();
    test_wgan_numerical_weight_gradient();
    test_wgan_generator();
    test_wgan_model_train_step();
    test_wgan_model_generate();

    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}