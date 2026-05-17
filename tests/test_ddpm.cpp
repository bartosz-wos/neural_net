// test_ddpm.cpp — DDPM unit tests
#include <iostream>
#include <cmath>
#include <cassert>
#include "nn/layers/generative/ddpm.h"
#include "nn/core/tensor.h"

using namespace std;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        cerr << "FAIL: " << msg << endl; \
        tests_failed++; \
    } else { \
        cout << "PASS: " << msg << endl; \
        tests_passed++; \
    } \
} while(0)

#define CHECK_FINITE(val, msg) do { \
    if (std::isnan(val) || std::isinf(val)) { \
        cerr << "FAIL: " << msg << " (got NaN/Inf: " << val << ")" << endl; \
        tests_failed++; \
    } else { \
        cout << "PASS: " << msg << " (val=" << val << ")" << endl; \
        tests_passed++; \
    } \
} while(0)

// -------------------------------------------------------------------
// Test 1: DDPMModel construction and training_forward loss is finite
// -------------------------------------------------------------------
void test_training_forward() {
    cout << "\n=== Test: training_forward ===" << endl;

    // Small params: in_ch=2, channels={8,16}, time_emb_dim=32, seq_len=8, T=50
    DDPMModel model(2, {8, 16}, 32, 8, 50);
    cout << "DDPMModel created." << endl;

    size_t feat_dim = static_cast<size_t>(2 * 8);  // in_ch * seq_len = 16
    Tensor x0(2, feat_dim);
    for (size_t i = 0; i < x0.rows; ++i)
        for (size_t j = 0; j < x0.cols; ++j)
            x0[i][j] = static_cast<float>(j % 10) * 0.1f;

    cout << "Input shape: " << x0.rows << "x" << x0.cols << endl;

    // Call training_forward
    Tensor eps_theta = model.training_forward(x0);
    cout << "UNet output shape: " << eps_theta.rows << "x" << eps_theta.cols << endl;

    // Check loss is finite
    float loss = model.loss(x0);
    CHECK_FINITE(loss, "loss is finite (no NaN/Inf)");

    // Check output shape matches input
    CHECK(eps_theta.rows == x0.rows && eps_theta.cols == x0.cols,
          "training_forward output shape matches input");

    // Check output values are finite
    bool all_finite = true;
    for (size_t i = 0; i < eps_theta.rows && all_finite; ++i)
        for (size_t j = 0; j < eps_theta.cols; ++j)
            if (std::isnan(eps_theta[i][j]) || std::isinf(eps_theta[i][j]))
                all_finite = false;
    CHECK(all_finite, "training_forward output has no NaN/Inf values");
}

// -------------------------------------------------------------------
// Test 2: denoise() output shape matches input
// -------------------------------------------------------------------
void test_denoise() {
    cout << "\n=== Test: denoise ===" << endl;

    DDPMModel model(2, {8, 16}, 32, 8, 50);
    size_t feat_dim = static_cast<size_t>(2 * 8);

    // Start from Gaussian noise
    Tensor x_t(1, feat_dim);
    for (size_t i = 0; i < x_t.rows; ++i)
        for (size_t j = 0; j < x_t.cols; ++j)
            x_t[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f;

    cout << "Input shape: " << x_t.rows << "x" << x_t.cols << endl;

    // Call denoise at t=25
    Tensor x_prev = model.denoise(x_t, 25);
    cout << "Denoised shape: " << x_prev.rows << "x" << x_prev.cols << endl;

    CHECK(x_prev.rows == x_t.rows && x_prev.cols == x_t.cols,
          "denoise output shape matches input");

    // Check values are finite
    bool all_finite = true;
    for (size_t i = 0; i < x_prev.rows && all_finite; ++i)
        for (size_t j = 0; j < x_prev.cols; ++j)
            if (std::isnan(x_prev[i][j]) || std::isinf(x_prev[i][j]))
                all_finite = false;
    CHECK(all_finite, "denoise output has no NaN/Inf values");

    // Check denoise at t=0 (should not add noise)
    Tensor x_t0(1, feat_dim);
    for (size_t j = 0; j < feat_dim; ++j) x_t0[0][j] = 0.5f;
    Tensor x_prev0 = model.denoise(x_t0, 0);
    CHECK(x_prev0.rows == x_t0.rows && x_prev0.cols == x_t0.cols,
          "denoise at t=0 output shape correct");
}

// -------------------------------------------------------------------
// Test 3: sample(1) output shape is (1, in_ch*seq_len)
// -------------------------------------------------------------------
void test_sample() {
    cout << "\n=== Test: sample ===" << endl;

    DDPMModel model(2, {8, 16}, 32, 8, 50);
    size_t feat_dim = static_cast<size_t>(2 * 8);

    cout << "Calling sample(1)..." << endl;
    Tensor sampled = model.sample(1);
    cout << "Sample shape: " << sampled.rows << "x" << sampled.cols << endl;

    CHECK(sampled.rows == 1 && sampled.cols == feat_dim,
          "sample(1) returns shape (1, in_ch*seq_len)");

    // Check values are finite
    bool all_finite = true;
    for (size_t i = 0; i < sampled.rows && all_finite; ++i)
        for (size_t j = 0; j < sampled.cols; ++j)
            if (std::isnan(sampled[i][j]) || std::isinf(sampled[i][j]))
                all_finite = false;
    CHECK(all_finite, "sample output has no NaN/Inf values");
}

// -------------------------------------------------------------------
// Test 4: training_forward with batch > 1
// -------------------------------------------------------------------
void test_training_batch() {
    cout << "\n=== Test: training_forward batch ===" << endl;

    DDPMModel model(2, {8, 16}, 32, 8, 50);
    size_t feat_dim = static_cast<size_t>(2 * 8);

    Tensor x0(4, feat_dim);
    for (size_t i = 0; i < x0.rows; ++i)
        for (size_t j = 0; j < x0.cols; ++j)
            x0[i][j] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;

    Tensor eps_theta = model.training_forward(x0);
    float loss = model.loss(x0);

    CHECK(eps_theta.rows == 4 && eps_theta.cols == feat_dim,
          "batch=4 training_forward shape correct");
    CHECK_FINITE(loss, "batch=4 loss is finite");
}

// -------------------------------------------------------------------
// Test 5: DenoisingUNet forward with integer timestep
// -------------------------------------------------------------------
void test_unet_forward() {
    cout << "\n=== Test: DenoisingUNet forward ===" << endl;

    DDPMModel model(2, {8, 16}, 32, 8, 50);
    size_t feat_dim = static_cast<size_t>(2 * 8);

    Tensor x(1, feat_dim);
    for (size_t j = 0; j < feat_dim; ++j)
        x[0][j] = static_cast<float>(j % 5) * 0.2f;

    // Forward with integer timestep
    Tensor out = model.unet().forward(x, 10);
    cout << "UNet forward shape: " << out.rows << "x" << out.cols << endl;

    CHECK(out.rows == 1 && out.cols == feat_dim,
          "UNet forward shape matches input");

    bool all_finite = true;
    for (size_t i = 0; i < out.rows && all_finite; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (std::isnan(out[i][j]) || std::isinf(out[i][j]))
                all_finite = false;
    CHECK(all_finite, "UNet forward output has no NaN/Inf");
}

// -------------------------------------------------------------------
// Main
// -------------------------------------------------------------------
int main() {
    cout << "========================================" << endl;
    cout << "         DDPM Unit Tests" << endl;
    cout << "========================================" << endl;

    try {
        test_training_forward();
        test_denoise();
        test_sample();
        test_training_batch();
        test_unet_forward();
    } catch (const std::exception& e) {
        cerr << "EXCEPTION: " << e.what() << endl;
        tests_failed++;
    }

    cout << "\n========================================" << endl;
    cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << endl;
    cout << "========================================" << endl;

    return tests_failed > 0 ? 1 : 0;
}