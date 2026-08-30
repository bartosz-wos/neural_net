// test_yarn_rope.cpp — Tests for YaRN (Yet another RoPE extensioN) layer.
//
// YaRN (Peng, Quesnelle, Fan, Shippole 2023, https://arxiv.org/abs/2309.00071)
// extends rotary position embeddings to longer contexts via:
//   1. NTK-aware per-dim frequency scaling:
//        freq_scale_by_dim[i] = 1 / (alpha · i / ((d/2) - 1) + 1)
//      dividing each theta_i by this factor stretches low-frequency dims
//      (large i) without extrapolating outside the trained range.
//   2. Attention temperature sqrt(1/t) where t = 1 - ramp_factor.
//
// At scale = 1 + alpha = 0, YaRN is bit-exact to vanilla RoPE.
//
// Tests (10):
//   1.  test_constructor_validates_dims
//   2.  test_per_dim_freq_scale_monotone
//   3.  test_vanilla_rope_equivalence_at_scale_1
//   4.  test_alpha_0_collapses_to_vanilla_rope
//   5.  test_scale_gt_1_diverges_from_vanilla
//   6.  test_forward_shape_and_finite
//   7.  test_input_gradient_fd
//   8.  test_attention_temperature
//   9.  test_mutation_alpha_changes_output
//   10. test_end_to_end_position_task

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <vector>
#include <algorithm>
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

static Tensor rand_tensor(size_t rows, size_t cols, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) x.data[i] = nd(rng);
    return x;
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double d = std::abs(a.data[i] - b.data[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static inline double block_mse(const Tensor& y, const Tensor& t) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y[i][j] - t[i][j];
            L += d * d;
        }
    return L / (2.0 * y.rows);
}

// ============================================================================
// Test 1: constructor validation
// ============================================================================
static void test_constructor_validates_dims() {
    cout << "--- Test 1: YaRNRoPE constructor validates dims ---" << endl;
    bool ok = true;

    // dim must be even
    try {
        YaRNRoPE bad(7, 64, 10000.0f, 1.0f, 0.1f, 0.0f);
        ok = check("dim=7 throws std::invalid_argument", false);
    } catch (const std::invalid_argument&) {
        ok &= check("dim=7 throws std::invalid_argument", true);
    }

    // max_seq_len must be > 0
    try {
        YaRNRoPE bad(8, 0, 10000.0f, 1.0f, 0.1f, 0.0f);
        ok = check("max_seq_len=0 throws", false);
    } catch (const std::invalid_argument&) {
        ok &= check("max_seq_len=0 throws", true);
    }

    // scale must be >= 1.0
    try {
        YaRNRoPE bad(8, 64, 10000.0f, 0.5f, 0.1f, 0.0f);
        ok = check("scale=0.5 throws", false);
    } catch (const std::invalid_argument&) {
        ok &= check("scale=0.5 throws", true);
    }

    // alpha must be >= 0
    try {
        YaRNRoPE bad(8, 64, 10000.0f, 1.0f, -0.1f, 0.0f);
        ok = check("alpha=-0.1 throws", false);
    } catch (const std::invalid_argument&) {
        ok &= check("alpha=-0.1 throws", true);
    }

    // ramp_factor must be in [0, 1]
    try {
        YaRNRoPE bad(8, 64, 10000.0f, 1.0f, 0.1f, 1.5f);
        ok = check("ramp_factor=1.5 throws", false);
    } catch (const std::invalid_argument&) {
        ok &= check("ramp_factor=1.5 throws", true);
    }
    try {
        YaRNRoPE bad(8, 64, 10000.0f, 1.0f, 0.1f, -0.1f);
        ok = check("ramp_factor=-0.1 throws", false);
    } catch (const std::invalid_argument&) {
        ok &= check("ramp_factor=-0.1 throws", true);
    }

    // valid constructor works
    try {
        YaRNRoPE good(8, 64, 10000.0f, 1.0f, 0.1f, 0.0f);
        ok &= check("valid constructor works", true);
    } catch (...) {
        ok &= check("valid constructor works", false);
    }

    // accessors
    YaRNRoPE yarn(8, 64, 10000.0f, 4.0f, 0.1f, 0.25f);
    ok &= check("dim() == 8", yarn.dim() == 8);
    ok &= check("max_seq_len() == 64", yarn.max_seq_len() == 64);
    ok &= check("base() == 10000.0f", yarn.base() == 10000.0f);
    ok &= check("scale() == 4.0f", yarn.scale() == 4.0f);
    ok &= check("alpha() == 0.1f", std::abs(yarn.alpha() - 0.1f) < 1e-9f);
    ok &= check("ramp_factor() == 0.25f", std::abs(yarn.ramp_factor() - 0.25f) < 1e-9f);
    ok &= check("name() == \"YaRNRoPE\"", yarn.name() == "YaRNRoPE");

    // no learnable parameters
    ok &= check("parameters() empty", yarn.parameters().empty());
}

// ============================================================================
// Test 2: per-dim freq_scale table monotonicity
// ============================================================================
static void test_per_dim_freq_scale_monotone() {
    cout << "--- Test 2: per-dim freq_scale_by_dim is monotone ---" << endl;
    bool ok = true;

    // dim=8, half_dim=4. With alpha=0.1, expected freq_scale_by_dim[i]:
    //   i=0: ramp=0/3=0,     factor = 1/(0.1·0 + 1) = 1.0000  (no scaling)
    //   i=1: ramp=1/3=0.333, factor = 1/(0.1·0.333 + 1) = 0.9677
    //   i=2: ramp=2/3=0.667, factor = 1/(0.1·0.667 + 1) = 0.9375
    //   i=3: ramp=3/3=1.000, factor = 1/(0.1·1.0 + 1)   = 0.9091  (max scaling)
    //
    // Smaller factor at larger i means we DIVIDE theta_i by a smaller
    // number at low-frequency dims, giving LARGER angles per position
    // — exactly what long-context extension needs (rotate faster,
    // hit more unique phases within the same number of positions).
    //
    // The freq_scale_by_dim table itself decreases monotonically from
    // 1.0 (at i=0) to 1/(1+alpha) (at i=half_dim-1).

    YaRNRoPE yarn(8, 64, 10000.0f, 1.0f, 0.1f, 0.0f);
    yarn.precompute_theta_freqs(8);

    const Tensor& tbl = yarn.freq_scale_table();
    ok &= check("freq_scale_table shape (4, 1) for dim=8",
                tbl.rows == 4 && tbl.cols == 1);

    double prev = tbl(0, 0);
    ok &= check("freq_scale_by_dim[0] == 1.0 (high-freq untouched)",
                std::abs(tbl(0, 0) - 1.0) < 1e-12);

    // Bounds: freq_scale_by_dim[i] ∈ [1/(1+alpha), 1] for i ∈ [0, half_dim-1].
    //   * i=0:               freq_scale = 1.0   (no scaling, high-freq dim)
    //   * i ∈ (0, half_dim-1): freq_scale ∈ (1/(1+alpha), 1)
    //   * i = half_dim-1:    freq_scale = 1/(1+alpha)  (max scaling, low-freq dim)
    //
    // Earlier entries are LARGER (less scaling) because the ramp
    // coefficient `i / (half_dim - 1)` is smaller.
    //
    // Note: alpha is stored as float (0.1f = 0.10000000149...), so the
    // exact bit-pattern is 1.0 / (alpha * ramp + 1.0) computed in double.
    double alpha_d = 0.1;  // user's intent for the test
    double lower_bound = 1.0 / (1.0 + alpha_d);  // 0.9090909090...
    for (size_t i = 1; i < 4; ++i) {
        double cur = tbl(i, 0);
        ok &= check("freq_scale_by_dim[" + to_string(i) +
                    "] < freq_scale_by_dim[" + to_string(i - 1) +
                    "] (monotone decreasing)",
                    cur < prev);
        if (i < 3) {
            // Strictly above the lower bound (only the last entry equals it)
            ok &= check("freq_scale_by_dim[" + to_string(i) +
                        "] > 1/(1+alpha) (strictly above lower bound)",
                        cur > lower_bound);
        }
        ok &= check("freq_scale_by_dim[" + to_string(i) +
                    "] < 1.0 (below the upper bound)",
                    cur < 1.0);
        prev = cur;
    }

    // Final value at i=half_dim-1 should be 1/(1+alpha) within float precision.
    // (float-stored alpha=0.1 introduces ~1e-9 error; we use 1e-8 tolerance.)
    ok &= check("freq_scale_by_dim[3] within 1e-8 of 1/(1+alpha)",
                std::abs(tbl(3, 0) - lower_bound) < 1e-8);
}

// ============================================================================
// Test 3: vanilla RoPE equivalence at scale=1
// ============================================================================
static void test_vanilla_rope_equivalence_at_scale_1() {
    cout << "--- Test 3: scale=1 + alpha=0 -> vanilla RoPE bit-exact ---" << endl;
    bool ok = true;

    // Construct both: vanilla RoPE and YaRN with scale=1, alpha=0
    RoPE vanilla(8, 64, 10000.0f);
    vanilla.precompute_theta_freqs(8);

    YaRNRoPE yarn(8, 64, 10000.0f, /*scale=*/1.0f, /*alpha=*/0.0f, /*ramp=*/0.0f);
    yarn.precompute_theta_freqs(8);

    // cos/sin caches should be identical
    double cache_diff = 0.0;
    for (size_t i = 0; i < vanilla.cos_cache.data.size(); ++i) {
        double d = std::abs(vanilla.cos_cache.data[i] - yarn.cos_cache.data[i]);
        if (d > cache_diff) cache_diff = d;
    }
    ok &= check("cos_cache identical between RoPE and YaRN(scale=1, alpha=0)",
                cache_diff == 0.0);

    double sin_cache_diff = 0.0;
    for (size_t i = 0; i < vanilla.sin_cache.data.size(); ++i) {
        double d = std::abs(vanilla.sin_cache.data[i] - yarn.sin_cache.data[i]);
        if (d > sin_cache_diff) sin_cache_diff = d;
    }
    ok &= check("sin_cache identical between RoPE and YaRN(scale=1, alpha=0)",
                sin_cache_diff == 0.0);

    // Forward should be bit-exact
    Tensor q = rand_tensor(2, 8 * 4, 1, 0.3);  // batch=2, seq=4, dim=8
    Tensor k = rand_tensor(2, 8 * 4, 2, 0.3);

    auto v_q = vanilla.forward(q, k);
    auto v_q_first = v_q.first;
    auto v_q_second = v_q.second;

    auto y_q = yarn.forward(q, k, k);
    auto y_q_first = std::get<0>(y_q);
    auto y_q_second = std::get<1>(y_q);

    ok &= check("q_rot bit-exact RoPE == YaRN(scale=1, alpha=0)",
                max_abs_diff(v_q_first, y_q_first) == 0.0);
    ok &= check("k_rot bit-exact RoPE == YaRN(scale=1, alpha=0)",
                max_abs_diff(v_q_second, y_q_second) == 0.0);
}

// ============================================================================
// Test 4: alpha=0 collapses to vanilla RoPE regardless of scale
// ============================================================================
static void test_alpha_0_collapses_to_vanilla_rope() {
    cout << "--- Test 4: alpha=0 collapses to vanilla RoPE even at scale>1 ---" << endl;
    bool ok = true;

    YaRNRoPE yarn(8, 64, 10000.0f, /*scale=*/16.0f, /*alpha=*/0.0f, /*ramp=*/0.0f);
    yarn.precompute_theta_freqs(8);

    // freq_scale_by_dim should all be 1.0
    const Tensor& tbl = yarn.freq_scale_table();
    bool all_ones = true;
    for (size_t i = 0; i < tbl.rows; ++i) {
        if (std::abs(tbl(i, 0) - 1.0) > 1e-12) {
            all_ones = false;
            break;
        }
    }
    ok &= check("alpha=0 -> freq_scale_by_dim all = 1.0", all_ones);

    // Compare forward output with vanilla RoPE (bit-exact, ignoring the
    // scale parameter which only affects the cache via alpha, which is 0)
    RoPE vanilla(8, 64, 10000.0f);
    vanilla.precompute_theta_freqs(8);

    Tensor q = rand_tensor(2, 8 * 4, 3, 0.3);
    Tensor k = rand_tensor(2, 8 * 4, 4, 0.3);

    auto v_q = vanilla.forward(q, k);
    auto y_q = yarn.forward(q, k, k);

    ok &= check("alpha=0,scale=16 -> bit-exact vanilla RoPE (q)",
                max_abs_diff(v_q.first, std::get<0>(y_q)) == 0.0);
    ok &= check("alpha=0,scale=16 -> bit-exact vanilla RoPE (k)",
                max_abs_diff(v_q.second, std::get<1>(y_q)) == 0.0);
}

// ============================================================================
// Test 5: scale > 1 + alpha > 0 diverges from vanilla
// ============================================================================
static void test_scale_gt_1_diverges_from_vanilla() {
    cout << "--- Test 5: scale=8,alpha=0.1 diverges from vanilla RoPE ---" << endl;
    bool ok = true;

    RoPE vanilla(8, 64, 10000.0f);
    vanilla.precompute_theta_freqs(8);

    YaRNRoPE yarn(8, 64, 10000.0f, /*scale=*/8.0f, /*alpha=*/0.1f, /*ramp=*/0.0f);
    yarn.precompute_theta_freqs(8);

    // cos/sin caches should differ
    double cache_diff = 0.0;
    for (size_t i = 0; i < vanilla.cos_cache.data.size(); ++i) {
        double d = std::abs(vanilla.cos_cache.data[i] - yarn.cos_cache.data[i]);
        if (d > cache_diff) cache_diff = d;
    }
    ok &= check("cos_cache differs between RoPE and YaRN(scale=8,alpha=0.1)",
                cache_diff > 1e-3);

    // Forward should also differ (not just caches)
    Tensor q = rand_tensor(2, 8 * 4, 5, 0.3);
    Tensor k = rand_tensor(2, 8 * 4, 6, 0.3);

    auto v_q = vanilla.forward(q, k);
    auto y_q = yarn.forward(q, k, k);

    ok &= check("q_rot differs (not bit-exact)",
                max_abs_diff(v_q.first, std::get<0>(y_q)) > 1e-3);

    // Output should still be finite
    Tensor& out = std::get<0>(y_q);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    }
    ok &= check("output is finite", finite);
}

// ============================================================================
// Test 6: forward shape and finiteness
// ============================================================================
static void test_forward_shape_and_finite() {
    cout << "--- Test 6: forward shape (2, 16) -> q,k,v (2, 16) ---" <<endl;
    bool ok = true;

    YaRNRoPE yarn(8, 64, 10000.0f, 4.0f, 0.1f, 0.0f);
    yarn.precompute_theta_freqs(4);  // seq=4, dim=8 -> cols=32

    Tensor q = rand_tensor(2, 32, 7, 0.3);
    Tensor k = rand_tensor(2, 32, 8, 0.3);
    Tensor v = rand_tensor(2, 32, 9, 0.3);

    auto result = yarn.forward(q, k, v);
    Tensor& q_out = std::get<0>(result);
    Tensor& k_out = std::get<1>(result);
    Tensor& v_out = std::get<2>(result);

    ok &= check("q_out shape (2, 32)", q_out.rows == 2 && q_out.cols == 32);
    ok &= check("k_out shape (2, 32)", k_out.rows == 2 && k_out.cols == 32);
    ok &= check("v_out shape (2, 32)", v_out.rows == 2 && v_out.cols == 32);

    auto all_finite = [](const Tensor& t) {
        for (size_t i = 0; i < t.data.size(); ++i) {
            if (!std::isfinite(t.data[i])) return false;
        }
        return true;
    };
    ok &= check("q_out all finite", all_finite(q_out));
    ok &= check("k_out all finite", all_finite(k_out));
    ok &= check("v_out all finite", all_finite(v_out));
}

// ============================================================================
// Test 7: input gradient FD vs analytical
// ============================================================================
static void test_input_gradient_fd() {
    cout << "--- Test 7: input gradient FD vs analytical ---" << endl;
    bool ok = true;

    YaRNRoPE yarn(8, 64, 10000.0f, 1.0f, 0.0f, 0.0f);  // scale=1, alpha=0 -> vanilla RoPE
    yarn.precompute_theta_freqs(4);

    Tensor q = rand_tensor(2, 32, 11, 0.3);
    Tensor k = rand_tensor(2, 32, 12, 0.3);
    Tensor v = rand_tensor(2, 32, 13, 0.3);

    Tensor target_q = rand_tensor(2, 32, 14, 0.3);
    Tensor target_k = rand_tensor(2, 32, 15, 0.3);
    Tensor target_v = rand_tensor(2, 32, 16, 0.3);

    auto result = yarn.forward(q, k, v);
    Tensor q_out = std::get<0>(result);
    Tensor k_out = std::get<1>(result);
    Tensor v_out = std::get<2>(result);

    // Build grad_q, grad_k, grad_v from MSE loss vs targets
    Tensor grad_q = q_out; grad_q.fill(0.0);
    Tensor grad_k = k_out; grad_k.fill(0.0);
    Tensor grad_v = v_out; grad_v.fill(0.0);
    for (size_t i = 0; i < q_out.rows; ++i)
        for (size_t j = 0; j < q_out.cols; ++j) {
            grad_q[i][j] = (q_out[i][j] - target_q[i][j]) / q_out.rows;
            grad_k[i][j] = (k_out[i][j] - target_k[i][j]) / q_out.rows;
            grad_v[i][j] = (v_out[i][j] - target_v[i][j]) / v_out.rows;
        }

    Tensor dq_analytical = yarn.backward_qkv(grad_q, grad_k, grad_v);

    // Numerical FD for q at a few positions
    double eps = 1e-5;
    bool all_pass = true;
    size_t test_count = 0;
    std::mt19937 rng(99);
    std::uniform_int_distribution<size_t> row_dist(0, q.rows - 1);
    std::uniform_int_distribution<size_t> col_dist(0, q.cols - 1);
    for (size_t trial = 0; trial < 10; ++trial) {
        size_t r = row_dist(rng);
        size_t c = col_dist(rng);
        double orig = q(r, c);

        q(r, c) = orig + eps;
        auto r_plus = yarn.forward(q, k, v);
        double L_plus = block_mse(std::get<0>(r_plus), target_q)
                      + block_mse(std::get<1>(r_plus), target_k)
                      + block_mse(std::get<2>(r_plus), target_v);

        q(r, c) = orig - eps;
        auto r_minus = yarn.forward(q, k, v);
        double L_minus = block_mse(std::get<0>(r_minus), target_q)
                       + block_mse(std::get<1>(r_minus), target_k)
                       + block_mse(std::get<2>(r_minus), target_v);

        q(r, c) = orig;
        double fd = (L_plus - L_minus) / (2.0 * eps);
        double ana = dq_analytical(r, c);
        double rel_err = std::abs(ana - fd) / std::max(std::abs(ana), std::abs(fd));
        if (rel_err > 1e-4) {
            cout << "    FD trial " << trial << " at (" << r << "," << c
                 << "): ana=" << ana << " fd=" << fd
                 << " rel_err=" << rel_err << endl;
            all_pass = false;
        }
        ++test_count;
    }
    ok &= check("input gradient FD vs analytical (10 trials, rel_err < 1e-4)",
                all_pass && test_count == 10);

    // Also check that the K and V gradients are populated and non-zero
    bool k_grad_nonzero = false, v_grad_nonzero = false;
    const Tensor& dK = yarn.backward_k();
    const Tensor& dV = yarn.backward_v();
    for (size_t i = 0; i < dK.data.size(); ++i) {
        if (std::abs(dK.data[i]) > 1e-9) { k_grad_nonzero = true; break; }
    }
    for (size_t i = 0; i < dV.data.size(); ++i) {
        if (std::abs(dV.data[i]) > 1e-9) { v_grad_nonzero = true; break; }
    }
    ok &= check("backward_k() populated and nonzero", k_grad_nonzero);
    ok &= check("backward_v() populated and nonzero", v_grad_nonzero);
}

// ============================================================================
// Test 8: attention temperature formula
// ============================================================================
static void test_attention_temperature() {
    cout << "--- Test 8: attention temperature formula ---" << endl;
    bool ok = true;

    YaRNRoPE cold(8, 64, 10000.0f, 1.0f, 0.1f, /*ramp=*/0.0f);
    ok &= check("ramp=0 -> temperature = 1.0 (no scaling)",
                std::abs(cold.attention_temperature() - 1.0) < 1e-12);

    YaRNRoPE warm(8, 64, 10000.0f, 1.0f, 0.1f, /*ramp=*/0.5f);
    // t = 1 - 0.5 = 0.5; sqrt(1/0.5) = sqrt(2) ≈ 1.4142
    ok &= check("ramp=0.5 -> temperature = sqrt(2)",
                std::abs(warm.attention_temperature() - std::sqrt(2.0)) < 1e-9);

    YaRNRoPE hottest(8, 64, 10000.0f, 1.0f, 0.1f, /*ramp=*/0.99f);
    // t = 1 - 0.99 = 0.01; sqrt(1/0.01) = sqrt(100) = 10.0
    // Note: 0.99f is stored as 0.9900000095..., so t = 0.00999999... and
    // sqrt(1/t) ≈ 10.0000047. Use 1e-4 tolerance.
    ok &= check("ramp=0.99 -> temperature ≈ 10 (within 1e-4)",
                std::abs(hottest.attention_temperature() - 10.0) < 1e-4);

    YaRNRoPE clamped(8, 64, 10000.0f, 1.0f, 0.1f, /*ramp=*/1.0f);
    // t = 0 -> clamped at 1e-12; sqrt(1/1e-12) = 1e6
    ok &= check("ramp=1.0 -> temperature clamped at 1e6",
                std::abs(clamped.attention_temperature() - 1e6) < 1e-3);

    // temperature_for_step: with ramp=0.5, step=0 -> sqrt(1/(1-0.5)) = sqrt(2)
    // with step=total -> sqrt(1/(1-0.5*(1-1))) = sqrt(1/(1-0)) = 1
    ok &= check("warm t_for_step(0, 100) = sqrt(2)",
                std::abs(warm.temperature_for_step(0, 100) - std::sqrt(2.0)) < 1e-9);
    ok &= check("warm t_for_step(100, 100) = 1",
                std::abs(warm.temperature_for_step(100, 100) - 1.0) < 1e-12);
    ok &= check("warm t_for_step(50, 100) = sqrt(1/(1-0.25)) = sqrt(4/3)",
                std::abs(warm.temperature_for_step(50, 100) - std::sqrt(4.0/3.0)) < 1e-9);
}

// ============================================================================
// Test 9: mutation — alpha=0.1 -> alpha=0.5 changes the output
// ============================================================================
static void test_mutation_alpha_changes_output() {
    cout << "--- Test 9: mutation — varying alpha changes output ---" << endl;
    bool ok = true;

    Tensor q = rand_tensor(2, 32, 17, 0.3);
    Tensor k = rand_tensor(2, 32, 18, 0.3);
    Tensor v = rand_tensor(2, 32, 19, 0.3);

    YaRNRoPE low_alpha(8, 64, 10000.0f, 8.0f, /*alpha=*/0.1f, 0.0f);
    low_alpha.precompute_theta_freqs(4);
    auto out_low = low_alpha.forward(q, k, v);

    YaRNRoPE high_alpha(8, 64, 10000.0f, 8.0f, /*alpha=*/0.5f, 0.0f);
    high_alpha.precompute_theta_freqs(4);
    auto out_high = high_alpha.forward(q, k, v);

    double diff = max_abs_diff(std::get<0>(out_low), std::get<0>(out_high));
    ok &= check("alpha=0.1 vs alpha=0.5 -> measurably different forward (q)",
                diff > 1e-3);

    double diff_k = max_abs_diff(std::get<1>(out_low), std::get<1>(out_high));
    ok &= check("alpha=0.1 vs alpha=0.5 -> measurably different forward (k)",
                diff_k > 1e-3);
}

// ============================================================================
// Test 10: end-to-end YaRN on a synthetic position-encoding task
//
// Construct Q/K such that the Q·K^T score depends on relative position.
// We pick a target rotation pattern: a fixed random K matrix, and Q
// matrices at each position that, after YaRN rotation, should match the
// rotated K. We minimize MSE loss over 50 SGD steps by adjusting only
// the rotation cache — but since the rotation is fixed (no learnable
// params), this test instead verifies that YaRN's rotation can be
// INVERTED analytically: applying the transpose-rotation to the output
// should recover the input.
//
// This is the rotation-equivariance property: R^T @ R @ x = x.
// ============================================================================
static void test_end_to_end_position_task() {
    cout << "--- Test 10: end-to-end YaRN rotation equivariant (R^T R x = x) ---" << endl;
    bool ok = true;

    YaRNRoPE yarn(8, 64, 10000.0f, 4.0f, 0.1f, 0.0f);
    yarn.precompute_theta_freqs(8);

    Tensor q = rand_tensor(2, 8 * 4, 21, 0.5);  // larger scale to make sure
    Tensor k = rand_tensor(2, 8 * 4, 22, 0.5);  // the rotation is nontrivial
    Tensor v = rand_tensor(2, 8 * 4, 23, 0.5);

    // Forward
    auto fwd = yarn.forward(q, k, v);

    // Build unit gradients (the rotation is a bijection, so any gradient
    // should recover the input exactly via R^T @ grad).
    Tensor grad_q(q.rows, q.cols);
    Tensor grad_k(k.rows, k.cols);
    Tensor grad_v(v.rows, v.cols);
    grad_q.fill(1.0);
    grad_k.fill(0.5);
    grad_v.fill(0.25);

    Tensor dq = yarn.backward_qkv(grad_q, grad_k, grad_v);

    // dq should be R^T @ grad_q. Since R^T @ R = I, applying forward
    // followed by backward (with all-one grads) should recover the input
    // exactly, MINUS the contribution from grad_k and grad_v which pass
    // through different cos/sin caches... actually all three use the
    // same cache, but we computed dq = R^T @ grad_q only. So dq != q.
    //
    // Instead: verify R^T @ R @ q == q by checking that
    // backward_qkv(forward(q, k, v) gradients = forward output, 0, 0)
    // applied to those gradients yields q.
    //
    // Cleaner: take q_out, treat it as grad_q, set grad_k = grad_v = 0,
    // run backward_qkv, and check that the result == q.
    Tensor zero_k(k.rows, k.cols); zero_k.fill(0.0);
    Tensor zero_v(v.rows, v.cols); zero_v.fill(0.0);

    Tensor q_recovered = yarn.backward_qkv(std::get<0>(fwd), zero_k, zero_v);

    double max_diff = max_abs_diff(q_recovered, q);
    ok &= check("R^T @ R @ q == q (rotation equivariant, max_diff < 1e-12)",
                max_diff < 1e-12);

    // Similarly for k and v (but those are populated separately via
    // backward_k() and backward_v(), not via backward_qkv's return).
    // We can check by passing only k/v gradients through backward_qkv
    // and reading the corresponding tensors.
    Tensor zero_q(q.rows, q.cols); zero_q.fill(0.0);
    yarn.backward_qkv(zero_q, std::get<1>(fwd), zero_v);
    double max_diff_k = max_abs_diff(yarn.backward_k(), k);
    ok &= check("R^T @ R @ k == k (via backward_k, max_diff < 1e-12)",
                max_diff_k < 1e-12);

    yarn.backward_qkv(zero_q, zero_k, std::get<2>(fwd));
    double max_diff_v = max_abs_diff(yarn.backward_v(), v);
    ok &= check("R^T @ R @ v == v (via backward_v, max_diff < 1e-12)",
                max_diff_v < 1e-12);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    cout << "=== YaRN (Yet another RoPE extensioN) Tests ===" << endl;
    test_constructor_validates_dims();
    test_per_dim_freq_scale_monotone();
    test_vanilla_rope_equivalence_at_scale_1();
    test_alpha_0_collapses_to_vanilla_rope();
    test_scale_gt_1_diverges_from_vanilla();
    test_forward_shape_and_finite();
    test_input_gradient_fd();
    test_attention_temperature();
    test_mutation_alpha_changes_output();
    test_end_to_end_position_task();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ==="
         << endl;
    return failed == 0 ? 0 : 1;
}
