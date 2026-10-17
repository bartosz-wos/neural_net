// Tests for post-training integer quantization (arXiv:1712.05877, Jacob et al.).
//
// Every expected value here is TRANSCRIBED FROM tests/ptq_oracle.py output, not
// from memory. The oracle self-checks against finite differences before
// emitting anything, and its self-checks themselves corrected four of my
// assumptions -- see the notes at each test.
//
// THE MEASURED FINDING THAT SHAPES THIS FILE: the quantize->dequantize forward
// is PIECEWISE CONSTANT in w, so a centred finite difference is EXACTLY ZERO on
// every non-maximal entry for a correct straight-through estimator AND for an
// arbitrarily wrong one alike. FD therefore CANNOT verify this code. Everything
// below is a VALUE assertion, plus mutation testing on the backward.

#include "nn/utils/ptq.h"
#include "nn/activations/activations.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>



static int g_pass = 0, g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s\n", name.c_str());
    }
}

static void check_close(double got, double want, double tol,
                        const std::string& name) {
    bool ok = std::fabs(got - want) <= tol;
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s: got %.17g want %.17g (tol %.3g)\n",
                    name.c_str(), got, want, tol);
    }
}

static Tensor make_w() {
    // The oracle's fixture, 3x4. max|W| = 3.0 (at [1][0] = -3.0), so the
    // per-tensor scale is 3/127 = 0.023622047244094488.
    Tensor w(3, 4);
    double d[3][4] = {
        {0.5, -1.25, 2.0, 0.125},
        {-3.0, 0.75, 0.0, 0.5},
        {0.1, 0.2, -0.4, 0.8},
    };
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) w[r][c] = d[r][c];
    return w;
}

static Tensor make_half_grid() {
    // Built to land EXACTLY on half-grid points. max = 254 -> scale = 2.0
    // exactly, and 127/2 = 63.5 steps.
    Tensor w(1, 6);
    double d[6] = {127.0, 253.0, 254.0, -127.0, -253.0, 1.0};
    for (size_t c = 0; c < 6; ++c) w[0][c] = d[c];
    return w;
}

// ---------------------------------------------------------------------------
// Test 1: the per-tensor scale is max|w| / 127, exactly.
// ---------------------------------------------------------------------------
static void test_per_tensor_scale() {
    Tensor w = make_w();
    double s = ptq_scale_per_tensor(w);
    check_close(s, 3.0 / 127.0, 0.0, "per_tensor scale = max|w|/127");
    check(s == 3.0 / 127.0, "per_tensor scale is exact, not approximate");

    // A tensor whose max is NOT in row 0, and which is NOT symmetric -- every
    // "natural" hand-written weight matrix tends to be both.
    Tensor t(2, 3);
    t[0][0] = -0.5; t[0][1] = 0.25; t[0][2] = 0.125;
    t[1][0] = 0.75; t[1][1] = -1.5; t[1][2] = 0.5;
    check_close(ptq_scale_per_tensor(t), 1.5 / 127.0, 0.0,
                "per_tensor scale takes the max over NEGATIVE entries too");
}

// ---------------------------------------------------------------------------
// Test 2: the integer codes, and the saturation property.
// ---------------------------------------------------------------------------
static void test_codes_and_saturation() {
    Tensor w = make_w();
    Tensor q = ptq_quantize_per_tensor(w);
    // Oracle: {21, -53, 85, 5}, {-127, 32, 0, 21}, {4, 8, -17, 34}
    int want[3][4] = {{21, -53, 85, 5}, {-127, 32, 0, 21}, {4, 8, -17, 34}};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            check(static_cast<int>(q[r][c]) == want[r][c],
                  "per_tensor codes match the oracle at [" +
                      std::to_string(r) + "][" + std::to_string(c) + "]");

    // D-saturation: the max-magnitude entry maps to EXACTLY 127 (and -127 for
    // the most negative). An implementation that uses 127.5 as the bound, or
    // that clamps at 126, passes every error test and fails this.
    check(static_cast<int>(q[1][0]) == -127, "most negative entry -> -127");

    // Every code is within the int8 range -- checked by explicit scan because
    // Tensor::operator() has NO bounds check and a bug here would be silent.
    bool in_range = true;
    for (size_t r = 0; r < q.rows; ++r)
        for (size_t c = 0; c < q.cols; ++c)
            if (q[r][c] < -127.0 || q[r][c] > 127.0) in_range = false;
    check(in_range, "every code lies in [-127, 127]");
}

// ---------------------------------------------------------------------------
// Test 3: DEQUANTIZATION REPRODUCES THE ORACLE'S w_hat -- the real gradient
// check for a piecewise-constant forward.
// ---------------------------------------------------------------------------
static void test_dequantize_matches_oracle() {
    Tensor w = make_w();
    double s = ptq_scale_per_tensor(w);
    Tensor wh = ptq_dequantize(ptq_quantize_per_tensor(w), s);
    // Oracle per-tensor w_hat
    double want[3][4] = {
        {0.49606299212598426, -1.2519685039370079, 2.0078740157480315,
         0.11811023622047244},
        {-3, 0.75590551181102361, 0, 0.49606299212598426},
        {0.094488188976377951, 0.1889763779527559, -0.40157480314960631,
         0.80314960629921262}};
    double worst = 0.0;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            worst = std::fmax(worst, std::fabs(wh[r][c] - want[r][c]));
    check_close(worst, 0.0, 1e-15, "dequantized values match the oracle");
}

// ---------------------------------------------------------------------------
// Test 4: D-ROUNDING -- half-away-from-zero, pinned on exact half-grid points.
//   numpy rint (half-to-EVEN) and C std::round disagree on 3 of these 6.
//   No tolerance test can see this; the fixture is built to hit it.
// ---------------------------------------------------------------------------
static void test_rounding_is_half_away_from_zero() {
    Tensor w = make_half_grid();
    double s = ptq_scale_per_tensor(w);
    check(s == 2.0, "half-grid fixture has scale exactly 2.0");
    Tensor q = ptq_quantize_per_tensor(w);
    // Oracle: std::round (half-AWAY) -> [64, 127, 127, -64, -127, 1]
    // numpy rint (half-to-EVEN) would give [64, 126, 127, -64, -126, 0]
    int want[6] = {64, 127, 127, -64, -127, 1};
    for (size_t c = 0; c < 6; ++c) {
        check(static_cast<int>(q[0][c]) == want[c],
              "rounding is half-away-from-zero at entry " + std::to_string(c));
    }
    // And the CONVERSE, so the assertion cannot be satisfied by accident:
    // an implementation using rint's rule must differ here.
    int rint_would[6] = {64, 126, 127, -64, -126, 0};
    bool differs_somewhere = false;
    for (size_t c = 0; c < 6; ++c)
        if (rint_would[c] != want[c]) differs_somewhere = true;
    check(differs_somewhere,
          "the fixture actually distinguishes half-away from half-to-even");
}

// ---------------------------------------------------------------------------
// Test 5: D-ZERO -- an all-zero tensor returns zeros, NOT NaN.
//   `scale = max|w|/127` is 0 here; dequantizing by a zero scale is the trap
//   the VQ-VAE session hit once (the reference's decay=1 returned all-NaN).
// ---------------------------------------------------------------------------
static void test_zero_tensor_is_not_nan() {
    Tensor z(2, 3);   // Tensor(n, m) is zero-filled by construction
    double s = ptq_scale_per_tensor(z);
    check(s == 0.0, "zero tensor has scale 0");
    Tensor q = ptq_quantize_per_tensor(z);
    Tensor wh = ptq_dequantize(q, s);
    bool all_zero = true, any_nan = false;
    for (size_t r = 0; r < q.rows; ++r)
        for (size_t c = 0; c < q.cols; ++c) {
            if (q[r][c] != 0.0) all_zero = false;
            if (wh[r][c] != 0.0) all_zero = false;
            if (std::isnan(wh[r][c])) any_nan = true;
        }
    check(all_zero, "zero tensor quantizes and dequantizes to exact zeros");
    check(!any_nan, "zero tensor produces no NaN");
}

// ---------------------------------------------------------------------------
// Test 6: D-SATURATION-PER-CHANNEL -- per-channel scales are per COLUMN, with
//   one scale per column of the (rows, cols) tensor.
// ---------------------------------------------------------------------------
static void test_per_channel_scales() {
    Tensor w = make_w();
    Tensor s = ptq_scale_per_channel(w);
    check(s.rows == 1 && s.cols == 4,
          "per_channel scales have shape (1, cols) -- one scale per column");
    // Oracle reduces with axis=0, so column maxima of W are:
    //   col0: max(|0.5|,|-3|,|0.1|) = 3
    //   col1: max(|-1.25|,|0.75|,|0.2|) = 1.25
    //   col2: max(|2|,|0|,|-0.4|) = 2
    //   col3: max(|0.125|,|0.5|,|0.8|) = 0.8
    double want[4] = {3.0 / 127.0, 1.25 / 127.0, 2.0 / 127.0, 0.8 / 127.0};
    for (size_t c = 0; c < 4; ++c)
        check_close(s[0][c], want[c], 1e-18,
                    "per_channel scale for column " + std::to_string(c));
}

static void test_per_channel_codes() {
    Tensor w = make_w();
    Tensor q = ptq_quantize_per_channel(w);
    // Oracle per-channel q:
    //   {21, -127, 127, 20}, {-127, 76, 0, 79}, {4, 20, -25, 127}
    int want[3][4] = {{21, -127, 127, 20}, {-127, 76, 0, 79}, {4, 20, -25, 127}};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            check(static_cast<int>(q[r][c]) == want[r][c],
                  "per_channel codes match the oracle at [" +
                      std::to_string(r) + "][" + std::to_string(c) + "]");
}

static void test_per_channel_w_hat_matches_oracle() {
    Tensor w = make_w();
    Tensor s = ptq_scale_per_channel(w);
    Tensor wh = ptq_dequantize_per_channel(ptq_quantize_per_channel(w), s);
    double want[3][4] = {
        {0.49606299212598426, -1.25, 2, 0.12598425196850396},
        {-3, 0.74803149606299213, 0, 0.4976377952755906},
        {0.094488188976377951, 0.19685039370078738, -0.39370078740157477,
         0.80000000000000004}};
    double worst = 0.0;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            worst = std::fmax(worst, std::fabs(wh[r][c] - want[r][c]));
    check_close(worst, 0.0, 1e-15, "per-channel w_hat matches the oracle");
}

// ---------------------------------------------------------------------------
// Test 7: per-channel DOMINATES per-tensor pointwise -- and this is a THEOREM,
//   not an observation. Per-channel error on column c is bounded by
//   max|w_c|/(2*127) and per-tensor by max|w_all|/(2*127), and max|w_c| <=
//   max|w_all|. A 3x3 decade sweep found no counterexample (see the oracle).
//   The suite asserts the theorem's CONSEQUENCE, not a hand-picked comparison.
// ---------------------------------------------------------------------------
static void test_per_channel_dominates() {
    Tensor w = make_w();
    Tensor qt = ptq_quantize_per_tensor(w);
    Tensor qc = ptq_quantize_per_channel(w);
    Tensor wt = ptq_dequantize(qt, ptq_scale_per_tensor(w));
    Tensor wc = ptq_dequantize_per_channel(qc, ptq_scale_per_channel(w));
    double l2t = ptq_relative_l2(w, wt);
    double l2c = ptq_relative_l2(w, wc);
    check(l2c < l2t, "per-channel has strictly smaller relative L2 here");
    check_close(l2t, 0.0045641204768499472, 1e-12,
                "per-tensor relative L2 matches the oracle");
    check_close(l2c, 0.0025341805049432847, 1e-12,
                "per-channel relative L2 matches the oracle");

    // The bound is CLOSED FORM and therefore testable without an oracle:
    // for a tensor normalised to max|w| == 1, the error is at most half a
    // grid step = 1/(2*127). Oracle measured 0.0039368721991914768 against
    // the bound 0.003937007874015748.
    Tensor u(3, 4);
    double d[3][4] = {{0.1, -0.2, 1.0, 0.3}, {-0.7, 0.4, 0.05, -0.9},
                      {0.6, -1.0, 0.2, 0.8}};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) u[r][c] = d[r][c];
    Tensor uh = ptq_dequantize(ptq_quantize_per_tensor(u),
                               ptq_scale_per_tensor(u));
    double worst = 0.0;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            worst = std::fmax(worst, std::fabs(u[r][c] - uh[r][c]));
    check(worst <= 1.0 / (2.0 * 127.0) + 1e-12,
          "error respects the closed-form half-grid-step bound");
}

// ---------------------------------------------------------------------------
// Test 8: D-FRESHNESS -- the returned gradient/code tensors never alias the
//   input, and quantize->dequantize is IDEMPOTENT (oracle: exactly 0.0).
// ---------------------------------------------------------------------------
static void test_idempotence_and_no_aliasing() {
    Tensor w = make_w();
    Tensor wh1 = ptq_dequantize(ptq_quantize_per_tensor(w),
                                ptq_scale_per_tensor(w));
    Tensor wh2 = ptq_dequantize(ptq_quantize_per_tensor(wh1),
                                ptq_scale_per_tensor(wh1));
    double worst = 0.0;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            worst = std::fmax(worst, std::fabs(wh1[r][c] - wh2[r][c]));
    check(worst == 0.0, "quantize->dequantize is exactly idempotent");

    // Mutating the input after quantizing must not change the result.
    Tensor w2 = make_w();
    Tensor q = ptq_quantize_per_tensor(w2);
    double before = q[0][0];
    w2[0][0] = 999.0;
    check(q[0][0] == before, "the returned code tensor does not alias its input");

    // A non-aliasing guard on the dequantized output too.
    Tensor w3 = make_w();
    double s3 = ptq_scale_per_tensor(w3);
    Tensor q3 = ptq_quantize_per_tensor(w3);
    Tensor out = ptq_dequantize(q3, s3);
    q3[0][0] = -42.0;
    check(out[0][0] != -42.0 * s3,
          "the dequantized tensor does not alias the codes");
}

// ---------------------------------------------------------------------------
// Test 9: D-EMPTY -- empty tensors throw rather than returning a silent NaN.
// ---------------------------------------------------------------------------
static void test_empty_throws() {
    bool threw = false;
    try {
        Tensor e(0, 0);
        (void)ptq_scale_per_tensor(e);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "an empty tensor throws in ptq_scale_per_tensor");
}

// ---------------------------------------------------------------------------
// Test 10: D-CODES-ARE-INTEGERS -- the code tensor holds exact integers.
//   A float "code" tensor with 21.0000000001 in it would still dequantize
//   correctly to within tolerance and would still be stored in int8 wrong.
// ---------------------------------------------------------------------------
static void test_codes_are_exact_integers() {
    Tensor w = make_w();
    Tensor q = ptq_quantize_per_tensor(w);
    bool integral = true;
    for (size_t r = 0; r < q.rows; ++r)
        for (size_t c = 0; c < q.cols; ++c)
            if (q[r][c] != std::floor(q[r][c] + 0.5)) integral = false;
    check(integral, "every code is an exact integer value");

    Tensor qc = ptq_quantize_per_channel(w);
    bool integral_c = true;
    for (size_t r = 0; r < qc.rows; ++r)
        for (size_t c = 0; c < qc.cols; ++c)
            if (qc[r][c] != std::floor(qc[r][c] + 0.5)) integral_c = false;
    check(integral_c, "every per-channel code is an exact integer value");
}

// ---------------------------------------------------------------------------
// Test 11: D-SHAPE -- quantizing a tensor of shape (1, N) and (N, 1).
//   Rank-1 weights are LEGAL and are exactly the shapes that expose a
//   per-tensor/per-channel confusion, because for a single row the column
//   reduction and the global reduction coincide.
// ---------------------------------------------------------------------------
static void test_rank_one_shapes() {
    Tensor row(1, 4);
    row[0][0] = 0.5; row[0][1] = -1.0; row[0][2] = 0.25; row[0][3] = 2.0;
    double s_row = ptq_scale_per_tensor(row);
    check_close(s_row, 2.0 / 127.0, 0.0, "(1,N) per-tensor scale");
    Tensor s_rowc = ptq_scale_per_channel(row);
    check(s_rowc.rows == 1 && s_rowc.cols == 4,
          "(1,N) per-channel keeps N scales even though there is one row");

    Tensor col(4, 1);
    col[0][0] = 0.5; col[1][0] = -1.0; col[2][0] = 0.25; col[3][0] = 2.0;
    check_close(ptq_scale_per_tensor(col), 2.0 / 127.0, 0.0,
                "(N,1) per-tensor scale");
    Tensor s_colc = ptq_scale_per_channel(col);
    check(s_colc.rows == 1 && s_colc.cols == 1,
          "(N,1) per-channel has exactly ONE scale (one column)");
    check_close(s_colc[0][0], 2.0 / 127.0, 1e-18,
                "(N,1) per-channel scale equals the global max/127");
}

// ---------------------------------------------------------------------------
// Test 12: D-STRAIGHT-THROUGH -- the STE is the IDENTITY on the gradient.
//   This is the formal definition; FD cannot check it (see the file header),
//   so it is asserted against its own definition, as an exact bit-equality.
// ---------------------------------------------------------------------------
static void test_straight_through_is_identity() {
    Tensor w = make_w();
    Tensor g(3, 4);
    double gd[3][4] = {{1.0, -2.0, 3.0, 0.5}, {-0.25, 1.5, 0.0, -1.0},
                       {2.0, 0.75, -0.5, 3.0}};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) g[r][c] = gd[r][c];

    Tensor out = ptq_straight_through(g);
    bool identical = (out.rows == g.rows && out.cols == g.cols);
    if (identical) {
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 4; ++c)
                if (out[r][c] != g[r][c]) identical = false;
    }
    check(identical, "the STE returns the gradient bit-exactly (identity)");
}

// ---------------------------------------------------------------------------
// Test 13: END-TO-END through a real Model. The point of PTQ is that a TRAINED
//   float model can be stored as int8 and still produce the same ARGMAX --
//   accuracy is the acceptance criterion, not reconstruction error.
//
//   LIVENESS: an accuracy comparison can pass vacuously if the model predicts
//   one class for everything (the `advRob = 0.5000` lesson). So the suite
//   asserts the quantized model SPLITS its predictions on the fixture.
// ---------------------------------------------------------------------------
static void test_end_to_end_accuracy_preserved() {
    Model m;
    m.add_layer(new Dense(6, 16));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(16, 3));

    // Two well-separated clusters so the trained model is confident.
    const int N = 60;
    Tensor X(N, 6), y(N, 3);
    srandom(20261017u);
    for (int i = 0; i < N; ++i) {
        int cls = (i < N / 2) ? 0 : 1;
        double base = (cls == 0) ? 1.0 : -1.0;
        for (size_t c = 0; c < 6; ++c)
            X[i][c] = base + 0.12 * ((double)rand() / RAND_MAX - 0.5);
        for (size_t c = 0; c < 3; ++c) y[i][c] = 0.0;
        y[i][cls] = 1.0;
    }
    m.train(X, y, 0.05, 300);

    // Pre-quantization predictions.
    Tensor logits_fp = m.forward(X);
    Tensor pred_fp = ptq_argmax_rows(logits_fp);
    int split_fp = 0;
    for (size_t i = 0; i < N; ++i) if (pred_fp[i][0] != 0) ++split_fp;
    check(split_fp > 0 && split_fp < (int)N,
          "the float model SPLITS its predictions (liveness: not a constant "
          "predictor)");

    // Quantize every weight in place, per-channel.
    PtqReport rep = ptq_quantize_model(m, PtqGranularity::kPerChannel);
    check(rep.num_tensors > 0, "quantize_model touched at least one tensor");

    Tensor logits_q = m.forward(X);
    Tensor pred_q = ptq_argmax_rows(logits_q);

    double acc_fp = ptq_accuracy(pred_fp, y);
    double acc_q = ptq_accuracy(pred_q, y);
    std::printf("  end-to-end: float acc=%.6f quantized acc=%.6f "
                "(%d tensors)\n", acc_fp, acc_q, rep.num_tensors);
    check(acc_fp > 0.95, "the float model fits the fixture");
    check(acc_q >= acc_fp - 0.05,
          "int8 post-training quantization preserves accuracy within 5%");

    int split_q = 0;
    for (size_t i = 0; i < N; ++i) if (pred_q[i][0] != 0) ++split_q;
    check(split_q > 0 && split_q < (int)N,
          "the quantized model still SPLITS its predictions (liveness)");
}

// ---------------------------------------------------------------------------
// Test 14: the STE makes quantize-aware TRAINING possible. A training loop
//   that quantizes after every step must still reduce the loss -- and the
//   weights must actually be on the grid afterwards.
// ---------------------------------------------------------------------------
static void test_quantize_aware_training_reduces_loss() {
    Model m;
    m.add_layer(new Dense(6, 16));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(16, 3));

    const int N = 60;
    Tensor X(N, 6), y(N, 3);
    srandom(31337u);
    for (int i = 0; i < N; ++i) {
        int cls = (i < N / 2) ? 0 : 1;
        double base = (cls == 0) ? 1.0 : -1.0;
        for (size_t c = 0; c < 6; ++c)
            X[i][c] = base + 0.12 * ((double)rand() / RAND_MAX - 0.5);
        for (size_t c = 0; c < 3; ++c) y[i][c] = 0.0;
        y[i][cls] = 1.0;
    }

    auto loss_now = [&]() {
        Tensor out = m.forward(X);
        Tensor g = (out - y) * (2.0 / static_cast<double>(X.cols));
        double l = 0.0;
        for (size_t i = 0; i < out.rows * out.cols; ++i)
            l += (out.data[i] - y.data[i]) * (out.data[i] - y.data[i]);
        return std::sqrt(l / static_cast<double>(out.rows * out.cols));
    };

    double l0 = loss_now();
    const int steps = 120;
    const double lr = 0.05;
    Adam opt(lr);
    for (int s = 0; s < steps; ++s) {
        Tensor out = m.forward(X);
        // Repo contract: Layer::backward is the gradient of
        // L = sum(grad_output * forward(x)) with NO 1/2 anywhere, so MSE needs
        // 2*(pred - y)/N -- not (pred - y)/N. A constant seed of (pred - y)/N
        // halves every gradient and was a real session's test bug.
        Tensor g = (out - y) * (2.0 / static_cast<double>(X.cols));
        m.backward(g, lr);
        opt.step(m);
        // Simulate-aware: after the update, project the weights back onto the
        // int8 grid, so the optimiser never sees an off-grid weight.
        ptq_quantize_model(m, PtqGranularity::kPerChannel);
    }
    double l1 = loss_now();
    std::printf("  quantize-aware: loss %.6f -> %.6f\n", l0, l1);
    check(l1 < l0, "quantize-aware training reduces the loss");

    // LIVENESS: a loss curve can fall because every hidden weight was zeroed
    // (a readout bias alone moves the loss). Assert the first layer still
    // carries non-trivial mass.
    double wsum = 0.0;
    for (Tensor* p : m.layers[0]->parameters()) {
        for (size_t i = 0; i < p->data.size(); ++i)
            wsum += std::fabs(p->data[i]);
    }
    check(wsum > 0.0, "the first layer still holds non-zero weights (liveness)");
}

// ---------------------------------------------------------------------------
// Test 16: added because mutation M12 SURVIVED, and the probe said why.
//   M12 removes the write-back `*p = w_hat;`, so `ptq_quantize_model` reports
//   work it does not do. The end-to-end accuracy assertion CANNOT see it: an
//   UN-quantized model is trivially within the 5% tolerance of the float one.
//   *A test that only checks an outcome cannot see a bug in how that outcome is
//   produced* -- the same lesson the ortho-penalty session recorded for its
//   gradient-accumulator overwrite.
//
//   The discriminator is GRID MEMBERSHIP: after quantization every weight must
//   satisfy |w/scale - round(w/scale)| == 0, i.e. it is exactly on the grid.
// ---------------------------------------------------------------------------
static void test_quantize_model_actually_writes_the_grid() {
    Model m;
    m.add_layer(new Dense(5, 7));
    m.add_layer(new Dense(7, 3));
    srandom(24680u);

    PtqReport rep = ptq_quantize_model(m, PtqGranularity::kPerChannel);
    check(rep.num_tensors > 0, "quantize_model reports touching tensors");

    // Every parameter must now lie EXACTLY on its per-channel int8 grid.
    double worst_off_grid = 0.0;
    int checked = 0;
    for (auto& layer : m.layers) {
        std::vector<Tensor*> ps = layer->parameters();
        for (Tensor* p : ps) {
            if (p->rows == 0 || p->cols == 0) continue;
            Tensor scales = ptq_scale_per_channel(*p);
            for (size_t i = 0; i < p->rows; ++i) {
                for (size_t j = 0; j < p->cols; ++j) {
                    double s = scales[0][j];
                    if (s == 0.0) continue;
                    double code = (*p)[i][j] / s;
                    worst_off_grid =
                        std::fmax(worst_off_grid,
                                  std::fabs(code - std::floor(code + 0.5)));
                    ++checked;
                }
            }
        }
    }
    std::printf("  grid membership: %d entries checked, worst off-grid = %.3e\n",
                checked, worst_off_grid);
    check(checked > 0, "the grid check actually examined weights (liveness)");
    // Tolerance rather than exactly 0.0 because `w_hat = q*scale` and then
    // `w_hat/scale` is a division round-trip, not the original integer.
    check(worst_off_grid < 1e-9,
          "every quantized weight lies on the int8 grid (catches M12)");

    // And the report must be non-degenerate: a report of zero error over a
    // non-zero tensor would itself be a lie.
    check(rep.max_abs_err > 0.0,
          "the report's max error is non-zero on non-trivial weights");
    check(rep.rel_l2_err > 0.0,
          "the report's relative L2 error is non-zero on non-trivial weights");
}

// ---------------------------------------------------------------------------
// Test 17: added because mutation M15 SURVIVED. The mismatched-scale-width
//   guard is unreachable from the internal callers, so only a DIRECT external
//   call can reach it -- and the suite made none.
// ---------------------------------------------------------------------------
static void test_dequantize_per_channel_rejects_mismatched_scales() {
    Tensor codes(2, 3);
    codes[0][0] = 1; codes[0][1] = 2; codes[0][2] = 3;
    codes[1][0] = -1; codes[1][1] = -2; codes[1][2] = -3;

    Tensor good(1, 3);
    good[0][0] = 0.1; good[0][1] = 0.2; good[0][2] = 0.3;
    Tensor out = ptq_dequantize_per_channel(codes, good);
    check(out.rows == 2 && out.cols == 3, "matching scales dequantize fine");

    bool threw_width = false;
    try {
        Tensor bad(1, 2);                      // wrong width
        bad[0][0] = 0.1; bad[0][1] = 0.2;
        (void)ptq_dequantize_per_channel(codes, bad);
    } catch (const std::invalid_argument&) {
        threw_width = true;
    }
    check(threw_width,
          "a scales tensor of the wrong WIDTH throws (catches M15)");

    bool threw_rows = false;
    try {
        Tensor bad(2, 3);                      // wrong height
        (void)ptq_dequantize_per_channel(codes, bad);
    } catch (const std::invalid_argument&) {
        threw_rows = true;
    }
    check(threw_rows, "a scales tensor of the wrong HEIGHT throws");
}

// ---------------------------------------------------------------------------
// Test 18: added because mutation M10 SURVIVED. `>` vs `>=` in the argmax is
//   invisible unless a fixture contains an EXACT tie.
//
//   MEASURED by exhaustive enumeration (probe): over all 120 logit vectors of
//   length 1-4 drawn from {0, 1, 2}, 105 contain an exact tie, and the two
//   rules DIFFER on 60 of them. So the tie is common and the discriminator is
//   cheap -- the suite simply had no tie in it.
// ---------------------------------------------------------------------------
static void test_argmax_tie_goes_to_lowest_index() {
    // (a) The tie is at the maximum: [1.0, 2.0, 2.0] -> lowest index 1, NOT 2.
    Tensor t(1, 3);
    t[0][0] = 1.0; t[0][1] = 2.0; t[0][2] = 2.0;
    check(ptq_argmax_rows(t)[0][0] == 1.0,
          "argmax tie at the top goes to the LOWEST index");

    // (b) A tie that does NOT include index 0, which `>=` would resolve to the
    //     last index and a fixture where class 0 wins would never see.
    Tensor u(1, 3);
    u[0][0] = 5.0; u[0][1] = 1.0; u[0][2] = 3.0;
    check(ptq_argmax_rows(u)[0][0] == 0.0,
          "a strictly-greater value at index 0 wins over later ones");

    // (c) Full tie across every column.
    Tensor v(1, 4);
    for (size_t c = 0; c < 4; ++c) v[0][c] = 3.0;
    check(ptq_argmax_rows(v)[0][0] == 0.0,
          "an all-equal row returns index 0");

    // (d) Ties on the two-row fixture: a strictly-greater value at index 1 with
    //     a tied value at index 2 -- the asymmetric case.
    Tensor w(2, 3);
    w[0][0] = 0.0; w[0][1] = 7.0; w[0][2] = 7.0;
    w[1][0] = 9.0; w[1][1] = 9.0; w[1][2] = 0.0;
    Tensor p = ptq_argmax_rows(w);
    check(p[0][0] == 1.0, "row 0 tie -> lowest index 1");
    check(p[1][0] == 0.0, "row 1 tie -> lowest index 0");

    // (e) A single-column row: the only index is 0.
    Tensor s(2, 1);
    s[0][0] = 4.0; s[1][0] = -4.0;
    Tensor sp = ptq_argmax_rows(s);
    check(sp[0][0] == 0.0 && sp[1][0] == 0.0,
          "a single-column row always returns index 0");
}

// ---------------------------------------------------------------------------
// Test 19: added because mutation M11 SURVIVED, and the probe localised it.
//   M11 replaces ptq_relative_l2's zero-denominator guard with `0.0 / 0.0`,
//   which the compiler evaluates to NaN (MEASURED: -nan, isnan == 1). The guard
//   fires only when the reference tensor `w` is EXACTLY all-zero, and no
//   fixture called the metric with one -- so the mutation was unreachable rather
//   than inert.
//
//   This is D4 on the METRIC side: 0/0 is the same trap as the quantizer's, in
//   the same file, one function over.
// ---------------------------------------------------------------------------
static void test_relative_l2_on_zero_tensor_is_zero_not_nan() {
    Tensor z(2, 3);                       // all-zero
    double v = ptq_relative_l2(z, z);
    std::printf("  relative_l2(zeros, zeros) = %.17g\n", v);
    check(!std::isnan(v), "relative_l2 of two zero tensors is NOT NaN (catches M11)");
    check(v == 0.0, "relative_l2 of two zero tensors is exactly 0");

    // The other branch of the same guard: a zero reference with a NON-zero
    // estimate must return the raw squared error, not NaN and not 0.
    Tensor est(2, 3);
    est[0][0] = 1.0; est[1][1] = 1.0;      // two unit errors
    double v2 = ptq_relative_l2(z, est);
    check(!std::isnan(v2), "zero reference with a non-zero estimate is not NaN");
    check_close(v2, 2.0, 1e-15,
                "zero reference returns the raw sum of squared errors");
}

// ---------------------------------------------------------------------------
// Test 20: mutation M08 was PROVED INERT and is re-checked every run by the
//   harness's INVERTED-verdict function (see tests/mutation_ptq.sh).
//
//   The measurement: ptq_dequantize's zero-scale guard returns zeros, and the
//   unguarded path computes codes * scale. For every FINITE scale s and finite
//   code c these agree when s == 0, because IEEE 754 defines c * 0.0 == 0.0 for
//   any finite c -- 127 * 0.0 is 0.0, not NaN. NaN requires 0 * inf or inf * 0,
//   and `scale` is a finite double here.
//   MEASURED over the cross product {0, 1e-320, 1e-12, 1e-6, 1} x
//   {0, +-1, +-127, 0.5}: ZERO differing cases.
//
//   So the guard documents D4 rather than implementing it, and the assertion
//   below pins the PROPERTY the guard was there to express.
// ---------------------------------------------------------------------------
static void test_dequantize_zero_scale_property() {
    Tensor codes(1, 4);
    codes[0][0] = 127.0; codes[0][1] = -127.0; codes[0][2] = 1.0; codes[0][3] = 0.0;
    Tensor out = ptq_dequantize(codes, 0.0);
    bool all_zero = true, any_nan = false;
    for (size_t j = 0; j < codes.cols; ++j) {
        if (out[0][j] != 0.0) all_zero = false;
        if (std::isnan(out[0][j])) any_nan = true;
    }
    check(all_zero,
          "dequantizing NON-ZERO codes by a zero scale gives exact zeros "
          "(IEEE 754: c*0.0 == 0.0)");
    check(!any_nan, "a zero scale never yields NaN even for non-zero codes");

    // A subnormal scale is finite and must behave like any other finite scale.
    Tensor tiny = ptq_dequantize(codes, 1e-320);
    check(!std::isnan(tiny[0][0]), "a subnormal scale does not produce NaN");
}

int main() {
    test_per_tensor_scale();
    test_codes_and_saturation();
    test_dequantize_matches_oracle();
    test_rounding_is_half_away_from_zero();
    test_zero_tensor_is_not_nan();
    test_per_channel_scales();
    test_per_channel_codes();
    test_per_channel_w_hat_matches_oracle();
    test_per_channel_dominates();
    test_idempotence_and_no_aliasing();
    test_empty_throws();
    test_codes_are_exact_integers();
    test_rank_one_shapes();
    test_straight_through_is_identity();
    test_end_to_end_accuracy_preserved();
    test_quantize_aware_training_reduces_loss();
    test_quantize_model_actually_writes_the_grid();
    test_dequantize_per_channel_rejects_mismatched_scales();
    test_argmax_tie_goes_to_lowest_index();
    test_relative_l2_on_zero_tensor_is_zero_not_nan();
    test_dequantize_zero_scale_property();

    std::printf("=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}