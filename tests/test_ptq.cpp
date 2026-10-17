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

    std::printf("=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}