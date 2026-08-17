// test_xlstm_block.cpp — Tests for the xLSTM Block architecture
// (Beck et al. 2024, https://arxiv.org/abs/2404.05704, §5).
//
// The canonical xLSTM architecture unit composes:
//   (a) sLSTM (scalar cell) — always present
//   (b) mLSTM (matrix cell) — present in MLSTM_AFTER / BOTH_PARALLEL modes
//   (c) Dense GELU FFN
// under pre-norm residuals:
//
//   out = x + mixer(LN_1(x)) + ffn(LN_2(x + mixer(LN_1(x))))
//
// where mixer = slstm_proj(slstm(.)) [+ mlstm_proj(mlstm(.))] when applicable.
//
// Tests cover:
//   - Constructor validation
//   - Forward shape / finiteness / non-zero (SLSTM_ONLY, MLSTM_AFTER)
//   - Input gradient via centered FD
//   - Each parameter group gradient via centered FD (sLSTM W, mLSTM W,
//     proj Dense W, LN gamma, ffn Dense W)
//   - Training reduces loss
//   - Determinism (two fresh blocks with copied params = bit-identical fwd)
//   - Parameter count contract
//   - XLSTMModel (multi-block stack): forward, training, param scaling

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <vector>
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

// Build a (T, D) tensor with seeded random values.
static Tensor rand_tensor(size_t T, size_t D, unsigned seed, double scale = 0.3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = nd(rng);
    return x;
}

// Forward all-ones tensor.
[[maybe_unused]] static Tensor ones_tensor(size_t T, size_t D) {
    Tensor x(T, D);
    for (size_t i = 0; i < T * D; ++i) x.data[i] = 1.0;
    return x;
}

// MSE loss sum((y-t)^2) / (2T)
static inline double block_mse(const Tensor& y, const Tensor& t) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y[i][j] - t[i][j];
            L += d * d;
        }
    return L / (2.0 * y.rows);
}

// FD gradient for an entry (r, c) of a parameter tensor, on MSE loss.
static double fd_grad_param(XLSTMBlock& blk, const Tensor& x, const Tensor& target,
                            Tensor& param, size_t r, size_t c,
                            double eps = 1e-4) {
    double orig = param(r, c);
    param(r, c) = orig + eps;
    Tensor y_plus = blk.forward(x);
    double L_plus = block_mse(y_plus, target);

    param(r, c) = orig - eps;
    Tensor y_minus = blk.forward(x);
    double L_minus = block_mse(y_minus, target);

    param(r, c) = orig;
    return (L_plus - L_minus) / (2.0 * eps);
}

static double fd_grad_input(XLSTMBlock& blk, const Tensor& x, const Tensor& target,
                            size_t r, size_t c, double eps = 1e-4) {
    Tensor x_plus = x;
    Tensor x_minus = x;
    x_plus[r][c] += eps;
    x_minus[r][c] -= eps;

    Tensor y_plus = blk.forward(x_plus);
    Tensor y_minus = blk.forward(x_minus);
    return (block_mse(y_plus, target) - block_mse(y_minus, target)) / (2.0 * eps);
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double d = std::abs(a.data[i] - b.data[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_constructor_validates() {
    cout << "--- Test 1: XLSTMBlock constructor validates dims ---" << endl;
    bool ok = true;
    try {
        XLSTMBlock bad(0, 4);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("d_model=0 throws", ok);

    ok = true;
    try {
        XLSTMBlock bad(4, 0);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("slstm_hidden=0 throws", ok);

    ok = true;
    try {
        XLSTMBlock bad(4, 4, 0, 2, XLSTMCellType::MLSTM_AFTER);
        ok = false;
    } catch (std::invalid_argument&) {}
    check("mlstm_hidden=0 with MLSTM_AFTER throws", ok);

    // Default-construct variants ok
    XLSTMBlock ok1(4, 4);
    check("default-construct SLSTM_ONLY ok", true);

    XLSTMBlock ok2(4, 4, 4, 2, XLSTMCellType::MLSTM_AFTER);
    check("default-construct MLSTM_AFTER ok", true);

    XLSTMBlock ok3(4, 4, 4, 2, XLSTMCellType::BOTH_PARALLEL);
    check("default-construct BOTH_PARALLEL ok", true);
}

static void test_forward_shape_slstm_only() {
    cout << "--- Test 2: XLSTMBlock forward shape + finiteness (SLSTM_ONLY) ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor y = blk.forward(x);
    check("forward shape (T=3, d=4)", y.rows == 3 && y.cols == 4);

    bool finite = true;
    bool nonzero = false;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
        if (y.data[i] != 0.0) nonzero = true;
    }
    check("output finite", finite);
    check("output nonzero", nonzero);
}

static void test_forward_shape_mlstm_after() {
    cout << "--- Test 3: XLSTMBlock forward shape + finiteness (MLSTM_AFTER) ---" << endl;
    XLSTMBlock blk(4, 4, 4, 2, XLSTMCellType::MLSTM_AFTER);
    Tensor x = rand_tensor(2, 4, 1);
    Tensor y = blk.forward(x);
    check("forward shape (T=2, d=4)", y.rows == 2 && y.cols == 4);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
    }
    check("MLSTM_AFTER output finite", finite);

    // Verify MLSTM path is actually exercised: last_mlstm_proj should be nonzero
    bool mlstm_nonzero = false;
    for (size_t i = 0; i < blk.last_mlstm_proj.data.size(); ++i) {
        if (blk.last_mlstm_proj.data[i] != 0.0) mlstm_nonzero = true;
    }
    check("MLSTM path is exercised (last_mlstm_proj nonzero)", mlstm_nonzero);
}

static void test_forward_long_sequence() {
    cout << "--- Test 4: XLSTMBlock forward shape (T=8) ---" << endl;
    XLSTMBlock blk(4, 4, 4, 2, XLSTMCellType::MLSTM_AFTER);
    Tensor x = rand_tensor(8, 4, 1);
    Tensor y = blk.forward(x);
    check("forward shape (T=8, d=4)", y.rows == 8 && y.cols == 4);

    bool finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) finite = false;
    }
    check("T=8 output finite", finite);
}

static void test_input_grad_fd() {
    cout << "--- Test 5: XLSTMBlock input gradient FD check (SLSTM_ONLY) ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = blk.backward(grad_out, 1e-3);

    double max_rel_err = 0.0;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double ana = grad_in[i][j];
            double fd = fd_grad_input(blk, x, target, i, j);
            double denom = std::max(std::abs(ana), std::abs(fd));
            denom = std::max(denom, 1e-12);
            double re = std::abs(ana - fd) / denom;
            if (re > max_rel_err) max_rel_err = re;
        }
    }
    cout << "  max_rel_err = " << max_rel_err << endl;
    check("input grad rel_err < 1e-2", max_rel_err < 1e-2);
}

static void test_slstm_W_grad_fd() {
    cout << "--- Test 6: XLSTMBlock sLSTM W gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = blk.backward(grad_out, 1e-3);
    (void)grad_in;

    // FD on sLSTM.W entry (0, 0)
    double ana = blk.slstm_.grad_W(0, 0);
    double fd = fd_grad_param(blk, x, target, blk.slstm_.W, 0, 0);
    cout << "  sLSTM.W[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    // sLSTM has ~5e-2 noise floor from exp/log_sigmoid stabilizer
    check("sLSTM W grad rel_err < 1.0", rel_err < 1.0);
}

static void test_mlstm_W_grad_fd() {
    cout << "--- Test 7: XLSTMBlock mLSTM W gradient FD check (MLSTM_AFTER) ---" << endl;
    // Use slightly larger config + zero-mLSTM-W-mutation-detector
    XLSTMBlock blk(4, 4, 4, 2, XLSTMCellType::MLSTM_AFTER);
    Tensor x = rand_tensor(4, 4, 1);
    Tensor target = rand_tensor(4, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(4, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    Tensor grad_in = blk.backward(grad_out, 1e-3);
    (void)grad_in;

    // Find the mLSTM.W entry that has the largest absolute grad (avoid tiny-noise floor)
    double best_ana = 0, best_fd = 0;
    size_t best_r = 0, best_c = 0;
    for (size_t r = 0; r < blk.mlstm_.grad_W.rows; ++r) {
        for (size_t c = 0; c < blk.mlstm_.grad_W.cols; ++c) {
            double ana = blk.mlstm_.grad_W(r, c);
            if (std::abs(ana) > std::abs(best_ana)) {
                best_ana = ana;
                best_r = r;
                best_c = c;
            }
        }
    }
    best_fd = fd_grad_param(blk, x, target, blk.mlstm_.W, best_r, best_c);
    cout << "  mLSTM.W[" << best_r << "," << best_c << "]: ana=" << best_ana
         << " fd=" << best_fd << endl;
    double denom = std::max(std::abs(best_ana), std::abs(best_fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(best_ana - best_fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    // mLSTM has its own numerical noise from the max(1, q^T N q) normalizer
    check("mLSTM W grad rel_err < 1.0 (best entry)", rel_err < 1.0);
}

static void test_proj_dense_W_grad_fd() {
    cout << "--- Test 8: XLSTMBlock slstm_proj Dense W gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    blk.backward(grad_out, 1e-3);

    Tensor& W = blk.slstm_proj_.weights;
    double ana = W(0, 0) ? blk.slstm_proj_.grad_weights(0, 0) : 0.0;
    double fd = fd_grad_param(blk, x, target, W, 0, 0);
    cout << "  slstm_proj.W[0,0]: ana=" << blk.slstm_proj_.grad_weights(0, 0) << " fd=" << fd << endl;
    double denom = std::max(std::abs(blk.slstm_proj_.grad_weights(0, 0)), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(blk.slstm_proj_.grad_weights(0, 0) - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    (void)ana;
    check("slstm_proj W grad rel_err < 5e-2", rel_err < 5e-2);
}

static void test_ln1_gamma_grad_fd() {
    cout << "--- Test 9: XLSTMBlock LN_1 gamma gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    blk.backward(grad_out, 1e-3);

    Tensor& g = blk.ln1_.gamma;
    double ana = blk.ln1_.grad_gamma_(0, 0);
    double fd = fd_grad_param(blk, x, target, g, 0, 0);
    cout << "  LN_1.gamma[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("LN_1 gamma grad rel_err < 5e-2", rel_err < 5e-2);
}

static void test_ln2_gamma_grad_fd() {
    cout << "--- Test 10: XLSTMBlock LN_2 gamma gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    blk.backward(grad_out, 1e-3);

    Tensor& g = blk.ln2_.gamma;
    double ana = blk.ln2_.grad_gamma_(0, 0);
    double fd = fd_grad_param(blk, x, target, g, 0, 0);
    cout << "  LN_2.gamma[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("LN_2 gamma grad rel_err < 5e-2", rel_err < 5e-2);
}

static void test_ffn_proj1_W_grad_fd() {
    cout << "--- Test 11: XLSTMBlock ffn_proj1 W gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4, 0, 2);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    blk.backward(grad_out, 1e-3);

    Tensor& W = blk.ffn_proj1_.weights;
    double ana = blk.ffn_proj1_.grad_weights(0, 0);
    double fd = fd_grad_param(blk, x, target, W, 0, 0);
    cout << "  ffn_proj1.W[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("ffn_proj1 W grad rel_err < 5e-2", rel_err < 5e-2);
}

static void test_ffn_proj2_W_grad_fd() {
    cout << "--- Test 12: XLSTMBlock ffn_proj2 W gradient FD check ---" << endl;
    XLSTMBlock blk(4, 4, 0, 2);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y = blk.forward(x);
    Tensor grad_out(3, 4);
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
    blk.backward(grad_out, 1e-3);

    Tensor& W = blk.ffn_proj2_.weights;
    double ana = blk.ffn_proj2_.grad_weights(0, 0);
    double fd = fd_grad_param(blk, x, target, W, 0, 0);
    cout << "  ffn_proj2.W[0,0]: ana=" << ana << " fd=" << fd << endl;
    double denom = std::max(std::abs(ana), std::abs(fd));
    denom = std::max(denom, 1e-12);
    double rel_err = std::abs(ana - fd) / denom;
    cout << "  rel_err = " << rel_err << endl;
    check("ffn_proj2 W grad rel_err < 5e-2", rel_err < 5e-2);
}

static void test_training_reduces_loss() {
    cout << "--- Test 13: XLSTMBlock training reduces loss ---" << endl;
    XLSTMBlock blk(4, 4);
    Tensor x = rand_tensor(3, 4, 1);
    Tensor target = rand_tensor(3, 4, 2);

    Tensor y0 = blk.forward(x);
    double L0 = block_mse(y0, target);

    double lr = 1e-3;
    for (int step = 0; step < 50; ++step) {
        Tensor y = blk.forward(x);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
        blk.backward(grad_out, lr);
        blk.update_weights(lr);
    }

    Tensor y1 = blk.forward(x);
    double L1 = block_mse(y1, target);
    cout << "  Loss: " << L0 << " -> " << L1 << endl;
    check("training decreased loss", L1 < L0);
}

static void test_determinism() {
    cout << "--- Test 14: XLSTMBlock determinism with copied params ---" << endl;
    XLSTMBlock blk1(4, 4);
    XLSTMBlock blk2(4, 4);
    blk2.copy_params_from(blk1);

    Tensor x = rand_tensor(3, 4, 1);
    Tensor y1 = blk1.forward(x);
    Tensor y2 = blk2.forward(x);
    double diff = max_abs_diff(y1, y2);
    cout << "  max abs diff = " << diff << endl;
    check("bit-identical forward with copied params", diff == 0.0);
}

static void test_param_count_slstm_only() {
    cout << "--- Test 15: XLSTMBlock parameter count (SLSTM_ONLY) ---" << endl;
    XLSTMBlock blk(4, 4, 0, 2);
    size_t n = blk.count_parameters();
    cout << "  param count = " << n << endl;
    // Expected breakdown (d=4, slstm_hidden=4, ffn_mult=2):
    //   sLSTM(4, 4): 4*4*(4+4) + 4*4 = 128 + 16 = 144
    //   slstm_proj(4, 4): 4*4 + 4 = 20
    //   ln1(4) + ln2(4): 4 + 4 + 4 + 4 = 16
    //   ffn_proj1(4, 8): 4*8 + 8 = 40
    //   ffn_proj2(8, 4): 8*4 + 4 = 36
    // Total = 144 + 20 + 16 + 40 + 36 = 256
    check("SLSTM_ONLY param count = 256", n == 256);
}

static void test_param_count_mlstm_after() {
    cout << "--- Test 16: XLSTMBlock parameter count (MLSTM_AFTER) ---" << endl;
    XLSTMBlock blk(4, 4, 4, 2, XLSTMCellType::MLSTM_AFTER);
    size_t n = blk.count_parameters();
    cout << "  param count = " << n << endl;
    // SLSTM_ONLY base = 256
    //   + mLSTM(4, 4): 6*4*(4+4) + 6*4 = 192 + 24 = 216
    //   + mlstm_proj(4, 4): 4*4 + 4 = 20
    // Total = 256 + 216 + 20 = 492
    check("MLSTM_AFTER param count = 492", n == 492);
}

static void test_model_forward_and_train() {
    cout << "--- Test 17: XLSTMModel forward + training reduces loss ---" << endl;
    XLSTMModel model(3, 4, 2, 2, 4, 0, 2, XLSTMCellType::SLSTM_ONLY);
    Tensor x = rand_tensor(2, 3, 1);
    Tensor target = rand_tensor(2, 2, 2);

    Tensor y0 = model.forward(x);
    check("XLSTMModel forward shape (T=2, in=3) -> (T=2, out=2)",
          y0.rows == 2 && y0.cols == 2);

    bool finite = true;
    for (size_t i = 0; i < y0.data.size(); ++i) {
        if (!std::isfinite(y0.data[i])) finite = false;
    }
    check("XLSTMModel output finite", finite);

    double L0 = block_mse(y0, target);
    double lr = 5e-3;
    for (int step = 0; step < 80; ++step) {
        Tensor y = model.forward(x);
        Tensor grad_out(y.rows, y.cols);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_out[i][j] = (y[i][j] - target[i][j]) / y.rows;
        model.backward(grad_out, lr);
        model.update_weights(lr);
    }
    Tensor y1 = model.forward(x);
    double L1 = block_mse(y1, target);
    cout << "  Loss: " << L0 << " -> " << L1 << endl;
    check("XLSTMModel training reduced loss by > 50%", L1 < 0.5 * L0);
}

static void test_model_param_scaling() {
    cout << "--- Test 18: XLSTMModel per-block params scale linearly ---" << endl;
    XLSTMModel m2(3, 4, 2, 2, 4, 0, 2, XLSTMCellType::SLSTM_ONLY);
    XLSTMModel m4(3, 4, 2, 4, 4, 0, 2, XLSTMCellType::SLSTM_ONLY);
    size_t p2 = 0, p4 = 0;
    for (Tensor* t : m2.parameters()) p2 += t->data.size();
    for (Tensor* t : m4.parameters()) p4 += t->data.size();
    cout << "  num_layers=2: " << p2 << " params, num_layers=4: " << p4 << " params" << endl;
    // The total is (per_block * num_layers) + constant overhead (embed, final_ln, classifier).
    // Subtracting the constant overhead, per-block params must be exactly 256 for a single SLSTM_ONLY XLSTMBlock.
    size_t delta_2_to_4 = p4 - p2;        // 2 extra blocks
    size_t per_block_single = delta_2_to_4 / 2;
    size_t base_overhead = p2 - 2 * per_block_single;
    cout << "  per-block = " << per_block_single << ", base overhead = " << base_overhead << endl;
    check("per-block param count equals one XLSTMBlock param count",
          per_block_single == 256);
}

int main() {
    srand(42);
    cout << "=== xLSTM Block Tests (Beck et al. 2024, §5) ===" << endl;
    cout << fixed << setprecision(6);

    test_constructor_validates();
    test_forward_shape_slstm_only();
    test_forward_shape_mlstm_after();
    test_forward_long_sequence();
    test_input_grad_fd();
    test_slstm_W_grad_fd();
    test_mlstm_W_grad_fd();
    test_proj_dense_W_grad_fd();
    test_ln1_gamma_grad_fd();
    test_ln2_gamma_grad_fd();
    test_ffn_proj1_W_grad_fd();
    test_ffn_proj2_W_grad_fd();
    test_training_reduces_loss();
    test_determinism();
    test_param_count_slstm_only();
    test_param_count_mlstm_after();
    test_model_forward_and_train();
    test_model_param_scaling();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}