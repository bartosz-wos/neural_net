// Tests for FastFormer (Wu et al. AAAI 2021)
// "FastFormer: Additive Attention Can Be All You Need"
// https://arxiv.org/abs/2108.09084
#include "nn/nn.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace nn;

static int g_pass = 0;
static int g_fail = 0;

struct CoutFlusher {
    CoutFlusher() {
        // unbuffered so crashes still show their tail
        std::cout.setf(std::ios::unitbuf);
    }
} _flusher;

#define CHECK(cond, name)                                                    \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else { ++g_fail; std::cout << "FAIL: " << name                        \
                                   << " (line " << __LINE__ << ")\n"; }      \
    } while (0)

#define CHECK_NEAR(a, b, tol, name)                                         \
    do {                                                                     \
        double aa = (a), bb = (b);                                           \
        if (std::abs(aa - bb) <= (tol)) { ++g_pass; }                        \
        else { ++g_fail; std::cout << "FAIL: " << name                       \
                                   << " expected " << bb << " got " << aa    \
                                   << " (line " << __LINE__ << ")\n"; }      \
    } while (0)


// ----------------------------------------------------------------------------
// Centered finite-difference numerical gradient for a single element (i, j)
// of the input tensor. Returns a scalar.
// ----------------------------------------------------------------------------
static double fd_input_grad(FastFormerAttention& m, Tensor& x, size_t i, size_t j,
                            const Tensor& grad_output, double eps = 1e-5) {
    Tensor x_orig = x.clone();
    // forward at +eps
    x(i, j) += eps;
    Tensor y_plus = m.forward(x);
    double f_plus = 0.0;
    for (size_t r = 0; r < y_plus.rows; ++r)
        for (size_t c = 0; c < y_plus.cols; ++c)
            f_plus += y_plus[r][c] * grad_output[r][c];
    m.zero_grad();
    x(i, j) -= 2 * eps;
    Tensor y_minus = m.forward(x);
    double f_minus = 0.0;
    for (size_t r = 0; r < y_minus.rows; ++r)
        for (size_t c = 0; c < y_minus.cols; ++c)
            f_minus += y_minus[r][c] * grad_output[r][c];
    x(i, j) = x_orig(i, j);
    return (f_plus - f_minus) / (2 * eps);
}

static double fd_param_grad(FastFormerAttention& m, Tensor& param, Tensor& grad_param,
                            Tensor* param_ptr, Tensor* grad_ptr, size_t i, size_t j,
                            Tensor& x, const Tensor& grad_output, double eps = 1e-5) {
    (void)param; (void)grad_param;
    double orig = (*param_ptr)(i, j);
    // Forward at +eps
    (*param_ptr)(i, j) = orig + eps;
    Tensor y_plus = m.forward(x);
    double f_plus = 0.0;
    for (size_t r = 0; r < y_plus.rows; ++r)
        for (size_t c = 0; c < y_plus.cols; ++c)
            f_plus += y_plus[r][c] * grad_output[r][c];
    (*param_ptr)(i, j) = orig - eps;
    Tensor y_minus = m.forward(x);
    double f_minus = 0.0;
    for (size_t r = 0; r < y_minus.rows; ++r)
        for (size_t c = 0; c < y_minus.cols; ++c)
            f_minus += y_minus[r][c] * grad_output[r][c];
    (*param_ptr)(i, j) = orig;
    return (f_plus - f_minus) / (2 * eps);
}

// Re-runs forward + backward at the unperturbed state to refresh analytical gradients.
static void refresh_grads(FastFormerAttention& m, Tensor& x, const Tensor& grad_output) {
    Tensor y = m.forward(x);
    m.zero_grad();
    m.backward(grad_output, 0.0);
    (void)y;
}


// =============================================================================
// Constructor validation tests
// =============================================================================
static void test_constructor_validation() {
    std::cout << "  [constructor validation]\n";
    bool ok1 = true;
    try { FastFormerAttention m(8, 2); } catch (...) { ok1 = false; }
    CHECK(ok1, "valid construction (d=8, H=2)");

    bool ok2 = false;
    try { FastFormerAttention m(0, 2); } catch (...) { ok2 = true; }
    CHECK(ok2, "throws on d_model=0");

    bool ok3 = false;
    try { FastFormerAttention m(8, 0); } catch (...) { ok3 = true; }
    CHECK(ok3, "throws on num_heads=0");

    bool ok4 = false;
    try { FastFormerAttention m(7, 2); } catch (...) { ok4 = true; }
    CHECK(ok4, "throws on non-divisible (d=7, H=2)");

    bool ok5 = true;
    try { FastFormerAttention m(8, 4); } catch (...) { ok5 = false; }
    CHECK(ok5, "valid (d=8, H=4)");

    bool ok6 = true;
    try { FastFormerAttention m(16, 1); } catch (...) { ok6 = false; }
    CHECK(ok6, "valid (d=16, H=1)");

    bool ok7 = true;
    try { FastFormerAttention m(8, 2, true); } catch (...) { ok7 = false; }
    CHECK(ok7, "valid causal construction");
}


// =============================================================================
// Forward shape & finiteness
// =============================================================================
static void test_forward_shape() {
    std::cout << "  [forward shape]\n";
    srand(42);
    FastFormerAttention m(8, 2);
    Tensor x = Tensor::random(4, 8, 0.3);
    Tensor y = m.forward(x);
    CHECK(y.rows == 4, "output rows == input rows");
    CHECK(y.cols == 8, "output cols == d_model");
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y[i][j])) finite = false;
    CHECK(finite, "all outputs finite");
}


// =============================================================================
// Forward non-triviality — different inputs produce different outputs
// =============================================================================
static void test_forward_nonzero() {
    std::cout << "  [forward non-triviality]\n";
    srand(42);
    FastFormerAttention m(8, 2);
    Tensor x1 = Tensor::random(4, 8, 0.3);
    Tensor x2 = Tensor::random(4, 8, 0.3);
    Tensor y1 = m.forward(x1);
    Tensor y2 = m.forward(x2);
    double diff = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j)
            diff += std::abs(y1[i][j] - y2[i][j]);
    CHECK(diff > 1e-3, "different inputs produce different outputs");

    // Perturb x1: output should change
    Tensor x1p = x1.clone();
    x1p(0, 0) += 0.5;
    Tensor y1p = m.forward(x1p);
    double diff2 = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j)
            diff2 += std::abs(y1[i][j] - y1p[i][j]);
    CHECK(diff2 > 1e-3, "perturbing input changes output");
}


// =============================================================================
// Causal mask: the alpha softmax is masked causal (future positions don't contribute)
// =============================================================================
static void test_causal_mask() {
    std::cout << "  [causal mask on alpha]\n";
    srand(7);
    FastFormerAttention m(4, 1, /*causal=*/true);
    // Initialize weights to large values so alpha is non-degenerate.
    m.W_q_.fill(0.5); m.W_k_.fill(0.5); m.W_v_.fill(0.5);
    m.W_p_.fill(0.5); m.W_b_.fill(0.5); m.W_o_.fill(0.5);
    m.W_gamma_.fill(0.5); m.b_gamma_[0][0] = 0.5;
    m.b_q_.fill(0.5); m.b_k_.fill(0.5); m.b_v_.fill(0.5);
    m.b_p_.fill(0.5); m.b_b_.fill(0.5); m.b_o_.fill(0.5);

    Tensor x = Tensor::random(4, 4, 0.3);
    m.forward(x);  // populate caches
    // alpha[0][0] should be nonzero; alpha[0][1..3] should be exactly 0 (causal mask).
    CHECK(m.last_alpha_(0, 0) > 0.99, "alpha[0][0] = 1 (causal mask forces self-attention at token 0)");
    for (size_t j = 1; j < 4; ++j) {
        CHECK(std::abs(m.last_alpha_(0, j)) < 1e-9, "alpha[0][j>0] = 0 (causal mask)");
    }

    // Non-causal: alpha[0][j] should be nonzero for all j.
    FastFormerAttention m2(4, 1, /*causal=*/false);
    m2.W_q_.fill(0.5); m2.W_k_.fill(0.5); m2.W_v_.fill(0.5);
    m2.W_p_.fill(0.5); m2.W_b_.fill(0.5); m2.W_o_.fill(0.5);
    m2.W_gamma_.fill(0.5); m2.b_gamma_[0][0] = 0.5;
    m2.b_q_.fill(0.5); m2.b_k_.fill(0.5); m2.b_v_.fill(0.5);
    m2.b_p_.fill(0.5); m2.b_b_.fill(0.5); m2.b_o_.fill(0.5);
    m2.forward(x);
    double max_off = 0.0;
    for (size_t j = 1; j < 4; ++j) max_off = std::max(max_off, m2.last_alpha_(0, j));
    CHECK(max_off > 0.05, "non-causal: alpha[0][j>0] > 0 (no mask)");
}


// =============================================================================
// FD gradient check on input (rel_err < 1e-3 with combined tolerance)
// =============================================================================
static void test_input_fd_grad() {
    std::cout << "  [input FD gradient]\n";
    srand(42);
    FastFormerAttention m(4, 1);  // num_heads=1 required for v1 backward
    Tensor x = Tensor::random(3, 4, 1.0);  // larger scale for non-degenerate softmax
    Tensor y = m.forward(x);
    Tensor grad_output = Tensor::random(3, 4, 0.5);
    m.zero_grad();
    Tensor dx = m.backward(grad_output, 0.0);

    double max_rel = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double fd = fd_input_grad(m, x, i, j, grad_output);
            double ana = dx(i, j);
            double scale = std::max(std::abs(fd), std::abs(ana));
            double denom = std::max(scale, 1e-12);
            double rel = std::abs(fd - ana) / denom;
            // Combine relative + absolute: if the gradient is tiny, an absolute floor is fine.
            if (rel > max_rel && (scale > 1e-6 || std::abs(fd - ana) > 1e-6)) max_rel = rel;
        }
    }
    // Refresh dx for any later checks (none currently, but be safe).
    refresh_grads(m, x, grad_output);
    CHECK(max_rel < 1e-3, "input gradient rel_err < 1e-3");
    if (max_rel >= 1e-3) std::cout << "    max_rel = " << max_rel << "\n";
}


// =============================================================================
// FD gradient check on W_q (one of the projection matrices)
// =============================================================================
static void test_param_fd_grad_Wq() {
    std::cout << "  [W_q FD gradient]\n";
    srand(42);
    FastFormerAttention m(4, 1);
    // Larger init for non-degenerate gradient signal.
    m.W_q_.fill(0.5); m.W_k_.fill(0.5); m.W_v_.fill(0.5);
    m.W_p_.fill(0.5); m.W_b_.fill(0.5); m.W_o_.fill(0.5);
    m.W_gamma_.fill(0.5); m.b_gamma_[0][0] = 0.5;
    m.b_q_.fill(0.5); m.b_k_.fill(0.5); m.b_v_.fill(0.5);
    m.b_p_.fill(0.5); m.b_b_.fill(0.5); m.b_o_.fill(0.5);

    Tensor x = Tensor::random(3, 4, 1.0);
    Tensor y = m.forward(x);
    Tensor grad_output = Tensor::random(3, 4, 0.5);
    m.zero_grad();
    m.backward(grad_output, 0.0);

    // Pick element (0, 0) of W_q
    size_t i = 0, j = 0;
    double fd = fd_param_grad(m, m.W_q_, m.grad_W_q_, &m.W_q_, &m.grad_W_q_, i, j, x, grad_output);
    refresh_grads(m, x, grad_output);
    double ana = m.grad_W_q_(i, j);
    double scale = std::max(std::abs(fd), std::abs(ana));
    double denom = std::max(scale, 1e-12);
    double rel = std::abs(fd - ana) / denom;
    bool ok = rel < 1e-3 || std::abs(fd - ana) < 1e-9;
    CHECK(ok, "W_q[0][0] gradient rel_err < 1e-3");
    if (!ok) std::cout << "    rel = " << rel << " ana=" << ana << " fd=" << fd << " scale=" << scale << "\n";
}


// =============================================================================
// FD gradient check on additional params (W_p, W_b, w_gamma, W_o)
// =============================================================================
static void test_param_fd_grad_more() {
    std::cout << "  [W_p, W_b, w_gamma, W_o FD gradients]\n";
    srand(42);
    FastFormerAttention m(4, 1);
    m.W_q_.fill(0.5); m.W_k_.fill(0.5); m.W_v_.fill(0.5);
    m.W_p_.fill(0.5); m.W_b_.fill(0.5); m.W_o_.fill(0.5);
    m.W_gamma_.fill(0.5); m.b_gamma_[0][0] = 0.5;
    m.b_q_.fill(0.5); m.b_k_.fill(0.5); m.b_v_.fill(0.5);
    m.b_p_.fill(0.5); m.b_b_.fill(0.5); m.b_o_.fill(0.5);

    Tensor x = Tensor::random(3, 4, 1.0);
    Tensor y = m.forward(x);
    Tensor grad_output = Tensor::random(3, 4, 0.5);
    m.zero_grad();
    m.backward(grad_output, 0.0);

    auto check = [&](Tensor* param, Tensor* grad, size_t i, size_t j, const char* nm) {
        double fd = fd_param_grad(m, *param, *grad, param, grad, i, j, x, grad_output);
        refresh_grads(m, x, grad_output);
        double ana = (*grad)(i, j);
        double scale = std::max(std::abs(fd), std::abs(ana));
        double denom = std::max(scale, 1e-12);
        double rel = std::abs(fd - ana) / denom;
        bool ok = rel < 1e-3 || std::abs(fd - ana) < 1e-9;
        CHECK(ok, std::string(nm) + std::string(" gradient rel_err < 1e-3"));
        if (!ok) std::cout << "    " << nm << " rel = " << rel << " ana=" << ana << " fd=" << fd << " scale=" << scale << "\n";
    };

    check(&m.W_p_, &m.grad_W_p_, 0, 0, "W_p");
    check(&m.W_b_, &m.grad_W_b_, 0, 0, "W_b");
    check(&m.W_gamma_, &m.grad_W_gamma_, 0, 0, "W_gamma");
    check(&m.W_o_, &m.grad_W_o_, 0, 0, "W_o");
}


// =============================================================================
// parameters()/gradients() contract — same count, pointers match order
// =============================================================================
static void test_params_gradients_contract() {
    std::cout << "  [parameters()/gradients() contract]\n";
    srand(42);
    FastFormerAttention m(8, 2);
    auto params = m.parameters();
    auto grads = m.gradients();
    CHECK(params.size() == 14, "14 parameters");
    CHECK(grads.size() == 14, "14 gradients");
    bool matches = true;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) matches = false;
    }
    CHECK(matches, "each param matches its grad shape");
}


// =============================================================================
// zero_grad clears all gradient tensors
// =============================================================================
static void test_zero_grad() {
    std::cout << "  [zero_grad]\n";
    srand(42);
    FastFormerAttention m(4, 1);  // num_heads=1 required for v1 backward
    Tensor x = Tensor::random(3, 4, 0.1);
    m.forward(x);
    Tensor go = Tensor::random(3, 4, 0.1);
    m.backward(go, 0.0);
    bool any_nonzero = false;
    for (auto* g : m.gradients()) {
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j)
                if (std::abs((*g)(i, j)) > 1e-12) any_nonzero = true;
    }
    CHECK(any_nonzero, "gradients non-zero after backward");
    m.zero_grad();
    bool all_zero = true;
    for (auto* g : m.gradients()) {
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j)
                if (std::abs((*g)(i, j)) > 1e-12) all_zero = false;
    }
    CHECK(all_zero, "all gradients zero after zero_grad");
}


// =============================================================================
// Determinism — two fresh models produce identical output for identical input
// =============================================================================
static void test_determinism() {
    std::cout << "  [determinism — fresh models]\n";
    srand(42);
    FastFormerAttention m1(8, 2);
    srand(42);
    FastFormerAttention m2(8, 2);
    Tensor x = Tensor::random(3, 8, 0.2);
    Tensor y1 = m1.forward(x);
    Tensor y2 = m2.forward(x);
    double max_diff = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j) {
            double d = std::abs(y1[i][j] - y2[i][j]);
            if (d > max_diff) max_diff = d;
        }
    CHECK(max_diff < 1e-12, "two fresh models identical under same seed");
}


// =============================================================================
// FastFormerBlock tests
// =============================================================================
static void test_block_forward_shape() {
    std::cout << "  [block forward shape]\n";
    srand(42);
    FastFormerBlock b(8, 2, /*ffn=*/0);
    Tensor x = Tensor::random(3, 8, 0.2);
    Tensor y = b.forward(x);
    CHECK(y.rows == 3 && y.cols == 8, "block output shape preserved (no FFN)");

    FastFormerBlock b2(8, 2, /*ffn=*/16);
    Tensor y2 = b2.forward(x);
    CHECK(y2.rows == 3 && y2.cols == 8, "block output shape preserved (with FFN)");
    bool finite = true;
    for (size_t i = 0; i < y2.rows; ++i)
        for (size_t j = 0; j < y2.cols; ++j)
            if (!std::isfinite(y2[i][j])) finite = false;
    CHECK(finite, "block output finite (with FFN)");
}


static void test_block_training() {
    std::cout << "  [block training reduces loss]\n";
    srand(42);
    FastFormerBlock b(8, 1, /*ffn=*/16);  // num_heads=1 required for v1 backward
    Tensor x = Tensor::random(4, 8, 0.3);
    // Target: small zero tensor — model should learn to push output toward 0.
    Tensor target(4, 8);
    target.fill(0.0);

    double lr = 0.01;
    double initial_loss = 0.0;
    for (int step = 0; step < 80; ++step) {
        b.zero_grad();
        Tensor y = b.forward(x);
        Tensor dy(4, 8);
        double loss = 0.0;
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 8; ++j) {
                double diff = y[i][j] - target[i][j];
                loss += diff * diff;
                dy[i][j] = 2.0 * diff / 32.0;
            }
        loss /= 32.0;
        if (step == 0) initial_loss = loss;
        b.backward(dy, 0.0);
        b.update_weights(lr);
    }
    // Re-run final forward for reported loss
    b.zero_grad();
    Tensor y_final = b.forward(x);
    double final_loss = 0.0;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j) {
            double diff = y_final[i][j] - target[i][j];
            final_loss += diff * diff;
        }
    final_loss /= 32.0;
    CHECK(final_loss < initial_loss, "block training reduces loss");
    std::cout << "    initial_loss = " << initial_loss << ", final_loss = " << final_loss << "\n";
}


// =============================================================================
// FastFormerModel tests
// =============================================================================
static void test_model_forward_shape() {
    std::cout << "  [model forward shape]\n";
    srand(42);
    FastFormerModel m(/*input_dim=*/4, /*d_model=*/8, /*output_dim=*/3,
                      /*seq_len=*/5, /*num_blocks=*/2, /*num_heads=*/1,  // H=1 for v1 backward
                      /*ffn_hidden=*/16);
    Tensor x = Tensor::random(4, 5, 0.2);
    Tensor y = m.forward(x);
    CHECK(y.rows == 3, "output rows == num_classes");
    CHECK(y.cols == 1, "output cols == 1");
    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y[i][j])) finite = false;
    CHECK(finite, "model output finite");
}


static void test_model_training() {
    std::cout << "  [model training reduces loss]\n";
    srand(42);
    FastFormerModel m(/*input_dim=*/4, /*d_model=*/8, /*output_dim=*/2,
                      /*seq_len=*/5, /*num_blocks=*/2, /*num_heads=*/1,  // H=1 for v1 backward
                      /*ffn_hidden=*/16);
    Tensor x = Tensor::random(4, 5, 0.2);
    Tensor target(2, 1);
    target.fill(0.0);

    double lr = 0.01;
    double initial_loss = 0.0;
    for (int step = 0; step < 80; ++step) {
        m.zero_grad();
        Tensor y = m.forward(x);
        Tensor dy(2, 1);
        double loss = 0.0;
        for (size_t i = 0; i < 2; ++i) {
            double diff = y[i][0] - target[i][0];
            loss += diff * diff;
            dy[i][0] = 2.0 * diff / 2.0;
        }
        loss /= 2.0;
        if (step == 0) initial_loss = loss;
        m.backward(dy, 0.0);
        m.update_weights(lr);
    }
    m.zero_grad();
    Tensor y_final = m.forward(x);
    double final_loss = 0.0;
    for (size_t i = 0; i < 2; ++i) {
        double diff = y_final[i][0] - target[i][0];
        final_loss += diff * diff;
    }
    final_loss /= 2.0;
    CHECK(final_loss < initial_loss, "model training reduces loss");
    std::cout << "    initial_loss = " << initial_loss << ", final_loss = " << final_loss << "\n";
}


// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "=== FastFormer Tests ===\n";
    test_constructor_validation();
    test_forward_shape();
    test_forward_nonzero();
    test_causal_mask();
    test_input_fd_grad();
    test_param_fd_grad_Wq();
    test_param_fd_grad_more();
    test_params_gradients_contract();
    test_zero_grad();
    test_determinism();
    test_block_forward_shape();
    test_block_training();
    test_model_forward_shape();
    test_model_training();

    std::cout << "=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
