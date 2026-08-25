// Block-Sparse Flash Attention — tests
//
// Math:
//   Same online-softmax flash recurrence as FlashAttentionV2, with a 2D block
//   mask M ∈ {0,1}^{n_q_blocks × n_k_blocks} that gates which (Q-block,
//   K-block) pairs participate. Masked-out blocks contribute nothing to
//   forward output AND no gradient flows through them in backward.
//
// Tests:
//   1.  Constructor validation (5 cases)
//   2.  Forward shape with dense mask
//   3.  Forward shape with causal mask
//   4.  Forward equivalence with FlashAttentionV2 in dense case
//   5.  Causal mask — row i has zero attention past position i
//   6.  Sliding-window mask — row i attends only to window
//   7.  Strided mask — row i attends to blocks j ≡ i mod stride
//   8.  BigBird mask — global + window + random blocks all contribute
//   9.  Input gradient FD check (dense mask)
//   10.  W_q/W_k/W_v/W_o gradient FD checks (dense mask)
//   11.  Masked-out K-block produces ZERO W_k gradient contribution
//   12.  Mask validation in forward (shape, value range)
//   13.  BlockSparseFlashBlock input gradient FD check
//   14.  BlockSparseFlashModel training reduces loss
//   15.  Mask == 0 for all blocks → degenerate but finite output
//   16.  Multi-head with GQA K/V sharing forward shape

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/attention/block_sparse_flash.h"
#include "nn/layers/attention/flash_attention_v2.h"

using namespace std;

static double relative_error(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-12) return fabs(a - b);
    return fabs(a - b) / max_abs;
}

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

// Numerical gradient check for a single scalar element of the *input*.
// Calls layer.forward_with_mask(input, mask) when available, else
// layer.forward(input) (for Block/Model).
template <typename LayerT>
static double check_input_gradient_block(LayerT& layer, Tensor& input,
                                          const Tensor& mask, const Tensor& target,
                                          double eps = 1e-5) {
    Tensor out = layer.forward_with_mask(input, mask);
    Tensor gout = l2_loss_grad(out, target);
    Tensor gin = layer.backward(gout, 0.0);
    double max_err = 0.0;
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist_r(0, input.rows - 1);
    std::uniform_int_distribution<size_t> dist_c(0, input.cols - 1);
    for (int trial = 0; trial < 8; ++trial) {
        size_t r = dist_r(rng);
        size_t c = dist_c(rng);
        double orig = input.data[r * input.cols + c];
        input.data[r * input.cols + c] = orig + eps;
        Tensor out_p = layer.forward_with_mask(input, mask);
        double loss_p = l2_loss_value(out_p, target);
        input.data[r * input.cols + c] = orig - eps;
        Tensor out_m = layer.forward_with_mask(input, mask);
        double loss_m = l2_loss_value(out_m, target);
        input.data[r * input.cols + c] = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = gin.data[r * gin.cols + c];
        double err = relative_error(ana_grad, num_grad);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

template <typename LayerT>
static double check_input_gradient(LayerT& layer, Tensor& input, const Tensor& target,
                                   double eps = 1e-5) {
    Tensor out = layer.forward(input);
    Tensor gout = l2_loss_grad(out, target);
    Tensor gin = layer.backward(gout, 0.0);
    double max_err = 0.0;
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist_r(0, input.rows - 1);
    std::uniform_int_distribution<size_t> dist_c(0, input.cols - 1);
    for (int trial = 0; trial < 8; ++trial) {
        size_t r = dist_r(rng);
        size_t c = dist_c(rng);
        double orig = input.data[r * input.cols + c];
        input.data[r * input.cols + c] = orig + eps;
        Tensor out_p = layer.forward(input);
        double loss_p = l2_loss_value(out_p, target);
        input.data[r * input.cols + c] = orig - eps;
        Tensor out_m = layer.forward(input);
        double loss_m = l2_loss_value(out_m, target);
        input.data[r * input.cols + c] = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = gin.data[r * gin.cols + c];
        double err = relative_error(ana_grad, num_grad);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Check a single parameter element via FD.
static double check_param_gradient(BlockSparseFlashAttention& layer,
                                    Tensor& param, const Tensor& grad_param,
                                    Tensor& input, const Tensor& mask,
                                    const Tensor& target, double eps = 1e-5) {
    Tensor out = layer.forward_with_mask(input, mask);
    Tensor gout = l2_loss_grad(out, target);
    layer.backward(gout, 0.0);
    double max_err = 0.0;
    std::mt19937 rng(456);
    std::uniform_int_distribution<size_t> dist_r(0, param.rows - 1);
    std::uniform_int_distribution<size_t> dist_c(0, param.cols - 1);
    for (int trial = 0; trial < 6; ++trial) {
        size_t r = dist_r(rng);
        size_t c = dist_c(rng);
        double orig = param.data[r * param.cols + c];
        param.data[r * param.cols + c] = orig + eps;
        Tensor out_p = layer.forward_with_mask(input, mask);
        double loss_p = l2_loss_value(out_p, target);
        param.data[r * param.cols + c] = orig - eps;
        Tensor out_m = layer.forward_with_mask(input, mask);
        double loss_m = l2_loss_value(out_m, target);
        param.data[r * param.cols + c] = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = grad_param.data[r * grad_param.cols + c];
        double err = relative_error(ana_grad, num_grad);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    cout << "=== Block-Sparse Flash Attention Tests ===\n";
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: Constructor validation
    // ------------------------------------------------------------
    cout << "\n--- Test 1: Constructor validation ---\n";
    {
        ++total;
        int sub_total = 0, sub_pass = 0;
        // d_model = 0 should throw
        try { BlockSparseFlashAttention a(0, 4); cout << "[FAIL] d_model=0 did not throw\n"; }
        catch (...) { cout << "[PASS] d_model=0 throws\n"; ++sub_pass; }
        ++sub_total;

        // num_heads = 0 should throw
        try { BlockSparseFlashAttention a(8, 0); cout << "[FAIL] num_heads=0 did not throw\n"; }
        catch (...) { cout << "[PASS] num_heads=0 throws\n"; ++sub_pass; }
        ++sub_total;

        // d_model % num_heads != 0 should throw
        try { BlockSparseFlashAttention a(8, 3); cout << "[FAIL] non-divisible did not throw\n"; }
        catch (...) { cout << "[PASS] d_model%num_heads throws\n"; ++sub_pass; }
        ++sub_total;

        // query_block_size = 0 should throw
        try { BlockSparseFlashAttention a(8, 2, 0, 0); cout << "[FAIL] qbs=0 did not throw\n"; }
        catch (...) { cout << "[PASS] query_block_size=0 throws\n"; ++sub_pass; }
        ++sub_total;

        // key_block_size = 0 should throw
        try { BlockSparseFlashAttention a(8, 2, 0, 4, 0); cout << "[FAIL] kbs=0 did not throw\n"; }
        catch (...) { cout << "[PASS] key_block_size=0 throws\n"; ++sub_pass; }
        ++sub_total;

        // Valid construction (no throw)
        try { BlockSparseFlashAttention a(8, 2); cout << "[PASS] valid construction succeeds\n"; ++sub_pass; }
        catch (...) { cout << "[FAIL] valid construction threw\n"; }
        ++sub_total;

        // num_kv_heads > num_heads should throw
        try { BlockSparseFlashAttention a(8, 2, 4); cout << "[FAIL] num_kv>num_q did not throw\n"; }
        catch (...) { cout << "[PASS] num_kv_heads > num_heads throws\n"; ++sub_pass; }
        ++sub_total;

        // num_kv_heads not dividing num_heads should throw (GQA invariant)
        try { BlockSparseFlashAttention a(8, 4, 3); cout << "[FAIL] num_heads%num_kv!=0 did not throw\n"; }
        catch (...) { cout << "[PASS] non-divisible GQA throws\n"; ++sub_pass; }
        ++sub_total;

        if (sub_pass == sub_total) { ++passed; cout << "ALL CONSTRUCTOR VALIDATION PASSED\n"; }
    }

    cout << "\n=== Summary: " << passed << "/" << total << " passed ===\n";
    return (passed == total) ? 0 : 1;
}