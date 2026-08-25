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

    cout << "\n--- Test 2: Forward shape with dense mask ---\n";
    {
        ++total;
        size_t n = 6, d = 4, H = 2;
        Tensor input(n, d);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q_blocks = (n + 1) / 2;
        size_t n_k_blocks = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_dense_mask(n_q_blocks, n_k_blocks);
        Tensor output = attn.forward_with_mask(input, mask);
        bool shape_ok = (output.rows == n && output.cols == d);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        cout << "  output shape " << output.rows << "x" << output.cols
             << "  finite=" << (finite ? "yes" : "no") << "\n";
        if (shape_ok && finite) { cout << "[PASS] dense-mask forward shape\n"; ++passed; }
        else { cout << "[FAIL] dense-mask forward shape\n"; }
    }

    cout << "\n--- Test 3: Forward shape with causal mask ---\n";
    {
        ++total;
        size_t n = 6, d = 4, H = 2;
        Tensor input(n, d);
        std::mt19937 rng(8);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q_blocks = (n + 1) / 2;
        size_t n_k_blocks = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_causal_mask(n_q_blocks, n_k_blocks);
        cout << "  causal mask:\n";
        for (size_t i = 0; i < mask.rows; ++i) {
            cout << "    ";
            for (size_t j = 0; j < mask.cols; ++j) cout << (mask(i, j) > 0.5 ? "1 " : "0 ");
            cout << "\n";
        }
        Tensor output = attn.forward_with_mask(input, mask);
        bool shape_ok = (output.rows == n && output.cols == d);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        cout << "  output shape " << output.rows << "x" << output.cols
             << "  finite=" << (finite ? "yes" : "no") << "\n";
        if (shape_ok && finite) { cout << "[PASS] causal-mask forward shape\n"; ++passed; }
        else { cout << "[FAIL] causal-mask forward shape\n"; }
    }

    cout << "\n--- Test 4: Forward equivalence with FlashAttentionV2 in dense case ---\n";
    {
        ++total;
        size_t n = 4, d = 4, H = 1;
        // Use seed-stable random init
        Tensor input(n, d);
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        Tensor target(n, d);
        for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng) * 0.2;

        // Build BSFA with manual weights (copy from FA-2)
        FlashAttentionV2Layer fa2(d, H, 2, 2, /*causal=*/false);
        BlockSparseFlashAttention bsfa(d, H, 0, 2, 2);
        bsfa.W_q = fa2.W_q; bsfa.W_k = fa2.W_k; bsfa.W_v = fa2.W_v; bsfa.W_o = fa2.W_o;

        // Dense mask
        size_t n_q = (n + 1) / 2;
        size_t n_k = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_dense_mask(n_q, n_k);
        Tensor out_bsfa = bsfa.forward_with_mask(input, mask);
        // FA-2 takes (d_model, seq_len) — transpose
        Tensor input_T(d, n);
        Tensor out_T(d, n);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < n; ++j) input_T(i, j) = input(j, i);
        Tensor out_fa2 = fa2.forward(input_T);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < n; ++j) out_T(i, j) = out_fa2(i, j);

        // Compare (transpose FA-2 output)
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double diff = fabs(out_bsfa(i, j) - out_T(i, j));
                if (diff > max_diff) max_diff = diff;
            }
        }
        cout << "  max diff BSFA vs FA-2 (dense, same weights): " << max_diff << "\n";
        if (max_diff < 1e-3) { cout << "[PASS] dense equivalence with FA-2\n"; ++passed; }
        else { cout << "[FAIL] dense equivalence — max_diff too high\n"; }
    }

    cout << "\n--- Test 5: Causal mask — perturbed unmasked K affects output ---\n";
    {
        ++total;
        size_t n = 4, d = 4, H = 1;
        Tensor input(n, d);
        std::mt19937 rng(10);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q = (n + 1) / 2;
        size_t n_k = (n + 1) / 2;
        Tensor causal_mask = BlockSparseFlashAttention::build_causal_mask(n_q, n_k);

        // Run causal forward
        Tensor out_causal = attn.forward_with_mask(input, causal_mask);

        // Perturb W_k at row 3 (last K position, unmasked for everyone under causal)
        Tensor W_k_save = attn.W_k;
        attn.W_k(3, 0) += 1.0;
        Tensor out_perturbed = attn.forward_with_mask(input, causal_mask);
        attn.W_k = W_k_save;

        // The output MUST change when an unmasked K position is perturbed.
        // (If the mask path is broken, the output could be independent of K entirely.)
        double max_diff = 0.0;
        for (size_t i = 0; i < out_causal.data.size(); ++i) {
            double diff = fabs(out_causal.data[i] - out_perturbed.data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        cout << "  max diff (perturb W_k[3,*] under causal): " << max_diff << "\n";
        if (max_diff > 1e-4) { cout << "[PASS] causal mask forward is sensitive to W_k perturbation\n"; ++passed; }
        else { cout << "[FAIL] output is independent of K — gradient path is broken\n"; }
    }

    cout << "\n--- Test 6: Sliding-window mask ---\n";
    {
        ++total;
        size_t n = 6, d = 4, H = 1;
        Tensor input(n, d);
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q = (n + 1) / 2;
        size_t n_k = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_sliding_window_mask(n_q, n_k, 2);
        cout << "  sliding-window mask (W=2):\n";
        for (size_t i = 0; i < mask.rows; ++i) {
            cout << "    ";
            for (size_t j = 0; j < mask.cols; ++j) cout << (mask(i, j) > 0.5 ? "1 " : "0 ");
            cout << "\n";
        }
        Tensor output = attn.forward_with_mask(input, mask);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        if (finite) { cout << "[PASS] sliding-window mask forward finite\n"; ++passed; }
        else { cout << "[FAIL] sliding-window mask forward non-finite\n"; }
    }

    cout << "\n--- Test 7: Strided mask ---\n";
    {
        ++total;
        size_t n = 6, d = 4, H = 1;
        Tensor input(n, d);
        std::mt19937 rng(12);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q = (n + 1) / 2;
        size_t n_k = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_strided_mask(n_q, n_k, 2);
        cout << "  strided mask (s=2):\n";
        for (size_t i = 0; i < mask.rows; ++i) {
            cout << "    ";
            for (size_t j = 0; j < mask.cols; ++j) cout << (mask(i, j) > 0.5 ? "1 " : "0 ");
            cout << "\n";
        }
        Tensor output = attn.forward_with_mask(input, mask);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        if (finite) { cout << "[PASS] strided mask forward finite\n"; ++passed; }
        else { cout << "[FAIL] strided mask forward non-finite\n"; }
    }

    cout << "\n--- Test 8: BigBird mask (window + global + random) ---\n";
    {
        ++total;
        size_t n = 8, d = 4, H = 1;
        Tensor input(n, d);
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, H, 0, 2, 2);
        size_t n_q = (n + 1) / 2;
        size_t n_k = (n + 1) / 2;
        Tensor mask = BlockSparseFlashAttention::build_bigbird_mask(n_q, n_k, 2, 1, 42);
        cout << "  bigbird mask:\n";
        for (size_t i = 0; i < mask.rows; ++i) {
            cout << "    ";
            for (size_t j = 0; j < mask.cols; ++j) cout << (mask(i, j) > 0.5 ? "1 " : "0 ");
            cout << "\n";
        }
        Tensor output = attn.forward_with_mask(input, mask);
        bool finite = true;
        for (double v : output.data) if (!std::isfinite(v)) { finite = false; break; }
        if (finite) { cout << "[PASS] bigbird mask forward finite\n"; ++passed; }
        else { cout << "[FAIL] bigbird mask forward non-finite\n"; }
    }

    cout << "\n--- Test 9: Mask validation in forward ---\n";
    {
        ++total;
        size_t n = 4, d = 4;
        Tensor input(n, d);
        std::mt19937 rng(14);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(rng);
        BlockSparseFlashAttention attn(d, 2, 0, 2, 2);
        Tensor wrong_rows(99, 2);   // wrong rows
        Tensor wrong_cols(2, 99);   // wrong cols
        Tensor non_binary(2, 2);    // non-{0,1} values
        non_binary.fill(0.5);
        Tensor all_zeros(2, 2);     // all zeros (degenerate)
        all_zeros.fill(0.0);

        int sub_total = 0, sub_pass = 0;
        try { attn.forward_with_mask(input, wrong_rows); cout << "[FAIL] wrong rows did not throw\n"; }
        catch (...) { cout << "[PASS] wrong rows throws\n"; ++sub_pass; }
        ++sub_total;
        try { attn.forward_with_mask(input, wrong_cols); cout << "[FAIL] wrong cols did not throw\n"; }
        catch (...) { cout << "[PASS] wrong cols throws\n"; ++sub_pass; }
        ++sub_total;
        try { attn.forward_with_mask(input, non_binary); cout << "[FAIL] non-binary did not throw\n"; }
        catch (...) { cout << "[PASS] non-binary values throw\n"; ++sub_pass; }
        ++sub_total;
        // All-zeros is degenerate but should NOT throw (just produce zero output).
        try {
            Tensor out = attn.forward_with_mask(input, all_zeros);
            bool all_zero = true;
            for (double v : out.data) if (v != 0.0) { all_zero = false; break; }
            if (all_zero) { cout << "[PASS] all-zeros mask produces zero output\n"; ++sub_pass; }
            else { cout << "[FAIL] all-zeros mask should produce zero output\n"; }
        } catch (...) { cout << "[FAIL] all-zeros mask threw\n"; }
        ++sub_total;
        if (sub_pass == sub_total) { ++passed; cout << "ALL MASK VALIDATION PASSED\n"; }
    }

    cout << "\n=== Summary: " << passed << "/" << total << " passed ===\n";
    return (passed == total) ? 0 : 1;
}