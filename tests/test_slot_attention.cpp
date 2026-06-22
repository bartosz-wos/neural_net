// Slot Attention — Locatello et al. 2020
//   "Object-Centric Learning with Slot Attention"
//   (https://arxiv.org/abs/2006.15055)
//
// Tests:
//   1. SlotAttention forward shape (N=3 inputs, D=2 features, K=2 slots)
//   2. SlotAttention output finite
//   3. SlotAttention slot count invariants (K fixed regardless of N)
//   4. SlotAttention permutation invariance to input order
//   5. SlotAttention input gradient check (rel_err < 1e-2)
//   6. SlotAttention W_k gradient check
//   7. SlotAttention W_v gradient check
//   8. SlotAttention W_q gradient check
//   9. SlotAttention mu (slot init) gradient check
//  10. SlotAttention training step reduces loss
//  11. SlotAttention parameters/gradients shape consistency
//  12. SlotAttention different num_iterations (T=2, T=4)
//  13. SlotAttentionBlock forward shape
//  14. SlotAttentionBlock input gradient check
//  15. SlotAttentionBlock training step reduces loss
//  16. SlotAttentionModel forward shape (N=3 -> K=2 -> 4-class)
//  17. SlotAttentionModel training step reduces loss

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nn/layers/attention/slot_attention.h"

using namespace std;

static double relative_error(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
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

// Helper: find parameter by shape signature (rows, cols).
struct ParamMatch {
    Tensor* p = nullptr;
    Tensor* g = nullptr;
    int seen = 0;
};
static ParamMatch find_param(vector<Tensor*>& params, vector<Tensor*>& grads,
                             size_t r, size_t c, int occurrence = 0) {
    ParamMatch pm;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == r && params[i]->cols == c) {
            if (pm.seen == occurrence) {
                pm.p = params[i];
                pm.g = grads[i];
                return pm;
            }
            pm.seen++;
        }
    }
    return pm;
}

int main() {
    cout << "=== Slot Attention (Locatello 2020) Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // Small tractable config: N=3 inputs, D=2 features, K=2 slots, T=2 iterations
    size_t N = 3, D = 2, K = 2, T = 2;

    // ------------------------------------------------------------
    // Test 1: forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 1: SlotAttention forward shape (N=3, D=2, K=2, T=2) ---\n";
    {
        ++total;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.1 * i - 0.05 * j + 0.1;

        SlotAttention attn(K, D, D, T);
        Tensor output = attn.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == K && output.cols == D) {
            cout << "[PASS] forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << K << "x" << D << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: output finite
    // ------------------------------------------------------------
    cout << "\n--- Test 2: SlotAttention output finite (N=4, D=2, K=3, T=2) ---\n";
    {
        ++total;
        size_t N2 = 4, D2 = 2, K2 = 3;
        Tensor input(N2, D2);
        for (size_t i = 0; i < N2; ++i)
            for (size_t j = 0; j < D2; ++j)
                input(i, j) = 0.3 * sin(0.5 * i) - 0.2 * j;

        SlotAttention attn(K2, D2, D2, 2);
        Tensor output = attn.forward(input);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ------------------------------------------------------------
    // Test 3: slot count invariant to N
    // ------------------------------------------------------------
    cout << "\n--- Test 3: SlotAttention output is always K slots regardless of N ---\n";
    {
        ++total;
        size_t K3 = 4;
        bool ok = true;
        for (size_t N_test : {size_t(2), size_t(5), size_t(8)}) {
            Tensor input(N_test, D);
            for (size_t i = 0; i < N_test; ++i)
                for (size_t j = 0; j < D; ++j)
                    input(i, j) = 0.1 * i + 0.05 * j;
            SlotAttention attn(K3, D, D, T);
            Tensor output = attn.forward(input);
            if (output.rows != K3) { ok = false; break; }
        }
        if (ok) {
            cout << "[PASS] output always K=" << K3 << " rows\n";
            ++passed;
        } else {
            cout << "[FAIL] output rows != K for some N\n";
        }
    }

    // ------------------------------------------------------------
    // Test 4: permutation invariance to input order
    // ------------------------------------------------------------
    cout << "\n--- Test 4: SlotAttention permutation-invariant to input order ---\n";
    {
        ++total;
        // Permuting input rows should permute attn weights but not change
        // the SET of slot values (since slots compete for inputs and the
        // double-softmax is invariant up to slot permutation).
        // We check: applying a row-permutation to inputs and re-running forward
        // should produce the SAME slot SET (slots can be permuted but the set
        // of slot values must match). For determinism with a fixed seed, we
        // expect the actual slot values to be permuted by the same permutation.
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.5 * i - 0.3 * j;

        // Permute: swap rows 0 and 2
        Tensor input_perm(N, D);
        input_perm(0, 0) = input(2, 0); input_perm(0, 1) = input(2, 1);
        input_perm(1, 0) = input(1, 0); input_perm(1, 1) = input(1, 1);
        input_perm(2, 0) = input(0, 0); input_perm(2, 1) = input(0, 1);

        SlotAttention attn(K, D, D, T);
        Tensor out1 = attn.forward(input);
        Tensor out2 = attn.forward(input_perm);

        // The slots should be the same SET (allowing any permutation of the K rows).
        // Sort each output by L2 norm of each slot and compare.
        vector<pair<double, size_t>> norm_id1, norm_id2;
        for (size_t k = 0; k < K; ++k) {
            double n1 = 0.0, n2 = 0.0;
            for (size_t j = 0; j < D; ++j) { n1 += out1(k, j) * out1(k, j); n2 += out2(k, j) * out2(k, j); }
            norm_id1.push_back({sqrt(n1), k});
            norm_id2.push_back({sqrt(n2), k});
        }
        sort(norm_id1.begin(), norm_id1.end());
        sort(norm_id2.begin(), norm_id2.end());

        double max_diff = 0.0;
        for (size_t k = 0; k < K; ++k) {
            size_t i1 = norm_id1[k].second, i2 = norm_id2[k].second;
            for (size_t j = 0; j < D; ++j) {
                max_diff = max(max_diff, fabs(out1(i1, j) - out2(i2, j)));
            }
        }
        if (max_diff < 1e-9) {
            cout << "[PASS] permutation invariance holds (max_diff=" << max_diff << ")\n";
            ++passed;
        } else {
            cout << "[FAIL] permutation invariance violated (max_diff=" << max_diff << ")\n";
        }
    }

    // ------------------------------------------------------------
    // Test 5: input gradient check (numerical vs analytical)
    // ------------------------------------------------------------
    cout << "\n--- Test 5: SlotAttention input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.5 * (i + 1) - 0.3 * (j + 1);

        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.2 * i - 0.1 * j + 1.0;

        SlotAttention attn(K, D, D, T);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        Tensor grad_x = attn.backward(grad_loss, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                double orig = input(i, j);
                input(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                input(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                input(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                double err = relative_error(num, ana);
                max_err = max(max_err, err);
            }
        }
        cout << "  max relative error = " << max_err << "\n";
        if (max_err < 1e-2) {
            cout << "[PASS] input gradient check (max_err < 1e-2)\n";
            ++passed;
        } else {
            cout << "[FAIL] input gradient check too loose\n";
        }
    }

    // ------------------------------------------------------------
    // Test 6: W_k gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 6: SlotAttention W_k gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.4 * i + 0.2 * j - 0.1;
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.1 * i - 0.3 * j + 0.5;

        SlotAttention attn(K, D, D, T);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        auto pm = find_param(params, grads, D, D, 0);  // first (D, D) is W_k
        if (!pm.p || !pm.g) {
            cout << "[FAIL] could not find W_k\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < D; ++i) for (size_t j = 0; j < D; ++j) {
                double orig = (*pm.p)(i, j);
                (*pm.p)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*pm.p)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*pm.p)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*pm.g)(i, j);
                max_err = max(max_err, relative_error(num, ana));
            }
            cout << "  max relative error = " << max_err << "\n";
            if (max_err < 1e-2) { cout << "[PASS] W_k gradient check\n"; ++passed; }
            else { cout << "[FAIL] W_k gradient check too loose\n"; }
        }
    }

    // ------------------------------------------------------------
    // Test 7: W_v gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 7: SlotAttention W_v gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.4 * i + 0.2 * j - 0.1;
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.1 * i - 0.3 * j + 0.5;

        SlotAttention attn(K, D, D, T);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        // params order: W_k, b_k, W_v, b_v, W_q, b_q, mu, ln_k.gamma, ln_k.beta,
        //               ln_v.gamma, ln_v.beta, ln_q.gamma, ln_q.beta,
        //               ln_mlp.gamma, ln_mlp.beta, mlp_fc1.weights, mlp_fc1.bias,
        //               mlp_fc2.weights, mlp_fc2.bias, W_zr, b_zr, W_h, b_h
        // W_v is (D, D) — second (D, D) in param list
        auto pm = find_param(params, grads, D, D, 1);
        if (!pm.p || !pm.g) {
            cout << "[FAIL] could not find W_v\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < D; ++i) for (size_t j = 0; j < D; ++j) {
                double orig = (*pm.p)(i, j);
                (*pm.p)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*pm.p)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*pm.p)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*pm.g)(i, j);
                max_err = max(max_err, relative_error(num, ana));
            }
            cout << "  max relative error = " << max_err << "\n";
            if (max_err < 1e-2) { cout << "[PASS] W_v gradient check\n"; ++passed; }
            else { cout << "[FAIL] W_v gradient check too loose\n"; }
        }
    }

    // ------------------------------------------------------------
    // Test 8: W_q gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 8: SlotAttention W_q gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.4 * i + 0.2 * j - 0.1;
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.1 * i - 0.3 * j + 0.5;

        SlotAttention attn(K, D, D, T);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        // W_q is third (D, D) in param list
        auto pm = find_param(params, grads, D, D, 2);
        if (!pm.p || !pm.g) {
            cout << "[FAIL] could not find W_q\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < D; ++i) for (size_t j = 0; j < D; ++j) {
                double orig = (*pm.p)(i, j);
                (*pm.p)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*pm.p)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*pm.p)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*pm.g)(i, j);
                max_err = max(max_err, relative_error(num, ana));
            }
            cout << "  max relative error = " << max_err << "\n";
            if (max_err < 1e-2) { cout << "[PASS] W_q gradient check\n"; ++passed; }
            else { cout << "[FAIL] W_q gradient check too loose\n"; }
        }
    }

    // ------------------------------------------------------------
    // Test 9: mu (slot init) gradient check
    // ------------------------------------------------------------
    cout << "\n--- Test 9: SlotAttention mu gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.4 * i + 0.2 * j - 0.1;
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.1 * i - 0.3 * j + 0.5;

        SlotAttention attn(K, D, D, T);
        Tensor out = attn.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        attn.zero_grad();
        attn.backward(grad_loss, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        // mu is (K, D) — first (K, D) param
        auto pm = find_param(params, grads, K, D, 0);
        if (!pm.p || !pm.g) {
            cout << "[FAIL] could not find mu\n";
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < K; ++i) for (size_t j = 0; j < D; ++j) {
                double orig = (*pm.p)(i, j);
                (*pm.p)(i, j) = orig + eps;
                Tensor out_p = attn.forward(input);
                double Lp = l2_loss_value(out_p, target);
                (*pm.p)(i, j) = orig - eps;
                Tensor out_m = attn.forward(input);
                double Lm = l2_loss_value(out_m, target);
                (*pm.p)(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*pm.g)(i, j);
                max_err = max(max_err, relative_error(num, ana));
            }
            cout << "  max relative error = " << max_err << "\n";
            if (max_err < 1e-2) { cout << "[PASS] mu gradient check\n"; ++passed; }
            else { cout << "[FAIL] mu gradient check too loose\n"; }
        }
    }

    // ------------------------------------------------------------
    // Test 10: training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 10: SlotAttention training step reduces loss ---\n";
    {
        ++total;
        Tensor input(N, D);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                input(i, j) = 0.3 * sin(i + j);
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        SlotAttention attn(K, D, D, T);
        double lr = 0.01;

        Tensor out0 = attn.forward(input);
        double L0 = l2_loss_value(out0, target);

        for (size_t step = 0; step < 30; ++step) {
            attn.zero_grad();
            Tensor out = attn.forward(input);
            Tensor grad = l2_loss_grad(out, target);
            attn.backward(grad, lr);
            attn.update_weights(lr);
        }
        Tensor out1 = attn.forward(input);
        double L1 = l2_loss_value(out1, target);
        cout << "  L0 = " << L0 << "  L1 = " << L1
             << "  reduction = " << (L0 > 0 ? (L0 - L1) / L0 * 100.0 : 0.0) << "%\n";
        if (L1 < L0) {
            cout << "[PASS] training reduced loss\n";
            ++passed;
        } else {
            cout << "[FAIL] training did not reduce loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 11: parameters/gradients shape consistency
    // ------------------------------------------------------------
    cout << "\n--- Test 11: SlotAttention parameters and gradients shape consistency ---\n";
    {
        ++total;
        SlotAttention attn(K, D, D, T);
        Tensor input(N, D);
        Tensor target(K, D);
        Tensor out = attn.forward(input);
        Tensor grad = l2_loss_grad(out, target);
        attn.backward(grad, 0.0);

        auto params = attn.parameters();
        auto grads = attn.gradients();
        bool ok = (params.size() == grads.size());
        if (ok) {
            for (size_t i = 0; i < params.size(); ++i) {
                if (params[i]->rows != grads[i]->rows ||
                    params[i]->cols != grads[i]->cols) { ok = false; break; }
            }
        }
        cout << "  num parameters = " << params.size() << "\n";
        if (ok) {
            cout << "[PASS] param/grad shapes consistent\n";
            ++passed;
        } else {
            cout << "[FAIL] param/grad shape mismatch\n";
        }
    }

    // ------------------------------------------------------------
    // Test 12: different num_iterations work
    // ------------------------------------------------------------
    cout << "\n--- Test 12: SlotAttention works with different num_iterations ---\n";
    {
        ++total;
        bool ok = true;
        for (size_t T_test : {size_t(1), size_t(2), size_t(4)}) {
            Tensor input(N, D);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < D; ++j)
                    input(i, j) = 0.1 * i + 0.05 * j;
            SlotAttention attn(K, D, D, T_test);
            Tensor out = attn.forward(input);
            if (out.rows != K || out.cols != D) { ok = false; break; }
            bool finite = true;
            for (size_t i = 0; i < out.rows && finite; ++i)
                for (size_t j = 0; j < out.cols; ++j)
                    if (!std::isfinite(out(i, j))) finite = false;
            if (!finite) { ok = false; break; }
        }
        if (ok) {
            cout << "[PASS] T=1,2,4 all work\n";
            ++passed;
        } else {
            cout << "[FAIL] some T value failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 13: SlotAttentionBlock forward shape (operates on slots)
    // ------------------------------------------------------------
    cout << "\n--- Test 13: SlotAttentionBlock forward shape (slots: K=2, D=2) ---\n";
    {
        ++total;
        // SlotAttentionBlock takes slots (K, D) and refines them → (K, D).
        SlotAttentionBlock blk(K, D, D, T);
        Tensor slots(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                slots(i, j) = 0.2 * i - 0.1 * j;
        Tensor out = blk.forward(slots);
        cout << "  Input: " << slots.rows << "x" << slots.cols
             << "  Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == K && out.cols == D) {
            cout << "[PASS] block forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << K << "x" << D << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 14: SlotAttentionBlock input gradient check (slots)
    // ------------------------------------------------------------
    cout << "\n--- Test 14: SlotAttentionBlock input gradient check ---\n";
    {
        ++total;
        double eps = 1e-5;
        Tensor slots(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                slots(i, j) = 0.3 * i + 0.1 * j - 0.05;
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.2 * i - 0.2 * j;

        SlotAttentionBlock blk(K, D, D, T);
        Tensor out = blk.forward(slots);
        Tensor grad = l2_loss_grad(out, target);
        blk.zero_grad();
        Tensor grad_x = blk.backward(grad, 0.0);

        double max_err = 0.0;
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                double orig = slots(i, j);
                slots(i, j) = orig + eps;
                Tensor out_p = blk.forward(slots);
                double Lp = l2_loss_value(out_p, target);
                slots(i, j) = orig - eps;
                Tensor out_m = blk.forward(slots);
                double Lm = l2_loss_value(out_m, target);
                slots(i, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = grad_x(i, j);
                max_err = max(max_err, relative_error(num, ana));
            }
        }
        cout << "  max relative error = " << max_err << "\n";
        if (max_err < 1e-2) {
            cout << "[PASS] block input gradient check\n";
            ++passed;
        } else {
            cout << "[FAIL] block input gradient too loose\n";
        }
    }

    // ------------------------------------------------------------
    // Test 15: SlotAttentionBlock training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 15: SlotAttentionBlock training step reduces loss ---\n";
    {
        ++total;
        Tensor slots(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                slots(i, j) = 0.3 * sin(i + j);
        Tensor target(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                target(i, j) = 0.2 * i - 0.1 * j;

        SlotAttentionBlock blk(K, D, D, T);
        double lr = 0.01;

        Tensor out0 = blk.forward(slots);
        double L0 = l2_loss_value(out0, target);

        for (size_t step = 0; step < 30; ++step) {
            blk.zero_grad();
            Tensor out = blk.forward(slots);
            Tensor grad = l2_loss_grad(out, target);
            blk.backward(grad, lr);
            blk.update_weights(lr);
        }
        Tensor out1 = blk.forward(slots);
        double L1 = l2_loss_value(out1, target);
        cout << "  L0 = " << L0 << "  L1 = " << L1
             << "  reduction = " << (L0 > 0 ? (L0 - L1) / L0 * 100.0 : 0.0) << "%\n";
        if (L1 < L0) {
            cout << "[PASS] block training reduced loss\n";
            ++passed;
        } else {
            cout << "[FAIL] block training did not reduce loss\n";
        }
    }

    // ------------------------------------------------------------
    // Test 16: SlotAttentionModel forward shape
    // ------------------------------------------------------------
    cout << "\n--- Test 16: SlotAttentionModel forward shape (N=3, D_in=3, K=2, D_out=4) ---\n";
    {
        ++total;
        size_t in_dim = 3, out_dim = 4;
        SlotAttentionModel model(K, D, in_dim, out_dim, 1, T);
        Tensor input(N, in_dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                input(i, j) = 0.2 * i - 0.1 * j;
        Tensor out = model.forward(input);
        cout << "  Input: " << input.rows << "x" << input.cols
             << "  Output: " << out.rows << "x" << out.cols << "\n";
        if (out.rows == K && out.cols == (long)out_dim) {
            cout << "[PASS] model forward shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << K << "x" << out_dim << "\n";
        }
    }

    // ------------------------------------------------------------
    // Test 17: SlotAttentionModel training step reduces loss
    // ------------------------------------------------------------
    cout << "\n--- Test 17: SlotAttentionModel training step reduces loss ---\n";
    {
        ++total;
        size_t in_dim = 3, out_dim = 4;
        SlotAttentionModel model(K, D, in_dim, out_dim, 1, T);
        double lr = 0.005;

        Tensor input(N, in_dim);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                input(i, j) = 0.3 * sin(i + j);
        Tensor target(K, out_dim);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < out_dim; ++j)
                target(i, j) = 0.1 * i - 0.2 * j;

        Tensor out0 = model.forward(input);
        double L0 = l2_loss_value(out0, target);

        for (size_t step = 0; step < 30; ++step) {
            model.zero_grad();
            Tensor out = model.forward(input);
            Tensor grad = l2_loss_grad(out, target);
            model.backward(grad, lr);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward(input);
        double L1 = l2_loss_value(out1, target);
        cout << "  L0 = " << L0 << "  L1 = " << L1
             << "  reduction = " << (L0 > 0 ? (L0 - L1) / L0 * 100.0 : 0.0) << "%\n";
        if (L1 < L0) {
            cout << "[PASS] model training reduced loss\n";
            ++passed;
        } else {
            cout << "[FAIL] model training did not reduce loss\n";
        }
    }

    cout << "\n=== Results: " << passed << " / " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
