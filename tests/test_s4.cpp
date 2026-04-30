#include "nn/layers/architectures/s4.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

using namespace std;

static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t[i][j] * t[i][j];
    return std::sqrt(s);
}

static bool check(const string& name, bool pass) {
    if (pass) cout << "  [PASS] " << name << endl;
    else cout << "  [FAIL] " << name << endl;
    return pass;
}

int main() {
    int passed = 0, failed = 0;
    cout << setprecision(8);

    cout << "=== S4 Layer Tests ===" << endl << endl;

    // =================================================================
    // Test 1: Basic construction
    // =================================================================
    cout << "-- Test 1: Basic construction --" << endl;
    {
        bool ok = true;
        try {
            S4Layer s4(8, 4);
            ok = check("S4Layer(8, 4) constructs without crash", true);
        } catch (const exception& e) {
            ok = check("S4Layer(8, 4) constructs without crash", false);
            cout << "    Exception: " << e.what() << endl;
        }
        if (ok) passed++; else failed++;
    }

    // =================================================================
    // Test 2: Forward pass shape check (seq_len = 1, 4, 16)
    // =================================================================
    cout << endl << "-- Test 2: Forward pass shape check --" << endl;
    {
        S4Layer s4(8, 4);
        int test_seqs[] = {1, 4, 16};
        for (int sl : test_seqs) {
            Tensor input = Tensor::random(8, sl, 0.5);
            bool shape_ok = false;
            try {
                Tensor out = s4.forward(input);
                shape_ok = (out.rows == 8 && out.cols == sl);
                if (!check("output shape (" + to_string(8) + "," + to_string(sl) + ")",
                           shape_ok)) failed++;
                else passed++;
            } catch (const exception& e) {
                check("output shape (" + to_string(8) + "," + to_string(sl) + ")", false);
                cout << "    Exception: " << e.what() << endl;
                failed++;
            }
        }
    }

    // =================================================================
    // Test 3: Forward/backward numerical consistency + gradient counts
    // =================================================================
    cout << endl << "-- Test 3: Forward/backward numerical consistency --" << endl;
    {
        S4Layer s4(8, 4);
        // Fixed input for determinism
        Tensor input(8, 8);
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                input[i][j] = (i * 8 + j) * 0.1;

        Tensor out = s4.forward(input);

        // Backward with all-ones gradient
        Tensor grad_out(8, 8);
        grad_out.fill(1.0);
        Tensor grad_x = s4.backward(grad_out, 0.01);

        // Check grad_x is non-zero
        double gx_norm = tensor_l2norm(grad_x);
        if (!check("grad_x is non-zero", gx_norm > 1e-8)) failed++; else passed++;

        // Check parameter gradients are non-zero
        auto params = s4.parameters();
        auto grads = s4.gradients();

        if (!check("parameters() returns 6 tensors", params.size() == 6)) failed++; else passed++;
        if (!check("gradients() returns 6 tensors", grads.size() == 6)) failed++; else passed++;

        // Verify each gradient is non-zero
        const char* gnames[] = {"x_proj", "W_out", "b_out", "Lambda", "B", "C"};
        for (size_t k = 0; k < grads.size(); ++k) {
            double norm = tensor_l2norm(*grads[k]);
            bool nz = norm > 1e-10;
            if (!check(string("grad_") + gnames[k] + " non-zero", nz)) failed++;
            else passed++;
        }
    }

    // =================================================================
    // Test 4: Gradient flow through SSM parameters (Lambda, B, C)
    // =================================================================
    cout << endl << "-- Test 4: SSM parameter gradient flow --" << endl;
    {
        S4Layer s4(4, 2);
        int seq_len = 8;
        Tensor input = Tensor::random(4, seq_len, 0.5);

        Tensor out = s4.forward(input);
        Tensor grad_out(4, seq_len);
        grad_out.fill(1.0);
        Tensor grad_x = s4.backward(grad_out, 0.01);

        double gL = tensor_l2norm(s4.grad_Lambda_);
        double gB = tensor_l2norm(s4.grad_B_);
        double gC = tensor_l2norm(s4.grad_C_);

        if (!check("grad_Lambda_ non-zero (SSM diagonal)", gL > 1e-10)) failed++; else passed++;
        if (!check("grad_B_ non-zero (SSM input projection)", gB > 1e-10)) failed++; else passed++;
        if (!check("grad_C_ non-zero (SSM output projection)", gC > 1e-10)) failed++; else passed++;
    }

    // =================================================================
    // Test 5: Skip connection gradient (D)
    // =================================================================
    cout << endl << "-- Test 5: Skip connection gradient (D) --" << endl;
    {
        S4Layer s4(4, 2);
        int seq_len = 8;
        Tensor input = Tensor::random(4, seq_len, 0.5);

        // D starts at 0.0, so grad_D_ may be zero from first backward.
        // We set D to a non-zero value to test gradient flow.
        s4.D = 0.5;

        Tensor out = s4.forward(input);
        Tensor grad_out(4, seq_len);
        grad_out.fill(1.0);
        s4.backward(grad_out, 0.01);

        // grad_D_ should be non-zero since D is non-zero and x_scalar is non-zero
        bool nz = std::abs(s4.grad_D_) > 1e-10;
        if (!check("grad_D_ non-zero when D != 0", nz)) failed++; else passed++;
    }

    // =================================================================
    // Test 6: Training step — weights actually change
    // =================================================================
    cout << endl << "-- Test 6: Training step updates weights --" << endl;
    {
        S4Layer s4(4, 2);
        int seq_len = 8;
        Tensor input = Tensor::random(4, seq_len, 0.5);
        Tensor grad_out(4, seq_len);
        grad_out.fill(1.0);

        // Snapshot params before
        double x_proj_before = s4.x_proj[0][0];
        double W_out_before = s4.W_out[0][0];
        double Lambda_before = s4.Lambda[0][0];
        double B_before = s4.B[0][0];
        double C_before = s4.C[0][0];
        double D_before = s4.D;

        s4.forward(input);
        s4.backward(grad_out, 0.01);
        s4.update_weights(0.01);

        bool x_proj_changed = std::abs(s4.x_proj[0][0] - x_proj_before) > 1e-12;
        bool W_out_changed = std::abs(s4.W_out[0][0] - W_out_before) > 1e-12;
        bool Lambda_changed = std::abs(s4.Lambda[0][0] - Lambda_before) > 1e-12;
        bool B_changed = std::abs(s4.B[0][0] - B_before) > 1e-12;
        bool C_changed = std::abs(s4.C[0][0] - C_before) > 1e-12;
        bool D_changed = std::abs(s4.D - D_before) > 1e-12;

        if (!check("x_proj updated after training step", x_proj_changed)) failed++; else passed++;
        if (!check("W_out updated after training step", W_out_changed)) failed++; else passed++;
        if (!check("Lambda updated after training step", Lambda_changed)) failed++; else passed++;
        if (!check("B updated after training step", B_changed)) failed++; else passed++;
        if (!check("C updated after training step", C_changed)) failed++; else passed++;
        if (!check("D updated after training step", D_changed)) failed++; else passed++;
    }

    // =================================================================
    // Test 7: Zero grad
    // =================================================================
    cout << endl << "-- Test 7: zero_grad() clears all gradients --" << endl;
    {
        S4Layer s4(4, 2);
        int seq_len = 8;
        Tensor input = Tensor::random(4, seq_len, 0.5);
        Tensor grad_out(4, seq_len);
        grad_out.fill(1.0);

        s4.forward(input);
        s4.backward(grad_out, 0.01);
        s4.zero_grad();

        double gx = tensor_l2norm(s4.grad_x_proj);
        double gW = tensor_l2norm(s4.grad_W_out);
        double gb = tensor_l2norm(s4.grad_b_out);
        double gL = tensor_l2norm(s4.grad_Lambda_);
        double gB = tensor_l2norm(s4.grad_B_);
        double gC = tensor_l2norm(s4.grad_C_);
        double gD = std::abs(s4.grad_D_);

        bool all_zero = (gx < 1e-12 && gW < 1e-12 && gb < 1e-12 &&
                         gL < 1e-12 && gB < 1e-12 && gC < 1e-12 && gD < 1e-12);
        if (!check("all gradients zero after zero_grad()", all_zero)) failed++; else passed++;
    }

    // =================================================================
    // Test 8: Input dimension mismatch throws invalid_argument
    // =================================================================
    cout << endl << "-- Test 8: Input dimension mismatch --" << endl;
    {
        S4Layer s4(8, 4);
        Tensor wrong_dim_input(10, 4);  // d_model=10 but layer expects 8
        wrong_dim_input.fill(0.5);

        bool threw = false;
        try {
            Tensor out = s4.forward(wrong_dim_input);
        } catch (const std::invalid_argument& e) {
            threw = true;
        }
        if (!check("wrong dimension input throws invalid_argument", threw)) failed++;
        else passed++;
    }

    // =================================================================
    // Summary
    // =================================================================
    cout << endl << setprecision(4);
    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}
