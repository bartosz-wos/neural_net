// Expire-Span Attention — Sukhbaatar et al. 2021, EMNLP
//   "Not All Memories are Created Equal: A Learnable, Per-Layer Memory Budget"
//   (https://arxiv.org/abs/2105.11850)
//
// Tests (numbers are 1-based, mirroring the plan tasks):
//   1.  ExpireSpanAttention constructor validation (5 invalid + 1 valid)
//   2.  ExpireSpanAttention forward shape + finiteness (n=6, d=8, Hq=4, Hkv=2)
//   3.  Hand-derived mask signature: all-span=1 ⇒ no mask below diagonal
//   4.  Hand-derived mask signature: zero span ⇒ self-only attention
//   5.  Hand-derived mask signature: span=0.5, S_max=4 ⇒ window of 2
//   6.  FD input gradient (soft-mask BPTT)  rel_err < 1e-4
//   7.  FD W_q, W_k, W_v, W_o gradients    rel_err < 1e-5
//   8.  FD W_span, b_span gradients        rel_err < 1e-4
//   9.  Cache shapes after forward (n=6)
//   10. Determinism (two layers seeded the same way ⇒ bit-exact)
//   11. zero_grad clears all 6 gradient tensors
//   12. update_weights moves all 6 parameters
//   13. Mutation: zero out W_span ⇒ output unchanged bit-exact
//   14. ExpireSpanBlock forward shape
//   15. ExpireSpanBlock input FD gradient
//   16. ExpireSpanModel forward shape
//   17. End-to-end training reduces MSE loss > 30%
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include "nn/layers/attention/expire_span.h"

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

int main() {
    cout << "=== Expire-Span Attention Tests ===" << endl;
    cout.setf(std::ios::unitbuf);
    int total = 0, passed = 0;

    // ------------------------------------------------------------
    // Test 1: constructor validation + accessors + name
    // ------------------------------------------------------------
    cout << "\n--- Test 1: ExpireSpanAttention constructor validation ---\n";
    {
        ++total;
        int sub_pass = 0;
        // d_model=0 throws
        try { ExpireSpanAttention a(0, 4, 2); sub_pass--; cout << "  d_model=0 did NOT throw\n"; }
        catch (const std::invalid_argument&) { sub_pass++; }
        // num_query_heads=0 throws
        try { ExpireSpanAttention a(8, 0, 2); sub_pass--; cout << "  num_query_heads=0 did NOT throw\n"; }
        catch (const std::invalid_argument&) { sub_pass++; }
        // num_kv_heads=0 throws
        try { ExpireSpanAttention a(8, 4, 0); sub_pass--; cout << "  num_kv_heads=0 did NOT throw\n"; }
        catch (const std::invalid_argument&) { sub_pass++; }
        // num_query_heads % num_kv_heads != 0 throws (4 % 3 != 0)
        try { ExpireSpanAttention a(8, 4, 3); sub_pass--; cout << "  bad divisibility did NOT throw\n"; }
        catch (const std::invalid_argument&) { sub_pass++; }
        // s_min < 0 throws
        try { ExpireSpanAttention a(8, 4, 2, 4, -0.1); sub_pass--; cout << "  s_min<0 did NOT throw\n"; }
        catch (const std::invalid_argument&) { sub_pass++; }

        // Valid construct — accessors
        ExpireSpanAttention a(8, 4, 2, 4);
        if (a.d_model() == 8 && a.num_heads() == 4 && a.num_kv_heads() == 2 &&
            a.S_max() == 4 && a.name() == "ExpireSpanAttention" &&
            a.head_dim() == 2 && a.s_min() == 1e-6) {
            sub_pass++;
        } else {
            cout << "  accessors wrong: d_model=" << a.d_model()
                 << " Hq=" << a.num_heads() << " Hkv=" << a.num_kv_heads()
                 << " S_max=" << a.S_max() << " head_dim=" << a.head_dim()
                 << " s_min=" << a.s_min() << " name=" << a.name() << "\n";
        }

        if (sub_pass == 6) {
            cout << "[PASS] all 6 constructor/accessor sub-tests passed\n";
            ++passed;
        } else {
            cout << "[FAIL] " << (6 - sub_pass) << " sub-test(s) failed\n";
        }
    }

    // ------------------------------------------------------------
    // Test 2: ExpireSpanAttention forward shape + finiteness
    // ------------------------------------------------------------
    cout << "\n--- Test 2: ExpireSpanAttention forward shape + finiteness ---\n";
    {
        ++total;
        size_t n = 6, d = 8;
        size_t num_q = 4, num_kv = 2;
        Tensor input = Tensor::random(n, d, 0.5);  // non-degenerate scale

        ExpireSpanAttention a(d, num_q, num_kv, /*S_max=*/4);
        Tensor output = a.forward(input);
        cout << "Input:  " << input.rows << "x" << input.cols
             << "  Output: " << output.rows << "x" << output.cols << "\n";

        bool shape_ok = (output.rows == n && output.cols == d);
        bool finite = true;
        bool nonzero = false;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j) {
                if (!std::isfinite(output(i, j))) finite = false;
                if (std::abs(output(i, j)) > 1e-12) nonzero = true;
            }

        if (shape_ok && finite && nonzero) {
            cout << "[PASS] forward shape OK, all outputs finite and nonzero\n";
            ++passed;
        } else {
            cout << "[FAIL] shape=" << shape_ok << " finite=" << finite << " nonzero=" << nonzero << "\n";
        }
    }

    cout << "\n=== Results: " << passed << "/" << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
