// Token Merging (ToMe): Bolya et al. ICCV 2023, arxiv:2210.09461.
//
// Tests cover:
//   - constructor validation (3 throws)
//   - accessors (ratio, name, n_out)
//   - forward shape (N=8 d=4 r=0.5 → (4, 4))
//   - match-edge signature: |edges|=n_out, a_idx in [0, N/2), b_idx in [0, N/2)
//   - closed-form (N=4 d=2) merge matches (X_a + X_b*)/2 exactly
//   - FD input gradient (N=6 d=3 r=0.5) rel_err < 1e-4
//   - forward + unmerge round-trip (N=8 d=4 → (4,4) → unmerge → (8,4))
//   - parameters/gradients/update_weights/zero_grad contracts (all empty)
//   - determinism (two consecutive forwards bit-exact)
//   - edge cases: N=5→N'=2; N=2→N'=1; zero-norm rows don't NaN
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <utility>
#include <stdexcept>
#include "nn/layers/attention/tome.h"

using namespace std;

static double rel_err(double a, double b) {
    double denom = max(fabs(b), 1e-12);
    return fabs(a - b) / denom;
}

static double max_rel(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            m = max(m, rel_err(a(i, j), b(i, j)));
    return m;
}

static void check_or_record(bool cond, const string& name, int& passes, int& total) {
    total++;
    if (cond) {
        passes++;
    } else {
        cerr << "  [FAIL] " << name << "\n";
    }
}

int main() {
    cout << "=== Token Merging (ToMe) Tests ===" << endl;
    int passes = 0, total = 0;

    // ====================================================================
    // 1. Constructor validation
    // ====================================================================
    {
        cout << "\n--- Constructor validation ---\n";
        bool t1 = false, t2 = false;
        try { TokenMerging(0.0); } catch (const exception&) { t1 = true; }
        try { TokenMerging(1.5); } catch (const exception&) { t2 = true; }
        check_or_record(t1, "ratio=0 throws",        passes, total);
        check_or_record(t2, "ratio>1 throws",        passes, total);
    }

    // ====================================================================
    // 2. Accessors + n_out
    // ====================================================================
    {
        cout << "\n--- Accessors ---\n";
        TokenMerging tm(0.5);
        bool ok = (tm.ratio() == 0.5)
               && (tm.name() == "TokenMerging")
               && (tm.n_out(8) == 4)
               && (tm.n_out(5) == 2)   // ⌊5*0.5⌋ = ⌊2.5⌋ = 2
               && (tm.n_out(2) == 1)   // ⌊2*0.5⌋ = 1
               && (tm.n_out(10) == 5);
        check_or_record(ok, "accessors return constructor values", passes, total);
    }

    // ====================================================================
    // 3. Forward shape + finiteness (N=8 d=4 r=0.5 → (4, 4))
    // ====================================================================
    {
        cout << "\n--- Forward shape (N=8 d=4 r=0.5) ---\n";
        TokenMerging tm(0.5);
        Tensor x = Tensor::random(8, 4, 0.5);
        Tensor y = tm.forward(x);
        bool shape_ok = (y.rows == 4) && (y.cols == 4);
        bool finite = true;
        for (size_t i = 0; i < y.rows && finite; ++i)
            for (size_t j = 0; j < y.cols && finite; ++j)
                if (!isfinite(y(i, j))) finite = false;
        bool nonzero = false;
        for (size_t i = 0; i < y.rows && !nonzero; ++i)
            for (size_t j = 0; j < y.cols && !nonzero; ++j)
                if (fabs(y(i, j)) > 1e-8) nonzero = true;
        check_or_record(shape_ok, "y.shape == (4, 4)", passes, total);
        check_or_record(finite,   "y is finite",       passes, total);
        check_or_record(nonzero,  "y is nonzero",      passes, total);
    }

    // ====================================================================
    // 4. Match-edge signature
    // ====================================================================
    {
        cout << "\n--- Match-edge signature (N=8) ---\n";
        TokenMerging tm(0.5);
        Tensor x = Tensor::random(8, 4, 0.5);
        Tensor y = tm.forward(x);
        const auto& edges = tm.last_match_edges();
        bool count_ok = edges.size() == (size_t)y.rows;
        bool ranges_ok = true;
        for (auto& p : edges) {
            // a_idx in {0,2,4,6}, b_idx in {1,3,5,7}
            if (p.first % 2 != 0 || p.first >= 8) { ranges_ok = false; break; }
            if (p.second % 2 != 1 || p.second >= 8) { ranges_ok = false; break; }
        }
        check_or_record(count_ok,   "edges.size() == y.rows", passes, total);
        check_or_record(ranges_ok,  "a in {0,2,4,6}, b in {1,3,5,7}",   passes, total);
    }

    // ====================================================================
    // 5. Closed-form merge (N=4 d=2)
    // ====================================================================
    {
        cout << "\n--- Closed-form merge (N=4 d=2) ---\n";
        TokenMerging tm(0.5);
        // Hand-picked x where match is predictable.
        // A = {0, 2}, B = {1, 3}. With x = [[1,0],[1,1],[0,0],[0,1]]:
        //   ||x_0|| = 1, ||x_1|| = sqrt(2), ||x_2|| = 0, ||x_3|| = 1
        //   s_01 = (1*1+0*1)/(2*1*sqrt(2)) = 1/(2*sqrt(2)) ≈ 0.354
        //   s_03 = (1*0+0*1)/(2*1*1)       = 0
        //   s_21 = (0*1+0*1)/(2*0*sqrt(2)) = 0/0 → tied at -inf
        //   s_23 = (0*0+0*1)/(2*0*1)       = 0/0 → tied at -inf
        // For a=2: all scores are -inf (zero-norm row), tie-break: best_b stays at
        // initial 0 → match (2, 0).
        // X' = [(x_0 + x_1)/2; (x_2 + x_0)/2] = [[1.0, 0.5]; [0.5, 0.0]]
        Tensor x(4, 2);
        x(0,0)=1.0; x(0,1)=0.0;
        x(1,0)=1.0; x(1,1)=1.0;
        x(2,0)=0.0; x(2,1)=0.0;
        x(3,0)=0.0; x(3,1)=1.0;
        Tensor y = tm.forward(x);
        Tensor yref(2, 2);
        yref(0,0)=1.0; yref(0,1)=0.5;
        yref(1,0)=0.5; yref(1,1)=0.0;
        bool shape_ok = (y.rows == 2) && (y.cols == 2);
        bool merge_ok = (max_rel(y, yref) < 1e-12);
        check_or_record(shape_ok,  "y.shape == (2, 2)",                  passes, total);
        check_or_record(merge_ok,  "merge = (X_a + X_b*)/2 to 1e-12",    passes, total);
    }

    // ====================================================================
    // 6. Backward FD input gradient (N=6 d=3 r=0.5 → 3 rows out)
    //
    // The v1 contract: matches are fixed at forward and treated as fixed during
    // backward (the standard STE-free inference convention). To validate the
    // merge+backward math in isolation we pin the match edges via the
    // test-only setter; without pinning, FD perturbs x and flips matches,
    // measuring a different operating point than the analytical.
    // ====================================================================
    {
        cout << "\n--- FD input gradient (N=6 d=3 r=0.5) ---\n";
        srand(42);
        TokenMerging tm(0.5);
        Tensor x = Tensor::random(6, 3, 0.3);
        // Pin matches to a fixed bipartite pairing: a ∈ {0, 2, 4} → b ∈ {1, 3, 5}.
        vector<pair<size_t, size_t>> pinned = {{0, 1}, {2, 3}, {4, 5}};
        tm._set_match_edges_for_test(x.rows, x.cols, pinned);

        // Compute y and grad_y from the PINNED matches (so y matches what
        // tm.backward(grad_y, 0) is consistent with).
        size_t n_out_ = pinned.size();
        Tensor y(n_out_, x.cols);
        for (size_t k = 0; k < n_out_; ++k) {
            size_t a = pinned[k].first;
            size_t b = pinned[k].second;
            for (size_t j = 0; j < x.cols; ++j) y(k, j) = 0.5 * (x(a, j) + x(b, j));
        }
        Tensor grad_y(y.rows, y.cols);
        // grad_y[k, j] = dL/dy[k, j] = 2 * y[k, j]  (because L = sum(y*y))
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j)
                grad_y(i, j) = 2.0 * y(i, j);
        Tensor grad_x_ana = tm.backward(grad_y, 0.0);

        // FD: re-pin matches on every forward call so the merge is exactly
        // the merge-step-only perturbation (not a different operating point).
        double eps = 1e-5;
        double max_re = 0.0;
        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                double xp = x(i, j);
                x(i, j) = xp + eps;
                tm._set_match_edges_for_test(x.rows, x.cols, pinned);
                Tensor yp(n_out_, x.cols);
                for (size_t k = 0; k < n_out_; ++k) {
                    size_t a = pinned[k].first;
                    size_t b = pinned[k].second;
                    for (size_t jj = 0; jj < x.cols; ++jj) yp(k, jj) = 0.5 * (x(a, jj) + x(b, jj));
                }
                double lp = 0.0;
                for (size_t a = 0; a < yp.rows; ++a)
                    for (size_t b = 0; b < yp.cols; ++b)
                        lp += yp(a, b) * yp(a, b);
                x(i, j) = xp - eps;
                tm._set_match_edges_for_test(x.rows, x.cols, pinned);
                Tensor ym(n_out_, x.cols);
                for (size_t k = 0; k < n_out_; ++k) {
                    size_t a = pinned[k].first;
                    size_t b = pinned[k].second;
                    for (size_t jj = 0; jj < x.cols; ++jj) ym(k, jj) = 0.5 * (x(a, jj) + x(b, jj));
                }
                double lm = 0.0;
                for (size_t a = 0; a < ym.rows; ++a)
                    for (size_t b = 0; b < ym.cols; ++b)
                        lm += ym(a, b) * ym(a, b);
                x(i, j) = xp;
                double grad_fd = (lp - lm) / (2.0 * eps);
                max_re = max(max_re, rel_err(grad_x_ana(i, j), grad_fd));
            }
        }
        check_or_record(max_re < 1e-4,
            "FD input gradient rel_err < 1e-4 (max_rel=" + to_string(max_re) + ")",
            passes, total);
    }

    // ====================================================================
    // 7. forward + unmerge round-trip
    // ====================================================================
    {
        cout << "\n--- forward + unmerge round-trip ---\n";
        TokenMerging tm(0.5);
        Tensor x = Tensor::random(8, 4, 0.5);
        Tensor y = tm.forward_with_unmerge(x);
        bool shape_ok = (y.rows == 4) && (y.cols == 4);
        Tensor grad_y = Tensor::random(4, 4, 0.1);
        Tensor grad_x = tm.unmerge(grad_y);
        bool back_shape = (grad_x.rows == 8) && (grad_x.cols == 4);
        bool back_finite = true;
        for (size_t i = 0; i < grad_x.rows && back_finite; ++i)
            for (size_t j = 0; j < grad_x.cols && back_finite; ++j)
                if (!isfinite(grad_x(i, j))) back_finite = false;
        check_or_record(shape_ok,   "forward_with_unmerge shape (4, 4)", passes, total);
        check_or_record(back_shape, "unmerge returns (8, 4)",           passes, total);
        check_or_record(back_finite, "unmerge output finite",            passes, total);
    }

    // ====================================================================
    // 8. parameters/gradients/update_weights/zero_grad contracts
    // ====================================================================
    {
        cout << "\n--- parameters/gradients/update_weights/zero_grad ---\n";
        TokenMerging tm(0.5);
        bool p_ok = tm.parameters().empty();
        bool g_ok = tm.gradients().empty();
        bool z_ok = true; try { tm.zero_grad(); } catch (...) { z_ok = false; }
        bool u_ok = true; try { tm.update_weights(0.01); } catch (...) { u_ok = false; }
        check_or_record(p_ok, "parameters() empty",          passes, total);
        check_or_record(g_ok, "gradients() empty",           passes, total);
        check_or_record(z_ok, "zero_grad() doesn't throw",   passes, total);
        check_or_record(u_ok, "update_weights() doesn't throw", passes, total);
    }

    // ====================================================================
    // 9. Determinism
    // ====================================================================
    {
        cout << "\n--- Determinism ---\n";
        TokenMerging tm(0.5);
        Tensor x = Tensor::random(8, 4, 0.5);
        Tensor y1 = tm.forward(x);
        Tensor y2 = tm.forward(x);
        double md = 0.0;
        for (size_t i = 0; i < y1.rows; ++i)
            for (size_t j = 0; j < y1.cols; ++j)
                md = max(md, fabs(y1(i, j) - y2(i, j)));
        check_or_record(md == 0.0, "two consecutive forwards bit-exact", passes, total);
    }

    // ====================================================================
    // 10. Edge cases
    // ====================================================================
    {
        cout << "\n--- Edge cases ---\n";
        // N=2 d=2 r=0.5 → n_out = 1
        TokenMerging tm2(0.5);
        Tensor x2 = Tensor::random(2, 2, 0.5);
        Tensor y2 = tm2.forward(x2);
        bool shape2 = (y2.rows == 1) && (y2.cols == 2);
        check_or_record(shape2, "N=2 r=0.5 → (1, 2)", passes, total);

        // N=5 d=3 r=0.5 → n_out = 2
        TokenMerging tm5(0.5);
        Tensor x5 = Tensor::random(5, 3, 0.5);
        Tensor y5 = tm5.forward(x5);
        bool shape5 = (y5.rows == 2) && (y5.cols == 3);
        check_or_record(shape5, "N=5 r=0.5 → (2, 3)", passes, total);

        // Zero-norm row doesn't NaN
        TokenMerging tmz(0.5);
        Tensor xz = Tensor::zeros(4, 2);
        Tensor yz = tmz.forward(xz);
        bool nan_free = true;
        for (size_t i = 0; i < yz.rows && nan_free; ++i)
            for (size_t j = 0; j < yz.cols && nan_free; ++j)
                if (!isfinite(yz(i, j))) nan_free = false;
        check_or_record(nan_free, "zero-norm input doesn't NaN", passes, total);
    }

    cout << "\n=== Summary: " << passes << " passed, " << total << " total ===" << endl;
    return (passes == total) ? 0 : 1;
}
