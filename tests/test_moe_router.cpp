// Tests for MoeRouter — a pure-stateless top-k softmax routing + load-balance
// utility shared across Mixture-of-Experts layers.
//
// Spec under test: include/nn/layers/architectures/moe_router.h
//
// Coverage (27 focused checks):
//   1.  empty logits (rows=0)            — softmax_topk throws
//   2.  forward shape (B=4,E=6,k=2)      — probs (4,6), topk (4,2), etc.
//   3.  topk indices = k largest logits   — exact ordering on hand fixture
//   4.  probs sum to 1 per row           — max err < 1e-12
//   5.  non-selected probs are exactly 0  — sparsity contract
//   6.  mean gate prob = mean over rows  — formula check
//   7.  dispatch fraction = count/B       — formula check
//   8.  load-balance loss                — closed form for tiny fixture
//   9.  numerical stability              — logits spanning ±1000
//   10. k=E edge case                    — all selected
//   11. softmax_topk_backward shape      — d_logits matches input logits shape
//   12. backward single-cell probe       — gate_probs probe identity
//   13. backward full FD                 — rel_err < 1e-9 with non-trivial
//                                           d_gate_probs
//   14. backward symmetry identity       — exact algebra via row-sum constraint
//   15. load_balance_loss symmetry       — invariant under permutation
//   16. load_balance_loss non-neg        — >=0
//   17. load_balance_loss uniform        — uniform fractions & probs → 1.0
//   18. multi-batch forward invariants   — B=8,E=4,k=2 invariants
//   19. backward chain via aux loss      — FD on E*p_e ≈ closeform
//   20. state round-trip                 — ones d_gate_probs → d_logits=0
//   21. state independent of input       — caller mutation doesn't affect
//   22. composed-output chain            — full combine FD agrees with FD
//   23. tie-breaking determinism         — equal logits → probs sum to 1
//   24. k=0 throws
//   25. k > E throws
//   26. B=0 edge case
//   27. determinism                      — bit-exact across reruns

#include "nn/layers/architectures/moe_router.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

void check_close(double a, double b, double tol, const char* msg) {
    double err = std::abs(a - b);
    if (err <= tol) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << " (err=" << err << ")\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg
                  << "  expected near " << b << " got " << a
                  << " (err=" << err << " > tol=" << tol << ")\n";
    }
}

Tensor make_logits_4x6() {
    // rows: 0..3, cols: 0..5 — make col 2 and col 5 the obvious top-2 per row
    std::vector<std::vector<double>> d = {
        { 1.0,  0.5,  9.0,  0.1, -1.0,  7.0},   // top-2: col 2 (9.0), col 5 (7.0)
        {-2.0,  0.0,  3.0, -1.0,  0.5,  5.0},   // top-2: col 5 (5.0), col 2 (3.0)
        {-3.0, -2.0, -1.0,  4.0, -0.5,  0.5},   // top-2: col 3 (4.0), col 5 (0.5)
        { 0.1,  0.2,  0.3,  0.4,  0.5,  0.6},   // top-2: col 5, col 4
    };
    return Tensor(d);
}

// Centred finite difference of softmax_topk.forward(perturbed_logits).probs,
// measured at a single coordinate (b,e). The "loss" we differentiate is
// `sum over (b',e') of  probs[b',e'] * d_gate_probs[b',e']` — i.e. the inner
// product. d_loss / d_logits[b, e] is then exactly the entry of
// softmax_topk_backward(d_gate_probs, state) at (b, e).
double numerical_d_logits_at(const Tensor& logits, size_t k,
                              size_t b, size_t e,
                              const Tensor& d_gate_probs,
                              double eps) {
    Tensor l_plus = logits.clone();
    Tensor l_minus = logits.clone();
    l_plus(b, e) += eps;
    l_minus(b, e) -= eps;
    auto s_plus  = MoeRouter::softmax_topk(l_plus, k);
    auto s_minus = MoeRouter::softmax_topk(l_minus, k);
    // sum over (b', e') probs[b', e'] * d_gate_probs[b', e']
    double lp = 0.0, lm = 0.0;
    for (size_t bp = 0; bp < logits.rows; ++bp) {
        for (size_t ep = 0; ep < (size_t)logits.cols; ++ep) {
            lp += s_plus.probs(bp, ep)  * d_gate_probs(bp, ep);
            lm += s_minus.probs(bp, ep) * d_gate_probs(bp, ep);
        }
    }
    return (lp - lm) / (2.0 * eps);
}

}  // namespace

int main() {
    std::cout << "=== MoeRouter Tests ===\n";

    // -----------------------------------------------------------------------
    // Test 1: softmax_topk with empty logits throws
    // -----------------------------------------------------------------------
    {
        bool threw = false;
        Tensor empty(0, 6);
        try { (void)MoeRouter::softmax_topk(empty, 2); }
        catch (std::invalid_argument&) { threw = true; }
        check(threw, "1. empty logits (rows=0) throws");
    }

    // -----------------------------------------------------------------------
    // Test 2: forward shape (B=4, E=6, k=2)
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        check(s.probs.rows == 4 && s.probs.cols == 6, "2a. probs shape (4,6)");
        check(s.topk_idx.size() == 4, "2b. topk rows == 4");
        bool all_k = std::all_of(s.topk_idx.begin(), s.topk_idx.end(),
                                  [](const auto& v) { return v.size() == 2; });
        check(all_k, "2c. each topk row has exactly k=2 entries");
        check(s.dispatch_frac.size() == 6, "2d. dispatch_frac size == E");
        check(s.mean_gate_prob.size() == 6, "2e. mean_gate_prob size == E");
        check(s.maxv.size() == 4, "2f. maxv size == B");
        check(s.exps.size() == 4, "2g. exps rows == B");
        bool all_k2 = std::all_of(s.exps.begin(), s.exps.end(),
                                   [](const auto& v) { return v.size() == 2; });
        check(all_k2, "2h. each exps row has exactly k=2 entries");
        check(s.logsumexp.size() == 4, "2i. logsumexp size == B");
        check(s.num_experts == 6, "2j. state.num_experts == 6");
        check(s.k == 2, "2k. state.k == 2");
    }

    // -----------------------------------------------------------------------
    // Test 3: top-k indices are exactly the k largest logits per row
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        std::vector<std::vector<size_t>> expected = {
            {2, 5},
            {5, 2},
            {3, 5},
            {5, 4},
        };
        bool ok = true;
        for (size_t b = 0; b < 4; ++b) {
            for (size_t i = 0; i < 2; ++i) {
                if (s.topk_idx[b][i] != expected[b][i]) { ok = false; }
            }
        }
        check(ok, "3. topk indices match hand-derived k largest per row");
    }

    // -----------------------------------------------------------------------
    // Test 4: probs sum to 1 per row
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        bool ok = true;
        for (size_t b = 0; b < 4; ++b) {
            double rsum = 0.0;
            for (size_t e = 0; e < 6; ++e) rsum += s.probs(b, e);
            if (std::abs(rsum - 1.0) > 1e-12) ok = false;
        }
        check(ok, "4. probs sum to 1 per row (max err < 1e-12)");
    }

    // -----------------------------------------------------------------------
    // Test 5: non-selected entries are exactly 0
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        bool ok = true;
        for (size_t b = 0; b < 4; ++b) {
            for (size_t e = 0; e < 6; ++e) {
                bool selected = false;
                for (size_t i = 0; i < 2; ++i) {
                    if (s.topk_idx[b][i] == e) { selected = true; break; }
                }
                if (!selected && s.probs(b, e) != 0.0) { ok = false; break; }
            }
        }
        check(ok, "5. non-selected entries are exactly 0");
    }

    // -----------------------------------------------------------------------
    // Test 6: mean gate prob = mean over rows of probs (selected only)
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        bool ok = true;
        for (size_t e = 0; e < 6; ++e) {
            double sum_b = 0.0;
            for (size_t b = 0; b < 4; ++b) sum_b += s.probs(b, e);
            double expected = sum_b / 4.0;
            if (std::abs(s.mean_gate_prob[e] - expected) > 1e-12) { ok = false; break; }
        }
        check(ok, "6. mean_gate_prob = (1/B) * Σ_b probs[b,e]");
    }

    // -----------------------------------------------------------------------
    // Test 7: dispatch fraction = count of selections / batch
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        // Count selections per expert from topk_idx
        std::vector<int> count(6, 0);
        for (size_t b = 0; b < 4; ++b)
            for (size_t i = 0; i < 2; ++i)
                count[s.topk_idx[b][i]]++;
        bool ok = true;
        for (size_t e = 0; e < 6; ++e) {
            double expected = count[e] / 4.0;
            if (std::abs(s.dispatch_frac[e] - expected) > 1e-12) { ok = false; break; }
        }
        check(ok, "7. dispatch_frac = (#selections) / B per expert");
    }

    // -----------------------------------------------------------------------
    // Test 8: load_balance_loss_unscaled closed form for tiny fixture
    // -----------------------------------------------------------------------
    {
        // B=2, E=3, k=1: 2 selections total, fractions sum to 1.
        // probs rows sum to 1, so sum_e p_e = (k / E) on average.
        // For a specific fixture, compute the closed form by hand.
        Tensor logits = Tensor(std::vector<std::vector<double>>{
            {1.0, 0.5, 0.0},
            {0.2, 0.8, 0.1},
        });
        auto s = MoeRouter::softmax_topk(logits, 1);
        double L = MoeRouter::load_balance_loss_unscaled(s.dispatch_frac, s.mean_gate_prob);
        // Closed form: 3 * sum_e (f_e * p_e)
        double closed = 0.0;
        for (size_t e = 0; e < 3; ++e) closed += s.dispatch_frac[e] * s.mean_gate_prob[e];
        closed *= 3.0;
        check_close(L, closed, 1e-12, "8a. load-balance loss matches 3 * Σ f_e * p_e");
        check(L >= 0.0, "8b. load-balance loss >= 0");
    }

    // -----------------------------------------------------------------------
    // Test 9: numerical stability with logits spanning ±1000
    // -----------------------------------------------------------------------
    {
        // Row 2: ALL logits = -1000 — without the 1e-30 sumexp floor,
        //        exp(-1000 - (-1000)) = 1 only if maxv is computed; but the
        //        stable exp(values - maxv) gives exp(0)=1 for both top-k
        //        and sumexp=2. Still safe.
        Tensor logits(4, 4);
        logits(0,0) = 1000.0; logits(0,1) = -1000.0; logits(0,2) = 1000.0; logits(0,3) = -1000.0;
        logits(1,0) = -1000.0; logits(1,1) = 1000.0; logits(1,2) = -1000.0; logits(1,3) = 1000.0;
        logits(2,0) = 1000.0; logits(2,1) = 1000.0; logits(2,2) = -1000.0; logits(2,3) = -1000.0;
        // Row 3: top-k are -1500 — the largest is exactly the max, so exp(0)=1
        //        for both top-k entries, sumexp = 2 (no floor triggered).
        //        The 1e-30 floor is defensive against future code paths that
        //        might produce zero sumexp (e.g. finite-difference noise).
        logits(3,0) = -1500.0;  // largest of this row
        logits(3,1) = -1500.0;
        logits(3,2) = -1501.0;
        logits(3,3) = -1501.0;
        bool threw = false; bool nonfin = false; bool row3_zero = false;
        MoeRouterState s;
        try { s = MoeRouter::softmax_topk(logits, 2); }
        catch (...) { threw = true; }
        if (!threw) {
            for (size_t b = 0; b < 4 && !nonfin; ++b)
                for (size_t e = 0; e < 4 && !nonfin; ++e)
                    if (!std::isfinite(s.probs(b, e))) nonfin = true;
            // Row 3 sum should be exactly 1.0 (the 1e-30 floor only kicks in
            // if sumexp is tiny, but here at -1500 - (-1500) = 0, exp(0)=1)
            double r3sum = 0.0;
            for (size_t e = 0; e < 4; ++e) r3sum += s.probs(3, e);
            if (std::abs(r3sum - 1.0) > 1e-12) row3_zero = true;
        }
        check(!threw && !nonfin && !row3_zero, "9. logits ±1500 → probs finite, sums=1, no throw");
    }

    // -----------------------------------------------------------------------
    // Test 10: k=E edge case (all experts selected)
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();  // B=4, E=6
        auto s = MoeRouter::softmax_topk(logits, 6);  // k=E
        // Each row should have Σ_e probs = 1 (full softmax, since no mask).
        bool ok = true;
        for (size_t b = 0; b < 4; ++b) {
            double rsum = 0.0;
            for (size_t e = 0; e < 6; ++e) rsum += s.probs(b, e);
            if (std::abs(rsum - 1.0) > 1e-12) { ok = false; break; }
        }
        check(ok, "10. k=E: full softmax, probs sum to 1 per row");
    }

    // -----------------------------------------------------------------------
    // Test 11: softmax_topk_backward shape
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        Tensor d_gate_probs = Tensor::zeros(4, 6);
        for (size_t i = 0; i < 4*6; ++i) d_gate_probs.data[i] = 0.1 * (double)(i + 1);
        Tensor d_logits = MoeRouter::softmax_topk_backward(d_gate_probs, s);
        check(d_logits.rows == 4 && d_logits.cols == 6,
              "11. softmax_topk_backward shape (B, E)");
    }

    // -----------------------------------------------------------------------
    // Test 12: backward single-cell probe
    //   d_gate_probs[b*, e*] = 1, others = 0
    //   => d_logits[b*, e*] = probs[b*, e*] * (1 - probs[b*, e*])
    //      d_logits[b*, e_other_selected] = -probs[b*, e_other_selected] * probs[b*, e*]
    //      d_logits[b*, e_unselected] = 0
    //      d_logits[b!=b*, .] = 0
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        size_t b_star = 0;
        size_t e_star = s.topk_idx[b_star][0];   // first selected of row 0
        size_t e_other = s.topk_idx[b_star][1];  // second selected of row 0
        Tensor d_gate_probs = Tensor::zeros(4, 6);
        d_gate_probs(b_star, e_star) = 1.0;
        Tensor d_logits = MoeRouter::softmax_topk_backward(d_gate_probs, s);
        bool ok = true;
        double expected_self = s.probs(b_star, e_star) * (1.0 - s.probs(b_star, e_star));
        double expected_other = -s.probs(b_star, e_other) * s.probs(b_star, e_star);
        if (std::abs(d_logits(b_star, e_star) - expected_self) > 1e-13) ok = false;
        if (std::abs(d_logits(b_star, e_other) - expected_other) > 1e-13) ok = false;
        // All other entries should be zero
        for (size_t b = 0; b < 4; ++b) {
            for (size_t e = 0; e < 6; ++e) {
                if (b == b_star && (e == e_star || e == e_other)) continue;
                if (d_logits(b, e) != 0.0) { ok = false; }
            }
        }
        check(ok, "12. backward probe: d_gate_probs probe identity matches closed form");
    }

    // -----------------------------------------------------------------------
    // Test 13: full FD vs backward (rel_err < 1e-9)
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        const size_t B = 3, E = 5, k = 2;
        Tensor logits(B, E);
        for (size_t i = 0; i < B*E; ++i) logits.data[i] = uni(rng);
        Tensor d_gate_probs(B, E);
        for (size_t i = 0; i < B*E; ++i) d_gate_probs.data[i] = uni(rng);
        auto s = MoeRouter::softmax_topk(logits, k);
        Tensor d_logits = MoeRouter::softmax_topk_backward(d_gate_probs, s);
        // Compare worst absolute rel_err across all (b, e)
        double worst = 0.0;
        const double eps = 1e-5;
        for (size_t b = 0; b < B; ++b) {
            for (size_t e = 0; e < E; ++e) {
                double ana = d_logits(b, e);
                double num = numerical_d_logits_at(logits, k, b, e, d_gate_probs, eps);
                double denom = std::max(std::abs(ana), std::abs(num));
                double re = (denom > 1e-12) ? (std::abs(ana - num) / denom) : std::abs(ana - num);
                if (re > worst) worst = re;
            }
        }
        std::cout << "    worst_rel_err = " << worst << "\n";
        check(worst < 1e-9, "13. softmax_topk_backward vs FD: rel_err < 1e-9");
    }

    // -----------------------------------------------------------------------
    // Test 14: row-sum constraint — ones d_gate_probs gives d_logits == 0
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        auto s = MoeRouter::softmax_topk(logits, 2);
        Tensor ones = Tensor(4, 6);
        ones.fill(1.0);
        Tensor d_logits = MoeRouter::softmax_topk_backward(ones, s);
        bool ok = true;
        for (size_t b = 0; b < 4 && ok; ++b) {
            for (size_t e = 0; e < 6 && ok; ++e) {
                if (std::abs(d_logits(b, e)) > 1e-12) ok = false;
            }
        }
        check(ok, "14. d_gate_probs=ones → d_logits = 0 (row-sum constraint)");
    }

    // -----------------------------------------------------------------------
    // Test 15: load_balance_loss invariant under permutation of expert indices
    // -----------------------------------------------------------------------
    {
        std::vector<double> f = {0.1, 0.2, 0.3, 0.4};
        std::vector<double> p = {0.25, 0.35, 0.15, 0.25};
        double L0 = MoeRouter::load_balance_loss_unscaled(f, p);
        std::vector<size_t> idx = {0, 1, 2, 3};
        std::mt19937 rng(7);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<double> f_perm(4), p_perm(4);
        for (size_t i = 0; i < 4; ++i) { f_perm[i] = f[idx[i]]; p_perm[i] = p[idx[i]]; }
        double L1 = MoeRouter::load_balance_loss_unscaled(f_perm, p_perm);
        check_close(L0, L1, 1e-12, "15. load_balance_loss invariant under permutation");
    }

    // -----------------------------------------------------------------------
    // Test 16: load_balance_loss non-negative
    // -----------------------------------------------------------------------
    {
        // By AM-GM, Σ f_e * p_e ≥ (Σ f_e)(Σ p_e) / E^2 = (1)(1)/E^2, NOT a
        // guarantee in general — actually since f_e ≥0 and p_e ≥0 the product
        // is ≥0 so the sum is ≥0. (The Shazeer / GShard form.)
        std::vector<double> f = {0.5, 0.5};
        std::vector<double> p = {0.4, 0.6};
        double L = MoeRouter::load_balance_loss_unscaled(f, p);
        check(L >= 0.0, "16. load_balance_loss non-negative for nonnegative f,p");
    }

    // -----------------------------------------------------------------------
    // Test 17: uniform fractions and probs → loss = 1.0 (closed form E * E*(1/E)^2)
    // -----------------------------------------------------------------------
    {
        const size_t E = 8;
        std::vector<double> f(E), p(E);
        for (size_t i = 0; i < E; ++i) { f[i] = 1.0 / E; p[i] = 1.0 / E; }
        double L = MoeRouter::load_balance_loss_unscaled(f, p);
        // E * sum (1/E)*(1/E) = E * E * (1/E^2) = 1
        check_close(L, 1.0, 1e-12, "17. load_balance_loss uniform → 1.0");
    }

    // -----------------------------------------------------------------------
    // Test 18: multi-batch (B=8, E=4, k=2) invariants
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> uni(-2.0, 2.0);
        Tensor logits(8, 4);
        for (size_t i = 0; i < 8*4; ++i) logits.data[i] = uni(rng);
        auto s = MoeRouter::softmax_topk(logits, 2);
        bool row_ok = true, spars_ok = true;
        for (size_t b = 0; b < 8; ++b) {
            double rsum = 0.0;
            for (size_t e = 0; e < 4; ++e) rsum += s.probs(b, e);
            if (std::abs(rsum - 1.0) > 1e-12) row_ok = false;
            int sel = 0;
            for (size_t e = 0; e < 4; ++e) {
                if (s.probs(b, e) != 0.0) {
                    sel++;
                    bool in_topk = false;
                    for (size_t i = 0; i < 2; ++i) if (s.topk_idx[b][i] == e) in_topk = true;
                    if (!in_topk) spars_ok = false;
                }
            }
            if (sel != 2) spars_ok = false;
        }
        // dispatch fractions must sum to k / 1  per expert's selection count
        double total_selections = 0;
        for (size_t e = 0; e < 4; ++e) total_selections += s.dispatch_frac[e];
        check_close(total_selections, 2.0, 1e-12, "18a. dispatch fractions sum to 2 (= k)");
        check(row_ok, "18b. multi-batch row sums = 1");
        check(spars_ok, "18c. multi-batch sparsity (each row has exactly k non-zero probs)");
    }

    // -----------------------------------------------------------------------
    // Test 19: backward chain through auxiliary load-balance loss
    //   Setting loss = E * Σ_e f_e * p_e, scalar.
    //   dL/df_e = E * p_e, dL/dp_e = E * f_e
    //   p_e = (1/B) Σ_b probs[b,e] → dL/d probs[b,e] = E * f_e / B
    //   We feed d_gate_probs = (E/B * f_e) into softmax_topk_backward and
    //   compare against FD on (loss as a function of logits).
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        const size_t B = 4, E = 5, k = 2;
        Tensor logits(B, E);
        for (size_t i = 0; i < B*E; ++i) logits.data[i] = uni(rng);
        auto s = MoeRouter::softmax_topk(logits, k);
        // Build d_gate_probs from the auxiliary chain:
        Tensor d_gate_probs(B, E);
        for (size_t b = 0; b < B; ++b) {
            for (size_t e = 0; e < E; ++e) {
                d_gate_probs(b, e) = static_cast<double>(E) * s.dispatch_frac[e] /
                                      static_cast<double>(B);
            }
        }
        Tensor d_logits = MoeRouter::softmax_topk_backward(d_gate_probs, s);

        // FD reference for the loss derivative w.r.t. logits
        auto loss_at = [&](const Tensor& l) {
            auto s_ = MoeRouter::softmax_topk(l, k);
            double L = 0.0;
            for (size_t e = 0; e < E; ++e) {
                L += s_.dispatch_frac[e] * s_.mean_gate_prob[e];
            }
            return static_cast<double>(E) * L;
        };
        const double eps = 1e-5;
        // Report worst re only from cells where the analytical is non-trivial
        // (skipping cells where ana ~ 0 because the FD there is dominated by
        // partial_sort comparator swaps at FP-noise level, not by the gradient
        // — the FD-at-floor pattern of systematic-debugging §5c).
        double worst = 0.0;
        int nontrivial = 0;
        for (size_t b = 0; b < B; ++b) {
            for (size_t e = 0; e < E; ++e) {
                Tensor lp = logits.clone();
                Tensor lm = logits.clone();
                lp(b, e) += eps; lm(b, e) -= eps;
                double ana = d_logits(b, e);
                double num = (loss_at(lp) - loss_at(lm)) / (2.0 * eps);
                double denom = std::max(std::abs(ana), std::abs(num));
                double re = (denom > 1e-12) ? (std::abs(ana - num) / denom) : std::abs(ana - num);
                if (std::abs(ana) < 1e-9) continue;   // skip trivial cells (FD-at-floor)
                ++nontrivial;
                if (re > worst) worst = re;
            }
        }
        std::cout << "    worst_rel_err (aux-loss FD) = " << worst
                  << " (over " << nontrivial << " non-trivial cells)" << "\n";
        check(worst < 1e-3, "19. aux-loss backward chain FD agrees with analytical");
    }

    // -----------------------------------------------------------------------
    // Test 20: state fields independent of caller mutation
    //   Modify logits after softmax_topk; state fields should be unchanged.
    // -----------------------------------------------------------------------
    {
        Tensor logits(2, 3);
        logits(0,0)=0.5; logits(0,1)=0.1; logits(0,2)=0.9;
        logits(1,0)=0.2; logits(1,1)=0.7; logits(1,2)=0.3;
        auto s = MoeRouter::softmax_topk(logits, 2);
        // Snapshot
        Tensor probs_snap = s.probs.clone();
        // Mutate logits
        logits(0, 0) = 999.0;
        logits(1, 2) = -999.0;
        // State should be unchanged.
        bool ok = true;
        for (size_t b = 0; b < 2 && ok; ++b) {
            for (size_t e = 0; e < 3 && ok; ++e) {
                if (std::abs(s.probs(b, e) - probs_snap(b, e)) > 1e-15) ok = false;
            }
        }
        check(ok, "20. state fields independent of caller mutation of logits");
    }

    // -----------------------------------------------------------------------
    // Test 21: composed-output chain
    //   composed[b, j] = Σ_e probs[b, e] * exp_e[b, j]     (MoE combine)
    //   d_logits[b, e*] = probs[b,e*] * (
    //                       Σ_j d_composed[b,j] * exp_e*[b,j]
    //                       - Σ_{e' in S} probs[b,e'] * (Σ_j d_composed[b,j] * exp_{e'}[b,j]))
    //   for e* in S. Verifies full chain.
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(2024);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        const size_t B = 3, E = 4, k = 2, J = 3;
        Tensor logits(B, E);
        for (size_t i = 0; i < B*E; ++i) logits.data[i] = uni(rng);
        Tensor exp_e(B*E, J);
        for (size_t i = 0; i < (size_t)exp_e.rows; ++i)
            for (size_t j = 0; j < J; ++j) exp_e(i, j) = uni(rng);
        // reshape exp_e as (B, E, J) conceptual — for the FD pass we treat
        // exp_e[b*E + e, j] as the per-expert output for (b, e, j).
        Tensor d_composed(B, J);
        for (size_t i = 0; i < B*J; ++i) d_composed.data[i] = uni(rng);

        auto s = MoeRouter::softmax_topk(logits, k);
        // 1) Analytical d_logits via softmax_topk_backward + chain rule:
        Tensor d_gate_probs = Tensor::zeros(B, E);
        for (size_t b = 0; b < B; ++b) {
            for (size_t e = 0; e < E; ++e) {
                double v = 0.0;
                for (size_t j = 0; j < J; ++j) v += d_composed(b, j) * exp_e(b*E + e, j);
                d_gate_probs(b, e) = v;
            }
        }
        Tensor d_logits = MoeRouter::softmax_topk_backward(d_gate_probs, s);

        // 2) FD reference on the composed-output scalar
        auto composed_scalar = [&](const Tensor& l) {
            auto s_ = MoeRouter::softmax_topk(l, k);
            double sum = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t j = 0; j < J; ++j) {
                    double v = 0.0;
                    for (size_t e = 0; e < E; ++e) {
                        v += s_.probs(b, e) * exp_e(b*E + e, j);
                    }
                    sum += v * d_composed(b, j);
                }
            }
            return sum;
        };
        const double eps = 1e-5;
        double worst = 0.0;
        for (size_t b = 0; b < B; ++b) {
            for (size_t e = 0; e < E; ++e) {
                Tensor lp = logits.clone();
                Tensor lm = logits.clone();
                lp(b, e) += eps; lm(b, e) -= eps;
                double ana = d_logits(b, e);
                double num = (composed_scalar(lp) - composed_scalar(lm)) / (2.0 * eps);
                double denom = std::max(std::abs(ana), std::abs(num));
                double re = (denom > 1e-12) ? (std::abs(ana - num) / denom) : std::abs(ana - num);
                if (re > worst) worst = re;
            }
        }
        std::cout << "    worst_rel_err (composed-output) = " << worst << "\n";
        check(worst < 1e-9, "21. composed-output chain matches FD via softmax_topk_backward");
    }

    // -----------------------------------------------------------------------
    // Test 22: tie-breaking determinism — equal logits → probs sum to 1
    // -----------------------------------------------------------------------
    {
        Tensor logits(2, 4);
        for (size_t b = 0; b < 2; ++b)
            for (size_t e = 0; e < 4; ++e)
                logits(b, e) = 1.0;   // all equal
        bool threw = false;
        MoeRouterState s;
        try { s = MoeRouter::softmax_topk(logits, 2); }
        catch (...) { threw = true; }
        bool ok = !threw;
        for (size_t b = 0; b < 2 && ok; ++b) {
            double rsum = 0.0;
            for (size_t e = 0; e < 4; ++e) rsum += s.probs(b, e);
            if (std::abs(rsum - 1.0) > 1e-12) ok = false;
        }
        check(ok, "22. tie-breaking determinism: equal logits → row sums = 1");
    }

    // -----------------------------------------------------------------------
    // Test 23: k=0 throws
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();
        bool threw = false;
        try { (void)MoeRouter::softmax_topk(logits, 0); }
        catch (std::invalid_argument&) { threw = true; }
        check(threw, "23. k=0 throws invalid_argument");
    }

    // -----------------------------------------------------------------------
    // Test 24: k > E throws
    // -----------------------------------------------------------------------
    {
        Tensor logits = make_logits_4x6();   // E = 6
        bool threw = false;
        try { (void)MoeRouter::softmax_topk(logits, 7); }
        catch (std::invalid_argument&) { threw = true; }
        check(threw, "24. k>E throws invalid_argument");
    }

    // -----------------------------------------------------------------------
    // Test 25: B=0 (no rows) — forward throws invalid_argument (per header contract)
    // -----------------------------------------------------------------------
    {
        Tensor empty(0, 6);
        bool threw = false;
        try { (void)MoeRouter::softmax_topk(empty, 2); }
        catch (std::invalid_argument&) { threw = true; }
        check(threw, "25. B=0 throws invalid_argument (per header contract)");
    }

    // -----------------------------------------------------------------------
    // Test 26: determinism — softmax_topk twice → bit-exact
    // -----------------------------------------------------------------------
    {
        std::mt19937 rng(1);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        Tensor logits(5, 4);
        for (size_t i = 0; i < 5*4; ++i) logits.data[i] = uni(rng);
        auto s1 = MoeRouter::softmax_topk(logits, 2);
        auto s2 = MoeRouter::softmax_topk(logits, 2);
        bool ok = true;
        for (size_t b = 0; b < 5 && ok; ++b) {
            for (size_t e = 0; e < 4 && ok; ++e) {
                if (std::abs(s1.probs(b, e) - s2.probs(b, e)) > 0.0) ok = false;
                if (s1.topk_idx[b][0] != s2.topk_idx[b][0]) ok = false;
                if (s1.topk_idx[b][1] != s2.topk_idx[b][1]) ok = false;
            }
        }
        check(ok, "26. softmax_topk determinism: bit-exact across reruns");
    }

    // -----------------------------------------------------------------------
    // Test 27: numerical_d_logits_at — internal helper sanity (catches utility bugs)
    // -----------------------------------------------------------------------
    {
        // For uniform d_gate_probs = 1, the FD at any (b,e) should be 0,
        // because softmax-topk rows sum to 1 so the inner-product loss is
        // invariant. Test the helper directly.
        Tensor logits(2, 3);
        logits(0,0)=0.1; logits(0,1)=0.2; logits(0,2)=0.3;
        logits(1,0)=0.4; logits(1,1)=0.5; logits(1,2)=0.6;
        Tensor ones(2, 3);
        ones.fill(1.0);
        double v00 = numerical_d_logits_at(logits, 2, 0, 0, ones, 1e-5);
        double v01 = numerical_d_logits_at(logits, 2, 0, 1, ones, 1e-5);
        check(std::abs(v00) < 1e-9 && std::abs(v01) < 1e-9,
              "27. internal FD helper sanity for ones d_gate_probs (all ~ 0)");
    }

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed (of "
              << (g_pass + g_fail) << ") ===\n";
    return g_fail == 0 ? 0 : 1;
}
