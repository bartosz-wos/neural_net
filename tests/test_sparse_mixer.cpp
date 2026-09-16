// test_sparse_mixer.cpp — Lee-Thorp & Ainslie, EMNLP Findings 2022
// "Sparse Mixers: Combining MoE and Mixing to build a more efficient BERT"
// (https://arxiv.org/abs/2205.12399)
//
// Three classes under test:
//   (1) LinearMixingSublayer — drop-in for self-attention in BERT-like
//       blocks. Math: y = T_h · T_seq · x where T_h : (d_model, d_model)
//       and T_seq : (seq_len, seq_len) are two dense token-INDEPENDENT
//       linear projections. No attention — just two matmuls.
//   (2) ExpertsChoiceMoE — Zhou et al. 2022 router. Each expert picks its
//       top-k tokens (k = capacity), so an expert ALWAYS fills its buffer
//       and a token may be routed to 0, 1, or multiple experts.
//   (3) SparseMixerBlock — pre-norm + mixing/attention sublayer +
//       residual + pre-norm + MoE/dense FFN sublayer + residual.
//   (4) SparseMixerModel — input proj → stack of blocks → mean-pool →
//       classifier.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "nn/layers/architectures/sparse_mixer.h"

using namespace std;

static double max_rel_err(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        double av = a.data[i], bv = b.data[i];
        double denom = max(fabs(av), fabs(bv));
        if (denom < 1e-8) denom = 1e-8;
        m = max(m, fabs(av - bv) / denom);
    }
    return m;
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

static void fill_random(Tensor& t, mt19937& gen, double scale = 0.3) {
    normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dis(gen);
}

// FD gradient check on the input to a callable (forward(input) -> output).
// Callable signature: Tensor(const Tensor&)
template <typename Callable>
static Tensor finite_diff_grad_input(Callable fn, Tensor input, const Tensor& target,
                                     double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input(i, j);
            input(i, j) = orig + eps;
            Tensor out_p = fn(input);
            double lp = l2_loss_value(out_p, target);
            input(i, j) = orig - eps;
            Tensor out_m = fn(input);
            double lm = l2_loss_value(out_m, target);
            input(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// FD gradient check on a parameter tensor — we mutate `param` while
// calling `fn` with the same `input`/`target` for both +eps and -eps.
template <typename Callable>
static Tensor finite_diff_grad_param(Callable fn, const Tensor& input, const Tensor& target,
                                     Tensor& param, double eps = 1e-5) {
    Tensor grad(param.rows, param.cols);
    for (size_t i = 0; i < param.rows; ++i) {
        for (size_t j = 0; j < param.cols; ++j) {
            double orig = param(i, j);
            param(i, j) = orig + eps;
            Tensor out_p = fn(input);
            double lp = l2_loss_value(out_p, target);
            param(i, j) = orig - eps;
            Tensor out_m = fn(input);
            double lm = l2_loss_value(out_m, target);
            param(i, j) = orig;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

int main() {
    cout << "=== Sparse Mixer Tests ===" << endl;
    cout.setf(ios::unitbuf);
    int total = 0, passed = 0;

    auto check = [&](bool ok, const string& msg) {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << msg << "\n"; }
        else    { cout << "[FAIL] " << msg << "\n"; }
    };

    mt19937 gen(42);

    // =====================================================================
    // LinearMixingSublayer tests
    // =====================================================================
    cout << "\n--- LinearMixingSublayer ---\n";

    // Test 1: Constructor validation
    {
        bool threw1 = false, threw2 = false, threw3 = false;
        try { LinearMixingSublayer l(0, 4); } catch (...) { threw1 = true; }
        try { LinearMixingSublayer l(4, 0); } catch (...) { threw2 = true; }
        try {
            LinearMixingSublayer l(4, 4);
            (void)l;
        } catch (...) { threw3 = true; }
        check(threw1, "constructor rejects d_model=0");
        check(threw2, "constructor rejects seq_len=0");
        check(!threw3, "constructor accepts valid (4,4)");
    }

    // Test 2: forward shape (B, S*D) -> (B, S*D) and finite/nonzero
    {
        size_t B = 2, S = 4, D = 6;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.2);

        LinearMixingSublayer mix(D, S);
        Tensor out = mix.forward(input);
        check(out.rows == B && out.cols == S * D, "forward shape preserved (B=2, S=4, D=6)");
        bool finite = true;
        double nonzero_norm = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i) {
            if (!isfinite(out.data[i])) finite = false;
            nonzero_norm += out.data[i] * out.data[i];
        }
        check(finite, "output is finite");
        check(nonzero_norm > 1e-3, "output is nonzero");
    }

    // Test 3: identity-W identity (W_h=I, W_seq=I, b=0) reproduces x exactly
    // Setup: x = (1, 4), T_h = I_4, T_seq = I_2, b = 0 → out = T_h · T_seq · x_flat_2x4
    // Actually the layer reshapes: forward((B, S*D)) does x_post_seq = x @ T_seq^T (then x_post_h = x_post_seq @ T_h^T).
    // For T_seq = I, T_h = I, bias=0: out must equal x exactly.
    {
        size_t B = 1, S = 2, D = 3;
        Tensor input(B, S * D);
        // set explicit values
        input(0,0) = 1.0; input(0,1) = 2.0; input(0,2) = 3.0;
        input(0,3) = 4.0; input(0,4) = 5.0; input(0,5) = 6.0;
        LinearMixingSublayer mix(D, S);
        // Force identity: W_h = I_3, W_seq = I_2, bias=0.
        for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j)
            mix.mix_W_h()(i, j) = (i == j) ? 1.0 : 0.0;
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j)
            mix.mix_W_seq()(i, j) = (i == j) ? 1.0 : 0.0;
        mix.mix_b_h().fill(0.0);
        Tensor out = mix.forward(input);
        check(max_rel_err(out, input) < 1e-12, "identity weights give identity forward");
    }

    // Test 4: input gradient FD vs analytical
    {
        size_t B = 1, S = 3, D = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);

        LinearMixingSublayer mix(D, S);
        // Randomize all weights non-trivially
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);

        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        // Analytical gradient
        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);
        Tensor ana = mix.grad_input();

        // FD gradient
        Tensor fd = finite_diff_grad_input(
            [&](const Tensor& x) { return mix.forward(x); },
            input, target);

        double rel = max_rel_err(ana, fd);
        cout << "  input grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-4, "input grad FD rel_err < 1e-4");
    }

    // Test 5: W_seq FD vs analytical
    {
        size_t B = 1, S = 3, D = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);

        LinearMixingSublayer mix(D, S);
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);

        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);
        Tensor ana = mix.grad_mix_W_seq();

        Tensor fd = finite_diff_grad_param(
            [&](const Tensor& x) { return mix.forward(x); },
            input, target, mix.mix_W_seq());

        double rel = max_rel_err(ana, fd);
        cout << "  W_seq grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-4, "W_seq grad FD rel_err < 1e-4");
    }

    // Test 6: W_h FD vs analytical
    {
        size_t B = 1, S = 3, D = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);

        LinearMixingSublayer mix(D, S);
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);

        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);
        Tensor ana = mix.grad_mix_W_h();

        Tensor fd = finite_diff_grad_param(
            [&](const Tensor& x) { return mix.forward(x); },
            input, target, mix.mix_W_h());

        double rel = max_rel_err(ana, fd);
        cout << "  W_h grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-4, "W_h grad FD rel_err < 1e-4");
    }

    // Test 7: b_h FD vs analytical
    {
        size_t B = 1, S = 3, D = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);

        LinearMixingSublayer mix(D, S);
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);

        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);
        Tensor ana = mix.grad_mix_b_h();

        Tensor fd = finite_diff_grad_param(
            [&](const Tensor& x) { return mix.forward(x); },
            input, target, mix.mix_b_h());

        double rel = max_rel_err(ana, fd);
        cout << "  b_h grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-4, "b_h grad FD rel_err < 1e-4");
    }

    // Test 8: zero_grad clears all gradients
    {
        LinearMixingSublayer mix(4, 3);
        Tensor input(2, 12);
        fill_random(input, gen, 0.3);
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);

        Tensor target(2, 12);
        fill_random(target, gen, 0.3);

        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);

        // Check that all 4 gradients are nonzero
        double norm = 0.0;
        for (double v : mix.grad_input().data) norm += v*v;
        for (double v : mix.grad_mix_W_h().data) norm += v*v;
        for (double v : mix.grad_mix_W_seq().data) norm += v*v;
        for (double v : mix.grad_mix_b_h().data) norm += v*v;
        check(norm > 1e-6, "gradients are nonzero before zero_grad");

        mix.zero_grad();
        double norm2 = 0.0;
        for (double v : mix.grad_input().data) norm2 += v*v;
        for (double v : mix.grad_mix_W_h().data) norm2 += v*v;
        for (double v : mix.grad_mix_W_seq().data) norm2 += v*v;
        for (double v : mix.grad_mix_b_h().data) norm2 += v*v;
        check(norm2 < 1e-12, "zero_grad clears all gradients");
    }

    // Test 9: update_weights moves parameters
    {
        LinearMixingSublayer mix(3, 2);
        Tensor input(1, 6);
        fill_random(input, gen, 0.3);
        fill_random(mix.mix_W_h(), gen, 0.2);
        fill_random(mix.mix_W_seq(), gen, 0.2);
        fill_random(mix.mix_b_h(), gen, 0.1);
        Tensor target(1, 6);
        fill_random(target, gen, 0.3);

        Tensor W_h_before = mix.mix_W_h().clone();
        Tensor W_seq_before = mix.mix_W_seq().clone();
        Tensor b_h_before = mix.mix_b_h().clone();

        Tensor out = mix.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        mix.backward(grad_out, 0.0);
        mix.update_weights(0.1);

        double moved_W_h = 0.0, moved_W_seq = 0.0, moved_b_h = 0.0;
        for (size_t i = 0; i < mix.mix_W_h().data.size(); ++i)
            moved_W_h += (mix.mix_W_h().data[i] - W_h_before.data[i]) *
                         (mix.mix_W_h().data[i] - W_h_before.data[i]);
        for (size_t i = 0; i < mix.mix_W_seq().data.size(); ++i)
            moved_W_seq += (mix.mix_W_seq().data[i] - W_seq_before.data[i]) *
                           (mix.mix_W_seq().data[i] - W_seq_before.data[i]);
        for (size_t i = 0; i < mix.mix_b_h().data.size(); ++i)
            moved_b_h += (mix.mix_b_h().data[i] - b_h_before.data[i]) *
                         (mix.mix_b_h().data[i] - b_h_before.data[i]);
        check(moved_W_h > 1e-6, "update_weights moves W_h");
        check(moved_W_seq > 1e-6, "update_weights moves W_seq");
        check(moved_b_h > 1e-6, "update_weights moves b_h");
    }

    // =====================================================================
    // ExpertsChoiceMoE tests
    // =====================================================================
    cout << "\n--- ExpertsChoiceMoE ---\n";

    // Test 10: Constructor validation
    {
        bool t1 = false, t2 = false, t3 = false;
        try { ExpertsChoiceMoE m(0, 4); } catch (...) { t1 = true; }
        try { ExpertsChoiceMoE m(8, 0); } catch (...) { t2 = true; }
        try {
            ExpertsChoiceMoE m(8, 4);
            (void)m;
        } catch (...) { t3 = true; }
        check(t1, "constructor rejects d_model=0");
        check(t2, "constructor rejects num_experts=0");
        check(!t3, "constructor accepts valid (8,4)");
    }

    // Test 11: Forward shape, finite, nonzero
    {
        size_t B = 4, D = 6, E = 3;
        Tensor input(B, D);
        fill_random(input, gen, 0.2);
        ExpertsChoiceMoE moe(D, E);
        Tensor out = moe.forward(input);
        check(out.rows == B && out.cols == D, "forward shape preserved (B=4, D=6)");
        bool finite = true; double nz = 0.0;
        for (double v : out.data) { if (!isfinite(v)) finite = false; nz += v*v; }
        check(finite, "output finite");
        check(nz > 1e-6, "output nonzero");
    }

    // Test 12: experts fill capacity — every expert processes exactly `cap` tokens
    {
        size_t B = 6, D = 4, E = 3;
        double cf = 1.0;
        ExpertsChoiceMoE moe(D, E, /*expert_hidden=*/0, cf);
        Tensor input(B, D);
        fill_random(input, gen, 0.3);
        moe.forward(input);
        size_t cap = (size_t)ceil((double)B / (double)E) * 1;  // cf=1.0
        size_t total_assigned = 0;
        for (size_t e = 0; e < E; ++e) {
            size_t cnt = moe.last_expert_assignment(e).size();
            total_assigned += cnt;
            check(cnt == cap, "expert " + to_string(e) + " fills to capacity " + to_string(cap));
        }
        check(total_assigned == E * cap, "total assigned == E * cap");
    }

    // Test 13: EC signature — capacity_factor controls cap linearly
    {
        size_t B = 6, D = 4, E = 3;
        Tensor input(B, D);
        fill_random(input, gen, 0.3);

        ExpertsChoiceMoE moe_lo(D, E, 0, 0.5);  // cf=0.5 → cap=1
        moe_lo.forward(input);
        size_t cap_lo = moe_lo.last_expert_assignment(0).size();
        check(cap_lo == 1, "cf=0.5 → cap=1");

        ExpertsChoiceMoE moe_hi(D, E, 0, 2.0);  // cf=2.0 → cap=4
        moe_hi.forward(input);
        size_t cap_hi = moe_hi.last_expert_assignment(0).size();
        check(cap_hi == 4, "cf=2.0 → cap=4");
    }

    // Test 14: input gradient FD vs analytical
    {
        size_t B = 4, D = 3, E = 2;
        ExpertsChoiceMoE moe(D, E);
        Tensor input(B, D);
        fill_random(input, gen, 0.3);
        fill_random(moe.router_W(), gen, 0.2);
        fill_random(moe.router_b(), gen, 0.1);

        Tensor target(B, D);
        fill_random(target, gen, 0.3);

        Tensor out = moe.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        moe.backward(grad_out, 0.0);
        Tensor ana = moe.grad_input();

        Tensor fd = finite_diff_grad_input(
            [&](const Tensor& x) { return moe.forward(x); },
            input, target);

        double rel = max_rel_err(ana, fd);
        cout << "  ExpertsChoiceMoE input grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-3, "input grad FD rel_err < 1e-3 (EC is non-trivial topology)");
    }

    // Test 15: W_router gradient is exactly zero (hard selection → no main-loss gradient)
    // This is a property of Experts-Choice routing. A load-balancing aux loss
    // would be the standard way to provide gradient to the router.
    {
        size_t B = 4, D = 3, E = 2;
        ExpertsChoiceMoE moe(D, E);
        Tensor input(B, D);
        fill_random(input, gen, 0.3);
        fill_random(moe.router_W(), gen, 0.2);
        fill_random(moe.router_b(), gen, 0.1);

        Tensor target(B, D);
        fill_random(target, gen, 0.3);

        Tensor out = moe.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        moe.backward(grad_out, 0.0);
        Tensor ana = moe.grad_router_W();

        // Verify analytical == 0 exactly
        double norm = 0.0;
        for (double v : ana.data) norm += v * v;
        cout << "  ExpertsChoiceMoE grad_router_W norm = " << scientific << setprecision(2) << norm << "\n";
        check(norm < 1e-12, "W_router gradient is exactly zero (hard EC routing)");
    }

    // Test 16: expert W1 FD vs analytical (verify expert backward flows)
    {
        size_t B = 4, D = 3, E = 2, H = 8;
        ExpertsChoiceMoE moe(D, E, H);
        Tensor input(B, D);
        fill_random(input, gen, 0.3);
        fill_random(moe.router_W(), gen, 0.2);
        fill_random(moe.router_b(), gen, 0.1);
        for (size_t e = 0; e < E; ++e) {
            fill_random(moe.expert_W1(e), gen, 0.2);
            fill_random(moe.expert_b1(e), gen, 0.1);
            fill_random(moe.expert_W2(e), gen, 0.2);
            fill_random(moe.expert_b2(e), gen, 0.1);
        }

        Tensor target(B, D);
        fill_random(target, gen, 0.3);

        Tensor out = moe.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        moe.backward(grad_out, 0.0);

        // Test on expert 0's W1
        Tensor& param = moe.expert_W1(0);
        Tensor fd = finite_diff_grad_param(
            [&](const Tensor& x) { return moe.forward(x); },
            input, target, param);
        Tensor ana = moe.grad_expert_W1(0);

        double rel = max_rel_err(ana, fd);
        cout << "  expert 0 W1 grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-3, "expert 0 W1 grad FD rel_err < 1e-3");
    }

    // =====================================================================
    // SparseMixerBlock tests
    // =====================================================================
    cout << "\n--- SparseMixerBlock ---\n";

    // Test 17: constructor validation (LINEAR_MIXING + DENSE)
    {
        bool t1 = false, t2 = false, t3 = false;
        try {
            SparseMixerBlock b(SparseMixerBlock::LINEAR_MIXING,
                               0, 4, /*num_experts=*/0, 0, 1.0, 4);
        } catch (...) { t1 = true; }
        try {
            SparseMixerBlock b(SparseMixerBlock::LINEAR_MIXING,
                               4, 0, /*num_experts=*/0, 0, 1.0, 4);
        } catch (...) { t2 = true; }
        try {
            SparseMixerBlock b(SparseMixerBlock::LINEAR_MIXING,
                               4, 4, /*num_experts=*/0, 0, 1.0, 4);
            (void)b;
        } catch (...) { t3 = true; }
        check(t1, "constructor rejects d_model=0");
        check(t2, "constructor rejects seq_len=0");
        check(!t3, "constructor accepts valid LINEAR_MIXING+DENSE");
    }

    // Test 18: forward shape + finite for LINEAR_MIXING + DENSE
    {
        size_t B = 2, S = 4, D = 6;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.2);
        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/0, 0, 1.0, 4);
        Tensor out = blk.forward(input);
        check(out.rows == B && out.cols == S * D, "LM+DENSE forward shape");
        bool finite = true; for (double v : out.data) if (!isfinite(v)) finite = false;
        check(finite, "LM+DENSE forward finite");
    }

    // Test 19: forward shape + finite for LINEAR_MIXING + MoE
    {
        size_t B = 3, S = 4, D = 5, E = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.2);
        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/E, 0, 1.0, 4);
        Tensor out = blk.forward(input);
        check(out.rows == B && out.cols == S * D, "LM+MoE forward shape");
        bool finite = true; for (double v : out.data) if (!isfinite(v)) finite = false;
        check(finite, "LM+MoE forward finite");
    }

    // Test 20: forward shape + finite for SELF_ATTENTION + DENSE
    {
        size_t B = 2, S = 4, D = 6;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.2);
        SparseMixerBlock blk(SparseMixerBlock::SELF_ATTENTION, D, S,
                             /*num_experts=*/0, 0, 1.0, 4);
        Tensor out = blk.forward(input);
        check(out.rows == B && out.cols == S * D, "SA+DENSE forward shape");
        bool finite = true; for (double v : out.data) if (!isfinite(v)) finite = false;
        check(finite, "SA+DENSE forward finite");
    }

    // Test 21: input FD for LINEAR_MIXING + DENSE block
    {
        size_t B = 1, S = 3, D = 4;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);
        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/0, 0, 1.0, 4);
        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        Tensor out = blk.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        blk.backward(grad_out, 0.0);
        Tensor ana = blk.grad_input();

        Tensor fd = finite_diff_grad_input(
            [&](const Tensor& x) { return blk.forward(x); },
            input, target);

        double rel = max_rel_err(ana, fd);
        cout << "  block LM+DENSE input grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-4, "LM+DENSE block input FD rel_err < 1e-4");
    }

    // Test 22: input FD for LINEAR_MIXING + MoE block
    {
        size_t B = 2, S = 3, D = 3, E = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);
        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/E, 0, 1.0, 4);
        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        Tensor out = blk.forward(input);
        Tensor grad_out = l2_loss_grad(out, target);
        blk.backward(grad_out, 0.0);
        Tensor ana = blk.grad_input();

        Tensor fd = finite_diff_grad_input(
            [&](const Tensor& x) { return blk.forward(x); },
            input, target);

        double rel = max_rel_err(ana, fd);
        cout << "  block LM+MoE input grad rel_err = " << scientific << setprecision(2) << rel << "\n";
        check(rel < 1e-3, "LM+MoE block input FD rel_err < 1e-3");
    }

    // Test 23: training reduces loss over 30 SGD steps for LM+DENSE
    {
        size_t B = 2, S = 3, D = 4;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/0, 0, 1.0, 4);

        Tensor out = blk.forward(input);
        double L0 = l2_loss_value(out, target);
        for (int step = 0; step < 30; ++step) {
            out = blk.forward(input);
            double L = l2_loss_value(out, target);
            Tensor g = l2_loss_grad(out, target);
            blk.backward(g, 0.01);
            blk.update_weights(0.01);
        }
        double Lf = l2_loss_value(blk.forward(input), target);
        cout << "  LM+Dense L0=" << fixed << setprecision(4) << L0
             << " Lf=" << Lf << "\n";
        check(Lf < L0, "training reduces loss (LM+Dense)");
    }

    // Test 24: training reduces loss over 30 SGD steps for LM+MoE
    {
        size_t B = 2, S = 3, D = 3, E = 2;
        Tensor input(B, S * D);
        fill_random(input, gen, 0.3);
        Tensor target(B, S * D);
        fill_random(target, gen, 0.3);

        SparseMixerBlock blk(SparseMixerBlock::LINEAR_MIXING, D, S,
                             /*num_experts=*/E, 0, 1.0, 4);

        Tensor out = blk.forward(input);
        double L0 = l2_loss_value(out, target);
        for (int step = 0; step < 30; ++step) {
            out = blk.forward(input);
            double L = l2_loss_value(out, target);
            Tensor g = l2_loss_grad(out, target);
            blk.backward(g, 0.01);
            blk.update_weights(0.01);
        }
        double Lf = l2_loss_value(blk.forward(input), target);
        cout << "  LM+MoE L0=" << fixed << setprecision(4) << L0
             << " Lf=" << Lf << "\n";
        check(Lf < L0, "training reduces loss (LM+MoE)");
    }

    // =====================================================================
    // SparseMixerModel tests
    // =====================================================================
    cout << "\n--- SparseMixerModel ---\n";

    // Test 25: constructor validation
    {
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { SparseMixerModel m(0, 4, 2, 3, 2, 1, 0, 0, 1.0, 4); } catch (...) { t1 = true; }
        try { SparseMixerModel m(2, 0, 2, 3, 2, 1, 0, 0, 1.0, 4); } catch (...) { t2 = true; }
        try { SparseMixerModel m(2, 4, 0, 3, 2, 1, 0, 0, 1.0, 4); } catch (...) { t3 = true; }
        try { SparseMixerModel m(2, 4, 2, 0, 2, 1, 0, 0, 1.0, 4); } catch (...) { t4 = true; }
        check(t1, "constructor rejects input_dim=0");
        check(t2, "constructor rejects d_model=0");
        check(t3, "constructor rejects output_dim=0");
        check(t4, "constructor rejects seq_len=0");
    }

    // Test 26: forward shape + finite
    {
        size_t B = 2, in_dim = 4, d_model = 6, out_dim = 3, seq_len = 3;
        size_t num_layers = 3, num_attn = 1;
        Tensor input(B, in_dim * seq_len);
        fill_random(input, gen, 0.2);
        SparseMixerModel m(in_dim, d_model, out_dim, seq_len,
                           num_layers, num_attn, /*num_experts=*/0, 0, 1.0, 4);
        Tensor out = m.forward(input);
        check(out.rows == B && out.cols == out_dim,
              "SparseMixerModel forward shape (B=" + to_string(B) + ", out_dim=" + to_string(out_dim) + ")");
        bool finite = true; for (double v : out.data) if (!isfinite(v)) finite = false;
        check(finite, "SparseMixerModel forward finite");
    }

    // Test 27: training reduces loss over 30 SGD steps
    {
        size_t B = 2, in_dim = 4, d_model = 4, out_dim = 3, seq_len = 3;
        size_t num_layers = 2, num_attn = 1;
        Tensor input(B, in_dim * seq_len);
        fill_random(input, gen, 0.3);
        Tensor target(B, out_dim);
        fill_random(target, gen, 0.3);

        SparseMixerModel m(in_dim, d_model, out_dim, seq_len,
                           num_layers, num_attn, /*num_experts=*/0, 0, 1.0, 4);

        Tensor out = m.forward(input);
        double L0 = l2_loss_value(out, target);
        for (int step = 0; step < 30; ++step) {
            out = m.forward(input);
            Tensor g = l2_loss_grad(out, target);
            m.backward(g, 0.01);
            m.update_weights(0.01);
        }
        double Lf = l2_loss_value(m.forward(input), target);
        cout << "  SparseMixerModel L0=" << fixed << setprecision(4) << L0
             << " Lf=" << Lf << "\n";
        check(Lf < L0, "SparseMixerModel training reduces loss");
    }

    // Test 28: SparseMixerModel with MoE in middle layers (1 LM+1 SA+1 LM-MoE)
    {
        size_t B = 2, in_dim = 3, d_model = 4, out_dim = 2, seq_len = 3;
        size_t num_layers = 3, num_attn = 1;
        Tensor input(B, in_dim * seq_len);
        fill_random(input, gen, 0.2);
        SparseMixerModel m(in_dim, d_model, out_dim, seq_len,
                           num_layers, num_attn, /*num_experts=*/2, 0, 1.0, 4);
        Tensor out = m.forward(input);
        check(out.rows == B && out.cols == out_dim,
              "SparseMixerModel + MoE forward shape");
        bool finite = true; for (double v : out.data) if (!isfinite(v)) finite = false;
        check(finite, "SparseMixerModel + MoE forward finite");
    }

    // Test 29: zero_grad clears all gradients on the model
    {
        SparseMixerModel m(3, 4, 2, 2, 2, 1, 0, 0, 1.0, 4);
        Tensor input(2, 6);
        fill_random(input, gen, 0.3);
        Tensor target(2, 2);
        fill_random(target, gen, 0.3);
        Tensor out = m.forward(input);
        m.backward(l2_loss_grad(out, target), 0.0);

        // Spot check one parameter
        double norm_before = 0.0;
        for (Tensor* g : m.gradients()) for (double v : g->data) norm_before += v*v;
        check(norm_before > 1e-6, "gradients nonzero before zero_grad");

        m.zero_grad();
        double norm_after = 0.0;
        for (Tensor* g : m.gradients()) for (double v : g->data) norm_after += v*v;
        check(norm_after < 1e-12, "zero_grad clears all gradients");
    }

    // =====================================================================
    // Summary
    // =====================================================================
    cout << "\n========================================" << endl;
    cout << "  Sparse Mixer Tests: " << passed << " / " << total << " passed" << endl;
    cout << "========================================" << endl;

    return (passed == total) ? 0 : 1;
}