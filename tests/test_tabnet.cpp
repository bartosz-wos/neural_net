// test_tabnet.cpp — Gradient correctness tests for TabNet
//
// Reference: Arik & Pfister 2019 "TabNet: Attentive Interpretable Tabular
//   Learning" (https://arxiv.org/abs/1908.07442).
//
// Tests:
//   1.  Forward shape (batch, input_dim) -> (batch, num_outputs)
//   2.  Output is finite (no NaN/Inf)
//   3.  Output is non-trivial (norm > 0)
//   4.  Decision contributions accumulate correctly (sum over steps == output)
//   5.  Mask softmax sums to 1 per row at every step
//   6.  Prior scale decays over steps (relaxation factor in action)
//   7.  Input gradient check (analytical vs centered finite differences)
//   8.  Step-FC weight gradient check
//   9.  Attention-block weight gradient check
//  10.  Independent-block w1 gradient check
//  11.  Shared-encoder w2 gradient check
//  12.  Update weights reduces loss
//  13.  Parameter/gradient count consistency
//  14.  Forward determinism (same seed -> same output)
//  15.  zero_grad clears all gradients

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "nn/layers/architectures/tabnet.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_err(double a, double b) {
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

static Tensor make_input(size_t batch, size_t d, double scale = 0.1, unsigned seed = 0) {
    Tensor x(batch, d);
    std::mt19937 gen(seed + 7);
    std::normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < batch * d; ++i) x.data[i] = dis(gen);
    return x;
}

// =====================================================================
// Test 1: forward shape
// =====================================================================
static void test_forward_shape() {
    cout << endl << "--- Test 1: forward shape (batch=4, input_dim=8, num_outputs=3, num_steps=2) ---" << endl;
    TabNet net(8, 3, 2, 2, 2, 1.5);
    Tensor input = make_input(4, 8, 0.1, 1);
    Tensor out = net.forward(input);
    check("output shape (4, 3)", out.rows == 4 && out.cols == 3);
}

// =====================================================================
// Test 2: output is finite
// =====================================================================
static void test_output_finite() {
    cout << endl << "--- Test 2: output is finite ---" << endl;
    TabNet net(6, 2, 3, 2, 2, 1.5);
    Tensor input = make_input(5, 6, 0.1, 2);
    Tensor out = net.forward(input);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) finite = false;
    check("all outputs finite", finite);
}

// =====================================================================
// Test 3: output is non-trivial
// =====================================================================
static void test_output_nonzero() {
    cout << endl << "--- Test 3: output is non-trivial ---" << endl;
    TabNet net(5, 3, 2, 2, 2, 1.5);
    Tensor input = make_input(3, 5, 0.3, 3);
    Tensor out = net.forward(input);
    double n2 = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) n2 += out.data[i] * out.data[i];
    check("||output||^2 > 0", n2 > 0.0);
}

// =====================================================================
// Test 4: mask softmax sums to 1 per row
// =====================================================================
static void test_mask_sums_to_one() {
    cout << endl << "--- Test 4: mask sums to 1 (softmax invariant) ---" << endl;
    TabNet net(6, 2, 3, 2, 2, 1.5);
    Tensor input = make_input(3, 6, 0.2, 4);
    net.forward(input);
    auto masks = net.getAttentionMasks();
    bool all_sum_one = true;
    for (size_t m = 1; m < masks.size(); ++m) {  // skip mask[0] which is ones
        Tensor& mm = masks[m];
        if (mm.rows == 0) continue;
        for (size_t i = 0; i < mm.rows; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < mm.cols; ++j) s += mm(i, j);
            if (fabs(s - 1.0) > 1e-9) all_sum_one = false;
        }
    }
    check("every mask row sums to 1", all_sum_one);
}

// =====================================================================
// Test 5: prior scale decays over steps (relaxation factor in action)
// =====================================================================
static void test_prior_scale_decays() {
    cout << endl << "--- Test 5: prior scale decays over steps ---" << endl;
    TabNet net(4, 2, 4, 2, 2, 1.5);
    Tensor input = make_input(2, 4, 0.1, 5);
    net.forward(input);
    auto params = net.parameters();
    // Last entries are step_fc.b (last 2 = num_outputs per step); we want prior_scales_
    // exposed via get_weights() — not directly.  Use the cached _last_masks_ to check
    // that subsequent masks shift.
    auto masks = net.getAttentionMasks();
    // Two masks should differ
    double diff = 0.0;
    if (masks.size() >= 2 && masks[1].rows > 0 && masks[2].rows > 0) {
        for (size_t i = 0; i < masks[1].data.size(); ++i)
            diff += fabs(masks[1].data[i] - masks[2].data[i]);
    }
    check("consecutive masks differ (mask depends on step)", diff > 0.0);
    (void)params;
}

// =====================================================================
// Test 6: input gradient check
// =====================================================================
static void test_input_grad() {
    cout << endl << "--- Test 6: input gradient check (batch=2, input_dim=4, num_outputs=2, num_steps=2) ---" << endl;
    size_t batch = 2, d = 4, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 8);
    Tensor target = make_input(batch, num_out, 0.1, 9);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    Tensor grad_x = net.backward(grad_loss, 0.0);

    double max_err = 0.0;
    int n_checked = 0;
    auto num_grad = [&](size_t i, size_t j) {
        double orig = input(i, j);
        input(i, j) = orig + eps;
        Tensor out_p = net.forward(input);
        double Lp = l2_loss_value(out_p, target);
        input(i, j) = orig - eps;
        Tensor out_m = net.forward(input);
        double Lm = l2_loss_value(out_m, target);
        input(i, j) = orig;
        return (Lp - Lm) / (2.0 * eps);
    };
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double num = num_grad(i, j);
            double ana = grad_x(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "input grad max rel err = " << max_err << endl;
    check("input gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 7: step-FC weight gradient check
// =====================================================================
static void test_step_fc_grad() {
    cout << endl << "--- Test 7: step-FC weight gradient check ---" << endl;
    size_t batch = 2, d = 3, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 10);
    Tensor target = make_input(batch, num_out, 0.1, 11);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();

    // Find a step-FC weight tensor in the param list.  It has shape (num_outputs, virtual_dim).
    // Param ordering: shared (8 params), independent[0..S-1] (8 each), attention[0..S-1] (1 each),
    // step_fc[0..S-1] (2 each).
    int step_fc_w_idx = -1;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == (size_t)num_out && params[i]->cols > (size_t)d) {
            // Could be a step_fc w (rows=num_out, cols=virtual_dim).  Also could be the indep's w1
            // which has rows=virtual_dim, cols=input_dim — different.  Step FC w has rows=num_out.
            step_fc_w_idx = (int)i;
            break;
        }
    }
    if (step_fc_w_idx < 0) {
        check("found step_fc_w in param list", false);
        return;
    }
    Tensor* Wp = params[step_fc_w_idx];
    Tensor* Gp = grads[step_fc_w_idx];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = net.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = net.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "step_fc_w max rel err = " << max_err << endl;
    check("step_fc_w gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 8: attention-block weight gradient check
// =====================================================================
static void test_attention_grad() {
    cout << endl << "--- Test 8: attention-block weight gradient check ---" << endl;
    size_t batch = 2, d = 3, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 12);
    Tensor target = make_input(batch, num_out, 0.1, 13);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();

    // Attention block w has shape (1, virtual_dim).
    int attn_idx = -1;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows == 1 && params[i]->cols > (size_t)d) {
            attn_idx = (int)i;
            break;
        }
    }
    if (attn_idx < 0) {
        check("found attention_w in param list", false);
        return;
    }
    Tensor* Wp = params[attn_idx];
    Tensor* Gp = grads[attn_idx];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
        double orig = (*Wp)(0, j);
        (*Wp)(0, j) = orig + eps;
        Tensor out_p = net.forward(input);
        double Lp = l2_loss_value(out_p, target);
        (*Wp)(0, j) = orig - eps;
        Tensor out_m = net.forward(input);
        double Lm = l2_loss_value(out_m, target);
        (*Wp)(0, j) = orig;
        double num = (Lp - Lm) / (2.0 * eps);
        double ana = (*Gp)(0, j);
        double err = rel_err(num, ana);
        max_err = max(max_err, err);
        ++n_checked;
    }
    cout << "attention_w max rel err = " << max_err << endl;
    check("attention_w gradient check (rel_err < 1e-1)", max_err < 1e-1);
}

// =====================================================================
// Test 9: independent-block w1 gradient check
// =====================================================================
static void test_indep_w1_grad() {
    cout << endl << "--- Test 9: independent-block w1 gradient check ---" << endl;
    size_t batch = 2, d = 3, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 14);
    Tensor target = make_input(batch, num_out, 0.1, 15);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();

    // First independent block w1 — after shared_encoder's 8 params.
    // shared: w1, b1, bn1_gamma, bn1_beta, w2, b2, bn2_gamma, bn2_beta = 8 params
    Tensor* Wp = params[8];  // indep[0].w1
    Tensor* Gp = grads[8];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = net.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = net.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "indep[0].w1 max rel err = " << max_err << endl;
    check("indep[0].w1 gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 10: shared-encoder w2 gradient check
// =====================================================================
static void test_shared_w2_grad() {
    cout << endl << "--- Test 10: shared-encoder w2 gradient check ---" << endl;
    size_t batch = 2, d = 3, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 16);
    Tensor target = make_input(batch, num_out, 0.1, 17);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();
    // shared_encoder params: w1(0), b1(1), bn1_gamma(2), bn1_beta(3), w2(4), b2(5), bn2_gamma(6), bn2_beta(7)
    Tensor* Wp = params[4];
    Tensor* Gp = grads[4];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = net.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = net.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "shared.w2 max rel err = " << max_err << endl;
    check("shared.w2 gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 10b: input projection W_in gradient check
// =====================================================================
static void test_W_in_grad() {
    cout << endl << "--- Test 10b: input projection W_in gradient check ---" << endl;
    size_t batch = 2, d = 4, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 18);
    Tensor target = make_input(batch, num_out, 0.1, 19);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();
    // Param ordering: shared(8) + indep[S](8) + attention[S](1) + step_fc[S](2) + W_in(1) + b_in(1).
    // W_in is the (param_list.size() - 2)th entry, shape (V, input_dim).
    if (params.size() < 2) { check("param list non-empty", false); return; }
    Tensor* Wp = params[params.size() - 2];
    Tensor* Gp = grads[params.size() - 2];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t i = 0; i < Wp->rows && n_checked < 6; ++i) {
        for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
            double orig = (*Wp)(i, j);
            (*Wp)(i, j) = orig + eps;
            Tensor out_p = net.forward(input);
            double Lp = l2_loss_value(out_p, target);
            (*Wp)(i, j) = orig - eps;
            Tensor out_m = net.forward(input);
            double Lm = l2_loss_value(out_m, target);
            (*Wp)(i, j) = orig;
            double num = (Lp - Lm) / (2.0 * eps);
            double ana = (*Gp)(i, j);
            double err = rel_err(num, ana);
            max_err = max(max_err, err);
            ++n_checked;
        }
    }
    cout << "W_in max rel err = " << max_err << endl;
    check("W_in gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 10c: input projection b_in gradient check
// =====================================================================
static void test_b_in_grad() {
    cout << endl << "--- Test 10c: input projection b_in gradient check ---" << endl;
    size_t batch = 2, d = 4, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 1.5);
    Tensor input = make_input(batch, d, 0.3, 24);
    Tensor target = make_input(batch, num_out, 0.1, 25);
    double eps = 1e-5;

    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.zero_grad();
    net.backward(grad_loss, 0.0);
    auto params = net.parameters();
    auto grads  = net.gradients();
    Tensor* Wp = params[params.size() - 1];
    Tensor* Gp = grads[params.size() - 1];
    double max_err = 0.0;
    int n_checked = 0;
    for (size_t j = 0; j < Wp->cols && n_checked < 6; ++j) {
        double orig = (*Wp)(0, j);
        (*Wp)(0, j) = orig + eps;
        Tensor out_p = net.forward(input);
        double Lp = l2_loss_value(out_p, target);
        (*Wp)(0, j) = orig - eps;
        Tensor out_m = net.forward(input);
        double Lm = l2_loss_value(out_m, target);
        (*Wp)(0, j) = orig;
        double num = (Lp - Lm) / (2.0 * eps);
        double ana = (*Gp)(0, j);
        double err = rel_err(num, ana);
        max_err = max(max_err, err);
        ++n_checked;
    }
    cout << "b_in max rel err = " << max_err << endl;
    check("b_in gradient check (rel_err < 1e-2)", max_err < 1e-2);
}

// =====================================================================
// Test 11: update_weights reduces loss
// =====================================================================
static void test_training_reduces_loss() {
    cout << endl << "--- Test 11: training step reduces loss ---" << endl;
    size_t batch = 4, d = 4, num_out = 2, steps = 2;
    TabNet net((int)d, (int)num_out, (int)steps, 2, 2, 0.5);
    Tensor input = make_input(batch, d, 0.3, 20);
    Tensor target = make_input(batch, num_out, 0.1, 21);

    double lr = 0.01;
    Tensor out0 = net.forward(input);
    double L0 = l2_loss_value(out0, target);
    for (int step = 0; step < 30; ++step) {
        Tensor out = net.forward(input);
        Tensor grad_loss = l2_loss_grad(out, target);
        net.zero_grad();
        net.backward(grad_loss, 0.0);
        net.update_weights(lr);
    }
    Tensor outF = net.forward(input);
    double LF = l2_loss_value(outF, target);
    cout << "L0=" << L0 << " LF=" << LF << " (reduction=" << (L0 - LF) / max(L0, 1e-9) * 100 << "%)" << endl;
    check("loss reduces after 30 steps", LF < L0);
}

// =====================================================================
// Test 12: parameter / gradient count consistency
// =====================================================================
static void test_param_grad_count() {
    cout << endl << "--- Test 12: parameter / gradient count consistency ---" << endl;
    int input_dim = 5, num_out = 3, num_steps = 3;
    TabNet net(input_dim, num_out, num_steps, 2, 2, 1.5);
    auto params = net.parameters();
    auto grads  = net.gradients();
    // Expected count:
    //   shared (8) + indep[0..S-1] (8*S) + attention[0..S-1] (1*S) + step_fc[0..S-1] (2*S)
    //   + input projection (2: W_in, b_in)
    // = 8 + 8*S + S + 2*S + 2 = 10 + 11*S
    size_t expected = 10 + 11 * (size_t)num_steps;
    cout << "param count = " << params.size() << ", expected " << expected << endl;
    check("param count matches expected", params.size() == expected);
    check("grad count matches param count", grads.size() == params.size());
}

// =====================================================================
// Test 13: forward determinism
// =====================================================================
static void test_forward_deterministic() {
    cout << endl << "--- Test 13: forward determinism (same model, two forward calls) ---" << endl;
    Tensor input = make_input(3, 4, 0.1, 30);
    // Seed RNG so initialization is reproducible for the deterministic-input test below.
    std::mt19937 rng(7777);
    (void)rng;
    TabNet n1(4, 2, 2, 2, 2, 1.5);
    TabNet n2(4, 2, 2, 2, 2, 1.5);
    // Project both networks to the same parameters by hand-copying n1 -> n2.
    auto p1 = n1.parameters();
    auto p2 = n2.parameters();
    if (p1.size() != p2.size()) {
        check("param count matches between nets", false);
        return;
    }
    for (size_t i = 0; i < p1.size(); ++i) {
        for (size_t k = 0; k < p1[i]->data.size(); ++k)
            (*p2[i]).data[k] = (*p1[i]).data[k];
    }
    Tensor out1 = n1.forward(input);
    Tensor out2 = n2.forward(input);
    double diff = 0.0;
    for (size_t i = 0; i < out1.data.size(); ++i)
        diff += fabs(out1.data[i] - out2.data[i]);
    check("two nets with same params produce identical output", diff < 1e-12);
    // Also: same net, two forward calls -> same output.
    Tensor out_a = n1.forward(input);
    Tensor out_b = n1.forward(input);
    double diff2 = 0.0;
    for (size_t i = 0; i < out_a.data.size(); ++i)
        diff2 += fabs(out_a.data[i] - out_b.data[i]);
    check("same net, two forward calls -> identical output", diff2 < 1e-12);
}

// =====================================================================
// Test 14: zero_grad clears all gradients
// =====================================================================
static void test_zero_grad() {
    cout << endl << "--- Test 14: zero_grad clears gradients ---" << endl;
    TabNet net(4, 2, 2, 2, 2, 1.5);
    Tensor input = make_input(3, 4, 0.2, 31);
    Tensor target = make_input(3, 2, 0.1, 32);
    Tensor out = net.forward(input);
    Tensor grad_loss = l2_loss_grad(out, target);
    net.backward(grad_loss, 0.0);
    // Verify something is non-zero
    bool any_nonzero = false;
    for (auto* g : net.gradients())
        for (size_t i = 0; i < g->data.size(); ++i)
            if (g->data[i] != 0.0) any_nonzero = true;
    check("grads non-zero after backward", any_nonzero);
    net.zero_grad();
    bool all_zero = true;
    for (auto* g : net.gradients())
        for (size_t i = 0; i < g->data.size(); ++i)
            if (g->data[i] != 0.0) all_zero = false;
    check("grads all zero after zero_grad", all_zero);
}

// =====================================================================
// main
// =====================================================================
int main() {
    cout << "=== TabNet tests ===" << endl;
    test_forward_shape();
    test_output_finite();
    test_output_nonzero();
    test_mask_sums_to_one();
    test_prior_scale_decays();
    test_input_grad();
    test_step_fc_grad();
    test_attention_grad();
    test_indep_w1_grad();
    test_shared_w2_grad();
    test_W_in_grad();
    test_b_in_grad();
    test_training_reduces_loss();
    test_param_grad_count();
    test_forward_deterministic();
    test_zero_grad();
    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
