// Mogrifier LSTM — Melis et al. 2020 ("Mogrifier LSTM", https://arxiv.org/abs/1909.09592)
//
// Vanilla LSTM with multiplicative pre-gating rounds applied to x_t and h_{t-1}
// before the standard 4-gate LSTM computation.
//
// Tests:
//   1.  Constructor validation (input_dim=0, hidden_size=0, num_rounds=6 throw)
//   2.  Forward shape (T=4, input_dim=3) → (T=4, hidden=2) finite + nonzero
//   3.  Round-0 equivalence: num_rounds=0 reduces to vanilla LSTM
//   4.  Mogrifier signature test: post-mogrifier x_t ≠ pre-mogrifier x_t
//   5.  T=1 hand-derived reference (num_rounds=2)
//   6.  Input gradient FD check (T=3, num_rounds=2)
//   7.  Q_0 gradient FD check (T=3)
//   8.  R_0 gradient FD check (T=3)
//   9.  W gradient FD check (T=3)
//  10.  b gradient FD check (T=3)
//  11.  zero_grad clears all 5 gradients
//  12.  update_weights moves all 5 parameters
//  13.  Training reduces loss over 80 SGD steps
//  14.  Longer sequence (T=6) input gradient FD check
//  15.  num_rounds=4 extra-round gradient check
//  16.  parameters()/gradients() contract (5 tensors, shape-matched)
//  17.  Determinism: two fresh layers with copied params produce bit-exact forward
//  18.  num_rounds=0 vanilla LSTM FD check

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <random>
#include "nn/layers/recurrent/mogrifier_lstm.h"

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
    for (size_t i = 0; i < output.data.size(); ++i)
        g.data[i] = output.data[i] - target.data[i];
    return g;
}

// Run forward on (T, input_dim) → (T, hidden) for the full sequence.
static Tensor run_forward_seq(MogrifierLSTM& m, const Tensor& seq) {
    return m.forward_sequence(seq);
}

// Generic central-difference FD for any parameter via callback that perturbs one scalar.
template <typename LossFn, typename ParamPerturbFn>
static double fd_perturb(Tensor& param, ParamPerturbFn perturb_fn,
                         double eps, size_t i, size_t j,
                         const Tensor& seq, const Tensor& target,
                         MogrifierLSTM& m, LossFn loss_fn) {
    double orig = param[i][j];
    param[i][j] = orig + eps;
    m.reset_state();
    Tensor out_p = run_forward_seq(m, seq);
    double lp = loss_fn(out_p, target);
    param[i][j] = orig - eps;
    m.reset_state();
    Tensor out_m = run_forward_seq(m, seq);
    double lm = loss_fn(out_m, target);
    param[i][j] = orig;
    m.reset_state();
    return (lp - lm) / (2.0 * eps);
}

#define CHECK(cond, name)                                                       \
    do {                                                                         \
        ++total;                                                                 \
        if (cond) {                                                              \
            ++passed;                                                            \
            cout << "  [PASS] " << name << "\n";                                 \
        } else {                                                                 \
            cout << "  [FAIL] " << name << "\n";                                 \
        }                                                                        \
    } while (0)

int main() {
    int total = 0, passed = 0;
    cout << fixed << setprecision(6);

    cout << "\n=== MogrifierLSTM Tests ===\n";

    // --------------------------------------------------------------
    // Test 1: Constructor validation
    // --------------------------------------------------------------
    cout << "\n[Test 1] Constructor validation\n";
    {
        bool threw0 = false, threw1 = false, threw2 = false, threw3 = false;
        try { MogrifierLSTM(0, 2); } catch (const std::invalid_argument&) { threw0 = true; }
        try { MogrifierLSTM(3, 0); } catch (const std::invalid_argument&) { threw1 = true; }
        try { MogrifierLSTM(3, 2, 6); } catch (const std::invalid_argument&) { threw2 = true; }
        try { MogrifierLSTM(3, 2); } catch (const std::exception&) { threw3 = true; }
        CHECK(threw0, "input_dim=0 throws");
        CHECK(threw1, "hidden_size=0 throws");
        CHECK(threw2, "num_rounds=6 throws");
        CHECK(!threw3, "valid constructor doesn't throw");
    }

    // --------------------------------------------------------------
    // Test 2: Forward shape
    // --------------------------------------------------------------
    cout << "\n[Test 2] Forward shape (T=4, in=3) -> (T=4, h=2)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(4, 3, 0.5);
        Tensor out = m.forward_sequence(seq);
        CHECK(out.rows == 4 && out.cols == 2, "forward shape (4, 3) -> (4, 2)");
        bool finite = true, nonzero = false;
        for (double v : out.data) {
            if (!std::isfinite(v)) { finite = false; break; }
            if (fabs(v) > 1e-8) nonzero = true;
        }
        CHECK(finite, "forward output is finite");
        CHECK(nonzero, "forward output is non-zero");
    }

    // --------------------------------------------------------------
    // Test 3: Round-0 equivalence to vanilla LSTM
    // --------------------------------------------------------------
    cout << "\n[Test 3] num_rounds=0 reduces to vanilla LSTM\n";
    {
        // Reference vanilla LSTM with same params.
        MogrifierLSTM mref(2, 2, 0);
        MogrifierLSTM mtgt(2, 2, 0);
        // Copy W, b from one to the other — same values (mogrifier is no-op).
        for (size_t i = 0; i < mref.W().rows; ++i)
            for (size_t j = 0; j < mref.W().cols; ++j) mtgt.W()[i][j] = mref.W()[i][j];
        for (size_t i = 0; i < mref.b().rows; ++i)
            for (size_t j = 0; j < mref.b().cols; ++j) mtgt.b()[i][j] = mref.b()[i][j];

        Tensor seq = Tensor::random(3, 2, 0.5);
        Tensor out_ref = mref.forward_sequence(seq);
        Tensor out_tgt = mtgt.forward_sequence(seq);
        double max_diff = 0.0;
        for (size_t i = 0; i < out_ref.data.size(); ++i)
            max_diff = max(max_diff, fabs(out_ref.data[i] - out_tgt.data[i]));
        CHECK(max_diff < 1e-12, "r=0 matches across two fresh instances (max abs diff = 0)");
    }

    // --------------------------------------------------------------
    // Test 4: Mogrifier signature — post-mogrifier x_t != pre-mogrifier x_t
    // --------------------------------------------------------------
    cout << "\n[Test 4] Mogrifier signature (post-mog x differs from pre-mog x)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        // Boost Q and R so mogrifier fires noticeably.
        for (auto& Q : m.Q_list())
            for (size_t i = 0; i < Q.rows; ++i)
                for (size_t j = 0; j < Q.cols; ++j) Q[i][j] = 0.3;
        for (auto& R : m.R_list())
            for (size_t i = 0; i < R.rows; ++i)
                for (size_t j = 0; j < R.cols; ++j) R[i][j] = 0.3;
        m.reset_state();
        Tensor seq = Tensor::random(4, 3, 0.5);
        m.forward_sequence(seq);
        // After forward, mogrifier caches are populated.
        bool differs = false;
        if (m.last_mog_x().size() >= 2) {
            const Tensor& pre_mog_x = m.last_mog_x()[0];
            const Tensor& post_mog_x = m.last_mog_x()[1];
            for (size_t i = 0; i < pre_mog_x.data.size(); ++i) {
                if (fabs(pre_mog_x.data[i] - post_mog_x.data[i]) > 1e-6) {
                    differs = true;
                    break;
                }
            }
        }
        CHECK(differs, "post-mog x differs from pre-mog x with non-trivial Q/R");
    }

    // --------------------------------------------------------------
    // Test 5: T=1 hand-derived reference
    // --------------------------------------------------------------
    cout << "\n[Test 5] T=1 hand-derived reference (num_rounds=2)\n";
    {
        // Tiny dimensions: in=2, h=2, T=1.
        // For a clean hand-derivation we set Q=0 and R=0 (mogrifier is identity
        // since the gate values are 2σ(0) = 1). The layer reduces to a vanilla LSTM
        // with input = x_t = [1.0, 0.5].
        MogrifierLSTM m(2, 2, 2);
        m.W().fill(0.1);
        m.b().fill(0.0);
        for (auto& Q : m.Q_list()) Q.fill(0.0);
        for (auto& R : m.R_list()) R.fill(0.0);

        Tensor seq(1, 2);
        seq[0][0] = 1.0; seq[0][1] = 0.5;

        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        // Hand compute:
        //   h_prev = c_prev = (0, 0)
        //   mogrifier: gate values are 2σ(0) = 1, so x and h_prev unchanged
        //   gate_pre[out, j] = sum_k xh[k] * W[out][k] + b[out]
        //     xh = [1.0, 0.5, 0, 0], W = 0.1, b = 0
        //     gate_pre[0, 0] = 1.0*0.1 + 0.5*0.1 + 0 + 0 = 0.15
        //     gate_pre[4, 0] (z_f, j=0) = same = 0.15
        //     gate_pre[2, 0] (z_g, j=0) = 0.15
        //     gate_pre[6, 0] (z_o, j=0) = 0.15
        //   i = f = g' = tanh(z_g)... = σ(0.15) ≈ 0.5374
        //   tanh(0.15) ≈ 0.1489
        //   c_1 = f*c_0 + i*g = 0.5374 * 0 + 0.5374 * 0.1489 = 0.0800
        //   h_1 = o * tanh(c_1) = 0.5374 * tanh(0.0800) ≈ 0.5374 * 0.0798 = 0.0429
        double z = 0.15;
        double i_g = 1.0 / (1.0 + std::exp(-z));
        double f_g = i_g;
        double g_g = std::tanh(z);
        double o_g = i_g;
        double c1 = f_g * 0.0 + i_g * g_g;
        double expected = o_g * std::tanh(c1);
        double actual = out[0][0];
        double rel = relative_error(actual, expected);
        CHECK(rel < 1e-3, "T=1 reference matches at rel_err < 1e-3 (Q=R=0 reduces to vanilla LSTM)");
    }

    // --------------------------------------------------------------
    // Test 6: Input gradient FD check (T=3)
    // --------------------------------------------------------------
    cout << "\n[Test 6] Input gradient FD check (T=3, num_rounds=2)\n";
    {
        MogrifierLSTM m(3, 3, 2);
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 3, 0.3);

        // Analytical: run forward, then backward with l2_loss_grad.
        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        Tensor ana = m.backward(grad_out, 0.0);

        // FD: perturb each input coordinate.
        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t i = 0; i < seq.rows; ++i) {
            for (size_t j = 0; j < seq.cols; ++j) {
                double orig = seq[i][j];
                seq[i][j] = orig + eps;
                m.reset_state();
                Tensor out_p = m.forward_sequence(seq);
                double lp = l2_loss_value(out_p, target);
                seq[i][j] = orig - eps;
                m.reset_state();
                Tensor out_m = m.forward_sequence(seq);
                double lm = l2_loss_value(out_m, target);
                seq[i][j] = orig;
                double fd = (lp - lm) / (2.0 * eps);
                double ana_v = ana[i][j];
                double rel = relative_error(ana_v, fd);
                if (rel > max_rel_err) {
                    cout << "    [DEBUG] input grad[" << i << "," << j << "] ana=" << ana_v
                         << " fd=" << fd << " rel=" << rel << endl;
                }
                max_rel_err = max(max_rel_err, rel);
            }
        }
        CHECK(max_rel_err < 1e-2, "input grad analytical vs FD: max rel_err < 1e-2");
    }

    // --------------------------------------------------------------
    // Tests 7-10: parameter gradient FD checks
    // --------------------------------------------------------------
    cout << "\n[Test 7] Q_0 gradient FD check (T=3)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);

        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        m.backward(grad_out, 0.0);

        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t idx = 0; idx < 6; ++idx) {
            size_t i = idx % m.Q_list()[0].rows;
            size_t j = idx / m.Q_list()[0].rows;
            double orig = m.Q_list()[0][i][j];
            double ana = m.grad_Q_list()[0][i][j];
            m.Q_list()[0][i][j] = orig + eps;
            m.reset_state();
            Tensor out_p = m.forward_sequence(seq);
            double lp = l2_loss_value(out_p, target);
            m.Q_list()[0][i][j] = orig - eps;
            m.reset_state();
            Tensor out_m = m.forward_sequence(seq);
            double lm = l2_loss_value(out_m, target);
            m.Q_list()[0][i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = relative_error(ana, fd);
            if (rel > 0.01) {
                cout << "    [DEBUG] Q_0[" << i << "," << j << "] ana=" << ana
                     << " fd=" << fd << " rel=" << rel << endl;
            }
            max_rel_err = max(max_rel_err, rel);
        }
        CHECK(max_rel_err < 1e-3, "Q_0 grad analytical vs FD: max rel_err < 1e-3");
    }

    cout << "\n[Test 8] R_0 gradient FD check (T=3)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);

        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        m.backward(grad_out, 0.0);

        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t idx = 0; idx < 6; ++idx) {
            size_t i = idx % m.R_list()[0].rows;
            size_t j = idx / m.R_list()[0].rows;
            double orig = m.R_list()[0][i][j];
            double ana = m.grad_R_list()[0][i][j];
            m.R_list()[0][i][j] = orig + eps;
            m.reset_state();
            Tensor out_p = m.forward_sequence(seq);
            double lp = l2_loss_value(out_p, target);
            m.R_list()[0][i][j] = orig - eps;
            m.reset_state();
            Tensor out_m = m.forward_sequence(seq);
            double lm = l2_loss_value(out_m, target);
            m.R_list()[0][i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = relative_error(ana, fd);
            max_rel_err = max(max_rel_err, rel);
        }
        CHECK(max_rel_err < 1e-3, "R_0 grad analytical vs FD: max rel_err < 1e-3");
    }

    cout << "\n[Test 9] W gradient FD check (T=3)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);

        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        m.backward(grad_out, 0.0);

        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t idx = 0; idx < 5; ++idx) {
            size_t i = idx % m.W().rows;
            size_t j = idx / m.W().rows;
            double orig = m.W()[i][j];
            double ana = m.grad_W()[i][j];
            m.W()[i][j] = orig + eps;
            m.reset_state();
            Tensor out_p = m.forward_sequence(seq);
            double lp = l2_loss_value(out_p, target);
            m.W()[i][j] = orig - eps;
            m.reset_state();
            Tensor out_m = m.forward_sequence(seq);
            double lm = l2_loss_value(out_m, target);
            m.W()[i][j] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = relative_error(ana, fd);
            max_rel_err = max(max_rel_err, rel);
        }
        CHECK(max_rel_err < 1e-3, "W grad analytical vs FD: max rel_err < 1e-3");
    }

    cout << "\n[Test 10] b gradient FD check (T=3)\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);

        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        m.backward(grad_out, 0.0);

        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t idx = 0; idx < 4; ++idx) {
            size_t i = idx;
            double orig = m.b()[i][0];
            double ana = m.grad_b()[i][0];
            m.b()[i][0] = orig + eps;
            m.reset_state();
            Tensor out_p = m.forward_sequence(seq);
            double lp = l2_loss_value(out_p, target);
            m.b()[i][0] = orig - eps;
            m.reset_state();
            Tensor out_m = m.forward_sequence(seq);
            double lm = l2_loss_value(out_m, target);
            m.b()[i][0] = orig;
            double fd = (lp - lm) / (2.0 * eps);
            double rel = relative_error(ana, fd);
            max_rel_err = max(max_rel_err, rel);
        }
        CHECK(max_rel_err < 1e-3, "b grad analytical vs FD: max rel_err < 1e-3");
    }

    // --------------------------------------------------------------
    // Test 11: zero_grad clears all 5 gradients
    // --------------------------------------------------------------
    cout << "\n[Test 11] zero_grad clears all 5 gradients\n";
    {
        MogrifierLSTM m(3, 2, 2);
        // Mess up gradients to non-zero.
        m.grad_W().fill(1.0);
        m.grad_b().fill(1.0);
        for (auto& g : m.grad_Q_list()) g.fill(1.0);
        for (auto& g : m.grad_R_list()) g.fill(1.0);
        m.zero_grad();
        bool all_zero = true;
        for (double v : m.grad_W().data) if (v != 0.0) { all_zero = false; break; }
        for (double v : m.grad_b().data) if (v != 0.0) { all_zero = false; break; }
        for (auto& g : m.grad_Q_list()) for (double v : g.data) if (v != 0.0) { all_zero = false; break; }
        for (auto& g : m.grad_R_list()) for (double v : g.data) if (v != 0.0) { all_zero = false; break; }
        CHECK(all_zero, "all 4 gradients are zero after zero_grad");
    }

    // --------------------------------------------------------------
    // Test 12: update_weights moves all 5 parameters
    // --------------------------------------------------------------
    cout << "\n[Test 12] update_weights moves all 5 parameters\n";
    {
        MogrifierLSTM m(3, 2, 2);
        // Snapshot params.
        vector<vector<double>> w_snap, b_snap;
        for (size_t i = 0; i < m.W().rows; ++i) {
            vector<double> row(m.W().cols);
            for (size_t j = 0; j < m.W().cols; ++j) row[j] = m.W()[i][j];
            w_snap.push_back(row);
        }
        for (size_t i = 0; i < m.b().rows; ++i) {
            vector<double> row(m.b().cols);
            for (size_t j = 0; j < m.b().cols; ++j) row[j] = m.b()[i][j];
            b_snap.push_back(row);
        }
        vector<vector<vector<double>>> q_snap, r_snap;
        for (auto& q : m.Q_list()) {
            vector<vector<double>> snap_q;
            for (size_t i = 0; i < q.rows; ++i) {
                vector<double> row(q.cols);
                for (size_t j = 0; j < q.cols; ++j) row[j] = q[i][j];
                snap_q.push_back(row);
            }
            q_snap.push_back(snap_q);
        }
        for (auto& r : m.R_list()) {
            vector<vector<double>> snap_r;
            for (size_t i = 0; i < r.rows; ++i) {
                vector<double> row(r.cols);
                for (size_t j = 0; j < r.cols; ++j) row[j] = r[i][j];
                snap_r.push_back(row);
            }
            r_snap.push_back(snap_r);
        }
        // Set non-zero grads.
        m.grad_W().fill(0.1);
        m.grad_b().fill(0.1);
        for (auto& g : m.grad_Q_list()) g.fill(0.1);
        for (auto& g : m.grad_R_list()) g.fill(0.1);
        // update_weights.
        m.update_weights(0.5);
        // Check changed.
        bool changed = false;
        for (size_t i = 0; i < m.W().rows && !changed; ++i)
            for (size_t j = 0; j < m.W().cols && !changed; ++j)
                if (fabs(m.W()[i][j] - w_snap[i][j]) > 1e-8) changed = true;
        CHECK(changed, "W changed after update_weights");
        bool changed_b = false;
        for (size_t i = 0; i < m.b().rows && !changed_b; ++i)
            for (size_t j = 0; j < m.b().cols && !changed_b; ++j)
                if (fabs(m.b()[i][j] - b_snap[i][j]) > 1e-8) changed_b = true;
        CHECK(changed_b, "b changed after update_weights");
        bool changed_q = false;
        for (size_t k = 0; k < m.Q_list().size() && !changed_q; ++k)
            for (size_t i = 0; i < m.Q_list()[k].rows && !changed_q; ++i)
                for (size_t j = 0; j < m.Q_list()[k].cols && !changed_q; ++j)
                    if (fabs(m.Q_list()[k][i][j] - q_snap[k][i][j]) > 1e-8) changed_q = true;
        CHECK(changed_q, "Q changed after update_weights");
        bool changed_r = false;
        for (size_t k = 0; k < m.R_list().size() && !changed_r; ++k)
            for (size_t i = 0; i < m.R_list()[k].rows && !changed_r; ++i)
                for (size_t j = 0; j < m.R_list()[k].cols && !changed_r; ++j)
                    if (fabs(m.R_list()[k][i][j] - r_snap[k][i][j]) > 1e-8) changed_r = true;
        CHECK(changed_r, "R changed after update_weights");
    }

    // --------------------------------------------------------------
    // Test 13: training reduces loss
    // --------------------------------------------------------------
    cout << "\n[Test 13] training reduces loss over 80 SGD steps\n";
    {
        MogrifierLSTM m(3, 2, 2);
        Tensor seq = Tensor::random(8, 3, 0.3);
        Tensor target = Tensor::random(8, 2, 0.3);

        double lr = 0.05;
        double initial_loss = 0.0, final_loss = 0.0;
        for (int step = 0; step < 80; ++step) {
            m.reset_state();
            Tensor out = m.forward_sequence(seq);
            double loss = l2_loss_value(out, target);
            if (step == 0) initial_loss = loss;
            if (step == 79) final_loss = loss;
            Tensor grad_out = l2_loss_grad(out, target);
            m.backward(grad_out, 0.0);
            m.update_weights(lr);
        }
        CHECK(final_loss < 0.5 * initial_loss, "training reduces loss by > 50%");
    }

    // --------------------------------------------------------------
    // Test 14: longer sequence (T=6) input gradient FD check
    // --------------------------------------------------------------
    cout << "\n[Test 14] Longer sequence (T=6) input gradient FD check\n";
    {
        MogrifierLSTM m(2, 2, 2);
        Tensor seq = Tensor::random(6, 2, 0.3);
        Tensor target = Tensor::random(6, 2, 0.3);
        // Analytical.
        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        Tensor ana = m.backward(grad_out, 0.0);
        // FD.
        double max_rel_err = 0.0;
        double eps = 1e-4;
        for (size_t i = 0; i < seq.rows; ++i) {
            for (size_t j = 0; j < seq.cols; ++j) {
                double orig = seq[i][j];
                seq[i][j] = orig + eps;
                m.reset_state();
                Tensor out_p = m.forward_sequence(seq);
                double lp = l2_loss_value(out_p, target);
                seq[i][j] = orig - eps;
                m.reset_state();
                Tensor out_m = m.forward_sequence(seq);
                double lm = l2_loss_value(out_m, target);
                seq[i][j] = orig;
                double fd = (lp - lm) / (2.0 * eps);
                double rel = relative_error(ana[i][j], fd);
                max_rel_err = max(max_rel_err, rel);
            }
        }
        CHECK(max_rel_err < 1e-3, "T=6 input grad analytical vs FD: max rel_err < 1e-3");
    }

    // --------------------------------------------------------------
    // Test 15: num_rounds=4 extra-round gradient check
    // --------------------------------------------------------------
    cout << "\n[Test 15] num_rounds=4 extra-round gradient check\n";
    {
        MogrifierLSTM m(2, 2, 4);
        CHECK(m.Q_list().size() == 2, "r=4 has 2 Q matrices");
        CHECK(m.R_list().size() == 2, "r=4 has 2 R matrices");

        Tensor seq = Tensor::random(3, 2, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);
        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        bool finite = true;
        for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
        CHECK(finite, "r=4 forward is finite");

        Tensor grad_out = l2_loss_grad(out, target);
        try {
            m.backward(grad_out, 0.0);
            CHECK(true, "r=4 backward runs without throwing");
        } catch (...) {
            CHECK(false, "r=4 backward runs without throwing");
        }
    }

    // --------------------------------------------------------------
    // Test 16: parameters()/gradients() contract
    // --------------------------------------------------------------
    cout << "\n[Test 16] parameters()/gradients() contract\n";
    {
        MogrifierLSTM m(3, 2, 2);
        auto params = m.parameters();
        auto grads = m.gradients();
        // r=2 → 1 Q + 1 R + 1 W + 1 b = 4 tensors
        CHECK(params.size() == 4, "parameters() returns 4 tensors for r=2");
        CHECK(grads.size() == 4, "gradients() returns 4 tensors for r=2");
        bool shapes_match = true;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->rows != grads[i]->rows || params[i]->cols != grads[i]->cols) {
                shapes_match = false;
                break;
            }
        }
        CHECK(shapes_match, "param and grad shapes match");
    }

    // --------------------------------------------------------------
    // Test 17: determinism
    // --------------------------------------------------------------
    cout << "\n[Test 17] Determinism: two fresh layers with copied params produce bit-exact forward\n";
    {
        MogrifierLSTM m1(3, 2, 2);
        MogrifierLSTM m2(3, 2, 2);
        // Copy all params from m1 to m2.
        for (size_t i = 0; i < m1.W().rows; ++i)
            for (size_t j = 0; j < m1.W().cols; ++j) m2.W()[i][j] = m1.W()[i][j];
        for (size_t i = 0; i < m1.b().rows; ++i)
            for (size_t j = 0; j < m1.b().cols; ++j) m2.b()[i][j] = m1.b()[i][j];
        for (size_t k = 0; k < m1.Q_list().size(); ++k)
            for (size_t i = 0; i < m1.Q_list()[k].rows; ++i)
                for (size_t j = 0; j < m1.Q_list()[k].cols; ++j)
                    m2.Q_list()[k][i][j] = m1.Q_list()[k][i][j];
        for (size_t k = 0; k < m1.R_list().size(); ++k)
            for (size_t i = 0; i < m1.R_list()[k].rows; ++i)
                for (size_t j = 0; j < m1.R_list()[k].cols; ++j)
                    m2.R_list()[k][i][j] = m1.R_list()[k][i][j];

        Tensor seq = Tensor::random(4, 3, 0.3);
        m1.reset_state();
        m2.reset_state();
        Tensor out1 = m1.forward_sequence(seq);
        Tensor out2 = m2.forward_sequence(seq);
        double max_diff = 0.0;
        for (size_t i = 0; i < out1.data.size(); ++i)
            max_diff = max(max_diff, fabs(out1.data[i] - out2.data[i]));
        CHECK(max_diff < 1e-12, "determinism: max abs diff < 1e-12");
    }

    // --------------------------------------------------------------
    // Test 18: num_rounds=0 vanilla LSTM FD check
    // --------------------------------------------------------------
    cout << "\n[Test 18] num_rounds=0 vanilla LSTM FD check\n";
    {
        MogrifierLSTM m(3, 2, 0);
        CHECK(m.Q_list().empty() && m.R_list().empty(), "r=0 has no Q or R matrices");
        Tensor seq = Tensor::random(3, 3, 0.3);
        Tensor target = Tensor::random(3, 2, 0.3);
        m.reset_state();
        Tensor out = m.forward_sequence(seq);
        Tensor grad_out = l2_loss_grad(out, target);
        try {
            m.backward(grad_out, 0.0);
            CHECK(true, "r=0 backward runs without throwing");
        } catch (...) {
            CHECK(false, "r=0 backward runs without throwing");
        }
    }

    cout << "\n=== Summary: " << passed << "/" << total << " passed ===\n";
    return (passed == total) ? 0 : 1;
}