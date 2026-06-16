#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/architectures/egnn.h"

using namespace std;

double relative_error(double a, double b) {
    if (fabs(b) < 1e-8) return fabs(a);
    return fabs(a - b) / max(fabs(b), 1e-8);
}

// =============================================================
// Helper: small fully-connected 4-node graph
// (0, 1, 2, 3) with all pairs connected, no self-loops
// =============================================================
Tensor make_adj_4nodes_full() {
    Tensor adj(4, 4);
    adj.fill(1.0);
    for (size_t i = 0; i < 4; ++i) adj(i, i) = 0.0;
    return adj;
}

// 3-node graph: 0-1, 1-2 (chain)
Tensor make_adj_3nodes_chain() {
    Tensor adj(3, 3);
    adj(0, 1) = 1; adj(1, 0) = 1;
    adj(1, 2) = 1; adj(2, 1) = 1;
    return adj;
}

// L2 loss grad: y - target
Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    return output - target;
}

double l2_loss_value(const Tensor& output, const Tensor& target) {
    Tensor d = output - target;
    double s = 0.0;
    for (size_t i = 0; i < d.rows; ++i)
        for (size_t j = 0; j < d.cols; ++j)
            s += 0.5 * d(i, j) * d(i, j);
    return s;
}

int main() {
    cout << "=== EGNN Tests ===" << endl;
    int total = 0, passed = 0;

    // ===============================================================
    // Test 1: EGNNLayer forward shape
    // ===============================================================
    cout << "\n--- Test 1: EGNNLayer forward shape ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, hidden = 5, coord_dim = 3;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f);
        Tensor input_x(N, coord_dim);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * (i + 1) + 0.05 * f;
            for (size_t c = 0; c < coord_dim; ++c) input_x(i, c) = 0.3 * i - 0.1 * c;
        }
        EGNNLayer layer(in_f, hidden, coord_dim);
        Tensor output = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_out = layer.get_last_x();
        cout << "Input h: " << input_h.rows << "x" << input_h.cols
             << "  Output h: " << output.rows << "x" << output.cols << "\n";
        cout << "Input x: " << input_x.rows << "x" << input_x.cols
             << "  Output x: " << x_out.rows << "x" << x_out.cols << "\n";
        if (output.rows == N && output.cols == hidden &&
            x_out.rows == N && x_out.cols == coord_dim) {
            cout << "[PASS] forward shapes correct\n";
            ++passed;
        } else {
            cout << "[FAIL] wrong shapes\n";
        }
    }

    // ===============================================================
    // Test 2: EGNNLayer output is finite
    // ===============================================================
    cout << "\n--- Test 2: EGNNLayer output is finite ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, hidden = 5;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.5 * sin(0.1 * i) + 0.3 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = cos(0.1 * i + 0.5 * c);
        }
        EGNNLayer layer(in_f, hidden);
        Tensor output = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_out = layer.get_last_x();
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        for (size_t i = 0; i < x_out.rows && finite; ++i)
            for (size_t j = 0; j < x_out.cols; ++j)
                if (!std::isfinite(x_out(i, j))) finite = false;
        if (finite) {
            cout << "[PASS] all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite output detected\n";
        }
    }

    // ===============================================================
    // Test 3: EGNNLayer handles isolated node (no neighbours)
    // ===============================================================
    cout << "\n--- Test 3: EGNNLayer with isolated node ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3;
        // Node 0 is isolated, 1-2-3 form a triangle
        Tensor adj(N, N);
        adj(1, 2) = 1; adj(2, 1) = 1;
        adj(2, 3) = 1; adj(3, 2) = 1;
        adj(1, 3) = 1; adj(3, 1) = 1;
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.2 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.05 * i * c;
        }
        EGNNLayer layer(in_f, hidden);
        Tensor output = layer.forward_with_adj(input_h, input_x, adj);
        bool finite = true;
        for (size_t i = 0; i < output.rows && finite; ++i)
            for (size_t j = 0; j < output.cols; ++j)
                if (!std::isfinite(output(i, j))) finite = false;
        if (finite && output.rows == N && output.cols == hidden) {
            cout << "[PASS] isolated node handled, all outputs finite\n";
            ++passed;
        } else {
            cout << "[FAIL] isolated node produced non-finite or wrong shape\n";
        }
    }

    // ===============================================================
    // Test 4: EGNNLayer coordinate output is rotation-equivariant
    // (forward-only test: rotate input, output should rotate by same R)
    // ===============================================================
    cout << "\n--- Test 4: EGNNLayer rotation equivariance ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.3 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.5 * sin(0.7 * i + 0.3 * c);
        }
        EGNNLayer layer(in_f, hidden);

        // 1) Compute x' = layer.forward(x)
        Tensor h_out = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_out = layer.get_last_x();

        // 2) Build a rotation matrix R (3x3 orthonormal, e.g. rotation about z by 0.5 rad)
        double theta = 0.5;
        Tensor R(3, 3);
        R(0, 0) = cos(theta);  R(0, 1) = -sin(theta); R(0, 2) = 0.0;
        R(1, 0) = sin(theta);  R(1, 1) = cos(theta);  R(1, 2) = 0.0;
        R(2, 0) = 0.0;         R(2, 1) = 0.0;         R(2, 2) = 1.0;

        // 3) Apply R to input_x: x_rot[i, c] = sum_k R[c, k] * x[i, k]
        Tensor x_rot(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                double s = 0.0;
                for (size_t k = 0; k < 3; ++k) s += R(c, k) * input_x(i, k);
                x_rot(i, c) = s;
            }
        }

        // 4) Compute layer.forward(input_h, x_rot) -> x_rot_out
        Tensor h_rot_out = layer.forward_with_adj(input_h, x_rot, adj);
        Tensor x_rot_out = layer.get_last_x();

        // 5) Compute R @ x_out for comparison
        Tensor x_out_rot(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                double s = 0.0;
                for (size_t k = 0; k < 3; ++k) s += R(c, k) * x_out(i, k);
                x_out_rot(i, c) = s;
            }
        }

        // 6) Compare x_rot_out vs R @ x_out
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                double diff = fabs(x_rot_out(i, c) - x_out_rot(i, c));
                if (diff > max_err) max_err = diff;
            }
        }
        cout << "Max rotation-equivariance error: " << max_err << "\n";
        if (max_err < 1e-9) {
            cout << "[PASS] coordinate output is exactly rotation-equivariant\n";
            ++passed;
        } else {
            cout << "[FAIL] equivariance broken: max_err = " << max_err << "\n";
        }
    }

    // ===============================================================
    // Test 5: EGNNLayer coordinate output is translation-equivariant
    // ===============================================================
    cout << "\n--- Test 5: EGNNLayer translation equivariance ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.3 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.5 * sin(0.7 * i + 0.3 * c);
        }
        EGNNLayer layer(in_f, hidden);

        Tensor h_out = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_out = layer.get_last_x();

        // Translate x by t = (1.0, -0.5, 0.7)
        Tensor x_t(N, 3);
        double t[3] = {1.0, -0.5, 0.7};
        for (size_t i = 0; i < N; ++i)
            for (size_t c = 0; c < 3; ++c)
                x_t(i, c) = input_x(i, c) + t[c];

        Tensor h_t_out = layer.forward_with_adj(input_h, x_t, adj);
        Tensor x_t_out = layer.get_last_x();

        // x_t_out should equal x_out + t (componentwise)
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                double expected = x_out(i, c) + t[c];
                double diff = fabs(x_t_out(i, c) - expected);
                if (diff > max_err) max_err = diff;
            }
        }
        cout << "Max translation-equivariance error: " << max_err << "\n";
        if (max_err < 1e-9) {
            cout << "[PASS] coordinate output is exactly translation-equivariant\n";
            ++passed;
        } else {
            cout << "[FAIL] equivariance broken: max_err = " << max_err << "\n";
        }
    }

    // ===============================================================
    // Test 6: EGNNLayer h-backward numerical gradient check
    // (perturb h, compare analytical d_h vs numerical)
    // ===============================================================
    cout << "\n--- Test 6: EGNNLayer h-backward numerical gradient check ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3, coord_dim = 3;
        Tensor adj = make_adj_3nodes_chain();
        Tensor input_h(N, in_f), input_x(N, coord_dim);
        Tensor target(N, hidden);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.2 + 0.1 * i - 0.05 * f;
            for (size_t c = 0; c < coord_dim; ++c) input_x(i, c) = 0.05 * i * c - 0.1;
            for (size_t f = 0; f < hidden; ++f) target(i, f) = 0.1 * f - 0.05 * i;
        }
        EGNNLayer layer(in_f, hidden, coord_dim);
        Tensor out = layer.forward_with_adj(input_h, input_x, adj);
        Tensor grad_loss = l2_loss_grad(out, target);
        Tensor d_input_h = layer.backward(grad_loss, 0.0);

        double eps = 1e-5;
        size_t ri = 1, rf = 0;
        double orig = input_h(ri, rf);
        input_h(ri, rf) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input_h, input_x, adj);
        double loss_p = l2_loss_value(out_p, target);
        input_h(ri, rf) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input_h, input_x, adj);
        double loss_m = l2_loss_value(out_m, target);
        input_h(ri, rf) = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = d_input_h(ri, rf);
        double rel = relative_error(num_grad, ana_grad);
        cout << "h[" << ri << "][" << rf << "] analytical: " << ana_grad
             << ", numerical: " << num_grad << ", rel_err: " << rel << "\n";
        if (rel < 1e-2) {
            cout << "[PASS] h input gradient close to numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] rel_err too high: " << rel << "\n";
        }
    }

    // ===============================================================
    // Test 7: EGNNLayer x-backward numerical gradient check
    // (perturb x, compare analytical d_x vs numerical)
    // ===============================================================
    cout << "\n--- Test 7: EGNNLayer x-backward numerical gradient check ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3, coord_dim = 3;
        Tensor adj = make_adj_3nodes_chain();
        Tensor input_h(N, in_f), input_x(N, coord_dim);
        Tensor target_x(N, coord_dim);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.2 + 0.1 * i - 0.05 * f;
            for (size_t c = 0; c < coord_dim; ++c) input_x(i, c) = 0.05 * i * c - 0.1;
            for (size_t c = 0; c < coord_dim; ++c) target_x(i, c) = 0.1 * c - 0.05 * i;
        }
        EGNNLayer layer(in_f, hidden, coord_dim);
        Tensor out_h = layer.forward_with_adj(input_h, input_x, adj);
        Tensor out_x = layer.get_last_x();
        Tensor grad_x_loss = l2_loss_grad(out_x, target_x);
        Tensor d_input_x = layer.backward_coord(grad_x_loss, 0.0);

        double eps = 1e-5;
        size_t ri = 0, rc = 1;
        double orig = input_x(ri, rc);
        input_x(ri, rc) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_p = layer.get_last_x();
        double loss_p = l2_loss_value(x_p, target_x);
        input_x(ri, rc) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_m = layer.get_last_x();
        double loss_m = l2_loss_value(x_m, target_x);
        input_x(ri, rc) = orig;
        double num_grad = (loss_p - loss_m) / (2.0 * eps);
        double ana_grad = d_input_x(ri, rc);
        double rel = relative_error(num_grad, ana_grad);
        cout << "x[" << ri << "][" << rc << "] analytical: " << ana_grad
             << ", numerical: " << num_grad << ", rel_err: " << rel << "\n";
        if (rel < 1e-2) {
            cout << "[PASS] x input gradient close to numerical\n";
            ++passed;
        } else {
            cout << "[FAIL] rel_err too high: " << rel << "\n";
        }
    }

    // ===============================================================
    // Test 8: EGNNLayer h output depends on x
    // (sanity check that x flows into the h output via dist2 -> phi_e;
    // verified by perturbing x in a single coordinate and checking that
    // the h output changes measurably. This is a probabilistic test —
    // with random init the magnitude can vary, but it should never be
    // exactly zero across many random restarts.)
    // ===============================================================
    cout << "\n--- Test 8: EGNNLayer h output depends on x ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3;
        Tensor adj = make_adj_3nodes_chain();
        Tensor input_h(N, in_f);
        Tensor input_x(N, 3);
        // Use substantial input values so dist2 produces non-trivial signals.
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.5 + 0.1 * i + 0.2 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.7 + 0.3 * i + 0.4 * c;
        }
        // Try multiple random restarts; pick the one that gives a non-zero
        // diff (we have a fixed seed in init_weights so weights are
        // deterministic — just check that *some* configuration yields a
        // measurable difference).
        EGNNLayer layer(in_f, hidden);
        Tensor out_base = layer.forward_with_adj(input_h, input_x, adj);
        // Perturb x[1][0] by +1.0 (substantial)
        double orig = input_x(1, 0);
        input_x(1, 0) = orig + 1.0;
        Tensor out_pert = layer.forward_with_adj(input_h, input_x, adj);
        input_x(1, 0) = orig;
        double diff_norm = 0.0;
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < hidden; ++f)
                diff_norm += fabs(out_base(i, f) - out_pert(i, f));
        cout << "||h_out_base - h_out_perturbed||_1 = " << diff_norm << "\n";
        if (diff_norm > 1e-4) {
            cout << "[PASS] h output changes when x changes (dist2 is used)\n";
            ++passed;
        } else {
            cout << "[FAIL] h output did not change — dist2 path is dead\n";
        }
    }

    // ===============================================================
    // Test 9: EGNNModel forward shape
    // ===============================================================
    cout << "\n--- Test 9: EGNNModel forward shape ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 4, out_f = 3, n_layers = 2;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.05 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.05 * i * c;
        }
        EGNNModel model(N, in_f, hidden, out_f, 3, n_layers);
        Tensor output = model.forward_with_adj(input_h, input_x, adj);
        cout << "Output: " << output.rows << "x" << output.cols << "\n";
        if (output.rows == N && output.cols == out_f) {
            cout << "[PASS] EGNNModel output shape correct\n";
            ++passed;
        } else {
            cout << "[FAIL] expected " << N << "x" << out_f << "\n";
        }
    }

    // ===============================================================
    // Test 10: EGNNModel training step reduces L2 loss
    // ===============================================================
    cout << "\n--- Test 10: EGNNModel training step reduces loss ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 4, out_f = 2, n_layers = 2;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.05 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.1 * i + 0.2 * c - 0.1;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * (f + 1) - 0.05 * i;
        }
        EGNNModel model(N, in_f, hidden, out_f, 3, n_layers);
        Tensor out0 = model.forward_with_adj(input_h, input_x, adj);
        double loss0 = l2_loss_value(out0, target);
        double lr = 0.005;
        for (int step = 0; step < 50; ++step) {
            Tensor out = model.forward_with_adj(input_h, input_x, adj);
            Tensor grad = l2_loss_grad(out, target);
            model.zero_grad();
            model.backward(grad, lr);
            model.update_weights(lr);
        }
        Tensor out1 = model.forward_with_adj(input_h, input_x, adj);
        double loss1 = l2_loss_value(out1, target);
        cout << "Loss before: " << loss0 << ", after: " << loss1 << "\n";
        if (loss1 < loss0) {
            cout << "[PASS] loss decreased after training\n";
            ++passed;
        } else {
            cout << "[FAIL] loss did not decrease\n";
        }
    }

    // ===============================================================
    // Test 11: EGNNModel with edge attributes
    // ===============================================================
    cout << "\n--- Test 11: EGNNModel with edge attributes ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3, out_f = 2, n_attrs = 2;
        Tensor adj = make_adj_3nodes_chain();
        Tensor input_h(N, in_f), input_x(N, 3);
        Tensor edge_attr(N * N, n_attrs);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.2 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.05 * i * c;
        }
        for (size_t i = 0; i < N * N; ++i) {
            for (size_t a = 0; a < n_attrs; ++a) {
                edge_attr(i, a) = 0.1 * i + 0.05 * a;
            }
        }
        EGNNModel model(N, in_f, hidden, out_f, 3, /*n_layers=*/2, n_attrs);
        Tensor out = model.forward_with_adj(input_h, input_x, adj, edge_attr);
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        if (finite && out.rows == N && out.cols == out_f) {
            cout << "[PASS] EGNNModel with edge attrs forward OK\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite or wrong shape\n";
        }
    }

    // ===============================================================
    // Test 12: EGNNLayer with edge weights
    // ===============================================================
    cout << "\n--- Test 12: EGNNLayer with explicit edge weights ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3;
        Tensor adj = make_adj_3nodes_chain();
        Tensor weights(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                weights(i, j) = (i == j) ? 0.0 : 0.5 + 0.1 * i + 0.05 * j;
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.2 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.05 * i * c;
        }
        EGNNLayer layer(in_f, hidden);
        Tensor out = layer.forward_with_adj(input_h, input_x, adj, Tensor(), weights);
        Tensor x_out = layer.get_last_x();
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        for (size_t i = 0; i < x_out.rows && finite; ++i)
            for (size_t j = 0; j < x_out.cols; ++j)
                if (!std::isfinite(x_out(i, j))) finite = false;
        if (finite && out.rows == N && out.cols == hidden) {
            cout << "[PASS] EGNNLayer with weights forward OK\n";
            ++passed;
        } else {
            cout << "[FAIL] non-finite or wrong shape\n";
        }
    }

    // ===============================================================
    // Test 13: EGNNModel parameter / gradient count
    // ===============================================================
    cout << "\n--- Test 13: EGNNModel parameters and gradients ---\n";
    {
        ++total;
        size_t N = 3, in_f = 2, hidden = 3, out_f = 2, n_layers = 2;
        EGNNModel model(N, in_f, hidden, out_f, 3, n_layers);
        auto params = model.parameters();
        auto grads = model.gradients();
        cout << "Parameter pointers: " << params.size() << ", gradient pointers: " << grads.size() << "\n";
        // Per EGNNLayer: phi_e (W, b) + phi_h (W, b) + phi_x (W, b) = 6 tensors
        // n_layers=2: 12 EGNN layer tensors
        // + input_proj (W, b) = 2
        // + classifier (W, b) = 2
        // Total: 2 + 12 + 2 = 16
        if (params.size() == 16 && grads.size() == 16) {
            cout << "[PASS] parameter/gradient count consistent (16 each)\n";
            ++passed;
        } else {
            cout << "[FAIL] expected 16, got " << params.size() << "/" << grads.size() << "\n";
        }
    }

    // ===============================================================
    // Test 14: EGNNLayer x-output is permutation-equivariant
    // (forward-only test: permute nodes, output should permute same way)
    // ===============================================================
    cout << "\n--- Test 14: EGNNLayer permutation equivariance ---\n";
    {
        ++total;
        size_t N = 4, in_f = 2, hidden = 3;
        Tensor adj = make_adj_4nodes_full();
        Tensor input_h(N, in_f), input_x(N, 3);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input_h(i, f) = 0.1 * i + 0.3 * f;
            for (size_t c = 0; c < 3; ++c) input_x(i, c) = 0.5 * sin(0.7 * i + 0.3 * c);
        }
        EGNNLayer layer(in_f, hidden);

        Tensor h_out = layer.forward_with_adj(input_h, input_x, adj);
        Tensor x_out = layer.get_last_x();

        // Permutation: pi = (2, 0, 3, 1) (swap 0<->2, 1<->3)
        size_t pi[4] = {2, 0, 3, 1};
        Tensor h_perm(N, in_f), x_perm(N, 3), adj_perm(N, N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) h_perm(i, f) = input_h(pi[i], f);
            for (size_t c = 0; c < 3; ++c) x_perm(i, c) = input_x(pi[i], c);
        }
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                adj_perm(i, j) = adj(pi[i], pi[j]);

        Tensor h_perm_out = layer.forward_with_adj(h_perm, x_perm, adj_perm);
        Tensor x_perm_out = layer.get_last_x();

        // x_perm_out[i] should equal x_out[pi[i]]
        double max_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                double expected = x_out(pi[i], c);
                double diff = fabs(x_perm_out(i, c) - expected);
                if (diff > max_err) max_err = diff;
            }
        }
        cout << "Max permutation-equivariance error: " << max_err << "\n";
        if (max_err < 1e-9) {
            cout << "[PASS] coordinate output is exactly permutation-equivariant\n";
            ++passed;
        } else {
            cout << "[FAIL] equivariance broken: max_err = " << max_err << "\n";
        }
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed)
         << " failed (of " << total << ") ===" << endl;
    return (passed == total) ? 0 : 1;
}
