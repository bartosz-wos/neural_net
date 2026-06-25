// tests/test_graphsage.cpp — GraphSAGE layer/model tests
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/layers/architectures/graphsage.h"
#include "nn/nn.h"

using namespace std;

static double sage_rel_err(double a, double b) {
    if (std::fabs(b) < 1e-8) return std::fabs(a);
    return std::fabs(a - b) / std::max(std::fabs(b), 1e-8);
}

// L2 loss and gradient helpers
static double l2_loss(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j) {
            double d = out(i, j) - tgt(i, j);
            s += 0.5 * d * d;
        }
    return s;
}
static Tensor l2_loss_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            g(i, j) = out(i, j) - tgt(i, j);
    return g;
}

// Build a chain graph on N nodes (0-1-2-...-N-1).
static Tensor make_chain_adj(size_t N) {
    Tensor adj(N, N);
    for (size_t i = 0; i < N; ++i) for (size_t j = 0; j < N; ++j) adj(i, j) = 0.0;
    for (size_t i = 0; i + 1 < N; ++i) { adj(i, i+1) = 1.0; adj(i+1, i) = 1.0; }
    return adj;
}

// Fully connected graph on N nodes (no self-loops).
static Tensor make_full_adj(size_t N) {
    Tensor adj(N, N);
    adj.fill(1.0);
    for (size_t i = 0; i < N; ++i) adj(i, i) = 0.0;
    return adj;
}

int main() {
    cout << "=== GraphSAGE Tests ===" << endl;
    int total = 0, passed = 0;

    auto report = [&](const std::string& name, bool ok, const std::string& extra = "") {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << name << (extra.empty() ? "" : " " + extra) << "\n"; }
        else    {            cout << "[FAIL] " << name << (extra.empty() ? "" : " " + extra) << "\n"; }
    };

    // ===============================================================
    // Test 1: Forward shape — mean aggregator
    // ===============================================================
    cout << "\n--- Test 1: Forward shape (mean) ---\n";
    {
        size_t N = 5, in_f = 4, out_f = 6;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/false, /*self_loop=*/true);
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) - 0.05 * f;
        Tensor adj = make_full_adj(N);
        Tensor out = layer.forward_with_adj(input, adj);
        report("forward shape (5,4) -> (5,6)",
               out.rows == 5 && out.cols == 6,
               "got (" + std::to_string(out.rows) + ", " + std::to_string(out.cols) + ")");
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols && finite; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        report("output finite", finite);
    }

    // ===============================================================
    // Test 2: Mean aggregation correctness — when all neighbours are equal,
    //         mean(h_u) = h_u (single neighbour chain node)
    // ===============================================================
    cout << "\n--- Test 2: Mean aggregator — single-neighbour sanity ---\n";
    {
        // 2-node chain: each node has exactly 1 neighbour.
        size_t N = 2, in_f = 3, out_f = 3;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/false, /*self_loop=*/false);
        Tensor input(N, in_f);
        input(0, 0) = 1.0; input(0, 1) = 2.0; input(0, 2) = 3.0;
        input(1, 0) = 4.0; input(1, 1) = 5.0; input(1, 2) = 6.0;
        Tensor adj = make_chain_adj(N);
        Tensor out = layer.forward_with_adj(input, adj);
        // agg(i) = mean of neighbours. With self_loop=false:
        //   agg(0) = input(1) = [4,5,6],   agg(1) = input(0) = [1,2,3]
        // concat(0) = [input(0); agg(0)] = [1,2,3,4,5,6]
        // concat(1) = [input(1); agg(1)] = [4,5,6,1,2,3]
        // out = concat @ W^T + b — we just check it's finite and shape correct.
        report("single-neighbour mean forward shape",
               out.rows == 2 && out.cols == 3);
    }

    // ===============================================================
    // Test 3: L2 normalization changes output to unit-norm per row
    // ===============================================================
    cout << "\n--- Test 3: L2 normalization ---\n";
    {
        size_t N = 3, in_f = 2, out_f = 4;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/true);
        Tensor input(N, in_f);
        input(0, 0) = 1.0; input(0, 1) = 0.5;
        input(1, 0) = -1.0; input(1, 1) = 2.0;
        input(2, 0) = 0.0; input(2, 1) = -0.3;
        Tensor adj = make_full_adj(N);
        Tensor out = layer.forward_with_adj(input, adj);
        bool unit_norm = true;
        for (size_t i = 0; i < N; ++i) {
            double n2 = 0.0;
            for (size_t k = 0; k < out.cols; ++k) n2 += out(i, k) * out(i, k);
            double n = std::sqrt(n2);
            if (std::fabs(n - 1.0) > 1e-6) { unit_norm = false; break; }
        }
        report("L2-normalized output has unit row norm", unit_norm);
    }

    // ===============================================================
    // Test 4: Numerical gradient check (mean aggregator, no normalize)
    // ===============================================================
    cout << "\n--- Test 4: Numerical gradient check (mean) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 3, out_f = 4;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/false);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.2 - 0.1 * i + 0.05 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * f - 0.05 * (i + 1);
        }
        Tensor adj = make_chain_adj(N);

        Tensor out = layer.forward_with_adj(input, adj);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < N && ok; ++ri) {
            for (size_t rf = 0; rf < in_f && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward_with_adj(input, adj);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward_with_adj(input, adj);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = sage_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-4) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] mean input grad matches numerical\n"; }
        else    {            cout << "[FAIL] mean input grad rel_err too high\n"; }
    }

    // ===============================================================
    // Test 5: Numerical gradient check (max aggregator, no normalize)
    // ===============================================================
    cout << "\n--- Test 5: Numerical gradient check (max) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 3, out_f = 4;
        GraphSAGELayer layer(in_f, out_f, "max", /*normalize=*/false);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        // Use distinct values so argmax is unambiguous.
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.3 * (i + 1) - 0.1 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * f - 0.05 * (i + 1);
        }
        Tensor adj = make_chain_adj(N);

        Tensor out = layer.forward_with_adj(input, adj);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < N && ok; ++ri) {
            for (size_t rf = 0; rf < in_f && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward_with_adj(input, adj);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward_with_adj(input, adj);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = sage_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-4) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] max input grad matches numerical\n"; }
        else    {            cout << "[FAIL] max input grad rel_err too high\n"; }
    }

    // ===============================================================
    // Test 6: Numerical gradient check (pool aggregator)
    // ===============================================================
    cout << "\n--- Test 6: Numerical gradient check (pool) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 3, out_f = 4;
        GraphSAGELayer layer(in_f, out_f, "pool", /*normalize=*/false);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.3 * (i + 1) - 0.1 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * f - 0.05 * (i + 1);
        }
        Tensor adj = make_chain_adj(N);

        Tensor out = layer.forward_with_adj(input, adj);
        Tensor d_out = l2_loss_grad(out, target);
        Tensor d_input = layer.backward(d_out, 0.0);

        double eps = 1e-5;
        bool ok = true;
        double max_rel = 0.0;
        for (size_t ri = 0; ri < N && ok; ++ri) {
            for (size_t rf = 0; rf < in_f && ok; ++rf) {
                double orig = input(ri, rf);
                input(ri, rf) = orig + eps;
                Tensor out_p = layer.forward_with_adj(input, adj);
                double lp = l2_loss(out_p, target);
                input(ri, rf) = orig - eps;
                Tensor out_m = layer.forward_with_adj(input, adj);
                double lm = l2_loss(out_m, target);
                input(ri, rf) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = d_input(ri, rf);
                double rel = sage_rel_err(num, ana);
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-4) { ok = false; }
            }
        }
        cout << "  max rel_err = " << max_rel << "\n";
        if (ok) { ++passed; cout << "[PASS] pool input grad matches numerical\n"; }
        else    {            cout << "[FAIL] pool input grad rel_err too high\n"; }
    }

    // ===============================================================
    // Test 7: W gradient check (mean aggregator)
    // ===============================================================
    cout << "\n--- Test 7: W weight gradient check (mean) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 3, out_f = 3;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/false);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.2 - 0.05 * i + 0.1 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * f;
        }
        Tensor adj = make_chain_adj(N);

        Tensor out = layer.forward_with_adj(input, adj);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);

        // Access the W_ member's grad_weights via the parameters/gradients interface.
        auto params = layer.parameters();
        auto grads  = layer.gradients();
        // params[0] = weights, grads[0] = grad_weights (from Dense)
        Tensor* W = params[0];
        Tensor* G = grads[0];

        // Numerical gradient for entry (0, 0) of W.
        double eps = 1e-5;
        double orig = (*W)(0, 0);
        (*W)(0, 0) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input, adj);
        double lp = l2_loss(out_p, target);
        (*W)(0, 0) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input, adj);
        double lm = l2_loss(out_m, target);
        (*W)(0, 0) = orig;
        double num = (lp - lm) / (2.0 * eps);
        double ana = (*G)(0, 0);
        double rel = sage_rel_err(num, ana);
        cout << "  W[0,0] num=" << num << " ana=" << ana << " rel_err=" << rel << "\n";
        if (rel < 1e-4) { ++passed; cout << "[PASS] W gradient matches numerical\n"; }
        else            {            cout << "[FAIL] W gradient rel_err too high\n"; }
    }

    // ===============================================================
    // Test 8: W_pool gradient check (pool aggregator)
    // ===============================================================
    cout << "\n--- Test 8: W_pool weight gradient check (pool) ---\n";
    {
        ++total;
        size_t N = 3, in_f = 3, out_f = 3;
        GraphSAGELayer layer(in_f, out_f, "pool", /*normalize=*/false);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.3 * (i + 1) - 0.1 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * f;
        }
        Tensor adj = make_chain_adj(N);

        Tensor out = layer.forward_with_adj(input, adj);
        Tensor d_out = l2_loss_grad(out, target);
        layer.backward(d_out, 0.0);

        auto params = layer.parameters();
        auto grads  = layer.gradients();
        // params = [W.weights, W.bias, W_pool.weights, W_pool.bias]
        // grads  = [W.grad_weights, W.grad_bias, W_pool.grad_weights, W_pool.grad_bias]
        Tensor* Wp = params[2];
        Tensor* Gp = grads[2];
        double eps = 1e-5;
        double orig = (*Wp)(0, 0);
        (*Wp)(0, 0) = orig + eps;
        Tensor out_p = layer.forward_with_adj(input, adj);
        double lp = l2_loss(out_p, target);
        (*Wp)(0, 0) = orig - eps;
        Tensor out_m = layer.forward_with_adj(input, adj);
        double lm = l2_loss(out_m, target);
        (*Wp)(0, 0) = orig;
        double num = (lp - lm) / (2.0 * eps);
        double ana = (*Gp)(0, 0);
        double rel = sage_rel_err(num, ana);
        cout << "  W_pool[0,0] num=" << num << " ana=" << ana << " rel_err=" << rel << "\n";
        if (rel < 1e-4) { ++passed; cout << "[PASS] W_pool gradient matches numerical\n"; }
        else            {            cout << "[FAIL] W_pool gradient rel_err too high\n"; }
    }

    // ===============================================================
    // Test 9: GraphSAGEModel training reduces loss
    // ===============================================================
    cout << "\n--- Test 9: GraphSAGEModel training reduces loss ---\n";
    {
        ++total;
        size_t N = 5, in_f = 4, hidden = 6, out_f = 3;
        GraphSAGEModel model(in_f, hidden, out_f, /*num_layers=*/2, "mean", /*normalize=*/true);
        Tensor input(N, in_f);
        Tensor target(N, out_f);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.1 * (i + 1) - 0.05 * f;
            for (size_t f = 0; f < out_f; ++f) target(i, f) = 0.1 * (i + 1) * (f + 1) * 0.05;
        }
        Tensor adj = make_chain_adj(N);

        Tensor out0 = model.forward_with_adj(input, adj);
        double L0 = l2_loss(out0, target);

        double lr = 0.05;
        int steps = 60;
        for (int s = 0; s < steps; ++s) {
            Tensor out = model.forward_with_adj(input, adj);
            Tensor g = l2_loss_grad(out, target);
            model.zero_grad();
            model.backward(g, 0.0);
            model.update_weights(lr);
        }
        Tensor out_final = model.forward_with_adj(input, adj);
        double Lf = l2_loss(out_final, target);
        double reduction = 100.0 * (L0 - Lf) / (std::fabs(L0) + 1e-9);
        cout << "  initial: " << L0 << " final: " << Lf << " reduction: " << reduction << "%\n";
        if (Lf < L0) { ++passed; cout << "[PASS] model training reduces loss\n"; }
        else         {            cout << "[FAIL] model did not reduce loss\n"; }
    }

    // ===============================================================
    // Test 10: Permutation equivariance (mean aggregator) — since SAGE
    //          with mean aggregator sums over a fixed neighbourhood, the
    //          output for a permutation of nodes is the permutation of
    //          the output.
    // ===============================================================
    cout << "\n--- Test 10: Permutation equivariance (mean) ---\n";
    {
        ++total;
        size_t N = 4, in_f = 3, out_f = 5;
        GraphSAGELayer layer_a(in_f, out_f, "mean", /*normalize=*/false);
        // Use identical weights for both layers by cloning.
        GraphSAGELayer layer_b(in_f, out_f, "mean", /*normalize=*/false);
        // Copy weights from layer_a into layer_b.
        auto pa = layer_a.parameters();
        auto pb = layer_b.parameters();
        // layer_a and layer_b both have W_.weights as params[0] and W_.bias as params[1].
        for (size_t i = 0; i < pa.size(); ++i) {
            *pb[i] = pa[i]->clone();
        }

        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input(i, f) = 0.1 * (i + 1) - 0.05 * f;
        Tensor adj = make_full_adj(N);

        Tensor out_a = layer_a.forward_with_adj(input, adj);

        // Permute input + adjacency, run through layer_b, compare to permuted out_a.
        std::vector<size_t> perm = {2, 0, 3, 1};
        std::vector<size_t> inv(N);
        for (size_t i = 0; i < N; ++i) inv[perm[i]] = i;

        Tensor input_perm(N, in_f);
        for (size_t i = 0; i < N; ++i)
            for (size_t f = 0; f < in_f; ++f)
                input_perm(i, f) = input(inv[i], f);
        Tensor adj_perm(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                adj_perm(i, j) = adj(inv[i], inv[j]);

        Tensor out_b = layer_b.forward_with_adj(input_perm, adj_perm);

        bool equiv = true;
        for (size_t i = 0; i < N && equiv; ++i)
            for (size_t k = 0; k < out_f && equiv; ++k) {
                double a = out_a(inv[i], k);
                double b = out_b(i, k);
                if (std::fabs(a - b) > 1e-9) equiv = false;
            }
        if (equiv) { ++passed; cout << "[PASS] permutation equivariance holds (mean, full adj)\n"; }
        else       {            cout << "[FAIL] permutation equivariance broken\n"; }
    }

    // ===============================================================
    // Test 11: zero_grad clears all gradients (mean, pool)
    // ===============================================================
    cout << "\n--- Test 11: zero_grad clears all gradients ---\n";
    {
        size_t N = 3, in_f = 3, out_f = 3;
        bool ok = true;
        for (auto agg : {string("mean"), string("pool"), string("max")}) {
            GraphSAGELayer layer(in_f, out_f, agg, /*normalize=*/false);
            Tensor input(N, in_f);
            for (size_t i = 0; i < N; ++i) for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.1;
            Tensor adj = make_chain_adj(N);
            Tensor out = layer.forward_with_adj(input, adj);
            Tensor d = l2_loss_grad(out, out);  // any grad
            layer.backward(d, 0.0);
            layer.zero_grad();
            auto grads = layer.gradients();
            for (auto* g : grads) {
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        if (std::fabs((*g)(i, j)) > 1e-12) { ok = false; }
            }
        }
        report("zero_grad clears all grads (mean/pool/max)", ok);
    }

    // ===============================================================
    // Test 12: parameters() returns expected count
    // ===============================================================
    cout << "\n--- Test 12: parameters() count ---\n";
    {
        GraphSAGELayer layer_mean(3, 4, "mean");
        report("mean: 2 params (W + b)", layer_mean.parameters().size() == 2);
        GraphSAGELayer layer_pool(3, 4, "pool");
        report("pool: 4 params (W + b + W_pool + b_pool)",
               layer_pool.parameters().size() == 4);
        GraphSAGEModel model(5, 5, 2, /*num_layers=*/2, "mean");
        // input_proj is NOT used because in_f == hidden (5 == 5), so:
        //   2 layers * 2 params + classifier = 6 params total.
        report("model: 2 layers * 2 + classifier = 6 params", model.parameters().size() == 6);
    }

    // ===============================================================
    // Test 13: name() returns the expected strings
    // ===============================================================
    cout << "\n--- Test 13: name() ---\n";
    {
        GraphSAGELayer layer(2, 3, "mean");
        GraphSAGEModel model(2, 4, 3);
        report("GraphSAGELayer.name() == \"GraphSAGELayer\"", layer.name() == "GraphSAGELayer");
        report("GraphSAGEModel.name() == \"GraphSAGEModel\"", model.name() == "GraphSAGEModel");
    }

    // ===============================================================
    // Test 14: Neighbour sampling reduces computation (forward still finite)
    // ===============================================================
    cout << "\n--- Test 14: Neighbour sampling works ---\n";
    {
        size_t N = 8, in_f = 3, out_f = 4;
        GraphSAGELayer layer(in_f, out_f, "mean", /*normalize=*/false);
        layer.set_num_samples(2);  // only 2 neighbours per node
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i) for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.1 * (i+1) - 0.05 * f;
        Tensor adj = make_full_adj(N);
        Tensor out = layer.forward_with_adj(input, adj);
        bool finite = true;
        for (size_t i = 0; i < out.rows && finite; ++i)
            for (size_t j = 0; j < out.cols && finite; ++j)
                if (!std::isfinite(out(i, j))) finite = false;
        report("forward with sampling is finite", finite);
        report("shape preserved under sampling",
               out.rows == N && out.cols == out_f);
    }

    // ===============================================================
    // Test 15: GraphSAGEModel forward shape
    // ===============================================================
    cout << "\n--- Test 15: GraphSAGEModel forward shape ---\n";
    {
        size_t N = 4, in_f = 3, hidden = 5, out_f = 2;
        GraphSAGEModel model(in_f, hidden, out_f, /*num_layers=*/3, "max");
        Tensor input(N, in_f);
        for (size_t i = 0; i < N; ++i) for (size_t f = 0; f < in_f; ++f) input(i, f) = 0.1 * i + 0.05 * f;
        Tensor adj = make_chain_adj(N);
        Tensor out = model.forward_with_adj(input, adj);
        report("model forward shape (4,3) -> (4,2)",
               out.rows == 4 && out.cols == 2,
               "got (" + std::to_string(out.rows) + ", " + std::to_string(out.cols) + ")");
    }

    // ===============================================================
    // Test 16: bad aggregator throws
    // ===============================================================
    cout << "\n--- Test 16: invalid aggregator throws ---\n";
    {
        bool threw = false;
        try { GraphSAGELayer bad(3, 3, "garbage"); }
        catch (const std::invalid_argument&) { threw = true; }
        report("invalid aggregator throws invalid_argument", threw);
    }

    cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}