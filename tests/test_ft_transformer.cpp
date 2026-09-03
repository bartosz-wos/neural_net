// test_ft_transformer.cpp — Gradient correctness tests for FT-Transformer
//   (Gorishniy et al. 2021, https://arxiv.org/abs/2106.11959)
//
// Pure-transformer architecture for tabular data with per-feature
// numerical embeddings + CLS-token aggregation.
//
// Tests:
//   1. NumericalFeatureTokenizer constructor validation
//   2. NumericalFeatureTokenizer forward shape & finiteness
//   3. NumericalFeatureTokenizer determinism (copy→forward bit-exact)
//   4. NumericalFeatureTokenizer input gradient FD check
//   5. NumericalFeatureTokenizer W gradient FD check
//   6. NumericalFeatureTokenizer b gradient FD check
//   7. CategoricalFeatureTokenizer constructor validation
//   8. CategoricalFeatureTokenizer forward shape & finiteness
//   9. CategoricalFeatureTokenizer input gradient FD check
//  10. CategoricalFeatureTokenizer E gradient FD check
//  11. FTTransformer constructor validation
//  12. FTTransformer forward shape & finiteness
//  13. FTTransformer output gradient flow (numerical input gradient is non-zero)
//  14. FTTransformer CLS token gradient updates weights (training reduces loss)
//  15. FTTransformer parameters()/gradients() contract

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/architectures/ft_transformer.h"

using namespace std;

static int passed = 0, failed = 0;
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

// ----------------------------------------------------------------------------
// NumericalFeatureTokenizer — Tests 1-6
// ----------------------------------------------------------------------------

static void test_num_tok_constructor() {
    cout << endl << "--- Test 1: NumericalFeatureTokenizer constructor validation ---" << endl;
    bool threw_d = false, threw_n = false, valid_ok = true;
    try { NumericalFeatureTokenizer bad(0, 3); } catch (const std::exception&) { threw_d = true; }
    try { NumericalFeatureTokenizer bad(4, 0); } catch (const std::exception&) { threw_n = true; }
    try { NumericalFeatureTokenizer good(4, 3); } catch (const std::exception&) { valid_ok = false; }
    cout << "  d_model=0 threw: " << threw_d
         << "  n_num=0 threw: " << threw_n
         << "  valid ok: " << valid_ok << endl;
    check("NumericalFeatureTokenizer constructor validates", threw_d && threw_n && valid_ok);
}

static void test_num_tok_forward_shape() {
    cout << endl << "--- Test 2: NumericalFeatureTokenizer forward shape ---" << endl;
    size_t B = 3, d_model = 4, n_num = 2;
    NumericalFeatureTokenizer tok(d_model, n_num);
    Tensor input(B, n_num);
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.1 * i + 0.05;
    }
    Tensor out = tok.forward(input);
    cout << "  input: " << B << "x" << n_num
         << "  output: " << out.rows << "x" << out.cols
         << " (expected " << B << "x" << n_num * d_model << ")" << endl;
    check("forward output shape", out.rows == B && out.cols == n_num * d_model);
    bool all_finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (!std::isfinite(out.data[i])) { all_finite = false; break; }
    }
    check("forward output is finite", all_finite);
}

static void test_num_tok_determinism() {
    cout << endl << "--- Test 3: NumericalFeatureTokenizer determinism ---" << endl;
    // Verify the same input produces the same output (forward is pure).
    NumericalFeatureTokenizer t1(3, 2);
    Tensor input(2, 2);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * i;

    Tensor o1 = t1.forward(input.clone());
    Tensor o1_again = t1.forward(input.clone());
    double max_diff = 0.0;
    for (size_t i = 0; i < o1.data.size(); ++i) {
        max_diff = max(max_diff, fabs(o1.data[i] - o1_again.data[i]));
    }
    cout << "  bit-exact (max diff): " << max_diff << endl;
    check("forward is deterministic", max_diff < 1e-12);
}

// Helper: get W_ and b_ for tests via a derived class
struct NumTokProbe : public NumericalFeatureTokenizer {
    NumTokProbe(size_t d, size_t n) : NumericalFeatureTokenizer(d, n) {}
    // Use the public accessors from the parent.
    using NumericalFeatureTokenizer::W;
    using NumericalFeatureTokenizer::b;
    using NumericalFeatureTokenizer::grad_W;
    using NumericalFeatureTokenizer::grad_b;
};

static void test_num_tok_input_grad() {
    cout << endl << "--- Test 4: NumericalFeatureTokenizer input gradient FD ---" << endl;
    // d_model = 3, n_num = 2, B = 2
    size_t B = 2, d_model = 3, n_num = 2;
    NumTokProbe tok(d_model, n_num);

    Tensor input(B, n_num);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * i;

    // Analytical input gradient
    Tensor out = tok.forward(input);
    // arbitrary grad (B, n_num * d_model)
    Tensor grad(B, n_num * d_model);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.3 * i - 0.1;
    Tensor dx_ana = tok.backward(grad, 0.0);

    // Finite differences
    double eps = 1e-5;
    Tensor dx_fd(B, n_num);
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < n_num; ++i) {
            double orig = input(b, i);
            input(b, i) = orig + eps;
            Tensor out_p = tok.forward(input);
            input(b, i) = orig - eps;
            Tensor out_m = tok.forward(input);
            input(b, i) = orig;

            double s = 0.0;
            // Loss = sum(out * grad)
            for (size_t k = 0; k < out_p.data.size(); ++k) {
                s += (out_p.data[k] - out_m.data[k]) * grad.data[k] / (2.0 * eps);
            }
            dx_fd(b, i) = s;
        }
    }
    double max_re = 0.0;
    for (size_t i = 0; i < dx_ana.data.size(); ++i) {
        max_re = max(max_re, rel_err(dx_ana.data[i], dx_fd.data[i]));
    }
    cout << "  max rel_err dx: " << max_re << endl;
    check("input gradient FD rel_err < 1e-3", max_re < 1e-3);
}

static void test_num_tok_W_grad() {
    cout << endl << "--- Test 5: NumericalFeatureTokenizer W gradient FD ---" << endl;
    size_t B = 2, d_model = 3, n_num = 2;
    NumTokProbe tok(d_model, n_num);

    Tensor input(B, n_num);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * i - 0.3;

    Tensor out = tok.forward(input);
    Tensor grad(B, n_num * d_model);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.4 * i - 0.2;

    tok.zero_grad();
    tok.backward(grad, 0.0);
    Tensor grad_W_ana = tok.grad_W().clone();

    // FD: perturb each W entry
    double eps = 1e-5;
    Tensor grad_W_fd = tok.grad_W().clone();
    grad_W_fd.fill(0.0);

    for (size_t j = 0; j < d_model; ++j) {
        for (size_t i = 0; i < n_num; ++i) {
            double orig = tok.W()(j, i);
            tok.W()(j, i) = orig + eps;
            Tensor out_p = tok.forward(input);
            tok.W()(j, i) = orig - eps;
            Tensor out_m = tok.forward(input);
            tok.W()(j, i) = orig;

            double s = 0.0;
            for (size_t k = 0; k < out_p.data.size(); ++k) {
                s += (out_p.data[k] - out_m.data[k]) * grad.data[k] / (2.0 * eps);
            }
            grad_W_fd(j, i) = s;
        }
    }
    double max_re = 0.0;
    for (size_t i = 0; i < grad_W_ana.data.size(); ++i) {
        max_re = max(max_re, rel_err(grad_W_ana.data[i], grad_W_fd.data[i]));
    }
    cout << "  max rel_err dW: " << max_re << endl;
    check("W gradient FD rel_err < 1e-3", max_re < 1e-3);
}

static void test_num_tok_b_grad() {
    cout << endl << "--- Test 6: NumericalFeatureTokenizer b gradient FD ---" << endl;
    size_t B = 2, d_model = 3, n_num = 2;
    NumTokProbe tok(d_model, n_num);

    Tensor input(B, n_num);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.2 * i - 0.3;

    Tensor out = tok.forward(input);
    Tensor grad(B, n_num * d_model);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.4 * i - 0.2;

    tok.zero_grad();
    tok.backward(grad, 0.0);
    Tensor grad_b_ana = tok.grad_b().clone();

    double eps = 1e-5;
    Tensor grad_b_fd = tok.grad_b().clone();
    grad_b_fd.fill(0.0);

    for (size_t j = 0; j < d_model; ++j) {
        double orig = tok.b()(0, j);
        tok.b()(0, j) = orig + eps;
        Tensor out_p = tok.forward(input);
        tok.b()(0, j) = orig - eps;
        Tensor out_m = tok.forward(input);
        tok.b()(0, j) = orig;

        double s = 0.0;
        for (size_t k = 0; k < out_p.data.size(); ++k) {
            s += (out_p.data[k] - out_m.data[k]) * grad.data[k] / (2.0 * eps);
        }
        grad_b_fd(0, j) = s;
    }
    double max_re = 0.0;
    for (size_t i = 0; i < grad_b_ana.data.size(); ++i) {
        max_re = max(max_re, rel_err(grad_b_ana.data[i], grad_b_fd.data[i]));
    }
    cout << "  max rel_err db: " << max_re << endl;
    check("b gradient FD rel_err < 1e-3", max_re < 1e-3);
}

// ----------------------------------------------------------------------------
// CategoricalFeatureTokenizer — Tests 7-10
// ----------------------------------------------------------------------------

struct CatTokProbe : public CategoricalFeatureTokenizer {
    CatTokProbe(size_t d, size_t n, size_t v) : CategoricalFeatureTokenizer(d, n, v) {}
    using CategoricalFeatureTokenizer::E;
    using CategoricalFeatureTokenizer::b_cat;
    using CategoricalFeatureTokenizer::grad_E;
    using CategoricalFeatureTokenizer::grad_b_cat;
};

static void test_cat_tok_constructor() {
    cout << endl << "--- Test 7: CategoricalFeatureTokenizer constructor validation ---" << endl;
    bool threw_d = false, threw_n = false, threw_v = false, valid_ok = true;
    try { CategoricalFeatureTokenizer bad(0, 3, 4); } catch (const std::exception&) { threw_d = true; }
    try { CategoricalFeatureTokenizer bad(4, 0, 4); } catch (const std::exception&) { threw_n = true; }
    try { CategoricalFeatureTokenizer bad(4, 3, 0); } catch (const std::exception&) { threw_v = true; }
    try { CategoricalFeatureTokenizer good(4, 3, 4); } catch (const std::exception&) { valid_ok = false; }
    cout << "  d_model=0 threw: " << threw_d
         << "  n_cat=0 threw: " << threw_n
         << "  vocab_max=0 threw: " << threw_v
         << "  valid ok: " << valid_ok << endl;
    check("CategoricalFeatureTokenizer constructor validates",
          threw_d && threw_n && threw_v && valid_ok);
}

static void test_cat_tok_forward_shape() {
    cout << endl << "--- Test 8: CategoricalFeatureTokenizer forward shape ---" << endl;
    size_t B = 3, d_model = 4, n_cat = 2, vocab_max = 5;
    CategoricalFeatureTokenizer tok(d_model, n_cat, vocab_max);
    Tensor input(B, n_cat * vocab_max);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * i;

    Tensor out = tok.forward(input);
    cout << "  input: " << B << "x" << n_cat * vocab_max
         << "  output: " << out.rows << "x" << out.cols
         << " (expected " << B << "x" << n_cat * d_model << ")" << endl;
    check("forward shape", out.rows == B && out.cols == n_cat * d_model);
    bool all_finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (!std::isfinite(out.data[i])) { all_finite = false; break; }
    }
    check("forward finite", all_finite);
}

static void test_cat_tok_input_grad() {
    cout << endl << "--- Test 9: CategoricalFeatureTokenizer input gradient FD ---" << endl;
    size_t B = 2, d_model = 3, n_cat = 2, vocab_max = 4;
    CatTokProbe tok(d_model, n_cat, vocab_max);

    Tensor input(B, n_cat * vocab_max);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * i + 0.05;

    Tensor out = tok.forward(input);
    Tensor grad(B, n_cat * d_model);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.3 * i - 0.1;

    Tensor dx_ana = tok.backward(grad, 0.0);

    double eps = 1e-5;
    Tensor dx_fd(B, n_cat * vocab_max);
    for (size_t b = 0; b < B; ++b) {
        for (size_t k = 0; k < n_cat * vocab_max; ++k) {
            double orig = input(b, k);
            input(b, k) = orig + eps;
            Tensor out_p = tok.forward(input);
            input(b, k) = orig - eps;
            Tensor out_m = tok.forward(input);
            input(b, k) = orig;

            double s = 0.0;
            for (size_t j = 0; j < out_p.data.size(); ++j) {
                s += (out_p.data[j] - out_m.data[j]) * grad.data[j] / (2.0 * eps);
            }
            dx_fd(b, k) = s;
        }
    }
    double max_re = 0.0;
    for (size_t i = 0; i < dx_ana.data.size(); ++i) {
        max_re = max(max_re, rel_err(dx_ana.data[i], dx_fd.data[i]));
    }
    cout << "  max rel_err dx: " << max_re << endl;
    check("categorical input grad FD rel_err < 1e-3", max_re < 1e-3);
}

static void test_cat_tok_E_grad() {
    cout << endl << "--- Test 10: CategoricalFeatureTokenizer E gradient FD ---" << endl;
    size_t B = 2, d_model = 3, n_cat = 2, vocab_max = 4;
    CatTokProbe tok(d_model, n_cat, vocab_max);

    Tensor input(B, n_cat * vocab_max);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.1 * i + 0.05;

    Tensor out = tok.forward(input);
    Tensor grad(B, n_cat * d_model);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.3 * i - 0.1;

    tok.zero_grad();
    tok.backward(grad, 0.0);
    Tensor grad_E_ana = tok.grad_E().clone();

    double eps = 1e-5;
    srand(42);
    int n_check = 5;
    double max_re = 0.0;
    for (int q = 0; q < n_check; ++q) {
        // E_ is stored as (n_cat * vocab_max, d_model).  Sample a random row, col.
        size_t i = (size_t)(rand() % n_cat);
        size_t k = (size_t)(rand() % vocab_max);
        size_t j = (size_t)(rand() % d_model);

        double orig = tok.E()(i * vocab_max + k, j);
        tok.E()(i * vocab_max + k, j) = orig + eps;
        Tensor out_p = tok.forward(input);
        tok.E()(i * vocab_max + k, j) = orig - eps;
        Tensor out_m = tok.forward(input);
        tok.E()(i * vocab_max + k, j) = orig;

        double s = 0.0;
        for (size_t r = 0; r < out_p.data.size(); ++r) {
            s += (out_p.data[r] - out_m.data[r]) * grad.data[r] / (2.0 * eps);
        }
        max_re = max(max_re, rel_err(s, grad_E_ana(i * vocab_max + k, j)));
    }
    cout << "  max rel_err dE (5 random): " << max_re << endl;
    check("categorical E gradient FD rel_err < 1e-3", max_re < 1e-3);
}

// ----------------------------------------------------------------------------
// FTTransformer — Tests 11-15
// ----------------------------------------------------------------------------

static void test_ft_constructor() {
    cout << endl << "--- Test 11: FTTransformer constructor validation ---" << endl;
    bool threw_d = false, threw_c = false, threw_h = false, threw_b = false, valid_ok = true;
    try { FTTransformer bad(0, 3, 4, 0, 0, 2, 1); } catch (const std::exception&) { threw_d = true; }
    try { FTTransformer bad(4, 0, 4, 0, 0, 2, 1); } catch (const std::exception&) { threw_c = true; }
    try { FTTransformer bad(4, 3, 4, 0, 0, 3, 1); } catch (const std::exception&) { threw_h = true; }  // d_model=4 not div by 3
    try { FTTransformer bad(4, 3, 4, 0, 0, 2, 0); } catch (const std::exception&) { threw_b = true; }
    try { FTTransformer good(4, 3, 4, 0, 0, 2, 1); } catch (const std::exception&) { valid_ok = false; }
    cout << "  d_model=0 threw: " << threw_d
         << "  n_classes=0 threw: " << threw_c
         << "  bad head dim threw: " << threw_h
         << "  num_blocks=0 threw: " << threw_b
         << "  valid ok: " << valid_ok << endl;
    check("FTTransformer constructor validates", threw_d && threw_c && threw_h && threw_b && valid_ok);
}

static void test_ft_forward_shape() {
    cout << endl << "--- Test 12: FTTransformer forward shape ---" << endl;
    size_t B = 2, d_model = 4, n_classes = 3, n_num = 3;
    FTTransformer ft(d_model, n_classes, n_num, 0, 0, /*heads=*/2, /*blocks=*/2);

    Tensor numerical(B, n_num);
    for (size_t i = 0; i < numerical.data.size(); ++i) numerical.data[i] = 0.1 * i;
    Tensor empty;  // no categorical

    Tensor logits = ft.forward(numerical, empty);
    cout << "  numerical: " << B << "x" << n_num
         << "  logits: " << logits.rows << "x" << logits.cols
         << " (expected " << B << "x" << n_classes << ")" << endl;
    check("forward logits shape", logits.rows == B && logits.cols == n_classes);
    bool all_finite = true;
    for (size_t i = 0; i < logits.data.size(); ++i) {
        if (!std::isfinite(logits.data[i])) { all_finite = false; break; }
    }
    check("forward logits finite", all_finite);
}

static void test_ft_backward_flow() {
    cout << endl << "--- Test 13: FTTransformer backward flow ---" << endl;
    size_t B = 2, d_model = 4, n_classes = 3, n_num = 3;
    FTTransformer ft(d_model, n_classes, n_num, 0, 0, 2, 2);

    Tensor numerical(B, n_num);
    for (size_t i = 0; i < numerical.data.size(); ++i) numerical.data[i] = 0.1 * i;
    Tensor empty;

    Tensor logits = ft.forward(numerical, empty);

    // Pick a non-trivial grad (e.g. (logits - target) for some target)
    Tensor target(B, n_classes);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.2 * i;

    Tensor grad(B, n_classes);
    for (size_t i = 0; i < B * n_classes; ++i) grad.data[i] = logits.data[i] - target.data[i];

    ft.zero_grad();
    auto out_back = ft.backward_full(grad, 0.0);

    cout << "  grad_numerical: " << out_back.grad_numerical.rows << "x" << out_back.grad_numerical.cols << endl;
    bool grad_nonzero = false;
    for (size_t i = 0; i < out_back.grad_numerical.data.size(); ++i) {
        if (fabs(out_back.grad_numerical.data[i]) > 1e-9) { grad_nonzero = true; break; }
    }
    cout << "  numerical gradient non-zero: " << grad_nonzero << endl;
    check("numerical input gradient is non-zero (gradient flows backward)", grad_nonzero);

    // Also check that gradients were accumulated in the parameters
    auto params = ft.parameters();
    auto grads = ft.gradients();
    bool any_grad_nonzero = false;
    for (size_t p = 0; p < grads.size(); ++p) {
        for (size_t i = 0; i < grads[p]->data.size(); ++i) {
            if (fabs(grads[p]->data[i]) > 1e-12) { any_grad_nonzero = true; break; }
        }
        if (any_grad_nonzero) break;
    }
    check("at least one parameter gradient is non-zero", any_grad_nonzero);
}

static void test_ft_training_reduces_loss() {
    cout << endl << "--- Test 14: FTTransformer training reduces loss ---" << endl;
    size_t B = 4, d_model = 4, n_classes = 2, n_num = 3;
    FTTransformer ft(d_model, n_classes, n_num, 0, 0, 2, 2);

    Tensor numerical(B, n_num);
    for (size_t i = 0; i < numerical.data.size(); ++i) numerical.data[i] = 0.1 * i - 0.2;
    Tensor empty;

    Tensor target(B, n_classes);
    for (size_t b = 0; b < B; ++b) target(b, 0) = 1.0;

    double lr = 0.01;
    double loss0 = 0.0;
    double lossf = 0.0;
    for (size_t step = 0; step < 30; ++step) {
        Tensor logits = ft.forward(numerical, empty);
        // MSE
        double loss = 0.0;
        for (size_t i = 0; i < logits.data.size(); ++i) {
            double d = logits.data[i] - target.data[i];
            loss += 0.5 * d * d;
        }
        if (step == 0)  loss0 = loss;
        if (step == 29) lossf = loss;

        // Gradient = (logits - target)
        Tensor grad(B, n_classes);
        for (size_t i = 0; i < logits.data.size(); ++i) grad.data[i] = logits.data[i] - target.data[i];

        ft.zero_grad();
        ft.backward_full(grad, 0.0);
        ft.update_weights(lr);
    }
    cout << "  L0=" << loss0 << "  L_final=" << lossf
         << "  reduction: " << (1.0 - lossf / loss0) * 100 << "%" << endl;
    check("training reduces loss by > 30%", lossf < loss0 * 0.7);
}

static void test_ft_param_grad_contract() {
    cout << endl << "--- Test 15: FTTransformer parameters()/gradients() contract ---" << endl;
    size_t B = 2, d_model = 4, n_classes = 3, n_num = 3, n_cat = 2, vocab_max = 4;
    FTTransformer ft(d_model, n_classes, n_num, n_cat, vocab_max, 2, 2);

    Tensor numerical(B, n_num);
    for (size_t i = 0; i < numerical.data.size(); ++i) numerical.data[i] = 0.1 * i;
    Tensor categorical(B, n_cat * vocab_max);
    for (size_t i = 0; i < categorical.data.size(); ++i) categorical.data[i] = 0.05 * i;

    Tensor logits = ft.forward(numerical, categorical);
    Tensor grad(B, n_classes);
    for (size_t i = 0; i < grad.data.size(); ++i) grad.data[i] = 0.1 * i;

    ft.zero_grad();
    ft.backward_full(grad, 0.0);

    auto params = ft.parameters();
    auto grads = ft.gradients();
    cout << "  #parameters: " << params.size()
         << "  #gradients: " << grads.size() << endl;
    check("parameters() and gradients() same count", params.size() == grads.size());
    bool shapes_match = true;
    for (size_t p = 0; p < params.size(); ++p) {
        if (params[p]->rows != grads[p]->rows || params[p]->cols != grads[p]->cols) {
            shapes_match = false;
            cout << "    param " << p << " shape (" << params[p]->rows << "," << params[p]->cols
                 << ") != grad shape (" << grads[p]->rows << "," << grads[p]->cols << ")" << endl;
            break;
        }
    }
    check("parameter/gradient shapes match", shapes_match);
}

int main() {
    cout << "=== FT-Transformer Tests ===" << endl;

    // NumericalFeatureTokenizer
    test_num_tok_constructor();
    test_num_tok_forward_shape();
    test_num_tok_determinism();
    test_num_tok_input_grad();
    test_num_tok_W_grad();
    test_num_tok_b_grad();

    // CategoricalFeatureTokenizer
    test_cat_tok_constructor();
    test_cat_tok_forward_shape();
    test_cat_tok_input_grad();
    test_cat_tok_E_grad();

    // FTTransformer
    test_ft_constructor();
    test_ft_forward_shape();
    test_ft_backward_flow();
    test_ft_training_reduces_loss();
    test_ft_param_grad_contract();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}