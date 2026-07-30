// ============================================================================
// GrokFast Optimizer test suite — Lee et al. 2024 (arXiv:2405.20233, NeurIPS 2024 Spotlight)
//
// Algorithm under test:
//   Per parameter θ, per step t:
//     1. buf_t     = α · buf_{t-1} + (1 − α) · grad_t               (EMA filter)
//     2. grad_filtered_t = grad_t + λ · buf_t                       (amplify slow)
//     3. θ ← inner_optimizer.step(grad_filtered_t)                  (inner sees filtered)
//
// The wrapper sits BETWEEN the backward pass (populates grad_t) and the
// inner optimizer's step() call. It mutates the stored gradient in-place
// to the filtered value, then delegates to inner.step(model).
//
// Defaults (paper §3.2 / Table 1):
//   lambda = 2.0   (amplification factor)
//   alpha  = 0.98  (EMA momentum)
//   inner  = Adam  (lr=1e-3, β1=0.9, β2=0.999, ε=1e-8)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <functional>
#include <memory>
#include "nn/optimizers/grokfast.h"
#include "nn/optimizers/optimizer.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"
#include "nn/core/tensor.h"

static int g_pass = 0;
static int g_fail = 0;
static std::string g_current_test;

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cout << "  [FAIL] (" << g_current_test << ") " \
                      << #cond << " @ line " << __LINE__ << "\n"; \
            ++g_fail; \
        } else { ++g_pass; } \
    } while (0)

#define ASSERT_NEAR(a, b, tol) \
    do { \
        double _a = (double)(a), _b = (double)(b), _tol = (double)(tol); \
        if (!(std::abs(_a - _b) <= _tol)) { \
            std::cout << "  [FAIL] (" << g_current_test << ") " \
                      << #a << "=" << _a << " vs " << #b << "=" << _b \
                      << " tol=" << _tol << " @ line " << __LINE__ << "\n"; \
            ++g_fail; \
        } else { ++g_pass; } \
    } while (0)

static void run(const std::string& name, std::function<void()> body) {
    g_current_test = name;
    std::cout << "\n" << name << "\n" << std::flush;
    body();
}

// ----- helpers -----
static void build_dense_model(Model& m, size_t in_f, size_t out_f, double init_value) {
    Dense* d = new Dense(in_f, out_f);
    for (size_t i = 0; i < d->weights.rows; ++i)
        for (size_t j = 0; j < d->weights.cols; ++j)
            d->weights[i][j] = init_value;
    for (size_t i = 0; i < d->bias.rows; ++i)
        for (size_t j = 0; j < d->bias.cols; ++j)
            d->bias[i][j] = init_value;
    m.add_layer(d);
}

static void set_grads_to_constant(Model& m, double v) {
    for (auto& layer : m.layers) {
        auto grads = layer->gradients();
        for (auto* g : grads) {
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    (*g)[i][j] = v;
        }
    }
}

static std::vector<double> snap(const Model& m) {
    std::vector<double> v;
    for (auto& layer : m.layers)
        for (auto* p : layer->parameters())
            for (size_t i = 0; i < p->rows; ++i)
                for (size_t j = 0; j < p->cols; ++j)
                    v.push_back((*p)[i][j]);
    return v;
}


int main() {
    std::cout << std::setprecision(10);
    std::cout << "=== GrokFast Optimizer Tests ===\n" << std::flush;

    // -------------------------------------------------------------------------
    // (1) Defaults round-trip
    // -------------------------------------------------------------------------
    run("(1) defaults round-trip", []{
        auto inner = std::make_unique<Adam>();
        GrokFast opt(std::move(inner));
        ASSERT_NEAR(opt.get_lambda(), 2.0, 0.0);
        ASSERT_NEAR(opt.get_alpha(), 0.98, 0.0);
        ASSERT_NEAR(opt.get_lr(), 1e-3, 0.0);   // inherited from Adam default
        ASSERT(opt.handles_weight_decay() == false);  // Adam doesn't do WD
        ASSERT(opt.last_num_params_filtered() == 0);
    });

    // -------------------------------------------------------------------------
    // (2) Non-default constructor
    // -------------------------------------------------------------------------
    run("(2) non-default constructor", []{
        auto inner = std::make_unique<Adam>(2e-3, 0.85, 0.999, 1e-6);
        GrokFast opt(std::move(inner), 5.0, 0.9);
        ASSERT_NEAR(opt.get_lambda(), 5.0, 0.0);
        ASSERT_NEAR(opt.get_alpha(), 0.9, 0.0);
        ASSERT_NEAR(opt.get_lr(), 2e-3, 0.0);
    });

    // -------------------------------------------------------------------------
    // (3) Null inner throws
    // -------------------------------------------------------------------------
    run("(3) null inner throws", []{
        bool threw = false;
        try { GrokFast opt(nullptr); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (4) lambda validation throws
    // -------------------------------------------------------------------------
    run("(4) lambda validation throws", []{
        bool threw = false;
        try { GrokFast opt(std::make_unique<Adam>(), -1.0); }
        catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (5) alpha validation throws
    // -------------------------------------------------------------------------
    run("(5) alpha validation throws", []{
        bool threw = false;
        try { GrokFast opt(std::make_unique<Adam>(), 2.0, -0.1); }
        catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        threw = false;
        try { GrokFast opt(std::make_unique<Adam>(), 2.0, 1.5); }   // alpha > 1 throws
        catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        // alpha == 0 is allowed (no-filtering limit)
        GrokFast opt0(std::make_unique<Adam>(), 2.0, 0.0);
        ASSERT_NEAR(opt0.get_alpha(), 0.0, 0.0);
        // alpha == 1.0 is allowed at the boundary (frozen filter, degenerate)
        GrokFast opt1(std::make_unique<Adam>(), 2.0, 1.0);
        ASSERT_NEAR(opt1.get_alpha(), 1.0, 0.0);
    });

    // -------------------------------------------------------------------------
    // (6) Closed-form first-step math
    //     Adam(lr=1, b1=0.5, b2=0.5, eps=1e-3), init=0, g=1:
    //       grad_filtered = 1 + 2.0 * (0.98*0 + 0.02*1) = 1 + 0.04 = 1.04
    //       m_t = 0.5*0 + 0.5*1.04 = 0.52
    //       v_t = 0.5*0 + 0.5*1.04^2 = 0.5408
    //       bias_corr1 = 0.5, bias_corr2 = 0.5
    //       m_hat = 1.04, v_hat = 1.0816
    //       update = 1.04 / (sqrt(1.0816) + 0.001) = 1.04 / 1.0410... ≈ 0.99908...
    //       param_1 = 0 - 1*0.99908... = -0.99908...
    //
    //     Compare GrokFast against a hand-computed expected param value to ~1e-6.
    // -------------------------------------------------------------------------
    run("(6) closed-form first-step math", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3), 2.0, 0.98);
        opt.step(m);

        double g_filtered = 1.0 + 2.0 * 0.02;  // 1.04
        double m_t = 0.5 * g_filtered;          // 0.52
        double v_t = 0.5 * g_filtered * g_filtered;  // 0.5408
        double m_hat = m_t / 0.5;               // 1.04
        double v_hat = v_t / 0.5;               // 1.0816
        double denom = std::sqrt(v_hat) + 1e-3;
        double expected = -1.0 * m_hat / denom;  // -0.99908...

        for (auto& layer : m.layers) {
            auto params = layer->parameters();
            for (auto* p : params) {
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        ASSERT_NEAR((*p)[i][j], expected, 1e-6);
            }
        }
    });

    // -------------------------------------------------------------------------
    // (7) lambda = 0 reduces to inner Adam bit-exactly
    //     With lambda=0, filtered grad == raw grad → step should match
    //     a plain Adam over the same gradient sequence.
    // -------------------------------------------------------------------------
    run("(7) lambda=0 reduces to plain Adam", []{
        Model m_g, m_a;
        build_dense_model(m_g, 2, 2, 0.3);
        build_dense_model(m_a, 2, 2, 0.3);

        GrokFast opt_g(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 0.0, 0.98);
        Adam opt_a(0.1, 0.9, 0.999, 1e-8);

        std::srand(7);
        for (int step = 0; step < 5; ++step) {
            // Same gradient pattern on both
            for (auto& layer : m_g.layers) {
                auto grads = layer->gradients();
                for (auto* g : grads) {
                    for (size_t i = 0; i < g->rows; ++i)
                        for (size_t j = 0; j < g->cols; ++j)
                            (*g)[i][j] = 0.1 * (double)((std::rand() % 7) - 3);
                }
            }
            for (size_t li = 0; li < m_g.layers.size(); ++li) {
                auto gg = m_g.layers[li]->gradients();
                auto ga = m_a.layers[li]->gradients();
                for (size_t k = 0; k < gg.size(); ++k) {
                    for (size_t i = 0; i < gg[k]->rows; ++i)
                        for (size_t j = 0; j < gg[k]->cols; ++j)
                            (*ga[k])[i][j] = (*gg[k])[i][j];
                }
            }
            opt_g.step(m_g);
            opt_a.step(m_a);
        }
        auto vg = snap(m_g), va = snap(m_a);
        ASSERT(vg.size() == va.size());
        for (size_t i = 0; i < vg.size(); ++i)
            ASSERT_NEAR(vg[i], va[i], 1e-12);
    });

    // -------------------------------------------------------------------------
    // (8) alpha=0 (no-filtering limit): buf_t == grad_t exactly
    //     With α=0, the EMA filter collapses to identity: buf_t = grad_t.
    //     So grad_filtered = grad + λ · grad = (1+λ) · grad.
    //     Then GrokFast(λ,α=0) should match an explicit (1+λ)·g Adam run.
    // -------------------------------------------------------------------------
    run("(8) alpha=0 reduces to scale*(1+lambda) factor on grad", []{
        double lambda = 5.0;
        // Reference Adam with grads scaled by (1+λ)
        Model m_ref;
        build_dense_model(m_ref, 2, 2, 0.0);
        Adam adam_ref(0.1, 0.5, 0.5, 1e-8);
        // Set grads to 1.0, scale manually to (1+λ)·1
        for (auto& layer : m_ref.layers) {
            auto grads = layer->gradients();
            for (auto* g : grads) {
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        (*g)[i][j] = (1.0 + lambda);
            }
        }
        adam_ref.step(m_ref);

        // GrokFast with alpha=0, grad=1 → filtered = (1+λ)·1
        Model m_g;
        build_dense_model(m_g, 2, 2, 0.0);
        set_grads_to_constant(m_g, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(0.1, 0.5, 0.5, 1e-8), lambda, 0.0);
        opt.step(m_g);

        for (size_t li = 0; li < m_g.layers.size(); ++li) {
            auto pr = m_ref.layers[li]->parameters();
            auto pg = m_g.layers[li]->parameters();
            for (size_t k = 0; k < pr.size(); ++k) {
                for (size_t i = 0; i < pr[k]->rows; ++i)
                    for (size_t j = 0; j < pr[k]->cols; ++j)
                        ASSERT_NEAR((*pr[k])[i][j], (*pg[k])[i][j], 1e-12);
            }
        }
    });

    // -------------------------------------------------------------------------
    // (9) handles_weight_decay delegates to inner
    // -------------------------------------------------------------------------
    run("(9) handles_weight_decay delegates to inner", []{
        GrokFast opt_c(std::make_unique<Adam>());
        ASSERT(opt_c.handles_weight_decay() == false);
    });

    // -------------------------------------------------------------------------
    // (10) State diagnostics: has_state after step
    // -------------------------------------------------------------------------
    run("(10) state diagnostics after first step", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3));
        auto* layer_ptr = m.layers[0].get();
        ASSERT(opt.has_state(layer_ptr, 0) == false);
        ASSERT(opt.has_state(layer_ptr, 1) == false);
        opt.step(m);
        ASSERT(opt.has_state(layer_ptr, 0) == true);
        ASSERT(opt.has_state(layer_ptr, 1) == true);
        ASSERT(opt.last_num_params_filtered() == 2);  // weights + bias
    });

    // -------------------------------------------------------------------------
    // (11) Empty model doesn't crash
    // -------------------------------------------------------------------------
    run("(11) works with a model that has no parameters", []{
        Model m;
        auto opt = GrokFast(std::make_unique<Adam>());
        opt.step(m);
        ASSERT(opt.last_num_params_filtered() == 0);
    });

    // -------------------------------------------------------------------------
    // (12) End-to-end: linear regression y = 2x with GrokFast(Adam)
    // -------------------------------------------------------------------------
    run("(12) end-to-end GrokFast reduces loss on y=2x", []{
        std::srand(42);
        Model m;
        Dense* d = new Dense(1, 1);
        for (size_t i = 0; i < d->weights.rows; ++i)
            for (size_t j = 0; j < d->weights.cols; ++j)
                d->weights[i][j] = 0.3 * (2.0 * std::rand() / RAND_MAX - 1.0);
        for (size_t i = 0; i < d->bias.rows; ++i)
            for (size_t j = 0; j < d->bias.cols; ++j)
                d->bias[i][j] = 0.0;
        m.add_layer(d);

        std::vector<std::vector<double>> X = {{0.5}, {1.0}, {1.5}, {2.0},
                                              {0.1}, {0.7}, {1.3}, {1.9}};
        std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 0.2, 1.4, 2.6, 3.8};

        auto opt = GrokFast(std::make_unique<Adam>(0.05, 0.9, 0.999, 1e-8));

        double loss0 = 1e9;
        for (int epoch = 0; epoch < 80; ++epoch) {
            double total_loss = 0;
            for (size_t s = 0; s < X.size(); ++s) {
                Tensor inp(1, 1);
                inp[0][0] = X[s][0];
                Tensor pred = d->forward(inp);
                double err = pred[0][0] - y[s];
                total_loss += 0.5 * err * err;
                Tensor grad(1, 1);
                grad[0][0] = err;
                d->backward(grad, 0.0);
            }
            if (epoch == 0) loss0 = total_loss;
            opt.step(m);
        }
        double total_loss = 0;
        for (size_t s = 0; s < X.size(); ++s) {
            Tensor inp(1, 1);
            inp[0][0] = X[s][0];
            Tensor pred = d->forward(inp);
            double err = pred[0][0] - y[s];
            total_loss += 0.5 * err * err;
        }
        ASSERT(total_loss < 0.5 * loss0);  // ≥50% reduction
    });

    // -------------------------------------------------------------------------
    // (13) Determinism: two fresh instances produce bit-identical updates
    // -------------------------------------------------------------------------
    run("(13) determinism over 5 steps", []{
        std::srand(0);
        auto run_one = []() {
            Model m;
            build_dense_model(m, 2, 2, 0.3);
            auto opt = GrokFast(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8));
            for (int step = 0; step < 5; ++step) {
                set_grads_to_constant(m, 1.0);
                opt.step(m);
            }
            return snap(m);
        };
        auto v1 = run_one();
        auto v2 = run_one();
        ASSERT(v1.size() == v2.size());
        for (size_t i = 0; i < v1.size(); ++i)
            ASSERT_NEAR(v1[i], v2[i], 1e-12);
    });

    // -------------------------------------------------------------------------
    // (14) Signature vs Adam: GrokFast(Adam) trajectory differs from Adam
    //      over 5 steps with non-trivial grads.
    // -------------------------------------------------------------------------
    run("(14) signature vs Adam: 5-step trajectory differs", []{
        std::srand(13);
        Model m_g, m_a;
        build_dense_model(m_g, 2, 2, 0.5);
        build_dense_model(m_a, 2, 2, 0.5);
        GrokFast opt_g(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8));
        Adam opt_a(0.1, 0.9, 0.999, 1e-8);

        for (int step = 0; step < 5; ++step) {
            for (auto& layer : m_g.layers) {
                auto grads = layer->gradients();
                for (auto* g : grads) {
                    for (size_t i = 0; i < g->rows; ++i)
                        for (size_t j = 0; j < g->cols; ++j)
                            (*g)[i][j] = 0.5 * (double)((std::rand() % 5) - 2);
                }
            }
            for (size_t li = 0; li < m_g.layers.size(); ++li) {
                auto gg = m_g.layers[li]->gradients();
                auto ga = m_a.layers[li]->gradients();
                for (size_t k = 0; k < gg.size(); ++k) {
                    for (size_t i = 0; i < gg[k]->rows; ++i)
                        for (size_t j = 0; j < gg[k]->cols; ++j)
                            (*ga[k])[i][j] = (*gg[k])[i][j];
                }
            }
            opt_g.step(m_g);
            opt_a.step(m_a);
        }
        auto vg = snap(m_g), va = snap(m_a);
        double dist2 = 0;
        for (size_t i = 0; i < vg.size(); ++i) {
            double d = vg[i] - va[i];
            dist2 += d * d;
        }
        ASSERT(dist2 > 1e-6);
    });

    // -------------------------------------------------------------------------
    // (15) Setters round-trip
    // -------------------------------------------------------------------------
    run("(15) setters round-trip", []{
        GrokFast opt(std::make_unique<Adam>());
        opt.set_lambda(3.5);
        ASSERT_NEAR(opt.get_lambda(), 3.5, 0.0);
        opt.set_alpha(0.5);
        ASSERT_NEAR(opt.get_alpha(), 0.5, 0.0);
        opt.set_lr(0.07);
        ASSERT_NEAR(opt.get_lr(), 0.07, 0.0);
    });

    // -------------------------------------------------------------------------
    // (16) Setter validation
    // -------------------------------------------------------------------------
    run("(16) setters validate input", []{
        GrokFast opt(std::make_unique<Adam>());
        bool threw = false;
        try { opt.set_lambda(-0.1); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        threw = false;
        try { opt.set_alpha(1.5); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        threw = false;
        try { opt.set_alpha(-0.1); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (17) inner() returns the wrapped optimizer
    // -------------------------------------------------------------------------
    run("(17) inner() returns the wrapped optimizer", []{
        auto inner = std::make_unique<Adam>(0.123, 0.5, 0.5, 1e-3);
        Adam* inner_raw = inner.get();
        GrokFast opt(std::move(inner));
        ASSERT(opt.inner() == inner_raw);
        // Adam::lr is the per-instance lr field; access via dynamic_cast since
        // Optimizer::lr is a separate base-class field (default 0.001).
        Adam* adam = dynamic_cast<Adam*>(opt.inner());
        ASSERT(adam != nullptr);
        ASSERT_NEAR(adam->lr, 0.123, 0.0);
        // get_lr() should also report the Adam-derived lr.
        ASSERT_NEAR(opt.get_lr(), 0.123, 0.0);
    });

    // -------------------------------------------------------------------------
    // (18) Multi-layer model: each layer gets independent buf
    // -------------------------------------------------------------------------
    run("(18) multi-layer: each layer tracked", []{
        Model m;
        build_dense_model(m, 2, 2, 0.3);
        build_dense_model(m, 3, 2, 0.3);   // second layer
        set_grads_to_constant(m, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8));
        opt.step(m);
        ASSERT(opt.has_state(m.layers[0].get(), 0) == true);
        ASSERT(opt.has_state(m.layers[1].get(), 0) == true);
        ASSERT(opt.last_num_params_filtered() == 4);  // 2 layers × 2 params each
    });

    // -------------------------------------------------------------------------
    // (19) State propagates across steps (buf persists)
    //     With constant gradient, buf eventually saturates to grad/(1-α).
    //     Verify by stepping twice and checking the second-step filtered
    //     gradient matches the closed-form expectation.
    // -------------------------------------------------------------------------
    run("(19) state propagates across steps", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3), 2.0, 0.5);
        // First step: buf = (1-0.5)*1 = 0.5, filtered = 1 + 2*0.5 = 2.0
        opt.step(m);
        // Set grads again to 1
        set_grads_to_constant(m, 1.0);
        // Second step: buf = 0.5*0.5 + 0.5*1 = 0.75, filtered = 1 + 2*0.75 = 2.5
        opt.step(m);
        // Expected second-step param value with g_filtered=2.5:
        //   m_t = 0.5*m_prev + 0.5*2.5. m_prev (after step 1) = 0.5*2.0 = 1.0
        //   m_t = 0.5*1.0 + 0.5*2.5 = 1.75
        //   v_t = 0.5*v_prev + 0.5*2.5^2. v_prev (after step 1) = 0.5*4.0 = 2.0
        //   v_t = 0.5*2.0 + 0.5*6.25 = 4.125
        //   bias_corr1 = 1-0.5^2 = 0.75, bias_corr2 = 0.75
        //   m_hat = 1.75/0.75 ≈ 2.3333
        //   v_hat = 4.125/0.75 = 5.5
        //   denom = sqrt(5.5) + 1e-3 ≈ 2.34537
        //   step_2_param = -1.0 * m_hat/denom = step_1_param - 1.0*m_hat/denom
        //   step_1_param ≈ -1.999
        //   step_2_param ≈ -1.999 - 2.3333/2.34537 ≈ -1.999 - 0.9949 ≈ -2.994
        double m_hat = 1.75 / 0.75;
        double v_hat = 4.125 / 0.75;
        double denom = std::sqrt(v_hat) + 1e-3;
        // step 1 had g_filtered=2.0, m_t=1.0, v_t=2.0, b1_corr=b2_corr=0.5
        double step1_param = -1.0 * (1.0 / 0.5) / (std::sqrt(2.0 / 0.5) + 1e-3);
        double step2_param = step1_param - 1.0 * (m_hat / denom);
        for (auto& layer : m.layers) {
            auto params = layer->parameters();
            for (auto* p : params) {
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        ASSERT_NEAR((*p)[i][j], step2_param, 1e-6);
            }
        }
    });

    // -------------------------------------------------------------------------
    // (20) Gradients are zeroed after step (delegated to inner)
    // -------------------------------------------------------------------------
    run("(20) gradients are zeroed after step", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = GrokFast(std::make_unique<Adam>(0.1, 0.5, 0.5, 1e-3));
        opt.step(m);
        for (auto& layer : m.layers) {
            for (auto* g : layer->gradients()) {
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        ASSERT_NEAR((*g)[i][j], 0.0, 0.0);
            }
        }
    });

    // -------------------------------------------------------------------------
    // (21) Loss decreases over 100 steps with sustained gradient signal
    // -------------------------------------------------------------------------
    run("(21) 100-step loss reduction with multi-feature regression", []{
        std::srand(7);
        Model m;
        Dense* d = new Dense(3, 1);
        for (size_t i = 0; i < d->weights.rows; ++i)
            for (size_t j = 0; j < d->weights.cols; ++j)
                d->weights[i][j] = 0.3 * (2.0 * std::rand() / RAND_MAX - 1.0);
        for (size_t i = 0; i < d->bias.rows; ++i)
            for (size_t j = 0; j < d->bias.cols; ++j)
                d->bias[i][j] = 0.0;
        m.add_layer(d);

        // y = 2*x0 - 1*x1 + 0.5*x2
        std::vector<std::vector<double>> X = {
            {0.5, 1.0, 2.0}, {1.0, 0.5, 1.5}, {1.5, 0.0, 0.5}, {2.0, 0.5, 0.2},
            {0.1, 0.7, 1.3}, {0.7, 1.1, 0.9}, {1.3, 0.2, 1.8}, {1.9, 0.4, 1.1}};
        std::vector<double> y;
        for (auto& x : X) y.push_back(2*x[0] - 1*x[1] + 0.5*x[2]);

        auto opt = GrokFast(std::make_unique<Adam>(0.05, 0.9, 0.999, 1e-8));

        double loss0 = 0;
        for (int epoch = 0; epoch < 100; ++epoch) {
            double total_loss = 0;
            for (size_t s = 0; s < X.size(); ++s) {
                // Input is (1, 3): batch=1, in_features=3. Dense(3,1) weight is (1,3).
                Tensor inp(1, 3);
                inp[0][0] = X[s][0]; inp[0][1] = X[s][1]; inp[0][2] = X[s][2];
                Tensor pred = d->forward(inp);
                double err = pred[0][0] - y[s];
                total_loss += 0.5 * err * err;
                Tensor grad(1, 1);
                grad[0][0] = err;
                d->backward(grad, 0.0);
            }
            if (epoch == 0) loss0 = total_loss;
            opt.step(m);
        }
        double total_loss = 0;
        for (size_t s = 0; s < X.size(); ++s) {
            Tensor inp(1, 3);
            inp[0][0] = X[s][0]; inp[0][1] = X[s][1]; inp[0][2] = X[s][2];
            Tensor pred = d->forward(inp);
            double err = pred[0][0] - y[s];
            total_loss += 0.5 * err * err;
        }
        ASSERT(total_loss < 0.1 * loss0);  // ≥90% reduction
    });

    // -------------------------------------------------------------------------
    // (22) Mutation test surrogate: lambda=2.0 path actually filters
    //      We verify that without filtering (lambda=0), and with filtering
    //      (lambda=2), we get different results — confirming the filter
    //      is NOT a no-op.
    // -------------------------------------------------------------------------
    run("(22) lambda=2 differs from lambda=0 (filter is non-vacuous)", []{
        std::srand(31);
        Model m_nofilter, m_filter;
        build_dense_model(m_nofilter, 2, 2, 0.3);
        build_dense_model(m_filter, 2, 2, 0.3);
        GrokFast opt_no(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 0.0, 0.98);
        GrokFast opt_yes(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 2.0, 0.98);
        for (int step = 0; step < 3; ++step) {
            // Mixed-sign grads to exercise buf
            for (auto& layer : m_nofilter.layers) {
                auto grads = layer->gradients();
                for (auto* g : grads) {
                    for (size_t i = 0; i < g->rows; ++i)
                        for (size_t j = 0; j < g->cols; ++j)
                            (*g)[i][j] = (std::rand() % 2 == 0) ? 1.0 : -1.0;
                }
            }
            for (size_t li = 0; li < m_nofilter.layers.size(); ++li) {
                auto gn = m_nofilter.layers[li]->gradients();
                auto gf = m_filter.layers[li]->gradients();
                for (size_t k = 0; k < gn.size(); ++k) {
                    for (size_t i = 0; i < gn[k]->rows; ++i)
                        for (size_t j = 0; j < gn[k]->cols; ++j)
                            (*gf[k])[i][j] = (*gn[k])[i][j];
                }
            }
            opt_no.step(m_nofilter);
            opt_yes.step(m_filter);
        }
        auto vn = snap(m_nofilter), vf = snap(m_filter);
        double dist2 = 0;
        for (size_t i = 0; i < vn.size(); ++i) {
            double d = vn[i] - vf[i];
            dist2 += d * d;
        }
        ASSERT(dist2 > 1e-6);
    });

    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}