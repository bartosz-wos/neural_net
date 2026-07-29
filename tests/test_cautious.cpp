// ============================================================================
// Cautious Optimizer test suite — Liang et al. 2024 (arXiv:2411.16085)
//
// Algorithm under test:
//   For each parameter:
//     1. snapshot param_before
//     2. inner.step(model) applies the inner optimizer's update direction
//        u_t = (param_after - param_before) / lr  (default inner: AdamW)
//     3. mask_t = (u_t * g_t > 0)  ∈ {0, 1}^N  (1 when direction agrees)
//     4. mask_mean_t = max(mask_t.mean(), eps_mask)   density compensation
//     5. restore param = param_before
//     6. param -= lr * (u_t * mask_t / mask_mean_t)
//
// The "one line of code" trick: the mask zeroes out update entries that
// would move the parameter against the gradient, then re-normalizes so
// the average step magnitude is preserved (compensation factor = 1/sum).
//
// Defaults: eps_mask = 1e-3, inner = AdamW(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <functional>
#include <memory>
#include "nn/optimizers/cautious.h"
#include "nn/optimizers/optimizer.h"
#include "nn/optimizers/adam_mini.h"  // not used, just keep header inclusion honest
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
    std::cout << "\n" << name << "\n";
    body();
}

// ----- helpers -----
static void build_dense_model(Model& m, size_t in_f, size_t out_f, double init_value) {
    Dense* d = new Dense(in_f, out_f);
    // set weights and bias to a constant
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

static void set_grads_alternating(Model& m, double a, double b) {
    for (auto& layer : m.layers) {
        auto grads = layer->gradients();
        for (auto* g : grads) {
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    (*g)[i][j] = ((i + j) % 2 == 0) ? a : b;
        }
    }
}


int main() {
    std::cout << std::setprecision(10);
    std::cout << "=== Cautious Optimizer Tests ===\n";

    // -------------------------------------------------------------------------
    // (1) Defaults round-trip
    // -------------------------------------------------------------------------
    run("(1) defaults round-trip", []{
        auto inner = std::make_unique<Adam>();
        Cautious opt(std::move(inner), 1e-3);
        ASSERT_NEAR(opt.get_eps_mask(), 1e-3, 0.0);
        ASSERT_NEAR(opt.get_lr(), 1e-3, 0.0);
        ASSERT(opt.handles_weight_decay() == false);  // Adam doesn't do WD
        ASSERT(opt.last_num_params_updated() == 0);   // no step yet
    });

    // -------------------------------------------------------------------------
    // (2) Non-default constructor
    // -------------------------------------------------------------------------
    run("(2) non-default constructor", []{
        auto inner = std::make_unique<Adam>(2e-3, 0.85, 0.999, 1e-6);
        Cautious opt(std::move(inner), 5e-4);
        ASSERT_NEAR(opt.get_eps_mask(), 5e-4, 0.0);
        ASSERT_NEAR(opt.get_lr(), 2e-3, 0.0);
    });

    // -------------------------------------------------------------------------
    // (3) Null inner throws
    // -------------------------------------------------------------------------
    run("(3) null inner throws", []{
        bool threw = false;
        try { Cautious opt(nullptr); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (4) eps_mask validation throws
    // -------------------------------------------------------------------------
    run("(4) eps_mask validation throws", []{
        bool threw = false;
        try { Cautious opt(std::make_unique<Adam>(), 0.0); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        threw = false;
        try { Cautious opt(std::make_unique<Adam>(), -1e-3); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (5) lr setter validation
    // -------------------------------------------------------------------------
    run("(5) lr setter validation throws", []{
        Cautious opt(std::make_unique<Adam>());
        bool threw = false;
        try { opt.set_lr(-0.001); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (6) eps_mask setter validation
    // -------------------------------------------------------------------------
    run("(6) eps_mask setter validation throws", []{
        Cautious opt(std::make_unique<Adam>());
        bool threw = false;
        try { opt.set_eps_mask(0.0); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
        threw = false;
        try { opt.set_eps_mask(-0.5); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });

    // -------------------------------------------------------------------------
    // (7) Closed-form first step with all-positive gradient → mask = 1, mean = 1
    //     → Cautious must produce IDENTICAL params to wrapped AdamW
    // -------------------------------------------------------------------------
    run("(7) all-positive grad: Cautious == Adam (mask=1 everywhere)", []{
        // Reference Adam
        Model m_ref;
        build_dense_model(m_ref, 2, 2, 0.0);
        set_grads_to_constant(m_ref, 1.0);
        Adam adam_ref(1.0, 0.5, 0.5, 1e-3);
        adam_ref.step(m_ref);

        // Cautious-wrapped Adam
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto inner = std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3);
        Cautious opt(std::move(inner), 1e-3);
        opt.step(m);

        // Compare weights
        for (size_t li = 0; li < m.layers.size(); ++li) {
            auto p_ref = m_ref.layers[li]->parameters();
            auto p     = m.layers[li]->parameters();
            for (size_t k = 0; k < p_ref.size(); ++k) {
                for (size_t i = 0; i < p_ref[k]->rows; ++i)
                    for (size_t j = 0; j < p_ref[k]->cols; ++j)
                        ASSERT_NEAR((*p_ref[k])[i][j], (*p[k])[i][j], 1e-12);
            }
        }
    });

    // -------------------------------------------------------------------------
    // (8) Closed-form first step with alternating ±1 gradient
    //     For AdamW with lr=1, b1=0.5, b2=0.5, eps=1e-3, init=0, g=±1:
    //       m_t = 0.5 * 0 + 0.5 * g = 0.5 * g
    //       v_t = 0.5 * 0 + 0.5 * 1 = 0.5
    //       bias_corr1 = 0.5, bias_corr2 = 0.5
    //       m_hat = m_t / 0.5 = g
    //       v_hat = v_t / 0.5 = 1
    //       update direction u_t = m_hat / (sqrt(v_hat) + eps) = g / (1 + 1e-3)
    //                          ≈ 0.999 g
    //     Cautious mask: mask = (u_t * g > 0) = (g² > 0) = 1 always (since u_t ∝ g)
    //     So even with alternating signs, the mask is uniformly 1!
    //     (This is because the AdamW update direction is proportional to g, so
    //     sign-agreement is automatic.)
    //
    //     To test the masking, we need a scenario where u_t ∝ g in the FIRST step
    //     but might NOT in later steps. We use a contrived scenario: g=[+1,-1] then
    //     g=[-1,+1] across two steps. That makes later-step u_t not proportional to
    //     current g.
    //
    //     For closed-form single-step, we use a one-D-per-row input where Adam's
    //     m coexists with opposing g in a different row → u_t might disagree.
    //
    //     Actually the simplest: when g has both signs but the bias-correction
    //     machinery is simple, m_t is just 0.5*g, so u_t = m_hat / (sqrt(v_hat)+eps)
    //     = g / (1+eps) ≈ g. So mask is 1 everywhere even with mixed-sign grad.
    //
    //     For a hard test of THE MASKING ITSELF, we artificially construct a case
    //     where m_t and g_t disagree. We do this by accumulating momentum before
    //     the cautious step: pre-fill inner optimizer's m_t buffer to a known value
    //     that points opposite to g. But we don't expose inner state. So we test
    //     the MASKING INDIRECTLY by comparing Cautious vs Adam trajectories over
    //     multiple steps with mixed-sign gradients — they MUST differ when mean
    //     mask < 1.
    //
    //     For closed-form first-step, we fall back to: with g = constant, mask = 1
    //     and Cautious = Adam. This is what test (7) verifies.
    // -------------------------------------------------------------------------
    run("(8) closed-form first-step with uniform g: mask mean = 1, no compensation", []{
        // Verify the per-param mask stats accumulator: after a step with uniform positive
        // gradient, the recorded mask_sum / n_entries should produce mean ≈ 1.
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto inner = std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3);
        Cautious opt(std::move(inner), 1e-3);
        opt.step(m);

        // The mask semantics: for AdamW with g=1 throughout, m_1 = 0.5, m_hat_1 = 1,
        // u_t = 1 / (1 + eps) > 0. So mask = (u_t * g > 0) = 1 everywhere.
        // recorded stats: mask_sum = num_entries, mask_mean = 1.
        // We verify via `last_stats()` returning the per-param (mask_sum, n_entries)
        // and the total mean = sum(mask_sum) / sum(n_entries).
        auto total_stats = opt.total_mask_stats();
        ASSERT_NEAR(total_stats.first, total_stats.second, 1e-9);  // mean == 1
    });

    // -------------------------------------------------------------------------
    // (9) Mixed-sign gradient over multiple steps → mask mean < 1 → Cautious
    //     takes a different trajectory than Adam.
    // -------------------------------------------------------------------------
    run("(9) mixed-sign grad: Cautious differs from Adam over multiple steps", []{
        // Set up identical models with same init
        Model m_cautious;
        build_dense_model(m_cautious, 2, 2, 0.5);
        Model m_adam;
        build_dense_model(m_adam, 2, 2, 0.5);

        // Print param snapshots
        auto snap = [](const Model& m) {
            std::vector<double> v;
            for (auto& layer : m.layers) {
                for (auto* p : layer->parameters())
                    for (size_t i = 0; i < p->rows; ++i)
                        for (size_t j = 0; j < p->cols; ++j)
                            v.push_back((*p)[i][j]);
            }
            return v;
        };
        auto v0 = snap(m_cautious);
        auto v0_adam = snap(m_adam);

        // Same params initially
        for (size_t i = 0; i < v0.size(); ++i)
            ASSERT_NEAR(v0[i], v0_adam[i], 1e-12);

        // Drive identical mixed-sign gradients
        auto opt_c = Cautious(std::make_unique<Adam>(0.1, 0.0, 0.0, 1e-3), 1e-3);
        Adam opt_a(0.1, 0.0, 0.0, 1e-3);

        for (int step = 0; step < 5; ++step) {
            // Reset gradients to mixed-sign pattern
            set_grads_alternating(m_cautious, 1.0, -1.0);
            set_grads_alternating(m_adam,     1.0, -1.0);
            opt_c.step(m_cautious);
            opt_a.step(m_adam);
        }

        // With β1=β2=0, no momentum accumulation, m_hat = g, so u_t = g / (1+eps).
        // Then mask = (g * g > 0) = 1 always. So Cautious should match Adam bit-exact.
        // This is a SUBTLE TEST: we need to force a scenario where u_t disagrees with g.
        // With β1=0, m_t = g, so m_hat = g, so u_t ∝ g. Inevitable mask=1.
        //
        // For Cautious to differ from Adam, we need momentum accumulation to
        // cause u_t to lag. Re-run with β1=0.9 (default).
        auto opt_c2 = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
        Adam opt_a2(0.1, 0.9, 0.999, 1e-8);

        // Reset both models
        Model m_c2, m_a2;
        build_dense_model(m_c2, 2, 2, 0.5);
        build_dense_model(m_a2, 2, 2, 0.5);

        // Use random ±1 gradients (NOT a fixed alternating pattern) so that
        // m_t diverges from g_t and the mask drops below 1 sometimes.
        std::srand(42);
        for (int step = 0; step < 8; ++step) {
            for (auto& ml : m_c2.layers) {
                auto grads = ml->gradients();
                for (auto* g : grads) {
                    for (size_t i = 0; i < g->rows; ++i)
                        for (size_t j = 0; j < g->cols; ++j)
                            (*g)[i][j] = (std::rand() % 2 == 0) ? 1.0 : -1.0;
                }
            }
            // Copy grads to m_a2
            for (size_t li = 0; li < m_c2.layers.size(); ++li) {
                auto gc = m_c2.layers[li]->gradients();
                auto ga = m_a2.layers[li]->gradients();
                for (size_t k = 0; k < gc.size(); ++k) {
                    for (size_t i = 0; i < gc[k]->rows; ++i)
                        for (size_t j = 0; j < gc[k]->cols; ++j)
                            (*ga[k])[i][j] = (*gc[k])[i][j];
                }
            }
            opt_c2.step(m_c2);
            opt_a2.step(m_a2);
        }

        auto vc = snap(m_c2);
        auto va = snap(m_a2);
        // They should differ when mask mean < 1 over these steps.
        // Compute L2 distance.
        double dist2 = 0;
        for (size_t i = 0; i < vc.size(); ++i) {
            double d = vc[i] - va[i];
            dist2 += d * d;
        }
        // We expect SOME difference. Threshold: > 1e-6.
        ASSERT(dist2 > 1e-6);
    });

    // -------------------------------------------------------------------------
    // (10) End-to-end: linear regression y = 2x with Cautious(Adam)
    // -------------------------------------------------------------------------
    run("(10) end-to-end Cautious reduces loss on y=2x", []{
        std::srand(42);
        Model m;
        Dense* d = new Dense(1, 1);
        // init small random
        for (size_t i = 0; i < d->weights.rows; ++i)
            for (size_t j = 0; j < d->weights.cols; ++j)
                d->weights[i][j] = 0.3 * (2.0 * std::rand() / RAND_MAX - 1.0);
        for (size_t i = 0; i < d->bias.rows; ++i)
            for (size_t j = 0; j < d->bias.cols; ++j)
                d->bias[i][j] = 0.0;
        m.add_layer(d);

        // Generate tiny dataset: y = 2x on 8 points
        std::vector<std::vector<double>> X = {{0.5}, {1.0}, {1.5}, {2.0},
                                              {0.1}, {0.7}, {1.3}, {1.9}};
        std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 0.2, 1.4, 2.6, 3.8};

        auto opt = Cautious(std::make_unique<Adam>(0.05, 0.9, 0.999, 1e-8), 1e-3);

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

        // Compute final loss
        double total_loss = 0;
        for (size_t s = 0; s < X.size(); ++s) {
            Tensor inp(1, 1);
            inp[0][0] = X[s][0];
            Tensor pred = d->forward(inp);
            double err = pred[0][0] - y[s];
            total_loss += 0.5 * err * err;
        }
        ASSERT(total_loss < 0.5 * loss0);  // at least 50% reduction
    });

    // -------------------------------------------------------------------------
    // (11) handles_weight_decay delegates to inner
    // -------------------------------------------------------------------------
    run("(11) handles_weight_decay delegates to inner", []{
        // Adam doesn't handle WD
        Cautious opt_c(std::make_unique<Adam>());
        ASSERT(opt_c.handles_weight_decay() == false);
        // AdamW-style: we don't have a built-in AdamW in this repo's optimizer.h,
        // so use a custom one. Sgd doesn't handle WD either.
        Cautious opt_sgd(std::make_unique<SGD>(0.1));
        ASSERT(opt_sgd.handles_weight_decay() == false);
    });

    // -------------------------------------------------------------------------
    // (12) Param/grad count mismatch guards
    // -------------------------------------------------------------------------
    run("(12) works with a model that has no parameters", []{
        // Empty model — should not crash
        Model m;
        auto opt = Cautious(std::make_unique<Adam>());
        opt.step(m);
        ASSERT(opt.last_num_params_updated() == 0);
    });

    // -------------------------------------------------------------------------
    // (13) Determinism: two fresh instances produce bit-identical updates
    // -------------------------------------------------------------------------
    run("(13) determinism over 5 steps", []{
        std::srand(0);
        auto run_one = []() {
            Model m;
            build_dense_model(m, 2, 2, 0.3);
            auto opt = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
            for (int step = 0; step < 5; ++step) {
                set_grads_alternating(m, 1.0, -1.0);
                opt.step(m);
            }
            std::vector<double> v;
            for (auto& layer : m.layers)
                for (auto* p : layer->parameters())
                    for (size_t i = 0; i < p->rows; ++i)
                        for (size_t j = 0; j < p->cols; ++j)
                            v.push_back((*p)[i][j]);
            return v;
        };
        auto v1 = run_one();
        auto v2 = run_one();
        ASSERT(v1.size() == v2.size());
        for (size_t i = 0; i < v1.size(); ++i)
            ASSERT_NEAR(v1[i], v2[i], 1e-12);
    });

    // -------------------------------------------------------------------------
    // (14) State diagnostics: has_state after first step
    // -------------------------------------------------------------------------
    run("(14) state diagnostics after first step", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = Cautious(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3), 1e-3);
        auto* layer_ptr = m.layers[0].get();
        ASSERT(opt.has_state(layer_ptr, 0) == false);
        ASSERT(opt.has_state(layer_ptr, 1) == false);
        opt.step(m);
        // After step, stats should be populated for each parameter
        ASSERT(opt.has_state(layer_ptr, 0) == true);
        ASSERT(opt.has_state(layer_ptr, 1) == true);
        ASSERT(opt.last_num_params_updated() == 2);  // weights + bias
    });

    // -------------------------------------------------------------------------
    // (15) Stats accessor reasonable values
    // -------------------------------------------------------------------------
    run("(15) stats after step: total mask_sum / total count", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = Cautious(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3), 1e-3);
        opt.step(m);

        auto stats = opt.total_mask_stats();
        // 5 params total (4 weights + 1 bias? No: weights (2,2) = 4 + bias (1,2) = 2 → 6)
        // Wait: Dense(2, 2) has weights (2,2) = 4 entries and bias (1,2) = 2 entries → 6 entries
        ASSERT(stats.second == 6);  // count
        ASSERT_NEAR(stats.first, stats.second, 1e-6);  // mean = 1 (all-positive grad)
    });

    // -------------------------------------------------------------------------
    // (16) set_lr forwards to inner
    // -------------------------------------------------------------------------
    run("(16) set_lr forwards to inner", []{
        Cautious opt(std::make_unique<Adam>(0.5, 0.9, 0.999, 1e-8));
        ASSERT_NEAR(opt.get_lr(), 0.5, 0.0);
        opt.set_lr(0.123);
        ASSERT_NEAR(opt.get_lr(), 0.123, 0.0);
    });

    // -------------------------------------------------------------------------
    // (17) Inner accessor returns the inner optimizer (read-only)
    // -------------------------------------------------------------------------
    run("(17) inner() returns the wrapped optimizer", []{
        auto inner = std::make_unique<Adam>(0.42, 0.9, 0.999, 1e-8);
        Cautious opt(std::move(inner));
        ASSERT(opt.inner() != nullptr);
        auto* adam = dynamic_cast<Adam*>(opt.inner());
        ASSERT(adam != nullptr);
        ASSERT_NEAR(adam->lr, 0.42, 0.0);
    });

    // -------------------------------------------------------------------------
    // (18) Verify mask-skipping behavior: when grad is exactly 0, AdamW does
    //      a finite no-op-step as lr * 0 = 0, but mask computation uses g.
    //      With g=0, mask = (u_t * 0 > 0) = 0 everywhere.
    //      → All-zero mask → density compensation clamps to eps_mask → final
    //        mask applied to u_t is 0/eps_mask = 0 → no step.
    // -------------------------------------------------------------------------
    run("(18) zero gradient: no parameter change", []{
        Model m;
        build_dense_model(m, 2, 2, 0.5);
        // Snapshot
        std::vector<double> before;
        for (auto& layer : m.layers)
            for (auto* p : layer->parameters())
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        before.push_back((*p)[i][j]);

        // Set grads to 0
        for (auto& layer : m.layers) {
            auto grads = layer->gradients();
            for (auto* g : grads) {
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        (*g)[i][j] = 0.0;
            }
        }
        auto opt = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
        opt.step(m);

        std::vector<double> after;
        for (auto& layer : m.layers)
            for (auto* p : layer->parameters())
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        after.push_back((*p)[i][j]);

        for (size_t i = 0; i < before.size(); ++i)
            ASSERT_NEAR(before[i], after[i], 1e-12);
    });

    // -------------------------------------------------------------------------
    // (19) Mask log records per-parameter stats
    // -------------------------------------------------------------------------
    run("(19) per-parameter stats: mask_sum on uniform grad = param size", []{
        Model m;
        build_dense_model(m, 2, 2, 0.0);
        set_grads_to_constant(m, 1.0);
        auto opt = Cautious(std::make_unique<Adam>(1.0, 0.5, 0.5, 1e-3), 1e-3);
        auto* layer_ptr = m.layers[0].get();
        opt.step(m);

        // Parameter 0 = weights (2x2 = 4 entries). Mask sum = 4 (all 1).
        // Parameter 1 = bias (1x2 = 2 entries). Mask sum = 2 (all 1).
        auto ps0 = opt.get_param_stats(layer_ptr, 0);
        auto ps1 = opt.get_param_stats(layer_ptr, 1);
        ASSERT_NEAR(ps0.first,  4.0, 1e-9);
        ASSERT_NEAR(ps0.second, 4.0, 1e-9);
        ASSERT_NEAR(ps1.first,  2.0, 1e-9);
        ASSERT_NEAR(ps1.second, 2.0, 1e-9);
    });

    // -------------------------------------------------------------------------
    // (20) Cautious vs Adam on real mixed-sign scenario: trajectories differ
    // -------------------------------------------------------------------------
    run("(20) Cautious vs Adam signature: 4-step mixed-sign differs", []{
        std::srand(7);
        auto build = []() {
            Model m;
            Dense* d = new Dense(2, 2);
            for (size_t i = 0; i < d->weights.rows; ++i)
                for (size_t j = 0; j < d->weights.cols; ++j)
                    d->weights[i][j] = 0.3 * (2.0 * std::rand() / RAND_MAX - 1.0);
            for (size_t i = 0; i < d->bias.rows; ++i)
                for (size_t j = 0; j < d->bias.cols; ++j)
                    d->bias[i][j] = 0.1;
            m.add_layer(d);
            return m;
        };
        Model m_c = build();
        Model m_a = build();
        // Snapshot values
        auto snap = [](const Model& m) {
            std::vector<double> v;
            for (auto& layer : m.layers)
                for (auto* p : layer->parameters())
                    for (size_t i = 0; i < p->rows; ++i)
                        for (size_t j = 0; j < p->cols; ++j)
                            v.push_back((*p)[i][j]);
            return v;
        };
        auto init_c = snap(m_c);
        auto init_a = snap(m_a);
        // Copy init into both
        auto copy_in = [](Model& m, const std::vector<double>& v) {
            size_t k = 0;
            for (auto& layer : m.layers)
                for (auto* p : layer->parameters())
                    for (size_t i = 0; i < p->rows; ++i)
                        for (size_t j = 0; j < p->cols; ++j)
                            (*p)[i][j] = v[k++];
        };
        copy_in(m_a, init_c);  // both models identical

        auto opt_c = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
        Adam opt_a(0.1, 0.9, 0.999, 1e-8);

        for (int step = 0; step < 5; ++step) {
            // Random-ish mixed-sign gradient
            std::srand(100 + step);
            for (auto& layer : m_c.layers) {
                auto grads = layer->gradients();
                for (auto* g : grads)
                    for (size_t i = 0; i < g->rows; ++i)
                        for (size_t j = 0; j < g->cols; ++j)
                            (*g)[i][j] = (std::rand() % 2 == 0) ? 1.0 : -1.0;
            }
            // Copy grads to m_a
            for (size_t li = 0; li < m_c.layers.size(); ++li) {
                auto gc = m_c.layers[li]->gradients();
                auto ga = m_a.layers[li]->gradients();
                for (size_t k = 0; k < gc.size(); ++k) {
                    for (size_t i = 0; i < gc[k]->rows; ++i)
                        for (size_t j = 0; j < gc[k]->cols; ++j)
                            (*ga[k])[i][j] = (*gc[k])[i][j];
                }
            }
            opt_c.step(m_c);
            opt_a.step(m_a);
        }

        auto vc = snap(m_c);
        auto va = snap(m_a);
        double dist2 = 0;
        for (size_t i = 0; i < vc.size(); ++i) {
            double d = vc[i] - va[i];
            dist2 += d * d;
        }
        // With random ±1 gradients over 5 steps, mask mean < 1 sometimes, so
        // Cautious should diverge from Adam.
        ASSERT(dist2 > 1e-6);
    });

    // -------------------------------------------------------------------------
    // (21) Signature: with eps_mask=very large, mask denominator becomes 1 always
    //      → Cautious = Adam (because mask / huge-eps = 0 except where mask=1,
    //      where it's 1/huge ≈ 0). Actually no: with eps_mask=1e6, mask_mean
    //      is clamped to 1e6, so mask / 1e6 = 1e-6 for agreeing entries → near-zero step.
    //      Better: with eps_mask=very small = eps, mask_mean = real_mean, no clamp.
    //      Equality with Adam holds when ALL mask entries are 1. Test with constant grad.
    // -------------------------------------------------------------------------
    run("(21) eps_mask setter round-trip", []{
        Cautious opt(std::make_unique<Adam>());
        opt.set_eps_mask(5e-4);
        ASSERT_NEAR(opt.get_eps_mask(), 5e-4, 0.0);
    });

    // -------------------------------------------------------------------------
    // (22) set_lr on inner Adam with new value should propagate
    // -------------------------------------------------------------------------
    run("(22) set_lr round-trip on inner", []{
        Cautious opt(std::make_unique<Adam>(0.001, 0.9, 0.999, 1e-8));
        opt.set_lr(0.007);
        ASSERT_NEAR(opt.get_lr(), 0.007, 0.0);
    });

    // -------------------------------------------------------------------------
    // (23) Cautious step on multi-layer model: each layer gets its state
    // -------------------------------------------------------------------------
    run("(23) multi-layer model: each layer tracked", []{
        Model m;
        Dense* d1 = new Dense(2, 3);
        Dense* d2 = new Dense(3, 1);
        for (size_t i = 0; i < d1->weights.rows; ++i)
            for (size_t j = 0; j < d1->weights.cols; ++j)
                d1->weights[i][j] = 0.1;
        for (size_t i = 0; i < d2->weights.rows; ++i)
            for (size_t j = 0; j < d2->weights.cols; ++j)
                d2->weights[i][j] = 0.1;
        m.add_layer(d1);
        m.add_layer(d2);

        // Initialize gradients
        for (auto& layer : m.layers) {
            auto grads = layer->gradients();
            for (auto* g : grads)
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        (*g)[i][j] = 1.0;
        }
        auto opt = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
        opt.step(m);

        // Each layer has 2 params (weights + bias) → 4 params total updated
        ASSERT(opt.last_num_params_updated() == 4);
    });

    // -------------------------------------------------------------------------
    // (24) Density compensation: total step magnitude preserved when mask < 1
    // -------------------------------------------------------------------------
    run("(24) density compensation: per-entry compensation matches Adam step", []{
        // Construct a scenario where mask=0.5 (4 entries, 2 dropped, 2 surviving).
        // Build Adam's m_t with positive gradient, then flip signs for half
        // the entries. Use a 4-param setting where we can control which
        // entries get masked.
        //
        // Setup: 2x2 weights with init=0. After 1 step with all g=+1, m_t=0.1.
        // Step 2: g = [+1, +1, -1, -1] (half matches, half disagrees).
        //   For (0,0): m_2 = 0.9*0.1 + 0.1*1 = 0.19, sign +. g=+1. mask=1.
        //   For (0,1): same. mask=1.
        //   For (1,0): m_2 = 0.9*0.1 + 0.1*(-1) = -0.01, sign -. g=-1. mask=1.
        //   For (1,1): same. mask=1.
        // mask=1 everywhere. So we can't force mask=0.5 with bias-corrected Adam.
        //
        // Alternative: Use Cautious with a "frozen" inner state by setting
        // pre-filled m_t via a long warmup. With 50 warmup steps of all +1,
        // m_t saturates. Then a sign flip won't move m_t much.
        //
        // Actually, the cleanest test is: skip the strict closed-form and
        // verify the COMPENSATED property via a controlled inner Adam state.
        // We construct the experiment where:
        //   - warmup with +1 gradient builds m_t (positive)
        //   - then induce sign disagreement by FLIPPING gradient for half entries
        //   - with m_t still positive for those entries, direction_t ∝ +1,
        //     but g = -1, so mask = 0 for those entries.
        //   - For the OTHER half where g stayed +1, direction_t ∝ +1 and g = +1,
        //     so mask = 1.
        //   - mask_mean = 0.5, compensation = 2x.
        //
        // To make this work, we need m_t to BE positive for all entries at the
        // start of step 2 (i.e., m_t does not change much with the new gradient).
        // With β1=0.9, after 50 warmup steps, m_t ≈ 1.
        // Step 51 with g_flipped_half: for the flipped entries, g=-1.
        //   m_51 = 0.9*1 + 0.1*(-1) = 0.8, still positive.
        //   direction_t ∝ 0.8 > 0. g = -1. mask = (0.8 * -1 > 0) = 0. ✓
        // For the non-flipped entries, g=+1.
        //   m_51 = 0.9*1 + 0.1*1 = 1.0, positive.
        //   direction_t ∝ 1.0 > 0. g = +1. mask = 1. ✓
        // mask_mean = 0.5, compensation = 2x.
        //
        // For the surviving entries, the compensated Cautious step should be
        // 2x the Adam step (since mask/mask_mean = 1/0.5 = 2).
        // For the dropped entries, Cautious step = 0.
        // So |Adam_step - Cautious_step| for surviving entries = Adam_step
        // (since Cautious = 2x Adam, but Adam doesn't compensate so step = Adam_step).
        // Without compensation, Cautious = 0.5x Adam, so diff = 0.5x Adam_step.
        //
        // We test this: surviving entries have |Adam_step - Cautious_step| ≈ Adam_step.
        Model m_c, m_a;
        build_dense_model(m_c, 2, 2, 0.0);
        build_dense_model(m_a, 2, 2, 0.0);
        set_grads_to_constant(m_c, 1.0);
        set_grads_to_constant(m_a, 1.0);
        auto opt_c = Cautious(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), 1e-3);
        Adam opt_a(0.1, 0.9, 0.999, 1e-8);
        // 50 warmup steps with all +1 gradient
        for (int step = 0; step < 50; ++step) {
            set_grads_to_constant(m_c, 1.0);
            set_grads_to_constant(m_a, 1.0);
            opt_c.step(m_c);
            opt_a.step(m_a);
        }
        // Now flip half the gradients
        for (auto& layer : m_c.layers) {
            auto grads = layer->gradients();
            for (auto* g : grads)
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        (*g)[i][j] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
        for (auto& layer : m_a.layers) {
            auto grads = layer->gradients();
            for (auto* g : grads)
                for (size_t i = 0; i < g->rows; ++i)
                    for (size_t j = 0; j < g->cols; ++j)
                        (*g)[i][j] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
        // Snapshot before step
        auto snap = [](const Model& m) {
            std::vector<double> v;
            for (auto& layer : m.layers)
                for (auto* p : layer->parameters())
                    for (size_t i = 0; i < p->rows; ++i)
                        for (size_t j = 0; j < p->cols; ++j)
                            v.push_back((*p)[i][j]);
            return v;
        };
        auto before_c = snap(m_c);
        auto before_a = snap(m_a);
        opt_c.step(m_c);
        opt_a.step(m_a);
        auto after_c = snap(m_c);
        auto after_a = snap(m_a);
        // Mask status
        auto stats = opt_c.total_mask_stats();
        // We expect mask_sum < count (some entries masked) IF compensation is in effect.
        // Verify compensation: for the surviving entries, |Cautious_step - Adam_step|
        // is approximately equal to Adam_step magnitude (since Cautious compensates
        // to 2x Adam step but Adam doesn't compensate).
        // For the dropped entries, |Cautious_step - Adam_step| ≈ Adam_step.
        // So overall, the L2 distance between cautious and adam steps should be
        // at least 50% of Adam's total step magnitude.
        double dist2 = 0, mag_a = 0;
        for (size_t i = 0; i < after_c.size(); ++i) {
            double d_c = after_c[i] - before_c[i];
            double d_a = after_a[i] - before_a[i];
            dist2 += (d_c - d_a) * (d_c - d_a);
            mag_a += std::abs(d_a);
        }
        // mag_a is Adam's total step magnitude; dist2 is the squared difference.
        // If compensation is active: Cautious_step ≈ 2*Adam_step for surviving,
        // 0 for dropped. Adam_step is the same magnitude. So diff per entry:
        //   surviving: |2*Adam - Adam| = Adam, so contributes Adam^2
        //   dropped: |0 - Adam| = Adam, so contributes Adam^2
        // Total diff per entry: 2*Adam^2.
        // L2 distance sqrt = sqrt(2) * Adam step magnitude.
        // Without compensation: surviving step = 0.5*Adam, dropped = 0.
        //   surviving: |0.5*Adam - Adam| = 0.5*Adam
        //   dropped: |0 - Adam| = Adam
        // Per-entry sum: 0.25*Adam^2 + 1*Adam^2 = 1.25*Adam^2.
        // L2 distance sqrt = sqrt(1.25) * Adam = 1.118 * Adam.
        // With COMPENSATION the total step magnitude (sum of |steps|) of Cautious
        // should match Adam's total step magnitude (masked entries contribute 0,
        // surviving entries contribute 2x their Adam step, so total = 1*Adam).
        // Without compensation: total = 0.5 * Adam magnitude.
        // Test: total_Cautious_step / total_Adam_step should be ~1.0 with compensation.
        double mag_c = 0;
        for (size_t i = 0; i < after_c.size(); ++i) {
            double d_c = after_c[i] - before_c[i];
            mag_c += std::abs(d_c);
        }
        // Some masking must have happened for the test to be meaningful.
        ASSERT(stats.first < stats.second);
        // Compensation ratio: with comp = 1.0, without comp = 0.5.
        // Use a loose tolerance: 0.7 < ratio < 1.3 (catches both off-by-2x bugs).
        // With mutation 2 (no comp), ratio ≈ 0.5 → fails.
        double ratio = (mag_a > 0) ? mag_c / mag_a : 1.0;
        ASSERT(ratio > 0.7);
        ASSERT(ratio < 1.3);
    });

    // -------------------------------------------------------------------------
    // (25) Cautious with SGD inner: still works (catches inner-type bug)
    // -------------------------------------------------------------------------
    run("(25) Cautious wraps SGD: lr inheritance from derived class", []{
        Cautious opt(std::make_unique<SGD>(0.07));
        ASSERT_NEAR(opt.get_lr(), 0.07, 0.0);
        opt.set_lr(0.13);
        ASSERT_NEAR(opt.get_lr(), 0.13, 0.0);
    });

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n=== Summary: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
