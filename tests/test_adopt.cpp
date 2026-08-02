#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include "nn/optimizers/adopt.h"
#include "nn/core/layer.h"
#include "nn/core/model.h"

namespace {
int passed = 0;
int failed = 0;
std::string current;

#define CHECK(expr) do { if (expr) { ++passed; } else { ++failed; std::cout << "[FAIL] " << current << ": " << #expr << " (line " << __LINE__ << ")\n"; } } while (0)
#define CHECK_NEAR(a,b,tol) do { double av=(a), bv=(b); if (std::abs(av-bv) <= (tol)) { ++passed; } else { ++failed; std::cout << "[FAIL] " << current << ": " << #a << "=" << av << " vs " << #b << "=" << bv << " (line " << __LINE__ << ")\n"; } } while (0)

void run(const std::string& name, const std::function<void()>& body) {
    current = name;
    std::cout << name << '\n';
    body();
}

Dense* add_dense(Model& model, size_t in, size_t out, double value) {
    auto* layer = new Dense(in, out);
    layer->weights.fill(value);
    layer->bias.fill(value);
    model.add_layer(layer);
    return layer;
}

void set_grads(Dense* layer, double value) {
    layer->grad_weights.fill(value);
    layer->grad_bias.fill(value);
}

class MalformedLayer : public Layer {
public:
    Tensor p1{1, 2};
    Tensor p2{1, 2};
    Tensor g1{1, 2};
    Tensor bad{2, 1};
    bool count_mismatch = false;
    bool shape_mismatch = false;

    Tensor forward(const Tensor& x) override { return x; }
    Tensor backward(const Tensor& g, double) override { return g; }
    std::vector<Tensor*> parameters() override { return count_mismatch ? std::vector<Tensor*>{&p1, &p2} : std::vector<Tensor*>{&p1}; }
    std::vector<Tensor*> gradients() override { return shape_mismatch ? std::vector<Tensor*>{&bad} : std::vector<Tensor*>{&g1}; }
    void zero_grad() override { g1.fill(0.0); bad.fill(0.0); }
    void update_weights(double) override {}
    Tensor get_weights() const override { return p1; }
    Tensor get_gradients() const override { return g1; }
    std::string name() const override { return "MalformedLayer"; }
};

void defaults_and_validation() {
    run("defaults and validation", [] {
        ADOPT opt;
        CHECK_NEAR(opt.get_lr(), 1e-3, 0.0);
        CHECK_NEAR(opt.get_beta1(), 0.9, 0.0);
        CHECK_NEAR(opt.get_beta2(), 0.9999, 0.0);
        CHECK_NEAR(opt.get_epsilon(), 1e-6, 0.0);
        CHECK_NEAR(opt.get_clip_exp(), 0.25, 0.0);
        CHECK_NEAR(opt.get_weight_decay(), 0.0, 0.0);
        CHECK(!opt.get_decoupled());
        CHECK(opt.get_t() == 1);
        CHECK(opt.handles_weight_decay());

        ADOPT custom(0.02, 0.5, 0.75, 1e-4, 0.5, 0.1, true);
        CHECK_NEAR(custom.get_lr(), 0.02, 0.0);
        CHECK_NEAR(custom.get_beta1(), 0.5, 0.0);
        CHECK_NEAR(custom.get_beta2(), 0.75, 0.0);
        CHECK_NEAR(custom.get_epsilon(), 1e-4, 0.0);
        CHECK_NEAR(custom.get_clip_exp(), 0.5, 0.0);
        CHECK_NEAR(custom.get_weight_decay(), 0.1, 0.0);
        CHECK(custom.get_decoupled());

        auto throws = [](const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        CHECK(throws([] { ADOPT(-1.0); }));
        CHECK(throws([] { ADOPT(1e-3, -0.1); }));
        CHECK(throws([] { ADOPT(1e-3, 1.0); }));
        CHECK(throws([] { ADOPT(1e-3, 0.9, -0.1); }));
        CHECK(throws([] { ADOPT(1e-3, 0.9, 1.0); }));
        CHECK(throws([] { ADOPT(1e-3, 0.9, 0.9999, 0.0); }));
        CHECK(throws([] { ADOPT(1e-3, 0.9, 0.9999, 1e-6, -0.1); }));
        CHECK(throws([] { ADOPT(1e-3, 0.9, 0.9999, 1e-6, 0.25, -0.1); }));

        CHECK(throws([&] { opt.set_lr(-1); }));
        CHECK(throws([&] { opt.set_beta1(1); }));
        CHECK(throws([&] { opt.set_beta2(1); }));
        CHECK(throws([&] { opt.set_epsilon(0); }));
        CHECK(throws([&] { opt.set_clip_exp(-0.1); }));
        CHECK(throws([&] { opt.set_weight_decay(-0.1); }));
        opt.set_lr(0.0); opt.set_beta1(0); opt.set_beta2(0); opt.set_clip_exp(0); opt.set_decoupled(true);
        CHECK_NEAR(opt.get_lr(), 0.0, 0.0);
        CHECK_NEAR(opt.get_beta1(), 0.0, 0.0);
        CHECK_NEAR(opt.get_beta2(), 0.0, 0.0);
        CHECK_NEAR(opt.get_clip_exp(), 0.0, 0.0);
        CHECK(opt.get_decoupled());
    });
}

void first_step_initializes_without_update() {
    run("first step initializes v without updating parameters", [] {
        Model model;
        Dense* d = add_dense(model, 3, 2, 0.5);
        ADOPT opt(0.1, 0.9, 0.5, 1e-6, 0.25, 0.0, false);
        CHECK(!opt.has_state(d));
        set_grads(d, 2.0);
        opt.step(model);
        CHECK(opt.has_state(d));
        CHECK(opt.num_params_with_state(d) == 2);
        CHECK(opt.get_t() == 2);
        CHECK_NEAR(d->weights[0][0], 0.5, 0.0);
        CHECK_NEAR(d->bias[0][0], 0.5, 0.0);
        CHECK_NEAR(opt.get_m(d, 0)[0][0], 0.0, 0.0);
        CHECK_NEAR(opt.get_v(d, 0)[0][0], 4.0, 0.0);
        CHECK(opt.get_m(d, 0).rows == d->weights.rows && opt.get_m(d, 0).cols == d->weights.cols);
        CHECK(opt.get_v(d, 1).rows == d->bias.rows && opt.get_v(d, 1).cols == d->bias.cols);
        CHECK_NEAR(d->grad_weights[0][0], 0.0, 0.0);
        CHECK_NEAR(d->grad_bias[0][0], 0.0, 0.0);
    });
}

void previous_second_moment_signature() {
    run("second step normalizes by previous v", [] {
        Model model;
        Dense* d = add_dense(model, 1, 1, 0.0);
        ADOPT opt(1.0, 0.0, 0.0, 1.0, 10.0, 0.0, false);
        set_grads(d, 2.0);
        opt.step(model);
        set_grads(d, 1.0);
        opt.step(model);
        CHECK_NEAR(d->weights[0][0], -0.5, 1e-12);
        CHECK_NEAR(d->bias[0][0], -0.5, 1e-12);
        CHECK_NEAR(opt.get_m(d, 0)[0][0], 0.5, 1e-12);
        CHECK_NEAR(opt.get_v(d, 0)[0][0], 1.0, 1e-12);
        CHECK(opt.get_t() == 3);
    });
}

void clipping_signature() {
    run("dynamic clipping bounds normalized gradient", [] {
        Model model;
        Dense* d = add_dense(model, 1, 1, 0.0);
        ADOPT opt(1.0, 0.0, 0.0, 1e-12, 0.25, 0.0, false);
        set_grads(d, 1.0); opt.step(model);
        set_grads(d, 1e6); opt.step(model);
        CHECK_NEAR(d->weights[0][0], -1.0, 1e-12);
        CHECK_NEAR(opt.get_m(d, 0)[0][0], 1.0, 1e-12);
        set_grads(d, -1e12); opt.step(model);
        CHECK_NEAR(d->weights[0][0], -1.0 + std::pow(2.0, 0.25), 1e-12);
        CHECK_NEAR(opt.get_m(d, 0)[0][0], -std::pow(2.0, 0.25), 1e-12);
    });
}

void weight_decay_modes() {
    run("coupled and decoupled weight decay", [] {
        Model coupled_model;
        Dense* c = add_dense(coupled_model, 1, 1, 2.0);
        ADOPT coupled(0.1, 0.0, 0.0, 1.0, 10.0, 0.5, false);
        set_grads(c, 0.0); coupled.step(coupled_model);
        CHECK_NEAR(coupled.get_v(c, 0)[0][0], 1.0, 1e-12); // (wd * p)^2
        set_grads(c, 0.0); coupled.step(coupled_model);
        CHECK_NEAR(c->weights[0][0], 1.9, 1e-12); // normalized coupled direction = 1

        Model decoupled_model;
        Dense* d = add_dense(decoupled_model, 1, 1, 2.0);
        ADOPT decoupled(0.1, 0.0, 0.0, 1.0, 10.0, 0.5, true);
        set_grads(d, 0.0); decoupled.step(decoupled_model);
        CHECK_NEAR(d->weights[0][0], 2.0, 0.0); // first step is initialization-only
        set_grads(d, 0.0); decoupled.step(decoupled_model);
        CHECK_NEAR(d->weights[0][0], 1.9, 1e-12);
        CHECK_NEAR(decoupled.get_v(d, 0)[0][0], 0.0, 0.0);
    });
}

void malformed_guards() {
    run("malformed layer guards", [] {
        {
            Model model; auto* bad = new MalformedLayer(); bad->count_mismatch = true; model.add_layer(bad);
            ADOPT opt; bool threw = false; try { opt.step(model); } catch (const std::logic_error&) { threw = true; }
            CHECK(threw);
        }
        {
            Model model; auto* bad = new MalformedLayer(); bad->shape_mismatch = true; model.add_layer(bad);
            ADOPT opt; bool threw = false; try { opt.step(model); } catch (const std::logic_error&) { threw = true; }
            CHECK(threw);
        }
    });
}

void state_isolation_and_determinism() {
    run("state isolation and deterministic trajectory", [] {
        Model a, b;
        Dense* a1 = add_dense(a, 2, 2, 0.3);
        Dense* a2 = add_dense(a, 2, 1, -0.2);
        Dense* b1 = add_dense(b, 2, 2, 0.3);
        Dense* b2 = add_dense(b, 2, 1, -0.2);
        ADOPT oa(0.05, 0.5, 0.75, 1e-3, 0.25, 0.01, true);
        ADOPT ob(0.05, 0.5, 0.75, 1e-3, 0.25, 0.01, true);
        for (int step = 0; step < 5; ++step) {
            double g1 = 0.2 + step * 0.1;
            double g2 = -0.4 + step * 0.03;
            set_grads(a1, g1); set_grads(a2, g2);
            set_grads(b1, g1); set_grads(b2, g2);
            oa.step(a); ob.step(b);
        }
        CHECK(oa.has_state(a1) && oa.has_state(a2));
        CHECK(&oa.get_m(a1, 0) != &oa.get_m(a2, 0));
        CHECK_NEAR(a1->weights[0][0], b1->weights[0][0], 0.0);
        CHECK_NEAR(a2->weights[0][0], b2->weights[0][0], 0.0);
        CHECK_NEAR(oa.get_v(a1, 0)[0][0], ob.get_v(b1, 0)[0][0], 0.0);
    });
}

void training_reduces_loss() {
    run("linear regression loss decreases", [] {
        Model model;
        Dense* d = add_dense(model, 1, 1, 0.0);
        ADOPT opt(0.05, 0.9, 0.99, 1e-6, 0.25, 0.0, true);
        double initial = 0.0, final = 0.0;
        for (int step = 0; step < 250; ++step) {
            double dw = 0.0, db = 0.0, loss = 0.0;
            for (int i = -5; i <= 5; ++i) {
                double x = i / 5.0;
                double y = 2.0 * x + 1.0;
                double pred = d->weights[0][0] * x + d->bias[0][0];
                double e = pred - y;
                loss += e * e;
                dw += 2.0 * e * x / 11.0;
                db += 2.0 * e / 11.0;
            }
            loss /= 11.0;
            if (step == 0) initial = loss;
            final = loss;
            d->grad_weights[0][0] = dw;
            d->grad_bias[0][0] = db;
            opt.step(model);
        }
        CHECK(std::isfinite(final));
        CHECK(final < initial * 0.05);
    });
}
}

int main() {
    defaults_and_validation();
    first_step_initializes_without_update();
    previous_second_moment_signature();
    clipping_signature();
    weight_decay_modes();
    malformed_guards();
    state_isolation_and_determinism();
    training_reduces_loss();
    std::cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
