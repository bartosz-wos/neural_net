// Behavioral tests for InMemoryEarlyStopping.
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/scheduler.h"
#include "nn/utils/early_stopping.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool condition) {
    if (condition) {
        cout << "  [PASS] " << name << '\n';
        ++passed;
    } else {
        cout << "  [FAIL] " << name << '\n';
        ++failed;
    }
}

static bool near(double actual, double expected, double tol = 1e-12) {
    return std::abs(actual - expected) <= tol;
}

template <typename Fn>
static bool throws_invalid_argument(Fn&& fn) {
    try {
        fn();
    } catch (const invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try {
        fn();
    } catch (const runtime_error&) {
        return true;
    } catch (...) {
    }
    return false;
}

class TestParamLayer : public Layer {
public:
    Tensor param;
    Tensor grad;

    TestParamLayer(size_t rows = 2, size_t cols = 3)
        : param(rows, cols), grad(rows, cols) {
        param.fill(0.0);
        grad.fill(0.0);
    }

    Tensor forward(const Tensor& input) override { return input; }
    Tensor backward(const Tensor& grad_output, double) override { return grad_output; }
    void update_weights(double) override {}
    Tensor get_weights() const override { return param; }
    Tensor get_gradients() const override { return grad; }
    vector<Tensor*> parameters() override { return {&param}; }
    vector<Tensor*> gradients() override { return {&grad}; }
    void zero_grad() override { grad.fill(0.0); }
};

static TestParamLayer* make_model(Model& model, size_t rows = 2, size_t cols = 3) {
    auto* layer = new TestParamLayer(rows, cols);
    model.add_layer(layer);
    return layer;
}

static void fill_parameter(TestParamLayer* layer, double value) {
    layer->param.fill(value);
}

static bool parameter_equals(const TestParamLayer* layer, double value) {
    for (double actual : layer->param.data) {
        if (!near(actual, value)) return false;
    }
    return true;
}

static void test_configuration_and_validation() {
    cout << "\n== configuration and validation ==\n";
    InMemoryEarlyStopping defaults;
    check("default patience = 5", defaults.patience() == 5);
    check("default min_delta = 0", near(defaults.min_delta(), 0.0));
    check("default mode minimizes", defaults.mode() == InMemoryEarlyStoppingMode::MINIMIZE);
    check("default restores best weights", defaults.restore_best_weights());
    check("starts without best metric", !defaults.has_best());
    check("starts running", !defaults.stopped());
    check("starts with zero observations", defaults.num_steps() == 0);
    check("starts with zero bad epochs", defaults.num_bad_epochs() == 0);
    check("starts with zero snapshot tensors", defaults.num_snapshot_params() == 0);

    InMemoryEarlyStopping configured(3, 0.25, InMemoryEarlyStoppingMode::MAXIMIZE, false);
    check("non-default patience", configured.patience() == 3);
    check("non-default min_delta", near(configured.min_delta(), 0.25));
    check("non-default mode", configured.mode() == InMemoryEarlyStoppingMode::MAXIMIZE);
    check("non-default restore flag", !configured.restore_best_weights());

    check("patience=0 rejected", throws_invalid_argument([] {
        InMemoryEarlyStopping invalid(0);
    }));
    check("negative min_delta rejected", throws_invalid_argument([] {
        InMemoryEarlyStopping invalid(1, -0.01);
    }));
    check("infinite min_delta rejected", throws_invalid_argument([] {
        InMemoryEarlyStopping invalid(1, numeric_limits<double>::infinity());
    }));
}

static void test_minimize_patience_and_delta() {
    cout << "\n== minimize mode, patience, and min_delta ==\n";
    Model model;
    make_model(model);
    InMemoryEarlyStopping stop(2, 0.1, InMemoryEarlyStoppingMode::MINIMIZE, false);

    check("first observation continues", !stop.step(1.0, model));
    check("first observation becomes best", stop.has_best() && near(stop.best_metric(), 1.0));
    check("best step is one-based observation 1", stop.best_step() == 1);
    check("first observation count = 1", stop.num_steps() == 1);

    check("delta equality is not improvement", !stop.step(0.9, model));
    check("delta equality increments bad epochs", stop.num_bad_epochs() == 1);
    check("metric inside delta stops at exact patience", stop.step(0.91, model));
    check("stopped flag set", stop.stopped());
    check("bad epochs equals patience", stop.num_bad_epochs() == 2);
    check("best metric remains baseline", near(stop.best_metric(), 1.0));
    check("observation count includes stopping step", stop.num_steps() == 3);
}

static void test_improvement_resets_patience() {
    cout << "\n== improvement resets patience ==\n";
    Model model;
    make_model(model);
    InMemoryEarlyStopping stop(2, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, false);

    stop.step(5.0, model);
    stop.step(6.0, model);
    check("one bad epoch accumulated", stop.num_bad_epochs() == 1);
    check("strict improvement continues", !stop.step(4.0, model));
    check("strict improvement resets bad epochs", stop.num_bad_epochs() == 0);
    check("strict improvement updates best", near(stop.best_metric(), 4.0));
    check("best step updated to observation 3", stop.best_step() == 3);
    check("first post-reset miss continues", !stop.step(4.5, model));
    check("second post-reset miss stops", stop.step(4.6, model));
}

static void test_maximize_mode() {
    cout << "\n== maximize mode ==\n";
    Model model;
    make_model(model);
    InMemoryEarlyStopping stop(2, 0.05, InMemoryEarlyStoppingMode::MAXIMIZE, false);

    stop.step(0.50, model);
    check("increase beyond delta improves", !stop.step(0.56, model));
    check("maximize best updated", near(stop.best_metric(), 0.56));
    check("smaller metric is bad", !stop.step(0.55, model));
    check("second smaller metric stops", stop.step(0.54, model));
}

static void test_automatic_best_weight_restore() {
    cout << "\n== automatic best-weight restore ==\n";
    Model model;
    TestParamLayer* layer = make_model(model);
    InMemoryEarlyStopping stop(2, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, true);

    fill_parameter(layer, 1.0);
    stop.step(0.5, model);
    check("best snapshot records one tensor", stop.num_snapshot_params() == 1);

    fill_parameter(layer, 9.0);
    stop.step(0.6, model);
    check("weights untouched before stop", parameter_equals(layer, 9.0));

    fill_parameter(layer, 10.0);
    check("stop boundary reached", stop.step(0.7, model));
    check("best weights restored at stop", parameter_equals(layer, 1.0));

    fill_parameter(layer, 77.0);
    const size_t steps_before = stop.num_steps();
    const size_t bad_before = stop.num_bad_epochs();
    check("step after stop remains true", stop.step(123.0, model));
    check("step after stop does not restore repeatedly", parameter_equals(layer, 77.0));
    check("step after stop does not increment observations", stop.num_steps() == steps_before);
    check("step after stop does not increment bad epochs", stop.num_bad_epochs() == bad_before);
}

static void test_manual_restore_and_deep_copy() {
    cout << "\n== manual restore and deep-copy isolation ==\n";
    Model model;
    TestParamLayer* layer = make_model(model);
    InMemoryEarlyStopping stop(3, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, false);

    fill_parameter(layer, 2.5);
    stop.step(1.0, model);
    fill_parameter(layer, -8.0);
    stop.restore_best(model);
    check("manual restore recovers snapshot", parameter_equals(layer, 2.5));

    fill_parameter(layer, 4.0);
    stop.step(0.5, model);
    fill_parameter(layer, 99.0);
    stop.restore_best(model);
    check("later improvement replaces snapshot", parameter_equals(layer, 4.0));

    Model no_best_model;
    make_model(no_best_model);
    InMemoryEarlyStopping no_best;
    check("restore before first observation throws", throws_runtime_error([&] {
        no_best.restore_best(no_best_model);
    }));
}

static void test_restore_disabled() {
    cout << "\n== disabled automatic restore ==\n";
    Model model;
    TestParamLayer* layer = make_model(model);
    InMemoryEarlyStopping stop(1, 0.0, InMemoryEarlyStoppingMode::MINIMIZE, false);

    fill_parameter(layer, 3.0);
    stop.step(1.0, model);
    fill_parameter(layer, 8.0);
    check("single miss stops", stop.step(2.0, model));
    check("disabled restore keeps current weights", parameter_equals(layer, 8.0));
    stop.restore_best(model);
    check("manual restore remains available", parameter_equals(layer, 3.0));
}

static void test_empty_model_and_topology_guards() {
    cout << "\n== empty model and topology guards ==\n";
    Model empty;
    InMemoryEarlyStopping empty_stop(1);
    check("empty model observation accepted", !empty_stop.step(1.0, empty));
    check("empty snapshot has zero tensors", empty_stop.num_snapshot_params() == 0);
    bool empty_restore_ok = true;
    try {
        empty_stop.restore_best(empty);
    } catch (...) {
        empty_restore_ok = false;
    }
    check("empty model restores safely", empty_restore_ok);

    Model count_model;
    make_model(count_model, 1, 1);
    InMemoryEarlyStopping count_stop(2);
    count_stop.step(1.0, count_model);
    make_model(count_model, 1, 1);
    check("parameter-count drift throws", throws_runtime_error([&] {
        count_stop.restore_best(count_model);
    }));

    Model shape_model;
    TestParamLayer* shape_layer = make_model(shape_model, 1, 2);
    InMemoryEarlyStopping shape_stop(2);
    shape_stop.step(1.0, shape_model);
    shape_layer->param = Tensor(2, 1);
    check("parameter-shape drift throws", throws_runtime_error([&] {
        shape_stop.restore_best(shape_model);
    }));
}

static void test_non_finite_metrics_are_transactional() {
    cout << "\n== non-finite metric validation ==\n";
    Model model;
    make_model(model);
    InMemoryEarlyStopping stop(2);

    check("NaN rejected", throws_invalid_argument([&] {
        stop.step(numeric_limits<double>::quiet_NaN(), model);
    }));
    check("+infinity rejected", throws_invalid_argument([&] {
        stop.step(numeric_limits<double>::infinity(), model);
    }));
    check("-infinity rejected", throws_invalid_argument([&] {
        stop.step(-numeric_limits<double>::infinity(), model);
    }));
    check("invalid metrics do not increment observations", stop.num_steps() == 0);
    check("invalid metrics do not create best state", !stop.has_best());
}

static void test_reset() {
    cout << "\n== reset ==\n";
    Model model;
    TestParamLayer* layer = make_model(model);
    InMemoryEarlyStopping stop(1, 0.2, InMemoryEarlyStoppingMode::MAXIMIZE, true);

    fill_parameter(layer, 5.0);
    stop.step(1.0, model);
    stop.step(0.0, model);
    check("precondition: stopped", stop.stopped());

    stop.reset();
    check("reset clears stopped flag", !stop.stopped());
    check("reset clears best flag", !stop.has_best());
    check("reset clears observations", stop.num_steps() == 0);
    check("reset clears bad epochs", stop.num_bad_epochs() == 0);
    check("reset clears snapshot", stop.num_snapshot_params() == 0);
    check("reset preserves patience", stop.patience() == 1);
    check("reset preserves min_delta", near(stop.min_delta(), 0.2));
    check("reset preserves mode", stop.mode() == InMemoryEarlyStoppingMode::MAXIMIZE);
    check("reset preserves restore flag", stop.restore_best_weights());
    check("new observation after reset accepted", !stop.step(2.0, model));
    check("new best after reset stored", near(stop.best_metric(), 2.0));
}

int main() {
    cout << "=== InMemoryEarlyStopping Tests ===\n";
    test_configuration_and_validation();
    test_minimize_patience_and_delta();
    test_improvement_resets_patience();
    test_maximize_mode();
    test_automatic_best_weight_restore();
    test_manual_restore_and_deep_copy();
    test_restore_disabled();
    test_empty_model_and_topology_guards();
    test_non_finite_metrics_are_transactional();
    test_reset();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
