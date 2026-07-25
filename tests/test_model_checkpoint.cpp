#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/utils/model_checkpoint.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool condition) {
    cout << "  [" << (condition ? "PASS" : "FAIL") << "] " << name << '\n';
    condition ? ++passed : ++failed;
}

template <typename Exception, typename Fn>
static bool throws_as(Fn&& fn) {
    try { fn(); } catch (const Exception&) { return true; } catch (...) {}
    return false;
}

static bool near(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

class MultiParamLayer : public Layer {
public:
    Tensor first;
    Tensor second;
    Tensor grad_first;
    Tensor grad_second;

    MultiParamLayer(size_t first_rows = 2, size_t first_cols = 3,
                    size_t second_rows = 1, size_t second_cols = 2)
        : first(first_rows, first_cols), second(second_rows, second_cols),
          grad_first(first_rows, first_cols), grad_second(second_rows, second_cols) {
        first.fill(0.0);
        second.fill(0.0);
        grad_first.fill(0.0);
        grad_second.fill(0.0);
    }

    Tensor forward(const Tensor& input) override { return input; }
    Tensor backward(const Tensor& grad_output, double) override { return grad_output; }
    void update_weights(double) override {}
    Tensor get_weights() const override { return first; }
    Tensor get_gradients() const override { return grad_first; }
    vector<Tensor*> parameters() override { return {&first, &second}; }
    vector<Tensor*> gradients() override { return {&grad_first, &grad_second}; }
    void zero_grad() override { grad_first.fill(0.0); grad_second.fill(0.0); }
};

static MultiParamLayer* add_layer(Model& model, size_t rows = 2, size_t cols = 3) {
    auto* layer = new MultiParamLayer(rows, cols);
    model.add_layer(layer);
    return layer;
}

static void fill_tensor(Tensor& tensor, double start) {
    for (size_t i = 0; i < tensor.data.size(); ++i) tensor.data[i] = start + static_cast<double>(i);
}

static bool tensor_equals(const Tensor& tensor, double start) {
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        if (!near(tensor.data[i], start + static_cast<double>(i))) return false;
    }
    return true;
}

static bool file_exists(const string& path) {
    ifstream in(path, ios::binary);
    return static_cast<bool>(in);
}

static string path_for(const string& name) {
    return "/tmp/neural_net_checkpoint_" + name + ".bin";
}

static void remove_files(const string& path) {
    remove(path.c_str());
    remove((path + ".tmp").c_str());
}

static void test_configuration_and_validation() {
    cout << "\n== configuration and validation ==\n";
    const string path = path_for("config");
    ModelCheckpoint defaults(path);
    check("path preserved", defaults.path() == path);
    check("defaults to best-only", defaults.save_best_only());
    check("default mode minimizes", defaults.mode() == ModelCheckpointMode::MINIMIZE);
    check("default min_delta zero", near(defaults.min_delta(), 0.0));
    check("starts without best", !defaults.has_best());
    check("starts with zero observations", defaults.num_steps() == 0);
    check("starts with zero saves", defaults.num_saved() == 0);

    ModelCheckpoint configured(path, false, ModelCheckpointMode::MAXIMIZE, 0.25);
    check("latest mode configured", !configured.save_best_only());
    check("maximize mode configured", configured.mode() == ModelCheckpointMode::MAXIMIZE);
    check("min_delta configured", near(configured.min_delta(), 0.25));

    check("empty path rejected", throws_as<invalid_argument>([] { ModelCheckpoint bad(""); }));
    check("negative min_delta rejected", throws_as<invalid_argument>([&] { ModelCheckpoint bad(path, true, ModelCheckpointMode::MINIMIZE, -0.1); }));
    check("infinite min_delta rejected", throws_as<invalid_argument>([&] { ModelCheckpoint bad(path, true, ModelCheckpointMode::MINIMIZE, numeric_limits<double>::infinity()); }));
    remove_files(path);
}

static void test_manual_round_trip_multiple_parameters() {
    cout << "\n== manual generic parameter round trip ==\n";
    const string path = path_for("roundtrip");
    remove_files(path);
    Model model;
    MultiParamLayer* a = add_layer(model, 2, 3);
    MultiParamLayer* b = add_layer(model, 1, 4);
    fill_tensor(a->first, 1.0);
    fill_tensor(a->second, 20.0);
    fill_tensor(b->first, -5.0);
    fill_tensor(b->second, 50.0);

    ModelCheckpoint checkpoint(path);
    checkpoint.save(model);
    check("manual save creates checkpoint", file_exists(path));
    check("manual save increments save count", checkpoint.num_saved() == 1);

    a->first.fill(99.0); a->second.fill(99.0);
    b->first.fill(99.0); b->second.fill(99.0);
    checkpoint.load(model);
    check("first layer first tensor restored", tensor_equals(a->first, 1.0));
    check("first layer second tensor restored", tensor_equals(a->second, 20.0));
    check("second layer first tensor restored", tensor_equals(b->first, -5.0));
    check("second layer second tensor restored", tensor_equals(b->second, 50.0));
    remove_files(path);
}

static void test_best_only_minimize_and_delta() {
    cout << "\n== best-only minimize and strict delta ==\n";
    const string path = path_for("minimize");
    remove_files(path);
    Model model;
    MultiParamLayer* layer = add_layer(model);
    ModelCheckpoint checkpoint(path, true, ModelCheckpointMode::MINIMIZE, 0.1);

    fill_tensor(layer->first, 1.0);
    check("first metric saves", checkpoint.step(0, 1.0, model));
    check("first best tracked", checkpoint.has_best() && near(checkpoint.best_metric(), 1.0));
    check("first best epoch tracked", checkpoint.best_epoch() == 0);

    fill_tensor(layer->first, 10.0);
    check("delta equality does not save", !checkpoint.step(1, 0.9, model));
    fill_tensor(layer->first, 20.0);
    check("worse metric does not save", !checkpoint.step(2, 1.2, model));
    fill_tensor(layer->first, 3.0);
    check("strict improvement saves", checkpoint.step(3, 0.89, model));
    check("best metric updated", near(checkpoint.best_metric(), 0.89));
    check("best epoch updated", checkpoint.best_epoch() == 3);
    check("only improvements saved", checkpoint.num_saved() == 2);
    check("all observations counted", checkpoint.num_steps() == 4);

    layer->first.fill(77.0);
    checkpoint.load(model);
    check("checkpoint contains strict best weights", tensor_equals(layer->first, 3.0));
    remove_files(path);
}

static void test_maximize_and_latest_modes() {
    cout << "\n== maximize and latest modes ==\n";
    const string best_path = path_for("maximize");
    remove_files(best_path);
    Model best_model;
    MultiParamLayer* best_layer = add_layer(best_model);
    ModelCheckpoint best(best_path, true, ModelCheckpointMode::MAXIMIZE, 0.05);
    fill_tensor(best_layer->first, 2.0); best.step(2, 0.5, best_model);
    fill_tensor(best_layer->first, 9.0);
    check("maximize delta equality does not save", !best.step(3, 0.55, best_model));
    fill_tensor(best_layer->first, 4.0);
    check("maximize strict improvement saves", best.step(4, 0.56, best_model));
    best_layer->first.fill(0.0); best.load(best_model);
    check("maximize checkpoint restores best", tensor_equals(best_layer->first, 4.0));

    const string latest_path = path_for("latest");
    remove_files(latest_path);
    Model latest_model;
    MultiParamLayer* latest_layer = add_layer(latest_model);
    ModelCheckpoint latest(latest_path, false);
    fill_tensor(latest_layer->first, 1.0); check("latest saves first", latest.step(0, 5.0, latest_model));
    fill_tensor(latest_layer->first, 7.0); check("latest saves worse metric", latest.step(1, 9.0, latest_model));
    latest_layer->first.fill(0.0); latest.load(latest_model);
    check("latest mode restores most recent weights", tensor_equals(latest_layer->first, 7.0));
    check("latest mode counts both saves", latest.num_saved() == 2);
    check("latest mode still tracks best metric", near(latest.best_metric(), 5.0) && latest.best_epoch() == 0);
    remove_files(best_path); remove_files(latest_path);
}

static void test_callback_reset_and_invalid_observations() {
    cout << "\n== callback, reset, and invalid observations ==\n";
    const string path = path_for("callback");
    remove_files(path);
    Model model;
    MultiParamLayer* layer = add_layer(model);
    ModelCheckpoint checkpoint(path);
    auto callback = checkpoint.callback(model);
    fill_tensor(layer->first, 6.0);
    callback(4, 9.0, 0.4);
    check("callback monitors validation loss", checkpoint.has_best() && near(checkpoint.best_metric(), 0.4));
    check("callback preserves epoch", checkpoint.best_epoch() == 4);

    const size_t saves_before = checkpoint.num_saved();
    const size_t steps_before = checkpoint.num_steps();
    check("negative callback epoch rejected", throws_as<invalid_argument>([&] { callback(-1, 1.0, 0.3); }));
    check("NaN metric rejected", throws_as<invalid_argument>([&] { checkpoint.step(5, numeric_limits<double>::quiet_NaN(), model); }));
    check("infinite metric rejected", throws_as<invalid_argument>([&] { checkpoint.step(5, numeric_limits<double>::infinity(), model); }));
    check("invalid observations do not increment steps", checkpoint.num_steps() == steps_before);
    check("invalid observations do not save", checkpoint.num_saved() == saves_before);

    checkpoint.reset();
    check("reset clears best", !checkpoint.has_best());
    check("reset clears steps", checkpoint.num_steps() == 0);
    check("reset clears save count", checkpoint.num_saved() == 0);
    check("reset preserves configuration", checkpoint.path() == path && checkpoint.save_best_only());
    remove_files(path);
}

static void test_load_validation_is_transactional() {
    cout << "\n== load validation and transactional restore ==\n";
    const string path = path_for("validation");
    remove_files(path);
    Model source;
    MultiParamLayer* source_layer = add_layer(source, 2, 3);
    fill_tensor(source_layer->first, 1.0);
    fill_tensor(source_layer->second, 10.0);
    ModelCheckpoint checkpoint(path);
    checkpoint.save(source);

    Model wrong_count;
    auto* count_layer = new MultiParamLayer(2, 3, 0, 0);
    wrong_count.add_layer(count_layer);
    count_layer->first.fill(44.0);
    check("parameter-count mismatch throws", throws_as<runtime_error>([&] { checkpoint.load(wrong_count); }));
    check("count mismatch leaves model unchanged", near(count_layer->first.data[0], 44.0));

    Model wrong_shape;
    MultiParamLayer* shape_layer = add_layer(wrong_shape, 3, 2);
    shape_layer->first.fill(55.0);
    shape_layer->second.fill(66.0);
    check("parameter-shape mismatch throws", throws_as<runtime_error>([&] { checkpoint.load(wrong_shape); }));
    check("shape mismatch leaves first tensor unchanged", near(shape_layer->first.data[0], 55.0));
    check("shape mismatch leaves later tensor unchanged", near(shape_layer->second.data[0], 66.0));

    const string missing = path_for("missing");
    remove_files(missing);
    ModelCheckpoint missing_checkpoint(missing);
    check("missing file throws", throws_as<runtime_error>([&] { missing_checkpoint.load(source); }));

    const string bad = path_for("bad_magic");
    { ofstream out(bad, ios::binary); out << "garbage"; }
    ModelCheckpoint bad_checkpoint(bad);
    check("bad magic throws", throws_as<runtime_error>([&] { bad_checkpoint.load(source); }));

    const string truncated = path_for("truncated");
    { ifstream in(path, ios::binary); ofstream out(truncated, ios::binary); vector<char> bytes(12); in.read(bytes.data(), bytes.size()); out.write(bytes.data(), in.gcount()); }
    ModelCheckpoint truncated_checkpoint(truncated);
    check("truncated checkpoint throws", throws_as<runtime_error>([&] { truncated_checkpoint.load(source); }));

    const string huge_count = path_for("huge_count");
    {
        ofstream out(huge_count, ios::binary);
        const char magic[] = {'N', 'N', 'P', 'A', 'R', 'M'};
        const uint8_t version = 1;
        const uint64_t count = numeric_limits<uint64_t>::max();
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }
    ModelCheckpoint huge_count_checkpoint(huge_count);
    check("absurd parameter count is rejected as checkpoint error", throws_as<runtime_error>([&] {
        huge_count_checkpoint.load(source);
    }));

    const string huge_shape = path_for("huge_shape");
    {
        ofstream out(huge_shape, ios::binary);
        const char magic[] = {'N', 'N', 'P', 'A', 'R', 'M'};
        const uint8_t version = 1;
        const uint64_t count = 2;
        const uint64_t huge = numeric_limits<uint64_t>::max();
        const uint64_t one = 1;
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(&huge), sizeof(huge));
        out.write(reinterpret_cast<const char*>(&one), sizeof(one));
    }
    ModelCheckpoint huge_shape_checkpoint(huge_shape);
    check("absurd tensor shape is rejected before allocation", throws_as<runtime_error>([&] {
        huge_shape_checkpoint.load(source);
    }));

    remove_files(path); remove_files(bad); remove_files(truncated);
    remove_files(huge_count); remove_files(huge_shape);
}

static void test_empty_model_round_trip() {
    cout << "\n== empty model ==\n";
    const string path = path_for("empty");
    remove_files(path);
    Model empty;
    ModelCheckpoint checkpoint(path);
    bool saved = true;
    try { checkpoint.save(empty); checkpoint.load(empty); } catch (...) { saved = false; }
    check("empty model saves and loads", saved);
    check("empty model checkpoint exists", file_exists(path));
    remove_files(path);
}

int main() {
    cout << "=== ModelCheckpoint Tests ===\n";
    test_configuration_and_validation();
    test_manual_round_trip_multiple_parameters();
    test_best_only_minimize_and_delta();
    test_maximize_and_latest_modes();
    test_callback_reset_and_invalid_observations();
    test_load_validation_is_transactional();
    test_empty_model_round_trip();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
