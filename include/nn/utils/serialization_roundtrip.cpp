#include "serialization_roundtrip.h"

SerializationRoundtripTest::Result SerializationRoundtripTest::run(
        Model& model, const Tensor& input,
        double tolerance, const std::string& path) {

    Result r;
    r.max_error = 0.0;

    // Forward pass before save
    Tensor pred_before = model.forward(input);
    r.message = "Before save: OK. ";

    // Save
    model.save(path);
    r.message += "Saved to " + path + ". ";

    // Load into a fresh model (same architecture)
    Model loaded;
    loaded.load(path);
    r.message += "Loaded from " + path + ". ";

    // Forward pass after load
    Tensor pred_after = loaded.forward(input);

    // Compare
    r.max_error = compare_tensors(pred_before, pred_after);
    r.passed = (r.max_error <= tolerance);

    if (r.passed) {
        r.message += "PASSED (max_err=" + std::to_string(r.max_error) + ").";
    } else {
        r.message += "FAILED (max_err=" + std::to_string(r.max_error) + " > " + std::to_string(tolerance) + ").";
    }

    return r;
}

SerializationRoundtripTest::Result SerializationRoundtripTest::run(
        Model& model, const std::vector<Tensor>& inputs,
        double tolerance, const std::string& path) {

    Result r;
    r.max_error = 0.0;
    std::vector<Tensor> preds_before, preds_after;

    // Forward pass before save
    for (const auto& input : inputs)
        preds_before.push_back(model.forward(input));

    model.save(path);

    Model loaded;
    loaded.load(path);

    for (const auto& input : inputs)
        preds_after.push_back(loaded.forward(input));

    double max_err = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        double err = compare_tensors(preds_before[i], preds_after[i]);
        max_err = std::max(max_err, err);
    }

    r.max_error = max_err;
    r.passed = (max_err <= tolerance);
    r.message = r.passed ? "PASSED" : "FAILED";

    return r;
}

SerializationRoundtripTest::Result SerializationRoundtripTest::run_stress(
        Model& model, const Tensor& input,
        size_t cycles, double tolerance) {

    Result r;
    r.max_error = 0.0;

    for (size_t c = 0; c < cycles; ++c) {
        std::string path = "/tmp/model_stress_" + std::to_string(c) + ".nn";
        model.save(path);

        Model loaded;
        loaded.load(path);

        Tensor pred = loaded.forward(input);
        double err = compare_tensors(model.forward(input), pred);
        r.max_error = std::max(r.max_error, err);

        if (err > tolerance) {
            r.passed = false;
            r.message = "FAILED at cycle " + std::to_string(c) + " (err=" + std::to_string(err) + ")";
            return r;
        }
    }

    r.passed = true;
    r.message = "PASSED " + std::to_string(cycles) + " cycles";
    return r;
}

double SerializationRoundtripTest::compare_tensors(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return 1e100; // huge error for size mismatch
    }
    double max_err = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            max_err = std::max(max_err, std::abs(a[i][j] - b[i][j]));
    return max_err;
}