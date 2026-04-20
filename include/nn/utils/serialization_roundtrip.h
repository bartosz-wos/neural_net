#ifndef SERIALIZATION_ROUNDTRIP_H
#define SERIALIZATION_ROUNDTRIP_H

#include "../core/model.h"
#include <string>
#include <vector>

// SerializationRoundtripTest: save → load → verify outputs match exactly.
// Tests that model serialization/deserialization is lossless.
class SerializationRoundtripTest {
public:
    // Run a roundtrip test: save model, load it, compare forward outputs.
    // tolerance: max acceptable absolute difference for outputs to be "equal".
    // Returns: {passed, max_error, message}
    struct Result {
        bool passed;
        double max_error;
        std::string message;
    };

    Result run(Model& model, const Tensor& input,
               double tolerance = 1e-6, const std::string& path = "/tmp/model_roundtrip.nn");

    // Overload: run multiple inputs
    Result run(Model& model, const std::vector<Tensor>& inputs,
               double tolerance = 1e-6, const std::string& path = "/tmp/model_roundtrip.nn");

    // Stress test: repeated save/load cycles
    Result run_stress(Model& model, const Tensor& input,
                      size_t cycles = 5, double tolerance = 1e-6);

private:
    double compare_tensors(const Tensor& a, const Tensor& b);
};

#endif