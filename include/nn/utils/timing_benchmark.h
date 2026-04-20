#ifndef TIMING_BENCHMARK_H
#define TIMING_BENCHMARK_H

#include "../core/layer.h"
#include "../core/model.h"
#include <vector>
#include <chrono>
#include <string>

// Layer timing benchmark — measures forward/backward pass time per layer.
// Usage: create benchmark, add layers, call run(), get per-layer report.
class TimingBenchmark {
public:
    TimingBenchmark() : total_time_ms_(0.0) {}

    struct LayerTiming {
        std::string name;
        size_t param_count;
        double forward_ms;
        double backward_ms;
        double total_ms;
    };

    // Run benchmark: repeat forward + backward n times per layer
    // Returns per-layer timing report
    std::vector<LayerTiming> run(const std::vector<Layer*>& layers,
                                  const Tensor& input,
                                  const Tensor& grad_output,
                                  size_t iterations = 10);

    // Run benchmark on a full Model
    std::vector<LayerTiming> run_model(Model& model,
                                       const Tensor& input,
                                       const Tensor& target,
                                       size_t iterations = 10);

    // Print timing report to stdout
    void print_report(const std::vector<LayerTiming>& timings,
                       double total_time_ms);

    double total_time_ms() const { return total_time_ms_; }

private:
    double total_time_ms_;
};

#endif