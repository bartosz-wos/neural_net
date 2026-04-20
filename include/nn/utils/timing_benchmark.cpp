#include "timing_benchmark.h"
#include "../core/model.h"
#include <iostream>
#include <iomanip>

std::vector<TimingBenchmark::LayerTiming> TimingBenchmark::run(
        const std::vector<Layer*>& layers,
        const Tensor& input,
        const Tensor& /*grad_output*/,
        size_t iterations) {

    std::vector<LayerTiming> timings;
    Tensor current = input;

    for (Layer* layer : layers) {
        LayerTiming lt;
        lt.name = layer->name();

        // Count params
        lt.param_count = 0;
        for (Tensor* p : layer->parameters())
            lt.param_count += p->rows * p->cols;

        // Warm-up
        current = layer->forward(current);
        layer->backward(Tensor(1,1), 0.0);
        layer->zero_grad();

        // Benchmark forward
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            current = layer->forward(input);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        lt.forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // Benchmark backward
        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            layer->backward(Tensor(1,1), 0.0);
            layer->zero_grad();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        lt.backward_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iterations;

        lt.total_ms = lt.forward_ms + lt.backward_ms;
        timings.push_back(lt);

        // Prepare input for next layer
        current = layer->forward(input); // re-run to get correct shape
    }

    return timings;
}

std::vector<TimingBenchmark::LayerTiming> TimingBenchmark::run_model(
        Model& model,
        const Tensor& input,
        const Tensor& target,
        size_t iterations) {

    std::vector<LayerTiming> timings;

    // MSE loss gradient
    Tensor pred = model.forward(input);
    Tensor grad_loss = pred - target;
    grad_loss = grad_loss * (2.0 / static_cast<double>(input.rows));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        model.forward(input);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_total = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        model.backward(grad_loss, 0.0);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_total = std::chrono::duration<double, std::milli>(t3 - t2).count() / iterations;

    total_time_ms_ = fwd_total + bwd_total;

    // Total timing
    LayerTiming total_lt;
    total_lt.name = "TOTAL";
    total_lt.forward_ms = fwd_total;
    total_lt.backward_ms = bwd_total;
    total_lt.total_ms = total_time_ms_;
    total_lt.param_count = model.param_count();
    timings.push_back(total_lt);

    // Per-layer timing
    for (size_t li = 0; li < model.layers.size(); ++li) {
        Layer* layer = model.layers[li].get();
        LayerTiming lt;
        lt.name = "layer_" + std::to_string(li) + "_" + layer->name();

        lt.param_count = 0;
        for (Tensor* p : layer->parameters())
            lt.param_count += p->rows * p->cols;

        auto t4 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            layer->forward(input);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        lt.forward_ms = std::chrono::duration<double, std::milli>(t5 - t4).count() / iterations;

        auto t6 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            layer->backward(Tensor(1,1), 0.0);
            layer->zero_grad();
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        lt.backward_ms = std::chrono::duration<double, std::milli>(t7 - t6).count() / iterations;

        lt.total_ms = lt.forward_ms + lt.backward_ms;
        timings.push_back(lt);
    }

    return timings;
}

void TimingBenchmark::print_report(const std::vector<LayerTiming>& timings,
                                    double total_ms) {
    std::cout << "\n=== Timing Benchmark ===\n";
    std::cout << std::left << std::setw(30) << "Layer"
              << std::right << std::setw(10) << "Params"
              << std::setw(14) << "Fwd (ms)"
              << std::setw(14) << "Bwd (ms)"
              << std::setw(14) << "Total (ms)" << "\n";
    std::cout << std::string(80, '-') << "\n";
    for (const auto& lt : timings) {
        std::cout << std::left << std::setw(30) << lt.name
                  << std::right << std::setw(10) << lt.param_count
                  << std::setw(14) << std::fixed << std::setprecision(3) << lt.forward_ms
                  << std::setw(14) << lt.backward_ms
                  << std::setw(14) << lt.total_ms << "\n";
    }
    std::cout << std::string(80, '-') << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n\n";
}