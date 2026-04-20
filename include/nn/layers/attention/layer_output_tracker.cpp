#include "layer_output_tracker.h"
#include "../../core/model.h"
#include <cmath>
#include <iostream>
#include <iomanip>

LayerOutputTracker::LayerOutputTracker(size_t report_every_n_steps,
                                         double zero_threshold,
                                         bool warn_on_vanishing,
                                         bool warn_on_exploding)
    : report_every_n_steps_(report_every_n_steps),
      zero_threshold_(zero_threshold),
      warn_on_vanishing_(warn_on_vanishing),
      warn_on_exploding_(warn_on_exploding),
      has_vanishing_(false), has_exploding_(false) {}

LayerOutputTracker::LayerStats LayerOutputTracker::compute_stats(
        const std::string& name, const Tensor& output, size_t step) {
    LayerStats s;
    s.name = name;
    s.step = step;

    s.min = output[0][0];
    s.max = output[0][0];
    double sum = 0.0;
    double sq_sum = 0.0;
    size_t zero_count = 0;
    size_t total = output.rows * output.cols;

    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            double v = output[i][j];
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
            sum += v;
            sq_sum += v * v;
            if (std::abs(v) < zero_threshold_) zero_count++;
        }
    }

    size_t N = total;
    s.mean = sum / N;
    double variance = (sq_sum / N) - s.mean * s.mean;
    s.std = std::sqrt(std::max(0.0, variance));
    s.pct_zero = 100.0 * zero_count / N;

    // Detect vanishing: std is very small relative to |mean|
    if (s.mean != 0 && s.std / std::abs(s.mean) < 1e-3) {
        has_vanishing_ = true;
        if (warn_on_vanishing_)
            std::cout << "[TRACKER] VANISHING at step " << step << " layer=" << name
                      << " mean=" << s.mean << " std=" << s.std << "\n";
    }

    // Detect exploding: values are very large
    if (s.max > 1e6 || std::abs(s.min) > 1e6) {
        has_exploding_ = true;
        if (warn_on_exploding_)
            std::cout << "[TRACKER] EXPLODING at step " << step << " layer=" << name
                      << " min=" << s.min << " max=" << s.max << "\n";
    }

    return s;
}

void LayerOutputTracker::track(const std::string& layer_name,
                                const Tensor& output, size_t step) {
    LayerStats s = compute_stats(layer_name, output, step);
    stats_.push_back(s);

    if (tracked_layers_.empty() || tracked_layers_.back() != layer_name)
        tracked_layers_.push_back(layer_name);
}

void LayerOutputTracker::track_model(Model& model, const Tensor& input, size_t step) {
    // Forward pass through model, tracking each layer
    Tensor x = input;
    for (size_t li = 0; li < model.layers.size(); ++li) {
        x = model.layers[li]->forward(x);
        std::string name = "layer_" + std::to_string(li);
        track(name, x, step);
    }
}

std::vector<LayerOutputTracker::LayerStats>
LayerOutputTracker::get_history(const std::string& layer_name) const {
    std::vector<LayerStats> result;
    for (const auto& s : stats_)
        if (s.name == layer_name)
            result.push_back(s);
    return result;
}

void LayerOutputTracker::print_report() {
    if (stats_.empty()) {
        std::cout << "No stats recorded.\n";
        return;
    }

    std::cout << "\n=== Layer Output Tracker Report ===\n";
    std::cout << std::left << std::setw(20) << "Layer"
              << std::right << std::setw(8) << "Step"
              << std::setw(12) << "Min"
              << std::setw(12) << "Max"
              << std::setw(12) << "Mean"
              << std::setw(10) << "Std"
              << std::setw(10) << "%Zero" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& s : stats_) {
        std::cout << std::left << std::setw(20) << s.name
                  << std::right << std::setw(8) << s.step
                  << std::setw(12) << std::scientific << std::setprecision(3) << s.min
                  << std::setw(12) << s.max
                  << std::setw(12) << s.mean
                  << std::setw(10) << s.std
                  << std::setw(10) << std::fixed << std::setprecision(1) << s.pct_zero << "%\n";
    }
    std::cout << "\n";
}

void LayerOutputTracker::save_to_csv(const std::string& filename) {
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "Could not open " << filename << " for writing\n";
        return;
    }
    f << "step,layer,name,min,max,mean,std,pct_zero\n";
    for (const auto& s : stats_) {
        f << s.step << ","
          << s.name << ","
          << std::scientific << s.min << ","
          << s.max << ","
          << s.mean << ","
          << s.std << ","
          << std::fixed << s.pct_zero << "\n";
    }
    f.close();
    std::cout << "Stats saved to " << filename << "\n";
}