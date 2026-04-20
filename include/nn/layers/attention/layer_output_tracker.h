#ifndef LAYER_OUTPUT_TRACKER_H
#define LAYER_OUTPUT_TRACKER_H

#include "../../core/layer.h"
#include "../../core/model.h"
#include <vector>
#include <string>
#include <fstream>

// Layer output tracker — monitors activation statistics per layer during training.
// Detects vanishing/exploding gradients, dead neurons, activation ranges.
// Can be used as a training callback to log stats per epoch/step.
class LayerOutputTracker {
public:
    struct LayerStats {
        std::string name;
        double min;
        double max;
        double mean;
        double std;
        double pct_zero;  // % of zero/negligible activations
        size_t step;
    };

    LayerOutputTracker(size_t report_every_n_steps = 100,
                       double zero_threshold = 1e-6,
                       bool warn_on_vanishing = true,
                       bool warn_on_exploding = true);

    // Track output of a single layer
    void track(const std::string& layer_name, const Tensor& output, size_t step);

    // Track output across a full Model
    void track_model(Model& model, const Tensor& input, size_t step);

    // Get history of stats for a layer
    std::vector<LayerStats> get_history(const std::string& layer_name) const;

    // Print report for all tracked layers
    void print_report();

    // Save stats to CSV file
    void save_to_csv(const std::string& filename);

    // Check if any layer has vanishing/exploding issues
    bool has_vanishing() const { return has_vanishing_; }
    bool has_exploding() const { return has_exploding_; }

    void reset() { stats_.clear(); has_vanishing_ = false; has_exploding_ = false; }

private:
    size_t report_every_n_steps_;
    double zero_threshold_;
    bool warn_on_vanishing_, warn_on_exploding_;
    bool has_vanishing_, has_exploding_;
    std::vector<LayerStats> stats_;
    std::vector<std::string> tracked_layers_;

    LayerStats compute_stats(const std::string& name, const Tensor& output, size_t step);
};

#endif