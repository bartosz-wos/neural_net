#ifndef MULTI_OUTPUT_MODEL_H
#define MULTI_OUTPUT_MODEL_H

#include "../core/model.h"

// MultiOutputModel: a model that can return outputs from intermediate layers.
// Specify which layer indices to "tap" (collect outputs from) during forward pass.
class MultiOutputModel {
public:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<size_t> output_indices_;  // layer indices whose outputs to return

    MultiOutputModel() = default;

    // Add a layer and optionally register it as an output tap
    void add(Layer* layer, bool tap_output = false) {
        layers.emplace_back(layer);
        if (tap_output)
            output_indices_.push_back(layers.size() - 1);
    }

    // Forward pass: returns outputs from tapped layers + final output
    // The tapped outputs are in order of layer index
    std::vector<Tensor> forward(const Tensor& input);

    // Standard single-output forward (from last layer)
    Tensor forward_single(const Tensor& input);

    // Gradient of loss w.r.t. tapped layer outputs (for distillation losses)
    std::vector<Tensor> backward(const std::vector<Tensor>& output_grads, double learning_rate);

    void update_weights(double learning_rate);
    void zero_grad();
    size_t param_count() const;
    size_t total_tapped_outputs() const { return output_indices_.size(); }
};

#endif
