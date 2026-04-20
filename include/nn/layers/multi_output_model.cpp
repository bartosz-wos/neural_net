#include "multi_output_model.h"

std::vector<Tensor> MultiOutputModel::forward(const Tensor& input) {
    std::vector<Tensor> tapped_outputs;
    Tensor current = input;

    for (size_t i = 0; i < layers.size(); ++i) {
        current = layers[i]->forward(current);
        // Check if this layer's output should be tapped
        for (size_t idx : output_indices_) {
            if (idx == i) {
                tapped_outputs.push_back(current);
                break;
            }
        }
    }
    return tapped_outputs; // last one is the final output
}

Tensor MultiOutputModel::forward_single(const Tensor& input) {
    Tensor current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<Tensor> MultiOutputModel::backward(const std::vector<Tensor>& output_grads,
                                              double learning_rate) {
    // output_grads should correspond to gradients w.r.t. tapped outputs (in order)
    // We backprop from the last layer toward the first, but since we don't store
    // intermediate activations for tapped layers, we can only provide gradients
    // for the full model's backward. This method is mainly for interface completeness.
    (void)output_grads;
    (void)learning_rate;
    return {}; // real implementation would need to cache forward activations per tap
}

void MultiOutputModel::update_weights(double learning_rate) {
    for (auto& layer : layers)
        layer->update_weights(learning_rate);
}

void MultiOutputModel::zero_grad() {
    for (auto& layer : layers)
        layer->zero_grad();
}

size_t MultiOutputModel::param_count() const {
    size_t total = 0;
    for (const auto& layer : layers)
        for (Tensor* p : layer->parameters())
            total += p->rows * p->cols;
    return total;
}
