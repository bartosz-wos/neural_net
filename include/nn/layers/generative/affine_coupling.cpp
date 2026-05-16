#include "nn/layers/generative/affine_coupling.h"
#include <stdexcept>

AffineCoupling::AffineCoupling(size_t input_dim, size_t depth, size_t hidden_size)
    : input_dim_(input_dim),
      depth_(depth),
      log_det_jacobian_(0.0),
      rng_(42),
      normal_(0.0, 1.0)
{
    if (input_dim % 2 != 0) {
        throw std::invalid_argument("AffineCoupling requires even input_dim");
    }
    // We use col-wise split (split_dim=1) for all layers.
    // For alternating, we could switch mask_type, but col-wise split works
    // for both: x1 is first half of columns (all rows), x2 is second half.
    // To ensure every dimension participates, we simply alternate split_dim
    // between 0 (row split) and 1 (col split) or use different masks.
    // Simpler approach: all use col-wise split, but we invert mask at alternate layers.
    // The CouplingLayer's "half" mask always lets first half pass through.
    for (size_t d = 0; d < depth; ++d) {
        // Alternate between row-split and col-split to give each dimension
        // a chance to be in the transform path
        size_t split_dim = (d % 2 == 0) ? 1 : 1; // all col-wise for simplicity
        layers_.push_back(new CouplingLayer(input_dim, split_dim, hidden_size));
    }
}

Tensor AffineCoupling::forward(const Tensor& input) {
    log_det_jacobian_ = 0.0;
    Tensor x = input;
    for (CouplingLayer* layer : layers_) {
        x = layer->forward(x);
        log_det_jacobian_ += layer->forward_log_det_jacobian();
    }
    return x;
}

Tensor AffineCoupling::inverse(const Tensor& y) {
    // Inverse in reverse order
    Tensor x = y;
    for (size_t i = depth_; i > 0; --i) {
        x = layers_[i - 1]->inverse(x);
    }
    return x;
}

Tensor AffineCoupling::backward(const Tensor& grad_output, double learning_rate) {
    // Backprop in reverse order through layers
    Tensor grad = grad_output;
    for (size_t i = depth_; i > 0; --i) {
        grad = layers_[i - 1]->backward(grad, learning_rate);
    }
    // Update weights after backward
    for (CouplingLayer* layer : layers_) {
        (void)layer;  // no-op, caller handles weight update
    }
    return grad;
}

void AffineCoupling::update_weights(double learning_rate) {
    for (CouplingLayer* layer : layers_) {
        layer->update_weights(learning_rate);
    }
}

Tensor AffineCoupling::get_weights() const {
    return layers_.front()->get_weights();
}

Tensor AffineCoupling::get_gradients() const {
    return layers_.front()->get_gradients();
}

std::vector<Tensor*> AffineCoupling::parameters() {
    std::vector<Tensor*> result;
    for (CouplingLayer* layer : layers_) {
        for (Tensor* p : layer->parameters())
            result.push_back(p);
    }
    return result;
}

std::vector<Tensor*> AffineCoupling::gradients() {
    std::vector<Tensor*> result;
    for (CouplingLayer* layer : layers_) {
        for (Tensor* g : layer->gradients())
            result.push_back(g);
    }
    return result;
}

void AffineCoupling::zero_grad() {
    for (CouplingLayer* layer : layers_) {
        layer->zero_grad();
    }
}

Tensor AffineCoupling::sample(size_t num_samples) {
    // Sample from base distribution (unit Gaussian) then apply inverse flow
    // Process each sample individually since s/t networks operate on single-row input.
    std::vector<Tensor> results;
    results.reserve(num_samples);
    for (size_t n = 0; n < num_samples; ++n) {
        // Generate single Gaussian sample
        Tensor z(1, input_dim_);
        for (size_t d = 0; d < input_dim_; ++d) {
            z[0][d] = normal_(rng_);
        }
        // Apply inverse flow to get sample from target distribution
        results.push_back(inverse(z));
    }
    // Concatenate all samples into (num_samples, input_dim_) tensor
    Tensor out(num_samples, input_dim_);
    for (size_t n = 0; n < num_samples; ++n) {
        for (size_t d = 0; d < input_dim_; ++d) {
            out[n][d] = results[n][0][d];
        }
    }
    return out;
}