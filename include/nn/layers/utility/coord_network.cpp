#include "coord_network.h"
#include "../../activations/activations.h"
#include <cmath>
#include <algorithm>

CoordinateNetwork::CoordinateNetwork(size_t coord_dim, size_t num_frequencies,
                                      const std::vector<size_t>& hidden_sizes,
                                      size_t output_dim, bool learn_frequencies)
    : coord_dim_(coord_dim), output_dim_(output_dim), hidden_sizes_(hidden_sizes)
{
    if (learn_frequencies) {
        fourier_ = std::make_unique<LearnedFourierFeatures>(coord_dim, num_frequencies);
    } else {
        fourier_ = std::make_unique<GaussianFourierFeatures>(coord_dim, num_frequencies, 1.0);
    }

    // Build MLP: Fourier output -> hidden[0] -> ... -> hidden[last] -> output
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(2 * num_frequencies);  // input to MLP
    for (size_t h : hidden_sizes) layer_sizes.push_back(h);
    layer_sizes.push_back(output_dim);

    for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        auto dense = std::make_unique<Dense>(layer_sizes[i], layer_sizes[i + 1]);
        dense->init_weights("xavier");
        mlp_layers_.push_back(std::move(dense));
    }
}

Tensor CoordinateNetwork::forward(const Tensor& coords) {
    last_fourier_out_ = fourier_->forward(coords);          // (N, 2F)
    Tensor x = last_fourier_out_;
    last_layer_inputs_.clear();

    // last_layer_inputs_[0] = Fourier output (pre-activation for first Dense)
    last_layer_inputs_.push_back(x.clone());

    for (size_t i = 0; i < mlp_layers_.size() - 1; ++i) {
        // Dense::forward stores its input in last_input
        x = mlp_layers_[i]->forward(x);
        // At this point, x is the output of Dense[i] (before ReLU), stored in mlp_layers_[i]->last_input
        // Store the pre-ReLU value
        last_layer_inputs_.push_back(x.clone());
        // Apply ReLU
        x = x.apply(ReLU());
    }
    // Final layer (no activation)
    x = mlp_layers_.back()->forward(x);
    last_layer_inputs_.push_back(x.clone());
    return x;
}

Tensor CoordinateNetwork::backward(const Tensor& grad_output, double learning_rate) {
    // Backprop through MLP in reverse order
    // For hidden layer at index i with ReLU:
    //   dL/d(pre_act) = (dL/d(post_act) ⊙ ReLU_mask) @ W^T
    //   where ReLU_mask = 1 if pre_act > 0, else 0
    // For output layer: just Dense backward (no activation)

    Tensor grad = grad_output;
    size_t num_layers = mlp_layers_.size();

    for (size_t li = 0; li < num_layers; ++li) {
        size_t layer_idx = num_layers - 1 - li;
        auto& layer = mlp_layers_[layer_idx];

        if (layer_idx == num_layers - 1) {
            // Final layer: no activation, Dense backward only
            grad = layer->backward(grad, 0.0);
        } else {
            // Hidden layer with ReLU
            // Get pre-activation input to this Dense layer
            // last_layer_inputs_[layer_idx + 1] = output of Dense[layer_idx] = pre-activation
            const Tensor& pre_act = layer->last_input;

            // Apply ReLU mask: dL/d(pre_act) = dL/d(post_act) ⊙ (pre_act > 0)
            Tensor grad_masked(grad.rows, grad.cols);
            for (size_t i = 0; i < grad.rows; ++i) {
                for (size_t j = 0; j < grad.cols; ++j) {
                    grad_masked[i][j] = (pre_act[i][j] > 0.0) ? grad[i][j] : 0.0;
                }
            }
            // Backprop through Dense
            grad = layer->backward(grad_masked, 0.0);
        }
    }

    // Backprop through FourierFeatures
    grad = fourier_->backward(grad, 0.0);

    // No gradient clipping for proper backward pass

    (void)learning_rate;
    return grad;
}

void CoordinateNetwork::update_weights(double learning_rate) {
    // Clip weight gradients per-layer to prevent instability
    double max_norm = 10.0;
    for (auto& layer : mlp_layers_) {
        for (Tensor* g : layer->gradients()) {
            double gnorm = 0.0;
            for (size_t i = 0; i < g->rows * g->cols; ++i)
                gnorm += (*g)[0][i] * (*g)[0][i];
            gnorm = std::sqrt(gnorm);
            if (gnorm > max_norm) {
                double scale = max_norm / gnorm;
                for (size_t i = 0; i < g->rows * g->cols; ++i)
                    (*g)[0][i] *= scale;
            }
        }
        layer->update_weights(learning_rate);
    }
}

void CoordinateNetwork::zero_grad() {
    fourier_->zero_grad();
    for (auto& layer : mlp_layers_) {
        layer->zero_grad();
    }
}

std::vector<Tensor*> CoordinateNetwork::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : mlp_layers_) {
        for (Tensor* p : layer->parameters()) params.push_back(p);
    }
    return params;
}

std::vector<Tensor*> CoordinateNetwork::gradients() {
    std::vector<Tensor*> grads;
    for (auto& layer : mlp_layers_) {
        for (Tensor* g : layer->gradients()) grads.push_back(g);
    }
    return grads;
}

Tensor CoordinateNetwork::get_weights() const {
    return Tensor(0, 0);
}

Tensor CoordinateNetwork::get_gradients() const {
    double gnorm = 0.0;
    for (auto& layer : mlp_layers_) {
        for (Tensor* g : layer->gradients()) {
            for (size_t i = 0; i < g->rows * g->cols; ++i)
                gnorm += (*g)[0][i] * (*g)[0][i];
        }
    }
    Tensor _t(1, 1); _t[0][0] = std::sqrt(gnorm); return _t;
}

std::vector<Tensor*> CoordinateNetwork::all_gradients() {
    std::vector<Tensor*> result;
    for (auto& layer : mlp_layers_) {
        for (Tensor* g : layer->gradients()) result.push_back(g);
    }
    return result;
}

// =============================================================================
// SIREN: Implicit Neural Representations with Periodic Activation Functions
// =============================================================================

SIREN::SIREN(size_t coord_dim, const std::vector<size_t>& hidden_sizes,
             size_t output_dim, double omega0)
    : coord_dim_(coord_dim), output_dim_(output_dim), omega0_(omega0)
{
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(coord_dim);   // raw coordinates in
    for (size_t h : hidden_sizes) layer_sizes.push_back(h);
    layer_sizes.push_back(output_dim);

    for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        auto dense = std::make_unique<Dense>(layer_sizes[i], layer_sizes[i + 1]);
        if (i == 0) {
            // First layer: init for sin activation
            dense->init_weights("xavier");
        } else {
            dense->init_weights("xavier");
        }
        layers_.push_back(std::move(dense));
    }
    init_weights_siren();
}

void SIREN::init_weights_siren() {
    // Sitzmann et al. 2020 initialization for sin activation:
    // First layer: w ~ U(-1, 1) * sqrt(6/fin) / omega0
    // Other layers: w ~ U(-1, 1) * sqrt(6/fin)
    size_t num_layers = layers_.size();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t li = 0; li < num_layers; ++li) {
        Tensor& w = layers_[li]->weights;
        size_t fin = w.cols;
        double scale = (li == 0) ? (std::sqrt(6.0 / static_cast<double>(fin)) / omega0_)
                                  : std::sqrt(6.0 / static_cast<double>(fin));
        for (size_t i = 0; i < w.rows; ++i) {
            for (size_t j = 0; j < w.cols; ++j) {
                w[i][j] = dist(rng) * scale;
            }
        }
        // Bias: uniform in [-pi, pi]
        Tensor& b = layers_[li]->bias;
        std::uniform_real_distribution<double> bias_dist(-M_PI, M_PI);
        for (size_t i = 0; i < b.cols; ++i) {
            b[0][i] = bias_dist(rng);
        }
    }
}

Tensor SIREN::forward(const Tensor& coords) {
    Tensor x = coords;
    last_activations_.clear();
    last_layer_inputs_.clear();
    last_pre_sin_.clear();  // Dense outputs before sin activation

    for (size_t li = 0; li < layers_.size(); ++li) {
        last_layer_inputs_.push_back(x.clone());
        x = layers_[li]->forward(x);
        last_pre_sin_.push_back(x.clone());  // store pre-sin value

        if (li == 0) {
            // First layer: sin(omega0 * (Wx + b))
            x = x.apply([this](double v){ return std::sin(omega0_ * v); });
        } else if (li < layers_.size() - 1) {
            // Hidden layers: sin(Wx + b)
            x = x.apply([](double v){ return std::sin(v); });
        }
        // Final layer: no activation

        if (li < layers_.size() - 1) {
            last_activations_.push_back(x.clone());
        }
    }
    return x;
}

Tensor SIREN::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, output_dim)
    // Chain rule through sin activations:
    // d/dx sin(z) = cos(z) * dz/dx
    // For layer_idx with pre_sin = Dense output z:
    //   dL/d(pre_sin) = dL/d(post_sin) * cos(pre_sin)
    //   dL/d(input) = (dL/d(pre_sin) @ W^T)

    Tensor grad = grad_output;
    size_t num_layers = layers_.size();

    for (size_t li = 0; li < num_layers; ++li) {
        size_t layer_idx = num_layers - 1 - li;

        if (layer_idx == num_layers - 1) {
            // Final layer: no activation, just Dense backward
            grad = layers_[layer_idx]->backward(grad, 0.0);
        } else {
            // sin activation
            // last_pre_sin_[layer_idx] = output of Dense[layer_idx] = pre-sin value z
            const Tensor& pre_sin = last_pre_sin_[layer_idx];

            // dL/d(pre_sin) = dL/d(post_sin) * cos(pre_sin)
            Tensor grad_sin(grad.rows, grad.cols);
            for (size_t i = 0; i < grad.rows; ++i) {
                for (size_t j = 0; j < grad.cols; ++j) {
                    grad_sin[i][j] = grad[i][j] * std::cos(pre_sin[i][j]);
                }
            }
            grad = layers_[layer_idx]->backward(grad_sin, 0.0);
        }
    }

    (void)learning_rate;
    return grad;
}

void SIREN::update_weights(double learning_rate) {
    // Clip weight gradients per-layer to prevent instability
    double max_norm = 10.0;
    for (auto& layer : layers_) {
        for (Tensor* g : layer->gradients()) {
            double gnorm = 0.0;
            for (size_t i = 0; i < g->rows * g->cols; ++i)
                gnorm += (*g)[0][i] * (*g)[0][i];
            gnorm = std::sqrt(gnorm);
            if (gnorm > max_norm) {
                double scale = max_norm / gnorm;
                for (size_t i = 0; i < g->rows * g->cols; ++i)
                    (*g)[0][i] *= scale;
            }
        }
        layer->update_weights(learning_rate);
    }
}

void SIREN::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}

std::vector<Tensor*> SIREN::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : layers_) {
        for (Tensor* p : layer->parameters()) params.push_back(p);
    }
    return params;
}

std::vector<Tensor*> SIREN::gradients() {
    std::vector<Tensor*> grads;
    for (auto& layer : layers_) {
        for (Tensor* g : layer->gradients()) grads.push_back(g);
    }
    return grads;
}

Tensor SIREN::get_weights() const {
    return Tensor(0, 0);
}

Tensor SIREN::get_gradients() const {
    double gnorm = 0.0;
    for (auto& layer : layers_) {
        for (Tensor* g : layer->gradients()) {
            for (size_t i = 0; i < g->rows * g->cols; ++i)
                gnorm += (*g)[0][i] * (*g)[0][i];
        }
    }
    Tensor _t(1, 1); _t[0][0] = std::sqrt(gnorm); return _t;
}