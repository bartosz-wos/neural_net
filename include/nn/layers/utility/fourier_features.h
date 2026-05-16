#ifndef FOURIER_FEATURES_H
#define FOURIER_FEATURES_H

#include "../../core/layer.h"
#include <random>
#include <cmath>

// Base class for Fourier feature encodings
// Encodes 2D (or ND) coordinates into a Fourier feature space for input to an MLP.
// Formula: gamma(x) = [cos(2π B x), sin(2π B x)] where B is a frequency matrix.
// Each row of B defines a frequency vector b_i, giving feature pair [cos(2π b_i^T x), sin(2π b_i^T x)].
class FourierFeatures : public Layer {
public:
    // input_dim: coordinate dimension (e.g., 2 for 2D)
    // num_frequencies: number of frequency vectors (output will be 2*num_frequencies features)
    FourierFeatures(size_t input_dim, size_t num_frequencies, double sigma = 1.0);

    Tensor forward(const Tensor& coords) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return frequencies_; }
    Tensor get_gradients() const override { return grad_frequencies_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FourierFeatures"; }

    size_t input_dim() const { return input_dim_; }
    size_t num_frequencies() const { return num_frequencies_; }
    size_t output_size() const { return 2 * num_frequencies_; }

protected:
    size_t input_dim_;
    size_t num_frequencies_;
    double sigma_;
    Tensor frequencies_;         // shape: (num_frequencies, input_dim)
    Tensor grad_frequencies_;    // shape: (num_frequencies, input_dim)
    Tensor last_coords_;         // cached input for backward
    Tensor last_encoded_;        // cached output for backward (N, 2*F)
    std::mt19937 rng_;

public:
};

// GaussianFourierFeatures: frequency matrix B sampled from N(0, sigma^2 I)
// This is the standard approach from Tancik et al. 2020.
class GaussianFourierFeatures : public FourierFeatures {
public:
    GaussianFourierFeatures(size_t input_dim, size_t num_frequencies, double sigma = 1.0);
    std::string name() const override { return "GaussianFourierFeatures"; }
};

// LearnedFourierFeatures: the frequency vectors are learnable parameters.
// After training, the learned frequencies can be used like Gaussian ones.
class LearnedFourierFeatures : public FourierFeatures {
public:
    LearnedFourierFeatures(size_t input_dim, size_t num_frequencies);
    std::string name() const override { return "LearnedFourierFeatures"; }
};

#endif