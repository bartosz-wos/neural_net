#ifndef SPECTRAL_NORM_H
#define SPECTRAL_NORM_H

#include "../../core/layer.h"

// Spectral Normalization (SNGAN - Miyato et al. 2018)
// Constrains the spectral norm (largest singular value) of a weight matrix to 1.
// W_normalized = gamma * W / sigma(W), where sigma(W) is computed via power iteration.

class SpectralNorm : public Layer {
public:
    Tensor gamma;   // learnable output scale (initialized to 1)
    Tensor u;       // left singular vector approximation (power iteration)
    Tensor v;       // right singular vector approximation (power iteration)
    Tensor last_W;  // current normalized weight matrix for gradient computation
    Tensor W;       // original weight matrix before normalization

    int power_iterations;  // number of power iteration steps (default 1)
    double eps;            // numerical stability constant

    // Constructor: W should be the weight matrix of a Dense layer (out_features, in_features)
    SpectralNorm(const Tensor& W, int power_iterations = 1, double eps = 1e-6);

    // Compute spectral norm via power iteration
    double compute_spectral_norm();

    // Power iteration step: normalize vector with given epsilon threshold
    static Tensor normalize(const Tensor& vec, double eps = 1e-6);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return last_W; }
    Tensor get_gradients() const override { return Tensor(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "SpectralNorm"; }
};

#endif