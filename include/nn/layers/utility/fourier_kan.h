#ifndef FOURIER_KAN_H
#define FOURIER_KAN_H

#include "../../core/layer.h"
#include <vector>
#include <memory>
#include <cstddef>

// =====================================================================
// FourierKAN — Xu et al. 2024
//   "FourierKAN: Highly Efficient KAN Based on Fast Fourier Transform"
//   https://arxiv.org/abs/2410.02803
//
// Per-edge learnable activation as a truncated real Fourier series on
// the bounded input domain [a, b] (default [-1, 1] — the standard
// Fourier basis is naturally defined on [-1, 1] without rescaling):
//
//   phi_{l,i,j}(x) = a_0_{l,i,j} / 2
//                  + Σ_{k=1..K} [ a_{l,i,j,k} · cos(k·π·x)
//                                + b_{l,i,j,k} · sin(k·π·x) ]
//
// We adopt the truncated form (no DC bias term in v1, matching the
// paper's reported ablation): start the cos terms at k=1, drop the
// constant-offset. This keeps the layer centered at zero on average
// and avoids an extra hyperparameter.
//
// Data layout per FourierKANLayer(in_features, out_features, num_freq):
//   a_coefs_  : (out, in * num_freq)  flattened, row i col j*K+k holds a_{i,j,k}
//   b_coefs_  : (out, in * num_freq)  flattened, row i col j*K+k holds b_{i,j,k}
//
// Initialization: a_coefs_ ~ N(0, 1/(in * K)) and b_coefs_ ~ N(0, 1/(in * K))
// so the initial phi has small magnitude — stable for downstream layers.
//
// FourierKANModel: stack of FourierKANLayers with a final layer mapping
// to out_dim. No inter-layer nonlinearity (matches the existing KAN
// wrapper convention; per-edge Fourier basis is already nonlinear).
// =====================================================================

class FourierKANLayer : public Layer {
public:
    // in_features, out_features: layer dimensions
    // num_freq: number of Fourier modes K (k = 1..K)
    FourierKANLayer(size_t in_features, size_t out_features, size_t num_freq = 8);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return a_coefs_; }
    Tensor get_gradients() const override { return grad_a_coefs_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FourierKANLayer"; }

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t num_freq() const { return num_freq_; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t num_freq_;

    Tensor a_coefs_;        // (out, in * K)
    Tensor b_coefs_;        // (out, in * K)

    Tensor grad_a_coefs_;   // (out, in * K)
    Tensor grad_b_coefs_;   // (out, in * K)

    // Forward cache
    Tensor last_input_;         // (B, in)
    // last_phi_(b, i*K_in + j*K + k) = cos(k*pi*x[b,j])  for grad_a chain
    //                                  + sin(k*pi*x[b,j])  for grad_b chain
    // Stored separately so the backward chain is a clean per-edge dot product.
    Tensor last_cos_;          // (B, out * in * K)
    Tensor last_sin_;          // (B, out * in * K)
    // last_dphi_dx_(b, i*K_in + j*K + k) = -k*pi*a[i,j,k]*sin(...) + k*pi*b[i,j,k]*cos(...)
    // cached so d_input[b, j] is just a per-j sum.
    Tensor last_dphi_dx_;      // (B, in * K)
};

// FourierKANModel: stack of FourierKANLayers, optional hidden dims,
// final layer maps to out_dim.
class FourierKANModel : public Layer {
public:
    FourierKANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
                    size_t out_dim, size_t num_freq = 8);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override {
        return layers_.empty() ? Tensor(0, 0) : layers_.front()->get_weights();
    }
    Tensor get_gradients() const override {
        return layers_.empty() ? Tensor(0, 0) : layers_.front()->get_gradients();
    }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FourierKANModel"; }

    const std::vector<std::unique_ptr<FourierKANLayer>>& layers() const { return layers_; }

private:
    std::vector<std::unique_ptr<FourierKANLayer>> layers_;
    std::vector<Tensor> last_layer_outputs_; // forward cache
};

#endif