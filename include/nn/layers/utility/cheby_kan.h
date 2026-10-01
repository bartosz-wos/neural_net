#ifndef CHEBY_KAN_H
#define CHEBY_KAN_H

#include "../../core/layer.h"
#include <vector>
#include <memory>
#include <cstddef>

// =====================================================================
// ChebyKAN — Sidharth SSIDC 2024
//   "ChebyKAN: A Kolmogorov-Arnold Network with Chebyshev Polynomials"
//   https://arxiv.org/abs/2404.10978
//
// Per-edge learnable activation as a Chebyshev-polynomial expansion of
// the first kind on the bounded input domain [-1, 1]:
//
//   phi_{l,i,j}(x) = Σ_{k=0..K} c_{l,i,j,k} · T_k(x)
//
// where T_k(x) is the Chebyshev polynomial of the first kind, defined by
// the recurrence
//
//   T_0(x) = 1,   T_1(x) = x,   T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x)   (k ≥ 2)
//
// and satisfying |T_k(x)| ≤ 1 on [-1, 1]. Unlike FourierKAN, the basis
// here is non-trigonometric and orthogonal w.r.t. the Chebyshev weight
// (1/√(1-x²)), making it the natural choice when the input lives on
// [-1, 1] (the standard KAN convention after the radial basis / silu
// input normalisation).
//
// Data layout per ChebyKANLayer(in_features, out_features, num_freq):
//   coefs_  : (out, in * (num_freq + 1))  flattened, row i col j*(K+1)+k
//             holds the coefficient c_{l,i,j,k} for edge (i, j) at order k.
//   (K+1) coefficients because we include the k=0 (constant) term.
//
// Initialization: coefs_ ~ N(0, 1/in) so the initial phi has small
// magnitude (sigma² = 1/in matches the canonical KAN paper's variance).
//
// ChebyKANModel: stack of ChebyKANLayers with a final layer mapping to
// out_dim. No inter-layer nonlinearity (matches the existing KAN and
// FourierKAN wrappers; per-edge Chebyshev basis is already nonlinear).
// =====================================================================

class ChebyKANLayer : public Layer {
public:
    // in_features, out_features: layer dimensions
    // num_freq: highest Chebyshev order K (we include orders 0..K inclusive)
    ChebyKANLayer(size_t in_features, size_t out_features, size_t num_freq = 8);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return coefs_; }
    Tensor get_gradients() const override { return grad_coefs_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ChebyKANLayer"; }

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t num_freq() const { return num_freq_; }
    size_t n_coefs() const { return num_freq_ + 1; }  // K+1 coefficients per edge

private:
    size_t in_features_;
    size_t out_features_;
    size_t num_freq_;

    Tensor coefs_;         // (out, in * (K+1))

    Tensor grad_coefs_;    // (out, in * (K+1))

    // Forward cache
    Tensor last_input_;    // (B, in)
    // last_T_(b, j*(K+1) + k) = T_k(x[b, j])
    // Cached so backward becomes a per-(b, j) inner-product with coefs.
    Tensor last_T_;        // (B, in * (K+1))
    // last_dphi_dx_(b, i*(in*(K+1)) + j*(K+1) + k) = c_{i,j,k} * T'_k(x[b, j])
    // Cached so d_input[b, j] is a single per-(b, j) sum over k.
    Tensor last_dphi_dx_;  // (B, out * in * (K+1))
};

// ChebyKANModel: stack of ChebyKANLayers, optional hidden dims,
// final layer maps to out_dim.
class ChebyKANModel : public Layer {
public:
    ChebyKANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
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
    std::string name() const override { return "ChebyKANModel"; }

    const std::vector<std::unique_ptr<ChebyKANLayer>>& layers() const { return layers_; }

private:
    std::vector<std::unique_ptr<ChebyKANLayer>> layers_;
    std::vector<Tensor> last_layer_outputs_; // forward cache
};

#endif
