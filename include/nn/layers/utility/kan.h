#ifndef KAN_H
#define KAN_H

#include "../../core/layer.h"
#include <vector>
#include <memory>
#include <cstddef>

// =====================================================================
// KAN: Kolmogorov-Arnold Networks — Liu et al. 2024
//   https://arxiv.org/abs/2404.19756
//
// Per-edge learnable activation
//   phi_{l,i,j}(x) = w_b * b(x) + w_s * spline(x; coefs_{l,i,j}, grid_l)
// where b(x) = silu(x) is the base activation, and the spline is a B-spline
// evaluated on a layer-shared clamped uniform knot grid (frozen during training).
//
// Data layout per KANLayer(in_features, out_features, num_grids, spline_order):
//   spline_coefs_  : (out_features, in_features * n_coefs)
//                    Row i, cols [j*n_coefs .. (j+1)*n_coefs - 1] are the spline
//                    coefficients for edge (i, j). n_coefs = num_grids + spline_order.
//   base_weight_   : (out_features, in_features)  — scalar per-edge base scaler
//   spline_weight_ : (out_features, in_features)  — scalar per-edge spline scaler
//   grid_          : (1, G + 2k + 1)              — frozen clamped uniform knots
//
// KANModel: stack of KANLayers (input -> hidden -> ... -> output_dim).
// Each KANLayer (except the last) is followed by an elementwise base activation
// (default silu) applied to the post-edge-aggregation output — same convention
// as the original KAN paper (where the "layer norm + activation" is just silu).
// =====================================================================

class KANLayer : public Layer {
public:
    // in_features, out_features: layer dimensions
    // num_grids: number of spline intervals G (the spline resolution)
    // spline_order: spline order k (degree k-1). Default 3 = cubic.
    // grid_range: [a, b] for the clamped uniform knot grid. Default [-2, 2].
    // grid_eps: NOT used in this version (grid is fixed at construction; no
    //   "grid extension" trick from the paper, which adds complexity but doesn't
    //   affect correctness for in-range inputs).
    KANLayer(size_t in_features, size_t out_features,
             size_t num_grids = 5, size_t spline_order = 3,
             double grid_low = -2.0, double grid_high = 2.0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return spline_coefs_; }
    Tensor get_gradients() const override { return grad_spline_coefs_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "KANLayer"; }

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t num_grids() const { return num_grids_; }
    size_t spline_order() const { return spline_order_; }

private:
    size_t in_features_;
    size_t out_features_;
    size_t num_grids_;     // G
    size_t spline_order_;  // k
    size_t n_coefs_;       // G + k
    size_t n_knots_;       // G + 2k + 1

    // Learnable parameters
    Tensor spline_coefs_;    // (out, in * n_coefs)
    Tensor base_weight_;     // (out, in)
    Tensor spline_weight_;   // (out, in)

    // Frozen (not learnable in this version) but kept as a parameter for API completeness
    Tensor grid_;            // (1, n_knots)

    // Gradients (only for the 3 learnable tensors)
    Tensor grad_spline_coefs_;
    Tensor grad_base_weight_;
    Tensor grad_spline_weight_;

    // Forward cache
    Tensor last_input_;          // (B, in)
    // post-aggregation output pre-base-activation (currently unused — no inter-layer activation yet)
    Tensor last_pre_act_;        // (B, out)
    // Per-(batch, edge) phi value, used for backward of inputs and weights
    // Stored as a flat (B, out * in) tensor — entry (b, i*in + j) = phi_{l,i,j}(x[b, j])
    Tensor last_phi_;            // (B, out * in)
    // Per-(batch, edge) base + spline contributions, used for the base/spline weight grads
    Tensor last_base_val_;       // (B, out * in) — base_weight[i,j] * silu(x[b, j]) (with sign convention)
    Tensor last_spline_val_;     // (B, out * in) — spline_weight[i,j] * spline(coefs, x[b, j])
};

// KANModel: stack of KANLayers. Optional per-layer base activation applied after
// each KANLayer (so the full network is a sequence of "edge activations then
// pointwise nonlinearity"). For now we omit the inter-layer nonlinearity to keep
// the gradient chain tight — the model can still learn because per-edge
// activations are themselves nonlinear.
class KANModel : public Layer {
public:
    // in_dim, hidden_dims: layer dimensions; final layer maps to out_dim.
    //   If hidden_dims is empty, model is a single KANLayer(in_dim -> out_dim).
    // num_grids, spline_order: passed to every KANLayer.
    KANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
             size_t out_dim, size_t num_grids = 5, size_t spline_order = 3);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return layers_.empty() ? Tensor(0, 0) : layers_.front()->get_weights(); }
    Tensor get_gradients() const override { return layers_.empty() ? Tensor(0, 0) : layers_.front()->get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "KANModel"; }

    const std::vector<std::unique_ptr<KANLayer>>& layers() const { return layers_; }

private:
    std::vector<std::unique_ptr<KANLayer>> layers_;
    std::vector<Tensor> last_layer_outputs_; // forward cache: output of layer i (= input to layer i+1)
};

#endif