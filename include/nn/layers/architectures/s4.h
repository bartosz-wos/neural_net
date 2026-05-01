#ifndef S4_H
#define S4_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// S4Layer: Structured State Space Sequence Model
// Implements the SSM with diagonal HIPPO initialization for long-range dependencies.
// Mathematical formulation (per sequence position t):
//   h'[t] = Lambda @ h[t] + B * u[t]
//   y[t]  = C @ h[t] + D * x[t]
// With convolution view for efficient parallel computation.
class S4Layer : public Layer {
public:
    // d_model: input/output dimension
    // d_state: SSM state dimension (number of HIPPO modes)
    S4Layer(int d_model, int d_state);
    ~S4Layer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return x_proj; }
    Tensor get_gradients() const override { return grad_x_proj; }
    std::string name() const override { return "S4Layer"; }

    // SSM parameters (public for gradient checks in tests)
    Tensor x_proj;     // (d_model, d_state*2) — input projection
    Tensor W_out;      // (d_state, d_model) — state-to-output projection
    Tensor b_out;      // (1, d_model) — output bias
    Tensor Lambda;     // (d_state,) — diagonal of state matrix A (real, HIPPO init)
    Tensor B;          // (d_state,) — input-to-state projection
    Tensor C;          // (d_state,) — state-to-output projection
    double D = 0.0;    // skip connection (scalar)

    // Gradient accumulators (named with trailing underscore to match test expectations)
    Tensor grad_x_proj;
    Tensor grad_W_out;
    Tensor grad_b_out;
    Tensor grad_Lambda_;
    Tensor grad_B_;
    Tensor grad_C_;
    double grad_D_ = 0.0;

private:
    int d_model_;
    int d_state_;
    int seq_len_cached_;
    bool has_cachedKernel_;
    Tensor cached_K_;   // SSM kernel for current sequence length
    Tensor last_input_;
    Tensor last_u_;     // (d_state*2, seq_len) — projected input for backward

    // Discretization via bilinear transform
    static double discretize_Lambda(double lam, double dt);
    static double discretize_B(double B_val, double lam, double dt);
};

#endif // S4_H
