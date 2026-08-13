#ifndef NEURAL_ODE_H
#define NEURAL_ODE_H

// ============================================================================
// Neural ODE / ODE-RNN — Chen, Rubanova, Du, Chen 2018, NeurIPS
//   "Neural Ordinary Differential Equations"
//   https://arxiv.org/abs/1806.07366
//
// De Brouwer et al. 2019, "GRU-ODE-Bayes: Comparing continuous-discrete
//   neural architectures for irregularly-sampled time series"
//   https://arxiv.org/abs/1905.04374
//
// ============================================================================
//
// This file provides a small continuous-depth model library:
//
//   1. namespace `odesolver` — free ODE-step functions (Euler, Midpoint, RK4,
//      Dormand-Prince RK45 fixed-step). Each takes a dynamics callable
//      f(h, t, x) and advances (h, t) by dt.
//
//   2. `ODEFunc` — a parameterized dynamics network computing
//      dh/dt = f(h, t, x). Default implementation: 2-layer MLP with tanh.
//      Exposes its own forward / backward / parameters / gradients.
//
//   3. `NeuralODE(input_dim, hidden_dim, output_dim, solver_type, dt)` —
//      continuous-depth block that integrates the ODE from t=0 to t=T (with
//      T = n_steps * dt) given a constant input x, then projects the final
//      hidden state to output_dim. Two backward modes:
//        - direct backprop through the solver (O(depth) memory)
//        - adjoint method (O(1) memory; matches direct at ~1e-5)
//
//   4. `ODERNN(input_dim, hidden_dim, output_dim, solver_type, dt)` —
//      sequence model that updates hidden state via a gated input injection
//      at each observation time, then evolves via ODE between observations
//      (De Brouwer et al. 2019). Each batch element can have its own
//      irregular observation-time sequence.
//
// All tensors are (rows=batch, cols=features) — single-row tensors are used
// to represent (1, features) hidden states.
//
// ----------------------------------------------------------------------------
// Math (single batch element, hidden state h in R^d):
//
//   NeuralODE forward:
//     h(0) = h_init       (zero by default)
//     for k = 0, ..., n_steps-1:
//       h(k+1) = solver_step(f, h(k), k*dt, dt, x)
//     y = output_proj(h(n_steps))
//
//   NeuralODE direct backward (per step, dh_{k+1}/dh_k = I + dt * J_f):
//     grad_h = grad_output_proj_back
//     for k = n_steps-1, ..., 0:
//       grad_h = grad_h @ (I + dt * J_f(h(k), k*dt, x))     # (1, hidden)
//       grad_ODEFunc += grad_h^T @ ...  (parameter gradients via Dense backward)
//
//   NeuralODE adjoint backward (continuous adjoint, Euler-step approximation):
//     a_h = grad_output_proj_back                              # (1, hidden)
//     for k = n_steps-1, ..., 0:
//       J_h = ∂f/∂h at (h(k), k*dt, x)                          # (hidden, hidden)
//       a_h = a_h + dt * a_h @ J_h                              # (1, hidden)
//       grad_ODEFunc += -dt * a_h^T @ ∂f/∂θ at (h(k), k*dt, x)
//
//   ODERNN forward (per observation index i, time t_i):
//     h_i = RNN_step(h_{i-1}, x_i)         (gated input injection)
//     h_{i+1} = ODESolve(h_i, t_i, t_{i+1})
//     y_i = output_proj(h_{i+1})
//
// ----------------------------------------------------------------------------
// Learnable parameters:
//   ODEFunc:           W1, b1, W2, b2      (Dense x 2 + tanh)
//   NeuralODE:         ODEFunc params + W_out, b_out (output projection)
//   ODERNN:            ODEFunc params + W_in, b_in, W_h, b_h (gated input)
//                      + W_out, b_out
//
// ============================================================================

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <string>
#include <functional>
#include <cstddef>
#include <memory>

namespace odesolver {

// Dynamics callable type: given (h, t, x) returns dh/dt of same shape as h.
using Dynamics = std::function<Tensor(const Tensor&, double, const Tensor&)>;

// Step functions. Each advances the state from t to t+dt using the
// specified scheme and returns the new state (same shape as h). The
// dynamics f must accept (h, t, x) and return dh/dt.
Tensor euler_step   (const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x);
Tensor midpoint_step(const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x);
Tensor rk4_step     (const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x);
Tensor dopri5_step  (const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x);

}  // namespace odesolver

// ODEFunc — parameterized dynamics network (the "vector field" f).
// Default implementation: 2-layer MLP with tanh. The first layer uses
// separate W_h, b_h (for h) and W_x, b_x (for x) projections, then sums
// the two pre-activations and applies tanh. The second layer (W_out, b_out)
// maps the hidden activation back to the dynamics output.
class ODEFunc : public Layer {
public:
    size_t input_dim_;
    size_t hidden_dim_;

    // Sublayers
    Dense dense1_;   // (hidden + input) -> hidden
    Dense dense2_;   // hidden -> hidden

    // Caches for backward
    Tensor last_h_;
    Tensor last_t_;  // scalar as (1,1)
    Tensor last_x_;
    Tensor last_pre_act_;  // post-tanh activations (hidden)

    ODEFunc(size_t input_dim, size_t hidden_dim);
    Tensor forward(const Tensor& h, double t, const Tensor& x);  // returns dh/dt
    Tensor forward(const Tensor& input) override;                // Layer interface — wraps (h, t, x) via last_h_/last_x_; rarely called directly
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ODEFunc"; }

    // Analytic Jacobian: ∂f/∂h at the LAST cached state. Shape (hidden, hidden).
    // Uses the chain rule for the 2-layer MLP + tanh: df/dh = J_tanh @ W2 @ J_dense1_h
    //   where J_tanh = diag(1 - tanh(z)^2), J_dense1_h = W1_h (the hidden-to-hidden slice).
    Tensor jacobian_h() const;

    // Per-parameter Jacobian column for parameter index p: ∂f/∂θ_p at the LAST cached state.
    // Returns a (1, hidden) row vector (i.e., the gradient contribution if the parameter
    // gradient is 1 for p and 0 elsewhere). Used by the adjoint method.
    // `param_layer` selects which Dense's parameter to compute the Jacobian for
    // ("dense1" or "dense2"); `param_kind` is "weights" or "bias".
    Tensor jacobian_param_col(const std::string& param_layer, const std::string& param_kind, size_t row, size_t col) const;

    // Helpers for tests: forward-with-cache version (saves inputs for jacobian_*).
    Tensor forward_with_cache(const Tensor& h, double t, const Tensor& x);
};

// NeuralODE — integrates ODE from t=0 to t=T, projects to output_dim.
class NeuralODE : public Layer {
public:
    size_t input_dim_;
    size_t hidden_dim_;
    size_t output_dim_;
    std::string solver_type_;  // "euler" | "midpoint" | "rk4" | "dopri5"
    size_t n_steps_;
    double dt_;

    ODEFunc odefunc_;
    Dense output_proj_;  // hidden -> output_dim

    // Trajectory cache for direct backward (one entry per step: h_k).
    std::vector<Tensor> trajectory_h_;
    std::vector<Tensor> trajectory_x_;  // constant across steps but cached per call
    std::vector<double> trajectory_t_;
    // Per-step grad accumulator for ODEFunc parameters (set up lazily during backward).
    std::vector<Tensor> grad_dense1_w_;
    std::vector<Tensor> grad_dense1_b_;
    std::vector<Tensor> grad_dense2_w_;
    std::vector<Tensor> grad_dense2_b_;

    // Adjoint-mode caches
    Tensor last_grad_input_;       // grad w.r.t. input x (returned by backward)
    Tensor last_grad_output_proj_w_;
    Tensor last_grad_output_proj_b_;
    bool use_adjoint_;

    NeuralODE(size_t input_dim, size_t hidden_dim, size_t output_dim,
              const std::string& solver_type = "euler",
              size_t n_steps = 10,
              double dt = 0.1,
              bool use_adjoint = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "NeuralODE"; }

    void set_use_adjoint(bool b) { use_adjoint_ = b; }
    bool is_using_adjoint() const { return use_adjoint_; }
};

// ODERNN — handles irregularly-sampled sequence observations
// (De Brouwer et al. 2019, GRU-ODE-Bayes).
class ODERNN : public Layer {
public:
    size_t input_dim_;
    size_t hidden_dim_;
    size_t output_dim_;
    std::string solver_type_;
    double dt_;

    ODEFunc odefunc_;   // between-observation dynamics
    Dense input_proj_;  // x -> hidden (for input injection)
    Dense hidden_proj_; // h -> hidden (for input injection)
    Dense output_proj_;// hidden -> output_dim

    // Caches
    std::vector<Tensor> trajectory_h_;      // hidden state after each step
    std::vector<Tensor> trajectory_x_;      // input at each observation
    std::vector<double> trajectory_t_;      // observation time
    std::vector<size_t> steps_per_seg_;     // number of ODE steps taken between observations

    Tensor last_grad_input_;

    ODERNN(size_t input_dim, size_t hidden_dim, size_t output_dim,
           const std::string& solver_type = "euler",
           double dt = 0.1);

    // X_seq: (T, input_dim); T_seq: (T,) — observation times (in arbitrary units).
    // Returns (T, output_dim) — predictions at each observation.
    Tensor forward_seq(const Tensor& X_seq, const std::vector<double>& T_seq);

    Tensor forward(const Tensor& input) override;  // not meaningful for sequences
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ODERNN"; }

    // Backward input gradient accessor (the sequence backward path returns a
    // (T, input_dim) tensor instead of fitting the (1, input_dim) Layer API).
    const Tensor& last_grad_input() const { return last_grad_input_; }
};

#endif  // NEURAL_ODE_H
