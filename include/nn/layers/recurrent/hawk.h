#ifndef HAWK_H
#define HAWK_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Hawk (RG-LRU) — De et al. 2024
//   "Griffin: Mixing Gated Linear Recurrences with Local Attention for
//    Efficient Language Models"
//   https://arxiv.org/abs/2402.19427
//
// ============================================================================
//
// The Real-Gated Linear Recurrence Unit (RG-LRU) is the central building
// block of the "Hawk" architecture from the Griffin paper. It is a
// scalar-state gated linear RNN with input-dependent per-channel decay.
//
// Canonical Hawk recurrence (per channel c, per time step t):
//
//   gate_input_t  = W_x · x_t + b_x                                in R^d
//   gate_t[c]     = sigmoid(gate_input_t[c])                        in (0, 1)
//   input_proj_t[c] = gate_t[c] * x_t[c]                            (gated input)
//   λ_t[c]        = exp(-exp(log_a_raw[c]) * exp(gate_input_t[c]))  in (0, 1)
//   h_t[c]        = λ_t[c] * h_{t-1}[c]
//                  + sqrt(1 - λ_t[c]^2) * input_proj_t[c]          (recurrence)
//   y_t[c]        = h_t[c]
//   y_t           = W_o · y_t + b_o
//
// State: scalar h ∈ R^d (one hidden per channel). No matrix state. This is
// the simplest of the gated-linear-recurrence family.
//
// Parameters (all learnable):
//   W_x (d, d), b_x (1, d)        — input projection (also produces the gate)
//   log_a_raw (1, d)              — unconstrained per-channel decay base
//   W_o (d, d), b_o (1, d)        — output projection
//
// Initialization convention (Griffin paper §3.1):
//   * W_x: xavier-uniform (Dense default).
//   * b_x: zero.
//   * log_a_raw: -3.0 init → exp(log_a_raw) ≈ 0.05, so the inner exponent
//     starts small (λ_t ≈ exp(-0.05 · exp(gi)) which is close to 1 for
//     negative gate inputs and decays slower for positive ones).
//   * W_o: xavier-uniform.
//   * b_o: zero.
//
// Shape convention: forward takes (T, d) and returns (T, d). The state
// sequence is cached as (T+1, d) (row 0 = initial state 0) for BPTT.
// ============================================================================

class HawkBlock : public Layer {
public:
    // d: input/output feature dim (must satisfy d > 0)
    explicit HawkBlock(size_t d);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_x.weights; }
    Tensor get_gradients() const override { return W_x.grad_weights; }
    std::string name() const override { return "HawkBlock"; }

    // Accessors for tests
    size_t d() const { return d_; }

    // ---- Public parameters ----
    size_t d_;
    Dense W_x;              // (d, d) — input projection (also produces gate)
    Tensor log_a_raw;       // (1, d) — unconstrained decay base
    Dense W_o;              // (d, d) — output projection

    // Hidden gradient buffers
    Tensor grad_log_a_raw_; // (1, d)

    // Forward caches (publicly readable for tests / debugging, like H3Block)
    Tensor last_input_;         // (T, d) — cloned
    Tensor last_gate_input_;    // (T, d) — W_x * x + b_x
    Tensor last_gates_;         // (T, d) — sigmoid(gate_input)
    Tensor last_lambda_;        // (T, d) — exp(-exp(log_a_raw) * exp(gate_input))
    Tensor last_h_;             // (T+1, d) — h_t (row 0 = 0)
    Tensor last_input_proj_;    // (T, d) — gate_t * x_t

    // Static helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
};

#endif // HAWK_H
