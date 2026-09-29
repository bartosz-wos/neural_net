#include "dynamic_tanh.h"
#include <cmath>
#include <stdexcept>

// DynamicTanh (DyT) — Zhu, Chen, He (CVPR 2025, arXiv:2503.10622)
// "Transformers without Normalization"
//
// out = gamma * tanh(alpha * x) + beta
//
// Three learnable parameters:
//   alpha: (1, 1)        scalar scaler — paper §7.1: init 0.5
//   gamma: (1, features) per-channel scale — init ones
//   beta:  (1, features) per-channel shift — init zeros
//
// Element-wise operation. No batch statistics computed.

DynamicTanh::DynamicTanh(size_t features, double init_alpha)
    : alpha(Tensor(1, 1))
    , gamma(Tensor(1, features))
    , beta(Tensor(1, features))
    , init_alpha(init_alpha)
    , features_(features)
{
    if (features == 0) {
        throw std::invalid_argument("DynamicTanh: features must be > 0");
    }
    // α ← init_alpha (scalar)
    alpha[0][0] = init_alpha;
    // γ ← 1 (per-channel)
    gamma.fill(1.0);
    // β ← 0 (per-channel)
    beta.fill(0.0);

    // Initialize grad buffers to zero so first zero_grad is a no-op.
    grad_alpha_ = Tensor(1, 1);
    grad_alpha_.fill(0.0);
    grad_gamma_ = Tensor(1, features);
    grad_gamma_.fill(0.0);
    grad_beta_  = Tensor(1, features);
    grad_beta_.fill(0.0);
}

Tensor DynamicTanh::forward(const Tensor& input) {
    const size_t batch    = input.rows;
    const size_t features = input.cols;
    if (features != features_) {
        throw std::invalid_argument("DynamicTanh::forward: feature dim mismatch (input "
                                    + std::to_string(features) + " vs "
                                    + std::to_string(features_) + ")");
    }

    last_input = input;
    last_tanh  = Tensor(batch, features);

    Tensor output(batch, features);
    const double a = alpha[0][0];

    for (size_t b = 0; b < batch; ++b) {
        for (size_t f = 0; f < features; ++f) {
            double t = std::tanh(a * input[b][f]);
            last_tanh[b][f] = t;
            output[b][f] = gamma[0][f] * t + beta[0][f];
        }
    }
    return output;
}

Tensor DynamicTanh::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t batch    = grad_output.rows;
    const size_t features = grad_output.cols;

    // Lazy-init on first backward call (matches LayerNorm/RMSNorm convention).
    // Subsequent calls accumulate across backward() invocations before zero_grad().
    if (grad_alpha_.rows == 0) {
        grad_alpha_ = Tensor(1, 1);
        grad_alpha_.fill(0.0);
    }
    if (grad_gamma_.rows == 0) {
        grad_gamma_ = Tensor(1, features);
        grad_gamma_.fill(0.0);
    }
    if (grad_beta_.rows == 0) {
        grad_beta_ = Tensor(1, features);
        grad_beta_.fill(0.0);
    }

    grad_x = Tensor(batch, features);

    const double a = alpha[0][0];

    // Reset accumulator for α (it is a single scalar — no accumulation across
    // batch or feature dims within a single backward).
    // For γ and β we accumulate per-channel.
    // (Note: the initial fill(0) above is for the FIRST backward call; on
    //  subsequent calls we want += accumulation, not overwrite. So we do NOT
    //  fill(0) here — we accumulate into the existing grads. For grad_alpha_,
    //  which is a single scalar, we accumulate by += over the entire batch×features.)

    double g_alpha_acc = 0.0;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t f = 0; f < features; ++f) {
            const double go    = grad_output[b][f];
            const double t     = last_tanh[b][f];          // tanh(α·x)
            const double otmt2 = 1.0 - t * t;               // d(tanh)/d(arg)
            const double x_bf  = last_input[b][f];

            // grad_x[b][f] = grad_out · γ · (1 - t²) · α
            grad_x[b][f] = go * gamma[0][f] * otmt2 * a;

            // contribution to grad_alpha from this cell: grad_out · γ · (1 - t²) · x
            g_alpha_acc += go * gamma[0][f] * otmt2 * x_bf;
        }
    }

    // grad_gamma[f] = Σ_b grad_out[b][f] · t[b][f]
    // grad_beta[f]  = Σ_b grad_out[b][f]
    // Accumulate (per LayerNorm/RMSNorm convention).
    for (size_t f = 0; f < features; ++f) {
        double g_gamma_f = 0.0;
        double g_beta_f  = 0.0;
        for (size_t b = 0; b < batch; ++b) {
            g_gamma_f += grad_output[b][f] * last_tanh[b][f];
            g_beta_f  += grad_output[b][f];
        }
        grad_gamma_[0][f] += g_gamma_f;
        grad_beta_[0][f]  += g_beta_f;
    }

    // grad_alpha is scalar — accumulate into the single element.
    grad_alpha_[0][0] += g_alpha_acc;

    return grad_x;
}

void DynamicTanh::update_weights(double learning_rate) {
    // γ and β updates
    for (size_t f = 0; f < features_; ++f) {
        gamma[0][f] -= learning_rate * grad_gamma_[0][f];
        beta[0][f]  -= learning_rate * grad_beta_[0][f];
    }
    // α scalar update
    alpha[0][0] -= learning_rate * grad_alpha_[0][0];
}

std::vector<Tensor*> DynamicTanh::parameters() {
    return {&alpha, &gamma, &beta};
}

std::vector<Tensor*> DynamicTanh::gradients() {
    return {&grad_alpha_, &grad_gamma_, &grad_beta_};
}

void DynamicTanh::zero_grad() {
    grad_alpha_.fill(0.0);
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);
}