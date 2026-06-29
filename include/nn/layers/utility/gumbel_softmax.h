#ifndef GUMBEL_SOFTMAX_H
#define GUMBEL_SOFTMAX_H

#include "../../core/layer.h"
#include <random>
#include <string>

// =====================================================================
// Gumbel-Softmax — Jang et al. 2017 "Categorical Reparameterization with
// Gumbel-Softmax" (https://arxiv.org/abs/1611.01144) and Maddison et al.
// 2017 "The Concrete Distribution: A Continuous Relaxation of Discrete
// Random Variables" (https://arxiv.org/abs/1611.00512).
//
// Forward:  y = softmax((logits + g) / temperature)
//   where g_i = -log(-log(u_i)),  u_i ~ Uniform(0, 1).
//
// The output is a continuous, differentiable proxy for sampling from the
// Categorical(logits) distribution. As temperature -> 0 the sample
// concentrates on the argmax; as temperature -> +inf it becomes uniform.
//
// hard = true (Straight-Through Estimator, Jang et al. §2.3):
//   The forward returns a one-hot vector one_hot(argmax(y_soft)).
//   The backward, however, flows through the *soft* y_soft — so the
//   upstream gradients are computed against a smooth relaxation, not the
//   discrete sample. This is the standard STE pattern: identical soft
//   gradient in both modes; only the forward output differs.
//
// training = false:
//   No Gumbel noise is added (the layer becomes the deterministic
//   temperature-scaled softmax). Useful at evaluation time when you want
//   a "soft" prediction rather than a sample.
//
// No learnable parameters — Layer interface returns empty parameters() /
// gradients() and update_weights() is a no-op.
// =====================================================================

class GumbelSoftmax : public Layer {
public:
    // temperature: positive. Smaller -> harder samples. Default 1.0.
    // hard: use the straight-through estimator (one-hot forward).
    // seed: RNG seed. 0 means "use std::random_device". The RNG is owned
    //       by the layer so test runs are reproducible when seed != 0.
    GumbelSoftmax(double temperature = 1.0, bool hard = false, unsigned seed = 0);

    // (B, K) logits -> (B, K) sample (rows sum to ~1; exactly 1 if hard).
    Tensor forward(const Tensor& logits) override;

    // (B, K) grad_output -> (B, K) grad_logits.
    // Uses the same soft-path derivative in both hard and soft modes
    // (the Straight-Through Estimator).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double /*learning_rate*/) override {}

    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}

    // Toggle training/eval. In eval mode no Gumbel noise is sampled.
    void set_training(bool t) { training_ = t; }

    // Adjust temperature (e.g. for an annealing schedule). Clamped > 0.
    void set_temperature(double t) { temperature_ = (t > 0.0 ? t : 1e-8); }

    double temperature() const { return temperature_; }
    bool hard() const { return hard_; }
    bool training() const { return training_; }

    std::string name() const override { return "GumbelSoftmax"; }

private:
    double temperature_;
    bool hard_;
    bool training_;
    unsigned seed_;
    std::mt19937 rng_;

    // Forward caches
    Tensor last_logits_;       // (B, K) — logits passed to forward()
    Tensor last_y_soft_;       // (B, K) — softmax((logits + g) / tau); basis of backward

    // Softmax with temperature (row-wise, numerically stable).
    static Tensor softmax_with_temperature(const Tensor& x, double temperature);

    // Sample a (rows, cols) Gumbel(0, 1) tensor.
    Tensor sample_gumbel(size_t rows, size_t cols);
};

#endif