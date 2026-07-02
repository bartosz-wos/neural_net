#ifndef HYPERNETWORK_H
#define HYPERNETWORK_H

#include "../../core/layer.h"
#include <vector>
#include <string>

// =========================================================================
// HyperNetwork
// =========================================================================
//
// Ha, Dai, Le 2016, "HyperNetworks" (https://arxiv.org/abs/1609.09106).
//
// A HyperNetwork is a small neural network whose output is the WEIGHTS of
// another (main) neural network. The generated weights depend on a context
// vector z — so the main network's behaviour is conditioned on z. This is
// useful for parameter sharing across related tasks, dynamic / conditional
// computation, weight compression, learned optimizers, and so on.
//
// In this implementation we generate a single Dense-like main layer with
// weights W ∈ R^{out×in} and bias b ∈ R^{1×out} from a context vector
// z ∈ R^{1×context_dim}. The hyper-network itself is a 2-layer MLP:
//
//     z_hidden = σ(z · W_h1^T + b_h1)           shape (1, hyper_hidden)
//     w_flat   = z_hidden · W_h2^T + b_h2       shape (1, in*out + out)
//
// where σ is the configured nonlinearity (default Tanh). The flat output
// is split into W (first in*out entries) and b (last out entries), both
// reshaped appropriately.
//
// Forward (one context, batched input):
//     (W, b) = generate(z)                       // one W, b per forward
//     y_i = x_i · W^T + b                        // (B, out)
//
// Backward:
//   ∂L/∂x flows back through the generated dense layer.
//   ∂L/∂(W, b) flows back through generate(), into hyper1_/hyper2_ params.
//
// The hyper-network's "input" is the context z — also returned via
// backward_with_context() so the caller can chain back into whatever
// produced z (a recurrent state, an embedding, etc.).
//
// NOTE: the generated W, b are NOT stored as Layer parameters — they
// are derived from the hyper-network's weights each forward pass. The
// only "trainable" parameters are W_h1, b_h1, W_h2, b_h2. zero_grad() /
// update_weights() / parameters() / gradients() all reflect this.
// =========================================================================
class HyperNetwork : public Layer {
public:
    // context_dim:    dimension of the conditioning vector z.
    // in_features:    dimension of the input to the GENERATED layer.
    // out_features:   dimension of the output of the GENERATED layer.
    // hyper_hidden:   hidden width of the hyper MLP (z → z_hidden).
    //                 The hyper-MLP is 2 layers deep.
    // use_bias:       whether the generated layer has a bias (default true).
    HyperNetwork(size_t context_dim,
                 size_t in_features,
                 size_t out_features,
                 size_t hyper_hidden = 32,
                 bool   use_bias    = true);

    // forward(x): requires last_context_ to be set via set_context().
    // Equivalent to forward_with_context(x, last_context_).
    Tensor forward(const Tensor& input) override;

    // Set the context vector (shape (1, context_dim) OR (context_dim,) via
    // the (1, dim) constructor). Stored until the next forward call.
    void set_context(const Tensor& z);

    // Forward with explicit context. Sets last_context_ and returns y.
    Tensor forward_with_context(const Tensor& input, const Tensor& z);

    // Standard backward (chain-rule compatible). Propagates ∂L/∂x AND
    // ∂L/∂context back into the hyper-network parameters.
    // Returns the gradient w.r.t. the INPUT x of shape (B, in_features).
    // The gradient w.r.t. the context z (shape (1, context_dim)) is also
    // computed and stored — call last_context_grad() to retrieve it.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    // Apply weight decay-style update to the hyper-network parameters
    // (W_h1, b_h1, W_h2, b_h2). Uses plain SGD: param -= lr * grad.
    void update_weights(double learning_rate) override;

    Tensor get_weights() const override { return hyper1_.weights; }
    Tensor get_gradients() const override { return hyper1_.grad_weights; }

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyperNetwork"; }

    // ----- inspectors -----
    size_t context_dim()  const { return context_dim_; }
    size_t in_features()  const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t hyper_hidden() const { return hyper_hidden_; }
    bool   use_bias()     const { return use_bias_; }
    const Tensor& last_context()      const { return last_context_; }
    const Tensor& last_context_grad() const { return last_context_grad_; }
    const Tensor& last_W()            const { return last_W_; }
    const Tensor& last_b()            const { return last_b_; }

    // Test-only accessors (write-enabled). Numerical gradient checks and
    // mutation tests need to perturb the inner Dense layers' weights/grads
    // without going through the full update loop.
    Dense&       hyper1()        { return hyper1_; }
    Dense&       hyper2()        { return hyper2_; }
    const Dense& hyper1() const  { return hyper1_; }
    const Dense& hyper2() const  { return hyper2_; }

private:
    // ----- dimensions -----
    size_t context_dim_;
    size_t in_features_;
    size_t out_features_;
    size_t hyper_hidden_;
    bool   use_bias_;
    size_t flat_size_;   // in_features * out_features + (use_bias_ ? out_features : 0)

    // ----- hyper MLP: z (1×context_dim) → flat weights (1×flat_size_) -----
    Dense hyper1_;   // (hyper_hidden, context_dim)
    Dense hyper2_;   // (flat_size_, hyper_hidden)

    // ----- caches -----
    Tensor last_context_;        // (1, context_dim) — the z used in forward
    Tensor last_context_grad_;   // (1, context_dim) — grad of loss w.r.t. z
    Tensor last_z_hidden_;       // (1, hyper_hidden) — post-activation hyper1
    Tensor last_z_hidden_pre_;   // (1, hyper_hidden) — pre-activation hyper1
    Tensor last_input_;          // (B, in_features)
    Tensor last_W_;               // (out_features, in_features) — generated weights
    Tensor last_b_;               // (1, out_features) — generated bias

    // ----- helpers -----
    // Run the hyper MLP on z, fill last_W_ and last_b_, also cache
    // last_z_hidden_ and last_z_hidden_pre_ for backward.
    void generate_weights(const Tensor& z);

    // Activation for the hidden hyper layer (Tanh by default).
    static double hyper_act(double x);
    static double hyper_act_deriv(double x);
};

#endif