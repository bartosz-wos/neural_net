#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../core/layer.h"
#include "../../activations/activations.h"

// SwiGLU: Swish-Gated Linear Unit
// From "GLU Variants Improve Transformer" (Shazeer 2020)
// FFN(x) = SiLU(W1 @ x) * (W2 @ x)
// W1, W2 are independent linear projections.
// The hidden dimension is dim_hidden (intermediate FFN size, typically 4 * dim_input).
//
// The Swish activation is: Swish(x) = x / (1 + exp(-x))
// The activation applied to W1 @ x is configurable via template parameter A
// (default: Swish / SiLU).
//
// Usage:
//   SwiGLU<> ffn(dim_input, dim_hidden);           // uses Swish (default)
//   SwiGLU<GELU> ffn_gelu(dim_input, dim_hidden); // uses GELU
//
template<typename A = Swish>
class SwiGLU : public Layer {
public:
    // dim_input:  input feature dimension
    // dim_hidden: intermediate FFN size (e.g. 4 * dim_input)
    // use_bias:   whether W1 and W2 have bias terms (default: true)
    SwiGLU(size_t dim_input, size_t dim_hidden, bool use_bias = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "SwiGLU"; }

    // Direct access to the two underlying Dense projections. The public
    // parameters()/gradients() accessors return a fresh vector of POINTERS,
    // which is all a caller needs for generic SGD; these return stable
    // addresses for callers that need to hold onto a specific tensor
    // (e.g. embedding a SwiGLU inside a composite Layer's parameter list).
    Tensor& w1_weights() { return w1_.weights; }
    Tensor& w1_bias()     { return w1_.bias; }
    Tensor& w2_weights() { return w2_.weights; }
    Tensor& w2_bias()     { return w2_.bias; }
    Tensor& grad_w1_weights() { return w1_.grad_weights; }
    Tensor& grad_w1_bias()     { return w1_.grad_bias; }
    Tensor& grad_w2_weights() { return w2_.grad_weights; }
    Tensor& grad_w2_bias()     { return w2_.grad_bias; }
    size_t dim_input()  const { return dim_input_; }
    size_t dim_hidden() const { return dim_hidden_; }

private:
    Dense w1_;                          // (dim_hidden, dim_input)
    Dense w2_;                          // (dim_hidden, dim_input)
    size_t dim_input_;
    size_t dim_hidden_;
    bool use_bias_;

    Tensor last_input_;   // cached: forward input, shape (batch, dim_input)
    Tensor last_h1_raw_;  // cached: W1 @ x + b1, PRE-activation, shape (batch, dim_hidden)
    Tensor last_h1_;      // cached: SiLU(W1 @ x), shape (batch, dim_hidden)
    Tensor last_h2_;      // cached: W2 @ x (gate), shape (batch, dim_hidden)
    Tensor last_output_;  // cached: forward output for debugging
};

#endif