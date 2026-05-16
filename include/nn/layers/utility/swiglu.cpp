#include "swiglu.h"

template<typename A>
SwiGLU<A>::SwiGLU(size_t dim_input, size_t dim_hidden, bool use_bias)
    : w1_(dim_input, dim_hidden),
      w2_(dim_input, dim_hidden),
      dim_input_(dim_input),
      dim_hidden_(dim_hidden),
      use_bias_(use_bias),
      last_input_(0, 0),
      last_h1_(0, 0),
      last_h2_(0, 0),
      last_output_(0, 0)
{
    // w1_ and w2_ use Dense's default Xavier init; biases stay zero
}

template<typename A>
Tensor SwiGLU<A>::forward(const Tensor& input) {
    // input shape: (batch, dim_input_)
    last_input_ = input.clone();

    // h1 = W1 @ x  (batch, dim_hidden)
    Tensor h1 = w1_.forward(input);
    // h2 = W2 @ x  (batch, dim_hidden)
    Tensor h2 = w2_.forward(input);

    // Apply activation to h1: swish(h1) = x * sigmoid(x)
    A act;
    last_h1_ = h1.apply(act);

    // Gate h2 (no activation)
    last_h2_ = h2;

    // Output = swish(h1) * h2  [Hadamard product]
    last_output_ = last_h1_.hadamard(last_h2_);

    return last_output_;
}

template<typename A>
Tensor SwiGLU<A>::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output shape: (batch, dim_hidden_)
    //
    // Forward:
    //   h1_raw = W1 @ x,  h2 = W2 @ x
    //   h1 = act(h1_raw)  (activation, default Swish = SiLU)
    //   y = h1 * h2
    //
    // Backward (per sample):
    //   dL/dh2 = dL/dy * h1
    //   dL/dh1 = dL/dy * h2 * act'(h1_raw)
    //   dL/dW1 = (dL/dh1)^T @ x
    //   dL/dW2 = (dL/dh2)^T @ x
    //   dL/dx  = dL/dh1 @ W1 + dL/dh2 @ W2
    //
    // For Swish activation, derivative: swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
    // Computed inline below — A template parameter allows other activations too

    // grad_h2 = grad_output .* last_h1_   (Hadamard — gate side)
    // last_h1_ = act(W1 @ x) = swish(h1_raw)
    Tensor grad_h2 = grad_output.hadamard(last_h1_);

    // For swish'(h1_raw): we need sigmoid(h1_raw), not swish value
    // last_h2_ holds W2 @ x (not the activation), so we need raw h1.
    // But we only cached last_h1_ = act(h1_raw). We need raw h1 for sigmoid.
    // Recompute h1_raw = W1 @ last_input_ to get sigmoid.
    Tensor h1_raw = w1_.last_input;           // (batch, dim_hidden)
    Tensor sigmoid_h1_raw(h1_raw.rows, h1_raw.cols);
    for (size_t i = 0; i < h1_raw.data.size(); ++i) {
        double x = h1_raw.data[i];
        sigmoid_h1_raw.data[i] = 1.0 / (1.0 + std::exp(-x));
    }
    // swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
    Tensor swish_deriv(last_h1_.rows, last_h1_.cols);
    for (size_t i = 0; i < last_h1_.data.size(); ++i) {
        double swish_x = last_h1_.data[i];
        double sig_x = sigmoid_h1_raw.data[i];
        swish_deriv.data[i] = swish_x + sig_x * (1.0 - swish_x);
    }

    // grad_h1_raw = grad_output .* last_h2_
    Tensor grad_h1_raw = grad_output.hadamard(last_h2_);
    // grad_h1 = grad_h1_raw .* swish_deriv
    Tensor grad_h1 = grad_h1_raw.hadamard(swish_deriv);

    // dL/dW1 = (dL/dh1)^T @ last_input_  -> (dim_hidden, dim_input)
    Tensor grad_w1 = grad_h1.transpose() * last_input_;
    w1_.grad_weights += grad_w1;

    // Accumulate bias gradient for W1
    if (use_bias_) {
        Tensor grad_b1(1, grad_h1.cols);
        for (size_t j = 0; j < grad_h1.cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < grad_h1.rows; ++i)
                sum += grad_h1[i][j];
            grad_b1[0][j] = sum;
        }
        w1_.grad_bias += grad_b1;
    }

    // dL/dW2 = (dL/dh2)^T @ last_input_  -> (dim_hidden, dim_input)
    Tensor grad_w2 = grad_h2.transpose() * last_input_;
    w2_.grad_weights += grad_w2;

    // Accumulate bias gradient for W2
    if (use_bias_) {
        Tensor grad_b2(1, grad_h2.cols);
        for (size_t j = 0; j < grad_h2.cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < grad_h2.rows; ++i)
                sum += grad_h2[i][j];
            grad_b2[0][j] = sum;
        }
        w2_.grad_bias += grad_b2;
    }

    // dL/d_input = dL/dh1 @ W1 + dL/dh2 @ W2  -> (batch, dim_input)
    // grad_h1: (batch, dim_hidden)  @  W1.weights: (dim_hidden, dim_input) -> (batch, dim_input)
    Tensor grad_input_h1 = grad_h1 * w1_.weights;
    // grad_h2: (batch, dim_hidden)  @  W2.weights: (dim_hidden, dim_input) -> (batch, dim_input)
    Tensor grad_input_h2 = grad_h2 * w2_.weights;
    Tensor grad_input = grad_input_h1 + grad_input_h2;

    return grad_input;
}

template<typename A>
void SwiGLU<A>::update_weights(double learning_rate) {
    w1_.update_weights(learning_rate);
    w2_.update_weights(learning_rate);
}

template<typename A>
std::vector<Tensor*> SwiGLU<A>::parameters() {
    std::vector<Tensor*> result;
    result.push_back(&w1_.weights);
    if (use_bias_) result.push_back(&w1_.bias);
    result.push_back(&w2_.weights);
    if (use_bias_) result.push_back(&w2_.bias);
    return result;
}

template<typename A>
std::vector<Tensor*> SwiGLU<A>::gradients() {
    std::vector<Tensor*> result;
    result.push_back(&w1_.grad_weights);
    if (use_bias_) result.push_back(&w1_.grad_bias);
    result.push_back(&w2_.grad_weights);
    if (use_bias_) result.push_back(&w2_.grad_bias);
    return result;
}

template<typename A>
void SwiGLU<A>::zero_grad() {
    w1_.zero_grad();
    w2_.zero_grad();
}

// Explicit instantiations (can be expanded as needed)
template class SwiGLU<Swish>;
// template class SwiGLU<GELU>;  // uncomment if GELU variant is needed