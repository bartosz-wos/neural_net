#include "hypernetwork.h"
#include <cmath>
#include <stdexcept>

// ----- helpers (Tanh activation in the hidden hyper layer) -----
double HyperNetwork::hyper_act(double x) {
    return std::tanh(x);
}
double HyperNetwork::hyper_act_deriv(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

// ----- ctor -----
HyperNetwork::HyperNetwork(size_t context_dim,
                           size_t in_features,
                           size_t out_features,
                           size_t hyper_hidden,
                           bool   use_bias)
    : context_dim_(context_dim),
      in_features_(in_features),
      out_features_(out_features),
      hyper_hidden_(hyper_hidden),
      use_bias_(use_bias),
      flat_size_(in_features * out_features + (use_bias ? out_features : 0)),
      hyper1_(context_dim, hyper_hidden),   // Dense(in, out) = (hyper_hidden, context_dim)
      hyper2_(hyper_hidden, flat_size_) {   // (flat_size_, hyper_hidden)
    if (context_dim == 0 || in_features == 0 || out_features == 0 ||
        hyper_hidden == 0) {
        throw std::invalid_argument("HyperNetwork: all dimensions must be > 0");
    }

    // Initialise hyper weights with smaller-than-default scale to avoid
    // saturating tanh at init (which would zero out dL/dz for the first
    // several steps). Dense uses xavier by default; we re-init with a
    // small uniform.
    hyper1_.init_weights("uniform");
    hyper2_.init_weights("uniform");
}

// ----- generate weights from context z -----
void HyperNetwork::generate_weights(const Tensor& z) {
    if (z.rows != 1 || z.cols != context_dim_) {
        throw std::invalid_argument(
            "HyperNetwork: context must be shape (1, context_dim)");
    }

    // Hyper layer 1: z (1, context_dim) -> z_pre (1, hyper_hidden) -> z_hidden (1, hyper_hidden)
    Tensor z_pre  = hyper1_.forward(z);                       // (1, hyper_hidden)
    Tensor z_hid  = z_pre.apply(hyper_act);                   // (1, hyper_hidden)

    // Hyper layer 2: z_hidden (1, hyper_hidden) -> w_flat (1, flat_size_)
    Tensor w_flat = hyper2_.forward(z_hid);                   // (1, flat_size_)

    // Cache
    last_context_      = z;
    last_z_hidden_pre_ = z_pre;
    last_z_hidden_     = z_hid;

    // Reshape: first in*out entries -> W (out_features, in_features)
    //          next out entries (if use_bias_) -> b (1, out_features)
    last_W_ = Tensor(out_features_, in_features_);
    for (size_t i = 0; i < out_features_; ++i) {
        for (size_t j = 0; j < in_features_; ++j) {
            last_W_(i, j) = w_flat(0, i * in_features_ + j);
        }
    }
    if (use_bias_) {
        last_b_ = Tensor(1, out_features_);
        for (size_t j = 0; j < out_features_; ++j) {
            last_b_(0, j) = w_flat(0, in_features_ * out_features_ + j);
        }
    } else {
        last_b_ = Tensor(1, out_features_);
        last_b_.fill(0.0);
    }
}

// ----- set context -----
void HyperNetwork::set_context(const Tensor& z) {
    if (z.rows != 1 || z.cols != context_dim_) {
        throw std::invalid_argument(
            "HyperNetwork: context must be shape (1, context_dim)");
    }
    last_context_ = z;
}

// ----- forward -----
Tensor HyperNetwork::forward(const Tensor& input) {
    // Require context to have been set
    if (last_context_.rows == 0 || last_context_.cols == 0) {
        throw std::runtime_error(
            "HyperNetwork::forward: call set_context() before forward()");
    }
    return forward_with_context(input, last_context_);
}

Tensor HyperNetwork::forward_with_context(const Tensor& input, const Tensor& z) {
    if (input.cols != in_features_) {
        throw std::invalid_argument(
            "HyperNetwork: input cols must match in_features_");
    }

    generate_weights(z);
    last_input_ = input;

    // Standard dense: y = x @ W^T + b
    Tensor y = input * last_W_.transpose();
    if (use_bias_) {
        for (size_t i = 0; i < y.rows; ++i) {
            for (size_t j = 0; j < y.cols; ++j) {
                y(i, j) += last_b_(0, j);
            }
        }
    }
    return y;
}

// ----- backward -----
Tensor HyperNetwork::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output shape: (B, out_features)
    if (grad_output.cols != out_features_) {
        throw std::invalid_argument(
            "HyperNetwork::backward: grad_output cols must match out_features_");
    }
    if (last_input_.rows == 0) {
        throw std::runtime_error(
            "HyperNetwork::backward: forward() must be called before backward()");
    }

    const size_t B = grad_output.rows;

    // --- ∂L/∂W = grad_output^T @ last_input_  (shape out_features × in_features) ---
    Tensor grad_W = grad_output.transpose() * last_input_;

    // --- ∂L/∂b = sum over batch rows of grad_output (shape 1 × out_features) ---
    Tensor grad_b(1, out_features_);
    grad_b.fill(0.0);
    if (use_bias_) {
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                grad_b(0, j) += grad_output(i, j);
            }
        }
    }

    // --- ∂L/∂x = grad_output @ W   (shape B × in_features) ---
    Tensor grad_x = grad_output * last_W_;

    // --- pack grad_W and grad_b into a flat vector (1, flat_size_) matching w_flat layout ---
    Tensor grad_w_flat(1, flat_size_);
    for (size_t i = 0; i < out_features_; ++i) {
        for (size_t j = 0; j < in_features_; ++j) {
            grad_w_flat(0, i * in_features_ + j) = grad_W(i, j);
        }
    }
    if (use_bias_) {
        for (size_t j = 0; j < out_features_; ++j) {
            grad_w_flat(0, in_features_ * out_features_ + j) = grad_b(0, j);
        }
    }

    // --- Backward through hyper2_:  z_hidden (1, hyper_hidden) -> w_flat (1, flat_size_) ---
    // Standard dense backward:
    //   grad_z_hidden = grad_w_flat @ hyper2_.weights            (shape 1 × hyper_hidden)
    //   grad_hyper2.W += grad_w_flat^T @ z_hidden                (shape flat_size × hyper_hidden)
    //   grad_hyper2.b += sum_batch grad_w_flat                   (always 1 row, so it's grad_w_flat)
    Tensor grad_z_hidden = grad_w_flat * hyper2_.weights;       // (1, hyper_hidden)

    {
        // Accumulate hyper2.W grad:  grad_w_flat^T (flat_size × 1) @ z_hidden (1 × hyper_hidden)
        Tensor gw_outer = grad_w_flat.transpose() * last_z_hidden_;
        hyper2_.grad_weights += gw_outer;
        // Bias: hyper2.bias has shape (1, flat_size_) — grad is the row itself
        hyper2_.grad_bias += grad_w_flat;
    }

    // --- Backward through hyper1 activation (Tanh) ---
    // grad_z_pre = grad_z_hidden * tanh'(z_pre) = grad_z_hidden * (1 - z_hidden^2)
    Tensor grad_z_pre(1, hyper_hidden_);
    for (size_t j = 0; j < hyper_hidden_; ++j) {
        double zh = last_z_hidden_(0, j);
        grad_z_pre(0, j) = grad_z_hidden(0, j) * (1.0 - zh * zh);
    }

    // --- Backward through hyper1_:  z (1, context_dim) -> z_pre (1, hyper_hidden) ---
    // grad_z = grad_z_pre @ hyper1_.weights                       (1 × context_dim)
    // grad_hyper1.W += grad_z_pre^T @ z                           (hyper_hidden × context_dim)
    // grad_hyper1.b += grad_z_pre
    Tensor grad_z = grad_z_pre * hyper1_.weights;               // (1, context_dim)
    last_context_grad_ = grad_z;

    {
        Tensor gz_outer = grad_z_pre.transpose() * last_context_;
        hyper1_.grad_weights += gz_outer;
        hyper1_.grad_bias += grad_z_pre;
    }

    return grad_x;
}

// ----- update_weights (plain SGD; the optimizer wrapper handles the real update) -----
void HyperNetwork::update_weights(double learning_rate) {
    hyper1_.update_weights(learning_rate);
    hyper2_.update_weights(learning_rate);
}

// ----- parameter / gradient accessors -----
std::vector<Tensor*> HyperNetwork::parameters() {
    auto p1 = hyper1_.parameters();
    auto p2 = hyper2_.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}
std::vector<Tensor*> HyperNetwork::gradients() {
    auto g1 = hyper1_.gradients();
    auto g2 = hyper2_.gradients();
    g1.insert(g1.end(), g2.begin(), g2.end());
    return g1;
}
void HyperNetwork::zero_grad() {
    hyper1_.zero_grad();
    hyper2_.zero_grad();
    last_context_grad_ = Tensor(1, context_dim_);
    last_context_grad_.fill(0.0);
}