#include "gmlp.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <stdexcept>

// ============================================================================
// gMLPBlock
// ============================================================================
//
// Forward:
//
//   z       = LN(x)                          # (S, d)
//   u       = fc_in(z)                      # (S, 2d)
//   u1, u2  = split last axis of u          # (S, d) each
//   v       = u1 * (W_spatial @ GELU(u2))   # (S, d)
//   v       = v + b_sgu                     # (S, d)
//   y       = fc_out(v)                     # (S, d)
//   out     = x + alpha * y
//
// Backward (with cached state):
//
//   grad_y      = alpha * grad_out
//   grad_alpha += sum(grad_out * y)
//   grad_v      = fc_out.backward(grad_y, lr)
//   grad_b_sgu  += sum_rows(grad_v)
//   grad_Wu2    = grad_v * u1
//   grad_u1     = grad_v * Wu2
//   grad_W_sp   = grad_Wu2 @ GELU(u2)^T
//   grad_gelu_u2 = W_spatial^T @ grad_Wu2
//   grad_u2     = grad_gelu_u2 * GELU'(u2)
//   grad_u      = concat([grad_u1, grad_u2])
//   grad_z      = fc_in.backward(grad_u, lr)
//   grad_x_ln   = norm_.backward(grad_z, lr)
//   grad_input  = grad_x_ln + grad_out
// ============================================================================

gMLPBlock::gMLPBlock(size_t dim, size_t seq_len, double alpha_init, bool sgu_bias)
    : dim_(dim), seq_len_(seq_len),
      norm_(dim),
      fc_in_(dim, 2 * dim),    // (2d, d)  — chunked in forward
      fc_out_(dim, dim),       // (d, d)
      W_spatial_(seq_len, seq_len),
      b_sgu_(1, dim),
      alpha_(1, 1),
      sgu_bias_(sgu_bias),
      grad_W_spatial_(seq_len, seq_len),
      grad_b_sgu_(1, dim),
      grad_alpha_(1, 1)
{
    // W_spatial: small init so block starts close to identity.
    // Use uniform in [-1/sqrt(seq_len), 1/sqrt(seq_len)] (Xavier-like).
    std::mt19937 gen(123);
    double bound = 1.0 / std::sqrt(static_cast<double>(seq_len));
    std::uniform_real_distribution<> dis(-bound, bound);
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t j = 0; j < seq_len; ++j)
            W_spatial_(i, j) = dis(gen);

    b_sgu_.fill(0.0);
    alpha_(0, 0) = alpha_init;

    grad_W_spatial_.fill(0.0);
    grad_b_sgu_.fill(0.0);
    grad_alpha_.fill(0.0);
}

Tensor gMLPBlock::forward(const Tensor& input) {
    // Cache residual input — we need it at the END of backward.
    last_residual_in_ = input.clone();
    last_input_ = input.clone();

    // Step 1: pre-norm
    Tensor z = norm_.forward(input);
    last_z_ = z;

    // Step 2: channel expansion: u = fc_in(z) -> (S, 2d)
    Tensor u = fc_in_.forward(z);
    last_u_ = u;

    if (u.cols != 2 * dim_) {
        throw std::runtime_error("gMLPBlock: fc_in output must have 2*dim columns");
    }
    if (u.rows != seq_len_) {
        throw std::runtime_error("gMLPBlock: input sequence length mismatch");
    }

    // Step 3: split last axis
    last_u1_ = Tensor(seq_len_, dim_);
    last_u2_ = Tensor(seq_len_, dim_);
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            last_u1_(i, j) = u(i, j);
            last_u2_(i, j) = u(i, dim_ + j);
        }
    }

    // Step 4: GELU(u2) then spatial mix
    last_gelu_u2_ = last_u2_.apply(GELU{});
    // Wu2 = W_spatial @ GELU(u2)  — (S, S) * (S, d) = (S, d)
    last_Wu2_ = W_spatial_ * last_gelu_u2_;

    // Step 5: elementwise gate
    last_v_ = last_u1_.hadamard(last_Wu2_);
    if (sgu_bias_) {
        for (size_t i = 0; i < seq_len_; ++i)
            for (size_t j = 0; j < dim_; ++j)
                last_v_(i, j) += b_sgu_(0, j);
    }

    // Step 6: channel projection
    Tensor y = fc_out_.forward(last_v_);
    last_y_ = y;

    // Step 7: residual with learnable scalar alpha
    Tensor output(seq_len_, dim_);
    double a = alpha_(0, 0);
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < dim_; ++j)
            output(i, j) = input(i, j) + a * y(i, j);

    return output;
}

Tensor gMLPBlock::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (S, d), shape (seq_len_, dim_)
    if (grad_output.rows != seq_len_ || grad_output.cols != dim_) {
        throw std::runtime_error("gMLPBlock::backward: grad_output shape mismatch");
    }

    double a = alpha_(0, 0);

    // Step 1: gradient through residual
    // out = x + alpha * y
    // grad_x (residual path) = grad_output
    // grad_y = alpha * grad_output
    // grad_alpha += sum(grad_output * y)   (scalar; alpha is shared across all (i,j))
    Tensor grad_y(grad_output.rows, grad_output.cols);
    double alpha_grad = 0.0;
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            grad_y(i, j) = a * grad_output(i, j);
            alpha_grad += grad_output(i, j) * last_y_(i, j);
        }
    }
    grad_alpha_(0, 0) += alpha_grad;

    // Step 2: backprop through fc_out
    Tensor grad_v = fc_out_.backward(grad_y, learning_rate);

    // Step 3: SGU bias gradient
    if (sgu_bias_) {
        for (size_t j = 0; j < dim_; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < seq_len_; ++i) sum += grad_v(i, j);
            grad_b_sgu_(0, j) += sum;
        }
    }

    // Step 4: elementwise gate gradient
    // v = u1 * Wu2
    // grad_u1 = grad_v * Wu2
    // grad_Wu2 = grad_v * u1
    Tensor grad_u1 = grad_v.hadamard(last_Wu2_);
    Tensor grad_Wu2 = grad_v.hadamard(last_u1_);

    // Step 5: spatial mixing backward
    // Wu2 = W_spatial @ GELU(u2)  (S, d) = (S, S) @ (S, d)
    // grad_W_spatial = grad_Wu2 @ GELU(u2)^T  (S, S) = (S, d) @ (d, S)
    // grad_gelu_u2   = W_spatial^T @ grad_Wu2 (S, d) = (S, S) @ (S, d)
    Tensor gelu_u2_T = last_gelu_u2_.transpose();  // (d, S)
    Tensor grad_W_step = grad_Wu2 * gelu_u2_T;     // (S, S)
    grad_W_spatial_ += grad_W_step;

    Tensor W_spatial_T = W_spatial_.transpose();   // (S, S)
    Tensor grad_gelu_u2 = W_spatial_T * grad_Wu2;  // (S, d)

    // Step 6: GELU'(u2) elementwise
    Tensor grad_u2 = last_u2_;
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            grad_u2(i, j) = grad_gelu_u2(i, j) * GELU{}.derivative(last_u2_(i, j));
        }
    }

    // Step 7: concatenate grad_u1 and grad_u2 along last axis -> (S, 2d)
    Tensor grad_u(seq_len_, 2 * dim_);
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            grad_u(i, j)           = grad_u1(i, j);
            grad_u(i, dim_ + j)    = grad_u2(i, j);
        }
    }

    // Step 8: backprop through fc_in
    Tensor grad_z = fc_in_.backward(grad_u, learning_rate);

    // Step 9: backprop through LN
    Tensor grad_x_ln = norm_.backward(grad_z, learning_rate);

    // Step 10: combine with residual gradient
    Tensor grad_input(seq_len_, dim_);
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            grad_input(i, j) = grad_x_ln(i, j) + grad_output(i, j);
        }
    }
    return grad_input;
}

void gMLPBlock::update_weights(double learning_rate) {
    // Update fc_in, fc_out
    fc_in_.update_weights(learning_rate);
    fc_out_.update_weights(learning_rate);
    // Update LN
    norm_.update_weights(learning_rate);

    // Update W_spatial
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            W_spatial_(i, j) -= learning_rate * grad_W_spatial_(i, j);
        }
    }
    // Update b_sgu
    if (sgu_bias_) {
        for (size_t j = 0; j < dim_; ++j) {
            b_sgu_(0, j) -= learning_rate * grad_b_sgu_(0, j);
        }
    }
    // Update alpha
    alpha_(0, 0) -= learning_rate * grad_alpha_(0, 0);
}

void gMLPBlock::zero_grad() {
    fc_in_.zero_grad();
    fc_out_.zero_grad();
    norm_.zero_grad();
    grad_W_spatial_.fill(0.0);
    grad_b_sgu_.fill(0.0);
    grad_alpha_.fill(0.0);
}

std::vector<Tensor*> gMLPBlock::parameters() {
    std::vector<Tensor*> p;
    auto fc_in_params = fc_in_.parameters();
    auto fc_out_params = fc_out_.parameters();
    auto norm_params = norm_.parameters();
    p.insert(p.end(), fc_in_params.begin(), fc_in_params.end());
    p.insert(p.end(), fc_out_params.begin(), fc_out_params.end());
    p.insert(p.end(), norm_params.begin(), norm_params.end());
    p.push_back(&W_spatial_);
    p.push_back(&b_sgu_);
    p.push_back(&alpha_);
    return p;
}

std::vector<Tensor*> gMLPBlock::gradients() {
    std::vector<Tensor*> g;
    auto fc_in_grads = fc_in_.gradients();
    auto fc_out_grads = fc_out_.gradients();
    auto norm_grads = norm_.gradients();
    g.insert(g.end(), fc_in_grads.begin(), fc_in_grads.end());
    g.insert(g.end(), fc_out_grads.begin(), fc_out_grads.end());
    g.insert(g.end(), norm_grads.begin(), norm_grads.end());
    g.push_back(&grad_W_spatial_);
    g.push_back(&grad_b_sgu_);
    g.push_back(&grad_alpha_);
    return g;
}

// ============================================================================
// gMLPModel
// ============================================================================

gMLPModel::gMLPModel(size_t dim, size_t seq_len, size_t out_features,
                     size_t num_blocks, double alpha_init, bool sgu_bias)
    : dim_(dim), seq_len_(seq_len), out_features_(out_features),
      num_blocks_(num_blocks), classifier_(dim, out_features)  // Dense(in=dim, out=out_features)
{
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_.emplace_back(dim, seq_len, alpha_init, sgu_bias);
    }
}

Tensor gMLPModel::forward(const Tensor& input) {
    last_input_ = input.clone();

    Tensor x = input;
    for (size_t i = 0; i < num_blocks_; ++i) {
        x = blocks_[i].forward(x);
    }
    // Classifier projects (S, d) -> (S, out_f)
    Tensor logits = classifier_.forward(x);
    return logits;
}

Tensor gMLPModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (S, out_features)
    // Backprop through classifier
    Tensor grad = classifier_.backward(grad_output, learning_rate);

    // Backprop through blocks in reverse order
    for (int i = static_cast<int>(num_blocks_) - 1; i >= 0; --i) {
        grad = blocks_[static_cast<size_t>(i)].backward(grad, learning_rate);
    }
    return grad;
}

void gMLPModel::update_weights(double learning_rate) {
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_[i].update_weights(learning_rate);
    }
    classifier_.update_weights(learning_rate);
}

void gMLPModel::zero_grad() {
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_[i].zero_grad();
    }
    classifier_.zero_grad();
}

std::vector<Tensor*> gMLPModel::parameters() {
    std::vector<Tensor*> p;
    for (size_t i = 0; i < num_blocks_; ++i) {
        auto block_params = blocks_[i].parameters();
        p.insert(p.end(), block_params.begin(), block_params.end());
    }
    auto clf_params = classifier_.parameters();
    p.insert(p.end(), clf_params.begin(), clf_params.end());
    return p;
}

std::vector<Tensor*> gMLPModel::gradients() {
    std::vector<Tensor*> g;
    for (size_t i = 0; i < num_blocks_; ++i) {
        auto block_grads = blocks_[i].gradients();
        g.insert(g.end(), block_grads.begin(), block_grads.end());
    }
    auto clf_grads = classifier_.gradients();
    g.insert(g.end(), clf_grads.begin(), clf_grads.end());
    return g;
}
