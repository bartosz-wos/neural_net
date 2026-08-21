#include "hyper_connection.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

// ============================================================================
// HyperConnection
// ============================================================================

HyperConnection::HyperConnection(size_t d_model, Layer* inner)
    : d_model_(d_model), inner_(inner) {
    if (d_model_ == 0) {
        throw std::invalid_argument("HyperConnection: d_model must be > 0");
    }
    if (!inner_) {
        throw std::invalid_argument("HyperConnection: inner layer must not be null");
    }
    // α_log initialized so sigmoid(α_log) ≈ 1: α_log = 10 → sigmoid ≈ 0.99995
    // β_log initialized so sigmoid(β_log) ≈ 0: β_log = -10 → sigmoid ≈ 4.5e-5
    // → at init the layer behaves as out ≈ x (standard residual).
    alpha_log_ = Tensor(1, d_model_);
    beta_log_ = Tensor(1, d_model_);
    grad_alpha_log_ = Tensor(1, d_model_);
    grad_beta_log_ = Tensor(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        alpha_log_[0][j] = 10.0;
        beta_log_[0][j] = -10.0;
    }
    last_alpha_ = Tensor(1, d_model_);
    last_beta_ = Tensor(1, d_model_);
}

Tensor HyperConnection::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("HyperConnection::forward: input.cols != d_model_");
    }
    last_input_ = input;

    // Compute sigmoid of log params
    for (size_t j = 0; j < d_model_; ++j) {
        last_alpha_[0][j] = 1.0 / (1.0 + std::exp(-alpha_log_[0][j]));
        last_beta_[0][j] = 1.0 / (1.0 + std::exp(-beta_log_[0][j]));
    }

    // Run inner sub-layer
    Tensor sub_out = inner_->forward(input);
    if (sub_out.rows != input.rows || sub_out.cols != d_model_) {
        throw std::runtime_error("HyperConnection::forward: inner sub-layer output shape mismatch");
    }
    last_sub_out_ = sub_out;

    // Apply gated merge: out = α ⊙ x + β ⊙ sub_out
    Tensor out(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            out[i][j] = last_alpha_[0][j] * input[i][j] + last_beta_[0][j] * sub_out[i][j];
        }
    }
    return out;
}

Tensor HyperConnection::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("HyperConnection::backward: grad_output.cols != d_model_");
    }
    const size_t n = grad_output.rows;

    // 1) d sub_out = β ⊙ d_out
    Tensor d_sub_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_sub_out[i][j] = last_beta_[0][j] * grad_output[i][j];
        }
    }

    // 2) d x from inner
    Tensor d_x_inner = inner_->backward(d_sub_out, learning_rate);

    // 3) d x from residual path = α ⊙ d_out
    Tensor d_x_residual(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_x_residual[i][j] = last_alpha_[0][j] * grad_output[i][j];
        }
    }

    // 4) total d x
    Tensor d_x(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_x[i][j] = d_x_residual[i][j] + d_x_inner[i][j];
        }
    }

    // 5) d α_log[0, j] = α(1-α) · Σ_t d_out[t, j] · x[t, j]
    // 6) d β_log[0, j] = β(1-β) · Σ_t d_out[t, j] · sub_out[t, j]
    for (size_t j = 0; j < d_model_; ++j) {
        double acc_a = 0.0;
        double acc_b = 0.0;
        for (size_t t = 0; t < n; ++t) {
            acc_a += grad_output[t][j] * last_input_[t][j];
            acc_b += grad_output[t][j] * last_sub_out_[t][j];
        }
        double alpha = last_alpha_[0][j];
        double beta = last_beta_[0][j];
        grad_alpha_log_[0][j] = alpha * (1.0 - alpha) * acc_a;
        grad_beta_log_[0][j] = beta * (1.0 - beta) * acc_b;
    }

    return d_x;
}

void HyperConnection::update_weights(double learning_rate) {
    // Apply SGD-style update to α_log, β_log (the parameters of this layer)
    for (size_t j = 0; j < d_model_; ++j) {
        alpha_log_[0][j] -= learning_rate * grad_alpha_log_[0][j];
        beta_log_[0][j] -= learning_rate * grad_beta_log_[0][j];
    }
    // Inner sub-layer uses its own update logic
    inner_->update_weights(learning_rate);
}

std::vector<Tensor*> HyperConnection::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&alpha_log_);
    params.push_back(&beta_log_);
    // Inner layer parameters
    std::vector<Tensor*> inner_p = inner_->parameters();
    for (Tensor* p : inner_p) params.push_back(p);
    return params;
}

std::vector<Tensor*> HyperConnection::gradients() {
    std::vector<Tensor*> grads;
    grads.push_back(&grad_alpha_log_);
    grads.push_back(&grad_beta_log_);
    std::vector<Tensor*> inner_g = inner_->gradients();
    for (Tensor* g : inner_g) grads.push_back(g);
    return grads;
}

void HyperConnection::zero_grad() {
    for (size_t i = 0; i < grad_alpha_log_.data.size(); ++i) grad_alpha_log_.data[i] = 0.0;
    for (size_t i = 0; i < grad_beta_log_.data.size(); ++i) grad_beta_log_.data[i] = 0.0;
    inner_->zero_grad();
}

// ============================================================================
// HyperConnectionBlock
// ============================================================================

HyperConnectionBlock::HyperConnectionBlock(size_t d_model, size_t ffn_dim)
    : d_model_(d_model), ffn_dim_(ffn_dim) {
    if (d_model_ == 0) throw std::invalid_argument("HyperConnectionBlock: d_model must be > 0");
    if (ffn_dim_ == 0) throw std::invalid_argument("HyperConnectionBlock: ffn_dim must be > 0");
    ln_ = std::make_unique<LayerNorm>(d_model_);
    fc1_ = std::make_unique<Dense>(d_model_, ffn_dim_);
    fc2_ = std::make_unique<Dense>(ffn_dim_, d_model_);

    // α, β learnable residual scalers
    alpha_log_ = Tensor(1, d_model_);
    beta_log_ = Tensor(1, d_model_);
    grad_alpha_log_ = Tensor(1, d_model_);
    grad_beta_log_ = Tensor(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        alpha_log_[0][j] = 10.0;   // sigmoid ≈ 1, alpha ≈ 1
        beta_log_[0][j] = -10.0;   // sigmoid ≈ 0, beta ≈ 0
    }

    last_input_ = Tensor(0, d_model_);
    last_ln_out_ = Tensor(0, d_model_);
    last_ffn_hidden_ = Tensor(0, ffn_dim_);
    last_sub_out_ = Tensor(0, d_model_);
    last_alpha_ = Tensor(1, d_model_);
    last_beta_ = Tensor(1, d_model_);
}

Tensor HyperConnectionBlock::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("HyperConnectionBlock::forward: input.cols != d_model_");
    }
    last_input_ = input;

    // 1) Pre-norm
    Tensor ln_out = ln_->forward(input);
    last_ln_out_ = ln_out;

    // 2) fc1: d_model -> ffn_dim
    Tensor fc1_out = fc1_->forward(ln_out);

    // 3) GELU
    Tensor gelu_out(fc1_out.rows, fc1_out.cols);
    for (size_t i = 0; i < fc1_out.data.size(); ++i) {
        double x = fc1_out.data[i];
        // GELU exact form: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_out.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    last_ffn_hidden_ = gelu_out;

    // 4) fc2: ffn_dim -> d_model
    Tensor sub_out = fc2_->forward(gelu_out);
    last_sub_out_ = sub_out;

    // 5) α, β
    for (size_t j = 0; j < d_model_; ++j) {
        last_alpha_[0][j] = 1.0 / (1.0 + std::exp(-alpha_log_[0][j]));
        last_beta_[0][j] = 1.0 / (1.0 + std::exp(-beta_log_[0][j]));
    }

    // 6) gated merge
    Tensor out(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            out[i][j] = last_alpha_[0][j] * input[i][j] + last_beta_[0][j] * sub_out[i][j];
        }
    }
    return out;
}

Tensor HyperConnectionBlock::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("HyperConnectionBlock::backward: grad_output.cols != d_model_");
    }
    const size_t n = grad_output.rows;

    // 1) d sub_out = β ⊙ d_out
    Tensor d_sub_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_sub_out[i][j] = last_beta_[0][j] * grad_output[i][j];
        }
    }

    // 2) d x from residual = α ⊙ d_out
    Tensor d_x_residual(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_x_residual[i][j] = last_alpha_[0][j] * grad_output[i][j];
        }
    }

    // 3) Backward through fc2: d_hidden = fc2.backward(d_sub_out)
    Tensor d_hidden = fc2_->backward(d_sub_out, learning_rate);

    // 4) Backward through GELU
    Tensor d_gelu(d_hidden.rows, d_hidden.cols);
    for (size_t i = 0; i < d_hidden.data.size(); ++i) {
        double x = last_ffn_hidden_.data[i];
        // GELU derivative (exact): 0.5 * (1 + erf(x/sqrt(2))) + x * phi(x)
        // where phi(x) = (1/sqrt(2π)) * exp(-x²/2)
        double phi = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
        double gelu_deriv = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))) + x * phi;
        d_gelu.data[i] = d_hidden.data[i] * gelu_deriv;
    }

    // 5) Backward through fc1: d_ln_out = fc1.backward(d_gelu)
    Tensor d_ln_out = fc1_->backward(d_gelu, learning_rate);

    // 6) Backward through LN: d_x_inner = ln.backward(d_ln_out)
    Tensor d_x_inner = ln_->backward(d_ln_out, learning_rate);

    // 7) Total d_x
    Tensor d_x(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_x[i][j] = d_x_residual[i][j] + d_x_inner[i][j];
        }
    }

    // 8) d α_log, d β_log (the same as HyperConnection)
    for (size_t j = 0; j < d_model_; ++j) {
        double acc_a = 0.0;
        double acc_b = 0.0;
        for (size_t t = 0; t < n; ++t) {
            acc_a += grad_output[t][j] * last_input_[t][j];
            acc_b += grad_output[t][j] * last_sub_out_[t][j];
        }
        double alpha = last_alpha_[0][j];
        double beta = last_beta_[0][j];
        // grad_alpha_log_ = α(1-α) · Σ_t d_out · x
        grad_alpha_log_[0][j] = alpha * (1.0 - alpha) * acc_a;
        grad_beta_log_[0][j] = beta * (1.0 - beta) * acc_b;
    }

    return d_x;
}

void HyperConnectionBlock::update_weights(double learning_rate) {
    // Update α, β
    for (size_t j = 0; j < d_model_; ++j) {
        alpha_log_[0][j] -= learning_rate * grad_alpha_log_[0][j];
        beta_log_[0][j] -= learning_rate * grad_beta_log_[0][j];
    }
    ln_->update_weights(learning_rate);
    fc1_->update_weights(learning_rate);
    fc2_->update_weights(learning_rate);
}

std::vector<Tensor*> HyperConnectionBlock::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&alpha_log_);
    p.push_back(&beta_log_);
    for (Tensor* t : ln_->parameters()) p.push_back(t);
    for (Tensor* t : fc1_->parameters()) p.push_back(t);
    for (Tensor* t : fc2_->parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> HyperConnectionBlock::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_alpha_log_);
    g.push_back(&grad_beta_log_);
    for (Tensor* t : ln_->gradients()) g.push_back(t);
    for (Tensor* t : fc1_->gradients()) g.push_back(t);
    for (Tensor* t : fc2_->gradients()) g.push_back(t);
    return g;
}

void HyperConnectionBlock::zero_grad() {
    for (size_t i = 0; i < grad_alpha_log_.data.size(); ++i) grad_alpha_log_.data[i] = 0.0;
    for (size_t i = 0; i < grad_beta_log_.data.size(); ++i) grad_beta_log_.data[i] = 0.0;
    ln_->zero_grad();
    fc1_->zero_grad();
    fc2_->zero_grad();
}

// ============================================================================
// HyperConnectionModel
// ============================================================================

HyperConnectionModel::HyperConnectionModel(size_t input_dim, size_t d_model, size_t output_dim,
                                           size_t num_blocks, size_t ffn_dim)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_blocks_(num_blocks), ffn_dim_(ffn_dim) {
    if (input_dim_ == 0 || d_model_ == 0 || output_dim_ == 0)
        throw std::invalid_argument("HyperConnectionModel: dims must be > 0");
    if (num_blocks_ == 0) throw std::invalid_argument("HyperConnectionModel: num_blocks must be > 0");
    input_proj_ = std::make_unique<Dense>(input_dim_, d_model_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_.push_back(std::make_unique<HyperConnectionBlock>(d_model_, ffn_dim_));
    }
    classifier_ = std::make_unique<Dense>(d_model_, output_dim_);

    last_input_ = Tensor(0, input_dim_);
    last_proj_ = Tensor(0, d_model_);
    block_outputs_.resize(num_blocks_ + 1);
}

Tensor HyperConnectionModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor proj = input_proj_->forward(input);
    last_proj_ = proj;

    block_outputs_[0] = proj;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        block_outputs_[i + 1] = blocks_[i]->forward(block_outputs_[i]);
    }
    Tensor out = classifier_->forward(block_outputs_[blocks_.size()]);
    return out;
}

Tensor HyperConnectionModel::backward(const Tensor& grad_output, double learning_rate) {
    // Classifier backward
    Tensor d_block_last = classifier_->backward(grad_output, learning_rate);

    // Blocks backward (reverse order)
    Tensor d_cur = d_block_last;
    for (size_t i = blocks_.size(); i > 0; --i) {
        d_cur = blocks_[i - 1]->backward(d_cur, learning_rate);
    }

    // Input projection backward
    Tensor d_input = input_proj_->backward(d_cur, learning_rate);
    return d_input;
}

void HyperConnectionModel::update_weights(double learning_rate) {
    input_proj_->update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    classifier_->update_weights(learning_rate);
}

std::vector<Tensor*> HyperConnectionModel::parameters() {
    std::vector<Tensor*> p;
    for (Tensor* t : input_proj_->parameters()) p.push_back(t);
    for (auto& b : blocks_) {
        for (Tensor* t : b->parameters()) p.push_back(t);
    }
    for (Tensor* t : classifier_->parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> HyperConnectionModel::gradients() {
    std::vector<Tensor*> g;
    for (Tensor* t : input_proj_->gradients()) g.push_back(t);
    for (auto& b : blocks_) {
        for (Tensor* t : b->gradients()) g.push_back(t);
    }
    for (Tensor* t : classifier_->gradients()) g.push_back(t);
    return g;
}

void HyperConnectionModel::zero_grad() {
    input_proj_->zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    classifier_->zero_grad();
}
