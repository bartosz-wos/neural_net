// Modern Hopfield Attention — Ramsauer et al. 2020
//   "Hopfield Networks is All You Need"
//
// See hopfield.h for the full mathematical formulation.
//
// This file implements:
//   * HopfieldAttention        — single modern Hopfield retrieval block.
//   * HopfieldBlock            — pre-LN → HopfieldAttention → residual →
//                                pre-LN → GELU FFN → residual.
//   * HopfieldModel            — stack of HopfieldBlocks + per-token
//                                classifier head.

#include "hopfield.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// d/dx softplus(x) = sigmoid(x) = 1 / (1 + exp(-x)).
static double sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
}

// ---------------------------------------------------------------------------
// HopfieldAttention
// ---------------------------------------------------------------------------

HopfieldAttention::HopfieldAttention(size_t d_model, size_t num_patterns)
    : d_model_(d_model),
      num_patterns_(num_patterns == 0 ? d_model : num_patterns),
      W_q(d_model, d_model),
      P_(num_patterns == 0 ? d_model : num_patterns, d_model),
      b_p_(1, num_patterns == 0 ? d_model : num_patterns),
      W_o(d_model, d_model),
      beta_log_(1, 1),
      grad_P_(num_patterns == 0 ? d_model : num_patterns, d_model),
      grad_b_p_(1, num_patterns == 0 ? d_model : num_patterns),
      grad_beta_log_(1, 1)
{
    if (d_model == 0) {
        throw std::invalid_argument("HopfieldAttention: d_model must be > 0");
    }
    // beta_log_ is initialized so that softplus(beta_log_) = 1 / sqrt(d_model),
    // matching the standard transformer attention scaling. We need
    // softplus(beta_log_) = 1/sqrt(d). softplus^{-1}(y) = log(exp(y) - 1).
    double target_beta = 1.0 / std::sqrt(static_cast<double>(d_model));
    if (target_beta > 0.0) {
        double e = std::exp(target_beta);
        beta_log_(0, 0) = std::log(e - 1.0);
    } else {
        beta_log_(0, 0) = 0.0;
    }

    // Initialize patterns P with small random values, like a Dense layer.
    // Xavier-ish init: N(0, sqrt(2 / (d_model + d_model)))
    std::mt19937 gen(42);
    double s = std::sqrt(2.0 / (2.0 * d_model));
    std::normal_distribution<> dis_p(0.0, s);
    for (size_t i = 0; i < P_.rows; ++i) {
        for (size_t j = 0; j < P_.cols; ++j) {
            P_(i, j) = dis_p(gen);
        }
    }
    for (size_t j = 0; j < b_p_.cols; ++j) {
        b_p_(0, j) = 0.0;
    }
    grad_beta_log_(0, 0) = 0.0;
}

Tensor HopfieldAttention::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("HopfieldAttention: input.cols must equal d_model");
    }
    last_input_ = input.clone();

    // (1) Q = X @ W_q^T  (Dense convention: W_q.weights is (d, d), W_q.forward
    //     computes y = X @ W_q.weights^T + b)
    last_Q_ = W_q.forward(input);

    // (2) scores = beta_pos * Q @ P^T  + b_p  — (n, m)  [P_ is (m, d)]
    size_t n = input.rows;
    double beta_pos = softplus(beta_log_(0, 0));
    last_beta_pos_ = beta_pos;
    Tensor scores(n, num_patterns_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < num_patterns_; ++s) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i) {
                acc += last_Q_(t, i) * P_(s, i);
            }
            scores(t, s) = beta_pos * acc + b_p_(0, s);
        }
    }

    // (3) row-softmax
    last_attn_ = Tensor(n, num_patterns_);
    for (size_t t = 0; t < n; ++t) {
        double m = scores(t, 0);
        for (size_t s = 1; s < num_patterns_; ++s) m = std::max(m, scores(t, s));
        double sum = 0.0;
        for (size_t s = 0; s < num_patterns_; ++s) {
            scores(t, s) = std::exp(scores(t, s) - m);
            sum += scores(t, s);
        }
        for (size_t s = 0; s < num_patterns_; ++s) {
            last_attn_(t, s) = scores(t, s) / sum;
        }
    }

    // (4) out_pre = attn @ P  — (n, d)
    last_out_pre_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t s = 0; s < num_patterns_; ++s) {
                acc += last_attn_(t, s) * P_(s, i);
            }
            last_out_pre_(t, i) = acc;
        }
    }

    // (5) out = out_pre @ W_o^T + b_o
    return W_o.forward(last_out_pre_);
}

Tensor HopfieldAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = last_input_.rows;
    W_q.zero_grad();
    W_o.zero_grad();
    grad_P_.fill(0.0);
    grad_b_p_.fill(0.0);
    grad_beta_log_(0, 0) = 0.0;

    // (1) Backward through W_o: y = x @ W_o^T + b_o
    //     grad_pre_o[t, k] = sum_j grad_output[t, j] * W_o.weights(j, k)
    //     grad_W_o[k, j]   += sum_t grad_output[t, k] * last_out_pre_[t, j]
    Tensor grad_pre_o(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_output(t, j) * W_o.weights(j, k);
            grad_pre_o(t, k) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_out_pre_(t, k);
            W_o.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // (2) Backward through attn @ P:  out_pre = attn @ P
    //     grad_attn[t, s] = sum_i grad_pre_o[t, i] * P[s, i]
    //     grad_P[s, i]   += sum_t grad_pre_o[t, i] * attn[t, s]
    Tensor grad_attn(n, num_patterns_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < num_patterns_; ++s) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i) acc += grad_pre_o(t, i) * P_(s, i);
            grad_attn(t, s) = acc;
        }
    }
    for (size_t s = 0; s < num_patterns_; ++s) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_pre_o(t, i) * last_attn_(t, s);
            grad_P_(s, i) += acc;
        }
    }

    // (3) Backward through softmax (row-wise):
    //     For row t, dL/dlogit[s] = attn[s] * (dL/dattn[s] - sum_s' attn[s'] * dL/dattn[s'])
    Tensor d_logits(n, num_patterns_);
    for (size_t t = 0; t < n; ++t) {
        double dot = 0.0;
        for (size_t s = 0; s < num_patterns_; ++s) {
            dot += last_attn_(t, s) * grad_attn(t, s);
        }
        for (size_t s = 0; s < num_patterns_; ++s) {
            d_logits(t, s) = last_attn_(t, s) * (grad_attn(t, s) - dot);
        }
    }

    // (4) Backward through scores = beta_pos * Q @ P^T + b_p
    //     dL/dQ[t, i]     = beta_pos * sum_s d_logits[t, s] * P[s, i]
    //     dL/dP[s, i]    += beta_pos * sum_t d_logits[t, s] * Q[t, i]
    //     dL/dbeta_pos   = sum_{t,s} d_logits[t, s] * (Q @ P^T)[t, s]
    //     dL/db_p[s]      = sum_t d_logits[t, s]
    double beta_pos = last_beta_pos_;
    Tensor dQ(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t s = 0; s < num_patterns_; ++s) {
                acc += d_logits(t, s) * P_(s, i);
            }
            dQ(t, i) = beta_pos * acc;
        }
    }
    for (size_t s = 0; s < num_patterns_; ++s) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) {
                acc += d_logits(t, s) * last_Q_(t, i);
            }
            grad_P_(s, i) += beta_pos * acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_logits(t, s);
        grad_b_p_(0, s) += b_acc;
    }
    // dL/dbeta_pos_: sum over (t, s) of d_logits[t, s] * (Q P^T)[t, s]
    double d_beta_pos = 0.0;
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < num_patterns_; ++s) {
            double qp = 0.0;
            for (size_t i = 0; i < d_model_; ++i) {
                qp += last_Q_(t, i) * P_(s, i);
            }
            d_beta_pos += d_logits(t, s) * qp;
        }
    }
    // dL/dbeta_log_ = dL/dbeta_pos_ * d(softplus)/dx = dL/dbeta_pos_ * sigmoid(beta_log_)
    grad_beta_log_(0, 0) = d_beta_pos * sigmoid(beta_log_(0, 0));

    // (5) Backward through W_q:  Q = X @ W_q^T + b_q  (Dense convention)
    //     grad_x[t, i] = sum_j dQ[t, j] * W_q.weights(j, i)
    //     grad_W_q[k, j] += sum_t dQ[t, k] * last_input_[t, j]
    //                       (here k is the OUT dim of W_q, j is the IN dim)
    Tensor grad_x(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += dQ(t, j) * W_q.weights(j, i);
            grad_x(t, i) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {  // j = OUT dim of W_q
        for (size_t k = 0; k < d_model_; ++k) {  // k = IN dim of W_q
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += dQ(t, j) * last_input_(t, k);
            W_q.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += dQ(t, j);
        W_q.grad_bias(0, j) += b_acc;
    }

    return grad_x;
}

void HopfieldAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < P_.rows; ++i) {
        for (size_t j = 0; j < P_.cols; ++j) {
            P_(i, j) -= learning_rate * grad_P_(i, j);
        }
    }
    for (size_t j = 0; j < b_p_.cols; ++j) {
        b_p_(0, j) -= learning_rate * grad_b_p_(0, j);
    }
    beta_log_(0, 0) -= learning_rate * grad_beta_log_(0, 0);
}

void HopfieldAttention::zero_grad() {
    W_q.zero_grad();
    W_o.zero_grad();
    grad_P_.fill(0.0);
    grad_b_p_.fill(0.0);
    grad_beta_log_(0, 0) = 0.0;
}

std::vector<Tensor*> HopfieldAttention::parameters() {
    return {&W_q.weights, &W_q.bias,
            &P_, &b_p_,
            &W_o.weights, &W_o.bias,
            &beta_log_};
}

std::vector<Tensor*> HopfieldAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias,
            &grad_P_, &grad_b_p_,
            &W_o.grad_weights, &W_o.grad_bias,
            &grad_beta_log_};
}

// ---------------------------------------------------------------------------
// HopfieldBlock
// ---------------------------------------------------------------------------

static inline double block_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
static inline double block_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

HopfieldBlock::HopfieldBlock(size_t d_model, size_t num_patterns, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, num_patterns),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model)
{
    if (d_model == 0) {
        throw std::invalid_argument("HopfieldBlock: d_model must be > 0");
    }
}

Tensor HopfieldBlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    // Pre-LN → HopfieldAttention → residual
    last_z1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_z1_);
    last_res1_ = input + last_attn_out_;

    // Pre-LN → FFN → residual
    last_z2_ = ln2_.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    last_h_act_ = Tensor(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.rows; ++i) {
        for (size_t j = 0; j < last_h_pre_.cols; ++j) {
            last_h_act_(i, j) = block_gelu(last_h_pre_(i, j));
        }
    }
    last_ffn_out_ = ffn_fc2_.forward(last_h_act_);
    return last_res1_ + last_ffn_out_;
}

Tensor HopfieldBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();

    size_t n = grad_output.rows;

    // (1) ffn_fc2 backward
    Tensor d_h_act(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) acc += grad_output(t, j) * ffn_fc2_.weights(j, k);
            d_h_act(t, k) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_h_act_(t, k);
            ffn_fc2_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        ffn_fc2_.grad_bias(0, j) += b_acc;
    }

    // (2) GELU backward on h_pre
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h_act(t, k) * block_gelu_deriv(last_h_pre_(t, k));
        }
    }

    // (3) ffn_fc1 backward
    Tensor d_z2(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < ffn_dim_; ++j) acc += d_h_pre(t, j) * ffn_fc1_.weights(j, k);
            d_z2(t, k) = acc;
        }
    }
    for (size_t j = 0; j < ffn_dim_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += d_h_pre(t, j) * last_z2_(t, k);
            ffn_fc1_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_h_pre(t, j);
        ffn_fc1_.grad_bias(0, j) += b_acc;
    }

    // (4) ln2 backward, then add the residual gradient
    Tensor d_res1_from_ffn = ln2_.backward(d_z2, 0.0);
    Tensor d_res1 = d_res1_from_ffn + grad_output;

    // (5) Direct residual contribution to d_x
    Tensor d_x = d_res1.clone();

    // (6) HopfieldAttention backward, then ln1.backward
    Tensor d_z1 = attn_.backward(d_res1, 0.0);
    Tensor d_x_from_ln1 = ln1_.backward(d_z1, 0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            d_x(t, k) += d_x_from_ln1(t, k);
        }
    }
    return d_x;
}

void HopfieldBlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void HopfieldBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> HopfieldBlock::parameters() {
    auto p = attn_.parameters();
    auto a = ln1_.parameters();
    auto b = ln2_.parameters();
    auto c = ffn_fc1_.parameters();
    auto d = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> HopfieldBlock::gradients() {
    auto g = attn_.gradients();
    auto a = ln1_.gradients();
    auto b = ln2_.gradients();
    auto c = ffn_fc1_.gradients();
    auto d = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}

// ---------------------------------------------------------------------------
// HopfieldModel
// ---------------------------------------------------------------------------

HopfieldModel::HopfieldModel(size_t d_model, size_t out_features,
                             size_t num_blocks, size_t num_patterns, size_t ffn_dim)
    : d_model_(d_model),
      out_features_(out_features),
      num_blocks_(num_blocks),
      classifier_(d_model, out_features)
{
    if (d_model == 0 || out_features == 0 || num_blocks == 0) {
        throw std::invalid_argument("HopfieldModel: d_model, out_features, num_blocks must be > 0");
    }
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.push_back(std::make_unique<HopfieldBlock>(d_model, num_patterns, ffn_dim));
    }
}

Tensor HopfieldModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    last_block_output_ = x.clone();
    return classifier_.forward(x);
}

Tensor HopfieldModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = grad_output.rows;
    classifier_.zero_grad();

    Tensor grad_x(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < out_features_; ++j) acc += grad_output(t, j) * classifier_.weights(j, k);
            grad_x(t, k) = acc;
        }
    }
    for (size_t j = 0; j < out_features_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_block_output_(t, k);
            classifier_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        classifier_.grad_bias(0, j) += b_acc;
    }

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_x = (*it)->backward(grad_x, 0.0);
    }
    return grad_x;
}

void HopfieldModel::update_weights(double learning_rate) {
    for (auto& block : blocks_) block->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void HopfieldModel::zero_grad() {
    for (auto& block : blocks_) block->zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> HopfieldModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& block : blocks_) {
        auto bp = block->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> HopfieldModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& block : blocks_) {
        auto bg = block->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
