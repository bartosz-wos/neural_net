#include "hyper_mixing.h"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// GELU (tanh approximation, matches activations.h::GELU).
// Used here directly so we can compute the per-token pre-activation backward
// without going through the Activation functor template machinery.
//
// IMPORTANT: this matches activations.h::GELU bit-exactly — including the
// [-4, 4] input clamp for CDF and the use of the unclamped x as the linear
// factor. Any drift between this and activations.h causes the analytical
// backward to disagree with the forward pass.
inline double gelu(double x) {
    double x_clamped = std::max(-4.0, std::min(4.0, x));
    const double a = std::sqrt(2.0 / M_PI);
    double cdf = 0.5 * (1.0 + std::tanh(a * (x_clamped + 0.044715 * x_clamped * x_clamped * x_clamped)));
    return x * cdf;
}

// d/dx GELU(x) (tanh-approx derivative). Matches activations.h::GELU::derivative.
inline double gelu_deriv(double x) {
    double x_clamped = std::max(-4.0, std::min(4.0, x));
    const double a = std::sqrt(2.0 / M_PI);
    double arg = a * (x_clamped + 0.044715 * x_clamped * x_clamped * x_clamped);
    double tanh_val = std::tanh(arg);
    double tanh_sq = tanh_val * tanh_val;
    double cdf = 0.5 * (1.0 + tanh_val);
    double pdf = 0.5 * a * (1.0 - tanh_sq) * (1.0 + 3.0 * 0.044715 * x_clamped * x_clamped);
    // FIX (matches activations.h Bug 8 fix): use x_clamped in x*pdf term.
    return cdf + x_clamped * pdf;
}

// Numerically stable row-wise softmax with max-subtraction.
// Mirrors the pattern from linformer.cpp.
Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j) {
            if (x[i][j] > row_max) row_max = x[i][j];
        }
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j) {
            result[i][j] *= inv;
        }
    }
    return result;
}

// Element-wise addition: c = a + b.
void tensor_add(const Tensor& a, const Tensor& b, Tensor& c) {
    for (size_t i = 0; i < a.data.size(); ++i) c.data[i] = a.data[i] + b.data[i];
}

}  // namespace

// ============================================================================
// HyperMixingLayer
// ============================================================================

HyperMixingLayer::HyperMixingLayer(size_t d_model, size_t mlp_hidden, size_t num_tokens)
    : d_model_(d_model),
      mlp_hidden_(mlp_hidden),
      num_tokens_(num_tokens),
      W_h_(d_model, num_tokens),
      W_1_(d_model, mlp_hidden),
      W_2_(mlp_hidden, d_model),
      W_3_(d_model, mlp_hidden),
      W_4_(mlp_hidden, d_model),
      final_ln_(d_model) {
    if (d_model == 0)
        throw std::invalid_argument("HyperMixingLayer: d_model must be > 0");
    if (mlp_hidden == 0)
        throw std::invalid_argument("HyperMixingLayer: mlp_hidden must be > 0");
    if (num_tokens == 0)
        throw std::invalid_argument("HyperMixingLayer: num_tokens must be > 0");
}

Tensor HyperMixingLayer::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("HyperMixingLayer::forward: input feature dim mismatch");
    if (input.rows != num_tokens_)
        throw std::invalid_argument(
            "HyperMixingLayer::forward: input.rows (" +
            std::to_string(input.rows) +
            ") must equal num_tokens (" + std::to_string(num_tokens_) +
            ") — the hypernetwork's W_h output is fixed at construction");
    const size_t T = input.rows;
    last_input = input;

    // --- Hypernetwork: R_logits = W_h · x  (T, num_tokens) ---
    last_R_logits = W_h_.forward(input);                 // (T, num_tokens)
    last_R = row_softmax(last_R_logits);                 // (T, num_tokens)

    // --- Channel MLP (per-token): U = W_2 · GELU(W_1 · x)  (T, d_model) ---
    // W_1: (d_model, mlp_hidden); W_1.forward(input) → (T, mlp_hidden).
    // GELU applied elementwise.
    // W_2: (mlp_hidden, d_model); W_2.forward(GELU(...)) → (T, d_model).
    last_U_pre = W_1_.forward(input);                    // (T, mlp_hidden)
    Tensor u_act(T, mlp_hidden_);
    for (size_t i = 0; i < last_U_pre.data.size(); ++i)
        u_act.data[i] = gelu(last_U_pre.data[i]);
    last_U = W_2_.forward(u_act);                        // (T, d_model)

    // --- Token mixing: H_T = R @ U  ---
    // H_T[t, c] = sum_s R[t, s] * U[s, c]
    last_H_T = last_R * last_U;                          // (T, d_model)

    // --- Channel mixing: Z = W_4 · GELU(W_3 · H_T) ---
    last_Z_pre = W_3_.forward(last_H_T);                 // (T, mlp_hidden)
    Tensor z_gelu(T, mlp_hidden_);
    for (size_t i = 0; i < last_Z_pre.data.size(); ++i)
        z_gelu.data[i] = gelu(last_Z_pre.data[i]);
    last_Z = W_4_.forward(z_gelu);                       // (T, d_model)

    // --- Residual + LayerNorm: out = LayerNorm(Z + x) ---
    last_residual = Tensor(T, d_model_);
    tensor_add(last_Z, last_input, last_residual);
    return final_ln_.forward(last_residual);
}

Tensor HyperMixingLayer::backward(const Tensor& grad_output, double learning_rate) {
    if (last_input.data.empty())
        throw std::logic_error("HyperMixingLayer::backward: forward was not called");
    if (grad_output.cols != d_model_)
        throw std::invalid_argument("HyperMixingLayer::backward: grad_output dim mismatch");
    const size_t T = last_input.rows;

    // --- (1) LayerNorm backward: grad_residual = d/d(residual) L ---
    Tensor grad_residual = final_ln_.backward(grad_output, learning_rate);   // (T, d_model)

    // The residual has TWO contributors: Z (from W_4 · GELU(W_3 · H_T)) and x.
    Tensor grad_x = Tensor(T, d_model_);
    std::fill(grad_x.data.begin(), grad_x.data.end(), 0.0);
    for (size_t i = 0; i < grad_residual.data.size(); ++i)
        grad_x.data[i] += grad_residual.data[i];                  // residual path

    // --- (2) Channel mixing backward (Z path): grad_Z_pre ---
    // Forward: Z_pre = W_3 · H_T,  Z_gelu = GELU(Z_pre),  Z = W_4 · Z_gelu.
    // backward through W_4 returns grad of (T, mlp_hidden) input = grad of Z_gelu.
    Tensor grad_z_gelu = W_4_.backward(grad_residual, learning_rate);  // (T, mlp_hidden)
    // GELU backward through Z_pre.
    Tensor grad_z_pre(T, mlp_hidden_);
    for (size_t i = 0; i < grad_z_pre.data.size(); ++i)
        grad_z_pre.data[i] = grad_z_gelu.data[i] * gelu_deriv(last_Z_pre.data[i]);
    // Backward through W_3 — returns grad_H_T = grad_z_pre · W_3 (T, d_model)
    // and accumulates grad_W_3, grad_b_3 internally.
    Tensor grad_H_T = W_3_.backward(grad_z_pre, learning_rate);       // (T, d_model)

    // --- (3) Token-mixing backward ---
    // Forward: H_T = R @ U  (R: (T,T), U: (T,d)).
    // grad_R = grad_H_T @ U^T  (T, T)
    // grad_U = R^T @ grad_H_T  (T, d_model)
    Tensor U_T = last_U.transpose();                                  // (d_model, T)
    Tensor grad_R = grad_H_T * U_T;                                   // (T, T)

    Tensor R_T = last_R.transpose();                                  // (T, T)
    Tensor grad_U = R_T * grad_H_T;                                    // (T, d_model)

    // --- (4) Channel MLP backward (U path): grad_U_pre ---
    // Forward: U_pre = W_1 · x,  U_act = GELU(U_pre),  U = W_2 · U_act.
    // Backward (reverse): grad_U (T, d_model) -> grad_U_act (T, mlp_hidden)
    // via W_2.backward, then grad_U_pre (T, mlp_hidden) via GELU', then
    // grad_x (T, d_model) via W_1.backward (also accumulates grad_W_1).
    Tensor grad_U_act = W_2_.backward(grad_U, learning_rate);    // (T, mlp_hidden)
    Tensor grad_U_pre(T, mlp_hidden_);
    for (size_t i = 0; i < grad_U_pre.data.size(); ++i)
        grad_U_pre.data[i] = grad_U_act.data[i] * gelu_deriv(last_U_pre.data[i]);
    // Backward through W_1 returns grad_x (T, d_model) AND accumulates grad_W_1, grad_b_1.
    Tensor grad_x_from_W1 = W_1_.backward(grad_U_pre, learning_rate);
    for (size_t i = 0; i < grad_x.data.size(); ++i)
        grad_x.data[i] += grad_x_from_W1.data[i];

    // --- (5) Hypernetwork backward ---
    // Forward: R_logits = W_h · x, R = row_softmax(R_logits) per row.
    // Row-softmax Jacobian: grad_R_logits[i, j] = R[i, j] * (grad_R[i, j] - sum_k grad_R[i, k] * R[i, k]).
    Tensor grad_R_logits(T, num_tokens_);
    std::fill(grad_R_logits.data.begin(), grad_R_logits.data.end(), 0.0);
    for (size_t i = 0; i < T; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < num_tokens_; ++j)
            dot += grad_R[i][j] * last_R[i][j];
        for (size_t j = 0; j < num_tokens_; ++j) {
            grad_R_logits[i][j] = last_R[i][j] * (grad_R[i][j] - dot);
        }
    }
    // Backward through W_h returns grad_x (T, d_model) AND accumulates grad_W_h, grad_b_h.
    Tensor grad_x_from_Wh = W_h_.backward(grad_R_logits, learning_rate);
    for (size_t i = 0; i < grad_x.data.size(); ++i)
        grad_x.data[i] += grad_x_from_Wh.data[i];

    return grad_x;
}

void HyperMixingLayer::update_weights(double learning_rate) {
    W_h_.update_weights(learning_rate);
    W_1_.update_weights(learning_rate);
    W_2_.update_weights(learning_rate);
    W_3_.update_weights(learning_rate);
    W_4_.update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
}

void HyperMixingLayer::zero_grad() {
    W_h_.zero_grad();
    W_1_.zero_grad();
    W_2_.zero_grad();
    W_3_.zero_grad();
    W_4_.zero_grad();
    final_ln_.zero_grad();
}

std::vector<Tensor*> HyperMixingLayer::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_h_.weights);
    p.push_back(&W_h_.bias);
    p.push_back(&W_1_.weights);
    p.push_back(&W_1_.bias);
    p.push_back(&W_2_.weights);
    p.push_back(&W_2_.bias);
    p.push_back(&W_3_.weights);
    p.push_back(&W_3_.bias);
    p.push_back(&W_4_.weights);
    p.push_back(&W_4_.bias);
    p.push_back(&final_ln_.gamma);
    p.push_back(&final_ln_.beta);
    return p;
}

std::vector<Tensor*> HyperMixingLayer::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&W_h_.grad_weights);
    g.push_back(&W_h_.grad_bias);
    g.push_back(&W_1_.grad_weights);
    g.push_back(&W_1_.grad_bias);
    g.push_back(&W_2_.grad_weights);
    g.push_back(&W_2_.grad_bias);
    g.push_back(&W_3_.grad_weights);
    g.push_back(&W_3_.grad_bias);
    g.push_back(&W_4_.grad_weights);
    g.push_back(&W_4_.grad_bias);
    g.push_back(&final_ln_.grad_gamma_);
    g.push_back(&final_ln_.grad_beta_);
    return g;
}

void HyperMixingLayer::copy_params_from(const HyperMixingLayer& other) {
    if (d_model_ != other.d_model_ || mlp_hidden_ != other.mlp_hidden_ ||
        num_tokens_ != other.num_tokens_) {
        throw std::invalid_argument("HyperMixingLayer::copy_params_from: shape mismatch");
    }
    W_h_.weights = other.W_h_.weights;
    W_h_.bias = other.W_h_.bias;
    W_1_.weights = other.W_1_.weights;
    W_1_.bias = other.W_1_.bias;
    W_2_.weights = other.W_2_.weights;
    W_2_.bias = other.W_2_.bias;
    W_3_.weights = other.W_3_.weights;
    W_3_.bias = other.W_3_.bias;
    W_4_.weights = other.W_4_.weights;
    W_4_.bias = other.W_4_.bias;
    final_ln_.gamma = other.final_ln_.gamma;
    final_ln_.beta = other.final_ln_.beta;
}

// ============================================================================
// HyperMixingModel
// ============================================================================

HyperMixingModel::HyperMixingModel(size_t input_dim, size_t d_model, size_t output_dim,
                                   size_t num_layers, size_t mlp_hidden, size_t num_tokens)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      mlp_hidden_(mlp_hidden),
      num_tokens_(num_tokens),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0) throw std::invalid_argument("HyperMixingModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("HyperMixingModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("HyperMixingModel: output_dim must be > 0");
    if (num_layers == 0) throw std::invalid_argument("HyperMixingModel: num_layers must be > 0");

    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(std::make_unique<HyperMixingLayer>(
            d_model, mlp_hidden, num_tokens));
    }
}

Tensor HyperMixingModel::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("HyperMixingModel::forward: input feature dim mismatch");
    Tensor x = embed_.forward(input);
    for (auto& blk : blocks_) {
        x = blk->forward(x);
    }
    x = final_ln_.forward(x);
    return classifier_.forward(x);
}

Tensor HyperMixingModel::backward(const Tensor& grad_output, double lr) {
    Tensor g = classifier_.backward(grad_output, lr);
    g = final_ln_.backward(g, lr);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        g = (*it)->backward(g, lr);
    }
    return embed_.backward(g, lr);
}

void HyperMixingModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void HyperMixingModel::zero_grad() {
    embed_.zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> HyperMixingModel::parameters() {
    std::vector<Tensor*> p;
    auto ep = embed_.parameters();
    p.insert(p.end(), ep.begin(), ep.end());
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fp = final_ln_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> HyperMixingModel::gradients() {
    std::vector<Tensor*> g;
    auto eg = embed_.gradients();
    g.insert(g.end(), eg.begin(), eg.end());
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fg = final_ln_.gradients();
    g.insert(g.end(), fg.begin(), fg.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
