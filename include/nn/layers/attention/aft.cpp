#include "aft.h"
#include <cmath>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// AFT (Attention Free Transformer) — Zhai et al. 2021
//
//   "An Attention Free Transformer"
//   https://arxiv.org/abs/2105.14103
//
// See aft.h for the full mathematical formulation. This file implements:
//
//   * AFTAttention : single-head AFT with learnable position bias w in R^{n x n}
//   * AFTBlock     : pre-LN -> AFT -> residual -> pre-LN -> FFN -> residual
//   * AFTModel     : stack of AFTBlocks + per-token classifier
//
// We follow the "Dense convention" used throughout this repo:
//   Dense::forward computes y = X @ W^T + b, with W stored as (out, in).
// So for AFT: Q = X @ W_q^T (Dense forward) etc.
// ----------------------------------------------------------------------------

static inline double aft_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

static inline double aft_gelu(double x) {
    // Standard GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static inline double aft_gelu_deriv(double x) {
    // gelu'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2pi)) * exp(-x^2/2)
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

// ============================================================================
// AFTAttention
// ============================================================================

AFTAttention::AFTAttention(size_t d_model, size_t max_seq_len)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      position_bias_(max_seq_len, max_seq_len),
      grad_position_bias_(max_seq_len, max_seq_len),
      d_model_(d_model),
      max_seq_len_(max_seq_len)
{
    if (d_model == 0 || max_seq_len == 0) {
        throw std::invalid_argument("AFTAttention: d_model and max_seq_len must be > 0");
    }
    // Initialize position bias to small random values centered on 0 so early
    // training doesn't favor any particular position offset. Scale 0.01.
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.01);
    for (size_t i = 0; i < max_seq_len_; ++i)
        for (size_t j = 0; j < max_seq_len_; ++j)
            position_bias_(i, j) = dis(gen);
    grad_position_bias_.fill(0.0);
}

Tensor AFTAttention::forward(const Tensor& input) {
    // input: (n, d_model). We expect n <= max_seq_len_.
    size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("AFTAttention: input.cols must equal d_model");
    }
    if (n > max_seq_len_) {
        throw std::invalid_argument("AFTAttention: input.rows exceeds max_seq_len");
    }
    last_input_ = input.clone();

    // Project to Q, K, V via Dense (y = X @ W^T + b).
    last_Q_ = W_q.forward(input);  // (n, d_model)
    last_K_ = W_k.forward(input);  // (n, d_model)
    last_V_ = W_v.forward(input);  // (n, d_model)

    // Compute A = sigmoid(Q), the per-channel query gate.
    last_A_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            last_A_(t, c) = aft_sigmoid(last_Q_(t, c));
        }
    }

    // For each output position t, compute:
    //   Y_t[c] = sum_s exp(K_s[c] + w_{t,s}) * V_s[c]
    //   Z_t[c] = sum_s exp(K_s[c] + w_{t,s})
    //   out_t[c] = A_t[c] * Y_t[c] / (Z_t[c] + eps)
    //
    // The sum over s is O(n) per output position, so the whole attention
    // computation is O(n * d_model + n^2). The n^2 term comes from the
    // position-bias precomputation (or from re-evaluating w_{t,s} inside
    // the inner loop).
    const double eps = 1e-6;
    Tensor output(n, d_model_);
    Tensor Y_local(n, d_model_);
    Tensor Z_local(n, d_model_);

    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double y_acc = 0.0;
            double z_acc = 0.0;
            for (size_t s = 0; s < n; ++s) {
                double w_ts = position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                y_acc += e * last_V_(s, c);
                z_acc += e;
            }
            Y_local(t, c) = y_acc;
            Z_local(t, c) = z_acc;
            output(t, c) = last_A_(t, c) * y_acc / (z_acc + eps);
        }
    }
    last_Y_ = Y_local;
    last_Z_ = Z_local;
    last_output_pre_wo_ = output.clone();

    // Final output projection: out = Y @ W_o^T + b_o
    Tensor final_output = W_o.forward(output);
    return final_output;
}

Tensor AFTAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = last_input_.rows;

    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();

    // (1) Backward through W_o. y = (n, d_model) -> out = y @ W_o^T + b_o.
    //   d_y[t, i] = sum_j grad_output[t, j] * W_o.weights[j, i]
    //   d_W_o[j, i] += sum_t grad_output[t, j] * y[t, i]
    //   d_b_o[j]   += sum_t grad_output[t, j]
    Tensor grad_y(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                acc += grad_output(t, j) * W_o.weights(j, i);
            }
            grad_y(t, i) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_output_pre_wo_(t, i);
            W_o.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // (2) Backward through the AFT formula.
    //
    //   out_t[c] = A_t[c] * Y_t[c] / (Z_t[c] + eps)
    //   A_t[c]   = sigmoid(Q_t[c])
    //   Y_t[c]   = sum_s exp(K_s[c] + w_{t,s}) * V_s[c]
    //   Z_t[c]   = sum_s exp(K_s[c] + w_{t,s})
    //
    // Chain rule:
    //   dA_t[c] = grad_y[t,c] * Y_t[c] / (Z_t[c] + eps)
    //   dY_t[c] = grad_y[t,c] * A_t[c] / (Z_t[c] + eps)
    //   dZ_t[c] = grad_y[t,c] * (-A_t[c] * Y_t[c]) / (Z_t[c] + eps)^2
    //
    //   dQ_t[c] = dA_t[c] * A_t[c] * (1 - A_t[c])   // sigmoid derivative
    //
    //   For dK_s, dV_s, dw_{t,s}:
    //     e_{t,s,c} = exp(K_s[c] + w_{t,s})
    //     dK_s[c]  += sum_t dY_t[c] * e_{t,s,c} * V_s[c]      + dZ_t[c] * e_{t,s,c}
    //     dV_s[c]  += sum_t dY_t[c] * e_{t,s,c}                // V_s enters linearly
    //     dw_{t,s} += sum_c [ dY_t[c] * e_{t,s,c} * V_s[c] + dZ_t[c] * e_{t,s,c} ]
    Tensor grad_Q(n, d_model_);
    Tensor grad_K(n, d_model_);
    Tensor grad_V(n, d_model_);
    grad_Q.fill(0.0);
    grad_K.fill(0.0);
    grad_V.fill(0.0);

    const double eps = 1e-6;
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double a_tc = last_A_(t, c);
            double y_tc = last_Y_(t, c);
            double z_tc = last_Z_(t, c);
            double dA = grad_y(t, c) * y_tc / (z_tc + eps);
            double dY = grad_y(t, c) * a_tc / (z_tc + eps);
            double dZ = grad_y(t, c) * (-a_tc * y_tc) / ((z_tc + eps) * (z_tc + eps));
            grad_Q(t, c) = dA * a_tc * (1.0 - a_tc);

            for (size_t s = 0; s < n; ++s) {
                double w_ts = position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                grad_K(s, c) += (dY * e * last_V_(s, c)) + (dZ * e);
                grad_V(s, c) += dY * e;
            }
        }
    }
    // Position-bias gradient in a (t, s)-outer loop for clarity:
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < n; ++s) {
            double gw = 0.0;
            for (size_t c = 0; c < d_model_; ++c) {
                double a_tc = last_A_(t, c);
                double y_tc = last_Y_(t, c);
                double z_tc = last_Z_(t, c);
                double dY = grad_y(t, c) * a_tc / (z_tc + eps);
                double dZ = grad_y(t, c) * (-a_tc * y_tc) / ((z_tc + eps) * (z_tc + eps));
                double w_ts = position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                gw += dY * e * last_V_(s, c) + dZ * e;
            }
            grad_position_bias_(t, s) += gw;
        }
    }

    // (3) Backward through W_q, W_k, W_v. For Dense y = X @ W^T + b:
    //   grad_W[i, k] += sum_t grad_y[t, i] * X[t, k]
    //   grad_b[i]   += sum_t grad_y[t, i]
    //   grad_X[t, k] = sum_i grad_y[t, i] * W[i, k]
    Tensor grad_input(n, d_model_);
    grad_input.fill(0.0);

    // W_q
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_Q(t, i) * last_input_(t, k);
            W_q.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_Q(t, i);
        W_q.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_Q(t, i) * W_q.weights(i, k);
    }
    // W_k
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_K(t, i) * last_input_(t, k);
            W_k.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_K(t, i);
        W_k.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_K(t, i) * W_k.weights(i, k);
    }
    // W_v
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_V(t, i) * last_input_(t, k);
            W_v.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_V(t, i);
        W_v.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_V(t, i) * W_v.weights(i, k);
    }

    return grad_input;
}

void AFTAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < max_seq_len_; ++i)
        for (size_t j = 0; j < max_seq_len_; ++j)
            position_bias_(i, j) -= learning_rate * grad_position_bias_(i, j);
}

void AFTAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    grad_position_bias_.fill(0.0);
}

std::vector<Tensor*> AFTAttention::parameters() {
    return {&W_q.weights, &W_q.bias, &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias, &W_o.weights, &W_o.bias, &position_bias_};
}

std::vector<Tensor*> AFTAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias, &W_k.grad_weights, &W_k.grad_bias,
            &W_v.grad_weights, &W_v.grad_bias, &W_o.grad_weights, &W_o.grad_bias,
            &grad_position_bias_};
}

// ============================================================================
// AFTBlock
// ============================================================================

AFTBlock::AFTBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim)
    : attn(d_model, max_seq_len),
      ln1(d_model),
      ln2(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    if (d_model == 0) throw std::invalid_argument("AFTBlock: d_model must be > 0");
}

Tensor AFTBlock::forward(const Tensor& input) {
    last_x_ = input.clone();

    // Pre-LN -> AFT -> residual
    last_z1_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_ = input + last_attn_out_;

    // Pre-LN -> FFN -> residual
    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    // GELU hidden
    Tensor h_act(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.rows; ++i)
        for (size_t j = 0; j < last_h_pre_.cols; ++j)
            h_act(i, j) = aft_gelu(last_h_pre_(i, j));
    last_ffn_h_ = h_act.clone();
    last_ffn_out_ = ffn_fc2_.forward(h_act);
    return last_res1_ + last_ffn_out_;
}

Tensor AFTBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Reverse path:
    //   out = res1 + ffn_out
    //   ffn_out = ffn_fc2.forward(gelu(ffn_fc1.forward(ln2(res1))))
    // For the residual sum: d_res1 = grad_output, d_ffn_out = grad_output.
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();

    size_t n = grad_output.rows;

    // (1) ffn_fc2 backward
    // ffn_out = h_act @ W_fc2^T + b_fc2
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
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_ffn_h_(t, k);
            ffn_fc2_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        ffn_fc2_.grad_bias(0, j) += b_acc;
    }

    // (2) GELU backward on h_pre (we cached last_h_pre_ in forward)
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h_act(t, k) * aft_gelu_deriv(last_h_pre_(t, k));
        }
    }

    // (3) ffn_fc1 backward
    // h_pre = z2 @ W_fc1^T + b_fc1
    // d_z2[t, k] = sum_j d_h_pre[t, j] * W_fc1.weights[j, k]
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

    // (4) ln2 backward: d_z2 -> d_res1
    // This is the gradient flowing through the FFN path: grad_output -> h_act
    //   -> h_pre -> z2 -> ln2 -> res1.
    Tensor d_res1_from_ffn = ln2.backward(d_z2, 0.0);
    // (5) The total d_res1 has two contributions:
    //   * from the FFN path: d_res1_from_ffn
    //   * from the direct residual path (out = res1 + ffn_out): grad_output
    //     (because d out / d res1 = 1).
    Tensor d_res1 = d_res1_from_ffn + grad_output;
    // (6) The total d_x (block input) also has two contributions:
    //   * from the residual direct path (res1 = x + attn): d_res1 * 1
    //   * from the ln1->attn chain: attn.backward(d_res1) -> ln1.backward
    Tensor d_x_total = d_res1.clone();
    // (7) AFT backward: d_res1 -> d_z1 (where z1 = ln1(x))
    Tensor d_z1 = attn.backward(d_res1, 0.0);
    // (8) ln1 backward: d_z1 -> d_x
    Tensor d_x_from_ln1 = ln1.backward(d_z1, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t k = 0; k < d_model_; ++k)
            d_x_total(t, k) += d_x_from_ln1(t, k);

    return d_x_total;
}

void AFTBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void AFTBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> AFTBlock::parameters() {
    auto p = attn.parameters();
    auto a = ln1.parameters();
    auto b = ln2.parameters();
    auto c = ffn_fc1_.parameters();
    auto d = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> AFTBlock::gradients() {
    auto g = attn.gradients();
    auto a = ln1.gradients();
    auto b = ln2.gradients();
    auto c = ffn_fc1_.gradients();
    auto d = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}

// ============================================================================
// AFTModel
// ============================================================================

AFTModel::AFTModel(size_t d_model, size_t max_seq_len, size_t out_features,
                   size_t num_blocks, size_t ffn_dim)
    : classifier_(d_model, out_features),
      d_model_(d_model),
      max_seq_len_(max_seq_len),
      out_features_(out_features),
      num_blocks_(num_blocks),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    if (d_model == 0 || max_seq_len == 0 || out_features == 0 || num_blocks == 0) {
        throw std::invalid_argument("AFTModel: d_model, max_seq_len, out_features, num_blocks must be > 0");
    }
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks.push_back(std::make_unique<AFTBlock>(d_model, max_seq_len, ffn_dim_));
    }
    last_block_output_ = Tensor(0, 0);  // placeholder
}

Tensor AFTModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& block : blocks) {
        x = block->forward(x);
    }
    last_block_output_ = x.clone();
    return classifier_.forward(x);
}

Tensor AFTModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = grad_output.rows;
    classifier_.zero_grad();

    // grad_x starts from the classifier backward
    // classifier: y = x @ W_c^T + b_c, so d_x[t, k] = sum_j grad_y[t, j] * W_c[j, k]
    Tensor grad_x(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t j = 0; j < out_features_; ++j) acc += grad_output(t, j) * classifier_.weights(j, k);
            grad_x(t, k) = acc;
        }
    }
    // grad_W_c[j, k] += sum_t grad_output[t, j] * x[t, k]
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

    // Backward through each block in reverse
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        grad_x = (*it)->backward(grad_x, 0.0);
    }
    return grad_x;
}

void AFTModel::update_weights(double learning_rate) {
    for (auto& block : blocks) block->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void AFTModel::zero_grad() {
    for (auto& block : blocks) block->zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> AFTModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& block : blocks) {
        auto bp = block->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> AFTModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& block : blocks) {
        auto bg = block->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

// ============================================================================
// AFTLocalAttention
// ============================================================================

AFTLocalAttention::AFTLocalAttention(size_t d_model, size_t max_seq_len, size_t window)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      relative_bias_(1, 2 * window - 1),
      grad_relative_bias_(1, 2 * window - 1),
      last_position_bias_(max_seq_len, max_seq_len),
      d_model_(d_model),
      max_seq_len_(max_seq_len),
      window_(window)
{
    if (d_model == 0 || max_seq_len == 0 || window == 0) {
        throw std::invalid_argument("AFTLocalAttention: d_model, max_seq_len, window must be > 0");
    }
    // Initialize relative bias to small random values centered on 0.
    std::mt19937 gen(43);
    std::normal_distribution<> dis(0.0, 0.01);
    for (size_t i = 0; i < relative_bias_.cols; ++i) {
        relative_bias_(0, i) = dis(gen);
    }
    grad_relative_bias_.fill(0.0);
}

Tensor AFTLocalAttention::forward(const Tensor& input) {
    size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("AFTLocalAttention: input.cols must equal d_model");
    }
    if (n > max_seq_len_) {
        throw std::invalid_argument("AFTLocalAttention: input.rows exceeds max_seq_len");
    }
    last_input_ = input.clone();

    // Materialize position bias from relative_bias_:
    //   w_{t,s} = relative_bias[t - s + (window - 1)]   if |t - s| < window
    //           = -1e9                                  otherwise
    for (size_t t = 0; t < max_seq_len_; ++t) {
        for (size_t s = 0; s < max_seq_len_; ++s) {
            long d = (long)t - (long)s;
            if (std::abs(d) < (long)window_) {
                last_position_bias_(t, s) = relative_bias_(0, d + (long)window_ - 1);
            } else {
                last_position_bias_(t, s) = -1e9;
            }
        }
    }

    // Project to Q, K, V
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    // A = sigmoid(Q)
    last_A_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            last_A_(t, c) = aft_sigmoid(last_Q_(t, c));
        }
    }

    const double eps = 1e-6;
    Tensor output(n, d_model_);
    Tensor Y_local(n, d_model_);
    Tensor Z_local(n, d_model_);

    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double y_acc = 0.0;
            double z_acc = 0.0;
            for (size_t s = 0; s < n; ++s) {
                double w_ts = last_position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                y_acc += e * last_V_(s, c);
                z_acc += e;
            }
            Y_local(t, c) = y_acc;
            Z_local(t, c) = z_acc;
            output(t, c) = last_A_(t, c) * y_acc / (z_acc + eps);
        }
    }
    last_Y_ = Y_local;
    last_Z_ = Z_local;
    last_output_pre_wo_ = output.clone();

    return W_o.forward(output);
}

Tensor AFTLocalAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = last_input_.rows;

    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();

    // (1) Backward through W_o
    Tensor grad_y(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                acc += grad_output(t, j) * W_o.weights(j, i);
            }
            grad_y(t, i) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_output_pre_wo_(t, i);
            W_o.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // (2) Backward through the AFT formula
    Tensor grad_Q(n, d_model_);
    Tensor grad_K(n, d_model_);
    Tensor grad_V(n, d_model_);
    grad_Q.fill(0.0);
    grad_K.fill(0.0);
    grad_V.fill(0.0);

    const double eps = 1e-6;
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double a_tc = last_A_(t, c);
            double y_tc = last_Y_(t, c);
            double z_tc = last_Z_(t, c);
            double dA = grad_y(t, c) * y_tc / (z_tc + eps);
            double dY = grad_y(t, c) * a_tc / (z_tc + eps);
            double dZ = grad_y(t, c) * (-a_tc * y_tc) / ((z_tc + eps) * (z_tc + eps));
            grad_Q(t, c) = dA * a_tc * (1.0 - a_tc);

            for (size_t s = 0; s < n; ++s) {
                double w_ts = last_position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                grad_K(s, c) += (dY * e * last_V_(s, c)) + (dZ * e);
                grad_V(s, c) += dY * e;
            }
        }
    }

    // Position-bias gradient: accumulate (t, s) -> relative offset index.
    // Only contribute when |t-s| < window; skip otherwise (those entries
    // are -1e9 and are not differentiable params).
    Tensor grad_pb_local(max_seq_len_, max_seq_len_);
    grad_pb_local.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < n; ++s) {
            long d = (long)t - (long)s;
            if (std::abs(d) >= (long)window_) continue;
            double gw = 0.0;
            for (size_t c = 0; c < d_model_; ++c) {
                double a_tc = last_A_(t, c);
                double y_tc = last_Y_(t, c);
                double z_tc = last_Z_(t, c);
                double dY = grad_y(t, c) * a_tc / (z_tc + eps);
                double dZ = grad_y(t, c) * (-a_tc * y_tc) / ((z_tc + eps) * (z_tc + eps));
                double w_ts = last_position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                gw += dY * e * last_V_(s, c) + dZ * e;
            }
            grad_pb_local(t, s) = gw;
            grad_relative_bias_(0, d + (long)window_ - 1) += gw;
        }
    }

    // (3) Backward through W_q, W_k, W_v
    Tensor grad_input(n, d_model_);
    grad_input.fill(0.0);

    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_Q(t, i) * last_input_(t, k);
            W_q.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_Q(t, i);
        W_q.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_Q(t, i) * W_q.weights(i, k);
    }
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_K(t, i) * last_input_(t, k);
            W_k.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_K(t, i);
        W_k.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_K(t, i) * W_k.weights(i, k);
    }
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_V(t, i) * last_input_(t, k);
            W_v.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_V(t, i);
        W_v.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_V(t, i) * W_v.weights(i, k);
    }

    return grad_input;
}

void AFTLocalAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < relative_bias_.cols; ++i) {
        relative_bias_(0, i) -= learning_rate * grad_relative_bias_(0, i);
    }
}

void AFTLocalAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    grad_relative_bias_.fill(0.0);
}

std::vector<Tensor*> AFTLocalAttention::parameters() {
    return {&W_q.weights, &W_q.bias, &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias, &W_o.weights, &W_o.bias, &relative_bias_};
}

std::vector<Tensor*> AFTLocalAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias, &W_k.grad_weights, &W_k.grad_bias,
            &W_v.grad_weights, &W_v.grad_bias, &W_o.grad_weights, &W_o.grad_bias,
            &grad_relative_bias_};
}

// ============================================================================
// AFTLocalBlock — mirrors AFTBlock with AFTLocalAttention in place of AFTAttention
// ============================================================================

AFTLocalBlock::AFTLocalBlock(size_t d_model, size_t max_seq_len, size_t window, size_t ffn_dim)
    : attn(d_model, max_seq_len, window),
      ln1(d_model),
      ln2(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    if (d_model == 0) throw std::invalid_argument("AFTLocalBlock: d_model must be > 0");
}

Tensor AFTLocalBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    last_z1_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_ = input + last_attn_out_;
    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    Tensor h_act(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.rows; ++i)
        for (size_t j = 0; j < last_h_pre_.cols; ++j)
            h_act(i, j) = aft_gelu(last_h_pre_(i, j));
    last_ffn_h_ = h_act.clone();
    last_ffn_out_ = ffn_fc2_.forward(h_act);
    return last_res1_ + last_ffn_out_;
}

Tensor AFTLocalBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    ln1.zero_grad();
    ln2.zero_grad();
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
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_ffn_h_(t, k);
            ffn_fc2_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        ffn_fc2_.grad_bias(0, j) += b_acc;
    }

    // (2) GELU backward
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h_act(t, k) * aft_gelu_deriv(last_h_pre_(t, k));
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

    // (4) ln2 backward
    Tensor d_res1_from_ffn = ln2.backward(d_z2, 0.0);
    Tensor d_res1 = d_res1_from_ffn + grad_output;
    Tensor d_x_total = d_res1.clone();
    Tensor d_z1 = attn.backward(d_res1, 0.0);
    Tensor d_x_from_ln1 = ln1.backward(d_z1, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t k = 0; k < d_model_; ++k)
            d_x_total(t, k) += d_x_from_ln1(t, k);

    return d_x_total;
}

void AFTLocalBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void AFTLocalBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> AFTLocalBlock::parameters() {
    auto p = attn.parameters();
    auto a = ln1.parameters();
    auto b = ln2.parameters();
    auto c = ffn_fc1_.parameters();
    auto d = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> AFTLocalBlock::gradients() {
    auto g = attn.gradients();
    auto a = ln1.gradients();
    auto b = ln2.gradients();
    auto c = ffn_fc1_.gradients();
    auto d = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}

// ============================================================================
// AFTConvAttention
// ============================================================================

AFTConvAttention::AFTConvAttention(size_t d_model, size_t max_seq_len, size_t rank)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      position_embedding_(max_seq_len, rank == 0 ? max_seq_len : rank),
      W_conv_(rank == 0 ? max_seq_len : rank, rank == 0 ? max_seq_len : rank),
      b_conv_(1, rank == 0 ? max_seq_len : rank),
      grad_position_embedding_(max_seq_len, rank == 0 ? max_seq_len : rank),
      grad_W_conv_(rank == 0 ? max_seq_len : rank, rank == 0 ? max_seq_len : rank),
      grad_b_conv_(1, rank == 0 ? max_seq_len : rank),
      d_model_(d_model),
      max_seq_len_(max_seq_len),
      rank_(rank == 0 ? max_seq_len : rank)
{
    if (d_model == 0 || max_seq_len == 0) {
        throw std::invalid_argument("AFTConvAttention: d_model, max_seq_len must be > 0");
    }
    if (rank > max_seq_len) {
        throw std::invalid_argument("AFTConvAttention: rank must be <= max_seq_len");
    }
    // Initialize position embedding to small random values.
    std::mt19937 gen(44);
    std::normal_distribution<> dis(0.0, 0.05);
    for (size_t i = 0; i < position_embedding_.rows; ++i)
        for (size_t j = 0; j < position_embedding_.cols; ++j)
            position_embedding_(i, j) = dis(gen);
    // 1x1 conv kernel: small random init, identity-like.
    for (size_t i = 0; i < W_conv_.rows; ++i) {
        for (size_t j = 0; j < W_conv_.cols; ++j) {
            W_conv_(i, j) = (i == j ? 0.1 : 0.0) + 0.01 * dis(gen);
        }
    }
    for (size_t i = 0; i < b_conv_.cols; ++i) b_conv_(0, i) = 0.0;
    grad_position_embedding_.fill(0.0);
    grad_W_conv_.fill(0.0);
    grad_b_conv_.fill(0.0);
}

Tensor AFTConvAttention::forward(const Tensor& input) {
    size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("AFTConvAttention: input.cols must equal d_model");
    }
    if (n > max_seq_len_) {
        throw std::invalid_argument("AFTConvAttention: input.rows exceeds max_seq_len");
    }
    last_input_ = input.clone();

    // Materialize position bias:
    //   q_t = pos_emb[t]   (rank_)
    //   q'_t = W_conv @ q_t + b_conv  (rank_)
    //   w_{t,s} = (1/K) * sum_k q'_t[k] * q'_s[k]   scalar
    last_q_ = Tensor(n, rank_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < rank_; ++k) {
            last_q_(t, k) = position_embedding_(t, k);
        }
    }
    last_qp_ = Tensor(n, rank_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < rank_; ++i) {
            double acc = b_conv_(0, i);
            for (size_t k = 0; k < rank_; ++k) acc += last_q_(t, k) * W_conv_(i, k);
            last_qp_(t, i) = acc;
        }
    }
    // Cache position bias (n, n) and scale by 1/K for the AFT formula.
    last_position_bias_ = Tensor(n, n);
    double inv_K = 1.0 / (double)rank_;
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < n; ++s) {
            double acc = 0.0;
            for (size_t k = 0; k < rank_; ++k) {
                acc += last_qp_(t, k) * last_qp_(s, k);
            }
            last_position_bias_(t, s) = acc * inv_K;
        }
    }

    // Project to Q, K, V
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    last_A_ = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            last_A_(t, c) = aft_sigmoid(last_Q_(t, c));
        }
    }

    const double eps = 1e-6;
    Tensor output(n, d_model_);
    Tensor Y_local(n, d_model_);
    Tensor Z_local(n, d_model_);

    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double y_acc = 0.0;
            double z_acc = 0.0;
            for (size_t s = 0; s < n; ++s) {
                double w_ts = last_position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                y_acc += e * last_V_(s, c);
                z_acc += e;
            }
            Y_local(t, c) = y_acc;
            Z_local(t, c) = z_acc;
            output(t, c) = last_A_(t, c) * y_acc / (z_acc + eps);
        }
    }
    last_Y_ = Y_local;
    last_Z_ = Z_local;
    last_output_pre_wo_ = output.clone();

    return W_o.forward(output);
}

Tensor AFTConvAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t n = last_input_.rows;
    double inv_K = 1.0 / (double)rank_;

    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();

    // (1) Backward through W_o
    Tensor grad_y(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                acc += grad_output(t, j) * W_o.weights(j, i);
            }
            grad_y(t, i) = acc;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_output_pre_wo_(t, i);
            W_o.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // (2) Backward through the AFT formula
    Tensor grad_Q(n, d_model_);
    Tensor grad_K(n, d_model_);
    Tensor grad_V(n, d_model_);
    grad_Q.fill(0.0);
    grad_K.fill(0.0);
    grad_V.fill(0.0);

    const double eps = 1e-6;
    // Materialize grad_pb(t, s) = dL/d_w_{t,s} = sum_c [ dY_t[c] * e * V_s[c] + dZ_t[c] * e ]
    // The 1/K factor from the bilinear form is applied separately when computing
    // d_q'_t[k] (since d_w_{t,s}/d_q'_t[k] = (1/K) * q'_s[k]).
    Tensor grad_pb(n, n);
    grad_pb.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            double a_tc = last_A_(t, c);
            double y_tc = last_Y_(t, c);
            double z_tc = last_Z_(t, c);
            double dA = grad_y(t, c) * y_tc / (z_tc + eps);
            double dY = grad_y(t, c) * a_tc / (z_tc + eps);
            double dZ = grad_y(t, c) * (-a_tc * y_tc) / ((z_tc + eps) * (z_tc + eps));
            grad_Q(t, c) = dA * a_tc * (1.0 - a_tc);

            for (size_t s = 0; s < n; ++s) {
                double w_ts = last_position_bias_(t, s);
                double e = std::exp(last_K_(s, c) + w_ts);
                grad_K(s, c) += (dY * e * last_V_(s, c)) + (dZ * e);
                grad_V(s, c) += dY * e;
                grad_pb(t, s) += dY * e * last_V_(s, c) + dZ * e;
            }
        }
    }

    // (3) Backward through the position-bias bilinear + 1x1 conv.
    // Forward: w_{t,s} = (1/K) * sum_k q'_t[k] * q'_s[k]
    //   d_w_{t,s} / d_q'_t[k] = (1/K) * q'_s[k]   (t-side: q'_t appears as the "t" factor)
    //   d_w_{t,s} / d_q'_s[k] = (1/K) * q'_t[k]   (s-side: q'_t appears as the "s" factor)
    // So the full dL/d_q'_t[k] has TWO contributions:
    //   dL/d_q'_t[k] = sum_s (dL/d_w_{t,s}) * (1/K) * q'_s[k]      (t-side)
    //                + sum_{t'} (dL/d_w_{t',t}) * (1/K) * q'_{t'}[k] (s-side)
    //
    // q'_t[i] = sum_k W_conv[i, k] * q_t[k] + b_conv[i]
    //   d_W_conv[i, k] = sum_t d_q'_t[i] * q_t[k]   (uses FULL d_q'_t, both sides)
    //   d_b_conv[i]    = sum_t d_q'_t[i]
    //   d_q_t[k]       = sum_i d_q'_t[i] * W_conv[i, k]
    //   d_pos_emb[t, k] = d_q_t[k]
    //
    // Compute d_q'_t[k] = t-side + s-side.
    // t-side:  d_qp(t, k)        = (1/K) * sum_s grad_pb(t, s) * q'_s[k]
    // s-side:  d_qp_sym(t, k)    = (1/K) * sum_{t'} grad_pb(t', t) * q'_{t'}[k]
    Tensor d_qp(n, rank_);
    d_qp.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < rank_; ++k) {
            double acc = 0.0;
            for (size_t s = 0; s < n; ++s) acc += grad_pb(t, s) * last_qp_(s, k);
            d_qp(t, k) = acc * inv_K;
        }
    }
    Tensor d_qp_sym(n, rank_);
    d_qp_sym.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < rank_; ++k) {
            double acc = 0.0;
            for (size_t t2 = 0; t2 < n; ++t2) acc += grad_pb(t2, t) * last_qp_(t2, k);
            d_qp_sym(t, k) = acc * inv_K;
        }
    }
    // d_W_conv[i, k] = sum_t (d_qp(t, i) + d_qp_sym(t, i)) * last_q_[t, k]
    for (size_t i = 0; i < rank_; ++i) {
        for (size_t k = 0; k < rank_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += (d_qp(t, i) + d_qp_sym(t, i)) * last_q_(t, k);
            grad_W_conv_(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += d_qp(t, i) + d_qp_sym(t, i);
        grad_b_conv_(0, i) += b_acc;
    }
    // d_q_t[t, k] = sum_i (d_qp(t, i) + d_qp_sym(t, i)) * W_conv[i, k]  -- d_pos_emb[t, k]
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < rank_; ++k) {
            double d_qt = 0.0;
            for (size_t i = 0; i < rank_; ++i) d_qt += (d_qp(t, i) + d_qp_sym(t, i)) * W_conv_(i, k);
            grad_position_embedding_(t, k) += d_qt;
        }
    }

    // (4) Backward through W_q, W_k, W_v
    Tensor grad_input(n, d_model_);
    grad_input.fill(0.0);

    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_Q(t, i) * last_input_(t, k);
            W_q.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_Q(t, i);
        W_q.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_Q(t, i) * W_q.weights(i, k);
    }
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_K(t, i) * last_input_(t, k);
            W_k.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_K(t, i);
        W_k.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_K(t, i) * W_k.weights(i, k);
    }
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double acc = 0.0;
            for (size_t t = 0; t < n; ++t) acc += grad_V(t, i) * last_input_(t, k);
            W_v.grad_weights(i, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_V(t, i);
        W_v.grad_bias(0, i) += b_acc;
        for (size_t t = 0; t < n; ++t)
            for (size_t k = 0; k < d_model_; ++k)
                grad_input(t, k) += grad_V(t, i) * W_v.weights(i, k);
    }

    return grad_input;
}

void AFTConvAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < position_embedding_.rows; ++i)
        for (size_t j = 0; j < position_embedding_.cols; ++j)
            position_embedding_(i, j) -= learning_rate * grad_position_embedding_(i, j);
    for (size_t i = 0; i < W_conv_.rows; ++i)
        for (size_t j = 0; j < W_conv_.cols; ++j)
            W_conv_(i, j) -= learning_rate * grad_W_conv_(i, j);
    for (size_t i = 0; i < b_conv_.cols; ++i)
        b_conv_(0, i) -= learning_rate * grad_b_conv_(0, i);
}

void AFTConvAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    grad_position_embedding_.fill(0.0);
    grad_W_conv_.fill(0.0);
    grad_b_conv_.fill(0.0);
}

std::vector<Tensor*> AFTConvAttention::parameters() {
    return {&W_q.weights, &W_q.bias, &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias, &W_o.weights, &W_o.bias,
            &position_embedding_, &W_conv_, &b_conv_};
}

std::vector<Tensor*> AFTConvAttention::gradients() {
    return {&W_q.grad_weights, &W_q.grad_bias, &W_k.grad_weights, &W_k.grad_bias,
            &W_v.grad_weights, &W_v.grad_bias, &W_o.grad_weights, &W_o.grad_bias,
            &grad_position_embedding_, &grad_W_conv_, &grad_b_conv_};
}

// ============================================================================
// AFTConvBlock
// ============================================================================

AFTConvBlock::AFTConvBlock(size_t d_model, size_t max_seq_len, size_t rank, size_t ffn_dim)
    : attn(d_model, max_seq_len, rank),
      ln1(d_model),
      ln2(d_model),
      ffn_fc1_(d_model, ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ffn_fc2_(ffn_dim == 0 ? 4 * d_model : ffn_dim, d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim)
{
    if (d_model == 0) throw std::invalid_argument("AFTConvBlock: d_model must be > 0");
}

Tensor AFTConvBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    last_z1_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_ = input + last_attn_out_;
    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    Tensor h_act(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.rows; ++i)
        for (size_t j = 0; j < last_h_pre_.cols; ++j)
            h_act(i, j) = aft_gelu(last_h_pre_(i, j));
    last_ffn_h_ = h_act.clone();
    last_ffn_out_ = ffn_fc2_.forward(h_act);
    return last_res1_ + last_ffn_out_;
}

Tensor AFTConvBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    ln1.zero_grad();
    ln2.zero_grad();
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
            for (size_t t = 0; t < n; ++t) acc += grad_output(t, j) * last_ffn_h_(t, k);
            ffn_fc2_.grad_weights(j, k) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < n; ++t) b_acc += grad_output(t, j);
        ffn_fc2_.grad_bias(0, j) += b_acc;
    }

    // (2) GELU backward
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h_act(t, k) * aft_gelu_deriv(last_h_pre_(t, k));
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

    // (4) ln2 backward
    Tensor d_res1_from_ffn = ln2.backward(d_z2, 0.0);
    Tensor d_res1 = d_res1_from_ffn + grad_output;
    Tensor d_x_total = d_res1.clone();
    Tensor d_z1 = attn.backward(d_res1, 0.0);
    Tensor d_x_from_ln1 = ln1.backward(d_z1, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t k = 0; k < d_model_; ++k)
            d_x_total(t, k) += d_x_from_ln1(t, k);

    return d_x_total;
}

void AFTConvBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void AFTConvBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> AFTConvBlock::parameters() {
    auto p = attn.parameters();
    auto a = ln1.parameters();
    auto b = ln2.parameters();
    auto c = ffn_fc1_.parameters();
    auto d = ffn_fc2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> AFTConvBlock::gradients() {
    auto g = attn.gradients();
    auto a = ln1.gradients();
    auto b = ln2.gradients();
    auto c = ffn_fc1_.gradients();
    auto d = ffn_fc2_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}
