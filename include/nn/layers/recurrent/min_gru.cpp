#include "min_gru.h"
#include <cmath>
#include <algorithm>
#include <random>

MinGRU::MinGRU(size_t input_dim, size_t hidden_size)
    : input_dim_(input_dim), hidden_size_(hidden_size), seq_len_(0),
      W_g_(input_dim, hidden_size), b_g_(1, hidden_size),
      W_h_(input_dim, hidden_size), b_h_(1, hidden_size),
      grad_W_g_(input_dim, hidden_size), grad_b_g_(1, hidden_size),
      grad_W_h_(input_dim, hidden_size), grad_b_h_(1, hidden_size),
      h_(1, hidden_size), last_output_(1, hidden_size),
      gates_(1, hidden_size), cand_(1, hidden_size),
      h_prev_(1, hidden_size) {
    init_weights();
    zero_grad();
}

void MinGRU::init_weights() {
    // Xavier-uniform init for the two input projections (same scale).
    double scale = std::sqrt(6.0 / (input_dim_ + hidden_size_));
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-scale, scale);
    for (size_t i = 0; i < W_g_.rows; ++i)
        for (size_t j = 0; j < W_g_.cols; ++j)
            W_g_[i][j] = dis(gen);
    for (size_t i = 0; i < W_h_.rows; ++i)
        for (size_t j = 0; j < W_h_.cols; ++j)
            W_h_[i][j] = dis(gen);
    // Forget-bias-style init for the gate bias (slightly positive so the
    // initial retention factor alpha = 1 - sigmoid(b_g) is around 0.5).
    for (size_t j = 0; j < b_g_.cols; ++j) b_g_[0][j] = 0.0;
    for (size_t j = 0; j < b_h_.cols; ++j) b_h_[0][j] = 0.0;
}

void MinGRU::zero_grad() {
    grad_W_g_.fill(0.0); grad_b_g_.fill(0.0);
    grad_W_h_.fill(0.0); grad_b_h_.fill(0.0);
    h_.fill(0.0);
}

Tensor MinGRU::forward(const Tensor& input) {
    h_prev_ = h_;
    Tensor x = input.rows > 1 ? input.get_row(0) : input;

    // gates = sigmoid(x @ W_g + b_g)
    for (size_t j = 0; j < hidden_size_; ++j) {
        double v = b_g_[0][j];
        for (size_t k = 0; k < input_dim_; ++k) v += x[0][k] * W_g_[k][j];
        gates_[0][j] = 1.0 / (1.0 + std::exp(-v));
    }
    // cand = x @ W_h + b_h
    for (size_t j = 0; j < hidden_size_; ++j) {
        double v = b_h_[0][j];
        for (size_t k = 0; k < input_dim_; ++k) v += x[0][k] * W_h_[k][j];
        cand_[0][j] = v;
    }
    // h = (1 - gates) * h_prev + gates * cand
    for (size_t j = 0; j < hidden_size_; ++j) {
        double g = gates_[0][j];
        h_[0][j] = (1.0 - g) * h_prev_[0][j] + g * cand_[0][j];
    }
    last_output_ = h_;
    return last_output_;
}

Tensor MinGRU::forward_sequence(const Tensor& seq) {
    size_t seq_len = seq.rows;
    seq_len_ = seq_len;
    inputs_ = seq;
    hidden_states_ = Tensor(seq_len + 1, hidden_size_);
    hidden_states_.fill(0.0);
    cached_gates_ = Tensor(seq_len, hidden_size_);
    cached_cand_ = Tensor(seq_len, hidden_size_);

    h_prev_ = h_;
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_row = seq.get_row(t);
        // gates_t
        for (size_t j = 0; j < hidden_size_; ++j) {
            double v = b_g_[0][j];
            for (size_t k = 0; k < input_dim_; ++k) v += x_row[0][k] * W_g_[k][j];
            double gv = 1.0 / (1.0 + std::exp(-v));
            gates_[0][j] = gv;
            cached_gates_[t][j] = gv;
        }
        // cand_t
        for (size_t j = 0; j < hidden_size_; ++j) {
            double v = b_h_[0][j];
            for (size_t k = 0; k < input_dim_; ++k) v += x_row[0][k] * W_h_[k][j];
            cand_[0][j] = v;
            cached_cand_[t][j] = v;
        }
        // h_t = (1 - gates_t) * h_{t-1} + gates_t * cand_t
        for (size_t j = 0; j < hidden_size_; ++j) {
            double g = gates_[0][j];
            h_[0][j] = (1.0 - g) * h_[0][j] + g * cand_[0][j];
            hidden_states_[t + 1][j] = h_[0][j];
        }
    }
    last_output_ = Tensor(seq_len, hidden_size_);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t j = 0; j < hidden_size_; ++j)
            last_output_[t][j] = hidden_states_[t + 1][j];
    return last_output_;
}

Tensor MinGRU::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t h = hidden_size_;

    // Single-step mode: return zero input gradient of shape (1, input_dim_).
    if (inputs_.rows == 0 || inputs_.cols == 0) {
        grad_W_g_.fill(0.0); grad_b_g_.fill(0.0);
        grad_W_h_.fill(0.0); grad_b_h_.fill(0.0);
        return Tensor(1, input_dim_);
    }

    size_t seq_len = inputs_.rows;
    Tensor grad_input(seq_len, input_dim_);
    grad_input.fill(0.0);

    grad_W_g_.fill(0.0); grad_b_g_.fill(0.0);
    grad_W_h_.fill(0.0); grad_b_h_.fill(0.0);

    Tensor grad_h(1, h);
    grad_h.fill(0.0);

    // BPTT from t = seq_len-1 down to 0
    for (size_t t = seq_len; t-- > 0; ) {
        // Recompute per-timestep caches from forward_sequence
        Tensor h_t(1, h);
        Tensor h_prev_t(1, h);
        for (size_t j = 0; j < h; ++j) {
            h_t[0][j]      = hidden_states_[t + 1][j];
            h_prev_t[0][j] = hidden_states_[t][j];
        }
        // Per-timestep caches for backward
        Tensor gates_t(1, h);
        Tensor cand_t(1, h);
        for (size_t j = 0; j < h; ++j) {
            gates_t[0][j] = cached_gates_[t][j];
            cand_t[0][j]  = cached_cand_[t][j];
        }

        if (t == seq_len - 1) {
            for (size_t j = 0; j < h; ++j) grad_h[0][j] = grad_output[t][j];
        } else {
            // Add the direct loss contribution at intermediate timesteps.
            for (size_t j = 0; j < h; ++j) grad_h[0][j] += grad_output[t][j];
        }

        // d/dg_t [h_t = (1-g_t)*h_prev_t + g_t*c_t] = c_t - h_prev_t
        // d/dc_t [h_t] = g_t
        // d/dh_prev_t (direct path) = 1 - g_t
        //
        // Gradients:
        //   dL/dg_t = grad_h * (c_t - h_prev_t)
        //   dL/dc_t = grad_h * g_t
        Tensor grad_g(1, h), grad_c(1, h);
        for (size_t j = 0; j < h; ++j) {
            grad_g[0][j] = grad_h[0][j] * (cand_t[0][j] - h_prev_t[0][j]);
            grad_c[0][j] = grad_h[0][j] * gates_t[0][j];
        }

        // dL/d(gate_logit) = grad_g * g * (1 - g)
        Tensor grad_gl(1, h);
        for (size_t j = 0; j < h; ++j) {
            double g = gates_t[0][j];
            grad_gl[0][j] = grad_g[0][j] * g * (1.0 - g);
        }
        // dL/dc_t is already direct (candidate has no nonlinearity in MinGRU).

        // --- Parameter gradients ---
        // W_g (input_dim, h): x_t^T * grad_gl
        Tensor x_t = inputs_.get_row(t);
        for (size_t i = 0; i < input_dim_; ++i)
            for (size_t j = 0; j < h; ++j)
                grad_W_g_[i][j] += x_t[0][i] * grad_gl[0][j];
        for (size_t j = 0; j < h; ++j)
            grad_b_g_[0][j] += grad_gl[0][j];

        // W_h (input_dim, h): x_t^T * grad_c
        for (size_t i = 0; i < input_dim_; ++i)
            for (size_t j = 0; j < h; ++j)
                grad_W_h_[i][j] += x_t[0][i] * grad_c[0][j];
        for (size_t j = 0; j < h; ++j)
            grad_b_h_[0][j] += grad_c[0][j];

        // --- Backprop to h_prev ---
        // Direct path: grad_h * (1 - g)
        Tensor grad_h_prev(1, h);
        for (size_t j = 0; j < h; ++j) {
            grad_h_prev[0][j] = grad_h[0][j] * (1.0 - gates_t[0][j]);
        }

        // --- Backprop to input x_t ---
        // x_t appears in both the gate and candidate pre-activations.
        for (size_t j = 0; j < input_dim_; ++j) {
            double gx = 0.0;
            for (size_t k = 0; k < h; ++k) {
                gx += grad_gl[0][k] * W_g_[j][k];
                gx += grad_c[0][k]  * W_h_[j][k];
            }
            grad_input[t][j] = gx;
        }

        grad_h = grad_h_prev;
    }

    return grad_input;
}

void MinGRU::update_weights(double learning_rate) {
    auto apply = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w[i][j] -= learning_rate * gw[i][j];
    };
    apply(W_g_, grad_W_g_); apply(b_g_, grad_b_g_);
    apply(W_h_, grad_W_h_); apply(b_h_, grad_b_h_);
}

std::vector<Tensor*> MinGRU::parameters() {
    return { &W_g_, &b_g_, &W_h_, &b_h_ };
}

std::vector<Tensor*> MinGRU::gradients() {
    return { &grad_W_g_, &grad_b_g_, &grad_W_h_, &grad_b_h_ };
}
