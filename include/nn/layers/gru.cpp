#include "gru.h"
#include <cmath>

GRU::GRU(size_t input_dim, size_t hidden_size)
    : input_dim_(input_dim), hidden_size_(hidden_size),
      W_zr_(input_dim, 2 * hidden_size), U_zr_(hidden_size, 2 * hidden_size),
      b_zr_(1, 2 * hidden_size),
      W_h_(input_dim, hidden_size), U_h_(hidden_size, hidden_size),
      b_h_(1, hidden_size),
      grad_W_zr_(input_dim, 2 * hidden_size), grad_U_zr_(hidden_size, 2 * hidden_size),
      grad_b_zr_(1, 2 * hidden_size),
      grad_W_h_(input_dim, hidden_size), grad_U_h_(hidden_size, hidden_size),
      grad_b_h_(1, hidden_size),
      h_(1, hidden_size), z_(1, hidden_size), r_(1, hidden_size), hc_(1, hidden_size),
      last_output_(1, hidden_size) {

    init_weights();
    zero_grad();
}

void GRU::init_weights() {
    double scale = std::sqrt(6.0 / (input_dim_ + hidden_size_));
    for (size_t i = 0; i < W_zr_.rows; ++i)
        for (size_t j = 0; j < W_zr_.cols; ++j)
            W_zr_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
    for (size_t i = 0; i < U_zr_.rows; ++i)
        for (size_t j = 0; j < U_zr_.cols; ++j)
            U_zr_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
    for (size_t j = 0; j < b_zr_.cols; ++j)
        b_zr_[0][j] = 0.0;

    scale = std::sqrt(6.0 / (input_dim_ + hidden_size_));
    for (size_t i = 0; i < W_h_.rows; ++i)
        for (size_t j = 0; j < W_h_.cols; ++j)
            W_h_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
    for (size_t i = 0; i < U_h_.rows; ++i)
        for (size_t j = 0; j < U_h_.cols; ++j)
            U_h_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
    for (size_t j = 0; j < b_h_.cols; ++j)
        b_h_[0][j] = 0.0;
}

void GRU::zero_grad() {
    grad_W_zr_.fill(0); grad_U_zr_.fill(0); grad_b_zr_.fill(0);
    grad_W_h_.fill(0); grad_U_h_.fill(0); grad_b_h_.fill(0);
    h_.fill(0);
}

void GRU::compute_gates(const Tensor& x, const Tensor& h_prev) {
    size_t h = hidden_size_;
    // x: (1, input_dim), h_prev: (1, h)

    // zr = sigmoid(x @ W_zr + h_prev @ U_zr + b_zr)
    // W_zr: (input_dim x 2h), U_zr: (h x 2h), b_zr: (1 x 2h)
    // zr: (1 x 2h)
    Tensor zr(1, 2 * h);
    for (size_t j = 0; j < 2 * h; ++j) {
        double val = b_zr_[0][j];
        for (size_t k = 0; k < input_dim_; ++k)
            val += x[0][k] * W_zr_[k][j];
        for (size_t k = 0; k < h; ++k)
            val += h_prev[0][k] * U_zr_[k][j];
        zr[0][j] = 1.0 / (1.0 + std::exp(-val));
    }
    for (size_t j = 0; j < h; ++j) {
        z_[0][j] = zr[0][j];
        r_[0][j] = zr[0][h + j];
    }

    // Candidate hidden state: hc = tanh(x @ W_h + (r * h_prev) @ U_h + b_h)
    Tensor rh(1, h);
    for (size_t j = 0; j < h; ++j) rh[0][j] = r_[0][j] * h_prev[0][j];

    for (size_t j = 0; j < h; ++j) {
        double val = b_h_[0][j];
        for (size_t k = 0; k < input_dim_; ++k)
            val += x[0][k] * W_h_[k][j];
        for (size_t k = 0; k < h; ++k)
            val += rh[0][k] * U_h_[k][j];
        hc_[0][j] = std::tanh(val);
    }
}

Tensor GRU::forward(const Tensor& input) {
    // Single step: input is (1, input_dim) or (batch, input_dim) — use only first row
    h_prev_ = h_; // save for potential backprop
    Tensor x = input.rows > 1 ? input.get_row(0) : input;
    compute_gates(x, h_);

    // h = (1 - z) * h_prev + z * hc  (element-wise)
    for (size_t j = 0; j < hidden_size_; ++j) {
        h_[0][j] = (1 - z_[0][j]) * h_prev_[0][j] + z_[0][j] * hc_[0][j];
    }
    last_output_ = h_;
    return last_output_;
}

Tensor GRU::forward_sequence(const Tensor& seq) {
    // seq: (seq_len, input_dim) — rows are time steps
    size_t seq_len = seq.rows;
    h_.fill(0);
    last_output_ = Tensor(seq_len, hidden_size_);
    h_prev_ = h_;
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_row = seq.get_row(t);
        compute_gates(x_row, h_);
        for (size_t j = 0; j < hidden_size_; ++j)
            h_[0][j] = (1 - z_[0][j]) * h_prev_[0][j] + z_[0][j] * hc_[0][j];
        for (size_t j = 0; j < hidden_size_; ++j)
            last_output_[t][j] = h_[0][j];
        h_prev_ = h_;
    }
    return last_output_;
}

Tensor GRU::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, input_dim_); // placeholder — real BPTT needed for sequence
}

void GRU::update_weights(double learning_rate) {
    auto apply = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w[i][j] -= learning_rate * gw[i][j];
    };
    apply(W_zr_, grad_W_zr_); apply(U_zr_, grad_U_zr_); apply(b_zr_, grad_b_zr_);
    apply(W_h_, grad_W_h_); apply(U_h_, grad_U_h_); apply(b_h_, grad_b_h_);
}

std::vector<Tensor*> GRU::parameters() {
    return { &W_zr_, &U_zr_, &b_zr_, &W_h_, &U_h_, &b_h_ };
}
std::vector<Tensor*> GRU::gradients() {
    return { &grad_W_zr_, &grad_U_zr_, &grad_b_zr_, &grad_W_h_, &grad_U_h_, &grad_b_h_ };
}
