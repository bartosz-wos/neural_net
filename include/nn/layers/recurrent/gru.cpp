#include "gru.h"
#include <cmath>
#include <algorithm>
#include <random>

GRU::GRU(size_t input_dim, size_t hidden_size)
    : input_dim_(input_dim), hidden_size_(hidden_size), seq_len_(0),
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
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-scale, scale);
    for (size_t i = 0; i < W_zr_.rows; ++i)
        for (size_t j = 0; j < W_zr_.cols; ++j)
            W_zr_[i][j] = dis(gen);
    for (size_t i = 0; i < U_zr_.rows; ++i)
        for (size_t j = 0; j < U_zr_.cols; ++j)
            U_zr_[i][j] = dis(gen);
    for (size_t j = 0; j < b_zr_.cols; ++j)
        b_zr_[0][j] = 0.0;

    scale = std::sqrt(6.0 / (input_dim_ + hidden_size_));
    std::uniform_real_distribution<> dis2(-scale, scale);
    for (size_t i = 0; i < W_h_.rows; ++i)
        for (size_t j = 0; j < W_h_.cols; ++j)
            W_h_[i][j] = dis2(gen);
    for (size_t i = 0; i < U_h_.rows; ++i)
        for (size_t j = 0; j < U_h_.cols; ++j)
            U_h_[i][j] = dis2(gen);
    for (size_t j = 0; j < b_h_.cols; ++j)
        b_h_[0][j] = 0.0;
}

void GRU::zero_grad() {
    grad_W_zr_.fill(0.0); grad_U_zr_.fill(0.0); grad_b_zr_.fill(0.0);
    grad_W_h_.fill(0.0); grad_U_h_.fill(0.0); grad_b_h_.fill(0.0);
    h_.fill(0.0);
}

void GRU::compute_gates(const Tensor& x, const Tensor& h_prev) {
    size_t h = hidden_size_;
    // x: (1, input_dim), h_prev: (1, h)
    // zr = sigmoid(x @ W_zr + h_prev @ U_zr + b_zr)
    // W_zr: (input_dim x 2h), U_zr: (h x 2h), b_zr: (1 x 2h)
    Tensor zr(1, 2 * h);
    for (size_t j = 0; j < 2 * h; ++j) {
        double val = b_zr_[0][j];
        for (size_t k = 0; k < input_dim_; ++k) val += x[0][k] * W_zr_[k][j];
        for (size_t k = 0; k < h; ++k)       val += h_prev[0][k] * U_zr_[k][j];
        zr[0][j] = 1.0 / (1.0 + std::exp(-val));
    }
    for (size_t j = 0; j < h; ++j) {
        z_[0][j] = zr[0][j];
        r_[0][j] = zr[0][h + j];
    }

    // Candidate: hc = tanh(x @ W_h + (r * h_prev) @ U_h + b_h)
    Tensor rh(1, h);
    for (size_t j = 0; j < h; ++j) rh[0][j] = r_[0][j] * h_prev[0][j];
    for (size_t j = 0; j < h; ++j) {
        double val = b_h_[0][j];
        for (size_t k = 0; k < input_dim_; ++k) val += x[0][k] * W_h_[k][j];
        for (size_t k = 0; k < h; ++k)          val += rh[0][k] * U_h_[k][j];
        hc_[0][j] = std::tanh(val);
    }
}

Tensor GRU::forward(const Tensor& input) {
    h_prev_ = h_;
    Tensor x = input.rows > 1 ? input.get_row(0) : input;
    compute_gates(x, h_);
    for (size_t j = 0; j < hidden_size_; ++j)
        h_[0][j] = (1 - z_[0][j]) * h_prev_[0][j] + z_[0][j] * hc_[0][j];
    last_output_ = h_;
    return last_output_;
}

Tensor GRU::forward_sequence(const Tensor& seq) {
    size_t seq_len = seq.rows;
    seq_len_ = seq_len;
    inputs_ = seq;
    hidden_states_ = Tensor(seq_len + 1, hidden_size_);
    hidden_states_.fill(0.0);
    h_prev_ = h_;
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_row = seq.get_row(t);
        compute_gates(x_row, h_);
        for (size_t j = 0; j < hidden_size_; ++j)
            h_[0][j] = (1 - z_[0][j]) * h_prev_[0][j] + z_[0][j] * hc_[0][j];
        for (size_t j = 0; j < hidden_size_; ++j)
            hidden_states_[t + 1][j] = h_[0][j];
        h_prev_ = h_;
    }
    last_output_ = Tensor(seq_len, hidden_size_);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t j = 0; j < hidden_size_; ++j)
            last_output_[t][j] = hidden_states_[t + 1][j];
    return last_output_;
}

Tensor GRU::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t h = hidden_size_;

    // Single-step mode: return zero gradient for input (stateless step)
    if (inputs_.rows == 0 || inputs_.cols == 0) {
        grad_W_zr_.fill(0.0); grad_U_zr_.fill(0.0); grad_b_zr_.fill(0.0);
        grad_W_h_.fill(0.0); grad_U_h_.fill(0.0); grad_b_h_.fill(0.0);
        return Tensor(1, input_dim_);
    }

    // --- Sequence mode (full BPTT) ---
    size_t seq_len = inputs_.rows;
    Tensor grad_h(1, h);
    Tensor grad_input(seq_len, input_dim_);
    grad_input.fill(0.0);

    grad_W_zr_.fill(0.0); grad_U_zr_.fill(0.0); grad_b_zr_.fill(0.0);
    grad_W_h_.fill(0.0); grad_U_h_.fill(0.0); grad_b_h_.fill(0.0);

    // BPTT from t = seq_len-1 down to 0
    for (size_t t = seq_len; t-- > 0; ) {
        // Get cached states
        Tensor h_t(1, h);
        for (size_t j = 0; j < h; ++j) h_t[0][j] = hidden_states_[t + 1][j];
        Tensor h_prev_t(1, h);
        for (size_t j = 0; j < h; ++j) h_prev_t[0][j] = hidden_states_[t][j];
        Tensor x_t = inputs_.get_row(t);

        // Recompute gates for this timestep
        compute_gates(x_t, h_prev_t);

        // Init grad_h from upstream (for last step = grad_output row t)
        if (t == seq_len - 1) grad_h.fill(0.0);
        if (t == seq_len - 1) {
            grad_h = grad_output.get_row(t);
        }

        // Gate derivatives
        Tensor dz(1, h);
        for (size_t j = 0; j < h; ++j) dz[0][j] = z_[0][j] * (1.0 - z_[0][j]);
        Tensor dr(1, h);
        for (size_t j = 0; j < h; ++j) dr[0][j] = r_[0][j] * (1.0 - r_[0][j]);
        Tensor dhc(1, h);
        for (size_t j = 0; j < h; ++j) {
            double t_val = hc_[0][j];
            dhc[0][j] = 1.0 - t_val * t_val;
        }

        // dL/d(zr_pre) = [dL/dz*dz, dL/dr*dr] stacked
        Tensor grad_zr(1, 2 * h);
        for (size_t j = 0; j < h; ++j) {
            double dl_dz = grad_h[0][j] * (hc_[0][j] - h_prev_t[0][j]);
            double dl_dhc = grad_h[0][j] * z_[0][j];
            double dl_dr = dl_dhc * dhc[0][j] * h_prev_t[0][j];
            grad_zr[0][j]     = dl_dz * dz[0][j];
            grad_zr[0][h + j] = dl_dr * dr[0][j];
        }

        // grad_hc_pre = dL/dhc * dhc = grad_h * z * dhc  (1, h)
        Tensor grad_hc_pre(1, h);
        for (size_t j = 0; j < h; ++j)
            grad_hc_pre[0][j] = grad_h[0][j] * z_[0][j] * dhc[0][j];

        // --- Parameter gradients ---
        // W_zr (input_dim, 2h): x_t^T * grad_zr
        for (size_t i = 0; i < input_dim_; ++i) {
            for (size_t j = 0; j < h; ++j) {
                grad_W_zr_[i][j]     += x_t[0][i] * grad_zr[0][j];
                grad_W_zr_[i][h + j] += x_t[0][i] * grad_zr[0][h + j];
            }
        }
        // U_zr (h, 2h): h_prev_t^T * grad_zr
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < h; ++j) {
                grad_U_zr_[i][j]     += h_prev_t[0][i] * grad_zr[0][j];
                grad_U_zr_[i][h + j] += h_prev_t[0][i] * grad_zr[0][h + j];
            }
        }
        for (size_t j = 0; j < 2 * h; ++j)
            grad_b_zr_[0][j] += grad_zr[0][j];

        // W_h (input_dim, h): x_t^T * grad_hc_pre
        for (size_t i = 0; i < input_dim_; ++i)
            for (size_t j = 0; j < h; ++j)
                grad_W_h_[i][j] += x_t[0][i] * grad_hc_pre[0][j];

        // U_h (h, h): (r*h_prev)^T * grad_hc_pre
        Tensor rh(1, h);
        for (size_t j = 0; j < h; ++j) rh[0][j] = r_[0][j] * h_prev_t[0][j];
        for (size_t i = 0; i < h; ++i)
            for (size_t j = 0; j < h; ++j)
                grad_U_h_[i][j] += rh[0][i] * grad_hc_pre[0][j];

        for (size_t j = 0; j < h; ++j)
            grad_b_h_[0][j] += grad_hc_pre[0][j];

        // --- Backprop to h_prev ---
        // 1. Direct path through z: d(h)/d(h_prev) = (1-z)
        Tensor grad_h_prev(1, h);
        for (size_t j = 0; j < h; ++j)
            grad_h_prev[0][j] = grad_h[0][j] * (1.0 - z_[0][j]);

        // 2. Gate pre-activation path: grad_zr * U_zr
        for (size_t j = 0; j < h; ++j) {
            double g = 0.0;
            for (size_t k = 0; k < 2 * h; ++k)
                g += grad_zr[0][k] * U_zr_[j][k];
            grad_h_prev[0][j] += g;
        }

        // 3. hc path: hadamard(grad_hc_pre * U_h^T, r)
        for (size_t j = 0; j < h; ++j) {
            double g = 0.0;
            for (size_t k = 0; k < h; ++k)
                g += grad_hc_pre[0][k] * U_h_[k][j];
            grad_h_prev[0][j] += g * r_[0][j];
        }

        // grad_input for timestep t
        for (size_t j = 0; j < input_dim_; ++j) {
            double gx = 0.0;
            for (size_t k = 0; k < h; ++k) {
                gx += grad_zr[0][k]       * W_zr_[j][k];
                gx += grad_zr[0][h + k]   * W_zr_[j][h + k];
                gx += grad_hc_pre[0][k]   * W_h_[j][k];
            }
            grad_input[t][j] = gx;
        }

        grad_h = grad_h_prev;
    }

    return grad_input;
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
