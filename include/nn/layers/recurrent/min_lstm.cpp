#include "min_lstm.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace {
inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
}

MinLSTM::MinLSTM(size_t input_dim, size_t hidden_size)
    : input_dim_(input_dim), hidden_size_(hidden_size), seq_len_(0),
      W_f_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      b_f_(1, hidden_size ? hidden_size : 1),
      W_i_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      b_i_(1, hidden_size ? hidden_size : 1),
      W_h_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      b_h_(1, hidden_size ? hidden_size : 1),
      grad_W_f_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      grad_b_f_(1, hidden_size ? hidden_size : 1),
      grad_W_i_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      grad_b_i_(1, hidden_size ? hidden_size : 1),
      grad_W_h_(input_dim ? input_dim : 1, hidden_size ? hidden_size : 1),
      grad_b_h_(1, hidden_size ? hidden_size : 1),
      h_(1, hidden_size ? hidden_size : 1),
      last_output_(1, hidden_size ? hidden_size : 1) {
    // Validate AFTER the initializer list so the dummy dimensions above keep
    // the Tensor constructors well-formed (no zero-size allocations).
    if (input_dim == 0) throw std::invalid_argument("MinLSTM: input_dim must be > 0");
    if (hidden_size == 0) throw std::invalid_argument("MinLSTM: hidden_size must be > 0");
    init_weights();
    zero_grad();
    h_.fill(0.0);
}

void MinLSTM::init_weights() {
    // Xavier-uniform for the three input projections, each with its own stream
    // so W_f, W_i and W_h are genuinely different matrices (a uniform or shared
    // init would make gradient checks degenerate — see TDD skill notes).
    double scale = std::sqrt(6.0 / (double)(input_dim_ + hidden_size_));
    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis(-scale, scale);
    for (size_t i = 0; i < W_f_.rows; ++i)
        for (size_t j = 0; j < W_f_.cols; ++j) W_f_[i][j] = dis(gen);
    for (size_t i = 0; i < W_i_.rows; ++i)
        for (size_t j = 0; j < W_i_.cols; ++j) W_i_[i][j] = dis(gen);
    for (size_t i = 0; i < W_h_.rows; ++i)
        for (size_t j = 0; j < W_h_.cols; ++j) W_h_[i][j] = dis(gen);

    // Forget-bias-style init: b_f slightly positive, b_i slightly negative, so
    // at init f' ~ 0.62 and i' ~ 0.38 (mild retention preference).
    for (size_t j = 0; j < b_f_.cols; ++j) b_f_[0][j] = 0.5;
    for (size_t j = 0; j < b_i_.cols; ++j) b_i_[0][j] = -0.5;
    for (size_t j = 0; j < b_h_.cols; ++j) b_h_[0][j] = 0.0;
}

void MinLSTM::zero_grad() {
    grad_W_f_.fill(0.0); grad_b_f_.fill(0.0);
    grad_W_i_.fill(0.0); grad_b_i_.fill(0.0);
    grad_W_h_.fill(0.0); grad_b_h_.fill(0.0);
}

Tensor MinLSTM::forward(const Tensor& input) {
    Tensor x = input.rows > 1 ? input.get_row(0) : input;
    Tensor h_prev = h_;
    for (size_t j = 0; j < hidden_size_; ++j) {
        double zf = b_f_[0][j], zi = b_i_[0][j], c = b_h_[0][j];
        for (size_t k = 0; k < input_dim_; ++k) {
            double xk = x[0][k];
            zf += xk * W_f_[k][j];
            zi += xk * W_i_[k][j];
            c  += xk * W_h_[k][j];
        }
        double f = sigmoid(zf), i = sigmoid(zi);
        double s = f + i;
        h_[0][j] = (f / s) * h_prev[0][j] + (i / s) * c;
    }
    // Single-step mode has no BPTT cache; invalidate it so backward() knows.
    inputs_ = Tensor();
    last_output_ = h_;
    return last_output_;
}

Tensor MinLSTM::forward_sequence(const Tensor& seq) {
    size_t T = seq.rows;
    seq_len_ = T;
    inputs_ = seq;
    hidden_states_ = Tensor(T + 1, hidden_size_);
    hidden_states_.fill(0.0);
    cached_f_      = Tensor(T, hidden_size_);
    cached_i_      = Tensor(T, hidden_size_);
    cached_f_norm_ = Tensor(T, hidden_size_);
    cached_i_norm_ = Tensor(T, hidden_size_);
    cached_cand_   = Tensor(T, hidden_size_);

    // Row 0 of hidden_states_ carries the incoming state h_{-1}.
    for (size_t j = 0; j < hidden_size_; ++j) hidden_states_[0][j] = h_[0][j];

    for (size_t t = 0; t < T; ++t) {
        Tensor x = seq.get_row(t);
        for (size_t j = 0; j < hidden_size_; ++j) {
            double zf = b_f_[0][j], zi = b_i_[0][j], c = b_h_[0][j];
            for (size_t k = 0; k < input_dim_; ++k) {
                double xk = x[0][k];
                zf += xk * W_f_[k][j];
                zi += xk * W_i_[k][j];
                c  += xk * W_h_[k][j];
            }
            double f = sigmoid(zf), i = sigmoid(zi);
            double s = f + i;
            double fn = f / s, in = i / s;

            cached_f_[t][j]      = f;
            cached_i_[t][j]      = i;
            cached_f_norm_[t][j] = fn;
            cached_i_norm_[t][j] = in;
            cached_cand_[t][j]   = c;

            double h_new = fn * hidden_states_[t][j] + in * c;
            hidden_states_[t + 1][j] = h_new;
            h_[0][j] = h_new;
        }
    }

    last_output_ = Tensor(T, hidden_size_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < hidden_size_; ++j)
            last_output_[t][j] = hidden_states_[t + 1][j];
    return last_output_;
}

Tensor MinLSTM::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Single-step mode: no BPTT cache, nothing to propagate.
    if (inputs_.rows == 0 || inputs_.cols == 0) {
        return Tensor(1, input_dim_);
    }

    size_t T = inputs_.rows;
    size_t H = hidden_size_;
    Tensor grad_input(T, input_dim_);
    grad_input.fill(0.0);

    // Gradient carried backwards through the h_{t-1} path.
    std::vector<double> grad_h(H, 0.0);

    for (size_t t = T; t-- > 0; ) {
        // Accumulate the direct loss contribution at this timestep.
        for (size_t j = 0; j < H; ++j) grad_h[j] += grad_output[t][j];

        std::vector<double> dz_f(H), dz_i(H), dc(H), grad_h_prev(H);

        for (size_t j = 0; j < H; ++j) {
            double gh     = grad_h[j];
            double f      = cached_f_[t][j];
            double i      = cached_i_[t][j];
            double fn     = cached_f_norm_[t][j];
            double in     = cached_i_norm_[t][j];
            double c      = cached_cand_[t][j];
            double h_prev = hidden_states_[t][j];
            double s      = f + i;
            double inv_s2 = 1.0 / (s * s);

            // h_t = fn * h_prev + in * c
            double d_fn = gh * h_prev;
            double d_in = gh * c;
            dc[j]       = gh * in;
            grad_h_prev[j] = gh * fn;

            // fn = f/(f+i), in = i/(f+i):
            //   d fn/d f =  i/s²   d fn/d i = -f/s²
            //   d in/d f = -i/s²   d in/d i =  f/s²
            double d_f = (i * inv_s2) * (d_fn - d_in);
            double d_i = (f * inv_s2) * (d_in - d_fn);

            // Through the sigmoids.
            dz_f[j] = d_f * f * (1.0 - f);
            dz_i[j] = d_i * i * (1.0 - i);
        }

        // Parameter gradients (x_t outer dz) and input gradient.
        Tensor x = inputs_.get_row(t);
        for (size_t k = 0; k < input_dim_; ++k) {
            double xk = x[0][k];
            double gx = 0.0;
            for (size_t j = 0; j < H; ++j) {
                grad_W_f_[k][j] += xk * dz_f[j];
                grad_W_i_[k][j] += xk * dz_i[j];
                grad_W_h_[k][j] += xk * dc[j];
                gx += dz_f[j] * W_f_[k][j];
                gx += dz_i[j] * W_i_[k][j];
                gx += dc[j]   * W_h_[k][j];
            }
            grad_input[t][k] = gx;
        }
        for (size_t j = 0; j < H; ++j) {
            grad_b_f_[0][j] += dz_f[j];
            grad_b_i_[0][j] += dz_i[j];
            grad_b_h_[0][j] += dc[j];
        }

        grad_h = grad_h_prev;
    }

    return grad_input;
}

void MinLSTM::update_weights(double learning_rate) {
    auto apply = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w[i][j] -= learning_rate * gw[i][j];
    };
    apply(W_f_, grad_W_f_); apply(b_f_, grad_b_f_);
    apply(W_i_, grad_W_i_); apply(b_i_, grad_b_i_);
    apply(W_h_, grad_W_h_); apply(b_h_, grad_b_h_);
}

std::vector<Tensor*> MinLSTM::parameters() {
    return { &W_f_, &b_f_, &W_i_, &b_i_, &W_h_, &b_h_ };
}

std::vector<Tensor*> MinLSTM::gradients() {
    return { &grad_W_f_, &grad_b_f_, &grad_W_i_, &grad_b_i_, &grad_W_h_, &grad_b_h_ };
}
