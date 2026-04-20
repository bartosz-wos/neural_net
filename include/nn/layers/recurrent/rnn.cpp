#include "rnn.h"
#include <cmath>
#include <random>
#include <stdexcept>

SimpleRNN::SimpleRNN(int in_dim, int hid, int seq)
    : input_dim(in_dim), hidden_size(hid), seq_len(seq),
      W_xh(hid, in_dim), W_hh(hid, hid), b(hid, 1),
      grad_W_xh(hid, in_dim), grad_W_hh(hid, hid), grad_b(hid, 1)
{
    double scale_xh = std::sqrt(2.0 / (in_dim + hid));
    double scale_hh = std::sqrt(2.0 / (hid + hid));
    std::mt19937 gen(42);
    std::normal_distribution<> dis_xh(0.0, scale_xh);
    std::normal_distribution<> dis_hh(0.0, scale_hh);
    for (int i = 0; i < hid; ++i) {
        for (int j = 0; j < in_dim; ++j) W_xh[i][j] = dis_xh(gen);
        for (int j = 0; j < hid; ++j) W_hh[i][j] = dis_hh(gen);
        b[i][0] = 0.0;
    }
    grad_W_xh.fill(0.0);
    grad_W_hh.fill(0.0);
    grad_b.fill(0.0);
}

Tensor SimpleRNN::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != seq_len * input_dim) {
        throw std::invalid_argument("SimpleRNN: input dimension mismatch");
    }

    // Cache original input
    inputs = Tensor(N, seq_len * input_dim);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < seq_len * input_dim; ++j)
            inputs[i][j] = input[i][j];

    // Allocate hidden_states cache: (seq_len+1) * N rows, hidden_size cols
    hidden_states = Tensor((seq_len + 1) * N, hidden_size);
    // Initialize h0 = 0
    for (int i = 0; i < N; ++i)
        for (int h = 0; h < hidden_size; ++h)
            hidden_states[i][h] = 0.0; // row i (t=0)

    // Forward over time steps
    for (int t = 0; t < seq_len; ++t) {
        // Build x_t (N, input_dim) from inputs
        Tensor x_t(N, input_dim);
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                x_t[i][d] = inputs[i][t * input_dim + d];
            }
        }

        // Get h_prev from hidden_states row t*N + i
        Tensor h_prev(N, hidden_size);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                h_prev[i][h] = hidden_states[t * N + i][h];
            }
        }

        // Linear: x_t * W_xh^T + h_prev * W_hh^T + b
        Tensor lin = x_t * W_xh.transpose() + h_prev * W_hh.transpose();
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                lin[i][h] += b[h][0];
            }
        }

        // tanh activation
        Tensor h_t = lin.apply([](double x) { return std::tanh(x); });

        // Store h_t into hidden_states at row (t+1)*N + i
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                hidden_states[(t+1) * N + i][h] = h_t[i][h];
            }
        }
    }

    // Output: final hidden state (last N rows)
    Tensor output(N, hidden_size);
    for (int i = 0; i < N; ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            output[i][h] = hidden_states[seq_len * N + i][h];
        }
    }
    return output;
}

Tensor SimpleRNN::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    if (grad_output.cols != hidden_size) {
        throw std::invalid_argument("SimpleRNN: grad_output dimension mismatch");
    }

    // Zero gradients
    grad_W_xh.fill(0.0);
    grad_W_hh.fill(0.0);
    grad_b.fill(0.0);

    // grad_input: same shape as original input (N, seq_len * input_dim)
    Tensor grad_input(N, seq_len * input_dim);
    grad_input.fill(0.0);

    // grad_h for the last time step (t = seq_len)
    Tensor grad_h(N, hidden_size);
    for (int i = 0; i < N; ++i)
        for (int h = 0; h < hidden_size; ++h)
            grad_h[i][h] = grad_output[i][h];

    // BPTT: from t = seq_len-1 down to 0
    for (int t = seq_len - 1; t >= 0; --t) {
        // Retrieve h_t = hidden_states[(t+1)*N + i]
        Tensor h_t(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                h_t[i][h] = hidden_states[(t+1) * N + i][h];

        // Retrieve h_prev = hidden_states[t*N + i]
        Tensor h_prev(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                h_prev[i][h] = hidden_states[t * N + i][h];

        // Retrieve x_t from inputs: slice columns [t*input_dim, (t+1)*input_dim)
        Tensor x_t(N, input_dim);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < input_dim; ++d)
                x_t[i][d] = inputs[i][t * input_dim + d];

        // Derivative of tanh at h_t: 1 - h_t^2
        Tensor d_tanh(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h) {
                double v = h_t[i][h];
                d_tanh[i][h] = 1.0 - v * v;
            }

        // grad_pre = grad_h * d_tanh (elementwise)
        Tensor grad_pre(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_pre[i][h] = grad_h[i][h] * d_tanh[i][h];

        // Gradients: grad_W_xh += grad_pre^T * x_t
        Tensor inc_W_xh = grad_pre.transpose() * x_t; // (H,N) * (N,D) = (H,D)
        for (int i = 0; i < grad_W_xh.rows; ++i)
            for (int j = 0; j < grad_W_xh.cols; ++j)
                grad_W_xh[i][j] += inc_W_xh[i][j];

        // grad_W_hh += grad_pre^T * h_prev
        Tensor inc_W_hh = grad_pre.transpose() * h_prev; // (H,N)*(N,H) = (H,H)
        for (int i = 0; i < grad_W_hh.rows; ++i)
            for (int j = 0; j < grad_W_hh.cols; ++j)
                grad_W_hh[i][j] += inc_W_hh[i][j];

        // grad_b += sum over batch of grad_pre (reduce rows)
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_b[h][0] += grad_pre[i][h];

        // grad_h for previous step: grad_h += grad_pre * W_hh
        // += because upstream grad_h (from output or next layer) carries additional signal
        grad_h = grad_pre * W_hh + grad_h; // (N,H)*(H,H) = (N,H)

        // grad_x_t = grad_pre * W_xh
        Tensor grad_x_t = grad_pre * W_xh; // (N,D)

        // Accumulate into grad_input at columns [t*input_dim, (t+1)*input_dim)
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < input_dim; ++d)
                grad_input[i][t * input_dim + d] += grad_x_t[i][d];
    }

    return grad_input;
}

void SimpleRNN::update_weights(double learning_rate) {
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            W_xh[i][j] -= learning_rate * grad_W_xh[i][j];
        }
        for (int j = 0; j < hidden_size; ++j) {
            W_hh[i][j] -= learning_rate * grad_W_hh[i][j];
        }
        b[i][0] -= learning_rate * grad_b[i][0];
    }
    // gradients will be zeroed by zero_grad() or at start of backprop
}

std::vector<Tensor*> SimpleRNN::parameters() {
    return {&W_xh, &W_hh, &b};
}

std::vector<Tensor*> SimpleRNN::gradients() {
    return {&grad_W_xh, &grad_W_hh, &grad_b};
}

void SimpleRNN::zero_grad() {
    grad_W_xh.fill(0.0);
    grad_W_hh.fill(0.0);
    grad_b.fill(0.0);
}

Tensor SimpleRNN::get_weights() const { return Tensor(0,0); }
Tensor SimpleRNN::get_gradients() const { return Tensor(0,0); }
