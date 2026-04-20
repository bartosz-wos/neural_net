#include "lstm.h"
#include <cmath>
#include <random>
#include <stdexcept>

LSTM::LSTM(int in_dim, int hid, int seq)
    : input_dim(in_dim), hidden_size(hid), seq_len(seq),
      W(4 * hid, hid + in_dim), b(4 * hid, 1),
      grad_W(4 * hid, hid + in_dim), grad_b(4 * hid, 1)
{
    // Xavier/Glorot init
    double scale = std::sqrt(2.0 / (in_dim + hid));
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);

    for (int i = 0; i < 4 * hid; ++i) {
        for (int j = 0; j < hid + in_dim; ++j) {
            W[i][j] = dis(gen);
        }
        b[i][0] = 0.0;
    }

    // Initialize forget gate biases to 1.0 (helps gradient flow early on)
    for (int i = hid; i < 2 * hid; ++i) {
        b[i][0] = 1.0;
    }

    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

Tensor LSTM::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != seq_len * input_dim) {
        throw std::invalid_argument("LSTM: input dimension mismatch");
    }

    // Cache raw input
    inputs = Tensor(N, seq_len * input_dim);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < seq_len * input_dim; ++j)
            inputs[i][j] = input[i][j];

    // Allocate hidden and cell state caches: (seq_len+1)*N rows, hidden cols each
    h_states = Tensor((seq_len + 1) * N, hidden_size);
    c_states = Tensor((seq_len + 1) * N, hidden_size);
    h_states.fill(0.0);
    c_states.fill(0.0);

    // Lazily initialize reusable buffers to match batch size N
    if (buf_x_t_.rows != (size_t)N || buf_x_t_.cols != (size_t)input_dim) {
        buf_x_t_    = Tensor(N, input_dim);
        buf_h_prev_ = Tensor(N, hidden_size);
        buf_c_prev_ = Tensor(N, hidden_size);
        buf_h_x_    = Tensor(N, hidden_size + input_dim);
        buf_gate_pre_ = Tensor(N, 4 * hidden_size);
        buf_i_gate_ = Tensor(N, hidden_size);
        buf_f_gate_ = Tensor(N, hidden_size);
        buf_o_gate_ = Tensor(N, hidden_size);
        buf_g_cand_ = Tensor(N, hidden_size);
    }

    // Forward through time
    for (int t = 0; t < seq_len; ++t) {
        // Extract x_t in-place into buf_x_t_
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < input_dim; ++d)
                buf_x_t_[i][d] = inputs[i][t * input_dim + d];

        // Extract h_prev in-place into buf_h_prev_
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                buf_h_prev_[i][h] = h_states[t * N + i][h];

        // Extract c_prev in-place into buf_c_prev_
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                buf_c_prev_[i][h] = c_states[t * N + i][h];

        // Concatenate [h_prev, x_t] in-place into buf_h_x_
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) buf_h_x_[i][h] = buf_h_prev_[i][h];
            for (int d = 0; d < input_dim; ++d) buf_h_x_[i][hidden_size + d] = buf_x_t_[i][d];
        }

        // Compute gate pre-activations: buf_h_x_ * W^T + b  ->  buf_gate_pre_
        // buf_gate_pre_ = buf_h_x_ * W.transpose()
        {
            const Tensor& hx = buf_h_x_;
            const Tensor& Wt = W;  // W is (4H, H+in), already stored as-is
            int H4 = 4 * hidden_size;
            int Hin = hidden_size + input_dim;
            // buf_gate_pre_[i][g] = sum_k hx[i][k] * W[g][k]
            for (int i = 0; i < N; ++i) {
                for (int g = 0; g < H4; ++g) {
                    double acc = b[g][0];
                    for (int k = 0; k < Hin; ++k)
                        acc += hx[i][k] * W[g][k];
                    buf_gate_pre_[i][g] = acc;
                }
            }
        }

        // Apply activations in-place into gate buffers
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                buf_i_gate_[i][h] = 1.0 / (1.0 + std::exp(-buf_gate_pre_[i][h]));
                buf_f_gate_[i][h] = 1.0 / (1.0 + std::exp(-buf_gate_pre_[i][hidden_size + h]));
                buf_o_gate_[i][h] = 1.0 / (1.0 + std::exp(-buf_gate_pre_[i][2 * hidden_size + h]));
                buf_g_cand_[i][h] = std::tanh(buf_gate_pre_[i][3 * hidden_size + h]);
            }
        }

        // c_t = f * c_prev + i * g_cand  (overwrite buf_c_prev_ as c_t)
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                buf_c_prev_[i][h] = buf_f_gate_[i][h] * buf_c_prev_[i][h]
                                 + buf_i_gate_[i][h] * buf_g_cand_[i][h];

        // h_t = o * tanh(c_t)  (reuse buf_h_prev_ as h_t)
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                buf_h_prev_[i][h] = buf_o_gate_[i][h] * std::tanh(buf_c_prev_[i][h]);

        // Store h_t and c_t into state caches
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h) {
                h_states[(t+1) * N + i][h] = buf_h_prev_[i][h];
                c_states[(t+1) * N + i][h] = buf_c_prev_[i][h];
            }
    }

    // Output = final hidden state h_L  (reuse buf_h_prev_ as output)
    for (int i = 0; i < N; ++i)
        for (int h = 0; h < hidden_size; ++h)
            buf_h_prev_[i][h] = h_states[seq_len * N + i][h];
    last_output_ = buf_h_prev_;
    return last_output_;
}

Tensor LSTM::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    if (grad_output.cols != hidden_size) {
        throw std::invalid_argument("LSTM: grad_output dimension mismatch");
    }

    // Zero gradients
    grad_W.fill(0.0);
    grad_b.fill(0.0);
    Tensor grad_input(N, seq_len * input_dim);
    grad_input.fill(0.0);

    // grad_h at final step = grad_output
    Tensor grad_h(N, hidden_size);
    for (int i = 0; i < N; ++i)
        for (int h = 0; h < hidden_size; ++h)
            grad_h[i][h] = grad_output[i][h];

    // Backward through time: t = L-1 ... 0
    for (int t = seq_len - 1; t >= 0; --t) {
        // FIX: Create zero-copy views into cached state buffers.
        // Use the public data pointer to construct lightweight references without allocation.
        double* h_t_ptr  = h_states.data.data() + (t+1) * N * hidden_size;
        double* c_t_ptr  = c_states.data.data() + (t+1) * N * hidden_size;
        double* h_prev_ptr = h_states.data.data() + t * N * hidden_size;
        double* c_prev_ptr = c_states.data.data() + t * N * hidden_size;
        // h_t(row, col) = h_t_ptr[row * hidden_size + col], etc.

        // Retrieve x_t (still needs a copy since we extract from packed inputs)
        Tensor x_t(N, input_dim);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < input_dim; ++d)
                x_t[i][d] = inputs[i][t * input_dim + d];

        // Compute gate values again (need them for derivatives)
        // Reconstruct h_x = [h_prev, x_t]
        Tensor h_x(N, hidden_size + input_dim);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) h_x[i][h] = h_prev_ptr[i * hidden_size + h];
            for (int d = 0; d < input_dim; ++d) h_x[i][hidden_size + d] = x_t[i][d];
        }
        Tensor gate_pre = h_x * W.transpose();
        for (int i = 0; i < N; ++i)
            for (int g = 0; g < 4 * hidden_size; ++g)
                gate_pre[i][g] += b[g][0];

        Tensor i_gate(N, hidden_size), f_gate(N, hidden_size), o_gate(N, hidden_size), g_cand(N, hidden_size);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                i_gate[i][h] = 1.0 / (1.0 + std::exp(-gate_pre[i][h]));
                f_gate[i][h] = 1.0 / (1.0 + std::exp(-gate_pre[i][hidden_size + h]));
                o_gate[i][h] = 1.0 / (1.0 + std::exp(-gate_pre[i][2 * hidden_size + h]));
                g_cand[i][h] = std::tanh(gate_pre[i][3 * hidden_size + h]);
            }
        }

        // Derivatives of gate activations
        Tensor d_i(N, hidden_size), d_f(N, hidden_size), d_o(N, hidden_size), d_g(N, hidden_size);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                d_i[i][h] = i_gate[i][h] * (1.0 - i_gate[i][h]);
                d_f[i][h] = f_gate[i][h] * (1.0 - f_gate[i][h]);
                d_o[i][h] = o_gate[i][h] * (1.0 - o_gate[i][h]);
                d_g[i][h] = 1.0 - g_cand[i][h] * g_cand[i][h];
            }
        }

        // tanh(c_t) and its derivative
        Tensor tanh_c(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                tanh_c[i][h] = std::tanh(c_t_ptr[i * hidden_size + h]);
        Tensor d_tanh_c(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                d_tanh_c[i][h] = 1.0 - tanh_c[i][h] * tanh_c[i][h];

        // dL/dc_t: comes from h_t (via o_t * tanh) and from next cell (c_{t+1} via forget gate).
        // We accumulate it here; the contribution from the next cell's forget gate is added
        // at the end of this BPTT step when we propagate grad_c backward.
        Tensor grad_c(N, hidden_size);

        // First, compute grad from h_t: dL/dh_t = grad_h (already)
        // dL/do_t = dL/dh_t * tanh(c_t)
        Tensor grad_o_pre(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_o_pre[i][h] = grad_h[i][h] * tanh_c[i][h];

        // dL/dc_t += dL/dh_t * o_t * d_tanh(c_t)
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_c[i][h] += grad_h[i][h] * o_gate[i][h] * d_tanh_c[i][h];

        // dL/dg_cand = dL/dc_t * i_t
        Tensor grad_g_pre(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_g_pre[i][h] = grad_c[i][h] * i_gate[i][h];

        // dL/di_t = dL/dc_t * g_cand
        Tensor grad_i_pre(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_i_pre[i][h] = grad_c[i][h] * g_cand[i][h];

        // dL/df_t = dL/dc_t * c_prev
        Tensor grad_f_pre(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_f_pre[i][h] = grad_c[i][h] * c_prev_ptr[i * hidden_size + h];

        // Now, we need to add contribution of c_t to next cell c_{t+1} via forget gate?
        // Actually in BPTT for LSTM, we also need to propagate through the cell state chain.
        // At time t, c_t influences c_{t+1} via f_{t+1}. We'll add that when we process t+1 (since we go backward).
        // So we need to accumulate grad_c backward: when we move to t-1, we will add f_t * current_grad_c to grad_c_prev.
        // Let's set that after computing grad_h for previous step.

        // Gradients w.r.t. gate pre-activations (before non-linearity):
        // dW_i, db_i accumulate: (d_i_pre)^T * h_x
        Tensor grad_i_final(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_i_final[i][h] = grad_i_pre[i][h] * d_i[i][h];
        Tensor grad_f_final(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_f_final[i][h] = grad_f_pre[i][h] * d_f[i][h];
        Tensor grad_o_final(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_o_final[i][h] = grad_o_pre[i][h] * d_o[i][h];
        Tensor grad_g_final(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_g_final[i][h] = grad_g_pre[i][h] * d_g[i][h];

        // Stack gradients for the 4 gates into a single matrix (N, 4H) to compute param grads via h_x^T
        // But easier: compute each separately: dW_i = (grad_i_final)^T * h_x, etc.

        // dW_i: (H,N) * (N, H+in) = (H, H+in)
        Tensor inc_i = grad_i_final.transpose() * h_x;
        Tensor inc_f = grad_f_final.transpose() * h_x;
        Tensor inc_o = grad_o_final.transpose() * h_x;
        Tensor inc_g = grad_g_final.transpose() * h_x;

        // Accumulate param gradients by placing in appropriate rows of grad_W (rows 0:H, H:2H, 2H:3H, 3H:4H)
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size + input_dim; ++j) {
                grad_W[i][j] += inc_i[i][j];
                grad_W[hidden_size + i][j] += inc_f[i][j];
                grad_W[2 * hidden_size + i][j] += inc_o[i][j];
                grad_W[3 * hidden_size + i][j] += inc_g[i][j];
            }
        }

        // db: sum over batch for each gate
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                grad_b[h][0] += grad_i_final[i][h];
                grad_b[hidden_size + h][0] += grad_f_final[i][h];
                grad_b[2 * hidden_size + h][0] += grad_o_final[i][h];
                grad_b[3 * hidden_size + h][0] += grad_g_final[i][h];
            }
        }

        // Backprop to h_prev and x_t:
        // grad_h_prev = (grad_i_pre * W_i_hprev_part) + (grad_f_pre * W_f_hprev_part) + (grad_o_pre * W_o_hprev_part) + (grad_g_pre * W_g_hprev_part)
        // where each W_* has shape (H, H) corresponding to slice of W^T.
        // d_h_x = [grad_i_pre, grad_f_pre, grad_o_pre, grad_g_pre] * W (since gate_pre = h_x * W^T, grad w.r.t h_x = grad_gate * W)
        // We have grad_gate stacked: (N, 4H). Compute: grad_h_x = grad_gate * W
        // Construct grad_gate_mat (N, 4H)
        Tensor grad_gate_mat(N, 4 * hidden_size);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) {
                grad_gate_mat[i][h] = grad_i_final[i][h];
                grad_gate_mat[i][hidden_size + h] = grad_f_final[i][h];
                grad_gate_mat[i][2 * hidden_size + h] = grad_o_final[i][h];
                grad_gate_mat[i][3 * hidden_size + h] = grad_g_final[i][h];
            }
        }
        Tensor grad_h_x = grad_gate_mat * W; // (N, H+in)

        // Split into grad_h_prev and grad_x_t
        Tensor grad_h_prev(N, hidden_size);
        Tensor grad_x_t(N, input_dim);
        for (int i = 0; i < N; ++i) {
            for (int h = 0; h < hidden_size; ++h) grad_h_prev[i][h] = grad_h_x[i][h];
            for (int d = 0; d < input_dim; ++d) grad_x_t[i][d] = grad_h_x[i][hidden_size + d];
        }

        // Also, grad_h_prev gets contribution from next cell's forget? Actually we already included all through gate_pre chain.
        // There's no direct path from c_{t+1} to h_prev (only through c_t and then to h_t, which we've accounted via grad_h at t+1).
        // So grad_h_prev is correct.

        // Accumulate grad_input at columns t*input_dim .. (t+1)*input_dim
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < input_dim; ++d)
                grad_input[i][t * input_dim + d] += grad_x_t[i][d];

        // FIX (Bug 11): Save the original grad_c BEFORE applying forget gate.
        // The current grad_c was used to compute gate gradients (i,f,o,g) above.
        // grad_c_prev = grad_c * f_gate flows to the previous cell; the current
        // grad_c stays intact for any downstream uses in this iteration.
        Tensor grad_c_prev(N, hidden_size);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hidden_size; ++h)
                grad_c_prev[i][h] = grad_c[i][h] * f_gate[i][h];

        // grad_h for next iteration (t-1) comes from backprop through gates
        grad_h = grad_h_prev;
        grad_c = grad_c_prev;
    }

    return grad_input;
}

void LSTM::update_weights(double learning_rate) {
    for (int i = 0; i < 4 * hidden_size; ++i) {
        for (int j = 0; j < hidden_size + input_dim; ++j) {
            W[i][j] -= learning_rate * grad_W[i][j];
        }
        b[i][0] -= learning_rate * grad_b[i][0];
    }
    // gradients will be zeroed by zero_grad()
}

std::vector<Tensor*> LSTM::parameters() {
    return {&W, &b};
}

std::vector<Tensor*> LSTM::gradients() {
    return {&grad_W, &grad_b};
}

void LSTM::zero_grad() {
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}
