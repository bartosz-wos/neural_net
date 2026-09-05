// Mogrifier LSTM — Melis et al. ICLR 2020 (https://arxiv.org/abs/1909.09592)
//
// See mogrifier_lstm.h for the math + parameter layout.

#include "mogrifier_lstm.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace {
inline double sigmoid(double z) {
    if (z > 0) {
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    }
}
} // namespace

MogrifierLSTM::MogrifierLSTM(size_t input_dim, size_t hidden_size, size_t num_rounds)
    : input_dim_(input_dim), hidden_size_(hidden_size), num_rounds_(num_rounds) {
    // Validate after default args resolved.
    if (input_dim == 0) throw std::invalid_argument("MogrifierLSTM: input_dim must be > 0");
    if (hidden_size == 0) throw std::invalid_argument("MogrifierLSTM: hidden_size must be > 0");
    if (num_rounds > 5) throw std::invalid_argument("MogrifierLSTM: num_rounds must be <= 5");

    // Stateful h, c.
    h_ = Tensor(1, hidden_size);
    c_ = Tensor(1, hidden_size);
    h_.fill(0.0);
    c_.fill(0.0);
    last_output_ = h_;
    cached_ = false;

    W_ = Tensor(4 * hidden_size, input_dim + hidden_size);
    b_ = Tensor(4 * hidden_size, 1);
    grad_W_ = Tensor(4 * hidden_size, input_dim + hidden_size);
    grad_b_ = Tensor(4 * hidden_size, 1);

    size_t n_Q = (num_rounds + 1) / 2;
    size_t n_R = num_rounds / 2;
    Q_list_.reserve(n_Q);
    R_list_.reserve(n_R);
    grad_Q_list_.reserve(n_Q);
    grad_R_list_.reserve(n_R);
    for (size_t k = 0; k < n_Q; ++k) {
        Q_list_.emplace_back(hidden_size, input_dim);
        grad_Q_list_.emplace_back(hidden_size, input_dim);
    }
    for (size_t k = 0; k < n_R; ++k) {
        R_list_.emplace_back(input_dim, hidden_size);
        grad_R_list_.emplace_back(input_dim, hidden_size);
    }

    init_weights();
    zero_grad();
}

void MogrifierLSTM::init_weights() {
    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis_full(-1.0, 1.0);

    // LSTM combined projection.
    double lstm_scale = std::sqrt(6.0 / (double)(input_dim_ + hidden_size_ + 4 * hidden_size_));
    for (size_t i = 0; i < W_.rows; ++i)
        for (size_t j = 0; j < W_.cols; ++j) {
            double v = dis_full(gen) * lstm_scale;
            W_[i][j] = v;
        }

    // LSTM bias: forget gate slice = 1.0; others = 0.0
    for (size_t i = 0; i < 4 * hidden_size_; ++i) b_[i][0] = 0.0;
    for (size_t j = 0; j < hidden_size_; ++j) b_[hidden_size_ + j][0] = 1.0;

    // Mogrifier Q / R: smaller scale (so initial gates are ~0.5 sigmoid → small
    // initial mogrification).
    double mog_scale = 0.1 * std::sqrt(6.0 / (double)(input_dim_ + hidden_size_));
    for (auto& Q : Q_list_)
        for (size_t i = 0; i < Q.rows; ++i)
            for (size_t j = 0; j < Q.cols; ++j) Q[i][j] = dis_full(gen) * mog_scale;
    for (auto& R : R_list_)
        for (size_t i = 0; i < R.rows; ++i)
            for (size_t j = 0; j < R.cols; ++j) R[i][j] = dis_full(gen) * mog_scale;
}

void MogrifierLSTM::zero_grad() {
    grad_W_.fill(0.0);
    grad_b_.fill(0.0);
    for (auto& g : grad_Q_list_) g.fill(0.0);
    for (auto& g : grad_R_list_) g.fill(0.0);
}

std::vector<Tensor*> MogrifierLSTM::parameters() {
    std::vector<Tensor*> p;
    p.reserve(2 + Q_list_.size() + R_list_.size());
    p.push_back(&W_);
    p.push_back(&b_);
    for (auto& Q : Q_list_) p.push_back(&Q);
    for (auto& R : R_list_) p.push_back(&R);
    return p;
}

std::vector<Tensor*> MogrifierLSTM::gradients() {
    std::vector<Tensor*> g;
    g.reserve(2 + grad_Q_list_.size() + grad_R_list_.size());
    g.push_back(&grad_W_);
    g.push_back(&grad_b_);
    for (auto& gq : grad_Q_list_) g.push_back(&gq);
    for (auto& gr : grad_R_list_) g.push_back(&gr);
    return g;
}

void MogrifierLSTM::reset_state() {
    h_.fill(0.0);
    c_.fill(0.0);
    cached_ = false;
}

// =============================================================================
// Forward
// =============================================================================
Tensor MogrifierLSTM::forward_sequence(const Tensor& seq) {
    assert(seq.cols == input_dim_);
    size_t T = seq.rows;

    // Snapshot input.
    inputs_ = seq;
    h_states_ = Tensor(T + 1, hidden_size_);
    c_states_ = Tensor(T + 1, hidden_size_);
    h_states_.fill(0.0);
    c_states_.fill(0.0);

    // Per-round caches: each round produces a (T, dim) tensor of values per timestep.
// cached_mog_x_[0] is the original x (= seq); cached_mog_x_[k] is x after round k.
// cached_mog_h_[0] is the original h_prev (= 0); cached_mog_h_[k] is h_prev after round k.
    size_t rounds_x = (num_rounds_ + 1) / 2; // number of x-mogrifying rounds (= ceil(r/2))
    size_t rounds_h = num_rounds_ / 2;        // number of h-mogrifying rounds (= floor(r/2))
    cached_mog_x_.clear();
    cached_mog_h_.clear();
    // Allocate (rounds_x + 1) entries for cached_mog_x_ and (rounds_h + 1) for cached_mog_h_,
    // each a (T, dim) tensor. Filled per-step inside the loop.
    cached_mog_x_.resize(rounds_x + 1);
    cached_mog_h_.resize(rounds_h + 1);
    for (auto& tx : cached_mog_x_) {
        tx = Tensor(T, input_dim_);
        tx.fill(0.0);
    }
    for (auto& th : cached_mog_h_) {
        th = Tensor(T, hidden_size_);
        th.fill(0.0);
    }

    cached_gates_ = Tensor(T, 4 * hidden_size_);

    // For each timestep:
    for (size_t t = 0; t < T; ++t) {
        // Step 1: Mogrifier pre-pass. Start with x = seq[t], h_prev = h_states_[t].
        Tensor x_cur = seq.get_row(t); // (1, input)
        Tensor h_prev = h_states_.get_row(t); // (1, hidden)

        // Cache the per-step state at the START of this step's rounds.
        for (size_t d = 0; d < input_dim_; ++d) cached_mog_x_[0][t][d] = x_cur[0][d];
        for (size_t d = 0; d < hidden_size_; ++d) cached_mog_h_[0][t][d] = h_prev[0][d];

        // For each round index k = 0, 1, 2, ..., num_rounds-1:
        //   If k is even (0, 2, 4, ...): x ← 2 σ(h_prev · Q^T) ⊙ x   (x-mogrifying)
        //   If k is odd (1, 3, 5, ...): h_prev ← 2 σ(x · R^T) ⊙ h_prev (h-mogrifying)
        // Note: with Dense layout (out, in), to compute `in · W^T` we compute
        // `sum_k in_k * W[out, k]`. For Q ∈ (hidden, input), "Q^T" maps input → hidden:
        //   gate_x[d] = sum_k h_prev[0][k] * Q[k][d]   where k ∈ hidden, d ∈ input
        // So the matmul "in @ Q^T" with Q of shape (hidden, input):
        //   out[d] = sum_{k ∈ hidden} in_h[k] * Q[k][d]   (in_h is h_prev)
        //
        // For R ∈ (input, hidden), "R^T" maps hidden → input:
        //   gate_h[d] = sum_{k ∈ hidden} x[d] * R[d][k]   where d ∈ input, k ∈ hidden
        // So matmul "x @ R^T" with R of shape (input, hidden):
        //   gate_h[d] = sum_k x[0][d] * R[d][k]   (x is (1, input))
        //
        // In both cases the matmul reduces to "sum_k input[k] * W[k][out_idx]" — standard
        // Dense forward.

        size_t x_round_idx = 0;
        size_t h_round_idx = 0;
        for (size_t k = 0; k < num_rounds_; ++k) {
            if (k % 2 == 0) {
                // x-mogrifying round: x_cur ← 2 σ(h_prev · Q_k^T) ⊙ x_cur
                Tensor& Q = Q_list_[x_round_idx++];
                Tensor gate(input_dim_, 1);
                gate.fill(0.0);
                for (size_t d = 0; d < input_dim_; ++d) {
                    double s = 0.0;
                    for (size_t kk = 0; kk < hidden_size_; ++kk) {
                        s += h_prev[0][kk] * Q[kk][d];
                    }
                    gate[d][0] = 2.0 * sigmoid(s);
                }
                // x_cur[d] *= gate[d]
                for (size_t d = 0; d < input_dim_; ++d) {
                    x_cur[0][d] = x_cur[0][d] * gate[d][0];
                }
                // Store after this round (per-timestep).
                for (size_t d = 0; d < input_dim_; ++d) {
                    cached_mog_x_[x_round_idx][t][d] = x_cur[0][d];
                }
            } else {
                // h-mogrifying round: h_prev ← 2 σ(x_cur · R_k^T) ⊙ h_prev
                Tensor& R = R_list_[h_round_idx++];
                Tensor gate(hidden_size_, 1);
                gate.fill(0.0);
                for (size_t d = 0; d < hidden_size_; ++d) {
                    double s = 0.0;
                    for (size_t kk = 0; kk < input_dim_; ++kk) {
                        s += x_cur[0][kk] * R[kk][d];
                    }
                    gate[d][0] = 2.0 * sigmoid(s);
                }
                for (size_t d = 0; d < hidden_size_; ++d) {
                    h_prev[0][d] = h_prev[0][d] * gate[d][0];
                }
                // Store per-timestep.
                for (size_t d = 0; d < hidden_size_; ++d) {
                    cached_mog_h_[h_round_idx][t][d] = h_prev[0][d];
                }
            }
        }
        // At this point, x_cur is the mogrified input (last x round), h_prev is
        // the mogrified h_prev (last h round). They have the right shapes for the LSTM.

        // Step 2: Standard LSTM gates on the mogrified [x_cur; h_prev].
        // Concatenate [x_cur; h_prev] into a single (1, input + hidden) tensor.
        Tensor xh(input_dim_ + hidden_size_, 1);
        for (size_t d = 0; d < input_dim_; ++d) xh[d][0] = x_cur[0][d];
        for (size_t d = 0; d < hidden_size_; ++d) xh[input_dim_ + d][0] = h_prev[0][d];

        // gate_pre = W · xh + b  (Dense convention: out = in @ W^T)
        // W shape (4·h, input + h). So gate_pre[out] = sum_k xh[k] * W[out][k]
        Tensor gate_pre(4 * hidden_size_, 1);
        for (size_t out = 0; out < 4 * hidden_size_; ++out) {
            double s = b_[out][0];
            for (size_t k = 0; k < input_dim_ + hidden_size_; ++k) {
                s += xh[k][0] * W_[out][k];
            }
            gate_pre[out][0] = s;
        }

        // Cache gate_pre for backward.
        for (size_t out = 0; out < 4 * hidden_size_; ++out) {
            cached_gates_[t][out] = gate_pre[out][0];
        }

        // Apply activations: gate_pre layout = [i (h), f (h), g (h), o (h)]
        // i = σ(z_i), f = σ(z_f), g = tanh(z_g), o = σ(z_o)
        Tensor h_new(1, hidden_size_);
        Tensor c_new(1, hidden_size_);
        Tensor c_prev = c_states_.get_row(t);
        for (size_t j = 0; j < hidden_size_; ++j) {
            double z_i = gate_pre[j][0];
            double z_f = gate_pre[hidden_size_ + j][0];
            double z_g = gate_pre[2 * hidden_size_ + j][0];
            double z_o = gate_pre[3 * hidden_size_ + j][0];
            double i_g = sigmoid(z_i);
            double f_g = sigmoid(z_f);
            double g_g = std::tanh(z_g);
            double o_g = sigmoid(z_o);
            double c_val = f_g * c_prev[0][j] + i_g * g_g;
            double h_val = o_g * std::tanh(c_val);
            h_new[0][j] = h_val;
            c_new[0][j] = c_val;
        }

        // Store in caches.
        for (size_t j = 0; j < hidden_size_; ++j) {
            h_states_[t + 1][j] = h_new[0][j];
            c_states_[t + 1][j] = c_new[0][j];
        }

        // Also update the stateful h_, c_ (for single-step API).
        for (size_t j = 0; j < hidden_size_; ++j) {
            h_[0][j] = h_new[0][j];
            c_[0][j] = c_new[0][j];
        }
    }

    // Build output: (T, hidden_size)
    Tensor out(T, hidden_size_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < hidden_size_; ++j)
            out[t][j] = h_states_[t + 1][j];

    last_output_ = h_;
    cached_ = true;
    return out;
}

Tensor MogrifierLSTM::forward(const Tensor& input) {
    // Single-step forward (no BPTT). Just runs one mogrifier + LSTM step using
    // the current h_, c_ state. Does NOT update caches.
    if (input.rows > 1) {
        // For a single row, treat it as one timestep.
        // Take just the first row.
        Tensor x1 = input.get_row(0);
        // ... handled below via inputs_.rows==0 path
    }
    Tensor x_cur(1, input_dim_);
    if (input.rows == 1) {
        for (size_t d = 0; d < input_dim_; ++d) x_cur[0][d] = input[0][d];
    } else {
        for (size_t d = 0; d < input_dim_; ++d) x_cur[0][d] = input[0][d];
    }
    Tensor h_prev(1, hidden_size_);
    for (size_t d = 0; d < hidden_size_; ++d) h_prev[0][d] = h_[0][d];

    size_t x_round_idx = 0, h_round_idx = 0;
    for (size_t k = 0; k < num_rounds_; ++k) {
        if (k % 2 == 0) {
            Tensor& Q = Q_list_[x_round_idx++];
            Tensor gate(input_dim_, 1);
            for (size_t d = 0; d < input_dim_; ++d) {
                double s = 0.0;
                for (size_t kk = 0; kk < hidden_size_; ++kk) s += h_prev[0][kk] * Q[kk][d];
                gate[d][0] = 2.0 * sigmoid(s);
            }
            for (size_t d = 0; d < input_dim_; ++d) x_cur[0][d] *= gate[d][0];
        } else {
            Tensor& R = R_list_[h_round_idx++];
            Tensor gate(hidden_size_, 1);
            for (size_t d = 0; d < hidden_size_; ++d) {
                double s = 0.0;
                for (size_t kk = 0; kk < input_dim_; ++kk) s += x_cur[0][kk] * R[kk][d];
                gate[d][0] = 2.0 * sigmoid(s);
            }
            for (size_t d = 0; d < hidden_size_; ++d) h_prev[0][d] *= gate[d][0];
        }
    }

    // LSTM gates.
    Tensor xh(input_dim_ + hidden_size_, 1);
    for (size_t d = 0; d < input_dim_; ++d) xh[d][0] = x_cur[0][d];
    for (size_t d = 0; d < hidden_size_; ++d) xh[input_dim_ + d][0] = h_prev[0][d];

    Tensor gate_pre(4 * hidden_size_, 1);
    for (size_t out = 0; out < 4 * hidden_size_; ++out) {
        double s = b_[out][0];
        for (size_t k = 0; k < input_dim_ + hidden_size_; ++k) s += xh[k][0] * W_[out][k];
        gate_pre[out][0] = s;
    }

    for (size_t j = 0; j < hidden_size_; ++j) {
        double z_i = gate_pre[j][0];
        double z_f = gate_pre[hidden_size_ + j][0];
        double z_g = gate_pre[2 * hidden_size_ + j][0];
        double z_o = gate_pre[3 * hidden_size_ + j][0];
        double i_g = sigmoid(z_i);
        double f_g = sigmoid(z_f);
        double g_g = std::tanh(z_g);
        double o_g = sigmoid(z_o);
        double c_val = f_g * c_[0][j] + i_g * g_g;
        double h_val = o_g * std::tanh(c_val);
        h_[0][j] = h_val;
        c_[0][j] = c_val;
    }
    last_output_ = h_;
    cached_ = false;
    return h_;
}

Tensor MogrifierLSTM::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    if (!cached_) {
        throw std::logic_error("MogrifierLSTM::backward requires forward_sequence to populate caches");
    }
    size_t T = grad_output.rows;
    if (T != inputs_.rows) {
        throw std::invalid_argument("MogrifierLSTM::backward: grad_output rows != inputs rows");
    }

    // 1. Standard LSTM backward through gates.
    // For each t, compute gradients to (z_i, z_f, z_g, z_o), then accumulate into
    // grad_W (outer product of [x_cur; h_prev] and the grad-gate vector), grad_b.
    // We also accumulate grad_x_cur[t] and grad_h_prev[t] (the *post-mogrifier*
    // gradients), then back through the mogrifier rounds.
    Tensor grad_inputs(T, input_dim_);
    grad_inputs.fill(0.0);
    Tensor grad_c(1, hidden_size_);
    grad_c.fill(0.0); // accumulated from future timesteps

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        // grad_h[t] = grad_output[t] (the only contribution at this step), plus
        // gh_carrier[t+1] from the next-step's mogrifier. gh_carrier[t+1] holds
        // gh from step t+1 = grad w.r.t. raw h_t (the LSTM's output at this step,
        // which is raw h_t — i.e., the next step's h_prev before mogrifier).
        Tensor grad_h(1, hidden_size_);
        for (size_t j = 0; j < hidden_size_; ++j) {
            grad_h[0][j] = grad_output[t][j];
            if (t + 1 < T) grad_h[0][j] += gh_carrier_[t + 1][j];
        }

        // c_t from cache.
        Tensor c_t = c_states_.get_row(t + 1);
        Tensor c_prev = c_states_.get_row(t);

        // gate_pre[t] from cache (4·hidden entries: i, f, g, o).
        // The cache stores them in the same concatenated layout: [i (h), f (h), g (h), o (h)].
        // Compute activations from gate_pre (for chain rule).
        Tensor grad_z(4 * hidden_size_, 1);
        grad_z.fill(0.0);
        for (size_t j = 0; j < hidden_size_; ++j) {
            double z_i = cached_gates_[t][j];
            double z_f = cached_gates_[t][hidden_size_ + j];
            double z_g = cached_gates_[t][2 * hidden_size_ + j];
            double z_o = cached_gates_[t][3 * hidden_size_ + j];
            double i_g = sigmoid(z_i);
            double f_g = sigmoid(z_f);
            double g_g = std::tanh(z_g);
            double o_g = sigmoid(z_o);

            // grad_c[t] = grad_h ⊙ o ⊙ (1 - tanh²(c_t)) + grad_c (from t+1)
            double grad_c_t_j = grad_h[0][j] * o_g * (1.0 - std::tanh(c_t[0][j]) * std::tanh(c_t[0][j]))
                                + grad_c[0][j];

            // grad_z_o = grad_h ⊙ tanh(c_t) ⊙ o(1-o)
            double tanh_c = std::tanh(c_t[0][j]);
            grad_z[3 * hidden_size_ + j][0] = grad_h[0][j] * tanh_c * o_g * (1.0 - o_g);

            // grad_z_g = grad_c_t ⊙ i ⊙ (1 - g²)
            grad_z[2 * hidden_size_ + j][0] = grad_c_t_j * i_g * (1.0 - g_g * g_g);

            // grad_z_i = grad_c_t ⊙ g ⊙ i(1-i)
            grad_z[j][0] = grad_c_t_j * g_g * i_g * (1.0 - i_g);

            // grad_z_f = grad_c_t ⊙ c_{t-1} ⊙ f(1-f)
            grad_z[hidden_size_ + j][0] = grad_c_t_j * c_prev[0][j] * f_g * (1.0 - f_g);

            // Carry grad_c back to t-1: grad_c_{t-1} = grad_c_t ⊙ f
            grad_c[0][j] = grad_c_t_j * f_g;
        }

        // 2. Accumulate grad_W (outer product of [x_cur; h_prev] with grad_z).
        // x_cur and h_prev are the *mogrified* inputs at step t.
        // We need to retrieve them. The cached_mog_x_ holds x after each round (in
        // round order). The "x_cur used for the LSTM" is the LAST x-mogrifying round's
        // output, which is cached_mog_x_[ceil(num_rounds/2)].
        // The "h_prev used for the LSTM" is the LAST h-mogrifying round's output,
        // cached_mog_h_[floor(num_rounds/2)].
        // Note: rounds_x = ceil(r/2); rounds_h = floor(r/2). cached_mog_x_.size()
        // = rounds_x + 1 (index 0 = original). cached_mog_h_.size() = rounds_h + 1.

        size_t rounds_x = (num_rounds_ + 1) / 2;
        size_t rounds_h = num_rounds_ / 2;

        Tensor x_cur_final(1, input_dim_);
        Tensor h_prev_final(1, hidden_size_);
        for (size_t d = 0; d < input_dim_; ++d) x_cur_final[0][d] = cached_mog_x_[rounds_x][t][d];
        for (size_t d = 0; d < hidden_size_; ++d) h_prev_final[0][d] = cached_mog_h_[rounds_h][t][d];

        // grad_W[out, k] += grad_z[out] * [x; h][k]
        for (size_t out = 0; out < 4 * hidden_size_; ++out) {
            double gz = grad_z[out][0];
            for (size_t d = 0; d < input_dim_; ++d) {
                grad_W_[out][d] += gz * x_cur_final[0][d];
            }
            for (size_t d = 0; d < hidden_size_; ++d) {
                grad_W_[out][input_dim_ + d] += gz * h_prev_final[0][d];
            }
            grad_b_[out][0] += gz;
        }

        // 3. Backprop to x_cur and h_prev (the *mogrified* versions).
        Tensor grad_x_mog(1, input_dim_);
        grad_x_mog.fill(0.0);
        Tensor grad_h_mog(1, hidden_size_);
        grad_h_mog.fill(0.0);
        for (size_t out = 0; out < 4 * hidden_size_; ++out) {
            double gz = grad_z[out][0];
            for (size_t d = 0; d < input_dim_; ++d) {
                grad_x_mog[0][d] += gz * W_[out][d];
            }
            for (size_t d = 0; d < hidden_size_; ++d) {
                grad_h_mog[0][d] += gz * W_[out][input_dim_ + d];
            }
        }

        // 4. Backward through mogrifier rounds (reverse order).
        // Replay forward, but now propagate gradients backward through each gate.
        // cached_mog_x_[k] for k = 0, 1, ..., rounds_x  (index 0 = original seq[t]).
        // cached_mog_h_[k] for k = 0, 1, ..., rounds_h  (index 0 = original h_{t-1}).
        // The forward sequence of operations:
        //   round 1 (k=0, x-mog): x_cur = 2 σ(h_prev Q_0^T) ⊙ x_cur; cached as cached_mog_x_[1]
        //   round 2 (k=1, h-mog): h_prev = 2 σ(x_cur R_0^T) ⊙ h_prev; cached as cached_mog_h_[1]
        //   round 3 (k=2, x-mog): x_cur = 2 σ(h_prev Q_1^T) ⊙ x_cur; cached as cached_mog_x_[2]
        //   round 4 (k=3, h-mog): h_prev = 2 σ(x_cur R_1^T) ⊙ h_prev; cached as cached_mog_h_[2]
        //
        // Backward: start with grad_x_mog (post-x-round, pre-next-h-round) and
        // grad_h_mog (post-h-round, pre-next-x-round). Walk rounds in reverse.
        //
        // For round k (0-indexed) of type:
        //   x-mog (k even): forward: x_after = gate ⊙ x_before
        //                   where gate[d] = 2 σ(h_after_this_round[k-1] Q_k^T)
        //                                 [uses h_after_prev_h_round if k > 0, else h_prev_orig]
        //                                 But cached_mog_h_[k/2] = h before this round.
        //                   actually: gate depends on h_prev which has been *further* mogrified
        //                   by all rounds up to round k-1 that modify h.
        //                   Cache lookup: cached_mog_h_[floor(k/2)] holds the h state
        //                   at the START of round k (i.e., just before round k executes).
        //                   Wait — our cache scheme: cached_mog_x_[k] is x AFTER round 2k-1
        //                   (i.e., after the k-th x-mogrifying round, 1-indexed).
        //                   cached_mog_h_[k] is h AFTER round 2k (i.e., after the k-th
        //                   h-mogrifying round, 1-indexed).
        //                   So round k (0-indexed):
        //                     if k is even: it's the (k/2 + 1)-th x-mogrifying round.
        //                       input h: cached_mog_h_[k/2]    (k/2 h-rounds happened first)
        //                       input x: cached_mog_x_[k/2]    (k/2 x-rounds already done)
        //                       output x: cached_mog_x_[k/2 + 1]
        //                       output h: cached_mog_h_[k/2]   (unchanged)
        //                     if k is odd: it's the (k/2 + 1)-th h-mogrifying round.
        //                       input h: cached_mog_h_[k/2]
        //                       input x: cached_mog_x_[(k+1)/2]  (x not modified since the last x-round)
        //                       output h: cached_mog_h_[k/2 + 1]
        //                       output x: cached_mog_x_[(k+1)/2]  (unchanged)
        // Backward:
        //   x-mog round (k even):
        //     gate_x[d] = 2 σ(sum_k cached_mog_h_[k/2][0][k] * Q[k/2][k][d])
        //     x_out[d] = gate_x[d] * x_in[d]
        //     x_in[d] = cached_mog_x_[k/2][t][d], x_out[d] = cached_mog_x_[k/2+1][t][d]
        //     grad_gate_x[d] = grad_x_out[d] * x_in[d]
        //     grad_h_pre = grad_gate_x @ Q_k    (shape: hidden)
        //     grad_Q_k += h_pre^T @ grad_gate_x  (shape: (hidden, input))
        //     grad_x_in[d] = grad_x_out[d] * gate_x[d]
        //
        //   h-mog round (k odd):
        //     gate_h[d] = 2 σ(sum_k cached_mog_x_[(k+1)/2][t][k] * R[k/2][k][d])
        //     h_out[d] = gate_h[d] * h_in[d]
        //     h_in[d] = cached_mog_h_[k/2][t][d], h_out[d] = cached_mog_h_[k/2+1][t][d]
        //     grad_gate_h[d] = grad_h_out[d] * h_in[d]
        //     grad_x_pre = grad_gate_h @ R_k    (shape: input)
        //     grad_R_k += x_pre^T @ grad_gate_h (shape: (input, hidden))
        //     grad_h_in[d] = grad_h_out[d] * gate_h[d]

        // We work in reverse. Start with grad_x_mog (input to round 2k+1, the LAST
        // x-mogrifying round's output — same as grad_x_mog computed above if rounds_x
        // is the last x-round, else we need to back through h-rounds first).
        //
        // Specifically: grad_x_mog above is the grad to the LAST x-mogrifying round's
        // output (cached_mog_x_[rounds_x]). grad_h_mog is the grad to the LAST
        // h-mogrifying round's output (cached_mog_h_[rounds_h]).
        //
        // Walk rounds in reverse, alternating:
        //   The last round (k = num_rounds - 1):
        //     if k is even: x-mogrifying. grad flows from grad_x_mog through the
        //                   gate to grad_x_in (cached_mog_x_[rounds_x - 1]).
        //                   No grad flows to h (h was input).
        //     if k is odd:  h-mogrifying. grad flows from grad_h_mog through the
        //                   gate to grad_h_in (cached_mog_h_[rounds_h - 1]).
        //                   No grad flows to x (x was input).
        //   Continue until round 0.

        // Run mogrifier backward through all rounds in reverse.
        // grad_to_x_post_last_round = grad_x_mog, grad_to_h_post_last_round = grad_h_mog.
        Tensor& gx = grad_x_mog;
        Tensor& gh = grad_h_mog;

        // Reverse order through rounds. For each round, we use the cached
        // pre-round input x or h, and the cached post-round output, plus the cached
        // pre-round state for the OTHER side (which determines the gate).

        for (int k_signed = (int)num_rounds_ - 1; k_signed >= 0; --k_signed) {
            size_t k = (size_t)k_signed;
            if (k % 2 == 0) {
                // x-mogrifying round k/2.
                size_t xr = k / 2;  // 0-indexed x-round number
                Tensor& Q = Q_list_[xr];
                // h used as input to this round's gate: cached_mog_h_[xr][t]
                // (after xr h-rounds, which precede this x-round).
                Tensor h_for_gate(1, hidden_size_);
                for (size_t j = 0; j < hidden_size_; ++j) h_for_gate[0][j] = cached_mog_h_[xr][t][j];
                // x_in: cached_mog_x_[xr][t]; x_out: cached_mog_x_[xr+1][t].
                Tensor x_in(1, input_dim_);
                Tensor x_out(1, input_dim_);
                for (size_t d = 0; d < input_dim_; ++d) {
                    x_in[0][d] = cached_mog_x_[xr][t][d];
                    x_out[0][d] = cached_mog_x_[xr+1][t][d];
                }
                // Recompute gate (we don't have it cached — small enough to recompute).
                Tensor gate(input_dim_, 1);
                for (size_t d = 0; d < input_dim_; ++d) {
                    double s = 0.0;
                    for (size_t kk = 0; kk < hidden_size_; ++kk) s += h_for_gate[0][kk] * Q[kk][d];
                    gate[d][0] = 2.0 * sigmoid(s);
                }
                // grad_gate[d] = gx[d] * x_in[d]
                Tensor grad_gate(input_dim_, 1);
                for (size_t d = 0; d < input_dim_; ++d) grad_gate[d][0] = gx[0][d] * x_in[0][d];
                // grad_s_x[d] = grad_gate[d] * d gate / d s_x
                //   gate = 2 σ(s), so d gate / d s = 2 σ(s)(1 - σ(s)) = gate (1 - gate/2)
                //   We store gate for clarity.
                Tensor grad_s(input_dim_, 1);
                for (size_t d = 0; d < input_dim_; ++d) {
                    grad_s[d][0] = grad_gate[d][0] * gate[d][0] * (1.0 - 0.5 * gate[d][0]);
                }
                // grad_h_pre += grad_s @ Q   (h_for_gate is the input)
                for (size_t j = 0; j < hidden_size_; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < input_dim_; ++d) s += grad_s[d][0] * Q[j][d];
                    gh[0][j] += s;
                }
                // grad_Q += h_for_gate^T @ grad_s
                for (size_t j = 0; j < hidden_size_; ++j) {
                    for (size_t d = 0; d < input_dim_; ++d) {
                        grad_Q_list_[xr][j][d] += h_for_gate[0][j] * grad_s[d][0];
                    }
                }
                // grad_x_in = gx * gate (x itself propagates unchanged but multiplied by gate)
                Tensor new_gx(1, input_dim_);
                for (size_t d = 0; d < input_dim_; ++d) {
                    new_gx[0][d] = gx[0][d] * gate[d][0];
                }
                gx = new_gx;
            } else {
                // h-mogrifying round (k-1)/2.
                size_t hr = (k - 1) / 2;  // 0-indexed h-round number
                Tensor& R = R_list_[hr];
                // x used as input to this round's gate: cached_mog_x_[hr+1][t]
                // (after hr+1 x-rounds, where the (hr+1)-th just happened).
                Tensor x_for_gate(1, input_dim_);
                for (size_t d = 0; d < input_dim_; ++d) x_for_gate[0][d] = cached_mog_x_[hr+1][t][d];
                // h_in: cached_mog_h_[hr][t]; h_out: cached_mog_h_[hr+1][t].
                Tensor h_in(1, hidden_size_);
                Tensor h_out(1, hidden_size_);
                for (size_t d = 0; d < hidden_size_; ++d) {
                    h_in[0][d] = cached_mog_h_[hr][t][d];
                    h_out[0][d] = cached_mog_h_[hr+1][t][d];
                }
                // Recompute gate.
                Tensor gate(hidden_size_, 1);
                for (size_t d = 0; d < hidden_size_; ++d) {
                    double s = 0.0;
                    for (size_t kk = 0; kk < input_dim_; ++kk) s += x_for_gate[0][kk] * R[kk][d];
                    gate[d][0] = 2.0 * sigmoid(s);
                }
                // grad_gate[d] = gh[d] * h_in[d]
                Tensor grad_gate(hidden_size_, 1);
                for (size_t d = 0; d < hidden_size_; ++d) grad_gate[d][0] = gh[0][d] * h_in[0][d];
                // grad_s_h[d] = grad_gate[d] * d gate / d s_h
                Tensor grad_s(hidden_size_, 1);
                for (size_t d = 0; d < hidden_size_; ++d) {
                    grad_s[d][0] = grad_gate[d][0] * gate[d][0] * (1.0 - 0.5 * gate[d][0]);
                }
                // grad_x_pre += grad_s @ R   (x_for_gate is the input)
                for (size_t j = 0; j < input_dim_; ++j) {
                    double s = 0.0;
                    for (size_t d = 0; d < hidden_size_; ++d) s += grad_s[d][0] * R[j][d];
                    gx[0][j] += s;
                }
                // grad_R += x_for_gate^T @ grad_s
                for (size_t j = 0; j < input_dim_; ++j) {
                    for (size_t d = 0; d < hidden_size_; ++d) {
                        grad_R_list_[hr][j][d] += x_for_gate[0][j] * grad_s[d][0];
                    }
                }
                // grad_h_in = gh * gate
                Tensor new_gh(1, hidden_size_);
                for (size_t d = 0; d < hidden_size_; ++d) {
                    new_gh[0][d] = gh[0][d] * gate[d][0];
                }
                gh = new_gh;
            }
        }

        // After all rounds: grad_inputs[t] += gx (the input grad at timestep t).
        // grad_h_{t-1} is carried via the gh variable — but gh propagates BACKWARD
        // in time. For t > 0, gh is the grad from this timestep's mogrifier w.r.t.
        // h_{t-1}, which adds to the grad_h_{t-1} accumulated from the LSTM step
        // at t-1 (the LSTM's grad_h_mog there). For t = 0, gh has no previous
        // timestep, so it's discarded (or returned as a grad_h_{-1} — we drop it).
        //
        // Note: the LSTM backward's grad_h_prev contributions at t come from the
        // grad_z@W path which is grad_h_mog (already computed at the start of this
        // timestep). For t > 0, the grad from mogrifier's h_{t-1} grad propagates
        // BACK to timestep t-1. We need to add gh (after all rounds) to the LSTM
        // backward's grad_h_prev at the previous timestep. We do this by storing
        // a per-timestep buffer and accumulating.
        // For simplicity, we use a separate Tensor grad_h_acc (size T+1, hidden)
        // initialized to 0; grad_h_acc[t+1] gets += gh_at_t (post-mogrifier).
        // But within this backward() loop, we can't easily store per-timestep grads
        // since we walk backwards. Instead, we add gh to grad_h (the grad used in
        // the LSTM step at t-1's backward pass).
        // Since we're going t-1 next, grad_h in that step is grad_output[t-1]. We'd
        // need to also add the carrier. Easiest: maintain a tensor grad_h_carrier
        // of shape (T+1, hidden) initialized to 0, and accumulate.
        //
        // For now, let's use a Tensor grad_h_carrier_ (caches member) and add to
        // grad_h at each step. But we don't want to mutate the input grad_output.
        // Alternative: accumulate in grad_h (a local) at each step.

        // For the FIRST step (t=0), gh is the grad to h_{-1} = 0 (the initial
        // hidden state is a constant, so this gradient is discarded — no param).
        // For t > 0, we ADD gh to grad_h[t-1] when processing t-1 next. Use a
        // separate accumulator tensor initialized to zero.
        // We'll add this in a second pass.
        // (Implementation note: stored in cached carrier below.)

        // Add gx (input grad) to grad_inputs[t].
        for (size_t d = 0; d < input_dim_; ++d) {
            grad_inputs[t][d] += gx[0][d];
        }

        // Stash gh into a carrier for use at step t-1. We use a member tensor gh_carrier_
        // (size T+1, hidden) that we reallocate when T changes. gh_carrier_[t]
        // holds the grad to raw h_{t-1} from step t, which step t-1 will read via
        // gh_carrier_[(t-1)+1] = gh_carrier_[t].
        if (gh_carrier_.rows != T + 1 || gh_carrier_.cols != hidden_size_) {
            gh_carrier_ = Tensor(T + 1, hidden_size_);
            gh_carrier_.fill(0.0);
        }
        for (size_t j = 0; j < hidden_size_; ++j) gh_carrier_[t][j] = gh[0][j];
    }

    return grad_inputs;
}

void MogrifierLSTM::update_weights(double learning_rate) {
    for (size_t i = 0; i < W_.rows; ++i)
        for (size_t j = 0; j < W_.cols; ++j)
            W_[i][j] -= learning_rate * grad_W_[i][j];
    for (size_t i = 0; i < b_.rows; ++i)
        for (size_t j = 0; j < b_.cols; ++j)
            b_[i][j] -= learning_rate * grad_b_[i][j];
    for (size_t k = 0; k < Q_list_.size(); ++k) {
        Tensor& Q = Q_list_[k];
        Tensor& gQ = grad_Q_list_[k];
        for (size_t i = 0; i < Q.rows; ++i)
            for (size_t j = 0; j < Q.cols; ++j)
                Q[i][j] -= learning_rate * gQ[i][j];
    }
    for (size_t k = 0; k < R_list_.size(); ++k) {
        Tensor& R = R_list_[k];
        Tensor& gR = grad_R_list_[k];
        for (size_t i = 0; i < R.rows; ++i)
            for (size_t j = 0; j < R.cols; ++j)
                R[i][j] -= learning_rate * gR[i][j];
    }
}