#include "lstm_bidirectional.h"

BiLSTM::BiLSTM(size_t input_dim, size_t hidden_size, size_t seq_len)
    : forward_lstm_(input_dim, hidden_size, seq_len),
      backward_lstm_(input_dim, hidden_size, seq_len),
      input_dim_(input_dim), hidden_size_(hidden_size), seq_len_(seq_len) {}

Tensor BiLSTM::forward(const Tensor& input) {
    size_t N = input.rows;

    // Forward LSTM on original sequence
    forward_lstm_.forward(input);
    const Tensor& h_fwd_states = forward_lstm_.h_states;

    // Build reversed-sequence input for backward LSTM
    Tensor input_rev(N, seq_len_ * input_dim_);
    for (size_t b = 0; b < N; ++b) {
        for (size_t t = 0; t < seq_len_; ++t) {
            size_t orig_t = seq_len_ - 1 - t;
            for (size_t j = 0; j < input_dim_; ++j)
                input_rev[b][t * input_dim_ + j] = input[b][orig_t * input_dim_ + j];
        }
    }

    // Backward LSTM on reversed sequence
    backward_lstm_.forward(input_rev);
    const Tensor& h_bwd_states = backward_lstm_.h_states;

    // Output: (batch, 2*hidden_size) — [last forward hidden; last backward hidden]
    last_output_ = Tensor(N, 2 * hidden_size_);
    for (size_t b = 0; b < N; ++b) {
        size_t fwd_idx = N * seq_len_ + b; // last hidden for batch b
        size_t bwd_idx = N * seq_len_ + b;
        for (size_t j = 0; j < hidden_size_; ++j) {
            last_output_[b][j] = h_fwd_states[fwd_idx][j];
            last_output_[b][hidden_size_ + j] = h_bwd_states[bwd_idx][j];
        }
    }
    return last_output_;
}

Tensor BiLSTM::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, seq_len_ * input_dim_);
}

void BiLSTM::update_weights(double learning_rate) {
    forward_lstm_.update_weights(learning_rate);
    backward_lstm_.update_weights(learning_rate);
}

void BiLSTM::zero_grad() {
    forward_lstm_.zero_grad();
    backward_lstm_.zero_grad();
}

std::vector<Tensor*> BiLSTM::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : forward_lstm_.parameters())
        result.push_back(p);
    for (Tensor* p : backward_lstm_.parameters())
        result.push_back(p);
    return result;
}
std::vector<Tensor*> BiLSTM::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : forward_lstm_.gradients())
        result.push_back(g);
    for (Tensor* g : backward_lstm_.gradients())
        result.push_back(g);
    return result;
}
