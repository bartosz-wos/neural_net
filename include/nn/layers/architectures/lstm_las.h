#ifndef LSTM_LAS_H
#define LSTM_LAS_H

#include "../../core/layer.h"
#include "../recurrent/lstm.h"
#include "../normalization/layer_norm.h"
#include "../dense/embedding.h"

// Listen-Attend-Spell: LSTM-based seq2seq with attention.
// Encoder: bidirectional LSTM processes source sequence.
// Decoder: LSTM controller with attention over encoder hidden states.
// Attention: computes attention weights over all encoder timesteps,
//            produces a context vector, concatenated with decoder input.
class LASEncoder : public Layer {
public:
    LASEncoder(size_t input_dim, size_t hidden_size, size_t num_layers = 1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    size_t get_seq_len() const { return seq_len_; }
    const Tensor& get_encoder_output() const { return h_states_; }

private:
    std::vector<LSTM> lstm_layers_;
    size_t input_dim_, hidden_size_, num_layers_;
    size_t seq_len_;
    Tensor h_states_;
    Tensor last_output_;
};

// Standalone attention helper (not a Layer to avoid forward signature issues)
class LASSelfAttention {
public:
    LASSelfAttention(size_t hidden_size);
    Tensor forward(const Tensor& query, const Tensor& keys, const Tensor& values);
    void update_weights(double learning_rate);
    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

private:
    size_t hidden_size_;
    Dense query_proj_;
    Dense key_proj_;
    Dense value_proj_;
};

class LASDecoder : public Layer {
public:
    LASDecoder(size_t input_dim, size_t hidden_size,
               const Tensor& encoder_hidden, size_t encoder_seq_len);
    // Overload that accepts encoder output for attention
    Tensor forward(const Tensor& input);
    Tensor forward(const Tensor& input, const Tensor& encoder_output);
    const Tensor& last_output() const { return last_output_; }
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    LSTM controller_;
    LASSelfAttention attention_;
    Dense context_proj_;
    Dense output_proj_;
    Tensor encoder_hidden_;
    size_t encoder_seq_len_;
    size_t hidden_size_;
    Tensor last_output_;
    Tensor last_context_;
    Tensor last_input_;  // cached decoder input for backward
};

class ListenAttendSpell : public Layer {
public:
    using Layer::forward;  // bring base forward(const Tensor&) into scope so both overloads are visible
    ListenAttendSpell(size_t vocab_size, size_t embedding_dim,
                       size_t encoder_hidden, size_t decoder_hidden,
                       size_t num_layers = 1);
    // input: (seq_len, embedding_dim), target: (target_seq_len, embedding_dim)
    // NOTE: forward(input, target) intentionally overloads base forward(const Tensor&)
    // ListenAttendSpell uses a 2-input forward for encoder-decoder architecture
    Tensor forward(const Tensor& input, const Tensor& target);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    // Greedy decoding
    Tensor decode(const Tensor& source, size_t max_len = 50);

private:
    Embedding encoder_embedding_;
    LASEncoder encoder_;
    Embedding decoder_embedding_;
    LASDecoder decoder_;
    Dense output_layer_;
    size_t vocab_size_;
    Tensor last_output_;
};

#endif
