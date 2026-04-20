#ifndef SEQ2SEQ_ATTENTION_H
#define SEQ2SEQ_ATTENTION_H

#include "../core/layer.h"
#include "recurrent/lstm.h"
#include <vector>

// AttentionLayer — content-based attention (Bahdanau-style).
// Input: (batch, hidden_size + seq_len * encoder_dim)
// first hidden_size cols = decoder state, rest = encoder outputs (flattened time steps)
class AttentionLayer : public Layer {
public:
    AttentionLayer(size_t hidden_size, size_t encoder_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_context, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    const Tensor& attention_weights() const { return alpha_; }

private:
    size_t hidden_size_, encoder_dim_, seq_len_;
    Tensor W_a_, v_a_;
    Tensor grad_W_a_, grad_v_a_;
    Tensor alpha_;
    Tensor last_input_;
    Tensor last_context_;
};

class Seq2SeqEncoder : public Layer {
public:
    Seq2SeqEncoder(size_t input_dim, size_t hidden_size, size_t seq_len);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    const Tensor& hidden_states() const { return lstm_.h_states; }

private:
    LSTM lstm_;
    size_t seq_len_;
};

// Full Seq2Seq model with encoder, attention, and decoder
class Seq2Seq : public Layer {
public:
    Seq2Seq(size_t input_dim, size_t output_dim, size_t hidden_size,
            size_t encoder_seq_len, size_t decoder_seq_len);
    Tensor forward(const Tensor& source) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    size_t encoder_dim_;
    Seq2SeqEncoder encoder_;
    AttentionLayer attention_;
    LSTM decoder_;
    Dense projection_;
    size_t encoder_seq_len_, decoder_seq_len_;
    Tensor last_output_;
};

#endif