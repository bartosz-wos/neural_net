#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../../core/layer.h"
#include "../../layers/normalization/layer_norm.h"
#include <vector>

class MultiHeadAttention : public Layer {
public:
    size_t d_model, num_heads, d_k;
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor last_q, last_k, last_v;
    Tensor last_scores;
    Tensor last_attn_out;
    Tensor last_x;
    size_t batch_size;

    MultiHeadAttention(size_t d_model, size_t num_heads);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

class TransformerBlock : public Layer {
public:
    size_t d_model, num_heads;
    MultiHeadAttention attn;
    LayerNorm ln1, ln2;
    Tensor W1, b1, W2, b2;
    Tensor grad_W1, grad_b1, grad_W2, grad_b2;
    Tensor last_x, last_attn_out, last_ffn_out, last_ffn_pregelu;

    TransformerBlock(size_t d_model, size_t num_heads);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

class PositionalEncoding : public Layer {
public:
    Tensor pe;
    PositionalEncoding(size_t max_len, size_t d_model);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return pe; }
    Tensor get_gradients() const override { return pe; }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
};

#endif