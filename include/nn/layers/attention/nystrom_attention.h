#ifndef NYSTROM_ATTENTION_H
#define NYSTROM_ATTENTION_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

class NystromAttention : public Layer {
public:
    using Layer::forward;  // bring base forward into scope so 3-arg overload is visible
    int embed_dim_;
    int num_heads_;
    int num_landmarks_;
    int head_dim_;
    float dropout_;
    float scale_;
    bool is_initialized_;

    // Learned projections: Q, K, V, output
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // Cache for backward pass
    Tensor last_query;  // (batch, seq * E) original query input
    Tensor last_key;    // (batch, seq * E) original key input
    Tensor last_value;  // (batch, seq * E) original value input
    Tensor last_q;       // (batch, seq * E) projected Q
    Tensor last_k;       // (batch, seq * E) projected K
    Tensor last_v;       // (batch, seq * E) projected V
    Tensor last_A_bar;   // (batch, num_heads, m, n) - landmark-to-all probs (flat per head)
    Tensor last_A_tilde; // (batch, num_heads, m, m) - landmark self-attn probs (flat per head)
    Tensor last_P;       // (batch, num_heads, m, n) - solution P (flat per head)
    Tensor last_landmark_v; // (batch, num_heads, m, head_dim) - landmark V values
    Tensor last_output_accum; // (batch, seq, embed_dim) pre-projection output

    size_t batch_size_;
    size_t seq_len_;
    std::vector<size_t> landmark_indices_;
    bool nystrom_path_used_;
    bool fallback_path_used_;

    NystromAttention(int embed_dim, int num_heads, int num_landmarks = 0, float dropout = 0.0f);

    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value);
    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    std::string name() const override { return "NystromAttention"; }
    void print() const;
    // Satisfy Layer abstract base — input is ignored, use forward(q,k,v) for attention
    Tensor forward(const Tensor& input) override {
        size_t seq = input.cols / embed_dim_;
        (void)seq; // unused, kept for documentation
        return forward(input, input, input);
    }

private:
    // Softmax over last axis
    Tensor softmax(const Tensor& x);
};

#endif