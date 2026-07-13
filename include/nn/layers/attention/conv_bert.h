#ifndef CONV_BERT_H
#define CONV_BERT_H

#include "../../core/layer.h"

#include <cstddef>
#include <string>
#include <vector>

// ConvBERT-style convolution-augmented attention.
//
// Reference: Ren et al., "ConvBERT: Improving BERT with Span-based Dynamic
// Convolution" (2020), https://arxiv.org/abs/2008.02496.
//
// This pedagogical variant implements the repository expansion contract:
// a global multi-head self-attention branch runs alongside a local
// Dense -> GLU -> depthwise Conv1D branch. A learnable scalar alpha mixes
// their outputs:
//
//   y = alpha * Attention(x) + (1 - alpha) * DepthwiseConv(GLU(Dense(x))).
//
// Input and output use the repository's token-major layout (tokens, d_model).
// Sequence length is dynamic; odd kernel sizes use zero padding to preserve it.
class ConvBertLayer : public Layer {
public:
    ConvBertLayer(size_t d_model, size_t kernel_size = 7,
                  size_t num_heads = 1, double alpha = 0.5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return query_projection_.weights; }
    Tensor get_gradients() const override { return query_projection_.grad_weights; }
    std::string name() const override { return "ConvBertLayer"; }

    size_t d_model() const { return d_model_; }
    size_t kernel_size() const { return kernel_size_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

    double alpha() const;
    void set_alpha(double value);

    const Tensor& last_attention_output() const { return last_attention_output_; }
    const Tensor& last_convolution_output() const { return last_convolution_output_; }

private:
    size_t d_model_;
    size_t kernel_size_;
    size_t num_heads_;
    size_t head_dim_;
    size_t padding_;
    double attention_scale_;

    Dense query_projection_;
    Dense key_projection_;
    Dense value_projection_;
    Dense attention_output_projection_;
    Dense convolution_projection_;  // d_model -> 2*d_model for GLU

    Tensor depthwise_weights_;       // (d_model, kernel_size)
    Tensor depthwise_bias_;          // (1, d_model)
    Tensor grad_depthwise_weights_;
    Tensor grad_depthwise_bias_;
    Tensor alpha_logit_;             // (1, 1), effective alpha = sigmoid(logit)
    Tensor grad_alpha_logit_;

    Tensor last_input_;
    Tensor last_query_;
    Tensor last_key_;
    Tensor last_value_;
    Tensor last_attention_probs_;    // (num_heads * tokens, tokens)
    Tensor last_attention_context_;  // (tokens, d_model)
    Tensor last_attention_output_;   // (tokens, d_model)

    Tensor last_convolution_pre_;    // (tokens, 2*d_model)
    Tensor last_gate_;               // (tokens, d_model)
    Tensor last_glu_;                // (tokens, d_model)
    Tensor last_convolution_output_; // (tokens, d_model)
};

#endif
