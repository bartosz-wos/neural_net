#ifndef VIT_H
#define VIT_H

#include "../../core/layer.h"
#include "../../layers/normalization/layer_norm.h"
#include "../../layers/convolutions/conv_layer.h"
#include "../../layers/attention/transformer.h"
#include <vector>

// ViTPatchEmbedding: split image into N patches and project each to d_model.
// Input:  (B, C, H, W)  -- flattened as (B, C*H*W)
// Output: (B, N, d_model) -- stored as (B, N*d_model) flat
class ViTPatchEmbedding : public Layer {
public:
    size_t patch_size;
    size_t C_in, H_in, W_in;
    size_t d_model;
    size_t num_patches;
    size_t H_patch, W_patch;
    Conv2D conv;

    ViTPatchEmbedding(size_t patch_size, size_t C_in, size_t H_in, size_t W_in,
                      size_t d_model);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return conv.weights; }
    Tensor get_gradients() const override { return conv.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

// ViTBlock: transformer encoder block for vision.
// Input: (B, (N+1)*d_model) flat representation of (B, N+1, d_model)
// Pre-norm -> MultiHeadAttention -> residual
// Pre-norm -> MLP (GeLU -> linear -> linear) -> residual
class ViTBlock : public Layer {
public:
    size_t d_model, num_heads, mlp_hidden, seq_len;
    MultiHeadAttention attn;
    LayerNorm ln1, ln2;
    Tensor W1, b1;
    Tensor grad_W1, grad_b1;
    Tensor W2, b2;
    Tensor grad_W2, grad_b2;
    Tensor last_x;           // (B, seq_len*d_model)
    Tensor last_ln1_out;     // (B, seq_len*d_model)
    Tensor last_attn_out;    // (B, seq_len*d_model) after residual
    Tensor last_ln2_out;     // (B, seq_len*d_model)
    Tensor last_ffn_pregelu;  // (B, seq_len*mlp_hidden)

    ViTBlock(size_t d_model, size_t num_heads, size_t mlp_hidden, size_t seq_len);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W1; }
    Tensor get_gradients() const override { return grad_W1; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
};

// ViT: Vision Transformer for image classification.
// Input: (B, C, H, W) -> Output: (B, num_classes)
class ViT : public Layer {
public:
    size_t patch_size;
    size_t d_model;
    size_t num_heads;
    size_t num_layers;
    size_t num_classes;
    size_t C_in, H_in, W_in;
    size_t num_patches;
    size_t N_plus_1;
    size_t mlp_hidden;

    ViTPatchEmbedding patch_embed;
    Tensor class_token;       // (1, d_model)
    Tensor pos_embedding;     // (N+1, d_model)
    std::vector<ViTBlock> transformer_blocks;
    LayerNorm ln;
    Dense head;               // (d_model, num_classes)

    Tensor last_patch_tokens;  // (B, N_plus_1 * d_model)
    Tensor last_cls;           // (B, d_model)

    ViT(size_t patch_size, size_t d_model, size_t num_heads, size_t num_layers,
        size_t C_in, size_t H_in, size_t W_in, size_t num_classes,
        double mlp_ratio = 4.0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return head.weights; }
    Tensor get_gradients() const override { return head.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t);

private:
    // Helper: element access on flat 3D tensor stored as 2D
    // flat[b][s*d_model + d] = value
    inline double flat3_at(const Tensor& flat, size_t b, size_t s, size_t d) const {
        return flat[b][s * d_model + d];
    }
    inline void flat3_set(Tensor& flat, size_t b, size_t s, size_t d, double val) {
        flat[b][s * d_model + d] = val;
    }
};

#endif
