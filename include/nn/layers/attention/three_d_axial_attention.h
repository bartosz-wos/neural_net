#ifndef THREE_D_AXIAL_ATTENTION_H
#define THREE_D_AXIAL_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// 3D Axial Attention — Ho, Kalchbrenner, Weissenborn, Salimans 2019 §3 (video)
//
// Idea: for a (H, W, T) volumetric grid flattened to (H*W*T, d_model) tokens,
// standard self-attention costs O((H*W*T)^2 * d). 3D axial attention factors
// the attention into THREE passes — one along each axis:
//   * H-axis pass: for each (j, t), attend over the H tokens of that col-on-T
//   * W-axis pass: for each (i, t), attend over the W tokens of that row-on-T
//   * T-axis pass: for each (i, j), attend over the T tokens of that (i, j) column
// Each axis costs O((H+W+T) * H*W*T * d) and the three passes together replace
// the full self-attention. Per-axis causal mask (causal_h / causal_w / causal_t)
// applied independently. Fuse is a residual sum (parameter-free):
//   output = out_h_pre + out_w_pre + out_t_pre + input
// matching Ho et al. 2019 §3 and the 2D axial attention convention.
//
// Conventions:
//   * Input:  (H*W*T, d_model)    — n = H*W*T tokens, d_model features
//   * Output: (H*W*T, d_model)
//   * d_model must be evenly divisible by num_heads
//   * Per-axis causal mask; defaults to causal=true on all three axes
//   * Biases on output projections only (Llama/Mistral style); no bias on Q/K/V
//
// Classes:
//   ThreeDAxialAttention         — the three-axis attention layer
//   ThreeDAxialAttentionBlock    — pre-LN → 3D axial attn → residual →
//                                   pre-LN → FFN → residual (optional FFN)
//   ThreeDAxialAttentionModel    — input projection → stack of blocks →
//                                   per-token head
// ============================================================================

class ThreeDAxialAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // H, W, T:        3D grid dimensions; input.rows must equal H*W*T
    // num_heads:      number of heads per axis (MHA-style; 1 head by default)
    // causal_h/w/t:   apply causal mask on the H / W / T axis (default true)
    ThreeDAxialAttention(size_t d_model, size_t H, size_t W, size_t T,
                         size_t num_heads = 1,
                         bool causal_h = true,
                         bool causal_w = true,
                         bool causal_t = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q_h; }
    Tensor get_gradients() const override { return grad_W_q_h; }
    std::string name() const override { return "ThreeDAxialAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t H()         const { return H_; }
    size_t W()         const { return W_; }
    size_t T()         const { return T_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    bool    causal_h() const { return causal_h_; }
    bool    causal_w() const { return causal_w_; }
    bool    causal_t() const { return causal_t_; }

    // Cache accessors (for tests + FD perturbation)
    const Tensor& last_attn_h() const { return last_attn_h_; }
    const Tensor& last_attn_w() const { return last_attn_w_; }
    const Tensor& last_attn_t() const { return last_attn_t_; }
    Tensor grad_input() const { return last_d_input_; }

    // Public parameter tensors (test access + mutation)
    // W_*_h are used by the H-axis pass; W_*_w by the W-axis; W_*_t by the T-axis.
    Tensor W_q_h, W_k_h, W_v_h, W_o_h;       // (d_model, d_model) each
    Tensor W_q_w, W_k_w, W_v_w, W_o_w;       // (d_model, d_model) each
    Tensor W_q_t, W_k_t, W_v_t, W_o_t;       // (d_model, d_model) each
    Tensor b_h, b_w, b_t;                    // (1, d_model) each
    Tensor grad_W_q_h, grad_W_k_h, grad_W_v_h, grad_W_o_h;
    Tensor grad_W_q_w, grad_W_k_w, grad_W_v_w, grad_W_o_w;
    Tensor grad_W_q_t, grad_W_k_t, grad_W_v_t, grad_W_o_t;
    Tensor grad_b_h, grad_b_w, grad_b_t;

private:
    size_t d_model_, H_, W_, T_, num_heads_, head_dim_;
    bool   causal_h_, causal_w_, causal_t_;
    double scale_;

    // BPTT cache (per-axis caches; each axis has its own — never share buffers)
    Tensor last_input_;                                  // (n, d_model)
    Tensor last_q_h_, last_k_h_, last_v_h_;              // (n, d_model)
    Tensor last_q_w_, last_k_w_, last_v_w_;              // (n, d_model)
    Tensor last_q_t_, last_k_t_, last_v_t_;              // (n, d_model)
    Tensor last_attn_h_, last_attn_w_, last_attn_t_;     // (n, n) softmax per axis
    Tensor last_head_out_h_, last_head_out_w_, last_head_out_t_;  // (n, d_model)
    Tensor last_out_h_, last_out_w_, last_out_t_;        // (n, d_model)
    Tensor last_d_input_;                                // (n, d_model)
};

// ============================================================================
// ThreeDAxialAttentionBlock — pre-LN → ThreeDAxialAttention → residual →
//                              pre-LN → GELU FFN → residual (optional FFN)
// ============================================================================

class ThreeDAxialAttentionBlock : public Layer {
public:
    // ffn_dim: FFN hidden dim. If 0, no FFN sub-layer (pure attention block).
    ThreeDAxialAttentionBlock(size_t d_model, size_t H, size_t W, size_t T,
                              size_t num_heads = 1,
                              size_t ffn_dim = 0,
                              bool causal_h = true,
                              bool causal_w = true,
                              bool causal_t = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn_.W_q_h; }
    Tensor get_gradients() const override { return attn_.grad_W_q_h; }
    std::string name() const override { return "ThreeDAxialAttentionBlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_, ffn_dim_;

    ThreeDAxialAttention attn_;
    LayerNorm ln1_;
    LayerNorm ln2_;
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    // BPTT cache
    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ffn_pre_;
    Tensor last_ffn_hidden_;
    Tensor last_ffn_out_;
    Tensor last_d_input_;
};

// ============================================================================
// ThreeDAxialAttentionModel — input projection → stack of blocks →
//                              per-token head
// ============================================================================

class ThreeDAxialAttentionModel : public Layer {
public:
    ThreeDAxialAttentionModel(size_t d_input,
                              size_t d_model,
                              size_t d_output,
                              size_t H, size_t W, size_t T,
                              size_t num_blocks = 2,
                              size_t num_heads = 1,
                              size_t ffn_dim = 0,
                              bool causal_h = true,
                              bool causal_w = true,
                              bool causal_t = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_.get_weights(); }
    Tensor get_gradients() const override { return W_in_.get_gradients(); }
    std::string name() const override { return "ThreeDAxialAttentionModel"; }

private:
    size_t d_input_, d_model_, d_output_;
    Dense W_in_, W_out_;
    LayerNorm ln_final_;
    std::vector<ThreeDAxialAttentionBlock> blocks_;

    Tensor last_input_;
    Tensor last_proj_;
};

#endif
