#ifndef FOX_H
#define FOX_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Forgetting Transformer (FoX) — "Forgetting Transformer: Softmax Attention
// with a Forget Gate", Lin, Yang, Sun et al., ICLR 2025
//   https://arxiv.org/abs/2503.02130
//
// Causal softmax attention augmented with a learnable, DATA-DEPENDENT forget
// gate that enters the pre-softmax scores as an additive log-decay bias.
//
// Per head h, per query token t, key token s (s <= t):
//
//   z[t, h]    = (X @ W_f)[t, h]                     forget-gate logit
//   f[t, h]    = sigmoid(z[t, h])           in (0, 1)
//   logf[t, h] = log f[t, h]                <= 0
//   D[t, h]    = sum_{i <= t} logf[i, h]             cumulative log-decay
//   bias[t, s] = D[t, h] - D[s, h] = sum_{i = s+1}^{t} logf[i, h]   <= 0
//   scores[t, s] = (Q_h[t] . K_h[s]) * scale + bias[t, s]     (s <= t)
//   scores[t, s] = -inf                                       (s >  t)
//   A_h        = row_softmax(scores)
//   head_out_h = A_h @ V_h
//   out        = concat_h(head_out_h) @ W_o
//
// Since exp(bias[t, s]) = prod_{i=s+1}^{t} f[i], the gate multiplicatively
// decays attention mass toward older keys — exactly the "Forgetting Attention"
// of the paper, expressible inside a FlashAttention kernel as an additive bias.
// As f -> 1 (large positive gate logits) bias -> 0 and the layer degenerates to
// vanilla causal multi-head attention; as f -> 0 it becomes strongly local.
//
// bias[t, t] == 0 always (empty sum), so the diagonal is never penalized.
//
// FoX-specific backward chain:
//   d_bias[t, s] = dS[t, s]
//   bias[t, s] depends on logf[i] for s < i <= t, hence
//     d_logf[i, h] = sum_{t >= i} sum_{s < i} dS_h[t, s]
//   d/dz log(sigmoid(z)) = 1 - sigmoid(z), so
//     dz[i, h] = d_logf[i, h] * (1 - f[i, h])
//   then the standard (X @ W_f) projection backward.
//
// Conventions (match SHLA / GQA / SlidingWindow in this repo):
//   * Input:  (n, d_model) — n tokens, d_model features
//   * Output: (n, d_model)
//   * d_model must be divisible by num_heads; head_dim = d_model / num_heads
//   * No biases on any projection
//   * Always causal (the forget gate is only meaningful causally)
//
// Parameters: W_q, W_k, W_v, W_o : (d_model, d_model);  W_f : (d_model, num_heads)
// ============================================================================

class FoXAttention : public Layer {
public:
    FoXAttention(size_t d_model, size_t num_heads);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "FoXAttention"; }

    // Accessors for tests
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }

    // Cached forward state (test access)
    const Tensor& get_last_forget() const { return last_f_; }       // (n, num_heads) in (0,1)
    const Tensor& get_last_cumlog() const { return last_D_; }       // (n, num_heads) <= 0
    const Tensor& get_last_attn()   const { return last_attn_; }    // (num_heads*n, n)
    Tensor grad_input() const { return last_d_input_; }

    // Forget bias for (head, query t, key s); 0 when s == t, <= 0 for s < t.
    double forget_bias(size_t h, size_t t, size_t s) const {
        return last_D_(t, h) - last_D_(s, h);
    }

    // Public parameters (test access)
    Tensor W_q, W_k, W_v, W_o, W_f;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_W_f;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double scale_;

    // BPTT cache
    Tensor last_input_;     // (n, d_model)
    Tensor last_q_;         // (n, d_model)
    Tensor last_k_;         // (n, d_model)
    Tensor last_v_;         // (n, d_model)
    Tensor last_f_;         // (n, num_heads)  sigmoid gate
    Tensor last_D_;         // (n, num_heads)  cumulative sum of log f
    Tensor last_attn_;      // (num_heads*n, n) causal softmax probs
    Tensor last_head_out_;  // (n, d_model)
    Tensor last_d_input_;
};

// ============================================================================
// FoXBlock — pre-LN → FoXAttention → residual → pre-LN → GELU FFN → residual
// ============================================================================

class FoXBlock : public Layer {
public:
    FoXBlock(size_t d_model, size_t num_heads, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "FoXBlock"; }

    FoXAttention& attn() { return attn_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_;
    size_t ffn_dim_;
    LayerNorm ln1_;
    FoXAttention attn_;
    LayerNorm ln2_;
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    Tensor last_res1_;
    Tensor last_ffn_hidden_;   // PRE-GELU
    Tensor last_d_input_;
};

// ============================================================================
// FoXModel — input projection → stack of FoXBlocks → output projection
// ============================================================================

class FoXModel : public Layer {
public:
    FoXModel(size_t input_dim, size_t d_model, size_t output_dim,
             size_t num_blocks, size_t num_heads, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "FoXModel"; }

    size_t num_blocks() const { return blocks_.size(); }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;
    std::vector<FoXBlock> blocks_;
    Tensor last_input_;
    Tensor last_proj_;
    Tensor last_block_out_;
};

#endif
