#ifndef AFT_H
#define AFT_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// AFT — Attention Free Transformer
//   Zhai et al. 2021, "An Attention Free Transformer"
//   (https://arxiv.org/abs/2105.14103)
//
// AFT replaces softmax attention with element-wise operations + sigmoid
// gating, achieving O(n) cost per layer (vs O(n^2) for softmax attention)
// while still capturing pairwise position interactions through a *learnable*
// position bias. The paper shows AFT is competitive with softmax attention
// on language modelling and image classification.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (per-position, no batching dim shown):
//
//   Q_t = x_t @ W_q                              in R^{d_model}
//   K_t = x_t @ W_k                              in R^{d_model}
//   V_t = x_t @ W_v                              in R^{d_model}
//   A_t = sigmoid(Q_t)                           in R^{d_model}    (query-side gate)
//
//   For t = 0..n-1:
//     Y_t = sum_s exp(K_s + w_{t,s}) * V_s        in R^{d_model}
//     Z_t = sum_s exp(K_s + w_{t,s})              in R^{d_model}
//     out_t = A_t * (Y_t / (Z_t + eps))           in R^{d_model}
//
// where w in R^{n x n} is a learnable position bias. Note that, unlike
// softmax attention, AFT allows Q and K to have different dimensions /
// uses a position-aware element-wise decomposition. The denominator Z_t
// acts as a per-channel "softmax" normalizer but is decomposed elementwise.
//
// Variants in the paper (we implement AFT-full, the most expressive):
//   * AFT-full  : w in R^{n x n}  — full learned position bias (this impl)
//   * AFT-local : w is local-windowed (e.g. w_{t,s} nonzero only for |t-s|<w)
//   * AFT-conv  : w_{t,s} is computed from a learned 1D conv on positions
//
// We follow the AFT-full specification. The position bias is initialized
// small (zero) so early training is essentially a per-channel scale of
// a sum over (exp(K_s) * V_s) values.
//
// ----------------------------------------------------------------------------
// Complexity:
//   Standard softmax attention:   O(n^2 * d) memory and compute
//   AFT-full (this impl):         O(n^2) for w storage + O(n * d) per step
//                                 (the inner loop is O(n * d) since the sum
//                                 over s collapses, not O(n^2 * d))
//
// ----------------------------------------------------------------------------
// Conventions (match GQA/Performer/Linformer in this repo):
//   * Input/Output: (n, d_model) — n tokens, d_model features
//   * W_q, W_k, W_v, W_o are Dense layers: y = x @ W^T + b, weights (out, in).
//     In forward we use the "Dense convention" — store W as (out, in) and
//     compute Y = X @ W^T. So for AFT: Q = X @ W_q^T (Dense forward).
//   * d_model must equal d_model (no head splitting in v1; could be added later
//     as multi-head AFT by reshaping Q/K/V into (n, num_heads, head_dim) and
//     summing over the head dim after the position mixing, but the v1 single-
//     head form is the most pedagogical and tractable for gradient checks).
//   * Block: pre-LN -> AFTAttention -> residual -> pre-LN -> GELU FFN -> residual
//   * Model: stack of `num_blocks` AFTBlocks + per-token classifier
// ----------------------------------------------------------------------------

class AFTAttention : public Layer {
public:
    // d_model: input/output feature dim (single-head v1)
    // max_seq_len: maximum number of tokens; position bias is (max_seq_len, max_seq_len)
    AFTAttention(size_t d_model, size_t max_seq_len);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "AFTAttention"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t max_seq_len() const { return max_seq_len_; }
    const Tensor& position_bias() const { return position_bias_; }

    // Public parameters
    Dense W_q;            // (d_model, d_model)
    Dense W_k;            // (d_model, d_model)
    Dense W_v;            // (d_model, d_model)
    Dense W_o;            // (d_model, d_model)
    Tensor position_bias_;// (max_seq_len, max_seq_len) — learnable

    // Gradient accumulator for position_bias_
    Tensor grad_position_bias_;

private:
    size_t d_model_;
    size_t max_seq_len_;

    // Caches for forward
    Tensor last_input_;           // (n, d_model)
    Tensor last_Q_;               // (n, d_model)
    Tensor last_K_;               // (n, d_model)
    Tensor last_V_;               // (n, d_model)
    Tensor last_A_;               // (n, d_model) — sigmoid(Q)
    Tensor last_Y_;               // (n, d_model) — AFT output before W_o
    Tensor last_Z_;               // (n, d_model) — denominator per-channel
    Tensor last_output_pre_wo_;   // (n, d_model) — A_t * Y_t / (Z_t + eps), the input to W_o
};

// ----------------------------------------------------------------------------
// AFTBlock — pre-LN -> AFTAttention -> residual -> pre-LN -> GELU FFN -> residual
// ----------------------------------------------------------------------------
class AFTBlock : public Layer {
public:
    AFTBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.W_q.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "AFTBlock"; }

    AFTAttention attn;
    LayerNorm ln1;        // (d_model,)
    LayerNorm ln2;        // (d_model,)
    Dense ffn_fc1_;       // (d_model, ffn_dim) — d_model -> ffn_dim
    Dense ffn_fc2_;       // (ffn_dim, d_model) — ffn_dim -> d_model

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;       // block input (for residual)
    Tensor last_z1_;      // ln1(x) — input to AFT
    Tensor last_attn_out_;// AFT output (for residual)
    Tensor last_res1_;    // x + attn_out
    Tensor last_z2_;      // ln2(res1) — input to FFN
    Tensor last_h_pre_;   // ffn_fc1(z2) — pre-activation FFN
    Tensor last_ffn_h_;   // gelu(h_pre) — FFN hidden
    Tensor last_ffn_out_; // ffn_fc2(h_act) — FFN output
};

// ----------------------------------------------------------------------------
// AFTModel — stack of AFTBlocks + per-token classifier
// ----------------------------------------------------------------------------
class AFTModel : public Layer {
public:
    // d_model, max_seq_len: model dims
    // num_blocks: number of stacked blocks
    // out_features: per-token classifier output dim
    // ffn_dim: hidden dim of FFN inside each block (default 4*d_model)
    AFTModel(size_t d_model, size_t max_seq_len, size_t out_features,
             size_t num_blocks, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return classifier_.weights; }
    Tensor get_gradients() const override { return classifier_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "AFTModel"; }

    std::vector<std::unique_ptr<AFTBlock>> blocks;
    Dense classifier_;    // (d_model, out_features) — per-token classifier

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t out_features_;
    size_t num_blocks_;
    size_t ffn_dim_;
    Tensor last_input_;
    Tensor last_block_output_;  // output of the last block (input to classifier)
};

#endif // AFT_H
