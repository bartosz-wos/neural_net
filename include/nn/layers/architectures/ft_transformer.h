#ifndef FT_TRANSFORMER_H
#define FT_TRANSFORMER_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../normalization/layer_norm.h"
#include "../attention/gqa.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// FT-Transformer — Gorishniy et al. 2021
//   "Revisiting Deep Learning Models for Tabular Data"
//   https://arxiv.org/abs/2106.11959
//
// A pure-transformer architecture for tabular data. The key insight is that
// every numerical feature becomes its own learnable embedding via a per-
// feature Linear(1 → d_model), categorical features become per-feature
// embedding lookups, and the resulting feature tokens are processed by a
// stack of pre-norm transformer blocks (a CLS token is prepended and used
// for aggregation).
//
// Architecture:
//
//   numerical input  (B, n_num) ──► NumericalFeatureTokenizer ──► (B, n_num, d_model)
//   categorical input (B, n_cat) ─► CategoricalFeatureTokenizer ─► (B, n_cat, d_model)
//                                            │
//                                            ▼
//                  tokens = [CLS_token; num_tokens; cat_tokens]   (B, 1+n_num+n_cat, d_model)
//                                            │
//                          N × GQABlock (pre-LN → attention → residual →
//                                        pre-LN → GELU FFN → residual)
//                                            │
//                                            ▼
//                                  CLS hidden state (B, d_model)
//                                            │
//                                            ▼
//                       LayerNorm → classifier MLP → logits (B, n_classes)
//
// Design choices (following the paper §3.1):
//
//   * The numerical feature tokenizer applies a per-feature Linear:
//         t_i = W_i · x_i + b_i
//     Implemented as a single (d_model, n_num) weight matrix so the forward
//     is just `tokens = x @ W^T + b` (broadcasting the bias across the batch
//     axis).
//
//   * Categorical features: in v1, the categorical inputs are PRE-EMBEDDED
//     by the caller into a (B, n_cat, vocab_i) one-hot OR dense-probability
//     tensor along the LAST axis. The tokenizer does a per-feature linear
//     `tokens_i = one_hot_i @ E_i` where E_i is a (vocab_i, d_model) embedding
//     table. This keeps the math trivial and lets the caller handle one-hot
//     / embedding lookup themselves.
//
//   * GQABlock is reused as-is — the paper recommends standard multi-head
//     self-attention, and GQABlock with num_kv_heads == num_query_heads
//     IS standard MHA (a special case of GQA).
//
//   * The CLS token is a learnable parameter prepended at every forward.
//     Its final hidden state (after the stack of blocks + final LayerNorm)
//     is fed to the classifier MLP.
//
// Conventions:
//   * B = batch dim, n_num = number of numerical features,
//     n_cat = number of categorical features, d_model = transformer dim,
//     n_classes = classifier output dim.
//   * Numerical input:  Tensor (B, n_num)
//   * Categorical input: Tensor (B, n_cat, vocab_max) — vocab_max is the
//     maximum vocabulary size across all categorical features. The tokenizer
//     masks unused slots for each feature (handled internally).
// ============================================================================

// ============================================================================
// NumericalFeatureTokenizer — per-feature Linear(1 → d_model)
//
// Maps (B, n_num) → (B, n_num, d_model) by applying a per-feature linear
// transformation. Weights stored as a (d_model, n_num) matrix so the forward
// is a single matmul.
//
//   tokens[b, i, j] = W[j, i] * x[b, i] + b[j]
//
// Backward propagates through both the input (dL/dx) and the parameters
// (dL/dW, dL/db).
// ============================================================================

class NumericalFeatureTokenizer : public Layer {
public:
    // d_model:        transformer feature dim
    // n_num_features: number of numerical input features
    NumericalFeatureTokenizer(size_t d_model, size_t n_num_features);

    // Input:  (B, n_num_features)
    // Output: (B, n_num_features, d_model)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_; }
    Tensor get_gradients() const override { return grad_W_; }
    std::string name() const override { return "NumericalFeatureTokenizer"; }

    size_t d_model() const { return d_model_; }
    size_t n_num_features() const { return n_num_; }

    // Direct parameter accessors (for tests / FD checks).
    Tensor& W() { return W_; }
    Tensor& b() { return b_; }
    const Tensor& W_const() const { return W_; }
    const Tensor& b_const() const { return b_; }
    const Tensor& grad_W() const { return grad_W_; }
    const Tensor& grad_b() const { return grad_b_; }

private:
    size_t d_model_, n_num_;
    Tensor W_;          // (d_model, n_num)
    Tensor b_;          // (1, d_model)
    Tensor grad_W_;     // (d_model, n_num)
    Tensor grad_b_;     // (1, d_model)

    // Forward cache
    Tensor last_input_; // (B, n_num)
    size_t last_B_;     // batch dim cached for backward
};

// ============================================================================
// CategoricalFeatureTokenizer — per-feature embedding lookup (v1: linear over
// pre-embedded input)
//
// Maps (B, n_cat, vocab_max) → (B, n_cat, d_model) by applying a per-feature
// linear transformation. For each categorical feature i:
//   tokens_i = one_hot_i @ E_i + b_i
// where E_i ∈ R^{vocab_max × d_model} and b_i ∈ R^{1 × d_model}. To keep
// the implementation uniform we store all E_i as a (n_cat, vocab_max, d_model)
// tensor.
//
// The user is responsible for one-hot-encoding their categorical inputs
// (so the input tensor is (B, n_cat, vocab_max) with rows summing to 1).
//
// Backward propagates dL/dx (the one-hot probability distribution) and
// dL/dE, dL/db.
// ============================================================================

class CategoricalFeatureTokenizer : public Layer {
public:
    // d_model:    transformer feature dim
    // n_cat:      number of categorical features
    // vocab_max:  vocabulary size per categorical feature (same for all v1)
    CategoricalFeatureTokenizer(size_t d_model, size_t n_cat, size_t vocab_max);

    // Input:  (B, n_cat, vocab_max)
    // Output: (B, n_cat, d_model)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return E_; }
    Tensor get_gradients() const override { return grad_E_; }
    std::string name() const override { return "CategoricalFeatureTokenizer"; }

    size_t d_model() const { return d_model_; }
    size_t n_cat() const { return n_cat_; }
    size_t vocab_max() const { return vocab_max_; }

    // Direct parameter accessors (for tests / FD checks).
    // E_ is stored as (n_cat * vocab_max, d_model) flat.
    Tensor& E() { return E_; }
    Tensor& b_cat() { return b_; }
    const Tensor& E_const() const { return E_; }
    const Tensor& b_cat_const() const { return b_; }
    const Tensor& grad_E() const { return grad_E_; }
    const Tensor& grad_b_cat() const { return grad_b_; }

private:
    size_t d_model_, n_cat_, vocab_max_;

    // E_: (n_cat, vocab_max, d_model) — per-feature embedding tables
    // b_: (n_cat, 1, d_model)          — per-feature biases
    Tensor E_, b_;
    Tensor grad_E_, grad_b_;

    // Forward cache
    Tensor last_input_; // (B, n_cat, vocab_max)
    size_t last_B_;
};

// ============================================================================
// FTBlock — single transformer block with CLS-token bookkeeping
//
// Wraps the GQABlock. The "block" includes:
//   * Input is already pre-concatenated with the CLS token (so the input
//     dimension is 1 + n_num + n_cat tokens).
//   * CLS-token gradient is propagated back to a learnable CLS parameter.
//
// For simplicity in v1, we DO NOT prepend/extract CLS internally — that
// is the FTTransformer's job. FTBlock just runs the GQABlock forward and
// backward.
// ============================================================================

// (Removed FTBlock as a separate class — the FTTransformer uses GQABlock
// directly. Keeping the GQABlock API simple makes the implementation
// easier to verify.)

// ============================================================================
// FTTransformer — full FT-Transformer model
//
// Composes:
//   * NumericalFeatureTokenizer (optional; skipped if n_num == 0)
//   * CategoricalFeatureTokenizer (optional; skipped if n_cat == 0)
//   * CLS token (always)
//   * N × GQABlock (with num_kv_heads == num_query_heads to be standard MHA)
//   * Final LayerNorm
//   * Classifier MLP: Linear(d_model → d_model) → ReLU → Linear(d_model → n_classes)
// ============================================================================

class FTTransformer : public Layer {
public:
    // d_model:    transformer feature dim
    // n_classes:  classifier output dim
    // n_num:      number of numerical features (0 to disable the tokenizer)
    // n_cat:      number of categorical features (0 to disable the tokenizer)
    // vocab_max:  vocabulary size per categorical feature (only used if n_cat > 0)
    // num_heads:  number of attention heads (must evenly divide d_model)
    // num_blocks: number of transformer blocks
    // ffn_dim:    FFN hidden dim (0 → defaults to 4 * d_model)
    FTTransformer(size_t d_model, size_t n_classes,
                  size_t n_num, size_t n_cat, size_t vocab_max,
                  size_t num_heads, size_t num_blocks,
                  size_t ffn_dim = 0);

    // Forward pass.
    //   numerical:  Tensor (B, n_num)        — pass empty Tensor if n_num==0
    //   categorical: Tensor (B, n_cat, vocab_max) — pass empty Tensor if n_cat==0
    // Returns logits (B, n_classes).
    Tensor forward(const Tensor& numerical, const Tensor& categorical);

    // Backward pass for the (numerical, categorical) inputs and the classifier.
    // Returns a Tensor that contains the gradient w.r.t. numerical input
    // (with shape (B, n_num)), followed (if n_cat > 0) by the gradient
    // w.r.t. categorical input. To extract the per-input gradients, use
    //   Tensor g_num  = grad_output_only(grad_total);
    //   Tensor g_cat  = grad_categorical_only(grad_total);
    // (where grad_total is what backward() returns).
    //
    // The single-Tensor Layer::backward contract is preserved by returning
    // a packed (B, n_num + n_cat * vocab_max) tensor for v1; if n_cat == 0
    // we return (B, n_num) directly.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    // Backward pass returning a struct of tensors.  Identical to backward(.)
    // for the Layer contract but returns the per-input gradient Tensors
    // separately for cleanliness.
    struct BackwardOutputs {
        Tensor grad_numerical;
        Tensor grad_categorical;
    };
    BackwardOutputs backward_full(const Tensor& grad_output, double learning_rate);

    // Layer overrides — forward(.) takes 1 Tensor by signature, so we route
    // through the (numerical, categorical) variant.
    Tensor forward(const Tensor& input) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "FTTransformer"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t n_classes() const { return n_classes_; }
    size_t n_num_features() const { return n_num_; }
    size_t n_cat_features() const { return n_cat_; }
    size_t vocab_max() const { return vocab_max_; }
    size_t num_heads() const { return num_heads_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t ffn_dim() const { return ffn_dim_; }

    // Internal accessors (for testing)
    Tensor& cls_token() { return cls_token_; }
    const Tensor& last_cls_hidden() const { return last_cls_hidden_; }
    const Tensor& last_numerical() const { return last_numerical_; }
    const Tensor& last_categorical() const { return last_categorical_; }
    const Tensor& last_concat() const { return last_concat_; }

private:
    size_t d_model_, n_classes_, n_num_, n_cat_, vocab_max_;
    size_t num_heads_, num_blocks_, ffn_dim_;

    // Optional tokenizers (null if the corresponding feature count is 0)
    std::unique_ptr<NumericalFeatureTokenizer> num_tok_;
    std::unique_ptr<CategoricalFeatureTokenizer> cat_tok_;

    // Learnable CLS token
    Tensor cls_token_;        // (1, d_model)
    Tensor grad_cls_token_;   // (1, d_model)

    // Stack of transformer blocks
    std::vector<std::unique_ptr<GQABlock>> blocks_;

    // Final LayerNorm
    LayerNorm final_ln_;

    // Classifier MLP: d_model → ffn_dim (ReLU) → n_classes
    Dense clf_fc1_;
    Dense clf_fc2_;

    // Forward caches
    Tensor last_numerical_;   // (B, n_num)        or empty
    Tensor last_categorical_;  // (B, n_cat, vocab) or empty
    Tensor last_concat_;      // (B, 1+n_num+n_cat, d_model) pre-block input
    Tensor last_cls_hidden_;  // (B, d_model) — post-LN CLS for backward routing
    Tensor last_clf_pre_relu_; // (B, ffn_dim) — classifier fc1 post-GELU
    Tensor last_clf_logits_;  // (B, n_classes) — classifier output

    // Last batch dim
    size_t last_B_;
};

#endif