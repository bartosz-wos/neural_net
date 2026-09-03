#include "ft_transformer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// NumericalFeatureTokenizer — per-feature Linear(1 → d_model)
//
// Forward: tokens[b, i, j] = W[j, i] * x[b, i] + b[j]
//
// We implement the forward as a batched matmul:
//   tokens_flat = x @ W^T + b     (B, n_num) × (n_num, d_model) → (B, d_model)
// but per-feature we need (B, n_num, d_model), so we treat each feature
// independently and add the bias once per feature.
// ============================================================================

NumericalFeatureTokenizer::NumericalFeatureTokenizer(size_t d_model, size_t n_num_features)
    : d_model_(d_model),
      n_num_(n_num_features),
      W_(Tensor::random(d_model, n_num_features, 0.1)),
      b_(Tensor::zeros(1, d_model)),
      grad_W_(d_model, n_num_features),
      grad_b_(1, d_model),
      last_input_(0, 0),
      last_B_(0)
{
    if (d_model == 0)        throw std::invalid_argument("NumericalFeatureTokenizer: d_model must be > 0");
    if (n_num_features == 0) throw std::invalid_argument("NumericalFeatureTokenizer: n_num_features must be > 0");
    grad_W_.fill(0.0);
    grad_b_.fill(0.0);
}

Tensor NumericalFeatureTokenizer::forward(const Tensor& input) {
    // Input: (B, n_num)
    if (input.cols != n_num_) {
        throw std::invalid_argument("NumericalFeatureTokenizer::forward: input cols != n_num");
    }
    last_input_ = input.clone();
    last_B_ = input.rows;

    // Output: (B, n_num, d_model)
    Tensor out(input.rows, n_num_ * d_model_);

    for (size_t b = 0; b < input.rows; ++b) {
        for (size_t i = 0; i < n_num_; ++i) {
            double x_bi = input(b, i);
            for (size_t j = 0; j < d_model_; ++j) {
                // tokens[b, i, j] = W[j, i] * x[b, i] + b[j]
                // Storing as a flat (B, n_num * d_model) row-major matrix:
                //   out(b, i * d_model + j) = W(j, i) * x(b, i) + b(0, j)
                out(b, i * d_model_ + j) = W_(j, i) * x_bi + b_(0, j);
            }
        }
    }
    return out;
}

Tensor NumericalFeatureTokenizer::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (B, n_num * d_model) — flat layout from forward.
    if (grad_output.rows != last_B_ || grad_output.cols != n_num_ * d_model_) {
        throw std::invalid_argument("NumericalFeatureTokenizer::backward: grad shape mismatch");
    }

    Tensor dx(last_B_, n_num_);

    for (size_t b = 0; b < last_B_; ++b) {
        for (size_t i = 0; i < n_num_; ++i) {
            double dx_bi = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                double g = grad_output(b, i * d_model_ + j);

                // dL/dW[j, i] += x[b, i] * g
                grad_W_(j, i) += last_input_(b, i) * g;

                // dL/db[j] += g
                grad_b_(0, j) += g;

                // dL/dx[b, i] += W[j, i] * g
                dx_bi += W_(j, i) * g;
            }
            dx(b, i) = dx_bi;
        }
    }
    return dx;
}

void NumericalFeatureTokenizer::update_weights(double learning_rate) {
    for (size_t i = 0; i < W_.data.size(); ++i) {
        W_.data[i] -= learning_rate * grad_W_.data[i];
    }
    for (size_t i = 0; i < b_.data.size(); ++i) {
        b_.data[i] -= learning_rate * grad_b_.data[i];
    }
}

void NumericalFeatureTokenizer::zero_grad() {
    grad_W_.fill(0.0);
    grad_b_.fill(0.0);
}

std::vector<Tensor*> NumericalFeatureTokenizer::parameters() {
    return {&W_, &b_};
}

std::vector<Tensor*> NumericalFeatureTokenizer::gradients() {
    return {&grad_W_, &grad_b_};
}

// ============================================================================
// CategoricalFeatureTokenizer — per-feature linear over pre-embedded input
//
// Input:  (B, n_cat, vocab_max)
// Output: (B, n_cat, d_model)
// Storage: E_ is (n_cat, vocab_max, d_model), b_ is (n_cat, 1, d_model)
//
// Forward (per feature i, sample b):
//   tokens[b, i, j] = sum_{k=0}^{vocab_max-1} input[b, i, k] * E_[i, k, j] + b_[i, 0, j]
//
// Backward: dL/dx[b, i, k] = sum_j dL/dtokens[b, i, j] * E_[i, k, j]
//           dL/dE_[i, k, j] = sum_b dL/dtokens[b, i, j] * input[b, i, k]
//           dL/db_[i, j]     = sum_b dL/dtokens[b, i, j]
// ============================================================================

CategoricalFeatureTokenizer::CategoricalFeatureTokenizer(size_t d_model, size_t n_cat, size_t vocab_max)
    : d_model_(d_model),
      n_cat_(n_cat),
      vocab_max_(vocab_max),
      // E_ stored as (n_cat * vocab_max, d_model): row (i * vocab_max + k) is E_[i, k, :]
      E_(Tensor::zeros(n_cat * vocab_max, d_model)),
      // b_ stored as (n_cat, d_model): row i is b_[i, :]
      b_(Tensor::zeros(n_cat, d_model)),
      grad_E_(n_cat * vocab_max, d_model),
      grad_b_(n_cat, d_model),
      last_input_(0, 0),
      last_B_(0)
{
    if (d_model == 0)   throw std::invalid_argument("CategoricalFeatureTokenizer: d_model must be > 0");
    if (n_cat == 0)     throw std::invalid_argument("CategoricalFeatureTokenizer: n_cat must be > 0");
    if (vocab_max == 0) throw std::invalid_argument("CategoricalFeatureTokenizer: vocab_max must be > 0");

    // Initialize E with small random values per (n_cat * vocab_max, d_model) row.
    for (size_t i = 0; i < E_.data.size(); ++i) {
        E_.data[i] = ((double) rand() / RAND_MAX - 0.5) * 0.1;
    }
    grad_E_.fill(0.0);
    grad_b_.fill(0.0);
}

Tensor CategoricalFeatureTokenizer::forward(const Tensor& input) {
    // input: (B, n_cat * vocab_max)
    if (input.cols != n_cat_ * vocab_max_) {
        throw std::invalid_argument("CategoricalFeatureTokenizer::forward: input shape mismatch");
    }
    if (input.rows == 0) {
        throw std::invalid_argument("CategoricalFeatureTokenizer::forward: empty batch");
    }
    last_input_ = input.clone();
    last_B_ = input.rows;

    // Output: (B, n_cat * d_model) — flat
    Tensor out(input.rows, n_cat_ * d_model_);

    for (size_t b = 0; b < input.rows; ++b) {
        for (size_t i = 0; i < n_cat_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double sum = b_(i, j);  // bias
                for (size_t k = 0; k < vocab_max_; ++k) {
                    double in_val = input(b, i * vocab_max_ + k);
                    // E_ row index for E_[i, k, :] is (i * vocab_max + k)
                    sum += in_val * E_(i * vocab_max_ + k, j);
                }
                out(b, i * d_model_ + j) = sum;
            }
        }
    }
    return out;
}

Tensor CategoricalFeatureTokenizer::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (B, n_cat * d_model)
    if (grad_output.rows != last_B_ || grad_output.cols != n_cat_ * d_model_) {
        throw std::invalid_argument("CategoricalFeatureTokenizer::backward: grad shape mismatch");
    }

    // dL/dx shape: (B, n_cat * vocab_max) — same flat layout as input
    Tensor dx(last_B_, n_cat_ * vocab_max_);

    for (size_t b = 0; b < last_B_; ++b) {
        for (size_t i = 0; i < n_cat_; ++i) {
            // dL/dx[b, i, k] = sum_j dL/dtokens[b, i, j] * E[i, k, j]
            for (size_t k = 0; k < vocab_max_; ++k) {
                double dx_val = 0.0;
                for (size_t j = 0; j < d_model_; ++j) {
                    double g = grad_output(b, i * d_model_ + j);
                    double E_ikj = E_(i * vocab_max_ + k, j);
                    dx_val += g * E_ikj;

                    // dL/dE[i, k, j] += input[b, i, k] * g
                    grad_E_(i * vocab_max_ + k, j) += last_input_(b, i * vocab_max_ + k) * g;

                    // dL/db[i, j] += g
                    grad_b_(i, j) += g;
                }
                dx(b, i * vocab_max_ + k) = dx_val;
            }
        }
    }
    return dx;
}

void CategoricalFeatureTokenizer::update_weights(double learning_rate) {
    for (size_t i = 0; i < E_.data.size(); ++i) {
        E_.data[i] -= learning_rate * grad_E_.data[i];
    }
    for (size_t i = 0; i < b_.data.size(); ++i) {
        b_.data[i] -= learning_rate * grad_b_.data[i];
    }
}

void CategoricalFeatureTokenizer::zero_grad() {
    grad_E_.fill(0.0);
    grad_b_.fill(0.0);
}

std::vector<Tensor*> CategoricalFeatureTokenizer::parameters() {
    return {&E_, &b_};
}

std::vector<Tensor*> CategoricalFeatureTokenizer::gradients() {
    return {&grad_E_, &grad_b_};
}

// ============================================================================
// FTTransformer — full model
// ============================================================================

FTTransformer::FTTransformer(size_t d_model, size_t n_classes,
                             size_t n_num, size_t n_cat, size_t vocab_max,
                             size_t num_heads, size_t num_blocks,
                             size_t ffn_dim)
    : d_model_(d_model),
      n_classes_(n_classes),
      n_num_(n_num),
      n_cat_(n_cat),
      vocab_max_(vocab_max),
      num_heads_(num_heads),
      num_blocks_(num_blocks),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      cls_token_(1, d_model),
      grad_cls_token_(1, d_model),
      final_ln_(d_model),
      clf_fc1_(d_model, ffn_dim_),
      clf_fc2_(ffn_dim_, n_classes),
      last_numerical_(0, 0),
      last_categorical_(0, 0, 0),
      last_concat_(0, 0),
      last_cls_hidden_(0, 0),
      last_clf_pre_relu_(0, 0),
      last_clf_logits_(0, 0),
      last_B_(0)
{
    if (d_model == 0)        throw std::invalid_argument("FTTransformer: d_model must be > 0");
    if (n_classes == 0)      throw std::invalid_argument("FTTransformer: n_classes must be > 0");
    if (d_model % num_heads != 0) throw std::invalid_argument("FTTransformer: d_model must be divisible by num_heads");
    if (num_blocks == 0)     throw std::invalid_argument("FTTransformer: num_blocks must be > 0");

    // CLS token init: small random
    for (size_t i = 0; i < cls_token_.data.size(); ++i) {
        cls_token_.data[i] = ((double) rand() / RAND_MAX - 0.5) * 0.05;
    }
    grad_cls_token_.fill(0.0);

    // Optional tokenizers
    if (n_num > 0) {
        num_tok_ = std::make_unique<NumericalFeatureTokenizer>(d_model, n_num);
    }
    if (n_cat > 0) {
        cat_tok_ = std::make_unique<CategoricalFeatureTokenizer>(d_model, n_cat, vocab_max);
    }

    // Stack of transformer blocks
    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        // num_kv_heads == num_heads → standard MHA
        blocks_.emplace_back(std::make_unique<GQABlock>(d_model, num_heads, num_heads, ffn_dim_));
    }
}

Tensor FTTransformer::forward(const Tensor& numerical, const Tensor& categorical) {
    size_t B = numerical.rows > 0 ? numerical.rows : categorical.rows;
    if (B == 0) throw std::invalid_argument("FTTransformer::forward: empty batch");
    if (n_num_ > 0 && numerical.rows != B) throw std::invalid_argument("FTTransformer::forward: num batch mismatch");
    if (n_cat_ > 0 && categorical.rows != B) throw std::invalid_argument("FTTransformer::forward: cat batch mismatch");

    last_B_ = B;
    last_numerical_ = numerical.clone();
    last_categorical_ = categorical.clone();

    // ---- Tokenization (numerical and/or categorical) ----
    size_t n_tok_features = n_num_ + n_cat_;
    Tensor features_flat;

    if (n_num_ > 0 && n_cat_ > 0) {
        // (B, n_num * d_model) and (B, n_cat * d_model) → concatenate along cols
        Tensor num_out = num_tok_->forward(numerical);
        Tensor cat_out = cat_tok_->forward(categorical);
        features_flat = num_out.concatenate(cat_out, /*along_cols=*/true);
    } else if (n_num_ > 0) {
        features_flat = num_tok_->forward(numerical);
    } else if (n_cat_ > 0) {
        features_flat = cat_tok_->forward(categorical);
    } else {
        throw std::invalid_argument("FTTransformer::forward: no features");
    }

    // ---- Prepend CLS token ----
    // features_flat is (B, n_tok_features * d_model).  We want to lay it out
    // as (B * total_tokens, d_model) for GQABlock (which treats rows = tokens,
    // cols = d_model features per token).
    size_t total_tokens = 1 + n_tok_features;
    Tensor tokens(B * total_tokens, d_model_);
    for (size_t b = 0; b < B; ++b) {
        // CLS at row b * total_tokens
        for (size_t j = 0; j < d_model_; ++j) {
            tokens(b * total_tokens, j) = cls_token_(0, j);
        }
        // Feature tokens: row b * total_tokens + (i + 1)
        for (size_t i = 0; i < n_tok_features; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                tokens(b * total_tokens + (i + 1), j) = features_flat(b, i * d_model_ + j);
            }
        }
    }
    last_concat_ = tokens.clone();

    // ---- Stack of GQABlocks ----
    Tensor x = tokens;
    for (auto& blk : blocks_) {
        x = blk->forward(x);
    }

    // ---- Final LayerNorm + extract CLS ----
    // x_ln has shape (B * total_tokens, d_model). LayerNorm is applied per-row.
    Tensor x_ln = final_ln_.forward(x);

    // Extract CLS at row b * total_tokens for each b.
    Tensor cls_hidden(B, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < d_model_; ++j) {
            cls_hidden(b, j) = x_ln(b * total_tokens, j);
        }
    }
    last_cls_hidden_ = cls_hidden.clone();

    // ---- Classifier MLP: d_model → ffn_dim (ReLU) → n_classes ----
    Tensor clf_pre = clf_fc1_.forward(cls_hidden);
    Tensor clf_act(clf_pre.rows, clf_pre.cols);
    for (size_t i = 0; i < clf_pre.data.size(); ++i) {
        clf_act.data[i] = clf_pre.data[i] > 0.0 ? clf_pre.data[i] : 0.0;
    }
    last_clf_pre_relu_ = clf_act.clone();

    Tensor logits = clf_fc2_.forward(clf_act);
    last_clf_logits_ = logits.clone();

    return logits;
}

FTTransformer::BackwardOutputs
FTTransformer::backward_full(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.rows != last_B_ || grad_output.cols != n_classes_) {
        throw std::invalid_argument("FTTransformer::backward: grad shape mismatch");
    }

    // ---- Backward through classifier MLP ----
    // logits = fc2(ReLU(fc1(cls_hidden)))
    // dL/dclf_act = grad_output @ W_fc2^T
    // dL/dcls_hidden = dL/dclf_act ⊙ ReLU'(clf_pre)  via Dense::backward paths
    Tensor grad_act = clf_fc2_.backward(grad_output, 0.0);  // (B, ffn_dim)
    // grad_act already includes the fc2 backward.

    // Now ReLU backward
    Tensor grad_pre(grad_act.rows, grad_act.cols);
    for (size_t i = 0; i < grad_act.data.size(); ++i) {
        grad_pre.data[i] = last_clf_pre_relu_.data[i] > 0.0 ? grad_act.data[i] : 0.0;
    }

    Tensor grad_cls_hidden = clf_fc1_.backward(grad_pre, 0.0);  // (B, d_model)

    // ---- Backward through final LayerNorm ----
    // grad_cls_hidden is (B, d_model). Place grad back into the
    // (B * total_tokens, d_model) buffer at CLS positions.
    size_t total_tokens_ln = 1 + n_num_ + n_cat_;
    Tensor grad_tokens(last_B_ * total_tokens_ln, d_model_);
    grad_tokens.fill(0.0);
    for (size_t b = 0; b < last_B_; ++b) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_tokens(b * total_tokens_ln, j) = grad_cls_hidden(b, j);
        }
    }
    Tensor grad_x_ln_input = final_ln_.backward(grad_tokens, 0.0);

    // ---- Backward through blocks ----
    Tensor grad_x = grad_x_ln_input;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_x = (*it)->backward(grad_x, 0.0);
    }
    // grad_x is now (B * total_tokens, d_model)

    // ---- Accumulate CLS-token gradient ----
    grad_cls_token_.fill(0.0);
    for (size_t b = 0; b < last_B_; ++b) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_cls_token_(0, j) += grad_x(b * total_tokens_ln, j);
        }
    }

    // ---- Split grad_x into (grad_num_tokens, grad_cat_tokens) ----
    BackwardOutputs result;
    size_t total_tokens = total_tokens_ln;

    if (n_num_ > 0) {
        Tensor grad_num_tokens(last_B_, n_num_ * d_model_);
        for (size_t b = 0; b < last_B_; ++b) {
            for (size_t i = 0; i < n_num_; ++i) {
                for (size_t j = 0; j < d_model_; ++j) {
                    grad_num_tokens(b, i * d_model_ + j) =
                        grad_x(b * total_tokens + (i + 1), j);
                }
            }
        }
        result.grad_numerical = num_tok_->backward(grad_num_tokens, 0.0);
    } else {
        result.grad_numerical = Tensor(0, 0);
    }

    if (n_cat_ > 0) {
        // Map grad_x tokens → grad_cat_emb tokens via the per-feature linear.
        Tensor grad_cat_emb(last_B_, n_cat_ * d_model_);
        for (size_t b = 0; b < last_B_; ++b) {
            for (size_t i = 0; i < n_cat_; ++i) {
                for (size_t j = 0; j < d_model_; ++j) {
                    grad_cat_emb(b, i * d_model_ + j) =
                        grad_x(b * total_tokens + (1 + n_num_ + i), j);
                }
            }
        }
        result.grad_categorical = cat_tok_->backward(grad_cat_emb, 0.0);
    } else {
        result.grad_categorical = Tensor(0, 0);
    }

    return result;
}

Tensor FTTransformer::backward(const Tensor& grad_output, double learning_rate) {
    auto out = backward_full(grad_output, learning_rate);

    // Pack into (B, n_num + n_cat * vocab_max) for the Layer contract.
    // If both numerical and categorical are present, concatenate along cols.
    // If only one is present, return that one directly.
    if (n_num_ == 0) {
        return out.grad_categorical;
    }
    if (n_cat_ == 0) {
        return out.grad_numerical;
    }
    return out.grad_numerical.concatenate(out.grad_categorical, /*along_cols=*/true);
}

Tensor FTTransformer::forward(const Tensor& /* input */) {
    // The single-Tensor signature is required by Layer but doesn't apply to
    // FTTransformer (it needs both numerical and categorical inputs).
    // Callers should use forward(numerical, categorical).
    throw std::logic_error(
        "FTTransformer: use forward(numerical, categorical); single-Tensor forward() is not supported");
}

void FTTransformer::update_weights(double learning_rate) {
    // CLS token
    for (size_t i = 0; i < cls_token_.data.size(); ++i) {
        cls_token_.data[i] -= learning_rate * grad_cls_token_.data[i];
    }
    // Tokenizers
    if (num_tok_) num_tok_->update_weights(learning_rate);
    if (cat_tok_) cat_tok_->update_weights(learning_rate);
    // Blocks
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    // Final LayerNorm + classifier
    final_ln_.update_weights(learning_rate);
    clf_fc1_.update_weights(learning_rate);
    clf_fc2_.update_weights(learning_rate);
}

void FTTransformer::zero_grad() {
    grad_cls_token_.fill(0.0);
    if (num_tok_) num_tok_->zero_grad();
    if (cat_tok_) cat_tok_->zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
    final_ln_.zero_grad();
    clf_fc1_.zero_grad();
    clf_fc2_.zero_grad();
}

std::vector<Tensor*> FTTransformer::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&cls_token_);
    if (num_tok_) {
        auto np = num_tok_->parameters();
        p.insert(p.end(), np.begin(), np.end());
    }
    if (cat_tok_) {
        auto cp = cat_tok_->parameters();
        p.insert(p.end(), cp.begin(), cp.end());
    }
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto lp = final_ln_.parameters();
    p.insert(p.end(), lp.begin(), lp.end());
    auto c1p = clf_fc1_.parameters();
    p.insert(p.end(), c1p.begin(), c1p.end());
    auto c2p = clf_fc2_.parameters();
    p.insert(p.end(), c2p.begin(), c2p.end());
    return p;
}

std::vector<Tensor*> FTTransformer::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_cls_token_);
    if (num_tok_) {
        auto ng = num_tok_->gradients();
        g.insert(g.end(), ng.begin(), ng.end());
    }
    if (cat_tok_) {
        auto cg = cat_tok_->gradients();
        g.insert(g.end(), cg.begin(), cg.end());
    }
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto lg = final_ln_.gradients();
    g.insert(g.end(), lg.begin(), lg.end());
    auto c1g = clf_fc1_.gradients();
    g.insert(g.end(), c1g.begin(), c1g.end());
    auto c2g = clf_fc2_.gradients();
    g.insert(g.end(), c2g.begin(), c2g.end());
    return g;
}

Tensor FTTransformer::get_weights() const {
    // For diagnostics — return the CLS token (smallest single-Tensor surface).
    return cls_token_;
}

Tensor FTTransformer::get_gradients() const {
    return grad_cls_token_;
}