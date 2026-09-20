#ifndef SET_TRANSFORMER_H
#define SET_TRANSFORMER_H

#include "../../core/layer.h"
#include <vector>

// ============================================================================
// Set Transformer — Lee et al. 2019 (ICML)
//   https://arxiv.org/abs/1810.00825
// ----------------------------------------------------------------------------
// A permutation-invariant architecture for set-structured inputs. Four
// composable building blocks:
//
//   (1) MAB(Q, K): Multihead Attention Block with Q from one source and K/V
//       from another (cross-attention). Pre-LN → MHA → residual →
//       pre-LN → FFN(GELU) → residual.
//
//   (2) SAB(X): Set Attention Block. SAB(X) = MAB(X, X).
//
//   (3) ISAB(X, I): Induced Set Attention Block. ISAB(X, I) = MAB(MAB(X, I), I).
//
//   (4) PMA(X, S): Pooling by Multihead Attention. PMA(X, S) = MAB(S, X).
//
//   (5) SetTransformer(input_dim, hidden_dim, num_heads, num_inds, num_seeds,
//       num_enc_blocks, num_dec_blocks, output_dim): full model.
//
// Convention (matches repo-wide): tensor layout is (n_tokens, d_model).
// Rows = tokens, columns = features. For Layer::forward(input), input must
// have cols == d_model and arbitrary rows == n_tokens.
//
// Single-batch (B=1) for v1 — standard Set Transformer convention.
//
// LayerNorm: we maintain our own mean/var caches because the q and kv inputs
// to MAB both go through pre-LN1 (with shared gamma/beta). The base LayerNorm
// has a single cache, so we'd lose information if we called it twice.
// We use plain Tensor fields for gamma and beta (not the LayerNorm object)
// and compute normalization manually.
// ============================================================================

class MAB : public Layer {
public:
    MAB(size_t d_model, size_t num_heads, size_t ffn_hidden = 0);

    // Cross-attention forward: Q from `q_input`, K/V from `kv_input`. Both
    // must have cols == d_model. Returns output of shape (n_q, d_model).
    Tensor forward(const Tensor& q_input, const Tensor& kv_input);

    // Single-input forward (self-attention case). Sets an internal flag so
    // backward knows to combine d_q and d_kv into one.
    Tensor forward(const Tensor& input) override;

    // Layer backward: returns d_q_input. (Self-attention: also incorporates
    // the kv side via the combine step in backward_full.)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    struct GradPair { Tensor d_q; Tensor d_kv; };
    GradPair backward_full(const Tensor& grad_output, double learning_rate);

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return W_q_; }
    Tensor get_gradients() const override { return grad_W_q_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "MAB"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }

    // Public weight accessors (for FD perturbation / tests)
    const Tensor& W_q() const { return W_q_; }
    const Tensor& W_k() const { return W_k_; }
    const Tensor& W_v() const { return W_v_; }
    const Tensor& W_O() const { return W_O_; }
    const Tensor& W_ffn() const { return W_ffn_; }
    const Tensor& W_ffn_out() const { return W_ffn_out_; }
    Tensor& W_q_mutable() { return W_q_; }
    Tensor& W_k_mutable() { return W_k_; }
    Tensor& W_v_mutable() { return W_v_; }
    Tensor& W_O_mutable() { return W_O_; }
    Tensor& W_ffn_mutable() { return W_ffn_; }
    Tensor& W_ffn_out_mutable() { return W_ffn_out_; }
    const Tensor& grad_W_q() const { return grad_W_q_; }
    const Tensor& grad_W_k() const { return grad_W_k_; }
    const Tensor& grad_W_v() const { return grad_W_v_; }
    const Tensor& grad_W_O() const { return grad_W_O_; }
    const Tensor& grad_W_ffn() const { return grad_W_ffn_; }
    const Tensor& grad_W_ffn_out() const { return grad_W_ffn_out_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t d_k_;
    size_t ffn_hidden_;
    bool last_was_self_;

    // MHA parameters (shape: (d_model, d_model))
    Tensor W_q_, W_k_, W_v_, W_O_;
    Tensor grad_W_q_, grad_W_k_, grad_W_v_, grad_W_O_;

    // FFN parameters: W_ffn (ffn_hidden, d_model), b_ffn (1, ffn_hidden),
    // W_ffn_out (d_model, ffn_hidden), b_ffn_out (1, d_model).
    Tensor W_ffn_, b_ffn_, W_ffn_out_, b_ffn_out_;
    Tensor grad_W_ffn_, grad_b_ffn_, grad_W_ffn_out_, grad_b_ffn_out_;

    // LayerNorm gamma/beta (1, d_model). gamma1/beta1 used for pre-LN1
    // (applied to both q and kv), gamma2/beta2 for pre-LN2.
    Tensor ln1_gamma_, ln1_beta_;
    Tensor ln2_gamma_, ln2_beta_;
    Tensor grad_ln1_gamma_, grad_ln1_beta_;
    Tensor grad_ln2_gamma_, grad_ln2_beta_;
    static constexpr double LN_EPS_ = 1e-5;

    // Forward caches (per-call, overwritten on next forward).
    Tensor last_q_in_raw_;     // (n_q, d_model)
    Tensor last_kv_in_raw_;    // (n_kv, d_model)

    Tensor last_q_normed_;     // (n_q, d_model)
    Tensor last_kv_normed_;    // (n_kv, d_model)
    // LN1 caches — q-side: mean (1, n_q), var (1, n_q), input (n_q, d_model)
    Tensor last_q_mean_, last_q_var_, last_q_pre_norm_;
    Tensor last_kv_mean_, last_kv_var_, last_kv_pre_norm_;

    Tensor last_Q_;            // (n_q, d_model)
    Tensor last_K_;            // (n_kv, d_model)
    Tensor last_V_;            // (n_kv, d_model)

    Tensor last_A_;            // (num_heads * n_q, n_kv) — softmax probs, flattened
    Tensor last_attn_concat_;  // (n_q, d_model)
    Tensor last_attn_out_;     // (n_q, d_model) — attn_concat @ W_O
    Tensor last_res1_;         // (n_q, d_model) — attn_out + q_in

    // LN2 caches — mean (1, n_q), var (1, n_q), input (n_q, d_model)
    Tensor last_res1_mean_, last_res1_var_, last_res1_pre_norm_;
    Tensor last_ln2_out_;      // (n_q, d_model)

    Tensor last_ffn_pre_;      // (n_q, ffn_hidden) — pre-activation
    Tensor last_ffn_act_;      // (n_q, ffn_hidden) — GELU(pre)
    Tensor last_ffn_out_;      // (n_q, d_model) — act @ W_ffn_out^T + b_ffn_out
    Tensor last_res2_;         // (n_q, d_model)
};

// ============================================================================
// SAB — Set Attention Block. SAB(X) = MAB(X, X).
// ============================================================================
class SAB : public Layer {
public:
    SAB(size_t d_model, size_t num_heads, size_t ffn_hidden = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return mab_.get_weights(); }
    Tensor get_gradients() const override { return mab_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "SAB"; }

    MAB& mab() { return mab_; }

private:
    MAB mab_;
};

// ============================================================================
// ISAB — Induced Set Attention Block. ISAB(X, I) = MAB(MAB(X, I), I).
// ============================================================================
class ISAB : public Layer {
public:
    ISAB(size_t d_model, size_t num_heads, size_t num_inds, size_t ffn_hidden = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return I_; }
    Tensor get_gradients() const override { return grad_I_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "ISAB"; }

    const Tensor& I() const { return I_; }

private:
    MAB mab1_;
    MAB mab2_;
    Tensor I_;        // (num_inds, d_model) — induced points, layout matches MAB input
    Tensor grad_I_;
};

// ============================================================================
// PMA — Pooling by Multihead Attention. PMA(X, S) = MAB(S, X).
// ============================================================================
class PMA : public Layer {
public:
    PMA(size_t d_model, size_t num_heads, size_t num_seeds, size_t ffn_hidden = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return S_; }
    Tensor get_gradients() const override { return grad_S_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "PMA"; }

    const Tensor& S() const { return S_; }
    size_t num_seeds() const { return num_seeds_; }

private:
    size_t num_seeds_;
    MAB mab_;
    Tensor S_;         // (num_seeds, d_model) — seed vectors, layout matches MAB input
    Tensor grad_S_;
};

// ============================================================================
// SetTransformer — full encoder-decoder model.
//
// Architecture:
//   proj = X @ W_in^T + b_in        (n_tokens, hidden_dim)
//   for i in enc_blocks: X = ISAB_i(X)
//   pooled = PMA(X, S)              (num_seeds, hidden_dim)
//   for i in dec_blocks: pooled = SAB_i(pooled)
//   flat = pooled.flatten()         (1, num_seeds * hidden_dim)
//   out = flat @ head^T + b_head    (1, output_dim)
//
// Single-batch (B=1) for v1.
// ============================================================================
class SetTransformer : public Layer {
public:
    SetTransformer(size_t input_dim, size_t hidden_dim, size_t num_heads,
                   size_t num_inds, size_t num_seeds,
                   size_t num_enc_blocks, size_t num_dec_blocks,
                   size_t output_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return head_.weights; }
    Tensor get_gradients() const override { return head_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "SetTransformer"; }

    size_t num_parameters() const;

private:
    size_t input_dim_, hidden_dim_, num_heads_;
    size_t num_inds_, num_seeds_, num_enc_blocks_, num_dec_blocks_, output_dim_;

    Dense input_proj_;       // (input_dim → hidden_dim) — manual forward for shape control
    std::vector<ISAB> enc_;
    PMA pma_;
    std::vector<SAB> dec_;
    Dense head_;             // (num_seeds * hidden_dim → output_dim) — manual forward

    // Forward caches
    Tensor last_input_;
    Tensor last_proj_;
    std::vector<Tensor> last_enc_;   // (n_tokens, hidden_dim) after each encoder
    Tensor last_pma_;                 // (num_seeds, hidden_dim)
    std::vector<Tensor> last_dec_;   // (num_seeds, hidden_dim) after each decoder
    Tensor last_flat_;                // (1, num_seeds * hidden_dim)
};

#endif // SET_TRANSFORMER_H