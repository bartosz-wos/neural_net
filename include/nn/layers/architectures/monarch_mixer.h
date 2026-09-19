#ifndef MONARCH_MIXER_H
#define MONARCH_MIXER_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cstdint>
#include <memory>

// ============================================================================
// Monarch Mixer (Dao et al. 2022, https://arxiv.org/abs/2204.00945,
//   "Monarch: Expressive Structured Matrices for Efficient Accurate Deep Learning")
//
// A Monarch matrix of dim n with block-size b is M = P · B · Q where:
//   - P, Q are FIXED permutations of size n (random init, frozen during training)
//   - B is block-diagonal with n/b^2 blocks of size (b, b) (the only learnable part)
//
// Constraints:
//   - n must be a perfect square (n = m^2) so the block-diagonal is square.
//   - b^2 must divide n (block count = n/b^2 must be (m/b)^2).
//   - b must divide m (= sqrt(n)).
//
// The Monarch Mixer REPLACES both the self-attention sublayer AND the MLP
// sublayer in a Transformer with Monarch matrices applied to the sequence axis
// (one Monarch per channel, mixing the seq_len positions) and to the channel
// axis (one shared Monarch applied per-token to the d_model channels). This
// gives O(N log N) sequence mixing and O(d log d) channel mixing at strictly
// less cost than O(N^2) attention or O(d^2) MLP.
// ============================================================================


// ============================================================================
// MonarchMatrix — a single Monarch matrix primitive. Applies M = P·B·Q to
// a (B, n) input, returning (B, n). Backward is the linear chain P^T·B^T·Q^T.
//
// For num_blocks > 1, multiple Monarch matrices are stacked: M = M_k · ... · M_1.
// ============================================================================
class MonarchMatrix : public Layer {
public:
    // n: matrix dimension (must be a perfect square).
    // b: block side (must divide sqrt(n) evenly).
    // num_blocks: number of stacked Monarch matrices (paper §2.4 — more expressive).
    // perm_seed: RNG seed for the random permutations P, Q.
    MonarchMatrix(size_t n, size_t b, size_t num_blocks = 1, uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMatrix"; }

    size_t n() const { return n_; }
    size_t b() const { return b_; }
    size_t m() const { return m_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t num_mb() const { return num_mb_; }

    // Per-Monarch perm accessors (indexed by Monarch index in [0, num_blocks)).
    const std::vector<size_t>& P(size_t i) const { return perm_P_[i]; }
    const std::vector<size_t>& Q(size_t i) const { return perm_Q_[i]; }

    // Block-values tensor: stored flat as (num_blocks * num_mb * b * b).
    // Accessor for flat (idx, block, i, j) → idx * num_mb * b * b + block * b * b + i * b + j.
    const Tensor& block_values() const { return block_values_; }
    Tensor& block_values() { return block_values_; }

private:
    size_t n_;
    size_t b_;
    size_t m_;             // sqrt(n)
    size_t num_blocks_;
    size_t num_mb_;        // n / b^2 (blocks per Monarch)

    std::vector<std::vector<size_t>> perm_P_;  // [num_blocks][n]
    std::vector<std::vector<size_t>> perm_Q_;  // [num_blocks][n]

    Tensor block_values_;       // (num_blocks * num_mb * b * b,) flat
    Tensor grad_block_values_;  // same shape

    // Caches: intermediates after each Monarch's P/B/Q application.
    // interm_Q_[k]  = Q_k · x_{k-1}       (B, n)
    // interm_BQ_[k] = B_k · (Q_k · x_{k-1}) (B, n)
    // interm_out_[k] = P_k · (B_k · ...)    (B, n)  — same as x_k = output of Monarch k
    std::vector<Tensor> interm_Q_;
    std::vector<Tensor> interm_BQ_;
    std::vector<Tensor> interm_out_;
    Tensor last_input_;   // (B, n)
};


// ============================================================================
// MonarchSequenceMix — applies (per-channel) Monarch matrices over the sequence
// axis. Each channel c of d_model has its OWN Monarch matrix, so the per-channel
// weights are independent. Input is (B, seq_len, d_model); output is the same.
// num_blocks controls how many Monarch matrices are stacked per channel.
// ============================================================================
class MonarchSequenceMix : public Layer {
public:
    MonarchSequenceMix(size_t seq_len, size_t d_model,
                       size_t block_size, size_t num_blocks = 1,
                       uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchSequenceMix"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }

private:
    size_t seq_len_, d_model_, block_size_, num_blocks_;
    size_t m_;          // sqrt(seq_len)
    size_t num_mb_;     // seq_len / block_size^2
    size_t blocks_per_channel_;  // num_blocks * num_mb

    Tensor blocks_;       // (d_model * num_blocks * num_mb * block_size * block_size,) flat
    Tensor grad_blocks_;  // same shape

    // Per-channel, per-Monarch permutations.
    std::vector<std::vector<std::vector<size_t>>> perm_P_;  // [d_model][num_blocks][seq_len]
    std::vector<std::vector<std::vector<size_t>>> perm_Q_;  // [d_model][num_blocks][seq_len]

    Tensor last_input_;  // (B, seq_len, d_model) — cached for backward
};


// ============================================================================
// MonarchChannelMix — applies a single (shared) Monarch matrix over the channel
// axis, per-token. Each token gets the same Monarch applied to its d_model-dim
// vector. Input is (B, seq_len, d_model); output is the same.
// num_blocks controls how many Monarch matrices are stacked (paper §2.4).
// ============================================================================
class MonarchChannelMix : public Layer {
public:
    MonarchChannelMix(size_t seq_len, size_t d_model, size_t block_size,
                      size_t num_blocks = 1, uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchChannelMix"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }

private:
    size_t seq_len_, d_model_, block_size_, num_blocks_;
    size_t m_;          // sqrt(d_model)
    size_t num_mb_;     // d_model / block_size^2

    Tensor blocks_;       // (num_blocks * num_mb * block_size * block_size,) flat
    Tensor grad_blocks_;  // same shape

    std::vector<std::vector<size_t>> perm_P_;  // [num_blocks][d_model]
    std::vector<std::vector<size_t>> perm_Q_;  // [num_blocks][d_model]

    Tensor last_input_;  // (B, seq_len, d_model) — cached for backward
};


// ============================================================================
// MonarchMixerBlock — pre-LN + MonarchSequenceMix + residual + pre-LN +
// MonarchChannelMix + residual. The Monarch Transformer block.
// ============================================================================
class MonarchMixerBlock : public Layer {
public:
    MonarchMixerBlock(size_t seq_len, size_t d_model,
                      size_t block_size_seq, size_t block_size_ch,
                      size_t num_blocks_seq = 1, size_t num_blocks_ch = 1,
                      uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMixerBlock"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }

    const LayerNorm& ln1() const { return ln1_; }
    const MonarchSequenceMix& seq_mix() const { return seq_mix_; }
    const LayerNorm& ln2() const { return ln2_; }
    const MonarchChannelMix& ch_mix() const { return ch_mix_; }

private:
    size_t seq_len_, d_model_;
    LayerNorm ln1_;
    MonarchSequenceMix seq_mix_;
    LayerNorm ln2_;
    MonarchChannelMix ch_mix_;

    // Caches
    Tensor last_input_;        // (B, seq_len, d_model)
    Tensor last_ln1_out_;      // (B, seq_len, d_model)
    Tensor last_res1_;         // (B, seq_len, d_model) = last_input_ + last_seq_mix_out_
    Tensor last_ln2_out_;      // (B, seq_len, d_model)
    Tensor last_output_;       // (B, seq_len, d_model) = last_res1_ + last_ch_mix_out_
};


// ============================================================================
// MonarchMixerModel — embed (Dense(input_dim → d_model)) + stack of
// MonarchMixerBlocks + final LayerNorm + mean-pool over sequence + classifier.
// ============================================================================
class MonarchMixerModel : public Layer {
public:
    MonarchMixerModel(size_t input_dim, size_t d_model, size_t output_dim,
                      size_t seq_len, size_t num_blocks = 2,
                      size_t block_size_seq = 0,   // 0 → sqrt(seq_len)
                      size_t block_size_ch = 0,   // 0 → sqrt(d_model)
                      uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMixerModel"; }

    size_t input_dim() const { return input_dim_; }
    size_t d_model() const { return d_model_; }
    size_t output_dim() const { return output_dim_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_blocks() const { return num_blocks_; }

    const Dense& embed() const { return embed_; }
    const Dense& classifier() const { return classifier_; }
    const LayerNorm& final_ln() const { return final_ln_; }

private:
    size_t input_dim_, d_model_, output_dim_, seq_len_, num_blocks_;
    size_t block_size_seq_, block_size_ch_;
    Dense embed_;
    std::vector<std::unique_ptr<MonarchMixerBlock>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;

    // Caches for backward
    Tensor last_embed_;       // (B, seq_len, d_model) — post-embed, pre-block
    std::vector<Tensor> last_block_outs_;  // post-block outputs (B, seq_len, d_model)
    Tensor last_pooled_;      // (B, d_model) — pre-classifier
    Tensor last_logits_;      // (B, output_dim) — post-classifier
};

#endif // MONARCH_MIXER_H