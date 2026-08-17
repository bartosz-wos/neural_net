#ifndef XLSTM_BLOCK_H
#define XLSTM_BLOCK_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/xlstm.h"
#include "../recurrent/mlstm.h"
#include <vector>
#include <memory>

// ============================================================================
// xLSTM Block (Beck et al. 2024, https://arxiv.org/abs/2404.05704,
// "xLSTM: Extended Long Short-Term Memory"), §5.
//
// The canonical xLSTM architecture unit. A block composes one or both xLSTM
// cell variants (sLSTM scalar-memory + mLSTM matrix-memory) with a 2-layer
// dense FFN, all under pre-norm residuals:
//
//   mixer_sub = slstm_proj(sLSTM(LN_1(x)))                  // always
//             [+ mlstm_proj(mLSTM(LN_1(x)))]                // when MLSTM_AFTER or BOTH_PARALLEL
//   ffn_sub   = ffn_proj2(GELU(ffn_proj1(LN_2(x + mixer_sub))))
//   out       = x + mixer_sub + ffn_sub
//
// Three composition modes for the mixer sublayer:
//
//   SLSTM_ONLY     — only sLSTM is used (paper §A.1 default)
//   MLSTM_AFTER    — sLSTM + mLSTM summed (sLSTM first, then mLSTM added)
//   BOTH_PARALLEL  — same as MLSTM_AFTER (paper §5 actually stacks them
//                    sequentially, but the parallel variant is a clean
//                    drop-in and is what most production implementations
//                    ship for parallelism)
//
// Layout convention: (T, d_model) end-to-end. sLSTM/mLSTM cells operate on
// (T, hidden); the block wraps each cell with a Dense(hidden, d_model)
// projector so the output shape matches the residual stream.
//
// State: each cell owns its own recurrence state (sLSTM: scalar c, normalizer
// n, stabilizer m; mLSTM: matrix C, normalizer N, stabilizer m). The block
// itself owns no state — state is per-cell, per-block (multi-block stacks
// have independent state per block).
//
// We follow the **pre-norm residual** convention used throughout this repo
// (Griffin, Jamba, Conformer), rather than the post-LN form in the paper.
// The gradient chain is otherwise identical to the paper's §5 specification.
// ============================================================================

enum class XLSTMCellType {
    SLSTM_ONLY = 0,
    MLSTM_AFTER = 1,
    BOTH_PARALLEL = 2
};

class XLSTMBlock : public Layer {
public:
    size_t d_model_;
    size_t slstm_hidden_;
    size_t mlstm_hidden_;
    size_t ffn_mult_;
    size_t ffn_hidden_;
    XLSTMCellType cell_type_;

    // Sublayers (public for test introspection)
    LayerNorm ln1_;                            // pre-norm for mixer
    LayerNorm ln2_;                            // pre-norm for FFN

    SLSTMCell slstm_;                          // (T, d_model) -> (T, slstm_hidden)
    Dense slstm_proj_;                         // (slstm_hidden, d_model) -> (T, d_model)
    MLSTMCell mlstm_;                          // (T, d_model) -> (T, mlstm_hidden) — only used in MLSTM_AFTER / BOTH_PARALLEL
    Dense mlstm_proj_;                         // (mlstm_hidden, d_model) -> (T, d_model) — only used in MLSTM_AFTER / BOTH_PARALLEL

    Dense ffn_proj1_;                          // (d_model, ffn_hidden) — FFN up-projection
    Dense ffn_proj2_;                          // (ffn_hidden, d_model) — FFN down-projection

    // Forward caches (public for tests)
    Tensor last_input;            // (T, d_model)
    Tensor last_ln1_out;          // (T, d_model)
    Tensor last_slstm_h;          // (T, slstm_hidden)
    Tensor last_slstm_proj;       // (T, d_model)
    Tensor last_mlstm_h;          // (T, mlstm_hidden)
    Tensor last_mlstm_proj;       // (T, d_model)
    Tensor last_mixer;            // (T, d_model) — sum of last_slstm_proj [+ last_mlstm_proj]
    Tensor last_residual1;        // (T, d_model) — x + last_mixer
    Tensor last_ln2_out;          // (T, d_model)
    Tensor last_ffn_hidden;       // (T, ffn_hidden)
    Tensor last_ffn_act;          // (T, ffn_hidden) — gelu(ffn_hidden)
    Tensor last_ffn_out;          // (T, d_model)

    XLSTMBlock(size_t d_model,
               size_t slstm_hidden,
               size_t mlstm_hidden = 0,
               size_t ffn_mult = 2,
               XLSTMCellType cell_type = XLSTMCellType::SLSTM_ONLY);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ln1_.gamma; }
    Tensor get_gradients() const override { return ln1_.grad_gamma_; }
    std::string name() const override { return "XLSTMBlock"; }

    // Deep-copy all learnable parameters from `other` into this block. Both
    // blocks must have identical shape configuration. Useful for determinism
    // tests.
    void copy_params_from(const XLSTMBlock& other);

    size_t num_blocks() const { return 1; }   // for stack contract parity

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t slstm_hidden() const { return slstm_hidden_; }
    size_t mlstm_hidden() const { return mlstm_hidden_; }
    size_t ffn_mult() const { return ffn_mult_; }
    XLSTMCellType cell_type() const { return cell_type_; }
    size_t count_parameters() const;
};

// Stack of XLSTMBlock + final LayerNorm + classifier. Multi-block stacks
// have independent cell state per block — block 0's h_T is NOT fed into
// block 1 as the initial hidden state. (The paper allows either mode; we
// follow the standard "fresh state per layer" convention used by all other
// stacks in this repo — Griffin, Jamba, Mamba-3 stacks — which gives better
// gradient flow than warm-starting each layer with the previous hidden.)
class XLSTMModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;

    Dense embed_;                                // (input_dim, d_model)
    std::vector<std::unique_ptr<XLSTMBlock>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;                           // (d_model, output_dim)

    XLSTMModel(size_t input_dim, size_t d_model, size_t output_dim,
               size_t num_layers, size_t slstm_hidden,
               size_t mlstm_hidden = 0, size_t ffn_mult = 2,
               XLSTMCellType cell_type = XLSTMCellType::SLSTM_ONLY);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed_.weights; }
    Tensor get_gradients() const override { return embed_.grad_weights; }
    std::string name() const override { return "XLSTMModel"; }
};

#endif // XLSTM_BLOCK_H