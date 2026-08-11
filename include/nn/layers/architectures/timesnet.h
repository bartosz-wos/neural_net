#ifndef TIMESNET_H
#define TIMESNET_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../normalization/layer_norm.h"
#include "../convolutions/conv_layer.h"
#include <vector>
#include <complex>
#include <cmath>
#include <memory>

// ============================================================================
// TimesNet (Wu et al. 2023, ICLR) — "TimesNet: Temporal 2D-Variation Modeling
// for General Time Series Analysis".  https://arxiv.org/abs/2210.02186
//
// Innovation: a 1-D time series contains multiple periodicities. TimesNet
// finds the top-K dominant periods via FFT, RESHAPES the (B, T, D) tensor
// into a 2-D (B, period, T/period, D) tensor (one slice per period), then
// runs a shared 2-D conv block over the stacked slices. This captures BOTH
// intraperiod (across the period axis) and interperiod (across the T/period
// axis) variations in one operator.
//
// Canonical TimesBlock (simplified for tractability):
//   fft_amp = FFT_amplitude(x)                              // (B, T)
//   p_1, ..., p_K = topk_periods(fft_amp, K)                // K integers
//   for k in 1..K:
//     x_k = reshape(x, (p_k, T/p_k))                        // (B, p_k, T/p_k, D)
//   stack into (B*K, max_period, max_len, D)
//   transpose to (B*K, D, max_period, max_len)  for Conv2D channel-major
//   y = Conv2DBlock(y)                                      // 2D conv stack
//   un-stack back, average over K
//   x = x + drop_path(conv_block(LN(x)))                    // residual
//   x = x + drop_path(ffn(LN(x)))                           // residual
//
// Tensor convention: all tensors in this file use (N, features) flat layout.
//   - For block input/output: (B*T, d_model)
//   - For TimesNet input:     (B*T, in_dim)
//   - For TimesNet output:    (B*pred_len, out_dim)    — caller knows B = rows/pred_len
// ============================================================================

// ----------------------------------------------------------------------------
// FFT helper: O(N^2) discrete Fourier transform, real input → complex output.
// Used for the small L ≤ 64 test cases; Cooley-Tukey is a clean drop-in
// upgrade but not needed for correctness verification.
// ----------------------------------------------------------------------------
std::vector<std::complex<double>> naive_dft(const std::vector<double>& x);

// ----------------------------------------------------------------------------
// Top-K dominant periods from a 1-D amplitude spectrum.
// amplitude[k] = |DFT[k]| for k = 0..N-1. We scan k=1..N/2 (skipping DC).
// Returns K periods sorted by descending amplitude. Each "period" is N/k
// (rounded). For non-divisor k, we keep the integer N/k and let the reshape
// pad to a common shape.
// ----------------------------------------------------------------------------
std::vector<int> topk_periods(const std::vector<double>& amplitude, int K);

// ----------------------------------------------------------------------------
// Conv2DBlock: 2 × (Conv2D 3×3, GELU) + 1×1 Conv2D projection back.
// Operates on (N, in_ch*H*W) flat input (channel-major: c*H*W + h*W + w).
// We always construct with H=W=seq_len so the conv block can be reused
// across all K periods (each period's tensor is zero-padded up to seq_len×seq_len).
// ----------------------------------------------------------------------------
class Conv2DBlock {
public:
    int in_channels_, hidden_channels_, out_channels_;
    int H_, W_;
    Conv2D conv1_;   // in_ch → hidden_ch, 3×3, pad 1
    Conv2D conv2_;   // hidden_ch → hidden_ch, 3×3, pad 1
    Conv2D proj_;    // hidden_ch → out_ch, 1×1

    // Caches
    Tensor last_input_;
    Tensor last_h1_pre_gelu_;
    Tensor last_h2_pre_gelu_;

    Conv2DBlock(int in_channels, int hidden_channels, int out_channels,
                int H, int W);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output, double learning_rate);
    void update_weights(double learning_rate);
    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

// ----------------------------------------------------------------------------
// TimesBlock: pre-norm → FFT period select → reshape → 2-D conv → average
//             → residual → pre-norm FFN → residual
//
// Input/output: (B*T, d_model) flat (where the row count = B*T tokens).
// ----------------------------------------------------------------------------
class TimesBlock : public Layer {
public:
    int d_model_;
    int seq_len_;
    int top_k_;
    int d_ff_;

    LayerNorm ln1_;   // before conv path
    LayerNorm ln2_;   // before FFN
    Conv2DBlock conv_block_;
    Dense ffn1_;
    Dense ffn2_;

    // Caches for backward
    Tensor last_input_;          // (B*T, d_model)
    Tensor last_ln1_out_;        // (B*T, d_model)
    Tensor last_ln2_out_;        // (B*T, d_model)
    Tensor last_ffn1_pre_gelu_;  // (B*T, d_ff)
    Tensor last_conv_out_;       // (B, T, d_model) — the conv block output before residual
    std::vector<int> last_periods_;
    int last_max_period_ = 0;
    int last_max_len_ = 0;
    bool last_used_fixed_ = false;

    // Fixed-period mode for FD gradient tests (skips FFT selection).
    std::vector<int> fixed_periods_;
    bool use_fixed_periods_ = false;

    void set_fixed_periods(std::vector<int> p) { fixed_periods_ = p; use_fixed_periods_ = true; }
    void clear_fixed_periods() { use_fixed_periods_ = false; fixed_periods_.clear(); }

    TimesBlock(int d_model, int seq_len, int top_k = 5, int d_ff = 0);
    ~TimesBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return ffn1_.get_weights(); }
    Tensor get_gradients() const override { return ffn1_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "TimesBlock"; }
};

// ----------------------------------------------------------------------------
// TimesNet: full forecasting model.
// Input:  (B*T, in_dim)   flat
// Output: (B*pred_len, out_dim)   flat
//
// Pipeline:
//   embed: (B*T, in_dim) → (B*T, d_model)
//   stack of TimesBlocks (input/output (B*T, d_model))
//   final_ln over (B*T, d_model)
//   slice last pred_len tokens per batch → (B*pred_len, d_model)
//   projector: (B*pred_len, d_model) → (B*pred_len, out_dim)
// ----------------------------------------------------------------------------
class TimesNet : public Layer {
public:
    int in_dim_;
    int out_dim_;
    int d_model_;
    int seq_len_;
    int pred_len_;
    int e_layers_;
    int top_k_;

    Dense embed_;
    std::vector<std::unique_ptr<TimesBlock>> blocks_;
    LayerNorm final_ln_;
    Dense projector_;

    Tensor last_input_;
    Tensor last_embed_out_;

    TimesNet(int in_dim, int out_dim, int seq_len, int pred_len,
             int d_model = 64, int e_layers = 2, int top_k = 5,
             int d_ff = 0);
    ~TimesNet() override = default;

    void set_fixed_periods(std::vector<int> p) {
        for (auto& blk : blocks_) blk->set_fixed_periods(p);
    }
    void clear_fixed_periods() {
        for (auto& blk : blocks_) blk->clear_fixed_periods();
    }

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return projector_.get_weights(); }
    Tensor get_gradients() const override { return projector_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "TimesNet"; }
};

#endif
