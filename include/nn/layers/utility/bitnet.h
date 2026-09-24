#ifndef BITNET_H
#define BITNET_H

#include "../../core/layer.h"
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// BitNet b1.58 — Ternary Linear Layer
//   Ma, Wang, Yang, Wei, Wang, Ma — 2024
//   "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
//   https://arxiv.org/abs/2402.17764
//
// Drop-in replacement for `Dense` where the WEIGHTS are quantized to
// ternary {-1, 0, +1} via absmean quantization (paper §3.1):
//
//   scale_w[o] = (1 / d_in) * Σ_j |W[o, j]|           // per-output-unit scale
//   W_hat[o,j] = clamp(W[o, j] / scale_w[o], -1, +1)
//   W_q[o, j]  = round(W_hat[o, j])                  // ∈ {-1, 0, +1} exactly
//
// Forward (paper Eq. 4):
//   Y = X · W_q^T · diag(scale_w)
//
// Straight-Through Estimator (STE) backward (paper §3.2): the round() is
// treated as identity in backward, EXCEPT at the clamp boundary:
//   ∂W_q/∂W ≈ (1 / scale_w[o])  if |W[o, j] / scale_w[o]| ≤ 1
//             0                  otherwise
// Implemented as a per-element mask M[o, j] = 1 iff |W[o, j]/scale_w[o]| ≤ 1.
//
// v1 implements weight-only quantization. Activation 8-bit absmean
// quantization (paper §3.2) is left for v2; the comment marks the seam.
//
// v1 choices:
//   - W is fp32 (not quantizing the storage), only the FORWARD uses W_q.
//   - scale_w is a learned per-output-unit parameter (paper §3.2 shows
//     a closed-form gradient through it; we expose grad_scale_w).
//   - Causal single-sample forward shape (N, d_in) -> (N, d_out).
// ============================================================================

class BitLinear : public Layer {
public:
    // d_in: input feature dim. d_out: output feature dim.
    // init: weight init scheme, default "xavier" (then absmean-quantized
    //       implicitly because W_q := round(W / scale_w) gives {-,0,+}).
    BitLinear(size_t d_in, size_t d_out,
              const std::string& init = "xavier");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_; }
    Tensor get_gradients() const override { return grad_W_; }
    std::string name() const override { return "BitLinear"; }

    // Accessors
    size_t d_in() const { return d_in_; }
    size_t d_out() const { return d_out_; }

    const Tensor& weights() const { return W_; }
    Tensor& weights() { return W_; }       // mutable for tests
    const Tensor& scale() const { return scale_w_; }
    Tensor& scale() { return scale_w_; }   // mutable for tests

    // Cached quantized weights from the most recent forward (ternary)
    const Tensor& last_W_q() const { return last_W_q_; }
    const Tensor& last_W_hat() const { return last_W_hat_; }
    const Tensor& last_scale_w() const { return last_scale_w_; }
    const Tensor& last_input() const { return last_input_; }
    const Tensor& last_Y_pre() const { return last_Y_pre_; }
    const Tensor& last_ste_mask() const { return last_ste_mask_; }

private:
    size_t d_in_;
    size_t d_out_;

    // fp32 master weights (will be quantized in forward).
    Tensor W_;
    // Per-output-unit absmean scale (1, d_out) — paper §3.2 says it is
    // jointly learned with W; gradient has a closed form.
    Tensor scale_w_;

    // Gradients
    Tensor grad_W_;        // (d_out, d_in)
    Tensor grad_scale_w_;  // (1, d_out)

    // Forward cache (read by backward)
    Tensor last_input_;    // (N, d_in)
    Tensor last_scale_w_;  // (1, d_out) — copy of scale_w_ at fwd time
    Tensor last_W_hat_;    // (d_out, d_in) — clamped W / scale_w
    Tensor last_W_q_;      // (d_out, d_in) — ∈ {-1, 0, +1}
    Tensor last_ste_mask_; // (d_out, d_in) — 1 iff |W/scale_w| ≤ 1 else 0
    Tensor last_Y_pre_;    // (N, d_out) — X · W_q^T (pre-scale broadcast)
};

// ============================================================================
// BitNetBlock — two BitLinear with SiLU between them (FFN-style).
//
//   y = BitLinear2(SiLU(BitLinear1(x)))
//
// No residual in v1 (the paper's BitNet has its own residual structure
// at the transformer-block level — out of scope for a primitive layer).
// ============================================================================

class BitNetBlock : public Layer {
public:
    // d_model: input/output feature dim (block is "same-dim" in v1).
    // ffn_mult: hidden dim = ffn_mult * d_model.
    BitNetBlock(size_t d_model, size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return lin1_->get_weights(); }
    Tensor get_gradients() const override { return lin1_->get_gradients(); }
    std::string name() const override { return "BitNetBlock"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    const BitLinear& lin1() const { return *lin1_; }
    const BitLinear& lin2() const { return *lin2_; }

private:
    size_t d_model_;
    size_t ffn_dim_;

    std::unique_ptr<BitLinear> lin1_;
    std::unique_ptr<BitLinear> lin2_;

    // Cache for the SiLU input (needed for SiLU backward).
    Tensor last_silu_in_;
};

#endif