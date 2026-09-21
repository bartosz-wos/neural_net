#ifndef MAMBA_BIDIRECTIONAL_H
#define MAMBA_BIDIRECTIONAL_H

#include "../../core/layer.h"
#include "mamba.h"
#include <vector>
#include <cmath>

// ============================================================================
// Bidirectional Mamba (BiMamba)
// ============================================================================
//
// A bidirectional extension of the canonical Mamba (S6) selective state-space
// model. Combines:
//   1. A FORWARD Mamba scan on the input  x[0..T-1].
//   2. A BACKWARD Mamba scan on the time-REVERSED input  x_rev[t] = x[T-1-t].
//   3. A learnable per-token, per-channel 2-way softmax mix that combines the
//      forward and backward gated outputs into the final layer output.
//
// Per-token, per-channel mix (the Hymba-style mixing pattern):
//   mix_logits[t, j]  = mix_proj(concat(forward_gated[t, j], backward_gated[t, j]))
//                    ∈ R^{T x 2*d_inner}        (one logit per direction per (token, channel))
//   mix[t, 0, j]      = softmax over 2 of mix_logits[t, :, j]
//   mix[t, 1, j]      = softmax over 2 of mix_logits[t, :, j]
//   gated_out[t, j]   = mix[t, 0, j] · forward_gated[t, j] + mix[t, 1, j] · backward_gated[t, j]
//   output[t, k]      = out_proj(gated_out[t, :])  ∈ R^{T x d_model}
//
// At init, mix_proj.bias is set so the 2-way softmax starts ≈ [0.5, 0.5]
// (uniform forward+backward) — letting the model learn the right balance.
//
// Mathematical justification:
//   * Forward Mamba sees the input in causal order; it can only aggregate past
//     information at position t (h_t depends on x_{0..t}).
//   * Backward Mamba sees the input in reverse causal order; it can only
//     aggregate FUTURE information at position t (h_rev_t depends on x_{t..T-1}).
//   * Mixing the two yields a NON-causal sequence representation — every
//     position has both past and future context. This is the same intuition
//     behind BiLSTM, BiGRU, and Bidirectional SSMs used in audio/vision Mamba
//     variants (e.g., Vision Mamba §2.2, Audio Mamba, BiMamba).
//
// References:
//   - Vision Mamba (Zhu et al. 2024) — https://arxiv.org/abs/2401.09417 §2.2
//     (the paper that popularized the "two Mamba scans + mix" pattern)
//   - Audio Mamba (https://arxiv.org/abs/2405.14736) — applied to audio sequences
//
// ----------------------------------------------------------------------------
// Backward (hand-derived):
//
// Forward path: y_f = forward_mamba.forward(x)                       in R^{T x d_inner}
//                grad_y_f → standard MambaBlock::backward  (already shipped, machine precision)
//
// Reverse input: x_rev[t] = x[T-1-t]                                 in R^{T x d_model}
//   dL/d(x_rev[t]) = grad_x_rev[t]
//   dL/d(x[t])     = grad_x_rev[T-1-t]                               (time-reversal permutation)
//
// Backward path: y_b = backward_mamba.forward(x_rev)
//                grad_y_b → backward_mamba.backward(grad_y_b, lr)    (same machinery, runs on x_rev)
//                grad_x_rev → permuted back to grad_x for the forward chain
//
// Mix chain (per token, per channel, 2-way softmax):
//   mix_logits[t, :, j] = mix_proj(forward_gated[t, j], backward_gated[t, j])
//   mix[t, 0, j] = e^l0 / (e^l0 + e^l1)
//   mix[t, 1, j] = e^l1 / (e^l0 + e^l1)
//   gated_out[t, j] = mix[0] · forward_gated[t, j] + mix[1] · backward_gated[t, j]
//
//   dL/dmix_logits[t, c, j] = grad_gated_out[t, j] · out[c] - grad_gated_out[t, j] · sum_c mix[t, c, j] · out[c]
//                            where  out[0] = forward_gated[t, j],  out[1] = backward_gated[t, j]
//   dL/d(mix_proj input)[t, c] = sum_j dL/dmix_logits[t, c, j] · mix_proj.weights[c, j]
//     → contributes to mix_proj.grad_weights via the standard Dense backward
//   dL/d(forward_gated)[t, j] = mix[t, 0, j] · grad_gated_out[t, j]
//   dL/d(backward_gated)[t, j] = mix[t, 1, j] · grad_gated_out[t, j]
//
// Reduction property: setting `mix[t, 0, j] = 1` everywhere (the "zero-mix"
// regression test) makes the layer reduce to the forward Mamba alone — the
// backward Mamba path contributes zero gradient and zero output. This is
// the canonical regression test that the bidirectional wiring is correct.
//
// ----------------------------------------------------------------------------
// Conventions match the existing MambaBlock:
//   * Input / output: (T, d_model)
//   * d_inner default = 2 * d_model (matches Mamba paper convention)
//   * Two independent MambaBlock instances (no weight sharing — paper says
//     independent scans learn complementary representations).
// ============================================================================

class MambaBidirectional : public Layer {
public:
    // d_model:    input/output feature dim
    // d_state:    SSM state dim for both scans
    // d_inner:    inner feature dim (default = 2 * d_model); same for both scans
    MambaBidirectional(size_t d_model, size_t d_state, size_t d_inner = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return forward_mamba.in_proj.weights; }
    Tensor get_gradients() const override { return forward_mamba.in_proj.grad_weights; }
    std::string name() const override { return "MambaBidirectional"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }
    size_t d_inner() const { return d_inner_; }

    // Mix projection: maps concat(forward_gated, backward_gated) per token to mix logits
    // Per (token, j) we have 2 logits → softmax gives [mix_forward, mix_backward]
    Dense mix_proj_;            // (2*d_inner, 2*d_inner)  WITH bias
    Dense out_proj_;            // (d_inner, d_model)     — output projection applied AFTER the mix
    MambaBlock forward_mamba;   // forward scan (we use its pre-proj `last_gated` as the mix input)
    MambaBlock backward_mamba;  // backward scan on reversed input (same convention)

    // Cached outputs (exposed as accessors for tests/inspectability)
    const Tensor& last_mix()           const { return last_mix_; }
    const Tensor& last_gated_out()     const { return last_gated_out_; }
    const Tensor& last_forward_gated() const { return last_forward_gated_; }
    const Tensor& last_backward_gated()const { return last_backward_gated_; }
    const Tensor& last_mix_logits()    const { return last_mix_logits_; }
    const Tensor& last_x_rev()         const { return last_x_rev_; }

private:
    size_t d_model_;
    size_t d_state_;
    size_t d_inner_;

    // Caches for forward / backward
    Tensor last_input_;         // (T, d_model)
    Tensor last_x_rev_;         // (T, d_model)  time-reversed copy of input
    Tensor last_forward_gated_; // (T, d_inner)  forward_mamba's last_gated_ output (post-silu*o)
    Tensor last_backward_gated_;// (T, d_inner)  backward_mamba's last_gated_ on reversed input
    Tensor last_mix_logits_;    // (T, 2*d_inner)  mix_proj output
    Tensor last_mix_;           // (T, 2*d_inner)  softmax over 2 (per token, per channel)
    Tensor last_gated_out_;     // (T, d_inner)   mix-weighted combination

    // Helpers
    static double softmax2(double l0, double l1, int which) {
        // numerically stable 2-way softmax, returns the which-th entry
        double m = std::max(l0, l1);
        double e0 = std::exp(l0 - m);
        double e1 = std::exp(l1 - m);
        double z  = e0 + e1;
        return (which == 0 ? e0 : e1) / z;
    }
};

#endif // MAMBA_BIDIRECTIONAL_H
