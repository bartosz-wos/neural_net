#ifndef POWER_ATTENTION_H
#define POWER_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// Power Attention — Poli et al., 2024
//   "Power Attention: Transformers Need Less Compute Than You Think"
//   https://arxiv.org/abs/2403.14248
// ============================================================================
//
// Standard softmax attention computes  A[i,j] = softmax(z[i,j])_j  where
// z[i,j] = (q_j · k_i) / sqrt(d_h). The "power" generalisation raises the
// logits to a learnable per-head power `p` BEFORE the softmax:
//
//     s[i,j] = p · z[i,j]
//     A[i,j] = exp(s[i,j] - max_i s[i,j]) / Σ_i exp(s[i',j] - max_i s[i',j])
//
// This is mathematically equivalent to a standard softmax with **learnable
// per-head temperature T = 1/p** — but parameterised in `p_log` space so the
// optimiser can move freely in either direction:
//
//     p[h] = softplus(p_log[h]) + p_eps            p > p_eps > 0
//
// The temperature interpretation gives a clean intuition:
//     p → ∞   (T → 0)         A → argmax           sharp, sparse
//     p = 1   (T = 1)         standard softmax
//     p → 0   (T → ∞)         A → uniform           smooth, diffuse
//
// The paper's empirical finding is that trained models learn *heterogeneous*
// per-head `p` values — some heads sharp, some diffuse — and that the
// distribution is itself informative for interpretability. Implementing it
// per-head (rather than per-layer) is the whole point.
//
// ----------------------------------------------------------------------------
// Backward derivation (hand-derived; the formulae in plain form)
//
// Let
//   s_k = p · z_k − p · z_max       (the shifted, scaled logits; z_max is per-query)
//   A_k = exp(s_k) / Σ_i exp(s_i)
//   dA_k given, sumA = Σ_k dA_k
//
// Standard softmax-backward identity:
//   dL/ds_k = A_k · (dA_k − sumA)        (so the gradient is positive when dA > 0
//                                         in a region where A is large)
//
// Then (and this is the *new* part vs. standard softmax):
//   dL/dz_k = p · dL/ds_k
//   dL/dp   = Σ_k z_k · dL/ds_k
//   dL/dp_log = dL/dp · softplus'(p_log)
//
// Q / K chain (same form as standard attention):
//   dL/dq_j += (p / sqrt(d_h)) · Σ_i dL/ds_i,j · k_i
//   dL/dk_i += (p / sqrt(d_h)) · Σ_j dL/ds_i,j · q_j
//
// In the gradient-check tests below, the chain through `p` is verified
// independently by perturbing `p_log` and checking d/d_p_log L.
//
// ----------------------------------------------------------------------------
// Conventions (match attention/diff_transformer.h and stick_breaking.h):
//   * Input / output: (N, d_model). N tokens, no explicit batch dim.
//   * W_q, W_k, W_v, W_o are `Dense` (weights shaped (out, in)); heads are
//     contiguous column slices of the flat (N, d_model) projections.
//   * `p_log` is a learnable per-head vector of length num_heads, init at 0
//     so initial p = softplus(0) + 0.1 ≈ 1.243 — within a factor of 1 of
//     standard softmax (T = 1) so the layer starts as a soft baseline.
//   * Block: pre-LN -> Power attn -> residual -> pre-LN -> GELU FFN -> residual
// ----------------------------------------------------------------------------

class PowerAttention : public Layer {
public:
    // d_model:   input/output feature dim; must be divisible by num_heads
    // num_heads: number of attention heads (default 1)
    // p_eps:     numerical floor on `p` so the optimiser can't push p -> 0
    PowerAttention(size_t d_model, size_t num_heads = 1, double p_eps = 0.1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "PowerAttention"; }

    // Accessors
    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    double inv_temp()  const { return inv_temp_; }
    double p_eps()     const { return p_eps_; }

    // Effective per-head power: p[h] = softplus(p_log_[h]) + p_eps_
    Tensor effective_p() const;

    // Attention map from the last forward, shape (num_heads * N, N):
    // row (h*N + j), col i  ==  A[i,j] for head h
    const Tensor& last_A() const { return last_A_; }
    // Per-(head, query) p * z_max, shape (num_heads, N), for diagnostics
    const Tensor& last_pz_max() const { return last_pz_max_; }
    const Tensor& last_input() const { return last_input_; }

    // Parameters (public so tests can perturb them directly for FD checks)
    Dense W_q, W_k, W_v, W_o;          // each (d_model, d_model)
    Tensor p_log_;                     // (num_heads,) learnable log-power
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_p_log_;                // (num_heads,)

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double inv_temp_;
    double p_eps_;

    // Forward caches
    Tensor last_input_;     // (N, d_model)
    Tensor last_Q_;         // (N, d_model)
    Tensor last_K_;         // (N, d_model)
    Tensor last_V_;         // (N, d_model)
    Tensor last_A_;         // (num_heads * N, N) — attention weights
    Tensor last_pz_max_;    // (num_heads, N) — per-(head, query) max(p·z_i)
    Tensor last_O_;         // (N, d_model) — pre-W_o output
    Tensor last_p_;         // (num_heads,) — effective per-head power at this forward
    size_t N_last_ = 0;
};

// ----------------------------------------------------------------------------
// PowerBlock — pre-LN -> Power attn -> residual -> pre-LN -> FFN -> residual
// ----------------------------------------------------------------------------
class PowerBlock : public Layer {
public:
    PowerBlock(size_t d_model, size_t num_heads = 1,
               size_t ffn_dim = 0, double p_eps = 0.1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return attn.W_q.weights; }
    Tensor get_gradients() const override { return attn.grad_W_q; }
    std::string name() const override { return "PowerBlock"; }

    PowerAttention attn;
    LayerNorm ln1, ln2;
    Dense ffn_fc1_;        // (ffn_dim, d_model)
    Dense ffn_fc2_;        // (d_model, ffn_dim)

private:
    size_t d_model_;
    size_t ffn_dim_;
    Tensor last_x_;
    Tensor last_z1_;        // ln1(x)
    Tensor last_attn_out_;
    Tensor last_res1_;      // z1 + attn_out
    Tensor last_z2_;        // ln2(res1)
    Tensor last_h_pre_;     // ffn_fc1(z2)
    Tensor last_h_act_;     // GELU(h_pre)
};

#endif // POWER_ATTENTION_H
