#ifndef RETNET_H
#define RETNET_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RetNet Retention — Sun et al. 2023
//   "Retentive Network: A Successor to Transformer for Large Language Models"
//   (arXiv:2307.08621)
//
// This file implements a single RetNet *retention layer* in the
// recurrence (single-step) form, which is the inference-time form and is
// also the form amenable to BPTT + numerical gradient checks.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (per layer, single-step recurrent form):
//
//   Inputs:  x_t in R^{d_model},  for t = 0..T-1
//
//   Step 1 — Q/K/V projections (Dense: y = x W^T + b, with W of shape (d, d)):
//     q_pre_t = W_Q · x_t + b_Q   in R^d
//     k_pre_t = W_K · x_t + b_K   in R^d
//     v_pre_t = W_V · x_t + b_V   in R^d
//
//   Step 2 — Per-channel decay γ (vector of d_model scalars, in (0, 1)).
//     Reparameterized as γ = sigmoid(gamma_raw) so it stays in (0, 1)
//     without manual clipping. Init at gamma_raw = 0 → γ = 0.5.
//
//   Step 3 — xPos rotation (Sun et al. 2022, RoPE-style pair rotation).
//     Q is rotated by +m·θ (forward) and K is rotated by -m·θ (inverse)
//     using the standard block rotation with θ_i = base^(-2i/(d/2)) for
//     i in [0, d/2). The forward/backward inverse symmetry is the
//     "relative-position" trick that lets the dot product Q·K encode
//     relative offsets instead of absolute ones.
//
//   Step 4 — Reshape into H heads of head_dim = d / H, then per head h:
//
//     S_t^(h) = diag(γ_h) · S_{t-1}^(h) + outer(k_t^(h), v_t^(h))
//                                             ∈ R^{head_dim × head_dim}
//     o_t^(h) = S_t^(h) · q_t^(h)              ∈ R^{head_dim}
//
//   Step 5 — Concat heads, output projection W_O ∈ R^{d×d}, b_O:
//     y_t = W_O · concat(o_t^(1), ..., o_t^(H)) + b_O   ∈ R^d
//
//   State: per head S^(h) ∈ R^{head_dim × head_dim}, total H·d_head·d_head
//          numbers. This is the O(d²/N_h) state that gives RetNet its
//          O(d²)-per-step inference cost (between RWKV's O(d) and full
//          softmax attention's O(T·d)).
//
// ----------------------------------------------------------------------------
// Backward (BPTT) — for each head, per step t (traversed t = T-1, ..., 0):
//
//   dq_t = S_t^T · d_o_t
//   dS_t += d_o_t ⊗ q_t
//   dk_t = dS_t · v_t             (from outer product, right factor)
//   dv_t = S_t^T_{prev-only}?? NO. dL/dv_h comes from the outer product:
//                                  v is the right factor of outer(k,v)=k v^T
//                                  so dL/dv_h = S_{t-1,h}^T · dS_t + γ*γ... ?
//   Wait — let me re-derive: outer(k,v) = k ⊗ v has entries
//     (k ⊗ v)[i,j] = k[i] * v[j]
//     dL/d(k ⊗ v)[i,j] = dS[i,j] (where dS is the dL/d(state) for this step).
//   So dL/dk[i] = sum_j dS[i,j] * v[j] = (dS · v)[i]
//      dL/dv[j] = sum_i dS[i,j] * k[i] = (k^T · dS)[j] = (dS^T · k)[j]
//
//   dS_{t-1} = diag(γ) · dS_t    (decay flows back to previous state)
//   dγ_h[i]  = sum_j dS_t[i,j] * S_{t-1}[i,j]    (decay multiplies the
//                                                  previous state row-wise)
//
//   Then dq_h, dk_h, dv_h are mapped back through xPos rotation inverse
//   (since Q was rotated by +m·θ and K by -m·θ, the inverse is a rotation
//   by -m·θ for Q and +m·θ for K — both are the same kind of operation
//   as the forward RoPE pair rotation), and dγ_h accumulated.
//
//   Then dq/dk/dv_pre are mapped through Dense.backward to give
//   dL/dW_Q, dL/db_Q, dL/dx_t (and dL/dW_K, dL/dW_V).
// ============================================================================

class RetNetRetention : public Layer {
public:
    // d_model: model dim (must be > 0 and even)
    // num_heads: number of heads (must divide d_model evenly)
    explicit RetNetRetention(size_t d_model, size_t num_heads = 1);

    // Forward pass on a full sequence.
    // input: (T, d_model)  ->  output: (T, d_model)
    Tensor forward(const Tensor& input) override;
    // Backward pass — grad_output: (T, d_model), returns grad_input: (T, d_model)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_Q.weights; }
    Tensor get_gradients() const override { return W_Q.grad_weights; }
    std::string name() const override { return "RetNetRetention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

    // ---- Public parameters (for gradient tests / debugging) ----
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double rope_base_;   // xPos base, default 100.0 (small for tractable grad check)

    Dense W_Q;              // (d, d)
    Dense W_K;              // (d, d)
    Dense W_V;              // (d, d)
    Dense W_O;              // (d, d)  output projection
    Tensor gamma_raw;       // (1, d)  unconstrained; γ = sigmoid(gamma_raw) ∈ (0,1)

    // Hidden gradient buffers
    Tensor grad_gamma_raw_;  // (1, d)

private:

    // BPTT cache (filled in forward, used in backward)
    Tensor last_input_;         // (T, d)  — cloned to avoid input corruption
    Tensor last_q_pre_;         // (T, d)
    Tensor last_k_pre_;         // (T, d)
    Tensor last_v_pre_;         // (T, d)
    // Per-head rotated q/k/v (stored flattened as (T, d) — head h = cols [h*D .. (h+1)*D))
    Tensor last_q_rot_;         // (T, d)  — q after xPos rotation (NO γ-scale on Q; Q is unscaled at retrieval)
    Tensor last_k_rot_;         // (T, d)  — k after xPos rotation (NO γ-scale on K; γ applies on S, not K)
    Tensor last_v_;             // (T, d)  — v (no rotation, no decay — v goes in raw)
    Tensor last_gamma_;         // (1, d)  — sigmoid(gamma_raw) — per-channel decay
    // Per-head state S_t ∈ R^{head_dim × head_dim} stored as (T+1, H*D*D) flat.
    // Row t contains [vec(S_t^(1)), vec(S_t^(2)), ..., vec(S_t^(H))] concatenated
    // where each vec(S_t^(h)) is row-major over the D×D matrix. Index helper:
    //   last_states_(t, h * D * D + i * D + j)  =  S_t^(h)[i, j]
    Tensor last_states_;        // (T+1, num_heads_ * head_dim_ * head_dim_)
    Tensor last_o_;             // (T, d)  output after concat of o_t^h, before W_O
};

#endif // RETNET_H
