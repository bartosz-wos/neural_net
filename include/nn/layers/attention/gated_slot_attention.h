#ifndef GATED_SLOT_ATTENTION_H
#define GATED_SLOT_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>

// ============================================================================
// Gated Slot Attention — Bauer et al., NeurIPS 2024
//   "Gated Slot Attention for Object-Centric Learning"
//   (https://arxiv.org/abs/2410.23790)
//
// An extension of Slot Attention (Locatello 2020) with a learned per-input
// gate that suppresses spurious cross-slot attention on out-of-distribution
// inputs. The gate `g = σ(W_g · x + b_g)` is element-wise multiplied with the
// LayerNorm'd input BEFORE the K/V projections — so the gate controls which
// input features are allowed to participate in the slot competition.
//
// Math (single block, T iterations):
//
//   g        = σ(W_g · x + b_g)              ∈ R^{N, input_dim}     [paper §3.2 "input gate"]
//   k        = (g ⊙ LN_k(x)) @ W_k^T + b_k   ∈ R^{N, slot_dim}
//   v        = (g ⊙ LN_v(x)) @ W_v^T + b_v   ∈ R^{N, slot_dim}
//   slots_0  = mu                            ∈ R^{K, slot_dim}      [learned init]
//   for t = 1..T:
//     q       = LN_q(slots) @ W_q^T + b_q     ∈ R^{K, slot_dim}
//     logits  = (q @ k^T) · dn^(-1/2)         ∈ R^{K, N}
//     attn    = softmax(logits, axis=slots)   column-softmax  (slot competition)
//     attn    = softmax(attn, axis=inputs)    row-softmax     (input distribution)
//     updates = attn @ v                      ∈ R^{K, slot_dim}
//     slots   = GRU(updates, slots)           per-slot stateless update
//     slots   = slots + MLP(LN(slots))        residual MLP refinement
//
// Backward additions over plain SlotAttention:
//   d_x_gated_k = d_k_proj @ W_k     (N, input_dim)
//   d_x_gated_v = d_v_proj @ W_v     (N, input_dim)
//   d_gate      = d_x_gated_k ⊙ LN_k(x) + d_x_gated_v ⊙ LN_v(x)
//   d_pre_sig   = d_gate ⊙ gate ⊙ (1 − gate)
//   dW_g       += x^T @ d_pre_sig     (input_dim, input_dim)
//   db_g       += Σ d_pre_sig
//   d_x       += d_pre_sig @ W_g^T
//   d_LN_k(x) += d_x_gated_k ⊙ g
//   d_LN_v(x) += d_x_gated_v ⊙ g
//
// Conventions:
//   Input X: (N, input_dim). Output slots: (K, slot_dim).
//   Default T = 3 iterations.
//   W_g has shape (input_dim, input_dim) — square projection. b_g has shape (1, input_dim).
//   All other projections / LayerNorm / MLP / GRU follow SlotAttention's
//   conventions bit-for-bit, so that with the gate forced to 0 or 1, the layer
//   collapses to vanilla SlotAttention.
//
// Public API:
//   * GatedSlotAttention(num_slots, slot_dim, input_dim, num_iterations=3, hidden_dim=0, epsilon=1e-8)
//       forward(input: (N, input_dim)) -> slots: (K, slot_dim)
//       backward(grad_output: (K, slot_dim), lr) -> grad_input: (N, input_dim)
//       parameters() / gradients() / zero_grad() — standard Layer interface
//       last_gate() returns the cached (N, input_dim) gate for inspection/tests
//
//   * GatedSlotAttentionModel(input_dim, slot_dim, num_slots, hidden_dim, output_dim, n_blocks=1, num_iterations=3)
//       input projection -> n_blocks of [GatedSlotAttention + FFN refinement]
//       -> per-slot classifier. forward: (N, input_dim) -> (K, output_dim).
// ============================================================================

class GatedSlotAttention : public Layer {
public:
    GatedSlotAttention(size_t num_slots, size_t slot_dim, size_t input_dim,
                       size_t num_iterations = 3, size_t hidden_dim = 0,
                       double epsilon = 1e-8);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_g_; }
    Tensor get_gradients() const override { return grad_W_g_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    size_t get_num_slots() const { return num_slots_; }
    size_t get_slot_dim() const { return slot_dim_; }
    size_t get_input_dim() const { return input_dim_; }
    const Tensor& last_gate() const { return last_gate_; }

private:
    size_t num_slots_;
    size_t slot_dim_;
    size_t input_dim_;
    size_t num_iterations_;
    size_t hidden_dim_;
    double epsilon_;

    // --- Gate projection (paper §3.2) ---
    // W_g: (input_dim, input_dim), b_g: (1, input_dim)
    Tensor W_g_, b_g_;
    Tensor grad_W_g_, grad_b_g_;

    // --- Q/K/V projections (Dense-style: weights (out, in) for x @ W^T + b) ---
    Tensor W_k_, W_v_, W_q_;
    Tensor b_k_, b_v_, b_q_;
    Tensor grad_W_k_, grad_W_v_, grad_W_q_;
    Tensor grad_b_k_, grad_b_v_, grad_b_q_;

    // --- Slot init mu: (K, slot_dim) ---
    Tensor mu_;
    Tensor grad_mu_;

    // --- LayerNorms ---
    LayerNorm ln_k_, ln_v_, ln_q_, ln_mlp_;

    // --- Residual MLP: Dense(D, H) -> ReLU -> Dense(H, D) ---
    Dense mlp_fc1_;
    Dense mlp_fc2_;

    // --- Per-slot GRU (shared weights across slots) ---
    // W_zr: (2D, 2D), W_h: (2D, D); b_zr: (1, 2D), b_h: (1, D)
    Tensor W_zr_, W_h_;
    Tensor b_zr_, b_h_;
    Tensor grad_W_zr_, grad_W_h_;
    Tensor grad_b_zr_, grad_b_h_;

    // --- Caches for BPTT ---
    struct IterCache {
        Tensor slots_pre_gru;       // (K, D) — slots BEFORE GRU at iter t
        Tensor slots_post_gru;      // (K, D) — slots AFTER  GRU at iter t (before residual MLP)
        Tensor slots_post_mlp;      // (K, D) — slots after residual MLP (= input to iter t+1)

        Tensor k_proj;              // (N, D)
        Tensor v_proj;              // (N, D)
        Tensor q_proj;              // (K, D)
        Tensor x_ln_k;              // (N, input_dim) — LayerNorm_k(x)
        Tensor x_ln_v;              // (N, input_dim) — LayerNorm_v(x)
        Tensor x_gated_k;           // (N, input_dim) — gate ⊙ x_ln_k (cached for backward)
        Tensor x_gated_v;           // (N, input_dim) — gate ⊙ x_ln_v
        Tensor gate;                // (N, input_dim) — cached gate at this iter

        Tensor slots_ln_q;          // (K, D) — LayerNorm_q(slots_pre_gru)
        Tensor slots_ln_mlp;        // (K, D) — LayerNorm(slots_post_gru) (input to residual MLP)
        Tensor mlp_h;               // (K, D) — first Dense + ReLU output
        Tensor logits;              // (K, N)
        Tensor attn1;               // (K, N) — col softmax
        Tensor attn2;               // (K, N) — row softmax
        Tensor updates;             // (K, D) — attn2 @ v_proj

        // GRU gate states (per-slot)
        Tensor z_gates;             // (K, D)
        Tensor r_gates;             // (K, D)
        Tensor s_hat;               // (K, D)
        Tensor rh;                  // (K, D) — r ⊙ s
    };
    std::vector<IterCache> cache_;

    // Cached input for input-grad
    Tensor last_input_;
    // Cached gate (computed once, applied at every iter)
    Tensor last_gate_;

    // ----- Helpers -----
    // Stateless per-slot GRU update.
    struct GruOut {
        Tensor new_s, z, r, s_hat, rh;
    };
    GruOut gru_forward(const Tensor& u, const Tensor& s);
    void gru_backward(const Tensor& grad_new_s,
                      const Tensor& u, const Tensor& s,
                      const GruOut& st,
                      Tensor& grad_u, Tensor& grad_s);

    static Tensor row_softmax(const Tensor& x);
    static Tensor col_softmax(const Tensor& x);
};


// ============================================================================
// GatedSlotAttentionModel
//   input projection -> [GatedSlotAttention + FFN block]×n_blocks -> per-slot classifier
//   forward: (N, input_dim) -> (K, output_dim)
// ============================================================================
class GatedSlotAttentionModel : public Layer {
public:
    GatedSlotAttentionModel(size_t input_dim, size_t slot_dim, size_t num_slots,
                            size_t hidden_dim, size_t output_dim,
                            size_t n_blocks = 1, size_t num_iterations = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return input_proj_.get_weights(); }
    Tensor get_gradients() const override { return input_proj_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    // Helper block: pre-LN -> per-slot FFN -> residual, on slots (K, slot_dim).
    class GsaBlock {
    public:
        GsaBlock(size_t slot_dim, size_t hidden_dim);
        Tensor forward(const Tensor& slots);
        Tensor backward(const Tensor& grad_output);
        void zero_grad();
        void update_weights(double lr);
        std::vector<Tensor*> parameters();
        std::vector<Tensor*> gradients();
    private:
        size_t hidden_dim_;
        LayerNorm ln_;
        Dense fc1_, fc2_;
        Tensor last_input_, last_ln_out_, last_ffn_h_;
    };

    Dense input_proj_;
    std::unique_ptr<GatedSlotAttention> attn_;
    std::vector<std::unique_ptr<GsaBlock>> blocks_;
    Dense classifier_;
    Tensor last_input_;
    Tensor last_slot_in_;
    Tensor last_attn_out_;
    Tensor last_block_out_;
    std::vector<Tensor> block_inputs_;
};

#endif // GATED_SLOT_ATTENTION_H
