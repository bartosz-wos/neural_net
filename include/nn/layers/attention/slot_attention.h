#ifndef SLOT_ATTENTION_H
#define SLOT_ATTENTION_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>

// ============================================================================
// Slot Attention — Locatello et al. 2020
//   "Object-Centric Learning with Slot Attention"
//   (https://arxiv.org/abs/2006.15055)
//
// Innovation: a permutation-invariant, iterative attention mechanism for
// decomposing a scene into a set of "slots" that each capture a distinct
// object or region. The key trick is a *double softmax* over both the slot
// axis (K slots) and the input axis (N inputs), which produces slot
// competition: each input is softly assigned to exactly one slot, while
// each slot has to compete for its share of inputs.
//
// Math (one Slot Attention block, T iterations):
//
//   slots_0 = mu                                              (K, D)  learned init
//   for t = 1..T:
//     k   = LayerNorm_k(x) @ W_k + b_k                        (N, D)
//     v   = LayerNorm_v(x) @ W_v + b_v                        (N, D)
//     q   = LayerNorm_q(slots) @ W_q + b_q                    (K, D)
//     logits = (q @ k^T) * scale          with scale = D^(-1/2)  (K, N)
//     attn   = softmax(logits, axis=slots)                   column softmax (each input distributes over slots — slot competition)
//     attn   = softmax(attn, axis=inputs)                    row softmax (each slot distributes over inputs)
//     updates = attn @ v                                     (K, D)
//     slots   = GRU_cell(updates, slots)                     per-slot stateless update; input=updates, hidden=slots
//     slots   = slots + MLP(LayerNorm(slots))                residual MLP refine
//
// Conventions:
//   * Input X: (N, D). Output slots: (K, D).
//   * Default T = 3 iterations.
//   * Slot init: learned parameter mu in R^{K, D} initialized small (0.1 scale).
//   * W_k, W_v, W_q are Dense-style: y = x @ W^T + b, shapes (D, D).
//   * MLP: 2-layer Dense with ReLU on hidden, both (D, D).
//   * GRU cell: stateless per-slot update, weights SHARED across slots.
//
// BPTT specifics:
//   * The double softmax splits the gradient chain through TWO softmaxes.
//     d_logits comes from BOTH paths (col-softmax backward to attn1, then
//     row-softmax backward from there). We accumulate them additively into
//     d_logits before projecting through W_k, W_v, W_q.
//   * Iterated unrolling: d_slot at iteration t flows forward from d_slot
//     at iteration t+1 PLUS the residual-MLP contribution. The chain
//     iterates T times in backward.
//   * Slot init mu participates in the BPTT chain only through slot_T.
//   * Slot init's gradient is d_mu = d_slot at t=1 BEFORE the LayerNorm_q
//     and W_q projection, after the GRU update.
//
// Public API:
//   * SlotAttention(num_slots, slot_dim, input_dim, num_iterations=3, hidden_dim=0, epsilon=1e-8)
//       forward(input: (N, input_dim)) -> slots: (K, slot_dim)
//       backward(grad_output: (K, slot_dim), lr) -> grad_input: (N, input_dim)
//       parameters() / gradients() / zero_grad() — standard Layer interface
//
//   * SlotAttentionBlock(num_slots, slot_dim, input_dim, num_iterations=3, hidden_dim=0)
//       pre-LN -> SlotAttention -> residual -> pre-LN -> 2-layer FFN -> residual
//       slots dimension is slot_dim (= input_dim by default; pre-LN before the
//       block ensures the input projection to slot_dim if they differ)
//
//   * SlotAttentionModel(num_slots, slot_dim, input_dim, out_dim, n_blocks=1, num_iterations=3, hidden_dim=0)
//       input projection -> stack of blocks -> per-slot classifier (linear to out_dim)
//       forward: (N, input_dim) -> (K, out_dim); each slot is classified separately.
// ============================================================================

class SlotAttention : public Layer {
public:
    SlotAttention(size_t num_slots, size_t slot_dim, size_t input_dim,
                  size_t num_iterations = 3, size_t hidden_dim = 0,
                  double epsilon = 1e-8);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return mu_; }
    Tensor get_gradients() const override { return grad_mu_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    size_t get_num_slots() const { return num_slots_; }

private:
    size_t num_slots_;
    size_t slot_dim_;
    size_t input_dim_;
    size_t num_iterations_;
    size_t hidden_dim_;
    double epsilon_;

    // Q/K/V projections: Dense-style y = x @ W^T + b.
    // For input projection, W_k and W_v are (slot_dim, input_dim) so that
    // k = LN(x) @ W_k^T has shape (N, slot_dim) — x is (N, input_dim).
    // Wait: we use Dense(in_features, out_features) so weights have shape
    // (out_features, in_features). Forward computes y = x @ W^T + b, so to
    // produce k = (N, slot_dim) from x_ln = (N, input_dim), we need
    // weights = (slot_dim, input_dim), i.e., Dense(input_dim, slot_dim).
    Tensor W_k_, W_v_, W_q_;     // W_k, W_v: Dense(input_dim, slot_dim); W_q: Dense(slot_dim, slot_dim)
    Tensor b_k_, b_v_, b_q_;     // (1, slot_dim)
    Tensor grad_W_k_, grad_W_v_, grad_W_q_;
    Tensor grad_b_k_, grad_b_v_, grad_b_q_;

    // Slot init mu: (K, D)
    Tensor mu_;
    Tensor grad_mu_;

    // LayerNorms for input (k, v) and slot (q) — each is a LayerNorm(D, eps).
    // LayerNorm normalizes per-row across features; with 1D (1, D) gamma/beta
    // it can be applied to both (N, D) and (K, D) inputs equally.
    LayerNorm ln_k_, ln_v_, ln_q_;
    LayerNorm ln_mlp_;  // before the residual MLP

    // Residual MLP — 2 Dense layers (slot_dim, slot_dim) with ReLU between.
    Dense mlp_fc1_;     // (slot_dim, slot_dim)
    Dense mlp_fc2_;     // (slot_dim, slot_dim)

    // GRU-like per-slot recurrent update — weights shared across slots,
    // stateless per call. Math:
    //   z = sigma(W_z @ [u; s] + b_z)
    //   r = sigma(W_r @ [u; s] + b_r)
    //   s_hat = tanh(W_h @ [u; r*s] + b_h)
    //   s' = (1-z)*s + z*s_hat
    // where u is the GRU input (updates[k, :]) and s is the slot (slots[k, :])
    // before the update. All weights are (slot_dim, 2*slot_dim) for [u;s]-based
    // gates, (slot_dim, slot_dim) for candidate. We store combined layouts:
    //   W_zr: (2*slot_dim, 2*slot_dim) acting on [u; s]                → (z, r) gates
    //   W_h:  (2*slot_dim, slot_dim) acting on [u; rh]                → candidate
    Tensor W_zr_, W_h_;
    Tensor b_zr_, b_h_;
    Tensor grad_W_zr_, grad_W_h_;
    Tensor grad_b_zr_, grad_b_h_;

    // ----- Caches for BPTT -----
    // For each iteration t (0..T-1), we cache everything needed to compute
    // gradients locally and to chain to t-1.
    struct IterCache {
        Tensor slots_pre_gru;     // (K, D) — slots BEFORE the GRU update at iter t
        Tensor slots_post_gru;    // (K, D) — slots AFTER the GRU update (before residual MLP)
        Tensor slots_post_mlp;    // (K, D) — slots after the residual MLP add (= input to next iter)

        Tensor k_proj;            // (N, D)
        Tensor v_proj;            // (N, D)
        Tensor q_proj;            // (K, D)
        Tensor x_ln;              // (N, D) — LayerNorm_k(x)
        Tensor v_ln;              // (N, D) — LayerNorm_v(x)
        Tensor slots_ln_q;        // (K, D) — LayerNorm_q(slots_pre_gru)
        Tensor slots_ln_mlp;      // (K, D) — LayerNorm(slots_post_gru) (input to residual MLP)
        Tensor mlp_h;             // (K, D) — first Dense + ReLU output (= input to fc2)
        Tensor logits;            // (K, N)
        Tensor attn1;             // (K, N) — softmax over slots (cols sum to 1)
        Tensor attn2;             // (K, N) — softmax over inputs (rows sum to 1)
        Tensor updates;           // (K, D) — attn2 @ v_proj

        // GRU gate states (per-slot): for BPTT we cache per-slot gates since
        // the gate math is independent across slots.
        Tensor z_gates;           // (K, D)
        Tensor r_gates;           // (K, D)
        Tensor s_hat;             // (K, D) — candidate
        Tensor rh;                // (K, D) — r * s (input to W_h path)
    };
    std::vector<IterCache> cache_;

    // Cached input (for input-grad)
    Tensor last_input_;

    // Helpers
    // Stateless GRU update: given (u: (K, D), s: (K, D)) produce new_s: (K, D).
    // Fills gates into z_out, r_out, s_hat_out, rh_out (all (K, D)).
    // Updates internal grad tensors in-place given grad_new_s (K, D).
    // Returns grad_u (K, D) and grad_s (K, D) w.r.t. its inputs.
    struct GruOut {
        Tensor new_s;
        Tensor z, r, s_hat, rh;
    };
    GruOut gru_forward(const Tensor& u, const Tensor& s);
    void gru_backward(const Tensor& grad_new_s,
                      const Tensor& u, const Tensor& s,
                      const GruOut& st,
                      Tensor& grad_u, Tensor& grad_s);

    // Row softmax (over columns axis of a (R, C) tensor)
    static Tensor row_softmax(const Tensor& x);
    // Column softmax (over rows axis of a (R, C) tensor)
    static Tensor col_softmax(const Tensor& x);

    // Helper: apply LayerNorm manually (since we cache a different kind of
    // normalization than the LayerNorm Layer provides). We use the LayerNorm
    // *class* for its learnable gamma/beta — but apply it row-wise (per-row
    // normalize across features), which is exactly what it does for a 2D
    // input. So we just call ln.forward(x) here. (No custom impl needed.)
};


// ============================================================================
// SlotAttentionBlock — per-slot FFN residual block
//
// Operates on slots (K, D) → slots (K, D). This is the "refinement block"
// that stacks after a SlotAttention to apply a per-slot feed-forward
// transformation with a residual connection. Architecture:
//
//   slots_res1 = LN(slots)
//   slots_ffn  = fc2( ReLU( fc1(slots_res1) ) )  with fc1: (hidden, D), fc2: (D, hidden)
//   out = slots + slots_ffn
//
// Where hidden defaults to 4*D if hidden_dim == 0. Per-slot weights are
// shared across slots (Dense convention applied batch-wise to the (K, D) tensor).
// ============================================================================
class SlotAttentionBlock : public Layer {
public:
    SlotAttentionBlock(size_t num_slots, size_t slot_dim, size_t input_dim,
                       size_t num_iterations = 3, size_t hidden_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    // The num_slots/num_iterations/input_dim args are accepted for API symmetry
    // with the rest of the model — input_dim is unused here (the block operates
    // purely on slot_dim), and num_slots/num_iterations are likewise unused
    // (we apply a per-slot Dense FFN regardless). We keep the signature for
    // uniform construction in SlotAttentionModel.
    size_t slot_dim_;
    size_t hidden_dim_;

    LayerNorm ln_;                    // (slot_dim)
    Dense ffn_fc1_;                   // (hidden_dim, slot_dim)
    Dense ffn_fc2_;                   // (slot_dim, hidden_dim)

    Tensor last_input_;
    Tensor last_ln_out_;              // LN output (pre-ReLU input)
    Tensor last_ffn_h_;               // ReLU output (post-activation)
};


// ============================================================================
// SlotAttentionModel — input projection -> stack of blocks -> per-slot classifier
// Input X: (N, input_dim). Output: (K, out_dim). Each slot classified independently.
// ============================================================================
class SlotAttentionModel : public Layer {
public:
    SlotAttentionModel(size_t num_slots, size_t slot_dim, size_t input_dim,
                       size_t out_dim, size_t n_blocks = 1,
                       size_t num_iterations = 3, size_t hidden_dim = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    Dense input_proj_;     // (slot_dim, input_dim)
    std::unique_ptr<SlotAttention> attn_;  // (N, D) → (K, D) — input → slots
    std::vector<std::unique_ptr<SlotAttentionBlock>> blocks_;  // per-slot FFN refinement, (K, D) → (K, D)
    Dense classifier_;     // (out_dim, slot_dim)
    Tensor last_input_;
    Tensor last_slot_in_;  // (K, slot_dim) — output of input_proj (first block's input)
    Tensor last_attn_out_; // (K, slot_dim) — output of attn_ (first block's input)
    Tensor last_block_out_;  // (K, slot_dim) — output of the LAST block, cached for classifier backward
    std::vector<Tensor> block_inputs_;  // cached input to each block
};

#endif // SLOT_ATTENTION_H
