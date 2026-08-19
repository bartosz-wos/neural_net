#ifndef MAMBA_BYTE_H
#define MAMBA_BYTE_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// MambaByte — Wang et al. 2024
//   "MambaByte: Token-free Selective State Space Model"
//   https://arxiv.org/abs/2401.13660
//
// A token-free, byte-level adaptation of Mamba (Gu & Dao 2023) that operates
// directly on bytes rather than tokenizer outputs.
//
// Mathematical formulation (per MambaByteBlock(d_model, d_state, d_inner, vocab_size, twss)):
//
//   Input:        bytes_t ∈ {0, ..., vocab_size-1}       (t = 0..T-1)
//
//   Step 1 — Byte embedding lookup:
//     x_t = W_emb[bytes_t]                              ∈ R^{1, d_model}
//     X ∈ R^{T, d_model}                                (stacked over t)
//
//   Step 2 — Selective SSM (identical to MambaBlock, no conv):
//     p_t       = in_proj(x_t)                          ∈ R^{1, 2*d_inner}
//     x_pre_t   = p_t[:d_inner] ;  g_t = p_t[d_inner:]
//     x̃_t       = silu(x_pre_t)
//     Δ_pre_t   = dt_proj(x_t)                          ∈ R^{d_inner}
//     Δ_t       = softplus(Δ_pre_t)                     > 0
//     B_t       = B_proj(x_t)                           ∈ R^{d_state}
//     C_t       = C_proj(x_t)                           ∈ R^{d_state}
//
//     Step 3 — Discretization (ZOH, scalar Δ_t per inner channel):
//     A = -exp(A_log)                                   ∈ R^{d_inner, d_state}
//     Ā_t = exp(Δ_t ⊗ A)                                ∈ R^{d_inner, d_state}
//     B̄_t = Δ_t ⊗ B_t                                  ∈ R^{d_inner, d_state}
//
//     Step 4 — Selective state recurrence:
//     h_0 = 0
//     h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x̃_t                 ∈ R^{d_inner, d_state}
//     y_t = C_t · h_t                                   ∈ R^{1, d_inner}
//
//     Step 5 — Output projection:
//     gated_t = silu(g_t) ⊙ y_t + D_skip ⊙ x̃_t
//     out_t   = out_proj(gated_t)                       ∈ R^{1, d_model}
//
//   Step 6 — Optional TWSS skip gate (paper §2.3):
//     out_t   = out_t + skip_gate ⊙ x_t                  if twss == true
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   Input:  (1, T)        byte indices in [0, vocab_size)
//   Output: (T, d_model)
//   Cache h, A_bar, B_bar: (T+1, d_inner, d_state)
//   Cache Δ, B, C, x_tilde, gated, etc.: (T, .)
//
// Backward:
//   - W_emb gradient: scatter add over rows indexed by bytes_t.
//   - Selective SSM: reverse-time BPTT over the cached Δ, B, C, Ā, B̄, h.
//   - Output projection: standard Dense backward.
//   - TWSS gradient: grad_skip_gate = sum_t grad_out_t ⊙ x_t.
// ----------------------------------------------------------------------------

class MambaByteBlock : public Layer {
public:
    // d_model:    input/output feature dim
    // d_state:    SSM state dim
    // d_inner:    inner feature dim (default = 2 * d_model, matches Mamba convention)
    // vocab_size: byte alphabet size (default 256)
    // twss:       if true, apply TWSS skip gate (paper §2.3)
    MambaByteBlock(size_t d_model, size_t d_state,
                   size_t d_inner = 0, size_t vocab_size = 256,
                   bool twss = false);

    // Forward: takes a (1, T) tensor of byte indices.
    Tensor forward(const Tensor& bytes) override;
    // Backward: takes (T, d_model) grad_output, returns (1, T) grad_bytes
    //           (gradient of the byte-index inputs; for embedding, used to
    //           update grad_W_emb via scatter).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_emb; }
    Tensor get_gradients() const override { return grad_W_emb_; }
    std::string name() const override { return "MambaByteBlock"; }

    // Accessors
    size_t d_model()    const { return d_model_; }
    size_t d_state()    const { return d_state_; }
    size_t d_inner()    const { return d_inner_; }
    size_t vocab_size() const { return vocab_size_; }
    bool   twss()       const { return twss_; }

    // ---- Parameters (public for tests) ----
    Tensor W_emb;           // (vocab_size, d_model)
    Tensor A_log;           // (d_inner, d_state)
    Tensor D_skip;          // (1, d_inner)
    Tensor skip_gate;       // (1, d_model)
    Dense in_proj;
    Dense out_proj;
    Dense dt_proj;
    Dense B_proj;
    Dense C_proj;

private:
    size_t d_model_, d_state_, d_inner_, vocab_size_;
    bool   twss_;

    // Cache (set in forward, used in backward)
public:
    Tensor last_p_;             // (T, 2*d_inner)   in_proj output
    Tensor last_bytes_;         // (1, T)
    Tensor last_embedded_;      // (T, d_model)
    Tensor last_x_pre_;         // (T, d_inner)
    Tensor last_g_;             // (T, d_inner)
    Tensor last_x_tilde_;       // (T, d_inner)
    Tensor last_Delta_;         // (T, d_inner)
    Tensor last_Delta_pre_;     // (T, d_inner)
    Tensor last_B_t_;           // (T, d_state)
    Tensor last_C_t_;           // (T, d_state)
    Tensor last_A_bar_;         // (T, d_inner, d_state)
    Tensor last_B_bar_;         // (T, d_inner, d_state)
    Tensor last_h_;             // (T+1, d_inner, d_state)        // h_0 ... h_{T-1}
    Tensor last_y_;             // (T, d_inner)
    Tensor last_gated_;         // (T, d_inner)
    Tensor last_out_proj_;      // (T, d_model) before any TWSS skip

    // Gradients (public so MambaByteModel::get_gradients() can read them)
    Tensor grad_W_emb_;         // (vocab_size, d_model)
    Tensor grad_A_log_;         // (d_inner, d_state)
    Tensor grad_D_skip_;        // (1, d_inner)
    Tensor grad_skip_gate_;     // (1, d_model)

private:

    // Numerically-stable helpers
    static double softplus(double x);
    static double sigmoid(double x);
    static double silu(double x);
};


class MambaByteModel : public Layer {
public:
    // input_dim:    raw input dim (we only use d_model for the SSM; input_dim kept
    //               in signature for consistency with Model::add_layer)
    // d_model:      SSM feature dim
    // output_dim:   number of output classes
    // num_layers:   number of stacked MambaByteBlocks
    // d_state:      passed to each block
    // d_inner:      passed to each block (default 0 → 2*d_model)
    // vocab_size:   passed to each block
    // twss:         passed to each block
    MambaByteModel(size_t input_dim, size_t d_model, size_t output_dim,
                   size_t num_layers, size_t d_state = 4,
                   size_t d_inner = 0, size_t vocab_size = 256,
                   bool twss = false);

    Tensor forward(const Tensor& bytes) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MambaByteModel"; }

    size_t num_layers() const { return blocks_.size(); }

private:
    size_t input_dim_, d_model_, output_dim_;
    std::vector<std::unique_ptr<MambaByteBlock>> blocks_;
    Dense classifier_;       // (d_model -> output_dim)
    Tensor last_block_out_;  // (T, d_model) — output of last block before classifier
};

#endif // MAMBA_BYTE_H
