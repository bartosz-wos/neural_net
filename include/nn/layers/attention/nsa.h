#ifndef NSA_H
#define NSA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// Native Sparse Attention (NSA) — DeepSeek-AI 2025
//   "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
//    Attention" — https://arxiv.org/abs/2502.11089
//
// Three parallel branches combined via a learned per-(token, head, branch)
// softmax gate (paper Eq. 5):
//
//   out_t = Σ_{c∈{cmp,slc,win}} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)
//
//   1. Compression (paper §3.3.1, Eq. 7):
//      Blockwise aggregation of keys/values via a per-head learnable MLP φ
//      applied to (block of l tokens + sinusoidal intra-block position
//      encoding). Sliding stride d < l. Produces n_cmp blocks per head.
//      → Coarse-grained global pattern recognition.
//
//   2. Selection (paper §3.3.2, Eq. 8-11):
//      Blockwise importance scores induced from compression attention scores
//      `p_cmp = softmax(q^T · K̃_cmp)` (Eq. 8), aggregated across heads in a
//      GQA group (Eq. 10), then top-n blocks retained (Eq. 11).
//      → Fine-grained important token preservation.
//
//   3. Sliding window (paper §3.3.3):
//      Last `w` tokens only. → Local-context shortcut.
//
// Per paper §3.3.3 ("we provide independent keys and values for three
// branches"), each branch has its OWN K/V projection — this prevents
// gradient interference between local and long-range branches.
//
// A single shared Q projection.
//
// Multi-head with K/V sharing via GQA convention (num_query_heads,
// num_kv_heads), mirroring GQA in this repo.
//
// Conventions (match the rest of the attention family):
//   * Input/Output: (n, d_model)
//   * d_model must be evenly divisible by num_query_heads (= head_dim)
//   * num_query_heads must be evenly divisible by num_kv_heads
//   * block_len > 0, stride in [1, block_len], top_n > 0, window_size > 0
//   * n must be >= block_len (so compression produces at least one block)
//   * n must be >= window_size (so sliding window fits)
//   * top_n <= (n - block_size)/stride + 1 (so selection produces <= top_n blocks)
//
// NSAAttention — the attention layer itself
// NSABlock     — pre-LN → NSA → residual → optional pre-LN FFN → residual
// NSAModel     — in_proj → N blocks → classifier
// ============================================================================

class NSAAttention : public Layer {
public:
    NSAAttention(size_t d_model,
                 size_t num_query_heads,
                 size_t num_kv_heads,
                 size_t block_len,       // l: compression block length
                 size_t stride,          // d: sliding stride between compression blocks
                 size_t top_n,           // n: top-n selection blocks retained
                 size_t window_size,     // w: sliding window size
                 size_t block_size = 0); // l': selection block size; 0 → defaults to stride

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "NSAAttention"; }

    // Accessors
    size_t d_model()        const { return d_model_; }
    size_t num_heads()      const { return num_query_heads_; }
    size_t num_kv_heads()   const { return num_kv_heads_; }
    size_t head_dim()       const { return head_dim_; }
    size_t block_len()      const { return block_len_; }
    size_t stride()         const { return stride_; }
    size_t top_n()          const { return top_n_; }
    size_t window_size()    const { return window_size_; }
    size_t block_size()     const { return block_size_; }

    // Cache sizes from last forward (computed from N)
    size_t n_cmp()          const { return n_cmp_; }
    size_t n_sel_blocks()   const { return n_sel_blocks_; }

    // Test helper: per-(token, head, branch) softmax gate from last forward.
    // Shape (n, num_heads, 3). Sums to 1 across the last dim.
    const Tensor& last_gate() const { return last_gate_; }

    // Test helpers: per-head attention maps from last forward.
    // last_attn_cmp_ : (num_query_heads, N, n_cmp)
    // last_attn_sel_ : (num_query_heads, N, top_n * block_size)
    // last_attn_win_ : (num_query_heads, N, window_size)
    const Tensor& last_attn_cmp() const { return last_attn_cmp_; }
    const Tensor& last_attn_sel() const { return last_attn_sel_; }
    const Tensor& last_attn_win() const { return last_attn_win_; }

    // Test helper: cached input gradient from last backward.
    Tensor grad_input() const { return last_d_input_; }

    // Public parameter tensors for tests (matches GQA / sliding_window convention)
    Tensor W_q, W_o;                                   // (d_model, d_model)
    Tensor W_k_cmp, W_v_cmp, W_k_sel, W_v_sel,
           W_k_win, W_v_win;                           // (d_model, d_model)
    Tensor W_phi_k, W_phi_v;                           // (num_heads, head_dim, l*head_dim)
    Tensor W_gate;                                     // (num_heads, 3, head_dim)
    Tensor grad_W_q, grad_W_o;
    Tensor grad_W_k_cmp, grad_W_v_cmp, grad_W_k_sel,
           grad_W_v_sel, grad_W_k_win, grad_W_v_win;
    Tensor grad_W_phi_k, grad_W_phi_v;
    Tensor grad_W_gate;

    // Mutation hooks (for non-vacuous mutation tests)
    void set_use_compression(bool b) { use_compression_ = b; }
    void set_use_selection(bool b)   { use_selection_   = b; }
    void set_use_window(bool b)      { use_window_      = b; }
    bool use_compression() const { return use_compression_; }
    bool use_selection()   const { return use_selection_;   }
    bool use_window()      const { return use_window_;      }

private:
    size_t d_model_;
    size_t num_query_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t group_size_;
    size_t block_len_;
    size_t stride_;
    size_t top_n_;
    size_t window_size_;
    size_t block_size_;
    size_t n_cmp_;          // (N - l) / d + 1
    size_t n_sel_blocks_;   // (N - l') / d + 1
    double scale_;          // 1 / sqrt(head_dim)

    bool use_compression_ = true;
    bool use_selection_   = true;
    bool use_window_      = true;

    // Fixed sinusoidal intra-block position encoding (l, head_dim).
    Tensor pos_enc_;

    // BPTT caches (filled by forward, consumed by backward)
    size_t N_last_ = 0;
    Tensor last_input_;
    Tensor last_Q_;
    Tensor last_K_cmp_, last_V_cmp_;
    Tensor last_K_sel_, last_V_sel_;
    Tensor last_K_win_, last_V_win_;
    Tensor last_K_tilde_cmp_, last_V_tilde_cmp_;  // compressed KV
    Tensor last_K_tilde_sel_, last_V_tilde_sel_;  // selected KV (concat'd blocks)
    Tensor last_K_tilde_win_, last_V_tilde_win_;  // sliding-window KV
    Tensor last_p_cmp_;                            // (H, N, n_cmp) compression attn scores
    Tensor last_p_cmp_per_head_;                   // (H, N, n_sel_blocks) aggregated importance per head
    Tensor last_top_idx_;                          // (H, N, top_n) selected block indices
    Tensor last_attn_cmp_, last_attn_sel_, last_attn_win_;
    Tensor last_gate_;                             // (N, num_heads, 3) softmax
    Tensor last_O_branches_;                       // (N, 3, d_model) per-branch output pre-gate
    Tensor last_O_;                                // (N, d_model) post-gate pre-W_o
    Tensor last_d_input_;
};

// ----------------------------------------------------------------------------
// NSABlock: pre-LN → NSA → residual → optional pre-LN FFN → residual
// ----------------------------------------------------------------------------
class NSABlock : public Layer {
public:
    NSABlock(size_t d_model, size_t num_query_heads, size_t num_kv_heads,
             size_t block_len, size_t stride, size_t top_n,
             size_t window_size, size_t block_size = 0, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return nsa_.W_q; }
    Tensor get_gradients() const override { return nsa_.grad_W_q; }
    std::string name() const override { return "NSABlock"; }

    size_t d_model() const { return d_model_; }
    size_t ffn_dim() const { return ffn_dim_; }
    Tensor grad_input() const { return last_d_input_; }

private:
    size_t d_model_, ffn_dim_;
    NSAAttention nsa_;
    LayerNorm ln1_, ln2_;

    Tensor W1_, b1_, W2_, b2_;
    Tensor grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    Tensor last_input_;
    Tensor last_ln1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_ln2_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_out_;
    Tensor last_d_input_;
};

// ----------------------------------------------------------------------------
// NSAModel: in_proj → N blocks → classifier
// ----------------------------------------------------------------------------
class NSAModel : public Layer {
public:
    NSAModel(size_t input_dim, size_t d_model, size_t output_dim,
             size_t num_blocks, size_t num_query_heads, size_t num_kv_heads,
             size_t block_len, size_t stride, size_t top_n,
             size_t window_size, size_t block_size = 0, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::string name() const override { return "NSAModel"; }

private:
    size_t input_dim_, d_model_, output_dim_;
    Tensor W_in_, b_in_, W_out_, b_out_;
    Tensor grad_W_in_, grad_b_in_, grad_W_out_, grad_b_out_;
    std::vector<NSABlock> blocks_;

    Tensor last_input_;
    Tensor last_proj_;
    Tensor last_block_out_;
};

#endif
