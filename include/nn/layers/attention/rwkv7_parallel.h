#ifndef RWKV7_PARALLEL_H
#define RWKV7_PARALLEL_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// RWKV-7 Parallel Attention — Peng et al. 2025
//   "RWKV-7 'Goose' with Expressive Dynamic State Evolution"
//   https://arxiv.org/abs/2503.14456
//
// Parallel-form variant of the generalized-delta-rule update in RWKV7TimeMix.
// The mathematics is identical to the recurrent form — within a chunk of size
// C the recurrence `wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t` is applied
// sequentially. The "parallel" name reflects the canonical kernel structure
// (BlinkDL's flash-rwkv7) where each chunk is processed in parallel via the
// parallel-scan algorithm on GPU; in our C++ reference implementation each
// chunk is sequential but produces the same per-token outputs.
//
// For `chunk_size >= T` the layer is mathematically equivalent to the
// recurrent form (RWKV7TimeMix) — same forward output and same gradients
// to FP64 tolerance. This makes the parallel form useful as:
//   1. A benchmark target (caches the same intermediates as recurrent)
//   2. A verification oracle (randomized test: forward / grad equivalence)
//   3. A testbed for chunked-kernel experiments
//
// Inputs/Outputs match RWKV7TimeMix: (T, d) → (T, d), same token-shift +
// W_r/W_k/W_v/W_d/W_a projections + sigmoid/tanh/L2-normalize/lerp + wkv
// recurrence + receptance-reads-wkv output.
//
// Parameter sharing: all public members (W_r/W_k/W_v/W_d/W_a/xi/alpha/mu_*)
// are stored as Dense or raw Tensor and can be COPIED FROM RWKV7TimeMix (same
// names, same shapes, same semantics). Copy-assignment helper
// `copy_params_from(const RWKV7TimeMix&)` provided.
// ============================================================================

class RWKV7ParallelAttention : public Layer {
public:
    // d: input/output feature dim (must be > 0 and divisible by num_heads)
    // num_heads: number of heads (default 1); head_dim = d / num_heads
    // chunk_size: parallel-scan chunk size (default 0 → uses T)
    explicit RWKV7ParallelAttention(size_t d, size_t num_heads = 1, size_t chunk_size = 0);

    // Private validating helper.
    RWKV7ParallelAttention(size_t d, size_t num_heads, size_t chunk_size, bool validate_tag);

    // Standard Layer API
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_r.weights; }
    Tensor get_gradients() const override { return W_r.grad_weights; }
    std::string name() const override { return "RWKV7ParallelAttention"; }

    // Accessors
    size_t d() const { return d_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t chunk_size() const { return chunk_size_; }

    // ---- Public parameters (mirror RWKV7TimeMix for direct copy) ----
    size_t d_;
    size_t num_heads_;
    size_t head_dim_;
    size_t chunk_size_;
    Dense W_r;
    Dense W_k;
    Dense W_v;
    Dense W_d;
    Dense W_a;
    Tensor xi;              // (1, d)
    Tensor alpha;           // (1, 1)
    Tensor mu_r;            // (1, d)
    Tensor mu_k;            // (1, d)
    Tensor mu_v;            // (1, d)
    Tensor mu_d;            // (1, d)
    Tensor mu_a;            // (1, d)

    // Hidden gradient buffers
    Tensor grad_xi_;
    Tensor grad_alpha_;
    Tensor grad_mu_r_;
    Tensor grad_mu_k_;
    Tensor grad_mu_v_;
    Tensor grad_mu_d_;
    Tensor grad_mu_a_;

    // BPTT cache (mirrors RWKV7TimeMix; row t of last_wkv_ holds wkv_t).
    Tensor last_input_;
    Tensor last_x_shift_;
    Tensor last_r_in_;
    Tensor last_k_in_;
    Tensor last_v_in_;
    Tensor last_d_in_;
    Tensor last_a_in_;
    Tensor last_r_;
    Tensor last_k_;
    Tensor last_v_;
    Tensor last_d_pre_;
    Tensor last_a_pre_;
    Tensor last_d_;
    Tensor last_w_;
    Tensor last_a_;
    Tensor last_kappa_;
    Tensor last_kappa_hat_;
    Tensor last_kappa_norm_;
    Tensor last_k_tilde_;
    Tensor last_wkv_;           // (T+1, num_heads * m * m); row 0 is wkv_{-1}=0
                                // But we keep T+1 rows; row t is wkv_t (after
                                // update). For chunk_size=1, wkv_t should
                                // equal wkv_{t-1} + v_t^T k̃_t for that chunk.
};

#endif // RWKV7_PARALLEL_H