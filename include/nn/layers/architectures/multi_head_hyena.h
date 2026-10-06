#ifndef MULTI_HEAD_HYENA_H
#define MULTI_HEAD_HYENA_H

#include "../../core/layer.h"
#include "../normalization/rms_norm.h"
#include "../utility/swiglu.h"
#include "hyena.h"
#include <vector>
#include <memory>

// ============================================================================
// MultiHeadHyena (MultiHyena) — multi-head long convolutions
//
//   Massaroli, Poli et al. 2023. "Laughing Hyena Distillery: Extracting
//   Compact Recurrences From Convolutions", arXiv:2310.18780, §4.
//
// This is the layer MAD (Poli, Massaro, Muntoni, Singhal, Beigi et al. 2024,
// arXiv:2403.17844v2) evaluates as its "Multi-Head Hyena" baseline — its
// reference [24] — in Appendix B.3.2 and C.2:
//   "Multi-Head Hyena [24]: number of heads: 16, state dimension of heads: 2,
//    filter order: 2, short filter order: 3."
//
// §4, verbatim:
//
//   1. "Given the projections q,k,v ∈ R^{L×D}, we split them into M chunks of
//       size N = D/M, q^m,k^m,v^m ∈ R^{L×N}."
//   2. "Each chunk is processed by a modified Hyena operator: first, we
//       perform the outer product of k^m and v^m along the spatial dimension,
//       z^m ≜ k^m ⊗ v^m ∈ R^{L×N×N}, apply a long convolution with filter h^m
//       to all N×N elements independently, then compute
//       y^m_t = (h^m * z^m)_t q^m_t, y^m ∈ R^{L×N} as shown in Figure 4.1."
//   3. "Finally, we compose y^1,…,y^m into a single output y ∈ R^{L×D} via
//       concatenation."
//
// WHY THIS IS NOT THE SHIPPED `HyenaOperator`
// (arXiv:2302.10866v3 Definition 3.1, in layers/architectures/hyena.h):
//
//   * Hyena_N featurizes with a long convolution over the CHANNEL axis of
//     order N (order=2 → two cascaded filter stages plus two gates). Here
//     there is exactly ONE long convolution per head, applied to an
//     outer-product plane.
//   * The defining featurization here is the OUTER PRODUCT k^m ⊗ v^m ∈ R^
//     {L×N×N}. That N²-ary featurization is what makes the layer state-sized
//     (the "outer-product head trick" MAD §2.2 attributes to this layer
//     family): state grows from N to N² per head while the filter count
//     stays at ONE per head.
//   * Crucially, `h^m` is a SINGLE filter applied to ALL N×N elements
//     ("to all N×N elements independently"). §4 is motivated by the finding
//     that "at initialization, filters correspond to high-dimensional SSMs,
//     and gradually converge to lower-dimensional representations during
//     training" — tying weights across the outer-product plane is the whole
//     thesis. A per-(i,j) filter would not be MultiHyena.
//   * `·q^m_t` is the HADAMARD product: (h*z)_t ∈ R^{N×N} times q^m_t ∈ R^N
//     reduces to R^N, which is what makes the concatenation of M heads give
//     back R^{L×D}.
//
// v1 scope decisions (see docs/plans/2026-10-06-multi-head-hyena.md §Decisions):
//   * The filter generator is the already-shipped `HyenaFilter` — MAD
//     App. B.3.2 attributes the filter featurization to this same ref [24],
//     so a second generator would duplicate a proven chain.
//   * No short depthwise convolution. §4's definition has none; the shipped
//     `HyenaOperator` already covers the short-conv variant.
//   * Naive O(L²) convolution, per the repo-wide convention documented in
//     hyena.h (keeps finite-difference gradient checks tractable at small L).
//
// Conventions: sequences are (L, d_model) row-major Tensors. Inside the
// operator the L×N×N outer-product plane is flattened to (L, N*N) with
// element c = i*N + j, so everything stays a 2-D Tensor.
// ============================================================================

class MultiHeadHyenaOperator : public Layer {
public:
    // Public config, trailing-underscore members so the accessors keep the
    // repo's `name()` convention without colliding.
    size_t d_model_, seq_len_, num_heads_, head_dim_, filter_order_;

    // d_model must be divisible by num_heads; head_dim = d_model / num_heads.
    MultiHeadHyenaOperator(size_t d_model, size_t seq_len, size_t num_heads,
                           size_t filter_order = 16);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return in_proj_.weights; }
    Tensor get_gradients() const override { return in_proj_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "MultiHeadHyenaOperator"; }

    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t filter_order() const { return filter_order_; }

    // --- Test / introspection surface -------------------------------------
    // Drive the operator directly from q,k,v (each (L, d_model)) instead of
    // via in_proj. The tests need this so their independent reference and the
    // implementation see IDENTICAL projections — otherwise the reference would
    // be testing in_proj as well as the §4 math.
    Tensor forward_from_qkv(const Tensor& q, const Tensor& k, const Tensor& v);

    // Cached filter h^m for head m: (L, N*N). One filter per head, shared
    // across that head's whole outer-product plane.
    Tensor last_filter(size_t head) const;

    // Cached outer product z^m for head m: (L, N*N), element c = i*N + j.
    Tensor last_outer(size_t head) const;

    // Cached long-convolution output w^m for head m: (L, N*N).
    Tensor last_conv(size_t head) const;

    // Gradient w.r.t. the operator input, (L, d_model), written by backward().
    Tensor last_input_grad() const { return last_input_grad_; }

    // Install a filter for one head, bypassing generation. Test-only: lets the
    // suite pin the shared-filter contract against a known non-degenerate h.
    void set_filter_for_test(size_t head, const Tensor& h);

// The two projections. Public so a composing Layer (MultiHeadHyenaBlock,
    // tests) can read them without an extra accessor pair — same convention as
    // the shipped `HyenaOperator`, which exposes `in_proj`/`out_proj` public.
    Dense in_proj_;    // (3*d_model, d_model)
    Dense out_proj_;   // (d_model, d_model)

private:
    // One filter GENERATOR per head, built with d_model = 1 so it emits a
    // SINGLE filter column — see the shared-filter note below.
    std::vector<std::unique_ptr<HyenaFilter>> filters_;

    // The shared filter actually used per head, (L, N*N): h[l][c] is the SAME
    // value for every c. §4 asks "can we reduce the total number of filters
    // without loss in quality?" and answers yes — one filter per head is
    // broadcast across that head's whole N×N outer-product plane, so M heads
    // use M filters instead of N²·M.
    std::vector<Tensor> last_h_;    // per head: (L, N*N)
    std::vector<bool> pinned_;       // true => that head's filter is an override
    Tensor override_h_;              // the pinned filter, used for the pinned head

    // Caches from the last forward (needed by backward).
    Tensor last_input_;        // (L, d_model)
    Tensor last_qkv_;          // (3L, d_model) — q,k,v stacked, rows [0,L)=q
    std::vector<Tensor> last_z_;       // per head: (L, N*N)
    std::vector<Tensor> last_w_;       // per head: (L, N*N) conv output (h*z)
    std::vector<Tensor> last_qh_;      // per head: (L, N)
    Tensor last_input_grad_;   // (L, d_model)
};

// ----------------------------------------------------------------------------
// MultiHeadHyenaBlock: pre-RMSNorm → MultiHeadHyenaOperator → residual
//                     → pre-RMSNorm → SwiGLU → down-projection → residual
//
// Appendix C.2 of MAD: "The channel mixer is replaced with SwiGLU, we use
// RMSNorm" — shared by the Hyena and Multi-Head Hyena families.
//
// The SwiGLU down-projection is REQUIRED, not optional: SwiGLU has no output
// projection of its own, so without it the channel mixer returns ffn_mult
// times too wide and the residual has nothing to add to. Same reason as
// MegalodonBlock and StripedHyenaBlock.
// ----------------------------------------------------------------------------
class MultiHeadHyenaBlock : public Layer {
public:
    size_t d_model_, seq_len_, num_heads_, ffn_mult_;

    // ffn_mult == 0 skips the channel mixer (used by FD checks to isolate
    // the sequence mixer chain).
    MultiHeadHyenaBlock(size_t d_model, size_t seq_len, size_t num_heads,
                        size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return operator_.in_proj_.weights; }
    Tensor get_gradients() const override { return operator_.in_proj_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "MultiHeadHyenaBlock"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t seq_len() const { return seq_len_; }
    size_t ffn_mult() const { return ffn_mult_; }

    Tensor last_input_grad() const { return last_input_grad_; }
    MultiHeadHyenaOperator* mixer() { return &operator_; }

private:
    MultiHeadHyenaOperator operator_;
    std::unique_ptr<RMSNorm> ln1_;
    std::unique_ptr<RMSNorm> ln2_;
    std::unique_ptr<SwiGLU<>> ffn_;        // null when ffn_mult_ == 0
    std::unique_ptr<Dense> ffn_down_;      // null when ffn_mult_ == 0

    Tensor last_input_, last_n1_, last_mix_, last_u_, last_n2_;
    Tensor last_input_grad_;
};

// ----------------------------------------------------------------------------
// MultiHeadHyenaModel: input projection → stack of blocks → final RMSNorm
//                      → mean-pool over the sequence → classifier.
// ----------------------------------------------------------------------------
class MultiHeadHyenaModel : public Layer {
public:
    MultiHeadHyenaModel(size_t input_dim, size_t d_model, size_t num_layers,
                        size_t output_dim, size_t seq_len, size_t num_heads,
                        size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return in_proj_.weights; }
    Tensor get_gradients() const override { return in_proj_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "MultiHeadHyenaModel"; }

    size_t num_layers() const { return num_layers_; }
    size_t seq_len() const { return seq_len_; }
    size_t output_dim() const { return output_dim_; }

    std::vector<MultiHeadHyenaBlock> blocks;

private:
    Dense in_proj_;      // (d_model, input_dim)
    std::unique_ptr<RMSNorm> final_norm_;
    Dense classifier_;   // (output_dim, d_model)

    size_t d_model_, num_layers_, output_dim_, seq_len_, num_heads_, ffn_mult_;
    Tensor last_input_, last_proj_, last_pooled_;
};

#endif