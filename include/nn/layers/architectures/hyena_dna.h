#ifndef HYENA_DNA_H
#define HYENA_DNA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <string>

// ============================================================================
// HyenaDNA (Nguyen, Poli, Faizi, Thomas, Birch-Sykes, Wornow, Patel, Rabideau,
// Massaroli, Bengio, Ermon, Baccus, Re 2023) — "HyenaDNA: Long-Range Genomic
// Sequence Modeling at Single Nucleotide Resolution"
// https://arxiv.org/abs/2306.15794
//
// Innovation over the shipped single-head Hyena operator (hyena.{h,cpp}):
// the operator is applied INDEPENDENTLY to num_heads channel groups, so the
// filter bank produces head_dim*(order-1) long-conv filters instead of one
// D-wide filter, and the per-channel skip term is a plain learnable `bias`
// applied after the long convolution rather than folded into the filter.
//
// Operator (arXiv:2306.15794 §3.1, Eq. 3.1):
//   (x_1, x_2, v) -> H(x_1, x_2) v,   H(x_1, x_2) = D_{x_2} T_h D_{x_1}
// where T_h is the Toeplitz matrix of a learnable long-conv filter h produced
// by a small implicit MLP, and D_x is element-wise gating by the projections x.
// The order-N form is the recurrence written out in forward() below.
//
// Per head h, with the short convolution applied to the concatenated
// projections:
//   u    = in_proj(x)                                (L, (order+1)*D)
//   uc   = short_depthwise_conv1d(u)                 causal, kernel short_filter_order
//   x_0..x_order, v = split(uc)                      per head, each (L, head_dim)
//   for o = order-1 .. 1:
//       v = v * x[o]                                 gate D_{x_o}
//       v = long_conv(v, k_o) + bias_o * v           T_h then per-channel skip
//   y    = v * x[0]                                  final gate
//   y    = concat_heads(y) ; out_proj(y)
//
// Filter (official HyenaFilter):
//   z = positional_embedding(L)                      (L, emb_dim)
//   h = Linear(emb_dim, P) -> Sin -> [Linear(P,P) -> Sin] * num_inner
//       -> Linear(P, D_f, bias=false)
//   h = (exp(-t * |deltas|) + shift) * h             ExponentialModulation
//
// Model (HyenaDNAModel): stack of blocks -> mean pool -> classifier.
// Block (HyenaDNABlock):
//   x = x + HyenaDNAOperator(LayerNorm(x))
//   x = x + GELU-FFN(LayerNorm(x))                   skipped when ffn_mult == 0
//
// Two paper utilities also ship here:
//   SequenceLengthWarmup (§3.2) — the staged 64 -> 128 -> ... schedule
//   SoftPrompting (§3.3, Eq. 3.2) — concat[embed(x_p), theta] with theta
//                                   the only trainable tensor
//
// All tensors in repo convention: row-major (rows, cols). Sequences are
// (L, d_model) and unbatched.
//
// The long convolutions use a naive O(L^2) recurrence (same rationale as
// hyena.cpp: tractable at the small L the gradient checks use, and a clean
// drop-in upgrade to an FFT-based evaluation).
// ============================================================================

// ----------------------------------------------------------------------------
// HyenaDNAFilter: the implicit long-conv filter bank (one filter per
// (head, order) pair) plus its per-channel skip bias.
//
// The filter output width is d_model, which the operator sets to
// head_dim * (order - 1) so that a single shared filter serves every head.
// Index layout of the (L, d_model) output, matching the reference
// `rearrange(k, "c l (v o) -> c o v l", v=head_dim, o=order-1)`:
//   k[o][v] == output[l][v * (order - 1) + o]
//   bias_o[v] == bias_param[v * (order - 1) + o]
// ----------------------------------------------------------------------------
class HyenaDNAFilter {
public:
    // Config as trailing-underscore members so the accessors below can keep the
    // repo's `name()`-style accessor convention (see striped_hyena.h) without
    // colliding with the member names.
    size_t d_model_;       // D_f — output width of one filter() call
    size_t l_max_;         // L — max sequence length
    size_t filter_order_;  // P — width of the implicit MLP
    size_t emb_dim_;       // positional-embedding dim (odd, >= 3)
    size_t num_inner_;     // number of inner Linear+Sin layers

    // --- learnable parameters
    Tensor mlp_in_W;       // (P, emb_dim)
    Tensor mlp_in_b;       // (1, P)
    Tensor sin_freq;       // (1, P) — learnable frequency of the Sin activation
    std::vector<Tensor> mlp_W;   // num_inner x (P, P)
    std::vector<Tensor> mlp_b;   // num_inner x (1, P)
    Tensor mlp_out_W;      // (D_f, P) — final projection, NO bias
    Tensor deltas;         // (1, D_f) — exponential-modulation decay rates
    Tensor bias;           // (1, D_f) — per-channel skip (the "D" in y = T v + D v)

    // --- gradient buffers
    Tensor grad_mlp_in_W, grad_mlp_in_b, grad_sin_freq;
    std::vector<Tensor> grad_mlp_W, grad_mlp_b;
    Tensor grad_mlp_out_W, grad_deltas, grad_bias;

    // --- cached forward state (for backward)
    Tensor last_z;         // (L, emb_dim)  positional embedding
    std::vector<Tensor> last_pre;   // sin inputs, one per Sin layer
    std::vector<Tensor> last_post;  // sin outputs, one per Sin layer
    Tensor last_h0;        // (L, D_f) pre-modulation filter
    Tensor last_decay;     // (L, D_f) exp(-t*|deltas|) + shift

    // ExponentialModulation defaults, from the reference implementation.
    static constexpr double kFastDecayPct = 0.3;
    static constexpr double kSlowDecayPct = 1.5;
    static constexpr double kModulationTarget = 1e-2;
    static constexpr double kModulationShift = 0.0;

    // emb_dim must be odd and >= 3 (time + cos + sin bands), as asserted in
    // the reference implementation.
    HyenaDNAFilter(size_t d_model, size_t l_max, size_t filter_order = 16,
                   size_t emb_dim = 3, size_t num_inner = 1);

    // Complex-exponential positional embedding (L, emb_dim):
    //   t   = linspace(0, 1, L)                        normalized so t_L = 1
    //   bands = (emb_dim - 1) / 2
    //   w   = 2*pi*linspace(0, L-1, L) / L
    //   f   = linspace(1e-4, bands-1, bands)
    //   z   = [t, cos(f*w), -sin(f*w)]
    Tensor positional_embedding(size_t L) const;

    // Generate the filter for sequence length L (L <= l_max). Caches
    // intermediates for backward.
    Tensor filter(size_t L);

    // Backward: grad_h is (L, D_f); accumulates into the grad_* buffers.
    void backward(const Tensor& grad_h);

    void zero_grad();

    // Manual SGD step over every parameter/grad pair. The filter holds raw
    // tensors rather than Dense layers, so this is its own method.
    void update_weights(double lr);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    size_t d_model() const { return d_model_; }
    size_t l_max() const { return l_max_; }
    size_t filter_order() const { return filter_order_; }
    size_t emb_dim() const { return emb_dim_; }
    size_t num_inner() const { return num_inner_; }

    // Test seams: the cached positional embedding and modulation decay.
    Tensor last_positional_embedding() const { return last_z; }
    Tensor last_decay_term() const { return last_decay; }
    const std::vector<Tensor>& last_sin_pre() const { return last_pre; }
    const std::vector<Tensor>& last_sin_post() const { return last_post; }
};

// ----------------------------------------------------------------------------
// HyenaDNAOperator: one multi-head Hyena operator layer.
// Input: (L, d_model). Output: (L, d_model).
// ----------------------------------------------------------------------------
class HyenaDNAOperator : public Layer {
public:
    size_t d_model_;          // D
    size_t l_max_;            // L
    size_t num_heads_;        // H
    size_t head_dim_;         // D / H
    size_t order_;            // order of the recurrence, >= 2
    size_t filter_order_;     // P
    size_t short_filter_order_;  // short depthwise conv kernel size

    Dense in_proj;            // D -> (order+1)*D, applied per token
    Dense out_proj;           // D -> D
    Tensor short_W;           // ((order+1)*D, short_filter_order)
    Tensor short_b;           // ((order+1)*D, 1)
    HyenaDNAFilter filter;    // output width head_dim*(order-1)

    // --- cached forward state
    Tensor last_input;        // (L, D)
    Tensor last_in_proj;      // (L, (order+1)*D)
    Tensor last_uc;           // (L, (order+1)*D) post short conv
    std::vector<Tensor> last_gate;      // (order) x (L, D)  the x_o gates
    std::vector<Tensor> last_v_before;  // (order) x (L, D)  v entering gate o
    std::vector<Tensor> last_v_gated;   // (order) x (L, D)  v entering conv o
    Tensor last_y_pre_out;    // (L, D) pre out_proj
    Tensor last_filter_h;     // (L, head_dim*(order-1))

    // --- gradient buffers
    Tensor grad_short_W, grad_short_b;

    HyenaDNAOperator(size_t d_model, size_t l_max, size_t num_heads = 1,
                     size_t order = 2, size_t filter_order = 16,
                     size_t short_filter_order = 3);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaDNAOperator"; }

    size_t d_model() const { return d_model_; }
    size_t l_max() const { return l_max_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t order() const { return order_; }
    size_t filter_order() const { return filter_order_; }
    size_t short_filter_order() const { return short_filter_order_; }
};

// ----------------------------------------------------------------------------
// HyenaDNABlock: pre-LN -> operator -> residual -> pre-LN -> GELU FFN -> residual
// ffn_mult == 0 skips the channel mixer entirely (used by the FD checks to
// isolate the sequence-mixer chain).
// ----------------------------------------------------------------------------
class HyenaDNABlock : public Layer {
public:
    size_t d_model_, l_max_, ffn_mult_;

    LayerNorm ln1, ln2;
    HyenaDNAOperator op;
    Dense ffn1;               // d_model -> ffn_mult * d_model
    Dense ffn2;               // ffn_mult * d_model -> d_model

    Tensor last_input;        // (L, D)
    Tensor last_res1;         // (L, D) block input after the first residual

    HyenaDNABlock(size_t d_model, size_t l_max, size_t num_heads = 1,
                  size_t order = 2, size_t filter_order = 16,
                  size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaDNABlock"; }

    size_t d_model() const { return d_model_; }
    size_t l_max() const { return l_max_; }
    size_t ffn_mult() const { return ffn_mult_; }
};

// ----------------------------------------------------------------------------
// HyenaDNAModel: stack of HyenaDNABlocks -> per-token mean pool -> classifier.
// Input: (L, d_model). Output: (num_classes,)  [1 x num_classes]
//
// v1 divergence from the paper: the paper's head is a per-token
// next-nucleotide LM head. A pooled classifier matches every other model
// wrapper in this repo (HyenaModel, StripedHyenaModel, MegalodonModel).
// ----------------------------------------------------------------------------
class HyenaDNAModel : public Layer {
public:
    size_t d_model_, l_max_, depth_, num_classes_;

    std::vector<HyenaDNABlock> blocks;
    Dense classifier;         // d_model -> num_classes

    Tensor last_input;        // (L, D)

    HyenaDNAModel(size_t d_model, size_t l_max, size_t depth, size_t num_classes,
                  size_t num_heads = 1, size_t order = 2,
                  size_t filter_order = 16, size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "HyenaDNAModel"; }

    size_t d_model() const { return d_model_; }
    size_t l_max() const { return l_max_; }
    size_t depth() const { return depth_; }
    size_t num_classes() const { return num_classes_; }
};

// ----------------------------------------------------------------------------
// SequenceLengthWarmup (arXiv:2306.15794 §3.2) — the training schedule that
// starts at L_1 = 64 and doubles the window at each stage while holding the
// global batch size constant. "By doing so, iterations at each consecutive
// stage will include more tokens, ensuring the scheduler can also act as a
// form of batch size warm-up."
//
// NOT a Layer: this is a training-loop concern with no tensors. Stage i runs
// for epoch_scale * 2^i epochs at sequence length start_len * 2^i.
// ----------------------------------------------------------------------------
class SequenceLengthWarmup {
public:
    size_t start_len_;    // L_1, default 64 (the paper's value)
    size_t num_stages_;   // number of doubling stages
    size_t epoch_scale_;  // epochs per stage at stage 0

    SequenceLengthWarmup(size_t start_len = 64, size_t num_stages = 8,
                         size_t epoch_scale = 1);

    size_t num_stages() const { return num_stages_; }
    size_t start_len() const { return start_len_; }
    size_t epoch_scale() const { return epoch_scale_; }

    // Stage index for an epoch (0-based). Past the last stage the schedule
    // saturates at the final stage rather than overflowing the shift.
    size_t stage_for_epoch(size_t epoch) const;
    // Piecewise-constant sequence length: start_len * 2^stage.
    size_t seq_len_for_epoch(size_t epoch) const;
    // Total epochs across all stages: epoch_scale * (2^num_stages - 1).
    size_t total_epochs() const;
    // Sequence length at the final stage: start_len * 2^(num_stages-1).
    size_t max_seq_len() const;
};

// ----------------------------------------------------------------------------
// SoftPrompting (arXiv:2306.15794 §3.3, Eq. 3.2):
//   x <- concat[embed(x_p), theta],   x in R^{(T+N) x D}
// theta is a learnable (N, D) prompt prepended (or appended) to the embedded
// input sequence, and is the ONLY thing optimized — every model parameter
// stays fixed.
//
// The caller is responsible for choosing N <= L - T; the concatenation
// arithmetic is the caller's. forward throws only on a d_model mismatch.
// ----------------------------------------------------------------------------
class SoftPrompting : public Layer {
public:
    size_t prompt_len_;   // N
    size_t d_model_;      // D
    bool at_front_;       // true: [theta, embedded]; false: [embedded, theta]

    Tensor prompt_;       // (N, D) — the trainable theta
    Tensor grad_prompt_;  // (N, D)

    Tensor last_embedded;  // (T, D) cached input

    SoftPrompting(size_t prompt_len, size_t d_model, bool at_front = true);

    Tensor forward(const Tensor& embedded) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return prompt_; }
    Tensor get_gradients() const override { return grad_prompt_; }
    std::vector<Tensor*> parameters() override { return {&prompt_}; }
    std::vector<Tensor*> gradients() override { return {&grad_prompt_}; }
    void zero_grad() override;

    size_t prompt_len() const { return prompt_len_; }
    size_t d_model() const { return d_model_; }
    bool at_front() const { return at_front_; }
    Tensor& prompt() { return prompt_; }
};

#endif
