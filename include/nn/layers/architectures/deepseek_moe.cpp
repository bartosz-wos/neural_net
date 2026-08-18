#include "deepseek_moe.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Numerically stable sigmoid.
inline double sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = std::exp(x);
    return z / (1.0 + z);
}

// SiLU (Swish-1) activation: x * sigmoid(x).
inline double silu(double x) {
    return x * sigmoid(x);
}

// d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//             = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
inline double silu_deriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 + x * (1.0 - s));
}

// Kaiming-He init (we use this for expert FFNs and the gate).
// Inputs: in_f (fan_in), out_f (fan_out).
void init_dense_he(Dense& d, size_t in_f, size_t out_f, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / static_cast<double>(in_f)));
    std::vector<double> w(in_f * out_f);
    for (auto& v : w) v = nd(rng);
    d.weights = Tensor(out_f, in_f, w.data());
    d.bias = Tensor(1, out_f);
    std::fill(d.bias.data.begin(), d.bias.data.end(), 0.0);
    d.grad_weights = Tensor(out_f, in_f);
    std::fill(d.grad_weights.data.begin(), d.grad_weights.data.end(), 0.0);
    d.grad_bias = Tensor(1, out_f);
    std::fill(d.grad_bias.data.begin(), d.grad_bias.data.end(), 0.0);
}

// Element-wise addition: c = a + b.
void tensor_add(const Tensor& a, const Tensor& b, Tensor& c) {
    for (size_t i = 0; i < a.data.size(); ++i) c.data[i] = a.data[i] + b.data[i];
}

}  // namespace

// ============================================================================
// DeepSeekMoELayer
// ============================================================================

DeepSeekMoELayer::DeepSeekMoELayer(size_t d_model, size_t d_expert,
                                   size_t num_routed, size_t num_shared,
                                   size_t top_k_routed)
    : d_model_(d_model),
      d_expert_(d_expert),
      num_routed_(num_routed),
      num_shared_(num_shared),
      top_k_routed_(num_routed > 0 ? top_k_routed : 0),
      seg_size_(num_routed > 0 ? d_model / num_routed : 0),
      W_g_(num_routed > 0 ? d_model : 1, num_routed > 0 ? num_routed : 1) {
    if (d_model == 0)
        throw std::invalid_argument("DeepSeekMoELayer: d_model must be > 0");
    if (d_expert == 0)
        throw std::invalid_argument("DeepSeekMoELayer: d_expert must be > 0");
    if (num_routed > 0) {
        if (d_model % num_routed != 0)
            throw std::invalid_argument("DeepSeekMoELayer: d_model must be divisible by num_routed");
        if (top_k_routed == 0)
            throw std::invalid_argument("DeepSeekMoELayer: top_k_routed must be > 0 when num_routed > 0");
        if (top_k_routed > num_routed)
            throw std::invalid_argument("DeepSeekMoELayer: top_k_routed cannot exceed num_routed");
    }

    // Allocate routed experts.
    experts_.reserve(num_routed);
    for (size_t i = 0; i < num_routed; ++i) {
        experts_.emplace_back(seg_size_, d_expert);
        init_dense_he(experts_[i].W1, seg_size_, d_expert, 0xD00D0001u + (unsigned)i * 7u);
        init_dense_he(experts_[i].W2, d_expert, seg_size_, 0xD00D0002u + (unsigned)i * 7u);
    }

    // Allocate shared experts.
    shared_experts_.reserve(num_shared);
    for (size_t j = 0; j < num_shared; ++j) {
        shared_experts_.emplace_back(d_model, d_expert);
        init_dense_he(shared_experts_[j].W1, d_model, d_expert, 0xBEEF0001u + (unsigned)j * 11u);
        init_dense_he(shared_experts_[j].W2, d_expert, d_model, 0xBEEF0002u + (unsigned)j * 11u);
    }
}

Tensor DeepSeekMoELayer::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("DeepSeekMoELayer::forward: input feature dim mismatch");
    const size_t T = input.rows;
    last_input = input;

    // --- Shared expert path: each shared FFN operates on full input ---
    last_shared_out = Tensor(T, d_model_);
    std::fill(last_shared_out.data.begin(), last_shared_out.data.end(), 0.0);
    for (size_t j = 0; j < num_shared_; ++j) {
        // W1 · x : (T, d_model) → (T, d_expert)
        Tensor z = shared_experts_[j].W1.forward(input);
        // SiLU
        Tensor h(z.rows, z.cols);
        for (size_t i = 0; i < z.data.size(); ++i) h.data[i] = silu(z.data[i]);
        // W2 · h : (T, d_expert) → (T, d_model)
        Tensor out = shared_experts_[j].W2.forward(h);
        for (size_t i = 0; i < out.data.size(); ++i) last_shared_out.data[i] += out.data[i];
    }

    // --- Routed path (or pure-shared short-circuit) ---
    last_routed_out = Tensor(T, d_model_);
    std::fill(last_routed_out.data.begin(), last_routed_out.data.end(), 0.0);

    if (num_routed_ == 0) {
        // Pure shared mode — nothing to do for the routed side.
        Tensor out(T, d_model_);
        tensor_add(last_shared_out, last_routed_out, out);
        last_load_balance_loss_ = 0.0;
        return out;
    }

    // Gate scores: s = sigmoid(W_g · x), shape (T, num_routed).
    last_gate_logits = W_g_.forward(input);                 // (T, num_routed)
    last_gate_scores = Tensor(T, num_routed_);
    for (size_t i = 0; i < last_gate_logits.data.size(); ++i)
        last_gate_scores.data[i] = sigmoid(last_gate_logits.data[i]);

    // Top-k selection + renormalization.
    last_top_k_indices = Tensor(T, top_k_routed_);
    last_top_k_weights = Tensor(T, top_k_routed_);

    // For load balance: per-expert selection fraction and per-expert mean prob.
    std::vector<double> sel_count(num_routed_, 0.0);    // per-expert, total selections across tokens
    std::vector<double> sum_prob (num_routed_, 0.0);     // per-expert, sum of gate probs over tokens

    for (size_t b = 0; b < T; ++b) {
        // Find top-k indices for this token.
        // Since `top_k_routed <= num_routed`, we just sort.
        std::vector<std::pair<double, size_t>> order;
        order.reserve(num_routed_);
        for (size_t i = 0; i < num_routed_; ++i)
            order.emplace_back(last_gate_scores[b][i], i);
        std::partial_sort(order.begin(), order.begin() + top_k_routed_, order.end(),
                          [](const std::pair<double, size_t>& a,
                             const std::pair<double, size_t>& b) {
                              return a.first > b.first;
                          });

        double sum_top = 0.0;
        for (size_t k = 0; k < top_k_routed_; ++k) sum_top += order[k].first;
        // Guard against degenerate case (shouldn't happen with sigmoid but be safe).
        if (sum_top < 1e-30) sum_top = 1e-30;

        for (size_t k = 0; k < top_k_routed_; ++k) {
            size_t ei = order[k].second;
            last_top_k_indices[b][k] = static_cast<double>(ei);
            last_top_k_weights[b][k] = order[k].first / sum_top;
            sel_count[ei] += 1.0;
        }
    }
    for (size_t e = 0; e < num_routed_; ++e) sum_prob[e] = last_gate_scores.data.size() > 0
        ? 0.0 : 0.0;
    for (size_t e = 0; e < num_routed_; ++e) {
        double s = 0.0;
        for (size_t b = 0; b < T; ++b) s += last_gate_scores[b][e];
        sum_prob[e] = s;
    }

    // Load-balance auxiliary loss: α · E · Σ_e (f_e · p_e).
    double total_selections = static_cast<double>(T * top_k_routed_);
    if (total_selections < 1e-9) total_selections = 1e-9;
    double lbl = 0.0;
    for (size_t e = 0; e < num_routed_; ++e) {
        double f_e = sel_count[e] / total_selections;
        double p_e = (T > 0 ? sum_prob[e] / static_cast<double>(T) : 0.0);
        lbl += f_e * p_e;
    }
    last_load_balance_loss_ = static_cast<double>(num_routed_) * lbl;

    // Run routed experts: each selected expert `e` operates on segment
    // `x_seg = input[:, e*seg_size : (e+1)*seg_size]` of shape (T, seg_size).
    //
    // For correctness in the backward, we evaluate each (token, selection)
    // pair exactly once — the gradient backpropagates through the same path.
    //
    // To keep the implementation simple and avoid per-expert token dispatch
    // tables, we run a 2-layer FFN over the full (T, seg_size) input segment
    // for each selected expert, multiply by the gate weight, and add to the
    // routed output at the segment positions.
    //
    // When top_k_routed == num_routed (deterministic-all-firing), this means
    // num_routed FFN calls (each on full batch). For test configs (small),
    // this is fine.

    // Forward caches for backward (per-expert intermediates).
    // For each routed expert e, we cache:
    //   last_per_expert_z[e]   — (T, d_expert)   W1 · x_seg (pre-SiLU)
    //   last_per_expert_h[e]   — (T, d_expert)   SiLU(W1 · x_seg)
    //   last_per_expert_out[e] — (T, seg_size)   W2 · SiLU(W1 · x_seg)
    // Reuse last_per_expert_z, last_per_expert_h, last_per_expert_seg from
    // header — but the header only declares them as single Tensors (flat),
    // not per-expert. To avoid shape mismatch we use Tensor::zeros of the
    // right shape and stash pointers via simple std::vector<Tensor> on the
    // local stack.

    // We use std::vector<std::vector<Tensor>> nested caches per selected
    // (token, expert), but for clean per-expert caches, we materialize
    // per-expert tensors below. The header's `last_per_expert_*` slots are
    // placeholders; we instead use local std::vector<Tensor> in forward and
    // store them as members by re-using the routing cache structure.

    // Actually, the header has only single Tensor slots — let me extend it
    // to per-expert vectors via a simple approach: forward returns the
    // result, and backward re-derives the inputs via forward-no-grad? No,
    // we need caches. Use std::vector<Tensor> fields. Re-shape header.

    // (See updated header below; for now, the .cpp uses vector caches.)
    // NOTE: To keep this self-contained, we declare static-sized caches in
    // the .cpp via a struct trick: use the header's single slots ONLY when
    // num_routed_== 1 AND top_k_routed_== 1. Otherwise, allocate via
    // static-thread-local — NO. The simplest fix is to update the header.
    //
    // Pragmatic choice: store the caches INSIDE forward via static locals —
    // BAD. Instead, allocate a dynamic dispatch table as a thread_local —
    // BAD. The cleanest path: use the existing single Tensor slots as a
    // FLATTENED (T*top_k, d_expert) buffer. Since each expert handles only
    // ITS segment, we can flatten over the routed path: row b*top_k + k
    // corresponds to expert index last_top_k_indices[b][k], token b. The
    // W1/W2 of expert e only ever sees segment inputs from tokens for which
    // e ∈ top-k. Re-evaluating those slices by collecting is also possible
    // but messy.
    //
    // The cleanest approach is: use the header slots as PER-EXPERT (one
    // forward pass per expert), each with shape (T_active, ...) where T_active
    // is the number of tokens that selected this expert. To keep this simple,
    // we ALWAYS run every expert on every token (even those not in top-k for
    // that expert), multiplying the output by the gate weight of zero for
    // non-selected slots. This gives bit-exact correspondence between forward
    // and backward (every expert runs on every token's segment) and avoids
    // the dispatch table entirely.
    //
    // This is the convention we'll use. Note that this is computationally
    // wasteful in the general case but correct, and matches the "all experts
    // fire on all tokens" view which is mathematically equivalent to top-k
    // routing when gate weights for non-selected experts are zero.

    // For each routed expert, run forward on the FULL segment input (T rows)
    // — i.e., we evaluate each expert on every token's segment. Multiply the
    // resulting (T, seg_size) output by the gate weight for that expert (per
    // token), and add to last_routed_out at the segment positions.

    for (size_t e = 0; e < num_routed_; ++e) {
        // Extract the segment for expert e: input[:, e*seg_size:(e+1)*seg_size].
        Tensor x_seg(T, seg_size_);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < seg_size_; ++j)
                x_seg[i][j] = input[i][e * seg_size_ + j];

        // W1 · x_seg : (T, d_expert) — use Dense::forward so it caches last_input.
        Tensor z = experts_[e].W1.forward(x_seg);   // (T, d_expert)
        // SiLU in place into a new tensor.
        Tensor h(T, d_expert_);
        for (size_t i = 0; i < z.data.size(); ++i) h.data[i] = silu(z.data[i]);
        // W2 · h : (T, seg_size)
        Tensor out = experts_[e].W2.forward(h);

        // Weight by gate probability s and add to routed_out at segment cols.
        // Note: we use the un-renormalized gate probability (sigma(W_g · x))
        // here — the renormalization is a post-scaling applied on top of the
        // weighted sum. This matches the paper's forward path:
        //   y_routed = sum_{i in top-k} (s_i / sum_{j in top-k} s_j) · FFN_i(x_seg_i)
        // We compute (s_i * FFN_i(...)) for each (token, expert) pair where
        // the expert is in top-k for that token, then divide by sum_top at
        // the end. To keep the code simple and exactly match the per-token
        // renormalization, we use the precomputed last_top_k_weights directly.
        for (size_t i = 0; i < T; ++i) {
            // Find the gate weight for expert e in token i (if it's in top-k).
            double w_e = 0.0;
            for (size_t k = 0; k < top_k_routed_; ++k) {
                if (static_cast<size_t>(last_top_k_indices[i][k]) == e) {
                    w_e = last_top_k_weights[i][k];
                    break;
                }
            }
            if (w_e == 0.0) continue;     // not selected for this token
            for (size_t j = 0; j < seg_size_; ++j) {
                last_routed_out[i][e * seg_size_ + j] += w_e * out[i][j];
            }
        }
    }

    // Final output = shared + routed.
    Tensor out(T, d_model_);
    tensor_add(last_shared_out, last_routed_out, out);
    return out;
}

Tensor DeepSeekMoELayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (last_input.data.empty())
        throw std::logic_error("DeepSeekMoELayer::backward: forward was not called");
    if (grad_output.cols != d_model_)
        throw std::invalid_argument("DeepSeekMoELayer::backward: grad_output dim mismatch");
    const size_t T = last_input.rows;

    // grad_input starts at zero — accumulate from shared + routed paths.
    Tensor grad_input(T, d_model_);
    std::fill(grad_input.data.begin(), grad_input.data.end(), 0.0);

    // --- Shared path backward ---
    // shared_out = sum_j W2_j · SiLU(W1_j · x)
    // grad from grad_output flows through each shared FFN.
    for (size_t j = 0; j < num_shared_; ++j) {
        // W2_j is shape (d_model, d_expert). Its backward expects grad of
        // (T, d_model) and returns grad of (T, d_expert).
        Tensor grad_h_pre = shared_experts_[j].W2.backward(grad_output, 0.0);   // (T, d_expert)
        // SiLU derivative. We need to re-derive the pre-SiLU activations —
        // simplest: re-run forward for shared expert j.
        Tensor z(T, d_expert_);
        for (size_t i = 0; i < T; ++i)
            for (size_t k = 0; k < d_expert_; ++k)
                z[i][k] = 0.0;
        // Recompute z = W1_j · x + b1_j  using the cached W1_j weights/bias.
        // We do this manually since Dense::backward would also accumulate
        // grad_weights/grad_bias and we want to chain through it.
        // Approach: call W1_j.backward(grad_z, 0.0) to get grad_x — but we
        // need grad_z. So compute z directly here.
        const Tensor& W1 = shared_experts_[j].W1.weights;
        const Tensor& b1 = shared_experts_[j].W1.bias;
        for (size_t i = 0; i < T; ++i) {
            for (size_t k = 0; k < d_expert_; ++k) {
                double s = b1[0][k];
                for (size_t j2 = 0; j2 < d_model_; ++j2) {
                    s += W1[k][j2] * last_input[i][j2];
                }
                z[i][k] = s;
            }
        }
        // grad through SiLU
        Tensor grad_z(T, d_expert_);
        for (size_t i = 0; i < grad_z.data.size(); ++i)
            grad_z.data[i] = grad_h_pre.data[i] * silu_deriv(z.data[i]);
        // W1_j backward: input was last_input; expect grad of (T, d_model).
        Tensor grad_x_via = shared_experts_[j].W1.backward(grad_z, 0.0);
        for (size_t i = 0; i < grad_input.data.size(); ++i)
            grad_input.data[i] += grad_x_via.data[i];
    }

    // --- Routed path backward (only when there are routed experts) ---
    if (num_routed_ > 0) {
        // For each routed expert e, run a 2-layer FFN backward gated by the
        // gate weight (which depends on W_g — see gate gradient below).
        //
        // Per-token, per-expert FFN gradient through W2 and W1:
        //
        //   h   = SiLU(z),   z = W1 · x_seg + b1
        //   out = W2 · h + b2
        //   contribution to y_routed at token i, segment e = w_e · out
        //
        //   grad_out_seg[i, j] = grad_output[i, e*seg + j] * w_e
        //
        // The W2 backward accumulates a (T, d_expert) gradient weighted by w_e.
        // But since we ran W2 on the FULL token batch, we need to weight each
        // token's gradient by w_e (zero for tokens where e ∉ top-k).
        //
        // Implementation: build per-token weighted grad_seg and pass through
        // W2.backward() with a custom accumulator.

        // We need a "fake" grad_output for W2 that has shape (T, seg_size)
        // and reflects per-token weighting.
        for (size_t e = 0; e < num_routed_; ++e) {
            // Build per-token gate weight w_e (0 if not in top-k).
            Tensor w(T, 1);
            for (size_t i = 0; i < T; ++i) {
                double w_e = 0.0;
                for (size_t k = 0; k < top_k_routed_; ++k) {
                    if (static_cast<size_t>(last_top_k_indices[i][k]) == e) {
                        w_e = last_top_k_weights[i][k];
                        break;
                    }
                }
                w[i][0] = w_e;
            }
            // grad_seg_e : (T, seg_size) = grad_output[:, e*seg:(e+1)*seg] ⊙ w
            Tensor grad_seg(T, seg_size_);
            for (size_t i = 0; i < T; ++i) {
                double w_e = w[i][0];
                for (size_t j = 0; j < seg_size_; ++j) {
                    grad_seg[i][j] = grad_output[i][e * seg_size_ + j] * w_e;
                }
            }
            // Backward through W2 (cached forward = (T, seg_size) = W2 · h)
            // W2 has cached last_input = h (shape (T, d_expert)).
            Tensor grad_h_pre = experts_[e].W2.backward(grad_seg, 0.0);   // (T, d_expert)

            // Re-derive z = W1 · x_seg + b1 for SiLU backward.
            Tensor x_seg(T, seg_size_);
            for (size_t i = 0; i < T; ++i)
                for (size_t j = 0; j < seg_size_; ++j)
                    x_seg[i][j] = last_input[i][e * seg_size_ + j];

            Tensor z(T, d_expert_);
            const Tensor& W1 = experts_[e].W1.weights;
            const Tensor& b1 = experts_[e].W1.bias;
            for (size_t i = 0; i < T; ++i) {
                for (size_t k = 0; k < d_expert_; ++k) {
                    double s = b1[0][k];
                    for (size_t j = 0; j < seg_size_; ++j)
                        s += W1[k][j] * x_seg[i][j];
                    z[i][k] = s;
                }
            }
            // SiLU backward
            Tensor grad_z(T, d_expert_);
            for (size_t i = 0; i < grad_z.data.size(); ++i)
                grad_z.data[i] = grad_h_pre.data[i] * silu_deriv(z.data[i]);
            // W1 backward — accumulates grad_weights/grad_bias for W1 and
            // returns grad_x_seg.
            Tensor grad_x_seg = experts_[e].W1.backward(grad_z, 0.0);
            // Add to grad_input at segment cols.
            for (size_t i = 0; i < T; ++i) {
                for (size_t j = 0; j < seg_size_; ++j) {
                    grad_input[i][e * seg_size_ + j] += grad_x_seg[i][j];
                }
            }
        }

        // --- Gate gradient (W_g) ---
        // The gate weights w_e for token i are renormalized:
        //   w_e = s_e / Σ_{j in top-k(i)} s_j
        // Let S_t = Σ_{j in top-k(i)} s_j (per token, scalar).
        //
        // The output at token i, segment e (for selected e):
        //   y_routed[i, e*seg:(e+1)*seg] = w_e · FFN_e(x_seg_e)
        //
        // So d(y)/d(s_e) = w_e · FFN_e(x_seg_e)  (for selected e)
        // and d(y)/d(s_j) for j ≠ e, j in top-k:
        //   = (d y / d w_e) · (d w_e / d s_j)
        //   = FFN_e(x_seg_e) · (-s_e / S_t^2)
        //
        // We treat gate-score gradients as a (T, num_routed) tensor.
        // grad_gate_scores[i, e] = sum over selected segments that expert e
        // contributes to, plus the renormalization coupling term.
        //
        // Then grad_W_g = grad_gate_scores ⊙ sigmoid'(z_gate) · x^T  via the
        // standard linear backward through W_g.

        Tensor grad_gate_scores(T, num_routed_);
        std::fill(grad_gate_scores.data.begin(), grad_gate_scores.data.end(), 0.0);

        // For each token, compute:
        //   contribution to grad_gate_scores[i, e] from the FFN output:
        //     if e in top-k:  sum_j ( grad_output[i, e*seg+j] * (FFN_e(x_seg)[i, j] / S_t) )
        //                    - sum_{k in top-k} ( grad_output[i, k*seg+j] * s_k * FFN_k_out[i,j] / S_t^2 )
        //
        // In compact form, let g_seg[i, e_seg_idx] = grad_output[i, e*seg+e_seg_idx]
        // (zero for non-selected experts), and let out_e_seg[i, j] = FFN_e(x_seg_e)[i, j]
        // (only available for selected experts — but we can re-derive or use
        // the fact that last_routed_out contains the weighted version).
        //
        // Easiest path: re-derive the per-expert FFN output forward (no grad)
        // for each selected expert per token. This costs another forward but
        // is correct and simple.

        // For each (token, top-k slot), we already know expert index e and
        // gate weight w_e. We re-run the FFN for that expert on the segment
        // to get out_e_seg. We also need s_e (= w_e * S_t). Since w_e and
        // S_t are derived from top-k selection, we can recover:
        //   s_e = w_e * S_t   (with S_t = sum over top-k scores)

        for (size_t i = 0; i < T; ++i) {
            // Recover S_t = sum of top-k gate scores for token i.
            double S_t = 0.0;
            for (size_t k = 0; k < top_k_routed_; ++k) {
                size_t e = static_cast<size_t>(last_top_k_indices[i][k]);
                S_t += last_gate_scores[i][e];
            }
            if (S_t < 1e-30) S_t = 1e-30;

            // Compute the per-expert raw output (FFN_e(x_seg_e)) for each
            // expert e in this token's top-k. Re-derive via forward.
            std::vector<Tensor> ffn_out(num_routed_);
            for (size_t k = 0; k < top_k_routed_; ++k) {
                size_t e = static_cast<size_t>(last_top_k_indices[i][k]);
                // x_seg_e = input[i, e*seg:(e+1)*seg] (1, seg_size)
                Tensor x_seg(1, seg_size_);
                for (size_t j = 0; j < seg_size_; ++j)
                    x_seg[0][j] = last_input[i][e * seg_size_ + j];
                // z = W1 · x_seg + b1
                Tensor z(1, d_expert_);
                const Tensor& W1 = experts_[e].W1.weights;
                const Tensor& b1 = experts_[e].W1.bias;
                for (size_t kk = 0; kk < d_expert_; ++kk) {
                    double s = b1[0][kk];
                    for (size_t jj = 0; jj < seg_size_; ++jj)
                        s += W1[kk][jj] * x_seg[0][jj];
                    z[0][kk] = s;
                }
                // h = SiLU(z)
                Tensor h(1, d_expert_);
                for (size_t kk = 0; kk < d_expert_; ++kk)
                    h[0][kk] = silu(z[0][kk]);
                // out = W2 · h + b2
                Tensor out(1, seg_size_);
                const Tensor& W2 = experts_[e].W2.weights;
                const Tensor& b2 = experts_[e].W2.bias;
                for (size_t jj = 0; jj < seg_size_; ++jj) {
                    double s = b2[0][jj];
                    for (size_t kk = 0; kk < d_expert_; ++kk)
                        s += W2[jj][kk] * h[0][kk];
                    out[0][jj] = s;
                }
                ffn_out[e] = out;
            }

            // Combined formula (derivation in comment below):
            //   grad_gate_scores[i, e] = (S_t · dot_e - sum_s_dot) / S_t^2
            //
            // d(y[i])/d(s_e) for e in top-k:
            //   = sum_{f in top-k} ∂w_f/∂s_e · <grad_output[i, f*seg:], FFN_f(x_seg_f)[i, :]>
            // where ∂w_f/∂s_e = (δ_{ef} · S_t - s_f) / S_t^2.
            //   = (S_t - s_e)/S_t^2 · dot_e + sum_{f != e} (-s_f/S_t^2) · dot_f
            //   = (S_t · dot_e - s_e · dot_e - sum_{f != e} s_f · dot_f) / S_t^2
            //   = (S_t · dot_e - sum_s_dot) / S_t^2
            //
            // where sum_s_dot = sum_{f in top-k} s_f · dot_f.

            // Pre-compute: sum_f s_f · FFN_f_dot_grad  (a single scalar per token).
            double sum_s_dot = 0.0;
            for (size_t k = 0; k < top_k_routed_; ++k) {
                size_t f = static_cast<size_t>(last_top_k_indices[i][k]);
                double s_f = last_gate_scores[i][f];
                double dot = 0.0;
                for (size_t j = 0; j < seg_size_; ++j)
                    dot += grad_output[i][f * seg_size_ + j] * ffn_out[f][0][j];
                sum_s_dot += s_f * dot;
            }
            double inv_S2 = 1.0 / (S_t * S_t);

            for (size_t k = 0; k < top_k_routed_; ++k) {
                size_t e = static_cast<size_t>(last_top_k_indices[i][k]);
                double s_e = last_gate_scores[i][e];
                double dot_e = 0.0;
                for (size_t j = 0; j < seg_size_; ++j)
                    dot_e += grad_output[i][e * seg_size_ + j] * ffn_out[e][0][j];
                // Combined formula:
                //   grad_gate_scores[i, e] = (S_t · dot_e - sum_s_dot) / S_t^2
                grad_gate_scores[i][e] += (S_t * dot_e - sum_s_dot) * inv_S2;
            }

            // Also: for non-selected experts f ∉ top-k, ∂w_f/∂s_e = 0, so no
            // contribution to grad_gate_scores from those slots (Term A is
            // also 0 since w_f = 0 means no FFN output contributes through
            // that path).
        }

        // Now: grad_gate_scores = grad_gate_scores ⊙ sigmoid'(z_gate)
        //   where z_gate = W_g · x.
        // sigmoid'(z) = sigma(z) * (1 - sigma(z))
        for (size_t i = 0; i < T; ++i) {
            for (size_t e = 0; e < num_routed_; ++e) {
                double s = last_gate_scores[i][e];
                double dsig = s * (1.0 - s);
                grad_gate_scores[i][e] *= dsig;
            }
        }

        // grad_W_g (shape (num_routed, d_model)) = grad_gate_scores^T · x
        // grad_b_g (shape (1, num_routed)) = sum over tokens of grad_gate_scores
        Tensor grad_W_g(num_routed_, d_model_);
        std::fill(grad_W_g.data.begin(), grad_W_g.data.end(), 0.0);
        Tensor grad_b_g(1, num_routed_);
        std::fill(grad_b_g.data.begin(), grad_b_g.data.end(), 0.0);
        for (size_t e = 0; e < num_routed_; ++e) {
            for (size_t j = 0; j < d_model_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < T; ++i) {
                    sum += grad_gate_scores[i][e] * last_input[i][j];
                }
                grad_W_g[e][j] = sum;
            }
        }
        for (size_t e = 0; e < num_routed_; ++e) {
            double sum = 0.0;
            for (size_t i = 0; i < T; ++i) sum += grad_gate_scores[i][e];
            grad_b_g[0][e] = sum;
        }

        // Accumulate into W_g_'s grad_weights / grad_bias (MUTATION HOOK
        // lives just below — search "MUTATION HOOK" to inject a stub here).
        for (size_t e = 0; e < num_routed_; ++e) {
            for (size_t j = 0; j < d_model_; ++j) {
                // MUTATION HOOK: this line is the gate-gradient accumulation
                // that test_mutation_w_g_grad_path can stub out (replace `+=`
                // with `=` or zero-fill) to prove the W_g FD test is
                // non-vacuous.
                W_g_.grad_weights[e][j] += grad_W_g[e][j];
            }
        }
        for (size_t e = 0; e < num_routed_; ++e) {
            W_g_.grad_bias[0][e] += grad_b_g[0][e];
        }

        // grad_x from the gate path = grad_gate_scores (pre-activation) · W_g
        // (this is the standard linear-layer backward through W_g).
        Tensor grad_x_gate(T, d_model_);
        std::fill(grad_x_gate.data.begin(), grad_x_gate.data.end(), 0.0);
        for (size_t i = 0; i < T; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double sum = 0.0;
                for (size_t e = 0; e < num_routed_; ++e) {
                    sum += grad_gate_scores[i][e] * W_g_.weights[e][j];
                }
                grad_x_gate[i][j] = sum;
            }
        }
        for (size_t i = 0; i < grad_input.data.size(); ++i)
            grad_input.data[i] += grad_x_gate.data[i];
    }

    return grad_input;
}

void DeepSeekMoELayer::update_weights(double learning_rate) {
    W_g_.update_weights(learning_rate);
    for (auto& e : experts_) {
        e.W1.update_weights(learning_rate);
        e.W2.update_weights(learning_rate);
    }
    for (auto& s : shared_experts_) {
        s.W1.update_weights(learning_rate);
        s.W2.update_weights(learning_rate);
    }
}

void DeepSeekMoELayer::zero_grad() {
    W_g_.zero_grad();
    for (auto& e : experts_) {
        e.W1.zero_grad();
        e.W2.zero_grad();
    }
    for (auto& s : shared_experts_) {
        s.W1.zero_grad();
        s.W2.zero_grad();
    }
}

std::vector<Tensor*> DeepSeekMoELayer::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_g_.weights);
    p.push_back(&W_g_.bias);
    for (auto& e : experts_) {
        p.push_back(&e.W1.weights);
        p.push_back(&e.W1.bias);
        p.push_back(&e.W2.weights);
        p.push_back(&e.W2.bias);
    }
    for (auto& s : shared_experts_) {
        p.push_back(&s.W1.weights);
        p.push_back(&s.W1.bias);
        p.push_back(&s.W2.weights);
        p.push_back(&s.W2.bias);
    }
    return p;
}

std::vector<Tensor*> DeepSeekMoELayer::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&W_g_.grad_weights);
    g.push_back(&W_g_.grad_bias);
    for (auto& e : experts_) {
        g.push_back(&e.W1.grad_weights);
        g.push_back(&e.W1.grad_bias);
        g.push_back(&e.W2.grad_weights);
        g.push_back(&e.W2.grad_bias);
    }
    for (auto& s : shared_experts_) {
        g.push_back(&s.W1.grad_weights);
        g.push_back(&s.W1.grad_bias);
        g.push_back(&s.W2.grad_weights);
        g.push_back(&s.W2.grad_bias);
    }
    return g;
}

void DeepSeekMoELayer::copy_params_from(const DeepSeekMoELayer& other) {
    if (d_model_ != other.d_model_ || d_expert_ != other.d_expert_ ||
        num_routed_ != other.num_routed_ || num_shared_ != other.num_shared_ ||
        top_k_routed_ != other.top_k_routed_) {
        throw std::invalid_argument("DeepSeekMoELayer::copy_params_from: shape mismatch");
    }
    W_g_.weights = other.W_g_.weights;
    W_g_.bias = other.W_g_.bias;
    for (size_t i = 0; i < num_routed_; ++i) {
        experts_[i].W1.weights = other.experts_[i].W1.weights;
        experts_[i].W1.bias = other.experts_[i].W1.bias;
        experts_[i].W2.weights = other.experts_[i].W2.weights;
        experts_[i].W2.bias = other.experts_[i].W2.bias;
    }
    for (size_t j = 0; j < num_shared_; ++j) {
        shared_experts_[j].W1.weights = other.shared_experts_[j].W1.weights;
        shared_experts_[j].W1.bias = other.shared_experts_[j].W1.bias;
        shared_experts_[j].W2.weights = other.shared_experts_[j].W2.weights;
        shared_experts_[j].W2.bias = other.shared_experts_[j].W2.bias;
    }
}

size_t DeepSeekMoELayer::count_parameters() const {
    size_t n = 0;
    n += W_g_.weights.data.size() + W_g_.bias.data.size();
    for (auto& e : experts_) {
        n += e.W1.weights.data.size() + e.W1.bias.data.size();
        n += e.W2.weights.data.size() + e.W2.bias.data.size();
    }
    for (auto& s : shared_experts_) {
        n += s.W1.weights.data.size() + s.W1.bias.data.size();
        n += s.W2.weights.data.size() + s.W2.bias.data.size();
    }
    return n;
}

// ============================================================================
// DeepSeekMoEModel
// ============================================================================

DeepSeekMoEModel::DeepSeekMoEModel(size_t input_dim, size_t d_model, size_t output_dim,
                                   size_t num_layers, size_t d_expert,
                                   size_t num_routed, size_t num_shared,
                                   size_t top_k_routed)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      d_expert_(d_expert),
      num_routed_(num_routed),
      num_shared_(num_shared),
      top_k_routed_(top_k_routed),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0) throw std::invalid_argument("DeepSeekMoEModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("DeepSeekMoEModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("DeepSeekMoEModel: output_dim must be > 0");
    if (num_layers == 0) throw std::invalid_argument("DeepSeekMoEModel: num_layers must be > 0");
    if (d_model % num_routed != 0 && num_routed > 0)
        throw std::invalid_argument("DeepSeekMoEModel: d_model must be divisible by num_routed");

    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(std::make_unique<DeepSeekMoELayer>(
            d_model, d_expert, num_routed, num_shared, top_k_routed));
    }
}

Tensor DeepSeekMoEModel::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("DeepSeekMoEModel::forward: input feature dim mismatch");
    Tensor x = embed_.forward(input);
    for (auto& blk : blocks_) {
        x = blk->forward(x);
    }
    x = final_ln_.forward(x);
    return classifier_.forward(x);
}

Tensor DeepSeekMoEModel::backward(const Tensor& grad_output, double lr) {
    Tensor g = classifier_.backward(grad_output, lr);
    g = final_ln_.backward(g, lr);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        g = (*it)->backward(g, lr);
    }
    return embed_.backward(g, lr);
}

void DeepSeekMoEModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void DeepSeekMoEModel::zero_grad() {
    embed_.zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> DeepSeekMoEModel::parameters() {
    std::vector<Tensor*> p;
    auto ep = embed_.parameters();
    p.insert(p.end(), ep.begin(), ep.end());
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fp = final_ln_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> DeepSeekMoEModel::gradients() {
    std::vector<Tensor*> g;
    auto eg = embed_.gradients();
    g.insert(g.end(), eg.begin(), eg.end());
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fg = final_ln_.gradients();
    g.insert(g.end(), fg.begin(), fg.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}