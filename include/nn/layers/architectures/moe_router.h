#ifndef MOE_ROUTER_H
#define MOE_ROUTER_H

#include "../../core/tensor.h"
#include <cstddef>
#include <vector>

// ============================================================================
// MoeRouter — pure-stateless top-k softmax routing + load-balance utility
// for Mixture-of-Experts layers.
//
// Today this math appears in two places in the repo, both as inline code:
//   1. include/nn/layers/architectures/sparse_moe.cpp::softmax_topk — proper
//      top-k softmax with stability (the canonical sparse MoE formulation,
//      Shazeer 2017 / Fedus 2022).
//   2. include/nn/layers/architectures/mixture_of_experts.cpp::SparseDispatcher
//      — older top-k with raw (non-softmax) scores.
//
// This class captures the FUNCTION that both call sites perform: given
// (B, E) gate logits and a k, return (B, E) gate probabilities with the
// top-k entries carrying the softmax-normalized weight and the rest set
// to zero, along with the per-expert dispatch and mean-gate statistics
// required for the standard Shazeer load-balance auxiliary loss.
//
// MoeRouter is NOT a Layer. It is a stateless utility — no parameters,
// no gradient buffers, no update/zero_grad cycle. MoE layers that adopt
// it call ::softmax_topk in their forward, cache the returned state, and
// call ::softmax_topk_backward in their backward.
// ============================================================================

// Snapshot of MoeRouter::softmax_topk — exactly what softmax_topk_backward
// needs to recompute the chain rule through the (top-k + max-stable softmax)
// operator. All fields are independent of the input logits object: callers
// may mutate or destroy the original tensor without affecting the state.
//
// Field semantics:
//   probs         (B, E)              soft top-k probabilities: probs.sum(axis=1)=1,
//                                    non-selected entries are exactly 0
//   topk_idx      (B, k)             indices (in [0, E)) of the k largest logits per
//                                    row, sorted descending by logit value
//   dispatch_frac (E,)                f_e = (# tokens for which e is in top-k) / B
//                                    sums to k (since each row contributes k selections)
//   mean_gate_prob (E,)               p_e = (1/B) * sum_b probs[b, e]
//                                    sums to k/E (each row contributes k of total probability 1)
//   maxv          (B,)                max of the k largest logits per row
//                                    (used in stability — see softmax_topk impl)
//   exps          (B, k)              exp(logits_topk[b,i] - maxv[b]) per row
//   logsumexp     (B,)                log(sum_i exps[b,i]) per row
//   num_experts    == E
//   k              == the constructor k
struct MoeRouterState {
    Tensor probs;
    std::vector<std::vector<size_t>> topk_idx;   // (B, k)
    std::vector<double> dispatch_frac;          // (E,)
    std::vector<double> mean_gate_prob;         // (E,)
    std::vector<double> maxv;                   // (B,)
    std::vector<std::vector<double>> exps;      // (B, k)
    std::vector<double> logsumexp;              // (B,)
    size_t num_experts = 0;
    size_t k = 0;
};

// Pure-stateless top-k softmax routing utility for Mixture-of-Experts layers.
//
// softmax_topk takes (B, E) gate LOGITS, computes (per-row) the k largest of
// those logits, applies a numerically-stable softmax over them, and writes
// g[b, e_selected] = softmax(...). Non-selected entries are 0.
//
// softmax_topk_backward computes dL/d(logits) given dL/d(probs). The chain
// rule for softmax in the top-k restricted space is:
//
//   For row b, let S = topk_idx[b] (size k).
//   Let s = Σ_{e in S} probs[b, e] * d_gate_probs[b, e]   (a row-scalar).
//   Then for e in S:     d_logits[b, e] = probs[b, e] * (d_gate_probs[b, e] - s)
//   For e not in S:      d_logits[b, e] = 0            (hard topk mask)
//
// (The "hard topk mask" here means the routing decision (S itself) is
// treated as a non-differentiable constant — which is the standard sparse
// MoE backward, since the per-row argmax is piecewise-constant in the
// logits. To get a "soft topk" backward one would add the
// probs * (1 - one_hot_topk) terms; we do not, to match sparse_moe.cpp.)
//
// load_balance_loss_unscaled computes the standard Shazeer-form auxiliary
// loss  E * Σ_e f_e * p_e. The user multiplies by α (aux_loss_coef) to
// match the Switch Transformer reference. The function is symmetric under
// permutation of expert indices.
class MoeRouter {
public:
    // Forward: numerically-stable top-k softmax of (B, E) logits.
    //   - logits:    (B, E) gate logits
    //   - k:         number of experts to route to per row (1 <= k <= E)
    // Returns:      MoeRouterState with all fields populated.
    // Throws:       std::invalid_argument if logits is empty (rows=0),
    //                if E == 0, if k == 0, or if k > E.
    static MoeRouterState softmax_topk(const Tensor& logits, size_t k);

    // Backward: dL/d(logits) given dL/d(probs).
    //   - d_gate_probs: (B, E) gradient of the loss w.r.t. softmax_topk output
    //   - state:        state returned from the matching softmax_topk call
    // Returns:         Tensor of shape (B, E) — gradient w.r.t. logits.
    // Throws:          std::invalid_argument if d_gate_probs.shape != state
    //                  (B, E) shape.
    static Tensor softmax_topk_backward(const Tensor& d_gate_probs,
                                         const MoeRouterState& state);

    // Shazeer 2017 form: E * Σ_e f_e * p_e. Caller multiplies by α.
    //   - dispatch_frac: (E,)  f_e per expert
    //   - mean_gate_prob: (E,)  p_e per expert
    // Returns:            double, the unscaled loss.
    // Throws:             std::invalid_argument if sizes mismatch or E == 0.
    static double load_balance_loss_unscaled(
        const std::vector<double>& dispatch_frac,
        const std::vector<double>& mean_gate_prob);
};

#endif // MOE_ROUTER_H
