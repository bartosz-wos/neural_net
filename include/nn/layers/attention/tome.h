#ifndef TOME_H
#define TOME_H

#include "../../core/layer.h"
#include <vector>
#include <utility>
#include <cstddef>

// ============================================================================
// Token Merging (ToMe) — Bolya, Fu, Dai, Zhang, Feichtenhofer, ICCV 2023
//   https://arxiv.org/abs/2210.09461
//
// Fast ViT inference primitive via bipartite soft token matching.
// Given X ∈ R^{N × d}, returns X' ∈ R^{N' × d} where N' = ⌊N · ratio⌋.
//
// Algorithm (paper §3.1, simplified for v1 — non-trainable):
//   1. n_keep = ⌊N · ratio⌋                                    output size
//   2. A = even indices of {0, …, N-1}                         ("a" side, |A| = ⌈N/2⌉)
//      B = odd  indices of {0, …, N-1}                         ("b" side, |B| = ⌊N/2⌋)
//   3. For each a ∈ A:
//        s_ab = (X_a · X_b) / (2·‖X_a‖·‖X_b‖ + eps)            normalised cosine
//        b*  = argmax_b s_ab                                   with smallest-b tie-break
//   4. For each matched (a_k, b*_k): X'_k = (X_{a_k} + X_{b*_k}) / 2
//      (when |A| > n_keep, keep the first n_keep matches by a-index — paper's
//       "iterative" convention; reduction is monotonic in original ordering).
//
// Backward (v1, fixed match): the match is treated as fixed during backward
// (matches ARE recomputed from X in forward, but the gradient doesn't flow
// back through the argmax — the standard STE-free convention used at inference
// and in the paper's reference implementation). The merge's only gradient
// contribution is the (1/2) scatter from X'_k back to X_{a_k} and X_{b*_k}.
//
// Parameters: NONE (v1 is non-trainable). v2 seam: add per-edge learnable
// temperature + STE on the matching edges for end-to-end training.
//
// Layer interface:
//   - forward(x)              → X' ∈ R^{N' × d}                 (Layer::forward)
//   - forward_with_unmerge(x) → (X', unmerge_map)               (paired API)
//   - backward(grad_y, lr)    → grad_x ∈ R^{N × d}              (Layer::backward)
//   - unmerge(grad_y)         → grad_x ∈ R^{N × d}              (conservative scatter)
// ============================================================================

class TokenMerging : public Layer {
public:
    explicit TokenMerging(double reduce_ratio = 0.5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "TokenMerging"; }

    // Accessors
    double ratio() const { return reduce_ratio_; }
    size_t n_out(size_t N) const;

    // Forward paired with an unmerge map (a_dst, b_dst) for each output row k.
    //   unmerge_map[2k]   = source a-index that contributed to X'_k
    //   unmerge_map[2k+1] = source b-index that contributed to X'_k
    Tensor forward_with_unmerge(const Tensor& input);

    // Unmerge grad_y into grad_x (conservative scatter: each source index
    // receives grad_y[k] / 2 if it contributed to output k, 0 otherwise).
    Tensor unmerge(const Tensor& grad_y);

    // Last match edges from forward / forward_with_unmerge:
    //   edges[k] = (a_k, b*_k) for the k-th output row.
    const std::vector<std::pair<size_t, size_t>>& last_match_edges() const {
        return last_edges_;
    }

    // Test-only: set match edges directly. Used by the FD input-gradient test
    // to fix the matches across perturbations (the v1 contract: matches are
    // computed at forward and treated as fixed during backward — without this
    // pinning, FD perturbs the input and flips the matches, computing FD at a
    // different operating point than analytical).
    void _set_match_edges_for_test(size_t N, size_t d,
                                    const std::vector<std::pair<size_t, size_t>>& edges) {
        last_N_ = N;
        last_d_ = d;
        last_N_out_ = std::min(edges.size(), n_out(N));
        last_edges_ = edges;
    }

private:
    double reduce_ratio_;
    size_t last_N_;
    size_t last_d_;
    size_t last_N_out_;
    std::vector<std::pair<size_t, size_t>> last_edges_;
};

#endif
