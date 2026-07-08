#ifndef CONTRASTIVE_LOSSES_H
#define CONTRASTIVE_LOSSES_H

#include "../core/tensor.h"
#include <vector>

// =============================================================================
// InfoNCE / NT-Xent Contrastive Loss
// =============================================================================
// The contrastive learning workhorse behind SimCLR, CLIP, MoCo, etc.
// Reference papers:
//   - Chen et al. 2020 "A Simple Framework for Contrastive Learning of
//     Visual Representations" (SimCLR), arXiv:2002.05709
//   - Oord et al. 2018 "Representation Learning with Contrastive Predictive
//     Coding" (CPC, original InfoNCE), arXiv:1807.03748
//   - Radford et al. 2021 "Learning Transferable Visual Models From Natural
//     Language Supervision" (CLIP), arXiv:2103.00020
//
// Given a batch of N examples with their augmented views, we get 2N
// representations z_0, z_0', z_1, z_1', ..., z_{N-1}, z_{N-1}' (total 2N rows).
// The standard SimCLR convention: positive of z_i (i in 0..2N-1) is the
// augmented view, i.e. z_{i XOR 1}.
//
// For each pair (i, j) where j is i's positive, the InfoNCE loss is:
//
//     L_ij = -log( exp(s_ij / T) / sum_{k != i} exp(s_ik / T) )
//
// where s_ij = sim(z_i, z_j) is a similarity (cosine or dot) and T is
// the temperature. The full loss is the mean of L_ij over all 2N anchors.
//
// Numerically-stable form using log-sum-exp:
//
//     L_ij = -(s_ij / T) + logsumexp_{k != i}(s_ik / T)
//
// Backward:
//     Define p_ik = softmax_k(s_i. / T) with the i-th entry zeroed out
//     (since we exclude the diagonal). Then for the L_ij contribution:
//        dL_ij/ds_ij = (1/T) * (p_ij - 1)
//        dL_ij/ds_ik = (1/T) * p_ik   for k != i, k != j
//     For cosine sim s_ij = z_i . z_j / (||z_i|| * ||z_j||), the chain
//     rule gives:
//        ds_ij/dz_i = (z_j / (||z_i|| ||z_j||)) - s_ij * z_i / ||z_i||^2
//        ds_ij/dz_j = (z_i / (||z_i|| ||z_j||)) - s_ij * z_j / ||z_j||^2
//     For dot-product sim s_ij = z_i . z_j:
//        ds_ij/dz_i = z_j,  ds_ij/dz_j = z_i
//
// Public API:
//   - InfoNCELoss(temperature, normalize=true, eps=1e-8)
//   - forward(z) -> Tensor(1, 1)         (mean InfoNCE over all 2N anchors)
//   - backward(z) -> Tensor(2N, D)       (grad w.r.t. z, averaged over 2N)
//   - optional positive_indices_ override: if non-empty, treats
//     positive_indices_[i] as the positive index of row i (instead of XOR 1).
//     This makes the loss usable for triplet-style positive/negative mining
//     beyond the simple SimCLR 2N convention.
//
// Reference invariants:
//   - When all pairs are equally similar (e.g. random init), L approaches
//     log(2N - 1) since the denominator is dominated by the 2N-1 negatives.
//   - When the positive similarity dominates, L drops toward 0.
//   - The loss is always >= 0 (a single log, since p_ij in (0, 1]).

class InfoNCELoss {
public:
    // temperature must be > 0; constructor asserts.
    // normalize=true -> cosine similarity (L2-normalize rows before dot).
    // eps is the floor on the denominator for cosine sim ||z_i|| * ||z_j||.
    explicit InfoNCELoss(double temperature = 0.5,
                         bool normalize = true,
                         double eps = 1e-8)
        : temperature_(temperature), normalize_(normalize), eps_(eps) {
        if (temperature_ <= 0.0) {
            // Defensive: paper convention is positive temperature.
            temperature_ = 1e-9;
        }
    }

    // Forward: takes a (2N, D) tensor of embeddings, returns 1x1 loss.
    // Convention: the positive of row i (0..2N-1) is row (i XOR 1).
    // Use set_positive_indices() to override.
    Tensor forward(const Tensor& z);

    // Backward: returns a (2N, D) tensor of gradients dL/dz.
    Tensor backward(const Tensor& z);

    // Optional: explicitly set the positive index of each row. Must have
    // size 2N, with values in 0..2N-1 and positive_indices_[i] != i.
    void set_positive_indices(const std::vector<int>& idx) {
        positive_indices_ = idx;
    }

    void clear_positive_indices() { positive_indices_.clear(); }

    double get_temperature() const { return temperature_; }
    bool get_normalize() const { return normalize_; }
    double get_eps() const { return eps_; }

    // Accessors for tests (mostly)
    const Tensor& last_logits() const { return last_logits_; }  // (2N, 2N) sim / T
    const Tensor& last_normalized() const { return last_normalized_; }  // (2N, D)
    const Tensor& last_probs() const { return last_probs_; }  // (2N, 2N) p_ij softmax

private:
    double temperature_;
    bool normalize_;
    double eps_;

    // Optional override for positive pairs (size 2N). Empty means use XOR.
    std::vector<int> positive_indices_;

    // Cached state from last forward.
    Tensor last_logits_;       // (2N, 2N) = sim / T
    Tensor last_normalized_;   // (2N, D) = input (post-L2-norm if normalize=true)
    Tensor last_probs_;        // (2N, 2N) softmax over k != i with diag = 0
    int last_2N_ = 0;          // cache size for backward
    int last_D_ = 0;
};

#endif