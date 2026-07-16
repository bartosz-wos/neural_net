#ifndef SEGMENTATION_LOSSES_H
#define SEGMENTATION_LOSSES_H

#include "../core/tensor.h"

// =============================================================================
// Segmentation Losses for Semantic Segmentation
// =============================================================================
//
// This header bundles four canonical segmentation losses used in medical
// imaging, U-Net, and general per-pixel classification pipelines:
//
//   1. Dice Loss            (Milletari et al. 2016, V-Net)
//   2. Tversky Loss         (Salehi et al. 2017)
//   3. Focal-Dice Loss      (Zhu et al. 2018)
//   4. Lovász-Hinge Loss    (Yu & Blaschko 2018)
//
// Convention for ALL losses in this header:
//   * Predictions are (N, C) tensors of probabilities in [0, 1]
//     (i.e. AFTER sigmoid / per-channel softmax).
//   * Targets are (N, C) tensors of {0, 1} masks (or probability masks —
//     the formulas generalize cleanly).
//   * Forward returns (1, 1) loss tensor, MEAN over batch.
//   * Backward returns gradient tensor of the same (N, C) shape as the
//     predictions.
//
// `eps` defaults to 1.0 throughout (literature standard for Dice / Tversky;
// guards against 0/0 at the empty-mask degenerate case). All losses assume
// the prediction tensor has rows >= 1 and cols >= 1 and raise std::invalid_argument
// on shape mismatch.


// =============================================================================
// Dice Loss (Milletari et al. 2016, V-Net)
// =============================================================================
// Per-batch-row Dice coefficient (using (N, C) layout, treating each cell
// (b, c) as a scalar "prediction" for that class):
//   D_bc = (2 * sum_n p_{bc,n} * t_{bc,n} + eps) / (sum_n p_{bc,n} + sum_n t_{bc,n} + eps)
// Loss per (batch, class) row: 1 - D_bc.
// Total loss returned: MEAN over (batch, class).
//
// Default eps = 1.0 — the literature default. We use the symmetric form
// (eps in numerator AND denominator), matching the segmentation-models-pytorch
// convention. Alpha/beta-free.
class DiceLoss {
public:
    explicit DiceLoss(double eps = 1.0) : eps_(eps) {}

    // Returns (1, 1) loss tensor.
    Tensor forward(const Tensor& pred, const Tensor& target);

    // Returns (N, C) gradient tensor — same shape as pred.
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_eps() const { return eps_; }

private:
    double eps_;
};


// =============================================================================
// Tversky Loss (Salehi et al. 2017)
// =============================================================================
// Tversky Index (generalizes Dice):
//   TI_bc = (sum_n p_n t_n + eps)
//         / (sum_n p_n t_n + α sum_n p_n (1-t_n) + β sum_n (1-p_n) t_n + eps)
// Loss per (b, c):  1 - TI_bc.
// α weights false positives (predicting 1 where target = 0),
// β weights false negatives (predicting 0 where target = 1).
// α = β = 0.5 recovers Dice (modulo the symmetric eps form).
class TverskyLoss {
public:
    TverskyLoss(double alpha = 0.5, double beta = 0.5, double eps = 1.0)
        : alpha_(alpha), beta_(beta), eps_(eps) {}

    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    double get_eps() const { return eps_; }

private:
    double alpha_, beta_, eps_;
};


// =============================================================================
// Focal-Dice Loss (Zhu et al. 2018)
// =============================================================================
// Combines Focal Modulation (Lin et al. 2017) with Dice/Tversky: per-row,
//   L[bc] = (1 - TI[bc]) * (1 - p_t[bc])^γ
// where TI is the standard Dice/Tversky index and p_t is the predicted
// probability of the TRUE class (p_t = p if t=1, else p_t = 1-p).
// γ = 0 reduces to plain Tversky/Dice. γ > 0 gives harder examples (low p_t)
// larger gradients. Default parameters match the original paper.
// We use the Tversky-style denominator so α / β trade off FP / FN.
class FocalDiceLoss {
public:
    FocalDiceLoss(double gamma = 1.0, double alpha = 0.5, double beta = 0.5, double eps = 1.0)
        : gamma_(gamma), alpha_(alpha), beta_(beta), eps_(eps) {}

    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_gamma() const { return gamma_; }
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    double get_eps() const { return eps_; }

private:
    double gamma_, alpha_, beta_, eps_;
};


// =============================================================================
// Lovász-Hinge Loss (Yu & Blaschko 2018)
// =============================================================================
// The convex extension of the hinge loss applied to the IoU surrogate.
// Operates on sigmoided predictions in [0, 1] and binary {0, 1} targets.
//
// Per-row algorithm:
//   1. errors[i] = (target[i] == 1) ? (1 - p[i]) : p[i]   (hinge pre-image)
//   2. Sort by descending errors → sorted_errors, order
//   3. gt_sorted[i] = target[order[i]]
//   4. cum_pos[i] = sum_{j<=i} gt_sorted[j],
//      cum_neg[i] = sum_{j<=i} (1 - gt_sorted[j])
//   5. jaccard_delta[i] = 1 - (cum_pos[i] - gt_sorted[i]) / (cum_pos[i] + cum_neg[i])
//   6. lovasz_grad[i]  = jaccard_delta[i] - jaccard_delta[i-1]  (with jd[-1] = 0)
//   7. row_loss = dot(sorted_errors, lovasz_grad)
//
// Backward propagates through the sort via the index permutation; the
// analytical gradient is d(row_loss)/dp[bc] = (1 - 2 * t[bc]) * lg[rank(b,c)]
// divided by (C * N) for the batch average.
class LovaszHingeLoss {
public:
    LovaszHingeLoss() {}

    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);
};


#endif
