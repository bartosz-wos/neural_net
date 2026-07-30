#include "label_smoothing.h"
#include "../activations/activations.h"
#include <cmath>

// Build the smooth-target tensor from raw targets.
// PyTorch convention: y_smooth[j] = (j==label) ? (1 - ε + ε/K) : (ε/K)
// This sums to 1 across j, which is necessary for the gradient formula
// dL/dz_i = (softmax_i - y_smooth_i) / B to be correct (the chain rule through
// softmax gives (Σ y_smooth)*p_i - y_smooth_i, which simplifies to p_i - y_smooth_i
// only when the smooth target distribution sums to 1).
// targets may be (batch, 1) integer class indices or (batch, K) soft labels.
static Tensor build_smooth_targets(const Tensor& targets, size_t batch, size_t K, double smoothing) {
    Tensor smooth_targets;
    bool hard_labels = (targets.cols == 1 || (targets.rows == batch && targets.cols != K));
    if (hard_labels) {
        smooth_targets = Tensor(batch, K);
        double true_val = 1.0 - smoothing + smoothing / static_cast<double>(K);
        double off_val  = smoothing / static_cast<double>(K);
        for (size_t b = 0; b < batch; ++b) {
            size_t label = static_cast<size_t>(targets[b][0] + 0.5);
            for (size_t j = 0; j < K; ++j) {
                smooth_targets[b][j] = (j == label) ? true_val : off_val;
            }
        }
    } else {
        smooth_targets = targets; // already soft
    }
    return smooth_targets;
}

Tensor LabelSmoothingCrossEntropy::forward(const Tensor& logits, const Tensor& targets) {
    last_logits_ = logits;
    size_t batch = logits.rows;
    size_t K = logits.cols;

    Tensor smooth_targets = build_smooth_targets(targets, batch, K, smoothing_);
    last_smooth_targets_ = smooth_targets;

    // Cross entropy with smooth targets: -sum y_smooth * log_softmax
    Tensor probs = Softmax()(logits);

    double loss = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < K; ++j) {
            if (probs[b][j] > 1e-15)
                loss -= smooth_targets[b][j] * std::log(probs[b][j]);
        }
    }
    Tensor loss_tensor(1, 1);
    loss_tensor(0, 0) = loss / batch;
    return loss_tensor;
}

Tensor LabelSmoothingCrossEntropy::backward(const Tensor& logits, const Tensor& targets) {
    // BUGFIX (was reading last_smooth_targets_ from a prior forward() call, which crashed
    // when backward() was called without a prior forward() — and was inconsistent with
    // softmax_cross_entropy_grad which computes its gradient from logits+targets alone).
    // Now compute smooth targets from the targets tensor directly.
    //
    // Gradient: (softmax - y_smooth) / B  (valid when Σ y_smooth = 1; we now use the
    // PyTorch convention that ensures this).
    size_t batch = logits.rows;
    size_t K = logits.cols;

    Tensor probs = Softmax()(logits);
    Tensor smooth_targets = build_smooth_targets(targets, batch, K, smoothing_);
    Tensor grad(batch, K);
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < K; ++j)
            grad[b][j] = (probs[b][j] - smooth_targets[b][j]) / static_cast<double>(batch);

    return grad;
}