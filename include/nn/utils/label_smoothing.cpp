#include "label_smoothing.h"
#include "../activations/activations.h"
#include <cmath>

Tensor LabelSmoothingCrossEntropy::forward(const Tensor& logits, const Tensor& targets) {
    last_logits_ = logits;
    size_t batch = logits.rows;
    size_t K = logits.cols;

    // Convert hard labels to one-hot soft targets if needed
    // targets may be (batch,) indices or (batch, K) soft labels
    bool hard_labels = (targets.cols == 1 || (targets.rows == batch && targets.cols != K));

    Tensor smooth_targets;
    if (hard_labels) {
        smooth_targets = Tensor(batch, K);
        for (size_t b = 0; b < batch; ++b) {
            size_t label = static_cast<size_t>(targets[b][0] + 0.5);
            for (size_t j = 0; j < K; ++j) {
                smooth_targets[b][j] = (j == label) ? (1.0 - smoothing_) : (smoothing_ / K);
            }
        }
    } else {
        smooth_targets = targets; // already soft
    }
    last_smooth_targets_ = smooth_targets;

    // Cross entropy with smooth targets: -sum y_smooth * log_softmax
    // Compute softmax
    Tensor probs = Softmax::softmax(logits);

    double loss = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < K; ++j) {
            if (probs[b][j] > 1e-15)
                loss -= smooth_targets[b][j] * std::log(probs[b][j]);
        }
    }
    return Tensor(1, 1, loss / batch);
}

Tensor LabelSmoothingCrossEntropy::backward(const Tensor& logits, const Tensor& targets) {
    (void)targets;
    size_t batch = logits.rows;
    size_t K = logits.cols;

    // Gradient: (softmax - smooth_targets) / batch
    Tensor probs = Softmax::softmax(logits);
    Tensor grad(batch, K);
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < K; ++j)
            grad[b][j] = (probs[b][j] - last_smooth_targets_[b][j]) / batch;

    return grad;
}