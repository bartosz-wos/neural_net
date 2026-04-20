#ifndef LABEL_SMOOTHING_H
#define LABEL_SMOOTHING_H

#include "../core/tensor.h"
#include <vector>

// Label smoothing cross-entropy loss.
// Replaces hard labels with soft targets: y_smooth = y * (1 - ε) + ε / K
// where K = num_classes, ε = smoothing parameter.
class LabelSmoothingCrossEntropy {
public:
    LabelSmoothingCrossEntropy(double smoothing = 0.1) : smoothing_(smoothing) {}

    // logits: (batch, num_classes), targets: (batch,) as class indices OR (batch, num_classes) as soft labels
    Tensor forward(const Tensor& logits, const Tensor& targets);
    Tensor backward(const Tensor& logits, const Tensor& targets);

    double get_smoothing() const { return smoothing_; }

private:
    double smoothing_;
    Tensor last_logits_;
    Tensor last_smooth_targets_;
};

#endif