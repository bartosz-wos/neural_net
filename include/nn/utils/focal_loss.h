#ifndef FOCAL_LOSS_H
#define FOCAL_LOSS_H

#include "../core/tensor.h"
#include "../activations/activations.h"

// Focal Loss for class-imbalanced classification (Lin et al. 2017).
// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
// γ (gamma): focusing parameter — down-weights easy examples
// α (alpha): class weighting balance factor
class FocalLoss {
public:
    FocalLoss(double gamma = 2.0, double alpha = 1.0)
        : gamma_(gamma), alpha_(alpha) {}

    // logits: (batch, num_classes), targets: (batch,) class indices
    Tensor forward(const Tensor& logits, const Tensor& targets);
    Tensor backward(const Tensor& logits, const Tensor& targets);

    double get_gamma() const { return gamma_; }
    double get_alpha() const { return alpha_; }

private:
    double gamma_; // focusing parameter
    double alpha_; // class weight
    Tensor last_probs_;
    Tensor last_targets_;
};

#endif