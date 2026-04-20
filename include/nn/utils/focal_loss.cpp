#include "focal_loss.h"
#include <cmath>

Tensor FocalLoss::forward(const Tensor& logits, const Tensor& targets) {
    size_t batch = logits.rows;
    size_t K = logits.cols;

    last_probs_ = Softmax::softmax(logits);
    last_targets_ = targets;

    double loss = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        size_t label = static_cast<size_t>(targets[b][0] + 0.5);
        double p_t = last_probs_[b][label];

        // p_t clamped to avoid log(0) and numerical issues
        p_t = std::max(1e-7, std::min(p_t, 1.0 - 1e-7));

        double pt_gamma = std::pow(1.0 - p_t, gamma_);
        loss += -alpha_ * pt_gamma * std::log(p_t);
    }

    return Tensor(1, 1, loss / batch);
}

Tensor FocalLoss::backward(const Tensor& logits, const Tensor& targets) {
    (void)targets;
    size_t batch = logits.rows;
    size_t K = logits.cols;

    Tensor probs = Softmax::softmax(logits);

    // Compute p_t for each sample and class
    Tensor grad(batch, K);
    for (size_t b = 0; b < batch; ++b) {
        size_t label = static_cast<size_t>(last_targets_[b][0] + 0.5);
        double p_t = probs[b][label];
        p_t = std::max(1e-7, std::min(p_t, 1.0 - 1e-7));

        double pt_gamma = std::pow(1.0 - p_t, gamma_);
        double coef = alpha_ * pt_gamma / p_t; // -α * (1-pt)^γ / pt  (sign absorbed)

        for (size_t j = 0; j < K; ++j) {
            double indicator = (j == label) ? 1.0 : 0.0;
            // dFL/dlogit_j = α * γ * (1-pt)^(γ-1) * pt * (indicator - prob_j)
            grad[b][j] = coef * gamma_ * std::pow(1.0 - p_t, gamma_ - 1.0) * p_t * (indicator - probs[b][j]);
        }
    }

    return grad;
}