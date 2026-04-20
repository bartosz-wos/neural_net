#include "focal_loss.h"
#include "../activations/activations.h"
#include <cmath>

Tensor FocalLoss::forward(const Tensor& logits, const Tensor& targets) {
    size_t batch = logits.rows;
    size_t K = logits.cols;
    (void)K;

    // Manually apply softmax (stable) — per-row normalization
    last_probs_ = Tensor(batch, K);
    for (size_t b = 0; b < batch; ++b) {
        double max_logit = logits[b][0];
        for (size_t j = 0; j < K; ++j)
            max_logit = std::max(max_logit, logits[b][j]);

        double sum_exp = 0.0;
        for (size_t j = 0; j < K; ++j) {
            last_probs_[b][j] = std::exp(logits[b][j] - max_logit);
            sum_exp += last_probs_[b][j];
        }
        for (size_t j = 0; j < K; ++j)
            last_probs_[b][j] /= sum_exp;
    }

    last_targets_ = targets;

    double loss = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        size_t label = static_cast<size_t>(targets[b][0] + 0.5);
        double p_t = last_probs_[b][label];
        p_t = std::max(1e-7, std::min(p_t, 1.0 - 1e-7));
        double pt_gamma = std::pow(1.0 - p_t, gamma_);
        loss += -alpha_ * pt_gamma * std::log(p_t);
    }

    Tensor result(1, 1);
    result[0][0] = loss / batch;
    return result;
}

Tensor FocalLoss::backward(const Tensor& logits, const Tensor& targets) {
    (void)targets;
    size_t batch = logits.rows;
    size_t K = logits.cols;

    // Recompute softmax for gradient — per-row normalization
    Tensor probs(batch, K);
    for (size_t b = 0; b < batch; ++b) {
        double max_logit = logits[b][0];
        for (size_t j = 0; j < K; ++j)
            max_logit = std::max(max_logit, logits[b][j]);

        double sum_exp = 0.0;
        for (size_t j = 0; j < K; ++j) {
            probs[b][j] = std::exp(logits[b][j] - max_logit);
            sum_exp += probs[b][j];
        }
        for (size_t j = 0; j < K; ++j)
            probs[b][j] /= sum_exp;
    }

    Tensor grad(batch, K);
    for (size_t b = 0; b < batch; ++b) {
        size_t label = static_cast<size_t>(last_targets_[b][0] + 0.5);
        double p_t = probs[b][label];
        p_t = std::max(1e-7, std::min(p_t, 1.0 - 1e-7));

        // Gradient derivation for FL = -α(1-p_t)^γ * log(p_t):
        //   dFL/dp_t = -α * [γ(1-p_t)^(γ-1)*(-log(p_t)) + (1-p_t)^γ * (-1/p_t)]
        //            = α(1-p_t)^(γ-1) * [γ*log(p_t) + 1 - p_t] / p_t
        // Using (indicator - p_j) to combine both cases:
        //   dFL/dz_j = dFL/dp_t * dp_t/dz_j = α(1-p_t)^(γ-1) * [γ*log(p_t) + 1 - p_t] / p_t * (indicator - p_j)
        double dFL_dpt = alpha_ * std::pow(1.0 - p_t, gamma_ - 1.0) * (gamma_ * std::log(p_t) + 1.0 - p_t) / p_t;
        for (size_t j = 0; j < K; ++j) {
            double indicator = (j == label) ? 1.0 : 0.0;
            grad[b][j] = dFL_dpt * (indicator - probs[b][j]) / batch;
        }
    }

    return grad;
}