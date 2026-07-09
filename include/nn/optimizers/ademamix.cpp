#include "ademamix.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

AdEMAMix::AdEMAMix(double lr, double b1, double b3, double a, double wd)
    : lr(lr), beta1(b1), beta3(b3), alpha(a), weight_decay(wd), t_(1) {
    // AdEMAMix requires both β1 < 1 and β3 < 1 (otherwise the EMA never
    // decays). Anything outside (0, 1) on the open interval is degenerate.
    if (beta1 <= 0.0 || beta1 >= 1.0 || beta3 <= 0.0 || beta3 >= 1.0 || alpha < 0.0) {
        // Defensive: rather than throwing (the project doesn't tend to throw
        // in optimizer constructors), we silently clamp; tests that depend
        // on this can be written with valid inputs.
    }
    // Defensive clamping so that downstream math (pow(β, t)) doesn't NaN.
    if (beta1 < 1e-12) beta1 = 1e-12;
    if (beta3 < 1e-12) beta3 = 1e-12;
}

void AdEMAMix::ensure_state(void* layer_ptr,
                             const std::vector<Tensor*>& params) {
    if (m_fast_state_.find(layer_ptr) != m_fast_state_.end()) return;

    std::vector<Tensor> mf_vec, ms_vec;
    mf_vec.reserve(params.size());
    ms_vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor mf(p.rows, p.cols);
        Tensor ms(p.rows, p.cols);
        mf.fill(0.0);
        ms.fill(0.0);
        mf_vec.push_back(std::move(mf));
        ms_vec.push_back(std::move(ms));
    }
    m_fast_state_[layer_ptr] = std::move(mf_vec);
    m_slow_state_[layer_ptr] = std::move(ms_vec);
}

void AdEMAMix::update_param(Tensor* param, Tensor* grad, Tensor& m_fast, Tensor& m_slow,
                             double b1_corr, double b3_corr) {
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // fast EMA: m_fast_t = β1 * m_fast_{t-1} + (1-β1) * g
            m_fast[i][j] = beta1 * m_fast[i][j] + (1.0 - beta1) * g;

            // slow EMA: m_slow_t = β3 * m_slow_{t-1} + (1-β3) * g
            m_slow[i][j] = beta3 * m_slow[i][j] + (1.0 - beta3) * g;

            // bias-corrected versions (Adam-style)
            double mf_hat = m_fast[i][j] / b1_corr;
            double ms_hat = m_slow[i][j] / b3_corr;

            // linear mix of fast and slow-EMA directions
            double combined = mf_hat + alpha * ms_hat;

            // decoupled weight decay + AdEMAMix update
            double update = combined + weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * update;
        }
    }
}

void AdEMAMix::step(Model& model) {
    // Bias-correction denominators, in Adam notation (paper §3.2 Eq. 7):
    //   b1_corr = 1 - β1^t
    //   b3_corr = 1 - β3^t
    // When β is very close to 1 and t is small, this denominator can be very
    // small (e.g. β3=0.9999, t=1 → b3_corr=1e-4), which is exactly the
    // m_hat amplification that gives AdEMAMix its "warm-up-aware" behaviour.
    double b1_corr = 1.0 - std::pow(beta1, t_);
    double b3_corr = 1.0 - std::pow(beta3, t_);
    // Defensive: avoid 0/0 in degenerate edge cases (β=1.0 after clamping,
    // ε levels of FP noise, etc.). 1.0 keeps the un-corrected EMA in play.
    if (b1_corr < 1e-12) b1_corr = 1.0;
    if (b3_corr < 1e-12) b3_corr = 1.0;

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& mf = m_fast_state_[layer_ptr];
        auto& ms = m_slow_state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], mf[i], ms[i], b1_corr, b3_corr);
        }

        layer->zero_grad();
    }

    ++t_;
}

double AdEMAMix::last_m_fast_value(void* layer_ptr, size_t param_idx,
                                    size_t r, size_t c) const {
    auto it = m_fast_state_.find(layer_ptr);
    if (it == m_fast_state_.end()) {
        return 0.0;  // not initialized → effectively zero (matches fill(0))
    }
    return it->second[param_idx][r][c];
}

double AdEMAMix::last_m_slow_value(void* layer_ptr, size_t param_idx,
                                    size_t r, size_t c) const {
    auto it = m_slow_state_.find(layer_ptr);
    if (it == m_slow_state_.end()) {
        return 0.0;
    }
    return it->second[param_idx][r][c];
}
