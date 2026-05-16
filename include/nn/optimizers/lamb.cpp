#include "lamb.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

LAMB::LAMB(double lr, double b1, double b2, double eps, double trust_gamma)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps),
      beta1_corr(1.0), beta2_corr(1.0),
      t(1), trust_ratio_gamma(trust_gamma) {}

double LAMB::param_norm(const Tensor* w) const {
    double norm_sq = 0.0;
    for (size_t r = 0; r < w->rows; ++r)
        for (size_t c = 0; c < w->cols; ++c)
            norm_sq += (*w)[r][c] * (*w)[r][c];
    return std::sqrt(norm_sq);
}

double LAMB::trust_ratio(double w_norm, double g_norm, double gamma) const {
    if (w_norm < 1e-12 || g_norm < 1e-12) return 1.0;
    double ratio = w_norm / g_norm;
    double lower = 1.0 / gamma;
    double upper = gamma;
    return std::max(lower, std::min(upper, ratio));
}

void LAMB::ensure_state(void* layer_ptr) {
    auto it = state_.find(layer_ptr);
    if (it != state_.end()) return;

    // We need the layer to get parameter dimensions.
    // Find it via model.layers iteration — caller ensures the layer is from model.
    // State will be populated in step() after we know the parameter shapes.
}

void LAMB::update_param(Tensor* param, Tensor* grad,
                        Tensor& m, Tensor& v,
                        double w_norm, double lr,
                        double epsilon, double trust_gamma) {
    // Compute gradient norm
    double g_norm = 0.0;
    for (size_t r = 0; r < grad->rows; ++r)
        for (size_t c = 0; c < grad->cols; ++c) {
            double g = (*grad)[r][c];
            g_norm += g * g;
        }
    g_norm = std::sqrt(g_norm);

    double r = trust_ratio(w_norm, g_norm, trust_gamma);

    // Bias corrections (computed in step())
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // First moment: m = beta1 * m + (1-beta1) * g
            m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;

            // Second moment: v = beta2 * v + (1-beta2) * g^2
            v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g;

            // Bias-corrected moments
            double m_hat = m[i][j] / b1_c;
            double v_hat = v[i][j] / b2_c;

            // LAMB update: w' = w - lr * r * m_hat / (sqrt(v_hat) + eps)
            (*param)[i][j] -= lr * r * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void LAMB::step(Model& model) {
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);
    beta1_corr = b1_c;
    beta2_corr = b2_c;

    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        // Initialize state if first time
        if (state_.find(layer_ptr) == state_.end()) {
            std::vector<std::pair<Tensor, Tensor>> vec;
            vec.reserve(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                Tensor& p = *params[i];
                Tensor m(p.rows, p.cols);
                m.fill(0.0);
                Tensor v(p.rows, p.cols);
                v.fill(0.0);
                vec.emplace_back(std::move(m), std::move(v));
            }
            state_[layer_ptr] = std::move(vec);
        }

        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];

            double w_norm = param_norm(param);

            update_param(param, grad,
                        state_vec[i].first,
                        state_vec[i].second,
                        w_norm, lr,
                        epsilon, trust_ratio_gamma);
        }

        layer->zero_grad();
    }

    t++;
}