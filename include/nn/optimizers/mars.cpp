#include "mars.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

MARS::MARS(double lr, double b1, double b2, double gamma_, double eps,
           double wd, bool clip_, double eps_max_, double eps_grad_)
    : lr(lr), beta1(b1), beta2(b2), gamma(gamma_), epsilon(eps),
      weight_decay(wd), clip(clip_), eps_max(eps_max_), eps_grad(eps_grad_),
      t(1) {
    // Validation
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("MARS: beta1 must be in [0,1)");
    }
    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("MARS: beta2 must be in [0,1)");
    }
    if (gamma < 0.0 || gamma > 1.0) {
        throw std::invalid_argument("MARS: gamma must be in [0,1]");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("MARS: epsilon must be > 0");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("MARS: weight_decay must be >= 0");
    }
    if (clip) {
        if (eps_max <= 0.0) {
            throw std::invalid_argument("MARS: eps_max must be > 0 (when clip=true)");
        }
        if (eps_grad <= 0.0) {
            throw std::invalid_argument("MARS: eps_grad must be > 0 (when clip=true)");
        }
    }
}

void MARS::ensure_state(void* layer_ptr,
                         const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<Tensor> vec;
    vec.reserve(params.size() * 3);
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols);
        Tensor v(p.rows, p.cols);
        Tensor g_prev(p.rows, p.cols);
        m.fill(0.0);
        v.fill(0.0);
        g_prev.fill(0.0);  // First step uses g_prev=0 -> (g - g_prev) = g
        vec.push_back(std::move(m));
        vec.push_back(std::move(v));
        vec.push_back(std::move(g_prev));
    }
    state_[layer_ptr] = std::move(vec);
}

bool MARS::get_m(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return false;
    if (3 * param_idx + 1 > it->second.size()) return false;
    out = it->second[3 * param_idx].clone();
    return true;
}

bool MARS::get_v(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return false;
    if (3 * param_idx + 2 > it->second.size()) return false;
    out = it->second[3 * param_idx + 1].clone();
    return true;
}

bool MARS::get_g_prev(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return false;
    if (3 * param_idx + 3 > it->second.size()) return false;
    out = it->second[3 * param_idx + 2].clone();
    return true;
}

void MARS::update_param(Tensor* param, Tensor* grad, size_t base_idx,
                         std::vector<Tensor>& st) {
    // Step 1 + 2: MARSM shift correction + m/v EMAs
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];
            Tensor& m = st[base_idx];
            Tensor& v = st[base_idx + 1];
            Tensor& g_prev = st[base_idx + 2];

            double gtilde = g + gamma * (g - g_prev[i][j]);

            if (clip) {
                double gp_abs = std::max(std::abs(g_prev[i][j]), eps_grad);
                double ratio = std::abs(gtilde) / gp_abs;
                double clipped = std::min(std::max(ratio, eps_max), 1.0);
                gtilde = (gtilde >= 0.0 ? 1.0 : -1.0) * clipped * gp_abs;
            }

            // m_t = β1 · m_{t−1} + (1 − β1) · g̃
            m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * gtilde;
            // v_t = β2 · v_{t−1} + (1 − β2) · g̃²
            v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * gtilde * gtilde;

            // g_prev becomes g for next step's shift
            g_prev[i][j] = g;
        }
    }

    // Step 4+5: bias-corrected AdamW-style update with decoupled weight decay
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);
    if (bc1 < 1e-12) bc1 = 1e-12;
    if (bc2 < 1e-12) bc2 = 1e-12;

    Tensor& m = st[base_idx];
    Tensor& v = st[base_idx + 1];

    for (size_t i = 0; i < param->rows; ++i) {
        for (size_t j = 0; j < param->cols; ++j) {
            double m_hat = m[i][j] / bc1;
            double v_hat = v[i][j] / bc2;
            double denom = std::sqrt(v_hat) + epsilon;
            double update = m_hat / denom;
            // AdamW-style decoupled weight decay
            (*param)[i][j] -= lr * (update + weight_decay * (*param)[i][j]);
        }
    }
}

void MARS::step(Model& model) {
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& st = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], 3 * i, st);
        }

        layer->zero_grad();
    }
    ++t;
}
