#include "sophia.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

Sophia::Sophia(double lr_, double b1, double b2, double eps, double rho_, int k, double wd)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps), rho(rho_),
      update_period(k), t(1), weight_decay(wd) {
    this->Optimizer::lr = lr_;
}

void Sophia::ensure_state(void* layer_ptr,
                          const std::vector<Tensor*>& params) {
    if (m_state_.find(layer_ptr) != m_state_.end()) return;

    std::vector<Tensor> mvec, hvec;
    mvec.reserve(params.size());
    hvec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols);
        Tensor h(p.rows, p.cols);
        m.fill(0.0);
        h.fill(0.0);
        mvec.push_back(std::move(m));
        hvec.push_back(std::move(h));
    }
    m_state_[layer_ptr] = std::move(mvec);
    h_state_[layer_ptr] = std::move(hvec);
}

void Sophia::set_hessian_estimates(void* layer_ptr,
                                   const std::vector<Tensor>& h_diag) {
    hessian_input_[layer_ptr] = h_diag;
    hessian_input_set_[layer_ptr] = true;
}

double Sophia::last_m_value(void* layer_ptr, size_t param_idx,
                            size_t r, size_t c) const {
    auto it = m_state_.find(layer_ptr);
    if (it == m_state_.end()) return 0.0;
    if (param_idx >= it->second.size()) return 0.0;
    return it->second[param_idx][r][c];
}

double Sophia::last_h_value(void* layer_ptr, size_t param_idx,
                            size_t r, size_t c) const {
    auto it = h_state_.find(layer_ptr);
    if (it == h_state_.end()) return 0.0;
    if (param_idx >= it->second.size()) return 0.0;
    return it->second[param_idx][r][c];
}

void Sophia::empirical_fisher(const Tensor& grad, Tensor& out) const {
    // Caller is expected to have constructed `out` with the same shape as `grad`.
    for (size_t i = 0; i < grad.rows; ++i) {
        for (size_t j = 0; j < grad.cols; ++j) {
            double g = grad[i][j];
            out[i][j] = g * g;
        }
    }
}

void Sophia::update_param(Tensor* param, Tensor* grad, Tensor& m, Tensor& h,
                          const Tensor* h_diag_src, double b1_corr) {
    for (size_t i = 0; i < param->rows; ++i) {
        for (size_t j = 0; j < param->cols; ++j) {
            double g = (*grad)[i][j];

            // m_t = β1 * m_{t-1} + (1 - β1) * g_t
            m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;

            // m̂_t = m_t / (1 - β1^t)
            double m_hat = m[i][j] / b1_corr;

            // h_diag at this step (external if provided, else empirical Fisher g ⊙ g)
            double h_d = h_diag_src ? (*h_diag_src)[i][j] : (g * g);

            // h_t = β2 * h_{t-1} + (1 - β2) * h_diag_t
            h[i][j] = beta2 * h[i][j] + (1.0 - beta2) * h_d;

            // update = clip(m̂_t / max(h_t, ε), -ρ, ρ)
            double divisor = std::max(h[i][j], epsilon);
            double raw = m_hat / divisor;
            if (raw > rho)        raw = rho;
            else if (raw < -rho)  raw = -rho;

            // θ -= lr * (update + wd * θ)
            double step = raw + weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * step;
        }
    }
}

void Sophia::step(Model& model) {
    double b1_corr = 1.0 - std::pow(beta1, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& m_vec = m_state_[layer_ptr];
        auto& h_vec = h_state_[layer_ptr];

        // Determine h_diag source per-layer: external if user supplied, else empirical Fisher.
        bool use_external = hessian_input_set_.count(layer_ptr) > 0;
        std::vector<Tensor> empirical_h;
        if (!use_external) {
            empirical_h.reserve(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                empirical_h.emplace_back(grads[i]->rows, grads[i]->cols);
                empirical_fisher(*grads[i], empirical_h.back());
            }
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Tensor* h_src = nullptr;
            if (use_external) {
                // The external input vector has one entry per parameter.
                h_src = &hessian_input_[layer_ptr][i];
            } else {
                h_src = &empirical_h[i];
            }
            update_param(params[i], grads[i], m_vec[i], h_vec[i],
                         h_src, b1_corr);
        }

        // Consume the externally-supplied Hessian (one-shot per call).
        hessian_input_set_.erase(layer_ptr);
        hessian_input_.erase(layer_ptr);

        layer->zero_grad();
    }

    t++;
}
