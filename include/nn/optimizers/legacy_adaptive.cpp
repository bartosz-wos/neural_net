#include "legacy_adaptive.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// AdaGrad
// =============================================================================

AdaGrad::AdaGrad(double lr, double eps, double wd)
    : lr(lr), epsilon(eps), weight_decay(wd) {}

void AdaGrad::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (sum_sq_state_.find(layer_ptr) != sum_sq_state_.end()) return;

    std::vector<Tensor> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor t(p.rows, p.cols);
        t.fill(0.0);
        vec.push_back(std::move(t));
    }
    sum_sq_state_[layer_ptr] = std::move(vec);
}

void AdaGrad::step(Model& model) {
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec = sum_sq_state_[layer_ptr];

        for (size_t pi = 0; pi < params.size(); ++pi) {
            Tensor* param = params[pi];
            Tensor* grad = grads[pi];
            Tensor& sum_sq = state_vec[pi];

            for (size_t i = 0; i < grad->rows; ++i) {
                for (size_t j = 0; j < grad->cols; ++j) {
                    double g = (*grad)[i][j];

                    // accumulate: sum_sq_t = sum_sq_{t-1} + g_t^2
                    sum_sq[i][j] += g * g;

                    // update: param -= lr * g / (sqrt(sum_sq) + eps)
                    double denom = std::sqrt(sum_sq[i][j]) + epsilon;
                    double update = g / denom;

                    if (weight_decay > 0.0) {
                        (*param)[i][j] -= lr * weight_decay * (*param)[i][j];
                    }
                    (*param)[i][j] -= lr * update;
                }
            }
        }

        layer->zero_grad();
    }
}

// =============================================================================
// AMSGrad
// =============================================================================

AMSGrad::AMSGrad(double lr, double b1, double b2, double eps, double wd)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps), t(1), weight_decay(wd) {}

void AMSGrad::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (m_state_.find(layer_ptr) != m_state_.end()) return;

    std::vector<Tensor> m_vec, v_vec, vhat_vec;
    m_vec.reserve(params.size());
    v_vec.reserve(params.size());
    vhat_vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols), v(p.rows, p.cols), vh(p.rows, p.cols);
        m.fill(0.0); v.fill(0.0); vh.fill(0.0);
        m_vec.push_back(std::move(m));
        v_vec.push_back(std::move(v));
        vhat_vec.push_back(std::move(vh));
    }
    m_state_[layer_ptr] = std::move(m_vec);
    v_state_[layer_ptr] = std::move(v_vec);
    vhat_state_[layer_ptr] = std::move(vhat_vec);
}

void AMSGrad::step(Model& model) {
    // AMSGrad intentionally does NOT bias-correct v_hat (Reddi 2018 §2)
    // — only m_hat is bias-corrected.
    double b1_c = 1.0 - std::pow(beta1, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& m_vec = m_state_[layer_ptr];
        auto& v_vec = v_state_[layer_ptr];
        auto& vhat_vec = vhat_state_[layer_ptr];

        for (size_t pi = 0; pi < params.size(); ++pi) {
            Tensor* param = params[pi];
            Tensor* grad = grads[pi];
            Tensor& m = m_vec[pi];
            Tensor& v = v_vec[pi];
            Tensor& vh = vhat_vec[pi];

            for (size_t i = 0; i < grad->rows; ++i) {
                for (size_t j = 0; j < grad->cols; ++j) {
                    double g = (*grad)[i][j];

                    m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;
                    v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g;

                    // KEY AMSGRAD STEP: v_hat_t = max(v_hat_{t-1}, v_t)
                    double v_new = v[i][j];
                    if (v_new > vh[i][j]) vh[i][j] = v_new;

                    // Use bias-corrected first moment m_hat
                    double m_hat = m[i][j] / b1_c;
                    // v_hat is NOT bias-corrected in AMSGrad (Reddi 2018)
                    double denom = std::sqrt(vh[i][j]) + epsilon;
                    double update = m_hat / denom;

                    if (weight_decay > 0.0) {
                        (*param)[i][j] -= lr * weight_decay * (*param)[i][j];
                    }
                    (*param)[i][j] -= lr * update;
                }
            }
        }

        layer->zero_grad();
    }
    t++;
}

// =============================================================================
// Nadam
// =============================================================================

Nadam::Nadam(double lr, double b1, double b2, double eps, double wd)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps), t(1), weight_decay(wd),
      prod_beta1_cum_(1.0) {}

void Nadam::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (m_state_.find(layer_ptr) != m_state_.end()) return;

    std::vector<Tensor> m_vec, v_vec;
    m_vec.reserve(params.size());
    v_vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols), v(p.rows, p.cols);
        m.fill(0.0); v.fill(0.0);
        m_vec.push_back(std::move(m));
        v_vec.push_back(std::move(v));
    }
    m_state_[layer_ptr] = std::move(m_vec);
    v_state_[layer_ptr] = std::move(v_vec);
}

void Nadam::step(Model& model) {
    // Update running product of (1 - beta1^k) for the bias correction denominator.
    // At step t, prod = (1-beta1)(1-beta1^2)...(1-beta1^t)
    prod_beta1_cum_ *= (1.0 - std::pow(beta1, t));
    double b2_c = 1.0 - std::pow(beta2, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& m_vec = m_state_[layer_ptr];
        auto& v_vec = v_state_[layer_ptr];

        for (size_t pi = 0; pi < params.size(); ++pi) {
            Tensor* param = params[pi];
            Tensor* grad = grads[pi];
            Tensor& m = m_vec[pi];
            Tensor& v = v_vec[pi];

            for (size_t i = 0; i < grad->rows; ++i) {
                for (size_t j = 0; j < grad->cols; ++j) {
                    double g = (*grad)[i][j];

                    // Standard Adam moment updates
                    m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;
                    v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g;

                    // Nesterov-lookahead first moment (schedule-free form):
                    //   m_bar = (beta1 * m_t + (1-beta1) * g_t) / (1 - prod(beta1, 1..t))
                    // = (beta1 / (1-prod)) * m_t + ((1-beta1) / (1-prod)) * g_t
                    double m_bar = (beta1 * m[i][j] + (1.0 - beta1) * g) / prod_beta1_cum_;
                    double v_hat = v[i][j] / b2_c;
                    double denom = std::sqrt(v_hat) + epsilon;
                    double update = m_bar / denom;

                    if (weight_decay > 0.0) {
                        (*param)[i][j] -= lr * weight_decay * (*param)[i][j];
                    }
                    (*param)[i][j] -= lr * update;
                }
            }
        }

        layer->zero_grad();
    }
    t++;
}

// =============================================================================
// Adamax
// =============================================================================

Adamax::Adamax(double lr, double b1, double b2, double eps, double wd)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps), t(1), weight_decay(wd) {}

void Adamax::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (m_state_.find(layer_ptr) != m_state_.end()) return;

    std::vector<Tensor> m_vec, u_vec;
    m_vec.reserve(params.size());
    u_vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols), u(p.rows, p.cols);
        m.fill(0.0); u.fill(0.0);
        m_vec.push_back(std::move(m));
        u_vec.push_back(std::move(u));
    }
    m_state_[layer_ptr] = std::move(m_vec);
    u_state_[layer_ptr] = std::move(u_vec);
}

void Adamax::step(Model& model) {
    double b1_c = 1.0 - std::pow(beta1, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& m_vec = m_state_[layer_ptr];
        auto& u_vec = u_state_[layer_ptr];

        for (size_t pi = 0; pi < params.size(); ++pi) {
            Tensor* param = params[pi];
            Tensor* grad = grads[pi];
            Tensor& m = m_vec[pi];
            Tensor& u = u_vec[pi];

            for (size_t i = 0; i < grad->rows; ++i) {
                for (size_t j = 0; j < grad->cols; ++j) {
                    double g = (*grad)[i][j];

                    // First moment: same as Adam
                    m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;

                    // L_inf norm: u_t = max(beta2 * u_{t-1}, |g_t|)
                    double abs_g = std::abs(g);
                    double u_prev = beta2 * u[i][j];
                    u[i][j] = (u_prev > abs_g) ? u_prev : abs_g;

                    // Update with bias-corrected m_hat and the L_inf denominator.
                    // No sqrt needed since u is already the L_inf norm.
                    double m_hat = m[i][j] / b1_c;
                    double update = m_hat / (u[i][j] + epsilon);

                    if (weight_decay > 0.0) {
                        (*param)[i][j] -= lr * weight_decay * (*param)[i][j];
                    }
                    (*param)[i][j] -= lr * update;
                }
            }
        }

        layer->zero_grad();
    }
    t++;
}

// =============================================================================
// AdaDelta
// =============================================================================

AdaDelta::AdaDelta(double rho, double eps, double wd)
    : rho(rho), epsilon(eps), weight_decay(wd) {}

void AdaDelta::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (eg2_state_.find(layer_ptr) != eg2_state_.end()) return;

    std::vector<Tensor> eg2_vec, ed_vec;
    eg2_vec.reserve(params.size());
    ed_vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor eg(p.rows, p.cols), ed(p.rows, p.cols);
        eg.fill(0.0); ed.fill(0.0);
        eg2_vec.push_back(std::move(eg));
        ed_vec.push_back(std::move(ed));
    }
    eg2_state_[layer_ptr] = std::move(eg2_vec);
    edelta2_state_[layer_ptr] = std::move(ed_vec);
}

void AdaDelta::step(Model& model) {
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& eg2_vec = eg2_state_[layer_ptr];
        auto& ed_vec = edelta2_state_[layer_ptr];

        for (size_t pi = 0; pi < params.size(); ++pi) {
            Tensor* param = params[pi];
            Tensor* grad = grads[pi];
            Tensor& eg2 = eg2_vec[pi];
            Tensor& ed2 = ed_vec[pi];

            for (size_t i = 0; i < grad->rows; ++i) {
                for (size_t j = 0; j < grad->cols; ++j) {
                    double g = (*grad)[i][j];

                    // E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
                    eg2[i][j] = rho * eg2[i][j] + (1.0 - rho) * g * g;

                    // RMS[g]_t = sqrt(E[g^2]_t + eps)
                    double rms_g = std::sqrt(eg2[i][j] + epsilon);
                    // RMS[delta]_{t-1} = sqrt(E[delta^2]_{t-1} + eps)
                    double rms_d = std::sqrt(ed2[i][j] + epsilon);

                    // Update step: update_t = -RMS[delta]_{t-1} / RMS[g]_t * g_t
                    double update = -(rms_d / rms_g) * g;

                    // E[delta^2]_t = rho * E[delta^2]_{t-1} + (1-rho) * update_t^2
                    ed2[i][j] = rho * ed2[i][j] + (1.0 - rho) * update * update;

                    if (weight_decay > 0.0) {
                        // AdaDelta-style decoupled weight decay (multiplicative shrinkage).
                        (*param)[i][j] *= (1.0 - weight_decay);
                    }
                    (*param)[i][j] += update;
                }
            }
        }

        layer->zero_grad();
    }
}