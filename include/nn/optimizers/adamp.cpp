#include "adamp.h"
#include "../core/layer.h"
#include "../core/model.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>

// ---- validation ----
void AdamP::validate(double lr_, double b1_, double b2_,
                     double eps_, double delta_, double wd_) {
    if (lr_ < 0.0)
        throw std::invalid_argument("AdamP: lr must be >= 0");
    if (b1_ < 0.0 || b1_ >= 1.0)
        throw std::invalid_argument("AdamP: beta1 must be in [0, 1)");
    if (b2_ < 0.0 || b2_ >= 1.0)
        throw std::invalid_argument("AdamP: beta2 must be in [0, 1)");
    if (eps_ <= 0.0)
        throw std::invalid_argument("AdamP: epsilon must be > 0");
    if (delta_ < -1.0 || delta_ > 1.0)
        throw std::invalid_argument("AdamP: delta must be in [-1, 1]");
    if (wd_ < 0.0)
        throw std::invalid_argument("AdamP: weight_decay must be >= 0");
}

// ---- constructor ----
AdamP::AdamP(double lr_, double b1_, double b2_,
             double eps_, double delta_, double wd_)
    : lr(lr_), beta1(b1_), beta2(b2_),
      epsilon(eps_), delta(delta_),
      weight_decay(wd_), t(1) {
    validate(lr_, b1_, b2_, eps_, delta_, wd_);
}

// ---- setters (with validation) ----
void AdamP::set_lr(double v) {
    validate(v, beta1, beta2, epsilon, delta, weight_decay);
    lr = v;
}
void AdamP::set_beta1(double v) {
    validate(lr, v, beta2, epsilon, delta, weight_decay);
    beta1 = v;
}
void AdamP::set_beta2(double v) {
    validate(lr, beta1, v, epsilon, delta, weight_decay);
    beta2 = v;
}
void AdamP::set_epsilon(double v) {
    validate(lr, beta1, beta2, v, delta, weight_decay);
    epsilon = v;
}
void AdamP::set_delta(double v) {
    validate(lr, beta1, beta2, epsilon, v, weight_decay);
    delta = v;
}
void AdamP::set_weight_decay(double v) {
    validate(lr, beta1, beta2, epsilon, delta, v);
    weight_decay = v;
}

// ---- state helpers ----
void AdamP::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<State> vec;
    vec.reserve(params.size());
    for (const auto* p : params) {
        State s;
        s.m = Tensor(p->rows, p->cols);
        s.v = Tensor(p->rows, p->cols);
        s.m.fill(0.0);
        s.v.fill(0.0);
        vec.push_back(std::move(s));
    }
    state_[layer_ptr] = std::move(vec);
}

bool AdamP::has_state(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return false;
    return param_idx < it->second.size();
}

bool AdamP::get_m(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].m;
    return true;
}

bool AdamP::get_v(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].v;
    return true;
}

// ---- per-parameter update (the AdamP kernel) ----
void AdamP::update_param(Tensor* param, Tensor* grad, State& st,
                         double lr, double b1, double b2,
                         double eps, double delta, double wd,
                         double b1_c, double b2_c) {
    const size_t R = grad->rows;
    const size_t C = grad->cols;

    // 1) Standard Adam m, v updates (per-element)
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            double g = (*grad)[i][j];
            st.m[i][j] = b1 * st.m[i][j] + (1.0 - b1) * g;
            st.v[i][j] = b2 * st.v[i][j] + (1.0 - b2) * g * g;
        }
    }

    // 2) Compute dot products BEFORE bias correction. The projection gate
    //    operates on the raw first-moment direction (paper §4 Algorithm 1).
    double w_dot_m = 0.0, w_sq = 0.0, m_sq = 0.0;
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            double w_ij = (*param)[i][j];
            double m_ij = st.m[i][j];
            w_dot_m += w_ij * m_ij;
            w_sq    += w_ij * w_ij;
            m_sq    += m_ij * m_ij;
        }
    }

    // 3) Apply projection gate
    //    cos_sim = (w · m) / (||w|| · ||m||)
    //    if cos_sim > delta: subtract (w · m) / (||w||² + eps) · w from m
    double cos_sim = 0.0;
    double w_norm = std::sqrt(w_sq);
    double m_norm = std::sqrt(m_sq);
    if (w_norm > 0.0 && m_norm > 0.0) {
        cos_sim = w_dot_m / (w_norm * m_norm + eps);
    }
    if (cos_sim > delta) {
        double scale = w_dot_m / (w_sq + eps);
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                st.m[i][j] -= scale * (*param)[i][j];
            }
        }
    }

    // 4) Bias correction and step
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            double m_hat = st.m[i][j] / b1_c;
            double v_hat = st.v[i][j] / b2_c;
            double w_ij = (*param)[i][j];
            double step = m_hat / (std::sqrt(v_hat) + eps);
            // decoupled weight decay
            if (wd > 0.0) {
                (*param)[i][j] = w_ij * (1.0 - lr * wd) - lr * step;
            } else {
                (*param)[i][j] = w_ij - lr * step;
            }
        }
    }
}

// ---- step ----
void AdamP::step(Model& model) {
    const double b1_c = 1.0 - std::pow(beta1, t);
    const double b2_c = 1.0 - std::pow(beta2, t);
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], state_vec[i],
                         lr, beta1, beta2, epsilon, delta, weight_decay,
                         b1_c, b2_c);
        }
        layer->zero_grad();
    }
    ++t;
}
