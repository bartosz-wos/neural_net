#include "stableadamw.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

StableAdamW::StableAdamW(double lr_, double b1, double b2, double eps, double wd)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      weight_decay(wd), t(1) {
    // Use validated setters — they throw on invalid input.
    set_lr(lr_);
    set_beta1(b1);
    set_beta2(b2);
    set_epsilon(eps);
    set_weight_decay(wd);
}

void StableAdamW::set_lr(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("StableAdamW: lr must be > 0");
    lr = v;
    Optimizer::lr = v;
}

void StableAdamW::set_beta1(double v) {
    if (!(v >= 0.0 && v < 1.0))
        throw std::invalid_argument("StableAdamW: beta1 must be in [0,1)");
    beta1 = v;
}

void StableAdamW::set_beta2(double v) {
    if (!(v >= 0.0 && v < 1.0))
        throw std::invalid_argument("StableAdamW: beta2 must be in [0,1)");
    beta2 = v;
}

void StableAdamW::set_epsilon(double v) {
    if (!(v > 0.0))
        throw std::invalid_argument("StableAdamW: epsilon must be > 0");
    epsilon = v;
}

void StableAdamW::set_weight_decay(double v) {
    if (!(v >= 0.0))
        throw std::invalid_argument("StableAdamW: weight_decay must be >= 0");
    weight_decay = v;
}

void StableAdamW::ensure_state(void* layer_ptr,
                                const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<ParamState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        ParamState st;
        st.m = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.v = Tensor(p.rows, p.cols);
        st.v.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

void StableAdamW::step(Model& model) {
    // Bias-correction denominators (Adam convention).
    //   bc1 = 1 - β1^t   bc2 = 1 - β2^t
    // When β is very close to 1 and t is small, this can be very small
    // (e.g. β1=0.9, t=1 → bc1=0.1, perfectly fine). We defensive-clamp
    // to avoid 0/0 in degenerate edge cases.
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);
    if (bc1 < 1e-12) bc1 = 1.0;
    if (bc2 < 1e-12) bc2 = 1.0;

    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::logic_error("StableAdamW: param/grad count mismatch");
        }
        ensure_state(layer_ptr, params);
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            if (grad->rows != param->rows || grad->cols != param->cols) {
                throw std::logic_error("StableAdamW: param/grad shape mismatch");
            }
            ParamState& st = st_vec[i];

            // Per-coordinate update: m, v, m_hat, v_hat, update, clip, apply.
            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g = (*grad)[r][c];

                    // m ← β1 * m + (1-β1) * g
                    st.m[r][c] = beta1 * st.m[r][c] + (1.0 - beta1) * g;

                    // v ← β2 * v + (1-β2) * g²
                    double g2 = g * g;
                    st.v[r][c] = beta2 * st.v[r][c] + (1.0 - beta2) * g2;

                    // Bias-corrected moments.
                    double m_hat = st.m[r][c] / bc1;
                    double v_hat = st.v[r][c] / bc2;

                    // Adam direction.
                    double denom = std::sqrt(v_hat) + epsilon;
                    double upd = m_hat / denom;

                    // KEY: update clipping. This is the StableAdamW innovation.
                    if (upd > 1.0) upd = 1.0;
                    else if (upd < -1.0) upd = -1.0;

                    // Decoupled weight decay (AdamW-style): θ *= (1 - lr*wd).
                    // Applied BEFORE the update so the WD is decoupled.
                    if (weight_decay > 0.0) {
                        (*param)[r][c] *= (1.0 - lr * weight_decay);
                    }

                    // Apply the clipped update.
                    (*param)[r][c] -= lr * upd;
                }
            }
        }

        layer->zero_grad();
    }

    ++t;
}

size_t StableAdamW::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    return it == state_.end() ? 0 : it->second.size();
}

const Tensor& StableAdamW::get_m(void* layer_ptr, size_t param_idx) const {
    static const Tensor empty;  // (0,0) default
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return empty;
    return it->second[param_idx].m;
}

const Tensor& StableAdamW::get_v(void* layer_ptr, size_t param_idx) const {
    static const Tensor empty;  // (0,0) default
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return empty;
    return it->second[param_idx].v;
}
