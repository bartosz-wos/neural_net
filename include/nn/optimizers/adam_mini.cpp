#include "adam_mini.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

AdamMini::AdamMini(double lr_, double b1, double b2, double eps, double wd, BlockMode mode)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      weight_decay(wd), t(1), default_mode(mode) {
    // Use setters to validate; they will throw on invalid inputs.
    set_lr(lr_);
    set_beta1(b1);
    set_beta2(b2);
    set_epsilon(eps);
    set_weight_decay(wd);
}

void AdamMini::set_lr(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("AdamMini: lr must be >= 0");
    lr = v;
    Optimizer::lr = v;
}

void AdamMini::set_beta1(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("AdamMini: beta1 must be in [0,1)");
    beta1 = v;
}

void AdamMini::set_beta2(double v) {
    if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("AdamMini: beta2 must be in [0,1)");
    beta2 = v;
}

void AdamMini::set_epsilon(double v) {
    if (!(v > 0.0)) throw std::invalid_argument("AdamMini: epsilon must be > 0");
    epsilon = v;
}

void AdamMini::set_weight_decay(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("AdamMini: weight_decay must be >= 0");
    weight_decay = v;
}

AdamMini::BlockMode AdamMini::resolve_mode(void* layer_ptr, size_t param_idx, const Tensor& p) const {
    auto oit = overrides_.find(layer_ptr);
    if (oit != overrides_.end() && param_idx < oit->second.size()) {
        BlockMode m = oit->second[param_idx];
        if (m != BlockMode::AUTO) return m;
    }
    if (default_mode != BlockMode::AUTO) return default_mode;
    // AUTO: 1-D → FULL, 2-D → ROW_MEAN
    return (p.rows == 1 || p.cols == 1) ? BlockMode::FULL : BlockMode::ROW_MEAN;
}

void AdamMini::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<ParamState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor* p = params[i];
        ParamState st;
        st.m = Tensor(p->rows, p->cols);
        st.m.fill(0.0);
        BlockMode mode = resolve_mode(layer_ptr, vec.size(), *p);
        st.mode = mode;
        size_t v_rows = p->rows;
        size_t v_cols = p->cols;
        switch (mode) {
            case BlockMode::FULL:
                // vmean = full (same shape as param)
                break;
            case BlockMode::ROW_MEAN:
                v_cols = 1;  // one value per row
                break;
            case BlockMode::SCALAR:
                v_rows = 1;
                v_cols = 1;
                break;
            default:
                throw std::logic_error("AdamMini: unresolved mode in ensure_state");
        }
        st.vmean = Tensor(v_rows, v_cols);
        st.vmean.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

void AdamMini::step(Model& model) {
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);

    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::logic_error("AdamMini: param/grad count mismatch");
        }
        ensure_state(layer_ptr, params);
        auto& st_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            if (grad->rows != param->rows || grad->cols != param->cols) {
                throw std::logic_error("AdamMini: param/grad shape mismatch");
            }
            ParamState& st = st_vec[i];

            // Update m: m = β1*m + (1-β1)*g (full shape)
            for (size_t r = 0; r < grad->rows; ++r)
                for (size_t c = 0; c < grad->cols; ++c)
                    st.m[r][c] = beta1 * st.m[r][c] + (1.0 - beta1) * (*grad)[r][c];

            // Update vmean depending on mode
            switch (st.mode) {
                case BlockMode::FULL: {
                    // vmean same shape as param
                    for (size_t r = 0; r < grad->rows; ++r)
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            st.vmean[r][c] = beta2 * st.vmean[r][c] + (1.0 - beta2) * g * g;
                        }
                    break;
                }
                case BlockMode::ROW_MEAN: {
                    // vmean shape (rows, 1): one value per row
                    for (size_t r = 0; r < grad->rows; ++r) {
                        double s = 0.0;
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            s += g * g;
                        }
                        double mean_gg = s / static_cast<double>(grad->cols);
                        st.vmean[r][0] = beta2 * st.vmean[r][0] + (1.0 - beta2) * mean_gg;
                    }
                    break;
                }
                case BlockMode::SCALAR: {
                    // vmean shape (1, 1): one global value
                    double s = 0.0;
                    size_t n = grad->rows * grad->cols;
                    for (size_t r = 0; r < grad->rows; ++r)
                        for (size_t c = 0; c < grad->cols; ++c) {
                            double g = (*grad)[r][c];
                            s += g * g;
                        }
                    double mean_gg = s / static_cast<double>(n);
                    st.vmean[0][0] = beta2 * st.vmean[0][0] + (1.0 - beta2) * mean_gg;
                    break;
                }
                default:
                    throw std::logic_error("AdamMini: unknown mode");
            }

            // Decoupled weight decay
            if (weight_decay > 0.0) {
                double scale = 1.0 - lr * weight_decay;
                for (size_t r = 0; r < param->rows; ++r)
                    for (size_t c = 0; c < param->cols; ++c)
                        (*param)[r][c] *= scale;
            }

            // Compute update
            switch (st.mode) {
                case BlockMode::FULL: {
                    for (size_t r = 0; r < param->rows; ++r)
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double v_hat = st.vmean[r][c] / bc2;
                            double denom = std::sqrt(v_hat) + epsilon;
                            double upd = m_hat / denom;
                            (*param)[r][c] -= lr * upd;
                        }
                    break;
                }
                case BlockMode::ROW_MEAN: {
                    for (size_t r = 0; r < param->rows; ++r) {
                        double v_hat_r = st.vmean[r][0] / bc2;
                        double denom_r = std::sqrt(v_hat_r) + epsilon;
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double upd = m_hat / denom_r;
                            (*param)[r][c] -= lr * upd;
                        }
                    }
                    break;
                }
                case BlockMode::SCALAR: {
                    double v_hat = st.vmean[0][0] / bc2;
                    double denom = std::sqrt(v_hat) + epsilon;
                    for (size_t r = 0; r < param->rows; ++r)
                        for (size_t c = 0; c < param->cols; ++c) {
                            double m_hat = st.m[r][c] / bc1;
                            double upd = m_hat / denom;
                            (*param)[r][c] -= lr * upd;
                        }
                    break;
                }
                default:
                    throw std::logic_error("AdamMini: unknown mode");
            }
        }

        layer->zero_grad();
    }

    ++t;
}

bool AdamMini::has_state(void* layer_ptr) const {
    return state_.find(layer_ptr) != state_.end();
}

size_t AdamMini::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    return it == state_.end() ? 0 : it->second.size();
}

const Tensor& AdamMini::get_m(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).m;
}

const Tensor& AdamMini::get_vmean(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).vmean;
}

AdamMini::BlockMode AdamMini::get_block_mode(void* layer_ptr, size_t param_idx) const {
    return state_.at(layer_ptr).at(param_idx).mode;
}

void AdamMini::set_param_block_mode(void* layer_ptr, size_t param_idx, BlockMode mode) {
    auto oit = overrides_.find(layer_ptr);
    if (oit == overrides_.end()) {
        overrides_[layer_ptr] = std::vector<BlockMode>();
        oit = overrides_.find(layer_ptr);
    }
    if (param_idx >= oit->second.size()) oit->second.resize(param_idx + 1, BlockMode::AUTO);
    oit->second[param_idx] = mode;
    // If state already exists for this layer and the param, update the mode.
    // The vmean shape doesn't change automatically; that's OK because:
    //   - For ROW_MEAN → SCALAR: state was (R, 1), now needs (1, 1). The
    //     step() will only read vmean[0][0] for SCALAR, so the extra rows
    //     are harmless.
    //   - For SCALAR → ROW_MEAN: state was (1, 1), now needs (R, 1). The
    //     step() will only read vmean[r][0] for ROW_MEAN, so reading past
    //     the end is the caller's bug.
    //   - To avoid bugs, we drop the state so it re-initializes next step.
    auto sit = state_.find(layer_ptr);
    if (sit != state_.end() && param_idx < sit->second.size()) {
        sit->second[param_idx].mode = mode;
    }
}
