#include "came.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

// =========================================================================
// CAME — Confidence-guided Adaptive Memory Efficient Optimization
// Luo et al. 2023 (https://arxiv.org/abs/2307.02047)
// Reference impl: https://github.com/yangluo7/CAME/blob/master/came_pytorch/CAME.py
//
// See came.h for the full algorithm. The key invariant is that 2-D
// parameters use a factored row/column EMA of g² (Adafactor-style memory
// savings) plus a confidence-guided residual scaling of the momentum
// direction, while 1-D parameters fall back to a per-element EMA with the
// residual step being identity (matching the canonical reference).
// =========================================================================

// ---- validation ----
void CAME::validate(double lr_,
                    double beta1_, double beta2_, double beta3_,
                    double eps1_, double eps2_,
                    double clip_threshold_, double weight_decay_) {
    if (lr_ <= 0.0) {
        throw std::invalid_argument("CAME: learning rate must be > 0");
    }
    if (beta1_ < 0.0 || beta1_ >= 1.0) {
        throw std::invalid_argument("CAME: beta1 must be in [0, 1)");
    }
    if (beta2_ < 0.0 || beta2_ >= 1.0) {
        throw std::invalid_argument("CAME: beta2 must be in [0, 1)");
    }
    if (beta3_ < 0.0 || beta3_ >= 1.0) {
        throw std::invalid_argument("CAME: beta3 must be in [0, 1)");
    }
    if (eps1_ <= 0.0) {
        throw std::invalid_argument("CAME: eps1 must be > 0");
    }
    if (eps2_ <= 0.0) {
        throw std::invalid_argument("CAME: eps2 must be > 0");
    }
    if (clip_threshold_ <= 0.0) {
        throw std::invalid_argument("CAME: clip_threshold must be > 0");
    }
    if (weight_decay_ < 0.0) {
        throw std::invalid_argument("CAME: weight decay must be >= 0");
    }
}

// ---- constructor ----
CAME::CAME(double lr_,
           double beta1_, double beta2_, double beta3_,
           double eps1_, double eps2_,
           double clip_threshold_, double weight_decay_)
    : lr(lr_),
      beta1(beta1_), beta2(beta2_), beta3(beta3_),
      eps1(eps1_), eps2(eps2_),
      clip_threshold(clip_threshold_),
      weight_decay(weight_decay_),
      t(0) {
    validate(lr_, beta1_, beta2_, beta3_, eps1_, eps2_, clip_threshold_, weight_decay_);
}

// ---- setters ----
void CAME::set_lr(double new_lr) {
    validate(new_lr, beta1, beta2, beta3, eps1, eps2, clip_threshold, weight_decay);
    lr = new_lr;
}
void CAME::set_beta1(double new_beta1) {
    validate(lr, new_beta1, beta2, beta3, eps1, eps2, clip_threshold, weight_decay);
    beta1 = new_beta1;
}
void CAME::set_beta2(double new_beta2) {
    validate(lr, beta1, new_beta2, beta3, eps1, eps2, clip_threshold, weight_decay);
    beta2 = new_beta2;
}
void CAME::set_beta3(double new_beta3) {
    validate(lr, beta1, beta2, new_beta3, eps1, eps2, clip_threshold, weight_decay);
    beta3 = new_beta3;
}
void CAME::set_eps1(double new_eps1) {
    validate(lr, beta1, beta2, beta3, new_eps1, eps2, clip_threshold, weight_decay);
    eps1 = new_eps1;
}
void CAME::set_eps2(double new_eps2) {
    validate(lr, beta1, beta2, beta3, eps1, new_eps2, clip_threshold, weight_decay);
    eps2 = new_eps2;
}
void CAME::set_clip_threshold(double new_clip) {
    validate(lr, beta1, beta2, beta3, eps1, eps2, new_clip, weight_decay);
    clip_threshold = new_clip;
}
void CAME::set_weight_decay(double new_wd) {
    validate(lr, beta1, beta2, beta3, eps1, eps2, clip_threshold, new_wd);
    weight_decay = new_wd;
}

// ---- RMS helper ----
double CAME::rms(const Tensor& t) {
    if (t.rows == 0 || t.cols == 0) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j) {
            double v = t[i][j];
            sum_sq += v * v;
        }
    return std::sqrt(sum_sq / static_cast<double>(t.rows * t.cols));
}

// ---- state accessors ----
Tensor CAME::get_exp_avg(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].exp_avg.clone();
}

bool CAME::get_exp_avg_sq_row(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    if (it->second[param_idx].is_1d) return false;
    out = it->second[param_idx].exp_avg_sq_row.clone();
    return true;
}

bool CAME::get_exp_avg_sq_col(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    if (it->second[param_idx].is_1d) return false;
    out = it->second[param_idx].exp_avg_sq_col.clone();
    return true;
}

bool CAME::get_exp_avg_res_row(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    if (it->second[param_idx].is_1d) return false;
    out = it->second[param_idx].exp_avg_res_row.clone();
    return true;
}

bool CAME::get_exp_avg_res_col(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    if (it->second[param_idx].is_1d) return false;
    out = it->second[param_idx].exp_avg_res_col.clone();
    return true;
}

bool CAME::get_exp_avg_sq(void* layer_ptr, size_t param_idx, Tensor& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    if (!it->second[param_idx].is_1d) return false;
    out = it->second[param_idx].exp_avg_sq.clone();
    return true;
}

bool CAME::get_rms(void* layer_ptr, size_t param_idx, double& out) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    out = it->second[param_idx].rms;
    return true;
}

bool CAME::is_1d(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) return false;
    return it->second[param_idx].is_1d;
}

// ---- ensure_state ----
// A parameter is "1-D" iff its shape is degenerate (one of rows/cols == 1).
// In the canonical reference, 1-D parameters use the per-element EMA path
// (no row/col factoring).
static bool is_degenerate_1d(const Tensor* param) {
    return param->rows == 1 || param->cols == 1;
}

void CAME::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<ParameterState> layer_state;
    layer_state.reserve(params.size());
    for (Tensor* parameter : params) {
        ParameterState s;
        const size_t r = parameter->rows;
        const size_t c = parameter->cols;
        s.is_1d = is_degenerate_1d(parameter);
        s.exp_avg = Tensor(r, c);
        s.exp_avg.fill(0.0);
        if (s.is_1d) {
            s.exp_avg_sq = Tensor(r, c);
            s.exp_avg_sq.fill(0.0);
        } else {
            s.exp_avg_sq_row = Tensor(r, 1);
            s.exp_avg_sq_row.fill(0.0);
            s.exp_avg_sq_col = Tensor(1, c);
            s.exp_avg_sq_col.fill(0.0);
            s.exp_avg_res_row = Tensor(r, 1);
            s.exp_avg_res_row.fill(0.0);
            s.exp_avg_res_col = Tensor(1, c);
            s.exp_avg_res_col.fill(0.0);
        }
        s.rms = 0.0;
        layer_state.push_back(std::move(s));
    }
    state_[layer_ptr] = std::move(layer_state);
}

// ---- update_param_1d ----
// Reference: per-element EMA + Adam-style momentum. Residual is identity.
//   raw        = g² + eps1
//   exp_avg_sq = β2 * exp_avg_sq + (1-β2) * raw
//   update     = g / sqrt(exp_avg_sq)
//   update    /= max(1, RMS(update) / clip_threshold)
//   exp_avg    = β1 * exp_avg + (1-β1) * update
//   update     = exp_avg
//   param     *= (1 − lr * wd)
//   param     -= lr * update
void CAME::update_param_1d(Tensor* param, Tensor* grad, ParameterState& state) {
    const size_t r = param->rows;
    const size_t c = param->cols;

    // Step 1: raw = g² + eps1, exp_avg_sq update, update = g / sqrt(...)
    Tensor update(r, c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            const double g = (*grad)[i][j];
            const double raw = g * g + eps1;
            state.exp_avg_sq[i][j] = beta2 * state.exp_avg_sq[i][j]
                                   + (1.0 - beta2) * raw;
            const double denom = std::sqrt(state.exp_avg_sq[i][j]);
            update[i][j] = g / denom;
        }
    }

    // Step 2: RMS-clip the update
    const double update_rms = rms(update);
    const double clip_factor = std::max(1.0, update_rms / clip_threshold);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            update[i][j] /= clip_factor;
    state.rms = update_rms / clip_factor;

    // Step 3: momentum
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            state.exp_avg[i][j] = beta1 * state.exp_avg[i][j]
                                + (1.0 - beta1) * update[i][j];
            // Step 4: 1-D residual is identity — update = exp_avg
            update[i][j] = state.exp_avg[i][j];
        }
    }

    // Step 5: decoupled weight decay (subtractive AdamW form)
    const double decay_factor = 1.0 - lr * weight_decay;
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            (*param)[i][j] = (*param)[i][j] * decay_factor - lr * update[i][j];
        }
    }
}

// ---- update_param_2d ----
// Reference: factored row/column EMA + residual confidence scaling.
//   raw_ij             = g²_ij + eps1
//   row_EMA_i          = β2 * row_EMA_i + (1-β2) * mean_j(raw_ij)
//   col_EMA_j          = β2 * col_EMA_j + (1-β2) * mean_i(raw_ij)
//   v_ij               = row_EMA_i * col_EMA_j / max(eps, mean_i(row_EMA_i))
//   update_ij          = g_ij / sqrt(v_ij)
//   update            /= max(1, RMS(update) / clip_threshold)
//   exp_avg_ij         = β1 * exp_avg_ij + (1-β1) * update_ij
//   res_ij             = (update_ij − exp_avg_ij)² + eps2
//   res_row_i          = β3 * res_row_i + (1-β3) * mean_j(res_ij)
//   res_col_j          = β3 * res_col_j + (1-β3) * mean_i(res_ij)
//   confidence_ij      = sqrt(res_row_i * res_col_j / max(eps, mean_i(res_row_i)))
//   update_ij         *= exp_avg_ij * confidence_ij
//   param_ij          *= (1 − lr * wd)
//   param_ij          -= lr * update_ij
void CAME::update_param_2d(Tensor* param, Tensor* grad, ParameterState& state) {
    const size_t r = param->rows;
    const size_t c = param->cols;

    // ---- Step 1: row/col EMA of raw = g² + eps1 ----
    // Compute row_means[i] = mean_j(raw[i][j]) and col_means[j] = mean_i(raw[i][j])
    // simultaneously, then update the row/col accumulators.
    Tensor update(r, c);
    for (size_t i = 0; i < r; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < c; ++j) {
            const double g = (*grad)[i][j];
            row_sum += g * g + eps1;
        }
        const double row_mean = row_sum / static_cast<double>(c);
        state.exp_avg_sq_row[i][0] = beta2 * state.exp_avg_sq_row[i][0]
                                   + (1.0 - beta2) * row_mean;
    }
    for (size_t j = 0; j < c; ++j) {
        double col_sum = 0.0;
        for (size_t i = 0; i < r; ++i) {
            const double g = (*grad)[i][j];
            col_sum += g * g + eps1;
        }
        const double col_mean = col_sum / static_cast<double>(r);
        state.exp_avg_sq_col[0][j] = beta2 * state.exp_avg_sq_col[0][j]
                                   + (1.0 - beta2) * col_mean;
    }

    // mean(row_EMA) for the denominator of v_ij
    double row_ema_sum = 0.0;
    for (size_t i = 0; i < r; ++i) row_ema_sum += state.exp_avg_sq_row[i][0];
    const double mean_row_ema = row_ema_sum / static_cast<double>(r);
    const double mean_row_ema_safe = std::max(mean_row_ema, eps1);

    // ---- Step 2: update = g / sqrt(row_EMA * col_EMA / mean(row_EMA)) ----
    for (size_t i = 0; i < r; ++i) {
        const double row_val = state.exp_avg_sq_row[i][0];
        for (size_t j = 0; j < c; ++j) {
            const double col_val = state.exp_avg_sq_col[0][j];
            const double v_ij = row_val * col_val / mean_row_ema_safe;
            const double g = (*grad)[i][j];
            update[i][j] = g / std::sqrt(v_ij);
        }
    }

    // ---- Step 3: RMS-clip the update ----
    const double update_rms = rms(update);
    const double clip_factor = std::max(1.0, update_rms / clip_threshold);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            update[i][j] /= clip_factor;
    state.rms = update_rms / clip_factor;

    // ---- Step 4: momentum (Adam-style first moment) ----
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            state.exp_avg[i][j] = beta1 * state.exp_avg[i][j]
                                + (1.0 - beta1) * update[i][j];
    // ---- Step 5: residual = (update - exp_avg)² + eps2 ----
    // Then update res_row, res_col, and re-scale update = exp_avg * confidence_factor.
    for (size_t i = 0; i < r; ++i) {
        double res_row_sum = 0.0;
        for (size_t j = 0; j < c; ++j) {
            const double diff = update[i][j] - state.exp_avg[i][j];
            res_row_sum += diff * diff + eps2;
        }
        const double res_row_mean = res_row_sum / static_cast<double>(c);
        state.exp_avg_res_row[i][0] = beta3 * state.exp_avg_res_row[i][0]
                                    + (1.0 - beta3) * res_row_mean;
    }
    for (size_t j = 0; j < c; ++j) {
        double res_col_sum = 0.0;
        for (size_t i = 0; i < r; ++i) {
            const double diff = update[i][j] - state.exp_avg[i][j];
            res_col_sum += diff * diff + eps2;
        }
        const double res_col_mean = res_col_sum / static_cast<double>(r);
        state.exp_avg_res_col[0][j] = beta3 * state.exp_avg_res_col[0][j]
                                    + (1.0 - beta3) * res_col_mean;
    }

    // mean(res_row) for the confidence-factor denominator
    double res_row_sum = 0.0;
    for (size_t i = 0; i < r; ++i) res_row_sum += state.exp_avg_res_row[i][0];
    const double mean_res_row = res_row_sum / static_cast<double>(r);
    const double mean_res_row_safe = std::max(mean_res_row, eps2);

    // ---- Step 6: scale update by exp_avg * sqrt(res_row * res_col / mean(res_row)) ----
    for (size_t i = 0; i < r; ++i) {
        const double res_row_val = state.exp_avg_res_row[i][0];
        for (size_t j = 0; j < c; ++j) {
            const double res_col_val = state.exp_avg_res_col[0][j];
            const double confidence = std::sqrt(res_row_val * res_col_val / mean_res_row_safe);
            update[i][j] = state.exp_avg[i][j] * confidence;
        }
    }

    // ---- Step 7: decoupled weight decay + parameter update ----
    const double decay_factor = 1.0 - lr * weight_decay;
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            (*param)[i][j] = (*param)[i][j] * decay_factor - lr * update[i][j];
        }
    }
}

// ---- step ----
void CAME::step(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "CAME: parameter and gradient counts must match");
        }

        ensure_state(layer_ptr, params);
        std::vector<ParameterState>& layer_state = state_[layer_ptr];
        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* param = params[param_idx];
            Tensor* grad = grads[param_idx];
            if (param->rows != grad->rows || param->cols != grad->cols) {
                throw std::runtime_error(
                    "CAME: parameter and gradient shapes must match");
            }
            if (layer_state[param_idx].is_1d) {
                update_param_1d(param, grad, layer_state[param_idx]);
            } else {
                update_param_2d(param, grad, layer_state[param_idx]);
            }
        }
        layer->zero_grad();
    }
    ++t;
}
