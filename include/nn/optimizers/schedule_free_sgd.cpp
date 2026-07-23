#include "schedule_free_sgd.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// =========================================================================
// Schedule-Free SGD — implementation
//
// See schedule_free_sgd.h for the algorithm and citations.
// Mirrors the canonical facebookresearch/schedule_free SGD implementation
// (same algorithm as schedule_free_adamw but without the Adam second-moment
// path: no β2, no exp_avg_sq, no bias correction).
// =========================================================================

// ---- validation ----
void ScheduleFreeSGD::validate(double lr_, double beta1_, double weight_decay_,
                              int warmup_steps_, double r_,
                              double weight_lr_power_) {
    if (lr_ <= 0.0) {
        throw std::invalid_argument("ScheduleFreeSGD: learning rate must be > 0");
    }
    if (beta1_ < 0.0 || beta1_ >= 1.0) {
        throw std::invalid_argument("ScheduleFreeSGD: beta1 must be in [0, 1)");
    }
    if (weight_decay_ < 0.0) {
        throw std::invalid_argument("ScheduleFreeSGD: weight decay must be >= 0");
    }
    if (warmup_steps_ < 0) {
        throw std::invalid_argument("ScheduleFreeSGD: warmup_steps must be >= 0");
    }
    if (r_ < 0.0) {
        throw std::invalid_argument("ScheduleFreeSGD: r must be >= 0");
    }
    if (weight_lr_power_ < 0.0) {
        throw std::invalid_argument("ScheduleFreeSGD: weight_lr_power must be >= 0");
    }
}

// ---- constructor ----
ScheduleFreeSGD::ScheduleFreeSGD(double lr_, double beta1_, double weight_decay_,
                                 int warmup_steps_, double r_,
                                 double weight_lr_power_)
    : lr(lr_), beta1(beta1_), weight_decay(weight_decay_),
      warmup_steps(warmup_steps_), r(r_), weight_lr_power(weight_lr_power_) {
    validate(lr_, beta1_, weight_decay_,
             warmup_steps_, r_, weight_lr_power_);
}

// ---- setters ----
void ScheduleFreeSGD::set_lr(double new_lr) {
    validate(new_lr, beta1, weight_decay, warmup_steps, r, weight_lr_power);
    lr = new_lr;
}
void ScheduleFreeSGD::set_beta1(double new_beta1) {
    validate(lr, new_beta1, weight_decay, warmup_steps, r, weight_lr_power);
    beta1 = new_beta1;
}
void ScheduleFreeSGD::set_weight_decay(double new_wd) {
    validate(lr, beta1, new_wd, warmup_steps, r, weight_lr_power);
    weight_decay = new_wd;
}
void ScheduleFreeSGD::set_warmup_steps(int new_warmup) {
    validate(lr, beta1, weight_decay, new_warmup, r, weight_lr_power);
    warmup_steps = new_warmup;
}
void ScheduleFreeSGD::set_r(double new_r) {
    validate(lr, beta1, weight_decay, warmup_steps, new_r, weight_lr_power);
    r = new_r;
}
void ScheduleFreeSGD::set_weight_lr_power(double new_p) {
    validate(lr, beta1, weight_decay, warmup_steps, r, new_p);
    weight_lr_power = new_p;
}

// ---- ckp1 helper ----
// ckp1 = weight / weight_sum where
//   weight      = ((k+1)^r) * (lr_max^weight_lr_power)
//   weight_sum += weight
// Matches the canonical schedule_free reference (Schedule-Free AdamW uses the
// same recurrence; see schedule_free_adamw.h).
double ScheduleFreeSGD::compute_ckp1(double /*sched*/) {
    const double kp1 = static_cast<double>(k_) + 1.0;
    const double weight = std::pow(kp1, r) * std::pow(lr_max_, weight_lr_power);
    weight_sum_ += weight;
    if (weight_sum_ <= 0.0) return 0.0;
    return weight / weight_sum_;
}

// ---- ensure_state ----
void ScheduleFreeSGD::ensure_state(void* layer_ptr,
                                   const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<ParameterState> layer_state;
    layer_state.reserve(params.size());
    for (Tensor* parameter : params) {
        ParameterState s;
        // z = clone of parameter (preserve current parameter values).
        // Clone carries the parameter's shape automatically — no need
        // to record rows/cols separately.
        s.z = parameter->clone();
        layer_state.push_back(std::move(s));
    }
    state_[layer_ptr] = std::move(layer_state);
}

// ---- get_z ----
Tensor ScheduleFreeSGD::get_z(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].z.clone();
}

// ---- update_param ----
// Per-parameter step. `param` currently holds `y` (averaged parameter).
//   u          = g + weight_decay * y           (coupled decay form)
//   z_{k+1}    = z_k - lr * u                   (the iterate step)
//   y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k + lr * (beta1*(1-ckp1) - 1) * u
void ScheduleFreeSGD::update_param(Tensor* param, Tensor* grad,
                                   ParameterState& st,
                                   double scheduled_lr, double ckp1) {
    const size_t rows = param->rows;
    const size_t cols = param->cols;
    const double one_minus_ckp1 = 1.0 - ckp1;
    const double y_coeff = scheduled_lr * (beta1 * one_minus_ckp1 - 1.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double g = (*grad)[i][j];
            const double y = (*param)[i][j];
            const double z_val = st.z[i][j];

            // u = g + weight_decay * y  (coupled form)
            double u = g;
            if (weight_decay > 0.0) {
                u += weight_decay * y;
            }

            // z_{k+1} = z_k - scheduled_lr * u
            st.z[i][j] = z_val - scheduled_lr * u;

            // y_{k+1} = ckp1 * z_{k+1} + (1-ckp1) * y_k + y_coeff * u
            (*param)[i][j] = ckp1 * st.z[i][j] + one_minus_ckp1 * y + y_coeff * u;
        }
    }
}

// ---- step ----
void ScheduleFreeSGD::step(Model& model) {
    if (!train_mode_) {
        throw std::runtime_error(
            "ScheduleFreeSGD::step() called while not in train mode. "
            "Call optimizer.train() before stepping (see README for the "
            "standard schedule_free .train()/.eval() API)."
        );
    }

    // Schedule: linear warmup_steps ramp from 0/(k+1) to 1.
    const double kp1 = static_cast<double>(k_) + 1.0;
    const double sched = (k_ < warmup_steps) ? (kp1 / static_cast<double>(warmup_steps))
                                              : 1.0;
    const double scheduled_lr = lr * sched;

    // Update lr_max and compute ckp1 (eval-point mixing fraction).
    lr_max_ = std::max(scheduled_lr, lr_max_);
    const double ckp1 = compute_ckp1(sched);

    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "ScheduleFreeSGD: parameter and gradient counts must match");
        }

        ensure_state(layer_ptr, params);
        std::vector<ParameterState>& layer_state = state_[layer_ptr];

        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* param = params[param_idx];
            Tensor* grad = grads[param_idx];
            if (param->rows != grad->rows || param->cols != grad->cols) {
                throw std::runtime_error(
                    "ScheduleFreeSGD: parameter and gradient shapes must match");
            }
            update_param(param, grad, layer_state[param_idx],
                         scheduled_lr, ckp1);
        }
        layer->zero_grad();
    }

    ++k_;
}

// ---- eval / train mode swaps ----
//
// Per the reference: when switching from train→eval, p holds y, and we
// lerp toward z with weight (1 - 1/beta1). The arithmetic gives
//   p_new = p + (1-1/beta1) * (z - p) = p/beta1 + (1-1/beta1)*z = x
// which is the eval point.
//
// When switching from eval→train, p holds x, and we lerp toward z with
// weight (1 - beta1) which gives
//   p_new = (1 - (1-beta1)) * x + (1-beta1) * z = beta1*x + (1-beta1)*z = y.
void ScheduleFreeSGD::eval(Model& model) {
    if (!train_mode_) return;  // already in eval mode, no-op
    const double w = 1.0 - 1.0 / beta1;
    for (auto& layer : model.layers) {
        std::vector<Tensor*> params = layer->parameters();
        auto it = state_.find(layer.get());
        if (it == state_.end()) continue;
        for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
            if (p_idx >= it->second.size()) continue;
            Tensor& z = it->second[p_idx].z;
            Tensor* param = params[p_idx];
            for (size_t i = 0; i < param->rows; ++i) {
                for (size_t j = 0; j < param->cols; ++j) {
                    const double py = (*param)[i][j];
                    const double zv = z[i][j];
                    (*param)[i][j] = py + w * (zv - py);
                }
            }
        }
    }
    train_mode_ = false;
}

void ScheduleFreeSGD::train(Model& model) {
    if (train_mode_) return;
    const double w = 1.0 - beta1;
    for (auto& layer : model.layers) {
        std::vector<Tensor*> params = layer->parameters();
        auto it = state_.find(layer.get());
        if (it == state_.end()) continue;
        for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
            if (p_idx >= it->second.size()) continue;
            Tensor& z = it->second[p_idx].z;
            Tensor* param = params[p_idx];
            for (size_t i = 0; i < param->rows; ++i) {
                for (size_t j = 0; j < param->cols; ++j) {
                    const double px = (*param)[i][j];
                    const double zv = z[i][j];
                    (*param)[i][j] = px + w * (zv - px);
                }
            }
        }
    }
    train_mode_ = true;
}