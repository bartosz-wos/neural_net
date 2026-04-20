#include "swa.h"
#include "../core/model.h"
#include <cmath>

SWAOptimizer::SWAOptimizer(Optimizer* inner, size_t start_after_steps)
    : inner_(inner), start_after_(start_after_steps), step_count_(0),
      initialized_(false), param_total_(0) {}

void SWAOptimizer::init_if_needed(const std::vector<Tensor*>& params) {
    if (initialized_) return;
    param_total_ = params.size();
    shadow_weights_.resize(param_total_);
    averaged_weights_.resize(param_total_);
    size_t idx = 0;
    for (Tensor* p : params) {
        // Init shadow to current weight values
        shadow_weights_[idx] = Tensor(p->rows, p->cols);
        for (size_t i = 0; i < p->rows; ++i)
            for (size_t j = 0; j < p->cols; ++j)
                shadow_weights_[idx][i][j] = (*p)[i][j];
        // Init averaged weights to zero
        averaged_weights_[idx] = Tensor(p->rows, p->cols);
        averaged_weights_[idx].fill(0.0);
        ++idx;
    }
    initialized_ = true;
    step_count_ = 0;
}

void SWAOptimizer::step(Model& model) {
    std::vector<Tensor*> params;
    for (auto& layer : model.layers)
        for (Tensor* p : layer->parameters())
            params.push_back(p);

    init_if_needed(params);

    inner_->step(model);

    if (step_count_ >= start_after_) {
        size_t avg_idx = 0;
        double n = static_cast<double>(step_count_ - start_after_ + 1);
        for (auto& layer : model.layers)
            for (Tensor* p : layer->parameters()) {
                // Bounds check: prevent out-of-bounds if parameter count changed
                if (avg_idx >= param_total_) break;
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j) {
                        double old_avg = averaged_weights_[avg_idx][i][j];
                        averaged_weights_[avg_idx][i][j] = (old_avg * (n - 1) + (*p)[i][j]) / n;
                    }
                ++avg_idx;
            }
    }
    ++step_count_;
}

void SWAOptimizer::swap_to_averaged(Model& model) {
    if (!initialized_) return;
    size_t idx = 0;
    for (auto& layer : model.layers)
        for (Tensor* p : layer->parameters()) {
            for (size_t i = 0; i < p->rows; ++i)
                for (size_t j = 0; j < p->cols; ++j)
                    (*p)[i][j] = averaged_weights_[idx][i][j];
            ++idx;
        }
}

void SWAOptimizer::record(Model& model) {
    if (!initialized_) return;
    size_t idx = 0;
    for (auto& layer : model.layers)
        for (Tensor* p : layer->parameters()) {
            for (size_t i = 0; i < p->rows; ++i)
                for (size_t j = 0; j < p->cols; ++j)
                    averaged_weights_[idx][i][j] = (*p)[i][j];
            ++idx;
        }
}

SWALRScheduler::SWALRScheduler(double start_lr, double swa_lr,
                                 size_t warmup_steps, size_t swa_start_step)
    : start_lr_(start_lr), swa_lr_(swa_lr), current_lr_(start_lr),
      warmup_steps_(warmup_steps), swa_start_step_(swa_start_step), step_count_(0) {}

void SWALRScheduler::update_lr() {
    if (step_count_ < warmup_steps_ && warmup_steps_ > 0) {
        current_lr_ = start_lr_ * (static_cast<double>(step_count_) / warmup_steps_);
    } else if (step_count_ >= swa_start_step_) {
        current_lr_ = swa_lr_;
    } else {
        current_lr_ = start_lr_;
    }
}