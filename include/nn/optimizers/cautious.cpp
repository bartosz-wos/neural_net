#include "cautious.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include "optimizer.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Cautious::Cautious(std::unique_ptr<Optimizer> inner, double eps_mask)
    : inner_(std::move(inner)), eps_mask_(eps_mask), last_num_params_updated_(0) {
    if (!inner_)
        throw std::invalid_argument("Cautious: inner optimizer must not be null");
    if (!(eps_mask > 0.0))
        throw std::invalid_argument("Cautious: eps_mask must be > 0");
    // Inherit the inner optimizer's lr into our Optimizer::lr (for scheduler
    // compatibility — schedulers write to Optimizer::lr).
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) this->lr = adam->lr;
    else if (auto* sgd = dynamic_cast<SGD*>(inner_.get())) this->lr = sgd->lr;
}
void Cautious::set_eps_mask(double v) {
    if (!(v > 0.0))
        throw std::invalid_argument("Cautious::set_eps_mask: must be > 0");
    eps_mask_ = v;
}

void Cautious::set_lr(double v) {
    if (!(v >= 0.0))
        throw std::invalid_argument("Cautious::set_lr: must be >= 0");
    if (inner_) inner_->lr = v;
    this->lr = v;
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) adam->lr = v;
    if (auto* sgd  = dynamic_cast<SGD*>(inner_.get()))  sgd->lr  = v;
}

double Cautious::get_lr() const {
    if (!inner_) return this->lr;
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) return adam->lr;
    if (auto* sgd  = dynamic_cast<SGD*>(inner_.get()))  return sgd->lr;
    return inner_->lr;
}

bool Cautious::handles_weight_decay() const {
    return inner_ ? inner_->handles_weight_decay() : false;
}

bool Cautious::has_state(void* layer_ptr, size_t param_idx) const {
    auto it = stats_.find(layer_ptr);
    if (it == stats_.end()) return false;
    return param_idx < it->second.size();
}

std::pair<double, double> Cautious::total_mask_stats() const {
    double s = 0.0, c = 0.0;
    for (auto& kv : stats_) {
        for (auto& e : kv.second) {
            s += e.mask_sum;
            c += e.count;
        }
    }
    return {s, c};
}

std::pair<double, double> Cautious::get_param_stats(void* layer_ptr, size_t param_idx) const {
    auto it = stats_.find(layer_ptr);
    if (it == stats_.end() || param_idx >= it->second.size())
        return {0.0, 0.0};
    return {it->second[param_idx].mask_sum, it->second[param_idx].count};
}

void Cautious::step(Model& model) {
    if (!inner_)
        throw std::logic_error("Cautious::step: inner optimizer is null");

    // Strategy: for each parameter, snapshot the value, let the inner optimizer
    // apply its update via `inner.step(model)` (which mutates parameters in
    // place), then RECONSTRUCT the update direction u_t = (param - before) / lr,
    // RESTORE the param to the snapshot, and APPLY the cautious mask.

    // Clear last-step stats.
    stats_.clear();
    last_num_params_updated_ = 0;

    // Collect snapshots and (param, grad) pointers per layer.
    struct Snap {
        Tensor before;        // copy of param BEFORE inner step
        Tensor grad;          // copy of gradient BEFORE inner step (inner.step zeroes grads)
        Tensor* param;
    };
    std::vector<std::vector<Snap>> per_layer_snaps;
    per_layer_snaps.reserve(model.layers.size());

    for (auto& layer : model.layers) {
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.size() != grads.size()) {
            throw std::logic_error("Cautious: parameter/gradient count mismatch");
        }
        std::vector<Snap> layer_snaps;
        layer_snaps.reserve(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            Snap s;
            s.before = *params[i];     // copy param
            s.grad = *grads[i];        // copy gradient (inner.step zeroes grads)
            s.param = params[i];
            layer_snaps.push_back(std::move(s));
        }
        per_layer_snaps.push_back(std::move(layer_snaps));
    }

    // Let the inner optimizer do its standard step.
    inner_->step(model);

    // For each parameter, reconstruct the update direction, restore the
    // parameter, and apply the cautious mask.
    double inner_lr = inner_->lr;
    if (!(inner_lr > 0.0)) {
        // Inner LR is zero — inner step is a no-op. Cautious is also a no-op.
        // Still record stats: for completeness, every entry has mask=0 (since
        // u_t = 0, anything * 0 > 0 is false).
        for (size_t li = 0; li < model.layers.size(); ++li) {
            auto* layer_ptr = model.layers[li].get();
            std::vector<Entry> entries;
            for (size_t pi = 0; pi < per_layer_snaps[li].size(); ++pi) {
                Entry e;
                e.mask_sum = 0.0;
                e.count = static_cast<double>(per_layer_snaps[li][pi].before.rows *
                                              per_layer_snaps[li][pi].before.cols);
                entries.push_back(e);
                ++last_num_params_updated_;
            }
            stats_[layer_ptr] = std::move(entries);
        }
        return;
    }

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        std::vector<Entry> entries;
        entries.reserve(per_layer_snaps[li].size());

        for (size_t pi = 0; pi < per_layer_snaps[li].size(); ++pi) {
            Snap& s = per_layer_snaps[li][pi];
            Tensor& p = *(s.param);
            const Tensor& before = s.before;
            const Tensor& g = s.grad;  // use the SNAPSHOT gradient

            // Compute direction_t = (before - p) / lr (the inner optimizer's step
            // DIRECTION, sign-flipped from the (param_after - param_before) delta).
            // For Adam with positive grad, direction_t > 0 (gradient-aligned).
            // Compute mask = (direction_t * g > 0), mask_sum, mask_mean.
            double mask_sum = 0.0;
            double count = static_cast<double>(p.rows * p.cols);
            for (size_t i = 0; i < p.rows; ++i) {
                for (size_t j = 0; j < p.cols; ++j) {
                    double direction_t = (before[i][j] - p[i][j]) / inner_lr;
                    double m = (direction_t * g[i][j] > 0.0) ? 1.0 : 0.0;
                    mask_sum += m;
                }
            }
            double mask_mean = (count > 0.0) ? (mask_sum / count) : 0.0;
            if (mask_mean < eps_mask_) mask_mean = eps_mask_;
            double inv_mask_mean = 1.0 / mask_mean;

            // Restore param to `before`, then apply cautious update.
            // Cautious update: param[i][j] = before[i][j] - lr * direction_t * m * inv_mask_mean
            for (size_t i = 0; i < p.rows; ++i) {
                for (size_t j = 0; j < p.cols; ++j) {
                    double direction_t = (before[i][j] - p[i][j]) / inner_lr;
                    double m = (direction_t * g[i][j] > 0.0) ? 1.0 : 0.0;
                    p[i][j] = before[i][j] - inner_lr * direction_t * m * inv_mask_mean;
                }
            }

            Entry e;
            e.mask_sum = mask_sum;
            e.count = count;
            entries.push_back(e);
            ++last_num_params_updated_;
        }

        stats_[layer_ptr] = std::move(entries);
    }
}
