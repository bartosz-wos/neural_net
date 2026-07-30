#include "grokfast.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include "optimizer.h"
#include <stdexcept>
#include <cmath>

GrokFast::GrokFast(std::unique_ptr<Optimizer> inner, double lambda, double alpha)
    : inner_(std::move(inner)), lambda_(lambda), alpha_(alpha),
      last_num_params_filtered_(0) {
    if (!inner_)
        throw std::invalid_argument("GrokFast: inner optimizer must not be null");
    if (!(lambda >= 0.0))
        throw std::invalid_argument("GrokFast: lambda must be >= 0");
    if (!(alpha >= 0.0 && alpha <= 1.0))
        throw std::invalid_argument("GrokFast: alpha must be in [0, 1]");
    // Inherit the inner optimizer's lr into our Optimizer::lr (for scheduler
    // compatibility — schedulers write to Optimizer::lr).
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) this->lr = adam->lr;
    else if (auto* sgd = dynamic_cast<SGD*>(inner_.get())) this->lr = sgd->lr;
    else this->lr = inner_->lr;
}

void GrokFast::set_lambda(double v) {
    if (!(v >= 0.0))
        throw std::invalid_argument("GrokFast::set_lambda: must be >= 0");
    lambda_ = v;
}

void GrokFast::set_alpha(double v) {
    if (!(v >= 0.0 && v <= 1.0))
        throw std::invalid_argument("GrokFast::set_alpha: must be in [0, 1]");
    alpha_ = v;
}

void GrokFast::set_lr(double v) {
    if (!(v >= 0.0))
        throw std::invalid_argument("GrokFast::set_lr: must be >= 0");
    if (inner_) inner_->lr = v;
    this->lr = v;
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) adam->lr = v;
    if (auto* sgd = dynamic_cast<SGD*>(inner_.get())) sgd->lr = v;
}

double GrokFast::get_lr() const {
    if (!inner_) return this->lr;
    if (auto* adam = dynamic_cast<Adam*>(inner_.get())) return adam->lr;
    if (auto* sgd  = dynamic_cast<SGD*>(inner_.get()))  return sgd->lr;
    return inner_->lr;
}

bool GrokFast::handles_weight_decay() const {
    return inner_ ? inner_->handles_weight_decay() : false;
}

bool GrokFast::has_state(void* layer_ptr, size_t param_idx) const {
    auto it = buf_state_.find(layer_ptr);
    if (it == buf_state_.end()) return false;
    return param_idx < it->second.size();
}

void GrokFast::step(Model& model) {
    if (!inner_)
        throw std::logic_error("GrokFast::step: inner optimizer is null");

    // Reset per-step diagnostic.
    last_num_params_filtered_ = 0;

    // For each parameter, lazily initialize the EMA-filter buffer and overwrite
    // the stored gradient with the filtered value: grad_filtered = grad + lambda * buf,
    // where buf = alpha * buf_prev + (1 - alpha) * grad.
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.size() != grads.size()) {
            throw std::logic_error("GrokFast: parameter/gradient count mismatch");
        }

        // Initialize state for this layer on first encounter.
        if (buf_state_.find(ptr) == buf_state_.end()) {
            buf_state_[ptr] = std::vector<Tensor>(params.size());
        }
        auto& buf_vec = buf_state_[ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];

            // Lazy init or shape-change re-init of the EMA buffer for this parameter.
            if (i >= buf_vec.size() ||
                buf_vec[i].rows != param->rows ||
                buf_vec[i].cols != param->cols) {
                buf_vec[i] = Tensor(param->rows, param->cols);
                buf_vec[i].fill(0.0);
            }
            Tensor& buf = buf_vec[i];

            // Elementwise: buf = alpha*buf + (1-alpha)*grad;  grad = grad + lambda*buf
            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g = (*grad)[r][c];
                    buf[r][c] = alpha_ * buf[r][c] + (1.0 - alpha_) * g;
                    (*grad)[r][c] = g + lambda_ * buf[r][c];
                }
            }

            ++last_num_params_filtered_;
        }
    }

    // Delegate to the inner optimizer, which now sees the filtered gradients.
    inner_->step(model);
}