#include "lr_finder.h"
#include "../optimizers/optimizer.h"
#include "../optimizers/optimizer_extended.h"
#include "../optimizers/rmsprop.h"
#include <typeinfo>
#include <cmath>
#include <stdexcept>
#include <limits>

LRFinder::LRFinder(double min_lr, double max_lr, int num_steps, double beta)
    : min_lr_(min_lr), max_lr_(max_lr), num_steps_(num_steps), beta_(beta) {}

std::vector<std::pair<double, double>> LRFinder::run(
    Model& model,
    const Tensor& X,
    const Tensor& y,
    Optimizer& opt)
{
    history_.clear();

    // Fresh optimizer with same concrete type and LR as the input optimizer.
    // We must clone the type (not just use Adam) to preserve the caller's optimizer
    // semantics (SGD, RMSProp, AdamW, etc.) during the range test.
    double base_lr = opt.lr;
    std::unique_ptr<Optimizer> worker_opt;

    if (dynamic_cast<Adam*>(&opt)) {
        worker_opt = std::make_unique<Adam>(base_lr);
    } else if (dynamic_cast<SGD*>(&opt)) {
        worker_opt = std::make_unique<SGD>(base_lr);
    } else if (dynamic_cast<RMSprop*>(&opt)) {
        worker_opt = std::make_unique<RMSprop>(base_lr);
    } else if (dynamic_cast<RMSProp*>(&opt)) {
        worker_opt = std::make_unique<RMSProp>(base_lr);
    } else if (dynamic_cast<AdamW*>(&opt)) {
        worker_opt = std::make_unique<AdamW>(base_lr);
    } else if (dynamic_cast<SGDNesterov*>(&opt)) {
        worker_opt = std::make_unique<SGDNesterov>(base_lr);
    } else if (dynamic_cast<WeightDecay*>(&opt)) {
        // WeightDecay wraps an inner optimizer; clone the inner one wrapped
        WeightDecay* wd = dynamic_cast<WeightDecay*>(&opt);
        double wd_factor = wd->wd();
        Optimizer* inner = wd->inner();
        if (dynamic_cast<Adam*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new Adam(base_lr), wd_factor);
        } else if (dynamic_cast<SGD*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new SGD(base_lr), wd_factor);
        } else if (dynamic_cast<RMSprop*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new RMSprop(base_lr), wd_factor);
        } else if (dynamic_cast<RMSProp*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new RMSProp(base_lr), wd_factor);
        } else if (dynamic_cast<AdamW*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new AdamW(base_lr), wd_factor);
        } else if (dynamic_cast<SGDNesterov*>(inner)) {
            worker_opt = std::make_unique<WeightDecay>(new SGDNesterov(base_lr), wd_factor);
        } else {
            worker_opt = std::make_unique<Adam>(base_lr);
        }
    } else {
        worker_opt = std::make_unique<Adam>(base_lr);
    }

    // Batch size: use full dataset per step (all rows)
    // Cycling is handled by modulo when num_steps > dataset size
    size_t dataset_size = X.rows;
    size_t batch_size = dataset_size; // full-batch by default for LR range test

    double smoothed_loss = 0.0;
    bool smoothed_init = false;

    // Exponential LR ramp: lr = min_lr * (max_lr/min_lr)^(step/num_steps)
    double lr_ratio = max_lr_ / min_lr_;
    double lr_exponent_scale = std::log(lr_ratio) / static_cast<double>(num_steps_);

    for (int step = 0; step < num_steps_; ++step) {
        // Compute current learning rate
        double lr = min_lr_ * std::exp(lr_exponent_scale * static_cast<double>(step));
        // Clamp to max_lr to be safe
        if (lr > max_lr_) lr = max_lr_;

        // Update the worker optimizer's LR
        worker_opt->lr = lr;

        // Select batch via cycling
        size_t batch_start = (step * batch_size) % dataset_size;
        size_t batch_end = batch_start + batch_size;
        if (batch_end > dataset_size) batch_end = dataset_size;
        size_t actual_batch_size = batch_end - batch_start;

        // Extract batch from X and y via manual copy.
        // NOTE: Tensor does not yet expose a slice() view method, so we copy
        // each row individually. This is correct but O(N) per step; a future
        // Tensor::slice(start_row, end_row) would be more efficient.
        Tensor x_batch(actual_batch_size, X.cols);
        Tensor y_batch(actual_batch_size, y.cols);
        for (size_t i = 0; i < actual_batch_size; ++i) {
            for (size_t j = 0; j < X.cols; ++j) {
                x_batch[i][j] = X[batch_start + i][j];
            }
            for (size_t j = 0; j < y.cols; ++j) {
                y_batch[i][j] = y[batch_start + i][j];
            }
        }

        // Forward pass
        Tensor prediction = model.forward(x_batch);

        // MSE loss: L = (1/N) * sum((pred - label)^2)
        // NOTE: We use MSE here rather than the model's built-in loss function.
        // The LRFinder does not have access to the model's loss, so MSE is a
        // reasonable stand-in for the range test. Smith's original paper uses
        // a classification loss; MSE is mathematically valid but may give
        // slightly different suggested LR for models with other losses.
        // Gradient w.r.t. prediction: dL/dpred = 2 * (pred - label) / N
        Tensor loss_grad(prediction.rows, prediction.cols);
        double raw_loss = 0.0;
        for (size_t i = 0; i < prediction.rows; ++i) {
            for (size_t j = 0; j < prediction.cols; ++j) {
                double diff = prediction[i][j] - y_batch[i][j];
                raw_loss += diff * diff;
                loss_grad[i][j] = 2.0 * diff / static_cast<double>(prediction.rows);
            }
        }
        raw_loss /= static_cast<double>(prediction.rows);

        // Backward pass
        model.backward(loss_grad, lr);

        // Optimizer step
        worker_opt->step(model);

        // Zero gradients after update
        for (auto& layer : model.layers) {
            layer->zero_grad();
        }

        // Exponential moving average of loss
        if (!smoothed_init) {
            smoothed_loss = raw_loss;
            smoothed_init = true;
        } else {
            smoothed_loss = beta_ * smoothed_loss + (1.0 - beta_) * raw_loss;
        }

        // Handle numerical instability
        if (std::isnan(smoothed_loss) || std::isinf(smoothed_loss)) {
            // Stop early if loss explodes
            if (step > 0) {
                break;
            }
        }

        history_.push_back({lr, smoothed_loss});
    }

    return history_;
}

double LRFinder::suggested_lr() const {
    if (history_.size() < 2) return min_lr_;

    // Find point of steepest descent: largest negative change in smoothed loss
    double best_change = 0.0;
    int best_idx = 0;

    for (size_t i = 0; i + 1 < history_.size(); ++i) {
        double change = history_[i + 1].second - history_[i].second;
        if (change < best_change) {
            best_change = change;
            best_idx = i;
        }
    }

    // Return the LR at the start of the steepest descent segment
    return history_[best_idx].first;
}

double LRFinder::lr_at_min_loss() const {
    if (history_.empty()) return min_lr_;

    int min_idx = 0;
    double min_loss = history_[0].second;

    for (size_t i = 1; i < history_.size(); ++i) {
        if (history_[i].second < min_loss) {
            min_loss = history_[i].second;
            min_idx = i;
        }
    }

    return history_[min_idx].first;
}
