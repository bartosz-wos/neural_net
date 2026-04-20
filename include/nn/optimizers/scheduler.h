#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "optimizer.h"

// Base scheduler: just wraps an Optimizer with LR adjustment
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual double get_lr() const = 0;
    virtual void step() = 0;
    virtual void apply(Model& model) = 0;
};

// StepLR: decay LR by gamma every step_size epochs
class StepLR : public LRScheduler {
    double initial_lr, lr, gamma;
    int step_size, epoch;
public:
    StepLR(double lr, int step_size = 10, double gamma = 0.1)
        : initial_lr(lr), lr(lr), gamma(gamma), step_size(step_size), epoch(0) {}
    double get_lr() const override { return lr; }
    void step() override { ++epoch; if (epoch % step_size == 0) lr *= gamma; }
    void apply(Model& model) override {}
};

// ExponentialLR: lr = initial_lr * gamma^epoch
class ExponentialLR : public LRScheduler {
    double initial_lr, lr, gamma;
    int epoch;
public:
    ExponentialLR(double lr, double gamma = 0.95)
        : initial_lr(lr), lr(lr), gamma(gamma), epoch(0) {}
    double get_lr() const override { return lr; }
    void step() override { ++epoch; lr = initial_lr * std::pow(gamma, epoch); }
    void apply(Model& model) override {}
};

// ReduceLROnPlateau: reduce LR when metric stops improving
class ReduceLROnPlateau : public LRScheduler {
    double lr, factor, min_lr;
    int patience, counter;
    double best_metric;
    bool first;
public:
    ReduceLROnPlateau(double lr, double factor = 0.5, int patience = 5, double min_lr = 1e-6)
        : lr(lr), factor(factor), min_lr(min_lr), patience(patience), counter(0), best_metric(1e9), first(true) {}
    double get_lr() const override { return lr; }
    void step() override { ++counter; }
    // Call this with current metric to check for improvement
    void check_metric(double metric) {
        if (first || metric < best_metric) {
            best_metric = metric;
            counter = 0;
            first = false;
        } else {
            ++counter;
            if (counter >= patience) {
                lr = std::max(lr * factor, min_lr);
                counter = 0;
            }
        }
    }
    void apply(Model& model) override {}
};

// CosineAnnealingLR: lr decays along cosine curve from initial_lr to min_lr
// T_max: number of epochs for one full cycle
// eta_min: minimum learning rate (default 0)
class CosineAnnealingLR : public LRScheduler {
    double initial_lr, lr, min_lr;
    int T_max, epoch;
public:
    CosineAnnealingLR(double lr, int T_max, double min_lr = 0.0)
        : initial_lr(lr), lr(lr), min_lr(min_lr), T_max(T_max), epoch(0) {}
    double get_lr() const override { return lr; }
    void step() override {
        ++epoch;
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + std::cos(std::acos(-1.0) * epoch / T_max));
    }
    void apply(Model& model) override {}
};

// EarlyStopping: monitors training loss and reverts to best model after patience epochs.
// min_delta: minimum improvement required to reset patience
// best_path: temporary file to store the best model checkpoint
// Usage: after each epoch, call check(loss, model) — returns true if early stopping triggered.
class EarlyStopping {
    double min_delta_;
    int patience_, counter_;
    double best_loss_;
    bool first_;
    std::string best_path_;
public:
    EarlyStopping(double min_delta = 1e-4, int patience = 10, const std::string& best_path = "best_model.nn")
        : min_delta_(min_delta), patience_(patience), counter_(0), best_loss_(1e9), first_(true), best_path_(best_path) {}

    // Returns true if early stopping triggered (model already reverted)
    bool check(double loss, Model& model) {
        if (first_ || loss < best_loss_ - min_delta_) {
            best_loss_ = loss;
            counter_ = 0;
            first_ = false;
            model.save(best_path_);  // checkpoint best model
        } else {
            ++counter_;
            if (counter_ >= patience_) {
                try {
                    Model best = Model::load(best_path_);
                    // Replace layers
                    model.layers = std::move(best.layers);
                } catch (...) {}
                return true;
            }
        }
        return false;
    }

    double best_loss() const { return best_loss_; }
    void reset() { best_loss_ = 1e9; counter_ = 0; first_ = true; }
};

#endif