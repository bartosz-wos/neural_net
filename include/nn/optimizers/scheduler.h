#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "optimizer.h"
#include "../core/model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// Base scheduler: just wraps an Optimizer with LR adjustment
class LRScheduler {
protected:
    Optimizer* optimizer_ = nullptr;
public:
    virtual ~LRScheduler() = default;
    virtual double get_lr() const = 0;
    virtual void step(Model& model) = 0;
    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }
    void apply(Model& model) { (void)model; if (optimizer_) optimizer_->lr = get_lr(); }
};

// StepLR: decay LR by gamma every step_size epochs
class StepLR : public LRScheduler {
    double initial_lr, lr, gamma;
    int step_size, epoch;
public:
    StepLR(double lr, int step_size = 10, double gamma = 0.1)
        : initial_lr(lr), lr(lr), gamma(gamma), step_size(step_size), epoch(0) {}
    double get_lr() const override { return lr; }
    void step(Model& model) override { ++epoch; if (epoch % step_size == 0) lr *= gamma; apply(model); }
};

// ExponentialLR: lr = initial_lr * gamma^epoch
class ExponentialLR : public LRScheduler {
    double initial_lr, lr, gamma;
    int epoch;
public:
    ExponentialLR(double lr, double gamma = 0.95)
        : initial_lr(lr), lr(lr), gamma(gamma), epoch(0) {}
    double get_lr() const override { return lr; }
    void step(Model& model) override { ++epoch; lr = initial_lr * std::pow(gamma, epoch); apply(model); }
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
    void step(Model& model) override { ++counter; apply(model); }
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
                if (optimizer_) optimizer_->lr = lr;
            }
        }
    }
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
    void step(Model& model) override {
        ++epoch;
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + std::cos(std::acos(-1.0) * epoch / T_max));
        apply(model);
    }
};

// CosineAnnealingWarmRestarts (SGDR — Loshchilov & Hutter 2017).
// lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(π * T_cur / T_i))
// where T_i = T_0 * T_mult^(i-1) for the i-th cycle (T_0 first, then doubles if T_mult=2).
// T_cur resets to 0 after each cycle. Standard "warm restart" pattern.
class CosineAnnealingWarmRestarts : public LRScheduler {
    double eta_max_, eta_min_, T_0_;
    int T_mult_;
    int last_epoch_;        // = T_cur (epoch within current cycle)
    int cycle_index_;       // which cycle we are in (0-indexed)
    double current_lr_;     // cached current LR
    // Cycle length for the i-th cycle (i = cycle_index_): T_0 * T_mult^i
    int cycle_length_() const {
        int len = T_0_;
        for (int i = 0; i < cycle_index_; ++i) len *= T_mult_;
        return len;
    }
    void recompute_() {
        double T_cur = static_cast<double>(last_epoch_);
        double T_i = static_cast<double>(cycle_length_());
        double cos_val = std::cos(M_PI * T_cur / T_i);
        current_lr_ = eta_min_ + 0.5 * (eta_max_ - eta_min_) * (1.0 + cos_val);
    }
public:
    CosineAnnealingWarmRestarts(double eta_max, int T_0,
                                int T_mult = 1, double eta_min = 0.0)
        : eta_max_(eta_max), eta_min_(eta_min), T_0_(T_0 <= 0 ? 1 : T_0),
          T_mult_(T_mult <= 1 ? 1 : T_mult),
          last_epoch_(0), cycle_index_(0), current_lr_(eta_max) {
        recompute_();  // initial lr at T_cur=0
    }
    double get_lr() const override { return current_lr_; }
    void step(Model& model) override {
        step_dry();
        apply(model);
    }

    // Advance epoch counter and recompute LR, but do NOT touch the optimizer
    // (for unit tests that want to inspect lr directly).
    void step_dry() {
        ++last_epoch_;
        // Hit end of current cycle? Reset T_cur and bump cycle_index_.
        if (last_epoch_ >= cycle_length_()) {
            last_epoch_ = 0;
            ++cycle_index_;
        }
        recompute_();
    }

    // Accessors
    double get_eta_max() const { return eta_max_; }
    double get_eta_min() const { return eta_min_; }
    int get_T_0() const { return T_0_; }
    int get_T_mult() const { return T_mult_; }
    int get_last_epoch() const { return last_epoch_; }
    int get_cycle_index() const { return cycle_index_; }
};

// MultiStepLR: decays LR by gamma at each milestone (epoch in the supplied list).
// After all milestones, LR stays at initial_lr * gamma^k where k = #milestones.
// std::sort is applied to the milestones internally so callers may pass them unsorted.
class MultiStepLR : public LRScheduler {
    double initial_lr_, lr_, gamma_;
    std::vector<int> milestones_;  // sorted ascending
    int last_epoch_;
public:
    MultiStepLR(double lr, std::vector<int> milestones, double gamma = 0.1)
        : initial_lr_(lr), lr_(lr), gamma_(gamma),
          last_epoch_(0) {
        std::sort(milestones.begin(), milestones.end());
        milestones_ = std::move(milestones);
    }
    double get_lr() const override { return lr_; }
    void step(Model& model) override {
        step_dry();
        apply(model);
    }

    void step_dry() {
        ++last_epoch_;
        if (std::binary_search(milestones_.begin(), milestones_.end(), last_epoch_))
            lr_ *= gamma_;
    }

    // Accessors
    const std::vector<int>& get_milestones() const { return milestones_; }
    double get_gamma() const { return gamma_; }
    double get_initial_lr() const { return initial_lr_; }
    int get_last_epoch() const { return last_epoch_; }
};

// PolynomialLR: lr = (initial - end) * (1 - epoch/max_epoch)^power + end
// (power=1 -> linear, power=2 -> quadratic, etc.). After max_epoch steps, lr=end.
// This is the same formula as torchvision's PolynomialLR.
class PolynomialLR : public LRScheduler {
    double initial_lr_, end_lr_, power_;
    int max_epoch_;
    int last_epoch_;
    double current_lr_;
public:
    PolynomialLR(double initial_lr, int max_epoch, double power = 1.0, double end_lr = 0.0)
        : initial_lr_(initial_lr), end_lr_(end_lr), power_(power),
          max_epoch_(max_epoch <= 0 ? 1 : max_epoch),
          last_epoch_(0), current_lr_(initial_lr) {}
    double get_lr() const override { return current_lr_; }
    void step(Model& model) override {
        step_dry();
        apply(model);
    }

    void step_dry() {
        ++last_epoch_;
        double t = std::min(static_cast<double>(last_epoch_) /
                            static_cast<double>(max_epoch_), 1.0);
        current_lr_ = (initial_lr_ - end_lr_) * std::pow(1.0 - t, power_) + end_lr_;
    }

    // Accessors
    double get_initial_lr() const { return initial_lr_; }
    double get_end_lr() const { return end_lr_; }
    double get_power() const { return power_; }
    int get_max_epoch() const { return max_epoch_; }
    int get_last_epoch() const { return last_epoch_; }
};

// CyclicLR — Smith 2017 ("Cyclical Learning Rates for Training Neural Networks").
// Oscillates LR between base_lr and max_lr in cycles of length 2*step_size.
// Three policies:
//   TRIANGULAR  — basic triangle wave (default)
//   TRIANGULAR2 — amplitude halves each cycle (so max_lr shrinks each cycle)
//   EXP_RANGE   — amplitude decays geometrically by gamma^step
// Standard triangular formula (from Smith 2017 §3.1):
//   cycle = floor(t / (2 * step_size))
//   x = |t / step_size - 2 * cycle - 1|
//   lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)
// For TRIANGULAR2, multiply (max_lr - base_lr) by 1 / 2^cycle.
// For EXP_RANGE, multiply (max_lr - base_lr) * max(0, 1 - x) by gamma^t.
class CyclicLR : public LRScheduler {
public:
    enum class Policy { TRIANGULAR, TRIANGULAR2, EXP_RANGE };
private:
    double base_lr_, max_lr_, gamma_;
    int step_size_;
    Policy policy_;
    int last_step_;  // global step counter
    double current_lr_;
    void recompute_() {
        int t = last_step_;
        int cycle = t / (2 * step_size_);
        double x = std::fabs(static_cast<double>(t) / step_size_ -
                             2.0 * cycle - 1.0);
        double scale = std::max(0.0, 1.0 - x);
        double amp = max_lr_ - base_lr_;
        if (policy_ == Policy::TRIANGULAR2) {
            amp /= std::pow(2.0, cycle);
        } else if (policy_ == Policy::EXP_RANGE) {
            scale *= std::pow(gamma_, t);
        }
        current_lr_ = base_lr_ + amp * scale;
    }
public:
    CyclicLR(double base_lr, double max_lr, int step_size,
             Policy policy = Policy::TRIANGULAR, double gamma = 0.99994)
        : base_lr_(base_lr), max_lr_(max_lr), gamma_(gamma),
          step_size_(step_size <= 0 ? 1 : step_size),
          policy_(policy), last_step_(0), current_lr_(max_lr) {
        recompute_();  // initial lr at t=0
    }
    double get_lr() const override { return current_lr_; }
    void step(Model& model) override {
        step_dry();
        apply(model);
    }

    void step_dry() {
        ++last_step_;
        recompute_();
    }

    // Accessors
    double get_base_lr() const { return base_lr_; }
    double get_max_lr() const { return max_lr_; }
    int get_step_size() const { return step_size_; }
    Policy get_policy() const { return policy_; }
    double get_gamma() const { return gamma_; }
    int get_last_step() const { return last_step_; }
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