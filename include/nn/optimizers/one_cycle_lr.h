#ifndef ONE_CYCLE_LR_H
#define ONE_CYCLE_LR_H

#include "scheduler.h"

// OneCycleLR: super-convergence learning rate schedule.
// Tracks step internally (call step() to advance).
// Phase 1 (warmup): lr grows from min_lr → max_lr over ~30% of total_steps
// Phase 2 (anneal): lr decays from max_lr → min_lr following cosine path
class OneCycleLR : public LRScheduler {
public:
    OneCycleLR(double max_lr, size_t total_steps,
               double min_lr = 1e-7, const std::string& anneal_strategy = "cos");
    void step(Model& model) override { ++step_; apply(model); }
    double get_lr() const override;
    void apply(Model& model);  // overrides LRScheduler::apply to call notify_optimizer
    void reset() { step_ = 0; }

private:
    double max_lr_, min_lr_;
    size_t total_steps_, warmup_steps_;
    std::string anneal_strategy_;
    mutable size_t step_;
};

#endif
