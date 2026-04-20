#ifndef SWA_H
#define SWA_H

#include "../core/layer.h"
#include "optimizer.h"
#include <vector>

class SWAOptimizer : public Optimizer {
public:
    SWAOptimizer(Optimizer* inner, size_t start_after_steps = 0);
    void step(Model& model) override;
    void swap_to_averaged(Model& model);
    void record(Model& model);
    size_t averaged_count() const { return step_count_ > start_after_ ? step_count_ - start_after_ : 0; }

private:
    std::unique_ptr<Optimizer> inner_;
    size_t start_after_;
    size_t step_count_;
    std::vector<Tensor> averaged_weights_; // one Tensor per parameter
    std::vector<Tensor> shadow_weights_;   // one Tensor per parameter
    bool initialized_;
    size_t param_total_;
    void init_if_needed(const std::vector<Tensor*>& params);
};

class SWALRScheduler {
public:
    SWALRScheduler(double start_lr, double swa_lr, size_t warmup_steps = 0,
                   size_t swa_start_step = 0);
    double get_lr() const { return current_lr_; }
    void step() { ++step_count_; update_lr(); }
    void reset() { step_count_ = 0; current_lr_ = start_lr_; }

private:
    double start_lr_, swa_lr_, current_lr_;
    size_t warmup_steps_, swa_start_step_, step_count_;
    void update_lr();
};

#endif