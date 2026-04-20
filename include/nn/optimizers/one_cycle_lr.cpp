#include "one_cycle_lr.h"
#include <cmath>
#include <algorithm>

OneCycleLR::OneCycleLR(double max_lr, size_t total_steps,
                        double min_lr, const std::string& anneal_strategy)
    : max_lr_(max_lr), min_lr_(min_lr),
      total_steps_(total_steps),
      warmup_steps_(total_steps / 3),
      anneal_strategy_(anneal_strategy),
      step_(0) {
    if (total_steps < 3) warmup_steps_ = 0;  // guard div/0
}

void OneCycleLR::step() {
    ++step_;
}

double OneCycleLR::get_lr() const {
    double s = static_cast<double>(step_);

    if (warmup_steps_ == 0) {
        // No warmup: start anneal immediately. First step (s=1) should be at max_lr,
        // so pct starts at 0 and ends at 1 at the final step (s=total_steps).
        if (s >= total_steps_) return min_lr_;
        double pct = (s - warmup_steps_) / static_cast<double>(total_steps_ - warmup_steps_);
        if (anneal_strategy_ == "cos") {
            double cos_val = (1.0 + std::cos(std::acos(-1.0) * pct)) / 2.0;
            return min_lr_ + (max_lr_ - min_lr_) * cos_val;
        } else {
            return min_lr_;
        }
    }

    if (s <= warmup_steps_) {
        // Warmup phase: linear increase
        return min_lr_ + (max_lr_ - min_lr_) * s / warmup_steps_;
    } else {
        // Anneal phase
        double pct = (s - warmup_steps_) / static_cast<double>(total_steps_ - warmup_steps_);
        if (anneal_strategy_ == "cos") {
            double cos_val = (1.0 + std::cos(std::acos(-1.0) * pct)) / 2.0;
            return min_lr_ + (max_lr_ - min_lr_) * cos_val;
        } else {
            double lin = max_lr_ - (max_lr_ - min_lr_) * std::clamp(pct, 0.0, 1.0);
            return lin;
        }
    }
}
