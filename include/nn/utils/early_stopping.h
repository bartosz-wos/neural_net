#ifndef EARLY_STOPPING_H
#define EARLY_STOPPING_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <cstddef>
#include <vector>

// Direction in which the monitored metric improves.
enum class InMemoryEarlyStoppingMode {
    MINIMIZE,
    MAXIMIZE
};

// Callback-friendly early stopping with an in-memory best-parameter snapshot.
//
// Call step(metric, model) once per validation epoch and stop training when it
// returns true. A metric must improve by strictly more than min_delta to reset
// patience. If restore_best_weights is enabled, the best snapshot is restored
// exactly once when patience is exhausted.
class InMemoryEarlyStopping {
public:
    explicit InMemoryEarlyStopping(size_t patience = 5,
                           double min_delta = 0.0,
                           InMemoryEarlyStoppingMode mode = InMemoryEarlyStoppingMode::MINIMIZE,
                           bool restore_best_weights = true);

    bool step(double metric, Model& model);
    void restore_best(Model& model) const;
    void reset();

    size_t patience() const { return patience_; }
    double min_delta() const { return min_delta_; }
    InMemoryEarlyStoppingMode mode() const { return mode_; }
    bool restore_best_weights() const { return restore_best_weights_; }

    bool has_best() const { return has_best_; }
    bool stopped() const { return stopped_; }
    double best_metric() const { return best_metric_; }
    size_t best_step() const { return best_step_; }
    size_t num_steps() const { return num_steps_; }
    size_t num_bad_epochs() const { return num_bad_epochs_; }
    size_t num_snapshot_params() const { return best_parameters_.size(); }

private:
    size_t patience_;
    double min_delta_;
    InMemoryEarlyStoppingMode mode_;
    bool restore_best_weights_;

    bool has_best_ = false;
    bool stopped_ = false;
    double best_metric_ = 0.0;
    size_t best_step_ = 0;
    size_t num_steps_ = 0;
    size_t num_bad_epochs_ = 0;
    std::vector<Tensor> best_parameters_;

    bool is_improvement(double metric) const;
    void snapshot(const Model& model);
    static std::vector<Tensor*> collect_parameters(Model& model);
};

#endif
