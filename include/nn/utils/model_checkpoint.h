#ifndef MODEL_CHECKPOINT_H
#define MODEL_CHECKPOINT_H

#include "../core/model.h"
#include <cstddef>
#include <functional>
#include <string>

// Direction in which the monitored checkpoint metric improves.
enum class ModelCheckpointMode {
    MINIMIZE,
    MAXIMIZE
};

// Generic parameter-only checkpointing for an already-constructed Model.
// The binary format stores every Layer::parameters() tensor in stable order;
// load() verifies the complete topology before mutating the destination model.
class ModelCheckpoint {
public:
    using EpochCallback = std::function<void(int, double, double)>;

    explicit ModelCheckpoint(
        std::string path,
        bool save_best_only = true,
        ModelCheckpointMode mode = ModelCheckpointMode::MINIMIZE,
        double min_delta = 0.0);

    bool step(size_t epoch, double metric, Model& model);
    EpochCallback callback(Model& model);
    void save(Model& model);
    void load(Model& model) const;
    void reset();

    const std::string& path() const { return path_; }
    bool save_best_only() const { return save_best_only_; }
    ModelCheckpointMode mode() const { return mode_; }
    double min_delta() const { return min_delta_; }

    bool has_best() const { return has_best_; }
    double best_metric() const { return best_metric_; }
    size_t best_epoch() const { return best_epoch_; }
    size_t num_steps() const { return num_steps_; }
    size_t num_saved() const { return num_saved_; }

private:
    std::string path_;
    bool save_best_only_;
    ModelCheckpointMode mode_;
    double min_delta_;

    bool has_best_ = false;
    double best_metric_ = 0.0;
    size_t best_epoch_ = 0;
    size_t num_steps_ = 0;
    size_t num_saved_ = 0;

    bool is_improvement(double metric) const;
};

#endif
