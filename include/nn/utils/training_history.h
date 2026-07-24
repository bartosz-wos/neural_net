#ifndef TRAINING_HISTORY_H
#define TRAINING_HISTORY_H

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct TrainingEpochRecord {
    size_t epoch;
    double train_loss;
    double val_loss;
    double learning_rate;
};

enum class TrainingMetric {
    TRAIN_LOSS,
    VAL_LOSS
};

enum class HistoryMode {
    MINIMIZE,
    MAXIMIZE
};

// Ordered in-memory epoch metrics with callback integration and CSV export.
class TrainingHistory {
public:
    using EpochCallback = std::function<void(int, double, double)>;

    void record(size_t epoch, double train_loss, double val_loss,
                double learning_rate = 0.0);
    EpochCallback callback(double learning_rate = 0.0);

    bool empty() const { return records_.empty(); }
    size_t size() const { return records_.size(); }
    const std::vector<TrainingEpochRecord>& records() const { return records_; }
    const TrainingEpochRecord& at(size_t index) const;
    const TrainingEpochRecord& latest() const;
    const TrainingEpochRecord& best_epoch(
        TrainingMetric metric,
        HistoryMode mode = HistoryMode::MINIMIZE) const;

    std::string to_csv() const;
    void save_csv(const std::string& path) const;
    void clear() { records_.clear(); }

private:
    std::vector<TrainingEpochRecord> records_;
};

#endif
