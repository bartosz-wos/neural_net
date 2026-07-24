#include "training_history.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

void TrainingHistory::record(size_t epoch, double train_loss, double val_loss,
                             double learning_rate) {
    if (!std::isfinite(train_loss) || !std::isfinite(val_loss) ||
        !std::isfinite(learning_rate) || learning_rate < 0.0) {
        throw std::invalid_argument(
            "TrainingHistory metrics must be finite and learning_rate non-negative");
    }
    if (!records_.empty() && epoch <= records_.back().epoch) {
        throw std::invalid_argument(
            "TrainingHistory epochs must be strictly increasing");
    }
    records_.push_back({epoch, train_loss, val_loss, learning_rate});
}

TrainingHistory::EpochCallback TrainingHistory::callback(double learning_rate) {
    if (!std::isfinite(learning_rate) || learning_rate < 0.0) {
        throw std::invalid_argument(
            "TrainingHistory callback learning_rate must be finite and non-negative");
    }
    return [this, learning_rate](int epoch, double train_loss, double val_loss) {
        if (epoch < 0) {
            throw std::invalid_argument("TrainingHistory epoch must be non-negative");
        }
        record(static_cast<size_t>(epoch), train_loss, val_loss, learning_rate);
    };
}

const TrainingEpochRecord& TrainingHistory::at(size_t index) const {
    return records_.at(index);
}

const TrainingEpochRecord& TrainingHistory::latest() const {
    if (records_.empty()) {
        throw std::runtime_error("TrainingHistory has no records");
    }
    return records_.back();
}

const TrainingEpochRecord& TrainingHistory::best_epoch(
    TrainingMetric metric, HistoryMode mode) const {
    if (records_.empty()) {
        throw std::runtime_error("TrainingHistory has no records");
    }

    size_t best = 0;
    const auto value = [metric](const TrainingEpochRecord& record) {
        return metric == TrainingMetric::TRAIN_LOSS
            ? record.train_loss
            : record.val_loss;
    };
    for (size_t i = 1; i < records_.size(); ++i) {
        const bool improves = mode == HistoryMode::MINIMIZE
            ? value(records_[i]) < value(records_[best])
            : value(records_[i]) > value(records_[best]);
        if (improves) best = i;
    }
    return records_[best];
}

std::string TrainingHistory::to_csv() const {
    std::ostringstream out;
    out << "epoch,train_loss,val_loss,learning_rate\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& record : records_) {
        out << record.epoch << ',' << record.train_loss << ',' << record.val_loss
            << ',' << record.learning_rate << '\n';
    }
    return out.str();
}

void TrainingHistory::save_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open training history CSV for writing: " + path);
    }
    out << to_csv();
    if (!out) {
        throw std::runtime_error("Failed to write training history CSV: " + path);
    }
}
