#include "model_checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr char CHECKPOINT_MAGIC[] = {'N', 'N', 'P', 'A', 'R', 'M'};
constexpr uint8_t CHECKPOINT_VERSION = 1;

std::vector<Tensor*> collect_parameters(Model& model) {
    std::vector<Tensor*> parameters;
    for (auto& layer : model.layers) {
        std::vector<Tensor*> layer_parameters = layer->parameters();
        for (Tensor* parameter : layer_parameters) {
            if (parameter == nullptr) {
                throw std::runtime_error("ModelCheckpoint: model exposes a null parameter");
            }
            parameters.push_back(parameter);
        }
    }
    return parameters;
}

void require_read(std::ifstream& input, char* destination, std::streamsize size,
                  const std::string& path) {
    if (size == 0) return;
    input.read(destination, size);
    if (!input) {
        throw std::runtime_error("ModelCheckpoint: truncated checkpoint: " + path);
    }
}

template <typename T>
void read_scalar(std::ifstream& input, T& value, const std::string& path) {
    require_read(input, reinterpret_cast<char*>(&value), sizeof(T), path);
}

template <typename T>
void write_scalar(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

ModelCheckpoint::ModelCheckpoint(std::string path, bool save_best_only,
                                 ModelCheckpointMode mode, double min_delta)
    : path_(std::move(path)),
      save_best_only_(save_best_only),
      mode_(mode),
      min_delta_(min_delta) {
    if (path_.empty()) {
        throw std::invalid_argument("ModelCheckpoint: path must not be empty");
    }
    if (!std::isfinite(min_delta_) || min_delta_ < 0.0) {
        throw std::invalid_argument(
            "ModelCheckpoint: min_delta must be finite and >= 0");
    }
}

bool ModelCheckpoint::is_improvement(double metric) const {
    if (!has_best_) return true;
    if (mode_ == ModelCheckpointMode::MINIMIZE) {
        return metric < best_metric_ - min_delta_;
    }
    return metric > best_metric_ + min_delta_;
}

bool ModelCheckpoint::step(size_t epoch, double metric, Model& model) {
    if (!std::isfinite(metric)) {
        throw std::invalid_argument("ModelCheckpoint: metric must be finite");
    }

    const bool improves = is_improvement(metric);
    const bool should_save = !save_best_only_ || improves;
    if (should_save) save(model);

    ++num_steps_;
    if (improves) {
        has_best_ = true;
        best_metric_ = metric;
        best_epoch_ = epoch;
    }
    return should_save;
}

ModelCheckpoint::EpochCallback ModelCheckpoint::callback(Model& model) {
    return [this, &model](int epoch, double, double val_loss) {
        if (epoch < 0) {
            throw std::invalid_argument("ModelCheckpoint: epoch must be non-negative");
        }
        step(static_cast<size_t>(epoch), val_loss, model);
    };
}

void ModelCheckpoint::save(Model& model) {
    const std::vector<Tensor*> parameters = collect_parameters(model);
    const std::string temporary_path = path_ + ".tmp";

    try {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "ModelCheckpoint: cannot open temporary checkpoint for writing: " +
                temporary_path);
        }

        output.write(CHECKPOINT_MAGIC, sizeof(CHECKPOINT_MAGIC));
        write_scalar(output, CHECKPOINT_VERSION);
        const uint64_t count = static_cast<uint64_t>(parameters.size());
        write_scalar(output, count);

        for (const Tensor* parameter : parameters) {
            const uint64_t rows = static_cast<uint64_t>(parameter->rows);
            const uint64_t cols = static_cast<uint64_t>(parameter->cols);
            write_scalar(output, rows);
            write_scalar(output, cols);
            if (!parameter->data.empty()) {
                output.write(
                    reinterpret_cast<const char*>(parameter->data.data()),
                    static_cast<std::streamsize>(sizeof(double) * parameter->data.size()));
            }
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "ModelCheckpoint: failed to write temporary checkpoint: " +
                temporary_path);
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "ModelCheckpoint: failed to close temporary checkpoint: " +
                temporary_path);
        }

        if (std::rename(temporary_path.c_str(), path_.c_str()) != 0) {
            throw std::runtime_error(
                "ModelCheckpoint: failed to replace checkpoint: " + path_);
        }
    } catch (...) {
        std::remove(temporary_path.c_str());
        throw;
    }
    ++num_saved_;
}

void ModelCheckpoint::load(Model& model) const {
    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "ModelCheckpoint: cannot open checkpoint for reading: " + path_);
    }

    char magic[sizeof(CHECKPOINT_MAGIC)] = {};
    require_read(input, magic, sizeof(magic), path_);
    for (size_t index = 0; index < sizeof(CHECKPOINT_MAGIC); ++index) {
        if (magic[index] != CHECKPOINT_MAGIC[index]) {
            throw std::runtime_error("ModelCheckpoint: invalid checkpoint magic: " + path_);
        }
    }

    uint8_t version = 0;
    read_scalar(input, version, path_);
    if (version != CHECKPOINT_VERSION) {
        throw std::runtime_error("ModelCheckpoint: unsupported checkpoint version");
    }

    uint64_t count = 0;
    read_scalar(input, count, path_);
    const std::vector<Tensor*> parameters = collect_parameters(model);
    if (count != static_cast<uint64_t>(parameters.size())) {
        throw std::runtime_error(
            "ModelCheckpoint: model parameter count does not match checkpoint");
    }

    std::vector<Tensor> loaded;
    loaded.reserve(parameters.size());
    for (size_t index = 0; index < parameters.size(); ++index) {
        uint64_t rows64 = 0;
        uint64_t cols64 = 0;
        read_scalar(input, rows64, path_);
        read_scalar(input, cols64, path_);
        const Tensor* expected = parameters[index];
        if (rows64 != static_cast<uint64_t>(expected->rows) ||
            cols64 != static_cast<uint64_t>(expected->cols)) {
            throw std::runtime_error(
                "ModelCheckpoint: model parameter shape does not match checkpoint");
        }

        Tensor tensor(expected->rows, expected->cols);
        if (!tensor.data.empty()) {
            const size_t bytes = sizeof(double) * tensor.data.size();
            if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error("ModelCheckpoint: tensor payload is too large");
            }
            require_read(input, reinterpret_cast<char*>(tensor.data.data()),
                         static_cast<std::streamsize>(bytes), path_);
        }
        loaded.push_back(std::move(tensor));
    }

    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("ModelCheckpoint: checkpoint contains trailing data");
    }
    if (!input.eof()) {
        throw std::runtime_error("ModelCheckpoint: failed while reading checkpoint");
    }

    for (size_t index = 0; index < parameters.size(); ++index) {
        parameters[index]->data = loaded[index].data;
    }
}

void ModelCheckpoint::reset() {
    has_best_ = false;
    best_metric_ = 0.0;
    best_epoch_ = 0;
    num_steps_ = 0;
    num_saved_ = 0;
}
