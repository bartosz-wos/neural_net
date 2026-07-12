#include "model_ema.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <cstddef>

// =============================================================================
// ModelEMA — Exponential Moving Average of model parameters.
// See model_ema.h for the algorithm and API documentation.
// =============================================================================

ModelEMA::ModelEMA(Model& model, double decay) : decay_(decay) {
    if (decay < 0.0 || decay > 1.0) {
        throw std::invalid_argument(
            "ModelEMA: decay must be in [0, 1], got " + std::to_string(decay));
    }
    std::vector<Tensor*> params = collect_params(model);
    shadow_.reserve(params.size());
    for (Tensor* p : params) {
        // Deep-copy the current parameter into the shadow buffer.
        shadow_.push_back(Tensor(p->rows, p->cols));
        copy_into(*p, shadow_.back());
    }
}

std::vector<Tensor*> ModelEMA::collect_params(Model& model) {
    std::vector<Tensor*> params;
    for (auto& layer : model.layers) {
        for (Tensor* p : layer->parameters()) {
            params.push_back(p);
        }
    }
    return params;
}

void ModelEMA::copy_into(const Tensor& src, Tensor& dst) {
    // dst.rows/cols should already match src.rows/cols from the caller.
    for (size_t i = 0; i < src.rows; ++i) {
        for (size_t j = 0; j < src.cols; ++j) {
            dst[i][j] = src[i][j];
        }
    }
}

void ModelEMA::ema_combine_into(const Tensor& src, Tensor& dst, double decay) {
    // dst[i][j] = decay * dst[i][j] + (1 - decay) * src[i][j]
    const double one_minus = 1.0 - decay;
    for (size_t i = 0; i < src.rows; ++i) {
        for (size_t j = 0; j < src.cols; ++j) {
            dst[i][j] = decay * dst[i][j] + one_minus * src[i][j];
        }
    }
}

void ModelEMA::step(Model& model) {
    std::vector<Tensor*> params = collect_params(model);
    // Defensive: if the model added or removed parameters since init, refuse
    // silently-silent corruption by throwing. Callers must reinitialize().
    if (params.size() != shadow_.size()) {
        throw std::logic_error(
            "ModelEMA::step: parameter count changed since initialization "
            "(was " + std::to_string(shadow_.size()) +
            ", now " + std::to_string(params.size()) +
            "). Call reinitialize(model) to rebuild the EMA state.");
    }
    for (size_t k = 0; k < params.size(); ++k) {
        Tensor* p = params[k];
        // Sanity: shape must match the shadow.
        if (p->rows != shadow_[k].rows || p->cols != shadow_[k].cols) {
            throw std::logic_error(
                "ModelEMA::step: parameter " + std::to_string(k) +
                " shape changed (was " + std::to_string(shadow_[k].rows) +
                "x" + std::to_string(shadow_[k].cols) +
                ", now " + std::to_string(p->rows) + "x" + std::to_string(p->cols) +
                "). Call reinitialize(model) to rebuild the EMA state.");
        }
        ema_combine_into(*p, shadow_[k], decay_);
    }
    ++step_count_;
}

void ModelEMA::reinitialize(Model& model) {
    std::vector<Tensor*> params = collect_params(model);
    shadow_.clear();
    shadow_.reserve(params.size());
    for (Tensor* p : params) {
        shadow_.push_back(Tensor(p->rows, p->cols));
        copy_into(*p, shadow_.back());
    }
    // Wipe any stored backup; the model architecture changed.
    stored_.clear();
    has_stored_ = false;
    step_count_ = 0;
}

void ModelEMA::store(Model& model) {
    std::vector<Tensor*> params = collect_params(model);
    if (stored_.size() != params.size()) {
        stored_.clear();
        stored_.reserve(params.size());
        for (size_t k = 0; k < params.size(); ++k) {
            stored_.push_back(Tensor(params[k]->rows, params[k]->cols));
        }
    }
    for (size_t k = 0; k < params.size(); ++k) {
        copy_into(*params[k], stored_[k]);
    }
    has_stored_ = true;
}

void ModelEMA::copy_to(Model& model) const {
    std::vector<Tensor*> params;
    for (auto& layer : model.layers) {
        for (Tensor* p : layer->parameters()) {
            params.push_back(p);
        }
    }
    if (params.size() != shadow_.size()) {
        throw std::logic_error(
            "ModelEMA::copy_to: parameter count mismatch (shadow=" +
            std::to_string(shadow_.size()) + ", model=" +
            std::to_string(params.size()) + "). Call reinitialize(model).");
    }
    for (size_t k = 0; k < params.size(); ++k) {
        Tensor* p = params[k];
        if (p->rows != shadow_[k].rows || p->cols != shadow_[k].cols) {
            throw std::logic_error(
                "ModelEMA::copy_to: parameter " + std::to_string(k) +
                " shape mismatch.");
        }
        copy_into(shadow_[k], *p);
    }
}

void ModelEMA::apply_shadow(Model& model) {
    store(model);
    copy_to(model);
}

void ModelEMA::restore(Model& model) {
    if (!has_stored_) {
        throw std::logic_error(
            "ModelEMA::restore: no stored weights to restore from. "
            "Call store(model) or apply_shadow(model) first.");
    }
    std::vector<Tensor*> params;
    for (auto& layer : model.layers) {
        for (Tensor* p : layer->parameters()) {
            params.push_back(p);
        }
    }
    if (params.size() != stored_.size()) {
        throw std::logic_error(
            "ModelEMA::restore: parameter count mismatch between stored "
            "backup and current model. The model architecture must not "
            "change between store() and restore().");
    }
    for (size_t k = 0; k < params.size(); ++k) {
        copy_into(stored_[k], *params[k]);
    }
}

Tensor ModelEMA::get_shadow(size_t i) const {
    if (i >= shadow_.size()) {
        throw std::out_of_range("ModelEMA::get_shadow: index out of range");
    }
    // Return a deep copy so callers can't mutate internal state.
    Tensor out(shadow_[i].rows, shadow_[i].cols);
    copy_into(shadow_[i], out);
    return out;
}

Tensor ModelEMA::get_stored(size_t i) const {
    if (!has_stored_) {
        return Tensor(0, 0);
    }
    if (i >= stored_.size()) {
        throw std::out_of_range("ModelEMA::get_stored: index out of range");
    }
    Tensor out(stored_[i].rows, stored_[i].cols);
    copy_into(stored_[i], out);
    return out;
}