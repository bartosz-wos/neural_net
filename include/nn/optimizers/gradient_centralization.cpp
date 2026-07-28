#include "gradient_centralization.h"
#include "../core/layer.h"
#include <stdexcept>
#include <algorithm>

GradientCentralization::GradientCentralization(Optimizer* inner, CenterMode mode)
    : inner_(inner), mode_(mode) {
    if (!inner_) {
        throw std::invalid_argument(
            "GradientCentralization: inner optimizer must not be null");
    }
    Optimizer::lr = inner_->lr;
}

bool GradientCentralization::handles_weight_decay() const {
    if (!inner_) return false;
    return inner_->handles_weight_decay();
}

void GradientCentralization::center_gradient(Tensor& grad) const {
    if (grad.rows == 0 || grad.cols == 0) return;
    if (grad.rows == 1 && grad.cols == 1) {
        // Scalar: mean = value, grad - value = 0 (no-op effectively, but we write
        // the zero anyway so the inner sees a zero gradient).
        grad[0][0] = 0.0;
        return;
    }

    if (grad.rows == 1) {
        // 1-D tensor (shape (1, C)): treat as COLUMN mode (single row).
        // mean[j] = grad[0][j] (it's the only row), so the centered gradient is 0.
        // This is the same as the scalar case generalized: row-mean of a single row
        // is the row itself, so center-step zeros it out. We don't need to do
        // anything special here — but write zeros explicitly for clarity.
        for (size_t j = 0; j < grad.cols; ++j) grad[0][j] = 0.0;
        return;
    }

    // 2-D tensor (R, C) with R >= 2.
    if (mode_ == CenterMode::ROW) {
        // Per-row centering: for each row i, subtract mean across cols.
        for (size_t i = 0; i < grad.rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < grad.cols; ++j) sum += grad[i][j];
            double mean = sum / static_cast<double>(grad.cols);
            for (size_t j = 0; j < grad.cols; ++j) grad[i][j] -= mean;
        }
    } else {
        // COLUMN mode: for each column j, subtract mean across rows.
        for (size_t j = 0; j < grad.cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < grad.rows; ++i) sum += grad[i][j];
            double mean = sum / static_cast<double>(grad.rows);
            for (size_t i = 0; i < grad.rows; ++i) grad[i][j] -= mean;
        }
    }
}

void GradientCentralization::step(Model& model) {
    // 1. Center every parameter's gradient in place.
    for (auto& layer : model.layers) {
        if (!layer) continue;
        auto grads = layer->gradients();
        for (Tensor* grad : grads) {
            if (grad && grad->rows > 0 && grad->cols > 0) {
                center_gradient(*grad);
            }
        }
    }
    // 2. Delegate to the inner optimizer (which will apply its update and zero grads).
    if (inner_) {
        inner_->step(model);
    }
    // Keep our base lr in sync with the inner for further scheduler queries.
    if (inner_) {
        Optimizer::lr = inner_->lr;
    }
}
