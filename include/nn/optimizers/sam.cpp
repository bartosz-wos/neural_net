#include "sam.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

SAM::SAM(Optimizer* inner, double rho, double eps_global, bool adaptive)
    : inner_(inner), rho_(rho), eps_global_(eps_global), adaptive_(adaptive) {
    if (rho_ < 0.0) {
        throw std::invalid_argument("SAM: rho must be >= 0");
    }
    if (eps_global_ <= 0.0) {
        throw std::invalid_argument("SAM: eps_global must be > 0");
    }
    // Inherit lr from inner for LR schedulers / display
    if (inner_) {
        this->Optimizer::lr = inner_->lr;
    }
}

void SAM::step(Model& /*model*/) {
    // SAM needs two forward+backward passes per step — calling step() once
    // bypasses the second pass and would compute a normal SGD-like update.
    // Callers must use the two-phase API: first_step + second_step.
    throw std::logic_error(
        "SAM::step() is not callable directly. Use first_step(model) after the "
        "first backward at w, run forward+backward again at w+epsilon, then "
        "call second_step(model). See the SAM test suite for an example.");
}

double SAM::compute_global_grad_norm(const Model& model) const {
    double sumsq = 0.0;
    for (const auto& layer : model.layers) {
        auto grads = layer->gradients();
        for (const Tensor* g : grads) {
            for (size_t r = 0; r < g->rows; ++r) {
                for (size_t c = 0; c < g->cols; ++c) {
                    double v = (*g)[r][c];
                    sumsq += v * v;
                }
            }
        }
    }
    return std::sqrt(sumsq);
}

double SAM::compute_param_grad_norm(const Tensor& grad) const {
    double sumsq = 0.0;
    for (size_t r = 0; r < grad.rows; ++r) {
        for (size_t c = 0; c < grad.cols; ++c) {
            double v = grad[r][c];
            sumsq += v * v;
        }
    }
    return std::sqrt(sumsq);
}

void SAM::first_step(Model& model) {
    if (perturbed_) {
        throw std::logic_error(
            "SAM::first_step called while already perturbed. Did you forget "
            "to call second_step on the previous iteration?");
    }

    // ---- 1. Snapshot current weights into saved_weights_ ----
    saved_weights_.clear();
    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer_ptr->parameters();
        std::vector<Tensor> snapshots;
        snapshots.reserve(params.size());
        for (Tensor* p : params) {
            Tensor copy(p->rows, p->cols);
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c = 0; c < p->cols; ++c) {
                    copy[r][c] = (*p)[r][c];
                }
            }
            snapshots.push_back(std::move(copy));
        }
        saved_weights_[(void*)layer_ptr] = std::move(snapshots);
    }

    // ---- 2. Compute gradient norm and perturbation magnitude ----
    double global_norm = compute_global_grad_norm(model);
    last_global_norm_ = global_norm;
    // Effective scale factor: rho * grad / max(||grad||, eps)
    // If ||grad|| is essentially 0 (zero-grad / dead model), the perturbation
    // is also 0 — the weights stay where they are. second_step will then run
    // the inner.step() with the gradients that are now 0 too, so the model
    // won't move. This matches the standard SAM behavior (degenerate case).
    double scale;
    if (adaptive_) {
        // Per-parameter perturbation. The standard adaptive-SAM formula is
        //   eps_w[i] = rho * grad[i] / (||grad_w||_param + eps)
        // We pass a per-parameter scale to the inner loop.
        scale = 1.0;  // unused in adaptive mode
    } else {
        // Standard SAM: a single scalar perturbation applied uniformly.
        scale = rho_ / std::max(global_norm, eps_global_);
    }
    last_eps_scalar_ = scale * std::max(global_norm, eps_global_);  // = rho_ in non-adaptive

    // ---- 3. Perturb weights ----
    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer_ptr->parameters();
        auto grads = layer_ptr->gradients();
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            if (p->rows == 0 || p->cols == 0) continue;

            if (adaptive_) {
                double param_norm = compute_param_grad_norm(*g);
                double local_scale = rho_ / std::max(param_norm, eps_global_);
                for (size_t r = 0; r < p->rows; ++r) {
                    for (size_t c = 0; c < p->cols; ++c) {
                        (*p)[r][c] += local_scale * (*g)[r][c];
                    }
                }
            } else {
                for (size_t r = 0; r < p->rows; ++r) {
                    for (size_t c = 0; c < p->cols; ++c) {
                        (*p)[r][c] += scale * (*g)[r][c];
                    }
                }
            }
        }
    }

    perturbed_ = true;
}

void SAM::second_step(Model& model) {
    if (!perturbed_) {
        throw std::logic_error(
            "SAM::second_step called without a matching first_step.");
    }
    if (!inner_) {
        throw std::logic_error("SAM: inner optimizer is null.");
    }

    // ---- 1. Restore original weights ----
    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer_ptr->parameters();
        auto it = saved_weights_.find((void*)layer_ptr);
        if (it == saved_weights_.end()) continue;
        const auto& snapshots = it->second;
        for (size_t i = 0; i < params.size() && i < snapshots.size(); ++i) {
            Tensor* p = params[i];
            const Tensor& snap = snapshots[i];
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c = 0; c < p->cols; ++c) {
                    (*p)[r][c] = snap[r][c];
                }
            }
        }
    }
    saved_weights_.clear();

    // ---- 2. Inner step (uses gradient computed AT the perturbed position) ----
    inner_->step(model);

    perturbed_ = false;
}