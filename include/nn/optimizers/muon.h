#ifndef MUON_H
#define MUON_H

#include "optimizer.h"
#include <map>
#include <vector>

// Muon: "Muon: An Optimizer for Hidden Layers in Neural Networks"
// Keller Jordan (2024, https://kellerjordan.github.io/posts/muon/)
//
// Key idea: standard SGD-momentum, then **Newton–Schulz orthogonalization** of
// the momentum buffer. The orthogonalized update has uniform singular values
// (it's a scaled orthogonal matrix), which empirically yields faster
// convergence than Adam on transformer hidden weights — and powered the
// NanoGPT speedrun records.
//
// Per-parameter update (for 2-D parameters):
//     m_t = β · m_{t-1} + g_t                              (SGD-momentum)
//     update = NewtonSchulz(m_t, ns_steps=5)               (orthogonalize)
//     update *= 0.2 · max(1, sqrt(out / in))               (paper scale)
//     param_t = param_{t-1} − lr · (update + wd · param_{t-1})  (AdamW-style wd)
//
// For 1-D parameters (biases, norms, embeddings), Muon falls back to plain
// SGD with momentum (Newton–Schulz requires a 2-D square-or-tall matrix
// structure to be well-defined).
//
// Newton–Schulz iteration (Bernstein–Jordan 2024 "Optimizing 2-D Kerneled
// Matrices"):  X ← a · X + b · X X^T X + c · (X X^T)^2 X  with the
// (3.4445, −4.7750, 2.0315) coefficients from the original Muon repo
// (https://github.com/KellerJordan/Muon). After ns_steps=5 iterations, X
// approximates U·V^T where X = U·Σ·V^T is the SVD — the singular values
// get squashed toward 1.
//
// Cautious variant (Liang et al. 2024 "Cautious Optimizers"):
// Optionally zero out the update wherever (update * g_t) < 0, i.e. where
// the orthogonalized direction disagrees in sign with the gradient. This
// empirically improves stability with no extra hyperparameter cost.
//
// Hyperparameters (paper / NanoGPT speedrun defaults):
//   - lr ∈ [0.001, 0.05]    : orthogonalization already normalizes the
//                               update, so the LR is much larger than Adam's.
//                               Typical: lr=0.02 for transformer training.
//   - β = 0.95              : SGD momentum coefficient
//   - ns_steps = 5          : Newton–Schulz iterations
//   - scale = 0.2           : empirical Frobenius-norm scale of an orthogonal
//                               matrix; multiplied by max(1, sqrt(out/in))
//   - wd ≥ 0                : decoupled weight decay (AdamW-style). 0 disables.
//   - cautious = false      : enable for the Cautious variant.
//
// State: per-parameter ONE Tensor (m, the momentum buffer), only for 2-D
// parameters. 1-D parameters still get an m buffer but skip the orthogonalize
// step.
class Muon : public Optimizer {
public:
    double lr;             // learning rate (typically 0.001-0.05, larger than Adam)
    double momentum;       // SGD momentum coefficient (β in the paper; default 0.95)
    int    ns_steps;       // Newton–Schulz iterations (default 5)
    double scale_const;    // 0.2 in the paper — empirical scale of orthogonal Frob norm
    double weight_decay;   // decoupled weight decay (default 0)
    bool   cautious;       // apply Cautious mask (default false)

    explicit Muon(double lr = 0.02,
                  double momentum = 0.95,
                  int    ns_steps = 5,
                  double scale_const = 0.2,
                  double weight_decay = 0.0,
                  bool   cautious = false);

    void step(Model& model) override;

    // Muon already applies weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // Testing accessors — read out momentum buffers for invariant checks.
    const std::map<void*, std::vector<Tensor>>& momentum_state() const { return state_; }
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    size_t num_steps() const { return num_steps_; }

    // Newton–Schulz step exposed for direct testing (5 iterations of the
    // cubic polynomial mapping X to (aX + b X X^T X + c (X X^T)^2 X)).
    static Tensor newton_schulz(const Tensor& X, int ns_steps = 5);

private:
    // Per-layer state: Layer* → vector of (m) tensors, one per parameter.
    std::map<void*, std::vector<Tensor>> state_;
    size_t num_steps_ = 0;

    // Lazily initialize the state for a layer's parameters.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Update a 2-D parameter via orthogonalized momentum.
    void update_2d(Tensor* param, Tensor* grad, Tensor& m);

    // Update a 1-D parameter via plain SGD-momentum (Newton–Schulz fallback).
    void update_1d(Tensor* param, Tensor* grad, Tensor& m);
};

#endif