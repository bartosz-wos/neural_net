#ifndef CAUTIOUS_H
#define CAUTIOUS_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <memory>
#include <utility>

class Model;
class Tensor;

// Cautious Optimizer — Liang, Chen, Liu, Liu 2024 (arXiv:2411.16085,
// "Cautious Optimizers: Improving Training with One Line of Code").
//
// Reference implementation: https://github.com/kyleliang919/C-Optim
//
// Idea: any momentum-based optimizer (AdamW, Lion, SGD+M, Muon, …) computes
// an update direction u_t. The original update is
//
//     param ← param − lr · u_t
//
// In many gradient steps, the per-component direction of u_t can DISAGREE
// with the per-component sign of the current gradient — i.e. the optimizer
// is about to push a parameter in the direction that INCREASES the loss for
// that parameter. The paper shows this is always a bad idea: zeroing those
// entries accelerates convergence and improves generalization (the
// HAMILTONIAN argument in the paper proves it preserves Adam's convergence
// guarantee under Lyapunov analysis).
//
// The "one line of code" trick: mask the update direction by the sign-agreement
// with the raw gradient, then re-normalize by the mask's mean to keep the
// average step magnitude roughly the same.
//
// Algorithm (per parameter, per step):
//
//     1. param_before ← param
//     2. inner.step(param)            // inner optimizer mutates param
//     3. u_t ← (param − param_before) / lr      // reconstructed direction
//     4. mask_t = (u_t * g_t > 0)               // 1 where sign agrees, 0 else
//     5. mask_mean_t = clamp(mask_t.mean(), eps_mask)
//     6. param ← param_before                   // restore
//     7. param ← param_before − lr · (u_t * mask_t / mask_mean_t)
//
// Edge cases:
//   - All-zero mask (worse than chance): mask_mean clamped to eps_mask →
//     compensating factor blows up but mask itself is 0 → final step is 0
//     (numerically zero). This is the "do nothing" fallback.
//   - All-ones mask: mask_mean = 1 → no compensation → identical to inner step.
//   - Tensor of size 1: mean is the single mask entry → identity or no step.
//
// The wrapper can wrap any Optimizer (default: Adam). The inner optimizer
// is moved into the wrapper and destroyed with it. `handles_weight_decay()`
// delegates to the inner optimizer.
class Cautious : public Optimizer {
public:
    // Per-parameter mask statistics, populated after each step.
    // mask_sum = number of entries where mask[i][j] = 1 over the param tensor
    // count = number of entries in the parameter tensor
    // Effective mean = mask_sum / count (NOT stored — divided on demand).
    struct Entry {
        double mask_sum;
        double count;
    };

    // Construct Cautious wrapping `inner`. The wrapper takes ownership of
    // `inner` via unique_ptr. `eps_mask` is the floor for the mask mean
    // used in the density compensation. Paper / reference uses 1e-3.
    explicit Cautious(std::unique_ptr<Optimizer> inner, double eps_mask = 1e-3);

    // Forward the call to the inner optimizer, but wrap each per-parameter
    // update with the cautious mask.
    void step(Model& model) override;

    // Weight decay is delegated to the inner optimizer (no double-application).
    bool handles_weight_decay() const override;

    // Setters.
    void set_eps_mask(double v);
    void set_lr(double v);  // forwards to inner

    // Accessors.
    double get_eps_mask() const { return eps_mask_; }
    double get_lr() const;  // returns inner_->lr
    Optimizer* inner() const { return inner_.get(); }

    // State diagnostics.
    bool has_state(void* layer_ptr, size_t param_idx) const;
    size_t last_num_params_updated() const { return last_num_params_updated_; }
    std::pair<double, double> total_mask_stats() const;  // sum, count
    std::pair<double, double> get_param_stats(void* layer_ptr, size_t param_idx) const;

private:
    std::unique_ptr<Optimizer> inner_;
    double eps_mask_;
    size_t last_num_params_updated_;
    std::map<void*, std::vector<Entry>> stats_;
};

#endif
