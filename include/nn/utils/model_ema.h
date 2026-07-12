#ifndef MODEL_EMA_H
#define MODEL_EMA_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <vector>
#include <stdexcept>

// =============================================================================
// ModelEMA — Exponential Moving Average of model parameters.
//
// Maintains a "shadow" copy of every parameter Tensor in the model. After each
// training step, the shadow is updated with the recursive formula:
//
//   shadow_t = decay * shadow_{t-1} + (1 - decay) * w_t
//
// During evaluation, the shadow weights are used (typically) in place of the
// raw weights — this gives smoother, less-noisy predictions than the raw
// weights and is empirically a strong regularizer. Used pervasively in
// modern training pipelines: DDPM/DDIM diffusion, GANs, BYOL, MoCo, MAE,
// timm ImageNet recipes, and most semi-supervised learning.
//
// Decay is typically 0.999 (fast) or 0.9999 (slow). At decay=0 the shadow
// equals the current weights; at decay=1 the shadow never changes.
//
// This is intentionally a "passive" wrapper — it does NOT inherit from
// Optimizer. The intended usage pattern is:
//
//   Optimizer opt(...);
//   ModelEMA ema(model, 0.999);
//   for (int step = 0; step < N; ++step) {
//       // ... forward, backward, opt.step(model)
//       ema.step(model);   // update shadow after the weights move
//   }
//   // At eval time:
//   ema.apply_shadow(model);
//   // ... compute metrics on validation set
//   ema.restore(model);
//
// API follows the timm naming convention (store / copy_to / apply_shadow /
// restore) plus an extra `step(model)` for the recursive EMA update.
//
// Implementation notes:
//   * State is lazy-initialized on the first step(): one Tensor per parameter
//     in the model. If new parameters are added later (unusual), rebuild the
//     EMA via `reinitialize(model)`.
//   * The shadow tensor owns its own data (deep copy). No aliasing with model
//     parameter memory.
//   * Parameter iteration order matches Model::parameters() in declaration
//     order — flatten layers, normalize layers, dense layers, attention
//     layers, etc. This makes apply_shadow/restore bit-exact.
//   * decay must lie in [0, 1] — constructor throws otherwise.
// =============================================================================

class ModelEMA {
public:
    // Construct an EMA wrapper bound to the given model. decay must be in
    // [0, 1]; out-of-range decays throw std::invalid_argument.
    explicit ModelEMA(Model& model, double decay = 0.999);

    // Update the shadow weights using the current model parameter values:
    //   shadow = decay * shadow + (1 - decay) * w
    // Call this AFTER the optimizer has moved the weights.
    void step(Model& model);

    // Re-initialize shadow to current model parameters. Useful when the
    // model architecture changes (e.g. adding a layer) or when you want to
    // "warm-start" the EMA from a freshly loaded checkpoint.
    void reinitialize(Model& model);

    // Save the current model weights into a backup buffer. Used in
    // combination with copy_to() to "snapshot" weights before swapping in
    // the shadow (which is exactly what apply_shadow does, but exposed
    // separately for flexibility — e.g. evaluate on shadow, then re-store
    // the original).
    void store(Model& model);

    // Copy the shadow weights into the model. This DESTROYS the current
    // model weights — pair with a store() first or a subsequent restore().
    void copy_to(Model& model) const;

    // Convenience: store then copy_to. The previous model weights remain
    // recoverable via restore().
    void apply_shadow(Model& model);

    // Restore the model weights that were saved via store() (or
    // apply_shadow()). Throws if no stored weights are available.
    void restore(Model& model);

    // ----- Accessors -----
    double decay() const { return decay_; }
    void set_decay(double d) {
        if (d < 0.0 || d > 1.0) {
            throw std::invalid_argument("ModelEMA::set_decay: decay must be in [0, 1]");
        }
        decay_ = d;
    }
    size_t step_count() const { return step_count_; }
    size_t num_params() const { return shadow_.size(); }

    // ----- Testing accessors -----
    // Read out the shadow Tensor for the i-th parameter (in the order they
    // appeared in model.layers at initialization). Returns a copy.
    Tensor get_shadow(size_t i) const;
    // Read out the most recently stored (apply_shadow backup) Tensor for the
    // i-th parameter. Returns a copy of an empty Tensor if no store has
    // happened yet.
    Tensor get_stored(size_t i) const;

private:
    double decay_;
    size_t step_count_ = 0;
    std::vector<Tensor> shadow_;   // one Tensor per parameter
    std::vector<Tensor> stored_;   // backup from store() / apply_shadow()
    bool has_stored_ = false;

    // Internal: collect parameter pointers in declaration order.
    static std::vector<Tensor*> collect_params(Model& model);
    // Internal: copy w -> dst element-wise (deep copy).
    static void copy_into(const Tensor& src, Tensor& dst);
    // Internal: deep element-wise multiply: dst[i][j] = decay * dst[i][j] + (1 - decay) * src[i][j]
    static void ema_combine_into(const Tensor& src, Tensor& dst, double decay);
};

#endif // MODEL_EMA_H