#ifndef ADAM_MINI_H
#define ADAM_MINI_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

// Adam-mini: "Adam-mini: Use Fewer Learning Rates To Gain More"
// Zhang et al. 2024 (https://arxiv.org/abs/2406.16793)
//
// Memory-efficient Adam variant. For each parameter:
//   - 2-D weight matrices: block-wise second moment reduced to ONE value
//     per row (mean of g² along columns). State size = rows + rows*cols
//     instead of 2*rows*cols.
//   - 1-D parameters (biases, norms): full per-element v (same as Adam).
//   - Configurable block-mode override.
//
// Per-parameter algorithm (2-D ROW_MEAN, the common case):
//   for each row i:
//     vmean_t[i] = β2 * vmean_{t-1}[i] + (1 - β2) * (1/cols) * Σ_j g_t[i,j]²
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t                              (full m)
//   m̂_t = m_t / (1 - β1^t)                                          (bias correction)
//   v̂_t = vmean_t / (1 - β2^t)                                       (broadcast to rows)
//   update[i,j] = m̂_t[i,j] / (√v̂_t[i] + ε)
//   if wd > 0: θ *= (1 - lr * wd)                                   (decoupled decay)
//   θ -= lr * update
//
// State per parameter: 2 tensors (m full-shape, vmean reduced).
// For 2-D (R, C): m is (R, C), vmean is (R, 1). Saves ~50% vs Adam.
// For 1-D:        m is (1, C), vmean is (1, C). Same as Adam (no savings needed).
class AdamMini : public Optimizer {
public:
    enum class BlockMode {
        AUTO,       // shape-based: 2-D → ROW_MEAN, 1-D → FULL
        FULL,       // per-element v (regular Adam)
        ROW_MEAN,   // vmean per row (Adam-mini main case)
        SCALAR      // one global vmean for the whole parameter
    };

    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;
    BlockMode default_mode;

    explicit AdamMini(double lr_ = 1e-3,
                      double b1 = 0.9,
                      double b2 = 0.999,
                      double eps = 1e-8,
                      double wd = 0.0,
                      BlockMode mode = BlockMode::AUTO);

    // Validated setters
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // State accessors
    bool has_state(void* layer_ptr) const;
    size_t num_params_with_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_vmean(void* layer_ptr, size_t param_idx) const;
    BlockMode get_block_mode(void* layer_ptr, size_t param_idx) const;

    // Override block-mode for a specific parameter.
    void set_param_block_mode(void* layer_ptr, size_t param_idx, BlockMode mode);

private:
    struct ParamState {
        Tensor m;       // first moment, full shape
        Tensor vmean;   // second moment, reduced shape (full for FULL mode)
        BlockMode mode;
    };
    std::map<void*, std::vector<ParamState>> state_;
    std::map<void*, std::vector<BlockMode>> overrides_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    BlockMode resolve_mode(void* layer_ptr, size_t param_idx, const Tensor& p) const;
};

#endif
