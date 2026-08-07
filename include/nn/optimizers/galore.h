#ifndef GALORE_H
#define GALORE_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

class Model;

// =========================================================================
// GaLore: Gradient Low-Rank Projection
//
// Zhao, J., Yang, Z., Wang, Y., Shen, Z., Zhang, Y., Rajan, R., Tan, Y., &
// Khan, S. (2024). "GaLore: Memory-Efficient LLM Training by Gradient
// Low-Rank Projection." https://arxiv.org/abs/2403.03508
//
// Core idea: the optimizer state for Adam on a parameter W of shape (m, n)
// is O(mn), which dominates memory at LLM scale. GaLore observes that the
// gradient matrix G has intrinsically low rank during training — instead of
// applying Adam to the full m×n gradient, project G onto a low-rank subspace
// of dimension r ≪ min(m, n), apply Adam in that subspace, then project the
// update back. This reduces optimizer-state memory by a factor of ~r/d or
// r/k while preserving convergence.
//
// === Algorithm (per-parameter, projection-only variant, Zhao §3.1) ===
//
// PER PARAMETER (shape (m, n), target rank r ≤ min(m, n)):
//   1. Gradient EMA: G_EMA_t = β2 · G_EMA_{t-1} + (1-β2) · G_t
//      (full m×n matrix, since the EMA is what we eigendecompose to get P)
//
//   2. Projection refresh (every `proj_update_interval` steps):
//      Let U Σ V^T = SVD_top-r(G_EMA_t). Then P_t = V_r ∈ ℝ^{n×r}.
//      (The right singular vectors of G_EMA give an orthonormal basis for
//      the row space of G_EMA, which is the column space of G^T.)
//      We store P_t and use it for the next interval steps.
//
//   3. Adam-in-subspace (every step):
//      g_low = G_t @ P_t                              (m, r) — low-rank grad
//      m_low = β1 · m_low + (1-β1) · g_low
//      v_low = β2 · v_low + (1-β2) · g_low²
//      m_hat = m_low / (1 - β1^t)
//      v_hat = v_low / (1 - β2^t)
//      update_low = m_hat / (sqrt(v_hat) + ε)
//      update = update_low @ P_t^T                    (m, n) — project back
//      θ -= lr · update
//
//   4. Decoupled weight decay (if weight_decay > 0):
//      θ *= (1 - lr · weight_decay)
//
// === Defaults (Zhao et al. 2024 §5, recommended for LLM fine-tuning) ===
//   lr = 0.001
//   beta1 = 0.9
//   beta2 = 0.999
//   epsilon = 1e-8
//   rank = 4                       (paper: r ∈ {2, 4, 8}; r=4 is a good default)
//   proj_update_interval = 200     (refresh projection every T steps)
//   weight_decay = 0.0             (decoupled, AdamW-style)
//   scale = 1.0                    (optional scaling on the projection)
//
// === Validation rules ===
//   lr > 0, ε > 0, rank ≥ 1, proj_update_interval ≥ 1, weight_decay ≥ 0, scale > 0
//   0 ≤ beta1 < 1, 0 ≤ beta2 < 1
//
// === State per parameter (lazy on first step) ===
//   P        (n × r) — projection matrix (top-r right singular vectors of G_EMA)
//   m_low    (m × r) — first moment in projected space
//   v_low    (m × r) — second moment in projected space
//   G_EMA    (m × n) — gradient EMA used for projection refresh
//   t_proj   last step index when P was refreshed
//   step_pt  current step counter (shared with global step but tracked per-param)
//
// === Step counter (Adam-style, starts at 1) ===
//   Maintained globally; incremented after each step() call.
//
// === Public API (parity with Prodigy/SOAP) ===
//   GaLore(...)                       — constructor with paper defaults
//   set_lr / set_beta1 / set_beta2 / set_epsilon
//   set_rank / set_proj_update_interval / set_weight_decay / set_scale
//   get_* accessors for every hyperparameter
//   step(Model&)
//   handles_weight_decay() → true (decoupled WD)
//   has_state(layer_ptr) / num_params_with_state(layer_ptr)
//   get_P / get_m_low / get_v_low / get_G_EMA / get_step_proj
//
// === Reference ===
//   https://arxiv.org/abs/2403.03508
//   PyTorch reference: https://github.com/jiaweizzhao/GaLore
// =========================================================================

class GaLore : public Optimizer {
public:
    // --- Public hyperparameters ---
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int    rank;
    int    proj_update_interval;
    double weight_decay;
    double scale;

    // --- Global step counter (Adam-style, starts at 1) ---
    int step_count;

    // --- Constructor with paper defaults ---
    explicit GaLore(double lr_             = 0.001,
                    double b1              = 0.9,
                    double b2              = 0.999,
                    double eps             = 1e-8,
                    int    rank_           = 4,
                    int    proj_interval   = 200,
                    double wd              = 0.0,
                    double scale_          = 1.0);

    // --- Validated setters (throw std::invalid_argument on invalid input) ---
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_rank(int v);
    void set_proj_update_interval(int v);
    void set_weight_decay(double v);
    void set_scale(double v);

    // --- Accessors ---
    double get_lr()                  const { return lr; }
    double get_beta1()               const { return beta1; }
    double get_beta2()               const { return beta2; }
    double get_epsilon()             const { return epsilon; }
    int    get_rank()                const { return rank; }
    int    get_proj_update_interval() const { return proj_update_interval; }
    double get_weight_decay()        const { return weight_decay; }
    double get_scale()               const { return scale; }
    int    get_step_count()        const { return step_count; }

    // --- Optimizer interface ---
    void step(Model& model) override;

    // Decoupled weight decay is applied internally.
    bool handles_weight_decay() const override { return true; }

    // --- State introspection (for tests + debugging) ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    size_t num_params_with_state(void* layer_ptr) const;

    // Retrieve the projection matrix, m_low, v_low, G_EMA, or step_proj
    // for (layer, param_idx). Returns empty (0, 0) Tensor if not seen.
    const Tensor& get_P(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_m_low(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_v_low(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_G_EMA(void* layer_ptr, size_t param_idx) const;
    int get_step_proj(void* layer_ptr, size_t param_idx) const;

private:
    // --- Per-parameter state ---
    struct ParamState {
        Tensor P;        // (n, r) projection matrix
        Tensor m_low;    // (m, r) first moment in projected space
        Tensor v_low;    // (m, r) second moment in projected space
        Tensor G_EMA;    // (m, n) gradient EMA
        int    t_proj;   // last step when P was refreshed (0 = uninitialized)
    };

    std::map<void*, std::vector<ParamState>> state_;

    // --- Helpers ---
    static void validate(double lr_, double b1, double b2, double eps,
                         int rank_, int proj_interval, double wd, double scale_);

    // Helper for the public get_* accessors — must be a member (or friend)
    // because ParamState is private.
    const ParamState* find_state(void* layer_ptr, size_t param_idx) const;

    // Lazy-init state for a layer on first encounter.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Refresh projection matrix P from G_EMA via top-r right singular vectors.
    // Uses economy-sized SVD computed via the eigendecomposition of G_EMA^T G_EMA.
    void refresh_projection(ParamState& st, size_t m, size_t n);

    // Per-parameter update.
    void update_param(Tensor* param, Tensor* grad, ParamState& st,
                      double b1_correction, double b2_correction);

    // Symmetric eigendecomposition via cyclic Jacobi rotations.
    // On entry, A is a symmetric (k × k) Tensor. On exit, Q holds the
    // orthogonal matrix of eigenvectors (columns) and eigenvalues is
    // filled with the eigenvalues of A. Q is initialized to identity.
    static void jacobi_eigendecompose(Tensor& A,
                                      Tensor& Q,
                                      Tensor& eigenvalues,
                                      int max_sweeps = 100,
                                      double tol = 1e-12);
};

#endif
