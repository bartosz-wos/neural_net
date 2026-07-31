#ifndef SOAP_H
#define SOAP_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// SOAP: ShampoO with Adam in the Preconditioner's eigenbasis
//
// Vyas, Morwani, Janson, Katharopoulos, Tibrewal, Jaderberg, Chen, Harvey,
// Doucet, Janson, Zhong, Linsley, Goyal, Zhai (2024),
// "SOAP: Improving and Stabilizing Shampoo using Adam"
// https://arxiv.org/abs/2409.11321  (NeurIPS 2024)
//
// The core idea: Shampoo uses two left/right preconditioners
//   L_t = β·L_{t-1} + (1-β)·G·G^T   (m×m)
//   R_t = β·R_{t-1} + (1-β)·G^T·G   (n×n)
// and applies the preconditioner via L^{-1/4}·G·R^{-1/4}. SOAP instead
// eigendecomposes L = Q_L·diag(λ_L)·Q_L^T, R = Q_R·diag(λ_R)·Q_R^T, rotates
// the gradient into the eigenbasis (G_rot = Q_L^T·G·Q_R), runs Adam in
// that basis, and rotates the update back. The result is the Adam
// convergence properties combined with Shampoo's higher-order preconditioning.
//
// Per step t (Vyas et al. 2024 §3, Algorithm 1):
//   if t mod precondition_frequency == 1 (or t == 1):
//       L_t = β2·L_{t-1} + (1-β2)·G·G^T
//       R_t = β2·R_{t-1} + (1-β2)·G^T·G
//       Q_L, λ_L = eigh(L_t)
//       Q_R, λ_R = eigh(R_t)
//   G_rot = Q_L^T · G · Q_R
//   M_t = β1·M_{t-1} + (1-β1)·G_rot
//   V_t = β2·V_{t-1} + (1-β2)·G_rot ⊙ G_rot
//   M̂_t = M_t / (1-β1^t)
//   V̂_t = V_t / (1-β2^t)
//   update_rot = lr · M̂_t / (sqrt(V̂_t) + ε)
//   update = Q_L · update_rot · Q_R^T
//   W := W - update
//
// Defaults (Vyas et al. 2024 §5.2, Llama-3-scale experiments):
//   lr = 3e-3, β1 = 0.95, β2 = 0.95, ε = 1e-8,
//   precondition_frequency = 10, weight_decay = 0
//
// Edge cases:
//   - m == 1 OR n == 1: skip eigendecomp for that side, use identity rotation
//   - m == 1 AND n == 1: scalar — plain Adam (no rotation needed)
//   - Invalid hyperparameters throw std::invalid_argument
//
// State per parameter:
//   L (m×m), R (n×n)  — symmetric covariance EMA
//   Q_L (m×m), Q_R (n×n) — cached eigenbases
//   M (m×n), V (m×n) — Adam moments in rotated space
// =========================================================================

class SOAP : public Optimizer {
public:
    // Public diagnostics.
    double beta1;
    double beta2;
    double epsilon;
    int    precondition_frequency;
    double weight_decay;
    int    t;  // next timestep, starts at 1

    explicit SOAP(double lr = 3e-3,
                  double beta1 = 0.95,
                  double beta2 = 0.95,
                  double epsilon = 1e-8,
                  int precondition_frequency = 10,
                  double weight_decay = 0.0);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated mutators.
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_epsilon(double new_epsilon);
    void set_precondition_frequency(int new_freq);
    void set_weight_decay(double new_wd);

    // Accessors.
    double get_lr() const { return Optimizer::lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_epsilon() const { return epsilon; }
    int    get_precondition_frequency() const { return precondition_frequency; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t() const { return t; }

    // State introspection (for tests / debugging).
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_L(void* layer_ptr, size_t param_idx) const;
    Tensor get_R(void* layer_ptr, size_t param_idx) const;
    Tensor get_Q_L(void* layer_ptr, size_t param_idx) const;
    Tensor get_Q_R(void* layer_ptr, size_t param_idx) const;
    Tensor get_M(void* layer_ptr, size_t param_idx) const;
    Tensor get_V(void* layer_ptr, size_t param_idx) const;

private:
    // Per-parameter state (private but accessed by friend helpers below).
    struct ParameterState {
        Tensor L;     // (m × m) symmetric covariance
        Tensor R;     // (n × n) symmetric covariance
        Tensor Q_L;   // (m × m) orthogonal eigenbasis of L (cached)
        Tensor Q_R;   // (n × n) orthogonal eigenbasis of R (cached)
        Tensor M;     // (m × n) Adam first moment in rotated space
        Tensor V;     // (m × n) Adam second moment in rotated space
    };

    std::map<void*, std::vector<ParameterState>> state_;

    // Helper for the public get_* accessors — must be a member (or friend)
    // because ParameterState is private.
    const ParameterState* find_state(void* layer_ptr, size_t param_idx) const;

    static void validate(double lr,
                         double beta1,
                         double beta2,
                         double epsilon,
                         int    precondition_frequency,
                         double weight_decay);

    // Lazy-init state for a layer on first encounter.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update.
    void update_param(Tensor* param,
                      Tensor* grad,
                      ParameterState& st,
                      double b1_correction,
                      double b2_correction);

    // Symmetric eigendecomposition via cyclic Jacobi rotations.
    // On entry, A is a symmetric (n × n) Tensor. On exit, Q holds the
    // orthogonal matrix of eigenvectors (columns) and eigenvalues is
    // filled with the eigenvalues of A. Q is initialized to identity.
    static void jacobi_eigendecompose(Tensor& A,
                                      Tensor& Q,
                                      Tensor& eigenvalues,
                                      int max_sweeps = 100,
                                      double tol = 1e-12);

    // Convenience: at step t, should we recompute the preconditioner?
    bool should_precondition(int step) const {
        return step == 1 || (step - 1) % precondition_frequency == 0;
    }
};

#endif