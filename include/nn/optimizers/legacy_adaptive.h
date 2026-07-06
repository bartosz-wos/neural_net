#ifndef LEGACY_ADAPTIVE_H
#define LEGACY_ADAPTIVE_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>
#include <cmath>

class Tensor;

// =============================================================================
// Five classic adaptive optimizers, all sharing the same per-parameter-state
// pattern (1-2 Tensors of state per parameter), grouped here because they were
// all invented in the 2010-2018 wave of adaptive methods:
//   - AdaGrad   (Duchi 2011) — accumulates squared gradients (per-param LR)
//   - AMSGrad   (Reddi 2018) — uses MAX of past squared gradients (v term)
//   - Nadam     (Dozat 2016) — Adam + Nesterov momentum
//   - Adamax    (Kingma 2014) — Adam with L_inf norm instead of L_2 in v
//   - AdaDelta  (Zeiler 2012) — per-parameter adaptive LR with no global LR
// =============================================================================

// AdaGrad (Duchi et al. 2011, "Adaptive Subgradient Methods for Online Learning
// and Stochastic Optimization", https://jmlr.org/papers/v12/duchi11a.html).
// Accumulates squared gradients to scale the learning rate per parameter.
//   sum_sq_g_t = sum_sq_g_{t-1} + g_t^2
//   param      = param - lr * g_t / (sqrt(sum_sq_g_t) + eps)
// Well-suited to sparse gradients (NLP, embeddings). Known issue: LR shrinks
// monotonically to zero (sum_sq_g grows without bound).
class AdaGrad : public Optimizer {
public:
    double lr;
    double epsilon;
    double weight_decay;

    explicit AdaGrad(double lr = 0.01, double eps = 1e-8, double wd = 0.0);

    void step(Model& model) override;

    // Testing accessors — peek into the state for invariant checks.
    const std::map<void*, std::vector<Tensor>>& sum_sq_state() const { return sum_sq_state_; }

private:
    // Per-parameter state: sum of squared gradients (accumulates forever).
    std::map<void*, std::vector<Tensor>> sum_sq_state_;
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

// AMSGrad (Reddi et al. 2018, "On the Convergence of Adam and Beyond",
// https://arxiv.org/abs/1904.09237). Adam variant that fixes Adam's
// convergence issue by using the MAX of past v_t values:
//   m_t      = beta1 * m_{t-1} + (1-beta1) * g_t
//   v_t      = beta2 * v_{t-1} + (1-beta2) * g_t^2
//   v_hat_t  = max(v_hat_{t-1}, v_t)   <-- key change
//   param    = param - lr * m_hat / (sqrt(v_hat) + eps)
// Guarantees convergence on convex problems where Adam does not.
class AMSGrad : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    double weight_decay;

    explicit AMSGrad(double lr = 0.001, double b1 = 0.9, double b2 = 0.999,
                     double eps = 1e-8, double wd = 0.0);

    void step(Model& model) override;

    // Testing accessors
    const std::map<void*, std::vector<Tensor>>& m_state() const { return m_state_; }
    const std::map<void*, std::vector<Tensor>>& v_state() const { return v_state_; }
    const std::map<void*, std::vector<Tensor>>& vhat_state() const { return vhat_state_; }

private:
    // Per-parameter state: (m, v, v_hat) — three Tensors per parameter.
    std::map<void*, std::vector<Tensor>> m_state_;
    std::map<void*, std::vector<Tensor>> v_state_;
    std::map<void*, std::vector<Tensor>> vhat_state_;
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

// Nadam (Dozat 2016, "Incorporating Nesterov Momentum into Adam",
// https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ). Adam with Nesterov
// momentum applied to the first moment. Two equivalent formulations; we use
// the "schedule-free" one (Tim Salimans' simplification, Appendix):
//   m_t      = beta1 * m_{t-1} + (1-beta1) * g_t
//   m_bar    = beta1 * m_t / (1 - prod(beta1, 1..t)) + (1-beta1) * g_t / (1 - prod(beta1, 1..t))
//   v_t      = beta2 * v_{t-1} + (1-beta2) * g_t^2
//   param    = param - lr * m_bar / (sqrt(v_hat) + eps)
// Empirically slightly better than Adam on some tasks; no extra state vs Adam.
class Nadam : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    double weight_decay;
    double prod_beta1_cum_;  // Running product of (1 - beta1^k)

    explicit Nadam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999,
                   double eps = 1e-8, double wd = 0.0);

    void step(Model& model) override;

private:
    std::map<void*, std::vector<Tensor>> m_state_;
    std::map<void*, std::vector<Tensor>> v_state_;
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

// Adamax (Kingma & Ba 2014, "Adam: A Method for Stochastic Optimization",
// https://arxiv.org/abs/1412.6980, §7). Adam where the L_2 norm of v_t
// is replaced with the L_inf norm — u_t = max(beta2 * u_{t-1}, |g_t|).
//   m_t = beta1 * m_{t-1} + (1-beta1) * g_t
//   u_t = max(beta2 * u_{t-1}, |g_t|)
//   param = param - lr / (1 - beta1^t) * m_t / u_t
// The L_inf norm is well-defined and stable when the gradient has very large
// sparse entries (NLP embeddings). No sqrt in the denominator.
class Adamax : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    double weight_decay;

    explicit Adamax(double lr = 0.002, double b1 = 0.9, double b2 = 0.999,
                    double eps = 1e-8, double wd = 0.0);

    void step(Model& model) override;

    // Testing accessors
    const std::map<void*, std::vector<Tensor>>& m_state() const { return m_state_; }
    const std::map<void*, std::vector<Tensor>>& u_state() const { return u_state_; }

private:
    std::map<void*, std::vector<Tensor>> m_state_;
    std::map<void*, std::vector<Tensor>> u_state_;  // L_inf state
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

// AdaDelta (Zeiler 2012, "ADADELTA: An Adaptive Learning Rate Method",
// https://arxiv.org/abs/1212.5701). Per-parameter adaptive learning rate with
// NO global LR hyperparameter. Uses two running averages: an exponential
// moving average of squared gradients (like RMSprop) and an exponential
// moving average of squared parameter updates (delta):
//   E[g^2]_t     = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
//   RMS[g]_t     = sqrt(E[g^2]_t + eps)
//   update_t     = -RMS[delta]_{t-1} / RMS[g]_t * g_t
//   param        = param + update_t
//   E[delta^2]_t = rho * E[delta^2]_{t-1} + (1-rho) * update_t^2
// "delta" is a stand-in for the running update; the -1/RMS[delta] term makes
// the units consistent. E[delta^2]_0 = 0 initially (sometimes 1; we use 0 to
// match the original paper).
class AdaDelta : public Optimizer {
public:
    double rho;      // decay for E[g^2] and E[delta^2]
    double epsilon;  // numerical stabilizer inside the sqrt
    double weight_decay;

    explicit AdaDelta(double rho = 0.9, double eps = 1e-6, double wd = 0.0);

    void step(Model& model) override;

    // Testing accessors
    const std::map<void*, std::vector<Tensor>>& eg2_state() const { return eg2_state_; }
    const std::map<void*, std::vector<Tensor>>& edelta2_state() const { return edelta2_state_; }

private:
    std::map<void*, std::vector<Tensor>> eg2_state_;    // E[g^2]
    std::map<void*, std::vector<Tensor>> edelta2_state_; // E[delta^2]
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

#endif