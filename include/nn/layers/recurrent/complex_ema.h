#ifndef COMPLEX_EMA_H
#define COMPLEX_EMA_H

#include "../../core/layer.h"

// ============================================================================
// ComplexEMA — MEGALODON §3.1 (Eq. 2-3)
//   "MEGALODON: Efficient LLM Pretraining and Inference with Unlimited
//   Context Length", https://arxiv.org/abs/2404.08801
//
// Mega's multi-dimensional damped EMA (Ma et al. 2023, Eq. 1) extended to the
// COMPLEX domain. For each input channel j and each complex hidden state
// k = 0..h-1:
//
//   h_{t,j,k} = p_{j,k} * x_{t,j} + q_{j,k} * h_{t-1,j,k}          (complex)
//   y_{t,j}   = Re( SUM_k eta_{j,k} * h_{t,j,k} ) + omega_j * x_{t,j}
//
// Parameterization (matching the official implementation's `_calc_coeffs`):
//
//   theta_{j,k} = (k+1) * sigmoid(theta_j) * (2*pi/h)                (Eq. 3)
//   p_{j,k}     = sigmoid(alpha_{j,k})                                  in (0,1)
//   q_{j,k}     = (1 - alpha_{j,k}*delta_{j,k}) * exp(i*theta_{j,k})
//   eta_{j,k}   in C, represented by (eta_re, eta_im), scaled by 1/sqrt(h)
//   omega_j     in R  (the residual/"FFTConv" path)
//
// Because |q| = 1 - alpha*delta < 1, the effective kernel decays — the
// paper's stated key property for long-sequence modeling.
//
// Deviation from Eq. 2 worth recording: the paper writes h_t with an explicit
// expansion matrix beta mapping the scalar channel to h dimensions. The
// official released implementation absorbs beta into alpha and adds the
// omega_j * x_{t,j} residual path, which is not in Eq. 2. We follow the
// released implementation, since that is what the 7B model was trained with.
//
// The repo's `Tensor` stores real doubles in a 2-D buffer only, so the complex
// hidden state and the complex eta are held as real (d, 2h) tensors with the
// columns interleaved: column 2*k is the real part of component k, column
// 2*k+1 is the imaginary part.
// ============================================================================

class ComplexEMA : public Layer {
public:
    // d_model: number of input/output channels (d in the paper)
    // ndim:    number of complex hidden states per channel (h in the paper)
    ComplexEMA(size_t d_model, size_t ndim = 16);
    ~ComplexEMA() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return alpha_; }
    Tensor get_gradients() const override { return grad_alpha_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "ComplexEMA"; }

    size_t d_model() const { return d_model_; }
    size_t ndim() const { return ndim_; }
    double eta_scale() const { return 1.0 / std::sqrt(static_cast<double>(ndim_)); }

    // Recompute the coefficient cache from the current parameters.
    // Called by forward(); exposed so tests can inspect p/q/eta directly.
    void compute_coeffs();

    // Coefficient / state caches (valid after forward or compute_coeffs).
    const Tensor& last_p() const { return p_; }        // (d, h)
    const Tensor& last_q_re() const { return q_re_; }  // (d, h)
    const Tensor& last_q_im() const { return q_im_; }  // (d, h)
    const Tensor& last_eta() const { return eta_; }    // (d, 2h), interleaved
    const Tensor& last_h() const { return h_state_; }  // (d, 2h) final state

    // Learnable parameters. Public for direct inspection in tests (same
    // convention as GatedDeltaNet's W_Q_ etc.). The per-parameter
    // `name(j, k)` accessors below are thin forwarders so callers can write
    // `e.alpha(j, k) = ...` instead of reaching into the tensor.
    Tensor alpha_;   // (d, h)
    Tensor delta_;   // (d, h)
    Tensor theta_;   // (d, 1)
    Tensor eta_re_;  // (d, h)
    Tensor eta_im_;  // (d, h)
    Tensor omega_;   // (d, 1)

    double& alpha(size_t j, size_t k) { return alpha_(j, k); }
    double& delta(size_t j, size_t k) { return delta_(j, k); }
    double& theta(size_t j, size_t k) { return theta_(j, k); }
    double& eta_re(size_t j, size_t k) { return eta_re_(j, k); }
    double& eta_im(size_t j, size_t k) { return eta_im_(j, k); }
    double& omega(size_t j, size_t k) { return omega_(j, k); }

    Tensor grad_alpha_;  // (d, h)
    Tensor grad_delta_;  // (d, h)
    Tensor grad_theta_;  // (d, 1)
    Tensor grad_eta_re_; // (d, h)
    Tensor grad_eta_im_; // (d, h)
    Tensor grad_omega_;  // (d, 1)

private:
    size_t d_model_;
    size_t ndim_;

    // Derived coefficients (recomputed from parameters on each forward).
    // eta is stored interleaved: [re0, im0, re1, im1, ...] in a (d, 2h) tensor,
    // matching the repo's 2-D-only Tensor.
    Tensor p_;       // (d, h)
    Tensor q_re_;    // (d, h)
    Tensor q_im_;    // (d, h)
    Tensor eta_;     // (d, 2h) = eta * eta_scale()

    // Forward cache.
    Tensor last_x_;    // (n, d)
    Tensor h_state_;   // (d, 2h) — hidden state AFTER the last timestep

    // Full hidden-state history (n, d, h, 2). The backward pass needs h at
    // every timestep: the adjoint of h_t is a sum over ALL future timesteps
    //   adj(h_t) = sum_{s>=t} G[s] * conj(eta) * conj(q)^(s-t)
    // so a single carried adjoint is not sufficient. Memory is O(n*d*h).
    Tensor h_hist_;    // (n, d*2h)

    Tensor grad_x_;    // (n, d)
};

#endif  // COMPLEX_EMA_H
