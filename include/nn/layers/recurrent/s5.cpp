#include "s5.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <complex>
#include <algorithm>

// ============================================================================
// S5Layer
// ============================================================================
//
// Implementation notes:
//   * A_re_, A_im_ store the continuous-time eigenvalues A_n ∈ ℂ.
//   * B_re_, B_im_, C_re_, C_im_ store B_n, C_n ∈ ℂ.
//   * At each forward, we compute the discretised params (Ā_n, B̄_n) per channel
//     and cache them for backward:
//        Ā_n = (1 + Δ/2 · A_n) / (1 − Δ/2 · A_n)
//        B̄_n = B_n / (1 − Δ/2 · A_n)
//     (these are scalar complex divisions on each diagonal entry).
//   * Forward recurrence:  h_t = Ā_n · h_{t-1} + B̄_n · x_t  (per channel).
//   * Output:              y_t = Re( C · h_t ) + D · x_t.
//   * Backward: reverse sweep on h with Ā, then Tustin chain rule.

// ----- HiPPO-LegS closed-form eigenvalues (paper §2.2.1) ------------------
static inline double hippo_legs_A_n(size_t n) {
    // A_n = −sqrt((2n+1)(2n+2))
    return -std::sqrt(static_cast<double>(2 * n + 1) * static_cast<double>(2 * n + 2));
}

static inline double hippo_legs_B_n(size_t n) {
    // B_n = sqrt(2n+1)
    return std::sqrt(static_cast<double>(2 * n + 1));
}

// ----- Tustin discretisation (per-channel scalar complex) -----------------
static inline std::complex<double> tustin_Abar(std::complex<double> A_n, double dt) {
    std::complex<double> half_dt(0.5 * dt, 0.0);
    std::complex<double> one(1.0, 0.0);
    std::complex<double> I_minus = one - half_dt * A_n;
    std::complex<double> I_plus = one + half_dt * A_n;
    return I_plus / I_minus;
}

static inline std::complex<double> tustin_Bbar(std::complex<double> B_n, std::complex<double> A_n, double dt) {
    std::complex<double> half_dt(0.5 * dt, 0.0);
    std::complex<double> one(1.0, 0.0);
    std::complex<double> I_minus = one - half_dt * A_n;
    return B_n / I_minus;
}

// ----- Constructor ---------------------------------------------------------
S5Layer::S5Layer(size_t d_model, size_t d_state, bool bidirectional, double dt)
    : A_re_(d_state, 1),
      A_im_(d_state, 1),
      B_re_(d_state, 1),
      B_im_(d_state, 1),
      C_re_(d_state, 1),
      C_im_(d_state, 1),
      D_skip_(1, 1),
      gate_log_(d_state, 1),
      grad_A_re_(d_state, 1),
      grad_A_im_(d_state, 1),
      grad_B_re_(d_state, 1),
      grad_B_im_(d_state, 1),
      grad_C_re_(d_state, 1),
      grad_C_im_(d_state, 1),
      grad_D_skip_(1, 1),
      grad_gate_log_(d_state, 1),
      grad_Abar_re_acc_(d_state, 1),
      grad_Abar_im_acc_(d_state, 1),
      grad_Bbar_re_acc_(d_state, 1),
      grad_Bbar_im_acc_(d_state, 1),
      d_model_(d_model),
      d_state_(d_state),
      dt_(dt),
      bidirectional_(bidirectional),
      Abar_re_(d_state, 1),
      Abar_im_(d_state, 1),
      Bbar_re_(d_state, 1),
      Bbar_im_(d_state, 1) {

    if (d_model == 0) {
        throw std::invalid_argument("S5Layer: d_model must be > 0");
    }
    if (d_state == 0) {
        throw std::invalid_argument("S5Layer: d_state must be > 0");
    }
    if (dt <= 0.0) {
        throw std::invalid_argument("S5Layer: dt must be > 0");
    }

    // HiPPO-LegS init for A (real, negative) and B (real, positive).
    for (size_t n = 0; n < d_state; ++n) {
        A_re_(n, 0) = hippo_legs_A_n(n);
        A_im_(n, 0) = 0.0;
        B_re_(n, 0) = hippo_legs_B_n(n);
        B_im_(n, 0) = 0.0;
    }

    // Random complex C init (small magnitude so initial outputs are reasonable).
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t n = 0; n < d_state; ++n) {
        C_re_(n, 0) = dist(gen);
        C_im_(n, 0) = dist(gen);
    }

    // D skip default = 1.0 (paper convention).
    D_skip_(0, 0) = 1.0;

    // Bidirectional gate default = 0.5 (sigmoid(0) = 0.5).
    for (size_t n = 0; n < d_state; ++n) {
        gate_log_(n, 0) = 0.0;
    }

    zero_grad();
}

// ----- zero_grad ------------------------------------------------------------
void S5Layer::zero_grad() {
    grad_A_re_.fill(0.0);
    grad_A_im_.fill(0.0);
    grad_B_re_.fill(0.0);
    grad_B_im_.fill(0.0);
    grad_C_re_.fill(0.0);
    grad_C_im_.fill(0.0);
    grad_D_skip_.fill(0.0);
    grad_gate_log_.fill(0.0);
    grad_Abar_re_acc_.fill(0.0);
    grad_Abar_im_acc_.fill(0.0);
    grad_Bbar_re_acc_.fill(0.0);
    grad_Bbar_im_acc_.fill(0.0);
}

// ----- parameters / gradients ----------------------------------------------
std::vector<Tensor*> S5Layer::parameters() {
    std::vector<Tensor*> p = {
        &A_re_, &A_im_,
        &B_re_, &B_im_,
        &C_re_, &C_im_,
        &D_skip_, &gate_log_
    };
    return p;
}

std::vector<Tensor*> S5Layer::gradients() {
    std::vector<Tensor*> g = {
        &grad_A_re_, &grad_A_im_,
        &grad_B_re_, &grad_B_im_,
        &grad_C_re_, &grad_C_im_,
        &grad_D_skip_, &grad_gate_log_
    };
    return g;
}

// ----- update_weights ------------------------------------------------------
void S5Layer::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& p, const Tensor& g) {
        for (size_t i = 0; i < p.rows; ++i)
            for (size_t j = 0; j < p.cols; ++j)
                p(i, j) -= learning_rate * g(i, j);
    };
    sgd(A_re_, grad_A_re_);
    sgd(A_im_, grad_A_im_);
    sgd(B_re_, grad_B_re_);
    sgd(B_im_, grad_B_im_);
    sgd(C_re_, grad_C_re_);
    sgd(C_im_, grad_C_im_);
    sgd(D_skip_, grad_D_skip_);
    sgd(gate_log_, grad_gate_log_);
}

// ----- forward --------------------------------------------------------------
//
// Input:  (T, d_model) real tensor.
// Output: (T, d_model) real tensor.
//
// Per channel (n ∈ [0, d_state)):
//   h_t[n] = Ā_n · h_{t-1}[n] + B̄_n · x_t   (for the forward direction)
//   y_t    = Re( C_n · h_t[n] ) summed over n  + D · x_t
//
// For bidirectional:
//   y_t^fwd = ... as above
//   y_t^bwd = ... running SSM backward over x (x_T, x_{T-1}, ..., x_0)
//   y_t     = sigmoid(gate[n]) · y_t^fwd + (1 - sigmoid(gate[n])) · y_t^bwd
//
// Notes on multi-channel broadcasting:
//   This implementation runs ONE channel per (d_model, d_state) pair — i.e.,
//   the SSM has d_model * d_state effective scalar recurrences. We reshape the
//   input as (T, d_model, d_state) where the per-(model, state) entry is the
//   MODEL-th channel's STATE-th SSM. We then run d_model × d_state parallel
//   recurrences, and sum across d_state to produce the (T, d_model) output.
//   This means A, B, C, gate are SHARED across d_model and only vary along
//   d_state. The Dense wrappers in S5Block handle cross-channel mixing.

Tensor S5Layer::forward(const Tensor& input) {
    const size_t T = input.rows;
    const size_t DM = input.cols;

    if (DM != d_model_) {
        throw std::invalid_argument("S5Layer: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("S5Layer: input must have at least one token");
    }

    last_input_ = input.clone();
    last_T_ = T;

    // Cache discretised params.
    for (size_t n = 0; n < d_state_; ++n) {
        std::complex<double> A_n(A_re_(n, 0), A_im_(n, 0));
        std::complex<double> B_n(B_re_(n, 0), B_im_(n, 0));
        std::complex<double> Abar = tustin_Abar(A_n, dt_);
        std::complex<double> Bbar = tustin_Bbar(B_n, A_n, dt_);
        Abar_re_(n, 0) = Abar.real();
        Abar_im_(n, 0) = Abar.imag();
        Bbar_re_(n, 0) = Bbar.real();
        Bbar_im_(n, 0) = Bbar.imag();
    }

    // Run forward SSM: cache h_t sequence (T × d_model × d_state complex).
    last_h_re_ = Tensor(T * d_model_ * d_state_, 1);
    last_h_im_ = Tensor(T * d_model_ * d_state_, 1);

    auto h_idx = [&](size_t t, size_t m, size_t n) {
        return (t * d_model_ + m) * d_state_ + n;
    };

    Tensor y(T, DM);
    const double Dval = D_skip_(0, 0);

    for (size_t t = 0; t < T; ++t) {
        // Multi-channel input: y_m = sum_n Re(C_n * h_t[m, n]) + D * x_t[m]
        // For now treat input as (T, DM) and broadcast SSM channels.
        // The h state for (m, n) starts at 0, then:
        //   h[m, n, t] = Ā_n * h[m, n, t-1] + B̄_n * x_t[m]
        for (size_t m = 0; m < DM; ++m) {
            const double x_tm = input(t, m);
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> Abar(Abar_re_(n, 0), Abar_im_(n, 0));
                std::complex<double> Bbar(Bbar_re_(n, 0), Bbar_im_(n, 0));
                std::complex<double> h_prev;
                if (t == 0) {
                    h_prev = std::complex<double>(0.0, 0.0);
                } else {
                    size_t prev = h_idx(t - 1, m, n);
                    h_prev = std::complex<double>(last_h_re_(prev, 0), last_h_im_(prev, 0));
                }
                std::complex<double> h_t_mn = Abar * h_prev + Bbar * x_tm;
                size_t cur = h_idx(t, m, n);
                last_h_re_(cur, 0) = h_t_mn.real();
                last_h_im_(cur, 0) = h_t_mn.imag();
            }
        }
        // Compute output y[t, m] = sum_n Re(C_n * h[m, n, t]) + D * x_t[m]
        for (size_t m = 0; m < DM; ++m) {
            std::complex<double> acc(0.0, 0.0);
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                size_t cur = h_idx(t, m, n);
                std::complex<double> h_mn(last_h_re_(cur, 0), last_h_im_(cur, 0));
                acc += C_n * h_mn;
            }
            y(t, m) = acc.real() + Dval * input(t, m);
        }
    }

    if (bidirectional_) {
        // Backward SSM: run over reversed input, then reverse the output.
        // We cache last_h_bwd_ as (T * d_model_ * d_state_, 1) with t in
        // reverse order: index k in [0, T) corresponds to forward-time T-1-k.
        last_h_bwd_re_ = Tensor(T * d_model_ * d_state_, 1);
        last_h_bwd_im_ = Tensor(T * d_model_ * d_state_, 1);
        last_T_bwd_ = T;

        // Run backward: t_bwd = 0 corresponds to forward-time t_fwd = T-1.
        for (size_t t_bwd = 0; t_bwd < T; ++t_bwd) {
            const size_t t_fwd = T - 1 - t_bwd;
            for (size_t m = 0; m < DM; ++m) {
                const double x_tm = input(t_fwd, m);
                for (size_t n = 0; n < d_state_; ++n) {
                    std::complex<double> Abar(Abar_re_(n, 0), Abar_im_(n, 0));
                    std::complex<double> Bbar(Bbar_re_(n, 0), Bbar_im_(n, 0));
                    std::complex<double> h_prev;
                    if (t_bwd == 0) {
                        h_prev = std::complex<double>(0.0, 0.0);
                    } else {
                        size_t prev = (t_bwd - 1) * d_model_ * d_state_ + m * d_state_ + n;
                        h_prev = std::complex<double>(last_h_bwd_re_(prev, 0), last_h_bwd_im_(prev, 0));
                    }
                    std::complex<double> h_t = Abar * h_prev + Bbar * x_tm;
                    size_t cur = t_bwd * d_model_ * d_state_ + m * d_state_ + n;
                    last_h_bwd_re_(cur, 0) = h_t.real();
                    last_h_bwd_im_(cur, 0) = h_t.imag();
                }
            }
        }

        // Mix forward and backward via per-channel gate.
        // gate[n] = sigmoid(gate_log_[n]) ∈ (0, 1).
        // output[t, m] = sum_n [ gate[n] * Re(C_n * h_fwd[t, m, n])
        //               + (1 - gate[n]) * Re(C_n * h_bwd[T-1-t, m, n]) ]
        //             + D * x[t, m]
        Tensor y_mixed(T, DM);
        for (size_t t = 0; t < T; ++t) {
            for (size_t m = 0; m < DM; ++m) {
                double acc = 0.0;
                for (size_t n = 0; n < d_state_; ++n) {
                    double gate = 1.0 / (1.0 + std::exp(-gate_log_(n, 0)));
                    std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                    size_t fwd_idx = h_idx(t, m, n);
                    std::complex<double> h_fwd(last_h_re_(fwd_idx, 0), last_h_im_(fwd_idx, 0));
                    // Backward SSM at "forward time" T-1-t = bwd-index t
                    size_t bwd_idx = t * d_model_ * d_state_ + m * d_state_ + n;
                    std::complex<double> h_bwd(last_h_bwd_re_(bwd_idx, 0), last_h_bwd_im_(bwd_idx, 0));
                    acc += gate * (C_n * h_fwd).real()
                         + (1.0 - gate) * (C_n * h_bwd).real();
                }
                y_mixed(t, m) = acc + Dval * input(t, m);
            }
        }
        return y_mixed;
    }

    return y;
}

// ----- backward -------------------------------------------------------------
//
// Given grad_output (T, DM):
//   1. Per channel n: dh_t = conj(C_n) · d_y_t for each (t, m).
//   2. Reverse sweep: for t = T-1 ... 0, accumulate dh_t · Ā_n into dh_{t-1}
//      (i.e., d_h[t-1] += Ā · d_h[t]).
//   3. dC_n = Σ_{t, m} conj(h[t, m, n]) · d_y[t, m].
//   4. dĀ_n = Σ_{t, m} d_h[t, m, n] · conj(h[t-1, m, n]).
//   5. dB̄_n = Σ_{t, m} d_h[t, m, n] · x[t, m] (real x — conjugates nothing).
//   6. Tustin chain rule: dA_n from dĀ_n and dB̄_n (per-channel scalar complex).
//   7. d_x[t, m] = Σ_n Re(conj(B̄_n) · dh[t, m, n]) + D · d_y[t, m].
// For bidirectional: gradient also flows through the backward SSM (independent
// parameters) and through the gate.
//
// To keep this tractable, we implement the BACKWARD for the unidirectional
// case first; bidirectional backward is in a follow-up commit.

Tensor S5Layer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t T = last_T_;
    const size_t DM = d_model_;
    if (T == 0) {
        return Tensor(grad_output.rows, grad_output.cols);
    }
    if (grad_output.rows != T || grad_output.cols != DM) {
        throw std::invalid_argument("S5Layer: grad_output shape mismatch");
    }

    // d_h[t, m, n] = conj(C_n) · d_y[t, m]  (allocate as flat tensor)
    Tensor dh_re(T * DM * d_state_, 1);
    Tensor dh_im(T * DM * d_state_, 1);
    dh_re.fill(0.0);
    dh_im.fill(0.0);

    auto h_idx = [&](size_t t, size_t m, size_t n) {
        return (t * DM + m) * d_state_ + n;
    };

    if (bidirectional_) {
        // For the bidirectional case, gradient flows through BOTH the forward
        // SSM and the backward SSM, scaled by gate[n] / (1-gate[n]) respectively.
        // We split d_y into two: d_y_fwd = gate * d_y, d_y_bwd = (1-gate) * d_y,
        // plus a gate-gradient term that adds (y_fwd - y_bwd) * d_y to the gate.
        // For simplicity in v1, run both directions with the same gradients as
        // unidirectional would see, scaled by gate / (1-gate).
        // Also accumulate dgate_log via d_y · (y_fwd - y_bwd) · σ'(gate_log).
        for (size_t t = 0; t < T; ++t) {
            for (size_t m = 0; m < DM; ++m) {
                const double dy = grad_output(t, m);
                for (size_t n = 0; n < d_state_; ++n) {
                    double gate_pre = gate_log_(n, 0);
                    double sig = 1.0 / (1.0 + std::exp(-gate_pre));
                    double sig_p = sig * (1.0 - sig);

                    // Recompute forward and backward h values for the gate-grad.
                    size_t fwd_idx = h_idx(t, m, n);
                    std::complex<double> h_fwd(last_h_re_(fwd_idx, 0), last_h_im_(fwd_idx, 0));
                    size_t bwd_idx = t * DM * d_state_ + m * d_state_ + n;
                    std::complex<double> h_bwd(last_h_bwd_re_(bwd_idx, 0), last_h_bwd_im_(bwd_idx, 0));
                    std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                    double diff = (C_n * h_fwd).real() - (C_n * h_bwd).real();
                    grad_gate_log_(n, 0) += dy * diff * sig_p;

                    // Forward direction gradient: scaled by gate.
                    std::complex<double> C_n_conj(C_n.real(), -C_n.imag());
                    std::complex<double> dh_t_fwd = C_n_conj * (sig * dy);
                    size_t cur_f = h_idx(t, m, n);
                    dh_re(cur_f, 0) += dh_t_fwd.real();
                    dh_im(cur_f, 0) += dh_t_fwd.imag();

                    // Backward direction gradient: scaled by (1 - gate).
                    std::complex<double> dh_t_bwd = C_n_conj * ((1.0 - sig) * dy);
                    size_t cur_b = bwd_idx;
                    dh_re(cur_b, 0) += dh_t_bwd.real();
                    dh_im(cur_b, 0) += dh_t_bwd.imag();
                }
            }
        }

        // Reverse sweep on FORWARD h: dh_{t-1} += Ā · dh_t.
        for (int t = static_cast<int>(T) - 1; t > 0; --t) {
            for (size_t m = 0; m < DM; ++m) {
                for (size_t n = 0; n < d_state_; ++n) {
                    std::complex<double> Abar(Abar_re_(n, 0), Abar_im_(n, 0));
                    size_t cur = h_idx(t, m, n);
                    std::complex<double> dh_t(dh_re(cur, 0), dh_im(cur, 0));
                    std::complex<double> propagated = Abar * dh_t;
                    size_t prev = h_idx(t - 1, m, n);
                    dh_re(prev, 0) += propagated.real();
                    dh_im(prev, 0) += propagated.imag();
                }
            }
        }

        // Reverse sweep on BACKWARD h: dh_{t_bwd-1} += Ā · dh_t_bwd.
        for (int t = static_cast<int>(T) - 1; t > 0; --t) {
            for (size_t m = 0; m < DM; ++m) {
                for (size_t n = 0; n < d_state_; ++n) {
                    std::complex<double> Abar(Abar_re_(n, 0), Abar_im_(n, 0));
                    size_t cur = t * DM * d_state_ + m * d_state_ + n;
                    std::complex<double> dh_t(dh_re(cur, 0), dh_im(cur, 0));
                    std::complex<double> propagated = Abar * dh_t;
                    size_t prev = (t - 1) * DM * d_state_ + m * d_state_ + n;
                    dh_re(prev, 0) += propagated.real();
                    dh_im(prev, 0) += propagated.imag();
                }
            }
        }

        // dC, dĀ, dB̄: same formulas but with the FORWARD h values.
        // (The backward-direction gradients contribute to A/B/C identically —
        //  both directions share the same parameters and the contribution
        //  structure is identical.)
        for (size_t t = 0; t < T; ++t) {
            for (size_t m = 0; m < DM; ++m) {
                for (size_t n = 0; n < d_state_; ++n) {
                    double gate = 1.0 / (1.0 + std::exp(-gate_log_(n, 0)));
                    std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                    std::complex<double> C_n_conj(C_n.real(), -C_n.imag());
                    size_t fwd_idx = h_idx(t, m, n);
                    std::complex<double> h_fwd(last_h_re_(fwd_idx, 0), last_h_im_(fwd_idx, 0));
                    std::complex<double> dh_fwd(dh_re(fwd_idx, 0), dh_im(fwd_idx, 0));
                    std::complex<double> h_bwd(last_h_bwd_re_(t * DM * d_state_ + m * d_state_ + n, 0),
                                                last_h_bwd_im_(t * DM * d_state_ + m * d_state_ + n, 0));
                    std::complex<double> dh_bwd(dh_re(t * DM * d_state_ + m * d_state_ + n, 0),
                                                dh_im(t * DM * d_state_ + m * d_state_ + n, 0));

                    // dC_n: gradient of Re(C·h) at output n, weighted by gate.
                    // y = gate * Re(C · h_fwd) + (1 - gate) * Re(C · h_bwd).
                    // ∂y/∂C_re = gate * h_fwd.re + (1 - gate) * h_bwd.re
                    // ∂y/∂C_im = -gate * h_fwd.im - (1 - gate) * h_bwd.im
                    grad_C_re_(n, 0) += gate * h_fwd.real() + (1.0 - gate) * h_bwd.real();
                    grad_C_im_(n, 0) += -gate * h_fwd.imag() - (1.0 - gate) * h_bwd.imag();

                    // dĀ_n, dB̄_n.
                    if (t > 0) {
                        std::complex<double> h_prev_fwd(last_h_re_(h_idx(t - 1, m, n), 0),
                                                         last_h_im_(h_idx(t - 1, m, n), 0));
                        std::complex<double> dAbar_t_fwd = dh_fwd * std::complex<double>(h_prev_fwd.real(), -h_prev_fwd.imag());
                        grad_Abar_re_acc_(n, 0) += dAbar_t_fwd.real();
                        grad_Abar_im_acc_(n, 0) += dAbar_t_fwd.imag();

                        std::complex<double> h_prev_bwd(last_h_bwd_re_((t - 1) * DM * d_state_ + m * d_state_ + n, 0),
                                                         last_h_bwd_im_((t - 1) * DM * d_state_ + m * d_state_ + n, 0));
                        std::complex<double> dAbar_t_bwd = dh_bwd * std::complex<double>(h_prev_bwd.real(), -h_prev_bwd.imag());
                        grad_Abar_re_acc_(n, 0) += dAbar_t_bwd.real();
                        grad_Abar_im_acc_(n, 0) += dAbar_t_bwd.imag();
                    }
                    // dB̄: dh · x_t (real x)
                    const double x_tm = last_input_(t, m);
                    grad_Bbar_re_acc_(n, 0) += dh_fwd.real() * x_tm;
                    grad_Bbar_im_acc_(n, 0) += dh_fwd.imag() * x_tm;
                    grad_Bbar_re_acc_(n, 0) += dh_bwd.real() * x_tm;
                    grad_Bbar_im_acc_(n, 0) += dh_bwd.imag() * x_tm;
                }
            }
        }

        // Tustin chain rule (per channel).
        for (size_t n = 0; n < d_state_; ++n) {
            std::complex<double> A_n(A_re_(n, 0), A_im_(n, 0));
            std::complex<double> B_n(B_re_(n, 0), B_im_(n, 0));
            std::complex<double> half_dt(0.5 * dt_, 0.0);
            std::complex<double> one(1.0, 0.0);
            std::complex<double> I_minus = one - half_dt * A_n;
            std::complex<double> I_plus = one + half_dt * A_n;
            std::complex<double> dAbar(grad_Abar_re_acc_(n, 0), grad_Abar_im_acc_(n, 0));
            std::complex<double> dBbar(grad_Bbar_re_acc_(n, 0), grad_Bbar_im_acc_(n, 0));
            std::complex<double> chain_Abar = half_dt * (I_plus + I_minus) / (I_minus * I_minus);
            std::complex<double> chain_Bbar = B_n * half_dt / (I_minus * I_minus);
            std::complex<double> dA_n = dAbar * chain_Abar + dBbar * chain_Bbar;
            grad_A_re_(n, 0) += dA_n.real();
            grad_A_im_(n, 0) += dA_n.imag();
            std::complex<double> dB_n = dBbar / I_minus;
            grad_B_re_(n, 0) += dB_n.real();
            grad_B_im_(n, 0) += dB_n.imag();
        }

        // d_x[t, m] = Σ_n Re(conj(B̄_n) · dh[t, m, n]) + D · d_y[t, m]
        Tensor d_input(T, DM);
        const double Dval = D_skip_(0, 0);
        for (size_t t = 0; t < T; ++t) {
            for (size_t m = 0; m < DM; ++m) {
                const double dy = grad_output(t, m);
                double acc = Dval * dy;
                for (size_t n = 0; n < d_state_; ++n) {
                    std::complex<double> Bbar_conj(Bbar_re_(n, 0), -Bbar_im_(n, 0));
                    // Forward direction contribution.
                    std::complex<double> dh_fwd(dh_re(h_idx(t, m, n), 0),
                                                dh_im(h_idx(t, m, n), 0));
                    acc += (Bbar_conj * dh_fwd).real();
                    // Backward direction contribution.
                    std::complex<double> dh_bwd(dh_re(t * DM * d_state_ + m * d_state_ + n, 0),
                                                dh_im(t * DM * d_state_ + m * d_state_ + n, 0));
                    acc += (Bbar_conj * dh_bwd).real();
                }
                d_input(t, m) = acc;
            }
        }
        // dD skip: Σ_{t, m} d_y[t, m] · x[t, m]
        for (size_t t = 0; t < T; ++t) {
            for (size_t m = 0; m < DM; ++m) {
                grad_D_skip_(0, 0) += grad_output(t, m) * last_input_(t, m);
            }
        }
        return d_input;
    }

    // ----- UNIDIRECTIONAL backward -----
    for (size_t t = 0; t < T; ++t) {
        for (size_t m = 0; m < DM; ++m) {
            const double dy = grad_output(t, m);
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                std::complex<double> C_n_conj(C_n.real(), -C_n.imag());
                std::complex<double> dh_t = C_n_conj * dy;
                size_t cur = h_idx(t, m, n);
                dh_re(cur, 0) += dh_t.real();
                dh_im(cur, 0) += dh_t.imag();
            }
        }
    }

    // Reverse sweep: dh_{t-1} += Ā · dh_t.
    for (int t = static_cast<int>(T) - 1; t > 0; --t) {
        for (size_t m = 0; m < DM; ++m) {
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> Abar(Abar_re_(n, 0), Abar_im_(n, 0));
                size_t cur = h_idx(t, m, n);
                std::complex<double> dh_t(dh_re(cur, 0), dh_im(cur, 0));
                std::complex<double> propagated = Abar * dh_t;
                size_t prev = h_idx(t - 1, m, n);
                dh_re(prev, 0) += propagated.real();
                dh_im(prev, 0) += propagated.imag();
            }
        }
    }

    // dC, dĀ, dB̄.
    for (size_t t = 0; t < T; ++t) {
        for (size_t m = 0; m < DM; ++m) {
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> C_n(C_re_(n, 0), C_im_(n, 0));
                size_t cur = h_idx(t, m, n);
                std::complex<double> h_t(last_h_re_(cur, 0), last_h_im_(cur, 0));
                std::complex<double> dh_t(dh_re(cur, 0), dh_im(cur, 0));

                // dC_n: dy_t = Re(C_n · h_t[n]) = C_re · h_re - C_im · h_im.
                //      ∂dy_t/∂C_re = h_re, ∂dy_t/∂C_im = -h_im.  (Real Wirtinger.)
                grad_C_re_(n, 0) += h_t.real();
                grad_C_im_(n, 0) += -h_t.imag();

                if (t > 0) {
                    std::complex<double> h_prev(last_h_re_(h_idx(t - 1, m, n), 0),
                                                 last_h_im_(h_idx(t - 1, m, n), 0));
                    std::complex<double> dAbar_t = dh_t * std::complex<double>(h_prev.real(), -h_prev.imag());
                    grad_Abar_re_acc_(n, 0) += dAbar_t.real();
                    grad_Abar_im_acc_(n, 0) += dAbar_t.imag();
                }
                const double x_tm = last_input_(t, m);
                grad_Bbar_re_acc_(n, 0) += dh_t.real() * x_tm;
                grad_Bbar_im_acc_(n, 0) += dh_t.imag() * x_tm;
            }
        }
    }

    // Tustin chain rule (per channel).
    for (size_t n = 0; n < d_state_; ++n) {
        std::complex<double> A_n(A_re_(n, 0), A_im_(n, 0));
        std::complex<double> B_n(B_re_(n, 0), B_im_(n, 0));
        std::complex<double> half_dt(0.5 * dt_, 0.0);
        std::complex<double> one(1.0, 0.0);
        std::complex<double> I_minus = one - half_dt * A_n;
        std::complex<double> I_plus = one + half_dt * A_n;
        std::complex<double> dAbar(grad_Abar_re_acc_(n, 0), grad_Abar_im_acc_(n, 0));
        std::complex<double> dBbar(grad_Bbar_re_acc_(n, 0), grad_Bbar_im_acc_(n, 0));
        std::complex<double> chain_Abar = half_dt * (I_plus + I_minus) / (I_minus * I_minus);
        std::complex<double> chain_Bbar = B_n * half_dt / (I_minus * I_minus);
        std::complex<double> dA_n = dAbar * chain_Abar + dBbar * chain_Bbar;
        grad_A_re_(n, 0) += dA_n.real();
        grad_A_im_(n, 0) += dA_n.imag();
        std::complex<double> dB_n = dBbar / I_minus;
        grad_B_re_(n, 0) += dB_n.real();
        grad_B_im_(n, 0) += dB_n.imag();
    }

    // d_x[t, m] = Σ_n Re(conj(B̄_n) · dh[t, m, n]) + D · d_y[t, m].
    Tensor d_input(T, DM);
    const double Dval = D_skip_(0, 0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t m = 0; m < DM; ++m) {
            const double dy = grad_output(t, m);
            double acc = Dval * dy;
            for (size_t n = 0; n < d_state_; ++n) {
                std::complex<double> Bbar_conj(Bbar_re_(n, 0), -Bbar_im_(n, 0));
                std::complex<double> dh_t(dh_re(h_idx(t, m, n), 0),
                                          dh_im(h_idx(t, m, n), 0));
                acc += (Bbar_conj * dh_t).real();
            }
            d_input(t, m) = acc;
        }
    }
    // dD skip: Σ_{t, m} d_y · x.
    for (size_t t = 0; t < T; ++t) {
        for (size_t m = 0; m < DM; ++m) {
            grad_D_skip_(0, 0) += grad_output(t, m) * last_input_(t, m);
        }
    }

    return d_input;
}

// ============================================================================
// S5Block
// ============================================================================

S5Block::S5Block(size_t d_model, size_t d_state, bool bidirectional, double dt)
    : dense_in_(d_model, d_model),
      s5_(d_model, d_state, bidirectional, dt),
      dense_out_(d_model, d_model) {}

Tensor S5Block::forward(const Tensor& input) {
    last_input_ = input.clone();
    last_proj_in_ = dense_in_.forward(input);
    Tensor ssm_out = s5_.forward(last_proj_in_);
    return dense_out_.forward(ssm_out);
}

Tensor S5Block::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad_ssm = dense_out_.backward(grad_output, learning_rate);
    Tensor grad_proj_in = s5_.backward(grad_ssm, learning_rate);
    return dense_in_.backward(grad_proj_in, learning_rate);
}

void S5Block::update_weights(double learning_rate) {
    dense_in_.update_weights(learning_rate);
    s5_.update_weights(learning_rate);
    dense_out_.update_weights(learning_rate);
}

void S5Block::zero_grad() {
    dense_in_.zero_grad();
    s5_.zero_grad();
    dense_out_.zero_grad();
}

std::vector<Tensor*> S5Block::parameters() {
    auto p = dense_in_.parameters();
    auto p2 = s5_.parameters();
    auto p3 = dense_out_.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), p3.begin(), p3.end());
    return p;
}

std::vector<Tensor*> S5Block::gradients() {
    auto g = dense_in_.gradients();
    auto g2 = s5_.gradients();
    auto g3 = dense_out_.gradients();
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), g3.begin(), g3.end());
    return g;
}