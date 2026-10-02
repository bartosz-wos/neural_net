#include "complex_ema.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

// ============================================================================
// Constructor
// ============================================================================
ComplexEMA::ComplexEMA(size_t d_model, size_t ndim)
    : d_model_(d_model), ndim_(ndim) {
    if (d_model == 0) throw std::invalid_argument("ComplexEMA: d_model must be > 0");
    if (ndim == 0) throw std::invalid_argument("ComplexEMA: ndim must be > 0");

    // Near-zero init keeps |q| = 1 - sigmoid(a)*sigmoid(d) close to 1 (long
    // memory) and spreads the phase evenly across the ndim rotations.
    alpha_  = Tensor::random(d_model_, ndim_, 0.01);
    delta_  = Tensor::random(d_model_, ndim_, 0.01);
    theta_  = Tensor::random(d_model_, 1, 0.01);
    eta_re_ = Tensor::random(d_model_, ndim_, 0.5);
    eta_im_ = Tensor::random(d_model_, ndim_, 0.5);
    omega_  = Tensor::random(d_model_, 1, 0.01);

    grad_alpha_  = Tensor::zeros(d_model_, ndim_);
    grad_delta_  = Tensor::zeros(d_model_, ndim_);
    grad_theta_  = Tensor::zeros(d_model_, 1);
    grad_eta_re_ = Tensor::zeros(d_model_, ndim_);
    grad_eta_im_ = Tensor::zeros(d_model_, ndim_);
    grad_omega_  = Tensor::zeros(d_model_, 1);
}

// ============================================================================
// compute_coeffs — p, q, eta derived from the current parameters.
// ============================================================================
void ComplexEMA::compute_coeffs() {
    const double scale = eta_scale();
    const double step = 2.0 * M_PI / static_cast<double>(ndim_);

    p_ = Tensor::zeros(d_model_, ndim_);
    q_re_ = Tensor::zeros(d_model_, ndim_);
    q_im_ = Tensor::zeros(d_model_, ndim_);
    eta_ = Tensor::zeros(d_model_, 2 * ndim_);

    for (size_t j = 0; j < d_model_; ++j) {
        // Eq. 3: one theta_j is spread over ndim phases by the (k+1) factor.
        const double base = sigmoid(theta_[j][0]) * step;
        for (size_t k = 0; k < ndim_; ++k) {
            const double p_jk = sigmoid(alpha_[j][k]);
            const double d_jk = sigmoid(delta_[j][k]);
            const double phi  = static_cast<double>(k + 1) * base;
            const double mag  = 1.0 - p_jk * d_jk;

            p_[j][k]    = p_jk;
            q_re_[j][k] = mag * std::cos(phi);
            q_im_[j][k] = mag * std::sin(phi);

            // eta is stored interleaved: [re0, im0, re1, im1, ...]
            eta_[j][2 * k]     = eta_re_[j][k] * scale;
            eta_[j][2 * k + 1] = eta_im_[j][k] * scale;
        }
    }
}

// ============================================================================
// forward — causal scan over the sequence dimension.
// ============================================================================
Tensor ComplexEMA::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("ComplexEMA::forward: input.cols must equal d_model");
    }
    const size_t n = input.rows;
    compute_coeffs();

    last_x_ = input;
    h_state_ = Tensor::zeros(d_model_, 2 * ndim_);      // (d, 2h) interleaved
    h_hist_  = Tensor::zeros(n, d_model_ * 2 * ndim_); // (n, d*2h) flattened
    Tensor out(n, d_model_);

    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < ndim_; ++k) {
                const size_t c = 2 * k;  // interleaved column for real part
                const double hr = h_state_[j][c];
                const double hi = h_state_[j][c + 1];
                // h_t = p * x_t + q * h_{t-1}   (complex multiply)
                const double nr = p_[j][k] * input[t][j] +
                                  q_re_[j][k] * hr - q_im_[j][k] * hi;
                const double ni = q_re_[j][k] * hi + q_im_[j][k] * hr;
                h_state_[j][c] = nr;
                h_state_[j][c + 1] = ni;
                h_hist_[t][j * 2 * ndim_ + c] = nr;
                h_hist_[t][j * 2 * ndim_ + c + 1] = ni;

                // Re(eta * h) = eta_re*h_re - eta_im*h_im
                acc += eta_[j][c] * nr - eta_[j][c + 1] * ni;
            }
            out[t][j] = acc + omega_[j][0] * input[t][j];
        }
    }
    return out;
}

// ============================================================================
// backward
//
// The adjoint of the hidden state is a closed-form sum over ALL future
// timesteps, because h_t reaches every output y_s (s >= t) through the complex
// recurrence:
//
//   adj(h_t) = sum_{s>=t} G[s] * conj(eta) * conj(q)^(s-t)
//   G[t]     = grad_output[t][j] * y_t
//
// with conj(eta) = (eta_re, -eta_im) and conj(q) = (q_re, -q_im).
// Evaluating this for every t is O(n^2 * d * h); with n the sequence length
// this dominates the layer's cost for long sequences.
// ============================================================================
Tensor ComplexEMA::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_x_.rows || grad_output.cols != d_model_) {
        throw std::invalid_argument("ComplexEMA::backward: grad_output shape mismatch");
    }
    const size_t n = last_x_.rows;
    const size_t w2 = 2 * ndim_;

    grad_alpha_  = Tensor::zeros(d_model_, ndim_);
    grad_delta_  = Tensor::zeros(d_model_, ndim_);
    grad_theta_  = Tensor::zeros(d_model_, 1);
    grad_eta_re_ = Tensor::zeros(d_model_, ndim_);
    grad_eta_im_ = Tensor::zeros(d_model_, ndim_);
    grad_omega_  = Tensor::zeros(d_model_, 1);
    grad_x_      = Tensor::zeros(n, d_model_);

    // G[t] is simply dL/dy for this timestep — the Layer interface hands us
    // the upstream gradient directly (same convention as RWKV6TimeMix::
    // backward, which does `W_o.grad_weights(j,i) += grad_output(t,j) * ...`
    // with no re-derivation of y). Do NOT multiply by the cached y again:
    // that would apply dL/dy -> (dL/dy) * y, which is a different chain rule.
    std::vector<double> G(n * d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            G[t * d_model_ + j] = grad_output[t][j];
        }
    }

    const double step = 2.0 * M_PI / static_cast<double>(ndim_);

    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < ndim_; ++k) {
            const double e0 =  eta_[j][2 * k];      // eta_re * scale
            const double e1 = -eta_[j][2 * k + 1];  // conj(eta) imag part
            const double c0 =  q_re_[j][k];          // conj(q) real part
            const double c1 = -q_im_[j][k];          // conj(q) imag part

            const double sig_theta = sigmoid(theta_[j][0]);
            const double sig_delta = sigmoid(delta_[j][k]);  // mag = 1 - p*sig_delta
            const double phi = static_cast<double>(k + 1) * sig_theta * step;
            const double cph = std::cos(phi), sph = std::sin(phi);
            const double mag = q_re_[j][k] * cph + q_im_[j][k] * sph;  // == 1 - p*delta

            // adj(h_t) = sum_{s>=t} G[s] * conj(eta) * conj(q)^(s-t)
            for (size_t t = 0; t < n; ++t) {
                double pr = 1.0, pi = 0.0;          // conj(q) running power
                double ar = 0.0, ai = 0.0;          // adj(h_t) for this (j,k)
                for (size_t s = t; s < n; ++s) {
                    const double tr = e0 * pr - e1 * pi;
                    const double ti = e0 * pi + e1 * pr;
                    const double g = G[s * d_model_ + j];
                    ar += g * tr;
                    ai += g * ti;
                    const double nr = pr * c0 - pi * c1;
                    const double ni = pr * c1 + pi * c0;
                    pr = nr; pi = ni;
                }

                // Direct p path: h_t = p*x_t + ...  => dL/dA += x_t * ar * dp/dA
                const double dp_da = p_[j][k] * (1.0 - p_[j][k]);
                grad_x_[t][j] += p_[j][k] * ar;
                grad_alpha_[j][k] += last_x_[t][j] * ar * dp_da;

                if (t == 0) continue;   // q multiplies h_{-1} = 0

                // q path: dL/dq = adj(h_t) . conj(h_{t-1})
                const double hr = h_hist_[t - 1][j * w2 + 2 * k];
                const double hi = h_hist_[t - 1][j * w2 + 2 * k + 1];
                const double dqr =  ar * hr + ai * hi;
                const double dqi = -ar * hi + ai * hr;

                // q = mag * e^{i phi}:  dq/dmag = (cos, sin), dq/dphi = mag*(-sin, cos)
                const double d_mag = dqr * cph + dqi * sph;
                const double d_phi = dqr * (-mag * sph) + dqi * (mag * cph);

                // mag = 1 - p*sig_delta  (note: sigmoid(delta), NOT delta)
                grad_alpha_[j][k] += d_mag * (-sig_delta * dp_da);
                grad_delta_[j][k] += d_mag * (-p_[j][k] * sig_delta *
                                              (1.0 - sig_delta));
                // phi = (k+1) * sigmoid(theta_j) * step
                grad_theta_[j][0] += d_phi * (static_cast<double>(k + 1) * step) *
                                     sig_theta * (1.0 - sig_theta);
            }
        }

        // eta and omega gradients depend only on the state at their own
        // timestep, so they need no adjoint sum.
        for (size_t t = 0; t < n; ++t) {
            for (size_t k = 0; k < ndim_; ++k) {
                const double g = G[t * d_model_ + j];
                grad_eta_re_[j][k] += g * h_hist_[t][j * w2 + 2 * k] * eta_scale();
                grad_eta_im_[j][k] -= g * h_hist_[t][j * w2 + 2 * k + 1] * eta_scale();
            }
            grad_omega_[j][0] += G[t * d_model_ + j] * last_x_[t][j];
            grad_x_[t][j] += G[t * d_model_ + j] * omega_[j][0];
        }
    }
    return grad_x_.clone();
}

// ============================================================================
// update_weights / zero_grad
// ============================================================================
void ComplexEMA::update_weights(double lr) {
    alpha_  -= grad_alpha_  * lr;
    delta_  -= grad_delta_  * lr;
    theta_  -= grad_theta_  * lr;
    eta_re_ -= grad_eta_re_ * lr;
    eta_im_ -= grad_eta_im_ * lr;
    omega_  -= grad_omega_  * lr;
}

void ComplexEMA::zero_grad() {
    grad_alpha_  = Tensor::zeros(d_model_, ndim_);
    grad_delta_  = Tensor::zeros(d_model_, ndim_);
    grad_theta_  = Tensor::zeros(d_model_, 1);
    grad_eta_re_ = Tensor::zeros(d_model_, ndim_);
    grad_eta_im_ = Tensor::zeros(d_model_, ndim_);
    grad_omega_  = Tensor::zeros(d_model_, 1);
}

std::vector<Tensor*> ComplexEMA::parameters() {
    return {&alpha_, &delta_, &theta_, &eta_re_, &eta_im_, &omega_};
}

std::vector<Tensor*> ComplexEMA::gradients() {
    return {&grad_alpha_, &grad_delta_, &grad_theta_,
            &grad_eta_re_, &grad_eta_im_, &grad_omega_};
}
