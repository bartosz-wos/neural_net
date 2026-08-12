#include "hawk.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// Hawk (RG-LRU) implementation
// ----------------------------------------------------------------------------
//
// Forward per channel c, per time step t:
//
//   gi_t         = W_x · x_t + b_x
//   g_t[c]       = sigmoid(gi_t[c])
//   inp_t[c]     = g_t[c] · x_t[c]
//   λ_t[c]       = exp(-exp(log_a_raw[c]) · exp(gi_t[c]))
//   h_t[c]       = λ_t[c] · h_{t-1}[c] + sqrt(1 - λ_t[c]²) · inp_t[c]
//   y_t[:]       = W_o · h_t + b_o
//
// Initialization (Griffin paper §3.1):
//   * W_x, W_o: xavier-uniform (Dense default).
//   * b_x, b_o: zero.
//   * log_a_raw: -3.0 init → exp(-3) ≈ 0.05 (small inner exponent).
//
// Backward pass derivation:
//   Let gH_t[c] = dL/dh_t[c]. The recurrence lets us compute gH backward:
//     gH[T] = 0
//     gH[t] = grad_h[t] + gH[t+1] · λ_{t+1}      for t = T-1, ..., 0
//
//   From h_t[c] = λ_t[c] · h_{t-1}[c] + sqrt(1-λ_t[c]²) · inp_t[c]:
//     grad_inp_t[c] = gH_t[c] · sqrt(1 - λ_t[c]²)
//     grad_λ_t[c]   = gH_t[c] · (h_{t-1}[c] - λ_t[c] · inp_t[c] / sqrt(1 - λ_t[c]²))
//
//   From inp_t[c] = g_t[c] · x_t[c]:
//     grad_g_t[c]        = grad_inp_t[c] · x_t[c]
//     grad_x_t_direct[c] = grad_inp_t[c] · g_t[c]      (direct path to x_t)
//
//   From g_t[c] = sigmoid(gi_t[c]):
//     grad_gi_t[c] += grad_g_t[c] · g_t[c] · (1 - g_t[c])
//
//   From λ_t[c] = exp(-exp(log_a_raw[c]) · exp(gi_t[c])):
//     dλ_t[c]/d(gi_t[c]) = -λ_t[c] · exp(log_a_raw[c]) · exp(gi_t[c])
//     grad_gi_t[c] += grad_λ_t[c] · (-λ_t[c] · exp(log_a_raw[c]) · exp(gi_t[c]))
//
//   From λ_t[c] = exp(-exp(log_a_raw[c]) · exp(gi_t[c])):
//     dλ_t[c]/d(log_a_raw[c]) = -λ_t[c] · exp(log_a_raw[c]) · exp(gi_t[c])
//     grad_log_a_raw[c] += grad_λ_t[c] · (-λ_t[c] · exp(log_a_raw[c]) · exp(gi_t[c]))
//
//   The grad_gi_t goes through W_x.backward to give grad_x_t (via W_x) and
//   accumulate W_x.grad_weights / W_x.grad_bias. The grad_x_t_direct
//   contribution is added directly to the returned grad_input.
//
// ============================================================================

// Helper: clamp exp(gi) to avoid overflow in the lambda exponent.
static inline double safe_exp(double x) {
    if (x > 50.0) return std::exp(50.0);
    return std::exp(x);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
HawkBlock::HawkBlock(size_t d)
    : d_(d),
      W_x(d, d),
      log_a_raw(1, d),
      W_o(d, d),
      grad_log_a_raw_(1, d) {
    if (d == 0) throw std::invalid_argument("HawkBlock: d must be > 0");

    // Init log_a_raw to -3.0 (Griffin paper §3.1)
    for (size_t i = 0; i < d; ++i) {
        log_a_raw[0][i] = -3.0;
        grad_log_a_raw_[0][i] = 0.0;
    }

    // Caches will be set in forward; just zero_Grad the parameters.
    W_x.zero_grad();
    W_o.zero_grad();
}

// ----------------------------------------------------------------------------
// Forward pass
// ----------------------------------------------------------------------------
Tensor HawkBlock::forward(const Tensor& input) {
    if (input.cols != d_) {
        throw std::invalid_argument("HawkBlock::forward: input cols must be d");
    }
    const size_t T = input.rows;

    // Cache the input (we re-use it in backward).
    last_input_ = input;
    last_gate_input_ = W_x.forward(input);  // (T, d)
    last_gates_ = Tensor(T, d_);
    last_lambda_ = Tensor(T, d_);
    last_input_proj_ = Tensor(T, d_);
    last_h_ = Tensor(T + 1, d_);
    for (size_t c = 0; c < d_; ++c) last_h_[0][c] = 0.0;  // h_0 = 0

    Tensor hidden(T, d_);  // (T, d) — pre-output hidden state
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_; ++c) {
            const double gi = last_gate_input_[t][c];
            const double g = sigmoid(gi);
            last_gates_[t][c] = g;

            // a_c = exp(log_a_raw[c]); exp_gi = exp(gi_t[c])
            const double a_base = std::exp(log_a_raw[0][c]);
            const double exp_gi = safe_exp(gi);
            const double lam = std::exp(-a_base * exp_gi);
            last_lambda_[t][c] = lam;

            // input_proj_t = gate_t * x_t
            const double inp = g * last_input_[t][c];
            last_input_proj_[t][c] = inp;

            // h_t = λ_t * h_{t-1} + sqrt(1 - λ_t²) * input_proj_t
            const double sqrt_term = std::sqrt(1.0 - lam * lam);
            const double h_prev = last_h_[t][c];
            const double h_new = lam * h_prev + sqrt_term * inp;
            last_h_[t + 1][c] = h_new;
            hidden[t][c] = h_new;
        }
    }

    // Output projection
    Tensor y = W_o.forward(hidden);  // (T, d)
    return y;
}

// ----------------------------------------------------------------------------
// Backward pass
// ----------------------------------------------------------------------------
Tensor HawkBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.cols != d_) {
        throw std::invalid_argument("HawkBlock::backward: grad_output cols must be d");
    }
    const size_t T = grad_output.rows;

    // 1) Output projection backward: returns grad_h (gradient at hidden state).
    Tensor grad_h = W_o.backward(grad_output, 0.0);  // (T, d)

    // 2) Compute gH[t] for all t by reverse sweep through the recurrence.
    //    For t = T-1, ..., 0:
    //      gH[t] = grad_h[t] + gH[t+1] * λ_{t+1}
    //    (i.e. the carrier from h_{t+1} to h_t uses λ_{t+1}, because
    //     h_{t+1} = λ_{t+1} · h_t + ...).
    //    Set gH[T] = 0 (placeholder; not used).
    Tensor gH = Tensor(T + 1, d_);
    for (size_t c = 0; c < d_; ++c) gH[T][c] = 0.0;
    if (T > 0) {
        for (int t_signed = static_cast<int>(T) - 1; t_signed >= 0; --t_signed) {
            const size_t t = static_cast<size_t>(t_signed);
            for (size_t c = 0; c < d_; ++c) {
                // gH[t] = grad_h[t] + gH[t+1] · λ_{t+1}
                // For t = T-1, t+1 = T, gH[T] = 0 (placeholder).
                // For t < T-1, the carrier is gH[t+1] * λ_{t+1}.
                const double lam_tplus1 = (t + 1 < T) ? last_lambda_[t + 1][c] : 0.0;
                gH[t][c] = grad_h[t][c] + gH[t + 1][c] * lam_tplus1;
            }
        }
    }

    // 3) Per-channel accumulators.
    Tensor grad_input_proj(T, d_);  // grad wrt input_proj_t
    Tensor grad_lambda(T, d_);      // grad wrt λ_t
    Tensor grad_gate_input(T, d_);  // grad wrt gi_t
    Tensor grad_input(T, d_);       // grad wrt x_t (direct path: input_proj = gate * x)

    // Placeholder for grad gate_input combined with W_x backward.
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_; ++c) {
            const double lam = last_lambda_[t][c];
            const double sqrt_term = std::sqrt(1.0 - lam * lam);
            const double inp_proj = last_input_proj_[t][c];
            const double h_prev = last_h_[t][c];
            const double gH_t = gH[t][c];

            // grad_inp_proj_t[c] = gH_t[c] · sqrt(1 - λ_t[c]²)
            const double g_inp = gH_t * sqrt_term;
            grad_input_proj[t][c] = g_inp;

            // grad_λ_t[c] = gH_t[c] · (h_{t-1}[c] - λ_t[c] · inp_t[c] / sqrt(1 - λ_t[c]²))
            const double g_lam = gH_t * (h_prev - lam * inp_proj / sqrt_term);
            grad_lambda[t][c] = g_lam;

            // grad_gate_t[c] = grad_inp_proj_t[c] · x_t[c]   (from input_proj = gate * x_t)
            const double grad_gate = g_inp * last_input_[t][c];
            // grad_gi_t_via_gate = grad_gate · gate · (1 - gate)
            const double g = last_gates_[t][c];
            const double grad_gi_via_gate = grad_gate * g * (1.0 - g);

            // grad_gi_t_via_λ = grad_λ · (-λ · a_c · exp_gi)
            const double a_base = std::exp(log_a_raw[0][c]);
            const double exp_gi = safe_exp(last_gate_input_[t][c]);
            const double d_lam_dgi = -lam * a_base * exp_gi;
            const double grad_gi_via_lam = g_lam * d_lam_dgi;

            grad_gate_input[t][c] = grad_gi_via_gate + grad_gi_via_lam;

            // grad_x_t (direct path through input_proj = gate * x): grad_inp_proj * gate
            grad_input[t][c] = g_inp * g;
        }
    }

    // 4) Feed grad_gate_input through W_x backward to get grad_x_via_Wx and
    //    accumulate W_x.grad_weights/bias.
    Tensor grad_x_via_Wx = W_x.backward(grad_gate_input, 0.0);  // (T, d)

    // 5) Combine grad_input (direct path) with grad_x_via_Wx.
    Tensor grad_input_full(T, d_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_; ++c) {
            grad_input_full[t][c] = grad_input[t][c] + grad_x_via_Wx[t][c];
        }
    }

    // 6) Accumulate grad_log_a_raw: dL/d(log_a_raw) = sum_t grad_λ_t · dλ_t/dlog_a_raw
    //    dλ/dlog_a_raw = -λ · a_c · exp_gi_t
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < d_; ++c) {
            const double lam = last_lambda_[t][c];
            const double a_base = std::exp(log_a_raw[0][c]);
            const double exp_gi = safe_exp(last_gate_input_[t][c]);
            const double d_lam_dlog_a_raw = -lam * a_base * exp_gi;
            grad_log_a_raw_[0][c] += grad_lambda[t][c] * d_lam_dlog_a_raw;
        }
    }

    return grad_input_full;
}

// ----------------------------------------------------------------------------
// Update weights
// ----------------------------------------------------------------------------
void HawkBlock::update_weights(double learning_rate) {
    W_x.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    for (size_t i = 0; i < d_; ++i) {
        log_a_raw[0][i] -= learning_rate * grad_log_a_raw_[0][i];
    }
}

// ----------------------------------------------------------------------------
// Zero gradients
// ----------------------------------------------------------------------------
void HawkBlock::zero_grad() {
    W_x.zero_grad();
    W_o.zero_grad();
    for (size_t i = 0; i < d_; ++i) grad_log_a_raw_[0][i] = 0.0;
}

// ----------------------------------------------------------------------------
// Parameters / gradients
// ----------------------------------------------------------------------------
std::vector<Tensor*> HawkBlock::parameters() {
    return {&W_x.weights, &W_x.bias, &log_a_raw, &W_o.weights, &W_o.bias};
}

std::vector<Tensor*> HawkBlock::gradients() {
    return {&W_x.grad_weights, &W_x.grad_bias, &grad_log_a_raw_,
            &W_o.grad_weights, &W_o.grad_bias};
}
