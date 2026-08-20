#include <iostream>
#include "linoss.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace {

// Compute Ax where A is the block-diagonal oscillator matrix
// for a (2*d_osc)-dim state. A is built from per-channel damping and omega.
inline void compute_Ax(const Tensor& x,
                       const std::vector<double>& damping,  // exp(log_damping[k]), size d_osc
                       const std::vector<double>& omega,    // freq[k], size d_osc
                       Tensor& Ax_out) {
    int two_d = (int)x.data.size();
    int d_osc = two_d / 2;
    for (int k = 0; k < d_osc; ++k) {
        double lam = damping[k];
        double om = omega[k];
        double x0 = x.data[2 * k];
        double x1 = x.data[2 * k + 1];
        // A_k = [[-lam, om], [-om, -lam]]
        Ax_out.data[2 * k]     = -lam * x0 + om * x1;
        Ax_out.data[2 * k + 1] = -om * x0 - lam * x1;
    }
}

// Compute (A^T) x for block-diagonal A. A^T block_k = [[-lam, -om], [om, -lam]].
inline void compute_ATx(const Tensor& x,
                        const std::vector<double>& damping,
                        const std::vector<double>& omega,
                        Tensor& ATx_out) {
    int two_d = (int)x.data.size();
    int d_osc = two_d / 2;
    for (int k = 0; k < d_osc; ++k) {
        double lam = damping[k];
        double om = omega[k];
        double x0 = x.data[2 * k];
        double x1 = x.data[2 * k + 1];
        ATx_out.data[2 * k]     = -lam * x0 - om * x1;
        ATx_out.data[2 * k + 1] = om * x0 - lam * x1;
    }
}

// For IM-LinOSS: given M = (I - dt/2 A), solve M y = rhs via 2×2 block inversion.
inline void solve_Minv_y(const std::vector<double>& damping,
                         const std::vector<double>& omega,
                         double dt,
                         const Tensor& rhs,
                         Tensor& y_out) {
    int two_d = (int)rhs.data.size();
    int d_osc = two_d / 2;
    double h = dt / 2.0;
    for (int k = 0; k < d_osc; ++k) {
        double lam = damping[k];
        double om = omega[k];
        // M_k = [[1 + h*lam, -h*om], [h*om, 1 + h*lam]]
        double a = 1.0 + h * lam;
        double b = -h * om;
        double c = h * om;
        double d = a;
        double det = a * d - b * c;  // a^2 + h^2 om^2
        double r0 = rhs.data[2 * k];
        double r1 = rhs.data[2 * k + 1];
        y_out.data[2 * k]     = (d * r0 - b * r1) / det;
        y_out.data[2 * k + 1] = (-c * r0 + a * r1) / det;
    }
}

// Same as solve_Minv_y but for M^T: blocks [[1 + h*lam, h*om], [-h*om, 1 + h*lam]].
inline void solve_MTinv_y(const std::vector<double>& damping,
                          const std::vector<double>& omega,
                          double dt,
                          const Tensor& rhs,
                          Tensor& y_out) {
    int two_d = (int)rhs.data.size();
    int d_osc = two_d / 2;
    double h = dt / 2.0;
    for (int k = 0; k < d_osc; ++k) {
        double lam = damping[k];
        double om = omega[k];
        double a = 1.0 + h * lam;
        double b = h * om;
        double c = -h * om;
        double d = a;
        double det = a * d - b * c;
        double r0 = rhs.data[2 * k];
        double r1 = rhs.data[2 * k + 1];
        y_out.data[2 * k]     = (d * r0 - b * r1) / det;
        y_out.data[2 * k + 1] = (-c * r0 + a * r1) / det;
    }
}

inline double softplus(double x) { return std::log1p(std::exp(x)); }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

// ============================================================================
// LinOSSCell
// ============================================================================

LinOSSCell::LinOSSCell(int d_input, int d_output, int d_osc, LinOSSType type)
    : d_input_(d_input), d_output_(d_output), d_osc_(d_osc), type_(type),
      B(d_input, 2 * d_osc),
      C(2 * d_osc, d_output),
      log_damping(d_osc, 1),
      freq(d_osc, 1),
      log_dt(1, 1),
      bias(d_output, 1),
      grad_B(d_input, 2 * d_osc),
      grad_C(2 * d_osc, d_output),
      grad_log_damping(d_osc, 1),
      grad_freq(d_osc, 1),
      grad_log_dt(1, 1),
      grad_bias(d_output, 1) {
    if (d_input <= 0 || d_output <= 0 || d_osc <= 0) {
        throw std::invalid_argument("LinOSSCell: d_input, d_output, d_osc must be positive");
    }

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 1.0);

    double scale_B = std::sqrt(2.0 / (double)d_input);
    for (int i = 0; i < d_input; ++i)
        for (int j = 0; j < 2 * d_osc; ++j)
            B[i][j] = dis(gen) * scale_B;

    double scale_C = std::sqrt(2.0 / (double)(2 * d_osc));
    for (int i = 0; i < 2 * d_osc; ++i)
        for (int j = 0; j < d_output; ++j)
            C[i][j] = dis(gen) * scale_C;

    // exp(log_damping) ≈ 0.5 → mild decay
    for (int k = 0; k < d_osc; ++k) log_damping[k][0] = std::log(0.5);

    for (int k = 0; k < d_osc; ++k) freq[k][0] = 0.5 + 0.3 * (double)k;

    // softplus(log_dt) ≈ 1.0 → log_dt = log(e - 1)
    log_dt[0][0] = std::log(std::exp(1.0) - 1.0);

    bias.fill(0.0);
    zero_grad();
    reset_state();
}

void LinOSSCell::reset_state() {
    last_input_ = Tensor();
    last_u_proj_ = Tensor();
    states_.clear();
    Ax_cache_.clear();
}

std::vector<Tensor*> LinOSSCell::parameters() {
    return {&B, &C, &log_damping, &freq, &log_dt, &bias};
}

std::vector<Tensor*> LinOSSCell::gradients() {
    return {&grad_B, &grad_C, &grad_log_damping, &grad_freq, &grad_log_dt, &grad_bias};
}

void LinOSSCell::zero_grad() {
    grad_B.fill(0.0);
    grad_C.fill(0.0);
    grad_log_damping.fill(0.0);
    grad_freq.fill(0.0);
    grad_log_dt.fill(0.0);
    grad_bias.fill(0.0);
}

void LinOSSCell::update_weights(double lr) {
    for (size_t i = 0; i < B.data.size(); ++i) B.data[i] -= lr * grad_B.data[i];
    for (size_t i = 0; i < C.data.size(); ++i) C.data[i] -= lr * grad_C.data[i];
    for (size_t i = 0; i < log_damping.data.size(); ++i) log_damping.data[i] -= lr * grad_log_damping.data[i];
    for (size_t i = 0; i < freq.data.size(); ++i) freq.data[i] -= lr * grad_freq.data[i];
    log_dt[0][0] -= lr * grad_log_dt[0][0];
    for (size_t i = 0; i < bias.data.size(); ++i) bias.data[i] -= lr * grad_bias.data[i];
}

Tensor LinOSSCell::forward(const Tensor& input) {
    int T = (int)input.rows;
    int two_d = 2 * d_osc_;

    last_input_ = input.clone();

    std::vector<double> damping(d_osc_), omega(d_osc_);
    for (int k = 0; k < d_osc_; ++k) {
        damping[k] = std::exp(log_damping[k][0]);
        omega[k] = freq[k][0];
    }
    double dt = softplus(log_dt[0][0]);

    // Pre-compute u_proj[t] = B · input[t], shape (T, two_d). Cached.
    Tensor u_proj(T, two_d);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < two_d; ++j) {
            double s = 0.0;
            for (int i = 0; i < d_input_; ++i) s += B[i][j] * input[t][i];
            u_proj[t][j] = s;
        }
    }
    last_u_proj_ = u_proj.clone();

    states_.clear();
    Ax_cache_.clear();
    states_.reserve(T + 1);
    Ax_cache_.reserve(T);

    Tensor x;
    x.rows = two_d; x.cols = 1;
    x.data.assign(two_d, 0.0);
    states_.push_back(x.clone());

    for (int t = 0; t < T; ++t) {
        if (type_ == LinOSSType::IMPLICIT_MIDPOINT) {
            // rhs = (I + dt/2 A) x_t + dt · u_proj[t]
            Tensor rhs;
            rhs.rows = two_d; rhs.cols = 1;
            rhs.data.assign(two_d, 0.0);
            double h = dt / 2.0;
            for (int k = 0; k < d_osc_; ++k) {
                double lam = damping[k];
                double om = omega[k];
                double x0 = x.data[2 * k];
                double x1 = x.data[2 * k + 1];
                double p0 = (1.0 - h * lam) * x0 + h * om * x1;
                double p1 = -h * om * x0 + (1.0 - h * lam) * x1;
                rhs.data[2 * k]     = p0 + dt * u_proj[t][2 * k];
                rhs.data[2 * k + 1] = p1 + dt * u_proj[t][2 * k + 1];
            }
            Tensor x_next;
            x_next.rows = two_d; x_next.cols = 1;
            x_next.data.assign(two_d, 0.0);
            solve_Minv_y(damping, omega, dt, rhs, x_next);
            // Cache A · x_t (used for grad_log_damping / grad_freq paths).
            Tensor Ax;
            Ax.rows = two_d; Ax.cols = 1;
            Ax.data.assign(two_d, 0.0);
            compute_Ax(x, damping, omega, Ax);
            Ax_cache_.push_back(Ax);
            x = x_next;
        } else {  // HEUN
            // k1 = A x_t + dt · u_proj[t]
            // x_hat = x_t + k1
            // k2 = A x_hat + dt · u_proj[t]
            // x_{t+1} = x_t + dt/2 (k1 + k2)
            Tensor Ax;
            Ax.rows = two_d; Ax.cols = 1;
            Ax.data.assign(two_d, 0.0);
            compute_Ax(x, damping, omega, Ax);
            // k1 = Ax + dt·u_proj
            Tensor k1;
            k1.rows = two_d; k1.cols = 1;
            k1.data.resize(two_d);
            for (int i = 0; i < two_d; ++i)
                k1.data[i] = Ax.data[i] + dt * u_proj[t][i];
            Tensor x_hat;
            x_hat.rows = two_d; x_hat.cols = 1;
            x_hat.data.resize(two_d);
            for (int i = 0; i < two_d; ++i)
                x_hat.data[i] = x.data[i] + k1.data[i];
            Tensor Ax_hat;
            Ax_hat.rows = two_d; Ax_hat.cols = 1;
            Ax_hat.data.assign(two_d, 0.0);
            compute_Ax(x_hat, damping, omega, Ax_hat);
            Tensor x_next;
            x_next.rows = two_d; x_next.cols = 1;
            x_next.data.resize(two_d);
            double h = dt / 2.0;
            for (int i = 0; i < two_d; ++i)
                x_next.data[i] = x.data[i] + h * (k1.data[i] + (Ax_hat.data[i] + dt * u_proj[t][i]));
            Ax_cache_.push_back(Ax);  // A · x_t cached for backward
            x = x_next;
        }
        states_.push_back(x.clone());
    }

    Tensor out(T, d_output_);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < d_output_; ++j) {
            double s = bias[j][0];
            for (int i = 0; i < two_d; ++i) s += C[i][j] * states_[t + 1].data[i];
            out[t][j] = s;
        }
    }
    return out;
}

Tensor LinOSSCell::backward(const Tensor& grad_output, double /*learning_rate*/) {
    int T = (int)last_input_.rows;
    int two_d = 2 * d_osc_;

    zero_grad();

    std::vector<double> damping(d_osc_), omega(d_osc_);
    for (int k = 0; k < d_osc_; ++k) {
        damping[k] = std::exp(log_damping[k][0]);
        omega[k] = freq[k][0];
    }
    double dt = softplus(log_dt[0][0]);
    double s_dt = sigmoid(log_dt[0][0]);  // d(softplus(log_dt))/d(log_dt)

    // grad_bias: sum_t grad_output[t]
    for (int j = 0; j < d_output_; ++j) {
        double s = 0.0;
        for (int t = 0; t < T; ++t) s += grad_output[t][j];
        grad_bias[j][0] = s;
    }

    // ---------- Single BPTT pass for grad_C, grad_B, grad_log_damping, grad_freq, grad_log_dt ----------
    // dstate carries the cumulative dL/dx_{t+1} (initially zero).
    Tensor dstate; dstate.rows = two_d; dstate.cols = 1;
    dstate.data.assign(two_d, 0.0);

    // grad_input accumulator (filled during this same loop)
    Tensor grad_input(T, d_input_);
    grad_input.fill(0.0);

    for (int t = T - 1; t >= 0; --t) {
        // 1. Add C·grad_output[t] to dstate (this is the direct contribution to dL/dx_{t+1}).
        for (int i = 0; i < two_d; ++i) {
            double s = 0.0;
            for (int j = 0; j < d_output_; ++j) s += C[i][j] * grad_output[t][j];
            dstate.data[i] += s;
        }

        // 2. grad_C accumulation: grad_C[i][j] += x_{t+1}[i] * grad_output[t][j]
        for (int i = 0; i < two_d; ++i)
            for (int j = 0; j < d_output_; ++j)
                grad_C[i][j] += states_[t + 1].data[i] * grad_output[t][j];

        // 3. Compute z_t = M^{-T} dstate and dx_t = J^T · dstate (the carriers).
        //    z_t is needed for grad_B and grad_input (because dL/du_proj[t] = dt · z_t).
        //    dx_t is needed for the parameter grads through A (log_damping, freq, log_dt)
        //    and for the next iteration's dstate (since dx_t = dL/dx_t represents the
        //    full carrier at time t).
        Tensor z_t;
        z_t.rows = two_d; z_t.cols = 1;
        z_t.data.assign(two_d, 0.0);
        Tensor dx_t;
        dx_t.rows = two_d; dx_t.cols = 1;
        dx_t.data.assign(two_d, 0.0);
        if (type_ == LinOSSType::IMPLICIT_MIDPOINT) {
            // z_t = M^{-T} dstate
            solve_MTinv_y(damping, omega, dt, dstate, z_t);
            // dx_t = (I + hA)^T z_t = z_t + h (A^T z_t)
            double h_im = dt / 2.0;
            for (int k = 0; k < d_osc_; ++k) {
                double lam = damping[k];
                double om = omega[k];
                double z0 = z_t.data[2 * k];
                double z1 = z_t.data[2 * k + 1];
                double ATz0 = -lam * z0 - om * z1;
                double ATz1 = om * z0 - lam * z1;
                dx_t.data[2 * k]     = z0 + h_im * ATz0;
                dx_t.data[2 * k + 1] = z1 + h_im * ATz1;
            }
        } else {
            // HEUN: dx_t = (I + dt A^T + dt^2/2 (A^T)^2) dstate
            Tensor ATdstate;
            ATdstate.rows = two_d; ATdstate.cols = 1;
            ATdstate.data.assign(two_d, 0.0);
            compute_ATx(dstate, damping, omega, ATdstate);
            Tensor AT2dstate;
            AT2dstate.rows = two_d; AT2dstate.cols = 1;
            AT2dstate.data.assign(two_d, 0.0);
            compute_ATx(ATdstate, damping, omega, AT2dstate);
            for (int i = 0; i < two_d; ++i)
                dx_t.data[i] = dstate.data[i] + dt * ATdstate.data[i]
                             + 0.5 * dt * dt * AT2dstate.data[i];
        }

        // 4. grad_B[i][j] accumulation: grad_B[i][j] += dt * z_t[j] * input[t][i]
        //    (dL/d(u_proj[t]) = dt · z_t where z_t = M^{-T} dL/dx_{t+1})
        for (int i = 0; i < d_input_; ++i)
            for (int j = 0; j < two_d; ++j)
                grad_B[i][j] += dt * z_t.data[j] * last_input_[t][i];

        // 5. grad_input[t][i] = sum_j dt * z_t[j] * B[i][j]
        for (int i = 0; i < d_input_; ++i) {
            double s = 0.0;
            for (int j = 0; j < two_d; ++j) s += dt * z_t.data[j] * B[i][j];
            grad_input[t][i] = s;
        }

        // 5b. Cache z_t for use in A-matrix gradient computation (per timestep).
        if (type_ == LinOSSType::IMPLICIT_MIDPOINT) {
            if ((int)z_t_per_t_.size() <= t) z_t_per_t_.resize(t + 1);
            // z_t.data is the AlignedAllocator vector; copy into std::vector<double>.
            z_t_per_t_[t].assign(z_t.data.begin(), z_t.data.end());
        }

// 6. Compute the parameter grads that flow through the A-matrix (use dx_t from step 3).
        //    For IM (compute grad_log_damping, grad_freq, grad_log_dt):
        //    These chain through A → N (= I + hA) → rhs, and A → M (= I - hA) → M^{-1}.
        //    We use the clean derivation: z_t = M^{-T} dL/dx_{t+1} (cached per timestep).

                // IM A-matrix gradient deferred to after the BPTT loop (see below).

        // Replace dstate with dx_t (for next iteration's BPTT carry).
        for (int i = 0; i < two_d; ++i) dstate.data[i] = dx_t.data[i];
    }
    // IM A-matrix gradient accumulation (deferred so all z_t values are cached).
    if (type_ == LinOSSType::IMPLICIT_MIDPOINT) {
        double h_im = dt / 2.0;

        // grad_log_damping[k]: ∂A/∂log_damping = -lam · I_2 at block k.
// dN/d(log_damping) = h · ∂A/∂log_damping = -h · lam · I_2 (affects rhs).
// dM/d(log_damping) = h · lam · I_2 (affects M^{-1}).
// dx_{t+1}/d(log_damping) = M^{-1} · (d rhs/d(log_damping)) + (d M^{-1}/d(log_damping)) · rhs
//                       = M^{-1} · (-h lam x_t) + (-h lam M^{-2}) · rhs
//                       = -h lam (M^{-1} x_t + M^{-2} rhs) = -h lam (M^{-1} x_t + x_{t+1})
//                                          (since M^{-1} rhs = x_{t+1})
// dL/d(log_damping)[k] = sum_t dL/dx_{t+1}[k] · dx_{t+1}/d(log_damping)
//                     = -h lam sum_t [z_t · x_t + z_t · x_{t+1}]
//                               where z_t = M^{-T} dL/dx_{t+1}
//                                            = M^{-T} (M^{-1} rhs = x_{t+1})  ← NO: z_t · M^{-1} rhs = z_t · x_{t+1}
//                     = -h lam sum_t z_t · (x_t + x_{t+1})
for (int k = 0; k < d_osc_; ++k) {
    double lam = damping[k];
    double inner_prod_sum = 0.0;
    for (int t_bpt = 0; t_bpt <= T - 1; ++t_bpt) {
        double z0 = z_t_per_t_[t_bpt][2 * k];
        double z1 = z_t_per_t_[t_bpt][2 * k + 1];
        double xt0 = states_[t_bpt].data[2 * k];
        double xt1 = states_[t_bpt].data[2 * k + 1];
        double xp0 = states_[t_bpt + 1].data[2 * k];
        double xp1 = states_[t_bpt + 1].data[2 * k + 1];
        inner_prod_sum += z0 * (xt0 + xp0) + z1 * (xt1 + xp1);
    }
    grad_log_damping[k][0] += -h_im * lam * inner_prod_sum;
}

        // grad_freq[k]: ∂A/∂ω at block k = [[0, 1], [-1, 0]].
        // dN/dω = h · [[0,1],[-1,0]] (affects rhs).
        // dM/dω = -h · [[0,1],[-1,0]] (affects M^{-1}).
        // dx_{t+1}/dω = h · M^{-1} · Dω_A · (x_t + x_{t+1}).
        // dL/dω[k] = h · sum_t <dL/dx_{t+1}, M^{-1} · Dω_A · (x_t + x_{t+1})>
        //          = h · sum_t <M^{-T} · dL/dx_{t+1}, Dω_A · (x_t + x_{t+1})>
        //          = h · sum_t <z_t, Dω_A · (x_t + x_{t+1})>
        // (the M^{-1} moves to z_t via <a, M^{-1}·b> = <M^{-T}·a, b>).
        for (int k = 0; k < d_osc_; ++k) {
            double h_im_local = dt / 2.0;
            double inner_prod_sum = 0.0;
            for (int t_bpt = 0; t_bpt <= T - 1; ++t_bpt) {
                double z0 = z_t_per_t_[t_bpt][2 * k];
                double z1 = z_t_per_t_[t_bpt][2 * k + 1];
                double xt0 = states_[t_bpt].data[2 * k];
                double xt1 = states_[t_bpt].data[2 * k + 1];
                double xp0 = states_[t_bpt + 1].data[2 * k];
                double xp1 = states_[t_bpt + 1].data[2 * k + 1];
                // Dω_A · (x_t + x_{t+1}) = ((xt1 + xp1), -(xt0 + xp0))
                inner_prod_sum += z0 * (xt1 + xp1) - z1 * (xt0 + xp0);
            }
            grad_freq[k][0] += h_im_local * inner_prod_sum;
        }

        // grad_log_dt: dx_{t+1}/dt = M^{-1} [(1/2) A (x_t + x_{t+1}) + u_proj[t]]
        // dL/dt = sum_t z_t · [(1/2) A (x_t + x_{t+1}) + u_proj[t]]
        double dL_dt_step = 0.0;
        for (int t_bpt = 0; t_bpt <= T - 1; ++t_bpt) {
            for (int k = 0; k < d_osc_; ++k) {
                double lam = damping[k];
                double om = omega[k];
                double xt0 = states_[t_bpt].data[2 * k];
                double xt1 = states_[t_bpt].data[2 * k + 1];
                double xp0 = states_[t_bpt + 1].data[2 * k];
                double xp1 = states_[t_bpt + 1].data[2 * k + 1];
                double Axtp0 = -lam * (xt0 + xp0) + om * (xt1 + xp1);
                double Axtp1 = -om * (xt0 + xp0) - lam * (xt1 + xp1);
                double up0 = last_u_proj_[t_bpt][2 * k];
                double up1 = last_u_proj_[t_bpt][2 * k + 1];
                double z0 = z_t_per_t_[t_bpt][2 * k];
                double z1 = z_t_per_t_[t_bpt][2 * k + 1];
                dL_dt_step += z0 * (0.5 * Axtp0 + up0) + z1 * (0.5 * Axtp1 + up1);
            }
        }
        grad_log_dt[0][0] += dL_dt_step * s_dt;
    }
    return grad_input;
}

// ============================================================================
// LinOSSModel
// ============================================================================

LinOSSModel::LinOSSModel(int d_input, int d_output, int d_model, int num_layers, int d_osc,
                         LinOSSType type) {
    if (d_input <= 0 || d_output <= 0 || d_model <= 0 || num_layers <= 0 || d_osc <= 0)
        throw std::invalid_argument("LinOSSModel: invalid dims");

    proj_in_.push_back(std::make_unique<Dense>(d_input, d_model));
    cells_.push_back(std::make_unique<LinOSSCell>(d_model, d_model, d_osc, type));
    for (int i = 1; i < num_layers; ++i) {
        proj_out_.push_back(std::make_unique<Dense>(d_model, d_model));
        cells_.push_back(std::make_unique<LinOSSCell>(d_model, d_model, d_osc, type));
    }
    classifier_ = std::make_unique<Dense>(d_model, d_output);
}

Tensor LinOSSModel::forward(const Tensor& input) {
    int T = (int)input.rows;
    last_input_ = input.clone();

    Tensor proj_x = proj_in_[0]->forward(input);
    last_embedded_.clear();
    last_embedded_.push_back(proj_x.clone());

    Tensor x = cells_[0]->forward(proj_x);
    last_embedded_.push_back(x.clone());

    for (size_t i = 1; i < cells_.size(); ++i) {
        Tensor proj_y = proj_out_[i - 1]->forward(x);
        last_embedded_.push_back(proj_y.clone());
        x = cells_[i]->forward(proj_y);
        last_embedded_.push_back(x.clone());
    }

    Tensor last_step(1, x.cols);
    for (size_t j = 0; j < x.cols; ++j) last_step[0][j] = x[T - 1][j];
    Tensor out = classifier_->forward(last_step);
    return out;
}

Tensor LinOSSModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    int T = (int)last_input_.rows;
    int d_model = cells_[0]->d_output_;

    Tensor grad_x(T, d_model);
    grad_x.fill(0.0);
    for (int j = 0; j < d_model; ++j) grad_x[T - 1][j] = grad_output[0][j];

    // Classifier backward (it has its own parameters; its grad_input is the grad through classifier).
    Tensor last_step(1, d_model);
    for (int j = 0; j < d_model; ++j)
        last_step[0][j] = last_embedded_[2 * cells_.size() - 1][T - 1][j];
    Tensor class_grad_in = classifier_->backward(grad_output, 0.0);
    // grad_x[T-1] should equal class_grad_in (after classifier). Let's add it for safety.
    for (int j = 0; j < d_model; ++j) grad_x[T - 1][j] = class_grad_in[0][j];

    for (int li = (int)cells_.size() - 1; li >= 0; --li) {
        Tensor cell_grad_in = cells_[li]->backward(grad_x, 0.0);
        if (li > 0) {
            Tensor proj_grad_in = proj_out_[li - 1]->backward(cell_grad_in, 0.0);
            grad_x = proj_grad_in;
        } else {
            proj_in_[0]->backward(cell_grad_in, 0.0);
        }
    }
    return Tensor();
}

void LinOSSModel::update_weights(double lr) {
    for (auto& p : proj_in_) p->update_weights(lr);
    for (auto& c : cells_) c->update_weights(lr);
    for (auto& p : proj_out_) p->update_weights(lr);
    classifier_->update_weights(lr);
}

void LinOSSModel::zero_grad() {
    for (auto& p : proj_in_) p->zero_grad();
    for (auto& c : cells_) c->zero_grad();
    for (auto& p : proj_out_) p->zero_grad();
    classifier_->zero_grad();
}

Tensor LinOSSModel::get_weights() const {
    return cells_[0]->B.clone();
}

Tensor LinOSSModel::get_gradients() const {
    return cells_[0]->grad_B.clone();
}

std::vector<Tensor*> LinOSSModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& pp : proj_in_) {
        auto sub = pp->parameters();
        p.insert(p.end(), sub.begin(), sub.end());
    }
    for (auto& c : cells_) {
        auto sub = c->parameters();
        p.insert(p.end(), sub.begin(), sub.end());
    }
    for (auto& pp : proj_out_) {
        auto sub = pp->parameters();
        p.insert(p.end(), sub.begin(), sub.end());
    }
    auto sub = classifier_->parameters();
    p.insert(p.end(), sub.begin(), sub.end());
    return p;
}

std::vector<Tensor*> LinOSSModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& pp : proj_in_) {
        auto sub = pp->gradients();
        g.insert(g.end(), sub.begin(), sub.end());
    }
    for (auto& c : cells_) {
        auto sub = c->gradients();
        g.insert(g.end(), sub.begin(), sub.end());
    }
    for (auto& pp : proj_out_) {
        auto sub = pp->gradients();
        g.insert(g.end(), sub.begin(), sub.end());
    }
    auto sub = classifier_->gradients();
    g.insert(g.end(), sub.begin(), sub.end());
    return g;
}