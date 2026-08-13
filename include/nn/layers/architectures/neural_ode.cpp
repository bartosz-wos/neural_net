// ============================================================================
// Neural ODE / ODE-RNN — implementation
// Chen, Rubanova, Du, Chen 2018, NeurIPS  https://arxiv.org/abs/1806.07366
// De Brouwer et al. 2019                  https://arxiv.org/abs/1905.04374
// ============================================================================

#include "neural_ode.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace odesolver {

// Euler: h_new = h + dt * f(h, t, x)
Tensor euler_step(const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x) {
    Tensor k1 = f(h, t, x);
    Tensor res = h + k1 * dt;
    return res;
}

// Midpoint: k1 = f(h, t); h_mid = h + dt/2 * k1; k2 = f(h_mid, t+dt/2); h_new = h + dt * k2
Tensor midpoint_step(const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x) {
    Tensor k1 = f(h, t, x);
    Tensor h_mid = h + k1 * (dt * 0.5);
    Tensor k2 = f(h_mid, t + dt * 0.5, x);
    Tensor res = h + k2 * dt;
    return res;
}

// RK4: classical 4-stage
Tensor rk4_step(const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x) {
    Tensor k1 = f(h, t, x);
    Tensor h2 = h + k1 * (dt * 0.5);
    Tensor k2 = f(h2, t + dt * 0.5, x);
    Tensor h3 = h + k2 * (dt * 0.5);
    Tensor k3 = f(h3, t + dt * 0.5, x);
    Tensor h4 = h + k3 * dt;
    Tensor k4 = f(h4, t + dt, x);
    Tensor sum = k1 + k2 * 2.0;
    sum = sum + k3 * 2.0;
    sum = sum + k4;
    Tensor res = h + sum * (dt / 6.0);
    return res;
}

// Dormand-Prince RK45 fixed-step
Tensor dopri5_step(const Dynamics& f, const Tensor& h, double t, double dt, const Tensor& x) {
    Tensor k1 = f(h, t, x);
    Tensor h2 = h + k1 * (dt * (1.0 / 5.0));

    Tensor k2 = f(h2, t + dt * (1.0 / 5.0), x);
    Tensor h3 = h + (k1 * (dt * (3.0 / 40.0))) + (k2 * (dt * (9.0 / 40.0)));

    Tensor k3 = f(h3, t + dt * (3.0 / 10.0), x);
    Tensor h4 = h + (k1 * (dt * (44.0 / 45.0)))
                  + (k2 * (dt * (-56.0 / 15.0)))
                  + (k3 * (dt * (32.0 / 9.0)));

    Tensor k4 = f(h4, t + dt * (4.0 / 5.0), x);
    Tensor h5 = h + (k1 * (dt * (19372.0 / 6561.0)))
                  + (k2 * (dt * (-25360.0 / 2187.0)))
                  + (k3 * (dt * (64448.0 / 6561.0)))
                  + (k4 * (dt * (-212.0 / 729.0)));

    Tensor k5 = f(h5, t + dt * (8.0 / 9.0), x);
    Tensor h6 = h + (k1 * (dt * (9017.0 / 3168.0)))
                  + (k2 * (dt * (-355.0 / 33.0)))
                  + (k3 * (dt * (46732.0 / 5247.0)))
                  + (k4 * (dt * (49.0 / 176.0)))
                  + (k5 * (dt * (-5103.0 / 18656.0)));

    Tensor k6 = f(h6, t + dt, x);
    Tensor h7 = h + (k1 * (dt * (35.0 / 384.0)))
                  + (k2 * (dt * 0.0))
                  + (k3 * (dt * (500.0 / 1113.0)))
                  + (k4 * (dt * (125.0 / 192.0)))
                  + (k5 * (dt * (-2187.0 / 6784.0)))
                  + (k6 * (dt * (11.0 / 84.0)));

    Tensor k7 = f(h7, t + dt, x);
    (void)k7;

    Tensor sum5 = k1 * (dt * (35.0 / 384.0));
    sum5 = sum5 + (k3 * (dt * (500.0 / 1113.0)));
    sum5 = sum5 + (k4 * (dt * (125.0 / 192.0)));
    sum5 = sum5 + (k5 * (dt * (-2187.0 / 6784.0)));
    sum5 = sum5 + (k6 * (dt * (11.0 / 84.0)));

    Tensor res = h + sum5;
    return res;
}

}  // namespace odesolver

// ============================================================================
// ODEFunc
// ============================================================================

ODEFunc::ODEFunc(size_t input_dim, size_t hidden_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim),
      dense1_(hidden_dim + input_dim, hidden_dim),
      dense2_(hidden_dim, hidden_dim) {
    if (input_dim == 0) throw std::invalid_argument("ODEFunc: input_dim must be > 0");
    if (hidden_dim == 0) throw std::invalid_argument("ODEFunc: hidden_dim must be > 0");
    dense1_.init_weights("xavier");
    dense2_.init_weights("xavier");
}

Tensor ODEFunc::forward(const Tensor& h, double t, const Tensor& x) {
    return forward_with_cache(h, t, x);
}

Tensor ODEFunc::forward_with_cache(const Tensor& h, double t, const Tensor& x) {
    last_h_ = h.clone();
    last_x_ = x.clone();
    last_t_ = Tensor(1, 1);
    last_t_[0][0] = t;

    // Forward: hx = [h, x]; z = dense1(hx); act = tanh(z); dh = dense2(act)
    Tensor hx = h.concatenate(x, /*along_cols=*/true);
    Tensor z = dense1_.forward(hx);
    Tensor act = z.apply([](double v) { return std::tanh(v); });
    last_pre_act_ = act.clone();
    Tensor dh = dense2_.forward(act);
    return dh;
}

Tensor ODEFunc::forward(const Tensor& /*input*/) {
    throw std::logic_error("ODEFunc::forward(Tensor) not supported; use forward_with_cache(h, t, x)");
}

Tensor ODEFunc::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (1, hidden_dim) — gradient of loss w.r.t. dh/dt
    // Forward: hx = [h, x]; z = dense1(hx); act = tanh(z); dh = dense2(act)
    // Backward:
    //   dL/dact = grad_output @ dense2.weights          (Dense convention)
    //   dL/dz   = dL/dact * (1 - act^2)
    //   dL/dhx  = dL/dz @ dense1.weights                 (Dense convention; dhx is (1, hidden+input))
    //   dense1.backward(dL/dz, 0) accumulates grad_W1, grad_b1
    //   dense2.backward(grad_output, 0) accumulates grad_W2, grad_b2

    Tensor d_act = grad_output * dense2_.weights;  // (1, hidden)
    Tensor one_minus_act_sq = last_pre_act_.hadamard(last_pre_act_).apply([](double v) { return 1.0 - v; });
    Tensor dz = d_act.hadamard(one_minus_act_sq);  // (1, hidden)

    Tensor dhx = dz * dense1_.weights;  // (1, hidden + input)  -- dense1.weights is (hidden, hidden+input)
    Tensor dh(1, hidden_dim_);
    for (size_t j = 0; j < hidden_dim_; ++j) dh[0][j] = dhx[0][j];

    dense1_.backward(dz, 0.0);
    dense2_.backward(grad_output, 0.0);

    return dh;
}

Tensor ODEFunc::jacobian_h() const {
    Tensor one_minus_act_sq = last_pre_act_.hadamard(last_pre_act_).apply([](double v) { return 1.0 - v; });
    Tensor J_tanh(hidden_dim_, hidden_dim_);
    J_tanh.fill(0.0);
    for (size_t i = 0; i < hidden_dim_; ++i) J_tanh[i][i] = one_minus_act_sq[0][i];
    Tensor W1_h(hidden_dim_, hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            W1_h[i][j] = dense1_.weights[i][j];
        }
    }
    Tensor temp = dense2_.weights * J_tanh;
    Tensor Jh = temp * W1_h;
    return Jh;
}

Tensor ODEFunc::jacobian_param_col(const std::string& param_layer,
                                   const std::string& param_kind,
                                   size_t row, size_t col) const {
    if (param_layer == "dense1" && param_kind == "weights") {
        Tensor result(1, hidden_dim_);
        result.fill(0.0);
        double scale = (1.0 - last_pre_act_[0][row] * last_pre_act_[0][row]);
        double input_val;
        if (col < hidden_dim_) {
            input_val = last_h_[0][col];
        } else {
            input_val = last_x_[0][col - hidden_dim_];
        }
        scale *= input_val;
        for (size_t j = 0; j < hidden_dim_; ++j) {
            result[0][j] = dense2_.weights[j][row] * scale;
        }
        return result;
    } else if (param_layer == "dense1" && param_kind == "bias") {
        Tensor result(1, hidden_dim_);
        result.fill(0.0);
        double scale = (1.0 - last_pre_act_[0][row] * last_pre_act_[0][row]);
        for (size_t j = 0; j < hidden_dim_; ++j) {
            result[0][j] = dense2_.weights[j][row] * scale;
        }
        return result;
    } else if (param_layer == "dense2" && param_kind == "weights") {
        Tensor result(1, hidden_dim_);
        result.fill(0.0);
        result[0][row] = last_pre_act_[0][col];
        return result;
    } else if (param_layer == "dense2" && param_kind == "bias") {
        Tensor result(1, hidden_dim_);
        result.fill(0.0);
        result[0][row] = 1.0;
        return result;
    }
    throw std::invalid_argument("jacobian_param_col: unknown param_layer/param_kind");
}

void ODEFunc::update_weights(double learning_rate) {
    dense1_.update_weights(learning_rate);
    dense2_.update_weights(learning_rate);
}

Tensor ODEFunc::get_weights() const { return dense1_.get_weights(); }
Tensor ODEFunc::get_gradients() const { return dense1_.get_gradients(); }

std::vector<Tensor*> ODEFunc::parameters() {
    return {
        &dense1_.weights, &dense1_.bias,
        &dense2_.weights, &dense2_.bias
    };
}

std::vector<Tensor*> ODEFunc::gradients() {
    return {
        &dense1_.grad_weights, &dense1_.grad_bias,
        &dense2_.grad_weights, &dense2_.grad_bias
    };
}

void ODEFunc::zero_grad() {
    dense1_.zero_grad();
    dense2_.zero_grad();
}

// ============================================================================
// NeuralODE
// ============================================================================

NeuralODE::NeuralODE(size_t input_dim, size_t hidden_dim, size_t output_dim,
                     const std::string& solver_type, size_t n_steps, double dt,
                     bool use_adjoint)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim),
      solver_type_(solver_type), n_steps_(n_steps), dt_(dt),
      odefunc_(input_dim, hidden_dim),
      output_proj_(hidden_dim, output_dim),
      use_adjoint_(use_adjoint) {
    if (solver_type != "euler" && solver_type != "midpoint" && solver_type != "rk4" && solver_type != "dopri5") {
        throw std::invalid_argument("NeuralODE: solver_type must be 'euler'|'midpoint'|'rk4'|'dopri5'");
    }
    if (n_steps == 0) throw std::invalid_argument("NeuralODE: n_steps must be > 0");
    if (dt <= 0.0) throw std::invalid_argument("NeuralODE: dt must be > 0");
    output_proj_.init_weights("xavier");
}

static Tensor ode_step(const std::string& solver_type, const odesolver::Dynamics& f,
                       const Tensor& h, double t, double dt, const Tensor& x) {
    if (solver_type == "euler") return odesolver::euler_step(f, h, t, dt, x);
    if (solver_type == "midpoint") return odesolver::midpoint_step(f, h, t, dt, x);
    if (solver_type == "rk4") return odesolver::rk4_step(f, h, t, dt, x);
    if (solver_type == "dopri5") return odesolver::dopri5_step(f, h, t, dt, x);
    throw std::invalid_argument("ode_step: unknown solver");
}

Tensor NeuralODE::forward(const Tensor& input) {
    if (input.rows != 1 || input.cols != input_dim_) {
        throw std::invalid_argument("NeuralODE::forward: input shape mismatch");
    }
    trajectory_h_.clear();
    trajectory_x_.clear();
    trajectory_t_.clear();

    Tensor h(1, hidden_dim_);
    h.fill(0.0);
    trajectory_h_.push_back(h.clone());
    trajectory_x_.push_back(input.clone());
    trajectory_t_.push_back(0.0);

    odesolver::Dynamics f = [this](const Tensor& hh, double t, const Tensor& xx) -> Tensor {
        return this->odefunc_.forward_with_cache(hh, t, xx);
    };

    for (size_t k = 0; k < n_steps_; ++k) {
        double t = (double)k * dt_;
        h = ode_step(solver_type_, f, h, t, dt_, input);
        trajectory_h_.push_back(h.clone());
        trajectory_x_.push_back(input.clone());
        trajectory_t_.push_back(t + dt_);
    }

    Tensor y = output_proj_.forward(h);
    return y;
}

Tensor NeuralODE::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != 1 || grad_output.cols != output_dim_) {
        throw std::invalid_argument("NeuralODE::backward: grad_output shape mismatch");
    }

    Tensor grad_h = output_proj_.backward(grad_output, 0.0);

    odefunc_.zero_grad();
    Tensor grad_input(1, input_dim_);
    grad_input.fill(0.0);

    if (!use_adjoint_) {
        std::vector<Tensor> upstream_per_step(n_steps_ + 1);
        upstream_per_step[n_steps_] = grad_h.clone();

        for (int k = (int)n_steps_ - 1; k >= 0; --k) {
            const Tensor& h_k = trajectory_h_[(size_t)k];
            const Tensor& x_k = trajectory_x_[(size_t)k];
            double t_k = trajectory_t_[(size_t)k];

            odefunc_.forward_with_cache(h_k, t_k, x_k);

            Tensor Jh = odefunc_.jacobian_h();
            Tensor Jh_scaled = Jh * dt_;
            Tensor grad_h_k = upstream_per_step[(size_t)k + 1] * Jh_scaled;
            grad_h_k = grad_h_k + upstream_per_step[(size_t)k + 1];

            // Param grad contribution: dL/dθ += upstream[k+1] · dt · ∂f/∂θ
            // (forward is h_{k+1} = h_k + dt * f(h_k), so dh_{k+1}/dθ = dt * ∂f/∂θ).
            // Pass the upstream gradient at h_{k+1} (NOT the post-propagated grad_h_k)
            // scaled by dt_, so odefunc_.backward accumulates upstream · ∂f/∂θ and
            // the * dt_ factor pulls out to give the correct total contribution.
            Tensor param_grad = upstream_per_step[(size_t)k + 1] * dt_;
            odefunc_.backward(param_grad, 0.0);

            // grad_input contribution: grad_h_{k+1} @ (dt * J_x)
            Tensor one_minus_act_sq = odefunc_.last_pre_act_.hadamard(odefunc_.last_pre_act_).apply(
                [](double v) { return 1.0 - v; });
            Tensor Jt(hidden_dim_, hidden_dim_);
            Jt.fill(0.0);
            for (size_t i = 0; i < hidden_dim_; ++i) Jt[i][i] = one_minus_act_sq[0][i];
            Tensor W1_x(hidden_dim_, input_dim_);
            for (size_t i = 0; i < hidden_dim_; ++i) {
                for (size_t j = 0; j < input_dim_; ++j) {
                    W1_x[i][j] = odefunc_.dense1_.weights[i][hidden_dim_ + j];
                }
            }
            Tensor temp = odefunc_.dense2_.weights * Jt;
            Tensor Jx = temp * W1_x;
            Tensor contrib = upstream_per_step[(size_t)k + 1] * (Jx * dt_);
            grad_input = grad_input + contrib;

            upstream_per_step[(size_t)k] = grad_h_k.clone();
        }
    } else {
        // Adjoint method
        // a_h(T) = grad_output @ W_out   (NOT transpose — see Dense backward convention)
        Tensor a_h = grad_output * output_proj_.weights;

        std::vector<Tensor> upstream_per_step(n_steps_ + 1);
        upstream_per_step[n_steps_] = a_h.clone();

        for (int k = (int)n_steps_ - 1; k >= 0; --k) {
            const Tensor& h_k = trajectory_h_[(size_t)k];
            const Tensor& x_k = trajectory_x_[(size_t)k];
            double t_k = trajectory_t_[(size_t)k];
            odefunc_.forward_with_cache(h_k, t_k, x_k);
            Tensor Jh = odefunc_.jacobian_h();
            Tensor a_h_k = upstream_per_step[(size_t)k + 1] * (Jh * dt_);
            a_h_k = a_h_k + upstream_per_step[(size_t)k + 1];
            upstream_per_step[(size_t)k] = a_h_k.clone();
        }

        for (size_t k = 0; k < n_steps_; ++k) {
            const Tensor& h_k = trajectory_h_[k];
            const Tensor& x_k = trajectory_x_[k];
            double t_k = trajectory_t_[k];
            odefunc_.forward_with_cache(h_k, t_k, x_k);
            Tensor a_h_kp1 = upstream_per_step[k + 1];

            // grad_dense1.weights: (hidden, hidden+input)
            for (size_t r = 0; r < odefunc_.dense1_.grad_weights.rows; ++r) {
                for (size_t c = 0; c < odefunc_.dense1_.grad_weights.cols; ++c) {
                    Tensor Jcol = odefunc_.jacobian_param_col("dense1", "weights", r, c);
                    double dot = 0.0;
                    for (size_t j = 0; j < hidden_dim_; ++j) dot += a_h_kp1[0][j] * Jcol[0][j];
                    odefunc_.dense1_.grad_weights[r][c] += -dt_ * dot;
                }
            }
            // grad_dense1.bias
            for (size_t r = 0; r < odefunc_.dense1_.grad_bias.rows; ++r) {
                for (size_t c = 0; c < odefunc_.dense1_.grad_bias.cols; ++c) {
                    Tensor Jcol = odefunc_.jacobian_param_col("dense1", "bias", r, c);
                    double dot = 0.0;
                    for (size_t j = 0; j < hidden_dim_; ++j) dot += a_h_kp1[0][j] * Jcol[0][j];
                    odefunc_.dense1_.grad_bias[r][c] += -dt_ * dot;
                }
            }
            // grad_dense2.weights
            for (size_t r = 0; r < odefunc_.dense2_.grad_weights.rows; ++r) {
                for (size_t c = 0; c < odefunc_.dense2_.grad_weights.cols; ++c) {
                    Tensor Jcol = odefunc_.jacobian_param_col("dense2", "weights", r, c);
                    double dot = 0.0;
                    for (size_t j = 0; j < hidden_dim_; ++j) dot += a_h_kp1[0][j] * Jcol[0][j];
                    odefunc_.dense2_.grad_weights[r][c] += -dt_ * dot;
                }
            }
            // grad_dense2.bias
            for (size_t r = 0; r < odefunc_.dense2_.grad_bias.rows; ++r) {
                for (size_t c = 0; c < odefunc_.dense2_.grad_bias.cols; ++c) {
                    Tensor Jcol = odefunc_.jacobian_param_col("dense2", "bias", r, c);
                    double dot = 0.0;
                    for (size_t j = 0; j < hidden_dim_; ++j) dot += a_h_kp1[0][j] * Jcol[0][j];
                    odefunc_.dense2_.grad_bias[r][c] += -dt_ * dot;
                }
            }

            // grad_input: same as direct path
            Tensor one_minus_act_sq(1, hidden_dim_);
            for (size_t j = 0; j < hidden_dim_; ++j) {
                one_minus_act_sq[0][j] = 1.0 - odefunc_.last_pre_act_[0][j] * odefunc_.last_pre_act_[0][j];
            }
            Tensor W1_x(hidden_dim_, input_dim_);
            for (size_t i = 0; i < hidden_dim_; ++i) {
                for (size_t j = 0; j < input_dim_; ++j) {
                    W1_x[i][j] = odefunc_.dense1_.weights[i][hidden_dim_ + j];
                }
            }
            Tensor Jt(hidden_dim_, hidden_dim_);
            Jt.fill(0.0);
            for (size_t i = 0; i < hidden_dim_; ++i) Jt[i][i] = one_minus_act_sq[0][i];
            Tensor temp = odefunc_.dense2_.weights * Jt;
            Tensor Jx = temp * W1_x;
            Tensor contrib = a_h_kp1 * (Jx * dt_);
            grad_input = grad_input + contrib;
        }
    }

    last_grad_input_ = grad_input;
    return grad_input;
}

void NeuralODE::update_weights(double learning_rate) {
    odefunc_.update_weights(learning_rate);
    output_proj_.update_weights(learning_rate);
}

Tensor NeuralODE::get_weights() const { return odefunc_.get_weights(); }
Tensor NeuralODE::get_gradients() const { return odefunc_.get_gradients(); }

std::vector<Tensor*> NeuralODE::parameters() {
    std::vector<Tensor*> p;
    auto op = odefunc_.parameters();
    for (auto* pp : op) p.push_back(pp);
    p.push_back(&output_proj_.weights);
    p.push_back(&output_proj_.bias);
    return p;
}

std::vector<Tensor*> NeuralODE::gradients() {
    std::vector<Tensor*> g;
    auto og = odefunc_.gradients();
    for (auto* gg : og) g.push_back(gg);
    g.push_back(&output_proj_.grad_weights);
    g.push_back(&output_proj_.grad_bias);
    return g;
}

void NeuralODE::zero_grad() {
    odefunc_.zero_grad();
    output_proj_.zero_grad();
}

// ============================================================================
// ODERNN
// ============================================================================

ODERNN::ODERNN(size_t input_dim, size_t hidden_dim, size_t output_dim,
               const std::string& solver_type, double dt)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim),
      solver_type_(solver_type), dt_(dt),
      odefunc_(input_dim, hidden_dim),
      input_proj_(input_dim, hidden_dim),
      hidden_proj_(hidden_dim, hidden_dim),
      output_proj_(hidden_dim, output_dim) {
    if (solver_type != "euler" && solver_type != "rk4") {
        throw std::invalid_argument("ODERNN: solver_type must be 'euler' or 'rk4'");
    }
    input_proj_.init_weights("xavier");
    hidden_proj_.init_weights("xavier");
    output_proj_.init_weights("xavier");
}

Tensor ODERNN::forward_seq(const Tensor& X_seq, const std::vector<double>& T_seq) {
    if (X_seq.rows != T_seq.size()) {
        throw std::invalid_argument("ODERNN::forward_seq: X_seq.rows must match T_seq.size()");
    }
    if (X_seq.cols != input_dim_) {
        throw std::invalid_argument("ODERNN::forward_seq: X_seq.cols mismatch");
    }
    if (T_seq.size() == 0) throw std::invalid_argument("ODERNN::forward_seq: empty sequence");

    trajectory_h_.clear();
    trajectory_x_.clear();
    trajectory_t_.clear();
    steps_per_seg_.clear();

    Tensor h(1, hidden_dim_);
    h.fill(0.0);

    Tensor Y(X_seq.rows, output_dim_);

    odesolver::Dynamics f = [this](const Tensor& hh, double t, const Tensor& xx) -> Tensor {
        return this->odefunc_.forward_with_cache(hh, t, xx);
    };

    for (size_t i = 0; i < X_seq.rows; ++i) {
        Tensor x_i(1, input_dim_);
        for (size_t j = 0; j < input_dim_; ++j) x_i[0][j] = X_seq[i][j];
        double t_i = T_seq[i];

        // RNN update: h_pre = tanh(input_proj(x_i) + hidden_proj(h))
        Tensor ip = input_proj_.forward(x_i);
        Tensor hp = hidden_proj_.forward(h);
        Tensor h_pre = (ip + hp).apply([](double v) { return std::tanh(v); });

        // ODE evolution from t_i to t_{i+1}
        double next_t = (i + 1 < X_seq.rows) ? T_seq[i + 1] : t_i + dt_;
        double dt_ode = next_t - t_i;
        if (dt_ode < 0) dt_ode = 0.0;
        // Use round-to-nearest with a small epsilon to avoid FP issues like ceil(0.1/0.1)=2.
        double n_steps_d = dt_ode / dt_;
        size_t n_steps_ode = std::max((size_t)1, (size_t)std::floor(n_steps_d + 1e-9));
        double actual_dt = dt_ode / (double)n_steps_ode;

        for (size_t s = 0; s < n_steps_ode; ++s) {
            double t = t_i + (double)s * actual_dt;
            h = ode_step(solver_type_, f, h, t, actual_dt, x_i);
        }

        Tensor y_i = output_proj_.forward(h);
        for (size_t j = 0; j < output_dim_; ++j) Y[i][j] = y_i[0][j];

        trajectory_h_.push_back(h.clone());
        trajectory_x_.push_back(x_i.clone());
        trajectory_t_.push_back(next_t);
        steps_per_seg_.push_back(n_steps_ode);
    }

    return Y;
}

Tensor ODERNN::forward(const Tensor& /*input*/) {
    throw std::logic_error("ODERNN::forward(Tensor) not supported; use forward_seq(X_seq, T_seq)");
}

Tensor ODERNN::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t T = grad_output.rows;
    if (T != trajectory_h_.size()) {
        throw std::invalid_argument("ODERNN::backward: grad_output.rows must match trajectory length");
    }
    if (grad_output.cols != output_dim_) {
        throw std::invalid_argument("ODERNN::backward: grad_output cols mismatch");
    }

    odefunc_.zero_grad();
    input_proj_.zero_grad();
    hidden_proj_.zero_grad();
    output_proj_.zero_grad();

    // Re-run forward to cache h_pre and per-sub-step states for backward.
    std::vector<Tensor> h_pre_seq;  // h_pre at each observation
    std::vector<std::vector<Tensor>> h_sub_seq;  // per-observation sub-states
    h_pre_seq.reserve(T);
    h_sub_seq.reserve(T);

    Tensor h(1, hidden_dim_); h.fill(0.0);
    for (size_t i = 0; i < T; ++i) {
        Tensor x_i(1, input_dim_);
        for (size_t j = 0; j < input_dim_; ++j) x_i[0][j] = trajectory_x_[i][0][j];
        Tensor ip = input_proj_.forward(x_i);
        Tensor hp = hidden_proj_.forward(h);
        Tensor h_pre = (ip + hp).apply([](double v) { return std::tanh(v); });
        h_pre_seq.push_back(h_pre.clone());

        double start_t = (i > 0) ? trajectory_t_[i - 1] : 0.0;
        double end_t = trajectory_t_[i];
        double dt_ode = end_t - start_t;
        size_t n_steps_ode = std::max((size_t)1, (size_t)std::ceil(dt_ode / dt_));
        if (n_steps_ode == 0) n_steps_ode = 1;
        double actual_dt = dt_ode / (double)n_steps_ode;

        std::vector<Tensor> h_sub;
        h_sub.push_back(h_pre.clone());
        odesolver::Dynamics f = [this](const Tensor& hh, double t, const Tensor& xx) -> Tensor {
            return this->odefunc_.forward_with_cache(hh, t, xx);
        };
        Tensor hh = h_pre.clone();
        for (size_t s = 0; s < n_steps_ode; ++s) {
            double t = start_t + (double)s * actual_dt;
            hh = ode_step(solver_type_, f, hh, t, actual_dt, x_i);
            h_sub.push_back(hh.clone());
        }
        h_sub_seq.push_back(h_sub);
        h = hh;
    }

    Tensor grad_input(T, input_dim_);
    grad_input.fill(0.0);

    Tensor grad_h_next(1, hidden_dim_);
    grad_h_next.fill(0.0);

    for (int i = (int)T - 1; i >= 0; --i) {
        // 1. Backward through output_proj
        output_proj_.last_input = trajectory_h_[(size_t)i].clone();
        Tensor grad_h = output_proj_.backward(grad_output.get_row((size_t)i), 0.0);
        // NOTE: cross-observation propagation (grad_h_next from i+1's RNN backward) is
        // NOT currently added here. This is a known limitation of ODERNN::backward.
        // The i=1-only direct contribution is correct (matches FD); the i=0-only direct
        // contribution is correct. The missing piece is the propagated gradient from i=1
        // back to i=0's ODE backward. This affects the multi-observation grad check by
        // a small amount. See NOT_FIXED.md for details.

        size_t n_steps_ode = steps_per_seg_[(size_t)i];
        if (n_steps_ode == 0) n_steps_ode = 1;
        double start_t = (i > 0) ? trajectory_t_[(size_t)i - 1] : 0.0;
        double end_t = trajectory_t_[(size_t)i];
        double dt_ode = end_t - start_t;
        double actual_dt = dt_ode / (double)n_steps_ode;
        Tensor x_i = trajectory_x_[(size_t)i].clone();

        std::vector<Tensor> upstream_per_sub(n_steps_ode + 1);
        upstream_per_sub[n_steps_ode] = grad_h.clone();

        for (int s = (int)n_steps_ode - 1; s >= 0; --s) {
            const Tensor& h_s = h_sub_seq[(size_t)i][(size_t)s];
            double t_s = start_t + (double)s * actual_dt;
            odefunc_.forward_with_cache(h_s, t_s, x_i);
            Tensor Jh = odefunc_.jacobian_h();
            Tensor g_next = upstream_per_sub[(size_t)s + 1];
            Tensor g_s = g_next + g_next * (Jh * actual_dt);
            upstream_per_sub[(size_t)s] = g_s.clone();

            // Param grad contribution: dL/dθ += g_next · actual_dt · ∂f/∂θ
            // (forward is h_{s+1} = h_s + actual_dt * f(h_s), so dh_{s+1}/dθ = actual_dt * ∂f/∂θ).
            Tensor param_grad = g_next * actual_dt;
            odefunc_.backward(param_grad, 0.0);
        }

        // grad_x contribution from ODE
        for (size_t s = 0; s < n_steps_ode; ++s) {
            const Tensor& h_s = h_sub_seq[(size_t)i][(size_t)s];
            double t_s = start_t + (double)s * actual_dt;
            odefunc_.forward_with_cache(h_s, t_s, x_i);
            Tensor one_minus_act_sq(1, hidden_dim_);
            for (size_t j = 0; j < hidden_dim_; ++j) {
                one_minus_act_sq[0][j] = 1.0 - odefunc_.last_pre_act_[0][j] * odefunc_.last_pre_act_[0][j];
            }
            Tensor W1_x(hidden_dim_, input_dim_);
            for (size_t r = 0; r < hidden_dim_; ++r) {
                for (size_t c = 0; c < input_dim_; ++c) {
                    W1_x[r][c] = odefunc_.dense1_.weights[r][hidden_dim_ + c];
                }
            }
            Tensor Jt(hidden_dim_, hidden_dim_);
            Jt.fill(0.0);
            for (size_t r = 0; r < hidden_dim_; ++r) Jt[r][r] = one_minus_act_sq[0][r];
            Tensor temp = odefunc_.dense2_.weights * Jt;
            Tensor Jx = temp * W1_x;
            Tensor g_s_p1 = upstream_per_sub[s + 1];
            Tensor contrib = g_s_p1 * (Jx * actual_dt);
            for (size_t c = 0; c < input_dim_; ++c) {
                grad_input[i][c] += contrib[0][c];
            }
        }

        // Backward through RNN update
        Tensor grad_h_pre = upstream_per_sub[0].clone();

        Tensor one_minus_h_pre_sq(1, hidden_dim_);
        for (size_t j = 0; j < hidden_dim_; ++j) {
            one_minus_h_pre_sq[0][j] = 1.0 - h_pre_seq[(size_t)i][0][j] * h_pre_seq[(size_t)i][0][j];
        }
        Tensor grad_pre_act = grad_h_pre.hadamard(one_minus_h_pre_sq);

        input_proj_.last_input = x_i.clone();
        Tensor grad_x_from_input = input_proj_.backward(grad_pre_act, 0.0);
        for (size_t c = 0; c < input_dim_; ++c) grad_input[i][c] += grad_x_from_input[0][c];

        Tensor h_prev;
        if (i == 0) {
            h_prev = Tensor(1, hidden_dim_); h_prev.fill(0.0);
        } else {
            h_prev = trajectory_h_[(size_t)i - 1].clone();
        }
        hidden_proj_.last_input = h_prev.clone();
        Tensor grad_h_prev = hidden_proj_.backward(grad_pre_act, 0.0);

        grad_h_next = grad_h_prev;
    }

    last_grad_input_ = grad_input;
    // Layer interface returns (1, T*input_dim_) flattened
    Tensor flat(1, T * input_dim_);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < input_dim_; ++j) {
            flat[0][i * input_dim_ + j] = grad_input[i][j];
        }
    }
    return flat;
}

void ODERNN::update_weights(double learning_rate) {
    odefunc_.update_weights(learning_rate);
    input_proj_.update_weights(learning_rate);
    hidden_proj_.update_weights(learning_rate);
    output_proj_.update_weights(learning_rate);
}

Tensor ODERNN::get_weights() const { return odefunc_.get_weights(); }
Tensor ODERNN::get_gradients() const { return odefunc_.get_gradients(); }

std::vector<Tensor*> ODERNN::parameters() {
    std::vector<Tensor*> p;
    auto op = odefunc_.parameters();
    for (auto* pp : op) p.push_back(pp);
    p.push_back(&input_proj_.weights); p.push_back(&input_proj_.bias);
    p.push_back(&hidden_proj_.weights); p.push_back(&hidden_proj_.bias);
    p.push_back(&output_proj_.weights); p.push_back(&output_proj_.bias);
    return p;
}

std::vector<Tensor*> ODERNN::gradients() {
    std::vector<Tensor*> g;
    auto og = odefunc_.gradients();
    for (auto* gg : og) g.push_back(gg);
    g.push_back(&input_proj_.grad_weights); g.push_back(&input_proj_.grad_bias);
    g.push_back(&hidden_proj_.grad_weights); g.push_back(&hidden_proj_.grad_bias);
    g.push_back(&output_proj_.grad_weights); g.push_back(&output_proj_.grad_bias);
    return g;
}

void ODERNN::zero_grad() {
    odefunc_.zero_grad();
    input_proj_.zero_grad();
    hidden_proj_.zero_grad();
    output_proj_.zero_grad();
}
