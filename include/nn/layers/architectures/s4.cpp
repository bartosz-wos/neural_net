#include "s4.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

S4Layer::S4Layer(int d_model, int d_state)
    : d_model_(d_model), d_state_(d_state),
      seq_len_cached_(0), has_cachedKernel_(false),
      grad_x_proj(d_model, d_state * 2),
      grad_W_out(d_state, d_model),
      grad_b_out(1, d_model),
      grad_Lambda_(d_state, 1),
      grad_B_(d_state, 1),
      grad_C_(d_state, 1),
      x_proj(d_model, d_state * 2),
      W_out(d_state, d_model),
      b_out(1, d_model),
      Lambda(d_state, 1),
      B(d_state, 1),
      C(d_state, 1),
      D(0.0) {

    if (d_model <= 0 || d_state <= 0) {
        throw std::invalid_argument("S4Layer: d_model and d_state must be positive");
    }

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 1.0);

    double scale_x = std::sqrt(2.0 / (d_model + d_state * 2));
    for (int i = 0; i < d_model; ++i)
        for (int j = 0; j < d_state * 2; ++j)
            x_proj[i][j] = dis(gen) * scale_x;

    double scale_w = std::sqrt(2.0 / (d_state + d_model));
    for (int i = 0; i < d_state; ++i)
        for (int j = 0; j < d_model; ++j)
            W_out[i][j] = dis(gen) * scale_w;

    b_out.fill(0.0);

    for (int n = 0; n < d_state; ++n)
        Lambda[n][0] = -0.5 * (2.0 * n + 1.0);

    for (int n = 0; n < d_state; ++n)
        B[n][0] = std::sqrt(2.0 * n + 1.0);

    for (int n = 0; n < d_state; ++n)
        C[n][0] = dis(gen) * 0.1;

    D = 0.0;
    zero_grad();
}

double S4Layer::discretize_Lambda(double lam, double dt) {
    return (1.0 + 0.5 * dt * lam) / (1.0 - 0.5 * dt * lam);
}

double S4Layer::discretize_B(double B_val, double lam, double dt) {
    return B_val / (1.0 - 0.5 * dt * lam);
}

Tensor S4Layer::forward(const Tensor& input) {
    int N = input.rows;
    int seq_len = static_cast<int>(input.cols);

    if (input.cols == 0) {
        throw std::invalid_argument("S4Layer: input has zero sequence length");
    }
    if (static_cast<int>(input.rows) != d_model_) {
        throw std::invalid_argument("S4Layer: input dimension mismatch");
    }

    last_input_ = input;

    // u_proj = x_proj^T @ input -> (d_state*2, seq_len)
    // x_proj_t (d_state*2, d_model) @ input (d_model, seq_len)
    Tensor x_proj_t = x_proj.transpose();
    Tensor u_proj = x_proj_t * input;

    last_u_ = u_proj;
    seq_len_cached_ = seq_len;
    has_cachedKernel_ = false;

    // Split u_proj into u_B and u_C (each d_state x seq_len)
    Tensor u_B(d_state_, seq_len);
    Tensor u_C(d_state_, seq_len);
    for (int n = 0; n < d_state_; ++n)
        for (int t = 0; t < seq_len; ++t) {
            u_B[n][t] = u_proj[n][t];
            u_C[n][t] = u_proj[d_state_ + n][t];
        }

    // Compute SSM kernel K (d_state, seq_len)
    // K[n][t] = C_n * u_C[n][t] + A_d_n * K[n][t-1] + B_d_n * u_B[n][t]
    Tensor K(d_state_, seq_len);
    K.fill(0.0);
    const double dt = 0.1;

    for (int n = 0; n < d_state_; ++n) {
        double lam = Lambda[n][0];
        double A_d = discretize_Lambda(lam, dt);
        double B_d = discretize_B(B[n][0], lam, dt);
        double C_val = C[n][0];
        double k_prev = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            double k_t = C_val * u_C[n][t] + A_d * k_prev + B_d * u_B[n][t];
            K[n][t] = k_t;
            k_prev = k_t;
        }
    }

    cached_K_ = K;
    has_cachedKernel_ = true;

    // y = W_out^T @ K + b_out + D * input
    // W_out: (d_state, d_model), K: (d_state, seq_len)
    // W_out^T @ K: (d_model, seq_len)
    Tensor y1 = W_out.transpose() * K;

    Tensor y(N, seq_len);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < seq_len; ++j)
            y[i][j] = y1[i][j] + b_out[0][i] + D * input[i][j];

    return y;
}

Tensor S4Layer::backward(const Tensor& grad_output, double /* learning_rate */) {
    int N = grad_output.rows;
    int seq_len = static_cast<int>(grad_output.cols);

    grad_x_proj.fill(0.0);
    grad_W_out.fill(0.0);
    grad_b_out.fill(0.0);
    grad_Lambda_.fill(0.0);
    grad_B_.fill(0.0);
    grad_C_.fill(0.0);
    grad_D_ = 0.0;

    const double dt = 0.1;

    // Reconstruct u_proj from cached last_input_
    // u_proj = x_proj^T @ input -> (d_state*2, seq_len)
    Tensor x_proj_t = x_proj.transpose();
    Tensor u_proj = x_proj_t * last_input_;

    // Split into u_B and u_C
    Tensor u_B(d_state_, seq_len);
    Tensor u_C(d_state_, seq_len);
    for (int n = 0; n < d_state_; ++n)
        for (int t = 0; t < seq_len; ++t) {
            u_B[n][t] = u_proj[n][t];
            u_C[n][t] = u_proj[d_state_ + n][t];
        }

    // Reconstruct kernel K
    Tensor K(d_state_, seq_len);
    K.fill(0.0);
    for (int n = 0; n < d_state_; ++n) {
        double lam = Lambda[n][0];
        double A_d = discretize_Lambda(lam, dt);
        double B_d = discretize_B(B[n][0], lam, dt);
        double C_val = C[n][0];
        double k_prev = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            double k_t = C_val * u_C[n][t] + A_d * k_prev + B_d * u_B[n][t];
            K[n][t] = k_t;
            k_prev = k_t;
        }
    }

    // dL/dK = W_out @ grad_output
    // W_out: (d_state, d_model), grad_output: (d_model, seq_len)
    // Result: (d_state, seq_len)
    Tensor grad_K = W_out * grad_output;

    // --- grad_C: dL/dK[n][t] * u_C[n][t]
    for (int n = 0; n < d_state_; ++n) {
        double acc = 0.0;
        for (int t = 0; t < seq_len; ++t)
            acc += grad_K[n][t] * u_C[n][t];
        grad_C_[n][0] = acc;
    }

    // --- grad_B ---
    for (int n = 0; n < d_state_; ++n) {
        double lam = Lambda[n][0];
        double A_d = discretize_Lambda(lam, dt);
        double B_val = B[n][0];
        double B_d = discretize_B(B_val, lam, dt);
        double dBdB = B_d / B_val;
        double acc = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            double power = 1.0;
            for (int j = 0; j <= t; ++j) {
                acc += grad_K[n][t] * power * u_B[n][j];
                power *= A_d;
            }
        }
        grad_B_[n][0] = acc * dBdB;
    }

    // --- grad_Lambda ---
    for (int n = 0; n < d_state_; ++n) {
        double lam = Lambda[n][0];
        double A_d = discretize_Lambda(lam, dt);
        double dAd_dlam = dt / std::pow(1.0 - 0.5 * dt * lam, 2.0);
        double acc = 0.0;
        for (int t = 1; t < seq_len; ++t) {
            double power = 1.0;
            double sum_k = 0.0;
            for (int s = 0; s < t; ++s) {
                sum_k += K[n][t - 1 - s] * power;
                power *= A_d;
            }
            acc += grad_K[n][t] * sum_k;
        }
        grad_Lambda_[n][0] = acc * dAd_dlam;
    }

    // --- grad_D ---
    {
        double acc = 0.0;
        for (int i = 0; i < N; ++i)
            for (int t = 0; t < seq_len; ++t)
                acc += grad_output[i][t] * last_input_[i][t];
        grad_D_ = acc;
    }

    // --- grad_W_out: grad_W_out[n][i] = sum_t grad_output[i][t] * K[n][t]
    // K (d_state, seq_len) * grad_output^T (seq_len, d_model) = (d_state, d_model)
    Tensor grad_output_t = grad_output.transpose();
    grad_W_out = K * grad_output_t;

    // --- grad_b_out ---
    for (int i = 0; i < d_model_; ++i) {
        double acc = 0.0;
        for (int t = 0; t < seq_len; ++t)
            acc += grad_output[i][t];
        grad_b_out[0][i] = acc;
    }

    // --- grad_u_B and grad_u_C for x_proj gradients ---
    // Backprop through SSM recurrence (backward through time):
    // grad_u_B[n][j] = sum_{t=j}^{L-1} grad_K[n][t] * B_d * A_d^{t-j}
    // grad_u_C[n][j] = sum_{t=j}^{L-1} grad_K[n][t] * C_val * A_d^{t-j}
    Tensor grad_u_B(d_state_, seq_len);
    Tensor grad_u_C(d_state_, seq_len);

    for (int n = 0; n < d_state_; ++n) {
        double lam = Lambda[n][0];
        double A_d = discretize_Lambda(lam, dt);
        double B_d = discretize_B(B[n][0], lam, dt);
        double C_val = C[n][0];

        for (int j = 0; j < seq_len; ++j) {
            double acc_B = 0.0;
            double acc_C = 0.0;
            double power = 1.0;
            for (int t = j; t < seq_len; ++t) {
                acc_B += grad_K[n][t] * power * B_d;
                acc_C += grad_K[n][t] * power * C_val;
                power *= A_d;
            }
            grad_u_B[n][j] = acc_B;
            grad_u_C[n][j] = acc_C;
        }
    }

    // --- grad_x_proj ---
    // u_proj = x_proj^T @ input
    // grad_x_proj[j][n] = sum_t grad_u_proj[n][t] * input[j][t]
    // grad_x_proj = input @ grad_u_proj^T = (d_model, seq_len) @ (seq_len, d_state*2) = (d_model, d_state*2)
    Tensor grad_u_proj(d_state_ * 2, seq_len);
    for (int n = 0; n < d_state_; ++n) {
        for (int t = 0; t < seq_len; ++t) {
            grad_u_proj[n][t] = grad_u_B[n][t];
            grad_u_proj[d_state_ + n][t] = grad_u_C[n][t];
        }
    }
    Tensor grad_u_proj_t = grad_u_proj.transpose();  // (seq_len, d_state*2)
    grad_x_proj = last_input_ * grad_u_proj_t;       // (d_model, d_state*2)

    // --- grad_input ---
    // dinput from SSM path: x_proj @ grad_u_proj = (d_model, d_state*2) @ (d_state*2, seq_len) = (d_model, seq_len)
    // dinput from skip: D * grad_output
    // total: grad_output * D + x_proj @ grad_u_proj
    Tensor grad_input_ssm = x_proj * grad_u_proj;  // (d_model, seq_len)

    Tensor grad_input(N, seq_len);
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < seq_len; ++t)
            grad_input[i][t] = grad_output[i][t] * D + grad_input_ssm[i][t];

    return grad_input;
}

void S4Layer::update_weights(double learning_rate) {
    for (int i = 0; i < d_model_; ++i)
        for (int j = 0; j < d_state_ * 2; ++j)
            x_proj[i][j] -= learning_rate * grad_x_proj[i][j];

    for (int i = 0; i < d_state_; ++i)
        for (int j = 0; j < d_model_; ++j)
            W_out[i][j] -= learning_rate * grad_W_out[i][j];

    for (int j = 0; j < d_model_; ++j)
        b_out[0][j] -= learning_rate * grad_b_out[0][j];

    for (int n = 0; n < d_state_; ++n)
        Lambda[n][0] -= learning_rate * grad_Lambda_[n][0];

    for (int n = 0; n < d_state_; ++n)
        B[n][0] -= learning_rate * grad_B_[n][0];

    for (int n = 0; n < d_state_; ++n)
        C[n][0] -= learning_rate * grad_C_[n][0];

    D -= learning_rate * grad_D_;
}

std::vector<Tensor*> S4Layer::parameters() {
    return {&x_proj, &W_out, &b_out, &Lambda, &B, &C};
}

std::vector<Tensor*> S4Layer::gradients() {
    return {&grad_x_proj, &grad_W_out, &grad_b_out, &grad_Lambda_, &grad_B_, &grad_C_};
}

void S4Layer::zero_grad() {
    grad_x_proj.fill(0.0);
    grad_W_out.fill(0.0);
    grad_b_out.fill(0.0);
    grad_Lambda_.fill(0.0);
    grad_B_.fill(0.0);
    grad_C_.fill(0.0);
    grad_D_ = 0.0;
}
