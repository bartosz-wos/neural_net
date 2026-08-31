#include "titans_mal.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ============================================================================
// Titans MAL (Memory as a Layer) — implementation
// ============================================================================

TitansMAL::TitansMAL(size_t d_model, size_t d_inner, size_t seg_len)
    : d_model_(d_model), seg_len_(seg_len),
      W_qkv_(d_model, 3 * d_model),
      W_alpha_(d_model + 1, 1),
      W_p_(d_model, d_model) {
    if (d_model == 0)
        throw std::invalid_argument("TitansMAL: d_model must be > 0");
    if (d_inner == 0) {
        d_inner_ = d_model;
    } else if (d_inner != d_model) {
        throw std::invalid_argument("TitansMAL: d_inner must equal d_model (v1 constraint)");
    } else {
        d_inner_ = d_inner;
    }
    if (seg_len != 0) seg_len_ = seg_len;

    // Initialize W_qkv with small Gaussian (Xavier-like).
    std::mt19937 rng(0xCAFE0003u);  // Distinct from MAC (0xCAFE0001) and MAG (0xCAFE0002).
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(static_cast<double>(d_model)));
    {
        std::vector<double> w(3 * d_model * d_model);
        for (auto& v : w) v = nd(rng);
        W_qkv_.weights = Tensor(3 * d_model, d_model, w.data());
        W_qkv_.bias = Tensor(1, 3 * d_model);
    }

    // Initialize W_alpha with small Gaussian — IMPORTANT: not zero, otherwise
    // the surprise-chain gradient test would pass vacuously.
    std::normal_distribution<double> nd_a(0.0, 1.0 / std::sqrt(static_cast<double>(d_model + 1)));
    {
        std::vector<double> w(1 * (d_model + 1));
        for (auto& v : w) v = 0.1 * nd_a(rng);
        W_alpha_.weights = Tensor(1, d_model + 1, w.data());
        W_alpha_.bias = Tensor(1, 1);
    }

    // Initialize W_p (MAL-specific input gate) with small Gaussian.
    {
        std::vector<double> w(d_model * d_model);
        for (auto& v : w) v = nd(rng);
        W_p_.weights = Tensor(d_model, d_model, w.data());
        W_p_.bias = Tensor(1, d_model);
    }

    // Initialize persistent memory M to zero.
    M_ = Tensor(d_model, d_model);
    std::fill(M_.data.begin(), M_.data.end(), 0.0);

    // Allocate gradient buffers.
    grad_W_qkv_w_ = Tensor(3 * d_model, d_model);
    grad_W_qkv_b_ = Tensor(1, 3 * d_model);
    grad_W_alpha_w_ = Tensor(1, d_model + 1);
    grad_W_alpha_b_ = Tensor(1, 1);
    grad_W_p_w_ = Tensor(d_model, d_model);
    grad_W_p_b_ = Tensor(1, d_model);
    grad_M_ = Tensor(d_model, d_model);
}

// ============================================================================
// Forward
// ============================================================================
Tensor TitansMAL::forward(const Tensor& input) {
    const size_t T = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("TitansMAL: input.cols must equal d_model");

    // 1) Joint projection: qkv = x · W_qkv^T + b_qkv   (k, v used; q unused)
    Tensor qkv = W_qkv_.forward(input);   // (T, 3*d_model)
    Tensor k_t(T, d_model_);
    Tensor v_t(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            k_t(t, j) = qkv(t, d_model_ + j);
            v_t(t, j) = qkv(t, 2 * d_model_ + j);
        }
    }

    // 2) MAL-specific input gate: p_t = sigmoid(W_p · x_t + b_p)   (T, d_model)
    Tensor p_t_pre = W_p_.forward(input);   // (T, d_model)
    Tensor p_t(T, d_model_);
    for (size_t i = 0; i < p_t_pre.data.size(); ++i) {
        p_t.data[i] = 1.0 / (1.0 + std::exp(-p_t_pre.data[i]));
    }

    // 3) x̃_t = p_t ⊙ x_t                                            (T, d_model)
    Tensor x_tilde(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            x_tilde(t, j) = p_t(t, j) * input(t, j);
        }
    }

    // 4-5) M-update + output (same structure as MAC, with q_t ← x̃_t)
    Tensor v_norm(T, 1);
    Tensor eta(T, 1);
    Tensor alpha(T, 1);
    Tensor M_t((T + 1) * d_model_, d_model_);
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            M_t(i, j) = M_(i, j);

    Tensor y(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        // v_norm[t] = ||v_t||_2
        double vnorm = 0.0;
        for (size_t j = 0; j < d_model_; ++j) vnorm += v_t(t, j) * v_t(t, j);
        vnorm = std::sqrt(vnorm);
        v_norm(t, 0) = vnorm;

        // alpha_input = concat[x_t, v_norm[t]] → (1, d_model+1)
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = input(t, j);
        alpha_in(0, d_model_) = vnorm;
        Tensor alpha_pre = W_alpha_.forward(alpha_in);
        double sigma_alpha = 1.0 / (1.0 + std::exp(-alpha_pre(0, 0)));

        // surprise η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
        Tensor mk_pred(d_model_, 1);
        double xnorm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t(t * d_model_ + i, j) * k_t(t, j);
            mk_pred(i, 0) = s;
            xnorm += input(t, i) * input(t, i);
        }
        xnorm = std::sqrt(xnorm);
        double diff_norm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double d_i = input(t, i) - mk_pred(i, 0);
            diff_norm += d_i * d_i;
        }
        diff_norm = std::sqrt(diff_norm);
        const double eps = 1e-8;
        double eta_t = diff_norm / (xnorm + eps);
        eta(t, 0) = eta_t;
        alpha(t, 0) = sigma_alpha * eta_t;

        // M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
        double a = alpha(t, 0);
        double om_a = 1.0 - a;
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double prev = M_t(t * d_model_ + i, j);
                double outer_vk = v_t(t, i) * k_t(t, j);
                M_t((t + 1) * d_model_ + i, j) = om_a * prev + a * outer_vk;
            }
        }

        // 6) MAL-specific output: y_t = M_t · x̃_t   (clean — no �)
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t((t + 1) * d_model_ + i, j) * x_tilde(t, j);
            y(t, i) = s;
        }
    }

    last_input_ = input.clone();
    last_k_t_ = k_t;
    last_v_t_ = v_t;
    last_v_norm_ = v_norm;
    last_eta_ = eta;
    last_alpha_ = alpha;
    last_M_t_ = M_t;
    last_p_t_ = p_t;
    last_x_tilde_ = x_tilde;
    last_y_t_ = y.clone();
    last_qkv_ = qkv;
    return y;
}

// ============================================================================
// Backward
// ============================================================================
Tensor TitansMAL::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    const size_t T = last_input_.rows;

    Tensor dk_t(T, d_model_);
    Tensor dv_t(T, d_model_);
    Tensor dx_t(T, d_model_);
    Tensor dx_tilde(T, d_model_);
    std::fill(dk_t.data.begin(), dk_t.data.end(), 0.0);
    std::fill(dv_t.data.begin(), dv_t.data.end(), 0.0);
    std::fill(dx_t.data.begin(), dx_t.data.end(), 0.0);
    std::fill(dx_tilde.data.begin(), dx_tilde.data.end(), 0.0);

    Tensor dM_carrier(d_model_, d_model_);
    std::fill(dM_carrier.data.begin(), dM_carrier.data.end(), 0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        double a = last_alpha_(t, 0);
        double om_a = 1.0 - a;
        double eta_t = last_eta_(t, 0);

        // ----- Output chain: y_t = M_t · x̃_t → dy_t -----
        //   dx_tilde[i] += dy[j] · M_t[j, i]
        //   dM_post[j, i] += dy[j] · x̃_t[i]
        Tensor dM_post(d_model_, d_model_);
        std::fill(dM_post.data.begin(), dM_post.data.end(), 0.0);
        for (size_t j = 0; j < d_model_; ++j) {
            double dy_j = grad_output(t, j);
            for (size_t i = 0; i < d_model_; ++i) {
                dx_tilde(t, i) += dy_j * last_M_t_((t + 1) * d_model_ + j, i);
                dM_post(j, i) += dy_j * last_x_tilde_(t, i);
            }
        }

        // Add the M-update carrier from later tokens.
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_post(i, j) += dM_carrier(i, j);

        // dα from M_t update: α_t contributes (v⊗k - M_prev) to M_t.
        double dalpha = 0.0;
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dalpha += dM_post(i, j) * (last_v_t_(t, i) * last_k_t_(t, j)
                                            - last_M_t_(t * d_model_ + i, j));

        // dL/dv_t and dL/dk_t from M_t update.
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                double c = dM_post(i, j) * a;
                dv_t(t, i) += c * last_k_t_(t, j);
                dk_t(t, j) += c * last_v_t_(t, i);
            }

        // dM_carrier → next iteration: (1-α) · dM_post.
        Tensor dM_new_carrier(d_model_, d_model_);
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_new_carrier(i, j) = om_a * dM_post(i, j);

        // ----- Surprise chain (identical to MAC) -----
        Tensor mk(d_model_, 1);
        Tensor diff(d_model_, 1);
        double N_d = 0.0, N_x = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += last_M_t_(t * d_model_ + i, j) * last_k_t_(t, j);
            mk(i, 0) = s;
            diff(i, 0) = last_input_(t, i) - s;
            N_d += diff(i, 0) * diff(i, 0);
            N_x += last_input_(t, i) * last_input_(t, i);
        }
        N_d = std::sqrt(N_d);
        N_x = std::sqrt(N_x);
        const double eps = 1e-8;

        // W_alpha gradient: dα = dz · σ(1-σ) · α_input.
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = last_input_(t, j);
        alpha_in(0, d_model_) = last_v_norm_(t, 0);
        double z_pre = W_alpha_.bias(0, 0);
        for (size_t j = 0; j < d_model_ + 1; ++j) z_pre += W_alpha_.weights(0, j) * alpha_in(0, j);
        double sigma_pre = 1.0 / (1.0 + std::exp(-z_pre));
        double sigma_pre_deriv = sigma_pre * (1.0 - sigma_pre);
        double deta = dalpha * sigma_pre;
        double dz_pre = dalpha * eta_t * sigma_pre_deriv;

        for (size_t j = 0; j < d_model_ + 1; ++j) {
            grad_W_alpha_w_(0, j) += dz_pre * alpha_in(0, j);
        }
        grad_W_alpha_b_(0, 0) += dz_pre;

        // dα_in → dx (input-side contributions from W_alpha).
        for (size_t j = 0; j < d_model_; ++j) {
            dx_t(t, j) += dz_pre * W_alpha_.weights(0, j);
        }
        if (last_v_norm_(t, 0) > eps) {
            double dvnorm = dz_pre * W_alpha_.weights(0, d_model_);
            for (size_t i = 0; i < d_model_; ++i) {
                dv_t(t, i) += dvnorm * last_v_t_(t, i) / last_v_norm_(t, 0);
            }
        }

        // dL/dη_t = deta. Routes to dx, dk, dM_{t-1}.
        if (N_x > eps && N_d > eps) {
            double inv_Nx = 1.0 / N_x;
            double inv_Nd = 1.0 / N_d;
            double inv_Nxpe = 1.0 / (N_x + eps);
            for (size_t i = 0; i < d_model_; ++i) {
                double deta_dx = deta * (diff(i, 0) * inv_Nd - eta_t * last_input_(t, i) * inv_Nx) * inv_Nxpe;
                dx_t(t, i) += deta_dx;
                for (size_t j = 0; j < d_model_; ++j) {
                    dM_new_carrier(i, j) += deta * (-inv_Nxpe * inv_Nd * diff(i, 0) * last_k_t_(t, j));
                    dk_t(t, j) += deta * (-inv_Nxpe * inv_Nd * last_M_t_(t * d_model_ + i, j) * diff(i, 0));
                }
            }
        }

        // dL/dW_qkv (k,v slices only — q unused) and dL/dW_qkv.bias.
        for (size_t i = 0; i < d_model_; ++i) {
            grad_W_qkv_b_(0, d_model_ + i) += dk_t(t, i);
            grad_W_qkv_b_(0, 2 * d_model_ + i) += dv_t(t, i);
        }
        for (size_t j = 0; j < 2 * d_model_; ++j) {
            double dqkv_tj = (j < d_model_) ? dk_t(t, j) : dv_t(t, j - d_model_);
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_qkv_w_(d_model_ + j, k) += dqkv_tj * last_input_(t, k);
            }
        }

        // ----- MAL-specific: input-gate chain through W_p -----
        // x̃_t = p_t � x_t. We already accumulated dx_tilde_t.
        //   dx[i] += dx_tilde[i] · p_t[i]                        (the ⊙ path)
        //   dp_t[i]  = dx_tilde[i] · x_t[i]                      (the gate derivative)
        Tensor dp_t(1, d_model_);
        for (size_t i = 0; i < d_model_; ++i) {
            dx_t(t, i) += dx_tilde(t, i) * last_p_t_(t, i);
            dp_t(0, i) = dx_tilde(t, i) * last_input_(t, i);
        }
        // p_t = sigmoid(W_p · x_t + b_p).  sigmoid'(z) = p(1-p).
        //   dW_p.weights[i, k] += dp_t[i] · p(1-p) · x_t[k]
        //   dW_p.bias[i]        += dp_t[i] · p(1-p)
        //   dx_t[k]             += dp_t[i] · p(1-p) · W_p.weights[i, k]
        for (size_t i = 0; i < d_model_; ++i) {
            double p_i = last_p_t_(t, i);
            double p_deriv = p_i * (1.0 - p_i);
            double dp_s = dp_t(0, i) * p_deriv;
            grad_W_p_b_(0, i) += dp_s;
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_p_w_(i, k) += dp_s * last_input_(t, k);
                dx_t(t, k) += dp_s * W_p_.weights(i, k);
            }
        }

        // Update dM_carrier for the NEXT iteration (token t-1).
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_carrier(i, j) = dM_new_carrier(i, j);
    }

    // After the loop, dM_carrier holds dL/dM_0 = dL/dM_ (persistent memory).
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_M_(i, j) = dM_carrier(i, j);

    // dL/dx_t from W_qkv (k,v slices only — q unused).
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += dk_t(t, j) * W_qkv_.weights(d_model_ + j, k);
            for (size_t j = 0; j < d_model_; ++j) s += dv_t(t, j) * W_qkv_.weights(2 * d_model_ + j, k);
            dx_t(t, k) += s;
        }
    }

    return dx_t;
}

// ============================================================================
// Optimizer interface
// ============================================================================
void TitansMAL::update_weights(double learning_rate) {
    W_qkv_.weights   = W_qkv_.weights   - grad_W_qkv_w_   * learning_rate;
    W_qkv_.bias      = W_qkv_.bias      - grad_W_qkv_b_   * learning_rate;
    W_alpha_.weights = W_alpha_.weights - grad_W_alpha_w_ * learning_rate;
    W_alpha_.bias    = W_alpha_.bias    - grad_W_alpha_b_ * learning_rate;
    W_p_.weights     = W_p_.weights     - grad_W_p_w_     * learning_rate;
    W_p_.bias        = W_p_.bias        - grad_W_p_b_     * learning_rate;
    M_ = M_ - grad_M_ * learning_rate;
}

void TitansMAL::zero_grad() {
    std::fill(W_qkv_.grad_weights.data.begin(), W_qkv_.grad_weights.data.end(), 0.0);
    std::fill(W_qkv_.grad_bias.data.begin(), W_qkv_.grad_bias.data.end(), 0.0);
    std::fill(W_alpha_.grad_weights.data.begin(), W_alpha_.grad_weights.data.end(), 0.0);
    std::fill(W_alpha_.grad_bias.data.begin(), W_alpha_.grad_bias.data.end(), 0.0);
    std::fill(W_p_.grad_weights.data.begin(), W_p_.grad_weights.data.end(), 0.0);
    std::fill(W_p_.grad_bias.data.begin(), W_p_.grad_bias.data.end(), 0.0);
    std::fill(grad_M_.data.begin(), grad_M_.data.end(), 0.0);
    std::fill(grad_W_qkv_w_.data.begin(), grad_W_qkv_w_.data.end(), 0.0);
    std::fill(grad_W_qkv_b_.data.begin(), grad_W_qkv_b_.data.end(), 0.0);
    std::fill(grad_W_alpha_w_.data.begin(), grad_W_alpha_w_.data.end(), 0.0);
    std::fill(grad_W_alpha_b_.data.begin(), grad_W_alpha_b_.data.end(), 0.0);
    std::fill(grad_W_p_w_.data.begin(), grad_W_p_w_.data.end(), 0.0);
    std::fill(grad_W_p_b_.data.begin(), grad_W_p_b_.data.end(), 0.0);
}

std::vector<Tensor*> TitansMAL::parameters() {
    return { &W_qkv_.weights, &W_qkv_.bias,
             &W_alpha_.weights, &W_alpha_.bias,
             &W_p_.weights, &W_p_.bias };
}

std::vector<Tensor*> TitansMAL::gradients() {
    return { &grad_W_qkv_w_, &grad_W_qkv_b_,
             &grad_W_alpha_w_, &grad_W_alpha_b_,
             &grad_W_p_w_, &grad_W_p_b_ };
}

void TitansMAL::copy_params_from(const TitansMAL& other) {
    W_qkv_.weights = other.W_qkv_.weights.clone();
    W_qkv_.bias = other.W_qkv_.bias.clone();
    W_alpha_.weights = other.W_alpha_.weights.clone();
    W_alpha_.bias = other.W_alpha_.bias.clone();
    W_p_.weights = other.W_p_.weights.clone();
    W_p_.bias = other.W_p_.bias.clone();
    M_ = other.M_.clone();
}
