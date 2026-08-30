#include "titans_mag.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ============================================================================
// Titans MAG (Memory as a Gate) — implementation
// ============================================================================

TitansMAG::TitansMAG(size_t d_model, size_t d_inner, size_t seg_len)
    : d_model_(d_model), seg_len_(seg_len),
      W_qkv_(d_model, 3 * d_model),
      W_alpha_(d_model + 1, 1) {
    if (d_model == 0)
        throw std::invalid_argument("TitansMAG: d_model must be > 0");
    if (d_inner == 0) {
        d_inner_ = d_model;
    } else if (d_inner != d_model) {
        throw std::invalid_argument("TitansMAG: d_inner must equal d_model (v1 constraint)");
    } else {
        d_inner_ = d_inner;
    }
    if (seg_len != 0) {
        seg_len_ = seg_len;
    }

    // Initialize W_qkv with small Gaussian (Xavier-like).
    std::mt19937 rng(0xCAFE0002u);  // Distinct from MAC's 0xCAFE0001u
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt((double)d_model));
    auto& W = W_qkv_.weights;
    for (size_t i = 0; i < W.rows; ++i)
        for (size_t j = 0; j < W.cols; ++j)
            W(i, j) = dist(rng);
    std::fill(W_qkv_.bias.data.begin(), W_qkv_.bias.data.end(), 0.0);

    // Initialize W_alpha with small Gaussian
    std::normal_distribution<double> dist_a(0.0, 1.0 / std::sqrt((double)(d_model + 1)));
    auto& Wa = W_alpha_.weights;
    for (size_t i = 0; i < Wa.rows; ++i)
        for (size_t j = 0; j < Wa.cols; ++j)
            Wa(i, j) = dist_a(rng);
    std::fill(W_alpha_.bias.data.begin(), W_alpha_.bias.data.end(), 0.0);

    // Initialize M = 0 (persistent memory blank at init)
    M_ = Tensor(d_model, d_model);
    std::fill(M_.data.begin(), M_.data.end(), 0.0);

    // Allocate gradient buffers.
    grad_W_qkv_w_ = Tensor(3 * d_model, d_model);
    grad_W_qkv_b_ = Tensor(1, 3 * d_model);
    grad_W_alpha_w_ = Tensor(1, d_model + 1);
    grad_W_alpha_b_ = Tensor(1, 1);
    grad_M_ = Tensor(d_model, d_model);
}

// ============================================================================
// Forward
// ============================================================================
Tensor TitansMAG::forward(const Tensor& input) {
    const size_t T = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("TitansMAG: input.cols must equal d_model");

    // 1) Joint projection: qkv = x · W_qkv^T + b_qkv
    Tensor qkv = W_qkv_.forward(input);   // (T, 3*d_model)
    Tensor q_t(T, d_model_);
    Tensor k_t(T, d_model_);
    Tensor v_t(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            q_t(t, j) = qkv(t, j);
            k_t(t, j) = qkv(t, d_model_ + j);
            v_t(t, j) = qkv(t, 2 * d_model_ + j);
        }
    }

    // Allocate cache
    Tensor v_norm(T, 1);
    Tensor eta(T, 1);
    Tensor alpha(T, 1);
    Tensor M_t((T + 1) * d_model_, d_model_);  // (T+1, d, d) flat layout
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            M_t(i, j) = M_(i, j);

    // 2-5) Per-token recurrence (steps 2-4 same as MAC; step 5 is MAG-specific)
    Tensor y(T, d_model_);
    Tensor mx(T, d_model_);  // (M_t · x_t) gate values for backward
    for (size_t t = 0; t < T; ++t) {
        // v_norm[t] = ||v_t||_2
        double vnorm = 0.0;
        for (size_t j = 0; j < d_model_; ++j) vnorm += v_t(t, j) * v_t(t, j);
        vnorm = std::sqrt(vnorm);
        v_norm(t, 0) = vnorm;

        // alpha_input[t] = concat[x_t, v_norm[t]]  → (1, d_model+1)
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

        // 5) MAG-specific output: y_t = (M_t · x_t) ⊙ x_t
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t((t + 1) * d_model_ + i, j) * input(t, j);
            mx(t, i) = s;     // cache for backward
            y(t, i) = s * input(t, i);
        }
    }

    last_input_ = input.clone();
    last_q_t_ = q_t;
    last_k_t_ = k_t;
    last_v_t_ = v_t;
    last_v_norm_ = v_norm;
    last_eta_ = eta;
    last_alpha_ = alpha;
    last_M_t_ = M_t;
    last_mx_ = mx;
    last_y_t_ = y.clone();
    last_qkv_ = qkv;
    return y;
}

// ============================================================================
// Backward
// ============================================================================
Tensor TitansMAG::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    const size_t T = last_input_.rows;

    Tensor dq_t(T, d_model_);
    Tensor dk_t(T, d_model_);
    Tensor dv_t(T, d_model_);
    Tensor dx_t(T, d_model_);
    Tensor dalphain_t(T, d_model_ + 1);
    std::fill(dalphain_t.data.begin(), dalphain_t.data.end(), 0.0);
    std::fill(dv_t.data.begin(), dv_t.data.end(), 0.0);
    std::fill(dk_t.data.begin(), dk_t.data.end(), 0.0);
    std::fill(dx_t.data.begin(), dx_t.data.end(), 0.0);

    // =========================================================================
    // MAG-specific: y_t = (M_t · x_t) ⊙ x_t
    //
    // y[j] = mx[j] · x[j]  where mx = M · x.
    // dy[j] = grad_output[t][j]
    //   dL/dmx[j] = dy[j] · x[j]
    //   dL/dx[j] (from ⊙) = dy[j] · mx[j]
    //   dL/dM[i,j] (from mx = M·x) = dL/dmx[j] · x[i]  (this contributes to dM_post)
    //
    // So per-token:
    //   dmx_t[j] = dy_t[j] · x_t[j]
    //   dx_t[j] += dy_t[j] · mx_t[j]      (the ⊙ direct path)
    //   dM_post[i,j] += dmx_t[j] · x_t[i] = dy_t[j] · x_t[j] · x_t[i]   (via mx = M·x)
    //
    // Then dM_post feeds into the same M-update recurrence as MAC.
    // =========================================================================

    Tensor dM_carrier(d_model_, d_model_);
    std::fill(dM_carrier.data.begin(), dM_carrier.data.end(), 0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        double a = last_alpha_(t, 0);
        double om_a = 1.0 - a;
        double eta_t = last_eta_(t, 0);

        // ----- MAG-specific: dL/dx from ⊙x, dL/dM from mx = M·x -----
        // y = (M·x) ⊙ x. Two paths into x:
        //   (a) Direct ⊙ path: y[j] = mx[j] * x[j]  →  dx[j] += dy[j] * mx[j]
        //   (b) Indirect M·x path: mx[j] = sum_k M[j,k] * x[k]  →  dx[k] += sum_j dy[j] * M[j,k] * x[j]
        //                       AND  dM[j,k] += dmx[j] * x[k] = (dy[j] * x[j]) * x[k]
        //
        // Indexing: i = M row / y row index, j = M col / x col index.
        //   (a) and (b) are TWO separate contributions to dx, summed below.
        Tensor dM_post(d_model_, d_model_);
        std::fill(dM_post.data.begin(), dM_post.data.end(), 0.0);
        for (size_t i = 0; i < d_model_; ++i) {
            double dy_i = grad_output(t, i);
            // (a) Direct path through ⊙: dx[i] += dy[i] * mx[i]
            dx_t(t, i) += dy_i * last_mx_(t, i);
            // (b) Indirect path through mx = M·x:
            //     dmx[i] = dy[i] * x[i]     (the gate derivative)
            //     dx[k] += sum_i dmx[i] * M[i,k] = sum_i dy[i] * x[i] * M[i,k]
            for (size_t j = 0; j < d_model_; ++j) {
                // dM[i,j] += dmx[i] * x[j] = (dy[i] * x[i]) * x[j]
                double dmx_i = dy_i * last_input_(t, i);
                dM_post(i, j) += dmx_i * last_input_(t, j);
                // dx[j] (indirect path) += dmx[i] * M[i,j] = dy[i] * x[i] * M[i,j]
                dx_t(t, j) += dmx_i * last_M_t_((t + 1) * d_model_ + i, j);
            }
        }

        // ----- Add the M-update carrier from later tokens (M_{t+1} = (1-α_{t+1})·M_t + ...) -----
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dM_post(i, j) += dM_carrier(i, j);
            }
        }

        // dL/dα_t from M_t update:
        //   α_t contributes to M_t[i,j] as: v_t[i]·k_t[j] - M_{t-1}[i,j]
        double dalpha = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dalpha += dM_post(i, j) * (last_v_t_(t, i) * last_k_t_(t, j)
                                            - last_M_t_(t * d_model_ + i, j));
            }
        }

        // dL/dv_t[i] = sum_j dM_post[i,j] · α_t · k_t[j]
        // dL/dk_t[j] = sum_i dM_post[i,j] · α_t · v_t[i]
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double c = dM_post(i, j) * a;
                dv_t(t, i) += c * last_k_t_(t, j);
                dk_t(t, j) += c * last_v_t_(t, i);
            }
        }

        // dL/dM_{t-1} carrier from M_t = (1-α_t)·M_{t-1} + α_t·outer(v,k)
        // = (1-α_t) · dM_post
        Tensor dM_new_carrier(d_model_, d_model_);
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dM_new_carrier(i, j) = om_a * dM_post(i, j);
            }
        }

        // ----- Surprise chain: η_t depends on M_{t-1} via η_t = ||x_t - M_{t-1} · k_t|| -----
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

        // z_pre = W_alpha · [x_t; ||v_t||] + b_alpha
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = last_input_(t, j);
        alpha_in(0, d_model_) = last_v_norm_(t, 0);
        double z_pre = W_alpha_.bias(0, 0);
        for (size_t j = 0; j < d_model_ + 1; ++j) z_pre += W_alpha_.weights(0, j) * alpha_in(0, j);
        double sigma_pre = 1.0 / (1.0 + std::exp(-z_pre));
        double sigma_pre_deriv = sigma_pre * (1.0 - sigma_pre);
        double deta = dalpha * sigma_pre;
        double dz_pre = dalpha * eta_t * sigma_pre_deriv;

        // dL/dW_alpha.weights(0, j) += dz_pre · alpha_in(0, j)
        for (size_t j = 0; j < d_model_ + 1; ++j) {
            grad_W_alpha_w_(0, j) += dz_pre * alpha_in(0, j);
            dalphain_t(t, j) += dz_pre * W_alpha_.weights(0, j);
        }
        grad_W_alpha_b_(0, 0) += dz_pre;

        // dα_in → dx, dvnorm
        for (size_t j = 0; j < d_model_; ++j) {
            dx_t(t, j) += dalphain_t(t, j);
        }
        if (last_v_norm_(t, 0) > eps) {
            double dvnorm = dalphain_t(t, d_model_);
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
                    double deta_dM = deta * (-inv_Nxpe * inv_Nd * diff(i, 0) * last_k_t_(t, j));
                    dM_new_carrier(i, j) += deta_dM;
                    // dη_t/dk_t[j] = -(1/(N_x+eps))·(1/N_d)·sum_i M_{t-1}[i,j]·diff[i]
                    dk_t(t, j) += deta * (-inv_Nxpe * inv_Nd * last_M_t_(t * d_model_ + i, j) * diff(i, 0));
                }
            }
        }

        // dL/dv_t and dL/dk_t propagate to dL/dW_qkv.
        //   dL/dW_qkv.bias[2d + i] += dL/dv_t[t, i]
        //   dL/dW_qkv.bias[d + j]  += dL/dk_t[t, j]
        // dL/dq_t is NOT needed for MAG (q_t doesn't appear in output) — but we still
        // computed q_t, so its parameter gradient is 0. We still must accumulate dL/dW_qkv.bias[j]
        // if q has any path... q_t doesn't enter y_t in MAG, so dL/dW_qkv's q-slice is 0.
        // We do NOT accumulate it. The test for q-slice will fail — but MAC has it for the
        // y_t = M_t · q_t path; MAG doesn't.
        for (size_t i = 0; i < d_model_; ++i) {
            grad_W_qkv_b_(0, d_model_ + i) += dk_t(t, i);
            grad_W_qkv_b_(0, 2 * d_model_ + i) += dv_t(t, i);
        }

        // dL/dW_qkv.weights: only k and v slices contribute (q is unused in MAG's output).
        for (size_t j = 0; j < 2 * d_model_; ++j) {
            double dqkv_tj = (j < d_model_) ? dk_t(t, j)
                                             : dv_t(t, j - d_model_);
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_qkv_w_(d_model_ + j, k) += dqkv_tj * last_input_(t, k);
            }
        }

        // Update dM_carrier for the NEXT iteration (token t-1).
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dM_carrier(i, j) = dM_new_carrier(i, j);
            }
        }
    }

    // After the loop, dM_carrier holds dL/dM_0 = dL/dM_ (persistent memory).
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_M_(i, j) = dM_carrier(i, j);
        }
    }

    // dL/dx_t also includes dL/d(W_qkv · x_t) — but ONLY k and v slices (q unused).
    //   dL/dx_t[k] += sum_j (dL/dqkv_t[j]) · W_qkv.weights[j, k]  for j in [d, 3d)
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            // k slice: j = d_model + i
            for (size_t j = 0; j < d_model_; ++j) {
                s += dk_t(t, j) * W_qkv_.weights(d_model_ + j, k);
            }
            // v slice: j = 2*d_model + i
            for (size_t j = 0; j < d_model_; ++j) {
                s += dv_t(t, j) * W_qkv_.weights(2 * d_model_ + j, k);
            }
            dx_t(t, k) += s;
        }
    }

    return dx_t;
}

// ============================================================================
// Optimizer interface
// ============================================================================
void TitansMAG::update_weights(double learning_rate) {
    W_qkv_.weights = W_qkv_.weights - grad_W_qkv_w_ * learning_rate;
    W_qkv_.bias   = W_qkv_.bias   - grad_W_qkv_b_ * learning_rate;
    W_alpha_.weights = W_alpha_.weights - grad_W_alpha_w_ * learning_rate;
    W_alpha_.bias   = W_alpha_.bias   - grad_W_alpha_b_ * learning_rate;
    M_ = M_ - grad_M_ * learning_rate;
}

void TitansMAG::zero_grad() {
    std::fill(W_qkv_.grad_weights.data.begin(), W_qkv_.grad_weights.data.end(), 0.0);
    std::fill(W_qkv_.grad_bias.data.begin(), W_qkv_.grad_bias.data.end(), 0.0);
    std::fill(W_alpha_.grad_weights.data.begin(), W_alpha_.grad_weights.data.end(), 0.0);
    std::fill(W_alpha_.grad_bias.data.begin(), W_alpha_.grad_bias.data.end(), 0.0);
    std::fill(grad_M_.data.begin(), grad_M_.data.end(), 0.0);
    std::fill(grad_W_qkv_w_.data.begin(), grad_W_qkv_w_.data.end(), 0.0);
    std::fill(grad_W_qkv_b_.data.begin(), grad_W_qkv_b_.data.end(), 0.0);
    std::fill(grad_W_alpha_w_.data.begin(), grad_W_alpha_w_.data.end(), 0.0);
    std::fill(grad_W_alpha_b_.data.begin(), grad_W_alpha_b_.data.end(), 0.0);
}

std::vector<Tensor*> TitansMAG::parameters() {
    return { &W_qkv_.weights, &W_qkv_.bias,
             &W_alpha_.weights, &W_alpha_.bias,
             &M_ };
}

std::vector<Tensor*> TitansMAG::gradients() {
    return { &grad_W_qkv_w_, &grad_W_qkv_b_,
             &grad_W_alpha_w_, &grad_W_alpha_b_,
             &grad_M_ };
}

void TitansMAG::copy_params_from(const TitansMAG& other) {
    W_qkv_.weights = other.W_qkv_.weights.clone();
    W_qkv_.bias = other.W_qkv_.bias.clone();
    W_alpha_.weights = other.W_alpha_.weights.clone();
    W_alpha_.bias = other.W_alpha_.bias.clone();
    M_ = other.M_.clone();
}
