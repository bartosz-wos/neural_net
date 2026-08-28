#include "titans_mac.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ============================================================================
// Titans MAC (Memory as a Context) — implementation
// ============================================================================

TitansMAC::TitansMAC(size_t d_model, size_t d_inner, size_t seg_len)
    : d_model_(d_model), seg_len_(seg_len),
      W_qkv_(d_model, 3 * d_model),
      W_alpha_(d_model + 1, 1) {
    if (d_model == 0)
        throw std::invalid_argument("TitansMAC: d_model must be > 0");
    if (d_inner == 0) {
        d_inner_ = d_model;
    } else if (d_inner != d_model) {
        throw std::invalid_argument("TitansMAC: d_inner must equal d_model (v1 constraint)");
    } else {
        d_inner_ = d_inner;
    }
    if (seg_len != 0) {
        // Reserved for future segment-forget implementation; currently no-op.
        seg_len_ = seg_len;
    }

    // Initialize W_qkv with small Gaussian (Xavier-like).
    std::mt19937 rng(0xCAFE0001u);
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(static_cast<double>(d_model)));
    {
        std::vector<double> w(3 * d_model * d_model);
        for (auto& v : w) v = nd(rng);
        W_qkv_.weights = Tensor(3 * d_model, d_model, w.data());
        W_qkv_.bias = Tensor(1, 3 * d_model);
    }

    // Initialize W_alpha with small Gaussian — IMPORTANT: not zero, otherwise
    // the surprise-chain gradient test (Test 7) passes vacuously.
    {
        std::vector<double> w(1 * (d_model + 1));
        for (auto& v : w) v = 0.1 * nd(rng);
        W_alpha_.weights = Tensor(1, d_model + 1, w.data());
        W_alpha_.bias = Tensor(1, 1);
    }

    // Initialize persistent memory M to zero. (Paper says small init or zero;
    // zero is the simplest and lets the surprise gradient be η_t = 1 at t=0.)
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
Tensor TitansMAC::forward(const Tensor& input) {
    const size_t T = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("TitansMAC: input.cols must equal d_model");

    // 1) Joint projection: qkv = x · W_qkv^T + b_qkv
    Tensor qkv = W_qkv_.forward(input);   // (T, 3*d_model)
    Tensor q_t(T, d_model_);
    Tensor k_t(T, d_model_);
    Tensor v_t(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            q_t(t, j) = qkv(t, j);                  // cols [0, d)
            k_t(t, j) = qkv(t, d_model_ + j);       // cols [d, 2d)
            v_t(t, j) = qkv(t, 2 * d_model_ + j);   // cols [2d, 3d)
        }
    }

    // Allocate cache
    Tensor v_norm(T, 1);
    Tensor eta(T, 1);
    Tensor alpha(T, 1);
    Tensor M_t((T + 1) * d_model_, d_model_);  // last_M_t flat layout (T+1, d, d)
    // last_M_t[0] = initial M (for reference). M_t[t*d_model + i, j] = M after token t.
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            M_t(i, j) = M_(i, j);   // M_{0} = initial

    // 2-5) Per-token recurrence
    Tensor y(T, d_model_);
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
        Tensor alpha_pre = W_alpha_.forward(alpha_in);   // (1, 1)
        double sigma_alpha = 1.0 / (1.0 + std::exp(-alpha_pre(0, 0)));

        // surprise η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
        // Compute M_{t-1} · k_t: prev_M is M_t[(t)*d:(t+1)*d, :], so its rows are the d×d matrix.
        Tensor mk(d_model_, 1);
        double xnorm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t(t * d_model_ + i, j) * k_t(t, j);
            mk(i, 0) = s;
            xnorm += input(t, i) * input(t, i);
        }
        xnorm = std::sqrt(xnorm);
        double diff_norm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double d_i = input(t, i) - mk(i, 0);
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

        // y_t = M_t · q_t  (use the freshly updated M_t)
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t((t + 1) * d_model_ + i, j) * q_t(t, j);
            y(t, i) = s;
        }
    }

    // Cache for backward. Store as private fields via direct attribute access —
    // we use a stash of named tensors to avoid growing the header for now.
    // (See below for how they're reconstructed in backward via last_input_)
    last_input_ = input.clone();
    // Stash via heap-allocated caches stored in a static map? No — simpler:
    // attach as fields directly to the class via this minimal stash.
    last_q_t_ = q_t;
    last_k_t_ = k_t;
    last_v_t_ = v_t;
    last_v_norm_ = v_norm;
    last_eta_ = eta;
    last_alpha_ = alpha;
    last_M_t_ = M_t;
    last_y_t_ = y.clone();
    last_qkv_ = qkv;
    return y;
}

// ============================================================================
// Backward
// ============================================================================
Tensor TitansMAC::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    const size_t T = last_input_.rows;

    // Allocate per-token local gradient accumulators.
    // Per-token gradient accumulators (intermediate, freed at end).
    Tensor dq_t(T, d_model_);
    Tensor dk_t(T, d_model_);
    Tensor dv_t(T, d_model_);
    Tensor dx_t(T, d_model_);
    Tensor dalphain_t(T, d_model_ + 1);  // dL/d(W_alpha·[x;||v||] + b_alpha) — pre-sigmoid
    std::fill(dalphain_t.data.begin(), dalphain_t.data.end(), 0.0);
    std::fill(dv_t.data.begin(), dv_t.data.end(), 0.0);
    std::fill(dk_t.data.begin(), dk_t.data.end(), 0.0);
    std::fill(dx_t.data.begin(), dx_t.data.end(), 0.0);

    // Step 1: dy_t → dq_t (the dM_t path is computed inside the main backward loop below).
    //   y_t[i] = sum_j M_t[i,j] · q_t[j]  ⇒  dL/dq_t[j] = sum_i M_t[i,j] · dL/dy_t[i]
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d_model_; ++i) {
            double dy = grad_output(t, i);
            for (size_t j = 0; j < d_model_; ++j) {
                dq_t(t, j) += last_M_t_((t + 1) * d_model_ + i, j) * dy;
            }
        }
    }

    // Process tokens from t = T-1 down to t = 0. We maintain:
    //   dM_carrier[i, j] = accumulated dL/dM_t[i, j] from the NEXT token's update (M_{t+1}
    //     depends on M_t via the (1-α) factor). Initially zero.
    //   dM_carrier is the contribution that flows backward from later tokens.
    Tensor dM_carrier(d_model_, d_model_);
    std::fill(dM_carrier.data.begin(), dM_carrier.data.end(), 0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        double a = last_alpha_(t, 0);
        double om_a = 1.0 - a;
        double eta_t = last_eta_(t, 0);

        // Total dL/dM_t (post-update) = dL/dy_t (via y_t = M_t · q_t) + dM_carrier
        // [the carrier is the dL/dM_t contribution from M_{t+1} = (1-α_{t+1})·M_t + ...]
        // but for token T-1, carrier is 0 (no next token).
        Tensor dM_post(d_model_, d_model_);
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dM_post(i, j) = grad_output(t, i) * last_q_t_(t, j) + dM_carrier(i, j);
            }
        }

        // dL/dα_t splits into two paths:
        //   (1) α_t contributes to M_t[i,j] as:  v_t[i]·k_t[j] - M_{t-1}[i,j]
        //   (2) α_t = σ_pre · η_t  →  dL/dσ_pre += dalpha · η_t  (handled via dz_pre below)
        //                          →  dL/dη_t   += dalpha · σ_pre  (handled via deta below)
        // dalpha below is path (1) only — paths (2) are accumulated later as deta and dz_pre.
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

        // ALSO: dL/dM_{t-1} from the surprise chain (η_t depends on M_{t-1}).
        //   Surprise: η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
        //   let mk = M_{t-1} · k_t    (d × 1)
        //   let diff = x_t - mk        (d × 1)
        //   diff_norm = ||diff||_2,  eta_t = diff_norm / (||x_t||_2 + eps)
        //
        //   dη_t/dx_t[i] = (diff[i]/N_d - η_t·x_t[i]/N_x) / (N_x + eps)
        //   dη_t/dk_t[j] = -(1/(N_x+eps))·(1/N_d)·(M_{t-1}^T · diff)[j]
        //   dη_t/dM_{t-1}[i,j] = -(1/(N_x+eps))·(1/N_d)·diff[i]·k_t[j]
        //
        // dL/dη_t arrives from:  α_t = σ_pre · η_t  (where σ_pre = sigmoid(W_α·[x;||v||])).
        // So dα_t/dη_t = σ_pre, giving dL/dη_t += σ_pre · dL/dα_t.
        // dα_t/dσ_pre = η_t, giving dL/dz_pre += η_t · σ_pre'(z_pre) · dL/dα_t
        //   (handled below as dz_pre = dalpha · η_t · σ_pre_deriv).
        //
        // Recompute the surprise inputs and σ_pre (they're cheap scalars).
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

        // z_pre = W_alpha · [x_t; ||v_t||] + b_alpha  (re-derived for the dz_pre path).
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = last_input_(t, j);
        alpha_in(0, d_model_) = last_v_norm_(t, 0);
        double z_pre = W_alpha_.bias(0, 0);
        for (size_t j = 0; j < d_model_ + 1; ++j) z_pre += W_alpha_.weights(0, j) * alpha_in(0, j);
        double sigma_pre = 1.0 / (1.0 + std::exp(-z_pre));
        double sigma_pre_deriv = sigma_pre * (1.0 - sigma_pre);
        double deta = dalpha * sigma_pre;
        double dz_pre = dalpha * eta_t * sigma_pre_deriv;

        // dL/dz_pre → dL/dW_alpha.weights(0, j) += dz_pre · alpha_in(0, j)
        //          → dL/dW_alpha.bias(0, 0) += dz_pre
        //          → dL/dalpha_in(0, j) += dz_pre · W_alpha.weights(0, j)
        for (size_t j = 0; j < d_model_ + 1; ++j) {
            grad_W_alpha_w_(0, j) += dz_pre * alpha_in(0, j);
            dalphain_t(t, j) += dz_pre * W_alpha_.weights(0, j);
        }
        grad_W_alpha_b_(0, 0) += dz_pre;

        // dL/dα_in(t, j) splits:
        //   alpha_in = concat[x_t, ||v_t||]
        //   For j in [0, d_model): dα_in(j)/dx_t[j] = 1 → dL/dx_t[j] += dalphain_t(j)
        //   For j = d_model: dα_in(j)/d(||v_t||) = 1 → dL/d||v_t|| += dalphain_t(d_model)
        //     and ||v_t|| has its own gradient flow: d||v_t||/dv_t[i] = v_t[i] / ||v_t||
        for (size_t j = 0; j < d_model_; ++j) {
            dx_t(t, j) += dalphain_t(t, j);
        }
        if (last_v_norm_(t, 0) > eps) {
            double dvnorm = dalphain_t(t, d_model_);
            for (size_t i = 0; i < d_model_; ++i) {
                dv_t(t, i) += dvnorm * last_v_t_(t, i) / last_v_norm_(t, 0);
            }
        }

        // Now the surprise chain: dL/dη_t = deta. Routes to dL/dx_t, dL/dk_t, dL/dM_{t-1}.
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
        // dL/dW_qkv.bias[2d + i] += dL/dv_t[t, i]   (col 2d + i of qkv)
        // dL/dW_qkv.bias[d + j]  += dL/dk_t[t, j]   (col d + j of qkv)
        // dL/dW_qkv.bias[j]      += dL/dq_t[t, j]   (col j of qkv)
        // (We already computed dq_t in step 1.)
        for (size_t i = 0; i < d_model_; ++i) {
            grad_W_qkv_b_(0, i) += dq_t(t, i);
            grad_W_qkv_b_(0, d_model_ + i) += dk_t(t, i);
            grad_W_qkv_b_(0, 2 * d_model_ + i) += dv_t(t, i);
        }

        // dL/dW_qkv.weights[j, k] += dL/dqkv[t, j] · x_t[k]
        // Similarly for k and v columns.
        for (size_t j = 0; j < 3 * d_model_; ++j) {
            double dqkv_tj = (j < d_model_) ? dq_t(t, j)
                            : (j < 2 * d_model_) ? dk_t(t, j - d_model_)
                                                 : dv_t(t, j - 2 * d_model_);
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_qkv_w_(j, k) += dqkv_tj * last_input_(t, k);
            }
        }

        // dL/dM_ (persistent memory) — for t=0, the initial M's gradient is dM_new_carrier
        // (which captures dL/dM_0 = dL/dM_{t-1} for t=0). For t > 0, the carrier is
        // dL/dM_{t-1} and feeds into the dM_post for the previous iteration.
        // Update dM_carrier for the NEXT iteration (token t-1).
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                dM_carrier(i, j) = dM_new_carrier(i, j);
            }
        }
    }

    // After the loop, dM_carrier holds dL/dM_0 = dL/dM_ (the persistent memory).
    // (The iteration t=0's dM_new_carrier captures dL/dM_{-1} = dL/dM_.)
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_M_(i, j) = dM_carrier(i, j);
        }
    }

    // The input gradient dL/dx_t also includes dL/d(W_qkv · x_t) — i.e., the projection
    // through W_qkv. The chain through q, k, v is:
    //   x_t → qkv_t = W_qkv · x_t  (linear, so dqkv_t/dx_t[k] = W_qkv.weights[j, k])
    //   dL/dx_t[k] += sum_j (dL/dqkv_t[j]) · W_qkv.weights[j, k]
    // We have dL/dqkv_t[j] = dq_t (j<d), dk_t (d ≤ j < 2d), dv_t (2d ≤ j < 3d).
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < 3 * d_model_; ++j) {
                double dqkv_tj = (j < d_model_) ? dq_t(t, j)
                                : (j < 2 * d_model_) ? dk_t(t, j - d_model_)
                                                     : dv_t(t, j - 2 * d_model_);
                s += dqkv_tj * W_qkv_.weights(j, k);
            }
            dx_t(t, k) += s;
        }
    }

    return dx_t;
}

// ============================================================================
// Optimizer interface
// ============================================================================
void TitansMAC::update_weights(double learning_rate) {
    // W_qkv: standard dense
    W_qkv_.weights = W_qkv_.weights - grad_W_qkv_w_ * learning_rate;
    W_qkv_.bias   = W_qkv_.bias   - grad_W_qkv_b_ * learning_rate;
    // W_alpha: standard dense
    W_alpha_.weights = W_alpha_.weights - grad_W_alpha_w_ * learning_rate;
    W_alpha_.bias   = W_alpha_.bias   - grad_W_alpha_b_ * learning_rate;
    // M (persistent memory): same SGD-style update
    M_ = M_ - grad_M_ * learning_rate;
}

void TitansMAC::zero_grad() {
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

std::vector<Tensor*> TitansMAC::parameters() {
    return { &W_qkv_.weights, &W_qkv_.bias,
             &W_alpha_.weights, &W_alpha_.bias,
             &M_ };
}

std::vector<Tensor*> TitansMAC::gradients() {
    return { &grad_W_qkv_w_, &grad_W_qkv_b_,
             &grad_W_alpha_w_, &grad_W_alpha_b_,
             &grad_M_ };
}

void TitansMAC::copy_params_from(const TitansMAC& other) {
    W_qkv_.weights = other.W_qkv_.weights.clone();
    W_qkv_.bias = other.W_qkv_.bias.clone();
    W_alpha_.weights = other.W_alpha_.weights.clone();
    W_alpha_.bias = other.W_alpha_.bias.clone();
    M_ = other.M_.clone();
}
