// ============================================================================
// FastFormer implementation
// ============================================================================
#include "fastformer.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

// ----------------------------------------------------------------------------
// row_softmax — row-wise softmax with max-subtraction trick.
// ----------------------------------------------------------------------------
inline Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j) {
            if (x[i][j] > row_max) row_max = x[i][j];
        }
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j) {
            result[i][j] *= inv;
        }
    }
    return result;
}

// ----------------------------------------------------------------------------
// row_mean — average each column across rows (returns (1, cols)).
// ----------------------------------------------------------------------------
inline Tensor row_mean(const Tensor& x) {
    Tensor out(1, x.cols);
    for (size_t j = 0; j < x.cols; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < x.rows; ++i) s += x[i][j];
        out[0][j] = s / (double)x.rows;
    }
    return out;
}

// ----------------------------------------------------------------------------
// sigmoid (in-place helper returning a fresh tensor)
// ----------------------------------------------------------------------------
inline Tensor sigmoid(const Tensor& x) {
    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            out[i][j] = 1.0 / (1.0 + std::exp(-x[i][j]));
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// GELU exact: 0.5 * x * (1 + erf(x / sqrt(2))).
// ----------------------------------------------------------------------------
inline Tensor gelu(const Tensor& x) {
    Tensor out(x.rows, x.cols);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            out[i][j] = 0.5 * x[i][j] * (1.0 + std::erf(x[i][j] * inv_sqrt2));
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// matmul C = A @ B where A:(m,k), B:(k,n) → C:(m,n)
// ----------------------------------------------------------------------------
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("matmul: dim mismatch");
    }
    Tensor C(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < A.cols; ++k) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    }
    return C;
}

// ----------------------------------------------------------------------------
// broadcast_add(A, b_row) where A:(m,n), b_row:(1,n) — adds b_row to each row.
// ----------------------------------------------------------------------------
inline Tensor row_bias_add(const Tensor& A, const Tensor& b) {
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < A.cols; ++j)
            C[i][j] = A[i][j] + b[0][j];
    return C;
}

} // namespace


// ============================================================================
// FastFormerAttention
// ============================================================================
FastFormerAttention::FastFormerAttention(size_t d_model, size_t num_heads, bool causal) {
    if (d_model == 0) throw std::invalid_argument("FastFormer: d_model must be > 0");
    if (num_heads == 0) throw std::invalid_argument("FastFormer: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("FastFormer: d_model must be divisible by num_heads");

    d_model_ = d_model;
    num_heads_ = num_heads;
    head_dim_ = d_model / num_heads;
    scale_ = 1.0 / std::sqrt((double)head_dim_);
    causal_ = causal;

    auto xr = [&](size_t r, size_t c) { return Tensor::random(r, c, 0.05); };
    auto zr = [&](size_t r, size_t c) { return Tensor::zeros(r, c); };

    W_q_ = xr(d_model_, d_model_);     b_q_ = zr(1, d_model_);
    W_k_ = xr(d_model_, d_model_);     b_k_ = zr(1, d_model_);
    W_v_ = xr(d_model_, d_model_);     b_v_ = zr(1, d_model_);
    W_p_ = xr(d_model_, 2 * d_model_); b_p_ = zr(1, d_model_);
    W_b_ = xr(d_model_, 2 * d_model_); b_b_ = zr(1, d_model_);
    W_gamma_ = xr(1, d_model_);        b_gamma_ = zr(1, 1);
    W_o_ = xr(d_model_, d_model_);     b_o_ = zr(1, d_model_);

    grad_W_q_ = zr(d_model_, d_model_);     grad_b_q_ = zr(1, d_model_);
    grad_W_k_ = zr(d_model_, d_model_);     grad_b_k_ = zr(1, d_model_);
    grad_W_v_ = zr(d_model_, d_model_);     grad_b_v_ = zr(1, d_model_);
    grad_W_p_ = zr(d_model_, 2 * d_model_); grad_b_p_ = zr(1, d_model_);
    grad_W_b_ = zr(d_model_, 2 * d_model_); grad_b_b_ = zr(1, d_model_);
    grad_W_gamma_ = zr(1, d_model_);       grad_b_gamma_ = zr(1, 1);
    grad_W_o_ = zr(d_model_, d_model_);     grad_b_o_ = zr(1, d_model_);
}


// ----------------------------------------------------------------------------
// Forward — multi-head: gating is SHARED across the full d_model, then per-head
// softmax attention is computed on per-head slices. Final output is W_o projection.
// (For v1, backward only supports num_heads=1; the forward path supports any H.)
// ----------------------------------------------------------------------------
Tensor FastFormerAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("FastFormer: input.cols must equal d_model");
    size_t n = input.rows;
    last_input_ = input;

    Tensor X = input;

    // QKV projections (shared across heads).
    Tensor Qpre = matmul(X, W_q_.transpose());          // (n, d_model_)
    Qpre = row_bias_add(Qpre, b_q_);
    Tensor Kpre = matmul(X, W_k_.transpose());
    Kpre = row_bias_add(Kpre, b_k_);
    Tensor Vpre = matmul(X, W_v_.transpose());
    Vpre = row_bias_add(Vpre, b_v_);

    // ===== Shared (full d_model) gating =====

    // 1. q_global = mean(Qpre over rows)  → (1, d_model_)
    Tensor q_global = row_mean(Qpre);

    // 2. p = sigmoid(W_p · [Q_i ; q_global] + b_p)  for each token i → (n, d_model_)
    //    Build [Q_i ; q_global] of shape (n, 2*d_model_).
    Tensor Qcat(n, 2 * d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            Qcat[i][j] = Qpre[i][j];
            Qcat[i][d_model_ + j] = q_global[0][j];
        }
    }
    Tensor p_logits = row_bias_add(matmul(Qcat, W_p_.transpose()), b_p_);
    Tensor p = sigmoid(p_logits);  // (n, d_model_)

    // K_mod = p ⊙ Kpre
    Tensor K_mod(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            K_mod[i][j] = p[i][j] * Kpre[i][j];

    // k_global = mean(K_mod over rows)
    Tensor k_global = row_mean(K_mod);

    // b_gate = sigmoid(W_b · [K_i ; k_global] + b_b)
    Tensor Kcat(n, 2 * d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            Kcat[i][j] = Kpre[i][j];
            Kcat[i][d_model_ + j] = k_global[0][j];
        }
    }
    Tensor b_logits = row_bias_add(matmul(Kcat, W_b_.transpose()), b_b_);
    Tensor b_gate = sigmoid(b_logits);  // (n, d_model_)

    // Q_mod = b_gate ⊙ Qpre
    Tensor Q_mod(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            Q_mod[i][j] = b_gate[i][j] * Qpre[i][j];

    // ===== Per-head softmax attention on per-head slices =====
    Tensor Y(n, d_model_);

    // Per-head caches (only valid for head 0; for num_heads=1 we use them directly).
    // For num_heads > 1 the backward of head-0 chains is correct in aggregate since
    // the gating is shared and the per-head contributions to dW_q/k/v are summed.
    // Note: backward is restricted to num_heads=1 in v1 (see backward() guard).
    last_q_ = Qpre;
    last_k_ = Kpre;
    last_v_ = Vpre;
    last_q_global_ = q_global;
    last_p_ = p;
    last_k_mod_ = K_mod;
    last_k_global_ = k_global;
    last_b_ = b_gate;
    last_q_mod_ = Q_mod;

    for (size_t h = 0; h < num_heads_; ++h) {
        size_t h0 = h * head_dim_;
        // Per-head slice of (n, head_dim_)
        Tensor Q_h(n, head_dim_), K_h(n, head_dim_), V_h(n, head_dim_);
        Tensor Qmod_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                Q_h[i][j]    = Qpre[i][h0 + j];
                K_h[i][j]    = Kpre[i][h0 + j];
                V_h[i][j]    = Vpre[i][h0 + j];
                Qmod_h[i][j] = Q_mod[i][h0 + j];
            }
        }

        // α_i = softmax(Q_mod · K^T / sqrt(d)) → (n, n)
        Tensor scores = matmul(Qmod_h, K_h.transpose());
        // Apply scaling
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                scores[i][j] *= scale_;
        // Causal mask
        if (causal_) {
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j)
                    scores[i][j] = -1e30;
        }
        Tensor alpha = row_softmax(scores);

        // c = α · V → (n, head_dim_)
        Tensor c = matmul(alpha, V_h);

        // γ = sigmoid(W_gamma · c + b_gamma)  → per-token scalar
        // W_gamma shape: (1, d_model_); for per-head we use the head-slice columns:
        //   γ_i = sigmoid( sum_{j ∈ head_h} W_gamma[0][h0+j] * c_i[j] + b_gamma )
        Tensor gamma(n, 1);
        for (size_t i = 0; i < n; ++i) {
            double s = b_gamma_[0][0];
            for (size_t j = 0; j < head_dim_; ++j) s += W_gamma_[0][h0 + j] * c[i][j];
            gamma[i][0] = 1.0 / (1.0 + std::exp(-s));
        }

        // out_h = (b_gate slice) ⊙ V_h + γ ⊙ c
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double gate = b_gate[i][h0 + j];
                Y[i][h0 + j] = gate * V_h[i][j] + gamma[i][0] * c[i][j];
            }
        }

        // Cache head-0's full per-token tensors for backward.
        if (h == 0) {
            last_alpha_ = alpha;
            last_c_ = c;
            last_gamma_ = gamma;
        }
    }

    // Output projection W_o: Y (n, d_model) @ W_o^T (d_model, d_model) + b_o
    Tensor output = row_bias_add(matmul(Y, W_o_.transpose()), b_o_);
    last_y_pre_ = Y;
    last_output_ = output;
    return output;
}


// ----------------------------------------------------------------------------
// Backward — full chain rule.
// For v1 we implement single-head path only (num_heads_ must be 1 for backward correctness).
// Multi-head gradient flow will be approximated by the test config setting num_heads=1.
// ----------------------------------------------------------------------------
Tensor FastFormerAttention::backward(const Tensor& grad_output, double /*lr*/) {
    if (num_heads_ != 1)
        throw std::logic_error("FastFormerAttention v1 backward requires num_heads=1");

    size_t n = last_input_.rows;
    size_t d = d_model_;

    // d_out → dY (input of W_o)
    // Y_pre = Y (after concat of heads); output = Y_pre @ W_o^T + b_o
    // output[i][k] = sum_j Y[i][j] * W_o[k][j] + b_o[k]
    // d/d(W_o[k][j]) = sum_i grad_output[i][k] * Y[i][j]
    //   → grad_W_o[k][j] = sum_i grad_output[i][k] * Y_pre[i][j]
    // grad_Y_pre[i][j] = sum_k grad_output[i][k] * W_o[k][j] = (grad_output @ W_o)[i][j]
    Tensor dYpre = matmul(grad_output, W_o_);  // (n, d) @ (d, d) = (n, d)
    const Tensor& Y_pre = last_y_pre_;
    for (size_t k = 0; k < d; ++k)
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += grad_output[i][k] * Y_pre[i][j];
            grad_W_o_[k][j] += s;
        }
    // grad_b_o[k] = sum_i grad_output[i][k]
    for (size_t k = 0; k < d; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += grad_output[i][k];
        grad_b_o_[0][k] += s;
    }

    // dYpre is the gradient w.r.t. the per-head concatenated Y_pre (the input of W_o).
    // Since num_heads=1, this is also the head-0 input gradient.
    Tensor& grad_head = dYpre;  // alias for clarity

    // Unpack caches for head 0.
    const Tensor& Q = last_q_;       // (n, d)
    const Tensor& K = last_k_;       // (n, d)
    const Tensor& V = last_v_;       // (n, d)
    const Tensor& Q_mod = last_q_mod_;
    const Tensor& b_gate = last_b_;   // (n, d)
    const Tensor& p_gate = last_p_;   // (n, d)
    const Tensor& alpha = last_alpha_;
    const Tensor& c = last_c_;
    const Tensor& gamma = last_gamma_;
    const Tensor& q_global = last_q_global_;
    const Tensor& k_global = last_k_global_;

    // d(b_gate ⊙ V) path: out_h = b_gate ⊙ V + γ ⊙ c
    // dV_path1 = b_gate ⊙ grad_head  (shape (n, d))
    // db_gate_path1 = V ⊙ grad_head
    Tensor dV(n, d);
    dV.fill(0.0);
    Tensor db_gate(n, d);
    db_gate.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            dV[i][j]      += b_gate[i][j] * grad_head[i][j];
            db_gate[i][j] += V[i][j]      * grad_head[i][j];
        }
    }

    // d(γ ⊙ c) path: γ_i = σ(w_gamma · c_i + b_gamma); dγ_i = sum_j grad_head[i][j] * c[i][j]
    // dlogit_gamma = γ_i * (1 - γ_i) * dγ_i
    // dc += γ_i * grad_head[i, :]
    // dc[i] += w_gamma^T * dlogit_gamma[i]    (broadcast across features)
    // W_gamma update: grad_W_gamma[0][j] = Σ_i dlogit_gamma[i] * c[i][j]
    // b_gamma update: grad_b_gamma[0][0] = Σ_i dlogit_gamma[i]
    Tensor dc(n, d);
    dc.fill(0.0);
    Tensor dlogit_gamma(n, 1);
    dlogit_gamma.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        double dgamma = 0.0;
        for (size_t j = 0; j < d; ++j) dgamma += grad_head[i][j] * c[i][j];
        double gv = gamma[i][0];
        dlogit_gamma[i][0] = gv * (1.0 - gv) * dgamma;
        for (size_t j = 0; j < d; ++j) dc[i][j] += gv * grad_head[i][j];
        // dc += w_gamma^T · dlogit_gamma[i]   =   w_gamma[0][j] * dlogit_gamma[i][0]
        for (size_t j = 0; j < d; ++j) dc[i][j] += W_gamma_[0][j] * dlogit_gamma[i][0];
    }
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dlogit_gamma[i][0] * c[i][j];
        grad_W_gamma_[0][j] += s;
    }
    for (size_t i = 0; i < n; ++i) grad_b_gamma_[0][0] += dlogit_gamma[i][0];

    // Back-prop through c = α · V
    // dV += α^T · dc        (V is shared; sum over the (n) row dim of alpha)
    // dα = dc · V^T         (per row of α)
    Tensor dalpha(n, n);
    dalpha.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t s_ = 0; s_ < n; ++s_) s += alpha[s_][i] * dc[s_][j];
            dV[i][j] += s;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += dc[i][k] * V[j][k];
            dalpha[i][j] = s;
        }
    }

    // Back-prop through softmax(α):
    // dscores[i][j] = α[i][j] * (dalpha[i][j] - Σ_k α[i][k] * dalpha[i][k])
    Tensor dscores(n, n);
    for (size_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (size_t k = 0; k < n; ++k) dot += alpha[i][k] * dalpha[i][k];
        for (size_t j = 0; j < n; ++j) {
            dscores[i][j] = alpha[i][j] * (dalpha[i][j] - dot);
        }
    }

    // Causal mask: positions where alpha was set to 0 should not propagate gradient.
    // The mask was applied pre-softmax by setting score=-1e30; those positions have
    // alpha ≈ 0 (after softmax). Forward itself was unaffected, but their dscores
    // contribution should be zero. We zero them explicitly to match the mask:
    if (causal_) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                dscores[i][j] = 0.0;
    }

    // scores was scaled by scale_ before softmax; we need to scale back.
    // scores_actual[i][j] = (Q_mod · K^T)[i][j] * scale_
    // dscores_unscaled = dscores * scale_
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            dscores[i][j] *= scale_;

    // Back-prop through Q_mod · K^T
    // dQ_mod[i][j] = Σ_k dscores[i][k] * K[k][j]
    // dK[i][j]    = Σ_k dscores[k][i] * Q_mod[k][j]
    Tensor dQ_mod(n, d);
    dQ_mod.fill(0.0);
    Tensor dK_from_scores(n, d);
    dK_from_scores.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s_q = 0.0;
            double s_k = 0.0;
            for (size_t k = 0; k < n; ++k) {
                s_q += dscores[i][k] * K[k][j];
                s_k += dscores[k][i] * Q_mod[k][j];
            }
            dQ_mod[i][j] = s_q;
            dK_from_scores[i][j] = s_k;
        }
    }

    // Back-prop through Q_mod = b_gate ⊙ Q → dQ += b_gate ⊙ dQ_mod, db_gate += Q ⊙ dQ_mod
    Tensor dQ(n, d);
    dQ.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            dQ[i][j]      += b_gate[i][j] * dQ_mod[i][j];
            db_gate[i][j] += Q[i][j]      * dQ_mod[i][j];
        }
    }

    // Back-prop through db_gate from the b_gate-as-σ-output path: b_gate is the
    // sigmoid output of W_b · [K_i ; k_global] + b_b.
    // b_gate = sigmoid(z_b)  →  dz_b = b_gate * (1 - b_gate) * db_gate
    // Then dK_i gets a contribution: dK_i += Σ_paths dK_i_path
    // The chain is: dz_b → [K_i; k_global] → K_i (and also to k_global).
    // First compute dK_from_db_gate: dK_i_feature_j += (W_b row j, col j_in_K) * dz_b[i]
    // The first d columns of W_b's input correspond to K_i, the last d to k_global.
    Tensor dz_b(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double v = b_gate[i][j];
            dz_b[i][j] = v * (1.0 - v) * db_gate[i][j];
        }
    }
    // dK_i += W_b[*, j_in_K] · dz_b[i]    (W_b: (d, 2d), first d cols are K_i)
    Tensor dK(n, d);
    dK.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t r = 0; r < d; ++r) s += W_b_[r][j] * dz_b[i][r];
            dK[i][j] += s;
        }
    }
    // dK_global: sum of W_b[*, j_in_k] · dz_b[i] over all i (k_global is shared)
    Tensor dk_global(1, d);
    dk_global.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t r = 0; r < d; ++r) s += W_b_[r][d + j] * dz_b[i][r];
            dk_global[0][j] += s;
        }
    }
    // Back-prop into K from scores: dK += dK_from_scores
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) dK[i][j] += dK_from_scores[i][j];

    // grad_W_b:  grad_W_b[r][j_in_input] = Σ_i dz_b[i][r] * input[i][j_in_input]
    // input = [K_i ; k_global]
    for (size_t r = 0; r < d; ++r) {
        for (size_t j_in = 0; j_in < 2 * d; ++j_in) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double inp_ij = (j_in < d) ? K[i][j_in] : k_global[0][j_in - d];
                s += dz_b[i][r] * inp_ij;
            }
            grad_W_b_[r][j_in] += s;
        }
    }
    // grad_b_b: grad_b_b[r] = Σ_i dz_b[i][r]
    for (size_t r = 0; r < d; ++r) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dz_b[i][r];
        grad_b_b_[0][r] += s;
    }

    // Now back-prop through K_mod = p_gate ⊙ K
    // dK_mod ← dk_global (broadcast: each row of K_mod gets dk_global / n)
    //   (the p_gate ⊙ K direction is added below as dK += p_gate ⊙ dK_mod)
    Tensor dK_mod(n, d);
    dK_mod.fill(0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            dK_mod[i][j] = dk_global[0][j] / (double)n;
        }

    // Back-prop through K_mod = p ⊙ K: dp_gate += K ⊙ dK_mod, dK += p ⊙ dK_mod
    Tensor dp_gate(n, d);
    dp_gate.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            dp_gate[i][j] += K[i][j] * dK_mod[i][j];
            dK[i][j]      += p_gate[i][j] * dK_mod[i][j];
        }
    }

    // Back-prop through p = sigmoid(z_p):
    // z_p = W_p · [Q_i ; q_global] + b_p
    // dz_p = p * (1 - p) * dp_gate
    Tensor dz_p(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double pv = p_gate[i][j];
            dz_p[i][j] = pv * (1.0 - pv) * dp_gate[i][j];
        }

    // grad_W_p[r][j_in_input] = Σ_i dz_p[i][r] * input[i][j_in_input]
    // input = [Q_i ; q_global]
    for (size_t r = 0; r < d; ++r) {
        for (size_t j_in = 0; j_in < 2 * d; ++j_in) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double inp_ij = (j_in < d) ? Q[i][j_in] : q_global[0][j_in - d];
                s += dz_p[i][r] * inp_ij;
            }
            grad_W_p_[r][j_in] += s;
        }
    }
    for (size_t r = 0; r < d; ++r) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dz_p[i][r];
        grad_b_p_[0][r] += s;
    }

    // Back-prop into Q from p = sigmoid(W_p · [Q; q_global]): q_global receives Σ_i dz_p · W_p[:, d:].
    // q_global = mean(Q) — but q_global here is the COMPUTED mean, not a free parameter. The
    // gradient flows back into each Q_i via the mean: dQ_i += dq_global / n (broadcast).
    Tensor dq_global(1, d);
    dq_global.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t r = 0; r < d; ++r) s += W_p_[r][d + j] * dz_p[i][r];
            dq_global[0][j] += s;
        }
    }
    // dQ_i += dq_global / n (broadcast from q_global mean), then also += W_p[:, :d] · dz_p[i]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double from_qmean = dq_global[0][j] / (double)n;
            double s = 0.0;
            for (size_t r = 0; r < d; ++r) s += W_p_[r][j] * dz_p[i][r];
            dQ[i][j] += from_qmean + s;
        }
    }

    // Now back-prop dQ into QKV projections. dQ, dK, dV are gradients w.r.t. the projected
    // values. The projections were Q = X @ W_q^T + b_q, etc.
    auto linear_backward = [&](const Tensor& dZ,
                               const Tensor& W,
                               Tensor& grad_W,
                               Tensor& grad_b,
                               Tensor& dX_out) {
        // dX = dZ · W   (W is (out, in), so dX is (n, in))
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < W.cols; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < W.rows; ++k) s += dZ[i][k] * W[k][j];
                dX_out[i][j] += s;
            }
        }
        // grad_W[k][j] = Σ_i dZ[i][k] * X[i][j]
        for (size_t k = 0; k < W.rows; ++k) {
            for (size_t j = 0; j < W.cols; ++j) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i) s += dZ[i][k] * last_input_[i][j];
                grad_W[k][j] += s;
            }
        }
        // grad_b[k] = Σ_i dZ[i][k]
        for (size_t k = 0; k < W.rows; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += dZ[i][k];
            grad_b[0][k] += s;
        }
    };

    Tensor dX(n, d);
    dX.fill(0.0);
    linear_backward(dQ, W_q_, grad_W_q_, grad_b_q_, dX);
    Tensor dK_input(n, d);
    dK_input.fill(0.0);
    linear_backward(dK, W_k_, grad_W_k_, grad_b_k_, dK_input);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) dX[i][j] += dK_input[i][j];
    Tensor dV_input(n, d);
    dV_input.fill(0.0);
    linear_backward(dV, W_v_, grad_W_v_, grad_b_v_, dV_input);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) dX[i][j] += dV_input[i][j];

    return dX;
}


void FastFormerAttention::update_weights(double lr) {
    auto sgd = [&](Tensor& W, Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] -= lr * gW[i][j];
    };
    auto sgdb = [&](Tensor& b, Tensor& gb) {
        for (size_t j = 0; j < b.cols; ++j) b[0][j] -= lr * gb[0][j];
    };
    sgd(W_q_, grad_W_q_); sgdb(b_q_, grad_b_q_);
    sgd(W_k_, grad_W_k_); sgdb(b_k_, grad_b_k_);
    sgd(W_v_, grad_W_v_); sgdb(b_v_, grad_b_v_);
    sgd(W_p_, grad_W_p_); sgdb(b_p_, grad_b_p_);
    sgd(W_b_, grad_W_b_); sgdb(b_b_, grad_b_b_);
    sgd(W_gamma_, grad_W_gamma_);
    b_gamma_[0][0] -= lr * grad_b_gamma_[0][0];
    sgd(W_o_, grad_W_o_); sgdb(b_o_, grad_b_o_);
}

void FastFormerAttention::zero_grad() {
    auto z = [&](Tensor& t) { t.fill(0.0); };
    z(grad_W_q_); z(grad_b_q_);
    z(grad_W_k_); z(grad_b_k_);
    z(grad_W_v_); z(grad_b_v_);
    z(grad_W_p_); z(grad_b_p_);
    z(grad_W_b_); z(grad_b_b_);
    z(grad_W_gamma_); z(grad_b_gamma_);
    z(grad_W_o_); z(grad_b_o_);
}

std::vector<Tensor*> FastFormerAttention::parameters() {
    return {&W_q_, &b_q_, &W_k_, &b_k_, &W_v_, &b_v_,
            &W_p_, &b_p_, &W_b_, &b_b_, &W_gamma_, &b_gamma_,
            &W_o_, &b_o_};
}

std::vector<Tensor*> FastFormerAttention::gradients() {
    return {&grad_W_q_, &grad_b_q_, &grad_W_k_, &grad_b_k_, &grad_W_v_, &grad_b_v_,
            &grad_W_p_, &grad_b_p_, &grad_W_b_, &grad_b_b_, &grad_W_gamma_, &grad_b_gamma_,
            &grad_W_o_, &grad_b_o_};
}


// ============================================================================
// FastFormerBlock
// ============================================================================
FastFormerBlock::FastFormerBlock(size_t d_model, size_t num_heads,
                                 size_t ffn_hidden, bool causal)
    : d_model_(d_model),
      ffn_hidden_(ffn_hidden),
      has_ffn_(ffn_hidden > 0),
      attn_(d_model, num_heads, causal),
      ln1_(d_model),
      ln2_(d_model) {
    if (has_ffn_) {
        W1_ = Tensor::random(ffn_hidden, d_model, 0.05);
        b1_ = Tensor::zeros(1, ffn_hidden);
        W2_ = Tensor::random(d_model, ffn_hidden, 0.05);
        b2_ = Tensor::zeros(1, d_model);
        grad_W1_ = Tensor::zeros(ffn_hidden, d_model);
        grad_b1_ = Tensor::zeros(1, ffn_hidden);
        grad_W2_ = Tensor::zeros(d_model, ffn_hidden);
        grad_b2_ = Tensor::zeros(1, d_model);
    } else {
        W1_ = Tensor::zeros(0, 0); b1_ = Tensor::zeros(0, 0);
        W2_ = Tensor::zeros(0, 0); b2_ = Tensor::zeros(0, 0);
        grad_W1_ = Tensor::zeros(0, 0); grad_b1_ = Tensor::zeros(0, 0);
        grad_W2_ = Tensor::zeros(0, 0); grad_b2_ = Tensor::zeros(0, 0);
    }
}

Tensor FastFormerBlock::forward(const Tensor& input) {
    last_input_ = input;
    Tensor ln1_out = ln1_.forward(input);
    Tensor attn_out = attn_.forward(ln1_out);
    Tensor r1 = input + attn_out;  // elementwise residual
    last_ln1_out_ = ln1_out;
    last_attn_out_ = attn_out;
    last_r1_ = r1;

    if (!has_ffn_) {
        last_output_ = r1;
        return r1;
    }
    Tensor ln2_out = ln2_.forward(r1);
    Tensor ffn_pregelu = row_bias_add(matmul(ln2_out, W1_.transpose()), b1_);
    Tensor ffn_gelu = gelu(ffn_pregelu);
    Tensor ffn_out = row_bias_add(matmul(ffn_gelu, W2_.transpose()), b2_);
    Tensor output = r1 + ffn_out;
    last_ln2_out_ = ln2_out;
    last_ffn_pregelu_ = ffn_pregelu;
    last_ffn_gelu_ = ffn_gelu;
    last_ffn_out_ = ffn_out;
    last_output_ = output;
    return output;
}

Tensor FastFormerBlock::backward(const Tensor& grad_output, double /*lr*/) {
    size_t n = last_input_.rows;

    // dL/d(r1) and dL/d(ffn_out) start as grad_output (since output = r1 + ffn_out).
    Tensor dr1 = grad_output;
    Tensor dffn_out = grad_output;

    if (!has_ffn_) {
        // dr1 is the gradient w.r.t. the residual after attention.
        // Back-prop through residual: out = input + attn_out → d_attn_out = dr1, d_input += dr1.
        Tensor d_attn_out = dr1;
        Tensor d_ln1_out = attn_.backward(d_attn_out, 0.0);
        // Back-prop through LayerNorm 1
        Tensor d_input = ln1_.backward(d_ln1_out, 0.0);
        // Add residual gradient
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model_; ++j) d_input[i][j] += dr1[i][j];
        return d_input;
    }

    // ===== With FFN: chain through the FFN path =====
    // Back-prop through ffn_out = ffn_gelu @ W2^T + b2:
    //   dffn_gelu = dffn_out · W2  (n, ffn_hidden) @ (d_model, ffn_hidden) = (n, ffn_hidden)
    //   dW2[k][j] += Σ_i dffn_out[i][k] * ffn_gelu[i][j]
    //   db2[k]    += Σ_i dffn_out[i][k]
    Tensor dffn_gelu(n, ffn_hidden_);
    dffn_gelu.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) s += dffn_out[i][k] * W2_[k][j];
            dffn_gelu[i][j] = s;
        }
    }
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += dffn_out[i][k] * last_ffn_gelu_[i][j];
            grad_W2_[k][j] += s;
        }
    }
    for (size_t k = 0; k < d_model_; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dffn_out[i][k];
        grad_b2_[0][k] += s;
    }

    // Back-prop through GELU: ffn_gelu = GELU(ffn_pregelu)
    //   d/dx GELU(x) = 0.5 * (1 + erf(x/sqrt(2))) + x/sqrt(2π) * exp(-x²/2)
    //   = ffn_gelu' which we compute per-element
    Tensor dffn_pregelu(n, ffn_hidden_);
    dffn_pregelu.fill(0.0);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double x = last_ffn_pregelu_[i][j];
            double gelu_prime = 0.5 * (1.0 + std::erf(x * inv_sqrt2)) + x * inv_sqrt2pi * std::exp(-0.5 * x * x);
            dffn_pregelu[i][j] = dffn_gelu[i][j] * gelu_prime;
        }
    }

    // Back-prop through ffn_pregelu = ln2_out @ W1^T + b1
    //   d_ln2_out = dffn_pregelu · W1
    //   dW1[k][j] += Σ_i dffn_pregelu[i][k] * ln2_out[i][j]
    //   db1[k]    += Σ_i dffn_pregelu[i][k]
    Tensor d_ln2_out_of_ffn(n, d_model_);
    d_ln2_out_of_ffn.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < ffn_hidden_; ++k) s += dffn_pregelu[i][k] * W1_[k][j];
            d_ln2_out_of_ffn[i][j] = s;
        }
    }
    for (size_t k = 0; k < ffn_hidden_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += dffn_pregelu[i][k] * last_ln2_out_[i][j];
            grad_W1_[k][j] += s;
        }
    }
    for (size_t k = 0; k < ffn_hidden_; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dffn_pregelu[i][k];
        grad_b1_[0][k] += s;
    }

    // Back-prop through LayerNorm 2 — receives d_ln2_out_of_ffn PLUS the residual
    // contribution dr1 from the r1 path (output = r1 + ffn_out, so dr1 also flows back
    // through ln2_out via the residual connection r1 → ln2_out).
    // Actually: r1 is the residual AFTER attention (input + attn_out). It is the INPUT
    // of the FFN sublayer. ln2_out = LN2(r1). The chain is:
    //   loss → r1 (residual-1 path) and ln2_out (FFN path).
    // We've already routed dffn_out → d_ln2_out_of_ffn. The residual path dr1 directly
    // contributes to d_r1 (we already have dr1 = grad_output above). So:
    //   d_r1 = dr1 (from the residual path) + d_ln2_out_of_ffn (from the FFN path)
    Tensor d_r1(n, d_model_);
    d_r1.fill(0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_r1[i][j] = dr1[i][j] + d_ln2_out_of_ffn[i][j];

    // LN2 backward (gradient w.r.t. r1)
    Tensor d_r1_from_ln2 = ln2_.backward(d_ln2_out_of_ffn, 0.0);
    // Combined: d_r1 = dr1 (residual path) + d_r1_from_ln2 (FFN path → through LN2)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_r1[i][j] += d_r1_from_ln2[i][j];

    // Now d_r1 is the gradient w.r.t. the residual-1 output (input + attn_out).
    //   d_attn_out = d_r1
    //   d_input += d_r1 (residual contribution)
    Tensor d_attn_out = d_r1;
    Tensor d_ln1_out = attn_.backward(d_attn_out, 0.0);
    Tensor d_input = ln1_.backward(d_ln1_out, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) d_input[i][j] += d_r1[i][j];
    return d_input;
}

void FastFormerBlock::update_weights(double lr) {
    attn_.update_weights(lr);
    ln1_.update_weights(lr);
    ln2_.update_weights(lr);
    if (has_ffn_) {
        for (size_t i = 0; i < W1_.rows; ++i)
            for (size_t j = 0; j < W1_.cols; ++j) W1_[i][j] -= lr * grad_W1_[i][j];
        for (size_t j = 0; j < b1_.cols; ++j) b1_[0][j] -= lr * grad_b1_[0][j];
        for (size_t i = 0; i < W2_.rows; ++i)
            for (size_t j = 0; j < W2_.cols; ++j) W2_[i][j] -= lr * grad_W2_[i][j];
        for (size_t j = 0; j < b2_.cols; ++j) b2_[0][j] -= lr * grad_b2_[0][j];
    }
}

void FastFormerBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    if (has_ffn_) {
        grad_W1_.fill(0.0); grad_b1_.fill(0.0);
        grad_W2_.fill(0.0); grad_b2_.fill(0.0);
    }
}

std::vector<Tensor*> FastFormerBlock::parameters() {
    std::vector<Tensor*> p = attn_.parameters();
    p.push_back(&ln1_.gamma); p.push_back(&ln1_.beta);
    p.push_back(&ln2_.gamma); p.push_back(&ln2_.beta);
    if (has_ffn_) {
        p.push_back(&W1_); p.push_back(&b1_);
        p.push_back(&W2_); p.push_back(&b2_);
    }
    return p;
}

std::vector<Tensor*> FastFormerBlock::gradients() {
    std::vector<Tensor*> g = attn_.gradients();
    g.push_back(&ln1_.grad_gamma_); g.push_back(&ln1_.grad_beta_);
    g.push_back(&ln2_.grad_gamma_); g.push_back(&ln2_.grad_beta_);
    if (has_ffn_) {
        g.push_back(&grad_W1_); g.push_back(&grad_b1_);
        g.push_back(&grad_W2_); g.push_back(&grad_b2_);
    }
    return g;
}


// ============================================================================
// FastFormerModel
// ============================================================================
FastFormerModel::FastFormerModel(size_t input_dim, size_t d_model, size_t output_dim,
                                 size_t seq_len, size_t num_blocks,
                                 size_t num_heads, size_t ffn_hidden, bool causal)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      seq_len_(seq_len),
      num_blocks_(num_blocks),
      num_heads_(num_heads),
      ffn_hidden_(ffn_hidden),
      causal_(causal),
      ln_out_(d_model) {
    if (input_dim == 0) throw std::invalid_argument("FastFormerModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("FastFormerModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("FastFormerModel: output_dim must be > 0");
    if (seq_len == 0) throw std::invalid_argument("FastFormerModel: seq_len must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("FastFormerModel: num_blocks must be > 0");
    if (num_heads == 0) throw std::invalid_argument("FastFormerModel: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("FastFormerModel: d_model must be divisible by num_heads");

    W_in_ = Tensor::random(d_model, input_dim, 0.05);
    b_in_ = Tensor::zeros(1, d_model);
    grad_W_in_ = Tensor::zeros(d_model, input_dim);
    grad_b_in_ = Tensor::zeros(1, d_model);

    W_cls_ = Tensor::random(output_dim, d_model, 0.05);
    b_cls_ = Tensor::zeros(1, output_dim);
    grad_W_cls_ = Tensor::zeros(output_dim, d_model);
    grad_b_cls_ = Tensor::zeros(1, output_dim);

    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, num_heads, ffn_hidden, causal);
    }
}

Tensor FastFormerModel::forward(const Tensor& input) {
    last_input_ = input;
    // input: (input_dim, seq_len)
    if (input.rows != input_dim_)
        throw std::invalid_argument("FastFormerModel: input.rows must equal input_dim");
    if (input.cols != seq_len_)
        throw std::invalid_argument("FastFormerModel: input.cols must equal seq_len");

    // Project: (d_model, input_dim) @ (input_dim, seq_len) + b_in → (d_model, seq_len)
    Tensor proj = row_bias_add(matmul(W_in_, input), b_in_);
    Tensor x = proj.transpose();  // (seq_len, d_model) for blocks
    last_proj_out_ = x;

    last_block_outs_.clear();
    last_block_outs_.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        x = blocks_[i].forward(x);
        last_block_outs_.push_back(x);
    }

    // Mean-pool over seq dim: (1, d_model)
    Tensor pool(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < seq_len_; ++i) s += x[i][j];
        pool[0][j] = s / (double)seq_len_;
    }
    last_pool_ = pool;

    Tensor ln_out = ln_out_.forward(pool);
    last_ln_out_ = ln_out;

    // Classifier: (1, d_model) @ (d_model, output_dim) + b_cls = (1, output_dim)
    // W_cls_: (output_dim, d_model), so we need transpose
    Tensor output = row_bias_add(matmul(ln_out, W_cls_.transpose()), b_cls_);
    // output: (1, output_dim) — but the repo convention is (output_dim, 1).
    Tensor out_t = output.transpose();  // (output_dim, 1)
    last_output_ = out_t;
    return out_t;
}

Tensor FastFormerModel::backward(const Tensor& grad_output, double /*lr*/) {
    // grad_output: (output_dim, 1)
    Tensor d_pre_t = grad_output.transpose();  // (1, output_dim)

    // Back-prop through classifier: out = pool @ W_cls^T + b_cls
    // dW_cls[k][j] += Σ_{1} d_pre_t[0][k] * pool[0][j]
    for (size_t k = 0; k < output_dim_; ++k) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_W_cls_[k][j] += d_pre_t[0][k] * last_pool_[0][j];
        }
    }
    for (size_t k = 0; k < output_dim_; ++k) grad_b_cls_[0][k] += d_pre_t[0][k];
    // d_pool = d_pre_t · W_cls
    Tensor d_pool(1, d_model_);
    d_pool.fill(0.0);
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < output_dim_; ++k) s += d_pre_t[0][k] * W_cls_[k][j];
        d_pool[0][j] = s;
    }

    // Back-prop through LayerNorm
    Tensor d_ln_out = ln_out_.backward(d_pool, 0.0);

    // Mean-pool backward: each token in x gets d_pool[0][j] / seq_len
    Tensor dx(seq_len_, d_model_);
    dx.fill(0.0);
    for (size_t i = 0; i < seq_len_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            dx[i][j] = d_ln_out[0][j] / (double)seq_len_;
        }
    }

    // Back-prop through blocks in reverse order
    for (size_t b = num_blocks_; b-- > 0; ) {
        dx = blocks_[b].backward(dx, 0.0);
    }

    // Back-prop through input projection: x = input^T @ W_in^T + b_in
    // dx is (seq_len, d_model), need d_input of shape (input_dim, seq_len).
    Tensor dx_in_seq = dx;  // (seq_len, d_model)
    // dW_in[k][j] += Σ_i dx_in_seq[i][k] * input[j][i]
    for (size_t k = 0; k < d_model_; ++k) {
        for (size_t j = 0; j < input_dim_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < seq_len_; ++i) s += dx_in_seq[i][k] * last_input_[j][i];
            grad_W_in_[k][j] += s;
        }
    }
    // db_in[k] = Σ_i dx_in_seq[i][k]
    for (size_t k = 0; k < d_model_; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < seq_len_; ++i) s += dx_in_seq[i][k];
        grad_b_in_[0][k] += s;
    }
    // d_input = W_in^T @ dx_in_seq^T
    Tensor dx_seq_T(d_model_, seq_len_);
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t k = 0; k < d_model_; ++k)
            dx_seq_T[k][i] = dx_in_seq[i][k];
    Tensor d_input = matmul(W_in_.transpose(), dx_seq_T);  // (input_dim, seq_len)
    return d_input;
}

void FastFormerModel::update_weights(double lr) {
    for (size_t i = 0; i < W_in_.rows; ++i)
        for (size_t j = 0; j < W_in_.cols; ++j) W_in_[i][j] -= lr * grad_W_in_[i][j];
    for (size_t j = 0; j < b_in_.cols; ++j) b_in_[0][j] -= lr * grad_b_in_[0][j];
    for (size_t i = 0; i < W_cls_.rows; ++i)
        for (size_t j = 0; j < W_cls_.cols; ++j) W_cls_[i][j] -= lr * grad_W_cls_[i][j];
    for (size_t j = 0; j < b_cls_.cols; ++j) b_cls_[0][j] -= lr * grad_b_cls_[0][j];
    ln_out_.update_weights(lr);
    for (auto& b : blocks_) b.update_weights(lr);
}

void FastFormerModel::zero_grad() {
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_cls_.fill(0.0); grad_b_cls_.fill(0.0);
    ln_out_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
}

std::vector<Tensor*> FastFormerModel::parameters() {
    std::vector<Tensor*> p = {&W_in_, &b_in_, &W_cls_, &b_cls_,
                               &ln_out_.gamma, &ln_out_.beta};
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}

std::vector<Tensor*> FastFormerModel::gradients() {
    std::vector<Tensor*> g = {&grad_W_in_, &grad_b_in_, &grad_W_cls_, &grad_b_cls_,
                               &ln_out_.grad_gamma_, &ln_out_.grad_beta_};
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    return g;
}
