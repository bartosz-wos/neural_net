// Slot Attention — Locatello et al. 2020
//   "Object-Centric Learning with Slot Attention"
//   (https://arxiv.org/abs/2006.15055)
//
// Implementation: see slot_attention.h for the full mathematical formulation.
// This file implements:
//   * SlotAttention       — single slot-attention block (T iterations).
//   * SlotAttentionBlock  — SlotAttention + FFN residual block.
//   * SlotAttentionModel  — input proj + stack of blocks + classifier.

#include "slot_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Local helpers
// ============================================================================
namespace {

inline double sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
}

// sigmoid derivative in terms of the precomputed sigmoid output s:
//   d/dx sigmoid(x) = s * (1 - s)
inline double sigmoid_deriv_from_s(double s) {
    return s * (1.0 - s);
}

// Initialize a (rows, cols) weight tensor with Xavier-uniform-style init.
inline void init_xavier(Tensor& t, std::mt19937& gen, double scale = 1.0) {
    double bound = scale * std::sqrt(6.0 / static_cast<double>(t.rows + t.cols));
    std::uniform_real_distribution<double> dis(-bound, bound);
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            t(i, j) = dis(gen);
}

}  // namespace

// ============================================================================
// SlotAttention
// ============================================================================

SlotAttention::SlotAttention(size_t num_slots, size_t slot_dim, size_t input_dim,
                             size_t num_iterations, size_t hidden_dim, double epsilon)
    : num_slots_(num_slots), slot_dim_(slot_dim), input_dim_(input_dim),
      num_iterations_(num_iterations == 0 ? 1 : num_iterations),
      hidden_dim_(hidden_dim == 0 ? 4 * slot_dim : hidden_dim),
      epsilon_(epsilon),
      W_k_(input_dim, slot_dim), W_v_(input_dim, slot_dim), W_q_(slot_dim, slot_dim),
      b_k_(1, slot_dim), b_v_(1, slot_dim), b_q_(1, slot_dim),
      grad_W_k_(input_dim, slot_dim), grad_W_v_(input_dim, slot_dim), grad_W_q_(slot_dim, slot_dim),
      grad_b_k_(1, slot_dim), grad_b_v_(1, slot_dim), grad_b_q_(1, slot_dim),
      mu_(num_slots, slot_dim),
      grad_mu_(num_slots, slot_dim),
      ln_k_(slot_dim, 1e-7), ln_v_(slot_dim, 1e-7), ln_q_(slot_dim, 1e-7),
      ln_mlp_(slot_dim, 1e-7),
      mlp_fc1_(slot_dim, slot_dim),
      mlp_fc2_(slot_dim, slot_dim),
      W_zr_(2 * slot_dim, 2 * slot_dim),    // W_zr acting on [u; s]
      W_h_(2 * slot_dim, slot_dim),         // W_h acting on [u; rh]
      b_zr_(1, 2 * slot_dim),
      b_h_(1, slot_dim),
      grad_W_zr_(2 * slot_dim, 2 * slot_dim),
      grad_W_h_(2 * slot_dim, slot_dim),
      grad_b_zr_(1, 2 * slot_dim),
      grad_b_h_(1, slot_dim)
{
    // Set up caches
    cache_.resize(num_iterations_);
    for (auto& c : cache_) {
        c.slots_pre_gru = Tensor(num_slots, slot_dim);
        c.slots_post_gru = Tensor(num_slots, slot_dim);
        c.slots_post_mlp = Tensor(num_slots, slot_dim);
        c.k_proj = Tensor(0, slot_dim);
        c.v_proj = Tensor(0, slot_dim);
        c.q_proj = Tensor(num_slots, slot_dim);
        c.x_ln = Tensor(0, slot_dim);
        c.v_ln = Tensor(0, slot_dim);
        c.slots_ln_q = Tensor(num_slots, slot_dim);
        c.slots_ln_mlp = Tensor(num_slots, slot_dim);
        c.mlp_h = Tensor(num_slots, slot_dim);
        c.logits = Tensor(num_slots, 0);
        c.attn1 = Tensor(num_slots, 0);
        c.attn2 = Tensor(num_slots, 0);
        c.updates = Tensor(num_slots, slot_dim);
        c.z_gates = Tensor(num_slots, slot_dim);
        c.r_gates = Tensor(num_slots, slot_dim);
        c.s_hat = Tensor(num_slots, slot_dim);
        c.rh = Tensor(num_slots, slot_dim);
    }
    last_input_ = Tensor(0, 0);

    // Weight init (Xavier-style)
    std::mt19937 gen(42);
    init_xavier(W_k_, gen); init_xavier(W_v_, gen); init_xavier(W_q_, gen);
    W_k_.fill(0.0); W_v_.fill(0.0); W_q_.fill(0.0);  // zero-init first (paper recommendation)
    init_xavier(b_k_, gen, 0.1); init_xavier(b_v_, gen, 0.1); init_xavier(b_q_, gen, 0.1);
    // mu is small random init (Locatello paper convention: small random slots)
    std::normal_distribution<double> dnorm(0.0, 0.1);
    for (size_t i = 0; i < mu_.rows; ++i)
        for (size_t j = 0; j < mu_.cols; ++j)
            mu_(i, j) = dnorm(gen);

    // GRU weights — combined W_zr and W_h
    init_xavier(W_zr_, gen, 0.5);
    init_xavier(W_h_, gen, 0.5);
    b_zr_.fill(0.0);
    b_h_.fill(0.0);

    zero_grad();
}

void SlotAttention::zero_grad() {
    grad_W_k_.fill(0.0); grad_W_v_.fill(0.0); grad_W_q_.fill(0.0);
    grad_b_k_.fill(0.0); grad_b_v_.fill(0.0); grad_b_q_.fill(0.0);
    grad_mu_.fill(0.0);
    ln_k_.zero_grad(); ln_v_.zero_grad(); ln_q_.zero_grad(); ln_mlp_.zero_grad();
    mlp_fc1_.zero_grad(); mlp_fc2_.zero_grad();
    grad_W_zr_.fill(0.0); grad_W_h_.fill(0.0);
    grad_b_zr_.fill(0.0); grad_b_h_.fill(0.0);
}

std::vector<Tensor*> SlotAttention::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_k_); p.push_back(&b_k_);
    p.push_back(&W_v_); p.push_back(&b_v_);
    p.push_back(&W_q_); p.push_back(&b_q_);
    p.push_back(&mu_);
    auto lk = ln_k_.parameters();   for (auto* x : lk) p.push_back(x);
    auto lv = ln_v_.parameters();   for (auto* x : lv) p.push_back(x);
    auto lq = ln_q_.parameters();   for (auto* x : lq) p.push_back(x);
    auto lm = ln_mlp_.parameters(); for (auto* x : lm) p.push_back(x);
    auto m1 = mlp_fc1_.parameters(); for (auto* x : m1) p.push_back(x);
    auto m2 = mlp_fc2_.parameters(); for (auto* x : m2) p.push_back(x);
    p.push_back(&W_zr_); p.push_back(&b_zr_);
    p.push_back(&W_h_); p.push_back(&b_h_);
    return p;
}

std::vector<Tensor*> SlotAttention::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_W_k_); g.push_back(&grad_b_k_);
    g.push_back(&grad_W_v_); g.push_back(&grad_b_v_);
    g.push_back(&grad_W_q_); g.push_back(&grad_b_q_);
    g.push_back(&grad_mu_);
    auto lk = ln_k_.gradients();   for (auto* x : lk) g.push_back(x);
    auto lv = ln_v_.gradients();   for (auto* x : lv) g.push_back(x);
    auto lq = ln_q_.gradients();   for (auto* x : lq) g.push_back(x);
    auto lm = ln_mlp_.gradients(); for (auto* x : lm) g.push_back(x);
    auto m1 = mlp_fc1_.gradients(); for (auto* x : m1) g.push_back(x);
    auto m2 = mlp_fc2_.gradients(); for (auto* x : m2) g.push_back(x);
    g.push_back(&grad_W_zr_); g.push_back(&grad_b_zr_);
    g.push_back(&grad_W_h_); g.push_back(&grad_b_h_);
    return g;
}

void SlotAttention::update_weights(double learning_rate) {
    // Use a simple SGD-style update
    auto params = parameters();
    auto grads = gradients();
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = 0; j < params[i]->data.size(); ++j) {
            params[i]->data[j] -= learning_rate * grads[i]->data[j];
        }
    }
}

// ----- helpers: row / col softmax -----
Tensor SlotAttention::row_softmax(const Tensor& x) {
    Tensor r(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double m = x(i, 0);
        for (size_t j = 1; j < x.cols; ++j) if (x(i, j) > m) m = x(i, j);
        double s = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            r(i, j) = std::exp(x(i, j) - m);
            s += r(i, j);
        }
        double inv = 1.0 / (s + 1e-12);
        for (size_t j = 0; j < x.cols; ++j) r(i, j) *= inv;
    }
    return r;
}

Tensor SlotAttention::col_softmax(const Tensor& x) {
    Tensor r(x.rows, x.cols);
    for (size_t j = 0; j < x.cols; ++j) {
        double m = x(0, j);
        for (size_t i = 1; i < x.rows; ++i) if (x(i, j) > m) m = x(i, j);
        double s = 0.0;
        for (size_t i = 0; i < x.rows; ++i) {
            r(i, j) = std::exp(x(i, j) - m);
            s += r(i, j);
        }
        double inv = 1.0 / (s + 1e-12);
        for (size_t i = 0; i < x.rows; ++i) r(i, j) *= inv;
    }
    return r;
}

// ----- GRU helpers (stateless, applied per-slot-batch-wise) -----
SlotAttention::GruOut SlotAttention::gru_forward(const Tensor& u, const Tensor& s) {
    // u, s: (K, D). W_zr: (2D, 2D) — first D rows for u, second D rows for s.
    // zr[j] = sigmoid( sum_k u[k]*W_zr[k,j] + sum_k s[k]*W_zr[D+k,j] + b_zr[j] )
    size_t K = u.rows, D = u.cols;
    GruOut out;
    out.new_s = Tensor(K, D);
    out.z = Tensor(K, D);
    out.r = Tensor(K, D);
    out.s_hat = Tensor(K, D);
    out.rh = Tensor(K, D);

    // Compute z, r and the combined [u; r*s] for candidate
    Tensor u_part(K, 2 * D);
    Tensor s_part(K, 2 * D);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < 2 * D; ++j) {
            double uu = 0.0, ss = 0.0;
            for (size_t k = 0; k < D; ++k) {
                uu += u(i, k) * W_zr_(k, j);
                ss += s(i, k) * W_zr_(D + k, j);
            }
            u_part(i, j) = uu;
            s_part(i, j) = ss;
        }
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double zr_pre = u_part(i, j) + s_part(i, j) + b_zr_(0, j);
            double z = sigmoid(zr_pre);
            out.z(i, j) = z;
            double rr_pre = u_part(i, D + j) + s_part(i, D + j) + b_zr_(0, D + j);
            double r = sigmoid(rr_pre);
            out.r(i, j) = r;
            out.rh(i, j) = r * s(i, j);
        }
    }

    // Candidate: s_hat = tanh(u @ W_h[:, 0:D] + (r*s) @ W_h[:, D:2D] + b_h)
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double v = b_h_(0, j);
            for (size_t k = 0; k < D; ++k) v += u(i, k) * W_h_(k, j);
            for (size_t k = 0; k < D; ++k) v += out.rh(i, k) * W_h_(D + k, j);
            out.s_hat(i, j) = std::tanh(v);
        }
    }

    // new_s = (1 - z) * s + z * s_hat
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j)
            out.new_s(i, j) = (1.0 - out.z(i, j)) * s(i, j) + out.z(i, j) * out.s_hat(i, j);

    return out;
}

void SlotAttention::gru_backward(const Tensor& grad_new_s,
                                 const Tensor& u, const Tensor& s,
                                 const GruOut& st,
                                 Tensor& grad_u, Tensor& grad_s) {
    // grad_new_s: (K, D). u, s: (K, D). st: from gru_forward.
    // Returns grad_u and grad_s w.r.t. inputs. Also accumulates into grad_W_zr_, grad_W_h_,
    // grad_b_zr_, grad_b_h_ (in-place; caller is expected to have zeroed them).
    size_t K = grad_new_s.rows, D = grad_new_s.cols;
    grad_u = Tensor(K, D);
    grad_s = Tensor(K, D);

    // From new_s = (1-z)*s + z*s_hat:
    //   dL/dz[i,j] = grad_new_s[i,j] * (s_hat[i,j] - s[i,j])
    //   dL/ds_hat[i,j] = grad_new_s[i,j] * z[i,j]
    //   dL/ds[i,j] (direct) = grad_new_s[i,j] * (1 - z[i,j])
    //   dL/dr[i,j] = dL/d(r*s)[i,j] * s[i,j]  (handled via rh backward)
    //   dL/d(r*s)[i,j] = sum_l dL/ds_hat[i,l] * W_h[D+j, l]
    //   dL/du[i,j] = sum_l (dL/dz[i,l] * W_zr[j,l] + dL/dr[i,l] * W_zr[j,D+l] + dL/ds_hat[i,l] * W_h[j,l])
    //   dL/ds[i,j] (full) = direct + sum_l (dL/dz[i,l] * W_zr[D+j,l] + dL/dr[i,l] * W_zr[D+j,D+l])

    // Step 1: dL/dz, dL/dr (via rh), dL/ds_hat
    Tensor dz(K, D), dr(K, D), ds_hat(K, D), drh(K, D);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            dz(i, j) = grad_new_s(i, j) * (st.s_hat(i, j) - s(i, j));
            ds_hat(i, j) = grad_new_s(i, j) * st.z(i, j);
        }
    }

    // drh[i,j] = sum_l ds_hat[i,l] * W_h[D+j, l]
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j) {
            double acc = 0.0;
            for (size_t l = 0; l < D; ++l) acc += ds_hat(i, l) * W_h_(D + j, l);
            drh(i, j) = acc;
        }

    // dr[i,j] = drh[i,j] * s[i,j] * sigmoid_deriv(r[i,j])
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j)
            dr(i, j) = drh(i, j) * s(i, j) * sigmoid_deriv_from_s(st.r(i, j));

    // Now: through z = sigmoid(zr_pre) with zr_pre[j] = sum_k u[k]*W_zr[k,j] + s[k]*W_zr[D+k,j] + b_zr[j]
    //   dL/dzr_pre[j] = dz[i,j] * sigmoid_deriv(z[i,j]) (per slot i)
    //   grad_W_zr[k, j] += sum_i dz[i,j] * sigmoid_deriv(z[i,j]) * u[i,k]
    //   grad_W_zr[D+k, j] += ... * s[i,k]
    //   grad_b_zr[j] += sum_i ...

    // Same for r:
    //   dL/drr_pre[j] = dr[i,j] * sigmoid_deriv(r[i,j])
    //   grad_W_zr[k, D+j] += sum_i (dL/drr_pre[i,j]) * u[i,k]
    //   grad_W_zr[D+k, D+j] += ... * s[i,k]
    //   grad_b_zr[D+j] += ...

    // dL/du[k] = sum_j ( dz[j] * sigmoid_deriv(z) * W_zr[k,j] + dr[j] * sigmoid_deriv(r) * W_zr[k, D+j] )
    //              + sum_l ds_hat[l] * W_h[k, l]   (s_hat = tanh(...) chain)
    //   Note the tanh derivative is 1 - tanh^2 = 1 - s_hat^2.
    //   ds_hat[i,j] already has dL/ds_hat applied; we need to apply tanh derivative here:
    //   dL/dtanh_pre[i,j] = ds_hat[i,j] * (1 - s_hat[i,j]^2)

    // Compute tanh_derivative contribution to ds_hat:
    Tensor d_tanh_pre(K, D);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j)
            d_tanh_pre(i, j) = ds_hat(i, j) * (1.0 - st.s_hat(i, j) * st.s_hat(i, j));

    // dL/du[i,k] = sum_j (dz[i,j] * sd_z * W_zr[k, j] + dr[i,j] * sd_r * W_zr[k, D+j])
    //             + sum_l d_tanh_pre[i, l] * W_h[k, l]
    // dL/ds[i,k] = sum_j (dz[i,j] * sd_z * W_zr[D+k, j] + dr[i,j] * sd_r * W_zr[D+k, D+j])
    //             + grad_new_s[i,k] * (1 - z[i,k])  (the direct (1-z)*s contribution)
    //             + drh[i,k] * r[i,k]   (because r*s path: dL/dr * r already gave dr, but the rh chain back into s is via drh already; we accounted drh = dL/d(rh) above, and rh = r*s -> dL/ds also gets rh * r contribution from the d(rh) path. Wait — drh is dL/d(rh), and rh = r*s, so dL/ds_direct_via_rh = drh * r. We add this.)
    // Hmm, but the direct (1-z)*s contribution was already accounted in grad_s via grad_new_s * (1-z).
    // Let me redo this carefully:
    //   new_s = (1-z)*s + z*s_hat
    //   grad_s (direct from new_s) = grad_new_s * (1-z)
    //   And s_hat = tanh(...) doesn't directly depend on s.
    //   But rh = r*s, where r = sigmoid(W_r@[u;s] + b_r). r DOES depend on s.
    //   So dL/ds also has: (path through r) -> (path through rh) -> (path through s_hat).
    //   We have dL/drh = sum_l dL/ds_hat[i,l] * W_h[D+j, l] (computed above)
    //   rh = r * s, so dL/ds_via_rh = drh * r.
    //   So total dL/ds = grad_new_s * (1-z)  +  drh * r  +  sum_l (dz * sd_z * W_zr[D+k,l] + dr * sd_r * W_zr[D+k,D+l])
    //                  (where the third term is the path through z and r via [u;s] linear combination with the s branch)
    //
    // Actually we also need dL/ds via the direct (1-z)*s contribution when z depends on s.
    // z depends on s via [u;s] -> W_zr[D+k, :], so dL/ds_via_z = sum_l (dz * sd_z * W_zr[D+k, l]).
    //
    // Let me re-derive cleanly:
    //
    // gates:    z[j] = sigmoid( zr_pre[j] ),  r[j] = sigmoid( rr_pre[j] )
    //   zr_pre[j] = sum_k u[k] W_zr[k,j]    +  sum_k s[k] W_zr[D+k,j]    + b_zr[j]
    //   rr_pre[j] = sum_k u[k] W_zr[k,D+j]  +  sum_k s[k] W_zr[D+k,D+j]  + b_zr[D+j]
    // rh[j] = r[j] * s[j]
    // s_hat[j] = tanh( sum_k u[k] W_h[k,j]  +  sum_k rh[k] W_h[D+k,j]   + b_h[j] )
    // new_s[j] = (1-z[j])*s[j] + z[j]*s_hat[j]
    //
    // Local grads (g = grad_new_s):
    //   d_new_s/dz[j] = -s[j] + s_hat[j]
    //   d_new_s/ds_hat[j] = z[j]
    //   d_new_s/ds[j] (direct, treating s only as direct input) = 1 - z[j]
    //
    //   grad_z[j] = g[j] * (s_hat[j] - s[j])
    //   grad_s_hat[j] = g[j] * z[j]
    //
    //   grad_s_direct[j] = g[j] * (1 - z[j])
    //
    //   d_s_hat/d_rh[k] = (1 - s_hat^2)[k] * W_h[D+k, j] (j is output of s_hat)
    //   d_rh[k]/d_r[k] = s[k]
    //   d_rh[k]/d_s[k] = r[k]
    //
    //   grad_rh[k] = sum_j grad_s_hat[j] * (1-s_hat^2)[j] * W_h[D+k, j]
    //   grad_r[k] (via rh) = grad_rh[k] * s[k]
    //   grad_r[k] (via sigmoid) = grad_r[k]_total * sigmoid_deriv(r[k])
    //   grad_s[k] (via rh) = grad_rh[k] * r[k]
    //
    //   grad_z_pre[j] = grad_z[j] * sigmoid_deriv(z[j])
    //   grad_r_pre[j] = grad_r[k]_total * sigmoid_deriv(r[k])
    //
    //   grad_u[a] = sum_j grad_z_pre[j] * W_zr[a, j]  +  sum_j grad_r_pre_total[j] * W_zr[a, D+j]
    //            + sum_j grad_s_hat[j] * (1-s_hat^2)[j] * W_h[a, j]
    //   grad_s[a] = grad_s_direct[a] + grad_s_via_rh[a]
    //             + sum_j grad_z_pre[j] * W_zr[D+a, j]  +  sum_j grad_r_pre_total[j] * W_zr[D+a, D+j]

    // OK let's implement it cleanly.

    Tensor grad_z_pre(K, D), grad_r_pre(K, D);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            grad_z_pre(i, j) = dz(i, j) * sigmoid_deriv_from_s(st.z(i, j));
            double grad_r_total = dr(i, j);
            grad_r_pre(i, j) = grad_r_total * sigmoid_deriv_from_s(st.r(i, j));
        }
    }

    // Accumulate gradients into grad_W_zr_, grad_b_zr_
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < 2 * D; ++j) {
            double gz = grad_z_pre(i, j % D);
            double gr = (j < D) ? 0.0 : grad_r_pre(i, j - D);
            double scale = gz + gr;
            for (size_t k = 0; k < D; ++k) {
                grad_W_zr_(k, j) += scale * u(i, k);
                grad_W_zr_(D + k, j) += scale * s(i, k);
            }
            if (j < D) grad_b_zr_(0, j) += scale;
            else       grad_b_zr_(0, j) += scale;
        }
    }
    // Fix the bias accumulation (above loop double-applied; recompute properly):
    for (size_t i = 0; i < D; ++i) {
        double zb = 0.0, rb = 0.0;
        for (size_t k = 0; k < K; ++k) {
            zb += grad_z_pre(k, i);
            rb += grad_r_pre(k, i);
        }
        grad_b_zr_(0, i) = zb;
        grad_b_zr_(0, D + i) = rb;
    }

    // Accumulate gradients into grad_W_h_, grad_b_h_
    // s_hat[j] = tanh( sum_k u[k] W_h[k,j] + sum_k rh[k] W_h[D+k,j] + b_h[j] )
    // d_tanh_pre[i,j] = grad_s_hat[i,j] * (1 - s_hat[i,j]^2)
    // grad_W_h[k, j] += sum_i d_tanh_pre[i,j] * u[i,k]
    // grad_W_h[D+k, j] += sum_i d_tanh_pre[i,j] * rh[i,k]
    // grad_b_h[j] += sum_i d_tanh_pre[i,j]
    for (size_t j = 0; j < D; ++j) {
        double bs = 0.0;
        for (size_t i = 0; i < K; ++i) {
            double d = d_tanh_pre(i, j);
            bs += d;
            for (size_t k = 0; k < D; ++k) {
                grad_W_h_(k, j) += d * u(i, k);
                grad_W_h_(D + k, j) += d * st.rh(i, k);
            }
        }
        grad_b_h_(0, j) += bs;
    }

    // Compute grad_u and grad_s
    for (size_t i = 0; i < K; ++i) {
        for (size_t a = 0; a < D; ++a) {
            double gu = 0.0, gs = 0.0;
            // Path through z and r gates (u branch of [u;s])
            for (size_t j = 0; j < D; ++j) {
                gu += grad_z_pre(i, j) * W_zr_(a, j);
                gu += grad_r_pre(i, j) * W_zr_(a, D + j);
                gs += grad_z_pre(i, j) * W_zr_(D + a, j);
                gs += grad_r_pre(i, j) * W_zr_(D + a, D + j);
            }
            // Path through s_hat -> W_h (u branch)
            for (size_t j = 0; j < D; ++j) {
                gu += d_tanh_pre(i, j) * W_h_(a, j);
            }
            // Path through rh -> s (rh = r * s, drh computed above)
            // dL/ds_via_rh = drh[i, a] * r[i, a]
            gs += drh(i, a) * st.r(i, a);
            // Direct (1-z)*s contribution to s
            gs += grad_new_s(i, a) * (1.0 - st.z(i, a));

            grad_u(i, a) = gu;
            grad_s(i, a) = gs;
        }
    }
}

// ----- Forward -----
Tensor SlotAttention::forward(const Tensor& input) {
    size_t N = input.rows;
    size_t D = slot_dim_;
    size_t K = num_slots_;

    last_input_ = input;

    // Resize cache for this N
    for (auto& c : cache_) {
        c.k_proj = Tensor(N, D);
        c.v_proj = Tensor(N, D);
        c.x_ln = Tensor(0, 0);  // will be reassigned below to (N, input_dim)
        c.v_ln = Tensor(0, 0);  // ditto
        c.logits = Tensor(K, N);
        c.attn1 = Tensor(K, N);
        c.attn2 = Tensor(K, N);
    }

    // Compute LN(x) once (it's the same at every iteration).
    // LN preserves shape, so x_ln and v_ln have the same shape as input: (N, input_dim).
    Tensor x_ln = ln_k_.forward(input);  // (N, input_dim)
    Tensor v_ln = ln_v_.forward(input);  // (N, input_dim)

    // For each iteration:
    Tensor slots = mu_.clone();
    double scale = 1.0 / std::sqrt(static_cast<double>(D));

    for (size_t t = 0; t < num_iterations_; ++t) {
        IterCache& c = cache_[t];

        // Slots LN_q + W_q + b_q: q = LN_q(slots) @ W_q^T + b_q
        Tensor slots_ln_q = ln_q_.forward(slots);
        c.slots_ln_q = slots_ln_q;
        // q_proj: (K, D) = slots_ln_q @ W_q^T + b_q
        // W_q^T shape (D, D), so q_proj = slots_ln_q * W_q^T, shape (K, D)
        c.q_proj = slots_ln_q * W_q_.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                c.q_proj(i, j) += b_q_(0, j);

        // k_proj: x_ln @ W_k^T + b_k  (N, D)
        c.x_ln = x_ln;  // cache for k proj grad
        c.k_proj = x_ln * W_k_.transpose();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                c.k_proj(i, j) += b_k_(0, j);

        // v_proj: v_ln @ W_v^T + b_v
        c.v_ln = v_ln;
        c.v_proj = v_ln * W_v_.transpose();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                c.v_proj(i, j) += b_v_(0, j);

        // logits = q_proj @ k_proj^T * scale  → (K, N)
        Tensor qk_t = c.q_proj * c.k_proj.transpose();  // (K, N)
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < N; ++j)
                c.logits(i, j) = qk_t(i, j) * scale;

        // Double softmax: col softmax (over slots), then row softmax (over inputs)
        c.attn1 = col_softmax(c.logits);   // each column sums to 1
        c.attn2 = row_softmax(c.attn1);    // each row sums to 1

        // updates = attn2 @ v_proj  (K, D)
        c.updates = c.attn2 * c.v_proj;

        // Cache slots pre-GRU
        c.slots_pre_gru = slots.clone();

        // GRU update: per-slot update with [u; s] input
        GruOut g = gru_forward(c.updates, slots);
        c.z_gates = g.z;
        c.r_gates = g.r;
        c.s_hat = g.s_hat;
        c.rh = g.rh;
        c.slots_post_gru = g.new_s;

        // Residual MLP: slots = slots_post_gru + fc2(relu(fc1(LN(slots_post_gru))))
        c.slots_ln_mlp = ln_mlp_.forward(c.slots_post_gru);
        // fc1: slots_ln_mlp @ W1^T + b1 (W1 is (D, D) per Dense convention)
        Tensor mlp_pre = c.slots_ln_mlp * mlp_fc1_.weights.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                mlp_pre(i, j) += mlp_fc1_.bias(0, j);
        // ReLU
        c.mlp_h = mlp_pre.apply([](double v) { return v > 0.0 ? v : 0.0; });
        // fc2
        Tensor mlp_out = c.mlp_h * mlp_fc2_.weights.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                mlp_out(i, j) += mlp_fc2_.bias(0, j);
        // Residual
        c.slots_post_mlp = c.slots_post_gru + mlp_out;
        slots = c.slots_post_mlp;
    }

    return slots;
}

// ----- Backward -----
Tensor SlotAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t N = last_input_.rows;
    size_t D = slot_dim_;
    size_t K = num_slots_;
    double scale = 1.0 / std::sqrt(static_cast<double>(D));

    // grad_output: (K, D) — dL/d(slots_T)
    Tensor d_slots = grad_output.clone();

    // Zero out param grads (we accumulate across iterations and into W_q/k/v/b_q/k/v,
    // mu, ln_k_, ln_v_, ln_q_, ln_mlp_, mlp_fc1_, mlp_fc2_, W_zr/h, b_zr/h).
    grad_W_q_.fill(0.0); grad_W_k_.fill(0.0); grad_W_v_.fill(0.0);
    grad_b_q_.fill(0.0); grad_b_k_.fill(0.0); grad_b_v_.fill(0.0);
    grad_mu_.fill(0.0);
    ln_k_.zero_grad(); ln_v_.zero_grad(); ln_q_.zero_grad(); ln_mlp_.zero_grad();
    mlp_fc1_.zero_grad(); mlp_fc2_.zero_grad();
    grad_W_zr_.fill(0.0); grad_W_h_.fill(0.0);
    grad_b_zr_.fill(0.0); grad_b_h_.fill(0.0);

    // Accumulator for input gradient d_x
    Tensor d_x(N, D);
    d_x.fill(0.0);

    // Iterate from T-1 down to 0
    for (int t = static_cast<int>(num_iterations_) - 1; t >= 0; --t) {
        IterCache& c = cache_[t];

        // ===== Residual MLP backward =====
        // c.slots_post_mlp = c.slots_post_gru + mlp_out
        // dL/d_slots_post_gru = d_slots (residual)
        // dL/d_mlp_out = d_slots
        Tensor d_mlp_out = d_slots.clone();
        Tensor d_slots_post_gru = d_slots.clone();

        // mlp_out = mlp_h @ W2^T + b2
        // dL/d_mlp_h = d_mlp_out @ W2
        // dL/d_W2 = d_mlp_out^T @ mlp_h
        // dL/d_b2 = sum over rows of d_mlp_out
        Tensor d_mlp_h = d_mlp_out * mlp_fc2_.weights;
        // d_W2[j, k] += sum_i d_mlp_out[i, j] * mlp_h[i, k]
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    mlp_fc2_.grad_weights(j, k) += d_mlp_out(i, j) * c.mlp_h(i, k);
                }
                mlp_fc2_.grad_bias(0, j) += d_mlp_out(i, j);
            }
        }

        // ReLU backward on mlp_pre
        Tensor d_mlp_pre = d_mlp_h.hadamard(c.mlp_h.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; }));
        // dL/d_W1[j, k] += sum_i d_mlp_pre[i, j] * slots_ln_mlp[i, k]
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    mlp_fc1_.grad_weights(j, k) += d_mlp_pre(i, j) * c.slots_ln_mlp(i, k);
                }
                mlp_fc1_.grad_bias(0, j) += d_mlp_pre(i, j);
            }
        }
        // dL/d_slots_ln_mlp = d_mlp_pre @ W1
        Tensor d_slots_ln_mlp = d_mlp_pre * mlp_fc1_.weights;

        // LN_mlp backward
        Tensor d_slots_post_gru_from_ln = d_slots_ln_mlp;
        Tensor d_x_from_ln = ln_mlp_.backward(d_slots_post_gru_from_ln, 0.0);
        // d_slots_post_gru += d_x_from_ln (LN doesn't add anything to d_slots_post_gru directly
        // because slots_ln_mlp is the OUTPUT of LN, and slots_post_gru is the INPUT)
        // Actually d_slots_post_gru (the input to LN_mlp) is what we want. ln_mlp_.backward()
        // returned grad w.r.t. input = d_slots_post_gru from the LN path.
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_post_gru(i, j) += d_x_from_ln(i, j);

        // ===== GRU backward =====
        // c.slots_post_gru = GRU(c.updates, c.slots_pre_gru)
        // d_slots_post_gru -> d_updates, d_slots_pre_gru
        Tensor d_updates_from_gru, d_slots_pre_gru;
        gru_backward(d_slots_post_gru, c.updates, c.slots_pre_gru,
                     {c.slots_post_gru, c.z_gates, c.r_gates, c.s_hat, c.rh},
                     d_updates_from_gru, d_slots_pre_gru);

        // ===== Updates = attn2 @ v_proj =====
        // dL/d_attn2 = d_updates_from_gru @ v_proj^T
        // dL/d_v_proj = attn2^T @ d_updates_from_gru
        Tensor d_attn2 = d_updates_from_gru * c.v_proj.transpose();
        Tensor d_v_proj = c.attn2.transpose() * d_updates_from_gru;
        // Also accumulate into d_x via the v_proj chain — handled after attn2 backward.

        // ===== Double softmax backward =====
        // attn2 = row_softmax(attn1)
        // dL/d_attn1 = row_softmax_backward(d_attn2, attn2)
        // attn1 = col_softmax(logits)
        // dL/d_logits = col_softmax_backward(d_attn1, attn1)
        // Then accumulate dL/d_logits into d_logits = scale * qk_t_grad (which is qk_t with no
        // other transform; qk_t = q_proj @ k_proj^T).
        // We have d_attn2: (K, N). Compute d_attn1 via row-softmax backward.
        Tensor d_attn1(K, N);
        for (size_t i = 0; i < K; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < N; ++j) dot += d_attn2(i, j) * c.attn2(i, j);
            for (size_t j = 0; j < N; ++j) {
                d_attn1(i, j) = c.attn2(i, j) * (d_attn2(i, j) - dot);
            }
        }

        // col_softmax backward
        // attn1[i, j] = exp(logits[i, j] - col_max[j]) / sum_i' exp(logits[i', j] - col_max[j])
        // dL/d_logits[i, j] = attn1[i, j] * (d_attn1[i, j] - sum_i' attn1[i', j] * d_attn1[i', j])
        Tensor d_logits(K, N);
        for (size_t j = 0; j < N; ++j) {
            double col_dot = 0.0;
            for (size_t i = 0; i < K; ++i) col_dot += c.attn1(i, j) * d_attn1(i, j);
            for (size_t i = 0; i < K; ++i) {
                d_logits(i, j) = c.attn1(i, j) * (d_attn1(i, j) - col_dot);
            }
        }

        // Apply scale factor
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < N; ++j)
                d_logits(i, j) *= scale;

        // ===== q_proj / k_proj / v_proj backward =====
        // qk_t = q_proj @ k_proj^T  (K, N)
        // dL/d_q_proj = d_logits @ k_proj  (K, D)
        // dL/d_k_proj = d_logits^T @ q_proj  (N, D)
        Tensor d_q_proj = d_logits * c.k_proj;
        Tensor d_k_proj = d_logits.transpose() * c.q_proj;
        // dL/d_v_proj already computed
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_v_proj(i, j) += 0.0;  // (placeholder for clarity; nothing to add)

        // ===== W_q / W_k / W_v / b_q / b_k / b_v grads =====
        // q_proj = slots_ln_q @ W_q^T + b_q  → dL/d_W_q[j, k] += sum_i d_q_proj[i, j] * slots_ln_q[i, k]
        // W_q is Dense(slot_dim, slot_dim) → weights (slot_dim, slot_dim). j,k in [0, slot_dim).
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    grad_W_q_(j, k) += d_q_proj(i, j) * c.slots_ln_q(i, k);
                }
                grad_b_q_(0, j) += d_q_proj(i, j);
            }
        }
        // dL/d_slots_ln_q = d_q_proj @ W_q  (K, D)
        Tensor d_slots_ln_q = d_q_proj * W_q_;

        // k_proj = x_ln @ W_k^T + b_k  → dL/d_W_k[j, k] += sum_i d_k_proj[i, j] * x_ln[i, k]
        // W_k is Dense(input_dim, slot_dim) → weights (slot_dim, input_dim). j in [0, slot_dim), k in [0, input_dim).
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < input_dim_; ++k) {
                    grad_W_k_(j, k) += d_k_proj(i, j) * c.x_ln(i, k);
                }
                grad_b_k_(0, j) += d_k_proj(i, j);
            }
        }
        // dL/d_x_ln = d_k_proj @ W_k  (N, input_dim)
        Tensor d_x_ln = d_k_proj * W_k_;

        // v_proj = v_ln @ W_v^T + b_v  → dL/d_W_v[j, k] += sum_i d_v_proj[i, j] * v_ln[i, k]
        // W_v is Dense(input_dim, slot_dim) → weights (slot_dim, input_dim). j in [0, slot_dim), k in [0, input_dim).
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < input_dim_; ++k) {
                    grad_W_v_(j, k) += d_v_proj(i, j) * c.v_ln(i, k);
                }
                grad_b_v_(0, j) += d_v_proj(i, j);
            }
        }
        // dL/d_v_ln = d_v_proj @ W_v  (N, input_dim)
        // (W_v_ is (slot_dim, input_dim), d_v_proj is (N, slot_dim) → product is (N, input_dim))
        // Wait: d_v_proj is (N, slot_dim), W_v_ is (slot_dim, input_dim). So d_v_proj * W_v_ is (N, input_dim).
        // But our cache v_ln is (N, slot_dim) (after LN)... actually v_ln is the LN(input) which is
        // (N, slot_dim). And we use it as input to the W_v^T projection to produce v_proj (N, slot_dim).
        // So d_v_ln should be (N, slot_dim). But here I'm computing (N, input_dim). Hmm.
        //
        // Wait let me re-derive. v_proj = v_ln @ W_v^T + b_v. So:
        //   v_ln: (N, slot_dim) ← output of LN_v applied to input
        //   W_v: (slot_dim, input_dim) → no, W_v has shape (slot_dim, input_dim) per Dense(input_dim, slot_dim).
        //   But then v_ln (N, slot_dim) @ W_v^T (input_dim, slot_dim) → requires slot_dim == input_dim. MISMATCH!
        //
        // OH! I see my confusion. v_ln is the LN of input, so v_ln has the SAME shape as input (N, input_dim).
        // The LN normalizes per-row across features, so it preserves shape. So v_ln is (N, input_dim), NOT (N, slot_dim).
        // Then v_ln (N, input_dim) @ W_v^T (input_dim, slot_dim) → (N, slot_dim). ✓
        //
        // I need to fix the cache shape too! c.v_ln is currently (N, slot_dim) but should be (N, input_dim).
        // Similarly for x_ln — should be (N, input_dim), not (N, slot_dim). Wait, in my forward:
        //
        //     Tensor x_ln = ln_k_.forward(input);  // LN preserves shape: (N, input_dim)
        //     Tensor v_ln = ln_v_.forward(input);  // (N, input_dim)
        //
        // Then I assigned:
        //     c.x_ln = x_ln;  // this IS (N, input_dim) since I just assigned the result of ln_k_.forward()
        //     c.v_ln = v_ln;  // (N, input_dim)
        //
        // BUT in the cache resize:
        //     c.x_ln = Tensor(N, D);  // D = slot_dim — WRONG!
        //     c.v_ln = Tensor(N, D);  // D = slot_dim — WRONG!
        //
        // The cache size initialization is wrong. Let me check.
        //
        // Looking at my forward: I do `c.x_ln = x_ln;` AFTER the resize, which OVERWRITES the resize.
        // And x_ln is the actual LN output (N, input_dim). So at runtime c.x_ln IS (N, input_dim).
        // Similarly c.v_ln IS (N, input_dim) at runtime.
        //
        // OK so the cache has the right shape AT RUNTIME because of the post-resize assignment.
        // Now back to the backward: c.v_ln (N, input_dim), c.x_ln (N, input_dim).
        //
        // v_proj = v_ln @ W_v^T + b_v. So:
        //   d_v_ln = d_v_proj @ W_v → (N, slot_dim) @ (slot_dim, input_dim) = (N, input_dim) ✓
        //   dL/d_W_v[j, k] += sum_i d_v_proj[i, j] * v_ln[i, k], j in [0, slot_dim), k in [0, input_dim) ✓
        //   dL/d_x_ln = d_k_proj @ W_k → (N, slot_dim) @ (slot_dim, input_dim) = (N, input_dim) ✓
        Tensor d_v_ln = d_v_proj * W_v_;

        // ===== LN_k backward =====
        // x_ln is the OUTPUT of ln_k_(input). ln_k_.last_x = input (since we called forward).
        Tensor d_x_from_k_path = ln_k_.backward(d_x_ln, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                d_x(i, j) += d_x_from_k_path(i, j);

        // ===== LN_v backward =====
        Tensor d_x_from_v_path = ln_v_.backward(d_v_ln, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                d_x(i, j) += d_x_from_v_path(i, j);

        // ===== LN_q backward =====
        // slots_ln_q is the OUTPUT of ln_q_(slots_pre_gru). ln_q_.last_x = slots_pre_gru.
        // dL/d_slots_pre_gru (from q path) = LN_q backward of d_slots_ln_q
        Tensor d_slots_pre_gru_from_q = ln_q_.backward(d_slots_ln_q, 0.0);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_pre_gru(i, j) += d_slots_pre_gru_from_q(i, j);

        // ===== Combine: d_slots at this iteration =====
        Tensor d_slots_iter(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_iter(i, j) = d_slots_pre_gru(i, j);

        // For t=0, d_slots_iter is the d_mu gradient (slot init). At t=0 slots_pre_gru = mu,
        // and we did call ln_q_.forward(slots) which set ln_q_.last_x = mu. So ln_q_.backward
        // correctly propagates the q-path gradient into d_mu.
        if (t == 0) {
            for (size_t i = 0; i < K; ++i)
                for (size_t j = 0; j < D; ++j)
                    grad_mu_(i, j) = d_slots_iter(i, j);
        }

        d_slots = d_slots_iter;
    }

    return d_x;
}

// ============================================================================
// SlotAttentionBlock
// ============================================================================

SlotAttentionBlock::SlotAttentionBlock(size_t num_slots, size_t slot_dim, size_t input_dim,
                                       size_t num_iterations, size_t hidden_dim)
    : slot_dim_(slot_dim),
      hidden_dim_(hidden_dim == 0 ? 4 * slot_dim : hidden_dim),
      ln_(slot_dim, 1e-7),
      ffn_fc1_(slot_dim, hidden_dim_ == 0 ? 4 * slot_dim : hidden_dim_),
      ffn_fc2_(hidden_dim_ == 0 ? 4 * slot_dim : hidden_dim_, slot_dim)
{
    (void)num_slots;
    (void)num_iterations;
    (void)input_dim;
}

Tensor SlotAttentionBlock::forward(const Tensor& input) {
    last_input_ = input;
    // pre-LN -> per-slot FFN -> residual
    last_ln_out_ = ln_.forward(input);
    Tensor ffn_h_pre = last_ln_out_ * ffn_fc1_.weights.transpose();
    for (size_t i = 0; i < ffn_h_pre.rows; ++i)
        for (size_t j = 0; j < ffn_fc1_.bias.cols; ++j)
            ffn_h_pre(i, j) += ffn_fc1_.bias(0, j);
    last_ffn_h_ = ffn_h_pre.apply([](double v) { return v > 0.0 ? v : 0.0; });
    Tensor ffn_out = last_ffn_h_ * ffn_fc2_.weights.transpose();
    for (size_t i = 0; i < ffn_out.rows; ++i)
        for (size_t j = 0; j < ffn_fc2_.bias.cols; ++j)
            ffn_out(i, j) += ffn_fc2_.bias(0, j);
    return input + ffn_out;
}

Tensor SlotAttentionBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // out = input + ffn_out
    // dL/d_input (from out) = grad_output
    // dL/d_ffn_out = grad_output
    Tensor d_input = grad_output.clone();
    Tensor d_ffn_out = grad_output.clone();

    // ffn_out = ReLU(ln_out @ W1^T + b1) @ W2^T + b2
    // dL/d_(ReLU out) = d_ffn_out @ W2
    Tensor d_ffn_h = d_ffn_out * ffn_fc2_.weights;
    ffn_fc2_.zero_grad();
    for (size_t i = 0; i < d_ffn_out.rows; ++i) {
        for (size_t j = 0; j < ffn_fc2_.weights.rows; ++j) {
            for (size_t k = 0; k < ffn_fc2_.weights.cols; ++k) {
                ffn_fc2_.grad_weights(j, k) += d_ffn_out(i, j) * last_ffn_h_(i, k);
            }
            ffn_fc2_.grad_bias(0, j) += d_ffn_out(i, j);
        }
    }
    // ReLU backward
    Tensor d_ffn_h_pre = d_ffn_h.hadamard(last_ffn_h_.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; }));
    ffn_fc1_.zero_grad();
    for (size_t i = 0; i < d_ffn_h_pre.rows; ++i) {
        for (size_t j = 0; j < ffn_fc1_.weights.rows; ++j) {
            for (size_t k = 0; k < ffn_fc1_.weights.cols; ++k) {
                ffn_fc1_.grad_weights(j, k) += d_ffn_h_pre(i, j) * last_ln_out_(i, k);
            }
            ffn_fc1_.grad_bias(0, j) += d_ffn_h_pre(i, j);
        }
    }
    // dL/d_ln_out = d_ffn_h_pre @ W1
    Tensor d_ln_out = d_ffn_h_pre * ffn_fc1_.weights;
    // LN backward
    Tensor d_input_from_ln = ln_.backward(d_ln_out, 0.0);
    for (size_t i = 0; i < d_input.rows; ++i)
        for (size_t j = 0; j < d_input.cols; ++j)
            d_input(i, j) += d_input_from_ln(i, j);

    return d_input;
}

Tensor SlotAttentionBlock::get_weights() const { return ffn_fc1_.get_weights(); }
Tensor SlotAttentionBlock::get_gradients() const { return ffn_fc1_.get_gradients(); }

std::vector<Tensor*> SlotAttentionBlock::parameters() {
    std::vector<Tensor*> p;
    auto l = ln_.parameters(); for (auto* x : l) p.push_back(x);
    auto f1 = ffn_fc1_.parameters(); for (auto* x : f1) p.push_back(x);
    auto f2 = ffn_fc2_.parameters(); for (auto* x : f2) p.push_back(x);
    return p;
}

std::vector<Tensor*> SlotAttentionBlock::gradients() {
    std::vector<Tensor*> g;
    auto l = ln_.gradients(); for (auto* x : l) g.push_back(x);
    auto f1 = ffn_fc1_.gradients(); for (auto* x : f1) g.push_back(x);
    auto f2 = ffn_fc2_.gradients(); for (auto* x : f2) g.push_back(x);
    return g;
}

void SlotAttentionBlock::zero_grad() {
    ln_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void SlotAttentionBlock::update_weights(double learning_rate) {
    ln_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

// ============================================================================
// SlotAttentionModel
// ============================================================================

SlotAttentionModel::SlotAttentionModel(size_t num_slots, size_t slot_dim, size_t input_dim,
                                       size_t out_dim, size_t n_blocks,
                                       size_t num_iterations, size_t hidden_dim)
    : input_proj_(input_dim, slot_dim),     // Dense(in=input_dim, out=slot_dim) → weights (slot_dim, input_dim)
      classifier_(slot_dim, out_dim)        // Dense(in=slot_dim, out=out_dim)  → weights (out_dim, slot_dim)
{
    // First, a single SlotAttention processes inputs (N, slot_dim) → slots (K, slot_dim).
    // Note: we pass slot_dim (not input_dim) as the SlotAttention's input_dim, because
    // the input_proj has already mapped the model's input_dim to slot_dim.
    // Then n_blocks refinement blocks operate on slots (K, slot_dim) → slots (K, slot_dim).
    attn_ = std::make_unique<SlotAttention>(num_slots, slot_dim, slot_dim, num_iterations, hidden_dim);
    for (size_t i = 0; i < n_blocks; ++i) {
        blocks_.emplace_back(new SlotAttentionBlock(num_slots, slot_dim, slot_dim, num_iterations, hidden_dim));
    }
    block_inputs_.resize(n_blocks);
}

Tensor SlotAttentionModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor x = input * input_proj_.weights.transpose();
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < input_proj_.bias.cols; ++j)
            x(i, j) += input_proj_.bias(0, j);
    last_slot_in_ = x;

    // SlotAttention: (N, slot_dim) → (K, slot_dim)
    last_attn_out_ = attn_->forward(x);

    // Per-slot refinement blocks: (K, slot_dim) → (K, slot_dim)
    Tensor s = last_attn_out_;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        block_inputs_[i] = s;
        s = blocks_[i]->forward(s);
    }
    last_block_out_ = s;

    // Classifier: per-slot linear → (K, out_dim)
    Tensor out = s * classifier_.weights.transpose();
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < classifier_.bias.cols; ++j)
            out(i, j) += classifier_.bias(0, j);
    return out;
}

Tensor SlotAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    // Classifier backward: dL/d_s (post-last-block) = grad_output @ classifier.W
    Tensor d_s = grad_output * classifier_.weights;
    classifier_.zero_grad();
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t j = 0; j < classifier_.weights.rows; ++j) {
            for (size_t k = 0; k < classifier_.weights.cols; ++k) {
                classifier_.grad_weights(j, k) += grad_output(i, j) * last_block_out_(i, k);
            }
            classifier_.grad_bias(0, j) += grad_output(i, j);
        }
    }

    // Backward through blocks in reverse
    for (size_t i = blocks_.size(); i-- > 0; ) {
        blocks_[i]->zero_grad();
        d_s = blocks_[i]->backward(d_s, learning_rate);
    }

    // SlotAttention backward (d_s is dL/d(last_attn_out_))
    attn_->zero_grad();
    Tensor d_slot_in = attn_->backward(d_s, learning_rate);

    // input_proj backward
    Tensor d_input_proj_input = d_slot_in * input_proj_.weights;
    input_proj_.zero_grad();
    for (size_t i = 0; i < d_slot_in.rows; ++i) {
        for (size_t j = 0; j < input_proj_.weights.rows; ++j) {
            for (size_t k = 0; k < input_proj_.weights.cols; ++k) {
                input_proj_.grad_weights(j, k) += d_slot_in(i, j) * last_input_(i, k);
            }
            input_proj_.grad_bias(0, j) += d_slot_in(i, j);
        }
    }
    return d_input_proj_input;
}

Tensor SlotAttentionModel::get_weights() const { return input_proj_.get_weights(); }
Tensor SlotAttentionModel::get_gradients() const { return input_proj_.get_gradients(); }

std::vector<Tensor*> SlotAttentionModel::parameters() {
    std::vector<Tensor*> p;
    auto ip = input_proj_.parameters();
    for (auto* x : ip) p.push_back(x);
    if (attn_) {
        auto a = attn_->parameters();
        for (auto* x : a) p.push_back(x);
    }
    for (auto& blk : blocks_) {
        auto b = blk->parameters();
        for (auto* x : b) p.push_back(x);
    }
    auto c = classifier_.parameters();
    for (auto* x : c) p.push_back(x);
    return p;
}

std::vector<Tensor*> SlotAttentionModel::gradients() {
    std::vector<Tensor*> g;
    auto ip = input_proj_.gradients();
    for (auto* x : ip) g.push_back(x);
    if (attn_) {
        auto a = attn_->gradients();
        for (auto* x : a) g.push_back(x);
    }
    for (auto& blk : blocks_) {
        auto b = blk->gradients();
        for (auto* x : b) g.push_back(x);
    }
    auto c = classifier_.gradients();
    for (auto* x : c) g.push_back(x);
    return g;
}

void SlotAttentionModel::zero_grad() {
    input_proj_.zero_grad();
    if (attn_) attn_->zero_grad();
    classifier_.zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
}

void SlotAttentionModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    if (attn_) attn_->update_weights(learning_rate);
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}
