// Gated Slot Attention — Bauer et al., NeurIPS 2024
//   "Gated Slot Attention for Object-Centric Learning"
//   (https://arxiv.org/abs/2410.23790)
//
// Implementation: see gated_slot_attention.h for the full mathematical formulation
// and a list of deviations from the paper. This file implements:
//   * GatedSlotAttention       — single gated slot-attention block (T iterations).
//   * GatedSlotAttentionModel  — input proj + stack of blocks + classifier.

#include "gated_slot_attention.h"

#include <cmath>
#include <iostream>
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

inline double sigmoid_stable(double x) {
    // Branched for numerical stability — avoids overflow in exp at |x| > 60.
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
}

inline double sigmoid_deriv_from_s(double s) {
    return s * (1.0 - s);
}

inline void init_xavier(Tensor& t, std::mt19937& gen, double scale = 1.0) {
    double bound = scale * std::sqrt(6.0 / static_cast<double>(t.rows + t.cols));
    std::uniform_real_distribution<double> dis(-bound, bound);
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            t(i, j) = dis(gen);
}

}  // namespace

// ============================================================================
// GatedSlotAttention
// ============================================================================

GatedSlotAttention::GatedSlotAttention(size_t num_slots, size_t slot_dim, size_t input_dim,
                                       size_t num_iterations, size_t hidden_dim, double epsilon)
    : num_slots_(num_slots), slot_dim_(slot_dim), input_dim_(input_dim),
      num_iterations_(num_iterations == 0 ? 1 : num_iterations),
      hidden_dim_(hidden_dim == 0 ? 4 * slot_dim : hidden_dim),
      epsilon_(epsilon),
      // Gate projection (paper §3.2)
      W_g_(input_dim, input_dim),
      b_g_(1, input_dim),
      grad_W_g_(input_dim, input_dim),
      grad_b_g_(1, input_dim),
      // Q/K/V projections — Dense(out, in) convention: weights are (out_features, in_features).
      //   W_k, W_v: out_features = slot_dim, in_features = input_dim → shape (slot_dim, input_dim)
      //   W_q: out_features = slot_dim, in_features = slot_dim     → shape (slot_dim, slot_dim)
      W_k_(slot_dim, input_dim), W_v_(slot_dim, input_dim), W_q_(slot_dim, slot_dim),
      b_k_(1, slot_dim), b_v_(1, slot_dim), b_q_(1, slot_dim),
      grad_W_k_(slot_dim, input_dim), grad_W_v_(slot_dim, input_dim), grad_W_q_(slot_dim, slot_dim),
      grad_b_k_(1, slot_dim), grad_b_v_(1, slot_dim), grad_b_q_(1, slot_dim),
      // Slot init
      mu_(num_slots, slot_dim),
      grad_mu_(num_slots, slot_dim),
      // LayerNorms
      ln_k_(input_dim, 1e-7), ln_v_(input_dim, 1e-7), ln_q_(slot_dim, 1e-7),
      ln_mlp_(slot_dim, 1e-7),
      // Residual MLP
      mlp_fc1_(slot_dim, slot_dim),
      mlp_fc2_(slot_dim, slot_dim),
      // Per-slot GRU weights
      W_zr_(2 * slot_dim, 2 * slot_dim),
      W_h_(2 * slot_dim, slot_dim),
      b_zr_(1, 2 * slot_dim),
      b_h_(1, slot_dim),
      grad_W_zr_(2 * slot_dim, 2 * slot_dim),
      grad_W_h_(2 * slot_dim, slot_dim),
      grad_b_zr_(1, 2 * slot_dim),
      grad_b_h_(1, slot_dim)
{
    if (num_slots == 0)  throw std::invalid_argument("GatedSlotAttention: num_slots must be > 0");
    if (slot_dim == 0)   throw std::invalid_argument("GatedSlotAttention: slot_dim must be > 0");
    if (input_dim == 0)   throw std::invalid_argument("GatedSlotAttention: input_dim must be > 0");
    if (num_iterations == 0) throw std::invalid_argument("GatedSlotAttention: num_iterations must be > 0");

    // Set up caches
    cache_.resize(num_iterations_);
    for (auto& c : cache_) {
        c.slots_pre_gru = Tensor(num_slots, slot_dim);
        c.slots_post_gru = Tensor(num_slots, slot_dim);
        c.slots_post_mlp = Tensor(num_slots, slot_dim);
        c.k_proj = Tensor(0, slot_dim);
        c.v_proj = Tensor(0, slot_dim);
        c.q_proj = Tensor(num_slots, slot_dim);
        c.x_ln_k = Tensor(0, input_dim);
        c.x_ln_v = Tensor(0, input_dim);
        c.x_gated_k = Tensor(0, input_dim);
        c.x_gated_v = Tensor(0, input_dim);
        c.gate = Tensor(0, input_dim);
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
    last_gate_ = Tensor(0, 0);

    // Weight init
    std::mt19937 gen(42);

    // Gate projection: small Xavier init so sigmoid is non-degenerate at init.
    init_xavier(W_g_, gen, 0.1);
    init_xavier(b_g_, gen, 0.1);

    // Q/K/V projections: zero-init first (paper recommendation for slot-attention).
    init_xavier(W_k_, gen); init_xavier(W_v_, gen); init_xavier(W_q_, gen);
    W_k_.fill(0.0); W_v_.fill(0.0); W_q_.fill(0.0);
    init_xavier(b_k_, gen, 0.1); init_xavier(b_v_, gen, 0.1); init_xavier(b_q_, gen, 0.1);

    // Slot init mu: small random init.
    std::normal_distribution<double> dnorm(0.0, 0.1);
    for (size_t i = 0; i < mu_.rows; ++i)
        for (size_t j = 0; j < mu_.cols; ++j)
            mu_(i, j) = dnorm(gen);

    // GRU weights
    init_xavier(W_zr_, gen, 0.5);
    init_xavier(W_h_, gen, 0.5);
    b_zr_.fill(0.0);
    b_h_.fill(0.0);

    zero_grad();
}

void GatedSlotAttention::zero_grad() {
    grad_W_k_.fill(0.0); grad_W_v_.fill(0.0); grad_W_q_.fill(0.0);
    grad_b_k_.fill(0.0); grad_b_v_.fill(0.0); grad_b_q_.fill(0.0);
    grad_mu_.fill(0.0);
    grad_W_g_.fill(0.0);
    grad_b_g_.fill(0.0);
    ln_k_.zero_grad(); ln_v_.zero_grad(); ln_q_.zero_grad(); ln_mlp_.zero_grad();
    mlp_fc1_.zero_grad(); mlp_fc2_.zero_grad();
    grad_W_zr_.fill(0.0); grad_W_h_.fill(0.0);
    grad_b_zr_.fill(0.0); grad_b_h_.fill(0.0);
}

std::vector<Tensor*> GatedSlotAttention::parameters() {
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
    // Gate projection — appended LAST so the test can locate it by index.
    p.push_back(&W_g_);
    p.push_back(&b_g_);
    return p;
}

std::vector<Tensor*> GatedSlotAttention::gradients() {
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
    g.push_back(&grad_W_g_);
    g.push_back(&grad_b_g_);
    return g;
}

void GatedSlotAttention::update_weights(double learning_rate) {
    auto params = parameters();
    auto grads = gradients();
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = 0; j < params[i]->data.size(); ++j) {
            params[i]->data[j] -= learning_rate * grads[i]->data[j];
        }
    }
}

// ----- helpers: row / col softmax -----
Tensor GatedSlotAttention::row_softmax(const Tensor& x) {
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

Tensor GatedSlotAttention::col_softmax(const Tensor& x) {
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
GatedSlotAttention::GruOut GatedSlotAttention::gru_forward(const Tensor& u, const Tensor& s) {
    size_t K = u.rows, D = u.cols;
    GruOut out;
    out.new_s = Tensor(K, D);
    out.z = Tensor(K, D);
    out.r = Tensor(K, D);
    out.s_hat = Tensor(K, D);
    out.rh = Tensor(K, D);

    // Compute z, r
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            double zr_pre = b_zr_(0, j);
            double rr_pre = b_zr_(0, D + j);
            for (size_t k = 0; k < D; ++k) {
                zr_pre += u(i, k) * W_zr_(k, j);
                rr_pre += u(i, k) * W_zr_(k, D + j);
            }
            for (size_t k = 0; k < D; ++k) {
                zr_pre += s(i, k) * W_zr_(D + k, j);
                rr_pre += s(i, k) * W_zr_(D + k, D + j);
            }
            out.z(i, j) = sigmoid_stable(zr_pre);
            out.r(i, j) = sigmoid_stable(rr_pre);
            out.rh(i, j) = out.r(i, j) * s(i, j);
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

void GatedSlotAttention::gru_backward(const Tensor& grad_new_s,
                                       const Tensor& u, const Tensor& s,
                                       const GruOut& st,
                                       Tensor& grad_u, Tensor& grad_s) {
    size_t K = grad_new_s.rows, D = grad_new_s.cols;
    grad_u = Tensor(K, D);
    grad_s = Tensor(K, D);

    // dz[i,j] = grad_new_s[i,j] * (s_hat - s)
    // ds_hat[i,j] = grad_new_s[i,j] * z[i,j]
    Tensor dz(K, D), ds_hat(K, D);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            dz(i, j) = grad_new_s(i, j) * (st.s_hat(i, j) - s(i, j));
            ds_hat(i, j) = grad_new_s(i, j) * st.z(i, j);
        }
    }

    // d_tanh_pre[i,j] = ds_hat[i,j] * (1 - s_hat[i,j]^2)
    Tensor d_tanh_pre(K, D);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j)
            d_tanh_pre(i, j) = ds_hat(i, j) * (1.0 - st.s_hat(i, j) * st.s_hat(i, j));

    // drh[i,j] = sum_l d_tanh_pre[i,l] * W_h[D+j, l]
    Tensor drh(K, D);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j) {
            double acc = 0.0;
            for (size_t l = 0; l < D; ++l) acc += d_tanh_pre(i, l) * W_h_(D + j, l);
            drh(i, j) = acc;
        }

    // dr[i,j] (via rh) = drh[i,j] * s[i,j], then * sigmoid_deriv(r)
    Tensor dr(K, D);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j)
            dr(i, j) = drh(i, j) * s(i, j) * sigmoid_deriv_from_s(st.r(i, j));

    // grad_z_pre = dz * sd_z, grad_r_pre = dr * sd_r
    Tensor grad_z_pre(K, D), grad_r_pre(K, D);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < D; ++j) {
            grad_z_pre(i, j) = dz(i, j) * sigmoid_deriv_from_s(st.z(i, j));
            grad_r_pre(i, j) = dr(i, j) * sigmoid_deriv_from_s(st.r(i, j));
        }

    // Accumulate grad_W_zr_, grad_W_h_, grad_b_zr_, grad_b_h_
    // For W_zr (2D, 2D): rows 0..D-1 take u input, rows D..2D-1 take s input
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < D; ++j) {
            // z branch (col j)
            double gz = grad_z_pre(i, j);
            for (size_t k = 0; k < D; ++k) {
                grad_W_zr_(k, j)     += gz * u(i, k);
                grad_W_zr_(D + k, j) += gz * s(i, k);
            }
            // r branch (col D+j)
            double gr = grad_r_pre(i, j);
            for (size_t k = 0; k < D; ++k) {
                grad_W_zr_(k, D + j)     += gr * u(i, k);
                grad_W_zr_(D + k, D + j) += gr * s(i, k);
            }
        }
    }
    for (size_t i = 0; i < D; ++i) {
        double zb = 0.0, rb = 0.0;
        for (size_t k = 0; k < K; ++k) {
            zb += grad_z_pre(k, i);
            rb += grad_r_pre(k, i);
        }
        grad_b_zr_(0, i)     += zb;
        grad_b_zr_(0, D + i) += rb;
    }

    // W_h: rows 0..D-1 take u, rows D..2D-1 take rh
    for (size_t j = 0; j < D; ++j) {
        double bs = 0.0;
        for (size_t i = 0; i < K; ++i) {
            double d = d_tanh_pre(i, j);
            bs += d;
            for (size_t k = 0; k < D; ++k) {
                grad_W_h_(k, j)     += d * u(i, k);
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
            // Path through rh -> s (rh = r * s, dL/drh → r)
            gs += drh(i, a) * st.r(i, a);
            // Direct (1-z)*s contribution to s
            gs += grad_new_s(i, a) * (1.0 - st.z(i, a));

            grad_u(i, a) = gu;
            grad_s(i, a) = gs;
        }
    }
}

// ----- Forward -----
Tensor GatedSlotAttention::forward(const Tensor& input) {
    size_t N = input.rows;
    size_t D = slot_dim_;
    size_t K = num_slots_;

    last_input_ = input;

    // Resize cache for this N
    for (auto& c : cache_) {
        c.k_proj = Tensor(N, D);
        c.v_proj = Tensor(N, D);
        c.q_proj = Tensor(K, D);
        c.x_ln_k = Tensor(N, input_dim_);
        c.x_ln_v = Tensor(N, input_dim_);
        c.x_gated_k = Tensor(N, input_dim_);
        c.x_gated_v = Tensor(N, input_dim_);
        c.gate = Tensor(N, input_dim_);
        c.logits = Tensor(K, N);
        c.attn1 = Tensor(K, N);
        c.attn2 = Tensor(K, N);
    }

    // LayerNorms of input (same at every iter)
    Tensor x_ln_k = ln_k_.forward(input);  // (N, input_dim)
    Tensor x_ln_v = ln_v_.forward(input);  // (N, input_dim)

    // Compute gate g = σ(W_g · x + b_g) ONCE — input doesn't change across iterations.
    // gate: (N, input_dim)
    Tensor gate = Tensor(N, input_dim_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < input_dim_; ++j) {
            double v = b_g_(0, j);
            for (size_t k = 0; k < input_dim_; ++k) v += input(i, k) * W_g_(k, j);
            gate(i, j) = sigmoid_stable(v);
        }
    }
    last_gate_ = gate;

    // Gated versions
    Tensor x_gated_k = Tensor(N, input_dim_);
    Tensor x_gated_v = Tensor(N, input_dim_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < input_dim_; ++j) {
            x_gated_k(i, j) = gate(i, j) * x_ln_k(i, j);
            x_gated_v(i, j) = gate(i, j) * x_ln_v(i, j);
        }

    Tensor slots = mu_.clone();
    double scale = 1.0 / std::sqrt(static_cast<double>(D));

    for (size_t t = 0; t < num_iterations_; ++t) {
        IterCache& c = cache_[t];

        // Slots LN_q + W_q + b_q
        Tensor slots_ln_q = ln_q_.forward(slots);
        c.slots_ln_q = slots_ln_q;
        c.q_proj = slots_ln_q * W_q_.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                c.q_proj(i, j) += b_q_(0, j);

        // k_proj = x_gated_k @ W_k^T + b_k
        c.x_ln_k = x_ln_k;
        c.x_ln_v = x_ln_v;
        c.gate = gate;
        c.x_gated_k = x_gated_k;
        c.x_gated_v = x_gated_v;
        c.k_proj = x_gated_k * W_k_.transpose();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                c.k_proj(i, j) += b_k_(0, j);

        // v_proj = x_gated_v @ W_v^T + b_v
        c.v_proj = x_gated_v * W_v_.transpose();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < D; ++j)
                c.v_proj(i, j) += b_v_(0, j);

        // logits = q_proj @ k_proj^T * scale
        Tensor qk_t = c.q_proj * c.k_proj.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < N; ++j)
                c.logits(i, j) = qk_t(i, j) * scale;

        // Double softmax
        c.attn1 = col_softmax(c.logits);
        c.attn2 = row_softmax(c.attn1);

        // updates = attn2 @ v_proj
        c.updates = c.attn2 * c.v_proj;

        // Cache slots pre-GRU
        c.slots_pre_gru = slots.clone();

        // GRU
        GruOut g = gru_forward(c.updates, slots);
        c.z_gates = g.z;
        c.r_gates = g.r;
        c.s_hat = g.s_hat;
        c.rh = g.rh;
        c.slots_post_gru = g.new_s;

        // Residual MLP
        c.slots_ln_mlp = ln_mlp_.forward(c.slots_post_gru);
        Tensor mlp_pre = c.slots_ln_mlp * mlp_fc1_.weights.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                mlp_pre(i, j) += mlp_fc1_.bias(0, j);
        c.mlp_h = mlp_pre.apply([](double v) { return v > 0.0 ? v : 0.0; });
        Tensor mlp_out = c.mlp_h * mlp_fc2_.weights.transpose();
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                mlp_out(i, j) += mlp_fc2_.bias(0, j);
        c.slots_post_mlp = c.slots_post_gru + mlp_out;
        slots = c.slots_post_mlp;
    }

    return slots;
}

// ----- Backward -----
Tensor GatedSlotAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t N = last_input_.rows;
    size_t D = slot_dim_;
    size_t K = num_slots_;
    double scale = 1.0 / std::sqrt(static_cast<double>(D));

    Tensor d_slots = grad_output.clone();

    // Zero param grads
    grad_W_q_.fill(0.0); grad_W_k_.fill(0.0); grad_W_v_.fill(0.0);
    grad_b_q_.fill(0.0); grad_b_k_.fill(0.0); grad_b_v_.fill(0.0);
    grad_mu_.fill(0.0);
    grad_W_g_.fill(0.0);
    grad_b_g_.fill(0.0);
    ln_k_.zero_grad(); ln_v_.zero_grad(); ln_q_.zero_grad(); ln_mlp_.zero_grad();
    mlp_fc1_.zero_grad(); mlp_fc2_.zero_grad();
    grad_W_zr_.fill(0.0); grad_W_h_.fill(0.0);
    grad_b_zr_.fill(0.0); grad_b_h_.fill(0.0);

    // Accumulator for input gradient
    Tensor d_x(N, input_dim_);
    d_x.fill(0.0);

    // Iterate from T-1 down to 0
    for (int t = static_cast<int>(num_iterations_) - 1; t >= 0; --t) {
        IterCache& c = cache_[t];

        // ===== Residual MLP backward =====
        Tensor d_mlp_out = d_slots.clone();
        Tensor d_slots_post_gru = d_slots.clone();

        // mlp_out = mlp_h @ W2^T + b2
        Tensor d_mlp_h = d_mlp_out * mlp_fc2_.weights;
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    mlp_fc2_.grad_weights(j, k) += d_mlp_out(i, j) * c.mlp_h(i, k);
                }
                mlp_fc2_.grad_bias(0, j) += d_mlp_out(i, j);
            }
        }
        // ReLU backward
        Tensor d_mlp_pre = d_mlp_h.hadamard(c.mlp_h.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; }));
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    mlp_fc1_.grad_weights(j, k) += d_mlp_pre(i, j) * c.slots_ln_mlp(i, k);
                }
                mlp_fc1_.grad_bias(0, j) += d_mlp_pre(i, j);
            }
        }
        Tensor d_slots_ln_mlp = d_mlp_pre * mlp_fc1_.weights;
        // LN_mlp backward
        Tensor d_x_from_ln = ln_mlp_.backward(d_slots_ln_mlp, 0.0);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_post_gru(i, j) += d_x_from_ln(i, j);

        // ===== GRU backward =====
        Tensor d_updates_from_gru, d_slots_pre_gru;
        gru_backward(d_slots_post_gru, c.updates, c.slots_pre_gru,
                     {c.slots_post_gru, c.z_gates, c.r_gates, c.s_hat, c.rh},
                     d_updates_from_gru, d_slots_pre_gru);

        // ===== Updates = attn2 @ v_proj =====
        Tensor d_attn2 = d_updates_from_gru * c.v_proj.transpose();
        Tensor d_v_proj = c.attn2.transpose() * d_updates_from_gru;

        // ===== Double softmax backward =====
        Tensor d_attn1(K, N);
        for (size_t i = 0; i < K; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < N; ++j) dot += d_attn2(i, j) * c.attn2(i, j);
            for (size_t j = 0; j < N; ++j) {
                d_attn1(i, j) = c.attn2(i, j) * (d_attn2(i, j) - dot);
            }
        }
        Tensor d_logits(K, N);
        for (size_t j = 0; j < N; ++j) {
            double col_dot = 0.0;
            for (size_t i = 0; i < K; ++i) col_dot += c.attn1(i, j) * d_attn1(i, j);
            for (size_t i = 0; i < K; ++i) {
                d_logits(i, j) = c.attn1(i, j) * (d_attn1(i, j) - col_dot);
            }
        }
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < N; ++j)
                d_logits(i, j) *= scale;

        // ===== q_proj / k_proj / v_proj backward =====
        Tensor d_q_proj = d_logits * c.k_proj;
        Tensor d_k_proj = d_logits.transpose() * c.q_proj;

        // ===== W_q / W_k / W_v / b_q / b_k / b_v grads =====
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    grad_W_q_(j, k) += d_q_proj(i, j) * c.slots_ln_q(i, k);
                }
                grad_b_q_(0, j) += d_q_proj(i, j);
            }
        }
        Tensor d_slots_ln_q = d_q_proj * W_q_;

        // k_proj = x_gated_k @ W_k^T + b_k  → grad_W_k[j, k] += sum_i d_k_proj[i, j] * x_gated_k[i, k]
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < input_dim_; ++k) {
                    grad_W_k_(j, k) += d_k_proj(i, j) * c.x_gated_k(i, k);
                }
                grad_b_k_(0, j) += d_k_proj(i, j);
            }
        }
        // d_x_gated_k = d_k_proj @ W_k  (N, input_dim)
        Tensor d_x_gated_k = d_k_proj * W_k_;

        // v_proj = x_gated_v @ W_v^T + b_v
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < D; ++j) {
                for (size_t k = 0; k < input_dim_; ++k) {
                    grad_W_v_(j, k) += d_v_proj(i, j) * c.x_gated_v(i, k);
                }
                grad_b_v_(0, j) += d_v_proj(i, j);
            }
        }
        // d_x_gated_v = d_v_proj @ W_v  (N, input_dim)
        Tensor d_x_gated_v = d_v_proj * W_v_;

        // ===== Gate chain (paper §3.2 — the GSA-specific path) =====
        // d_gate[i, j] = d_x_gated_k[i, j] * x_ln_k[i, j] + d_x_gated_v[i, j] * x_ln_v[i, j]
        Tensor d_gate(N, input_dim_);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < input_dim_; ++j) {
                d_gate(i, j) = d_x_gated_k(i, j) * c.x_ln_k(i, j)
                             + d_x_gated_v(i, j) * c.x_ln_v(i, j);
            }
        }
        // d_pre_sig = d_gate * gate * (1 - gate)  (sigmoid derivative)
        Tensor d_pre_sig(N, input_dim_);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < input_dim_; ++j) {
                d_pre_sig(i, j) = d_gate(i, j) * c.gate(i, j) * (1.0 - c.gate(i, j));
            }
        }
        // dW_g[k, j] += sum_i d_pre_sig[i, j] * input[i, k]  (W_g is (input_dim, input_dim))
        for (size_t k = 0; k < input_dim_; ++k) {
            for (size_t j = 0; j < input_dim_; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i) acc += d_pre_sig(i, j) * last_input_(i, k);
                grad_W_g_(k, j) += acc;
            }
        }
        // db_g[j] += sum_i d_pre_sig[i, j]
        for (size_t j = 0; j < input_dim_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < N; ++i) acc += d_pre_sig(i, j);
            grad_b_g_(0, j) += acc;
        }
        // dx_from_gate += d_pre_sig @ W_g  (W_g is (input_dim, input_dim), d_pre_sig is (N, input_dim))
        Tensor d_x_from_gate = d_pre_sig * W_g_;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input_dim_; ++j)
                d_x(i, j) += d_x_from_gate(i, j);

        // ===== LN_k backward (input gate ⊙ LN_k(x) so d_x_gated → x_ln via ⊙ gate) =====
        // d_LN_k_out (i.e., x_ln_k) = d_x_gated_k ⊙ gate
        Tensor d_x_ln_k(N, input_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input_dim_; ++j)
                d_x_ln_k(i, j) = d_x_gated_k(i, j) * c.gate(i, j);
        Tensor d_x_from_k_path = ln_k_.backward(d_x_ln_k, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input_dim_; ++j)
                d_x(i, j) += d_x_from_k_path(i, j);

        // ===== LN_v backward =====
        Tensor d_x_ln_v(N, input_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input_dim_; ++j)
                d_x_ln_v(i, j) = d_x_gated_v(i, j) * c.gate(i, j);
        Tensor d_x_from_v_path = ln_v_.backward(d_x_ln_v, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input_dim_; ++j)
                d_x(i, j) += d_x_from_v_path(i, j);

        // ===== LN_q backward =====
        Tensor d_slots_pre_gru_from_q = ln_q_.backward(d_slots_ln_q, 0.0);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_pre_gru(i, j) += d_slots_pre_gru_from_q(i, j);

        // ===== Combine: d_slots at this iteration =====
        Tensor d_slots_iter(K, D);
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < D; ++j)
                d_slots_iter(i, j) = d_slots_pre_gru(i, j);

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
// GatedSlotAttentionModel::GsaBlock (per-slot FFN refinement block)
// ============================================================================

GatedSlotAttentionModel::GsaBlock::GsaBlock(size_t slot_dim, size_t hidden_dim)
    : hidden_dim_(hidden_dim == 0 ? 4 * slot_dim : hidden_dim),
      ln_(slot_dim, 1e-7),
      fc1_(slot_dim, hidden_dim_),
      fc2_(hidden_dim_, slot_dim) {}

Tensor GatedSlotAttentionModel::GsaBlock::forward(const Tensor& slots) {
    last_input_ = slots;
    last_ln_out_ = ln_.forward(slots);
    Tensor h_pre = last_ln_out_ * fc1_.weights.transpose();
    for (size_t i = 0; i < h_pre.rows; ++i)
        for (size_t j = 0; j < fc1_.bias.cols; ++j)
            h_pre(i, j) += fc1_.bias(0, j);
    last_ffn_h_ = h_pre.apply([](double v) { return v > 0.0 ? v : 0.0; });
    Tensor out = last_ffn_h_ * fc2_.weights.transpose();
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < fc2_.bias.cols; ++j)
            out(i, j) += fc2_.bias(0, j);
    return slots + out;
}

Tensor GatedSlotAttentionModel::GsaBlock::backward(const Tensor& grad_output) {
    Tensor d_input = grad_output.clone();
    Tensor d_ffn_out = grad_output.clone();

    Tensor d_h = d_ffn_out * fc2_.weights;
    fc2_.zero_grad();
    for (size_t i = 0; i < d_ffn_out.rows; ++i) {
        for (size_t j = 0; j < fc2_.weights.rows; ++j) {
            for (size_t k = 0; k < fc2_.weights.cols; ++k) {
                fc2_.grad_weights(j, k) += d_ffn_out(i, j) * last_ffn_h_(i, k);
            }
            fc2_.grad_bias(0, j) += d_ffn_out(i, j);
        }
    }
    Tensor d_h_pre = d_h.hadamard(last_ffn_h_.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; }));
    fc1_.zero_grad();
    for (size_t i = 0; i < d_h_pre.rows; ++i) {
        for (size_t j = 0; j < fc1_.weights.rows; ++j) {
            for (size_t k = 0; k < fc1_.weights.cols; ++k) {
                fc1_.grad_weights(j, k) += d_h_pre(i, j) * last_ln_out_(i, k);
            }
            fc1_.grad_bias(0, j) += d_h_pre(i, j);
        }
    }
    Tensor d_ln_out = d_h_pre * fc1_.weights;
    Tensor d_input_from_ln = ln_.backward(d_ln_out, 0.0);
    for (size_t i = 0; i < d_input.rows; ++i)
        for (size_t j = 0; j < d_input.cols; ++j)
            d_input(i, j) += d_input_from_ln(i, j);
    return d_input;
}

void GatedSlotAttentionModel::GsaBlock::zero_grad() {
    ln_.zero_grad();
    fc1_.zero_grad();
    fc2_.zero_grad();
}

void GatedSlotAttentionModel::GsaBlock::update_weights(double lr) {
    ln_.update_weights(lr);
    fc1_.update_weights(lr);
    fc2_.update_weights(lr);
}

std::vector<Tensor*> GatedSlotAttentionModel::GsaBlock::parameters() {
    std::vector<Tensor*> p;
    auto l = ln_.parameters(); for (auto* x : l) p.push_back(x);
    auto f1 = fc1_.parameters(); for (auto* x : f1) p.push_back(x);
    auto f2 = fc2_.parameters(); for (auto* x : f2) p.push_back(x);
    return p;
}

std::vector<Tensor*> GatedSlotAttentionModel::GsaBlock::gradients() {
    std::vector<Tensor*> g;
    auto g1 = ln_.gradients();   for (auto* x : g1) g.push_back(x);
    auto g2 = fc1_.gradients();  for (auto* x : g2) g.push_back(x);
    auto g3 = fc2_.gradients();  for (auto* x : g3) g.push_back(x);
    return g;
}

// ============================================================================
// GatedSlotAttentionModel
// ============================================================================

GatedSlotAttentionModel::GatedSlotAttentionModel(size_t input_dim, size_t slot_dim, size_t num_slots,
                                                  size_t hidden_dim, size_t output_dim,
                                                  size_t n_blocks, size_t num_iterations)
    : input_proj_(input_dim, slot_dim),
      classifier_(slot_dim, output_dim)
{
    attn_ = std::make_unique<GatedSlotAttention>(num_slots, slot_dim, slot_dim, num_iterations, hidden_dim);
    for (size_t i = 0; i < n_blocks; ++i) {
        blocks_.emplace_back(new GsaBlock(slot_dim, hidden_dim));
    }
    block_inputs_.resize(n_blocks);
}

Tensor GatedSlotAttentionModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor x = input * input_proj_.weights.transpose();
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < input_proj_.bias.cols; ++j)
            x(i, j) += input_proj_.bias(0, j);
    last_slot_in_ = x;

    last_attn_out_ = attn_->forward(x);

    Tensor s = last_attn_out_;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        block_inputs_[i] = s;
        s = blocks_[i]->forward(s);
    }
    last_block_out_ = s;

    Tensor out = s * classifier_.weights.transpose();
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < classifier_.bias.cols; ++j)
            out(i, j) += classifier_.bias(0, j);
    return out;
}

Tensor GatedSlotAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    // Classifier backward
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

    // Backward through blocks
    for (size_t i = blocks_.size(); i-- > 0; ) {
        blocks_[i]->zero_grad();
        d_s = blocks_[i]->backward(d_s);
    }

    // SlotAttention backward
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

std::vector<Tensor*> GatedSlotAttentionModel::parameters() {
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

std::vector<Tensor*> GatedSlotAttentionModel::gradients() {
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

void GatedSlotAttentionModel::zero_grad() {
    input_proj_.zero_grad();
    if (attn_) attn_->zero_grad();
    classifier_.zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
}

void GatedSlotAttentionModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    if (attn_) attn_->update_weights(learning_rate);
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}
