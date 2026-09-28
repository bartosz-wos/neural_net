#include "linear_attention.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Linear Attention — Katharopoulos et al. 2020
//   "Transformers are RNNs: Fast Autoregressive Transformers with Linear
//    Attention" (ICML 2020, https://arxiv.org/abs/2006.16236)
// ============================================================================
//
// This file implements the canonical kernel-feature-map linear attention from
// §3.1 of the paper. The math is:
//
//   φ(x)    = elu(x) + 1                              (the §3.1 feature map)
//   KV      = φ(K)^T V                                (d_model × d_model)
//   Z       = φ(K)^T 1_n                              (d_model,)
//   context = (φ(Q) @ KV) / (φ(Q) @ Z + eps_safe)    (n × d_model)
//
// with Q = x W_q^T + b_q, K = x W_k^T + b_k, V = x W_v^T + b_v, and
// out = context W_o^T + b_o. The forward is O(n · d_model^2) — linear in n.
//
// All gradients are derived analytically below and verified by finite
// differences in the test suite.

// ----------------------------------------------------------------------------
// Deterministic init RNG (matches the rest of this repo's attention layers)
// ----------------------------------------------------------------------------
static std::mt19937& linear_attention_global_rng() {
    static std::mt19937 gen(42);
    return gen;
}

// elu(x) + 1, vectorised over a (rows, cols) tensor.
// elu(x) = x if x > 0, exp(x) - 1 otherwise.
static inline double elu_plus_one(double x) {
    return x > 0.0 ? (x + 1.0) : (std::exp(x));
}

// derivative of elu(x) + 1: 1 if x > 0, exp(x) otherwise.
static inline double elu_plus_one_deriv(double x) {
    return x > 0.0 ? 1.0 : (std::exp(x));
}

// ----------------------------------------------------------------------------
// LinearAttention
// ----------------------------------------------------------------------------
LinearAttention::LinearAttention(size_t d_model)
    : d_model_(d_model),
      eps_safe_(1e-6),
      W_q(d_model, d_model), W_k(d_model, d_model),
      W_v(d_model, d_model), W_o(d_model, d_model),
      b_q(1, d_model), b_k(1, d_model),
      b_v(1, d_model), b_o(1, d_model),
      grad_W_q(d_model, d_model), grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model), grad_W_o(d_model, d_model),
      grad_b_q(1, d_model), grad_b_k(1, d_model),
      grad_b_v(1, d_model), grad_b_o(1, d_model),
      has_forward_cache_(false) {

    if (d_model == 0) {
        throw std::invalid_argument("LinearAttention: d_model must be > 0");
    }

    // Xavier-uniform init on Q/K/V/O; biases zero.
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    const double bound = std::sqrt(6.0 / static_cast<double>(d_model + d_model));
    for (size_t i = 0; i < W_q.rows; ++i)
        for (size_t j = 0; j < W_q.cols; ++j) {
            W_q(i, j) = bound * dis(linear_attention_global_rng());
            W_k(i, j) = bound * dis(linear_attention_global_rng());
            W_v(i, j) = bound * dis(linear_attention_global_rng());
            W_o(i, j) = bound * dis(linear_attention_global_rng());
        }
    b_q.fill(0.0); b_k.fill(0.0); b_v.fill(0.0); b_o.fill(0.0);

    // Zero-init gradients
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0);
    grad_b_v.fill(0.0); grad_b_o.fill(0.0);
}

Tensor LinearAttention::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("LinearAttention: input.cols != d_model_");
    }
    const size_t n = input.rows;

    // Cache input for backward
    last_input_ = input.clone();
    has_forward_cache_ = true;

    // Projections: y = x W^T + b. Tensor::operator* is matmul.
    last_Q_ = (input * W_q.transpose());
    for (size_t j = 0; j < d_model_; ++j) {
        double bj = b_q(0, j);
        for (size_t i = 0; i < n; ++i) last_Q_(i, j) += bj;
    }
    last_K_ = (input * W_k.transpose());
    for (size_t j = 0; j < d_model_; ++j) {
        double bj = b_k(0, j);
        for (size_t i = 0; i < n; ++i) last_K_(i, j) += bj;
    }
    last_V_ = (input * W_v.transpose());
    for (size_t j = 0; j < d_model_; ++j) {
        double bj = b_v(0, j);
        for (size_t i = 0; i < n; ++i) last_V_(i, j) += bj;
    }

    // φ(Q), φ(K) — feature maps
    last_phiQ_ = Tensor(n, d_model_);
    last_phiK_ = Tensor(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_phiQ_(i, j) = elu_plus_one(last_Q_(i, j));
            last_phiK_(i, j) = elu_plus_one(last_K_(i, j));
        }
    }

    // KV = φ(K)^T V  ∈ R^{d_model × d_model}
    last_KV_ = last_phiK_.transpose() * last_V_;

    // Z = φ(K)^T 1  ∈ R^{d_model}
    last_Z_ = Tensor(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += last_phiK_(i, j);
        last_Z_(0, j) = s;
    }

    // numerator = φ(Q) @ KV    ∈ R^{n × d_model}
    Tensor numerator = last_phiQ_ * last_KV_;

    // denominator = φ(Q) @ Z   ∈ R^{n}  → broadcast across columns
    Tensor denominator(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < d_model_; ++j) s += last_phiQ_(i, j) * last_Z_(0, j);
        denominator(i, 0) = s + eps_safe_;
    }

    // context = numerator / denominator  (per-row divide — broadcast column-wise)
    last_context_ = Tensor(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        double inv = 1.0 / denominator(i, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            last_context_(i, j) = numerator(i, j) * inv;
        }
    }

    // Output projection: out = context W_o^T + b_o
    Tensor out = last_context_ * W_o.transpose();
    for (size_t j = 0; j < d_model_; ++j) {
        double bj = b_o(0, j);
        for (size_t i = 0; i < n; ++i) out(i, j) += bj;
    }
    return out;
}

Tensor LinearAttention::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (!has_forward_cache_) {
        throw std::logic_error("LinearAttention::backward called without forward cache");
    }
    const size_t n = last_input_.rows;

    // --------------------------------------------------------------------
    // Step 1 — gradient of output projection (context W_o^T + b_o)
    // --------------------------------------------------------------------
    // out[i, j] = sum_k context[i, k] · W_o^T[k, j] + b_o[j]
    //           = sum_k context[i, k] · W_o[j, k]      + b_o[j]
    //
    // d(out[i, j]) / d(W_o[a, b]) = context[i, b] · δ(j == a)  ⇒  dW_o[a, b] = sum_i dL/d(out[i, a]) · context[i, b]
    // i.e.  grad_W_o = grad_output^T @ context              (d × d)
    //
    // grad_context[i, j] = sum_k grad_output[i, k] · W_o[k, j]
    // i.e.  grad_context = grad_output @ W_o              (n × d)
    Tensor grad_context = grad_output * W_o;
    grad_W_o = grad_output.transpose() * last_context_;  // (d,n) @ (n,d) = (d,d)
    // grad_b_o = Σ_i grad_output(i, :)  (broadcast over rows)
    grad_b_o.fill(0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_b_o(0, j) += grad_output(i, j);

    // --------------------------------------------------------------------
    // Step 2 — gradient of the per-row divide  context = numerator / denom
    // --------------------------------------------------------------------
    // Let u_i = denom_i (scalar per row), c_i = numerator[i, :] (row vec).
    // context[i, :] = c_i / u_i
    // dL/dc_i = g_i / u_i
    // dL/du_i = -Σ_j g_{ij} · c_{ij} / u_i²   (derivative of (n/u) w.r.t. u is -n/u²)
    //
    // We reconstruct u_i = φ(Q)[i, ·] · Z + eps_safe  (column-1 of denom)
    // And we have numerator[i, :] cached implicitly via the context * u_i
    // relationship: numerator = context · u  (broadcast).
    Tensor grad_numer = Tensor(n, d_model_);
    Tensor grad_denom_row = Tensor(n, 1);  // gradient w.r.t. u_i (per row)
    for (size_t i = 0; i < n; ++i) {
        double u = 0.0;
        for (size_t j = 0; j < d_model_; ++j) u += last_phiQ_(i, j) * last_Z_(0, j);
        u += eps_safe_;
        double inv_u = 1.0 / u;
        double inv_u2 = inv_u * inv_u;
        double dot_g_numer = 0.0;
        for (size_t j = 0; j < d_model_; ++j) {
            grad_numer(i, j) = grad_context(i, j) * inv_u;
            // numerator[i, j] = context[i, j] * u  (reconstruct from cache)
            double numer_ij = last_context_(i, j) * u;
            dot_g_numer += grad_context(i, j) * numer_ij;
        }
        // dL/du_i = -sum_j g_ij · numerator_ij / u²
        grad_denom_row(i, 0) = -dot_g_numer * inv_u2;
    }

    // --------------------------------------------------------------------
    // Step 3 — gradient of numerator = φ(Q) @ KV
    // --------------------------------------------------------------------
    // numerator[i, j] = sum_k φ(Q)[i, k] · KV[k, j]
    // dφ(Q)[i, k]  += sum_j grad_numer[i, j] · KV[k, j]   →  grad_phiQ_num = grad_numer @ KV^T
    // dKV[k, j]    += sum_i φ(Q)[i, k] · grad_numer[i, j] →  grad_KV      = φ(Q)^T  @ grad_numer
    Tensor grad_phiQ_num = grad_numer * last_KV_.transpose();        // (n,d) @ (d,d) = (n,d)
    Tensor grad_KV       = last_phiQ_.transpose() * grad_numer;       // (d,n) @ (n,d) = (d,d)

    // --------------------------------------------------------------------
    // Step 4 — gradient of denominator = φ(Q) · Z  +  eps_safe
    // --------------------------------------------------------------------
    // dZ     = φ(Q)^T @ grad_denom_row     (d,)
    // dφQ   += grad_denom_row @ Z^T       (n × d) — broadcast across columns
    Tensor grad_Z_from_denom = Tensor(1, d_model_);
    for (size_t j = 0; j < d_model_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += last_phiQ_(i, j) * grad_denom_row(i, 0);
        grad_Z_from_denom(0, j) = s;
    }
    // Broadcast grad_denom_row * Z   (n × 1) * (1 × d) → (n × d)
    Tensor grad_phiQ_denom(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_phiQ_denom(i, j) = grad_denom_row(i, 0) * last_Z_(0, j);

    // Combined φQ gradient
    Tensor grad_phiQ = grad_phiQ_num + grad_phiQ_denom;

    // --------------------------------------------------------------------
    // Step 5 — gradient of KV = φ(K)^T V  AND  Z = φ(K)^T 1
    // --------------------------------------------------------------------
    // KV[k, j] = sum_i φ(K)[i, k] · V[i, j]
    // dV[i, j]   += sum_k φ(K)[i, k] · grad_KV[k, j]    → dV = φ(K) @ grad_KV
    // dφ(K)[i, k] += sum_j V[i, j] · grad_KV[k, j]      → dφ(K) = V @ grad_KV
    Tensor grad_V              = last_phiK_ * grad_KV;            // (n,d) @ (d,d) = (n,d)
    Tensor grad_phiK_KV_branch = last_V_ * grad_KV.transpose();  // (n,d) @ (d,d) = (n,d)  [via V @ grad_KV]
    // (Note: V @ grad_KV gives dφ(K) contribution from the KV term; via
    //  matmul, last_V_ * grad_KV.transpose() = V · grad_KV^T  which expands
    //  as sum_j V[i,j] · grad_KV[k,j] — exactly dφ(K)[i,k] = sum_j V[i,j] · grad_KV[k,j].)
    Tensor grad_phiK_Z_branch = Tensor(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_phiK_Z_branch(i, j) = grad_Z_from_denom(0, j);

    Tensor grad_phiK = grad_phiK_KV_branch + grad_phiK_Z_branch;

    // --------------------------------------------------------------------
    // Step 6 — back through the φ(·) = elu(·)+1 feature map
    // --------------------------------------------------------------------
    // dQ = grad_phiQ ⊙ φ'(Q)   where φ'(Q)[i,j] = 1 if Q[i,j] > 0 else exp(Q[i,j])
    // dK = grad_phiK ⊙ φ'(K)
    Tensor grad_Q(n, d_model_), grad_K(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_Q(i, j) = grad_phiQ(i, j) * elu_plus_one_deriv(last_Q_(i, j));
            grad_K(i, j) = grad_phiK(i, j) * elu_plus_one_deriv(last_K_(i, j));
        }
    }

    // --------------------------------------------------------------------
    // Step 7 — back through Q/K/V projections (Q = x W^T + b)
    // --------------------------------------------------------------------
    // grad_W_q = grad_Q^T @ x   (d × n) @ (n × d) → (d × d)
    // grad_W_k = grad_K^T @ x
    // grad_W_v = grad_V^T @ x
    // grad_input = grad_Q @ W_q + grad_K @ W_k + grad_V @ W_v   (all broadcast to (n × d))
    // grad_b_q = Σ_i grad_Q[i, :]
    // grad_b_k = Σ_i grad_K[i, :]
    // grad_b_v = Σ_i grad_V[i, :]
    grad_W_q = grad_Q.transpose() * last_input_;
    grad_W_k = grad_K.transpose() * last_input_;
    grad_W_v = grad_V.transpose() * last_input_;

    Tensor grad_input = grad_Q * W_q;
    Tensor grad_input_k = grad_K * W_k;
    Tensor grad_input_v = grad_V * W_v;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_input(i, j) += grad_input_k(i, j) + grad_input_v(i, j);

    grad_b_q.fill(0.0);
    grad_b_k.fill(0.0);
    grad_b_v.fill(0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j) {
            grad_b_q(0, j) += grad_Q(i, j);
            grad_b_k(0, j) += grad_K(i, j);
            grad_b_v(0, j) += grad_V(i, j);
        }

    return grad_input;
}

void LinearAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W(i, j) -= learning_rate * gW(i, j);
    };
    auto sgd_b = [&](Tensor& b, Tensor& gb) {
        for (size_t j = 0; j < b.cols; ++j) b(0, j) -= learning_rate * gb(0, j);
    };
    sgd(W_q, grad_W_q); sgd_b(b_q, grad_b_q);
    sgd(W_k, grad_W_k); sgd_b(b_k, grad_b_k);
    sgd(W_v, grad_W_v); sgd_b(b_v, grad_b_v);
    sgd(W_o, grad_W_o); sgd_b(b_o, grad_b_o);
}

void LinearAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    grad_b_q.fill(0.0); grad_b_k.fill(0.0);
    grad_b_v.fill(0.0); grad_b_o.fill(0.0);
    // Note: we deliberately do NOT clear has_forward_cache_ here. The cache
    // is invalidated by the next forward() call (which overwrites it).
    // zero_grad() should not affect the backward's ability to run.
}

std::vector<Tensor*> LinearAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &b_q, &b_k, &b_v, &b_o};
}

std::vector<Tensor*> LinearAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o,
            &grad_b_q, &grad_b_k, &grad_b_v, &grad_b_o};
}

// ============================================================================
// LinearAttentionBlock — pre-LN residual transformer block
// ============================================================================
LinearAttentionBlock::LinearAttentionBlock(size_t d_model, size_t ffn_hidden)
    : d_model_(d_model),
      ffn_hidden_(ffn_hidden == 0 ? d_model : ffn_hidden),
      ln1_(d_model), ln2_(d_model),
      attn_(d_model),
      ffn1_(d_model, ffn_hidden_),
      ffn2_(ffn_hidden_, d_model) {

    if (d_model == 0) {
        throw std::invalid_argument("LinearAttentionBlock: d_model must be > 0");
    }
}

Tensor LinearAttentionBlock::forward(const Tensor& input) {
    last_input_ = input.clone();
    const size_t n = input.rows;

    // Block sublayer 1 — pre-LN + linear-attention + residual
    last_normed1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_normed1_);
    last_residual1_ = Tensor(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_residual1_(i, j) = input(i, j) + last_attn_out_(i, j);

    // Block sublayer 2 — pre-LN + 2-layer GELU FFN + residual
    last_normed2_ = ln2_.forward(last_residual1_);
    last_ffn1_pre_ = ffn1_.forward(last_normed2_);
    // GELU activation (numerically-stable sigmoid-form)
    last_ffn1_out_ = Tensor(last_ffn1_pre_.rows, last_ffn1_pre_.cols);
    for (size_t i = 0; i < last_ffn1_pre_.rows; ++i) {
        for (size_t j = 0; j < last_ffn1_pre_.cols; ++j) {
            double x = last_ffn1_pre_(i, j);
            // GELU(x) = x · Φ(x)  ≈ x · 0.5 · (1 + tanh(√(2/π)(x + 0.044715 x³)))
            double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
            double gelu = 0.5 * x * (1.0 + std::tanh(inner));
            last_ffn1_out_(i, j) = gelu;
        }
    }
    last_ffn2_out_ = ffn2_.forward(last_ffn1_out_);

    Tensor out(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            out(i, j) = last_residual1_(i, j) + last_ffn2_out_(i, j);
    return out;
}

Tensor LinearAttentionBlock::backward(const Tensor& grad_output, double learning_rate) {
    const size_t n = grad_output.rows;

    // ---- Back through residual-2: grad_residual1 += grad_output, grad_ffn2_out = grad_output
    Tensor grad_ffn2_out = grad_output;
    Tensor grad_residual1 = grad_output;

    // ---- Back through ffn2 + GELU + ffn1 + ln2
    // We backprop through each Dense separately. The Layer interface's
    // backward does NOT do the GELU, so we replicate GELU's derivative here.
    Tensor grad_ffn1_pre_act = ffn2_.backward(grad_ffn2_out, learning_rate);
    // Exact GELU derivative at the pre-activation x:
    //   GELU(x)  = x · Φ(x),  Φ(x) = 0.5(1 + erf(x/√2))
    //   GELU'(x) = Φ(x) + x · φ(x),  φ(x) = (1/√(2π)) exp(-x²/2)
    Tensor grad_ffn1 = Tensor(n, ffn_hidden_);
    const double sqrt_2pi = std::sqrt(2.0 * M_PI);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < ffn_hidden_; ++j) {
            double x = last_ffn1_pre_(i, j);
            double phi = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
            double dens = std::exp(-0.5 * x * x) / sqrt_2pi;
            double gelu_prime = phi + x * dens;
            grad_ffn1(i, j) = grad_ffn1_pre_act(i, j) * gelu_prime;
        }
    }
    // Back through ffn1 (Dense::backward handles its own weights+biases+input grad)
    Tensor grad_normed2 = ffn1_.backward(grad_ffn1, learning_rate);
    // Back through ln2
    Tensor grad_residual1_from_ln = ln2_.backward(grad_normed2, learning_rate);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_residual1(i, j) += grad_residual1_from_ln(i, j);

    // ---- Back through residual-1: grad_input += grad_residual1, grad_attn_out = grad_residual1
    Tensor grad_input = grad_residual1;
    Tensor grad_attn_out = grad_residual1;

    // ---- Back through ln1 + attention
    Tensor grad_normed1 = attn_.backward(grad_attn_out, learning_rate);
    Tensor grad_input_from_ln = ln1_.backward(grad_normed1, learning_rate);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_input(i, j) += grad_input_from_ln(i, j);

    return grad_input;
}

void LinearAttentionBlock::update_weights(double learning_rate) {
    ln1_.update_weights(learning_rate);
    attn_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn1_.update_weights(learning_rate);
    ffn2_.update_weights(learning_rate);
}

void LinearAttentionBlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn1_.zero_grad();
    ffn2_.zero_grad();
}

std::vector<Tensor*> LinearAttentionBlock::parameters() {
    auto p = ln1_.parameters();
    auto a = attn_.parameters();
    auto l = ln2_.parameters();
    auto f1 = ffn1_.parameters();
    auto f2 = ffn2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), l.begin(), l.end());
    p.insert(p.end(), f1.begin(), f1.end());
    p.insert(p.end(), f2.begin(), f2.end());
    return p;
}

std::vector<Tensor*> LinearAttentionBlock::gradients() {
    auto g = ln1_.gradients();
    auto ga = attn_.gradients();
    auto gl = ln2_.gradients();
    auto gf1 = ffn1_.gradients();
    auto gf2 = ffn2_.gradients();
    g.insert(g.end(), ga.begin(), ga.end());
    g.insert(g.end(), gl.begin(), gl.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}

Tensor LinearAttentionBlock::get_weights() const {
    return attn_.get_weights();
}

Tensor LinearAttentionBlock::get_gradients() const {
    return attn_.get_gradients();
}

// ============================================================================
// LinearAttentionModel — input projection + stack of blocks + classifier
// ============================================================================
LinearAttentionModel::LinearAttentionModel(size_t d_input, size_t d_model, size_t d_output,
                                           size_t num_blocks, size_t num_heads, size_t ffn_hidden)
    : d_input_(d_input), d_model_(d_model), d_output_(d_output),
      num_blocks_(num_blocks == 0 ? 1 : num_blocks),
      num_heads_(num_heads == 0 ? 1 : num_heads),
      input_proj_(d_input, d_model),
      classifier_(d_model, d_output) {

    if (d_input == 0 || d_model == 0 || d_output == 0) {
        throw std::invalid_argument(
            "LinearAttentionModel: d_input/d_model/d_output must all be > 0");
    }
    if (d_model % num_heads_ != 0) {
        throw std::invalid_argument(
            "LinearAttentionModel: d_model must be divisible by num_heads");
    }
    blocks_.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_.emplace_back(d_model_, ffn_hidden);
    }
}

Tensor LinearAttentionModel::forward(const Tensor& input) {
    Tensor x = input_proj_.forward(input);
    for (auto& blk : blocks_) {
        x = blk.forward(x);
    }
    // Mean-pool over sequence axis (per-row mean) — common for set/seq classification.
    // For variable-length this is a reasonable default; caller can substitute.
    Tensor pooled(1, d_model_);
    pooled.fill(0.0);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            pooled(0, j) += x(i, j);
    if (x.rows > 0) {
        double inv = 1.0 / static_cast<double>(x.rows);
        for (size_t j = 0; j < d_model_; ++j) pooled(0, j) *= inv;
    }
    return classifier_.forward(pooled);
}

Tensor LinearAttentionModel::backward(const Tensor& grad_output, double learning_rate) {
    // Back through classifier
    Tensor grad_pooled = classifier_.backward(grad_output, learning_rate);

    // Back through mean-pool: each input row gets grad_pooled / n. We recover
    // the pre-pool sequence length from the first block's cached input.
    if (blocks_.empty()) {
        // No blocks — model is just input_proj -> mean-pool -> classifier.
        // Need a sequence length: use 1 (the model is degenerate, gradients
        // w.r.t. input are grad_pooled broadcast through input_proj).
        Tensor grad_blocks_out(1, d_model_);
        grad_blocks_out = grad_pooled;  // broadcast
        return input_proj_.backward(grad_blocks_out, learning_rate);
    }
    const size_t seq_len = blocks_[0].last_input_rows();
    Tensor grad_blocks_out(seq_len > 0 ? seq_len : 1, d_model_);
    double inv = seq_len > 0 ? 1.0 / static_cast<double>(seq_len) : 1.0;
    for (size_t i = 0; i < grad_blocks_out.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_blocks_out(i, j) = grad_pooled(0, j) * inv;

    // Back through blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad_blocks_out = it->backward(grad_blocks_out, learning_rate);
    }
    // Back through input projection
    return input_proj_.backward(grad_blocks_out, learning_rate);
}

void LinearAttentionModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void LinearAttentionModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& blk : blocks_) blk.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> LinearAttentionModel::parameters() {
    std::vector<Tensor*> p = input_proj_.parameters();
    for (auto& blk : blocks_) {
        auto bp = blk.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> LinearAttentionModel::gradients() {
    std::vector<Tensor*> g = input_proj_.gradients();
    for (auto& blk : blocks_) {
        auto bg = blk.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

Tensor LinearAttentionModel::get_weights() const {
    return input_proj_.get_weights();
}

Tensor LinearAttentionModel::get_gradients() const {
    return input_proj_.get_gradients();
}
