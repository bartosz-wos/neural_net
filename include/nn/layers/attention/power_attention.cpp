// Power Attention — Poli et al., 2024
//   "Power Attention: Transformers Need Less Compute Than You Think"
//   https://arxiv.org/abs/2403.14248
//
// See power_attention.h for the full formulation and backward derivation.
//
// Implementation notes (the bits easy to get wrong):
//   * The forward uses  s[i,j] = p · (z[i,j] − z_max)  before softmax. This
//     subtracts the per-query max INSIDE the p-scaled logits — important
//     because p is per-head, not per-head-and-query.
//   * Backward computes  dL/dz_k = p · dL/ds_k  (multiply the standard
//     softmax-backward by the same p), and dL/dp_log through softplus'.
//   * p_log_ is unconstrained (we apply softplus + p_eps in forward only),
//     so the optimiser can search p in (p_eps, ∞).
//   * Projection parameter grads go into raw grad_W_* tensors (not via
//     Dense::backward) because the per-head chain needs manual index control.

#include "power_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

namespace {

// Numerically stable softplus: log(1 + exp(x)).
inline double pa_softplus(double x) {
    return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
}

inline double pa_softplus_deriv_from_logit(double x) {
    // softplus'(x) = sigmoid(x) ; we use this to backprop through p_log.
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

inline double pa_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double pa_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

} // namespace

// ============================================================================
// PowerAttention
// ============================================================================

PowerAttention::PowerAttention(size_t d_model, size_t num_heads, double p_eps)
    : W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      p_log_(num_heads ? num_heads : 1, 1),
      grad_W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_p_log_(num_heads ? num_heads : 1, 1),
      d_model_(d_model), num_heads_(num_heads),
      head_dim_((num_heads && d_model && d_model % num_heads == 0)
                    ? d_model / num_heads : 1),
      inv_temp_(0.0), p_eps_(p_eps)
{
    if (d_model == 0)
        throw std::invalid_argument("PowerAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("PowerAttention: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "PowerAttention: d_model must be divisible by num_heads");
    if (!(p_eps >= 0.0))
        throw std::invalid_argument("PowerAttention: p_eps must be >= 0");

    inv_temp_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_p_log_.fill(0.0);

    // p_log_ init at 0  =>  p = softplus(0) + p_eps ≈ 1.243 — within a factor
    // of 1 of standard softmax (T = 1 / p ≈ 0.8).  The layer starts as a
    // slightly-sharper-than-softmax baseline, then learns the optimal mix.
    p_log_.fill(0.0);
}

Tensor PowerAttention::effective_p() const {
    Tensor p(num_heads_, 1);
    for (size_t h = 0; h < num_heads_; ++h)
        p(h, 0) = pa_softplus(p_log_(h, 0)) + p_eps_;
    return p;
}

Tensor PowerAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument(
            "PowerAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("PowerAttention: input.rows must be > 0");

    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;
    N_last_ = N;
    last_input_ = input.clone();

    // Compute per-head effective power and stash for backward.
    last_p_ = Tensor(H, 1);
    for (size_t h = 0; h < H; ++h)
        last_p_(h, 0) = pa_softplus(p_log_(h, 0)) + p_eps_;

    // Projections: Q = X @ W_q^T + b, etc. (Dense stores (out, in)).
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    last_A_      = Tensor(H * N, N);
    last_pz_max_ = Tensor(H, N);
    last_O_      = Tensor(N, d_model_);
    last_A_.fill(0.0);
    last_pz_max_.fill(0.0);
    last_O_.fill(0.0);

    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        const double p_h = last_p_(h, 0);

        for (size_t j = 0; j < N; ++j) {
            const size_t arow = h * N + j;

            // z[i,j] = inv_temp * (q_j · k_i)
            // p * z[i,j] = p_h * inv_temp_ * (q_j · k_i)
            // We absorb inv_temp_ into z:  s[i,j] = p_h · z[i,j]
            double pz_max = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += last_Q_(j, off + d) * last_K_(i, off + d);
                double pz = p_h * dot * inv_temp_;
                last_pz_max_(h, j) = (i == 0) ? pz : std::max(last_pz_max_(h, j), pz);
                pz_max = last_pz_max_(h, j);
            }

            // exp(p·z - p·z_max) and softmax over i.
            double denom = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += last_Q_(j, off + d) * last_K_(i, off + d);
                double pz = p_h * dot * inv_temp_;
                double e = std::exp(pz - pz_max);
                last_A_(arow, i) = e;
                denom += e;
            }
            double inv_denom = (denom > 0.0) ? (1.0 / denom) : 0.0;
            for (size_t i = 0; i < N; ++i)
                last_A_(arow, i) *= inv_denom;

            // o_j = Σ_i A[i,j] · v_i
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i)
                    acc += last_A_(arow, i) * last_V_(i, off + d);
                last_O_(j, off + d) = acc;
            }
        }
    }

    return W_o.forward(last_O_);
}

Tensor PowerAttention::backward(const Tensor& grad_output,
                                double /* learning_rate */) {
    const size_t N = N_last_;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;

    if (grad_output.rows != N || grad_output.cols != d_model_)
        throw std::invalid_argument(
            "PowerAttention::backward: grad_output shape mismatch");

    // --- Output projection: Y = O @ W_o^T + b_o ---
    // dO = dY @ W_o ; dW_o += dY^T @ O ; db_o += colsum(dY)
    Tensor dO(N, d_model_);
    dO.fill(0.0);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i)
                acc += grad_output(t, i) * W_o.weights(i, j);
            dO(t, j) = acc;
        }
    for (size_t i = 0; i < d_model_; ++i) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) {
            bacc += grad_output(t, i);
            for (size_t j = 0; j < d_model_; ++j)
                grad_W_o(i, j) += grad_output(t, i) * last_O_(t, j);
        }
        W_o.grad_bias[0][i] += bacc;
    }

    // --- Per-head power-attention backward (single pass per head) ---
    //
    // Forward:
    //   z[i,j] = inv_temp * (q_j · k_i)
    //   p      = softplus(p_log) + p_eps                   (per-head)
    //   s[i,j] = p · z[i,j]
    //   A[i,j] = softmax_i(s[i,j])                         (numerically stable via max-shift)
    //   o[j, off+d] = Σ_i A[i,j] · V[i, off+d]
    //
    // Backward (one head, h):
    //   dA[i,j]   = Σ_d dO[j, off+d] · V[i, off+d]
    //   dV[i,off+d] += A[i,j] · dO[j, off+d]
    //   sumA[j]   = Σ_i dA[i,j]
    //   dS[i,j]   = A[i,j] · (dA[i,j] - sumA[j])          (softmax backward identity)
    //   dZ[i,j]   = p_h · dS[i,j]                          (chain through p)
    //   dQ[j,off+d] += inv_temp · dZ[i,j] · K[i, off+d]   (chain through q_j)
    //   dK[i,off+d] += inv_temp · dZ[i,j] · Q[j, off+d]   (chain through k_i)
    //   dp_log[h] += Σ_ij (dA[i,j] - sumA[j]) · A[i,j] · z[i,j]
    //                 × softplus'(p_log[h])                (chain through softplus)
    //
    // dW_q[off+d, col] += Σ_j dQ[j, off+d] · X[j, col]
    // dW_k[off+d, col] += Σ_j dK[j, off+d] · X[j, col]
    // dW_v[off+d, col] += Σ_i dV[i, off+d] · X[i, col]
    //
    // Bias: db_q[off+d] += Σ_j dQ[j, off+d] ; db_k[off+d] += Σ_j dK[j, off+d]
    //       db_v[d]     += Σ_i dV[i, d]
    //
    // Input grad:  dX[j, col] = Σ_d (dQ[j, d] · W_q[d, col] + dK[j, d] · W_k[d, col])
    //                          + Σ_d dV[j, d] · W_v[d, col]

    Tensor dV(N, d_model_);
    dV.fill(0.0);
    Tensor dQ_full(N, d_model_);
    Tensor dK_full(N, d_model_);
    dQ_full.fill(0.0);
    dK_full.fill(0.0);

    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        const double p_h = last_p_(h, 0);

        // 1) dA from dO & V, accumulate dV (head-local contribution)
        Tensor dA_local(N, N);
        dA_local.fill(0.0);
        for (size_t j = 0; j < N; ++j) {
            for (size_t d = 0; d < dh; ++d) {
                const double do_j = dO(j, off + d);
                for (size_t i = 0; i < N; ++i) {
                    dA_local(i, j) += do_j * last_V_(i, off + d);
                    dV(i, off + d) += last_A_(h * N + j, i) * do_j;
                }
            }
        }

        // 2) Head-local dQ, dK, dp_log accumulator
        Tensor dQ(N, d_model_);
        Tensor dK(N, d_model_);
        dQ.fill(0.0);
        dK.fill(0.0);
        double dp_log_acc_h = 0.0;

        for (size_t j = 0; j < N; ++j) {
            // sumA[j] = Σ_i A[i,j] * dA[i,j]    (NOT Σ_i dA[i,j] — wrong!)
            // The standard softmax-backward is:
            //   dL/ds_m = A_m * (dA_m − Σ_i A_i · dA_i)
            // so the Σ term is weighted by A_i, not raw dA_i.
            double sumA_w = 0.0;
            for (size_t i = 0; i < N; ++i)
                sumA_w += last_A_(h * N + j, i) * dA_local(i, j);

            for (size_t i = 0; i < N; ++i) {
                const double a  = last_A_(h * N + j, i);
                const double da = dA_local(i, j);
                const double dS_ij = a * (da - sumA_w);
                const double dZ_ij = p_h * dS_ij;

                // Chain into dQ[j] and dK[i] for this (i, j) pair
                for (size_t d = 0; d < dh; ++d) {
                    dQ(j, off + d) += inv_temp_ * dZ_ij * last_K_(i, off + d);
                    dK(i, off + d) += inv_temp_ * dZ_ij * last_Q_(j, off + d);
                }

                // p_log chain: dL/dp_log += (da − A-weighted sumA) · a · z[i,j] (then × softplus')
                // z[i,j] = inv_temp * (q_j · k_i)
                // dS_ij = a * (da - sumA_w), so:
                // dL/dp += sum_ij z[i,j] · dS_ij
                double qk = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    qk += last_Q_(j, off + d) * last_K_(i, off + d);
                dp_log_acc_h += dS_ij * qk * inv_temp_;
            }
        }
        // softplus'(p_log[h]) factor on the p_log chain
        grad_p_log_(h, 0) += dp_log_acc_h *
                             pa_softplus_deriv_from_logit(p_log_(h, 0));

        // 3) Accumulate W_q / W_k gradients and biases from dQ / dK.
        //    Also accumulate dQ_full / dK_full across heads for input grad.
        for (size_t j = 0; j < N; ++j) {
            for (size_t d = 0; d < dh; ++d) {
                const double dQ_j = dQ(j, off + d);
                const double dK_j = dK(j, off + d);
                W_q.grad_bias[0][off + d] += dQ_j;
                W_k.grad_bias[0][off + d] += dK_j;
                dQ_full(j, off + d) += dQ_j;
                dK_full(j, off + d) += dK_j;
                for (size_t col = 0; col < d_model_; ++col) {
                    grad_W_q(off + d, col) += dQ_j * last_input_(j, col);
                    grad_W_k(off + d, col) += dK_j * last_input_(j, col);
                }
            }
        }
    }

    // 4) grad_W_v from dV: grad_W_v[d, col] += Σ_i dV[i, d] · X[i, col]
    //                       db_v[d]        += Σ_i dV[i, d]
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < d_model_; ++d) {
            const double dv = dV(i, d);
            W_v.grad_bias[0][d] += dv;
            for (size_t col = 0; col < d_model_; ++col)
                grad_W_v(d, col) += dv * last_input_(i, col);
        }

    // 5) Input grad:  dX[j, col] = Σ_d (dQ_full[j, d] · W_q[d, col]
    //                                       + dK_full[j, d] · W_k[d, col])
    //                          + Σ_d dV[j, d] · W_v[d, col]
    Tensor dX(N, d_model_);
    dX.fill(0.0);
    for (size_t j = 0; j < N; ++j) {
        for (size_t d = 0; d < d_model_; ++d) {
            const double dQ_jd = dQ_full(j, d);
            const double dK_jd = dK_full(j, d);
            const double dV_jd = dV(j, d);
            for (size_t col = 0; col < d_model_; ++col) {
                dX(j, col) += dQ_jd * W_q.weights(d, col)
                           +  dK_jd * W_k.weights(d, col)
                           +  dV_jd * W_v.weights(d, col);
            }
        }
    }

    return dX;
}

void PowerAttention::update_weights(double learning_rate) {
    // Update the raw projection weights/biases using OUR accumulated grads
    // (grad_W_q / grad_W_k / grad_W_v / grad_W_o + W_q/W_k/W_v/W_o.grad_bias).
    // Note: we do NOT delegate to Dense::update_weights here, because Dense's
    // own `grad_weights` was never written to (we accumulate into raw
    // `grad_W_*` tensors in the per-head backward).  We subtract the
    // learning-rate-scaled gradient from the corresponding Tensor in place.
    auto step = [learning_rate](Tensor& w, const Tensor& g) {
        for (size_t k = 0; k < w.data.size(); ++k)
            w.data[k] -= learning_rate * g.data[k];
    };
    step(W_q.weights, grad_W_q);
    step(W_k.weights, grad_W_k);
    step(W_v.weights, grad_W_v);
    step(W_o.weights, grad_W_o);
    step(W_q.bias, W_q.grad_bias);
    step(W_k.bias, W_k.grad_bias);
    step(W_v.bias, W_v.grad_bias);
    step(W_o.bias, W_o.grad_bias);

    // Plain SGD on p_log:  p_log -= lr · grad_p_log
    for (size_t h = 0; h < num_heads_; ++h)
        p_log_(h, 0) -= learning_rate * grad_p_log_(h, 0);
}

void PowerAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_p_log_.fill(0.0);
}

std::vector<Tensor*> PowerAttention::parameters() {
    return {
        &W_q.weights, &W_q.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &W_o.weights, &W_o.bias,
        &p_log_
    };
}

std::vector<Tensor*> PowerAttention::gradients() {
    return {
        &grad_W_q, &W_q.grad_bias,
        &grad_W_k, &W_k.grad_bias,
        &grad_W_v, &W_v.grad_bias,
        &grad_W_o, &W_o.grad_bias,
        &grad_p_log_
    };
}

// ============================================================================
// PowerBlock
// ============================================================================

PowerBlock::PowerBlock(size_t d_model, size_t num_heads, size_t ffn_dim, double p_eps)
    : attn(d_model, num_heads, p_eps),
      ln1(d_model), ln2(d_model),
      ffn_fc1_(d_model, ffn_dim ? ffn_dim : 4 * d_model),
      ffn_fc2_(ffn_dim ? ffn_dim : 4 * d_model, d_model),
      d_model_(d_model),
      ffn_dim_(ffn_dim ? ffn_dim : 4 * d_model)
{
    if (d_model == 0)
        throw std::invalid_argument("PowerBlock: d_model must be > 0");
}

Tensor PowerBlock::forward(const Tensor& input) {
    last_x_ = input.clone();

    last_z1_ = ln1.forward(input);
    Tensor attn_out = attn.forward(last_z1_);
    last_attn_out_ = attn_out.clone();

    // res1 = z1 + attn_out
    last_res1_ = Tensor(input.rows, d_model_);
    last_res1_.fill(0.0);
    for (size_t i = 0; i < input.data.size(); ++i)
        last_res1_.data[i] = last_z1_.data[i] + attn_out.data[i];

    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    last_h_act_ = Tensor(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < last_h_pre_.data.size(); ++i)
        last_h_act_.data[i] = pa_gelu(last_h_pre_.data[i]);
    Tensor h_out = ffn_fc2_.forward(last_h_act_);

    Tensor out(input.rows, d_model_);
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = last_res1_.data[i] + h_out.data[i];
    return out;
}

Tensor PowerBlock::backward(const Tensor& grad_output, double learning_rate) {
    // out = res1 + ffn_out  ->  d_res1 = grad_output + (chain through FFN)
    Tensor d_h_act = ffn_fc2_.backward(grad_output, learning_rate);
    // d_h_pre = d_h_act * GELU'(h_pre)
    Tensor d_h_pre(last_h_pre_.rows, last_h_pre_.cols);
    for (size_t i = 0; i < d_h_pre.data.size(); ++i)
        d_h_pre.data[i] = d_h_act.data[i] * pa_gelu_deriv(last_h_pre_.data[i]);
    Tensor d_z2 = ffn_fc1_.backward(d_h_pre, learning_rate);

    // z2 = ln2(res1) -> must route d_z2 THROUGH ln2, not add it directly.
    Tensor d_res1 = grad_output.clone();
    Tensor d_res1_from_ln2 = ln2.backward(d_z2, learning_rate);
    for (size_t i = 0; i < d_res1.data.size(); ++i)
        d_res1.data[i] += d_res1_from_ln2.data[i];

    // res1 = z1 + attn_out
    Tensor d_z1 = d_res1.clone();
    Tensor d_z1_from_attn = attn.backward(d_res1, learning_rate);
    for (size_t i = 0; i < d_z1.data.size(); ++i)
        d_z1.data[i] += d_z1_from_attn.data[i];

    // z1 = ln1(x)
    return ln1.backward(d_z1, learning_rate);
}

void PowerBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

void PowerBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> PowerBlock::parameters() {
    auto pa = attn.parameters();
    auto pl1 = ln1.parameters();
    auto pl2 = ln2.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    std::vector<Tensor*> all;
    all.reserve(pa.size() + pl1.size() + pl2.size() + pf1.size() + pf2.size());
    for (auto* p : pa) all.push_back(p);
    for (auto* p : pl1) all.push_back(p);
    for (auto* p : pl2) all.push_back(p);
    for (auto* p : pf1) all.push_back(p);
    for (auto* p : pf2) all.push_back(p);
    return all;
}

std::vector<Tensor*> PowerBlock::gradients() {
    auto ga = attn.gradients();
    auto gl1 = ln1.gradients();
    auto gl2 = ln2.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    std::vector<Tensor*> all;
    all.reserve(ga.size() + gl1.size() + gl2.size() + gf1.size() + gf2.size());
    for (auto* g : ga) all.push_back(g);
    for (auto* g : gl1) all.push_back(g);
    for (auto* g : gl2) all.push_back(g);
    for (auto* g : gf1) all.push_back(g);
    for (auto* g : gf2) all.push_back(g);
    return all;
}
