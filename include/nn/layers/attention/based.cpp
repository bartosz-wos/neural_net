// Based Linear Attention — Arora et al. 2024
//   "Simple linear attention language models balance the recall-throughput
//    tradeoff" (https://arxiv.org/abs/2402.18668)
//
// Single-head causal Taylor-approximated linear attention. See based.h for
// the math.
//
// Backward derivation (the bits that are easy to get wrong):
//
//   Forward (per query i, for all keys j ≤ i, value j):
//     A[i, j] = phi_q[i] · phi_k[j]     (with j ≤ i for causal)
//     y[i]    = sum_{j≤i} A[i,j] V[j]
//     z[i]    = sum_{j≤i} phi_q[i] · phi_k[j] + eps   (per-query normalizer)
//     out[i]  = y[i] / z[i]
//
//   Backward (grad_output = dL/dout):
//     d_y[i] = grad_output[i] / z[i]                       (chain rule on 1/z)
//     d_z[i] = - (grad_output[i] · y[i]) / z[i]²           (chain rule on 1/z)
//
//   // d_y → d_V, d_A:
//     d_V[j] += sum_i A[i,j] d_y[i]   (for i ≥ j)
//     d_A[i, j] = d_y[i] · V[j]        (for j ≤ i)
//
//   // d_A → d_phi_q, d_phi_k:
//     d_phi_q[i] += sum_j d_A[i, j] · phi_k[j]   (j ≤ i)
//     d_phi_k[j] += sum_i d_A[i, j] · phi_q[i]   (i ≥ j)
//
//   // d_z contribution to d_phi_q, d_phi_k (the cumsum path):
//     z[i] = phi_q[i] · Ksum[i]    where Ksum[i] = sum_{j≤i} phi_k[j]
//     d_phi_q[i] += d_z[i] · Ksum[i]
//     For each j, Ksum[i] depends on phi_k[j] for all i ≥ j, so:
//       d_phi_k[j] += sum_{i ≥ j} d_z[i] · phi_q[i]
//
//   // d_phi → d_q_pre, d_k_pre via TaylorExp backward (see below).
//
// TaylorExp backward (for 2nd-order phi with phi_dim = 1 + d + d²):
//   phi[0]           = 1                              (constant, no gradient)
//   phi[1..d]        = q[k] / rrd       k = 0..d-1
//   phi[d+1 + i*d+j] = q[i]*q[j] / (r2 * rd)   i,j = 0..d-1
//
//   d_q[k]:
//     d/ d_q[k] phi[1+k]       = 1 / rrd
//     d/ d_q[k] phi[d+1+i*d+j] = (delta_ik q[j] + q[i] delta_jk) / (r2*rd)
//                              = (q[j] (i==k) + q[i] (j==k)) / (r2*rd)
//
//   So: d_q[k] = sum_m d_phi[m] · d phi[m] / d q[k]
//
//   d_q[k] = d_phi[1+k] / rrd
//          + sum_{j=0}^{d-1} d_phi[d+1+k*d+j] * q[j] / (r2*rd)
//          + sum_{i=0}^{d-1} d_phi[d+1+i*d+k] * q[i] / (r2*rd)
//
//   Same for d_k[k] (replace q with k). The two matrix terms are equal
//   (symmetric), so we can sum them in one pass.

#include "based.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// ---------------------------------------------------------------------------
// taylor_phi: compute phi(x) for an (n, d) tensor x.
//   output: (n, 1 + d + d²)
//   phi[i, 0]              = 1
//   phi[i, 1+k]            = x[i, k] / rrd   for k = 0..d-1
//   phi[i, 1+d + i0*d+j0]  = x[i, i0] * x[i, j0] / (r2*rd)   for i0, j0 = 0..d-1
// ---------------------------------------------------------------------------
static Tensor taylor_phi(const Tensor& x, size_t d, double r2, double rrd, double rd) {
    const size_t n = x.rows;
    const size_t phi_dim = 1 + d + d * d;
    Tensor out(n, phi_dim);
    const double inv_r2_rd = 1.0 / (r2 * rd);
    for (size_t t = 0; t < n; ++t) {
        out(t, 0) = 1.0;
        for (size_t k = 0; k < d; ++k) {
            out(t, 1 + k) = x(t, k) / rrd;
        }
        // Outer product flattened (row-major): (d+1) + i*d + j for i,j in 0..d-1
        for (size_t i = 0; i < d; ++i) {
            double xi = x(t, i);
            for (size_t j = 0; j < d; ++j) {
                out(t, 1 + d + i * d + j) = xi * x(t, j) * inv_r2_rd;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// taylor_phi_backward: given d_phi (n, phi_dim) and the pre-phix x (n, d),
//   return d_x (n, d).
//   See derivation in the header comment.
// ---------------------------------------------------------------------------
static Tensor taylor_phi_backward(const Tensor& d_phi, const Tensor& x, size_t d,
                                  double r2, double rrd, double rd) {
    const size_t n = x.rows;
    Tensor d_x(n, d);
    d_x.fill(0.0);
    const double inv_r2_rd = 1.0 / (r2 * rd);
    for (size_t t = 0; t < n; ++t) {
        // d_x[k] from phi[1+k] = x[k] / rrd  →  d_x[k] += d_phi[1+k] / rrd
        for (size_t k = 0; k < d; ++k) {
            d_x(t, k) = d_phi(t, 1 + k) / rrd;
        }
        // d_x[k] from the outer-product entries.
        // phi[d+1 + i*d + j] = x[i]*x[j] * inv_r2_rd
        // d phi / d x[k]:
        //   = (x[j] if i==k else 0) * inv_r2_rd  +  (x[i] if j==k else 0) * inv_r2_rd
        //   (contribution from row i of the outer product with k as the left index)
        //   + (contribution from column j of the outer product with k as the right index)
        // d_x[k] += sum_{j=0..d-1} d_phi[d+1 + k*d + j] * x[j] * inv_r2_rd
        //         + sum_{i=0..d-1} d_phi[d+1 + i*d + k] * x[i] * inv_r2_rd
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < d; ++j) s += d_phi(t, 1 + d + k * d + j) * x(t, j);
            for (size_t i = 0; i < d; ++i) s += d_phi(t, 1 + d + i * d + k) * x(t, i);
            d_x(t, k) += s * inv_r2_rd;
        }
    }
    return d_x;
}

// ===========================================================================
// BasedAttention
// ===========================================================================

BasedAttention::BasedAttention(size_t d_model, size_t seq_len, size_t feature_dim,
                               double eps)
    : d_model_(d_model), seq_len_(seq_len), feature_dim_(feature_dim),
      phi_dim_(1 + feature_dim_ + feature_dim_ * feature_dim_),
      eps_(eps),
      W_q(d_model, feature_dim), W_k(d_model, feature_dim),
      W_v(d_model, d_model),   W_o(d_model, d_model),
      grad_W_q(d_model, feature_dim), grad_W_k(d_model, feature_dim),
      grad_W_v(d_model, d_model),   grad_W_o(d_model, d_model) {
    if (d_model == 0) throw std::invalid_argument("BasedAttention: d_model must be > 0");
    if (seq_len == 0) throw std::invalid_argument("BasedAttention: seq_len must be > 0");
    if (feature_dim == 0) throw std::invalid_argument("BasedAttention: feature_dim must be > 0");

    // Xavier-uniform init for Q, K, V (output dim) projections; small init
    // for W_o (the output projection — paper convention is small here).
    auto xavier_init = [&](Tensor& W, std::mt19937& gen) {
        double s = std::sqrt(6.0 / (double)(W.rows + W.cols));
        std::uniform_real_distribution<> d(-s, s);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) = d(gen);
    };
    std::mt19937 gen(20240902);
    xavier_init(W_q, gen);
    xavier_init(W_k, gen);
    xavier_init(W_v, gen);
    xavier_init(W_o, gen);
    zero_grad();
}

void BasedAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
}

Tensor BasedAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (n != seq_len_) {
        throw std::invalid_argument("BasedAttention: input.rows != seq_len_");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument("BasedAttention: input.cols != d_model_");
    }
    last_input_ = input.clone();

    // Q/K/V projections: y = x W (no bias — matches Based paper).
    // Repo convention (matching Linformer / Performer): W has shape (in_dim, out_dim)
    // so y = x W gives the projected result.
    last_q_pre_ = input * W_q;   // (n, feature_dim)
    last_k_pre_ = input * W_k;   // (n, feature_dim)
    last_v_     = input * W_v;   // (n, d_model)

    // Taylor feature map on Q and K.
    const double d = (double)feature_dim_;
    const double rd  = std::sqrt(d);
    const double r2  = std::sqrt(2.0);
    const double rrd = std::sqrt(rd);
    last_phi_q_ = taylor_phi(last_q_pre_, feature_dim_, r2, rrd, rd);
    last_phi_k_ = taylor_phi(last_k_pre_, feature_dim_, r2, rrd, rd);

    // A = tril(phi_q · phi_k^T)  — (n, n)
    last_A_ = last_phi_q_ * last_phi_k_.transpose();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            last_A_(i, j) = 0.0;
        }
    }

    // Ksum[i] = sum_{j ≤ i} phi_k[j]  (n, phi_dim) — cumsum along seq axis
    last_Ksum_ = Tensor(n, phi_dim_);
    last_Ksum_.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t kk = 0; kk < phi_dim_; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j <= i; ++j) s += last_phi_k_(j, kk);
            last_Ksum_(i, kk) = s;
        }
    }

    // z[i] = phi_q[i] · Ksum[i] + eps  (n,)
    last_z_ = Tensor(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t kk = 0; kk < phi_dim_; ++kk) s += last_phi_q_(i, kk) * last_Ksum_(i, kk);
        last_z_(i, 0) = s + eps_;
    }

    // y = A · V  (n, d_model)  then  out_pre = y / z
    last_out_pre_ = last_A_ * last_v_;   // (n, d_model)
    for (size_t i = 0; i < n; ++i) {
        double zinv = 1.0 / last_z_(i, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            last_out_pre_(i, j) *= zinv;
        }
    }

    // Output projection: out = out_pre · W_o  (n, d_model)
    Tensor result = last_out_pre_ * W_o;
    last_output_ = result;
    return result;
}

Tensor BasedAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = seq_len_;
    const size_t d = d_model_;
    const size_t fd = feature_dim_;
    const size_t pd = phi_dim_;

    // 1) d_output → d_out_pre  via W_o:  out = out_pre · W_o
    //    d_out_pre[i, j] = sum_k grad_output[i, k] * W_o[j, k]
    Tensor d_out_pre(n, d);
    d_out_pre.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += grad_output(i, k) * W_o(j, k);
            d_out_pre(i, j) = s;
        }
    }
    // grad for W_o: dW_o[j, k] += sum_i d_out[i, k] * out_pre[i, j]
    for (size_t j = 0; j < d; ++j) {
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += grad_output(i, k) * last_out_pre_(i, j);
            grad_W_o(j, k) += s;
        }
    }

    // 2) d_out_pre → d_y, d_z  via out_pre[i] = y[i] / z[i]
    //    d_y[i] = d_out_pre[i] / z[i]
    //    d_z[i] = - sum_j (d_out_pre[i,j] * y[i,j]) / z[i]²
    //             = - sum_j (d_out_pre[i,j] * out_pre[i,j]) / z[i]
    //             (since y = out_pre * z, so y/z² = out_pre/z)
    Tensor d_y(n, d);
    d_y.fill(0.0);
    Tensor d_z(n, 1);
    d_z.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        double z = last_z_(i, 0);
        double zinv = 1.0 / z;
        for (size_t j = 0; j < d; ++j) {
            d_y(i, j) = d_out_pre(i, j) * zinv;
            // d_z += -d_out_pre * out_pre / z   (NOT z² — see derivation above)
            d_z(i, 0) -= d_out_pre(i, j) * last_out_pre_(i, j) * zinv;
        }
    }

    // 3) d_y → d_A, d_V  via y[i] = sum_{j≤i} A[i,j] V[j]
    //    d_A[i, j] = sum_k d_y[i, k] * V[j, k]   (for j ≤ i)
    //    d_V[j, k] = sum_{i ≥ j} d_y[i, k] * A[i, j]
    Tensor d_A(n, n);
    d_A.fill(0.0);
    Tensor d_V(n, d);
    d_V.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) s += d_y(i, k) * last_v_(j, k);
            d_A(i, j) = s;
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t i = j; i < n; ++i) s += d_y(i, k) * last_A_(i, j);
            d_V(j, k) = s;
        }
    }

    // 4) d_V → dW_v, d_v_input  via V = input · W_v
    //    V[i, j] = sum_d input[i, d] * W_v[d, j]
    //    dW_v[d, j] += sum_i d_V[i, j] * input[i, d]
    //    d_input_v[i, d] = sum_j d_V[i, j] * W_v[d, j]
    for (size_t d = 0; d < d_model_; ++d) {  // row of W_v
        for (size_t j = 0; j < d_model_; ++j) {  // col of W_v
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d_V(i, j) * last_input_(i, d);
            grad_W_v(d, j) += s;
        }
    }
    Tensor d_input_from_v(n, d_model_);
    d_input_from_v.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < d_model_; ++d) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += d_V(i, j) * W_v(d, j);
            d_input_from_v(i, d) = s;
        }
    }

    // 5) d_A → d_phi_q, d_phi_k  via A[i, j] = phi_q[i] · phi_k[j]   (j ≤ i)
    //    d_phi_q[i] += sum_{j≤i} d_A[i, j] * phi_k[j]
    //    d_phi_k[j] += sum_{i≥j} d_A[i, j] * phi_q[i]
    Tensor d_phi_q(n, pd);
    d_phi_q.fill(0.0);
    Tensor d_phi_k(n, pd);
    d_phi_k.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t kk = 0; kk < pd; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j <= i; ++j) s += d_A(i, j) * last_phi_k_(j, kk);
            d_phi_q(i, kk) = s;
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t kk = 0; kk < pd; ++kk) {
            double s = 0.0;
            for (size_t i = j; i < n; ++i) s += d_A(i, j) * last_phi_q_(i, kk);
            d_phi_k(j, kk) = s;
        }
    }

    // 6) d_z → d_phi_q, d_phi_k  via z[i] = phi_q[i] · Ksum[i] + eps
    //    d_phi_q[i] += d_z[i] * Ksum[i]
    //    Ksum[i] depends on phi_k[j] for all j ≤ i, so:
    //    d_phi_k[j] += sum_{i≥j} d_z[i] * phi_q[i]
    for (size_t i = 0; i < n; ++i) {
        for (size_t kk = 0; kk < pd; ++kk) {
            d_phi_q(i, kk) += d_z(i, 0) * last_Ksum_(i, kk);
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t kk = 0; kk < pd; ++kk) {
            double s = 0.0;
            for (size_t i = j; i < n; ++i) s += d_z(i, 0) * last_phi_q_(i, kk);
            d_phi_k(j, kk) += s;
        }
    }

    // 7) d_phi_q, d_phi_k → d_q_pre, d_k_pre  via taylor_phi_backward.
    const double dd = (double)fd;
    const double rd_  = std::sqrt(dd);
    const double r2_  = std::sqrt(2.0);
    const double rrd_ = std::sqrt(rd_);
    Tensor d_q_pre = taylor_phi_backward(d_phi_q, last_q_pre_, fd, r2_, rrd_, rd_);
    Tensor d_k_pre = taylor_phi_backward(d_phi_k, last_k_pre_, fd, r2_, rrd_, rd_);

    // 8) d_q_pre, d_k_pre → d_input, dW_q, dW_k  via y = input · W
    //    q_pre[i, k] = sum_d input[i, d] * W_q[d, k]
    //    dW_q[d, k] += sum_i d_q_pre[i, k] * input[i, d]
    //    d_input_q[i, d] += sum_k d_q_pre[i, k] * W_q[d, k]   (and same for k)
    for (size_t d_dim = 0; d_dim < d_model_; ++d_dim) {
        for (size_t k = 0; k < fd; ++k) {
            double sq = 0.0, sk = 0.0;
            for (size_t i = 0; i < n; ++i) {
                sq += d_q_pre(i, k) * last_input_(i, d_dim);
                sk += d_k_pre(i, k) * last_input_(i, d_dim);
            }
            grad_W_q(d_dim, k) += sq;
            grad_W_k(d_dim, k) += sk;
        }
    }
    Tensor d_input_from_qk(n, d_model_);
    d_input_from_qk.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d_dim = 0; d_dim < d_model_; ++d_dim) {
            double sq = 0.0, sk = 0.0;
            for (size_t k = 0; k < fd; ++k) {
                sq += d_q_pre(i, k) * W_q(d_dim, k);
                sk += d_k_pre(i, k) * W_k(d_dim, k);
            }
            d_input_from_qk(i, d_dim) = sq + sk;
        }
    }

    // 9) d_input = d_input_from_v + d_input_from_qk
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            d_input(i, k) = d_input_from_v(i, k) + d_input_from_qk(i, k);
        }
    }
    return d_input;
}

void BasedAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) -= learning_rate * gW(i, j);
    };
    sgd(W_q, grad_W_q);
    sgd(W_k, grad_W_k);
    sgd(W_v, grad_W_v);
    sgd(W_o, grad_W_o);
}

std::vector<Tensor*> BasedAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}
std::vector<Tensor*> BasedAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}

// ===========================================================================
// BasedBlock
// ===========================================================================

BasedBlock::BasedBlock(size_t d_model, size_t seq_len, size_t feature_dim, size_t ffn_dim)
    : d_model_(d_model), attn(d_model, seq_len, feature_dim),
      ln1(d_model), ln2(d_model),
      W1(d_model, ffn_dim ? ffn_dim : 4 * d_model), b1(1, ffn_dim ? ffn_dim : 4 * d_model),
      W2(ffn_dim ? ffn_dim : 4 * d_model, d_model), b2(1, d_model),
      grad_W1(d_model, ffn_dim ? ffn_dim : 4 * d_model), grad_b1(1, ffn_dim ? ffn_dim : 4 * d_model),
      grad_W2(ffn_dim ? ffn_dim : 4 * d_model, d_model), grad_b2(1, d_model) {
    if (d_model == 0) throw std::invalid_argument("BasedBlock: d_model must be > 0");
    if (seq_len == 0)  throw std::invalid_argument("BasedBlock: seq_len must be > 0");

    // Xavier-uniform init for the FFN matrices.
    auto xavier_init = [&](Tensor& W) {
        double s = std::sqrt(6.0 / (double)(W.rows + W.cols));
        std::mt19937 gen(0xBACE);
        std::uniform_real_distribution<> d(-s, s);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) = d(gen);
    };
    xavier_init(W1);
    xavier_init(W2);
    b1.fill(0.0);
    b2.fill(0.0);
    zero_grad();
}

void BasedBlock::zero_grad() {
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
}

Tensor BasedBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    last_ln1_out_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_ln1_out_);
    last_resid1_ = Tensor(input.rows, input.cols);
    for (size_t i = 0; i < input.rows * input.cols; ++i)
        last_resid1_.data[i] = input.data[i] + last_attn_out_.data[i];

    last_ln2_out_ = ln2.forward(last_resid1_);
    // W1 is (d_model, ffn_dim); result is (n, ffn_dim)
    last_ffn_pregelu_ = last_ln2_out_ * W1;
    for (size_t j = 0; j < last_ffn_pregelu_.cols; ++j) {
        double bj = b1(0, j);
        for (size_t i = 0; i < last_ffn_pregelu_.rows; ++i)
            last_ffn_pregelu_(i, j) += bj;
    }
    // GELU activation (tanh approximation, matches common convention).
    Tensor h(last_ffn_pregelu_.rows, last_ffn_pregelu_.cols);
    for (size_t i = 0; i < h.rows; ++i) {
        for (size_t j = 0; j < h.cols; ++j) {
            double x = last_ffn_pregelu_(i, j);
            double t = std::tanh(0.7978845608 * (x + 0.044715 * x * x * x));
            h(i, j) = 0.5 * x * (1.0 + t);
        }
    }
    // W2 is (ffn_dim, d_model); result is (n, d_model)
    last_ffn_out_ = h * W2;
    for (size_t j = 0; j < last_ffn_out_.cols; ++j) {
        double bj = b2(0, j);
        for (size_t i = 0; i < last_ffn_out_.rows; ++i)
            last_ffn_out_(i, j) += bj;
    }
    Tensor out(last_resid1_.rows, last_resid1_.cols);
    for (size_t i = 0; i < out.rows * out.cols; ++i)
        out.data[i] = last_resid1_.data[i] + last_ffn_out_.data[i];
    return out;
}

Tensor BasedBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    const size_t d = d_model_;
    const size_t ffn = W2.rows;  // = W1.cols

    // Recompute the post-GELU activation h from the cached pre-GELU.
    Tensor h(n, ffn);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < ffn; ++k) {
            double x = last_ffn_pregelu_(i, k);
            double t = std::tanh(0.7978845608 * (x + 0.044715 * x * x * x));
            h(i, k) = 0.5 * x * (1.0 + t);
        }
    }

    // d_resid1 = grad_output (residual connection)
    Tensor d_resid1(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) d_resid1(i, j) = grad_output(i, j);

    // FFN path: d_ffn_out = grad_output (residual).
    // d_h[i, k] = sum_j d_ffn_out[i, j] * W2[k, j]
    Tensor d_h(n, ffn);
    d_h.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < ffn; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < d; ++j) s += grad_output(i, j) * W2(k, j);
            d_h(i, k) = s;
        }
    }
    // dW2[k, j] += sum_i grad_output[i, j] * h[i, k]
    for (size_t k = 0; k < ffn; ++k) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += grad_output(i, j) * h(i, k);
            grad_W2(k, j) += s;
        }
    }
    // db2[j] += sum_i grad_output[i, j]
    for (size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += grad_output(i, j);
        grad_b2(0, j) += s;
    }
    // d_pregelu = d_h ⊙ gelu'(pregelu)
    Tensor d_pregelu(n, ffn);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < ffn; ++k) {
            double x = last_ffn_pregelu_(i, k);
            double t = std::tanh(0.7978845608 * (x + 0.044715 * x * x * x));
            double gelu_prime = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) *
                                0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
            d_pregelu(i, k) = d_h(i, k) * gelu_prime;
        }
    }
    // dW1[j, k] += sum_i d_pregelu[i, k] * ln2_out[i, j]
    for (size_t j = 0; j < d; ++j) {
        for (size_t k = 0; k < ffn; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d_pregelu(i, k) * last_ln2_out_(i, j);
            grad_W1(j, k) += s;
        }
    }
    for (size_t k = 0; k < ffn; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += d_pregelu(i, k);
        grad_b1(0, k) += s;
    }
    // d_ln2_out[i, j] = sum_k d_pregelu[i, k] * W1[j, k]
    Tensor d_ln2_out(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < ffn; ++k) s += d_pregelu(i, k) * W1(j, k);
            d_ln2_out(i, j) = s;
        }
    }
    // d_resid1 += d_ln2_out (LN2 backward path)
    Tensor d_resid1_from_ln2 = ln2.backward(d_ln2_out, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) d_resid1(i, j) += d_resid1_from_ln2(i, j);

    // Now d_resid1 = d(attn_out) at the attn-out point (and d_input via the inner +).
    // The inner residual "resid1 = input + attn_out" sends d_resid1 to BOTH the
    // attn path (via attn_out) and the input path (via the +).
    // d_ln1_out = attn.backward(d_resid1, 0)
    Tensor d_ln1_out = attn.backward(d_resid1, 0.0);
    Tensor d_input_from_ln1 = ln1.backward(d_ln1_out, 0.0);
    // d_input = d_resid1 (from inner +) + d_ln1_back (from attn path)
    Tensor d_input(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            d_input(i, j) = d_resid1(i, j) + d_input_from_ln1(i, j);
        }
    }
    return d_input;
}

void BasedBlock::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) -= learning_rate * gW(i, j);
    };
    auto sgd_b = [&](Tensor& b, const Tensor& gb) {
        for (size_t j = 0; j < b.cols; ++j) b(0, j) -= learning_rate * gb(0, j);
    };
    sgd(W1, grad_W1);  sgd_b(b1, grad_b1);
    sgd(W2, grad_W2);  sgd_b(b2, grad_b2);
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
}

std::vector<Tensor*> BasedBlock::parameters() {
    return {&W1, &b1, &W2, &b2};
}
std::vector<Tensor*> BasedBlock::gradients() {
    return {&grad_W1, &grad_b1, &grad_W2, &grad_b2};
}

// ===========================================================================
// BasedModel
// ===========================================================================

BasedModel::BasedModel(size_t in_dim, size_t d_model, size_t seq_len,
                       size_t out_features, size_t num_blocks,
                       size_t feature_dim, size_t ffn_dim)
    : d_model_(d_model), num_blocks_(num_blocks), out_features_(out_features),
      blocks_(),
      final_ln_(d_model),
      W_out_(d_model, out_features), b_out_(1, out_features),
      grad_W_out_(d_model, out_features), grad_b_out_(1, out_features),
      W_in_(in_dim, d_model), b_in_(1, d_model),
      grad_W_in_(in_dim, d_model), grad_b_in_(1, d_model) {
    if (in_dim == 0) throw std::invalid_argument("BasedModel: in_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("BasedModel: d_model must be > 0");
    if (seq_len == 0) throw std::invalid_argument("BasedModel: seq_len must be > 0");
    if (out_features == 0) throw std::invalid_argument("BasedModel: out_features must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("BasedModel: num_blocks must be > 0");

    blocks_.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, seq_len, feature_dim, ffn_dim);
    }
    auto xavier_init = [&](Tensor& W) {
        double s = std::sqrt(6.0 / (double)(W.rows + W.cols));
        std::mt19937 gen(0xDA7A);
        std::uniform_real_distribution<> d(-s, s);
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) = d(gen);
    };
    xavier_init(W_in_);
    b_in_.fill(0.0);
    xavier_init(W_out_);
    b_out_.fill(0.0);
    zero_grad();
}

void BasedModel::zero_grad() {
    grad_W_in_.fill(0.0); grad_b_in_.fill(0.0);
    grad_W_out_.fill(0.0); grad_b_out_.fill(0.0);
    for (auto& b : blocks_) b.zero_grad();
    final_ln_.zero_grad();
}

Tensor BasedModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Input projection: in_proj = input · W_in + b_in
    // (W_in_ is shape (in_dim, d_model); input is (n, in_dim); result is (n, d_model))
    last_in_proj_ = input * W_in_;
    for (size_t j = 0; j < (size_t)W_in_.cols; ++j) {
        double bj = b_in_(0, j);
        for (size_t i = 0; i < input.rows; ++i) last_in_proj_(i, j) += bj;
    }
    Tensor x = last_in_proj_;
    for (auto& b : blocks_) {
        x = b.forward(x);
    }
    last_final_ln_ = final_ln_.forward(x);
    // Output projection: logits = last_final_ln · W_out + b_out
    // (W_out_ is shape (d_model, out_features); last_final_ln is (n, d_model); result is (n, out_features))
    last_logits_ = last_final_ln_ * W_out_;
    for (size_t j = 0; j < (size_t)W_out_.cols; ++j) {
        double bj = b_out_(0, j);
        for (size_t i = 0; i < last_logits_.rows; ++i) last_logits_(i, j) += bj;
    }
    return last_logits_;
}

Tensor BasedModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    // d_logits = grad_output
    // d_final_ln = d_logits · W_out
    Tensor d_final_ln(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < out_features_; ++k) s += grad_output(i, k) * W_out_(j, k);
            d_final_ln(i, j) = s;
        }
    }
    // dW_out[j, k] += sum_i grad_output[i, k] * last_final_ln[i, j]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < out_features_; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += grad_output(i, k) * last_final_ln_(i, j);
            grad_W_out_(j, k) += s;
        }
    }
    for (size_t k = 0; k < out_features_; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += grad_output(i, k);
        grad_b_out_(0, k) += s;
    }
    // d_last_block_out = final_ln.backward(d_final_ln)
    Tensor d_block_out = final_ln_.backward(d_final_ln, 0.0);
    // Run block backward in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_block_out = it->backward(d_block_out, 0.0);
    }
    // d_in_proj = d_block_out (residual through input projection)
    Tensor d_in_proj = d_block_out;
    // d_input = d_in_proj · W_in
    Tensor d_input(n, W_in_.rows);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < (size_t)W_in_.rows; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k) s += d_in_proj(i, k) * W_in_(j, k);
            d_input(i, j) = s;
        }
    }
    // dW_in[j, k] += sum_i d_in_proj[i, k] * last_input[i, j]
    for (size_t j = 0; j < (size_t)W_in_.rows; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += d_in_proj(i, k) * last_input_(i, j);
            grad_W_in_(j, k) += s;
        }
    }
    for (size_t k = 0; k < d_model_; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += d_in_proj(i, k);
        grad_b_in_(0, k) += s;
    }
    return d_input;
}

void BasedModel::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j) W(i, j) -= learning_rate * gW(i, j);
    };
    auto sgd_b = [&](Tensor& b, const Tensor& gb) {
        for (size_t j = 0; j < b.cols; ++j) b(0, j) -= learning_rate * gb(0, j);
    };
    sgd(W_in_, grad_W_in_);  sgd_b(b_in_, grad_b_in_);
    sgd(W_out_, grad_W_out_); sgd_b(b_out_, grad_b_out_);
    for (auto& b : blocks_) b.update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
}

std::vector<Tensor*> BasedModel::parameters() {
    return {&W_in_, &b_in_, &W_out_, &b_out_};
}
std::vector<Tensor*> BasedModel::gradients() {
    return {&grad_W_in_, &grad_b_in_, &grad_W_out_, &grad_b_out_};
}
