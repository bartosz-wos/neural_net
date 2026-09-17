// ============================================================================
// Mixture of Softmaxes (MoS) — implementation
// ============================================================================
//
// Forward (matches Yang et al. 2018 §2):
//
//   z[h, k, j, i] = inv_temp * Σ_{c ∈ head h} Q[j, c] * K[i, c]
//                 + Σ_{c ∈ head h} B_z[k, c]
//   P[h, k, j, i] = softmax_i(z[h, k, j, i])
//   O[h, k, j, d] = Σ_i P[h, k, j, i] * V_k[k, i, h_off(d) + d]
//   g[h, j, k]    = Σ_{c ∈ head h} Q[j, c] * W_g[k, c] + B_g[k, 0]
//   Pi[h, j, k]   = softmax_k(g[h, j, k])
//   Y[h, j, d]    = Σ_k Pi[h, j, k] * O[h, k, j, d]
//   Y_full[j, c]  = W_o[c, :] @ concat_h Y[h, j, :] + b_o[c]
//
// All 3D+ tensors are stored as 2D Tensor with FLAT indexing.
// Conventions:
//   * W_v[K × d × d]: flat (K, d*d), index W_v(k, out_d * d + in_d). So the
//     linear projection is V_k[k, n, c_out] = Σ_{c_in} input[n, c_in] * W_v(k, c_out * d + c_in) + B_v[k, c_out].
//   * V_k[K × N × d]: flat (K, N*d), index V_k(k, n * d + c).
//   * d_O[H × K × N × dh]: flat (H, K * N * dh), index d_O(h, k * N * dh + j * dh + d).
//   * d_Pi[H × N × K]: flat (H, N * K), index d_Pi(h, j * K + k).
//   * d_P[H*K × N × N]: flat (H*K, N*N), index d_P(hk, j * N + i).
//   * d_V_k[K × N × d]: flat (K, N * d), index d_V_k(k, n * d + c).
//   * d_Z[H*K × N × N]: flat (H*K, N*N), index d_Z(hk, j * N + i).
//   * d_g[H × N × K]: flat (H, N * K), index d_g(h, j * K + k).
//   * O[H × N × dh]: flat (H, N * dh), index O(h, j * dh + d).
//   * Pi[H × N × K]: flat (H, N * K), index Pi(h, j * K + k).
//   * g_logits[H × N × K]: same as Pi.
// ============================================================================

#include "mixture_of_softmaxes.h"
#include <cmath>
#include <stdexcept>

MixtureOfSoftmaxesAttention::MixtureOfSoftmaxesAttention(size_t d_model, size_t num_heads, size_t K)
    : W_q(d_model, d_model), W_k(d_model, d_model), W_o(d_model, d_model),
      W_v(K, d_model * d_model), B_v(K, d_model),
      B_z(K, d_model), W_g(K, d_model), B_g(K, 1),
      grad_W_q(d_model, d_model), grad_b_q(1, d_model),
      grad_W_k(d_model, d_model), grad_b_k(1, d_model),
      grad_W_o(d_model, d_model), grad_b_o(1, d_model),
      grad_W_v(K, d_model * d_model), grad_B_v(K, d_model),
      grad_B_z(K, d_model), grad_W_g(K, d_model), grad_B_g(K, 1) {
    if (d_model == 0)
        throw std::invalid_argument("MixtureOfSoftmaxes: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MixtureOfSoftmaxes: num_heads must be > 0");
    if (K == 0)
        throw std::invalid_argument("MixtureOfSoftmaxes: K (num_branches) must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("MixtureOfSoftmaxes: d_model must be divisible by num_heads");
    if (d_model < num_heads)
        throw std::invalid_argument("MixtureOfSoftmaxes: d_model must be >= num_heads");

    d_model_ = d_model;
    num_heads_ = num_heads;
    K_ = K;
    head_dim_ = d_model_ / num_heads_;
    inv_temp_ = 1.0 / std::sqrt(double(head_dim_));
    size_t_K_ = K_;

    W_g.fill(0.0);
    B_g.fill(0.0);
}

Tensor MixtureOfSoftmaxesAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("MixtureOfSoftmaxes: input.cols must equal d_model");
    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;
    last_input_ = input;

    last_Q_ = W_q.forward(input);  // (N, d_model)
    last_K_ = W_k.forward(input);  // (N, d_model)

    // V_k[k, n, c] flat-indexed as V_k_(k, n * d_model + c)
    // = Σ_c' input[n, c'] * W_v(k, c * d_model + c') + B_v(k, c)
    last_V_k_ = Tensor(K_, N * d_model_);
    for (size_t k = 0; k < K_; ++k) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < d_model_; ++c) {
                double s = B_v(k, c);
                for (size_t cp = 0; cp < d_model_; ++cp)
                    s += input(n, cp) * W_v(k, c * d_model_ + cp);
                last_V_k_(k, n * d_model_ + c) = s;
            }
        }
    }

    // Per-(head, branch) logits + softmax. last_Z_(h*K + k, j*N + i) etc.
    last_Z_ = Tensor(H * K_, N * N);
    last_P_ = Tensor(H * K_, N * N);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t k = 0; k < K_; ++k) {
            const size_t hk = h * K_ + k;
            for (size_t j = 0; j < N; ++j) {
                double bias_sum = 0.0;
                for (size_t d = 0; d < dh; ++d) bias_sum += B_z(k, h_off + d);
                for (size_t i = 0; i < N; ++i) {
                    double s = 0.0;
                    for (size_t d = 0; d < dh; ++d)
                        s += last_Q_(j, h_off + d) * last_K_(i, h_off + d);
                    last_Z_(hk, j * N + i) = s * inv_temp_ + bias_sum;
                }
                double maxv = last_Z_(hk, j * N + 0);
                for (size_t i = 1; i < N; ++i)
                    if (last_Z_(hk, j * N + i) > maxv) maxv = last_Z_(hk, j * N + i);
                double sumexp = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    double e = std::exp(last_Z_(hk, j * N + i) - maxv);
                    last_P_(hk, j * N + i) = e;
                    sumexp += e;
                }
                if (sumexp < 1e-30) sumexp = 1e-30;
                for (size_t i = 0; i < N; ++i)
                    last_P_(hk, j * N + i) /= sumexp;
            }
        }
    }

    // Gating.
    last_g_logits_ = Tensor(H, N * K_);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < K_; ++k) {
                double s = B_g(k, 0);
                for (size_t d = 0; d < dh; ++d)
                    s += last_Q_(j, h_off + d) * W_g(k, h_off + d);
                last_g_logits_(h, j * K_ + k) = s;
            }
    }
    last_Pi_ = Tensor(H, N * K_);
    for (size_t h = 0; h < H; ++h)
        for (size_t j = 0; j < N; ++j) {
            double maxv = last_g_logits_(h, j * K_ + 0);
            for (size_t k = 1; k < K_; ++k)
                if (last_g_logits_(h, j * K_ + k) > maxv) maxv = last_g_logits_(h, j * K_ + k);
            double sumexp = 0.0;
            for (size_t k = 0; k < K_; ++k) {
                double e = std::exp(last_g_logits_(h, j * K_ + k) - maxv);
                last_Pi_(h, j * K_ + k) = e;
                sumexp += e;
            }
            if (sumexp < 1e-30) sumexp = 1e-30;
            for (size_t k = 0; k < K_; ++k) last_Pi_(h, j * K_ + k) /= sumexp;
        }

    // Y[h, j, d] flat-indexed as last_O_(h, j * dh + d) =
    //   Σ_k Pi[h, j, k] * O_branch[h, k, j, d]
    // where O_branch[h, k, j, d] = Σ_i P[hk, j*N+i] * V_k_(k, i*d + h_off + d).
    // Cache last_O_branch_(h, k * N * dh + j * dh + d) for the d_Pi chain.
    last_O_branch_ = Tensor(H, K_ * N * dh);
    last_O_ = Tensor(H, N * dh);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d) {
                double s = 0.0;
                for (size_t k = 0; k < K_; ++k) {
                    double branch = 0.0;
                    for (size_t i = 0; i < N; ++i)
                        branch += last_P_(h * K_ + k, j * N + i) *
                                  last_V_k_(k, i * d_model_ + h_off + d);
                    last_O_branch_(h, k * N * dh + j * dh + d) = branch;
                    s += last_Pi_(h, j * K_ + k) * branch;
                }
                last_O_(h, j * dh + d) = s;
            }
    }

    // Concat + W_o.
    Tensor flat(N, d_model_);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                flat(j, h_off + d) = last_O_(h, j * dh + d);
    }
    last_flat_ = flat;
    last_Y_ = W_o.forward(flat);
    return last_Y_;
}

Tensor MixtureOfSoftmaxesAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != last_input_.rows || grad_output.cols != d_model_)
        throw std::invalid_argument("MixtureOfSoftmaxes: grad_output shape mismatch");
    const size_t N = last_input_.rows;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;

    Tensor dY_flat = W_o.backward(grad_output, 0.0);
    W_o.grad_weights.fill(0.0);
    W_o.grad_bias.fill(0.0);

    // dY[h, j, d] = dY_flat(j, h_off + d)
    Tensor dY(H, N * dh);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t j = 0; j < N; ++j)
            for (size_t d = 0; d < dh; ++d)
                dY(h, j * dh + d) = dY_flat(j, h_off + d);
    }

    // d_O flat (H, K*N*dh), d_Pi flat (H, N*K).
    // d_Pi[h,j,k] = Σ_d O_branch[h,k,j,d] * dY[h,j,d] (per-branch O, not Y)
    Tensor d_O(H, K_ * N * dh);
    Tensor d_Pi(H, N * K_);
    for (size_t h = 0; h < H; ++h)
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K_; ++k)
                for (size_t d = 0; d < dh; ++d)
                    d_O(h, k * N * dh + j * dh + d) =
                        last_Pi_(h, j * K_ + k) * dY(h, j * dh + d);
            for (size_t k = 0; k < K_; ++k) {
                double s = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    s += last_O_branch_(h, k * N * dh + j * dh + d) * dY(h, j * dh + d);
                d_Pi(h, j * K_ + k) = s;
            }
        }

    // d_P, d_V_k. d_V_k is indexed by KEY position i (not query j).
    Tensor d_P(H * K_, N * N);
    Tensor d_V_k(K_, N * d_model_);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t k = 0; k < K_; ++k) {
            const size_t hk = h * K_ + k;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < N; ++i) {
                    double s = 0.0;
                    for (size_t d = 0; d < dh; ++d)
                        s += last_V_k_(k, i * d_model_ + h_off + d) *
                             d_O(h, k * N * dh + j * dh + d);
                    d_P(hk, j * N + i) = s;
                }
            }
            for (size_t i = 0; i < N; ++i)
                for (size_t d = 0; d < dh; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < N; ++j)
                        acc += last_P_(hk, j * N + i) * d_O(h, k * N * dh + j * dh + d);
                    d_V_k(k, i * d_model_ + h_off + d) += acc;
                }
        }
    }

    // Softmax backward on Z rows.
    Tensor d_Z(H * K_, N * N);
    for (size_t hk = 0; hk < H * K_; ++hk)
        for (size_t j = 0; j < N; ++j) {
            double dot = 0.0;
            for (size_t i = 0; i < N; ++i)
                dot += last_P_(hk, j * N + i) * d_P(hk, j * N + i);
            for (size_t i = 0; i < N; ++i)
                d_Z(hk, j * N + i) =
                    last_P_(hk, j * N + i) * (d_P(hk, j * N + i) - dot);
        }

    // Gating softmax backward.
    Tensor d_g(H, N * K_);
    for (size_t h = 0; h < H; ++h)
        for (size_t j = 0; j < N; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < K_; ++k)
                dot += last_Pi_(h, j * K_ + k) * d_Pi(h, j * K_ + k);
            for (size_t k = 0; k < K_; ++k)
                d_g(h, j * K_ + k) = last_Pi_(h, j * K_ + k) *
                                       (d_Pi(h, j * K_ + k) - dot);
        }

    // grad_B_z[k, h_off(d) + d] += Σ_{j, i} d_Z[h, k, j, i]
    for (size_t k = 0; k < K_; ++k)
        for (size_t h = 0; h < H; ++h) {
            const size_t h_off = h * dh;
            const size_t hk = h * K_ + k;
            for (size_t d = 0; d < dh; ++d) {
                double s = 0.0;
                for (size_t j = 0; j < N; ++j)
                    for (size_t i = 0; i < N; ++i)
                        s += d_Z(hk, j * N + i);
                grad_B_z(k, h_off + d) += s;
            }
        }

    // grad_B_v[k, c] += Σ_j d_V_k[k, j * d + c]
    for (size_t k = 0; k < K_; ++k)
        for (size_t c = 0; c < d_model_; ++c)
            for (size_t j = 0; j < N; ++j)
                grad_B_v(k, c) += d_V_k(k, j * d_model_ + c);

    // grad_W_v[k, c * d + cp] += Σ_j input[j, cp] * d_V_k[k, j * d + c]
    for (size_t k = 0; k < K_; ++k)
        for (size_t c = 0; c < d_model_; ++c)
            for (size_t cp = 0; cp < d_model_; ++cp)
                for (size_t j = 0; j < N; ++j)
                    grad_W_v(k, c * d_model_ + cp) += last_input_(j, cp) * d_V_k(k, j * d_model_ + c);

    // grad_W_g[k, c] += Σ_{h: c ∈ head h} Σ_j d_g[h, j, k] * Q[j, c]
    for (size_t k = 0; k < K_; ++k)
        for (size_t c = 0; c < d_model_; ++c)
            for (size_t h = 0; h < H; ++h) {
                const size_t h_off = h * dh;
                if (c < h_off || c >= h_off + dh) continue;
                for (size_t j = 0; j < N; ++j)
                    grad_W_g(k, c) += d_g(h, j * K_ + k) * last_Q_(j, c);
            }

    // grad_B_g[k, 0] += Σ_{h, j} d_g[h, j, k]
    for (size_t k = 0; k < K_; ++k)
        for (size_t h = 0; h < H; ++h)
            for (size_t j = 0; j < N; ++j)
                grad_B_g(k, 0) += d_g(h, j * K_ + k);

    // d_Q, d_K, grad_W_q / grad_b_q / grad_W_k / grad_b_k
    Tensor d_Q(N, d_model_);
    Tensor d_K(N, d_model_);
    for (size_t h = 0; h < H; ++h) {
        const size_t h_off = h * dh;
        for (size_t c = h_off; c < h_off + dh; ++c) {
            for (size_t j = 0; j < N; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < K_; ++k) {
                    const size_t hk = h * K_ + k;
                    for (size_t i = 0; i < N; ++i)
                        s += last_K_(i, c) * d_Z(hk, j * N + i);
                }
                double g_part = 0.0;
                for (size_t k = 0; k < K_; ++k)
                    g_part += W_g(k, c) * d_g(h, j * K_ + k);
                d_Q(j, c) = s * inv_temp_ + g_part;
            }
            for (size_t i = 0; i < N; ++i) {
                double s = 0.0;
                for (size_t k = 0; k < K_; ++k) {
                    const size_t hk = h * K_ + k;
                    for (size_t j = 0; j < N; ++j)
                        s += last_Q_(j, c) * d_Z(hk, j * N + i);
                }
                d_K(i, c) = s * inv_temp_;
            }
        }
    }

    // grad_W_q / grad_b_q / grad_W_k / grad_b_k
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t cp = 0; cp < d_model_; ++cp)
            for (size_t n = 0; n < N; ++n)
                grad_W_q(c, cp) += d_Q(n, c) * last_input_(n, cp);
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t n = 0; n < N; ++n)
            grad_b_q(0, c) += d_Q(n, c);
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t cp = 0; cp < d_model_; ++cp)
            for (size_t n = 0; n < N; ++n)
                grad_W_k(c, cp) += d_K(n, c) * last_input_(n, cp);
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t n = 0; n < N; ++n)
            grad_b_k(0, c) += d_K(n, c);

    // grad_W_o / grad_b_o from grad_output and last_flat_ (W_o input).
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t cc = 0; cc < d_model_; ++cc)
            for (size_t n = 0; n < N; ++n)
                grad_W_o(c, cc) += grad_output(n, c) * last_flat_(n, cc);
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t n = 0; n < N; ++n)
            grad_b_o(0, c) += grad_output(n, c);

    // d_input.
    Tensor d_input(N, d_model_);
    for (size_t n = 0; n < N; ++n)
        for (size_t cp = 0; cp < d_model_; ++cp) {
            double from_v = 0.0;
            for (size_t k = 0; k < K_; ++k)
                for (size_t c = 0; c < d_model_; ++c)
                    from_v += W_v(k, c * d_model_ + cp) * d_V_k(k, n * d_model_ + c);
            double from_q = 0.0;
            for (size_t c = 0; c < d_model_; ++c)
                from_q += d_Q(n, c) * W_q.weights(c, cp);
            double from_k = 0.0;
            for (size_t c = 0; c < d_model_; ++c)
                from_k += d_K(n, c) * W_k.weights(c, cp);
            d_input(n, cp) = from_v + from_q + from_k;
        }

    return d_input;
}

void MixtureOfSoftmaxesAttention::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& p, const Tensor& g) {
        for (size_t i = 0; i < p.data.size(); ++i)
            p.data[i] -= learning_rate * g.data[i];
    };
    sgd(W_q.weights, grad_W_q);
    sgd(W_q.bias,    grad_b_q);
    sgd(W_k.weights, grad_W_k);
    sgd(W_k.bias,    grad_b_k);
    sgd(W_o.weights, grad_W_o);
    sgd(W_o.bias,    grad_b_o);
    sgd(W_v,         grad_W_v);
    sgd(B_v,         grad_B_v);
    sgd(B_z,         grad_B_z);
    sgd(W_g,         grad_W_g);
    sgd(B_g,         grad_B_g);
}

void MixtureOfSoftmaxesAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_b_q.fill(0.0);
    grad_W_k.fill(0.0); grad_b_k.fill(0.0);
    grad_W_o.fill(0.0); grad_b_o.fill(0.0);
    grad_W_v.fill(0.0); grad_B_v.fill(0.0);
    grad_B_z.fill(0.0); grad_W_g.fill(0.0); grad_B_g.fill(0.0);
    W_q.zero_grad();
    W_k.zero_grad();
    W_o.zero_grad();
}

std::vector<Tensor*> MixtureOfSoftmaxesAttention::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_q.weights);
    p.push_back(&W_q.bias);
    p.push_back(&W_k.weights);
    p.push_back(&W_k.bias);
    p.push_back(&W_o.weights);
    p.push_back(&W_o.bias);
    p.push_back(&W_v);
    p.push_back(&B_v);
    p.push_back(&B_z);
    p.push_back(&W_g);
    p.push_back(&B_g);
    return p;
}

std::vector<Tensor*> MixtureOfSoftmaxesAttention::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&grad_W_q);
    g.push_back(&grad_b_q);
    g.push_back(&grad_W_k);
    g.push_back(&grad_b_k);
    g.push_back(&grad_W_o);
    g.push_back(&grad_b_o);
    g.push_back(&grad_W_v);
    g.push_back(&grad_B_v);
    g.push_back(&grad_B_z);
    g.push_back(&grad_W_g);
    g.push_back(&grad_B_g);
    return g;
}