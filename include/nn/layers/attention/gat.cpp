// GAT: Graph Attention Network layer.
// Veličković et al. "Graph Attention Networks" ICLR 2018.
//
// Per-head math is in the header docstring. Three gradient paths flow back from h'_i:
//   (1) h'_i = LeakyReLU(sum_j α_ij Wh_j)  → dL/dα, dL/dWh (direct path)
//   (2) e_ij = a^T [Wh_i || Wh_j]          → dL/dWh via attention scores
//   (3) Wh = X @ W^T                       → dL/dX (input) and dL/dW
// All three are summed.

#include "gat.h"
#include <cmath>
#include <cstdio>
#include <random>

GATLayer::GATLayer(size_t in_features, size_t out_features, size_t num_heads, bool concat_heads)
    : num_heads_(num_heads), concat_heads_(concat_heads),
      in_features_(in_features), out_features_(out_features),
      last_output_(1, out_features) {

    if (concat_heads_) {
        // out_features = num_heads * head_dim
        if (out_features % num_heads != 0) {
            // Fall back: set head_dim = 1, then num_heads must equal out_features
            head_dim_ = 1;
        } else {
            head_dim_ = out_features / num_heads;
        }
    } else {
        // Average heads: each head outputs out_features
        head_dim_ = out_features;
    }

    std::mt19937 gen(123);
    double std_w = std::sqrt(2.0 / static_cast<double>(in_features));
    double std_a = std::sqrt(2.0 / static_cast<double>(2 * head_dim_));
    std::normal_distribution<> nd(0.0, 1.0);

    heads_.resize(num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        GATHeadParams hp;
        hp.W = Tensor(head_dim_, in_features);
        for (size_t i = 0; i < head_dim_; ++i)
            for (size_t j = 0; j < in_features; ++j)
                hp.W(i, j) = nd(gen) * std_w;

        hp.a = Tensor(2 * head_dim_, 1);
        for (size_t i = 0; i < 2 * head_dim_; ++i)
            hp.a(i, 0) = nd(gen) * std_a;

        hp.grad_W = Tensor(head_dim_, in_features);
        hp.grad_a = Tensor(2 * head_dim_, 1);
        heads_[h] = std::move(hp);
    }
}

Tensor GATLayer::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GATLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (N, in_features)
    // adj:   (N, N)
    size_t N = input.rows;
    const double leaky_slope = 0.2;

    last_input_ = input;
    adj_ = adj;

    last_Wh_heads_.assign(num_heads_, Tensor(N, head_dim_));
    last_alpha_ = Tensor(N, N * num_heads_);
    last_e_ = Tensor(N, N * num_heads_);
    last_head_pre_ = Tensor(N, head_dim_ * num_heads_);

    std::vector<Tensor> head_outputs(num_heads_, Tensor(N, head_dim_));

    for (size_t h = 0; h < num_heads_; ++h) {
        const Tensor& W = heads_[h].W;
        const Tensor& a = heads_[h].a;

        // Wh = input @ W^T   → (N, head_dim)
        // Wh[i][j] = sum_k input[i][k] * W[j][k]
        Tensor Wh(N, head_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < in_features_; ++k)
                    sum += input(i, k) * W(j, k);
                Wh(i, j) = sum;
            }
        last_Wh_heads_[h] = Wh;

        // Compute e_ij = LeakyReLU( a^T [Wh_i || Wh_j] ) for all i, j.
        // Mask non-neighbors (adj[i][j] == 0 → e = -inf so softmax -> 0).
        Tensor e(N, N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (adj(i, j) < 1e-9) {
                    e(i, j) = -1e9;  // masked: will softmax to 0
                    continue;
                }
                double dot = 0.0;
                // First head_dim components: a[0..F'-1] · Wh_i
                for (size_t k = 0; k < head_dim_; ++k)
                    dot += a(k, 0) * Wh(i, k);
                // Second head_dim components: a[F'..2F'-1] · Wh_j
                for (size_t k = 0; k < head_dim_; ++k)
                    dot += a(head_dim_ + k, 0) * Wh(j, k);
                e(i, j) = (dot > 0.0) ? dot : leaky_slope * dot;  // LeakyReLU
            }
        }

        // Store pre-softmax e (post-LeakyReLU) for backward
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                last_e_(i, h * N + j) = e(i, j);

        // Row-softmax over j: α_ij = exp(e_ij) / sum_k exp(e_ik)
        for (size_t i = 0; i < N; ++i) {
            double max_e = e(i, 0);
            for (size_t j = 1; j < N; ++j) max_e = std::max(max_e, e(i, j));
            double sum_exp = 0.0;
            for (size_t j = 0; j < N; ++j) {
                e(i, j) = std::exp(e(i, j) - max_e);
                sum_exp += e(i, j);
            }
            if (sum_exp < 1e-30) sum_exp = 1e-30;  // safety
            for (size_t j = 0; j < N; ++j)
                e(i, j) /= sum_exp;
        }

        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                last_alpha_(i, h * N + j) = e(i, j);

        // head_pre[i][j] = sum_k α_ik * Wh[k][j]   (pre-LeakyReLU)
        Tensor head_pre(N, head_dim_);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k)
                    sum += e(i, k) * Wh(k, j);
                head_pre(i, j) = sum;
            }
        }
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                last_head_pre_(i, h * head_dim_ + j) = head_pre(i, j);

        // Apply LeakyReLU to head_pre to get final head output
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j) {
                double v = head_pre(i, j);
                head_outputs[h](i, j) = (v > 0.0) ? v : leaky_slope * v;
            }
    }

    // Combine heads: concat or average
    if (concat_heads_) {
        size_t total_dim = head_dim_ * num_heads_;
        last_output_ = Tensor(N, total_dim);
        for (size_t h = 0; h < num_heads_; ++h)
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < head_dim_; ++j)
                    last_output_(i, h * head_dim_ + j) = head_outputs[h](i, j);
    } else {
        last_output_ = head_outputs[0];
        for (size_t h = 1; h < num_heads_; ++h)
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < head_dim_; ++j)
                    last_output_(i, j) += head_outputs[h](i, j);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                last_output_(i, j) /= static_cast<double>(num_heads_);
    }

    return last_output_;
}

Tensor GATLayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features). For concat: each head's slice [h*head_dim, (h+1)*head_dim).
    // For average: each head receives grad_output / num_heads.
    size_t N = grad_output.rows;
    const double leaky_slope = 0.2;

    Tensor grad_input(N, in_features_);
    grad_input.fill(0.0);

    // For each head: build its own per-head grad, then backprop into W, a, and input.
    for (size_t h = 0; h < num_heads_; ++h) {
        const Tensor& Wh = last_Wh_heads_[h];
        Tensor& W = heads_[h].W;
        Tensor& a = heads_[h].a;
        Tensor& grad_W = heads_[h].grad_W;
        Tensor& grad_a = heads_[h].grad_a;

        // Extract alpha (N, N) for this head from cached last_alpha_
        Tensor alpha(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                alpha(i, j) = last_alpha_(i, h * N + j);

        // Extract pre-softmax LeakyReLU scores e_ij
        Tensor e(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                e(i, j) = last_e_(i, h * N + j);

        // === dL/d(head_output_after_LeakyReLU) ===
        // For this head, slice grad_output.
        Tensor grad_head_out(N, head_dim_);  // dL/d(head_output_pre_LeakyReLU)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double g;
                if (concat_heads_) {
                    g = grad_output(i, h * head_dim_ + j);
                } else {
                    g = grad_output(i, j) / static_cast<double>(num_heads_);
                }
                // Apply LeakyReLU' (the head_output = LeakyReLU(head_pre))
                double v = last_head_pre_(i, h * head_dim_ + j);
                double deriv = (v > 0.0) ? 1.0 : leaky_slope;
                grad_head_out(i, j) = g * deriv;
            }
        }

        // === Direct path: head_pre = α @ Wh ===
        // dL/dWh (direct)  : grad_Wh[j][k] = sum_i α_ij * grad_head_out[i][k]
        // dL/dα (direct)   : grad_alpha[i][j] = sum_k grad_head_out[i][k] * Wh[j][k]
        Tensor grad_Wh_direct(N, head_dim_);
        grad_Wh_direct.fill(0.0);
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < head_dim_; ++k) {
                double sum = 0.0;
                for (size_t i = 0; i < N; ++i)
                    sum += alpha(i, j) * grad_head_out(i, k);
                grad_Wh_direct(j, k) = sum;
            }

        Tensor grad_alpha_direct(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < head_dim_; ++k)
                    sum += grad_head_out(i, k) * Wh(j, k);
                grad_alpha_direct(i, j) = sum;
            }

        // === Backward through softmax (row-wise) ===
        // dL/de_ij = α_ij * (dL/dα_ij - sum_k α_ik * dL/dα_ik)
        Tensor grad_e(N, N);
        for (size_t i = 0; i < N; ++i) {
            double sum_contrib = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum_contrib += alpha(i, k) * grad_alpha_direct(i, k);
            for (size_t j = 0; j < N; ++j)
                grad_e(i, j) = alpha(i, j) * (grad_alpha_direct(i, j) - sum_contrib);
        }

        // === Backward through LeakyReLU (e_ij = LeakyReLU(dot_ij)) ===
        Tensor grad_dot(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) {
                double leaku_deriv = (e(i, j) > 0.0) ? 1.0 : leaky_slope;
                grad_dot(i, j) = grad_e(i, j) * leaku_deriv;
            }

        // === dL/dW_h via two paths ===
        // (1) Direct: dL/dW_h[j][k] += sum_i grad_Wh_direct[j][k] * input[i][k]
        //     i.e. dL/dW = grad_Wh_direct.T @ input
        // (2) Indirect via attention scores:
        //     dot_ij = sum_k a[k] * Wh_i[k] + sum_k a[F'+k] * Wh_j[k]
        //     dL/dWh_i[k] += sum_j grad_dot(i,j) * a[k]
        //     dL/dWh_j[k] += sum_i grad_dot(i,j) * a[F'+k]
        //     When i==j, dot_ii uses Wh_i twice with the SAME a vector; partial
        //     of dot_ii w.r.t. Wh_i[k] is a[k] + a[F'+k] — we sum both halves
        //     rather than masking, which is correct.
        Tensor grad_Wh_indirect(N, head_dim_);
        grad_Wh_indirect.fill(0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < head_dim_; ++k) {
                double sum_i = 0.0;
                for (size_t j = 0; j < N; ++j)
                    sum_i += grad_dot(i, j) * a(k, 0);
                grad_Wh_indirect(i, k) += sum_i;
            }
        }
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < head_dim_; ++k) {
                double sum_j = 0.0;
                for (size_t i = 0; i < N; ++i)
                    sum_j += grad_dot(i, j) * a(head_dim_ + k, 0);
                grad_Wh_indirect(j, k) += sum_j;
            }
        }

        Tensor grad_Wh(N, head_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                grad_Wh(i, j) = grad_Wh_direct(i, j) + grad_Wh_indirect(i, j);

        // dL/dW_h = grad_Wh.T @ input  → (head_dim, in_features)
        for (size_t i = 0; i < head_dim_; ++i) {
            for (size_t k = 0; k < in_features_; ++k) {
                double sum = 0.0;
                for (size_t n = 0; n < N; ++n)
                    sum += grad_Wh(n, i) * last_input_(n, k);
                grad_W(i, k) = sum;
            }
        }

        // === dL/dW_h via Wh = input @ W^T → dL/dinput += grad_Wh @ W  ===
        Tensor grad_input_h(N, in_features_);
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < in_features_; ++k) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j)
                    sum += grad_Wh(i, j) * W(j, k);
                grad_input_h(i, k) = sum;
            }
        for (size_t i = 0; i < N; ++i)
            for (size_t k = 0; k < in_features_; ++k)
                grad_input(i, k) += grad_input_h(i, k);

        // === dL/da via attention scores ===
        // dot_ij = a[k] * Wh_i[k] (k in [0, F'))  +  a[F'+k] * Wh_j[k] (k in [0, F'))
        // Mask: only consider j such that adj(i, j) > 0.
        grad_a.fill(0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (adj_(i, j) < 1e-9) continue;  // masked in forward
                // First half: a[k] contributes grad_dot(i,j) * Wh_i[k]
                for (size_t k = 0; k < head_dim_; ++k)
                    grad_a(k, 0) += grad_dot(i, j) * Wh(i, k);
                // Second half: a[F'+k] contributes grad_dot(i,j) * Wh_j[k]
                for (size_t k = 0; k < head_dim_; ++k)
                    grad_a(head_dim_ + k, 0) += grad_dot(i, j) * Wh(j, k);
            }
        }

        // === SGD update on W and a ===
        for (size_t i = 0; i < head_dim_; ++i)
            for (size_t j = 0; j < in_features_; ++j)
                W(i, j) -= learning_rate * grad_W(i, j);
        for (size_t i = 0; i < 2 * head_dim_; ++i)
            a(i, 0) -= learning_rate * grad_a(i, 0);
    }

    return grad_input;
}

void GATLayer::update_weights(double learning_rate) {
    // Weights are already updated in backward() (matches GCNLayer pattern).
    (void)learning_rate;
}

void GATLayer::zero_grad() {
    for (auto& hp : heads_) {
        hp.grad_W.fill(0.0);
        hp.grad_a.fill(0.0);
    }
}

std::vector<Tensor*> GATLayer::parameters() {
    std::vector<Tensor*> result;
    for (auto& hp : heads_) {
        result.push_back(&hp.W);
        result.push_back(&hp.a);
    }
    return result;
}

std::vector<Tensor*> GATLayer::gradients() {
    std::vector<Tensor*> result;
    for (auto& hp : heads_) {
        result.push_back(&hp.grad_W);
        result.push_back(&hp.grad_a);
    }
    return result;
}
