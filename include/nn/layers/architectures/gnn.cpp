#include "gnn.h"
#include <cmath>
#include <cstdio>
#include <random>

// === GCNLayer ===

GCNLayer::GCNLayer(size_t in_features, size_t out_features)
    : W_(in_features, out_features), last_output_(1, out_features) {}

Tensor GCNLayer::forward(const Tensor& input) {
    // Requires adjacency matrix — use forward_with_adj
    (void)input;
    return last_output_;
}

Tensor GCNLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // adj: (num_nodes, num_nodes) adjacency matrix
    // input: (num_nodes, in_features)
    // Normalize: D^{-1/2} A D^{-1/2}
    size_t N = adj.rows;

    // Compute degree vector
    Tensor d(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj[i][j];
        d[i][0] = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }

    // Normalized adjacency: D^{-1/2} A D^{-1/2}
    Tensor adj_norm(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            adj_norm[i][j] = d[i][0] * adj[i][j] * d[j][0];

    adj_norm_ = adj_norm; // store for backward

    // AXW: (N,N) @ (N, in_features) @ (in_features, out_features)

    // AXW: (N,N) @ (N, in_features) @ (in_features, out_features)
    // Step 1: AX = adj_norm @ input
    Tensor AX(N, input.cols);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < input.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += adj_norm[i][k] * input[k][j];
            AX[i][j] = sum;
        }

    // Step 2: AXW = AX @ W_
    Tensor AW(N, W_.get_weights().cols);
    const Tensor& W = W_.get_weights();
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < W.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < AX.cols; ++k)
                sum += AX[i][k] * W[k][j];
            AW[i][j] = sum;
        }

    // ReLU
    relu_mask_.resize(N);
    for (size_t i = 0; i < AW.rows; ++i) {
        relu_mask_[i].resize(AW.cols);
        for (size_t j = 0; j < AW.cols; ++j) {
            relu_mask_[i][j] = (AW[i][j] > 0.0) ? 1.0 : 0.0;
            AW[i][j] = std::max(0.0, AW[i][j]);
        }
    }

    last_output_ = AW;
    return last_output_;
}

Tensor GCNLayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (num_nodes, out_features)
    // Forward: input → AX = adj_norm @ input → AW = AX @ W → ReLU → output
    // Backprop: dL/dAW (ReLU) → dL/dAX = dL/dAW @ W^T → dL/dinput = adj_norm^T @ dL/dAX
    // ReLU gradient: elementwise multiply by relu_mask (AW > 0)
    size_t N = grad_output.rows;
    const Tensor& W = W_.get_weights();

    // Apply ReLU mask to grad_output to get grad w.r.t. AW
    // relu_mask_[i][j] = 1 if last_AW_[i][j] > 0, else 0
    Tensor grad_AW(N, W.cols);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < (size_t)W.cols; ++j)
            grad_AW[i][j] = (i < relu_mask_.size() && j < relu_mask_[i].size() && relu_mask_[i][j])
                            ? grad_output[i][j] : 0.0;

    // dL/dAX = grad_AW @ W^T
    Tensor grad_AX(N, W.rows);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < W.rows; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < W.cols; ++k)
                sum += grad_AW[i][k] * W[j][k];
            grad_AX[i][j] = sum;
        }

    // dL/dinput = adj_norm^T @ grad_AX
    Tensor grad_input(N, W.rows);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < W.rows; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += adj_norm_[k][i] * grad_AX[k][j];
            grad_input[i][j] = sum;
        }

    // Update W via Dense's backward on the projected input
    // W_.backward expects grad_wrt_output (grad_AW) and computes grad_wrt_input = grad_AW @ W^T
    // Then updates W using last_input_ (which was the AX input)
    W_.backward(grad_AW, learning_rate);

    return grad_input;
}

void GCNLayer::update_weights(double learning_rate) {
    W_.update_weights(learning_rate);
}

void GCNLayer::zero_grad() { W_.zero_grad(); }

std::vector<Tensor*> GCNLayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GCNLayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_.gradients()) result.push_back(g);
    return result;
}

// === GATLayer ===
//
// Veličković et al. "Graph Attention Networks" ICLR 2018.
//
// Per head h, with head_dim F':
//   Wh_i = W_h @ h_i                 ∈ R^{F'}
//   e_ij = LeakyReLU( a_h^T [ Wh_i || Wh_j ] )   ∈ R     (a_h ∈ R^{2F'})
//   α_ij = softmax_j( e_ij )         over j ∈ N(i) ∪ {i}   (we use row-softmax)
//   h'_i = LeakyReLU( sum_j α_ij Wh_j )            ∈ R^{F'}
//
// Multi-head: concat or average across heads. Output dim = out_features.
//
// Backward (key idea — corrected):
//   Three paths flow back from h'_i:
//     (1) h'_i = LeakyReLU(sum_j α_ij Wh_j)  → dL/dα, dL/dWh (direct path)
//     (2) e_ij = a^T [Wh_i || Wh_j]  → dL/dWh via attention scores
//     (3) Wh = X @ W^T                → dL/dX (input) and dL/dW
//   All three must be summed.
//
// LeakyReLU slope α_LR = 0.2 (per paper).

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
        // Equivalent to: Wh[i] = sum_k input[i][k] * W.T[k]
        // Use: Wh[i][j] = sum_k input[i][k] * W[j][k]
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
        // dL/ddot_ij = grad_e(i,j) * 1{raw > 0} + grad_e(i,j) * slope{raw ≤ 0}
        // where raw is the un-LeakyRelu'd value. We have e_ij = LeakyReLU(raw_ij).
        // So: leaku_deriv = 1 if e_ij > 0, else slope. Then dL/d(dot_ij) = grad_e(i,j) * leaku_deriv.
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
        //     dL/dWh_j[k] += sum_i grad_dot(i,j) * a[F'+k]  (for i != j; for i==j,
        //            we must NOT double-count, since Wh_i == Wh_j)
        //     Note: when i==j, dot_ii uses Wh_i twice with the SAME a vector.
        //     The actual partial of dot_ii w.r.t. Wh_i[k] is a[k] + a[F'+k].
        //     So summing both halves is correct, but we should NOT mask i==j.
        //     The direct α path uses α_ii; the indirect path uses dot_ii.
        //     Together: dL/dWh = (direct) + (indirect via a).
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
        // Mask: only consider j such that adj(i, j) > 0. For i==j, both halves apply.
        // (When adj(i,i)==0, dot_ii contribution to loss is zero because e_ii = -1e9.)
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

// === GraphNetwork ===

GraphNetwork::GraphNetwork(const std::vector<size_t>& hidden_dims, bool use_gat)
    : use_gat_(use_gat), last_output_(1, 1) {

    for (size_t i = 0; i + 1 < hidden_dims.size(); ++i) {
        if (use_gat)
            gat_layers_.emplace_back(hidden_dims[i], hidden_dims[i + 1], 4, true);
        else
            gcn_layers_.emplace_back(hidden_dims[i], hidden_dims[i + 1]);
    }
}

Tensor GraphNetwork::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GraphNetwork::forward_with_adj(const Tensor& input, const Tensor& adj) {
    Tensor x = input;
    if (use_gat_) {
        for (auto& layer : gat_layers_)
            x = layer.forward_with_adj(x, adj);
    } else {
        for (auto& layer : gcn_layers_)
            x = layer.forward_with_adj(x, adj);
    }
    last_output_ = x;
    return last_output_;
}

Tensor GraphNetwork::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void GraphNetwork::update_weights(double learning_rate) {
    if (use_gat_) {
        for (auto& l : gat_layers_) l.update_weights(learning_rate);
    } else {
        for (auto& l : gcn_layers_) l.update_weights(learning_rate);
    }
}

void GraphNetwork::zero_grad() {
    if (use_gat_) {
        for (auto& l : gat_layers_) l.zero_grad();
    } else {
        for (auto& l : gcn_layers_) l.zero_grad();
    }
}

std::vector<Tensor*> GraphNetwork::parameters() {
    std::vector<Tensor*> result;
    if (use_gat_) {
        for (auto& l : gat_layers_)
            for (Tensor* p : l.parameters()) result.push_back(p);
    } else {
        for (auto& l : gcn_layers_)
            for (Tensor* p : l.parameters()) result.push_back(p);
    }
    return result;
}

std::vector<Tensor*> GraphNetwork::gradients() {
    std::vector<Tensor*> result;
    if (use_gat_) {
        for (auto& l : gat_layers_)
            for (Tensor* g : l.gradients()) result.push_back(g);
    } else {
        for (auto& l : gcn_layers_)
            for (Tensor* g : l.gradients()) result.push_back(g);
    }
    return result;
}
