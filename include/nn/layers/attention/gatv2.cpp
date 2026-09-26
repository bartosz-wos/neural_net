// GATv2: Improved Graph Attention Network layer.
// Brody, Alon, Yahav. "How Attentive are Graph Attention Networks?" ICLR 2023.
// https://arxiv.org/abs/2105.14491
//
// Per-head math is in the header docstring. Three gradient paths flow back from h'_i:
//   (1) h'_i = LeakyReLU(sum_j α_ij Wh_j)     → dL/dα, dL/dWh (direct path)
//   (2) e_ij = a^T · (W · [h_i || h_j])       → dL/dWh via attention scores (indirect)
//   (3) Wh = X @ W^T                          → dL/dX (input) and dL/dW
// All three are summed.
//
// The KEY difference from GAT v1 is that v2's score uses a SINGLE W applied to the
// concat input (not two half-Ws), and a SINGLE a of dimension head_dim (not 2*head_dim).
// Both halves of the indirect contribution to grad_Wh use the SAME a[k] (not a[k] and
// a[F'+k]).

#include "gatv2.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

GATv2Layer::GATv2Layer(size_t in_features, size_t out_features, size_t num_heads, bool concat_heads)
    : num_heads_(num_heads), concat_heads_(concat_heads),
      in_features_(in_features), out_features_(out_features),
      last_output_(1, out_features) {

    if (in_features == 0)
        throw std::invalid_argument("GATv2Layer: in_features must be > 0");
    if (out_features == 0)
        throw std::invalid_argument("GATv2Layer: out_features must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("GATv2Layer: num_heads must be > 0");

    if (concat_heads_) {
        // out_features = num_heads * head_dim
        if (out_features % num_heads != 0) {
            head_dim_ = 1;
        } else {
            head_dim_ = out_features / num_heads;
        }
    } else {
        head_dim_ = out_features;
    }

    std::mt19937 gen(123);
    double std_w = std::sqrt(2.0 / static_cast<double>(in_features));
    double std_a = std::sqrt(2.0 / static_cast<double>(head_dim_));
    std::normal_distribution<> nd(0.0, 1.0);

    heads_.resize(num_heads_);
    for (size_t h = 0; h < num_heads_; ++h) {
        GATv2HeadParams hp;
        hp.W = Tensor(head_dim_, in_features);
        for (size_t i = 0; i < head_dim_; ++i)
            for (size_t j = 0; j < in_features; ++j)
                hp.W(i, j) = nd(gen) * std_w;

        hp.a = Tensor(head_dim_, 1);  // single head_dim, NOT 2*head_dim (v2 distinction)
        for (size_t i = 0; i < head_dim_; ++i)
            hp.a(i, 0) = nd(gen) * std_a;

        hp.grad_W = Tensor(head_dim_, in_features);
        hp.grad_a = Tensor(head_dim_, 1);
        heads_[h] = std::move(hp);
    }
}

Tensor GATv2Layer::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GATv2Layer::forward_with_adj(const Tensor& input, const Tensor& adj) {
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
                    e(i, j) = -1e9;
                    continue;
                }
                // V2: a is F'-dimensional, applied to the 2F'-dim concat [Wh_i || Wh_j].
                // We compute it as sum over k of a[k] * (Wh_i[k] for k < F', Wh_j[k-F'] for k >= F').
                // Since a has F' components, the dot over 2F' is:
                //   score = sum_{k=0}^{F'-1} a[k] * Wh_i[k]  +  sum_{k=0}^{F'-1} a[k] * Wh_j[k]
                // Both halves use the SAME a[k] (this is v2's distinguishing property).
                double dot = 0.0;
                for (size_t k = 0; k < head_dim_; ++k)
                    dot += a(k, 0) * Wh(i, k);    // i-half
                for (size_t k = 0; k < head_dim_; ++k)
                    dot += a(k, 0) * Wh(j, k);    // j-half (SAME a[k])
                e(i, j) = (dot > 0.0) ? dot : leaky_slope * dot;
            }
        }

        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                last_e_(i, h * N + j) = e(i, j);

        // Row-softmax over j
        for (size_t i = 0; i < N; ++i) {
            double max_e = e(i, 0);
            for (size_t j = 1; j < N; ++j) max_e = std::max(max_e, e(i, j));
            double sum_exp = 0.0;
            for (size_t j = 0; j < N; ++j) {
                e(i, j) = std::exp(e(i, j) - max_e);
                sum_exp += e(i, j);
            }
            if (sum_exp < 1e-30) sum_exp = 1e-30;
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

        // Apply LeakyReLU
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j) {
                double v = head_pre(i, j);
                head_outputs[h](i, j) = (v > 0.0) ? v : leaky_slope * v;
            }
    }

    // Combine heads
    if (concat_heads_) {
        size_t total_dim = head_dim_ * num_heads_;
        last_output_ = Tensor(N, total_dim);
        for (size_t h = 0; h < num_heads_; ++h)
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < head_dim_; ++j)
                    last_output_(i, h * head_dim_ + j) = head_outputs[h](i, j);
    } else {
        last_output_ = head_outputs[0];
        for (size_t h = 1; h < num_heads_; ++h) {
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < head_dim_; ++j)
                    last_output_(i, j) += head_outputs[h](i, j);
        }
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                last_output_(i, j) /= static_cast<double>(num_heads_);
    }

    return last_output_;
}

Tensor GATv2Layer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features)
    size_t N = grad_output.rows;
    const double leaky_slope = 0.2;

    Tensor grad_input(N, in_features_);
    grad_input.fill(0.0);

    for (size_t h = 0; h < num_heads_; ++h) {
        const Tensor& Wh = last_Wh_heads_[h];
        Tensor& W = heads_[h].W;
        Tensor& a = heads_[h].a;
        Tensor& grad_W = heads_[h].grad_W;
        Tensor& grad_a = heads_[h].grad_a;

        Tensor alpha(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                alpha(i, j) = last_alpha_(i, h * N + j);

        Tensor e(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                e(i, j) = last_e_(i, h * N + j);

        // === dL/d(head_pre) ===
        Tensor grad_head_out(N, head_dim_);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double g;
                if (concat_heads_) {
                    g = grad_output(i, h * head_dim_ + j);
                } else {
                    g = grad_output(i, j) / static_cast<double>(num_heads_);
                }
                double v = last_head_pre_(i, h * head_dim_ + j);
                double deriv = (v > 0.0) ? 1.0 : leaky_slope;
                grad_head_out(i, j) = g * deriv;
            }
        }

        // === Direct path: head_pre = α @ Wh ===
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

        // === Softmax backward ===
        Tensor grad_e(N, N);
        for (size_t i = 0; i < N; ++i) {
            double sum_contrib = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum_contrib += alpha(i, k) * grad_alpha_direct(i, k);
            for (size_t j = 0; j < N; ++j)
                grad_e(i, j) = alpha(i, j) * (grad_alpha_direct(i, j) - sum_contrib);
        }

        // === LeakyReLU backward ===
        Tensor grad_dot(N, N);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) {
                double leaku_deriv = (e(i, j) > 0.0) ? 1.0 : leaky_slope;
                grad_dot(i, j) = grad_e(i, j) * leaku_deriv;
            }

        // === Indirect contributions to grad_Wh via the score path ===
        // V2: s_ij = [Wh_i || Wh_j], score = a^T s_ij = sum_{k=0}^{F'-1} a[k]*(Wh_i[k] + Wh_j[k]).
        // So dWh_i[k] += sum_j grad_dot[i,j] * a[k]   (i-half, uses a[k])
        //    dWh_j[k] += sum_i grad_dot[i,j] * a[k]   (j-half, SAME a[k])
        // Both halves use the SAME a[k] — this is v2's distinguishing property.
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
                    sum_j += grad_dot(i, j) * a(k, 0);  // SAME a[k], not a[F'+k] (v2)
                grad_Wh_indirect(j, k) += sum_j;
            }
        }

        Tensor grad_Wh(N, head_dim_);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                grad_Wh(i, j) = grad_Wh_direct(i, j) + grad_Wh_indirect(i, j);

        // === dL/dW via Wh = X @ W^T ===
        // V2: W is a single (F', in) matrix used for BOTH halves of the concat input.
        // grad_W[k, c] = sum_n grad_Wh[n, k] * input[n, c]   (single matmul, no half-split)
        for (size_t i = 0; i < head_dim_; ++i) {
            for (size_t k = 0; k < in_features_; ++k) {
                double sum = 0.0;
                for (size_t n = 0; n < N; ++n)
                    sum += grad_Wh(n, i) * last_input_(n, k);
                grad_W(i, k) = sum;
            }
        }

        // === dL/dinput via Wh = X @ W^T ===
        // V2: grad_input[n, c] = sum_k grad_Wh[n, k] * W[k, c] (matmul through single W).
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
        // V2: score = a^T s_ij where s_ij = [Wh_i || Wh_j].
        // grad_a[k] = sum_{i, j: adj(i,j)>0} grad_dot[i, j] * s_ij[k]
        //   for k < F': s_ij[k] = Wh_i[k]
        //   for k >= F': s_ij[k] = Wh_j[k - F']
        // But v2 has only ONE a of dimension F' (not 2F'). The a vector is the SAME
        // for both halves of s_ij — that is, a[k] for k in [0, F') is used in BOTH
        // the i-half dot product AND the j-half dot product (see forward).
        //
        // Mathematically: score = sum_{k=0}^{F'-1} a[k] * Wh_i[k] + sum_{k=0}^{F'-1} a[k] * Wh_j[k].
        // So d(a[k]) = sum_{i,j adj} grad_dot[i,j] * (Wh_i[k] + Wh_j[k]).
        grad_a.fill(0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (adj_(i, j) < 1e-9) continue;
                for (size_t k = 0; k < head_dim_; ++k) {
                    // V2: grad_a[k] += grad_dot[i,j] * (Wh_i[k] + Wh_j[k])
                    // Both halves contribute, but a[k] is one entry — the i-half and
                    // j-half contributions are SUMMED into a single a[k] entry.
                    grad_a(k, 0) += grad_dot(i, j) * (Wh(i, k) + Wh(j, k));
                }
            }
        }

        // === SGD update on W and a (inline, matching GAT v1 pattern) ===
        for (size_t i = 0; i < head_dim_; ++i)
            for (size_t j = 0; j < in_features_; ++j)
                W(i, j) -= learning_rate * grad_W(i, j);
        for (size_t i = 0; i < head_dim_; ++i)
            a(i, 0) -= learning_rate * grad_a(i, 0);
    }

    return grad_input;
}

void GATv2Layer::update_weights(double learning_rate) {
    (void)learning_rate;
}

void GATv2Layer::zero_grad() {
    for (auto& hp : heads_) {
        hp.grad_W.fill(0.0);
        hp.grad_a.fill(0.0);
    }
}

std::vector<Tensor*> GATv2Layer::parameters() {
    std::vector<Tensor*> result;
    for (auto& hp : heads_) {
        result.push_back(&hp.W);
        result.push_back(&hp.a);
    }
    return result;
}

std::vector<Tensor*> GATv2Layer::gradients() {
    std::vector<Tensor*> result;
    for (auto& hp : heads_) {
        result.push_back(&hp.grad_W);
        result.push_back(&hp.grad_a);
    }
    return result;
}

// ============================================================================
// GATv2Model — minimal stack of GATv2Layers.
// ============================================================================

GATv2Model::GATv2Model(size_t in_features, size_t hidden_features, size_t out_features,
                       size_t num_layers, size_t num_heads)
    : in_features_(in_features), hidden_features_(hidden_features),
      out_features_(out_features), num_layers_(num_layers), num_heads_(num_heads) {
    if (in_features == 0) throw std::invalid_argument("GATv2Model: in_features must be > 0");
    if (hidden_features == 0) throw std::invalid_argument("GATv2Model: hidden_features must be > 0");
    if (out_features == 0) throw std::invalid_argument("GATv2Model: out_features must be > 0");
    if (num_layers == 0) throw std::invalid_argument("GATv2Model: num_layers must be > 0");
    if (num_heads == 0) throw std::invalid_argument("GATv2Model: num_heads must be > 0");

    layers_.reserve(num_layers_);
    // First layer takes in_features_, all others hidden_features_
    layers_.push_back(GATv2Layer(in_features_, hidden_features_, num_heads_, true));
    for (size_t l = 1; l < num_layers_; ++l) {
        layers_.push_back(GATv2Layer(hidden_features_, hidden_features_, num_heads_, true));
    }
}

Tensor GATv2Model::forward(const Tensor& input) {
    (void)input;
    return Tensor();
}

Tensor GATv2Model::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = input;
    adj_ = adj;
    layer_inputs_.clear();
    Tensor cur = input;
    for (size_t l = 0; l < num_layers_; ++l) {
        layer_inputs_.push_back(cur);
        cur = layers_[l].forward_with_adj(cur, adj);
    }
    return cur;
}

Tensor GATv2Model::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = grad_output;
    for (size_t l = num_layers_; l > 0; --l) {
        grad = layers_[l - 1].backward(grad, learning_rate);
    }
    return grad;
}

void GATv2Model::update_weights(double learning_rate) {
    for (auto& l : layers_) l.update_weights(learning_rate);
}

void GATv2Model::zero_grad() {
    for (auto& l : layers_) l.zero_grad();
}

std::vector<Tensor*> GATv2Model::parameters() {
    std::vector<Tensor*> result;
    for (auto& l : layers_) {
        auto p = l.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

std::vector<Tensor*> GATv2Model::gradients() {
    std::vector<Tensor*> result;
    for (auto& l : layers_) {
        auto g = l.gradients();
        result.insert(result.end(), g.begin(), g.end());
    }
    return result;
}
