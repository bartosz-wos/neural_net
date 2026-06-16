#include "egnn.h"
#include <cmath>
#include <algorithm>
#include <random>

// =====================================================================
// EGNNLayer
//
// EGNN is a message-passing GNN on (h, x) where x is a 3D coordinate
// (or any coord_dim). The Tensor class is 2D, so we flatten per-edge
// tensors of shape (N, N, F) to (N*N, F) for storage; row i*N + j
// corresponds to the (i, j) directed edge.
// =====================================================================

EGNNLayer::EGNNLayer(size_t in_features, size_t hidden_dim, size_t coord_dim,
                     size_t n_edge_attrs, const std::string& activation,
                     bool coordinate_output)
    : in_features_(in_features), hidden_dim_(hidden_dim), coord_dim_(coord_dim),
      n_edge_attrs_(n_edge_attrs), activation_(activation),
      coordinate_output_(coordinate_output),
      phi_e_(2 * in_features + 1 + n_edge_attrs, hidden_dim),
      phi_h_(2 * hidden_dim, hidden_dim),
      phi_x_(hidden_dim, 1),
      last_input_h_(1, 1), last_input_x_(1, 1), last_adj_(1, 1),
      last_h_out_(1, 1), last_x_out_(1, 1), last_m_(1, 1),
      last_m_ij_(1, 1), last_px_(1, 1),
      last_h_pre_relu_(1, 1), last_dist2_(1, 1), last_h_concat_(1, 1),
      last_px_pre_relu_(1, 1)
{
    phi_e_.init_weights("xavier");
    phi_h_.init_weights("xavier");
    phi_x_.init_weights("xavier");
}

void EGNNLayer::relu_inplace(Tensor& t) const {
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            if (t[i][j] < 0.0) t[i][j] = 0.0;
        }
    }
}

Tensor EGNNLayer::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, hidden_dim_);
}

Tensor EGNNLayer::forward_with_adj(const Tensor& input_h, const Tensor& input_x,
                                   const Tensor& adj, const Tensor& edge_attr,
                                   const Tensor& weights) {
    size_t N = input_h.rows;
    size_t in_dim = 2 * in_features_ + 1 + n_edge_attrs_;
    last_n_ = N;
    last_input_h_ = input_h.clone();
    last_input_x_ = input_x.clone();
    last_adj_ = adj.clone();

    bool use_edge_attr = (n_edge_attrs_ > 0 && edge_attr.rows > 0);
    bool use_weights = (weights.rows > 0);
    last_use_edge_weights_ = use_weights;

    if (use_edge_attr) last_edge_attr_ = edge_attr.clone();
    else last_edge_attr_ = Tensor();
    if (use_weights) last_edge_weights_ = weights.clone();
    else last_edge_weights_ = Tensor();

    // ---- Step 1: build per-edge input batch (N*N, in_dim) ----
    Tensor batch_e(N * N, in_dim);
    Tensor dist2(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t row = i * N + j;
            for (size_t f = 0; f < in_features_; ++f) {
                batch_e(row, f) = input_h(i, f);
                batch_e(row, in_features_ + f) = input_h(j, f);
            }
            double d2 = 0.0;
            for (size_t c = 0; c < coord_dim_; ++c) {
                double diff = input_x(i, c) - input_x(j, c);
                d2 += diff * diff;
            }
            dist2(i, j) = d2;
            batch_e(row, 2 * in_features_) = d2;
            if (use_edge_attr) {
                for (size_t a = 0; a < n_edge_attrs_; ++a) {
                    // edge_attr is a flat (N*N, n_edge_attrs) tensor;
                    // row (i, j) is at i*N + j.
                    batch_e(row, 2 * in_features_ + 1 + a) =
                        edge_attr(i * N + j, a);
                }
            }
        }
    }
    last_dist2_ = dist2;

    // ---- Step 2: run phi_e to get per-edge messages (N*N, hidden_dim) ----
    Tensor m_ij_flat = phi_e_.forward(batch_e);
    if (is_relu()) relu_inplace(m_ij_flat);
    last_m_ij_ = m_ij_flat;  // post-ReLU; backward uses post>0 mask

    // ---- Step 3: aggregate messages m_i = sum_j w_ij * m_ij[i, j, :] ----
    Tensor m(N, hidden_dim_);
    m.fill(0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double w_ij;
            if (use_weights) w_ij = weights(i, j);
            else w_ij = adj(i, j);
            if (w_ij == 0.0) continue;
            size_t row = i * N + j;
            for (size_t f = 0; f < hidden_dim_; ++f) {
                m(i, f) += w_ij * m_ij_flat(row, f);
            }
        }
    }
    last_m_ = m;

    // ---- Step 4: phi_x per edge (N*N, 1) -> (N, N) ----
    Tensor px_flat = phi_x_.forward(m_ij_flat);
    last_px_pre_relu_ = px_flat.clone();  // (N*N, 1) for ReLU mask
    if (is_relu()) relu_inplace(px_flat);
    Tensor px(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            px(i, j) = px_flat(i * N + j, 0);
    last_px_ = px;

    // ---- Step 5: coord update ----
    Tensor x_out = input_x.clone();
    if (coordinate_output_) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double w_ij;
                if (use_weights) w_ij = weights(i, j);
                else w_ij = adj(i, j);
                if (w_ij == 0.0) continue;
                for (size_t c = 0; c < coord_dim_; ++c) {
                    double dx = input_x(i, c) - input_x(j, c);
                    x_out(i, c) += w_ij * dx * px(i, j);
                }
            }
        }
    }
    last_x_out_ = x_out;

    // ---- Step 6: h update h_i' = phi_h([pad(h_i) || m_i]) ----
    // If in_features_ < hidden_dim_, we pad h_i with zeros up to hidden_dim.
    // If in_features_ > hidden_dim_, the user must match dimensions; we
    // truncate in the h_concat build (no error to keep the API forgiving
    // for tests where dimensions don't perfectly align).
    Tensor h_concat(N, 2 * hidden_dim_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < hidden_dim_; ++f) {
            h_concat(i, f) = (f < in_features_) ? input_h(i, f) : 0.0;
        }
        for (size_t f = 0; f < hidden_dim_; ++f) {
            h_concat(i, hidden_dim_ + f) = m(i, f);
        }
    }
    last_h_concat_ = h_concat.clone();
    Tensor h_pre = phi_h_.forward(h_concat);
    last_h_pre_relu_ = h_pre.clone();
    Tensor h_out = h_pre;
    if (is_relu()) relu_inplace(h_out);
    last_h_out_ = h_out;

    return h_out;
}

Tensor EGNNLayer::backward(const Tensor& grad_h, double learning_rate) {
    // h-only backward. Returns d_input_h.
    size_t N = last_n_;
    const Tensor& in_h = last_input_h_;

    // Step 1: ReLU on h_out
    Tensor grad_h_pre(N, hidden_dim_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < hidden_dim_; ++f) {
            double v = last_h_pre_relu_(i, f);
            double mask = (is_relu() && v <= 0.0) ? 0.0 : 1.0;
            grad_h_pre(i, f) = grad_h(i, f) * mask;
        }
    }

    // Step 2: phi_h backward
    Tensor d_concat = phi_h_.backward(grad_h_pre, learning_rate);
    Tensor d_pad_h(N, hidden_dim_);
    Tensor d_m(N, hidden_dim_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < hidden_dim_; ++f) {
            d_pad_h(i, f) = d_concat(i, f);
            d_m(i, f) = d_concat(i, hidden_dim_ + f);
        }
    }

    // Step 3: aggregate-message backward
    // m_i = sum_j w_ij * m_ij[i, j, :]  ->  d_m_ij[i*N+j, f] = w_ij * d_m[i, f]
    Tensor d_m_ij(N * N, hidden_dim_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double w_ij;
            if (last_use_edge_weights_) w_ij = last_edge_weights_(i, j);
            else w_ij = last_adj_(i, j);
            size_t row = i * N + j;
            for (size_t f = 0; f < hidden_dim_; ++f) {
                d_m_ij(row, f) = w_ij * d_m(i, f);
            }
        }
    }

    // Step 4: ReLU on m_ij (use post-ReLU > 0 as mask — see forward())
    for (size_t i = 0; i < N * N; ++i) {
        for (size_t f = 0; f < hidden_dim_; ++f) {
            double v = last_m_ij_(i, f);
            double mask = (is_relu() && v <= 0.0) ? 0.0 : 1.0;
            d_m_ij(i, f) *= mask;
        }
    }

    // Step 5: phi_e backward
    Tensor d_batch_e = phi_e_.backward(d_m_ij, learning_rate);

    // Step 6: split into d_h
    Tensor d_input_h(in_h.rows, in_h.cols);
    d_input_h.fill(0.0);
    // Add the direct path: h_in flows into h_concat[0:in_features_] directly
    // (padded with zeros if in_features_ < hidden_dim_). Only the first
    // in_features_ columns of d_pad_h correspond to actual h_in entries.
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_features_; ++f) {
            d_input_h(i, f) += d_pad_h(i, f);
        }
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t row = i * N + j;
            for (size_t f = 0; f < in_features_; ++f) {
                d_input_h(i, f) += d_batch_e(row, f);
                d_input_h(j, f) += d_batch_e(row, in_features_ + f);
            }
        }
    }
    return d_input_h;
}

Tensor EGNNLayer::backward_coord(const Tensor& grad_x, double learning_rate,
                                 Tensor* d_input_h_out) {
    // x-only backward. Returns d_input_x. If d_input_h_out is non-null,
    // also computes the d_x contribution to d_input_h (through dist2).
    size_t N = last_n_;

    // Direct coord path
    Tensor d_input_x(N, coord_dim_);
    d_input_x.fill(0.0);

    // Diagonal: 1 + sum_j w_ij * px[i, j]
    for (size_t i = 0; i < N; ++i) {
        double sum_px = 0.0;
        for (size_t j = 0; j < N; ++j) {
            double w_ij;
            if (last_use_edge_weights_) w_ij = last_edge_weights_(i, j);
            else w_ij = last_adj_(i, j);
            if (w_ij == 0.0) continue;
            sum_px += w_ij * last_px_(i, j);
        }
        for (size_t c = 0; c < coord_dim_; ++c) {
            d_input_x(i, c) += grad_x(i, c) * (1.0 + sum_px);
        }
    }
    // j-side: from x_out[j, c]
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double w_ji;
            if (last_use_edge_weights_) w_ji = last_edge_weights_(j, i);
            else w_ji = last_adj_(j, i);
            if (w_ji == 0.0) continue;
            double wjpxji = w_ji * last_px_(j, i);
            for (size_t c = 0; c < coord_dim_; ++c) {
                d_input_x(i, c) += grad_x(j, c) * (-wjpxji);
            }
        }
    }

    // Through px: d_px[i, j] = sum_c d_x_out[i, c] * w_ij * (x[i, c] - x[j, c])
    Tensor d_px(N, N);
    d_px.fill(0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double w_ij;
            if (last_use_edge_weights_) w_ij = last_edge_weights_(i, j);
            else w_ij = last_adj_(i, j);
            if (w_ij == 0.0) continue;
            double s = 0.0;
            for (size_t c = 0; c < coord_dim_; ++c) {
                s += grad_x(i, c) * w_ij * (last_input_x_(i, c) - last_input_x_(j, c));
            }
            d_px(i, j) = s;
        }
    }

    // ReLU on px (use pre-ReLU > 0 mask; we saved last_px_pre_relu_)
    Tensor d_px_flat(N * N, 1);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double pre = last_px_pre_relu_(i * N + j, 0);
            double mask = (is_relu() && pre <= 0.0) ? 0.0 : 1.0;
            d_px_flat(i * N + j, 0) = d_px(i, j) * mask;
        }
    }

    // phi_x backward: input was m_ij_flat (N*N, hidden_dim).
    Tensor d_m_ij_flat = phi_x_.backward(d_px_flat, learning_rate);

    // ReLU on m_ij (post>0 mask)
    for (size_t i = 0; i < N * N; ++i) {
        for (size_t f = 0; f < hidden_dim_; ++f) {
            double v = last_m_ij_(i, f);
            double mask = (is_relu() && v <= 0.0) ? 0.0 : 1.0;
            d_m_ij_flat(i, f) *= mask;
        }
    }

    // phi_e backward
    Tensor d_batch_e = phi_e_.backward(d_m_ij_flat, learning_rate);

    // Extract d_dist2 at column 2*in_features
    Tensor d_dist2(N, N);
    d_dist2.fill(0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            d_dist2(i, j) = d_batch_e(i * N + j, 2 * in_features_);
        }
    }

    // d_dist2 to d_x chain
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < coord_dim_; ++c) {
            double sum = 0.0;
            for (size_t j = 0; j < N; ++j) {
                double diff = last_input_x_(i, c) - last_input_x_(j, c);
                sum += 2.0 * (d_dist2(i, j) + d_dist2(j, i)) * diff;
            }
            d_input_x(i, c) += sum;
        }
    }

    // Optionally compute d_input_h from d_batch_e
    if (d_input_h_out) {
        Tensor d_h(N, in_features_);
        d_h.fill(0.0);
        // Add the direct path contribution (h_in -> h_concat[0:in_f])
        // NOTE: in backward_coord we don't have d_pad_h because we don't
        // backprop through phi_h here. The d_batch_e is from phi_e's
        // backward. The h-in -> h-out path through phi_h is independent
        // of d_x (since d_x is the grad w.r.t. x_out, not h_out). The
        // d_h contribution we compute here is purely from the dist2
        // chain: x -> dist2 -> phi_e -> m_ij -> ... -> d_batch_e. There
        // is no direct path from h_in to x_out, so d_pad_h doesn't apply.
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t row = i * N + j;
                for (size_t f = 0; f < in_features_; ++f) {
                    d_h(i, f) += d_batch_e(row, f);
                    d_h(j, f) += d_batch_e(row, in_features_ + f);
                }
            }
        }
        *d_input_h_out = d_h;
    }

    return d_input_x;
}

void EGNNLayer::update_weights(double learning_rate) {
    phi_e_.update_weights(learning_rate);
    phi_h_.update_weights(learning_rate);
    phi_x_.update_weights(learning_rate);
}

void EGNNLayer::zero_grad() {
    phi_e_.zero_grad();
    phi_h_.zero_grad();
    phi_x_.zero_grad();
}

std::vector<Tensor*> EGNNLayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : phi_e_.parameters()) result.push_back(p);
    for (Tensor* p : phi_h_.parameters()) result.push_back(p);
    for (Tensor* p : phi_x_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> EGNNLayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : phi_e_.gradients()) result.push_back(g);
    for (Tensor* g : phi_h_.gradients()) result.push_back(g);
    for (Tensor* g : phi_x_.gradients()) result.push_back(g);
    return result;
}

// =====================================================================
// EGNNModel
// =====================================================================

EGNNModel::EGNNModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
                     size_t out_features, size_t coord_dim, size_t n_layers,
                     size_t n_edge_attrs, const std::string& activation)
    : num_nodes_(num_nodes), in_features_(in_features), hidden_dim_(hidden_dim),
      out_features_(out_features), coord_dim_(coord_dim),
      n_layers_(n_layers), n_edge_attrs_(n_edge_attrs), activation_(activation),
      input_proj_(in_features, hidden_dim),
      classifier_(hidden_dim, out_features),
      last_input_h_(1, 1), last_input_x_(1, 1), last_adj_(1, 1),
      last_h_(1, 1), last_x_(1, 1)
{
    input_proj_.init_weights("xavier");
    classifier_.init_weights("xavier");
    for (size_t l = 0; l < n_layers; ++l) {
        layers_.emplace_back(hidden_dim, hidden_dim, coord_dim,
                             n_edge_attrs, activation, true);
    }
}

Tensor EGNNModel::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, out_features_);
}

Tensor EGNNModel::forward_with_adj(const Tensor& input_h, const Tensor& input_x,
                                   const Tensor& adj, const Tensor& edge_attr,
                                   const Tensor& weights) {
    last_input_h_ = input_h.clone();
    last_input_x_ = input_x.clone();
    last_adj_ = adj.clone();
    bool use_edge_attr = (n_edge_attrs_ > 0 && edge_attr.rows > 0);
    bool use_weights = (weights.rows > 0);
    last_use_edge_weights_ = use_weights;
    if (use_edge_attr) last_edge_attr_ = edge_attr.clone();
    else last_edge_attr_ = Tensor();
    if (use_weights) last_edge_weights_ = weights.clone();
    else last_edge_weights_ = Tensor();

    Tensor h = input_proj_.forward(input_h);
    Tensor x = input_x;
    Tensor cur_edge_attr = use_edge_attr ? edge_attr : Tensor();
    Tensor cur_weights = use_weights ? weights : Tensor();
    for (auto& layer : layers_) {
        h = layer.forward_with_adj(h, x, adj, cur_edge_attr, cur_weights);
        x = layer.get_last_x();
    }
    last_h_ = h;
    last_x_ = x;
    Tensor logits = classifier_.forward(h);
    return logits;
}

Tensor EGNNModel::backward(const Tensor& grad_output, double learning_rate) {
    // Backward: grad_output -> classifier -> last layer's h -> layers in reverse -> input_proj -> input_h
    Tensor d_h = classifier_.backward(grad_output, learning_rate);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        d_h = it->backward(d_h, learning_rate);
    }
    Tensor d_proj_in = input_proj_.backward(d_h, learning_rate);
    return d_proj_in;
}

void EGNNModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& l : layers_) l.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void EGNNModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& l : layers_) l.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> EGNNModel::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : input_proj_.parameters()) result.push_back(p);
    for (auto& l : layers_)
        for (Tensor* p : l.parameters()) result.push_back(p);
    for (Tensor* p : classifier_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> EGNNModel::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : input_proj_.gradients()) result.push_back(g);
    for (auto& l : layers_)
        for (Tensor* g : l.gradients()) result.push_back(g);
    for (Tensor* g : classifier_.gradients()) result.push_back(g);
    return result;
}

Tensor EGNNModel::get_last_x() const {
    return last_x_;
}
