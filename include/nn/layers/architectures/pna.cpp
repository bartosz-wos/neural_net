#include "pna.h"
#include <limits>
#include <cassert>
#include <stdexcept>

// =====================================================================
// PNALayer
// =====================================================================

PNALayer::PNALayer(size_t in_features, size_t out_features, double deg_log_delta)
    : in_features_(in_features),
      out_features_(out_features),
      deg_log_delta_(deg_log_delta),
      post_agg_(in_features * num_aggregators_ * num_scalers_, out_features),
      last_input_(1, 1),
      adj_(1, 1),
      last_concat_(1, 1),
      last_deg_log_(1, 1),
      num_nodes_(0) {
    post_agg_.init_weights("xavier");
    last_agg_values_.resize(num_aggregators_);
}

Tensor PNALayer::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, 1);
}

std::vector<size_t> PNALayer::compute_degrees(const Tensor& adj) const {
    size_t N = adj.rows;
    std::vector<size_t> deg(N, 0);
    for (size_t i = 0; i < N; ++i) {
        size_t s = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 1e-9) s++;
        }
        deg[i] = s;
    }
    return deg;
}

Tensor PNALayer::agg_mean(const Tensor& input, const Tensor& adj) const {
    size_t N = input.rows;
    size_t F = input.cols;
    Tensor out = Tensor::zeros(N, F);
    for (size_t i = 0; i < N; ++i) {
        size_t cnt = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 1e-9) {
                for (size_t f = 0; f < F; ++f) {
                    out(i, f) += input(j, f);
                }
                cnt++;
            }
        }
        if (cnt > 0) {
            double inv = 1.0 / static_cast<double>(cnt);
            for (size_t f = 0; f < F; ++f) out(i, f) *= inv;
        }
    }
    return out;
}

Tensor PNALayer::agg_max(const Tensor& input, const Tensor& adj,
                          std::vector<std::vector<size_t>>& argmax) const {
    size_t N = input.rows;
    size_t F = input.cols;
    Tensor out(N, F);
    argmax.assign(N, std::vector<size_t>(F, 0));
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < F; ++f) {
            double best = -std::numeric_limits<double>::infinity();
            size_t best_j = 0;
            for (size_t j = 0; j < N; ++j) {
                if (adj(i, j) > 1e-9) {
                    double v = input(j, f);
                    if (v > best) { best = v; best_j = j; }
                }
            }
            // If a node has no neighbours, use 0.0 and self as placeholder
            if (!std::isfinite(best)) { best = 0.0; best_j = 0; }
            out(i, f) = best;
            argmax[i][f] = best_j;
        }
    }
    return out;
}

Tensor PNALayer::agg_min(const Tensor& input, const Tensor& adj,
                          std::vector<std::vector<size_t>>& argmin) const {
    size_t N = input.rows;
    size_t F = input.cols;
    Tensor out(N, F);
    argmin.assign(N, std::vector<size_t>(F, 0));
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < F; ++f) {
            double best = std::numeric_limits<double>::infinity();
            size_t best_j = 0;
            for (size_t j = 0; j < N; ++j) {
                if (adj(i, j) > 1e-9) {
                    double v = input(j, f);
                    if (v < best) { best = v; best_j = j; }
                }
            }
            if (!std::isfinite(best)) { best = 0.0; best_j = 0; }
            out(i, f) = best;
            argmin[i][f] = best_j;
        }
    }
    return out;
}

Tensor PNALayer::agg_std(const Tensor& input, const Tensor& adj,
                          const Tensor& mean_agg) const {
    // std = sqrt(E[(x - mean)^2]) over neighbours
    size_t N = input.rows;
    size_t F = input.cols;
    Tensor out = Tensor::zeros(N, F);
    for (size_t i = 0; i < N; ++i) {
        size_t cnt = 0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 1e-9) {
                for (size_t f = 0; f < F; ++f) {
                    double d = input(j, f) - mean_agg(i, f);
                    out(i, f) += d * d;
                }
                cnt++;
            }
        }
        if (cnt > 1) {
            double inv = 1.0 / static_cast<double>(cnt);
            for (size_t f = 0; f < F; ++f) {
                out(i, f) = std::sqrt(out(i, f) * inv);
            }
        } else if (cnt == 1) {
            for (size_t f = 0; f < F; ++f) out(i, f) = 0.0;
        }
    }
    return out;
}

Tensor PNALayer::apply_scaler_identity(const Tensor& agg) const {
    return agg;
}

Tensor PNALayer::apply_scaler_amplification(const Tensor& agg,
                                              const std::vector<double>& deg_log) const {
    size_t N = agg.rows;
    size_t F = agg.cols;
    Tensor out(N, F);
    for (size_t i = 0; i < N; ++i) {
        double s = deg_log[i] / deg_log_delta_;
        for (size_t f = 0; f < F; ++f) out(i, f) = s * agg(i, f);
    }
    return out;
}

Tensor PNALayer::apply_scaler_attenuation(const Tensor& agg,
                                            const std::vector<double>& deg_log) const {
    size_t N = agg.rows;
    size_t F = agg.cols;
    Tensor out(N, F);
    for (size_t i = 0; i < N; ++i) {
        // Guard against deg_log == 0 (no neighbours): fall back to 0
        // contribution from this scaler to keep things finite.
        double s = (deg_log[i] > 1e-12) ? (deg_log_delta_ / deg_log[i]) : 0.0;
        for (size_t f = 0; f < F; ++f) out(i, f) = s * agg(i, f);
    }
    return out;
}

Tensor PNALayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    size_t F = input.cols;
    if (F != in_features_) {
        throw std::invalid_argument("PNALayer::forward_with_adj: input feature dim mismatch");
    }
    num_nodes_ = N;
    last_input_ = input.clone();
    adj_ = adj;

    // Compute degree log
    std::vector<size_t> deg = compute_degrees(adj);
    std::vector<double> deg_log(N);
    for (size_t i = 0; i < N; ++i) deg_log[i] = std::log(static_cast<double>(deg[i]) + 1.0);
    // Cache as a (N, 1) tensor for interface uniformity
    last_deg_log_ = Tensor(N, 1);
    for (size_t i = 0; i < N; ++i) last_deg_log_(i, 0) = deg_log[i];

    // Compute aggregators
    last_agg_values_[0] = agg_mean(input, adj);                                                 // mean
    last_agg_values_[1] = agg_max(input, adj, last_argmax_);                                    // max
    last_agg_values_[2] = agg_min(input, adj, last_argmin_);                                    // min
    last_agg_values_[3] = agg_std(input, adj, last_agg_values_[0]);                              // std

    // Apply scalers and concatenate
    // Layout: [agg_0_s0, agg_0_s1, agg_0_s2, agg_1_s0, ..., agg_3_s2] along columns.
    // Output column dim: F * 4 * 3 = F * 12
    size_t agg_dim = F * num_aggregators_ * num_scalers_;
    Tensor concat(N, agg_dim);
    size_t offset = 0;
    for (size_t a = 0; a < num_aggregators_; ++a) {
        Tensor s0 = apply_scaler_identity(last_agg_values_[a]);
        Tensor s1 = apply_scaler_amplification(last_agg_values_[a], deg_log);
        Tensor s2 = apply_scaler_attenuation(last_agg_values_[a], deg_log);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < F; ++f) {
                concat(i, offset + 0 * F + f) = s0(i, f);
                concat(i, offset + 1 * F + f) = s1(i, f);
                concat(i, offset + 2 * F + f) = s2(i, f);
            }
        }
        offset += F * num_scalers_;
    }
    last_concat_ = concat;

    // Post-aggregation dense
    return post_agg_.forward(concat);
}

Tensor PNALayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features)
    // 1. Backprop through post-agg dense
    // 2. Reconstruct grad_concat = grad_output * W_post^T  (handled by Dense::backward)
    Tensor grad_concat = post_agg_.backward(grad_output, learning_rate);
    size_t N = grad_concat.rows;
    size_t concat_dim = grad_concat.cols;
    size_t F = in_features_;
    if (concat_dim != F * num_aggregators_ * num_scalers_) {
        throw std::invalid_argument("PNALayer::backward: grad_concat dim mismatch");
    }

    // 3. For each (aggregator, scaler) compute the gradient w.r.t. the
    //    aggregator output, then backprop through the aggregator to
    //    grad_input. Aggregator grads for std require a chain through
    //    mean as well; we account for that.
    Tensor grad_input = Tensor::zeros(N, F);
    std::vector<double> deg_log(N);
    for (size_t i = 0; i < N; ++i) deg_log[i] = last_deg_log_(i, 0);

    size_t offset = 0;
    for (size_t a = 0; a < num_aggregators_; ++a) {
        // Pull grad of (agg, scaler) outputs for this aggregator
        Tensor grad_s0 = Tensor(N, F);
        Tensor grad_s1 = Tensor(N, F);
        Tensor grad_s2 = Tensor(N, F);
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < F; ++f) {
                grad_s0(i, f) = grad_concat(i, offset + 0 * F + f);
                grad_s1(i, f) = grad_concat(i, offset + 1 * F + f);
                grad_s2(i, f) = grad_concat(i, offset + 2 * F + f);
            }
        }
        // Backprop scalers: each scaler is a scalar multiplication
        // identity:     ds0 = grad_s0
        // amplification: ds1 = (deg_log / delta) * grad_s1
        // attenuation:   ds2 = (delta / deg_log) * grad_s2   (or 0 if deg_log==0)
        Tensor grad_agg = Tensor::zeros(N, F);
        for (size_t i = 0; i < N; ++i) {
            double a_amp = deg_log[i] / deg_log_delta_;
            double a_att = (deg_log[i] > 1e-12) ? (deg_log_delta_ / deg_log[i]) : 0.0;
            for (size_t f = 0; f < F; ++f) {
                grad_agg(i, f) = grad_s0(i, f) + a_amp * grad_s1(i, f) + a_att * grad_s2(i, f);
            }
        }

        // Backprop each aggregator to grad_input
        if (a == 0) {
            // mean: d/dx_j of (1/cnt) * sum_l x_l = (1/cnt) per neighbour.
            // grad_input[j][f] += sum_{i: j in N(i)} (1/cnt_i) * grad_agg[i][f]
            for (size_t i = 0; i < N; ++i) {
                size_t cnt = 0;
                for (size_t j = 0; j < N; ++j) if (adj_(i, j) > 1e-9) cnt++;
                if (cnt == 0) continue;
                double inv = 1.0 / static_cast<double>(cnt);
                for (size_t j = 0; j < N; ++j) {
                    if (adj_(i, j) > 1e-9) {
                        for (size_t f = 0; f < F; ++f) {
                            grad_input(j, f) += inv * grad_agg(i, f);
                        }
                    }
                }
            }
        } else if (a == 1) {
            // max: gradient goes only to argmax neighbour
            for (size_t i = 0; i < N; ++i) {
                for (size_t f = 0; f < F; ++f) {
                    size_t j = last_argmax_[i][f];
                    grad_input(j, f) += grad_agg(i, f);
                }
            }
        } else if (a == 2) {
            // min: gradient goes only to argmin neighbour
            for (size_t i = 0; i < N; ++i) {
                for (size_t f = 0; f < F; ++f) {
                    size_t j = last_argmin_[i][f];
                    grad_input(j, f) += grad_agg(i, f);
                }
            }
        } else {
            // std: agg_std(input, mean_agg) = sqrt( (1/cnt) * sum_j (input_j - mean)^2 )
            // d agg_i / d input_j = (1/cnt) * (input_j - mean_i) / agg_i    if j in N(i)
            // Chain through mean: d agg_i / d input_j (via mean) = -(1/cnt) * d agg_i / d mean_i
            //                      d mean_i / d input_j = (1/cnt)
            // Combined:
            //   if j in N(i):
            //     d agg_i / d input_j = (input_j - mean_i) / (cnt * agg_i)  -  (1/cnt) * (1/cnt) * sum_l (input_l - mean_i) / agg_i
            //                         = (input_j - mean_i) / (cnt * agg_i)  -  0   (since mean is the centroid)
            // We add the contribution; the mean-grad part averages to 0.
            const Tensor& mean_agg = last_agg_values_[0];
            const Tensor& std_agg = last_agg_values_[3];
            for (size_t i = 0; i < N; ++i) {
                size_t cnt = 0;
                for (size_t j = 0; j < N; ++j) if (adj_(i, j) > 1e-9) cnt++;
                if (cnt == 0) continue;
                double inv = 1.0 / static_cast<double>(cnt);
                for (size_t j = 0; j < N; ++j) {
                    if (adj_(i, j) > 1e-9) {
                        for (size_t f = 0; f < F; ++f) {
                            double std_val = std_agg(i, f);
                            if (std_val < 1e-12) continue;
                            double deriv = (last_input_(j, f) - mean_agg(i, f)) * inv / std_val;
                            grad_input(j, f) += deriv * grad_agg(i, f);
                        }
                    }
                }
            }
        }
        offset += F * num_scalers_;
    }

    (void)learning_rate;  // not used in this layer
    return grad_input;
}

void PNALayer::update_weights(double learning_rate) {
    post_agg_.update_weights(learning_rate);
}

void PNALayer::zero_grad() {
    post_agg_.zero_grad();
}

std::vector<Tensor*> PNALayer::parameters() {
    std::vector<Tensor*> p = post_agg_.parameters();
    return p;
}

std::vector<Tensor*> PNALayer::gradients() {
    std::vector<Tensor*> g = post_agg_.gradients();
    return g;
}

// =====================================================================
// PNAModel
// =====================================================================

PNAModel::PNAModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
                   size_t out_features, size_t num_layers, double deg_log_delta)
    : num_nodes_(num_nodes),
      in_features_(in_features),
      hidden_dim_(hidden_dim),
      out_features_(out_features),
      num_layers_(num_layers),
      input_proj_(in_features, hidden_dim),
      classifier_(hidden_dim, out_features),
      last_input_(1, 1),
      last_input_proj_(1, 1),
      adj_(1, 1) {
    input_proj_.init_weights("xavier");
    classifier_.init_weights("xavier");
    pna_layers_.reserve(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
        pna_layers_.emplace_back(hidden_dim, hidden_dim, deg_log_delta);
    }
    layer_inputs_.resize(num_layers);
}

Tensor PNAModel::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, 1);
}

Tensor PNAModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = input.clone();
    adj_ = adj;
    // Project input
    Tensor x = input_proj_.forward(input);
    last_input_proj_ = x.clone();
    // Stack PNA layers with ReLU between. We cache the pre-ReLU PNA output
    // of each layer; that is the mask used by the corresponding ReLU
    // backward step.
    for (size_t l = 0; l < num_layers_; ++l) {
        x = pna_layers_[l].forward_with_adj(x, adj);
        Tensor pre_relu = x.clone();
        // In-place ReLU
        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                if (x(i, j) < 0.0) x(i, j) = 0.0;
            }
        }
        layer_inputs_[l] = pre_relu;
    }
    return classifier_.forward(x);
}

Tensor PNAModel::backward(const Tensor& grad_output, double learning_rate) {
    // 1. Backprop classifier
    Tensor grad = classifier_.backward(grad_output, learning_rate);
    // 2. Backprop ReLU + PNA layers in reverse
    for (size_t l = num_layers_; l > 0; --l) {
        // Backprop through ReLU
        Tensor& in_relu = layer_inputs_[l - 1];
        for (size_t i = 0; i < grad.rows; ++i) {
            for (size_t j = 0; j < grad.cols; ++j) {
                if (in_relu(i, j) <= 0.0) grad(i, j) = 0.0;
            }
        }
        grad = pna_layers_[l - 1].backward(grad, learning_rate);
    }
    // 3. Backprop input projection
    grad = input_proj_.backward(grad, learning_rate);
    return grad;
}

void PNAModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (size_t l = 0; l < num_layers_; ++l) {
        pna_layers_[l].update_weights(learning_rate);
    }
    classifier_.update_weights(learning_rate);
}

void PNAModel::zero_grad() {
    input_proj_.zero_grad();
    for (size_t l = 0; l < num_layers_; ++l) {
        pna_layers_[l].zero_grad();
    }
    classifier_.zero_grad();
}

std::vector<Tensor*> PNAModel::parameters() {
    std::vector<Tensor*> p;
    auto ip = input_proj_.parameters();
    auto cp = classifier_.parameters();
    p.insert(p.end(), ip.begin(), ip.end());
    for (size_t l = 0; l < num_layers_; ++l) {
        auto pp = pna_layers_[l].parameters();
        p.insert(p.end(), pp.begin(), pp.end());
    }
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> PNAModel::gradients() {
    std::vector<Tensor*> g;
    auto ig = input_proj_.gradients();
    auto cg = classifier_.gradients();
    g.insert(g.end(), ig.begin(), ig.end());
    for (size_t l = 0; l < num_layers_; ++l) {
        auto pg = pna_layers_[l].gradients();
        g.insert(g.end(), pg.begin(), pg.end());
    }
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
