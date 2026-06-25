#include "graphsage.h"
#include <cassert>
#include <stdexcept>
#include <random>
#include <iostream>
#include <unordered_set>

// =====================================================================
// GraphSAGELayer
// =====================================================================

GraphSAGELayer::GraphSAGELayer(size_t in_features, size_t out_features,
                               const std::string& aggregator, bool normalize,
                               bool self_loop)
    : in_features_(in_features),
      out_features_(out_features),
      aggregator_(aggregator),
      normalize_(normalize),
      self_loop_(self_loop),
      // W: CONCAT(h_i, h_{N(i)}) (size 2 * in_features) -> out_features
      W_(2 * in_features, out_features) {
    if (aggregator_ != "mean" && aggregator_ != "pool" && aggregator_ != "max") {
        throw std::invalid_argument("GraphSAGELayer: aggregator must be one of {mean, pool, max}");
    }
    W_.init_weights("xavier");
    if (aggregator_ == "pool") {
        // W_pool: per-neighbour transform before max-aggregation
        W_pool_ = std::make_unique<Dense>(in_features, in_features);
        W_pool_->init_weights("xavier");
        has_pool_ = true;
    }
}

// ---------------------------------------------------------------------
// Sampling helper: returns the neighbour index list for node i.
// If self_loop_ is true, the node itself is appended at the END of the
// list (so it can be excluded from aggregation but kept for the CONCAT).
// ---------------------------------------------------------------------
static std::vector<size_t>
sample_neighbors(size_t i, const Tensor& adj, size_t num_samples) {
    size_t N = adj.rows;
    std::vector<size_t> all;
    all.reserve(N);
    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;
        if (adj(i, j) > 1e-9) all.push_back(j);
    }
    if (num_samples == 0 || num_samples >= all.size()) {
        return all;  // no sampling
    }
    // Uniform random sample without replacement.
    // Use std::mt19937 seeded by deterministic per-(i, num_samples) hash for
    // reproducibility — we still get randomness across nodes.
    std::seed_seq sseq{static_cast<uint32_t>(i), static_cast<uint32_t>(num_samples),
                        static_cast<uint32_t>(adj.rows)};
    std::mt19937 rng(sseq);
    std::vector<size_t> out;
    out.reserve(num_samples);
    // Partial Fisher–Yates: swap-and-take-first-K.
    for (size_t k = 0; k < num_samples; ++k) {
        std::uniform_int_distribution<size_t> dist(k, all.size() - 1);
        size_t idx = dist(rng);
        std::swap(all[k], all[idx]);
        out.push_back(all[k]);
    }
    return out;
}

Tensor GraphSAGELayer::forward(const Tensor& input) {
    // No adjacency provided — treat as fully connected (every node is every
    // other node's neighbour, except self unless self_loop_).
    size_t N = input.rows;
    Tensor adj(N, N);
    adj.fill(1.0);
    for (size_t i = 0; i < N; ++i) {
        if (!self_loop_) adj(i, i) = 0.0;
    }
    return forward_with_adj(input, adj);
}

Tensor GraphSAGELayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    if (adj.rows != N || adj.cols != N) {
        throw std::invalid_argument("GraphSAGELayer: adjacency must be square N x N");
    }
    if (input.cols != in_features_) {
        throw std::invalid_argument("GraphSAGELayer: input.cols != in_features_");
    }

    // Cache state for backward.
    last_input_ = input.clone();
    adj_ = adj.clone();
    last_neighbors_.assign(N, std::vector<size_t>());

    // -------- Step 1: aggregate neighbours --------
    Tensor agg(N, in_features_);
    if (aggregator_ == "mean") {
        for (size_t i = 0; i < N; ++i) {
            std::vector<size_t> nbrs = sample_neighbors(i, adj, num_samples_);
            last_neighbors_[i] = nbrs;
            if (nbrs.empty()) continue;
            double inv = 1.0 / static_cast<double>(nbrs.size());
            for (size_t f = 0; f < in_features_; ++f) agg(i, f) = 0.0;
            for (size_t j : nbrs) {
                for (size_t f = 0; f < in_features_; ++f) {
                    agg(i, f) += input(j, f);
                }
            }
            for (size_t f = 0; f < in_features_; ++f) agg(i, f) *= inv;
        }
    } else if (aggregator_ == "pool" || aggregator_ == "max") {
        // For pool/max we need to track argmax (which neighbour produced
        // the per-feature maximum) for the backward pass.
        last_argmax_.assign(N, std::vector<size_t>(in_features_, 0));

        // Compute per-neighbour "candidate" embedding for each neighbour:
        //   - "pool": sigma(W_pool · h_u + b_pool)
        //   - "max" : h_u (no transform)
        // Then take element-wise max over neighbours per (i, feature).
        for (size_t i = 0; i < N; ++i) {
            std::vector<size_t> nbrs = sample_neighbors(i, adj, num_samples_);
            last_neighbors_[i] = nbrs;
            if (nbrs.empty()) {
                for (size_t f = 0; f < in_features_; ++f) agg(i, f) = 0.0;
                continue;
            }
            for (size_t f = 0; f < in_features_; ++f) {
                double best = -std::numeric_limits<double>::infinity();
                size_t best_j = nbrs[0];
                for (size_t j : nbrs) {
                    double v = input(j, f);
                    if (aggregator_ == "pool") {
                        // Apply pool Dense row-wise to input.
                        double s = 0.0;
                        for (size_t k = 0; k < in_features_; ++k) {
                            s += W_pool_->weights(f, k) * input(j, k);
                        }
                        s += W_pool_->bias(0, f);
                        v = 1.0 / (1.0 + std::exp(-s));  // sigmoid
                    }
                    if (v > best) { best = v; best_j = j; }
                }
                agg(i, f) = best;
                last_argmax_[i][f] = best_j;
            }
        }
    }

    // -------- Step 2: CONCAT(h_i, agg) and dense --------
    Tensor concat(N, 2 * in_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_features_; ++f) {
            concat(i, f) = input(i, f);
            concat(i, in_features_ + f) = agg(i, f);
        }
    }
    last_concat_ = concat.clone();

    // Dense forward: y = x @ W^T + b
    last_pre_act_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < out_features_; ++k) {
            double s = W_.bias(0, k);
            for (size_t j = 0; j < 2 * in_features_; ++j) {
                s += concat(i, j) * W_.weights(k, j);
            }
            last_pre_act_(i, k) = s;
        }
    }

    // -------- Step 3: activation (ReLU) + optional L2 normalize --------
    Tensor act(N, out_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < out_features_; ++k) {
            double v = last_pre_act_(i, k);
            act(i, k) = v > 0.0 ? v : 0.0;
        }
    }
    last_output_ = act.clone();

    Tensor out(N, out_features_);
    if (normalize_) {
        // Per-row L2 normalization. eps = 1e-12 for stability.
        const double eps = 1e-12;
        for (size_t i = 0; i < N; ++i) {
            double sumsq = 0.0;
            for (size_t k = 0; k < out_features_; ++k) {
                double v = act(i, k);
                sumsq += v * v;
            }
            double norm = std::sqrt(sumsq + eps);
            for (size_t k = 0; k < out_features_; ++k) {
                out(i, k) = act(i, k) / norm;
            }
        }
        last_norm_out_ = out.clone();
    } else {
        last_norm_out_ = out.clone();
        out = act;
    }

    has_cache_ = true;
    return out;
}

Tensor GraphSAGELayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (!has_cache_) {
        throw std::runtime_error("GraphSAGELayer::backward called before forward_with_adj");
    }
    size_t N = last_input_.rows;
    const double eps = 1e-12;

    // -------- Step 3 backward: L2-normalize (if used) --------
    Tensor d_act(N, out_features_);
    if (normalize_) {
        // d/dx (x / ||x||_2) = (I - x x^T / ||x||^2) / ||x||
        for (size_t i = 0; i < N; ++i) {
            double sumsq = 0.0;
            for (size_t k = 0; k < out_features_; ++k) {
                double v = last_output_(i, k);
                sumsq += v * v;
            }
            double norm = std::sqrt(sumsq + eps);
            double inv_norm = 1.0 / norm;
            double inv_norm3 = inv_norm / (sumsq + eps);  // 1 / (||x||^3)
            for (size_t k = 0; k < out_features_; ++k) {
                double g = grad_output(i, k);
                double s = 0.0;
                for (size_t j = 0; j < out_features_; ++j) {
                    double kj = (k == j) ? 1.0 : 0.0;
                    s += g * (kj - last_output_(i, k) * last_output_(i, j) * inv_norm3);
                }
                d_act(i, k) = s * inv_norm;
            }
        }
    } else {
        d_act = grad_output.clone();
    }

    // -------- Step 3 backward: ReLU --------
    Tensor d_pre(N, out_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < out_features_; ++k) {
            double v = last_pre_act_(i, k);
            d_pre(i, k) = (v > 0.0) ? d_act(i, k) : 0.0;
        }
    }

    // -------- Step 2 backward: Dense (CONCAT(h_i, h_{N(i)})) --------
    // grad_W_: (out, 2*in) += sum_i d_pre(i, :) outer concat(i, :)
    // grad_b_: (1, out) += sum_i d_pre(i, :)
    // grad_concat: (N, 2*in) = d_pre @ W_  (since y = x W^T + b, dx = dy W)
    Tensor grad_concat(N, 2 * in_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 2 * in_features_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < out_features_; ++k) {
                s += d_pre(i, k) * W_.weights(k, j);
            }
            grad_concat(i, j) = s;
        }
    }
    // Update W_/b_ gradients in-place.
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 2 * in_features_; ++j) {
            double c = last_concat_(i, j);
            for (size_t k = 0; k < out_features_; ++k) {
                W_.grad_weights(k, j) += d_pre(i, k) * c;
            }
        }
        for (size_t k = 0; k < out_features_; ++k) {
            W_.grad_bias(0, k) += d_pre(i, k);
        }
    }

    // Split grad_concat into self and aggregate branches.
    Tensor d_self(N, in_features_);
    Tensor d_agg(N, in_features_);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_features_; ++f) {
            d_self(i, f) = grad_concat(i, f);
            d_agg(i, f)   = grad_concat(i, in_features_ + f);
        }
    }

    // -------- Step 1 backward: aggregator --------
    Tensor d_input(N, in_features_);
    for (size_t i = 0; i < N; ++i) d_input(i, 0) = 0.0;  // touched but unused
    d_input.fill(0.0);

    if (aggregator_ == "mean") {
        // d_input[j] gets += d_agg(i, :) / count for every i that has j as neighbour.
        for (size_t i = 0; i < N; ++i) {
            const auto& nbrs = last_neighbors_[i];
            if (nbrs.empty()) continue;
            double inv = 1.0 / static_cast<double>(nbrs.size());
            for (size_t j : nbrs) {
                for (size_t f = 0; f < in_features_; ++f) {
                    d_input(j, f) += d_agg(i, f) * inv;
                }
            }
        }
    } else if (aggregator_ == "max") {
        // Element-wise max: each feature f of agg(i,:) comes from
        // the neighbour last_argmax_[i][f]; route the gradient there.
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_features_; ++f) {
                size_t j = last_argmax_[i][f];
                d_input(j, f) += d_agg(i, f);
            }
        }
    } else if (aggregator_ == "pool") {
        // For each (i, f), agg(i, f) = sigmoid(W_pool[f,:] · h_{j*} + b_pool[f])
        // where j* = argmax neighbour. The gradient d_agg(i, f) flows through
        // sigmoid (which is computed during forward using cached pool-pre-sigmoid).
        // We need to recompute pool_pre = sigmoid(W_pool · input + b_pool) for
        // each (neighbour, feature) and pick the argmax per (i, f).
        // But we cached last_argmax_, so we re-evaluate sigma for the winning
        // neighbour to get the right sigmoid derivative value.
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_features_; ++f) {
                size_t j_star = last_argmax_[i][f];
                // Recompute pre-activation at (j_star, f) using current W_pool_->
                double pre = 0.0;
                for (size_t k = 0; k < in_features_; ++k) {
                    pre += W_pool_->weights(f, k) * last_input_(j_star, k);
                }
                pre += W_pool_->bias(0, f);
                double sig = 1.0 / (1.0 + std::exp(-pre));
                double d_sig = sig * (1.0 - sig);
                double upstream = d_agg(i, f) * d_sig;
                // d_W_pool_[f, k] += upstream * input(j_star, k)
                for (size_t k = 0; k < in_features_; ++k) {
                    W_pool_->grad_weights(f, k) += upstream * last_input_(j_star, k);
                }
                W_pool_->grad_bias(0, f) += upstream;
                // d_input[j_star, k] += upstream * W_pool_[f, k]
                for (size_t k = 0; k < in_features_; ++k) {
                    d_input(j_star, k) += upstream * W_pool_->weights(f, k);
                }
            }
        }
    }

    // -------- Final: add self contribution --------
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_features_; ++f) {
            d_input(i, f) += d_self(i, f);
        }
    }

    return d_input;
}

void GraphSAGELayer::update_weights(double learning_rate) {
    W_.update_weights(learning_rate);
    if (has_pool_) W_pool_->update_weights(learning_rate);
}

void GraphSAGELayer::zero_grad() {
    W_.zero_grad();
    if (has_pool_) W_pool_->zero_grad();
}

std::vector<Tensor*> GraphSAGELayer::parameters() {
    std::vector<Tensor*> out = W_.parameters();
    if (has_pool_) {
        auto pool_p = W_pool_->parameters();
        out.insert(out.end(), pool_p.begin(), pool_p.end());
    }
    return out;
}

std::vector<Tensor*> GraphSAGELayer::gradients() {
    std::vector<Tensor*> out = W_.gradients();
    if (has_pool_) {
        auto pool_g = W_pool_->gradients();
        out.insert(out.end(), pool_g.begin(), pool_g.end());
    }
    return out;
}

// =====================================================================
// GraphSAGEModel
// =====================================================================

GraphSAGEModel::GraphSAGEModel(size_t in_features, size_t hidden_dim,
                               size_t out_features, size_t num_layers,
                               const std::string& aggregator, bool normalize)
    : input_proj_(in_features, hidden_dim),
      classifier_(hidden_dim, out_features) {
    if (num_layers == 0) {
        throw std::invalid_argument("GraphSAGEModel: num_layers must be >= 1");
    }
    use_proj_ = (in_features != hidden_dim);
    if (use_proj_) input_proj_.init_weights("xavier");
    classifier_.init_weights("xavier");

    // Layer 0 takes input_features -> hidden_dim if not projected, else hidden_dim -> hidden_dim.
    // We always feed hidden_dim to subsequent layers for stability.
    layers_.push_back(new GraphSAGELayer(hidden_dim, hidden_dim, aggregator, normalize, true));
    for (size_t l = 1; l < num_layers; ++l) {
        layers_.push_back(new GraphSAGELayer(hidden_dim, hidden_dim, aggregator, normalize, true));
    }
}

GraphSAGEModel::~GraphSAGEModel() {
    for (auto* l : layers_) delete l;
}

Tensor GraphSAGEModel::forward(const Tensor& input) {
    size_t N = input.rows;
    Tensor adj(N, N);
    adj.fill(1.0);
    for (size_t i = 0; i < N; ++i) adj(i, i) = 0.0;
    return forward_with_adj(input, adj);
}

Tensor GraphSAGEModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    Tensor h = use_proj_ ? input_proj_.forward(input) : input;
    for (auto* l : layers_) {
        h = l->forward_with_adj(h, adj);
    }
    return classifier_.forward(h);
}

Tensor GraphSAGEModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor g = classifier_.backward(grad_output, 0.0);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        g = (*it)->backward(g, 0.0);
    }
    if (use_proj_) {
        g = input_proj_.backward(g, 0.0);
    }
    return g;
}

void GraphSAGEModel::update_weights(double learning_rate) {
    if (use_proj_) input_proj_.update_weights(learning_rate);
    for (auto* l : layers_) l->update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void GraphSAGEModel::zero_grad() {
    if (use_proj_) input_proj_.zero_grad();
    for (auto* l : layers_) l->zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> GraphSAGEModel::parameters() {
    std::vector<Tensor*> out;
    if (use_proj_) {
        auto p = input_proj_.parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    for (auto* l : layers_) {
        auto p = l->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    auto cp = classifier_.parameters();
    out.insert(out.end(), cp.begin(), cp.end());
    return out;
}

std::vector<Tensor*> GraphSAGEModel::gradients() {
    std::vector<Tensor*> out;
    if (use_proj_) {
        auto g = input_proj_.gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    for (auto* l : layers_) {
        auto g = l->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    auto cg = classifier_.gradients();
    out.insert(out.end(), cg.begin(), cg.end());
    return out;
}