#include "edgeconv.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

// =====================================================================
// EdgeConvLayer
// =====================================================================

EdgeConvLayer::EdgeConvLayer(size_t in_features, size_t out_features, size_t k,
                             bool self_loops)
    : in_features_(in_features), out_features_(out_features), k_(k),
      self_loops_(self_loops),
      // MLP: input is concat(h_i, h_j - h_i) of size 2*in_features.
      // Hidden dim: 2 * out_features (paper convention).
      mlp_fc1_(2 * in_features, 2 * out_features),
      mlp_fc2_(2 * out_features, out_features),
      last_input_(1, 1), num_nodes_(0)
{
    // Initialise MLP weights
    mlp_fc1_.init_weights("xavier");
    mlp_fc2_.init_weights("xavier");
}

Tensor EdgeConvLayer::forward(const Tensor& input) {
    // Default: no adjacency constraint. We pass a fully-connected
    // virtual adjacency so the k-NN search runs unconstrained.
    size_t N = input.rows;
    Tensor adj(N, N);
    adj.fill(1.0);
    for (size_t i = 0; i < N; ++i) adj(i, i) = 0.0;
    return forward_with_adj(input, adj);
}

Tensor EdgeConvLayer::forward_with_adj(const Tensor& input, const Tensor& /*adj*/) {
    // input: (num_nodes, in_features)
    size_t N = input.rows;
    num_nodes_ = N;
    last_input_ = input.clone();

    // ---- Step 1: build k-NN graph in feature space ----
    last_neighbors_.assign(N, std::vector<size_t>());
    last_argmax_.assign(N, std::vector<size_t>(out_features_, 0));

    // For each node i, compute squared distance to every node j (j != i
    // unless self_loops_ is true). Pick the k smallest distances.
    std::vector<double> dists(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j == i && !self_loops_) {
                dists[j] = std::numeric_limits<double>::infinity();
                continue;
            }
            double d2 = 0.0;
            for (size_t f = 0; f < in_features_; ++f) {
                double diff = input(i, f) - input(j, f);
                d2 += diff * diff;
            }
            dists[j] = d2;
        }
        // If self_loops_, include i in the candidate set with distance 0.
        if (self_loops_) dists[i] = 0.0;

        // Pick k smallest. Use a partial sort: collect (dist, idx) pairs,
        // sort ascending, take first k.
        std::vector<std::pair<double, size_t>> pairs;
        pairs.reserve(N);
        for (size_t j = 0; j < N; ++j) {
            pairs.emplace_back(dists[j], j);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const std::pair<double, size_t>& a,
                     const std::pair<double, size_t>& b) {
                      return a.first < b.first;
                  });
        size_t kk = std::min(k_, N);
        for (size_t s = 0; s < kk; ++s) {
            last_neighbors_[i].push_back(pairs[s].second);
        }
    }

    // ---- Step 2 & 3: for each (i, j in N(i)) build edge feature and
    //                   run through MLP ----
    //
    // The MLP takes concat(h_i, h_j - h_i) of size 2*in_features.
    // Output per (i,j) pair: (out_features,).
    // We collect all messages into a (sum_neighbors, 2*in_features) tensor
    // and run Dense once. We must remember the (i, j) indices to route
    // gradients correctly in backward.
    //
    // Total message count: sum_i |N(i)|. With self_loops_=true and k_=20
    // this is up to N*k_ messages. We allocate a flat buffer.
    size_t total_msgs = 0;
    for (size_t i = 0; i < N; ++i) total_msgs += last_neighbors_[i].size();
    if (total_msgs == 0) {
        // Degenerate: no neighbours. Return zeros of right shape.
        Tensor out = Tensor::zeros(N, out_features_);
        last_concat_msgs_ = Tensor(0, 2 * in_features_);
        return out;
    }

    Tensor concat_msgs(total_msgs, 2 * in_features_);
    // Map flat message index -> (i, j) for backward.
    std::vector<std::pair<size_t, size_t>> msg_index;  // (i, j_neighbour)
    msg_index.reserve(total_msgs);
    size_t row = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j : last_neighbors_[i]) {
            // concat: [h_i || h_j - h_i]
            for (size_t f = 0; f < in_features_; ++f) {
                concat_msgs(row, f) = input(i, f);
                concat_msgs(row, in_features_ + f) = input(j, f) - input(i, f);
            }
            msg_index.emplace_back(i, j);
            ++row;
        }
    }
    last_concat_msgs_ = concat_msgs.clone();

    // Run MLP: fc1 -> ReLU -> fc2
    // (ReLU is applied implicitly via the Dense's expectation that the
    // user applies a nonlinearity; we must apply it here manually so
    // Dense.backward of fc1 sees the correct pre-activation cache.)
    // We mimic the standard pattern: fc1.forward, then manually ReLU,
    // then fc2.forward. Dense caches its last_input. After ReLU we
    // need to override the fc1's cached last_input with the post-ReLU
    // value so that fc1.backward computes the correct gradient through
    // the ReLU.
    Tensor fc1_out = mlp_fc1_.forward(concat_msgs);
    // Apply ReLU in place
    for (size_t i = 0; i < fc1_out.rows; ++i) {
        for (size_t f = 0; f < fc1_out.cols; ++f) {
            if (fc1_out(i, f) < 0.0) fc1_out(i, f) = 0.0;
        }
    }
    // Override fc1's cached input with post-ReLU output so that
    // fc1.backward computes gradient w.r.t. post-ReLU, which is
    // equivalent to chain-rule through the ReLU.
    mlp_fc1_.last_input = fc1_out.clone();

    Tensor msgs = mlp_fc2_.forward(fc1_out);
    // msgs is (total_msgs, out_features_)

    // Precompute (i, s) -> flat row index in msgs.
    std::vector<std::vector<size_t>> lookup(N);
    {
        size_t row = 0;
        for (size_t i = 0; i < N; ++i) {
            lookup[i].assign(last_neighbors_[i].size(), 0);
            for (size_t s = 0; s < last_neighbors_[i].size(); ++s, ++row) {
                lookup[i][s] = row;
            }
        }
    }

    // ---- Step 4: max aggregation per (i, f) ----
    Tensor out = Tensor::zeros(N, out_features_);
    for (size_t i = 0; i < N; ++i) {
        const auto& nbrs = last_neighbors_[i];
        if (nbrs.empty()) {
            // No neighbours: out is zeros; argmax stays 0.
            continue;
        }
        for (size_t f = 0; f < out_features_; ++f) {
            double best = -std::numeric_limits<double>::infinity();
            size_t best_idx = 0;
            for (size_t s = 0; s < nbrs.size(); ++s) {
                double v = msgs(lookup[i][s], f);
                if (v > best) {
                    best = v;
                    best_idx = s;  // index INTO nbrs
                }
            }
            out(i, f) = best;
            last_argmax_[i][f] = best_idx;
        }
    }

    (void)msg_index;  // not used in the optimised path

    return out;
}

Tensor EdgeConvLayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features_)
    size_t N = num_nodes_;
    if (N == 0) return Tensor(grad_output.rows, grad_output.cols);
    const auto& nbrs_all = last_neighbors_;
    size_t total_msgs = 0;
    for (size_t i = 0; i < N; ++i) total_msgs += nbrs_all[i].size();
    if (total_msgs == 0) {
        return Tensor::zeros(N, in_features_);
    }

    // ---- 1. Backprop through max aggregator ----
    // For each (i, f): grad_output(i, f) flows to msgs[argmax(i,f)][f].
    // Build a (total_msgs, out_features_) tensor of incoming grads for msgs.
    Tensor grad_msgs = Tensor::zeros(total_msgs, out_features_);
    // We need a fast lookup from (i, j) -> flat row index in last_concat_msgs_.
    // Precompute it.
    std::vector<std::vector<int>> lookup(N);  // lookup[i][j_in_nbrs] = flat_row
    {
        size_t row = 0;
        for (size_t i = 0; i < N; ++i) {
            lookup[i].assign(nbrs_all[i].size(), -1);
            for (size_t s = 0; s < nbrs_all[i].size(); ++s, ++row) {
                lookup[i][s] = (int)row;
            }
        }
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < out_features_; ++f) {
            size_t s = last_argmax_[i][f];
            if (s >= nbrs_all[i].size()) continue;
            int row = lookup[i][s];
            if (row < 0) continue;
            grad_msgs((size_t)row, f) += grad_output(i, f);
        }
    }

    // ---- 2. Backprop through MLP (fc2 then ReLU then fc1) ----
    // fc2: msgs (post-ReLU-of-fc1) -> grad_msgs. Use Dense.backward.
    Tensor grad_fc1_out = mlp_fc2_.backward(grad_msgs, learning_rate);
    // grad_fc1_out is gradient w.r.t. the post-ReLU input of fc1.
    // Backprop through ReLU: grad = grad_fc1_out * (post-ReLU > 0).
    // mlp_fc1_.last_input is the post-ReLU value (we set it in forward).
    for (size_t i = 0; i < grad_fc1_out.rows; ++i) {
        for (size_t f = 0; f < grad_fc1_out.cols; ++f) {
            if (mlp_fc1_.last_input(i, f) <= 0.0) {
                grad_fc1_out(i, f) = 0.0;
            }
        }
    }
    // fc1 backward: this gives us grad_concat_msgs.
    // But: mlp_fc1_.last_input is post-ReLU, and we set it in forward
    // to override. The Dense backward computes grad w.r.t. its cached
    // last_input. We want grad w.r.t. the *pre-ReLU* fc1 output, which
    // is what grad_fc1_out is (after we masked ReLU). So set
    // mlp_fc1_.last_input back to the pre-ReLU output to compute
    // grad_concat_msgs correctly.
    // (We cached last_concat_msgs_ for inspection, but the pre-ReLU
    // fc1 output is not stored. Re-run fc1 forward on the same input
    // to get it.)
    Tensor fc1_pre_relu = mlp_fc1_.forward(last_concat_msgs_);
    mlp_fc1_.last_input = fc1_pre_relu.clone();  // restore pre-ReLU

    Tensor grad_concat = mlp_fc1_.backward(grad_fc1_out, learning_rate);
    // grad_concat: (total_msgs, 2*in_features_)

    // ---- 3. Backprop through edge function and gather ----
    // For each message (i, j): concat = [h_i || h_j - h_i].
    //   d(h_i, f)     += grad_concat[row, f]
    //   d(h_j, f)     += grad_concat[row, in_features_ + f]
    //   d(h_i, in+f)  -= grad_concat[row, in_features_ + f]   (from -h_i term)
    // Combined contribution to h_i from this message:
    //   grad_x[i, f] += grad_concat[row, f] - grad_concat[row, in_features_ + f]
    // Contribution to h_j from this message:
    //   grad_x[j, f] += grad_concat[row, in_features_ + f]
    Tensor grad_x = Tensor::zeros(N, in_features_);
    {
        size_t row = 0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t s = 0; s < nbrs_all[i].size(); ++s, ++row) {
                size_t j = nbrs_all[i][s];
                for (size_t f = 0; f < in_features_; ++f) {
                    double g_self = grad_concat(row, f);
                    double g_edge = grad_concat(row, in_features_ + f);
                    grad_x(i, f) += g_self - g_edge;
                    grad_x(j, f) += g_edge;
                }
            }
        }
    }

    return grad_x;
}

void EdgeConvLayer::update_weights(double learning_rate) {
    mlp_fc1_.update_weights(learning_rate);
    mlp_fc2_.update_weights(learning_rate);
}

void EdgeConvLayer::zero_grad() {
    mlp_fc1_.zero_grad();
    mlp_fc2_.zero_grad();
}

std::vector<Tensor*> EdgeConvLayer::parameters() {
    auto p1 = mlp_fc1_.parameters();
    auto p2 = mlp_fc2_.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

std::vector<Tensor*> EdgeConvLayer::gradients() {
    auto g1 = mlp_fc1_.gradients();
    auto g2 = mlp_fc2_.gradients();
    g1.insert(g1.end(), g2.begin(), g2.end());
    return g1;
}

// =====================================================================
// EdgeConvModel
// =====================================================================

EdgeConvModel::EdgeConvModel(size_t in_features, size_t hidden_dim,
                             size_t out_features, size_t num_classes,
                             size_t num_layers, size_t k)
    : in_features_(in_features), hidden_dim_(hidden_dim),
      out_features_(out_features), num_classes_(num_classes),
      num_layers_(num_layers),
      input_proj_(in_features, hidden_dim),
      head_proj_(hidden_dim, out_features),
      classifier_(out_features, num_classes),
      last_input_(1, 1)
{
    // Stack of EdgeConv layers, all with hidden_dim
    for (size_t i = 0; i < num_layers_; ++i) {
        edge_layers_.emplace_back(hidden_dim, hidden_dim, k, /*self_loops=*/true);
    }
    input_proj_.init_weights("xavier");
    head_proj_.init_weights("xavier");
    if (num_classes_ > 0) classifier_.init_weights("xavier");
}

Tensor EdgeConvModel::forward(const Tensor& input) {
    size_t N = input.rows;
    Tensor adj(N, N);
    adj.fill(1.0);
    for (size_t i = 0; i < N; ++i) adj(i, i) = 0.0;
    return forward_with_adj(input, adj);
}

Tensor EdgeConvModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = input.clone();
    adj_ = adj;
    layer_pre_relu_.clear();
    layer_pre_relu_.reserve(num_layers_);

    // Input projection
    Tensor h = input_proj_.forward(input);

    // EdgeConv stack with ReLU between layers
    for (size_t i = 0; i < num_layers_; ++i) {
        h = edge_layers_[i].forward_with_adj(h, adj);
        layer_pre_relu_.push_back(h.clone());
        // Apply ReLU in place
        for (size_t r = 0; r < h.rows; ++r) {
            for (size_t c = 0; c < h.cols; ++c) {
                if (h(r, c) < 0.0) h(r, c) = 0.0;
            }
        }
    }

    // h is (N, hidden_dim). Project to (N, out_features), then classify
    // per node.
    last_head_input_ = h.clone();
    Tensor h_proj = head_proj_.forward(h);
    if (num_classes_ == 0) return h_proj;

    // Per-node classification: classifier applied row-wise.
    Tensor logits = classifier_.forward(h_proj);
    return logits;
}

Tensor EdgeConvModel::backward(const Tensor& grad_output, double learning_rate) {
    // Backward through classifier (if any), then head projection, then
    // ReLU masks for each layer in reverse, then through edge layers in
    // reverse, then input projection.
    Tensor grad_h;
    if (num_classes_ > 0) {
        Tensor grad_post_head = classifier_.backward(grad_output, learning_rate);
        grad_h = head_proj_.backward(grad_post_head, learning_rate);
    } else {
        grad_h = head_proj_.backward(grad_output, learning_rate);
    }

    for (int li = (int)num_layers_ - 1; li >= 0; --li) {
        // Apply ReLU mask using the cached pre-ReLU value
        Tensor masked = grad_h;
        const Tensor& pre = layer_pre_relu_[(size_t)li];
        for (size_t r = 0; r < masked.rows; ++r) {
            for (size_t c = 0; c < masked.cols; ++c) {
                if (pre(r, c) <= 0.0) masked(r, c) = 0.0;
            }
        }
        grad_h = edge_layers_[(size_t)li].backward(masked, learning_rate);
    }

    // Backprop through input projection
    grad_h = input_proj_.backward(grad_h, learning_rate);
    return grad_h;
}

void EdgeConvModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& l : edge_layers_) l.update_weights(learning_rate);
    head_proj_.update_weights(learning_rate);
    if (num_classes_ > 0) classifier_.update_weights(learning_rate);
}

void EdgeConvModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& l : edge_layers_) l.zero_grad();
    head_proj_.zero_grad();
    if (num_classes_ > 0) classifier_.zero_grad();
}

std::vector<Tensor*> EdgeConvModel::parameters() {
    std::vector<Tensor*> p = input_proj_.parameters();
    for (auto& l : edge_layers_) {
        auto pp = l.parameters();
        p.insert(p.end(), pp.begin(), pp.end());
    }
    auto ph = head_proj_.parameters();
    p.insert(p.end(), ph.begin(), ph.end());
    if (num_classes_ > 0) {
        auto pc = classifier_.parameters();
        p.insert(p.end(), pc.begin(), pc.end());
    }
    return p;
}

std::vector<Tensor*> EdgeConvModel::gradients() {
    std::vector<Tensor*> g = input_proj_.gradients();
    for (auto& l : edge_layers_) {
        auto gg = l.gradients();
        g.insert(g.end(), gg.begin(), gg.end());
    }
    auto gh = head_proj_.gradients();
    g.insert(g.end(), gh.begin(), gh.end());
    if (num_classes_ > 0) {
        auto gc = classifier_.gradients();
        g.insert(g.end(), gc.begin(), gc.end());
    }
    return g;
}
