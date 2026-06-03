#include "patchy_san.h"
#include <map>
#include <set>

// === PatchySANLayer ===

PatchySANLayer::PatchySANLayer(size_t in_features, size_t out_features,
                               size_t w, size_t k)
    : in_features_(in_features), out_features_(out_features), w_(w), k_(k),
      output_dense_(out_features, out_features),
      last_patch_sum_(1, 1) {

    // w position-wise Dense layers, each mapping in_features -> out_features
    conv_layers_.clear();
    conv_layers_.reserve(w_);
    for (size_t p = 0; p < w_; ++p) {
        Dense d(in_features, out_features);
        d.init_weights("xavier");
        conv_layers_.push_back(d);
    }
    output_dense_.init_weights("xavier");
}

std::vector<uint64_t> PatchySANLayer::compute_labels(const Tensor& adj) const {
    // 1-WL-style iterative relabeling: iterate label refinement.
    // Initial label = degree of node. Then label_new[i] = hash( degree_i, multiset of label_j for j in N(i) ).
    // We do a small fixed number of iterations (3).
    size_t N = adj.rows;
    std::vector<uint64_t> labels(N, 0);
    std::vector<size_t> degrees(N, 0);

    for (size_t i = 0; i < N; ++i) {
        size_t d = 0;
        for (size_t j = 0; j < N; ++j) if (adj(i, j) != 0.0) ++d;
        degrees[i] = d;
        labels[i] = static_cast<uint64_t>(d);
    }

    const size_t num_iters = 3;
    for (size_t it = 0; it < num_iters; ++it) {
        // Build a new label by hashing (degree, sorted multiset of neighbor labels)
        std::vector<uint64_t> new_labels(N, 0);
        for (size_t i = 0; i < N; ++i) {
            std::vector<uint64_t> nbr_labels;
            nbr_labels.reserve(degrees[i]);
            for (size_t j = 0; j < N; ++j) {
                if (adj(i, j) != 0.0) nbr_labels.push_back(labels[j]);
            }
            std::sort(nbr_labels.begin(), nbr_labels.end());

            // FNV-1a-like 64-bit hash
            uint64_t h = 1469598103934665603ULL;
            // Mix in degree
            h ^= static_cast<uint64_t>(degrees[i]);
            h *= 1099511628211ULL;
            for (uint64_t l : nbr_labels) {
                h ^= l;
                h *= 1099511628211ULL;
            }
            new_labels[i] = h;
        }
        labels.swap(new_labels);
    }
    return labels;
}

std::vector<size_t> PatchySANLayer::build_patch(size_t anchor, const Tensor& adj) const {
    // Returns a sequence of up to w_ node ids, starting with `anchor`.
    // BFS up to k_ hops, take first (w_ - 1) neighbors (excluding anchor) discovered,
    // then sort the collected set (except anchor) by (label, id) for canonical ordering.
    size_t N = adj.rows;
    if (N == 0 || w_ == 0) return {};

    std::vector<uint64_t> labels = compute_labels(adj);

    std::vector<size_t> result;
    result.reserve(w_);

    // BFS up to depth k_
    std::vector<bool> visited(N, false);
    std::queue<std::pair<size_t, size_t>> q;  // (node, depth)
    q.push({anchor, 0});
    visited[anchor] = true;

    // Collect neighbors in BFS order, but we'll re-sort by (label, id)
    std::vector<size_t> nbrs;
    while (!q.empty() && (result.size() + nbrs.size()) < (w_ - 1)) {
        auto [u, d] = q.front();
        q.pop();
        for (size_t v = 0; v < N; ++v) {
            if (visited[v]) continue;
            if (adj(u, v) == 0.0) continue;
            visited[v] = true;
            nbrs.push_back(v);
            if (d + 1 < k_) {
                q.push({v, d + 1});
            }
        }
    }

    // Sort neighbors by (label, id) — canonical order
    std::vector<std::pair<uint64_t, size_t>> keyed;
    keyed.reserve(nbrs.size());
    for (size_t v : nbrs) keyed.emplace_back(labels[v], v);
    std::sort(keyed.begin(), keyed.end());

    result.push_back(anchor);
    for (auto& p : keyed) {
        if (result.size() >= w_) break;
        result.push_back(p.second);
    }
    return result;
}

Tensor PatchySANLayer::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, 1);
}

Tensor PatchySANLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (N, in_features), adj: (N, N)
    size_t N = input.rows;
    if (N == 0) return Tensor(0, out_features_);

    last_input_ = input;
    adj_ = adj;
    last_patches_.clear();
    last_patches_.reserve(N);

    // For each anchor i, build a patch of length w_ and apply the position-wise Dense,
    // then sum the position outputs to get a per-node representation.
    Tensor sum_features(N, out_features_);
    sum_features.fill(0.0);

    for (size_t i = 0; i < N; ++i) {
        std::vector<size_t> patch = build_patch(i, adj);
        // Pad to length w_ by repeating the last id (or zero vector if empty)
        if (patch.size() < w_) {
            size_t filler = patch.empty() ? 0 : patch.back();
            while (patch.size() < w_) patch.push_back(filler);
        } else if (patch.size() > w_) {
            patch.resize(w_);
        }
        last_patches_.push_back(patch);

        for (size_t p = 0; p < w_; ++p) {
            // Build a (1, in_features) row from the p-th node in the patch
            Tensor row(1, in_features_);
            size_t nid = patch[p];
            for (size_t f = 0; f < in_features_; ++f)
                row(0, f) = input(nid, f);

            Tensor h = conv_layers_[p].forward(row);  // (1, out_features_)
            for (size_t f = 0; f < out_features_; ++f)
                sum_features(i, f) += h(0, f);
        }
    }

    last_patch_sum_ = sum_features.clone();

    // Final Dense producing the output
    Tensor output = output_dense_.forward(sum_features);  // (N, out_features_)
    return output;
}

Tensor PatchySANLayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features_)
    size_t N = grad_output.rows;

    (void)learning_rate;

    // Backward through output_dense_
    Tensor grad_sum = output_dense_.backward(grad_output, learning_rate);  // (N, out_features_)

    // Distribute grad_sum to each position in the patch
    // For each anchor i, each position p gets grad_sum[i] added.
    // The gradient for input(nid, f) is sum over patches containing nid at position p of
    // grad_sum[i, f] for that anchor. We need to map back to input gradients AND through conv_layers_[p].
    Tensor grad_input(N, in_features_);
    grad_input.fill(0.0);

    for (size_t i = 0; i < N; ++i) {
        const auto& patch = last_patches_[i];
        for (size_t p = 0; p < w_; ++p) {
            // grad over conv_layers_[p] output: grad_sum[i, :]
            Tensor grad_row_input(1, out_features_);
            for (size_t f = 0; f < out_features_; ++f)
                grad_row_input(0, f) = grad_sum(i, f);

            Tensor grad_row = conv_layers_[p].backward(grad_row_input, learning_rate);
            size_t nid = patch[p];
            for (size_t f = 0; f < in_features_; ++f)
                grad_input(nid, f) += grad_row(0, f);
        }
    }

    return grad_input;
}

void PatchySANLayer::update_weights(double learning_rate) {
    for (auto& d : conv_layers_) d.update_weights(learning_rate);
    output_dense_.update_weights(learning_rate);
}

void PatchySANLayer::zero_grad() {
    for (auto& d : conv_layers_) d.zero_grad();
    output_dense_.zero_grad();
}

std::vector<Tensor*> PatchySANLayer::parameters() {
    std::vector<Tensor*> result;
    for (auto& d : conv_layers_) {
        auto p = d.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto p = output_dense_.parameters();
    result.insert(result.end(), p.begin(), p.end());
    return result;
}

std::vector<Tensor*> PatchySANLayer::gradients() {
    std::vector<Tensor*> result;
    for (auto& d : conv_layers_) {
        auto g = d.gradients();
        result.insert(result.end(), g.begin(), g.end());
    }
    auto g = output_dense_.gradients();
    result.insert(result.end(), g.begin(), g.end());
    return result;
}

// === PatchySANModel ===

PatchySANModel::PatchySANModel(size_t in_features, size_t hidden_dim, size_t out_features,
                               size_t w, size_t k)
    : in_features_(in_features), hidden_dim_(hidden_dim), out_features_(out_features),
      w_(w), k_(k),
      input_proj_(in_features, hidden_dim),
      patchy_layer_(hidden_dim, hidden_dim, w, k),
      classifier_(hidden_dim, out_features),
      last_proj_(1, 1),
      last_output_(1, 1) {
    input_proj_.init_weights("xavier");
    classifier_.init_weights("xavier");
}

Tensor PatchySANModel::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor PatchySANModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // Project node features into hidden_dim
    last_input_ = input;
    adj_ = adj;
    last_proj_ = input_proj_.forward(input);
    Tensor h = patchy_layer_.forward_with_adj(last_proj_, adj);
    last_output_ = classifier_.forward(h);
    return last_output_;
}

Tensor PatchySANModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor g1 = classifier_.backward(grad_output, learning_rate);
    Tensor g2 = patchy_layer_.backward(g1, learning_rate);
    Tensor g3 = input_proj_.backward(g2, learning_rate);
    return g3;
}

void PatchySANModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    patchy_layer_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void PatchySANModel::zero_grad() {
    input_proj_.zero_grad();
    patchy_layer_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> PatchySANModel::parameters() {
    std::vector<Tensor*> result;
    auto p1 = input_proj_.parameters();
    result.insert(result.end(), p1.begin(), p1.end());
    auto p2 = patchy_layer_.parameters();
    result.insert(result.end(), p2.begin(), p2.end());
    auto p3 = classifier_.parameters();
    result.insert(result.end(), p3.begin(), p3.end());
    return result;
}

std::vector<Tensor*> PatchySANModel::gradients() {
    std::vector<Tensor*> result;
    auto g1 = input_proj_.gradients();
    result.insert(result.end(), g1.begin(), g1.end());
    auto g2 = patchy_layer_.gradients();
    result.insert(result.end(), g2.begin(), g2.end());
    auto g3 = classifier_.gradients();
    result.insert(result.end(), g3.begin(), g3.end());
    return result;
}
