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
