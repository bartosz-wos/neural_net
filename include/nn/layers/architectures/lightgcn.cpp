#include "lightgcn.h"
#include <algorithm>

// === LightGCNLayer ===

LightGCNLayer::LightGCNLayer(size_t num_layers, bool learnable_combination)
    : num_layers_(num_layers), learnable_combination_(learnable_combination) {
    // Initialize combination weights uniformly: alpha_k = 1/(K+1)
    size_t K = num_layers_ + 1;  // we have outputs H^{(0)}...H^{(K)}
    alpha_ = Tensor(K, 1);
    for (size_t i = 0; i < K; ++i)
        alpha_[i][0] = 1.0 / K;
    
    // Gradient holder for learnable alpha
    if (learnable_combination_) {
        // alpha_ is already initialized, will be softmax'd in forward
    }
}

void LightGCNLayer::normalize_adjacency(const Tensor& adj) {
    // adj: (N, N) adjacency matrix
    // Compute D^{-1/2} (A + I) D^{-1/2}
    size_t N = adj.rows;
    
    // Add self-loops: A' = A + I
    // Compute degree vector (including self-loops)
    Tensor d(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj[i][j];
        deg += 1.0;  // self-loop
        d[i][0] = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    
    // Normalized: D^{-1/2} (A+I) D^{-1/2}
    adj_norm_ = Tensor(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double aij = (i == j) ? adj[i][j] + 1.0 : adj[i][j];  // add self-loop
            adj_norm_[i][j] = d[i][0] * aij * d[j][0];
        }
    }
}

Tensor LightGCNLayer::compute_output() const {
    // Weighted sum of layer outputs: sum_k alpha_k * H^{(k)}
    size_t K = layer_outputs_.size();
    size_t N = layer_outputs_[0].rows;
    size_t dim = layer_outputs_[0].cols;
    
    Tensor result(N, dim);
    result.fill(0.0);
    
    for (size_t k = 0; k < K; ++k) {
        double ak = alpha_[k][0];
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < dim; ++j)
                result[i][j] += ak * layer_outputs_[k][i][j];
    }
    return result;
}

Tensor LightGCNLayer::forward(const Tensor& input) {
    (void)input;
    return Tensor(1, 1);
}

Tensor LightGCNLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (num_nodes, embedding_dim)
    // adj: (num_nodes, num_nodes) adjacency matrix
    size_t N = input.rows;
    size_t dim = input.cols;
    
    // Normalize adjacency (with self-loops)
    normalize_adjacency(adj);
    
    // Softmax alpha if learnable
    if (learnable_combination_) {
        // softmax over alpha
        double max_alpha = alpha_[0][0];
        for (size_t i = 1; i < alpha_.rows; ++i)
            max_alpha = std::max(max_alpha, alpha_[i][0]);
        
        double sum_exp = 0.0;
        for (size_t i = 0; i < alpha_.rows; ++i) {
            alpha_[i][0] = std::exp(alpha_[i][0] - max_alpha);
            sum_exp += alpha_[i][0];
        }
        for (size_t i = 0; i < alpha_.rows; ++i)
            alpha_[i][0] /= sum_exp;
    }
    
    // Propagate K times, storing all outputs
    layer_outputs_.clear();
    layer_outputs_.reserve(num_layers_ + 1);
    
    Tensor H = input;  // H^{(0)}
    layer_outputs_.push_back(H);
    
    for (size_t k = 0; k < num_layers_; ++k) {
        // H^{(k+1)} = adj_norm_ @ H^{(k)}
        Tensor H_next(N, dim);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (size_t m = 0; m < N; ++m)
                    sum += adj_norm_[i][m] * H[m][j];
                H_next[i][j] = sum;
            }
        }
        H = H_next;
        layer_outputs_.push_back(H);
    }
    
    return compute_output();
}

Tensor LightGCNLayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (num_nodes, embedding_dim)
    // dL/dH_final = grad_output
    // H_final = sum_k alpha_k * H^{(k)}
    // dL/dH^{(k)} = alpha_k * grad_output  (chain rule through weighted sum)
    // Then backprop through each propagation step:
    //   H^{(k)} = adj_norm_ @ H^{(k-1)}
    //   dL/dH^{(k-1)} = adj_norm_^T @ dL/dH^{(k)}}
    
    size_t N = grad_output.rows;
    size_t dim = layer_outputs_[0].cols;
    size_t K = layer_outputs_.size();  // K = num_layers_ + 1
    
    (void)learning_rate;
    
    // Step 1: Compute gradient w.r.t. each layer output
    // grad_layer[k] = dL/dH^{(k)} = alpha_k * grad_output
    std::vector<Tensor> grad_layer(K);
    for (size_t k = 0; k < K; ++k) {
        grad_layer[k] = Tensor(N, dim);
        double ak = alpha_[k][0];
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < dim; ++j)
                grad_layer[k][i][j] = ak * grad_output[i][j];
    }
    
    // Step 2: Backprop through propagation steps (from K-1 down to 0)
    // dL/dH^{(k)} += adj_norm_^T @ dL/dH^{(k+1)}} for k = 0..K-2
    for (size_t k = K - 1; k > 0; --k) {
        // grad_layer[k] already has contribution from output
        // Add contribution from layer k via propagation
        // dL/dH^{(k-1)} += adj_norm_^T @ grad_layer[k]
        Tensor grad_H_k = grad_layer[k];
        
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (size_t m = 0; m < N; ++m)
                    sum += adj_norm_[m][i] * grad_H_k[m][j];  // adj_norm_^T
                grad_layer[k-1][i][j] += sum;
            }
        }
    }
    
    // Step 3: Update alpha if learnable
    // dL/dalpha_k = sum_{i,j} dL/dH^{(k)}[i][j] * H^{(k)}[i][j]
    if (learnable_combination_) {
        Tensor grad_alpha(K, 1);
        for (size_t k = 0; k < K; ++k) {
            double sum = 0.0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < dim; ++j)
                    sum += grad_layer[k][i][j] * layer_outputs_[k][i][j];
            grad_alpha[k][0] = sum;
        }
        
        // Gradient descent on alpha (simple update, ignoring softmax gradient for simplicity)
        for (size_t k = 0; k < K; ++k)
            alpha_[k][0] -= learning_rate * grad_alpha[k][0];
        
        // Re-normalize alpha with softmax
        double max_alpha = alpha_[0][0];
        for (size_t i = 1; i < K; ++i)
            max_alpha = std::max(max_alpha, alpha_[i][0]);
        
        double sum_exp = 0.0;
        for (size_t i = 0; i < K; ++i) {
            alpha_[i][0] = std::exp(alpha_[i][0] - max_alpha);
            sum_exp += alpha_[i][0];
        }
        for (size_t i = 0; i < K; ++i)
            alpha_[i][0] /= sum_exp;
    }
    
    // Return gradient w.r.t. original input H^{(0)}
    return grad_layer[0];
}

void LightGCNLayer::update_weights(double learning_rate) {
    (void)learning_rate;
    // No learnable weights in propagation
}

void LightGCNLayer::zero_grad() {
    // No gradients to zero (alpha is updated directly in backward)
}

std::vector<Tensor*> LightGCNLayer::parameters() {
    // No parameters in propagation layers
    return {};
}

std::vector<Tensor*> LightGCNLayer::gradients() {
    return {};
}

// === LightGCNModel ===

LightGCNModel::LightGCNModel(size_t num_nodes, size_t in_features, size_t embedding_dim,
                             size_t num_layers, bool learnable_combination)
    : num_nodes_(num_nodes), in_features_(in_features), embedding_dim_(embedding_dim),
      num_layers_(num_layers), learnable_combination_(learnable_combination),
      has_learned_embeddings_(false),
      gcn_layer_(num_layers, learnable_combination),
      last_output_(1, 1) {
    
    // Initialize embeddings if in_features doesn't match embedding_dim
    // or if we want learned embeddings regardless
    if (in_features == embedding_dim) {
        // Use raw features as input, no learned embeddings
        has_learned_embeddings_ = false;
    } else {
        // Need a projection layer: embed raw features to embedding_dim
        // Actually for LightGCN, we typically either:
        // 1. Use learned embeddings directly (num_nodes, embedding_dim)
        // 2. Use raw features (num_nodes, in_features) as is
        // For simplicity, use learned embeddings when in_features != embedding_dim
        has_learned_embeddings_ = true;
        embedding_ = Tensor::random(num_nodes, embedding_dim, 0.1);
        grad_embedding_ = Tensor(num_nodes, embedding_dim);
    }
}

Tensor LightGCNModel::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor LightGCNModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: either (num_nodes,) integer indices OR (num_nodes, in_features) features
    // adj: (num_nodes, num_nodes) adjacency matrix
    
    last_input_ = input;
    adj_ = adj;
    
    Tensor emb_input;
    
    if (input.cols == 1 && input.rows == num_nodes_) {
        // Integer indices: lookup embeddings
        // This means input is (num_nodes,) of indices - but our Tensor doesn't store int
        // Actually this case is ambiguous with (num_nodes, 1) features
        // We distinguish by: if values are close to integers in range [0, num_nodes), treat as indices
        // For simplicity, assume: if input.cols == 1 and all values are in [0, num_nodes), it's indices
        bool looks_like_indices = true;
        for (size_t i = 0; i < input.rows && looks_like_indices; ++i) {
            double v = input[i][0];
            if (std::abs(v - std::round(v)) > 1e-6 || v < 0 || v >= num_nodes_) {
                looks_like_indices = false;
            }
        }
        
        if (looks_like_indices && has_learned_embeddings_) {
            // Lookup embeddings
            emb_input = Tensor(num_nodes_, embedding_dim_);
            for (size_t i = 0; i < num_nodes_; ++i) {
                size_t idx = static_cast<size_t>(std::round(input[i][0]));
                if (idx < num_nodes_) {
                    for (size_t j = 0; j < embedding_dim_; ++j)
                        emb_input[i][j] = embedding_[idx][j];
                }
            }
        } else {
            // Treat as features, project if needed
            if (has_learned_embeddings_) {
                // input is features, but we have learned embeddings - use features as is
                // LightGCN accepts (num_nodes, in_features) where in_features = embedding_dim_
                emb_input = input;
            } else {
                emb_input = input;
            }
        }
    } else {
        // Multi-dimensional input: treat as features
        emb_input = input;
    }
    
    last_emb_input_ = emb_input;
    last_output_ = gcn_layer_.forward_with_adj(emb_input, adj);
    
    return last_output_;
}

Tensor LightGCNModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (num_nodes, embedding_dim)
    
    // Backward through LightGCNLayer
    Tensor grad_emb_input = gcn_layer_.backward(grad_output, learning_rate);
    
    // Gradient w.r.t. embeddings (if learned)
    if (has_learned_embeddings_) {
        // grad_emb_input is (num_nodes, embedding_dim)
        // This is gradient flowing into the embedding lookup
        // For embedding lookup: dL/dembedding[j] += sum over nodes i where input[i]==j of grad_emb_input[i]
        
        // Re-compute which nodes correspond to which embeddings
        // We need to know the indices that were used in forward
        // For index-based input:
        bool looks_like_indices = true;
        for (size_t i = 0; i < last_input_.rows && looks_like_indices; ++i) {
            double v = last_input_[i][0];
            if (std::abs(v - std::round(v)) > 1e-6 || v < 0 || v >= num_nodes_) {
                looks_like_indices = false;
            }
        }
        
        if (looks_like_indices) {
            grad_embedding_.fill(0.0);
            for (size_t i = 0; i < last_input_.rows; ++i) {
                size_t idx = static_cast<size_t>(std::round(last_input_[i][0]));
                if (idx < num_nodes_) {
                    for (size_t j = 0; j < embedding_dim_; ++j)
                        grad_embedding_[idx][j] += grad_emb_input[i][j];
                }
            }
            
            // Gradient descent on embeddings
            for (size_t i = 0; i < num_nodes_; ++i)
                for (size_t j = 0; j < embedding_dim_; ++j)
                    embedding_[i][j] -= learning_rate * grad_embedding_[i][j];
        }
    }
    
    // Return gradient w.r.t. original input (indices or features)
    // For features: this is the gradient w.r.t. input features
    // For indices: we don't backprop into indices (they're integers)
    return Tensor(1, 1);  // no meaningful gradient for original input
}

void LightGCNModel::update_weights(double learning_rate) {
    gcn_layer_.update_weights(learning_rate);
}

void LightGCNModel::zero_grad() {
    gcn_layer_.zero_grad();
    if (has_learned_embeddings_)
        grad_embedding_.fill(0.0);
}

std::vector<Tensor*> LightGCNModel::parameters() {
    std::vector<Tensor*> result;
    if (has_learned_embeddings_)
        result.push_back(&embedding_);
    return result;
}

std::vector<Tensor*> LightGCNModel::gradients() {
    std::vector<Tensor*> result;
    if (has_learned_embeddings_)
        result.push_back(&grad_embedding_);
    return result;
}