#include "gin.h"
#include <cmath>

// === GINLayer ===

GINLayer::GINLayer(size_t in_features, size_t out_features, size_t hidden_dim,
                   size_t num_mlp_layers)
    : one_plus_eps_(1.0),
      last_input_(1, 1), last_agg_(1, 1), adj_(1, 1),
      num_nodes_(0), in_features_(in_features)
{
    if (num_mlp_layers == 1) {
        // Single linear layer
        fc_layers_.emplace_back(in_features, out_features);
    } else {
        // First layer: in -> hidden
        fc_layers_.emplace_back(in_features, hidden_dim);
        bn_layers_.emplace_back(hidden_dim);
        // Middle layers: hidden -> hidden (if more than 2 total)
        for (size_t i = 2; i < num_mlp_layers - 1; ++i) {
            fc_layers_.emplace_back(hidden_dim, hidden_dim);
            bn_layers_.emplace_back(hidden_dim);
        }
        // Last layer: hidden -> out
        fc_layers_.emplace_back(hidden_dim, out_features);
    }

    // Initialize first layer weights
    if (!fc_layers_.empty()) {
        fc_layers_[0].init_weights("xavier");
    }
}

Tensor GINLayer::forward(const Tensor& input) {
    (void)input;
    return last_agg_;  // last_agg_ holds the MLP output
}

Tensor GINLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (num_nodes, in_features)
    // adj: (num_nodes, num_nodes) adjacency matrix (no self-loops)
    // GIN: h'_{k+1} = MLP( (1+eps_k) * h_k + sum_{j in N(i)} h_j )
    size_t N = input.rows;
    num_nodes_ = N;
    last_input_ = input.clone();  // clone to avoid corruption when input is modified in-place
    adj_ = adj;

    // Aggregate: sum_{j in N(i)} h_j
    // agg[i][f] = sum_{j where adj[i][j]=1} input[j][f]
    Tensor agg = Tensor::zeros(N, input.cols);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj[i][j] > 1e-9) {
                for (size_t f = 0; f < input.cols; ++f) {
                    agg[i][f] += input[j][f];
                }
            }
        }
    }

    // Combine: (1+eps)*h_i + agg
    // This is the input to the MLP
    last_agg_ = Tensor::zeros(N, input.cols);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < input.cols; ++f) {
            last_agg_[i][f] = one_plus_eps_ * input[i][f] + agg[i][f];
        }
    }

    // MLP forward: fc[0] -> bn[0] -> ReLU -> ... -> fc[last]
    Tensor x = last_agg_;

    // Process hidden layers (all but last fc layer)
    size_t num_hidden = fc_layers_.size() - 1;
    pre_bn_outputs_.clear();
    pre_bn_outputs_.resize(num_hidden);

    for (size_t layer_idx = 0; layer_idx < num_hidden; ++layer_idx) {
        // Dense
        x = fc_layers_[layer_idx].forward(x);

        // Store pre-BN output for backward (fc output before BatchNorm)
        // MUST clone — x is modified in-place by subsequent operations
        pre_bn_outputs_[layer_idx] = x.clone();

        // BatchNorm
        bn_layers_[layer_idx].set_training(true);
        x = bn_layers_[layer_idx].forward(x);

        // ReLU (in-place)
        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                x[i][j] = std::max(0.0, x[i][j]);
            }
        }
    }

    // Final fc layer (no BN/ReLU)
    x = fc_layers_.back().forward(x);

    // Store final MLP output as last_agg_ (overwriting the pre-MLP value)
    last_agg_ = x;
    return last_agg_;
}

Tensor GINLayer::backward(const Tensor& grad_output, double learning_rate) {
    size_t N = num_nodes_;
    size_t in_feat = in_features_;

    // grad_output: (N, out_features) from upstream gradient
    // We need to backprop through:
    //   1. Final Dense layer (fc_layers_.back())
    //   2. Hidden layers in reverse: ReLU -> BatchNorm -> Dense (for each)
    //   3. The combine step: last_agg_ = (1+eps)*input + agg

    Tensor grad = grad_output;

    // 1. Backward through final Dense layer
    // grad_fc_input: gradient w.r.t. the input of the final fc (the output of last ReLU)
    Dense& final_fc = fc_layers_.back();
    Tensor grad_fc_input = final_fc.backward(grad, learning_rate);
    final_fc.update_weights(learning_rate);
    grad = grad_fc_input;  // now grad is w.r.t. output of last hidden layer (post-ReLU)

    if (num_nodes_ == 3 && in_features_ == 2) {
        fprintf(stderr, "DEBUG after FC1 backward (dL/d_relu_out):\n");
        for (size_t i = 0; i < N; ++i) {
            fprintf(stderr, "  grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
        }
    }

    // 2. Backward through hidden layers in reverse order
    size_t num_hidden = fc_layers_.size() - 1;

    for (size_t layer_idx = num_hidden; layer_idx > 0; --layer_idx) {
        size_t rev_idx = layer_idx - 1;

        // ReLU backward: grad *= (pre_relu > 0)
        // pre_bn_outputs_[rev_idx] is the fc output before BN (and before ReLU)
        if (num_nodes_ == 3 && in_features_ == 2) {
            fprintf(stderr, "DEBUG MLP backward layer_idx=%zu rev_idx=%zu\n", layer_idx, rev_idx);
            fprintf(stderr, "  Before ReLU mask: grad[0..2][0..1]:\n");
            for (size_t i = 0; i < N; ++i) {
                fprintf(stderr, "    grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
            }
            fprintf(stderr, "  pre_bn_outputs_[%zu] (should be BN input):\n", rev_idx);
            for (size_t i = 0; i < N; ++i) {
                fprintf(stderr, "    pre_bn[%zu][%zu]: %.10f %.10f\n", rev_idx, i, pre_bn_outputs_[rev_idx](i, 0), pre_bn_outputs_[rev_idx](i, 1));
            }
        }

        for (size_t i = 0; i < grad.rows; ++i) {
            for (size_t j = 0; j < grad.cols; ++j) {
                if (pre_bn_outputs_[rev_idx](i, j) <= 0.0) {
                    grad[i][j] = 0.0;
                }
            }
        }

        if (num_nodes_ == 3 && in_features_ == 2) {
            fprintf(stderr, "  After ReLU mask: grad[0..2][0..1]:\n");
            for (size_t i = 0; i < N; ++i) {
                fprintf(stderr, "    grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
            }
        }

        // BatchNorm backward
        // grad is w.r.t. BN output, we need grad w.r.t. BN input (fc output)
        grad = bn_layers_[rev_idx].backward(grad, 0.0);

        if (num_nodes_ == 3 && in_features_ == 2) {
            fprintf(stderr, "  After BN backward: grad[0..2][0..1]:\n");
            for (size_t i = 0; i < N; ++i) {
                fprintf(stderr, "    grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
            }
        }

        // Dense backward: get gradient w.r.t. this layer's input
        Tensor grad_before_fc = fc_layers_[rev_idx].backward(grad, learning_rate);
        fc_layers_[rev_idx].update_weights(learning_rate);
        grad = grad_before_fc;  // now grad is w.r.t. last_agg_ (MLP input)

        if (num_nodes_ == 3 && in_features_ == 2) {
            fprintf(stderr, "  After FC[%zu] backward: grad[0..2][0..1] (dL/d_last_agg):\n", rev_idx);
            for (size_t i = 0; i < N; ++i) {
                fprintf(stderr, "    grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
            }
        }
    }

    // grad is now w.r.t. last_agg_ (the combined (1+eps)*input + agg before MLP)
    // last_agg_ = (1+eps)*input + agg
    // dL/d_input from self-loop: (1+eps) * grad
    // dL/d_input from aggregation: grad flows backward through the aggregation

    Tensor grad_input = Tensor::zeros(N, in_feat);

    // Self-loop contribution: (1+eps) * grad
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_feat; ++f) {
            grad_input[i][f] = one_plus_eps_ * grad(i, f);
        }
    }

    if (num_nodes_ == 3 && in_features_ == 2) {
        fprintf(stderr, "DEBUG aggregation backward:\n");
        fprintf(stderr, "  last_input_ (original input):\n");
        for (size_t i = 0; i < N; ++i) {
            fprintf(stderr, "    last_input_[%zu]: %.10f %.10f\n", i, last_input_(i, 0), last_input_(i, 1));
        }
        fprintf(stderr, "  grad (w.r.t. last_agg_/MLP input) before agg backward:\n");
        for (size_t i = 0; i < N; ++i) {
            fprintf(stderr, "    grad[%zu]: %.10f %.10f\n", i, grad(i, 0), grad(i, 1));
        }
    }

    // Aggregation backward:
    // Forward: agg[i][f] = sum_{j where adj[i][j]=1} input[j][f]
    // So input[j][f] contributes to agg[i][f] for all i where adj[i][j]=1
    // In backward: grad flows from i to j where adj[i][j]=1 (j contributed to i)
    // i.e., for each edge j->i (adj[i][j]=1 means j contributes to i),
    // grad[j][f] += grad_agg[i][f]
    // OR equivalently: adj[j][i]=1 means j receives gradient from i
    // Let's think carefully:
    //   forward: agg[i] = sum_{j in N(i)} input[j]
    //   dL/d_input[j] += sum_{i: j in N(i)} dL/d_agg[i]
    //   i.e., j contributes to i, so gradient flows back from i to j
    //   adj[i][j]=1 means j contributes to i, so gradient flows i -> j
    //   So: grad_input(j, f) += grad(i, f) for all i where adj[i][j]=1
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj_(i, j) > 1e-9) {  // adj[i][j]=1: j contributes to i (gradient flows i -> j)
                for (size_t f = 0; f < in_feat; ++f) {
                    grad_input(j, f) += grad(i, f);
                }
            }
        }
    }

    if (num_nodes_ == 3 && in_features_ == 2) {
        fprintf(stderr, "  After aggregation backward (grad_input):\n");
        for (size_t i = 0; i < N; ++i) {
            fprintf(stderr, "    grad_input[%zu]: %.10f %.10f\n", i, grad_input(i, 0), grad_input(i, 1));
        }
        fprintf(stderr, "  Input values (for reference):\n");
        for (size_t i = 0; i < N; ++i) {
            fprintf(stderr, "    input[%zu]: %.10f %.10f\n", i, last_input_(i, 0), last_input_(i, 1));
        }
    }

    // 3. Update one_plus_eps_ via gradient from self-loop term
    // dL/d_eps = sum_{i,f} input[i][f] * grad[i][f]
    // since d((1+eps)*x)/d_eps = x
    // Only update when learning_rate > 0 (e.g., skip during numerical gradient checks with lr=0)
    if (learning_rate > 0) {
        double grad_eps_val = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t f = 0; f < in_feat; ++f) {
                grad_eps_val += last_input_(i, f) * grad(i, f);
            }
        }
        one_plus_eps_ -= learning_rate * grad_eps_val;
    }

    return grad_input;
}

void GINLayer::update_weights(double learning_rate) {
    for (auto& fc : fc_layers_) fc.update_weights(learning_rate);
    for (auto& bn : bn_layers_) bn.update_weights(learning_rate);
}

void GINLayer::zero_grad() {
    for (auto& fc : fc_layers_) fc.zero_grad();
    for (auto& bn : bn_layers_) bn.zero_grad();
    one_plus_eps_ = 1.0;
}

std::vector<Tensor*> GINLayer::parameters() {
    std::vector<Tensor*> result;
    for (auto& fc : fc_layers_) {
        for (Tensor* p : fc.parameters()) result.push_back(p);
    }
    for (auto& bn : bn_layers_) {
        for (Tensor* p : bn.parameters()) result.push_back(p);
    }
    return result;
}

std::vector<Tensor*> GINLayer::gradients() {
    std::vector<Tensor*> result;
    for (auto& fc : fc_layers_) {
        for (Tensor* g : fc.gradients()) result.push_back(g);
    }
    for (auto& bn : bn_layers_) {
        for (Tensor* g : bn.gradients()) result.push_back(g);
    }
    return result;
}

// === GIN0Layer ===

GIN0Layer::GIN0Layer(size_t in_features, size_t out_features)
    : W_(in_features, out_features),
      last_input_(1, 1), input_plus_agg_(1, 1), adj_(1, 1),
      num_nodes_(0), out_features_(out_features)
{
    W_.init_weights("xavier");
}

Tensor GIN0Layer::forward(const Tensor& input) {
    (void)input;
    return last_agg_;
}

Tensor GIN0Layer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    num_nodes_ = N;
    last_input_ = input.clone();  // clone to avoid corruption when input is modified in-place
    adj_ = adj;

    // Aggregate: sum_{j in N(i)} h_j
    Tensor agg = Tensor::zeros(N, input.cols);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj[i][j] > 1e-9) {
                for (size_t f = 0; f < input.cols; ++f) {
                    agg[i][f] += input[j][f];
                }
            }
        }
    }

    // GIN0: h'_i = W @ (h_i + sum_{j in N(i)} h_j)
    // Compute (h_i + sum_neighbor) = input + agg, store for backward
    input_plus_agg_ = Tensor::zeros(N, input.cols);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < input.cols; ++f) {
            input_plus_agg_[i][f] = input[i][f] + agg[i][f];
        }
    }

    // Apply linear transform: output = input_plus_agg @ W^T
    // W has shape (out_features, in_features), stored row-major
    // output[i][k] = sum_j input_plus_agg[i][j] * W[k][j]
    last_agg_ = Tensor::zeros(N, out_features_);
    const Tensor& W = W_.weights;
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < out_features_; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < input.cols; ++j) {
                sum += input_plus_agg_(i, j) * W(k, j);
            }
            last_agg_(i, k) = sum;
        }
    }
    return last_agg_;
}

Tensor GIN0Layer::backward(const Tensor& grad_output, double learning_rate) {
    size_t N = num_nodes_;
    size_t in_feat = (size_t)last_input_.cols;

    // grad_output: (N, out_features)
    // W: (out_features, in_features)
    //
    // Forward: out = (input + agg) @ W^T
    //   out[i][k] = sum_j (input_plus_agg[i][j]) * W^T[k][j] = sum_j (input_plus_agg[i][j]) * W[j][k]
    //   Wait no, W^T[k][j] = W[j][k]
    //   out[i][k] = sum_j input_plus_agg[i][j] * W^T[k][j]
    //   = sum_j input_plus_agg[i][j] * W[j][k]  (W row j, col k)
    //
    // grad_agg = grad_output @ W   [chain rule: d/d(input_plus_agg) = d/d(out) @ W]
    // grad_agg[i][j] = sum_k grad_output[i][k] * W[k][j]
    // W[k][j] = W(k, j) using Tensor::operator()(row, col)
    Tensor grad_agg = Tensor::zeros(N, in_feat);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < in_feat; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < (size_t)grad_output.cols; ++k) {
                sum += grad_output(i, k) * W_.weights(k, j);
            }
            grad_agg(i, j) = sum;
        }
    }

    // dL/dW = grad_agg^T @ input_plus_agg
    // grad_W[out_row][in_col] = sum_i grad_agg[i][out_row] * input_plus_agg[i][in_col]
    // Wait, dimension analysis:
    //   grad_agg: (N, in_feat) — gradient w.r.t. input_plus_agg rows
    //   input_plus_agg: (N, in_feat)
    //   dL/dW should be (out_feat, in_feat)
    //   grad_W[k][j] = sum_i grad_agg[i][k] * input_plus_agg[i][j]
    //   Let's verify: out[i][k] = sum_j input_plus_agg[i][j] * W[k][j]
    //   d(out[i][k])/d(W[k][j]) = input_plus_agg[i][j]
    //   So dL/dW[k][j] = sum_i dL/d(out[i][k]) * d(out[i][k])/d(W[k][j])
    //                 = sum_i grad_output[i][k] * input_plus_agg[i][j]
    Tensor grad_W = Tensor::zeros(out_features_, in_feat);
    for (size_t k = 0; k < out_features_; ++k) {
        for (size_t j = 0; j < in_feat; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < N; ++i) {
                sum += grad_agg(i, k) * input_plus_agg_(i, j);
            }
            grad_W(k, j) = sum;
        }
    }

    // Update W manually
    Tensor& W_ref = W_.weights;
    for (size_t k = 0; k < out_features_; ++k) {
        for (size_t j = 0; j < in_feat; ++j) {
            W_ref(k, j) -= learning_rate * grad_W(k, j);
        }
    }

    // grad_input from (input + agg):
    // input_plus_agg = input + agg
    // dL/d_input = dL/d_agg (self-loop: 1) + aggregation gradient
    Tensor grad_input = Tensor::zeros(N, in_feat);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < in_feat; ++f) {
            grad_input(i, f) = grad_agg(i, f);  // self-loop: 1 * grad_agg
        }
    }

    // Aggregation backward:
    // forward: agg[i][f] = sum_{j where adj[i][j]=1} input[j][f]
    // gradient flows: grad_input[j][f] += sum_{i: adj[i][j]=1} grad_agg[i][f]
    // i.e., for each edge j->i (adj[i][j]=1), grad flows i -> j
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj_(i, j) > 1e-9) {  // adj[i][j]=1: j contributes to i (gradient flows i -> j)
                for (size_t f = 0; f < in_feat; ++f) {
                    grad_input(j, f) += grad_agg(i, f);
                }
            }
        }
    }

    return grad_input;
}

void GIN0Layer::update_weights(double learning_rate) {
    (void)learning_rate;
    // Weights are updated manually in backward()
    // W_.update_weights would also work if it didn't double-update
}

void GIN0Layer::zero_grad() {
    W_.zero_grad();
}

std::vector<Tensor*> GIN0Layer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GIN0Layer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_.gradients()) result.push_back(g);
    return result;
}