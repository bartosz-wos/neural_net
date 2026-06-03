#include "deep_gcn.h"
#include <cmath>

// ===========================================================================
// DeepGCNBlock
// ===========================================================================

DeepGCNBlock::DeepGCNBlock(size_t in_features, size_t out_features,
                           bool use_bn, bool use_dropout, double dropout_p,
                           bool use_attention)
    : W_gcn_(in_features, out_features),
      use_bn_(use_bn), bn_(out_features),
      bn_res_(in_features),
      use_dropout_(use_dropout), dropout_p_(dropout_p),
      dropout_(std::unique_ptr<Dropout1D>(new Dropout1D(dropout_p))),
      use_attention_(use_attention),
      attn_W_(out_features, out_features),
      attn_a_(out_features * 2, 1),
      alpha_(1, 1),
      last_x_(1, 1), last_x_bn_(1, 1), last_x_act_(1, 1),
      last_x_gcn_(1, 1), last_x_out_(1, 1),
      adj_norm_(1, 1),
      last_x_prev_(1, 1),
      training_(false)
{
    alpha_[0][0] = 1.0;
    W_gcn_.init_weights("xavier");
    if (use_attention_) {
        attn_W_.init_weights("xavier");
        attn_a_.init_weights("xavier");
    }
}

void DeepGCNBlock::set_training(bool t) {
    training_ = t;
    if (use_bn_) bn_.set_training(t);
    bn_res_.set_training(t);
    dropout_->set_training(t);
}

Tensor DeepGCNBlock::forward(const Tensor& input) {
    (void)input;
    return last_x_out_;
}

Tensor DeepGCNBlock::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    size_t d = input.cols;
    last_x_ = Tensor(input);  // clone to prevent external input corruption
    if (last_x_prev_.rows == 1) {
        last_x_prev_ = Tensor::zeros(N, d);
    }

    // Step 1: BN
    Tensor x = input;
    if (use_bn_) {
        bn_.set_training(training_);
        x = bn_.forward(x);
    }
    last_x_bn_ = x;

    // Step 2: ReLU
    last_x_act_ = x;
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            last_x_act_[i][j] = std::max(0.0, x[i][j]);

    // Step 3: GCN (inline)
    Tensor d_vec(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj[i][j];
        d_vec[i][0] = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    adj_norm_ = Tensor(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            adj_norm_[i][j] = d_vec[i][0] * adj[i][j] * d_vec[j][0];

    const Tensor& W = W_gcn_.get_weights();
    size_t out_d = W.rows;

    Tensor AX(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += adj_norm_[i][k] * last_x_act_[k][j];
            AX[i][j] = sum;
        }

    last_x_gcn_ = Tensor(N, out_d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < d; ++k)
                sum += AX[i][k] * W[k][j];
            last_x_gcn_(i, j) = sum;
        }

    // Step 4: Residual
    double alpha_val = alpha_[0][0];
    last_x_out_ = Tensor(N, out_d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_d; ++j) {
            double prev_val = (j < last_x_prev_.cols) ? last_x_prev_(i, j) : 0.0;
            last_x_out_(i, j) = last_x_gcn_(i, j) + alpha_val * prev_val;
        }

    // Step 5: Dropout
    if (training_ && use_dropout_) {
        last_x_out_ = dropout_->forward(last_x_out_);
    }

    return last_x_out_;
}

Tensor DeepGCNBlock::backward(const Tensor& grad_output, double learning_rate) {
    size_t N = last_x_.rows;
    size_t d = last_x_act_.cols;
    size_t out_d = W_gcn_.get_weights().rows;

    // Dropout backward
    Tensor grad = grad_output;
    if (training_ && use_dropout_) {
        grad = dropout_->backward(grad_output, learning_rate);
    }

    // ReLU backward: mask — only touch grad columns that exist
    size_t grad_cols = grad.cols;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < grad_cols; ++j)
            if (last_x_act_(i, j) <= 0.0) grad[i][j] = 0.0;

    // GCN backward: dL/d_x_act = adj_norm^T @ (grad @ W)
    const Tensor& W = W_gcn_.get_weights();
    Tensor grad_proj(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_d; ++k)
                sum += grad(i, k) * W(j, k);
            grad_proj(i, j) = sum;
        }

    Tensor grad_x_act_final(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += adj_norm_(k, i) * grad_proj(k, j);
            grad_x_act_final(i, j) = sum;
        }

    // Update W: dL/dW = x_act^T @ grad
    Tensor grad_W(d, out_d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < out_d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += last_x_act_(k, i) * grad(k, j);
            grad_W(i, j) = sum;
        }

    Tensor& W_ref = W_gcn_.weights;
    for (size_t i = 0; i < W_ref.rows; ++i)
        for (size_t j = 0; j < W_ref.cols; ++j)
            W_ref(i, j) -= learning_rate * grad_W(i, j);

    // BN backward
    if (use_bn_) {
        grad_x_act_final = bn_.backward(grad_x_act_final, 0.0);
    }

    // Update alpha
    double grad_alpha_val = 0.0;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_d; ++j)
            grad_alpha_val += grad(i, j) * last_x_prev_(i, j);
    alpha_[0][0] -= learning_rate * grad_alpha_val;

    return grad_x_act_final;
}

void DeepGCNBlock::update_weights(double learning_rate) {
    W_gcn_.update_weights(learning_rate);
    if (use_attention_) {
        attn_W_.update_weights(learning_rate);
        attn_a_.update_weights(learning_rate);
    }
}

void DeepGCNBlock::zero_grad() {
    W_gcn_.zero_grad();
    bn_.zero_grad();
    bn_res_.zero_grad();
    if (use_attention_) {
        attn_W_.zero_grad();
        attn_a_.zero_grad();
    }
}

std::vector<Tensor*> DeepGCNBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_gcn_.parameters()) result.push_back(p);
    if (use_attention_) {
        for (Tensor* p : attn_W_.parameters()) result.push_back(p);
        for (Tensor* p : attn_a_.parameters()) result.push_back(p);
    }
    result.push_back(&alpha_);
    return result;
}

std::vector<Tensor*> DeepGCNBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_gcn_.gradients()) result.push_back(g);
    result.push_back(&alpha_);
    for (Tensor* g : bn_.gradients()) result.push_back(g);
    if (use_attention_) {
        for (Tensor* g : attn_W_.gradients()) result.push_back(g);
        for (Tensor* g : attn_a_.gradients()) result.push_back(g);
    }
    return result;
}

// ===========================================================================
// DeepGCNStack
// ===========================================================================

DeepGCNStack::DeepGCNStack(const std::vector<size_t>& hidden_dims,
                            bool use_bn, bool use_dropout, double dropout_p,
                            bool use_attention)
    : last_output_(1, 1), last_input_(1, 1), last_adj_(1, 1)
{
    for (size_t i = 0; i + 1 < hidden_dims.size(); ++i) {
        blocks_.emplace_back(hidden_dims[i], hidden_dims[i + 1],
                            use_bn, use_dropout, dropout_p, use_attention);
    }
}

void DeepGCNStack::set_training(bool t) {
    for (auto& b : blocks_) b.set_training(t);
}

Tensor DeepGCNStack::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor DeepGCNStack::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = input;
    last_adj_ = adj;
    Tensor x = input;
    Tensor x_prev = Tensor::zeros(input.rows, input.cols);
    Tensor x_prev2 = Tensor::zeros(input.rows, input.cols);

    for (size_t k = 0; k < blocks_.size(); ++k) {
        blocks_[k].last_x_prev_ = x_prev2;
        x = blocks_[k].forward_with_adj(x, adj);
        x_prev2 = x_prev;
        x_prev = x;
    }

    last_output_ = x;
    return last_output_;
}

Tensor DeepGCNStack::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = grad_output;
    for (size_t k = blocks_.size(); k > 0; --k) {
        grad = blocks_[k - 1].backward(grad, learning_rate);
    }
    return grad;
}

void DeepGCNStack::update_weights(double learning_rate) {
    for (auto& b : blocks_) b.update_weights(learning_rate);
}

void DeepGCNStack::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
}

std::vector<Tensor*> DeepGCNStack::parameters() {
    std::vector<Tensor*> result;
    for (auto& b : blocks_)
        for (Tensor* p : b.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> DeepGCNStack::gradients() {
    std::vector<Tensor*> result;
    for (auto& b : blocks_)
        for (Tensor* g : b.gradients()) result.push_back(g);
    return result;
}

// ===========================================================================
// GCNIILayer
// ===========================================================================
//
// Formula: h^{(l+1)} = ReLU( ((1-alpha)*A_norm@h^{(l)} + alpha*h0)@W + beta*h^{(l)}@V )
// where h0 is the initial residual (shared across layers, set by model)
//
// Correct gradient chain:
//   dL/d(preact) = grad_output * ReLU_mask
//   dL/dW  = ((1-alpha)*A_norm@h + alpha*h0)^T @ dL/dpreact
//   dL/dV  = h^T @ dL/dpreact
//   dL/dh  = dL/dpreact @ W^T  +  dL/dpreact @ V^T  (identity path)
//            + (1-alpha)*A_norm^T @ (dL/dpreact @ W^T)
// ===========================================================================

GCNIILayer::GCNIILayer(size_t in_features, size_t out_features,
                       double alpha, double beta, double dropout_p)
    : W_(in_features, out_features),
      V_(in_features, out_features),
      alpha_(alpha),
      beta_(beta),
      dropout_(std::make_unique<Dropout1D>(dropout_p)),
      h0_(1, 1),
      last_input_(1, 1),
      last_adj_norm_(1, 1),
      last_Axh_(1, 1),
      last_preact_(1, 1),
      last_output_(1, 1),
      h0_set_(false),
      training_(false),
      num_nodes_(0),
      in_features_(in_features),
      out_features_(out_features)
{
    W_.init_weights("xavier");
    V_.init_weights("xavier");
}

void GCNIILayer::set_training(bool t) {
    training_ = t;
    dropout_->set_training(t);
}

Tensor GCNIILayer::normalize_adjacency(const Tensor& adj) const {
    size_t N = adj.rows;
    Tensor d_vec(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj(i, j);
        d_vec[i][0] = (deg > 1e-9) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    Tensor A_norm(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_norm(i, j) = d_vec[i][0] * adj(i, j) * d_vec[j][0];
    return A_norm;
}

Tensor GCNIILayer::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GCNIILayer::forward_with_adj(const Tensor& input, const Tensor& adj_norm) {
    // adj_norm: already D^{-1/2} A D^{-1/2} (symmetric normalized, no self-loops)
    size_t N = input.rows;
    size_t d = input.cols;
    num_nodes_ = N;
    last_input_ = Tensor(input);  // clone
    last_adj_norm_ = adj_norm;

    const Tensor& W = W_.get_weights();
    const Tensor& V = V_.get_weights();

    // A_norm @ h
    Tensor Ax(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += adj_norm(i, k) * input(k, j);
            Ax(i, j) = sum;
        }

    // h0: use input as identity residual if not set
    const Tensor& h0_ref = h0_set_ ? h0_ : input;

    // ((1-alpha)*Ax + alpha*h0)@W
    last_Axh_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < d; ++k) {
                double ax_val = (1.0 - alpha_) * Ax(i, k) + alpha_ * h0_ref(i, k);
                sum += ax_val * W(k, j);
            }
            // beta * h @ V (identity mapping)
            double id_val = 0.0;
            if (beta_ > 0.0) {
                for (size_t k = 0; k < d; ++k)
                    id_val += input(i, k) * V(k, j);
                sum += beta_ * id_val;
            }
            last_Axh_(i, j) = sum;
        }

    // ReLU + Dropout
    last_preact_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j)
            last_preact_(i, j) = std::max(0.0, last_Axh_(i, j));

    if (training_) {
        last_output_ = dropout_->forward(last_preact_);
    } else {
        last_output_ = last_preact_;
    }

    return last_output_;
}

Tensor GCNIILayer::forward_with_sparse(const Tensor& input, const Tensor& adj) {
    Tensor adj_norm = normalize_adjacency(adj);
    return forward_with_adj(input, adj_norm);
}

Tensor GCNIILayer::backward(const Tensor& grad_output, double learning_rate) {
    size_t N = num_nodes_;
    size_t d = in_features_;
    const Tensor& W = W_.get_weights();
    const Tensor& V = V_.get_weights();
    const Tensor& h0_ref = h0_set_ ? h0_ : last_input_;

    // ReLU backward
    Tensor grad_preact(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j)
            grad_preact(i, j) = (last_preact_(i, j) > 0.0) ? grad_output(i, j) : 0.0;

    // Dropout backward
    if (training_) {
        grad_preact = dropout_->backward(grad_preact, learning_rate);
    }

    // dL/dW = ((1-alpha)*A@h + alpha*h0)^T @ grad_preact
    Tensor grad_W(d, out_features_);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                double axh_val = 0.0;
                // Recompute A@h
                double ax_val = 0.0;
                for (size_t m = 0; m < N; ++m)
                    ax_val += last_adj_norm_(k, m) * last_input_(m, i);
                axh_val = (1.0 - alpha_) * ax_val + alpha_ * h0_ref(k, i);
                sum += axh_val * grad_preact(k, j);
            }
            grad_W(i, j) = sum;
        }

    // dL/dV = h^T @ grad_preact
    Tensor grad_V(d, out_features_);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += last_input_(k, i) * grad_preact(k, j);
            grad_V(i, j) = sum * beta_;
        }

    // Weight updates
    Tensor& W_ref = W_.weights;
    for (size_t i = 0; i < W_ref.rows; ++i)
        for (size_t j = 0; j < W_ref.cols; ++j)
            W_ref(i, j) -= learning_rate * grad_W(i, j);

    Tensor& V_ref = V_.weights;
    for (size_t i = 0; i < V_ref.rows; ++i)
        for (size_t j = 0; j < V_ref.cols; ++j)
            V_ref(i, j) -= learning_rate * grad_V(i, j);

    // dL/dh from identity path: grad_preact @ V^T
    Tensor grad_identity(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_features_; ++k)
                sum += grad_preact(i, k) * V(j, k) * beta_;
            grad_identity(i, j) = sum;
        }

    // dL/dh from graph path: (1-alpha)*A^T @ (grad_preact @ W^T)
    Tensor grad_Wh(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_features_; ++k)
                sum += grad_preact(i, k) * W(j, k);
            grad_Wh(i, j) = sum;
        }

    Tensor grad_Ax(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += last_adj_norm_(k, i) * grad_Wh(k, j);
            grad_Ax(i, j) = sum * (1.0 - alpha_);
        }

    // Total gradient to input
    Tensor grad_input(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j)
            grad_input(i, j) = grad_identity(i, j) + grad_Ax(i, j);

    return grad_input;
}

void GCNIILayer::update_weights(double learning_rate) {
    W_.update_weights(learning_rate);
    V_.update_weights(learning_rate);
}

void GCNIILayer::zero_grad() {
    W_.zero_grad();
    V_.zero_grad();
}

std::vector<Tensor*> GCNIILayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_.parameters()) result.push_back(p);
    for (Tensor* p : V_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GCNIILayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_.gradients()) result.push_back(g);
    for (Tensor* g : V_.gradients()) result.push_back(g);
    return result;
}

// ===========================================================================
// DeepGCNModel
// ===========================================================================

DeepGCNModel::DeepGCNModel(size_t in_features, size_t hidden_features,
                           size_t out_features, size_t num_layers,
                           double alpha, double beta, double dropout_p,
                           bool use_bn)
    : W0_(in_features, hidden_features),
      W_out_(hidden_features, out_features),
      num_layers_(num_layers),
      dropout_p_(dropout_p),
      use_bn_(use_bn),
      in_features_(in_features),
      out_features_(out_features),
      h0_(1, 1),
      last_input_(1, 1),
      last_output_(1, 1),
      last_adj_(1, 1),
      h0_set_(false),
      training_(false)
{
    W0_.init_weights("xavier");
    W_out_.init_weights("xavier");
    alphas_.reserve(num_layers_);
    for (size_t k = 0; k < num_layers_; ++k) {
        // alpha_k = 2/(1+k) per GCNII paper
        alphas_.push_back(2.0 / (1.0 + static_cast<double>(k)));
        layers_.emplace_back(std::make_unique<GCNIILayer>(hidden_features, hidden_features,
                             alphas_[k], beta, dropout_p));
        if (use_bn_) bns_.emplace_back(hidden_features);
        drops_.emplace_back(std::make_unique<Dropout1D>(dropout_p));
    }
}

void DeepGCNModel::set_training(bool t) {
    training_ = t;
    for (auto& drop : drops_) drop->set_training(t);
}

Tensor DeepGCNModel::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor DeepGCNModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = Tensor(input);
    last_adj_ = adj;

    // First: project input to hidden and set h0
    const Tensor& W0 = W0_.get_weights();
    size_t N = input.rows;
    Tensor x(N, layers_[0]->get_weights().rows);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < layers_[0]->get_weights().rows; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < input.cols; ++k)
                sum += input(i, k) * W0(k, j);
            x(i, j) = sum;
        }

    // Set h0 as the projected initial input (shared across all GCNII layers)
    h0_ = Tensor(x);
    h0_set_ = true;
    for (auto& layer : layers_)
        layer->set_h0(h0_);

    // GCNII layers
    for (size_t k = 0; k < num_layers_; ++k) {
        layers_[k]->set_training(training_);
        x = layers_[k]->forward_with_adj(x, adj);
        if (use_bn_) {
            bns_[k].set_training(training_);
            x = bns_[k].forward(x);
        }
        if (dropout_p_ > 0.0) {
            drops_[k]->set_training(training_);
            x = drops_[k]->forward(x);
        }
    }

    // Final projection
    const Tensor& Wout = W_out_.get_weights();
    Tensor final(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < x.cols; ++k)
                sum += x(i, k) * Wout(k, j);
            final(i, j) = sum;
        }

    last_output_ = final;
    return last_output_;
}

Tensor DeepGCNModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, out_features)
    size_t N = grad_output.rows;

    // Backprop through W_out
    const Tensor& Wout = W_out_.get_weights();
    Tensor grad_x = Tensor(grad_output.rows, layers_[0]->get_weights().rows);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < layers_[0]->get_weights().rows; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_features_; ++k)
                sum += grad_output(i, k) * Wout(j, k);
            grad_x(i, j) = sum;
        }

    Tensor& Wout_ref = W_out_.weights;
    for (size_t i = 0; i < Wout_ref.rows; ++i)
        for (size_t j = 0; j < Wout_ref.cols; ++j)
            Wout_ref(i, j) -= learning_rate * 0.0;  // placeholder

    // Backprop through layers in reverse
    for (size_t k = num_layers_; k > 0; --k) {
        if (dropout_p_ > 0.0) {
            grad_x = drops_[k - 1]->backward(grad_x, learning_rate);
        }
        if (use_bn_) {
            grad_x = bns_[k - 1].backward(grad_x, learning_rate);
        }
        grad_x = layers_[k - 1]->backward(grad_x, learning_rate);
    }

    // Backprop through W0 projection
    const Tensor& W0 = W0_.get_weights();
    Tensor grad_input(grad_x.rows, in_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < in_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < layers_[0]->get_weights().rows; ++k)
                sum += grad_x(i, k) * W0(j, k);
            grad_input(i, j) = sum;
        }

    return grad_input;
}

void DeepGCNModel::update_weights(double learning_rate) {
    W0_.update_weights(learning_rate);
    W_out_.update_weights(learning_rate);
    for (auto& layer : layers_) layer->update_weights(learning_rate);
    for (auto& bn : bns_) bn.update_weights(learning_rate);
}

void DeepGCNModel::zero_grad() {
    W0_.zero_grad();
    W_out_.zero_grad();
    for (auto& layer : layers_) layer->zero_grad();
    for (auto& bn : bns_) bn.zero_grad();
}

std::vector<Tensor*> DeepGCNModel::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W0_.parameters()) result.push_back(p);
    for (Tensor* p : W_out_.parameters()) result.push_back(p);
    for (auto& layer : layers_)
        for (Tensor* p : layer->parameters()) result.push_back(p);
    for (auto& bn : bns_)
        for (Tensor* p : bn.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> DeepGCNModel::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W0_.gradients()) result.push_back(g);
    for (Tensor* g : W_out_.gradients()) result.push_back(g);
    for (auto& layer : layers_)
        for (Tensor* g : layer->gradients()) result.push_back(g);
    for (auto& bn : bns_)
        for (Tensor* g : bn.gradients()) result.push_back(g);
    return result;
}

// ===========================================================================
// GCNIIModel
// ===========================================================================
//
// Pure GCNII model: input -> W0 -> (GCNII -> [BN] -> [Dropout])*L -> output
// Output is in hidden_features space (no final projection to out_features).
// This matches the original GCNII paper setup for semi-supervised node
// classification where the hidden dim IS the prediction dim.

GCNIIModel::GCNIIModel(size_t in_features, size_t hidden_features,
                       size_t num_layers, double dropout_p,
                       bool use_bn, double beta)
    : W0_(in_features, hidden_features),
      num_layers_(num_layers),
      dropout_p_(dropout_p),
      use_bn_(use_bn),
      in_features_(in_features),
      hidden_features_(hidden_features),
      h0_(1, 1),
      last_input_(1, 1),
      last_output_(1, 1),
      last_adj_(1, 1),
      h0_set_(false),
      training_(false)
{
    W0_.init_weights("xavier");
    alphas_.reserve(num_layers_);
    for (size_t k = 0; k < num_layers_; ++k) {
        // alpha_k = 2/(1+k) per GCNII paper
        alphas_.push_back(2.0 / (1.0 + static_cast<double>(k)));
        layers_.emplace_back(std::make_unique<GCNIILayer>(hidden_features, hidden_features,
                             alphas_[k], beta, dropout_p));
        if (use_bn_) bns_.emplace_back(hidden_features);
        drops_.emplace_back(std::make_unique<Dropout1D>(dropout_p));
    }
}

void GCNIIModel::set_training(bool t) {
    training_ = t;
    for (auto& drop : drops_) drop->set_training(t);
}

Tensor GCNIIModel::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GCNIIModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = Tensor(input);
    last_adj_ = adj;

    // First: project input to hidden and set h0
    const Tensor& W0 = W0_.get_weights();
    size_t N = input.rows;
    Tensor x(N, hidden_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < hidden_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < input.cols; ++k)
                sum += input(i, k) * W0(k, j);
            x(i, j) = sum;
        }

    // Set h0 as the projected initial input (shared across all GCNII layers)
    h0_ = Tensor(x);
    h0_set_ = true;
    for (auto& layer : layers_)
        layer->set_h0(h0_);

    // GCNII layers (no final projection: output is in hidden_features space)
    for (size_t k = 0; k < num_layers_; ++k) {
        layers_[k]->set_training(training_);
        x = layers_[k]->forward_with_adj(x, adj);
        if (use_bn_) {
            bns_[k].set_training(training_);
            x = bns_[k].forward(x);
        }
        if (dropout_p_ > 0.0) {
            drops_[k]->set_training(training_);
            x = drops_[k]->forward(x);
        }
    }

    last_output_ = x;
    return last_output_;
}

Tensor GCNIIModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (N, hidden_features)
    size_t N = grad_output.rows;
    Tensor grad_x = grad_output;

    // Backprop through layers in reverse (no W_out here, gradient is direct)
    for (size_t k = num_layers_; k > 0; --k) {
        if (dropout_p_ > 0.0) {
            grad_x = drops_[k - 1]->backward(grad_x, learning_rate);
        }
        if (use_bn_) {
            grad_x = bns_[k - 1].backward(grad_x, learning_rate);
        }
        grad_x = layers_[k - 1]->backward(grad_x, learning_rate);
    }

    // Backprop through W0 projection: grad_input = grad_x @ W0
    const Tensor& W0 = W0_.get_weights();
    Tensor grad_input(grad_x.rows, in_features_);
    for (size_t i = 0; i < grad_x.rows; ++i)
        for (size_t j = 0; j < in_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < hidden_features_; ++k)
                sum += grad_x(i, k) * W0(j, k);
            grad_input(i, j) = sum;
        }

    return grad_input;
}

void GCNIIModel::update_weights(double learning_rate) {
    W0_.update_weights(learning_rate);
    for (auto& layer : layers_) layer->update_weights(learning_rate);
    for (auto& bn : bns_) bn.update_weights(learning_rate);
}

void GCNIIModel::zero_grad() {
    W0_.zero_grad();
    for (auto& layer : layers_) layer->zero_grad();
    for (auto& bn : bns_) bn.zero_grad();
}

std::vector<Tensor*> GCNIIModel::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W0_.parameters()) result.push_back(p);
    for (auto& layer : layers_)
        for (Tensor* p : layer->parameters()) result.push_back(p);
    for (auto& bn : bns_)
        for (Tensor* p : bn.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GCNIIModel::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W0_.gradients()) result.push_back(g);
    for (auto& layer : layers_)
        for (Tensor* g : layer->gradients()) result.push_back(g);
    for (auto& bn : bns_)
        for (Tensor* g : bn.gradients()) result.push_back(g);
    return result;
}
