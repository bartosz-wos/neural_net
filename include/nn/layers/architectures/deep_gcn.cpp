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
    last_x_ = input;
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
            last_x_gcn_[i][j] = sum;
        }

    // Step 4: Residual
    double alpha_val = alpha_[0][0];
    last_x_out_ = Tensor(N, out_d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_d; ++j) {
            double prev_val = (j < last_x_prev_.cols) ? last_x_prev_(i, j) : 0.0;
            last_x_out_[i][j] = last_x_gcn_(i, j) + alpha_val * prev_val;
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
    // NOTE: when d > out_d (upsampling block), columns grad_cols..d-1 of last_x_act_
    // are stale and don't contribute to grad since grad only has grad_cols columns.
    // No need to touch them.

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

GCNIILayer::GCNIILayer(size_t in_features, size_t out_features)
    : W_(in_features, out_features),
      W0_(in_features, out_features),
      adj_(1, 1),
      num_nodes_(0), out_features_(out_features),
      last_input_(1, 1),
      last_Pxh_(1, 1), last_Ixh0_(1, 1),
      last_preact_(1, 1),
      alpha_(0.5)
{
    W_.init_weights("xavier");
    W0_.init_weights("xavier");
}

Tensor GCNIILayer::forward(const Tensor& input) {
    (void)input;
    return last_preact_;
}

Tensor GCNIILayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    size_t d = input.cols;
    num_nodes_ = N;
    last_input_ = input;
    adj_ = adj;

    // Compute degree vector
    Tensor d_vec(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj[i][j];
        d_vec[i][0] = (deg > 1e-9) ? 1.0 / deg : 0.0;
    }

    // P = A @ D^{-1}
    Tensor P(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            P[i][j] = adj[i][j] * d_vec[j][0];

    const Tensor& W = W_.get_weights();
    const Tensor& W0 = W0_.get_weights();

    // Path 1: Ph = P @ input
    Tensor Ph(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += P[i][k] * input[k][j];
            Ph(i, j) = sum;
        }

    // PhW = Ph @ W^T
    last_Pxh_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < d; ++k)
                sum += Ph(i, k) * W[k][j];
            last_Pxh_(i, j) = sum;
        }

    // Path 2: h @ W0
    last_Ixh0_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < d; ++k)
                sum += input[i][k] * W0[k][j];
            last_Ixh0_(i, j) = sum;
        }

    // Pre-activation + ReLU
    last_preact_ = Tensor(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j)
            last_preact_[i][j] = std::max(0.0, last_Pxh_(i, j) + last_Ixh0_(i, j));

    return last_preact_;
}

Tensor GCNIILayer::backward(const Tensor& grad_output, double learning_rate) {
    size_t N = num_nodes_;
    size_t d = last_input_.cols;
    const Tensor& W = W_.get_weights();
    const Tensor& W0 = W0_.get_weights();

    // ReLU gradient
    Tensor grad_preact(N, out_features_);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_features_; ++j)
            grad_preact[i][j] = (last_preact_(i, j) > 0.0) ? grad_output(i, j) : 0.0;

    // Recompute P = A @ D^{-1}
    Tensor d_vec(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j)
            deg += adj_(i, j);
        d_vec[i][0] = (deg > 1e-9) ? 1.0 / deg : 0.0;
    }
    Tensor P(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            P[i][j] = adj_(i, j) * d_vec[j][0];

    // Ph = P @ input
    Tensor Ph(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += P[i][k] * last_input_(k, j);
            Ph(i, j) = sum;
        }

    // dL/dW = Ph^T @ grad_preact
    Tensor grad_W(d, out_features_);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += Ph(k, i) * grad_preact(k, j);
            grad_W(i, j) = sum;
        }

    // dL/dW0 = input^T @ grad_preact
    Tensor grad_W0(d, out_features_);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < out_features_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += last_input_(k, i) * grad_preact(k, j);
            grad_W0(i, j) = sum;
        }

    // Update W with weight decay, W0 without
    double wd = 0.01;
    Tensor& W_ref = W_.weights;
    for (size_t i = 0; i < W_ref.rows; ++i)
        for (size_t j = 0; j < W_ref.cols; ++j)
            W_ref(i, j) = (1.0 - learning_rate * wd) * W_ref(i, j) - learning_rate * grad_W(i, j);

    Tensor& W0_ref = W0_.weights;
    for (size_t i = 0; i < W0_ref.rows; ++i)
        for (size_t j = 0; j < W0_ref.cols; ++j)
            W0_ref(i, j) -= learning_rate * grad_W0(i, j);

    // Gradient to input from Path 1: P^T @ grad_preact @ W^T
    Tensor grad_Ph(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_features_; ++k)
                sum += grad_preact(i, k) * W(j, k);
            grad_Ph(i, j) = sum;
        }

    Tensor grad_input_P(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += P[k][i] * grad_Ph(k, j);
            grad_input_P(i, j) = sum;
        }

    // Gradient to input from Path 2: grad_preact @ W0^T
    Tensor grad_input_I(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < out_features_; ++k)
                sum += grad_preact(i, k) * W0(j, k);
            grad_input_I(i, j) = sum;
        }

    Tensor grad_input(N, d);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j)
            grad_input(i, j) = grad_input_P(i, j) + grad_input_I(i, j);

    return grad_input;
}

void GCNIILayer::update_weights(double learning_rate) {
    W_.update_weights(learning_rate);
    W0_.update_weights(learning_rate);
}

void GCNIILayer::zero_grad() {
    W_.zero_grad();
    W0_.zero_grad();
}

std::vector<Tensor*> GCNIILayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_.parameters()) result.push_back(p);
    for (Tensor* p : W0_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GCNIILayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_.gradients()) result.push_back(g);
    for (Tensor* g : W0_.gradients()) result.push_back(g);
    return result;
}

// ===========================================================================
// GCNIIModel
// ===========================================================================

GCNIIModel::GCNIIModel(size_t in_features, size_t hidden_features,
                        size_t num_layers, double dropout_p,
                        bool use_bn, double beta)
    : num_layers_(num_layers), dropout_p_(dropout_p),
      use_bn_(use_bn), beta_(beta),
      W0_shared_(in_features, hidden_features),
      last_output_(1, 1), last_input_(1, 1), last_adj_(1, 1)
{
    W0_shared_.init_weights("xavier");

    for (size_t k = 0; k < num_layers_; ++k) {
        double alpha = 2.0 / (1.0 + static_cast<double>(k));
        alphas_.push_back(alpha);
        layers_.emplace_back(in_features, hidden_features);
        if (use_bn_) bns_.emplace_back(hidden_features);
        drops_.emplace_back(std::unique_ptr<Dropout1D>(new Dropout1D(dropout_p)));
    }
}

void GCNIIModel::set_training(bool t) {
    for (auto& drop : drops_) drop->set_training(t);
}

Tensor GCNIIModel::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor GCNIIModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    last_input_ = input;
    last_adj_ = adj;
    Tensor x = input;

    for (size_t k = 0; k < num_layers_; ++k) {
        x = layers_[k].forward_with_adj(x, adj);
        if (use_bn_) {
            bns_[k].set_training(true);
            x = bns_[k].forward(x);
        }
        if (dropout_p_ > 0.0) {
            drops_[k]->set_training(true);
            x = drops_[k]->forward(x);
        }
    }

    last_output_ = x;
    return last_output_;
}

Tensor GCNIIModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = grad_output;

    for (size_t k = num_layers_; k > 0; --k) {
        if (dropout_p_ > 0.0) {
            grad = drops_[k - 1]->backward(grad, learning_rate);
        }
        if (use_bn_) {
            grad = bns_[k - 1].backward(grad, learning_rate);
        }
        grad = layers_[k - 1].backward(grad, learning_rate);
    }

    return grad;
}

void GCNIIModel::update_weights(double learning_rate) {
    for (auto& layer : layers_) layer.update_weights(learning_rate);
    for (auto& bn : bns_) bn.update_weights(learning_rate);
}

void GCNIIModel::zero_grad() {
    for (auto& layer : layers_) layer.zero_grad();
    for (auto& bn : bns_) bn.zero_grad();
}

std::vector<Tensor*> GCNIIModel::parameters() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* p : layer.parameters()) result.push_back(p);
    for (auto& bn : bns_)
        for (Tensor* p : bn.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GCNIIModel::gradients() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* g : layer.gradients()) result.push_back(g);
    for (auto& bn : bns_)
        for (Tensor* g : bn.gradients()) result.push_back(g);
    return result;
}