#include "cnn_models.h"
#include <cmath>

// === VGGBlock ===

VGGBlock::VGGBlock(size_t in_channels, size_t filters[], size_t n_layers,
                   bool use_bn, size_t pool_stride)
    : use_bn_(use_bn), pool_(2, 2, pool_stride, pool_stride), n_layers_(n_layers) {
    size_t ch = in_channels;
    for (size_t i = 0; i < n_layers; ++i) {
        convs_.emplace_back(ch, filters[i], 3, 3, 224, 224, 1, 1, 1, 1);
        if (use_bn) {
            bn_gamma_.emplace_back(filters[i], 1);
            bn_beta_.emplace_back(filters[i], 1);
            running_mean_.emplace_back(filters[i], 1);
            running_var_.emplace_back(filters[i], 1);
        }
        ch = filters[i];
    }
}

Tensor VGGBlock::forward(const Tensor& input) {
    last_output_ = input;
    // Detect H, W from input: cols = in_channels * H * W
    size_t ch = last_output_.cols / (H * W); // infer channels (not used further, but correct)
    (void)ch; // suppress unused warning

    for (size_t i = 0; i < n_layers_; ++i) {
        last_output_ = convs_[i].forward(last_output_);

        if (use_bn_ && i < bn_gamma_.size()) {
            // Batch norm along channels (assuming channels first layout)
            size_t C = bn_gamma_[i].rows;
            size_t S = last_output_.cols / C;
            Tensor normalized(last_output_.rows, last_output_.cols);
            for (size_t b = 0; b < last_output_.rows; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    double mean = running_mean_[i][c][0];
                    double var = running_var_[i][c][0] + 1e-5;
                    double gamma = bn_gamma_[i][c][0];
                    double beta = bn_beta_[i][c][0];
                    for (size_t s = 0; s < S; ++s) {
                        double x = last_output_[b][c * S + s];
                        normalized[b][c * S + s] = gamma * (x - mean) / std::sqrt(var) + beta;
                    }
                }
            }
            last_output_ = normalized;
        }

        // ReLU activation
        for (size_t b = 0; b < last_output_.rows; ++b)
            for (size_t j = 0; j < last_output_.cols; ++j)
                last_output_[b][j] = std::max(0.0, last_output_[b][j]);
    }

    // Pool
    last_output_ = pool_.forward(last_output_);
    return last_output_;
}

Tensor VGGBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void VGGBlock::update_weights(double learning_rate) {
    for (auto& c : convs_) c.update_weights(learning_rate);
}

void VGGBlock::zero_grad() {
    for (auto& c : convs_) c.zero_grad();
}

std::vector<Tensor*> VGGBlock::parameters() {
    std::vector<Tensor*> result;
    for (auto& c : convs_)
        for (Tensor* p : c.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> VGGBlock::gradients() {
    std::vector<Tensor*> result;
    for (auto& c : convs_)
        for (Tensor* g : c.gradients())
            result.push_back(g);
    return result;
}

// === LeNet-5 ===

LeNet5::LeNet5(size_t num_classes)
    : conv1_(1, 6, 5, 5, 32, 32, 1, 1, 2, 2),
      conv2_(6, 16, 5, 5, 14, 14, 1, 1, 0, 0),
      pool1_(2, 2, 2, 2), pool2_(2, 2, 2, 2),
      fc1_(16 * 5 * 5, 120), fc2_(120, 84), fc3_(84, num_classes) {}

Tensor LeNet5::forward(const Tensor& input) {
    // Conv1 -> ReLU -> Pool1
    Tensor x = conv1_.forward(input);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool1_.forward(x);

    // Conv2 -> ReLU -> Pool2
    x = conv2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool2_.forward(x);

    // Flatten -> FC1 -> ReLU -> FC2 -> ReLU -> FC3
    x = flatten_.forward(x);
    x = fc1_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = fc2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    last_output_ = fc3_.forward(x);
    return last_output_;
}

Tensor LeNet5::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void LeNet5::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv2_.update_weights(learning_rate);
    fc1_.update_weights(learning_rate);
    fc2_.update_weights(learning_rate);
    fc3_.update_weights(learning_rate);
}

void LeNet5::zero_grad() {
    conv1_.zero_grad(); conv2_.zero_grad();
    fc1_.zero_grad(); fc2_.zero_grad(); fc3_.zero_grad();
}

std::vector<Tensor*> LeNet5::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    for (Tensor* p : fc3_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> LeNet5::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv2_.gradients()) result.push_back(g);
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    for (Tensor* g : fc3_.gradients()) result.push_back(g);
    return result;
}

// === AlexNet ===

AlexNet::AlexNet(size_t num_classes, bool use_lrbn)
    : use_lrbn_(use_lrbn),
      conv1_(3, 96, 11, 11, 55, 55, 4, 4, 3, 3),
      conv2_(96, 256, 5, 5, 27, 27, 1, 1, 2, 2),
      conv3_(256, 384, 3, 3, 13, 13, 1, 1, 1, 1),
      conv4_(384, 384, 3, 3, 13, 13, 1, 1, 1, 1),
      conv5_(384, 256, 3, 3, 13, 13, 1, 1, 1, 1),
      pool1_(3, 3, 2, 2), pool2_(3, 3, 2, 2), pool3_(3, 3, 2, 2),
      flatten_(),
      fc1_(6 * 6 * 256, 4096), fc2_(4096, 4096), fc3_(4096, num_classes) {}

Tensor AlexNet::forward(const Tensor& input) {
    // conv1 -> relu -> pool1 -> lrn
    Tensor x = conv1_.forward(input);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool1_.forward(x);

    // conv2 -> relu -> pool2 -> lrn
    x = conv2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool2_.forward(x);

    // conv3 -> relu -> conv4 -> relu -> conv5 -> relu -> pool3
    x = conv3_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = conv4_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = conv5_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool3_.forward(x);

    // Flatten -> fc1 -> relu -> dropout -> fc2 -> relu -> dropout -> fc3
    x = flatten_.forward(x);
    x = fc1_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = fc2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    last_output_ = fc3_.forward(x);
    return last_output_;
}

Tensor AlexNet::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void AlexNet::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate); conv2_.update_weights(learning_rate);
    conv3_.update_weights(learning_rate); conv4_.update_weights(learning_rate);
    conv5_.update_weights(learning_rate);
    fc1_.update_weights(learning_rate); fc2_.update_weights(learning_rate); fc3_.update_weights(learning_rate);
}

void AlexNet::zero_grad() {
    conv1_.zero_grad(); conv2_.zero_grad(); conv3_.zero_grad(); conv4_.zero_grad(); conv5_.zero_grad();
    fc1_.zero_grad(); fc2_.zero_grad(); fc3_.zero_grad();
}

std::vector<Tensor*> AlexNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    for (Tensor* p : conv3_.parameters()) result.push_back(p);
    for (Tensor* p : conv4_.parameters()) result.push_back(p);
    for (Tensor* p : conv5_.parameters()) result.push_back(p);
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    for (Tensor* p : fc3_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> AlexNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv2_.gradients()) result.push_back(g);
    for (Tensor* g : conv3_.gradients()) result.push_back(g);
    for (Tensor* g : conv4_.gradients()) result.push_back(g);
    for (Tensor* g : conv5_.gradients()) result.push_back(g);
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    for (Tensor* g : fc3_.gradients()) result.push_back(g);
    return result;
}