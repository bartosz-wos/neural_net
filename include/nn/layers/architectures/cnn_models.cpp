#include "cnn_models.h"
#include <cmath>

// === VGGBlock ===

VGGBlock::VGGBlock(size_t in_channels, size_t filters[], size_t n_layers,
                   bool use_bn, size_t pool_stride)
    // The pooling layers are SHAPE-STATIC (`MaxPool2D(kH, kW, H_in, W_in,
    // stride_h, stride_w)` bakes the spatial dims in at construction). The
    // conv stack below is built at a fixed 224x224 input with 3x3 kernels and
    // pad 1, so the spatial size is PRESERVED through the convs and the pool
    // receives a 224x224 map. The previous
    // `pool_(2, 2, pool_stride, pool_stride)` passed the STRIDE in the
    // H_in/W_in slots, which built a pool over a `pool_stride x pool_stride`
    // map -- for the default stride 2 that is a 2x2 input with a 2x2 kernel,
    // i.e. a 1x1 output.
    // See docs/plans/2026-10-28-pooling-geometry.md.
    : use_bn_(use_bn), pool_(2, 2, 224, 224, pool_stride, pool_stride), n_layers_(n_layers) {
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

    for (size_t i = 0; i < n_layers_; ++i) {
        last_output_ = convs_[i].forward(last_output_);

        if (use_bn_ && i < bn_gamma_.size()) {
            // Batch norm along channels (assuming channels first layout)
            size_t C = bn_gamma_[i].rows;
            size_t S = last_output_.cols / C;
            // Compute batch statistics
            Tensor batch_mean(1, C), batch_var(1, C);
            for (size_t c = 0; c < C; ++c) {
                double m = 0.0;
                for (size_t b = 0; b < last_output_.rows; ++b) {
                    for (size_t s = 0; s < S; ++s)
                        m += last_output_[b][c * S + s];
                }
                m /= (last_output_.rows * S);
                batch_mean[0][c] = m;
                double v = 0.0;
                for (size_t b = 0; b < last_output_.rows; ++b) {
                    for (size_t s = 0; s < S; ++s) {
                        double d = last_output_[b][c * S + s] - m;
                        v += d * d;
                    }
                }
                v /= (last_output_.rows * S);
                batch_var[0][c] = v;
            }
            // Update running stats (momentum=0.01)
            double mom = 0.01;
            for (size_t c = 0; c < C; ++c) {
                running_mean_[i][c][0] = (1 - mom) * running_mean_[i][c][0] + mom * batch_mean[0][c];
                running_var_[i][c][0] = (1 - mom) * running_var_[i][c][0] + mom * batch_var[0][c];
            }
            Tensor normalized(last_output_.rows, last_output_.cols);
            for (size_t b = 0; b < last_output_.rows; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    double mean = batch_mean[0][c];
                    double var = batch_var[0][c] + 1e-5;
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
    // The spatial chain, derived from the layers' own declared shapes:
    //   conv1 (5x5, pad 0, stride 1) on 32x32 -> 28x28 -> pool1 (2x2, s2) -> 14x14
    //   conv2 (5x5, pad 0, stride 1) on 14x14 -> 10x10 -> pool2 (2x2, s2) ->  5x5
    //   flatten -> 16 * 5 * 5 = 400  ->  fc1 (400, 120)
    // The `fc1_(16 * 5 * 5, 120)` below is the CROSS-CHECK: it is only
    // consistent with conv1 having pad 0 (28x28 -> 14x14 -> 10x10 -> 5x5).
    // With pad 2 the chain gives 32x32 -> 16x16 -> 12x12 -> 6x6 -> 576, which
    // contradicts fc1's declared 400. conv1's pad was 2, which made the whole
    // network inconsistent -- and the network could never run, because it was
    // ABSTRACT (fixed in commit 9fae6e3) and its pools were built with the
    // stride in the H_in/W_in slots (fixed in the pooling commit).
    : conv1_(1, 6, 5, 5, 32, 32, 1, 1, 0, 0),
      conv2_(6, 16, 5, 5, 14, 14, 1, 1, 0, 0),
      // The pooling layers are SHAPE-STATIC: `MaxPool2D(kH, kW, H_in, W_in,
      // stride_h, stride_w)` bakes the spatial dims in at construction, and
      // `forward` asserts `input.cols % (H * W) == 0`. The dims below are the
      // spatial size of the map EACH pool actually receives:
      //   pool1: conv1 output -> 28x28, pooled to 14x14
      //   pool2: conv2 output -> 10x10, pooled to  5x5
      // The previous `pool1_(2, 2, 2, 2), pool2_(2, 2, 2, 2)` passed the
      // STRIDE in the H_in/W_in slots, which collapsed both pools to a 2x2
      // input and a 1x1 output.
      // See docs/plans/2026-10-28-pooling-geometry.md.
      pool1_(2, 2, 28, 28, 2, 2), pool2_(2, 2, 10, 10, 2, 2),
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
    // The spatial chain, derived from the layers' own declared shapes:
    //   conv1 (11x11, s4, pad3) on 224x224 -> 55x55 -> pool1 (3x3, s2) -> 27x27
    //   conv2 (5x5, s1, pad2)   on 27x27   -> 27x27 -> pool2 (3x3, s2) -> 13x13
    //   conv3/4/5 (3x3, s1, pad1) on 13x13 -> 13x13 -> pool3 (3x3, s2) ->  6x6
    //   flatten -> 6 * 6 * 256 = 9216  ->  fc1 (9216, 4096)
    // `conv1_` was declared with H_in = W_in = 55, but 55x55 is conv1's
    // OUTPUT size, not its input: with an 11x11 kernel, stride 4 and pad 3,
    // the OUTPUT of a 224x224 input is exactly 55x55. Declaring the input as
    // 55 made conv1 expect `3 * 55 * 55` columns while the chain fed it
    // `3 * 224 * 224`, and the forward died on Conv2D's shape assertion.
    // The `fc1_(6 * 6 * 256, 4096)` below is the CROSS-CHECK that pins the
    // whole chain: only a 6x6 spatial map reaching fc1 gives 9216.
    : use_lrbn_(use_lrbn),
      conv1_(3, 96, 11, 11, 224, 224, 4, 4, 3, 3),
      conv2_(96, 256, 5, 5, 27, 27, 1, 1, 2, 2),
      conv3_(256, 384, 3, 3, 13, 13, 1, 1, 1, 1),
      conv4_(384, 384, 3, 3, 13, 13, 1, 1, 1, 1),
      conv5_(384, 256, 3, 3, 13, 13, 1, 1, 1, 1),
      // The pooling layers are SHAPE-STATIC (`MaxPool2D(kH, kW, H_in, W_in,
      // stride_h, stride_w)` bakes the spatial dims in at construction). The
      // dims below are the spatial size of the map EACH pool actually
      // receives, from the 224x224 input:
      //   pool1: conv1 output (11x11, s4, pad3) -> 55x55, pooled to 27x27
      //   pool2: conv2 output (5x5, s1, pad2)  -> 27x27, pooled to 13x13
      //   pool3: conv5 output (3x3, s1, pad1)  -> 13x13, pooled to  6x6
      // The previous `pool1_(3, 3, 2, 2), ...` passed the STRIDE in the
      // H_in/W_in slots, giving H_out = (2 - 3) / 1 + 1 = 0 and throwing
      // "MaxPool2D: output dimensions non-positive" at construction.
      // See docs/plans/2026-10-28-pooling-geometry.md.
      pool1_(3, 3, 55, 55, 2, 2), pool2_(3, 3, 27, 27, 2, 2), pool3_(3, 3, 13, 13, 2, 2),
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