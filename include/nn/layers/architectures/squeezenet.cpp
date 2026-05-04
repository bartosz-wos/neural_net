#include "squeezenet.h"

FireModule::FireModule(size_t in_channels, size_t squeeze_ch,
                       size_t expand1_ch, size_t expand3_ch,
                       size_t H_in, size_t W_in)
    : squeeze_(in_channels, squeeze_ch, 1, 1, H_in, W_in, 1, 1, 0, 0),
      expand1x1_(squeeze_ch, expand1_ch, 1, 1, H_in, W_in, 1, 1, 0, 0),
      expand3x3_(squeeze_ch, expand3_ch, 3, 3, H_in, W_in, 1, 1, 1, 1),
      expand_ch_(expand1_ch + expand3_ch),
      H_out_(H_in), W_out_(W_in),
      last_output_(1, expand_ch_ * H_out_ * W_out_) {}

Tensor FireModule::forward(const Tensor& input) {
    // Squeeze: H_in x W_in x in_channels -> H_out x W_out x squeeze_ch
    Tensor x = squeeze_.forward(input);
    // H_out x W_out x squeeze_ch

    // Expand: two branches, concat along channel
    Tensor e1 = expand1x1_.forward(x);
    Tensor e3 = expand3x3_.forward(x);
    // Both: H_out x W_out x expand_ch_

    // Concat e1 + e3 along channel
    last_output_ = Tensor(e1.rows, (expand_ch_) * H_out_ * W_out_);
    for (size_t b = 0; b < e1.rows; ++b) {
        size_t out_idx = 0;
        for (size_t h = 0; h < H_out_; ++h) {
            for (size_t w = 0; w < W_out_; ++w) {
                for (size_t c = 0; c < static_cast<size_t>(expand1x1_.out_channels); ++c)
                    last_output_[b][out_idx++] = e1[b][c * H_out_ * W_out_ + h * W_out_ + w];
                for (size_t c = 0; c < static_cast<size_t>(expand3x3_.out_channels); ++c)
                    last_output_[b][out_idx++] = e3[b][c * H_out_ * W_out_ + h * W_out_ + w];
            }
        }
    }
    return last_output_;
}

Tensor FireModule::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, expand_ch * H_out * W_out)
    // Forward: squeeze → e1=expand1x1(x), e3=expand3x3(x) → concat(e1, e3)
    // Concat along channel: e1 gets first expand1_ch, e3 gets next expand3_ch
    // To backprop: split grad along channel dimension, backprop each expand, sum for squeeze

    size_t batch = grad_output.rows;
    size_t e1_ch = expand1x1_.out_channels;
    size_t e3_ch = expand3x3_.out_channels;
    size_t spatial = H_out_ * W_out_;

    // Split gradient: grad_e1 = first e1_ch channels, grad_e3 = next e3_ch channels
    Tensor grad_e1(batch, e1_ch * spatial);
    Tensor grad_e3(batch, e3_ch * spatial);
    for (size_t b = 0; b < batch; ++b) {
        // grad_output layout: [e1_ch * spatial channels] [e3_ch * spatial channels]
        for (size_t s = 0; s < e1_ch * spatial; ++s)
            grad_e1[b][s] = grad_output[b][s];
        for (size_t s = 0; s < e3_ch * spatial; ++s)
            grad_e3[b][s] = grad_output[b][e1_ch * spatial + s];
    }

    // Backprop expand3x3 (before reshape): needs spatial reformat
    // expand3x3_ outputs (batch, e3_ch * spatial) where layout is per-channel spatial
    // But its forward stores output in column-major spatial layout
    // grad_e3 format matches its forward output format, so pass directly
    Tensor grad_x3 = expand3x3_.backward(grad_e3, learning_rate);
    Tensor grad_x1 = expand1x1_.backward(grad_e1, learning_rate);

    // Sum gradients for squeeze input (squeeze got both branches as input)
    // Squeeze forward: input → squeeze_ → output (squeeze_ch channels)
    // Both expand branches got the same squeeze output as input
    // So grad_to_squeeze = grad_x1 + grad_x3 (since both contributed to final concat)
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < grad_x1.cols; ++j)
            grad_x1[b][j] += grad_x3[b][j];

    // Backprop squeeze conv
    return squeeze_.backward(grad_x1, learning_rate);
}

void FireModule::update_weights(double learning_rate) {
    squeeze_.update_weights(learning_rate);
    expand1x1_.update_weights(learning_rate);
    expand3x3_.update_weights(learning_rate);
}

void FireModule::zero_grad() {
    squeeze_.zero_grad();
    expand1x1_.zero_grad();
    expand3x3_.zero_grad();
}

std::vector<Tensor*> FireModule::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : squeeze_.parameters()) result.push_back(p);
    for (Tensor* p : expand1x1_.parameters()) result.push_back(p);
    for (Tensor* p : expand3x3_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> FireModule::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : squeeze_.gradients()) result.push_back(g);
    for (Tensor* g : expand1x1_.gradients()) result.push_back(g);
    for (Tensor* g : expand3x3_.gradients()) result.push_back(g);
    return result;
}

SqueezeNet::SqueezeNet(size_t num_classes, size_t H_in, size_t W_in)
    : conv1_(3, 96, 7, 7, H_in, W_in, 2, 2, 3, 3),
      pool1_(3, 3, 2, 2),
      fire_modules_(),
      pool_final_(3, 3, 2, 2),
      conv10_(96, num_classes, 1, 1, 7, 7, 1, 1, 0, 0),
      flatten_(),
      last_output_(1, num_classes) {

    // After conv1 + pool: H=55, W=55, C=96
    size_t H = (H_in - 7) / 2 + 1; // ~55
    size_t W = (W_in - 7) / 2 + 1;

    // Fire modules: fire1 (96->16->64+64, H=55), fire2 (128->16->64+64)
    // After pool1, H=27, W=27, C=96
    H = (H - 3) / 2 + 1;
    W = (W - 3) / 2 + 1;
    size_t ch = 96;
    for (size_t i = 0; i < 8; ++i) { // 8 fire modules
        size_t squeeze = std::max((size_t)16, ch / 4);
        fire_modules_.emplace_back(ch, squeeze, 64, 64, H, W);
        ch = 64 + 64; // 128 channels after concat
        if (i == 1 || i == 3 || i == 5 || i == 7) {
            // After pool: H,W halved
            H = (H - 3) / 2 + 1;
            W = (W - 3) / 2 + 1;
        }
    }
}

Tensor SqueezeNet::forward(const Tensor& input) {
    // conv1 -> ReLU -> pool1
    Tensor x = conv1_.forward(input);
    // Apply ReLU manually
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = pool1_.forward(x);

    // Fire modules
    for (auto& fm : fire_modules_) {
        x = fm.forward(x);
    }

    x = pool_final_.forward(x);
    x = conv10_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);
    x = flatten_.forward(x);

    last_output_ = x;
    return last_output_;
}

Tensor SqueezeNet::backward(const Tensor& grad_output, double learning_rate) {
    // Backprop: conv10 → pool_final → fire_modules (reverse) → pool1 → conv1
    // forward: conv1 → ReLU → pool1 → fire0 → fire1 → pool → fire2 → ... → fire7 → pool_final → conv10 → ReLU
    // conv10 output is (batch, num_classes) directly, no flatten in SqueezeNet::forward
    // flatten_ was called but its output wasn't stored or used in last_output_
    // So grad_output is already (batch, num_classes) from conv10

    // Backprop conv10: (batch, num_classes) → (batch, conv10_in_ch * conv10_spatial)
    Tensor grad = conv10_.backward(grad_output, learning_rate);

    // ReLU gradient after conv10 (approximate - no cache)

    // Backprop pool_final (MaxPool2D - not a sub-layer, gradient passes through)

    // Backprop through fire modules in reverse
    for (auto it = fire_modules_.rbegin(); it != fire_modules_.rend(); ++it)
        grad = it->backward(grad, learning_rate);

    // Backprop pool1 (MaxPool2D - not a sub-layer)

    // ReLU after conv1 (approximate - no cache)

    // Backprop conv1
    grad = conv1_.backward(grad, learning_rate);
    return grad;
}

void SqueezeNet::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv10_.update_weights(learning_rate);
    for (auto& fm : fire_modules_)
        fm.update_weights(learning_rate);
}

void SqueezeNet::zero_grad() {
    conv1_.zero_grad();
    conv10_.zero_grad();
    for (auto& fm : fire_modules_)
        fm.zero_grad();
}

std::vector<Tensor*> SqueezeNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv10_.parameters()) result.push_back(p);
    for (auto& fm : fire_modules_)
        for (Tensor* p : fm.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> SqueezeNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv10_.gradients()) result.push_back(g);
    for (auto& fm : fire_modules_)
        for (Tensor* g : fm.gradients())
            result.push_back(g);
    return result;
}