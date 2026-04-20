#include "cnn_models_vgg.h"
#include <cmath>
#include <algorithm>

VGGBlock::VGGBlock(int in_channels, int out_channels, int num_convs, int pool_size)
    : num_convs_(num_convs) {
    int ch = in_channels;
    for (int i = 0; i < num_convs; i++) {
        convs_.emplace_back(ch, out_channels, 3, 1, 1, 0);
        ch = out_channels;
    }
}

Tensor VGGBlock::forward(const Tensor& x) {
    Tensor cur = x;
    for (int i = 0; i < num_convs_; i++) {
        cur = convs_[i].forward(cur);
        for (size_t r = 0; r < cur.rows; ++r)
        for (size_t c = 0; c < cur.cols; ++c)
            cur[r][c] = std::max(0.0, cur[r][c]);
    }
    return cur;
}

Tensor VGGBlock::backward(const Tensor& grad_output, double lr) {
    (void)grad_output; (void)lr;
    return Tensor(grad_output.rows, grad_output.cols);
}

VGG11::VGG11(int num_classes, int in_channels) {
    model_.add_layer(new Conv2D(in_channels, 64, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(64, 128, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(128, 256, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(256, 256, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(256, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Flatten());
    model_.add_layer(new Dense(512, 4096));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Dense(4096, 4096));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Dense(4096, num_classes));
}

Tensor VGG11::forward(const Tensor& x) { return model_.forward(x); }
Tensor VGG11::backward(const Tensor& grad, double lr) { return model_.backward(grad, lr); }

VGG16::VGG16(int num_classes, int in_channels) {
    model_.add_layer(new Conv2D(in_channels, 64, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(64, 64, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(64, 128, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(128, 128, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(128, 256, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(256, 256, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(256, 256, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(256, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Conv2D(512, 512, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Flatten());
    model_.add_layer(new Dense(512, 4096));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Dense(4096, 4096));
    model_.add_layer(new Activation<ReLU>(ReLU{}));
    model_.add_layer(new Dense(4096, num_classes));
}

Tensor VGG16::forward(const Tensor& x) { return model_.forward(x); }
Tensor VGG16::backward(const Tensor& grad, double lr) { return model_.backward(grad, lr); }

ResNeXtBlock::ResNeXtBlock(int in_channels, int bottleneck_width, int cardinality, int stride)
    : in_channels_(in_channels), cardinality_(cardinality), bottleneck_width_(bottleneck_width), stride_(stride) {
    int inner_channels = bottleneck_width_ * cardinality_;
    int out_ch = inner_channels * 2;
    bneck_ = Conv2D(in_channels_, inner_channels, 1, 1, 1, 0);
    gconv_ = Conv2D(inner_channels, inner_channels, 3, 1, 1, 0);
    final_ = Conv2D(inner_channels, out_ch, 1, 1, 1, 0);
}

Tensor ResNeXtBlock::forward(const Tensor& x) {
    int inner_channels = bottleneck_width_ * cardinality_;
    int out_ch = inner_channels * 2;

    Tensor out = bneck_.forward(x);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            out[i][j] = std::max(0.0, out[i][j]);

    out = gconv_.forward(out);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            out[i][j] = std::max(0.0, out[i][j]);

    out = final_.forward(out);

    if (stride_ == 1 && in_channels_ == out_ch) {
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                out[i][j] += x[i][j];
    }
    return out;
}

Tensor ResNeXtBlock::backward(const Tensor& grad_output, double lr) {
    Tensor grad = grad_output;
    grad = final_.backward(grad, lr);
    // ReLU backward on gconv output
    for (size_t i = 0; i < grad.rows; ++i)
        for (size_t j = 0; j < grad.cols; ++j)
            grad[i][j] = grad[i][j] > 0 ? grad[i][j] : 0.0;
    grad = gconv_.backward(grad, lr);
    // ReLU backward on bneck output
    for (size_t i = 0; i < grad.rows; ++i)
        for (size_t j = 0; j < grad.cols; ++j)
            grad[i][j] = grad[i][j] > 0 ? grad[i][j] : 0.0;
    grad = bneck_.backward(grad, lr);
    // Residual gradient: identity contribution
    if (stride_ == 1 && in_channels_ == grad_output.cols / (bneck_.get_weights().rows)) {
        for (size_t i = 0; i < grad.rows; ++i)
            for (size_t j = 0; j < grad.cols; ++j)
                grad[i][j] += grad_output[i][j];
    }
    return grad;
}

void ResNeXtBlock::update_weights(double lr) {
    bneck_.update_weights(lr);
    gconv_.update_weights(lr);
    final_.update_weights(lr);
}

void ResNeXtBlock::zero_grad() {
    bneck_.zero_grad();
    gconv_.zero_grad();
    final_.zero_grad();
}

std::vector<Tensor*> ResNeXtBlock::parameters() {
    std::vector<Tensor*> r;
    for (Tensor* p : bneck_.parameters()) r.push_back(p);
    for (Tensor* p : gconv_.parameters()) r.push_back(p);
    for (Tensor* p : final_.parameters()) r.push_back(p);
    return r;
}

std::vector<Tensor*> ResNeXtBlock::gradients() {
    std::vector<Tensor*> r;
    for (Tensor* g : bneck_.gradients()) r.push_back(g);
    for (Tensor* g : gconv_.gradients()) r.push_back(g);
    for (Tensor* g : final_.gradients()) r.push_back(g);
    return r;
}

ResNeXt29::ResNeXt29(int num_classes, int in_channels, int cardinality, int bottleneck_width) {
    model_.add_layer(new Conv2D(in_channels, 64, 3, 1, 1, 0));
    model_.add_layer(new Activation<ReLU>(ReLU{}));

    model_.add_layer(new ResNeXtBlock(64, bottleneck_width, cardinality, 1));
    model_.add_layer(new ResNeXtBlock(128, bottleneck_width, cardinality, 1));
    model_.add_layer(new ResNeXtBlock(128, bottleneck_width, cardinality, 2));
    model_.add_layer(new ResNeXtBlock(256, bottleneck_width, cardinality, 1));
    model_.add_layer(new ResNeXtBlock(512, bottleneck_width, cardinality, 1));

    model_.add_layer(new Flatten());
    model_.add_layer(new Dense(512, num_classes));
}

Tensor ResNeXt29::forward(const Tensor& x) { return model_.forward(x); }
Tensor ResNeXt29::backward(const Tensor& grad, double lr) { return model_.backward(grad, lr); }
