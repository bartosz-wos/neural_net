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
    // Backward through pool (if VGGBlock owns one — note: vgg.cpp VGGBlock doesn't use pool)
    // VGG11/VGG16 delegate to model_.backward(), so this is only for standalone VGGBlock usage.
    // For the composite VGG11/VGG16, gradients flow through model_.backward() correctly.
    // Standalone VGGBlock: backprop through convs in reverse, then ReLU.
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
    : in_channels_(in_channels), cardinality_(cardinality), bottleneck_width_(bottleneck_width), stride_(stride) {}

Tensor ResNeXtBlock::forward(const Tensor& x) {
    int inner_channels = bottleneck_width_ * cardinality_;
    int out_ch = inner_channels * 2;

    Conv2D bneck(in_channels_, inner_channels, 1, 1, 1, 0);
    Tensor out = bneck.forward(x);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            out[i][j] = std::max(0.0, out[i][j]);

    Conv2D gconv(inner_channels, inner_channels, 3, 1, 1, 0);
    out = gconv.forward(out);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            out[i][j] = std::max(0.0, out[i][j]);

    Conv2D final(in_channels_, out_ch, 1, 1, 1, 0);
    out = final.forward(out);

    if (stride_ == 1 && in_channels_ == out_ch) {
        out = out + x;
    }
    return out;
}

Tensor ResNeXtBlock::backward(const Tensor& grad_output, double) {
    return grad_output;
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
