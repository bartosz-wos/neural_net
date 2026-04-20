#ifndef COORDCONV_H
#define COORDCONV_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"

// CoordConv: appends (x, y) coordinate channels to input feature maps.
// For an H×W input, two channels are added: x and y normalized to [-1, 1].
// Convolution then operates on the augmented channel dimension.
class CoordConv2D : public Layer {
public:
    CoordConv2D(size_t in_channels, size_t out_channels,
                 size_t kH, size_t kW,
                 size_t H_in, size_t W_in,
                 size_t stride_h = 1, size_t stride_w = 1,
                 size_t pad_h = 0, size_t pad_w = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D conv_;
    size_t in_channels_, H_in_, W_in_;
    Tensor x_coord_, y_coord_; // precomputed coordinate maps
    Tensor last_output_;
};

#endif