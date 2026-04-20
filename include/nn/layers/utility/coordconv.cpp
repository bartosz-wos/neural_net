#include "coordconv.h"
#include <cmath>

CoordConv2D::CoordConv2D(size_t in_channels, size_t out_channels,
                          size_t kH, size_t kW,
                          size_t H_in, size_t W_in,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w)
    : in_channels_(in_channels), H_in_(H_in), W_in_(W_in),
      conv_(in_channels + 2, out_channels, kH, kW, H_in, W_in,
            stride_h, stride_w, pad_h, pad_w),
      x_coord_(H_in, W_in), y_coord_(H_in, W_in),
      last_output_(1, out_channels) {

    // Precompute normalized coordinate maps
    for (size_t h = 0; h < H_in_; ++h) {
        for (size_t w = 0; w < W_in_; ++w) {
            x_coord_[h][w] = (W_in_ > 1) ? (2.0 * w / (W_in_ - 1) - 1.0) : 0.0;
            y_coord_[h][w] = (H_in_ > 1) ? (2.0 * h / (H_in_ - 1) - 1.0) : 0.0;
        }
    }
}

Tensor CoordConv2D::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t spatial = H_in_ * W_in_;

    // Build coord-augmented tensor: insert x,y channels before each spatial position
    Tensor aug_input(batch, (in_channels_ + 2) * spatial);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < spatial; ++s) {
            size_t h = s / W_in_;
            size_t w = s % W_in_;
            size_t in_off = s * in_channels_;
            size_t out_off = s * (in_channels_ + 2);

            // x channel
            aug_input[b][out_off] = x_coord_[h][w];
            // y channel
            aug_input[b][out_off + 1] = y_coord_[h][w];
            // original channels
            for (size_t c = 0; c < in_channels_; ++c)
                aug_input[b][out_off + 2 + c] = input[b][in_off + c];
        }
    }

    last_output_ = conv_.forward(aug_input);
    return last_output_;
}

Tensor CoordConv2D::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void CoordConv2D::update_weights(double learning_rate) {
    conv_.update_weights(learning_rate);
}

void CoordConv2D::zero_grad() { conv_.zero_grad(); }

std::vector<Tensor*> CoordConv2D::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> CoordConv2D::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv_.gradients()) result.push_back(g);
    return result;
}