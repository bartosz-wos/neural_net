#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../../core/layer.h"

class Conv2D : public Layer {
public:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int H, W; // input spatial dimensions

    // weight shape: (out_channels, in_channels * kernel_h * kernel_w)
    Tensor weights;
    Tensor bias;     // (out_channels, 1)
    Tensor grad_weights;
    Tensor grad_bias;

    // Cached for backward
    Tensor last_input; // stored as (N, in_channels*H*W)
    Tensor col;        // im2col matrix: (in_channels*kernel_h*kernel_w, N*H_out*W_out)

    Conv2D();  // default constructor (for use as class member)
    Conv2D(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
           int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0,
           int dilation_h = 1, int dilation_w = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }

protected:
    int H_out, W_out; // computed in forward, stored for backward
    // Helper: im2col (supports dilation)
    static Tensor im2col(const Tensor& input, int N, int C, int H, int W,
                         int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int dilation_h, int dilation_w,
                         int& H_out, int& W_out);
    // Helper: col2im (accumulate into provided tensor, supports dilation)
    static void col2im(Tensor& grad_input, const Tensor& grad_col,
                       int N, int C, int H, int W,
                       int kH, int kW, int stride_h, int stride_w,
                       int pad_h, int pad_w, int dilation_h, int dilation_w);
};

#endif
