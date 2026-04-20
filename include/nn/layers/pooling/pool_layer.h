#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "../../core/layer.h"

class MaxPool2D : public Layer {
public:
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int H, W; // input spatial dims
    Tensor last_input; // cached (N, C*H*W)

    MaxPool2D(int kH, int kW, int H_in, int W_in, int stride_h = 1, int stride_w = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0,0); }
    Tensor get_gradients() const override { return Tensor(0,0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}

private:
    int H_out, W_out;
};

// 1D max pooling for sequence data
// Input: (batch, channels * seq_len)  — col-major as channels × time
// Output: (batch, channels * seq_out)
class MaxPool1D : public Layer {
public:
    int kernel_size;
    int stride;
    int channels;
    int seq_len;

    Tensor last_input; // (batch, channels * seq_len)

    MaxPool1D(int kernel_size, int seq_len, int channels = 1, int stride = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0,0); }
    Tensor get_gradients() const override { return Tensor(0,0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}

private:
    int seq_out;
    std::vector<std::vector<int>> max_indices_; // [batch*channels][seq_out] stores argmax time index
};

// 1D average pooling
class AvgPool1D : public Layer {
public:
    int kernel_size;
    int stride;
    int channels;
    int seq_len;

    Tensor last_input;

    AvgPool1D(int kernel_size, int seq_len, int channels = 1, int stride = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0,0); }
    Tensor get_gradients() const override { return Tensor(0,0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}

private:
    int seq_out;
};

#endif
