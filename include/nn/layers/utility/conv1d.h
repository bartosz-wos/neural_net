#ifndef CONV1D_H
#define CONV1D_H

#include "../../core/layer.h"

// 1D convolution for 1D signal / sequence data.
// Input: (batch, in_channels * seq_len) — col-major as channels × time
// Output: (batch, out_channels * seq_out)
// weight shape: (out_channels, in_channels * kernel_size)
class Conv1D : public Layer {
public:
    int in_channels, out_channels;
    int kernel_size;
    int stride;
    int pad;
    int seq_len;

    Tensor weights;   // (out_channels, in_channels * kernel_size)
    Tensor bias;      // (out_channels, 1)
    Tensor grad_weights;
    Tensor grad_bias;

    Tensor last_input; // (batch, in_channels * seq_len)
    Tensor col;       // (in_channels * kernel_size, batch * seq_out)

    Conv1D(int in_ch, int out_ch, int kernel_size, int seq_len,
           int stride = 1, int pad = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }

private:
    int seq_out; // computed in forward
};

#endif
