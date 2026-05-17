#ifndef CONV1D_TRANSPOSE_H
#define CONV1D_TRANSPOSE_H

#include "../../core/layer.h"

// =============================================================================
// TransposedConv1D — transposed convolution (aka deconv, fractionally strided conv)
// Used for upsampling: maps (batch, in_channels * seq_in) -> (batch, out_channels * seq_out)
// where seq_out = (seq_in - 1) * stride - 2 * pad + kernel_size + output_pad
// =============================================================================
class TransposedConv1D : public Layer {
public:
    int in_channels, out_channels;
    int kernel_size;
    int stride;
    int pad;
    int output_pad;
    int seq_in; // input sequence length

    Tensor weights;    // (in_channels, out_channels * kernel_size) — note: swapped layout vs Conv1D
    Tensor bias;       // (out_channels, 1)
    Tensor grad_weights;
    Tensor grad_bias;

    Tensor last_input; // (batch, in_channels * seq_in)

    TransposedConv1D() = default;
    TransposedConv1D(int in_ch, int out_ch, int kernel_size, int seq_in,
                     int stride = 2, int pad = 1, int output_pad = 0);

    // seq_out computed as: (seq_in - 1) * stride - 2 * pad + kernel_size + output_pad
    int seq_out() const;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }
};

// =============================================================================
// Upsample1D — nearest-neighbor 2x upsampling for 1D sequences
// Input:  (batch, channels * seq_in)
// Output: (batch, channels * (seq_in * 2))
// =============================================================================
class Upsample1D : public Layer {
public:
    explicit Upsample1D(int channels, int seq_in);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "Upsample1D"; }

private:
    int channels_;
    int seq_in_;
    int seq_out_;
    Tensor last_input_;
};

#endif // CONV1D_TRANSPOSE_H