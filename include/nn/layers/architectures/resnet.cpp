#include "resnet.h"
#include <cmath>

ResBlock::ResBlock(size_t in_channels, size_t out_channels, size_t kernel_size,
                   size_t stride, size_t H_in, size_t W_in)
    : conv1_(in_channels, out_channels, kernel_size, kernel_size, H_in, W_in,
             stride, stride, 0, 0),
      conv2_(out_channels, out_channels, kernel_size, kernel_size,
             (H_in - kernel_size) / stride + 1, (W_in - kernel_size) / stride + 1,
             1, 1, 0, 0),
      needs_projection_(in_channels != out_channels || stride != 1),
      out_channels_(out_channels),
      H_out_((H_in - kernel_size) / stride + 1),
      W_out_((W_in - kernel_size) / stride + 1),
      last_output_(1, out_channels * H_out_ * W_out_) {}

Tensor ResBlock::forward(const Tensor& input) {
    last_input_ = input;
    last_output_ = conv1_.forward(input);
    last_output_ = conv2_.forward(last_output_);

    // Element-wise residual add
    size_t batch = input.rows;
    size_t out_cols = last_output_.cols;
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            last_output_[i][j] += input[i][j];
    return last_output_;
}

Tensor ResBlock::backward(const Tensor& grad_output, double learning_rate) {
    // Backprop through conv2, then conv1
    Tensor grad = conv2_.backward(grad_output, learning_rate);
    grad = conv1_.backward(grad, learning_rate);
    // Residual gradient: dL/dx += grad_output * 1 (identity = 1)
    // Only add if shapes match (identity path was used)
    if (grad.rows == grad_output.rows && grad.cols == grad_output.cols) {
        for (size_t i = 0; i < grad.rows; ++i)
            for (size_t j = 0; j < grad.cols; ++j)
                grad[i][j] += grad_output[i][j];
    }
    return grad;
}

void ResBlock::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv2_.update_weights(learning_rate);
}

void ResBlock::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
}

std::vector<Tensor*> ResBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> ResBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.gradients()) result.push_back(p);
    for (Tensor* p : conv2_.gradients()) result.push_back(p);
    return result;
}

// === ResNet ===

ResNet::ResNet(size_t input_channels, size_t num_classes, size_t H_in, size_t W_in,
               const std::vector<size_t>& channels_per_stage,
               const std::vector<size_t>& strides_per_stage,
               size_t blocks_per_stage)
    : stem_(input_channels, channels_per_stage[0], 7, 7, H_in, W_in, 2, 2, 3, 3),
      fc_(channels_per_stage.back(), num_classes),
      last_output_(1, num_classes) {

    size_t H = (H_in - 7) / 2 + 1;
    size_t W = (W_in - 7) / 2 + 1;
    size_t in_ch = channels_per_stage[0];

    for (size_t stage = 0; stage < channels_per_stage.size(); ++stage) {
        size_t out_ch = channels_per_stage[stage];
        size_t stride = strides_per_stage[stage];

        stages_.emplace_back(in_ch, out_ch, 3, stride, H, W);
        H = (H - 3) / stride + 1;
        W = (W - 3) / stride + 1;

        for (size_t b = 1; b < blocks_per_stage; ++b) {
            stages_.emplace_back(out_ch, out_ch, 3, 1, H, W);
        }
        in_ch = out_ch;
    }
}

Tensor ResNet::forward(const Tensor& input) {
    Tensor x = stem_.forward(input);
    for (auto& stage : stages_)
        x = stage.forward(x);
    // Flatten + fc
    size_t batch = x.rows;
    Tensor flat(batch, x.cols);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            flat[i][j] = x[i][j];
    last_output_ = fc_.forward(flat);
    return last_output_;
}

Tensor ResNet::backward(const Tensor& grad_output, double learning_rate) {
    // Backprop through fc layer
    Tensor grad_fc = fc_.backward(grad_output, learning_rate);
    // Unflatten gradient to stage output shape
    size_t batch = grad_fc.rows;
    size_t stage_cols = stages_.back().out_channels()
                        * stages_.back().H_out() * stages_.back().W_out();
    Tensor grad_stage(batch, stage_cols);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < stage_cols; ++j)
            grad_stage[i][j] = grad_fc[i][j];
    // Backprop through stages in reverse order
    Tensor grad = grad_stage;
    for (auto it = stages_.rbegin(); it != stages_.rend(); ++it)
        grad = it->backward(grad, learning_rate);
    // Backprop through stem
    grad = stem_.backward(grad, learning_rate);
    return grad;
}

void ResNet::update_weights(double learning_rate) {
    stem_.update_weights(learning_rate);
    for (auto& stage : stages_) stage.update_weights(learning_rate);
    fc_.update_weights(learning_rate);
}

void ResNet::zero_grad() {
    stem_.zero_grad();
    for (auto& stage : stages_) stage.zero_grad();
    fc_.zero_grad();
}

std::vector<Tensor*> ResNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : stem_.parameters()) result.push_back(p);
    for (auto& stage : stages_)
        for (Tensor* p : stage.parameters()) result.push_back(p);
    for (Tensor* p : fc_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> ResNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* p : stem_.gradients()) result.push_back(p);
    for (auto& stage : stages_)
        for (Tensor* p : stage.gradients()) result.push_back(p);
    for (Tensor* p : fc_.gradients()) result.push_back(p);
    return result;
}