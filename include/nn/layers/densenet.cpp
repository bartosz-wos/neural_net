#include "densenet.h"
#include <cmath>

DenseBlock::DenseBlock(size_t in_channels, size_t growth_rate, size_t num_layers)
    : growth_rate_(growth_rate), num_layers_(num_layers) {

    for (size_t i = 0; i < num_layers_; ++i) {
        fc_layers_.emplace_back(in_channels, growth_rate_);  // Fixed: Dense(in_channels, growth_rate)
        concat_buffers_.emplace_back(1, 1);
    }
    last_output_ = Tensor(1, in_channels);
}

Tensor DenseBlock::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t in_ch = input.cols;

    last_output_ = input;
    concat_buffers_[0] = input;

    size_t current_ch = in_ch;
    for (size_t l = 0; l < num_layers_; ++l) {
        // Extract current_ch columns as input to FC layer (all batch rows)
        Tensor fc_input(batch, current_ch);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < current_ch; ++j)
                fc_input[b][j] = last_output_[b][j];
        Tensor out = fc_layers_[l].forward(fc_input);
        for (size_t b = 0; b < out.rows; ++b)
            for (size_t j = 0; j < out.cols; ++j)
                out[b][j] = std::max(0.0, out[b][j]);

        Tensor concatenated(batch, current_ch + growth_rate_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < current_ch; ++j)
                concatenated[b][j] = last_output_[b][j];
            for (size_t j = 0; j < growth_rate_; ++j)
                concatenated[b][current_ch + j] = out[b][j];
        }
        last_output_ = concatenated;
        current_ch += growth_rate_;
        concat_buffers_[l] = out;
    }
    return last_output_;
}

Tensor DenseBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void DenseBlock::update_weights(double learning_rate) {
    for (auto& fc : fc_layers_)
        fc.update_weights(learning_rate);
}

void DenseBlock::zero_grad() {
    for (auto& fc : fc_layers_)
        fc.zero_grad();
}

std::vector<Tensor*> DenseBlock::parameters() {
    std::vector<Tensor*> result;
    for (auto& fc : fc_layers_)
        for (Tensor* p : fc.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> DenseBlock::gradients() {
    std::vector<Tensor*> result;
    for (auto& fc : fc_layers_)
        for (Tensor* g : fc.gradients())
            result.push_back(g);
    return result;
}

TransitionLayer::TransitionLayer(size_t in_channels, size_t out_channels, size_t H, size_t W)
    : conv_(in_channels, out_channels, 1, 1, H, W, 1, 1, 0, 0),
      out_channels_(out_channels), H_(H), W_(W) {}

Tensor TransitionLayer::forward(const Tensor& input) {
    last_output_ = conv_.forward(input);
    for (size_t i = 0; i < last_output_.rows; ++i)
        for (size_t j = 0; j < last_output_.cols; ++j)
            last_output_[i][j] = std::max(0.0, last_output_[i][j]);
    // Downsample via strided view — simplify by just returning conv output
    // (pool would need spatial dimensions which we flatten)
    return last_output_;
}

Tensor TransitionLayer::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void TransitionLayer::update_weights(double learning_rate) {
    conv_.update_weights(learning_rate);
}

void TransitionLayer::zero_grad() { conv_.zero_grad(); }

std::vector<Tensor*> TransitionLayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv_.parameters())
        result.push_back(p);
    return result;
}

std::vector<Tensor*> TransitionLayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv_.gradients())
        result.push_back(g);
    return result;
}

DenseNet::DenseNet(size_t initial_channels, size_t growth_rate,
                   const std::vector<size_t>& layers_per_block,
                   size_t num_classes, size_t H_in, size_t W_in)
    : stem_(3, initial_channels, 3, 3, H_in, W_in, 2, 2, 1, 1),
      fc_(1, num_classes),
      last_output_(1, num_classes) {

    size_t H = (H_in - 3) / 2 + 1;
    size_t W = (W_in - 3) / 2 + 1;
    size_t channels = initial_channels;

    for (size_t b = 0; b < layers_per_block.size(); ++b) {
        blocks_.emplace_back(channels, growth_rate, layers_per_block[b]);
        channels += growth_rate * layers_per_block[b];

        if (b < layers_per_block.size() - 1) {
            size_t trans_out = channels / 2;
            transitions_.emplace_back(channels, trans_out, H, W);
            channels = trans_out;
        }
    }
}

Tensor DenseNet::forward(const Tensor& input) {
    Tensor x = stem_.forward(input);
    size_t batch = x.rows;
    size_t ch = x.cols;

    for (size_t b = 0; b < blocks_.size(); ++b) {
        x = blocks_[b].forward(x);
        if (b < transitions_.size())
            x = transitions_[b].forward(x);
    }

    last_output_ = fc_.forward(x);
    return last_output_;
}

Tensor DenseNet::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void DenseNet::update_weights(double learning_rate) {
    stem_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk.update_weights(learning_rate);
    for (auto& t : transitions_) t.update_weights(learning_rate);
    fc_.update_weights(learning_rate);
}

void DenseNet::zero_grad() {
    stem_.zero_grad();
    for (auto& blk : blocks_) blk.zero_grad();
    for (auto& t : transitions_) t.zero_grad();
    fc_.zero_grad();
}

std::vector<Tensor*> DenseNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : stem_.parameters()) result.push_back(p);
    for (auto& blk : blocks_)
        for (Tensor* p : blk.parameters()) result.push_back(p);
    for (auto& t : transitions_)
        for (Tensor* p : t.parameters()) result.push_back(p);
    for (Tensor* p : fc_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> DenseNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : stem_.gradients()) result.push_back(g);
    for (auto& blk : blocks_)
        for (Tensor* g : blk.gradients()) result.push_back(g);
    for (auto& t : transitions_)
        for (Tensor* g : t.gradients()) result.push_back(g);
    for (Tensor* g : fc_.gradients()) result.push_back(g);
    return result;
}