#ifndef DENSENET_H
#define DENSENET_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include "../pooling/pool_layer.h"

class DenseBlock : public Layer {
public:
    DenseBlock(size_t in_channels, size_t growth_rate, size_t num_layers);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    size_t growth_rate_;
    std::vector<Dense> fc_layers_;
    size_t num_layers_;
    std::vector<Tensor> concat_buffers_;
    Tensor last_output_;
};

class TransitionLayer : public Layer {
public:
    TransitionLayer(size_t in_channels, size_t out_channels, size_t H, size_t W);
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
    size_t out_channels_;
    Tensor last_output_;
    size_t H_, W_;
};

class DenseNet : public Layer {
public:
    DenseNet(size_t initial_channels, size_t growth_rate,
             const std::vector<size_t>& layers_per_block,
             size_t num_classes, size_t H_in = 32, size_t W_in = 32);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D stem_;
    std::vector<DenseBlock> blocks_;
    std::vector<TransitionLayer> transitions_;
    Dense fc_;
    Tensor last_output_;
};

#endif