#ifndef CNN_MODELS_VGG_H
#define CNN_MODELS_VGG_H

#include "../core/layer.h"
#include "../core/model.h"
#include "convolutions/conv_layer.h"
#include "pooling/pool_layer.h"
#include "dense/flatten.h"
#include "../activations/activations.h"

// VGG Block: sequence of n conv layers with same channels, followed by pool
class VGGBlock : public Layer {
public:
    VGGBlock(int in_channels, int out_channels, int num_convs, int pool_size = 2);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "VGGBlock"; }

private:
    std::vector<Conv2D> convs_;
    int num_convs_;
};

// VGG-11/13/16/19
class VGG11 : public Layer {
public:
    VGG11(int num_classes = 10, int in_channels = 3);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "VGG11"; }

    Model& model() { return model_; }

private:
    Model model_;
};

class VGG16 : public Layer {
public:
    VGG16(int num_classes = 10, int in_channels = 3);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "VGG16"; }

    Model& model() { return model_; }

private:
    Model model_;
};

// ResNeXt block with grouped convolutions
// Paper: https://arxiv.org/abs/1611.05431
class ResNeXtBlock : public Layer {
public:
    ResNeXtBlock(int in_channels, int bottleneck_width, int cardinality = 32, int stride = 1);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "ResNeXtBlock"; }

private:
    int in_channels_;
    int out_channels_;
    int cardinality_;
    int bottleneck_width_;
    int stride_;
    std::vector<Conv2D> group_convs_;
};

class ResNeXt29 : public Layer {
public:
    ResNeXt29(int num_classes = 10, int in_channels = 3, int cardinality = 32, int bottleneck_width = 4);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "ResNeXt29"; }

    Model& model() { return model_; }

private:
    Model model_;
};

#endif
