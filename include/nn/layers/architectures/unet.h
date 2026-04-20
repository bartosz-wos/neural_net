#ifndef UNET_H
#define UNET_H

#include "../../core/layer.h"
#include "../../core/model.h"
#include "../convolutions/conv_layer.h"
#include "../pooling/pool_layer.h"
#include "../dense/flatten.h"
#include "../../activations/activations.h"

// U-Net: encoder-decoder with skip connections for image segmentation
// Paper: https://arxiv.org/abs/1505.04597
// Model-based implementation
class UNet : public Layer {
public:
    UNet(int in_channels = 3, int num_classes = 1, int depth = 4, int base_channels = 64);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
    std::string name() const override { return "UNet"; }

    Model& model() { return model_; }

private:
    Model model_;
};

#endif
