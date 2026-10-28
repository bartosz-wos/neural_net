#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include "../pooling/pool_layer.h"
#include "../dense/flatten.h"

// SqueezeNet: Fire module — squeeze 1x1 conv followed by expand 1x1 + 3x3 conv.
// Very parameter-efficient, similar accuracy to AlexNet with 50x fewer params.
class FireModule : public Layer {
public:
    // in_channels: input feature maps
    // squeeze_ch: output channels of squeeze 1x1 conv
    // expand1_ch: output channels of expand 1x1 conv
    // expand3_ch: output channels of expand 3x3 conv (spatial)
    FireModule(size_t in_channels, size_t squeeze_ch,
               size_t expand1_ch, size_t expand3_ch,
               size_t H_in, size_t W_in);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    Conv2D squeeze_;     // 1x1 conv
    Conv2D expand1x1_;    // 1x1 expand
    Conv2D expand3x3_;     // 3x3 expand (padded to same spatial size)
    size_t expand_ch_;     // expand1 + expand3 total channels
    size_t H_out_, W_out_;
    Tensor last_output_;
};

class SqueezeNet : public Layer {
public:
    SqueezeNet(size_t num_classes = 1000, size_t H_in = 224, size_t W_in = 224);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

private:
    // -------------------------------------------------------------------------
    // THE SPATIAL CHAIN, AND WHY IT IS COMPUTED BY STATIC HELPERS.
    //
    // Every layer here is SHAPE-STATIC: `Conv2D` and `MaxPool2D` bake their
    // spatial dims in at construction. The constructor therefore has to know
    // the spatial size of every stage BEFORE it can size the layers, and the
    // member-initialiser list runs BEFORE the constructor body -- so the sizes
    // must come from functions callable in the initialiser list. These three
    // static helpers ARE the chain, and they are the single source of truth:
    // the constructor body uses them too, so the layers and the forward pass
    // cannot drift apart.
    //
    //   conv1  (7x7, stride 2, pad 3)  on H_in      -> conv1_out
    //   pool1  (3x3, stride 2)                       -> pool_out
    //   fire0, fire1, POOL, fire2, fire3, POOL, ...   -> 4 intermediate pools
    //   conv10 (1x1)                                 -> final spatial
    //   global average pool                          -> (batch, num_classes)
    // -------------------------------------------------------------------------
    static size_t conv1_out(size_t h) { return (h + 6 - 7) / 2 + 1; }
    static size_t pool_out(size_t h) { return (h - 3) / 2 + 1; }
    // The spatial size after fire8's pool -- the size conv10 and the global
    // average pool are built for.
    static size_t final_spatial(size_t h) {
        size_t s = pool_out(conv1_out(h));
        for (size_t i = 0; i < 8; ++i)
            if (i == 1 || i == 3 || i == 5 || i == 7) s = pool_out(s);
        return s;
    }
    // True when a pool follows fire module `i` (0-indexed).
    static bool pool_follows(size_t i) { return i == 1 || i == 3 || i == 5 || i == 7; }

    Conv2D conv1_;  // initial conv
    MaxPool2D pool1_;
    std::vector<FireModule> fire_modules_;
    // The four pools that follow fire modules 2, 4, 6 and 8. The PREVIOUS
    // constructor tracked a halved spatial size for these pools but the
    // forward pass never performed them, so the two halves of the class
    // disagreed about the shape of the network and the class could not even
    // be constructed.
    std::vector<MaxPool2D> pool_mid_;
    Conv2D conv10_; // final 1x1 conv to num_classes
    AvgPool2D global_pool_;  // global average pool -> (batch, num_classes)
    Tensor last_output_;
};

#endif