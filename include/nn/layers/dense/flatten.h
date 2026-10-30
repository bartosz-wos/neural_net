#ifndef FLATTEN_H
#define FLATTEN_H

#include "../../core/layer.h"

// ===========================================================================
// Flatten -- AND IT IS A NO-OP. READ THIS BEFORE USING IT.
//
// ---------------------------------------------------------------------------
// THE CONTRACT, STATED PLAINLY BECAUSE THE HEADER USED TO MISSTATE IT
// ---------------------------------------------------------------------------
//
//     THIS LAYER DOES NOT FLATTEN ANYTHING.
//
// `Flatten::forward` returns its input UNCHANGED, and `Flatten::backward`
// returns its gradient unchanged. That is the whole implementation
// (include/nn/layers/dense/flatten.cpp):
//
//     Tensor Flatten::forward(const Tensor& input) {
//         // Just return as-is, the data format already flattens spatial dims
//         return input;
//     }
//
// **THE SHAPE IS UNCHANGED.** A (batch, C*H*W) input comes out as
// (batch, C*H*W). A (batch, C, H, W) input -- which this repo's `Tensor`
// cannot even represent, since it is two-dimensional -- is not collapsed
// either, because there is nothing to collapse it with.
//
// **THE OLD HEADER WAS WRONG.** It said "Flattens spatial dimensions into a
// single channel", which is not what the class does and never was. If you
// are reading this for the first time, that sentence was a trap: a layer
// that "flattens" but does not will produce a wrong shape SILENTLY, and
// only the next `Dense` will notice -- because a `Dense` whose input width
// does not match its weight matrix either throws loudly or, worse, reads a
// wrong slice of the buffer.
//
// ---------------------------------------------------------------------------
// WHY IT IS A NO-OP, AND WHY THAT IS CORRECT HERE
// ---------------------------------------------------------------------------
//
// **IN THIS REPO EVERY LAYER'S FORWARD OUTPUT IS ALREADY FLAT**, i.e.
// (N, C*H*W). There is no spatial map left to collapse, so the layer has
// nothing to do. The class exists because the CANONICAL CNN ARCHITECTURES
// have a "flatten" stage in their diagrams, and the repo's `LeNet5`,
// `AlexNet` and `VGG` all instantiate one (see `layers/architectures/
// cnn_models.h` and `cnn_models_vgg.cpp`).
//
// There, it is CORRECT, because the tensor feeding it is already flat.
//
// ---------------------------------------------------------------------------
// THE TRAP THIS CLASS SETS, AND WHAT TO DO INSTEAD
// ---------------------------------------------------------------------------
//
// **DO NOT USE `Flatten` TO COLLAPOSE A SPATIAL MAP.** It will not, and the
// failure is silent.
//
// This is not hypothetical: `SqueezeNet` used a `Flatten` as if it collapsed
// a spatial map, and its output was `(batch, num_classes * H * W)` instead
// of `(batch, num_classes)`. It was fixed in commit `5f84f5d` by replacing
// the `Flatten` with a real GLOBAL AVERAGE POOL -- which is the correct
// operation, because averaging over the spatial axes reduces (C, H, W) to
// (C,) while genuinely using the spatial extent.
//
// **IF YOU NEED A REAL FLATTEN, USE A GLOBAL AVERAGE POOL** (or an adaptive
// average pool with the right output size). `layers/pooling/pool_layer.h`
// provides the pooling layers; `layers/generative/capsnet.cpp` shows the
// idiom for computing a per-sample spatial reduction by hand.
//
// **CHANGING THIS CLASS TO ACTUALLY FLATTEN IS A REPO-WIDE DECISION, NOT A
// BUG FIX**, and it has deliberately NOT been done: every layer that relies
// on the current no-op behaviour would break, because their tensors are
// already flat and a real flatten of a (N, C*H*W) tensor is the identity
// anyway -- but a real flatten of a (N, C*H*W) tensor that is NOT what the
// caller wanted would silently change every downstream width. See
// NOT_FIXED.md.
//
// **SO: this class is here for the CNN architectures' diagrams, it is
// correct where it is used, and it is the WRONG TOOL anywhere else.**
// ===========================================================================

class Flatten : public Layer {
public:
    Flatten() {}
    // Returns the input UNCHANGED. See the contract above.
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /* learning_rate */) override;
    void update_weights(double) override {}
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {}; }
    std::vector<Tensor*> gradients() override { return {}; }
    void zero_grad() override {}
};

#endif
