#ifndef SKIP_CONNECTION_H
#define SKIP_CONNECTION_H

#include "../core/layer.h"

// SkipConnection: wraps a layer as x + f(x) residual.
// If dimensions of input and output differ, the skip path is projected
// to match the output dimensions via a learnable linear projection.
class SkipConnection : public Layer {
public:
    // Takes ownership of `inner` — the residual branch
    explicit SkipConnection(Layer* inner);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return Tensor(0,0); }
    Tensor get_gradients() const override { return Tensor(0,0); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    Layer* inner_layer() const { return inner_.get(); }

private:
    std::unique_ptr<Layer> inner_;
    std::unique_ptr<Dense> shortcut_; // projection if needed
    Tensor last_input_;
    bool needs_projection_ = false;
};

#endif
