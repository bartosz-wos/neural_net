#ifndef FF_LAYER_H
#define FF_LAYER_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>

// FFLayer: a single layer in Hinton's Forward-Forward Algorithm.
//
// Hinton, "The Forward-Forward Algorithm: Some preliminary investigations",
// 2022 (https://www.cs.toronto.edu/~hinton/FFA.pdf).
//
// Standard BP-free design:
//   y = relu(W @ x + b)
//   goodness(y) = mean(y_i^2)            // scalar per sample
//
// The layer is trained WITHOUT backprop. Instead it runs TWO forward passes:
//   * positive pass: input is a positive sample (data + correct label) →
//     we want goodness to be ABOVE a threshold θ.
//   * negative pass: input is a negative sample (data + wrong / no label) →
//     we want goodness to be BELOW θ.
//
// Local loss (per Hinton 2022 §3.1): a logistic-style function of the
// goodness:
//   L+ = -log σ(g_pos - θ)     (push g_pos up)
//   L- = -log σ(θ - g_neg)     (push g_neg down)
// so the layer's local objective L = L+ + L-.
//
// Local gradient w.r.t. the linear pre-activation h = W @ x + b:
//   dL+/dg_pos = -σ(θ - g_pos)
//   dL-/dg_neg = +σ(g_neg - θ)
//
// and then for each sample:
//   dL/dh = dL/dg * 2/N * (y > 0) ⊙ y   (squared-mask through ReLU)
//   dL/dW = dL/dh @ x^T
//   dL/db = dL/dh
//   dL/dx = W^T @ dL/dh
//
// We implement `train_ff(pos_x, neg_x)` which does both passes and applies
// the local weight update with a SMALL FF learning rate. The standard
// `Layer::backward` is also implemented (for compatibility / debugging)
// and propagates dL/dx via the standard chain rule — useful for sanity
// checks but NOT how FFLayer is meant to be trained.
//
// Layer::forward simply computes y = relu(W @ x + b) and caches x and y.
//
class FFLayer : public Layer {
public:
    // in_features: input dimension (e.g. data_dim + num_classes when the
    //             first layer takes a label-bias concatenation).
    // out_features: number of units in this layer.
    // threshold:    goodness threshold θ. Hinton's paper uses θ = 2.0 for
    //               MNIST. Default 2.0.
    // lr_ff:        local FF learning rate (paper uses 0.03 for MNIST
    //               first-layer experiments). Default 0.03.
    FFLayer(size_t in_features, size_t out_features,
            double threshold = 2.0, double lr_ff = 0.03);

    // Standard forward: y = relu(W @ x + b), caches x and y for backward.
    // Input: (B, in_features). Output: (B, out_features).
    Tensor forward(const Tensor& input) override;

    // FF training step. Runs forward on pos_x AND neg_x, computes the
    // local goodness gradient, and updates W and b in-place. Returns the
    // (g_pos, g_neg) pair for diagnostics.
    //
    // This is the ONLY way FFLayer is meant to learn — backprop is not
    // used. The standard backward() is provided only for compatibility
    // with the Layer interface.
    std::pair<double, double> train_ff(const Tensor& pos_x,
                                        const Tensor& neg_x);

    // Standard backward (chain-rule compatible). Propagates dL/dx back
    // through the layer using the cached activations. NOT used during
    // FF training — only for sanity checks / hybrid usage.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    // Apply weight decay-style step using lr_ff_ (the FF learning rate).
    // Provided to satisfy the Layer interface — actual training should use
    // train_ff().
    void update_weights(double learning_rate) override;

    Tensor get_weights() const override { return W; }
    Tensor get_gradients() const override { return grad_W; }

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    std::string name() const override { return "FFLayer"; }

    // Inspectors
    double threshold() const { return threshold_; }
    double lr_ff() const { return lr_ff_; }
    size_t in_features() const { return W.cols; }
    size_t out_features() const { return W.rows; }

    // Set the FF learning rate (paper-recommended sweep).
    void set_lr_ff(double lr) { lr_ff_ = lr; }
    void set_threshold(double theta) { threshold_ = theta; }

private:
    Tensor W;        // (out, in)
    Tensor b;        // (1, out)
    Tensor grad_W;   // (out, in)
    Tensor grad_b;   // (1, out)
    Tensor last_input;   // (B, in)
    Tensor last_output;  // (B, out), post-activation

    double threshold_;
    double lr_ff_;

    // Helpers
    void init_weights();
    static double sigmoid(double x);
};

// ============================================================================
// FFNetwork: a stack of FFLayers for end-to-end FF training.
//
// Architecture:
//   layer 0: takes (raw_input, one_hot(label)) → hidden_0
//   layer i: takes hidden_{i-1} → hidden_i   (i = 1..L-1)
//   final "softmax" / classification: not done by FF — instead we report
//   per-label goodness summed across all layers, and pick the argmax
//   label (Hinton's "match the label whose goodness is highest").
//
// FFNetwork exposes:
//   * forward_with_label(x, label): returns the activation stack. Used by
//     the caller to generate positive samples for layer-wise training.
//   * negative_sample(label): returns a one-hot label different from
//     `label` (random uniform over the other num_classes - 1 labels).
//   * train_step(x, label): runs an FF training step — positive pass on
//     (x, true_label) and negative pass on (x, false_label). Updates all
//     layers locally.
//   * predict(x): returns argmax_label [sum_goodness_layer_i(x ⊕ label)].
//     Inference is "try each label, pick the one with the highest
//     goodness".
// ============================================================================
class FFNetwork : public Layer {
public:
    // input_dim:  raw input dimension (no label).
    // hidden_dims: sizes of the hidden FF layers. Must be non-empty.
    // num_classes: number of labels.
    // threshold:    goodness threshold shared by all layers (default 2.0).
    // lr_ff:        shared FF learning rate (default 0.03).
    FFNetwork(size_t input_dim,
              const std::vector<size_t>& hidden_dims,
              size_t num_classes,
              double threshold = 2.0,
              double lr_ff = 0.03);

    // Standard forward: takes a TENSOR whose cols == input_dim + num_classes
    // (i.e. label is one-hot-concatenated to the input). Runs all layers
    // and returns the FINAL activation. Provided for the Layer interface.
    Tensor forward(const Tensor& input) override;

    // Inference: returns argmax over labels (summed goodness).
    int predict(const Tensor& x);

    // Inference: returns per-class goodness vector (1, num_classes).
    Tensor predict_goodness(const Tensor& x);

    // Single training step. x is (B, input_dim). label is the integer
    // true label (0..num_classes - 1) for ALL B samples (we use the same
    // label for every sample in the batch — the FF paper trains per-sample
    // and we follow that convention). Returns a vector of (g_pos, g_neg)
    // per layer for diagnostics.
    std::vector<std::pair<double, double>> train_step(const Tensor& x, int label);

    // Helper: build a positive (x ⊕ one_hot_label) batch.
    Tensor make_positive(const Tensor& x, int label) const;

    // Helper: build a negative batch by either using a random wrong label
    // (default), a specific wrong label, or a zero/erased label.
    Tensor make_negative_random(const Tensor& x, int true_label) const;
    Tensor make_negative_label(const Tensor& x, int false_label) const;

    // Standard backward/parameters interface (delegates to the last layer).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return layers_.back()->get_weights(); }
    Tensor get_gradients() const override { return layers_.back()->get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    std::string name() const override { return "FFNetwork"; }

    // Inspectors
    size_t num_classes() const { return num_classes_; }
    size_t input_dim() const { return input_dim_; }
    const std::vector<size_t>& hidden_dims() const { return hidden_dims_; }
    size_t num_layers() const { return layers_.size(); }

    // Access an individual layer (for advanced usage / tests).
    FFLayer* layer(size_t i) { return layers_[i].get(); }
    const FFLayer* layer(size_t i) const { return layers_[i].get(); }

private:
    std::vector<std::unique_ptr<FFLayer>> layers_;
    std::vector<size_t> hidden_dims_;
    size_t input_dim_;
    size_t num_classes_;
    double threshold_;
    double lr_ff_;
};

#endif