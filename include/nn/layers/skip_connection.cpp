#include "skip_connection.h"
#include "convolutions/conv_layer.h"
#include <cassert>

SkipConnection::SkipConnection(Layer* inner)
    : inner_(inner), needs_projection_(false)
{}

Tensor SkipConnection::forward(const Tensor& input) {
    last_input_ = input;
    Tensor out = inner_->forward(input);

    // Determine output size: out is (batch, inner_out_features)
    // If out.cols != input.cols, we need a projection
    if (out.cols != input.cols) {
        if (!shortcut_) {
            size_t in_feat = input.cols;
            size_t out_feat = out.cols;
            shortcut_ = std::make_unique<Dense>(in_feat, out_feat);
        }
        needs_projection_ = true;
    } else {
        needs_projection_ = false;
    }

    if (needs_projection_) {
        // Project input to match output dimension
        Tensor projected = shortcut_->forward(input); // (batch, out_feat)
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                out[i][j] += projected[i][j];
    } else {
        // Direct add: input and output must have same shape
        assert(out.rows == input.rows && out.cols == input.cols);
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                out[i][j] += input[i][j];
    }
    return out;
}

Tensor SkipConnection::backward(const Tensor& grad_output, double learning_rate) {
    // Residual: out = inner(x) + proj(x)  [if projection]
    //           out = inner(x) + x        [if no projection]
    // dL/dx = dL/dinner * dinner/dx + dL/dproj * dproj/dx  [projection case]
    // dL/dx = dL/dinner * dinner/dx + dL/dout * 1           [no projection case]

    if (needs_projection_) {
        // grad_output: (batch, out_feat)
        // inner_->backward(grad_output) returns grad w.r.t. inner's input → (batch, in_feat)
        // shortcut_->backward(grad_output) returns grad w.r.t. shortcut's input → (batch, in_feat)
        // Both gradients flow to the same input — sum them.
        Tensor shortcut_grad = shortcut_->backward(grad_output, learning_rate);
        Tensor inner_grad = inner_->backward(grad_output, learning_rate);
        // shortcut_grad and inner_grad both have shape (batch, in_feat)
        Tensor grad_input(grad_output.rows, shortcut_grad.cols);
        for (size_t i = 0; i < grad_input.rows; ++i)
            for (size_t j = 0; j < grad_input.cols; ++j)
                grad_input[i][j] = shortcut_grad[i][j] + inner_grad[i][j];
        return grad_input;
    } else {
        // grad flows to inner layer and identity path
        Tensor inner_grad = inner_->backward(grad_output, learning_rate);
        // dL/dx += grad_output (identity gradient)
        for (size_t i = 0; i < inner_grad.rows; ++i)
            for (size_t j = 0; j < inner_grad.cols; ++j)
                inner_grad[i][j] += grad_output[i][j];
        return inner_grad;
    }
}

void SkipConnection::update_weights(double learning_rate) {
    inner_->update_weights(learning_rate);
    if (needs_projection_)
        shortcut_->update_weights(learning_rate);
}

std::vector<Tensor*> SkipConnection::parameters() {
    std::vector<Tensor*> params = inner_->parameters();
    if (shortcut_)
        for (Tensor* p : shortcut_->parameters())
            params.push_back(p);
    return params;
}

std::vector<Tensor*> SkipConnection::gradients() {
    std::vector<Tensor*> grads = inner_->gradients();
    if (shortcut_)
        for (Tensor* g : shortcut_->gradients())
            grads.push_back(g);
    return grads;
}

void SkipConnection::zero_grad() {
    inner_->zero_grad();
    if (shortcut_)
        shortcut_->zero_grad();
}
