#include "skip_connection.h"
#include "convolutions/conv_layer.h"

SkipConnection::SkipConnection(Layer* inner)
    : inner_(inner)
{}

Tensor SkipConnection::forward(const Tensor& input) {
    last_input_ = input;
    Tensor out = inner_->forward(input);

    // Determine output size: out is (batch, inner_out_features)
    // If out.cols != input.cols, we need a projection
    if (out.cols != input.cols) {
        // Recompute projection whenever dims change (not just first pass)
        size_t in_feat = input.cols;
        size_t out_feat = out.cols;
        shortcut_ = std::make_unique<Dense>(in_feat, out_feat);
        needs_projection_ = true;
    }

    if (needs_projection_) {
        // Project input to match output dimension
        Tensor projected = shortcut_->forward(input); // (batch, out_feat)
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                out[i][j] += projected[i][j];
    } else {
        // Direct add: input and output must have same shape
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                out[i][j] += input[i][j];
    }
    return out;
}

Tensor SkipConnection::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: dL/d(out) where out = F(x) + proj(x)
    // dL/dx = dL/dout + dL/dF * dF/dx  (via chain rule through inner)
    // but for residual: dL/dx = dL/dout + dL/dout_proj (through projection)

    if (needs_projection_) {
        // grad_output split into two paths: shortcut gradient + inner gradient
        // First, grad through projection shortcut
        Tensor shortcut_grad = shortcut_->backward(grad_output, learning_rate);

        // Then, grad through inner branch
        Tensor inner_grad = inner_->backward(grad_output, learning_rate);

        // Combine: grad to input from both paths
        Tensor grad_input(grad_output.rows, shortcut_grad.cols);
        for (size_t i = 0; i < grad_input.rows; ++i)
            for (size_t j = 0; j < grad_input.cols; ++j)
                grad_input[i][j] = shortcut_grad[i][j] + inner_grad[i][j];
        return grad_input;
    } else {
        // grad goes to both input (skip) and inner layer
        Tensor inner_grad = inner_->backward(grad_output, learning_rate);
        // dL/dx += grad_output (identity path)
        // inner_grad already has dL/dx through F(x), now add identity contribution
        Tensor grad_input = inner_grad;
        for (size_t i = 0; i < grad_input.rows; ++i)
            for (size_t j = 0; j < grad_input.cols; ++j)
                grad_input[i][j] += grad_output[i][j];
        return grad_input;
    }
}

void SkipConnection::update_weights(double learning_rate) {
    inner_->update_weights(learning_rate);
    if (needs_projection_)
        shortcut_->update_weights(learning_rate);
}

std::vector<Tensor*> SkipConnection::parameters() {
    std::vector<Tensor*> params = inner_->parameters();
    if (needs_projection_)
        for (Tensor* p : shortcut_->parameters())
            params.push_back(p);
    return params;
}

std::vector<Tensor*> SkipConnection::gradients() {
    std::vector<Tensor*> grads = inner_->gradients();
    if (needs_projection_)
        for (Tensor* g : shortcut_->gradients())
            grads.push_back(g);
    return grads;
}

void SkipConnection::zero_grad() {
    inner_->zero_grad();
    if (needs_projection_)
        shortcut_->zero_grad();
}
