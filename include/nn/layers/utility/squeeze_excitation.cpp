#include "squeeze_excitation.h"
#include <cmath>

SEBlock::SEBlock(size_t in_channels, size_t reduction)
    : fc1_(in_channels, std::max(in_channels / reduction, size_t(1))),
      fc2_(std::max(in_channels / reduction, size_t(1)), in_channels),
      in_channels_(in_channels), reduction_(reduction),
      last_excitation_(1, in_channels) {}

Tensor SEBlock::forward(const Tensor& input) {
    last_input_ = input;
    size_t batch = input.rows;
    size_t channels = in_channels_;
    size_t spatial = input.cols / channels;

    // Global average pooling: squeeze each channel to a scalar
    Tensor gap(batch, channels);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t s = 0; s < spatial; ++s)
                sum += input[b][c * spatial + s];
            gap[b][c] = sum / spatial;
        }
    }

    // FC → ReLU → FC → Sigmoid
    last_gap_ = gap;
    Tensor z = fc1_.forward(gap);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t j = 0; j < z.cols; ++j)
            z[i][j] = std::max(0.0, z[i][j]);

    last_fc1_relu_ = z;
    z = fc2_.forward(z);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t j = 0; j < z.cols; ++j)
            z[i][j] = 1.0 / (1.0 + std::exp(-z[i][j])); // sigmoid

    last_excitation_ = z; // (batch, channels)
    return scale_channels(input, z);  // apply SE attention to input channels
}

Tensor SEBlock::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, channels * spatial)
    // last_excitation_: (batch, channels)
    // last_fc1_relu_: (batch, reduced_channels)
    // last_gap_: (batch, channels)
    // last_input_: (batch, channels * spatial)
    size_t batch = last_input_.rows;
    size_t channels = in_channels_;
    size_t spatial = last_input_.cols / channels;

    // Backprop through scale_channels: dL/d(excitation) = sum over spatial of (dL/d(out) * input)
    // d(out)/d(excitation) = input, so grad_excitation[b][c] = sum_s grad_output[b][c*S+s] * input[b][c*S+s]
    Tensor grad_excitation(batch, channels);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t s = 0; s < spatial; ++s)
                sum += grad_output[b][c * spatial + s] * last_input_[b][c * spatial + s];
            grad_excitation[b][c] = sum;
        }
    }

    // dL/d(excitation) through sigmoid: grad_fc2_input[b][c] = grad_excitation[b][c] * excitation[b][c] * (1 - excitation[b][c])
    Tensor grad_sigmoid(batch, channels);
    for (size_t b = 0; b < batch; ++b)
        for (size_t c = 0; c < channels; ++c) {
            double ex = last_excitation_[b][c];
            grad_sigmoid[b][c] = grad_excitation[b][c] * ex * (1.0 - ex);
        }

    // Backprop fc2
    Tensor grad_fc1_relu = fc2_.backward(grad_sigmoid, learning_rate);

    // ReLU derivative: pass through only positive elements
    // (elements where last_fc1_relu > 0 were kept; others had grad 0)
    Tensor grad_fc1(batch, grad_fc1_relu.cols);
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < grad_fc1_relu.cols; ++j)
            grad_fc1[b][j] = last_fc1_relu_[b][j] > 0 ? grad_fc1_relu[b][j] : 0.0;

    // Backprop fc1: backprop from fc1_relu gradient to fc1 input (gap)
    fc1_.backward(grad_fc1, learning_rate);

    // Recover dL/d(gap) = dL/d(fc1_input) = dL/d(fc1_relu) * W_fc1^T
    // grad_fc1 is dL/d(fc1_relu), weights W_fc1: (out_features=r_reduced, in_features=channels)
    // We need dL/d(gap)[b][c] = sum_r grad_fc1[b][r] * W_fc1[r][c]
    Tensor grad_gap(batch, channels);
    const Tensor& W1 = fc1_.weights;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t r = 0; r < grad_fc1.cols; ++r)
                sum += grad_fc1[b][r] * W1[r][c];
            grad_gap[b][c] = sum;
        }
    }

    // Distribute grad_gap evenly across spatial positions
    Tensor grad_input(batch, channels * spatial);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double g = grad_gap[b][c] / spatial;
            for (size_t s = 0; s < spatial; ++s)
                grad_input[b][c * spatial + s] = g;
        }
    }

    return grad_input;
}

void SEBlock::update_weights(double learning_rate) {
    fc1_.update_weights(learning_rate);
    fc2_.update_weights(learning_rate);
}

void SEBlock::zero_grad() {
    fc1_.zero_grad();
    fc2_.zero_grad();
}

std::vector<Tensor*> SEBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> SEBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    return result;
}

Tensor SEBlock::scale_channels(const Tensor& input, const Tensor& excitation) {
    size_t batch = input.rows;
    size_t channels = excitation.cols;
    size_t spatial = input.cols / channels;

    Tensor output(batch, input.cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            double w = excitation[b][c];
            for (size_t s = 0; s < spatial; ++s)
                output[b][c * spatial + s] = input[b][c * spatial + s] * w;
        }
    }
    return output;
}

// === SEResNetBlock ===

SEResNetBlock::SEResNetBlock(size_t in_channels, size_t out_channels,
                               size_t reduction, size_t H, size_t W)
    : H_(H), W_(W),
      has_skip_(in_channels != out_channels),
      conv1_(in_channels, out_channels, 3, 3, H, W, 1, 1, 1, 1),
      conv2_(out_channels, out_channels, 3, 3, H, W, 1, 1, 1, 1),
      se_(out_channels, reduction),
      skip_conv_(in_channels, out_channels, 1, 1, H, W, 1, 1, 0, 0),
      last_output_(1, out_channels * H * W) {}

Tensor SEResNetBlock::forward(const Tensor& input) {
    last_se_input_ = input; // cache before SE (conv2 output)
    Tensor x = conv1_.forward(input);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);

    x = conv2_.forward(x);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x[i][j] = std::max(0.0, x[i][j]);

    // SE excitation + channel scaling
    x = SEBlock::scale_channels(x, se_.forward(x));

    // Skip connection
    if (has_skip_) {
        last_skip_output_ = skip_conv_.forward(input);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j)
                x[i][j] += last_skip_output_[i][j];
    } else {
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j)
                x[i][j] += input[i][j];
    }

    last_output_ = x;
    return last_output_;
}

Tensor SEResNetBlock::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, out_channels * H * W)
    // Forward: input → conv1(ReLU) → conv2(ReLU) → SE.scale → [+ skip]
    // Backward: grad_output → SE.scale back → conv2 back → conv1 back → skip back

    size_t batch = grad_output.rows;
    size_t out_ch = conv2_.out_channels;
    size_t H_out = H_;
    size_t W_out = W_;
    size_t spatial = H_out * W_out;


    // 1) Backprop through skip connection
    // grad_split: (out_ch * spatial, batch) where for each element:
    //   has_skip_ ? grad_output : grad_output
    // The skip path adds to the main path.
    // For has_skip_ case: grad from output splits to conv2 path AND skip path (equal contribution since addition)
    // For no skip case: grad flows entirely to conv2 path
    Tensor grad_se = grad_output;
    if (has_skip_) {
        // Add skip gradient to main gradient (addition backprop)
        // grad after adding skip contribution: grad_output + grad_skip_from_identity
        // But skip_conv_ output was added to x, so dL/d(skip) = dL/d(x) * 1 for each element
        // The skip path: skip_conv_.forward(input), grad flows through skip_conv_
        // We need to backprop skip_conv_
        grad_se = grad_output; // keep for SE path
        // Backprop skip conv
        skip_conv_.backward(grad_output, learning_rate);
    }
    // For no-skip case, identity path: dL/d(input) += grad_output directly
    // (handled at end)


    // 2) Backprop through SE scale_channels
    // d(out)/d(excitation) = last_se_input_ (conv2 output before SE scaling)
    // d(out)/d(conv2_output) = excitation
    // We need to split grad_output into grad_excitation and grad_conv2_input
    Tensor grad_excitation(batch, out_ch);
    Tensor grad_conv2_input(batch, out_ch * spatial);
    // last_se_input_ is (batch, out_ch * spatial), last_excitation_ is (batch, out_ch)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < out_ch; ++c) {
            // grad_excitation[c] = sum over spatial of grad_output[c*spatial+s] * last_se_input_[c*spatial+s]
            double sum_ex = 0.0;
            for (size_t s = 0; s < spatial; ++s) {
                double g = grad_output[b][c * spatial + s];
                grad_conv2_input[b][c * spatial + s] = g * se_.last_excitation_[b][c];
                sum_ex += g * last_se_input_[b][c * spatial + s];
            }
            grad_excitation[b][c] = sum_ex;
        }
    }

    // 3) SE backward: backprop se_.forward (which includes GAP + fc1 + fc2 + sigmoid)
    // se_.backward returns gradient w.r.t. its input (last_se_input_ = conv2 raw output)
    Tensor grad_se_input = se_.backward(grad_excitation, learning_rate);

    // 4) Backprop conv2: conv2 was forward: conv2_.forward(conv1_relu_output) → raw → ReLU
    // conv2_.backward expects grad w.r.t. conv2 output (after ReLU), returns grad w.r.t. conv2 input (raw)
    // grad_se_input IS grad w.r.t. conv2 raw output (from SE backward)
    // Actually conv2_.backward computes d(loss)/d(conv2_input) given d(loss)/d(conv2_output)
    // conv2 output shape: (batch, out_ch * H * W), conv2_input shape same (conv1_relu output)
    // grad_se_input is the gradient w.r.t. conv2 output after ReLU
    // So we pass grad_se_input directly
    Tensor grad_conv1_relu_output = conv2_.backward(grad_se_input, learning_rate);

    // 5) Backprop conv1: conv1 was forward: conv1_.forward(input) → raw → ReLU
    // conv1_.backward computes d(loss)/d(conv1_input) given d(loss)/d(conv1_output)
    // conv1 output = conv1_relu_output = conv2 input
    // grad_conv1_relu_output is d(loss)/d(conv1_relu_output)
    // We need d(loss)/d(conv1_raw_output) = d(loss)/d(conv1_relu_output) * ReLU_mask
    // conv1_.backward internally uses conv1_.last_input (raw conv1 output) to apply ReLU mask
    Tensor grad_block_input = conv1_.backward(grad_conv1_relu_output, learning_rate);

    // 6) Add identity gradient for no-skip case (direct addition backprop)
    if (!has_skip_) {
        for (size_t i = 0; i < grad_block_input.rows; ++i)
            for (size_t j = 0; j < grad_block_input.cols; ++j)
                grad_block_input[i][j] += grad_output[i][j];
    }

    return grad_block_input;
}

void SEResNetBlock::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv2_.update_weights(learning_rate);
    se_.update_weights(learning_rate);
    skip_conv_.update_weights(learning_rate);
}

void SEResNetBlock::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
    se_.zero_grad();
    skip_conv_.zero_grad();
}

std::vector<Tensor*> SEResNetBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    for (Tensor* p : se_.parameters()) result.push_back(p);
    for (Tensor* p : skip_conv_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> SEResNetBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv2_.gradients()) result.push_back(g);
    for (Tensor* g : se_.gradients()) result.push_back(g);
    for (Tensor* g : skip_conv_.gradients()) result.push_back(g);
    return result;
}