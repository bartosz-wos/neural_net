#include "nn/layers/generative/coupling_layer.h"
#include <cmath>
#include <stdexcept>

CouplingLayer::CouplingLayer(size_t input_dim, size_t split_dim, size_t hidden_size)
    : input_dim_(input_dim),
      split_dim_(split_dim),
      hidden_size_(hidden_size == 0 ? std::max(size_t(4), input_dim / 4) : hidden_size),
      half_dim_(input_dim / 2),
      // s_net: half_dim -> hidden_size -> half_dim (outputs log_scale per element)
      s_net_(half_dim_, hidden_size_),
      s_scale_(hidden_size_, half_dim_),
      t_net_(half_dim_, hidden_size_),
      t_scale_(hidden_size_, half_dim_),
      rng_(42),
      dL_d_log_det_(-1.0)
{
    if (input_dim % 2 != 0) {
        throw std::invalid_argument("CouplingLayer requires even input_dim");
    }
    s_net_.init_weights("xavier");
    s_scale_.init_weights("xavier");
    t_net_.init_weights("xavier");
    t_scale_.init_weights("xavier");
}

void CouplingLayer::compute_st_from_x1(const Tensor& x1) {
    Tensor h = s_net_.forward(x1);
    log_scale_ = s_scale_.forward(h);
    h = t_net_.forward(x1);
    translation_ = t_scale_.forward(h);
}

Tensor CouplingLayer::forward(const Tensor& input) {
    last_input_ = input.clone();

    if (split_dim_ == 1) {
        // Col-wise split: x1 = left half cols, x2 = right half cols
        size_t half_cols = input.cols / 2;
        size_t rows = input.rows;
        size_t actual_half = rows * half_cols;

        // Build x1 flat: all elements in left half, flattened to (1, rows*half_cols)
        x1_ = Tensor(1, actual_half);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < half_cols; ++j)
                x1_[0][idx++] = input[i][j];

        // s(x1), t(x1)
        compute_st_from_x1(x1_);

        // Output: clone input then overwrite right half with transform
        Tensor y = input.clone();
        idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = half_cols; j < input.cols; ++j) {
                double s = log_scale_[0][idx];
                // Clip s to prevent exp(s) overflow
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                double t = translation_[0][idx];
                y[i][j] = std::exp(s) * input[i][j] + t;
                idx++;
            }

        // Cache x2 flat for backward
        x2_ = Tensor(1, actual_half);
        idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = half_cols; j < input.cols; ++j)
                x2_[0][idx++] = input[i][j];

        // Log det = sum of s(x1) elements
        log_det_ = 0.0;
        for (size_t k = 0; k < actual_half; ++k)
            log_det_ += log_scale_[0][k];

        return y;
    } else {
        // split_dim_ == 0: row-wise split
        size_t half_rows = input.rows / 2;
        size_t cols = input.cols;
        size_t actual_half = half_rows * cols;

        // Build x1 flat: all elements in top half rows, flattened to (1, half_rows*cols)
        x1_ = Tensor(1, actual_half);
        size_t idx = 0;
        for (size_t i = 0; i < half_rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                x1_[0][idx++] = input[i][j];

        // s(x1), t(x1)
        compute_st_from_x1(x1_);

        // Output: clone input then overwrite bottom half with transform
        Tensor y = input.clone();
        idx = 0;
        for (size_t i = half_rows; i < input.rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double s = log_scale_[0][idx];
                // Clip s to prevent exp(s) overflow
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                double t = translation_[0][idx];
                y[i][j] = std::exp(s) * input[i][j] + t;
                idx++;
            }

        // Cache x2 flat for backward
        x2_ = Tensor(1, actual_half);
        idx = 0;
        for (size_t i = half_rows; i < input.rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                x2_[0][idx++] = input[i][j];

        // Log det = sum of s(x1) elements
        log_det_ = 0.0;
        for (size_t k = 0; k < actual_half; ++k)
            log_det_ += log_scale_[0][k];

        return y;
    }
}

Tensor CouplingLayer::inverse(const Tensor& y) {
    // Inverse: x1 = y1 (pass-through), x2 = (y2 - t) / exp(s)
    Tensor x = y.clone();

    if (split_dim_ == 1) {
        size_t half_cols = y.cols / 2;
        size_t rows = y.rows;
        size_t actual_half = rows * half_cols;

        // x1_flat: flattened left half to (1, rows*half_cols)
        Tensor x1_flat(1, actual_half);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < half_cols; ++j)
                x1_flat[0][idx++] = y[i][j];

        compute_st_from_x1(x1_flat);

        idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = half_cols; j < y.cols; ++j) {
                double s = log_scale_[0][idx];
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                double scale = std::exp(s);
                double trans = translation_[0][idx];
                x[i][j] = (y[i][j] - trans) / scale;
                idx++;
            }

    } else {
        size_t half_rows = y.rows / 2;
        size_t cols = y.cols;
        size_t actual_half = half_rows * cols;

        Tensor x1_flat(1, actual_half);
        size_t idx = 0;
        for (size_t i = 0; i < half_rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                x1_flat[0][idx++] = y[i][j];

        compute_st_from_x1(x1_flat);

        idx = 0;
        for (size_t i = half_rows; i < y.rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double s = log_scale_[0][idx];
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                double scale = std::exp(s);
                double trans = translation_[0][idx];
                x[i][j] = (y[i][j] - trans) / scale;
                idx++;
            }
    }

    return x;
}

Tensor CouplingLayer::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Forward: y1 = x1, y2 = exp(s) * x2 + t
    // L = loss(y) - log_det  (for MLE)
    // dL/dx2_i = dL/d(y2_i) * exp(s_i)
    // dL/ds_i = dL/d(y2_i) * exp(s_i) * x2_i - 1  (MLE: -log_det gradient)
    // dL/dt_i = dL/d(y2_i)
    // dL/dx1 = grad_y1 (pass-through) + s_net.backward(dL/ds) + t_net.backward(dL/dt)

    if (split_dim_ == 1) {
        size_t half_cols = grad_output.cols / 2;
        size_t rows = grad_output.rows;
        size_t actual_half = rows * half_cols;

        // grad_y2: right half cols
        // grad_x2 = grad_y2 * exp(s)
        Tensor grad_y2_flat(1, actual_half);
        Tensor grad_x2_flat(1, actual_half);
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = half_cols; j < grad_output.cols; ++j) {
                double gy2 = grad_output[i][j];
                double s = log_scale_[0][idx];
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                grad_y2_flat[0][idx] = gy2;
                grad_x2_flat[0][idx] = gy2 * std::exp(s);
                idx++;
            }

        // dL/ds = grad_x2 * x2 + dL_d_log_det_ (MLE contribution: -1)
        Tensor grad_s = grad_x2_flat.hadamard(x2_);
        for (size_t k = 0; k < actual_half; ++k)
            grad_s[0][k] += dL_d_log_det_;

        // dL/dt = grad_y2
        Tensor grad_t = grad_y2_flat;

        // Backprop through s_scale_ (s: hidden -> half_dim)
        // then s_net_ (h: half_dim -> hidden)
        Tensor grad_h_s = s_scale_.backward(grad_s, 0.0);
        Tensor grad_x1_from_s = s_net_.backward(grad_h_s, 0.0);

        // Backprop through t_scale_ then t_net_
        Tensor grad_h_t = t_scale_.backward(grad_t, 0.0);
        Tensor grad_x1_from_t = t_net_.backward(grad_h_t, 0.0);

        // dL/d(x1) combined
        Tensor grad_x1_combined = grad_x1_from_s + grad_x1_from_t;

        // Build full grad_input
        Tensor grad_input(grad_output.rows, grad_output.cols);

        // Left half (x1): grad_y1 (pass-through) + combined from s,t nets
        idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < half_cols; ++j) {
                grad_input[i][j] = grad_output[i][j] + grad_x1_combined[0][idx];
                idx++;
            }

        // Right half (x2): grad_x2
        idx = 0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = half_cols; j < grad_output.cols; ++j) {
                grad_input[i][j] = grad_x2_flat[0][idx];
                idx++;
            }

        return grad_input;

    } else {
        // split_dim_ == 0: row-wise split
        size_t half_rows = grad_output.rows / 2;
        size_t cols = grad_output.cols;
        size_t actual_half = half_rows * cols;

        Tensor grad_y2_flat(1, actual_half);
        Tensor grad_x2_flat(1, actual_half);
        size_t idx = 0;
        for (size_t i = half_rows; i < grad_output.rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double gy2 = grad_output[i][j];
                double s = log_scale_[0][idx];
                if (s > 20.0) s = 20.0;
                if (s < -20.0) s = -20.0;
                grad_y2_flat[0][idx] = gy2;
                grad_x2_flat[0][idx] = gy2 * std::exp(s);
                idx++;
            }

        Tensor grad_s = grad_x2_flat.hadamard(x2_);
        for (size_t k = 0; k < actual_half; ++k)
            grad_s[0][k] += dL_d_log_det_;

        Tensor grad_t = grad_y2_flat;

        Tensor grad_h_s = s_scale_.backward(grad_s, 0.0);
        Tensor grad_x1_from_s = s_net_.backward(grad_h_s, 0.0);

        Tensor grad_h_t = t_scale_.backward(grad_t, 0.0);
        Tensor grad_x1_from_t = t_net_.backward(grad_h_t, 0.0);

        Tensor grad_x1_combined = grad_x1_from_s + grad_x1_from_t;

        Tensor grad_input(grad_output.rows, grad_output.cols);

        // Top half (x1): grad_y1 + combined
        idx = 0;
        for (size_t i = 0; i < half_rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                grad_input[i][j] = grad_output[i][j] + grad_x1_combined[0][idx];
                idx++;
            }

        // Bottom half (x2): grad_x2
        idx = 0;
        for (size_t i = half_rows; i < grad_output.rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                grad_input[i][j] = grad_x2_flat[0][idx];
                idx++;
            }

        return grad_input;
    }
}

void CouplingLayer::update_weights(double learning_rate) {
    s_net_.update_weights(learning_rate);
    s_scale_.update_weights(learning_rate);
    t_net_.update_weights(learning_rate);
    t_scale_.update_weights(learning_rate);
}

std::vector<Tensor*> CouplingLayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : s_net_.parameters()) result.push_back(p);
    for (Tensor* p : s_scale_.parameters()) result.push_back(p);
    for (Tensor* p : t_net_.parameters()) result.push_back(p);
    for (Tensor* p : t_scale_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> CouplingLayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : s_net_.gradients()) result.push_back(g);
    for (Tensor* g : s_scale_.gradients()) result.push_back(g);
    for (Tensor* g : t_net_.gradients()) result.push_back(g);
    for (Tensor* g : t_scale_.gradients()) result.push_back(g);
    return result;
}

void CouplingLayer::zero_grad() {
    s_net_.zero_grad();
    s_scale_.zero_grad();
    t_net_.zero_grad();
    t_scale_.zero_grad();
}