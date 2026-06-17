#include "layer_norm.h"
#include <cmath>

LayerNorm::LayerNorm(size_t features, double eps)
    : gamma(Tensor::random(1, features, 0.01))
    , beta(Tensor::zeros(1, features))
    , eps(eps)
    , training(true)
    , grad_gamma_(1, features)
    , grad_beta_(1, features)
    , grad_x(0, 0)
{
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);
}

Tensor LayerNorm::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t features = input.cols;

    last_x = input;
    last_mean = Tensor(1, batch);
    last_var = Tensor(1, batch);

    Tensor output(batch, features);

    for (size_t b = 0; b < batch; ++b) {
        double mean = 0.0;
        for (size_t f = 0; f < features; ++f)
            mean += input[b][f];
        mean /= features;
        last_mean[0][b] = mean;

        double var = 0.0;
        for (size_t f = 0; f < features; ++f) {
            double diff = input[b][f] - mean;
            var += diff * diff;
        }
        var /= features;
        var = std::max(var, 1e-7);
        last_var[0][b] = var;

        double sqrt_var = std::sqrt(var + eps);
        for (size_t f = 0; f < features; ++f) {
            double norm = (input[b][f] - mean) / sqrt_var;
            output[b][f] = gamma[0][f] * norm + beta[0][f];
        }
    }
    return output;
}

Tensor LayerNorm::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t batch = grad_output.rows;
    size_t features = grad_output.cols;

    // Initialize (on first backward call within an optimizer step) and accumulate
    // across multiple backward() calls before zero_grad() is called.
    if (grad_gamma_.rows == 0) grad_gamma_ = Tensor(1, features);
    if (grad_beta_.rows == 0)  grad_beta_  = Tensor(1, features);
    // NOTE: no fill(0.0) here — accumulation is += so multiple backward calls
    // between zero_grad() calls correctly sum their contributions.

    grad_x = Tensor(batch, features);

    for (size_t f = 0; f < features; ++f) {
        double g_g = 0.0, g_b = 0.0;
        for (size_t b = 0; b < batch; ++b) {
            g_g += grad_output[b][f] * ((last_x[b][f] - last_mean[0][b]) /
                                         std::sqrt(last_var[0][b] + eps));
            g_b += grad_output[b][f];
        }
        grad_gamma_[0][f] += g_g;
        grad_beta_[0][f] += g_b;
    }

    for (size_t b = 0; b < batch; ++b) {
        double var = last_var[0][b];
        var = std::max(var, 1e-7);
        double sqrt_var = std::sqrt(var + eps);
        double inv_var = 1.0 / sqrt_var;

        Tensor dNorm(1, features);
        for (size_t f = 0; f < features; ++f) {
            dNorm[0][f] = grad_output[b][f] * gamma[0][f];
        }

        double dVar = 0.0;
        for (size_t f = 0; f < features; ++f) {
            dVar += dNorm[0][f] * (last_x[b][f] - last_mean[0][b]);
        }
        dVar *= -0.5 * inv_var * inv_var * inv_var;

        double dMu = 0.0;
        for (size_t f = 0; f < features; ++f) {
            dMu -= dNorm[0][f] * inv_var;
        }
        // NOTE: the sum(x - mean) term vanishes (equals 0 by definition of mean),
        // so no correction needed here. The old code incorrectly used last_mean[0][b].

        for (size_t f = 0; f < features; ++f) {
            double dx_norm = dNorm[0][f] * inv_var;
            double dx_var = dVar * 2.0 * (last_x[b][f] - last_mean[0][b]) / features;
            double dx_mu = dMu / features;
            grad_x[b][f] = dx_norm + dx_var + dx_mu;
        }
    }
    return grad_x;
}

void LayerNorm::update_weights(double learning_rate) {
    for (size_t f = 0; f < gamma.cols; ++f) {
        gamma[0][f] -= learning_rate * grad_gamma_[0][f];
        beta[0][f] -= learning_rate * grad_beta_[0][f];
    }
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> LayerNorm::gradients() {
    return {&grad_gamma_, &grad_beta_};
}

void LayerNorm::zero_grad() {
    grad_gamma_.fill(0.0);
    grad_beta_.fill(0.0);
}

// ============== Dropout ==============

Tensor Dropout::forward(const Tensor& input) {
    if (!training) return input;

    mask = Tensor(input.rows, input.cols);
    Tensor output(input.rows, input.cols);

    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double r = (double)rand() / RAND_MAX;
            mask[i][j] = (r < p) ? 0.0 : 1.0;
            output[i][j] = input[i][j] * mask[i][j];
        }
    }
    return output;
}

Tensor Dropout::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (!training) return grad_output;
    Tensor grad(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i)
        for (size_t j = 0; j < grad_output.cols; ++j)
            grad[i][j] = grad_output[i][j] * mask[i][j];
    return grad;
}