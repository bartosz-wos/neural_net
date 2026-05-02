#include "rms_norm.h"
#include <cmath>

RMSNorm::RMSNorm(size_t features, double eps)
    : gamma(Tensor::random(1, features, 0.01))
    , eps(eps)
    , training(true)
{
}

Tensor RMSNorm::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t features = input.cols;

    last_x = input;
    last_rms = Tensor(1, batch);

    Tensor output(batch, features);

    for (size_t b = 0; b < batch; ++b) {
        double sum_sq = 0.0;
        for (size_t f = 0; f < features; ++f) {
            sum_sq += input[b][f] * input[b][f];
        }
        double rms = std::sqrt(sum_sq / features + eps);
        last_rms[0][b] = rms;

        for (size_t f = 0; f < features; ++f) {
            output[b][f] = input[b][f] * gamma[0][f] / rms;
        }
    }
    return output;
}

Tensor RMSNorm::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t batch = grad_output.rows;
    size_t features = grad_output.cols;

    // Initialize on first backward call; accumulate across calls before zero_grad()
    if (grad_gamma_.rows == 0) grad_gamma_ = Tensor(1, features);

    grad_x = Tensor(batch, features);

    // grad_gamma[f] = sum_b(grad_output[b][f] * (last_x[b][f] / last_rms[b]))
    for (size_t f = 0; f < features; ++f) {
        double g_g = 0.0;
        for (size_t b = 0; b < batch; ++b) {
            g_g += grad_output[b][f] * last_x[b][f] / last_rms[0][b];
        }
        grad_gamma_[0][f] += g_g;
    }

    // grad_x[b][f] = sum_g(grad_output[b][g] * gamma[g] / last_rms[b]
    //                     * (delta_{g,f} - last_x[b][g] * last_x[b][f] / (features * last_rms[b]^2)))
    for (size_t b = 0; b < batch; ++b) {
        double rms = last_rms[0][b];
        double rms_sq = rms * rms;

        for (size_t f = 0; f < features; ++f) {
            double gx = 0.0;
            for (size_t g = 0; g < features; ++g) {
                double delta = (g == f) ? 1.0 : 0.0;
                double correction = last_x[b][g] * last_x[b][f] / (features * rms_sq);
                double coeff = grad_output[b][g] * gamma[0][g] / rms * (delta - correction);
                gx += coeff;
            }
            grad_x[b][f] = gx;
        }
    }

    return grad_x;
}

void RMSNorm::update_weights(double learning_rate) {
    for (size_t f = 0; f < gamma.cols; ++f) {
        gamma[0][f] -= learning_rate * grad_gamma_[0][f];
    }
}

std::vector<Tensor*> RMSNorm::parameters() {
    return {&gamma};
}

std::vector<Tensor*> RMSNorm::gradients() {
    return {&grad_gamma_};
}

void RMSNorm::zero_grad() {
    grad_gamma_.fill(0.0);
}