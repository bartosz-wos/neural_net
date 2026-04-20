#include "batch_norm.h"
#include <cmath>

BatchNorm1D::BatchNorm1D(size_t features, double eps, double momentum)
    : gamma(Tensor::random(1, features, 0.01))
    , beta(Tensor::zeros(1, features))
    , eps(eps), momentum(momentum)
    , running_mean(Tensor::zeros(1, features))
    , running_var(Tensor::zeros(1, features))
    , grad_gamma_(1, features)
    , grad_beta_(1, features)
    , grad_x(0, 0)
    , training(true)
{
    running_var.fill(1.0);
}

Tensor BatchNorm1D::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t features = input.cols;

    last_x = input;
    last_mean = Tensor(1, features);
    last_var = Tensor(1, features);
    Tensor output(batch, features);

    if (training) {
        // Compute mean over batch (axis=0) for each feature
        for (size_t f = 0; f < features; ++f) {
            double m = 0.0;
            for (size_t b = 0; b < batch; ++b) m += input[b][f];
            m /= batch;
            last_mean[0][f] = m;

            double v = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                double d = input[b][f] - m;
                v += d * d;
            }
            v /= batch;
            last_var[0][f] = v;

            double inv_std = 1.0 / std::sqrt(v + eps);
            for (size_t b = 0; b < batch; ++b) {
                double norm = (input[b][f] - m) * inv_std;
                output[b][f] = gamma[0][f] * norm + beta[0][f];
            }
        }
        for (size_t f = 0; f < features; ++f) {
            running_mean[0][f] = (1 - momentum) * running_mean[0][f] + momentum * last_mean[0][f];
            running_var[0][f] = (1 - momentum) * running_var[0][f] + momentum * last_var[0][f];
        }
    } else {
        for (size_t f = 0; f < features; ++f) {
            double m = running_mean[0][f];
            double v = running_var[0][f];
            double inv_std = 1.0 / std::sqrt(v + eps);
            for (size_t b = 0; b < batch; ++b) {
                double norm = (input[b][f] - m) * inv_std;
                output[b][f] = gamma[0][f] * norm + beta[0][f];
            }
        }
    }
    return output;
}

void BatchNorm1D::update_weights(double learning_rate) {
    for (size_t f = 0; f < gamma.cols; ++f) {
        gamma[0][f] -= learning_rate * grad_gamma_[0][f];
        beta[0][f] -= learning_rate * grad_beta_[0][f];
    }
}

std::vector<Tensor*> BatchNorm1D::parameters() { return {&gamma, &beta}; }
std::vector<Tensor*> BatchNorm1D::gradients() { return {&grad_gamma_, &grad_beta_}; }
void BatchNorm1D::zero_grad() { grad_gamma_.fill(0.0); grad_beta_.fill(0.0); }

Tensor BatchNorm1D::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t batch = grad_output.rows;
    size_t features = grad_output.cols;

    grad_gamma_ = Tensor(1, features);
    grad_beta_ = Tensor(1, features);
    grad_x = Tensor(batch, features);

    // Accumulate dL/d(var) and dL/d(mu) across all samples per feature
    for (size_t f = 0; f < features; ++f) {
        double inv_std_f = 1.0 / std::sqrt(last_var[0][f] + eps);
        double g_g = 0.0, g_b = 0.0;
        double g_var = 0.0, g_mu = 0.0;

        for (size_t b = 0; b < batch; ++b) {
            double x = last_x[b][f];
            double mu_f = last_mean[0][f];
            double norm = (x - mu_f) * inv_std_f;
            double dz = grad_output[b][f] * gamma[0][f];

            g_b += grad_output[b][f];
            g_g += grad_output[b][f] * norm;

            // dL/d(var) accum: dz * (x-mu) * (-0.5) * inv_std^3
            g_var += dz * (x - mu_f) * (-0.5) * inv_std_f * inv_std_f * inv_std_f;
            // dL/d(mu) accum: dz * (-inv_std)
            g_mu += dz * (-inv_std_f);
        }

        grad_gamma_[0][f] = g_g;
        grad_beta_[0][f] = g_b;

        // Now compute dL/dx for each sample using accumulated g_var and g_mu
        for (size_t b = 0; b < batch; ++b) {
            double x = last_x[b][f];
            double mu_f = last_mean[0][f];
            double v = last_var[0][f];
            v = std::max(v, eps);  // guard near-zero variance
            double inv_std_f = 1.0 / std::sqrt(v + eps);

            double dz = grad_output[b][f] * gamma[0][f];
            // dL/dx = dz * inv_std + dvar * 2*(x-mu)/N + dmu/N
            grad_x[b][f] = dz * inv_std_f
                         + g_var * 2.0 * (x - mu_f) / static_cast<double>(batch)
                         + g_mu / static_cast<double>(batch);
        }
    }
    return grad_x;
}
