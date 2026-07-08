#include "instance_norm.h"
#include <cmath>

// ============================================================================
// InstanceNorm1D
// ============================================================================

InstanceNorm1D::InstanceNorm1D(size_t channels, double eps)
    : gamma(Tensor(1, channels))
    , beta(Tensor(1, channels))
    , eps(eps)
    , last_mean(0, 0)
    , last_var(0, 0)
    , grad_gamma(1, channels)
    , grad_beta(1, channels)
    , grad_x(0, 0)
    , training(true)
{
    // Initialize gamma=1, beta=0
    for (size_t c = 0; c < channels; ++c) {
        gamma[0][c] = 1.0;
        beta[0][c]  = 0.0;
        grad_gamma[0][c] = 0.0;
        grad_beta[0][c]  = 0.0;
    }
}

Tensor InstanceNorm1D::forward(const Tensor& input) {
    size_t N = input.rows;
    size_t C = input.cols;
    size_t M = C;  // reduce over the feature axis

    last_x = input;
    last_mean = Tensor(N, 1);
    last_var  = Tensor(N, 1);

    Tensor output(N, C);

    for (size_t n = 0; n < N; ++n) {
        double mean = 0.0;
        for (size_t c = 0; c < C; ++c) mean += input[n][c];
        mean /= static_cast<double>(M);
        last_mean[n][0] = mean;

        double var = 0.0;
        for (size_t c = 0; c < C; ++c) {
            double d = input[n][c] - mean;
            var += d * d;
        }
        var /= static_cast<double>(M);
        last_var[n][0] = var;

        double inv_std = 1.0 / std::sqrt(var + eps);
        for (size_t c = 0; c < C; ++c) {
            double norm = (input[n][c] - mean) * inv_std;
            output[n][c] = gamma[0][c] * norm + beta[0][c];
        }
    }

    return output;
}

Tensor InstanceNorm1D::backward(const Tensor& grad_output, double /*lr*/) {
    size_t N = grad_output.rows;
    size_t C = grad_output.cols;
    size_t M = C;

    grad_x = Tensor(N, C);

    for (size_t n = 0; n < N; ++n) {
        // Recompute mean / var from cached input (we could just use last_mean,
        // but recomputing ensures consistency even if forward and backward are
        // called with perturbed inputs during finite-difference checking.)
        double mean = 0.0;
        for (size_t c = 0; c < C; ++c) mean += last_x[n][c];
        mean /= static_cast<double>(M);
        double var = 0.0;
        for (size_t c = 0; c < C; ++c) {
            double d = last_x[n][c] - mean;
            var += d * d;
        }
        var /= static_cast<double>(M);
        double inv_std = 1.0 / std::sqrt(var + eps);
        double inv_std3 = inv_std * inv_std * inv_std;

        // Accumulate per-sample dgamma / dbeta contributions (over channels)
        for (size_t c = 0; c < C; ++c) {
            double norm = (last_x[n][c] - mean) * inv_std;
            grad_gamma[0][c] += grad_output[n][c] * norm;
            grad_beta[0][c]  += grad_output[n][c];
        }

        // Accumulate per-sample dvar / dmean across channels
        // dL/dvar = -0.5 * sum_m (dL/dy_m * gamma_m) * (x_m - mu) * inv_std^3
        // dL/dmean = -inv_std * sum_m (dL/dy_m * gamma_m)
        double dL_dvar  = 0.0;
        double dL_dmean = 0.0;
        for (size_t c = 0; c < C; ++c) {
            double grad_y = grad_output[n][c] * gamma[0][c];
            double diff = last_x[n][c] - mean;
            dL_dvar  += grad_y * diff * (-0.5) * inv_std3;
            dL_dmean += grad_y * (-inv_std);
        }

        // Compute dL/dx element-by-element:
        //   dx[m] = gamma_m * dL/dy_m * inv_std             (dnorm term)
        //         + dL/dvar * 2 * (x_m - mu) / M             (dvar term)
        //         + dL/dmean / M                             (dmean term)
        for (size_t c = 0; c < C; ++c) {
            double grad_y = grad_output[n][c] * gamma[0][c];
            double diff = last_x[n][c] - mean;
            double dx_norm  = grad_y * inv_std;
            double dx_var   = dL_dvar * 2.0 * diff / static_cast<double>(M);
            double dx_mean  = dL_dmean / static_cast<double>(M);
            grad_x[n][c] = dx_norm + dx_var + dx_mean;
        }
    }

    return grad_x;
}

void InstanceNorm1D::update_weights(double learning_rate) {
    for (size_t c = 0; c < gamma.cols; ++c) {
        gamma[0][c] -= learning_rate * grad_gamma[0][c];
        beta[0][c]  -= learning_rate * grad_beta[0][c];
    }
}

std::vector<Tensor*> InstanceNorm1D::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> InstanceNorm1D::gradients() {
    return {&grad_gamma, &grad_beta};
}

void InstanceNorm1D::zero_grad() {
    grad_gamma.fill(0.0);
    grad_beta.fill(0.0);
}

// ============================================================================
// InstanceNorm2D
// ============================================================================

InstanceNorm2D::InstanceNorm2D(int num_channels, int H, int W, double eps)
    : gamma(Tensor(1, num_channels))
    , beta(Tensor(1, num_channels))
    , eps(eps)
    , num_channels_(num_channels)
    , spatial_(H * W)
    , last_mean(0, 0)
    , last_var(0, 0)
    , grad_gamma(1, num_channels)
    , grad_beta(1, num_channels)
    , grad_x(0, 0)
    , training(true)
{
    for (int c = 0; c < num_channels; ++c) {
        gamma[0][c] = 1.0;
        beta[0][c]  = 0.0;
        grad_gamma[0][c] = 0.0;
        grad_beta[0][c]  = 0.0;
    }
}

Tensor InstanceNorm2D::forward(const Tensor& input) {
    int N = (int)input.rows;
    int C = num_channels_;
    int S = spatial_;
    if ((int)input.cols != C * S) {
        // Reshape is not implicit — the caller must pre-flatten to (N, C*H*W).
        // We do not throw here, just rely on the indexing below; if the columns
        // don't match, results will be garbage.
    }

    last_x = input;
    last_mean = Tensor(N, C);
    last_var  = Tensor(N, C);

    Tensor output(N, C * S);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // Slice [c*S, (c+1)*S)
            int base = c * S;
            double mean = 0.0;
            for (int s = 0; s < S; ++s) mean += input[n][base + s];
            mean /= static_cast<double>(S);
            last_mean[n][c] = mean;

            double var = 0.0;
            for (int s = 0; s < S; ++s) {
                double d = input[n][base + s] - mean;
                var += d * d;
            }
            var /= static_cast<double>(S);
            last_var[n][c] = var;

            double inv_std = 1.0 / std::sqrt(var + eps);
            for (int s = 0; s < S; ++s) {
                int idx = base + s;
                double norm = (input[n][idx] - mean) * inv_std;
                output[n][idx] = gamma[0][c] * norm + beta[0][c];
            }
        }
    }

    return output;
}

Tensor InstanceNorm2D::backward(const Tensor& grad_output, double /*lr*/) {
    int N = (int)grad_output.rows;
    int C = num_channels_;
    int S = spatial_;

    grad_x = Tensor(N, C * S);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            int base = c * S;

            // Recompute mean/var from cached input
            double mean = 0.0;
            for (int s = 0; s < S; ++s) mean += last_x[n][base + s];
            mean /= static_cast<double>(S);
            double var = 0.0;
            for (int s = 0; s < S; ++s) {
                double d = last_x[n][base + s] - mean;
                var += d * d;
            }
            var /= static_cast<double>(S);
            double inv_std = 1.0 / std::sqrt(var + eps);
            double inv_std3 = inv_std * inv_std * inv_std;

            // Per-channel dgamma / dbeta (accumulate over spatial)
            for (int s = 0; s < S; ++s) {
                int idx = base + s;
                double norm = (last_x[n][idx] - mean) * inv_std;
                grad_gamma[0][c] += grad_output[n][idx] * norm;
                grad_beta[0][c]  += grad_output[n][idx];
            }

            // Per-(sample, channel) dvar / dmean
            double dL_dvar  = 0.0;
            double dL_dmean = 0.0;
            for (int s = 0; s < S; ++s) {
                int idx = base + s;
                double grad_y = grad_output[n][idx] * gamma[0][c];
                double diff = last_x[n][idx] - mean;
                dL_dvar  += grad_y * diff * (-0.5) * inv_std3;
                dL_dmean += grad_y * (-inv_std);
            }

            // Per-element dx
            for (int s = 0; s < S; ++s) {
                int idx = base + s;
                double grad_y = grad_output[n][idx] * gamma[0][c];
                double diff = last_x[n][idx] - mean;
                double dx_norm = grad_y * inv_std;
                double dx_var  = dL_dvar * 2.0 * diff / static_cast<double>(S);
                double dx_mean = dL_dmean / static_cast<double>(S);
                grad_x[n][idx] = dx_norm + dx_var + dx_mean;
            }
        }
    }

    return grad_x;
}

void InstanceNorm2D::update_weights(double learning_rate) {
    for (int c = 0; c < num_channels_; ++c) {
        gamma[0][c] -= learning_rate * grad_gamma[0][c];
        beta[0][c]  -= learning_rate * grad_beta[0][c];
    }
}

std::vector<Tensor*> InstanceNorm2D::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> InstanceNorm2D::gradients() {
    return {&grad_gamma, &grad_beta};
}

void InstanceNorm2D::zero_grad() {
    grad_gamma.fill(0.0);
    grad_beta.fill(0.0);
}
