#include "fourier_features.h"
#include <cmath>
#include <algorithm>

FourierFeatures::FourierFeatures(size_t input_dim, size_t num_frequencies, double sigma)
    : input_dim_(input_dim), num_frequencies_(num_frequencies), sigma_(sigma),
      frequencies_(num_frequencies, input_dim),
      grad_frequencies_(num_frequencies, input_dim),
      last_coords_(0, 0), last_encoded_(0, 0) {}

// Forward: coords is (N, input_dim), output is (N, 2*num_frequencies)
Tensor FourierFeatures::forward(const Tensor& coords) {
    size_t N = coords.rows;
    size_t F = num_frequencies_;
    last_coords_ = coords;
    last_encoded_ = Tensor(N, 2 * F);

    const double* freq_data = frequencies_.data.data();
    double* out_data = last_encoded_.data.data();

    for (size_t n = 0; n < N; ++n) {
        for (size_t f = 0; f < F; ++f) {
            // Compute dot product b_f^T x
            double dot = 0.0;
            for (size_t d = 0; d < input_dim_; ++d) {
                dot += freq_data[f * input_dim_ + d] * coords[n][d];
            }
            double angle = 2.0 * M_PI * dot;
            out_data[n * (2 * F) + f]         = std::cos(angle);  // cos
            out_data[n * (2 * F) + F + f]     = std::sin(angle);  // sin
        }
    }

    return last_encoded_;
}

// Backward: grad_output is (N, 2*F)
// d/dx [cos(2π b^T x)] = -2π sin(2π b^T x) * b
// d/dx [sin(2π b^T x)] =  2π cos(2π b^T x) * b
// grad_frequencies_: for each sample n and frequency f, accumulate
//   dL/df_f += sum over outputs of dL/dcos/sin * d(cos/sin)/df
//   d(cos)/df = -2π x * sin(2π b^T x)
//   d(sin)/df =  2π x * cos(2π b^T x)
Tensor FourierFeatures::backward(const Tensor& grad_output, double /*learning_rate*/) {
    size_t N = grad_output.rows;
    size_t F = num_frequencies_;

    // Gradient w.r.t. frequencies (accumulate across samples and cos/sin parts)
    grad_frequencies_.fill(0.0);
    double* grad_freq_data = grad_frequencies_.data.data();
    const double* freq_data = frequencies_.data.data();

    // grad w.r.t. input coordinates
    Tensor grad_input(N, input_dim_);

    for (size_t n = 0; n < N; ++n) {
        for (size_t f = 0; f < F; ++f) {
            double dot = 0.0;
            for (size_t d = 0; d < input_dim_; ++d) {
                dot += freq_data[f * input_dim_ + d] * last_coords_[n][d];
            }
            double angle = 2.0 * M_PI * dot;
            double cos_val = std::cos(angle);
            double sin_val = std::sin(angle);

            double grad_cos = grad_output[n][f];
            double grad_sin = grad_output[n][F + f];

            // dL/df_f for each input dimension d: accumulate from both cos and sin paths
            // d(cos)/df_d = -2π * x_d * sin(angle)
            // d(sin)/df_d =  2π * x_d * cos(angle)
            for (size_t d = 0; d < input_dim_; ++d) {
                double x_d = last_coords_[n][d];
                double dcos_df = -2.0 * M_PI * x_d * sin_val;
                double dsin_df =  2.0 * M_PI * x_d * cos_val;
                grad_freq_data[f * input_dim_ + d] += grad_cos * dcos_df + grad_sin * dsin_df;
            }

            // Gradient w.r.t. input coordinates x
            // dL/dx_d = sum_f (dL/dcos_f * dcos/dx_d + dL/dsin_f * dsin/dx_d)
            // dcos/dx_d = -2π * b_f[d] * sin(angle)
            // dsin/dx_d =  2π * b_f[d] * cos(angle)
            for (size_t d = 0; d < input_dim_; ++d) {
                double b_fd = freq_data[f * input_dim_ + d];
                double dcos_dx = -2.0 * M_PI * b_fd * sin_val;
                double dsin_dx =  2.0 * M_PI * b_fd * cos_val;
                grad_input[n][d] += grad_cos * dcos_dx + grad_sin * dsin_dx;
            }
        }
    }

    return grad_input;
}

void FourierFeatures::zero_grad() {
    grad_frequencies_.fill(0.0);
}

std::vector<Tensor*> FourierFeatures::parameters() {
    return {&frequencies_};
}

std::vector<Tensor*> FourierFeatures::gradients() {
    return {&grad_frequencies_};
}

// GaussianFourierFeatures: sample B from N(0, sigma^2 I)
GaussianFourierFeatures::GaussianFourierFeatures(size_t input_dim, size_t num_frequencies, double sigma)
    : FourierFeatures(input_dim, num_frequencies, sigma)
{
    std::normal_distribution<double> dist(0.0, sigma);
    for (size_t f = 0; f < num_frequencies_; ++f) {
        for (size_t d = 0; d < input_dim_; ++d) {
            frequencies_.data[f * input_dim_ + d] = dist(rng_);
        }
    }
}

// LearnedFourierFeatures: initialize frequencies randomly from N(0, sigma^2 I), sigma=1.0
LearnedFourierFeatures::LearnedFourierFeatures(size_t input_dim, size_t num_frequencies)
    : FourierFeatures(input_dim, num_frequencies, 1.0)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t f = 0; f < num_frequencies_; ++f) {
        for (size_t d = 0; d < input_dim_; ++d) {
            frequencies_.data[f * input_dim_ + d] = dist(rng_);
        }
    }
}