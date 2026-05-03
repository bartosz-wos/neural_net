#include "spectral_norm.h"
#include <cmath>
#include <stdexcept>

SpectralNorm::SpectralNorm(const Tensor& W, int power_iterations, double eps)
    : gamma(Tensor(1, 1)),           // scalar, initialized to 1
      u(Tensor(W.rows, 1)),          // left singular vector (out_features, 1)
      v(Tensor(W.cols, 1)),          // right singular vector (in_features, 1)
      last_W(W.rows, W.cols),
      W(W),
      power_iterations(power_iterations),
      eps(eps)
{
    if (W.rows == 0 || W.cols == 0) {
        throw std::invalid_argument("SpectralNorm: weight matrix must be non-empty");
    }

    // Initialize gamma to 1 (scalar)
    gamma[0][0] = 1.0;

    // Initialize u and v with random vectors (normalized)
    // Using a simple random init for the power iteration starting vectors
    for (size_t i = 0; i < u.rows; ++i) {
        u[i][0] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    for (size_t i = 0; i < v.rows; ++i) {
        v[i][0] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Normalize initial u
    double u_norm = 0.0;
    for (size_t i = 0; i < u.rows; ++i)
        u_norm += u[i][0] * u[i][0];
    u_norm = std::sqrt(u_norm);
    if (u_norm > eps) {
        for (size_t i = 0; i < u.rows; ++i)
            u[i][0] /= u_norm;
    }

    // Normalize initial v
    double v_norm = 0.0;
    for (size_t i = 0; i < v.rows; ++i)
        v_norm += v[i][0] * v[i][0];
    v_norm = std::sqrt(v_norm);
    if (v_norm > eps) {
        for (size_t i = 0; i < v.rows; ++i)
            v[i][0] /= v_norm;
    }
}

double SpectralNorm::compute_spectral_norm() {
    // Power iteration to find the largest singular value
    // u = normalize(W @ v)
    // v = normalize(W.T @ u)
    // sigma = ||W @ v|| (or equivalently ||W.T @ u||)
    // We do `power_iterations` full iterations; after the last, we compute the norm

    for (int iter = 0; iter < power_iterations; ++iter) {
        // v_new = normalize(W.T @ u)
        // W.T is (in_features, out_features), u is (out_features, 1)
        // Result v_new is (in_features, 1)
        Tensor v_new(v.rows, 1);
        for (size_t j = 0; j < W.cols; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < W.rows; ++i) {
                sum += W[i][j] * u[i][0];  // W[i][j] = W[j][i]^T
            }
            v_new[j][0] = sum;
        }

        // Normalize v_new
        double v_norm = 0.0;
        for (size_t i = 0; i < v_new.rows; ++i)
            v_norm += v_new[i][0] * v_new[i][0];
        v_norm = std::sqrt(v_norm);
        if (v_norm > eps) {
            for (size_t i = 0; i < v_new.rows; ++i)
                v[i][0] = v_new[i][0] / v_norm;
        }

        // u_new = normalize(W @ v)
        // W is (out_features, in_features), v is (in_features, 1)
        // Result u_new is (out_features, 1)
        Tensor u_new(u.rows, 1);
        for (size_t i = 0; i < W.rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < W.cols; ++j) {
                sum += W[i][j] * v[j][0];
            }
            u_new[i][0] = sum;
        }

        // Normalize u_new
        double u_norm = 0.0;
        for (size_t i = 0; i < u_new.rows; ++i)
            u_norm += u_new[i][0] * u_new[i][0];
        u_norm = std::sqrt(u_norm);
        if (u_norm > eps) {
            for (size_t i = 0; i < u_new.rows; ++i)
                u[i][0] = u_new[i][0] / u_norm;
        }
    }

    // After power iterations, compute sigma = ||W @ v|| / ||u||
    // Since u is normalized to 1, sigma = ||W @ v||
    // Actually, we want sigma = u^T @ W @ v / ||u|| / ||v||
    // If both u and v are normalized (||u||=||v||=1), then:
    // sigma = u^T @ W @ v
    // Which equals ||W @ v|| when u = normalize(W @ v)
    // And sigma = ||W @ v|| since u is normalized to length 1
    double sigma = 0.0;
    for (size_t i = 0; i < W.rows; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < W.cols; ++j) {
            dot += W[i][j] * v[j][0];
        }
        sigma += u[i][0] * dot;  // u[i][0] is the normalized left singular vector
    }
    return std::abs(sigma);
}

Tensor SpectralNorm::normalize(const Tensor& vec, double eps) {
    Tensor result(vec.rows, vec.cols);
    double norm = 0.0;
    for (size_t i = 0; i < vec.rows; ++i) {
        for (size_t j = 0; j < vec.cols; ++j) {
            norm += vec.data[i * vec.cols + j] * vec.data[i * vec.cols + j];
        }
    }
    norm = std::sqrt(norm);
    if (norm > eps) {
        for (size_t i = 0; i < vec.rows; ++i) {
            for (size_t j = 0; j < vec.cols; ++j) {
                result.data[i * vec.cols + j] = vec.data[i * vec.cols + j] / norm;
            }
        }
    } else {
        result = vec;
    }
    return result;
}

Tensor SpectralNorm::forward(const Tensor& input) {
    // Compute spectral norm of W
    double sigma = compute_spectral_norm();
    if (sigma < eps) sigma = eps;

    // W_normalized = gamma * W / sigma
    // gamma is a scalar (stored as 1x1 tensor)
    double gamma_val = gamma[0][0];

    last_W = Tensor(W.rows, W.cols);
    for (size_t i = 0; i < W.rows; ++i) {
        for (size_t j = 0; j < W.cols; ++j) {
            last_W[i][j] = gamma_val * W[i][j] / sigma;
        }
    }

    // y = input @ W_normalized^T
    // last_W is (out_features, in_features), so last_W^T is (in_features, out_features)
    // input is (batch, in_features)
    // output is (batch, out_features)
    Tensor W_T = last_W.transpose();
    Tensor output = input * W_T;
    return output;
}

Tensor SpectralNorm::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (batch, out_features)
    // last_W: (out_features, in_features), last_W^T: (in_features, out_features)
    // grad_input = grad_output @ last_W
    // grad_input: (batch, in_features)
    Tensor grad_input = grad_output * last_W;
    return grad_input;
}

void SpectralNorm::update_weights(double /* learning_rate */) {
    // SpectralNorm doesn't have its own weights to update.
    // The original layer's weights are updated elsewhere.
    // We just re-normalize after the weight update.
    // This is called after the wrapped layer's update_weights.
}

std::vector<Tensor*> SpectralNorm::parameters() {
    return {&gamma};
}

std::vector<Tensor*> SpectralNorm::gradients() {
    return {};  // gamma gradient is computed during backward (if needed)
}

void SpectralNorm::zero_grad() {
    // No persistent gradients in SpectralNorm
}