#include "capsnet.h"
#include <cmath>

CapsuleLayer::CapsuleLayer(size_t input_dim, size_t num_capsules,
                             size_t dim_capsule, size_t num_routing)
    : num_capsules_(num_capsules), dim_capsule_(dim_capsule),
      num_routing_(num_routing), input_dim_(input_dim),
      last_v_(1, num_capsules * dim_capsule),
      last_b_(1, num_capsules) {

    for (size_t j = 0; j < num_capsules_; ++j) {
        Tensor Wj(input_dim, dim_capsule_);
        for (size_t i = 0; i < input_dim; ++i)
            for (size_t k = 0; k < dim_capsule_; ++k)
                Wj[i][k] = (rand() / RAND_MAX - 0.5) * 0.1;
        W_.push_back(Wj);
    }
}

Tensor CapsuleLayer::forward(const Tensor& input) {
    size_t batch = input.rows;

    // Compute predicted vectors \hat{u}_j = input · W_j for each output capsule
    Tensor u_hat(batch, num_capsules_ * dim_capsule_);
    for (size_t j = 0; j < num_capsules_; ++j) {
        const Tensor& Wj = W_[j];
        for (size_t b = 0; b < batch; ++b) {
            for (size_t k = 0; k < dim_capsule_; ++k) {
                double sum = 0.0;
                for (size_t i = 0; i < input_dim_; ++i)
                    sum += input[b][i] * Wj[i][k];
                u_hat[b][j * dim_capsule_ + k] = sum;
            }
        }
    }

    // Dynamic routing: coupling coefficients c_ij
    size_t num_input_capsules = 1;
    Tensor c(batch, num_capsules_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < num_capsules_; ++j)
            c[b][j] = 1.0 / num_capsules_;

    for (size_t r = 0; r < num_routing_; ++r) {
        Tensor s(batch, num_capsules_ * dim_capsule_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < num_capsules_; ++j) {
                for (size_t k = 0; k < dim_capsule_; ++k) {
                    double sum = 0.0;
                    for (size_t i = 0; i < num_input_capsules; ++i) {
                        sum += c[b][j]
                             * u_hat[b][j * dim_capsule_ + k];
                    }
                    s[b][j * dim_capsule_ + k] = sum;
                }
            }
        }

        // Squash
        last_v_ = Tensor(batch, num_capsules_ * dim_capsule_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < num_capsules_; ++j) {
                double norm_sq = 0.0;
                for (size_t k = 0; k < dim_capsule_; ++k)
                    norm_sq += s[b][j * dim_capsule_ + k] * s[b][j * dim_capsule_ + k];
                double norm = std::sqrt(norm_sq) + 1e-8;
                double scale = norm_sq / (1.0 + norm_sq);
                for (size_t k = 0; k < dim_capsule_; ++k)
                    last_v_[b][j * dim_capsule_ + k] = scale * s[b][j * dim_capsule_ + k] / norm;
            }
        }

        // Update coupling coefficients
        if (r < num_routing_ - 1) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t j = 0; j < num_capsules_; ++j) {
                    double agreement = 0.0;
                    for (size_t k = 0; k < dim_capsule_; ++k)
                        agreement += u_hat[b][j * dim_capsule_ + k] * last_v_[b][j * dim_capsule_ + k];
                    c[b][j] = c[b][j] * std::exp(agreement);
                }
                double sum_c = 0.0;
                for (size_t j = 0; j < num_capsules_; ++j)
                    sum_c += c[b][j];
                if (sum_c > 0)
                    for (size_t j = 0; j < num_capsules_; ++j)
                        c[b][j] /= sum_c;
            }
        }
    }

    return last_v_;
}

Tensor CapsuleLayer::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void CapsuleLayer::update_weights(double learning_rate) {
    (void)learning_rate;
}

void CapsuleLayer::zero_grad() {}

std::vector<Tensor*> CapsuleLayer::parameters() {
    std::vector<Tensor*> result;
    for (auto& W : W_) result.push_back(&W);
    return result;
}

std::vector<Tensor*> CapsuleLayer::gradients() {
    return {};
}

// === CapsNet ===

CapsNet::CapsNet(size_t input_channels, size_t H, size_t W,
                 size_t num_classes, size_t dim_capsule,
                 size_t primary_dim, size_t primary_channels,
                 size_t num_routing)
    : primary_caps_fc_(primary_channels, primary_dim * 32),
      digit_caps_(primary_dim * 32, num_classes, dim_capsule, num_routing),
      fc1_(num_classes * dim_capsule, 512),
      fc2_(512, 1024),
      fc3_(1024, H * W * input_channels),
      last_capsule_output_(1, 1), dim_capsule_(dim_capsule) {}

Tensor CapsNet::forward(const Tensor& input) {
    last_input_ = input;
    size_t batch = input.rows;

    // PrimaryCaps: flatten + FC + squash
    Tensor x = primary_caps_fc_.forward(input);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double v = x[i][j];
            double norm_sq = v * v;
            x[i][j] = (norm_sq / (1.0 + norm_sq)) * v / (std::sqrt(norm_sq) + 1e-8);
        }
    }

    // DigitCaps via dynamic routing
    last_capsule_output_ = digit_caps_.forward(x);

    // Compute length of each capsule vector as prediction confidence
    size_t dim_caps = digit_caps_.dim_capsule();
    Tensor lengths(batch, last_capsule_output_.cols / dim_caps);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < lengths.cols; ++c) {
            double norm_sq = 0.0;
            for (size_t k = 0; k < dim_caps; ++k)
                norm_sq += last_capsule_output_[b][c * dim_caps + k]
                         * last_capsule_output_[b][c * dim_caps + k];
            lengths[b][c] = std::sqrt(norm_sq);
        }
    }

    return lengths;
}

Tensor CapsNet::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void CapsNet::update_weights(double learning_rate) {
    primary_caps_fc_.update_weights(learning_rate);
    digit_caps_.update_weights(learning_rate);
    fc1_.update_weights(learning_rate);
    fc2_.update_weights(learning_rate);
    fc3_.update_weights(learning_rate);
}

void CapsNet::zero_grad() {
    primary_caps_fc_.zero_grad();
    digit_caps_.zero_grad();
    fc1_.zero_grad();
    fc2_.zero_grad();
    fc3_.zero_grad();
}

std::vector<Tensor*> CapsNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : primary_caps_fc_.parameters()) result.push_back(p);
    for (Tensor* p : digit_caps_.parameters()) result.push_back(p);
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    for (Tensor* p : fc3_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> CapsNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : primary_caps_fc_.gradients()) result.push_back(g);
    for (Tensor* g : digit_caps_.gradients()) result.push_back(g);
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    for (Tensor* g : fc3_.gradients()) result.push_back(g);
    return result;
}

double CapsNet::reconstruction_loss(const Tensor& input,
                                     const Tensor& digit_capsules,
                                     size_t correct_label) {
    (void)input; (void)digit_capsules; (void)correct_label;
    return 0.0;
}