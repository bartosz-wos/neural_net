#include "fourier_kan.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// =====================================================================
// FourierKANLayer implementation
//
// Per-edge Fourier-series activation on [-1, 1] (the natural domain of
// {cos(k*pi*x), sin(k*pi*x)}):
//   phi_{i,j}(x) = Σ_{k=1..K} a_{i,j,k} * cos(k*pi*x)
//                       + b_{i,j,k} * sin(k*pi*x)
//
// Parameter layout:
//   a_coefs_, b_coefs_ : (out, in*K) — flat.
//     Index into row i, column j*K + (k-1) for the (i, j, k)-th edge.
//
// Forward (per row b):
//   out[b, i] = Σ_j Σ_k [ a[i, j*K + (k-1)] * cos(k*pi*x[b, j])
//                       + b[i, j*K + (k-1)] * sin(k*pi*x[b, j]) ]
//
// Backward:
//   d_a[i, j*K + (k-1)] += Σ_b grad_out[b, i] * cos(k*pi*x[b, j])
//   d_b[i, j*K + (k-1)] += Σ_b grad_out[b, i] * sin(k*pi*x[b, j])
//   d_input[b, j]      += Σ_i Σ_k grad_out[b, i] *
//                            [ -k*pi*a[i, j*K + (k-1)] * sin(k*pi*x[b, j])
//                             + k*pi*b[i, j*K + (k-1)] * cos(k*pi*x[b, j]) ]
// =====================================================================

FourierKANLayer::FourierKANLayer(size_t in_features, size_t out_features, size_t num_freq)
    : in_features_(in_features), out_features_(out_features), num_freq_(num_freq)
{
    if (in_features == 0)
        throw std::invalid_argument("FourierKANLayer: in_features must be > 0");
    if (out_features == 0)
        throw std::invalid_argument("FourierKANLayer: out_features must be > 0");
    if (num_freq == 0)
        throw std::invalid_argument("FourierKANLayer: num_freq must be > 0");

    size_t n_coefs = in_features * num_freq_;

    // Initialize a and b from small N(0, sigma^2), sigma = sqrt(2/(in*K))
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(2.0 / static_cast<double>(n_coefs));
        std::normal_distribution<double> dist(0.0, scale);
        a_coefs_ = Tensor(out_features_, n_coefs);
        b_coefs_ = Tensor(out_features_, n_coefs);
        for (size_t i = 0; i < a_coefs_.data.size(); ++i) {
            a_coefs_.data[i] = dist(rng);
            b_coefs_.data[i] = dist(rng);
        }
    }

    grad_a_coefs_ = Tensor(out_features_, n_coefs);
    grad_b_coefs_ = Tensor(out_features_, n_coefs);
}

Tensor FourierKANLayer::forward(const Tensor& input) {
    last_input_ = input.clone();
    const size_t B = input.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t K = num_freq_;

    // Cache cos(k*pi*x) and sin(k*pi*x) per (batch, edge, k)
    // Layout: last_cos_(b, i*K_in + j*K + (k-1)) where K_in = in*K
    // We need cos/sin for each (b, j, k) — out_f-independent. We store per
    // (b, j*K + (k-1)) to avoid the i dimension (cos/sin don't depend on i).
    Tensor cos_xp = Tensor(B, in_f * K);   // cos_xp(b, j*K + (k-1)) = cos(k*pi*x[b, j])
    Tensor sin_xp = Tensor(B, in_f * K);   // sin_xp(b, j*K + (k-1)) = sin(k*pi*x[b, j])

    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            double x = input(b, j);
            for (size_t kk = 1; kk <= K; ++kk) {
                double arg = static_cast<double>(kk) * M_PI * x;
                cos_xp(b, j * K + (kk - 1)) = std::cos(arg);
                sin_xp(b, j * K + (kk - 1)) = std::sin(arg);
            }
        }
    }
    last_cos_ = std::move(cos_xp);
    last_sin_ = std::move(sin_xp);

    // Also cache the dphi/dx contribution per (b, j, k) so backward is a clean
    // per-(b, j) sum.
    // dphi_dx_kk = -kk*pi * a[i,j,kk] * sin(...) + kk*pi * b[i,j,kk] * cos(...)
    // This depends on i (through a and b) so we cache per (b, i, j, kk).
    // Layout: last_dphi_dx_(b, i * (in*K) + j*K + (k-1))
    last_dphi_dx_ = Tensor(B, out_f * in_f * K);
    Tensor out(B, out_f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < in_f; ++j) {
                for (size_t kk = 1; kk <= K; ++kk) {
                    double cs = last_cos_(b, j * K + (kk - 1));
                    double sn = last_sin_(b, j * K + (kk - 1));
                    double a = a_coefs_(i, j * K + (kk - 1));
                    double bv = b_coefs_(i, j * K + (kk - 1));
                    sum += a * cs + bv * sn;
                    // dphi/dx = -kk*pi*a*sin + kk*pi*b*cos
                    double dphi_dx = static_cast<double>(kk) * M_PI *
                                     (-a * sn + bv * cs);
                    last_dphi_dx_(b, i * (in_f * K) + j * K + (kk - 1)) = dphi_dx;
                }
            }
            out(b, i) = sum;
        }
    }
    return out;
}

Tensor FourierKANLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t B = last_input_.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t K = num_freq_;

    Tensor grad_input(B, in_f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double g_out = grad_output(b, i);
            for (size_t j = 0; j < in_f; ++j) {
                for (size_t kk = 1; kk <= K; ++kk) {
                    double cs = last_cos_(b, j * K + (kk - 1));
                    double sn = last_sin_(b, j * K + (kk - 1));
                    // d_a[i, j*K + (k-1)] += grad_out[b, i] * cos(...)
                    grad_a_coefs_(i, j * K + (kk - 1)) += g_out * cs;
                    // d_b[i, j*K + (k-1)] += grad_out[b, i] * sin(...)
                    grad_b_coefs_(i, j * K + (kk - 1)) += g_out * sn;
                    // d_input[b, j] += grad_out[b, i] * dphi/dx
                    grad_input(b, j) += g_out *
                        last_dphi_dx_(b, i * (in_f * K) + j * K + (kk - 1));
                }
            }
        }
    }
    return grad_input;
}

void FourierKANLayer::update_weights(double learning_rate) {
    auto sgd = [&](Tensor& w, const Tensor& g) {
        for (size_t i = 0; i < w.data.size(); ++i) {
            w.data[i] -= learning_rate * g.data[i];
        }
    };
    sgd(a_coefs_, grad_a_coefs_);
    sgd(b_coefs_, grad_b_coefs_);
}

std::vector<Tensor*> FourierKANLayer::parameters() {
    return { &a_coefs_, &b_coefs_ };
}

std::vector<Tensor*> FourierKANLayer::gradients() {
    return { &grad_a_coefs_, &grad_b_coefs_ };
}

void FourierKANLayer::zero_grad() {
    grad_a_coefs_.fill(0.0);
    grad_b_coefs_.fill(0.0);
}

// =====================================================================
// FourierKANModel implementation
// =====================================================================

FourierKANModel::FourierKANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
                                 size_t out_dim, size_t num_freq) {
    if (in_dim == 0)
        throw std::invalid_argument("FourierKANModel: in_dim must be > 0");
    if (out_dim == 0)
        throw std::invalid_argument("FourierKANModel: out_dim must be > 0");
    if (num_freq == 0)
        throw std::invalid_argument("FourierKANModel: num_freq must be > 0");
    for (size_t h : hidden_dims) {
        if (h == 0)
            throw std::invalid_argument("FourierKANModel: hidden dim 0 not allowed");
    }

    std::vector<size_t> dims;
    dims.push_back(in_dim);
    for (size_t h : hidden_dims) dims.push_back(h);
    dims.push_back(out_dim);

    for (size_t l = 0; l + 1 < dims.size(); ++l) {
        layers_.push_back(std::make_unique<FourierKANLayer>(dims[l], dims[l + 1], num_freq));
    }
    last_layer_outputs_.resize(layers_.size());
}

Tensor FourierKANModel::forward(const Tensor& input) {
    Tensor cur = input;
    for (size_t l = 0; l < layers_.size(); ++l) {
        cur = layers_[l]->forward(cur);
        last_layer_outputs_[l] = cur.clone();
    }
    return cur;
}

Tensor FourierKANModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor cur_grad = grad_output;
    for (size_t l = layers_.size(); l > 0; --l) {
        size_t idx = l - 1;
        Tensor next_grad = layers_[idx]->backward(cur_grad, learning_rate);
        if (idx == 0) {
            // input gradient — return it.
            return next_grad;
        }
        cur_grad = next_grad;
    }
    return Tensor(0, 0);  // unreachable for non-empty model
}

void FourierKANModel::update_weights(double learning_rate) {
    for (auto& l : layers_) l->update_weights(learning_rate);
}

std::vector<Tensor*> FourierKANModel::parameters() {
    std::vector<Tensor*> all;
    for (auto& l : layers_) {
        auto p = l->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Tensor*> FourierKANModel::gradients() {
    std::vector<Tensor*> all;
    for (auto& l : layers_) {
        auto g = l->gradients();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

void FourierKANModel::zero_grad() {
    for (auto& l : layers_) l->zero_grad();
}