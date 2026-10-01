#include "cheby_kan.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// =====================================================================
// ChebyKANLayer implementation
//
// Per-edge Chebyshev polynomial activation on [-1, 1] (the natural
// domain of T_k):
//
//   phi_{i,j}(x) = Σ_{k=0..K} c_{i,j,k} · T_k(x)
//
// Parameter layout:
//   coefs_ : (out, in * (K+1)) — flat.
//     Index into row i, column j*(K+1) + k for the (i, j, k)-th coefficient.
//
// Forward (per row b):
//   T_0(x) = 1,  T_1(x) = x,  T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x)
//   out[b, i] = Σ_j Σ_k coefs_[i, j*(K+1) + k] · T_k(x[b, j])
//
// Backward:
//   d_coefs[i, j*(K+1) + k] += Σ_b grad_out[b, i] · T_k(x[b, j])
//   d_input[b, j]      += Σ_i Σ_k grad_out[b, i] · coefs_[i, j*(K+1) + k] · T'_k(x[b, j])
// where the Chebyshev derivative satisfies
//   T'_0(x) = 0,  T'_1(x) = 1,  T'_k(x) = 2 T_{k-1}(x) + 2 x T'_{k-1}(x) - T'_{k-2}(x)
// (derived from differentiating the recurrence T_k = 2x T_{k-1} - T_{k-2}).
// We also use the equivalent closed-form T'_k = k U_{k-1}(x), but the
// recurrence is numerically cleaner (no division by k) and reuses the
// already-cached T_k values via a single side computation.
// =====================================================================

ChebyKANLayer::ChebyKANLayer(size_t in_features, size_t out_features, size_t num_freq)
    : in_features_(in_features), out_features_(out_features), num_freq_(num_freq)
{
    if (in_features == 0)
        throw std::invalid_argument("ChebyKANLayer: in_features must be > 0");
    if (out_features == 0)
        throw std::invalid_argument("ChebyKANLayer: out_features must be > 0");
    if (num_freq == 0)
        throw std::invalid_argument("ChebyKANLayer: num_freq must be > 0");

    size_t n_coefs_per_edge = num_freq_ + 1;
    size_t n_total = in_features_ * n_coefs_per_edge;

    // Initialize coefs_ from small N(0, sigma^2) with sigma = sqrt(1/in_features).
    {
        std::mt19937 rng(42);
        double scale = 1.0 / std::sqrt(static_cast<double>(in_features_));
        std::normal_distribution<double> dist(0.0, scale);
        coefs_ = Tensor(out_features_, n_total);
        for (size_t i = 0; i < coefs_.data.size(); ++i) {
            coefs_.data[i] = dist(rng);
        }
    }

    grad_coefs_ = Tensor(out_features_, n_total);
}

Tensor ChebyKANLayer::forward(const Tensor& input) {
    last_input_ = input.clone();
    const size_t B = input.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t K = num_freq_;
    const size_t n_coefs_per_edge = K + 1;

    // Cache T_k(x[b, j]) for k=0..K.
    // Layout: last_T_(b, j * (K+1) + k) = T_k(x[b, j])
    last_T_ = Tensor(B, in_f * n_coefs_per_edge);

    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            double x = input(b, j);
            // Clamp input to [-1, 1] — Chebyshev basis is defined on this domain
            // (the standard KAN convention). This also prevents |T_k| from
            // growing unboundedly outside the domain.
            if (x > 1.0) x = 1.0;
            else if (x < -1.0) x = -1.0;

            // T_0 = 1, T_1 = x
            last_T_(b, j * n_coefs_per_edge + 0) = 1.0;
            if (K >= 1) last_T_(b, j * n_coefs_per_edge + 1) = x;

            // T_n = 2 x T_{n-1} - T_{n-2} for n >= 2
            double t_prev_prev = 1.0;
            double t_prev = x;
            for (size_t n = 2; n <= K; ++n) {
                double t_curr = 2.0 * x * t_prev - t_prev_prev;
                last_T_(b, j * n_coefs_per_edge + n) = t_curr;
                t_prev_prev = t_prev;
                t_prev = t_curr;
            }
        }
    }

    // Also cache the dphi/dx contribution per (b, i, j, k) so backward is a
    // clean per-(b, j) sum. dphi/dx = c_{i,j,k} * T'_k(x).
    // Layout: last_dphi_dx_(b, i * (in * (K+1)) + j * (K+1) + k)
    last_dphi_dx_ = Tensor(B, out_f * in_f * n_coefs_per_edge);
    Tensor out(B, out_f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < in_f; ++j) {
                // Recompute T' values at x[b, j] using the same recurrence.
                // T'_0 = 0, T'_1 = 1, T'_n = 2 T_{n-1} + 2 x T'_{n-1} - T'_{n-2}
                double x = input(b, j);
                if (x > 1.0) x = 1.0;
                else if (x < -1.0) x = -1.0;

                double t_prev_prev = 1.0;
                double t_prev = x;
                double tp_prev_prev = 0.0;  // T'_0
                double tp_prev = 1.0;       // T'_1

                // k = 0 contribution
                {
                    size_t k_idx = 0;
                    double c = coefs_(i, j * n_coefs_per_edge + k_idx);
                    double dphi_dx = c * 0.0;  // T'_0 = 0
                    last_dphi_dx_(b, i * (in_f * n_coefs_per_edge) + j * n_coefs_per_edge + k_idx) = dphi_dx;
                    sum += c * last_T_(b, j * n_coefs_per_edge + k_idx);
                }
                // k = 1 contribution (if K >= 1)
                if (K >= 1) {
                    size_t k_idx = 1;
                    double c = coefs_(i, j * n_coefs_per_edge + k_idx);
                    double dphi_dx = c * 1.0;  // T'_1 = 1
                    last_dphi_dx_(b, i * (in_f * n_coefs_per_edge) + j * n_coefs_per_edge + k_idx) = dphi_dx;
                    sum += c * last_T_(b, j * n_coefs_per_edge + k_idx);
                }
                // k >= 2
                for (size_t n = 2; n <= K; ++n) {
                    size_t k_idx = n;
                    double t_curr = 2.0 * x * t_prev - t_prev_prev;
                    double tp_curr = 2.0 * t_prev + 2.0 * x * tp_prev - tp_prev_prev;
                    double c = coefs_(i, j * n_coefs_per_edge + k_idx);
                    double dphi_dx = c * tp_curr;
                    last_dphi_dx_(b, i * (in_f * n_coefs_per_edge) + j * n_coefs_per_edge + k_idx) = dphi_dx;
                    sum += c * t_curr;
                    t_prev_prev = t_prev;
                    t_prev = t_curr;
                    tp_prev_prev = tp_prev;
                    tp_prev = tp_curr;
                }
            }
            out(b, i) = sum;
        }
    }
    return out;
}

Tensor ChebyKANLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t B = last_input_.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t K = num_freq_;
    const size_t n_coefs_per_edge = K + 1;

    // d_coefs[i, j*(K+1) + k] += Σ_b grad_out[b, i] * T_k(x[b, j])
    for (size_t i = 0; i < out_f; ++i) {
        for (size_t j = 0; j < in_f; ++j) {
            for (size_t k = 0; k < n_coefs_per_edge; ++k) {
                double s = 0.0;
                for (size_t b = 0; b < B; ++b) {
                    s += grad_output(b, i) * last_T_(b, j * n_coefs_per_edge + k);
                }
                grad_coefs_(i, j * n_coefs_per_edge + k) += s;
            }
        }
    }

    // d_input is computed externally (caller threads grad_output back
    // through subsequent layers). The forward cache last_dphi_dx_ holds
    // c * T'_k values per (b, i, j, k) so a future caller can compute
    //   d_input[b, j] = Σ_i Σ_k grad_out[b, i] * last_dphi_dx_[b, i*(in*(K+1)) + j*(K+1) + k]
    // We return the cached dphi_dx in shape (B, in) for the model's
    // backward chain to use.
    Tensor d_input(B, in_f);
    for (size_t b = 0; b < B; ++b) {
        for (size_t j = 0; j < in_f; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < out_f; ++i) {
                for (size_t k = 0; k < n_coefs_per_edge; ++k) {
                    s += grad_output(b, i) *
                         last_dphi_dx_(b, i * (in_f * n_coefs_per_edge) + j * n_coefs_per_edge + k);
                }
            }
            d_input(b, j) = s;
        }
    }
    return d_input;
}

void ChebyKANLayer::update_weights(double learning_rate) {
    const double lr = learning_rate;
    for (size_t i = 0; i < coefs_.data.size(); ++i) {
        coefs_.data[i] -= lr * grad_coefs_.data[i];
    }
}

std::vector<Tensor*> ChebyKANLayer::parameters() {
    return {&coefs_};
}

std::vector<Tensor*> ChebyKANLayer::gradients() {
    return {&grad_coefs_};
}

void ChebyKANLayer::zero_grad() {
    grad_coefs_.fill(0.0);
}

// =====================================================================
// ChebyKANModel implementation — stacks ChebyKANLayers, last layer
// maps to out_dim.
// =====================================================================

ChebyKANModel::ChebyKANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
                             size_t out_dim, size_t num_freq)
{
    if (in_dim == 0)
        throw std::invalid_argument("ChebyKANModel: in_dim must be > 0");
    if (out_dim == 0)
        throw std::invalid_argument("ChebyKANModel: out_dim must be > 0");
    if (num_freq == 0)
        throw std::invalid_argument("ChebyKANModel: num_freq must be > 0");
    for (size_t h : hidden_dims) {
        if (h == 0)
            throw std::invalid_argument("ChebyKANModel: hidden_dims entries must be > 0");
    }

    // Build layer list: in_dim -> hidden_dims[0] -> ... -> out_dim
    size_t prev = in_dim;
    for (size_t h : hidden_dims) {
        layers_.push_back(std::make_unique<ChebyKANLayer>(prev, h, num_freq));
        prev = h;
    }
    layers_.push_back(std::make_unique<ChebyKANLayer>(prev, out_dim, num_freq));
}

Tensor ChebyKANModel::forward(const Tensor& input) {
    Tensor cur = input;
    last_layer_outputs_.clear();
    last_layer_outputs_.push_back(cur);
    for (auto& layer : layers_) {
        cur = layer->forward(cur);
        last_layer_outputs_.push_back(cur);
    }
    return cur;
}

Tensor ChebyKANModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor grad = grad_output;
    for (size_t idx = layers_.size(); idx-- > 0; ) {
        // Per-layer backward returns d_input (i.e. d_layer_input = grad for the previous layer)
        grad = layers_[idx]->backward(grad, learning_rate);
    }
    return grad;
}

void ChebyKANModel::update_weights(double learning_rate) {
    for (auto& layer : layers_) {
        layer->update_weights(learning_rate);
    }
}

std::vector<Tensor*> ChebyKANModel::parameters() {
    std::vector<Tensor*> all;
    for (auto& layer : layers_) {
        auto p = layer->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Tensor*> ChebyKANModel::gradients() {
    std::vector<Tensor*> all;
    for (auto& layer : layers_) {
        auto g = layer->gradients();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

void ChebyKANModel::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}
