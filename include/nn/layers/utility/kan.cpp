#include "kan.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// =====================================================================
// KANLayer implementation
// =====================================================================

KANLayer::KANLayer(size_t in_features, size_t out_features,
                   size_t num_grids, size_t spline_order,
                   double grid_low, double grid_high)
    : in_features_(in_features),
      out_features_(out_features),
      num_grids_(num_grids),
      spline_order_(spline_order),
      n_coefs_(num_grids + spline_order),
      n_knots_(num_grids + 2 * spline_order)  // canonical: n_coefs + k = G + 2k
{
    if (in_features == 0 || out_features == 0)
        throw std::invalid_argument("KANLayer: in/out features must be > 0");
    if (spline_order < 1)
        throw std::invalid_argument("KANLayer: spline_order must be >= 1");
    if (num_grids < 1)
        throw std::invalid_argument("KANLayer: num_grids must be >= 1");
    if (grid_high <= grid_low)
        throw std::invalid_argument("KANLayer: grid_high must be > grid_low");

    // Canonical clamped-uniform B-spline knot vector (NURBS Book convention):
    //   n_coefs = G + k control points, degree p = k - 1
    //   n_knots = n_coefs + k = G + 2k
    //   t_0..t_{k-1} = a (k copies at left boundary)
    //   t_k..t_{k+G-1} = a+h, a+2h, ..., a+G*h = b (G interior knots from a to b inclusive)
    //   t_{k+G}..t_{k+G+k-1} = b (k copies at right boundary)
    // Note: t_{k+G-1} = b and t_{k+G} = b coincide, giving k+1 copies of b at the right.
    // This is the standard convention; the basis sums to 1 (partition of unity) over [a, b].

    // Initialize learnable parameters with small random values (PyTorch KAN init scheme).
    // spline_coefs: shape (out, in * n_coefs), init from N(0, sigma^2) with sigma from Kaiming.
    {
        std::mt19937 rng(42);
        double scale = std::sqrt(2.0 / static_cast<double>(in_features * n_coefs_));
        std::normal_distribution<double> dist(0.0, scale);
        spline_coefs_ = Tensor(out_features, in_features * n_coefs_);
        for (auto& v : spline_coefs_.data) v = dist(rng);
    }
    // base_weight: shape (out, in), init from N(0, sigma^2) — small.
    {
        std::mt19937 rng(43);
        double scale = std::sqrt(2.0 / static_cast<double>(in_features));
        std::normal_distribution<double> dist(0.0, scale);
        base_weight_ = Tensor(out_features, in_features);
        for (auto& v : base_weight_.data) v = dist(rng);
    }
    // spline_weight: shape (out, in), init to 1.0 (spline term dominates initially)
    spline_weight_ = Tensor(out_features, in_features);
    spline_weight_.fill(1.0);

    // Build clamped uniform knot grid (canonical NURBS convention):
    //   t_0..t_{k-1}            = grid_low   (k copies at left boundary)
    //   t_{k+i} for i in 0..G-1  = grid_low + (i+1)*h  (G interior knots from a+h to b inclusive)
    //   t_{k+G}..t_{k+2k-1}     = grid_high  (k copies at right boundary)
    // Total: k + G + k = G + 2k = n_knots.
    // Note: t_{k+G-1} = grid_high = b coincides with t_{k+G}, so we have k+1 copies of b at the right.
    grid_ = Tensor(1, n_knots_);
    double h = (grid_high - grid_low) / static_cast<double>(num_grids);
    size_t k = spline_order_;
    for (size_t i = 0; i < k; ++i)        grid_.data[i] = grid_low;
    for (size_t i = 0; i < num_grids; ++i) grid_.data[k + i] = grid_low + static_cast<double>(i + 1) * h;
    for (size_t i = 0; i < k; ++i)        grid_.data[k + num_grids + i] = grid_high;

    // Initialize gradient tensors (zero-filled)
    grad_spline_coefs_ = Tensor(out_features, in_features * n_coefs_);
    grad_base_weight_ = Tensor(out_features, in_features);
    grad_spline_weight_ = Tensor(out_features, in_features);
}

namespace kan_detail {

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline double silu(double x) {
    if (x > 0.0) return x / (1.0 + std::exp(-x));
    // Numerically stable alternative for x < 0
    double ex = std::exp(x);
    return x * ex / (1.0 + ex);
}
// silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
inline double silu_deriv(double x) {
    double s = (x > 0.0) ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
    return s * (1.0 + x * (1.0 - s));
}

// Evaluate a clamped B-spline of order k (degree k-1) with the given knot vector
// and coefficients at point x.
// Returns the spline value sum_{i=0}^{n_coefs-1} c_i * B_{i,k}(x).
//
// Uses Cox-de Boor recursion (Algorithm A2.1 from The NURBS Book, but written
// as a clear nested-loop implementation that doesn't rely on subtle working-memory
// bookkeeping — correctness first, speed second). For typical KAN sizes (k <= 8,
// n_coefs <= 32) this is plenty fast.
inline double bspline_eval(double x, const double* knots, size_t n_knots,
                          const double* coefs, size_t n_coefs, size_t k) {
    // Special case: x outside the knot range — clamp to boundary coefficient
    if (x <= knots[0]) return coefs[0];
    if (x >= knots[n_knots - 1]) return coefs[n_coefs - 1];

    // Compute all B_{c, k}(x) for c = 0..n_coefs using Cox-de Boor.
    // N_prev[c] = B_{c, p}(x); we iterate p = 0..k-1.
    // Allocate on stack — k is bounded.
    double N_prev[32];
    double N_curr[32];
    if (n_coefs > 32 || k > 32) return 0.0; // sanity; not expected for KAN
    for (size_t c = 0; c < n_coefs; ++c) N_prev[c] = 0.0;

    // p = 0: N_{c, 0}(x) = 1 iff knots[c] <= x < knots[c+1], else 0
    for (size_t c = 0; c < n_coefs; ++c) {
        if (c + 1 < n_knots && knots[c] <= x && x < knots[c + 1]) {
            N_prev[c] = 1.0;
        }
    }

    // p = 1..k-1
    for (size_t p = 1; p < k; ++p) {
        for (size_t c = 0; c < n_coefs; ++c) {
            double left = 0.0;
            double right = 0.0;
            if (c + p < n_knots && knots[c + p] != knots[c]) {
                left = (x - knots[c]) / (knots[c + p] - knots[c]) * N_prev[c];
            }
            if (c + p + 1 < n_knots && knots[c + p + 1] != knots[c + 1]) {
                right = (knots[c + p + 1] - x) / (knots[c + p + 1] - knots[c + 1]) * N_prev[c + 1];
            }
            N_curr[c] = left + right;
        }
        // Copy N_curr → N_prev for the next iteration.
        for (size_t c = 0; c < n_coefs; ++c) N_prev[c] = N_curr[c];
    }

    // Sum coefs * basis
    double s = 0.0;
    for (size_t c = 0; c < n_coefs; ++c) s += coefs[c] * N_prev[c];
    return s;
}

// Evaluate a single B-spline basis function B_{c, k}(x) at point x, returning
// its value. Used for the spline_coef gradient.
inline double bspline_basis(double x, const double* knots, size_t n_knots,
                            size_t n_coefs, size_t k, size_t c_target) {
    if (x <= knots[0]) return (c_target == 0) ? 1.0 : 0.0;
    if (x >= knots[n_knots - 1]) return (c_target == n_coefs - 1) ? 1.0 : 0.0;

    double N_prev[32];
    double N_curr[32];
    if (n_coefs > 32 || k > 32) return 0.0;
    for (size_t c = 0; c < n_coefs; ++c) N_prev[c] = 0.0;

    // p = 0
    for (size_t c = 0; c < n_coefs; ++c) {
        if (c + 1 < n_knots && knots[c] <= x && x < knots[c + 1]) N_prev[c] = 1.0;
    }

    // p = 1..k-1
    for (size_t p = 1; p < k; ++p) {
        for (size_t c = 0; c < n_coefs; ++c) {
            double left = 0.0, right = 0.0;
            if (c + p < n_knots && knots[c + p] != knots[c]) {
                left = (x - knots[c]) / (knots[c + p] - knots[c]) * N_prev[c];
            }
            if (c + p + 1 < n_knots && knots[c + p + 1] != knots[c + 1]) {
                right = (knots[c + p + 1] - x) / (knots[c + p + 1] - knots[c + 1]) * N_prev[c + 1];
            }
            N_curr[c] = left + right;
        }
        for (size_t c = 0; c < n_coefs; ++c) N_prev[c] = N_curr[c];
    }
    return N_prev[c_target];
}

} // namespace kan_detail

Tensor KANLayer::forward(const Tensor& input) {
    using namespace kan_detail;
    last_input_ = input.clone();
    const size_t B = input.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t k = spline_order_;
    const size_t nc = n_coefs_;

    last_phi_ = Tensor(B, out_f * in_f);
    last_base_val_ = Tensor(B, out_f * in_f);
    last_spline_val_ = Tensor(B, out_f * in_f);
    last_pre_act_ = Tensor(B, out_f);

    Tensor out(B, out_f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double sum_phi = 0.0;
            for (size_t j = 0; j < in_f; ++j) {
                double x = input(b, j);
                // base term: base_weight[i,j] * silu(x)
                double base_v = base_weight_(i, j) * silu(x);
                // spline term: spline_weight[i,j] * spline(coefs_for_edge, x)
                const double* edge_coefs = spline_coefs_.data.data() + (i * in_f + j) * nc;
                double spline_v_raw = bspline_eval(x, grid_.data.data(), n_knots_,
                                                   edge_coefs, nc, k);
                double spline_v = spline_weight_(i, j) * spline_v_raw;
                double phi = base_v + spline_v;
                last_phi_(b, i * in_f + j) = phi;
                last_base_val_(b, i * in_f + j) = base_v;
                last_spline_val_(b, i * in_f + j) = spline_v;
                sum_phi += phi;
            }
            out(b, i) = sum_phi;
            last_pre_act_(b, i) = sum_phi;
        }
    }
    return out;
}

Tensor KANLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    using namespace kan_detail;
    const size_t B = last_input_.rows;
    const size_t in_f = in_features_;
    const size_t out_f = out_features_;
    const size_t k = spline_order_;
    const size_t nc = n_coefs_;

    // Accumulate gradients into grad_*_ tensors (we use += so multi-call backward works).
    // grad_output: (B, out_f)
    Tensor grad_input(B, in_f);

    // For each batch element, distribute grad_output across phi entries, then
    // accumulate parameter gradients.
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < out_f; ++i) {
            double g_out = grad_output(b, i);
            for (size_t j = 0; j < in_f; ++j) {
                double d_phi = g_out; // dL/dphi = dL/dout (one phi contributes to one out)

                // grad_base_weight[i, j] += d_phi * silu(x)
                double x = last_input_(b, j);
                grad_base_weight_(i, j) += d_phi * silu(x);

                // grad_spline_weight[i, j] += d_phi * spline_raw(x[b, j])
                const double* edge_coefs = spline_coefs_.data.data() + (i * in_f + j) * nc;
                double spline_raw = bspline_eval(x, grid_.data.data(), n_knots_,
                                                  edge_coefs, nc, k);
                grad_spline_weight_(i, j) += d_phi * spline_raw;

                // grad_spline_coefs[i, j*nc + c] += d_phi * spline_weight[i, j] * B_{c,k}(x)
                double sw = spline_weight_(i, j);
                for (size_t c = 0; c < nc; ++c) {
                    double B_c = bspline_basis(x, grid_.data.data(), n_knots_, nc, k, c);
                    grad_spline_coefs_(i, j * nc + c) += d_phi * sw * B_c;
                }

                // grad_input[b, j] += d_phi * (base_weight[i, j] * silu'(x) + spline_weight[i, j] * spline'(x))
                // spline'(x) = sum_c coefs[c] * B'_{c, k}(x). We compute it via finite differences
                // for simplicity and robustness. The relative error is small at the grid resolution.
                // Note: gradient is via the analytical chain — for proper BPTT, we should use the
                // analytical B-spline derivative. Numerical derivative works for verifying correctness
                // but adds a tiny amount of O(eps^2) noise. To keep things exact, we use a closed-form
                // spline derivative: spline'(x) = (k-1) * sum_c (coefs[c+1] - coefs[c]) / (knots[c+k] - knots[c+1]) * B_{c, k-1}(x)
                // implemented below.
                double spline_deriv = 0.0;
                if (k >= 2) {
                    // Spline derivative (NURBS Book Theorem 3.4 with degree p = k-1):
                    //   spline'(x) = sum_{c=0..nc-2} (coefs[c+1] - coefs[c]) * (k-1) / (knots[c+k] - knots[c+1]) * B_{c, k-1}(x)
                    // where B_{c, k-1} is a degree-(k-2) B-spline on the interior knot vector
                    // (knots[1], knots[2], ..., knots[n_knots-1]) — that's n_knots - 1 knots total.
                    // CRITICAL: the denominator is knots[c+k] - knots[c+1], NOT knots[c+k+1] - knots[c+1].
                    // (with p = k-1, U_{i+p+1} = U_{i+k}.)
                    size_t k_deriv = k - 1;
                    size_t nc_deriv = nc - 1;
                    size_t n_knots_deriv = n_knots_ - 1;
                    for (size_t c = 0; c + 1 < nc; ++c) {
                        double t_right = grid_.data[c + k];   // U_{i+p+1} = U_{c+k}
                        double t_left = grid_.data[c + 1];    // U_{i+1} = U_{c+1}
                        double denom = t_right - t_left;
                        if (denom <= 0.0) continue;
                        double d_c = (k_deriv) * (edge_coefs[c + 1] - edge_coefs[c]) / denom;
                        double B_c_d = bspline_basis(x, grid_.data.data() + 1, n_knots_deriv,
                                                      nc_deriv, k_deriv, c);
                        spline_deriv += d_c * B_c_d;
                    }
                }
                double d_input_local = base_weight_(i, j) * silu_deriv(x) + spline_weight_(i, j) * spline_deriv;
                grad_input(b, j) += d_phi * d_input_local;
            }
        }
    }
    return grad_input;
}

void KANLayer::update_weights(double learning_rate) {
    // Plain SGD update; users can swap to other optimizers externally.
    // (The library's standard update scheme.)
    auto sgd_update = [&](Tensor& w, const Tensor& g) {
        for (size_t k = 0; k < w.data.size(); ++k) {
            w.data[k] -= learning_rate * g.data[k];
        }
    };
    sgd_update(spline_coefs_, grad_spline_coefs_);
    sgd_update(base_weight_, grad_base_weight_);
    sgd_update(spline_weight_, grad_spline_weight_);
}

std::vector<Tensor*> KANLayer::parameters() {
    // Order matters for tests: spline_coefs, base_weight, spline_weight, grid
    return { &spline_coefs_, &base_weight_, &spline_weight_, &grid_ };
}

std::vector<Tensor*> KANLayer::gradients() {
    return { &grad_spline_coefs_, &grad_base_weight_, &grad_spline_weight_ };
}

void KANLayer::zero_grad() {
    grad_spline_coefs_.fill(0.0);
    grad_base_weight_.fill(0.0);
    grad_spline_weight_.fill(0.0);
}

// =====================================================================
// KANModel implementation
// =====================================================================

KANModel::KANModel(size_t in_dim, const std::vector<size_t>& hidden_dims,
                   size_t out_dim, size_t num_grids, size_t spline_order) {
    std::vector<size_t> sizes;
    sizes.push_back(in_dim);
    for (size_t h : hidden_dims) sizes.push_back(h);
    sizes.push_back(out_dim);
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        layers_.push_back(std::make_unique<KANLayer>(sizes[i], sizes[i + 1],
                                                      num_grids, spline_order));
    }
    last_layer_outputs_.resize(layers_.size());
}

Tensor KANModel::forward(const Tensor& input) {
    Tensor x = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i]->forward(x);
        last_layer_outputs_[i] = x;
    }
    return x;
}

Tensor KANModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor g = grad_output;
    for (size_t i = layers_.size(); i > 0; --i) {
        size_t idx = i - 1;
        Tensor grad_in = layers_[idx]->backward(g, /*learning_rate=*/0.0);
        g = grad_in;
    }
    return g;
}

void KANModel::update_weights(double learning_rate) {
    for (auto& l : layers_) l->update_weights(learning_rate);
}

std::vector<Tensor*> KANModel::parameters() {
    std::vector<Tensor*> all;
    for (auto& l : layers_) {
        for (auto* p : l->parameters()) all.push_back(p);
    }
    return all;
}

std::vector<Tensor*> KANModel::gradients() {
    std::vector<Tensor*> all;
    for (auto& l : layers_) {
        for (auto* g : l->gradients()) all.push_back(g);
    }
    return all;
}

void KANModel::zero_grad() {
    for (auto& l : layers_) l->zero_grad();
}