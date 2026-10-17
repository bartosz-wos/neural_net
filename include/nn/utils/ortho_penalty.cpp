#include "ortho_penalty.h"

#include "../core/layer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// One contiguous (s, c) block starting at column `base`. The offset MUST
// advance by BOTH s and c -- the first version indexed `w(b, j*c + k)` only,
// which never used `i`, so every row of a slice read the same `c` columns.
Tensor view_slice(const Tensor& w, size_t b, size_t base, size_t s, size_t c) {
    Tensor m(s, c);
    for (size_t i = 0; i < s; ++i)
        for (size_t k = 0; k < c; ++k) m(i, k) = w(b, base + i * c + k);
    return m;
}

void require_nonempty(const Tensor& w, const char* who) {
    if (w.rows == 0 || w.cols == 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": weight tensor must have non-zero rows AND cols "
                                    "(got " + std::to_string(w.rows) + "x" +
                                    std::to_string(w.cols) + ")");
    }
}

// Column sums (length c) and row sums (length r). The adjoint needs both, and
// they are the ONLY quantities the penalty depends on -- see the header's (1).
void row_col_sums(const Tensor& w, std::vector<double>& col, std::vector<double>& row) {
    col.assign(w.cols, 0.0);
    row.assign(w.rows, 0.0);
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t k = 0; k < w.cols; ++k) {
            const double v = w(i, k);
            col[k] += v;
            row[i] += v;
        }
}

// m1 = mean(W W^T - I_r) = (||u||^2 - r)/r^2,  m2 = mean(W^T W - I_c)
//                                   = (||t||^2 - c)/c^2.
void grams(const std::vector<double>& col, const std::vector<double>& row,
           size_t r, size_t c, double& m1, double& m2) {
    double uu = 0.0, tt = 0.0;
    for (double v : col) uu += v * v;
    for (double v : row) tt += v * v;
    // NOTE the `- r` and `- c`, not `- r*r` / `- c*c`: the mean runs over all
    // r*r ENTRIES, and the identity matrix contributes exactly r of them.
    m1 = (uu - static_cast<double>(r)) / (static_cast<double>(r) * static_cast<double>(r));
    m2 = (tt - static_cast<double>(c)) / (static_cast<double>(c) * static_cast<double>(c));
}

double penalty_from_sums(const std::vector<double>& col, const std::vector<double>& row,
                         size_t r, size_t c) {
    double m1, m2;
    grams(col, row, r, c, m1, m2);
    const double s = 1.0 / (static_cast<double>(r) * static_cast<double>(c));
    return s * (m1 * m1 + m2 * m2);
}

// The adjoint. Written as the sum of a COLUMN term (every row equal) and a ROW
// term (every column equal), which is what makes it separable in (i, k) -- a
// per-element Gram adjoint cannot produce that shape, so the structure is
// itself a discriminator against a plausible wrong implementation.
Tensor gradient_from_sums(const std::vector<double>& col, const std::vector<double>& row,
                          size_t r, size_t c) {
    double m1, m2;
    grams(col, row, r, c, m1, m2);
    const double s = 4.0 / (static_cast<double>(r) * static_cast<double>(c));
    const double inv_r2 = 1.0 / (static_cast<double>(r) * static_cast<double>(r));
    const double inv_c2 = 1.0 / (static_cast<double>(c) * static_cast<double>(c));

    Tensor g = Tensor::zeros(r, c);
    for (size_t i = 0; i < r; ++i) {
        const double beta = s * m2 * row[i] * inv_c2;
        for (size_t k = 0; k < c; ++k) {
            const double alpha = s * m1 * col[k] * inv_r2;
            g(i, k) = alpha + beta;
        }
    }
    return g;
}

Tensor view_mode(const Tensor& w, size_t mode, size_t r, size_t c) {
    Tensor m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t k = 0; k < c; ++k) m(i, k) = w(mode, i * c + k);
    return m;
}

void require_mode_shape(const Tensor& w, size_t r, size_t c, const char* who) {
    require_nonempty(w, who);
    if (r == 0 || c == 0) {
        throw std::invalid_argument(std::string(who) + ": r and c must be non-zero");
    }
    if (w.cols != r * c) {
        throw std::invalid_argument(std::string(who) +
                                    ": w.cols must equal r*c (got " +
                                    std::to_string(w.cols) + ", need " +
                                    std::to_string(r * c) + ")");
    }
}

// The per-slice layout needs its own guard: w.cols must equal
// num_slices * s * c, which is a STRICTER condition than cols == s*c.
void require_slice_shape(const Tensor& w, size_t num_slices, size_t s, size_t c,
                         const char* who) {
    require_nonempty(w, who);
    if (num_slices == 0 || s == 0 || c == 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": num_slices, s and c must all be non-zero");
    }
    const size_t need = num_slices * s * c;
    if (w.cols != need) {
        throw std::invalid_argument(std::string(who) +
                                    ": w.cols must equal num_slices*s*c = " +
                                    std::to_string(need) + " (got " +
                                    std::to_string(w.cols) + "). Without this check "
                                    "the slice loop would read past the row.");
    }
}

// One-sided power iteration with a FIXED step count: deterministic, no RNG, and
// so a repeated call on the same input is bit-identical (D10).
constexpr int kPowerSteps = 100;

}  // namespace

double ortho_penalty(const Tensor& w) {
    require_nonempty(w, "ortho_penalty");
    std::vector<double> col, row;
    row_col_sums(w, col, row);
    return penalty_from_sums(col, row, w.rows, w.cols);
}

Tensor ortho_penalty_gradient(const Tensor& w) {
    require_nonempty(w, "ortho_penalty_gradient");
    std::vector<double> col, row;
    row_col_sums(w, col, row);
    return gradient_from_sums(col, row, w.rows, w.cols);
}

double ortho_penalty_lam(const Tensor& w, double lambda) {
    return lambda * ortho_penalty(w);
}

Tensor ortho_penalty_lam_gradient(const Tensor& w, double lambda) {
    Tensor g = ortho_penalty_gradient(w);
    for (size_t i = 0; i < g.rows; ++i)
        for (size_t k = 0; k < g.cols; ++k) g(i, k) *= lambda;
    return g;
}

double ortho_penalty_modes(const Tensor& w, size_t r, size_t c) {
    require_mode_shape(w, r, c, "ortho_penalty_modes");
    double acc = 0.0;
    for (size_t b = 0; b < w.rows; ++b) acc += ortho_penalty(view_mode(w, b, r, c));
    return acc / static_cast<double>(w.rows);
}

Tensor ortho_penalty_modes_gradient(const Tensor& w, size_t r, size_t c) {
    require_mode_shape(w, r, c, "ortho_penalty_modes_gradient");
    Tensor g = Tensor::zeros(w.rows, w.cols);
    const double inv = 1.0 / static_cast<double>(w.rows);
    for (size_t b = 0; b < w.rows; ++b) {
        Tensor gm = ortho_penalty_gradient(view_mode(w, b, r, c));
        for (size_t i = 0; i < r; ++i)
            for (size_t k = 0; k < c; ++k) g(b, i * c + k) = gm(i, k) * inv;
    }
    return g;
}

double ortho_penalty_slices(const Tensor& w, size_t num_slices, size_t s, size_t c) {
    require_slice_shape(w, num_slices, s, c, "ortho_penalty_slices");
    double acc = 0.0;
    for (size_t b = 0; b < w.rows; ++b)
        for (size_t j = 0; j < num_slices; ++j)
            acc += ortho_penalty(view_slice(w, b, j * s * c, s, c));
    return acc / static_cast<double>(num_slices * w.rows);
}

Tensor ortho_penalty_slices_gradient(const Tensor& w, size_t num_slices, size_t s, size_t c) {
    require_slice_shape(w, num_slices, s, c, "ortho_penalty_slices_gradient");
    Tensor g = Tensor::zeros(w.rows, w.cols);
    const double inv = 1.0 / static_cast<double>(num_slices * w.rows);
    for (size_t b = 0; b < w.rows; ++b)
        for (size_t j = 0; j < num_slices; ++j) {
            const size_t base = j * s * c;
            Tensor gs = ortho_penalty_gradient(view_slice(w, b, base, s, c));
            for (size_t i = 0; i < s; ++i)
                for (size_t k = 0; k < c; ++k) g(b, base + i * c + k) = gs(i, k) * inv;
        }
    return g;
}

double ortho_spectral_norm(const Tensor& w) {
    require_nonempty(w, "ortho_spectral_norm");
    // Start from all-ones: deterministic, unlike a random start, so repeated
    // calls are bit-identical (D10). For the zero matrix every step is 0/0, so
    // guard explicitly rather than dividing by a vanishing norm.
    std::vector<double> v(w.cols, 0.0);
    double nv = 0.0;
    for (size_t k = 0; k < w.cols; ++k) { v[k] = 1.0; nv += 1.0; }
    nv = std::sqrt(nv);
    if (nv == 0.0) return 0.0;
    for (size_t k = 0; k < w.cols; ++k) v[k] /= nv;

    for (int step = 0; step < kPowerSteps; ++step) {
        std::vector<double> u(w.rows, 0.0);
        double nu = 0.0;
        for (size_t i = 0; i < w.rows; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < w.cols; ++k) s += w(i, k) * v[k];
            u[i] = s;
            nu += s * s;
        }
        nu = std::sqrt(nu);
        if (nu == 0.0) return 0.0;
        for (size_t i = 0; i < w.rows; ++i) u[i] /= nu;

        double nv2 = 0.0;
        for (size_t k = 0; k < w.cols; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < w.rows; ++i) s += w(i, k) * u[i];
            v[k] = s;
            nv2 += s * s;
        }
        nv2 = std::sqrt(nv2);
        if (nv2 == 0.0) return 0.0;
        for (size_t k = 0; k < w.cols; ++k) v[k] /= nv2;
    }

    // Rayleigh quotient u^T W v with the converged (normalised) u.
    std::vector<double> u(w.rows, 0.0);
    double nu = 0.0;
    for (size_t i = 0; i < w.rows; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < w.cols; ++k) s += w(i, k) * v[k];
        u[i] = s;
        nu += s * s;
    }
    nu = std::sqrt(nu);
    if (nu == 0.0) return 0.0;
    double rv = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t k = 0; k < w.cols; ++k) rv += (u[i] / nu) * w(i, k) * v[k];
    return std::fabs(rv);
}

double ortho_spectral_norm_inverse(const Tensor& w) {
    const double sv = ortho_spectral_norm(w);
    if (sv == 0.0) {
        throw std::invalid_argument(
            "ortho_spectral_norm_inverse: the matrix is singular (sv1 == 0), so "
            "1/sv1 is not finite. Renormalising the weight would produce inf/NaN "
            "in every downstream update; refusing is the only safe answer.");
    }
    return 1.0 / sv;
}

double ortho_penalty_one_sided(const Tensor& w) {
    return std::fabs(ortho_spectral_norm(w) - 1.0);
}

double ortho_penalty_spectral(const Tensor& w) {
    const double sv = ortho_spectral_norm(w);
    // A zero matrix is not a scaled rotation; excluding it by definition keeps
    // the penalty from pulling a zeroed-out layer back to a unit scale.
    if (sv == 0.0) return 0.0;
    const double d = sv - 1.0;
    return d * d;
}

Tensor ortho_penalty_spectral_gradient(const Tensor& w) {
    require_nonempty(w, "ortho_penalty_spectral_gradient");
    // d/dW (sv1 - 1)^2. With sigma the largest singular value and
    // (u, v) its normalised singular vectors: d sigma / dW = u v^T, so
    //     dL/dW = 2(sigma - 1) * u v^T.
    // (The sub-dominant terms have zero derivative because sigma is the largest
    //  eigenvalue of W^T W with a non-degenerate gap on generic input; this is
    //  the same first-order approximation the paper's own spectral term makes,
    //  and the test verifies it against centred finite differences.)
    std::vector<double> v(w.cols, 1.0);
    double nv = std::sqrt(static_cast<double>(w.cols));
    for (size_t k = 0; k < w.cols; ++k) v[k] /= nv;
    std::vector<double> u;
    for (int step = 0; step < kPowerSteps; ++step) {
        std::vector<double> ut(w.rows, 0.0);
        double nu = 0.0;
        for (size_t i = 0; i < w.rows; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < w.cols; ++k) s += w(i, k) * v[k];
            ut[i] = s;
            nu += s * s;
        }
        nu = std::sqrt(nu);
        if (nu == 0.0) return Tensor::zeros(w.rows, w.cols);
        for (size_t i = 0; i < w.rows; ++i) ut[i] /= nu;
        u = ut;

        double nv2 = 0.0;
        for (size_t k = 0; k < w.cols; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < w.rows; ++i) s += w(i, k) * u[i];
            v[k] = s;
            nv2 += s * s;
        }
        nv2 = std::sqrt(nv2);
        if (nv2 == 0.0) return Tensor::zeros(w.rows, w.cols);
        for (size_t k = 0; k < w.cols; ++k) v[k] /= nv2;
    }

    double sigma = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t k = 0; k < w.cols; ++k) sigma += u[i] * w(i, k) * v[k];
    if (sigma == 0.0) return Tensor::zeros(w.rows, w.cols);

    // NOTE: NO division by sigma here. d sigma / dW = u v^T exactly (the
    // Rayleigh quotient's envelope theorem kills the d u, d v terms), so
    // d (sigma-1)^2 / dW = 2(sigma-1) u v^T. Dividing by sigma here was the
    // original bug, and it had the cleanest fingerprint in the file: the
    // analytic gradient came out exactly 1/sigma times FD on EVERY fixture
    // (measured rel_err 0.6417 = 1 - 1/2.791 on a 3x3 fixture, and 1/3 on
    // diag(3,1)). A clean constant ratio across every fixture is the
    // scale-factor signature, not a structural error.
    const double coef = 2.0 * (sigma - 1.0);
    Tensor g = Tensor::zeros(w.rows, w.cols);
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t k = 0; k < w.cols; ++k) g(i, k) = coef * u[i] * v[k];
    return g;
}

// ---------------------------------------------------------------------------
// Model-level
// ---------------------------------------------------------------------------

double OrthoPenaltyLoss::evaluate(Model& model) const {
    if (model.layers.empty()) {
        throw std::invalid_argument(
            "OrthoPenaltyLoss::evaluate: the model has no layers, so there are no "
            "weights to regularise. (This is the silent-no-op failure mode: the "
            "penalty would otherwise score 0 forever and look like it worked.)");
    }
    double acc = 0.0;
    size_t counted = 0;
    for (auto& layer : model.layers) {
        auto params = layer->parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            const Tensor& p = *params[i];
            // A 1-D bias has no Gram matrix to regularise: it is a SHIFT, and
            // shifting a layer does not affect its conditioning. Penalising it
            // would add a term with no meaning and a nonzero value.
            if (p.rows < 2 || p.cols < 2) continue;
            acc += ortho_penalty(p);
            ++counted;
        }
    }
    // Scale ONCE, by lambda (D1/D9). When nothing was counted (a model of pure
    // biases) the result is exactly 0.
    return counted == 0 ? 0.0 : lambda_ * acc;
}

Tensor OrthoPenaltyLoss::gradient_wrt(const Tensor& param) const {
    if (param.rows < 2 || param.cols < 2) return Tensor::zeros(param.rows, param.cols);
    return ortho_penalty_gradient(param);
}

void OrthoPenaltyLoss::apply_to_gradients(Model& model, double lambda) const {
    if (model.layers.empty()) {
        throw std::invalid_argument(
            "OrthoPenaltyLoss::apply_to_gradients: the model has no layers, so there "
            "are no weights to regularise.");
    }
    const double lam = lambda < 0.0 ? lambda_ : lambda;
    for (auto& layer : model.layers) {
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.size() != grads.size() || params.empty()) continue;
        for (size_t i = 0; i < params.size(); ++i) {
            const Tensor& p = *params[i];
            if (p.rows < 2 || p.cols < 2) continue;
            Tensor gp = ortho_penalty_gradient(p);
            for (size_t r = 0; r < p.rows; ++r)
                for (size_t k = 0; k < p.cols; ++k) (*grads[i])(r, k) += lam * gp(r, k);
        }
    }
}