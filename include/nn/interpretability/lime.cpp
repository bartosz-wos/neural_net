#include "lime.h"

#include "../core/tensor.h"
#include "shapley.h"  // model_output

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

// Local xorshift64* generator + Box-Muller normal.
//
// Deliberately NOT std::rand()/std::normal_distribution: this repo has a
// documented history of cross-suite RNG flakiness (the test_adaln_zero
// intermittent-NaN entry in NOT_FIXED.md), so an attribution pass must not
// perturb the RNG stream of an unrelated suite. Same rationale and same
// high-bits discipline as shapley.cpp's Lcg.
struct Rng {
    uint64_t state;
    double spare = 0.0;
    bool has_spare = false;
    explicit Rng(unsigned seed)
        : state(static_cast<uint64_t>(seed) * 2654435761u + 1u) {}

    uint64_t next64() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    }
    // Uniform in [0, 1) from the WELL-MIXED high bits (shapley.cpp documents
    // that reading the low, correlated end of an LCG state gives a measurably
    // non-uniform stream).
    double uniform() {
        return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0);
    }
    // Standard normal via Box-Muller, caching the second variate.
    //
    // BUG FOUND 2026-10-08 — the cached spare was stored as
    //   static_cast<unsigned long>(r * sin(theta) * 1e6)
    // and then returned via static_cast<double>(spare), i.e. the 1e6 scale
    // factor was applied on the way IN and never divided back out. Every
    // second draw was therefore a ~1e20 integer rather than a variate: the
    // neighborhood became astronomically large, every distance overflowed
    // exp() to 0, all kernel weights except row 0 underflowed to 0, and the
    // centered normal equations became exactly 0 = 0, so the solve returned
    // beta = 0 for every feature. The symptom (a perfectly linear model
    // explained with all-zero coefficients and score 0) looks like a ridge or
    // ranking bug; it was a single unit-conversion slip in the sampler.
    //
    // The lesson: cache the spare as the SAME type you return. Mixing a scaled
    // integer cache with a double return type is a silent 1e6 error.
    double normal() {
        if (has_spare) { has_spare = false; return spare; }
        double u1 = uniform();
        const double u2 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;  // log(0) guard
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * 3.14159265358979323846 * u2;
        spare = r * std::sin(theta);
        has_spare = true;
        return r * std::cos(theta);
    }
};

// Solve A x = b by Gaussian elimination with partial pivoting. `A` is row-major
// n x n and is modified in place. Returns false when a pivot is numerically
// zero (singular), leaving the caller to apply the ridge.
//
// Mirrors the helper in shapley.cpp rather than sharing it: that one is in an
// anonymous namespace, and a shared internal solver header for two callers is
// a larger refactor than this feature warrants.
bool solve_dense(std::vector<double>& A, std::vector<double>& b, size_t n) {
    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        double best = std::abs(A[col * n + col]);
        for (size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-300) return false;
        if (pivot != col) {
            // Swap the WHOLE row, not just the pivot element. This was the
            // first version's bug: swapping only A[col*n+col] left the rest of
            // the pivot row in place, which zeroed every solve.
            for (size_t c = 0; c < n; ++c) std::swap(A[col * n + c], A[pivot * n + c]);
            std::swap(b[col], b[pivot]);
        }
        const double d = A[col * n + col];
        for (size_t r = col + 1; r < n; ++r) {
            const double factor = A[r * n + col] / d;
            if (factor == 0.0) continue;
            for (size_t c = col; c < n; ++c) A[r * n + c] -= factor * A[col * n + c];
            b[r] -= factor * b[col];
        }
    }
    for (size_t ri = n; ri-- > 0;) {
        double s = b[ri];
        for (size_t c = ri + 1; c < n; ++c) s -= A[ri * n + c] * b[c];
        b[ri] = s / A[ri * n + ri];
    }
    return true;
}

} // namespace

double exp_kernel(double d, double kernel_width) {
    // sqrt(exp(-(d/omega)^2)) — the sqrt() applies to the RESULT of exp().
    // See the header: exp(-d^2/w^2) instead is a factor-of-two-in-the-exponent
    // bug that still looks like a perfectly reasonable kernel.
    const double t = d / kernel_width;
    return std::sqrt(std::exp(-t * t));
}

double exp_kernel_width(size_t n_features) {
    if (n_features == 0) {
        throw std::invalid_argument("exp_kernel_width: n_features must be >= 1");
    }
    return std::sqrt(static_cast<double>(n_features)) * 0.75;
}

LocalSurrogate fit_local_surrogate(const std::vector<std::vector<double>>& z,
                                   const std::vector<double>& y,
                                   const std::vector<double>& weights,
                                   double alpha) {
    if (alpha < 0.0) {
        throw std::invalid_argument("fit_local_surrogate: alpha must be >= 0");
    }
    if (z.empty()) {
        throw std::invalid_argument("fit_local_surrogate: design matrix must have >= 1 column");
    }
    const size_t n = y.size();
    if (n == 0) {
        throw std::invalid_argument("fit_local_surrogate: n_samples must be >= 1");
    }
    if (weights.size() != n) {
        throw std::invalid_argument("fit_local_surrogate: weights length must match responses");
    }
    const size_t p = z.size();
    for (size_t j = 0; j < p; ++j) {
        if (z[j].size() != n) {
            throw std::invalid_argument("fit_local_surrogate: every design column must have n_samples entries");
        }
    }

    // --- Weighted centering (sklearn's _preprocess_data with fit_intercept) ---
    // Centering by the SAMPLE-WEIGHTED mean, not the unweighted one. LIME's
    // kernel weights are strongly non-uniform by construction, so this is not
    // a rounding detail.
    double sw = 0.0;
    for (size_t i = 0; i < n; ++i) sw += weights[i];
    if (std::abs(sw) < 1e-300) {
        throw std::invalid_argument("fit_local_surrogate: sample weights sum to zero");
    }
    std::vector<double> xm(p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += weights[i] * z[j][i];
        xm[j] = s / sw;
    }
    double ym = 0.0;
    for (size_t i = 0; i < n; ++i) ym += weights[i] * y[i];
    ym /= sw;

    // --- Normal equations: (X'WX + alpha*I) b = X'W(yc) -----------------------
    // alpha is the reference's Ridge penalty and is NOT divided by n_samples:
    // sklearn's objective is ||y - Xw||^2 + alpha*||w||^2 (verified in
    // sklearn/linear_model/_ridge.py). Dropping the +alpha*I term makes alpha
    // inert and silently reduces this to ordinary least squares.
    std::vector<double> A(p * p, 0.0);
    std::vector<double> b(p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        for (size_t k = j; k < p; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) {
                s += weights[i] * (z[j][i] - xm[j]) * (z[k][i] - xm[k]);
            }
            A[j * p + k] = s;
            A[k * p + j] = s;
        }
        A[j * p + j] += alpha;
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += weights[i] * (z[j][i] - xm[j]) * (y[i] - ym);
        b[j] = s;
    }

    if (!solve_dense(A, b, p)) {
        // Singular: fall back to a trace-scaled ridge so the solve still
        // returns something finite. At this scale it only engages on a
        // genuinely rank-deficient design (e.g. fewer distinct points than
        // features), and perturbs the answer at the 1e-10 relative level.
        std::vector<double> A2(p * p, 0.0);
        std::vector<double> b2 = b;
        double trace = 0.0;
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < p; ++k) A2[j * p + k] = A[j * p + k];
            trace += A[j * p + j];
        }
        const double lam = (trace > 0.0) ? 1e-10 * trace : 1e-10;
        for (size_t j = 0; j < p; ++j) A2[j * p + j] += lam;
        solve_dense(A2, b2, p);
        b = b2;
    }

    // --- Reconstruct the intercept (sklearn's _set_intercept) ---------------
    double intercept = ym;
    for (size_t j = 0; j < p; ++j) intercept -= xm[j] * b[j];

    // --- Weighted R^2, sklearn's Ridge.score ---------------------------------
    // Both sums carry the weights, and SS_tot is about the WEIGHTED mean of y.
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double pred = intercept;
        for (size_t j = 0; j < p; ++j) pred += b[j] * z[j][i];
        ss_res += weights[i] * (y[i] - pred) * (y[i] - pred);
        ss_tot += weights[i] * (y[i] - ym) * (y[i] - ym);
    }

    LocalSurrogate s;
    s.coefficients = b;
    s.intercept = intercept;
    s.score = (std::abs(ss_tot) < 1e-300) ? 0.0 : 1.0 - ss_res / ss_tot;
    return s;
}

LimeResult lime_explain(Model& model,
                        const Tensor& x,
                        const LimeConfig& cfg,
                        size_t target) {
    if (cfg.n_samples == 0) {
        throw std::invalid_argument("lime_explain: n_samples must be >= 1");
    }
    if (x.rows != 1) {
        throw std::invalid_argument("lime_explain: x must be a single row (the instance to explain)");
    }
    const size_t m = x.cols;
    if (m == 0) {
        throw std::invalid_argument("lime_explain: input must have at least one feature");
    }

    // Resolve the standardization. Empty tensors default to mean 0 / scale 1,
    // which reproduces "no standardization" rather than failing — that is the
    // reference's behavior when the training data has no spread to record.
    std::vector<double> mu(m, 0.0), sd(m, 1.0);
    if (cfg.mean.rows > 0) {
        if (cfg.mean.cols != m) {
            throw std::invalid_argument("lime_explain: mean width must equal n_features");
        }
        for (size_t j = 0; j < m; ++j) mu[j] = cfg.mean[0][j];
    }
    if (cfg.std.rows > 0) {
        if (cfg.std.cols != m) {
            throw std::invalid_argument("lime_explain: std width must equal n_features");
        }
        for (size_t j = 0; j < m; ++j) sd[j] = cfg.std[0][j];
    }
    for (size_t j = 0; j < m; ++j) {
        if (!(sd[j] > 0.0)) {
            throw std::invalid_argument("lime_explain: every standard deviation must be positive");
        }
    }

    // Target validation happens BEFORE any model call so a bad target cannot
    // leave the model's gradients half-touched.
    {
        for (auto& lp : model.layers) lp->zero_grad();
        const Tensor probe = model.forward(x);
        if (target >= probe.cols) {
            throw std::out_of_range(
                "lime_explain: target index out of range for model output");
        }
    }

    const size_t n = cfg.n_samples;
    const double omega = (cfg.kernel_width > 0.0) ? cfg.kernel_width
                                                  : exp_kernel_width(m);

    // --- Build the neighborhood ----------------------------------------------
    // raw[i][j] = z[i][j]*sd_j + mu_j  (the reference perturbs around the
    // TRAINING MEAN, lime_tabular.py __data_inverse with
    // sample_around_instance=False), then row 0 is REPLACED by the instance
    // itself so it always appears in the fit with kernel weight 1.0.
    std::vector<std::vector<double>> z(m, std::vector<double>(n, 0.0));
    std::vector<double> raw(n * m, 0.0);
    Rng rng(cfg.seed);
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            z[j][i] = rng.normal();
            raw[i * m + j] = z[j][i] * sd[j] + mu[j];
        }
    }
    for (size_t j = 0; j < m; ++j) {
        z[j][0] = (x[0][j] - mu[j]) / sd[j];  // standardized instance
        raw[0 * m + j] = x[0][j];
    }

    // --- Distances in STANDARDIZED space (lime_tabular explain_instance) ------
    // `z` is already standardized by construction, so the distance metric is
    // the plain euclidean norm of the z-row difference. Computing it on the
    // raw rows instead would weight a feature by its sd twice.
    std::vector<double> dist(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < m; ++j) {
            const double dz = z[j][i] - z[j][0];
            s += dz * dz;
        }
        dist[i] = std::sqrt(s);
    }

    // --- Evaluate the black box on the whole neighborhood in ONE pass --------
    Tensor batch = Tensor::zeros(n, m);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            batch[i][j] = raw[i * m + j];

    std::vector<double> responses(n, 0.0);
    std::vector<double> weights(n, 0.0);
    for (size_t i = 0; i < n; ++i) weights[i] = exp_kernel(dist[i], omega);

    {
        Tensor out = model.forward(batch);
        for (size_t i = 0; i < n; ++i) responses[i] = out[i][target];
    }

    // --- Fit, then rank by |b| (lime_base.py's 'highest_weights') ------------
    LocalSurrogate fit = fit_local_surrogate(z, responses, weights, cfg.alpha);

    LimeResult r;
    const size_t p = fit.coefficients.size();
    r.coefficients = fit.coefficients;
    r.intercept = fit.intercept;
    r.score = fit.score;
    r.n_features = m;
    r.n_samples = n;
    r.target = target;
    r.config = cfg;
    r.config.mean = Tensor::zeros(1, m);
    r.config.std = Tensor::zeros(1, m);
    for (size_t j = 0; j < m; ++j) {
        r.config.mean[0][j] = mu[j];
        r.config.std[0][j] = sd[j];
    }
    r.config.kernel_width = omega;

    // g(z_0): the surrogate's own answer at the instance. This is what LIME
    // reports as the local prediction; for an exactly linear black box at
    // alpha = 0 it reproduces f(x) to machine precision.
    r.local_prediction = fit.intercept;
    for (size_t j = 0; j < p; ++j) r.local_prediction += fit.coefficients[j] * z[j][0];
    r.base_prediction = model_output(model, x, target);

    // --- Feature selection: keep the largest |b| --------------------------
    // The reference fits on ALL features first (highest_weights) and then
    // re-fits on the selected subset. num_features == 0 means "keep all".
    std::vector<size_t> order(p);
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::abs(fit.coefficients[a]) > std::abs(fit.coefficients[b]);
    });

    size_t keep = p;
    if (cfg.num_features > 0 && cfg.num_features < p) keep = cfg.num_features;

    if (keep < p) {
        // Re-fit on the selected subset ONLY, so the returned coefficients
        // are the ones of the model the user actually sees. Re-ranking without
        // re-fitting would report weights from a different fit.
        std::vector<std::vector<double>> zs;
        for (size_t i = 0; i < keep; ++i) zs.push_back(z[order[i]]);
        LocalSurrogate sub = fit_local_surrogate(zs, responses, weights, cfg.alpha);
        r.coefficients = sub.coefficients;
        r.intercept = sub.intercept;
        r.score = sub.score;
        r.local_prediction = sub.intercept;
        for (size_t i = 0; i < keep; ++i)
            r.local_prediction += sub.coefficients[i] * zs[i][0];
    }

    r.ranking.assign(order.begin(), order.begin() + static_cast<long>(keep));
    r.importances.resize(keep);
    for (size_t i = 0; i < keep; ++i) r.importances[i] = std::abs(r.coefficients[i]);

    // Zero gradients again on the way out. MUTATION NOTE: removing this line is
    // NOT catchable by any test, and that is provable rather than a coverage
    // gap: nothing between here and the entry zero_grad writes a parameter
    // gradient (lime_explain never calls backward()), so the two zeroings are
    // behaviourally IDENTICAL. The entry zeroing exists to discard a caller's
    // stale accumulated gradients; this one is defence in depth for future
    // changes that might add a backward pass. The suite pins the OBSERVABLE
    // contract (gradients are zero on return) rather than the line that
    // delivers it.
    for (auto& lp : model.layers) lp->zero_grad();
    return r;
}