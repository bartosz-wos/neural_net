#include "partial_dependence.h"

#include "../core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

// Local xorshift64* generator.
//
// Deliberately NOT std::rand()/std::normal_distribution: this repo has a
// documented history of cross-suite RNG flakiness (the test_adaln_zero
// intermittent-NaN entry in NOT_FIXED.md), so an attribution pass must not
// perturb the RNG stream of an unrelated suite. Same rationale and same
// high-bits discipline as shapley.cpp's Lcg and lime.cpp's Rng.
struct Rng {
    uint64_t state;
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
    // Uniform integer in [0, n) WITHOUT modulo bias via rejection sampling.
    // Plain `next64() % n` is biased toward the low end of the range, which for
    // the small `n` here (instance counts) is a visible skew in which instances
    // get curves.
    size_t below(size_t n) {
        const uint64_t limit = UINT64_MAX - (UINT64_MAX % n);
        uint64_t r;
        do { r = next64(); } while (r >= limit);
        return static_cast<size_t>(r % n);
    }
};

// numpy's 'linear' quantile (the np.quantile default): for sorted `v` and
// probability q, h = q*(n-1), i = floor(h), frac = h - i, and
//
//   value = v[i]*(1-frac) + v[i+1]*frac
//
// The interpolation is the load-bearing detail — sklearn's _nanquantile calls
// np.nanquantile with no `method=` argument, so 'linear' is what the reference
// computes. 'lower' / 'higher' / 'nearest' are equally defensible-looking
// alternatives that produce a different axis.
double quantile_linear_sorted(const std::vector<double>& sorted, double q) {
    const size_t n = sorted.size();
    if (n == 1) return sorted[0];
    const double h = q * static_cast<double>(n - 1);
    const size_t i = static_cast<size_t>(std::floor(h));
    if (i + 1 >= n) return sorted.back();
    const double frac = h - static_cast<double>(i);
    return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
}

void check_pct_range(double lo, double hi) {
    if (lo < 0.0 || lo > 1.0 || hi < 0.0 || hi > 1.0) {
        throw std::invalid_argument("pd_grid: percentiles must be in [0, 1]");
    }
    if (lo >= hi) {
        throw std::invalid_argument("pd_grid: percentile_lo must be < percentile_hi");
    }
}

} // namespace

std::vector<double> pd_grid(const Tensor& X,
                            size_t feature,
                            double percentile_lo,
                            double percentile_hi,
                            size_t grid_resolution) {
    if (X.rows == 0) {
        throw std::invalid_argument("pd_grid: X must have at least one row");
    }
    if (feature >= X.cols) {
        throw std::invalid_argument("pd_grid: feature index out of range");
    }
    if (grid_resolution <= 1) {
        throw std::invalid_argument("pd_grid: grid_resolution must be > 1");
    }
    check_pct_range(percentile_lo, percentile_hi);

    // Sort a COPY of the column: the caller's X must come back untouched, and an
    // unsorted column has to give the same axis as a sorted one.
    std::vector<double> column(X.rows);
    for (size_t i = 0; i < X.rows; ++i) column[i] = X[i][feature];
    std::sort(column.begin(), column.end());

    // Low-cardinality fallback (partial_dependence.py:138-142). The reference's
    // condition is `uniques.shape[0] < grid_resolution` — STRICTLY less than — so
    // a column with exactly grid_resolution distinct values still takes the
    // linspace branch. Using `<=` here would silently change the axis of every
    // small-domain integer feature.
    //
    // BUG FOUND 2026-10-09 — the unique count below must NOT be obtained from
    // std::unique on `column` itself. std::unique does not just count: it
    // COMPACTS the range in place, rewriting [0,0,5,5] as [0,5,5,5]. On the
    // fallback branch that compaction happens to be harmless (the erase that
    // follows throws away the tail), but on the LINALSPHERE branch — every time
    // n_uniques >= grid_resolution, i.e. the common case — the quantiles are
    // then computed from a buffer that is no longer sorted. Symptom: a column of
    // {0,5,0,5} at q=0.05 returned lo=0.75 (interpolating toward the wrong
    // order statistic) instead of 0.0, i.e. the axis was shifted by a factor
    // that grows with the column's spread. The adjacent-value loop below is
    // non-mutating; a std::unique call here would corrupt the sort order.
    size_t n_uniques = 0;
    for (size_t i = 0; i < column.size(); ++i) {
        if (i == 0 || column[i] != column[i - 1]) ++n_uniques;
    }
    if (n_uniques < grid_resolution) {
        std::vector<double> uniques;
        uniques.reserve(n_uniques);
        for (size_t i = 0; i < column.size(); ++i) {
            if (i == 0 || column[i] != column[i - 1]) uniques.push_back(column[i]);
        }
        return uniques;
    }

    const double lo = quantile_linear_sorted(column, percentile_lo);
    const double hi = quantile_linear_sorted(column, percentile_hi);

    // Degenerate axis (partial_dependence.py:144-148). The reference RAISES here
    // rather than returning a zero-width axis; a silently flat curve would read
    // as "this feature has no effect" rather than "this column has no spread",
    // which are completely different claims. np.allclose's default tolerance is
    // rtol=1e-5, atol=1e-8.
    if (std::abs(lo - hi) <= 1e-8 + 1e-5 * std::abs(hi)) {
        throw std::invalid_argument(
            "pd_grid: the percentile range is degenerate (column has no spread "
            "between the two percentiles); choose wider percentiles");
    }

    std::vector<double> axis(grid_resolution);
    for (size_t k = 0; k < grid_resolution; ++k) {
        const double t = static_cast<double>(k) / static_cast<double>(grid_resolution - 1);
        axis[k] = lo + (hi - lo) * t;  // np.linspace(..., endpoint=True)
    }
    // Pin the endpoints so the first and last entries are EXACTLY the empirical
    // quantiles rather than a rounding of lo + (hi-lo)*1.0.
    axis.front() = lo;
    axis.back() = hi;
    return axis;
}

std::vector<double> average_ice(const std::vector<std::vector<double>>& ice) {
    if (ice.empty()) {
        throw std::invalid_argument("average_ice: matrix must have at least one instance");
    }
    const size_t n_points = ice[0].size();
    if (n_points == 0) {
        throw std::invalid_argument("average_ice: curves must have at least one point");
    }
    for (size_t i = 0; i < ice.size(); ++i) {
        if (ice[i].size() != n_points) {
            throw std::invalid_argument("average_ice: every curve must have the same length");
        }
    }
    // The reference averages with `weights=None` (partial_dependence.py:305),
    // i.e. the plain UNWEIGHTED mean over instances.
    std::vector<double> avg(n_points, 0.0);
    for (size_t i = 0; i < ice.size(); ++i) {
        for (size_t k = 0; k < n_points; ++k) avg[k] += ice[i][k];
    }
    const double n = static_cast<double>(ice.size());
    for (size_t k = 0; k < n_points; ++k) avg[k] /= n;
    return avg;
}

std::vector<double> center_pd(const std::vector<double>& pd) {
    if (pd.empty()) {
        throw std::invalid_argument("center_pd: curve must have at least one point");
    }
    const double base = pd[0];
    std::vector<double> out(pd.size());
    for (size_t k = 0; k < pd.size(); ++k) out[k] = pd[k] - base;
    return out;
}

void center_ice(std::vector<std::vector<double>>& ice) {
    if (ice.empty()) {
        throw std::invalid_argument("center_ice: matrix must have at least one instance");
    }
    for (size_t i = 0; i < ice.size(); ++i) {
        if (ice[i].empty()) {
            throw std::invalid_argument("center_ice: curves must have at least one point");
        }
        const double base = ice[i][0];
        for (size_t k = 0; k < ice[i].size(); ++k) ice[i][k] -= base;
    }
}

PartialDependenceResult partial_dependence(Model& model,
                                           const Tensor& X,
                                           size_t feature,
                                           const PartialDependenceConfig& cfg,
                                           size_t target) {
    // Grid first: it validates X and `feature` before anything touches the model.
    const std::vector<double> grid = pd_grid(
        X, feature, cfg.percentile_lo, cfg.percentile_hi, cfg.grid_resolution);
    const size_t n_points = grid.size();
    const size_t n = X.rows;

    // Target validation happens BEFORE any model call so a bad target cannot
    // leave the model's gradients half-touched (same ordering as lime_explain).
    {
        for (auto& lp : model.layers) lp->zero_grad();
        const Tensor probe = model.forward(X);
        if (target >= probe.cols) {
            throw std::out_of_range(
                "partial_dependence: target index out of range for model output");
        }
    }

    // --- Evaluate the whole grid in ONE batched forward pass -----------------
    // Row layout is (grid point k) x (instance i), i.e. row k*n + i. Batching
    // this way is the same reason lime_explain evaluates its neighborhood in a
    // single pass: the model is row-independent, so the batch is just a
    // concatenation.
    //
    // The batch is ALLOCATED at its true size (n_points*n rows). Note the repo's
    // Tensor has no resize: `Tensor::clone()` sizes `data` to rows*cols, so
    // growing `rows` afterwards would leave the buffer short and every write
    // past `n*cols` a heap overflow.
    Tensor batch = Tensor::zeros(n_points * n, X.cols);
    for (size_t k = 0; k < n_points; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < X.cols; ++j) {
                batch(k * n + i, j) = (j == feature) ? grid[k] : X[i][j];
            }
        }
    }

    std::vector<std::vector<double>> responses(n, std::vector<double>(n_points, 0.0));
    {
        Tensor out = model.forward(batch);
        for (size_t k = 0; k < n_points; ++k) {
            for (size_t i = 0; i < n; ++i) {
                responses[i][k] = out[k * n + i][target];
            }
        }
    }

    // The PD average is over EVERY instance, independent of `subsample`
    // (partial_dependence.py:498-499: "the full dataset is still used to
    // calculate averaged partial dependence when kind='both'"). `subsample`
    // limits how many individual curves are KEPT, not what is averaged.
    PartialDependenceResult r;
    r.average = average_ice(responses);

    // Which instances get a retained curve. subsample == 0 means "keep all".
    std::vector<size_t> keep_idx;
    if (cfg.subsample == 0 || cfg.subsample >= n) {
        keep_idx.resize(n);
        for (size_t i = 0; i < n; ++i) keep_idx[i] = i;
    } else {
        Rng rng(cfg.seed);
        // Without replacement (a partial Fisher-Yates prefix), so the retained
        // set is a genuine SUBSET — drawing with replacement would duplicate
        // instances and show the same curve twice in an ICE plot.
        std::vector<size_t> pool(n);
        for (size_t i = 0; i < n; ++i) pool[i] = i;
        for (size_t k = 0; k < cfg.subsample; ++k) {
            const size_t pick = k + rng.below(n - k);
            std::swap(pool[k], pool[pick]);
            keep_idx.push_back(pool[k]);
        }
    }

    r.individual.reserve(keep_idx.size());
    for (size_t idx : keep_idx) r.individual.push_back(responses[idx]);

    if (cfg.centered) {
        r.average = center_pd(r.average);
        center_ice(r.individual);
    }

    r.grid = grid;
    r.feature = feature;
    r.target = target;
    r.n_points = n_points;
    r.n_instances = n;
    r.n_curves = r.individual.size();
    r.centered = cfg.centered;

    // Zero gradients again on the way out, matching lime_explain. See that file's
    // mutation note: nothing between here and the entry zero_grad writes a
    // parameter gradient (this pass never calls backward()), so the two zeroings
    // are behaviourally identical and the observable contract is what matters.
    for (auto& lp : model.layers) lp->zero_grad();
    return r;
}