#include "sample_quality.h"

#include "linear_algebra.h"

#include <cmath>
#include <stdexcept>
#include <string>

// ===========================================================================
// utils/sample_quality.cpp -- the sample-quality metrics: the Frechet
// Inception Distance, the Inception Score, and the sample Gaussian
// estimator they are built on.
//
// THE SPECIFICATION is tests/sample_quality_oracle.py (float64 numpy +
// scipy), which was run BEFORE this file existed. Its discriminating power
// is MEASURED, not narrated: tests/sample_quality_mutate_oracle.py breaks
// the oracle's model code in seventeen specific ways and requires that it
// FAIL each time. The C++ fixtures are EMITTED by that oracle
// (tests/splice_sample_quality_fixture.py), never hand-transcribed.
//
// THE FID PATH COMPOSES `utils::frechet_distance`. There is exactly ONE
// copy of the Gaussian-to-Gaussian distance in this repo -- see the header
// for why that matters.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// ---------------------------------------------------------------------------
// Guards. EVERY message is DISTINCT, for the reason this repo has hit
// several separate times: a needle that appears in two different messages
// discriminates nothing between them, and the mutation harness then
// reports a survivor that is really a missing test.
// ---------------------------------------------------------------------------
void validate_feature_matrix(const Tensor& features, const char* who) {
    if (features.rows == 0)
        throw std::invalid_argument(std::string(who) +
                                    ": the feature matrix has no rows");
    if (features.cols == 0)
        throw std::invalid_argument(std::string(who) +
                                    ": the feature matrix has no columns");
    if (features.rows < 2)
        throw std::invalid_argument(std::string(who) +
                                    ": the feature matrix needs at least 2 "
                                    "rows, because the covariance estimator "
                                    "divides by n-1");
    for (size_t i = 0; i < features.rows; ++i) {
        for (size_t j = 0; j < features.cols; ++j) {
            if (!std::isfinite(features(i, j)))
                throw std::invalid_argument(std::string(who) +
                                            ": the feature matrix has a "
                                            "non-finite entry");
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// sample_gaussian_parameters -- the empirical mean and covariance.
//
// THE UNBIASED ESTIMATOR: divide by n-1. See the header's C2.
//
// THE COVARIANCE IS BUILT BY THE CENTRED OUTER SUM,
// `sum_i (x_i - mu)(x_i - mu)^T`, rather than by `E[xx^T] - mu mu^T`. The
// two are algebraically equal, but the second form is the "naive" one and
// it CATASTROPHICALLY CANCELS when the mean is large relative to the
// spread: `E[xx^T]` and `mu mu^T` are then both large and nearly equal,
// and their difference loses most of its significant digits. The centred
// form never subtracts two large numbers, so it keeps full precision. This
// is the same reason numpy's `cov` centres first.
// ---------------------------------------------------------------------------
GaussianParams sample_gaussian_parameters(const Tensor& features) {
    validate_feature_matrix(features, "sample_gaussian_parameters");

    const size_t n = features.rows;
    const size_t d = features.cols;

    // The mean: a (1, d) ROW, matching this repo's convention for a
    // vector-valued quantity.
    Tensor mean(1, d);
    for (size_t j = 0; j < d; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += features(i, j);
        mean(0, j) = sum / static_cast<double>(n);
    }

    // The covariance: the centred outer sum, divided by n-1.
    Tensor covariance(d, d);
    const double denom = static_cast<double>(n - 1);
    for (size_t j = 0; j < d; ++j) {
        for (size_t k = j; k < d; ++k) {
            double acc = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double dj = features(i, j) - mean(0, j);
                const double dk = features(i, k) - mean(0, k);
                acc += dj * dk;
            }
            const double v = acc / denom;
            covariance(j, k) = v;
            covariance(k, j) = v;   // symmetric by construction
        }
    }

    GaussianParams out;
    out.mean = mean;
    out.covariance = covariance;
    return out;
}

// ---------------------------------------------------------------------------
// frechet_inception_distance -- FID of two feature sets.
//
// **FID IS THE SQUARED FREMCHET DISTANCE** (the header's C1), and the
// shipped `frechet_distance` is REUSED rather than reimplemented (C3).
//
// THE SQUARE IS TAKEN AFTER THE CLAMP INSIDE `frechet_distance`, which is
// what makes `FID(X, X) == 0` EXACT (C4): the shipped routine clamps its
// radicand at zero from below, so the distance -- and therefore its
// square -- is never negative.
// ---------------------------------------------------------------------------
double frechet_inception_distance(const Tensor& real_features,
                                  const Tensor& fake_features) {
    validate_feature_matrix(real_features, "frechet_inception_distance");
    validate_feature_matrix(fake_features, "frechet_inception_distance");
    if (real_features.cols != fake_features.cols)
        throw std::invalid_argument(
            "frechet_inception_distance: the two feature matrices have "
            "different numbers of columns");

    const GaussianParams real = sample_gaussian_parameters(real_features);
    const GaussianParams fake = sample_gaussian_parameters(fake_features);

    const double d = frechet_distance(real.mean, real.covariance,
                                       fake.mean, fake.covariance);
    return d * d;
}

// ---------------------------------------------------------------------------
// inception_score -- the Inception Score of a class-probability matrix.
//
// THE `0 * log(0/q) = 0` CONVENTION (the header's C5) is implemented by
// ZEROING THE LOGARITHM where the probability is zero, so the term is
// `0 * (0 - log q) == 0` rather than `0 * (-inf) == NaN`.
//
// **THERE IS EXACTLY ONE GUARD FOR THAT CONDITION.** A first draft had
// two -- the zeroed logarithm AND an outer `where` that discarded the
// whole term -- and the outer one was REDUNDANT, which meant the
// mutation "remove the zero guard" removed a guard that was not carrying
// the load and the mutation SURVIVED. Two guards for one condition is one
// guard too many, and the redundant one makes the live one untested.
// ---------------------------------------------------------------------------
double inception_score(const Tensor& probabilities, int splits) {
    if (probabilities.rows == 0)
        throw std::invalid_argument(
            "inception_score: the probability matrix has no rows");
    if (probabilities.cols == 0)
        throw std::invalid_argument(
            "inception_score: the probability matrix has no columns");
    if (splits < 1)
        throw std::invalid_argument(
            "inception_score: the number of splits must be at least 1");
    if (static_cast<size_t>(splits) > probabilities.rows)
        throw std::invalid_argument(
            "inception_score: the number of splits exceeds the number of "
            "rows, so a group would be empty");

    const size_t n = probabilities.rows;
    const size_t k = probabilities.cols;

    // Every row must be a DISTRIBUTION: non-negative, finite, and summing
    // to 1. The metric is a KL divergence, which is only a divergence
    // between distributions.
    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (size_t c = 0; c < k; ++c) {
            const double p = probabilities(i, c);
            if (!std::isfinite(p))
                throw std::invalid_argument(
                    "inception_score: the probability matrix has a "
                    "non-finite entry");
            if (p < 0.0)
                throw std::invalid_argument(
                    "inception_score: the probability matrix has a negative "
                    "entry");
            row_sum += p;
        }
        if (std::fabs(row_sum - 1.0) > 1e-6)
            throw std::invalid_argument(
                "inception_score: a row of the probability matrix does not "
                "sum to 1");
    }

    double total = 0.0;
    size_t groups = 0;
    for (int s = 0; s < splits; ++s) {
        const size_t lo = static_cast<size_t>(s) * n / static_cast<size_t>(splits);
        const size_t hi = static_cast<size_t>(s + 1) * n / static_cast<size_t>(splits);
        const size_t rows = hi - lo;
        if (rows == 0) continue;

        // The marginal: the mean of THIS group's rows, over the classes.
        Tensor marginal(1, k);
        for (size_t c = 0; c < k; ++c) {
            double sum = 0.0;
            for (size_t i = lo; i < hi; ++i) sum += probabilities(i, c);
            marginal(0, c) = sum / static_cast<double>(rows);
        }

        // The mean KL over the group's rows.
        double kl_sum = 0.0;
        for (size_t i = lo; i < hi; ++i) {
            double kl = 0.0;
            for (size_t c = 0; c < k; ++c) {
                const double p = probabilities(i, c);
                // THE GUARD -- AND IT IS THE ONLY ONE FOR THIS CONDITION.
                // A zero probability contributes NOTHING, and the logarithm
                // is never evaluated for it, so no -inf is ever produced
                // and the term cannot become NaN.
                //
                // **THERE IS NO SECOND GUARD ON `q`.** A first draft had
                // one (`if (q <= 0.0) continue;`), and it was DEAD CODE:
                // the marginal is the mean of this group's probabilities,
                // so `q > 0` whenever any row has `p > 0` in that class --
                // `q >= p / rows > 0`. A dead guard is not harmless: it
                // makes the LIVE guard look covered, and this repo has
                // paid for exactly that (the conformal session's
                // unreachable `!found` throw caught the bad label, so the
                // live range check went untested and its mutation
                // SURVIVED). The invariant is stated here instead:
                // **q > 0 whenever p > 0**, so `std::log(q)` is finite.
                if (p <= 0.0) continue;
                const double q = marginal(0, c);
                kl += p * (std::log(p) - std::log(q));
            }
            kl_sum += kl;
        }

        total += std::exp(kl_sum / static_cast<double>(rows));
        ++groups;
    }

    return total / static_cast<double>(groups);
}

}  // namespace utils
}  // namespace nn
