#include "sliced_wasserstein.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// utils/sliced_wasserstein.cpp -- the sliced Wasserstein distance.
//
// THE SPECIFICATION is tests/sliced_wasserstein_oracle.py (float64 numpy),
// which was run BEFORE this file existed and caught four defects in its own
// first draft. See that file's header for what they were.
//
// The three things that are easy to get wrong, and that the test suite
// pins:
//
//   1. THE SORTING. The 1-D W1 is the L1 distance between the SORTED
//      projections. Sorting MINIMISES the sum of absolute differences --
//      that is what optimal 1-D transport IS -- so a missing sort
//      OVERESTIMATES the distance.
//
//   2. THE UNIT DIRECTION. The projection is onto a unit vector, and a
//      non-unit direction is REJECTED rather than silently rescaled.
//
//   3. THE GENERAL (UNEQUAL-n) 1-D FORM. The shortcut
//      `mean(|sorted_p - sorted_q|)` is valid only for equal sample
//      counts.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// The tolerance on the unit-length check.
//
// A direction's length is computed as sqrt(sum of squares), so it is not
// exactly 1.0 even when the direction was constructed to be unit. The
// tolerance is loose enough for that rounding and tight enough that a
// direction that is off by a factor of 2 is still rejected.
constexpr double kUnitTolerance = 1e-9;

// ---------------------------------------------------------------------------
// The values of a Tensor, in row-major order, as a flat vector.
//
// The 1-D Wasserstein distance cares about the VALUES, not the shape, so
// this is the right flattening for it: a (n, 1) and a (1, n) Tensor with
// the same values are the same sample.
//
// **THIS IS WHERE THE ORACLE'S TRANSPOSED-MATMUL BUG HAD ITS MIRROR
// IMAGE.** The oracle's `project` returned (n, k) when the caller wanted
// (k, n), and the symptom was that the metric had silently become a
// metric on ONE POINT AT A TIME. Here the analogous mistake would be
// flattening the PROJECTION matrix in the wrong order, so this helper is
// the ONLY place a sample is turned into a list of values.
// ---------------------------------------------------------------------------
std::vector<double> values_of(const Tensor& t) {
    std::vector<double> v;
    v.reserve(t.rows * t.cols);
    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t j = 0; j < t.cols; ++j) {
            v.push_back(t(i, j));
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// The 1-D Wasserstein-1 distance, computed EXACTLY.
//
//     W_1(P, Q) = integral |F_P(t) - F_Q(t)| dt
//
// The merged support is walked once. On each gap between consecutive
// distinct values the CDF difference is CONSTANT -- there is no support
// point strictly inside the gap -- so the contribution is that constant
// times the gap's width.
//
// **THE CDF IS EVALUATED WITH "<=", AND THAT MATTERS AT A SHARED VALUE.**
// Both distributions must have already counted a shared value before the
// gap to its right opens, so `upper_bound` (which counts the values <=
// the point) is the right search, not `lower_bound`.
// ---------------------------------------------------------------------------
double wasserstein1_impl(std::vector<double> p, std::vector<double> q) {
    if (p.empty() || q.empty()) return 0.0;
    std::sort(p.begin(), p.end());
    std::sort(q.begin(), q.end());

    // The merged support.
    std::vector<double> pts;
    pts.reserve(p.size() + q.size());
    std::set_union(p.begin(), p.end(), q.begin(), q.end(), std::back_inserter(pts));

    double total = 0.0;
    const double np = static_cast<double>(p.size());
    const double nq = static_cast<double>(q.size());
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const size_t cp = static_cast<size_t>(
            std::upper_bound(p.begin(), p.end(), pts[i]) - p.begin());
        const size_t cq = static_cast<size_t>(
            std::upper_bound(q.begin(), q.end(), pts[i]) - q.begin());
        const double fp = static_cast<double>(cp) / np;
        const double fq = static_cast<double>(cq) / nq;
        total += std::abs(fp - fq) * (pts[i + 1] - pts[i]);
    }
    return total;
}

// ---------------------------------------------------------------------------
// The projection of a sample set onto ONE direction, as a flat vector.
//
// The 1-D W1 of two projections is the W1 of the two flat vectors, so the
// projection never has to be materialised as a matrix.
// ---------------------------------------------------------------------------
std::vector<double> project_onto(const Tensor& samples, const Tensor& directions,
                                 size_t k) {
    std::vector<double> out;
    out.reserve(samples.rows);
    for (size_t i = 0; i < samples.rows; ++i) {
        double t = 0.0;
        for (size_t j = 0; j < samples.cols; ++j) {
            t += samples(i, j) * directions(k, j);
        }
        out.push_back(t);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The guards. EVERY message here is DISTINCT.
//
// A substring that appears in two different messages discriminates nothing
// between them, and this repo has hit that trap several separate times
// (each time the assertion searched for a word the OTHER guard's message
// also contained, so the mutant survived).
// ---------------------------------------------------------------------------
void validate_inputs(const Tensor& P, const Tensor& Q,
                     const Tensor& directions, const char* who) {
    if (P.rows == 0 || P.cols == 0)
        throw std::invalid_argument(std::string(who) + ": the first sample set is empty");
    if (Q.rows == 0 || Q.cols == 0)
        throw std::invalid_argument(std::string(who) + ": the second sample set is empty");
    if (directions.rows == 0 || directions.cols == 0)
        throw std::invalid_argument(std::string(who) + ": the direction set is empty");
    if (P.cols != Q.cols)
        throw std::invalid_argument(
            std::string(who) + ": the two sample sets have different dimensions");
    if (directions.cols != P.cols)
        throw std::invalid_argument(
            std::string(who) + ": the directions' dimension does not match the samples'");

    // THE UNIT CHECK. A non-unit direction scales the answer by its length,
    // which is a DIFFERENT metric from the one the definition names.
    for (size_t k = 0; k < directions.rows; ++k) {
        double norm_sq = 0.0;
        for (size_t j = 0; j < directions.cols; ++j) {
            norm_sq += directions(k, j) * directions(k, j);
        }
        const double norm = std::sqrt(norm_sq);
        if (!(std::abs(norm - 1.0) <= kUnitTolerance)) {
            throw std::invalid_argument(
                std::string(who) + ": every direction must be a unit vector");
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// wasserstein1 -- the public 1-D Wasserstein-1 distance.
// ---------------------------------------------------------------------------
double wasserstein1(const Tensor& p, const Tensor& q) {
    if (p.rows == 0 || p.cols == 0)
        throw std::invalid_argument("wasserstein1: the first sample is empty");
    if (q.rows == 0 || q.cols == 0)
        throw std::invalid_argument("wasserstein1: the second sample is empty");
    return wasserstein1_impl(values_of(p), values_of(q));
}

// ---------------------------------------------------------------------------
// sliced_wasserstein_distance -- the MEAN over the given directions.
// ---------------------------------------------------------------------------
double sliced_wasserstein_distance(const Tensor& P, const Tensor& Q,
                                   const Tensor& directions) {
    validate_inputs(P, Q, directions, "sliced_wasserstein_distance");

    double total = 0.0;
    for (size_t k = 0; k < directions.rows; ++k) {
        total += wasserstein1_impl(project_onto(P, directions, k),
                                   project_onto(Q, directions, k));
    }
    return total / static_cast<double>(directions.rows);
}

// ---------------------------------------------------------------------------
// max_sliced_wasserstein_distance -- the MAX over the given directions.
// ---------------------------------------------------------------------------
double max_sliced_wasserstein_distance(const Tensor& P, const Tensor& Q,
                                        const Tensor& directions) {
    validate_inputs(P, Q, directions, "max_sliced_wasserstein_distance");

    double best = -1.0;
    for (size_t k = 0; k < directions.rows; ++k) {
        const double w = wasserstein1_impl(project_onto(P, directions, k),
                                           project_onto(Q, directions, k));
        if (w > best) best = w;
    }
    return best;
}

// ---------------------------------------------------------------------------
// random_unit_directions -- uniform draws on the unit sphere.
//
// A standard normal vector, normalised, is uniform on the sphere. The
// zero-norm case has probability zero, but the division must not be a NaN
// generator if it ever happens, so it is handled explicitly.
// ---------------------------------------------------------------------------
Tensor random_unit_directions(int num_projections, int dim, uint64_t seed) {
    if (num_projections < 1)
        throw std::invalid_argument(
            "random_unit_directions: the projection count must be >= 1");
    if (dim < 1)
        throw std::invalid_argument(
            "random_unit_directions: the dimension must be >= 1");

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    Tensor dirs(static_cast<size_t>(num_projections), static_cast<size_t>(dim));
    for (size_t i = 0; i < dirs.rows; ++i) {
        double norm_sq = 0.0;
        for (size_t j = 0; j < dirs.cols; ++j) {
            const double v = normal(rng);
            dirs(i, j) = v;
            norm_sq += v * v;
        }
        double norm = std::sqrt(norm_sq);
        if (norm == 0.0) {
            // Probability zero, but a NaN here would poison every
            // downstream number silently.
            dirs(i, 0) = 1.0;
            norm = 1.0;
        }
        for (size_t j = 0; j < dirs.cols; ++j) dirs(i, j) /= norm;
    }
    return dirs;
}

// ---------------------------------------------------------------------------
// sliced_wasserstein_distance -- the random-direction form.
//
// The seed is part of the contract: the same seed gives the same answer.
// The directions are unit by construction, so the unit check in
// `validate_inputs` passes by definition rather than by luck.
// ---------------------------------------------------------------------------
double sliced_wasserstein_distance(const Tensor& P, const Tensor& Q,
                                   int num_projections, uint64_t seed) {
    if (num_projections < 1)
        throw std::invalid_argument(
            "sliced_wasserstein_distance: the projection count must be >= 1");
    const Tensor dirs =
        random_unit_directions(num_projections, static_cast<int>(P.cols), seed);
    return sliced_wasserstein_distance(P, Q, dirs);
}

double max_sliced_wasserstein_distance(const Tensor& P, const Tensor& Q,
                                        int num_projections, uint64_t seed) {
    if (num_projections < 1)
        throw std::invalid_argument(
            "max_sliced_wasserstein_distance: the projection count must be >= 1");
    const Tensor dirs =
        random_unit_directions(num_projections, static_cast<int>(P.cols), seed);
    return max_sliced_wasserstein_distance(P, Q, dirs);
}

}  // namespace utils
}  // namespace nn
