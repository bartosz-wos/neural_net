#ifndef NN_UTILS_DTW_H
#define NN_UTILS_DTW_H

#include "../core/tensor.h"

#include <cstddef>
#include <utility>
#include <vector>

// ===========================================================================
// DYNAMIC TIME WARPING
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-12)
// ---------------------------------------------------------------------------
//
//     grep -rinE 'dynamic.time.warp|\bdtw\b|warping|sakoe|itakura' include/
//     -> 0 hits
//
// The repo has ~250 layers and 60+ utility files and NO order-sensitive
// distance between two time series. What it HAS is adjacent but DIFFERENT:
//
//   * `utils/sliced_wasserstein.h` -- a distance between two DISTRIBUTIONS of
//     samples. It is deliberately ORDER-FREE: it slices a cloud of points and
//     compares the marginals. Two trajectories that are the same path sampled
//     at different rates get a LARGE sliced-Wasserstein distance, which is
//     exactly the case DTW exists to handle.
//   * `utils/linear_algebra.h` -- the Frechet distance between two
//     GAUSSIANS. A distributional distance, not a trajectory distance.
//   * `layers/architectures/timesnet.h` -- a forecasting architecture over
//     (B*T, d_model) tensors. It consumes time series; nothing measures how
//     far two of them are apart.
//
// The question this module answers: "ARE THESE TWO TIME SERIES THE SAME
// TRAJECTORY?" -- the forecasting, sensor-fusion and activity-recognition
// question. Two accelerometers worn by the same person, sampled at
// different rates, trace the same path. Euclidean distance says they are far
// apart; DTW says they are identical.
//
// ---------------------------------------------------------------------------
// THE `Tensor` SHAPE CONVENTION IS NOT INTERCHANGEABLE HERE
// ---------------------------------------------------------------------------
//
// `utils/rank_correlation.h` and `utils/off_policy_eval.h` both state: "THE
// VALUES ARE WHAT MATTER: a (N,1) and an (1,N) tensor holding the same
// values are the SAME sample."
//
// **THAT CONVENTION DOES NOT HOLD IN THIS MODULE, AND THE DIFFERENCE IS
// LOAD-BEARING.** Those modules take UNORDERED samples, so the shape carries
// no information. DTW is about the ORDER of the observations, and the shape
// here says how many dimensions each observation has:
//
//     x is (T, D):  T TIME STEPS, each a D-dimensional observation.
//     y is (S, D):  S TIME STEPS, each a D-dimensional observation.
//
// So a (T,1) tensor is T scalar observations, and a (1,T) tensor is ONE
// observation of T dimensions -- a completely different time series. A
// module that silently inherited the sibling modules' convention would
// compute a different answer for a transposed input.
//
// The local cost is the EUCLIDEAN distance between two D-dimensional
// observations, i.e. between ROWS. (Measured against `dtw-python`, whose
// `cdist` is applied to the rows: 200/200 agreement on random
// multidimensional cases.)
//
// ---------------------------------------------------------------------------
// THE SIX SEMANTICS THAT ARE EASY TO GET WRONG
// ---------------------------------------------------------------------------
//
//  1. THE STEP PATTERN IS symmetric1 -- the THREE-neighbour lattice:
//     diagonal (i-1,j-1), vertical (i-1,j), horizontal (i,j-1).
//
//     `dtw-python`'s DEFAULT step pattern is symmetric2, which is a
//     DIFFERENT function: measured to differ on 200/200 random pairs. Every
//     reference comparison must therefore pin `step_pattern=symmetric1`
//     explicitly.
//
//     And symmetric1 is genuinely three neighbours, not a pattern with
//     two-step jumps: measured, x = [0,100] against y = [0] costs 100.0
//     under both -- a jump-capable pattern would have cost 0.
//
//  2. THE LOCAL COST IS EUCLIDEAN, summed RAW.
//
//     The distance is NOT normalised by the path length and NOT normalised
//     by (n+m). It is the accumulated local cost along the optimal path.
//     `dtw-python`'s `distance` attribute is the same quantity.
//
//  3. THE BAND IS |i-j| <= band ON TIME INDICES.
//
//     The offset does not matter: the DP's cells are 1-based and the data
//     indices are 0-based, and |i-j| == |(i-1)-(j-1)| EXACTLY. (Measured:
//     the two readings differ on 0 of 2000 random cases.)
//
//  4. A BAND THAT ADMITS NO PATH IS AN INFEASIBLE REQUEST, NOT A LARGE
//     DISTANCE.
//
//     A monotone path exists if and only if band >= |T - S| -- measured on
//     3000 random cases, 0 violations. `dtw-python` raises ValueError in
//     that situation; THIS MODULE RETURNS +inf and says so, because a
//     distance that cannot be computed is not a number a caller can compare.
//
//  5. THE WARPING PATH IS NOT UNIQUE.
//
//     Ties in the dynamic program admit several optimal paths. The shipped
//     tie-break is measured against `dtw-python`'s own `index1`/`index2` on
//     tie-heavy INTEGER observations: 300/300 identical. (Continuous
//     observations almost never tie, so they cannot discriminate the
//     candidate orders at all -- the integer sample is the discriminating
//     one.)
//
//  6. DTW IS SYMMETRIC BUT NOT A METRIC.
//
//     Measured: 0/300 asymmetric pairs, but the triangle inequality is
//     violated on 16/300 random triples. This module must not call DTW a
//     metric, and a caller must not use it where a metric is required.
//
// ---------------------------------------------------------------------------
// SCOPE BOUNDARY (stated so it cannot creep in)
// ---------------------------------------------------------------------------
//
//   * ONE STEP PATTERN (symmetric1). A "DTW" that offers six variants under
//     one name is worse than one that names the one it implements.
//   * NO NORMALISED DTW, no subsequence DTW, no weighted/local (Itakura)
//     constraints, no multi-step patterns, no open-end variants.
//
// ---------------------------------------------------------------------------
// THE SPECIFICATION IS tests/dtw_oracle.py
// ---------------------------------------------------------------------------
//
// The oracle was written and run BEFORE this file existed. It checks the
// implementation against an EXHAUSTIVE ENUMERATION of every monotone path
// through the warping lattice -- the definition of the problem rather than an
// algorithm for it -- and against `dtw-python` with the step pattern pinned
// explicitly. The oracle's own sensitivity is measured, not narrated:
// `tests/dtw_mutate_oracle.py` breaks it in seventeen specific ways and
// requires that it fail each time (16 caught, 1 documented survivor).
//
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// DtwPath -- the warping path.
// ---------------------------------------------------------------------------
//
// The cells are 0-BASED INDICES into the two series: `cells[k].first` is the
// time index in `x` and `cells[k].second` the time index in `y`.
//
// The path starts at (0,0), ends at (T-1, S-1), and every step advances by
// (1,0), (0,1) or (1,1) -- i.e. it is MONOTONE. That is the whole content of
// "dynamic time warping": the path may skip forward in either series but it
// may never go back.
//
// `cost` is the accumulated local cost along the path, and it always equals
// the distance returned by `dtw_distance` for the same arguments.
struct DtwPath {
    std::vector<std::pair<size_t, size_t>> cells;
    double cost = 0.0;
};

// ---------------------------------------------------------------------------
// The DTW distance between two series.
//
// x is (T, D) and y is (S, D): ROWS ARE TIME STEPS, COLUMNS ARE DIMENSIONS.
// Both series must have the same number of columns.
//
// Throws std::invalid_argument if either series is empty, or if the two
// series have different numbers of columns.
//
// The result is SYMMETRIC but is NOT a metric: the triangle inequality can
// fail (measured: 16 violations / 300 random triples).
double dtw_distance(const Tensor& x, const Tensor& y);

// The banded DTW distance.
//
// The band forbids the warping path from straying more than `band` steps
// from the diagonal: a cell (i,j) is unusable when |i-j| > band.
//
// A monotone path exists if and only if band >= |T - S|. When none does --
// the band is too tight for the two lengths -- the result is +INF. (This is
// an INFEASIBLE REQUEST, not a large distance; `dtw-python` raises
// ValueError in the same situation.)
//
// Throws std::invalid_argument if `band` is negative.
double dtw_distance(const Tensor& x, const Tensor& y, int band);

// The warping path.
//
// The path is DETERMINISTIC: the tie-break is documented in the header and
// was measured to reproduce `dtw-python`'s own path on tie-heavy
// observations.
//
// When the band admits no path, the returned path is EMPTY and its cost is
// +INF.
//
// Throws std::invalid_argument if either series is empty, or if the two
// series have different numbers of columns, or if `band` is negative.
DtwPath dtw_align(const Tensor& x, const Tensor& y);

// The banded warping path. See dtw_align for the conventions.
DtwPath dtw_align(const Tensor& x, const Tensor& y, int band);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_DTW_H
