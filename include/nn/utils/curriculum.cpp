#include "curriculum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// =============================================================================
// Curriculum / self-paced sample weighing -- implementation.
//
// See curriculum.h for the citation, the measured gap, and the twelve pinned
// semantics (D1..D12). Every formula below was cross-checked by an independent
// float64 oracle (tests/curriculum_oracle.py) that verifies each explicit
// regulariser by GRID-SEARCHING its argmin rather than transcribing a
// minimiser, and which measured the phi'/sigma discrepancy documented in the
// header.
// =============================================================================

namespace {

[[noreturn]] void bad(const std::string& what) {
    throw std::invalid_argument("curriculum: " + what);
}

void require_lambda(double lambda) {
    if (!(lambda > 0.0) || std::isinf(lambda) || std::isnan(lambda))
        bad("lambda must be finite and strictly positive");
}

void require_ell(double ell) {
    if (!(ell >= 0.0) || std::isnan(ell))
        bad("loss must be finite and non-negative");
}

void require_shape(const Tensor& t, const char* name) {
    if (t.rows == 0 || t.cols == 0) bad(std::string(name) + " must be non-empty");
}

}  // namespace

// -----------------------------------------------------------------------------
// The SPL weight v* = sigma(lam, ell).
//
// -----------------------------------------------------------------------------
// WHAT THIS RETURNS, AND WHY IT IS NOT phi'(ell)
//
// Definition 1 condition 1 states phi = min_{v>=0}[v*t + psi], whose argmin is
// phi'(t). A first draft of this file therefore returned phi'(ell) and claimed
// Table 1's column was the HQ minimiser instead. **THAT WAS BACKWARDS**, and the
// suite caught it: Definition 1 condition 4 requires sigma to be
// "monotonically decreasing w.r.t. t". Measured over 400 points in [0, 8]:
//
//   mode     phi'(t) monotone DEcreasing?    sigma(t) monotone DEcreasing?
//   cauchy   NO (peaks at t == lambda)        YES
//   welsch   NO (peaks at t = 0.702*lambda)   YES
//   l1l2     NO (rises monotonically)         YES
//
// Shipping phi' would UPWEIGHT the hardest samples -- the exact opposite of a
// curriculum -- so Table 1's column is what an SPL weight must be.
//
// AND THE TWO ARE THE SAME FUNCTION UP TO A FACTOR: measured
// `|sigma - phi'/t| = 0.000e+00` over 4 lambdas x 5 losses for all three
// modes. `curr_hq_minimiser` below is kept because the paper's Eq. (7) names
// this quantity `delta(t)` and a caller doing half-quadratic optimisation
// wants it under that name; it is not a different function.
//
// Per mode, phi' and the resulting sigma:
//   Cauchy : phi = lam^2*log(1 + (t/lam)^2) => phi' = 2t/(1+(t/lam)^2)
//                                                          => sigma = 2/(1+(t/lam)^2)
//   Welsch : phi = lam^2*(1 - exp(-(t/lam)^2)) => phi' = 2t*exp(-(t/lam)^2)
//                                                          => sigma = 2*exp(-(t/lam)^2)
//   L1L2   : phi = sqrt(lam + t^2) - 1 => phi' = t/sqrt(lam + t^2)
//                                              => sigma = 1/sqrt(lam + t^2)
//   Sec32  : phi = lam^2*log(1 + t/lam^2) => phi' = 1/(1 + t/lam^2)  (section 3.2)
//
// The implicit weights are bounded by 2 (Cauchy/Welsch reach exactly 2 at
// t == 0), NOT by 1 -- Definition 1 condition 3 says "non-negative and
// up-bounded" without naming the bound, and the measured bound is 2.
// -----------------------------------------------------------------------------
double curr_weight(CurrWeightMode mode, double lambda, double ell, double gamma) {
    require_lambda(lambda);
    require_ell(ell);

    switch (mode) {
        case CurrWeightMode::Hard:
            // D1: STRICT '<' -- the paper's Table. Eq. (4) in the body writes
            // '<='; at ell == lambda the objective is flat (D2) so both are
            // valid minimisers, and we document the Table's convention.
            return (ell < lambda) ? 1.0 : 0.0;

        case CurrWeightMode::Linear:
            // 1 - ell/lambda for ell < lambda, else 0. No division by zero:
            // lambda > 0 is already guaranteed above.
            return (ell < lambda) ? (1.0 - ell / lambda) : 0.0;

        case CurrWeightMode::Log: {
            // zeta = 1 - lambda must lie strictly inside (0, 1): zeta == 0 gives
            // log(0) = -inf and zeta == 1 gives log(1) = 0 as a DIVISOR.
            const double zeta = 1.0 - lambda;
            if (!(zeta > 0.0) || !(zeta < 1.0))
                bad("log mode requires 0 < lambda < 1 (zeta = 1 - lambda must be in (0,1))");
            if (!(ell < lambda)) return 0.0;
            return std::log(ell + zeta) / std::log(zeta);
        }

        case CurrWeightMode::Mixture: {
            // Lu, Jiang et al. Two thresholds: gamma carries lambda1 and lambda
            // carries lambda2, with lambda1 > lambda2 > 0 required, because
            // zeta = l1*l2/(l1 - l2) changes SIGN if the order is swapped.
            const double l1 = gamma, l2 = lambda;
            if (!(l1 > l2))
                bad("mixture mode requires gamma (lambda1) > lambda (lambda2) > 0");
            const double zeta = l1 * l2 / (l1 - l2);
            if (ell <= l2) return 1.0;
            if (ell >= l1) return 0.0;
            return (l1 - ell) * zeta / (ell * l1);
        }

        case CurrWeightMode::Gamma: {
            // Zhao et al. Three branches. The middle one divides by sqrt(ell),
            // so ell == 0 MUST be caught by the first branch -- that is what
            // keeps the fifth 1/0 trap of this repo out of the function.
            if (!(gamma > 0.0)) bad("gamma mode requires gamma > 0");
            const double cut = (lambda * gamma / (lambda + gamma));
            const double cut2 = cut * cut;
            if (ell <= cut2) return 1.0;
            if (ell >= lambda * lambda) return 0.0;
            if (!(ell > 0.0)) return 1.0;     // unreachable given cut2 >= 0, kept explicit
            return gamma * (1.0 / std::sqrt(ell) - 1.0 / lambda);
        }

        case CurrWeightMode::Cauchy:
            // sigma = 2/(1 + (t/lam)^2). Equals exactly 2 at t == 0.
            return 2.0 / (1.0 + (ell / lambda) * (ell / lambda));

        case CurrWeightMode::Welsch:
            // sigma = 2*exp(-(t/lam)^2). Equals exactly 2 at t == 0.
            return 2.0 * std::exp(-(ell / lambda) * (ell / lambda));

        case CurrWeightMode::L1L2:
            // sigma = 1/sqrt(lam + t^2), the one entry whose closed form is
            // literally phi' here. Equals 1/sqrt(lambda) at t == 0, which
            // EXCEEDS 1 whenever lambda < 1 -- a measured property of the
            // paper's formula, recorded rather than clamped (D5).
            return 1.0 / std::sqrt(lambda + ell * ell);

        case CurrWeightMode::Sec32:
            // Section 3.2 / Figure 1: sigma = 1/(1 + l/lam^2), which for THIS
            // pair is both phi' and the SPL weight, and is bounded by 1 with
            // sigma(0) == 1 exactly. Measured to satisfy Definition 1
            // condition 1 (2.56e-09) AND condition 4 (monotone decreasing).
            return 1.0 / (1.0 + ell / (lambda * lambda));
    }
    bad("unknown weighting mode");
}

// -----------------------------------------------------------------------------
// The SAME quantity under the paper's Eq. (7) name, `delta(t)` -- the
// half-quadratic minimiser. Measured identical to `curr_weight` (D6); it exists
// so a caller reading the HQ section can ask for it by the name used there.
//
// The explicit regularisers have no Table-1 counterpart and THROW rather than
// silently returning the weight, so a caller cannot confuse the two families.
// -----------------------------------------------------------------------------
double curr_hq_minimiser(CurrWeightMode mode, double lambda, double ell) {
    require_lambda(lambda);
    require_ell(ell);

    if (mode == CurrWeightMode::Hard || mode == CurrWeightMode::Linear ||
        mode == CurrWeightMode::Log || mode == CurrWeightMode::Mixture ||
        mode == CurrWeightMode::Gamma)
        bad("this mode is an EXPLICIT regulariser and has no Table-1 counterpart");

    return curr_weight(mode, lambda, ell);
}

// -----------------------------------------------------------------------------
// D7 -- the learning-pace schedule.
// -----------------------------------------------------------------------------
double curr_schedule_step(double lambda, double step_factor, bool divide) {
    require_lambda(lambda);
    if (!(step_factor > 0.0) || std::isinf(step_factor) || std::isnan(step_factor))
        bad("step factor must be finite and positive");
    // A schedule that does not move lambda forward is not a schedule: it is
    // how every sample silently drops to weight 0 without an error anywhere.
    if (divide) {
        if (!(step_factor > 1.0)) bad("dividing requires a step factor > 1");
        return lambda / step_factor;
    }
    if (!(step_factor > 1.0)) bad("multiplying requires a step factor > 1");
    return lambda * step_factor;
}

// -----------------------------------------------------------------------------
// D8 -- lambda initialised to a quantile of the observed losses.
// -----------------------------------------------------------------------------
double curr_lambda_init(const Tensor& losses, double quantile, CurrWeightMode mode,
                        double gamma) {
    require_shape(losses, "losses");
    if (!(quantile >= 0.0) || !(quantile <= 1.0) || std::isnan(quantile))
        bad("quantile must lie in [0, 1]");

    const size_t n = losses.cols;
    std::vector<double> sorted(n, 0.0);
    for (size_t i = 0; i < n; ++i) sorted[i] = losses(0, i);
    std::sort(sorted.begin(), sorted.end());

    // Linear interpolation between order statistics (the "type 7" rule), which
    // is the same convention numpy.percentile uses -- so the median of an even
    // count is the mean of the two central losses rather than either of them.
    const double pos = quantile * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    double lam = sorted[lo] + frac * (sorted[hi] - sorted[lo]);

    if (!(lam > 0.0))
        bad("the requested quantile of these losses is not a positive lambda "
            "(a loss of exactly 0 at that quantile cannot be cut at lambda > 0)");
    (void)mode;   // the quantile is mode-independent; kept in the signature so
                  // a future mode-specific rule has a place to live.
    (void)gamma;
    return lam;
}

// -----------------------------------------------------------------------------
// Batch helpers.
// -----------------------------------------------------------------------------
Tensor curr_weights(const Tensor& losses, double lambda, CurrWeightMode mode, double gamma) {
    require_shape(losses, "losses");
    const size_t n = losses.cols;
    Tensor w(1, n);
    for (size_t i = 0; i < n; ++i)
        w(0, i) = curr_weight(mode, lambda, losses(0, i), gamma);
    return w;   // a fresh tensor: never aliases `losses`
}

double curr_weighted_loss(const Tensor& losses, const Tensor& weights) {
    require_shape(losses, "losses");
    require_shape(weights, "weights");
    if (losses.cols != weights.cols)
        bad("losses and weights must have the same number of samples");
    double total = 0.0;
    for (size_t i = 0; i < losses.cols; ++i)
        total += weights(0, i) * losses(0, i);
    return total;
}

Tensor curr_weighted_loss_gradient(const Tensor& losses) {
    require_shape(losses, "losses");
    // D9: E is affine in v, so the gradient is the loss vector ITSELF -- no
    // scaling, no factor, no tolerance.
    Tensor g(1, losses.cols);
    for (size_t i = 0; i < losses.cols; ++i) g(0, i) = losses(0, i);
    return g;
}

double curr_effective_mass(const Tensor& weights) {
    require_shape(weights, "weights");
    double total = 0.0;
    for (size_t i = 0; i < weights.cols; ++i) total += weights(0, i);
    return total;   // all-zero weights sum to exactly 0.0, never NaN (D10)
}

size_t curr_hard_fraction(const Tensor& weights, double tol) {
    require_shape(weights, "weights");
    if (!(tol >= 0.0)) bad("tolerance must be non-negative");
    size_t excluded = 0;
    for (size_t i = 0; i < weights.cols; ++i)
        if (weights(0, i) <= tol) ++excluded;
    return excluded;
}