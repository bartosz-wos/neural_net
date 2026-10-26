#include "conformal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// ===========================================================================
// utils/conformal.cpp -- split conformal prediction.
//
// The specification is tests/conformal_oracle.py (64 checks, green), which
// found SIX defects in its own first draft. Two of them are worth restating
// here because they decided the shape of the code:
//
//   1. THE CALIBRATION INDEX IS COMPUTED EXACTLY. The textbook float
//      expression ceil((n+1)*(1-alpha)) is WRONG for some (n, alpha) pairs
//      (measured at alpha = 2/3, n = 2: the float form gives k = 1 where the
//      exact form gives k = 2), and the error is a WHOLE CALIBRATION POINT.
//      So alpha is decomposed into its EXACT binary value and the index is
//      computed in 128-bit integer arithmetic.
//
//   2. AN INSUFFICIENT CALIBRATION SET GIVES +inf, NOT A CLAMPED INDEX.
//      With n = 4 and alpha = 0.1 the required index is 5 > 4, the
//      `inf{...}` form of the theorem is +infinity, and the prediction set
//      is the ENTIRE LABEL SPACE. Clamping the index to n would return a
//      finite threshold and break the coverage guarantee.
//
// The reference for the theorems is the header.
// ===========================================================================

namespace nn {
namespace utils {

namespace {

// Decompose a finite, non-negative double into its EXACT binary value:
//     alpha = numerator / 2^shift
// with `numerator` an unsigned __int128 and `shift` the number of binary
// places. `alpha` must be finite and in [0, 1].
//
// Why exact: `ceil((n+1)*(1-alpha))` computed in floating point is off by one
// for some (n, alpha) -- notably alpha = 2/3, where the double 0.666... is
// slightly BELOW 2/3 and the product lands just below an integer. Using the
// exact value of the double the user passed removes the ambiguity: the
// answer is the mathematically correct one for THAT double.
//
// The decomposition is exact because every finite double is m * 2^e with m
// an integer. `frexp` gives the mantissa and exponent; the mantissa is
// converted to an integer and shifted into place.
void decompose_exact(double x, unsigned __int128& num, int& shift) {
    // frexp returns f in [0.5, 1) and x = f * 2^exp, so x = f * 2^exp with
    // f = m / 2^53 for a double.
    int exp = 0;
    double f = std::frexp(x, &exp);
    // f has at most 53 significant bits; scale it to an integer mantissa.
    std::uint64_t mant = static_cast<std::uint64_t>(std::ldexp(f, 53));
    num = static_cast<unsigned __int128>(mant);
    shift = 53 - exp;
    if (shift < 0) {
        // Only reachable for x >= 1, i.e. x == 1.0 exactly here (alpha <= 1).
        num <<= -shift;
        shift = 0;
    }
}

// ceil((n+1) * (1 - alpha)) computed exactly.
//
// Uses the identity ceil(m - m*alpha) == m - floor(m*alpha) for integer m
// and real alpha, with m = n+1. floor(m*alpha) is computed in exact integer
// arithmetic: m*alpha = m*num / 2^shift, so
//
//     floor(m*alpha) = (m*num) >> shift
//
// and the shift is exact for a non-negative numerator.
size_t exact_index(size_t n, double alpha) {
    const size_t m = n + 1;
    unsigned __int128 num = 0;
    int shift = 0;
    decompose_exact(alpha, num, shift);

    // m * num can be large; 128 bits hold it comfortably for any realistic n.
    unsigned __int128 prod = static_cast<unsigned __int128>(m) * num;
    unsigned __int128 floored = prod >> shift;
    return m - static_cast<size_t>(floored);
}

void check_alpha(double alpha) {
    if (!std::isfinite(alpha)) {
        throw std::invalid_argument("conformal: alpha must be finite");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("conformal: alpha must be in [0, 1], got " +
                                    std::to_string(alpha));
    }
}

}  // namespace

size_t conformal_quantile_index(size_t n, double alpha) {
    if (n == 0) {
        throw std::invalid_argument("conformal: calibration set must be non-empty");
    }
    check_alpha(alpha);
    return exact_index(n, alpha);
}

bool conformal_quantile_is_infinite(size_t n, double alpha) {
    return conformal_quantile_index(n, alpha) > n;
}

double conformal_quantile(const Tensor& scores, double alpha) {
    if (scores.rows == 0 || scores.cols == 0) {
        throw std::invalid_argument("conformal: calibration scores must be non-empty");
    }
    check_alpha(alpha);

    // EVERY score is checked for finiteness BEFORE anything else. This is not
    // cosmetic: the `k > n` early return below would otherwise short-circuit
    // the check and return +infinity for a calibration set that contains a
    // NaN, silently turning a corrupted calibration set into "the threshold
    // is infinite" instead of an error. A NaN that reaches `nth_element` is
    // also not guaranteed to be selected at all, so the check cannot be left
    // to the selection step.
    const size_t n = scores.rows;
    std::vector<double> v;
    v.reserve(n * scores.cols);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < scores.cols; ++j) {
            const double s = scores(i, j);
            if (!std::isfinite(s)) {
                throw std::invalid_argument("conformal: calibration scores must be finite");
            }
            v.push_back(s);
        }
    }

    const size_t k = conformal_quantile_index(n, alpha);
    if (k > n) {
        // Not enough calibration data to certify the requested coverage: the
        // `inf{...}` form of the theorem is +infinity.
        //
        // THIS GUARD IS NOT REDUNDANT WITH THE `v.size() < k` ONE BELOW, and
        // the mutation harness proved it: removing THIS line left the suite
        // GREEN, because every fixture used a ONE-COLUMN score tensor, for
        // which `v.size() == n` and the two guards are equivalent.
        //
        // They DIVERGE for a MULTI-COLUMN tensor, where `v.size() == n*cols`
        // exceeds `n`: the `v.size() < k` guard then never fires, and this
        // one is the ONLY thing that returns +inf. Without it, a (4, 3)
        // calibration tensor at alpha = 0.1 would return a FINITE threshold
        // where the theorem says +infinity -- silently breaking the coverage
        // guarantee instead of reporting that the calibration set is too
        // small.
        return std::numeric_limits<double>::infinity();
    }
    if (v.size() < k) {
        // More entries were required than the tensor holds.
        return std::numeric_limits<double>::infinity();
    }
    // The quantile is an ORDER STATISTIC -- the k-th smallest score, exactly.
    // Sorting a copy leaves the caller's tensor untouched.
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k - 1), v.end());
    return v[k - 1];
}

Tensor aps_score(const Tensor& probs, const Tensor& labels) {
    if (probs.rows != labels.rows) {
        throw std::invalid_argument("conformal: aps_score row mismatch");
    }
    if (probs.cols == 0) {
        throw std::invalid_argument("conformal: aps_score needs at least one class");
    }
    const size_t K = probs.cols;

    Tensor out(probs.rows, 1);
    std::vector<size_t> order(K);
    std::vector<double> cum(K);

    for (size_t i = 0; i < probs.rows; ++i) {
        // The label of row i: `labels` is (n, 1), so read column 0. A
        // non-integer or out-of-range index throws.
        const double lbl = labels(i, 0);
        if (!std::isfinite(lbl)) {
            throw std::invalid_argument("conformal: aps_score label must be finite");
        }
        const long long y = static_cast<long long>(std::llround(lbl));
        if (static_cast<double>(y) != lbl) {
            throw std::invalid_argument("conformal: aps_score label must be an integer");
        }
        if (y < 0 || y >= static_cast<long long>(K)) {
            throw std::invalid_argument("conformal: aps_score label out of range");
        }

        // Rank the classes by probability DESCENDING, ties broken by class
        // index ASCENDING. `std::stable_sort` with a `>` comparison keeps the
        // original (index) order within a tie, which is exactly the rule.
        for (size_t c = 0; c < K; ++c) order[c] = c;
        std::stable_sort(order.begin(), order.end(),
                         [&probs, i](size_t a, size_t b) {
                             return probs(i, a) > probs(i, b);
                         });

        // The cumulative mass THROUGH each class; the score is the mass
        // through the TRUE class.
        //
        // The `order` vector is a PERMUTATION of {0..K-1}, so an in-range
        // label is ALWAYS present in it. The range check above has already
        // rejected every out-of-range label, so there is no "not found" case
        // to handle here -- an earlier draft carried a `found` flag and a
        // `!found` throw, and the mutation harness proved that flag was
        // UNREACHABLE (removing the range guard M20 left the suite green,
        // because the dead `!found` throw caught the bad label instead).
        // Two guards for one condition is one guard too many: the dead one
        // made the live one untested.
        double running = 0.0;
        double score = 0.0;
        for (size_t r = 0; r < K; ++r) {
            running += probs(i, order[r]);
            cum[r] = running;
            if (order[r] == static_cast<size_t>(y)) {
                score = running;
            }
        }
        out(i, 0) = score;
    }
    return out;
}

Tensor cqr_score(const Tensor& lower, const Tensor& upper, const Tensor& y) {
    if (lower.rows != upper.rows || lower.rows != y.rows ||
        lower.cols != upper.cols || lower.cols != y.cols) {
        throw std::invalid_argument("conformal: cqr_score shape mismatch");
    }
    Tensor out(lower.rows, lower.cols);
    for (size_t i = 0; i < lower.rows; ++i) {
        for (size_t j = 0; j < lower.cols; ++j) {
            // The score is a MAX of the two one-sided gaps. It is NOT clamped
            // at zero: a target strictly inside the interval scores NEGATIVE,
            // and that negativity is what lets CQR shrink the interval.
            const double a = lower(i, j) - y(i, j);
            const double b = y(i, j) - upper(i, j);
            out(i, j) = std::max(a, b);
        }
    }
    return out;
}

std::vector<int> aps_prediction_set(const Tensor& probs_row, double qhat) {
    if (probs_row.rows != 1) {
        throw std::invalid_argument("conformal: aps_prediction_set expects one row");
    }
    const size_t K = probs_row.cols;
    if (K == 0) {
        throw std::invalid_argument("conformal: aps_prediction_set needs at least one class");
    }

    std::vector<size_t> order(K);
    for (size_t c = 0; c < K; ++c) order[c] = c;
    std::stable_sort(order.begin(), order.end(),
                     [&probs_row](size_t a, size_t b) {
                         return probs_row(0, a) > probs_row(0, b);
                     });

    // Walk the classes in rank order, accumulating mass, and STOP at the
    // first class whose cumulative mass reaches qhat. The crossing class IS
    // kept (`>=`, not `>`).
    std::vector<int> out;
    double running = 0.0;
    for (size_t r = 0; r < K; ++r) {
        running += probs_row(0, order[r]);
        out.push_back(static_cast<int>(order[r]));
        if (running >= qhat) {
            break;
        }
    }

    // The class indices are returned sorted ASCENDING: the set is order-free,
    // but a deterministic order makes the result comparable.
    std::sort(out.begin(), out.end());
    return out;
}

Tensor cqr_interval(double lower, double upper, double qhat) {
    Tensor out(1, 2);
    // The calibration error is added to BOTH ends: [lower - qhat, upper + qhat].
    out(0, 0) = lower - qhat;
    out(0, 1) = upper + qhat;
    return out;
}

}  // namespace utils
}  // namespace nn
