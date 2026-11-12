#include "ordinal_regression.h"

#include <cmath>
#include <limits>
#include <stdexcept>

// ===========================================================================
// The numerically careful primitives.
//
// EVERY one of these exists because the naive form loses information that
// the model needs. The header documents the mathematics; this file documents
// the arithmetic.
// ===========================================================================

namespace {

// sigmoid(x), computed so that neither exp() overflows nor the result
// rounds to exactly 1.0 when it should be strictly less than 1.
inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

// softplus(x) = log(1 + e^x), computed without overflowing for large x.
inline double softplus(double x) {
    if (x > 0.0) {
        return x + std::log1p(std::exp(-x));
    }
    return std::log1p(std::exp(x));
}

// log(sigmoid(x)) = -softplus(-x). Computed in log space so that it stays
// accurate for x << 0, where the probability itself underflows.
inline double log_sigmoid(double x) {
    return -softplus(-x);
}

// log(1 - e^{-gap}) for gap > 0.
//
// THE MINUS SIGN IS LOAD-BEARING: 1 - e^{-gap} is positive and SMALL, and
// computing it as 1.0 - exp(-gap) loses every significant digit when the
// gap is tiny (for gap = 3.7e-44 the naive form is EXACTLY 0.0). The
// expm1 form keeps full relative accuracy down to gaps of ~1e-308.
//
// For gap <= 0 the result is -inf, which is the honest limit: a
// non-positive gap means a zero-width class, whose probability is zero.
inline double log1mexp_neg(double gap) {
    if (gap <= 0.0) return -std::numeric_limits<double>::infinity();
    // log(-expm1(-gap))
    return std::log(-std::expm1(-gap));
}

// q = 1 / (e^{gap} - 1), the derivative of log(1 - e^{-gap}) w.r.t. gap.
//
// expm1 keeps the accuracy for tiny gaps: e^{gap} - 1 == gap for a gap that
// small, so q == 1/gap, and expm1 returns exactly that.
//
// A non-positive gap yields +inf, which is the correct limit of q.
inline double gap_q(double gap) {
    if (gap <= 0.0) return std::numeric_limits<double>::infinity();
    return 1.0 / std::expm1(gap);
}

}  // namespace

// ===========================================================================
// CONSTRUCTION
// ===========================================================================

OrdinalRegression::OrdinalRegression(size_t in_features, size_t num_classes)
    : in_features_(in_features),
      num_classes_(num_classes),
      beta_(1, in_features),
      raw_cuts_(1, num_classes > 0 ? num_classes - 1 : 0),
      grad_beta_(1, in_features),
      grad_raw_cuts_(1, num_classes > 0 ? num_classes - 1 : 0),
      last_input_(0, 0) {
    if (num_classes < 2) {
        throw std::invalid_argument(
            "OrdinalRegression: num_classes must be at least 2 (there must "
            "be at least one cut point)");
    }
    if (in_features == 0) {
        throw std::invalid_argument(
            "OrdinalRegression: in_features must be at least 1");
    }
    beta_.fill(0.0);
    raw_cuts_.fill(0.0);
    grad_beta_.fill(0.0);
    grad_raw_cuts_.fill(0.0);
}

void OrdinalRegression::zero_grad() {
    grad_beta_.fill(0.0);
    grad_raw_cuts_.fill(0.0);
}

// ===========================================================================
// THE FORWARD PASS
// ===========================================================================
//
// For each row:
//   eta = x . beta
//   a_k = theta_k - eta
//   log P_1     = log sigma(a_1)
//   log P_k     = log sigma(a_k) + log sigma(-a_{k-1}) + log(1 - e^{-gap})
//   log P_K     = log sigma(-a_{K-1})
//
// THE GAPS ARE CARRIED SEPARATELY from the cumulative sum of the thetas.
// The cumulative sum can ABSORB a tiny gap (theta_1 = 100 with a gap of
// 3.7e-44 gives theta_2 == theta_1 exactly in float64), and the gap enters
// the probability, so it must never be recovered by differencing.

Tensor OrdinalRegression::log_probs(const Tensor& input) const {
    if (input.cols != in_features_) {
        throw std::invalid_argument(
            "OrdinalRegression: input width " +
            std::to_string(input.cols) + " does not match in_features " +
            std::to_string(in_features_));
    }

    const size_t K = num_classes_;
    const size_t D = in_features_;
    const size_t N = input.rows;

    // ---- the cut points and the gaps, computed SEPARATELY ----
    const size_t n_cuts = K - 1;
    std::vector<double> theta(n_cuts);
    std::vector<double> gap(n_cuts > 0 ? n_cuts - 1 : 0);
    theta[0] = raw_cuts_[0][0];
    for (size_t j = 0; j + 1 < n_cuts; ++j) {
        gap[j] = softplus(raw_cuts_[0][j + 1]);
        theta[j + 1] = theta[j] + gap[j];
    }

    Tensor out(N, K);

    for (size_t i = 0; i < N; ++i) {
        // ---- the latent score ----
        double eta = 0.0;
        for (size_t d = 0; d < D; ++d) eta += input[i][d] * beta_[0][d];

        // ---- the a values ----
        std::vector<double> a(n_cuts);
        for (size_t j = 0; j < n_cuts; ++j) a[j] = theta[j] - eta;

        // ---- class 0: the first cut point is the lower boundary ----
        out[i][0] = log_sigmoid(a[0]);

        // ---- the interior classes ----
        for (size_t k = 1; k + 1 < K; ++k) {
            out[i][k] = log_sigmoid(a[k]) + log_sigmoid(-a[k - 1]) +
                        log1mexp_neg(gap[k - 1]);
        }

        // ---- the last class ----
        out[i][K - 1] = log_sigmoid(-a[n_cuts - 1]);
    }

    return out;
}

Tensor OrdinalRegression::forward(const Tensor& input) {
    Tensor out = log_probs(input);
    last_input_ = input.clone();
    return out;
}

// ===========================================================================
// THE BACKWARD PASS
// ===========================================================================
//
// `grad_output` is dL/d(log P), shape (N, K).
//
// THE GRADIENT IS COMPUTED IN LOG SPACE. In probability space the factor
// sigma(a)(1 - sigma(a)) is EXACTLY ZERO wherever sigma(a) has rounded to
// 1.0, which is exactly the regime where the model is interesting.
//
// THE GRADIENT IS DIFFERENTIATED IN (u, GAPS) COORDINATES, where u is the
// first cut point and the gaps are the K-2 independent increments.
//
// WHY, AND NOT IN THE THETA COORDINATES:
//
//   The q term -- q_j = 1/(e^{gap_j} - 1) -- appears with OPPOSITE signs in
//   the two theta-partials of an interior class, so it CANCELS when the two
//   partials are summed. With a tiny gap q is astronomically large (2.7e43
//   for a 3.7e-44 gap) and the cancellation destroys the answer: summing
//   the theta-partials returns EXACTLY 0 where the true gradient is -1.
//
//   In (u, gaps) coordinates the q term never has to cancel:
//     * u shifts EVERY cut point, which is exactly equivalent to shifting
//       eta the other way, so dL/du = -dL/deta and it carries no q at all.
//     * a gap shift touches EVERY class at or above it, and for an interior
//       class k there are THREE cases:
//           j <= k-2 : both theta_k and theta_{k-1} shift -> (e_k - c_{k-1})
//           j == k-1 : only theta_k shifts, and the gap itself changes
//                      -> (e_k + q_{k-1})
//           j >= k   : nothing
//     * the chain gap_j = softplus(r_{j+1}) multiplies by sigmoid(r_{j+1}),
//       and sigmoid(r_{j+1}) * q_j is formed from O(1) quantities.
//
// The three per-row quantities the engine needs:
//
//   c_k = sigma(a_k)            the cumulative probability
//   e_k = sigmoid(-a_k)         1 - c_k, COMPUTED STABLY. Never 1.0 - c_k:
//                                where c_k has rounded to 1.0 the naive form
//                                is EXACTLY 0.0 while the true value is
//                                4.2e-18, and that factor can carry the
//                                ENTIRE gradient.
//   q_j = 1/(e^{gap_j} - 1)     the gap's own derivative
//
// ---------------------------------------------------------------------------
// THE GRADIENT OF EACH log P_k WITH RESPECT TO eta
// ---------------------------------------------------------------------------
//   class 0     : -(e_0)
//   class k     :  c_{k-1} - e_k
//   class K-1   :  c_{K-2}
//
// (The gap is independent of eta, so it contributes nothing here.)
//
// ---------------------------------------------------------------------------
// THE GRADIENT OF EACH log P_k WITH RESPECT TO THE GAPS
// ---------------------------------------------------------------------------
// For the interior class k, the coefficient of gap j is
//
//   (e_k - c_{k-1})   for j <= k-2        (the "bulk" case)
//   (e_k + q_{k-1})   for j == k-1        (the "special" case)
//   0                for j >= k
//
// and the LAST class contributes -c_{K-2} to EVERY gap, because a gap shift
// moves every cut point above it.
//
// Summing over the classes: gap j receives the bulk coefficients of all
// classes k >= j+2, plus the special coefficient of class j+1, plus the
// last class's -c_{K-2}. That is what the suffix sum below computes.

Tensor OrdinalRegression::backward(const Tensor& grad_output, double /* lr */) {
    if (grad_output.rows != last_input_.rows ||
        grad_output.cols != num_classes_) {
        throw std::invalid_argument(
            "OrdinalRegression::backward: grad_output must be (N, K)");
    }

    const size_t K = num_classes_;
    const size_t D = in_features_;
    const size_t N = last_input_.rows;
    const size_t n_cuts = K - 1;

    // ---- the cut points and the gaps, computed SEPARATELY (see forward) ----
    std::vector<double> theta(n_cuts);
    std::vector<double> gap(n_cuts > 0 ? n_cuts - 1 : 0);
    std::vector<double> q(n_cuts > 0 ? n_cuts - 1 : 0);
    theta[0] = raw_cuts_[0][0];
    for (size_t j = 0; j + 1 < n_cuts; ++j) {
        gap[j] = softplus(raw_cuts_[0][j + 1]);
        q[j] = gap_q(gap[j]);
        theta[j + 1] = theta[j] + gap[j];
    }

    // The accumulated dL/dgap, one per gap.
    std::vector<double> dL_dgap(n_cuts > 0 ? n_cuts - 1 : 0, 0.0);
    // The per-class coefficients, accumulated over the rows.
    std::vector<double> bulk(K, 0.0);      // the "j <= k-2" coefficient
    std::vector<double> special(K, 0.0);   // the "j == k-1" coefficient

    // dL/deta, accumulated per row, and dL/dbeta.
    std::vector<double> dL_deta(N, 0.0);

    for (size_t i = 0; i < N; ++i) {
        double eta = 0.0;
        for (size_t d = 0; d < D; ++d) eta += last_input_[i][d] * beta_[0][d];

        std::vector<double> a(n_cuts);
        std::vector<double> c(n_cuts), e(n_cuts);
        for (size_t j = 0; j < n_cuts; ++j) {
            a[j] = theta[j] - eta;
            c[j] = sigmoid(a[j]);
            // e = 1 - c, computed STABLY. Never 1.0 - c.
            e[j] = sigmoid(-a[j]);
        }

        // ---- dL/deta ----
        double acc_eta = 0.0;
        for (size_t k = 0; k < K; ++k) {
            const double g = grad_output[i][k];
            if (g == 0.0) continue;
            if (k == 0) {
                acc_eta += g * (-e[0]);
            } else if (k == K - 1) {
                acc_eta += g * (c[K - 2]);
            } else {
                acc_eta += g * (c[k - 1] - e[k]);
            }
        }
        dL_deta[i] = acc_eta;

        // ---- the gap coefficients, per class ----
        for (size_t k = 0; k < K; ++k) {
            const double g = grad_output[i][k];
            if (g == 0.0) continue;
            if (k == 0) {
                // Class 0 has no gap term; its coefficient is the shift of
                // u, which is handled by -dL/deta.
                bulk[k] += g * e[0];
            } else if (k == K - 1) {
                // The last class contributes -c_{K-2} to EVERY gap.
                bulk[k] += g * (-c[K - 2]);
            } else {
                bulk[k] += g * (e[k] - c[k - 1]);
                special[k] += g * (e[k] + q[k - 1]);
            }
        }
    }

    // ---- dL/dgap: the suffix sum over the bulk coefficients, plus the
    //      special coefficient of class j+1 ----
    //
    // gap j receives bulk[k] for every k >= j+2, and special[j+1].
    double suffix = 0.0;
    for (size_t j = n_cuts - 1; j-- > 0;) {
        // j runs n_cuts-2 down to 0
        suffix += bulk[j + 2];
        dL_dgap[j] = suffix + special[j + 1];
    }
    // (When n_cuts == 1 there is no gap at all and the loop does not run.)

    // ---- dL/dbeta ----
    for (size_t i = 0; i < N; ++i) {
        if (dL_deta[i] == 0.0) continue;
        for (size_t d = 0; d < D; ++d) {
            grad_beta_[0][d] += dL_deta[i] * last_input_[i][d];
        }
    }

    // ---- dL/draw ----
    //
    // r_1 is u: shifting it shifts every cut point, which is equivalent to
    // shifting eta the other way, so dL/du = -dL/deta.
    double dL_du = 0.0;
    for (size_t i = 0; i < N; ++i) dL_du -= dL_deta[i];
    grad_raw_cuts_[0][0] += dL_du;

    // r_{j+2} parameterises gap_j through softplus, so
    // dL/dr_{j+2} = dL/dgap_j * sigmoid(r_{j+2}).
    for (size_t j = 0; j + 1 < n_cuts; ++j) {
        grad_raw_cuts_[0][j + 1] += dL_dgap[j] * sigmoid(raw_cuts_[0][j + 1]);
    }

    // ---- dL/dinput ----
    // The input enters ONLY through eta = x . beta, so
    // dL/dx_{i,d} = (dL/deta_i) * beta_d.
    Tensor grad_input(N, D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            grad_input[i][d] = dL_deta[i] * beta_[0][d];
        }
    }
    return grad_input;
}

// ===========================================================================
// THE UPDATE
// ===========================================================================

void OrdinalRegression::update_weights(double learning_rate) {
    for (size_t d = 0; d < in_features_; ++d) {
        beta_[0][d] -= learning_rate * grad_beta_[0][d];
    }
    for (size_t k = 0; k + 1 < num_classes_; ++k) {
        raw_cuts_[0][k] -= learning_rate * grad_raw_cuts_[0][k];
    }
}

// ===========================================================================
// THE READOUTS
// ===========================================================================

Tensor OrdinalRegression::cut_points() const {
    const size_t n_cuts = num_classes_ - 1;
    Tensor out(1, n_cuts);
    if (n_cuts == 0) return out;
    out[0][0] = raw_cuts_[0][0];
    for (size_t j = 0; j + 1 < n_cuts; ++j) {
        out[0][j + 1] = out[0][j] + softplus(raw_cuts_[0][j + 1]);
    }
    return out;
}

Tensor OrdinalRegression::cut_gaps() const {
    const size_t n_gaps = num_classes_ - 1 > 0 ? num_classes_ - 2 : 0;
    Tensor out(1, n_gaps);
    for (size_t j = 0; j < n_gaps; ++j) {
        // From the RAW parameter, never by differencing the cut points: the
        // cumulative sum can absorb a tiny gap, and differencing would then
        // report a gap of 0.
        out[0][j] = softplus(raw_cuts_[0][j + 1]);
    }
    return out;
}

Tensor OrdinalRegression::expected_class(const Tensor& input) const {
    Tensor logp = log_probs(input);
    Tensor out(input.rows, 1);
    for (size_t i = 0; i < input.rows; ++i) {
        double expected = 0.0;
        for (size_t k = 0; k < num_classes_; ++k) {
            // exp() of the log probability. A probability that underflows to
            // zero contributes nothing to the expectation, which is correct:
            // its weight is far below the resolution of the sum.
            expected += static_cast<double>(k) * std::exp(logp[i][k]);
        }
        out[i][0] = expected;
    }
    return out;
}

// ===========================================================================
// THE PARAMETER ACCESSORS
// ===========================================================================

double OrdinalRegression::beta_at(size_t d) const {
    if (d >= in_features_) {
        throw std::out_of_range("OrdinalRegression::beta_at: index out of range");
    }
    return beta_[0][d];
}

void OrdinalRegression::set_beta(size_t d, double value) {
    if (d >= in_features_) {
        throw std::out_of_range("OrdinalRegression::set_beta: index out of range");
    }
    beta_[0][d] = value;
}

double OrdinalRegression::raw_cut_at(size_t k) const {
    if (k + 1 >= num_classes_) {
        throw std::out_of_range("OrdinalRegression::raw_cut_at: index out of range");
    }
    return raw_cuts_[0][k];
}

void OrdinalRegression::set_raw_cut(size_t k, double value) {
    if (k + 1 >= num_classes_) {
        throw std::out_of_range("OrdinalRegression::set_raw_cut: index out of range");
    }
    raw_cuts_[0][k] = value;
}
