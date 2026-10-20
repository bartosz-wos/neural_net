#include "sinkhorn.h"

#include <cmath>
#include <stdexcept>

namespace nn {
namespace utils {
namespace {

// The `-1` in H(P) = -sum P (log P - 1) is the constant that makes H the
// GENERALISED entropy, and it is what lets the debiased divergence vanish
// exactly on the diagonal. Sum P == 1, so the term contributes exactly -1.
constexpr double kGeneralizedEntropyOffset = 1.0;

// ---------------------------------------------------------------------------
// Guards. EVERY message here is distinct. A substring that appears in two
// different messages discriminates nothing between them, and this repo has now
// hit that trap three separate times (each time the assertion searched for a
// word the *other* guard's message also contained, so the mutant survived).
// ---------------------------------------------------------------------------
void validate_cost(const Tensor& C, const Tensor& a, const Tensor& b) {
    if (C.rows == 0 || C.cols == 0)
        throw std::invalid_argument("sinkhorn: C is empty");
    if (a.rows == 0 || a.cols == 0)
        throw std::invalid_argument("sinkhorn: a is empty");
    if (b.rows == 0 || b.cols == 0)
        throw std::invalid_argument("sinkhorn: b is empty");
    if (C.rows != a.rows || C.cols != b.rows)
        throw std::invalid_argument(
            "sinkhorn: C shape does not match the marginals (C.rows vs a.rows, "
            "C.cols vs b.rows)");
    if (a.cols != 1 || b.cols != 1)
        throw std::invalid_argument(
            "sinkhorn: each marginal must be a single column, since Tensor is "
            "2-D only");
}

// Weights must be strictly positive: log(0) is -inf and a single zero weight
// poisons every downstream number. This is the repo's recurring 1/0 trap
// (muP's `inf` from a dropped guard, VQ-VAE's `decay=1`, PTQ's all-zero
// tensor, ptq_relative_l2, and this file's own all-zero-plan case), and the
// consistent response across all of them is to REJECT rather than to return a
// plausible-looking infinity.
void validate_weights(const Tensor& w, const char* name) {
    for (size_t i = 0; i < w.rows; ++i) {
        double v = w(i, 0);
        if (!std::isfinite(v))
            throw std::invalid_argument(
                std::string("sinkhorn: ") + name + " contains a non-finite weight");
        if (v <= 0.0)
            throw std::invalid_argument(
                std::string("sinkhorn: ") + name +
                " contains a non-positive weight; every weight must be "
                "strictly positive");
    }
}

void validate_eps(double eps) {
    if (!std::isfinite(eps) || eps <= 0.0)
        throw std::invalid_argument("sinkhorn: eps must be finite and > 0");
}

// Renormalise a marginal to sum 1, so a caller who passes (0.5, 0.5) and a
// caller who passes (1, 1) get the same problem. The deviation is reported by
// the caller's residual check rather than silently accepted.
double marginal_sum(const Tensor& w) {
    double s = 0.0;
    for (size_t i = 0; i < w.rows; ++i) s += w(i, 0);
    return s;
}

// logsumexp over a contiguous row of M, max-subtracted. The max-subtraction
// is not an optimisation: at eps = 1e-3 the un-subtracted exp underflows to
// exactly 0 for all but the largest element and the whole row reads as -inf.
double logsumexp(const double* row, size_t n) {
    double mx = row[0];
    for (size_t j = 1; j < n; ++j)
        if (row[j] > mx) mx = row[j];
    if (!std::isfinite(mx)) return mx;
    double acc = 0.0;
    for (size_t j = 0; j < n; ++j) acc += std::exp(row[j] - mx);
    return mx + std::log(acc);
}

Tensor uniform(size_t n) {
    Tensor t = Tensor::zeros(n, 1);
    for (size_t i = 0; i < n; ++i) t(i, 0) = 1.0 / static_cast<double>(n);
    return t;
}

}  // namespace

// ---------------------------------------------------------------------------
double plan_marginal_error(const Tensor& P, const Tensor& a, const Tensor& b) {
    validate_cost(P, a, b);
    double worst = 0.0;
    for (size_t i = 0; i < P.rows; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < P.cols; ++j) s += P(i, j);
        worst = std::max(worst, std::fabs(s - a(i, 0)));
    }
    for (size_t j = 0; j < P.cols; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < P.rows; ++i) s += P(i, j);
        worst = std::max(worst, std::fabs(s - b(j, 0)));
    }
    return worst;
}

// ---------------------------------------------------------------------------
SinkhornReport sinkhorn_log(const Tensor& C, const Tensor& a, const Tensor& b,
                            double eps, int max_iter, double tol,
                            Tensor* plan_out, Tensor* f_out, Tensor* g_out) {
    validate_cost(C, a, b);
    validate_weights(a, "a");
    validate_weights(b, "b");
    validate_eps(eps);
    if (max_iter < 1)
        throw std::invalid_argument("sinkhorn: max_iter must be >= 1");

    const size_t n = C.rows;
    const size_t m = C.cols;

    const double sa = marginal_sum(a);
    const double sb = marginal_sum(b);
    if (sa <= 0.0 || sb <= 0.0)
        throw std::invalid_argument("sinkhorn: a and b must each have positive mass");

    // Normalised copies of the logs. Dividing by the sum rather than requiring
    // it to be 1 is what makes the solver usable with hand-written weights; a
    // solver that demanded sum == 1 exactly would reject legitimate input for
    // no mathematical reason.
    std::vector<double> la(n), lb(m);
    for (size_t i = 0; i < n; ++i) la[i] = std::log(a(i, 0) / sa);
    for (size_t j = 0; j < m; ++j) lb[j] = std::log(b(j, 0) / sb);

    std::vector<double> f(n, 0.0), g(m, 0.0);
    // `row` holds a ROW of M (m entries) and `colm` a COLUMN (n entries). Both
    // buffers are sized from the axis they are indexed over -- the mistake of
    // sizing one from the other axis is a heap overflow for every rectangular
    // problem, and the suite deliberately covers BOTH (7,5) and (3,7) so that
    // neither direction can be the one that breaks.
    std::vector<double> row(m);

    SinkhornReport rep;
    rep.marginal_residual = std::numeric_limits<double>::infinity();

    for (int it = 1; it <= max_iter; ++it) {
        // ROW UPDATE. The row marginal of exp(M) is exp(logsumexp_j M_ij), so
        // the fixed point f_i += eps * (log a_i - lse_i) drives that marginal to
        // a_i. This ACCUMULATES.
        //
        // A first draft of the oracle wrote the NEGATED step, f = -eps*(la-lse),
        // which gives lse^new = 2*lse - la: the sweep DOUBLES the error rather
        // than cancelling it and reaches NaN in ~40 iterations. Measured, not
        // reasoned -- that divergence is what identified the sign.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j)
                row[j] = (-C(i, j) + f[i] + g[j]) / eps;
            double lse = logsumexp(row.data(), m);
            f[i] += eps * (la[i] - lse);
        }
        // COLUMN UPDATE, same step on the other axis. The max-subtraction is
        // done PER COLUMN here; reusing the row's max would underflow on the
        // column axis, which is a different set of exponentials.
        //
        // The buffer holds a COLUMN, so it is sized n and indexed over n.
        // A first draft sized it m and indexed it over n, which is a
        // heap-buffer-overflow for every rectangular (n > m) problem -- caught
        // by the suite's shapes (7,5) and (3,7), and by ASan. `Tensor::operator()`
        // has no bounds check either, so the same class of mistake in a tensor
        // loop is silent at -O2.
        std::vector<double> colm(n);
        for (size_t j = 0; j < m; ++j) {
            for (size_t i = 0; i < n; ++i)
                colm[i] = (-C(i, j) + f[i] + g[j]) / eps;
            double lse = logsumexp(colm.data(), n);
            g[j] += eps * (lb[j] - lse);
        }

        // Residual, computed on the PLAN rather than on the duals: the duals can
        // agree while the plan is still off, and it is the plan a caller uses.
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double mx = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < m; ++j) {
                double v = (-C(i, j) + f[i] + g[j]) / eps;
                if (v > mx) mx = v;
            }
            double s = 0.0;
            for (size_t j = 0; j < m; ++j)
                s += std::exp((-C(i, j) + f[i] + g[j]) / eps - mx);
            worst = std::max(worst, std::fabs(std::exp(mx) * s - a(i, 0) / sa));
        }
        for (size_t j = 0; j < m; ++j) {
            double mx = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < n; ++i) {
                double v = (-C(i, j) + f[i] + g[j]) / eps;
                if (v > mx) mx = v;
            }
            double s = 0.0;
            for (size_t i = 0; i < n; ++i)
                s += std::exp((-C(i, j) + f[i] + g[j]) / eps - mx);
            worst = std::max(worst, std::fabs(std::exp(mx) * s - b(j, 0) / sb));
        }

        rep.iterations = it;
        rep.marginal_residual = worst;
        if (!(worst > tol)) {  // also catches NaN, which compares false here
            rep.converged = true;
            break;
        }
        if (!std::isfinite(worst)) {
            // A non-finite residual cannot be reduced by more iterations; it
            // means the duals have left the representable range. Reporting
            // "converged" here would be a lie, so this is returned unconverged
            // and the caller sees a non-finite residual.
            break;
        }
    }

    if (plan_out) {
        // The plan is materialised as
        //     P_ij = a_i * exp(M_ij - lse_i),      lse_i = logsumexp_j M_ij
        // which is algebraically exp(M_ij) but computed so that the row sums
        // are a_i BY CONSTRUCTION rather than by cancellation of a large
        // exponential.
        //
        // A first draft wrote the row-max form, P_ij = exp(M_ij - mx_i), and
        // FORGOT to multiply back by exp(mx_i). The solver's internal residual
        // was computed correctly and reported 9.294e-15 while the plan it
        // handed back had row sums off by orders of magnitude -- so the suite's
        // "converged" and "residual" checks both passed and only the marginals
        // of the RETURNED plan caught it. That is the value of asserting on the
        // output rather than on the solver's own report of itself: a solver that
        // grades its own homework will happily grade it correctly.
        Tensor P = Tensor::zeros(n, m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j)
                row[j] = (-C(i, j) + f[i] + g[j]) / eps;
            double lse = logsumexp(row.data(), m);
            double ai = a(i, 0) / sa;
            for (size_t j = 0; j < m; ++j)
                P(i, j) = ai * std::exp(row[j] - lse);
        }
        *plan_out = P;
    }
    if (f_out) {
        Tensor fo = Tensor::zeros(n, 1);
        for (size_t i = 0; i < n; ++i) fo(i, 0) = f[i];
        *f_out = fo;
    }
    if (g_out) {
        Tensor go = Tensor::zeros(m, 1);
        for (size_t j = 0; j < m; ++j) go(j, 0) = g[j];
        *g_out = go;
    }
    return rep;
}

// ---------------------------------------------------------------------------
double ot_eps(const Tensor& P, const Tensor& C, double eps) {
    if (P.rows != C.rows || P.cols != C.cols)
        throw std::invalid_argument("sinkhorn: plan and cost shapes do not match");
    validate_eps(eps);

    double cost = 0.0;
    double entropy = 0.0;
    for (size_t i = 0; i < P.rows; ++i) {
        for (size_t j = 0; j < P.cols; ++j) {
            double p = P(i, j);
            cost += p * C(i, j);
            // A plan entry that is EXACTLY zero is legitimate and frequent: at
            // eps = 1e-3 the optimal plan for [[0,1],[1,0]] is [0.5 0; 0 0.5].
            // Such an entry contributes nothing to the entropy, and skipping it
            // is what keeps this finite instead of 0 * (-inf) = NaN.
            if (p > 0.0) entropy -= p * (std::log(p) - kGeneralizedEntropyOffset);
        }
    }
    return cost - eps * entropy;
}

// ---------------------------------------------------------------------------
SinkhornCosts sinkhorn_sqdist(const Tensor& X, const Tensor& Y) {
    if (X.rows == 0 || X.cols == 0)
        throw std::invalid_argument("sinkhorn: X is empty");
    if (Y.rows == 0 || Y.cols == 0)
        throw std::invalid_argument("sinkhorn: Y is empty");
    if (X.cols != Y.cols)
        throw std::invalid_argument(
            "sinkhorn: X and Y column counts do not match");

    const size_t n = X.rows, m = Y.rows, d = X.cols;
    SinkhornCosts out;
    out.cxy = Tensor::zeros(n, m);
    out.cxx = Tensor::zeros(n, n);
    out.cyy = Tensor::zeros(m, m);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) {
                double diff = X(i, k) - Y(j, k);
                s += diff * diff;
            }
            out.cxy(i, j) = s;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) {
                double diff = X(i, k) - X(j, k);
                s += diff * diff;
            }
            out.cxx(i, j) = s;
        }
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < d; ++k) {
                double diff = Y(i, k) - Y(j, k);
                s += diff * diff;
            }
            out.cyy(i, j) = s;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
double sinkhorn_divergence(const Tensor& X, const Tensor& Y, const Tensor& a,
                           const Tensor& b, const Tensor& cxy, const Tensor& cxx,
                           const Tensor& cyy, double eps, int max_iter, double tol,
                           Tensor* plan_xy_out, Tensor* plan_xx_out,
                           Tensor* plan_yy_out, SinkhornReport* report_out) {
    // X and Y are validated through the cost shapes rather than used directly:
    // this entry point is transport-plan-first, and the caller may have built
    // the costs by other means. Validating that the point sets agree with the
    // costs' shapes catches the mismatch a caller can actually make.
    if (X.rows != cxy.rows || Y.rows != cxy.cols)
        throw std::invalid_argument(
            "sinkhorn: point-set rows do not match the cost matrix shape");
    if (X.cols != Y.cols)
        throw std::invalid_argument(
            "sinkhorn: X and Y column counts do not match");

    SinkhornReport rxy = sinkhorn_log(cxy, a, b, eps, max_iter, tol, plan_xy_out);
    Tensor Pxx, Pyy;
    SinkhornReport rxx = sinkhorn_log(cxx, a, a, eps, max_iter, tol, &Pxx);
    SinkhornReport ryy = sinkhorn_log(cyy, b, b, eps, max_iter, tol, &Pyy);

    // The cross plan is needed for its VALUE even when the caller did not ask
    // for it, so it is materialised here rather than being dropped from the sum.
    // NOTE this is a plain branch, not a comma expression: the first draft
    // packed the solve into the false branch of `?:` so the assignment would
    // only run if the pointer were null, which aliases `Pxy_local` into its own
    // argument -- an argument-evaluation-order hazard, not a style complaint.
    Tensor Pxy_local;
    const Tensor& Pxy = plan_xy_out ? *plan_xy_out : Pxy_local;
    if (!plan_xy_out)
        sinkhorn_log(cxy, a, b, eps, max_iter, tol, &Pxy_local);

    double v_xy = ot_eps(Pxy, cxy, eps);
    double v_xx = ot_eps(Pxx, cxx, eps);
    double v_yy = ot_eps(Pyy, cyy, eps);

    // THE DEBIASING. These two -1/2 terms are the entire content of Feydy et
    // al.'s divergence and their ABSENCE is what the tiny-displacement test
    // catches: the plain value is -0.147292507453 where the debiased one is
    // 3.000000000003e-06, a factor of 49000.
    double S = v_xy - 0.5 * v_xx - 0.5 * v_yy;

    if (plan_xx_out) *plan_xx_out = Pxx;
    if (plan_yy_out) *plan_yy_out = Pyy;
    if (report_out) {
        // Report the WORST of the three solves: a caller asking "did this
        // converge" must not be told yes by the cross solve while the two self
        // solves are still far off.
        SinkhornReport worst = rxy;
        if (rxx.marginal_residual > worst.marginal_residual) worst = rxx;
        if (ryy.marginal_residual > worst.marginal_residual) worst = ryy;
        worst.converged = rxy.converged && rxx.converged && ryy.converged;
        worst.iterations = std::max(rxy.iterations,
                                    std::max(rxx.iterations, ryy.iterations));
        *report_out = worst;
    }
    return S;
}

// ---------------------------------------------------------------------------
Tensor sinkhorn_point_gradient(const Tensor& X, const Tensor& Y, const Tensor& a,
                               const Tensor& b, const Tensor& cxy, const Tensor& cxx,
                               const Tensor& cyy, double eps, int max_iter,
                               double tol, Tensor* grad_y_out) {
    if (X.rows != cxy.rows || Y.rows != cxy.cols)
        throw std::invalid_argument(
            "sinkhorn: point-set rows do not match the cost matrix shape");
    if (cxx.rows != X.rows || cxx.cols != X.rows)
        throw std::invalid_argument(
            "sinkhorn: cxx shape does not match the X point set");
    if (cyy.rows != Y.rows || cyy.cols != Y.rows)
        throw std::invalid_argument(
            "sinkhorn: cyy shape does not match the Y point set");
    if (X.cols != Y.cols)
        throw std::invalid_argument(
            "sinkhorn: X and Y column counts do not match");

    const size_t n = X.rows, m = Y.rows, d = X.cols;

    Tensor Pxy, Pxx;
    sinkhorn_log(cxy, a, b, eps, max_iter, tol, &Pxy);
    sinkhorn_log(cxx, a, a, eps, max_iter, tol, &Pxx);

    double sa = marginal_sum(a);
    double sb = marginal_sum(b);

    Tensor gx = Tensor::zeros(n, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d; ++k) {
            // CROSS TERM, coefficient 2. X[i,k] is the first endpoint of
            // C_xy[i,j] exactly once, so the ordinary chain rule gives 2.
            double s = 0.0;
            for (size_t j = 0; j < m; ++j)
                s += Pxy(i, j) * 2.0 * (X(i, k) - Y(j, k));

            // SELF TERM, coefficient -2. THIS IS THE CONSTANT THE ORACLE'S OWN
            // FIRST DRAFT GOT WRONG BY A FACTOR OF TWO.
            //
            // S' = -1/2 sum_{p,q} P_xx[p,q] C_xx[p,q], and X[i,k] appears in
            // that sum TWICE: as the first endpoint of C_xx[i,q], and as the
            // SECOND endpoint of C_xx[p,i] with the opposite sign. With
            // G = -1/2 P_xx symmetric the two halves are equal:
            //     d/dX[i,k] = 2 sum_q G[i,q](X[i,k]-X[q,k])
            //                 + 2 sum_p G[i,p](X[i,k]-X[p,k])
            //               = 4 sum_{q != i} G[i,q](X[i,k]-X[q,k])
            // and with G carrying the -1/2 that is a net coefficient of -2.
            // Using -1 counts ONE of the two endpoints; measured, it gives
            // 3.6e-02 relative error where -2 gives 7.3e-10. A clean factor of
            // two on one term with everything else at 1e-10 is the signature
            // of a missing symmetric half, not a tolerance problem.
            double self = 0.0;
            for (size_t q = 0; q < n; ++q) {
                // C_xx[i,i] == 0, so this term's distance factor is zero and it
                // would contribute nothing even without the guard. It is kept
                // because it states the intent and saves O(1) work.
                if (q == i) continue;
                self += Pxx(i, q) * (X(i, k) - X(q, k));
            }
            s += (-2.0) * self;

            // The weights are normalised inside the solver, so the derivative
            // carries the SAME 1/s normalisation: a enters as a/sum(a).
            gx(i, k) = s / sa;
        }
    }

    if (grad_y_out) {
        // dS/dY is the same expression with X <-> Y, a <-> b, P_xy transposed,
        // and P_yy in place of P_xx. The -2 and the 2 are UNCHANGED, which is
        // the real check that the symmetric derivation was applied to both
        // sides rather than to one of them.
        Tensor Pyy;
        sinkhorn_log(cyy, b, b, eps, max_iter, tol, &Pyy);
        Tensor gy = Tensor::zeros(m, d);
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < d; ++k) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i)
                    s += Pxy(i, j) * 2.0 * (X(i, k) - Y(j, k));
                double self = 0.0;
                for (size_t q = 0; q < m; ++q) {
                    if (q == j) continue;
                    self += Pyy(j, q) * (Y(j, k) - Y(q, k));
                }
                s += (-2.0) * self;
                gy(j, k) = -s / sb;
            }
        }
        *grad_y_out = gy;
    }

    return gx;
}

// ---------------------------------------------------------------------------
Tensor barycentric_projection(const Tensor& P, const Tensor& X, const Tensor& Y) {
    if (P.rows == 0 || P.cols == 0)
        throw std::invalid_argument("sinkhorn: plan is empty");
    if (X.rows == 0 || X.cols == 0)
        throw std::invalid_argument("sinkhorn: X is empty");
    if (Y.rows == 0 || Y.cols == 0)
        throw std::invalid_argument("sinkhorn: Y is empty");
    if (P.rows != X.rows)
        throw std::invalid_argument(
            "sinkhorn: plan rows do not match the source point count");
    if (P.cols != Y.rows)
        throw std::invalid_argument(
            "sinkhorn: plan columns do not match the target point count");
    if (X.cols != Y.cols)
        throw std::invalid_argument(
            "sinkhorn: X and Y column counts do not match");

    const size_t n = P.rows, m = P.cols, d = X.cols;
    Tensor out = Tensor::zeros(n, d);
    for (size_t i = 0; i < n; ++i) {
        // The 1/a_i weighting is MANDATORY, not cosmetic. Without it the
        // output is sum_j P_ij Y[j], which is not the barycentric image of a
        // measure with masses a_i -- for a plan whose rows sum to 1/n that is a
        // factor of n too small.
        double mass = 0.0;
        for (size_t j = 0; j < m; ++j) mass += P(i, j);
        if (!(std::fabs(mass) > 0.0))
            throw std::invalid_argument(
                "sinkhorn: a plan row with zero mass has no barycentric image");
        for (size_t k = 0; k < d; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < m; ++j) s += P(i, j) * Y(j, k);
            out(i, k) = s / mass;
        }
    }
    return out;
}

// ===========================================================================
SinkhornLoss::SinkhornLoss(double eps, int max_iter, double tol)
    : eps_(eps), max_iter_(max_iter), tol_(tol) {
    validate_eps(eps_);
    if (max_iter_ < 1)
        throw std::invalid_argument("sinkhorn: max_iter must be >= 1");
    if (!(tol_ > 0.0) || !std::isfinite(tol_))
        throw std::invalid_argument("sinkhorn: tol must be finite and > 0");
}

void SinkhornLoss::solve(const Tensor& out, const Tensor& ref) {
    if (out.rows == 0 || out.cols == 0)
        throw std::invalid_argument("sinkhorn: output is empty");
    if (ref.rows == 0 || ref.cols == 0)
        throw std::invalid_argument("sinkhorn: reference is empty");
    if (out.cols != ref.cols)
        throw std::invalid_argument(
            "sinkhorn: output and reference column counts do not match");

    costs_ = sinkhorn_sqdist(out, ref);
    a_ = uniform(out.rows);
    b_ = uniform(ref.rows);

    // ONE set of solves produces the cross plan (which the caller can read
    // back), the two self plans (needed only for their VALUES), the report and
    // the value. Solving three times per forward would have been correct and
    // wasteful -- and the duals come free with the solve.
    last_value_ = sinkhorn_divergence(out, ref, a_, b_, costs_.cxy, costs_.cxx,
                                      costs_.cyy, eps_, max_iter_, tol_, &plan_,
                                      nullptr, nullptr, &report_);
    // The duals are a SECOND cross solve's only other output, and a caller
    // doing many consecutive steps can warm-start from them. They are cheap
    // relative to the self-plans, which dominate the cost.
    sinkhorn_log(costs_.cxy, a_, b_, eps_, max_iter_, tol_, nullptr, &duals_,
                 nullptr);
}

double SinkhornLoss::forward(const Tensor& out, const Tensor& ref) {
    solve(out, ref);
    return last_value_;
}

Tensor SinkhornLoss::backward(const Tensor& out, const Tensor& ref) {
    // A caller who has not called forward() must not get a gradient from an
    // unsolved state, so the plans are solved here if necessary.
    if (costs_.cxy.rows != out.rows || costs_.cxy.cols != ref.rows)
        solve(out, ref);
    return sinkhorn_point_gradient(out, ref, a_, b_, costs_.cxy, costs_.cxx,
                                   costs_.cyy, eps_, max_iter_, tol_);
}

}  // namespace utils
}  // namespace nn
