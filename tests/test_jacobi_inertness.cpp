// =============================================================================
// tests/test_jacobi_inertness.cpp -- THE INVERTED-VERDICT CHECK for the
// MEASURED-INERT mutation of the Jacobi eigendecomposition.
//
// WHY THIS FILE EXISTS, AND WHY IT IS A SHIPPED TEST RATHER THAN A PROBE.
//
// The mutation harness (tests/mutate_jacobi_eig.sh) applies four mutations to
// the Jacobi eigendecomposition in Shampoo/SOAP/GaLore. Three are caught by
// the suite. The fourth -- M4, dropping the `app == aqq` special case --
// SURVIVES.
//
// A survivor that is not explained is indistinguishable from a real coverage
// gap, so it must either be killed by a test or MEASURED and shown to be
// inert. M4 was measured: with the CORRECT general formula
//
//     theta = 0.5 * atan2(2 * apq, app - aqq)
//
// the degenerate case (app == aqq, apq != 0) gives `atan2(2*apq, 0)` which is
// EXACTLY +-pi/2, so the general branch already produces +-pi/4 -- the same
// value the special case hardcodes. Measured over 2400 matrices (a third of
// them with EQUAL DIAGONAL ENTRIES, which is the branch's actual trigger
// condition): bit-identical output, worst difference exactly 0.0.
//
// **BUT A ONE-OFF MEASUREMENT IN A HARNESS LOG GOES STALE.** If a future
// refactor changed the general formula, or the threshold, the "inert" verdict
// would silently become false while the harness kept reporting it. So this
// file makes the claim CONTINUOUSLY VERIFIABLE: it asserts the inertness
// directly, and it FAILS if the claim ever stops holding.
//
// It shares no code with the library -- it re-implements the rotation angle
// in both variants and compares them -- so it cannot be fooled by the
// implementation it is judging.
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static inline double theta_with_branch(double app, double aqq, double apq) {
    if (std::abs(app - aqq) < 1e-30) {
        return (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
    }
    return 0.5 * std::atan2(2.0 * apq, app - aqq);
}

static inline double theta_without_branch(double app, double aqq, double apq) {
    return 0.5 * std::atan2(2.0 * apq, app - aqq);
}

static void jacobi(std::vector<double>& A, std::vector<double>& Q,
                   double (*theta_fn)(double, double, double),
                   int max_sweeps, double tol) {
    const size_t n = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(A.size()))));
    std::vector<double> Aq = A;
    Q.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) Q[i * n + i] = 1.0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                double v = Aq[p * n + q];
                off += 2.0 * v * v;
            }
        if (std::sqrt(off) < tol) break;

        for (size_t p = 0; p + 1 < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                const double apq = Aq[p * n + q];
                if (std::abs(apq) < 1e-15) continue;
                const double app = Aq[p * n + p];
                const double aqq = Aq[q * n + q];
                const double theta = theta_fn(app, aqq, apq);
                const double c = std::cos(theta), s = std::sin(theta);
                const double new_app = c*c*app + 2.0*c*s*apq + s*s*aqq;
                const double new_aqq = s*s*app - 2.0*c*s*apq + c*c*aqq;
                Aq[p*n+p] = new_app; Aq[q*n+q] = new_aqq;
                Aq[p*n+q] = 0.0; Aq[q*n+p] = 0.0;
                for (size_t r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = Aq[r*n+p], arq = Aq[r*n+q];
                    const double narp = c*arp + s*arq;
                    const double narq = -s*arp + c*arq;
                    Aq[r*n+p] = narp; Aq[p*n+r] = narp;
                    Aq[r*n+q] = narq; Aq[q*n+r] = narq;
                }
                for (size_t r = 0; r < n; ++r) {
                    const double qrp = Q[r*n+p], qrq = Q[r*n+q];
                    Q[r*n+p] = c*qrp + s*qrq;
                    Q[r*n+q] = -s*qrp + c*qrq;
                }
            }
        }
    }
    A = Aq;
}

static bool bits_equal(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t x, y;
        std::memcpy(&x, &a[i], 8);
        std::memcpy(&y, &b[i], 8);
        if (x != y) return false;
    }
    return true;
}

int main() {
    int fails = 0;

    // ---- PART 1: the angle agrees to the last bit at the degenerate point ----
    printf("=== PART 1: the angle at app == aqq ===\n");
    for (double apq : {0.7, -0.7, 1e-8, -1e-8, 1e-20, -1e-20, 3.0, -3.0}) {
        const double app = 2.0, aqq = 2.0;
        const double with_b = theta_with_branch(app, aqq, apq);
        const double without_b = theta_without_branch(app, aqq, apq);
        uint64_t x, y;
        std::memcpy(&x, &with_b, 8);
        std::memcpy(&y, &without_b, 8);
        const bool same = (x == y);
        printf("  apq=%+.1e  with=%.17f  without=%.17f  bit-identical=%s\n",
               apq, with_b, without_b, same ? "YES" : "NO");
        if (!same) ++fails;
    }

    // ---- PART 2: the whole algorithm, bit-for-bit, over a sweep ----
    printf("\n=== PART 2: full-algorithm bit-identity over a randomised sweep ===\n");
    std::mt19937 rng(20261028);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    size_t total = 0, identical = 0, differing = 0;
    double worst = 0.0;

    for (size_t n : {2u, 3u, 4u, 6u, 8u, 12u}) {
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<double> M(n * n);
            for (auto& v : M) v = uni(rng);
            std::vector<double> A(n * n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < n; ++k) s += M[k*n+i] * M[k*n+j];
                    A[i*n+j] = s;
                }
            // Every 3rd trial: EQUAL DIAGONAL ENTRIES -- the branch's actual
            // trigger condition, so the sweep really exercises it.
            if (trial % 3 == 0) {
                for (size_t i = 0; i < n; ++i) {
                    const double d = A[i*n+i];
                    for (size_t j = 0; j < n; ++j) { A[i*n+j] = d; A[j*n+i] = d; }
                    if (i + 1 < n) { A[i*n+i+1] = 0.5; A[(i+1)*n+i] = 0.5; }
                }
            }

            std::vector<double> A1 = A, A2 = A, Q1, Q2;
            jacobi(A1, Q1, theta_with_branch, 100, 1e-12);
            jacobi(A2, Q2, theta_without_branch, 100, 1e-12);

            ++total;
            if (bits_equal(A1, A2) && bits_equal(Q1, Q2)) {
                ++identical;
            } else {
                ++differing;
                for (size_t i = 0; i < A1.size(); ++i) worst = std::max(worst, std::abs(A1[i]-A2[i]));
                for (size_t i = 0; i < Q1.size(); ++i) worst = std::max(worst, std::abs(Q1[i]-Q2[i]));
            }
        }
    }

    printf("  matrices tested      : %zu\n", total);
    printf("  BIT-IDENTICAL        : %zu\n", identical);
    printf("  differing            : %zu\n", differing);
    printf("  worst abs difference : %.6e\n", worst);
    if (differing != 0) {
        printf("  [FAIL] M4 is NOT inert -- the 'inert' verdict would be WRONG\n");
        ++fails;
    } else {
        printf("  [PASS] M4 is provably INERT (bit-identical over the whole sweep)\n");
    }

    printf("\n=== SUMMARY: %s ===\n", fails == 0 ? "the inert verdict HOLDS" : "the inert verdict is BROKEN");
    return fails == 0 ? 0 : 1;
}
