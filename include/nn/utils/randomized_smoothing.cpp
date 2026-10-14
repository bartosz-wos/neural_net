#include "randomized_smoothing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "losses.h"

namespace {

void require_sigma(double sigma, const char* who) {
    if (!(sigma >= 0.0)) {
        throw std::invalid_argument(std::string(who) +
                                    ": sigma must be >= 0 (got " +
                                    std::to_string(sigma) + ")");
    }
}

void require_alpha(double alpha, const char* who) {
    if (!(alpha > 0.0) || !(alpha < 1.0)) {
        throw std::invalid_argument(std::string(who) +
                                    ": alpha must be in (0, 1) (got " +
                                    std::to_string(alpha) + ")");
    }
}

// ---------------------------------------------------------------------------
// Binomial tail probabilities, written as explicit sums.
//
// The recurrences below are the standard cumulative-binomial identities
// (K&R 6.2). They are used rather than lgamma so that the k = n and k = 0
// endpoints -- which are exactly the cases this file's callers hit most often
// (an unanimous or a fully-discordant sample) -- return 0.0 and 1.0 EXACTLY
// rather than picking up a cancellation error from a log-gamma difference.
// ---------------------------------------------------------------------------
double binomial_pmf(int k, int n, double p) {
    if (k < 0 || k > n) return 0.0;
    if (p <= 0.0) return k == 0 ? 1.0 : 0.0;
    if (p >= 1.0) return k == n ? 1.0 : 0.0;
    return std::exp(std::lgamma(n + 1.0) - std::lgamma(k + 1.0) -
                    std::lgamma(n - k + 1.0) + k * std::log(p) +
                    (n - k) * std::log1p(-p));
}

double binomial_cdf(int k, int n, double p) {
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    double sum = 0.0;
    for (int i = 0; i <= k; ++i) sum += binomial_pmf(i, n, p);
    return sum;
}

double binomial_sf(int k, int n, double p) {
    // P(X >= k) = 1 - P(X <= k-1), but the latter loses all precision for large
    // k, so sum the upper tail directly instead.
    if (k <= 0) return 1.0;
    if (k > n) return 0.0;
    double sum = 0.0;
    for (int i = k; i <= n; ++i) sum += binomial_pmf(i, n, p);
    return sum;
}

// ---------------------------------------------------------------------------
// Continued-fraction expansion for the incomplete beta function (Lentz's
// method), used by `regularized_incomplete_beta`.
// ---------------------------------------------------------------------------
double beta_continued_fraction(double a, double b, double x) {
    constexpr int kMaxIterations = 300;
    constexpr double kEpsilon = 3e-16;
    constexpr double kTiny = 1e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIterations; ++m) {
        const int m2 = 2 * m;

        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;

        if (std::fabs(del - 1.0) < kEpsilon) break;
    }
    return h;
}

// The regularised incomplete beta function I_x(a, b).
double regularized_incomplete_beta(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (a <= 0.0 || b <= 0.0) return std::numeric_limits<double>::quiet_NaN();

    const double log_front = std::lgamma(a + b) - std::lgamma(a) -
                             std::lgamma(b) + a * std::log(x) +
                             b * std::log1p(-x);
    const double front = std::exp(log_front);

    // I_x(a,b) is small for x < (a+1)/(a+b+2); evaluating the mirrored form
    // there avoids losing every significant digit to cancellation.
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * beta_continued_fraction(a, b, x) / a;
    }
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b;
}

// Invert I_x(a, b) = p for x by bisection. I_x is strictly increasing in x on
// (0, 1) for a, b > 0, which is what makes bisection valid here.
double regularized_incomplete_beta_inverse(double a, double b, double p) {
    if (!(p > 0.0) || !(p < 1.0)) {
        throw std::invalid_argument(
            "clopper_pearson_lower: the target probability must be in (0, 1)");
    }
    double lo = 0.0;
    double hi = 1.0;
    // 200 halvings of [0, 1] is far past double resolution; the loop normally
    // exits on the width test after ~60 iterations.
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (regularized_incomplete_beta(a, b, mid) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo <= 1e-17 * std::max(1.0, mid)) break;
    }
    return 0.5 * (lo + hi);
}

}  // namespace

// ===========================================================================
// Phi and Phi^-1
// ===========================================================================
double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_quantile(double p) {
    if (!(p > 0.0) || !(p < 1.0)) {
        throw std::invalid_argument(
            "normal_quantile: p must be in (0, 1) (got " + std::to_string(p) +
            "). Phi^-1(1) is +infinity, so a caller reaching p == 1 has an "
            "over-confident bound rather than a valid input.");
    }
    // Phi^-1(0.5) IS exactly 0 by the function's own symmetry, and returning the
    // bisection's stopping residual (-7.4e-17) instead makes the answer depend
    // on the bracket rather than on the mathematics. This matters here because
    // `certified_radius(sigma, p_lower)` must return a radius of exactly 0 at
    // p_lower == 1/2 for D2's threshold to be a clean comparison.
    if (p == 0.5) return 0.0;
    // D9/bisection: Phi is strictly increasing, so a bracket search is exact.
    double lo = -40.0;
    double hi = 40.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (normal_cdf(mid) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo <= 1e-17 * std::max(1.0, std::fabs(mid))) break;
    }
    return 0.5 * (lo + hi);
}

// ===========================================================================
// Clopper-Pearson lower bound  (D6, D7)
// ===========================================================================
double clopper_pearson_lower(int k, int n, double alpha) {
    if (n <= 0) {
        throw std::invalid_argument(
            "clopper_pearson_lower: n must be >= 1 (got " + std::to_string(n) + ")");
    }
    if (k < 0 || k > n) {
        throw std::invalid_argument(
            "clopper_pearson_lower: k must be in [0, n] (got k=" +
            std::to_string(k) + ", n=" + std::to_string(n) + ")");
    }
    require_alpha(alpha, "clopper_pearson_lower");

    // D7: zero observed successes gives a bound of EXACTLY zero. The closed
    // form below would ask for Beta^-1(alpha; 0, n+1), and Beta(0, .) is
    // degenerate -- scipy returns NaN there, which would poison a batch.
    if (k == 0) return 0.0;

    // D6: Beta^-1(alpha; k, n-k+1). NOTE the `alpha`, not `2*alpha`: the
    // paper's appendix cites `proportion_confint(..., alpha=2*alpha,
    // method="beta")`, but statsmodels halves alpha internally for its default
    // two-sided alternative, so the 2x cancels and this is the correct one-sided
    // (1-alpha) bound. Verified numerically: at this point P(X >= k) == alpha
    // exactly, whereas the 2*alpha form gives P(X >= k) == 2*alpha.
    return regularized_incomplete_beta_inverse(static_cast<double>(k),
                                               static_cast<double>(n - k) + 1.0,
                                               alpha);
}

// ===========================================================================
// Two-sided binomial test at p = 0.5  (D4)
// ===========================================================================
double binom_pvalue_half(int n_a, int n_b) {
    if (n_a < 0 || n_b < 0) {
        throw std::invalid_argument(
            "binom_pvalue_half: both counts must be >= 0 (got n_a=" +
            std::to_string(n_a) + ", n_b=" + std::to_string(n_b) + ")");
    }
    const int n = n_a + n_b;
    // A vacuous test is not evidence of anything, so it does not reject.
    if (n == 0) return 1.0;

    const double lower = binomial_cdf(n_a, n, 0.5);
    const double upper = binomial_sf(n_a, n, 0.5);
    return std::min(1.0, 2.0 * std::min(lower, upper));
}

// ===========================================================================
// Theorem 1  (D8)
// ===========================================================================
double certified_radius_from_probs(double sigma, double p_a, double p_b) {
    if (!(sigma > 0.0)) {
        throw std::invalid_argument(
            "certified_radius_from_probs: sigma must be > 0 (got " +
            std::to_string(sigma) + ")");
    }
    if (!(p_a > 0.0) || !(p_a <= 1.0) || !(p_b >= 0.0) || !(p_b <= 1.0)) {
        throw std::invalid_argument(
            "certified_radius_from_probs: need 0 < p_a <= 1 and 0 <= p_b <= 1");
    }
    if (p_a < p_b) {
        throw std::invalid_argument(
            "certified_radius_from_probs: p_a (" + std::to_string(p_a) +
            ") must be >= p_b (" + std::to_string(p_b) +
            "); a 'top' class less likely than the runner-up gives a negative "
            "radius, which hides the caller's bug instead of reporting it");
    }
    return 0.5 * sigma * (normal_quantile(p_a) - normal_quantile(p_b));
}

double certified_radius(double sigma, double p_lower) {
    if (!(sigma > 0.0)) {
        throw std::invalid_argument(
            "certified_radius: sigma must be > 0 (got " +
            std::to_string(sigma) + ")");
    }
    // Reuse the general form with the runner-up bound CERTIFY assumes, so the
    // two public entry points cannot drift apart (D8).
    return certified_radius_from_probs(sigma, p_lower, 1.0 - p_lower);
}

// ===========================================================================
// Theorem 6  (Appendix D)
// ===========================================================================
double certified_accuracy_lower_bound(int y, int m, double alpha, double rho) {
    if (m < 0) {
        throw std::invalid_argument(
            "certified_accuracy_lower_bound: m must be >= 0 (got " +
            std::to_string(m) + ")");
    }
    if (y < 0 || y > m) {
        throw std::invalid_argument(
            "certified_accuracy_lower_bound: y must be in [0, m] (got y=" +
            std::to_string(y) + ", m=" + std::to_string(m) + ")");
    }
    require_alpha(alpha, "certified_accuracy_lower_bound");
    require_alpha(rho, "certified_accuracy_lower_bound");
    if (m == 0) return 0.0;

    const double log_inv_rho = std::log(1.0 / rho);
    // Bernstein's inequality term from the proof (p. 24). max(0, .) because the
    // radicand goes negative for small m and large alpha, where the bound is
    // vacuous rather than undefined.
    const double radicand =
        2.0 * static_cast<double>(m) * alpha * (1.0 - alpha) * log_inv_rho -
        log_inv_rho / 3.0;
    const double slack =
        radicand > 0.0 ? std::sqrt(radicand) : 0.0;

    // Deliberately NOT clamped to [0, 1]: this is a statistic, and a negative
    // value legitimately means "no certified examples at this confidence".
    return (1.0 / (1.0 - alpha)) *
           (static_cast<double>(y) - alpha * static_cast<double>(m) - slack) /
           static_cast<double>(m);
}

// ===========================================================================
// SAMPLEUNDERNOISE  (D1, D9)
// ===========================================================================
std::vector<int> sample_under_noise(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples,
                                    unsigned seed) {
    require_sigma(sigma, "sample_under_noise");
    if (num_samples < 0) {
        throw std::invalid_argument(
            "sample_under_noise: num_samples must be >= 0 (got " +
            std::to_string(num_samples) + ")");
    }
    if (num_samples == 0) return {};

    if (x.rows != 1) {
        throw std::invalid_argument(
            "sample_under_noise: x must hold exactly ONE example per call "
            "(got " + std::to_string(x.rows) + " rows). PREDICT and CERTIFY "
            "certify around a single point; certify a batch by looping.");
    }

    // D9: a local engine plus an EXPLICIT Box-Muller transform, not
    // std::normal_distribution -- the distribution objects are not portable
    // across standard library implementations, so a saved seed would not
    // reproduce the same certified radius on a different toolchain.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const size_t dim = x.cols;
    std::vector<int> counts;
    Tensor noisy(1, dim);

    for (int i = 0; i < num_samples; ++i) {
        // Box-Muller. `uniform` returns [0, 1), so 1 - u is in (0, 1] and the
        // logarithm's argument is strictly positive.
        //
        // Each pair of draws yields TWO INDEPENDENT standard normals, so
        // consecutive coordinates are filled pairwise. (An earlier draft tried
        // to fan one pair out across coordinates with a phase rotation
        // 2*pi*j/2; that is only correct for dim == 2 and silently produces
        // correlated, wrong-magnitude noise for dim >= 3.)
        size_t j = 0;
        while (j < dim) {
            const double u1 = uniform(gen);
            const double u2 = uniform(gen);
            const double mag = sigma * std::sqrt(-2.0 * std::log(1.0 - u1));
            const double angle = 2.0 * M_PI * u2;

            // D1: additive, isotropic, standard deviation sigma.
            noisy[0][j] = x[0][j] + mag * std::cos(angle);
            if (j + 1 < dim) {
                noisy[0][j + 1] = x[0][j + 1] + mag * std::sin(angle);
            }
            j += 2;
        }

        const int cls = f(noisy);
        if (cls < 0) {
            throw std::invalid_argument(
                "sample_under_noise: the base classifier returned class " +
                std::to_string(cls) + ", but class indices must be >= 0");
        }
        if (static_cast<size_t>(cls) >= counts.size()) {
            counts.resize(static_cast<size_t>(cls) + 1, 0);
        }
        ++counts[static_cast<size_t>(cls)];
    }
    return counts;
}

// ===========================================================================
// PREDICT  (D3, D4)
// ===========================================================================
namespace {

// Index of the largest count, and the runner-up count (D3). Ties resolve to the
// LOWEST class index, which is deterministic; a tie is not evidence either way,
// so the p-value test below is what actually decides abstention.
int top_two(const std::vector<int>& counts, int& n_b) {
    n_b = 0;
    if (counts.empty()) return -1;

    int best = 0;   // index of the largest count
    int best_val = counts[0];
    int second_val = 0;   // largest count strictly below best_val
    for (size_t c = 1; c < counts.size(); ++c) {
        const int v = counts[c];
        if (v > best_val) {
            // The old best becomes the runner-up. Its COUNT goes into second_val
            // -- the first draft wrote `second = best` here, which stored the
            // winning class INDEX into a slot that holds a count, and the bug
            // only bites when counts[0] == 0 (a classifier that never emits
            // class 0). That silently turned n_b into 0 for every such input.
            second_val = best_val;
            best_val = v;
            best = static_cast<int>(c);
        } else if (v > second_val) {
            second_val = v;
        }
    }
    n_b = second_val;
    return best;
}

}  // namespace

SmoothedPrediction smoothed_predict(const BaseClassifierFn& f, const Tensor& x,
                                    double sigma, int num_samples, double alpha,
                                    unsigned seed) {
    require_sigma(sigma, "smoothed_predict");
    require_alpha(alpha, "smoothed_predict");
    if (num_samples < 1) {
        throw std::invalid_argument(
            "smoothed_predict: num_samples must be >= 1 (got " +
            std::to_string(num_samples) + ")");
    }

    const std::vector<int> counts =
        sample_under_noise(f, x, sigma, num_samples, seed);

    SmoothedPrediction out;
    int n_b = 0;
    const int top = top_two(counts, n_b);
    if (top < 0) {
        // No samples produced a class; abstain rather than guess.
        return out;
    }
    const int n_a = counts[static_cast<size_t>(top)];

    out.class_index = top;
    out.n_a = n_a;
    out.n_b = n_b;
    // D3: the test is on the TOP TWO counts only, with n = n_a + n_b -- NOT the
    // full sample count. D4: two-sided at p = 0.5.
    out.p_value = binom_pvalue_half(n_a, n_b);
    // D4: abstain iff the p-value EXCEEDS alpha.
    out.abstained = out.p_value > alpha;
    return out;
}

// ===========================================================================
// CERTIFY  (D5, D2)
// ===========================================================================
CertifiedPrediction certify(const BaseClassifierFn& f, const Tensor& x,
                            const SmoothingConfig& cfg) {
    require_sigma(cfg.sigma, "certify");
    require_alpha(cfg.alpha, "certify");
    if (cfg.num_guess_samples < 1) {
        throw std::invalid_argument(
            "certify: num_guess_samples must be >= 1 (got " +
            std::to_string(cfg.num_guess_samples) + ")");
    }
    if (cfg.num_samples < 1) {
        throw std::invalid_argument(
            "certify: num_samples must be >= 1 (got " +
            std::to_string(cfg.num_samples) + ")");
    }

    CertifiedPrediction out;

    // Round 1: guess the class. D5: these samples are DISCARDED -- the
    // Clopper-Pearson bound is built from the second round only, because the
    // class was CHOSEN using the first, so pooling them would be a bound on a
    // statistic that was selected on the data it is computed from.
    const std::vector<int> guess_counts =
        sample_under_noise(f, x, cfg.sigma, cfg.num_guess_samples, cfg.seed);
    int guess_runner_up = 0;
    const int guess_top = top_two(guess_counts, guess_runner_up);
    if (guess_top < 0) {
        return out;  // abstain
    }
    out.class_index = guess_top;

    // Round 2: estimate p_A on a FRESH set of samples.
    const std::vector<int> counts =
        sample_under_noise(f, x, cfg.sigma, cfg.num_samples, cfg.seed + 1u);
    if (counts.empty() ||
        static_cast<size_t>(out.class_index) >= counts.size()) {
        // The class vanished entirely in the second round; p_A = 0 and the
        // Clopper-Pearson bound at k = 0 is exactly 0 (D7), so abstain below.
        out.count = 0;
        out.p_lower = 0.0;
        out.radius = 0.0;
        return out;
    }
    out.count = counts[static_cast<size_t>(out.class_index)];

    out.p_lower = clopper_pearson_lower(out.count, cfg.num_samples, cfg.alpha);

    // D2: STRICTLY greater than 1/2. At p_lower == 1/2 the radius is exactly 0,
    // which certifies nothing, so it must abstain rather than return R = 0.
    if (!(out.p_lower > 0.5)) {
        out.abstained = true;
        out.radius = 0.0;
        return out;
    }
    out.abstained = false;
    out.radius = certified_radius(cfg.sigma, out.p_lower);
    return out;
}

// ===========================================================================
// Training under smoothing  (§4 / Appendix J.2)
// ===========================================================================
void smoothing_train(Model& model, const Tensor& X, const Tensor& y,
                     const SmoothingConfig& cfg, double learning_rate, int epochs) {
    if (model.layers.empty()) {
        throw std::invalid_argument(
            "smoothing_train: the model has no layers, so there is nothing to "
            "train");
    }
    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "smoothing_train: X.rows (" + std::to_string(X.rows) +
            ") must equal y.rows (" + std::to_string(y.rows) + ")");
    }
    if (X.cols == 0 || y.cols == 0) {
        throw std::invalid_argument("smoothing_train: X and y must be non-empty");
    }
    require_sigma(cfg.sigma, "smoothing_train");
    require_alpha(cfg.alpha, "smoothing_train");

    CrossEntropyLoss ce;
    Tensor go(1, 1);
    go(0, 0) = 1.0;

    // One engine for the whole run, advanced across epochs, so two epochs do
    // not replay the same noise (D9).
    std::mt19937 gen(cfg.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            const Tensor sample(1, X.cols, &X.data[i * X.cols]);
            const Tensor label(1, 1, &y.data[i * y.cols]);

            // D1: additive isotropic Gaussian noise of standard deviation
            // sigma. Each Box-Muller pair yields two INDEPENDENT standard
            // normals, filled pairwise.
            Tensor noisy(1, X.cols);
            size_t j = 0;
            while (j < X.cols) {
                const double u1 = uniform(gen);
                const double u2 = uniform(gen);
                const double mag =
                    cfg.sigma * std::sqrt(-2.0 * std::log(1.0 - u1));
                const double angle = 2.0 * M_PI * u2;
                noisy[0][j] = sample[0][j] + mag * std::cos(angle);
                if (j + 1 < X.cols) {
                    noisy[0][j + 1] = sample[0][j + 1] + mag * std::sin(angle);
                }
                j += 2;
            }

            Tensor logits = model.forward(noisy);
            total_loss += ce.forward(logits, label)[0][0];

            Tensor d_logits = ce.backward(logits, label, go);
            model.backward(d_logits, 0.0);
            for (auto& layer : model.layers) {
                layer->update_weights(learning_rate);
                layer->zero_grad();
            }
        }
        for (auto& layer : model.layers) layer->zero_grad();
    }
}