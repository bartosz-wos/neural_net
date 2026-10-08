#include "shapley.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

const size_t kExactShapleyMaxFeatures = 12;

namespace {

// Local LCG (Numerical Recipes constants). Deliberately NOT std::rand():
// attribution must not perturb the RNG stream of an unrelated suite, and this
// repo has a documented history of cross-suite RNG flakiness (the
// test_adaln_zero intermittent-NaN entry in NOT_FIXED.md).
//
// BUG FOUND 2026-10-07 — POOR OUTPUT ENTROPY. The first version returned
//     (state >> 33) & 0xFFFFFFFF
// i.e. bits 33..64 of the state. An LCG's HIGH bits are its well-distributed
// bits; the shift chosen here reads a window that includes the LOW, highly
// correlated end of the state, and the resulting stream is measurably
// NON-uniform. That silently corrupted the coalition-size sampler: driving an
// inverse-CDF with a biased u() yields a biased size distribution. Measured at
// M=4 against the SHAP kernel target p(d) = (0.200, 0.200, 0.600):
//     broken generator -> (0.400, 0.400, 0.200)
//     fixed generator  -> (0.200, 0.200, 0.600)
// The tell is that a BIASED estimator gets WORSE, not better, when handed more
// samples — a bias, not variance. This is the same shape as the sampler bug
// it masked, which is why that one stayed hidden.
//
// The fix is the standard xorshift64* finalizer on the whole state, which is
// what makes the high bits usable for both the float uniform and `below()`.
struct Lcg {
    uint64_t state;
    explicit Lcg(unsigned seed)
        : state(static_cast<uint64_t>(seed) * 2654435761u + 1u) {}
    uint64_t next64() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    }
    uint32_t next() {
        return static_cast<uint32_t>((next64() >> 32) & 0xFFFFFFFFu);
    }
    // Uniform in [0, 1), from the WELL-MIXED high bits.
    double uniform() {
        return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0);
    }
    // Uniform in [0, n) via multiply-shift. Modulo (`next() % n`) would be
    // biased whenever n does not divide 2^32; the multiply-shift keeps the
    // full precision of the 53-bit double mantissa.
    size_t below(size_t n) {
        if (n == 0) return 0;
        return static_cast<size_t>(uniform() * static_cast<double>(n));
    }
};

// Build a masked input: features in `mask` come from `x`, the rest from
// `baseline`. A feature is "present" iff its bit is set.
Tensor apply_mask(const Tensor& x, const Tensor& baseline,
                  const std::vector<bool>& mask) {
    Tensor out = Tensor::zeros(x.rows, x.cols);
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c)
            out[r][c] = mask[c] ? x[r][c] : baseline[r][c];
    return out;
}

// Popcount over features 0..M-1.
size_t mask_weight(const std::vector<bool>& mask) {
    size_t w = 0;
    for (size_t i = 0; i < mask.size(); ++i) if (mask[i]) ++w;
    return w;
}

void validate_inputs(const Tensor& x, const Tensor& baseline,
                     const char* who) {
    if (x.cols == 0) {
        throw std::invalid_argument(std::string(who) + ": input must have at least one feature");
    }
    if (x.rows != baseline.rows || x.cols != baseline.cols) {
        throw std::invalid_argument(std::string(who) + ": baseline shape must match input");
    }
}

// Solve A x = b by Gaussian elimination with partial pivoting.
// `A` is row-major n x n, modified in place. Returns false if a pivot is
// numerically zero (singular), leaving the caller to apply the ridge.
bool solve_dense(std::vector<double>& A, std::vector<double>& b, size_t n) {
    for (size_t col = 0; col < n; ++col) {
        // Partial pivoting: find the row with the largest |A[row][col]|.
        size_t pivot = col;
        double best = std::abs(A[col * n + col]);
        for (size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-300) return false;
        if (pivot != col) {
            for (size_t c = 0; c < n; ++c) std::swap(A[col * n + c], A[pivot * n + c]);
            std::swap(b[col], b[pivot]);
        }
        const double d = A[col * n + col];
        for (size_t r = col + 1; r < n; ++r) {
            const double factor = A[r * n + col] / d;
            if (factor == 0.0) continue;
            for (size_t c = col; c < n; ++c) A[r * n + c] -= factor * A[col * n + c];
            b[r] -= factor * b[col];
        }
    }
    // Back substitution.
    for (size_t ri = n; ri-- > 0;) {
        double s = b[ri];
        for (size_t c = ri + 1; c < n; ++c) s -= A[ri * n + c] * b[c];
        b[ri] = s / A[ri * n + ri];
    }
    return true;
}

} // namespace

double model_output(Model& model, const Tensor& input, size_t target) {
    if (input.cols == 0) {
        throw std::invalid_argument("model_output: input must have at least one feature");
    }
    // Attribution evaluates the model but never backpropagates, so no
    // parameter gradient is touched. Zero anyway to guarantee no residue if a
    // caller ran a training step before us (Dense::backward accumulates).
    for (auto& lp : model.layers) lp->zero_grad();

    Tensor out = model.forward(input);
    if (target >= out.cols) {
        throw std::out_of_range(
            "model_output: target index out of range for model output");
    }
    // Average over rows: a multi-row input is a batch of one game per row, but
    // the Shapley game here is defined on the row-averaged output so the
    // attributions stay a single vector (see header, D4).
    double s = 0.0;
    for (size_t i = 0; i < out.rows; ++i) s += out[i][target];
    return s / static_cast<double>(out.rows);
}

void kernel_weights(size_t m, Tensor& weights) {
    if (m == 0) {
        throw std::invalid_argument("kernel_weights: m must be >= 1");
    }
    // weights[d] = 1 / (m * C(m-1, d)).
    // C(m-1, d) built via a Pascal row over 0..m-1 (exact integers, no overflow
    // for the m <= 20 range where the weights are numerically sane anyway).
    std::vector<double> binom(m + 1, 0.0);
    binom[0] = 1.0;
    for (size_t i = 1; i <= m - 1; ++i)
        for (size_t j = i; j >= 1; --j) binom[j] = binom[j] + binom[j - 1];

    weights = Tensor::zeros(1, m + 1);
    const double dm = static_cast<double>(m);
    for (size_t d = 0; d <= m; ++d) {
        // d == m: C(m-1, m) == 0, so the raw formula divides by zero. The
        // convention used here is the d == 0 value, 1/m (the reference
        // implementation never evaluates this row: the efficiency constraint
        // determines the full coalition's contribution instead).
        weights[0][d] = (binom[d] > 0.0) ? 1.0 / (dm * binom[d]) : 1.0 / dm;
    }
}

ShapleyResult finalize_shapley(Model& /*model*/,
                               const Tensor& raw,
                               double fx,
                               double fbaseline,
                               size_t n_coalitions,
                               const std::string& method,
                               bool apply_efficiency_residual) {
    const size_t m = raw.cols;
    if (m == 0) throw std::invalid_argument("finalize_shapley: empty attribution vector");

    ShapleyResult r;
    r.attributions = raw.clone();
    r.base_value = fbaseline;
    r.n_features = m;
    r.n_coalitions = n_coalitions;
    r.method = method;

    // The efficiency residual: the reference implementation's
    //   phi[nonzero_inds[-1]] = (fx - fnull) - sum(w)
    // (shap/explainers/_kernel.py:786). The last feature absorbs the rounding
    // and truncation error of the other M-1, so sum(phi) == fx - fnull to
    // machine precision for any coalition set, including a badly
    // under-sampled one.
    //
    // It is applied ONLY to the sampled estimators (kernel / permutation),
    // never to exact_shapley — and that distinction is load-bearing:
    //
    // BUG FOUND 2026-10-07. exact_shapley enumerates the coalition lattice
    // exhaustively, so its answer is ALREADY efficient to machine precision;
    // the residual is a no-op there in exact arithmetic. In floating point it
    // is not merely unnecessary but ACTIVELY HARMFUL: it overwrites the
    // exactly-computed phi[M-1] with whatever the other M-1 happened to miss,
    // which breaks the SYMMETRY axiom. A model f(x) = x0 + x1 on x = (3, 5),
    // baseline 0 has phi = (3, 5) exactly; the residual rewrote it as (3, 5)
    // -> phi_1 = 8 - 3 = 5, which happens to agree only because the residual
    // was zero. Under any asymmetry in the sampling the last feature absorbs
    // an error it did not cause, and the attributions become an artifact of
    // FEATURE ORDER rather than of the game.
    //
    // SamplingWithoutReplacement in the reference applies the residual for the
    // same reason SamplingWithoutReplacement needs it: its M! orderings are
    // a sample, not the whole lattice.
    if (apply_efficiency_residual) {
        double partial = 0.0;
        for (size_t i = 0; i + 1 < m; ++i) partial += r.attributions[0][i];
        r.attributions[0][m - 1] = (fx - fbaseline) - partial;
    }

    r.completeness_delta = r.attributions.sum() - (fx - fbaseline);
    return r;
}

ShapleyResult exact_shapley(Model& model,
                            const Tensor& x,
                            const Tensor& baseline,
                            size_t target) {
    validate_inputs(x, baseline, "exact_shapley");
    const size_t m = x.cols;
    if (m > kExactShapleyMaxFeatures) {
        throw std::out_of_range(
            "exact_shapley: too many features for exhaustive enumeration "
            "(use kernel_shap)");
    }

    const double fbaseline = model_output(model, baseline, target);
    const double fx = model_output(model, x, target);

    Tensor kw;
    kernel_weights(m, kw);

    Tensor phi = Tensor::zeros(1, m);
    size_t coalitions = 0;

    for (size_t i = 0; i < m; ++i) {
        double acc = 0.0;
        // Enumerate every subset S of the OTHER features.
        const size_t others = m - 1;
        for (uint64_t sub = 0; sub < (1ULL << others); ++sub) {
            std::vector<bool> without_i(m, false), with_i(m, false);
            size_t k = 0;
            for (size_t f = 0; f < m; ++f) {
                if (f == i) continue;
                if (sub & (1ULL << k)) without_i[f] = true;
                ++k;
            }
            with_i = without_i;
            with_i[i] = true;

            // Shapley weight |S|!(M-|S|-1)!/M! = 1 / (M * C(M-1, |S|)).
            // Identical in closed form to the SHAP kernel weight — reuse the
            // table computed above so the exact and kernel estimators cannot
            // drift apart.
            const size_t sz = mask_weight(without_i);
            const double w = kw[0][sz];

            const double v_without = model_output(model, apply_mask(x, baseline, without_i), target);
            const double v_with = model_output(model, apply_mask(x, baseline, with_i), target);
            coalitions += 2;
            acc += w * (v_with - v_without);
        }
        phi[0][i] = acc;
    }

    ShapleyResult r = finalize_shapley(model, phi, fx, fbaseline, coalitions, "exact", false);
    return r;
}

ShapleyResult kernel_shap(Model& model,
                          const Tensor& x,
                          const Tensor& baseline,
                          size_t target,
                          size_t n_samples,
                          unsigned seed) {
    validate_inputs(x, baseline, "kernel_shap");
    if (n_samples == 0) {
        throw std::invalid_argument("kernel_shap: n_samples must be >= 1");
    }
    const size_t m = x.cols;
    const double fbaseline = model_output(model, baseline, target);
    const double fx = model_output(model, x, target);

    Tensor kw;
    kernel_weights(m, kw);

    // ---- KernelSHAP sampling ------------------------------------------------
    //
    // Faithful to shap/explainers/_kernel.py:419-470. Two structural facts
    // there are easy to miss and both matter:
    //
    //  1. THE KERNEL WEIGHT IS (M-1)/(d*(M-d)), not 1/(M*C(M-1,d)). The
    //     latter is algebraically correct for the EXACT Shapley sum but is
    //     NOT the sampling/least-squares weight of KernelSHAP. The reference
    //     builds weight_vector[(i-1)] = (M-1)/(i*(M-i)), doubles the first
    //     floor((M-1)/2) entries, and normalizes the vector to sum to 1.
    //  2. SUBSETS ARE PAIRED WITH THEIR COMPLEMENTS. For each subset S of
    //     size d <= M-d the reference ALSO evaluates its complement, and
    //     halves the per-sample weight accordingly. This antithetic pairing
    //     is why only sizes 1..ceil((M-1)/2) appear in the weight vector: the
    //     complement carries size M-d.
    //
    // The first implementation of this function sampled sizes uniformly and
    // weighted rows by kernel_weights(m)[|z|]. That produced a systematic bias
    // (~2.5e-2 mean max-error against exact_shapley on a 4-feature MLP,
    // FLAT in n_samples — the signature of bias, not variance) whereas the
    // correct paired scheme below reaches the same order as the exhaustive
    // least-squares solution.
    const size_t num_subset_sizes = (m + 1) / 2;   // ceil((M-1)/2)
    const size_t num_paired = m / 2;                // floor((M-1)/2)

    std::vector<double> weight_vector(num_subset_sizes, 0.0);
    for (size_t i = 1; i <= num_subset_sizes; ++i) {
        weight_vector[i - 1] =
            static_cast<double>(m - 1) / (static_cast<double>(i) * static_cast<double>(m - i));
        if (i <= num_paired) weight_vector[i - 1] *= 2.0;
    }
    {
        double total = 0.0;
        for (double wv : weight_vector) total += wv;
        if (total > 0.0) for (double& wv : weight_vector) wv /= total;
    }

    Lcg rng(seed);
    std::vector<std::vector<double>> rows;  // design matrix
    std::vector<double> ys;
    std::vector<double> ws;

    const size_t want = std::max<size_t>(n_samples, 2 * m + 2);
    std::vector<size_t> perm(m);


    for (size_t s = 0; s < want; ) {
        // Which subset size to draw from: inverse-CDF over weight_vector.
        double total = 0.0;
        for (double wv : weight_vector) total += wv;
        const double u = rng.uniform() * total;
        size_t d = 1;
        double acc = 0.0;
        for (size_t i = 1; i <= num_subset_sizes; ++i) {
            acc += weight_vector[i - 1];
            d = i;
            if (u < acc) break;
        }

        // Draw a uniform size-d subset by partial Fisher-Yates over indices.
        for (size_t f = 0; f < m; ++f) perm[f] = f;
        for (size_t k = 0; k < d; ++k) {
            const size_t pick = k + rng.below(m - k);
            std::swap(perm[k], perm[pick]);
        }

        const bool paired = (d <= num_paired);
        // Per-sample weight: the size class's share, divided over the number of
        // subsets at that size, halved again when the complement is paired.
        double w = weight_vector[d - 1];
        {
            // binom(M, d) computed exactly for the small d we can reach.
            double num = 1.0;
            for (size_t k = 0; k < d; ++k) num = num * static_cast<double>(m - k) / static_cast<double>(k + 1);
            w /= num;
        }
        if (paired) w /= 2.0;

        // Emit the subset, then its complement when paired.
        std::vector<bool> mask(m, false);
        for (size_t k = 0; k < d; ++k) mask[perm[k]] = true;
        rows.push_back([&] {
            std::vector<double> r(m, 0.0);
            for (size_t f = 0; f < m; ++f) r[f] = mask[f] ? 1.0 : 0.0;
            return r; }());
        ys.push_back(model_output(model, apply_mask(x, baseline, mask), target) - fbaseline);
        ws.push_back(w);
        ++s;

        if (paired && s < want) {
            std::vector<bool> comp(m, true);
            for (size_t k = 0; k < d; ++k) comp[perm[k]] = false;
            rows.push_back([&] {
                std::vector<double> r(m, 0.0);
                for (size_t f = 0; f < m; ++f) r[f] = comp[f] ? 1.0 : 0.0;
                return r; }());
            ys.push_back(model_output(model, apply_mask(x, baseline, comp), target) - fbaseline);
            ws.push_back(w);
            ++s;
        }

    }

    // Solve for the first M-1 features; the last is the efficiency residual.
    //
    // BUG FOUND 2026-10-07 — THE EFFICIENCY-CENTERED DESIGN MATRIX. The first
    // version accumulated the plain design matrix Z and the plain target
    // y = v(z) - f(empty). That solves a regression with an implicit intercept
    // pinned at 0, which is the wrong problem: SHAP's efficiency axiom says
    // the attributions must already account for the FULL output change
    // f(x) - f(baseline), so the regression has to be written relative to the
    // feature we solve for as the residual.
    //
    // The reference does exactly this (shap/explainers/_kernel.py:738-741):
    //     eyAdj2 = eyAdj - maskMatrix[:, last] * (fx - fnull)
    //     etmp   = Z - maskMatrix[:, last]
    // i.e. both the design matrix and the target are CENTERED on the last
    // feature. Solving the centered system for the other M-1, then setting
    //     phi[last] = (fx - fnull) - sum(w)
    // recovers the efficiency constraint by construction (same line 786).
    //
    // Isolated proof of the bug: on the linear game f(x) = sum c_i x_i, where
    // phi_i = c_i*(x_i - b_i) analytically, the PLAIN formulation returned
    // (3.25, 5.0, 7.75) against an oracle of (0.75, 2.5, 5.25) — and it did
    // so even with EXHAUSTIVE coalitions and exact SHAP weights, proving the
    // sampling and the weights were fine and the system being solved was
    // wrong. The CENTERED formulation returns (0.75, 2.5, 5.25) exactly.
    const size_t p = m - 1;
    const double total_change = fx - fbaseline;
    std::vector<double> raw(m, 0.0);
    if (p > 0) {
        // Centered design: e_r[i] = Z[r][i] - Z[r][p] for i < p, and
        // y2_r = ys[r] - Z[r][p] * (fx - fbaseline).
        std::vector<double> E(p), A(p * p, 0.0), b(p, 0.0);
        auto accumulate = [&]() {
            for (size_t r = 0; r < rows.size(); ++r) {
                const std::vector<double>& X = rows[r];
                const double last = X[p];
                const double W = ws[r];
                const double Y = ys[r] - last * total_change;
                // BUG FOUND 2026-10-07 — STALE `E` ON THE `continue` PATH. This
                // buffer is declared OUTSIDE the loop and refreshed per entry.
                // The zero-skip below therefore left E[j] holding a value from
                // the PREVIOUS row for every j after a skipped i, which leaked a
                // stale cross-term into A. This is the "two distinct
                // quantities in one scratch buffer" class of bug, and it is
                // invisible in any configuration where no E[i] is ever zero —
                // which is exactly why it survived every earlier test: the
                // sampled design almost always contains a coalition where some
                // feature matches the residual feature, so the skip DOES fire,
                // but the corruption is spread across rows and looks like
                // sampling noise rather than a deterministic bug.
                for (size_t i = 0; i < p; ++i) E[i] = X[i] - last;
                for (size_t i = 0; i < p; ++i) {
                    if (E[i] == 0.0) continue;
                    for (size_t j = 0; j < p; ++j) A[i * p + j] += W * E[i] * E[j];
                    b[i] += W * E[i] * Y;
                }
            }
        };
        accumulate();

        std::vector<double> A0 = A;
        if (!solve_dense(A, b, p)) {
            // Ridge fallback (documented deviation D3): the reference
            // implementation falls back to lstsq; this repo has none, so a
            // trace-scaled ridge takes its place. At 1e-10*trace this only
            // engages on a genuinely rank-deficient system.
            double trace = 0.0;
            for (size_t i = 0; i < p; ++i) trace += A0[i * p + i];
            const double lam = 1e-10 * (trace > 0.0 ? trace : 1.0);
            for (size_t i = 0; i < p; ++i) A0[i * p + i] += lam;
            b.assign(p, 0.0);
            accumulate();
            A = A0;
            solve_dense(A, b, p);
        }
        for (size_t i = 0; i < p; ++i) raw[i] = b[i];
    }

    Tensor raw_t = Tensor::zeros(1, m);
    for (size_t i = 0; i < m; ++i) raw_t[0][i] = raw[i];
    return finalize_shapley(model, raw_t, fx, fbaseline, rows.size(), "kernel", true);
}

ShapleyResult path_dependent_shap(Model& model,
                                  const Tensor& x,
                                  const Tensor& baseline,
                                  size_t target,
                                  size_t n_permutations,
                                  unsigned seed) {
    validate_inputs(x, baseline, "path_dependent_shap");
    if (n_permutations == 0) {
        throw std::invalid_argument("path_dependent_shap: n_permutations must be >= 1");
    }
    const size_t m = x.cols;
    if (m < 1) {
        throw std::invalid_argument(
            "path_dependent_shap: needs at least 1 feature");
    }
    const double fbaseline = model_output(model, baseline, target);
    const double fx = model_output(model, x, target);

    Lcg rng(seed);
    std::vector<double> phi(m, 0.0);
    size_t coalitions = 0;

    std::vector<size_t> order(m);
    // Antithetic sampling (shap/explainers/_permutation.py:189-221): each
    // iteration walks the ordering FORWARD and then the REVERSED ordering, so
    // every prefix of the forward pass is matched by a suffix of the reverse
    // pass. The reference divides by 2*npermutations for exactly this reason.
    for (size_t p = 0; p < n_permutations; ++p) {
        for (size_t i = 0; i < m; ++i) order[i] = i;
        // Fisher-Yates with the local LCG.
        for (size_t i = m; i-- > 1;) {
            const size_t j = rng.below(i + 1);
            std::swap(order[i], order[j]);
        }

        // Forward pass: feature order[k] joins a prefix of size k.
        std::vector<bool> prefix(m, false);
        double v_prefix = fbaseline;
        for (size_t k = 0; k < m; ++k) {
            const size_t f = order[k];
            prefix[f] = true;
            const double v_with = model_output(model, apply_mask(x, baseline, prefix), target);
            ++coalitions;
            phi[f] += (v_with - v_prefix);
            v_prefix = v_with;
        }

        // Backward (antithetic) pass: walk DOWN from the full coalition,
        // stripping features off the END of the ordering. Feature order[k]
        // joins the SUFFIX {order[k+1..m-1]} — the complement of the prefix it
        // was credited with in the forward pass. Pairing each prefix with the
        // matching suffix is the variance reduction the reference performs, and
        // it is why the denominator is 2*n_permutations rather than
        // n_permutations.
        //
        // BUG FOUND 2026-10-07 — THE LOOP BOUND DROPPED THE order[0] STEP. This
        // loop read `for (size_t k = m; k-- > 1;)`, which visits
        // k = m-1, m-2, ..., 1 and SKIPS k = 0. That is not a rounding detail:
        // the skipped step is the one where feature order[0] joins the suffix
        // {order[1..m-1]}, which is precisely the pairing with the reference's
        // `for ind in inds: row_values[ind] += outputs[i] - outputs[i+1]`
        // first iteration (shap/explainers/_permutation.py:213-216). Verified
        // against the reference source rather than assumed: those masks are
        // DELTA/XOR masks (shap/utils/_masked_model.py:91,
        // `delta_mask = mask ^ last_mask`), so entry i+1 toggles feature
        // inds[i] OFF as the sequence walks inds[0], inds[1], ... downward
        // from the full coalition. Walking the same array in reverse WITHOUT
        // the extra step credits order[0] once (forward only) and every other
        // position twice, while still dividing by 2*n_permutations.
        //
        // The resulting bias is exactly 1 - 1/(2m): with m positions, each
        // permutation credits its order[0] feature once and the other m-1
        // twice, so the mean is (2 - 1/m)/2 = 1 - 1/(2m) of the truth. It is
        // uniform across features (order[0] is uniformly random), it is
        // systematic rather than random, and — the tell that it is a bias and
        // not noise — it does NOT shrink as n_permutations grows.
        //
        // Isolated proof on the identity game f(x) = x, where every marginal is
        // exactly x_i for every coalition and the estimator is therefore
        // EXACT for any budget. Measured mean bias over 40 seeds was
        // -1.50e-01 / +6.7e-02 / +8.3e-02 at n_permutations = 100 and at
        // 100000 — a 1000x budget increase moving the bias in the third
        // decimal, i.e. zero convergence. Predicted by the 5/6 factor at
        // m=3: 0.9 * 5/6 = 0.75 (measured 0.7501) and -0.4 * 5/6 = -0.3333
        // (measured -0.3333). Third feature differs only because
        // finalize_shapley's efficiency residual overwrites phi[m-1] with the
        // residual of the other m-1 — correct behaviour, and it is why the
        // probe's third entry is not the raw estimator output.
        std::vector<bool> suffix(m, true);
        double v_suffix = fx;
        for (size_t k = m; k-- > 0;) {
            const size_t f = order[k];
            suffix[f] = false;
            const double v_without = model_output(model, apply_mask(x, baseline, suffix), target);
            ++coalitions;
            phi[f] += (v_suffix - v_without);
            v_suffix = v_without;
        }
    }

    Tensor raw = Tensor::zeros(1, m);
    for (size_t i = 0; i < m; ++i)
        raw[0][i] = phi[i] / static_cast<double>(2 * n_permutations);
    return finalize_shapley(model, raw, fx, fbaseline, coalitions, "permutation", true);
}