#include "vqcodebook.h"
#include "gmm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ===========================================================================
// Helpers
// ===========================================================================

// `Tensor::operator()` indexes `data[r * cols + c]` with NO bounds check, so a
// shape mismatch is silent heap corruption rather than an exception. The repo
// has already been bitten by exactly this in the VAT fixture, where a
// one-past-the-end write landed on the adjacent heap block.
static void require_same_shape(const Tensor& a, const Tensor& b,
                               const char* what) {
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::invalid_argument(
            std::string("vqcb: ") + what + " shape mismatch (" +
            std::to_string(a.rows) + "x" + std::to_string(a.cols) + " vs " +
            std::to_string(b.rows) + "x" + std::to_string(b.cols) + ")");
}

// ===========================================================================
// k-means codebook initialisation
// ===========================================================================
Tensor vqcb_kmeans_initialise(const Tensor& z, size_t num_codes,
                              unsigned int seed, int max_iter) {
    if (z.rows == 0 || z.cols == 0)
        throw std::invalid_argument("vqcb_kmeans_initialise: empty input");
    // D8. Checked HERE, before delegating, so the message names the constraint
    // the caller actually violated. The shipped KMeans::fit also throws, but
    // its message ("n_samples < n_clusters") is about k-means, not about a
    // codebook, and a caller reading it would not know a codebook was
    // involved.
    if (z.rows < num_codes)
        throw std::invalid_argument(
            "vqcb_kmeans_initialise: latent batch (" + std::to_string(z.rows) +
            ") must be >= num_codes (" + std::to_string(num_codes) +
            ") -- zero-padding a short batch would produce a degenerate "
            "codebook");
    if (num_codes == 0)
        throw std::invalid_argument("vqcb_kmeans_initialise: num_codes == 0");

    // Reuse the shipped Lloyd's implementation rather than writing a second
    // one. KMeans::fit throws on n_samples < n_clusters, which cannot happen
    // here given the check above.
    KMeans km(num_codes, static_cast<size_t>(max_iter > 0 ? max_iter : 1),
              1e-4, seed);
    km.fit(z);

    // THE TRANSPOSE. `cluster_centers()` is (K, D) -- one centre per ROW -- and
    // this repo's codebook convention is (D, K) with COLUMN k IS e_k
    // (vq_vae.h D3, matching Sonnet's self._w). Missing this is the single
    // easiest mistake in the function: with a square fixture nothing catches
    // it, and it silently pairs every code with the wrong dimension.
    const Tensor& centers = km.cluster_centers();
    if (centers.rows != num_codes || centers.cols != z.cols)
        throw std::runtime_error(
            "vqcb_kmeans_initialise: KMeans returned a (K, D) codebook of the "
            "wrong shape -- internal contract violation");
    Tensor out(z.cols, num_codes);
    for (size_t k = 0; k < num_codes; ++k)
        for (size_t d = 0; d < z.cols; ++d)
            out(d, k) = centers(k, d);
    return out;
}

// ===========================================================================
// Row ordering by descending error
// ===========================================================================
std::vector<size_t> vqcb_rows_by_error_desc(const Tensor& rows,
                                            const Tensor& z,
                                            const Tensor& z_q) {
    require_same_shape(rows, z, "rows vs z");
    require_same_shape(rows, z_q, "rows vs z_q");
    if (rows.rows == 0)
        throw std::invalid_argument("vqcb_rows_by_error_desc: empty rows");

    const size_t n = rows.rows;
    std::vector<double> err(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t d = 0; d < rows.cols; ++d) {
            const double diff = z_q(i, d) - z(i, d);
            s += diff * diff;
        }
        err[i] = s;
    }

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;

    // D7: DESCENDING error, ties by LOWEST ROW INDEX.
    //
    // The tie-break is written as an explicit second key rather than relying on
    // the sort's stability, because `std::stable_sort` would give the same
    // answer here (the input order IS ascending index) but the intent would be
    // invisible to a reader. An EXACT tie is only observable on a fixture
    // where two errors are bit-identical -- no tolerance-based test can see
    // one, which is why the suite builds such a fixture deliberately.
    std::sort(order.begin(), order.end(), [&err](size_t a, size_t b) {
        if (err[a] != err[b]) return err[a] > err[b];   // descending
        return a < b;                                    // lowest index wins
    });
    return order;
}

// ===========================================================================
// LRU accounting
// ===========================================================================
void vqcb_lru_tick(Tensor& ages, const std::vector<size_t>& used) {
    const size_t k_count = ages.cols;

    // Validate BEFORE writing anything, so a bad index cannot leave the ages
    // half-updated. `Tensor::operator()` does no bounds checking.
    for (size_t k : used)
        if (k >= k_count)
            throw std::invalid_argument(
                "vqcb_lru_tick: code index " + std::to_string(k) +
                " out of range for a codebook with " + std::to_string(k_count) +
                " codes");

    for (size_t k = 0; k < k_count; ++k) ages(0, k) += 1.0;
    for (size_t k : used) ages(0, k) = 0.0;   // idempotent on duplicates
}

// ===========================================================================
// LRU replacement
// ===========================================================================
size_t vqcb_lru_replace(Tensor& codebook, Tensor& ages, const Tensor& rows,
                        double lifespan) {
    if (codebook.rows != rows.cols)
        throw std::invalid_argument(
            "vqcb_lru_replace: replacement rows are (N, " +
            std::to_string(rows.cols) + ") but the codebook has embedding dim " +
            std::to_string(codebook.rows));
    if (ages.cols != codebook.cols)
        throw std::invalid_argument(
            "vqcb_lru_replace: ages has " + std::to_string(ages.cols) +
            " entries but the codebook has " + std::to_string(codebook.cols) +
            " codes");

    const size_t k_count = codebook.cols;

    // Collect the dead slots FIRST, before mutating anything, so an
    // impossible-to-satisfy request fails without a partially-written
    // codebook.
    std::vector<size_t> dead;
    for (size_t k = 0; k < k_count; ++k)
        // D6: the threshold is >=, so a code unused for EXACTLY `lifespan`
        // iterations is replaced.
        if (ages(0, k) >= lifespan) dead.push_back(k);

    if (dead.empty()) return 0;

    if (rows.rows == 0)
        throw std::invalid_argument(
            "vqcb_lru_replace: " + std::to_string(dead.size()) +
            " code(s) are dead but no replacement rows were supplied -- a dead "
            "code receives EXACTLY zero gradient (paper §3.1: \"Cz \\ Qz does "
            "not receive any gradient\"), so it cannot be recovered by "
            "training; a replacement vector must be supplied");

    // The replacement source. `rows` is a pool of encoder outputs; the error is
    // measured against the CURRENT codebook, i.e. how badly each candidate is
    // fit right now. Rows already assigned to a live code are still eligible:
    // the pool is the encoder's output distribution, not the live codebook.
    //
    // Build the paired (N, D) z_q ONCE -- each row's nearest current code. This
    // is the same nearest-code rule as the quantizer's, including its
    // FIRST-WINS tie-break (strict `<`), so the pool's ordering is measured
    // against the same geometry the quantizer will see next pass.
    Tensor zq_rows(rows.rows, codebook.rows);
    for (size_t i = 0; i < rows.rows; ++i) {
        size_t best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < k_count; ++k) {
            double s = 0.0;
            for (size_t d = 0; d < codebook.rows; ++d) {
                const double diff = codebook(d, k) - rows(i, d);
                s += diff * diff;
            }
            if (s < best_d) { best_d = s; best = k; }
        }
        for (size_t d = 0; d < codebook.rows; ++d) zq_rows(i, d) = codebook(d, best);
    }

    const std::vector<size_t> order =
        vqcb_rows_by_error_desc(rows, rows, zq_rows);

    // Consume the pool CYCLICALLY, so more dead codes than rows is legal.
    for (size_t j = 0; j < dead.size(); ++j) {
        const size_t slot = dead[j];
        const size_t src = order[j % order.size()];
        for (size_t d = 0; d < codebook.rows; ++d)
            codebook(d, slot) = rows(src, d);
        ages(0, slot) = 0.0;      // D6
    }
    return dead.size();
}

// ===========================================================================
// Committedness loss (Eq. 8-9)
// ===========================================================================
//
// Both terms are the same scalar. See the header for why that makes every
// value test incapable of separating them.

// (1 - beta) * d(z_e, sg[z_q]), with d = 0.5*mean over all M*D elements.
double vqcb_committedness_term1(const Tensor& z, const Tensor& z_q,
                                double beta) {
    require_same_shape(z, z_q, "z vs z_q");
    if (z.rows == 0 || z.cols == 0)
        throw std::invalid_argument("vqcb_committedness_term1: empty latents");
    double s = 0.0;
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d) {
            const double diff = z(i, d) - z_q(i, d);
            s += diff * diff;
        }
    return (1.0 - beta) * 0.5 * s /
           static_cast<double>(z.rows * z.cols);
}

// beta * d(sg[z_e], z_q) -- the same scalar, weighted the other way.
double vqcb_committedness_term2(const Tensor& z, const Tensor& z_q,
                                double beta) {
    require_same_shape(z, z_q, "z vs z_q");
    if (z.rows == 0 || z.cols == 0)
        throw std::invalid_argument("vqcb_committedness_term2: empty latents");
    double s = 0.0;
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d) {
            const double diff = z_q(i, d) - z(i, d);
            s += diff * diff;
        }
    return beta * 0.5 * s / static_cast<double>(z.rows * z.cols);
}

double vqcb_committedness_loss(const Tensor& z, const Tensor& z_q, double beta) {
    return vqcb_committedness_term1(z, z_q, beta) +
           vqcb_committedness_term2(z, z_q, beta);
}

// d/dz_e. Term 1 is the only term that touches z_e -- term 2 carries sg[z_e]
// (D4) -- so the factor is (1 - beta), NOT beta.
Tensor vqcb_committedness_gradient_ze(const Tensor& z, const Tensor& z_q,
                                      double beta) {
    require_same_shape(z, z_q, "z vs z_q");
    if (z.rows == 0 || z.cols == 0)
        throw std::invalid_argument("vqcb_committedness_gradient_ze: empty");
    const double n = static_cast<double>(z.rows * z.cols);
    Tensor g(z.rows, z.cols);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d)
            g(i, d) = (1.0 - beta) * (z(i, d) - z_q(i, d)) / n;
    return g;
}

// d/dz_q. Term 2 is the only term that touches z_q -- term 1 carries sg[z_q]
// (D4) -- so the factor is beta. The sign is (z_q - z_e): the codebook moves
// TOWARD the encoder output under gradient DESCENT, so the gradient must point
// away from z_e.
Tensor vqcb_committedness_gradient_codebook(const Tensor& z, const Tensor& z_q,
                                            double beta) {
    require_same_shape(z, z_q, "z vs z_q");
    if (z.rows == 0 || z.cols == 0)
        throw std::invalid_argument(
            "vqcb_committedness_gradient_codebook: empty");
    const double n = static_cast<double>(z.rows * z.cols);
    Tensor g(z.rows, z.cols);
    for (size_t i = 0; i < z.rows; ++i)
        for (size_t d = 0; d < z.cols; ++d)
            g(i, d) = beta * (z_q(i, d) - z(i, d)) / n;
    return g;
}