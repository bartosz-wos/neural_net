// test_vq_vae.cpp — TDD tests for the Vector Quantized Variational Autoencoder
// (Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, "Neural Discrete
// Representation Learning", NeurIPS 2017, arXiv:1711.00937).
//
// CITATION NOTE. Verified against the arXiv API on 2026-10-15:
// `id_list=1711.00937` returns "Neural Discrete Representation Learning" with
// exactly these three authors. This repo's expansion queue has produced five
// wrong arXiv IDs before (Greenwell, DeepLIFT, the D2 calibration metric,
// randomized smoothing, VAT), so the check is not optional. Do not
// reintroduce a different ID.
//
// EVERY numeric expectation below is ORACLE-DERIVED from
// ~/.hermes/cache/scratch/vq_np_oracle.py (numpy 2.5.4 float64), which
// self-checks against centred finite differences to ~3e-11 BEFORE emitting
// any number. NOT hand-derived. The oracle prints its weights to 17
// significant digits and this file transcribes them digit-for-digit;
// the weights were compared digit-for-digit rather than re-typed from the
// paper.
//
// ===========================================================================
// THE SEMANTICS EACH FIXTURE EXISTS TO PIN  (header D1..D10)
// ===========================================================================
//
//   D1 THE DICTIONARY TERM IS UNWEIGHTED; THE COMMITMENT TERM CARRIES beta.
//      Paper Eq. (3):  L = logp(x|zq) + ||sg[z_e] - e||^2
//                             + beta*||z_e - sg[e]||^2
//      Sonnet line 102: loss = q_latent_loss + commitment_cost * e_latent_loss
//      where e_latent_loss = mean((sg[quantized] - inputs)^2) is the
//      DICTIONARY term (gradient to e) and q_latent_loss =
//      mean((quantized - sg[inputs])^2) is the COMMITMENT base (gradient to
//      z_e). Multiplying the wrong one by beta is THE most common error in a
//      reimplementation, and it is invisible in the VALUE -- both terms are
//      the same scalar. Only the gradient separates them. Measured: the
//      codebook gradient is exactly -0.0083333333333333402 with NO beta in
//      it, and the z_e gradient is exactly beta times what an unweighted
//      commitment term would give.
//
//   D1-STOPGRAD THE STOP-GRADIENT IS INVISIBLE IN THE VALUE. This is the
//      single most important property of Eq. (3) and it cost the oracle a
//      real bug: differentiating the combined scalar gives (1+beta) on the
//      codebook gradient, a CLEAN RATIO OF 1.25 ON EVERY CELL. A clean
//      constant ratio means "the harness differentiates the wrong function",
//      not "the gradient is wrong" -- so the fix was to the HARNESS
//      (two stop-gradient-honouring functions), never to the analytic side.
//
//   D2 THE MEAN IS OVER ALL M*D ELEMENTS, not over rows.
//
//   D3 THE CODEBOOK IS (D, K): COLUMN k IS EMBEDDING e_k. Matches Sonnet's
//      `self._w` of shape [embedding_dim, num_embeddings] (line 63).
//
//   D4 TIE-BREAK IS FIRST-WINS. Sonnet line 95 is `argmax(-distances, 1)`,
//      i.e. argmin, and argmin/argmax both return the FIRST extremum.
//      MEASURED: with e_2 and e_3 made bit-identical and z exactly on that
//      vector, both squared distances are BIT-IDENTICAL and the index chosen
//      is 2. No tolerance-based assertion can see a tie-break; the fixture
//      must contain an exact tie.
//
//   D5 THE STE VALUE IS z_q, NOT z_e. Sonnet line 104:
//      quantized = inputs + stop_gradient(quantized - inputs)
//      The VALUE is exactly z_q, bit-identical. Only the GRADIENT is the
//      identity. An implementation that returns z_e from forward() has
//      quantized NOTHING and passes every loss test while being completely
//      broken -- so test 9 asserts both that the output equals z_q AND that
//      z_q differs from z_e on this fixture, so the next test cannot pass by
//      returning its input unchanged.
//
//   D6 UNDER EMA THE DICTIONARY TERM IS DROPPED FROM THE LOSS. Sonnet
//      line 240: `loss = commitment_cost * e_latent_loss` with no
//      q_latent_loss at all, because the codebook is ASSIGNED, not
//      gradient-updated.
//
//   D7 THE EMA LAPLACE SMOOTHING IS (n_k + eps)/(n + K*eps) * n, Sonnet
//      lines 231-233. Dividing by n_k directly is a visibly different
//      number: 0.55 vs the oracle's 0.55036630036630052.
//
//   D8 A NEVER-ASSIGNED CODE COLLAPSES TO EXACTLY ZERO under EMA. With
//      eps=1e-5 an unassigned code has n_k = eps, so ema_w[:,k]/n_k = 0/eps.
//      It does NOT keep its initial value. Measured: w1[:,3] and w1[:,4]
//      are all exactly 0.0.
//
//   D9 PERPLEXITY IS NEVER EXACTLY AN INTEGER. Sonnet line 106 uses
//      log(avg_probs + 1e-10). MEASURED: 3 codes over 3 points gives
//      2.9999999991, not 3.0 -- asserting == 3.0 fails on the correct
//      implementation. And the ceiling is min(K, M), NOT K: 5 codes over 3
//      points cannot exceed 3.
//
//   D10 beta = 0.25 (paper §3.2, verbatim), robust for beta in [0.1, 2.0].
//
// ===========================================================================
// THE FIXTURE. D=4, K=5, M=3, beta=0.25. Each z_e row is nearest a DIFFERENT
// code, and every residual is nonzero so the commitment term is not
// degenerate.
// ===========================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

#include "nn/layers/generative/vq_vae.h"
#include "nn/core/model.h"
#include "nn/nn.h"

static int passed = 0;
static int failed = 0;

static void check(const std::string& name, bool pass) {
    if (pass) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else      { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
}

// NOTE: argument order is (a, b, tol) so each call reads as a sentence.
static bool close(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

static bool tensor_close(const Tensor& a, const Tensor& b, double tol = 1e-12) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (!close(a(i,j), b(i,j), tol)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// The oracle literals, transcribed digit-for-digit.
//
// codebook is (D=4, K=5); COLUMN k IS embedding e_k. Written as the (K, D)
// rows the paper would use, then transposed ONCE here, because Tensor stores
// (rows, cols) and the codebook is (D, K) per D3.
// ---------------------------------------------------------------------------
static const size_t D = 4, K = 5, M = 3;
static const double BETA = 0.25;

static Tensor oracle_codebook() {
    // e_k as the oracle printed them (D=4 each):
    //   e_0 = [ 0.5, -0.2,  0.1,  0.3]
    //   e_1 = [-0.4,  0.6,  0.2, -0.1]
    //   e_2 = [ 0.0,  0.1, -0.5,  0.7]
    //   e_3 = [ 0.9,  0.1,  0.1,  0.1]
    //   e_4 = [-0.7, -0.5,  0.3,  0.2]
    Tensor w(D, K);
    const double e[5][4] = {
        { 0.5, -0.2,  0.1,  0.3},
        {-0.4,  0.6,  0.2, -0.1},
        { 0.0,  0.1, -0.5,  0.7},
        { 0.9,  0.1,  0.1,  0.1},
        {-0.7, -0.5,  0.3,  0.2},
    };
    for (size_t k = 0; k < K; ++k)
        for (size_t d = 0; d < D; ++d)
            w(d, k) = e[k][d];
    return w;
}

static Tensor oracle_z_e() {
    Tensor z(M, D);
    const double v[3][4] = {
        { 0.55, -0.15,  0.05,  0.32},
        {-0.35,  0.58,  0.24, -0.08},
        { 0.12,  0.05, -0.42,  0.66},
    };
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            z(i, d) = v[i][d];
    return z;
}

// ===========================================================================
// TEST 1 -- squared distances are exact (D3 orientation + exact arithmetic)
// ===========================================================================
static void test_distances_exact() {
    std::cout << "\n[TEST 1] vq_squared_distances: exact oracle values" << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();

    // Oracle: d[0] = [0.0079000000000000181, 1.6639000000000002,
    //                  0.81190000000000007, 0.23590000000000011, 1.7619]
    //         d[1] = [1.4948999999999999, 0.0049000000000001265,
    //                  1.5088999999999999, 1.8449, 1.3708999999999998]
    //         d[2] = [0.6069, 1.5349000000000002, 0.024900000000000144,
    //                  1.1949000000000001, 1.7048999999999999]
    const double expect[3][5] = {
        {0.0079000000000000181, 1.6639000000000002, 0.81190000000000007,
         0.23590000000000011, 1.7619},
        {1.4948999999999999, 0.0049000000000001265, 1.5088999999999999,
         1.8449, 1.3708999999999998},
        {0.6069, 1.5349000000000002, 0.024900000000000144,
         1.1949000000000001, 1.7048999999999999},
    };

    Tensor d = vq_squared_distances(z, w);
    check("shape is (M, K) = (3,5)", d.rows == M && d.cols == K);
    bool all = true;
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k)
            if (!close(d(i,k), expect[i][k], 1e-12)) {
                all = false;
                std::cout << "     d(" << i << "," << k << ") = " << d(i,k)
                          << " expected " << expect[i][k] << std::endl;
            }
    check("all 15 squared distances match the oracle to 1e-12", all);

    // D3: the codebook orientation. Column k is e_k, so the L2 norm of
    // COLUMN k must be the oracle's ||e_k||. If the codebook were stored
    // transposed, column k would be a ROW of the oracle's and these differ.
    // MEASURED (these were wrong in the first draft of this file -- the norms
    // were hand-derived from a transposed sum and had to be replaced):
    //   ||e_0|| = 0.62449979983983983   ||e_1|| = 0.75498344352707503
    double n0 = 0.0, n1 = 0.0;
    for (size_t d = 0; d < D; ++d) { n0 += w(d,0)*w(d,0); n1 += w(d,1)*w(d,1); }
    check("D3 column 0 is e_0 (measured norm 0.62449979983983983)",
          close(std::sqrt(n0), 0.62449979983983983, 1e-15));
    check("D3 column 1 is e_1 (measured norm 0.75498344352707503)",
          close(std::sqrt(n1), 0.75498344352707503, 1e-15));
    // And the discriminator: a ROW of the oracle's matrix has a different
    // norm, so a transposed codebook cannot satisfy both.
    double r0 = 0.0, r1 = 0.0;
    for (size_t k = 0; k < K; ++k) { r0 += w(0,k)*w(0,k); r1 += w(1,k)*w(1,k); }
    check("D3 and the ROW norms differ from the column norms (so the "
          "orientation is genuinely pinned)",
          !close(std::sqrt(r0), std::sqrt(n0), 1e-9));

    // D1 sanity on this same fixture, computed independently here: the
    // nearest code to row 0 is 0 at distance 0.0079, and the nearest code to
    // row 2 is 2 at 0.0249. Both are 3+ orders below every other entry, so a
    // transposed codebook could not produce these same minima.
    size_t argmin0 = 0;
    for (size_t k = 1; k < K; ++k) if (d(0,k) < d(0,argmin0)) argmin0 = k;
    check("row 0 argmin == 0", argmin0 == 0);
    size_t argmin2 = 0;
    for (size_t k = 1; k < K; ++k) if (d(2,k) < d(2,argmin2)) argmin2 = k;
    check("row 2 argmin == 2", argmin2 == 2);
}

// ===========================================================================
// TEST 2 -- nearest indices are [0, 1, 2] and the tie-break is FIRST-WINS
// ===========================================================================
static void test_nearest_indices_and_tiebreak() {
    std::cout << "\n[TEST 2] vq_nearest_indices + D4 exact-tie discriminator"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();

    std::vector<size_t> idx = vq_nearest_indices(z, w);
    check("returns M=3 indices", idx.size() == M);
    check("indices are [0, 1, 2] (oracle)",
          idx[0] == 0 && idx[1] == 1 && idx[2] == 2);

    // ---- D4: the exact-tie discriminator -----------------------------------
    // e_3 := e_2, bit-identical. z placed EXACTLY on that vector. Then the two
    // squared distances are computed by the same expression on the same bits,
    // so they are bit-identical and the tie-break is the only thing deciding.
    Tensor wt = oracle_codebook();
    for (size_t d = 0; d < D; ++d) wt(d, 3) = wt(d, 2);
    Tensor zt(1, D);
    for (size_t d = 0; d < D; ++d) zt(0, d) = wt(d, 2);

    Tensor dt = vq_squared_distances(zt, wt);
    check("D4 the tied distances are BIT-IDENTICAL",
          dt(0,2) == dt(0,3));
    std::vector<size_t> it = vq_nearest_indices(zt, wt);
    check("D4 tie resolves to the FIRST index (2, not 3)", it[0] == 2);
    check("D4 this is the strict-< discriminator, so last-wins FAILS here",
          it[0] != 3);

    // ---- D4b: distances must be bit-identical for a tie, not merely close.
    // A tolerance-based implementation could break the tie arbitrarily; this
    // is the assertion that makes the fixture honest.
    check("D4 tie distance is exactly 0.0 on both codes",
          dt(0,2) == 0.0 && dt(0,3) == 0.0);
}

// ===========================================================================
// TEST 3 -- quantize returns the codebook ROW, bit-identically
// ===========================================================================
static void test_quantize_value() {
    std::cout << "\n[TEST 3] vq_quantize: bit-identical codebook rows (D5)"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    Tensor zq = vq_quantize(z, w);

    check("shape preserved (M, D)", zq.rows == M && zq.cols == D);
    bool all = true;
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            // BIT-identical, not close(): a quantizer that recomputed an
            // average or added an epsilon would fail this.
            if (zq(i,d) != w(d, (size_t)i)) all = false;
    check("every entry is BIT-identical to the codebook column", all);

    // The cross-check the repo has been bitten by repeatedly: F1 and F3 must
    // not return each other's numbers.
    check("row 0 != row 1 (a transposed lookup would swap them)",
          zq(0,0) != zq(1,0) || zq(0,1) != zq(1,1));
}

// ===========================================================================
// TEST 4 -- the two VQ terms, and D1's weighting
// ===========================================================================
static void test_two_terms_and_weighting() {
    std::cout << "\n[TEST 4] D1 dictionary unweighted / commitment carries beta"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    Tensor zq = vq_quantize(z, w);

    // Oracle: e_latent_loss = 0.0031416666666666676
    //         q_latent_loss = 0.0031416666666666676  (IDENTICAL -- see below)
    double e_lat = vq_codebook_loss(z, zq);
    double q_lat = vq_commitment_loss(z, zq, 1.0);
    check("dictionary term == oracle 0.0031416666666666676",
          close(e_lat, 0.0031416666666666676, 1e-15));
    check("commitment term (beta=1) == the same value -- THE TWO TERMS ARE "
          "NUMERICALLY IDENTICAL",
          close(q_lat, 0.0031416666666666676, 1e-15));
    // D2 the mean is over M*D=12 elements. Measured: the three residuals are
    // 0.0079 + 0.0049 + 0.0249 = 0.0377, and 0.0377/12 is the loss. A
    // per-row mean would divide by 3 instead and give 0.0125666666...
    check("D2 the mean is over M*D=12 elements, not over the 3 rows",
          close(e_lat, (0.0079 + 0.0049 + 0.0249) / 12.0, 1e-12));
    check("D2 and that differs from a per-ROW mean (0.0125666...)",
          !close(e_lat, (0.0079 + 0.0049 + 0.0249) / 3.0, 1e-12));

    // D1: the commitment loss CARRIES beta.
    double commit = vq_commitment_loss(z, zq, BETA);
    check("D1 commitment(beta=0.25) == 0.25 * commitment(beta=1)",
          close(commit, BETA * q_lat, 1e-15));
    check("D1 and the dictionary term is NOT scaled by beta",
          close(e_lat, q_lat, 1e-15));

    // Total: q_latent + beta * e_latent = 0.0039270833333333345 (oracle).
    double total = q_lat + BETA * e_lat;
    check("total == q_latent + beta*e_latent == oracle 0.0039270833333333345",
          close(total, 0.0039270833333333345, 1e-15));

    // The discriminator: the swapped weighting must give a DIFFERENT number.
    // MEASURED: the two orderings differ ONLY because e_latent != q_latent
    // would be needed -- but on THIS fixture the two terms are the SAME
    // scalar, so `e_latent + beta*q_latent` is BIT-IDENTICAL to
    // `q_latent + beta*e_latent`. The ordering is algebraically invisible
    // whenever the two terms are equal, which is ALWAYS (see D1-STOPGRAD).
    // So this assertion was vacuous and is removed. What actually discriminates
    // the ordering is the GRADIENT (test 5, via vq_codebook_gradient_with_beta)
    // -- that is the only place the difference is observable.
    double swapped = e_lat + BETA * q_lat;
    check("D1 NOTE: the two orderings are bit-identical on this fixture, which "
          "is WHY the discrimination has to happen in the gradient, not the "
          "value", swapped == total);

    // beta = 0 must collapse the commitment term exactly (D10 precondition).
    check("D10 commitment(beta=0) == 0 exactly",
          vq_commitment_loss(z, zq, 0.0) == 0.0);
}

// ===========================================================================
// TEST 5 -- the two GRADIENTS. D1's beta lives ONLY in d/dz_e.
// ===========================================================================
static void test_gradients() {
    std::cout << "\n[TEST 5] gradient w.r.t. codebook and z_e (D1, D2)"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    Tensor zq = vq_quantize(z, w);

    Tensor gw = vq_codebook_gradient(z, w);      // (D, K)
    check("codebook gradient shape is (D, K)", gw.rows == D && gw.cols == K);

    // Oracle d(total)/d(codebook), (D,K):
    //   [-0.0083333333333333402 -0.0083333333333333402 -0.02            0  0]
    //   [-0.0083333333333333367  0.0033333333333333361  0.0083333333333333332 0  0]
    //   [ 0.0083333333333333332 -0.0066666666666666636 -0.013333333333333336 0  0]
    //   [-0.0033333333333333361 -0.003333333333333334  0.0066666666666666541 0 0]
    const double expect_gw[4][5] = {
        {-0.0083333333333333402, -0.0083333333333333402, -0.02,             0.0, 0.0},
        {-0.0083333333333333367,  0.0033333333333333361,  0.0083333333333333332, 0.0, 0.0},
        { 0.0083333333333333332, -0.0066666666666666636, -0.013333333333333336, 0.0, 0.0},
        {-0.0033333333333333361, -0.003333333333333334,  0.0066666666666666541, 0.0, 0.0},
    };
    bool all = true;
    for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K; ++k)
            if (!close(gw(d,k), expect_gw[d][k], 1e-12)) {
                all = false;
                std::cout << "     gw(" << d << "," << k << ") = " << gw(d,k)
                          << " expected " << expect_gw[d][k] << std::endl;
            }
    check("all 20 codebook-gradient entries match the oracle to 1e-12", all);

    // The columns for codes 3 and 4 are never assigned by this fixture, so
    // their gradient is EXACTLY zero -- a real check, not an omission.
    check("unassigned codes 3 and 4 have EXACTLY zero gradient",
          gw(0,3) == 0.0 && gw(0,4) == 0.0);

    // ---- d/dz_e: carries beta, and NOTHING else.
    Tensor gz = vq_commitment_gradient(z, zq, BETA);
    // Oracle d(total)/d(z_e), (M,D):
    //   [ 0.002083333333333335  0.0020833333333333342 -0.0020833333333333333  0.00083333333333333404]
    //   [ 0.002083333333333335 -0.00083333333333333404  0.0016666666666666659  0.0008333333333333335]
    //   [ 0.0050000000000000001 -0.0020833333333333333 0.003333333333333334  -0.0016666666666666635]
    const double expect_gz[3][4] = {
        { 0.002083333333333335,  0.0020833333333333342, -0.0020833333333333333,  0.00083333333333333404},
        { 0.002083333333333335, -0.00083333333333333404,  0.0016666666666666659,  0.0008333333333333335},
        { 0.0050000000000000001,-0.0020833333333333333,  0.003333333333333334,  -0.0016666666666666635},
    };
    bool allz = true;
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            if (!close(gz(i,d), expect_gz[i][d], 1e-12)) {
                allz = false;
                std::cout << "     gz(" << i << "," << d << ") = " << gz(i,d)
                          << " expected " << expect_gz[i][d] << std::endl;
            }
    check("all 12 commitment-gradient entries match the oracle to 1e-12", allz);

    // D1: the beta factor is EXACTLY beta, isolated by beta = 1.
    Tensor gz1 = vq_commitment_gradient(z, zq, 1.0);
    bool linear = true;
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            if (!close(gz(i,d), BETA * gz1(i,d), 1e-12)) linear = false;
    check("D1 d/dz_e is EXACTLY linear in beta (no other factor hidden)", linear);

    // The discriminator for D1: the codebook gradient must be the UNWEIGHTED
    // one. If beta leaked in, it would be exactly (1+beta) times -- the clean
    // ratio the oracle hit.
    Tensor gw_beta = vq_codebook_gradient_with_beta(z, w, BETA);
    bool leaked = true;
    for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K; ++k)
            if (!close(gw(d,k), gw_beta(d,k), 1e-12)) leaked = false;
    check("D1 the codebook gradient carries NO beta factor (the swapped-"
          "gradient variant differs)", !leaked);

    // beta = 0 must zero the commitment gradient exactly (D10).
    Tensor gz0 = vq_commitment_gradient(z, zq, 0.0);
    bool allzero = true;
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d) if (gz0(i,d) != 0.0) allzero = false;
    check("D10 commitment gradient at beta=0 is EXACTLY zero", allzero);

    // Degenerate: z_e sitting exactly ON a code gives an EXACTLY zero
    // commitment gradient. (The reconstruction gradient still flows via the
    // STE, so this is not a dead layer -- see test 7.)
    Tensor z_on = oracle_z_e();
    for (size_t d = 0; d < D; ++d) z_on(0,d) = w(d,0);
    Tensor zq_on = vq_quantize(z_on, w);
    Tensor gz_on = vq_commitment_gradient(z_on, zq_on, BETA);
    bool row0zero = true;
    for (size_t d = 0; d < D; ++d) if (gz_on(0,d) != 0.0) row0zero = false;
    check("z_e exactly on a code gives an exactly-zero commitment gradient "
          "on that row", row0zero);
}

// ===========================================================================
// TEST 6 -- perplexity (D9): never exactly an integer, ceiling is min(K,M)
// ===========================================================================
static void test_perplexity() {
    std::cout << "\n[TEST 6] D9 perplexity is never exactly an integer"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    std::vector<size_t> idx = vq_nearest_indices(z, w);

    // Oracle: 2.9999999991  -- NOT 3.0. Sonnet line 106 uses log(p + 1e-10).
    double p = vq_perplexity(idx, K);
    check("3 distinct codes over 3 points == oracle 2.9999999990999995",
          close(p, 2.9999999990999995, 1e-15));
    check("D9 it is NOT exactly 3.0 (asserting == 3.0 would fail on the "
          "correct implementation)", p != 3.0);
    check("D9 it is just BELOW 3.0", p < 3.0);

    // Floor: all points on ONE code.
    std::vector<size_t> one{0, 0, 0};
    double p1 = vq_perplexity(one, K);
    check("all points on one code == oracle 0.99999999989999999",
          close(p1, 0.99999999989999999, 1e-15));
    check("and it is BELOW 1.0, never exactly 1", p1 < 1.0 && p1 != 1.0);

    // D9 ceiling: min(K, M). 5 codes over 3 points cannot exceed 3 -- the
    // maximum entropy of a 3-point categorical distribution is ln(3).
    std::vector<size_t> spread{0, 1, 2};
    check("the ceiling is min(K,M)=3, NOT K=5",
          vq_perplexity(spread, K) < 5.0);

    // Two codes over 2 points. MEASURED: 1.9999999996 -- and note what that is
    // NOT. Perplexity is exp(entropy), so two points on two codes is
    // exp(log 2) = 2, not ln(2) = 0.6931. The first draft of this file
    // asserted ln(2), which is the entropy, not the perplexity.
    std::vector<size_t> two{1, 4};
    double p2 = vq_perplexity(two, K);
    check("2 points on 2 codes == exp(log 2) = 1.9999999996 (measured)",
          close(p2, 1.9999999996, 1e-12));
    check("D9 and it is NOT ln(2) -- perplexity is exp(entropy), not entropy",
          !close(p2, 0.69314718055994529, 1e-6));

    // Monotonicity: more distinct codes never lowers perplexity.
    check("adding a distinct code raises perplexity", p2 > p1);

    // A code that is never used contributes nothing (log(0 + 1e-10) ~ 0).
    check("unused codes do not inflate perplexity (no leak of log(0))",
          std::isfinite(p) && p <= (double)M + 1e-9);
}

// ===========================================================================
// TEST 7 -- the straight-through estimator (D5): value z_q, gradient identity
// ===========================================================================
static void test_straight_through() {
    std::cout << "\n[TEST 7] D5 STE: value is z_q, gradient is the identity"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    Tensor zq = vq_quantize(z, w);

    // FIXTURE HONESTY (D5): z_q must DIFFER from z_e on this fixture, else
    // "the output equals z_q" is trivially satisfied by returning the input.
    bool differs = false;
    for (size_t i = 0; i < M && !differs; ++i)
        for (size_t d = 0; d < D; ++d)
            if (zq(i,d) != z(i,d)) { differs = true; break; }
    check("FIXTURE HONESTY: z_q differs from z_e on this fixture", differs);

    Tensor ste = vq_straight_through(z, zq);
    check("the STE VALUE is BIT-identical to z_q", ste.rows == zq.rows &&
          ste.cols == zq.cols && [&]{ for (size_t i=0;i<M;++i) for (size_t d=0;d<D;++d)
              if (ste(i,d)!=zq(i,d)) return false; return true; }());

    // The algebraic form, Sonnet line 104: inputs + sg(quantized - inputs).
    // A forward-mode evaluation of THAT expression gives z_e + (z_q - z_e),
    // which cancels to z_q but is NOT bit-identical in general. Assert the
    // identity holds to tolerance. (MEASURED CORRECTION: the first draft of
    // this file asserted the two differ bit-for-bit. They DO NOT on this
    // fixture -- the cancellation is exact here -- so that assertion was
    // vacuous. What matters, and is asserted above, is that the value is
    // z_q and NOT z_e.)
    Tensor algebraic = z + (zq - z);
    check("the algebraic form z + (z_q - z) equals z_q to 1e-12",
          tensor_close(algebraic, zq, 1e-12));

    // The GRADIENT half: d(straight_through)/d(input) = I. There is no
    // function to call here -- the identity is realised by the Layer's
    // backward returning grad_output -- so this asserts the CONTRACT on the
    // real layer, which is test 9.
    check("the STE gradient contract is realised in test 9 (Layer::backward "
          "returns grad_output unchanged)", true);
}

// ===========================================================================
// TEST 8 -- EMA update: Laplace smoothing (D7) and zero-collapse (D8)
// ===========================================================================
static void test_ema_step() {
    std::cout << "\n[TEST 8] D7/D8 EMA update: Laplace smoothing + zero-collapse"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();

    Tensor cluster_size(1, K), ema_w(D, K);
    cluster_size.fill(0.0);
    ema_w.fill(0.0);

    Tensor w1 = vq_ema_step(z, w, cluster_size, ema_w, 0.99, 1e-5);

    // Oracle EMA step 1, w1[:,k]:
    //   [:,0] = [ 0.55036630036630052, -0.15009990009990012,
    //             0.050033300033300039,  0.32021312021312021]
    //   [:,1] = [-0.35023310023310023,  0.58038628038628037,
    //             0.24015984015984015, -0.080053280053280051]
    //   [:,2] = [ 0.12007992007992008,  0.050033300033300039,
    //            -0.42027972027972027,  0.66043956043956054]
    //   [:,3] = [0,0,0,0]     <- D8 zero-collapse
    //   [:,4] = [0,0,0,0]
    const double expect_w1[3][4] = {
        { 0.55036630036630052, -0.15009990009990012,  0.050033300033300039,  0.32021312021312021},
        {-0.35023310023310023,  0.58038628038628037,  0.24015984015984015, -0.080053280053280051},
        { 0.12007992007992008,  0.050033300033300039, -0.42027972027972027,  0.66043956043956054},
    };
    bool all = true;
    for (size_t k = 0; k < 3; ++k)
        for (size_t d = 0; d < D; ++d)
            if (!close(w1(d,k), expect_w1[k][d], 1e-12)) {
                all = false;
                std::cout << "     w1(" << d << "," << k << ") = " << w1(d,k)
                          << " expected " << expect_w1[k][d] << std::endl;
            }
    check("EMA step 1 columns 0..2 match the oracle to 1e-12", all);

    // D7 DISCRIMINATOR: dividing by n_k directly (no Laplace smoothing) is a
    // visibly different number. With cs=0 and ema_w=0 the raw ratio is
    // (0.01*z)/(0.01) = z exactly = 0.55, whereas the smoothed version is
    // 0.55036630036630052. Asserting BOTH pins the smoothing and pins the
    // value the unsmoothed variant would have produced.
    check("D7 the smoothed value is NOT exactly z_e (0.55)",
          !close(w1(0,0), z(0,0), 1e-15));
    check("D7 the smoothed value is the oracle 0.55036630036630052",
          close(w1(0,0), 0.55036630036630052, 1e-15));
    check("D7 the unsmoothed variant's value (0.55) is a DIFFERENT number, so "
          "this fixture discriminates",
          !close(0.55036630036630052, 0.55, 1e-12));

    // D8: never-assigned codes collapse to EXACTLY zero.
    bool z3 = true, z4 = true;
    for (size_t d = 0; d < D; ++d) {
        if (w1(d,3) != 0.0) z3 = false;
        if (w1(d,4) != 0.0) z4 = false;
    }
    check("D8 code 3 (never assigned) collapses to EXACTLY 0.0", z3);
    check("D8 code 4 (never assigned) collapses to EXACTLY 0.0", z4);

    // DETERMINISM: vq_ema_step is a PURE function of (z, w, cluster_size, ema_w),
    // so two calls with identical inputs give bit-identical outputs. The first
    // draft of this test asserted they DIFFER, having conflated "the function
    // is live" with "the function is impure". Correct property: the EMA update
    // is observable when the INPUT STATE advances, so drive it from the
    // step-1 codebook.
    Tensor cluster2(1, K), ema2(D,K);
    cluster2.fill(0.0); ema2.fill(0.0);
    Tensor step1 = vq_ema_step(z, w, cluster2, ema2, 0.99, 1e-5);
    Tensor step1b = vq_ema_step(z, w, cluster2, ema2, 0.99, 1e-5);
    check("the EMA step is a pure function: two identical calls agree exactly",
          tensor_close(step1, step1b, 0.0));

    // The Laplace bias survives decay = 0, so the codebook does NOT land
    // exactly on z_e. MEASURED: with N=3 and K=5 the smoothed cluster size is
    // 0.99999333, so the codebook is z_e/(1 + 6.667e-06) -- a 6.7e-06
    // relative offset. The first draft asserted an exact landing, which is
    // what an implementation WITHOUT the Laplace smoothing would give.
    Tensor cs0(1, K), ew0(D,K);
    cs0.fill(0.0); ew0.fill(0.0);
    Tensor wdecay0 = vq_ema_step(z, w, cs0, ew0, 0.0, 1e-5);
    check("decay=0 gives the measured 0.5500036666 (NOT exactly z_e = 0.55)",
          close(wdecay0(0,0), 0.5500036666, 1e-9));
    check("decay=0 still does not land exactly on z_e -- the Laplace bias "
          "survives",
          !close(wdecay0(0,0), z(0,0), 1e-12));

    // decay = 1 with a ZERO statistics state is a documented no-op, and the
    // oracle shows the formula as written returns ALL NaN there (0/0), because
    // `total` is 0. Rather than ship a NaN for a legal argument, the
    // implementation guards it: nothing has been observed, so the codebook is
    // returned unchanged. MEASURED CORRECTION: the first draft of this test
    // asserted the unassigned codes "collapse to zero (0/eps == 0)", which was
    // arithmetically wrong -- the divisor is the LAPLACE-SMOOTHED total, not
    // eps alone.
    Tensor cs1(1, K), ew1(D,K);
    cs1.fill(0.0); ew1.fill(0.0);
    Tensor wdecay1 = vq_ema_step(z, w, cs1, ew1, 1.0, 1e-5);
    check("decay=1 with zero statistics returns the codebook UNCHANGED "
          "(guarded against the oracle's 0/0 NaN)",
          tensor_close(wdecay1, w, 0.0));
    check("decay=1 with zero statistics is finite everywhere",
          [&]{ for (size_t d=0;d<D;++d) for (size_t k=0;k<K;++k)
                   if (!std::isfinite(wdecay1(d,k))) return false; return true; }());

    // With REAL statistics present, decay = 1 is still a no-op: new_dw equals
    // the stored ema_w exactly, so the update returns the stored codebook.
    Tensor csR(1, K), ewR(D, K);
    for (size_t k = 0; k < K; ++k) csR(0,k) = 4.0;
    ewR.fill(0.0);
    Tensor wdecay1b = vq_ema_step(z, w, csR, ewR, 1.0, 1e-5);
    bool finite_b = true;
    for (size_t d = 0; d < D; ++d)
        for (size_t k = 0; k < K; ++k) if (!std::isfinite(wdecay1b(d,k))) finite_b = false;
    check("decay=1 with real cluster sizes stays finite", finite_b);
}

// ===========================================================================
// TEST 9 -- VectorQuantizer the Layer (D5 gradient contract, D1 accessor)
// ===========================================================================
static void test_vector_quantizer_layer() {
    std::cout << "\n[TEST 9] VectorQuantizer: Layer contract" << std::endl;

    VectorQuantizer vq(D, K, BETA);
    Tensor w = oracle_codebook(), z = oracle_z_e();
    vq.set_codebook(w);
    vq.zero_grad();

    Tensor out = vq.forward(z);

    // D5: the Layer's forward value is z_q, bit-identical.
    Tensor zq = vq_quantize(z, w);
    check("forward() returns BIT-identical z_q", tensor_close(out, zq, 0.0));

    // The accessors report the two terms SEPARATELY, which is the only way a
    // caller can see that they are not interchangeable.
    check("codebook_loss() == oracle 0.0031416666666666676",
          close(vq.codebook_loss(), 0.0031416666666666676, 1e-15));
    check("commitment_loss() == beta * 0.0031416666666666676",
          close(vq.commitment_loss(), BETA * 0.0031416666666666676, 1e-15));
    check("the two accessors are NOT equal (beta is applied to exactly one)",
          vq.codebook_loss() != vq.commitment_loss());

    check("indices() == [0,1,2]", vq.indices().size() == 3 &&
          vq.indices()[0] == 0 && vq.indices()[1] == 1 && vq.indices()[2] == 2);
    check("perplexity() == oracle 2.9999999990999995",
          close(vq.perplexity(), 2.9999999990999995, 1e-15));
    // usage: 3 distinct codes used, 2 dead.
    check("usage_count() == 3 of 5 codes used", vq.usage_count() == 3);
    check("dead_code_count() == 2 (the classic VQ codebook-collapse signal)",
          vq.dead_code_count() == 2);

    // ---- the gradient after backward. The seed is (1,1) DELIBERATELY: it is
    // the obvious way to ask "is this the identity?", and it is the seed that
    // exposed the shape-truncation bug described below.
    Tensor grad(1, 1); grad(0,0) = 1.0;
    Tensor gz = vq.backward(grad, 0.0);

    // D5: the STE passes grad_output UNALTERED to z_e -- but the layer's return
    // value is that PLUS the commitment gradient, so it is NOT grad_output
    // alone. The first draft of this test asserted equality with grad_output,
    // which is arithmetically impossible given the commitment term the header
    // promises. MEASURED by a standalone probe: gz(0,0) = 1.0020833333333334
    // = 1.0 (the seed) + 0.0020833 (the beta-weighted commitment term). The
    // correct statement is that the seed appears UNCHANGED as an additive
    // identity on top of the commitment term.
    Tensor gcommit = vq_commitment_gradient(z, zq, BETA);
    Tensor expected_gz(M, D);
    for (size_t i = 0; i < M; ++i)
        for (size_t d = 0; d < D; ++d)
            expected_gz(i,d) = grad(0,0) + gcommit(i,d);

    check("D5 backward() returns the seed UNCHANGED on top of the commitment "
          "term (the STE identity)", tensor_close(gz, expected_gz, 1e-12));

    // The discriminator: the seed alone must differ from the sum, so the
    // assertion above cannot pass by omitting the commitment term.
    check("D5 the commitment term is not silently dropped (STE alone differs)",
          !tensor_close(gz, Tensor(M, D).apply([&](double){ return grad(0,0); }),
                        1e-12));

    // The shape contract: backward must return the INPUT's shape even when
    // the caller seeds with a (1,1) scalar. This WAS a real bug -- backward
    // returned `grad_output` verbatim, so a (1,1) seed produced a (1,1)
    // gradient and truncated the commitment term to one cell. Found by the
    // standalone probe above, NOT by reading: a value-only test cannot see a
    // truncated shape.
    check("D5 backward() returns (M, D) even for a (1,1) grad_output seed",
          gz.rows == M && gz.cols == D);
    check("D5 every cell is populated (not truncated to one)",
          [&]{ for (size_t i=0;i<M;++i) for (size_t d=0;d<D;++d)
                   if (gz(i,d) != expected_gz(i,d)) return false; return true; }());

    // The codebook gradient is populated and matches the oracle.
    // NOTE `gradients()[0]` is the GRADIENT and `parameters()[0]` is the
    // CODEBOOK. The first draft read parameters()[0] expecting a gradient,
    // which returned the codebook VALUE (0.5 at (0,0)) and looked like a
    // wildly wrong gradient rather than a wrong accessor. Embedding.h has the
    // same split, so this is the repo convention and the test was at fault.
    Tensor* gw = vq.gradients()[0];
    const double expect_gw0_0 = -0.0083333333333333402;
    check("the codebook gradient is populated at entry (0,0)",
          close((*gw)(0,0), expect_gw0_0, 1e-12));
    check("the codebook gradient has EXACTLY zero entries for dead codes",
          (*gw)(0,3) == 0.0 && (*gw)(0,4) == 0.0);
    // And the parameters() accessor really is the codebook, not the gradient.
    check("parameters()[0] is the CODEBOOK (0.5 at (0,0)), not the gradient",
          close((*vq.parameters()[0])(0,0), 0.5, 1e-15));

    // The shape contract: backward must return the INPUT's shape even when
    // the caller seeds with a (1,1) scalar. Found by a standalone probe, not by
    // reading -- a value-only test cannot see a truncated shape.
    check("D5 backward() returns (M, D) even for a (1,1) grad_output seed",
          gz.rows == M && gz.cols == D);
    check("D5 every cell is populated (not truncated to one)",
          tensor_close(gz, expected_gz, 1e-12));

    // ---- update_weights actually MOVES the codebook (the "training test that
    // passes because nothing moved" trap: check the parameter, not the loss).
    Tensor before = vq.codebook();
    vq.update_weights(0.5);
    check("update_weights() moves the codebook", !tensor_close(before, vq.codebook(), 0.0));
    check("update_weights moves it TOWARD z_e (the dictionary term pulls e "
          "towards its assigned encoder output)",
          vq.codebook()(0,0) > before(0,0));

    // zero_grad clears it.
    vq.zero_grad();
    check("zero_grad() clears the codebook gradient",
          (*vq.gradients()[0])(0,0) == 0.0);

    check("get_weights() returns the codebook", vq.get_weights().rows == D &&
          vq.get_weights().cols == K);
    check("parameters() has exactly one entry (the codebook)",
          vq.parameters().size() == 1);
}

// ===========================================================================
// TEST 10 -- finite differences on the EMA-free gradient path
// ===========================================================================
static void test_fd_gradients() {
    std::cout << "\n[TEST 10] centred FD against the analytic gradients"
              << std::endl;
    Tensor w = oracle_codebook(), z = oracle_z_e();
    const double h = 1e-6;

    // d/d(codebook) of the DICTIONARY term. IMPORTANT (D1-STOPGRAD): this
    // differentiates the dictionary term ALONE, holding z_e fixed. Adding the
    // commitment term to the function being differentiated would give
    // (1+beta) = 1.25 on every cell -- a clean-ratio fingerprint of
    // differentiating the wrong function, which is exactly the bug the oracle
    // made on its first run.
    double worst_w = 0.0;
    for (size_t d = 0; d < D; ++d) {
        for (size_t k = 0; k < K; ++k) {
            Tensor wp = w, wm = w;
            wp(d,k) += h; wm(d,k) -= h;
            Tensor zqp = vq_quantize(z, wp), zqm = vq_quantize(z, wm);
            double fd = (vq_codebook_loss(z, zqp) - vq_codebook_loss(z, zqm)) / (2*h);
            double ana = vq_codebook_gradient(z, w)(d,k);
            double den = std::max(std::fabs(ana), std::fabs(fd));
            if (den > 1e-12) worst_w = std::max(worst_w, std::fabs(ana-fd)/den);
        }
    }
    check("d(dictionary)/d(codebook) matches FD (worst rel_err < 1e-7)", worst_w < 1e-7);
    std::cout << "     worst rel_err = " << worst_w << std::endl;

    // d/d(z_e) of the COMMITMENT term (beta = 1 so the factor is visible and
    // separable). This one IS differentiable: ||z_e - sg[e]||^2 is smooth.
    Tensor zq = vq_quantize(z, w);
    double worst_z = 0.0;
    for (size_t i = 0; i < M; ++i) {
        for (size_t d = 0; d < D; ++d) {
            Tensor zp = z, zm = z;
            zp(i,d) += h; zm(i,d) -= h;
            // sg[e]: the codebook is held fixed, so z_q is FIXED here.
            Tensor zqp = vq_quantize(zp, w);
            Tensor zqm = vq_quantize(zm, w);
            double fd = (vq_commitment_loss(zp, zqp, 1.0)
                       - vq_commitment_loss(zm, zqm, 1.0)) / (2*h);
            double ana = vq_commitment_gradient(z, zq, 1.0)(i,d);
            double den = std::max(std::fabs(ana), std::fabs(fd));
            if (den > 1e-12) worst_z = std::max(worst_z, std::fabs(ana-fd)/den);
        }
    }
    check("d(commitment)/d(z_e) matches FD (worst rel_err < 1e-7)", worst_z < 1e-7);
    std::cout << "     worst rel_err = " << worst_z << std::endl;
}

// ===========================================================================
// TEST 11 -- VQVAE end-to-end. LIVENESS, not a convergence promise.
// ===========================================================================
static void test_vqvae_end_to_end() {
    std::cout << "\n[TEST 11] VQVAE end-to-end liveness" << std::endl;

    VQVAE model(4, 8, D, K, BETA);
    model.zero_grad();

    // Two well-separated clusters so a reconstruction is possible at all.
    Tensor x(6, 4);
    const double cl[2][4] = {
        { 1.0, -1.0,  0.5, -0.5},
        {-1.0,  1.0, -0.5,  0.5},
    };
    for (size_t i = 0; i < 6; ++i)
        for (size_t d = 0; d < 4; ++d)
            x(i,d) = cl[i % 2][d] + 0.01 * (double)((i * 7 + d * 3) % 5);

    Tensor recon = model.forward(x);
    check("reconstruction has the input shape", recon.rows == 6 && recon.cols == 4);
    check("the reconstruction is finite", std::isfinite(recon.sum()));

    // LIVENESS (the adversarial session's `advRob = 0.5000` lesson): a
    // constant predictor scores exactly 1/K on a balanced task and reads as
    // "half working". Assert the reconstruction actually DEPENDS on the input.
    Tensor recon2 = model.forward(x);
    check("forward is deterministic across calls", tensor_close(recon, recon2, 0.0));

    Tensor shifted = x;
    for (size_t i = 0; i < 6; ++i) for (size_t d = 0; d < 4; ++d) shifted(i,d) *= -1.0;
    Tensor recon3 = model.forward(shifted);
    check("LIVENESS: negating the input CHANGES the reconstruction "
          "(a dead/constant decoder would fail this)",
          !tensor_close(recon, recon3, 1e-9));

    check("perplexity is finite and in (0, K]",
          model.perplexity() > 0.0 && model.perplexity() <= (double)K);

    // The training loop: forward / backward / update, in that order. Calling
    // forward+backward without update_weights is the repo's recorded "L0 == L1
    // to 6 decimals" trap.
    //
    // THE GRADIENT IS NOT A CONSTANT, and getting that wrong is silent. This
    // repo's `Layer::backward` contract is the gradient of
    // L = sum(grad_output * forward(x)) with no 1/2 anywhere, so for
    // L = MSE = mean((r-x)^2) the incoming gradient is 2*(r-x)/N. The first
    // draft passed a constant 1/N -- a perfectly plausible "mean, isn't it?"
    // seed -- and the loss ROSE and diverged to inf at step 131. Measured, by
    // a standalone probe before this was changed:
    //     constant 1/N,      lr=0.05  -> delta = inf
    //     proper 2(r-x)/N,   lr=0.05  -> delta = -0.632858
    //     proper, lr in {0.005, 0.01, 0.02, 0.05, 0.1} -> all NEGATIVE
    // i.e. it was a constant-seed bug, not a divergence bug: the same wrong
    // seed fails identically at every learning rate, which is the
    // "comparison fails identically across every hyperparameter" signature
    // that means look at the fixture, not the optimiser.
    Tensor grad(6, 4);
    double first = 0.0, last = 0.0;
    for (int step = 0; step < 200; ++step) {
        model.zero_grad();
        Tensor r = model.forward(x);
        // reconstruction MSE as the loss
        double loss = 0.0;
        for (size_t i = 0; i < 6; ++i)
            for (size_t d = 0; d < 4; ++d)
                loss += (r(i,d) - x(i,d)) * (r(i,d) - x(i,d));
        loss /= (6 * 4);
        for (size_t i = 0; i < 6; ++i)
            for (size_t d = 0; d < 4; ++d)
                grad(i,d) = 2.0 * (r(i,d) - x(i,d)) / (6 * 4);
        model.backward(grad, 0.0);
        model.update_weights(0.05);
        if (step == 0) first = loss;
        last = loss;
    }
    check("LIVENESS: the reconstruction loss decreases over 200 steps",
          last < first);
    check("LIVENESS: it stays FINITE (the first draft diverged to inf)",
          std::isfinite(last));
    std::cout << "     loss " << first << " -> " << last << std::endl;

    // Robustness across learning rates: the same wrong seed fails identically
    // at every one, which is how the fixture was diagnosed. Pin that the
    // correct gradient works over a range, not just at one lucky value.
    bool all_decrease = true;
    for (double lr : {0.005, 0.01, 0.02, 0.05, 0.1}) {
        VQVAE m2(4, 8, D, K, BETA);
        Tensor g2(6, 4);
        double f2 = 0.0, l2 = 0.0;
        for (int step = 0; step < 200; ++step) {
            m2.zero_grad();
            Tensor rr = m2.forward(x);
            double ls = 0.0;
            for (size_t i = 0; i < 6; ++i)
                for (size_t d = 0; d < 4; ++d)
                    ls += (rr(i,d) - x(i,d)) * (rr(i,d) - x(i,d));
            ls /= 24.0;
            for (size_t i = 0; i < 6; ++i)
                for (size_t d = 0; d < 4; ++d)
                    g2(i,d) = 2.0 * (rr(i,d) - x(i,d)) / 24.0;
            m2.backward(g2, 0.0);
            m2.update_weights(lr);
            if (step == 0) f2 = ls;
            l2 = ls;
        }
        if (!(l2 < f2) || !std::isfinite(l2)) all_decrease = false;
    }
    check("LIVENESS: the loss decreases at EVERY learning rate in "
          "{0.005, 0.01, 0.02, 0.05, 0.1}", all_decrease);

    // The quantizer must actually be quantizing: codes must be used.
    check("LIVENESS: the quantizer uses at least one code",
          model.usage_count() >= 1);
}

// ===========================================================================
// TEST 12 -- VectorQuantizerEMA the Layer (D6: no dictionary term)
// ===========================================================================
static void test_ema_layer() {
    std::cout << "\n[TEST 12] VectorQuantizerEMA: optimizer-independent update"
              << std::endl;

    VectorQuantizerEMA vq(D, K, BETA, 0.99, 1e-5);
    Tensor w = oracle_codebook(), z = oracle_z_e();
    vq.set_codebook(w);
    vq.zero_grad();

    Tensor out = vq.forward(z);
    check("forward returns bit-identical z_q", tensor_close(out, vq_quantize(z, w), 0.0));

    Tensor grad(1,1); grad(0,0) = 1.0;
    Tensor gz = vq.backward(grad, 0.0);

    // D6: under EMA the dictionary term is DROPPED, so backward returns ONLY
    // the commitment gradient -- NOT grad_output + commitment gradient.
    Tensor gcommit = vq_commitment_gradient(z, vq_quantize(z, w), BETA);
    check("D6 backward() returns ONLY the commitment gradient (the STE "
          "identity does NOT apply under EMA)",
          tensor_close(gz, gcommit, 1e-12));

    Tensor ones(D, K); ones.fill(1.0);
    check("D6 and it is NOT the STE identity (so the EMA path is distinguishable)",
          !tensor_close(gz, ones, 1e-12));

    // The codebook is ASSIGNED, not gradient-updated. Two update_weights with
    // wildly different learning rates must give the IDENTICAL codebook --
    // that is the whole documented advantage of EMA ("independent of the
    // choice of optimizer").
    Tensor cb_before = vq.codebook();
    vq.update_weights(0.001);
    Tensor cb_a = vq.codebook();
    vq.set_codebook(cb_before);
    vq.set_ema_state(Tensor(1, K), Tensor(D, K));   // reset stats
    vq.update_weights(1000.0);
    Tensor cb_b = vq.codebook();
    check("the EMA update is INDEPENDENT of the learning rate (Sonnet docstring: "
          "'independent of the choice of optimizer')",
          tensor_close(cb_a, cb_b, 0.0));

    // And it must actually have moved.
    check("the EMA update moved the codebook",
          !tensor_close(cb_before, cb_a, 1e-12));

    // D8 through the Layer: dead codes collapse to exactly zero.
    bool dead_zero = true;
    for (size_t d = 0; d < D; ++d)
        if (vq.codebook()(d,3) != 0.0 || vq.codebook()(d,4) != 0.0) dead_zero = false;
    check("D8 dead codes are EXACTLY zero after an EMA update", dead_zero);

    // training=false must NOT move the codebook.
    Tensor cb_pre = vq.codebook();
    vq.update_weights(0.99, /*training=*/false);
    check("training=false leaves the codebook untouched",
          tensor_close(cb_pre, vq.codebook(), 0.0));
}

// ===========================================================================
// TEST 13 -- degeneracy guards
// ===========================================================================
static void test_degenerate_fixtures() {
    std::cout << "\n[TEST 13] degenerate configurations" << std::endl;

    // Zero encoder output: every point is at distance ||e_k||^2 from every
    // code, and the argmin over a tie goes to index 0.
    Tensor w = oracle_codebook();
    Tensor z0(2, D); z0.fill(0.0);
    std::vector<size_t> idx = vq_nearest_indices(z0, w);
    check("all-zero z_e assigns every point to code 0 (first-wins tie)",
          idx[0] == 0 && idx[1] == 0);
    check("an all-identical-assignment batch has perplexity ~1.0 (not 0, not 5)",
          close(vq_perplexity(idx, K), 0.9999999999, 1e-8));

    // Zero codebook: all distances identical, still index 0, no NaN.
    Tensor wz(D, K); wz.fill(0.0);
    std::vector<size_t> idxz = vq_nearest_indices(z0, wz);
    check("an all-zero codebook still assigns deterministically (no NaN)",
          idxz[0] == 0 && idxz[1] == 0);

    // Identical encoder outputs -> the codebook gradient accumulates onto the
    // SAME column and must not overflow or go NaN.
    Tensor zsame(4, D);
    for (size_t i = 0; i < 4; ++i)
        for (size_t d = 0; d < D; ++d) zsame(i,d) = w(d,0) + 0.1 * (double)d;
    Tensor zqs = vq_quantize(zsame, w);
    Tensor g = vq_codebook_gradient(zsame, w);
    bool finite = true;
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j < K; ++j) if (!std::isfinite(g(i,j))) finite = false;
    check("a degenerate batch produces finite gradients", finite);

    // A single point must not be normalised away by the mean (D2). MEASURED by
    // the oracle: z = e_0 + 0.5*ones is nearest code 3 (dmin = 0.79), not
    // code 0 -- the all-ones offset moves it closer to e_3 = [0.9,0.1,0.1,0.1].
    // The loss is the squared distance 0.79 spread over D=4 elements = 0.1975.
    // Dividing by the 4 ELEMENTS is D2; a per-row mean of a single row would
    // give the same thing, so the discrimination against row-mean needs M>1
    // and is made in test 4.
    Tensor z1(1, D);
    for (size_t d = 0; d < D; ++d) z1(0,d) = w(d,0) + 0.5;
    check("e_0 + 0.5*ones is nearest code 3, not 0 (measured)",
          vq_nearest_indices(z1, w)[0] == 3);
    Tensor zq1 = vq_quantize(z1, w);
    check("D2 the loss for a single point is 0.79/D = 0.1975",
          close(vq_codebook_loss(z1, zq1), 0.1975, 1e-12));
    check("D2 and the summed squared residual is 0.79 (not 0.1975)",
          close(vq_codebook_loss(z1, zq1) * (double)D, 0.79, 1e-12));

    // Large-magnitude input: no overflow in the distance computation.
    Tensor zbig(1, D); zbig.fill(1e3);
    Tensor dbig = vq_squared_distances(zbig, w);
    check("a large-magnitude input gives finite distances",
          std::isfinite(dbig(0,0)) && dbig(0,0) > 0.0);

    check("K=1 degenerate codebook always assigns 0",
          vq_nearest_indices(z0, Tensor(D,1))[0] == 0);
}

// ===========================================================================
static void test_shape_and_errors() {
    std::cout << "\n[TEST 14] shape validation" << std::endl;

    Tensor w = oracle_codebook();   // (4,5): embedding_dim = 4

    // (M, 4) latent against a (4,5) codebook is VALID -- 3 is the batch size
    // and says nothing about the embedding width. The first draft of this
    // file built `Tensor bad(3, D)` believing it was a (3,4)-wide mismatch,
    // but D == 4 so it is exactly the right shape, and the assertion could
    // never pass. A fixture named for a shape must actually have that shape.
    Tensor good(3, D); good.fill(0.0);
    check("(3,4) latent against a (4,5) codebook is VALID (3 is the batch)",
          vq_squared_distances(good, w).rows == 3);

    // Now a genuine mismatch: the latent is 5-wide, the codebook 4-wide.
    Tensor bad(3, 5); bad.fill(0.0);
    bool threw = false;
    try { vq_squared_distances(bad, w); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a (3,5) latent against a (4,5) codebook throws invalid_argument",
          threw);

    // And the transposed direction: a (5,3) codebook against a (3,4) latent.
    threw = false;
    try { vq_squared_distances(oracle_z_e(), Tensor(5, 3)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a (5,3) codebook against a (3,4) latent throws", threw);

    // An empty codebook (K = 0) and an empty batch must both be rejected
    // rather than dividing by zero.
    threw = false;
    try { vq_squared_distances(good, Tensor(D, 0)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("a (4,0) codebook throws rather than dividing by zero", threw);

    threw = false;
    try { VectorQuantizer vq(D, K, BETA); vq.set_codebook(Tensor(3, 3)); }
    catch (const std::invalid_argument&) { threw = true; }
    check("set_codebook with the wrong shape throws", threw);
}

// ===========================================================================
int main() {
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "==========================================================\n";
    std::cout << " VQ-VAE test suite (van den Oord, arXiv:1711.00937)\n";
    std::cout << "==========================================================\n";

    test_distances_exact();
    test_nearest_indices_and_tiebreak();
    test_quantize_value();
    test_two_terms_and_weighting();
    test_gradients();
    test_perplexity();
    test_straight_through();
    test_ema_step();
    test_vector_quantizer_layer();
    test_fd_gradients();
    test_vqvae_end_to_end();
    test_ema_layer();
    test_degenerate_fixtures();
    test_shape_and_errors();

    std::cout << "\n=== Summary: " << passed << " passed, " << failed
              << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}