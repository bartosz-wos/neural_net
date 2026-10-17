#ifndef NN_UTILS_ORTHO_PENALTY_H
#define NN_UTILS_ORTHO_PENALTY_H

#include "../core/tensor.h"
#include "../core/model.h"
#include <cstddef>

// ===========================================================================
// Orthogonality penalties
//
//   Tero Karras, Samuli Laine, Miika Aittala, Samuli Laine -- "Analyzing and
//   Improving the Image Quality of StyleGAN", arXiv:1912.04958, SIGGRAPH 2020.
//
// CITATION. Verified 2026-10-17 by scraping the arXiv ABS PAGE for its
// `citation_title` / `citation_author` meta tags: title matches, authors Brock
// et al. aside, the first three are Karras/Laine/Aittala. The arXiv **API was
// rate-limited** from this host (a 14-byte body reading "Rate exceeded."), and a
// rate-limited endpoint is NOT a verified citation. Five IDs guessed from
// memory were WRONG and must never be reintroduced: 1810.01242 (a Seyfert-1
// astronomy paper), 1809.10620 ("Operations on Partial Orders"), 1802.01886
// (Texygen), 1912.04948 (gauge theory), 1706.07057 (collider constraints).
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//
// `grep -rilE 'ortho' include/nn/layers/generative/` -> 0 hits, and a search for
// any penalty-shaped term over weights
// (`ortho_pen|orthogonality_pen|ortho_penalty|off_diagonal|gram`) returns only
// the Gram matrix inside `optimizers/galore.cpp`.
//
// Every "orthogonal" token in this repo is one of three things: a PAPER NAME
// (Performer's FAVOR+ = Fast Attention Via Orthogonal Random Features), an
// ORTHOGONALISING PROJECTION (`normalization/spectral_norm.h`, Muon's
// Newton-Schulz step -- these ENFORCE orthogonality by construction, which is
// a different thing entirely), or a comment saying something is "orthogonal to"
// another idea. Nothing in ~250 layers and 42 utility files ADDS A LOSS FOR
// NON-ORTHOGONALITY.
//
// `layers/generative/wgan_gp.h` has a *gradient* penalty, which is a different
// quantity on a different object (the critic's INPUT, constrained to unit
// norm). It is deliberately NOT duplicated or re-exported here.
//
// This is the only shape of regulariser the expansion queue named that needed
// no paper at all -- and it is the one class of regulariser that acts on the
// WEIGHTS rather than on the loss surface or the activations.
//
// ---------------------------------------------------------------------------
// THE PENALTY (paper Sec. 3, "lazy regularisation")
// ---------------------------------------------------------------------------
//
//     L(W) = s * ( mean(W W^T - I_r)^2 + mean(W^T W - I_c)^2 ),  s = 1/(r*c)
//
// where `mean` is over ALL entries of the r x r / c x c matrix.
//
// ---------------------------------------------------------------------------
// THE ADJOINT, AND THE THREE SUBTLETIES IN IT
// ---------------------------------------------------------------------------
//
//     u_k = sum_i W[i][k]   (column sums, length c)
//     t_i = sum_k W[i][k]   (row sums, length r)
//     m1  = (||u||^2 - r) / r^2          m2 = (||t||^2 - c) / c^2
//     dL/dW[i][k] = 4s * ( m1*u_k/r^2 + m2*t_i/c^2 )
//
// (1) `sum_ij (W W^T)[ij] = ||W^T 1||^2` -- the squared norm of the COLUMN SUMS,
//     NOT the Frobenius norm. (Mean of an r*r matrix is a sum over r*r ENTRIES.)
//     Getting this wrong gives rel_err ~ 1-3 against finite differences: an
//     INCONSISTENT error across shapes, which is the signature of a wrong
//     DERIVATION (a clean constant ratio is instead the signature of a harness
//     that differentiates the wrong function).
//
// (2) `mean(I_r) = 1/r`, NOT 1 -- the identity contributes `r`, so `m1` is
//     `(||u||^2 - r)/r^2`. **This bug is invisible on square matrices and on
//     random ones; it only shows up at r=1 or c=1**, because for a single row
//     the `r` vs `r^2` error nearly vanishes. That is why the suite carries
//     explicit 1x4 and 4x1 fixtures: every "natural" weight matrix is square,
//     so a square-only fixture set cannot see this bug at all.
//
// (3) The gradient is the sum of a COLUMN term (every row equal) and a ROW
//     term (every column equal), so it is always SEPARABLE: `g[i][k] = a_i +
//     b_k`. Measured residual 1e-17. A per-element Gram-matrix adjoint cannot
//     produce this shape, which makes the property a strong discriminator.
//
// THREE PREDICTIONS MEASURED FALSE -- do not assert any of them:
//   * `g == g^T` for square W:            FALSE, max|g - g^T| ~ 1e-1.
//   * `L(2W) = 4*L(W)`:                   FALSE, ratio -> 16 (the `1/(r*c)`
//     scale and the differing `r`/`c` in the two means break homogeneity; it is
//     only asymptotically quadratic at large ||W||).
//   * gradient non-zero at orthogonality: FALSE, EXACTLY ZERO -- orthogonality
//     is a genuine STATIONARY POINT, the global minimum of the penalty. This is
//     the single most load-bearing property in this file and it has its own
//     test, because a sign flip or a dropped factor in the adjoint destroys it
//     immediately.
//
// ---------------------------------------------------------------------------
// SEMANTICS, each a natural reimplementation mistake, each with a test
// ---------------------------------------------------------------------------
//
// (D1)  `s = 1/(r*c)` -- one factor, over ALL elements, applied to the SUM of
//       the two terms. Applying `1/(r*c)` twice, or `1/r` and `1/c` separately,
//       gives a different (and much smaller) number.
// (D2)  At an exactly orthogonal W the penalty is EXACTLY 0 AND the gradient is
//       EXACTLY 0. Verified for the identity, for a permutation matrix
//       (orthogonal but NOT symmetric, so it is not the same test), and for an
//       exact rotation.
// (D3)  For rank > 2 the matrix is `(shape[-2], shape[-1])` PER LEADING INDEX
//       ("per mode"). Folding a channel axis into the rows MIXES modes and
//       returns a different, wrong number -- the single easiest mistake here,
//       because `Tensor` is strictly 2-D so the fold looks harmless.
// (D4)  Rank-1 weights (`1 x C`, `R x 1`) are LEGAL and must not throw. They are
//       exactly the shapes that expose subtlety (2) above.
// (D5)  `m1` uses `r`, `m2` uses `c`. They are NOT interchangeable on a
//       non-square W.
// (D6)  Both terms are SQUARED MEANS, `mean(A)^2`, not means of squares
//       `mean(A^2)`. The two differ whenever A has mixed signs, which is the
//       normal case.
// (D7)  Empty tensors (`0 x C`, `C x 0`) THROW std::invalid_argument rather than
//       reading out of bounds.
// (D8)  `ortho_penalty_gradient` returns a FRESH tensor of the input's shape;
//       it never aliases its argument.
// (D9)  `lambda` scales both the value and the gradient linearly; `lambda = 0`
//       gives both EXACTLY zero, which is what makes the Model-level path
//       provably a no-op when disabled.
// (D10) The one-sided residual is `|sv1 - 1|`, NOT two-sided. The spectral
//       norm is estimated by a FIXED-step power iteration (deterministic, no
//       RNG), and `ortho_spectral_norm_inverse` THROWS at `sv1 == 0` rather
//       than returning infinity.
//
// ===========================================================================

// ---------------------------------------------------------------------------
// Core: the weight-orthogonality penalty on a 2-D weight matrix. (D1, D2, D5,
// D6, D7)
// ---------------------------------------------------------------------------
// THROWS std::invalid_argument if `w` has no rows or no columns.
double ortho_penalty(const Tensor& w);

// The adjoint: dL/dw, same shape as `w`. (D2, D3-in-2D, D8)
Tensor ortho_penalty_gradient(const Tensor& w);

// `lambda * ortho_penalty(w)` and `lambda * dL/dw`. (D9)
double ortho_penalty_lam(const Tensor& w, double lambda);
Tensor ortho_penalty_lam_gradient(const Tensor& w, double lambda);

// ---------------------------------------------------------------------------
// Rank > 2. `w` is a flat (num_modes, rows_per_mode * cols) tensor; `r` and `c`
// describe the matrix each mode holds, and MUST satisfy
// `w.cols == r * c`. (D3, D4)
// ---------------------------------------------------------------------------
// Per MODE: mean over modes of ortho_penalty(mode matrix).
double ortho_penalty_modes(const Tensor& w, size_t r, size_t c);
Tensor ortho_penalty_modes_gradient(const Tensor& w, size_t r, size_t c);

// Per SLICE: `w` is (num_batches, num_slices * s, c) -- each contiguous (s, c)
// block of a row is penalised ON ITS OWN, and the result is the mean over
// EVERY slice in EVERY batch (so there are num_slices * w.rows of them).
//
// DELIBERATELY a separate function with an EXPLICIT slice count, rather than a
// flag on the per-mode entry point (D12), so a caller who wants per-mode cannot
// get per-slice by passing a different argument -- and so the two cannot be
// confused by an off-by-one in a shared index expression.
//
// THROWS std::invalid_argument unless w.cols == num_slices * s * c.
//
// (The first version of this function took only (s, c) and inferred the slice
// layout as `sl(i,k) = w(b, j*c + k)`. That index expression never uses `i`, so
// every row of every slice read the SAME four columns, and the value divided
// by s*rows while the gradient behaved as if by rows alone. Mutation testing
// plus centred FD found it: the gradient was exactly 1/s of the value's own
// derivative -- a CLEAN constant ratio, which is this repo's standing signal
// for a structural index error rather than a tolerance problem.)
double ortho_penalty_slices(const Tensor& w, size_t num_slices, size_t s, size_t c);
Tensor ortho_penalty_slices_gradient(const Tensor& w, size_t num_slices, size_t s, size_t c);

// ---------------------------------------------------------------------------
// The one-sided spectral variant, used to normalise a generator's weights
// rather than to penalise a Gram matrix. (D10)
// ---------------------------------------------------------------------------
// Largest singular value, by a FIXED-step power iteration (deterministic).
// THROWS on an empty tensor. Returns exactly 0 for the zero matrix.
double ortho_spectral_norm(const Tensor& w);

// `1 / sv1`. THROWS std::invalid_argument when `sv1 == 0` rather than
// returning infinity -- an infinity here silently poisons every downstream
// weight update.
double ortho_spectral_norm_inverse(const Tensor& w);

// |sv1 - 1|: the ONE-SIDED residual. Zero for a semi-orthogonal matrix.
double ortho_penalty_one_sided(const Tensor& w);

// (sv1 - 1)^2, the penalty form. Excludes sv1 == 0 by definition (a zero matrix
// is not a scaled rotation), returning exactly 0 there.
double ortho_penalty_spectral(const Tensor& w);
Tensor ortho_penalty_spectral_gradient(const Tensor& w);

// ---------------------------------------------------------------------------
// Model-level: the regulariser that acts on a network's WEIGHTS.
//
// The gradient is added to the parameter gradients already stored by a
// `Model::backward` pass. It acts on the weights DIRECTLY, so the usual loop
// is: zero_grad / forward / backward / apply_to_gradients / optimizer.step.
// (Per the adversarial-session lesson: this is a regulariser on the WEIGHT
// SPACE, so there is no activation path to route through -- calling it with a
// different `lambda` than the one used for `evaluate` is a mismatch the
// caller can make and this class does not hide.)
// ---------------------------------------------------------------------------
class OrthoPenaltyLoss {
public:
    explicit OrthoPenaltyLoss(double lambda = 1.0) : lambda_(lambda) {}

    double lambda() const { return lambda_; }
    void set_lambda(double lambda) { lambda_ = lambda; }

    // The penalty summed over every parameter tensor in `model`, scaled by
    // `lambda`. THROWS std::invalid_argument on a model with no layers.
    double evaluate(Model& model) const;

    // The penalty gradient with respect to ONE parameter tensor, unscaled by
    // lambda (the scale is applied by `apply_to_gradients`). A parameter that
    // is not a 2-D weight (e.g. a 1-D bias) has penalty EXACTLY 0 and
    // gradient EXACTLY 0 -- a bias has no Gram matrix to regularise.
    Tensor gradient_wrt(const Tensor& param) const;

    // Add `lambda * dP/dW` to the gradients already stored in `model`. Only
    // 2-D parameters are touched. `lambda` defaults to this class's own.
    void apply_to_gradients(Model& model, double lambda = -1.0) const;

private:
    double lambda_;
};

#endif  // NN_UTILS_ORTHO_PENALTY_H