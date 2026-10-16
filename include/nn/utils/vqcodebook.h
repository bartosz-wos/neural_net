#ifndef NN_UTILS_VQCODEBOOK_H
#define NN_UTILS_VQCODEBOOK_H

#include "../core/tensor.h"
#include <cstddef>
#include <vector>

// ===========================================================================
// VQ codebook maintenance
//
//   Minyoung Huh, Brian Cheung, Pulkit Agrawal, Phillip Isola --
//   "Straightening Out the Straight-Through Estimator: Overcoming
//   Optimization Challenges in Vector Quantized Networks",
//   arXiv:2305.08842 (ICML 2023).
//
// CITATION. Verified against the arXiv API on 2026-10-16 via the query
// `ti:"Straightening Out the Straight-Through Estimator"`, which returns
// exactly this paper with exactly these four authors. This repo's expansion
// queue has produced EIGHT wrong arXiv IDs before, so the check is not
// optional. THREE further candidate IDs were guessed WRONG while scoping this
// and are recorded so they are never reintroduced:
//     arXiv:1911.05863  -> "Regularity results for a nonlinear
//                           elliptic-parabolic system with oscillating
//                           coefficients" (a PDE paper)
//     arXiv:1912.04997  -> a PGL_2 representation-theory MSc thesis
//     arXiv:2107.03312  -> 0 results
//     "Reviving Dead Codes in VQ-VAE"    -> 0 arXiv results by title search
//     "Input Sanitization for VQ-VAE"   -> 0 arXiv results by title search
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//
// `layers/generative/vq_vae.h` EXPOSES `dead_code_count()` on both
// `VectorQuantizer` and `VectorQuantizerEMA`, with the comment "A rising
// dead-code count is the classic codebook-collapse signal, which is why it is
// exposed rather than only inferred."
//
// Measured by content grep before this file was written:
//     grep -rn 'dead_code_count\|usage_count' --include=*.h --include=*.cpp .
// -> ONLY vq_vae.{h,cpp} itself and tests/test_vq_vae.cpp. Nothing in the repo
// ACTS on it.
//
// So the quantizer detects the one failure mode its own method is known for
// and then does nothing about it. This file is the missing half: the paper's
// diagnosis (§3.1) plus the two mechanisms it concludes are worth shipping --
// k-means INITIALISATION and an LRU REPLACEMENT policy.
//
// ---------------------------------------------------------------------------
// WHY A REPLACEMENT POLICY IS NECESSARY RATHER THAN OPTIONAL
// ---------------------------------------------------------------------------
//
// The obvious objection to any collapse remedy is "just train longer". The
// paper answers it, and the answer is a structural one worth reading before
// writing any of this. Section 3.1 rewrites the commitment loss as
//
//     min_Cz D(Pz, Cz) = 1/|Pz| * sum_{zi in Pz} min_{cj in Cz} d(zi, cj)   (13)
//
// -- an average over `P_z` but a minimisation over `C_z`. The consequence:
//
//     "Here the disjoint set Cz \ Qz does not receive any gradient and is not
//      trained."
//
// A dead code is not merely slow to recover -- it is receiving EXACTLY ZERO
// gradient forever, because no row of `z_e` ever selects it, so the term that
// would move it is never instantiated. No number of additional SGD steps
// revives it. This is why `vq_codebook_gradient` in vq_vae.h returns exactly
// zero for unassigned codes (that file's D8-adjacent property), and it is why
// a policy that must WRITE the code is not optional.
//
// The paper is explicit that replacing symptoms is not the whole answer --
// "these methods address the symptoms of the collapse by replacing inactive
// codes rather than re-solving why they became inactive in the first place" --
// and its own answer (affine re-parameterisation, §4.1) is deliberately OUT OF
// SCOPE here because it changes the quantizer's parameterisation rather than
// adding a utility. What IS shipped is the part that is composable.
//
// ---------------------------------------------------------------------------
// API shape
// ---------------------------------------------------------------------------
//
// Every numerical core is a FREE FUNCTION over `Tensor`s, so this file is
// testable with no `Model`, no optimizer and no encoder -- the same model-free
// property `utils/vat.h` and `utils/randomized_smoothing.h` were built with.
//
// Shapes. The codebook is (D, K) with COLUMN k IS embedding e_k, matching
// `vq_vae.h` D3 and Sonnet's `self._w`. `ages` is a (1, K) tensor owned by the
// caller, chosen to be layout-compatible with `VectorQuantizerEMA::
// cluster_size()` (also (K)) so the two compose without a conversion.
//
// ===========================================================================

// ---------------------------------------------------------------------------
// k-MEANS CODEBOOK INITIALISATION  (paper §3, and §3.2's "One approach is to
// ensure the quantization error is small at initialisation using K-means")
// ---------------------------------------------------------------------------
//
// `z` is (M, D) -- the encoder outputs. Returns a (D, K) codebook whose
// COLUMNS are the k-means CENTRES of `z`'s rows, reusing the already-shipped
// `KMeans` class in `gmm.h` rather than a second implementation of Lloyd's
// algorithm in this file.
//
// THROWS std::invalid_argument when M < K (D8). The shipped `KMeans::fit`
// already throws on that condition; silent zero-padding would produce a
// degenerate codebook that looks like a legitimate one. Also throws on an
// empty `z`.
//
// Determinism (D9): reproducible for a given `seed`. The returned centre for
// a well-separated fixture is that cluster's MEAN -- asserted against
// `sklearn.cluster.KMeans` 1.9.1 in the test suite, not against a
// transcription of it.
Tensor vqcb_kmeans_initialise(const Tensor& z, size_t num_codes,
                              unsigned int seed = 42u, int max_iter = 100);

// ---------------------------------------------------------------------------
// LRU REPLACEMENT POLICY  (paper §3, "Replacement policy")
// ---------------------------------------------------------------------------
//
// Paper §3, verbatim: "Replacement policies require careful tuning, and we
// find that using a least-recently-used (LRU) policy with a life-span of 20
// iteration works the best -- if the code does not get used for 20 training
// iterations, it gets replaced with a random embedding vector. When using
// replacement policies, the active codes are left unchanged, and the overall
// performance of the model does not degrade."
//
// The lifespan of 20 is the paper's own; it is a PARAMETER here, not a
// constant, because the sentence that establishes it ("works the best") is
// specific to their setup (1024 codes of dim 512, ResNet18, ImageNet100) and
// does not transfer to a general library.
//
// Ordering of `ages`: element k is the iteration count since code k was last
// USED. A code with `ages(k) >= lifespan` is replaced.
//
// D5: a code that is NOT replaced is left BYTE-IDENTICAL. "The active codes
// are left unchanged" is the paper's own claim and it is the property that
// makes this policy safe to leave switched on.
//
// D7: replacement vectors are taken from `rows` (an (N, D) pool of encoder
// outputs) in DESCENDING reconstruction error, ties broken by LOWEST ROW
// INDEX. The paper's wording for Zeghidour et al. is "randomly sampled
// embedding", but §3's replacement analysis and the whole input-sanitization
// family move high-error inputs in, and a random draw from an untrained
// encoder is a worse source than the encoder's own worst-fitting samples. The
// pool is consumed CYCLICALLY, so `num_replaced` may exceed the pool size.
//
// RETURNS the number of codes replaced. Resets `ages` to 0 at exactly the
// replaced slots and leaves every other age untouched (D6).
//
// THROWS std::invalid_argument if there are dead codes but `rows` is empty --
// inventing a vector would be worse than refusing.
size_t vqcb_lru_replace(Tensor& codebook, Tensor& ages, const Tensor& rows,
                        double lifespan);

// Advance every code's age by one, then reset the ages of `used` to 0.
// D10: called ONCE PER TRAINING ITERATION by the caller -- not once per row.
// A per-row call would age codes 10x too fast for a batch of 10 and silently
// replace codes that are in active use.
//
// THROWS std::invalid_argument on an out-of-range index in `used`.
// `Tensor::operator()` is `data[r * cols + c]` with NO bounds check, so an
// unvalidated index is heap corruption, not an exception (the repo has been
// bitten by exactly this in the VAT fixture). Duplicate indices are
// idempotent, not an error.
void vqcb_lru_tick(Tensor& ages, const std::vector<size_t>& used);

// Row indices of `rows` ordered by DESCENDING squared error against `z_q`,
// ties broken by LOWEST ROW INDEX. Exposed because the ordering rule is part of
// the contract (D7) and no tolerance-based test can observe a tie-break -- only
// an EXACT tie can.
//
// THROWS std::invalid_argument if `rows` and `z_q` disagree in shape.
std::vector<size_t> vqcb_rows_by_error_desc(const Tensor& rows,
                                            const Tensor& z,
                                            const Tensor& z_q);

// ---------------------------------------------------------------------------
// COMMITTEDNESS LOSS  (paper Eq. 8-9)
// ---------------------------------------------------------------------------
//
//     L_cmt(z_e, z_q) = (1 - beta) * d(z_e, sg[z_q])    <- pulls z_e to the code
//                     + beta      * d(sg[z_e], z_q)    <- pulls the code to z_e
//
// with, on p.4, verbatim: "A good rule of thumb is to set alpha = 10 and
// beta = 0.9 when using euclidean distance dl2(ze, zq) := 1/2 ||ze - zq||^2_2
// (Van Den Oord et al., 2017)."
//
// THE TWO TERMS ARE THE SAME SCALAR. sg[.] is the identity at forward time, so
// `d(z_e, z_q) == d(z_q, z_e)` and
//
//     L_cmt = (1-beta)*Q + beta*Q = Q,   Q = 0.5*mean((z_e - z_q)^2)
//
// INDEPENDENT OF BETA ENTIRELY. The measured values bear this out: beta=0 and
// beta=1 both give 0.008472222222222221 on the suite's fixture. **No value
// test can distinguish the committedness loss from a plain halved squared
// error, or distinguish the two terms, or distinguish this loss from van den
// Oord's dictionary term up to the factor of 2.** Only the two SEPARATE
// gradients can. That is why `vqcb_committedness_gradient_ze` and
// `vqcb_committedness_gradient_codebook` exist as separate entry points and
// why neither of them may be derived from the other by re-reading the loss.
//
// D2: `d(.,.)` is 0.5*||a-b||^2, NOT ||a-b||^2. Asserted as a RATIO against
// an independently computed unhalved mean, because a factor of 2 on a single
// scalar is invisible to any value assertion on that scalar.
//
// D3: the mean is over all M*D elements, matching `vq_vae.h` D2, so the two
// implementations' beta values are directly comparable.
//
// D4: sg[.] goes on the operand BEING PULLED TOWARD THE OTHER. Term 1 pulls
// z_e toward z_q so z_q is stopped; term 2 pulls the code toward z_e so z_e is
// stopped. Inverted, the LOSS is unchanged and every value test passes.
//
// The individual terms are exposed because -- exactly as in vq_vae.h, and for
// exactly the same reason -- a caller cannot otherwise see which is which.
double vqcb_committedness_term1(const Tensor& z, const Tensor& z_q, double beta);
double vqcb_committedness_term2(const Tensor& z, const Tensor& z_q, double beta);
double vqcb_committedness_loss(const Tensor& z, const Tensor& z_q, double beta);

// d(committedness)/d(z_e) = (1 - beta) * (z_e - z_q) / (M*D), shape (M, D).
// At beta = 1 this is EXACTLY ZERO -- there is nothing pulling the encoder.
Tensor vqcb_committedness_gradient_ze(const Tensor& z, const Tensor& z_q,
                                      double beta);

// d(committedness)/d(z_q) = beta * (z_q - z_e) / (M*D), shape (M, D).
//
// d(z_q)/dz_e is the straight-through path and belongs to the quantizer, not
// here; this is the codebook-facing half, and it is what makes beta's
// POSITION (D1) observable at all. Note the shape is (M, D) rather than
// (D, K): the loss depends on the codes only through the SELECTED ones, so
// the caller accumulates this onto the codebook by assignment, exactly as
// `vq_codebook_gradient` does.
Tensor vqcb_committedness_gradient_codebook(const Tensor& z, const Tensor& z_q,
                                            double beta);

#endif // NN_UTILS_VQCODEBOOK_H