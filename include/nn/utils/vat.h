#ifndef NN_UTILS_VAT_H
#define NN_UTILS_VAT_H

#include "../core/tensor.h"
#include "../core/model.h"
#include "adversarial.h"        // GradientFn
#include <cstddef>
#include <functional>
#include <vector>

// ===========================================================================
// Virtual Adversarial Training
//
//   Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Shin Ishii --
//   "Virtual Adversarial Training: A Regularization Method for Supervised and
//   Semi-Supervised Learning", ICLR 2018, arXiv:1704.03976.
//
// CITATION WARNING. arXiv:1704.03941 is NOT this paper -- that ID is "Analysis
// of pacemaker activity in a two-component model of some brainstem neurons".
// Do not reintroduce it.
//
// THE IDEA. Adversarial training (see `adversarial.{h,cpp}`, Madry arXiv:1706.06083)
// maximises the loss over a ball and descends on the loss at the maximiser. That
// requires LABELS: the inner objective needs a y. VAT replaces the label loss
// with a divergence between the model's own output distribution at x and at the
// worst nearby point:
//
//     r_vadv = argmax_{||r||_2 <= eps}  KL( p(y|x) || p(y|x+r) )
//     R_vadv =            KL( p(y|x) || p(y|x+r_vadv) )
//     L     =  L_labelled(theta)  +  alpha * R_vadv(theta)
//
// So VAT needs NO LABELS AT ALL and no adversarial example at test time. It is
// the missing third corner of this repo's robustness triangle: `adversarial.h`
// SEARCHES for a worst-case delta, `randomized_smoothing.h` BOUNDS how far any
// attack can move the classifier, and this file REGULARISES the model to be
// smooth in between.
//
// WHY IT LIVES HERE AND NOT IN layers/: the attack is not a layer. It consumes
// logits and a gradient and returns an input; nothing is learned by it. Same
// model-free shape as `adversarial.h`, which means the entire power-iteration
// core is testable with a synthetic oracle and no network.
//
// WHY IT IS CHEAP HERE: the repo's `Layer::backward` already RETURNS dL/d(input)
// rather than storing it, so one power iteration is the SIGN of an ordinary
// backward pass driven by a batch-mean KL. No base-class change.
//
// ===========================================================================
// Semantics, and the rules a reimplementation gets wrong
// ===========================================================================
//
// (D1) THE KL ARGUMENT ORDER IS clean || perturbed. NOT the reverse. Paper Eq.
//      (17): D[p(y|x), p(y|x+r)] = sum_y p(y|x) log(p(y|x) / p(y|x+r)). The
//      reference's parameter NAMES are misleading -- `kl_divergence_with_logit
//      (q_logit, p_logit)` in layers.py:137 passes the CLEAN logit as the first
//      argument. What matters is the position, and clean-first is what both the
//      paper and both reference implementations do.
//      Measured on the test fixture: KL(clean||adv) = 0.32342102212705642,
//      KL(adv||clean) = 0.32363019005210997. KL is asymmetric and the two differ
//      in the 6th decimal -- invisible by eye, fatal to a 1e-9 tolerance.
//
// (D2) THE CLEAN BRANCH IS DETACHED. `vat.py:55` applies `tf.stop_gradient` to
//      the clean logits, so
//          d/d(perturbed logits) [ mean_i KL(clean_i || p_i) ] = (p_k - q_k) / N
//      -- a ONE-SIDED KL whose gradient flows only through the perturbed branch.
//      The SIGN is (p - q)/N, not (q - p)/N. This was settled by finite
//      differences, not by reading: an oracle written with the opposite sign
//      disagreed at rel_err ~ 2.0, the signature of an exactly inverted
//      gradient. Note also the 1/N: the batch mean inside the KL helper.
//
// (D3) THE PROBE IS FORMED BEFORE THE GRADIENT STEP, AND eps IS APPLIED AFTER.
//      `r_probe = xi * normalise(d)`; then take the gradient at that probe; then
//      `r_vadv = eps * normalise(g)`. The chain rule therefore carries a factor
//      of xi, which the final normalisation divides back out -- so xi does not
//      change the DIRECTION of a single-iteration result, only where the
//      linearisation is evaluated. That is exactly why xi must be small.
//
// (D4) THE GRADIENT VANISHES EXACTLY AT r = 0. KL(p(y|x) || p(y|x+r)) is
//      QUADRATIC in r, so dKL/dr = 0 at r = 0 -- measured `max|dKL| = 0` and
//      `KL(clean||clean) = 0` to the last bit. Two consequences:
//        (a) A centred finite-difference gradient check taken at r = 0 is
//            VACUOUS. It divides floating-point noise by 2*eps and compares it
//            against a zero analytical value, so it "passes" for a correct AND
//            an incorrect implementation. Test at a SHIFTED point instead.
//        (b) It is also why a power iteration is needed at all, and why the
//            Hessian of the KL at r = 0 (PSD) is what makes it converge.
//
// (D5) THE REFERENCE DISTRIBUTION IS CACHED, NEVER RECOMPUTED AT THE PROBE.
//      `vat.py:44` aliases `logit_p = logit` -- the clean forward happens once,
//      outside the loop. A from-scratch implementation that recomputes p_bar
//      inside the differentiated graph gets p == q and therefore a gradient of
//      EXACTLY zero (measured `||g|| = 0.0`), i.e. a perturbation that is pure
//      noise. This is a silent failure: it produces finite, plausible numbers.
//
// (D6) ||r_vadv|| IS eps * 0.9999997, NOT eps. `get_normalized_vector`
//      (vat.py:33-36) divides by `sqrt(1e-6 + sum d^2)` -- the `1e-6` biases the
//      norm BELOW one. Measured 7.9999983496010874 for eps = 8. So a test that
//      asserts `||r_vadv|| == eps` fails on the CORRECT implementation, and one
//      that asserts `||r_vadv|| <= eps` passes on a broken one. Pin the exact
//      value.
//
// (D7) THE LOSS IS MONOTONE NON-DECREASING IN eps. r_vadv is an argmax over the
//      ball, so widening the ball can only raise the inner maximum. This is the
//      property that distinguishes a real power iteration from an arbitrary
//      perturbation.
//
// (D8) ENTROPY MINIMISATION IS NEGATIVE ENTROPY, AND IT IS ADDED. The
//      reference's `entropy_y_x` (layers.py:145-148) returns
//      `-mean_i sum_j p_ij log p_ij`, so ADDING it under gradient DESCENT
//      minimises the entropy, i.e. sharpens predictions -- the opposite of the
//      intuition that a function called "entropy" should be maximised. Entropy
//      minimisation is an OPTIONAL extra term (the reference's `--method=vatent`),
//      never a replacement for the VAT term.
//
// (D9) THE OUTPUT-LAYER WEIGHT IS LOAD-BEARING IN THE BACKWARD.
//      dlogit_k/dx = sum_h W2[k,h] * sech^2(h) * W1[h,c]. An implementation that
//      writes the VJP without the W2 factor does not merely change the
//      magnitude -- for a two-class head it collapses the result to ~1e-17,
//      because dl[k=1] = -dl[k=0] exactly and the hidden terms then cancel.
//      Measured: analytic (no W2) = -9.305e-18, FD = -3.099e-09. In this repo
//      the factor is supplied by `Dense::backward`, so the risk is not writing
//      the VJP by hand but bypassing the model -- hence `vat_loss` takes a
//      LogitsFn + GradientFn pair rather than a hand-rolled Jacobian.
//
// (D10) NORMALISATION IS PER-SAMPLE OVER EVERY NON-BATCH AXIS. `vat.py:34-35`
//      reduces over `range(1, len(shape))` with keep_dims. A global norm over
//      the whole batch silently gives differently-scaled rows.
//
// (D11) THE 1e-12 + max|d| STEP IS AN OVERFLOW GUARD THAT CANCELS. It is
//      mathematically a no-op between the two divides and cannot affect the
//      direction -- but it is load-bearing for numerical safety on large
//      gradients, and the SEPARATE 1e-6 in the second divide is NOT a no-op.
//      A zero vector must give zero, not 0/0 = NaN.
// ===========================================================================

// The model's forward map. This is the ONLY interface to a network that the
// power iteration needs, which is what keeps the core testable without a Model.
using LogitsFn = std::function<Tensor(const Tensor&)>;

// The initial search direction. The paper draws it from an iid Gaussian
// (Algorithm 1 step 2); it is injectable because std::normal_distribution is
// not bit-reproducible across standard libraries, and a test that asserts exact
// digits on a Gaussian is a flaky test. A caller wanting the paper's behaviour
// uses the default; a caller wanting reproducibility supplies a fixed one.
using DirectionFn = std::function<Tensor(const Tensor&)>;

// Per-row log-softmax, max-shifted (the reference's form, layers.py:131-135).
Tensor vat_log_softmax(const Tensor& logits);

// Per-row softmax, max-shifted.
Tensor vat_softmax(const Tensor& logits);

// KL(clean || perturbed) averaged over the batch  (D1).
//
// Both arguments are LOGITS, not probabilities. Requires the same shape and a
// non-empty batch. KL(p||p) is EXACTLY 0 (not 1e-17) because both branches
// evaluate the same expression.
double kl_divergence(const Tensor& clean_logits, const Tensor& perturbed_logits);

// KL(clean || perturbed) PER ROW, shape (B, 1). Exposed because the per-sample
// form is what a weighted or thresholded objective needs.
Tensor kl_divergence_per_sample(const Tensor& clean_logits,
                                const Tensor& perturbed_logits);

// d/d(perturbed logits) of `kl_divergence`, treating `clean_logits` as FIXED
// (D2). Returns (p_k - q_k) / N.
//
// This is the seed gradient the power iteration back-propagates. `clean_logits`
// must be a CACHED tensor from the clean forward pass (D5), never recomputed at
// the probe.
Tensor dkl_divergence_dlogits(const Tensor& clean_logits,
                              const Tensor& perturbed_logits);

// Per-row L2 normalisation in the reference's guarded two-step form  (D6/D10/D11):
//     d /= (1e-12 + max|d|)
//     d /= sqrt(1e-6 + sum d^2)
// Same shape as the input. A zero row yields zero, never NaN.
Tensor normalise_direction(const Tensor& d);

// Configuration. The defaults are the reference's own flag defaults
// (vat.py:10-12) and the paper's CIFAR-10 setting.
struct VATConfig {
    // Radius of the virtual-adversarial ball. Paper/reference: 8.0 for CIFAR-10.
    double epsilon = 8.0;
    // Finite-difference step for the second-order approximation, Eq. (12).
    // Must satisfy xi << epsilon; 1e-6 in all of the paper's experiments.
    // xi == 1 is NOT a meaningful setting (it voids the linearisation) and the
    // header documents that, but it is not forbidden.
    double xi = 1e-6;
    // Power iterations. The paper: "only one power iteration was sufficient".
    int power_iterations = 1;
    // Initial direction. Empty means "use the library's seeded Gaussian";
    // VATConfig::default_direction() is always non-empty by construction.
    DirectionFn direction;
    // Verbose per-epoch logging from VATTrainer.
    bool verbose = false;

    // The library default direction: an iid Gaussian, seeded deterministically
    // from `seed` so a run is reproducible. Matches Algorithm 1 step 2's
    // distribution while staying bit-reproducible for a given seed.
    static DirectionFn default_direction();
    static DirectionFn gaussian_direction(unsigned seed);
};

// The virtual-adversarial perturbation (paper Eq. 14, `vat.py:39-50`):
//     d     = direction(x)
//     for _ in range(power_iterations):
//         d = xi * normalise(d)                 # probe, BEFORE the gradient
//         d = grad( dKL(clean || f(x+d)) / dr ) # stop-gradient, detaches
//     return eps * normalise(d)                 # AFTER the loop
//
// `grad` must be the gradient of the batch-mean KL with the CLEAN logits held
// fixed -- the same `GradientFn` shape `adversarial.h` uses, so a caller can
// pass one factory that serves both. `clean_logits` may be supplied to save a
// forward pass; if it is empty the function computes and caches it itself
// (still detached, D5).
//
// Throws std::invalid_argument on: a non-positive epsilon, a non-positive xi,
// a negative power_iterations, an empty batch, or a missing `grad`.
Tensor virtual_adversarial_perturbation(const Tensor& x, const LogitsFn& forward,
                                        const GradientFn& grad,
                                        const VATConfig& cfg,
                                        const Tensor& clean_logits = Tensor());

// The VAT loss: KL(clean || perturbed) at the virtual-adversarial point
// (paper Eq. 5). Equals 0 exactly when the model is locally invariant.
double vat_loss(const Tensor& x, const LogitsFn& forward,
                const GradientFn& grad, const VATConfig& cfg,
                const Tensor& clean_logits = Tensor());

// NEGATIVE entropy of the predictive distribution, batch-averaged  (D8).
// Returns -mean_i sum_j p_ij log p_ij. Uniform over K classes gives log(K).
// ADDING this under gradient DESCENT sharpens the predictions.
double entropy_negative(const Tensor& logits);

// d/d(logits) of `entropy_negative`:
//     dE/dz_ik = -(1/N) * p_ik * (log p_ik + H_i),  H_i = -sum_k p_ik log p_ik
// The MINUS is load-bearing: measured by centred finite differences, the
// un-negated form disagrees at rel_err EXACTLY 2.0 -- the same inverted-
// gradient fingerprint that D2's KL sign produced.
Tensor d_entropy_negative_dlogits(const Tensor& logits);

// ---------------------------------------------------------------------------
// VATTrainer -- the outer minimisation of paper Eq. (8):
//     min_theta  L(D_l, theta) + alpha * R_vadv(theta)
// ---------------------------------------------------------------------------
class VATTrainer {
public:
    struct Config {
        int epochs = 10;
        double learning_rate = 0.01;
        // Weight on the VAT term. The reference HARD-CODES alpha = 1 (there is
        // no scale flag in vat.py), so 1.0 is the faithful default.
        double alpha = 1.0;
        // When false this degenerates to plain softmax cross-entropy training,
        // which is what makes the "is the VAT path a silent no-op?" check in
        // the test suite possible.
        bool use_vat = true;
        // Optional entropy-minimisation term (D8). Off by default: the paper's
        // main method is `vat`, not `vatent`.
        bool use_entropy_min = false;
        double entropy_weight = 1.0;
        VATConfig vat;
        // Seed for the Gaussian direction, so a run is reproducible.
        unsigned seed = 0;
    };

    explicit VATTrainer(Model& model);

    // Train on `X` (one example per row) and integer class indices in `y`.
    // `epoch_losses`, if non-null, receives the mean total loss of each epoch.
    //
    // Uses the repo's canonical `CrossEntropyLoss` (utils/losses.h), matching
    // `adversarial_train`.
    //
    // Throws std::invalid_argument on: a model with no layers, X.rows != y.rows,
    // an empty X or y, or an invalid config (alpha < 0, epochs < 0).
    void train(const Tensor& X, const Tensor& y, const Config& cfg,
               std::vector<double>* epoch_losses = nullptr);

    // The VAT (or CE, or CE+VAT) loss the last training step used. Useful for
    // a caller composing a custom loop; not needed for `train` itself.
    double last_loss() const { return last_loss_; }

private:
    Model& model_;
    double last_loss_ = 0.0;
};

#endif  // NN_UTILS_VAT_H