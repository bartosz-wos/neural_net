#ifndef NN_UTILS_ADVERSARIAL_H
#define NN_UTILS_ADVERSARIAL_H

#include "../core/tensor.h"
#include "../core/model.h"
#include <cstddef>
#include <functional>

// ===========================================================================
// Adversarial robustness — FGSM and L-infinity PGD
//
//   Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras,
//   Adrian Vladu — "Towards Deep Learning Models Resistant to Adversarial
//   Attacks", ICLR 2018, arXiv:1706.06083.
//
// THE IDEA. Robust optimisation (the paper's Eq. 2.1) is a SADDLE POINT:
//
//     min_theta  E_{(x,y)~D} [ max_{delta in S} L(theta, x + delta, y) ]
//
// An adversary picks the perturbation that MAXIMISES the loss, and the model is
// then trained to minimise the loss AT that worst perturbation. So this file
// provides the inner problem — an attacker — and `adversarial_train` implements
// the outer one.
//
// WHY IT LIVES HERE AND NOT IN layers/: the attack is not a layer. It consumes
// an input and a gradient and returns an input; nothing is learned by it. This
// is the same model-free shape as knowledge_distillation and
// temperature_scaling, and it means the attack core is testable with a
// synthetic gradient oracle and no network at all.
//
// WHY IT IS CHEAP HERE (this is the reason it was deferred until now): the
// repo's `Layer::backward` already RETURNS dL/d(input) rather than storing it,
// and no layer's `backward()` writes to a parameter (verified by grep across
// all ~325 backward implementations). So an attack step is the SIGN of an
// ordinary backward pass, and an adversarial training step is a training loop
// with a perturbed input. No base-class change, zero risk to existing layers.
//
// ===========================================================================
// Semantics, and the rules a reimplementation gets wrong
// ===========================================================================
//
// (D1) IT IS ASCENT. The attack is a MAXIMISATION of the loss -- "a simple
//      one-step scheme for maximizing the inner part of the saddle point
//      formulation" (§2.1). The perturbation is `+ eps * sgn(grad)`, never
//      `-`. A descent implementation produces an attack that IMPROVES the loss
//      and would make adversarial training a no-op that converges to a
//      vulnerable model. Measured: the attack raises the loss from
//      0.73522905727170718 to 0.76273603563171388 on the end-to-end fixture.
//
// (D2) sign(0) IS 0, NOT +1. This is the single most likely bug in the whole
//      feature. `grad >= 0 ? 1 : -1` -- the natural way to write the three-way
//      case in C++ if you forget that zero has no direction -- moves a
//      zero-gradient coordinate by a FULL eps. PyTorch was checked directly:
//      `torch.sign(0.0) == 0.0` and `torch.sign(-0.0) == 0.0`. A coordinate
//      whose gradient is exactly zero must not move at all, which is what the
//      mixed-sign fixture (two exact zeros) pins. This is invisible on any
//      fixture without an exact zero, so no tolerance-style test can find it.
//
// (D3) THE PROJECTION IS CENTRED ON x_nat -- THE ORIGINAL INPUT -- not on the
//      current iterate and not on the random start. This is settled by the
//      authors' own reference implementation, not by the paper:
//          x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
//      (MadryLab/mnist_challenge/pgd_attack.py:53). Without it the attack is
//      no longer a bounded attack at all. Discriminator measured at eps=0.3,
//      alpha=0.1, k=7: the correct projection gives max|delta| = 0.3; the
//      x_t-centred variant gives 0.7.
//
// (D4) THE STEP IS APPLIED BEFORE THE PROJECTION, every iteration. PGD with
//      k=1 is therefore bit-identical to FGSM at the same eps, and the suite
//      asserts that identity rather than re-deriving it.
//
// (D5) THE DATA-RANGE CLAMP IS OPT-IN, AND IT IS APPLIED LAST. The paper's
//      formula has no clamp. The reference implementation clamps to [0, 1]
//      because MNIST pixels live there, and does so AFTER the ball projection,
//      including on the random start. Clamping first, clamping implicitly, or
//      clamping only inside the loop are three different functions. On the
//      mixed-sign fixture at eps=0.3 the pure form gives row 0 =
//      [0.8, -0.55, 1.05, -0.425] and the clamped form gives [0.8, 0, 1, 0] --
//      visibly, numerically different, so the suite pins both.
//
// (D6) eps AND alpha ARE INDEPENDENT. eps is the radius of the ball; alpha is
//      the size of each step. The authors' own challenge config uses
//      eps = 0.3 with a = 0.01 -- a 30x ratio (mnist_challenge/config.json) --
//      so defaulting alpha to eps is visibly wrong at the settings the paper
//      actually used.
//
// (D7) k IS THE NUMBER OF GRADIENT EVALUATIONS. k = 0 is the identity attack
//      (returns the input untouched, and never calls the gradient function),
//      not "one step".
//
// (D8) THE ATTACK MUST NOT TOUCH A PARAMETER. It is an evaluation-time
//      operation. `Layer::backward(grad, 0.0)` accumulates into `grad_*` but
//      writes no weight; this file additionally zero_grad()s on the way out so
//      an attack cannot contaminate a surrounding training loop.
//
// (D9) THE L2 VARIANT NORMALISES, THE LINF VARIANT SIGNS. `l2_fgsm` divides the
//      gradient by its per-ROW L2 norm; a row with an exactly zero gradient has
//      no direction and is left unchanged rather than producing 0/0 = NaN.
//      The norm is per ROW, not global: rows with different gradient magnitudes
//      get proportionally different per-coordinate steps, and a global norm
//      silently produces a different perturbation.
// ===========================================================================

// Elementwise sign with sign(0) == 0 (D2). Same shape as the input.
Tensor sign_tensor(const Tensor& x);

// Optional clamp of the data to a valid range (D5). The defaults describe
// UNBOUNDED data: no clamp is applied unless the caller opts in. Note these are
// NOT image defaults -- [0,1] belongs to MNIST/CIFAR, not to this library.
struct DataRange {
    bool enabled = false;
    double lo = 0.0;
    double hi = 1.0;

    // Unbounded: no clamp is ever applied.
    DataRange() = default;
    // Clamp to [lo_, hi_]. Throws at attack time (not here) if lo_ > hi_, so a
    // bad range is reported by the call that uses it.
    DataRange(double lo_, double hi_) : enabled(true), lo(lo_), hi(hi_) {}

    static DataRange none() { return DataRange(); }
    static DataRange unit() { return DataRange(0.0, 1.0); }
};

// The gradient oracle an attack consumes: given the CURRENT input, return
// dL/d(input) -- the same quantity `Layer::backward` returns. This is the whole
// interface between the attack and any model.
using GradientFn = std::function<Tensor(const Tensor&)>;

// ---------------------------------------------------------------------------
// FGSM (paper §2.1):  x_adv = x + eps * sgn(grad_x L)
// ---------------------------------------------------------------------------
// `epsilon` must be > 0 and `range` must be well-formed (lo <= hi) or
// std::invalid_argument is thrown. The pure paper formula -- no clamp unless
// one is requested.
Tensor linf_fgsm(const Tensor& x, double epsilon, const GradientFn& grad,
                 DataRange range = DataRange());

// ---------------------------------------------------------------------------
// L-inf PGD (paper §2.1):  x_{t+1} = Pi_{x+S} ( x_t + alpha * sgn(grad_x L) )
// ---------------------------------------------------------------------------
// `epsilon` > 0, `alpha` > 0 and `num_steps` >= 0, else std::invalid_argument.
// With `random_start` the iterate begins at clip(x + U(-eps, eps), range),
// which is the reference implementation's default attack mode. `seed` makes
// that draw reproducible; the default 0 is deterministic.
//
// NOTE: `random_start` defaults to FALSE here even though PGDConfig (the
// training config) defaults it to TRUE. The asymmetry is deliberate. The
// authors' own challenge config sets `"random_start": true` for BOTH training
// and evaluation, and random start is what makes the attack STRONGER (it
// escapes the saddle point the deterministic start lands in). So the library's
// training default is strong. But a single-shot attack call defaults to the
// deterministic start because a caller who writes
// `linf_pgd(x, eps, alpha, k, grad)` almost always wants a REPRODUCIBLE
// adversarial example for a regression test, a demo, or a saved artefact --
// and silently handing back a different example on every call is the worse
// surprise. Pass random_start = true explicitly when you want the stronger
// attack; random_restart_pgd() does it for you.
Tensor linf_pgd(const Tensor& x, double epsilon, double alpha, int num_steps,
                const GradientFn& grad, DataRange range = DataRange(),
                bool random_start = false, unsigned seed = 0);

// ---------------------------------------------------------------------------
// Random-restart PGD: run `num_restarts` independent attacks and keep the
// worst one by the supplied objective.
//
// Madry observed (§3.2) that over many random restarts the final loss is
// well-concentrated -- so the best-of-n is a cheap, genuinely stronger attack
// than a single run, and it is the standard baseline for "strongest attack".
//
// `objective` maps an input to a scalar to MAXIMISE (higher = stronger). If it
// is empty the returned attack is simply the LAST restart, which keeps the
// function usable without an objective. `num_restarts` >= 1, else
// std::invalid_argument.
Tensor random_restart_pgd(const Tensor& x, double epsilon, double alpha,
                          int num_steps, int num_restarts,
                          const GradientFn& grad, DataRange range = DataRange(),
                          unsigned seed = 0,
                          std::function<double(const Tensor&)> objective = nullptr);

// ---------------------------------------------------------------------------
// L2 FGSM:  x_adv = x + eps * grad / ||grad||_2   (per row, D9)
// ---------------------------------------------------------------------------
// `epsilon` must be > 0. A row whose gradient is exactly zero is left
// unchanged (no direction, no perturbation) rather than producing NaN.
Tensor l2_fgsm(const Tensor& x, double epsilon, const GradientFn& grad);

// ---------------------------------------------------------------------------
// Configuration for adversarial training (the paper's outer problem).
//
// The defaults are the authors' own MNIST challenge scale: epsilon = 0.3 with
// 40 steps of size 0.01 (mnist_challenge/config.json). alpha is deliberately NOT
// tied to epsilon (D6) -- the paper's ratio between them is 30x.
// ---------------------------------------------------------------------------
struct PGDConfig {
    double epsilon = 0.3;
    double alpha = 0.01;
    int num_steps = 40;
    bool random_start = true;
    unsigned seed = 0;
    // Print per-epoch adversarial loss. Off by default so this is usable from a
    // library caller without polluting stdout.
    bool verbose = false;
};

// ---------------------------------------------------------------------------
// Train `model` against the PGD adversary in `cfg` (the outer minimisation of
// Eq. 2.1).
//
// Per sample: build the adversarial input, run one forward/backward on THAT
// input, then apply the SGD update. This is Danskin's theorem in practice --
// the gradient of the outer problem is taken at an approximate inner maximiser,
// which §3.3 notes is not strictly justified for a non-smooth model but is what
// the paper trains with and is what works.
//
// Uses softmax cross-entropy, matching `Model::train_cross_entropy`.
//
// Throws std::invalid_argument on: a model with no layers, X.rows != y.rows,
// an X/y column mismatch, or an invalid `cfg` (epsilon <= 0, alpha <= 0).
// `X` holds one example per row; `y` holds integer class indices in a column.
// ---------------------------------------------------------------------------
void adversarial_train(Model& model, const Tensor& X, const Tensor& y,
                       const PGDConfig& cfg, double learning_rate, int epochs);

#endif  // NN_UTILS_ADVERSARIAL_H