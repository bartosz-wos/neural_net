#include "vat.h"

#include "../activations/activations.h"   // Tanh (not used here, but keeps the
                                           // activation header's log_softmax
                                           // contract aligned with ours)
#include "losses.h"                        // CrossEntropyLoss

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// ============================================================================
// Virtual Adversarial Training -- Miyato et al., arXiv:1704.03976 (ICLR 2018).
//
// The derivations live in the header (D1..D11). Two of them are worth
// restating here because they are the lines an implementer gets wrong, and both
// were caught by the test suite's own finite-difference checks:
//
//   the KL ARGUMENT ORDER (D1), which is clean-first and not symmetric:
//       KL(clean || adv) = 0.32342102212705642
//       KL(adv  || clean) = 0.32363019005210997
//
//   and the CLEAN BRANCH DETACHMENT (D2), whose gradient is (p - q)/N:
//       (p - q)/N  ->  FD agrees
//       (q - p)/N  ->  FD disagrees at rel_err ~ 2.0  (an exact sign flip)
// ============================================================================

namespace {

void require(bool ok, const std::string& msg) {
    if (!ok) throw std::invalid_argument(msg);
}

void validate_config(const VATConfig& cfg) {
    require(cfg.epsilon > 0.0,
            "vat: epsilon must be > 0, got " + std::to_string(cfg.epsilon) +
            " (it is the radius of the virtual-adversarial ball)");
    require(cfg.xi > 0.0,
            "vat: xi must be > 0, got " + std::to_string(cfg.xi) +
            " (it is the finite-difference step; it must be << epsilon)");
    require(cfg.power_iterations >= 0,
            "vat: power_iterations must be >= 0, got " +
            std::to_string(cfg.power_iterations));
}

// Max-shifted per-row log-softmax, the reference's form (layers.py:131-135).
// Shifting by the row max is what makes exp() safe for large logits; it is not
// an approximation.
Tensor log_softmax_rows(const Tensor& logits) {
    Tensor out(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i) {
        double mx = logits(i, 0);
        for (size_t j = 1; j < logits.cols; ++j) mx = std::max(mx, logits(i, j));
        double sum = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) sum += std::exp(logits(i, j) - mx);
        const double log_sum = mx + std::log(sum);
        for (size_t j = 0; j < logits.cols; ++j) out(i, j) = logits(i, j) - log_sum;
    }
    return out;
}

Tensor softmax_rows(const Tensor& logits) {
    Tensor ls = log_softmax_rows(logits);
    Tensor out(ls.rows, ls.cols);
    for (size_t i = 0; i < ls.rows; ++i)
        for (size_t j = 0; j < ls.cols; ++j) out(i, j) = std::exp(ls(i, j));
    return out;
}

void validate_logits_pair(const Tensor& a, const Tensor& b) {
    require(a.rows == b.rows && a.cols == b.cols,
            "vat: logit tensors must have the same shape, got (" +
            std::to_string(a.rows) + ", " + std::to_string(a.cols) + ") and (" +
            std::to_string(b.rows) + ", " + std::to_string(b.cols) + ")");
    require(a.rows > 0 && a.cols > 0, "vat: logit tensors must be non-empty");
}

}  // namespace

// ---------------------------------------------------------------------------
Tensor vat_log_softmax(const Tensor& logits) {
    require(logits.rows > 0 && logits.cols > 0,
            "vat_log_softmax: input must be non-empty");
    return log_softmax_rows(logits);
}

Tensor vat_softmax(const Tensor& logits) {
    require(logits.rows > 0 && logits.cols > 0,
            "vat_softmax: input must be non-empty");
    return softmax_rows(logits);
}

// ---------------------------------------------------------------------------
// D1: the ORDER. `clean` is the FIRST argument of the divergence, which is the
// whole point -- KL is asymmetric, and the reference's confusingly-named
// parameters (`q_logit` is the clean logit) hide that from a reader.
Tensor kl_divergence_per_sample(const Tensor& clean_logits,
                                const Tensor& perturbed_logits) {
    validate_logits_pair(clean_logits, perturbed_logits);
    Tensor q = softmax_rows(clean_logits);      // the reference's p(y|x)
    Tensor ls_clean = log_softmax_rows(clean_logits);
    Tensor ls_pert = log_softmax_rows(perturbed_logits);
    Tensor out(clean_logits.rows, 1);
    for (size_t i = 0; i < clean_logits.rows; ++i) {
        double acc = 0.0;
        for (size_t k = 0; k < clean_logits.cols; ++k) {
            // p * (log p - log q), with p's own log taken from the SAME
            // max-shifted expression, so KL(p||p) cancels to exactly 0.
            acc += q(i, k) * (ls_clean(i, k) - ls_pert(i, k));
        }
        out(i, 0) = acc;
    }
    return out;
}

double kl_divergence(const Tensor& clean_logits, const Tensor& perturbed_logits) {
    Tensor per = kl_divergence_per_sample(clean_logits, perturbed_logits);
    double sum = 0.0;
    for (size_t i = 0; i < per.rows; ++i) sum += per(i, 0);
    return sum / static_cast<double>(per.rows);
}

// ---------------------------------------------------------------------------
// D2: d/d(perturbed logits) of the BATCH-MEAN KL, with `clean` held FIXED
// (the reference's `tf.stop_gradient` at vat.py:55).  = (p_k - q_k) / N.
Tensor dkl_divergence_dlogits(const Tensor& clean_logits,
                              const Tensor& perturbed_logits) {
    validate_logits_pair(clean_logits, perturbed_logits);
    Tensor q = softmax_rows(clean_logits);
    Tensor p = softmax_rows(perturbed_logits);
    const double inv_n = 1.0 / static_cast<double>(clean_logits.rows);
    Tensor out(perturbed_logits.rows, perturbed_logits.cols);
    for (size_t i = 0; i < perturbed_logits.rows; ++i)
        for (size_t k = 0; k < perturbed_logits.cols; ++k)
            out(i, k) = (p(i, k) - q(i, k)) * inv_n;
    return out;
}

// ---------------------------------------------------------------------------
// D6, D10, D11: the reference's guarded two-step per-row normalisation.
Tensor normalise_direction(const Tensor& d) {
    require(d.rows > 0 && d.cols > 0,
            "normalise_direction: input must be non-empty");
    Tensor out(d.rows, d.cols);
    for (size_t i = 0; i < d.rows; ++i) {
        double max_abs = 0.0;
        for (size_t j = 0; j < d.cols; ++j) max_abs = std::max(max_abs, std::fabs(d(i, j)));
        // The first divide is an overflow guard that cancels with the second
        // (D11); a zero row must not become 0/0 here.
        if (max_abs == 0.0) continue;                 // leaves the row exactly 0
        const double inv_first = 1.0 / (1e-12 + max_abs);
        double sq = 0.0;
        for (size_t j = 0; j < d.cols; ++j) {
            const double v = d(i, j) * inv_first;
            out(i, j) = v;
            sq += v * v;
        }
        const double inv_second = 1.0 / std::sqrt(1e-6 + sq);   // NOT a no-op (D6)
        for (size_t j = 0; j < d.cols; ++j) out(i, j) *= inv_second;
    }
    return out;
}

// ---------------------------------------------------------------------------
DirectionFn VATConfig::gaussian_direction(unsigned seed) {
    // Algorithm 1 step 2: iid Gaussian, one draw over the whole tensor. The
    // seed is explicit because std::normal_distribution is not bit-reproducible
    // across standard libraries -- within THIS library's build it is, so a
    // caller can reproduce a run exactly by passing the same seed.
    return [seed](const Tensor& x) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        Tensor d(x.rows, x.cols);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j) d(i, j) = dist(gen);
        return d;
    };
}

DirectionFn VATConfig::default_direction() { return gaussian_direction(0); }

// ---------------------------------------------------------------------------
// The power iteration (D3, D5).
//
// `grad` is d/d(input) of the batch-mean KL(clean || forward(.)) with the clean
// logits held fixed. That is exactly `GradientFn` from `adversarial.h` -- the
// same callback shape -- so one factory can serve both this and the shipped PGD
// attacker. It receives the FULL input (x + r), not the displacement r, which
// is what makes D4 hold: the KL's gradient is exactly zero at x, and only
// exactly zero when the probe is evaluated AT x.
Tensor virtual_adversarial_perturbation(const Tensor& x, const LogitsFn& forward,
                                        const GradientFn& grad,
                                        const VATConfig& cfg,
                                        const Tensor& clean_logits) {
    validate_config(cfg);
    require(static_cast<bool>(forward), "vat: a LogitsFn is required");
    require(static_cast<bool>(grad), "vat: a GradientFn is required");
    require(x.rows > 0 && x.cols > 0, "vat: the input batch must be non-empty");

    // D5: the clean prediction is computed ONCE, outside the loop, and is a
    // plain Tensor (no tape, nothing to detach later) -- so a caller cannot
    // accidentally differentiate through the reference branch.
    Tensor cached = (clean_logits.rows == x.rows && clean_logits.cols > 0)
                        ? clean_logits
                        : forward(x);

    DirectionFn dir = cfg.direction;
    if (!static_cast<bool>(dir)) dir = VATConfig::gaussian_direction(0);

    Tensor d = dir(x);
    if (d.rows != x.rows || d.cols != x.cols) {
        throw std::invalid_argument(
            "vat: the direction function returned shape (" +
            std::to_string(d.rows) + ", " + std::to_string(d.cols) +
            ") for an input of shape (" + std::to_string(x.rows) + ", " +
            std::to_string(x.cols) + ")");
    }

    for (int k = 0; k < cfg.power_iterations; ++k) {
        // D3: the probe DISPLACEMENT is formed before the gradient step and
        // scaled by xi. The gradient is then evaluated at the FULL INPUT
        // x + probe -- the reference computes `forward(x + d)` (vat.py:45), not
        // `forward(d)`. Getting this wrong evaluates the KL at the ORIGIN
        // instead of at the probe, and the resulting r_vadv does not match the
        // reference at all (measured: rows 1 and 2 come out sign-flipped).
        Tensor probe = normalise_direction(d);
        for (size_t i = 0; i < probe.rows; ++i)
            for (size_t j = 0; j < probe.cols; ++j) probe(i, j) *= cfg.xi;

        // The reference's `tf.stop_gradient(grad)` (vat.py:48): the direction
        // found here is a CONSTANT for the rest of the iteration, so the next
        // round re-normalises it rather than differentiating through it.
        d = grad(x + probe);

        if (d.rows != x.rows || d.cols != x.cols) {
            throw std::invalid_argument(
                "vat: the gradient function returned shape (" +
                std::to_string(d.rows) + ", " + std::to_string(d.cols) +
                "), which does not match the input shape (" +
                std::to_string(x.rows) + ", " + std::to_string(x.cols) + ")");
        }
    }

    // D3/D6: eps is applied AFTER the loop, and the result is normalised, so
    // ||r_vadv|| = eps * 0.9999997 rather than eps.
    Tensor r = normalise_direction(d);
    for (size_t i = 0; i < r.rows; ++i)
        for (size_t j = 0; j < r.cols; ++j) r(i, j) *= cfg.epsilon;
    return r;
}

// ---------------------------------------------------------------------------
double vat_loss(const Tensor& x, const LogitsFn& forward,
                const GradientFn& grad, const VATConfig& cfg,
                const Tensor& clean_logits) {
    Tensor r = virtual_adversarial_perturbation(x, forward, grad, cfg, clean_logits);
    Tensor clean = (clean_logits.rows == x.rows && clean_logits.cols > 0)
                       ? clean_logits
                       : forward(x);
    // A FRESH forward at the virtual-adversarial point: the probe logits are
    // not reused, because r_vadv and the probe are different points.
    return kl_divergence(clean, forward(x + r));
}

// ---------------------------------------------------------------------------
// D8: NEGATIVE entropy. The name says what it returns; adding it under
// gradient descent therefore MINIMISES entropy (sharpens predictions).
//
// THE MINUS IS THE WHOLE POINT. The reference's `entropy_y_x`
// (layers.py:145-148) returns `-mean_i sum_j p_ij log p_ij`, so a UNIFORM
// distribution gives +log(K) and a confident one gives ~0 from BELOW. Writing
// `+mean sum p log p` -- the natural thing to type -- returns -log(K) for
// uniform, and every sign-sensitive assertion inverts.
double entropy_negative(const Tensor& logits) {
    require(logits.rows > 0 && logits.cols > 0,
            "entropy_negative: input must be non-empty");
    Tensor p = softmax_rows(logits);
    Tensor ls = log_softmax_rows(logits);
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i)
        for (size_t k = 0; k < logits.cols; ++k)
            total += p(i, k) * ls(i, k);          // sum_i sum_k p log p  (negative)
    return -total / static_cast<double>(logits.rows);      // <-- negated
}

// d/d(logits) of `entropy_negative`, for the optional entropy-minimisation term.
//
//      E = -(1/N) sum_i sum_k p_ik log p_ik
//      dE/dz_ik = -(1/N) * p_ik * (log p_ik + H_i),   H_i = -sum_k p_ik log p_ik
//
// The MINUS is not a matter of taste. Measured by centred finite differences on
// a random (4,3) logit matrix: the un-negated form p*(log p + H) disagrees at
// rel_err EXACTLY 2.0 -- the signature of an inverted gradient, the same
// fingerprint the dKL/dlogits sign flip produced (D2).
Tensor d_entropy_negative_dlogits(const Tensor& logits) {
    require(logits.rows > 0 && logits.cols > 0,
            "d_entropy_negative_dlogits: input must be non-empty");
    Tensor p = softmax_rows(logits);
    Tensor ls = log_softmax_rows(logits);
    const double inv_n = 1.0 / static_cast<double>(logits.rows);
    Tensor out(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i) {
        double H = 0.0;
        for (size_t k = 0; k < logits.cols; ++k) H -= p(i, k) * ls(i, k);
        for (size_t k = 0; k < logits.cols; ++k)
            out(i, k) = -p(i, k) * (ls(i, k) + H) * inv_n;
    }
    return out;
}

// ---------------------------------------------------------------------------
VATTrainer::VATTrainer(Model& model) : model_(model) {}

void VATTrainer::train(const Tensor& X, const Tensor& y, const Config& cfg,
                       std::vector<double>* epoch_losses) {
    require(!model_.layers.empty(),
            "VATTrainer: the model has no layers, so there is nothing to train");
    require(X.rows == y.rows,
            "VATTrainer: X.rows (" + std::to_string(X.rows) +
            ") must equal y.rows (" + std::to_string(y.rows) + ")");
    require(X.cols > 0 && y.cols > 0, "VATTrainer: X and y must be non-empty");
    require(X.rows > 0, "VATTrainer: X has no rows");
    require(cfg.epochs >= 0, "VATTrainer: epochs must be >= 0");
    require(cfg.alpha >= 0.0,
            "VATTrainer: alpha must be >= 0, got " + std::to_string(cfg.alpha));
    validate_config(cfg.vat);

    CrossEntropyLoss ce;
    Tensor go(1, 1);
    go(0, 0) = 1.0;

    // The direction is seeded once per run so a whole training run is
    // reproducible from `cfg.seed`.
    VATConfig vat_cfg = cfg.vat;
    if (!static_cast<bool>(vat_cfg.direction))
        vat_cfg.direction = VATConfig::gaussian_direction(cfg.seed);

    LogitsFn forward = [this](const Tensor& t) { return model_.forward(t); };

    // d/d(input) of the batch-mean KL(clean || forward(.)), with `clean` held
    // FIXED (D2). This is the `GradientFn` the power iteration consumes, and it
    // is the same callback shape `adversarial.h` uses.
    auto make_kl_gradient = [this, &forward](const Tensor& clean) {
        return [this, &forward, clean](const Tensor& t) {
            Tensor logits = forward(t);
            Tensor dl = dkl_divergence_dlogits(clean, logits);
            for (auto& layer : model_.layers) layer->zero_grad();
            return model_.backward(dl, 0.0);
        };
    };

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        double total = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            const Tensor sample(1, X.cols, &X.data[i * X.cols]);
            const Tensor label(1, 1, &y.data[i * y.cols]);

            // Gradients must not survive between steps. The VAT path runs
            // several backward passes per sample, so this is not optional.
            for (auto& layer : model_.layers) layer->zero_grad();

            Tensor logits = model_.forward(sample);
            const double ce_loss = ce.forward(logits, label)[0][0];
            double step_loss = ce_loss;

            // The SUPERVISED branch. This gradient is deliberately NOT
            // zero_grad'd before the VAT branch runs -- the two terms are
            // summed, exactly as the reference sums nll_loss + vat_loss
            // (vat.py:68), so the parameter gradients ACCUMULATE.
            Tensor d_logits = ce.backward(logits, label, go);
            model_.backward(d_logits, 0.0);

            if (cfg.use_vat && cfg.alpha > 0.0) {
                const Tensor clean = logits;
                Tensor r = virtual_adversarial_perturbation(
                    sample, forward, make_kl_gradient(clean), vat_cfg, clean);

                // The outer VAT term. NOTE this is a FRESH forward at
                // x + r_vadv: the probe logits are a different point and must
                // not be reused.
                Tensor adv_logits = model_.forward(sample + r);
                step_loss += cfg.alpha * kl_divergence(clean, adv_logits);

                // D2: the gradient flows only through the PERTURBED branch,
                // since the clean one is a constant. The power-iteration
                // gradient above was already zero_grad'd away, so this backward
                // adds the VAT term on top of the supervised gradient.
                Tensor d_adv = dkl_divergence_dlogits(clean, adv_logits);
                for (size_t r0 = 0; r0 < d_adv.rows; ++r0)
                    for (size_t k = 0; k < d_adv.cols; ++k)
                        d_adv(r0, k) *= cfg.alpha;
                model_.backward(d_adv, 0.0);
            }

            if (cfg.use_entropy_min) {
                // D8: entropy_y_x is NEGATIVE entropy, so ADDING it under
                // gradient descent sharpens the predictions.
                const Tensor clean = model_.forward(sample);
                step_loss += cfg.entropy_weight * entropy_negative(clean);
                Tensor d_ent = d_entropy_negative_dlogits(clean);
                for (size_t r0 = 0; r0 < d_ent.rows; ++r0)
                    for (size_t k = 0; k < d_ent.cols; ++k)
                        d_ent(r0, k) *= cfg.entropy_weight;
                model_.backward(d_ent, 0.0);
            }

            for (auto& layer : model_.layers) layer->update_weights(cfg.learning_rate);
            for (auto& layer : model_.layers) layer->zero_grad();
            total += step_loss;
        }

        last_loss_ = total / static_cast<double>(X.rows);
        if (epoch_losses) epoch_losses->push_back(last_loss_);

        if (cfg.vat.verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs
                      << " VAT loss: " << last_loss_ << std::endl;
        }
    }
}