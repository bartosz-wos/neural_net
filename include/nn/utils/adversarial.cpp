#include "adversarial.h"

#include "../activations/activations.h"   // Softmax
#include "losses.h"                        // CrossEntropyLoss

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// ============================================================================
// Adversarial robustness — Madry et al., arXiv:1706.06083 (ICLR 2018).
//
// The derivations are in the header (D1..D9). The one place worth restating
// here is the PROJECTION, because it is the line an implementer gets wrong:
//
//     x <- clip(x, x_nat - eps, x_nat + eps)      <- centred on x_nat (D3)
//     x <- clip(x, x_nat + eps, x_nat + eps)      <- centred on x_t   : WRONG
//
// and the one that costs the most to debug silently:
//
//     sgn = g > 0 ? +1 : g < 0 ? -1 : 0            <- correct            (D2)
//     sgn = g >= 0 ? +1 : -1                       <- moves a zero by eps
// ============================================================================

namespace {

void validate_epsilon(double eps) {
    if (!(eps > 0.0)) {
        throw std::invalid_argument("epsilon must be > 0, got " + std::to_string(eps));
    }
}

void validate_range(const DataRange& r) {
    if (r.enabled && !(r.lo <= r.hi)) {
        throw std::invalid_argument(
            "data range must satisfy lo <= hi, got lo=" + std::to_string(r.lo) +
            " hi=" + std::to_string(r.hi));
    }
}

void validate_alpha(double a) {
    if (!(a > 0.0)) {
        throw std::invalid_argument("step size alpha must be > 0, got " + std::to_string(a));
    }
}

// A gradient of the wrong shape means the oracle is not the function the caller
// thinks it is. Taking `grad[i][j]` anyway would read out of bounds.
void validate_grad(const Tensor& g, const Tensor& x) {
    if (g.rows != x.rows || g.cols != x.cols) {
        throw std::invalid_argument(
            "gradient shape (" + std::to_string(g.rows) + ", " +
            std::to_string(g.cols) + ") does not match the input shape (" +
            std::to_string(x.rows) + ", " + std::to_string(x.cols) + ")");
    }
}

void validate_x(const Tensor& x) {
    if (x.rows == 0 || x.cols == 0) {
        throw std::invalid_argument("input must have at least one row and one column");
    }
}

double clamp01(double v, const DataRange& r) {
    if (!r.enabled) return v;
    return std::min(r.hi, std::max(r.lo, v));
}

}  // namespace

// ---------------------------------------------------------------------------
// D2. The three-way sign. Written as three separate comparisons rather than
// `g >= 0` on purpose: the `>=` version is the bug this header documents.
// ---------------------------------------------------------------------------
Tensor sign_tensor(const Tensor& x) {
    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            const double g = x[i][j];
            if (g > 0.0)      out[i][j] = 1.0;
            else if (g < 0.0) out[i][j] = -1.0;
            else              out[i][j] = 0.0;   // D2: no direction, no movement
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// FGSM. One ascent step of size eps.
// ---------------------------------------------------------------------------
Tensor linf_fgsm(const Tensor& x, double epsilon, const GradientFn& grad,
                 DataRange range) {
    validate_x(x);
    validate_epsilon(epsilon);
    validate_range(range);
    if (!grad) {
        throw std::invalid_argument("gradient function must not be empty");
    }

    const Tensor g = grad(x);
    validate_grad(g, x);
    const Tensor s = sign_tensor(g);   // D2

    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            // D1: ASCENT. The attack maximises the loss.
            out[i][j] = clamp01(x[i][j] + epsilon * s[i][j], range);   // D5
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// L-inf PGD. D3 (projection centred on x_nat) and D4 (step before project)
// are the load-bearing lines.
// ---------------------------------------------------------------------------
Tensor linf_pgd(const Tensor& x, double epsilon, double alpha, int num_steps,
                const GradientFn& grad, DataRange range, bool random_start,
                unsigned seed) {
    validate_x(x);
    validate_epsilon(epsilon);
    validate_alpha(alpha);
    validate_range(range);
    if (num_steps < 0) {
        throw std::invalid_argument("num_steps must be >= 0, got " +
                                    std::to_string(num_steps));
    }
    if (!grad) {
        throw std::invalid_argument("gradient function must not be empty");
    }

    // D7: k = 0 is the identity attack, and it must not evaluate the gradient.
    if (num_steps == 0) {
        Tensor out = x;
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j) out[i][j] = clamp01(x[i][j], range);
        return out;
    }

    const Tensor& x_nat = x;   // the projection centre, fixed for all steps (D3)

    Tensor current(x.rows, x.cols);
    if (random_start) {
        // Reference implementation, in its order: draw U(-eps, eps), add, then
        // clip to the data range.
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-epsilon, epsilon);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j)
                current[i][j] = clamp01(x[i][j] + dist(rng), range);
    } else {
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < x.cols; ++j) current[i][j] = x[i][j];
    }

    for (int step = 0; step < num_steps; ++step) {
        const Tensor g = grad(current);
        validate_grad(g, current);
        const Tensor s = sign_tensor(g);   // D2

        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                // D4: the step comes FIRST ...
                double v = current[i][j] + alpha * s[i][j];   // D1: ascent

                // ... and THEN the projection onto the eps-ball around x_nat.
                const double centre = x_nat[i][j];   // D3: x_nat, NOT current
                if (v < centre - epsilon) v = centre - epsilon;
                if (v > centre + epsilon) v = centre + epsilon;

                // D5: the data-range clamp is last, and opt-in.
                current[i][j] = clamp01(v, range);
            }
        }
    }
    return current;
}

// ---------------------------------------------------------------------------
// Random restarts, keeping the worst (highest-objective) attack.
// ---------------------------------------------------------------------------
Tensor random_restart_pgd(const Tensor& x, double epsilon, double alpha,
                          int num_steps, int num_restarts,
                          const GradientFn& grad, DataRange range,
                          unsigned seed,
                          std::function<double(const Tensor&)> objective) {
    validate_x(x);
    validate_epsilon(epsilon);
    validate_alpha(alpha);
    if (num_restarts < 1) {
        throw std::invalid_argument("num_restarts must be >= 1, got " +
                                    std::to_string(num_restarts));
    }
    if (!grad) {
        throw std::invalid_argument("gradient function must not be empty");
    }

    Tensor best;
    double best_score = 0.0;
    for (int r = 0; r < num_restarts; ++r) {
        // Each restart gets its own stream offset, so restart 2 is not a replay
        // of restart 1 when num_steps == 0.
        Tensor cand = linf_pgd(x, epsilon, alpha, num_steps, grad, range,
                               /*random_start=*/true, seed + 7919u * static_cast<unsigned>(r));
        if (r == 0) {
            best = cand;
            if (objective) best_score = objective(best);
            continue;
        }
        if (!objective) {
            best = cand;   // no objective: the last restart is the answer
            continue;
        }
        const double score = objective(cand);
        // "Worst" = the attack that raises the loss the most.
        if (score > best_score) {
            best = cand;
            best_score = score;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// L2 FGSM: normalise instead of sign (D9).
// ---------------------------------------------------------------------------
Tensor l2_fgsm(const Tensor& x, double epsilon, const GradientFn& grad) {
    validate_x(x);
    validate_epsilon(epsilon);
    if (!grad) {
        throw std::invalid_argument("gradient function must not be empty");
    }

    const Tensor g = grad(x);
    validate_grad(g, x);

    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double norm2 = 0.0;
        for (size_t j = 0; j < x.cols; ++j) norm2 += g[i][j] * g[i][j];
        const double norm = std::sqrt(norm2);
        if (norm == 0.0) {
            // D9: a zero gradient has no direction. Leave the row alone rather
            // than dividing by zero and emitting NaN into the training set.
            for (size_t j = 0; j < x.cols; ++j) out[i][j] = x[i][j];
            continue;
        }
        for (size_t j = 0; j < x.cols; ++j) {
            out[i][j] = x[i][j] + epsilon * g[i][j] / norm;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Adversarial training — the outer minimisation of Eq. 2.1.
// ---------------------------------------------------------------------------
void adversarial_train(Model& model, const Tensor& X, const Tensor& y,
                       const PGDConfig& cfg, double learning_rate, int epochs) {
    if (model.layers.empty()) {
        throw std::invalid_argument(
            "adversarial_train: the model has no layers, so there is nothing to attack");
    }
    if (X.rows != y.rows) {
        throw std::invalid_argument(
            "adversarial_train: X.rows (" + std::to_string(X.rows) +
            ") must equal y.rows (" + std::to_string(y.rows) + ")");
    }
    if (X.cols == 0 || y.cols == 0) {
        throw std::invalid_argument("adversarial_train: X and y must be non-empty");
    }
    if (X.rows == 0) {
        throw std::invalid_argument("adversarial_train: X has no rows");
    }
    validate_epsilon(cfg.epsilon);
    validate_alpha(cfg.alpha);
    if (cfg.num_steps < 0) {
        throw std::invalid_argument(
            "adversarial_train: num_steps must be >= 0, got " +
            std::to_string(cfg.num_steps));
    }

    CrossEntropyLoss ce;
    Tensor go(1, 1);
    go(0, 0) = 1.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            const Tensor sample(1, X.cols, &X.data[i * X.cols]);
            const Tensor label(1, 1, &y.data[i * y.cols]);

            // Inner problem: find the perturbation that maximises the loss.
            // The gradient is taken with respect to the INPUT, which is what
            // Model::backward returns. D8: no parameter is touched by this.
            const GradientFn input_grad = [&](const Tensor& cur) {
                Tensor logits = model.forward(cur);
                Tensor dl = ce.backward(logits, label, go);
                return model.backward(dl, 0.0);
            };

            Tensor adv = linf_pgd(sample, cfg.epsilon, cfg.alpha, cfg.num_steps,
                                  input_grad, DataRange(), cfg.random_start,
                                  cfg.seed + static_cast<unsigned>(i) +
                                      104729u * static_cast<unsigned>(epoch));

            // Outer problem: descend on the loss AT that perturbation.
            //
            // This uses the repo's CANONICAL CrossEntropyLoss (utils/losses.h),
            // which takes INTEGER class indices -- not the older
            // Softmax::cross_entropy_loss, which takes a one-hot matrix. The two
            // conventions coexist in this codebase; using the canonical one is
            // what makes `adversarial_train`'s documented label format true.
            // (The first version of this function used the one-hot helper and
            // consequently threw "Tensor dimensions mismatch" on every integer
            // label column.)
            Tensor logits = model.forward(adv);
            total_loss += ce.forward(logits, label)[0][0];

            Tensor d_logits = ce.backward(logits, label, go);   // d(CE)/d(logits)
            model.backward(d_logits, 0.0);
            for (auto& layer : model.layers) {
                layer->update_weights(learning_rate);
                layer->zero_grad();
            }
        }

        // D8: the attack pass above accumulates into grad_* too. Clear it, so a
        // caller's surrounding training loop never sees a stale gradient.
        for (auto& layer : model.layers) layer->zero_grad();

        if (cfg.verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " adversarial CE loss: "
                      << total_loss / static_cast<double>(X.rows) << std::endl;
        }
    }
}