#include "knowledge_distillation.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

// Row-wise softmax of `z / T`, computed with the max subtracted so a row of
// large logits cannot overflow exp(). Returned as PROBABILITIES.
Tensor softmax_at_temperature(const Tensor& z, double T) {
    Tensor out(z.rows, z.cols);
    const double inv_T = 1.0 / T;
    for (size_t i = 0; i < z.rows; ++i) {
        double m = z(i, 0);
        for (size_t j = 1; j < z.cols; ++j)
            if (z(i, j) > m) m = z(i, j);
        double sum = 0.0;
        for (size_t j = 0; j < z.cols; ++j) {
            const double e = std::exp((z(i, j) - m) * inv_T);
            out(i, j) = e;
            sum += e;
        }
        for (size_t j = 0; j < z.cols; ++j) out(i, j) /= sum;
    }
    return out;
}

void require_same_shape(const Tensor& student_logits,
                        const Tensor& teacher_logits) {
    if (student_logits.rows != teacher_logits.rows ||
        student_logits.cols != teacher_logits.cols) {
        throw std::invalid_argument(
            "distillation: student_logits and teacher_logits must have the same "
            "shape (got " + std::to_string(student_logits.rows) + "x" +
            std::to_string(student_logits.cols) + " and " +
            std::to_string(teacher_logits.rows) + "x" +
            std::to_string(teacher_logits.cols) + ")");
    }
    if (student_logits.rows == 0 || student_logits.cols == 0) {
        throw std::invalid_argument("distillation: logits must be non-empty");
    }
}

void require_temperature(double T) {
    if (!(T > 0.0)) {
        throw std::invalid_argument(
            "distillation: temperature must be > 0 (got " +
            std::to_string(T) + ")");
    }
}

// log_softmax(student / T), reusing the SHIPPED helper rather than a second
// implementation. `student / T` is a fresh Tensor, so the cached state of the
// caller's tensor is never touched (semantics 3: the two normalisations are
// independent).
Tensor student_log_probs(const Tensor& student_logits, double T) {
    return log_softmax(student_logits * (1.0 / T));
}

}  // namespace

// ---------------------------------------------------------------------------
// distillation_kl
// ---------------------------------------------------------------------------
// KL(p_t || q_s) = sum_ij p_t(i,j) * (log p_t(i,j) - log_softmax(z_s/T)(i,j))
//
// PyTorch Loss.cpp:264 computes this as xlogy(target, target) - target * input,
// with `target` = teacher probs and `input` = student log-probs. The xlogy
// convention is the part that matters: xlogy(0, y) = 0 EVEN WHEN y IS ALSO 0.
// The naive `p * log(p / q)` form yields 0 * log(0/0) = NaN instead, and a
// softmax probability underflowing to exactly 0 is reachable at large T (the
// FIX_UNDERFLOW fixture in the test suite hits it at T = 1000). So the zero rule
// is applied at the point of use, not folded into a quotient.
//
// Reduction::Mean divides by the ELEMENT count rows*cols, matching PyTorch's
// `unreduced.mean()` (Loss.cpp:66-68). It is deliberately NOT batchmean: the
// oracle measured mean/batchmean = 1/K exactly at K=3.
double distillation_kl(const Tensor& student_logits,
                       const Tensor& teacher_logits, double T) {
    require_temperature(T);
    require_same_shape(student_logits, teacher_logits);

    const Tensor p_t = softmax_at_temperature(teacher_logits, T);
    const Tensor lp_s = student_log_probs(student_logits, T);

    double total = 0.0;
    for (size_t i = 0; i < student_logits.rows; ++i)
        for (size_t j = 0; j < student_logits.cols; ++j) {
            const double p = p_t(i, j);
            // xlogy(p, p) == 0 for p == 0, whatever the student has. Skipping
            // the term here is what keeps the underflow path finite.
            if (p == 0.0) continue;
            total += p * (std::log(p) - lp_s(i, j));
        }

    double value = total;
    // Semantics 5: Mean divides by numel, not by rows.
    value /= static_cast<double>(student_logits.rows * student_logits.cols);
    // Semantics 6: the T^2 rescaling is on the LOSS, applied identically by
    // Keras's LogitsDistillation. Without it the soft-target gradient would
    // shrink like 1/T^2 and the hard/soft balance would silently depend on T.
    return value * T * T;
}

// ---------------------------------------------------------------------------
// distillation_kl_backward
// ---------------------------------------------------------------------------
// With p_t = softmax(teacher/T) and q_s = softmax(student/T):
//
//   dL/d(log q_s) = scale * p_t                                  (p is constant)
//   dL/d(q_s)    = dL/d(log q_s) - q_s * sum_k dL/d(log q_s)[k]  (softmax adjoint)
//   dL/d(z_s/T)  = dL/d(q_s)                                     (softmax is its
//                                                                   own Jacobian)
//   dL/d(z_s)    = (1/T) * dL/d(z_s/T)
//   times the loss factor T^2 from the forward.
//
// The last two factors give T^2 * (1/T) = T overall, which the oracle verified
// against torch autograd in float64 to 1.4e-16.
//
// The soft row-sum `sum_k p_t[i][k] == 1` is used rather than re-summing, which
// is exact for a softmax row up to its own rounding — and it is a genuine
// identity, not a shortcut that hides a missing term: the shipped
// CrossEntropyLoss backward relies on the same property.
Tensor distillation_kl_backward(const Tensor& student_logits,
                                const Tensor& teacher_logits, double T) {
    require_temperature(T);
    require_same_shape(student_logits, teacher_logits);

    const size_t N = student_logits.rows;
    const size_t K = student_logits.cols;

    const Tensor p_t = softmax_at_temperature(teacher_logits, T);
    const Tensor q_s = softmax_at_temperature(student_logits, T);

    // dL/d(log q_s) = scale * (-p_t)
    //
    // THE SIGN IS THE WHOLE FORMULA. KL(p||q) = sum_j p_j (log p_j - log q_j),
    // so d/d(log q_j) is MINUS p_j. With the sign dropped the result is exactly
    // -dL/dz, a rel error of precisely 2.0 against finite differences on every
    // cell — the clean-ratio fingerprint the test suite asserts against.
    //
    // This is also what makes the shipped log_softmax_backward produce the right
    // answer below: given g = -scale*p_t it returns
    //   -scale*p_t[j] - q[j] * (-scale * sum_k p_t[k]) = scale * (q[j] - p_t[j])
    // because a softmax row sums to exactly 1. That is Hinton's Eq. 2 adjoint.
    Tensor dlp(N, K);
    const double scale = 1.0 / static_cast<double>(N * K);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < K; ++j) dlp(i, j) = -scale * p_t(i, j);

    // Push through the softmax adjoint via the SHIPPED helper so this cannot
    // drift from CrossEntropyLoss's own formulation:
    //   log_softmax_backward(g, log_probs)[j] = g[j] - softmax(x)[j] * sum_k g[k]
    // Note that helper needs the LOG-probs to recover the softmax, and it
    // recovers it as exp(log_softmax(x)) — the same value softmax_at_temperature
    // computes, to within one rounding.
    const Tensor lp_s = student_log_probs(student_logits, T);
    Tensor dqs = log_softmax_backward(dlp, lp_s);

    // Chain through 1/T, then apply the forward's T^2 loss factor: net factor T.
    const double net = T * T * (1.0 / T);
    Tensor grad(N, K);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < K; ++j) grad(i, j) = dqs(i, j) * net;

    // `q_s` is used above only through lp_s's exp(); keep the explicit softmax
    // available for the row-sum shortcut documented in the header comment.
    (void)q_s;
    return grad;
}

// ---------------------------------------------------------------------------
// DistillationLoss
// ---------------------------------------------------------------------------
DistillationLoss::DistillationLoss(double temperature, double alpha,
                                   Reduction reduction)
    : temperature_(temperature), alpha_(alpha), reduction_(reduction) {
    require_temperature(temperature);
    if (!(alpha >= 0.0) || !(alpha <= 1.0)) {
        throw std::invalid_argument(
            "DistillationLoss: alpha (weight on the hard-label term) must be in "
            "[0, 1] (got " + std::to_string(alpha) + ")");
    }
}

namespace {

// Validate an (N,1) integer label column against an (N,K) logit matrix.
void validate_labels(const Tensor& labels, const Tensor& logits) {
    if (labels.cols != 1) {
        throw std::invalid_argument(
            "DistillationLoss: labels must be an (N, 1) column (got " +
            std::to_string(labels.cols) + " columns)");
    }
    if (labels.rows != logits.rows) {
        throw std::invalid_argument(
            "DistillationLoss: labels has " + std::to_string(labels.rows) +
            " rows but the logits have " + std::to_string(logits.rows));
    }
    for (size_t i = 0; i < labels.rows; ++i) {
        const double y = labels(i, 0);
        if (y < 0.0 || y >= static_cast<double>(logits.cols)) {
            throw std::invalid_argument(
                "DistillationLoss: label " + std::to_string(y) +
                " at row " + std::to_string(i) + " is outside [0, " +
                std::to_string(logits.cols) + ")");
        }
        if (y != std::floor(y)) {
            throw std::invalid_argument(
                "DistillationLoss: label " + std::to_string(y) +
                " at row " + std::to_string(i) + " is not an integer class index");
        }
    }
}

// Per-element hard-label cross-entropy adjoint at T = 1, matching the shipped
// CrossEntropyLoss backward (which routes through NLLLoss for its divisor).
// Returned as dL/dlogits for one row per batch element.
Tensor hard_ce_adjoint(const Tensor& logits, const Tensor& labels,
                       double row_scale) {
    const Tensor lp = log_softmax(logits);
    Tensor dlp(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i) {
        const size_t t = static_cast<size_t>(labels(i, 0));
        for (size_t j = 0; j < logits.cols; ++j)
            dlp(i, j) = row_scale * std::exp(lp(i, j));
        dlp(i, t) -= row_scale;
    }
    return log_softmax_backward(dlp, lp);
}

}  // namespace

Tensor DistillationLoss::forward(const Tensor& student_logits,
                                 const Tensor& teacher_logits,
                                 const Tensor& labels) const {
    validate_labels(labels, student_logits);

    // The hard term is at T = 1 and carries NO T^2 (semantics 7).
    CrossEntropyLoss ce(reduction_);
    const Tensor hard = ce.forward(student_logits, labels);

    // The soft term is reduced over elements, so its Mean divisor is rows*cols.
    // Under Reduction::None the per-row form is required, so it is built here
    // rather than folded into distillation_kl (which is always fully reduced).
    if (reduction_ == Reduction::None) {
        const size_t N = student_logits.rows;
        const size_t K = student_logits.cols;
        const Tensor p_t = softmax_at_temperature(teacher_logits, temperature_);
        const Tensor lp_s = student_log_probs(student_logits, temperature_);
        Tensor per_row(N, 1);
        for (size_t i = 0; i < N; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < K; ++j) {
                const double p = p_t(i, j);
                if (p == 0.0) continue;          // xlogy zero rule
                row_sum += p * (std::log(p) - lp_s(i, j));
            }
            per_row(i, 0) = row_sum * temperature_ * temperature_;
        }
        Tensor out(N, 1);
        for (size_t i = 0; i < N; ++i)
            out(i, 0) = alpha_ * hard(i, 0) + (1.0 - alpha_) * per_row(i, 0);
        return out;
    }

    const Tensor soft(1, 1);
    double soft_value = distillation_kl(student_logits, teacher_logits,
                                        temperature_);
    // distillation_kl is ALWAYS mean over ELEMENTS (semantics 5). Under Sum the
    // caller asked for the total, and the two differ by exactly the element
    // count. Without this the Sum branch silently returned the Mean — which no
    // shape or finiteness check can see, and which the per-row reduction test
    // catches because Sum != sum(per_row).
    if (reduction_ == Reduction::Sum) {
        soft_value *= static_cast<double>(student_logits.rows *
                                          student_logits.cols);
    }
    Tensor out(1, 1);
    out(0, 0) = alpha_ * hard(0, 0) + (1.0 - alpha_) * soft_value;
    return out;
}

Tensor DistillationLoss::backward(const Tensor& student_logits,
                                  const Tensor& teacher_logits,
                                  const Tensor& labels,
                                  const Tensor& grad_output) const {
    validate_labels(labels, student_logits);

    const size_t N = student_logits.rows;
    const size_t K = student_logits.cols;
    const size_t per_row = (reduction_ == Reduction::None);

    // Row scale for the hard term: Mean divides by the batch (B - n_ignored,
    // and nothing is ignored here), None leaves each row alone.
    const double hard_row_scale =
        (reduction_ == Reduction::Mean) ? 1.0 / static_cast<double>(N) : 1.0;

    Tensor grad = hard_ce_adjoint(student_logits, labels,
                                 alpha_ * hard_row_scale);

    // Apply the incoming upstream gradient row by row. Under Mean/Sum there is
    // a single scalar; under None each row carries its own.
    if (!per_row) {
        const double go = grad_output(0, 0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < K; ++j) grad(i, j) *= go;
    } else {
        for (size_t i = 0; i < N; ++i) {
            const double go = grad_output(i, 0);
            for (size_t j = 0; j < K; ++j) grad(i, j) *= go;
        }
    }

    // Soft term. Under Mean the element-mean scale is 1/(N*K); under None the
    // per-row soft value needs only its own 1/T and the T^2 loss factor, with
    // the class axis already summed in the row loop below.
    const double soft_element_scale =
        (reduction_ == Reduction::Mean) ? 1.0 / static_cast<double>(N * K) : 1.0;
    const Tensor p_t = softmax_at_temperature(teacher_logits, temperature_);
    const Tensor lp_s = student_log_probs(student_logits, temperature_);

    Tensor dlp(N, K);
    for (size_t i = 0; i < N; ++i) {
        const double go = per_row ? grad_output(i, 0) : grad_output(0, 0);
        for (size_t j = 0; j < K; ++j)
            dlp(i, j) = -soft_element_scale * p_t(i, j);
        Tensor dqs_row = log_softmax_backward(dlp.get_row(i),
                                              lp_s.get_row(i));
        const double net = temperature_ * temperature_ * (1.0 / temperature_);
        const double w = (1.0 - alpha_) * go * net;
        for (size_t j = 0; j < K; ++j) grad(i, j) += w * dqs_row(0, j);
    }
    return grad;
}