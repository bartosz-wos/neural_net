#include "losses.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// ============================================================================
// Canonical classification losses.
//
// Semantics verified against PyTorch source on 2026-10-11, not from memory:
//   aten/src/ATen/native/LossNLL.cpp:166-305   nll_loss_out_frame
//   aten/src/ATen/native/LossNLL.cpp:345-416   nll_loss_backward_out_frame
//   aten/src/ATen/native/Loss.cpp:349-364      binary_cross_entropy_with_logits
// ============================================================================

Tensor log_softmax(const Tensor& x) {
    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double mx = x[i][0];
        for (size_t j = 0; j < x.cols; ++j) mx = std::max(mx, x[i][j]);
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) sum += std::exp(x[i][j] - mx);
        const double lse = mx + std::log(sum);
        for (size_t j = 0; j < x.cols; ++j) out[i][j] = x[i][j] - lse;
    }
    return out;
}

Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& log_probs) {
    // dL/dx_j = g_j - softmax(x)_j * sum_k g_k   (the sum is over the row)
    Tensor out(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < grad_output.cols; ++j) sum += grad_output[i][j];
        for (size_t j = 0; j < grad_output.cols; ++j) {
            out[i][j] = grad_output[i][j] - std::exp(log_probs[i][j]) * sum;
        }
    }
    return out;
}

// Shared validation: every class index must be in range unless ignored (D8).
static void validate_targets(const Tensor& targets, size_t n_rows,
                             size_t n_classes, int ignore_index,
                             const char* who) {
    if (targets.rows != n_rows)
        throw std::invalid_argument(std::string(who) +
            ": targets must have one row per input row");
    for (size_t b = 0; b < n_rows; ++b) {
        const int t = static_cast<int>(targets[b][0]);
        if (t == ignore_index) continue;
        if (t < 0 || static_cast<size_t>(t) >= n_classes)
            throw std::invalid_argument(std::string(who) +
                ": target " + std::to_string(t) + " is out of bounds for " +
                std::to_string(n_classes) + " classes");
    }
}

static Tensor reduce_scalar(double value, Reduction r, size_t n_rows) {
    (void)n_rows;
    Tensor out(1, 1);
    out[0][0] = value;
    (void)r;
    return out;
}

// ---------------------------------------------------------------------------
// NLLLoss
// ---------------------------------------------------------------------------
NLLLoss::NLLLoss(Reduction r, int ignore_index)
    : reduction_(r), ignore_index_(ignore_index) {}

void NLLLoss::set_weight(const Tensor& weight) {
    // An unset weight is the default-constructed empty Tensor; treat it as
    // "no class weights" rather than a malformed one.
    if (weight.rows == 0 || weight.cols == 0) { weight_ = Tensor(); return; }
    if (weight.rows != 1)
        throw std::invalid_argument("NLLLoss: weight must be a (1, C) row vector");
    weight_ = weight.clone();
}

double NLLLoss::class_weight(size_t cls) const {
    if (weight_.rows == 0 || weight_.cols == 0) return 1.0;
    return weight_[0][cls];
}

void NLLLoss::compute_total_weight(const Tensor& targets) const {
    // LossNLL.cpp:288-292 — the divisor is the batch size MINUS the ignored
    // rows, or the summed class weights when a weight tensor is supplied.
    // Using the plain batch size here is the bug this class exists to avoid.
    if (weight_.rows == 0 || weight_.cols == 0) {
        size_t kept = 0;
        for (size_t b = 0; b < targets.rows; ++b)
            if (static_cast<int>(targets[b][0]) != ignore_index_) ++kept;
        total_weight_ = static_cast<double>(kept);
    } else {
        double s = 0.0;
        for (size_t b = 0; b < targets.rows; ++b) {
            const int t = static_cast<int>(targets[b][0]);
            if (t == ignore_index_) continue;
            s += class_weight(static_cast<size_t>(t));
        }
        total_weight_ = s;
    }
}

Tensor NLLLoss::forward(const Tensor& log_probs, const Tensor& targets) const {
    validate_targets(targets, log_probs.rows, log_probs.cols, ignore_index_, "NLLLoss");
    compute_total_weight(targets);

    double total = 0.0;
    Tensor per_row(reduction_ == Reduction::None ? targets.rows : 1, 1);
    for (size_t b = 0; b < log_probs.rows; ++b) {
        const int t = static_cast<int>(targets[b][0]);
        const double v = (t == ignore_index_)
            ? 0.0
            : -log_probs[b][static_cast<size_t>(t)] * class_weight(static_cast<size_t>(t));
        total += v;
        if (reduction_ == Reduction::None) per_row[b][0] = v;
    }
    if (reduction_ == Reduction::None) return per_row;
    // Mean divides by total_weight; Sum does not divide at all.
    const double denom = (reduction_ == Reduction::Mean) ? total_weight_ : 1.0;
    return reduce_scalar(total / denom, reduction_, log_probs.rows);
}

Tensor NLLLoss::backward(const Tensor& log_probs, const Tensor& targets,
                         const Tensor& grad_output) const {
    validate_targets(targets, log_probs.rows, log_probs.cols, ignore_index_, "NLLLoss");
    compute_total_weight(targets);

    Tensor g(log_probs.rows, log_probs.cols);
    g.fill(0.0);
    // LossNLL.cpp:376-379 — the gradient lands on the target column only, and
    // an ignored row keeps an EXACTLY zero row (not merely a small one).
    for (size_t b = 0; b < log_probs.rows; ++b) {
        const int t = static_cast<int>(targets[b][0]);
        if (t == ignore_index_) continue;
        const double go = (reduction_ == Reduction::None)
            ? grad_output[b][0]
            : grad_output[0][0];
        const double denom = (reduction_ == Reduction::Mean) ? total_weight_ : 1.0;
        g[b][static_cast<size_t>(t)] =
            -class_weight(static_cast<size_t>(t)) * go / denom;
    }
    return g;
}

// ---------------------------------------------------------------------------
// CrossEntropyLoss — softmax cross-entropy over raw logits (D4)
// ---------------------------------------------------------------------------
CrossEntropyLoss::CrossEntropyLoss(Reduction r, int ignore_index)
    : reduction_(r), ignore_index_(ignore_index) {}

void CrossEntropyLoss::set_weight(const Tensor& weight) {
    NLLLoss nll(reduction_, ignore_index_);
    nll.set_weight(weight);
    weight_ = weight.clone();
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) const {
    NLLLoss nll(reduction_, ignore_index_);
    nll.set_weight(weight_);
    return nll.forward(log_softmax(logits), targets);
}

Tensor CrossEntropyLoss::backward(const Tensor& logits, const Tensor& targets,
                                  const Tensor& grad_output) const {
    // Route through an NLLLoss so total_weight_ is computed by the SAME code
    // path as forward — a second, divergent copy of that arithmetic is how the
    // batch-size-vs-(B - n_ignored) divisor bug gets reintroduced.
    NLLLoss nll(reduction_, ignore_index_);
    nll.set_weight(weight_);
    nll.forward(logits, targets);   // side effect: populates total_weight

    const Tensor lp = log_softmax(logits);
    // dL/dlogits = log_softmax_backward(softmax - onehot, log_softmax).
    // The one-hot term is dropped where the target is ignored so the whole row
    // stays exactly zero (D2) — log_softmax is NOT zero there, so skipping this
    // would leak spurious gradient into padded rows.
    Tensor dlp(logits.rows, logits.cols);
    for (size_t b = 0; b < logits.rows; ++b) {
        const int t = static_cast<int>(targets[b][0]);
        const double go = (reduction_ == Reduction::None)
            ? grad_output[b][0]
            : grad_output[0][0];
        const double denom = (reduction_ == Reduction::Mean) ? nll.total_weight() : 1.0;
        // No class weight configured means w == 1, not 0 — defaulting to 0 here
        // silently zeroes the entire gradient, which is why an unweighted CE
        // backward must be read against `weight_.rows == 0`, not `weight_.empty()`.
        const double w = (t == ignore_index_) ? 0.0
            : (weight_.rows == 0 ? 1.0 : weight_[0][static_cast<size_t>(t)]);
        const double scale = go * w / denom;
        for (size_t j = 0; j < logits.cols; ++j) {
            dlp[b][j] = scale * std::exp(lp[b][j]);
        }
        if (t != ignore_index_) dlp[b][static_cast<size_t>(t)] -= scale;
    }
    total_weight_ = nll.total_weight();
    return log_softmax_backward(dlp, lp);
}

// ---------------------------------------------------------------------------
// BCEWithLogitsLoss (Loss.cpp:349-364)
// ---------------------------------------------------------------------------
BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction r) : reduction_(r) {}

void BCEWithLogitsLoss::set_weight(const Tensor& weight) {
    weight_ = weight.clone();
}

void BCEWithLogitsLoss::set_pos_weight(const Tensor& pos_weight) {
    pos_weight_ = pos_weight.clone();
}

// Stable pointwise loss and its exact adjoint, derived together.
//
//   L(z, t) = (1 - t) * z - logw * log_sigmoid(z)
//   logw    = (pos_weight - 1) * t + 1        (Loss.cpp:353)
//
// This form is stable for |z| up to 1e4 and needs no max-subtraction, because
// log_sigmoid(z) = -log1p(exp(-z)) is computed with exp(-z) — never exp(z) —
// and -log1p(exp(-z)) underflows harmlessly for large positive z (→ 0) while
// -log1p(exp(-|z|)) is used for the z < 0 branch, where exp(-z) would overflow.
// PyTorch writes the same expression as (1-target)*input - log_sigmoid(input)
// and relies on ATen's log_sigmoid for exactly this reason.
//
// dL/dz = (1 - t) - logw * (1 - sigmoid(z))          [1 - sigmoid(z) = sigmoid(-z)]
static inline double bce_point(double z, double t, double logw, double* dz) {
    const double lse = (z >= 0.0) ? std::log1p(std::exp(-z))
                                  : -z + std::log1p(std::exp(z));
    const double log_sigmoid = -lse;
    const double sig = (z >= 0.0) ? 1.0 / (1.0 + std::exp(-z))
                                  : std::exp(z) / (1.0 + std::exp(z));
    *dz = (1.0 - t) - logw * (1.0 - sig);
    return (1.0 - t) * z - logw * log_sigmoid;
}

// The logw multiplier for one element, 1.0 when pos_weight is unset.
static inline double bce_log_weight(double t, const Tensor& pos_weight) {
    if (pos_weight.rows == 0) return 1.0;
    return (pos_weight[0][0] - 1.0) * t + 1.0;
}

Tensor BCEWithLogitsLoss::forward(const Tensor& logits, const Tensor& targets) const {
    if (logits.rows != targets.rows || logits.cols != targets.cols)
        throw std::invalid_argument(
            "BCEWithLogitsLoss: logits and targets must have the same shape");
    const bool per_row = (reduction_ == Reduction::None);
    // Reduction::None reduces along the CLASS axis only (Loss.cpp:363 ->
    // apply_loss_reduction(None) = sum over class_dim), leaving one value per
    // row. Summing over the whole tensor instead would make a (2,3) input
    // return a (1,1) tensor indistinguishable from the fully reduced value.
    Tensor out(per_row ? logits.rows : 1, 1);
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        double row_total = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) {
            double dz = 0.0;
            const double logw = bce_log_weight(targets[i][j], pos_weight_);
            double v = bce_point(logits[i][j], targets[i][j], logw, &dz);
            if (weight_.rows > 0) v *= weight_[i % weight_.rows][j % weight_.cols];
            row_total += v;
            total += v;
        }
        if (per_row) out[i][0] = row_total;
    }
    if (per_row) return out;
    // D7: mean divides by the ELEMENT count (numel), not the row count.
    if (reduction_ == Reduction::Mean) total /= static_cast<double>(logits.rows * logits.cols);
    Tensor res(1, 1);
    res[0][0] = total;
    return res;
}

Tensor BCEWithLogitsLoss::backward(const Tensor& logits, const Tensor& targets,
                                   const Tensor& grad_output) const {
    const size_t numel = logits.rows * logits.cols;
    Tensor g(logits.rows, logits.cols);
    for (size_t i = 0; i < logits.rows; ++i) {
        for (size_t j = 0; j < logits.cols; ++j) {
            double dz = 0.0;
            const double logw = bce_log_weight(targets[i][j], pos_weight_);
            bce_point(logits[i][j], targets[i][j], logw, &dz);
            const double go = (reduction_ == Reduction::None)
                ? grad_output[i][0]
                : grad_output[0][0];
            double v = dz * go;
            if (reduction_ == Reduction::Mean) v /= static_cast<double>(numel);
            if (weight_.rows > 0) v *= weight_[i % weight_.rows][j % weight_.cols];
            g[i][j] = v;
        }
    }
    return g;
}