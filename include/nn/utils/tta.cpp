#include "tta.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument(msg);
}

}  // namespace

// ---------------------------------------------------------------------------
// D1 -- bundle accessors.
//
// `Tensor` is strictly 2-D, so a (T, N, K) logit bundle is carried as
// (T, N*K), TRANSFORM-MAJOR: element (t, n*K + k) is the logit of class k for
// row n under transform t.
//
// Deriving loop bounds from the buffer rather than from a sibling quantity is
// deliberate. The alternative -- computing N as `bundle.cols / num_classes` and
// then indexing as if the buffer were N*T rows -- is how a 1-past-the-end read
// gets in, and `Tensor::operator()` does no bounds checking, so such a read is
// silent heap corruption rather than an exception.
// ---------------------------------------------------------------------------
size_t tta_num_transforms(const Tensor& bundle, size_t num_classes) {
    require(num_classes > 0, "tta_num_transforms: num_classes must be > 0");
    require(bundle.cols % num_classes == 0,
            "tta_num_transforms: bundle width must be a multiple of num_classes, "
            "got width " + std::to_string(bundle.cols) + " and num_classes " +
            std::to_string(num_classes));
    require(bundle.rows > 0, "tta_num_transforms: bundle must have rows");
    return bundle.rows;
}

Tensor tta_transform_view(const Tensor& bundle, size_t transform,
                          size_t num_classes) {
    require(num_classes > 0,
            "tta_transform_view: num_classes must be > 0");
    require(transform < bundle.rows,
            "tta_transform_view: transform " + std::to_string(transform) +
            " is out of range for a bundle with " +
            std::to_string(bundle.rows) + " transforms");
    require(bundle.cols % num_classes == 0,
            "tta_transform_view: bundle width must be a multiple of num_classes, "
            "got width " + std::to_string(bundle.cols) + " and num_classes " +
            std::to_string(num_classes));

    const size_t n = bundle.cols / num_classes;      // derived from the buffer
    Tensor out(n, num_classes);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < num_classes; ++j)
            out(i, j) = bundle(transform, i * num_classes + j);
    return out;
}

// ---------------------------------------------------------------------------
// D4 -- argmax, ties to the LOWEST class index.
//
// Strict `>` (not `>=`) matches the existing `argmax_row` in `metrics.cpp:38`,
// so a TTA prediction and a metric computed from the same tensor cannot disagree
// about a tie. A `>=` here would make the answer depend on iteration order.
// ---------------------------------------------------------------------------
std::vector<size_t> tta_argmax(const Tensor& t) {
    require(t.rows > 0 && t.cols > 0, "tta_argmax: tensor must be non-empty");
    std::vector<size_t> out(t.rows, 0);
    for (size_t i = 0; i < t.rows; ++i) {
        size_t best = 0;
        for (size_t j = 1; j < t.cols; ++j)
            if (t(i, j) > t(i, best)) best = j;
        out[i] = best;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Transforms.
//
// `x` is (N, H*W): a batch of flattened images, one image per row.
// ---------------------------------------------------------------------------
Tensor tta_identity(const Tensor& x) { return x.clone(); }

// D10 -- the flip reverses the WIDTH axis WITHIN each image row.
//
// Note the loop: `r` runs over IMAGES and `c` over the image's own columns. A
// whole-buffer reversal would run one loop over `cols` and would additionally
// reverse the order of the IMAGES, which is a different transform. The suite
// asserts the two are distinguishable rather than assuming this is obvious.
Tensor tta_hflip(const Tensor& x, size_t height, size_t width) {
    require(height > 0, "tta_hflip: height must be > 0");
    require(width > 0, "tta_hflip: width must be > 0");
    require(height * width == x.cols,
            "tta_hflip: height*width must equal the flattened width, got " +
            std::to_string(height) + "*" + std::to_string(width) + " != " +
            std::to_string(x.cols));

    Tensor out(x.rows, x.cols);
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t i = 0; i < height; ++i)
            for (size_t j = 0; j < width; ++j)
                out(r, i * width + (width - 1 - j)) = x(r, i * width + j);
    return out;
}

std::vector<Tensor> tta_augment(const Tensor& x, size_t height, size_t width) {
    std::vector<Tensor> views;
    views.push_back(tta_identity(x));
    views.push_back(tta_hflip(x, height, width));
    return views;
}

// ---------------------------------------------------------------------------
// D9 -- the flip report.
//
// The counts are SEPARATE fields, not a net number. The paper's headline is
// that a net improvement can still convert correct predictions into incorrect
// ones, so a report that exposes only the net change cannot see its own
// failure mode.
// ---------------------------------------------------------------------------
TtaFlipReport tta_flip_report(const Tensor& base, const Tensor& candidate,
                              const std::vector<int>& labels) {
    require(base.rows == candidate.rows,
            "tta_flip_report: base and candidate must have the same row count, "
            "got " + std::to_string(base.rows) + " and " +
            std::to_string(candidate.rows));
    require(labels.size() == base.rows,
            "tta_flip_report: labels size must equal the row count, got " +
            std::to_string(labels.size()) + " labels for " +
            std::to_string(base.rows) + " rows");
    require(base.cols == candidate.cols,
            "tta_flip_report: base and candidate must have the same column "
            "count, got " + std::to_string(base.cols) + " and " +
            std::to_string(candidate.cols));
    require(base.cols > 0, "tta_flip_report: prediction tensors must be non-empty");

    std::vector<size_t> pb = tta_argmax(base);
    std::vector<size_t> pc = tta_argmax(candidate);

    TtaFlipReport r;
    for (size_t i = 0; i < pb.size(); ++i) {
        if (pb[i] == pc[i]) continue;
        ++r.changed;
        // A changed row has exactly two possible outcomes, so
        // changed == c2i + i2c is an identity rather than a coincidence.
        const bool base_right = (pb[i] == static_cast<size_t>(labels[i]));
        if (base_right) ++r.correct_to_incorrect;
        else            ++r.incorrect_to_correct;
    }
    return r;
}

// ===========================================================================
// AGGREGATIONS
// ===========================================================================
namespace {

// Shared validation + weight normalisation (D8).
struct BundlePlan {
    size_t num_transforms;
    size_t num_rows;
    size_t num_classes;
    std::vector<double> w;          // normalised, sums to 1
};

BundlePlan plan_aggregation(const Tensor& bundle, size_t num_classes,
                            const std::vector<double>& weights) {
    require(num_classes > 0,
            "tta: num_classes must be > 0");
    require(bundle.rows > 0,
            "tta: the bundle must have at least one transform");
    require(bundle.cols % num_classes == 0,
            "tta: bundle width must be a multiple of num_classes, got width " +
            std::to_string(bundle.cols) + " and num_classes " +
            std::to_string(num_classes));

    // D8: T == 1 is not an aggregation. Returning the single member's logits
    // unchanged would silently hide a mis-wired pipeline that was supposed to
    // apply several transforms.
    require(bundle.rows > 1,
            "tta: aggregation requires at least 2 transforms, got " +
            std::to_string(bundle.rows) +
            " -- a single transform is not an aggregation");

    BundlePlan p;
    p.num_transforms = bundle.rows;
    p.num_classes = num_classes;
    p.num_rows = bundle.cols / num_classes;

    if (weights.empty()) {
        p.w.assign(p.num_transforms,
                   1.0 / static_cast<double>(p.num_transforms));
        return p;
    }

    require(weights.size() == p.num_transforms,
            "tta: weights must have one entry per transform, got " +
            std::to_string(weights.size()) + " for " +
            std::to_string(p.num_transforms) + " transforms");

    double total = 0.0;
    for (double v : weights) {
        require(v >= 0.0, "tta: weights must be non-negative");
        total += v;
    }
    require(total > 0.0, "tta: weights must not sum to zero");
    p.w.resize(weights.size());
    for (size_t i = 0; i < weights.size(); ++i)
        p.w[i] = weights[i] / total;
    return p;
}

// Numerically stable softmax of one row, written straight into `out`.
void softmax_row_into(const Tensor& bundle, size_t transform, size_t row,
                      size_t k, Tensor& out) {
    double mx = bundle(transform, row * k);
    for (size_t j = 1; j < k; ++j)
        mx = std::max(mx, bundle(transform, row * k + j));
    double sum = 0.0;
    for (size_t j = 0; j < k; ++j)
        sum += std::exp(bundle(transform, row * k + j) - mx);
    for (size_t j = 0; j < k; ++j)
        out(row, j) = std::exp(bundle(transform, row * k + j) - mx) / sum;
}

}  // namespace

// D2 -- returns LOGITS, not probabilities.
Tensor tta_mean_logits(const Tensor& bundle, size_t num_classes,
                       const std::vector<double>& weights) {
    const BundlePlan p = plan_aggregation(bundle, num_classes, weights);
    Tensor out(p.num_rows, p.num_classes);
    for (size_t t = 0; t < p.num_transforms; ++t)
        for (size_t n = 0; n < p.num_rows; ++n)
            for (size_t c = 0; c < p.num_classes; ++c)
                out(n, c) += p.w[t] * bundle(t, n * p.num_classes + c);
    return out;
}

Tensor tta_mean_probs(const Tensor& bundle, size_t num_classes,
                      const std::vector<double>& weights) {
    const BundlePlan p = plan_aggregation(bundle, num_classes, weights);
    Tensor probs(p.num_rows, p.num_classes);
    Tensor out(p.num_rows, p.num_classes);
    for (size_t t = 0; t < p.num_transforms; ++t) {
        for (size_t n = 0; n < p.num_rows; ++n) {
            softmax_row_into(bundle, t, n, p.num_classes, probs);
            for (size_t c = 0; c < p.num_classes; ++c)
                out(n, c) += p.w[t] * probs(n, c);
        }
    }
    return out;
}

// D3 -- the normalisation here is the whole point.
//
// Computed in LOG space: each member contributes `log_softmax(z_t)` directly
// rather than `log(softmax(z_t))`, so no intermediate is ever a logarithm of a
// probability that may have underflowed to exactly zero (which happens once a
// logit gap exceeds ~745 -- the regime the knowledge-distillation suite had to
// build a separate fixture for).
Tensor tta_geo_mean_probs(const Tensor& bundle, size_t num_classes,
                          const std::vector<double>& weights) {
    const BundlePlan p = plan_aggregation(bundle, num_classes, weights);
    Tensor acc(p.num_rows, p.num_classes);
    for (size_t t = 0; t < p.num_transforms; ++t)
        for (size_t n = 0; n < p.num_rows; ++n) {
            double mx = bundle(t, n * p.num_classes);
            for (size_t c = 1; c < p.num_classes; ++c)
                mx = std::max(mx, bundle(t, n * p.num_classes + c));
            double sum = 0.0;
            for (size_t c = 0; c < p.num_classes; ++c)
                sum += std::exp(bundle(t, n * p.num_classes + c) - mx);
            const double log_sum = mx + std::log(sum);
            for (size_t c = 0; c < p.num_classes; ++c)
                acc(n, c) += p.w[t] * (bundle(t, n * p.num_classes + c) - log_sum);
        }

    Tensor out(p.num_rows, p.num_classes);
    for (size_t n = 0; n < p.num_rows; ++n) {
        double mx = acc(n, 0);
        for (size_t c = 1; c < p.num_classes; ++c) mx = std::max(mx, acc(n, c));
        double sum = 0.0;
        for (size_t c = 0; c < p.num_classes; ++c) sum += std::exp(acc(n, c) - mx);
        for (size_t c = 0; c < p.num_classes; ++c)
            out(n, c) = std::exp(acc(n, c) - mx) / sum;
    }
    return out;
}

// D5 / D6 -- plurality over per-transform argmaxes, returned one-hot.
Tensor tta_majority_vote(const Tensor& bundle, size_t num_classes) {
    require(num_classes > 0, "tta_majority_vote: num_classes must be > 0");
    require(bundle.rows > 0, "tta_majority_vote: bundle must have transforms");
    require(bundle.cols % num_classes == 0,
            "tta_majority_vote: bundle width must be a multiple of num_classes");
    require(bundle.rows > 1,
            "tta_majority_vote: a vote needs at least 2 transforms, got " +
            std::to_string(bundle.rows));

    const size_t n = bundle.cols / num_classes;
    Tensor out(n, num_classes);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> votes(num_classes, 0.0);
        for (size_t t = 0; t < bundle.rows; ++t) {
            // D4 applies here too: the per-transform argmax must use the same
            // tie-break as tta_argmax, or a tied member would vote for a
            // different class depending on which helper is used.
            size_t best = 0;
            for (size_t c = 1; c < num_classes; ++c)
                if (bundle(t, i * num_classes + c) >
                    bundle(t, i * num_classes + best)) best = c;
            votes[best] += 1.0;
        }
        // D5: ties go to the LOWEST class index (strict `>`), NOT to the
        // earliest transform that voted. Those rules disagree on a 1-1 split.
        size_t best = 0;
        for (size_t c = 1; c < num_classes; ++c)
            if (votes[c] > votes[best]) best = c;
        out(i, best) = 1.0;
    }
    return out;
}

Tensor tta_aggregate(const Tensor& bundle, size_t num_classes,
                    TtaAggregation how, const std::vector<double>& weights) {
    switch (how) {
        case TtaAggregation::MeanLogits:
            return tta_mean_logits(bundle, num_classes, weights);
        case TtaAggregation::MeanProbs:
            return tta_mean_probs(bundle, num_classes, weights);
        case TtaAggregation::GeoMeanProbs:
            return tta_geo_mean_probs(bundle, num_classes, weights);
        case TtaAggregation::MajorityVote:
            // A vote has no notion of a weight, so accepting one here would
            // silently drop it. Refuse rather than ignore.
            require(weights.empty(),
                    "tta_aggregate: majority_vote does not accept weights");
            return tta_majority_vote(bundle, num_classes);
    }
    throw std::invalid_argument("tta_aggregate: unknown aggregation");
}