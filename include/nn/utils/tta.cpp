#include "tta.h"
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