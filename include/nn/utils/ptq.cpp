#include "ptq.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>



// ---------------------------------------------------------------------------
// Guards. `Tensor::operator()` is `data[r*cols + c]` with NO bounds check, so
// an out-of-range index here is silent heap corruption. Every public entry
// point validates its input first.
// ---------------------------------------------------------------------------
namespace {

void require_non_empty(const Tensor& t, const char* who) {
    if (t.rows == 0 || t.cols == 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": tensor is empty (" +
                                    std::to_string(t.rows) + "x" +
                                    std::to_string(t.cols) + ")");
    }
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* who) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument(
            std::string(who) + ": shape mismatch " +
            std::to_string(a.rows) + "x" + std::to_string(a.cols) + " vs " +
            std::to_string(b.rows) + "x" + std::to_string(b.cols));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Scales
// ---------------------------------------------------------------------------

double ptq_scale_per_tensor(const Tensor& w) {
    require_non_empty(w, "ptq_scale_per_tensor");
    double amax = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            amax = std::max(amax, std::fabs(w[i][j]));
    // D4: an all-zero tensor has scale 0, which is a LEGAL answer. Callers
    // must not divide by it -- `ptq_dequantize` special-cases it instead of
    // producing NaN.
    return amax / PTQ_QMAX;
}

Tensor ptq_scale_per_channel(const Tensor& w) {
    require_non_empty(w, "ptq_scale_per_channel");
    // D3: one scale per COLUMN of the (rows, cols) tensor -- the output-feature
    // axis of a weight matrix, which is the axis the paper scales. So a (1, N)
    // tensor keeps N scales and an (N, 1) tensor keeps ONE. Reducing over the
    // wrong axis would make those two cases identical.
    Tensor scales(1, w.cols);
    for (size_t j = 0; j < w.cols; ++j) {
        double amax = 0.0;
        for (size_t i = 0; i < w.rows; ++i)
            amax = std::max(amax, std::fabs(w[i][j]));
        scales[0][j] = amax / PTQ_QMAX;
    }
    return scales;
}

// ---------------------------------------------------------------------------
// Quantize / dequantize
// ---------------------------------------------------------------------------

namespace {

// Round to the nearest grid point, then clamp.
//
// D1: `std::round` is ROUND-HALF-AWAY-FROM-ZERO. The alternative in wide use
// (numpy's `rint`, and it is what a `std::nearbyint` with the default rounding
// mode gives) is ROUND-HALF-TO-EVEN. They disagree at exact half-grid points:
// on w/scale = [63.5, 126.5, 127.0, -63.5, -126.5, 0.5] this returns
// [64, 127, 127, -64, -127, 1] and half-to-even returns
// [64, 126, 127, -64, -126, 0] -- a 3-of-6 disagreement that no tolerance
// test can observe.
//
// D2: clamp to [-127, +127] inclusive. `PTQ_QMAX` is 127 and NOT 128, because
// the signed int8 grid is [-128, 127] but a symmetric quantizer leaves -128
// unused so the grid is symmetric about zero.
Tensor round_to_grid(const Tensor& w, const Tensor& scales, bool per_channel) {
    Tensor q(w.rows, w.cols);
    for (size_t i = 0; i < w.rows; ++i) {
        for (size_t j = 0; j < w.cols; ++j) {
            const double s = per_channel ? scales[0][j] : scales[0][0];
            if (s == 0.0) {
                // D4: a zero channel has no grid. Every value in it is already
                // 0 (a zero scale means max|w_j| == 0), so the code is 0 and no
                // division is performed -- which is what keeps this off 0/0.
                q[i][j] = 0.0;
                continue;
            }
            double r = std::round(w[i][j] / s);
            r = std::max(-PTQ_QMAX, std::min(PTQ_QMAX, r));
            q[i][j] = r;
        }
    }
    return q;
}

}  // namespace

Tensor ptq_quantize_per_tensor(const Tensor& w) {
    require_non_empty(w, "ptq_quantize_per_tensor");
    Tensor scales(1, 1);
    scales[0][0] = ptq_scale_per_tensor(w);
    return round_to_grid(w, scales, /*per_channel=*/false);
}

Tensor ptq_quantize_per_channel(const Tensor& w) {
    require_non_empty(w, "ptq_quantize_per_channel");
    return round_to_grid(w, ptq_scale_per_channel(w), /*per_channel=*/true);
}

Tensor ptq_dequantize(const Tensor& codes, double scale) {
    require_non_empty(codes, "ptq_dequantize");
    Tensor out(codes.rows, codes.cols);
    if (scale == 0.0) {
        // D4: multiply-by-zero is defined and gives exactly 0. The trap is the
        // `0/0` in the quantizer, which is handled above; here a zero scale
        // must still not yield NaN.
        return Tensor::zeros(codes.rows, codes.cols);
    }
    for (size_t i = 0; i < codes.rows; ++i)
        for (size_t j = 0; j < codes.cols; ++j) out[i][j] = codes[i][j] * scale;
    return out;
}

Tensor ptq_dequantize_per_channel(const Tensor& codes, const Tensor& scales) {
    require_non_empty(codes, "ptq_dequantize_per_channel");
    require_non_empty(scales, "ptq_dequantize_per_channel");
    if (scales.rows != 1 || scales.cols != codes.cols) {
        throw std::invalid_argument(
            std::string("ptq_dequantize_per_channel: scales must be (1, cols) "
                        "matching codes, got ") +
            std::to_string(scales.rows) + "x" + std::to_string(scales.cols) +
            " vs codes " + std::to_string(codes.rows) + "x" +
            std::to_string(codes.cols));
    }
    Tensor out(codes.rows, codes.cols);
    for (size_t i = 0; i < codes.rows; ++i) {
        for (size_t j = 0; j < codes.cols; ++j) {
            const double s = scales[0][j];
            // D4 per-channel: a column whose max magnitude is 0 has scale 0,
            // and its codes are all 0, so the product is defined 0 -- but the
            // explicit branch keeps this off a 0*inf and documents the case.
            out[i][j] = (s == 0.0) ? 0.0 : codes[i][j] * s;
        }
    }
    return out;
}

Tensor ptq_quantize_dequantize(const Tensor& w, PtqGranularity g) {
    if (g == PtqGranularity::kPerTensor) {
        return ptq_dequantize(ptq_quantize_per_tensor(w),
                              ptq_scale_per_tensor(w));
    }
    return ptq_dequantize_per_channel(ptq_quantize_per_channel(w),
                                      ptq_scale_per_channel(w));
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

double ptq_max_abs_error(const Tensor& w, const Tensor& w_hat) {
    require_non_empty(w, "ptq_max_abs_error");
    require_same_shape(w, w_hat, "ptq_max_abs_error");
    double worst = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            worst = std::max(worst, std::fabs(w[i][j] - w_hat[i][j]));
    return worst;
}

double ptq_relative_l2(const Tensor& w, const Tensor& w_hat) {
    require_non_empty(w, "ptq_relative_l2");
    require_same_shape(w, w_hat, "ptq_relative_l2");
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < w.rows; ++i) {
        for (size_t j = 0; j < w.cols; ++j) {
            const double d = w[i][j] - w_hat[i][j];
            num += d * d;
            den += w[i][j] * w[i][j];
        }
    }
    // D4 again, on the metric side: an all-zero reference is not a division.
    if (den == 0.0) return num == 0.0 ? 0.0 : num;
    return std::sqrt(num) / std::sqrt(den);
}

// ---------------------------------------------------------------------------
// Straight-through estimator
// ---------------------------------------------------------------------------

Tensor ptq_straight_through(const Tensor& grad) {
    require_non_empty(grad, "ptq_straight_through");
    // The STE's definition IS the identity on the gradient: the forward value
    // is quantized, the backward passes the incoming gradient through
    // unchanged. It is written out rather than aliased so the intent is
    // visible at the call site -- and the suite asserts bit-equality against
    // this definition, because a centred FD cannot see a piecewise-constant
    // forward at all (see the header).
    return grad.clone();
}

// ---------------------------------------------------------------------------
// Model-level
// ---------------------------------------------------------------------------

PtqReport ptq_quantize_model(Model& model, PtqGranularity g) {
    PtqReport rep;
    double num_sq = 0.0, den_sq = 0.0;
    for (auto& layer : model.layers) {
        std::vector<Tensor*> params = layer->parameters();
        for (Tensor* p : params) {
            if (p == nullptr) continue;
            if (p->rows == 0 || p->cols == 0) continue;   // e.g. Activation

            Tensor w_hat = ptq_quantize_dequantize(*p, g);

            rep.max_abs_err =
                std::max(rep.max_abs_err, ptq_max_abs_error(*p, w_hat));
            for (size_t i = 0; i < p->rows; ++i) {
                for (size_t j = 0; j < p->cols; ++j) {
                    const double d = (*p)[i][j] - w_hat[i][j];
                    num_sq += d * d;
                    den_sq += (*p)[i][j] * (*p)[i][j];
                }
            }
            // Write the quantized values back in place: this is a
            // POST-TRAINING projection, so the model keeps running in float
            // and only its weights sit on the int8 grid.
            *p = w_hat;
            ++rep.num_tensors;
        }
    }
    rep.rel_l2_err = (den_sq == 0.0) ? 0.0 : std::sqrt(num_sq / den_sq);
    return rep;
}

Tensor ptq_argmax_rows(const Tensor& logits) {
    require_non_empty(logits, "ptq_argmax_rows");
    Tensor out(logits.rows, 1);
    for (size_t i = 0; i < logits.rows; ++i) {
        size_t best = 0;
        for (size_t j = 1; j < logits.cols; ++j)
            // Strict `>` so a tie goes to the LOWEST column index, matching
            // `metrics.cpp`. `>=` would give the LAST index on a tie, which a
            // two-class fixture with equal logits can distinguish.
            if (logits[i][j] > logits[i][best]) best = j;
        out[i][0] = static_cast<double>(best);
    }
    return out;
}

double ptq_accuracy(const Tensor& pred, const Tensor& targets) {
    require_non_empty(pred, "ptq_accuracy");
    require_non_empty(targets, "ptq_accuracy");
    // `pred` is (rows, 1) of class indices; `targets` is (rows, K). They need
    // NOT have the same width -- a prediction is an index, a target is a
    // distribution -- so only the ROW count is a shape requirement. Asserting
    // equal widths here is the bug the first GREEN run hit.
    if (pred.rows != targets.rows) {
        throw std::invalid_argument(
            std::string("ptq_accuracy: row count mismatch ") +
            std::to_string(pred.rows) + " vs " + std::to_string(targets.rows));
    }
    if (pred.cols != 1) {
        throw std::invalid_argument(
            std::string("ptq_accuracy: pred must be (rows, 1) of class "
                        "indices, got cols=") +
            std::to_string(pred.cols));
    }
    size_t correct = 0;
    for (size_t i = 0; i < pred.rows; ++i) {
        size_t true_class = 0;
        for (size_t j = 1; j < targets.cols; ++j)
            if (targets[i][j] > targets[i][true_class]) true_class = j;
        if (static_cast<size_t>(pred[i][0]) == true_class) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(pred.rows);
}

