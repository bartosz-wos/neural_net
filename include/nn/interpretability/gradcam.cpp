#include "gradcam.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// Validate that a (1, C*h*w) tensor is exactly that shape. Both the activation
// and the gradient flow through here, so the geometry contract has one home.
void check_map(const Tensor& t, size_t C, size_t h, size_t w, const char* what) {
    if (t.rows == 0 || t.cols == 0) {
        throw std::invalid_argument(
            std::string("grad_cam: ") + what + " tensor is empty");
    }
    if (t.rows != 1) {
        throw std::invalid_argument(
            std::string("grad_cam: ") + what +
            " must be a single example (rows == 1); the GAP weights are "
            "per-example and a batch would be averaged into one map");
    }
    if (C * h * w != t.cols) {
        throw std::invalid_argument(
            std::string("grad_cam: ") + what + " width " + std::to_string(t.cols) +
            " is inconsistent with channels*h*w = " +
            std::to_string(C * h * w));
    }
}

}  // namespace

Tensor grad_cam_weights(const Tensor& act, const Tensor& act_grad,
                        size_t channels, size_t h, size_t w) {
    if (channels == 0 || h == 0 || w == 0) {
        throw std::invalid_argument(
            "grad_cam_weights: channels, h and w must all be non-zero");
    }
    check_map(act, channels, h, w, "activation");
    check_map(act_grad, channels, h, w, "gradient");

    // Eq. 1: alpha[k] = (1/Z) * sum_{i,j} dL/dA_ij^k, where the sum runs over the
    // SPATIAL dims only. The critical detail: this is a mean over each channel's
    // own h*w block, NOT a mean over the whole flattened row. Averaging over the
    // row would divide by channels*h*w and shrink every weight by `channels` —
    // a constant factor that looks plausible and is wrong.
    const double denom = static_cast<double>(h * w);
    Tensor alpha(1, channels);
    for (size_t c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (size_t s = 0; s < h * w; ++s) {
            acc += act_grad[0][c * h * w + s];
        }
        alpha[0][c] = acc / denom;
    }
    return alpha;
}

Tensor upsample_bilinear(const Tensor& cam, size_t C, size_t h, size_t w,
                         size_t H, size_t W) {
    if (C == 0 || h == 0 || w == 0 || H == 0 || W == 0) {
        throw std::invalid_argument(
            "upsample_bilinear: every dimension must be non-zero");
    }
    check_map(cam, C, h, w, "cam");

    Tensor out(1, C * H * W);
    const double sh = static_cast<double>(h);
    const double sw = static_cast<double>(w);

    for (size_t c = 0; c < C; ++c) {
        for (size_t y = 0; y < H; ++y) {
            // Align corners: map output pixel y to source coordinate
            // y * (h-1)/(H-1), which makes the four corners exact and makes an
            // identity resize (H==h, W==w) reproduce the input bit-for-bit.
            const double fy = (H == 1) ? 0.0 : (static_cast<double>(y) * (sh - 1.0) /
                                                static_cast<double>(H - 1));
            size_t y0 = static_cast<size_t>(std::floor(fy));
            size_t y1 = std::min(y0 + 1, h - 1);
            const double wy = fy - static_cast<double>(y0);

            for (size_t x = 0; x < W; ++x) {
                const double fx = (W == 1) ? 0.0 : (static_cast<double>(x) * (sw - 1.0) /
                                                    static_cast<double>(W - 1));
                size_t x0 = static_cast<size_t>(std::floor(fx));
                size_t x1 = std::min(x0 + 1, w - 1);
                const double wx = fx - static_cast<double>(x0);

                const double v00 = cam[0][c * h * w + y0 * w + x0];
                const double v01 = cam[0][c * h * w + y0 * w + x1];
                const double v10 = cam[0][c * h * w + y1 * w + x0];
                const double v11 = cam[0][c * h * w + y1 * w + x1];

                const double top = v00 * (1.0 - wx) + v01 * wx;
                const double bot = v10 * (1.0 - wx) + v11 * wx;
                out[0][c * H * W + y * W + x] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    return out;
}

GradCAM grad_cam(Model& model, const Tensor& input,
                 size_t target_layer, size_t target, bool relu,
                 size_t channels, size_t h, size_t w) {
    if (model.layers.empty()) {
        throw std::invalid_argument("grad_cam: model has no layers");
    }
    if (target_layer >= model.layers.size()) {
        throw std::out_of_range(
            "grad_cam: target_layer " + std::to_string(target_layer) +
            " is out of range (model has " + std::to_string(model.layers.size()) +
            " layers)");
    }
    if (channels == 0 || h == 0 || w == 0) {
        throw std::invalid_argument(
            "grad_cam: channels, h and w must all be non-zero (they describe "
            "the target layer's feature map)");
    }

    // Dense::backward and friends ACCUMULATE into grad_weights. Zero first so
    // this attribution pass neither adds to nor reads stale training state.
    for (auto& l : model.layers) l->zero_grad();

    // ---- Forward pass, snapshotting the target layer's activation ----
    Tensor current = input;
    Tensor act;
    for (size_t i = 0; i < model.layers.size(); ++i) {
        current = model.layers[i]->forward(current);
        if (i == target_layer) act = current;
    }
    const Tensor& out = current;

    if (out.rows != 1) {
        throw std::invalid_argument(
            "grad_cam: this method requires a single example (batch size 1)");
    }
    if (target >= out.cols) {
        throw std::out_of_range(
            "grad_cam: target " + std::to_string(target) +
            " is out of range for model output width " + std::to_string(out.cols));
    }
    check_map(act, channels, h, w, "target-layer activation");

    // ---- Backward pass, keeping the gradient at the target layer ----
    // Seeding with a one-hot on `target` makes the pass compute
    // d(logit_target)/d(anything), so the logit is attributed rather than a
    // softmax probability (which saturates and has vanishing gradients).
    Tensor seed(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i) seed[i][target] = 1.0;

    // Layer::backward RETURNS dL/d(input). Walking in reverse, the call for the
    // layer AFTER the target layer returns exactly dL/d(activation of the
    // target layer) — that is the tensor Eq. 1 averages. When the target layer
    // is the last one, dL/d(its own activation) is the seed itself, and no
    // capture is needed.
    Tensor act_grad;
    Tensor grad = seed;
    for (size_t i = model.layers.size(); i-- > 0;) {
        if (i == target_layer) {
            // `grad` entering here is dL/d(this layer's OUTPUT), which is what
            // Eq. 1 wants when the target layer is the final one.
            act_grad = grad;
            break;
        }
        Tensor next = model.layers[i]->backward(grad, 0.0);
        if (i == target_layer + 1) {
            act_grad = next;   // dL/d(target layer's activation)
        }
        grad = next;
    }

    // Leave no residue: attribution is not a training step.
    for (auto& l : model.layers) l->zero_grad();

    // ---- Eq. 1 then Eq. 2 ----
    Tensor alpha = grad_cam_weights(act, act_grad, channels, h, w);

    Tensor cam(1, h * w);
    for (size_t s = 0; s < h * w; ++s) {
        double acc = 0.0;
        for (size_t c = 0; c < channels; ++c) {
            acc += alpha[0][c] * act[0][c * h * w + s];
        }
        // Eq. 2's ReLU is applied ONCE to the summed combination. Applying it
        // per channel before the sum is a different map that happens to agree on
        // an all-positive fixture.
        cam[0][s] = relu ? std::max(0.0, acc) : acc;
    }

    GradCAM result;
    result.cam = cam;
    result.alpha = alpha;
    result.channels = channels;
    result.height = h;
    result.width = w;
    result.target_layer = target_layer;
    result.target = target;
    return result;
}