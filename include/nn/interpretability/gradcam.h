#ifndef GRADCAM_H
#define GRADCAM_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <cstddef>

// =============================================================================
// Grad-CAM — gradient-based class activation maps
//
// Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra, "Grad-CAM: Visual
// Explanations from Deep Networks via Gradient-based Localization", ICCV 2017,
// arXiv:1610.02391.
//
//   alpha_k^c  = (1/Z) * sum_i sum_j  dy^c / dA_ij^k      Eq. 1 (global average
//                                                        pooling of dL/dA)
//   L_Grad-CAM = ReLU( sum_k alpha_k^c * A^k )           Eq. 2
//
// The other five shipped methods all attribute a SCALAR output to the MODEL
// INPUT or to an input FEATURE. Grad-CAM is the first here that localises an
// answer to a SPATIAL MAP inside the network — the coarse heatmap of Fig. 1.
// That makes it the right tool when the input is an image-like (N, C*H*W) grid
// and "which region" is the question.
//
// -----------------------------------------------------------------------------
// WHY NO REPO-WIDE HOOK WAS NEEDED. The expansion queue recorded Grad-CAM as
// needing "an optional no-op cache_input_grad_ on the base Layer or a hook
// vector on Model", i.e. a change rippling through ~250 layers. That is
// unnecessary, for two properties the codebase already has:
//
//   1. Model::layers is a PUBLIC std::vector<std::unique_ptr<Layer>>, so the
//      forward/backward chain can be driven by hand from outside.
//   2. Layer::backward RETURNS dL/d(input) (core/layer.cpp:62-84), rather than
//      storing it. The gradient at an intermediate layer is therefore simply the
//      tensor returned by the call for the layer just after it.
//
// So grad_cam() runs the forward pass itself, snapshots the target layer's
// activation, seeds the output with a one-hot on `target`, then walks
// model.layers in reverse and keeps the tensor returned by the call that
// follows the target layer. Zero base-class change, zero risk to the ~250
// existing layers.
//
// -----------------------------------------------------------------------------
// CONVENTIONS
//
//  * Conv2D emits (N, C*H_out*W_out) with column c*(H_out*W_out) + i*W_out + j
//    (channel-major, row-major within the channel; verified in conv_layer.cpp
//    :143-151 and the im2col ordering at :70). A (C, h, w) view therefore needs
//    NO transposition.
//  * The caller supplies `channels`, `h`, `w` because the target layer's SPATIAL
//    geometry is not recoverable from the flat tensor alone — the same channel
//    count times a different grid factor is a different tensor. When they are
//    inconsistent with the actual activation width, this throws rather than
//    silently reshaping wrong.
//  * N (batch) MUST be 1 and is enforced by a throw. The GAP weights are
//    per-example; this repo's Tensor flattens (N, C*H*W) into one row block, so
//    a batch would silently produce ONE map averaged over examples.
//  * `target` selects the output COLUMN being explained. Grad-CAM attributes the
//    LOGIT, never a softmax probability, for the same reason Integrated
//    Gradients does (attribution.h:31-36): probabilities saturate and their
//    gradients vanish.
//
// -----------------------------------------------------------------------------
// WHERE THE MATH WAS READ FROM (not from memory):
//   ar5iv HTML of arXiv:1610.02391 — Eq. 1 and Eq. 2 verbatim.
//   captum/attr/_core/layer/grad_cam.py:217-228 — mean over dim 2.. (SPATIAL
//     dims only; the batch dim is not reduced).
//   ...grad_cam.py:237-242 — the weights multiply the activations and THEN a
//     single F.relu is applied to the result. ReLU is applied ONCE at the end,
//     not per channel before the weighted sum; the two orders agree on an
//     all-positive fixture and differ on a mixed-sign one, which is exactly the
//     kind of near-miss the test suite pins down.
// =============================================================================

// Result of a Grad-CAM pass.
struct GradCAM {
    // The CAM itself, (1, h*w) — ReLU(sum_k alpha_k * A^k) with the channel sum
    // collapsed. This is the coarse heatmap; upsample_bilinear() resizes it.
    Tensor cam;
    // Per-channel importance weights alpha, shape (1, channels) — Eq. 1.
    Tensor alpha;
    // Geometry of the target layer's feature map, as supplied by the caller.
    size_t channels = 0, height = 0, width = 0;
    // Which layer was explained and which output column was the target.
    size_t target_layer = 0;
    size_t target = 0;
};

// Global-average-pool the activation gradient into per-channel weights.
//
// Eq. 1: alpha[c] = (1/(h*w)) * sum_{i,j} grad[0][c*h*w + i*w + j].
//
// `act` and `act_grad` must both be (1, channels*h*w). `act` is accepted so the
// caller can be checked against the same shape contract in one place; this
// function reads only `act_grad`. Throws std::invalid_argument on a zero
// channels/h/w, an empty tensor, or a width inconsistent with channels*h*w.
Tensor grad_cam_weights(const Tensor& act, const Tensor& act_grad,
                        size_t channels, size_t h, size_t w);

// Bilinear resize of a (1, C*h*w) CAM to (1, C*H*W).
//
// Grad-CAM produces a map the size of the last conv feature map (14x14 for a
// 224x224 input at VGG16); Fig. 1 of the paper upsamples it to the input size
// for overlay. Alignment corners are matched exactly, so an exact integer-
// multiple resize is lossless at the borders and an identity resize returns the
// input bit-for-bit. Throws std::invalid_argument on a zero dimension or an
// inconsistent source width.
Tensor upsample_bilinear(const Tensor& cam, size_t C, size_t h, size_t w,
                         size_t H, size_t W);

// Grad-CAM for output column `target` taken at the feature map of layer
// `target_layer`.
//
// Performs ONE forward pass (snapshotting the target layer's activation) and
// ONE backward pass. `channels`/`h`/`w` describe the target layer's feature
// map and must satisfy channels*h*w == its output width.
//
// `relu` applies Eq. 2's ReLU (default true, the paper's method). Pass false
// for the raw weighted combination, which is useful for inspecting the
// pre-ReLU map — a signed map that some follow-up work (HiResCAM, ablation CAM)
// is defined against.
//
// Throws std::invalid_argument on an empty model, an out-of-range layer index,
// a zero channels/h/w, an inconsistent width, or a batch larger than 1;
// std::out_of_range when `target` exceeds the model's output width.
GradCAM grad_cam(Model& model, const Tensor& input,
                 size_t target_layer, size_t target = 0, bool relu = true,
                 size_t channels = 0, size_t h = 0, size_t w = 0);

#endif // GRADCAM_H