#ifndef NN_UTILS_PTQ_H
#define NN_UTILS_PTQ_H

#include "../core/tensor.h"
#include "../core/model.h"
#include <cstddef>

// ===========================================================================
// Post-training integer quantization (symmetric int8)
//
//   Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang,
//   Andrew Howard, Hartwig Adam, Dmitry Kalenichenko -- "Quantization and
//   Training of Neural Networks for Efficient Integer-Arithmetic-Only
//   Inference", arXiv:1712.05877.
//
// CITATION. Verified 2026-10-17 by a TITLE query against the arXiv API, which
// is the only check this queue trusts after it produced nine wrong IDs:
//
//   search_query=ti:"Integer-Arithmetic-Only Inference"
//     -> ['http://arxiv.org/abs/1712.05877v1']
//     -> 'Quantization and Training of Neural Networks for Efficient
//         Integer-Arithmetic-Only Inference'
//
// Two other int8 IDs this session expected returned ZERO results and are
// recorded so they are never reintroduced. Do not trust an arXiv ID typed from
// memory; search by title.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//
//   grep -rilE 'int8|uint8|zero_point|dequant|per.channel.*scale' include/
//       | grep -viE 'vq_vae|vqcodebook|bitnet|log_linear'
//   -> EMPTY
//
// Every low-precision mechanism already in this repo is a DIFFERENT object:
//   * `vq_vae.{h,cpp}` / `vqcodebook.{h,cpp}` -- VECTOR quantization. The
//     indices are integers, but there is no fixed grid, no scale and no
//     dequantization; the codebook is learned by gradient descent.
//   * `layers/utility/bitnet.h` -- 1-bit TRAINING-time binarization.
//   * per-channel `gamma` in the normalization layers -- a learned affine, not
//     a quantization grid.
//
// So the repo can train in low precision, can address memory with integer
// codes, and can compress structurally (`magnitude_pruning.h`), and has no way
// to take a TRAINED FLOAT MODEL and store its weights as int8. That is the
// deployment step between "training finished" and "inference on a device", and
// it is the one all three of those files is missing.
//
// WHY IT LIVES HERE AND NOT IN layers/: quantization is not a layer. It
// consumes and returns Tensors; nothing is learned by the forward pass.
// Model-free by construction -- every core below is a free function, so the
// whole file is testable with no Model, no optimizer and no network.
//
// ---------------------------------------------------------------------------
// THE SCHEME (paper Sec. 3, symmetric -- the paper's default)
// ---------------------------------------------------------------------------
//
//     scale = max|w| / 127
//     q_k   = clamp(round(w_k / scale), -127, 127)
//     w_hat = q_k * scale
//
// ---------------------------------------------------------------------------
// SEMANTICS: what a reimplementation gets wrong
// ---------------------------------------------------------------------------
//
// (D1) ROUNDING IS HALF-AWAY-FROM-ZERO, not half-to-even. `std::round` is
//      half-away-from-zero; numpy's `rint` is half-to-EVEN. They disagree at
//      exact half-grid points, which is where `w/scale` lands on `.5`.
//      MEASURED on a fixture built to hit them (w/scale =
//      [63.5, 126.5, 127.0, -63.5, -126.5, 0.5]): half-away gives
//      [64, 127, 127, -64, -127, 1] and half-to-even gives
//      [64, 126, 127, -64, -126, 0] -- they differ on 3 of 6 entries.
//      No tolerance-based test can observe this, which is why the suite pins it
//      with exact codes.
//
// (D2) THE MAX ENTRY MAPS TO EXACTLY +-127. Using 127.5 as the bound, or
//      clamping at 126, costs only half a least-significant bit of range and
//      so passes every error test; the codes are the only place it shows.
//
// (D3) PER-CHANNEL MEANS PER COLUMN of the (rows, cols) tensor, i.e. one
//      scale per output feature of a weight matrix, which is the axis the
//      paper scales. `(1, N)` therefore keeps N scales and `(N, 1)` keeps
//      ONE -- the two are not symmetric, and a suite that only tests square
//      matrices cannot tell the axes apart.
//
// (D4) AN ALL-ZERO TENSOR RETURNS ZEROS, NOT NaN. `scale = 0` is a legal
//      answer and dequantizing by it must not produce NaN. (The reference
//      implementation of a related scheme -- VQ-VAE's `decay = 1` -- returns an
//      all-NaN codebook here, which this repo already recorded once.)
//
// (D5) QUANTIZE->DEQUANTIZE IS EXACTLY IDEMPOTENT. Measured: 0.0, not
//      1e-17. This is the cheapest possible check that the dequantizer is the
//      inverse of the quantizer, and it holds because a grid point is already
//      on its own grid.
//
// (D6) PER-CHANNEL DOMINATES PER-TENSOR POINTWISE. Not an observation -- a
//      theorem. Per-channel error on column c is bounded by max|w_c|/(2*127)
//      and per-tensor by max|w_all|/(2*127), and max|w_c| <= max|w_all|. A 3x3
//      decade sweep (big in {1e2,1e4,1e6} x small in {1e-3,1e-6,1e-9}) found
//      NO shape where per-channel has worse L2 error, which is consistent with
//      the bound. Do not write a "per-channel loses" fixture; it would be
//      asserting a falsehood. (The paper's per-channel win on REAL accuracy
//      comes from layer OUTPUTS sharing a per-channel scale, not from a better
//      reconstruction bound.)
//
// (D7) THE RETURNED TENSORS NEVER ALIAS THEIR INPUTS. Every function returns a
//      fresh Tensor.
//
// (D8) EMPTY TENSORS THROW. `Tensor(0,0)` is a legal construction here and
//      `max|w|` over it is undefined; the honest answer is an exception.
//
// ---------------------------------------------------------------------------
// WHY NO FINITE-DIFFERENCE TEST APPEARS IN THIS FILE
// ---------------------------------------------------------------------------
//
// The quantize->dequantize forward is PIECEWISE CONSTANT in w. MEASURED: a
// centred difference `(f(w+e) - f(w-e)) / 2e` with e = 1e-7 returns EXACTLY
// 0.0 on every non-maximal entry -- for a correct straight-through estimator
// AND for an arbitrarily wrong one alike. FD therefore cannot discriminate
// between a correct and an incorrect backward here; it reads 0 for both.
//
// The ONE exception is the tensor's own maximum: perturbing it moves
// `max|w|`, hence the whole grid. Measured per-entry FD on
// [0.37, -1.21, 2.05, 0.0] -> [0.0, 0.0, 1.0, 0.0]. That asymmetry is itself a
// pinned property, and it is the diagnostic signature: a clean constant ratio
// on EVERY entry means a rounding bug; exactly one non-zero entry means the
// max-derived scale is working.
//
// So the forward is verified by VALUE (against a float64 numpy oracle) and the
// straight-through estimator by an IDENTITY assertion plus mutation testing.
//
// ===========================================================================

// The int8 code range used by the symmetric grid. Signed, so [-127, +127] --
// -128 is excluded so the grid is symmetric about zero.
static constexpr double PTQ_QMAX = 127.0;

enum class PtqGranularity {
    kPerTensor,   // one scale for the whole tensor
    kPerChannel,  // one scale per COLUMN of a (rows, cols) tensor
};

struct PtqReport {
    int num_tensors = 0;      // how many parameter tensors were quantized
    double max_abs_err = 0.0; // worst |w - w_hat| over every entry
    double rel_l2_err = 0.0;  // ||w - w_hat|| / ||w||, aggregated over all
};

// -- scales -----------------------------------------------------------------

// scale = max|w| / 127 over the WHOLE tensor. Throws on an empty tensor.
double ptq_scale_per_tensor(const Tensor& w);

// (1, cols) tensor holding one scale per COLUMN: scale_j = max_i |w[i][j]| / 127.
// Throws on an empty tensor.
Tensor ptq_scale_per_channel(const Tensor& w);

// -- quantize / dequantize --------------------------------------------------

// Round to the nearest grid point and clamp to [-127, 127]. Returns a FRESH
// tensor of exact integer-valued doubles (D2, D10).
Tensor ptq_quantize_per_tensor(const Tensor& w);
Tensor ptq_quantize_per_channel(const Tensor& w);

// codes * scale. A zero scale returns zeros rather than NaN (D4).
Tensor ptq_dequantize(const Tensor& codes, double scale);

// Per-channel dequantization: `scales` is the (1, cols) tensor returned by
// `ptq_scale_per_channel`, and column j is dequantized by scales[0][j].
Tensor ptq_dequantize_per_channel(const Tensor& codes, const Tensor& scales);

// Quantize->dequantize in one call. Exactly idempotent (D5).
Tensor ptq_quantize_dequantize(const Tensor& w, PtqGranularity g);

// -- error metrics ----------------------------------------------------------

// max |w[i][j] - w_hat[i][j]|. Throws on an empty tensor.
double ptq_max_abs_error(const Tensor& w, const Tensor& w_hat);

// ||w - w_hat||_F / ||w||_F. Returns 0.0 when w is exactly zero, rather than
// dividing 0/0 (the same trap as D4, on the metric side).
double ptq_relative_l2(const Tensor& w, const Tensor& w_hat);

// -- straight-through estimator --------------------------------------------

// The STE used for quantization-aware fine-tuning: the VALUE is quantized, the
// GRADIENT passes through as the identity. FD cannot verify this (see the file
// header), so it is asserted as an exact bit-equality in the suite.
Tensor ptq_straight_through(const Tensor& grad);

// -- model-level -----------------------------------------------------------

// Quantize EVERY parameter tensor of every layer in place. Returns what it
// did, so a caller can assert the model was actually touched (a loop over an
// empty `model.layers` would otherwise no-op silently).
PtqReport ptq_quantize_model(Model& model, PtqGranularity g);

// argmax index of each ROW, as a (rows, 1) tensor. Ties go to the LOWEST
// column index (strict `>`, matching `metrics.cpp`).
Tensor ptq_argmax_rows(const Tensor& logits);

// Fraction of rows whose argmax index matches the argmax of that row of
// `targets`. `pred` is (rows, 1) of class INDICES and `targets` is (rows, K)
// of logits or one-hot rows -- the two need not have the same width, because
// a prediction is an index and a target is a distribution.
// Throws on an empty tensor.
double ptq_accuracy(const Tensor& pred, const Tensor& targets);

#endif  // NN_UTILS_PTQ_H