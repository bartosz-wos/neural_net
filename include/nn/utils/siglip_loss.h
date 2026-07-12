#ifndef SIGLIP_LOSS_H
#define SIGLIP_LOSS_H

#include "../core/tensor.h"
#include <utility>
#include <vector>

// =============================================================================
// SigLIP-style Sigmoid Loss for Language-Image Pre-Training
// =============================================================================
// Reference: Zhai et al. 2023 "Sigmoid Loss for Language Image Pre-Training"
// (https://arxiv.org/abs/2303.15343)
//
// The original CLIP/InfoNCE loss normalises a softmax partition function over
// the N x N image-text similarity matrix, which makes it scale as O(N^2) memory
// and compute and prevents very large batches. SigLIP replaces the softmax
// cross-entropy with an elementwise sigmoid loss — *no partition function* —
// so the loss is O(N) and can be pushed to very large batches.
//
// Per-row loss:
//     L[i] = (1/N) * sum_j softplus(-z*[i][j] * t[i][j])
// where
//     t[i][j] = +1   if i == j   (positive pair)
//              = -1   otherwise  (negative pair)
//     z*[i][j] = scale * S[i][j] + bias
//     S[i][j] = cos(z_img[i], z_txt[j])   (default, normalize=true)
//            = z_img[i] . z_txt[j]         (dot product, normalize=false)
//
// Total loss returned by forward: L = (1/N) * sum_i L[i]  (mean over batch rows).
//
// Public API:
//   - SigLIPLoss(scale=100.0, bias=0.0, eps=1e-8)
//       scale must be > 0 (defensively clamped if <= 0).
//       bias can be any value (learned offset; default 0).
//   - forward(z_img, z_txt) -> Tensor(1, 1)
//   - backward(z_img, z_txt) -> std::pair<Tensor, Tensor>  (grad_img, grad_txt)
//   - set_normalize(bool)        // toggle cosine vs dot product
//   - get_*  accessors
//   - last_*  cache accessors (S, z*, targets, normalized inputs)
//
// Backward chain (math):
//   d softplus(-z*) / dz*  =  -sigmoid(-z*)
//                         =  sigmoid(z*) - 1
//   d softplus(-z* * t) / dz*  =  -t * (sigmoid(z* * t) - 1)
//   dL[i]/dz*[i][j] = (1/N) * [dL[i]/d softplus(...)] * (-t[i][j] * (sigmoid(z*[i][j]*t[i][j]) - 1))
//                   = (1/N) * t[i][j] * (1 - sigmoid(z*[i][j] * t[i][j]))
//   dL/dS = (1/N) * scale * t * (1 - sigmoid(z* * t))
//   Then dL/dS propagates to dL/dz_img, dL/dz_txt through the cosine sim
//   chain (L2 normalization + inner product), same as in InfoNCE.
//
// Reference invariants:
//   - At scale=0, z* = bias; the loss is constant in S and equals the mean
//     softplus. With bias=0 this is exactly log(2) for every entry.
//   - The loss is always >= 0 because softplus(x) >= 0 for all x.
//   - The loss is symmetric under (z_img, z_txt) swap because t is symmetric
//     and S[img,txt][i][j] = S[img,txt][j][i] when z_img==z_txt; in general
//     we get a swap of grad_img and grad_txt but the loss is preserved as long
//     as the similarity matrix is symmetric in the inputs (it isn't always
//     so we test only the on-diagonal-positivity case for symmetry).
//
// Numerical stability:
//   softplus(x) is implemented as
//       max(x, 0) + log(1 + exp(-|x|))
//   which is finite for any real x (avoids overflow when x is large positive).
//   In our case x = -z* * t. The bias and scale are scalars so z* is finite
//   for any finite input.
//
// Implementation notes:
//   - The targets t are baked into the cache; if you ever want to override
//     the diagonal-positivity assumption (e.g. noisy labels), subclass or
//     extend. For the canonical SigLIP we keep t = +1 on the diagonal.
//   - normalize=true (cosine) is the original CLIP-style preprocessing and
//     is the default. normalize=false lets the loss operate on raw dot
//     products, useful for unit tests with hand-derived values and when
//     users already L2-normalize their embeddings upstream.

class SigLIPLoss {
public:
    // scale must be > 0 (defensively clamped to a small positive if <= 0).
    // bias is a learnable/scalar offset; default 0.
    // eps is the floor for L2-normalization (avoids divide-by-zero).
    // normalize=true -> cosine similarity (L2-normalize inputs first).
    explicit SigLIPLoss(double scale = 100.0,
                        double bias = 0.0,
                        double eps = 1e-8)
        : scale_(scale), bias_(bias), eps_(eps), normalize_(true) {
        if (scale_ <= 0.0) {
            // Defensive: SigLIP loss with non-positive scale is ill-defined
            // (z* becomes a constant in S, gradient through S vanishes).
            scale_ = 1e-9;
        }
    }

    // Forward: takes (N, D) image and (N, D) text embeddings, returns 1x1 loss.
    Tensor forward(const Tensor& z_img, const Tensor& z_txt);

    // Backward: returns (grad_z_img, grad_z_txt) each of shape (N, D).
    std::pair<Tensor, Tensor> backward(const Tensor& z_img, const Tensor& z_txt);

    // Toggle cosine vs dot product similarity.
    void set_normalize(bool b) { normalize_ = b; }

    // Accessors
    double get_scale() const { return scale_; }
    double get_bias() const { return bias_; }
    double get_eps() const { return eps_; }
    bool get_normalize() const { return normalize_; }

    // Cache accessors (populated by forward; valid until next forward).
    const Tensor& last_similarity()  const { return last_S_; }        // (N, N)
    const Tensor& last_zstar()       const { return last_zstar_; }    // (N, N)
    const Tensor& last_targets()     const { return last_targets_; }  // (N, N), t[i][j]
    const Tensor& last_img_normalized() const { return last_z_img_norm_; }  // (N, D) post-L2 if normalize=true
    const Tensor& last_txt_normalized() const { return last_z_txt_norm_; }  // (N, D)
    int last_N() const { return last_N_; }
    int last_D() const { return last_D_; }

private:
    double scale_;
    double bias_;
    double eps_;
    bool   normalize_;

    // Cached state from last forward.
    Tensor last_S_;             // (N, N) similarity (cosine or dot)
    Tensor last_zstar_;         // (N, N) = scale * S + bias
    Tensor last_targets_;       // (N, N) with t[i][j] = 2*I[i,j] - 1
    Tensor last_z_img_norm_;    // (N, D) post-L2-norm (or raw if normalize=false)
    Tensor last_z_txt_norm_;    // (N, D) post-L2-norm (or raw if normalize=false)
    int last_N_ = 0;
    int last_D_ = 0;
};

#endif
