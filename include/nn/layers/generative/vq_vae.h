#ifndef NN_LAYERS_GENERATIVE_VQ_VAE_H
#define NN_LAYERS_GENERATIVE_VQ_VAE_H

#include "../../core/layer.h"
#include <cstddef>
#include <vector>

// ===========================================================================
// Vector Quantized Variational Autoencoder
//
//   Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu --
//   "Neural Discrete Representation Learning", NeurIPS 2017,
//   arXiv:1711.00937.
//
// CITATION. Verified against the arXiv API on 2026-10-15: `id_list=1711.00937`
// returns "Neural Discrete Representation Learning" with exactly these three
// authors. This repo's expansion queue has produced five wrong arXiv IDs
// before, so the check is not optional. Do not substitute another ID.
//
// THE GAP THIS FILLS. The repo had a full continuous-latent generative family
// -- `vae.{h,cpp}` (amortised Gaussian), `ddpm.h`, `dit.h`, `consistency.h`,
// `neural_spline_flow.h`, `flow_matching.h` -- and NOTHING with a DISCRETE
// latent. Verified absent by content grep before implementing:
//     grep -ril 'quantiz\|codebook\|straight.through\|commitment' include/
// -> 0 hits outside unrelated files. The paper's premise is that a discrete
// latent beats a continuous one of the same capacity (§4.1), and that
// comparison is impossible to run in this repo without this file.
//
// ---------------------------------------------------------------------------
// THE ONE PROPERTY THAT DEFINES THIS METHOD
// ---------------------------------------------------------------------------
//
// Paper Eq. (2):
//     z_q(x) = e_k,   k = argmin_j || z_e(x) - e_j ||_2
//
// so the encoder's output z_e is SNAPPED to the nearest codebook vector before
// the decoder ever sees it. The decoder therefore never sees z_e.
//
// That snap has NO derivative: argmin is piecewise constant. The paper
// §3.2 resolves this with the STRAIGHT-THROUGH ESTIMATOR -- "during the
// backwards pass the gradient is passed unaltered to the encoder". So:
//
//     forward:   value     = z_q(x)   <-- the NEAREST CODE, bit-identical
//     backward:  gradient  = dL/dz_q  <-- copied to dL/dz_e unchanged
//
// The gradient half is easy to get right and the VALUE half is easy to get
// wrong. An implementation that returns z_e from forward() has quantized
// nothing at all: it is a plain autoencoder wearing a VQ-VAE's name, it
// trains fine, and every loss-based test still passes. That is why D5 in the
// test suite is a headline assertion and why the fixture is checked to be
// one where z_q and z_e actually differ.
//
// ---------------------------------------------------------------------------
// THE LOSS, AND WHY IT IS MORE SUBTLE THAN IT LOOKS
// ---------------------------------------------------------------------------
//
// Paper Eq. (3):
//     L = log p(x|z_q(x))  +  || sg[z_e] - e ||^2  +  beta*|| z_e - sg[e] ||^2
//                                     ^^^^^^^            ^^^^
//                                     dictionary          commitment
//
// sg[.] is the stop-gradient: identity at forward time, zero derivative.
//
//     term 2 (DICTIONARY) moves the CODEBOOK vectors e toward z_e. The encoder
//                          is stopped out of it.
//     term 3 (COMMITMENT) moves the ENCODER OUTPUT z_e toward e. The codebook
//                          is stopped out of it. beta = 0.25 (paper §3.2).
//
// *** THE SUBTLETY, AND IT IS THE WHOLE POINT ***
//
// The two terms are the SAME SCALAR. ||sg[z_e] - e||^2 and ||z_e - sg[e]||^2
// are numerically identical -- both equal ||z_e - z_q||^2 -- because sg does
// not change a value. They differ ONLY in their DEPENDENCIES:
//
//     dictionary  is a function of the CODEBOOK alone
//     commitment  is a function of z_e alone
//
// So no value test can tell the two terms apart, and -- this is the part that
// will bite anyone writing a finite-difference check -- DIFFERENTIATING THE
// COMBINED SCALAR IS WRONG. A naive forward-mode pass differentiates both
// terms through z_q and returns (1 + beta) on the codebook gradient. That is a
// CLEAN CONSTANT RATIO of 1.25 on every single cell, which is the signature of
// "your harness differentiates the wrong function", not "your gradient is
// wrong". Fix the HARNESS. See tests/test_vq_vae.cpp test 10, which
// differentiates each term separately and never their sum.
//
// The same trap has a second face in `vq_commitment_gradient`: it is the ONLY
// path that carries a gradient to z_e apart from the straight-through, and it
// carries a factor of exactly beta and nothing else.
//
// ---------------------------------------------------------------------------
// EMA VARIANT (D6)
// ---------------------------------------------------------------------------
//
// Paper §3.2 notes the dictionary "can alternatively also be updated as a
// function of moving averages of z_e". DeepMind's Sonnet ships that variant
// (`VectorQuantizerEMA`) and it is here too, because it changes the layer's
// CONTRACT rather than just its details:
//
//   * the codebook is not gradient-updated at all -- it is ASSIGNED from the
//     moving averages, so `update_weights(lr)` ignores lr entirely;
//   * the dictionary term is therefore DROPPED from the returned loss;
//   * the straight-through identity does NOT apply to the returned loss, and
//     the class says so rather than inheriting a misleading contract.
//
// ---------------------------------------------------------------------------
// API shape
// ---------------------------------------------------------------------------
//
// Every numerical core is a FREE FUNCTION over Tensors, so the whole quantizer
// is testable with no Model, no optimizer and no encoder -- the same
// model-free property `utils/vat.h` and `utils/randomized_smoothing.h` were
// built with. The Layer classes are thin adapters over them.
//
// Shapes. The codebook is (D, K): COLUMN k IS embedding e_k. That matches
// Sonnet's `self._w` of shape [embedding_dim, num_embeddings] and makes
// `sum(w**2, 0)` the per-code L2 norm with no transposition (D3).

// ---------------------------------------------------------------------------
// Free functions -- the testable core.
// ---------------------------------------------------------------------------

// Squared L2 distance from each row of `z` (M, D) to each column of `w`
// (D, K), returning (M, K). Same expansion as the reference
// (|x|^2 - 2x.w + |w|^2), which avoids the explicit (M,D)x(D,K) difference.
Tensor vq_squared_distances(const Tensor& z, const Tensor& w);

// Index of the nearest codebook column for each row of `z`.
// TIE-BREAK IS FIRST-WINS (D4): the reference is `argmax(-distances, 1)`, and
// both argmax and argmin return the first extremum. Only a fixture containing
// an EXACT tie can observe this -- a tolerance-based implementation breaks
// ties arbitrarily and no approximate assertion can tell.
std::vector<size_t> vq_nearest_indices(const Tensor& z, const Tensor& w);

// The quantized output: row i of the result is column indices[i] of `w`,
// copied BIT-IDENTICALLY (never recomputed as an average).
Tensor vq_quantize(const Tensor& z, const Tensor& w);

// The DICTIONARY term, UNWEIGHTED: mean over all M*D elements of
// (z_q - z_e)^2. This is the loss that updates the codebook (D1, D2).
double vq_codebook_loss(const Tensor& z, const Tensor& z_q);

// The COMMITMENT term: beta * mean over all M*D elements of (z_e - z_q)^2.
// This is the loss that updates the encoder output (D1, D2).
double vq_commitment_loss(const Tensor& z, const Tensor& z_q, double beta);

// d(dictionary)/d(codebook), shape (D, K). Entries for codes assigned by no
// row are EXACTLY zero -- not small, not stale. Takes the CODEBOOK (not z_q)
// because K -- the codebook width -- is needed for the gradient's shape and is
// not derivable from the latent width.
Tensor vq_codebook_gradient(const Tensor& z, const Tensor& w);

// The WRONG-variant helper used only by the D1 discriminator in the tests:
// the same quantity with beta applied to the codebook gradient as well.
// A correct implementation and this one must DIFFER; if they agree, the test
// cannot discriminate the two orderings and the fixture is vacuous.
Tensor vq_codebook_gradient_with_beta(const Tensor& z, const Tensor& w, double beta);

// d(commitment)/d(z_e), shape (M, D) = 2*beta*(z_e - z_q)/(M*D). Carries a
// factor of EXACTLY beta and nothing else.
Tensor vq_commitment_gradient(const Tensor& z, const Tensor& z_q, double beta);

// The straight-through output. VALUE == z_q bit-identically; only the
// gradient is the identity. Note the algebraic form in the reference,
// `z + sg(z_q - z)`, evaluates to z_q in exact arithmetic but is NOT
// bit-identical in floating point -- so this returns z_q directly.
Tensor vq_straight_through(const Tensor& z, const Tensor& z_q);

// Perplexity of the code usage, `exp(-sum(avg * log(avg + 1e-10)))`.
// NEVER exactly an integer: the +1e-10 inside the log biases it strictly
// below the count (D9). The ceiling is min(K, M), not K.
double vq_perplexity(const std::vector<size_t>& indices, size_t num_codes);

// One EMA step of the moving-average codebook (Sonnet lines 224-236):
//     n_k  <- decay*n_k + (1-decay)*count_k
//     dw   <- decay*dw  + (1-decay)*sum_{i: idx_i = k} z_e[i]
//     N    <- sum_k n_k
//     n_k  <- (n_k + eps)/(N + K*eps) * N      <-- Laplace smoothing (D7)
//     w    <- dw / n_k
// A code assigned by NO row collapses to EXACTLY ZERO, because its n_k is eps
// and its dw is 0 (D8). Returned by value; the caller owns the state.
Tensor vq_ema_step(const Tensor& z, const Tensor& w,
                   const Tensor& cluster_size, const Tensor& ema_w,
                   double decay, double epsilon);

// ---------------------------------------------------------------------------
// VectorQuantizer -- the loss-based quantizer of Eq. (3).
// ---------------------------------------------------------------------------
class VectorQuantizer : public Layer {
public:
    VectorQuantizer(size_t embedding_dim, size_t num_embeddings, double beta);

    Tensor forward(const Tensor& input) override;
    // Returns grad_output UNALTERED (the straight-through identity) PLUS the
    // beta-weighted commitment gradient, which is the one path the STE cannot
    // supply. Accumulates the dictionary gradient into the codebook's gradient.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    // Plain SGD on the codebook. Deliberately does NOT touch the encoding.
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return codebook_; }
    Tensor get_gradients() const override { return grad_codebook_; }
    std::vector<Tensor*> parameters() override { return {&codebook_}; }
    std::vector<Tensor*> gradients() override { return {&grad_codebook_}; }
    std::string name() const override { return "VectorQuantizer"; }

    // Set an explicit codebook. Throws std::invalid_argument on a shape
    // mismatch rather than silently reinterpreting the memory.
    void set_codebook(const Tensor& w);
    Tensor codebook() const { return codebook_.clone(); }

    // The two terms REPORTED SEPARATELY. They are numerically identical
    // before beta and different after, and exposing them separately is the
    // only way a caller can see which is which (D1).
    double codebook_loss() const { return codebook_loss_; }
    double commitment_loss() const { return commitment_loss_; }
    double total_loss() const { return codebook_loss_ + commitment_loss_; }

    double perplexity() const { return perplexity_; }
    const std::vector<size_t>& indices() const { return indices_; }
    // Codes hit by at least one row in the last forward pass, and the
    // complement. A rising dead-code count is the classic codebook-collapse
    // signal, which is why it is exposed rather than only inferred.
    size_t usage_count() const { return used_codes_.size(); }
    size_t dead_code_count() const { return dead_codes_.size(); }

    size_t embedding_dim() const { return embedding_dim_; }
    size_t num_embeddings() const { return num_embeddings_; }
    double beta() const { return beta_; }

private:
    size_t embedding_dim_, num_embeddings_;
    double beta_;

    Tensor codebook_;        // (D, K) -- COLUMN k IS e_k (D3)
    Tensor grad_codebook_;   // (D, K)

    Tensor last_z_e_, last_z_q_;
    std::vector<size_t> indices_;
    std::vector<size_t> used_codes_, dead_codes_;
    double codebook_loss_ = 0.0, commitment_loss_ = 0.0, perplexity_ = 0.0;
};

// ---------------------------------------------------------------------------
// VectorQuantizerEMA -- the moving-average variant.
//
// THE CONTRACT DIFFERS from VectorQuantizer and the differences are load-
// bearing, so they are stated rather than inherited:
//
//   * `update_weights(lr)` IGNORES lr. The codebook is assigned from moving
//     averages, so the update is independent of the optimizer -- which is the
//     entire documented advantage of this variant.
//   * The DICTIONARY term is dropped from `total_loss()` (D6): the codebook is
//     never gradient-updated, so a dictionary term would be reporting a
//     quantity that nothing acts on.
//   * `backward()` returns ONLY the commitment gradient. There is no
//     straight-through path here: with no reconstruction gradient reaching the
//     codebook, the only gradient that acts is the commitment one.
//   * `update_weights(lr, training=false)` is a no-op, so evaluation does not
//     pollute the statistics.
// ---------------------------------------------------------------------------
class VectorQuantizerEMA : public Layer {
public:
    VectorQuantizerEMA(size_t embedding_dim, size_t num_embeddings, double beta,
                       double decay = 0.99, double epsilon = 1e-5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    // `learning_rate` is deliberately UNUSED -- see the class comment. Passing
    // `training = false` leaves both the codebook and the statistics
    // untouched, so evaluation never pollutes the running averages.
    //
    // NOTE there is deliberately NO default argument here: a defaulted
    // `update_weights(double, bool = true)` alongside the virtual
    // `update_weights(double)` is an AMBIGUOUS overload for every one-argument
    // call, and the repo compiles with -Wall -Wextra. Two names, no defaults.
    void update_weights(double learning_rate, bool training);
    void update_weights(double learning_rate) override {
        update_weights(learning_rate, true);
    }
    void zero_grad() override;

    Tensor get_weights() const override { return codebook_; }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::vector<Tensor*> parameters() override { return {&codebook_}; }
    std::vector<Tensor*> gradients() override { return {}; }
    std::string name() const override { return "VectorQuantizerEMA"; }

    void set_codebook(const Tensor& w);
    // Replace the EMA statistics (cluster sizes and accumulated embeddings).
    void set_ema_state(const Tensor& cluster_size, const Tensor& ema_w);
    Tensor codebook() const { return codebook_.clone(); }
    Tensor cluster_size() const { return cluster_size_.clone(); }

    double codebook_loss() const { return 0.0; }        // dropped (D6)
    double commitment_loss() const { return commitment_loss_; }
    double total_loss() const { return commitment_loss_; }
    double perplexity() const { return perplexity_; }
    const std::vector<size_t>& indices() const { return indices_; }
    size_t usage_count() const { return used_codes_.size(); }
    size_t dead_code_count() const { return dead_codes_.size(); }

    double beta() const { return beta_; }
    double decay() const { return decay_; }
    double epsilon() const { return epsilon_; }

private:
    size_t embedding_dim_, num_embeddings_;
    double beta_, decay_, epsilon_;

    Tensor codebook_;        // (D, K)
    Tensor cluster_size_;    // (K)
    Tensor ema_w_;           // (D, K)

    Tensor last_z_e_, last_z_q_;
    std::vector<size_t> indices_;
    std::vector<size_t> used_codes_, dead_codes_;
    double commitment_loss_ = 0.0, perplexity_ = 0.0;
};

// ---------------------------------------------------------------------------
// VQVAE -- encoder -> VectorQuantizer -> decoder, as one Layer.
//
// The encoder is a single Dense to `embedding_dim`; the decoder a Dense to
// `hidden_dim` plus a Dense back to `input_dim`. The quantizer sits between
// them and is what makes this a VQ-VAE rather than an autoencoder: the
// decoder receives z_q, never z_e.
// ---------------------------------------------------------------------------
class VQVAE : public Layer {
public:
    VQVAE(size_t input_dim, size_t hidden_dim, size_t embedding_dim,
          size_t num_embeddings, double beta = 0.25);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return quantizer_.get_weights(); }
    Tensor get_gradients() const override { return quantizer_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "VQVAE"; }

    // Expose the two VQ terms so a caller can log codebook health during
    // training without reaching into the member.
    double codebook_loss() const { return quantizer_.codebook_loss(); }
    double commitment_loss() const { return quantizer_.commitment_loss(); }
    double perplexity() const { return quantizer_.perplexity(); }
    size_t usage_count() const { return quantizer_.usage_count(); }
    size_t dead_code_count() const { return quantizer_.dead_code_count(); }

    VectorQuantizer& quantizer() { return quantizer_; }

private:
    size_t input_dim_, hidden_dim_, embedding_dim_, num_embeddings_;

    Dense encoder_, decoder_hidden_, decoder_out_;
    VectorQuantizer quantizer_;

    Tensor last_input_, last_z_e_, last_h_dec_;
};

#endif // NN_LAYERS_GENERATIVE_VQ_VAE_H