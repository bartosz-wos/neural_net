#ifndef VAE_H
#define VAE_H

#include "../../core/layer.h"
#include <vector>
#include <random>

// Variational Autoencoder — encoder maps input to latent mean+logvar,
// reparameterization trick samples latent, decoder reconstructs.
// Loss = reconstruction_loss + KL_divergence
//
// ---------------------------------------------------------------------------
// THE LOSS CONVENTIONS, AND WHY THEY ARE WHAT THEY ARE
// ---------------------------------------------------------------------------
//
// **BOTH COMPONENTS OF `total_loss()` ARE PER-SAMPLE MEANS.** The
// reconstruction loss is the mean over the batch of the per-row squared
// error, and the KL is the mean over the batch of the per-row KL. This was
// NOT true of the KL until 2026-10-29: it was a SUM over the batch while
// the reconstruction loss was a MEAN, so the two components of
// `total_loss()` were on different scales AND the total loss changed with
// the batch size even for fixed data -- which made the effective beta (the
// KL's weight, hard-coded to 1.0 here) a property of the BATCH SIZE rather
// than of the loss.
//
// The argument is not a matter of taste:
//
//   * The repo's own convention agrees. `utils/losses.h` documents that the
//     mean reduction divides by `numel`, NOT by `rows` (D7), and
//     `utils/contrastive_losses.cpp` divides its total by `X.rows`. **A
//     LOSS THAT SCALES WITH THE BATCH SIZE IS NOT A PER-SAMPLE QUANTITY.**
//   * The KL is a function of `mu` and `logvar` ONLY -- the
//     reparameterisation's noise does not enter it (measured: two forwards
//     of the same input give a BIT-IDENTICAL KL while the reconstruction
//     loss differs). So the reconstruction term is a MEAN over a
//     STOCHASTIC quantity and the KL was a SUM over a DETERMINISTIC one.
//
// Pinned by tests/test_vae.cpp: a 2-row batch of identical rows must give
// the SAME KL as a 1-row batch of one of them.
//
// **THE GRADIENT MATCHES THE LOSS.** `backward()`'s KL contribution is
// `(+mu) / N` and `0.5*(exp(logvar) - 1) / N`. The mu term's SIGN was also
// wrong until 2026-10-29 -- the closed form's `- mu^2` carries a minus, so
// the derivative is POSITIVE, and the code used `-mu` with a comment
// asserting the sign was negative. **A COMMENT THAT ASSERTS A SIGN IS NOT
// EVIDENCE OF ONE.** With the sign inverted, gradient descent moved `mu`
// AWAY from zero -- the direction that INCREASES the KL -- so the KL term
// was actively pushing the encoder away from the prior.
//
// Pinned by tests/test_vae_grad.cpp, against finite differences on the
// encoder's BIAS (which shifts mu and logvar directly, so no weight path or
// ReLU is confounded with the KL's own contribution).
//
// ---------------------------------------------------------------------------
// THE FINITE-DIFFERENCE CAVEAT, STATED SO A FUTURE READER DOES NOT WASTE
// A SESSION ON IT
// ---------------------------------------------------------------------------
//
// **THE FINITE DIFFERENCE OF `total_loss()` IS NOT A VALID DERIVATIVE.** The
// reparameterisation draws a fresh `eps ~ N(0,1)` on every forward, so the
// central difference of `total_loss()` compares two UNCORRELATED noisy
// outputs. The finite difference of `kl_loss()` alone IS valid, because the
// KL does not depend on the noise. This is the repo's recorded
// "stochastic forward" trap.
class VAE : public Layer {
public:
    VAE(size_t input_dim, size_t hidden_dim, size_t latent_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    // A VAE has MANY parameter tensors (encoder fc1, encoder mu, encoder
    // logvar, decoder fc1, decoder out), so there is no single weight matrix
    // to expose. The empty tensor is this repo's established answer for a
    // multi-parameter layer -- see `architectures/resnet.h:15-16`. WITHOUT
    // THESE TWO OVERRIDES THE CLASS IS ABSTRACT, because `Layer` declares
    // `get_weights`/`get_gradients` pure virtual.
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    // Sample from latent space (encoder -> reparameterize -> decoder)
    Tensor sample(const Tensor& input);

    // Getters for loss components
    double reconstruction_loss() const { return recon_loss_; }
    double kl_loss() const { return kl_loss_; }
    double total_loss() const { return recon_loss_ + kl_loss_; }

private:
    size_t input_dim_, hidden_dim_, latent_dim_;

    // Encoder: input -> hidden -> {mean, logvar}
    Dense enc_fc1_, enc_fc_mu_, enc_fc_logvar_;

    // Decoder: latent -> hidden -> reconstruction
    Dense dec_fc1_, dec_fc_out_;

    double recon_loss_ = 0.0, kl_loss_ = 0.0;
    std::normal_distribution<double> normal_dist_{0, 1};
    std::mt19937 rng_;

    Tensor z_placeholder_; // sampled latent for reparameterization
    Tensor last_recon_;    // reconstruction output

    // Cached activations for backward pass
    Tensor last_h_;        // encoder ReLU output (enc_fc1_ output)
    Tensor last_mu_;       // encoder mean output
    Tensor last_logvar_;   // encoder logvar output
    Tensor last_h_dec_;    // decoder ReLU output (dec_fc1_ output)

    Tensor reparameterize(const Tensor& mu, const Tensor& logvar);
};

#endif