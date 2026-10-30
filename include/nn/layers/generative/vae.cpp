#include "vae.h"

VAE::VAE(size_t input_dim, size_t hidden_dim, size_t latent_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), latent_dim_(latent_dim),
      enc_fc1_(input_dim, hidden_dim),
      enc_fc_mu_(hidden_dim, latent_dim),
      enc_fc_logvar_(hidden_dim, latent_dim),
      dec_fc1_(latent_dim, hidden_dim),
      dec_fc_out_(hidden_dim, input_dim),
      rng_(42) {}

Tensor VAE::reparameterize(const Tensor& mu, const Tensor& logvar) {
    // z = mu + sigma * epsilon, where sigma = exp(0.5 * logvar)
    Tensor z(mu.rows, mu.cols);
    for (size_t i = 0; i < mu.rows; ++i) {
        for (size_t j = 0; j < mu.cols; ++j) {
            double sigma = std::exp(0.5 * logvar[i][j]);
            double epsilon = normal_dist_(rng_);
            z[i][j] = mu[i][j] + sigma * epsilon;
        }
    }
    z_placeholder_ = z;
    return z;
}

Tensor VAE::forward(const Tensor& input) {
    // Encode: x -> h -> {mu, logvar}
    Tensor h = enc_fc1_.forward(input);
    h = h.apply([](double x) { return std::max(0.0, x); }); // ReLU
    last_h_ = h;  // cache for backward

    last_mu_ = enc_fc_mu_.forward(h);
    last_logvar_ = enc_fc_logvar_.forward(h);

    // Sample latent
    Tensor z = reparameterize(last_mu_, last_logvar_);

    // Decode: z -> h -> recon
    last_h_dec_ = dec_fc1_.forward(z);
    last_h_dec_ = last_h_dec_.apply([](double x) { return std::max(0.0, x); }); // ReLU
    last_recon_ = dec_fc_out_.forward(last_h_dec_);

    // Compute losses
    recon_loss_ = 0.0;
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            recon_loss_ += (last_recon_[i][j] - input[i][j]) * (last_recon_[i][j] - input[i][j]);
    recon_loss_ /= input.rows;

    kl_loss_ = 0.0;
    for (size_t i = 0; i < last_mu_.rows; ++i)
        for (size_t j = 0; j < last_mu_.cols; ++j)
            kl_loss_ += -0.5 * (1.0 + last_logvar_[i][j] - last_mu_[i][j] * last_mu_[i][j] - std::exp(last_logvar_[i][j]));
    // THE KL IS A MEAN OVER THE BATCH, exactly as the reconstruction loss
    // above already is.
    //
    // **IT USED TO BE A SUM, AND THE TWO COMPONENTS OF total_loss() WERE
    // THEREFORE ON DIFFERENT SCALES.** Measured before this fix: a 2-row
    // batch of identical data gave a KL of exactly TWICE the 1-row batch's
    // (0.751304738877 against 0.375652369439 -- a ratio of exactly 2.0),
    // while the reconstruction loss stayed O(1). So the total loss changed
    // with the batch size even for fixed data, and the effective beta (the
    // KL's weight, hard-coded to 1.0 in total_loss()) was a property of
    // the BATCH SIZE rather than of the loss.
    //
    // **THE ARGUMENT FOR THE MEAN IS NOT A MATTER OF TASTE.** The KL is a
    // function of `mu` and `logvar` ONLY -- the reparameterisation's noise
    // does not enter it (measured: two forwards of the same input give a
    // BIT-IDENTICAL KL while the reconstruction loss differs). So the
    // reconstruction term is a MEAN over a STOCHASTIC quantity and the KL
    // was a SUM over a DETERMINISTIC one, and a total loss whose
    // batch-scaling depends on which term you look at is not a coherent
    // per-sample objective.
    //
    // The repo's own convention agrees: `utils/losses.h` documents that the
    // mean reduction divides by `numel`, NOT by `rows` (D7), and
    // `utils/contrastive_losses.cpp` divides its total by `X.rows`. **A
    // LOSS THAT SCALES WITH THE BATCH SIZE IS NOT A PER-SAMPLE QUANTITY.**
    //
    // Pinned by tests/test_vae.cpp: a 2-row batch of identical rows must
    // give the SAME KL as a 1-row batch of one of them.
    kl_loss_ /= static_cast<double>(input.rows);

    return last_recon_;
}

Tensor VAE::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    // ---- 1. Backprop reconstruction loss through decoder ----
    // dL/dx_recon = grad_output * 2 * (recon - input) / N
    // Since we don't have target input here, we propagate grad_output directly.
    Tensor grad_recon = grad_output;

    // Decoder backward: dec_fc_out_ -> ReLU-gated dec_fc1_ -> reparam
    Tensor grad_h_dec = dec_fc_out_.backward(grad_recon, 0.0);  // dL/d(h_dec)

    // ReLU backward: gate with (last_h_dec_ > 0)
    for (size_t i = 0; i < grad_h_dec.rows; ++i)
        for (size_t j = 0; j < grad_h_dec.cols; ++j)
            if (last_h_dec_[i][j] <= 0.0) grad_h_dec[i][j] = 0.0;

    // dL/dz = grad through dec_fc1_ weights
    Tensor grad_z = dec_fc1_.backward(grad_h_dec, 0.0);  // dL/dz

    // ---- 2. Backprop through reparameterization + KL loss ----
    // KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    //
    // **THE SIGN ON THE mu TERM WAS INVERTED, AND THE 1/N WAS MISSING.**
    // The closed form carries `- mu^2` with a MINUS in front of it, so
    //
    //     d/dmu [ -0.5 * (1 + logvar - mu^2 - exp(logvar)) ]
    //         = -0.5 * ( -2*mu )
    //         = +mu
    //
    // and the old code used `-mu_ij` -- with a comment asserting the sign
    // was negative. **A COMMENT THAT ASSERTS A SIGN IS NOT EVIDENCE OF
    // ONE.** Measured against the finite difference, the analytical
    // gradient's ratio to the true one was exactly -1.0 for a 1-row batch,
    // -2.0 for 2 rows and -3.0 for 3.
    //
    // **AND THE SIGN WAS NOT COSMETIC.** The KL is minimised at mu = 0, so
    // the gradient must have the SAME SIGN as mu, and gradient descent
    // (which subtracts the gradient) must move mu toward zero. With the
    // sign inverted it moved AWAY: measured, one training step took the
    // mu-bias from +0.300000 to +0.542474 -- the KL term was actively
    // pushing the encoder away from the prior.
    //
    // The 1/N is the batch mean that `forward()` now applies to the loss
    // (see the note there), and the gradient must match its loss. Measured
    // ratios for the logvar gradient were +1.0, +2.0, +3.0 for N = 1, 2, 3
    // -- the sign was right and the 1/N was missing.
    //
    // Reparam: z = mu + exp(0.5*logvar) * eps,  eps = (z - mu) / exp(0.5*logvar)
    // dL/dmu += dL/dz (direct) + KL contribution
    // dL/dlogvar += dL/dz * 0.5 * (z - mu) + KL contribution

    size_t latent = last_mu_.rows;
    size_t latent_cols = last_mu_.cols;
    // The batch size, for the 1/N that matches the mean-form loss.
    const double inv_batch = 1.0 / static_cast<double>(last_mu_.rows);
    Tensor grad_mu_enc(latent, latent_cols);
    Tensor grad_logvar_enc(latent, latent_cols);

    for (size_t i = 0; i < latent; ++i) {
        for (size_t j = 0; j < latent_cols; ++j) {
            double mu_ij = last_mu_[i][j];
            double logvar_ij = last_logvar_[i][j];
            double z_ij = z_placeholder_[i][j];
            double sigma = std::exp(0.5 * logvar_ij);

            // KL divergence gradient contributions
            // KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
            // dKL/dmu = +mu_ij  (the closed form's -mu^2 carries a minus,
            //                     so the derivative is POSITIVE)
            grad_mu_enc[i][j] = mu_ij * inv_batch;
            // dKL/dlogvar = 0.5 * (exp(logvar) - 1.0)
            grad_logvar_enc[i][j] = 0.5 * (std::exp(logvar_ij) - 1.0) * inv_batch;

            // Reparameterization gradient contributions
            // dL/dmu += grad_z (direct path z=mu+...)
            grad_mu_enc[i][j] += grad_z[i][j];
            // dL/dlogvar += grad_z * 0.5 * (z - mu)  [via eps path: d(z)/d(logvar) = 0.5*(z-mu)]
            grad_logvar_enc[i][j] += grad_z[i][j] * 0.5 * (z_ij - mu_ij);
        }
    }

    // ---- 3. Backprop through encoder ----
    // grad_mu_enc and grad_logvar_enc are dL/d(enc_fc_mu output) and dL/d(enc_fc_logvar output)
    // enc_fc_mu_.backward and enc_fc_logvar_.backward use their last_input (= last_h_)
    Tensor grad_h = enc_fc_mu_.backward(grad_mu_enc, 0.0);
    Tensor grad_h2 = enc_fc_logvar_.backward(grad_logvar_enc, 0.0);
    for (size_t i = 0; i < grad_h.rows; ++i)
        for (size_t j = 0; j < grad_h.cols; ++j)
            grad_h[i][j] += grad_h2[i][j];

    // ReLU backward on enc_fc1_ output
    for (size_t i = 0; i < grad_h.rows; ++i)
        for (size_t j = 0; j < grad_h.cols; ++j)
            if (last_h_[i][j] <= 0.0) grad_h[i][j] = 0.0;

    // Backprop through enc_fc1_ to get input gradient
    Tensor grad_input = enc_fc1_.backward(grad_h, 0.0);
    return grad_input;
}

void VAE::update_weights(double learning_rate) {
    enc_fc1_.update_weights(learning_rate);
    enc_fc_mu_.update_weights(learning_rate);
    enc_fc_logvar_.update_weights(learning_rate);
    dec_fc1_.update_weights(learning_rate);
    dec_fc_out_.update_weights(learning_rate);
}

void VAE::zero_grad() {
    enc_fc1_.zero_grad();
    enc_fc_mu_.zero_grad();
    enc_fc_logvar_.zero_grad();
    dec_fc1_.zero_grad();
    dec_fc_out_.zero_grad();
}

std::vector<Tensor*> VAE::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : enc_fc1_.parameters()) result.push_back(p);
    for (Tensor* p : enc_fc_mu_.parameters()) result.push_back(p);
    for (Tensor* p : enc_fc_logvar_.parameters()) result.push_back(p);
    for (Tensor* p : dec_fc1_.parameters()) result.push_back(p);
    for (Tensor* p : dec_fc_out_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> VAE::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : enc_fc1_.gradients()) result.push_back(g);
    for (Tensor* g : enc_fc_mu_.gradients()) result.push_back(g);
    for (Tensor* g : enc_fc_logvar_.gradients()) result.push_back(g);
    for (Tensor* g : dec_fc1_.gradients()) result.push_back(g);
    for (Tensor* g : dec_fc_out_.gradients()) result.push_back(g);
    return result;
}

Tensor VAE::sample(const Tensor& input) {
    Tensor h = enc_fc1_.forward(input);
    h = h.apply([](double x) { return std::max(0.0, x); });
    Tensor mu = enc_fc_mu_.forward(h);
    Tensor logvar = enc_fc_logvar_.forward(h);
    Tensor z = reparameterize(mu, logvar);

    Tensor h_dec = dec_fc1_.forward(z);
    h_dec = h_dec.apply([](double x) { return std::max(0.0, x); });
    return dec_fc_out_.forward(h_dec);
}