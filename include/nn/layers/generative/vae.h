#ifndef VAE_H
#define VAE_H

#include "../../core/layer.h"
#include <vector>
#include <random>

// Variational Autoencoder — encoder maps input to latent mean+logvar,
// reparameterization trick samples latent, decoder reconstructs.
// Loss = reconstruction_loss + KL_divergence
class VAE : public Layer {
public:
    VAE(size_t input_dim, size_t hidden_dim, size_t latent_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

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

    Tensor reparameterize(const Tensor& mu, const Tensor& logvar);
};

#endif