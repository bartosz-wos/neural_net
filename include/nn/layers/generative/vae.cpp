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

    Tensor mu = enc_fc_mu_.forward(h);
    Tensor logvar = enc_fc_logvar_.forward(h);

    // Sample latent
    Tensor z = reparameterize(mu, logvar);

    // Decode: z -> h -> recon
    Tensor h_dec = dec_fc1_.forward(z);
    h_dec = h_dec.apply([](double x) { return std::max(0.0, x); }); // ReLU
    last_recon_ = dec_fc_out_.forward(h_dec);

    // Compute losses
    recon_loss_ = 0.0;
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            recon_loss_ += (last_recon_[i][j] - input[i][j]) * (last_recon_[i][j] - input[i][j]);
    recon_loss_ /= input.rows;

    kl_loss_ = 0.0;
    for (size_t i = 0; i < mu.rows; ++i)
        for (size_t j = 0; j < mu.cols; ++j)
            kl_loss_ += -0.5 * (1.0 + logvar[i][j] - mu[i][j] * mu[i][j] - std::exp(logvar[i][j]));

    return last_recon_;
}

Tensor VAE::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, input_dim_); // placeholder
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