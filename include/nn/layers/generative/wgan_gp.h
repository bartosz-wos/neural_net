#ifndef WGAN_GP_H
#define WGAN_GP_H

#include "../../core/tensor.h"
#include "../../core/layer.h"
#include "../../activations/activations.h"
#include <vector>
#include <random>
#include <memory>

// =====================================================================
// WGANDiscriminator (Critic) — no sigmoid, outputs raw score
// =====================================================================
class WGANDiscriminator : public Layer {
public:
    WGANDiscriminator(size_t input_dim, size_t hidden_dim, size_t num_layers = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::string name() const override { return "WGANDiscriminator"; }

    // Custom backward from a pre-computed output gradient (d_loss/d_output).
    // Accumulates into weight gradients without going through the standard
    // single-last_input interface.
    void backward_from(const Tensor& grad_output);

    // Compute gradient penalty: GP = E_xhat~Unif [ (||grad_xhat|| - 1)^2 ]
    // Returns the penalty value; also caches x_hat for backward.
    double gradient_penalty(const Tensor& real, const Tensor& fake, double lambda = 10.0);

    // Backward through gradient penalty: d(GP)/d(critic_input).
    // Must be called AFTER gradient_penalty().
    void backward_gradient_penalty();

    // Reset cached inputs (call before a new D forward sequence)
    void reset_cached_inputs();

    // Accessors for trainer
    const Tensor& last_output() const { return last_output_; }
    size_t num_layers() const { return layers_.size(); }
    Dense& layer(size_t i) { return layers_[i]; }
    const Dense& layer(size_t i) const { return layers_[i]; }
    size_t layer_input_size(size_t i) const;
    size_t layer_output_size(size_t i) const;
    LeakyReLU& activation(size_t i) { return activations_[i]; }

private:
    std::vector<Dense> layers_;
    std::vector<LeakyReLU> activations_;
    std::vector<Tensor> cached_inputs_;   // input at each layer
    Tensor last_output_;
    Tensor last_x_hat_;                   // cached for gradient penalty
    std::vector<Tensor> grad_x_hat_layers_; // grad w.r.t. each layer's input (for GP)
    std::mt19937 rng_;                    // RNG for gradient penalty sampling
};

// =====================================================================
// WGANGenerator — latent z -> fake sample
// =====================================================================
class WGANGenerator : public Layer {
public:
    WGANGenerator(size_t latent_dim, size_t hidden_dim, size_t output_dim,
                  size_t num_layers = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::string name() const override { return "WGANGenerator"; }

    // Accessors for trainer
    size_t num_layers() const { return layers_.size(); }
    Dense& layer(size_t i) { return layers_[i]; }
    const Dense& layer(size_t i) const { return layers_[i]; }

    size_t latent_dim() const { return latent_dim_; }
    size_t output_dim() const { return output_dim_; }

private:
    size_t latent_dim_, hidden_dim_, output_dim_, num_layers_;
    std::vector<Dense> layers_;
    std::vector<LeakyReLU> activations_;
    std::vector<Tensor> cached_inputs_;
    Tensor last_output_;
};

// =====================================================================
// GradientPenaltyLoss — standalone loss for R1 / WGAN-GP style penalties
// =====================================================================
class GradientPenaltyLoss {
public:
    // Compute gradient penalty.
    // For WGAN-GP (r1_style=false): penalty = E_xhat [ (||grad_xhat|| - 1)^2 ]
    //   where grad_xhat = d(model_output) / d(x_hat)
    //   and model_output = critic(x_hat)
    // For R1 (r1_style=true): penalty = E_x [ ||grad_x||^2 ]
    //   where grad_x = d(critic(x)) / d(x) for real samples x
    //
    // Returns the scalar penalty value.
    static double compute(const Tensor& x, const Tensor& model_output,
                         const Tensor& grad_wrt_x_hat,
                         bool r1_style = false);

    // Compute R1 gradient penalty directly.
    // d_loss/d_x = grad_wrt_x^T * model_jacobian
    // penalty = ||d_loss/d_x||^2 / N
    static double compute_r1(const Tensor& real_samples,
                             const Tensor& grad_wrt_real);
};

// =====================================================================
// WGANTrainer — orchestrates WGAN training with gradient penalty
// =====================================================================
class WGANTrainer {
public:
    // clip_value: weight clipping for basic WGAN (not needed with GP, use 0.0 to disable)
    WGANTrainer(double clip_value = 0.0, unsigned seed = 42);

    // Single training step.
    // Returns {d_loss_value, g_loss_value, gp_value}
    // - n_critic: number of D updates per G update (default 5)
    void train_step(WGANDiscriminator& critic,
                    WGANGenerator& generator,
                    const Tensor& real_batch,
                    size_t latent_dim,
                    double lambda_gp = 10.0,
                    int n_critic = 5);

    // Utility: generate random latent samples
    Tensor sample_latent(size_t batch_size, size_t latent_dim);

    void set_lambda_gp(double lambda) { lambda_gp_ = lambda; }
    double lambda_gp() const { return lambda_gp_; }

private:
    double lambda_gp_ = 10.0;
    double clip_value_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    std::normal_distribution<double> normal_{0.0, 1.0};

    // Clip critic weights (basic WGAN, not needed with GP)
    void clip_weights(WGANDiscriminator& critic, double clip_val);

    // Compute D loss = E[D(real)] - E[D(fake)] + lambda * GP
    double compute_d_loss(const Tensor& d_real, const Tensor& d_fake,
                          double gp) const;

    // Compute G loss = -E[D(fake)]
    double compute_g_loss(const Tensor& d_fake) const;
};

// =====================================================================
// WGANModel — combines D and G into a single trainable Model
// =====================================================================
class WGANModel {
public:
    WGANModel(size_t latent_dim, size_t data_dim,
              size_t hidden_dim_d = 512, size_t hidden_dim_g = 512,
              size_t num_layers_d = 3, size_t num_layers_g = 3,
              unsigned seed = 42);

    WGANDiscriminator& critic() { return *critic_; }
    WGANGenerator& generator() { return *generator_; }
    const WGANDiscriminator& critic() const { return *critic_; }
    const WGANGenerator& generator() const { return *generator_; }

    // Single training step
    std::tuple<double, double, double> train_step(
        const Tensor& real_batch,
        double lambda_gp = 10.0,
        int n_critic = 5);

    // Generate samples from noise
    Tensor generate(const Tensor& z) const;
    Tensor generate(size_t num_samples) const;

    void set_lambda_gp(double lambda) { trainer_.set_lambda_gp(lambda); }

private:
    std::unique_ptr<WGANDiscriminator> critic_;
    std::unique_ptr<WGANGenerator> generator_;
    WGANTrainer trainer_;
};

#endif // WGAN_GP_H
