#include "wgan_gp.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// =====================================================================
// Helper: accumulate weight gradients directly for a Dense layer.
// grad_output: (batch, out_features)
// input:       (batch, in_features) — the input associated with this grad
// =====================================================================
static void accumulate_dense_grad(Dense& layer, const Tensor& grad_output,
                                  const Tensor& input) {
    if (input.rows == 0 || input.cols == 0) return;
    // grad_w += grad_output^T * input
    // grad_output: (batch, out_i), input: (batch, in_i) -> (out_i, in_i)
    // grad_w[m][n] = sum_r grad_output[r][m] * input[r][n]
    Tensor grad_w(grad_output.cols, input.cols);
    for (size_t m = 0; m < grad_w.rows; ++m) {
        for (size_t n = 0; n < grad_w.cols; ++n) {
            double sum = 0.0;
            for (size_t r = 0; r < grad_output.rows; ++r) {
                sum += grad_output[r][m] * input[r][n];
            }
            grad_w[m][n] = sum;
        }
    }
    layer.grad_weights += grad_w;

    // Bias gradient: sum over batch for each output unit
    for (size_t j = 0; j < grad_output.cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_output.rows; ++i)
            sum += grad_output[i][j];
        layer.grad_bias[0][j] += sum;
    }
}

// =====================================================================
// WGANDiscriminator (Critic)
// - input_dim -> hidden -> ... -> hidden -> 1 (no sigmoid, raw score)
// =====================================================================
WGANDiscriminator::WGANDiscriminator(size_t input_dim, size_t hidden_dim,
                                     size_t num_layers)
    : rng_(42)
{
    layers_.reserve(num_layers + 1);
    activations_.reserve(num_layers + 1);
    cached_inputs_.reserve(num_layers + 1);

    size_t dim = input_dim;
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.emplace_back(dim, hidden_dim);
        activations_.emplace_back(0.01);  // LeakyReLU slope = 0.01
        cached_inputs_.emplace_back();    // empty placeholder
        dim = hidden_dim;
    }
    // Final layer: hidden_dim -> 1 (no activation)
    layers_.emplace_back(dim, 1);
    activations_.emplace_back(0.01);
    cached_inputs_.emplace_back();
}

size_t WGANDiscriminator::layer_input_size(size_t i) const {
    return layers_[i].weights.cols;
}

size_t WGANDiscriminator::layer_output_size(size_t i) const {
    return layers_[i].weights.rows;
}

Tensor WGANDiscriminator::forward(const Tensor& input) {
    // Cache input for each layer
    cached_inputs_[0] = input;

    Tensor x = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        // Forward through Dense
        x = layers_[i].forward(x);
        // Cache the pre-activation output for layer i+1 (input to next layer)
        if (i + 1 < cached_inputs_.size())
            cached_inputs_[i + 1] = x;
        // Apply LeakyReLU for all but the last layer
        if (i < layers_.size() - 1)
            x = activations_[i](x);
    }
    last_output_ = x;
    return x;
}

void WGANDiscriminator::reset_cached_inputs() {
    for (auto& c : cached_inputs_)
        c = Tensor(0, 0);
}

void WGANDiscriminator::backward_from(const Tensor& grad_output) {
    // Accumulate weight gradients using the cached inputs from forward passes.
    // grad_output: (batch, 1) — d_loss/d(output) for each sample
    // We backprop through each layer, accumulating into the layer gradients.
    // grad: d_loss/d(output of layer i) = dL/d(post-activation of layer i)
    //
    // For weight gradients, we need dL/d(pre-activation of layer i), not dL/d(post-activation).
    // dL/d(pre-activation) = dL/d(post-activation) * leakyrelu'(pre-activation)
    //
    // The Dense::backward gives us grad_input = dL/d(pre-activation of previous layer),
    // which is correct for propagation. But for weight gradients in accumulate_dense_grad,
    // we need to multiply by leakyrelu'(pre_i) before the outer product.
    Tensor grad = grad_output;

    for (size_t i = layers_.size(); i-- > 0; ) {
        // Compute dL/d(pre-activation) by applying activation derivative
        // For output layer: dL/dpre = dL/dpost (identity)
        // For hidden layers: dL/dpre = dL/dpost * leakyrelu'(pre)
        //
        // cached_inputs_[0] = network input (to layer 0)
        // cached_inputs_[i+1] = pre-activation output of layer i (before leakyrelu)
        // So for layer i's leakyrelu, we need cached_inputs_[i+1]
        Tensor grad_pre = grad;
        if (i < layers_.size() - 1) {
            double alpha = activations_[i].slope;
            const Tensor& pre_i = cached_inputs_[i + 1];  // pre-activation for layer i
            for (size_t r = 0; r < grad_pre.rows; ++r) {
                for (size_t c = 0; c < grad_pre.cols; ++c) {
                    double pre_val = pre_i(r, c);
                    double slope = (pre_val > 0.0) ? 1.0 : alpha;
                    grad_pre(r, c) *= slope;
                }
            }
        }

        // Accumulate weight gradients: grad_w += grad_pre^T * input
        // For weight gradients, we need the activation OUTPUT (h), not pre-activation (pre)
        // cached_inputs_[i] = pre-activation output of layer i-1 (for i > 0)
        // cached_inputs_[0] = network input (for layer 0)
        // So for layer 0: input = cached_inputs_[0] (real)
        // For layer i > 0: input = leakyrelu(cached_inputs_[i])
        if (cached_inputs_[i].rows > 0 && cached_inputs_[i].cols > 0) {
            Tensor input_for_grad = cached_inputs_[i];
            // For layers > 0, apply activation to get the actual input
            if (i > 0) {
                double alpha = activations_[i - 1].slope;  // activation for previous layer
                for (size_t r = 0; r < input_for_grad.rows; ++r) {
                    for (size_t c = 0; c < input_for_grad.cols; ++c) {
                        double val = input_for_grad(r, c);
                        input_for_grad(r, c) = (val > 0.0) ? val : alpha * val;
                    }
                }
            }
            accumulate_dense_grad(layers_[i], grad_pre, input_for_grad);
        }

        // Propagate gradient to previous layer's output (input of current layer)
        if (i > 0) {
            grad = layers_[i].backward(grad, 0.0);  // d_loss/d(input of layer i) = dL/d(pre-activation of layer i-1)
        }
    }
}

double WGANDiscriminator::gradient_penalty(const Tensor& real, const Tensor& fake,
                                           double lambda) {
    // Sample alpha ~ Uniform(0, 1)
    size_t batch = real.rows;
    size_t input_dim = real.cols;
    Tensor alpha(batch, 1);
    for (size_t i = 0; i < batch; ++i)
        alpha[i][0] = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);

    // Interpolation: x_hat = alpha * real + (1 - alpha) * fake
    // Tile alpha from (batch, 1) to (batch, input_dim) for proper broadcasting
    Tensor alpha_tiled(batch, input_dim);
    Tensor one_minus_alpha_tiled(batch, input_dim);
    for (size_t i = 0; i < batch; ++i) {
        double a = alpha[i][0];
        double om = 1.0 - a;
        for (size_t j = 0; j < input_dim; ++j) {
            alpha_tiled[i][j] = a;
            one_minus_alpha_tiled[i][j] = om;
        }
    }
    last_x_hat_ = alpha_tiled.hadamard(real) + one_minus_alpha_tiled.hadamard(fake);

    // Forward through critic on x_hat
    // Cached inputs are set inside forward()
    Tensor d_x_hat = forward(last_x_hat_); // last_x_hat_ cached via forward

    // Compute gradient of D(x_hat) w.r.t. x_hat.
    // We backprop ones through the critic.
    // grad_x_hat_layers_[k] = d(D(x_hat))/d(cached_input of layer k)
    grad_x_hat_layers_.resize(layers_.size());

    Tensor grad_ones(batch, 1);
    grad_ones.fill(1.0);
    Tensor grad = grad_ones;

    for (size_t i = layers_.size(); i-- > 0; ) {
        if (cached_inputs_[i].rows > 0 && cached_inputs_[i].cols > 0) {
            grad_x_hat_layers_[i] = layers_[i].backward(grad, 0.0);
            grad = grad_x_hat_layers_[i];
        } else {
            grad_x_hat_layers_[i] = Tensor(0, 0);
        }
    }
    // grad_x_hat_layers_[0] = d(D(x_hat))/d(x_hat)

    // Compute ||grad|| for each sample and the penalty
    double total_penalty = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        double norm_sq = 0.0;
        for (size_t j = 0; j < grad_x_hat_layers_[0].cols; ++j) {
            double g = grad_x_hat_layers_[0][i][j];
            norm_sq += g * g;
        }
        double norm = std::sqrt(std::max(norm_sq, 1e-12));
        double diff = norm - 1.0;
        total_penalty += diff * diff;
    }

    return lambda * total_penalty / static_cast<double>(batch);
}

void WGANDiscriminator::backward_gradient_penalty() {
    // Gradient penalty backward pass.
    //
    // For each sample r:
    //   coeff_r = 2 * (||grad_x_hat_layers_[0][r]|| - 1) / ||grad_x_hat_layers_[0][r]||
    //
    // d(GP)/d(W_i) = sum_r coeff_r * d(D)/d(W_i)[r]
    //               = sum_r coeff_r * cached_inputs_[i][r]^T  [for output layer]
    //
    // d(GP)/d(pre_i) propagation:
    //   d(GP)/d(pre_i) = d(GP)/d(pre_{i+1}) @ W_{i+1} * leakyReLU'
    //
    // Cached inputs mapping:
    //   cached_inputs_[0] = input to layer 0 (x)
    //   cached_inputs_[i] = output of layer i-1 = pre_{i-1} (input to layer i)
    size_t batch = last_x_hat_.rows;


    // coeff: per-sample gradient penalty coefficient, shape (batch, 1)
    Tensor coeff(batch, 1);
    for (size_t r = 0; r < batch; ++r) {
        double norm_sq = 0.0;
        for (size_t c = 0; c < grad_x_hat_layers_[0].cols; ++c) {
            double g = grad_x_hat_layers_[0][r][c];
            norm_sq += g * g;
        }
        double norm = std::sqrt(std::max(norm_sq, 1e-12));
        double diff = norm - 1.0;
        coeff[r][0] = 2.0 * diff / norm;
    }

    // grad_pre: d(GP)/d(pre_i) — starts as d(GP)/d(pre_n) for last layer
    // For last layer: d(GP)/d(pre_n) = coeff * 1 (identity Jacobian for scalar output)
    // grad_pre is also used to compute d(GP)/d(pre_{i-1}) = d(GP)/d(pre_i) @ W_i
    Tensor grad_pre(batch, layers_.back().weights.rows);
    for (size_t r = 0; r < batch; ++r) {
        grad_pre[r][0] = coeff[r][0];
    }

    for (size_t i = layers_.size(); i-- > 0; ) {
        if (cached_inputs_[i].rows > 0 && cached_inputs_[i].cols > 0) {
            const Dense& layer = layers_[i];
            size_t out_i = layer.weights.rows;
            size_t in_i = layer.weights.cols;

            // d(GP)/d(W_i)[m][n] = sum_r coeff[r] * grad_x_hat_layers_[i][r][m] * cached_inputs_[i][r][n]
            Tensor grad_w(out_i, in_i);
            for (size_t m = 0; m < out_i; ++m) {
                for (size_t n = 0; n < in_i; ++n) {
                    double sum = 0.0;
                    for (size_t r = 0; r < batch; ++r) {
                        sum += coeff[r][0] * grad_x_hat_layers_[i][r][m] * cached_inputs_[i][r][n];
                    }
                    grad_w[m][n] = sum;
                }
            }

            // Accumulate GP gradient into layer.grad_weights
            if (layer.grad_weights.rows != grad_w.rows || layer.grad_weights.cols != grad_w.cols) {
                layers_[i].grad_weights = Tensor(grad_w.rows, grad_w.cols);
            }
            layers_[i].grad_weights += grad_w;

            // Propagate to previous layer: d(GP)/d(pre_{i-1}) = d(GP)/d(pre_i) @ W_i * leakyReLU'
            if (i > 0) {
                // grad_pre: (batch, out_i), W_i: (out_i, in_i)
                // grad_pre @ W_i: (batch, out_i) @ (out_i, in_i) = (batch, in_i)
                Tensor grad_input(batch, layer.weights.cols);
                for (size_t r = 0; r < batch; ++r) {
                    for (size_t c = 0; c < layer.weights.cols; ++c) {
                        double sum = 0.0;
                        for (size_t k = 0; k < out_i; ++k) {
                            sum += grad_pre[r][k] * layer.weights[k][c];
                        }
                        grad_input[r][c] = sum;
                    }
                }
                // Apply LeakyReLU' slope from cached_inputs_[i] (which is pre_i)
                const Tensor& slope_input = cached_inputs_[i];
                for (size_t r = 0; r < batch; ++r) {
                    for (size_t c = 0; c < layer.weights.cols; ++c) {
                        double x = slope_input(r, c);
                        double s = (x > 0.0) ? 1.0 : 0.01;
                        grad_input[r][c] *= s;
                    }
                }
                grad_pre = grad_input;
            }
        }
    }
}

Tensor WGANDiscriminator::backward(const Tensor& grad_output, double /* learning_rate */) {
    // Standard Layer::backward for compatibility.
    // Note: for WGAN, use backward_from() and backward_gradient_penalty() instead.
    (void)grad_output;
    return Tensor(0, 0);
}

void WGANDiscriminator::update_weights(double learning_rate) {
    for (auto& layer : layers_) {
        layer.update_weights(learning_rate);
    }
}

void WGANDiscriminator::zero_grad() {
    for (auto& layer : layers_) {
        layer.zero_grad();
    }
}

std::vector<Tensor*> WGANDiscriminator::parameters() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* p : layer.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> WGANDiscriminator::gradients() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* g : layer.gradients())
            result.push_back(g);
    return result;
}

// =====================================================================
// WGANGenerator
// - latent_dim -> hidden -> ... -> hidden -> output_dim
// =====================================================================
WGANGenerator::WGANGenerator(size_t latent_dim, size_t hidden_dim, size_t output_dim,
                             size_t num_layers)
    : latent_dim_(latent_dim), hidden_dim_(hidden_dim),
      output_dim_(output_dim), num_layers_(num_layers)
{
    layers_.reserve(num_layers + 1);
    activations_.reserve(num_layers + 1);
    cached_inputs_.reserve(num_layers + 1);

    size_t dim = latent_dim;
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.emplace_back(dim, hidden_dim);
        activations_.emplace_back(0.01);
        cached_inputs_.emplace_back();
        dim = hidden_dim;
    }
    layers_.emplace_back(dim, output_dim);
    activations_.emplace_back(0.01);
    cached_inputs_.emplace_back();
}

Tensor WGANGenerator::forward(const Tensor& input) {
    cached_inputs_[0] = input;
    Tensor x = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i].forward(x);
        if (i + 1 < cached_inputs_.size())
            cached_inputs_[i + 1] = x;
        if (i < layers_.size() - 1)
            x = activations_[i](x);
    }
    last_output_ = x;
    return x;
}

Tensor WGANGenerator::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: d_loss/d(output) coming from critic's backward
    // We backprop through all layers using the standard Dense backward,
    // accumulating into layer gradients.
    Tensor grad = grad_output;
    for (size_t i = layers_.size(); i-- > 0; ) {
        grad = layers_[i].backward(grad, 0.0);
    }
    return grad;
}

void WGANGenerator::update_weights(double learning_rate) {
    for (auto& layer : layers_)
        layer.update_weights(learning_rate);
}

void WGANGenerator::zero_grad() {
    for (auto& layer : layers_)
        layer.zero_grad();
}

std::vector<Tensor*> WGANGenerator::parameters() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* p : layer.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> WGANGenerator::gradients() {
    std::vector<Tensor*> result;
    for (auto& layer : layers_)
        for (Tensor* g : layer.gradients())
            result.push_back(g);
    return result;
}

// =====================================================================
// GradientPenaltyLoss
// =====================================================================
double GradientPenaltyLoss::compute(const Tensor& /* x */,
                                    const Tensor& /* model_output */,
                                    const Tensor& grad_wrt_x_hat,
                                    bool r1_style) {
    if (r1_style) {
        // R1 penalty: ||grad||^2
        double penalty = 0.0;
        for (size_t i = 0; i < grad_wrt_x_hat.rows; ++i) {
            double norm_sq = 0.0;
            for (size_t j = 0; j < grad_wrt_x_hat.cols; ++j) {
                double g = grad_wrt_x_hat[i][j];
                norm_sq += g * g;
            }
            penalty += norm_sq;
        }
        return penalty / static_cast<double>(grad_wrt_x_hat.rows);
    } else {
        // WGAN-GP: (||grad|| - 1)^2
        double penalty = 0.0;
        for (size_t i = 0; i < grad_wrt_x_hat.rows; ++i) {
            double norm_sq = 0.0;
            for (size_t j = 0; j < grad_wrt_x_hat.cols; ++j) {
                double g = grad_wrt_x_hat[i][j];
                norm_sq += g * g;
            }
            double norm = std::sqrt(std::max(norm_sq, 1e-12));
            double diff = norm - 1.0;
            penalty += diff * diff;
        }
        return penalty / static_cast<double>(grad_wrt_x_hat.rows);
    }
}

double GradientPenaltyLoss::compute_r1(const Tensor& /* real_samples */,
                                       const Tensor& grad_wrt_real) {
    double penalty = 0.0;
    size_t batch = grad_wrt_real.rows;
    for (size_t i = 0; i < batch; ++i) {
        double norm_sq = 0.0;
        for (size_t j = 0; j < grad_wrt_real.cols; ++j) {
            double g = grad_wrt_real[i][j];
            norm_sq += g * g;
        }
        penalty += norm_sq;
    }
    return penalty / static_cast<double>(batch);
}

// =====================================================================
// WGANTrainer
// =====================================================================
WGANTrainer::WGANTrainer(double clip_value, unsigned seed)
    : clip_value_(clip_value), rng_(seed) {}

Tensor WGANTrainer::sample_latent(size_t batch_size, size_t latent_dim) {
    Tensor z(batch_size, latent_dim);
    for (size_t i = 0; i < batch_size; ++i)
        for (size_t j = 0; j < latent_dim; ++j)
            z[i][j] = normal_(rng_);
    return z;
}

double WGANTrainer::compute_d_loss(const Tensor& d_real, const Tensor& d_fake,
                                    double gp) const {
    double n = static_cast<double>(d_real.rows);
    return (d_real.sum() - d_fake.sum()) / n + lambda_gp_ * gp;
}

double WGANTrainer::compute_g_loss(const Tensor& d_fake) const {
    return -d_fake.sum() / static_cast<double>(d_fake.rows);
}

void WGANTrainer::clip_weights(WGANDiscriminator& critic, double clip_val) {
    for (size_t l = 0; l < critic.num_layers(); ++l) {
        Dense& layer = critic.layer(l);
        for (size_t i = 0; i < layer.weights.rows; ++i) {
            for (size_t j = 0; j < layer.weights.cols; ++j) {
                double w = layer.weights[i][j];
                layer.weights[i][j] = std::max(-clip_val, std::min(clip_val, w));
            }
        }
        for (size_t j = 0; j < layer.bias.cols; ++j) {
            double b = layer.bias[0][j];
            layer.bias[0][j] = std::max(-clip_val, std::min(clip_val, b));
        }
    }
}

void WGANTrainer::train_step(WGANDiscriminator& critic,
                             WGANGenerator& generator,
                             const Tensor& real_batch,
                             size_t latent_dim,
                             double lambda_gp,
                             int /* n_critic */) {
    lambda_gp_ = lambda_gp;
    const size_t batch = real_batch.rows;
    const double n = static_cast<double>(batch);

    // ---- Generate fake samples ----
    Tensor z = sample_latent(batch, latent_dim);
    Tensor fake_batch = generator.forward(z); // G forward

    // ---- D forward on real ----
    critic.reset_cached_inputs();
    Tensor d_real = critic.forward(real_batch);

    // ---- D forward on fake ----
    critic.reset_cached_inputs();
    Tensor d_fake = critic.forward(fake_batch);

    // ---- Gradient penalty ----
    double gp = critic.gradient_penalty(real_batch, fake_batch, lambda_gp);
    (void)gp;

    // ---- D update: accumulate gradients from real, fake, and GP ----
    critic.zero_grad();

    // (a) Real sample gradient: d_loss/d(D(real)) = +1/N
    {
        critic.reset_cached_inputs();
        critic.forward(real_batch);
        Tensor grad_real(batch, 1);
        for (size_t i = 0; i < batch; ++i) grad_real[i][0] = 1.0 / n;
        critic.backward_from(grad_real);
    }

    // (b) Fake sample gradient: d_loss/d(D(fake)) = -1/N
    {
        critic.reset_cached_inputs();
        critic.forward(fake_batch);
        Tensor grad_fake(batch, 1);
        for (size_t i = 0; i < batch; ++i) grad_fake[i][0] = -1.0 / n;
        critic.backward_from(grad_fake);
    }

    // (c) GP gradient: accumulate via backward_gradient_penalty()
    {
        // gradient_penalty was already called above and cached x_hat forward.
        // Now backpropagate the GP through critic.
        critic.backward_gradient_penalty();
    }

    // Clip weights if requested (basic WGAN; not needed with GP)
    if (clip_value_ > 0.0)
        clip_weights(critic, clip_value_);

    // ---- G update ----
    // g_loss = -mean(D(fake)) = -1/N * sum_i D(fake)_i
    // d_g_loss/d(D(fake)_i) = -1/N
    // Backprop through critic: d_g_loss/d(fake) = -1/N * d(D(fake))/d(fake)
    // Then through generator: d_g_loss/d(G_weights) = d_g_loss/d(fake) * d(fake)/d(G_weights)

    generator.zero_grad();
    {
        // Re-forward fake through critic to get proper last_input chain
        critic.reset_cached_inputs();
        d_fake = critic.forward(fake_batch);

        // d_loss/d(D(fake)) = -1/N for each element
        Tensor grad_g(batch, 1);
        for (size_t i = 0; i < batch; ++i) grad_g[i][0] = -1.0 / n;

        // Backprop through critic to get grad w.r.t. fake (critic's input)
        // This accumulates into critic gradients as a side effect.
        // We reset critic grads afterwards.
        Tensor grad = grad_g;
        for (size_t i = critic.num_layers(); i-- > 0; ) {
            grad = critic.layer(i).backward(grad, 0.0);
        }
        // grad = d(D(fake))/d(fake) * (-1/N) = upstream gradient for generator

        // Now backprop through generator using grad as upstream gradient
        generator.backward(grad, 0.0);

        // Reset critic gradients accumulated during G backward (not needed for G update)
        critic.zero_grad();
    }
}

// =====================================================================
// WGANModel
// =====================================================================
WGANModel::WGANModel(size_t latent_dim, size_t data_dim,
                     size_t hidden_dim_d, size_t hidden_dim_g,
                     size_t num_layers_d, size_t num_layers_g,
                     unsigned seed)
    : critic_(std::make_unique<WGANDiscriminator>(data_dim, hidden_dim_d, num_layers_d)),
      generator_(std::make_unique<WGANGenerator>(latent_dim, hidden_dim_g, data_dim, num_layers_g)),
      trainer_(0.0, seed)
{}

std::tuple<double, double, double> WGANModel::train_step(
    const Tensor& real_batch,
    double lambda_gp,
    int n_critic) {

    size_t latent_dim = generator_->latent_dim();
    size_t batch_size = real_batch.rows;
    const double n = static_cast<double>(batch_size);

    // For n_critic > 1, do multiple D updates per G update.
    // In this implementation, we do 1 D update per G update.
    // The n_critic parameter is accepted for API compatibility.
    (void)n_critic;

    // ---- G forward ----
    Tensor z = trainer_.sample_latent(batch_size, latent_dim);
    Tensor fake_batch = generator_->forward(z);

    // ---- D forward on real ----
    critic_->reset_cached_inputs();
    Tensor d_real = critic_->forward(real_batch);

    // ---- D forward on fake ----
    critic_->reset_cached_inputs();
    Tensor d_fake = critic_->forward(fake_batch);

    // ---- Gradient penalty ----
    double gp = critic_->gradient_penalty(real_batch, fake_batch, lambda_gp);

    // ---- D loss value ----
    double d_loss = (d_real.sum() - d_fake.sum()) / n + lambda_gp * gp;
    double g_loss = -d_fake.sum() / n;

    // ---- D update ----
    critic_->zero_grad();

    // Real gradient
    {
        critic_->reset_cached_inputs();
        critic_->forward(real_batch);
        Tensor grad_real(batch_size, 1);
        for (size_t i = 0; i < batch_size; ++i) grad_real[i][0] = 1.0 / n;
        critic_->backward_from(grad_real);
    }

    // Fake gradient
    {
        critic_->reset_cached_inputs();
        critic_->forward(fake_batch);
        Tensor grad_fake(batch_size, 1);
        for (size_t i = 0; i < batch_size; ++i) grad_fake[i][0] = -1.0 / n;
        critic_->backward_from(grad_fake);
    }

    // GP gradient
    {
        (void)critic_->gradient_penalty(real_batch, fake_batch, lambda_gp);
        critic_->backward_gradient_penalty();
    }

    // ---- G update ----
    generator_->zero_grad();
    {
        // Re-forward fake through critic
        critic_->reset_cached_inputs();
        d_fake = critic_->forward(fake_batch);

        Tensor grad_g(batch_size, 1);
        for (size_t i = 0; i < batch_size; ++i) grad_g[i][0] = -1.0 / n;

        // Backprop through critic to get grad w.r.t. fake
        // (accumulates into critic gradients as side effect)
        Tensor grad = grad_g;
        for (size_t i = critic_->num_layers(); i-- > 0; ) {
            grad = critic_->layer(i).backward(grad, 0.0);
        }
        // grad = d(D(fake))/d(fake) * (-1/N)
        generator_->backward(grad, 0.0);

        // Reset critic gradients accumulated during G backward
        critic_->zero_grad();
    }

    return {d_loss, g_loss, gp};
}

Tensor WGANModel::generate(const Tensor& z) const {
    return generator_->forward(z);
}

Tensor WGANModel::generate(size_t num_samples) const {
    size_t latent_dim = generator_->latent_dim();
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    Tensor z(num_samples, latent_dim);
    for (size_t i = 0; i < num_samples; ++i)
        for (size_t j = 0; j < latent_dim; ++j)
            z[i][j] = dist(gen);
    return generator_->forward(z);
}
