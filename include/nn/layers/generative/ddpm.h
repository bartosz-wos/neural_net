#ifndef NN_GENERATIVE_DDPM_H
#define NN_GENERATIVE_DDPM_H

#include "../../core/tensor.h"
#include "../../core/layer.h"
#include "../../activations/activations.h"
#include "../utility/conv1d.h"
#include "../utility/conv1d_transpose.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

// =============================================================================
// DDPM — Denoising Diffusion Probabilistic Models
//
// Forward process (noising):
//   q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
//   x_t = sqrt(alphabar_t) * x_0 + sqrt(1-alphabar_t) * epsilon
//
// Reverse process (denoising):
//   p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
//   We predict epsilon_theta and use the analytic DDPM mean formula.
//
// =============================================================================

// =============================================================================
// NoiseScheduler — manages the forward diffusion noise schedule (linear schedule)
// =============================================================================
class NoiseScheduler {
public:
    // Default constructor (for use as a class member)
    NoiseScheduler();

    NoiseScheduler(int T, float beta_start, float beta_end);

    void initialize(float beta_start, float beta_end, int T);

    // Computes the full linear noise schedule
    void compute_schedule();

    // ----- Accessors -----
    float sqrt_alphas_cumprod(int t) const;
    float sqrt_one_minus_alphas_cumprod(int t) const;
    float sqrt_recip_alphas(int t) const;
    float extract(std::vector<float>& vec, int t) const;
    float posterior_variance(int t) const;

    // Noise schedule parameters
    int T() const { return T_; }
    float beta_start() const { return beta_start_; }
    float beta_end() const { return beta_end_; }

    // Forward process: apply t steps of noise to x0
    // x_t = sqrt(alphabar_t) * x0 + sqrt(1-alphabar_t) * noise
    Tensor q_sample(const Tensor& x0, int t, const Tensor& noise) const;

    // Precomputed arrays
    std::vector<float> betas;          // size T
    std::vector<float> alphas;          // size T
    std::vector<float> alphas_cumprod; // size T
    std::vector<float> alphas_cumprod_prev; // size T+1

private:
    int T_;
    float beta_start_;
    float beta_end_;
};

// =============================================================================
// TimeEmbedding — sinusoidal positional embeddings for timestep t
// =============================================================================
class TimeEmbedding {
public:
    explicit TimeEmbedding(int hidden_dim);
    Tensor forward(int t) const;

private:
    int hidden_dim_;
};

// =============================================================================
// ResBlock — 1D residual block with time embedding injection
// =============================================================================
class DDPMResBlock : public Layer {
public:
    DDPMResBlock(int in_channels, int out_channels, int time_emb_dim, int seq_len);
    ~DDPMResBlock();

    // Layer interface (required override — discards time_emb)
    Tensor forward(const Tensor& x) override;
    // Two-arg forward with time embedding (used by DenoisingUNet)
    Tensor forward(const Tensor& x, const Tensor& time_emb);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return conv1_.weights; }
    Tensor get_gradients() const override { return conv1_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "DDPMResBlock"; }
    int in_ch()  const { return in_channels_; }
    int out_ch() const { return out_channels_; }
    int seq_len() const { return seq_len_; }

private:
    int in_channels_;
    int out_channels_;
    int seq_len_;
    static bool initialized_;  // set true after first forward pass (conv2 zero-init)

    Conv1D conv1_;
    Conv1D conv2_;
    Dense  time_mlp_;
    GELU   act_;
    Tensor last_x_;
    Tensor last_time_emb_;
    Tensor last_h_;
};

// =============================================================================
// SkipTransform — 1x1 Conv1D to project skip connection channels to mid_ch
// Used to match encoder skip channels to the bottleneck channel count before concat
// =============================================================================
class SkipTransform : public Layer {
public:
    SkipTransform(int in_channels, int out_channels, int seq_len);
    ~SkipTransform();

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return conv_.weights; }
    Tensor get_gradients() const override { return conv_.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "SkipTransform"; }
    int in_ch()  const { return conv_.in_channels; }
    int out_ch() const { return conv_.out_channels; }

private:
    Conv1D conv_;
    Tensor last_x_;
};

// =============================================================================
// DenoisingUNet — 1D UNet for denoising time-series / sequence data
//
// Architecture:
//   - TimeEmbedding: t -> hidden_dim
//   - Encoder: channels with ResBlocks and downsample convs
//   - Middle: two ResBlocks (no skip)
//   - Decoder: skip_transforms (1x1 conv to mid_ch) + concat + Upsample1D + ResBlocks
//   - Output: Conv1D -> in_channels
//
// Input shape: (batch, in_features) where in_features = in_channels * seq_len
// Output shape: (batch, in_features) — same shape as input (predicts noise epsilon)
// =============================================================================
class DenoisingUNet {
public:
    DenoisingUNet(int in_channels, const std::vector<int>& channels,
                  int time_emb_dim, int seq_len);
    ~DenoisingUNet();

    // Forward with integer timestep
    Tensor forward(const Tensor& x, int t);

    // Forward with pre-computed time embedding
    Tensor forward(const Tensor& x, const Tensor& time_emb);

    // Backward pass
    Tensor backward(const Tensor& grad_output, double learning_rate);

    void update_weights(double learning_rate);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    // Public time embedding access
    Tensor time_embedding(int t) const { return time_emb_.forward(t); }

    int input_channels() const { return in_channels_; }

private:
    int in_channels_;
    std::vector<int> channels_;
    int time_emb_dim_;
    int seq_len_;

    TimeEmbedding time_emb_;

    // Encoder path
    std::vector<DDPMResBlock*>   down_blocks_;
    std::vector<Conv1D*>     down_sample_;  // stride-2 conv for downsampling
    std::vector<int>         encoder_channels_;

    // Middle
    DDPMResBlock* middle_block1_;
    DDPMResBlock* middle_block2_;

    // Decoder path
    std::vector<DDPMResBlock*>     up_blocks_;
    std::vector<Upsample1D*>   up_sample_;    // 2x nearest-neighbor upsampling
    std::vector<SkipTransform*> skip_transforms_; // 1x1 convs on skip connections
    std::vector<int>           decoder_channels_;

    // Output conv: last encoder channel -> in_channels
    Conv1D conv_out_;

    Tensor last_x_;
    Tensor last_time_emb_;
    std::vector<Tensor> skip_connections_;
    int enc_levels_;
};

// =============================================================================
// DDPMModel — complete DDPM model combining scheduler + UNet
//
// DDPM loss: L = E_{t, epsilon}[ ||epsilon - epsilon_theta(x_t, t)||^2 ]
// where x_t = sqrt(alphabar_t) * x0 + sqrt(1-alphabar_t) * epsilon
// =============================================================================
class DDPMModel : public Layer {
public:
    DDPMModel(int in_channels, const std::vector<int>& channels,
              int time_emb_dim, int seq_len, int diffusion_steps,
              float beta_start = 1e-4f, float beta_end = 0.02f);
    ~DDPMModel();

    // Build the model (called automatically by constructor)
    void setup();

    // ----- DDPM forward pass (training) -----
    // Samples random timestep t, generates noise, returns MSE loss
    Tensor training_forward(const Tensor& x0);

    // Returns the noise-prediction loss for a given x0
    float loss(const Tensor& x0);

    // Single reverse denoising step: x_{t-1} = p_theta(x_t, t)
    Tensor denoise(const Tensor& x_t, int t);

    // Full reverse diffusion sampling loop
    // Returns generated sample(s) of shape (num_samples, in_channels * seq_len)
    Tensor sample(int num_samples);

    // ----- Layer interface -----
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "DDPMModel"; }

    // Accessors
    DenoisingUNet& unet() { return *unet_; }
    const NoiseScheduler& scheduler() const { return scheduler_; }

private:
    int in_channels_;
    std::vector<int> channels_;
    int time_emb_dim_;
    int seq_len_;
    int diffusion_steps_;
    float beta_start_;
    float beta_end_;

    std::unique_ptr<DenoisingUNet> unet_;
    NoiseScheduler scheduler_;

    std::mt19937 rng_;
    std::uniform_int_distribution<int> dist_t_;
    std::normal_distribution<float> normal_;

    Tensor last_x0_;
    Tensor last_xt_;
    Tensor last_noise_;
    int    last_t_;
    float  last_loss_;
};

#endif // NN_GENERATIVE_DDPM_H