#ifndef CONSISTENCY_H
#define CONSISTENCY_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include <vector>
#include <memory>
#include <random>

// =====================================================================
// Forward declaration
// =====================================================================
class UNetDenoiser;

// =====================================================================
// ConvTranspose2D — transposed conv (fractional stride 2) for upsampling
// (defined first so DecBlock can contain it by value)
// =====================================================================
class ConvTranspose2D : public Layer {
public:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int output_pad_h, output_pad_w;
    int H, W; // input spatial dimensions

    Tensor weights;
    Tensor bias;
    Tensor grad_weights;
    Tensor grad_bias;
    Tensor last_input;

    ConvTranspose2D() = default;
    ConvTranspose2D(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
                   int stride_h, int stride_w, int pad_h, int pad_w,
                   int output_pad_h = 1, int output_pad_w = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return weights; }
    Tensor get_gradients() const override { return grad_weights; }
    std::string name() const override { return "ConvTranspose2D"; }
};

// =====================================================================
// TimeMLP — two-layer MLP for time conditioning at each resolution level
// =====================================================================
struct TimeMLP {
    Dense* fc1 = nullptr;
    Dense* fc2 = nullptr;
    TimeMLP() = default;
    TimeMLP(int in_dim, int hidden_dim, int out_dim);
};

// =====================================================================
// EncBlock — one encoder block: two convs + downsample + time MLP
// =====================================================================
struct EncBlock {
    int out_channels = 0;
    Conv2D conv1;
    Conv2D conv2;
    Conv2D downsample;
};

// =====================================================================
// DecBlock — one decoder block: upsample + two convs
// =====================================================================
struct DecBlock {
    int out_channels = 0;
    ConvTranspose2D upsample;
    Conv2D conv1;
    Conv2D conv2;
};

// =====================================================================
// ConsistencyTimeEmbedding — sinusoidal positional-style time embedding
// =====================================================================
class ConsistencyTimeEmbedding {
public:
    ConsistencyTimeEmbedding(int dim, int max_len = 1024);

    Tensor embed(double t) const;

private:
    int dim_;
    int half_;
    int max_len_;
    std::vector<double> freq_;
};

// =====================================================================
// ConsistencyStudent — U-Net-style student with time + class conditioning
// =====================================================================
class ConsistencyStudent : public Layer {
public:
    ConsistencyStudent(int channels, int time_dim, int num_classes,
                        int depth, int base_ch, int H, int W);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::string name() const override { return "ConsistencyStudent"; }

    void set_condition(double t, int class_label) {
        last_t_ = t;
        last_class_label_ = class_label;
    }

private:
    Tensor silu(const Tensor& x);
    Tensor add_time(const Tensor& feat, const Tensor& te, int out_ch);
    Tensor reshape_4d(const Tensor& flat, int N, int C, int H, int W) const;
    Tensor flatten_4d(const Tensor& tensor, int N, int C, int H, int W) const;
    static Tensor upsample2x(const Tensor& x, int N, int C, int H, int W);

    int channels_;
    int time_dim_;
    int num_classes_;
    int depth_;
    int base_ch_;
    int H_;
    int W_;

    static constexpr int class_embed_dim_ = 64;

    ConsistencyTimeEmbedding time_embed_;
    std::vector<Tensor> class_embed_;
    std::unique_ptr<Dense> class_proj_;

    double last_t_ = 0.0;
    int last_class_label_ = -1;
    Tensor last_input_;

    std::vector<EncBlock> enc_blocks_;
    std::vector<DecBlock> dec_blocks_;
    std::vector<Conv2D> skip_transforms_;
    std::vector<TimeMLP> time_mlps_;
    std::vector<Tensor> grad_bufs_;

    Conv2D mid_conv1_;
    Conv2D mid_conv2_;
    Conv2D final_conv_;
};

// =====================================================================
// UNetDenoiser — teacher denoiser: wraps a UNet with time conditioning
// =====================================================================
class UNetDenoiser {
public:
    UNetDenoiser();
    void set_condition(double t, int class_label);
    Tensor forward(const Tensor& x) const;

private:
    double last_t_;
    int last_class_label_;
    ConsistencyTimeEmbedding time_embed_;
};

// =====================================================================
// ConsistencyModel — teacher-student distillation for consistency training
// =====================================================================
class ConsistencyModel : public Layer {
public:
    ConsistencyModel(UNetDenoiser* teacher, ConsistencyStudent* student,
                     int T, int distillage_steps,
                     int min_step = 0, int max_step = -1);

    // Training: single distillation step, returns L2 consistency loss
    double distill(const Tensor& x_t, int t);

    // Sampling: single-step and multi-step consistency sampling
    Tensor sample(const Tensor& x_t, int t) const;
    Tensor sample_multistep(const Tensor& x_start, int num_steps);
    Tensor consistency_correction(const Tensor& x, int t, int t_next) const;
    Tensor sample_with_cfg(const Tensor& x_t, int t, int class_label,
                           double guidance_scale);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override { return Tensor(0, 0); }
    Tensor get_gradients() const override { return Tensor(0, 0); }
    std::string name() const override { return "ConsistencyModel"; }

    void set_condition(double t, int class_label) {
        last_t_ = t;
        last_class_label_ = class_label;
    }

private:
    void build_distillation_schedule();

    UNetDenoiser* teacher_ = nullptr;
    ConsistencyStudent* student_ = nullptr;

    int T_;
    int distillage_steps_;
    int min_step_;
    int max_step_;
    int step_size_;
    int current_distill_step_ = 0;
    std::vector<int> distillation_schedule_;

    double last_t_ = 0.0;
    int last_class_label_ = -1;

    Tensor teacher_pred_;
    Tensor student_pred_;
    Tensor loss_buf_;
    Tensor sample_buf_;
    Tensor grad_buf_;

    std::mt19937 rng_;
};

#endif // CONSISTENCY_H
