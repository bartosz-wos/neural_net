#ifndef NN_NORMALIZATION_ADALN_ZERO_H
#define NN_NORMALIZATION_ADALN_ZERO_H

#include "../../core/layer.h"
#include <vector>

// ============================================================================
// AdaLN-Zero — DiT-style adaptive layer norm modulation with zero-init gate.
//   Peebles & Xie 2023 "Scalable Diffusion Models with Transformers" (DiT)
//   https://arxiv.org/abs/2212.09748
//   Esser et al. 2024 "Scaling Rectified Flow Transformers for High-Resolution
//   Image Synthesis" (SD3) https://arxiv.org/abs/2403.06304
//
// A single AdaLNModulation produces (shift, scale, gate) ∈ R^{B×d_model} from
// a conditioning vector via a 2-layer Dense MLP with SiLU; proj2 is initialized
// to zero so the block is identity at the start of training.
//
// AdaLNZeroBlock consumes these tensors to apply, for each token:
//     y = x + gate ⊙ ( (1 + scale) * LayerNorm(x) + shift )
//
// This single class is enough to verify the math and to integrate into Model.
// ============================================================================

class AdaLNModulation {
public:
    AdaLNModulation(size_t cond_dim, size_t d_model, size_t hidden_mult = 4);

    // cond: (B, cond_dim). Returns a vector of 3 tensors (shift, scale, gate),
    // each of shape (B, d_model).
    std::vector<Tensor> forward(const Tensor& cond);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void update_weights(double lr);
    void zero_grad();

    Dense&  proj1() { return proj1_; }
    Dense&  proj2() { return proj2_; }
    const Tensor& last_pre_silu() const { return last_pre_silu_; }
    const Tensor& last_hidden() const { return last_hidden_; }

    // Gradient aliases — the modulation's gradient buffers are the Dense's
    // own gradient buffers, so any write here flows directly into optimizer
    // updates. We provide these aliases for clarity at the call site.
    Tensor& grad_proj1_w() { return proj1_.grad_weights; }
    Tensor& grad_proj1_b() { return proj1_.grad_bias; }
    Tensor& grad_proj2_w() { return proj2_.grad_weights; }
    Tensor& grad_proj2_b() { return proj2_.grad_bias; }

private:
    Dense proj1_;   // (hidden, cond_dim)
    Dense proj2_;   // (3*d_model, hidden)  — zero-init
    size_t cond_dim_;
    size_t d_model_;
    size_t hidden_;

    Tensor last_pre_silu_;   // (B, hidden), pre-SiLU value (needed for SiLU backward)
    Tensor last_hidden_;     // (B, hidden), post-SiLU value (input to proj2)
};

class AdaLNZeroBlock : public Layer {
public:
    AdaLNZeroBlock(size_t cond_dim, size_t d_model,
                   size_t hidden_mult = 4, double eps = 1e-6);

    Tensor forward(const Tensor& input, const Tensor& cond);
    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights()  const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "AdaLNZeroBlock"; }

    AdaLNModulation& modulation() { return adaln_; }

private:
    size_t cond_dim_;
    size_t d_model_;
    double eps_;

    AdaLNModulation adaln_;
    Tensor gamma_;     // (1, d_model), init to 1
    Tensor beta_;      // (1, d_model), init to 0
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // Caches
    Tensor last_x_;
    Tensor last_cond_;
    Tensor last_mu_;
    Tensor last_rstd_;
    Tensor last_normed_;      // raw (x - μ) * rstd  (BEFORE gamma/beta)
    Tensor last_h_;           // post-affine (gamma=1, beta=0 → == last_normed_)
    Tensor last_shift_;
    Tensor last_scale_;
    Tensor last_gate_;
    Tensor last_y_mod_;
    Tensor last_y_;
};

#endif
