#ifndef MIXTURE_OF_SOFTMAXES_H
#define MIXTURE_OF_SOFTMAXES_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Mixture of Softmaxes (MoS) Attention
//   Yang, Chen, Liu, Li, Wang, Yi, Wong, Xia — ICLR 2018
//   "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model"
//   https://arxiv.org/abs/1711.03953
//
// The "softmax bottleneck" (Yang et al. 2018): a single softmax
//   P[i,j] = exp(z[i,j]) / Σ_{i'} exp(z[i',j])
// is at most rank-1 in log-space. The downstream value projection `W_v`
// therefore expresses at most a rank-1 transformation of the softmax output,
// which empirically throttles language-model perplexity.
//
// Fix: replace the single softmax with a learned MIXTURE of K softmaxes.
// Each branch has its own value projection (so the per-branch projection
// lives in a richer matrix space), and a separate gating network
//   Pi[j] = softmax_k(W_g · Q[j] + b_g)
// mixes the branches. The combined output
//   Y[j] = Σ_k Pi[j, k] · softmax_i((Q[j]·K[i] + B_z[k,d])/√d) · V_k[i]
// is unconstrained in log-space rank. K=1 recovers standard MHA exactly.
//
// ============================================================================
// Conventions
//   * Input / output: (N, d_model). N tokens, d_model feature dim.
//   * H = num_heads, K = num mixture branches.
//   * head_dim = d_model / H. d_model must be divisible by H.
//   * W_q, W_k, W_o: shared Dense (d_model, d_model) — standard MHA.
//   * W_v : (K, d_model, d_model) per-branch per-head value projection.
//   * B_v : (K, d_model) per-branch value bias.
//   * B_z : (K, d_model) per-branch additive logit bias (Yang et al. Eq. 7).
//   * W_g : (K, d_model) gating weights (per head). One bias B_g : (K, 1)
//           shared across heads for simplicity (equivalent to K biases per
//           head — the (K, d_model) per-head gate decomposes naturally).
//   * All forward caches stored as raw Tensors (not via Dense) because the
//     per-branch softmax + per-head gating chain is hand-derived and needs
//     manual control.
// ============================================================================

class MixtureOfSoftmaxesAttention : public Layer {
public:
    MixtureOfSoftmaxesAttention(size_t d_model, size_t num_heads = 1, size_t K = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "MixtureOfSoftmaxesAttention"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t num_branches() const { return K_; }
    double inv_temp() const { return inv_temp_; }

    // Forward caches (public so tests can inspect / perturb)
    Dense W_q, W_k, W_o;            // shared (d_model, d_model)
    Tensor W_v;                     // (K, d_model, d_model)
    Tensor B_v;                     // (K, d_model)
    Tensor B_z;                     // (K, d_model)
    Tensor W_g;                     // (K, d_model) — gating weights (per-head slice of d_model)
    Tensor B_g;                     // (K, 1) — gating bias

    Tensor grad_W_q, grad_b_q;      // (d_model, d_model), (1, d_model)
    Tensor grad_W_k, grad_b_k;
    Tensor grad_W_o, grad_b_o;
    Tensor grad_W_v, grad_B_v, grad_B_z, grad_W_g, grad_B_g;

    const Tensor& last_P() const { return last_P_; }    // (H*K, N*N)
    const Tensor& last_Pi() const { return last_Pi_; }  // (H, N*K)
    const Tensor& last_input() const { return last_input_; }

private:
    // Order matches constructor init list to silence -Wreorder.
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double inv_temp_;
    size_t K_;
    size_t size_t_K_;

    Tensor last_input_;             // (N, d_model)
    Tensor last_Q_, last_K_;        // (N, d_model)
    Tensor last_V_k_;               // (K, N, d_model) per-branch V projection applied
    Tensor last_Z_;                 // (H*K, N*N)    per-(head, branch) raw logits
    Tensor last_P_;                 // (H*K, N*N)   per-(head, branch) attention maps (rows are j*N+i)
    Tensor last_Pi_;                // (H, N*K)     per-head gating (rows are j*K+k)
    Tensor last_g_logits_;          // (H, N, K)    pre-gating-softmax logits (= Q[h] @ W_g^T + B_g)
    Tensor last_O_branch_;          // (H, K*N*dh)  per-(head, branch) pre-W_o output
    Tensor last_O_;                 // (H, N, head_dim) per-head pre-W_o output (= Σ_k Pi O_branch)
    Tensor last_Y_;                 // (N, d_model)  post-W_o output (kept for grad)
    Tensor last_flat_;              // (N, d_model)  W_o input (kept for grad_W_o)
};

#endif // MIXTURE_OF_SOFTMAXES_H