#ifndef CAPSNET_H
#define CAPSNET_H

#include "../../core/layer.h"
#include <vector>
#include <random>

// ============================================================================
// CapsuleLayer — Sabour, Frosst, Hinton 2017
//   "Dynamic Routing Between Capsules"
//   (https://arxiv.org/abs/1710.09829)
//
// Innovation: vector-valued neurons ("capsules") whose activation encodes both
// the *probability* of an entity being present (via the vector's LENGTH) and
// the *instantiation parameters* of that entity (via the vector's DIRECTION).
// Routing is the iterative procedure that decides which lower-level capsule
// feeds into which higher-level capsule, by REACHING AGREEMENT via the inner
// product of the prediction with the current output vector.
//
// Math:
//
//   Input:    u ∈ R^{B × I × D_in}   (B batches, I input capsules, D_in dims each)
//   Output:   v ∈ R^{B × J × D}      (J output capsules, D dims each)
//
//   Per output capsule j and input capsule i, a learned transformation matrix
//   W[j] ∈ R^{D × D_in} produces a "prediction vector":
//
//       û[b, j, i, k] = sum_{d_in} W[j][d_in, k] · u[b, i, d_in]
//
//   The routing loop (R iterations) finds coupling coefficients c[b, i, j]
//   by softmax over the output-capsule axis j:
//
//       b_1[b, i, j] = 0
//       for r = 1..R:
//         c_r[b, i, j] = softmax_j(b_r[b, i, j])
//         s_r[b, j, k] = sum_i c_r[b, i, j] · û[b, j, i, k]
//         v_r[b, j, k] = squash(s_r[b, j, k])
//         if r < R:
//           b_{r+1}[b, i, j] = b_r[b, i, j] + sum_k û[b, j, i, k] · v_r[b, j, k]
//
//   where squash(x) = x · ||x||² / (1 + ||x||²) / ||x|| = x · ||x|| / (1 + ||x||²).
//   Squash maps a vector to one whose LENGTH lies in [0, 1), preserving direction.
//
//   The final output is v_R, returned as (B, J*D).
//
// Backward (BPTT-through-routing):
//   The chain runs over (1) the squash Jacobian per output capsule, (2) the
//   softmax-over-j Jacobian, (3) the per-iteration agreement carrier, and
//   (4) the linear transform û = u W[j]^T. Detailed recipe in capsnet.cpp.
//
// Public API:
//   CapsuleLayer(num_input_capsules, input_capsule_dim, num_capsules, dim_capsule, num_routing=3)
//       forward(input: (B, I*D_in)) -> output: (B, J*D)
//       backward(grad_output: (B, J*D), lr) -> grad_input: (B, I*D_in)
//
//   All other Layer interface methods (parameters/gradients/zero_grad/update_weights)
//   are standard. parameters()/gradients() return J tensors of shape (D_in, D).
// ============================================================================

class CapsuleLayer : public Layer {
public:
    CapsuleLayer(size_t num_input_capsules, size_t input_capsule_dim,
                 size_t num_capsules, size_t dim_capsule,
                 size_t num_routing = 3);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "CapsuleLayer"; }

    size_t num_input_capsules() const { return num_input_capsules_; }
    size_t input_capsule_dim()  const { return input_capsule_dim_; }
    size_t num_capsules()       const { return num_capsules_; }
    size_t dim_capsule()        const { return dim_capsule_; }
    size_t num_routing()        const { return num_routing_; }

private:
    size_t num_input_capsules_;
    size_t input_capsule_dim_;
    size_t num_capsules_;
    size_t dim_capsule_;
    size_t num_routing_;
    double epsilon_ = 1e-8;

    // One transformation matrix per output capsule. W_[j] has shape (D_in, D):
    //   û[b, j, i, k] = sum_{d_in} W_[j][d_in, k] · u[b, i, d_in]
    std::vector<Tensor> W_;
    std::vector<Tensor> grad_W_;

    // ----- Caches for BPTT -----
    Tensor last_input_;       // (B, I*D_in) — for input-grad
    Tensor last_u_hat_;       // (B, J*I*D) — predictions per output capsule per input capsule
    Tensor last_c_;           // (B, I*J) — coupling coefficients from last routing iter
    // Per-iter caches for the routing loop.
    std::vector<Tensor> iter_s_;   // pre-squash, each (B, J*D)
    std::vector<Tensor> iter_v_;   // post-squash, each (B, J*D)
    std::vector<Tensor> iter_c_;   // couplings, each (B, I*J)
};

// ============================================================================
// CapsNet — full MNIST-style capsule network (kept for API compatibility).
// The forward path is the original sketch: primary capsules (flatten → Dense
// → squash) → digit capsules (dynamic routing) → length-of-capsule
// classifier. Backward through the routing + reconstruction decoder is NOT
// implemented (still returns Tensor(1, 1)) — only the CapsuleLayer itself
// has full BPTT. This class is preserved so existing code that includes
// capsnet.h still compiles; for new development use CapsuleLayer directly.
// ============================================================================
class CapsNet : public Layer {
public:
    CapsNet(size_t input_channels, size_t H, size_t W,
            size_t num_classes, size_t dim_capsule = 16,
            size_t primary_dim = 8, size_t primary_channels = 32 * 6 * 6,
            size_t num_routing = 3);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    double reconstruction_loss(const Tensor& input,
                               const Tensor& digit_capsules,
                               size_t correct_label);

private:
    Dense primary_caps_fc_;
    CapsuleLayer digit_caps_;
    Dense fc1_, fc2_, fc3_;
    Tensor last_input_;
    Tensor last_capsule_output_;
    size_t dim_capsule_;
};

#endif