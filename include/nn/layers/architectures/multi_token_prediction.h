#ifndef MULTI_TOKEN_PREDICTION_H
#define MULTI_TOKEN_PREDICTION_H

#include "../../core/layer.h"
#include <vector>
#include <memory>

// ============================================================================
// Multi-Token Prediction Head
//   Gloeckle, Du, Gunter, Mao, Neubig, Tsvetkov (Meta, 2024)
//   "Better & Faster Large Language Models via Multi-token Prediction"
//   https://arxiv.org/abs/2404.19737
//
// Replaces the single next-token head with n_future independent Dense-style
// heads that share the trunk representation. Per head k in {0..n_future-1}:
//   y_hat_k[t, v] = h[t, :] @ W_k[v, :]^T + b_k[v]      in R^((N, vocab))
//   p_k[t, v]     = softmax(y_hat_k[t, :])[v]
// Loss (Eq. 4):
//   L_k = - (1 / N_valid_k) * sum_t mask_k[t] * log p_k[t, y^(k)[t]]
//   L   = sum_k L_k                  (L > 0 — standard CE)
//
// where mask_k[t] = 1 iff t + k + 1 < N (we have a target at position t + k + 1).
//
// Backward (paper §2.2):
//   d_logits_k[t, v] = (1 / N_valid_k) * mask_k[t] * (p_k[t, v] - 1{v=y_k[t]})
//   dW_k[v, c]       = sum_t d_logits_k[t, v] * h[t, c]
//   db_k[v]          = sum_t d_logits_k[t, v]
//   d_trunk[t, c]    = sum_k sum_v d_logits_k[t, v] * W_k[v, c]   <- SUM
//
// The SUM aggregation on the last line is the defining design: the trunk
// receives the gradient from ALL heads, not the gradient of any single one.
// A hand-derived test asserts this directly (Test 4b) and is the signature
// that distinguishes the multi-head design from a "single head picked first"
// mistake.
//
// Conventions (match ngpt.h / stick_breaking.h):
//   * Targets are an integer-valued Tensor of shape (N, n_future);
//     targets(t, k) is the class index for head k's prediction at token t.
//   * No explicit batch dimension.
//   * parameters()/gradients() return shape-matched pairs in the order:
//        W_0, b_0, W_1, b_1, ..., W_{n_future-1}, b_{n_future-1}
//   * The forwarding API takes (h: (N, d_model), targets: (N, n_future)) and
//     returns the scalar summed CE loss; backward returns d_trunk (N, d_model).
//
// Deliberate deviations from the paper:
//   1. Loss is NOT decoupled into per-head training-stream heads (the paper
//      trains each head with a fully-shared trunk — we follow that exactly).
//   2. Per-head bias b_k IS kept (paper ablations show ~0 effect; the test
//      scaffolding here relies on it for the hand-derived Test 4a).
//   3. Inference uses head[n_future - 1]'s logits (paper §4.1 standard).
//   4. No tied input/output embeddings (paper ablations show ~0 benefit at
//      the test vocab sizes; keeps the impl simple).
// ============================================================================

// ----------------------------------------------------------------------------
// MultiTokenPredictionHead — the head module itself.
//
// NOT a Layer: the forward API differs (takes targets, returns scalar loss,
// not "apply to single tensor"); the model's integration shows the canonical
// pattern (a wrapper Model assembles input proj + head + classifier).
// ----------------------------------------------------------------------------
class MultiTokenPredictionHead {
public:
    MultiTokenPredictionHead(size_t d_model, size_t vocab_size, size_t n_future);

    // Forward. h is (N, d_model); targets is (N, n_future) of integer class
    // indices. Returns the scalar summed CE loss L.
    double forward(const Tensor& h, const Tensor& targets);

    // Backward. Computes d_logits for every head, accumulates per-head grad_W
    // and grad_b, returns d_trunk (N, d_model). `learning_rate` is unused
    // here (weight update is a separate update_weights call).
    Tensor backward(double learning_rate);

    // Apply -lr * grad for every per-head parameter (W_k, b_k pair).
    void update_weights(double learning_rate);

    // Zero every per-head grad.
    void zero_grad();

    // 2 * n_future parameters in the order W_0, b_0, W_1, b_1, ..., W_{n-1}, b_{n-1}.
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    // Accessors
    size_t d_model()    const { return d_model_; }
    size_t vocab_size() const { return vocab_; }
    size_t n_future()   const { return n_future_; }
    std::vector<Tensor>& weights() { return W_; }
    std::vector<Tensor>& biases()  { return b_; }
    std::vector<Tensor>& grad_weights() { return grad_W_; }
    std::vector<Tensor>& grad_biases()  { return grad_b_; }
    const std::vector<Tensor>& weights() const { return W_; }
    const std::vector<Tensor>& biases()  const { return b_; }
    const std::vector<Tensor>& grad_weights() const { return grad_W_; }
    const std::vector<Tensor>& grad_biases()  const { return grad_b_; }
    // last_logits_[k] is (N, vocab), the per-head logits from the most recent forward.
    const std::vector<Tensor>& last_logits() const { return last_logits_; }
    // last_h_ is the (N, d_model) trunk representation that the head saw.
    const Tensor& last_input() const { return last_h_; }

private:
    size_t d_model_, vocab_, n_future_;

    std::vector<Tensor> W_;        // n_future tensors, each (vocab, d_model)
    std::vector<Tensor> b_;        // n_future tensors, each (1, vocab)
    std::vector<Tensor> grad_W_;   // matching shapes
    std::vector<Tensor> grad_b_;   // matching shapes

    // Caches
    Tensor last_h_;                                    // (N, d_model)
    std::vector<Tensor> last_logits_;                  // n_future × (N, vocab)
    std::vector<Tensor> last_p_;                       // n_future × (N, vocab) softmax probs
    std::vector<size_t> last_N_valid_;                 // n_future long, valid-mask counts
    Tensor last_targets_;                              // (N, n_future)
};

// ----------------------------------------------------------------------------
// MultiTokenPredictionModel — Dense input projection + the head, wrapped as a
// Layer. forward(input) returns the n_future-th head's logits (the deepest
// head's prediction, paper §4.1 inference convention). The summed-CE training
// loop lives in the test (matches the NGPTModel convention — test computes
// loss against shifted targets, calls backward, calls update_weights).
// ----------------------------------------------------------------------------
class MultiTokenPredictionModel : public Layer {
public:
    MultiTokenPredictionModel(size_t input_dim, size_t d_model, size_t vocab_size,
                              size_t n_future);

    // Input: (N, input_dim) -> logits: (N, vocab) from the deepest head.
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return head_.weights()[0]; }
    Tensor get_gradients() const override { return head_.grad_weights()[0]; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "MultiTokenPredictionModel"; }

    size_t input_dim() const { return input_dim_; }
    size_t d_model()   const { return d_model_; }
    size_t vocab_size() const { return vocab_; }
    size_t n_future()   const { return n_future_; }
    MultiTokenPredictionHead& head() { return head_; }
    Dense& input_proj() { return input_proj_; }
    const Dense& input_proj() const { return input_proj_; }

private:
    size_t input_dim_, d_model_, vocab_, n_future_;
    Dense input_proj_;
    MultiTokenPredictionHead head_;

    // Caches
    Tensor last_trunk_;       // output of input_proj, (N, d_model)
    Tensor last_head_logits_; // output of forward, (N, vocab) — the deepest head
};

#endif // MULTI_TOKEN_PREDICTION_H
