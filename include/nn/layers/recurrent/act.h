#ifndef ACT_H
#define ACT_H

#include "../../core/layer.h"

// ACTRecurrentLayer — Adaptive Computation Time (Graves 2016)
// https://arxiv.org/abs/1603.08983
//
// Wraps a simple tanh RNN cell with a learnable halting head. At each step
// t the cell computes (h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b), y_t = h_t)
// and a halting logit p_t = sigmoid(W_halt @ y_t + b_halt). A running sum
// N_t = sum_{s <= t} p_s tracks how much "compute budget" has been spent.
// The cell halts at the first step where N_{t-1} < 1 AND N_{t-1} + p_t >= 1
// (or at the last step if the sum never crosses 1). The output is a
// per-step weighted average of y_t, where the final-step weight is the
// remainder budget 1 - N_{t-1}, so the per-step weights always sum to 1.
// The ponder penalty is exposed as a configurable hyperparameter that adds
// a constant per-step contribution to the halting-logit gradient.
//
// N=1 (single sample per forward call) for v1 — batched ACT with per-sample
// variable halting is a future extension.
class ACTRecurrentLayer : public Layer {
public:
    ACTRecurrentLayer(int input_dim, int hidden_size, int max_steps,
                      double ponder_penalty = 0.01);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "ACTRecurrentLayer"; }

    // Accessors
    int input_dim() const { return input_dim_; }
    int hidden_size() const { return hidden_size_; }
    int max_steps() const { return max_steps_; }
    double ponder_penalty() const { return ponder_penalty_; }

    // ACT-specific introspection (post-forward)
    int actual_steps() const { return halt_step_ + 1; }
    double halt_probability(int t) const { return halt_probs_[0][t]; }
    double weight_at(int t) const { return weights_[0][t]; }
    double halt_step() const { return static_cast<double>(halt_step_); }
    double ponder_cost() const;

    // Test-only setters (kept public for the test harness; production code
    // should use the random init or the default init).
    void set_W_xh(int i, int j, double v) { W_xh_[i][j] = v; }
    void set_W_hh(int i, int j, double v) { W_hh_[i][j] = v; }
    void set_b(int i, double v) { b_[i][0] = v; }
    void set_W_halt(int i, int j, double v) { W_halt_[i][j] = v; }
    void set_b_halt(double v) { b_halt_[0][0] = v; }
    void zero_all_weights();
    void random_init(unsigned seed);

    // Test-only direct value accessors (for FD comparison)
    double W_xh_at(int i, int j) const { return W_xh_[i][j]; }
    double W_hh_at(int i, int j) const { return W_hh_[i][j]; }
    double b_at(int i) const { return b_[i][0]; }
    double W_halt_at(int i, int j) const { return W_halt_[i][j]; }
    double b_halt_at() const { return b_halt_[0][0]; }

    // Test-only tensor references (for mutation testing / snapshot/restore)
    Tensor& W_xh_accessor() { return W_xh_; }
    Tensor& W_hh_accessor() { return W_hh_; }
    Tensor& b_accessor() { return b_; }
    Tensor& W_halt_accessor() { return W_halt_; }
    Tensor& grad_W_xh_accessor() { return grad_W_xh_; }
    Tensor& grad_W_hh_accessor() { return grad_W_hh_; }
    Tensor& grad_b_accessor() { return grad_b_; }
    Tensor& grad_W_halt_accessor() { return grad_W_halt_; }
    Tensor& grad_b_halt_accessor() { return grad_b_halt_; }

    // Test-only direct gradient accessors (for FD comparison)
    double grad_W_xh_at(int i, int j) const { return grad_W_xh_[i][j]; }
    double grad_W_hh_at(int i, int j) const { return grad_W_hh_[i][j]; }
    double grad_b_at(int i) const { return grad_b_[i][0]; }
    double grad_W_halt_at(int i, int j) const { return grad_W_halt_[i][j]; }
    double grad_b_halt_at() const { return grad_b_halt_[0][0]; }

private:
    int input_dim_;
    int hidden_size_;
    int max_steps_;
    double ponder_penalty_;

    // RNN cell weights: tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
    Tensor W_xh_;   // (hidden, input)
    Tensor W_hh_;   // (hidden, hidden)
    Tensor b_;      // (hidden, 1)

    Tensor grad_W_xh_;
    Tensor grad_W_hh_;
    Tensor grad_b_;

    // Halting head: p_t = sigmoid(W_halt @ y_t + b_halt)
    Tensor W_halt_; // (hidden, 1)
    Tensor b_halt_; // (1, 1)

    Tensor grad_W_halt_;
    Tensor grad_b_halt_;

    // Caches (rebuilt every forward)
    Tensor inputs_;         // (1, max_steps * input_dim) — copy of input
    Tensor hidden_states_;  // (max_steps + 1, hidden) — h_0 .. h_T
    Tensor outputs_;        // (max_steps, hidden) — y_0 .. y_{T-1}
    Tensor halt_probs_;     // (1, max_steps) — p_0 .. p_{T-1}
    Tensor halt_logits_;    // (1, max_steps) — z_0 .. z_{T-1} (pre-sigmoid)
    Tensor weights_;        // (1, max_steps) — per-step output weights
    int    halt_step_;      // step where halting happened
    double total_ponder_;   // sum of p_t for t in [0, halt_step]
};

#endif