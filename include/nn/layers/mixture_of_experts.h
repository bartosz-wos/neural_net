#ifndef MIXTURE_OF_EXPERTS_H
#define MIXTURE_OF_EXPERTS_H

#include "../core/layer.h"
#include <vector>

// Mixture of Experts (MoE) with sparse gating.
// Each expert is an independent feedforward network.
// A gating network selects the top-k experts for each input.
// Load balancing loss encourages even expert utilization.
class SparseDispatcher {
public:
    SparseDispatcher(size_t num_experts, size_t k);
    std::vector<Tensor> dispatch(const Tensor& input, const Tensor& gate_values);
    std::vector<Tensor> combine(const std::vector<Tensor>& expert_outputs,
                                 const Tensor& gate_values);
    double load_balance_loss(const Tensor& gate_values,
                              const std::vector<double>& expert_counts);
    std::vector<double> get_expert_counts() const { return expert_counts_; }

private:
    size_t num_experts_, k_;
    std::vector<double> expert_counts_;
};

class MoELayer : public Layer {
public:
    // hidden_dim: input/output dim, num_experts: number of FFN experts,
    // k: top-k experts to use, expert_capacity: max tokens per expert (for capacity routing)
    MoELayer(size_t hidden_dim, size_t num_experts,
               size_t k = 2, size_t expert_capacity = 4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    double get_load_balance_loss() const { return load_balance_loss_; }

private:
    Dense gating_network_;    // maps input to num_experts gate scores
    std::vector<Dense> experts_; // each expert: 2-layer FFN
    size_t num_experts_, k_, expert_capacity_;
    SparseDispatcher dispatcher_;
    Tensor last_output_;
    double load_balance_loss_;
};

#endif