#include "mixture_of_experts.h"
#include <cmath>
#include <algorithm>

SparseDispatcher::SparseDispatcher(size_t num_experts, size_t k)
    : num_experts_(num_experts), k_(k), expert_counts_(num_experts, 0.0) {}

std::vector<Tensor> SparseDispatcher::dispatch(const Tensor& input,
                                                 const Tensor& gate_values) {
    // gate_values: (batch, num_experts)
    // Returns k tensors: each is the input sliced/selected for expert i
    // Simplified: select top-k based on gate values, zero out other experts
    size_t batch = input.rows;

    std::vector<Tensor> dispatched(batch * num_experts_, Tensor(0, 0));
    std::fill(expert_counts_.begin(), expert_counts_.end(), 0.0);

    for (size_t b = 0; b < batch; ++b) {
        // Find top-k expert indices
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t e = 0; e < num_experts_; ++e)
            sorted.emplace_back(gate_values[b][e], e);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < k_; ++i) {
            size_t expert_idx = sorted[i].second;
            dispatched[b * num_experts_ + expert_idx] = input.get_row(b);
            expert_counts_[expert_idx] += 1.0;
        }
    }

    return dispatched;
}

std::vector<Tensor> SparseDispatcher::combine(const std::vector<Tensor>& expert_outputs,
                                               const Tensor& gate_values) {
    // Combines expert outputs using gate weights for top-k experts per sample
    // expert_outputs: (batch * num_experts_) tensor per expert slot
    // gate_values: (batch, num_experts_)
    // Returns: per-sample combined outputs (batch, hidden_dim)
    size_t batch = gate_values.rows;
    size_t dim = expert_outputs.size() > 0 && expert_outputs[0].rows > 0
                 ? expert_outputs[0].cols : 0;

    std::vector<Tensor> combined(batch, Tensor(0, 0));

    for (size_t b = 0; b < batch; ++b) {
        // Find top-k expert indices for this sample
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t e = 0; e < num_experts_; ++e)
            sorted.emplace_back(gate_values[b][e], e);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        Tensor out(1, dim); for (size_t j = 0; j < dim; ++j) out[0][j] = 0.0;
        for (size_t i = 0; i < k_; ++i) {
            size_t expert_idx = sorted[i].second;
            const Tensor& expert_out = expert_outputs[b * num_experts_ + expert_idx];
            if (expert_out.rows > 0) {
                double g = sorted[i].first;
                for (size_t j = 0; j < dim; ++j)
                    out[0][j] += g * expert_out[0][j];
            }
        }
        combined[b] = out;
    }
    return combined;
}

double SparseDispatcher::load_balance_loss(const Tensor& gate_values,
                                            const std::vector<double>& expert_counts) {
    size_t batch = gate_values.rows;
    double num_experts = static_cast<double>(num_experts_);

    // Compute mean gate values per expert
    double total = 0.0;
    for (size_t e = 0; e < num_experts_; ++e)
        total += expert_counts[e];
    if (total < 1e-9) return 0.0;

    double load_balance = 0.0;
    for (size_t e = 0; e < num_experts_; ++e) {
        double fraction_experts = expert_counts[e] / total;
        double fraction_tokens = 0.0;
        for (size_t b = 0; b < batch; ++b)
            fraction_tokens += gate_values[b][e];
        fraction_tokens /= batch;
        load_balance += fraction_experts * fraction_tokens;
    }

    // Loss = num_experts * sum(frac_experts * frac_tokens)
    return num_experts * load_balance;
}

// === MoELayer ===

MoELayer::MoELayer(size_t hidden_dim, size_t num_experts,
                     size_t k, size_t expert_capacity)
    : num_experts_(num_experts), k_(k), expert_capacity_(expert_capacity),
      gating_network_(hidden_dim, num_experts),
      dispatcher_(num_experts, k),
      load_balance_loss_(0.0),
      last_output_(1, hidden_dim) {

    for (size_t e = 0; e < num_experts_; ++e) {
        experts_.emplace_back(hidden_dim, hidden_dim); // 1-layer experts
    }
}

Tensor MoELayer::forward(const Tensor& input) {
    // Gating: compute gate scores
    gate_values_ = gating_network_.forward(input); // (batch, num_experts)

    // Dispatch inputs to top-k experts
    auto dispatched = dispatcher_.dispatch(input, gate_values_);

    // Run each expert that received a non-empty input
    expert_outputs_ = std::vector<Tensor>(num_experts_ * input.rows, Tensor(1, 0));
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t b = 0; b < input.rows; ++b) {
            size_t idx = b * num_experts_ + e;
            if (dispatched[idx].rows > 0) {
                expert_outputs_[idx] = experts_[e].forward(dispatched[idx]);
            }
        }
    }

    // Combine: weighted sum of expert outputs using gate values
    // Only top-k are used, others contribute 0
    last_output_ = Tensor(input.rows, input.cols);
    for (size_t b = 0; b < input.rows; ++b) {
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t e = 0; e < num_experts_; ++e)
            sorted.emplace_back(gate_values_[b][e], e);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < k_; ++i) {
            size_t expert_idx = sorted[i].second;
            double gate_weight = sorted[i].first;
            const Tensor& out = expert_outputs_[b * num_experts_ + expert_idx];
            if (out.rows > 0) {
                for (size_t j = 0; j < last_output_.cols; ++j)
                    last_output_[b][j] += gate_weight * out[0][j];
            }
        }
    }

    load_balance_loss_ = dispatcher_.load_balance_loss(gate_values_,
                                                        dispatcher_.get_expert_counts());
    return last_output_;
}

Tensor MoELayer::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (batch, hidden_dim)
    // Backprop through:
    // 1. Combine step: grad flows to selected experts via gate weights
    // 2. Each expert forward
    // 3. Gating network
    size_t batch = grad_output.rows;
    size_t hidden_dim = grad_output.cols;

    // Gradient w.r.t. each expert input (dispatched input)
    std::vector<Tensor> grad_expert_inputs(num_experts_ * batch, Tensor(0, 0));

    for (size_t b = 0; b < batch; ++b) {
        // Get top-k experts for this sample from stored gate_values_
        std::vector<std::pair<double, size_t>> sorted;
        for (size_t e = 0; e < num_experts_; ++e)
            sorted.emplace_back(gate_values_[b][e], e);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < k_; ++i) {
            size_t expert_idx = sorted[i].second;
            double gate_weight = sorted[i].first;
            Tensor grad_exp_input(1, hidden_dim);
            for (size_t j = 0; j < hidden_dim; ++j)
                grad_exp_input[0][j] = grad_output[b][j] * gate_weight;
            grad_expert_inputs[b * num_experts_ + expert_idx] = grad_exp_input;
        }
    }

    // Backprop through each expert
    for (size_t e = 0; e < num_experts_; ++e) {
        for (size_t b = 0; b < batch; ++b) {
            size_t idx = b * num_experts_ + e;
            if (expert_outputs_[idx].rows > 0) {
                experts_[e].backward(grad_expert_inputs[idx], learning_rate);
            }
        }
    }

    // Gradient through gating network (straight-through estimator for Top-K)
    // d(gate_values)/d(input) = 1 (identity passthrough for sorting)
    Tensor grad_gating = gating_network_.backward(grad_output, learning_rate);

    // Gradient to input
    return grad_gating;
}
void MoELayer::update_weights(double learning_rate) {
    gating_network_.update_weights(learning_rate);
    for (auto& e : experts_) e.update_weights(learning_rate);
}

void MoELayer::zero_grad() {
    gating_network_.zero_grad();
    for (auto& e : experts_) e.zero_grad();
}

std::vector<Tensor*> MoELayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : gating_network_.parameters()) result.push_back(p);
    for (auto& e : experts_)
        for (Tensor* p : e.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> MoELayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : gating_network_.gradients()) result.push_back(g);
    for (auto& e : experts_)
        for (Tensor* g : e.gradients()) result.push_back(g);
    return result;
}