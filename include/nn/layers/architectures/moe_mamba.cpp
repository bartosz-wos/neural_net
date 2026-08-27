#include "moe_mamba.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// MoEMambaBlock
// ============================================================================

MoEMambaBlock::MoEMambaBlock(size_t d_model, size_t n_heads, size_t num_experts,
                             size_t expert_d_inner, double capacity_factor,
                             double aux_loss_alpha)
    : d_model_(d_model), n_heads_(n_heads), num_experts_(num_experts),
      capacity_factor_(capacity_factor), aux_loss_alpha_(aux_loss_alpha),
      W_g_(d_model, num_experts) {
    if (d_model == 0)
        throw std::invalid_argument("MoEMambaBlock: d_model must be > 0");
    if (n_heads == 0)
        throw std::invalid_argument("MoEMambaBlock: n_heads must be > 0");
    if (num_experts == 0)
        throw std::invalid_argument("MoEMambaBlock: num_experts must be > 0");

    if (expert_d_inner == 0) {
        expert_d_inner_ = 2 * d_model;
    } else {
        expert_d_inner_ = expert_d_inner;
    }
    if (expert_d_inner_ % n_heads != 0) {
        throw std::invalid_argument(
            "MoEMambaBlock: expert_d_inner must be divisible by n_heads");
    }
    if (capacity_factor <= 0.0)
        throw std::invalid_argument("MoEMambaBlock: capacity_factor must be > 0");
    if (aux_loss_alpha < 0.0)
        throw std::invalid_argument("MoEMambaBlock: aux_loss_alpha must be >= 0");

    // Initialize W_g with small Gaussian (Xavier-like) — std = 1/sqrt(d_model).
    std::mt19937 rng(0xCAFE0001u);
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(static_cast<double>(d_model)));
    std::vector<double> w(num_experts * d_model);
    for (auto& v : w) v = nd(rng);
    W_g_.weights = Tensor(num_experts, d_model, w.data());
    W_g_.bias = Tensor(1, num_experts);
    W_g_.grad_weights = Tensor(num_experts, d_model);
    W_g_.grad_bias = Tensor(1, num_experts);
    std::fill(W_g_.grad_weights.data.begin(), W_g_.grad_weights.data.end(), 0.0);
    std::fill(W_g_.grad_bias.data.begin(), W_g_.grad_bias.data.end(), 0.0);

    experts_.reserve(num_experts_);
    for (size_t e = 0; e < num_experts_; ++e) {
        experts_.emplace_back(
            std::make_unique<Mamba2Block>(d_model_, n_heads_, expert_d_inner_));
    }

    // Allocate cache fields with correct shapes (empty for now).
    last_gate_logits_        = Tensor(0, num_experts_);
    last_gate_scores_        = Tensor(0, num_experts_);
    last_route_indices_      = Tensor(0, 1);
    last_route_mask_         = Tensor(0, num_experts_);
    last_capacity_mask_      = Tensor(0, num_experts_);
    last_input_              = Tensor(0, d_model_);
    last_token_expert_count_ = Tensor(1, num_experts_);
    last_load_balance_loss_  = 0.0;
}

Tensor MoEMambaBlock::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("MoEMambaBlock: input.cols must equal d_model");
    last_input_ = input.clone();

    // Router: gate_logits = x · W_g^T  (Dense.forward).
    last_gate_logits_ = W_g_.forward(input);  // (T, num_experts)

    // Sigmoid + argmax + one-hot route mask.
    last_gate_scores_   = Tensor(T, num_experts_);
    last_route_indices_ = Tensor(T, 1);
    last_route_mask_    = Tensor(T, num_experts_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t e = 0; e < num_experts_; ++e) {
            double z = last_gate_logits_(t, e);
            last_gate_scores_(t, e) = 1.0 / (1.0 + std::exp(-z));
        }
        size_t best_e = 0;
        double best_s = last_gate_scores_(t, 0);
        for (size_t e = 1; e < num_experts_; ++e) {
            if (last_gate_scores_(t, e) > best_s) {
                best_s = last_gate_scores_(t, e);
                best_e = e;
            }
        }
        last_route_indices_(t, 0) = static_cast<double>(best_e);
        for (size_t e = 0; e < num_experts_; ++e)
            last_route_mask_(t, e) = (e == best_e) ? 1.0 : 0.0;
    }

    // Count tokens per expert (pre-capacity).
    last_token_expert_count_.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        size_t e = static_cast<size_t>(last_route_indices_(t, 0));
        last_token_expert_count_(0, e) += 1.0;
    }

    // Capacity: ceil(T * cap_factor / N). Min 1.
    size_t capacity = static_cast<size_t>(
        std::ceil(static_cast<double>(T) * capacity_factor_ /
                  static_cast<double>(num_experts_)));
    if (capacity < 1) capacity = 1;

    last_capacity_mask_ = Tensor(T, num_experts_);
    Tensor expert_count_so_far(1, num_experts_);
    expert_count_so_far.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        size_t e = static_cast<size_t>(last_route_indices_(t, 0));
        bool allow = (expert_count_so_far(0, e) < static_cast<double>(capacity));
        for (size_t i = 0; i < num_experts_; ++i)
            last_capacity_mask_(t, i) = (i == e && allow) ? 1.0 : 0.0;
        if (allow) expert_count_so_far(0, e) += 1.0;
    }

    // Forward each expert on the FULL input; mask its output by capacity_mask.
    Tensor output(T, d_model_);
    output.fill(0.0);
    for (size_t e = 0; e < num_experts_; ++e) {
        Tensor expert_out = experts_[e]->forward(input);
        for (size_t t = 0; t < T; ++t) {
            double m = last_capacity_mask_(t, e);
            if (m > 0.0) {
                for (size_t j = 0; j < d_model_; ++j)
                    output(t, j) += m * expert_out(t, j);
            }
        }
    }

    // Auxiliary load-balance loss.
    if (T > 0) {
        double sum_loss = 0.0;
        for (size_t e = 0; e < num_experts_; ++e) {
            double f_e = last_token_expert_count_(0, e) / static_cast<double>(T);
            double p_e = 0.0;
            for (size_t t = 0; t < T; ++t) p_e += last_gate_scores_(t, e);
            p_e /= static_cast<double>(T);
            sum_loss += f_e * p_e;
        }
        last_load_balance_loss_ =
            aux_loss_alpha_ * static_cast<double>(num_experts_) * sum_loss;
    } else {
        last_load_balance_loss_ = 0.0;
    }

    return output;
}

Tensor MoEMambaBlock::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = grad_output.rows;
    if (grad_output.cols != d_model_)
        throw std::invalid_argument("MoEMambaBlock: grad_output.cols must equal d_model");

    // Sum per-expert input gradients.
    Tensor grad_input(T, d_model_);
    grad_input.fill(0.0);

    for (size_t e = 0; e < num_experts_; ++e) {
        Tensor expert_grad_out(T, d_model_);
        for (size_t t = 0; t < T; ++t) {
            double m = last_capacity_mask_(t, e);
            for (size_t j = 0; j < d_model_; ++j)
                expert_grad_out(t, j) = m * grad_output(t, j);
        }
        Tensor expert_grad_in =
            experts_[e]->backward(expert_grad_out, learning_rate);
        grad_input += expert_grad_in;
    }

    // Router gradient (load-balance loss only — Switch routing is non-differentiable).
    // d L_aux / d gate_scores[t, e] = alpha * N * f_e / T  (constant per row e).
    // Chain through sigmoid: d gate_scores / d gate_logits = s * (1 - s).
    Tensor grad_gate_logits(T, num_experts_);
    if (aux_loss_alpha_ > 0.0 && T > 0) {
        for (size_t e = 0; e < num_experts_; ++e) {
            double f_e = last_token_expert_count_(0, e) / static_cast<double>(T);
            double base = aux_loss_alpha_ * static_cast<double>(num_experts_)
                          * f_e / static_cast<double>(T);
            for (size_t t = 0; t < T; ++t) {
                double s = last_gate_scores_(t, e);
                grad_gate_logits(t, e) = base * s * (1.0 - s);
            }
        }
    } else {
        grad_gate_logits.fill(0.0);
    }

    // W_g.backward chains through its own Dense gradient update + returns d_input.
    Tensor grad_W_g_input = W_g_.backward(grad_gate_logits, learning_rate);
    grad_input += grad_W_g_input;

    return grad_input;
}

void MoEMambaBlock::update_weights(double learning_rate) {
    W_g_.update_weights(learning_rate);
    for (auto& e : experts_) e->update_weights(learning_rate);
}

void MoEMambaBlock::zero_grad() {
    W_g_.zero_grad();
    for (auto& e : experts_) e->zero_grad();
}

std::vector<Tensor*> MoEMambaBlock::parameters() {
    std::vector<Tensor*> p = W_g_.parameters();
    for (auto& e : experts_) {
        auto ep = e->parameters();
        p.insert(p.end(), ep.begin(), ep.end());
    }
    return p;
}

std::vector<Tensor*> MoEMambaBlock::gradients() {
    std::vector<Tensor*> g = W_g_.gradients();
    for (auto& e : experts_) {
        auto eg = e->gradients();
        g.insert(g.end(), eg.begin(), eg.end());
    }
    return g;
}

void MoEMambaBlock::copy_params_from(const MoEMambaBlock& other) {
    if (other.d_model_ != d_model_ || other.n_heads_ != n_heads_ ||
        other.num_experts_ != num_experts_ ||
        other.expert_d_inner_ != expert_d_inner_) {
        throw std::invalid_argument(
            "MoEMambaBlock::copy_params_from: architecture mismatch");
    }
    W_g_.weights = other.W_g_.weights.clone();
    W_g_.bias    = other.W_g_.bias.clone();
    for (size_t i = 0; i < num_experts_; ++i) {
        const Mamba2Block& src = *other.experts_[i];
        Mamba2Block& dst       = *experts_[i];
        dst.in_proj.weights  = src.in_proj.weights.clone();
        dst.in_proj.bias     = src.in_proj.bias.clone();
        dst.out_proj.weights = src.out_proj.weights.clone();
        dst.out_proj.bias    = src.out_proj.bias.clone();
        dst.a_proj.weights   = src.a_proj.weights.clone();
        dst.a_proj.bias      = src.a_proj.bias.clone();
        dst.b_proj.weights   = src.b_proj.weights.clone();
        dst.b_proj.bias      = src.b_proj.bias.clone();
        dst.k_proj.weights   = src.k_proj.weights.clone();
        dst.k_proj.bias      = src.k_proj.bias.clone();
        dst.q_proj.weights   = src.q_proj.weights.clone();
        dst.q_proj.bias      = src.q_proj.bias.clone();
        dst.D_skip           = src.D_skip.clone();
        dst.dt_bias          = src.dt_bias.clone();
    }
}

size_t MoEMambaBlock::count_parameters() const {
    // Count of distinct learnable tensors in this block (matches parameters().size()).
    size_t total = 2;  // W_g_.weights + W_g_.bias
    for (const auto& e : experts_) total += e->parameters().size();
    return total;
}

// ============================================================================
// MoEMambaModel
// ============================================================================

MoEMambaModel::MoEMambaModel(size_t input_dim, size_t d_model, size_t output_dim,
                             size_t num_layers, size_t n_heads, size_t num_experts,
                             size_t expert_d_inner, double capacity_factor,
                             double aux_loss_alpha)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_layers_(num_layers), n_heads_(n_heads), num_experts_(num_experts),
      expert_d_inner_(expert_d_inner),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0)
        throw std::invalid_argument("MoEMambaModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("MoEMambaModel: d_model must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("MoEMambaModel: output_dim must be > 0");
    if (num_layers == 0)
        throw std::invalid_argument("MoEMambaModel: num_layers must be > 0");

    blocks_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
        blocks_.emplace_back(std::make_unique<MoEMambaBlock>(
            d_model_, n_heads_, num_experts_, expert_d_inner_,
            capacity_factor, aux_loss_alpha));
    }
}

Tensor MoEMambaModel::forward(const Tensor& input) {
    Tensor h = embed_.forward(input);
    for (auto& b : blocks_) h = b->forward(h);
    h = final_ln_.forward(h);
    return classifier_.forward(h);
}

Tensor MoEMambaModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor g = classifier_.backward(grad_output, learning_rate);
    g = final_ln_.backward(g, learning_rate);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it)
        g = (*it)->backward(g, learning_rate);
    g = embed_.backward(g, learning_rate);
    return g;
}

void MoEMambaModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void MoEMambaModel::zero_grad() {
    embed_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> MoEMambaModel::parameters() {
    std::vector<Tensor*> p = embed_.parameters();
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fp = final_ln_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> MoEMambaModel::gradients() {
    std::vector<Tensor*> g = embed_.gradients();
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fg = final_ln_.gradients();
    g.insert(g.end(), fg.begin(), fg.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}