// soft_moe.cpp — Soft Mixture of Experts (Puigcerver et al. ICLR 2024).
#include "soft_moe.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// SoftMoELayer
// ---------------------------------------------------------------------------
SoftMoELayer::SoftMoELayer(size_t d_model, size_t num_experts,
                           size_t slots_per_expert, size_t d_expert)
    : d_model_(d_model),
      num_experts_(num_experts),
      slots_per_expert_(slots_per_expert),
      d_expert_(d_expert),
      num_slots_(num_experts * slots_per_expert),
      W_disp_(d_model, num_slots_) {       // (E*S, d_model)  — fixed at construction
    if (d_model == 0)
        throw std::invalid_argument("SoftMoELayer: d_model must be > 0");
    if (num_experts == 0)
        throw std::invalid_argument("SoftMoELayer: num_experts must be > 0");
    if (slots_per_expert == 0)
        throw std::invalid_argument("SoftMoELayer: slots_per_expert must be > 0");
    if (d_expert == 0)
        throw std::invalid_argument("SoftMoELayer: d_expert must be > 0");

    // One Dense per slot for FFN so each retains its own last_input cache.
    W1_.reserve(num_slots_);
    W2_.reserve(num_slots_);
    for (size_t i = 0; i < num_slots_; ++i) {
        W1_.emplace_back(d_model, d_expert);   // W1: (d_expert, d_model)
        W2_.emplace_back(d_expert, d_model);   // W2: (d_model, d_expert)
    }

    // W_comb_ is lazily initialized on first forward, once we know T.
}

void SoftMoELayer::ensure_W_comb_(size_t T) {
    if (!W_comb_initialized_) {
        W_comb_.clear();
        W_comb_.reserve(num_slots_);
        for (size_t i = 0; i < num_slots_; ++i) {
            W_comb_.emplace_back(d_model_, T);   // weights: (T, d_model)
        }
        W_comb_initialized_ = true;
    }
}

Tensor SoftMoELayer::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("SoftMoELayer::forward: input cols != d_model");
    const size_t T = input.rows;
    ensure_W_comb_(T);

    // 1. Dispatch logits and softmax per-row over slots.
    last_input = input;
    last_D_logits = W_disp_.forward(input);                 // (T, E*S)
    last_D_ = Tensor(T, num_slots_);
    for (size_t t = 0; t < T; ++t) {
        double mx = last_D_logits[t][0];
        for (size_t j = 1; j < num_slots_; ++j)
            if (last_D_logits[t][j] > mx) mx = last_D_logits[t][j];
        double sum = 0.0;
        for (size_t j = 0; j < num_slots_; ++j) {
            last_D_[t][j] = std::exp(last_D_logits[t][j] - mx);
            sum += last_D_[t][j];
        }
        for (size_t j = 0; j < num_slots_; ++j)
            last_D_[t][j] /= sum;
    }

    // 2. Slot inputs: X' = D^T @ X   -> (E*S, d_model)
    last_Xp_ = Tensor(num_slots_, d_model_);
    for (size_t i = 0; i < num_slots_; ++i)
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t t = 0; t < T; ++t)
                s += last_D_[t][i] * input[t][k];
            last_Xp_[i][k] = s;
        }

    // 3. Per-slot FFN.
    expert_h_pre_.clear(); expert_h_pre_.reserve(num_slots_);
    expert_h_act_.clear(); expert_h_act_.reserve(num_slots_);
    last_Y_ = Tensor(num_slots_, d_model_);
    for (size_t slot_idx = 0; slot_idx < num_slots_; ++slot_idx) {
        // Single slot input (1, d_model).
        Tensor slot_in(1, d_model_);
        for (size_t k = 0; k < d_model_; ++k)
            slot_in[0][k] = last_Xp_[slot_idx][k];

        Tensor h_pre = W1_[slot_idx].forward(slot_in);     // (1, d_expert)
        Tensor h_act(1, d_expert_);
        for (size_t j = 0; j < d_expert_; ++j)
            h_act[0][j] = h_pre[0][j] > 0.0 ? h_pre[0][j] : 0.0;
        Tensor y_slot = W2_[slot_idx].forward(h_act);       // (1, d_model)

        expert_h_pre_.push_back(h_pre);
        expert_h_act_.push_back(h_act);
        for (size_t k = 0; k < d_model_; ++k)
            last_Y_[slot_idx][k] = y_slot[0][k];
    }

    // 4. Per-slot combine logits and softmax over tokens.
    last_C_logits = Tensor(num_slots_, T);
    last_C_ = Tensor(num_slots_, T);
    for (size_t slot_idx = 0; slot_idx < num_slots_; ++slot_idx) {
        Tensor slot_in(1, d_model_);
        for (size_t k = 0; k < d_model_; ++k)
            slot_in[0][k] = last_Xp_[slot_idx][k];

        Tensor c_logits = W_comb_[slot_idx].forward(slot_in);  // (1, T)
        for (size_t t = 0; t < T; ++t)
            last_C_logits[slot_idx][t] = c_logits[0][t];

        // softmax over tokens for this slot
        double mx = last_C_logits[slot_idx][0];
        for (size_t j = 1; j < T; ++j)
            if (last_C_logits[slot_idx][j] > mx) mx = last_C_logits[slot_idx][j];
        double sum = 0.0;
        for (size_t j = 0; j < T; ++j) {
            last_C_[slot_idx][j] = std::exp(last_C_logits[slot_idx][j] - mx);
            sum += last_C_[slot_idx][j];
        }
        for (size_t j = 0; j < T; ++j)
            last_C_[slot_idx][j] /= sum;
    }

    // 5. Output = C^T @ Y    -> (T, d_model)
    Tensor output(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t i = 0; i < num_slots_; ++i)
                s += last_C_[i][t] * last_Y_[i][k];
            output[t][k] = s;
        }
    return output;
}

Tensor SoftMoELayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t T = last_input.rows;
    ensure_W_comb_(T);  // safety

    // === output = C^T @ Y ===
    // dC[i, t] = sum_k grad_output[t, k] * Y[i, k]                (E*S, T)
    // dY[i, k] = sum_t C[i, t] * grad_output[t, k]               (E*S, d_model)
    Tensor dC(num_slots_, T);
    Tensor dY(num_slots_, d_model_);
    for (size_t i = 0; i < num_slots_; ++i) {
        for (size_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                s += grad_output[t][k] * last_Y_[i][k];
            dC[i][t] = s;
        }
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t t = 0; t < T; ++t)
                s += last_C_[i][t] * grad_output[t][k];
            dY[i][k] = s;
        }
    }

    // softmax-over-tokens backward for C
    // dC_logits[i, t] = C[i, t] * (dC[i, t] - sum_u C[i, u] * dC[i, u])
    Tensor dC_logits(num_slots_, T);
    for (size_t i = 0; i < num_slots_; ++i) {
        double dot = 0.0;
        for (size_t u = 0; u < T; ++u)
            dot += last_C_[i][u] * dC[i][u];
        for (size_t t = 0; t < T; ++t)
            dC_logits[i][t] = last_C_[i][t] * (dC[i][t] - dot);
    }

    // Per-slot combine backward.
    Tensor dXp_from_C(num_slots_, d_model_);
    for (size_t slot_idx = 0; slot_idx < num_slots_; ++slot_idx) {
        Tensor grad_c_logits(1, T);
        for (size_t t = 0; t < T; ++t)
            grad_c_logits[0][t] = dC_logits[slot_idx][t];
        Tensor grad_slot_in = W_comb_[slot_idx].backward(grad_c_logits, 0.0);  // (1, d_model)
        for (size_t k = 0; k < d_model_; ++k)
            dXp_from_C[slot_idx][k] = grad_slot_in[0][k];
    }

    // === per-slot FFN backward ===
    Tensor dXp_from_Y(num_slots_, d_model_);
    dXp_from_Y.fill(0.0);
    for (size_t slot_idx = 0; slot_idx < num_slots_; ++slot_idx) {
        Tensor grad_h_act_in(1, d_model_);
        for (size_t k = 0; k < d_model_; ++k)
            grad_h_act_in[0][k] = dY[slot_idx][k];

        Tensor d_h_pre = W2_[slot_idx].backward(grad_h_act_in, 0.0);   // (1, d_expert)
        Tensor d_h_act(1, d_expert_);
        const Tensor& h_pre_cached = expert_h_pre_[slot_idx];
        for (size_t j = 0; j < d_expert_; ++j) {
            double v = h_pre_cached[0][j];
            d_h_act[0][j] = v > 0.0 ? d_h_pre[0][j] : 0.0;
        }
        Tensor grad_slot_in = W1_[slot_idx].backward(d_h_act, 0.0);    // (1, d_model)
        for (size_t k = 0; k < d_model_; ++k)
            dXp_from_Y[slot_idx][k] = grad_slot_in[0][k];
    }

    // Total slot-input gradient
    Tensor dXp(num_slots_, d_model_);
    for (size_t i = 0; i < num_slots_; ++i)
        for (size_t k = 0; k < d_model_; ++k)
            dXp[i][k] = dXp_from_Y[i][k] + dXp_from_C[i][k];

    // === X' = D^T @ X ===
    // dL/dD[t, i] = sum_k dXp[i, k] * X[t, k]
    Tensor dD(T, num_slots_);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < num_slots_; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < d_model_; ++k)
                s += dXp[i][k] * last_input[t][k];
            dD[t][i] = s;
        }

    // softmax-over-slots backward for D
    Tensor dD_logits(T, num_slots_);
    for (size_t t = 0; t < T; ++t) {
        double dot = 0.0;
        for (size_t u = 0; u < num_slots_; ++u)
            dot += last_D_[t][u] * dD[t][u];
        for (size_t i = 0; i < num_slots_; ++i)
            dD_logits[t][i] = last_D_[t][i] * (dD[t][i] - dot);
    }

    // W_disp_.backward updates W_disp.grad_weights.
    // The full chain for dL/dX[t, k] has TWO contributions:
    //   (a) indirect via D -> D_logits -> W_disp:   dD_logits @ W_disp_weights
    //   (b) direct via X -> X' (X' = D^T @ X):     sum_i dXp[i, k] * D[t, i]
    // The returned grad_input from Dense::backward is (a).  Add (b) to get the full.
    Tensor grad_input = W_disp_.backward(dD_logits, 0.0);
    for (size_t t = 0; t < T; ++t)
        for (size_t k = 0; k < d_model_; ++k)
            for (size_t i = 0; i < num_slots_; ++i)
                grad_input[t][k] += dXp[i][k] * last_D_[t][i];
    return grad_input;
}

void SoftMoELayer::update_weights(double learning_rate) {
    W_disp_.update_weights(learning_rate);
    for (auto& w : W_comb_) w.update_weights(learning_rate);
    for (size_t i = 0; i < num_slots_; ++i) {
        W1_[i].update_weights(learning_rate);
        W2_[i].update_weights(learning_rate);
    }
}

void SoftMoELayer::zero_grad() {
    W_disp_.zero_grad();
    for (auto& w : W_comb_) w.zero_grad();
    for (size_t i = 0; i < num_slots_; ++i) {
        W1_[i].zero_grad();
        W2_[i].zero_grad();
    }
}

std::vector<Tensor*> SoftMoELayer::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&W_disp_.weights);
    p.push_back(&W_disp_.bias);
    for (auto& w : W_comb_) {
        p.push_back(&w.weights);
        p.push_back(&w.bias);
    }
    for (size_t i = 0; i < num_slots_; ++i) {
        p.push_back(&W1_[i].weights);
        p.push_back(&W1_[i].bias);
        p.push_back(&W2_[i].weights);
        p.push_back(&W2_[i].bias);
    }
    return p;
}

std::vector<Tensor*> SoftMoELayer::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&W_disp_.grad_weights);
    g.push_back(&W_disp_.grad_bias);
    for (auto& w : W_comb_) {
        g.push_back(&w.grad_weights);
        g.push_back(&w.grad_bias);
    }
    for (size_t i = 0; i < num_slots_; ++i) {
        g.push_back(&W1_[i].grad_weights);
        g.push_back(&W1_[i].grad_bias);
        g.push_back(&W2_[i].grad_weights);
        g.push_back(&W2_[i].grad_bias);
    }
    return g;
}

void SoftMoELayer::copy_params_from(const SoftMoELayer& other) {
    if (d_model_ != other.d_model_ || num_experts_ != other.num_experts_ ||
        slots_per_expert_ != other.slots_per_expert_ || d_expert_ != other.d_expert_) {
        throw std::invalid_argument("SoftMoELayer::copy_params_from: shape mismatch");
    }
    W_disp_.weights = other.W_disp_.weights;
    W_disp_.bias = other.W_disp_.bias;
    if (W_comb_initialized_ && other.W_comb_initialized_) {
        for (size_t i = 0; i < num_slots_; ++i) {
            W_comb_[i].weights = other.W_comb_[i].weights;
            W_comb_[i].bias = other.W_comb_[i].bias;
        }
    }
    for (size_t i = 0; i < num_slots_; ++i) {
        W1_[i].weights = other.W1_[i].weights;
        W1_[i].bias = other.W1_[i].bias;
        W2_[i].weights = other.W2_[i].weights;
        W2_[i].bias = other.W2_[i].bias;
    }
}

size_t SoftMoELayer::count_parameters() const {
    size_t n = 0;
    n += W_disp_.weights.data.size() + W_disp_.bias.data.size();
    for (auto& w : W_comb_) {
        n += w.weights.data.size() + w.bias.data.size();
    }
    for (size_t i = 0; i < num_slots_; ++i) {
        n += W1_[i].weights.data.size() + W1_[i].bias.data.size();
        n += W2_[i].weights.data.size() + W2_[i].bias.data.size();
    }
    return n;
}

// ---------------------------------------------------------------------------
// SoftMoEModel
// ---------------------------------------------------------------------------
SoftMoEModel::SoftMoEModel(size_t input_dim, size_t d_model, size_t output_dim,
                           size_t num_layers, size_t num_experts,
                           size_t slots_per_expert, size_t d_expert)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      num_experts_(num_experts),
      slots_per_expert_(slots_per_expert),
      d_expert_(d_expert),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0 || d_model == 0 || output_dim == 0)
        throw std::invalid_argument("SoftMoEModel: input/d_model/output must be > 0");
    if (num_layers == 0)
        throw std::invalid_argument("SoftMoEModel: num_layers must be > 0");

    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.push_back(std::make_unique<SoftMoELayer>(d_model, num_experts,
                                                        slots_per_expert, d_expert));
    }
}

Tensor SoftMoEModel::forward(const Tensor& input) {
    Tensor h = embed_.forward(input);                       // (T, d_model)
    for (auto& b : blocks_) {
        h = b->forward(h);
    }
    h = final_ln_.forward(h);
    // Mean-pool over tokens
    Tensor pooled(1, d_model_);
    for (size_t k = 0; k < d_model_; ++k) {
        double s = 0.0;
        for (size_t t = 0; t < h.rows; ++t) s += h[t][k];
        pooled[0][k] = s / h.rows;
    }
    return classifier_.forward(pooled);                     // (1, output_dim)
}

Tensor SoftMoEModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor grad_pooled = classifier_.backward(grad_output, 0.0);    // (1, d_model)

    size_t T = embed_.last_input.rows;
    Tensor grad_h(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t k = 0; k < d_model_; ++k)
            grad_h[t][k] = grad_pooled[0][k] / static_cast<double>(T);

    Tensor grad_post_ln = final_ln_.backward(grad_h, 0.0);
    for (size_t i = num_layers_; i > 0; --i) {
        grad_post_ln = blocks_[i - 1]->backward(grad_post_ln, 0.0);
    }
    (void)embed_.backward(grad_post_ln, 0.0);
    return grad_post_ln;
}

void SoftMoEModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void SoftMoEModel::zero_grad() {
    embed_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> SoftMoEModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&embed_.weights);
    p.push_back(&embed_.bias);
    for (auto& b : blocks_) {
        auto sub = b->parameters();
        p.insert(p.end(), sub.begin(), sub.end());
    }
    p.push_back(&final_ln_.gamma);
    p.push_back(&final_ln_.beta);
    p.push_back(&classifier_.weights);
    p.push_back(&classifier_.bias);
    return p;
}

std::vector<Tensor*> SoftMoEModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&embed_.grad_weights);
    g.push_back(&embed_.grad_bias);
    for (auto& b : blocks_) {
        auto sub = b->gradients();
        g.insert(g.end(), sub.begin(), sub.end());
    }
    g.push_back(&final_ln_.grad_gamma_);
    g.push_back(&final_ln_.grad_beta_);
    g.push_back(&classifier_.grad_weights);
    g.push_back(&classifier_.grad_bias);
    return g;
}