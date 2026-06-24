#include "mixture_of_depths.h"
#include <cmath>
#include <algorithm>
#include <random>

// ============================================================================
// FFNSequential — small helper Layer that does ReLU(ffn_fc1(x)) -> ffn_fc2.
// Used as the inner block of MoDBlock (real MoD wraps a transformer block;
// we use a 2-layer FFN as a tractable, fast, and parameter-rich inner block
// that exercises the routing machinery well).
// ============================================================================

class FFNSequential : public Layer {
public:
    Dense fc1;
    Dense fc2;
    Tensor last_h_pre;       // pre-ReLU cache (for ReLU backward)

    FFNSequential(size_t in_dim, size_t hidden_dim)
        : fc1(in_dim, hidden_dim), fc2(hidden_dim, in_dim) {}

    Tensor forward(const Tensor& x) override {
        Tensor h_pre = fc1.forward(x);
        last_h_pre = h_pre;
        Tensor h_act(h_pre.rows, h_pre.cols);
        for (size_t i = 0; i < h_pre.data.size(); ++i) {
            double v = h_pre.data[i];
            h_act.data[i] = v > 0.0 ? v : 0.0;
        }
        return fc2.forward(h_act);
    }
    Tensor backward(const Tensor& grad, double lr) override {
        Tensor dh_act = fc2.backward(grad, lr);
        Tensor dh_pre(dh_act.rows, dh_act.cols);
        for (size_t i = 0; i < dh_pre.data.size(); ++i) {
            double v = last_h_pre.data[i];
            dh_pre.data[i] = dh_act.data[i] * (v > 0.0 ? 1.0 : 0.0);
        }
        return fc1.backward(dh_pre, lr);
    }
    void update_weights(double lr) override { fc1.update_weights(lr); fc2.update_weights(lr); }
    void zero_grad() override { fc1.zero_grad(); fc2.zero_grad(); }
    Tensor get_weights() const override { return fc1.get_weights(); }
    Tensor get_gradients() const override { return fc1.get_gradients(); }
    std::vector<Tensor*> parameters() override {
        auto p = fc1.parameters();
        auto q = fc2.parameters();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }
    std::vector<Tensor*> gradients() override {
        auto p = fc1.gradients();
        auto q = fc2.gradients();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }
    std::string name() const override { return "FFNSequential"; }
};

// ============================================================================
// MoDLayer
// ============================================================================
//
// See mixture_of_depths.h for the full design write-up.

MoDLayer::MoDLayer(size_t d_model, double capacity_factor, double aux_loss_coef,
                   Layer* inner)
    : d_model_(d_model),
      capacity_factor_(std::max(0.0, std::min(1.0, capacity_factor))),
      capacity_(0),
      aux_loss_coef_(aux_loss_coef),
      inner_(inner),
      load_balance_loss_(0.0),
      rng_(42),
      W_router_(1, d_model),
      b_router_(1, 1),
      grad_W_router_(1, d_model),
      grad_b_router_(1, 1)
{
    // Small init for the router so initial routing is close to random.
    std::normal_distribution<> dis(0.0, 0.05);
    for (size_t j = 0; j < d_model_; ++j) W_router_(0, j) = dis(rng_);
    b_router_(0, 0) = 0.0;
    grad_W_router_.fill(0.0);
    grad_b_router_.fill(0.0);
}

std::vector<Tensor*> MoDLayer::parameters() {
    std::vector<Tensor*> p = { &W_router_, &b_router_ };
    if (inner_) {
        auto inner_p = inner_->parameters();
        p.insert(p.end(), inner_p.begin(), inner_p.end());
    }
    return p;
}

std::vector<Tensor*> MoDLayer::gradients() {
    std::vector<Tensor*> g = { &grad_W_router_, &grad_b_router_ };
    if (inner_) {
        auto inner_g = inner_->gradients();
        g.insert(g.end(), inner_g.begin(), inner_g.end());
    }
    return g;
}

void MoDLayer::zero_grad() {
    grad_W_router_.fill(0.0);
    grad_b_router_.fill(0.0);
    if (inner_) inner_->zero_grad();
}

void MoDLayer::update_weights(double lr) {
    // SGD update for router
    for (size_t j = 0; j < d_model_; ++j)
        W_router_(0, j) -= lr * grad_W_router_(0, j);
    b_router_(0, 0) -= lr * grad_b_router_(0, 0);
    if (inner_) inner_->update_weights(lr);
}

// ----------------------------------------------------------------------------
// top-k selection helper — partial sort by score descending, take first capacity
// ----------------------------------------------------------------------------
void MoDLayer::select_top_k(const std::vector<double>& scores,
                            std::vector<size_t>& indices_out) {
    size_t n = scores.size();
    indices_out.resize(n);
    for (size_t i = 0; i < n; ++i) indices_out[i] = i;
    std::partial_sort(indices_out.begin(), indices_out.begin() + capacity_, indices_out.end(),
                      [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    indices_out.resize(capacity_);
}

// ----------------------------------------------------------------------------
// forward
// ----------------------------------------------------------------------------
Tensor MoDLayer::forward(const Tensor& input) {
    size_t n = input.rows;
    input_ = input.clone();

    // 1) Router logits = input @ W_router^T + b_router   (Dense convention)
    //    W_router is (1, d_model), so logits is (n, 1).
    router_logits_ = Tensor(n, 1);
    for (size_t t = 0; t < n; ++t) {
        double s = b_router_(0, 0);
        for (size_t j = 0; j < d_model_; ++j) s += input(t, j) * W_router_(0, j);
        router_logits_(t, 0) = s;
    }

    // 2) Router probs = sigmoid(router_logits)
    router_probs_ = Tensor(n, 1);
    for (size_t t = 0; t < n; ++t) {
        double z = router_logits_(t, 0);
        router_probs_(t, 0) = 1.0 / (1.0 + std::exp(-z));
    }

    // 3) Compute capacity from current batch size.
    //    When capacity_factor_ is exactly 0, no tokens should be selected.
    //    Otherwise round up to at least 1 (so a non-zero capacity factor
    //    always routes at least one token when n >= 1).
    size_t cap = static_cast<size_t>(capacity_factor_ * static_cast<double>(n) + 1e-9);
    if (cap == 0 && capacity_factor_ > 0.0) cap = 1;
    if (cap > n) cap = n;
    capacity_ = cap;

    // 4) Top-k selection on the (n, 1) logits.
    std::vector<double> scores(n);
    for (size_t t = 0; t < n; ++t) scores[t] = router_logits_(t, 0);
    select_top_k(scores, selected_indices_);

    // 5) Build mask (n, 1): 1.0 if selected, 0.0 otherwise
    mask_ = Tensor(n, 1);
    mask_.fill(0.0);
    for (size_t idx : selected_indices_) mask_(idx, 0) = 1.0;

    // 6) Run inner on the FULL batch (same pattern as SparseMoE — clean BPTT)
    sub_out_ = inner_ ? inner_->forward(input) : input;

    // 7) Apply the mask: y = x + mask * sub_out
    Tensor output(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        double m = mask_(t, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            output(t, j) = input(t, j) + m * sub_out_(t, j);
        }
    }

    // 8) Aux loss: L_aux = alpha * n * (mean(mask) - capacity_factor)^2
    double mean_mask = 0.0;
    for (size_t t = 0; t < n; ++t) mean_mask += mask_(t, 0);
    mean_mask /= static_cast<double>(n);
    load_balance_loss_ = aux_loss_coef_ * static_cast<double>(n) *
                         (mean_mask - capacity_factor_) * (mean_mask - capacity_factor_);

    return output;
}

// ----------------------------------------------------------------------------
// backward
// ----------------------------------------------------------------------------
Tensor MoDLayer::backward(const Tensor& grad_output, double lr) {
    size_t n = grad_output.rows;

    // 1) Mask the gradient for the inner sub-layer: grad_sub_out = grad_out * mask
    Tensor grad_sub_out(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        double m = mask_(t, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            grad_sub_out(t, j) = m * grad_output(t, j);
        }
    }

    // 2) Inner sub-layer backward (gets only the masked gradient)
    Tensor d_input_inner(n, d_model_);
    d_input_inner.fill(0.0);
    if (inner_) d_input_inner = inner_->backward(grad_sub_out, lr);

    // 3) Residual path: y = x + mask * sub_out
    //    d_residual = grad_out (since y = x + ... and dL/dx from the residual
    //    is just grad_out — the residual path passes gradient through 1:1).
    Tensor d_input(n, d_model_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            d_input(t, j) = grad_output(t, j) + d_input_inner(t, j);
        }
    }

    // 4) Router backward (aux-loss path only — the top-k mask is hard).
    //    L_aux = alpha * n * (mean(mask) - cap)^2
    //    dL/d mean(mask) = alpha * n * 2 * (mean(mask) - cap)
    //    dL/d mask[t] = (1/n) * dL/d mean(mask) = 2 * alpha * (mean(mask) - cap)
    //    dL/d router_probs[t] = dL/d mask[t]
    //    dL/d router_logits[t] = dL/d router_probs[t] * p * (1 - p)
    Tensor d_router_logits(n, 1);
    if (n > 0) {
        double mean_mask = 0.0;
        for (size_t t = 0; t < n; ++t) mean_mask += mask_(t, 0);
        mean_mask /= static_cast<double>(n);
        double d_mean_mask = 2.0 * aux_loss_coef_ * (mean_mask - capacity_factor_);
        // dL/d router_probs[t] = (1/n) * dL/d mean(mask)
        double d_probs = d_mean_mask / static_cast<double>(n);
        for (size_t t = 0; t < n; ++t) {
            double p = router_probs_(t, 0);
            d_router_logits(t, 0) = d_probs * p * (1.0 - p);
        }
    }

    // 5) Router Dense backward: gate_logits = input @ W_router^T + b_router
    //    W_router is (1, d_model), so:
    //      dW_router[0, j] += sum_t d_router_logits[t, 0] * input[t, j]
    //      d_input[t, j]  += d_router_logits[t, 0] * W_router[0, j]
    //      d_b_router[0, 0] += sum_t d_router_logits[t, 0]
    for (size_t j = 0; j < d_model_; ++j) {
        double s = grad_W_router_(0, j);
        for (size_t t = 0; t < n; ++t) s += d_router_logits(t, 0) * input_(t, j);
        grad_W_router_(0, j) = s;
    }
    double db = grad_b_router_(0, 0);
    for (size_t t = 0; t < n; ++t) db += d_router_logits(t, 0);
    grad_b_router_(0, 0) = db;
    for (size_t t = 0; t < n; ++t) {
        double d = d_router_logits(t, 0);
        for (size_t j = 0; j < d_model_; ++j) {
            d_input(t, j) += d * W_router_(0, j);
        }
    }

    return d_input;
}

Tensor MoDLayer::get_weights() const { return W_router_; }
Tensor MoDLayer::get_gradients() const { return grad_W_router_; }

// ============================================================================
// MoDBlock — pre-LN → MoD-wrapped FFN chain → residual
// ============================================================================

MoDBlock::MoDBlock(size_t d_model, double capacity_factor, size_t ffn_dim,
                   double aux_loss_coef)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      capacity_factor_(capacity_factor),
      ln1_(d_model),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),       // in=d_model, out=ffn_dim (used in forward)
      ffn_fc2_(ffn_dim_, d_model),       // in=ffn_dim, out=d_model (used in forward)
      mod1_(d_model, capacity_factor, aux_loss_coef,
            new FFNSequential(d_model, ffn_dim_))  // inner FFN-as-Layer
{}

std::vector<Tensor*> MoDBlock::parameters() {
    auto p = ln1_.parameters();
    auto a = mod1_.parameters();
    auto q = ln2_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

std::vector<Tensor*> MoDBlock::gradients() {
    auto p = ln1_.gradients();
    auto a = mod1_.gradients();
    auto q = ln2_.gradients();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

void MoDBlock::zero_grad() {
    ln1_.zero_grad();
    mod1_.zero_grad();
    ln2_.zero_grad();
}

void MoDBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    mod1_.update_weights(lr);
    ln2_.update_weights(lr);
}

Tensor MoDBlock::forward(const Tensor& input) {
    // Architecture:
    //   z1 = ln1(input)
    //   h_pre = ffn_fc1(z1)              # (n, ffn_dim)
    //   h_act = relu(h_pre)
    //   ffn_residual = ffn_fc2(h_act)    # (n, d_model)
    //   z2 = ln2(input + ffn_residual)
    //   gated = mod1(z2)                 # MoD-routed refinement of z2
    //   output = input + gated
    //
    // The MoD wrap on `mod1(z2)` is the key: only `capacity_factor` of the
    // tokens get the gated refinement applied; the rest get identity (skip).
    last_input_ = input;
    Tensor z1 = ln1_.forward(input);
    Tensor h_pre = ffn_fc1_.forward(z1);
    last_h_pre_ = h_pre;
    Tensor h_act(h_pre.rows, h_pre.cols);
    for (size_t i = 0; i < h_pre.data.size(); ++i) {
        double v = h_pre.data[i];
        h_act.data[i] = v > 0.0 ? v : 0.0;
    }
    Tensor ffn_residual = ffn_fc2_.forward(h_act);   // (n, d_model)
    Tensor z2 = ln2_.forward(input + ffn_residual);
    Tensor gated = mod1_.forward(z2);
    Tensor output = input + gated;
    return output;
}

Tensor MoDBlock::backward(const Tensor& grad_output, double lr) {
    // output = input + mod1(ln2(input + ffn_fc2(relu(ffn_fc1(ln1(input))))))
    //
    // Chain rule (in reverse):
    //   d_gated = grad_output (from output = input + gated, residual)
    //   d_z2 = mod1.backward(d_gated, lr)            # into ln2's input
    //   d_(input + ffn_residual) = ln2.backward(d_z2, lr) + grad_output  (residual)
    //   d_ffn_residual = d_(input + ffn_residual)
    //   d_h_act = ffn_fc2.backward(d_ffn_residual, lr)
    //   d_h_pre = d_h_act * (h_pre > 0)              # ReLU mask
    //   d_z1 = ffn_fc1.backward(d_h_pre, lr)
    //   d_input_from_ln1 = ln1.backward(d_z1, lr)
    //   d_input_total = grad_output (outer residual)
    //                  + d_(input + ffn_residual) (inner residual from input + ffn_residual)
    //                  + d_input_from_ln1 (ln1 chain)
    Tensor d_z2 = mod1_.backward(grad_output, lr);
    Tensor d_inner_residual = ln2_.backward(d_z2, lr);
    Tensor d_ffn_residual = d_inner_residual;
    Tensor d_h_act = ffn_fc2_.backward(d_ffn_residual, lr);
    // ReLU backward using cached h_pre
    Tensor d_h_pre(d_h_act.rows, d_h_act.cols);
    for (size_t i = 0; i < d_h_pre.data.size(); ++i) {
        double v = last_h_pre_.data[i];
        d_h_pre.data[i] = d_h_act.data[i] * (v > 0.0 ? 1.0 : 0.0);
    }
    Tensor d_z1 = ffn_fc1_.backward(d_h_pre, lr);
    Tensor d_input_from_ln1 = ln1_.backward(d_z1, lr);
    Tensor d_input(d_input_from_ln1.rows, d_input_from_ln1.cols);
    for (size_t i = 0; i < d_input.data.size(); ++i) {
        d_input.data[i] = grad_output.data[i] + d_inner_residual.data[i] +
                          d_input_from_ln1.data[i];
    }
    return d_input;
}

// ============================================================================
// MoDModel — stack of MoDBlocks + classifier head
// ============================================================================

MoDModel::MoDModel(size_t d_model, size_t out_features, size_t num_blocks,
                   double capacity_factor, size_t ffn_dim, double aux_loss_coef)
    : d_model_(d_model),
      out_features_(out_features),
      total_aux_loss_(0.0),
      blocks_(),
      classifier_(d_model, out_features_)
{
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(d_model, capacity_factor, ffn_dim, aux_loss_coef);
    }
}

std::vector<Tensor*> MoDModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> MoDModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}

void MoDModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
    total_aux_loss_ = 0.0;
}

void MoDModel::update_weights(double lr) {
    for (auto& b : blocks_) b.update_weights(lr);
    classifier_.update_weights(lr);
}

Tensor MoDModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor h = input;
    total_aux_loss_ = 0.0;
    for (auto& b : blocks_) {
        h = b.forward(h);
        total_aux_loss_ += b.get_load_balance_loss();
    }
    return classifier_.forward(h);
}

Tensor MoDModel::backward(const Tensor& grad_output, double lr) {
    Tensor d = classifier_.backward(grad_output, lr);
    // Reverse-order through blocks
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d = it->backward(d, lr);
    }
    return d;
}
