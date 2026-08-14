// ============================================================================
// HGRN (Hierarchically Gated Recurrent Neural Network) — Mehrdad 2023
//   "Hierarchically Gated Recurrent Neural Network"
//   https://arxiv.org/abs/2307.02226
//
// Per-channel scalar-state linear RNN with separate forget/input gates (and
// optionally an output gate). See hgrn.h for full math description.
// ============================================================================

#include "nn/layers/recurrent/hgrn.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// HGRNCell
// ---------------------------------------------------------------------------

double HGRNCell::sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

HGRNCell::HGRNCell(size_t input_dim, size_t hidden_dim, bool use_output_gate)
    : W_f(input_dim, hidden_dim),
      W_i(input_dim, hidden_dim),
      W_z(input_dim, hidden_dim),
      W_o(input_dim, hidden_dim),
      input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      use_output_gate_(use_output_gate)
{
    if (input_dim == 0 || hidden_dim == 0) {
        throw std::invalid_argument("HGRNCell: input_dim and hidden_dim must be > 0");
    }
}

Tensor HGRNCell::forward(const Tensor& input) {
    if (input.cols != input_dim_) {
        throw std::invalid_argument("HGRNCell::forward: input.cols must equal input_dim");
    }
    const size_t T = input.rows;

    // Cache input
    cache_input_ = input.clone();

    // Project all tokens at once via Dense (Dense.forward is batched).
    Tensor pre_f = W_f.forward(input);   // (T, hidden_dim)
    Tensor pre_i = W_i.forward(input);
    Tensor pre_z = W_z.forward(input);

    // Allocate caches
    cache_pre_f_ = Tensor(T, hidden_dim_);
    cache_pre_i_ = Tensor(T, hidden_dim_);
    cache_pre_z_ = Tensor(T, hidden_dim_);
    cache_pre_o_ = Tensor(T, hidden_dim_);   // unused when !use_output_gate_
    cache_f_ = Tensor(T, hidden_dim_);
    cache_i_ = Tensor(T, hidden_dim_);
    cache_z_ = Tensor(T, hidden_dim_);
    cache_o_ = Tensor(T, hidden_dim_);       // unused when !use_output_gate_
    cache_c_ = Tensor(T, hidden_dim_);

    Tensor pre_o(T, hidden_dim_);
    if (use_output_gate_) {
        pre_o = W_o.forward(input);
    } else {
        pre_o.fill(0.0);
    }

    Tensor output(T, hidden_dim_);

    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < hidden_dim_; ++c) {
            const double fpre = pre_f[t][c];
            const double ipre = pre_i[t][c];
            const double zpre = pre_z[t][c];
            const double opre = pre_o[t][c];
            const double ft = sigmoid(fpre);
            const double it_ = sigmoid(ipre);
            const double zt = std::tanh(zpre);
            const double ot = use_output_gate_ ? sigmoid(opre) : 1.0;
            const double c_prev = (t == 0) ? 0.0 : cache_c_[t - 1][c];
            const double ct = ft * c_prev + it_ * zt;

            cache_pre_f_[t][c] = fpre;
            cache_pre_i_[t][c] = ipre;
            cache_pre_z_[t][c] = zpre;
            cache_pre_o_[t][c] = opre;
            cache_f_[t][c] = ft;
            cache_i_[t][c] = it_;
            cache_z_[t][c] = zt;
            cache_o_[t][c] = ot;
            cache_c_[t][c] = ct;

            if (use_output_gate_) {
                output[t][c] = ot * ct;
            } else {
                output[t][c] = ct;
            }
        }
    }
    return output;
}

Tensor HGRNCell::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != cache_c_.rows || grad_output.cols != hidden_dim_) {
        throw std::invalid_argument("HGRNCell::backward: grad_output shape mismatch");
    }
    const size_t T = grad_output.rows;

    // Reverse-sweep grad_c, then compute gradients w.r.t. f_pre, i_pre, z_pre, o_pre.
    // grad_c[t] is the gradient of the loss w.r.t. c_t.
    Tensor grad_c(T, hidden_dim_);
    grad_c.fill(0.0);
    Tensor grad_o_pre(T, hidden_dim_);   // for HGRN-2
    grad_o_pre.fill(0.0);

    // Step 1: init grad_c[T-1] from grad_output[T-1], then sweep backward.
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < hidden_dim_; ++c) {
            const double gh = grad_output[t][c];
            if (use_output_gate_) {
                // h_t = o_t * c_t → grad_c_t (contribution from h_t) = o_t * grad_h_t
                grad_c[t][c] += cache_o_[t][c] * gh;
                // grad_o_t = c_t * grad_h_t; grad_o_pre = grad_o_t * sigmoid'(o_pre)
                const double o = cache_o_[t][c];
                grad_o_pre[t][c] = cache_c_[t][c] * gh * o * (1.0 - o);
            } else {
                // h_t = c_t
                grad_c[t][c] += gh;
            }
        }
    }
    // Step 2: carry forward grad_c[t-1] += grad_c[t] * f_t
    for (size_t t = T; t-- > 1;) {  // t = T-1, T-2, ..., 1; then add to t-1
        for (size_t c = 0; c < hidden_dim_; ++c) {
            grad_c[t - 1][c] += grad_c[t][c] * cache_f_[t][c];
        }
    }

    // Step 3: per-channel gradients for f_pre, i_pre, z_pre.
    // grad_f_pre[t][c] = grad_c[t][c] * c_{t-1}[c] * sigmoid'(f_pre[t][c])
    //                  = grad_c[t][c] * c_{t-1}[c] * f_t[c] * (1 - f_t[c])
    // grad_i_pre[t][c] = grad_c[t][c] * z_t[c] * i_t[c] * (1 - i_t[c])
    // grad_z_pre[t][c] = grad_c[t][c] * i_t[c] * (1 - z_t[c]^2)
    Tensor grad_f_pre(T, hidden_dim_), grad_i_pre(T, hidden_dim_), grad_z_pre(T, hidden_dim_);
    grad_f_pre.fill(0.0); grad_i_pre.fill(0.0); grad_z_pre.fill(0.0);

    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < hidden_dim_; ++c) {
            const double gc = grad_c[t][c];
            const double c_prev = (t == 0) ? 0.0 : cache_c_[t - 1][c];
            const double f = cache_f_[t][c];
            const double it_ = cache_i_[t][c];
            const double z = cache_z_[t][c];
            grad_f_pre[t][c] = gc * c_prev * f * (1.0 - f);
            grad_i_pre[t][c] = gc * z * it_ * (1.0 - it_);
            grad_z_pre[t][c] = gc * it_ * (1.0 - z * z);
        }
    }

    // Step 4: hand the pre-activation gradients to Dense::backward to compute
    // grad_W, grad_b, and grad_x. Each Dense keeps its own last_input — since
    // we called W_f.forward(input) etc. with the SAME input, all three have
    // identical last_input. The grad_x contributions sum into grad_input.
    //
    // Note: Dense::backward ACCUMULATES into grad_weights/grad_bias, so we
    // must zero them first (the caller may have done so already via zero_grad,
    // but to be safe and idempotent we rely on the caller's contract — i.e.,
    // backward assumes zero_grad was called). For tests, this matches the
    // repo convention.
    Tensor grad_x = W_f.backward(grad_f_pre, 0.0);
    Tensor gx_i  = W_i.backward(grad_i_pre, 0.0);
    Tensor gx_z  = W_z.backward(grad_z_pre, 0.0);
    grad_x = grad_x + gx_i + gx_z;

    if (use_output_gate_) {
        Tensor gx_o = W_o.backward(grad_o_pre, 0.0);
        grad_x = grad_x + gx_o;
    }

    return grad_x;
}

void HGRNCell::update_weights(double learning_rate) {
    W_f.update_weights(learning_rate);
    W_i.update_weights(learning_rate);
    W_z.update_weights(learning_rate);
    if (use_output_gate_) {
        W_o.update_weights(learning_rate);
    }
}

void HGRNCell::zero_grad() {
    W_f.zero_grad();
    W_i.zero_grad();
    W_z.zero_grad();
    if (use_output_gate_) {
        W_o.zero_grad();
    }
}

std::vector<Tensor*> HGRNCell::parameters() {
    std::vector<Tensor*> p = {&W_f.weights, &W_f.bias,
                              &W_i.weights, &W_i.bias,
                              &W_z.weights, &W_z.bias};
    if (use_output_gate_) {
        p.push_back(&W_o.weights);
        p.push_back(&W_o.bias);
    }
    return p;
}

std::vector<Tensor*> HGRNCell::gradients() {
    std::vector<Tensor*> g = {&W_f.grad_weights, &W_f.grad_bias,
                              &W_i.grad_weights, &W_i.grad_bias,
                              &W_z.grad_weights, &W_z.grad_bias};
    if (use_output_gate_) {
        g.push_back(&W_o.grad_weights);
        g.push_back(&W_o.grad_bias);
    }
    return g;
}

// ---------------------------------------------------------------------------
// HGRNModel
// ---------------------------------------------------------------------------

HGRNModel::HGRNModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
                     size_t num_layers, bool use_output_gate)
    : classifier(hidden_dim, output_dim)
{
    if (input_dim == 0 || hidden_dim == 0 || output_dim == 0 || num_layers == 0) {
        throw std::invalid_argument("HGRNModel: dims and num_layers must be > 0");
    }
    cells.reserve(num_layers);
    size_t in_d = input_dim;
    for (size_t i = 0; i < num_layers; ++i) {
        cells.push_back(std::make_unique<HGRNCell>(in_d, hidden_dim, use_output_gate));
        in_d = hidden_dim;  // subsequent cells take hidden_dim features
    }
}

Tensor HGRNModel::forward(const Tensor& input) {
    if (input.cols != cells.front()->input_dim()) {
        throw std::invalid_argument("HGRNModel::forward: input cols must match first cell input_dim");
    }
    last_input_ = input.clone();

    Tensor x = input;
    last_cell_outputs_.clear();
    last_cell_outputs_.reserve(cells.size());
    for (auto& cell : cells) {
        x = cell->forward(x);
        last_cell_outputs_.push_back(x.clone());
    }

    // Take the last timestep (1, hidden_dim) — `x.get_row(T-1)` returns row as (1, hidden_dim).
    const size_t T = x.rows;
    Tensor last_row = x.get_row(T - 1);   // (1, hidden_dim)
    Tensor y = classifier.forward(last_row);  // (1, output_dim)
    return y;
}

Tensor HGRNModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != 1 || grad_output.cols != classifier.weights.rows) {
        throw std::invalid_argument("HGRNModel::backward: grad_output shape mismatch");
    }
    // Classifier backward: returns (1, hidden_dim)
    Tensor grad_last_row = classifier.backward(grad_output, 0.0);

    // Broadcast to (T, hidden_dim) for the last cell's backward.
    const Tensor& last_cell_out = last_cell_outputs_.back();
    const size_t T = last_cell_out.rows;
    const size_t H = last_cell_out.cols;
    Tensor grad_cell(T, H);
    grad_cell.fill(0.0);
    for (size_t c = 0; c < H; ++c) {
        grad_cell[T - 1][c] = grad_last_row[0][c];
    }

    // Sweep cells in reverse order.
    Tensor grad_to_input = grad_cell;
    for (size_t i = cells.size(); i-- > 0;) {
        Tensor grad_input_for_layer = cells[i]->backward(grad_to_input, 0.0);
        if (i == 0) {
            // We don't propagate further — the model "input" is the user-supplied input.
            return grad_input_for_layer;
        }
        // Next layer's grad_input has the same shape as that layer's input.
        // For cells 1..L-1, the "input" is the previous cell's output (T, hidden_dim).
        // grad_to_input should equal grad_input_for_layer (shape matches).
        grad_to_input = grad_input_for_layer;
    }
    // Unreachable; return empty.
    return Tensor();
}

void HGRNModel::update_weights(double learning_rate) {
    for (auto& cell : cells) cell->update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}

void HGRNModel::zero_grad() {
    for (auto& cell : cells) cell->zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> HGRNModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& cell : cells) {
        auto cp = cell->parameters();
        p.insert(p.end(), cp.begin(), cp.end());
    }
    auto cp = classifier.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> HGRNModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& cell : cells) {
        auto cg = cell->gradients();
        g.insert(g.end(), cg.begin(), cg.end());
    }
    auto cg = classifier.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
