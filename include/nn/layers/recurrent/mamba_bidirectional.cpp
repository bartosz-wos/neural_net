#include "mamba_bidirectional.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// Bidirectional Mamba (BiMamba) — Implementation
// ============================================================================
//
// Architecture:
//   y_f  = forward_mamba.forward(x)             (T, d_inner) — forward SSM scan
//   x_rev[t] = x[T-1-t]                          — time-reversed input
//   y_b  = backward_mamba.forward(x_rev)         (T, d_inner) — backward SSM scan
//
//   mix_input[t, c]   = concat(y_f[t, c], y_b[t, c])     (T, 2*d_inner)
//   mix_logits[t, c]  = mix_proj_.forward(mix_input)[t, c]
//   mix[t, 0, j]      = exp(mix_logits[t, j]) / Z_j(t)
//   mix[t, 1, j]      = exp(mix_logits[t, j+d_inner]) / Z_j(t)
//                       where  Z_j(t) = exp(mix_logits[t, j]) + exp(mix_logits[t, j+d_inner])
//   gated_out[t, j]   = mix[t, 0, j] · y_f[t, j] + mix[t, 1, j] · y_b[t, j]
//   output[t, k]      = output_proj(gated_out[t, :])   (T, d_model)
//
// (Note: in this implementation, the gated_out is the FINAL input to the
// output projection — NOT a second Dense. We reuse the standard
// MambaBlock::backward via a special path where we feed gated_out to a
// pre-existing MambaBlock's `forward_with_input` path. To keep the code
// simple and avoid touching MambaBlock, we instead implement the
// MambaBidirectional::backward explicitly with the per-channel softmax mix.)
//
// Backward:
//   dL/dgated_out  = ... (via output projection backward)
//   dL/dmix_logits = mix * (dL/dgated_out . out - sum_c mix . (dL/dgated_out . out))
//                              where out[0]=y_f, out[1]=y_b
//   dL/d(mix_input)[t, c] = mix_proj_.backward(dL/dmix_logits)[t, c]
//   dL/dy_f[t, j]  = dL/dgated_out[t, j] · mix[t, 0, j] + (mix projection input chain to j)
//   dL/dy_b[t, j]  = dL/dgated_out[t, j] · mix[t, 1, j] + (mix projection input chain to j+d_inner)
//
//   Then chain into forward_mamba.backward(dL/dy_f, lr) and
//   backward_mamba.backward(dL/dy_b, lr) — both are existing, machine-precision
//   verified implementations.
//
//   The time-reversal chain for the backward Mamba is handled by feeding
//   `x_rev` as input during forward, then taking `grad_x_rev` from backward
//   and applying `grad_x[t] = grad_x_rev[T-1-t]` to recover dL/dx_orig for
//   the input projection.
//
// (Full per-token output projection gradient derivation is below.)
// ============================================================================

static inline double mb_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}
static inline double mb_silu(double x) {
    return x * mb_sigmoid(x);
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

MambaBidirectional::MambaBidirectional(size_t d_model, size_t d_state, size_t d_inner)
    : mix_proj_(1, 1),  // placeholder (1,1) so Dense doesn't throw; reassigned below
      out_proj_(d_inner, d_model),  // BiMamba's own output projection (d_inner -> d_model)
      forward_mamba(d_model, d_state, d_inner),
      backward_mamba(d_model, d_state, d_inner),
      d_model_(d_model), d_state_(d_state)
{
    if (d_model == 0 || d_state == 0 || d_inner == 0) {
        throw std::invalid_argument("MambaBidirectional: d_model, d_state, d_inner must be > 0");
    }
    d_inner_ = d_inner;

    // mix_proj: takes (T, 2*d_inner) → (T, 2*d_inner). Per (token, j) the
    // output[c=0] is the forward-mix logit for channel j, output[c=1*d_inner+j]
    // is the backward-mix logit for channel j.
    mix_proj_ = Dense(2 * d_inner, 2 * d_inner);
    // Initialize mix_proj.bias to 0 → softmax gives [0.5, 0.5] per (token, j) initially.
    mix_proj_.bias.fill(0.0);
    // BiMamba's out_proj has independent init — Xavier via Dense default.
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

Tensor MambaBidirectional::forward(const Tensor& input) {
    size_t T = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("MambaBidirectional: input.cols must equal d_model");
    }
    if (T < 1) {
        throw std::invalid_argument("MambaBidirectional: input must have at least one token");
    }
    last_input_ = input.clone();

    // Time-reversed input
    last_x_rev_ = Tensor(T, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            last_x_rev_(t, j) = input(T - 1 - t, j);

    // Forward Mamba scan — capture the GATED output (silu(g) * o, BEFORE out_proj)
    // because that's what the mix projection should combine. The output projection
    // is applied AFTER the mix (so it sees the combined mix-weighted gated output).
    forward_mamba.forward(input);
    Tensor y_f = forward_mamba.last_gated();
    last_forward_gated_ = y_f.clone();

    // Backward Mamba scan on the reversed input — capture the gated output (pre-proj).
    backward_mamba.forward(last_x_rev_);
    Tensor y_b = backward_mamba.last_gated();
    last_backward_gated_ = y_b.clone();

    // Build mix input = concat(forward_gated, backward_gated) per token
    Tensor mix_input(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_inner_; ++j) {
            mix_input(t, j)              = y_f(t, j);
            mix_input(t, d_inner_ + j)   = y_b(t, j);
        }
    }

    // Mix projection
    last_mix_logits_ = mix_proj_.forward(mix_input);
    // Softmax over the 2 logits per (token, j)
    last_mix_ = Tensor(T, 2 * d_inner_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_inner_; ++j) {
            double l0 = last_mix_logits_(t, j);
            double l1 = last_mix_logits_(t, d_inner_ + j);
            double m  = std::max(l0, l1);
            double e0 = std::exp(l0 - m);
            double e1 = std::exp(l1 - m);
            double z  = e0 + e1;
            last_mix_(t, j)            = e0 / z;
            last_mix_(t, d_inner_ + j) = e1 / z;
        }
    }

    // Gated output = mix-weighted combination
    last_gated_out_ = Tensor(T, d_inner_);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < d_inner_; ++j)
            last_gated_out_(t, j) = last_mix_(t, j) * y_f(t, j)
                                  + last_mix_(t, d_inner_ + j) * y_b(t, j);

    // Output projection (BiMamba's own Dense: d_inner -> d_model)
    //   output[t, k] = sum_j last_gated_out_[t, j] * out_proj_.weights(k, j) + out_proj_.bias(0, k)
    Tensor output = out_proj_.forward(last_gated_out_);
    return output;
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
//
// Step 1: chain through output projection (Dense math, w shaped (d_model, d_inner))
//   dL/dlast_gated_out[t, j] = sum_k dL/doutput[t, k] * w_out[k, j]
//   dL/dw_out[k, j]         += dL/doutput[t, k] * last_gated_out[t, j]
//   dL/db_out[k]            += sum_t dL/doutput[t, k]
//
// Step 2: chain through mix (per (token, j), 2-way softmax)
//   gated_out[t, j] = mix[t, 0, j] * y_f[t, j] + mix[t, 1, j] * y_b[t, j]
//   where mix[t, 0, j] = e^l0 / (e^l0 + e^l1), mix[t, 1, j] = e^l1 / (e^l0 + e^l1),
//         l0 = mix_logits[t, j], l1 = mix_logits[t, j+d_inner]
//
//   dL/dmix_logits[t, j]           = grad_gated_out[t, j] * (y_f[t, j] - gated_out[t, j])
//   dL/dmix_logits[t, j+d_inner]   = grad_gated_out[t, j] * (y_b[t, j] - gated_out[t, j])
//   (the standard 2-way softmax Jacobian with the mix-weighted output form)
//
//   dL/dy_f[t, j]  = grad_gated_out[t, j] * mix[t, 0, j]
//   dL/dy_b[t, j]  = grad_gated_out[t, j] * mix[t, 1, j]
//
// Step 3: chain through mix_proj (Dense backward)
//   grad_mix_input = mix_proj_.backward(grad_mix_logits, lr)
//   The first d_inner_ channels of grad_mix_input are an additional
//   dL/dy_f contribution (from the input chain), and the second d_inner_
//   channels are an additional dL/dy_b contribution.
//
//   Specifically: grad_mix_input[t, c] for c in [0, d_inner) is dL/dy_f[t, c]
//                 grad_mix_input[t, c] for c in [d_inner, 2*d_inner) is dL/dy_b[t, c-d_inner]
//   So we add:
//     dL/dy_f[t, j] += grad_mix_input[t, j]
//     dL/dy_b[t, j] += grad_mix_input[t, d_inner_ + j]
//
// Step 4: chain into forward_mamba.backward(dL/dy_f, lr)  → grad_x_fwd (T, d_model)
//        and backward_mamba.backward(dL/dy_b, lr)         → grad_x_rev (T, d_model)
//
// Step 5: time-reversal chain for backward Mamba's contribution to dL/dx_orig:
//   grad_x_rev[t] = dL/dx_rev[t]  (the backward Mamba's chain on x_rev)
//   grad_x_orig[t] += grad_x_rev[T-1-t]
//
// Step 6: total grad_input = grad_x_fwd + grad_x_rev_permuted.

Tensor MambaBidirectional::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = last_input_.rows;
    if (grad_output.rows != T || grad_output.cols != d_model_) {
        throw std::invalid_argument("MambaBidirectional: grad_output shape mismatch");
    }

    // Zero all internal grads (forward_mamba, backward_mamba, mix_proj)
    forward_mamba.zero_grad();
    backward_mamba.zero_grad();
    mix_proj_.zero_grad();

    // ------ Step 1: output projection (Dense backward on out_proj_) ------
    // grad_last_gated_out[t, j] = sum_k grad_output[t, k] * out_proj_.weights(k, j)
    Tensor grad_gated_out = out_proj_.backward(grad_output, learning_rate);
    // (out_proj_.backward also updates out_proj_.grad_weights and out_proj_.grad_bias internally)

    // ------ Step 2: chain through mix (per (token, j), 2-way softmax) ------
    //
    // Per (token, j): gated_out[t, j] = mix[t, 0, j] * y_f[t, j] + mix[t, 1, j] * y_b[t, j]
    // with mix[t, c, j] = e^l_c / (e^l_0 + e^l_1).
    //
    // Jacobian of the softmax:
    //   dmix[0]/dl_0 = mix[0] * (1 - mix[0])     dmix[0]/dl_1 = -mix[0] * mix[1]
    //   dmix[1]/dl_0 = -mix[1] * mix[0]           dmix[1]/dl_1 = mix[1] * (1 - mix[1])
    //
    // Chain to gated_out:
    //   d(gated_out)/dl_0 = mix[0] * (y_f - gated_out)     (matches the closed form below)
    //   d(gated_out)/dl_1 = mix[1] * (y_b - gated_out)
    //
    // Therefore:
    //   grad_mix_logits[t, j]          = g_go * mix[0] * (y_f - gated_out)
    //   grad_mix_logits[t, j+d_inner]  = g_go * mix[1] * (y_b - gated_out)
    //
    // Direct gradient contribution to y_f and y_b:
    //   grad_y_f[t, j] += g_go * mix[0]
    //   grad_y_b[t, j] += g_go * mix[1]
    Tensor grad_y_f(T, d_inner_);
    Tensor grad_y_b(T, d_inner_);
    grad_y_f.fill(0.0);
    grad_y_b.fill(0.0);

    Tensor grad_mix_logits(T, 2 * d_inner_);
    grad_mix_logits.fill(0.0);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_inner_; ++j) {
            double g_go = grad_gated_out(t, j);
            double m0 = last_mix_(t, j);
            double m1 = last_mix_(t, d_inner_ + j);
            grad_y_f(t, j) += g_go * m0;
            grad_y_b(t, j) += g_go * m1;
            double yf = last_forward_gated_(t, j);
            double yb = last_backward_gated_(t, j);
            double go = last_gated_out_(t, j);
            grad_mix_logits(t, j)            = g_go * m0 * (yf - go);
            grad_mix_logits(t, d_inner_ + j) = g_go * m1 * (yb - go);
        }
    }

    // ------ Step 3: chain through mix_proj (Dense backward) ------
    // grad_mix_input = mix_proj_.backward(grad_mix_logits, lr)
    // The backward call mutates mix_proj_.grad_weights and mix_proj_.grad_bias
    // and returns grad_mix_input = dL/dmix_input.
    Tensor grad_mix_input = mix_proj_.backward(grad_mix_logits, learning_rate);

    // Add the mix_proj input chain to grad_y_f / grad_y_b
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_inner_; ++j) {
            grad_y_f(t, j) += grad_mix_input(t, j);
            grad_y_b(t, j) += grad_mix_input(t, d_inner_ + j);
        }
    }

    // ------ Step 4: chain into forward_mamba / backward_mamba ------
    // Use backward_from_gated so that grad_y_f is treated as dL/d(MambaBlock::last_gated_)
    // directly — bypassing the out_proj chain in MambaBlock, which BiMamba does NOT
    // want (it has its own out_proj_ that was already chained through in Step 1).
    Tensor grad_x_fwd = forward_mamba.backward_from_gated(grad_y_f, learning_rate);
    Tensor grad_x_rev = backward_mamba.backward_from_gated(grad_y_b, learning_rate);

    // ------ Step 5: time-reversal chain for backward Mamba ------
    // grad_x_orig[t] += grad_x_rev[T-1-t]
    Tensor grad_input(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            grad_input(t, j) = grad_x_fwd(t, j) + grad_x_rev(T - 1 - t, j);
        }
    }
    return grad_input;
}

// ----------------------------------------------------------------------------
// update_weights
// ----------------------------------------------------------------------------

void MambaBidirectional::update_weights(double learning_rate) {
    mix_proj_.update_weights(learning_rate);
    out_proj_.update_weights(learning_rate);
    forward_mamba.update_weights(learning_rate);
    backward_mamba.update_weights(learning_rate);
}

// ----------------------------------------------------------------------------
// zero_grad
// ----------------------------------------------------------------------------

void MambaBidirectional::zero_grad() {
    mix_proj_.zero_grad();
    out_proj_.zero_grad();
    forward_mamba.zero_grad();
    backward_mamba.zero_grad();
}

// ----------------------------------------------------------------------------
// parameters / gradients
// ----------------------------------------------------------------------------

std::vector<Tensor*> MambaBidirectional::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&mix_proj_.weights);
    p.push_back(&mix_proj_.bias);
    p.push_back(&out_proj_.weights);
    p.push_back(&out_proj_.bias);
    auto fp = forward_mamba.parameters();
    for (auto* pp : fp) p.push_back(pp);
    auto bp = backward_mamba.parameters();
    for (auto* pp : bp) p.push_back(pp);
    return p;
}

std::vector<Tensor*> MambaBidirectional::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&mix_proj_.grad_weights);
    g.push_back(&mix_proj_.grad_bias);
    g.push_back(&out_proj_.grad_weights);
    g.push_back(&out_proj_.grad_bias);
    auto fg = forward_mamba.gradients();
    for (auto* gg : fg) g.push_back(gg);
    auto bg = backward_mamba.gradients();
    for (auto* gg : bg) g.push_back(gg);
    return g;
}
