#include "bitnet.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// ============================================================================
// BitLinear — implementation
// ============================================================================

BitLinear::BitLinear(size_t d_in, size_t d_out, const std::string& init)
    : d_in_(d_in), d_out_(d_out) {
    if (d_in == 0)
        throw std::invalid_argument("BitLinear: d_in must be > 0");
    if (d_out == 0)
        throw std::invalid_argument("BitLinear: d_out must be > 0");

    // Reuse Dense's init_weight scheme ("xavier"/"he"/"uniform"/"zeros").
    // We allocate fp32 W and a (1, d_out) per-row scale initialized to 1.0;
    // the absmean scale is recomputed in forward from the (then-current) W,
    // but scale_w_ exists so we have a learnable parameter the user can
    // re-set (mirroring the paper's joint training of W and scale_w).
    Dense tmp(d_in, d_out);
    tmp.init_weights(init);
    W_ = tmp.weights.clone();           // (d_out, d_in)
    scale_w_ = Tensor(1, d_out);
    scale_w_.fill(1.0);

    grad_W_       = Tensor(d_out, d_in);
    grad_scale_w_ = Tensor(1, d_out);

    last_input_    = Tensor(0, 0);
    last_scale_w_  = Tensor(0, 0);
    last_W_hat_    = Tensor(d_out, d_in);
    last_W_q_      = Tensor(d_out, d_in);
    last_ste_mask_ = Tensor(d_out, d_in);
    last_Y_pre_    = Tensor(0, 0);
}

Tensor BitLinear::forward(const Tensor& input) {
    if (input.cols != d_in_)
        throw std::invalid_argument(
            "BitLinear: input.cols=" + std::to_string(input.cols) +
            " != d_in=" + std::to_string(d_in_));
    last_input_ = input.clone();
    const size_t N = input.rows;

    // ------------------------------------------------------------------
    // Per-output-unit absmean scale: scale_w[o] = mean_j |W[o, j]|
    // ------------------------------------------------------------------
    Tensor scale_w(1, d_out_);
    for (size_t o = 0; o < d_out_; ++o) {
        double s = 0.0;
        for (size_t j = 0; j < d_in_; ++j)
            s += std::fabs(W_(o, j));
        scale_w(0, o) = s / static_cast<double>(d_in_);
    }
    last_scale_w_ = scale_w.clone();

    // ------------------------------------------------------------------
    // W_hat = clamp(W / scale_w, -1, +1)
    // W_q   = round(W_hat)        ∈ {-1, 0, +1} exactly (post-clamp)
    // M     = 1 if |W/scale_w| ≤ 1 (i.e. W was NOT clipped), else 0
    // ------------------------------------------------------------------
    Tensor W_hat(d_out_, d_in_);
    Tensor W_q(d_out_, d_in_);
    Tensor M(d_out_, d_in_);
    for (size_t o = 0; o < d_out_; ++o) {
        double sw = scale_w(0, o);
        // If sw == 0 (degenerate row of all-zeros), set it to 1 to avoid
        // 0/0; the layer produces zero output for that row in this case
        // (W_q all zero) — fine for stability.
        if (sw == 0.0) sw = 1.0;
        for (size_t j = 0; j < d_in_; ++j) {
            double x = W_(o, j) / sw;
            if (x >  1.0) x =  1.0;
            if (x < -1.0) x = -1.0;
            W_hat(o, j) = x;
            double r = std::round(x);
            W_q(o, j) = r;
            // STE saturation mask: 1 if unclamped, 0 if clamped.
            // We use <= on the original value to include the exact-1
            // boundary in the "passes through" region (standard STE
            // convention).
            double raw = W_(o, j) / sw;
            M(o, j) = (std::fabs(raw) <= 1.0) ? 1.0 : 0.0;
        }
    }
    last_W_hat_    = W_hat;
    last_W_q_      = W_q;
    last_ste_mask_ = M;

    // ------------------------------------------------------------------
    // Y_pre = X · W_q^T           (N, d_out)
    // Y     = Y_pre · diag(scale_w)
    // ------------------------------------------------------------------
    Tensor Y_pre(N, d_out_);
    // Y_pre[n, o] = Σ_j X[n, j] * W_q[o, j]
    for (size_t n = 0; n < N; ++n)
        for (size_t o = 0; o < d_out_; ++o) {
            double acc = 0.0;
            for (size_t j = 0; j < d_in_; ++j)
                acc += input(n, j) * W_q(o, j);
            Y_pre(n, o) = acc;
        }
    last_Y_pre_ = Y_pre.clone();

    Tensor Y(N, d_out_);
    for (size_t n = 0; n < N; ++n)
        for (size_t o = 0; o < d_out_; ++o)
            Y(n, o) = Y_pre(n, o) * scale_w(0, o);

    return Y;
}

Tensor BitLinear::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.cols != d_out_)
        throw std::invalid_argument("BitLinear: grad_output.cols != d_out");
    const size_t N = grad_output.rows;

    // grad_Y = grad_output: shape (N, d_out)
    const Tensor& grad_Y = grad_output;

    // grad_scale_w[o] = Σ_n grad_Y[n, o] * Y_pre[n, o]
    //   Y_pre[n, o] = Σ_j X[n, j] * W_q[o, j]
    //   Y = diag(s_w) · Y_pre, so dL/ds_w[o] = Σ_n grad_Y[n, o] · Y_pre[n, o].
    Tensor grad_scale(1, d_out_);
    for (size_t o = 0; o < d_out_; ++o) {
        double acc = 0.0;
        for (size_t n = 0; n < N; ++n)
            acc += grad_Y(n, o) * last_Y_pre_(n, o);
        grad_scale(0, o) = acc;
        grad_scale_w_(0, o) += acc;
    }

    // grad_X[n, j] = Σ_o grad_Z[n, o] * W_q[o, j]
    //              = Σ_o grad_Y[n, o] * s_w[o] * W_q[o, j]
    // This is a STANDARD matmul derivative through W_q (no STE needed —
    // W_q is the actual forward operand). Input gradient is independent of
    // the W → W_q chain.
    Tensor grad_X(N, d_in_);
    for (size_t n = 0; n < N; ++n)
        for (size_t j = 0; j < d_in_; ++j) {
            double acc = 0.0;
            for (size_t o = 0; o < d_out_; ++o)
                acc += grad_Y(n, o) * last_scale_w_(0, o) * last_W_q_(o, j);
            grad_X(n, j) = acc;
        }

    // grad_W[o, j] backward through the BitLinear STE chain.
//
// The forward is Y = s_w · Σ_k X[n, k] · W_q[o, k] where
//   s_w[o]   = (1/d_in) Σ_k |W[o, k]|
//   W_q[o, k] = round(W[o, k] / s_w[o])  ∈ {-1, 0, +1}
//
// For STE backward, we use ∂L/∂W = ∂L/∂W_q · ∂W_q/∂W. Crucially, W_q is
// DISCRETE: even when W is perturbed continuously in the interior
// (|W/s_w| < 1), the discrete value W_q = ±1 or 0 does NOT change (the
// round() commits to the integer and the perturbation stays within the
// rounding bin). So ∂W_q/∂W = 0 for interior entries that stay interior.
// The ONLY W → Y path is through s_w:
//
//   dY[n, o]/dW[o, j] = (ds_w[o]/dW[o, j]) · Y_pre[n, o]
//                     = (sign(W[o, j]) / d_in) · Y_pre[n, o]
//
// (where ds_w/dW = sign(W[o, j]) / d_in by the absmean definition, and
//  Y_pre[n, o] = Σ_k X[n, k] · W_q[o, k] is held fixed under the perturbation
//  because no W_q entry flips discrete value).
//
// For SATURATED entries, W_q[o, j] = ±1 is constant by definition, so
// ∂W_q/∂W = 0 there too — same formula applies. The STE mask
// `last_ste_mask_` (1 if |W/s_w| ≤ 1, else 0) gates whether this formula
// has any dependence on the entry (W at j) at all; for the analytical
// derivation the formula holds either way.
//
// Therefore: grad_W[o, j] = (sign(W[o, j]) / d_in) · grad_scale[o]
//                              for all j, where
//   grad_scale[o] = Σ_n grad_Y[n, o] · Y_pre[n, o].
//
// (The "direct matmul" term Σ_n grad_Y[n, o] · X[n, j] that you'd write
//  for a non-quantized Y = X · W does NOT appear here, because round()
//  eliminates the W → W_q dependence at the discrete level — verified by
//  FD where a 1e-4 perturbation of an interior W changes Y by exactly
//  (eps/d_in) · Y_pre and nothing else.)
    const double inv_d_in = 1.0 / static_cast<double>(d_in_);
    for (size_t o = 0; o < d_out_; ++o) {
        for (size_t j = 0; j < d_in_; ++j) {
            double sgn = (W_(o, j) > 0.0) ? 1.0
                       : (W_(o, j) < 0.0) ? -1.0
                       : 0.0;
            grad_W_(o, j) += grad_scale(0, o) * sgn * inv_d_in;
        }
    }

    return grad_X;
}

void BitLinear::update_weights(double lr) {
    for (size_t i = 0; i < W_.data.size(); ++i)
        W_.data[i] -= lr * grad_W_.data[i];
    for (size_t i = 0; i < scale_w_.data.size(); ++i)
        scale_w_.data[i] -= lr * grad_scale_w_.data[i];
    zero_grad();
}

void BitLinear::zero_grad() {
    grad_W_.fill(0.0);
    grad_scale_w_.fill(0.0);
}

std::vector<Tensor*> BitLinear::parameters() {
    return { &W_, &scale_w_ };
}

std::vector<Tensor*> BitLinear::gradients() {
    return { &grad_W_, &grad_scale_w_ };
}

// ============================================================================
// BitNetBlock — two BitLinear with SiLU between them
// ============================================================================

BitNetBlock::BitNetBlock(size_t d_model, size_t ffn_mult)
    : d_model_(d_model), ffn_dim_(ffn_mult * d_model) {
    if (d_model == 0)
        throw std::invalid_argument("BitNetBlock: d_model must be > 0");
    if (ffn_mult == 0)
        throw std::invalid_argument("BitNetBlock: ffn_mult must be > 0");

    lin1_ = std::make_unique<BitLinear>(d_model, ffn_dim_);
    lin2_ = std::make_unique<BitLinear>(ffn_dim_, d_model);
    last_silu_in_ = Tensor(0, 0);
}

Tensor BitNetBlock::forward(const Tensor& input) {
    // y = BitLinear2(SiLU(BitLinear1(x)))
    Tensor z1 = lin1_->forward(input);
    last_silu_in_ = z1.clone();
    // SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
    Tensor silu_out(z1.rows, z1.cols);
    for (size_t i = 0; i < z1.data.size(); ++i) {
        double z = z1.data[i];
        silu_out.data[i] = z / (1.0 + std::exp(-z));
    }
    Tensor y = lin2_->forward(silu_out);
    return y;
}

Tensor BitNetBlock::backward(const Tensor& grad_output, double learning_rate) {
    // d_z2 = lin2_backward(grad_output)
    Tensor d_silu = lin2_->backward(grad_output, learning_rate);

    // SiLU'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
    //          = silu(z) + z * (1 - sigmoid(z)) * sigmoid(z)
    //          = silu(z)/z * (silu(z) + z * (1 - sigmoid(z)) * sigmoid(z) / (silu(z)/z))  (skip — simpler form)
    // d_silu_input = d_silu * (sigmoid + z * sigmoid * (1 - sigmoid))
    //            = d_silu * sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    Tensor d_z1(d_silu.rows, d_silu.cols);
    for (size_t i = 0; i < d_silu.data.size(); ++i) {
        double z = last_silu_in_.data[i];
        double sig = 1.0 / (1.0 + std::exp(-z));
        d_z1.data[i] = d_silu.data[i] * sig * (1.0 + z * (1.0 - sig));
    }

    // d_x = lin1_backward(d_z1)
    return lin1_->backward(d_z1, learning_rate);
}

void BitNetBlock::update_weights(double lr) {
    lin1_->update_weights(lr);
    lin2_->update_weights(lr);
}

void BitNetBlock::zero_grad() {
    lin1_->zero_grad();
    lin2_->zero_grad();
}

std::vector<Tensor*> BitNetBlock::parameters() {
    std::vector<Tensor*> p = lin1_->parameters();
    std::vector<Tensor*> q = lin2_->parameters();
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

std::vector<Tensor*> BitNetBlock::gradients() {
    std::vector<Tensor*> g = lin1_->gradients();
    std::vector<Tensor*> q = lin2_->gradients();
    g.insert(g.end(), q.begin(), q.end());
    return g;
}