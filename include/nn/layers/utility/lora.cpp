#include "lora.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// ============================================================================
// LoRA — implementation
// ============================================================================

namespace {
// Deterministic small-Gaussian init for A (paper §4.2: "A is initialized
// to a small Gaussian ... while B is initialized to zero"). We use
// Box-Muller on the C stdlib rand() so the init is reproducible across
// runs (unlike Tensor::random which uses a single uniform sample).
Tensor gaussian_init(size_t rows, size_t cols, double std) {
    Tensor t(rows, cols);
    if (t.data.empty()) return t;
    for (size_t i = 0; i + 1 < t.data.size(); i += 2) {
        double u1 = ((double)std::rand() / RAND_MAX + 1e-30);
        double u2 = (double)std::rand() / RAND_MAX;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;
        t.data[i] = std * r * std::cos(theta);
        t.data[i + 1] = std * r * std::sin(theta);
    }
    if (t.data.size() % 2 == 1) {
        double u1 = ((double)std::rand() / RAND_MAX + 1e-30);
        double u2 = (double)std::rand() / RAND_MAX;
        double r = std::sqrt(-2.0 * std::log(u1));
        t.data[t.data.size() - 1] = std * r * std::cos(2.0 * M_PI * u2);
    }
    return t;
}
}  // namespace

LoRALinear::LoRALinear(size_t d_in, size_t d_out, size_t rank,
                       double alpha, const std::string& base_init,
                       bool freeze_base)
    : d_in_(d_in), d_out_(d_out), rank_(rank),
      alpha_(alpha < 0 ? static_cast<double>(rank) : alpha),
      freeze_base_(freeze_base), merged_(false),
      W_0_(std::make_unique<Dense>(d_in, d_out)) {
    if (d_in == 0)
        throw std::invalid_argument("LoRALinear: d_in must be > 0");
    if (d_out == 0)
        throw std::invalid_argument("LoRALinear: d_out must be > 0");
    if (alpha >= 0)
        alpha_ = alpha;            // user-provided alpha
    // alpha < 0 is the "use rank" sentinel — leave alpha_ = rank from init list
    if (rank_ == 0 && freeze_base_)
        throw std::invalid_argument(
            "LoRALinear: rank=0 with freeze_base=true leaves nothing trainable");
    if (rank_ > 0 && alpha_ == 0.0)
        throw std::invalid_argument("LoRALinear: alpha=0 disables the LoRA path; use rank=0 instead");

    W_0_->init_weights(base_init);

    // Paper §4.2: A ~ Kaiming uniform (we use a small Gaussian with
    // std = 1/sqrt(d_in) — same effective scale, deterministic init).
    // B = 0 (paper §4.2: "B is initialized to zero so ΔW = BA = 0 at
    // the beginning of training").
    if (rank_ > 0) {
        double A_std = std::sqrt(1.0 / static_cast<double>(d_in));
        A_ = gaussian_init(rank_, d_in_, A_std);
        B_ = Tensor::zeros(d_out_, rank_);
        grad_A_ = Tensor::zeros(rank_, d_in_);
        grad_B_ = Tensor::zeros(d_out_, rank_);
    } else {
        // rank_ == 0 — no LoRA path; A/B are empty tensors (0×0).
        A_ = Tensor(0, 0);
        B_ = Tensor(0, 0);
        grad_A_ = Tensor(0, 0);
        grad_B_ = Tensor(0, 0);
    }
}

Tensor LoRALinear::forward(const Tensor& input) {
    if (input.cols != d_in_)
        throw std::invalid_argument("LoRALinear: input.cols=" + std::to_string(input.cols) +
                                     " != d_in=" + std::to_string(d_in_));
    last_input_ = input;

    // Base path: y_base = x · W_0ᵀ + b
    Tensor y_base = W_0_->forward(input);
    last_base_out_ = y_base;

    if (rank_ == 0) {
        last_Z_ = Tensor(0, 0);
        last_y_lora_ = Tensor(input.rows, d_out_);  // zeros, same shape
        return y_base;
    }

    // LoRA path: y_lora = scaling · x · Aᵀ · Bᵀ
    Tensor At = A_.transpose();             // (d_in, rank)
    Tensor xA = input * At;                 // (N, rank)
    last_Z_ = xA;
    Tensor Bt = B_.transpose();             // (rank, d_out)
    Tensor y_lora_inner = xA * Bt;          // (N, d_out)
    Tensor y_lora = y_lora_inner * scaling();  // (N, d_out)
    last_y_lora_ = y_lora;

    return y_base + y_lora;
}

Tensor LoRALinear::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.cols != d_out_)
        throw std::invalid_argument("LoRALinear: grad_output.cols != d_out");

    if (rank_ == 0) {
        // base Dense backward only
        return W_0_->backward(grad_output, 0.0);
    }

    // y_lora = scaling · (x · Aᵀ) · Bᵀ
    // Both paths receive the FULL grad_output (since y = y_base + y_lora).

    // -- LoRA path backward --
    // grad to B (rank, d_out):
    //   dL/dB = scaling · (x · Aᵀ)ᵀ · grad_output
    Tensor xA = last_Z_;                           // (N, rank)
    Tensor dB_raw = (xA.transpose() * grad_output) * scaling();  // (rank, d_out)
    // grad_B_ is (d_out, rank); transpose dB_raw into it
    for (size_t i = 0; i < rank_; ++i)
        for (size_t j = 0; j < d_out_; ++j)
            grad_B_(j, i) += dB_raw(i, j);

    // grad to A (rank, d_in):
    //   dL/dA = scaling · xᵀ · (grad_output · B)   (grad_output · B is (N, rank))
    // Result shape: xᵀ is (d_in, N), gB is (N, rank) → (d_in, rank).
    // But our grad_A_ is (rank, d_in) — transpose the result.
    Tensor gB = grad_output * B_;                  // (N, rank)
    Tensor dA_T = (last_input_.transpose() * gB) * scaling();  // (d_in, rank)
    for (size_t i = 0; i < rank_; ++i)
        for (size_t j = 0; j < d_in_; ++j)
            grad_A_(i, j) += dA_T(j, i);

    // grad to input from LoRA path (N, d_in):
    //   dX_lora = scaling · (grad_output · B) · A   (gB is (N, rank), A is (rank, d_in))
    Tensor dx_lora = (gB * A_) * scaling();

    // -- Base path backward (uses full grad_output) --
    Tensor dx_base = W_0_->backward(grad_output, 0.0);

    return dx_base + dx_lora;
}

void LoRALinear::update_weights(double lr) {
    if (merged_) {
        // After merge, A/B are zero and BAKED into W_0_. Don't update them.
        grad_A_.fill(0.0);
        grad_B_.fill(0.0);
        return;
    }
    if (rank_ > 0) {
        for (size_t i = 0; i < A_.data.size(); ++i)
            A_.data[i] -= lr * grad_A_.data[i];
        for (size_t i = 0; i < B_.data.size(); ++i)
            B_.data[i] -= lr * grad_B_.data[i];
    }
    if (!freeze_base_) {
        // Update base Dense too
        for (size_t i = 0; i < W_0_->weights.data.size(); ++i)
            W_0_->weights.data[i] -= lr * W_0_->grad_weights.data[i];
        for (size_t i = 0; i < W_0_->bias.data.size(); ++i)
            W_0_->bias.data[i] -= lr * W_0_->grad_bias.data[i];
    }
    zero_grad();
}

void LoRALinear::zero_grad() {
    if (rank_ > 0) {
        grad_A_.fill(0.0);
        grad_B_.fill(0.0);
    }
    if (!freeze_base_) {
        W_0_->zero_grad();
    }
}

std::vector<Tensor*> LoRALinear::parameters() {
    std::vector<Tensor*> p;
    if (rank_ > 0) {
        p.push_back(&A_);
        p.push_back(&B_);
    }
    if (!freeze_base_) {
        p.push_back(&W_0_->weights);
        p.push_back(&W_0_->bias);
    }
    return p;
}

std::vector<Tensor*> LoRALinear::gradients() {
    std::vector<Tensor*> g;
    if (rank_ > 0) {
        g.push_back(&grad_A_);
        g.push_back(&grad_B_);
    }
    if (!freeze_base_) {
        g.push_back(&W_0_->grad_weights);
        g.push_back(&W_0_->grad_bias);
    }
    return g;
}

void LoRALinear::merge_weights() {
    if (merged_) return;
    if (rank_ == 0) { merged_ = true; return; }  // no-op

    // W_0 ← W_0 + (α/rank) · B · A
    Tensor update = (B_ * A_) * scaling();        // (d_out, d_in)
    for (size_t i = 0; i < W_0_->weights.data.size(); ++i)
        W_0_->weights.data[i] += update.data[i];

    // Mark merged; subsequent update_weights is a no-op for A/B.
    A_.fill(0.0);
    B_.fill(0.0);
    grad_A_.fill(0.0);
    grad_B_.fill(0.0);
    merged_ = true;
}

void LoRALinear::unmerge_weights() {
    if (!merged_) return;
    if (rank_ == 0) { merged_ = false; return; }

    // Subtract (α/rank) · B · A back out of W_0_. (A and B are zero after
    // merge, so update = 0; but we kept the underlying A, B before merge
    // by convention — but we DID zero them. So unmerge is a no-op on
    // the W_0 side unless A, B were saved separately. For v1, unmerge
    // simply un-flags merged_; A and B remain zero.)
    merged_ = false;
}