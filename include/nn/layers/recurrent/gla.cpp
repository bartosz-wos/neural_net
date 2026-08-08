#include "gla.h"
#include "../../core/tensor.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>

// Initialize projections to a small random scale — small enough that the
// recurrence stays numerically stable for forward checks, large enough
// that gradient checks are non-vacuous.
static Tensor gla_rand_init(size_t rows, size_t cols) {
    Tensor t(rows, cols);
    double s = 0.5 / std::sqrt(static_cast<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t[i][j] = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0 * s;
        }
    }
    return t;
}

GatedLinearAttention::GatedLinearAttention(size_t d_model, size_t n_heads, size_t head_dim)
    : W_Q_(d_model, 0), W_K_(d_model, 0), W_V_(d_model, 0),
      W_O_(0, 0), W_gate_(d_model, 0),
      d_model_(d_model),
      n_heads_(n_heads),
      head_dim_(head_dim == 0 ? d_model / n_heads : head_dim),
      d_inner_(n_heads * (head_dim == 0 ? d_model / n_heads : head_dim))
{
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model and n_heads must be > 0");
    }
    if (head_dim_ == 0 || d_model_ % head_dim_ != 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model must divide evenly by head_dim");
    }
    // Standard convention: d_inner = d_model (so head_dim = d_model / n_heads).
    d_inner_ = d_model;

    // Build Dense projections (x · W^T + b convention)
    W_Q_   = Dense(d_model, d_inner_);
    W_K_   = Dense(d_model, d_inner_);
    W_V_   = Dense(d_model, d_inner_);
    W_O_   = Dense(d_inner_, d_model);
    W_gate_ = Dense(d_model, n_heads_);  // per-head α_t = sigmoid(W_gate · x_t)

    // Override Dense default init with a Xavier-ish small scale
    W_Q_.weights    = gla_rand_init(d_inner_, d_model_);
    W_K_.weights    = gla_rand_init(d_inner_, d_model_);
    W_V_.weights    = gla_rand_init(d_inner_, d_model_);
    W_O_.weights    = gla_rand_init(d_model_, d_inner_);
    W_gate_.weights = gla_rand_init(n_heads_, d_model_);
    // Biases default to zero in Dense
}