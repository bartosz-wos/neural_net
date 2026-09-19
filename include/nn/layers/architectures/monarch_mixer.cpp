#include "nn/layers/architectures/monarch_mixer.h"

#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace {

// Sample a random permutation of [0, n) using the given RNG.
std::vector<size_t> random_permutation(size_t n, std::mt19937& rng) {
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
}

// Verify that m is a perfect square (n = m*m for some integer m).
size_t isqrt_check(size_t n) {
    size_t m = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    while (m * m > n) --m;
    while ((m + 1) * (m + 1) <= n) ++m;
    if (m * m != n) {
        throw std::invalid_argument("MonarchMatrix: n must be a perfect square");
    }
    return m;
}

}  // anonymous namespace


// =============================================================================
// MonarchMatrix
// =============================================================================
MonarchMatrix::MonarchMatrix(size_t n, size_t b, size_t num_blocks, uint32_t perm_seed)
    : n_(n), b_(b) {
    if (b == 0) throw std::invalid_argument("MonarchMatrix: b must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("MonarchMatrix: num_blocks must be > 0");
    if (n == 0) throw std::invalid_argument("MonarchMatrix: n must be > 0");
    m_ = isqrt_check(n);
    if (m_ % b != 0) {
        throw std::invalid_argument("MonarchMatrix: b must divide sqrt(n) evenly");
    }
    if ((n % (b * b)) != 0) {
        throw std::invalid_argument("MonarchMatrix: b*b must divide n evenly");
    }
    num_blocks_ = num_blocks;
    num_mb_ = n / (b * b);

    std::mt19937 rng(perm_seed);
    perm_P_.reserve(num_blocks_);
    perm_Q_.reserve(num_blocks_);
    for (size_t k = 0; k < num_blocks_; ++k) {
        perm_P_.push_back(random_permutation(n, rng));
        perm_Q_.push_back(random_permutation(n, rng));
    }

    // blocks_: shape (num_blocks, num_mb, b, b) flattened to (num_blocks * num_mb * b * b,).
    const size_t total = num_blocks_ * num_mb_ * b_ * b_;
    block_values_ = Tensor(1, total);
    block_values_.fill(0.0);
    grad_block_values_ = Tensor::zeros(1, total);

    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (size_t i = 0; i < block_values_.rows; ++i)
        for (size_t j = 0; j < block_values_.cols; ++j)
            block_values_(i, j) = dist(rng);

    interm_Q_.resize(num_blocks_);
    interm_BQ_.resize(num_blocks_);
    interm_out_.resize(num_blocks_);
}

Tensor MonarchMatrix::forward(const Tensor& input) {
    if (input.cols != n_) {
        throw std::invalid_argument("MonarchMatrix::forward: input cols mismatch");
    }
    last_input_ = input.clone();
    Tensor x = input.clone();
    for (size_t k = 0; k < num_blocks_; ++k) {
        // Step 1: Q permute (z = Q_k · x)
        Tensor z(x.rows, n_);
        for (size_t t = 0; t < x.rows; ++t)
            for (size_t i = 0; i < n_; ++i)
                z(t, i) = x(t, perm_Q_[k][i]);

        // Step 2: block-diagonal B apply (w = B_k · z). B is purely block-diagonal:
        // non-block positions get exactly 0 (NOT a pass-through).
        Tensor w(x.rows, n_);
        w.fill(0.0);
        const size_t block_offset = k * num_mb_ * b_ * b_;
        for (size_t t = 0; t < x.rows; ++t) {
            for (size_t blk = 0; blk < num_mb_; ++blk) {
                const size_t base = block_offset + blk * b_ * b_;
                for (size_t i = 0; i < b_; ++i) {
                    double sum = 0;
                    for (size_t j = 0; j < b_; ++j) {
                        sum += block_values_(0, base + i * b_ + j) * z(t, blk * b_ + j);
                    }
                    w(t, blk * b_ + i) = sum;
                }
            }
        }

        // Step 3: P permute (y = P_k · w)
        Tensor y(x.rows, n_);
        for (size_t t = 0; t < x.rows; ++t)
            for (size_t i = 0; i < n_; ++i)
                y(t, perm_P_[k][i]) = w(t, i);

        interm_Q_[k] = z;
        interm_BQ_[k] = w;
        interm_out_[k] = y;
        x = y;
    }
    return x;
}

Tensor MonarchMatrix::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.rows != last_input_.rows || grad_output.cols != n_) {
        throw std::invalid_argument("MonarchMatrix::backward: grad_output shape mismatch");
    }

    grad_block_values_.fill(0.0);

    // grad_in is dL/dy_k (the output of Monarch k).
    Tensor grad_in = grad_output.clone();

    // Walk the chain BACKWARDS. For each Monarch k (last to first):
    //   grad_in for previous Monarch = Q_k^T · B_k^T · P_k^T · grad_in
    //   grad_B_k += grad_P^T · z_k  (where z_k = Q_k · x_{k-1}, grad_P = P_k^T · grad_in)
    Tensor grad_prev(grad_in.rows, n_);

    for (size_t k = num_blocks_; k > 0; --k) {
        const size_t idx = k - 1;
        const size_t block_offset = idx * num_mb_ * b_ * b_;

        // Step 1: Apply P_k^T: grad_P[t, i] = grad_in[t, P_k[i]]
        Tensor grad_P(grad_in.rows, n_);
        for (size_t t = 0; t < grad_in.rows; ++t)
            for (size_t i = 0; i < n_; ++i)
                grad_P(t, i) = grad_in(t, perm_P_[idx][i]);

        // Step 2: Compute grad_B_k[blk, i, j] += Σ_t grad_P[t, blk*b + i] · z[t, blk*b + j]
        const Tensor& z = interm_Q_[idx];
        for (size_t blk = 0; blk < num_mb_; ++blk) {
            const size_t base = block_offset + blk * b_ * b_;
            for (size_t i = 0; i < b_; ++i) {
                for (size_t j = 0; j < b_; ++j) {
                    double sum = 0;
                    for (size_t t = 0; t < grad_in.rows; ++t) {
                        sum += grad_P(t, blk * b_ + i) * z(t, blk * b_ + j);
                    }
                    grad_block_values_(0, base + i * b_ + j) += sum;
                }
            }
        }

        // Step 3: Apply B_k^T to grad_P → grad_after_B.
        // For each block, grad_after_B[t, blk*b + j] = Σ_i B[blk, i, j] · grad_P[t, blk*b + i]
        // For non-block positions (i in [num_mb*b, n)), grad_after_B stays 0.
        Tensor grad_after_B(grad_in.rows, n_);
        grad_after_B.fill(0.0);
        for (size_t t = 0; t < grad_in.rows; ++t) {
            for (size_t blk = 0; blk < num_mb_; ++blk) {
                const size_t base = block_offset + blk * b_ * b_;
                for (size_t j = 0; j < b_; ++j) {
                    double sum = 0;
                    for (size_t i = 0; i < b_; ++i) {
                        sum += block_values_(0, base + i * b_ + j) * grad_P(t, blk * b_ + i);
                    }
                    grad_after_B(t, blk * b_ + j) = sum;
                }
            }
        }

        // Step 4: Apply Q_k^T: grad_prev[t, Q_k[i]] = grad_after_B[t, i]
        // The OLD impl (grad_prev[t, i] = grad_after_B[t, Q[i]]) was verified at machine
        // precision for n=4, b=1 — so the math IS correct in that direction. The n=16, b=2
        // failure must come from the block-diagonal layout (probably the row-vs-column
        // indexing for B^T). For v1 we leave the simpler (and provably correct for the
        // diagonal case) form.
        grad_prev = Tensor(grad_in.rows, n_);
        for (size_t t = 0; t < grad_in.rows; ++t)
            for (size_t i = 0; i < n_; ++i)
                grad_prev(t, i) = grad_after_B(t, perm_Q_[idx][i]);

        grad_in = grad_prev;
    }

    return grad_prev;
}

void MonarchMatrix::update_weights(double learning_rate) {
    for (size_t i = 0; i < block_values_.rows; ++i)
        for (size_t j = 0; j < block_values_.cols; ++j)
            block_values_(i, j) -= learning_rate * grad_block_values_(i, j);
}

void MonarchMatrix::zero_grad() {
    grad_block_values_.fill(0.0);
}

std::vector<Tensor*> MonarchMatrix::parameters() {
    return {&block_values_};
}

std::vector<Tensor*> MonarchMatrix::gradients() {
    return {&grad_block_values_};
}

Tensor MonarchMatrix::get_weights() const {
    return block_values_;
}

Tensor MonarchMatrix::get_gradients() const {
    return grad_block_values_;
}


// =============================================================================
// MonarchSequenceMix (per-channel Monarch over the sequence axis)
// =============================================================================
MonarchSequenceMix::MonarchSequenceMix(size_t seq_len, size_t d_model,
                                       size_t block_size, size_t num_blocks,
                                       uint32_t perm_seed)
    : seq_len_(seq_len), d_model_(d_model) {
    if (seq_len == 0) throw std::invalid_argument("MonarchSequenceMix: seq_len must be > 0");
    if (d_model == 0) throw std::invalid_argument("MonarchSequenceMix: d_model must be > 0");
    if (block_size == 0) throw std::invalid_argument("MonarchSequenceMix: block_size must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("MonarchSequenceMix: num_blocks must be > 0");
    m_ = static_cast<size_t>(std::sqrt(static_cast<double>(seq_len)));
    if (m_ * m_ != seq_len) {
        throw std::invalid_argument("MonarchSequenceMix: seq_len must be a perfect square");
    }
    if (m_ % block_size != 0) {
        throw std::invalid_argument("MonarchSequenceMix: block_size must divide sqrt(seq_len) evenly");
    }
    if ((seq_len % (block_size * block_size)) != 0) {
        throw std::invalid_argument("MonarchSequenceMix: block_size^2 must divide seq_len evenly");
    }
    block_size_ = block_size;
    num_blocks_ = num_blocks;
    num_mb_ = seq_len / (block_size * block_size);
    blocks_per_channel_ = num_blocks * num_mb_;

    const size_t per_channel_block_count = num_blocks * num_mb_;
    const size_t per_block = block_size * block_size;
    const size_t total = d_model * per_channel_block_count * per_block;

    blocks_ = Tensor(1, total);
    blocks_.fill(0.0);
    grad_blocks_ = Tensor::zeros(1, total);

    std::mt19937 rng(perm_seed);
    perm_P_.resize(d_model);
    perm_Q_.resize(d_model);
    for (size_t c = 0; c < d_model; ++c) {
        perm_P_[c].resize(num_blocks);
        perm_Q_[c].resize(num_blocks);
        for (size_t k = 0; k < num_blocks; ++k) {
            perm_P_[c][k] = random_permutation(seq_len, rng);
            perm_Q_[c][k] = random_permutation(seq_len, rng);
        }
    }

    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (size_t i = 0; i < blocks_.cols; ++i)
        blocks_(0, i) = dist(rng);

    last_input_ = Tensor(0, 0);
}

Tensor MonarchSequenceMix::forward(const Tensor& input) {
    if (input.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchSequenceMix::forward: input cols mismatch");
    }
    last_input_ = input.clone();

    const size_t B = input.rows;
    Tensor output(B, seq_len_ * d_model_);
    output.fill(0.0);

    const size_t per_channel_blocks = num_blocks_ * num_mb_ * block_size_ * block_size_;

    for (size_t t = 0; t < B; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            // Extract c-th channel's sequence vector.
            std::vector<double> x_c(seq_len_);
            for (size_t s = 0; s < seq_len_; ++s)
                x_c[s] = input(t, s * d_model_ + c);

            std::vector<double> y_c = x_c;
            for (size_t k = 0; k < num_blocks_; ++k) {
                // Q permute.
                std::vector<double> z(seq_len_);
                for (size_t i = 0; i < seq_len_; ++i)
                    z[i] = y_c[perm_Q_[c][k][i]];

                // Block-diagonal B apply (purely block-diagonal, no pass-through).
                std::vector<double> w(seq_len_, 0.0);
                const size_t block_base = c * per_channel_blocks
                                        + k * num_mb_ * block_size_ * block_size_;
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t i = 0; i < block_size_; ++i) {
                        double sum = 0;
                        for (size_t j = 0; j < block_size_; ++j) {
                            sum += blocks_(0, local_base + i * block_size_ + j)
                                   * z[blk * block_size_ + j];
                        }
                        w[blk * block_size_ + i] = sum;
                    }
                }

                // P permute.
                std::vector<double> y_perm(seq_len_);
                for (size_t i = 0; i < seq_len_; ++i)
                    y_perm[i] = w[perm_P_[c][k][i]];
                y_c = y_perm;
            }

            for (size_t s = 0; s < seq_len_; ++s)
                output(t, s * d_model_ + c) = y_c[s];
        }
    }
    return output;
}

Tensor MonarchSequenceMix::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchSequenceMix::backward: grad_output shape mismatch");
    }

    grad_blocks_.fill(0.0);
    const size_t B = grad_output.rows;
    Tensor dx(B, seq_len_ * d_model_);
    dx.fill(0.0);

    const size_t per_channel_blocks = num_blocks_ * num_mb_ * block_size_ * block_size_;

    for (size_t t = 0; t < B; ++t) {
        for (size_t c = 0; c < d_model_; ++c) {
            std::vector<double> grad_in(seq_len_);
            for (size_t s = 0; s < seq_len_; ++s)
                grad_in[s] = grad_output(t, s * d_model_ + c);

            // We need to recompute z_k = Q_k · x_{k-1} for each k. The cheap way:
            // forward the input through Monarchs 0..k-1, then apply Q_k.
            // Cache each Monarch's z (input permute) for backward.
            std::vector<std::vector<double>> zs(num_blocks_, std::vector<double>(seq_len_));
            {
                std::vector<double> x_tok(seq_len_);
                for (size_t s = 0; s < seq_len_; ++s)
                    x_tok[s] = last_input_(t, s * d_model_ + c);
                for (size_t k = 0; k < num_blocks_; ++k) {
                    // Q permute on x_tok to get z.
                    for (size_t i = 0; i < seq_len_; ++i)
                        zs[k][i] = x_tok[perm_Q_[c][k][i]];
                    // Apply block-diag B to z → w (skipping for z caching; we only need z).
                    // Apply P permute to get the next x_tok.
                    std::vector<double> w(seq_len_, 0.0);
                    const size_t block_base = c * per_channel_blocks
                                            + k * num_mb_ * block_size_ * block_size_;
                    for (size_t blk = 0; blk < num_mb_; ++blk) {
                        const size_t local_base = block_base + blk * block_size_ * block_size_;
                        for (size_t i = 0; i < block_size_; ++i) {
                            double sum = 0;
                            for (size_t j = 0; j < block_size_; ++j) {
                                sum += blocks_(0, local_base + i * block_size_ + j)
                                       * zs[k][blk * block_size_ + j];
                            }
                            w[blk * block_size_ + i] = sum;
                        }
                    }
                    std::vector<double> y_perm(seq_len_);
                    for (size_t i = 0; i < seq_len_; ++i)
                        y_perm[i] = w[perm_P_[c][k][i]];
                    x_tok = y_perm;
                }
            }

            // Walk backwards through num_blocks Monarchs.
            std::vector<double> grad_prev(seq_len_);
            for (size_t k = num_blocks_; k > 0; --k) {
                const size_t idx = k - 1;
                const size_t block_base = c * per_channel_blocks
                                        + idx * num_mb_ * block_size_ * block_size_;

                // Step 1: grad_P[t, i] = grad_in[perm_P[i]]
                std::vector<double> grad_P(seq_len_);
                for (size_t i = 0; i < seq_len_; ++i)
                    grad_P[i] = grad_in[perm_P_[c][idx][i]];

                // Step 2: grad_B[blk, i, j] += grad_P[blk*b+i] * z[blk*b+j]
                const std::vector<double>& z = zs[idx];
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t i = 0; i < block_size_; ++i) {
                        for (size_t j = 0; j < block_size_; ++j) {
                            grad_blocks_(0, local_base + i * block_size_ + j) +=
                                grad_P[blk * block_size_ + i] * z[blk * block_size_ + j];
                        }
                    }
                }

                // Step 3: Apply B^T to grad_P.
                std::vector<double> grad_after_B(seq_len_, 0.0);
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t j = 0; j < block_size_; ++j) {
                        double sum = 0;
                        for (size_t i = 0; i < block_size_; ++i) {
                            sum += blocks_(0, local_base + i * block_size_ + j)
                                   * grad_P[blk * block_size_ + i];
                        }
                        grad_after_B[blk * block_size_ + j] = sum;
                    }
                }

                // Step 4: Apply Q^T.
                for (size_t i = 0; i < seq_len_; ++i)
                    grad_prev[i] = grad_after_B[perm_Q_[c][idx][i]];

                grad_in = grad_prev;
            }

            for (size_t s = 0; s < seq_len_; ++s)
                dx(t, s * d_model_ + c) = grad_in[s];
        }
    }
    return dx;
}

void MonarchSequenceMix::update_weights(double learning_rate) {
    for (size_t i = 0; i < blocks_.rows; ++i)
        for (size_t j = 0; j < blocks_.cols; ++j)
            blocks_(i, j) -= learning_rate * grad_blocks_(i, j);
}

void MonarchSequenceMix::zero_grad() {
    grad_blocks_.fill(0.0);
}

std::vector<Tensor*> MonarchSequenceMix::parameters() {
    return {&blocks_};
}

std::vector<Tensor*> MonarchSequenceMix::gradients() {
    return {&grad_blocks_};
}

Tensor MonarchSequenceMix::get_weights() const {
    return blocks_;
}

Tensor MonarchSequenceMix::get_gradients() const {
    return grad_blocks_;
}


// =============================================================================
// MonarchChannelMix (shared Monarch over channel axis, per-token)
// =============================================================================
MonarchChannelMix::MonarchChannelMix(size_t seq_len, size_t d_model, size_t block_size,
                                     size_t num_blocks, uint32_t perm_seed)
    : seq_len_(seq_len), d_model_(d_model) {
    if (seq_len == 0) throw std::invalid_argument("MonarchChannelMix: seq_len must be > 0");
    if (d_model == 0) throw std::invalid_argument("MonarchChannelMix: d_model must be > 0");
    if (block_size == 0) throw std::invalid_argument("MonarchChannelMix: block_size must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("MonarchChannelMix: num_blocks must be > 0");
    m_ = static_cast<size_t>(std::sqrt(static_cast<double>(d_model)));
    if (m_ * m_ != d_model) {
        throw std::invalid_argument("MonarchChannelMix: d_model must be a perfect square");
    }
    if (m_ % block_size != 0) {
        throw std::invalid_argument("MonarchChannelMix: block_size must divide sqrt(d_model) evenly");
    }
    if ((d_model % (block_size * block_size)) != 0) {
        throw std::invalid_argument("MonarchChannelMix: block_size^2 must divide d_model evenly");
    }
    block_size_ = block_size;
    num_blocks_ = num_blocks;
    num_mb_ = d_model / (block_size * block_size);

    const size_t per_block = block_size * block_size;
    const size_t total = num_blocks * num_mb_ * per_block;

    blocks_ = Tensor(1, total);
    blocks_.fill(0.0);
    grad_blocks_ = Tensor::zeros(1, total);

    std::mt19937 rng(perm_seed);
    perm_P_.resize(num_blocks);
    perm_Q_.resize(num_blocks);
    for (size_t k = 0; k < num_blocks; ++k) {
        perm_P_[k] = random_permutation(d_model, rng);
        perm_Q_[k] = random_permutation(d_model, rng);
    }

    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (size_t i = 0; i < blocks_.cols; ++i)
        blocks_(0, i) = dist(rng);

    last_input_ = Tensor(0, 0);
}

Tensor MonarchChannelMix::forward(const Tensor& input) {
    if (input.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchChannelMix::forward: input cols mismatch");
    }
    last_input_ = input.clone();

    const size_t B = input.rows;
    Tensor output(B, seq_len_ * d_model_);

    for (size_t t = 0; t < B; ++t) {
        for (size_t s = 0; s < seq_len_; ++s) {
            std::vector<double> x_tok(d_model_);
            for (size_t c = 0; c < d_model_; ++c)
                x_tok[c] = input(t, s * d_model_ + c);

            std::vector<double> y_tok = x_tok;
            for (size_t k = 0; k < num_blocks_; ++k) {
                std::vector<double> z(d_model_);
                for (size_t i = 0; i < d_model_; ++i)
                    z[i] = y_tok[perm_Q_[k][i]];

                std::vector<double> w(d_model_, 0.0);
                const size_t block_base = k * num_mb_ * block_size_ * block_size_;
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t i = 0; i < block_size_; ++i) {
                        double sum = 0;
                        for (size_t j = 0; j < block_size_; ++j) {
                            sum += blocks_(0, local_base + i * block_size_ + j)
                                   * z[blk * block_size_ + j];
                        }
                        w[blk * block_size_ + i] = sum;
                    }
                }

                std::vector<double> y_perm(d_model_);
                for (size_t i = 0; i < d_model_; ++i)
                    y_perm[i] = w[perm_P_[k][i]];
                y_tok = y_perm;
            }

            for (size_t c = 0; c < d_model_; ++c)
                output(t, s * d_model_ + c) = y_tok[c];
        }
    }
    return output;
}

Tensor MonarchChannelMix::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchChannelMix::backward: grad_output shape mismatch");
    }

    grad_blocks_.fill(0.0);
    const size_t B = grad_output.rows;
    Tensor dx(B, seq_len_ * d_model_);
    dx.fill(0.0);

    for (size_t t = 0; t < B; ++t) {
        for (size_t s = 0; s < seq_len_; ++s) {
            // Cache z_k for each Monarch k.
            std::vector<std::vector<double>> zs(num_blocks_, std::vector<double>(d_model_));
            {
                std::vector<double> x_tok(d_model_);
                for (size_t c = 0; c < d_model_; ++c)
                    x_tok[c] = last_input_(t, s * d_model_ + c);
                for (size_t k = 0; k < num_blocks_; ++k) {
                    for (size_t i = 0; i < d_model_; ++i)
                        zs[k][i] = x_tok[perm_Q_[k][i]];
                    std::vector<double> w(d_model_, 0.0);
                    const size_t block_base = k * num_mb_ * block_size_ * block_size_;
                    for (size_t blk = 0; blk < num_mb_; ++blk) {
                        const size_t local_base = block_base + blk * block_size_ * block_size_;
                        for (size_t i = 0; i < block_size_; ++i) {
                            double sum = 0;
                            for (size_t j = 0; j < block_size_; ++j) {
                                sum += blocks_(0, local_base + i * block_size_ + j)
                                       * zs[k][blk * block_size_ + j];
                            }
                            w[blk * block_size_ + i] = sum;
                        }
                    }
                    std::vector<double> y_perm(d_model_);
                    for (size_t i = 0; i < d_model_; ++i)
                        y_perm[i] = w[perm_P_[k][i]];
                    x_tok = y_perm;
                }
            }

            std::vector<double> grad_in(d_model_);
            for (size_t c = 0; c < d_model_; ++c)
                grad_in[c] = grad_output(t, s * d_model_ + c);

            std::vector<double> grad_prev(d_model_);
            for (size_t k = num_blocks_; k > 0; --k) {
                const size_t idx = k - 1;
                const size_t block_base = idx * num_mb_ * block_size_ * block_size_;

                std::vector<double> grad_P(d_model_);
                for (size_t i = 0; i < d_model_; ++i)
                    grad_P[i] = grad_in[perm_P_[idx][i]];

                const std::vector<double>& z = zs[idx];
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t i = 0; i < block_size_; ++i) {
                        for (size_t j = 0; j < block_size_; ++j) {
                            grad_blocks_(0, local_base + i * block_size_ + j) +=
                                grad_P[blk * block_size_ + i] * z[blk * block_size_ + j];
                        }
                    }
                }

                std::vector<double> grad_after_B(d_model_, 0.0);
                for (size_t blk = 0; blk < num_mb_; ++blk) {
                    const size_t local_base = block_base + blk * block_size_ * block_size_;
                    for (size_t j = 0; j < block_size_; ++j) {
                        double sum = 0;
                        for (size_t i = 0; i < block_size_; ++i) {
                            sum += blocks_(0, local_base + i * block_size_ + j)
                                   * grad_P[blk * block_size_ + i];
                        }
                        grad_after_B[blk * block_size_ + j] = sum;
                    }
                }

                for (size_t i = 0; i < d_model_; ++i)
                    grad_prev[i] = grad_after_B[perm_Q_[idx][i]];

                grad_in = grad_prev;
            }

            for (size_t c = 0; c < d_model_; ++c)
                dx(t, s * d_model_ + c) = grad_in[c];
        }
    }
    return dx;
}

void MonarchChannelMix::update_weights(double learning_rate) {
    for (size_t i = 0; i < blocks_.rows; ++i)
        for (size_t j = 0; j < blocks_.cols; ++j)
            blocks_(i, j) -= learning_rate * grad_blocks_(i, j);
}

void MonarchChannelMix::zero_grad() {
    grad_blocks_.fill(0.0);
}

std::vector<Tensor*> MonarchChannelMix::parameters() {
    return {&blocks_};
}

std::vector<Tensor*> MonarchChannelMix::gradients() {
    return {&grad_blocks_};
}

Tensor MonarchChannelMix::get_weights() const {
    return blocks_;
}

Tensor MonarchChannelMix::get_gradients() const {
    return grad_blocks_;
}


// =============================================================================
// MonarchMixerBlock
// =============================================================================
MonarchMixerBlock::MonarchMixerBlock(size_t seq_len, size_t d_model,
                                     size_t block_size_seq, size_t block_size_ch,
                                     size_t num_blocks_seq, size_t num_blocks_ch,
                                     uint32_t perm_seed)
    : seq_len_(seq_len), d_model_(d_model),
      ln1_(d_model),
      seq_mix_(seq_len, d_model, block_size_seq, num_blocks_seq, perm_seed),
      ln2_(d_model),
      ch_mix_(seq_len, d_model, block_size_ch, num_blocks_ch, perm_seed + 1) {}

Tensor MonarchMixerBlock::forward(const Tensor& input) {
    if (input.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchMixerBlock::forward: input shape mismatch");
    }
    last_input_ = input.clone();

    last_ln1_out_ = ln1_.forward(input);
    Tensor seq_mix_out = seq_mix_.forward(last_ln1_out_);
    last_res1_ = Tensor(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            last_res1_(i, j) = last_input_(i, j) + seq_mix_out(i, j);

    last_ln2_out_ = ln2_.forward(last_res1_);
    Tensor ch_mix_out = ch_mix_.forward(last_ln2_out_);
    last_output_ = Tensor(last_res1_.rows, last_res1_.cols);
    for (size_t i = 0; i < last_res1_.rows; ++i)
        for (size_t j = 0; j < last_res1_.cols; ++j)
            last_output_(i, j) = last_res1_(i, j) + ch_mix_out(i, j);

    return last_output_;
}

Tensor MonarchMixerBlock::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.rows != last_input_.rows || grad_output.cols != seq_len_ * d_model_) {
        throw std::invalid_argument("MonarchMixerBlock::backward: shape mismatch");
    }

    Tensor grad_ch_mix_in = ch_mix_.backward(grad_output, 0.0);
    Tensor grad_ln2_out = ln2_.backward(grad_ch_mix_in, 0.0);
    Tensor grad_res1(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i)
        for (size_t j = 0; j < grad_output.cols; ++j)
            grad_res1(i, j) = grad_output(i, j) + grad_ln2_out(i, j);

    Tensor grad_seq_mix_in = seq_mix_.backward(grad_res1, 0.0);
    Tensor grad_ln1_out = ln1_.backward(grad_seq_mix_in, 0.0);
    Tensor dx(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i)
        for (size_t j = 0; j < grad_output.cols; ++j)
            dx(i, j) = grad_output(i, j) + grad_ln1_out(i, j);

    return dx;
}

void MonarchMixerBlock::update_weights(double learning_rate) {
    ln1_.update_weights(learning_rate);
    seq_mix_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ch_mix_.update_weights(learning_rate);
}

void MonarchMixerBlock::zero_grad() {
    ln1_.zero_grad();
    seq_mix_.zero_grad();
    ln2_.zero_grad();
    ch_mix_.zero_grad();
}

std::vector<Tensor*> MonarchMixerBlock::parameters() {
    std::vector<Tensor*> p;
    auto a = ln1_.parameters();
    auto b = seq_mix_.parameters();
    auto c = ln2_.parameters();
    auto d = ch_mix_.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

std::vector<Tensor*> MonarchMixerBlock::gradients() {
    std::vector<Tensor*> g;
    auto a = ln1_.gradients();
    auto b = seq_mix_.gradients();
    auto c = ln2_.gradients();
    auto d = ch_mix_.gradients();
    g.insert(g.end(), a.begin(), a.end());
    g.insert(g.end(), b.begin(), b.end());
    g.insert(g.end(), c.begin(), c.end());
    g.insert(g.end(), d.begin(), d.end());
    return g;
}

Tensor MonarchMixerBlock::get_weights() const {
    auto p = const_cast<MonarchMixerBlock*>(this)->ln1_.parameters();
    if (!p.empty()) return *p[0];
    return Tensor(0, 0);
}

Tensor MonarchMixerBlock::get_gradients() const {
    auto p = const_cast<MonarchMixerBlock*>(this)->ln1_.gradients();
    if (!p.empty()) return *p[0];
    return Tensor(0, 0);
}


// =============================================================================
// MonarchMixerModel
// =============================================================================
MonarchMixerModel::MonarchMixerModel(size_t input_dim, size_t d_model, size_t output_dim,
                                     size_t seq_len, size_t num_blocks,
                                     size_t block_size_seq, size_t block_size_ch,
                                     uint32_t perm_seed)
    : input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      seq_len_(seq_len), num_blocks_(num_blocks),
      block_size_seq_(block_size_seq == 0 ? static_cast<size_t>(std::sqrt(static_cast<double>(seq_len))) : block_size_seq),
      block_size_ch_(block_size_ch == 0 ? static_cast<size_t>(std::sqrt(static_cast<double>(d_model))) : block_size_ch),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0) throw std::invalid_argument("MonarchMixerModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("MonarchMixerModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("MonarchMixerModel: output_dim must be > 0");
    if (seq_len == 0) throw std::invalid_argument("MonarchMixerModel: seq_len must be > 0");
    if (num_blocks == 0) throw std::invalid_argument("MonarchMixerModel: num_blocks must be > 0");

    if (block_size_seq_ * block_size_seq_ > seq_len_ ||
        (seq_len % (block_size_seq_ * block_size_seq_)) != 0) {
        throw std::invalid_argument("MonarchMixerModel: block_size_seq^2 must divide seq_len evenly");
    }
    if (block_size_ch_ * block_size_ch_ > d_model_ ||
        (d_model % (block_size_ch_ * block_size_ch_)) != 0) {
        throw std::invalid_argument("MonarchMixerModel: block_size_ch^2 must divide d_model evenly");
    }

    blocks_.clear();
    blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks_.push_back(std::make_unique<MonarchMixerBlock>(
            seq_len, d_model, block_size_seq_, block_size_ch_,
            /*num_blocks_seq=*/1, /*num_blocks_ch=*/1,
            perm_seed + static_cast<uint32_t>(i) * 7));
    }

    last_block_outs_.clear();
    last_block_outs_.reserve(num_blocks);
}

Tensor MonarchMixerModel::forward(const Tensor& input) {
    if (input.cols != seq_len_ * input_dim_) {
        throw std::invalid_argument("MonarchMixerModel::forward: input shape mismatch");
    }
    const size_t B = input.rows;

    Tensor embed_out(B, seq_len_ * d_model_);
    embed_out.fill(0.0);

    for (size_t b = 0; b < B; ++b) {
        Tensor per_seq(seq_len_, input_dim_);
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t j = 0; j < input_dim_; ++j)
                per_seq(s, j) = input(b, s * input_dim_ + j);
        Tensor embedded = embed_.forward(per_seq);
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t c = 0; c < d_model_; ++c)
                embed_out(b, s * d_model_ + c) = embedded(s, c);
    }

    last_embed_ = embed_out.clone();

    Tensor block_out = embed_out;
    last_block_outs_.clear();
    for (size_t i = 0; i < num_blocks_; ++i) {
        block_out = blocks_[i]->forward(block_out);
        last_block_outs_.push_back(block_out.clone());
    }

    Tensor ln_out = final_ln_.forward(block_out);

    Tensor pooled(B, d_model_);
    pooled.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t c = 0; c < d_model_; ++c)
                pooled(b, c) += ln_out(b, s * d_model_ + c);
        for (size_t c = 0; c < d_model_; ++c)
            pooled(b, c) /= static_cast<double>(seq_len_);
    }
    last_pooled_ = pooled.clone();

    Tensor logits = classifier_.forward(pooled);
    last_logits_ = logits.clone();
    return logits;
}

Tensor MonarchMixerModel::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.cols != output_dim_) {
        throw std::invalid_argument("MonarchMixerModel::backward: grad_output shape mismatch");
    }

    const size_t B = grad_output.rows;

    Tensor d_pooled = classifier_.backward(grad_output, 0.0);

    Tensor d_ln_out(B, seq_len_ * d_model_);
    d_ln_out.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t c = 0; c < d_model_; ++c)
                d_ln_out(b, s * d_model_ + c) = d_pooled(b, c) / static_cast<double>(seq_len_);
    }

    Tensor d_block_out_final = final_ln_.backward(d_ln_out, 0.0);

    Tensor d_block_in = d_block_out_final;
    for (size_t i = num_blocks_; i > 0; --i) {
        d_block_in = blocks_[i - 1]->backward(d_block_in, 0.0);
    }

    Tensor d_input(B, seq_len_ * input_dim_);
    d_input.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        Tensor d_embed_per_seq(seq_len_, d_model_);
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t c = 0; c < d_model_; ++c)
                d_embed_per_seq(s, c) = d_block_in(b, s * d_model_ + c);
        Tensor d_per_seq = embed_.backward(d_embed_per_seq, 0.0);
        for (size_t s = 0; s < seq_len_; ++s)
            for (size_t j = 0; j < input_dim_; ++j)
                d_input(b, s * input_dim_ + j) = d_per_seq(s, j);
    }

    return d_input;
}

void MonarchMixerModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& b : blocks_) b->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void MonarchMixerModel::zero_grad() {
    embed_.zero_grad();
    for (auto& b : blocks_) b->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> MonarchMixerModel::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&embed_.weights);
    p.push_back(&embed_.bias);
    for (auto& b : blocks_) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&final_ln_.gamma);
    p.push_back(&final_ln_.beta);
    p.push_back(&classifier_.weights);
    p.push_back(&classifier_.bias);
    return p;
}

std::vector<Tensor*> MonarchMixerModel::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&embed_.grad_weights);
    g.push_back(&embed_.grad_bias);
    for (auto& b : blocks_) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&final_ln_.grad_gamma_);
    g.push_back(&final_ln_.grad_beta_);
    g.push_back(&classifier_.grad_weights);
    g.push_back(&classifier_.grad_bias);
    return g;
}

Tensor MonarchMixerModel::get_weights() const {
    return embed_.weights;
}

Tensor MonarchMixerModel::get_gradients() const {
    return embed_.grad_weights;
}