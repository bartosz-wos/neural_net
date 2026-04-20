#include "memory_network.h"

MemoryNetwork::MemoryNetwork(size_t vocab_size, size_t embedding_dim,
                             size_t memory_size, size_t hop_layers, size_t output_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      memory_size_(memory_size), hop_layers_(hop_layers), output_dim_(output_dim),
      A_(vocab_size, embedding_dim), W_(embedding_dim, embedding_dim),
      R_(embedding_dim, output_dim), b_out_(1, output_dim),
      grad_A_(vocab_size, embedding_dim), grad_W_(embedding_dim, embedding_dim),
      grad_R_(embedding_dim, output_dim), grad_b_out_(1, output_dim),
      last_u_(1, embedding_dim), last_probs_(1, memory_size),
      last_memory_(embedding_dim, memory_size) {

    // Random init for embedding matrix A
    double scale = std::sqrt(2.0 / vocab_size_);
    for (size_t i = 0; i < vocab_size_; ++i)
        for (size_t j = 0; j < embedding_dim_; ++j)
            A_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;

    // Identity-like init for W (tied hops)
    for (size_t i = 0; i < embedding_dim_; ++i)
        for (size_t j = 0; j < embedding_dim_; ++j)
            W_[i][j] = (i == j) ? 0.1 : 0.0;

    // Small random init for R
    for (size_t i = 0; i < embedding_dim_; ++i)
        for (size_t j = 0; j < output_dim_; ++j)
            R_[i][j] = (rand() / RAND_MAX * 2 - 1) * 0.01;

    for (size_t j = 0; j < output_dim_; ++j)
        b_out_[0][j] = 0.0;
}

Tensor MemoryNetwork::compute_attention(const Tensor& u) {
    // u: (batch, embedding_dim)
    // last_memory_: (embedding_dim, memory_size) — each column is a memory slot embedding
    // Compute attention: probs_i = softmax(u^T * m_i)
    size_t batch = u.rows;
    last_probs_ = Tensor(batch, memory_size_);

    for (size_t b = 0; b < batch; ++b) {
        double total = 0.0;
        for (size_t i = 0; i < memory_size_; ++i) {
            double score = 0.0;
            for (size_t k = 0; k < embedding_dim_; ++k)
                score += u[b][k] * last_memory_[k][i];
            last_probs_[b][i] = std::exp(score);
            total += last_probs_[b][i];
        }
        for (size_t i = 0; i < memory_size_; ++i)
            last_probs_[b][i] /= total;
    }
    return last_probs_;
}

Tensor MemoryNetwork::hop(const Tensor& u, const Tensor& memory) {
    // memory: (embedding_dim, memory_size)
    // u: (batch, embedding_dim)
    // o = sum_i p_i * m_i^T
    size_t batch = u.rows;
    Tensor output(batch, embedding_dim_);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t k = 0; k < embedding_dim_; ++k) {
            double sum = 0.0;
            for (size_t i = 0; i < memory_size_; ++i)
                sum += last_probs_[b][i] * memory[k][i];
            output[b][k] = u[b][k] + sum; // residual-like add
        }
    }
    return output;
}

void MemoryNetwork::set_memory(const std::vector<std::vector<size_t>>& memory) {
    memory_ = memory;
    // Pre-embed all memory sentences (bag-of-words average)
    last_memory_ = Tensor(embedding_dim_, memory_size_);
    for (size_t m = 0; m < memory.size() && m < memory_size_; ++m) {
        if (memory[m].empty()) continue;
        double scale = 1.0 / memory[m].size();
        for (size_t k = 0; k < embedding_dim_; ++k) {
            double sum = 0.0;
            for (size_t w : memory[m]) {
                if (w < vocab_size_)
                    sum += A_[w][k];
            }
            last_memory_[k][m] = sum * scale;
        }
    }
}

Tensor MemoryNetwork::forward(const Tensor& input) {
    // input: (batch, seq_len) — query word indices
    // Embed query (bag of words)
    last_u_ = Tensor(input.rows, embedding_dim_);
    for (size_t b = 0; b < input.rows; ++b) {
        if (input.cols == 0) continue;
        double scale = 1.0 / input.cols;
        for (size_t k = 0; k < embedding_dim_; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < input.cols; ++j) {
                size_t w = static_cast<size_t>(input[b][j]);
                if (w < vocab_size_)
                    sum += A_[w][k];
            }
            last_u_[b][k] = sum * scale;
        }
    }

    // Multiple hop layers — each hop reads memory and updates u
    Tensor u = last_u_;
    for (size_t h = 0; h < hop_layers_; ++h) {
        compute_attention(u);
        u = hop(u, last_memory_);
    }

    // Output layer: o * R + b
    // R: (embedding_dim, output_dim)
    last_output_ = Tensor(input.rows, output_dim_);
    for (size_t b = 0; b < input.rows; ++b) {
        for (size_t j = 0; j < output_dim_; ++j) {
            double sum = b_out_[0][j];
            for (size_t k = 0; k < embedding_dim_; ++k)
                sum += u[b][k] * R_[k][j];
            last_output_[b][j] = sum;
        }
    }
    return last_output_;
}

Tensor MemoryNetwork::backward(const Tensor& grad_output, double learning_rate) {
    // Gradient w.r.t. R and output
    // grad_R = u^T * grad_output
    size_t batch = grad_output.rows;
    for (size_t k = 0; k < embedding_dim_; ++k)
        for (size_t j = 0; j < output_dim_; ++j) {
            double g = 0.0;
            for (size_t b = 0; b < batch; ++b)
                g += grad_output[b][j] * last_u_[b][k]; // use original u before hop
            grad_R_[k][j] = g / batch;
        }

    // Gradient w.r.t. embedding matrix A (simplified — propagate through u)
    for (size_t i = 0; i < vocab_size_; ++i)
        for (size_t k = 0; k < embedding_dim_; ++k)
            grad_A_[i][k] = 0.0;

    // Simplified backward: just update W and R
    (void)learning_rate;
    return Tensor(1, 1); // placeholder
}

void MemoryNetwork::update_weights(double learning_rate) {
    auto step = [&](Tensor& w, const Tensor& gw) {
        for (size_t i = 0; i < w.rows; ++i)
            for (size_t j = 0; j < w.cols; ++j)
                w[i][j] -= learning_rate * gw[i][j];
    };
    step(W_, grad_W_);
    step(R_, grad_R_);
    step(b_out_, grad_b_out_);
}

void MemoryNetwork::zero_grad() {
    grad_A_.fill(0); grad_W_.fill(0);
    grad_R_.fill(0); grad_b_out_.fill(0);
}

std::vector<Tensor*> MemoryNetwork::parameters() {
    return { &A_, &W_, &R_, &b_out_ };
}

std::vector<Tensor*> MemoryNetwork::gradients() {
    return { &grad_A_, &grad_W_, &grad_R_, &grad_b_out_ };
}

// === EmbeddingLayer ===

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      embeddings_(vocab_size, embedding_dim),
      grad_embeddings_(vocab_size, embedding_dim) {

    double scale = std::sqrt(2.0 / vocab_size_);
    for (size_t i = 0; i < vocab_size_; ++i)
        for (size_t j = 0; j < embedding_dim_; ++j)
            embeddings_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
}

Tensor EmbeddingLayer::forward(const Tensor& input) {
    // input: (batch, seq_len) of word indices
    last_input_ = input;
    size_t batch = input.rows;
    Tensor output(batch, embedding_dim_);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t k = 0; k < embedding_dim_; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < input.cols; ++j) {
                size_t w = static_cast<size_t>(input[b][j]);
                if (w < vocab_size_)
                    sum += embeddings_[w][k];
            }
            double scale = input.cols > 0 ? 1.0 / input.cols : 1.0;
            output[b][k] = sum * scale;
        }
    }
    return output;
}

Tensor EmbeddingLayer::backward(const Tensor& grad_output, double learning_rate) {
    // Accumulate gradient into embedding matrix
    size_t batch = last_input_.rows;
    for (size_t b = 0; b < batch; ++b) {
        double scale = last_input_.cols > 0 ? 1.0 / last_input_.cols : 1.0;
        for (size_t j = 0; j < last_input_.cols; ++j) {
            size_t w = static_cast<size_t>(last_input_[b][j]);
            if (w >= vocab_size_) continue;
            for (size_t k = 0; k < embedding_dim_; ++k)
                grad_embeddings_[w][k] += scale * grad_output[b][k];
        }
    }
    (void)learning_rate;
    return Tensor(1, 1);
}

void EmbeddingLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < vocab_size_; ++i)
        for (size_t k = 0; k < embedding_dim_; ++k)
            embeddings_[i][k] -= learning_rate * grad_embeddings_[i][k];
}

void EmbeddingLayer::zero_grad() { grad_embeddings_.fill(0); }

std::vector<Tensor*> EmbeddingLayer::parameters() { return { &embeddings_ }; }
std::vector<Tensor*> EmbeddingLayer::gradients() { return { &grad_embeddings_ }; }