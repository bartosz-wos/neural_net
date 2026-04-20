#include "embedding.h"
#include <random>
#include <cmath>

Embedding::Embedding(int vocab, int dim)
    : vocab_size(vocab), dim(dim),
      table(vocab, dim), grad_table(vocab, dim)
{
    // Xavier init for embeddings
    double scale = std::sqrt(2.0 / dim);
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);
    for (int i = 0; i < vocab; ++i)
        for (int j = 0; j < dim; ++j)
            table[i][j] = dis(gen);
    grad_table.fill(0.0);
}

Tensor Embedding::forward(const Tensor& input) {
    // input: (batch, seq_len) — token IDs as integers
    int batch = input.rows;
    int seq_len = input.cols;
    cache.clear();
    cache.resize(batch);

    Tensor output(batch, seq_len * dim);
    for (int b = 0; b < batch; ++b) {
        cache[b].resize(seq_len);
        for (int t = 0; t < seq_len; ++t) {
            int token_id = static_cast<int>(input[b][t]);
            if (token_id < 0) token_id = 0;
            if (token_id >= vocab_size) token_id = 0;
            cache[b][t] = token_id;

            // Copy embedding row into output slice
            for (int d = 0; d < dim; ++d) {
                output[b][t * dim + d] = table[token_id][d];
            }
        }
    }
    return output;
}

Tensor Embedding::backward(const Tensor& grad_output, double /* learning_rate */) {
    int batch = (int)cache.size();
    int seq_len = cache.empty() ? 0 : (int)cache[0].size();
    grad_table.fill(0.0);

    // grad_output: (batch, seq_len * dim)
    // Each gradient w.r.t. embedding is averaged over occurrences
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int token_id = cache[b][t];
            for (int d = 0; d < dim; ++d) {
                grad_table[token_id][d] += grad_output[b][t * dim + d];
            }
        }
    }

    // grad_input: same shape as input (IDs don't flow gradient)
    return Tensor(grad_output.rows, grad_output.cols);
}

void Embedding::update_weights(double learning_rate) {
    for (int i = 0; i < vocab_size; ++i)
        for (int j = 0; j < dim; ++j)
            table[i][j] -= learning_rate * grad_table[i][j];
}

void Embedding::zero_grad() {
    grad_table.fill(0.0);
}