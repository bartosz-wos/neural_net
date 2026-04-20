#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../core/layer.h"

// Learnable embedding table: maps token IDs to dense vectors
class Embedding : public Layer {
public:
    int vocab_size;
    int dim;
    Tensor table;     // (vocab_size, dim)
    Tensor grad_table; // accumulated gradients

    Embedding(int vocab_size, int dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return table; }
    Tensor get_gradients() const override { return grad_table; }
    std::vector<Tensor*> parameters() override { return {&table}; }
    std::vector<Tensor*> gradients() override { return {&grad_table}; }
    void zero_grad() override;

private:
    std::vector<std::vector<int>> cache; // cached token sequences (for backward)
};

#endif