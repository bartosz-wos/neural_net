#ifndef MEMORY_NETWORK_H
#define MEMORY_NETWORK_H

#include "../../core/layer.h"
#include <vector>
#include <string>

// MemN2N: End-to-end Memory Network for question answering.
// Embeds a knowledge base into memory slots, processes queries through hop layers,
// and produces an answer. Supports multiple hops and tied weights.
class MemoryNetwork : public Layer {
public:
    // vocab_size: size of word vocabulary
    // embedding_dim: dimension of word embeddings
    // memory_size: max number of memory slots
    // hop_layers: number of hop layers (each re-reads memory)
    // output_dim: answer vocabulary / label size
    MemoryNetwork(size_t vocab_size, size_t embedding_dim,
                  size_t memory_size, size_t hop_layers, size_t output_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    // Set the knowledge base (list of memory sentences)
    void set_memory(const std::vector<std::vector<size_t>>& memory);

    // Compute attention probabilities over memory given current u (query embedding)
    Tensor compute_attention(const Tensor& u);

private:
    size_t vocab_size_, embedding_dim_, memory_size_, hop_layers_, output_dim_;

    // Embedding matrix A: (vocab_size, embedding_dim)
    Tensor A_;
    // Hop weight matrix W: (embedding_dim, embedding_dim) — tied across hops
    Tensor W_;
    // Output weight matrix R: (embedding_dim, output_dim) — only on last hop
    Tensor R_;
    // Bias for output layer
    Tensor b_out_;

    // Gradients
    Tensor grad_A_, grad_W_, grad_R_, grad_b_out_;

    // Internal state
    Tensor last_u_;        // final query embedding after hops
    Tensor last_probs_;    // attention probabilities over memory
    Tensor last_memory_;   // embedded memory (embedding_dim, memory_size)
    Tensor last_output_;
    Tensor last_input_;     // store input for backward (batch, seq_len)
    std::vector<std::vector<size_t>> memory_;

    // Single hop: u = u + softmax(u^T * m_i) * m_i^T
    Tensor hop(const Tensor& u, const Tensor& memory);
};

// Simple embedding lookup layer (bag-of-words style sum of word embeddings)
class EmbeddingLayer : public Layer {
public:
    EmbeddingLayer(size_t vocab_size, size_t embedding_dim);
    // input: (batch, seq_len) of word indices
    // output: (batch, embedding_dim) — sum of word vectors (or mean)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void set_vocab_embedding(const Tensor& embedding) { embeddings_ = embedding; }

private:
    size_t vocab_size_, embedding_dim_;
    Tensor embeddings_;     // (vocab_size, embedding_dim) — learned
    Tensor grad_embeddings_;
    Tensor last_input_;      // store for backward
};

#endif