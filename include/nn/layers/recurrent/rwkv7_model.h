#ifndef RWKV7_MODEL_H
#define RWKV7_MODEL_H

#include "rwkv7.h"
#include <vector>
#include <memory>

// ============================================================================
// RWKV7Model — Stack of RWKV7TimeMix cells + classifier
//   Peng et al. 2025 "RWKV-7 Goose with Expressive Dynamic State Evolution"
//   https://arxiv.org/abs/2503.14456
//
// A model that takes (T, input_dim) → (1, output_dim) via:
//   1. embed: (T, input_dim) → (T, d)            Dense(input_dim, d)
//   2. cell[0..L-1].forward: (T, d) → (T, d)     each cell takes the prev output
//   3. last-step extract: (T, d) → (1, d)
//   4. classifier: (1, d) → (1, output_dim)      Dense(d, output_dim)
//
// v1 simplification: no channel-mixing (FFN) blocks; just the RWKV-7 time-mix
// per layer. This exercises the generalized delta rule cleanly while keeping
// the test surface tractable.
// ============================================================================

class RWKV7Model : public Layer {
public:
    RWKV7Model(size_t input_dim, size_t d, size_t output_dim,
               size_t num_heads = 1, size_t num_layers = 1);

    // (T, input_dim) → (1, output_dim)
    Tensor forward(const Tensor& input) override;
    // grad_output: (1, output_dim), returns grad_input: (T, input_dim)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed.weights; }
    Tensor get_gradients() const override { return embed.grad_weights; }
    std::string name() const override { return "RWKV7Model"; }

    size_t input_dim() const { return input_dim_; }
    size_t d() const { return d_; }
    size_t output_dim() const { return output_dim_; }
    size_t num_layers() const { return num_layers_; }
    size_t num_heads() const { return num_heads_; }

    // ---- Public sub-modules ----
    size_t input_dim_;
    size_t d_;
    size_t output_dim_;
    size_t num_layers_;
    size_t num_heads_;
    Dense embed;             // (input_dim, d)
    std::vector<std::unique_ptr<RWKV7TimeMix>> cells;
    Dense classifier;        // (d, output_dim)

    // BPTT cache (last input + intermediate cell outputs)
    Tensor last_input_;      // (T, input_dim)
    std::vector<Tensor> cell_outputs_;  // cell_outputs_[l] is the (T, d) output of layer l
};

#endif // RWKV7_MODEL_H