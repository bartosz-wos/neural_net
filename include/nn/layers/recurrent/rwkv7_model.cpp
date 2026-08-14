#include "rwkv7_model.h"
#include <stdexcept>

// ============================================================================
// RWKV7Model implementation
// ============================================================================

RWKV7Model::RWKV7Model(size_t input_dim, size_t d, size_t output_dim,
                       size_t num_heads, size_t num_layers)
    : input_dim_(input_dim), d_(d), output_dim_(output_dim),
      num_layers_(num_layers), num_heads_(num_heads),
      embed(input_dim, d),
      classifier(d, output_dim)
{
    if (input_dim == 0 || d == 0 || output_dim == 0 || num_layers == 0) {
        throw std::invalid_argument("RWKV7Model: all dims must be > 0");
    }
    cells.reserve(num_layers);
    for (size_t l = 0; l < num_layers_; ++l) {
        cells.push_back(std::make_unique<RWKV7TimeMix>(d, num_heads));
    }
}

Tensor RWKV7Model::forward(const Tensor& input) {
    if (input.cols != input_dim_) {
        throw std::invalid_argument("RWKV7Model: input.cols must equal input_dim");
    }
    last_input_ = input.clone();

    // 1. embed
    Tensor x = embed.forward(input);   // (T, d)

    // 2. stack of cells
    cell_outputs_.clear();
    cell_outputs_.push_back(x);
    for (size_t l = 0; l < num_layers_; ++l) {
        x = cells[l]->forward(x);  // (T, d)
        cell_outputs_.push_back(x);
    }

    // 3. last-step extract: take the LAST row of the final cell output.
    Tensor last(1, d_);
    for (size_t j = 0; j < d_; ++j) {
        last(0, j) = cell_outputs_.back()(cell_outputs_.back().rows - 1, j);
    }

    // 4. classifier
    return classifier.forward(last);  // (1, output_dim)
}

Tensor RWKV7Model::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != 1 || grad_output.cols != output_dim_) {
        throw std::invalid_argument("RWKV7Model: grad_output must be (1, output_dim)");
    }

    // Backward through classifier → last (1, d)
    Tensor grad_last = classifier.backward(grad_output, 0.0);

    // Spread grad_last across the last row only (rest is zero)
    Tensor grad_cell_outputs_last = Tensor(cell_outputs_.back().rows, d_);
    grad_cell_outputs_last.fill(0.0);
    for (size_t j = 0; j < d_; ++j) {
        grad_cell_outputs_last(grad_cell_outputs_last.rows - 1, j) = grad_last(0, j);
    }

    // Backward through cells in reverse
    Tensor grad_x = grad_cell_outputs_last;
    for (size_t l = num_layers_; l > 0; --l) {
        grad_x = cells[l - 1]->backward(grad_x, 0.0);  // (T, d)
    }

    // Backward through embed → grad_input (T, input_dim)
    Tensor grad_input = embed.backward(grad_x, 0.0);

    return grad_input;
}

void RWKV7Model::update_weights(double learning_rate) {
    embed.update_weights(learning_rate);
    for (auto& c : cells) c->update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}

void RWKV7Model::zero_grad() {
    embed.zero_grad();
    for (auto& c : cells) c->zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> RWKV7Model::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&embed.weights);
    p.push_back(&embed.bias);
    for (auto& c : cells) {
        auto cp = c->parameters();
        for (Tensor* t : cp) p.push_back(t);
    }
    p.push_back(&classifier.weights);
    p.push_back(&classifier.bias);
    return p;
}

std::vector<Tensor*> RWKV7Model::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&embed.grad_weights);
    g.push_back(&embed.grad_bias);
    for (auto& c : cells) {
        auto cg = c->gradients();
        for (Tensor* t : cg) g.push_back(t);
    }
    g.push_back(&classifier.grad_weights);
    g.push_back(&classifier.grad_bias);
    return g;
}