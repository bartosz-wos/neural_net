#include "ff_layer.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {
// PRNG for label sampling in FFNetwork::make_negative_random().
// Seeded with a fixed value for reproducibility in tests.
static std::mt19937 ff_rng_(424242);
} // namespace

// =============================================================================
// FFLayer
// =============================================================================

double FFLayer::sigmoid(double x) {
    // Numerically stable sigmoid
    if (x >= 0.0) {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(x);
        return e / (1.0 + e);
    }
}

FFLayer::FFLayer(size_t in_features, size_t out_features,
                 double threshold, double lr_ff)
    : W(static_cast<size_t>(out_features), static_cast<size_t>(in_features)),
      b(1, out_features),
      grad_W(static_cast<size_t>(out_features), static_cast<size_t>(in_features)),
      grad_b(1, out_features),
      last_input(0, 0),
      last_output(0, 0),
      threshold_(threshold),
      lr_ff_(lr_ff) {
    init_weights();
}

void FFLayer::init_weights() {
    // Standard normal init scaled by 1/sqrt(in). For an FF layer the
    // activation distribution is the key statistic — Xavier/Glorot init
    // keeps the post-ReLU activations well-conditioned at start.
    double scale = 1.0 / std::sqrt(static_cast<double>(W.cols));
    std::normal_distribution<double> dist(0.0, scale);
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] = dist(ff_rng_);
    }
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] = 0.0;
    }
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

Tensor FFLayer::forward(const Tensor& input) {
    // y = relu(W @ x + b)
    last_input = input.clone();
    const size_t B = input.rows;
    const size_t in_dim = W.cols;
    const size_t out_dim = W.rows;
    Tensor y(B, out_dim);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            double s = b(0, j);
            for (size_t k = 0; k < in_dim; ++k) {
                s += W(j, k) * input(i, k);
            }
            y(i, j) = (s > 0.0) ? s : 0.0;
        }
    }
    last_output = y.clone();
    return y;
}

std::pair<double, double> FFLayer::train_ff(const Tensor& pos_x,
                                              const Tensor& neg_x) {
    // Run positive forward
    Tensor pos_y = forward(pos_x);
    // Run negative forward
    Tensor neg_y = forward(neg_x);

    const size_t B_pos = pos_y.rows;
    const size_t B_neg = neg_y.rows;
    const size_t out_dim = pos_y.cols;

    // Compute goodnesses (per-sample mean of squared activations).
    double g_pos = 0.0;
    for (size_t i = 0; i < B_pos; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < out_dim; ++j) s += pos_y(i, j) * pos_y(i, j);
        g_pos += s / static_cast<double>(out_dim);
    }
    g_pos /= static_cast<double>(B_pos);

    double g_neg = 0.0;
    for (size_t i = 0; i < B_neg; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < out_dim; ++j) s += neg_y(i, j) * neg_y(i, j);
        g_neg += s / static_cast<double>(out_dim);
    }
    g_neg /= static_cast<double>(B_neg);

    // Local goodness gradients.
    // dL+/dg_pos = -σ(θ - g_pos) = -(1 - σ(g_pos - θ))
    // dL-/dg_neg = +σ(g_neg - θ)
    double sg_pos = sigmoid(g_pos - threshold_);    // σ(g_pos - θ)
    double sg_neg = sigmoid(g_neg - threshold_);    // σ(g_neg - θ)
    double dL_dg_pos = -(1.0 - sg_pos);            // = -σ(θ - g_pos)
    double dL_dg_neg = +sg_neg;                    // = +σ(g_neg - θ)

    // Now backpropagate through the per-sample squared-mean-of-activations
    // and ReLU. For each sample:
    //   g = (1/D) sum_j y_j^2
    //   d g / d h_j = (2/D) y_j  (h is pre-activation; y = relu(h))
    //   d g / d y_j = (2/D) y_j,  but y_j = relu(h_j), so
    //   d g / d h_j = (2/D) y_j * 1[h_j > 0]
    //
    // So dL/dh (per sample) = dL/dg * (2/D) * y .* relu_mask
    //
    // For the full batch:
    //   grad_h_pos[b, j] = dL_dg_pos * (2/D) * y_pos[b, j] * 1[y_pos[b, j] > 0]
    //   grad_h_neg[b, j] = dL_dg_neg * (2/D) * y_neg[b, j] * 1[y_neg[b, j] > 0]
    //
    // Then dL/dW = grad_h^T @ x  (summed over batch), dL/db = sum grad_h over batch.

    // Accumulate gradients over the batch.
    grad_W.fill(0.0);
    grad_b.fill(0.0);

    // Positive contributions
    for (size_t i = 0; i < B_pos; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            double yj = pos_y(i, j);
            double mask = (yj > 0.0) ? 1.0 : 0.0;
            double dh = dL_dg_pos * (2.0 / static_cast<double>(out_dim)) * yj * mask;
            for (size_t k = 0; k < W.cols; ++k) {
                grad_W(j, k) += dh * pos_x(i, k);
            }
            grad_b(0, j) += dh;
        }
    }
    // Negative contributions
    for (size_t i = 0; i < B_neg; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            double yj = neg_y(i, j);
            double mask = (yj > 0.0) ? 1.0 : 0.0;
            double dh = dL_dg_neg * (2.0 / static_cast<double>(out_dim)) * yj * mask;
            for (size_t k = 0; k < W.cols; ++k) {
                grad_W(j, k) += dh * neg_x(i, k);
            }
            grad_b(0, j) += dh;
        }
    }

    // Average the gradients over both batches' total size (so the lr is
    // batch-size invariant). This is a normalization choice — the paper
    // is loose on the convention; we use a symmetric average.
    double norm = 1.0 / static_cast<double>(B_pos + B_neg);
    for (size_t i = 0; i < grad_W.data.size(); ++i) grad_W.data[i] *= norm;
    for (size_t i = 0; i < grad_b.data.size(); ++i) grad_b.data[i] *= norm;

    // Apply the local FF update. FFLayer trains WITHOUT backprop, so we
    // apply the gradient in-place using lr_ff_.
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= lr_ff_ * grad_W.data[i];
    }
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] -= lr_ff_ * grad_b.data[i];
    }

    return std::make_pair(g_pos, g_neg);
}

Tensor FFLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // Standard chain rule: dx = W^T @ (grad_output .* relu_mask).
    //
    // This is NOT used during FF training — the layer is meant to learn via
    // train_ff(). Provided for compatibility / gradient-check sanity tests.
    const size_t B = grad_output.rows;
    const size_t out_dim = grad_output.cols;
    const size_t in_dim = W.cols;
    Tensor grad_input(B, in_dim);
    grad_input.fill(0.0);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            double mask = (last_output(i, j) > 0.0) ? 1.0 : 0.0;
            double g = grad_output(i, j) * mask;
            for (size_t k = 0; k < in_dim; ++k) {
                grad_input(i, k) += W(j, k) * g;
            }
        }
    }
    return grad_input;
}

void FFLayer::update_weights(double /*learning_rate*/) {
    // FF training is already applied inside train_ff(); this method is a
    // no-op for compatibility. If a user invokes Layer-style training with
    // an externally provided gradient (e.g. for a gradient check), we
    // apply it using lr_ff_ as the step size so the layer still updates.
    if (grad_W.data.empty()) return;
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= lr_ff_ * grad_W.data[i];
    }
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] -= lr_ff_ * grad_b.data[i];
    }
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

std::vector<Tensor*> FFLayer::parameters() {
    return {&W, &b};
}

std::vector<Tensor*> FFLayer::gradients() {
    return {&grad_W, &grad_b};
}

void FFLayer::zero_grad() {
    grad_W.fill(0.0);
    grad_b.fill(0.0);
}

// =============================================================================
// FFNetwork
// =============================================================================

FFNetwork::FFNetwork(size_t input_dim,
                     const std::vector<size_t>& hidden_dims,
                     size_t num_classes,
                     double threshold,
                     double lr_ff)
    : layers_(),
      hidden_dims_(hidden_dims),
      input_dim_(input_dim),
      num_classes_(num_classes),
      threshold_(threshold),
      lr_ff_(lr_ff) {
    if (hidden_dims.empty()) {
        // Degenerate case: a single hidden layer of size 1. We require at
        // least one layer; bail with a sensible default.
        hidden_dims_.push_back(std::max<size_t>(1, input_dim));
    }
    // Layer 0 takes (input_dim + num_classes) → hidden_dims[0]
    // Layer i takes hidden_dims[i-1] → hidden_dims[i]
    size_t prev_dim = input_dim + num_classes;
    for (size_t h : hidden_dims_) {
        layers_.emplace_back(std::make_unique<FFLayer>(prev_dim, h, threshold, lr_ff));
        prev_dim = h;
    }
}

Tensor FFNetwork::forward(const Tensor& input) {
    // Forward through all layers; return the final activation.
    Tensor x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

Tensor FFNetwork::make_positive(const Tensor& x, int label) const {
    // Build (x ⊕ one_hot(label)) with the one-hot as a ROW-APPEND? No —
    // columns append: out[b, k] = (x[b, k] for k < input_dim, else
    // one_hot(label)[k - input_dim]).
    if (x.cols != input_dim_) {
        // Shape mismatch — return a zero tensor to keep the layer from
        // crashing. (Caller should make sure cols match.)
        Tensor z(x.rows, input_dim_ + num_classes_);
        z.fill(0.0);
        return z;
    }
    Tensor out(x.rows, input_dim_ + num_classes_);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t k = 0; k < input_dim_; ++k) out(i, k) = x(i, k);
        for (size_t k = 0; k < num_classes_; ++k) out(i, input_dim_ + k) = 0.0;
        if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
            out(i, input_dim_ + static_cast<size_t>(label)) = 1.0;
        }
    }
    return out;
}

Tensor FFNetwork::make_negative_random(const Tensor& x, int true_label) const {
    // Sample a wrong label uniformly from {0..num_classes-1} \ {true_label}.
    if (num_classes_ <= 1) return make_positive(x, true_label);  // pathological
    std::uniform_int_distribution<int> dist(0, static_cast<int>(num_classes_) - 1);
    int wrong = dist(ff_rng_);
    if (wrong == true_label) {
        // Shift by 1 to avoid collision (modular — symmetric anyway).
        wrong = (wrong + 1) % static_cast<int>(num_classes_);
    }
    return make_positive(x, wrong);
}

Tensor FFNetwork::make_negative_label(const Tensor& x, int false_label) const {
    return make_positive(x, false_label);
}

std::vector<std::pair<double, double>> FFNetwork::train_step(const Tensor& x,
                                                              int label) {
    // Build positive and negative batches.
    Tensor pos_x = make_positive(x, label);
    Tensor neg_x = make_negative_random(x, label);

    // Train each layer greedily: layer i takes the previous layer's output
    // as input. For the positive sample we feed pos_x into layer 0, then
    // forward through to build pos_x for the next layer; same for neg_x.
    std::vector<std::pair<double, double>> per_layer;
    Tensor pos_act = pos_x;
    Tensor neg_act = neg_x;
    for (auto& layer : layers_) {
        // Train this layer on the current activations.
        auto stats = layer->train_ff(pos_act, neg_act);
        per_layer.push_back(stats);
        // Re-forward to get the activations for the NEXT layer. We need
        // pos_act = layer->forward(pos_act), but train_ff already called
        // forward internally. So we recompute:
        pos_act = layer->forward(pos_act);
        neg_act = layer->forward(neg_act);
    }
    return per_layer;
}

Tensor FFNetwork::predict_goodness(const Tensor& x) {
    // For each candidate label, run all layers forward with x ⊕ label,
    // compute per-layer goodness, sum. Return (1, num_classes).
    Tensor goodness(1, num_classes_);
    goodness.fill(0.0);
    for (int label = 0; label < static_cast<int>(num_classes_); ++label) {
        Tensor biased = make_positive(x, label);
        Tensor act = biased;
        double total_g = 0.0;
        for (auto& layer : layers_) {
            act = layer->forward(act);
            double g_sample = 0.0;
            for (size_t i = 0; i < act.rows; ++i) {
                for (size_t j = 0; j < act.cols; ++j) {
                    g_sample += act(i, j) * act(i, j);
                }
            }
            g_sample /= static_cast<double>(act.rows * act.cols);
            total_g += g_sample;
        }
        goodness(0, static_cast<size_t>(label)) = total_g;
    }
    return goodness;
}

int FFNetwork::predict(const Tensor& x) {
    Tensor g = predict_goodness(x);
    int best = 0;
    double best_g = g(0, 0);
    for (size_t k = 1; k < num_classes_; ++k) {
        if (g(0, k) > best_g) { best_g = g(0, k); best = static_cast<int>(k); }
    }
    return best;
}

Tensor FFNetwork::backward(const Tensor& grad_output, double learning_rate) {
    // Chain rule through all layers. Used for gradient checks / debugging.
    Tensor g = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        g = (*it)->backward(g, learning_rate);
    }
    return g;
}

void FFNetwork::update_weights(double learning_rate) {
    // In standard (non-FF) usage, propagate the gradient update. In FF
    // training, layers are updated via train_ff() and this is a no-op for
    // the params.
    for (auto& layer : layers_) {
        layer->update_weights(learning_rate);
    }
}

std::vector<Tensor*> FFNetwork::parameters() {
    std::vector<Tensor*> p;
    for (auto& layer : layers_) {
        auto pp = layer->parameters();
        for (auto* t : pp) p.push_back(t);
    }
    return p;
}

std::vector<Tensor*> FFNetwork::gradients() {
    std::vector<Tensor*> g;
    for (auto& layer : layers_) {
        auto gg = layer->gradients();
        for (auto* t : gg) g.push_back(t);
    }
    return g;
}

void FFNetwork::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}