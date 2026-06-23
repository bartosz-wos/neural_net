#include "span_extractor.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

SpanExtractor::SpanExtractor(size_t d_model, bool use_refinement)
    : d_model_(d_model),
      use_refinement_(use_refinement),
      refinement_(nullptr)
{
    if (use_refinement_) {
        refinement_ = std::make_unique<Dense>(d_model, d_model);
    }
}

Tensor SpanExtractor::extract(const Tensor& sequence,
                              const std::vector<int>& starts,
                              const std::vector<int>& ends) {
    // sequence: (n, d)
    // starts: vector<int> length B
    // ends:   vector<int> length B
    const size_t n = sequence.rows;
    const size_t d = sequence.cols;
    if (d != d_model_) {
        throw std::runtime_error("SpanExtractor: sequence feature dim != d_model");
    }
    if (starts.size() != ends.size()) {
        throw std::runtime_error("SpanExtractor: starts and ends must have same length");
    }
    const size_t B = starts.size();

    // Cache the inputs for the backward pass.
    last_sequence_ = sequence.clone();
    last_starts_ = starts;
    last_ends_ = ends;

    // Select rows: start_emb[i] = sequence[starts[i], :], end_emb[i] = sequence[ends[i], :]
    Tensor start_emb(B, d);
    Tensor end_emb(B, d);
    for (size_t i = 0; i < B; ++i) {
        if (starts[i] < 0 || static_cast<size_t>(starts[i]) >= n) {
            throw std::runtime_error("SpanExtractor: start index out of range");
        }
        if (ends[i] < 0 || static_cast<size_t>(ends[i]) >= n) {
            throw std::runtime_error("SpanExtractor: end index out of range");
        }
        for (size_t j = 0; j < d; ++j) {
            start_emb[i][j] = sequence[starts[i]][j];
            end_emb[i][j]   = sequence[ends[i]][j];
        }
    }

    // Cache pre-refine embeddings for backward.
    last_pre_refine_start_ = start_emb.clone();
    last_pre_refine_end_   = end_emb.clone();

    // Apply per-token refinement if enabled: each token (start, end) flows
    // through a Dense(d, d) + tanh. We implement the refinement inline so
    // the backward can accumulate gradients from BOTH start and end into
    // the same shared Dense weights — sharing the Dense across both sides
    // is the standard "applied to each selected token" pattern.
    if (use_refinement_) {
        // Forward through y = x @ W^T + b, then tanh. (Dense stores W as
        // (d_out, d_in); x is (B, d_in); y is (B, d_out).)
        const Tensor& W = refinement_->weights;  // (d, d)
        const Tensor& b = refinement_->bias;     // (1, d)
        Tensor out_start(B, d), out_end(B, d);
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double ys = 0.0, ye = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    ys += W[j][k] * start_emb[i][k];
                    ye += W[j][k] * end_emb[i][k];
                }
                ys += b[0][j];
                ye += b[0][j];
                out_start[i][j] = std::tanh(ys);
                out_end[i][j]   = std::tanh(ye);
            }
        }
        start_emb = out_start;
        end_emb = out_end;
    }

    // Stack into (B, 2d) row-major: cols [0, d) are start, [d, 2d) are end.
    Tensor out(B, 2 * d);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < d; ++j) {
            out[i][j]     = start_emb[i][j];
            out[i][d + j] = end_emb[i][j];
        }
    }
    return out;
}

// Backward pass for span extraction. Accepts grad_output of shape (B, 2d)
// (the same layout returned by extract), and returns grad_input of shape
// (n, d) with gradients scattered to the start/end index rows.
Tensor SpanExtractor::backward_span(const Tensor& grad_output,
                                     double learning_rate) {
    (void)learning_rate;
    const size_t B = last_starts_.size();
    const size_t n = last_sequence_.rows;
    const size_t d = d_model_;

    // grad_output: (B, 2d). Split into grad_start (B, d) and grad_end (B, d).
    // Convention: grad_output[i, j] for j < d → grad_start[i, j];
    //            grad_output[i, j] for j ≥ d → grad_end[i, j-d].
    Tensor grad_start(B, d), grad_end(B, d);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < d; ++j) {
            grad_start[i][j] = grad_output[i][j];
            grad_end[i][j]   = grad_output[i][d + j];
        }
    }

    // If refinement is used, we need to chain through tanh + Dense backward.
    // The gradient signal is: dL/dx = dL/dy * (1 - tanh²(z)) * W
    // (where z = x @ W^T + b and y = tanh(z)).
    if (use_refinement_) {
        // For each (i, j): d_y = grad_output[i, j]
        //                 d_z = d_y * (1 - y[i, j]^2)
        //                 d_W[j, k] += d_z * x[i, k]
        //                 d_b[0, j] += d_z
        //                 d_x[i, k] += d_z * W[j, k]
        // We accumulate dW and db over BOTH start and end indices because
        // they share the same Dense weights.
        Tensor& dW = refinement_->grad_weights;
        Tensor& db = refinement_->grad_bias;
        // dW, db must be zeroed by the caller (zero_grad); we just accumulate.

        // Compute d_x for start side
        Tensor d_x_start(B, d);
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double y = grad_start[i][j];  // dL/dy at the post-tanh output
                // We need the pre-tanh value to compute (1 - y^2) where y
                // is the cached post-tanh. But we cached post-refine values
                // in last_pre_refine_start_/last_pre_refine_end_; we need
                // the post-tanh activations, not pre-tanh.
                //
                // Re-derive: y_post = tanh(z), so d_z = d_y * (1 - y_post^2).
                // We need y_post (the post-tanh output). We re-compute the
                // forward inline here for backward consistency.
                double ys = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    ys += refinement_->weights[j][k] * last_pre_refine_start_[i][k];
                }
                ys += refinement_->bias[0][j];
                double y_post = std::tanh(ys);
                double d_z = y * (1.0 - y_post * y_post);
                for (size_t k = 0; k < d; ++k) {
                    dW[j][k] += d_z * last_pre_refine_start_[i][k];
                    d_x_start[i][k] += d_z * refinement_->weights[j][k];
                }
                db[0][j] += d_z;
            }
        }
        // Now do the same for end side
        Tensor d_x_end(B, d);
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < d; ++j) {
                double y = grad_end[i][j];
                double ye = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    ye += refinement_->weights[j][k] * last_pre_refine_end_[i][k];
                }
                ye += refinement_->bias[0][j];
                double y_post = std::tanh(ye);
                double d_z = y * (1.0 - y_post * y_post);
                for (size_t k = 0; k < d; ++k) {
                    dW[j][k] += d_z * last_pre_refine_end_[i][k];
                    d_x_end[i][k] += d_z * refinement_->weights[j][k];
                }
                db[0][j] += d_z;
            }
        }
        grad_start = d_x_start;
        grad_end = d_x_end;
    }

    // Scatter grad_start[i] to sequence row last_starts_[i], and grad_end[i]
    // to sequence row last_ends_[i]. Multiple batch entries can index the
    // same row — accumulate (sum) their gradients.
    Tensor grad_input(n, d);
    grad_input.fill(0.0);
    for (size_t i = 0; i < B; ++i) {
        int si = last_starts_[i];
        int ei = last_ends_[i];
        for (size_t j = 0; j < d; ++j) {
            grad_input[si][j] += grad_start[i][j];
            grad_input[ei][j] += grad_end[i][j];
        }
    }
    return grad_input;
}

Tensor SpanExtractor::forward(const Tensor& input) {
    // Single-arg Layer forward: treat as a no-op pass-through that returns
    // the input unchanged. The real entry point is extract(). For
    // compatibility with Model's chain (which calls forward/backward on
    // each layer in sequence), this lets the SpanExtractor sit in a model
    // stack without breaking the interface.
    (void)input;
    // Return a 1x1 zero tensor — model chains shouldn't actually call this
    // for span extraction; they should call extract() directly.
    Tensor out(1, 1);
    out.fill(0.0);
    return out;
}

Tensor SpanExtractor::backward(const Tensor& grad_output, double learning_rate) {
    // The single-arg backward is unsupported — SpanExtractor's backward is
    // called via extract()'s complementary path, not via Layer's standard
    // chain. Return zeros of the appropriate shape.
    (void)learning_rate;
    Tensor z(grad_output.rows, grad_output.cols);
    z.fill(0.0);
    return z;
}

void SpanExtractor::update_weights(double learning_rate) {
    if (use_refinement_) {
        refinement_->update_weights(learning_rate);
    }
}

std::vector<Tensor*> SpanExtractor::parameters() {
    if (use_refinement_) {
        return refinement_->parameters();
    }
    return {};
}

std::vector<Tensor*> SpanExtractor::gradients() {
    if (use_refinement_) {
        return refinement_->gradients();
    }
    return {};
}

void SpanExtractor::zero_grad() {
    if (use_refinement_) {
        refinement_->zero_grad();
    }
}

Tensor SpanExtractor::get_weights() const {
    if (use_refinement_) {
        return refinement_->get_weights();
    }
    return Tensor(0, 0);
}

Tensor SpanExtractor::get_gradients() const {
    if (use_refinement_) {
        return refinement_->get_gradients();
    }
    return Tensor(0, 0);
}
