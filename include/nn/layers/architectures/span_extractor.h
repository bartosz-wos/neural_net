#ifndef SPAN_EXTRACTOR_H
#define SPAN_EXTRACTOR_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>

// SpanExtractor: SQuAD-style span extraction head.
//
// Given a sequence of (n, d) token embeddings (typically the output of a
// transformer encoder) and per-example (start, end) integer index tensors
// of length batch, returns for each example a (2, d) span representation
// consisting of the start-token embedding stacked over the end-token
// embedding:
//
//   start_emb_i = sequence[start_indices[i], :]   ∈ R^d
//   end_emb_i   = sequence[end_indices[i],   :]   ∈ R^d
//   span_i      = stack(start_emb_i, end_emb_i)   ∈ R^{2×d}
//
// Optional learned refinement: a per-token Dense(d, d) followed by a
// nonlinearity is applied to the span embeddings before they are returned.
// This lets the QA head learn task-specific transformations of the
// selected span tokens (a standard trick from the SQuAD literature).
//
// Backward: the start/end indices are *non-differentiable* integer
// selections, so the gradient w.r.t. sequence flows via standard
// index-select backward — grad_start and grad_end are scattered into the
// corresponding rows of the (n, d) gradient tensor; the refinement layer's
// gradients are accumulated on its weights and bias.

class SpanExtractor : public Layer {
public:
    // d_model: feature dimension of the input sequence embeddings.
    // use_refinement: if true, adds a Dense(d_model, d_model) + tanh refinement
    //                 to each selected span token (start and end separately).
    explicit SpanExtractor(size_t d_model, bool use_refinement = true);

    // Standard Layer forward interface is not the typical entry point for
    // span extraction (we need the index tensors), but we provide a single-
    // arg forward that treats the first half of the columns as start and
    // the second half as end (for compatibility with Model's chain).
    Tensor forward(const Tensor& input) override;

    // Span-extraction entry point:
    //   sequence:  (n, d_model) — token-level encodings
    //   starts:    vector<int> length B — start token index per batch example
    //   ends:      vector<int> length B — end token index per batch example
    // Returns:
    //   (B, 2, d_model) — for each batch example, [start_emb; end_emb]
    Tensor extract(const Tensor& sequence,
                   const std::vector<int>& starts,
                   const std::vector<int>& ends);

    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    // Span-extraction backward: takes the (B, 2d) grad_output (same shape
    // as extract's return) and returns (n, d) grad_input with gradients
    // scattered to the start/end index rows. Also accumulates gradients
    // on the refinement Dense (if any).
    Tensor backward_span(const Tensor& grad_output, double learning_rate);
    void update_weights(double learning_rate) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "SpanExtractor"; }

    // Accessors for testing / inspection.
    const Tensor& last_sequence() const { return last_sequence_; }
    const std::vector<int>& last_starts() const { return last_starts_; }
    const std::vector<int>& last_ends() const { return last_ends_; }
    bool use_refinement() const { return use_refinement_; }
    Dense* refinement() { return refinement_.get(); }

private:
    size_t d_model_;
    bool use_refinement_;
    std::unique_ptr<Dense> refinement_;  // per-token Dense(d, d) + tanh

    // Caches from the most recent extract() call (used by backward).
    Tensor last_sequence_;             // (n, d)
    std::vector<int> last_starts_;     // (B,)
    std::vector<int> last_ends_;       // (B,)
    Tensor last_pre_refine_start_;     // (B, d) — selected start embedding before refinement
    Tensor last_pre_refine_end_;       // (B, d) — selected end embedding before refinement
};

#endif
