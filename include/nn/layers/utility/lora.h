#ifndef LORA_H
#define LORA_H

#include "../../core/layer.h"
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// LoRA — Low-Rank Adaptation of dense layers
//   Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — 2021
//   "LoRA: Low-Rank Adaptation of Large Language Models"
//   https://arxiv.org/abs/2106.09685
//
// Drop-in adapter for a frozen Dense: adds a TRAINABLE low-rank update
//   ΔW = (α / rank) · B · A   ∈ ℝ^{d_out × d_in}
// to the base weight W_0. Forward becomes
//   y = x · (W_0 + ΔW)ᵀ + b
//     = x · W_0ᵀ + (α/rank) · x · Aᵀ · Bᵀ + b
// Both paths share the input — the B=0 initialization (§4.2) means the LoRA
// path contributes EXACTLY ZERO at construction, so the layer's initial
// forward is bit-exact equal to the base Dense (the regression test for
// "LoRA breaks the base on day 1").
//
// Inference-time merge: merge_weights() folds the update into the base
//   W' = W_0 + (α/rank) · B · A
// so a forward becomes a single matmul (paper §4.2, no extra FLOPs at
// deployment). After merge, A/B should NOT be updated further — the
// training switch is "merged_" which short-circuits update_weights().
//
// rank = 0 boundary: when the user wants pure base Dense with NO LoRA
// path (e.g. r=0 ablation in the paper), rank=0 means the LoRA path is
// identically zero. Allowed ONLY if freeze_base=true (otherwise nothing
// is trainable, which is rejected).
// ============================================================================

class LoRALinear : public Layer {
public:
    // d_in / d_out: base Dense dimensions.
    // rank: low-rank decomposition rank. r=0 → LoRA is a no-op (paper §4.1).
    // alpha: scalar; scaling factor is (alpha / rank). Default: alpha = rank,
    //        so scaling = 1.0 (paper convention).
    // base_init: passed to Dense.init_weights() — "xavier" (default),
    //        "he", "uniform", "zeros".
    // freeze_base: if true (default), the base W_0/b are not trained. If
    //        false, LoRA trains both the low-rank update AND the base
    //        (full fine-tuning via the same rank-1 path).
    LoRALinear(size_t d_in, size_t d_out, size_t rank,
               double alpha = -1.0,
               const std::string& base_init = "xavier",
               bool freeze_base = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_0_->weights; }
    Tensor get_gradients() const override { return grad_A_; }
    std::string name() const override { return "LoRALinear"; }

    // Accessors
    size_t d_in() const { return d_in_; }
    size_t d_out() const { return d_out_; }
    size_t rank() const { return rank_; }
    double alpha() const { return alpha_; }
    double scaling() const {
        return rank_ == 0 ? 0.0 : alpha_ / static_cast<double>(rank_);
    }
    bool is_merged() const { return merged_; }
    bool base_frozen() const { return freeze_base_; }

    const Tensor& A() const { return A_; }
    Tensor& A() { return A_; }            // mutable accessor for tests
    const Tensor& B() const { return B_; }
    Tensor& B() { return B_; }            // mutable accessor for tests
    const Tensor& base_weights() const { return W_0_->weights; }
    Tensor& base_weights() { return W_0_->weights; }  // mutable for tests
    const Tensor& base_bias() const { return W_0_->bias; }
    Tensor& base_bias() { return W_0_->bias; }        // mutable for tests

    // LoRA gradients — exposed for FD tests (typically private in other
    // layers; LoRA uses public because the FD tests need direct read access
    // to verify the analytical backward).
    const Tensor& grad_A() const { return grad_A_; }
    const Tensor& grad_B() const { return grad_B_; }

    // Merge: W_0 ← W_0 + (α/rank) · B · A, then disable A/B training.
    // Forward after merge ≡ base Dense forward (bit-exact), zero extra FLOPs.
    void merge_weights();

    // Unmerge: revert a merge (subtract the contribution back out, leave
    // A/B zeroed as after merge). To continue training post-merge, do
    // unmerge → restore A, B from a snapshot → resume SGD.
    void unmerge_weights();

private:
    size_t d_in_;
    size_t d_out_;
    size_t rank_;
    double alpha_;
    bool freeze_base_;
    bool merged_;

    // Base Dense (frozen if freeze_base_ is true)
    std::unique_ptr<Dense> W_0_;

    // LoRA params: A (rank, d_in), B (d_out, rank)
    Tensor A_;
    Tensor B_;

    // LoRA grads (accumulated in backward, zeroed in update_weights /
    // zero_grad). Base Dense grads come from W_0_->grad_weights /
    // grad_bias (when freeze_base is false).
    Tensor grad_A_;
    Tensor grad_B_;

    // Forward cache (used by backward). Public for FD tests; not part of
    // the public API contract.
    Tensor last_input_;    // (N, d_in)
    Tensor last_Z_;        // (N, rank)  = last_input_ · Aᵀ
    Tensor last_y_lora_;   // (N, d_out) = scaling · last_Z_ · Bᵀ
    Tensor last_base_out_; // (N, d_out) = base Dense forward (unused for now)
};

#endif