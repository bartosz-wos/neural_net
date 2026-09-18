#ifndef DCN_V2_H
#define DCN_V2_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
// Dense lives in ../../core/layer.h, included above.
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// Deep & Cross Network v2 (DCN-v2) — Wang, Fu, Fu, Wang, Liu, Zhang 2021
//   "DCN V2: Improved Deep & Cross Network and Practical Lessons for
//    Web-scale Learning to Rank Systems"
//   https://arxiv.org/abs/2008.13535
//
// Three classes:
//
//   (1) CrossLayer      — one low-rank cross layer (stateless helper struct).
//   (2) CrossNetwork    — stack of CrossLayer; the cross tower. Layer.
//   (3) DCNv2           — CrossNetwork + a deep MLP tower + a stack Dense.
//                          The full DCN-v2 model. Layer.
//   (4) DCNv2Regression — convenience wrapper around DCNv2 for regression /
//                          binary classification with sensible defaults.
//
// Math (low-rank cross, paper §3.1 Eq. 6):
//   Given input x_0 ∈ R^{B × d}, and layer l = 1, …, L:
//     p_l = C_l · x_{l-1}      (B, r)   C_l ∈ R^{r × d}
//     q_l = U_l · p_l          (B, d)   U_l ∈ R^{d × r}
//     x_l = x_0 ⊙ q_l + x_{l-1}         (elementwise cross with x_0 + residual)
//
// The two factors (C, U) parameterize an effective (d, d) weight W = U · C
// at cost 2·d·r ≪ d².  When r = d and U = I, the layer is exactly the
// original full-rank cross of DCN-v1.  CrossNetwork::forward / backward
// implement both cases uniformly; Test 9 verifies the r = d bit-exact match.
//
// Backward chain (paper §3.1 + standard chain rule):
//   Given d x_l (B, d), for one layer with cached x_0, x_{l-1}, p_l, q_l:
//     dq = x_0 ⊙ d x_l                          (B, d)
//     dU = dq^T · p_l                           (d, r)
//     dC = (U^T · dq)^T · x_{l-1}               (r, d)
//     d x_{l-1} = d x_l + (U · C)^T · dq        (B, d)
//     d x_0 (accumulated) += d x_l ⊙ q_l
//   Layers walk BACKWARDS; d x_0 accumulates contributions from every layer.
//
// Deep tower: standard MLP (ReLU + Dense).  The deep-tower input is the
// ORIGINAL input (not x_L) — this is paper §3.1's "parallel structure":
//
//     ┌── CrossNetwork ──► x_L ──┐
//     │                         ▼
//   x ─┤                    [x_L; h_deep]
//     │                         ▼
//     └── DeepMLP ─────► h_deep ─► StackLayer (Dense) ─► output
//
// Stack layer is a single Dense that maps [x_L ‖ h_deep] ∈ R^{B × (d + |deep|)}
// to the output dimension.  This is the "v2" mixing — v1 cross was just
// concatenated with deep and fed to a sigmoid without an extra Dense.
//
// Design decisions:
//   - ReLU on deep tower (paper §3.1).
//   - No bias on cross layer output (paper convention).
//   - r validation in CrossLayer: r == 0 throws, r > d throws.
//   - DCNv2: input_dim == d (the cross dim) == the deep-tower input dim.
//   - DCNv2Regression defaults: cross_layers=3, deep_hidden=64, deep_layers=2,
//     cross_r = max(1, input_dim / 4).
// ============================================================================


// ============================================================================
// CrossLayer — one low-rank cross layer.  Plain struct, not a Layer.
// ============================================================================
struct CrossLayer {
    size_t d;          // input/output dim
    size_t r;          // low-rank dim
    Tensor C;          // (r, d)
    Tensor U;          // (d, r)
    Tensor gC;         // (r, d) gradient
    Tensor gU;         // (d, r) gradient
    Tensor last_p;     // (B, r) cached p_l = C · x_{l-1}
    Tensor last_q;     // (B, d) cached q_l = U · p_l
    Tensor last_x_lm1; // (B, d) cached x_{l-1}

    CrossLayer(size_t d_, size_t r_);
    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

// ============================================================================
// CrossNetwork — stack of CrossLayer, all sharing the original input x_0.
// ============================================================================
class CrossNetwork : public Layer {
public:
    CrossNetwork(size_t d, size_t num_layers, size_t r);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "CrossNetwork"; }

    size_t d() const { return d_; }
    size_t num_layers() const { return num_layers_; }
    size_t r() const { return r_; }
    const std::vector<CrossLayer>& layers() const { return layers_; }
    std::vector<CrossLayer>& layers() { return layers_; }
    const Tensor& last_x0() const { return last_x0_; }
    const Tensor& last_xL() const { return last_xL_; }

private:
    size_t d_, num_layers_, r_;
    std::vector<CrossLayer> layers_;
    Tensor last_x0_;        // (B, d) cached input
    Tensor last_xL_;        // (B, d) cached final output
};


// ============================================================================
// DCNv2 — full Deep & Cross Network v2 model.
// ============================================================================
class DCNv2 : public Layer {
public:
    DCNv2(size_t input_dim, size_t cross_layers, size_t cross_r,
          const std::vector<size_t>& deep_dims, size_t output_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "DCNv2"; }

    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }
    const CrossNetwork& cross() const { return cross_; }
    CrossNetwork& cross() { return cross_; }

    // Test setters / accessors (NOT for production use)
    Dense& stack_for_test() { return stack_; }
    Dense& deep_for_test(size_t i) { return *deep_[i]; }

    // Deep-copy helper for FD gradient check tests.
    DCNv2 clone() const;

    // Non-copyable (unique_ptr members); use clone() if you need a copy.
    DCNv2(const DCNv2&) = delete;
    DCNv2& operator=(const DCNv2&) = delete;
    DCNv2(DCNv2&&) = default;
    DCNv2& operator=(DCNv2&&) = default;

private:
    size_t input_dim_, output_dim_;
    std::vector<size_t> deep_dims_;     // for clone() and introspection
    CrossNetwork cross_;
    std::vector<std::unique_ptr<Dense>> deep_;   // ReLU MLPs (Dense only; ReLU applied manually in forward / backward)
    Dense stack_;                                // [x_L; h_deep] → output_dim
    Tensor last_xL_;                             // cross output (B, d)
    Tensor last_h_deep_;                         // deep tower final output (B, last_deep_dim)
    std::vector<Tensor> last_h_pre_relu_;        // per-deep-layer pre-ReLU activations
    Tensor last_concat_;                         // [x_L ‖ h_deep] (B, d + last_deep_dim)
};


// ============================================================================
// DCNv2Regression — convenience wrapper around DCNv2.
// ============================================================================
class DCNv2Regression : public Layer {
public:
    DCNv2Regression(size_t input_dim, size_t output_dim = 1,
                    size_t cross_r = 0,           // 0 → max(1, input_dim / 4) default
                    size_t cross_layers = 3,
                    size_t deep_hidden = 64,
                    size_t deep_layers = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "DCNv2Regression"; }

    size_t input_dim() const { return model_.input_dim(); }
    size_t output_dim() const { return model_.output_dim(); }
    const DCNv2& model() const { return model_; }
    size_t cross_layers() const { return model_.cross().num_layers(); }
    size_t cross_r() const { return model_.cross().r(); }

private:
    DCNv2 model_;
};

#endif // DCN_V2_H