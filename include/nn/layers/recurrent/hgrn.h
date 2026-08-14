#ifndef HGRN_H
#define HGRN_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// HGRN (Hierarchically Gated Recurrent Neural Network) — Mehrdad, Cevher,
//        Giannakis 2023
//   "Hierarchically Gated Recurrent Neural Network"
//   https://arxiv.org/abs/2307.02226
//
// HGRN is a per-channel scalar-state linear RNN with separate forget/input
// gates (and optionally an output gate). Compared to the existing recurrent
// layers in this repo, HGRN fills the gap between GRU/LSTM (full additive
// gates, no state coupling) and the linear-attention family (GLA, Mamba,
// DeltaNet — single scalar decay).
//
// ----------------------------------------------------------------------------
// Mathematical formulation (Algorithm 1 in the paper, single-state per
// channel version):
//
//   Input:        x ∈ R^{T × d_in}
//   Parameters:   W_f, W_i, W_z ∈ R^{d_in × d_hidden}      (gate projections)
//                 b_f, b_i, b_z ∈ R^{d_hidden}
//                 W_o, b_o ∈ R^{d_in × d_hidden}           (output gate, HGRN-2 only)
//
//   Per-token gate pre-activations (broadcasted over T):
//     f_pre_t[c] = W_f x_t + b_f       (forget gate pre-activation)
//     i_pre_t[c] = W_i x_t + b_i       (input gate pre-activation)
//     z_pre_t[c] = W_z x_t + b_z       (candidate pre-activation)
//     o_pre_t[c] = W_o x_t + b_o       (output gate pre-activation, HGRN-2 only)
//
//   Per-channel scalar-state recurrence:
//     f_t[c] = sigmoid(f_pre_t[c])     ∈ (0, 1)
//     i_t[c] = sigmoid(i_pre_t[c])     ∈ (0, 1)
//     z_t[c] = tanh(z_pre_t[c])        ∈ (-1, 1)
//     o_t[c] = sigmoid(o_pre_t[c])     ∈ (0, 1)             (HGRN-2 only)
//     c_0 = 0
//     c_t[c] = f_t[c] * c_{t-1}[c] + i_t[c] * z_t[c]
//     h_t[c] = c_t[c]                                    (HGRN-1)
//     h_t[c] = o_t[c] * c_t[c]                           (HGRN-2)
//
// ----------------------------------------------------------------------------
// Shape conventions (consistent with GLA / Hawk / H3 / RWKV in this repo):
//
//   Input x:           (T, input_dim)
//   Gate pre-activations: (T, hidden_dim) each
//   Hidden state c:    (T, hidden_dim) — flattened cache of c_0..c_{T-1}
//                      (c_0 is conceptually 0, but the cache stores c_t for
//                       t=0..T-1, with c_{-1}=0 used in the BPTT formula)
//   Output h:          (T, hidden_dim)
//
// ----------------------------------------------------------------------------
// Backward pass (single-step BPTT):
//
//   Reverse sweep (T-1 → 0):
//     For HGRN-1:
//       grad_c[t] = grad_h[t]
//     For HGRN-2:
//       grad_c[t][c] = grad_h[t][c] * o_t[c]
//       grad_o_pre[t][c] = grad_h[t][c] * c_t[c] * o_t[c] * (1 - o_t[c])
//     Then carry forward: grad_c[t-1][c] += grad_c[t][c] * f_t[c]
//
//   Per-channel parameter gradients (for each t):
//     grad_f_pre[t][c] = grad_c[t][c] * c_{t-1}[c] * f_t[c] * (1 - f_t[c])
//     grad_i_pre[t][c] = grad_c[t][c] * z_t[c]       * i_t[c] * (1 - i_t[c])
//     grad_z_pre[t][c] = grad_c[t][c] * i_t[c]       * (1 - z_t[c]^2)
//
//   The pre-activation gradients are passed to W_f.backward / W_i.backward /
//   W_z.backward (and W_o.backward for HGRN-2). Dense::backward internally
//   computes grad_W += grad_output^T * last_input and grad_b += sum over
//   batch, and returns grad_output * W (= grad_x contribution). All three
//   Dense backward calls share the same `last_input` (the original x), so
//   the grad_x contributions accumulate correctly via +.
//
//   Notes on dimensions:
//     * Dense stores weights as (out_features, in_features). We pass
//       (T, in_features) to Dense.forward, which gives (T, out_features)
//       for the pre-activations. W_f(in=d_in, out=d_hidden) → pre (T, d_hidden).
//     * The "grad_output" shape for W_f.backward is therefore (T, d_hidden),
//       matching grad_f_pre.
//
// ============================================================================

class HGRNCell : public Layer {
public:
    // input_dim:   input feature dimension (d_in)
    // hidden_dim:  hidden state dimension (d_hidden)
    // use_output_gate: if true, implements HGRN-2 (output = o * c);
    //                  if false, implements HGRN-1 (output = c).
    HGRNCell(size_t input_dim, size_t hidden_dim, bool use_output_gate = false);

    // Input:  (T, input_dim)   Output: (T, hidden_dim)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_f.weights; }
    Tensor get_gradients() const override { return W_f.grad_weights; }
    std::string name() const override { return "HGRNCell"; }

    // Accessors for tests
    size_t input_dim() const { return input_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }
    bool use_output_gate() const { return use_output_gate_; }

    // Public Dense accessors (for tests). W_o is default-constructed when
    // use_output_gate=false and does not participate in forward/backward.
    Dense W_f;
    Dense W_i;
    Dense W_z;
    Dense W_o;

private:
    size_t input_dim_;
    size_t hidden_dim_;
    bool use_output_gate_;

    // Forward caches (per-token, per-channel). All shape (T, hidden_dim).
    Tensor cache_input_;        // (T, input_dim)
    Tensor cache_pre_f_;        // (T, hidden_dim) — sigmoid pre-activation
    Tensor cache_pre_i_;        // (T, hidden_dim)
    Tensor cache_pre_z_;        // (T, hidden_dim)
    Tensor cache_pre_o_;        // (T, hidden_dim) — only used if use_output_gate_
    Tensor cache_f_;            // (T, hidden_dim) — sigmoid(pre_f)
    Tensor cache_i_;            // (T, hidden_dim) — sigmoid(pre_i)
    Tensor cache_z_;            // (T, hidden_dim) — tanh(pre_z)
    Tensor cache_o_;            // (T, hidden_dim) — sigmoid(pre_o), only HGRN-2
    Tensor cache_c_;            // (T, hidden_dim) — c_t values

    static double sigmoid(double x);
};

// ============================================================================
// HGRNModel — stacks HGRNCells + last-timestep classifier.
//
// forward: (T, input_dim) → (1, output_dim)
//   x → HGRNCell_1 → ... → HGRNCell_L → h_L → slice last row → Dense classifier → ŷ
// =============================================================================
class HGRNModel : public Layer {
public:
    // input_dim:    input feature dimension
    // hidden_dim:   hidden dim of every HGRN cell
    // output_dim:   classifier output dim
    // num_layers:   number of HGRN cells to stack
    // use_output_gate: if true, every cell uses HGRN-2; else HGRN-1
    HGRNModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
              size_t num_layers = 1, bool use_output_gate = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return cells.empty() ? Tensor() : cells.front()->W_f.weights; }
    Tensor get_gradients() const override { return cells.empty() ? Tensor() : cells.front()->W_f.grad_weights; }
    std::string name() const override { return "HGRNModel"; }

    size_t num_layers() const { return cells.size(); }

    // Public accessors (for tests)
    std::vector<std::unique_ptr<HGRNCell>> cells;
    Dense classifier;     // (hidden_dim → output_dim)

    // Cache: last input + last cell outputs (for backward)
    Tensor last_input_;       // (T, input_dim)
    std::vector<Tensor> last_cell_outputs_;   // per-cell (T, hidden_dim)
};

#endif // HGRN_H
