#ifndef SPIKING_LIF_H
#define SPIKING_LIF_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <cmath>
#include <vector>
#include <memory>

// ============================================================================
// Spiking Neural Networks — LIF neurons trained with surrogate gradients
//
// Source of truth: Neftci, Gutierrez, Bohte, "Surrogate Gradient Learning in
// Spiking Neural Networks", arXiv:1901.09948v2 — Box 1, Eq. 4-5, Sec. IV-A and
// Sec. V-B (surrogate gradients). The equation numbering below refers to that
// paper.
//
// ---------------------------------------------------------------------------
// Discrete-time LIF dynamics (Eq. 4-5), with the survey's stated
// normalizations U_rest = 0, R = 1:
//
//   I_t = α · I_{t-1} + Σ_j W_ij · X_t[j] + b_i          (Eq. 4: synaptic current)
//       + Σ_j V_ij · S_{t-1}[j]                          (Eq. 2 recurrent term,
//                                                        present only when
//                                                        use_recurrence = true)
//   U_t = β · U_{t-1} + I_t − ϑ · S_{t-1}                 (Eq. 5: membrane)
//   S_t = Θ(U_t − ϑ)                                      (hard Heaviside, spikes in {0,1})
//
//   α = exp(−Δt/τ_syn) ∈ (0,1)   β = exp(−Δt/τ_mem) ∈ (0,1)
//
// ---------------------------------------------------------------------------
// THE TWO CONVENTIONS THE TESTS PIN
//
// (1) The reset term uses S_{t-1}, NOT S_t.  Eq. 5 carries "−S_i[n] · ϑ" while
//     the spike is emitted as S_i[n] = Θ(U_i[n] − ϑ), so a spike at step t is
//     subtracted from the membrane at step t+1.  A "reset immediately after
//     firing" implementation (using S_t) produces a membrane trace shifted by
//     one step.  Both are self-consistent; only this one matches the equations.
//
// (2) Θ(0) = 1, i.e. S = (U >= ϑ).  The Heaviside at exactly 0 is a
//     measure-zero detail of the forward pass, but it must be pinned because
//     the hand-derived T=3 trace lands exactly on the threshold at t=0.
//
// Note that the reset term −ϑ·S_{t-1} is a CONSTANT with respect to every
// parameter and to U, so it contributes NOTHING to any gradient — it changes
// the forward trace only.  Its absence from the backward is not an omission.
//
// ---------------------------------------------------------------------------
// Surrogate gradient (Sec. V-B, Fig. 3/4)
//
// The true dΘ/dU is zero everywhere except at U = ϑ, where it is ill-defined,
// which makes an SNN's loss landscape a plateau (Fig. 4a).  Surrogate-gradient
// methods replace it, during BACKWARD ONLY, with the derivative of a smooth
// function — the forward pass stays hard-binary, which is what preserves the
// energy/fault-tolerance property of SNNs.
//
// This implements the fast-sigmoid ("SuperSpike") family, which the survey
// attributes to Zenke & Ganguli (its ref [2]):
//
//     forward:   S_t = Θ(U_t − ϑ)                          (hard, unchanged)
//     backward:  ∂S/∂U  ≈  1 / (γ · |U_t − ϑ| + 1)
//
// γ (learnable, default 10) is the sharpness: larger γ is narrower and closer
// to a true derivative.  The survey's stated virtue is that the surrogate is
// continuous and finite, so BPTT keeps working where the true gradient is a
// plateau (Test 11 pins this: non-zero at |U−ϑ| = 11 where the true gradient
// is exactly 0).
//
// ---------------------------------------------------------------------------
// Shape conventions
//   X:                       (T, input_dim)
//   I, U, S:                 (T, hidden_size)
//   W_:                      (hidden_size, input_dim)
//   b_:                      (hidden_size, 1)      <- column, indexed b_[i][0]
//   V_:                      (hidden_size, hidden_size)   [recurrent only]
//   gamma_:                  (1, 1)
//
//   S is stored as a real tensor holding EXACTLY 0.0 or 1.0 (the binary spike
//   train), not as spike indices: the (T, hidden) layout is what lets the
//   backward loops stay dense, matching every other recurrent layer here.
// ============================================================================

// ----------------------------------------------------------------------------
// SurrogateSpike — stateless spike nonlinearity + surrogate derivative
// ----------------------------------------------------------------------------
struct SurrogateSpike {
    // Forward: hard Heaviside. Theta(0) = 1.
    static double spike(double u, double threshold) {
        return (u >= threshold) ? 1.0 : 0.0;
    }

    // Surrogate derivative (fast-sigmoid family): 1 / (gamma*|u-theta| + 1)
    static double surrogate_derivative(double u, double threshold, double gamma) {
        return 1.0 / (gamma * std::abs(u - threshold) + 1.0);
    }

    // d/dgamma of the above:  -|u-theta| / (gamma*|u-theta| + 1)^2
    static double surrogate_derivative_dgamma(double u, double threshold, double gamma) {
        double a = std::abs(u - threshold);
        return -a / ((gamma * a + 1.0) * (gamma * a + 1.0));
    }
};

// ----------------------------------------------------------------------------
// LIFNeuron — one layer of leaky integrate-and-fire neurons
// ----------------------------------------------------------------------------
class LIFNeuron : public Layer {
public:
    LIFNeuron(size_t input_dim, size_t hidden_size,
              double alpha = 0.5, double beta = 0.5,
              double threshold = 1.0, double gamma = 10.0,
              bool use_recurrence = false);

    // Input: (T, input_dim) -> Output: (T, hidden_size), values exactly {0,1}
    Tensor forward(const Tensor& input) override;

    // Receives grad_output (T, hidden_size) w.r.t. the SPIKE output and
    // returns grad_input (T, input_dim). Surrogate substitution is applied
    // internally. Parameter gradients accumulate per the Layer convention.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_; }
    Tensor get_gradients() const override { return grad_W_; }
    std::string name() const override { return "LIFNeuron"; }

    size_t input_dim() const { return input_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    double threshold() const { return threshold_; }
    double gamma() const { return gamma_[0][0]; }
    bool uses_recurrence() const { return use_recurrence_; }

    // Test introspection: cached values from the most recent forward call.
    Tensor last_spikes() const { return cache_S_; }
    Tensor last_membrane() const { return cache_U_; }
    Tensor last_current() const { return cache_I_; }

public:
    // Public parameter tensors (tests set them directly).
    Tensor W_, b_, gamma_;
    Tensor V_;                       // only allocated when use_recurrence

    // Public gradient tensors.
    Tensor grad_W_, grad_b_, grad_gamma_;
    Tensor grad_V_;                  // only allocated when use_recurrence

private:
    size_t input_dim_;
    size_t hidden_size_;
    double alpha_;                   // synaptic decay, (0,1)
    double beta_;                    // membrane decay, (0,1)
    double threshold_;               // firing threshold ϑ
    bool use_recurrence_;

    Tensor cache_X_;                 // (T, input_dim)
    Tensor cache_I_;                 // (T, hidden)
    Tensor cache_U_;                 // (T, hidden)
    Tensor cache_S_;                 // (T, hidden)  binary spikes
};

// ----------------------------------------------------------------------------
// SpikingNet — rate encoder -> stack of LIF layers -> rate readout
// ----------------------------------------------------------------------------
class SpikingNet : public Layer {
public:
    SpikingNet(size_t input_dim, size_t hidden_size, size_t output_dim,
               size_t num_layers = 2,
               double alpha = 0.5, double beta = 0.5,
               double threshold = 1.0, double gamma = 10.0,
               double encode_frac = 0.5, double encode_gain = 1.0);

    // Input: (B, input_dim) -> Output: (B, output_dim)
    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "SpikingNet"; }

    size_t num_layers() const { return layers_.size(); }
    Tensor layer_spikes(size_t l) const;
    Tensor last_encoded_spikes() const { return cache_encoded_; }

private:
    // Rate encoding: (B, input_dim) -> (B, input_dim) spike train.
    // Each sample is presented for one timestep; a spike is emitted where the
    // value exceeds gain * (the max |x| in that row), scaled by rank.
    Tensor encode_rate(const Tensor& input);

    size_t input_dim_;
    size_t hidden_size_;
    size_t output_dim_;
    double alpha_, beta_, threshold_, gamma_;
    double encode_frac_;              // fraction of features per row that spike
    double encode_gain_;               // scale on the binary train into layer 0

    std::vector<std::unique_ptr<LIFNeuron>> layers_;
    std::unique_ptr<Dense> readout_;
    Tensor cache_encoded_;           // (B, input_dim) binary
    std::vector<Tensor> cache_layer_out_;  // spikes of each layer
};

#endif  // SPIKING_LIF_H