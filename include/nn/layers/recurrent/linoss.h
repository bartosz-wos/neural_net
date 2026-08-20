#ifndef LINOSS_H
#define LINOSS_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <memory>

// Forward decl — Dense lives in core/layer.h.
class Dense;

// ============================================================================
// LinOSS — Linear Oscillator State Space Layer
// Orvieto, Smith, Gu, Fernando, Goyal et al. 2024/2025,
// "From Principles to Learning Systems: State Space Models That Need No
// Discretization" (ICLR 2025), https://arxiv.org/abs/2410.03943
//
// Each oscillator pair (k = 0..d_osc-1) is a 2×2 block in continuous time:
//     A_k = [[ -exp(lambda_k),       omega_k  ],
//            [     -omega_k,     -exp(lambda_k) ]]
// so A's eigenvalues are -exp(lambda_k) ± i·omega_k — exponentially stable
// by construction. B ∈ R^{(2*d_osc, d_input)} and C ∈ R^{(d_output, 2*d_osc)}
// are dense projections.
//
// Two discretization flavors (both stable):
//   * IMPLICIT_MIDPOINT — implicit, A-stable for any dt>0.
//                         Solve (I - dt/2·A) x_{k+1} = (I + dt/2·A) x_k
//                                                   + dt · B · u_k
//                         via closed-form 2×2 block inversion.
//   * HEUN             — explicit Heun (trapezoidal).
//                         x_hat = x_k + dt · (A x_k + B u_k)
//                         x_{k+1} = x_k + dt/2 · [(A x_k + B u_k)
//                                                 + (A x_hat + B u_k)]
//
// All discretization constants (dt) are exposed as a learned scalar log_dt.
// Per-channel damping and frequency are learned via log_damping and freq.
// ============================================================================

enum class LinOSSType { IMPLICIT_MIDPOINT, HEUN };

class LinOSSCell : public Layer {
public:
    LinOSSCell(int d_input, int d_output, int d_osc,
               LinOSSType type = LinOSSType::IMPLICIT_MIDPOINT);

    Tensor forward(const Tensor& input) override;       // (T, d_input) → (T, d_output)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return B; }
    Tensor get_gradients() const override { return grad_B; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LinOSSCell"; }

    // Reset only the recurrent state (not gradients) — useful for FD re-evals.
    void reset_state();

    // Learned parameters (public for FD gradient checks).
    Tensor B;             // (d_input, 2*d_osc)
    Tensor C;             // (2*d_osc, d_output)
    Tensor log_damping;   // (d_osc,) — exp() → damping per oscillator
    Tensor freq;          // (d_osc,) — natural frequency per oscillator
    Tensor log_dt;        // (1, 1) scalar — softplus(dt) gives the step size
    Tensor bias;          // (d_output,) — output bias

    // Gradient accumulators.
    Tensor grad_B, grad_C, grad_log_damping, grad_freq, grad_log_dt, grad_bias;

    // Cached z_t = M^{-T} dL/dx_{t+1} per timestep, used for parameter gradients
    // through the A-matrix.
    std::vector<std::vector<double>> z_t_per_t_;  // z_t_per_t_[t][i]

    LinOSSType type_;
    int d_input_, d_output_, d_osc_;

private:
    // Cached forward intermediates for backward().
    Tensor last_input_;          // (T, d_input)
    Tensor last_u_proj_;         // (T, 2*d_osc)  — B · input per step
    std::vector<Tensor> states_; // T+1 cached x_t (2*d_osc,) each
    std::vector<Tensor> Ax_cache_; // T cached A · x_t for Heun's "k2"
};

// Stack of LinOSSCells + final-step classifier.
class LinOSSModel : public Layer {
public:
    LinOSSModel(int d_input, int d_output, int d_model, int num_layers, int d_osc,
                LinOSSType type = LinOSSType::IMPLICIT_MIDPOINT);

    Tensor forward(const Tensor& input) override;       // (T, d_input) → (d_output,)
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LinOSSModel"; }

    int num_layers() const { return static_cast<int>(cells_.size()); }

private:
    std::vector<std::unique_ptr<LinOSSCell>> cells_;
    std::vector<std::unique_ptr<class Dense>> proj_in_;   // d_input → d_model (first layer only)
    std::vector<std::unique_ptr<class Dense>> proj_out_;  // d_model → d_model (between layers)
    std::unique_ptr<class Dense> classifier_;             // d_model → d_output

    Tensor last_input_;      // (T, d_input)
    std::vector<Tensor> last_embedded_; // T-list of per-step activations for BPTT
};

#endif // LINOSS_H