#ifndef NN_GENERATIVE_FLOW_MATCHING_H
#define NN_GENERATIVE_FLOW_MATCHING_H

#include "../../core/tensor.h"
#include "../../core/layer.h"
#include "../../activations/activations.h"
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <utility>

// ============================================================================
// Flow Matching — Lipman, Chen, Benjamini, Nickel, Le, Lipman (ICLR 2023)
//   "Flow Matching for Generative Modeling" (https://arxiv.org/abs/2210.02747)
//
// Three classes below:
//   * `FlowMatching` — base loss / trainer.
//       Forward path:   x_t = (1 - t) * x0 + t * x1
//       Target vel:     v_target = x1 - x0
//       Loss:           mean(|| v_pred(x_t, t) - v_target ||²)
//   * `ConditionalFlowMatching` — Lipman §3.2 (informally the σ_min path)
//       Forward path:   x_t = α_t * x1 + (1 - α_t) * x0,
//                       α_t = 1 - (1 - σ_min) * t
//       Target vel:     v_target = x1 - (1 - σ_min) * x0
//       Loss:           same MSE form
//       The σ_min > 0 detour breaks the determinism of the marginal path so
//       the network cannot trivially solve the loss by predicting a constant.
//   * `OptimalTransportFlowMatching` — Tong et al. (ICML 2024)
//       Same loss as FlowMatching but we permute the rows of x1 (and y1 if
//       class-conditional) using a minibatch OT assignment before computing
//       x_t and v_target. The OT cost is squared-L2; the assignment is greedy
//       nearest-neighbour per row (sufficient for training, O(N²)).
//
// ============================================================================

// =============================================================================
// GaussianMixture2D — 2-cluster Gaussian pair data for FM training tests
// =============================================================================
class GaussianMixture2D {
public:
    GaussianMixture2D(int n_per_cluster = 64, int dim = 2,
                      double scale = 1.0, double separation = 4.0,
                      unsigned seed = 42);

    // Returns (x0, x1), each of shape (2*n_per_cluster, dim).
    // x0 is drawn from a mixture of two Gaussians centred at (-sep/2, -sep/2)
    // and (+sep/2, +sep/2); x1 is drawn from the same mixture but with each
    // row independently resampled (so pairing is "random", suitable for OT FM).
    std::pair<Tensor, Tensor> sample_pair();

    int n_per_cluster() const { return n_per_cluster_; }
    int dim() const { return dim_; }

private:
    int n_per_cluster_;
    int dim_;
    double scale_;
    double separation_;
    unsigned seed_;
};

// =============================================================================
// FMTimeEmbedding — sinusoidal time embedding (Vaswani-style)
// =============================================================================
class FMTimeEmbedding {
public:
    explicit FMTimeEmbedding(int hidden_dim = 64);

    // t is a scalar in [0, 1]. Returns shape (1, hidden_dim).
    Tensor forward(double t) const;

    int hidden_dim() const { return hidden_dim_; }

private:
    int hidden_dim_;
};

// =============================================================================
// ClassEmbedding — learned per-class embedding
// =============================================================================
class ClassEmbedding {
public:
    ClassEmbedding(int num_classes, int hidden_dim = 64,
                   unsigned seed = 0);

    // Single-label path: returns shape (1, hidden_dim).
    Tensor forward(int label) const;

    // One-hot batch path: input is (N, num_classes), returns (N, hidden_dim).
    Tensor forward(const Tensor& one_hot) const;

    int num_classes() const { return num_classes_; }
    int hidden_dim() const { return hidden_dim_; }

    // Direct access to the embedding matrix for test inspection.
    Tensor weights() const { return embeddings_; }

private:
    int num_classes_;
    int hidden_dim_;
    Tensor embeddings_;  // (num_classes, hidden_dim)
};

// =============================================================================
// FlowMatchingNet — the velocity-prediction network
//
//   Input  shape: (N, data_dim + 1 + num_classes)
//                   [ x_t (data_dim) | t (1) | y_onehot (num_classes) ]
//   Hidden:       Dense(data_dim + 1 + num_classes → hidden_dim)
//                 SiLU activation
//                 Dense(hidden_dim → data_dim)
//                 Output is residual:   v_pred = W2·act(W1·in + b1) + b2 + in[:, :data_dim]
//   Output shape: (N, data_dim)
// =============================================================================
class FlowMatchingNet : public Layer {
public:
    FlowMatchingNet(int data_dim, int hidden_dim = 64, int num_classes = 0,
                    unsigned seed = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    Tensor get_weights() const override;
    Tensor get_gradients() const override;

    std::string name() const override { return "FlowMatchingNet"; }

    int data_dim() const { return data_dim_; }
    int hidden_dim() const { return hidden_dim_; }
    int num_classes() const { return num_classes_; }
    int input_dim() const { return input_dim_; }

    // Internal Dense accessors for tests / FD-grad probing.
    Dense* dense1() { return dense1_.get(); }
    Dense* dense2() { return dense2_.get(); }

private:
    int data_dim_;
    int hidden_dim_;
    int num_classes_;
    int input_dim_;  // data_dim + 1 + num_classes

    std::unique_ptr<Dense> dense1_;
    std::unique_ptr<Dense> dense2_;

    Tensor last_input_;     // (N, input_dim) — the concatenated input
    Tensor last_hidden_;    // (N, hidden_dim) — pre-activation of layer 1
    Tensor last_act_;       // (N, hidden_dim) — SiLU(hidden)
    Tensor last_residual_;  // (N, data_dim) — input[:, :data_dim] for residual
    Tensor last_velocity_;  // (N, data_dim) — final output
};

// =============================================================================
// FlowMatching — base class (also used for "pure" FM with σ_min = 0 and no OT)
//
// Construct with (data_dim, hidden_dim, num_classes, sigma_min, use_ot).
//   - sigma_min = 0 (default) and use_ot = false (default) → standard FM
//   - sigma_min > 0 and use_ot = false → use ConditionalFlowMatching
//   - sigma_min = 0 and use_ot = true  → use OptimalTransportFlowMatching
//   - sigma_min > 0 and use_ot = true  → both (advanced)
//
// The class itself can be instantiated directly with any combination; the
// Conditional / OT subclasses below are convenience aliases.
// =============================================================================
class FlowMatching {
public:
    FlowMatching(int data_dim, int hidden_dim = 64, int num_classes = 0,
                 double sigma_min = 0.0, bool use_ot = false,
                 unsigned seed = 0);

    virtual ~FlowMatching() = default;

    // Compute the FM loss. x0, x1 are (N, data_dim).
    // Returns a (1, 1) tensor containing the mean-over-batch MSE.
    virtual Tensor forward(const Tensor& x0, const Tensor& x1);

    // Forward with a FIXED per-row t (for FD gradient testing).
    // t_col is (N, 1) and must satisfy 0 <= t[i][0] <= 1.
    virtual Tensor forward_with_t(const Tensor& x0, const Tensor& x1, const Tensor& t_col);

    // Backward pass — populates the velocity-net's grad_weights / grad_bias.
    // Returns the gradient of the loss w.r.t. the net input (typically not used).
    virtual Tensor backward();

    // Apply one SGD-style update step to the velocity-net parameters.
    // (Caller can also use net.update_weights(lr) directly.)
    virtual void update_weights(double lr);

    // Euler ODE sampler. class_labels.size() must equal n_samples if
    // num_classes > 0 (else throw). Returns last_samples() and also caches it.
    virtual void sample(int n_samples, int n_steps = 50,
                        const std::vector<int>& class_labels = {},
                        unsigned seed = 0);

    // ----- Accessors -----
    FlowMatchingNet& net() { return *net_; }
    const FlowMatchingNet& get_net() const { return *net_; }
    FlowMatchingNet& get_net_mut() { return *net_; }

    Tensor last_loss() const { return last_loss_; }
    Tensor last_x_t() const { return last_x_t_; }
    Tensor last_v_target() const { return last_v_target_; }
    Tensor last_v_pred() const { return last_v_pred_; }
    Tensor last_t_vec() const { return last_t_vec_; }
    Tensor last_samples() const { return last_samples_; }
    Tensor last_x1_perm() const { return last_x1_perm_; }
    Tensor last_one_hot() const { return last_one_hot_; }

    int data_dim() const { return data_dim_; }
    int hidden_dim() const { return hidden_dim_; }
    int num_classes() const { return num_classes_; }
    double sigma_min() const { return sigma_min_; }
    bool use_ot() const { return use_ot_; }

    // Random sampling state for t in [0, 1].
    std::mt19937& rng() { return rng_; }

    // Static utility: build a one-hot (N, K) tensor from a label vector.
    static Tensor one_hot(const std::vector<int>& labels, int num_classes);

protected:
    int data_dim_;
    int hidden_dim_;
    int num_classes_;
    double sigma_min_;
    bool use_ot_;
    unsigned seed_;

    std::unique_ptr<FlowMatchingNet> net_;
    std::mt19937 rng_;

    Tensor last_loss_;
    Tensor last_x_t_;       // (N, data_dim)
    Tensor last_v_target_;  // (N, data_dim)
    Tensor last_v_pred_;    // (N, data_dim)
    Tensor last_t_vec_;     // (N, 1)
    Tensor last_samples_;   // (n_samples, data_dim)
    Tensor last_x1_perm_;   // (N, data_dim) — after OT permutation
    Tensor last_one_hot_;   // (N, num_classes) — empty if num_classes == 0

    // Helpers
    Tensor interpolate(const Tensor& x0, const Tensor& x1, const Tensor& t_col);
    Tensor target_velocity(const Tensor& x0, const Tensor& x1);
    Tensor greedy_ot_assign(const Tensor& x0, const Tensor& x1);
    Tensor build_net_input(const Tensor& x_t, const Tensor& t_col);
};

// =============================================================================
// ConditionalFlowMatching — Lipman §3.2 with σ_min > 0
// =============================================================================
class ConditionalFlowMatching : public FlowMatching {
public:
    ConditionalFlowMatching(int data_dim, int hidden_dim = 64,
                            int num_classes = 0, double sigma_min = 0.1,
                            unsigned seed = 0);

    Tensor forward(const Tensor& x0, const Tensor& x1) override;
};

// =============================================================================
// OptimalTransportFlowMatching — Tong et al. 2023 with minibatch OT
// =============================================================================
class OptimalTransportFlowMatching : public FlowMatching {
public:
    OptimalTransportFlowMatching(int data_dim, int hidden_dim = 64,
                                  int num_classes = 0, double sigma_min = 0.0,
                                  unsigned seed = 0);

    Tensor forward(const Tensor& x0, const Tensor& x1) override;
};

#endif // NN_GENERATIVE_FLOW_MATCHING_H
