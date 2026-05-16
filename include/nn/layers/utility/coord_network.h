#ifndef COORD_NETWORK_H
#define COORD_NETWORK_H

#include "../../core/layer.h"
#include "../../core/model.h"
#include "fourier_features.h"

// CoordinateNetwork: an MLP that takes Fourier-encoded coordinates and outputs scalar or vector.
// Used for implicit neural representations (SDF, occupancy fields) and learning functions
// on 2D/3D spatial domains.
//
// Architecture:
//   coords (N, coord_dim) -> FourierFeatures -> (N, 2*F) -> MLP -> (N, output_dim)
class CoordinateNetwork : public Layer {
public:
    // coord_dim: coordinate dimension (2 for 2D, 3 for 3D)
    // num_frequencies: number of Fourier frequency vectors
    // hidden_sizes: hidden layer dimensions of the MLP
    // output_dim: output dimension (default 1 for scalar SDF)
    // learn_frequencies: if true, use LearnedFourierFeatures; else GaussianFourierFeatures
    CoordinateNetwork(size_t coord_dim, size_t num_frequencies,
                      const std::vector<size_t>& hidden_sizes,
                      size_t output_dim = 1,
                      bool learn_frequencies = false);

    Tensor forward(const Tensor& coords) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> all_gradients();
    std::string name() const override { return "CoordinateNetwork"; }

    // Access sub-layer
    FourierFeatures& fourier() { return *fourier_; }
    const FourierFeatures& fourier() const { return *fourier_; }

private:
    std::unique_ptr<FourierFeatures> fourier_;
    std::vector<std::unique_ptr<Dense>> mlp_layers_;
    size_t coord_dim_;
    size_t output_dim_;
    std::vector<size_t> hidden_sizes_;
    Tensor last_fourier_out_;
    std::vector<Tensor> last_layer_inputs_;
};

// SIREN: Implicit Neural Representations with Periodic Activation Functions
// Uses sin activation after first layer + special weight initialization
// From "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann et al., 2020)
class SIREN : public Layer {
public:
    SIREN(size_t coord_dim, const std::vector<size_t>& hidden_sizes,
          size_t output_dim = 1, double omega0 = 30.0);

    Tensor forward(const Tensor& coords) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "SIREN"; }

private:
    void init_weights_siren();
    std::vector<std::unique_ptr<Dense>> layers_;
    double omega0_;
    size_t coord_dim_;
    size_t output_dim_;
    std::vector<Tensor> last_activations_;
    std::vector<Tensor> last_layer_inputs_;
    std::vector<Tensor> last_pre_sin_;  // Dense outputs before sin activation
};

#endif