#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../core/layer.h"

// RMS Normalization: normalize by RMS of features, no mean subtraction, no beta
// y = (x / RMS) * gamma, where RMS = sqrt(mean(x²) + eps)
class RMSNorm : public Layer {
public:
    Tensor gamma;  // scale (features)
    double eps;
    Tensor last_rms;
    Tensor last_x;
    bool training;
    Tensor grad_gamma_;
    Tensor grad_x;

    RMSNorm(size_t features, double eps = 1e-7);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training = t; }
    std::string name() const override { return "RMSNorm"; }
};

#endif