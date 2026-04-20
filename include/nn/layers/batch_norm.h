#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "../core/layer.h"
#include <vector>

class BatchNorm1D : public Layer {
public:
    Tensor gamma, beta;
    double eps, momentum;
    Tensor running_mean, running_var;
    Tensor last_x, last_mean, last_var;
    Tensor grad_gamma_, grad_beta_, grad_x;
    bool training;

    BatchNorm1D(size_t features, double eps = 1e-5, double momentum = 0.1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /* learning_rate */) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return gamma; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training = t; }
};

#endif