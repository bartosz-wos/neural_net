#ifndef GROUP_NORM_H
#define GROUP_NORM_H

#include "../../core/layer.h"

// GroupNorm: divide channels into G groups, normalize within each group
// Paper: https://arxiv.org/abs/1803.08494
// Input: (batch, channels * height * width) — flatten CNN output
class GroupNorm : public Layer {
public:
    GroupNorm(int num_groups, int num_channels, float eps = 1e-5);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double) override {}
    // NOTE: forward(input, target) intentionally hides base forward(const Tensor&)
    Tensor get_weights() const override { return gamma_; }
    Tensor get_gradients() const override { return grad_gamma_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "GroupNorm"; }

    int num_groups() const { return num_groups_; }
    int num_channels() const { return num_channels_; }

private:
    int num_groups_;
    int num_channels_;
    float eps_;

    Tensor gamma_; // scale (num_channels, 1)
    Tensor beta_;  // shift (num_channels, 1)
    Tensor grad_gamma_;
    Tensor grad_beta_;

    Tensor last_x_;
    Tensor last_mean_;
    Tensor last_var_;
    int last_spatial_;
    bool train_mode_ = true;
};

#endif
