#ifndef GROUP_CONV_H
#define GROUP_CONV_H

#include "../../core/layer.h"
#include "../convolutions/conv_layer.h"
#include <vector>
#include <stdexcept>

// Grouped Convolution: split input/output channels into num_groups groups,
// apply independent Conv2D per group, then concatenate outputs.
// Input:  (N, in_channels * H * W)
// Output: (N, out_channels * H_out * W_out)
// where in_channels and out_channels must be divisible by num_groups.
class GroupConv : public Layer {
public:
    // Constructor
    GroupConv(int in_channels, int out_channels, int kernel_size,
              int num_groups, int stride = 1, int padding = 0,
              int dilation = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "GroupConv"; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int num_groups_;
    int stride_;
    int padding_;
    int dilation_;
    int H_;          // input spatial height
    int W_;          // input spatial width
    int H_out_;
    int W_out_;
    int channels_per_group_;
    int out_channels_per_group_;

    std::vector<Conv2D> group_convs_;  // one Conv2D per group
    Tensor last_input_;
    std::vector<Tensor> last_group_outputs_; // cached outputs per group
};

#endif