#include "../convolutions/conv_layer.h"

// Conv2D with dilated (atrous) convolution.
// dilation_rate: spacing between kernel elements (dilation=1 → standard conv).
// Useful for semantic segmentation, detection with dilated kernels.
class DilatedConv2D : public Conv2D {
public:
    DilatedConv2D(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
                  int dilation_rate, int stride = 1, int pad = 0)
        : Conv2D(in_ch, out_ch, kH, kW, H_in, W_in,
                 stride, stride, pad, pad,
                 dilation_rate, dilation_rate)
    {}
};
