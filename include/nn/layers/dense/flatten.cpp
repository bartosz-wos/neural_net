#include "flatten.h"

Tensor Flatten::forward(const Tensor& input) {
    // Just return as-is, the data format already flattens spatial dims
    return input;
}

Tensor Flatten::backward(const Tensor& grad_output, double /* learning_rate */) {
    return grad_output;
}