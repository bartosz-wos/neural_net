#ifndef CLIP_GRAD_NORM_H
#define CLIP_GRAD_NORM_H

#include "../core/tensor.h"
#include <vector>
#include <cmath>

// clip_grad_norm_: clips gradients by global norm.
// Returns the total norm before clipping.
// Usage: double total_norm = clip_grad_norm_(params, max_norm);
// After calling, gradients in each parameter are scaled in-place.
double clip_grad_norm_(const std::vector<Tensor*>& params, double max_norm);

#endif