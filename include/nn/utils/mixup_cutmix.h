#ifndef MIXUP_CUTMIX_H
#define MIXUP_CUTMIX_H

#include <random>
#include <vector>
#include "../core/tensor.h"

// MixUp: linearly interpolate between two samples and their labels
// Paper: https://arxiv.org/abs/1710.09412
// CutMix: cut and paste patches between samples
// Paper: https://arxiv.org/abs/1905.04899

struct MixupResult {
    Tensor x;       // augmented input
    Tensor y_a;     // original labels (for loss)
    Tensor y_b;     // mixed labels (for loss)
    float lambda;   // interpolation factor
};

class Mixup {
public:
    Mixup(float alpha = 0.2) : alpha_(alpha), rng_(std::random_device{}()) {}

    MixupResult apply(const Tensor& X, const Tensor& y);

private:
    float alpha_;
    std::mt19937 rng_;
};

struct CutMixResult {
    Tensor x;
    Tensor y_a;
    Tensor y_b;
    float lambda;
};

class CutMix {
public:
    CutMix(float alpha = 1.0) : alpha_(alpha), rng_(std::random_device{}()) {}

    CutMixResult apply(const Tensor& X, const Tensor& y, int C, int H, int W);

private:
    float alpha_;
    std::mt19937 rng_;
};

// Combined augmentation that randomly chooses MixUp or CutMix
struct AutoAugmentResult {
    Tensor x;
    Tensor y_a;
    Tensor y_b;
    float lambda;
    enum Type { MIXUP, CUTMIX } type;
};

class AutoAugment {
public:
    AutoAugment(float mixup_alpha = 0.2, float cutmix_alpha = 1.0f)
        : mixup_(mixup_alpha), cutmix_(cutmix_alpha), rng_(std::random_device{}()) {}

    AutoAugmentResult apply(const Tensor& X, const Tensor& y, int C, int H, int W);

private:
    Mixup mixup_;
    CutMix cutmix_;
    std::mt19937 rng_;
};

#endif
