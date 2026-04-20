#include "mixup_cutmix.h"
#include <algorithm>
#include <cmath>

MixupResult Mixup::apply(const Tensor& X, const Tensor& y) {
    std::uniform_real_distribution<float> dist(0.3f, 0.7f);

    int N = (int)X.rows;
    int total_features = (int)X.cols;

    float lambda = dist(rng_);

    // Pick random index for mixing partner
    std::uniform_int_distribution<int> idx_dist(0, N - 1);

    Tensor x_mixed = X; // each row is one sample
    Tensor y_a = y;
    Tensor y_b = y;

    for (int n = 0; n < N; n++) {
        int idx = idx_dist(rng_);
        for (int f = 0; f < total_features; f++) {
            x_mixed[n][f] = lambda * X[n][f] + (1 - lambda) * X[idx][f];
        }
    }

    MixupResult r;
    r.x = x_mixed;
    r.y_a = y_a;
    r.y_b = y_b;
    r.lambda = lambda;
    return r;
}

CutMixResult CutMix::apply(const Tensor& X, const Tensor& y, int C, int H, int W) {
    std::uniform_real_distribution<float> dist(0.3f, 0.7f);

    int N = (int)X.rows;
    int spatial = C * H * W;

    float lambda = dist(rng_);

    // Sample bounding box area = lambda * H * W
    float area = lambda * H * W;
    int target_h = std::uniform_int_distribution<int>(1, (int)std::sqrt(area) * 2)(rng_);
    int target_w = (int)(area / target_h);
    target_h = std::min(target_h, H);
    target_w = std::min(target_w, W);

    // Sample top-left corner
    int y_start = std::uniform_int_distribution<int>(0, H - target_h)(rng_);
    int x_start = std::uniform_int_distribution<int>(0, W - target_w)(rng_);

    // Pick random index for source
    std::uniform_int_distribution<int> idx_dist(0, N - 1);

    Tensor x_mixed = X;
    Tensor y_a = y;
    Tensor y_b = y;

    for (int n = 0; n < N; n++) {
        int idx = idx_dist(rng_);
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int flat_idx = c * H * W + h * W + w;
                    if (h >= y_start && h < y_start + target_h && w >= x_start && w < x_start + target_w) {
                        int src_flat = c * H * W + h * W + w;
                        x_mixed[n][flat_idx] = X[idx][src_flat];
                    }
                }
            }
        }
    }

    CutMixResult r;
    r.x = x_mixed;
    r.y_a = y_a;
    r.y_b = y_b;
    r.lambda = lambda;
    return r;
}

AutoAugmentResult AutoAugment::apply(const Tensor& X, const Tensor& y, int C, int H, int W) {
    std::uniform_int_distribution<int> type_dist(0, 1);
    int type = type_dist(rng_);

    AutoAugmentResult r;
    if (type == 0) {
        MixupResult m = mixup_.apply(X, y);
        r.x = m.x;
        r.y_a = m.y_a;
        r.y_b = m.y_b;
        r.lambda = m.lambda;
        r.type = AutoAugmentResult::MIXUP;
    } else {
        CutMixResult c = cutmix_.apply(X, y, C, H, W);
        r.x = c.x;
        r.y_a = c.y_a;
        r.y_b = c.y_b;
        r.lambda = c.lambda;
        r.type = AutoAugmentResult::CUTMIX;
    }
    return r;
}
