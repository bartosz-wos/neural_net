#include "mixup_cutmix.h"
#include <algorithm>
#include <cmath>
#include <random>

MixupResult Mixup::apply(const Tensor& X, const Tensor& y) {
    // Sample lambda from Beta(alpha, alpha) — standard MixUp formulation
    // Beta(a,b) sampled as X/(X+Y) with X~Gamma(a,1), Y~Gamma(b,1)
    std::gamma_distribution<double> dist_alpha(alpha_);
    double g1 = dist_alpha(rng_);
    double g2 = dist_alpha(rng_);
    float lambda = static_cast<float>(g1 / (g1 + g2));

    int N = (int)X.rows;
    int total_features = (int)X.cols;

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
    // Sample lambda from Beta(alpha, alpha) — same as MixUp
    std::gamma_distribution<double> g_alpha(alpha_);
    double g1 = g_alpha(rng_);
    double g2 = g_alpha(rng_);
    float lambda = static_cast<float>(g1 / (g1 + g2));

    int N = (int)X.rows;

    // Sample bounding box dimensions from lambda proportional area
    // Standard CutMix: w = sqrt(λ) * W, h = sqrt(λ) * H (aspect ratio preserved)
    float sqrt_lambda = std::sqrt(lambda);
    int target_h = std::max(1, (int)(sqrt_lambda * H));
    int target_w = std::max(1, (int)(sqrt_lambda * W));
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
