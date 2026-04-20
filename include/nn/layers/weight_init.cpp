#include "weight_init.h"
#include <random>

static std::mt19937 rng_(42);

double WeightInitializer::xavier_limit(size_t fan_in, size_t fan_out) {
    return std::sqrt(6.0 / (fan_in + fan_out));
}

double WeightInitializer::kaiming_limit(size_t fan_in) {
    return std::sqrt(2.0 / fan_in);
}

void WeightInitializer::xavier(Tensor& w, size_t fan_in, size_t fan_out,
                               const std::string& activation) {
    bool is_relu = (activation == "relu" || activation == "leaky");
    double limit = is_relu ? std::sqrt(2.0 / fan_in) : xavier_limit(fan_in, fan_out);
    uniform(w, limit);
}

void WeightInitializer::kaiming(Tensor& w, size_t fan_in, const std::string& activation) {
    double limit = kaiming_limit(fan_in);
    if (activation == "leaky") limit *= std::sqrt(2.0); // adjustment for LeakyReLU
    uniform(w, limit);
}

void WeightInitializer::normal(Tensor& w, double mean, double std) {
    std::normal_distribution<double> dist(mean, std);
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            w[i][j] = dist(rng_);
}

void WeightInitializer::uniform(Tensor& w, double limit) {
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            w[i][j] = dist(rng_);
}

void WeightInitializer::initialize(Layer* layer, Strategy strategy,
                                   const std::string& activation) {
    Dense* dense = dynamic_cast<Dense*>(layer);
    if (!dense) return;

    Tensor& w = dense->weights;
    size_t fan_in = w.cols;
    size_t fan_out = w.rows;

    switch (strategy) {
        case Strategy::Xavier:
            xavier(w, fan_in, fan_out, activation);
            break;
        case Strategy::KaimingHe:
            kaiming(w, fan_in, activation);
            break;
        case Strategy::Normal:
            normal(w);
            break;
        case Strategy::Uniform:
            uniform(w);
            break;
    }
}