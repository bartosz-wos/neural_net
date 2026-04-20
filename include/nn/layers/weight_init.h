#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include "../core/layer.h"

// Weight initialization strategies for different layer types and activations.
// Usage: WeightInitializer::xavier_for_dense(weights, fan_in, fan_out, activation="relu")
class WeightInitializer {
public:
    enum class Strategy { Xavier, KaimingHe, Normal, Uniform };

    // Xavier/Glorot for sigmoid/tanh — works with weights where fan_in and fan_out are known
    static void xavier(Tensor& w, size_t fan_in, size_t fan_out,
                       const std::string& activation = "");

    // Kaiming/He for ReLU/LeakyReLU — use fan_in and activation="relu"
    static void kaiming(Tensor& w, size_t fan_in, const std::string& activation = "relu");

    // Normal distribution with given mean and std
    static void normal(Tensor& w, double mean = 0.0, double std = 0.01);

    // Uniform distribution in [-limit, limit]
    static void uniform(Tensor& w, double limit = 0.01);

    // Auto-detect strategy based on layer type and apply
    static void initialize(Layer* layer, Strategy strategy = Strategy::Xavier,
                           const std::string& activation = "relu");

private:
    static double xavier_limit(size_t fan_in, size_t fan_out);
    static double kaiming_limit(size_t fan_in);
};

#endif