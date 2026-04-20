#ifndef OPTIMIZER_EXTENDED_H
#define OPTIMIZER_EXTENDED_H

#include <memory>
#include "optimizer.h"
#include <map>

class RMSprop : public Optimizer {
public:
    double lr, alpha, eps;
    RMSprop(double lr = 0.01, double alpha = 0.99, double eps = 1e-8)
        : lr(lr), alpha(alpha), eps(eps) {}
    void step(Model& model) override;
private:
    std::map<void*, std::vector<Tensor>> cache_;
};

class AdamW : public Optimizer {
public:
    double lr, beta1, beta2, epsilon, weight_decay;
    int t;
    AdamW(double lr = 0.001, double b1 = 0.9, double b2 = 0.999,
          double eps = 1e-8, double wd = 0.01)
        : lr(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(1) {}
    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }
private:
    std::map<void*, std::vector<Tensor>> m_state;
    std::map<void*, std::vector<Tensor>> v_state;
};

class SGDNesterov : public Optimizer {
public:
    double lr, momentum;
    SGDNesterov(double lr = 0.01, double momentum = 0.9)
        : lr(lr), momentum(momentum) {}
    void step(Model& model) override;
private:
    std::map<void*, std::vector<Tensor>> velocity_;
};

// WeightDecay: optimizer wrapper that adds L2 regularization.
// Wraps any optimizer and adds weight_decay * weight to gradients before the inner step.
class WeightDecay : public Optimizer {
public:
    WeightDecay(Optimizer* inner, double weight_decay)
        : inner_(inner), weight_decay_(weight_decay) {}
    void step(Model& model) override;
    Optimizer* inner() const { return inner_.get(); }
    double wd() const { return weight_decay_; }
private:
    std::unique_ptr<Optimizer> inner_;
    double weight_decay_;
};

#endif