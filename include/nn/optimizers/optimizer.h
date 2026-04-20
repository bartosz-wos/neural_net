#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <map>

class Model;
class Tensor;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Model& model) = 0;
    // Clip all gradients to max_norm (L2 norm). Returns the total gradient norm before clipping.
    static double clip_grad_norm_(Model& model, double max_norm);
};

class SGD : public Optimizer {
public:
    double lr;
    explicit SGD(double lr) : lr(lr) {}
    void step(Model& model) override;
};

class Adam : public Optimizer {
public:
    double lr, beta1, beta2, epsilon;
    int t;

    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-7);
    void step(Model& model) override;

private:
    // Per-layer state: maps Layer* to vector of first/second moment Tensors
    std::map<void*, std::vector<Tensor>> m_state;
    std::map<void*, std::vector<Tensor>> v_state;
};

#endif
