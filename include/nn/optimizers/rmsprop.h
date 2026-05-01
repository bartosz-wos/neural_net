#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"
#include <map>
#include <vector>

class RMSProp : public Optimizer {
public:
    double lr, alpha, epsilon, weight_decay;
    int t;

    RMSProp(double lr = 0.001, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0);
    void step(Model& model) override;
    bool handles_weight_decay() const override { return weight_decay > 0; }

private:
    std::map<void*, std::vector<Tensor>> eg_state;  // E[g²] running average
};

#endif  // RMSPROP_H
