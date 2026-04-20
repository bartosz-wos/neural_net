#include "optimizer.h"
#include "layer.h"
#include <cmath>

class SGD : public Optimizer {
public:
    double lr;
    SGD(double lr) : lr(lr) {}
    void step(Model& model) override;
};

class Adam : public Optimizer {
public:
    double lr;
    double beta1, beta2, epsilon;
    int t; // timestep

    // For each layer we'll store first and second moment vectors for weights and bias
    // We'll use maps or augment layer? Since we don't have per-layer storage,
    // we'll use a simple approach: during step, we'll allocate and store in optimizer
    // using dynamic casts. Not super clean but works for demo.

    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : lr(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void step(Model& model) override;
};

#endif // OPTIMIZER_H
