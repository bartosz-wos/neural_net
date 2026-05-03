#ifndef LOOKAHEAD_H
#define LOOKAHEAD_H

#include "optimizer.h"
#include "../core/model.h"
#include "../core/layer.h"
#include <map>
#include <vector>
#include <memory>

class Lookahead : public Optimizer {
public:
    explicit Lookahead(Optimizer* inner, int k = 6, double alpha = 0.5);
    void step(Model& model) override;

    Optimizer* inner() const { return inner_.get(); }
    int get_k() const { return k_; }
    double get_alpha() const { return alpha_; }

private:
    void sync_slow_weights(Model& model);
    void update_slow_weights(Model& model);

    std::unique_ptr<Optimizer> inner_;
    int k_ = 6;
    double alpha_ = 0.5;
    int steps_ = 0;
    // maps layer address -> vector of slow (checkpointed) parameter copies
    std::map<void*, std::vector<Tensor>> slow_weights_;
};

#endif
