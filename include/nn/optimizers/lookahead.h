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
    bool handles_weight_decay() const override {
        return inner_->handles_weight_decay();
    }

    Optimizer* inner() const { return inner_.get(); }
    int get_k() const { return k_; }
    double get_alpha() const { return alpha_; }
    size_t num_steps() const { return steps_; }
    bool has_state(void* layer_ptr) const {
        return slow_weights_.find(layer_ptr) != slow_weights_.end();
    }
    Tensor get_slow_weight(void* layer_ptr, size_t param_idx) const;

private:
    void initialize_slow_weights(Model& model);
    void synchronize(Model& model);

    std::unique_ptr<Optimizer> inner_;
    int k_;
    double alpha_;
    size_t steps_;
    // Maps Layer* to slow parameter copies in parameters() order.
    std::map<void*, std::vector<Tensor>> slow_weights_;
};

#endif
