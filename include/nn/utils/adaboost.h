#ifndef ADABOOST_H
#define ADABOOST_H

#include "tabular_ensemble.h"
#include <vector>

// AdaBoost — boosting with sequential sample reweighting and weighted vote.
class AdaBoost {
public:
    AdaBoost(size_t n_learners = 50);
    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    double score(const Tensor& X, const Tensor& y) const;

    size_t n_learners() const { return stumps_.size(); }

private:
    size_t n_learners_;
    std::vector<DecisionStump> stumps_;
    std::vector<double> alphas_;
};

#endif