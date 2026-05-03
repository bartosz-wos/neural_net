#ifndef LR_FINDER_H
#define LR_FINDER_H

#include "../core/tensor.h"
#include "../core/model.h"
#include "../optimizers/optimizer.h"
#include <vector>
#include <utility>

// Learning Rate Finder — implements Leslie Smith's LR Range Test.
// Call run() to get loss trajectory, then use suggested_lr() to get optimal LR.
class LRFinder {
public:
    // min_lr: starting learning rate (default 1e-7)
    // max_lr: ceiling learning rate (default 1.0 or 10.0)
    // num_steps: number of steps in the range test
    // beta: smoothing factor for exponential moving average of loss (0.98 = Smith default)
    LRFinder(double min_lr = 1e-7, double max_lr = 1.0,
             int num_steps = 100, double beta = 0.98);

    // Run the LR range test on a model.
    // X: input tensor (rows=samples, cols=features)
    // y: target tensor (rows=samples, cols=targets)
    // opt: optimizer to use (will be cloned internally)
    // Returns: vector of (learning_rate, smoothed_loss) pairs
    std::vector<std::pair<double, double>> run(
        Model& model,
        const Tensor& X,
        const Tensor& y,
        Optimizer& opt
    );

    // Get suggested LR: the point with the highest loss derivative (steepest descent)
    double suggested_lr() const;

    // Get the LR at minimum smoothed loss
    double lr_at_min_loss() const;

    // Smoothing parameter beta getter/setter
    double beta() const { return beta_; }
    void set_beta(double b) { beta_ = b; }

private:
    double min_lr_, max_lr_;
    int num_steps_;
    double beta_;
    std::vector<std::pair<double, double>> history_; // (lr, smoothed_loss)
};

#endif