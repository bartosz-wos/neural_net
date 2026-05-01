#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include <vector>
#include <string>
#include "../optimizers/optimizer.h"

class Model {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    // Gradient accumulation support
    int accum_steps_ = 1;    // 1 = no accumulation (regular training)
    int step_count_ = 0;     // global step counter

    void set_accumulate_steps(int n);       // Number of steps to accumulate before update
    int get_accumulate_steps() const { return accum_steps_; }
    double accumulated_grad_norm();          // Returns L2 norm of summed gradients
    void apply_gradAccum(Optimizer& opt);    // Divide accumulated grads by accum_steps_, update weights, zero grads
    bool should_update() const;              // returns (step_count_ % accum_steps_ == 0)
    void step();                             // increment step_count_; returns true when should_update()
    int step_count() const { return step_count_; }

    Model() = default;
    void add_layer(Layer* layer);  // takes ownership
    Tensor forward(const Tensor& input);
    Tensor backward(Tensor grad, double learning_rate);
    void train(const Tensor& X, const Tensor& y, double learning_rate, int epochs);
    void train(const Tensor& X, const Tensor& y, Optimizer& opt, int epochs);
    // Mini-batch training with DataLoader
    void train(const Tensor& X, const Tensor& y, Optimizer& opt, int epochs,
               size_t batch_size, bool shuffle = true, unsigned seed = 42);
    void train_cross_entropy(const Tensor& X, const Tensor& y, double learning_rate, int epochs);
    double evaluate(const Tensor& X, const Tensor& y);
    void save(const std::string& path);
    static Model load(const std::string& path);
    void summary() const;  // prints layer name, output shape, param count
    size_t param_count() const;  // total number of parameters
};

// Convenience constructors
Model create_mlp(const std::vector<size_t>& layer_sizes, const std::string& activation = "relu");
Model create_mlp_ex(const std::vector<size_t>& sizes, const std::string& activation = "relu", bool batchnorm = false, double dropout = 0.0);

#endif
