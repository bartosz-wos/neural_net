#include <iostream>
#include <cmath>
#include <iomanip>
#include "nn/utils/focal_loss.h"
#include "nn/core/tensor.h"

int main() {
    std::cout << std::setprecision(12);
    
    FocalLoss loss(2.0, 1.0);
    
    // Batch=1 for simplicity
    Tensor logits(1, 3);
    logits[0][0] = 1.0;
    logits[0][1] = 0.5;
    logits[0][2] = -0.3;
    
    Tensor targets(1, 1);
    targets[0][0] = 0;
    
    // Forward pass
    Tensor result = loss.forward(logits, targets);
    std::cout << "Loss = " << result[0][0] << "\n";
    
    // Backward
    Tensor grad = loss.backward(logits, targets);
    std::cout << "Analytical grad: " << grad[0][0] << " " << grad[0][1] << " " << grad[0][2] << "\n";
    
    // Numerical gradient for z_0
    double eps = 1e-3;
    double orig_z0 = logits[0][0];
    
    logits[0][0] = orig_z0 + eps;
    Tensor loss_plus = loss.forward(logits, targets);
    
    logits[0][0] = orig_z0 - eps;
    Tensor loss_minus = loss.forward(logits, targets);
    
    logits[0][0] = orig_z0;
    
    double grad_num = (loss_plus[0][0] - loss_minus[0][0]) / (2.0 * eps);
    std::cout << "Numerical grad for z_0: " << grad_num << "\n";
    std::cout << "Diff: " << std::abs(grad_num - grad[0][0]) << "\n";
    
    return 0;
}
