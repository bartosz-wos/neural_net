// Simple single-layer network to verify gradient
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    // Create a simple 1-layer network: input -> Dense(1) -> output
    // Actually, let's just test with a simple tensor operation
    // Let me manually compute gradients for a tiny network
    
    // For a 1-layer network: y = x @ W^T, L = sum(y)
    // dL/dW = sum_r x[r]^T * 1 = x^T (for single output)
    
    Tensor x(2, 2);  // batch=2, dim=2
    x[0][0] = 0.5; x[0][1] = -0.5;
    x[1][0] = 1.0; x[1][1] = 0.0;
    
    // Single Dense layer: out=1, in=2
    Tensor W(1, 2);  // 1 output, 2 inputs
    W[0][0] = -0.246;
    W[0][1] = 0.0;
    
    // Forward: y = x @ W^T
    // x is (2,2), W^T is (2,1), y is (2,1)
    Tensor y(2, 1);
    for (int r = 0; r < 2; ++r) {
        y[r][0] = x[r][0] * W[0][0] + x[r][1] * W[0][1];
    }
    cout << "y = " << y[0][0] << ", " << y[1][0] << endl;
    
    // L = sum(y)
    double L = y[0][0] + y[1][0];
    cout << "L = " << L << endl;
    
    // dL/dW[0][0] = x[0][0] + x[1][0] = 0.5 + 1.0 = 1.5
    // dL/dW[0][1] = x[0][1] + x[1][1] = -0.5 + 0.0 = -0.5
    cout << "Analytical dL/dW[0][0] = " << (x[0][0] + x[1][0]) << endl;
    cout << "Analytical dL/dW[0][1] = " << (x[0][1] + x[1][1]) << endl;
    
    // Numerical
    double eps = 1e-4;
    W[0][0] = -0.246 + eps;
    double L_plus = (x[0][0] * W[0][0] + x[0][1] * W[0][1]) + 
                    (x[1][0] * W[0][0] + x[1][1] * W[0][1]);
    W[0][0] = -0.246 - eps;
    double L_minus = (x[0][0] * W[0][0] + x[0][1] * W[0][1]) + 
                     (x[1][0] * W[0][0] + x[1][1] * W[0][1]);
    W[0][0] = -0.246;
    double num_dW00 = (L_plus - L_minus) / (2 * eps);
    cout << "Numerical dL/dW[0][0] = " << num_dW00 << endl;
    
    return 0;
}
