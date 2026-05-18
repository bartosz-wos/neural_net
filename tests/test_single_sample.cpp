// Single sample gradient test
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    srand(42);
    
    WGANDiscriminator critic(2, 8, 2);
    
    Dense& w0 = critic.layer(0);
    
    // Just use ONE sample
    Tensor input(1, 2);
    input[0][0] = 0.5; input[0][1] = -0.5;
    
    double alpha = 0.01;
    
    // Manual forward
    // pre0 = input @ W0^T
    // h0 = leakyrelu(pre0)
    // pre1 = h0 @ W1^T
    // h1 = leakyrelu(pre1)
    // pre2 = h1 @ W2^T
    // D = pre2
    
    // Manual gradient for just layer 0
    // L = D (scalar for batch=1)
    // dL/dpre2 = [[1]]
    // dL/dh1 = dL/dpre2 @ W2 = [[W2[0][0], ..., W2[0][7]]]
    // dL/dpre1 = dL/dh1 * leakyrelu'(pre1) = dL/dh1 * (1 or alpha)
    // dL/dh0 = dL/dpre1 @ W1
    // dL/dpre0 = dL/dh0 * leakyrelu'(pre0)
    // dL/dW0[j][i] = dL/dpre0[0][j] * input[0][i]
    
    double orig_w00 = w0.weights[0][0];
    double eps = 1e-4;
    
    // Numerical gradient for single sample
    w0.weights[0][0] = orig_w00 + eps;
    critic.reset_cached_inputs();
    Tensor out_p = critic.forward(input);
    double loss_plus = out_p[0][0];
    
    w0.weights[0][0] = orig_w00 - eps;
    critic.reset_cached_inputs();
    Tensor out_m = critic.forward(input);
    double loss_minus = out_m[0][0];
    
    w0.weights[0][0] = orig_w00;
    
    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    cout << "Single sample numerical gradient w0[0][0] = " << num_grad << endl;
    
    // Analytical gradient
    critic.zero_grad();
    Tensor grad_out(1, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(input);
    critic.backward_from(grad_out);
    
    cout << "Single sample analytical gradient w0[0][0] = " << w0.grad_weights[0][0] << endl;
    
    double rel_err = abs(num_grad - w0.grad_weights[0][0]) / (abs(num_grad) + abs(w0.grad_weights[0][0]) + 1e-8);
    cout << "Relative error = " << rel_err << endl;
    
    return 0;
}
