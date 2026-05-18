// Detailed trace of gradient computation
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    WGANDiscriminator critic(2, 8, 2);
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    // Get weights
    Dense& w0 = critic.layer(0);
    Dense& w1 = critic.layer(1);
    Dense& w2 = critic.layer(2);
    
    cout << "w0.weights[0][0] = " << w0.weights[0][0] << endl;
    cout << "w1.weights[0][0] = " << w1.weights[0][0] << endl;
    cout << "w2.weights[0][0] = " << w2.weights[0][0] << endl;
    
    // Compute pre0 = real @ w0.weights^T
    Tensor pre0(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            double sum = 0.0;
            for (int j = 0; j < 2; ++j) {
                sum += real[r][j] * w0.weights[m][j];
            }
            pre0[r][m] = sum;
        }
    }
    
    cout << "\npre0 (real):" << endl;
    for (int r = 0; r < 2; ++r) {
        cout << "  [" << r << "]: " << pre0[r][0] << ", " << pre0[r][1] << ", ..." << endl;
    }
    
    // Compute pre0 for fake
    Tensor pre0_fake(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            double sum = 0.0;
            for (int j = 0; j < 2; ++j) {
                sum += fake[r][j] * w0.weights[m][j];
            }
            pre0_fake[r][m] = sum;
        }
    }
    
    // Manual gradient computation
    double alpha = 0.01;
    
    // dD/dW0[0][0] = sum_r real[r][0] * (sum_k W2[0][k] * W1[0][k]) * leaky_relu_prime(pre0[r][0])
    // Compute sum_k W2[0][k] * W1[0][k]
    double sum_k = 0.0;
    for (int k = 0; k < 8; ++k) {
        sum_k += w2.weights[0][k] * w1.weights[k][0];
    }
    cout << "\nsum_k W2[0][k] * W1[k][0] = " << sum_k << endl;
    
    double dD_dW0_00_real = 0.0;
    for (int r = 0; r < 2; ++r) {
        double leaky_prime = (pre0[r][0] > 0) ? 1.0 : alpha;
        dD_dW0_00_real += real[r][0] * sum_k * leaky_prime;
    }
    cout << "Manual dD_real/dW0[0][0] = " << dD_dW0_00_real << endl;
    
    double dD_dW0_00_fake = 0.0;
    for (int r = 0; r < 2; ++r) {
        double leaky_prime = (pre0_fake[r][0] > 0) ? 1.0 : alpha;
        dD_dW0_00_fake += fake[r][0] * sum_k * leaky_prime;
    }
    cout << "Manual dD_fake/dW0[0][0] = " << dD_dW0_00_fake << endl;
    cout << "Manual total dL/dW0[0][0] = " << dD_dW0_00_real + dD_dW0_00_fake << endl;
    
    // Now let me check what the analytical gradient is
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    
    cout << "\nAnalytical grad_weights[0][0] = " << w0.grad_weights[0][0] << endl;
    
    // What does the numerical gradient check give?
    double orig_w00 = w0.weights[0][0];
    double eps = 1e-4;
    
    w0.weights[0][0] = orig_w00 + eps;
    critic.reset_cached_inputs();
    Tensor out_r_p = critic.forward(real);
    double loss_r_plus = out_r_p[0][0] + out_r_p[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_p = critic.forward(fake);
    double loss_f_plus = out_f_p[0][0] + out_f_p[1][0];
    
    w0.weights[0][0] = orig_w00 - eps;
    critic.reset_cached_inputs();
    Tensor out_r_m = critic.forward(real);
    double loss_r_minus = out_r_m[0][0] + out_r_m[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_m = critic.forward(fake);
    double loss_f_minus = out_f_m[0][0] + out_f_m[1][0];
    
    w0.weights[0][0] = orig_w00;
    
    double num_grad = (loss_r_plus + loss_f_plus - loss_r_minus - loss_f_minus) / (2.0 * eps);
    cout << "Numerical grad = " << num_grad << endl;
    
    return 0;
}
