// Detailed gradient flow trace
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    WGANDiscriminator critic(2, 8, 2);
    
    Dense& w0 = critic.layer(0);
    Dense& w1 = critic.layer(1);
    Dense& w2 = critic.layer(2);
    
    cout << "w0: " << w0.weights.rows << "x" << w0.weights.cols << endl;
    cout << "w1: " << w1.weights.rows << "x" << w1.weights.cols << endl;
    cout << "w2: " << w2.weights.rows << "x" << w2.weights.cols << endl;
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    // Forward pass to get cached values
    critic.reset_cached_inputs();
    Tensor out = critic.forward(real);
    
    cout << "\nOutput: " << out[0][0] << ", " << out[1][0] << endl;
    
    // Now let's manually compute gradients
    // D(real) = sum_r D[r] (we use dD/dr = 1 for each output)
    
    // For a simple 3-layer network:
    // pre0 = real @ W0^T  (2x2 @ 2x8 = 2x8)
    // h0 = leakyrelu(pre0)
    // pre1 = h0 @ W1^T  (2x8 @ 8x8 = 2x8)
    // h1 = leakyrelu(pre1)
    // pre2 = h1 @ W2^T  (2x8 @ 8x1 = 2x1)
    // D = pre2
    
    // dL/dW0[i][j] = sum_r dL/dD[r] * dD[r]/dW0[i][j]
    // dL/dD[r] = 1 for all r (since L = sum(D))
    // dD[r]/dW0[i][j] = d(pre2[r])/dW0[i][j]
    // = sum_m d(pre2[r])/dh1[m] * dh1[m]/dpre1[m] * dpre1[m]/dh0 * dh0/dpre0 * dpre0/dW0[i][j]
    
    // Let me compute dD/dW0[0][0] numerically with a simpler approach
    // dD/dW0[i][j] = sum_r dD[r]/dW0[i][j]
    
    // dD[r]/dW0[i][j] = h0[r][j] * (sum_m W2[0][m] * dh1[m]/dpre1[m] * W1[i][m]) * leakyrelu'(pre0[r][i])
    
    // Let me compute this step by step
    double alpha = 0.01;
    
    // Compute pre0 = real @ W0^T
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
    
    // Compute h0 = leakyrelu(pre0)
    Tensor h0(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            h0[r][m] = (pre0[r][m] > 0) ? pre0[r][m] : alpha * pre0[r][m];
        }
    }
    
    // Compute pre1 = h0 @ W1^T
    Tensor pre1(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            double sum = 0.0;
            for (int k = 0; k < 8; ++k) {
                sum += h0[r][k] * w1.weights[m][k];
            }
            pre1[r][m] = sum;
        }
    }
    
    // Compute h1 = leakyrelu(pre1)
    Tensor h1(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            h1[r][m] = (pre1[r][m] > 0) ? pre1[r][m] : alpha * pre1[r][m];
        }
    }
    
    // pre2 = h1 @ W2^T = D
    Tensor pre2(2, 1);
    for (int r = 0; r < 2; ++r) {
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += h1[r][k] * w2.weights[0][k];
        }
        pre2[r][0] = sum;
    }
    
    cout << "\nManual pre2 (D): " << pre2[0][0] << ", " << pre2[1][0] << endl;
    cout << "Forward pre2: " << out[0][0] << ", " << out[1][0] << endl;
    
    // dD/dW0[i][j] for i=0, j=0
    // = sum_r dD[r]/dW0[0][0]
    // = sum_r h0[r][0] * (sum_m W2[0][m] * dh1[r][m]/dpre1[r][m] * W1[0][m]) * leakyrelu'(pre0[r][0])
    
    // dh1[r][m]/dpre1[r][m] = 1 if pre1[r][m] > 0, else alpha
    // leakyrelu'(pre0[r][0]) = 1 if pre0[r][0] > 0, else alpha
    
    double dD_dW0_00 = 0.0;
    for (int r = 0; r < 2; ++r) {
        // sum_m W2[0][m] * dh1/dpre1 * W1[0][m]
        double sum_m = 0.0;
        for (int m = 0; m < 8; ++m) {
            double dh1_dpre1 = (pre1[r][m] > 0) ? 1.0 : alpha;
            sum_m += w2.weights[0][m] * dh1_dpre1 * w1.weights[m][0];
        }
        double dpre0_leaky = (pre0[r][0] > 0) ? 1.0 : alpha;
        dD_dW0_00 += h0[r][0] * sum_m * dpre0_leaky;
    }
    
    cout << "\nManual dD/dW0[0][0] = " << dD_dW0_00 << endl;
    
    // Now do the same for the full loss: L = D(real) + D(fake)
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    // Compute pre0_fake
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
    
    // Compute h0_fake = leakyrelu(pre0_fake)
    Tensor h0_fake(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            h0_fake[r][m] = (pre0_fake[r][m] > 0) ? pre0_fake[r][m] : alpha * pre0_fake[r][m];
        }
    }
    
    // Compute pre1_fake = h0_fake @ W1^T
    Tensor pre1_fake(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            double sum = 0.0;
            for (int k = 0; k < 8; ++k) {
                sum += h0_fake[r][k] * w1.weights[m][k];
            }
            pre1_fake[r][m] = sum;
        }
    }
    
    // Compute h1_fake = leakyrelu(pre1_fake)
    Tensor h1_fake(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            h1_fake[r][m] = (pre1_fake[r][m] > 0) ? pre1_fake[r][m] : alpha * pre1_fake[r][m];
        }
    }
    
    // Compute pre2_fake = h1_fake @ W2^T
    Tensor pre2_fake(2, 1);
    for (int r = 0; r < 2; ++r) {
        double sum = 0.0;
        for (int k = 0; k < 8; ++k) {
            sum += h1_fake[r][k] * w2.weights[0][k];
        }
        pre2_fake[r][0] = sum;
    }
    
    double dD_fake_dW0_00 = 0.0;
    for (int r = 0; r < 2; ++r) {
        double sum_m = 0.0;
        for (int m = 0; m < 8; ++m) {
            double dh1_dpre1 = (pre1_fake[r][m] > 0) ? 1.0 : alpha;
            sum_m += w2.weights[0][m] * dh1_dpre1 * w1.weights[m][0];
        }
        double dpre0_leaky = (pre0_fake[r][0] > 0) ? 1.0 : alpha;
        dD_fake_dW0_00 += h0_fake[r][0] * sum_m * dpre0_leaky;
    }
    
    cout << "Manual dD_fake/dW0[0][0] = " << dD_fake_dW0_00 << endl;
    cout << "Manual total dL/dW0[0][0] = " << dD_dW0_00 + dD_fake_dW0_00 << endl;
    
    // Numerical gradient
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
    cout << "\nNumerical gradient = " << num_grad << endl;
    
    // Analytical gradient from backward_from
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    
    cout << "Analytical gradient = " << w0.grad_weights[0][0] << endl;
    
    return 0;
}
