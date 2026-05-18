// Detailed gradient computation trace
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
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    double alpha = 0.01;
    
    // Manual forward for real
    // Layer 0: pre0 = real @ W0^T, h0 = leakyrelu(pre0)
    Tensor pre0_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            pre0_real[r][m] = real[r][0] * w0.weights[m][0] + real[r][1] * w0.weights[m][1];
    
    Tensor h0_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h0_real[r][m] = (pre0_real[r][m] > 0) ? pre0_real[r][m] : alpha * pre0_real[r][m];
    
    // Layer 1: pre1 = h0 @ W1^T, h1 = leakyrelu(pre1)
    Tensor pre1_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            pre1_real[r][m] = 0;
            for (int k = 0; k < 8; ++k)
                pre1_real[r][m] += h0_real[r][k] * w1.weights[m][k];
        }
    
    Tensor h1_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h1_real[r][m] = (pre1_real[r][m] > 0) ? pre1_real[r][m] : alpha * pre1_real[r][m];
    
    // Layer 2: pre2 = h1 @ W2^T
    Tensor pre2_real(2, 1);
    for (int r = 0; r < 2; ++r) {
        pre2_real[r][0] = 0;
        for (int k = 0; k < 8; ++k)
            pre2_real[r][0] += h1_real[r][k] * w2.weights[0][k];
    }
    
    // Manual gradient for real
    // dL/dpre2 = [[1], [1]] (identity)
    // dL/dh1 = dL/dpre2 @ W2 = [[1], [1]] @ W2
    // dL/dpre1 = dL/dh1 * leakyrelu'(pre1)
    // dL/dh0 = dL/dpre1 @ W1
    // dL/dpre0 = dL/dh0 * leakyrelu'(pre0)
    
    // For real:
    // dL/dpre2 = [[1], [1]]
    // dL/dh1[r][k] = sum_m dL/dpre2[r][m] * W2[m][k] = 1 * W2[0][k] for each r
    // dL/dpre1[r][k] = dL/dh1[r][k] * leakyrelu'(pre1[r][k])
    
    // Compute dL/dpre1 for real
    double dl_dpre1_real[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int k = 0; k < 8; ++k) {
            double dl_dh1 = w2.weights[0][k];  // dL/dpre2[r][0] = 1
            double leaky = (pre1_real[r][k] > 0) ? 1.0 : alpha;
            dl_dpre1_real[r][k] = dl_dh1 * leaky;
        }
    }
    
    // dL/dh0 = dL/dpre1 @ W1
    double dl_dh0_real[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 8; ++j) {
            dl_dh0_real[r][j] = 0;
            for (int k = 0; k < 8; ++k) {
                dl_dh0_real[r][j] += dl_dpre1_real[r][k] * w1.weights[k][j];
            }
        }
    }
    
    // dL/dpre0 = dL/dh0 * leakyrelu'(pre0)
    double dl_dpre0_real[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 8; ++j) {
            double leaky = (pre0_real[r][j] > 0) ? 1.0 : alpha;
            dl_dpre0_real[r][j] = dl_dh0_real[r][j] * leaky;
        }
    }
    
    // dL/dW0[j][i] = sum_r dL/dpre0[r][j] * real[r][i]
    double dl_dW0_real_00 = 0;
    for (int r = 0; r < 2; ++r) {
        dl_dW0_real_00 += dl_dpre0_real[r][0] * real[r][0];
    }
    cout << "Manual dL/dW0[0][0] for real = " << dl_dW0_real_00 << endl;
    
    // Do the same for fake
    Tensor pre0_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            pre0_fake[r][m] = fake[r][0] * w0.weights[m][0] + fake[r][1] * w0.weights[m][1];
    
    Tensor h0_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h0_fake[r][m] = (pre0_fake[r][m] > 0) ? pre0_fake[r][m] : alpha * pre0_fake[r][m];
    
    Tensor pre1_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            pre1_fake[r][m] = 0;
            for (int k = 0; k < 8; ++k)
                pre1_fake[r][m] += h0_fake[r][k] * w1.weights[m][k];
        }
    
    Tensor h1_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h1_fake[r][m] = (pre1_fake[r][m] > 0) ? pre1_fake[r][m] : alpha * pre1_fake[r][m];
    
    Tensor pre2_fake(2, 1);
    for (int r = 0; r < 2; ++r) {
        pre2_fake[r][0] = 0;
        for (int k = 0; k < 8; ++k)
            pre2_fake[r][0] += h1_fake[r][k] * w2.weights[0][k];
    }
    
    double dl_dpre1_fake[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int k = 0; k < 8; ++k) {
            double dl_dh1 = w2.weights[0][k];
            double leaky = (pre1_fake[r][k] > 0) ? 1.0 : alpha;
            dl_dpre1_fake[r][k] = dl_dh1 * leaky;
        }
    }
    
    double dl_dh0_fake[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 8; ++j) {
            dl_dh0_fake[r][j] = 0;
            for (int k = 0; k < 8; ++k) {
                dl_dh0_fake[r][j] += dl_dpre1_fake[r][k] * w1.weights[k][j];
            }
        }
    }
    
    double dl_dpre0_fake[2][8];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 8; ++j) {
            double leaky = (pre0_fake[r][j] > 0) ? 1.0 : alpha;
            dl_dpre0_fake[r][j] = dl_dh0_fake[r][j] * leaky;
        }
    }
    
    double dl_dW0_fake_00 = 0;
    for (int r = 0; r < 2; ++r) {
        dl_dW0_fake_00 += dl_dpre0_fake[r][0] * fake[r][0];
    }
    cout << "Manual dL/dW0[0][0] for fake = " << dl_dW0_fake_00 << endl;
    
    cout << "Manual total dL/dW0[0][0] = " << dl_dW0_real_00 + dl_dW0_fake_00 << endl;
    
    // Now compare with network
    cout << "\n=== Network ===" << endl;
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    cout << "After real, grad = " << w0.grad_weights[0][0] << endl;
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    cout << "After fake, grad = " << w0.grad_weights[0][0] << endl;
    
    return 0;
}
