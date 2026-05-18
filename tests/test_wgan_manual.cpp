// Manual verification of WGAN gradient
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
        for (int m = 0; m < 8; ++m) {
            pre0_real[r][m] = real[r][0] * w0.weights[m][0] + real[r][1] * w0.weights[m][1];
        }
    
    Tensor h0_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            h0_real[r][m] = (pre0_real[r][m] > 0) ? pre0_real[r][m] : alpha * pre0_real[r][m];
        }
    
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
        for (int m = 0; m < 8; ++m) {
            h1_real[r][m] = (pre1_real[r][m] > 0) ? pre1_real[r][m] : alpha * pre1_real[r][m];
        }
    
    // Layer 2: pre2 = h1 @ W2^T (no activation)
    Tensor D_real(2, 1);
    for (int r = 0; r < 2; ++r) {
        D_real[r][0] = 0;
        for (int k = 0; k < 8; ++k)
            D_real[r][0] += h1_real[r][k] * w2.weights[0][k];
    }
    
    cout << "Manual D(real): " << D_real[0][0] << ", " << D_real[1][0] << endl;
    cout << "Forward D(real): " << critic.forward(real)[0][0] << ", " << critic.forward(real)[1][0] << endl;
    
    // Manual gradient computation
    // L = D(real)[0] + D(real)[1]
    // dL/dW0[j][i] = sum_r dL/dD[r] * dD[r]/dW0[j][i]
    // dL/dD[r] = 1
    
    // dD[r]/dW0[j][i] = dD[r]/dh0 * dh0/dpre0 * dpre0/dW0[j][i]
    // dD[r]/dh0 = d(h1 @ W2^T)/dh0 = W2
    // dh0/dpre0 = leakyrelu'(pre0)
    // dpre0/dW0[j][i] = real[r][i]  (since pre0[r][j] = sum_i real[r][i] * W0[j][i])
    
    // So: dD[r]/dW0[j][i] = real[r][i] * leakyrelu'(pre0[r][j]) * sum_k W2[0][k] * dpre1[r][k]/dh0[r][j]
    // dpre1[r][k]/dh0[r][j] = W1[k][j]
    
    // Let me compute this more directly:
    // dD[r]/dW0[j][i] = real[r][i] * leakyrelu'(pre0[r][j]) * sum_k W2[0][k] * W1[k][j] * leakyrelu'(pre1[r][k])
    
    // For w0.weights[0][0] (j=0, i=0):
    double dD_dW0_00_real = 0.0;
    for (int r = 0; r < 2; ++r) {
        double sum_k = 0.0;
        for (int k = 0; k < 8; ++k) {
            double dh1_dpre1 = (pre1_real[r][k] > 0) ? 1.0 : alpha;
            sum_k += w2.weights[0][k] * w1.weights[k][0] * dh1_dpre1;
        }
        double dh0_dpre0 = (pre0_real[r][0] > 0) ? 1.0 : alpha;
        dD_dW0_00_real += real[r][0] * dh0_dpre0 * sum_k;
    }
    
    // Same for fake
    Tensor pre0_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            pre0_fake[r][m] = fake[r][0] * w0.weights[m][0] + fake[r][1] * w0.weights[m][1];
        }
    
    Tensor h0_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            h0_fake[r][m] = (pre0_fake[r][m] > 0) ? pre0_fake[r][m] : alpha * pre0_fake[r][m];
        }
    
    Tensor pre1_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            pre1_fake[r][m] = 0;
            for (int k = 0; k < 8; ++k)
                pre1_fake[r][m] += h0_fake[r][k] * w1.weights[m][k];
        }
    
    Tensor h1_fake(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            h1_fake[r][m] = (pre1_fake[r][m] > 0) ? pre1_fake[r][m] : alpha * pre1_fake[r][m];
        }
    
    Tensor D_fake(2, 1);
    for (int r = 0; r < 2; ++r) {
        D_fake[r][0] = 0;
        for (int k = 0; k < 8; ++k)
            D_fake[r][0] += h1_fake[r][k] * w2.weights[0][k];
    }
    
    cout << "\nManual D(fake): " << D_fake[0][0] << ", " << D_fake[1][0] << endl;
    
    double dD_dW0_00_fake = 0.0;
    for (int r = 0; r < 2; ++r) {
        double sum_k = 0.0;
        for (int k = 0; k < 8; ++k) {
            double dh1_dpre1 = (pre1_fake[r][k] > 0) ? 1.0 : alpha;
            sum_k += w2.weights[0][k] * w1.weights[k][0] * dh1_dpre1;
        }
        double dh0_dpre0 = (pre0_fake[r][0] > 0) ? 1.0 : alpha;
        dD_dW0_00_fake += fake[r][0] * dh0_dpre0 * sum_k;
    }
    
    cout << "\ndD(real)/dW0[0][0] = " << dD_dW0_00_real << endl;
    cout << "dD(fake)/dW0[0][0] = " << dD_dW0_00_fake << endl;
    cout << "Total dL/dW0[0][0] = " << dD_dW0_00_real + dD_dW0_00_fake << endl;
    
    // Compare with what the network computes
    cout << "\n--- Network backward_from ---" << endl;
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    cout << "After backward_from(real), w0.grad_weights[0][0] = " << w0.grad_weights[0][0] << endl;
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    cout << "After backward_from(fake), w0.grad_weights[0][0] = " << w0.grad_weights[0][0] << endl;
    
    return 0;
}
