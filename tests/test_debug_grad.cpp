// Debug gradient computation
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
    Dense& w1 = critic.layer(1);
    Dense& w2 = critic.layer(2);
    
    // Single sample fake
    Tensor fake(1, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    
    double alpha = 0.01;
    
    // Manual forward for fake
    // pre0 = fake @ W0^T, dims (1, 8)
    // h0 = leakyrelu(pre0)
    // pre1 = h0 @ W1^T, dims (1, 8)
    // h1 = leakyrelu(pre1)
    // pre2 = h1 @ W2^T, dims (1, 1)
    
    // pre0[j] = fake[0]*W0[j][0] + fake[1]*W0[j][1]
    double pre0[8];
    for (int j = 0; j < 8; ++j) {
        pre0[j] = fake[0][0] * w0.weights[j][0] + fake[0][1] * w0.weights[j][1];
    }
    
    // h0[j] = leakyrelu(pre0[j])
    double h0[8];
    for (int j = 0; j < 8; ++j) {
        h0[j] = (pre0[j] > 0) ? pre0[j] : alpha * pre0[j];
    }
    
    // pre1[k] = sum_j h0[j] * W1[k][j]
    double pre1[8];
    for (int k = 0; k < 8; ++k) {
        pre1[k] = 0;
        for (int j = 0; j < 8; ++j) {
            pre1[k] += h0[j] * w1.weights[k][j];
        }
    }
    
    // h1[k] = leakyrelu(pre1[k])
    double h1[8];
    for (int k = 0; k < 8; ++k) {
        h1[k] = (pre1[k] > 0) ? pre1[k] : alpha * pre1[k];
    }
    
    // pre2 = sum_k h1[k] * W2[0][k]
    double pre2 = 0;
    for (int k = 0; k < 8; ++k) {
        pre2 += h1[k] * w2.weights[0][k];
    }
    
    cout << "Manual pre2 (D) = " << pre2 << endl;
    
    // Now backward
    // dL/dpre2 = 1
    // dL/dh1[k] = W2[0][k]
    // dL/dpre1[k] = dL/dh1[k] * leakyrelu'(pre1[k])
    // dL/dh0[j] = sum_k dL/dpre1[k] * W1[k][j]
    // dL/dpre0[j] = dL/dh0[j] * leakyrelu'(pre0[j])
    // dL/dW0[j][i] = dL/dpre0[j] * input[i]
    
    double dl_dpre1[8];
    for (int k = 0; k < 8; ++k) {
        double leaky = (pre1[k] > 0) ? 1.0 : alpha;
        dl_dpre1[k] = w2.weights[0][k] * leaky;
    }
    
    double dl_dh0[8];
    for (int j = 0; j < 8; ++j) {
        dl_dh0[j] = 0;
        for (int k = 0; k < 8; ++k) {
            dl_dh0[j] += dl_dpre1[k] * w1.weights[k][j];
        }
    }
    
    double dl_dpre0[8];
    for (int j = 0; j < 8; ++j) {
        double leaky = (pre0[j] > 0) ? 1.0 : alpha;
        dl_dpre0[j] = dl_dh0[j] * leaky;
    }
    
    double dl_dW0_00 = dl_dpre0[0] * fake[0][0];
    double dl_dW0_01 = dl_dpre0[0] * fake[0][1];
    double dl_dW0_10 = dl_dpre0[1] * fake[0][0];
    double dl_dW0_11 = dl_dpre0[1] * fake[0][1];
    
    cout << "Manual dL/dW0[0][0] = " << dl_dW0_00 << endl;
    cout << "Manual dL/dW0[0][1] = " << dl_dW0_01 << endl;
    cout << "Manual dL/dW0[1][0] = " << dl_dW0_10 << endl;
    cout << "Manual dL/dW0[1][1] = " << dl_dW0_11 << endl;
    
    // Now compute with network
    critic.zero_grad();
    Tensor grad_out(1, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    
    cout << "\nNetwork dL/dW0[0][0] = " << w0.grad_weights[0][0] << endl;
    cout << "Network dL/dW0[0][1] = " << w0.grad_weights[0][1] << endl;
    cout << "Network dL/dW0[1][0] = " << w0.grad_weights[1][0] << endl;
    cout << "Network dL/dW0[1][1] = " << w0.grad_weights[1][1] << endl;
    
    // Check some intermediate values
    cout << "\n=== Debug ===" << endl;
    cout << "w2.weights[0][0] = " << w2.weights[0][0] << endl;
    cout << "w2.weights[0][1] = " << w2.weights[0][1] << endl;
    
    return 0;
}
