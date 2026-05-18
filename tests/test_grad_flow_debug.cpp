// Debug gradient flow
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
    
    double alpha = 0.01;
    
    // Manual forward and backward
    // Layer 0: pre0 = real @ W0^T, h0 = leakyrelu(pre0)
    Tensor pre0(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            pre0[r][m] = real[r][0] * w0.weights[m][0] + real[r][1] * w0.weights[m][1];
    
    // h0 = leakyrelu(pre0)
    Tensor h0(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h0[r][m] = (pre0[r][m] > 0) ? pre0[r][m] : alpha * pre0[r][m];
    
    // Layer 1: pre1 = h0 @ W1^T, h1 = leakyrelu(pre1)
    Tensor pre1(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m) {
            pre1[r][m] = 0;
            for (int k = 0; k < 8; ++k)
                pre1[r][m] += h0[r][k] * w1.weights[m][k];
        }
    
    Tensor h1(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int m = 0; m < 8; ++m)
            h1[r][m] = (pre1[r][m] > 0) ? pre1[r][m] : alpha * pre1[r][m];
    
    // Layer 2: pre2 = h1 @ W2^T
    Tensor pre2(2, 1);
    for (int r = 0; r < 2; ++r) {
        pre2[r][0] = 0;
        for (int k = 0; k < 8; ++k)
            pre2[r][0] += h1[r][k] * w2.weights[0][k];
    }
    
    cout << "pre2 (D): " << pre2[0][0] << ", " << pre2[1][0] << endl;
    
    // Backward: grad_output = [1, 1]^T
    // dL/dpre2 = [1, 1]^T (identity for output layer)
    // dL/dh1 = dL/dpre2 @ W2
    // dL/dpre1 = dL/dh1 * leakyrelu'(pre1)
    // dL/dh0 = dL/dpre1 @ W1
    // dL/dpre0 = dL/dh0 * leakyrelu'(pre0)
    
    // dL/dpre2 = [1, 1]^T
    double dl_dpre2_0 = 1.0;
    double dl_dpre2_1 = 1.0;
    
    // dL/dh1 = dL/dpre2 @ W2
    // dh1[r][k]/dpre1[r][k] = leakyrelu'(pre1[r][k])
    // dL/dh1[r][k] = dL/dpre2[r][0] * W2[0][k] (since pre2[r][0] = sum_k h1[r][k] * W2[0][k])
    // = 1 * W2[0][k] = W2[0][k]
    // But we need to incorporate the leakyrelu' for pre1!
    
    // dL/dpre1[r][k] = dL/dh1[r][k] * leakyrelu'(pre1[r][k])
    // = W2[0][k] * leakyrelu'(pre1[r][k])
    double dl_dpre1_0[8], dl_dpre1_1[8];
    for (int k = 0; k < 8; ++k) {
        double leaky1_0 = (pre1[0][k] > 0) ? 1.0 : alpha;
        double leaky1_1 = (pre1[1][k] > 0) ? 1.0 : alpha;
        dl_dpre1_0[k] = w2.weights[0][k] * leaky1_0;
        dl_dpre1_1[k] = w2.weights[0][k] * leaky1_1;
    }
    
    // dL/dh0 = dL/dpre1 @ W1
    // dL/dh0[r][j] = sum_k dL/dpre1[r][k] * W1[k][j]
    double dl_dh0_0[8], dl_dh0_1[8];
    for (int j = 0; j < 8; ++j) {
        dl_dh0_0[j] = 0;
        dl_dh0_1[j] = 0;
        for (int k = 0; k < 8; ++k) {
            dl_dh0_0[j] += dl_dpre1_0[k] * w1.weights[k][j];
            dl_dh0_1[j] += dl_dpre1_1[k] * w1.weights[k][j];
        }
    }
    
    // dL/dpre0[r][j] = dL/dh0[r][j] * leakyrelu'(pre0[r][j])
    double dl_dpre0_0[8], dl_dpre0_1[8];
    for (int j = 0; j < 8; ++j) {
        double leaky0_0 = (pre0[0][j] > 0) ? 1.0 : alpha;
        double leaky0_1 = (pre0[1][j] > 0) ? 1.0 : alpha;
        dl_dpre0_0[j] = dl_dh0_0[j] * leaky0_0;
        dl_dpre0_1[j] = dl_dh0_1[j] * leaky0_1;
    }
    
    cout << "\ndL/dpre0[0][0] = " << dl_dpre0_0[0] << endl;
    cout << "dL/dpre0[1][0] = " << dl_dpre0_1[0] << endl;
    
    // Weight gradient for layer 0: dL/dW0[j][i] = sum_r dL/dpre0[r][j] * input[r][i]
    // = sum_r dL/dpre0[r][j] * real[r][i]
    double dl_dW0_00 = dl_dpre0_0[0] * real[0][0] + dl_dpre0_1[0] * real[1][0];
    cout << "\nManual dL/dW0[0][0] = " << dl_dW0_00 << endl;
    
    // Now compare with what the network computes
    cout << "\n=== Network ===" << endl;
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    cout << "Network dL/dW0[0][0] = " << w0.grad_weights[0][0] << endl;
    
    // Let me trace through backward_from manually to see what's happening
    
    // grad = grad_output = [[1], [1]]
    // i = 2: layer 2 (output)
    //   grad_pre = grad (identity)
    //   grad_w2 += grad_pre^T * cached_inputs_[2]
    //   cached_inputs_[2] = pre1
    //   grad_w2[m][n] = sum_r grad[r][m] * pre1[r][n]
    //   For m=0, n=0: grad_w2[0][0] = grad[0][0] * pre1[0][0] + grad[1][0] * pre1[1][0]
    //   Wait, but this uses pre1, not h1! That's wrong!
    
    // Actually wait - cached_inputs_[i] is the INPUT to layer i, which is the OUTPUT of the previous layer's activation.
    // For layer 2, cached_inputs_[2] = output of layer 1 = pre1 before activation? NO!
    // 
    // Let me re-trace the forward:
    // x = input (real)
    // x = layers_[0].forward(x) -> pre0, cached_inputs_[1] = pre0
    // x = leakyrelu(pre0) -> h0
    // x = layers_[1].forward(x) -> pre1, cached_inputs_[2] = pre1
    // x = leakyrelu(pre1) -> h1
    // x = layers_[2].forward(x) -> pre2
    // 
    // So cached_inputs_[2] = pre1, but the actual input to layer 2 is h1!
    // So cached_inputs_[2] is WRONG for computing the weight gradient at layer 2!
    // We need h1 (the activation output), not pre1 (the pre-activation input).
    
    // This is the bug! For layer 2, cached_inputs_[2] = pre1, but we need h1 = leakyrelu(pre1).
    // Same for layer 1: cached_inputs_[1] = pre0, but we need h0 = leakyrelu(pre0).
    // For layer 0: cached_inputs_[0] = real (input), which is correct.
    
    // So for layer 2, the weight gradient should use h1, not pre1!
    // But we don't cache h1... we only cache pre0, pre1, etc.
    
    // The fix: we need to cache the activation outputs (h), not just the pre-activations (pre).
    // Or we need to recompute h from pre when computing weight gradients.
    
    cout << "\n=== Analysis ===" << endl;
    cout << "cached_inputs_[0] = real (correct for layer 0 input)" << endl;
    cout << "cached_inputs_[1] = pre0 (but need h0 for layer 1 weight grad)" << endl;
    cout << "cached_inputs_[2] = pre1 (but need h1 for layer 2 weight grad)" << endl;
    cout << "\nThe weight gradient computation uses pre-activation instead of post-activation!" << endl;
    
    return 0;
}
