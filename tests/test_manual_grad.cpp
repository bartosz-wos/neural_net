// Manually compute gradient to verify the formula
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
    
    // Manual forward and gradient for a single sample
    // D[0] = pre2[0] = sum_k h1[0][k] * W2[0][k]
    // h1[0][k] = leakyrelu(pre1[0][k])
    // pre1[0][k] = sum_m h0[0][m] * W1[k][m]
    // h0[0][m] = leakyrelu(pre0[0][m])
    // pre0[0][m] = sum_j x[0][j] * W0[j][m] = sum_j x[0][j] * w0.weights[m][j]
    
    // So D[0]/W0[m][j] = x[0][j] * leakyrelu_slope(pre0[0][m]) * 
    //                    sum_k (W2[0][k] * W1[k][m] * leakyrelu_slope(pre1[0][k]))
    
    // Compute pre0 for sample 0
    double pre0_0_m[8];
    double h0_0_m[8];
    for (int m = 0; m < 8; ++m) {
        pre0_0_m[m] = real(0, 0) * w0.weights[m][0] + real(0, 1) * w0.weights[m][1];
        h0_0_m[m] = (pre0_0_m[m] > 0) ? pre0_0_m[m] : alpha * pre0_0_m[m];
    }
    
    // Compute pre1 for sample 0
    double pre1_0_k[8];
    double h1_0_k[8];
    for (int k = 0; k < 8; ++k) {
        pre1_0_k[k] = 0.0;
        for (int m = 0; m < 8; ++m) {
            pre1_0_k[k] += h0_0_m[m] * w1.weights[k][m];
        }
        h1_0_k[k] = (pre1_0_k[k] > 0) ? pre1_0_k[k] : alpha * pre1_0_k[k];
    }
    
    // Compute D[0] for sample 0
    double D0 = 0.0;
    for (int k = 0; k < 8; ++k) {
        D0 += h1_0_k[k] * w2.weights[0][k];
    }
    cout << "Sample 0 D[0] = " << D0 << endl;
    cout << "Forward D[0] = " << critic.forward(real)(0, 0) << endl;
    
    // Compute dD[0]/dW0[m][j] for sample 0
    cout << "\ndD[0]/dW0 for sample 0:" << endl;
    for (int m = 0; m < 2; ++m) {
        for (int j = 0; j < 2; ++j) {
            double sum_k = 0.0;
            for (int k = 0; k < 8; ++k) {
                double dpre1_dh0 = w1.weights[k][m];
                double dh1_dpre1 = (pre1_0_k[k] > 0) ? 1.0 : alpha;
                sum_k += w2.weights[0][k] * dpre1_dh0 * dh1_dpre1;
            }
            double dh0_dpre0 = (pre0_0_m[m] > 0) ? 1.0 : alpha;
            double dD_dW0 = real(0, j) * dh0_dpre0 * sum_k;
            cout << "  m=" << m << " j=" << j << ": dD_dW0 = " << dD_dW0 << endl;
        }
    }
    
    // Wait, I think I made an error. Let me re-derive.
    // D = pre2 = h1 @ W2^T
    // D[m] = sum_k h1[m][k] * W2[m][k]
    // h1 = leakyrelu(pre1)
    // pre1 = h0 @ W1^T
    // pre1[m][k] = sum_j h0[m][j] * W1[k][j]
    // h0 = leakyrelu(pre0)
    // pre0 = x @ W0^T
    // pre0[m][j] = sum_i x[m][i] * W0[j][i]
    
    // dD/dW0[j][i] = sum_m dD[m]/dW0[j][i]
    // dD[m]/dW0[j][i] = sum_k dD[m]/dh1[m][k] * dh1[m][k]/dpre1[m][k] * dpre1[m][k]/dh0[m][j] * dh0[m][j]/dpre0[m][j] * dpre0[m][j]/dW0[j][i]
    
    // dD[m]/dh1[m][k] = W2[m][k]
    // dh1[m][k]/dpre1[m][k] = leakyrelu'(pre1[m][k])
    // dpre1[m][k]/dh0[m][j] = W1[k][j]
    // dh0[m][j]/dpre0[m][j] = leakyrelu'(pre0[m][j])
    // dpre0[m][j]/dW0[j][i] = x[m][i]
    
    // So: dD[m]/dW0[j][i] = x[m][i] * leakyrelu'(pre0[m][j]) * 
    //                       sum_k (W2[m][k] * W1[k][j] * leakyrelu'(pre1[m][k]))
    
    // This is for weights W0[j][i] where W0 is the untransposed weight matrix
    // But w0.weights is (out=8, in=2) = W0^T, so w0.weights[j][i] = W0[j][i]
    
    cout << "\nCorrected dD[0]/dW0 formula:" << endl;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            double sum_k = 0.0;
            for (int k = 0; k < 8; ++k) {
                double dh1_dpre1 = (pre1_0_k[k] > 0) ? 1.0 : alpha;
                sum_k += w2.weights[0][k] * w1.weights[k][j] * dh1_dpre1;
            }
            double dh0_dpre0 = (pre0_0_m[j] > 0) ? 1.0 : alpha;
            double dD_dW0 = real(0, i) * dh0_dpre0 * sum_k;
            cout << "  W0[" << j << "][" << i << "]: dD_dW0 = " << dD_dW0 << endl;
        }
    }
    
    // The issue: w0.weights[m][j] vs W0[m][j]
    // In the formula: pre0[m][j] = sum_i x[i] * W0[j][i]
    // If w0.weights is stored as W0 (not W0^T), then:
    // pre0 = x @ w0.weights^T (forward uses @ W^T)
    // So pre0[m][j] = sum_i x[i] * w0.weights[j][i]?
    
    // Wait, let me check the Dense::forward
    // Tensor forward(const Tensor& input) override {
    //     last_input = input;
    //     return input * weights.transpose() + bias;
    // }
    // input @ W^T = pre0
    // So pre0[j][m] = sum_i input[j][i] * weights[m][i]
    // pre0[j][m] is row j, col m
    
    // So w0.weights[m][i] = W^T[m][i] = W[i][m]
    // And pre0[j][m] = sum_i x[j][i] * W[i][m]
    
    // This means W_untransposed[j][i] = w0.weights[m][i] where m is the column index in pre0
    // So W_untransposed[j][m] = w0.weights[m][j]
    
    // For dpre0/dW0: pre0[j][m] = sum_i x[j][i] * W[i][m]
    // W is the untransposed weight matrix, so W[i][m] = w0.weights[m][i]
    // dpre0[j][m]/dW[i][m] = x[j][i]
    
    // OK so the gradient formula using w0.weights should be:
    // dD/dw0.weights[m][i] = x[j][i] * dD/dpre0[j][m] summed over j
    // = x[j][i] * leakyrelu'(pre0[j][m]) * sum_k (W2[0][k] * dD/dpre1[j][k])
    // Where dD/dpre1[j][k] = W1[k][j] * leakyrelu'(pre1[j][k])
    
    // Let me re-verify by writing the full gradient manually
    
    return 0;
}
