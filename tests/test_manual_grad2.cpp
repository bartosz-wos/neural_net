// Correct manual gradient check
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
    
    // Compute all intermediate values for real
    // pre0[r][j] = sum_i real[r][i] * w0.weights[j][i]
    Tensor pre0_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 8; ++j)
            pre0_real[r][j] = real[r][0] * w0.weights[j][0] + real[r][1] * w0.weights[j][1];
    
    // pre1[r][k] = sum_j h0[r][j] * w1.weights[k][j]
    // where h0 = leakyrelu(pre0)
    Tensor h0_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 8; ++j)
            h0_real[r][j] = (pre0_real[r][j] > 0) ? pre0_real[r][j] : alpha * pre0_real[r][j];
    
    Tensor pre1_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int k = 0; k < 8; ++k) {
            pre1_real[r][k] = 0;
            for (int j = 0; j < 8; ++j)
                pre1_real[r][k] += h0_real[r][j] * w1.weights[k][j];
        }
    
    Tensor h1_real(2, 8);
    for (int r = 0; r < 2; ++r)
        for (int k = 0; k < 8; ++k)
            h1_real[r][k] = (pre1_real[r][k] > 0) ? pre1_real[r][k] : alpha * pre1_real[r][k];
    
    // dD[0]/dh0[0][j] = sum_k w2.weights[0][k] * w1.weights[k][j] * leakyrelu'(pre1[0][k])
    // dD[1]/dh0[1][j] = sum_k w2.weights[0][k] * w1.weights[k][j] * leakyrelu'(pre1[1][k])
    
    // dD[r]/dpre0[r][j] = dD[r]/dh0[r][j] * leakyrelu'(pre0[r][j])
    
    // dD[r]/dW0[j][i] = real[r][i] * dD[r]/dpre0[r][j]
    
    // L = sum_r D[r], so dL/dW0[j][i] = sum_r real[r][i] * dD[r]/dpre0[r][j]
    
    // For w0.weights[j][i] (j=0, i=0):
    cout << "Computing dL/dW0[0][0] (correct formula)..." << endl;
    double dL_dW0_00 = 0.0;
    for (int r = 0; r < 2; ++r) {
        // dD[r]/dh0[r][j]
        double dD_dh0 = 0.0;
        for (int k = 0; k < 8; ++k) {
            double dh1_dpre1 = (pre1_real[r][k] > 0) ? 1.0 : alpha;
            dD_dh0 += w2.weights[0][k] * w1.weights[k][0] * dh1_dpre1;
        }
        // dD[r]/dpre0[r][0]
        double dD_dpre0 = dD_dh0 * ((pre0_real[r][0] > 0) ? 1.0 : alpha);
        // dL/dW0[0][0] contribution from sample r
        dL_dW0_00 += real[r][0] * dD_dpre0;
        cout << "  r=" << r << ": dD_dh0=" << dD_dh0 << ", dD_dpre0=" << dD_dpre0 
             << ", real[r][0]=" << real[r][0] << ", contrib=" << real[r][0] * dD_dpre0 << endl;
    }
    cout << "Manual dL/dW0[0][0] = " << dL_dW0_00 << endl;
    
    // Numerical gradient check
    double orig = w0.weights[0][0];
    double eps = 1e-4;
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    // L(w+eps) = D(real,w+eps) + D(fake,w+eps)
    w0.weights[0][0] = orig + eps;
    critic.reset_cached_inputs();
    double L_plus_real = 0;
    Tensor out = critic.forward(real);
    for (int r = 0; r < 2; ++r) L_plus_real += out[r][0];
    critic.reset_cached_inputs();
    double L_plus_fake = 0;
    out = critic.forward(fake);
    for (int r = 0; r < 2; ++r) L_plus_fake += out[r][0];
    
    w0.weights[0][0] = orig - eps;
    critic.reset_cached_inputs();
    double L_minus_real = 0;
    out = critic.forward(real);
    for (int r = 0; r < 2; ++r) L_minus_real += out[r][0];
    critic.reset_cached_inputs();
    double L_minus_fake = 0;
    out = critic.forward(fake);
    for (int r = 0; r < 2; ++r) L_minus_fake += out[r][0];
    w0.weights[0][0] = orig;
    
    double num_grad = ((L_plus_real + L_plus_fake) - (L_minus_real + L_minus_fake)) / (2.0 * eps);
    cout << "Numerical dL/dW0[0][0] = " << num_grad << endl;
    cout << "Manual  dL/dW0[0][0] = " << dL_dW0_00 << endl;
    
    return 0;
}
