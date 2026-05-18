// Test to identify exactly what gets cached
#include <iostream>
#include <iomanip>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    WGANDiscriminator critic(2, 8, 2);
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    // Forward pass
    critic.reset_cached_inputs();
    Tensor out = critic.forward(real);
    
    cout << "=== Cached inputs after forward(real) ===" << endl;
    for (int i = 0; i < 3; ++i) {
        cout << "cached_inputs_[" << i << "] shape: " 
             << critic.cached_inputs_[i].rows << "x" << critic.cached_inputs_[i].cols << endl;
        if (critic.cached_inputs_[i].rows > 0 && critic.cached_inputs_[i].cols > 0) {
            cout << "  [0][0] = " << critic.cached_inputs_[i][0][0] << endl;
            cout << "  [0][1] = " << critic.cached_inputs_[i][0][1] << endl;
        }
    }
    
    // Layer 0 forward
    Dense& w0 = critic.layer(0);
    cout << "\n=== Manual pre0 computation ===" << endl;
    // pre0 = real @ w0.weights^T
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
    cout << "pre0[0][0] = " << pre0[0][0] << endl;
    cout << "pre0[0][1] = " << pre0[0][1] << endl;
    cout << "pre0[1][0] = " << pre0[1][0] << endl;
    
    // Layer 1: input is leaky(pre0)
    double alpha = 0.01;
    Tensor h0(2, 8);
    for (int r = 0; r < 2; ++r) {
        for (int m = 0; m < 8; ++m) {
            h0[r][m] = (pre0[r][m] > 0) ? pre0[r][m] : alpha * pre0[r][m];
        }
    }
    cout << "\nh0[0][0] = " << h0[0][0] << " (should match cached_inputs_[1][0][0])" << endl;
    cout << "cached_inputs_[1][0][0] = " << critic.cached_inputs_[1][0][0] << endl;
    
    // Now check: does cached_inputs_[1] match pre0 or h0?
    cout << "\n=== Comparison ===" << endl;
    cout << "cached_inputs_[1][0][0] = " << critic.cached_inputs_[1][0][0] << endl;
    cout << "pre0[0][0] = " << pre0[0][0] << endl;
    cout << "h0[0][0] = " << h0[0][0] << endl;
    cout << "\nCached inputs_[1] matches: " << endl;
    cout << "  pre0: " << (abs(critic.cached_inputs_[1][0][0] - pre0[0][0]) < 1e-10 ? "YES" : "NO") << endl;
    cout << "  h0: " << (abs(critic.cached_inputs_[1][0][0] - h0[0][0]) < 1e-10 ? "YES" : "NO") << endl;
    
    return 0;
}
