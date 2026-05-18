// Verify the caching issue by checking what values we get
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    WGANDiscriminator critic(2, 8, 2);
    
    // Get layer references
    Dense& w0 = critic.layer(0);
    Dense& w1 = critic.layer(1);
    Dense& w2 = critic.layer(2);
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    // Forward pass
    critic.reset_cached_inputs();
    Tensor out = critic.forward(real);
    
    // Now manually compute pre0
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
    
    cout << "w0.last_input[0][0] = " << w0.last_input[0][0] << endl;
    cout << "w0.last_input[1][0] = " << w0.last_input[1][0] << endl;
    cout << "real[0][0] = " << real[0][0] << endl;
    cout << "real[1][0] = " << real[1][0] << endl;
    
    cout << "\npre0[0][0] = " << pre0[0][0] << endl;
    cout << "w0.last_input = pre0? " << (abs(w0.last_input[0][0] - pre0[0][0]) < 1e-10 ? "YES" : "NO") << endl;
    
    // After forward, w0.last_input = input to Dense forward = the x before matmul
    // In forward: Tensor out = input @ W^T + b
    // So input is the original tensor, not pre0
    // w0.last_input = real (the input)
    
    // And cached_inputs_[1] = w0.forward(real) = pre0 (the output of Dense)
    
    cout << "\ncached_inputs_[1][0][0] = " << critic.cached_inputs_[1][0][0] << endl;
    cout << "pre0[0][0] = " << pre0[0][0] << endl;
    cout << "cached_inputs_[1] == pre0? " << (abs(critic.cached_inputs_[1][0][0] - pre0[0][0]) < 1e-10 ? "YES" : "NO") << endl;
    
    return 0;
}
