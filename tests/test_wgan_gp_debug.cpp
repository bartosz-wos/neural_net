// Debug test 5 with more detail
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
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
    
    Dense& w1 = critic.layer(0);
    cout << "w1.weights[0][0] = " << w1.weights[0][0] << endl;
    cout << "w1.weights shape: " << w1.weights.rows << "x" << w1.weights.cols << endl;
    
    // Let's trace what happens in the first forward pass for real
    cout << "\n=== Forward real ===" << endl;
    critic.reset_cached_inputs();
    Tensor out_real = critic.forward(real);
    cout << "D(real) output: " << out_real[0][0] << ", " << out_real[1][0] << endl;
    cout << "Sum = " << out_real[0][0] + out_real[1][0] << endl;
    
    // Print cached inputs for each layer
    cout << "\nCached inputs shapes:" << endl;
    for (size_t i = 0; i < critic.num_layers(); ++i) {
        cout << "  layer " << i << " last_input: " << critic.layer(i).last_input.rows << "x" << critic.layer(i).last_input.cols << endl;
    }
    
    // Now trace backward_from
    cout << "\n=== backward_from for real ===" << endl;
    cout << "grad_out = [[1],[1]]" << endl;
    
    Tensor grad_out_real(2, 1);
    grad_out_real[0][0] = 1.0;
    grad_out_real[1][0] = 1.0;
    
    critic.zero_grad();
    critic.forward(real);
    critic.backward_from(grad_out_real);
    
    cout << "After backward_from:" << endl;
    for (size_t i = 0; i < critic.num_layers(); ++i) {
        cout << "  layer " << i << " grad_weights norm: " << critic.layer(i).grad_weights.rows << "x" << critic.layer(i).grad_weights.cols << endl;
        cout << "    grad_weights[0][0] = " << critic.layer(i).grad_weights[0][0] << endl;
    }
    
    // Now do the same for fake
    cout << "\n=== Forward fake ===" << endl;
    critic.reset_cached_inputs();
    Tensor out_fake = critic.forward(fake);
    cout << "D(fake) output: " << out_fake[0][0] << ", " << out_fake[1][0] << endl;
    
    cout << "\n=== backward_from for fake ===" << endl;
    Tensor grad_out_fake(2, 1);
    grad_out_fake[0][0] = 1.0;
    grad_out_fake[1][0] = 1.0;
    
    critic.forward(fake);
    critic.backward_from(grad_out_fake);
    
    cout << "After backward_from (both):" << endl;
    for (size_t i = 0; i < critic.num_layers(); ++i) {
        cout << "  layer " << i << " grad_weights[0][0] = " << critic.layer(i).grad_weights[0][0] << endl;
    }
    
    // Now verify numerical gradient
    cout << "\n=== Numerical gradient check ===" << endl;
    double orig_w00 = w1.weights[0][0];
    double eps = 1e-4;
    
    // w + eps
    w1.weights[0][0] = orig_w00 + eps;
    critic.reset_cached_inputs();
    Tensor out_r_p = critic.forward(real);
    double loss_r_plus = out_r_p[0][0] + out_r_p[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_p = critic.forward(fake);
    double loss_f_plus = out_f_p[0][0] + out_f_p[1][0];
    double loss_plus = loss_r_plus + loss_f_plus;
    
    // w - eps
    w1.weights[0][0] = orig_w00 - eps;
    critic.reset_cached_inputs();
    Tensor out_r_m = critic.forward(real);
    double loss_r_minus = out_r_m[0][0] + out_r_m[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_m = critic.forward(fake);
    double loss_f_minus = out_f_m[0][0] + out_f_m[1][0];
    double loss_minus = loss_r_minus + loss_f_minus;
    
    w1.weights[0][0] = orig_w00;
    
    double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    cout << "D(real) w+eps: " << out_r_p[0][0] << ", " << out_r_p[1][0] << endl;
    cout << "D(fake) w+eps: " << out_f_p[0][0] << ", " << out_f_p[1][0] << endl;
    cout << "D(real) w-eps: " << out_r_m[0][0] << ", " << out_r_m[1][0] << endl;
    cout << "D(fake) w-eps: " << out_f_m[0][0] << ", " << out_f_m[1][0] << endl;
    cout << "loss_plus = " << loss_plus << ", loss_minus = " << loss_minus << endl;
    cout << "num_grad = " << num_grad << endl;
    
    cout << "\nFinal ana_grad = " << w1.grad_weights[0][0] << endl;
    
    return 0;
}
