// Compare numerical vs analytical gradients for w1
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
    
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    cout << "\n=== Numerical gradient for w1.weights[0][0] ===" << endl;
    
    // Compute numerical gradient for w1
    double orig_w10 = w1.weights[0][0];
    double eps = 1e-4;
    
    w1.weights[0][0] = orig_w10 + eps;
    critic.reset_cached_inputs();
    Tensor out_r_p = critic.forward(real);
    double loss_r_plus = out_r_p[0][0] + out_r_p[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_p = critic.forward(fake);
    double loss_f_plus = out_f_p[0][0] + out_f_p[1][0];
    
    w1.weights[0][0] = orig_w10 - eps;
    critic.reset_cached_inputs();
    Tensor out_r_m = critic.forward(real);
    double loss_r_minus = out_r_m[0][0] + out_r_m[1][0];
    critic.reset_cached_inputs();
    Tensor out_f_m = critic.forward(fake);
    double loss_f_minus = out_f_m[0][0] + out_f_m[1][0];
    
    w1.weights[0][0] = orig_w10;
    
    double num_grad_w1 = (loss_r_plus + loss_f_plus - loss_r_minus - loss_f_minus) / (2.0 * eps);
    cout << "Numerical gradient w1[0][0] = " << num_grad_w1 << endl;
    
    // Analytical gradient for w1
    critic.zero_grad();
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.reset_cached_inputs();
    critic.forward(real);
    critic.backward_from(grad_out);
    critic.reset_cached_inputs();
    critic.forward(fake);
    critic.backward_from(grad_out);
    
    cout << "Analytical gradient w1[0][0] = " << w1.grad_weights[0][0] << endl;
    
    double rel_err = abs(num_grad_w1 - w1.grad_weights[0][0]) / (abs(num_grad_w1) + abs(w1.grad_weights[0][0]) + 1e-8);
    cout << "Relative error = " << rel_err << endl;
    
    cout << "\n=== Numerical gradient for w0.weights[0][0] ===" << endl;
    
    double orig_w00 = w0.weights[0][0];
    
    w0.weights[0][0] = orig_w00 + eps;
    critic.reset_cached_inputs();
    out_r_p = critic.forward(real);
    loss_r_plus = out_r_p[0][0] + out_r_p[1][0];
    critic.reset_cached_inputs();
    out_f_p = critic.forward(fake);
    loss_f_plus = out_f_p[0][0] + out_f_p[1][0];
    
    w0.weights[0][0] = orig_w00 - eps;
    critic.reset_cached_inputs();
    out_r_m = critic.forward(real);
    loss_r_minus = out_r_m[0][0] + out_r_m[1][0];
    critic.reset_cached_inputs();
    out_f_m = critic.forward(fake);
    loss_f_minus = out_f_m[0][0] + out_f_m[1][0];
    
    w0.weights[0][0] = orig_w00;
    
    double num_grad_w0 = (loss_r_plus + loss_f_plus - loss_r_minus - loss_f_minus) / (2.0 * eps);
    cout << "Numerical gradient w0[0][0] = " << num_grad_w0 << endl;
    
    cout << "Analytical gradient w0[0][0] = " << w0.grad_weights[0][0] << endl;
    
    return 0;
}
