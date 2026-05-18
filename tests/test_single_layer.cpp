// Test single layer gradient to isolate the issue
#include <iostream>
#include <iomanip>
#include <cmath>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"
#include "nn/layers/dense/dense.h"

using namespace std;

// Simple single-layer network to test gradient
class SimpleCritic {
public:
    vector<Dense> layers;
    
    SimpleCritic(int input_dim, int output_dim) {
        layers.push_back(Dense(input_dim, output_dim));
    }
    
    Dense& layer(int i) { return layers[i]; }
    size_t num_layers() const { return layers.size(); }
    
    Tensor forward(const Tensor& input) {
        cached_inputs_.clear();
        Tensor out = input;
        for (auto& layer : layers) {
            cached_inputs_.push_back(out);
            out = layer.forward(out);
        }
        return out;
    }
    
    void zero_grad() {
        for (auto& layer : layers) {
            layer.grad_weights.fill(0.0);
            if (layer.use_bias_) layer.grad_bias.fill(0.0);
        }
    }
    
    void backward_from(const Tensor& grad_out) {
        // Start from last layer
        Tensor grad = grad_out;
        for (int i = num_layers() - 1; i >= 0; --i) {
            layers[i].backward(grad);
            // grad for next layer (input grad)
            grad = layers[i].grad_input;
        }
    }
    
    vector<Tensor> cached_inputs_;
};

int main() {
    cout << setprecision(12);
    
    // Single layer: 2 -> 1
    SimpleCritic critic(2, 1);
    
    // Use same inputs as test 5
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    Tensor fake(2, 2);
    fake[0][0] = -0.5; fake[0][1] = 0.5;
    fake[1][0] = 0.0; fake[1][1] = -1.0;
    
    Dense& w = critic.layer(0);
    cout << "w.weights[0][0] = " << w.weights[0][0] << endl;
    cout << "w.weights shape: " << w.weights.rows << "x" << w.weights.cols << endl;
    
    double orig_w00 = w.weights[0][0];
    double eps = 1e-4;
    
    // Numerical gradient
    w.weights[0][0] = orig_w00 + eps;
    Tensor out_r_p = critic.forward(real);
    double loss_r_plus = out_r_p[0][0] + out_r_p[1][0];
    Tensor out_f_p = critic.forward(fake);
    double loss_f_plus = out_f_p[0][0] + out_f_p[1][0];
    
    w.weights[0][0] = orig_w00 - eps;
    Tensor out_r_m = critic.forward(real);
    double loss_r_minus = out_r_m[0][0] + out_r_m[1][0];
    Tensor out_f_m = critic.forward(fake);
    double loss_f_minus = out_f_m[0][0] + out_f_m[1][0];
    
    w.weights[0][0] = orig_w00;
    
    double num_grad = (loss_r_plus + loss_f_plus - loss_r_minus - loss_f_minus) / (2.0 * eps);
    
    // Analytical gradient
    critic.zero_grad();
    critic.forward(real);
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    critic.backward_from(grad_out);
    
    critic.forward(fake);
    critic.backward_from(grad_out);
    
    double ana_grad = w.grad_weights[0][0];
    
    cout << "\n=== Single layer test ===" << endl;
    cout << "D(real) w+eps: " << out_r_p[0][0] << ", " << out_r_p[1][0] << endl;
    cout << "D(fake) w+eps: " << out_f_p[0][0] << ", " << out_f_p[1][0] << endl;
    cout << "D(real) w-eps: " << out_r_m[0][0] << ", " << out_r_m[1][0] << endl;
    cout << "D(fake) w-eps: " << out_f_m[0][0] << ", " << out_f_m[1][0] << endl;
    cout << "num_grad = " << num_grad << endl;
    cout << "ana_grad = " << ana_grad << endl;
    
    double err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-8);
    cout << "rel_error = " << err << endl;
    
    return 0;
}
