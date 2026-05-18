// More detailed debug to trace what's happening in the gradient computation
#include <iostream>
#include <iomanip>
#include "nn/layers/generative/wgan_gp.h"
#include "nn/core/tensor.h"

using namespace std;

int main() {
    cout << setprecision(12);
    
    WGANDiscriminator critic(2, 8, 2);
    
    // Get weights reference
    Dense& w1 = critic.layer(0);
    Dense& w2 = critic.layer(1);
    Dense& w3 = critic.layer(2);
    
    cout << "Initial w1.weights[0][0] = " << w1.weights[0][0] << endl;
    cout << "w2.weights shape: " << w2.weights.rows << "x" << w2.weights.cols << endl;
    cout << "w3.weights shape: " << w3.weights.rows << "x" << w3.weights.cols << endl;
    
    // Set up inputs
    Tensor real(2, 2);
    real[0][0] = 0.5; real[0][1] = -0.5;
    real[1][0] = 1.0; real[1][1] = 0.0;
    
    // Fresh: just do forward then backward with grad_out = [[1],[1]]
    critic.zero_grad();
    critic.reset_cached_inputs();
    Tensor out_real = critic.forward(real);
    
    cout << "\n=== Forward pass ===" << endl;
    cout << "Output: " << out_real[0][0] << ", " << out_real[1][0] << endl;
    
    // Print layer 0 last_input
    cout << "\nLayer 0 last_input:" << endl;
    for (int r = 0; r < critic.layer(0).last_input.rows; ++r) {
        for (int c = 0; c < critic.layer(0).last_input.cols; ++c) {
            cout << "  [" << r << "][" << c << "] = " << critic.layer(0).last_input[r][c] << endl;
        }
    }
    
    // Print layer 1 last_input
    cout << "\nLayer 1 last_input:" << endl;
    for (int r = 0; r < critic.layer(1).last_input.rows; ++r) {
        for (int c = 0; c < critic.layer(1).last_input.cols; ++c) {
            cout << "  [" << r << "][" << c << "] = " << critic.layer(1).last_input[r][c] << endl;
        }
    }
    
    // Print layer 2 last_input (output layer)
    cout << "\nLayer 2 last_input:" << endl;
    for (int r = 0; r < critic.layer(2).last_input.rows; ++r) {
        for (int c = 0; c < critic.layer(2).last_input.cols; ++c) {
            cout << "  [" << r << "][" << c << "] = " << critic.layer(2).last_input[r][c] << endl;
        }
    }
    
    // Now backward_from
    cout << "\n=== Backward pass ===" << endl;
    cout << "grad_out = [[1],[1]]" << endl;
    
    Tensor grad_out(2, 1);
    grad_out[0][0] = 1.0;
    grad_out[1][0] = 1.0;
    
    critic.backward_from(grad_out);
    
    cout << "\nAfter backward, grad_weights:" << endl;
    for (int i = 0; i < 3; ++i) {
        cout << "Layer " << i << " grad_weights[0][0] = " << critic.layer(i).grad_weights[0][0] << endl;
    }
    
    // Now let me trace what happens in backward_from step by step
    cout << "\n=== Manual trace of backward_from ===" << endl;
    
    // Layer 2 (output layer) backward
    // grad_out for layer 2 is what we passed in: [[1],[1]]
    // accumulate_dense_grad(w3, grad_out, cached_input=layer2.last_input)
    // grad_w3 = grad_out^T @ layer2.last_input
    cout << "Layer 2: grad_out = [[1],[1]], cached input shape " << w3.last_input.rows << "x" << w3.last_input.cols << endl;
    cout << "grad_w3[0][0] = " << w3.grad_weights[0][0] << endl;
    
    // Layer 1 backward
    // First we compute grad for layer 1's pre-activation
    // The backward_from calls layer->backward(grad) which does:
    // For LeakyReLU: grad_pre = leakyReLU_prime(pre_activation) * grad
    // Then: accumulate_dense_grad for weights
    // Then: grad_input = W^T @ grad_pre (this becomes grad_out for next layer)
    
    // Since grad_out for layer 2 is [[1],[1]] and layer2 is Dense (identity activation),
    // the backward pass from layer 2 would compute grad_input for layer 1
    
    // But wait - the WGAN uses backward_from which calls layer->backward(grad) for each layer
    // So the grad going into layer 1 is the grad of output w.r.t. layer 1's output
    
    // Let me trace what grad is passed to layer 1's backward
    // This is computed inside backward_from
    
    return 0;
}
