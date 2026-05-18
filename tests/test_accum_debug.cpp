// Debug accumulate_dense_grad
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
    
    // Forward pass for real
    critic.reset_cached_inputs();
    Tensor out_real = critic.forward(real);
    cout << "D(real): " << out_real[0][0] << ", " << out_real[1][0] << endl;
    
    // Backward pass for real - manually compute what should happen
    // grad_output = [[1], [1]] (dL/dD = 1 for each sample)
    Tensor grad_out(2, 1);
    grad_out.fill(1.0);
    
    // The grad going into layer 2 is grad_out itself (identity for output layer)
    // For accumulate_dense_grad at layer 2:
    // grad_w2[0][k] = sum_r grad_out[r][0] * h1[r][k]
    
    // But we need to cache h1 (output of layer 1, before layer 2)
    // cached_inputs_[2] should be the input to layer 2 (output of layer 1)
    
    // Let's trace: forward does:
    // x = input (real)
    // x = layers_[0].forward(x)  -> pre0
    // x = leakyrelu(pre0) -> h0
    // x = layers_[1].forward(x)  -> pre1
    // x = leakyrelu(pre1) -> h1
    // x = layers_[2].forward(x)  -> pre2 (final output)
    // cached_inputs_[0] = input (real)
    // cached_inputs_[1] = pre0
    // cached_inputs_[2] = pre1 (input to layer 2 = output of layer 1 after leakyrelu? NO!)
    
    // Wait, let me re-read forward:
    // x = layers_[i].forward(x)  // Dense forward returns pre-activation
    // if (i + 1 < cached_inputs_.size())
    //     cached_inputs_[i + 1] = x;  // Cache the output of Dense (pre-activation)
    // if (i < layers_.size() - 1)
    //     x = activations_[i](x);  // Apply activation
    
    // So cached_inputs_[1] = pre0 (output of layer 0 Dense, before leakyrelu)
    // Then x = leakyrelu(pre0) = h0
    // x = layers_[1].forward(h0) -> pre1
    // cached_inputs_[2] = pre1 (output of layer 1 Dense, before leakyrelu)
    // Then x = leakyrelu(pre1) = h1
    // x = layers_[2].forward(h1) -> pre2
    // No caching for layer 2 output (last layer)
    
    // So for layer 2:
    // cached_inputs_[2] = pre1 (NOT h1!)
    // input to layer 2 is h1 (output of leakyrelu applied to pre1)
    
    // But accumulate_dense_grad uses cached_inputs_[i] which is the pre-activation!
    // That's wrong - the weight gradient should use the actual input to the layer (h1, not pre1)!
    
    // Wait no, let me re-check:
    // cached_inputs_[i] is the input to layer i
    // For i=0: cached_inputs_[0] = real (input to network)
    // For i=1: cached_inputs_[1] = pre0 (output of layer 0, input to layer 0's activation = pre0)
    // For i=2: cached_inputs_[2] = pre1 (output of layer 1, input to layer 1's activation = pre1)
    
    // So for layer 0, input is real (correct)
    // For layer 1, input is pre0, but we want h0 (the output after leakyrelu)
    // For layer 2, input is pre1, but we want h1 (the output after leakyrelu)
    
    // The issue is: we cache pre-activation but we need the post-activation (h) for the weight gradient!
    // grad_w1 = grad_pre1^T * h1, not grad_pre1^T * pre1!
    
    // But wait - the forward pass applies leakyrelu and caches the result:
    // x = activations_[i](x);  // This modifies x to be h
    // Then next layer: x = layers_[i+1].forward(x);  // This uses h as input!
    
    // So after forward, x is the output of the activation, not the pre-activation.
    // But we cache BEFORE applying activation...
    
    // Let me verify:
    // layers_[0].forward(x) returns pre0 and stores x (original input) in last_input
    // cached_inputs_[1] = pre0 (the returned value)
    // Then x = leakyrelu(pre0) overwrites x
    // layers_[1].forward(x) uses leakyrelu(pre0) as input, and returns pre1
    // So cached_inputs_[2] = pre1 (NOT h1)
    
    // But the actual input to layer 1 was h0 = leakyrelu(pre0)
    // So for computing grad_w1 = grad_pre1^T * h0, we need h0 but cached_inputs_[1] gives us pre0!
    
    // This is the bug! cached_inputs_[i] stores pre-activation but we need the post-activation (activation output) for the weight gradient!
    
    cout << "\nLet me verify by printing shapes and checking..." << endl;
    cout << "layers_.size() = " << critic.layers_.size() << endl;
    
    // Actually, wait. Let me trace through more carefully.
    // 
    // Layer 0 (i=0):
    // x = layers_[0].forward(x) -> pre0
    // cached_inputs_[1] = pre0
    // x = activations_[0](x) -> h0
    // 
    // Layer 1 (i=1):
    // x = layers_[1].forward(x) -> pre1 (where x was h0)
    // cached_inputs_[2] = pre1
    // x = activations_[1](x) -> h1
    // 
    // Layer 2 (i=2):
    // x = layers_[2].forward(x) -> pre2 (where x was h1)
    // No caching (last layer)
    
    // So cached_inputs_[0] = input to layer 0 = real
    // cached_inputs_[1] = pre0 = output of layer 0 Dense, before leakyrelu
    // cached_inputs_[2] = pre1 = output of layer 1 Dense, before leakyrelu
    
    // For weight gradient at layer 0:
    // grad_w0 = grad_pre0^T * cached_inputs_[0] = grad_pre0^T * real ✓
    
    // For weight gradient at layer 1:
    // grad_w1 = grad_pre1^T * cached_inputs_[1] = grad_pre1^T * pre0
    // But we need grad_w1 = grad_pre1^T * h0 = grad_pre1^T * leakyrelu(pre0)
    // So we use the wrong input! pre0 instead of h0!
    
    // For weight gradient at layer 2:
    // grad_w2 = grad_pre2^T * cached_inputs_[2] = grad_pre2^T * pre1
    // But we need grad_w2 = grad_pre2^T * h1 = grad_pre2^T * leakyrelu(pre1)
    // So we use the wrong input! pre1 instead of h1!
    
    // THIS IS THE BUG!
    
    // But wait - the numerical gradient test uses loss = D(real) + D(fake)
    // For the W[0][0] weight in layer 0, the gradient dL/dW0[0][0] involves:
    // - dD(real)/dW0[0][0] = real[0][0] * dD/dpre0[0][0] + real[1][0] * dD/dpre0[1][0]
    // - dD(fake)/dW0[0][0] = fake[0][0] * dD/dpre0[0][0] + fake[1][0] * dD/dpre0[1][0]
    
    // Where dD/dpre0 = dD/dh0 * dh0/dpre0 = dD/dh0 * leakyrelu'(pre0)
    // And dD/dh0 = sum_k (W2[0][k] * W1[k][0] * leakyrelu'(pre1))
    
    // But when we use cached_inputs_[0] = real, that's correct!
    // So layer 0 should be correct...
    
    // Let me check if maybe the issue is that cached_inputs_[i] for i>0 is pre-activation but we need post-activation...
    
    // Actually, wait - for layer 0, cached_inputs_[0] = real (the original input) which is correct
    // For layer 1, cached_inputs_[1] = pre0, but the actual input to layer 1 was h0 = leakyrelu(pre0)
    // For layer 2, cached_inputs_[2] = pre1, but the actual input to layer 2 was h1 = leakyrelu(pre1)
    
    // Let me trace what the network is actually computing for layer 1's weight gradient:
    // accumulate_dense_grad(layers_[1], grad_pre1, cached_inputs_[1])
    // cached_inputs_[1] = pre0 (the WRONG value - should be h0 = leakyrelu(pre0))
    
    // So grad_w1[m][n] = sum_r grad_pre1[r][m] * pre0[r][n]
    // But it SHOULD be: grad_w1[m][n] = sum_r grad_pre1[r][m] * h0[r][n]
    
    // This means for layer 1 and layer 2, the weight gradients are computed with wrong inputs!
    
    // But the test is for W[0][0] which is layer 0, not layer 1 or layer 2...
    // Unless... w1 is being used to refer to something else in the test?
    
    // Let me check the test:
    // Dense& w1 = critic.layer(0);  // This is layer 0!
    
    // So we're testing layer 0's weight gradient. 
    // For layer 0, cached_inputs_[0] = real (correct input).
    // So the input is correct.
    
    // The issue must be in grad_pre0 then...
    // grad_pre0 is computed as grad * leakyrelu'(pre0) where grad came from layer 1's backward.
    
    // grad = layers_[1].backward(grad, 0.0);
    // layers_[1].backward returns grad_input = grad_output * weights
    // grad_output = grad_pre1 (after leakyrelu multiplication at layer 1)
    // weights = w1.weights
    
    // So grad = grad_pre1 * w1.weights
    // = dL/dpre1 * w1.weights
    
    // Then at layer 0:
    // grad_pre0 = grad * leakyrelu'(pre0)
    // = dL/dpre1 * w1.weights * leakyrelu'(pre0)
    // = dL/dpre0 (correct!)
    
    // So grad_pre0 should be correct...
    
    // But wait, there's an issue in how the gradient flows through the network.
    // When we compute grad_pre0, we use the leakyrelu'(pre0) from cached_inputs_[0].
    // But cached_inputs_[0] is the input to the network (real), not pre0!
    
    // Let me re-check the code:
    // const Tensor& pre_i = cached_inputs_[i];  // pre-activation for layer i
    
    // For i=0: cached_inputs_[0] = input (real), NOT pre0!
    // For i=1: cached_inputs_[1] = pre0
    // For i=2: cached_inputs_[2] = pre1
    
    // So when computing grad_pre0, we use cached_inputs_[0] which is real, not pre0!
    // This is wrong! We should be using pre0 to compute the leakyrelu derivative!
    
    // The code says:
    // const Tensor& pre_i = cached_inputs_[i];  // pre-activation for layer i
    // But cached_inputs_[i] is NOT the pre-activation for layer i!
    // cached_inputs_[0] is the network input, not pre0
    // cached_inputs_[1] is pre0
    // cached_inputs_[2] is pre1
    
    // So the comment is wrong and the indexing is off by one!
    
    // The correct mapping should be:
    // cached_inputs_[0] = input (to layer 0) = pre0's input
    // cached_inputs_[1] = output of layer 0 (pre0 before activation)
    // cached_inputs_[2] = output of layer 1 (pre1 before activation)
    
    // For computing grad_pre0, we need pre0, but cached_inputs_[0] is the original input!
    // We should be using cached_inputs_[1] for pre0!
    
    // This is the bug! The cached_inputs_ array is indexed wrong for the backward pass!
    
    // Let me verify by checking the forward pass again:
    // cached_inputs_[0] = input;  // Set before loop
    // for (size_t i = 0; i < layers_.size(); ++i) {
    //     x = layers_[i].forward(x);
    //     if (i + 1 < cached_inputs_.size())
    //         cached_inputs_[i + 1] = x;
    //     if (i < layers_.size() - 1)
    //         x = activations_[i](x);
    // }
    
    // So cached_inputs_[0] = input (original)
    // cached_inputs_[1] = output of layer 0 = pre0
    // cached_inputs_[2] = output of layer 1 = pre1
    
    // But in backward_from:
    // for (size_t i = layers_.size(); i-- > 0; ) {
    //     const Tensor& pre_i = cached_inputs_[i];  // pre-activation for layer i
    //     ...
    // }
    
    // For i=0: pre_i = cached_inputs_[0] = input (WRONG! should be pre0 = cached_inputs_[1])
    // For i=1: pre_i = cached_inputs_[1] = pre0 (correct for layer 1)
    // For i=2: pre_i = cached_inputs_[2] = pre1 (correct for layer 2)
    
    // So the indexing is off by one for layer 0!
    // This explains the bug - for layer 0, we're using the wrong pre-activation values!
    
    // Let me verify: for layer 0, we use cached_inputs_[0] which is real (input)
    // But we should use cached_inputs_[1] which is pre0!
    
    // This means:
    // grad_pre0 = grad * slope where slope = (input > 0) ? 1 : alpha
    // But it should be: grad_pre0 = grad * slope where slope = (pre0 > 0) ? 1 : alpha
    
    // Since input != pre0, we get the wrong leakyrelu' values!
    // This would cause the gradient to be completely wrong!
    
    cout << "\nCHECK: cached_inputs_[0] is input (real), not pre0!" << endl;
    cout << "cached_inputs_[0].rows = " << critic.cached_inputs_[0].rows << endl;
    cout << "cached_inputs_[0].cols = " << critic.cached_inputs_[0].cols << endl;
    cout << "Should be using cached_inputs_[1] for pre0 in layer 0!" << endl;
    
    return 0;
}
