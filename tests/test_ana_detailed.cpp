// More detailed analytical gradient computation
#include <iostream>
#include <iomanip>
#include <cmath>
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
    
    // Get weights
    Dense& w0 = critic.layer(0);
    Dense& w1 = critic.layer(1);
    Dense& w2 = critic.layer(2);
    
    // Let's trace the backward pass step by step
    // For real:
    // grad_output = [[1],[1]] (2x1)
    
    // Layer 2 backward:
    // grad_pre2 = grad_output * 1 (no activation) = [[1],[1]]
    // grad_w2 = grad_pre2^T @ h1 = [[1,1]] @ h1 = h1[0][:] + h1[1][:]
    // grad_h1_input = grad_pre2 @ w2.weights = [[1],[1]] @ w2.weights(1,8) = 1x8 row vector repeated for each sample
    
    // Actually, Dense::backward(grad) computes:
    // grad_input = grad @ W^T (where W is weights matrix out_dim x in_dim)
    // So for layer 2: grad_input2 = grad @ w2.weights^T
    // w2.weights is (1, 8), w2.weights^T is (8, 1)
    // grad is (2, 1)
    // grad_input2 = (2, 1) @ (1, 8) = (2, 8) ??? Wait, matmul shapes
    
    // Let me check Dense::backward signature
    cout << "w2.weights shape: " << w2.weights.rows << "x" << w2.weights.cols << endl;
    cout << "grad_output shape: 2x1" << endl;
    
    // In accumulate_dense_grad:
    // grad_w = grad_output^T @ input
    // grad_output: (batch, out_features)
    // input: (batch, in_features)
    
    // For layer 2: out_features = 1, in_features = 8
    // grad_output = (2, 1), input (h1) = (2, 8)
    // grad_w2 = (1, 2) @ (2, 8) = (1, 8) -- correct
    
    // Then Dense::backward computes grad_input = grad @ W^T
    // grad = (2, 1), W^T = (8, 1)
    // grad_input = (2, 1) @ (8, 1) -- this doesn't work!
    
    // Unless... W is stored as (in_features, out_features)?
    // Let me check
    
    cout << "\nw0.weights: " << w0.weights.rows << "x" << w0.weights.cols << " (in_features=2, out_features=8)" << endl;
    cout << "w1.weights: " << w1.weights.rows << "x" << w1.weights.cols << " (in_features=8, out_features=8)" << endl;
    cout << "w2.weights: " << w2.weights.rows << "x" << w2.weights.cols << " (in_features=8, out_features=1)" << endl;
    
    // So weights are (out_features, in_features)
    // forward: input (batch, in) @ W^T (in, out) = (batch, out) ✓
    // backward(grad): grad (batch, out) @ W (out, in) = (batch, in) ✓
    
    // So for layer 2:
    // grad_input = grad @ W2 = (2, 1) @ (1, 8) = (2, 8) ✓
    
    // But wait, in the backward_from loop, the grad being passed to accumulate_dense_grad
    // for layer i is the same grad that goes into layer->backward for layer i.
    // This grad is dL/d(output of layer i), not dL/d(pre_activation of layer i).
    
    // For layer 2: grad = dL/d(output of layer 2) = [[1],[1]]
    // This is what gets used in accumulate_dense_grad for layer 2
    // But for layer 2, output of layer 2 IS the pre-activation (no activation)
    
    // For layer 1: grad = dL/d(output of layer 1) = dL/d(h1) = grad_input from layer 2
    // This is dL/d(post-leakyrelu of layer 1)
    
    // But accumulate_dense_grad for layer 1 uses grad = dL/d(h1)
    // and input = cached_inputs_[1] = h0 (post-leakyrelu of layer 0)
    
    // But for a correct gradient, we need dL/dpre1, not dL/dh1!
    // dL/dW1 = dL/dh1 * dh1/dpre1 @ h0^T = dL/dh1 * leakyrelu'(pre1) @ h0^T
    
    // So the issue is that accumulate_dense_grad is using the WRONG gradient!
    // It's using dL/d(output) which is dL/d(post-activation), but for computing
    // weight gradients, we need dL/d(pre-activation) for layers with activations.
    
    // Let me check what Dense::backward actually does with the grad...
    
    // Actually, wait. Let me re-read backward_from:
    // for (size_t i = layers_.size(); i-- > 0; ) {
    //     accumulate_dense_grad(layers_[i], grad, cached_inputs_[i]);
    //     if (i > 0) {
    //         grad = layers_[i].backward(grad, 0.0);
    //     }
    // }
    
    // For i = 2:
    //   grad_w2 += grad_output^T @ cached_input[2]
    //   grad = layers_[2].backward(grad) = grad @ W2^T (since no activation)
    // For i = 1:
    //   grad_w1 += grad_input_for_layer1^T @ cached_input[1]
    //   grad = layers_[1].backward(grad)  -- but grad here is dL/dh1, not dL/dpre1!
    
    // The problem: accumulate_dense_grad for layer 1 uses grad = dL/dh1
    // But it should use grad = dL/dpre1 = dL/dh1 * leakyrelu'(pre1)
    
    // The backward call after accumulate doesn't fix this because accumulate
    // already used the wrong grad!
    
    // Actually wait. Let me think again...
    // The grad flowing backward through the network IS dL/d(output of layer i)
    // For layer 1, grad = dL/dh1
    
    // But for weight gradients:
    // dL/dW1 = dL/dpre1 @ h0^T
    // dL/dpre1 = dL/dh1 * dh1/dpre1 = dL/dh1 * leakyrelu'(pre1)
    
    // So the issue is: accumulate_dense_grad is using dL/dh1 but it should
    // use dL/dpre1!
    
    // For layers WITHOUT activation (like layer 2), dL/dpre2 = dL/dpost2 = dL/doutput2
    // so there's no issue.
    
    // For layers WITH activation, there's a mismatch.
    
    // But wait... in the backward pass, layer->backward(grad) DOES apply the
    // activation derivative! So the grad flowing to the previous layer IS
    // dL/dpre1 (after applying leakyrelu' to dL/dh1).
    
    // But accumulate_dense_grad is called BEFORE the backward that applies
    // the activation derivative! So accumulate uses grad = dL/dpost, not dL/dpre!
    
    // IS THIS THE BUG?!?!?!
    
    // Let me verify:
    // For layer 1 in backward_from:
    // 1. accumulate_dense_grad(layers_[1], grad, cached_inputs_[1])
    //    This uses grad = dL/dh1 (the grad flowing into layer 1 from layer 2)
    //    But for correct weight gradients, we need dL/dpre1 = dL/dh1 * leakyrelu'(pre1)
    //
    // 2. grad = layers_[1].backward(grad)
    //    This returns: dL/dpre1 = dL/dh1 * leakyrelu'(pre1) (for input grad)
    //    and also computes layer1.grad_input = dL/dpre1 @ W1^T
    //
    // But the damage is done - accumulate already used the wrong gradient!
    
    // Wait, but the grad used in accumulate for layer i is the grad flowing
    // INTO layer i from layer i+1, which IS dL/d(output of layer i).
    
    // For layer 1, output of layer 1 is h1 = leakyrelu(pre1).
    // So dL/d(output of layer 1) = dL/dh1.
    // But dL/dpre1 = dL/dh1 * leakyrelu'(pre1).
    
    // So yes, accumulate_dense_grad for layer 1 uses dL/dh1 instead of dL/dpre1.
    // This is wrong!
    
    // The fix would be to either:
    // 1. Have accumulate use the gradient AFTER applying activation derivative
    // 2. Or change the order: call backward FIRST (which applies activation derivative
    //    and returns grad_input), then use that for accumulate
    
    // But wait... let me reconsider. In the backward pass:
    // - grad_input (returned by backward) is dL/d(input to layer) which is dL/dpre for the previous layer
    // - But the weight gradient computation inside Dense::backward uses dL/dpre (not dL/dpost)
    
    // So Dense::backward internally computes correct weight gradients because it has
    // access to pre-activation values.
    
    // But backward_from uses accumulate_dense_grad which bypasses Dense::backward!
    // So the weight gradients in Dense::backward are NOT being used.
    
    // Let me verify that Dense::backward actually computes weight gradients...
    
    // In Layer base class, there's a virtual backward method.
    // For Dense, it probably computes weight gradients using pre-activation values.
    // But backward_from calls accumulate_dense_grad separately, which OVERWRITES
    // or ACCUMULATES to layer.grad_weights.
    
    // Actually wait - accumulate_dense_grad does +=, not =.
    // So it accumulates ON TOP of what Dense::backward computed.
    
    // But in backward_from, we call backward(grad) which computes gradients internally,
    // AND we also call accumulate_dense_grad which adds more to grad_weights.
    
    // This seems like double computation of gradients?
    // Or maybe the gradients from Dense::backward are not what we want?
    
    // Let me check: does layer->backward(grad, 0.0) accumulate or overwrite grad_weights?
    
    return 0;
}
