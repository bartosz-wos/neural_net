# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Done

- **TanhPlus + Snake activations**: Implement `TanhPlus(x) = x + tanh(x)` (smooth, unbounded above, bounded below at ~-0.76) and `Snake(x) = x + (1/beta) * sin^2(beta*x)` (periodic activation, learnable beta). Both go in `include/nn/activations/activations.h` and `activations.cpp`. Snake is from the paper "Snake: Sinusoidal Activation Functions" — good for periodic pattern recognition.

- **Spectral Normalization**: Implement spectral_norm_ as a layer wrapper that constrains the spectral norm of weight matrices (Lipshitz constant = 1). Essential for stabilizing GANs (SNGAN) and differentiable physics. Implement as a wrapper class SpectralNorm that normalizes W by its largest singular value each forward pass. Add to `include/nn/layers/normalization/spectral_norm.{h,cpp}`.

- **RMSNorm**: Implement RMSNorm layer (Root Mean Square Normalization, used in LLaMA/Mistral). Implement RMSNorm layer (Root Mean Square Normalization, used in LLaMA/Mistral). Formula: y = (x / RMS) * gamma where RMS = sqrt(mean(x^2) + eps). Remove mean-centering (unlike LayerNorm) — more numerically stable and faster. Added to layers/normalization/rms_norm.{h,cpp}.

- **DataLoader / Dataset infrastructure**: Implement MNIST-style mini-batch loader with shuffle, batch iteration, TensorDataset wrapper, and integrate into Model::train() method. Add DataLoader class that wraps a Dataset and yields mini-batches. Implement a Trainer class that handles epoch-level training loops with DataLoader support.

- **FlashAttention-style memory-efficient attention**: Implement memory-efficient multi-head attention with tiled computation, online softmax normalization, and exact gradient computation. Include FlashAttentionLayer class with forward/backward passes that compute attention in O(sequence_length) memory instead of O(sequence_length^2). Add CPU tiling fallback for non-AVX systems. Benchmark against standard attention for correctness and memory savings.

- **Fix GATLayer backward pass**: The GATLayer backward() in gnn.cpp currently returns a stub tensor. Implement full attention gradient backprop: dL/dW_heads, dL/d_a_heads, and dL/d_input. This requires computing attention weight gradients via softmax jacobian and chaining through the LeakyReLU attention scores.

- **GELU + SwiGLU**: Implement GELU activation (Gaussian Error Linear Unit, from "Gaussian Error Linear Units (GELU)" — used in BERT, GPT, ViT) with exact formula: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. Also implement SwiGLU (Swish-Gated Linear Unit from "GLU Variants Improve Transformer") as a block: `SiLU(x) * W * x` with a gating branch. Add both to `include/nn/activations/activations.h` and `activations.cpp`.
