# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Ideas

## Done

- **GELU + SwiGLU**: Already implemented — GELU in `activations.h`/`.cpp`, SwiGLU as a template class in `layers/utility/swiglu.{h,cpp}`.

## Done

- **TanhPlus + Snake activations**: Implement `TanhPlus(x) = x + tanh(x)` (smooth, unbounded above, bounded below at ~-0.76) and `Snake(x) = x + (1/beta) * sin^2(beta*x)` (periodic activation, learnable beta). Both go in `include/nn/activations/activations.h` and `activations.cpp`. Snake is from the paper "Snake: Sinusoidal Activation Functions" — good for periodic pattern recognition.

- **Normalizing Flow layers**: Implemented RealNVP-style invertible flows with AffineCoupling stack, coupling layers, forward/inverse passes, log-det Jacobian. Fixed batch dimension mismatch bug in CouplingLayer::inverse() and AffineCoupling::sample() — s/t networks operate on single-row input, not batched. Tests: 12/12 pass.

- **Coordinate Networks (CoN)**: Implement MLPs augmented with Fourier features for learning high-frequency functions on spatial domains. Good for implicit neural representations and SDFs.

- **Spectral Normalization**: Implement spectral_norm_ as a layer wrapper that constrains the spectral norm of weight matrices (Lipshitz constant = 1). Essential for stabilizing GANs (SNGAN) and differentiable physics. Implement as a wrapper class SpectralNorm that normalizes W by its largest singular value each forward pass. Add to `include/nn/layers/normalization/spectral_norm.{h,cpp}`.

- **RMSNorm**: Implement RMSNorm layer (Root Mean Square Normalization, used in LLaMA/Mistral). Formula: y = (x / RMS) * gamma where RMS = sqrt(mean(x^2) + eps). Remove mean-centering (unlike LayerNorm) — more numerically stable and faster. Added to layers/normalization/rms_norm.{h,cpp}.

- **DataLoader / Dataset infrastructure**: Implement MNIST-style mini-batch loader with shuffle, batch iteration, TensorDataset wrapper, and integrate into Model::train() method. Add DataLoader class that wraps a Dataset and yields mini-batches. Implement a Trainer class that handles epoch-level training loops with DataLoader support.