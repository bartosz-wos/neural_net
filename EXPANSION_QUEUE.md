# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Ideas

- **GIN (Graph Isomorphism Network)**: Implement GIN layer: h_{k+1} = MLP((1+eps_k) * h_k + sum_{j in N(i)} h_j). Most powerful GNN variant for graph classification. Add to `layers/architectures/gin.{h,cpp}`.

- **DeepGCN / GCNII**: Advanced GCN variants with residual connections and MLPs for deep graphs. Add to `layers/architectures/deep_gcn.{h,cpp}`.

- **Nyström attention**: Efficient attention via low-rank approximation using Nyström method. Approximates softmax(KQ)V with landmark points. Good for long sequences. Add to `layers/attention/nystrom_attention.{h,cpp}`.

- **CRATE**: Convolutions with Rectified Activations — uses channel attention with ReLU. Implement as a lightweight efficient architecture.

- **LightGCN**: Simplified Graph Convolutional Network — removes non-linearities and uses only linear propagation for recommendation. Very simple but effective. Add to `layers/architectures/lightgcn.{h,cpp}`.

## Done

- **AdaBelief optimizer**: Implemented — belief variance instead of second moment. Uses residual variance E[(grad - m_prev)^2] to measure gradient "surprise". Better generalization than Adam in image classification and language modeling. AdamW-style weight decay. Full bias-corrected moment update. Tests: 4/4 pass.

- **Group Conv / DepthwiseSeparable / Ghost Module / ECA / CBAM / SCConv / CoordAttention**: Implemented modern conv variants: GroupConv (grouped convolutions), DepthwiseSeparableConv (DW+PW), GhostModule (cheap feature maps from cheap operations), ECA (efficient channel attention), CBAM (channel+spatial attention), SCConv (spatial and channel reconstruction), CoordAttention. Added to `layers/convolutions/` and `layers/normalization/`.

- **LAMB optimizer**: Implemented — Layer-wise Adaptive Moment estimation with trust ratio normalization. Add to `optimizers/lamb.{h,cpp}` and registered. Full bias-corrected moment update with per-parameter trust ratio clamping.

- **GELU + SwiGLU**: Already implemented — GELU in `activations.h`/`.cpp`, SwiGLU as a template class in `layers/utility/swiglu.{h,cpp}`.

- **TanhPlus + Snake activations**: Implement `TanhPlus(x) = x + tanh(x)` (smooth, unbounded above, bounded below at ~-0.76) and `Snake(x) = x + (1/beta) * sin^2(beta*x)` (periodic activation, learnable beta). Both go in `include/nn/activations/activations.h` and `activations.cpp`. Snake is from the paper "Snake: Sinusoidal Activation Functions" — good for periodic pattern recognition.

- **Normalizing Flow layers**: Implemented RealNVP-style invertible flows with AffineCoupling stack, coupling layers, forward/inverse passes, log-det Jacobian. Fixed batch dimension mismatch bug in CouplingLayer::inverse() and AffineCoupling::sample() — s/t networks operate on single-row input, not batched. Tests: 12/12 pass.

- **Coordinate Networks (CoN)**: Implement MLPs augmented with Fourier features for learning high-frequency functions on spatial domains. Good for implicit neural representations and SDFs.

- **Spectral Normalization**: Implement spectral_norm_ as a layer wrapper that constrains the spectral norm of weight matrices (Lipshitz constant = 1). Essential for stabilizing GANs (SNGAN) and differentiable physics. Implement as a wrapper class SpectralNorm that normalizes W by its largest singular value each forward pass. Add to `include/nn/layers/normalization/spectral_norm.{h,cpp}`.

- **RMSNorm**: Implement RMSNorm layer (Root Mean Square Normalization, used in LLaMA/Mistral). Formula: y = (x / RMS) * gamma where RMS = sqrt(mean(x^2) + eps). Remove mean-centering (unlike LayerNorm) — more numerically stable and faster. Added to layers/normalization/rms_norm.{h,cpp}.

- **DataLoader / Dataset infrastructure**: Implement MNIST-style mini-batch loader with shuffle, batch iteration, TensorDataset wrapper, and integrate into Model::train() method. Add DataLoader class that wraps a Dataset and yields mini-batches. Implement a Trainer class that handles epoch-level training loops with DataLoader support.