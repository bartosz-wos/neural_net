# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- **spectral_norm**: Spectral normalization for weight matrices — stabilizes GAN training. `spectral_norm_(W, n_power_iters=1)` utility
- **pixel_shuffle**: Pixel shuffle (depth-to-space) for super-resolution upscaling. `PixelShuffle` with subpixel_conv
- **wgan_gp**: Wasserstein GAN with gradient penalty (WGAN-GP). `WGAN_GP` loss with gradient penalty term, `discriminator_optimizer_step()`
- **attention_pooling**: Attention-based pooling — learnable query, key, value projection + softmax similarity. `AttentionPooling` layer

## Done

- **coordconv** ✅ — `CoordConv2D` injects x,y normalized coords as channels before conv
- **gnn** ✅ — `GCNLayer` (normalized adjacency), `GATLayer` (multi-head attention with LeakyReLU), `GraphNetwork`
- **squeeze_excitation** ✅ — `SEBlock` (GAP→FC→ReLU→FC→Sigmoid), `SEResNetBlock` residual with channel attention
- **mixture_of_experts** ✅ — `MoELayer` with gating + top-k dispatch, `SparseDispatcher` with load balancing loss

- **capsnet** ✅ — `CapsuleLayer` with dynamic routing, `CapsNet` encoder-decoder
- **clip_grad_norm** ✅ — `clip_grad_norm_()` by global norm
- **one_cycle_lr** ✅ — `OneCycleLR` warmup + cosine annealing
- **focal_loss** ✅ — `FocalLoss` with γ focusing parameter
- **label_smoothing** ✅ — `LabelSmoothingCrossEntropy`

- **densenet**, **mobilenet_v2**, **serialization_roundtrip_test**, **lstm_las**, **memory_network**, **squeezenet**, **numerical_stability_tests**, **layer_timing_benchmark**, **layer_output_tracker**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d