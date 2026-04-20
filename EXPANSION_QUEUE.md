# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- **capsnet**: Capsule Network — capsules with dynamic routing by agreement, Mask-R佛教练, reconstruction loss. `CapsuleLayer`, `CapsNet`
- **coordconv**: CoordConv — appends (x,y) coordinate channels to input feature maps. `CoordConv2D`
- **gnn**: Graph Neural Network — GCN layer (Aggregate→Combine with adjacency), GAT (multi-head attention). `GCNLayer`, `GATLayer`, `GraphNetwork`
- **squeeze_excitation**: Squeeze-Excitation block — global avg pool → FC → ReLU → FC → Sigmoid, channel attention. `SEBlock`, `SEResNetBlock`
- **mixture_of_experts**: Mixture of Experts — sparse gating, load balancing loss, top-k routing. `MoELayer`, `SparseDispatcher`
- **focal_loss**: Focal Loss — for class-imbalanced classification. γ focus parameter, α weighting. `FocalLoss`
- **clip_grad_norm**: Gradient clipping by global norm — `clip_grad_norm_(params, max_norm)` utility
- **one_cycle_lr**: OneCycleLR scheduler — super-convergence: warmup → max_lr → anneal. `OneCycleLR`
- **label_smoothing**: Label smoothing cross-entropy — soft targets instead of hard labels. `LabelSmoothingCrossEntropy`

## Done

- **densenet** ✅
- **mobilenet_v2** ✅
- **serialization_roundtrip_test** ✅
- **lstm_las** ✅
- **memory_network** ✅
- **squeezenet** ✅
- **numerical_stability_tests** ✅
- **layer_timing_benchmark** ✅
- **layer_output_tracker** ✅

- **resnet_blocks**, **vae**, **lstm_bidirectional**, **spatial_dropout**, **weight_init_schemes**, **gru_layer**, **seq2seq_attention**, **swa**, **build_cnn_models**, **normalize_weights**
- **gradient_check**, **weight_decay_wrapper**, **multi_output_model**, **skip_connection**, **1d_pooling**, **mish_activation**, **early_stopping**, **model_summary**, **binary_serialization**, **softmax_cross_entropy_logits**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d