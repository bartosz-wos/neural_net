# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

_(empty — round 4 complete)_

## Done

- **densenet** ✅ — `DenseBlock` (dense concat connections), `TransitionLayer` (compression), `DenseNet` encoder
- **mobilenet_v2** ✅ — `InvertedResidual` (expand→depthwise→project), ReLU6, skip conn, `MobileNetV2`
- **serialization_roundtrip_test** ✅ — `SerializationRoundtripTest` with save→load→verify, multi-input, stress test
- **lstm_las** ✅ — `LASEncoder`, `LASSelfAttention`, `LASDecoder`, `ListenAttendSpell`
- **memory_network** ✅ — MemN2N with `EmbeddingLayer`, hop layers, attention over memory slots
- **squeezenet** ✅ — `FireModule` (squeeze 1x1 → expand 1x1+3x3), `SqueezeNet` model
- **numerical_stability_tests** ✅ — NaN/Inf stress, softmax overflow, boundary ops, gradient clipping, activation extremes
- **layer_timing_benchmark** ✅ — `TimingBenchmark` with per-layer ms profiling, `run_model()`, CSV export
- **layer_output_tracker** ✅ — monitors activation min/max/mean/std/%zero per step, vanishing/exploding detection, CSV logging

- **resnet_blocks**, **vae**, **lstm_bidirectional**, **spatial_dropout**, **weight_init_schemes**, **gru_layer**, **seq2seq_attention**, **swa**, **build_cnn_models**, **normalize_weights**
- **gradient_check**, **weight_decay_wrapper**, **multi_output_model**, **skip_connection**, **1d_pooling**, **mish_activation**, **early_stopping**, **model_summary**, **binary_serialization**, **softmax_cross_entropy_logits**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d