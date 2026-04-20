# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- `group_norm` — GroupNorm (divide channels into groups, normalize within group)
- `mixup_cutmix` — MixUp and CutMix data augmentation
- `unet` — U-Net (encoder-decoder with skip connections for segmentation)
- `cnn_models_vgg` — VGG-11/13/16/19 + ResNeXt block (block with grouped convolutions)
- `triplet_loss_siamese` — TripletLoss + SiameseNetwork (metric learning, contrastive loss)

## Done

- **round-9-refactor** ✅ — reorganized layers/ into subdirectories (attention, recurrent, normalization, convolutions, pooling, dense, generative, graph)
- **elastic_net** ✅, **random_forest** ✅, **isolation_forest** ✅, **lightgbm_style** ✅
- **tabular_ensemble** ✅, **adaboost** ✅, **gradient_boosting** ✅, **xgboost_style** ✅
- **coordconv** ✅, **gnn** ✅, **squeeze_excitation** ✅, **mixture_of_experts** ✅
- **capsnet** ✅, **clip_grad_norm** ✅, **one_cycle_lr** ✅, **focal_loss** ✅, **label_smoothing** ✅
- **densenet** ✅, **mobilenet_v2** ✅, **serialization_roundtrip_test** ✅, **lstm_las** ✅, **memory_network** ✅, **squeezenet** ✅, **numerical_stability_tests** ✅, **layer_timing_benchmark** ✅, **layer_output_tracker** ✅

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d
