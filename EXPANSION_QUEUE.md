# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

_(round 9 queue — to be filled)_

## Done

- **elastic_net** ✅ — `ElasticNet` (L1+L2 combined penalty), coordinate descent solver, soft-thresholding
- **random_forest** ✅ — `RandomForest` (bagging + random feature subset, info gain / MSE splits), majority vote / mean
- **isolation_forest** ✅ — `IsolationForest` (random splits, anomaly score = 2^{-avg_path/c(n)}), binary anomaly predictions
- **lightgbm_style** ✅ — `HistogramBoosting` (feature binning, histogram-based gradient boosting, best-first leaf growth)

- **tabular_ensemble** ✅, **adaboost** ✅, **gradient_boosting** ✅, **xgboost_style** ✅
- **coordconv** ✅, **gnn** ✅, **squeeze_excitation** ✅, **mixture_of_experts** ✅
- **capsnet** ✅, **clip_grad_norm** ✅, **one_cycle_lr** ✅, **focal_loss** ✅, **label_smoothing** ✅
- **densenet**, **mobilenet_v2**, **serialization_roundtrip_test**, **lstm_las**, **memory_network**, **squeezenet**, **numerical_stability_tests**, **layer_timing_benchmark**, **layer_output_tracker**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d