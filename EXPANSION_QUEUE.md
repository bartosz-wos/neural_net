# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- **catboost_style**: CatBoost-style boosting — ordered boosting (anti-leakage), symmetric trees, `PerceptronCriterion` for classification, `CatBoostClassifier`
- **random_forest**: Random Forest — bagging + random feature subset at each split. `RandomForest` with parallel trees, majority vote
- **isolation_forest**: Isolation Forest — anomaly detection via random splits, anomaly score = 2^{-(avg_depth / c(n))}. `IsolationForest`
- **lightgbm_style**: LightGBM-style — histogram-based binning (feature binning for speed), leaf-wise growth (best-first), `HistogramBoosting`
- **elastic_net**: ElasticNet regularization — combined L1 (Lasso) + L2 (Ridge) penalty on weights, `ElasticNet` utility

## Done

- **tabular_ensemble** ✅ — `TabularDataset`, `DecisionStump` (Gini-split search)
- **adaboost** ✅ — `AdaBoost` with sample reweighting + weighted vote
- **gradient_boosting** ✅ — `GradientBoosting`, `RegressionTree`
- **xgboost_style** ✅ — `XGBoostTree` (λ, γ, min_child_weight), `XGBoostClassifier`

- **coordconv** ✅, **gnn** ✅, **squeeze_excitation** ✅, **mixture_of_experts** ✅
- **capsnet** ✅, **clip_grad_norm** ✅, **one_cycle_lr** ✅, **focal_loss** ✅, **label_smoothing** ✅
- **densenet**, **mobilenet_v2**, **serialization_roundtrip_test**, **lstm_las**, **memory_network**, **squeezenet**, **numerical_stability_tests**, **layer_timing_benchmark**, **layer_output_tracker**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d