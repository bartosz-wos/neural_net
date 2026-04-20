# Expansion Queue (LIFO)

Items are implemented from the bottom up (last item = next to pop).
After completing an item, move it to the "Done" section.

## Queue (bottom = next to pop)

- **tabular_ensemble**: Tabular data Ensemble base — dataset loader for tabular CSV, train/test split, minmax normalization. `TabularDataset`, `TabularLoader`
- **decision_stump**: Decision stump (1-level decision tree) as weak learner. `DecisionStump` with Gini/entropy split search
- **adaboost**: AdaBoost — sequential boosting with sample reweighting, weighted vote. `AdaBoost` with `DecisionStump` weak learners
- **gradient_boosting**: Generic Gradient Boosting — sequential learners fit negative gradient of loss, `GradientBoosting` with customizable loss (MSE, cross-entropy)
- **xgboost_style**: XGBoost-style boosting — regularization term (γT + λ/2 * ||w||^2), approximate histogram binning, gradient statistics per node. `XGBoostTree`, `XGBoostClassifier`

## Done

- **coordconv** ✅
- **gnn** ✅
- **squeeze_excitation** ✅
- **mixture_of_experts** ✅
- **capsnet** ✅, **clip_grad_norm** ✅, **one_cycle_lr** ✅, **focal_loss** ✅, **label_smoothing** ✅
- **densenet**, **mobilenet_v2**, **serialization_roundtrip_test**, **lstm_las**, **memory_network**, **squeezenet**, **numerical_stability_tests**, **layer_timing_benchmark**, **layer_output_tracker**

## Pre-queue Done

mlp_helper, dataloader, cosine_annealing_lr, pool_optimizer_step, model_serialization (text), weight_init, gelu_activation, leaky_relu, conv1d, dilated_conv2d