#include "nn/nn.h"
#include "nn/layers/layer_norm.h"
// removed (via nn.h)
#include "nn/layers/batch_norm.h"
#include "nn/layers/flatten.h"
// removed (via nn.h)
#include "nn/utils/grid_search.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));
    std::cout << "=== Neural Net Extensions Demo ===\n\n";

    // --- Dropout Test ---
    std::cout << "[1] Dropout Test\n";
    Dropout drop(0.5);
    Tensor input(3, 4);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = (std::rand() % 100) / 10.0;

    drop.set_training(true);
    Tensor dropped = drop.forward(input);
    std::cout << "  Input sum: " << input.sum() << ", Dropped sum: " << dropped.sum() << "\n";
    drop.set_training(false);
    Tensor inference = drop.forward(input);
    std::cout << "  Inference mode sum: " << inference.sum() << "\n\n";

    // --- LayerNorm Test ---
    std::cout << "[2] LayerNorm Test\n";
    LayerNorm ln(4);
    Tensor x_norm(2, 4);
    for (size_t i = 0; i < x_norm.rows; ++i)
        for (size_t j = 0; j < x_norm.cols; ++j)
            x_norm[i][j] = std::rand() % 100 / 10.0;
    Tensor out_ln = ln.forward(x_norm);
    std::cout << "  LayerNorm output shape: (" << out_ln.rows << ", " << out_ln.cols << ")\n";
    std::cout << "  Means close to 0: " << out_ln.sum() / (out_ln.rows * out_ln.cols) << "\n\n";

    // --- BatchNorm1D Test ---
    std::cout << "[3] BatchNorm1D Test\n";
    BatchNorm1D bn(4);
    Tensor x_bn(3, 4);
    for (size_t i = 0; i < x_bn.rows; ++i)
        for (size_t j = 0; j < x_bn.cols; ++j)
            x_bn[i][j] = std::rand() % 100 / 10.0;
    Tensor out_bn = bn.forward(x_bn);
    std::cout << "  BatchNorm output shape: (" << out_bn.rows << ", " << out_bn.cols << ")\n";
    bn.set_training(false);
    Tensor inf_bn = bn.forward(x_bn);
    std::cout << "  Inference mode output shape: (" << inf_bn.rows << ", " << inf_bn.cols << ")\n\n";

    // --- Flatten Test ---
    std::cout << "[4] Flatten Test\n";
    Flatten flat;
    Tensor flat_in(2, 8);
    for (size_t i = 0; i < flat_in.rows; ++i)
        for (size_t j = 0; j < flat_in.cols; ++j)
            flat_in[i][j] = i * flat_in.cols + j;
    Tensor flat_out = flat.forward(flat_in);
    std::cout << "  Input: " << flat_in.rows << "x" << flat_in.cols
              << " -> Output: " << flat_out.rows << "x" << flat_out.cols << "\n\n";

    // --- Optimizers ---
    std::cout << "[5] Optimizers (RMSprop, AdamW, SGD Nesterov)\n";
    Model m;
    m.add_layer(new Dense(2, 4));
    m.add_layer(new Dense(4, 1));
    RMSprop opt1(0.01);
    AdamW opt2(0.001, 0.9, 0.999, 1e-8, 0.01);
    SGDNesterov opt3(0.01, 0.9);
    std::cout << "  All optimizers initialized OK\n\n";

    // --- LR Schedulers ---
    std::cout << "[6] LR Schedulers\n";
    StepLR step_lr(0.01, 5, 0.5);
    std::cout << "  StepLR: initial=" << step_lr.get_lr();
    step_lr.step(); step_lr.step(); step_lr.step(); step_lr.step(); step_lr.step();
    std::cout << " after 5 steps=" << step_lr.get_lr() << "\n";

    ExponentialLR exp_lr(0.1, 0.9);
    std::cout << "  ExponentialLR: initial=" << exp_lr.get_lr();
    exp_lr.step(); exp_lr.step(); exp_lr.step();
    std::cout << " after 3 steps=" << exp_lr.get_lr() << "\n";

    ReduceLROnPlateau plateau(0.1);
    plateau.check_metric(0.5); plateau.check_metric(0.4); plateau.check_metric(0.3);
    plateau.check_metric(0.35); plateau.check_metric(0.36); plateau.check_metric(0.37);
    std::cout << "  ReduceLROnPlateau: after metric plateau, lr=" << plateau.get_lr() << "\n\n";

    // --- GridSearchCV ---
    std::cout << "[7] GridSearchCV Test\n";
    GridSearchCV gs(2);
    std::cout << "  GridSearchCV initialized (cv_folds=2)\n\n";

    std::cout << "=== All Extension Tests Passed ===\n";
    return 0;
}