#ifndef TRAINER_H
#define TRAINER_H

#include "../core/model.h"
#include "../optimizers/optimizer.h"
#include "../layers/utility/dataloader.h"
#include <functional>

struct TrainerConfig {
    int epochs = 10;
    int accum_steps = 1;  // gradient accumulation steps (1 = no accumulation)
};

// Trainer: orchestrates the training loop with full control over
// epochs, batch iteration, callbacks, and logging.
class Trainer {
public:
    using EpochCallback = std::function<void(int epoch, double train_loss, double val_loss)>;
    using BatchCallback = std::function<void(int epoch, int batch_idx, double batch_loss)>;

    Trainer() = default;

    // Train with a DataLoader (mini-batch)
    void train(DataLoader& train_loader,
               DataLoader* val_loader,
               Model& model,
               Optimizer& optimizer,
               int epochs);

    // Train with explicit config
    void train(Model& model, const DataLoader& train_loader,
               const DataLoader* val_loader, Optimizer& opt, TrainerConfig config);

    // Single-batch training step (for manual loops)
    void train_step(Model& model, Optimizer& optimizer,
                    const Tensor& x_batch, const Tensor& y_batch,
                    double& loss_out);

    // Set callbacks
    void set_epoch_callback(EpochCallback cb) { epoch_cb_ = std::move(cb); }
    void set_batch_callback(BatchCallback cb) { batch_cb_ = std::move(cb); }
    void set_verbose(bool v) { verbose_ = v; }

private:
    EpochCallback epoch_cb_;
    BatchCallback batch_cb_;
    bool verbose_ = false;
};

// Computes cross-entropy loss for a batch given logits (N x C) and labels (N x C or N x 1)
double batch_cross_entropy(const Tensor& logits, const Tensor& targets);

// Utility: convert a vector of (1, C) target tensors into a single (batch_size, C) Tensor
Tensor targets_to_tensor(const std::vector<Tensor>& targets);

#endif
