#include "trainer.h"
#include "../activations/activations.h"
#include <iostream>

// Utility: convert a vector of (1, C) target tensors into a single (batch_size, C) Tensor
static Tensor targets_to_tensor_impl(const std::vector<Tensor>& targets) {
    if (targets.empty()) return Tensor(0, 0);
    size_t batch_size = targets.size();
    size_t cols = targets[0].cols;
    Tensor result(batch_size, cols);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t c = 0; c < cols; ++c) {
            result(i, c) = targets[i](0, c);
        }
    }
    return result;
}

void Trainer::train_step(Model& model, Optimizer& optimizer,
                         const Tensor& x_batch, const Tensor& y_batch,
                         double& loss_out) {
    // Forward pass
    Tensor logits = model.forward(x_batch);

    // Compute loss
    loss_out = softmax_cross_entropy(logits, y_batch);

    // Backward pass
    Tensor grad = softmax_cross_entropy_grad(logits, y_batch);
    model.backward(grad, 0.0); // learning rate handled by optimizer

    // Optimizer step
    optimizer.step(model);
}

void Trainer::train(DataLoader& train_loader,
                    DataLoader* val_loader,
                    Model& model,
                    Optimizer& optimizer,
                    int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_loader.reset();
        double total_loss = 0.0;
        int batch_count = 0;

        while (train_loader.has_next()) {
            DataLoader::Batch batch = train_loader.next();
            // Stack individual (1, features) samples into (batch_size, features) tensors
            Tensor X_batch(batch.batch_size, batch.data[0].cols);
            for (size_t i = 0; i < batch.batch_size; ++i) {
                for (size_t c = 0; c < X_batch.cols; ++c) {
                    X_batch(i, c) = batch.data[i](0, c);
                }
            }
            Tensor y_batch = targets_to_tensor_impl(batch.targets);

            double batch_loss = 0.0;
            train_step(model, optimizer, X_batch, y_batch, batch_loss);
            total_loss += batch_loss;
            batch_count++;

            if (batch_cb_) batch_cb_(epoch, batch_count, batch_loss);
        }

        double avg_train_loss = batch_count > 0 ? total_loss / batch_count : 0.0;
        double val_loss = 0.0;

        if (val_loader) {
            val_loader->reset();
            double val_total = 0.0;
            int val_count = 0;
            while (val_loader->has_next()) {
                DataLoader::Batch batch = val_loader->next();
                Tensor X_batch(batch.batch_size, batch.data[0].cols);
                for (size_t i = 0; i < batch.batch_size; ++i) {
                    for (size_t c = 0; c < X_batch.cols; ++c) {
                        X_batch(i, c) = batch.data[i](0, c);
                    }
                }
                Tensor y_batch = targets_to_tensor_impl(batch.targets);
                Tensor logits = model.forward(X_batch);
                val_total += softmax_cross_entropy(logits, y_batch);
                val_count++;
            }
            val_loss = val_count > 0 ? val_total / val_count : 0.0;
        }

        if (epoch_cb_) epoch_cb_(epoch, avg_train_loss, val_loss);
        if (verbose_) {
            std::cout << "Epoch " << epoch << " — train_loss: " << avg_train_loss
                      << " val_loss: " << val_loss << std::endl;
        }
    }
}

void Trainer::train(Model& model, const DataLoader& train_loader,
                       const DataLoader* val_loader, Optimizer& opt, TrainerConfig config) {
    model.set_accumulate_steps(config.accum_steps);

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        // NOTE: const cast needed because DataLoader doesn't expose non-const iterator
        auto& loader = const_cast<DataLoader&>(train_loader);
        loader.reset();
        double total_loss = 0.0;
        int batch_count = 0;

        while (loader.has_next()) {
            DataLoader::Batch batch = loader.next();
            Tensor X_batch(batch.batch_size, batch.data[0].cols);
            for (size_t i = 0; i < batch.batch_size; ++i) {
                for (size_t c = 0; c < X_batch.cols; ++c) {
                    X_batch(i, c) = batch.data[i](0, c);
                }
            }
            Tensor y_batch = targets_to_tensor_impl(batch.targets);

            // Forward pass
            Tensor logits = model.forward(X_batch);
            double batch_loss = softmax_cross_entropy(logits, y_batch);
            total_loss += batch_loss;

            // Backward pass
            Tensor grad = softmax_cross_entropy_grad(logits, y_batch);
            model.backward(grad, 0.0);

            // Gradient accumulation step
            model.step();
            if (model.should_update()) {
                model.apply_gradAccum(opt);
            }

            batch_count++;
            if (batch_cb_) batch_cb_(epoch, batch_count, batch_loss);
        }

        // Final update if we haven't reached accum_steps
        if (!model.should_update() && model.step_count() > 0) {
            model.apply_gradAccum(opt);
        }

        double avg_train_loss = batch_count > 0 ? total_loss / batch_count : 0.0;
        double val_loss = 0.0;

        if (val_loader) {
            auto& vloader = const_cast<DataLoader&>(*val_loader);
            vloader.reset();
            double val_total = 0.0;
            int val_count = 0;
            while (vloader.has_next()) {
                DataLoader::Batch batch = vloader.next();
                Tensor X_batch(batch.batch_size, batch.data[0].cols);
                for (size_t i = 0; i < batch.batch_size; ++i) {
                    for (size_t c = 0; c < X_batch.cols; ++c) {
                        X_batch(i, c) = batch.data[i](0, c);
                    }
                }
                Tensor y_batch = targets_to_tensor_impl(batch.targets);
                Tensor logits = model.forward(X_batch);
                val_total += softmax_cross_entropy(logits, y_batch);
                val_count++;
            }
            val_loss = val_count > 0 ? val_total / val_count : 0.0;
        }

        if (epoch_cb_) epoch_cb_(epoch, avg_train_loss, val_loss);
        if (verbose_) {
            std::cout << "Epoch " << epoch << " — train_loss: " << avg_train_loss
                      << " val_loss: " << val_loss << std::endl;
        }
    }
}

double batch_cross_entropy(const Tensor& logits, const Tensor& targets) {
    return softmax_cross_entropy(logits, targets);
}

Tensor targets_to_tensor(const std::vector<Tensor>& targets) {
    return targets_to_tensor_impl(targets);
}
