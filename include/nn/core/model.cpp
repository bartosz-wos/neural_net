#include "model.h"
#include "../activations/activations.h"
#include <fstream>
#include <cstdint>
#include <cstring>
#include "../layers/normalization/batch_norm.h"
#include "../layers/normalization/layer_norm.h"
#include "../layers/convolutions/conv_layer.h"
#include "../layers/dense/embedding.h"
#include "../layers/recurrent/lstm.h"
#include "../layers/skip_connection.h"
#include "../layers/pooling/pool_layer.h"
#include "../layers/dense/flatten.h"
#include "../layers/attention/transformer.h"
#include "../layers/conv1d.h"
#include "../layers/recurrent/rnn.h"
#include <iostream>

void Model::add_layer(Layer* layer) {
    layers.emplace_back(layer);
}

Tensor Model::forward(const Tensor& input) {
    Tensor current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

Tensor Model::backward(Tensor grad, double learning_rate) {
    // Backprop through layers in reverse order
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
    return grad;
}

void Model::train(const Tensor& X, const Tensor& y, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            Tensor sample(1, X.cols, &X.data[i * X.cols]);
            Tensor label(1, y.cols, &y.data[i * y.cols]);

            // Forward
            Tensor prediction = forward(sample);

            // Compute loss gradient (MSE: 2*(pred - label) / N, but for single sample simpler)
            Tensor grad = (prediction - label) * (2.0 / static_cast<double>(X.cols));

            // Backward
            backward(grad, learning_rate);

            // Update weights
            for (auto& layer : layers) {
                layer->update_weights(learning_rate);
                layer->zero_grad();
            }

            // Accumulate MSE loss
            double loss = 0.0;
            for (size_t j = 0; j < prediction.cols; ++j) {
                double diff = prediction[0][j] - label[0][j];
                loss += diff * diff;
            }
            total_loss += loss;
        }
        double avg_loss = total_loss / static_cast<double>(X.rows);
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << std::endl;
        }
    }
}

void Model::train_cross_entropy(const Tensor& X, const Tensor& y, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            Tensor sample(1, X.cols, &X.data[i * X.cols]);
            Tensor label(1, y.cols, &y.data[i * y.cols]);

            // Forward pass to get logits
            Tensor logits = forward(sample);
            Tensor probs = Softmax()(logits);
            double loss = Softmax::cross_entropy_loss(logits, label);
            total_loss += loss;

            // Gradient: dL/dz = softmax(z) - y (for one-hot)
            Tensor grad = probs - label;

            backward(grad, learning_rate);
            for (auto& layer : layers) {
                layer->update_weights(learning_rate);
                layer->zero_grad();
            }
        }
        double avg_loss = total_loss / static_cast<double>(X.rows);
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " CE loss: " << avg_loss << std::endl;
        }
    }
    // void return
}

double Model::evaluate(const Tensor& X, const Tensor& y) {
    double total_loss = 0.0;
    for (size_t i = 0; i < X.rows; ++i) {
        Tensor sample(1, X.cols, &X.data[i * X.cols]);
        Tensor label(1, y.cols, &y.data[i * y.cols]);
        Tensor prediction = forward(sample);
        double loss = 0.0;
        for (size_t j = 0; j < prediction.cols; ++j) {
            double diff = prediction[0][j] - label[0][j];
            loss += diff * diff;
        }
        total_loss += loss;
    }
    return total_loss / static_cast<double>(X.rows);
}

void Model::save(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + path);
    // Magic header
    out.write("NNBIN", 5);
    uint8_t version = 1;
    out.write(reinterpret_cast<char*>(&version), 1);
    uint32_t n = static_cast<uint32_t>(layers.size());
    out.write(reinterpret_cast<char*>(&n), 4);
    for (const auto& layer : layers) {
        if (auto* d = dynamic_cast<Dense*>(layer.get())) {
            uint8_t type = 0; out.write(reinterpret_cast<char*>(&type), 1);
            uint32_t rows = static_cast<uint32_t>(d->weights.rows);
            uint32_t cols = static_cast<uint32_t>(d->weights.cols);
            out.write(reinterpret_cast<char*>(&rows), 4);
            out.write(reinterpret_cast<char*>(&cols), 4);
            out.write(reinterpret_cast<char*>(d->weights.data.data()), sizeof(double) * rows * cols);
            uint32_t bcols = static_cast<uint32_t>(d->bias.cols);
            out.write(reinterpret_cast<char*>(&bcols), 4);
            out.write(reinterpret_cast<char*>(d->bias.data.data()), sizeof(double) * bcols);
        } else if (auto* bn = dynamic_cast<BatchNorm1D*>(layer.get())) {
            uint8_t type = 1; out.write(reinterpret_cast<char*>(&type), 1);
            uint32_t ch = static_cast<uint32_t>(bn->gamma.rows);
            out.write(reinterpret_cast<char*>(&ch), 4);
            out.write(reinterpret_cast<char*>(bn->gamma.data.data()), sizeof(double) * ch);
            out.write(reinterpret_cast<char*>(bn->beta.data.data()), sizeof(double) * ch);
            out.write(reinterpret_cast<char*>(bn->running_mean.data.data()), sizeof(double) * ch);
            out.write(reinterpret_cast<char*>(bn->running_var.data.data()), sizeof(double) * ch);
        } else if (auto* ln = dynamic_cast<LayerNorm*>(layer.get())) {
            uint8_t type = 2; out.write(reinterpret_cast<char*>(&type), 1);
            uint32_t dim = static_cast<uint32_t>(ln->gamma.rows);
            out.write(reinterpret_cast<char*>(&dim), 4);
            out.write(reinterpret_cast<char*>(ln->gamma.data.data()), sizeof(double) * dim);
            out.write(reinterpret_cast<char*>(ln->beta.data.data()), sizeof(double) * dim);
        } else {
            uint8_t type = 255; out.write(reinterpret_cast<char*>(&type), 1);
        }
    }
    out.close();
}

Model Model::load(const std::string& path) {
    Model model;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading: " + path);
    char magic[6] = {0};
    in.read(magic, 5);
    if (std::string(magic) != "NNBIN") throw std::runtime_error("Invalid model file: bad magic");
    uint8_t version; in.read(reinterpret_cast<char*>(&version), 1);
    uint32_t n; in.read(reinterpret_cast<char*>(&n), 4);
    for (uint32_t idx = 0; idx < n; ++idx) {
        uint8_t type; in.read(reinterpret_cast<char*>(&type), 1);
        if (type == 0) { // Dense
            uint32_t rows, cols;
            in.read(reinterpret_cast<char*>(&rows), 4);
            in.read(reinterpret_cast<char*>(&cols), 4);
            Dense* d = new Dense(cols, rows);
            in.read(reinterpret_cast<char*>(d->weights.data.data()), sizeof(double) * rows * cols);
            uint32_t bcols; in.read(reinterpret_cast<char*>(&bcols), 4);
            in.read(reinterpret_cast<char*>(d->bias.data.data()), sizeof(double) * bcols);
            model.add_layer(d);
        } else if (type == 1) { // BatchNorm1D
            uint32_t ch; in.read(reinterpret_cast<char*>(&ch), 4);
            BatchNorm1D* bn = new BatchNorm1D(ch);
            in.read(reinterpret_cast<char*>(bn->gamma.data.data()), sizeof(double) * ch);
            in.read(reinterpret_cast<char*>(bn->beta.data.data()), sizeof(double) * ch);
            in.read(reinterpret_cast<char*>(bn->running_mean.data.data()), sizeof(double) * ch);
            in.read(reinterpret_cast<char*>(bn->running_var.data.data()), sizeof(double) * ch);
            model.add_layer(bn);
        } else if (type == 2) { // LayerNorm
            uint32_t dim; in.read(reinterpret_cast<char*>(&dim), 4);
            LayerNorm* ln = new LayerNorm(dim);
            in.read(reinterpret_cast<char*>(ln->gamma.data.data()), sizeof(double) * dim);
            in.read(reinterpret_cast<char*>(ln->beta.data.data()), sizeof(double) * dim);
            model.add_layer(ln);
        } else {
            // type == 255: skip layer (Conv2D, Embedding, RNN, Activation)
        }
    }
    return model;
}


Model create_mlp(const std::vector<size_t>& sizes, const std::string& activation) {
    Model model;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        model.add_layer(new Dense(sizes[i], sizes[i+1]));
        if (i + 1 < sizes.size() - 1) {
            if (activation == "relu") {
                model.add_layer(new Activation<ReLU>(ReLU{}));
            } else if (activation == "sigmoid") {
                model.add_layer(new Activation<Sigmoid>(Sigmoid{}));
            } else if (activation == "tanh") {
                model.add_layer(new Activation<Tanh>(Tanh{}));
            } else {
                model.add_layer(new Activation<ReLU>(ReLU{}));  // default
            }
        }
    }
    return model;
}

void Model::train(const Tensor& X, const Tensor& y, Optimizer& opt, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.rows; ++i) {
            Tensor sample(1, X.cols, &X.data[i * X.cols]);
            Tensor label(1, y.cols, &y.data[i * y.cols]);

            // Forward
            Tensor prediction = forward(sample);

            // Compute loss gradient (MSE)
            Tensor grad = (prediction - label) * (2.0 / static_cast<double>(X.cols));

            // Backward
            backward(grad, 0.0);

            // Optimizer step
            opt.step(*this);

            // Accumulate MSE loss
            double loss = 0.0;
            for (size_t j = 0; j < prediction.cols; ++j) {
                double diff = prediction[0][j] - label[0][j];
                loss += diff * diff;
            }
            total_loss += loss;
        }
        double avg_loss = total_loss / static_cast<double>(X.rows);
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << std::endl;
        }
    }
}

// Extended MLP creator with optional BatchNorm and Dropout per hidden layer.
// sizes: [input_dim, hidden1, hidden2, ..., output_dim]
// batchnorm: if true, adds BatchNorm1D after each hidden Dense (before activation)
// dropout: probability [0,1), 0 means no dropout
Model create_mlp_ex(const std::vector<size_t>& sizes,
                    const std::string& activation,
                    bool batchnorm,
                    double dropout) {
    Model model;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bool is_hidden = (i + 2 < sizes.size());
        model.add_layer(new Dense(sizes[i], sizes[i+1]));
        if (is_hidden) {
            if (batchnorm) {
                model.add_layer(new BatchNorm1D(sizes[i+1]));
            }
            if (dropout > 0.0) {
                model.add_layer(new Dropout(dropout));
            }
            if (activation == "relu") {
                model.add_layer(new Activation<ReLU>(ReLU{}));
            } else if (activation == "sigmoid") {
                model.add_layer(new Activation<Sigmoid>(Sigmoid{}));
            } else if (activation == "tanh") {
                model.add_layer(new Activation<Tanh>(Tanh{}));
            } else if (activation == "leakyrelu") {
                model.add_layer(new Activation<LeakyReLU>(LeakyReLU{}));
            } else if (activation == "gelu") {
                model.add_layer(new Activation<GELU>(GELU{}));
            } else {
                model.add_layer(new Activation<ReLU>(ReLU{}));
            }
        }
    }
    return model;
}

void Model::summary() const {
    size_t total_params = 0;
    std::cout << "Model Summary\n";
    std::cout << "────────────────────────────────────────────\n";
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        std::string name;
        size_t params = 0;
        size_t out_rows = 0, out_cols = 0;

        if (auto* d = dynamic_cast<Dense*>(layer.get())) {
            name = "Dense";
            params = d->weights.rows * d->weights.cols + d->bias.rows * d->bias.cols;
            out_rows = 0; out_cols = d->weights.rows;
        } else if (dynamic_cast<Conv2D*>(layer.get())) {
            name = "Conv2D";
            // rough estimate
            out_rows = 0; out_cols = 0;
        } else if (dynamic_cast<BatchNorm1D*>(layer.get())) {
            name = "BatchNorm1D";
            params = 0; // tracked separately
            out_rows = 0; out_cols = 0;
        } else if (dynamic_cast<LayerNorm*>(layer.get())) {
            name = "LayerNorm";
            params = 0;
            out_rows = 0; out_cols = 0;
        } else if (dynamic_cast<Dropout*>(layer.get())) {
            name = "Dropout";
        } else if (dynamic_cast<Embedding*>(layer.get())) {
            name = "Embedding";
        } else if (dynamic_cast<LSTM*>(layer.get())) {
            name = "LSTM";
        } else if (dynamic_cast<SimpleRNN*>(layer.get())) {
            name = "SimpleRNN";
        } else if (dynamic_cast<SkipConnection*>(layer.get())) {
            name = "SkipConnection";
        } else if (dynamic_cast<MaxPool2D*>(layer.get())) {
            name = "MaxPool2D";
        } else if (dynamic_cast<Flatten*>(layer.get())) {
            name = "Flatten";
        } else if (dynamic_cast<TransformerBlock*>(layer.get())) {
            name = "TransformerBlock";
        } else if (dynamic_cast<MultiHeadAttention*>(layer.get())) {
            name = "MultiHeadAttention";
        } else if (dynamic_cast<PositionalEncoding*>(layer.get())) {
            name = "PositionalEncoding";
        } else {
            name = "Activation/Other";
        }

        if (params > 0) {
            total_params += params;
            std::cout << "  [" << i << "] " << name << "  params=" << params << "\n";
        } else {
            std::cout << "  [" << i << "] " << name << "\n";
        }
    }
    std::cout << "────────────────────────────────────────────\n";
    std::cout << "Total trainable params: " << param_count() << "\n";
}

size_t Model::param_count() const {
    size_t total = 0;
    for (const auto& layer : layers) {
        for (Tensor* p : layer->parameters()) {
            total += p->rows * p->cols;
        }
    }
    return total;
}
