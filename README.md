# Neural Net — C++ MLP Framework

A minimal but functional C++ neural network framework with automatic differentiation.

## Layout

```
neural_net/
├── include/nn/
│   ├── nn.h                        # umbrella: #include "nn/nn.h"
│   ├── core/                       # Tensor, Layer, Model
│   ├── activations/               # ReLU, Sigmoid, Tanh, Softmax, GELU, ...
│   ├── optimizers/                # SGD, Adam, RMSprop, AdamW, OneCycleLR, SWA
│   ├── utils/                     # GridSearchCV, tokenizer, gradient check
│   ├── layers/
│   │   ├── attention/            # MultiHeadAttention, TransformerBlock, LayerOutputTracker
│   │   ├── convolutions/        # Conv2D (im2col), Conv1D, DilatedConv2D, CoordConv
│   │   ├── pooling/             # MaxPool2D, AvgPool2D
│   │   ├── recurrent/            # LSTM, SimpleRNN, GRU, BidirectionalLSTM
│   │   ├── dense/               # Embedding, Flatten
│   │   ├── normalization/       # BatchNorm1D, LayerNorm, WeightNorm
│   │   ├── composite/           # SkipConnection, MultiOutputModel, ResNet, Seq2SeqAttention
│   │   ├── generative/          # VAE, CapsuleNet
│   │   └── graph/               # GCN, GAT, GraphNetwork
│   └── utils/
│       ├── ensemble/             # AdaBoost, RandomForest, IsolationForest, GradientBoosting, XGBoostStyle, LightGBMStyle
│       ├── loss/                 # FocalLoss, LabelSmoothing, ElasticNet
│       ├── training/             # ClipGradNorm, OneCycleLR, TimingBenchmark, LayerOutputTracker
│       └── numerical/            # NumericalStability, SerializationRoundtrip
├── demo*.cpp                       # demos
├── Makefile
└── EXPANSION_QUEUE.md
```

## Usage

```cpp
#include "nn/nn.h"

Model model;
model.add_layer(new Dense(2, 8));
model.add_layer(new Activation<ReLU>(ReLU{}));
model.add_layer(new Dense(8, 1));

Adam optimizer(0.01);
model.train(X, y, optimizer, epochs);
```

## Building

```bash
make all       # build all demos
make clean     # clean build/
make <target>  # build specific demo
./build/<demo> # run
```

## Features

**Core**
- `tensor` — matrix ops, elementwise, transpose, reshape, reductions, Hadamard
- `layer` — Dense (Xavier init) + Activation wrapper template
- `model` — sequential container with train/evaluate/save/load

**Activations** — ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Softplus, GELU, Swish

**Layers**
- Conv2D (im2col, padding, strides, dilation), Conv1D, DilatedConv2D, CoordConv
- MaxPool2D, AvgPool2D
- LSTM (full BPTT), SimpleRNN, GRU, BidirectionalLSTM
- Embedding (lookup table with gradient accumulation)
- LayerNorm, BatchNorm1D, WeightNorm, Dropout
- TransformerBlock, MultiHeadAttention, PositionalEncoding
- SkipConnection, ResNet (ResNet18/34/50/101), DenseNet, SqueezeNet, MobileNetV2
- MemN2N (Memory Network), ListenAttendSpell (LAS), MoE (Mixture of Experts)
- GCN, GAT (graph neural networks)
- VAE (Variational Autoencoder), CapsuleNet

**Optimizers** — SGD, Adam, RMSprop, AdamW, SGD+Nesterov, clip_grad_norm_

**Schedulers** — StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, SWA

**Training Tools** — GridSearchCV, DataLoader + Dataset, TimingBenchmark, LayerOutputTracker

**Losses** — FocalLoss, LabelSmoothingCrossEntropy, ElasticNet

**Ensemble** — AdaBoost, RandomForest, IsolationForest, GradientBoosting, XGBoostStyle, LightGBMStyle

## Demos

| Demo | Description |
|------|-------------|
| `nn_demo` | XOR, Boston Housing |
| `xor_big` | Large XOR dataset |
| `multiclass` | 3-class classification |
| `cnn_xor` | CNN on XOR |
| `cnn_multiclass` | CNN multi-class |
| `rnn_airline` | SimpleRNN airline passengers |
| `lstm_airline` | LSTM airline passengers |
| `embedding_demo` | Char-level embedding |
| `extensions_demo` | Norm, Dropout, GridSearch, LR schedulers |
| `transformer_demo` | Transformer, MHA, PositionalEncoding |

## Dependencies

- C++17 compiler (g++) — no external libraries
