# Neural Net — C++ MLP Framework

A minimal but functional C++ neural network framework with automatic differentiation.

## Layout

```
neural_net/
├── include/nn/
│   ├── nn.h                        # umbrella: #include "nn/nn.h"
│   ├── tensor.h / tensor.cpp       # matrix ops, elementwise, transpose, reshape
│   ├── layer.h  / layer.cpp        # Dense, Activation wrapper template
│   ├── model.h  / model.cpp        # sequential model container
│   ├── activations/                # all activation functions
│   │   ├── activations.h
│   │   └── activations.cpp
│   ├── optimizers/                 # optimizers + schedulers
│   │   ├── optimizer.h/.cpp
│   │   ├── optimizer_extended.h/.cpp
│   │   ├── scheduler.h
│   │   └── optimizer_sgd_adam.h
│   ├── layers/                     # all layer types
│   │   ├── conv_layer.h/.cpp      # Conv2D (im2col, dilation)
│   │   ├── conv1d.h               # Conv1D
│   │   ├── dilated_conv2d.h       # DilatedConv2D subclass
│   │   ├── pool_layer.h/.cpp      # MaxPool2D, AvgPool2D
│   │   ├── lstm.h/.cpp            # LSTM
│   │   ├── rnn.h/.cpp             # SimpleRNN
│   │   ├── embedding.h/.cpp       # word embeddings
│   │   ├── layer_norm.h/.cpp      # LayerNorm + Dropout
│   │   ├── batch_norm.h/.cpp       # BatchNorm1D
│   │   ├── flatten.h/.cpp         # Flatten
│   │   ├── transformer.h/.cpp      # MHA, TransformerBlock, PositionalEncoding
│   │   └── dataloader.h           # Dataset, TensorDataset, DataLoader
│   └── utils/
│       ├── grid_search.h/.cpp     # GridSearchCV
│       └── tokenizer.h/.cpp       # char-level tokenizer
├── demo*.cpp                        # demos
├── Makefile
├── README.md
└── EXPANSION_QUEUE.md               # pending features
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
- `layer` — Dense (Xavier init) + Activation wrapper template; `init_weights(scheme)`
- `model` — sequential container with train/evaluate/save/load

**Activations** — ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Softplus, GELU, Swish

**Layers**
- Conv2D (im2col, padding, strides, dilation), Conv1D, DilatedConv2D
- MaxPool2D, AvgPool2D
- LSTM (full BPTT), SimpleRNN
- Embedding (lookup table with gradient accumulation)
- LayerNorm, BatchNorm1D, Dropout
- TransformerBlock, MultiHeadAttention, PositionalEncoding

**Optimizers** — SGD, Adam, RMSprop, AdamW, SGD+Nesterov, `clip_grad_norm_()`

**Schedulers** — StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR

**Training Tools** — GridSearchCV, DataLoader + Dataset, create_mlp_ex()

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
