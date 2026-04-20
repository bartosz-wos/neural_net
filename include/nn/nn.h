#ifndef NN_H
#define NN_H

// Public umbrella header — all library components
// Include path: -I<root>/include
#include "core/tensor.h"
#include "core/layer.h"
#include "core/model.h"
#include "activations/activations.h"
#include "optimizers/optimizer.h"
#include "optimizers/optimizer_extended.h"
#include "optimizers/scheduler.h"
#include "optimizers/one_cycle_lr.h"
#include "utils/grid_search.h"
#include "utils/tokenizer.h"
#include "utils/gradient_check.h"
#include "utils/numerical_stability.h"
#include "utils/timing_benchmark.h"
#include "utils/serialization_roundtrip.h"
#include "utils/label_smoothing.h"
#include "utils/clip_grad_norm.h"
#include "utils/focal_loss.h"
#include "utils/tabular_ensemble.h"
#include "utils/adaboost.h"
#include "utils/gradient_boosting.h"
#include "utils/elastic_net.h"
#include "utils/random_forest.h"
#include "utils/isolation_forest.h"
#include "utils/lightgbm_style.h"
#include "utils/mixup_cutmix.h"
#include "utils/triplet_loss_siamese.h"

// Layers — convolutional
#include "layers/convolutions/conv_layer.h"
#include "layers/pooling/pool_layer.h"
#include "layers/conv1d.h"
#include "layers/coordconv.h"

// Layers — recurrent
#include "layers/recurrent/lstm.h"
#include "layers/recurrent/rnn.h"
#include "layers/recurrent/gru.h"
#include "layers/recurrent/lstm_bidirectional.h"

// Layers — dense / embeddings
#include "layers/dense/embedding.h"
#include "layers/dense/flatten.h"

// Layers — normalization
#include "layers/normalization/batch_norm.h"
#include "layers/normalization/layer_norm.h"
#include "layers/normalization/weight_norm.h"
#include "layers/normalization/group_norm.h"

// Layers — attention
#include "layers/attention/transformer.h"
#include "layers/attention/layer_output_tracker.h"

// Layers — composite (architecture blocks)
#include "layers/skip_connection.h"
#include "layers/multi_output_model.h"
#include "layers/resnet.h"
#include "layers/densenet.h"
#include "layers/mobilenet_v2.h"
#include "layers/squeezenet.h"
#include "layers/memory_network.h"
#include "layers/squeeze_excitation.h"
#include "layers/seq2seq_attention.h"
#include "layers/mixture_of_experts.h"
#include "layers/lstm_las.h"

// Layers — generative
#include "layers/generative/vae.h"
#include "layers/generative/capsnet.h"

// Layers — graph
#include "layers/gnn.h"
#include "layers/unet.h"
#include "layers/cnn_models_vgg.h"

#endif