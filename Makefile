CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native
INCLUDES = -Iinclude

BUILD_DIR = build

.PHONY: all clean

all: nn_demo xor_big multiclass cnn_xor cnn_multiclass rnn_airline lstm_airline embedding_demo extensions transformer_demo

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/activations $(BUILD_DIR)/optimizers \
	$(BUILD_DIR)/layers/attention $(BUILD_DIR)/layers/recurrent \
	$(BUILD_DIR)/layers/normalization $(BUILD_DIR)/layers/convolutions \
	$(BUILD_DIR)/layers/pooling $(BUILD_DIR)/layers/dense \
	$(BUILD_DIR)/layers/generative $(BUILD_DIR)/layers/graph \
	$(BUILD_DIR)/layers/composite $(BUILD_DIR)/utils

# Core
$(BUILD_DIR)/core/tensor.cpp.o: include/nn/core/tensor.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/core/layer.cpp.o: include/nn/core/layer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/core/model.cpp.o: include/nn/core/model.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Activations
$(BUILD_DIR)/activations/activations.cpp.o: include/nn/activations/activations.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Optimizers
$(BUILD_DIR)/optimizers/optimizer.cpp.o: include/nn/optimizers/optimizer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/optimizers/optimizer_extended.cpp.o: include/nn/optimizers/optimizer_extended.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/optimizers/swa.cpp.o: include/nn/optimizers/swa.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/optimizers/one_cycle_lr.cpp.o: include/nn/optimizers/one_cycle_lr.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Utils
$(BUILD_DIR)/utils/grid_search.cpp.o: include/nn/utils/grid_search.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/tokenizer.cpp.o: include/nn/utils/tokenizer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/numerical_stability.cpp.o: include/nn/utils/numerical_stability.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/timing_benchmark.cpp.o: include/nn/utils/timing_benchmark.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/serialization_roundtrip.cpp.o: include/nn/utils/serialization_roundtrip.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/clip_grad_norm.cpp.o: include/nn/utils/clip_grad_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/focal_loss.cpp.o: include/nn/utils/focal_loss.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/tabular_ensemble.cpp.o: include/nn/utils/tabular_ensemble.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/adaboost.cpp.o: include/nn/utils/adaboost.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/gradient_boosting.cpp.o: include/nn/utils/gradient_boosting.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/elastic_net.cpp.o: include/nn/utils/elastic_net.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/random_forest.cpp.o: include/nn/utils/random_forest.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/isolation_forest.cpp.o: include/nn/utils/isolation_forest.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/lightgbm_style.cpp.o: include/nn/utils/lightgbm_style.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/mixup_cutmix.cpp.o: include/nn/utils/mixup_cutmix.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/utils/triplet_loss_siamese.cpp.o: include/nn/utils/triplet_loss_siamese.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — convolutions
$(BUILD_DIR)/layers/convolutions/conv_layer.cpp.o: include/nn/layers/convolutions/conv_layer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/pooling/pool_layer.cpp.o: include/nn/layers/pooling/pool_layer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/conv1d.cpp.o: include/nn/layers/conv1d.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/coordconv.cpp.o: include/nn/layers/coordconv.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — recurrent
$(BUILD_DIR)/layers/recurrent/lstm.cpp.o: include/nn/layers/recurrent/lstm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/recurrent/rnn.cpp.o: include/nn/layers/recurrent/rnn.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/recurrent/gru.cpp.o: include/nn/layers/recurrent/gru.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/recurrent/lstm_bidirectional.cpp.o: include/nn/layers/recurrent/lstm_bidirectional.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — dense
$(BUILD_DIR)/layers/dense/embedding.cpp.o: include/nn/layers/dense/embedding.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/dense/flatten.cpp.o: include/nn/layers/dense/flatten.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — normalization
$(BUILD_DIR)/layers/normalization/batch_norm.cpp.o: include/nn/layers/normalization/batch_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/normalization/layer_norm.cpp.o: include/nn/layers/normalization/layer_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/normalization/weight_norm.cpp.o: include/nn/layers/normalization/weight_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/normalization/group_norm.cpp.o: include/nn/layers/normalization/group_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/unet.cpp.o: include/nn/layers/unet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/cnn_models_vgg.cpp.o: include/nn/layers/cnn_models_vgg.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — attention
$(BUILD_DIR)/layers/attention/transformer.cpp.o: include/nn/layers/attention/transformer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/attention/layer_output_tracker.cpp.o: include/nn/layers/attention/layer_output_tracker.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — composite
$(BUILD_DIR)/layers/skip_connection.cpp.o: include/nn/layers/skip_connection.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/multi_output_model.cpp.o: include/nn/layers/multi_output_model.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/resnet.cpp.o: include/nn/layers/resnet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/densenet.cpp.o: include/nn/layers/densenet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/mobilenet_v2.cpp.o: include/nn/layers/mobilenet_v2.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/squeezenet.cpp.o: include/nn/layers/squeezenet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/memory_network.cpp.o: include/nn/layers/memory_network.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/squeeze_excitation.cpp.o: include/nn/layers/squeeze_excitation.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/seq2seq_attention.cpp.o: include/nn/layers/seq2seq_attention.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/mixture_of_experts.cpp.o: include/nn/layers/mixture_of_experts.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/lstm_las.cpp.o: include/nn/layers/lstm_las.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — generative
$(BUILD_DIR)/layers/generative/vae.cpp.o: include/nn/layers/generative/vae.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/generative/capsnet.cpp.o: include/nn/layers/generative/capsnet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers — graph
$(BUILD_DIR)/layers/graph/gnn.cpp.o: include/nn/layers/gnn.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

LIB_OBJS = \
	$(BUILD_DIR)/core/tensor.cpp.o \
	$(BUILD_DIR)/core/layer.cpp.o \
	$(BUILD_DIR)/core/model.cpp.o \
	$(BUILD_DIR)/activations/activations.cpp.o \
	$(BUILD_DIR)/optimizers/optimizer.cpp.o \
	$(BUILD_DIR)/optimizers/optimizer_extended.cpp.o \
	$(BUILD_DIR)/optimizers/swa.cpp.o \
	$(BUILD_DIR)/optimizers/one_cycle_lr.cpp.o \
	$(BUILD_DIR)/utils/grid_search.cpp.o \
	$(BUILD_DIR)/utils/tokenizer.cpp.o \
	$(BUILD_DIR)/utils/numerical_stability.cpp.o \
	$(BUILD_DIR)/utils/timing_benchmark.cpp.o \
	$(BUILD_DIR)/utils/serialization_roundtrip.cpp.o \
	$(BUILD_DIR)/utils/clip_grad_norm.cpp.o \
	$(BUILD_DIR)/utils/focal_loss.cpp.o \
	$(BUILD_DIR)/utils/tabular_ensemble.cpp.o \
	$(BUILD_DIR)/utils/adaboost.cpp.o \
	$(BUILD_DIR)/utils/gradient_boosting.cpp.o \
	$(BUILD_DIR)/utils/elastic_net.cpp.o \
	$(BUILD_DIR)/utils/random_forest.cpp.o \
	$(BUILD_DIR)/utils/isolation_forest.cpp.o \
	$(BUILD_DIR)/utils/lightgbm_style.cpp.o \
	$(BUILD_DIR)/utils/mixup_cutmix.cpp.o \
	$(BUILD_DIR)/utils/triplet_loss_siamese.cpp.o \
	$(BUILD_DIR)/layers/convolutions/conv_layer.cpp.o \
	$(BUILD_DIR)/layers/pooling/pool_layer.cpp.o \
	$(BUILD_DIR)/layers/conv1d.cpp.o \
	$(BUILD_DIR)/layers/coordconv.cpp.o \
	$(BUILD_DIR)/layers/recurrent/lstm.cpp.o \
	$(BUILD_DIR)/layers/recurrent/rnn.cpp.o \
	$(BUILD_DIR)/layers/recurrent/gru.cpp.o \
	$(BUILD_DIR)/layers/recurrent/lstm_bidirectional.cpp.o \
	$(BUILD_DIR)/layers/dense/embedding.cpp.o \
	$(BUILD_DIR)/layers/dense/flatten.cpp.o \
	$(BUILD_DIR)/layers/normalization/batch_norm.cpp.o \
	$(BUILD_DIR)/layers/normalization/layer_norm.cpp.o \
	$(BUILD_DIR)/layers/normalization/weight_norm.cpp.o \
	$(BUILD_DIR)/layers/normalization/group_norm.cpp.o \
	$(BUILD_DIR)/layers/attention/transformer.cpp.o \
	$(BUILD_DIR)/layers/attention/layer_output_tracker.cpp.o \
	$(BUILD_DIR)/layers/skip_connection.cpp.o \
	$(BUILD_DIR)/layers/multi_output_model.cpp.o \
	$(BUILD_DIR)/layers/resnet.cpp.o \
	$(BUILD_DIR)/layers/densenet.cpp.o \
	$(BUILD_DIR)/layers/mobilenet_v2.cpp.o \
	$(BUILD_DIR)/layers/squeezenet.cpp.o \
	$(BUILD_DIR)/layers/memory_network.cpp.o \
	$(BUILD_DIR)/layers/squeeze_excitation.cpp.o \
	$(BUILD_DIR)/layers/seq2seq_attention.cpp.o \
	$(BUILD_DIR)/layers/mixture_of_experts.cpp.o \
	$(BUILD_DIR)/layers/lstm_las.cpp.o \
	$(BUILD_DIR)/layers/generative/vae.cpp.o \
	$(BUILD_DIR)/layers/generative/capsnet.cpp.o \
	$(BUILD_DIR)/layers/graph/gnn.cpp.o \
	$(BUILD_DIR)/layers/unet.cpp.o \
	$(BUILD_DIR)/layers/cnn_models_vgg.cpp.o

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

nn_demo: $(LIB_OBJS) $(BUILD_DIR)/demo.o
	$(CXX) $^ -o $(BUILD_DIR)/nn_demo
xor_big: $(LIB_OBJS) $(BUILD_DIR)/demo_big.o
	$(CXX) $^ -o $(BUILD_DIR)/xor_big
multiclass: $(LIB_OBJS) $(BUILD_DIR)/demo_multiclass.o
	$(CXX) $^ -o $(BUILD_DIR)/multiclass
cnn_xor: $(LIB_OBJS) $(BUILD_DIR)/demo_cnn_xor.o
	$(CXX) $^ -o $(BUILD_DIR)/cnn_xor
cnn_multiclass: $(LIB_OBJS) $(BUILD_DIR)/demo_cnn_multiclass.o
	$(CXX) $^ -o $(BUILD_DIR)/cnn_multiclass
rnn_airline: $(LIB_OBJS) $(BUILD_DIR)/demo_rnn_airline.o
	$(CXX) $^ -o $(BUILD_DIR)/rnn_airline
lstm_airline: $(LIB_OBJS) $(BUILD_DIR)/demo_lstm_airline.o
	$(CXX) $^ -o $(BUILD_DIR)/lstm_airline
embedding_demo: $(LIB_OBJS) $(BUILD_DIR)/demo_embedding.o
	$(CXX) $^ -o $(BUILD_DIR)/embedding_demo
extensions: $(LIB_OBJS) $(BUILD_DIR)/demo_extensions.o
	$(CXX) $^ -o $(BUILD_DIR)/extensions_demo
transformer_demo: $(LIB_OBJS) $(BUILD_DIR)/demo_transformer.o
	$(CXX) $^ -o $(BUILD_DIR)/transformer_demo

clean:
	rm -rf $(BUILD_DIR)