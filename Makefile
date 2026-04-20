CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native
INCLUDES = -Iinclude

BUILD_DIR = build

.PHONY: all clean

all: nn_demo xor_big multiclass cnn_xor cnn_multiclass rnn_airline lstm_airline embedding_demo extensions transformer_demo

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/activations $(BUILD_DIR)/optimizers $(BUILD_DIR)/layers $(BUILD_DIR)/utils

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
\n$(BUILD_DIR)/utils/focal_loss.cpp.o: include/nn/utils/focal_loss.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Layers
$(BUILD_DIR)/layers/conv_layer.cpp.o: include/nn/layers/conv_layer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/pool_layer.cpp.o: include/nn/layers/pool_layer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/lstm.cpp.o: include/nn/layers/lstm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/rnn.cpp.o: include/nn/layers/rnn.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/embedding.cpp.o: include/nn/layers/embedding.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/layer_norm.cpp.o: include/nn/layers/layer_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/batch_norm.cpp.o: include/nn/layers/batch_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/flatten.cpp.o: include/nn/layers/flatten.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/transformer.cpp.o: include/nn/layers/transformer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/conv1d.cpp.o: include/nn/layers/conv1d.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/skip_connection.cpp.o: include/nn/layers/skip_connection.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/multi_output_model.cpp.o: include/nn/layers/multi_output_model.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/gru.cpp.o: include/nn/layers/gru.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/weight_norm.cpp.o: include/nn/layers/weight_norm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/spatial_dropout.cpp.o: include/nn/layers/spatial_dropout.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/weight_init.cpp.o: include/nn/layers/weight_init.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/lstm_bidirectional.cpp.o: include/nn/layers/lstm_bidirectional.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/resnet.cpp.o: include/nn/layers/resnet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/seq2seq_attention.cpp.o: include/nn/layers/seq2seq_attention.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/vae.cpp.o: include/nn/layers/vae.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/cnn_models.cpp.o: include/nn/layers/cnn_models.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/memory_network.cpp.o: include/nn/layers/memory_network.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/squeezenet.cpp.o: include/nn/layers/squeezenet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/layer_output_tracker.cpp.o: include/nn/layers/layer_output_tracker.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/densenet.cpp.o: include/nn/layers/densenet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/mobilenet_v2.cpp.o: include/nn/layers/mobilenet_v2.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/layers/lstm_las.cpp.o: include/nn/layers/lstm_las.cpp | $(BUILD_DIR)
\n$(BUILD_DIR)/layers/capsnet.cpp.o: include/nn/layers/capsnet.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
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
	$(BUILD_DIR)/layers/conv_layer.cpp.o \
	$(BUILD_DIR)/layers/pool_layer.cpp.o \
	$(BUILD_DIR)/layers/lstm.cpp.o \
	$(BUILD_DIR)/layers/rnn.cpp.o \
	$(BUILD_DIR)/layers/embedding.cpp.o \
	$(BUILD_DIR)/layers/layer_norm.cpp.o \
	$(BUILD_DIR)/layers/batch_norm.cpp.o \
	$(BUILD_DIR)/layers/flatten.cpp.o \
	$(BUILD_DIR)/layers/transformer.cpp.o \
	$(BUILD_DIR)/layers/conv1d.cpp.o \
	$(BUILD_DIR)/layers/skip_connection.cpp.o \
	$(BUILD_DIR)/layers/multi_output_model.cpp.o \
	$(BUILD_DIR)/layers/gru.cpp.o \
	$(BUILD_DIR)/layers/weight_norm.cpp.o \
	$(BUILD_DIR)/layers/spatial_dropout.cpp.o \
	$(BUILD_DIR)/layers/weight_init.cpp.o \
	$(BUILD_DIR)/layers/lstm_bidirectional.cpp.o \
	$(BUILD_DIR)/layers/resnet.cpp.o \
	$(BUILD_DIR)/layers/seq2seq_attention.cpp.o \
	$(BUILD_DIR)/layers/vae.cpp.o \
	$(BUILD_DIR)/layers/cnn_models.cpp.o \
	$(BUILD_DIR)/layers/memory_network.cpp.o \
	$(BUILD_DIR)/layers/squeezenet.cpp.o \
	$(BUILD_DIR)/layers/layer_output_tracker.cpp.o \
	$(BUILD_DIR)/layers/densenet.cpp.o \
	$(BUILD_DIR)/layers/mobilenet_v2.cpp.o \
	$(BUILD_DIR)/layers/lstm_las.cpp.o

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