CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native
INCLUDES = -Iinclude

BUILD_DIR = build
DEMOS_DIR = demos

# Auto-discover source files (wildcard only goes 1 level deep with **, so chain)
LIB_SRCS := $(wildcard include/nn/*.cpp) \
	    $(wildcard include/nn/*/*.cpp) \
	    $(wildcard include/nn/*/*/*.cpp)

LIB_OBJS := $(LIB_SRCS:include/nn/%.cpp=$(BUILD_DIR)/%.o)

# All unique dirs needed
ALL_DIRS := $(sort $(dir $(LIB_OBJS)))

.PHONY: all clean setup tests run_tests

all: setup \
	$(BUILD_DIR)/nn_demo \
	$(BUILD_DIR)/demo_big \
	$(BUILD_DIR)/demo_multiclass \
	$(BUILD_DIR)/demo_cnn_xor \
	$(BUILD_DIR)/demo_cnn_multiclass \
	$(BUILD_DIR)/demo_rnn_airline \
	$(BUILD_DIR)/demo_lstm_airline \
	$(BUILD_DIR)/demo_embedding \
	$(BUILD_DIR)/demo_extensions \
	$(BUILD_DIR)/demo_transformer \
	$(BUILD_DIR)/demo_s4 \
	$(BUILD_DIR)/demo_lookahead

setup:
	@mkdir -p $(BUILD_DIR) $(ALL_DIRS)

# Library objects
$(BUILD_DIR)/%.o: include/nn/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Demo single (demo.cpp)
$(BUILD_DIR)/demo.o: $(DEMOS_DIR)/demo.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Demo others (big, multiclass, cnn_xor, etc.)
$(BUILD_DIR)/demo%.o: $(DEMOS_DIR)/demo%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link nn_demo (special case)
$(BUILD_DIR)/nn_demo: $(LIB_OBJS) $(BUILD_DIR)/demo.o
	$(CXX) $^ -o $@

# Link other demos
$(BUILD_DIR)/demo%: $(LIB_OBJS) $(BUILD_DIR)/demo%.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/demo_s4: $(LIB_OBJS) $(BUILD_DIR)/demo_s4.o
	$(CXX) $^ -o $@

# Test single-file compilation rules
$(BUILD_DIR)/test%.o: tests/test%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# =================================================================
# Test targets
# =================================================================
$(BUILD_DIR)/test_s4: $(LIB_OBJS) $(BUILD_DIR)/test_s4.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gradient_check: $(LIB_OBJS) $(BUILD_DIR)/test_gradient_check.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_crate: $(LIB_OBJS) $(BUILD_DIR)/test_crate.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rmsnorm: $(LIB_OBJS) $(BUILD_DIR)/test_rmsnorm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_wgan_gp: $(LIB_OBJS) $(BUILD_DIR)/test_wgan_gp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_flash_attention: $(LIB_OBJS) $(BUILD_DIR)/test_flash_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_vit: $(LIB_OBJS) $(BUILD_DIR)/test_vit.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rope: $(LIB_OBJS) $(BUILD_DIR)/test_rope.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_focal_simple: $(LIB_OBJS) $(BUILD_DIR)/test_focal_simple.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gat_gradient: $(LIB_OBJS) $(BUILD_DIR)/test_gat_gradient.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gat_verify: $(LIB_OBJS) $(BUILD_DIR)/test_gat_verify.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gat_attention: $(LIB_OBJS) $(BUILD_DIR)/test_gat_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_coord_network: $(LIB_OBJS) $(BUILD_DIR)/test_coord_network.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_realnvp: $(LIB_OBJS) $(BUILD_DIR)/test_realnvp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ddpm: $(LIB_OBJS) $(BUILD_DIR)/test_ddpm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adabelief: $(LIB_OBJS) $(BUILD_DIR)/test_adabelief.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_avgpool2d: $(LIB_OBJS) $(BUILD_DIR)/test_avgpool2d.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gin: $(LIB_OBJS) $(BUILD_DIR)/test_gin.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_deep_gcn: $(LIB_OBJS) $(BUILD_DIR)/test_deep_gcn.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lightgcn: $(LIB_OBJS) $(BUILD_DIR)/test_lightgcn.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_patchy_san: $(LIB_OBJS) $(BUILD_DIR)/test_patchy_san.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_pna: $(LIB_OBJS) $(BUILD_DIR)/test_pna.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_nystrom_attention: $(LIB_OBJS) $(BUILD_DIR)/test_nystrom_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_edgeconv: $(LIB_OBJS) $(BUILD_DIR)/test_edgeconv.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dmon: $(LIB_OBJS) $(BUILD_DIR)/test_dmon.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gmlp: $(LIB_OBJS) $(BUILD_DIR)/test_gmlp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_linformer: $(LIB_OBJS) $(BUILD_DIR)/test_linformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba: $(LIB_OBJS) $(BUILD_DIR)/test_mamba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_suite: $(LIB_OBJS) $(BUILD_DIR)/test_suite.o
	$(CXX) $^ -o $@

tests: setup $(BUILD_DIR)/test_s4 $(BUILD_DIR)/test_adabelief $(BUILD_DIR)/test_gradient_check $(BUILD_DIR)/test_rmsnorm $(BUILD_DIR)/test_wgan_gp $(BUILD_DIR)/test_flash_attention $(BUILD_DIR)/test_vit $(BUILD_DIR)/test_gat_gradient $(BUILD_DIR)/test_gat_verify $(BUILD_DIR)/test_gat_attention $(BUILD_DIR)/test_coord_network $(BUILD_DIR)/test_avgpool2d $(BUILD_DIR)/test_gin $(BUILD_DIR)/test_realnvp $(BUILD_DIR)/test_ddpm $(BUILD_DIR)/test_nystrom_attention $(BUILD_DIR)/test_deep_gcn $(BUILD_DIR)/test_lightgcn $(BUILD_DIR)/test_patchy_san $(BUILD_DIR)/test_pna $(BUILD_DIR)/test_edgeconv $(BUILD_DIR)/test_dmon $(BUILD_DIR)/test_gmlp $(BUILD_DIR)/test_linformer $(BUILD_DIR)/test_mamba

run_tests: tests
	@echo "=== Running S4 Tests ===" && ./$(BUILD_DIR)/test_s4
	@echo "=== Running Gradient Checks ===" && ./$(BUILD_DIR)/test_gradient_check
	@echo "=== Running Linformer Tests ===" && ./$(BUILD_DIR)/test_linformer
	@echo "=== Running Mamba Tests ===" && ./$(BUILD_DIR)/test_mamba

clean:
	rm -rf $(BUILD_DIR)