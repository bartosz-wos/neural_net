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

.PHONY: all clean setup

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
	$(BUILD_DIR)/demo_transformer

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

clean:
	rm -rf $(BUILD_DIR)
