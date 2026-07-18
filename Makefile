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

$(BUILD_DIR)/test_adaln_zero: $(LIB_OBJS) $(BUILD_DIR)/test_adaln_zero.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_wgan_gp: $(LIB_OBJS) $(BUILD_DIR)/test_wgan_gp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_flash_attention: $(LIB_OBJS) $(BUILD_DIR)/test_flash_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_flash_attention_v2: $(LIB_OBJS) $(BUILD_DIR)/test_flash_attention_v2.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_vit: $(LIB_OBJS) $(BUILD_DIR)/test_vit.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rope: $(LIB_OBJS) $(BUILD_DIR)/test_rope.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rope_v: $(LIB_OBJS) $(BUILD_DIR)/test_rope_v.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_distribution_losses: $(LIB_OBJS) $(BUILD_DIR)/test_distribution_losses.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_segmentation_losses: $(LIB_OBJS) $(BUILD_DIR)/test_segmentation_losses.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mmd_loss: $(LIB_OBJS) $(BUILD_DIR)/test_mmd_loss.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_contrastive_losses: $(LIB_OBJS) $(BUILD_DIR)/test_contrastive_losses.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_siglip_loss: $(LIB_OBJS) $(BUILD_DIR)/test_siglip_loss.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_activations: $(LIB_OBJS) $(BUILD_DIR)/test_activations.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_legacy_adaptive: $(LIB_OBJS) $(BUILD_DIR)/test_legacy_adaptive.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ademamix: $(LIB_OBJS) $(BUILD_DIR)/test_ademamix.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_muon: $(LIB_OBJS) $(BUILD_DIR)/test_muon.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adafactor: $(LIB_OBJS) $(BUILD_DIR)/test_adafactor.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sgd_nesterov: $(LIB_OBJS) $(BUILD_DIR)/test_sgd_nesterov.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lr_schedulers: $(LIB_OBJS) $(BUILD_DIR)/test_lr_schedulers.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gmm: $(LIB_OBJS) $(BUILD_DIR)/test_gmm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_metrics: $(LIB_OBJS) $(BUILD_DIR)/test_metrics.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_model_ema: $(LIB_OBJS) $(BUILD_DIR)/test_model_ema.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dataloader: $(LIB_OBJS) $(BUILD_DIR)/test_dataloader.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_instance_norm: $(LIB_OBJS) $(BUILD_DIR)/test_instance_norm.o
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

$(BUILD_DIR)/test_neural_spline_flow: $(LIB_OBJS) $(BUILD_DIR)/test_neural_spline_flow.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ddpm: $(LIB_OBJS) $(BUILD_DIR)/test_ddpm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adabelief: $(LIB_OBJS) $(BUILD_DIR)/test_adabelief.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_yogi: $(LIB_OBJS) $(BUILD_DIR)/test_yogi.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_radam: $(LIB_OBJS) $(BUILD_DIR)/test_radam.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lion: $(LIB_OBJS) $(BUILD_DIR)/test_lion.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sophia: $(LIB_OBJS) $(BUILD_DIR)/test_sophia.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sam: $(LIB_OBJS) $(BUILD_DIR)/test_sam.o
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

$(BUILD_DIR)/test_xlstm: $(LIB_OBJS) $(BUILD_DIR)/test_xlstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba2: $(LIB_OBJS) $(BUILD_DIR)/test_mamba2.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_retnet: $(LIB_OBJS) $(BUILD_DIR)/test_retnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mlstm: $(LIB_OBJS) $(BUILD_DIR)/test_mlstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_performer: $(LIB_OBJS) $(BUILD_DIR)/test_performer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gqa: $(LIB_OBJS) $(BUILD_DIR)/test_gqa.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_aft: $(LIB_OBJS) $(BUILD_DIR)/test_aft.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_aft_local: $(LIB_OBJS) $(BUILD_DIR)/test_aft_local.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_aft_conv: $(LIB_OBJS) $(BUILD_DIR)/test_aft_conv.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lsh_attention: $(LIB_OBJS) $(BUILD_DIR)/test_lsh_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_conv_attention: $(LIB_OBJS) $(BUILD_DIR)/test_conv_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_conv_bert: $(LIB_OBJS) $(BUILD_DIR)/test_conv_bert.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_pixelcnn: $(LIB_OBJS) $(BUILD_DIR)/test_pixelcnn.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_egnn: $(LIB_OBJS) $(BUILD_DIR)/test_egnn.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sparse_moe: $(LIB_OBJS) $(BUILD_DIR)/test_sparse_moe.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_knn_classifier: $(LIB_OBJS) $(BUILD_DIR)/test_knn_classifier.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_tree_lstm: $(LIB_OBJS) $(BUILD_DIR)/test_tree_lstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hopfield: $(LIB_OBJS) $(BUILD_DIR)/test_hopfield.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_slot_attention: $(LIB_OBJS) $(BUILD_DIR)/test_slot_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_fnet: $(LIB_OBJS) $(BUILD_DIR)/test_fnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_magnitude_pruning: $(LIB_OBJS) $(BUILD_DIR)/test_magnitude_pruning.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_stochastic_depth: $(LIB_OBJS) $(BUILD_DIR)/test_stochastic_depth.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ff_layer: $(LIB_OBJS) $(BUILD_DIR)/test_ff_layer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_span_extractor: $(LIB_OBJS) $(BUILD_DIR)/test_span_extractor.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mla: $(LIB_OBJS) $(BUILD_DIR)/test_mla.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mixture_of_depths: $(LIB_OBJS) $(BUILD_DIR)/test_mixture_of_depths.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_graphsage: $(LIB_OBJS) $(BUILD_DIR)/test_graphsage.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_capsule: $(LIB_OBJS) $(BUILD_DIR)/test_capsule.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_kan: $(LIB_OBJS) $(BUILD_DIR)/test_kan.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gumbel_softmax: $(LIB_OBJS) $(BUILD_DIR)/test_gumbel_softmax.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_bigbird: $(LIB_OBJS) $(BUILD_DIR)/test_bigbird.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_tabnet: $(LIB_OBJS) $(BUILD_DIR)/test_tabnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_alibi: $(LIB_OBJS) $(BUILD_DIR)/test_alibi.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_spatial_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_spatial_transformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_transformer_decoder: $(LIB_OBJS) $(BUILD_DIR)/test_transformer_decoder.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hypernetwork: $(LIB_OBJS) $(BUILD_DIR)/test_hypernetwork.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lookahead: $(LIB_OBJS) $(BUILD_DIR)/test_lookahead.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_suite: $(LIB_OBJS) $(BUILD_DIR)/test_suite.o
	$(CXX) $^ -o $@

tests: setup $(BUILD_DIR)/test_realnvp $(BUILD_DIR)/test_neural_spline_flow $(BUILD_DIR)/test_ddpm $(BUILD_DIR)/test_adabelief \
$(BUILD_DIR)/test_lion $(BUILD_DIR)/test_sophia $(BUILD_DIR)/test_sam \
$(BUILD_DIR)/test_gradient_check \
$(BUILD_DIR)/test_rmsnorm $(BUILD_DIR)/test_wgan_gp $(BUILD_DIR)/test_flash_attention $(BUILD_DIR)/test_flash_attention_v2 \
$(BUILD_DIR)/test_vit $(BUILD_DIR)/test_distribution_losses $(BUILD_DIR)/test_mmd_loss $(BUILD_DIR)/test_contrastive_losses \
$(BUILD_DIR)/test_siglip_loss $(BUILD_DIR)/test_metrics $(BUILD_DIR)/test_model_ema $(BUILD_DIR)/test_dataloader \
$(BUILD_DIR)/test_activations $(BUILD_DIR)/test_legacy_adaptive $(BUILD_DIR)/test_gat_gradient $(BUILD_DIR)/test_gat_verify \
$(BUILD_DIR)/test_gat_attention $(BUILD_DIR)/test_coord_network $(BUILD_DIR)/test_avgpool2d $(BUILD_DIR)/test_gin \
$(BUILD_DIR)/test_ddpm $(BUILD_DIR)/test_nystrom_attention $(BUILD_DIR)/test_deep_gcn $(BUILD_DIR)/test_lightgcn \
$(BUILD_DIR)/test_patchy_san $(BUILD_DIR)/test_pna $(BUILD_DIR)/test_edgeconv $(BUILD_DIR)/test_dmon \
$(BUILD_DIR)/test_gmlp $(BUILD_DIR)/test_linformer $(BUILD_DIR)/test_mamba $(BUILD_DIR)/test_xlstm \
$(BUILD_DIR)/test_mamba2 $(BUILD_DIR)/test_rwkv $(BUILD_DIR)/test_retnet $(BUILD_DIR)/test_mlstm \
$(BUILD_DIR)/test_performer $(BUILD_DIR)/test_gqa $(BUILD_DIR)/test_aft $(BUILD_DIR)/test_aft_local \
$(BUILD_DIR)/test_aft_conv $(BUILD_DIR)/test_lsh_attention $(BUILD_DIR)/test_conv_attention $(BUILD_DIR)/test_conv_bert \
$(BUILD_DIR)/test_pixelcnn $(BUILD_DIR)/test_egnn $(BUILD_DIR)/test_sparse_moe $(BUILD_DIR)/test_knn_classifier \
$(BUILD_DIR)/test_tree_lstm $(BUILD_DIR)/test_hopfield $(BUILD_DIR)/test_rope_v $(BUILD_DIR)/test_slot_attention \
$(BUILD_DIR)/test_fnet $(BUILD_DIR)/test_magnitude_pruning $(BUILD_DIR)/test_stochastic_depth $(BUILD_DIR)/test_ff_layer \
$(BUILD_DIR)/test_span_extractor $(BUILD_DIR)/test_mla $(BUILD_DIR)/test_mixture_of_depths $(BUILD_DIR)/test_graphsage \
$(BUILD_DIR)/test_capsule $(BUILD_DIR)/test_kan $(BUILD_DIR)/test_gumbel_softmax $(BUILD_DIR)/test_bigbird \
$(BUILD_DIR)/test_tabnet $(BUILD_DIR)/test_alibi $(BUILD_DIR)/test_spatial_transformer $(BUILD_DIR)/test_transformer_decoder \
$(BUILD_DIR)/test_hypernetwork $(BUILD_DIR)/test_instance_norm $(BUILD_DIR)/test_ademamix $(BUILD_DIR)/test_sgd_nesterov \
$(BUILD_DIR)/test_lr_schedulers $(BUILD_DIR)/test_muon \
$(BUILD_DIR)/test_gmm $(BUILD_DIR)/test_adaln_zero $(BUILD_DIR)/test_adafactor $(BUILD_DIR)/test_segmentation_losses \
$(BUILD_DIR)/test_yogi $(BUILD_DIR)/test_radam

run_tests: tests
	@echo "=== Running FlashAttention-2 Tests ===" && ./$(BUILD_DIR)/test_flash_attention_v2
	@echo "=== Running S4 Tests ===" && ./$(BUILD_DIR)/test_s4
	@echo "=== Running Neural Spline Flow Tests ===" && ./$(BUILD_DIR)/test_neural_spline_flow
	@echo "=== Running Gradient Checks ===" && ./$(BUILD_DIR)/test_gradient_check
	@echo "=== Running Linformer Tests ===" && ./$(BUILD_DIR)/test_linformer
	@echo "=== Running Mamba Tests ===" && ./$(BUILD_DIR)/test_mamba
	@echo "=== Running xLSTM Tests ===" && ./$(BUILD_DIR)/test_xlstm
	@echo "=== Running Mamba-2 Tests ===" && ./$(BUILD_DIR)/test_mamba2
	@echo "=== Running RWKV Tests ===" && ./$(BUILD_DIR)/test_rwkv
	@echo "=== Running RetNet Tests ===" && ./$(BUILD_DIR)/test_retnet
	@echo "=== Running mLSTM Tests ===" && ./$(BUILD_DIR)/test_mlstm
	@echo "=== Running Performer Tests ===" && ./$(BUILD_DIR)/test_performer
	@echo "=== Running GQA Tests ===" && ./$(BUILD_DIR)/test_gqa
	@echo "=== Running AFT Tests ===" && ./$(BUILD_DIR)/test_aft
	@echo "=== Running AFT-Local Tests ===" && ./$(BUILD_DIR)/test_aft_local
	@echo "=== Running AFT-Conv Tests ===" && ./$(BUILD_DIR)/test_aft_conv
	@echo "=== Running Conv Attention Tests ===" && ./$(BUILD_DIR)/test_conv_attention
	@echo "=== Running ConvBERT Tests ===" && ./$(BUILD_DIR)/test_conv_bert
	@echo "=== Running PixelCNN Tests ===" && ./$(BUILD_DIR)/test_pixelcnn
	@echo "=== Running EGNN Tests ===" && ./$(BUILD_DIR)/test_egnn
	@echo "=== Running Sparse MoE Tests ===" && ./$(BUILD_DIR)/test_sparse_moe
	@echo "=== Running KNN Classifier Tests ===" && ./$(BUILD_DIR)/test_knn_classifier
	@echo "=== Running Tree-LSTM Tests ===" && ./$(BUILD_DIR)/test_tree_lstm
	@echo "=== Running Modern Hopfield Tests ===" && ./$(BUILD_DIR)/test_hopfield
	@echo "=== Running RoPEWithV Tests ===" && ./$(BUILD_DIR)/test_rope_v
	@echo "=== Running Slot Attention Tests ===" && ./$(BUILD_DIR)/test_slot_attention
	@echo "=== Running FNet Tests ===" && ./$(BUILD_DIR)/test_fnet
	@echo "=== Running Magnitude Pruning Tests ===" && ./$(BUILD_DIR)/test_magnitude_pruning
	@echo "=== Running Stochastic Depth Tests ===" && ./$(BUILD_DIR)/test_stochastic_depth
	@echo "=== Running FF Layer Tests ===" && ./$(BUILD_DIR)/test_ff_layer
	@echo "=== Running Span Extractor Tests ===" && ./$(BUILD_DIR)/test_span_extractor
	@echo "=== Running MLA Tests ===" && ./$(BUILD_DIR)/test_mla
	@echo "=== Running Mixture-of-Depths Tests ===" && ./$(BUILD_DIR)/test_mixture_of_depths
	@echo "=== Running GraphSAGE Tests ===" && ./$(BUILD_DIR)/test_graphsage
	@echo "=== Running CapsuleLayer Tests ===" && ./$(BUILD_DIR)/test_capsule
	@echo "=== Running KAN Tests ===" && ./$(BUILD_DIR)/test_kan
	@echo "=== Running Gumbel-Softmax Tests ===" && ./$(BUILD_DIR)/test_gumbel_softmax
	@echo "=== Running Lion Tests ===" && ./$(BUILD_DIR)/test_lion
	@echo "=== Running Sophia Tests ===" && ./$(BUILD_DIR)/test_sophia
	@echo "=== Running SAM Tests ===" && ./$(BUILD_DIR)/test_sam
	@echo "=== Running BigBird Tests ===" && ./$(BUILD_DIR)/test_bigbird
	@echo "=== Running TabNet Tests ===" && ./$(BUILD_DIR)/test_tabnet
	@echo "=== Running Lookahead Tests ===" && ./$(BUILD_DIR)/test_lookahead
	@echo "=== Running ALiBi Tests ===" && ./$(BUILD_DIR)/test_alibi
	@echo "=== Running Spatial Transformer Tests ===" && ./$(BUILD_DIR)/test_spatial_transformer
	@echo "=== Running Transformer Decoder Tests ===" && ./$(BUILD_DIR)/test_transformer_decoder
	@echo "=== Running HyperNetwork Tests ===" && ./$(BUILD_DIR)/test_hypernetwork
	@echo "=== Running Distribution Losses Tests ===" && ./$(BUILD_DIR)/test_distribution_losses
	@echo "=== Running MMD Loss Tests ===" && ./$(BUILD_DIR)/test_mmd_loss
	@echo "=== Running Contrastive Losses Tests ===" && ./$(BUILD_DIR)/test_contrastive_losses
	@echo "=== Running SigLIP Loss Tests ===" && ./$(BUILD_DIR)/test_siglip_loss
	@echo "=== Running Metrics Tests ===" && ./$(BUILD_DIR)/test_metrics
	@echo "=== Running Model EMA Tests ===" && ./$(BUILD_DIR)/test_model_ema
	@echo "=== Running DataLoader Tests ===" && ./$(BUILD_DIR)/test_dataloader
	@echo "=== Running Activations Tests ===" && ./$(BUILD_DIR)/test_activations
	@echo "=== Running Legacy Adaptive Optimizers Tests ===" && ./$(BUILD_DIR)/test_legacy_adaptive
	@echo "=== Running InstanceNorm Tests ===" && ./$(BUILD_DIR)/test_instance_norm
	@echo "=== Running AdEMAMix Tests ===" && ./$(BUILD_DIR)/test_ademamix
	@echo "=== Running SGD-Nesterov Tests ===" && ./$(BUILD_DIR)/test_sgd_nesterov
	@echo "=== Running LR Schedulers Tests ===" && ./$(BUILD_DIR)/test_lr_schedulers
	@echo "=== Running Muon Tests ===" && ./$(BUILD_DIR)/test_muon
	@echo "=== Running GMM Tests ===" && ./$(BUILD_DIR)/test_gmm
	@echo "=== Running AdaLN-Zero Tests ===" && ./$(BUILD_DIR)/test_adaln_zero
	@echo "=== Running Adafactor Tests ===" && ./$(BUILD_DIR)/test_adafactor
	@echo "=== Running Segmentation Losses Tests ===" && ./$(BUILD_DIR)/test_segmentation_losses
	@echo "=== Running Yogi Tests ===" && ./$(BUILD_DIR)/test_yogi
	@echo "=== Running RAdam Tests ===" && ./$(BUILD_DIR)/test_radam

clean:
	rm -rf $(BUILD_DIR)