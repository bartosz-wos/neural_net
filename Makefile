CXX = g++
# -MMD -MP: emit a .d sidecar per translation unit recording every header it
# included, and add phony targets for headers that get deleted. Without this the
# build has NO header dependency tracking: changing a class layout in a header
# (adding a member to SwiGLU / MegalodonBlock) recompiles only the .cpp that
# owns it, every other translation unit keeps the OLD layout, and linking the
# two produces an ODR violation that surfaces far from the cause (double free
# in a destructor, or silently stale accessor values). See the -include below.
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -march=native -MMD -MP
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

# Header dependency sidecars emitted by -MMD (see CXXFLAGS). These are
# included as makefiles so that editing a header rebuilds every .cpp that
# includes it. Without this line make never sees the .d files and header edits
# silently produce stale objects.
DEPS := $(LIB_OBJS:.o=.d) \
        $(patsubst $(DEMOS_DIR)/%.cpp,$(BUILD_DIR)/%.d,$(wildcard $(DEMOS_DIR)/*.cpp)) \
        $(patsubst tests/%.cpp,$(BUILD_DIR)/test_%.d,$(wildcard tests/*.cpp))
-include $(DEPS)

# make treats objects built only to satisfy an implicit rule (the demos, which
# have no explicit $(BUILD_DIR)/demo_x target) as INTERMEDIATE files and deletes
# them after linking, which defeats incremental builds — every `make all` would
# relink every demo from scratch. .SECONDARY keeps them.
.SECONDARY:

.PHONY: all clean setup tests run_tests nuke-deps

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

$(BUILD_DIR)/test_spectral_norm: $(LIB_OBJS) $(BUILD_DIR)/test_spectral_norm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dynamic_tanh: $(LIB_OBJS) $(BUILD_DIR)/test_dynamic_tanh.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_layer_scale: $(LIB_OBJS) $(BUILD_DIR)/test_layer_scale.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_qk_norm: $(LIB_OBJS) $(BUILD_DIR)/test_qk_norm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_cheby_kan: $(LIB_OBJS) $(BUILD_DIR)/test_cheby_kan.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_convnext: $(LIB_OBJS) $(BUILD_DIR)/test_convnext.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_timestep_norm: $(LIB_OBJS) $(BUILD_DIR)/test_timestep_norm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_timestep_norm_grad: $(LIB_OBJS) $(BUILD_DIR)/test_timestep_norm_grad.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_complex_ema: $(LIB_OBJS) $(BUILD_DIR)/test_complex_ema.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_complex_ema_grad: $(LIB_OBJS) $(BUILD_DIR)/test_complex_ema_grad.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_megalodon: $(LIB_OBJS) $(BUILD_DIR)/test_megalodon.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_swiglu: $(LIB_OBJS) $(BUILD_DIR)/test_swiglu.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mha_backward: $(LIB_OBJS) $(BUILD_DIR)/test_mha_backward.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_megalodon_grad: $(LIB_OBJS) $(BUILD_DIR)/test_megalodon_grad.o
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

$(BUILD_DIR)/test_yarn_rope: $(LIB_OBJS) $(BUILD_DIR)/test_yarn_rope.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_distribution_losses: $(LIB_OBJS) $(BUILD_DIR)/test_distribution_losses.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_self_supervised_losses: $(LIB_OBJS) $(BUILD_DIR)/test_self_supervised_losses.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_label_smoothing: $(LIB_OBJS) $(BUILD_DIR)/test_label_smoothing.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_clip_grad_norm: $(LIB_OBJS) $(BUILD_DIR)/test_clip_grad_norm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mixup_cutmix: $(LIB_OBJS) $(BUILD_DIR)/test_mixup_cutmix.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_elastic_net: $(LIB_OBJS) $(BUILD_DIR)/test_elastic_net.o
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

$(BUILD_DIR)/test_early_stopping: $(LIB_OBJS) $(BUILD_DIR)/test_early_stopping.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_training_history: $(LIB_OBJS) $(BUILD_DIR)/test_training_history.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_model_checkpoint: $(LIB_OBJS) $(BUILD_DIR)/test_model_checkpoint.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dataloader: $(LIB_OBJS) $(BUILD_DIR)/test_dataloader.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_cross_validation: $(LIB_OBJS) $(BUILD_DIR)/test_cross_validation.o
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

$(BUILD_DIR)/test_diffgrad: $(LIB_OBJS) $(BUILD_DIR)/test_diffgrad.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_yogi: $(LIB_OBJS) $(BUILD_DIR)/test_yogi.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_radam: $(LIB_OBJS) $(BUILD_DIR)/test_radam.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adan: $(LIB_OBJS) $(BUILD_DIR)/test_adan.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mars: $(LIB_OBJS) $(BUILD_DIR)/test_mars.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lars: $(LIB_OBJS) $(BUILD_DIR)/test_lars.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_novograd: $(LIB_OBJS) $(BUILD_DIR)/test_novograd.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_came: $(LIB_OBJS) $(BUILD_DIR)/test_came.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_schedule_free_adamw: $(LIB_OBJS) $(BUILD_DIR)/test_schedule_free_adamw.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_schedule_free_sgd: $(LIB_OBJS) $(BUILD_DIR)/test_schedule_free_sgd.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_signum: $(LIB_OBJS) $(BUILD_DIR)/test_signum.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adam_mini: $(LIB_OBJS) $(BUILD_DIR)/test_adam_mini.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dadaptation: $(LIB_OBJS) $(BUILD_DIR)/test_dadaptation.o
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

$(BUILD_DIR)/test_mlp_mixer: $(LIB_OBJS) $(BUILD_DIR)/test_mlp_mixer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hyena: $(LIB_OBJS) $(BUILD_DIR)/test_hyena.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_striped_hyena: $(LIB_OBJS) $(BUILD_DIR)/test_striped_hyena.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hyena_dna: $(LIB_OBJS) $(BUILD_DIR)/test_hyena_dna.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_conformer: $(LIB_OBJS) $(BUILD_DIR)/test_conformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_linformer: $(LIB_OBJS) $(BUILD_DIR)/test_linformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba: $(LIB_OBJS) $(BUILD_DIR)/test_mamba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba_conv: $(LIB_OBJS) $(BUILD_DIR)/test_mamba_conv.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_xlstm: $(LIB_OBJS) $(BUILD_DIR)/test_xlstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_min_gru: $(LIB_OBJS) $(BUILD_DIR)/test_min_gru.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_min_lstm: $(LIB_OBJS) $(BUILD_DIR)/test_min_lstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mogrifier_lstm: $(LIB_OBJS) $(BUILD_DIR)/test_mogrifier_lstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba2: $(LIB_OBJS) $(BUILD_DIR)/test_mamba2.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_retnet: $(LIB_OBJS) $(BUILD_DIR)/test_retnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mlstm: $(LIB_OBJS) $(BUILD_DIR)/test_mlstm.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_h3: $(LIB_OBJS) $(BUILD_DIR)/test_h3.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_deltanet: $(LIB_OBJS) $(BUILD_DIR)/test_deltanet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gated_deltanet: $(LIB_OBJS) $(BUILD_DIR)/test_gated_deltanet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_s5: $(LIB_OBJS) $(BUILD_DIR)/test_s5.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_act: $(LIB_OBJS) $(BUILD_DIR)/test_act.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_switch_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_switch_transformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_vision_mamba: $(LIB_OBJS) $(BUILD_DIR)/test_vision_mamba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_axial_attention: $(LIB_OBJS) $(BUILD_DIR)/test_axial_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_three_d_axial_attention: $(LIB_OBJS) $(BUILD_DIR)/test_three_d_axial_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sparse_mixer: $(LIB_OBJS) $(BUILD_DIR)/test_sparse_mixer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mixture_of_softmaxes: $(LIB_OBJS) $(BUILD_DIR)/test_mixture_of_softmaxes.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_focal_modulation: $(LIB_OBJS) $(BUILD_DIR)/test_focal_modulation.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba3: $(LIB_OBJS) $(BUILD_DIR)/test_mamba3.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_jamba: $(LIB_OBJS) $(BUILD_DIR)/test_jamba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_griffin: $(LIB_OBJS) $(BUILD_DIR)/test_griffin.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_xlstm_block: $(LIB_OBJS) $(BUILD_DIR)/test_xlstm_block.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_deepseek_moe: $(LIB_OBJS) $(BUILD_DIR)/test_deepseek_moe.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hyper_mixing: $(LIB_OBJS) $(BUILD_DIR)/test_hyper_mixing.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mega: $(LIB_OBJS) $(BUILD_DIR)/test_mega.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mambabyte: $(LIB_OBJS) $(BUILD_DIR)/test_mambabyte.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gla: $(LIB_OBJS) $(BUILD_DIR)/test_gla.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ltc: $(LIB_OBJS) $(BUILD_DIR)/test_ltc.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_neural_ode: $(LIB_OBJS) $(BUILD_DIR)/test_neural_ode.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_flow_matching: $(LIB_OBJS) $(BUILD_DIR)/test_flow_matching.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dit: $(LIB_OBJS) $(BUILD_DIR)/test_dit.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_timesnet: $(LIB_OBJS) $(BUILD_DIR)/test_timesnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hawk: $(LIB_OBJS) $(BUILD_DIR)/test_hawk.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hgrn: $(LIB_OBJS) $(BUILD_DIR)/test_hgrn.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv7: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv7.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv5: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv5.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv6: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv6.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_performer: $(LIB_OBJS) $(BUILD_DIR)/test_performer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_based: $(LIB_OBJS) $(BUILD_DIR)/test_based.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ft_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_ft_transformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_cosformer: $(LIB_OBJS) $(BUILD_DIR)/test_cosformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gqa: $(LIB_OBJS) $(BUILD_DIR)/test_gqa.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_expire_span: $(LIB_OBJS) $(BUILD_DIR)/test_expire_span.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_aft: $(LIB_OBJS) $(BUILD_DIR)/test_aft.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gau: $(LIB_OBJS) $(BUILD_DIR)/test_gau.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gatv2: $(LIB_OBJS) $(BUILD_DIR)/test_gatv2.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sagpool: $(LIB_OBJS) $(BUILD_DIR)/test_sagpool.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_tome: $(LIB_OBJS) $(BUILD_DIR)/test_tome.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_clustered_attention: $(LIB_OBJS) $(BUILD_DIR)/test_clustered_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_window_attention: $(LIB_OBJS) $(BUILD_DIR)/test_window_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_linear_attention: $(LIB_OBJS) $(BUILD_DIR)/test_linear_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lambda_layer: $(LIB_OBJS) $(BUILD_DIR)/test_lambda_layer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_stick_breaking: $(LIB_OBJS) $(BUILD_DIR)/test_stick_breaking.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ngpt: $(LIB_OBJS) $(BUILD_DIR)/test_ngpt.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_multi_token_prediction: $(LIB_OBJS) $(BUILD_DIR)/test_multi_token_prediction.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_tokenformer: $(LIB_OBJS) $(BUILD_DIR)/test_tokenformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_deformable_attention: $(LIB_OBJS) $(BUILD_DIR)/test_deformable_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_multi_scale_deformable_attention: $(LIB_OBJS) $(BUILD_DIR)/test_multi_scale_deformable_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_power_attention: $(LIB_OBJS) $(BUILD_DIR)/test_power_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gated_slot_attention: $(LIB_OBJS) $(BUILD_DIR)/test_gated_slot_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hilo_attention: $(LIB_OBJS) $(BUILD_DIR)/test_hilo_attention.o
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

$(BUILD_DIR)/test_moe_router: $(LIB_OBJS) $(BUILD_DIR)/test_moe_router.o
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

$(BUILD_DIR)/test_lora: $(LIB_OBJS) $(BUILD_DIR)/test_lora.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_bitnet: $(LIB_OBJS) $(BUILD_DIR)/test_bitnet.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_dcn_v2: $(LIB_OBJS) $(BUILD_DIR)/test_dcn_v2.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_monarch_mixer: $(LIB_OBJS) $(BUILD_DIR)/test_monarch_mixer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_set_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_set_transformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_fastformer: $(LIB_OBJS) $(BUILD_DIR)/test_fastformer.o
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

$(BUILD_DIR)/test_fourier_kan: $(LIB_OBJS) $(BUILD_DIR)/test_fourier_kan.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lightgbm_style: $(LIB_OBJS) $(BUILD_DIR)/test_lightgbm_style.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_gradient_centralization: $(LIB_OBJS) $(BUILD_DIR)/test_gradient_centralization.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adamp: $(LIB_OBJS) $(BUILD_DIR)/test_adamp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_lamb: $(LIB_OBJS) $(BUILD_DIR)/test_lamb.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_cautious: $(LIB_OBJS) $(BUILD_DIR)/test_cautious.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_grokfast: $(LIB_OBJS) $(BUILD_DIR)/test_grokfast.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_soap: $(LIB_OBJS) $(BUILD_DIR)/test_soap.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_shampoo: $(LIB_OBJS) $(BUILD_DIR)/test_shampoo.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_prodigy: $(LIB_OBJS) $(BUILD_DIR)/test_prodigy.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_stableadamw: $(LIB_OBJS) $(BUILD_DIR)/test_stableadamw.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_adopt: $(LIB_OBJS) $(BUILD_DIR)/test_adopt.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_swa: $(LIB_OBJS) $(BUILD_DIR)/test_swa.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_galore: $(LIB_OBJS) $(BUILD_DIR)/test_galore.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_apollo: $(LIB_OBJS) $(BUILD_DIR)/test_apollo.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_linoss: $(LIB_OBJS) $(BUILD_DIR)/test_linoss.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_agent_attention: $(LIB_OBJS) $(BUILD_DIR)/test_agent_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hyper_connection: $(LIB_OBJS) $(BUILD_DIR)/test_hyper_connection.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_diff_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_diff_transformer.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_hymba: $(LIB_OBJS) $(BUILD_DIR)/test_hymba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_log_linear_attention: $(LIB_OBJS) $(BUILD_DIR)/test_log_linear_attention.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_sliding_window: $(LIB_OBJS) $(BUILD_DIR)/test_sliding_window.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_nsa: $(LIB_OBJS) $(BUILD_DIR)/test_nsa.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_shla: $(LIB_OBJS) $(BUILD_DIR)/test_shla.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_fox: $(LIB_OBJS) $(BUILD_DIR)/test_fox.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mosa: $(LIB_OBJS) $(BUILD_DIR)/test_mosa.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_block_sparse_flash: $(LIB_OBJS) $(BUILD_DIR)/test_block_sparse_flash.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_rwkv7_parallel: $(LIB_OBJS) $(BUILD_DIR)/test_rwkv7_parallel.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_moe_mamba: $(LIB_OBJS) $(BUILD_DIR)/test_moe_mamba.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_soft_moe: $(LIB_OBJS) $(BUILD_DIR)/test_soft_moe.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_titans_mac: $(LIB_OBJS) $(BUILD_DIR)/test_titans_mac.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_titans_mag: $(LIB_OBJS) $(BUILD_DIR)/test_titans_mag.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_titans_mal: $(LIB_OBJS) $(BUILD_DIR)/test_titans_mal.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ttt_linear: $(LIB_OBJS) $(BUILD_DIR)/test_ttt_linear.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_ttt_mlp: $(LIB_OBJS) $(BUILD_DIR)/test_ttt_mlp.o
	$(CXX) $^ -o $@

$(BUILD_DIR)/test_mamba_bidirectional: $(LIB_OBJS) $(BUILD_DIR)/test_mamba_bidirectional.o
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

tests: setup $(BUILD_DIR)/test_realnvp $(BUILD_DIR)/test_neural_spline_flow $(BUILD_DIR)/test_ddpm $(BUILD_DIR)/test_adabelief $(BUILD_DIR)/test_s4 \
$(BUILD_DIR)/test_lion $(BUILD_DIR)/test_sophia $(BUILD_DIR)/test_sam \
$(BUILD_DIR)/test_gradient_check \
$(BUILD_DIR)/test_rmsnorm $(BUILD_DIR)/test_spectral_norm $(BUILD_DIR)/test_dynamic_tanh $(BUILD_DIR)/test_layer_scale $(BUILD_DIR)/test_wgan_gp $(BUILD_DIR)/test_flash_attention $(BUILD_DIR)/test_flash_attention_v2 \
$(BUILD_DIR)/test_vit $(BUILD_DIR)/test_distribution_losses $(BUILD_DIR)/test_self_supervised_losses $(BUILD_DIR)/test_mmd_loss $(BUILD_DIR)/test_contrastive_losses \
$(BUILD_DIR)/test_siglip_loss $(BUILD_DIR)/test_metrics $(BUILD_DIR)/test_model_ema $(BUILD_DIR)/test_early_stopping $(BUILD_DIR)/test_training_history $(BUILD_DIR)/test_model_checkpoint $(BUILD_DIR)/test_dataloader $(BUILD_DIR)/test_cross_validation $(BUILD_DIR)/test_lightgbm_style $(BUILD_DIR)/test_label_smoothing $(BUILD_DIR)/test_clip_grad_norm $(BUILD_DIR)/test_mixup_cutmix $(BUILD_DIR)/test_elastic_net \
$(BUILD_DIR)/test_activations $(BUILD_DIR)/test_legacy_adaptive $(BUILD_DIR)/test_gat_gradient $(BUILD_DIR)/test_gat_verify \
$(BUILD_DIR)/test_gat_attention $(BUILD_DIR)/test_coord_network $(BUILD_DIR)/test_avgpool2d $(BUILD_DIR)/test_gin \
$(BUILD_DIR)/test_ddpm $(BUILD_DIR)/test_nystrom_attention $(BUILD_DIR)/test_deep_gcn $(BUILD_DIR)/test_lightgcn \
$(BUILD_DIR)/test_patchy_san $(BUILD_DIR)/test_pna $(BUILD_DIR)/test_edgeconv $(BUILD_DIR)/test_dmon \
$(BUILD_DIR)/test_mha_backward $(BUILD_DIR)/test_gmlp $(BUILD_DIR)/test_mlp_mixer $(BUILD_DIR)/test_hyena $(BUILD_DIR)/test_striped_hyena $(BUILD_DIR)/test_hyena_dna $(BUILD_DIR)/test_convnext $(BUILD_DIR)/test_conformer $(BUILD_DIR)/test_linformer $(BUILD_DIR)/test_mamba $(BUILD_DIR)/test_xlstm $(BUILD_DIR)/test_min_gru $(BUILD_DIR)/test_min_lstm $(BUILD_DIR)/test_mogrifier_lstm \
$(BUILD_DIR)/test_mamba2 $(BUILD_DIR)/test_rwkv $(BUILD_DIR)/test_retnet $(BUILD_DIR)/test_mlstm $(BUILD_DIR)/test_h3 $(BUILD_DIR)/test_deltanet $(BUILD_DIR)/test_gated_deltanet $(BUILD_DIR)/test_mamba3 $(BUILD_DIR)/test_jamba $(BUILD_DIR)/test_gla $(BUILD_DIR)/test_griffin $(BUILD_DIR)/test_xlstm_block $(BUILD_DIR)/test_deepseek_moe $(BUILD_DIR)/test_hyper_mixing $(BUILD_DIR)/test_mega $(BUILD_DIR)/test_mambabyte \
$(BUILD_DIR)/test_flow_matching \
$(BUILD_DIR)/test_dit \
$(BUILD_DIR)/test_timesnet \
$(BUILD_DIR)/test_hawk \
$(BUILD_DIR)/test_performer $(BUILD_DIR)/test_cosformer $(BUILD_DIR)/test_gqa $(BUILD_DIR)/test_expire_span $(BUILD_DIR)/test_aft $(BUILD_DIR)/test_aft_local \
$(BUILD_DIR)/test_aft_conv $(BUILD_DIR)/test_lsh_attention $(BUILD_DIR)/test_conv_attention $(BUILD_DIR)/test_conv_bert \
$(BUILD_DIR)/test_pixelcnn $(BUILD_DIR)/test_egnn $(BUILD_DIR)/test_sparse_moe $(BUILD_DIR)/test_moe_router $(BUILD_DIR)/test_knn_classifier \
$(BUILD_DIR)/test_tree_lstm $(BUILD_DIR)/test_hopfield $(BUILD_DIR)/test_rope_v $(BUILD_DIR)/test_yarn_rope $(BUILD_DIR)/test_slot_attention \
$(BUILD_DIR)/test_fnet $(BUILD_DIR)/test_magnitude_pruning $(BUILD_DIR)/test_stochastic_depth $(BUILD_DIR)/test_ff_layer $(BUILD_DIR)/test_lora $(BUILD_DIR)/test_bitnet $(BUILD_DIR)/test_dcn_v2 \
$(BUILD_DIR)/test_monarch_mixer \
$(BUILD_DIR)/test_set_transformer \
$(BUILD_DIR)/test_fastformer \
$(BUILD_DIR)/test_gatv2 \
$(BUILD_DIR)/test_sagpool \
$(BUILD_DIR)/test_tome \
$(BUILD_DIR)/test_clustered_attention \
$(BUILD_DIR)/test_window_attention \
$(BUILD_DIR)/test_linear_attention \
$(BUILD_DIR)/test_span_extractor $(BUILD_DIR)/test_mla $(BUILD_DIR)/test_mixture_of_depths $(BUILD_DIR)/test_graphsage \
$(BUILD_DIR)/test_capsule $(BUILD_DIR)/test_kan $(BUILD_DIR)/test_fourier_kan $(BUILD_DIR)/test_gumbel_softmax $(BUILD_DIR)/test_bigbird \
$(BUILD_DIR)/test_tabnet $(BUILD_DIR)/test_alibi $(BUILD_DIR)/test_spatial_transformer $(BUILD_DIR)/test_transformer_decoder \
$(BUILD_DIR)/test_hypernetwork $(BUILD_DIR)/test_instance_norm $(BUILD_DIR)/test_ademamix $(BUILD_DIR)/test_sgd_nesterov \
$(BUILD_DIR)/test_lr_schedulers $(BUILD_DIR)/test_muon \
$(BUILD_DIR)/test_gmm $(BUILD_DIR)/test_adaln_zero $(BUILD_DIR)/test_adafactor $(BUILD_DIR)/test_segmentation_losses \
$(BUILD_DIR)/test_yogi $(BUILD_DIR)/test_radam $(BUILD_DIR)/test_adan $(BUILD_DIR)/test_lookahead $(BUILD_DIR)/test_diffgrad \
$(BUILD_DIR)/test_mars $(BUILD_DIR)/test_lars $(BUILD_DIR)/test_novograd $(BUILD_DIR)/test_came $(BUILD_DIR)/test_schedule_free_adamw \
$(BUILD_DIR)/test_schedule_free_sgd $(BUILD_DIR)/test_signum $(BUILD_DIR)/test_adam_mini $(BUILD_DIR)/test_dadaptation $(BUILD_DIR)/test_gradient_centralization $(BUILD_DIR)/test_adamp $(BUILD_DIR)/test_lamb $(BUILD_DIR)/test_cautious $(BUILD_DIR)/test_grokfast $(BUILD_DIR)/test_soap $(BUILD_DIR)/test_shampoo $(BUILD_DIR)/test_prodigy $(BUILD_DIR)/test_stableadamw $(BUILD_DIR)/test_adopt $(BUILD_DIR)/test_swa $(BUILD_DIR)/test_galore $(BUILD_DIR)/test_apollo $(BUILD_DIR)/test_ltc $(BUILD_DIR)/test_neural_ode $(BUILD_DIR)/test_hgrn $(BUILD_DIR)/test_rwkv7 $(BUILD_DIR)/test_rwkv5 $(BUILD_DIR)/test_rwkv6 $(BUILD_DIR)/test_linoss $(BUILD_DIR)/test_agent_attention $(BUILD_DIR)/test_hyper_connection $(BUILD_DIR)/test_diff_transformer $(BUILD_DIR)/test_hymba $(BUILD_DIR)/test_log_linear_attention $(BUILD_DIR)/test_sliding_window $(BUILD_DIR)/test_nsa $(BUILD_DIR)/test_shla $(BUILD_DIR)/test_block_sparse_flash $(BUILD_DIR)/test_rwkv7_parallel $(BUILD_DIR)/test_moe_mamba $(BUILD_DIR)/test_titans_mac $(BUILD_DIR)/test_titans_mag $(BUILD_DIR)/test_titans_mal $(BUILD_DIR)/test_fox $(BUILD_DIR)/test_mosa $(BUILD_DIR)/test_soft_moe $(BUILD_DIR)/test_ttt_linear $(BUILD_DIR)/test_ttt_mlp $(BUILD_DIR)/test_mamba_bidirectional $(BUILD_DIR)/test_based $(BUILD_DIR)/test_ft_transformer $(BUILD_DIR)/test_gau $(BUILD_DIR)/test_lambda_layer $(BUILD_DIR)/test_mamba_conv $(BUILD_DIR)/test_stick_breaking $(BUILD_DIR)/test_ngpt $(BUILD_DIR)/test_multi_token_prediction $(BUILD_DIR)/test_tokenformer $(BUILD_DIR)/test_deformable_attention $(BUILD_DIR)/test_multi_scale_deformable_attention $(BUILD_DIR)/test_power_attention $(BUILD_DIR)/test_gated_slot_attention $(BUILD_DIR)/test_hilo_attention $(BUILD_DIR)/test_s5 $(BUILD_DIR)/test_act $(BUILD_DIR)/test_switch_transformer $(BUILD_DIR)/test_vision_mamba $(BUILD_DIR)/test_axial_attention $(BUILD_DIR)/test_three_d_axial_attention $(BUILD_DIR)/test_sparse_mixer $(BUILD_DIR)/test_mixture_of_softmaxes $(BUILD_DIR)/test_focal_modulation $(BUILD_DIR)/test_qk_norm $(BUILD_DIR)/test_cheby_kan $(BUILD_DIR)/test_timestep_norm $(BUILD_DIR)/test_timestep_norm_grad $(BUILD_DIR)/test_complex_ema $(BUILD_DIR)/test_complex_ema_grad $(BUILD_DIR)/test_swiglu $(BUILD_DIR)/test_megalodon $(BUILD_DIR)/test_megalodon_grad

run_tests: tests
	@rm -f .run_tests_failed
	@echo "=== Running test_flash_attention_v2 ==="; if ./$(BUILD_DIR)/test_flash_attention_v2; then :; else echo "test_flash_attention_v2" >> .run_tests_failed; fi
	@echo "=== Running test_s4 ==="; if ./$(BUILD_DIR)/test_s4; then :; else echo "test_s4" >> .run_tests_failed; fi
	@echo "=== Running test_neural_spline_flow ==="; if ./$(BUILD_DIR)/test_neural_spline_flow; then :; else echo "test_neural_spline_flow" >> .run_tests_failed; fi
	@echo "=== Running test_realnvp ==="; if ./$(BUILD_DIR)/test_realnvp; then :; else echo "test_realnvp" >> .run_tests_failed; fi
	@echo "=== Running test_ddpm ==="; if ./$(BUILD_DIR)/test_ddpm; then :; else echo "test_ddpm" >> .run_tests_failed; fi
	@echo "=== Running test_adabelief ==="; if ./$(BUILD_DIR)/test_adabelief; then :; else echo "test_adabelief" >> .run_tests_failed; fi
	@echo "=== Running test_gradient_check ==="; if ./$(BUILD_DIR)/test_gradient_check; then :; else echo "test_gradient_check" >> .run_tests_failed; fi
	@echo "=== Running test_flash_attention ==="; if ./$(BUILD_DIR)/test_flash_attention; then :; else echo "test_flash_attention" >> .run_tests_failed; fi
	@echo "=== Running test_rmsnorm ==="; if ./$(BUILD_DIR)/test_rmsnorm; then :; else echo "test_rmsnorm" >> .run_tests_failed; fi
	@echo "=== Running test_spectral_norm ==="; if ./$(BUILD_DIR)/test_spectral_norm; then :; else echo "test_spectral_norm" >> .run_tests_failed; fi
	@echo "=== Running test_dynamic_tanh ==="; if ./$(BUILD_DIR)/test_dynamic_tanh; then :; else echo "test_dynamic_tanh" >> .run_tests_failed; fi
	@echo "=== Running test_layer_scale ==="; if ./$(BUILD_DIR)/test_layer_scale; then :; else echo "test_layer_scale" >> .run_tests_failed; fi
	@echo "=== Running test_avgpool2d ==="; if ./$(BUILD_DIR)/test_avgpool2d; then :; else echo "test_avgpool2d" >> .run_tests_failed; fi
	@echo "=== Running test_deep_gcn ==="; if ./$(BUILD_DIR)/test_deep_gcn; then :; else echo "test_deep_gcn" >> .run_tests_failed; fi
	@echo "=== Running test_dmon ==="; if ./$(BUILD_DIR)/test_dmon; then :; else echo "test_dmon" >> .run_tests_failed; fi
	@echo "=== Running test_edgeconv ==="; if ./$(BUILD_DIR)/test_edgeconv; then :; else echo "test_edgeconv" >> .run_tests_failed; fi
	@echo "=== Running test_gat_gradient ==="; if ./$(BUILD_DIR)/test_gat_gradient; then :; else echo "test_gat_gradient" >> .run_tests_failed; fi
	@echo "=== Running test_gat_verify ==="; if ./$(BUILD_DIR)/test_gat_verify; then :; else echo "test_gat_verify" >> .run_tests_failed; fi
	@echo "=== Running test_gin ==="; if ./$(BUILD_DIR)/test_gin; then :; else echo "test_gin" >> .run_tests_failed; fi
	@echo "=== Running test_gmlp ==="; if ./$(BUILD_DIR)/test_gmlp; then :; else echo "test_gmlp" >> .run_tests_failed; fi
	@echo "=== Running test_mlp_mixer ==="; if ./$(BUILD_DIR)/test_mlp_mixer; then :; else echo "test_mlp_mixer" >> .run_tests_failed; fi
	@echo "=== Running test_hyena ==="; if ./$(BUILD_DIR)/test_hyena; then :; else echo "test_hyena" >> .run_tests_failed; fi
	@echo "=== Running test_conformer ==="; if ./$(BUILD_DIR)/test_conformer; then :; else echo "test_conformer" >> .run_tests_failed; fi
	@echo "=== Running test_lightgcn ==="; if ./$(BUILD_DIR)/test_lightgcn; then :; else echo "test_lightgcn" >> .run_tests_failed; fi
	@echo "=== Running test_lsh_attention ==="; if ./$(BUILD_DIR)/test_lsh_attention; then :; else echo "test_lsh_attention" >> .run_tests_failed; fi
	@echo "=== Running test_nystrom_attention ==="; if ./$(BUILD_DIR)/test_nystrom_attention; then :; else echo "test_nystrom_attention" >> .run_tests_failed; fi
	@echo "=== Running test_patchy_san ==="; if ./$(BUILD_DIR)/test_patchy_san; then :; else echo "test_patchy_san" >> .run_tests_failed; fi
	@echo "=== Running test_pna ==="; if ./$(BUILD_DIR)/test_pna; then :; else echo "test_pna" >> .run_tests_failed; fi
	@echo "=== Running test_linformer ==="; if ./$(BUILD_DIR)/test_linformer; then :; else echo "test_linformer" >> .run_tests_failed; fi
	@echo "=== Running test_mamba_conv ==="; if ./$(BUILD_DIR)/test_mamba_conv; then :; else echo "test_mamba_conv" >> .run_tests_failed; fi
	@echo "=== Running test_mamba ==="; if ./$(BUILD_DIR)/test_mamba; then :; else echo "test_mamba" >> .run_tests_failed; fi
	@echo "=== Running test_xlstm ==="; if ./$(BUILD_DIR)/test_xlstm; then :; else echo "test_xlstm" >> .run_tests_failed; fi
	@echo "=== Running test_min_gru ==="; if ./$(BUILD_DIR)/test_min_gru; then :; else echo "test_min_gru" >> .run_tests_failed; fi
	@echo "=== Running test_min_lstm ==="; if ./$(BUILD_DIR)/test_min_lstm; then :; else echo "test_min_lstm" >> .run_tests_failed; fi
	@echo "=== Running test_mogrifier_lstm ==="; if ./$(BUILD_DIR)/test_mogrifier_lstm; then :; else echo "test_mogrifier_lstm" >> .run_tests_failed; fi
	@echo "=== Running test_mamba2 ==="; if ./$(BUILD_DIR)/test_mamba2; then :; else echo "test_mamba2" >> .run_tests_failed; fi
	@echo "=== Running test_rwkv ==="; if ./$(BUILD_DIR)/test_rwkv; then :; else echo "test_rwkv" >> .run_tests_failed; fi
	@echo "=== Running test_retnet ==="; if ./$(BUILD_DIR)/test_retnet; then :; else echo "test_retnet" >> .run_tests_failed; fi
	@echo "=== Running test_mlstm ==="; if ./$(BUILD_DIR)/test_mlstm; then :; else echo "test_mlstm" >> .run_tests_failed; fi
	@echo "=== Running test_h3 ==="; if ./$(BUILD_DIR)/test_h3; then :; else echo "test_h3" >> .run_tests_failed; fi
	@echo "=== Running test_deltanet ==="; if ./$(BUILD_DIR)/test_deltanet; then :; else echo "test_deltanet" >> .run_tests_failed; fi
	@echo "=== Running test_gated_deltanet ==="; if ./$(BUILD_DIR)/test_gated_deltanet; then :; else echo "test_gated_deltanet" >> .run_tests_failed; fi
	@echo "=== Running test_mamba3 ==="; if ./$(BUILD_DIR)/test_mamba3; then :; else echo "test_mamba3" >> .run_tests_failed; fi
	@echo "=== Running test_jamba ==="; if ./$(BUILD_DIR)/test_jamba; then :; else echo "test_jamba" >> .run_tests_failed; fi
	@echo "=== Running test_griffin ==="; if ./$(BUILD_DIR)/test_griffin; then :; else echo "test_griffin" >> .run_tests_failed; fi
	@echo "=== Running test_xlstm_block ==="; if ./$(BUILD_DIR)/test_xlstm_block; then :; else echo "test_xlstm_block" >> .run_tests_failed; fi
	@echo "=== Running test_deepseek_moe ==="; if ./$(BUILD_DIR)/test_deepseek_moe; then :; else echo "test_deepseek_moe" >> .run_tests_failed; fi
	@echo "=== Running test_hyper_mixing ==="; if ./$(BUILD_DIR)/test_hyper_mixing; then :; else echo "test_hyper_mixing" >> .run_tests_failed; fi
	@echo "=== Running test_mega ==="; if ./$(BUILD_DIR)/test_mega; then :; else echo "test_mega" >> .run_tests_failed; fi
	@echo "=== Running test_mambabyte ==="; if ./$(BUILD_DIR)/test_mambabyte; then :; else echo "test_mambabyte" >> .run_tests_failed; fi
	@echo "=== Running test_gla ==="; if ./$(BUILD_DIR)/test_gla; then :; else echo "test_gla" >> .run_tests_failed; fi
	@echo "=== Running test_flow_matching ==="; if ./$(BUILD_DIR)/test_flow_matching; then :; else echo "test_flow_matching" >> .run_tests_failed; fi
	@echo "=== Running test_dit ==="; if ./$(BUILD_DIR)/test_dit; then :; else echo "test_dit" >> .run_tests_failed; fi
	@echo "=== Running test_timesnet ==="; if ./$(BUILD_DIR)/test_timesnet; then :; else echo "test_timesnet" >> .run_tests_failed; fi
	@echo "=== Running test_hawk ==="; if ./$(BUILD_DIR)/test_hawk; then :; else echo "test_hawk" >> .run_tests_failed; fi
	@echo "=== Running test_performer ==="; if ./$(BUILD_DIR)/test_performer; then :; else echo "test_performer" >> .run_tests_failed; fi
	@echo "=== Running test_cosformer ==="; if ./$(BUILD_DIR)/test_cosformer; then :; else echo "test_cosformer" >> .run_tests_failed; fi
	@echo "=== Running test_gqa ==="; if ./$(BUILD_DIR)/test_gqa; then :; else echo "test_gqa" >> .run_tests_failed; fi
	@echo "=== Running test_expire_span ==="; if ./$(BUILD_DIR)/test_expire_span; then :; else echo "test_expire_span" >> .run_tests_failed; fi
	@echo "=== Running test_aft ==="; if ./$(BUILD_DIR)/test_aft; then :; else echo "test_aft" >> .run_tests_failed; fi
	@echo "=== Running test_aft_local ==="; if ./$(BUILD_DIR)/test_aft_local; then :; else echo "test_aft_local" >> .run_tests_failed; fi
	@echo "=== Running test_aft_conv ==="; if ./$(BUILD_DIR)/test_aft_conv; then :; else echo "test_aft_conv" >> .run_tests_failed; fi
	@echo "=== Running test_conv_attention ==="; if ./$(BUILD_DIR)/test_conv_attention; then :; else echo "test_conv_attention" >> .run_tests_failed; fi
	@echo "=== Running test_conv_bert ==="; if ./$(BUILD_DIR)/test_conv_bert; then :; else echo "test_conv_bert" >> .run_tests_failed; fi
	@echo "=== Running test_pixelcnn ==="; if ./$(BUILD_DIR)/test_pixelcnn; then :; else echo "test_pixelcnn" >> .run_tests_failed; fi
	@echo "=== Running test_egnn ==="; if ./$(BUILD_DIR)/test_egnn; then :; else echo "test_egnn" >> .run_tests_failed; fi
	@echo "=== Running test_sparse_moe ==="; if ./$(BUILD_DIR)/test_sparse_moe; then :; else echo "test_sparse_moe" >> .run_tests_failed; fi
	@echo "=== Running test_moe_router ==="; if ./$(BUILD_DIR)/test_moe_router; then :; else echo "test_moe_router" >> .run_tests_failed; fi
	@echo "=== Running test_knn_classifier ==="; if ./$(BUILD_DIR)/test_knn_classifier; then :; else echo "test_knn_classifier" >> .run_tests_failed; fi
	@echo "=== Running test_tree_lstm ==="; if ./$(BUILD_DIR)/test_tree_lstm; then :; else echo "test_tree_lstm" >> .run_tests_failed; fi
	@echo "=== Running test_hopfield ==="; if ./$(BUILD_DIR)/test_hopfield; then :; else echo "test_hopfield" >> .run_tests_failed; fi
	@echo "=== Running test_rope_v ==="; if ./$(BUILD_DIR)/test_rope_v; then :; else echo "test_rope_v" >> .run_tests_failed; fi
	@echo "=== Running test_yarn_rope ==="; if ./$(BUILD_DIR)/test_yarn_rope; then :; else echo "test_yarn_rope" >> .run_tests_failed; fi
	@echo "=== Running test_slot_attention ==="; if ./$(BUILD_DIR)/test_slot_attention; then :; else echo "test_slot_attention" >> .run_tests_failed; fi
	@echo "=== Running test_fnet ==="; if ./$(BUILD_DIR)/test_fnet; then :; else echo "test_fnet" >> .run_tests_failed; fi
	@echo "=== Running test_magnitude_pruning ==="; if ./$(BUILD_DIR)/test_magnitude_pruning; then :; else echo "test_magnitude_pruning" >> .run_tests_failed; fi
	@echo "=== Running test_stochastic_depth ==="; if ./$(BUILD_DIR)/test_stochastic_depth; then :; else echo "test_stochastic_depth" >> .run_tests_failed; fi
	@echo "=== Running test_lora ==="; if ./$(BUILD_DIR)/test_lora; then :; else echo "test_lora" >> .run_tests_failed; fi
	@echo "=== Running test_bitnet ==="; if ./$(BUILD_DIR)/test_bitnet; then :; else echo "test_bitnet" >> .run_tests_failed; fi
	@echo "=== Running test_dcn_v2 ==="; if ./$(BUILD_DIR)/test_dcn_v2; then :; else echo "test_dcn_v2" >> .run_tests_failed; fi
	@echo "=== Running test_monarch_mixer ==="; if ./$(BUILD_DIR)/test_monarch_mixer; then :; else echo "test_monarch_mixer" >> .run_tests_failed; fi
	@echo "=== Running test_set_transformer ==="; if ./$(BUILD_DIR)/test_set_transformer; then :; else echo "test_set_transformer" >> .run_tests_failed; fi
	@echo "=== Running test_ff_layer ==="; if ./$(BUILD_DIR)/test_ff_layer; then :; else echo "test_ff_layer" >> .run_tests_failed; fi
	@echo "=== Running test_span_extractor ==="; if ./$(BUILD_DIR)/test_span_extractor; then :; else echo "test_span_extractor" >> .run_tests_failed; fi
	@echo "=== Running test_mla ==="; if ./$(BUILD_DIR)/test_mla; then :; else echo "test_mla" >> .run_tests_failed; fi
	@echo "=== Running test_mixture_of_depths ==="; if ./$(BUILD_DIR)/test_mixture_of_depths; then :; else echo "test_mixture_of_depths" >> .run_tests_failed; fi
	@echo "=== Running test_graphsage ==="; if ./$(BUILD_DIR)/test_graphsage; then :; else echo "test_graphsage" >> .run_tests_failed; fi
	@echo "=== Running test_capsule ==="; if ./$(BUILD_DIR)/test_capsule; then :; else echo "test_capsule" >> .run_tests_failed; fi
	@echo "=== Running test_kan ==="; if ./$(BUILD_DIR)/test_kan; then :; else echo "test_kan" >> .run_tests_failed; fi
	@echo "=== Running test_fourier_kan ==="; if ./$(BUILD_DIR)/test_fourier_kan; then :; else echo "test_fourier_kan" >> .run_tests_failed; fi
	@echo "=== Running test_gumbel_softmax ==="; if ./$(BUILD_DIR)/test_gumbel_softmax; then :; else echo "test_gumbel_softmax" >> .run_tests_failed; fi
	@echo "=== Running test_lion ==="; if ./$(BUILD_DIR)/test_lion; then :; else echo "test_lion" >> .run_tests_failed; fi
	@echo "=== Running test_sophia ==="; if ./$(BUILD_DIR)/test_sophia; then :; else echo "test_sophia" >> .run_tests_failed; fi
	@echo "=== Running test_sam ==="; if ./$(BUILD_DIR)/test_sam; then :; else echo "test_sam" >> .run_tests_failed; fi
	@echo "=== Running test_bigbird ==="; if ./$(BUILD_DIR)/test_bigbird; then :; else echo "test_bigbird" >> .run_tests_failed; fi
	@echo "=== Running test_tabnet ==="; if ./$(BUILD_DIR)/test_tabnet; then :; else echo "test_tabnet" >> .run_tests_failed; fi
	@echo "=== Running test_lookahead ==="; if ./$(BUILD_DIR)/test_lookahead; then :; else echo "test_lookahead" >> .run_tests_failed; fi
	@echo "=== Running test_diffgrad ==="; if ./$(BUILD_DIR)/test_diffgrad; then :; else echo "test_diffgrad" >> .run_tests_failed; fi
	@echo "=== Running test_alibi ==="; if ./$(BUILD_DIR)/test_alibi; then :; else echo "test_alibi" >> .run_tests_failed; fi
	@echo "=== Running test_spatial_transformer ==="; if ./$(BUILD_DIR)/test_spatial_transformer; then :; else echo "test_spatial_transformer" >> .run_tests_failed; fi
	@echo "=== Running test_transformer_decoder ==="; if ./$(BUILD_DIR)/test_transformer_decoder; then :; else echo "test_transformer_decoder" >> .run_tests_failed; fi
	@echo "=== Running test_hypernetwork ==="; if ./$(BUILD_DIR)/test_hypernetwork; then :; else echo "test_hypernetwork" >> .run_tests_failed; fi
	@echo "=== Running test_distribution_losses ==="; if ./$(BUILD_DIR)/test_distribution_losses; then :; else echo "test_distribution_losses" >> .run_tests_failed; fi
	@echo "=== Running test_self_supervised_losses ==="; if ./$(BUILD_DIR)/test_self_supervised_losses; then :; else echo "test_self_supervised_losses" >> .run_tests_failed; fi
	@echo "=== Running test_mmd_loss ==="; if ./$(BUILD_DIR)/test_mmd_loss; then :; else echo "test_mmd_loss" >> .run_tests_failed; fi
	@echo "=== Running test_contrastive_losses ==="; if ./$(BUILD_DIR)/test_contrastive_losses; then :; else echo "test_contrastive_losses" >> .run_tests_failed; fi
	@echo "=== Running test_siglip_loss ==="; if ./$(BUILD_DIR)/test_siglip_loss; then :; else echo "test_siglip_loss" >> .run_tests_failed; fi
	@echo "=== Running test_metrics ==="; if ./$(BUILD_DIR)/test_metrics; then :; else echo "test_metrics" >> .run_tests_failed; fi
	@echo "=== Running test_model_ema ==="; if ./$(BUILD_DIR)/test_model_ema; then :; else echo "test_model_ema" >> .run_tests_failed; fi
	@echo "=== Running test_early_stopping ==="; if ./$(BUILD_DIR)/test_early_stopping; then :; else echo "test_early_stopping" >> .run_tests_failed; fi
	@echo "=== Running test_training_history ==="; if ./$(BUILD_DIR)/test_training_history; then :; else echo "test_training_history" >> .run_tests_failed; fi
	@echo "=== Running test_model_checkpoint ==="; if ./$(BUILD_DIR)/test_model_checkpoint; then :; else echo "test_model_checkpoint" >> .run_tests_failed; fi
	@echo "=== Running test_dataloader ==="; if ./$(BUILD_DIR)/test_dataloader; then :; else echo "test_dataloader" >> .run_tests_failed; fi
	@echo "=== Running test_cross_validation ==="; if ./$(BUILD_DIR)/test_cross_validation; then :; else echo "test_cross_validation" >> .run_tests_failed; fi
	@echo "=== Running test_lightgbm_style ==="; if ./$(BUILD_DIR)/test_lightgbm_style; then :; else echo "test_lightgbm_style" >> .run_tests_failed; fi
	@echo "=== Running test_label_smoothing ==="; if ./$(BUILD_DIR)/test_label_smoothing; then :; else echo "test_label_smoothing" >> .run_tests_failed; fi
	@echo "=== Running test_clip_grad_norm ==="; if ./$(BUILD_DIR)/test_clip_grad_norm; then :; else echo "test_clip_grad_norm" >> .run_tests_failed; fi
	@echo "=== Running test_mixup_cutmix ==="; if ./$(BUILD_DIR)/test_mixup_cutmix; then :; else echo "test_mixup_cutmix" >> .run_tests_failed; fi
	@echo "=== Running test_elastic_net ==="; if ./$(BUILD_DIR)/test_elastic_net; then :; else echo "test_elastic_net" >> .run_tests_failed; fi
	@echo "=== Running test_activations ==="; if ./$(BUILD_DIR)/test_activations; then :; else echo "test_activations" >> .run_tests_failed; fi
	@echo "=== Running test_legacy_adaptive ==="; if ./$(BUILD_DIR)/test_legacy_adaptive; then :; else echo "test_legacy_adaptive" >> .run_tests_failed; fi
	@echo "=== Running test_instance_norm ==="; if ./$(BUILD_DIR)/test_instance_norm; then :; else echo "test_instance_norm" >> .run_tests_failed; fi
	@echo "=== Running test_ademamix ==="; if ./$(BUILD_DIR)/test_ademamix; then :; else echo "test_ademamix" >> .run_tests_failed; fi
	@echo "=== Running test_sgd_nesterov ==="; if ./$(BUILD_DIR)/test_sgd_nesterov; then :; else echo "test_sgd_nesterov" >> .run_tests_failed; fi
	@echo "=== Running test_lr_schedulers ==="; if ./$(BUILD_DIR)/test_lr_schedulers; then :; else echo "test_lr_schedulers" >> .run_tests_failed; fi
	@echo "=== Running test_muon ==="; if ./$(BUILD_DIR)/test_muon; then :; else echo "test_muon" >> .run_tests_failed; fi
	@echo "=== Running test_gmm ==="; if ./$(BUILD_DIR)/test_gmm; then :; else echo "test_gmm" >> .run_tests_failed; fi
	@echo "=== Running test_adaln_zero ==="; if ./$(BUILD_DIR)/test_adaln_zero; then :; else echo "test_adaln_zero" >> .run_tests_failed; fi
	@echo "=== Running test_adafactor ==="; if ./$(BUILD_DIR)/test_adafactor; then :; else echo "test_adafactor" >> .run_tests_failed; fi
	@echo "=== Running test_segmentation_losses ==="; if ./$(BUILD_DIR)/test_segmentation_losses; then :; else echo "test_segmentation_losses" >> .run_tests_failed; fi
	@echo "=== Running test_yogi ==="; if ./$(BUILD_DIR)/test_yogi; then :; else echo "test_yogi" >> .run_tests_failed; fi
	@echo "=== Running test_radam ==="; if ./$(BUILD_DIR)/test_radam; then :; else echo "test_radam" >> .run_tests_failed; fi
	@echo "=== Running test_adan ==="; if ./$(BUILD_DIR)/test_adan; then :; else echo "test_adan" >> .run_tests_failed; fi
	@echo "=== Running test_mars ==="; if ./$(BUILD_DIR)/test_mars; then :; else echo "test_mars" >> .run_tests_failed; fi
	@echo "=== Running test_lars ==="; if ./$(BUILD_DIR)/test_lars; then :; else echo "test_lars" >> .run_tests_failed; fi
	@echo "=== Running test_novograd ==="; if ./$(BUILD_DIR)/test_novograd; then :; else echo "test_novograd" >> .run_tests_failed; fi
	@echo "=== Running test_came ==="; if ./$(BUILD_DIR)/test_came; then :; else echo "test_came" >> .run_tests_failed; fi
	@echo "=== Running test_schedule_free_adamw ==="; if ./$(BUILD_DIR)/test_schedule_free_adamw; then :; else echo "test_schedule_free_adamw" >> .run_tests_failed; fi
	@echo "=== Running test_schedule_free_sgd ==="; if ./$(BUILD_DIR)/test_schedule_free_sgd; then :; else echo "test_schedule_free_sgd" >> .run_tests_failed; fi
	@echo "=== Running test_signum ==="; if ./$(BUILD_DIR)/test_signum; then :; else echo "test_signum" >> .run_tests_failed; fi
	@echo "=== Running test_adam_mini ==="; if ./$(BUILD_DIR)/test_adam_mini; then :; else echo "test_adam_mini" >> .run_tests_failed; fi
	@echo "=== Running test_dadaptation ==="; if ./$(BUILD_DIR)/test_dadaptation; then :; else echo "test_dadaptation" >> .run_tests_failed; fi
	@echo "=== Running test_gradient_centralization ==="; if ./$(BUILD_DIR)/test_gradient_centralization; then :; else echo "test_gradient_centralization" >> .run_tests_failed; fi
	@echo "=== Running test_adamp ==="; if ./$(BUILD_DIR)/test_adamp; then :; else echo "test_adamp" >> .run_tests_failed; fi
	@echo "=== Running test_lamb ==="; if ./$(BUILD_DIR)/test_lamb; then :; else echo "test_lamb" >> .run_tests_failed; fi
	@echo "=== Running test_cautious ==="; if ./$(BUILD_DIR)/test_cautious; then :; else echo "test_cautious" >> .run_tests_failed; fi
	@echo "=== Running test_grokfast ==="; if ./$(BUILD_DIR)/test_grokfast; then :; else echo "test_grokfast" >> .run_tests_failed; fi
	@echo "=== Running test_soap ==="; if ./$(BUILD_DIR)/test_soap; then :; else echo "test_soap" >> .run_tests_failed; fi
	@echo "=== Running test_shampoo ==="; if ./$(BUILD_DIR)/test_shampoo; then :; else echo "test_shampoo" >> .run_tests_failed; fi
	@echo "=== Running test_prodigy ==="; if ./$(BUILD_DIR)/test_prodigy; then :; else echo "test_prodigy" >> .run_tests_failed; fi
	@echo "=== Running test_stableadamw ==="; if ./$(BUILD_DIR)/test_stableadamw; then :; else echo "test_stableadamw" >> .run_tests_failed; fi
	@echo "=== Running test_adopt ==="; if ./$(BUILD_DIR)/test_adopt; then :; else echo "test_adopt" >> .run_tests_failed; fi
	@echo "=== Running test_swa ==="; if ./$(BUILD_DIR)/test_swa; then :; else echo "test_swa" >> .run_tests_failed; fi
	@echo "=== Running test_galore ==="; if ./$(BUILD_DIR)/test_galore; then :; else echo "test_galore" >> .run_tests_failed; fi
	@echo "=== Running test_apollo ==="; if ./$(BUILD_DIR)/test_apollo; then :; else echo "test_apollo" >> .run_tests_failed; fi
	@echo "=== Running test_ltc ==="; if ./$(BUILD_DIR)/test_ltc; then :; else echo "test_ltc" >> .run_tests_failed; fi
	@echo "=== Running test_neural_ode ==="; if ./$(BUILD_DIR)/test_neural_ode; then :; else echo "test_neural_ode" >> .run_tests_failed; fi
	@echo "=== Running test_hgrn ==="; if ./$(BUILD_DIR)/test_hgrn; then :; else echo "test_hgrn" >> .run_tests_failed; fi
	@echo "=== Running test_rwkv7 ==="; if ./$(BUILD_DIR)/test_rwkv7; then :; else echo "test_rwkv7" >> .run_tests_failed; fi
	@echo "=== Running test_rwkv5 ==="; if ./$(BUILD_DIR)/test_rwkv5; then :; else echo "test_rwkv5" >> .run_tests_failed; fi
	@echo "=== Running test_rwkv6 ==="; if ./$(BUILD_DIR)/test_rwkv6; then :; else echo "test_rwkv6" >> .run_tests_failed; fi
	@echo "=== Running test_linoss ==="; if ./$(BUILD_DIR)/test_linoss; then :; else echo "test_linoss" >> .run_tests_failed; fi
	@echo "=== Running test_agent_attention ==="; if ./$(BUILD_DIR)/test_agent_attention; then :; else echo "test_agent_attention" >> .run_tests_failed; fi
	@echo "=== Running test_hyper_connection ==="; if ./$(BUILD_DIR)/test_hyper_connection; then :; else echo "test_hyper_connection" >> .run_tests_failed; fi
	@echo "=== Running test_diff_transformer ==="; if ./$(BUILD_DIR)/test_diff_transformer; then :; else echo "test_diff_transformer" >> .run_tests_failed; fi
	@echo "=== Running test_hymba ==="; if ./$(BUILD_DIR)/test_hymba; then :; else echo "test_hymba" >> .run_tests_failed; fi
	@echo "=== Running test_log_linear_attention ==="; if ./$(BUILD_DIR)/test_log_linear_attention; then :; else echo "test_log_linear_attention" >> .run_tests_failed; fi
	@echo "=== Running test_sliding_window ==="; if ./$(BUILD_DIR)/test_sliding_window; then :; else echo "test_sliding_window" >> .run_tests_failed; fi
	@echo "=== Running test_nsa ==="; if ./$(BUILD_DIR)/test_nsa; then :; else echo "test_nsa" >> .run_tests_failed; fi
	@echo "=== Running test_shla ==="; if ./$(BUILD_DIR)/test_shla; then :; else echo "test_shla" >> .run_tests_failed; fi
	@echo "=== Running test_block_sparse_flash ==="; if ./$(BUILD_DIR)/test_block_sparse_flash; then :; else echo "test_block_sparse_flash" >> .run_tests_failed; fi
	@echo "=== Running test_rwkv7_parallel ==="; if ./$(BUILD_DIR)/test_rwkv7_parallel; then :; else echo "test_rwkv7_parallel" >> .run_tests_failed; fi
	@echo "=== Running test_moe_mamba ==="; if ./$(BUILD_DIR)/test_moe_mamba; then :; else echo "test_moe_mamba" >> .run_tests_failed; fi
	@echo "=== Running test_titans_mac ==="; if ./$(BUILD_DIR)/test_titans_mac; then :; else echo "test_titans_mac" >> .run_tests_failed; fi
	@echo "=== Running test_titans_mag ==="; if ./$(BUILD_DIR)/test_titans_mag; then :; else echo "test_titans_mag" >> .run_tests_failed; fi
	@echo "=== Running test_titans_mal ==="; if ./$(BUILD_DIR)/test_titans_mal; then :; else echo "test_titans_mal" >> .run_tests_failed; fi
	@echo "=== Running test_fox ==="; if ./$(BUILD_DIR)/test_fox; then :; else echo "test_fox" >> .run_tests_failed; fi
	@echo "=== Running test_mosa ==="; if ./$(BUILD_DIR)/test_mosa; then :; else echo "test_mosa" >> .run_tests_failed; fi
	@echo "=== Running test_soft_moe ==="; if ./$(BUILD_DIR)/test_soft_moe; then :; else echo "test_soft_moe" >> .run_tests_failed; fi
	@echo "=== Running test_ttt_linear ==="; if ./$(BUILD_DIR)/test_ttt_linear; then :; else echo "test_ttt_linear" >> .run_tests_failed; fi
	@echo "=== Running test_ttt_mlp ==="; if ./$(BUILD_DIR)/test_ttt_mlp; then :; else echo "test_ttt_mlp" >> .run_tests_failed; fi
	@echo "=== Running test_mamba_bidirectional ==="; if ./$(BUILD_DIR)/test_mamba_bidirectional; then :; else echo "test_mamba_bidirectional" >> .run_tests_failed; fi
	@echo "=== Running test_based ==="; if ./$(BUILD_DIR)/test_based; then :; else echo "test_based" >> .run_tests_failed; fi
	@echo "=== Running test_ft_transformer ==="; if ./$(BUILD_DIR)/test_ft_transformer; then :; else echo "test_ft_transformer" >> .run_tests_failed; fi
	@echo "=== Running test_gau ==="; if ./$(BUILD_DIR)/test_gau; then :; else echo "test_gau" >> .run_tests_failed; fi
	@echo "=== Running test_lambda_layer ==="; if ./$(BUILD_DIR)/test_lambda_layer; then :; else echo "test_lambda_layer" >> .run_tests_failed; fi
	@echo "=== Running test_stick_breaking ==="; if ./$(BUILD_DIR)/test_stick_breaking; then :; else echo "test_stick_breaking" >> .run_tests_failed; fi
	@echo "=== Running test_ngpt ==="; if ./$(BUILD_DIR)/test_ngpt; then :; else echo "test_ngpt" >> .run_tests_failed; fi
	@echo "=== Running test_multi_token_prediction ==="; if ./$(BUILD_DIR)/test_multi_token_prediction; then :; else echo "test_multi_token_prediction" >> .run_tests_failed; fi
	@echo "=== Running test_tokenformer ==="; if ./$(BUILD_DIR)/test_tokenformer; then :; else echo "test_tokenformer" >> .run_tests_failed; fi
	@echo "=== Running test_deformable_attention ==="; if ./$(BUILD_DIR)/test_deformable_attention; then :; else echo "test_deformable_attention" >> .run_tests_failed; fi
	@echo "=== Running test_multi_scale_deformable_attention ==="; if ./$(BUILD_DIR)/test_multi_scale_deformable_attention; then :; else echo "test_multi_scale_deformable_attention" >> .run_tests_failed; fi
	@echo "=== Running test_power_attention ==="; if ./$(BUILD_DIR)/test_power_attention; then :; else echo "test_power_attention" >> .run_tests_failed; fi
	@echo "=== Running test_gated_slot_attention ==="; if ./$(BUILD_DIR)/test_gated_slot_attention; then :; else echo "test_gated_slot_attention" >> .run_tests_failed; fi
	@echo "=== Running test_hilo_attention ==="; if ./$(BUILD_DIR)/test_hilo_attention; then :; else echo "test_hilo_attention" >> .run_tests_failed; fi
	@echo "=== Running test_s5 ==="; if ./$(BUILD_DIR)/test_s5; then :; else echo "test_s5" >> .run_tests_failed; fi
	@echo "=== Running test_act ==="; if ./$(BUILD_DIR)/test_act; then :; else echo "test_act" >> .run_tests_failed; fi
	@echo "=== Running test_switch_transformer ==="; if ./$(BUILD_DIR)/test_switch_transformer; then :; else echo "test_switch_transformer" >> .run_tests_failed; fi
	@echo "=== Running test_vision_mamba ==="; if ./$(BUILD_DIR)/test_vision_mamba; then :; else echo "test_vision_mamba" >> .run_tests_failed; fi
	@echo "=== Running test_axial_attention ==="; if ./$(BUILD_DIR)/test_axial_attention; then :; else echo "test_axial_attention" >> .run_tests_failed; fi
	@echo "=== Running test_three_d_axial_attention ==="; if ./$(BUILD_DIR)/test_three_d_axial_attention; then :; else echo "test_three_d_axial_attention" >> .run_tests_failed; fi
	@echo "=== Running test_sparse_mixer ==="; if ./$(BUILD_DIR)/test_sparse_mixer; then :; else echo "test_sparse_mixer" >> .run_tests_failed; fi
	@echo "=== Running test_focal_modulation ==="; if ./$(BUILD_DIR)/test_focal_modulation; then :; else echo "test_focal_modulation" >> .run_tests_failed; fi
	@echo "=== Running test_mixture_of_softmaxes ==="; if ./$(BUILD_DIR)/test_mixture_of_softmaxes; then :; else echo "test_mixture_of_softmaxes" >> .run_tests_failed; fi
	@echo "=== Running test_fastformer ==="; if ./$(BUILD_DIR)/test_fastformer; then :; else echo "test_fastformer" >> .run_tests_failed; fi
	@echo "=== Running test_gatv2 ==="; if ./$(BUILD_DIR)/test_gatv2; then :; else echo "test_gatv2" >> .run_tests_failed; fi
	@echo "=== Running test_sagpool ==="; if ./$(BUILD_DIR)/test_sagpool; then :; else echo "test_sagpool" >> .run_tests_failed; fi
	@echo "=== Running test_tome ==="; if ./$(BUILD_DIR)/test_tome; then :; else echo "test_tome" >> .run_tests_failed; fi
	@echo "=== Running test_clustered_attention ==="; if ./$(BUILD_DIR)/test_clustered_attention; then :; else echo "test_clustered_attention" >> .run_tests_failed; fi
	@echo "=== Running test_window_attention ==="; if ./$(BUILD_DIR)/test_window_attention; then :; else echo "test_window_attention" >> .run_tests_failed; fi
	@echo "=== Running test_linear_attention ==="; if ./$(BUILD_DIR)/test_linear_attention; then :; else echo "test_linear_attention" >> .run_tests_failed; fi
	@echo "=== Running test_qk_norm ==="; if ./$(BUILD_DIR)/test_qk_norm; then :; else echo "test_qk_norm" >> .run_tests_failed; fi
	@echo "=== Running test_cheby_kan ==="; if ./$(BUILD_DIR)/test_cheby_kan; then :; else echo "test_cheby_kan" >> .run_tests_failed; fi
	@echo "=== Running test_timestep_norm ==="; if ./$(BUILD_DIR)/test_timestep_norm; then :; else echo "test_timestep_norm" >> .run_tests_failed; fi
	@echo "=== Running test_timestep_norm_grad ==="; if ./$(BUILD_DIR)/test_timestep_norm_grad; then :; else echo "test_timestep_norm_grad" >> .run_tests_failed; fi
	@echo "=== Running test_complex_ema ==="; if ./$(BUILD_DIR)/test_complex_ema; then :; else echo "test_complex_ema" >> .run_tests_failed; fi
	@echo "=== Running test_complex_ema_grad ==="; if ./$(BUILD_DIR)/test_complex_ema_grad; then :; else echo "test_complex_ema_grad" >> .run_tests_failed; fi
	@echo "=== Running test_swiglu ==="; if ./$(BUILD_DIR)/test_swiglu; then :; else echo "test_swiglu" >> .run_tests_failed; fi
	@echo "=== Running test_megalodon ==="; if ./$(BUILD_DIR)/test_megalodon; then :; else echo "test_megalodon" >> .run_tests_failed; fi
	@echo "=== Running test_megalodon_grad ==="; if ./$(BUILD_DIR)/test_megalodon_grad; then :; else echo "test_megalodon_grad" >> .run_tests_failed; fi
	@echo "=== Running test_striped_hyena ==="; if ./$(BUILD_DIR)/test_striped_hyena; then :; else echo "test_striped_hyena" >> .run_tests_failed; fi
	@echo "=== Running test_hyena_dna ==="; if ./$(BUILD_DIR)/test_hyena_dna; then :; else echo "test_hyena_dna" >> .run_tests_failed; fi
	@echo "=== Running test_convnext ==="; if ./$(BUILD_DIR)/test_convnext; then :; else echo "test_convnext" >> .run_tests_failed; fi
	@echo "=== Running test_mha_backward ==="; if ./$(BUILD_DIR)/test_mha_backward; then :; else echo "test_mha_backward" >> .run_tests_failed; fi
	@if [ -s .run_tests_failed ]; then \
		echo ""; echo "================ FAILING SUITES ================"; \
		sed "s|^|  |" .run_tests_failed; \
		echo "=================================================="; \
		echo ""; rm -f .run_tests_failed; \
		echo "run_tests: FAILED (listed above). Every suite ran."; \
		exit 1; \
	else \
		echo ""; echo "run_tests: all 224 suites PASSED"; \
		exit 0; \
	fi
clean:
	rm -rf $(BUILD_DIR)

# Escape hatch for the ODR-violation failure mode that -MMD now prevents in the
# normal case. If a header layout change ever still produces a stale object
# (e.g. a header swapped wholesale, or a .d sidecar pruned), nuke-deps forces a
# full rebuild, which is the only reliable recovery. A full rebuild is ~40 s.
nuke-deps:
	@find $(BUILD_DIR) -name '*.o' -delete
	@echo "Removed all object files. Next build will be a full rebuild."

