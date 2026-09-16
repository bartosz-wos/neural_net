# Sparse Mixer — Lee-Thorp & Ainslie, EMNLP Findings 2022

> **For Hermes:** Implement task-by-task using strict TDD.

**Goal:** Ship the Sparse Mixer architecture (https://arxiv.org/abs/2205.12399) as three C++ classes in `include/nn/layers/architectures/sparse_mixer.{h,cpp}` plus tests in `tests/test_sparse_mixer.cpp`, registered in `nn.h` umbrella and `Makefile`.

**Architecture:**
- `LinearMixingSublayer(d_model, seq_len)` — drop-in for self-attention in BERT-like blocks. Math: `y = T_h · T_seq · x` where `T_h : (d_model, d_model)` and `T_seq : (seq_len, seq_len)` are two dense, token-INDEPENDENT linear projections. Two Dense layers. No attention — just two matmuls.
- `ExpertsChoiceMoE(d_model, num_experts, expert_hidden=0, capacity_factor=1.0)` — Experts-Choice (Zhou et al. 2022) router. Each expert picks its top-k tokens (k = capacity), so an expert ALWAYS fills its buffer and a token may be routed to 0, 1, or multiple experts. Output is mean-pool over experts that selected the token (so any token gets at least the residual through, even if no expert picked it). No load-balancing loss needed because every expert fills capacity. Cached: gate logits (B, E), expert→token assignment (E, cap_e), per-expert cached inputs (cap_e, d_model).
- `SparseMixerBlock(d_model, seq_len, mode, num_experts=0, expert_hidden=0, capacity_factor=1.0, ffn_mult=4)` — pre-norm + mixing/attention sublayer + residual + pre-norm + MoE/dense FFN sublayer + residual. `mode ∈ {LINEAR_MIXING, SELF_ATTENTION}` selects the first sublayer. Second sublayer is `ExpertsChoiceMoE` if `num_experts > 0`, else a dense 2-layer GELU FFN with `ffn_mult` hidden width. Attention sublayer uses a hand-rolled single-head Q/K/V/O attention on the pre-norm output (the paper uses standard BERT attention in the top-4 layers).
- `SparseMixerModel(input_dim, d_model, output_dim, seq_len, num_layers, num_attention_layers, num_experts, expert_hidden=0, capacity_factor=1.0, ffn_mult=4)` — Dense input projection → stack of `num_layers` SparseMixerBlocks where the LAST `num_attention_layers` use `SELF_ATTENTION` mode and the first `num_layers - num_attention_layers` use `LINEAR_MIXING` mode (matching the paper's "Linear, 4 TOP Attention layers" arrangement) → mean-pool over seq_len → final Dense classifier.

**Tech Stack:** C++17, hand-rolled Tensors, `Dense`/`LayerNorm`/`SwitchMoELayer`-style internal helpers. TDD discipline per `test-driven-development` skill.

**Distinct from existing repo:**
- vs `MlpMixerBlock`: MLP-Mixer uses TWO MLPs (token-mix + channel-mix), Sparse Mixer uses ONE linear mix (sequence-only) + MoE FFN.
- vs `SparseMoELayer`/`SwitchMoELayer`: those are Tokens-Choice top-k. Experts-Choice inverts the assignment direction.

**TDD plan (bite-sized):**

Task 1: LinearMixingSublayer header + constructor validation.
Task 2: LinearMixingSublayer forward (shape, finite, nonzero).
Task 3: LinearMixingSublayer backward (input FD + W_seq/W_h FD at machine precision).
Task 4: ExpertsChoiceMoE header + constructor validation.
Task 5: ExpertsChoiceMoE forward (shape, finite, nonzero, experts all fill capacity, EC signature).
Task 6: ExpertsChoiceMoE backward (input FD + router W FD + one expert W1 FD).
Task 7: SparseMixerBlock header + constructor validation (LINEAR_MIXING + DENSE, LINEAR_MIXING + MoE, SELF_ATTENTION + DENSE, SELF_ATTENTION + MoE).
Task 8: SparseMixerBlock forward shape + finiteness for all 4 configurations.
Task 9: SparseMixerBlock input FD (LINEAR_MIXING + DENSE).
Task 10: SparseMixerModel forward shape + training reduces loss.
Task 11: Register in nn.h umbrella + Makefile (build/test_sparse_mixer rule, run_tests echo). Verify 0 regressions on existing tests.
Task 12: Write summary doc bullet to EXPANSION_QUEUE.md ## Done.

**Failure mode mitigations (from systematic-debugging skill):**
- EC-MoE's `mean-pool over assigned experts` has a division-by-zero when a token is unassigned (no expert picked it) → guard with `if count == 0, output = 0` (the residual still flows through the block's first add).
- Hand-derive the LinearMixingSublayer reference at d_model=1, seq_len=2 with W_h=W_seq=I, b=0 → `y == x` exactly.
- Hand-derive EC-MoE for n_tokens=4, n_experts=2, cap=2 → expert 0 picks {top2 of probs[0,:]}, expert 1 picks {top2 of probs[1,:]} deterministically.
- Use random init with non-trivial scale 0.3 for all FD checks.