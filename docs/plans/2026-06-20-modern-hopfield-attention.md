# Modern Hopfield Attention — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement a Modern Hopfield Network attention layer (Ramsauer et al. 2020) that generalizes transformer attention as a Hopfield retrieval over a learnable set of "patterns" (memories). The layer is a `Layer` that fits into the existing attention stack convention.

**Architecture:**
- **HopfieldAttention**: Single-layer modern Hopfield retrieval. The layer holds a learnable pattern memory `P ∈ R^{m × d}` (m stored patterns of dim d) plus learnable `W_q, W_o` projections. Forward: `Q = X @ W_q^T`, `attn = softmax(β * Q @ P^T)`, `out = (attn @ P) @ W_o^T`. Inverse temperature β is a learnable parameter (per-head share; one scalar per layer is the common convention).
- **HopfieldBlock**: pre-LN → HopfieldAttention → residual → pre-LN → GELU FFN → residual (matches `ConvAttentionBlock` / `GATBlock` / etc.).
- **HopfieldModel**: stack of HopfieldBlocks + per-token classifier.

**Tech Stack:** C++17, matches existing layer/attention conventions. Pure CPU, no external deps. Tested with numerical gradient checks against analytical backward.

**Reference paper:**
- Ramsauer, H. et al. (2020). "Hopfield Networks is All You Need." https://arxiv.org/abs/2008.02217
- The key insight: `softmax(β * X @ P^T) @ P` is mathematically equivalent to a Hopfield network energy-minimization retrieval over stored patterns P. With β = 1/√d (or β learnable), this recovers the standard transformer attention.

**Files:**
- Create: `include/nn/layers/attention/hopfield.h`
- Create: `include/nn/layers/attention/hopfield.cpp`
- Create: `tests/test_hopfield.cpp`
- Modify: `include/nn/nn.h` (add `#include "layers/attention/hopfield.h"`)
- Modify: `Makefile` (add build target, test target, and add to `tests`/`run_tests`)
- Modify: `EXPANSION_QUEUE.md` (move entry to `## Done` after successful tests)

**Verification:**
- All 20+ tests pass at machine precision (rel_err < 1e-4 for input/weight gradients, < 1e-2 for bias/last-row gradients, finite outputs, loss reduction on training step).
- `make tests` and `make run_tests` both pass.
