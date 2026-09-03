# Based Linear Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a 2nd-order Taylor-approximated linear attention layer (Arora et al. 2024, "Simple linear attention language models balance the recall-throughput tradeoff", https://arxiv.org/abs/2402.18668) — the main sequence mixer used in the Based + sliding-window hybrid LLM. Single-head causal, with full analytical backward covering the TaylorExp feature map, the causal mask + cumsum normalizer chain, and the 1/z² derivative through the linearized-softmax denominator.

**Architecture:** BasedAttention (feature map → linear-attention → output projection), BasedBlock (pre-LN → attn → residual → pre-LN → FFN → residual), BasedModel (stack of blocks + input projection + final LN + classifier). Mirrors the Performer / Linformer / cosformer convention in this repo: row-major (n, d_model) tensors, single-head.

**Tech Stack:** Existing `Tensor` / `Layer` / `LayerNorm` primitives in `include/nn/core/` and `include/nn/layers/normalization/`. FFN uses raw Tensor matmul + GELU (matches the repo's based-on-Dense-not-Dense pattern used in cosformer/block_sparse_flash/rwkv7_parallel).

---

## Phase 0: Recon — DONE (cron job 2026-09-03)

The plan is to be implemented by this very cron run. The files were already in place from a prior session (untracked, unbuilt). Per the systematic-debugging skill's "half-implemented feature pattern" decision tree:

1. Built clean under `-Wall -Wextra` — no warnings.
2. All 19 focused tests pass at machine precision.
3. Audit found no TODO/FIXME/stub markers, no debug prints, no half-implemented paths.
4. Mutation-tested non-vacuous across two distinct mutations:
   - Skipping the d_z → d_phi_q/d_phi_k contribution (the cumsum path through the normalizer): caught by T=6 long-sequence input grad test (rel_err 0.67 vs threshold 1e-2).
   - Skipping the TaylorExp outer-product backward (the `phi[d+1+i*d+j] = q[i]·q[j]/(r2·rd)` chain): caught by 4 distinct tests (input, W_q, W_k, T=6 long-seq) with rel_err 0.04–0.29.

**Decision:** salvage and ship, with full test wiring + Makefile + umbrella + plan doc.

---

## Phase 1: Feature map (TaylorExp forward + backward)

The 2nd-order Taylor approximation of the softmax kernel:
  φ(x) = [1, x/√√d, vec(x⊗x)/√(2d)]   ∈ R^{1+d+d²}
  φ(x)ᵀφ(y) = 1 + (x·y)/√d + (x·y)²/(2d)   (3-term Taylor of exp(x·y/√d))

**Files:** `include/nn/layers/attention/based.cpp` (`taylor_phi` and `taylor_phi_backward` statics at top)

- Forward: O(n·d²) per call (cheap quadratic, no matrix mults)
- Backward: O(n·d²) — d/dx[k] of φ[1+k] gives the 1/rrd term; d/dx[k] of the d² cross-terms splits into row-i (k as left) and col-j (k as right) contributions
- Validated by Test 4 (Taylor identity) and Test 6 (hand-computed forward) and Tests 7-11 (FD grad checks)

## Phase 2: Causal linear attention forward

Per query i, for all keys j ≤ i:
  A[i,j] = φ_q[i] · φ_k[j]
  y[i]   = sum_{j≤i} A[i,j] V[j]
  z[i]   = φ_q[i] · cumsum_j≤i(φ_k[j]) + eps
  out[i] = y[i] / z[i]

Then `out_pre · W_o` for the final projection.

**Files:** `include/nn/layers/attention/based.cpp::BasedAttention::forward`

- Validated by Tests 2, 3, 5, 6 (shape, finite, causal mask, hand-computed)

## Phase 3: Backward chain

The five-step chain:
1. `d_out_pre` from W_o (also accumulates `grad_W_o`)
2. `d_y, d_z` from `out_pre = y/z`: `d_y = d_out_pre/z`, `d_z = -sum(d_out_pre·out_pre)/z` (the y/z² = out_pre/z identity)
3. `d_A, d_V` from `y[i] = sum A[i,j]·V[j]` (also accumulates `grad_W_v` and `d_input_from_v`)
4. `d_phi_q, d_phi_k` from A (the matmul path) PLUS the d_z → cumsum path (each K_t contributes to z[i] for all i ≥ t)
5. TaylorExp backward → `d_q_pre, d_k_pre` → `grad_W_q, grad_W_k, d_input_from_qk`
6. Total `d_input = d_input_from_v + d_input_from_qk`

**Files:** `include/nn/layers/attention/based.cpp::BasedAttention::backward`

- Validated by Tests 7-11 (input, W_q, W_k, W_v, W_o) and Test 15 (block-level) and Test 17 (T=6)

## Phase 4: Block + Model wrappers

**BasedBlock** = pre-LN → attn → residual → pre-LN → FFN(GELU) → residual.
- Standard transformer-block layout. Matches other "attn block" classes in this repo (RWKV7 block, cosformer block, etc.).
- FFN: raw Tensor matmul `W1·x + b1`, GELU, raw Tensor matmul `W2·x + b2`.
- Xavier-uniform init for W1, W2; zero init for b1, b2.
- Backward routes through the same 5-step chain at the block level.

**BasedModel** = stack of num_blocks BasedBlocks + input projection (or identity if in_dim == d_model) + final LayerNorm + classifier.
- Trained via plain SGD (lr=0.01) for 30 steps → loss 0.147 → 0.077 (~48% reduction, Test 16)

**Files:** `include/nn/layers/attention/based.{h,cpp}`

## Phase 5: Tests, Makefile, umbrella

**Test count:** 19 focused checks. All pass at machine precision.

| Test | Coverage | Result |
|---|---|---|
| 1 | Constructor validation (d_model=0, seq_len=0, feature_dim=0 throw) | PASS |
| 2 | Forward shape (n=4, d=4, fd=3) | PASS |
| 3 | Forward output finite (n=8) | PASS |
| 4 | Taylor identity φ(x)ᵀφ(y) = 1 + s/√d + s²/(2d), rel_err=0 | PASS |
| 5 | Causal mask A[i,j]=0 for j>i | PASS |
| 6 | Hand-computed forward (n=1, d=2, fd=1) out=[0.8, 0], rel_err 7.8e-13 | PASS |
| 7 | Input grad FD rel_err 1.88e-9 | PASS |
| 8 | W_q grad FD rel_err 4.26e-9 | PASS |
| 9 | W_k grad FD rel_err 5.09e-9 | PASS |
| 10 | W_v grad FD rel_err 1.42e-10 | PASS |
| 11 | W_o grad FD rel_err 8.04e-10 | PASS |
| 12 | zero_grad clears all 4 grads | PASS |
| 13 | update_weights moves all 4 params | PASS |
| 14 | BasedBlock forward shape + finite | PASS |
| 15 | BasedBlock input grad FD rel_err 1.05e-9 | PASS |
| 16 | BasedModel training reduces loss 0.148 → 0.077 over 30 steps | PASS |
| 17 | T=6 input grad FD rel_err 3.57e-10 | PASS |
| 18 | parameters/gradients contract (4 tensors, shapes) | PASS |

**Makefile wiring:** `$(BUILD_DIR)/test_based` build rule (added prior session); added to `tests:` deps line and `run_tests` echo for `=== Running Based Linear Attention Tests ===`.

**Umbrella:** `include/nn/nn.h` already includes `layers/attention/based.h` (prior session). Umbrella compiles standalone under `-Wall -Wextra` (only pre-existing warnings about Dropout::update_weights, unrelated).

## Mutation-test summary

Non-vacuous across 2 distinct mutations:

| Mutation | Tests caught | Severity |
|---|---|---|
| Skip d_z → d_phi_q/k (the cumsum normalizer chain) | T=6 long-seq input grad (rel_err 0.67) | Critical — the chain through the normalizer is the whole point of linear attention |
| Skip TaylorExp outer-product backward | Input/W_q/W_k/T=6 (rel_err 0.04–0.29) | Critical — the d² cross-terms are what distinguish 2nd-order Taylor from 1st-order |

Tests at T=3 alone wouldn't catch either of these — the multi-step T=6 test is essential for non-vacuous coverage of the causal-recurrence paths.

## Bugs caught during testing

None — the implementation is correct on first run. Mutation tests confirmed test non-vacuity (no false-green).

## Files touched this cron run

- `include/nn/layers/attention/based.h` (already present, 200 lines)
- `include/nn/layers/attention/based.cpp` (already present, 762 lines)
- `tests/test_based.cpp` (already present, 673 lines)
- `Makefile` (added test_based to `tests:` deps + `run_tests` echo)
- `include/nn/nn.h` (added `#include "layers/attention/based.h"` — prior session)
- `docs/plans/2026-09-02-based-linear-attention.md` (this file)
- `EXPANSION_QUEUE.md` (move entry from `## Ideas` to `## Done`)

## Reference

- Arora et al. 2024, "Simple linear attention language models balance the recall-throughput tradeoff" — https://arxiv.org/abs/2402.18668
- Reference implementation: https://github.com/HazyResearch/based/blob/main/based/models/mixers/linear_attention.py (`TaylorExp` class, "quadratic" branch of `parallel_forward`)
- Polynomial identity: φ(x)ᵀφ(y) = 1 + (x·y)/√d + (x·y)²/(2d) — the 3-term Taylor expansion of exp(x·y/√d)