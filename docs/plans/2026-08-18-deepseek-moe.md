# DeepSeekMoE Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the canonical DeepSeekMoE architecture (fine-grained expert segmentation + shared expert isolation) — a `DeepSeekMoELayer` and `DeepSeekMoEModel` with full analytical backward and machine-precision gradient checks against centered FD.

**Architecture:** Two classes:
1. `DeepSeekMoELayer(d_model, d_expert, num_routed, num_shared, top_k_routed)` — DeepSeek-style MoE FFN. Two distinct expert populations:
   - **`num_routed` fine-grained routed experts** — each covers a *segment* of the intermediate dim of size `d_model / m` (segment_size × num_routed = d_model), so the routed experts act on small, specialized sub-features. Top-`top_k_routed` routing per token, soft-combined via sigmoid gating (sigmoid-on-score, the standard DeepSeek-V2/V3 choice).
   - **`num_shared` shared experts** — always-on, each is a full `d_model → d_expert → d_model` FFN that processes the entire input. No routing. Their contributions are summed (no gating weight).
   - Output: `y = sum_{i in top-k} gate_i · Expert_i(x) + sum_{j in shared} SharedExpert_j(x)`.
2. `DeepSeekMoEModel(input_dim, d_model, output_dim, num_layers, d_expert, num_routed, num_shared, top_k_routed)` — Per block: pre-LayerNorm → DeepSeekMoELayer → residual. Stack of N blocks → final LayerNorm → classifier.

**Tech Stack:** Existing infra — `Layer`/`Tensor`/`Dense` from core, reuses `LayerNorm` (`include/nn/layers/normalization/layer_norm.h`). Self-contained expert FFNs (each is two `Dense` layers with SiLU activation, built compositionally).

**Reference:** DeepSeek-AI 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (https://arxiv.org/abs/2401.06066). Two innovations:
1. **Fine-grained expert segmentation**: split each expert's intermediate dim into `m` smaller pieces, producing `mN` finer experts, selecting `mK` per token. Increases combinatorial flexibility without changing parameter count.
2. **Shared expert isolation**: carve out `K_s` experts that always fire (no routing) and capture common knowledge, freeing the routed experts to specialize.

The paper (Algorithm 1) uses sigmoid gating for the routed experts (not softmax) and the affinity `s_{i,t} = σ(W_g · u_t) · x_{i,t}` where `x_{i,t}` is the per-segment-gated input. We follow this convention.

---

## Background

The repo already ships:
- `MoELayer` and `SparseDispatcher` in `include/nn/layers/architectures/mixture_of_experts.{h,cpp}` — top-k MoE with softmax gating, each expert is a 1-layer `Dense` (input → input).
- `SparseMoE` / `JambaBlock` — use the existing MoE for MoE FFN sublayers.

What's missing: the **DeepSeekMoE** architecture unit — the two innovations (fine-grained segmentation + shared experts) are not present in any existing MoE layer in the repo. DeepSeek-V2/V3 (which scaled this idea) are the current state-of-the-art open-source MoE architectures, so adding this layer is high-value.

**Key design decisions** (locked, from the paper):

| Decision | Choice | Rationale |
|---|---|---|
| Gate nonlinearity | sigmoid (per expert) | DeepSeek-V2/V3 default; bounded outputs, no need for the softmax renormalization trick |
| Routing score | `s_i = σ(W_g · u_t)` (scalar per expert) | Standard DeepSeekMoE §2.1 |
| Top-k selection | k largest scores; outputs weighted by `s_i / Σ_{j∈top-k} s_j` (renormalized) | Paper Eq. 2; the renormalization keeps gradient magnitudes bounded |
| Segmented expert input | split input into `num_routed` equal segments of size `d_model / num_routed`; each expert receives only its segment | §2.1 fine-grained segmentation |
| Segmented expert output | each expert outputs a segment of size `d_model / num_routed`; results are concatenated (sum over the routed path) | Each expert is self-contained on its segment |
| Shared experts | full FFN `d_model → d_expert → d_model`, no gating, all fire | §2.1 shared expert isolation |
| Output | `y = routed_combined + shared_combined` (sum, since both are in `d_model` space) | The two paths are independent |
| Load-balancing auxiliary loss | `α · E · Σ_e (f_e · p_e)` (Skipwich et al. 2024 / Shazeer et al. 2017) | Standard for MoE; we expose `load_balance_loss()` but don't add to the main loss (callers can add) |

**Sanity check against the paper:** DeepSeekMoE-16B uses 64 routed experts (segmented from 4 base experts × 16 fine-grained), 2 shared experts, top-6 routing. We support arbitrary `num_routed`, `num_shared`, `top_k_routed` as long as `top_k_routed ≤ num_routed` and `d_model % num_routed == 0` (so the segments divide evenly).

**Edge case handled:** `num_shared == 0` is allowed (pure routed MoE). `num_routed == 0` is also allowed (pure shared experts, no routing). The general case is `num_routed > 0 && num_shared > 0`.

---

## Conventions

- `Layer` interface: `forward(T, d) → (T, d)`, `backward(grad_out, lr) → grad_in`
- (T, d_model) layout end-to-end; routed experts operate on `(T, segment_size)` (segment_size = d_model / num_routed)
- Pre-norm residuals (matches Griffin/Jamba/xLSTM_Block convention in this repo)
- Public members following the Griffin/Jamba style — `forward`/`backward` plus accessors for tests

## Files to create

- `include/nn/layers/architectures/deepseek_moe.h` — header for `DeepSeekMoELayer`, `DeepSeekMoEModel`
- `include/nn/layers/architectures/deepseek_moe.cpp` — implementation
- `tests/test_deepseek_moe.cpp` — focused test suite
- `docs/plans/2026-08-18-deepseek-moe.md` — this plan

## Files to modify

- `include/nn/nn.h` — add `#include "layers/architectures/deepseek_moe.h"` after the xLSTM Block include
- `Makefile` — add `build/test_deepseek_moe` compile rule, add `$(BUILD_DIR)/test_deepseek_moe` to the `tests:` target deps, and add `@echo "=== Running DeepSeekMoE Tests ===" && ./$(BUILD_DIR)/test_deepseek_moe` to the `run_tests` recipe

---

## Implementation tasks

Each task is sized for one TDD cycle: write failing test, watch fail, write minimal code, watch pass.

### Task 1: Header skeleton + test scaffold

**Step 1:** Write `tests/test_deepseek_moe.cpp` with the empty main + a `test_constructor_validates_dims` that fails (no header → compile error).

**Step 2:** Create `include/nn/layers/architectures/deepseek_moe.h` with class declarations but throwing `std::logic_error("not implemented")` for `forward/backward`. Confirm compile and the test reports "feature missing".

**Step 3:** Add `DeepSeekMoE` to `nn.h` umbrella and add Makefile rule (`build/test_deepseek_moe` compile rule, `tests:` deps line, `=== Running DeepSeekMoE Tests ===` echo in `run_tests`). Confirm `make build/test_deepseek_moe` succeeds (compiles, links).

**Validation:** `make build/test_deepseek_moe` produces a binary, the test fails (the `forward`/`backward` throws).

### Task 2: Forward shape + finiteness (routed + shared)

**Test:** `DeepSeekMoELayer(d_model=8, d_expert=16, num_routed=4, num_shared=2, top_k_routed=2).forward((T=3, d=8))` returns `(3, 8)`, all finite, nonzero.

**Implement:** Forward path:
```
x = input                                                  // (T, d_model)
# Shared expert path
for j in 0..num_shared-1:
    shared_out += FFN_j(x)                                  // (T, d_model), summed

# Routed expert path
gate_scores = sigmoid(W_g · x)                              // (T, num_routed)
top_k_indices, top_k_weights = top_k_with_renormalize(gate_scores, k=top_k_routed)
# For each token, for each selected expert i:
#   - segment_index = i (each expert routes to its own segment)
#   - segment_input = x[:, i*seg:(i+1)*seg]
#   - expert_out = Expert_i(segment_input)  // (T, seg)
#   - routed_out += top_k_weight[b, i] * expert_out at the segment positions
routed_out = segmented_expert_combine(x, top_k_indices, top_k_weights, num_routed)

out = shared_out + routed_out
```

Cache: `last_input`, `last_gate_scores`, `last_top_k_indices`, `last_top_k_weights`, `last_shared_intermediates`, `last_shared_outputs`, `last_routed_expert_outputs_per_segment`, `last_shared_out`, `last_routed_out`.

**Validation:** `1 passed`. Test 1: forward shape + finite + nonzero.

### Task 3: Input gradient via centered FD (full routed+shared)

**Test:** Centered finite-difference check on `L = sum(out²)` for input `x` (T=2, d=8). Rel_err tolerance < 1e-4.

**Implement:** Backward that propagates grad through:
1. Routed expert path: gate gradient + segment weights + per-expert FFN gradients (the W1, b1, W2, b2 of the selected experts)
2. Shared expert path: per-shared FFN gradients
3. Summation.

For the gated combination: `grad_x at segment i = sum_{t in selected_i} w_{t,i} · FFN_i'(...)` where `w_{t,i}` is the renormalized gate weight (and the gate score itself contributes `∂y/∂s_i = (FFN_i(x_seg_i) - mean) · top_k_renormal` terms — simplified analytically).

**Implementation hint:** Since segments are independent (each expert reads only its own segment, with no cross-segment influence), the gradient chain simplifies naturally:
- `grad_x[:, i*seg:(i+1)*seg] = ∑_{k where expert_i ∈ top-k} w_gated · Expert_i.backward(grad_segment_out)[grad_in]`
- Gate score gradient: `grad_s_i = w_renorm · dot(FFN_i_out, grad_segment_out)` (with the renormalization Jacobian).

**Validation:** `2 passed`. Test 2: input gradient FD rel_err < 1e-4.

### Task 4: W_g (gate weights) gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `W_g.weights` (num_routed × d_model). Rel_err < 1e-4.

**Implement:** In backward, also accumulate `grad_W_g[b, i] = sigmoid_deriv(gate_pre[b, i]) · renormal_grad` (the gradient of the renormalized gate weight w.r.t. the pre-activation, propagated to W_g via the linear W_g · x).

**Validation:** `3 passed`. Test 3: W_g gradient FD rel_err < 1e-4.

### Task 5: Routed expert W1 (first FFN weight) gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `experts_[0].W1.weights` (d_expert × segment_size). Rel_err < 1e-4.

**Implement:** Standard 2-layer FFN backward, gated: `grad_W1[b] = grad_hidden[b]^T · segment_input[b]` where `grad_hidden = grad_out_segment · W2^T ⊙ silu'(z)`.

**Validation:** `4 passed`. Test 4: routed expert W1 gradient FD rel_err < 1e-4.

### Task 6: Shared expert W1 gradient via FD

**Test:** Centered FD check on `L = sum(out²)` for `shared_experts_[0].W1.weights` (d_expert × d_model). Rel_err < 1e-4.

**Implement:** Standard 2-layer FFN backward (no gating). Mirrors the routed expert but with `d_model` input instead of `segment_size`.

**Validation:** `5 passed`. Test 5: shared expert W1 gradient FD rel_err < 1e-4.

### Task 7: Edge cases — pure-routed, pure-shared, zero top-k

**Test 7a:** `DeepSeekMoELayer(d_model=4, d_expert=8, num_routed=4, num_shared=0, top_k_routed=2)` — all routed, no shared. Forward shape + finite + nonzero. Input gradient FD rel_err < 1e-4.

**Test 7b:** `DeepSeekMoELayer(d_model=4, d_expert=8, num_routed=0, num_shared=2, top_k_routed=1)` — pure shared (no routing). Forward shape + finite + nonzero. Input gradient FD rel_err < 1e-4.

**Test 7c (disable path):** `DeepSeekMoELayer(d_model=4, d_expert=8, num_routed=4, num_shared=2, top_k_routed=4)` — top-k = num_routed, all experts fire. Test: in this mode, the routed-path output should be deterministic (gate weights still gated, but the partition is unique → easier to verify).

**Validation:** `8 passed`. Tests 6–8: edge cases.

### Task 8: Determinism — bit-exact forward with copied params

**Test:** Two `DeepSeekMoELayer` instances. Construct first, snapshot all parameters. Construct second, copy params into second. Forward both on the same input. Assert bit-exact (max abs diff = 0).

**Implementation hint:** Construct two layers separately, then `for each param p1, p2: p2->data[i] = p1->data[i];` for all parameter tensors. Need to enumerate all parameters across both routed and shared FFN, gating, and the routed-experts W1/W2/b1/b2.

**Validation:** `9 passed`. Test 9: determinism.

### Task 9: Training reduces loss

**Test:** Construct `DeepSeekMoELayer(d_model=4, d_expert=8, num_routed=2, num_shared=1, top_k_routed=1)`. 50 SGD steps with lr=1e-2, MSE loss against target `y = 0.5 * x`. Assert loss strictly decreases by > 30% (or, if loss is noisy, the final loss is < 0.7 × initial loss).

**Implementation:** Standard loop: `zero_grad → forward → mse_grad → backward → update_weights`. Use `Layer::forward` / `backward` per the standard interface.

**Validation:** `10 passed`. Test 10: training reduces loss.

### Task 10: DeepSeekMoEModel (stack) — forward shape + param/grad count

**Test:** `DeepSeekMoEModel(input_dim=3, d_model=4, output_dim=2, num_layers=2, d_expert=8, num_routed=2, num_shared=1, top_k_routed=1).forward((T=2, in=3))` returns `(2, 2)`, finite, nonzero. Assert `parameters().size()` and `gradients().size()` are non-empty and match.

**Implement:** Embed (Dense input_dim → d_model) → stack of N blocks (each: pre-LayerNorm → DeepSeekMoELayer → residual) → final LayerNorm → classifier (Dense d_model → output_dim).

**Validation:** `11 passed`. Test 11: model forward shape + finite.

### Task 11: DeepSeekMoEModel training reduces loss

**Test:** 80 SGD steps with lr=1e-2 on `DeepSeekMoEModel(input_dim=3, d_model=4, output_dim=2, num_layers=2, ...)`. Assert loss decreases by > 30%.

**Validation:** `12 passed`. Test 12: model training reduces loss.

### Task 12: Load-balance loss + accessors

**Test 12a:** `load_balance_loss()` returns a non-negative scalar; increases as more tokens share an expert; stays finite.

**Test 12b:** `num_routed()`, `num_shared()`, `top_k_routed()`, `d_model()`, `d_expert()` accessors return the values passed to the constructor.

**Validation:** `14 passed`. Tests 13–14: aux loss + accessors.

### Task 13: Mutate-stub regression test

**Test:** Disable the gate score gradient (stub it out). The W_g gradient FD check should fail. This proves the W_g gradient test is non-vacuous.

**Validation:** `15 passed`. Test 15: mutation catches missing gate-gradient path.

### Task 14: Run full test suite + no regressions

**Step 1:** `make build/test_deepseek_moe` — confirm it compiles.
**Step 2:** `make build/test_deepseek_moe && ./build/test_deepseek_moe` — confirm all 15 tests pass.
**Step 3:** `make run_tests` — confirm the existing 130+ test suite still runs (no regressions, ignoring the pre-existing deferred failures in NOT_FIXED.md).

**Validation:** All 15 tests pass. Existing tests still pass.

### Task 15: Cleanup pass + commit

**Step 1:** Look for stray debug artifacts in `tests/` and root (`debug_*`, `*.bak`, `ref_*`). Delete any that exist.
**Step 2:** `git add include/nn/layers/architectures/deepseek_moe.h include/nn/layers/architectures/deepseek_moe.cpp tests/test_deepseek_moe.cpp include/nn/nn.h Makefile docs/plans/2026-08-18-deepseek-moe.md`
**Step 3:** Commit: `feat(architectures): DeepSeekMoE — fine-grained expert segmentation + shared expert isolation (15/15 tests)`
**Step 4:** Push with `git push origin master`.

**Step 5:** Update `EXPANSION_QUEUE.md` — move the DeepSeekMoE entry from `## Ideas` to `## Done` with a summary of test results.

---

## Risks & pitfalls

- **Numerical stability of sigmoid gate scores.** Use `expit`-style helper with stable `_x ≥ 0` branching. Should already be in `nn/optimizers/...` or `core/numerical_stability.h` — check before adding.
- **Top-k selection is non-differentiable.** The choice of which experts fire is a discrete operation; we use straight-through routing (gradient flows through the gate weights but not through the index selection). Make this explicit in the header.
- **Segment bookkeeping.** Off-by-one in `i*seg:(i+1)*seg` if `d_model % num_routed != 0`. Enforce in constructor.
- **Gate gradient chain.** The renormalization `w_i = s_i / Σ_{j∈top-k} s_j` introduces a coupling term: `∂w_i/∂s_j = (δ_{ij} Σ_{j'} s_{j'} - s_i) / (Σ_{j'} s_{j'})²` for `j ∈ top-k`, else 0. Easy to forget. Verify with the FD check.
- **centroid test fixture convention.** All test tensors use `rand_tensor(T, D, seed, scale=0.3)` style helpers — match the convention used by other tests in this repo.
- **Memory of the routed path.** With `num_routed` 2-layer FFNs (each with d_expert × seg × 2 weights + biases), the parameter count is `num_routed * (2 * d_expert * seg + 2 * d_expert)`. For `(d_model=8, d_expert=16, num_routed=4, seg=2)`: 4 × (2·16·2 + 2·16) = 384 params for routed. Plus shared: `num_shared * (2 · d_expert · d_model + 2 · d_expert)`. Total is small for tests.

---

## Definition of done

- [ ] `include/nn/layers/architectures/deepseek_moe.{h,cpp}` exists, with `DeepSeekMoELayer` and `DeepSeekMoEModel` classes
- [ ] `tests/test_deepseek_moe.cpp` exists with 15+ focused tests
- [ ] `make build/test_deepseek_moe` compiles
- [ ] `./build/test_deepseek_moe` runs with 15+ passing tests
- [ ] `make run_tests` shows no new regressions (existing 130+ tests still pass)
- [ ] `include/nn/nn.h` includes the new header
- [ ] `Makefile` has the build rule, the `tests:` deps line, and the echo line in `run_tests`
- [ ] `EXPANSION_QUEUE.md` has the DeepSeekMoE entry moved from `## Ideas` to `## Done`
- [ ] Plan saved at `docs/plans/2026-08-18-deepseek-moe.md`
- [ ] Changes committed and pushed to `origin/master`
