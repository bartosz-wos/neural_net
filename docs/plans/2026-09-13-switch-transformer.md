# Switch Transformer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a faithful C++ implementation of Fedus, Zoph, Shazeer 2022 "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (https://arxiv.org/abs/2101.03961) as `SwitchTransformerLayer` + `SwitchTransformerBlock` + `SwitchTransformerModel`, focusing on the three distinctive Switch-Transformer ingredients that the existing `SparseMoELayer` does NOT have: (1) **hard top-1 routing**, (2) **expert capacity factor with overflow-token dropping**, (3) **router z-loss** (paper Eq. 5) for stability, plus the canonical differentiable load-balancing loss (paper Eq. 4).

**Architecture:** A Switch FFN block has the form `LN → SwitchMoE(x) → residual`, where the MoE routes each token to its single top-1 expert, applies the capacity cap, and returns the expert output weighted by the router probability. Backward differentiates through the load-balancing auxiliary loss and z-loss so the router learns to balance. The full `SwitchTransformerModel` stacks N blocks with a pre-block LN, then an unnormalized classifier.

**Tech Stack:** Existing `Dense`, `LayerNorm`, `Tensor`. New `switch_transformer.{h,cpp}` under `include/nn/layers/architectures/`.

---

## Distinctive vs existing

The repo already has `SparseMoELayer` (Shazeer 2017, top-k routing) and `mixture_of_experts.cpp`. What's NEW here:

1. **Top-1 only (Switch routing).** `SparseMoELayer` supports `k≥1`; Switch Transformer fixes k=1 (paper §1.1 "simplifies the MoE routing algorithm to compute a single expert per token").
2. **Expert capacity factor** (paper §3.1, Eq. 2): each expert can process at most `capacity = ceil(tokens / num_experts) · capacity_factor` tokens; overflow tokens are dropped (output set to 0, gradient to 0 for those expert positions). This is the *only* paper that introduces explicit token dropping via a scalar hyperparameter.
3. **Router z-loss** (paper Eq. 5): `L_z = (1/B) · Σ_b log(Σ_e exp(z_e)²)` — penalizes large router logits, the paper's key stability fix that lets them train at scale. We expose this as a separate accessor (`get_z_loss()`) the user can add to the loss.
4. **SwitchFFNBlock** that wires LN → SwitchMoE → residual with the paper's expert FFN.

Everything else (Xavier router init, He expert init, 2-layer ReLU FFN, raw Tensor params, batched forward) follows the `SparseMoELayer` pattern.

---

## Task 1: SwitchMoELayer — RED 1 (constructor validation + accessors)

**Files:**
- Create: `include/nn/layers/architectures/switch_transformer.h`
- Create: `tests/test_switch_transformer.cpp`

**Step 1:** In `switch_transformer.h`, declare:

```cpp
class SwitchMoELayer : public Layer {
public:
    SwitchMoELayer(size_t d_model, size_t num_experts,
                   size_t expert_hidden = 0,
                   double capacity_factor = 1.25,
                   double aux_loss_coef = 0.01,
                   double z_loss_coef = 0.001);
    size_t d_model() const;
    size_t num_experts() const;
    size_t expert_hidden() const;
    double capacity_factor() const;
    double aux_loss_coef() const;
    double z_loss_coef() const;
    // ... forward/backward/parameters/gradients/zero_grad/update_weights/name ...
private:
    size_t d_model_, num_experts_, expert_hidden_;
    double capacity_factor_, aux_loss_coef_, z_loss_coef_;
    // Router + experts (same shape convention as SparseMoELayer)
    Tensor W_router_, b_router_, grad_W_router_, grad_b_router_;
    std::vector<Tensor> W1_, W2_, b1_, b2_, gW1_, gW2_, gb1_, gb2_;
    // Forward caches
    Tensor input_, gate_logits_, gate_probs_, top1_idx_;
    Tensor dispatch_mask_; // (tokens, num_experts) — 1 iff (b,e) routed AND under capacity, else 0
    Tensor expert_dispatch_; // (tokens,) — for each token, which expert it was routed to
    Tensor token_dropped_; // (tokens,) — 1 if the token's expert was over capacity, else 0
    Tensor expert_h_pre_, expert_h_act_, expert_outputs_; // per-expert (capacity_or_all, d_model/hidden)
    double load_balance_loss_;
    double z_loss_;
    std::vector<double> dispatch_frac_;
    std::vector<double> mean_router_prob_;
};
```

**Step 2:** Create `tests/test_switch_transformer.cpp` with Test 1:
- 5-case constructor validation: `d_model=0` throws, `num_experts=0` throws, `capacity_factor ≤ 0` throws, `aux_loss_coef < 0` throws, `z_loss_coef < 0` throws.
- Valid construct with d_model=8, num_experts=4, expert_hidden=16, capacity_factor=1.5 → accessors return input values; `name() == "SwitchMoELayer"`.

**Step 3:** Verify RED — `g++ -c tests/test_switch_transformer.cpp -Iinclude -std=c++17` should fail because the header doesn't exist yet.

**Step 4:** Add the header file with the declaration above (forward methods throw `std::logic_error("not implemented")`).

**Step 5:** Verify RED → GREEN by adding the implementation stubs that just throw on forward/backward but satisfy the constructor + accessors.

**Step 6:** Commit `feat(architectures): SwitchMoELayer header + constructor validation`.

---

## Task 2: SwitchMoELayer — RED 2 (forward shape, finiteness, top-1 routing)

**Step 1:** Add to test file:
- Test 2: forward shape `(B=6, d_model=8) → (B, d_model)`, finite, no NaN/Inf.
- Test 3: top-1 routing signature — `last_top1_indices()` (a new accessor `top1_indices()` we add) has exactly one expert per row (it's a single index per row, not a vector).
- Test 4: with `W_router` zeroed and `b_router` set so expert e is dominant for row b, that (b, e) is the top-1 for row b.

**Step 2:** Verify RED.

**Step 3:** Implement `forward()` to compute `gate_logits = X @ W_router^T + b_router`, find `top1_idx[b] = argmax_e gate_logits[b, e]`, dispatch each token to its top-1 expert (skipping tokens that overflow capacity), call `expert(b) = W2 · ReLU(W1 · x + b1) + b2`, accumulate `output[b] = router_prob[b, top1] * expert(top1)[b]` if not dropped, else 0. Cache everything for backward.

**Step 4:** Verify GREEN.

**Step 5:** Commit `feat(architectures): SwitchMoELayer forward (top-1 routing)`.

---

## Task 3: SwitchMoELayer — RED 3 (capacity factor dropping)

**Step 1:** Add:
- Test 5: with `capacity_factor = 1.0` and num_experts=4, B=8 tokens → capacity_per_expert = ceil(8/4)·1.0 = 2. Force all 8 tokens to route to expert 0 (set `b_router[0] = +100`, others = -100). Then exactly `2` of the 8 output rows should be the expert's output, and the other 6 should be exactly zero (token dropped due to capacity overflow).
- Test 6: with `capacity_factor = 2.0`, same config → capacity_per_expert = ceil(8/4)·2.0 = 4. Then 4 of 8 tokens get expert output, 4 are zero.

**Step 2:** Verify RED.

**Step 3:** Implement capacity cap:
```cpp
size_t cap_per = static_cast<size_t>(std::ceil(static_cast<double>(B) / num_experts_ * capacity_factor_));
// Per-expert counter, dispatch_mask[b, e] = 1 iff token b's top1==e AND counter[e] < cap
```
Make the per-expert cache size `capacity` so the cached pre-acts are over only the dispatched tokens (not all B), and fill output from the cap-limited cache.

**Step 4:** Verify GREEN.

**Step 5:** Commit `feat(architectures): SwitchMoELayer capacity factor + token dropping`.

---

## Task 4: SwitchMoELayer — RED 4 (aux loss + z-loss formulas)

**Step 1:** Add:
- Test 7: `get_load_balance_loss()` matches the paper Eq. 4 formula: `aux_loss_coef · num_experts · Σ_e f_e · p_e` where `f_e` = fraction of tokens routed to e (NOT counting dropped tokens; same convention as paper) and `p_e` = mean router probability for e.
- Test 8: `get_z_loss()` matches the paper Eq. 5 formula: `z_loss_coef · (1/B) · Σ_b log(Σ_e exp(gate_logits[b, e])²)`. Verify on a hand-computed case: B=2, num_experts=3, gate_logits=[[1,2,3],[0,0,0]] → Σ_e exp(1)² = e², Σ_e exp(2)² = e⁴, Σ_e exp(3)² = e⁶ → row-0 inner = e²+e⁴+e⁶ → log = log(e²+e⁴+e⁶); row-1 inner = 3 → log(3); mean = (log(e²+e⁴+e⁶) + log(3))/2; multiply by z_loss_coef.

**Step 2:** Verify RED.

**Step 3:** Implement both losses:
```cpp
// Aux loss (Eq. 4): alpha · N · sum_e (f_e · p_e)
double alpha = aux_loss_coef_;
double N = static_cast<double>(num_experts_);
double sum_f_p = 0.0;
for (size_t e = 0; e < num_experts_; ++e)
    sum_f_p += dispatch_frac_[e] * mean_router_prob_[e];
load_balance_loss_ = alpha * N * sum_f_p;

// Z-loss (Eq. 5): z_coef · (1/B) · Σ_b log(Σ_e exp(logit_e)^2)
double total = 0.0;
for (size_t b = 0; b < B; ++b) {
    double inner = 0.0;
    for (size_t e = 0; e < num_experts_; ++e) {
        double v = std::exp(gate_logits_(b, e));
        inner += v * v;
    }
    total += std::log(inner + 1e-30);  // numerical floor
}
z_loss_ = z_loss_coef_ * (total / B);
```

**Step 4:** Verify GREEN.

**Step 5:** Commit `feat(architectures): SwitchMoELayer aux + z losses`.

---

## Task 5: SwitchMoELayer — RED 5 (backward + parameter gradients)

**Step 1:** Add:
- Test 9: FD input gradient at rel_err < 1e-4 with B=4, d_model=4, num_experts=2 (small config for tractable FD).
- Test 10: FD `W_router` gradient at rel_err < 1e-4 (random non-uniform init).
- Test 11: FD `W1[0]` and `W2[0]` gradients at rel_err < 1e-4 for the EXPERT that received at least one token (use random init + small num_experts so dispatch is non-degenerate).
- Test 12: With all tokens dropped (capacity_factor=0.5, num_experts=2, B=4, all 4 tokens forced to expert 0 → cap_per=ceil(4/2)·0.5=1, so 3 tokens dropped): `output` is all zeros, `grad_input` is all zeros.

**Step 2:** Verify RED.

**Step 3:** Implement backward:
- `grad_output (B, d_model)`: for each (b, e) where dispatch_mask[b,e]=1, compute `grad_expert_out[b, e, :] = router_prob[b, e] * grad_output[b, :]`. Otherwise 0.
- Per expert: standard 2-layer FFN backward through `W1/W2/b1/b2`.
- Router gradient: for each non-dropped (b, e): `d_router_prob[b, e] = Σ_j grad_output[b, j] · expert_e_out[b, j]`. Note `router_prob[b, e] = softmax_e(gate_logits[b, :])` even though we only selected top-1 (paper §2.2: "we use the same softmax distribution for the gate, not just the top-1 hard choice"). So `d_logits[b, e] = router_prob[b, e] · (d_router_prob[b, e] − Σ_e' router_prob[b, e'] · d_router_prob[b, e'])`. The Z-loss adds `d_logits[b, e] += z_loss_coef · (2/B) · softmax(gate_logits[b, e])` (chain: `log Σ exp² → (2·exp(2·z_e)) / Σ exp²`).
- Input gradient: `d_input += d_logits @ W_router`.

**Step 4:** Verify GREEN. Add `gate_probs_` cache (full softmax, not just top-1) for the router backward.

**Step 5:** Commit `feat(architectures): SwitchMoELayer backward (full chain)`.

---

## Task 6: SwitchMoELayer — RED 6 (update_weights + zero_grad + training reduces loss)

**Step 1:** Add:
- Test 13: `update_weights(0.05)` moves all `W_router`, `b_router`, every expert's `W1`/`W2`/`b1`/`b2` by > 1e-10 from initial.
- Test 14: `zero_grad()` clears every grad tensor (sum < 1e-15).
- Test 15: `parameters()` returns `2 + 4·num_experts` tensors, `gradients()` returns the same count with shape-matched pairs.
- Test 16: end-to-end SGD training on a 4-token, d_model=4, num_experts=2 regression task reduces MSE loss > 30% over 50 steps.

**Step 2:** Verify RED.

**Step 3:** Implement `update_weights` as in-place `param -= lr · grad` for each tensor; `zero_grad` zeros every grad.

**Step 4:** Verify GREEN.

**Step 5:** Commit `feat(architectures): SwitchMoELayer update/zero/training`.

---

## Task 7: SwitchTransformerBlock — pre-LN + SwitchMoE + residual

**Files:**
- Append to `switch_transformer.h`: `SwitchTransformerBlock(size_t d_model, size_t num_experts, size_t expert_hidden = 0, double capacity_factor = 1.25, double aux_loss_coef = 0.01, double z_loss_coef = 0.001)`.

**Step 1:** Test 17: block forward shape `(B, d_model) → (B, d_model)`, finite.
- Test 18: identity-style signature — with `W_router` zeroed + `b_router = -100` (uniform low routing → aux loss dominates → gradient flow still valid). Actually a cleaner signature: block input FD at rel_err < 1e-4 (B=4, d_model=4, num_experts=2).
- Test 19: `get_aux_loss()` and `get_z_loss()` are accessible via the block.
- Test 20: end-to-end block training reduces loss > 20% over 30 SGD steps.

**Step 2:** Implement as `LN → SwitchMoE → residual`. Use the same `LayerNorm` already in `include/nn/layers/normalization/layer_norm.h` (the standard one used by every other block in the repo).

**Step 3:** Verify GREEN.

**Step 4:** Commit `feat(architectures): SwitchTransformerBlock`.

---

## Task 8: SwitchTransformerModel — full stack

**Files:**
- Append to header.

**Step 1:** Test 21: `SwitchTransformerModel(input_dim=4, d_model=8, num_experts=4, output_dim=3, num_blocks=2)` forward shape `(B, 4) → (B, 3)`, finite.
- Test 22: model training reduces loss > 30% over 50 steps on a 4-class toy task.
- Test 23: `get_aux_loss()` and `get_z_loss()` of the model return the SUM of the per-block losses (the canonical convention: sum across blocks).
- Test 24: mutation test — zeroing `W_router` of any block changes the model output by max diff > 1e-6 (proves the layer is wired in, not a no-op).

**Step 2:** Implement as `Dense(input_dim → d_model) → LayerNorm → N × SwitchTransformerBlock → Dense(d_model → output_dim)`.

**Step 3:** Verify GREEN.

**Step 4:** Commit `feat(architectures): SwitchTransformerModel`.

---

## Task 9: Wire-up + umbrella + Makefile + plan

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/architectures/switch_transformer.h"` after `sparse_moe.h`.
- Modify: `Makefile` — add `$(BUILD_DIR)/test_switch_transformer` rule, add to `tests:` deps list, add echo line in `run_tests`.
- Modify: `EXPANSION_QUEUE.md` — move "Switch Transformer" bullet to `## Done` with a summary.

**Step 1:** Run `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` to verify the umbrella compiles standalone.
**Step 2:** Run `make build/test_switch_transformer && ./build/test_switch_transformer` → all checks pass.
**Step 3:** Run `make tests` to verify no regressions in other suites.
**Step 4:** Commit `chore: register SwitchTransformer in umbrella + Makefile`.
**Step 5:** Commit `docs: plan for Switch Transformer (Fedus et al. 2022)`.
**Step 6:** Push to master.

---

## Acceptance criteria

- All 24 tests pass at machine precision or rel_err < 1e-4.
- End-to-end training reduces loss > 30% (model) and > 20% (block).
- Umbrella `nn.h` compiles standalone.
- No regressions in existing test suites.
