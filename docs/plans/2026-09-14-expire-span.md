# Expire-Span Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a faithful C++ implementation of Sukhbaatar, Ju, Poff, Roller, Szlam, Weston, Lewis 2021 "Not All Memories are Created Equal: A Learnable, Per-Layer Memory Budget" (https://arxiv.org/abs/2105.11850 / EMNLP 2021) — Expire-Span attention — as `ExpireSpanAttention` + `ExpireSpanBlock` + `ExpireSpanModel`. Expire-Span learns a per-token scalar "span" `s_i ∈ [0, 1]` (via a small Dense head on the input) that controls how many future tokens this token should remain accessible to. At query position t, only the keys whose `effective_age = t - i ≤ s_i · S_max` (paper §2.2) are KEPT in the softmax; the rest get the additive mask `-1e9` (paper §3.2 / Eq. 8). This is the "learnable per-layer memory budget" — a sparse attention pattern whose ROUTING (which keys survive) is data-dependent.

**Architecture:** Each layer attaches a small MLP `s_i = σ(W_s · x_i + b_s) ∈ [0, 1]` to the input. The cache stores `last_span[i]`. At forward, for each query i, the key j's effective age is `(i - j)`; it survives iff `(i - j) ≤ last_span[j] · S_max`. The additive mask is `-1e9` for dropped keys, `0` for survivors. Backward propagates through (a) the softmax chain, (b) the span-head Dense (input span-head W_s, b_s, AND the cross-impact: `ds_j` affects `dW_s[j]` for the token j whose span we computed), and (c) Q/K/V/O projections. The span is recomputed every layer (it's a per-layer budget, not a per-token lifetime).

**Tech Stack:** Existing `Dense`, `LayerNorm`, `Tensor`, `softmax`. New `expire_span.{h,cpp}` under `include/nn/layers/attention/`.

**Design choices:**
- **Why Expire-Span** for the "sparse attention with token coalescing" slot: it's the cleanest tractable variant of "learned per-layer memory budget" — the routing (which keys survive) is computed from the input itself (a small Dense), the keep/drop decision is a single threshold (no clustering, no Gumbel, no top-K across batch), and the full hand-derived BPTT chain is well-defined. CoLT5 needs heavy/light branches plus a separate learned router (more bookkeeping). MagicDraft is bespoke. Expire-Span is the canonical "learn to forget" sparse attention paper.
- **Why a learned span head per layer** (not per-block stack): the paper's per-layer budget is the whole point. A block pattern (LN → expire → residual → LN → FFN → residual) is exposed via `ExpireSpanBlock`. The Model wraps N blocks.
- **`s_i` shape and clamping:** raw `z = W_s · x_i + b_s`; `s_i = clamp(σ(z), s_min, 1.0)` with `s_min = 1e-6` to avoid zero spans (paper §3.2 — they also clamp to avoid log(0)). We store `last_z[i]`, `last_span[i]`, `last_survive_mask[i, j]`. Backward uses the span clamped at forward time but propagates through the un-clamped sigmoid.
- **Effective age:** `effective_age(i, j) = i - j` (i.e., query i, key j, j ≤ i causal). Drop iff `effective_age > last_span[j] * S_max`. The mask is `M[i,j] = -1e9 if (i > j AND effective_age > last_span[j]*S_max) else 0`. Past tokens survive iff their own span "covers" the query position.
- **Initialization:** `W_s, b_s = 0` (sigmoid(0) = 0.5 — all tokens start with a half-window budget, paper default). `S_max` is a user-controlled knob, default `S_max = n` (no pruning at default).
- **No Q/K/V biases** (Mistral-style), matching the modern attention layers already in the repo.

---

## Distinctive vs existing

The repo already has many sparse / linear attention variants: `sliding_window` (fixed W, not learned), `lsh_attention`, `bigbird`, `block_sparse_flash`, `nsa` (native sparse — recent). What's NEW here:

1. **Per-token learned span** (not a fixed window). Each key has its own `s_j ∈ [0,1]` controlling how many future queries it serves. A "fresh" content token (e.g. a new entity) keeps a large span; a "filler" token expires quickly.
2. **Span-head Dense on every layer.** A `W_s, b_s` projection per attention layer. Backward must propagate `ds_j` back through this projection AND through the keep/drop mask into the Q/K/V chain.
3. **`effective_age(i, j) = i - j` (causal)** — paper §2.2 / Eq. 5. The mask is `M[i, j] = -1e9 iff (j < i AND (i - j) > s_j * S_max)`.
4. **End-to-end training reduces loss on a long-context synthetic task** (sparse recall-style).

Everything else (multi-head attention with K/V sharing via GQA convention, Q/K/V/O projections, softmax chain, FD-tested backward, block wrapper with pre-LN + FFN) follows the existing `sliding_window.h` / `gqa.h` pattern.

---

## Task 1: ExpireSpanAttention — RED 1 (constructor validation + accessors)

**Files:**
- Create: `include/nn/layers/attention/expire_span.h`
- Create: `tests/test_expire_span.cpp`

**Step 1:** In `expire_span.h`, declare:

```cpp
class ExpireSpanAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // num_query_heads: number of Q heads
    // num_kv_heads:   number of distinct K and V heads (must divide num_query_heads)
    // S_max:          max effective age in tokens (default = 0 → use n at forward time)
    // s_min:          floor for the sigmoid span (paper §3.2; default 1e-6)
    ExpireSpanAttention(size_t d_model,
                        size_t num_query_heads,
                        size_t num_kv_heads,
                        size_t S_max = 0,
                        double s_min = 1e-6);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "ExpireSpanAttention"; }

    // Test accessors
    size_t d_model()       const { return d_model_; }
    size_t num_heads()     const { return num_query_heads_; }
    size_t num_kv_heads()  const { return num_kv_heads_; }
    size_t head_dim()      const { return head_dim_; }
    size_t S_max()         const { return S_max_; }
    double s_min()         const { return s_min_; }
    double scale()         const { return scale_; }

    // Cache accessors (for FD and signature tests)
    const Tensor& last_z()         const { return last_z_; }        // (n, 1)
    const Tensor& last_span()      const { return last_span_; }     // (n, 1) — clamped
    const Tensor& last_mask()      const { return last_mask_; }     // (n, n), 0 or -1e9
    const Tensor& last_attn_head(size_t h) const { return last_attn_by_head_[h]; }
    Tensor grad_input()            const { return last_d_input_; }

    // Public for test access (matches GQA / cosformer / sliding_window convention).
    Tensor W_q, W_k, W_v, W_o;          // (d_model, d_model)
    Tensor W_span, b_span;              // (1, d_model), (1, 1)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_W_span, grad_b_span;

private:
    size_t d_model_, num_query_heads_, num_kv_heads_, head_dim_, group_size_;
    size_t S_max_;
    double s_min_, scale_;

    // BPTT cache
    Tensor last_input_;        // (n, d_model)
    Tensor last_q_, last_k_, last_v_;   // (n, d_model)
    Tensor last_attn_;         // (Hq * n, n) — per-head softmax output
    std::vector<Tensor> last_attn_by_head_;
    Tensor last_head_out_;     // (n, d_model)
    Tensor last_z_, last_span_;          // (n, 1)
    Tensor last_mask_;         // (n, n) — additive mask applied
    Tensor last_d_input_;      // (n, d_model)
};
```

**Step 2:** Create `tests/test_expire_span.cpp` with Test 1:
- 5-case constructor validation: `d_model=0` throws, `num_query_heads=0` throws, `num_kv_heads=0` throws, `num_query_heads % num_kv_heads != 0` throws, `s_min < 0` throws.
- Valid construct with d_model=8, num_query_heads=4, num_kv_heads=2, S_max=4 → accessors return values; `name() == "ExpireSpanAttention"`.

**Step 3:** Verify RED — `g++ -c tests/test_expire_span.cpp -Iinclude -std=c++17` should fail because the header doesn't exist yet.

**Step 4:** Add the header file with the declaration above (forward/backward throw `std::logic_error("not implemented")`).

**Step 5:** Verify RED → GREEN by adding the implementation stubs that just throw on forward/backward but satisfy the constructor + accessors.

**Step 6:** Commit `feat(attention): ExpireSpanAttention header + constructor validation`.

---

## Task 2: ExpireSpanAttention — RED 2 (forward shape, finiteness, span init)

**Files:**
- Modify: `include/nn/layers/attention/expire_span.{h,cpp}`
- Modify: `tests/test_expire_span.cpp`

**Step 1:** Write Test 2 in `tests/test_expire_span.cpp`:
- Construct with d_model=8, num_query_heads=4, num_kv_heads=2, S_max=4.
- `forward(Tensor::random(n=6, 8))` returns Tensor shape (6, 8), all finite, all non-zero.

**Step 2:** Run and confirm RED (`forward` throws).

**Step 3:** Implement `forward` minimally:
- Compute `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v` (no biases; GQA-style reshape into heads).
- Compute `z = X @ W_span^T + b_span` (shape (n, 1)). `last_z_ = z`.
- `span = clamp(sigmoid(z), s_min, 1.0)`. `last_span_ = span`.
- For each (i, j) with `j ≤ i` (causal), drop iff `i - j > span[j] * S_max_eff` where `S_max_eff = (S_max == 0 ? n : S_max)`. Build `last_mask_` (n, n) with `0` for survivors, `-1e9` for drops. Diagonal and lower-triangle-with-survive span: `0`. Upper-triangle (j > i): `-1e9` (causal).
- Per-head scaled-dot-product with the additive mask.
- Concat heads, project via `W_o`. `last_head_out_ = concat; last_q_/last_k_/last_v_` = projected Q/K/V. Return `out`.

**Step 4:** Run and confirm GREEN.

**Step 5:** Commit `feat(attention): ExpireSpanAttention forward (shape, span head, mask, softmax)`.

---

## Task 3: ExpireSpanAttention — RED 3 (hand-derived mask signature tests)

These are the "this is wired up correctly" sanity tests — they don't require gradient chains, just the forward pass.

**Files:**
- Modify: `tests/test_expire_span.cpp`

**Step 1:** Add Tests 3-5:

- **Test 3 — all-span=1 (sigmoid of +∞) ⇒ no mask applied, falls back to causal softmax.** Build a layer, set `W_span` and `b_span` to a large positive value (e.g. 50) so `sigmoid(z) ≈ 1.0` for every token. Forward with a known small input; verify the mask matrix `last_mask_` is ALL ZEROS in the lower triangle and all `-1e9` in the strict upper triangle. (The mask tensor itself, not the output.) Also verify that `last_span_` is approximately 1.0 for every entry.

- **Test 4 — zero span ⇒ only self-attention.** Set `W_span = 0`, `b_span = -50` (so sigmoid → ~0). Forward with known input. Verify `last_span_` is all ≈ 0 (clamped to `s_min`). Verify that `last_mask_[i, j] = -1e9` for all `j < i` (because `effective_age > 0 = s_min * S_max`) and `last_mask_[i, i] = 0` (i - i = 0 ≤ 0, survive). Verify output is approximately `V[i] @ softmax([0, -1e9, ...]) = V[i]` — i.e., each query attends only to itself. (The output is `W_o @ V[i]` projected.)

- **Test 5 — span = 0.5, S_max = 4 ⇒ window of 2.** Force `last_span_ = 0.5 * ones(n, 1)` by setting `sigmoid(z) = 0.5` (z = 0). S_max = 4. So `effective_age threshold = 0.5 * 4 = 2`. For query i, keys j ∈ [max(0, i-2), i] survive. Verify the mask: `last_mask_[i, j] = -1e9 iff j < i-2 (in causal valid range)`. The diagonal and the 2 most recent lower-triangle entries are 0.

**Step 2:** Run; all should pass (GREEN).

**Step 3:** Commit `test(attention): ExpireSpanAttention mask signature tests`.

---

## Task 4: ExpireSpanAttention — RED 4 (gradient checks via FD)

The big one. Hand-derived BPTT through Q/K/V/O + softmax + mask + span head.

**Files:**
- Modify: `include/nn/layers/attention/expire_span.{h,cpp}`
- Modify: `tests/test_expire_span.cpp`

**Step 1:** Implement `backward` properly:
- Receive `grad_output` (n, d_model). Compute `d_head_concat = grad_output @ W_o^T`. Reshape into per-head `d_O_h` (n, head_dim).
- For each head: `d_attn_h = d_O_h @ V_h^T`. `d_V_h = A_h^T @ d_O_h`. Then softmax backward: `d_pre_softmax_h = (d_attn_h - sum_rows(d_attn_h ⊙ A_h, keepdim)) ⊙ A_h`. (The mask is additive in pre-softmax, so it does NOT enter the chain beyond the fact that A_h already incorporates it — `d_pre_softmax_h` already has the effect of the mask baked in via the A_h values being zero on masked positions.)
- `d_K_h = d_pre_softmax_h^T @ Q_h * scale`. `d_Q_h = d_pre_softmax_h @ K_h * scale`. (With GQA, group `Q` heads to the right `K/V` head — `group_size_ = num_query_heads_ / num_kv_heads_`.)
- Reduce per-head grads to per-projection grads (sum across heads): `d_W_q += X^T @ d_Q`, etc. Standard multi-head reduction.
- For the span head: `d_pre_softmax_h` on a dropped position is ~0 (because A_h is 0 there). On a surviving position, `d_pre_softmax_h` is nonzero. For a given key j, `ds_j` = the chain: `ds_j * S_max = ∂(mask threshold) / ∂s_j` enters via which entries of `d_pre_softmax` are nonzero. Concretely: `ds_j` contributes to `d_pre_softmax_h[i, j]` (where i ≥ j) through `d_pre_softmax_h[i, j]` is nonzero iff `last_mask_[i, j] = 0`, i.e., iff `(i - j) ≤ s_j * S_max_eff`. So `d_pre_softmax_h[i, j]` contributes to `ds_j` only when the (i, j) edge survives. Sum across all surviving `(i, j)` pairs.
  - Actually, the span `s_j` doesn't directly enter the pre-softmax value — it enters via the MASK, which is applied additively. So `d_pre_softmax_h[i, j]` already implicitly accounts for the mask. The mask is essentially a piecewise-constant function of `s_j`: `M[i, j] = -1e9` if `i - j > s_j * S_max_eff`, else `0`. The gradient of `M` w.r.t. `s_j` is `0` everywhere except at the boundary `i - j = s_j * S_max_eff` (where it's a Dirac delta — which we ignore in practice because it's a measure-zero set). So **`d_pre_softmax_h` doesn't directly contribute to `ds_j` in the interior of the keep/drop regions**. The span head's gradient comes ENTIRELY through `d_pre_softmax_h` on the SURVIVING edges, where the chain is: `ds_j` doesn't appear explicitly in the surviving-edge value (because the mask is `0` there). The only path from `s_j` to the loss is through which edges are kept vs dropped — but inside a region of constant keep/drop, this is locally flat. The actual gradient path is: `ds_j → dz_j → dW_span, db_span` via the sigmoid. **`dW_span[c] = Σ_i ds_j * x_i[c] * sigmoid'(z_j)`** where `ds_j` is the gradient of the loss w.r.t. `span[j]`.
  - The crucial realization: **`ds_j` is NOT derivable from `d_pre_softmax_h` directly.** It comes from a separate path: the SOFTMAX at query i reads `M[i, j]` — but `M[i, j]` is a piecewise constant in `s_j`. So a "standard" gradient through the mask gives 0 inside regions. **The way to actually compute `ds_j` is via a finite-difference-style re-evaluation, OR via the observation that in the SOFT MAX, the gradient of the OUTPUT w.r.t. `M[i, j]` is `d_attn_h[i, j] = d_pre_softmax_h[i, j] - A_h[i, j] * sum_k d_pre_softmax_h[i, k] * A_h[i, k]`. On a DROPPED position, `A_h[i, j] ≈ 0`, so the term is `d_pre_softmax_h[i, j]` — but `d_pre_softmax_h[i, j]` on a dropped position is also 0 (it's the gradient of the output w.r.t. a value that, in the actual forward, was `-1e9` pre-softmax — and the gradient of `softmax(x - 1e9)` w.r.t. `x` at the dropped position is 0). So **`ds_j` is actually ZERO under the piecewise-constant mask assumption**. This is a known issue with hard masking.**
  - **THE FIX (paper §3.2 + our implementation choice):** use a SOFT mask. Replace the hard threshold with a sigmoid-based soft mask: `M[i, j] = -soft_threshold * sigmoid((i - j - s_j * S_max_eff) / temperature)` where `soft_threshold = 1e9` and `temperature` is small (e.g. 0.1). This is differentiable. With temperature=0.1, at the boundary `i - j = s_j * S_max_eff`, `sigmoid(0) = 0.5`, so the mask value is `-5e8` (essentially `-1e9`). Far from the boundary, it saturates to `0` (survive) or `-1e9` (drop). The gradient is well-defined everywhere via the sigmoid derivative.
  - **Implementation choice:** store `last_soft_mask[i, j] = -soft_threshold * sigmoid((i - j - s_j * S_max_eff) / temperature)` and propagate `dM[i, j]` through the standard softmax chain. Then `ds_j = S_max_eff * Σ_i dM[i, j] * ∂sigmoid/∂(s_j * S_max_eff) * 1/S_max_eff * 1 = Σ_i dM[i, j] * sigmoid(...) * (1 - sigmoid(...)) / temperature * (-1)`. The chain is straightforward.
  - This makes the gradient check tractable AND matches the spirit of the paper (the paper uses STE / straight-through for hard masking, but a soft mask is a common differentiable relaxation that's been used in many similar papers and gives the same gradient direction).

- After computing `ds[n, 1]`: `dz = ds ⊙ sigmoid'(z) = ds ⊙ sigmoid(z) ⊙ (1 - sigmoid(z))`. `dW_span += X^T @ dz` (shape (1, d_model)). `db_span += sum(dz)`. And `d_input += dz @ W_span` (the input projection through the span head adds to `d_input`).
- Standard input gradient: `d_input += d_Q @ W_q^T + d_K @ W_k^T + d_V @ W_v^T` (with per-head reshape reduction).

**Step 2:** Add gradient tests:
- **Test 6 — FD input gradient.** `n=4`, `d_model=8`, `Hq=2`, `Hkv=1`, `S_max=4`, `temperature=0.5`. Random non-uniform init (mandatory). Compare analytical `grad_input` against central-difference FD on `forward`. Target: `rel_err < 1e-4`. The FD step is `1e-5`. The non-uniform init ensures the span varies token-by-token.
- **Test 7 — FD W_q, W_k, W_v, W_o gradient.** Same config. All four projections' gradients should match FD to `rel_err < 1e-5`.
- **Test 8 — FD W_span, b_span gradient.** This is the new piece. Random init for `W_span`/`b_span` too. `rel_err < 1e-4`.

If any test fails: consult the systematic-debugging skill. Common bugs:
- Forgot to scale by `temperature` in the sigmoid derivative.
- `ds_j` accumulated the wrong sign (`-1` from `-soft_threshold * sigmoid(...)` derivative w.r.t. `(i - j - s_j * S_max)` is `sigmoid * (1 - sigmoid)`, but the derivative w.r.t. `s_j` has an extra `-1/S_max_eff` from `-(i - j - s_j * S_max_eff)` chain).
- Forgot to add `d_input += dz @ W_span` (the cross-path from span head back to input).

**Step 3:** Run; all should pass.

**Step 4:** Commit `feat(attention): ExpireSpanAttention backward with hand-derived BPTT and soft mask; gradient checks pass`.

---

## Task 5: ExpireSpanAttention — RED 5 (determinism + cache shape + update_weights + zero_grad)

**Files:**
- Modify: `tests/test_expire_span.cpp`

**Step 1:** Add Tests 9-13:
- **Test 9 — cache shapes.** After a forward with n=6, `last_q_/k_/v_` are (6, 8), `last_attn_by_head_[h]` are (6, 6), `last_z_` is (6, 1), `last_span_` is (6, 1), `last_mask_` is (6, 6).
- **Test 10 — determinism.** Two fresh layers with the same seed (`Tensor::seed(42)`) produce bit-exact forward output.
- **Test 11 — zero_grad clears all 6 gradient tensors** (`W_q, W_k, W_v, W_o, W_span, b_span`). Sum < 1e-15.
- **Test 12 — update_weights moves all 6.** Save copy of params, run forward+backward+update_weights(lr=0.01), verify each param changed by non-zero amount.
- **Test 13 — mutation test: zero out W_span ⇒ output is unchanged bit-exact (because sigmoid(0)=0.5 still spans all positions) AND the gradient still flows through the OTHER params.** This proves the span head is wired into the chain. Then a stronger mutation: set W_span so sigmoid(z)=1.0 for token 0, 0.0 for token 1. Verify the mask for token 0 is "all survive" and for token 1 is "self only" (the boundary cases from Test 4 / 5).

**Step 2:** Run; all should pass.

**Step 3:** Commit `test(attention): ExpireSpanAttention cache, determinism, update_weights, mutation tests`.

---

## Task 6: ExpireSpanBlock + ExpireSpanModel

**Files:**
- Modify: `include/nn/layers/attention/expire_span.{h,cpp}`
- Modify: `tests/test_expire_span.cpp`

**Step 1:** Add `ExpireSpanBlock` (pre-LN → expire attn → residual → pre-LN → GELU FFN → residual):
- Constructor: `(d_model, num_query_heads, num_kv_heads, S_max=0, s_min=1e-6, ffn_dim=0)`.
- `parameters()` returns params of attn + (W1, b1, W2, b2 of FFN if ffn_dim > 0).
- `forward(input)` returns `input + attn(LN(input))`; then if ffn_dim > 0: `+ FFN(LN(...))`.
- `backward` chains all the gradients.

**Step 2:** Add `ExpireSpanModel`:
- Constructor: `(d_input, d_model, d_output, num_blocks, num_query_heads, num_kv_heads, S_max=0, s_min=1e-6, ffn_dim=0)`.
- `forward(x)`: input projection → stack of blocks → classifier.

**Step 3:** Add Tests 14-17:
- **Test 14 — block forward shape.** `n=6`, `d_model=8`. Block returns (6, 8), finite, nonzero.
- **Test 15 — block input FD.** `n=6, d_model=8, Hq=2, Hkv=1, ffn_dim=16`. Block input gradient matches FD to `rel_err < 1e-3` (looser because the LN chain + FFN chain amplifies noise).
- **Test 16 — model forward.** `(4, 3) → (4, 5)` finite, nonzero.
- **Test 17 — end-to-end training.** MSE loss on a synthetic `y = W_target · x` target reduces over 60 SGD steps at lr=0.01. (Loss should drop >30%.)

**Step 4:** Commit `feat(attention): ExpireSpanBlock + ExpireSpanModel with end-to-end training test`.

---

## Task 7: Register in umbrella + Makefile + run full suite

**Files:**
- Modify: `include/nn/nn.h` (add `#include "layers/attention/expire_span.h"` near the other attention headers)
- Modify: `Makefile` (add `build/test_expire_span` rule, append `$(BUILD_DIR)/test_expire_span` to the `tests:` deps, add `=== Running Expire-Span Tests ===` echo in `run_tests`)

**Step 1:** Add to `include/nn/nn.h`.

**Step 2:** Verify standalone compile: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` → no errors.

**Step 3:** Add Makefile rules.

**Step 4:** Run `make tests` to verify build of the new test target.

**Step 5:** Run `./build/test_expire_span` — all 17+ tests pass.

**Step 6:** Run `make run_tests` — verify no regressions in existing tests.

**Step 7:** Commit `chore: register Expire-Span in umbrella + Makefile`.

---

## Out of scope (v1)

- **STE-style hard mask with straight-through estimator.** We use a soft mask for differentiability. The paper uses a hard mask with STE; that's an alternative implementation choice that would also work but doesn't add new gradient-check coverage.
- **Multi-modal variants (e.g. CoLT5's heavy/light branches).** Pure Expire-Span is the chosen tractable variant.
- **Per-position span with token-position features (e.g. cross-attention variants).** Expire-Span is causal self-attention in v1.

---

## Success criteria

- [ ] All 17+ focused checks pass at machine precision (or rel_err < 1e-3 for block-level).
- [ ] Mutation-tested non-vacuous: stubbing out the span head changes the output AND its gradients are caught by at least 2 tests.
- [ ] End-to-end training reduces MSE loss > 30%.
- [ ] `make run_tests` shows no regressions.
- [ ] Umbrella compiles standalone.
- [ ] Clean build under `-Wall -Wextra`.