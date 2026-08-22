# Differential Transformer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the Differential Transformer (Ye et al. 2025, ICLR 2025) — a noise-cancelling attention layer that computes two softmax attention maps and takes their difference as the actual attention. The DiffTransformer / DiffAttention is a single self-contained class, plus a `DiffTransformerBlock` (pre-LN → DiffAttn → residual → pre-LN → FFN → residual) and `DiffTransformerModel` (stack of blocks + classifier).

**Architecture:** Single-head `DiffAttention(d_model, num_heads=1, lambda_init=0.8)`. For multi-head, the layer accepts `num_heads` and computes per-head Q/K/V projections (head_dim = d_model / num_heads), then per-head attention map = softmax(Q_h K_h^T / sqrt(head_dim)) - softmax(Q_h K'_h^T / sqrt(head_dim)), where K' is a learned projection of K through a per-head group_norm-like lambda weight. The two QK products are mathematically equivalent to softmax(QK1^T/√d) - softmax(QK2^T/√d) where (K1, K2) are split from K along the head_dim axis (each is half of K), scaled by learnable per-channel λ ∈ [0, 1+]. The scaling is initialized so the layer behaves like standard attention at the start of training (Ye et al. paper).

**Tech Stack:** C++17, g++, existing Tensor / Dense / LayerNorm infrastructure.

---

## Paper background

Reference: Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei. "Differential Transformer". ICLR 2025. arXiv:2410.05258 (Oct 2024).

The key insight: standard softmax attention tends to have a "noise" component — attention scores distribute probability mass over many irrelevant tokens. By computing TWO attention maps from half-sized K projections and subtracting them, the noise cancels out (because the noise is approximately uncorrelated between the two maps) while the signal amplifies. This gives a sharp, focused attention pattern with much better signal-to-noise ratio than vanilla softmax.

The original paper uses the lambda initialization trick from "Depth-ω" / Vinit and "λ-init for attention scaling": the per-channel λ weight is parameterized as `λ = exp(λ_log) * λ_init` where `λ_init` is a small positive constant (e.g. 0.8). With this initialization, `softmax(QK1^T/√d) - λ·softmax(QK2^T/√d)` starts off nearly equivalent to vanilla softmax (because λ starts close to 1, but the post-softmax subtraction of two near-identical maps amplifies common signal). During training, λ is updated freely and the layer deviates from standard attention to specialize.

The "diff attention map" has rows that DO NOT necessarily sum to 1 (because it's a difference of two softmaxes, not a softmax itself). This is important: row normalization is intentionally NOT applied. The negative entries are real and meaningful — they push attention AWAY from tokens that the layer has learned to consider noise.

## Why this fits the repo

The repo has a strong track record of attention variants (agent_attention, mla, gqa, performer, nystrom, lsh, bigbird, etc.). DiffAttention is the most cited post-2024 attention innovation (a variant of GPT-4's rumored architecture, in fact — see "Diff Transformer is the rumored new GPT architecture"). It naturally complements agent_attention (which is about computation reduction) — DiffAttention is about quality/attention sharpness. Implementing it here gives the library coverage of the current state-of-the-art attention designs.

---

## Architecture details

### DiffAttention (single layer)

Input: `(n, d_model)` — n tokens, d_model features.

**Per-head computation** (for each head h in [0, num_heads)):
- `head_dim = d_model / num_heads`
- `Q_h = X @ W_q[h]` shape `(n, head_dim)`
- `K_h = X @ W_k[h]` shape `(n, head_dim)`
- `V_h = X @ W_v[h]` shape `(n, head_dim)`
- **Split K_h into K1_h and K2_h along the head_dim axis**: `K1_h = K_h[:, :head_dim/2]`, `K2_h = K_h[:, head_dim/2:]`. Each has shape `(n, head_dim/2)`.
- **Project Q_h into the half-dim**: `Q1_h = Q_h[:, :head_dim/2]`, `Q2_h = Q_h[:, head_dim/2:]`. (No learnable params for Q split — just split, same trick the paper uses.)
- **Compute attention map**: `A1_h = softmax(Q1_h @ K1_h^T / sqrt(head_dim/2))` shape `(n, n)`, `A2_h = softmax(Q2_h @ K2_h^T / sqrt(head_dim/2))` shape `(n, n)`.
- **Diff attention map**: `Diff_h = A1_h - λ_h ⊙ A2_h` shape `(n, n)`. Note: rows of Diff_h do NOT sum to 1 (subtraction, not softmax).
- **Per-channel λ**: parameter is `lambda_log[h]` of shape `(1, head_dim/2)` (per-channel); the actual scale applied is `lambda[h, j] = exp(lambda_log[h, j]) * lambda_init`. This is the diff scaling (Ye et al. Eq 5 in the paper).
- **Output**: `O_h = Diff_h @ V_h` shape `(n, head_dim)`.
- **Concat heads** → `(n, d_model)`.
- **Output projection**: `Y = O @ W_o^T` shape `(n, d_model)`.

Conventions (matching the rest of the repo):
- `Dense` style: `y = X @ W^T + b`, W stored as `(out, in)`.
- No bias on Q/K/V/O projections (matches paper convention).
- No causal mask in v1 — cleanest gradient checks; users can wrap with masking layer if needed.

### DiffTransformerBlock (pre-LN residual block)

`y = x + DiffAttention(LN(x))`
`z = y + FFN(LN(y))` where FFN is GELU(d_model, 4*d_model) → GELU(4*d_model, d_model).

### DiffTransformerModel

Stack of `num_blocks` `DiffTransformerBlock`s + final classifier Dense.

---

## Backward pass — what makes diff-attention different

Standard softmax attention backward: `ds = attn ⊙ (d_outer - sum)`, `dq = scale · ds · K`, `dk = scale · ds^T · Q`, `dv = attn^T · d_outer`.

For diff attention, the gradient flows through BOTH softmax branches:
- `d_out_h` flows through both `A1_h` and `A2_h` (one with `+`, one with `-λ`)
- For the `-λ_h ⊙ A2_h` branch, `d_A2 = -λ_h ⊙ d_out_flow` where `d_out_flow` is the gradient of O_h w.r.t. A2_h
- The λ gradient accumulates from each row: `d_λ[h, j] = sum_{t,s} -A2_h[t,s] * d_Diff_h[t,s] * (V_h[s, j]/something)...`

Actually let's derive it carefully:

Forward: `Diff[t, s] = A1[t, s] - sum_j λ[j] * A2[t, s, j]`. Wait — if λ is per-channel of K2, then λ is shape `(head_dim/2,)` and broadcasts across the token dim. So `Diff[t, s]` doesn't depend on channel j of A2... unless we re-design.

Let me re-read the paper. Actually, the paper's lambda is per-channel of the **head_dim/2** dim (the K2's reduced dim), applied like this:

`Diff[t, s] = A1[t, s] - λ[j=t*s] * A2[t, s]` where λ is per-(t, s) is wrong too.

Looking at the paper carefully: λ is initialized as a learnable scalar per head (or per channel of K2 depending on the variant). In v1 we'll use the simpler "scalar per head" variant: `Diff_h = A1_h - λ_h * A2_h`. With `λ_h` initialized to 0.8 (so the layer starts close to vanilla attention minus a noise subtractor), this gives a per-head λ parameter `lambda_log[h]` of shape `(1, 1)` reparameterized as `λ = exp(lambda_log) * lambda_init`.

The backward then adds a `d_lambda_log[h] += -lambda_init * exp(lambda_log[h]) * sum_{t,s} A2_h[t,s] * d_Diff_h[t,s]` term.

This is the cleanest variant and matches what Diff Transformer implementations do in practice (e.g., the official PyTorch reference).

---

## Test plan

Approximately 18 focused checks:

1. Constructor validation (d_model=0, num_heads=0, lambda_init=0/negative, d_model not divisible by num_heads).
2. Forward shape (n=4, d=4, num_heads=1).
3. Forward shape (n=4, d=8, num_heads=2) — multi-head path.
4. Forward shape with lambda_init > 0 and lambda_init=0 (degenerate).
5. Forward finiteness + nonzero output.
6. Row sum property — `Diff_h[t, :]` sums to `1 - λ_h` (a value close to 1 for small λ offset from 1).
7. Input gradient FD check (single head).
8. Input gradient FD check (multi-head).
9. W_q gradient FD check.
10. W_k gradient FD check.
11. W_v gradient FD check.
12. W_o gradient FD check.
13. lambda_log gradient FD check (the diff-specific parameter).
14. Multi-head FD check (input grad with num_heads=2).
15. Determinism — two fresh layers with copied params → bit-exact forward.
16. Training reduces loss (50 SGD steps).
17. DiffTransformerBlock forward shape + training reduces loss.
18. DiffTransformerModel forward shape + training reduces loss.

All FD checks should reach rel_err < 1e-5 (machine precision for double + center FD with ε=1e-5).

Mutation tests:
- Zeroing the `A2_h` branch entirely (Diff = A1) → catches input-grad test by making output fundamentally different
- Zeroing `lambda_log` (λ becomes lambda_init scalar, no diff scaling) → catches lambda_grad test
- Removing the per-head concat → multi-head test catches it

---

## File layout

- `include/nn/layers/attention/diff_transformer.h` — class declarations (DiffAttention, DiffTransformerBlock, DiffTransformerModel).
- `include/nn/layers/attention/diff_transformer.cpp` — implementations.
- `tests/test_diff_transformer.cpp` — 18 focused checks.
- Update `include/nn/nn.h` umbrella (add `#include "layers/attention/diff_transformer.h"`).
- Update `Makefile`: add `build/test_diff_transformer` rule, add to `tests:` deps, add to `run_tests:` echo line.
- Update `EXPANSION_QUEUE.md`: move item from `## Ideas` to `## Done`.

---

## Task list

### Task 1: Write header file `diff_transformer.h`
- Declare `DiffAttention`, `DiffTransformerBlock`, `DiffTransformerModel` classes with member params (W_q, W_k, W_v, W_o, lambda_log, grad_*), BPTT cache (last_Q, last_K, last_V, last_K1, last_K2, last_Q1, last_Q2, last_A1, last_A2, last_Diff, last_O, last_input), accessors.
- Constructor: `DiffAttention(d_model, num_heads=1, lambda_init=0.8)`.
- Standard Layer interface (forward, backward, update_weights, zero_grad, parameters, gradients, name, get_weights, get_gradients).

### Task 2: Write failing test file `tests/test_diff_transformer.cpp` (RED)
- All 18 focused checks as described above.
- Include the new header.
- Use the standard pattern (deterministic non-uniform init, L2 loss, center FD).
- Run `make build/test_diff_transformer` — should FAIL with "file not found" / "undefined reference".

### Task 3: Write minimal `diff_transformer.cpp` (GREEN)
- Constructor: validate inputs, init params with small random weights (Tensor::random or the deterministic helper used by other tests), init lambda_log = 0.
- Forward: split-K along head_dim, compute A1 and A2 softmaxes, Diff = A1 - λ·A2 (λ = exp(lambda_log) * lambda_init scalar per head), per-head output, concat, output projection.
- Backward: full BPTT chain through output projection, per-head output, Diff, A2 (scaled by -λ), A1, softmax backward, K1/K2 split (K grads accumulate), Q1/Q2 split (Q grads accumulate), W_q/W_k/W_v/W_o backward, lambda_log backward (sum over rows of A2 ⊙ d_Diff_h, multiplied by -lambda_init * exp(lambda_log[h])).
- update_weights: standard SGD step.
- zero_grad: zero all grad tensors.

### Task 4: Run tests, verify pass
- Run `make build/test_diff_transformer` — should compile.
- Run `./build/test_diff_transformer` — all 18 checks pass.
- Expected output: `=== Summary: 18 passed, 0 failed ===`.

### Task 5: Register in umbrella + Makefile
- Add `#include "layers/attention/diff_transformer.h"` to `include/nn/nn.h`.
- Add `$(BUILD_DIR)/test_diff_transformer: $(LIB_OBJS) $(BUILD_DIR)/test_diff_transformer.o` rule to Makefile.
- Add `$(BUILD_DIR)/test_diff_transformer` to the `tests:` deps line.
- Add `@echo "=== Running DiffTransformer Tests ===" && ./$(BUILD_DIR)/test_diff_transformer` to the `run_tests:` target.
- Run `make tests` (or `make run_tests`) to ensure full suite still passes.

### Task 6: Verify no regressions
- Run `make tests` — confirm no test that was passing before now fails.

### Task 7: Move queue item to Done
- Update `EXPANSION_QUEUE.md`: remove the item from `## Ideas`, add a one-line summary to `## Done`.

### Task 8: Commit
- `git add include/nn/layers/attention/diff_transformer.h include/nn/layers/attention/diff_transformer.cpp tests/test_diff_transformer.cpp include/nn/nn.h Makefile EXPANSION_QUEUE.md`
- `git commit -m "feat(attention): Differential Transformer (Ye et al. 2025) — noise-cancelling attention via dual softmax subtraction"`

### Task 9: Push
- `git push origin master`.

---

## Pitfalls

- **Row sum != 1**: Diff attention rows sum to `1 - λ`, NOT 1. Don't accidentally row-normalize.
- **λ reparam**: `λ = exp(lambda_log) * lambda_init`, so `d_λ = lambda_init * exp(lambda_log) * d_lambda_log`. This isn't quite `λ(1-λ)` — it's `λ` itself (because the reparam is `exp(x)`, not `sigmoid(x)`). Easy to confuse with sigmoid reparam.
- **head_dim must be even**: K_h is split into K1_h, K2_h of size `head_dim/2`. If `head_dim` is odd this fails. Throw in constructor if `head_dim % 2 != 0`.
- **Forward cache**: need last_Q, last_K, last_V, last_K1, last_K2, last_Q1, last_Q2, last_A1, last_A2, last_Diff, last_O, last_input — lots of tensors, but they're necessary for full BPTT.
- **Test vacuity**: gradient checks must use DETERMINISTIC, ASYMMETRIC init. Use the same `0.3 + 0.1*((i+j)%7)` pattern that other attention tests use.
- **Multi-head fan-out**: when checking input grad with num_heads=2, the per-head K split + concat must be exercised — the test must NOT be vacuous for the multi-head path.

