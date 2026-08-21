# Hyper-Connections Implementation Plan

> **For Hermes:** Use test-driven-development skill. Build test-first.

**Goal:** Implement Hyper-Connections (DeepSeek-AI 2025, https://arxiv.org/abs/2409.19606) — a learnable generalization of the residual connection that replaces `y = x + f(x)` with `y = α ⊙ x + β ⊙ f(x)` for learnable per-channel `α, β ∈ (0,1)`. Drop-in replacement for the standard residual in any block (attention, FFN, MoE, Mamba, etc.).

**Architecture:** A simple Layer (`HyperConnection`) that wraps any inner sub-layer. Two variants:
1. **Scalar width (1 stream)** — per-channel `α, β ∈ R^d`, broadcasting across the batch.
2. **Wide width (n parallel streams)** — `(n, d)` streams mixed through learnable `(n, n)` input / output maps.

**Tech Stack:** C++17, Tensor class, existing Layer patterns. Composes with all existing layers (Dense, MambaBlock, Attention, FFN).

---

## Paper reference

DeepSeek-AI. **"Hyper-Connections."** arXiv:2409.19606 (2025).

https://arxiv.org/abs/2409.19606

## Core idea

Standard residual: `x_{l+1} = x_l + f(x_l)`. The residual strength is implicit (always 1).

**Hyper-Connections** make the residual learnable per-channel:
```
x_{l+1} = α_l ⊙ x_l + β_l ⊙ f(x_l)
```
with `α_l, β_l ∈ R^d` initialized to `α = 1, β = 0` (so the layer starts as identity — standard residual — and learns to deviate).

The paper's full version uses n parallel streams with `(n, n)` input / output mixing matrices. For this implementation we focus on the **scalar-width** version (n=1), which is the cleanest form and recovers standard residual at init.

## Math

### Per-layer
- `α_log ∈ R^d`, `β_log ∈ R^d` raw parameters
- `α = sigmoid(α_log) ∈ (0, 1)` — bounded
- `β = β_scale · sigmoid(β_log) ∈ (0, β_scale)`

`β_scale = 1.0` is the canonical "Hyper-Connection" init; the residual magnitude at init is the same as standard residual.

### Forward (n_samples, d_model)
```
sub_out = inner(x)                              // (n, d)
out     = α * x + β * sub_out                   // broadcasts (1, d) across (n, d)
```

### Backward (n_samples, d_model)
```
d_sub_out = β * d_out                           // (n, d)
d_inner   = inner.backward(d_sub_out, lr)      // returns d_input_inner
d_x       = α * d_out + d_inner                 // α from residual path + inner path

d_α = sum_t d_out[t, :] * x[t, :]              // (1, d)
d_β = sum_t d_out[t, :] * sub_out[t, :]        // (1, d)

d_α_log = d_α * α * (1 - α)                    // sigmoid derivative
d_β_log = β_scale * d_β * β * (1 - β) / β_scale * sigmoid'(β_log)
```

### Block wrapper (residual + pre-norm FFN)

```
y = x + α_2 * (FFN(LN(x))) + β_2 * (... no, simpler:)

y = α * x + β * inner(LN(x))     # wrapped as a single block
```

The block wraps pre-norm + sub-layer, and HyperConnection replaces the implicit `+ x` residual.

## File layout

- `include/nn/layers/utility/hyper_connection.h` — class declarations
- `include/nn/layers/utility/hyper_connection.cpp` — implementation
- `tests/test_hyper_connection.cpp` — focused tests
- Register in `include/nn/nn.h` umbrella
- Register in `Makefile` (build rule + tests: deps + run_tests echo)

## Tests (~16 focused checks)

1.  Constructor validation (d_model=0 throws, inner=nullptr throws)
2.  Forward shape (n=4, d=8)
3.  Forward output is finite + nonzero
4.  At init (α→1, β→0): output ≈ input to machine precision (recovers standard residual)
5.  Forced α=0.5, β=0.5: output = 0.5 * x + 0.5 * inner(x) to machine precision
6.  Inner-layer forward cache correctly used in backward (test with identity inner)
7.  Input gradient FD check (rel_err < 1e-9)
8.  α_log gradient FD check (rel_err < 1e-9)
9.  β_log gradient FD check (rel_err < 1e-9)
10. α_log gradient is nonzero when d_out nonzero AND x nonzero (sanity)
11. β_log gradient is nonzero when d_out nonzero AND sub_out nonzero
12. HyperConnectionBlock: pre-LN → Dense → HyperConnection, forward shape
13. HyperConnectionBlock: input gradient FD check
14. HyperConnectionBlock: training reduces loss (50 SGD steps)
15. HyperConnectionModel: full model with input projection + N blocks + classifier
16. Mutation test: zeroing α/β gradient → input grad test fails

## Style notes

- Match the Mixture-of-Depths pattern (parameter ownership, move semantics, depth-first structure).
- Use `β_scale = 1.0` (DeepSeek paper canonical value).
- Init `α_log = 0, β_log → -∞` so `sigmoid(β_log) ≈ 0` at init (perfect residual identity).
  In practice use `β_log = -10.0` to avoid log(0) issues with gradients at init.
- All gradients are tested against finite differences; mutation-tested.
