# Forgetting Transformer (FoX) Implementation Plan

**Goal:** Add `FoXAttention` / `FoXBlock` / `FoXModel` — causal softmax attention with a
learnable data-dependent *forget gate* applied as an additive log-decay bias (Lin, Yang,
Sun, et al., ICLR 2025, "Forgetting Transformer: Softmax Attention with a Forget Gate",
https://arxiv.org/abs/2503.02130).

**Architecture:**

Per head `h`, per token `t`:

```
z[t, h]    = (X W_f)[t, h]                      # forget-gate logit
f[t, h]    = sigmoid(z[t, h]) in (0, 1)
logf[t, h] = log f[t, h]                        # <= 0
D[t, h]    = sum_{i<=t} logf[i, h]              # cumulative log-decay
bias[t, s] = D[t, h] - D[s, h] = sum_{i=s+1..t} logf[i, h]   (<= 0, s <= t)
scores[t, s] = (Q_h[t] . K_h[s]) * scale + bias[t, s]   for s <= t, else -inf
A          = row_softmax(scores)                # causal
head_out   = A @ V_h
out        = concat_h(head_out) @ W_o
```

The forget bias is exactly `log(prod_{i=s+1..t} f_i)`, so `exp(bias)` multiplicatively
decays attention to older keys — FlashAttention-compatible "Forgetting Attention" in the
paper. When `f -> 1` the layer reduces to vanilla causal attention.

**Backward (the FoX-specific chain):**
`d_bias[t, s] = dS[t, s]`; since `bias[t,s] = sum_{i=s+1..t} logf[i]`,
`d_logf[i] = sum_{t>=i} sum_{s<i} dS[t, s]`. Then `dz = d_logf * (1 - f)` because
`d/dz log(sigmoid(z)) = 1 - sigmoid(z)`. Then the standard `W_f` projection backward.

**Tech Stack:** C++17, repo `Tensor`, `Layer`, `LayerNorm`, `Dense`; no external deps.

**Files:**
- Create `include/nn/layers/attention/fox.h` / `.cpp`
- Create `tests/test_fox.cpp`
- Modify `include/nn/nn.h` (umbrella include), `Makefile` (build rule + `tests:` + `run_tests:`)

## Tasks (TDD, one behavior per cycle)

1. Write `tests/test_fox.cpp` with constructor-validation + forward-shape tests. Run → RED (no header).
2. Implement constructor + validation + forward. Run → GREEN.
3. Add causality test (perturbing `x[s]`, `s > t`, leaves `y[t]` bit-exact). RED → implement causal mask.
4. Add forget-gate invariant tests (`f in (0,1)`, `bias <= 0`, `bias[t,t] == 0`).
5. Add input-gradient FD check + all 5 parameter FD checks (`W_q/W_k/W_v/W_o/W_f`).
   `W_f` is the signature parameter — its FD check is what distinguishes FoX from causal MHA.
6. Add "reduces to vanilla causal attention as f -> 1" test: force `W_f` large positive,
   compare against a `bias == 0` reference computed in the test.
7. Add `FoXBlock` (pre-LN + residual + GELU FFN) forward + FD check.
8. Add `FoXModel` forward shape + training-reduces-loss.
9. Mutation test: stub `d_logf` accumulation → `W_f` grad check must fail.
10. Register in umbrella + Makefile; run full `make run_tests`; commit.

**Verification:** `make build/test_fox && ./build/test_fox` → all checks pass; FD rel_err < 1e-4.
