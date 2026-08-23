# Log-Linear Attention Implementation Plan

> **For Hermes:** Use test-driven-development skill. Build test-first.

**Goal:** Implement Log-Linear Attention (Guo, Yang, Goel, Xing, Dao, Kim, ICLR 2026, https://arxiv.org/abs/2506.04761) — a sequence-modeling primitive that generalizes linear attention / state-space models to use a logarithmically-growing set of Fenwick-tree partitioned hidden states, balancing the efficiency of linear attention with the expressivity of softmax attention. Implement the **Log-Linear Mamba-2 variant** (the simpler of the two case studies in the paper, §3.1 + §3.4 with the Mamba-2 A transition) using the **recurrent form** (Eq. 3) for tractable gradient checks.

**Architecture:** A single `LogLinearAttention(d_model, n_heads, d_state, d_inner=0, max_levels=0)` class. Block structure:

```
x ──► in_proj ──► [x_ssm, gate]
       ├─ a_t = σ(dt_bias + a_proj(x_t))              per-head scalar decay
       ├─ b_t = b_proj(x_t)                            per-head value matrix (d_state × d_head flattened)
       ├─ k_t = k_proj(x_t)                            per-head key matrix
       ├─ q_t = q_proj(x_t)                            per-head query matrix
       ├─ λ_t = λ_proj(x_t)                            per-(token, level) lambda (L levels)
       │
       └─ For t = 0..T-1:
            Fenwick-tree state update:
              lssb_t = lssb(t+1) (least significant set bit of (t+1))
              S^(0)_t = b_t ⊗ k_t                                          (immediate contribution)
              for ℓ in 1..min(lssb_t, L-1): S^(ℓ)_t = 0
              for ℓ == lssb_t + 1 (if ℓ < L): S^(ℓ)_t = Σ_{ℓ' < ℓ} S^(ℓ')_{t-1}  (promotion)
              for ℓ > lssb_t + 1: S^(ℓ)_t = S^(ℓ)_{t-1}                    (carry forward)
            Apply decay a_t to all levels (multiplicative on state):
              S^(ℓ)_t ← diag(a_t)·S^(ℓ)_t  for all ℓ
            Output:
              o_t[h] = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · q_t[h]^T · S^(ℓ)_t[h]
       │
y_t = silu(g_t) ⊙ (o_t + D ⊙ x_ssm_t) → out_proj → out
```

**Tech Stack:** C++17, g++, existing Tensor / Dense infrastructure. No new dependencies.

---

## Paper reference

Guo, Yang, Goel, Xing, Dao, Kim. **"Log-Linear Attention."** ICLR 2026, arXiv:2506.04761 (Jun 2025).

https://arxiv.org/abs/2506.04761

Code: https://github.com/HanGuo97/log-linear-attention

## Core idea

Linear attention / SSMs use a **fixed-size hidden state** to model context — this is a fundamental limitation for associative recall tasks. Softmax attention has variable-size state (one key-value pair per token, O(T) memory at decoding) but O(T²) compute. **Log-linear attention** strikes a middle ground by maintaining a Fenwick-tree partitioned set of `L = O(log T)` states, with recent tokens kept at high resolution (level 0, 1 token each) and distant tokens summarized more coarsely (level ℓ has 2^(ℓ-1) tokens per bucket).

The masking matrix `M_H` is a **hierarchical (H) matrix** with `L = ⌈log₂ T⌉ + 1` levels; the recurrence is `S^(ℓ)_t` per level, with Fenwick-tree promotion: when the level-0 state "fills" (every 2 steps), it gets promoted to level 1; when level 1 fills (every 4 steps), it gets promoted to level 2, etc. This decouples memory (O(log T)) from compute (O(T log T)).

## Math

### Per-token forward (single head h, single level ℓ)

For head h, the per-level state is `S^(ℓ)_t[h] ∈ R^{d_head × d_state}` (head_dim × d_state, matching Mamba-2's head structure). The output combines contributions from all L levels weighted by learnable per-token per-head λ:

```
o_t[h, dh] = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · Σ_{ds=0}^{d_state-1} q_t[h, dh, ds] · S^(ℓ)_t[h, dh, ds]
```

### Fenwick tree state update (Eq. in §3.1 / page 5, recurrence)

Let `lssb(t+1) = index of least significant set bit in binary of (t+1)`. For token t (0-indexed):

```
S^(ℓ)_t[h] = {
    b_t[h] ⊗ k_t[h]                  if ℓ == 0                           // immediate
    0                                  if 0 < ℓ ≤ lssb(t+1)              // cleared (promoted up)
    Σ_{ℓ'=0}^{ℓ-1} S^(ℓ')_{t-1}[h]   if ℓ == lssb(t+1) + 1  and ℓ < L    // merge & promote
    diag(a_t[h]) · S^(ℓ)_{t-1}[h]    if ℓ > lssb(t+1) + 1                 // carry forward with decay
}
```

Note: the paper's recurrence applies `a_t · S^(ℓ)_{t-1}` as a multiplicative decay on the carry-forward path. We unify: every level gets the decay `S^(ℓ)_t ← diag(a_t) · S^(ℓ)_t` after construction.

### Parameter count

For `n_heads=H`, `d_inner=H·d_head`, `d_state=N`:
- in_proj: 2 × d_inner × d_model + 2 × d_inner
- out_proj: d_model × d_inner + d_inner
- a_proj: n_heads × d_model + n_heads (sigmoid log-decay input)
- b_proj: d_inner × d_model + d_inner (per-head b matrix rows = d_head × d_state wait, here we use d_inner for full matrix)
- k_proj, q_proj: same as b_proj
- λ_proj: n_heads × L × d_model + n_heads × L (per-(token, head, level))
- D_skip: d_inner (per-channel skip)
- dt_bias: n_heads

So total params ≈ (2d_inner·d_model + 2d_inner) + (d_model·d_inner + d_inner) + n_heads·(d_model+1) + 3·(d_inner·d_model + d_inner) + n_heads·L·(d_model+1) + d_inner + n_heads
                ≈ 7·d_inner·d_model + 7·d_inner + 2·n_heads + n_heads·L·d_model + ...

For default d_model=4, n_heads=2, d_inner=4 (head_dim=2), d_state=2, L=3: ~ 7·16 + 28 + 4 + 12·3 + ... ~ 200+ params.

## File layout

```
include/nn/layers/recurrent/log_linear_attention.{h,cpp}    — LogLinearAttention class + LogLinearAttentionModel
tests/test_log_linear_attention.cpp                         — focused checks
```

`LogLinearAttentionModel(input_dim, d_model, output_dim, num_layers, n_heads, d_state, d_inner=0, max_levels=0)` is a stack of N LogLinearAttention blocks with input projection + classifier.

## Implementation order (TDD red-green-refactor)

1. Test 1: forward shape with single head (1, T=4, d=4, n_heads=1, d_state=2, L=2)
2. Test 2: forward shape with multi-head (T=4, d=4, n_heads=2, d_state=2, L=3)
3. Test 3: forward finiteness + nonzero output
4. Test 4: hand-derived single-token T=1 reference output (compute by hand)
5. Test 5: input gradient FD check (rel_err < 1e-4)
6. Test 6: a_proj weights gradient FD check (exercises decay path)
7. Test 7: b_proj weights gradient FD check (exercises level-0 immediate state path)
8. Test 8: q_proj weights gradient FD check (exercises the λ-weighted output contraction)
9. Test 9: λ_proj weights gradient FD check (exercises the per-level mixing)
10. Test 10: D_skip gradient FD check
11. Test 11: dt_bias gradient FD check
12. Test 12: training reduces loss (single block, 50 SGD steps)
13. Test 13: Fenwick tree promotion correctness (manual sequence: at t=0..3, verify which level each token's b_t⊗k_t ends up in)
14. Test 14: zero-λ → only level-0 contributes (sanity for λ_proj backward)
15. Test 15: log-linear model forward shape
16. Test 16: log-linear model training reduces loss (2-block stack)
17. Test 17: parameters/gradients shape consistency
18. Test 18: determinism — two fresh LogLinearAttention with copied params produce bit-exact forward
19. Test 19: λ_proj shape (n_heads, L) per-head multiplicity matches
20. Test 20: at L=1, log-linear attention degenerates to a single-state Mamba-2 (with extra λ^(0)=1)

## Backward pass — Fenwick tree BPTT

The forward recurrence `S^(ℓ)_t` has three cases (immediate / cleared / merge-promoted / carry-forward). The backward pass must propagate `d S^(ℓ)_t` back through the case that was active at each (t, ℓ):

- **Case ℓ=0 (immediate)**: `S^(0)_t` is built fresh from b_t ⊗ k_t. ∂L/∂b_t[h, dh, ds] = Σ_t dS^(0)_t[h, dh, ds] · k_t[h, dh, ds] (wait — b is head_dim×d_state, but here we use d_inner flattened where head h's slice is [h·d_head, (h+1)·d_head) for head dim and full d_state for state). Need to be careful with shapes.
- **Case cleared (0 < ℓ ≤ lssb)**: backward grads flow into the promotion of level ℓ (since the S that was there got promoted).
- **Case merge-promoted (ℓ = lssb+1)**: `S^(ℓ)_t = Σ_{ℓ'<ℓ} S^(ℓ')_{t-1} · diag(a_t)`. So backward dS^(ℓ)_t[h, dh, ds] contributes to dS^(ℓ')_{t-1}[h, dh, ds] · a_t[h], for all ℓ' < ℓ.
- **Case carry (ℓ > lssb+1)**: `S^(ℓ)_t = diag(a_t) · S^(ℓ)_{t-1}`. Backward dS^(ℓ)_t propagates to dS^(ℓ)_{t-1} · a_t[h].

The state gradient also needs the **decay-back** term: `d a_t[h]` gets += sum_{dh, ds} dS^(ℓ)_t[h, dh, ds] · S^(ℓ)_{t-1}[h, dh, ds] · a_t[h] · (1 - a_t[h]) for all ℓ (since a_t = sigmoid(a_proj_out)).

Wait — careful. The carry-forward applies `diag(a_t)`, so `S^(ℓ)_t = diag(a_t) · S^(ℓ)_{t-1}`. Then `dL/dS^(ℓ)_{t-1} += diag(a_t) · dL/dS^(ℓ)_t`, and `dL/d(a_t[h]) += Σ_{dh, ds} dS^(ℓ)_t[h, dh, ds] · S^(ℓ)_{t-1}[h, dh, ds]`.

For the immediate / cleared / merge-promoted cases, the new S is built FROM the inputs, so the backward flows through them. For the cleared case (S was zeroed), there is no carry and the level contributes nothing to the output (λ^(ℓ)_t's contribution was already consumed at t-1 and t-2, etc.). But we still need to track that S^(ℓ)_t = 0, so any downstream use has S^(ℓ)_t = 0; this matters for the merge-promoted case at t+1.

Actually for the recurrence, "cleared" means S^(ℓ)_t = 0 at this step — its contribution to o_t is 0 (since it multiplies q_t · 0 = 0). The gradient dS^(ℓ)_t at the cleared level is just 0 (since S^(ℓ)_t contributes nothing to o_t). At t+1, the promotion at level ℓ+1 will pull from levels < ℓ+1 (which includes level ℓ), so the gradient at the cleared level at step t+1 must come through the merge-promoted chain at level ℓ+1, level ℓ+2, etc. But since S^(ℓ)_t = 0 at step t (cleared), there's nothing to propagate backward from there.

Hmm wait. Let me think again. The merge at level ℓ+1 pulls from ALL levels < ℓ+1 at step t-1. So at step t (when level ℓ is cleared), the level ℓ+1's merge pulls from levels 0..ℓ at step t-1 (the previous step). The level ℓ at step t-1 had its own content. So at step t-1, level ℓ had non-zero content (unless t-1 was also a "cleared" tick for ℓ). This is getting recursive. We just track the gradient dS^(ℓ)_t at each step and route backward through the case that was active at that step.

For the cleared case: dS^(ℓ)_t is zero (no contribution to o_t) — backward gradient for level ℓ at step t is 0. So nothing flows back through the level ℓ at step t.

But wait: actually the cleared case means the level ℓ state is set to 0 at step t. The "merge-promoted" case at step t+1 pulls from levels < ℓ+1 at step t. If level ℓ at step t is 0, then it contributes nothing to the merge. But at step t+1 the merge includes level ℓ at step t (which is 0) — that's fine, contributes 0. So no backward flow issue.

OK so the BPTT is straightforward: at each (t, ℓ), determine which case was active in the forward pass (stored in cache), and apply the corresponding backward rule.

## Verification approach

Each test gets its own focused configuration (small T, small d, small n_heads) so FD checks are tractable. Random init for all weights and inputs (deterministic via seed). L2 loss for gradient checks.

## Pitfalls to watch

- **State shape**: Mamba-2 has per-head H of shape (d_head, d_state). For us, log-linear attention has L levels per head, each (d_head, d_state). Cache as a flat Tensor of shape `(T * L, d_inner, d_state)` (or `(T+1, L, d_inner, d_state)` flattened).
- **Fenwick tree "promotion" timing**: at t=0, lssb(1)=0, so level 0 is immediate, level 1+ are cleared. At t=1, lssb(2)=1, so level 0 is immediate, level 1 gets Σ_{ℓ'<1} S^(ℓ')_0 = S^(0)_0, levels 2+ are cleared. At t=3, lssb(4)=2, level 0 immediate, levels 1 cleared, level 2 gets Σ_{ℓ'<2} S^(ℓ')_2 = S^(0)_2 + S^(1)_2, levels 3+ cleared.
- **λ_proj output**: per token, per head, per level. Total shape `(T, n_heads * L)`. Or restructure to `(T, n_heads, L)`.
- **a_t decay is per-head scalar**: applies to all states at all levels uniformly (since a is per head).
- **Skip + gating path**: in_proj output is `(T, 2*d_inner)` — first d_inner is x_ssm, second is gate (post-silu). D_skip is `(1, d_inner)`.
- **Decay application timing**: in Mamba-2 we apply a_t to the carry-forward AND to the immediate path? Looking at the Mamba-2 reference, `H_t = diag(a_t) · H_{t-1} + b_t ⊗ k_t`, so the carry-forward applies diag(a_t). The immediate path doesn't apply a_t (it just contributes b_t ⊗ k_t). We follow the same convention.