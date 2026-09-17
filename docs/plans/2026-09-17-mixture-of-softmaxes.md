# Mixture of Softmaxes (MoS) Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Mixture-of-Softmaxes attention (Yang et al., ICLR 2018, https://arxiv.org/abs/1711.03953, "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model") — replaces a single softmax in attention with a learned mixture of K softmaxes. Each softmax branch has its own logit projection and value projection; a separate gating network mixes them. K=1 must bit-exactly recover standard scaled-dot-product attention.

**Architecture:** Standard MHA structure with K softmax branches per head:

For input `x ∈ R^{N × d_model}`, with `num_heads = H` and `K` mixture branches:

1. **Q/K projections (shared across mixture branches)** — `W_q, W_k ∈ R^{d_model × d_model}` per the paper, so `Q = x · W_q^T + b_q`, `K = x · W_k^T + b_k`. (Yang et al. share Q/K across branches.)
2. **Per-(head, mixture) value projections** — `W_v ∈ R^{K × d_model × d_model}` and `b_v ∈ R^{K × d_model}` — `V_{h,k} = x · W_v[h,k]^T + b_v[h,k]`. (Each branch has its own value projection — the source of the "softmax bottleneck" fix.)
3. **Per-(head, mixture) logit projection** — `W_z ∈ R^{K × d_model × d_model}` and `b_z ∈ R^{K × d_model}` — `Z_{h,k} = x · W_z[h,k]^T + b_z[h,k]` (shared Q/K for the dot product gives the same `Z`; per-branch `W_z` lets the branches see different slices if desired — but for v1 we share Z across branches, so a single `W_z` suffices).
4. **Per-head branch softmax** — `P_{h,k}[i,j] = softmax_i(Z[h,k,j] · K[h,k,i] / sqrt(d_h))` over positions i.
5. **Per-head branch output** — `O_{h,k}[j] = Σ_i P_{h,k}[i,j] · V_{h,k}[i]`. (This is the "high-rank output" path.)
6. **Per-head mixing weights from the query** — `Π_h[j] = softmax_k(W_g[h] · Q_h[j] + b_g[h])` over K branches. (Gating depends on Q_h, not Z, so it's query-conditioned.)
7. **Per-head final pre-W_o output** — `Y_h[j] = Σ_k Π_h[j,k] · O_{h,k}[j]`.
8. **Concat heads + W_o** — standard MHA closure.

Total parameters per attention layer:
- `W_q, W_k, b_q, b_k` (shared): 2 × d² + 2 × d = 2d² + 2d
- `W_v` (per-branch per-head): K × d² + K × d = K · (d² + d)
- `W_z` (shared since branches see same Q/K): d² + d (v1 simplification)
- `W_g, b_g` (gating): K × d + K = K · (d + 1)
- `W_o, b_o`: d² + d

For H=2, d=4, K=2: 2·(16+4) + 2·(16+4) + (16+4) + 2·(4+1) + (16+4) = 40 + 40 + 20 + 10 + 20 = **130**.

For K=1 we recover standard MHA exactly: `W_v[0]` is the standard `W_v`, `W_z` is the standard `W_q·W_k^T` reformulation, `W_g` is degenerate (one branch), so the `Π · O` collapse is a no-op. Test 4 verifies this bit-exact equivalence.

**Tech Stack:** C++17, `Tensor`, `Dense`, `LayerNorm`, hand-derived gradients (MoS chain through Z, P, V, Π).

## Design Decisions

- **Shared Q/K across mixture branches** (matches paper). This keeps `d_model²` projections instead of `K · d²` and the gradient chain through Q/K is standard softmax-backward. Branches differ only in `W_z` (a per-branch bias on Z), `W_v`, and the gating `Π`.
  - Actually: simpler v1 — let `Z` be SHARED across mixture branches (one `W_q·K`/`W_k^T` projection per head) and branches differ in (a) a per-branch additive bias on Z (`B_z[h,k,d]`, scalar per head-mixture-feature), and (b) `W_v[h,k]`, plus (c) the gating `Π`. This matches Yang et al. 2018 Eq. 7 exactly. **Decision: per-branch bias on Z, not per-branch W_z.**
- **No attention mask / causal in v1** — full bidirectional attention. Per-row softmax over all positions i. Same convention as the existing `StickBreakingAttention` tests which use causal masking there but bidirectional here is cleaner.
- **K ≥ 1**, integer. Constructor validates K ≥ 1. Default K=1 should be exactly MHA-recoverable.
- **d_model must be divisible by num_heads** — same as MHA convention. head_dim = d_model / num_heads.

## Phase 0 — Setup files

### Task 0.1: Create header `include/nn/layers/attention/mixture_of_softmaxes.h`

Three classes:

```cpp
class MixtureOfSoftmaxesAttention : public Layer {
public:
    MixtureOfSoftmaxesAttention(size_t d_model, size_t num_heads = 1, size_t K = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "MixtureOfSoftmaxesAttention"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t num_branches() const { return K_; }
    double inv_temp() const { return inv_temp_; }

    // Forward caches
    Dense W_q, W_k, W_o;  // shared (d_model, d_model)
    Tensor W_v;            // (K, d_model, d_model) — per-branch per-head V projection flattened
    Tensor B_v;            // (K, d_model)
    Tensor B_z;            // (K, d_model)  — per-branch additive bias on logits
    Tensor W_g;            // (K, d_model)  — gating weights (shared across heads — paper Eq. 7)
    Tensor B_g;            // (K, 1)         — gating bias

    Tensor grad_W_q, grad_W_k, grad_W_o;
    Tensor grad_W_v, grad_B_v, grad_B_z, grad_W_g, grad_B_g;

    const Tensor& last_P() const { return last_P_; }      // (H*K, N, N) per-branch attention maps
    const Tensor& last_Pi() const { return last_Pi_; }    // (H, N, K) gating weights
    const Tensor& last_O() const { return last_O_; }      // (H, N, d_h) pre-W_o output
private:
    // Forward cache + size_t scalars
};
```

Parameter count for H=2, d=4, K=2: 130 (computed above).

### Task 0.2: Create test file `tests/test_mixture_of_softmaxes.cpp`

Six test categories:
1. Constructor validation
2. Forward shape + finiteness
3. K=1 bit-exact MHA recovery
4. K=2 vs K=1 differs (signature that branches are active)
5. Input gradient FD (K=1 and K=2)
6. Param gradient FD (W_v, B_v, B_z, W_g, B_g) with random non-uniform init
7. zero_grad + update_weights + parameters()/gradients() contract
8. K=4 training reduces loss over N steps

---

## Phase 1 — TDD per task

### Task 1: Constructor + forward + finiteness

**Files:** tests/test_mixture_of_softmaxes.cpp, include/nn/layers/attention/mixture_of_softmaxes.{h,cpp}.

Tests:
- (a) Construction throws on `d_model=0`, `num_heads=0`, `K=0`, d_model not divisible by num_heads, d_model < num_heads.
- (b) Construction with (4, 1, 1), (4, 2, 1), (4, 2, 2), (8, 2, 4) all succeed.
- (c) Forward shape (N=5, d_model=4 → (5, 4)) finite, nonzero.
- (d) Forward shape (N=5, d_model=8, H=2, K=4 → (5, 8)) finite, nonzero.
- (e) last_P shape (H*K, N, N), last_Pi shape (H, N, K).
- (f) per-branch softmax rows sum to 1; per-branch gating rows sum to 1.

### Task 2: K=1 bit-exact MHA recovery

**Files:** same.

With K=1, the gating softmax produces `Pi[h, j, 0] = 1.0` for every (h, j). The forward should reduce to standard MHA: `O = softmax(QK^T / sqrt(d_h)) · V`, then concat + W_o.

Tests:
- Construct MixtureOfSoftmaxesAttention(d=4, H=2, K=1).
- Set W_q = some MHA-W_q, W_k = MHA-W_k, W_v = MHA-W_v (as the single branch slice), W_o = MHA-W_o, B_z = 0, W_g = arbitrary (unused), B_g = 0.
- Forward; compare against a hand-computed MHA reference (just dot products + softmax + matmul). Bit-exact to 1e-13.

This is the regression test that catches the "K=1 silently uses only branch 0" bug.

### Task 3: K=2 vs K=1 differs

Construct with K=2 vs K=1, identical non-degenerate params (B_v[1] = 0 so branch 1 contributes 0 to V but gating still nontrivial — or simpler: B_v[1] != B_v[0]). Forward outputs must differ at rel_err > 1e-3.

Catches "K parameter is unused" bugs.

### Task 4: Input gradient FD

Tests:
- (N=4, d=4, H=1, K=1): input grad FD rel_err < 1e-5.
- (N=4, d=4, H=2, K=2): input grad FD rel_err < 1e-4.
- (N=6, d=4, H=2, K=4): input grad FD rel_err < 1e-3.

### Task 5: Parameter gradient FD (random non-uniform init)

For each of: W_q.weights, W_k.weights, W_v, B_v, B_z, W_g, B_g, W_o.weights — FD check at rel_err < 1e-4. With `Tensor::random(scale=0.3)` non-uniform init to avoid row-vs-column confusion. For the K=1 case, W_g and B_g gradients should be exactly 0 (no branches to gate over) — that's the regression test.

### Task 6: zero_grad + update_weights + parameters()/gradients() contract

Verify counts match (H=2, K=2 → W_q, W_k, W_o (3) + W_v, B_v, B_z, W_g, B_g (5) = 8 param tensors, same for grads). update_weights changes all of them.

### Task 7: K=4 end-to-end training

Build a small encoder-style task: input (B=4, S=4, d_model=4), target (B=4, output=4). Train MoS(d=4, H=1, K=4) for 50 SGD steps at lr=0.01. Loss should drop by > 30%.

---

## Phase 2 — Wrap

- Add `#include "layers/attention/mixture_of_softmaxes.h"` to `include/nn/nn.h`.
- Add `build/test_mixture_of_softmaxes` rule and `=== Running Mixture of Softmaxes Tests ===` echo in Makefile.
- Add `build/test_mixture_of_softmaxes` to the `tests:` target.
- Confirm: build, run, all green, no warnings, no regressions in adjacent tests.