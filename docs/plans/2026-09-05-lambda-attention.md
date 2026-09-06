# Lambda Layer (Bello et al. 2021) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a hand-rolled `LambdaAttention` layer to `include/nn/layers/attention/lambda_layer.{h,cpp}` that implements the single-head version of Lambda Layer from "LambdaNetworks: Modeling Long-Range Interactions Without Attention" (https://arxiv.org/abs/2102.08602), with full analytical backward, FD-grade gradient checks, plus a `LambdaBlock` and `LambdaModel` stack for end-to-end training. Register in `include/nn/nn.h` umbrella and add a `build/test_lambda_layer` rule to the Makefile.

**Architecture:**
- `LambdaAttention(d_model, max_seq_len, k_depth=0, causal=false)` — single-head Lambda layer with input dim `d_model`, learned Q/K/V projections, position embeddings of shape `(max_seq_len, max_seq_len, k_depth)`, optional causal mask. Output shape `(n, d_model)`.
- `LambdaBlock(d_model, max_seq_len, ffn_dim=0, k_depth=0, causal=false)` — pre-LN → Lambda → residual → pre-LN → GELU FFN → residual, matching the pattern used by `GAUBlock`.
- `LambdaModel(d_model, max_seq_len, out_features, num_blocks, ffn_dim=0, k_depth=0, causal=false)` — stack of blocks + classifier, matching `GAUModel`.

**Tech Stack:** Existing C++17 / hand-rolled tensor library; layer convention matches the rest of `attention/`.

---

## Reference: the math (single-head, self-context C = X)

Given input X ∈ R^{n×d_model}:

1. **Project**: Q = X · W_Q ∈ R^{n×k}, K = X · W_K ∈ R^{n×k}, V = X · W_V ∈ R^{n×d_model}.
2. **Normalize keys** across positions: K̄[m, kk] = exp(K[m, kk]) / Σ_{m'} exp(K[m', kk]) (per-column softmax over positions, for each kk ∈ [0, k)).
3. **Content lambda** (shared across query positions): λ_c[kk, v] = Σ_m K̄[m, kk] · V[m, v] ∈ R^{k × d_model}.
4. **Position lambda** (per-query-position): λ_p[n, kk, v] = Σ_m E[n, m, kk] · V[m, v] ∈ R^{n × k × d_model}, where E ∈ R^{max_seq_len × max_seq_len × k} is a learnable relative-position-embedding tensor.
5. **Combine** (broadcast λ_c over n): λ[n, kk, v] = λ_c[kk, v] + λ_p[n, kk, v].
6. **Apply lambda to query**: Y[n, v] = Σ_kk Q[n, kk] · λ[n, kk, v] ∈ R^{n × d_model}.

For causal mode (single-head): zero out E[n, m, kk] for m > n BEFORE computing λ_p. (This still gives a well-defined, gradient-friendly computation — we just don't compute the future-position contribution.)

**Multi-query generalization (k_depth < d_model / u for u heads) is OUT OF SCOPE for v1** — single-head (k_depth = d_model, v = d_model) keeps the math clean and the test surface small. The constructor accepts `k_depth=0 → k_depth = d_model`.

---

## Why this layer is interesting for the repo

- It's a fundamentally different way of capturing long-range interactions from softmax attention: there is **no `softmax(Q·K^T)` and no attention map**, just two cheap aggregations (K̄^T V and E^T V) that share a query-side matmul `Q · λ`.
- Memory cost: Θ(n² · k) for E (positions) instead of Θ(n² · d) for an attention map. With k ≤ d, this is the dominant memory win.
- Computes per-position a `(k, d_model)` "lambda matrix" that contextualizes the query. Conceptually adjacent to **fast-weight programmers**, **perceiver cross-attention** (different shape), and **DELTA-net** (state-based). None of the existing repo layers cover this slot.

---

## Tasks

### Task 1: Write the .h file

**Files:**
- Create: `include/nn/layers/attention/lambda_layer.h`
- Will be modified later: `include/nn/nn.h` (umbrella include), `Makefile`, `tests/test_lambda_layer.cpp`.

**Step 1.1:** Define the header. Three classes: `LambdaAttention`, `LambdaBlock`, `LambdaModel`. Public parameters on `LambdaAttention`: `W_Q (d_model, k)`, `W_K (d_model, k)`, `W_V (d_model, d_model)`, `position_embeddings (max_seq_len, max_seq_len, k)` (stored as `Tensor position_emb_; // (max_seq_len, max_seq_len * k)` reshaped via strides in backward; see step 1.2). Plus matching `grad_*` tensors. Block has 2 LayerNorms + 2 dense weights. Model has a stack of unique_ptr<LambdaBlock> + classifier W/b.

**Concrete fields for `LambdaAttention`:**
```cpp
class LambdaAttention : public Layer {
public:
    LambdaAttention(size_t d_model, size_t max_seq_len, size_t k_depth = 0, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_Q; }
    Tensor get_gradients() const override { return grad_W_Q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LambdaAttention"; }

    size_t d_model() const { return d_model_; }
    size_t max_seq_len() const { return max_seq_len_; }
    size_t k_depth() const { return k_depth_; }
    bool causal() const { return causal_; }

    // Public for FD tests
    Tensor W_Q, W_K, W_V;
    Tensor grad_W_Q, grad_W_K, grad_W_V;
    Tensor position_emb_;        // (max_seq_len, max_seq_len * k)
    Tensor grad_position_emb_;  // (max_seq_len, max_seq_len * k)

    // Forward caches (mutable across forward calls)
    Tensor last_input_;          // (n, d_model)
    Tensor last_Q_, last_K_, last_V_;
    Tensor last_K_softmax_;      // (n, k) — softmaxed K
    Tensor last_lambda_content_; // (k, d_model)
    Tensor last_lambda_pos_;     // (n, k, d_model)
    Tensor last_lambda_total_;   // (n, k, d_model)
    Tensor last_output_;         // (n, d_model)  -- pre-final-clamp / unscaled

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t k_depth_;
    bool causal_;

    // Caches for backward
    Tensor last_E_used_;         // (n, n, k) — after causal masking (if any)
};
```

Use std::vector<Tensor*> for parameters() — there will be 4 (W_Q, W_K, W_V, position_emb_). Use 4 for gradients().

**Step 1.2:** Position-embedding storage choice. We store `position_emb_` as a `(max_seq_len, max_seq_len * k)` tensor (column-major reshape of the natural 3D `(max_seq_len, max_seq_len, k)`). Indexing: `position_emb_(n, m*k + kk) == E[n, m, kk]`. This keeps storage as 2D (matching the rest of the codebase), and we use plain 2-D iteration in forward/backward. Document this clearly in the header.

**Verify:** `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/layers/attention/lambda_layer.h"'` compiles cleanly (will fail until Task 3 creates the .cpp; that's fine — this task only creates the .h).

**Step 1.3:** Commit:
```bash
git add include/nn/layers/attention/lambda_layer.h
git commit -m "feat(attention): LambdaAttention header (Bello et al. 2021) — single-head Q/K/V + per-position lambdas"
```

---

### Task 2: Write the failing test file

**Files:**
- Create: `tests/test_lambda_layer.cpp`

**Tests to write (each must FAIL before the .cpp is implemented):**

1. Constructor validation: d_model=0 throws; max_seq_len=0 throws; k_depth=0 → defaults to d_model (passes); valid constructor doesn't throw.
2. Forward shape: input (n=4, d=4) → output (4, 4).
3. Forward finite: n=8, d=4 input → output all finite.
4. Causal mask signature: in causal mode, perturbing V[m] for m > n leaves Y[n] bit-exact unchanged (3 (n, m) pairs checked).
5. Hand-derived forward (n=2, d=2, k=2, forced weights, small ints) matches a Python-script-style manual recomputation at rel_err < 1e-5 (write the manual reference in C++ inside the test).
6. **Input gradient FD check** (n=3, d=3, eps=1e-5): max abs / max abs < 1e-3.
7. W_Q gradient FD check (rel_err < 1e-3).
8. W_K gradient FD check (rel_err < 1e-3).
9. W_V gradient FD check (rel_err < 1e-3).
10. position_emb gradient FD check (rel_err < 1e-3).
11. zero_grad clears all 4 gradients.
12. update_weights moves all 4 parameters.
13. LambdaBlock forward shape (n, d) → (n, d).
14. LambdaBlock input gradient FD check.
15. LambdaModel training reduces loss over 30 SGD steps.
16. parameters()/gradients() contract: 4 tensors, shape-matched.
17. **Mutation test**: zeroing `grad_W_V` accumulator in backward → input FD rel_err rises (proves the W_V path is exercised).
18. Longer sequence (T=6) input gradient FD check.
19. **Causal mode gradient FD check**: with causal=true, input FD vs analytical rel_err < 1e-3 (proves the masking path is correct).

**Helper functions to define:**
- `static double rel_err(double a, double b)` — same as GAU.
- `static double l2_loss_value(const Tensor&, const Tensor&)` — same as GAU.
- `static Tensor l2_loss_grad(const Tensor&, const Tensor&)` — same as GAU.

**Step 2.1:** Write the full test file. The check()/passed/failed pattern matches `test_gau.cpp`. Use the same static counters and signature.

**Step 2.2:** Try to build (will fail — no .cpp yet). Don't run it.

**Step 2.3:** Commit:
```bash
git add tests/test_lambda_layer.cpp
git commit -m "test: add 19 failing checks for LambdaAttention (Bello et al. 2021)"
```

---

### Task 3: Implement LambdaAttention::forward (RED → GREEN for Tests 1–5)

**Files:**
- Create: `include/nn/layers/attention/lambda_layer.cpp`

**Step 3.1:** Constructor body: validate d_model_/max_seq_len_ > 0; k_depth_ = (k_depth == 0) ? d_model_ : k_depth_; allocate tensors:
```cpp
W_Q = Tensor::random(d_model_, k_depth_, 0.02);
W_K = Tensor::random(d_model_, k_depth_, 0.02);
W_V = Tensor::random(d_model_, d_model_, 0.02);
position_emb_ = Tensor(max_seq_len_, max_seq_len_ * k_depth_);
// init position_emb_ small-random per (n, m, kk) — use std::mt19937 seeded with 0x1da
```

**Step 3.2:** `forward` body:
1. `last_input_ = input.clone();`
2. `Q = input · W_Q` (matmul: `(n, d_model) @ (d_model, k)` → `(n, k)`).
3. `K = input · W_K` → `(n, k)`.
4. `V = input · W_V` → `(n, d_model)`.
5. `K̄ = col_softmax(K)` → `(n, k)` (per-column softmax over the `n` axis; equivalent to row_softmax(K^T) and then transpose).
6. `λ_c = K̄^T · V` → `(k, d_model)`.
7. `λ_p = einsum('nmk,bmv->bnkv', E, V)`:
   - E_slice = last_E_used_ : (n, n, k) where E_slice[n, m, kk] = position_emb_(n, m*k + kk), and if causal_ then E_slice[n, m, kk] = 0 for m > n.
   - λ_p[n, kk, v] = Σ_m E_slice[n, m, kk] * V[m, v] → (n, k, d_model).
8. `λ_total[n, kk, v] = λ_c[kk, v] + λ_p[n, kk, v]` → (n, k, d_model).
9. `Y[n, v] = Σ_kk Q[n, kk] * λ_total[n, kk, v]` → (n, d_model).
10. Cache everything for backward. Return Y.

**Step 3.3:** `backward` body (gradient flow — this is the meaty part):

Given `grad_output ∈ R^{n × d_model}` (upstream gradient), compute gradients w.r.t. W_Q, W_K, W_V, position_emb_, and return `grad_input ∈ R^{n × d_model}`.

Work in `dL/dY → dL/dλ_total → dL/dλ_c, dL/dλ_p → dL/dQ (and grad_W_Q), dL/dK̄ → dL/dK → grad_W_K, dL/dV → grad_W_V, and dL/dλ_p → grad_position_emb_`.

Notation: `g = grad_output`. Compute `gY` = g.

**(a) dL/dQ from Y = Q · λ_total**:
`Y[n, v] = Σ_kk Q[n, kk] · λ_total[n, kk, v]`
→ `dL/dQ[n, kk] = Σ_v gY[n, v] · λ_total[n, kk, v]`
→ `grad_input_Q = (n, k)` matrix.
→ `grad_W_Q[i, kk] = Σ_n X[n, i] · dL/dQ[n, kk]` (use last_input_ here; careful: input is (n, d) and W_Q is (d, k), so Q[n, kk] = Σ_i X[n, i] · W_Q[i, kk] → dL/dW_Q[i, kk] = Σ_n X[n, i] · dL/dQ[n, kk]).

**(b) dL/dλ_total from Y = Q · λ_total**:
`dL/dλ_total[n, kk, v] = gY[n, v] · Q[n, kk]` → `(n, k, d_model)` matrix.
Then split:
- `dL/dλ_c[kk, v] = Σ_n dL/dλ_total[n, kk, v]` (shared across n).
- `dL/dλ_p[n, kk, v] = dL/dλ_total[n, kk, v]` (per-n).

**(c) dL/dV from λ_c = K̄^T · V**:
`λ_c[kk, v] = Σ_m K̄[m, kk] · V[m, v]`
→ `dL/dV[m, v] += Σ_kk K̄[m, kk] · dL/dλ_c[kk, v]`
= `dL/dV[m, :] = K̄[m, :] · dL/dλ_c` (matrix-vector per m).

**(d) dL/dV from λ_p**:
`λ_p[n, kk, v] = Σ_m E_slice[n, m, kk] · V[m, v]`
→ `dL/dV[m, v] += Σ_{n, kk} E_slice[n, m, kk] · dL/dλ_p[n, kk, v]`.

**(e) dL/dK̄ from λ_c**:
`dL/dK̄[m, kk] = Σ_v V[m, v] · dL/dλ_c[kk, v]` → `(n, k)` matrix.
Then `dL/dK_softmax_pre = softmax_backward(dL/dK̄, K̄)` (per-column softmax backprop).

**Softmax backward** (per-column over positions, for column kk):
- Input: `K̄[m, kk]` = exp(K[m, kk]) / Z[kk] where Z[kk] = Σ_m exp(K[m, kk]).
- Let `dL/dK̄_pre[m, kk] = K̄[m, kk] · (dL/dK̄[m, kk] − Σ_{m'} K̄[m', kk] · dL/dK̄[m', kk])`.
- Compute this in a loop.

**(f) dL/dW_K**:
`K = X · W_K` → `dL/dW_K[i, kk] = Σ_n X[n, i] · dL/dK_softmax_pre[n, kk]`.

**(g) dL/dW_V**:
`V = X · W_V` → `dL/dW_V[i, v] = Σ_n X[n, i] · dL/dV[n, v]`.

**(h) dL/dposition_emb_**:
`λ_p[n, kk, v] = Σ_m E_slice[n, m, kk] · V[m, v]`
→ `dL/dE_slice[n, m, kk] = Σ_v dL/dλ_p[n, kk, v] · V[m, v]`
→ `dL/dposition_emb_(n, m*k + kk) = dL/dE_slice[n, m, kk]` if !causal_ or m ≤ n; else 0.

**(i) dL/dX (grad_input)** — three contributions: from Q, from K (after softmax + linear), from V (after lambda + linear):
- From V: `dL/dX_from_V[n, i] = Σ_v dL/dV[n, v] · W_V[i, v]`
- From K: `dL/dX_from_K[n, i] = Σ_kk dL/dK_softmax_pre[n, kk] · W_K[i, kk]`
- From Q: `dL/dX_from_Q[n, i] = Σ_kk dL/dQ[n, kk] · W_Q[i, kk]`
- `grad_input[n, :] = dL/dX_from_Q + dL/dX_from_K + dL/dX_from_V`.

**Step 3.4:** Run tests 1–6. Expected: tests 1–5 pass; test 6 (input grad FD) — the most likely place to get the chain wrong — needs to hit rel_err < 1e-3.

If test 6 fails with a SPECIFIC factor (e.g., 0.5, 2.0, sign flip): the bug is usually (a) forgetting the dL/dK̄ → dL/dK_softmax_pre softmax chain, (b) forgetting one of the three grad_input contributions, (c) missing the contribution from `λ_c` vs `λ_p` in `dL/dλ_total`, or (d) missing the dL/dλ_c aggregation step. Run all three contributions in isolation (set the other two to zero) to bisect.

**Step 3.5:** Commit when tests 1–6 are green:
```bash
git add include/nn/layers/attention/lambda_layer.cpp
git commit -m "feat(attention): LambdaAttention forward + backward (single-head, content+position lambdas)"
```

---

### Task 4: W_Q, W_K, W_V, position_emb_ gradient FD checks (Tests 7–10)

**Already implemented in Step 3.3 (the FD check at the test level just verifies the grad tensors are populated correctly).** Run them, confirm rel_err < 1e-3 on each. If any fail, the bug is in the corresponding grad_W_* accumulation loop in Step 3.3 — print the analytical and FD values for the failing entries and compare manually.

Commit:
```bash
git add tests/test_lambda_layer.cpp
git commit -m "test: W_Q, W_K, W_V, position_emb_ gradient FD checks for LambdaAttention"
```

---

### Task 5: zero_grad, update_weights, parameters()/gradients() contract (Tests 11, 12, 16)

**Step 5.1:** Implement `zero_grad`: zero all 4 grad tensors.
**Step 5.2:** Implement `update_weights(lr)`: subtract `lr * grad_*` from each W_* and position_emb_ (in-place).
**Step 5.3:** Implement `parameters()` returning `{&W_Q, &W_K, &W_V, &position_emb_}`; `gradients()` returning `{&grad_W_Q, &grad_W_K, &grad_W_V, &grad_position_emb_}`. Each Tensor is a public member so the pointer is stable.

**Verify:** Tests 11, 12, 16 pass. Commit:
```bash
git add include/nn/layers/attention/lambda_layer.cpp
git commit -m "feat(attention): LambdaAttention zero_grad / update_weights / parameter contract"
```

---

### Task 6: LambdaBlock (Tests 13, 14)

**Files:**
- Modify: `include/nn/layers/attention/lambda_layer.h` (add class)
- Modify: `include/nn/layers/attention/lambda_layer.cpp` (implement)

**Architecture** (mirror `GAUBlock`):
```cpp
class LambdaBlock : public Layer {
public:
    LambdaBlock(size_t d_model, size_t max_seq_len, size_t ffn_dim = 0, size_t k_depth = 0, bool causal = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return attn.W_Q; }
    Tensor get_gradients() const override { return attn.grad_W_Q; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "LambdaBlock"; }

    LambdaAttention attn;
    LayerNorm ln1, ln2;
    Tensor W_ffn1_, b_ffn1_, W_ffn2_, b_ffn2_;
    Tensor grad_W_ffn1_, grad_b_ffn1_, grad_W_ffn2_, grad_b_ffn2_;

private:
    size_t d_model_, ffn_dim_;
    Tensor last_x_, last_z1_, last_attn_out_, last_res1_, last_z2_, last_h_pre_, last_h_act_, last_ffn_out_;
};
```

**Forward:**
- `ln1(x)` → `z1`
- `attn(z1)` → `attn_out` (residual addition: `res1 = z1 + attn_out`)
- `ln2(res1)` → `z2`
- FFN: `h_pre = z2 · W_ffn1 + b_ffn1 ; h_act = GELU(h_pre) ; ffn_out = h_act · W_ffn2 + b_ffn2`
- `output = res1 + ffn_out`
- Cache everything.

**Backward:** chain through FFN → add → ln2 backward → add → ln1 backward → attn backward. Use the same recipe as GAUBlock / Hybrid pattern.

**Verify:** Tests 13, 14 pass. Commit:
```bash
git add include/nn/layers/attention/lambda_layer.{h,cpp}
git commit -m "feat(attention): LambdaBlock (pre-LN → Lambda → residual → FFN)"
```

---

### Task 7: LambdaModel (Test 15)

**Files:** Modify .h and .cpp.

**Architecture:**
```cpp
class LambdaModel : public Layer {
public:
    LambdaModel(size_t d_model, size_t max_seq_len, size_t out_features, size_t num_blocks, size_t ffn_dim = 0, size_t k_depth = 0, bool causal = false);
    // standard forward/backward/update_weights/zero_grad/parameters/gradients/name
    std::vector<std::unique_ptr<LambdaBlock>> blocks;
    Tensor classifier_W_, classifier_b_;
    Tensor grad_classifier_W_, grad_classifier_b_;
private:
    Tensor last_input_, last_block_output_;
};
```

**Forward:**
- `x = input`
- for each block: `x = block->forward(x)`
- `logits = x · classifier_W + classifier_b` (shape (n, out_features))

**Backward:** chain through classifier → (no final LN here, matching GAUModel) → stack of blocks in reverse.

**Training test:** build model, run 30 SGD steps with lr=0.01 on a synthetic (n=4, d=4) → (4, 3) regression task, check loss decreases > 20%.

Commit:
```bash
git add include/nn/layers/attention/lambda_layer.{h,cpp}
git commit -m "feat(attention): LambdaModel (stack + classifier) — 30 SGD-step training sanity"
```

---

### Task 8: Mutation test (Test 17)

**Files:** Modify `tests/test_lambda_layer.cpp`.

**The mutation:** Stub out `grad_W_V` accumulation in `backward` (e.g. wrap the loop with `if (false) { /* grad_W_V loop */ }`). Re-run the test suite. Expected: Test 6 (input grad FD) now fails because `grad_input_from_V` is zeroed. If it does fail with rel_err > 1e-1, the test is exercising the W_V path → un-stub → re-run → confirm green.

Commit:
```bash
git add tests/test_lambda_layer.cpp
git commit -m "test: mutation test for grad_W_V — proves W_V gradient path is exercised"
```

---

### Task 9: Longer-sequence FD check (Test 18)

**Files:** Modify `tests/test_lambda_layer.cpp`.

n=6, d=3 → input FD rel_err < 1e-3. This stresses the per-position lambda path more than n=3. If it fails, the bug is in the dL/dλ_p → dL/dV aggregation step (off-by-one in the m summation).

Commit:
```bash
git add tests/test_lambda_layer.cpp
git commit -m "test: longer-sequence (T=6) input gradient FD check for LambdaAttention"
```

---

### Task 10: Causal mode FD check (Test 19)

**Files:** Modify `tests/test_lambda_layer.cpp` and `include/nn/layers/attention/lambda_layer.cpp`.

In causal mode, the masking path zeros E[n, m, :] for m > n BEFORE the λ_p computation. The backward must respect the same mask (dL/dposition_emb_[n, m*k+kk] = 0 for m > n).

Implement the mask in forward: build `E_used_[n, m*k+kk] = position_emb_[n, m*k+kk] if m <= n else 0`. Use `E_used_` (cached) in both forward (λ_p computation) and backward (grad_position_emb_ accumulation). This is cleaner than re-deriving the mask condition twice.

Verify Test 19 passes (input FD rel_err < 1e-3 in causal mode, n=3, d=3). Also verify the causal signature (Test 4) still passes.

Commit:
```bash
git add tests/test_lambda_layer.cpp include/nn/layers/attention/lambda_layer.cpp
git commit -m "feat(attention): causal masking path — both forward and backward respect the m > n mask"
```

---

### Task 11: Umbrella header + Makefile registration

**Files:**
- Modify: `include/nn/nn.h` (add `#include "layers/attention/lambda_layer.h"` after `gau.h`)
- Modify: `Makefile` (add `$(BUILD_DIR)/test_lambda_layer` rule, add it to `tests:` deps line, add `=== Running Lambda Layer Tests ===` echo in `run_tests`)

**Step 11.1:** Standalone umbrella compile check:
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: clean compile (no warnings, no errors).

**Step 11.2:** Build the test:
```bash
make build/test_lambda_layer
```
Expected: clean build.

**Step 11.3:** Run the test:
```bash
make run_tests 2>&1 | grep -A 30 "Lambda Layer"
```
Expected: all 19 checks pass, "Summary: 19 passed, 0 failed".

**Step 11.4:** Run a quick no-regression check on the closest sibling test suites:
```bash
./build/test_gau && ./build/test_mogrifier_lstm && ./build/test_based && ./build/test_nystrom_attention
```
Expected: all green.

**Step 11.5:** Commit:
```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register LambdaAttention in umbrella header + Makefile"
```

---

### Task 12: Implementation plan + final cleanup

**Files:**
- Modify: `EXPANSION_QUEUE.md` — move the entry from `## Ideas` to `## Done`.
- Create: a short implementation-plan summary committed to the queue's `## Done` section (matches the pattern of the other entries: paper reference, classes, params, test counts, mutation-test result).

**Step 12.1:** Add the entry to `## Done` with the standard structure (paper link, class list, parameter count, test counts and pass rates, mutation-test summary, plan-file reference to `docs/plans/2026-09-05-lambda-attention.md`).

**Step 12.2:** Commit:
```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark LambdaAttention as Done in EXPANSION_QUEUE"
```

**Step 12.3:** Push:
```bash
git push origin master
```

---

## Bite-size

12 tasks × ~5 minutes each = ~60 minutes wall-clock. Each task produces a self-contained commit.

## Reference

- Paper: https://arxiv.org/abs/2102.08602 (Bello, Hou, Fed友谊, Xiong, Zoph, Shlens, Cui 2021, "LambdaNetworks: Modeling Long-Range Interactions Without Attention").
- Repo sibling templates: `gau.{h,cpp}` (262-line header, 586-line .cpp), `linformer.cpp` (545-line .cpp with `row_softmax` helper).

## Risks

- **Off-by-factor**: most likely bug is missing the dL/dK̄ → dL/dK_softmax_pre softmax chain in backward. Diagnostic: print analytical grad_input vs FD for one entry; if they differ by a constant factor, suspect softmax chain; if by a sign, suspect a `+` vs `-`; if by 0.5 or 2, suspect a missing `1/n` or `2x`.
- **Wrong m-index in causal mask**: causal signature test (Test 4) catches `Y[n] changing when V[m] for m > n changes`; this is the strongest bug detector.
- **FD test too tight**: use `eps=1e-5` for FD; if a parameter has very small magnitudes, the FD noise dominates. The repo's existing convention (rel_err < 1e-3) is forgiving enough for stable gradients.
- **Mutation test vacuous**: zeroing `grad_W_V` and seeing test 6 fail is the proof that grad_input has a V-path. If the mutation doesn't break the test, the test wasn't really testing what we thought.

## Handoff

After all 12 tasks, plan is done. Use subagent-driven-development if the orchestrator wants each task dispatched with two-stage review (spec compliance then code quality). For the self-driven variant, the cron job can just execute tasks in order, run tests after each task, and commit atomically.
