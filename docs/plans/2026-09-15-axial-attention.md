# Axial Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement Axial Attention (Ho et al. 2019 / Wang et al. 2020) for the `neural_net` C++ ML repo, exposing row-axis + column-axis attention for `(H*W, d)` sequences with per-axis Q/K/V projections and a residual combine.

**Architecture:** Two-axis attention that splits a flat `(H*W, d)` sequence into an `H × W` grid and runs (1) row attention over the W tokens of each H-row, then (2) column attention over the H tokens of each W-column. Each axis has its own Q/K/V/O projections; the two outputs are summed (a parameter-free fuse, matching Ho et al. 2019 §3.1 and the Axial Transformer codebase convention). Causal mask is applied independently per axis. For video (3D), an additional depth-axis pass can be layered on top — out of scope for v1 (see Decisions §2.3).

**Tech Stack:** C++17, hand-rolled matmul-style loops on `Tensor`, `Dense` for projections, `LayerNorm` for pre-LN blocks.

## References

- Ho, Kalchbrenner, Weissenborn, Salimans 2019, "Axial Attention in Multidimensional Transformers" (https://arxiv.org/abs/1912.12180) — original axial attention for image and video.
- Wang, Li, Khabsa, Lin, Ma, Radev 2020, "Linformer: Self-Attention with Linear Complexity" — concurrent linear-complexity transformer paper, included here for context only.
- Bello, Du, Lin, Shlens, Zoph 2023, "Attention Augmented Convolution Networks" — practical instantiation using axial attention.

## Two-axis design (H, W)

```
Input: (H*W, d_model)  -- laid out row-major as X[i*W + j, :], i ∈ [0,H), j ∈ [0,W)
        ↓ reshape (no copy) → (H, W, d_model)
Row-axis attention:    for each row i ∈ [0,H), attend over j ∈ [0,W)
                       → out_row has shape (H, W, d_model), with per-j summary
Column-axis attention: for each column j ∈ [0,W), attend over i ∈ [0,H)
                       → out_col has shape (H, W, d_model), with per-i summary
Fuse:                  out = out_row + out_col + input  (residual)
                       → reshape back to (H*W, d_model)
```

For each axis, the Q/K/V projections live in their own parameter set (`W_q_row, W_k_row, W_v_row, W_o_row` and `W_q_col, W_k_col, W_v_col, W_o_col`). The two attention passes are independent (no parameter sharing). Per-axis mask is applied additively to `QK^T / sqrt(d_head)`. The row-axis mask is causal along the W direction (j₁ > j₂ → drop for row i). The column-axis mask is causal along the H direction (i₁ > i₂ → drop for column j).

For `H=1` or `W=1` the corresponding axis is a degenerate row/col of length 1 → attention over a single key is `softmax(scores) = [1]`, which is a no-op projection — covered by the FD test. For `H=W=1` the layer is the identity (plus bias).

---

## Task 1: Header file `axial_attention.h` — class skeletons

**Files:** Create `include/nn/layers/attention/axial_attention.h`.

Define three classes:

```cpp
class AxialAttention : public Layer {
public:
    // d_model:        input/output feature dim
    // num_heads:      number of heads (MHA-style; for v1 use 1 head per axis)
    // H_, W_:         2D grid dimensions; input.rows() MUST equal H*W
    // causal:         apply per-axis causal mask (default true)
    AxialAttention(size_t d_model, size_t H, size_t W,
                   size_t num_heads = 1, bool causal = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q_row; }
    Tensor get_gradients() const override { return grad_W_q_row; }
    std::string name() const override { return "AxialAttention"; }

    // Accessors
    size_t d_model() const  { return d_model_; }
    size_t H() const        { return H_; }
    size_t W() const        { return W_; }
    size_t num_heads() const { return num_heads_; }
    bool causal() const     { return causal_; }

    // Public tensors (for test access & FD perturbation)
    Tensor W_q_row, W_k_row, W_v_row, W_o_row;   // (d_model, d_model) each
    Tensor W_q_col, W_k_col, W_v_col, W_o_col;   // (d_model, d_model) each
    Tensor b_row, b_col;                          // (1, d_model) each — pre-output bias
    Tensor grad_W_q_row, grad_W_k_row, grad_W_v_row, grad_W_o_row;
    Tensor grad_W_q_col, grad_W_k_col, grad_W_v_col, grad_W_o_col;
    Tensor grad_b_row, grad_b_col;

private:
    size_t d_model_, H_, W_, num_heads_, head_dim_;
    bool causal_;
    double scale_;

    // BPTT cache
    Tensor last_input_;        // (H*W, d_model)
    Tensor last_q_row, last_k_row, last_v_row;  // (H*W, d_model)
    Tensor last_q_col, last_k_col, last_v_col;
    Tensor last_attn_row, last_attn_col;        // (H*W, H*W) flattened, per axis
    Tensor last_head_out_row, last_head_out_col;
    Tensor last_out_row, last_out_col;          // (H*W, d_model) post-attention pre-proj
    Tensor last_d_input_;       // (H*W, d_model)
};

class AxialAttentionBlock : public Layer {
    // pre-LN → AxialAttention → residual → pre-LN → GELU FFN → residual
public:
    AxialAttentionBlock(size_t d_model, size_t H, size_t W,
                        size_t num_heads = 1,
                        size_t ffn_dim = 0,
                        bool causal = true);
    // ...
};

class AxialAttentionModel : public Layer {
    // input projection → N blocks → final LN → classifier
public:
    AxialAttentionModel(size_t d_input, size_t d_model, size_t d_output,
                        size_t H, size_t W,
                        size_t num_blocks = 2,
                        size_t num_heads = 1,
                        size_t ffn_dim = 0,
                        bool causal = true);
    // ...
};
```

**Constructor validation:**
- `d_model == 0` → throw
- `H == 0 || W == 0` → throw
- `d_model % num_heads != 0` → throw
- Constructor must initialize all 10 weight tensors (8 projections + 2 biases) with `Tensor::random(scale=0.02)`.

**Step 1:** Write the header.

**Step 2:** Compile-only check: `g++ -std=c++17 -Iinclude -fsyntax-only -x c++ - <<<'#include "nn/layers/attention/axial_attention.h"'` — confirm syntax is OK. Expected: silent success.

**Step 3:** Commit.

```bash
git add include/nn/layers/attention/axial_attention.h
git commit -m "feat(attention): AxialAttention header (3 classes, constructor validation)"
```

---

## Task 2: Test file scaffold + constructor test (Test 1)

**Files:** Create `tests/test_axial_attention.cpp`.

Include the header. Define helpers: `relative_error(a, b)`, `l2_loss_value(out, tgt)`, `l2_loss_grad(out, tgt)`. Main starts with Test 1: constructor validation for all three classes.

**Test 1 cases (AxialAttention):**
- `d_model=0` throws `std::invalid_argument`
- `H=0` throws
- `W=0` throws
- `d_model=4, num_heads=3` (4 % 3 != 0) throws
- Valid `(d=8, H=3, W=4, Hq=2)` constructs without throw; accessors return correct values; `parameters()` returns 10 tensors; `gradients()` returns 10 tensors.

**Test 2 cases (AxialAttentionBlock):**
- `d_model=0, H=2, W=2` throws
- `H=0` throws
- `W=0` throws
- Valid `(d=8, H=2, W=3, ffn=16)` constructs; `name()` returns "AxialAttentionBlock"; `parameters()` returns `1 (AxialAttention) + 2 (LN) + 4 (ffn) = 7` tensors.

**Test 3 cases (AxialAttentionModel):**
- `d_input=0` throws
- `d_model=0` throws
- `num_blocks=0` throws
- Valid `(d_input=4, d_model=8, d_output=3, H=2, W=3, num_blocks=2)` constructs; `name()` returns "AxialAttentionModel"; `parameters()` returns `2 + 2 + num_blocks * 7 = 18` tensors.

**Step 1:** Write the failing test (constructor validation + accessors).

**Step 2:** Run `make build/test_axial_attention` — expect link failure (no impl yet).

**Step 3:** Add a stub `.cpp` with empty implementations that just compile. Don't commit the stub.

**Step 4:** Run `build/test_axial_attention` — expect 3 passes (or whatever the stubs let pass).

**Step 5:** Commit.

```bash
git add tests/test_axial_attention.cpp
git commit -m "test(attention): AxialAttention constructor + accessors (Tests 1-3)"
```

---

## Task 3: AxialAttention forward + finiteness (Tests 4-6)

**Files:** `include/nn/layers/attention/axial_attention.cpp`.

Implement forward:
1. Validate `input.rows == H * W` and `input.cols == d_model`.
2. Cache `last_input_ = input.clone()`.
3. Compute `Q_row, K_row, V_row = input @ W_q_row, input @ W_k_row, input @ W_v_row` (shape (H*W, d_model) each). Add `b_row` after the output projection only (not on Q/K/V — matches ExpireSpanAttention convention).
4. Build the row-attention mask:
   - For token `(i, j) ∈ [0,H) × [0,W)` and key `(i', j')`:
     - if `i' != i`: drop (different row, not part of row-axis attention)
     - if `causal && j' > j`: drop
     - else: keep
   - Mask is `M_row[i*W+j, i'*W+j'] = 0 if keep else -1e9`.
5. Compute row-axis attention:
   - `scores_row = Q_row @ K_row^T / sqrt(d_head)`  shape `(H*W, H*W)`
   - Add row mask
   - `A_row = row_softmax(scores_row)`
   - `head_out_row = A_row @ V_row`  shape `(H*W, d_model)`
   - `out_row_pre = head_out_row @ W_o_row`  shape `(H*W, d_model)`
6. Repeat for column axis with the column mask:
   - For token `(i, j)` and key `(i', j')`: drop if `j' != j`, drop if `causal && i' > i`, else keep.
   - Compute `out_col_pre` similarly.
7. `output = out_row_pre + out_col_pre + input`  (residual sum, plus the residual is the input — no bias added separately because each axis already has `b_row`/`b_col` after the output projection; actually NO bias to keep simple — drop `b_row`/`b_col` for v1).

Update header to remove `b_row`/`b_col` if you decide to drop them in implementation. Pick: drop biases; the output projection has no bias; the residual sum is the fusion.

Actually — keep biases (cleaner fusion), but document that they enter as `+ b_row` and `+ b_col` additively in `output = (head_out_row @ W_o_row + b_row) + (head_out_col @ W_o_col + b_col) + input`. This is fine.

8. `last_attn_row, last_attn_col, last_head_out_row, last_head_out_col, last_q_row, last_k_row, last_v_row, last_q_col, last_k_col, last_v_col, last_out_row, last_out_col` — all populated for backward.

**Test 4: Forward shape** — `(H=3, W=4, d=8, Hq=2)` input `(12, 8)` → output `(12, 8)`. Finite (no NaN/Inf). Nonzero.

**Test 5: H=1, W=1** (degenerate single token) — input `(1, d)` → output `(1, d)` finite, equals `input + 2 * (b_row + b_col) + 0` projected through W_o on a zero attention. Just check finiteness + nonzero if biases are nonzero.

**Test 6: W=1 column-only path** — `H=4, W=1, d=8` input `(4, 8)` → output `(4, 8)` finite. This tests the column-axis path runs alone when the row-axis is degenerate.

**Step 1:** Implement forward.

**Step 2:** Run `build/test_axial_attention` — expect Tests 4-6 pass; Tests 1-3 should still pass.

**Step 3:** Mutation: zero out `W_o_row` and verify `output != 0` (proves the column axis and residual still flow even if the row projection is dead). Add this as a sanity check (don't promote to a separate test, just print).

**Step 4:** Commit.

```bash
git add include/nn/layers/attention/axial_attention.cpp
git commit -m "feat(attention): AxialAttention forward (row + column axis, residual fuse)"
```

---

## Task 4: Mask signature tests (Tests 7-10)

The mask is the distinctive test of axial attention — must verify rows and columns see only their axis.

**Test 7: row-axis mask** — read `last_attn_row` after forward on `(H=3, W=4)`. For each row i, every token `(i', j')` with `i' != i` must have row-axis attention contribution `0` (because the QK product for that key was -1e9 and softmax killed it). Verify: `max(|A_row[i*W+j, i'*W+j']|) < 1e-9` whenever `i' != i`.

**Test 8: column-axis mask** — same for `last_attn_col`: `max(|A_col[i*W+j, i'*W+j']|) < 1e-9` whenever `j' != j`.

**Test 9: causal mask** — `causal=true`. For row-axis: `A_row[i*W+j, i*W+j'] = 0` when `j' > j`. For column-axis: `A_col[i*W+j, i'*W+j'] = 0` when `i' > i`.

**Test 10: non-causal** — `causal=false`. For a query at `(i, j)` in column axis, every key `i' < i` at column j has nonzero attention. Verify at least one `A_col[i*W+j, (i-1)*W+j] > 0` (using seeded small input that gives a deterministic nonzero).

**Step 1:** Add these tests.

**Step 2:** Run. Expect all pass.

**Step 3:** Commit.

```bash
git add tests/test_axial_attention.cpp
git commit -m "test(attention): AxialAttention mask signatures (row/col/causal)"
```

---

## Task 5: AxialAttention backward (Tests 11-14)

Implement backward:
1. `d_output` shape `(H*W, d_model)`.
2. Split: `d_out_row = d_output`, `d_out_col = d_output` (each axis carries the full gradient initially — the gradient w.r.t. the residual sum is split equally between the two axes).

   Actually — for correctness, both axes receive `d_output` directly (since `output = out_row_pre + out_col_pre + input`, both `d_out_row_pre = d_output` and `d_out_col_pre = d_output`).

3. **For each axis independently** (same algorithm as a standard multi-head attention backward — reference ExpireSpanAttention lines 290-419):
   - `d_head_out = d_out_axis @ W_o^T`  shape `(H*W, d_model)`
   - `grad_W_o[k, j] += sum_t d_out_axis[t, j] * head_out_axis[t, k]`
   - For each head h:
     - `d_V_h = A^T @ d_head_out_h`
     - `d_A_h = d_head_out_h @ V_h^T`
     - `d_pre_soft = A * (d_A - row_sum(d_A * A))`  (with row-sum already absorbing the masked positions, which are 0 in A)
     - `d_Q_h = scale * d_pre_soft @ K_h`
     - `d_K_h = scale * d_pre_soft^T @ Q_h`
   - Accumulate into `d_Q_acc, d_K_acc, d_V_acc` and per-axis `grad_W_q, grad_W_k, grad_W_v`.
   - Note: row-axis and column-axis each have their own Q/K/V/O. So `grad_W_q_row, grad_W_k_row, grad_W_v_row` accumulate only the row-axis gradient; same for `*_col`. The two passes don't share parameters.

4. **Propagate through input projection** (since Q/K/V = input @ W_q/k/v):
   - `d_input_from_row = d_Q_row @ W_q_row^T + d_K_row @ W_k_row^T + d_V_row @ W_v_row^T`
   - `d_input_from_col = d_Q_col @ W_q_col^T + d_K_col @ W_k_col^T + d_V_col @ W_v_col^T`
   - `d_input_total = d_input_from_row + d_input_from_col + d_output`  (the residual term adds `d_output` itself)
   - `grad_W_q_row[k, j] += sum_t input[t, k] * d_Q_row[t, j]`  (and similar for K, V)
   - Same for column axis.
   - `grad_b_row[j] += sum_t d_out_axis[t, j]`

**Test 11: FD input gradient** — `Tensor input(H*W, d);` random init. `attn.forward(input)` → `attn.backward(l2_loss_grad(...))` → compare `attn.last_d_input_` to FD with `eps=1e-5`. `rel_err < 1e-4` with random non-uniform init (mandatory — uniform init masks row-vs-column confusion in the matmul backward).

**Test 12: FD W_q_row, W_k_row, W_v_row, W_o_row** — `rel_err < 1e-5`.

**Test 13: FD W_q_col, W_k_col, W_v_col, W_o_col** — `rel_err < 1e-5`.

**Test 14: FD b_row, b_col** — `rel_err < 1e-5`.

**Step 1:** Implement backward.

**Step 2:** Add tests.

**Step 3:** Run. Expect 14 tests pass. (If any fails, follow systematic-debugging.)

**Step 4:** Commit.

```bash
git add include/nn/layers/attention/axial_attention.cpp tests/test_axial_attention.cpp
git commit -m "feat(attention): AxialAttention backward + FD gradient tests"
```

---

## Task 6: Block + Model forward (Tests 15-17)

Implement `AxialAttentionBlock::forward`:
- Pre-LN: `z1 = LN1(input)`
- `attn_out = axial_attention.forward(z1)`
- `z2 = LN2(input + attn_out)`  (or `LN2(attn_out + input)` — either; pick `z2 = LN2(input + attn_out)`)
- If `ffn_dim > 0`:
  - `ffn_h = GELU(ffn_fc1(z2))`
  - `ffn_out = ffn_fc2(ffn_h)`
  - `output = input + attn_out + ffn_out`
- Else:
  - `output = input + attn_out`
- Cache: `last_input_, last_z1_, last_attn_out_, last_res1_, last_z2_, last_ffn_pre_, last_ffn_hidden_, last_ffn_out_, last_d_input_`.

**AxialAttentionModel::forward**:
- `proj = W_in_(input)`
- For each block: `proj = block.forward(proj)`
- `final = LN_final(proj)`
- `output = W_out_(final)`  — but only the mean over the H*W tokens or the first token? For v1 use mean pooling (avoids needing a CLS embedding), output shape `(N, d_output)`.

**Test 15:** `AxialAttentionBlock.forward((H*W, d))` returns `(H*W, d)` finite nonzero. With `ffn_dim=0` and `ffn_dim=16`.

**Test 16:** `AxialAttentionModel.forward((H*W, d_input))` returns `(1, d_output)` (after mean pool) — wait, the model takes a single flat input `(H*W, d_input)` and outputs `(1, d_output)`. No batch dim in this repo's convention.

Actually look at ExpireSpanModel: forward takes `(n, d_input)` and returns `(n, d_output)`. For axial, the input is `(H*W, d_input)` and output should also be `(H*W, d_output)` if we don't pool. Decide: **don't pool**. Output is `(H*W, d_output)` — per-token classification / regression head. Simpler and consistent with the repo.

**Test 16 revised:** `(H=2, W=3, d_input=4, d_model=8, d_output=3, num_blocks=2)` forward `(6, 4)` → `(6, 3)` finite.

**Test 17:** `AxialAttentionBlock::backward` + FD input gradient at `rel_err < 1e-3` (looser threshold because of LN+FFN chain).

**Step 1:** Implement Block + Model.

**Step 2:** Add tests.

**Step 3:** Run.

**Step 4:** Commit.

```bash
git add include/nn/layers/attention/axial_attention.cpp tests/test_axial_attention.cpp
git commit -m "feat(attention): AxialAttentionBlock + AxialAttentionModel + tests"
```

---

## Task 7: Training + integration tests (Tests 18-19)

**Test 18:** `AxialAttentionModel` end-to-end training reduces MSE loss > 30% in 60 SGD steps at `lr=0.01` on a synthetic task (e.g., predict row index from column index).

**Test 19:** `update_weights` moves all 10 parameter groups (max diff > 1e-10).

**Step 1:** Add tests.

**Step 2:** Run.

**Step 3:** Commit.

```bash
git add tests/test_axial_attention.cpp
git commit -m "test(attention): AxialAttention end-to-end training + update_weights"
```

---

## Task 8: Register in umbrella + Makefile

**Files:**
- Modify `include/nn/nn.h`: add `#include "layers/attention/axial_attention.h"` after `expire_span.h`.
- Modify `Makefile`:
  - Add `$(BUILD_DIR)/test_axial_attention: $(LIB_OBJS) $(BUILD_DIR)/test_axial_attention.o` + link rule.
  - Add `$(BUILD_DIR)/test_axial_attention` to the `tests:` deps list.
  - Add `@echo "=== Running Axial Attention Tests ===" && ./$(BUILD_DIR)/test_axial_attention` in `run_tests`.

**Step 1:** Edit.

**Step 2:** Verify umbrella compiles standalone:
```bash
g++ -std=c++17 -Iinclude -fsyntax-only -x c++ - <<<'#include "nn/nn.h"'
```
Expected: silent success.

**Step 3:** Run `make tests` and confirm the new test runs and passes.

**Step 4:** Commit.

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register AxialAttention in umbrella nn.h and Makefile"
```

---

## Task 9: Cleanup + docs

- Delete `*.o` stray files in repo root (gitignored but bloat).
- Update `EXPANSION_QUEUE.md`: move "3D Attention / Axial Attention" bullet from the Ideas section to the Done section with a one-paragraph summary referencing the test counts and where it lives.

**Step 1:** Clean up.

**Step 2:** Commit.

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark Axial Attention as Done in EXPANSION_QUEUE"
```

**Step 3:** `git push origin master`.

---

## Decisions

1. **Two-axis interface only for v1.** Three-axis (video / volumetric) extension via a depth-axis pass lives in §2.3.
2. **No GQA in v1.** All heads are MHA-style. Adding `num_kv_heads` later is straightforward.
3. **Per-axis causal independently.** Row-axis causal along W, column-axis causal along H. Same `causal` flag toggles both.
4. **Biases on output projections only.** Q/K/V projections have no bias (Llama/Mistral style).
5. **Residual fuse.** `output = out_row_pre + out_col_pre + input`. No learnable combine (Ho et al. 2019 §3.1).

## Out of scope (v1)

- 3D / video axial attention (depth-axis pass)
- GQA / MQA per-axis
- Relative position bias along each axis (paper §3.2)
- Positional embeddings