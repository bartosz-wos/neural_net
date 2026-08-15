# Gated DeltaNet (Yang, Kautz, Hatamizadeh 2025, ICLR 2025) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the Gated DeltaNet layer — combining the Mamba2/GLA per-head decay gate with the DeltaNet delta-rule write — in a clean multi-head CPU layer with full analytical BPTT and FD-verified gradients.

**Architecture:** Single multi-head sequence layer. Per head h, per time t, the recurrence is:

  `S_t = α_t · S_{t-1} · (I − β_t · k_t · k_t^T) + β_t · v_t · k_t^T`

  `o_t[h] = S_t[h] · q_t[h]`

followed by per-head concat + output projection. The gate `α_t = sigmoid(W_gate · x_t)` ∈ (0,1) per head lets the layer rapidly erase memory, while the delta-rule term `β_t · outer(k_t, v_t − S_{t-1}·k_t)` provides targeted updates. This is the "gated delta rule" that powers Gated DeltaNet (Yang et al. 2025, ICLR 2025, arxiv:2412.06464) and the Mamba-3 baseline.

**Tech Stack:** Hand-rolled C++17. Reuses the repo's existing `Dense`, `Tensor`, `Layer` conventions. Follows `DeltaNet` / `GLA` style: input (T, d_model), output (T, d_model), 5 learnable parameter tensors (6 with the new gate), per-head (head_dim × head_dim) state, single-step BPTT with explicit per-head loops.

**Reference:**
- https://arxiv.org/abs/2412.06464 (paper)
- https://github.com/NVlabs/GatedDeltaNet (NVIDIA reference)

**Math summary** (per head h, per time t):

```
q_t = W_Q · x_t                                (R^d_inner)
k_t_raw = W_K · x_t                            (R^d_inner)
v_t = W_V · x_t                                (R^d_inner)
β_t_pre = W_β · x_t; β_t = sigmoid(β_t_pre)    (R^n_heads, scalar per head)
α_t_pre = W_gate · x_t; α_t = sigmoid(α_t_pre)(R^n_heads, scalar per head)

for each head h:
  k_norm = ||k_t_raw[h]||₂ + 1e-12
  k_t[h] = (β_t[h] / k_norm) · k_t_raw[h]      (delta-rule k-mag normalization)

  S_t[h] = α_t[h] · S_{t-1}[h] · (I − β_t[h] · outer(k_t[h], k_t[h]))
         + β_t[h] · outer(k_t[h], v_t[h])

  o_t[h] = S_t[h] · q_t[h]                     (R^head_dim)

y_t = W_O · concat([o_t[0]; ...; o_t[H-1]])    (R^d_model)
```

**Learnable parameters (6 Dense layers → 12 tensors, weights + bias):**

1. `W_Q` (d_model → d_inner) — query projection
2. `W_K` (d_model → d_inner) — key projection
3. `W_V` (d_model → d_inner) — value projection
4. `W_O` (d_inner → d_model) — output projection
5. `W_β` (d_model → n_heads) — delta-rule write strength (per head, pre-sigmoid)
6. `W_gate` (d_model → n_heads) — Mamba2-style decay gate (per head, pre-sigmoid)

**Default dims:** `d_inner = d_model`, `head_dim = d_model / n_heads`. Requires `d_model % n_heads == 0`.

---

## Tasks (TDD, bite-sized)

### Task 1: Scaffold the GatedDeltaNet header

**Objective:** Create `include/nn/layers/recurrent/gated_deltanet.h` with the class declaration, following the DeltaNet header pattern (caches, public Denses, constructors, forward/backward signature, accessors).

**Files:**
- Create: `include/nn/layers/recurrent/gated_deltanet.h`

**Step 1: Write header content**

Define:
- `class GatedDeltaNet : public Layer`
- Constructor: `GatedDeltaNet(size_t d_model, size_t n_heads, size_t head_dim = 0)`
- Public Dense: `W_Q_, W_K_, W_V_, W_O_, W_beta_, W_gate_`
- Override: `forward`, `backward`, `update_weights`, `zero_grad`, `parameters`, `gradients`, `get_weights`, `get_gradients`, `name()="GatedDeltaNet"`
- Introspection: `last_state()`, `d_model()`, `n_heads()`, `head_dim()`, `d_inner()`
- Private cache fields: `cache_x_, cache_q_, cache_k_, cache_v_, cache_k_scaled_, cache_beta_pre_, cache_beta_, cache_gate_pre_, cache_gate_, cache_concat_o_, cache_S_`
- Private gradient buffers: `grad_q_, grad_k_, grad_v_, grad_k_scaled_, grad_x_`

**Step 2: Verify header compiles**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/layers/recurrent/gated_deltanet.h"'`
Expected: PASS (header self-sufficient)

**Step 3: Commit**

```bash
git add include/nn/layers/recurrent/gated_deltanet.h
git commit -m "feat(recurrent): GatedDeltaNet header (Yang 2025)"
```

### Task 2: Constructor + skeleton forward

**Objective:** Build the constructor (validation + Dense init) and a stub `forward` that returns zeros of the right shape. Write a failing test first.

**Files:**
- Create: `include/nn/layers/recurrent/gated_deltanet.cpp`
- Create: `tests/test_gated_deltanet.cpp`

**Step 1: Write a failing test (in `test_gated_deltanet.cpp`)**

```cpp
TEST("constructor validation throws on d_model=0")
TEST("constructor validation throws on n_heads=0")
TEST("constructor validation throws on d_model % n_heads != 0")
TEST("forward shape (T=3, d_model=4, n_heads=2) returns (3, 4)")
TEST("forward output is finite")
TEST("last_state() returns (n_heads, head_dim * head_dim) tensor")
```

These tests should FAIL on the first run (no implementation yet, header-only).

**Step 2: Write the implementation skeleton**

- Delegate constructor pattern: default-construct Dense then assign (same as DeltaNet)
- `head_dim_ = d_model_ / n_heads_`, `d_inner_ = d_model_`
- `init_weights("uniform")` for all 6 projections, biases zero
- `forward`: project (T, d_inner) for q/k/v and (T, n_heads) for β/gate, then return `Tensor(T, d_model_)` filled with zeros (placeholder)

**Step 3: Run tests, watch them pass for the *real* reasons**

Run: `make build/test_gated_deltanet && ./build/test_gated_deltanet`
Expected: 6/6 (constructor validation + shape + finite + state)

**Step 4: Commit**

```bash
git add include/nn/layers/recurrent/gated_deltanet.cpp tests/test_gated_deltanet.cpp
git commit -m "feat(recurrent): GatedDeltaNet constructor + skeleton forward"
```

### Task 3: Real forward — gate, β-scaled k, gated delta recurrence

**Objective:** Implement the actual recurrence `S_t = α_t · S_{t-1} · (I − β_t · k_t · k_t^T) + β_t · outer(k_t, v_t)`. Verify with hand-derived forward + output is finite + state norm grows.

**Files:**
- Modify: `include/nn/layers/recurrent/gated_deltanet.cpp`

**Step 1: Add failing tests**

```cpp
TEST("forward output differs from zero (real computation, not stub)")
TEST("forward output magnitude > some threshold (real values)")
TEST("last_state() norm > 0 (state accumulation works)")
TEST("hand-derived forward on (T=2, d_model=4, n_heads=2, head_dim=2) matches expected")
TEST("determinism — two fresh layers with copied params produce bit-identical forward")
```

**Step 2: Implement**

Inside `forward`:
1. Cache x_, run projections q/k/v/β_pre/gate_pre
2. Sigmoid β_t, gate_t (per-head)
3. Per-head k-magnitude normalization: `k_t[h] *= β_t[h] / ||k_t_raw[h]||`
4. Per-head gated delta recurrence:
   - `S_0[h] = 0`
   - For t in 0..T-1:
     - `I_term[h] = I - β_t[h] · outer(k_t[h], k_t[h])`     (head_dim × head_dim, computed in-place)
     - `S_t[h] = gate_t[h] · (S_{t-1}[h] · I_term[h]) + β_t[h] · outer(k_t[h], v_t[h])`
     - `o_t[h] = S_t[h] · q_t[h]`
     - Cache S_{t-1} (the state BEFORE update) into `cache_S_[t]`
5. Concat heads, project through W_O.

**Step 3: Verify**

Run: `./build/test_gated_deltanet`
Expected: tests 6/6 from Task 2 still pass + new tests pass for the right reasons.

**Step 4: Commit**

```bash
git add include/nn/layers/recurrent/gated_deltanet.cpp tests/test_gated_deltanet.cpp
git commit -m "feat(recurrent): GatedDeltaNet forward (gate + delta rule)"
```

### Task 4: Hand-derived signature test (forward closed-form)

**Objective:** Build a hand-derived single-head forward for a tiny case and assert it matches the implementation. This is the "soul" test that catches any future regression in the recurrence.

**Files:**
- Modify: `tests/test_gated_deltanet.cpp`

**Step 1: Hand-derive the closed form**

For `d_model = 2, n_heads = 1, head_dim = 2, T = 1`:

- Set `W_Q = [[1, 0], [0, 1]]`, `W_K = [[1, 0], [0, 1]]`, `W_V = [[1, 0], [0, 1]]`, `W_O = identity`, `W_β = [[1, 1]]`, `W_gate = [[0, 0]]`
- With `x = [[a, b]]` (1×2), `gate = sigmoid(0) = 0.5`, `β = sigmoid(a + b)`
- `k_raw = [a, b]`, `k_norm = sqrt(a² + b²)`, `k_t = (β / k_norm) · [a, b]`
- `I_term = I - β · outer(k_t, k_t) = I - β² · outer([a,b]/k_norm, [a,b]/k_norm)`
- `S_0 = 0`, so `S_1 = gate · 0 + β · outer(k_t, v_t) = β · outer(k_t, [a, b])`
- `o_1 = S_1 · q_1 = β · k_t · (k_t · q_1)` — i.e. projection of q_1 onto k_t then times β

**Step 2: Write the test**

For `a=0.6, b=0.8, β=sigmoid(1.4)=0.8022, gate=0.5, k_t = (β/1.0) · [0.6, 0.8]`:
- `S_1 = β · outer(k_t, [0.6, 0.8]) = β · [[0.36, 0.48], [0.48, 0.64]]`
- `q_1 = [0.6, 0.8]`, `o_1 = S_1 · q_1 = β · [0.36·0.6 + 0.48·0.8, 0.48·0.6 + 0.64·0.8] = β · [0.6, 0.8]`
- `y_1 = W_O · o_1 = o_1` (identity) → expected `[0.4813, 0.6417]`

**Step 3: Verify**

Run: `./build/test_gated_deltanet`
Expected: hand-derived matches at rel_err < 1e-10.

**Step 4: Commit**

```bash
git add tests/test_gated_deltanet.cpp
git commit -m "test(gated_deltanet): hand-derived forward signature"
```

### Task 5: Backward through W_O + output chain

**Objective:** Implement the gradient flow through W_O and the per-head `o_t[h] = S_t[h] · q_t[h]` chain.

**Files:**
- Modify: `include/nn/layers/recurrent/gated_deltanet.cpp`

**Step 1: Add failing test**

```cpp
TEST("backward returns (T, d_model) grad_x shape")
TEST("backward grad_x is finite")
```

**Step 2: Implement `backward` (start)**

- Backward through W_O: `grad_concat_o = W_O_.backward(grad_output, 0.0)`
- For each head: `gS_t[h, i, j] += grad_concat_o[t, h*head_dim + i] · q_t[h, j]` (output-side)
- For each head: `grad_q_t[h, i] = sum_j gS_t[h, i, j] · grad_concat_o[t, h*head_dim + j]`... wait, actually `o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]`, so `grad_q_t[h, i] = sum_j S_t[h, j, i] · grad_concat_o[t, h*head_dim + j]` (transposed index)

**Step 3: Run, watch pass**

**Step 4: Commit**

### Task 6: Full BPTT through the gated delta recurrence

**Objective:** Implement the backward propagation of gS_t back to gS_{t-1} via the gated delta update. This is the hard part — need to derive carefully.

**Files:**
- Modify: `include/nn/layers/recurrent/gated_deltanet.cpp`

**Step 1: Derive (mirror DeltaNet's pattern)**

For the gated delta update:
  `S_t[h] = gate_t[h] · S_{t-1}[h] · (I − β_t[h] · k_t[h] · k_t[h]^T) + β_t[h] · k_t[h] · v_t[h]^T`

Let `G = I − β · k k^T` (the "delete" matrix), so `S_t = gate · S_{t-1} · G + β · outer(k, v)`.

Contributions to gS_{t-1} from gS_t:

(a) Linear in S_{t-1}: `gS_{t-1}[h] += gate_t[h] · gS_t[h] · G^T`
    (because S_t = ... + gate · S_{t-1} · G, dS_t/dS_{t-1} = gate · G^T when applied on the left)

(b) From v_t = v_t (no S_{t-1} dependence): zero direct contribution to S_{t-1}.

Contributions to g_k_t, g_v_t, g_gate_t, g_beta_t (via W_β), and via the chain through g_k_scaled → g_k_raw:

Use the same strategy as DeltaNet's backward: cache S_{t-1}, recompute Sk = S_{t-1}·k, k·Sk, gate, β, v_t on demand, then:
- `g_v_t[h, j] = sum_i gS_t[h, i, j] · β · k_t[h, i]`
- `g_k_t[h, i]` from 4 sources: outer-product (β·sum_j gS·v), the Sk path through g_v_t (β·G^T·g_v), and the alpha path through k_t.
- The gate path: `g_gate[h] = sum_{i,j} gS_t[h, i, j] · (S_{t-1}[h] · G^T)_{i, j}`.
- The β path (via W_β): `g_beta[h]` from `S_t = gate · S_{t-1} · G + β · outer(k, v)` where `G = I − β · kk^T`. Two contributions: (i) `dS_t/dβ · gS_t = (-gate · S_{t-1} · kk^T + outer(k, v) - β · outer(k, v) · 0) · gS_t`... actually since `outer(k,v) = k·v^T` and β is a scalar multiplier, `∂(β·k·v^T)/∂β = k·v^T = outer(k,v)`. So `g_β += sum_{i,j} gS_t[h, i, j] · outer(k_t, v_t)[i, j] = <gS_t[h], outer(k_t, v_t)>`. Plus from the G path: `g_β += sum_{i,j} gS_t[h, i, j] · gate · S_{t-1}[h] · dG/dβ_{i,j}` where `dG/dβ = -kk^T`, so `g_β += -gate · sum_{i,j} gS_t[h,i,j] · S_{t-1}[h,i,j'] · k_t[h,i] · k_t[h,j']` — careful with the S_{t-1}·G contraction.

This is intricate. To keep it tractable, we'll implement the full derivation in the code with a per-head scratch tensor, mirroring the DeltaNet structure.

**Step 2: Add failing test**

```cpp
TEST("input gradient (T=3, d=4, h=2) rel_err < 1e-4 vs centered FD")
```

This is the key FD vs analytical check for the recurrence BPTT. If it fails, the derivation is wrong.

**Step 3: Run, debug, fix until rel_err < 1e-4**

**Step 4: Commit**

```bash
git add include/nn/layers/recurrent/gated_deltanet.cpp
git commit -m "feat(recurrent): GatedDeltaNet BPTT (gate + delta rule)"
```

### Task 7: Parameter gradients

**Objective:** After BPTT produces grad_q, grad_k, grad_v, run backward through each Dense to accumulate gradient on W_Q, W_K, W_V, W_β, W_gate.

**Files:**
- Modify: `include/nn/layers/recurrent/gated_deltanet.cpp`

**Step 1: Add failing tests**

```cpp
TEST("W_Q.weights grad rel_err < 1e-4 vs FD")
TEST("W_K.weights grad rel_err < 1e-4 vs FD")
TEST("W_V.weights grad rel_err < 1e-4 vs FD")
TEST("W_O.weights grad rel_err < 1e-4 vs FD")
TEST("W_beta.weights grad rel_err < 1e-4 vs FD")
TEST("W_gate.weights grad rel_err < 1e-4 vs FD")
```

**Step 2: Implement**

After computing grad_q_, grad_k_, grad_v_, grad_k_scaled_, grad_beta_pre_, grad_gate_pre_:
- `W_Q_.backward(grad_q_, 0.0)` → gx_q (accumulates W_Q grad internally)
- `W_K_.backward(grad_k_, 0.0)` → gx_k
- `W_V_.backward(grad_v_, 0.0)` → gx_v
- `W_beta_.backward(grad_beta_pre_, 0.0)` → gx_beta
- `W_gate_.backward(grad_gate_pre_, 0.0)` → gx_gate
- Sum: `grad_x_ = gx_q + gx_k + gx_v + gx_beta + gx_gate`
- Return `grad_x_`

**Step 3: Verify**

Run: `./build/test_gated_deltanet`
Expected: all FD checks pass at rel_err < 1e-4.

**Step 4: Commit**

```bash
git add include/nn/layers/recurrent/gated_deltanet.cpp tests/test_gated_deltanet.cpp
git commit -m "feat(recurrent): GatedDeltaNet parameter gradients (FD-verified)"
```

### Task 8: Training reduces loss + multi-head integration

**Objective:** Verify the layer can train end-to-end on a synthetic regression task, and that multi-head works.

**Files:**
- Modify: `tests/test_gated_deltanet.cpp`

**Step 1: Add tests**

```cpp
TEST("training reduces MSE loss on synthetic seq-regression (50 SGD steps, >50%)")
TEST("multi-head (n_heads=3, head_dim=2, d_model=6) forward shape + finite")
TEST("update_weights moves all 12 parameter tensors")
TEST("zero_grad clears all 12 parameter gradients")
TEST("parameters() returns 12 tensors (6 Denses × weights+bias)")
TEST("gradients() returns 12 tensors")
TEST("longer sequence (T=8, d_model=4, n_heads=2) input grad FD rel_err < 1e-3")
```

**Step 2: Verify all pass**

**Step 3: Commit**

```bash
git add tests/test_gated_deltanet.cpp
git commit -m "test(gated_deltanet): training + multi-head integration"
```

### Task 9: Umbrella + Makefile registration

**Objective:** Add `#include "layers/recurrent/gated_deltanet.h"` to `include/nn/nn.h`. Register the test in `Makefile` (compile rule, tests: deps, run_tests: echo line).

**Files:**
- Modify: `include/nn/nn.h` (add the include in the recurrent section)
- Modify: `Makefile` (add `build/test_gated_deltanet` rule, append to `tests:` deps, add `=== Running Gated DeltaNet Tests ===` echo line)

**Step 1: Edit nn.h** — insert `#include "layers/recurrent/gated_deltanet.h"` after `#include "layers/recurrent/deltanet.h"`

**Step 2: Edit Makefile** — three additions:
- New compile rule block (after the `deltanet` rule):
  ```make
  $(BUILD_DIR)/test_gated_deltanet: $(LIB_OBJS) $(BUILD_DIR)/test_gated_deltanet.o
      $(CXX) $^ -o $@
  ```
- Append `$(BUILD_DIR)/test_gated_deltanet` to the `tests:` deps line
- Add `=== Running Gated DeltaNet Tests === && ./$(BUILD_DIR)/test_gated_deltanet` to `run_tests:`

**Step 3: Run `make tests` to confirm aggregate build still works**

Run: `make tests 2>&1 | tail -30`
Expected: all 95+ test binaries compile, including test_gated_deltanet

**Step 4: Run `make run_tests`**

Expected: GatedDeltaNet tests pass alongside all the others. Note: pre-existing intermittent failures (test_adaln_zero, test_wgan_gp, test_vit, test_coord_network, test_gat_attention, test_odernn_grad_check) are documented in NOT_FIXED.md and are NOT blockers.

**Step 5: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register GatedDeltaNet in umbrella and Makefile"
```

### Task 10: Plan update + queue entry + push

**Objective:** Move this plan to Done in EXPANSION_QUEUE.md with a one-line summary, and push to origin.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Edit EXPANSION_QUEUE.md**

Move the GatedDeltaNet entry from `## Ideas` to `## Done` with the focused test count.

**Step 2: Push**

```bash
git push origin master
```

**Step 3: Done.**

---

## Verification gates

- All 6+ focused checks pass at machine precision on every test rerun
- Aggregate `make tests` compiles all binaries including the new one
- `make run_tests` runs the GatedDeltaNet suite alongside the rest
- Existing tests have no regressions (NOT_FIXED.md omissions are unchanged)
- The hand-derived forward test catches any future regression in the recurrence

## Anti-patterns to avoid

- **Don't skip the FD-vs-analytical gradient check.** It's the only thing that catches backward derivation bugs.
- **Don't initialize Dense weights to large values.** The gated delta recurrence is sensitive to scale; use uniform [-1/sqrt(fan_in), 1/sqrt(fan_in)] like DeltaNet.
- **Don't forget the gate (W_gate).** It's the whole point of Gated DeltaNet vs DeltaNet.
- **Don't write a single combined `backward` that mixes the gate and delta paths.** Keep them separate, derive each chain, sum the contributions.
- **Don't release with the zero-return stub from Task 2.** If a test for "real computation" still passes with the stub, that's a vacuous test — strengthen it.