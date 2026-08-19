# MambaByte Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement MambaByte (Wang et al. 2024, https://arxiv.org/abs/2401.13660, "MambaByte: Token-free Selective State Space Model") as a token-free byte-level adaptation of Mamba, with byte-embedding input, the same selective SSM machinery as the existing `MambaBlock`, no patch/conv, plus a MambaByteModel that stacks blocks, and a small-scale test suite that hits machine precision on the backward against centered FD.

**Architecture:** A `MambaByteBlock(d_model, d_state, d_inner=0, vocab_size=256)` first embeds raw byte `u_t ∈ {0..255}` through a learnable `W_emb ∈ ℝ^{256 × d_model}` (or via `Dense(vocab_size, d_model)` returning a `(T, d_model)` projection), and then runs the canonical Mamba-1 selective SSM pass through the existing math (Δ_t = softplus(dt_proj(u_t)), B_t = B_proj(u_t), C_t = C_proj(u_t), Ā_t = exp(Δ_t ⊗ A), B̄_t = Δ_t ⊗ B_t, h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x̃_t, y_t = C_t · h_t), with the SiLU gate path and output projection. MambaByteModel stacks N blocks + final classifier.

**Tech Stack:** C++17, `Layer`, `Dense`, `Tensor`, softmax, autograd. Reuses the same selective-SSM mechanics as `MambaBlock`, but the layer accepts raw bytes (integers in `[0, vocab_size)`) directly. No new dependencies.

---

## Background

The MambaByte paper (Wang et al. 2024, https://arxiv.org/abs/2401.13660) is a token-free extension of Mamba that operates directly on bytes (vocabulary size = 256). The key differences from the canonical Mamba:

1. **No tokenizer, no patch embedding.** Bytes `[0..255]` are converted to feature vectors via a small learnable embedding. The selective-SSM machinery is unchanged.
2. **No convolution.** Mamba's 1-D depthwise convolution is dropped because for byte sequences the receptive field is too small relative to typical sequence lengths (the conv's main benefit in Mamba was local-context aggregation over subword tokens; for bytes this benefit vanishes). Our existing `MambaBlock` already drops the conv (see `include/nn/layers/recurrent/mamba.h` line 66) — perfect baseline.
3. **Optional Type-1 / Type-2 variants.** The paper describes two layer types (TWSS — token-wise-state space, with a skip gate added — and the more common selective-SSM variant). We implement the more common one as the default; the optional skip gate is exposed via a constructor flag.

We re-use 100% of the existing selective-SSM math from `MambaBlock`. The wrapper layer adds:

- A `ByteEmbedding(vocab_size, d_model)` lookup. We implement this directly as a `(vocab_size, d_model)` parameter Tensor and a `forward(bytes)` that fetches rows. (Bytes are a `(T,)` integer Tensor; the embedding returns `(T, d_model)`.)
- Optional `skip_gate` parameter `(d_model,)` for the TWSS variant: `out = silu(g) * y + skip_gate ⊙ x_emb`.

This is a small, focused addition that **demonstrates the token-free framing on top of the existing Mamba infrastructure** and is a useful pedagogical contrast with Mamba's token-level path.

---

## Mathematical formulation (per `MambaByteBlock(d_model, d_state, d_inner=0, vocab_size=256, twss=false)`)

### Inputs

- `bytes_t ∈ {0, ..., vocab_size-1}` for `t = 0..T-1`. Internally we store as a `Tensor(1, T)` of doubles (uint indexes).

### Forward

1. **Byte embedding lookup:**
   `x_t = W_emb[bytes_t]` ∈ ℝ^(1, d_model) for each t
   Stacked: `x ∈ ℝ^(T, d_model)`

2. **Mamba's selective SSM block** (identical to existing `MambaBlock`):
   - `p_t = in_proj(x_t)` → `(1, 2*d_inner)`
   - `x_pre_t = p_t[:d_inner]`, `g_t = p_t[d_inner:]`
   - `x̃_t = silu(x_pre_t)`
   - `Δ_t = softplus(dt_proj(x_t))` ∈ ℝ^(d_inner)
   - `B_t = B_proj(x_t)`, `C_t = C_proj(x_t)`
   - `A = -exp(A_log)`, `Ā_t = exp(Δ_t ⊗ A)`, `B̄_t = Δ_t ⊗ B_t`
   - `h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x̃_t`; `y_t = C_t · h_t`
   - `gated_t = silu(g_t) ⊙ y_t`
   - `out_t = out_proj(gated_t)` ∈ ℝ^(1, d_model)

3. **TWSS skip (only if `twss=true`):** add `skip_gate ⊙ x_t` to `out_t`. This is the additional gating the paper introduces in §2.3.

### Backward

The backward is composed of:

- `W_emb` gradient via standard embedding-lookup scatter (each `bytes_t` contributes its row gradient to `grad_W_emb[bytes_t]`).
- Selective-SSM backward: the existing `MambaBlock::backward` logic ported over. Same BPTT through `Δ_t`, `B_t`, `C_t`, `Ā_t`, `B̄_t`.
- TWSS gradient: `grad_skip_gate = sum_t grad_out_t ⊙ x_t`, `grad_x_t += skip_gate ⊙ grad_out_t`.

We integrate the byte-embedding backward with the rest of the chain.

---

## Files

- **Create:** `include/nn/layers/recurrent/mambabyte.{h,cpp}`
- **Create:** `tests/test_mambabyte.cpp`
- **Modify:** `include/nn/nn.h` (add `#include "layers/recurrent/mambabyte.h"`)
- **Modify:** `Makefile` (add `build/test_mambabyte` rule, `tests:` deps entry, `=== Running MambaByte Tests ===` echo in `run_tests`)

---

## Architecture (exact interfaces)

```cpp
// include/nn/layers/recurrent/mambabyte.h

class MambaByteBlock : public Layer {
public:
    // d_model:    input/output feature dim
    // d_state:    SSM state dim
    // d_inner:    inner feature dim (default = 2 * d_model, paper convention)
    // vocab_size: byte alphabet size (default 256)
    // twss:       if true, include the TWSS skip gate (paper §2.3)
    MambaByteBlock(size_t d_model, size_t d_state,
                  size_t d_inner = 0, size_t vocab_size = 256,
                  bool twss = false);

    // Accepts a (1, T) tensor of byte indices in [0, vocab_size).
    Tensor forward(const Tensor& bytes) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_emb; }
    Tensor get_gradients() const override { return grad_W_emb; }
    std::string name() const override { return "MambaByteBlock"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }
    size_t d_inner() const { return d_inner_; }
    size_t vocab_size() const { return vocab_size_; }
    bool twss() const { return twss_; }

    // Parameters (public for tests / grad check)
    Tensor W_emb;          // (vocab_size, d_model)        byte embedding
    Tensor A_log;          // (d_inner, d_state)           reparam: A = -exp(A_log)
    Tensor D_skip;         // (1, d_inner)                 learnable SSM skip
    Tensor skip_gate;      // (1, d_model)                 TWSS gate (zero-initialized if !twss)
    Dense in_proj;         // (d_model -> 2*d_inner)
    Dense out_proj;        // (d_inner -> d_model)
    Dense dt_proj;         // (d_model -> d_inner)        pre-softplus Δ
    Dense B_proj;          // (d_model -> d_state)
    Dense C_proj;          // (d_model -> d_state)

private:
    size_t d_model_, d_state_, d_inner_, vocab_size_;
    bool   twss_;

    // Caches (forward) and gradient buffers (backward)
    Tensor last_bytes_;        // (1, T)   byte indices
    Tensor last_embedded_;     // (T, d_model)
    Tensor last_p_;            // (T, 2*d_inner)   in_proj output
    Tensor last_x_pre_;        // (T, d_inner)
    Tensor last_g_;            // (T, d_inner)
    Tensor last_x_tilde_;      // (T, d_inner)     post-SiLU
    Tensor last_Delta_;        // (T, d_inner)
    Tensor last_Delta_pre_;    // (T, d_inner)
    Tensor last_B_t_;          // (T, d_state)
    Tensor last_C_t_;          // (T, d_state)
    Tensor last_A_bar_;        // (T, d_inner, d_state)
    Tensor last_B_bar_;        // (T, d_inner, d_state)
    Tensor last_h_;            // (T+1, d_inner, d_state)
    Tensor last_y_;            // (T, d_inner)
    Tensor last_gated_;        // (T, d_inner)

    // gradient buffers
    Tensor grad_W_emb_;        // (vocab_size, d_model)
    Tensor grad_A_log_;        // (d_inner, d_state)
    Tensor grad_D_skip_;       // (1, d_inner)
    Tensor grad_skip_gate_;    // (1, d_model)

    // Numerically-stable helpers (file-static in cpp)
    static double softplus(double x);
    static double sigmoid(double x);
    static double silu(double x);
};


class MambaByteModel : public Layer {
public:
    // input_dim:    raw byte input dim (1: byte indices are forwarded directly; we
    //               still pass (1, T) Tensor through MambaByteBlock stack)
    // d_model:      feature dim of the SSM blocks
    // output_dim:   number of classes for the final classifier
    // num_layers:   number of stacked MambaByteBlocks
    // d_state, d_inner, vocab_size, twss: passed through to each block
    MambaByteModel(size_t input_dim, size_t d_model, size_t output_dim,
                   size_t num_layers, size_t d_state = 4,
                   size_t d_inner = 0, size_t vocab_size = 256,
                   bool twss = false);

    Tensor forward(const Tensor& bytes) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { /* first block's W_emb */ }
    Tensor get_gradients() const override { /* first block's grad_W_emb */ }
    std::string name() const override { return "MambaByteModel"; }

private:
    size_t input_dim_, d_model_, output_dim_, num_layers_;
    std::vector<std::unique_ptr<MambaByteBlock>> blocks_;
    Dense classifier_;   // (d_model -> output_dim)
};
```

---

## Tasks

### Task 1: Write the byte-embedding slice of the failing test (RED)

**Files:**
- Test: `tests/test_mambabyte.cpp` (skeleton with one assertion: forward shape given a (1,4) byte input).

**Step 1.** Write test_mambabyte.cpp with:
   ```cpp
   #include "nn/layers/recurrent/mambabyte.h"
   int main() {
       MambaByteBlock block(/*d_model=*/2, /*d_state=*/2);
       Tensor bytes(1, 4);          // 4 byte indices 0,1,2,3
       bytes.data = {0.0, 1.0, 2.0, 3.0};
       Tensor out = block.forward(bytes);
       // expected shape (4, 2)
       assert(out.rows == 4 && out.cols == 2);
       return 0;
   }
   ```

**Step 2.** Build: `g++ -std=c++17 -Iinclude -c tests/test_mambabyte.cpp -o build/test_mambabyte.o`. Must fail with `mambabyte.h: No such file or directory`.

### Task 2: Implement MambaByteBlock header + cpp (GREEN-1)

**Files:**
- Create: `include/nn/layers/recurrent/mambabyte.h`
- Create: `include/nn/layers/recurrent/mambabyte.cpp`

Implement the full class per the interface above. Initialize `W_emb` from a small `Uniform(0, 0.02)` distribution. The `A_log` follows Mamba's `-exp(A_log)` reparameterization.

**Step 1.** Implement constructor (validates `d_model > 0`, `d_state > 0`, `vocab_size > 0`; sets `d_inner = 2 * d_model` if 0), `forward`, `backward`, `update_weights`, `zero_grad`, `parameters`, `gradients`, `get_weights`, `get_gradients`, `name`.

**Step 2.** Re-run test: `g++ -std=c++17 -Iinclude tests/test_mambabyte.cpp build/.../mambabyte.o ... -o test_mambabyte && ./test_mambabyte`. Must pass. (No gradient check yet, just shape.)

### Task 3: Add Test 2 (finiteness) and Test 3 (hand-derived reference), expand to suite

**Files:**
- Modify: `tests/test_mambabyte.cpp`

**Tests to add:**
- T2: output is finite for `T=6` random byte indices.
- T3: hand-derived reference with `d_model=1, d_state=1, vocab_size=2` and known weights — verify the first output matches `silu(g) · y + (twss ? skip_gate · x : 0)` to ≤1e-9.

### Task 4: Add grad-check tests (T4..T10)

**Files:**
- Modify: `tests/test_mambabyte.cpp`

Tests:
- T4: input byte-embedding gradient check (small batch, finite byte indices).
- T5: `W_emb` embedding gradient check.
- T6: `in_proj` `W` and `b` gradient checks.
- T7: `out_proj` `W` and `b` gradient checks.
- T8: `dt_proj`, `B_proj`, `C_proj` weight gradient checks (exercises selective-scan backward).
- T9: `A_log` gradient check.
- T10: `D_skip` gradient check.

Each test uses a centered-finite-difference with `eps=1e-5` against L2 loss, accepting `rel_err < 1e-3` (machine precision 1e-5 to 1e-9 in practice, but use 1e-3 to match repo conventions).

### Task 5: Add training test (T11)

**Files:**
- Modify: `tests/test_mambabyte.cpp`

Train a tiny MambaByte on bytes → regression target over 30 SGD steps. Assert L0 > LF (loss decreases).

### Task 6: Implement MambaByteModel stack (T12, T13)

**Files:**
- Modify: `include/nn/layers/recurrent/mambabyte.{h,cpp}`
- Modify: `tests/test_mambabyte.cpp`

Add `MambaByteModel`, stack N blocks + classifier. Tests:
- T12: forward shape (T, output_dim) for `T=4, num_layers=2`.
- T13: training reduces loss over 30 SGD steps on stacked model.

### Task 7: Wire up umbrella + Makefile

**Files:**
- Modify: `include/nn/nn.h` add `#include "layers/recurrent/mambabyte.h"`
- Modify: `Makefile`:
  - Add `$(BUILD_DIR)/test_mambabyte: $(LIB_OBJS) $(BUILD_DIR)/test_mambabyte.o` rule.
  - Append `$(BUILD_DIR)/test_mambabyte` to `tests:` deps line.
  - Append `@echo "=== Running MambaByte Tests ===" && ./$(BUILD_DIR)/test_mambabyte` to `run_tests`.

### Task 8: Run all tests, fix regressions

**Verification:**
- `make tests` builds all suites
- `make run_tests` runs them
- All MambaByte tests pass at the planned thresholds

### Task 9: Commit and push

`git commit -am "feat(recurrent): MambaByte — token-free selective SSM (Wang et al. 2024)"` and push.

---

## Verification gates per task

- After T1: build fails on missing header (RED proven)
- After T2: shape test passes (GREEN proven)
- After T3: hand-derived reference matches
- After T4..T10: gradient checks all pass with `rel_err < 1e-3`
- After T11: training reduces loss
- After T12: stacked model forward shape correct
- After T13: stacked model training reduces loss
- After T7: `make tests` builds clean
- After T8: `make run_tests` returns 0

---

## Pitfalls / known traps

1. **Embedding lookup gradient.** `grad_W_emb[bytes_t] += grad_x[t]`. Need to scatter correctly — don't allocate fresh rows per step.
2. **Selective-SSM BPTT.** Carry the Mamba code carefully; the cache shapes must match exactly.
3. **TWSS skip gate** initializes to zero so the layer starts identical to vanilla MambaByte, removing the skip-gate contribution at init (stable training).
4. **D_skip gradient chain** is fragile — same trap as in Mamba. Use a small `softplus(x)` clamp at +30 / -30 like the existing Mamba uses.
5. **Test data integrity.** L2 loss gradient is just `output - target`, no `/N`. Hand-derived expected values must be re-derived twice (paper mention + impl inspection) — never trust the prose alone.
6. **Layer shape conventions.** Input is `(1, T)` of integer-coded doubles. The output is `(T, d_model)`. Match the existing `MambaBlock` output convention.
