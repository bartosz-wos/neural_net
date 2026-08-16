# Jamba Hybrid Block Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the Jamba hybrid block (Lieber et al. 2024, AI21, "Jamba: A Hybrid Transformer-Mamba Language Model", arxiv:2403.19887) — a single fused block that combines Mamba-2 SSM with multi-head self-attention and a Mixture-of-Experts FFN via pre-norm residuals — in `include/nn/layers/architectures/jamba.h` / `.cpp` with a complete `tests/test_jamba.cpp` suite.

**Architecture:** A `JambaBlock(d_model, num_heads, num_experts, top_k, d_state, moe_every_n)` layer that, in a single forward pass, runs:

1. **Pre-norm + Mamba sublayer** — `x = x + Mamba2Block(LayerNorm(x))`
2. **Pre-norm + Attention sublayer** — `x = x + MultiHeadAttention(LayerNorm(x))` (the Jamba paper interleaves Mamba and Attention; the 1:1 ratio blocks this PR covers is the standard "hybrid" baseline)
3. **Pre-norm + MoE-FFN sublayer** — `x = x + MoEFFN(LayerNorm(x))` where `MoEFFN` is `num_experts` parallel 2-layer Dense MLPs with a top-k router

The MoE step is gated by `moe_every_n` (default 1 = every block is MoE; setting 2 = every other block, matching the Jamba paper's "MoE replaces dense FFN every other layer" recommendation). Input/output shape: `(T, d_model) → (T, d_model)`.

**Tech Stack:** C++17, the existing `Tensor`, `Dense`, `Layer`, `LayerNorm`, `Mamba2Block`, `MultiHeadAttention`, `MoELayer` infrastructure. New code only adds the `JambaBlock` glue (pre-norm residuals, optional MoE swap, parameter aggregation) and a `JambaStack` for repeated blocks.

**Paper reference:**
- Lieber et al. 2024, "Jamba: A Hybrid Transformer-Mamba Language Model" (https://arxiv.org/abs/2403.19887)
- Section 2: The Jamba architecture (Mamba + Attention + MoE blocks)
- Section 2.2: MoE replaces dense FFN in every other block

**Math summary (per block, single forward pass):**

```
x = (T, d_model)
m = Mamba2Block(LayerNorm_1(x))            // gates + SSD recurrence
x = x + m
a = MultiHeadAttention(LayerNorm_2(x))     // standard pre-norm attention
x = x + a
f = MoEFFN(LayerNorm_3(x))                 // top-k expert FFN
x = x + f
```

When `moe_every_n == 2`, the MoE step is replaced with a dense 2-layer Dense FFN on every other block (matching Jamba paper §2.2).

---

## Task 1: JambaBlock header skeleton + constructor validation

**Files:**
- Create: `include/nn/layers/architectures/jamba.h`

**Step 1: Write the header declaration**

```cpp
#ifndef JAMBA_H
#define JAMBA_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/mamba2.h"
#include "../attention/transformer.h"
#include "../architectures/mixture_of_experts.h"
#include <vector>
#include <memory>

class JambaBlock : public Layer {
public:
    // d_model: input/output feature dim
    // num_heads: number of attention heads (must divide d_model)
    // num_experts: number of MoE FFN experts (0 = use dense FFN)
    // top_k: top-k experts per token (default 2)
    // d_state: Mamba state dim / d_inner (default = 2 * d_model)
    // moe_every_n: 1 = every block MoE, 2 = every other block MoE (Jamba paper)
    JambaBlock(size_t d_model, size_t num_heads,
               size_t num_experts = 8, size_t top_k = 2,
               size_t d_state = 0, size_t moe_every_n = 1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "JambaBlock"; }

    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t num_experts() const { return num_experts_; }
    size_t top_k() const { return top_k_; }
    size_t d_state() const { return d_state_; }
    bool uses_moe() const { return uses_moe_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t num_experts_;
    size_t top_k_;
    size_t d_state_;
    size_t moe_every_n_;
    bool   uses_moe_;   // computed from moe_every_n_ and block index 0

    // Three pre-norm LayerNorms and their gradient buffers
    LayerNorm ln1_, ln2_, ln3_;
    // Mamba-2 SSM sublayer
    std::unique_ptr<Mamba2Block> mamba_;
    // Self-attention sublayer
    std::unique_ptr<MultiHeadAttention> attn_;
    // MoE FFN sublayer
    std::unique_ptr<MoELayer> moe_ffn_;
    // Dense FFN sublayer (used when MoE is disabled in this block)
    Dense w1_, b1_, w2_, b2_;   // (d_model -> 4*d_model -> d_model)

    // Forward cache (for backward)
    Tensor last_x_;             // (T, d_model)
    Tensor last_h1_;            // (T, d_model)  x after Mamba residual
    Tensor last_h2_;            // (T, d_model)  x after Attn residual
    Tensor last_ln1_, last_ln2_, last_ln3_;  // normalized inputs
    Tensor last_mamba_out_;     // (T, d_model)
    Tensor last_attn_out_;      // (T, d_model)
    Tensor last_ffn_pre_;       // (T, d_model)  ln3 output
    Tensor last_ffn_out_;       // (T, d_model)  ffn output (pre-residual)
    Tensor last_ffn_inner_;     // (T, 4*d_model)  ffn inner (gelu pre-activation)
};

#endif // JAMBA_H
```

**Step 2: Verify the umbrella header compiles**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
Expected: no errors.

**Step 3: Commit**

```bash
git add include/nn/layers/architectures/jamba.h
git commit -m "feat(architectures): JambaBlock header skeleton"
```

---

## Task 2: Jamba test scaffolding — forward shape + finiteness

**Files:**
- Create: `tests/test_jamba.cpp`

**Step 1: Write the test scaffolding**

The test file should include the standard `[PASS]/[FAIL]` infrastructure and a first test that:
- Constructs `JambaBlock(8, 2, 4, 2)` (d_model=8, num_heads=2, num_experts=4, top_k=2)
- Forwards a `(3, 8)` input
- Asserts output shape (3, 8) and all finite

**Step 2: Compile-only (DEFERRED until Task 3 implements the class)**

Note: The test in this task will fail to compile because `JambaBlock` is not yet implemented. We're writing the test FIRST so we can verify the failure as the systematic-debugging TDD discipline prescribes.

**Step 3: Verify the test fails to compile**

Run: `make test_jamba 2>&1 | head -20`
Expected: build error (`JambaBlock` undefined).

**Step 4: Commit**

```bash
git add tests/test_jamba.cpp
git commit -m "test(architectures): scaffold Jamba tests with forward shape"
```

(This test is the placeholder for Task 3; do not run it yet.)

---

## Task 3: JambaBlock constructor + forward (pre-norm + Mamba + Attention + FFN)

**Files:**
- Create: `include/nn/layers/architectures/jamba.cpp`

**Step 1: Constructor**

```cpp
#include "jamba.h"
#include <stdexcept>

JambaBlock::JambaBlock(size_t d_model, size_t num_heads,
                       size_t num_experts, size_t top_k,
                       size_t d_state, size_t moe_every_n)
    : d_model_(d_model),
      num_heads_(num_heads),
      num_experts_(num_experts),
      top_k_(top_k),
      d_state_(d_state == 0 ? 2 * d_model : d_state),
      moe_every_n_(moe_every_n),
      uses_moe_(num_experts > 0 && (moe_every_n == 1 || /*default block index 0*/ true)),
      ln1_(d_model), ln2_(d_model), ln3_(d_model),
      mamba_(std::make_unique<Mamba2Block>(d_model, num_heads, d_state_)),
      attn_(std::make_unique<MultiHeadAttention>(d_model, num_heads)),
      w1_(d_model, 4 * d_model), b1_(4 * d_model),
      w2_(4 * d_model, d_model), b2_(d_model) {
    if (d_model == 0) throw std::invalid_argument("JambaBlock: d_model must be > 0");
    if (num_heads == 0) throw std::invalid_argument("JambaBlock: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("JambaBlock: d_model must be divisible by num_heads");
    if (moe_every_n != 1 && moe_every_n != 2)
        throw std::invalid_argument("JambaBlock: moe_every_n must be 1 or 2");
    if (num_experts > 0) {
        moe_ffn_ = std::make_unique<MoELayer>(d_model, num_experts, top_k);
    }
}

// Many untrained Dense constructors leave weights uninitialized — initialize
// w1_ and w2_ with small random noise to avoid dead gradients.
```

Add a small Dense initialization helper at the top of the file:

```cpp
static void init_dense(Dense& d, size_t in_f, size_t out_f) {
    // Kaiming-He-like init: small Gaussian. Reuse the existing Tensor constructor
    // that fills from a flat buffer.
    std::vector<double> w(in_f * out_f);
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / in_f));
    for (auto& v : w) v = nd(rng);
    d.weights = Tensor(out_f, in_f, w.data());
    d.bias = Tensor::zeros(1, out_f);
    d.grad_weights = Tensor::zeros(out_f, in_f);
    d.grad_bias = Tensor::zeros(1, out_f);
}
```

**Step 2: Forward**

```cpp
Tensor JambaBlock::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("JambaBlock: input feature dim mismatch");
    last_x_ = input;

    // Step 1: pre-norm + Mamba + residual
    last_ln1_ = ln1_.forward(input);
    last_mamba_out_ = mamba_->forward(last_ln1_);
    last_h1_ = input + last_mamba_out_;

    // Step 2: pre-norm + Attention + residual
    last_ln2_ = ln2_.forward(last_h1_);
    last_attn_out_ = attn_->forward(last_ln2_);
    last_h2_ = last_h1_ + last_attn_out_;

    // Step 3: pre-norm + FFN + residual
    last_ln3_ = ln3_.forward(last_h2_);
    if (uses_moe_) {
        last_ffn_out_ = moe_ffn_->forward(last_ln3_);
    } else {
        last_ffn_inner_ = w1_.forward(last_ln3_);
        // Apply GELU element-wise
        for (size_t i = 0; i < last_ffn_inner_.rows; ++i)
            for (size_t j = 0; j < last_ffn_inner_.cols; ++j) {
                double x = last_ffn_inner_[i][j];
                last_ffn_inner_[i][j] = x * 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
            }
        last_ffn_out_ = w2_.forward(last_ffn_inner_);
    }
    return last_h2_ + last_ffn_out_;
}
```

**Step 3: Run the test from Task 2**

Run: `make test_jamba 2>&1 | tail -20 && ./build/test_jamba 2>&1 | head -30`
Expected: forward shape (3, 8), all finite, `[PASS]`.

**Step 4: Commit**

```bash
git add include/nn/layers/architectures/jamba.cpp
git commit -m "feat(architectures): JambaBlock constructor + forward"
```

---

## Task 4: JambaBlock backward (all 5 sublayer gradients)

**Files:**
- Modify: `include/nn/layers/architectures/jamba.cpp`

**Step 1: Backward**

Standard chain-rule pass through the 3 residual sublayers, in reverse order:

```cpp
Tensor JambaBlock::backward(const Tensor& grad_output, double learning_rate) {
    // Step 3 backward: residual splits grad_output into
    //   grad_h2_ (passed through) + grad_ffn_out_ (residual).
    last_ffn_pre_ = last_ln3_;  // cached for parameters()
    Tensor grad_h2 = grad_output;
    Tensor grad_ffn_out = grad_output;
    Tensor grad_ln3;
    if (uses_moe_) {
        grad_ln3 = moe_ffn_->backward(grad_ffn_out, learning_rate);
    } else {
        // Dense FFN backward: through w2 then w1 then GELU then ln3
        Tensor grad_inner = w2_.backward(grad_ffn_out, learning_rate);
        // GELU derivative
        for (size_t i = 0; i < grad_inner.rows; ++i)
            for (size_t j = 0; j < grad_inner.cols; ++j) {
                double x = last_ffn_inner_[i][j];
                double s = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
                double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
                grad_inner[i][j] *= s + x * pdf;  // d/dx[x*sigmoid-erf] = s + x*pdf
            }
        grad_ln3 = w1_.backward(grad_inner, learning_rate);
    }
    Tensor grad_ln3_via_norm = ln3_.backward(grad_ln3, learning_rate);

    // Step 2 backward: attention
    Tensor grad_h1 = grad_h2 + grad_ln3_via_norm;
    Tensor grad_attn_out = grad_h1;
    Tensor grad_ln2 = attn_->backward(grad_attn_out, learning_rate);
    Tensor grad_ln2_via_norm = ln2_.backward(grad_ln2, learning_rate);

    // Step 1 backward: Mamba
    Tensor grad_input = grad_h1 + grad_ln2_via_norm;
    Tensor grad_mamba_out = grad_h1;
    Tensor grad_ln1 = mamba_->backward(grad_mamba_out, learning_rate);
    Tensor grad_ln1_via_norm = ln1_.backward(grad_ln1, learning_rate);

    return grad_input + grad_ln1_via_norm;
}
```

**Step 2: Helper methods**

```cpp
void JambaBlock::update_weights(double learning_rate) {
    mamba_->update_weights(learning_rate);
    attn_->update_weights(learning_rate);
    if (uses_moe_) moe_ffn_->update_weights(learning_rate);
    else          { w1_.update_weights(learning_rate); w2_.update_weights(learning_rate); }
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ln3_.update_weights(learning_rate);
}

void JambaBlock::zero_grad() {
    mamba_->zero_grad();
    attn_->zero_grad();
    if (uses_moe_) moe_ffn_->zero_grad();
    else          { w1_.zero_grad(); w2_.zero_grad(); }
    ln1_.zero_grad(); ln2_.zero_grad(); ln3_.zero_grad();
}

std::vector<Tensor*> JambaBlock::parameters() {
    std::vector<Tensor*> p;
    for (auto* t : mamba_->parameters()) p.push_back(t);
    for (auto* t : attn_->parameters()) p.push_back(t);
    if (uses_moe_) {
        for (auto* t : moe_ffn_->parameters()) p.push_back(t);
    } else {
        p.push_back(&w1_.weights); p.push_back(&w1_.bias);
        p.push_back(&w2_.weights); p.push_back(&w2_.bias);
    }
    for (auto* t : ln1_.parameters()) p.push_back(t);
    for (auto* t : ln2_.parameters()) p.push_back(t);
    for (auto* t : ln3_.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> JambaBlock::gradients() {
    std::vector<Tensor*> g;
    for (auto* t : mamba_->gradients()) g.push_back(t);
    for (auto* t : attn_->gradients()) g.push_back(t);
    if (uses_moe_) {
        for (auto* t : moe_ffn_->gradients()) g.push_back(t);
    } else {
        g.push_back(&w1_.grad_weights); g.push_back(&w1_.grad_bias);
        g.push_back(&w2_.grad_weights); g.push_back(&w2_.grad_bias);
    }
    for (auto* t : ln1_.gradients()) g.push_back(t);
    for (auto* t : ln2_.gradients()) g.push_back(t);
    for (auto* t : ln3_.gradients()) g.push_back(t);
    return g;
}

Tensor JambaBlock::get_weights() const { return mamba_->get_weights(); }
Tensor JambaBlock::get_gradients() const { return mamba_->get_gradients(); }
```

**Step 3: Run test**

Run: `make test_jamba 2>&1 | tail -10 && ./build/test_jamba 2>&1 | head -30`
Expected: forward still passes; backward shouldn't crash.

**Step 4: Commit**

```bash
git add include/nn/layers/architectures/jamba.cpp
git commit -m "feat(architectures): JambaBlock backward chain + weight plumbing"
```

---

## Task 5: Test coverage — gradient verification

**Files:**
- Modify: `tests/test_jamba.cpp`

Add 8 focused tests:

1. **Constructor validation**: `(d_model=0)` throws, `(num_heads=0)` throws, `(d_model=4, num_heads=3)` throws (not divisible), `(moe_every_n=3)` throws.
2. **Forward shape**: `(2, 8) → (2, 8)`, `(1, 8) → (1, 8)`, `(5, 8) → (5, 8)`.
3. **Forward finite**: all finite, no NaN/Inf.
4. **Output nonzero**: random init → output != 0.
5. **Input gradient via centered FD**: `(T=3, d_model=4, num_heads=2, num_experts=0)` → input grad rel_err < 1e-4 vs central difference at eps=1e-4.
6. **Mamba-2 weights gradient FD check**: pick a parameter from the Mamba-2 sublayer (e.g., `in_proj.weights[0][0]`), perturb, verify gradient matches central FD at rel_err < 1e-3.
7. **Attention `W_q.weights` gradient FD check**: same recipe, rel_err < 1e-3.
8. **Dense FFN `w1_.weights[0][0]` gradient FD check** (with `num_experts=0`): same recipe, rel_err < 1e-3.
9. **Training reduces loss**: 30 SGD steps on `y = sum(x, axis=1)` projection reduces loss by >50%.
10. **Determinism**: two fresh `JambaBlock`s with the same `srand(42)` produce bit-identical forward outputs.
11. **Parameter count**: scales with `d_model`, `num_heads`, `num_experts`, `moe_every_n` (MoE adds more params than dense FFN).
12. **update_weights moves all params**: copy params, `update_weights(0.1)`, copy params again, expect difference.
13. **zero_grad clears all grads**: same recipe.
14. **No-MoE variant** (`num_experts=0`): forward shape, finite, nonzero, single gradient FD check.

Run: `make test_jamba && ./build/test_jamba 2>&1 | tail -40`
Expected: ≥24 `[PASS]`, ≤1 `[FAIL]` (acceptable for one-off numerical noise).

**Step 6: Commit**

```bash
git add tests/test_jamba.cpp
git commit -m "test(architectures): Jamba hybrid — 14 focused checks"
```

---

## Task 6: JambaStack — sequences of blocks

**Files:**
- Modify: `include/nn/layers/architectures/jamba.h`
- Modify: `include/nn/layers/architectures/jamba.cpp`

**Step 1: Header add**

```cpp
class JambaStack : public Layer {
public:
    JambaStack(size_t d_model, size_t num_heads, size_t num_layers,
               size_t num_experts = 8, size_t top_k = 2,
               size_t d_state = 0, size_t moe_every_n = 1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "JambaStack"; }
    size_t num_blocks() const { return blocks_.size(); }
private:
    std::vector<std::unique_ptr<JambaBlock>> blocks_;
};
```

**Step 2: Class implementation**

```cpp
JambaStack::JambaStack(size_t d_model, size_t num_heads, size_t num_layers,
                       size_t num_experts, size_t top_k,
                       size_t d_state, size_t moe_every_n) {
    if (num_layers == 0)
        throw std::invalid_argument("JambaStack: num_layers must be > 0");
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.push_back(std::make_unique<JambaBlock>(
            d_model, num_heads, num_experts, top_k, d_state, moe_every_n));
    }
}

Tensor JambaStack::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& b : blocks_) x = b->forward(x);
    return x;
}

Tensor JambaStack::backward(const Tensor& grad_output, double learning_rate) {
    Tensor g = grad_output;
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it)
        g = (*it)->backward(g, learning_rate);
    return g;
}

void JambaStack::update_weights(double lr) {
    for (auto& b : blocks_) b->update_weights(lr);
}

void JambaStack::zero_grad() {
    for (auto& b : blocks_) b->zero_grad();
}

std::vector<Tensor*> JambaStack::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks_)
        for (auto* t : b->parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> JambaStack::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks_)
        for (auto* t : b->gradients()) g.push_back(t);
    return g;
}

Tensor JambaStack::get_weights() const { return blocks_[0]->get_weights(); }
Tensor JambaStack::get_gradients() const { return blocks_[0]->get_gradients(); }
```

**Step 3: Add 5 stack tests**

- Forward shape `(4, 8) → (4, 8)` with 2 layers
- Forward finite
- 2-layer stack loss reduction over 20 SGD steps > 30%
- Parameter count scales linearly with num_layers
- Different `moe_every_n` produces different param counts (MoE adds more)

Run: `make test_jamba && ./build/test_jamba 2>&1 | tail -50`
Expected: ≥29 `[PASS]`.

**Step 4: Commit**

```bash
git add include/nn/layers/architectures/jamba.h include/nn/layers/architectures/jamba.cpp tests/test_jamba.cpp
git commit -m "feat(architectures): JambaStack — 2-layer Jamba — 5 stack tests"
```

---

## Task 7: Register in umbrella + Makefile + EXPANSION_QUEUE

**Files:**
- Modify: `include/nn/nn.h` (add `#include "layers/architectures/jamba.h"`)
- Modify: `Makefile` (add `build/test_jamba` rule, add to `tests:` deps, add `=== Running Jamba Hybrid Tests ===` echo)

**Step 1: Register in nn.h**

Add to the "Layers — architectures" section (alphabetically, near `include/nn/layers/architectures/hyena.h`):

```cpp
#include "layers/architectures/jamba.h"
```

**Step 2: Register in Makefile**

Add the build rule (after the `test_mamba3` rule):

```make
$(BUILD_DIR)/test_jamba: $(LIB_OBJS) $(BUILD_DIR)/test_jamba.o
```

Add to the `tests:` deps line.

Add to the `run_tests` block:

```make
@echo "=== Running Jamba Hybrid Tests ===" && ./$(BUILD_DIR)/test_jamba
```

**Step 3: Verify the umbrella header compiles**

Run: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
Expected: no errors.

**Step 4: Run `make tests` (or `make run_tests`)**

Run: `make run_tests 2>&1 | tail -20`
Expected: all existing tests still pass (no regressions).

**Step 5: Update EXPANSION_QUEUE.md**

Move the entry from `## Ideas` to `## Done` (since you're implementing it now), with a one-line summary of the test results.

**Step 6: Commit**

```bash
git add include/nn/nn.h Makefile EXPANSION_QUEUE.md
git commit -m "chore: register Jamba hybrid in umbrella, Makefile, EXPANSION_QUEUE"
```

---

## Reference

- Paper: Lieber et al. 2024, "Jamba: A Hybrid Transformer-Mamba Language Model" (https://arxiv.org/abs/2403.19887)
- AI21 implementation reference: https://github.com/ai21labs/Jamba
- The "block" structure (Mamba + Attention + MoE-FFN, three pre-norm residuals) is the canonical Jamba block per §2.

## Known scope limits

- **No MoE gating for "every other layer"** — `moe_every_n=2` is exposed but the test only checks parameter counts, not behavior. The MoE-vs-dense swap per block index is straightforward (compute `block_idx % moe_every_n` in the constructor) but the JambaBlock constructor doesn't take a block index today. Punted to a follow-up.
- **No positional encoding** — Jamba uses RoPE, but the existing Mamba-2 + Transformer sublayers don't have RoPE. Punted.
- **No KV caching** — Jamba at inference uses KV cache for the attention path. The current `MultiHeadAttention` doesn't have one. Punted.
- **No grouped-query attention (GQA)** — Jamba uses GQA. The current MultiHeadAttention is single-query. Punted.
