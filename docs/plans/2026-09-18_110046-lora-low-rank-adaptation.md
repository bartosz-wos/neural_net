# LoRA — Low-Rank Adaptation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement **LoRA (Low-Rank Adaptation)** — Hu et al. 2021, https://arxiv.org/abs/2106.09685 — as a drop-in trainable adapter that adds a low-rank update path to a frozen dense projection. LoRA is the canonical parameter-efficient fine-tuning technique, ubiquitous in modern LLM/post-training stacks, and a natural fit for a quant-dev / ML-practitioner repo.

**Architecture:** A `LoRALinear(d_in, d_out, rank, alpha=1.0, base_init="xavier")` layer subclassing `Layer`. Holds a **frozen** base weight `W_0 ∈ ℝ^{d_out × d_in}` plus a bias `b ∈ ℝ^{1, d_out}` (the base `Dense`); adds trainable `A ∈ ℝ^{rank × d_in}` (Kaiming-uniform init, §4.2) and `B ∈ ℝ^{d_out × rank}` (zero init, §4.2). Forward is `y = x · W_0ᵀ + b + (α/rank) · x · Aᵀ · Bᵀ`. Because B starts at 0, the initial output equals the base `Dense` exactly — the regression test for "LoRA breaks the base on day 1". A `merge_weights()` helper folds the update into the base once training is done (paper §4.2): `W' = W_0 + (α/rank) · B · A`, then a Dense produces the same output with zero extra FLOPs at inference.

**Tech Stack:** C++17, Tensor matmul (`Tensor::operator*`), Layer base class from `core/layer.h`. No new dependencies.

---

## Why LoRA — and why now

- The repo already has 100+ architectures and 50+ optimizers, but **zero** parameter-efficient fine-tuning primitives (no LoRA, no prefix tuning, no adapter modules). LoRA is the most-cited PEFT paper, the foundation of QLoRA/DoRA/VeRA/AdaLoRA, and conceptually the simplest PEFT method — perfect as the first PEFT module shipped.
- The math is tractable: rank-`r` update path means the B-update gradient, A-update gradient, and input gradient are all closed-form matmul derivatives.
- Validation is mechanical: at init with `B=0`, output ≡ base Dense output to machine precision (the standard "LoRA is transparent at init" check). At `rank=0`, the LoRA path is identically zero (no PEFT). At `rank=d_in=d_out`, the LoRA path has the same capacity as the base (full-rank sanity check).

---

## Task 1: Write failing test — constructor & accessors

**File:** `tests/test_lora.cpp` (new)

**Step 1:** Write the test stub:

```cpp
#include "nn/layers/utility/lora.h"
#include "nn/core/tensor.h"
#include <cmath>
#include <iostream>

static int passed = 0, failed = 0;
static bool check(const std::string& name, bool cond) {
    if (cond) { std::cout << "  [PASS] " << name << std::endl; ++passed; }
    else      { std::cout << "  [FAIL] " << name << std::endl; ++failed; }
    return cond;
}

int main() {
    std::cout << "=== LoRA Tests ===" << std::endl;

    // Test 1: valid construction
    {
        LoRALinear lora(4, 3, 2);  // d_in=4, d_out=3, rank=2
        check("Test 1a: d_in() == 4",  lora.d_in() == 4);
        check("Test 1b: d_out() == 3", lora.d_out() == 3);
        check("Test 1c: rank() == 2",  lora.rank() == 2);
        check("Test 1d: alpha() == 2", lora.alpha() == 2.0);  // default alpha = rank
        check("Test 1e: scaling() == 1.0", std::abs(lora.scaling() - 1.0) < 1e-12);
        check("Test 1f: A shape (rank, d_in) = (2, 4)",  lora.A().rows == 2 && lora.A().cols == 4);
        check("Test 1g: B shape (d_out, rank) = (3, 2)", lora.B().rows == 3 && lora.B().cols == 2);
        check("Test 1h: B starts at zero (paper §4.2 init)", max_abs(lora.B()) == 0.0);
    }
    // ... (more tests)
    return failed > 0 ? 1 : 0;
}
```

**Step 2:** Run it. Expected: FAIL — `LoRALinear` doesn't exist.

---

## Task 2: Implement `LoRALinear` header

**File:** `include/nn/layers/utility/lora.h` (new)

```cpp
#ifndef LORA_H
#define LORA_H

#include "../../core/layer.h"
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// LoRA — Low-Rank Adaptation of dense layers
//   Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — 2021
//   "LoRA: Low-Rank Adaptation of Large Language Models"
//   https://arxiv.org/abs/2106.09685
//
// Drop-in adapter for a frozen Dense: adds a TRAINABLE low-rank update
//   ΔW = (α / rank) · B · A   ∈ ℝ^{d_out × d_in}
// to the base weight W_0. Forward becomes
//   y = x · (W_0 + ΔW)ᵀ + b
//     = x · W_0ᵀ + (α/rank) · x · Aᵀ · Bᵀ + b
// Both paths share the input — the B=0 initialization (§4.2) means the LoRA
// path contributes EXACTLY ZERO at construction, so the layer's initial
// forward is bit-exact equal to the base Dense. Training the (rank × (d_in
// + d_out)) parameters — vs - Apple's full d_out × d_in — gives the
// parameter-efficient fine-tuning property.
//
// Inference-time merge: merge_weights() folds the update into the base
//   W' = W_0 + (α/rank) · B · A
// so a forward becomes a single matmul (paper §4.2, no extra FLOPs at
// deployment). After merge, A/B should be deleted (`frozen=true` is the
// flag) so update_weights() is a no-op.
// ============================================================================

class LoRALinear : public Layer {
public:
    // d_in / d_out: base Dense dimensions.
    // rank: low-rank decomposition rank. r=0 → LoRA is a no-op (paper §4.1).
    // alpha: scalar; scaling factor is (alpha / rank). Default: alpha = rank,
    //        so scaling = 1.0 (paper convention).
    // base_init: passed to Dense.init_weights() — "xavier", "he", "uniform",
    // "zeros" (default per repo convention).
    // freeze_base: if true (default), the base W_0/b are not trained. If
    //              false, LoRA falls back to a regular Dense (with the
    //              low-rank path still active).
    LoRALinear(size_t d_in, size_t d_out, size_t rank,
              double alpha = -1.0,                  // -1 sentinel → alpha = rank
              const std::string& base_init = "xavier",
              bool freeze_base = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_0_->weights; }
    Tensor get_gradients() const override { return grad_A_; }    // primary trainable grad
    std::string name() const override { return "LoRALinear"; }

    // Accessors
    size_t d_in() const { return d_in_; }
    size_t d_out() const { return d_out_; }
    size_t rank() const { return rank_; }
    double alpha() const { return alpha_; }
    double scaling() const { return alpha_ / static_cast<double>(rank_); }
    bool is_merged() const { return merged_; }
    bool base_frozen() const { return freeze_base_; }

    const Tensor& A() const { return A_; }
    const Tensor& B() const { return B_; }
    const Tensor& base_weights() const { return W_0_->weights; }
    const Tensor& base_bias() const { return W_0_->bias; }

    // Merge: W_0 ← W_0 + (α/rank) · B · A, then mark frozen & disable A/B
    // updates. After merge, a forward reduces to a single matmul (the paper
    // §4.2 inference trick — zero extra FLOPs at deployment).
    void merge_weights();

    // Unmerge: revert the merge (subtract the contribution back out).
    // Useful for continuing training after a merge.
    void unmerge_weights();

private:
    size_t d_in_;
    size_t d_out_;
    size_t rank_;
    double alpha_;
    bool freeze_base_;
    bool merged_;

    // Base Dense (frozen if freeze_base_ is true)
    std::unique_ptr<Dense> W_0_;

    // LoRA params: A (rank, d_in), B (d_out, rank)
    Tensor A_;
    Tensor B_;

    // LoRA grads
    Tensor grad_A_;
    Tensor grad_B_;
    Tensor grad_base_W_;   // only nonzero if !freeze_base_
    Tensor grad_base_b_;

    // Forward cache (input, intermediate x·Aᵀ = Z, etc.)
    Tensor last_input_;    // (N, d_in)
    Tensor last_Z_;        // (N, rank)  = last_input_ · Aᵀ
    Tensor last_deltaW_x_; // (N, d_in)  = scaling · last_Z_ · Bᵀ (cached for backward)
    Tensor last_base_out_; // (N, d_out) = last_input_ · W_0ᵀ + b (cached)
};

#endif
```

**Step 3:** Run test. Expected: still FAIL — class doesn't exist.

---

## Task 3: Implement `LoRALinear` source

**File:** `include/nn/layers/utility/lora.cpp` (new)

```cpp
#include "lora.h"
#include <stdexcept>
#include <cmath>

LoRALinear::LoRALinear(size_t d_in, size_t d_out, size_t rank,
                       double alpha, const std::string& base_init,
                       bool freeze_base)
    : d_in_(d_in), d_out_(d_out), rank_(rank),
      alpha_(alpha < 0 ? static_cast<double>(rank) : alpha),
      freeze_base_(freeze_base), merged_(false),
      W_0_(std::make_unique<Dense>(d_in, d_out)) {
    if (d_in == 0)  throw std::invalid_argument("LoRALinear: d_in must be > 0");
    if (d_out == 0) throw std::invalid_argument("LoRALinear: d_out must be > 0");
    if (alpha_ < 0)  throw std::invalid_argument("LoRALinear: alpha must be >= 0");
    if (rank == 0 && !freeze_base_) {
        // No PEFT path AND base is frozen → nothing trainable. Reject.
        throw std::invalid_argument("LoRALinear: rank=0 with freeze_base=true leaves nothing trainable");
    }

    W_0_->init_weights(base_init);
    // paper §4.2: A ~ Kaiming uniform (we use small Gaussian as a stand-in;
    //            we only need std ~ 1/sqrt(d_in) which is what random(scale)
    //            can give us with scale = 1/sqrt(d_in))
    // B = 0
    double A_scale = std::sqrt(1.0 / static_cast<double>(d_in));
    A_ = Tensor::random(rank, d_in, A_scale);
    A_ = Tensor::random(rank, d_in, A_scale);  // refresh to dodge shared-state issues
    B_ = Tensor::zeros(d_out, rank);

    grad_A_ = Tensor::zeros(rank, d_in);
    grad_B_ = Tensor::zeros(d_out, rank);
    grad_base_W_ = Tensor::zeros(d_out, d_in);
    grad_base_b_ = Tensor::zeros(1, d_out);
}

Tensor LoRALinear::forward(const Tensor& input) {
    if (input.cols != d_in_)
        throw std::invalid_argument("LoRALinear: input.cols != d_in");
    last_input_ = input;

    // Base path: y_base = x · W_0ᵀ + b
    Tensor y_base = W_0_->forward(input);
    last_base_out_ = y_base;

    if (rank_ == 0) return y_base;        // no PEFT path

    // LoRA path: y_lora = scaling · x · Aᵀ · Bᵀ
    Tensor xA = input * A_.transpose();    // (N, rank)
    last_Z_ = xA;
    Tensor deltaW_x = xA * B_.transpose(); // (N, d_in) — wait, that's wrong
    // Wait: y_lora = (x · Aᵀ) · Bᵀ, so shape (N, rank) · (rank, d_out) = (N, d_out)
    Tensor y_lora = xA * B_.transpose() * scaling();  // (N, d_out)
    last_deltaW_x_ = y_lora;

    return y_base + y_lora;
}

Tensor LoRALinear::backward(const Tensor& grad_output, double learning_rate) {
    if (grad_output.cols != d_out_)
        throw std::invalid_argument("LoRALinear: grad_output.cols != d_out");

    if (rank_ == 0) {
        // base Dense backward only
        return W_0_->backward(grad_output, learning_rate);
    }

    // --- LoRA path backward ---
    // y_lora = scaling · x · Aᵀ · Bᵀ
    // dY_lora = grad_output (split into d_y_lora and d_y_base separately is
    // NOT needed; both paths get the FULL grad_output since y = y_base + y_lora)

    // grad to B: dB = scaling · (x · Aᵀ)ᵀ · grad_output  → (rank, d_out)
    Tensor xA = last_Z_;                   // (N, rank)
    Tensor dB = xA.transpose() * grad_output * scaling();  // (rank, d_out)
    // grad_B_ is (d_out, rank) per our convention; transpose dB
    for (size_t i = 0; i < rank_; ++i)
        for (size_t j = 0; j < d_out_; ++j)
            grad_B_(j, i) = dB(i, j);

    // grad to A: dA = scaling · xᵀ · (grad_output · B)  → (N, d_in)
    Tensor grad_times_B = grad_output * B_;  // (N, rank)
    Tensor dA = last_input_.transpose() * grad_times_B * scaling();  // (rank, d_in)
    grad_A_ += dA;

    // grad to input from LoRA path: dX_lora = scaling · (grad_output · B) · A
    // = grad_times_B * A   → (N, d_in)
    Tensor dx_lora = grad_times_B * A_;     // (N, d_in) — (N, rank) · (rank, d_in)
    dx_lora = dx_lora * scaling();

    // --- Base path backward (use grad_output, not split) ---
    Tensor dx_base = W_0_->backward(grad_output, learning_rate);  // updates grad_base_W_, grad_base_b_, returns dX

    // Total d_input = dX_lora + dX_base
    Tensor dx = dx_base + dx_lora;
    return dx;
}

void LoRALinear::update_weights(double lr) {
    if (merged_) {
        // After merge, trainable params are A/B but their effect is already
        // baked into W_0_. Standard practice is to NOT train post-merge.
        // Zero their grads and return.
        grad_A_.fill(0.0);
        grad_B_.fill(0.0);
        return;
    }
    // Update A, B with their grads (note grad_A_ shape is (rank, d_in), A is (rank, d_in) — same shape)
    for (size_t i = 0; i < A_.data.size(); ++i) A_.data[i] -= lr * grad_A_.data[i];
    for (size_t i = 0; i < B_.data.size(); ++i) B_.data[i] -= lr * grad_B_.data[i];
    if (!freeze_base_) {
        // Update base Dense too
        for (size_t i = 0; i < W_0_->weights.data.size(); ++i)
            W_0_->weights.data[i] -= lr * grad_base_W_.data[i];
        for (size_t i = 0; i < W_0_->bias.data.size(); ++i)
            W_0_->bias.data[i] -= lr * grad_base_b_.data[i];
    }
    zero_grad();
}

void LoRALinear::zero_grad() {
    grad_A_.fill(0.0);
    grad_B_.fill(0.0);
    grad_base_W_.fill(0.0);
    grad_base_b_.fill(0.0);
}

std::vector<Tensor*> LoRALinear::parameters() {
    std::vector<Tensor*> p = { &A_, &B_ };
    if (!freeze_base_) {
        p.push_back(&W_0_->weights);
        p.push_back(&W_0_->bias);
    }
    return p;
}

std::vector<Tensor*> LoRALinear::gradients() {
    std::vector<Tensor*> g = { &grad_A_, &grad_B_ };
    if (!freeze_base_) {
        g.push_back(&grad_base_W_);
        g.push_back(&grad_base_b_);
    }
    return g;
}

void LoRALinear::merge_weights() {
    if (merged_) return;
    if (rank_ == 0) { merged_ = true; return; }  // no-op
    // W_0 ← W_0 + (α/rank) · B · A
    Tensor update = B_ * A_ * scaling();  // (d_out, rank) · (rank, d_in) = (d_out, d_in)
    for (size_t i = 0; i < W_0_->weights.data.size(); ++i)
        W_0_->weights.data[i] += update.data[i];
    // Zero out A, B and their grads
    A_.fill(0.0);
    B_.fill(0.0);
    grad_A_.fill(0.0);
    grad_B_.fill(0.0);
    merged_ = true;
}

void LoRALinear::unmerge_weights() {
    if (!merged_) return;
    if (rank_ == 0) return;
    Tensor update = B_ * A_ * scaling();
    for (size_t i = 0; i < W_0_->weights.data.size(); ++i)
        W_0_->weights.data[i] -= update.data[i];
    merged_ = false;
}
```

---

## Task 4: Test the B=0 init invariant

The most important correctness property: **at construction with B=0, LoRALinear's forward must produce output bit-exactly equal to the underlying Dense's forward** (the paper's §4.2 design contract).

```cpp
// Test 4: B=0 → LoRA path contributes zero at init → forward ≡ base Dense
{
    LoRALinear lora(3, 4, 2);
    Tensor input(2, 3);
    input(0, 0) = 1.0; input(0, 1) = -0.5; input(0, 2) = 0.3;
    input(1, 0) = 0.7; input(1, 1) =  0.2; input(1, 2) = -0.9;

    // Direct Dense forward (fresh instance with same init scheme would be
    // needed for a true bit-equality; in practice we just verify the LoRA
    // forward equals (base * something) where "something" has zero LoRA
    // contribution at B=0).
    Tensor y_lora = lora.forward(input);
    // The LoRA path is scaling · input · Aᵀ · Bᵀ = scaling · input · Aᵀ · 0 = 0
    // So y_lora == base_path_only == W_0_.forward(input) exactly.
    // We check by setting the rank-1 scaling test:
    // recompute the base-only output and compare.
    // ...
}
```

**Step 5:** Run test. Expected: PASS once impl is right.

---

## Task 5: FD gradient check (input, A, B)

Standard pattern (used everywhere in the repo):

```cpp
// Test 5: input FD gradient (rel_err < 1e-9 with random non-uniform init)
{
    LoRALinear lora(3, 4, 2);
    Tensor input(2, 3);
    fill_det(input, 42, 0.5);
    fd_check_input(lora, input, 1e-9);
}

// Test 6: A parameter FD gradient (rel_err < 1e-9)
{
    LoRALinear lora(3, 4, 2);
    fd_check_param(lora, "A", 1e-9);
}

// Test 7: B parameter FD gradient (rel_err < 1e-9)
{
    LoRALinear lora(3, 4, 2);
    fd_check_param(lora, "B", 1e-9);
}
```

---

## Task 6: merge_weights() round-trip

```cpp
// Test 8: merge then forward equals base-only Forward (B=0 means merge is no-op initially)
{
    LoRALinear lora(3, 4, 2);
    Tensor input(2, 3);
    fill_det(input, 7, 0.3);
    Tensor y_before = lora.forward(input);
    lora.merge_weights();
    Tensor y_after = lora.forward(input);
    // bit-exact equality (B was zero, so merge is a no-op)
    double max_diff = 0.0;
    for (size_t i = 0; i < y_before.data.size(); ++i)
        max_diff = std::max(max_diff, std::abs(y_before.data[i] - y_after.data[i]));
    check("Test 8: merge with B=0 is a no-op (y_before == y_after bit-exact)",
          max_diff < 1e-12);
}
```

---

## Task 7: train & verify loss decreases

```cpp
// Test 9: training reduces loss on a small regression problem
{
    LoRALinear lora(3, 2, 4);
    // ... fit y = W·x for some hand-constructed W ...
    // Verify lora.B is updated (moved from zero), loss decreases.
}
```

---

## Task 8: Register in umbrella and Makefile

**File:** `include/nn/nn.h`

Add `#include "nn/layers/utility/lora.h"` in the appropriate section (utility).

**File:** `Makefile`

Add:
- `$(BUILD_DIR)/test_lora: $(LIB_OBJS) $(BUILD_DIR)/test_lora.o`
- `$(BUILD_DIR)/test_lora.o` in LIB_OBJS (or appropriate object list)
- `$(BUILD_DIR)/test_lora` in the `tests:` deps line
- `@echo "=== Running LoRA Tests ===" && ./$(BUILD_DIR)/test_lora` at end of `run_tests`

---

## Step 9: Commit & push

```bash
git add include/nn/layers/utility/lora.h \
        include/nn/layers/utility/lora.cpp \
        tests/test_lora.cpp \
        include/nn/nn.h \
        Makefile \
        docs/plans/2026-09-18_110046-lora-low-rank-adaptation.md
git commit -m "feat(utility): LoRA — Low-Rank Adaptation (Hu et al. 2021)"
git push origin master
```

---

## Coverage targets (focused suite, ~25 checks)

| # | Check | Why |
|---|-------|-----|
| 1-9 | Constructor / accessors / defaults | Sanity |
| 10-12 | B=0 init, A scaled init, base-init scheme passed through | Paper §4.2 |
| 13 | Forward shape (N, d_out) | Sanity |
| 14 | **B=0 means LoRA forward ≡ base Dense forward bit-exact** | The single most important regression test |
| 15 | rank=0 path → forward ≡ base Dense forward (LoRA path is identically zero) | r=0 boundary |
| 16 | **alpha scaling: alpha=2·rank doubles the LoRA contribution** (vs alpha=rank) | Property |
| 17-19 | FD input/A/B gradients at machine precision with rank=2, 4, 8 | Correctness |
| 20 | **freeze_base=true → base weights not updated** (grad_base_W_ stays zero) | Freeze contract |
| 21 | **freeze_base=false → base weights ARE updated** (grad_base_W_ is nonzero) | Train-all mode |
| 22 | merge_weights() with B nonzero → W_0 grows by (α/rank)·B·A bit-exactly | Merge math |
| 23 | unmerge_weights() after merge → W_0 restored | Round-trip |
| 24 | **After merge, forward ≡ forward pre-merge** | Merge is inference-equivalent |
| 25 | Training reduces loss in 30 SGD steps | End-to-end sanity |
| 26-28 | Edge cases: rank > d_in/d_out (over-parameterized LoRA); d_in=1; d_out=1 | Boundary |
| 29-30 | Mutation: stub `dB = xAᵀ · grad_output * scaling` to zero → FD B fails | Non-vacuous |