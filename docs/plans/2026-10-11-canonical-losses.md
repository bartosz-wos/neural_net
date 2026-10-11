# Canonical Classification Losses (Cross-Entropy / NLL / BCEWithLogits) — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the first-class classification-loss family this repo has never had: `CrossEntropyLoss`, `NLLLoss`, and `BCEWithLogitsLoss`, with `ignore_index`, per-class `weight`, and the three PyTorch reductions.

**Architecture:** Three stateless-ish loss classes in `include/nn/utils/losses.{h,cpp}`, following the existing `FocalLoss` / `LabelSmoothingCrossEntropy` / `HuberLoss` convention (`forward(logits, targets) -> Tensor(1,1)`, `backward(logits, targets) -> grad of same shape as logits`). Shared private helpers for `log_softmax` and for the `ignore_index` / `weight` / reduction bookkeeping so the three classes cannot drift apart. `Reduction` is a new repo-wide `enum class`; it is declared in the new header and guarded against a future second definition.

**Tech Stack:** C++17, the repo's `Tensor` (2-D, real, `double`), plain `std::vector` — no new dependencies.

---

## Phase 0 — Verification of the gap (DONE 2026-10-11)

Searched by content, not filename:

| Symbol | Result |
|---|---|
| `CrossEntropyLoss`, `NLLLoss`, `BCEWithLogitsLoss` | **ABSENT** |
| `ignore_index` | **ABSENT** (grep across all of `include/`) |
| `class_weight` / `weight` on any loss | **ABSENT** |
| `Reduction` enum / `reduction=` arg on any loss | **ABSENT** (`cbam.h:15` `reduction` is a *channel* MLP ratio, `tome.h` is a comment) |

What exists instead: `Softmax::cross_entropy_loss` / `Softmax::cross_entropy_grad` (two static helpers on the activation struct that **require one-hot labels** and hardcode mean reduction), `FocalLoss`, `LabelSmoothingCrossEntropy`, `HuberLoss`, `QuantileLoss`, `KLDivergence`, `JSDivergence`, `triplet_loss_siamese`, `contrastive_losses`, `self_supervised_losses`, `siglip_loss`.

**Why this is the right next feature and not a re-pop.** The queue's `## Ideas` block was fully consumed by Grad-CAM on 2026-10-10; the six interpretability families are complete. The deepest gap in a library with ~250 layers, 38+ optimizers and 6 interpretability methods is that **it has no canonical way to train a classifier**. Every existing loss is either a special case (focal, quantile, KL), a variant of CE that duplicates this work under a different name (`LabelSmoothingCrossEntropy`), or a non-classification objective. Nothing handles class indices, `ignore_index` (padding masks — the single most common real-world need in sequence/NLP models already in this repo), or per-class weights (imbalanced classes — which is exactly what the shipped `FocalLoss` was written to work around, rather than addressing directly).

**Overlap that must be handled, not ignored.** `LabelSmoothingCrossEntropy` and `FocalLoss` overlap with what this adds. Per YAGNI and "lean toward fixes/refactors over deletions": **do not refactor them**. They are shipped, tested, mutation-verified, and consumed. This feature adds the canonical entry points; a future session can decide whether `LabelSmoothingCrossEntropy` becomes a thin wrapper. Record that as a follow-up, do not do it now.

## Semantics — read from PyTorch source, not memory (2026-10-11)

All from `aten/src/ATen/native/LossNLL.cpp` and `Loss.cpp`, fetched and read at the cited lines.

**NLL forward** (`LossNLL.cpp:252-304`), with `nllloss_nd(input, target, weight, reduction, ignore_index)`:
- Per non-ignored row: `loss_b = -input[b][target[b]] * w[target[b]]`, else `loss_b = 0`.
- `Reduction::Sum` → plain sum of `loss_b`.
- `Reduction::Mean` → `sum(loss_b) / total_weight` where **`total_weight = batch_size - num_ignored` when no class weight is given** (`:288-289`), and `= Σ_{b non-ignored} w[target[b]]` when a weight IS given (`:290-292`).
  **This is the single most mis-implemented detail in every hand-rolled version of this loss, and the one this plan's headline test targets.** The divisor is *not* the batch size whenever `ignore_index` or class weights are in play.
- `Reduction::None` → per-row vector of `loss_b`.
- Out-of-bounds target → error (`:259-262`).

**NLL backward** (`LossNLL.cpp:345-416`):
- `grad_input[b][target[b]] = -w[target[b]] * grad_output / total_weight` for `Reduction::Mean`, `= -w[target[b]] * grad_output` for `Sum`, and `= -w[target[b]] * grad_output[b]` for `None` (per-row `grad_output[b]`, `:378`).
- Ignored rows get an all-zero gradient row (`:373-374`, `:409-410`).

**Cross-entropy** (`LossNLL.cpp:632+`) is `nll_loss(log_softmax(logits), ...)` — softmax is folded into the loss, so callers pass **raw logits**. With `label_smoothing = 0` the prob-target branch (`:549-557`) reduces to exactly the same `-Σ_j t_j · logsoftmax_j` and its mean divides by `input.numel() / n_classes` = batch size.

**Class-weight broadcasting in the prob-target branch** (`:524-546`): weight is a `(C,)` vector viewed as `(1, C, 1, …)` and multiplies elementwise; the mean divides the **weighted** element-sum by the **unweighted** `numel / n_classes` (batch size). Note this **differs from the index-target branch**, which divides by `Σ w[target_b]`. Both are recorded in the header; `CrossEntropyLoss` here takes index targets only (the prob-target branch is out of scope — see Out of scope).

**BCEWithLogits** (`Loss.cpp:349-364`): `loss = (1 - target) * input - log_sigmoid(input)`, then `* weight` elementwise, then reduce. `log_sigmoid(x) = -softplus(-x)`, computed stably. `pos_weight` reweights the positive term as `log_weight = (pos_weight - 1) * target + 1`, multiplying into `log_sigmoid_input` (`:351-355`). `Reduction::Mean` divides by `numel` (not by a weight sum) via `apply_loss_reduction`.

**Semantics the repo must NOT get wrong, listed as decisions:**

| # | Decision | Rationale |
|---|---|---|
| D1 | `total_weight` = `B - n_ignored` (unweighted) or `Σ_{b ∉ ignored} w[t_b]` (weighted). Mean divides by it. | `LossNLL.cpp:288-292`. A batch-size divisor is wrong the moment `ignore_index` or class weights exist. |
| D2 | `ignore_index` is **per-row** and its gradient row is **exactly zero**, not merely small. | `:373-374`, `:409-410`. `log_softmax` is not zero for ignored rows; forgetting this adds spurious gradient. |
| D3 | Class-weight `mean` divides by the **weight sum**, not the batch size. | `:290-292`. Distinct from the prob-target branch (`:539`) — a genuine PyTorch inconsistency, documented, not silently unified. |
| D4 | Callers pass **raw logits** to CE; no prior softmax. | `LossNLL.cpp:632+`. |
| D5 | Targets are **class indices in an `(N,1)` tensor** (repo convention: everything is 2-D). Both `FocalLoss` and `LabelSmoothingCrossEntropy` read `targets[b][0]`. | Repo-wide 2-D `Tensor`; matching the neighbours beats matching PyTorch's `(N,)`. |
| D6 | `Reduction` is a new repo-wide `enum class` named `Reduction`, guarded by `#ifndef NN_REDUCTION_ENUM_DEFINED`. | Verified no collision. The guard makes a future second definition a compile error instead of a silent redeclaration. |
| D7 | `BCEWithLogitsLoss` mean divides by `numel` (= `N*C` for a `(N,C)` tensor), **not** by `N`. | `apply_loss_reduction` at `Loss.cpp:363`. This is a real trap: "mean over batch" and "mean over elements" differ for a `(N,C)` tensor and only coincide at `C=1`. |
| D8 | Out-of-range target throws `std::invalid_argument`. | PyTorch errors (`:259-262`); a silent wrap/UB would be worse. |

---

## Task 1: `Reduction` enum + header skeleton

**Objective:** Declare the shared `Reduction` enum and the three loss class interfaces. No implementations.

**Files:**
- Create: `include/nn/utils/losses.h`

**Step 1: Write the header**

```cpp
#ifndef LOSSES_H
#define LOSSES_H

#include "../core/tensor.h"
#include <string>

// Reduction modes, matching PyTorch's `reduction=` argument
// (aten/src/ATen/native/LossNLL.cpp:542-546).
#ifndef NN_REDUCTION_ENUM_DEFINED
#define NN_REDUCTION_ENUM_DEFINED
enum class Reduction { None, Mean, Sum };
#endif

// ---------------------------------------------------------------------------
// log_softmax helpers (public: callers building a custom loss need them, and
// they are the numerically-stable path CrossEntropyLoss itself takes).
// log_softmax(x)[i][j] = x[i][j] - max_j x[i][j] - log(sum_k exp(x[i][k] - max))
// The max subtraction is what keeps a +1e4 logit row from overflowing exp().
// ---------------------------------------------------------------------------
Tensor log_softmax(const Tensor& x);
Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& log_probs);

// ---------------------------------------------------------------------------
// NLLLoss — negative log-likelihood of an ALREADY-log-softmax'd input.
// loss_b = -input[b][target[b]] * w[target[b]]   (0 for ignored rows)
// mean: sum / total_weight, total_weight = B - n_ignored, or the weight sum
//       when class weights are given (LossNLL.cpp:288-292 — D1)
// ---------------------------------------------------------------------------
class NLLLoss {
public:
    NLLLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100);
    void set_weight(const Tensor& weight);          // (1, C) or (C, 1)
    Tensor forward(const Tensor& log_probs, const Tensor& targets) const;
    Tensor backward(const Tensor& log_probs, const Tensor& targets,
                    const Tensor& grad_output) const;
    // Cache `total_weight` from the last forward so backward can use it.
    double total_weight() const { return total_weight_; }
    Reduction reduction() const { return reduction_; }
    int ignore_index() const { return ignore_index_; }

private:
    void compute_total_weight(const Tensor& targets, size_t num_classes) const;
    Reduction reduction_;
    int ignore_index_;
    Tensor weight_;
    mutable double total_weight_ = 1.0;
};

// ---------------------------------------------------------------------------
// CrossEntropyLoss — softmax cross-entropy over RAW LOGITS (D4).
// Equivalent to NLLLoss(log_softmax(logits)) but folds the softmax in and
// computes the gradient in logit space directly.
// ---------------------------------------------------------------------------
class CrossEntropyLoss {
public:
    CrossEntropyLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100);
    void set_weight(const Tensor& weight);
    Tensor forward(const Tensor& logits, const Tensor& targets) const;
    Tensor backward(const Tensor& logits, const Tensor& targets,
                    const Tensor& grad_output) const;
    double total_weight() const { return total_weight_; }
    Reduction reduction() const { return reduction_; }
    int ignore_index() const { return ignore_index_; }

private:
    Reduction reduction_;
    int ignore_index_;
    Tensor weight_;
    mutable double total_weight_ = 1.0;
};

// ---------------------------------------------------------------------------
// BCEWithLogitsLoss — binary cross-entropy from logits (no sigmoid needed).
// loss = (1 - target) * input - log_sigmoid(input), stably computed as
//        max(x,0) - x*target + log1p(exp(-|x|))   (Loss.cpp:349-364)
// mean divides by numel = rows*cols, NOT by rows (D7).
// ---------------------------------------------------------------------------
class BCEWithLogitsLoss {
public:
    BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);
    void set_weight(const Tensor& weight);      // elementwise, same shape as input
    void set_pos_weight(const Tensor& pos_weight);  // (1, 1); >= 0
    Tensor forward(const Tensor& logits, const Tensor& targets) const;
    Tensor backward(const Tensor& logits, const Tensor& targets,
                    const Tensor& grad_output) const;
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
    Tensor weight_;
    Tensor pos_weight_;
};

#endif
```

**Step 2: Compile the header standalone**

Run:
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/utils/losses.h"'
```
Expected: no output (exit 0). A focused test including only this header cannot catch cross-header collisions, which is why this step is separate from the umbrella check in Task 5.

**Step 3: Commit**

```bash
git add include/nn/utils/losses.h
git commit -m "feat(losses): header for CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss"
```

---

## Task 2: `log_softmax` + RED test for NLLLoss

**Objective:** Implement the numerically-stable `log_softmax` helper and write the first failing NLL test.

**Files:**
- Create: `include/nn/utils/losses.cpp`
- Create: `tests/test_losses.cpp`

**Step 1: Write the failing test (RED)**

In `tests/test_losses.cpp`, using the repo's existing harness idiom (copy the `ASSERT` / `ASSERT_NEAR` / `run(...)` macros verbatim from `tests/test_label_smoothing.cpp:29-57`):

```cpp
#include "nn/utils/losses.h"
#include "nn/core/tensor.h"

// A hand-derived log_softmax on a 2-class row. logit = [0, ln 2].
// softmax([0, ln2]) = [1/3, 2/3]  (exp(0)=1, exp(ln2)=2, sum=3)
// log_softmax      = [-ln 3, ln2 - ln 3]
static void test_log_softmax_hand() {
    run("(1a) log_softmax hand-derived", [] {
        Tensor x(1, 2); x[0][0] = 0.0; x[0][1] = std::log(2.0);
        Tensor lp = log_softmax(x);
        ASSERT_NEAR(lp[0][0], -std::log(3.0), 1e-12);
        ASSERT_NEAR(lp[0][1], std::log(2.0) - std::log(3.0), 1e-12);
        // softmax of a log_softmax row sums to 1
        double s = std::exp(lp[0][0]) + std::exp(lp[0][1]);
        ASSERT_NEAR(s, 1.0, 1e-12);
    });
}

// NLL of a PERFECT prediction is ~0; of a maximally wrong one it is large.
static void test_nll_perfect_and_wrong() {
    run("(1b) NLL perfect vs wrong", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor lp(2, 2);
        // row 0: target 0, log-prob ~ -1e-9  (near-certain)
        lp[0][0] = -1e-9; lp[0][1] = -20.0;
        // row 1: target 1, log-prob ~ -20    (wrong, target has tiny prob)
        lp[1][0] = -20.0; lp[1][1] = -20.0;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        Tensor perfect = loss.forward(lp, t);
        ASSERT(perfect[0][0] >= 0.0);
        ASSERT(perfect[0][0] < 1e-6);
        // Uniform log-probs (-ln2) for K=2 give loss ln2 exactly.
        Tensor uni(1, 2); uni[0][0] = -std::log(2.0); uni[0][1] = -std::log(2.0);
        Tensor t1(1, 1); t1[0][0] = 0.0;
        NLLLoss l2(Reduction::Mean, -100);
        ASSERT_NEAR(l2.forward(uni, t1)[0][0], std::log(2.0), 1e-12);
    });
}
```

**Step 2: Run the test to verify it fails (RED)**

Run:
```bash
make -j8 build/test_losses 2>&1 | tail -5
```
Expected: **compile/link failure** — `log_softmax` and `NLLLoss` are declared but not defined. This is the expected RED for a from-scratch file; the harness must also be able to compile once the `.cpp` exists but before the bodies do. See Task 3 Step 0.

**Step 3: Commit**

```bash
git add tests/test_losses.cpp
git commit -m "test(losses): RED suite — NLL perfect/wrong, log_softmax hand-derived"
```

---

## Task 3: Implement `log_softmax`, `NLLLoss` forward/backward

**Objective:** Turn the suite GREEN for NLL.

**Files:**
- Create: `include/nn/utils/losses.cpp`

**Step 1: `log_softmax`**

```cpp
#include "losses.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Tensor log_softmax(const Tensor& x) {
    Tensor out(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double mx = x[i][0];
        for (size_t j = 0; j < x.cols; ++j) mx = std::max(mx, x[i][j]);
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) sum += std::exp(x[i][j] - mx);
        const double lse = mx + std::log(sum);
        for (size_t j = 0; j < x.cols; ++j) out[i][j] = x[i][j] - lse;
    }
    return out;
}

Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& log_probs) {
    // dL/dx_j = g_j - softmax(x)_j * sum_k g_k
    Tensor sum_grad(grad_output.rows, 1);
    for (size_t i = 0; i < grad_output.rows; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < grad_output.cols; ++j) s += grad_output[i][j];
        sum_grad[i][0] = s;
    }
    Tensor out(grad_output.rows, grad_output.cols);
    for (size_t i = 0; i < grad_output.rows; ++i)
        for (size_t j = 0; j < grad_output.cols; ++j)
            out[i][j] = grad_output[i][j] -
                        std::exp(log_probs[i][j]) * sum_grad[i][0];
    return out;
}
```

**Step 2: `NLLLoss`**

```cpp
NLLLoss::NLLLoss(Reduction r, int ignore_index)
    : reduction_(r), ignore_index_(ignore_index) {}

void NLLLoss::set_weight(const Tensor& w) {
    if (w.rows * w.cols != 1 && w.rows != 1 && w.cols != 1)
        throw std::invalid_argument("NLLLoss: weight must have one entry per class");
    weight_ = w.clone();
}

void NLLLoss::compute_total_weight(const Tensor& targets, size_t num_classes) const {
    if (weight_.rows == 0 || weight_.cols == 0) {
        size_t kept = 0;
        for (size_t b = 0; b < targets.rows; ++b)
            if (static_cast<int>(targets[b][0]) != ignore_index_) ++kept;
        total_weight_ = static_cast<double>(kept);
    } else {
        double s = 0.0;
        for (size_t b = 0; b < targets.rows; ++b) {
            int t = static_cast<int>(targets[b][0]);
            if (t == ignore_index_) continue;
            s += weight_[t % weight_.rows][0];
        }
        total_weight_ = s;
    }
    (void)num_classes;
}
```

`forward` returns `(1,1)` for `Sum`/`Mean` and a `(B,1)` tensor for `None`; `backward` puts `-w[t_b] * g / total_weight` at `[b][t_b]` and zero elsewhere (ignored rows entirely zero). `Reduction::None` backward takes a per-row `grad_output` of shape `(B,1)` and writes `-w[t_b]*grad_output[b][0]` at `[b][t_b]` (`LossNLL.cpp:378`). Validate `0 <= t < cols` and throw `std::invalid_argument` otherwise (D8).

**Step 3: Run to verify GREEN**

```bash
make -j8 build/test_losses && ./build/test_losses
```
Expected: all assertions pass, summary line prints `N passed, 0 failed`, exit 0.

**Step 4: Mutation check (do not skip)**

Break `total_weight_` to `static_cast<double>(targets.rows)` (the batch-size divisor this plan exists to avoid) and confirm the `ignore_index` test from Task 4 fails. Confirm it *does* fail before restoring — a mutation that passes means the test is vacuous.

**Step 5: Commit**

```bash
git add include/nn/utils/losses.cpp
git commit -m "feat(losses): log_softmax + NLLLoss forward/backward"
```

---

## Task 4: `ignore_index`, class weight, reductions — the headline tests

**Objective:** Pin D1/D2/D3, the three details every hand-rolled version gets wrong.

**Files:**
- Modify: `tests/test_losses.cpp`

**Step 1: Write the failing tests (RED for the ignore/weight paths)**

```cpp
// D1 — mean divides by (B - n_ignored), NOT by B. This is the headline test.
static void test_ignore_index_mean_divisor() {
    run("(4a) ignore_index mean divisor", [] {
        NLLLoss loss(Reduction::Mean, /*ignore_index=*/1);
        Tensor lp(2, 2);
        lp[0][0] = 0.0; lp[0][1] = -std::log(3.0);  // target 0 -> loss 0
        lp[1][0] = 0.0; lp[1][1] = -1.0;            // ignored row
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        // Only row 0 contributes and its loss is 0, so the loss is 0 either
        // way -- the DIVISOR is what the second half pins.
        Tensor out = loss.forward(lp, t);
        ASSERT_NEAR(out[0][0], 0.0, 1e-12);
        ASSERT_NEAR(loss.total_weight(), 1.0, 0.0);   // 2 - 1 ignored
        // Give row 0 a non-zero loss so the divisor is observable.
        lp[0][0] = -1.0; lp[0][1] = -std::log(3.0);
        out = loss.forward(lp, t);
        ASSERT_NEAR(out[0][0], 1.0, 1e-12);            // 1.0 / 1.0
        ASSERT_NEAR(loss.total_weight(), 1.0, 0.0);
    });
}

// D3 — with class weights, mean divides by the WEIGHT SUM.
static void test_weighted_mean_divisor() {
    run("(4b) class-weight mean divisor", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor w(1, 3); w[0][0] = 1.0; w[0][1] = 2.0; w[0][2] = 3.0;
        loss.set_weight(w);
        Tensor lp(2, 3);
        lp[0][0] = -1.0; lp[0][1] = -5.0; lp[0][2] = -5.0;  // t=0, w=1 -> 1.0
        lp[1][0] = -5.0; lp[1][1] = -2.0; lp[1][2] = -5.0;  // t=1, w=2 -> 4.0
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        Tensor out = loss.forward(lp, t);
        ASSERT_NEAR(loss.total_weight(), 3.0, 0.0);      // 1 + 2
        ASSERT_NEAR(out[0][0], 5.0 / 3.0, 1e-12);
    });
}

// D2 — an ignored row's gradient is EXACTLY zero, not merely small.
static void test_ignored_row_gradient_is_zero() {
    run("(4c) ignored row grad exactly zero", [] {
        NLLLoss loss(Reduction::Mean, 1);
        Tensor lp(2, 2);
        lp[0][0] = -0.5; lp[0][1] = -2.0;
        lp[1][0] = -0.7; lp[1][1] = -0.3;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;   // row 1 ignored
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = loss.backward(lp, t, go);
        ASSERT_NEAR(g[0][0], -1.0, 1e-12);
        ASSERT(g[0][1] == 0.0);
        ASSERT(g[1][0] == 0.0 && g[1][1] == 0.0);   // EXACTLY zero
    });
}

// Reduction::Sum must NOT divide; Reduction::None returns per-row.
static void test_sum_and_none() {
    run("(4d) sum and none reductions", [] {
        Tensor lp(2, 2);
        lp[0][0] = -1.0; lp[0][1] = -2.0;
        lp[1][0] = -3.0; lp[1][1] = -4.0;
        Tensor t(2, 1); t[0][0] = 0.0; t[1][0] = 1.0;
        NLLLoss sum(Reduction::Sum, -100);
        ASSERT_NEAR(sum.forward(lp, t)[0][0], 1.0 + 4.0, 1e-12);
        NLLLoss none(Reduction::None, -100);
        Tensor per = none.forward(lp, t);
        ASSERT(per.rows == 2 && per.cols == 1);
        ASSERT_NEAR(per[0][0], 1.0, 1e-12);
        ASSERT_NEAR(per[1][0], 4.0, 1e-12);
    });
}

// Out-of-range target must throw, not wrap or read out of bounds.
static void test_out_of_range_throws() {
    run("(4e) out-of-range target throws", [] {
        NLLLoss loss(Reduction::Mean, -100);
        Tensor lp(1, 2); lp[0][0] = -0.5; lp[0][1] = -0.5;
        Tensor t(1, 1); t[0][0] = 7.0;
        bool threw = false;
        try { loss.forward(lp, t); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    });
}
```

**Step 2: Run to verify the new tests FAIL for the right reason**

```bash
make -j8 build/test_losses && ./build/test_losses; echo "exit=$?"
```
Expected: tests (4a)–(4e) fail; earlier tests still pass. A test that fails because of a typo is not RED — re-read the failure message.

**Step 3: Implement the ignore/weight paths** in `losses.cpp` per D1/D2/D3.

**Step 4: Run to verify GREEN**

```bash
./build/test_losses; echo "exit=$?"
```
Expected: `0 failed`, exit 0.

**Step 5: Mutation-check all three decisions.** Three separate mutations, each must be caught:
- divisor `total_weight_` → batch size → (4a) must fail.
- ignored-row gradient `-= ...` → `+=` or drop the `continue` → (4c) must fail.
- weighted divisor → batch size → (4b) must fail.

**Step 6: Commit**

```bash
git add include/nn/utils/losses.cpp tests/test_losses.cpp
git commit -m "feat(losses): ignore_index, class weights, three reductions"
```

---

## Task 5: `CrossEntropyLoss`

**Objective:** CE over raw logits, folding in the softmax.

**Files:**
- Modify: `include/nn/utils/losses.cpp`

**Step 1: Write the failing test (RED)**

```cpp
// CE over logits == NLL(log_softmax(logits)). The equivalence IS the spec.
static void test_ce_equals_nll_of_log_softmax() {
    run("(5a) CE == NLL(log_softmax(logits))", [] {
        Tensor logits(3, 4);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                logits[i][j] = std::sin(3.0 * i + 1.7 * j) * 2.0;
        Tensor t(3, 1); t[0][0] = 0.0; t[1][0] = 3.0; t[2][0] = 2.0;
        NLLLoss nll(Reduction::Mean, -100);
        Tensor expected = nll.forward(log_softmax(logits), t);
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor got = ce.forward(logits, t);
        ASSERT_NEAR(got[0][0], expected[0][0], 1e-12);
    });
}

// Gradient: dCE/dlogits = softmax - onehot, over the batch. Known closed form.
static void test_ce_gradient_closed_form() {
    run("(5b) CE gradient == softmax - onehot", [] {
        Tensor logits(2, 3);
        logits[0][0] = 0.5; logits[0][1] = -1.0; logits[0][2] = 2.0;
        logits[1][0] = 0.0; logits[1][1] = 0.25; logits[1][2] = -0.5;
        Tensor t(2, 1); t[0][0] = 1.0; t[1][0] = 2.0;
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = ce.backward(logits, t, go);
        for (size_t i = 0; i < 2; ++i) {
            Tensor p = log_softmax(logits).apply([](double v){ return std::exp(v); });
            double rowsum = 0.0;
            for (size_t j = 0; j < 3; ++j) rowsum += g[i][j];
            ASSERT_NEAR(rowsum, 0.0, 1e-12);   // softmax-1 sums to 0 per row
            for (size_t j = 0; j < 3; ++j) {
                double onehot = (j == static_cast<size_t>(t[i][0])) ? 1.0 : 0.0;
                ASSERT_NEAR(g[i][j], (p[i][j] - onehot) / 2.0, 1e-12);
            }
        }
    });
}

// D4: passing ALREADY-softmaxed values must give a DIFFERENT, larger loss.
static void test_ce_expects_raw_logits() {
    run("(5c) CE expects raw logits, not probabilities", [] {
        Tensor probs(1, 3); probs[0][0] = 0.2; probs[0][1] = 0.5; probs[0][2] = 0.3;
        Tensor t(1, 1); t[0][0] = 0.0;
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor as_probs = ce.forward(probs, t);
        // Correct usage: -log p_0 = 1.609
        ASSERT_NEAR(as_probs[0][0], 1.6094379124341003, 1e-9);
        // Feeding the true LOGITS [ln .2, ln .5, ln .3] must give the same value.
        Tensor logits(1, 3);
        logits[0][0] = std::log(0.2); logits[0][1] = std::log(0.5);
        logits[0][2] = std::log(0.3);
        ASSERT_NEAR(ce.forward(logits, t)[0][0], as_probs[0][0], 1e-12);
    });
}

// FD oracle: the gradient is smooth in the logits, so centered FD is valid here.
static void test_ce_input_gradient_fd() {
    run("(5d) CE input gradient vs finite differences", [] {
        CrossEntropyLoss ce(Reduction::Mean, -100);
        Tensor logits(2, 3);
        logits[0][0] = 0.3; logits[0][1] = -0.8; logits[0][2] = 1.1;
        logits[1][0] = -0.2; logits[1][1] = 0.9; logits[1][2] = 0.4;
        Tensor t(2, 1); t[0][0] = 2.0; t[1][0] = 0.0;
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = ce.backward(logits, t, go);
        const double eps = 1e-6;
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 3; ++j) {
            Tensor up = logits.clone(); up[i][j] += eps;
            Tensor dn = logits.clone(); dn[i][j] -= eps;
            double num = (ce.forward(up, t)[0][0] - ce.forward(dn, t)[0][0]) / (2 * eps);
            double rel = std::abs(num - g[i][j]) / std::max(1e-9, std::abs(num));
            ASSERT(rel < 1e-5);
        }
    });
}
```

**Step 2: Run, confirm RED.** **Step 3: Implement.** CE `forward` = `NLLLoss::forward(log_softmax(logits), targets)`; CE `backward` = `log_softmax_backward(softmax - onehot-weighted, log_softmax(logits))`, with the ignored-row rows forced to exactly zero (D2).

**Step 4: Run, confirm GREEN + full suite.** **Step 5: Commit.**

---

## Task 6: `BCEWithLogitsLoss`

**Objective:** Binary CE from logits, stable and correctly reduced.

**Files:**
- Modify: `include/nn/utils/losses.cpp`

**Step 1: Write the failing test (RED)**

```cpp
// Stable form: max(x,0) - x*t + log1p(exp(-|x|))  (Loss.cpp:349-364)
static void test_bce_hand_derived() {
    run("(6a) BCE hand-derived", [] {
        BCEWithLogitsLoss loss(Reduction::None);
        Tensor z(2, 1); z[0][0] = 0.0; z[1][0] = 2.0;
        Tensor t(2, 1); t[0][0] = 1.0; t[1][0] = 0.0;
        Tensor per = loss.forward(z, t);
        ASSERT(per.rows == 2 && per.cols == 1);
        ASSERT_NEAR(per[0][0], std::log(2.0), 1e-12);                       // -log s(0)
        ASSERT_NEAR(per[1][0], 2.0 + std::log1p(std::exp(-2.0)), 1e-12);    // -log s(-2)
    });
}

// D7: mean divides by numel = rows*cols, NOT by rows. Differs from mean
// over batch the moment cols > 1.
static void test_bce_mean_divides_by_numel() {
    run("(6b) BCE mean divides by numel not rows", [] {
        Tensor z(2, 3); z.fill(0.0);
        Tensor t(2, 3); t.fill(1.0);
        BCEWithLogitsLoss none(Reduction::None);
        Tensor per = none.forward(z, t);
        double total = per[0][0] + per[0][1] + per[0][2] +
                       per[1][0] + per[1][1] + per[1][2];
        BCEWithLogitsLoss mean(Reduction::Mean);
        ASSERT_NEAR(mean.forward(z, t)[0][0], total / 6.0, 1e-12);
    });
}

// Extreme logits must not overflow: |z| = 1e4 gives a finite loss.
static void test_bce_extreme_logits() {
    run("(6c) BCE stable at |z| = 1e4", [] {
        BCEWithLogitsLoss loss(Reduction::Mean);
        Tensor z(1, 1); z[0][0] = 1e4;
        Tensor t(1, 1); t[0][0] = 0.0;
        double v = loss.forward(z, t)[0][0];
        ASSERT(std::isfinite(v));
        ASSERT_NEAR(v, 1e4, 1e-6);           // loss ≈ max(x,0) - x*t = 1e4
        Tensor g = loss.backward(z, t, Tensor(1,1));
        ASSERT(std::isfinite(g[0][0]));
    });
}

// FD oracle on the logit gradient.
static void test_bce_gradient_fd() {
    run("(6d) BCE gradient vs finite differences", [] {
        BCEWithLogitsLoss loss(Reduction::Mean);
        Tensor z(2, 2);
        z[0][0] = 0.4; z[0][1] = -1.2; z[1][0] = 0.7; z[1][1] = 2.1;
        Tensor t(2, 2);
        t[0][0] = 1.0; t[0][1] = 0.0; t[1][0] = 0.0; t[1][1] = 1.0;
        Tensor go(1, 1); go[0][0] = 1.0;
        Tensor g = loss.backward(z, t, go);
        const double eps = 1e-6;
        for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) {
            Tensor up = z.clone(); up[i][j] += eps;
            Tensor dn = z.clone(); dn[i][j] -= eps;
            double num = (loss.forward(up, t)[0][0] - loss.forward(dn, t)[0][0]) / (2 * eps);
            ASSERT(std::abs(num - g[i][j]) / std::max(1e-9, std::abs(num)) < 1e-5);
        }
    });
}

// pos_weight reweights the positive term (Loss.cpp:351-355).
static void test_bce_pos_weight() {
    run("(6e) BCE pos_weight", [] {
        BCEWithLogitsLoss loss(Reduction::None);
        Tensor pw(1, 1); pw[0][0] = 2.0;
        loss.set_pos_weight(pw);
        Tensor z(1, 2); z[0][0] = 0.5; z[0][1] = -0.5;
        Tensor t(1, 2); t[0][0] = 1.0; t[0][1] = 0.0;
        Tensor per = loss.forward(z, t);
        BCEWithLogitsLoss plain(Reduction::None);
        double log_sigmoid = std::log(1.0 + std::exp(-0.5));
        // positive column gains an extra factor pos_weight on the log_sigmoid
        ASSERT_NEAR(per[0][0], 0.5 - 2.0 * log_sigmoid, 1e-12);
        ASSERT_NEAR(per[0][1], -0.5 - log_sigmoid, 1e-12);
        (void)plain;
    });
}
```

**Step 2: Run, confirm RED.** **Step 3: Implement** per the stable form, `pos_weight`, elementwise `weight`, and the D7 divisor. **Step 4: GREEN.** **Step 5: Commit.**

---

## Task 7: Wire into `nn.h` and the Makefile

**Objective:** Register the header and the test so `make run_tests` executes it.

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`

**Step 1: Add the include** to `include/nn/nn.h` next to the other `utils/` losses (e.g. after `#include "utils/focal_loss.h"` if present, else after `utils/label_smoothing.h`).

**Step 2: Verify the umbrella header compiles STANDALONE** (the check a focused test cannot do):

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: no output, exit 0.

**Step 3: Add the Makefile rule**, matching the existing per-test shape:
```make
$(BUILD_DIR)/test_losses: $(LIB_OBJS) $(BUILD_DIR)/test_losses.o
	$(CXX) $^ -o $@
```
Then add `$(BUILD_DIR)/test_losses` to the `tests:` prerequisite list and a `run_tests` line:
```make
	@echo "=== Running test_losses ==="; if ./$(BUILD_DIR)/test_losses; then :; else echo "test_losses" >> .run_tests_failed; fi
```
**The `tests:` prerequisite line is the one that gets forgotten** — `NOT_FIXED.md`'s 2026-10-04 entry records exactly this bug for `test_mixture_of_softmaxes` (a link rule and a `run_tests` line but no `tests:` entry, so the binary was never built).

**Step 4: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore(build): register losses in nn.h and the test suite"
```

---

## Task 8: Full-suite regression + docs

**Objective:** No regressions; queue and NOT_FIXED updated.

**Files:**
- Modify: `EXPANSION_QUEUE.md`

**Step 1: Run the full suite**
```bash
make -j8 run_tests 2>&1 | tail -20
cat .run_tests_failed 2>/dev/null
```
Expected: `test_losses` absent from `.run_tests_failed`. Pre-existing failures are listed in `NOT_FIXED.md` and are expected (`test_neural_ode` is the only real one under the aggregate target).

**Step 2: Move the entry to `## Done`** with a one-line summary and the check count.

**Step 3: Commit**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark canonical losses Done in EXPANSION_QUEUE"
```

---

## Self-contradiction scan

- The three decisions D1/D2/D3 are stated once each, in the Decisions table, and each is pinned by exactly one named test: (4a), (4c), (4b) respectively. No task states a conflicting divisor.
- `backward` signature is `(logits, targets, grad_output)` in the header, in Task 3, and in every test. Consistent.
- Task 2 says "compile/link failure" is the expected RED because the `.cpp` does not exist yet; Task 3 makes the file exist. That is stated explicitly in Task 2 Step 2 rather than left for the implementer to discover.
- Out of scope is stated once (prob-target CE branch, 2-D input `(C,)` single-sample mode, `(N, C, d1, d2)` spatial NLL) and no task contradicts it.

## Out of scope for v1

- The **probability-target** CE branch (`LossNLL.cpp:498-563`) and its different mean divisor (`numel/n_classes`, D3's exception). Index targets only.
- Single-sample `(C,)` input mode (the repo is 2-D throughout).
- `nll_loss2d` / `nll_loss_nd` for `(N,C,d1,d2)` spatial inputs (`LossNLL.cpp:126-133`). The repo's segmentation losses cover that surface.
- Refactoring `LabelSmoothingCrossEntropy` / `FocalLoss` into wrappers over the new class. Recorded as a follow-up; doing it now would touch two shipped, mutation-verified suites for no user-visible gain.

## References

- PyTorch `aten/src/ATen/native/LossNLL.cpp` — read 2026-10-11: `nll_loss_out_frame:166-305` (forward, total_weight), `nll_loss_backward_out_frame:345-416` (backward, ignored rows), `cross_entropy_loss_prob_target:498-563`, `cross_entropy_loss_label_smoothing:566-630`.
- PyTorch `aten/src/ATen/native/Loss.cpp:349-364` — `binary_cross_entropy_with_logits`.
- Repo neighbours read for convention: `include/nn/utils/focal_loss.{h,cpp}`, `include/nn/utils/label_smoothing.h`, `include/nn/utils/distribution_losses.h`, `include/nn/activations/activations.h:30-43`.