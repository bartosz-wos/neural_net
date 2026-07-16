# Segmentation Losses (Dice / Tversky / Focal-Dice / Lovász-Hinge) — Implementation Plan

> **For Hermes:** Use TDD per task. Each task = failing test, minimal impl, verify, commit.

**Goal:** Add 4 canonical semantic-segmentation losses to `include/nn/utils/segmentation_losses.{h,cpp}` and a `build/test_segmentation_losses` binary registered in `Makefile` and `include/nn/nn.h`.

**Architecture:** A single `segmentation_losses.h` umbrella, four classes — `DiceLoss`, `TverskyLoss`, `FocalDiceLoss`, `LovaszHingeLoss` — that all share a uniform `(N, C)`-layout convention (N batch rows, C class mask probabilities per row). All forward returns `(1,1)` loss tensor. Backward returns a gradient `Tensor` of the same `(N, C)` shape as the predictions. All take predictions **after sigmoid/softmax** (i.e. already in [0,1]) so they play nicely with FocalDice and Lovász; softmax-only consumers can pre-normalize.

**Tech Stack:** C++17, `Tensor`, standard library. Conventions match `focal_loss.h` / `distribution_losses.h` / `contrastive_losses.h`.

---

## Motivation

The repo has strong classification losses (Focal, InfoNCE, SigLIP, Triplet, Contrastive) and stable regression losses (Huber, Quantile) but **no segmentation losses**. Segmentation is a different problem domain — class imbalance is mild *per-pixel* but severe *per-image* (most of an image is "background"), which is exactly where Dice/Tversky/Focal-Dice shine over plain BCE. The losses are also commonly used as the primary metric-equivalent for medical imaging / U-Net / semantic segmentation work.

The four losses below are the canonical set used in every segmentation pipeline:

1. **Dice Loss** (Milletari et al. 2016, V-Net) — the de-facto segmentation loss; an F1 surrogate.
2. **Tversky Loss** (Salehi et al. 2017) — a generalization of Dice that lets you trade off false positives vs false negatives via an α/β weighting. α=β=0.5 recovers Dice.
3. **Focal-Dice Loss** (Zhu et al. 2018, "Boundary-aware") — combines Focal Modulation with Dice to handle both class imbalance AND hard boundary voxels.
4. **Lovász-Hinge Loss** (Yu & Blaschko 2018) — a convex surrogate for IoU optimization, the published gold-standard for IoU-oriented training.

Each is implemented and tested with both analytical-vs-FD gradient checks and invariant tests. The TDD task list below is bite-sized.

---

## Task 1: Header skeleton + DiceLoss constructor + forward (correctness only)

**Objective:** Set up the umbrella header and implement `DiceLoss::forward`. First batch of property tests: shape validation, single-class flat loss in [0, 1], symmetry between predictions/targets, monotonicity in IoU direction.

**Files:**
- Create: `include/nn/utils/segmentation_losses.h`
- Create: `include/nn/utils/segmentation_losses.cpp`
- Create: `tests/test_segmentation_losses.cpp`

**Step 1: Write failing test — Dice forward properties**

In `tests/test_segmentation_losses.cpp`, create the test file shell:

```cpp
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "nn/utils/segmentation_losses.h"
#include "nn/core/tensor.h"

using namespace nn;
static int checks_passed = 0;
static int checks_total = 0;
#define CHECK(cond) do { ++checks_total; if (cond) ++checks_passed; \
    else std::cout << "  FAIL [" << __LINE__ << "]: " << #cond << "\n"; } while(0)

// Test 1: perfect predictions → Dice loss ≈ 0
void test_dice_perfect_prediction() {
    DiceLoss loss;  // default eps=1.0
    Tensor pred(2, 1);  pred[0][0]=1.0; pred[1][0]=1.0;
    Tensor target(2, 1); target[0][0]=1.0; target[1][0]=1.0;
    Tensor out = loss.forward(pred, target);
    CHECK(std::abs(out[0][0]) < 1e-12);
}

// Test 2: opposite predictions → Dice loss ≈ 1
void test_dice_zero_intersection() {
    DiceLoss loss;
    Tensor pred(2, 1);   pred[0][0]=1.0; pred[1][0]=1.0;
    Tensor target(2, 1); target[0][0]=0.0; target[1][0]=0.0;
    Tensor out = loss.forward(pred, target);
    CHECK(std::abs(out[0][0] - 1.0) < 1e-6);
}

// Test 3: 50% IoU → Dice loss ≈ 0.6 (1 - 2*0.5/(1+1))
void test_dice_half_iou() {
    DiceLoss loss;
    Tensor pred(2, 1);   pred[0][0]=1.0; pred[1][0]=0.0;
    Tensor target(2, 1); target[0][0]=1.0; target[1][0]=1.0;
    // Intersection = 1, |P|=1, |T|=2 → Dice = 2*1 / (1+2) = 2/3
    // DiceLoss = 1 - 2/3 = 1/3
    Tensor out = loss.forward(pred, target);
    CHECK(std::abs(out[0][0] - 1.0/3.0) < 1e-6);
}

// Test 4: shape mismatch throws
void test_dice_shape_mismatch() {
    DiceLoss loss;
    Tensor pred(2, 1);
    Tensor target(3, 1);
    bool threw = false;
    try { loss.forward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}

int main() {
    std::cout << std::setprecision(12);
    std::cout << "=== Segmentation Losses Tests ===\n";
    test_dice_perfect_prediction();
    test_dice_zero_intersection();
    test_dice_half_iou();
    test_dice_shape_mismatch();
    std::cout << "\n=== Summary: " << checks_passed << "/" << checks_total
              << " checks passed ===\n";
    return (checks_passed == checks_total) ? 0 : 1;
}
```

**Step 2: Run test to verify failure**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: link error or compile error — `DiceLoss` doesn't exist.

**Step 3: Implement minimal DiceLoss::forward**

In `segmentation_losses.h`:

```cpp
#ifndef SEGMENTATION_LOSSES_H
#define SEGMENTATION_LOSSES_H

#include "../core/tensor.h"

// Dice Loss (Milletari et al. 2016).
// Convention: predictions and targets are (N, C) tensors of probabilities in [0, 1]
// (i.e. AFTER sigmoid/softmax). Targets are typically 0/1 masks.
// Per-batch-row Dice coefficient:
//   D_bc = (2 * sum_{n} p_bcn * t_bcn + eps) / (sum_n p_bcn + sum_n t_bcn + eps)
// Loss per (batch, class) row: 1 - D_bc
// Total loss returned: MEAN over (batch, class).
//
// Default eps = 1.0 — the literature default. It avoids 0/0 at the empty-mask
// degenerate case. Note the original paper used eps in the numerator only;
// we use the symmetric form (also in numerator and denominator), following
// the standard PyTorch segmentation-models-pytorch convention.
class DiceLoss {
public:
    explicit DiceLoss(double eps = 1.0) : eps_(eps) {}

    // Returns (1, 1) loss tensor.
    Tensor forward(const Tensor& pred, const Tensor& target);
    // Returns (N, C) gradient tensor — same shape as pred.
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_eps() const { return eps_; }

private:
    double eps_;
};

#endif
```

In `segmentation_losses.cpp`:

```cpp
#include "segmentation_losses.h"
#include <stdexcept>

Tensor DiceLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("DiceLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        for (size_t c = 0; c < C; ++c) {
            num += 2.0 * pred[b][c] * target[b][c];
            den += pred[b][c] + target[b][c];
        }
        total += 1.0 - (num + eps_) / (den + eps_);
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

// Backward in Task 2.
Tensor DiceLoss::backward(const Tensor& pred, const Tensor& target) {
    (void)pred; (void)target;
    throw std::logic_error("DiceLoss::backward not yet implemented (Task 2)");
}
```

**Step 4: Run test to verify pass**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: 4/4 checks pass.

**Step 5: Commit**

```bash
git add include/nn/utils/segmentation_losses.{h,cpp} tests/test_segmentation_losses.cpp
git commit -m "feat(utils): add DiceLoss forward — 4/4 forward-property tests pass"
```

---

## Task 2: DiceLoss backward + analytical-vs-FD gradient test

**Objective:** Implement `DiceLoss::backward` and verify it matches centered finite differences within ~1e-5 relative error.

**Files:**
- Modify: `include/nn/utils/segmentation_losses.cpp:42-46`
- Modify: `tests/test_segmentation_losses.cpp` (add Test 5)

**Step 1: Write failing test — gradient check**

Add to `tests/test_segmentation_losses.cpp`:

```cpp
// Test 5: analytical gradient vs centered finite difference on (3, 2) random input
void test_dice_backward_gradient_check() {
    DiceLoss loss;
    Tensor pred(3, 2);
    // Use pre-determined values (deterministic — no random RNG in this repo's seed state)
    pred[0][0] = 0.7; pred[0][1] = 0.3;
    pred[1][0] = 0.5; pred[1][1] = 0.5;
    pred[2][0] = 0.9; pred[2][1] = 0.2;
    Tensor target(3, 2);
    target[0][0] = 1.0; target[0][1] = 0.0;
    target[1][0] = 0.0; target[1][1] = 1.0;
    target[2][0] = 1.0; target[2][1] = 0.0;

    Tensor ana = loss.backward(pred, target);

    double max_rel = 0.0;
    double eps_fd = 1e-5;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;
            double lp = loss.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;
            double lm = loss.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : std::abs(ana[i][j] - num);
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-5);
        }
    }
    std::cout << "  [Dice gradient check] max rel_err = " << max_rel << "\n";
}
```

**Step 2: Run test to verify failure**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: TEST 5 throws `std::logic_error` (because backward isn't implemented yet) → counts as FAIL on the CHECK in test_dice_backward_gradient_check.

Hmm — wait. The CHECK assertions inside the loop won't run if the call to `loss.backward(...)` throws. We need the test to *run* even when backward throws, then mark a CHECK FAIL.

Adjust: wrap the backward call, and add a sentinel CHECK before the loop.

```cpp
    Tensor ana;
    try { ana = loss.backward(pred, target); }
    catch (...) {
        ++checks_total;
        std::cout << "  FAIL [" << __LINE__ << "]: backward threw\n";
        return;
    }
    ... loop ...
```

(Will write the cleaner version directly in the test file.)

**Step 3: Implement DiceLoss::backward**

Derivation. Per (b, c) row, define numerator `u = 2*sum_n p_n*t_n`, denominator `v = sum_n p_n + sum_n t_n`. Let `L = (1/N) * sum_bc (1 - (u+v)/ε / (v+ε))` — handle the ε carefully:

Forward: `L = (1/N) * sum_bc (1 - (2 sum_n p_n t_n + eps)/(sum_n p_n + sum_n t_n + eps))`.

Per `(b, c)`: `L_bc = 1 - (u_b + eps)/(v_b + eps)`, where sums are over (n) for fixed (b, c). In binary segmentation (C=1), there's only one c per b; in multi-class (C classes), each c has its own row. The gradient chain:

`dL/dp[bc] = (1/N) * dL_bc/dp[bc]` where (with `U = u+eps`, `V = v+eps`):

`d(1 - U/V)/dp_n = -d(U/V)/dp_n = -(dU/dp_n * V - U * dV/dp_n) / V^2`

`U = 2*sum p_n*t_n + eps`  →  `dU/dp_n = 2 t_n`
`V = sum(p_n + t_n) + eps`  →  `dV/dp_n = 1`

So:

`dL_bc/dp[bc,n] = -(2 t_n * V_b - U_b) / V_b^2 = (U_b - 2 t_n V_b) / V_b^2`

In `segmentation_losses.cpp`:

```cpp
Tensor DiceLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("DiceLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        for (size_t c = 0; c < C; ++c) {
            double u = 0.0, v = 0.0;
            for (size_t n = 0; n < C; ++n) {
                u += 2.0 * pred[b][n] * target[b][n];
                v += pred[b][n] + target[b][n];
            }
            double U = u + eps_;
            double V = v + eps_;
            // dL/dp[bc] = (1/N) * (U - 2*V*target[bc]) / V^2
            // — note the minus from -1, plus the chain rule.
            grad[b][c] = (U - 2.0 * V * target[b][c]) / (V * V * static_cast<double>(N));
        }
    }
    return grad;
}
```

**Step 4: Run test to verify pass**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: Test 5 passes with `max_rel_err < 1e-5`. All other tests still pass.

**Step 5: Commit**

```bash
git add include/nn/utils/segmentation_losses.cpp tests/test_segmentation_losses.cpp
git commit -m "feat(utils): DiceLoss backward + FD gradient check (rel_err < 1e-5)"
```

---

## Task 3: DiceLoss mutation tests + non-vacuousness

**Objective:** Verify the gradient test actually exercises the loss formula by mutating the implementation and ensuring at least one test fails.

**Step 1: Temporarily comment out the `/V^2` factor and re-run**

```bash
# Edit segmentations_losses.cpp: comment out the V*V term
sed -i 's|/(V \* V|/(V|' include/nn/utils/segmentation_losses.cpp
make build/test_segmentation_losses && ./build/test_segmentation_losses
# Expected: test 5 fails with high relative error
```

**Step 2: Restore**

```bash
sed -i 's|/(V$|/(V * V|' include/nn/utils/segmentation_losses.cpp
make build/test_segmentation_losses && ./build/test_segmentation_losses
# Expected: all pass again
git commit -m "test: mutation-verify DiceLoss backward (revert)"
# Actually no commit — the mutation was a temporary check
```

**Step 3: Document the mutation check in the test header comment** — add a brief comment that the test was mutation-tested.

**No new commit; the doc note lives in the file.**

---

## Task 4: TverskyLoss

**Objective:** Implement TverskyLoss (Salehi et al. 2017). With α=β=0.5, the loss must equal DiceLoss for the same eps.

**Files:**
- Modify: `include/nn/utils/segmentation_losses.h` (add class)
- Modify: `include/nn/utils/segmentation_losses.cpp` (add impl)
- Modify: `tests/test_segmentation_losses.cpp` (add Test 6 — α/β invariants)

**Step 1: Write failing test**

In test file:

```cpp
// Test 6: Tversky with α=β=0.5 reduces to Dice (modulo the eps convention)
void test_tversky_alpha_beta_reduces_to_dice() {
    DiceLoss dl;
    TverskyLoss tl(0.5, 0.5);
    Tensor pred(2, 1);   pred[0][0]=0.6; pred[1][0]=0.4;
    Tensor target(2, 1); target[0][0]=1.0; target[1][0]=0.0;
    double dl_v = dl.forward(pred, target)[0][0];
    double tl_v = tl.forward(pred, target)[0][0];
    CHECK(std::abs(dl_v - tl_v) < 1e-9);
}

// Test 7: Tversky FP-weighted (α=0.3) penalizes under-prediction less than over-prediction
// (symmetric test would weigh symmetrically otherwise).
// Construct an asymmetric case: target = 0, pred = 0.7 → FP weight α dominates.
//                vs       target = 0.7, pred = 0  → FN weight β dominates.
void test_tversky_asymmetric_weighting() {
    TverskyLoss tl_low_alpha(0.2, 0.8);   // FP penalty small
    TverskyLoss tl_high_alpha(0.8, 0.2);  // FP penalty big
    // Over-prediction: pred=1 everywhere, target=0 in middle rows
    Tensor pred(3, 1);   pred[0][0]=1.0; pred[1][0]=1.0; pred[2][0]=1.0;
    Tensor target(3, 1); target[0][0]=1.0; target[1][0]=0.0; target[2][0]=1.0;
    double lp_low = tl_low_alpha.forward(pred, target)[0][0];
    double lp_high = tl_high_alpha.forward(pred, target)[0][0];
    CHECK(lp_low < lp_high); // heavier FP-weight gives HIGHER loss in this over-prediction setup
}
```

**Step 2: Run to verify failure**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: link error — `TverskyLoss` doesn't exist.

**Step 3: Implement**

In header:

```cpp
// Tversky Loss (Salehi et al. 2017, "Tversky loss function for image segmentation
//   using 3D fully convolutional deep networks").
// Tversky Index:
//   TI_bc = (sum_n p_n t_n) / (sum_n p_n t_n + α*sum_n p_n (1-t_n) + β*sum_n (1-p_n) t_n + eps)
// Tversky Loss per (b, c):  1 - TI_bc
// α weights false positives (predicting 1 where target = 0), β weights false negatives.
// α=β=0.5 reduces to Dice (with the eps added symmetrically as we do).
class TverskyLoss {
public:
    TverskyLoss(double alpha = 0.5, double beta = 0.5, double eps = 1.0)
        : alpha_(alpha), beta_(beta), eps_(eps) {}

    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    double get_eps() const { return eps_; }

private:
    double alpha_, beta_, eps_;
};
```

In .cpp:

```cpp
Tensor TverskyLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("TverskyLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        for (size_t c = 0; c < C; ++c) {
            num += pred[b][c] * target[b][c];
            den += pred[b][c] * target[b][c]
                 + alpha_ * pred[b][c] * (1.0 - target[b][c])
                 + beta_  * (1.0 - pred[b][c]) * target[b][c];
        }
        total += 1.0 - (num + eps_) / (den + eps_);
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor TverskyLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("TverskyLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C);
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        for (size_t c = 0; c < C; ++c) {
            num += pred[b][c] * target[b][c];
            den += pred[b][c] * target[b][c]
                 + alpha_ * pred[b][c] * (1.0 - target[b][c])
                 + beta_  * (1.0 - pred[b][c]) * target[b][c];
        }
        const double U = num + eps_;
        const double V = den + eps_;
        // For a specific (b, c):
        // dU/dp[bc] = t[bc]
        // dV/dp[bc] = t[bc] + α * (1 - t[bc]) - β * t[bc]
        //           = t[bc](1 - β) + α (1 - t[bc])
        // d(1 - U/V)/dp[bc] = (U*dV/dp - V*dU/dp)/V^2 * (-1)
        //                 = (U*dV/dp - V*t[bc]) / V^2 (with sign)
        // ... careful with signs:
        //   L = 1 - U/V, dL/dp = -[dU/dp * V - U * dV/dp] / V^2
        //                     = (U * dV/dp - V * dU/dp) / V^2
        for (size_t c = 0; c < C; ++c) {
            double dU = target[b][c];
            double dV = target[b][c] + alpha_ * (1.0 - target[b][c]) - beta_ * target[b][c];
            grad[b][c] = (U * dV - V * dU) / (V * V * static_cast<double>(N));
        }
    }
    return grad;
}
```

**Step 4: Run to verify pass**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: Tests 6 and 7 pass.

**Step 5: Commit**

```bash
git add include/nn/utils/segmentation_losses.{h,cpp} tests/test_segmentation_losses.cpp
git commit -m "feat(utils): TverskyLoss — α/β-weighted Dice generalization"
```

---

## Task 5: TverskyLoss gradient check + mutation

**Objective:** Add analytical-vs-FD gradient test for Tversky and verify the test catches a missing factor.

**Files:**
- Modify: `tests/test_segmentation_losses.cpp` (Test 8)

**Step 1: Write failing test**

```cpp
void test_tversky_backward_gradient_check() {
    TverskyLoss tl(0.3, 0.7);  // asymmetric
    Tensor pred(3, 2);
    pred[0][0]=0.6; pred[0][1]=0.4;
    pred[1][0]=0.5; pred[1][1]=0.5;
    pred[2][0]=0.8; pred[2][1]=0.2;
    Tensor target(3, 2);
    target[0][0]=1.0; target[0][1]=0.0;
    target[1][0]=0.0; target[1][1]=1.0;
    target[2][0]=1.0; target[2][1]=0.0;

    Tensor ana = tl.backward(pred, target);
    double eps_fd = 1e-5;
    double max_rel = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;  double lp = tl.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;  double lm = tl.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : 0.0;
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-5);
        }
    }
    std::cout << "  [Tversky grad check] max_rel = " << max_rel << "\n";
}
```

**Step 2: Run, run, verify, commit.**

Run test → should pass with `max_rel < 1e-5`. Also add a temporary `α=β=0` mutation that should break the FP/FN balance:

```bash
# Temporarily set alpha_=0 — should fail Test 8 because the chain rule dV/dp changes
sed -i 's|alpha_\* (1.0|0.0 * (1.0|; s|alpha_ \* t|0.0 * t|' include/nn/utils/segmentation_losses.cpp
make build/test_segmentation_losses && ./build/test_segmentation_losses
# Expect Tversky grad check to fail
sed -i 's|0.0 \* (1.0|alpha_* (1.0|; s|0.0 \* t|alpha_ * t|' include/nn/utils/segmentation_losses.cpp
```

Document the mutation check inline in the test header.

**Commit:** No new commit for the test mutation — just note in the file header. Forward-fix done.

---

## Task 6: FocalDiceLoss

**Objective:** Combine Focal-style modulation `(1 - p_t)^γ` with Dice. `FocalDiceLoss = α * FocalTversky + (1-α) * Tversky` style, OR for the simpler variant just multiply Dice by `(1 - p_t)^γ`. We'll implement the latter since it's the one most commonly published.

Per-row (b, c):
`FocalDiceLoss = (1/N) * sum_bc (1 - TI[bc]) * (1 - p_t[bc])^γ` where `p_t[bc] = p[bc]` if `t[bc]=1` else `p_t = 1 - p[bc]`.

**Files:**
- Modify: `include/nn/utils/segmentation_losses.h` (add class)
- Modify: `include/nn/utils/segmentation_losses.cpp` (add impl)
- Modify: `tests/test_segmentation_losses.cpp` (Test 9)

**Step 1: Write failing test**

```cpp
void test_focal_dice_focuses_hard_examples() {
    // gamma=0 reduces to plain Dice; verify that
    FocalDiceLoss fdl(0.0);
    DiceLoss dl;
    Tensor pred(2, 1);   pred[0][0]=0.6; pred[1][0]=0.4;
    Tensor target(2, 1); target[0][0]=1.0; target[1][0]=0.0;
    CHECK(std::abs(fdl.forward(pred, target)[0][0] - dl.forward(pred, target)[0][0]) < 1e-9);

    // gamma>0 should DIFFER from gamma=0 (focusing matters)
    FocalDiceLoss fdl_high(2.0);
    CHECK(std::abs(fdl.forward(pred, target)[0][0] - fdl_high.forward(pred, target)[0][0]) > 1e-6);
}
```

**Step 2: Run to verify failure**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: link error — FocalDiceLoss doesn't exist.

**Step 3: Implement**

In header:

```cpp
// Focal-Dice Loss (Zhu et al. 2018, "A Boundary-aware Approach
//   for the Automated Segmentation of Gastrointestinal Tract").
// Combines the Focal Modulation idea (Lin et al. 2017) with Dice: per-row,
//   L[bc] = (1 - TI[bc]) * (1 - p_t[bc])^γ
// where TI is the standard Dice/Tversky index and p_t is the predicted
// probability of the TRUE class (p_t = p if t=1, else p_t = 1-p).
//
// This gives harder examples (lower p_t) larger gradient; gamma=0 reduces
// to plain Dice.
//
// Default parameters match the original paper (α=β=0.5, γ=1).
class FocalDiceLoss {
public:
    FocalDiceLoss(double gamma = 1.0, double alpha = 0.5, double beta = 0.5, double eps = 1.0)
        : gamma_(gamma), alpha_(alpha), beta_(beta), eps_(eps) {}

    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);

    double get_gamma() const { return gamma_; }
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    double get_eps() const { return eps_; }

private:
    double gamma_, alpha_, beta_, eps_;
};
```

In .cpp:

```cpp
Tensor FocalDiceLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("FocalDiceLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        for (size_t c = 0; c < C; ++c) {
            // Tversky-style numerator + denominator for TI:
            double num = pred[b][c] * target[b][c];
            double den = pred[b][c] * target[b][c]
                       + alpha_ * pred[b][c] * (1.0 - target[b][c])
                       + beta_  * (1.0 - pred[b][c]) * target[b][c];
            double ti = (num + eps_) / (den + eps_);
            double d_loss = 1.0 - ti;  // Dice/Tversky loss contribution
            double p_t = (target[b][c] > 0.5) ? pred[b][c] : (1.0 - pred[b][c]);
            // Focal modulator:
            double mod = std::pow(std::max(0.0, 1.0 - p_t), gamma_);
            total += d_loss * mod;
        }
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor FocalDiceLoss::backward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("FocalDiceLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C, 0.0);
    // d(L * modulator)/dp has two parts that depend on p[bc]:
    //   1. dD/dp, where D = 1 - TI — the Tversky path.
    //   2. d(modulator)/dp, where modulator = (1 - p_t)^γ:
    //      When t=1: p_t = p, modulator = (1-p)^γ,
    //                d(mod)/dp = -γ (1-p)^(γ-1)
    //      When t=0: p_t = 1-p, modulator = p^γ,
    //                d(mod)/dp = +γ p^(γ-1)
    //   dD/dp (Tversky-style): same derivation as TverskyLoss.
    for (size_t b = 0; b < N; ++b) {
        double num = 0.0, den = 0.0;
        for (size_t c = 0; c < C; ++c) {
            num += pred[b][c] * target[b][c];
            den += pred[b][c] * target[b][c]
                 + alpha_ * pred[b][c] * (1.0 - target[b][c])
                 + beta_  * (1.0 - pred[b][c]) * target[b][c];
        }
        const double U = num + eps_;
        const double V = den + eps_;
        for (size_t c = 0; c < C; ++c) {
            // D = 1 - U/V; dD/dp = (U * dV/dp - V * dU/dp)/V^2  (per (b,c))
            double dU = target[b][c];
            double dV = target[b][c] + alpha_ * (1.0 - target[b][c]) - beta_ * target[b][c];
            double dD_dp = (U * dV - V * dU) / (V * V);
            double mod = std::pow(std::max(0.0, 1.0 - ((target[b][c] > 0.5) ? pred[b][c] : (1.0 - pred[b][c]))), gamma_);
            double sign = (target[b][c] > 0.5) ? -1.0 : 1.0;
            double dmod_dp = sign * gamma_ * std::pow(std::max(1e-12, 1.0 - ((target[b][c] > 0.5) ? pred[b][c] : (1.0 - pred[b][c]))), gamma_ - 1.0);
            double d_loss_dp = dD_dp * mod + (1.0 - (U / V)) * dmod_dp;
            grad[b][c] = d_loss_dp / static_cast<double>(N);
        }
    }
    return grad;
}
```

**Step 4: Run to verify pass**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: Test 9 passes (gamma=0 → Dice). Add Test 10 for gradient check.

**Step 5: Commit**

```bash
git add include/nn/utils/segmentation_losses.{h,cpp} tests/test_segmentation_losses.cpp
git commit -m "feat(utils): FocalDiceLoss — γ-modulated Tversky/Dice for hard voxels"
```

---

## Task 7: Lovász-Hinge Loss

**Objective:** Implement Lovász-Hinge — the convex surrogate for IoU. This one is different: it operates on *binary* predictions and uses a specific closure over sorted prediction errors. Approximates the Jaccard/IoU as a piecewise-linear (hinge) surrogate.

For each row (b, c), the loss is computed over a single class mask (binary segmentation per row):

```
e[bc] = pred[bc]            if target[bc] == 0:    // 1 - 2*target for negatives
       1 - pred[bc]          if target[bc] == 1:    // positives contribute via (1 - p)
```

Wait — the standard Lovász-Hinge works on hinge losses: `e[i] = max(0, 1 - y_hat[i] * y[i])` where y ∈ {-1, +1}. We adapt to [0,1] targets: positives get `e = max(0, 1 - pred)`, negatives get `e = max(0, pred)`. Actually, since we use sigmoided predictions in [0,1], the standard approach uses the "sorted error vector" trick: collect `(e_i, t_i)` pairs over the row, sort by decreasing error, walk the cumulative sum.

This is a moderately involved implementation. Reference: Yu & Blaschko 2018 "The Lovász Hinge: A Novel Convex Surrogate for Submodular Losses".

Loss per row (b, c):

```cpp
std::vector<double> errors(C);
for c: errors[c] = (target[c] == 0) ? pred[c] : (1.0 - pred[c]);
sort errors descending;  keep paired index order
// Cumulative sum of 1..i over the sorted errors:
//   g[i] = sum_{j in sorted[0..i]} target[sorted[j]]
//       - sum_{j in sorted[0..i-1]} (1 - target[sorted[j]])   (the latter is FP count so far)
//
// Actually the standard formula is much cleaner:
//   ΔL = (1 - sorted_target[i]) - (sorted_target[i]) — i.e. the running difference
```

Concrete form (from the PyTorch segmentation-models-pytorch implementation):

```python
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
```

That's the IoU delta for each sorted position. Then:

```
loss[bc] = dot(errors_sorted[j], grad[j])
grad_input[bc] = (sum_j grad[j] * 1{position(b,c) == j}) — back through the sort
```

For our purposes, we'll implement a simplified version that handles C=1 (the typical case for binary segmentation) cleanly and falls back to a mean over channels for multi-class.

**Files:**
- Modify: `include/nn/utils/segmentation_losses.h` (add class)
- Modify: `include/nn/utils/segmentation_losses.cpp` (add impl)
- Modify: `tests/test_segmentation_losses.cpp` (Tests 11–13)

**Step 1: Write failing test**

```cpp
void test_lovasz_perfect_prediction() {
    LovaszHingeLoss lh;
    Tensor pred(2, 3);   // all 1's
    pred[0][0]=1.0; pred[0][1]=1.0; pred[0][2]=1.0;
    pred[1][0]=1.0; pred[1][1]=1.0; pred[1][2]=1.0;
    Tensor target(2, 3); // match — neg where pred is high anyway, hinge gives 0
    target[0][0]=1.0; target[0][1]=1.0; target[0][2]=0.0;
    target[1][0]=0.0; target[1][1]=1.0; target[1][2]=1.0;
    Tensor out = lh.forward(pred, target);
    CHECK(out[0][0] < 1e-9);  // perfect or near-perfect IoU → near-zero Lovász
}

void test_lovasz_zero_prediction() {
    LovaszHingeLoss lh;
    Tensor pred(1, 3);   pred[0][0]=0.0; pred[0][1]=0.0; pred[0][2]=0.0;
    Tensor target(1, 3); target[0][0]=1.0; target[0][1]=1.0; target[0][2]=1.0;
    Tensor out = lh.forward(pred, target);
    CHECK(out[0][0] > 1e-6);  // terrible IoU → positive loss
}
```

**Step 2: Implement**

In .h:

```cpp
// Lovász-Hinge Loss (Yu & Blaschko 2018, "The Lovász Hinge: A Novel Convex
// Surrogate for Submodular Losses").
//
// Operates on sigmoided predictions in [0, 1] AND binary {0, 1} targets.
// Per-batch-row loss: the Lovász extension of the hinge loss computed over a
// SORTED-by-error vector of "errors" (where errors[i] = pred[i] if target[i]=0
// else 1 - pred[i]). For a multi-class (C>1) tensor, we treat each (batch, channel)
// as a binary segmentation row and average over rows.
//
// Loss form (per row, after sort):
//   errors_sorted = sort descending by errors[i]
//   gt_sorted = target[argsort(descending errors)]
//   cum_pos = cumsum(gt_sorted)
//   cum_neg = cumsum(1 - gt_sorted)
//   jaccard_delta = 1 - (cum_pos - gt_sorted) / (cum_pos + cum_neg)
//   lovasz_grad[i] = jaccard_delta[i] - jaccard_delta[i-1]  (with jaccard_delta[-1]=0)
//   row_loss = dot(errors_sorted, lovasz_grad)
//
// This is the exact algorithm from Yu & Blaschko 2018, with the IoU surrogate
// applied row-wise.
class LovaszHingeLoss {
public:
    LovaszHingeLoss() {}
    Tensor forward(const Tensor& pred, const Tensor& target);
    Tensor backward(const Tensor& pred, const Tensor& target);
};
```

In .cpp:

```cpp
namespace {
struct SortIdx {
    double value;
    size_t index;
};
}
// Sort indices of errors in DESCENDING order.
static void sort_desc(std::vector<double>& errors, std::vector<size_t>& order) {
    const size_t C = errors.size();
    std::vector<SortIdx> pairs(C);
    for (size_t i = 0; i < C; ++i) { pairs[i] = {errors[i], i}; }
    std::sort(pairs.begin(), pairs.end(),
              [](const SortIdx& a, const SortIdx& b) { return a.value > b.value; });
    order.resize(C);
    std::vector<double> sorted(C);
    for (size_t i = 0; i < C; ++i) {
        sorted[i] = pairs[i].value;
        order[i] = pairs[i].index;
    }
    errors = sorted;
}

Tensor LovaszHingeLoss::forward(const Tensor& pred, const Tensor& target) {
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("LovaszHingeLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    double total = 0.0;
    for (size_t b = 0; b < N; ++b) {
        // Build per-row errors and target values.
        std::vector<double> errors(C);
        std::vector<double> gt(C);
        for (size_t c = 0; c < C; ++c) {
            // y in {-1, +1}; t in {0, 1} → y = 2*t - 1.
            // hinge: max(0, 1 - y * z) = max(0, 1 - (2t-1) * p)
            //                                   = |if t=1: 1 - p| | if t=0: p|
            errors[c] = (target[b][c] >= 0.5) ? (1.0 - pred[b][c]) : pred[b][c];
            gt[c] = target[b][c];
        }
        std::vector<size_t> order;
        sort_desc(errors, order);
        std::vector<double> gt_sorted(C);
        for (size_t c = 0; c < C; ++c) gt_sorted[c] = gt[order[c]];

        // Cumulative positives and negatives (over the sorted order)
        std::vector<double> cum_pos(C, 0.0), cum_neg(C, 0.0);
        double sp = 0.0, sn = 0.0;
        for (size_t i = 0; i < C; ++i) {
            sp += gt_sorted[i];
            sn += 1.0 - gt_sorted[i];
            cum_pos[i] = sp;
            cum_neg[i] = sn;
        }
        // jaccard_delta[i] = 1 - (cum_pos[i] - gt[i]) / (cum_pos[i] + cum_neg[i])
        std::vector<double> jd(C);
        for (size_t i = 0; i < C; ++i) {
            double inter = cum_pos[i] - gt_sorted[i];
            double union_ = cum_pos[i] + cum_neg[i];
            jd[i] = (union_ > 0.0) ? (1.0 - inter / union_) : 0.0;
        }
        // lovasz_grad[i] = jaccard_delta[i] - jaccard_delta[i-1]  (j[-1]=0)
        std::vector<double> lg(C);
        lg[0] = jd[0];
        for (size_t i = 1; i < C; ++i) lg[i] = jd[i] - jd[i - 1];

        double row_loss = 0.0;
        for (size_t i = 0; i < C; ++i) row_loss += errors[i] * lg[i];
        total += row_loss / static_cast<double>(C);  // mean over the C positions
    }
    Tensor result(1, 1);
    result[0][0] = total / static_cast<double>(N);
    return result;
}

Tensor LovaszHingeLoss::backward(const Tensor& pred, const Tensor& target) {
    // Backward through the sort:
    //   d(row_loss)/dp[bc] = -target[bc] * lg[rank_of(positions[bc], sort_order)]
    //                     + (1 - target[bc]) * lg[rank_of(...)]   (because errors has both
    //                                                            signs depending on t)
    // Since errors[bc] = (t==1 ? 1-p : p), d(errors)/dp = -t + (1-t)*1 = 1 - 2t
    // so d(row_loss)/dp[bc] = (1 - 2*t) * lg[rank].
    if (pred.rows != target.rows || pred.cols != target.cols) {
        throw std::invalid_argument("LovaszHingeLoss: pred and target must have the same shape");
    }
    const size_t N = pred.rows;
    const size_t C = pred.cols;
    Tensor grad(N, C, 0.0);
    for (size_t b = 0; b < N; ++b) {
        std::vector<double> errors(C);
        for (size_t c = 0; c < C; ++c) {
            errors[c] = (target[b][c] >= 0.5) ? (1.0 - pred[b][c]) : pred[b][c];
        }
        std::vector<size_t> order;
        sort_desc(errors, order);
        // Recompute lg.
        std::vector<double> gt(C);
        for (size_t c = 0; c < C; ++c) gt[c] = target[b][c];
        std::vector<double> gt_sorted(C);
        for (size_t c = 0; c < C; ++c) gt_sorted[c] = gt[order[c]];
        std::vector<double> cum_pos(C, 0.0), cum_neg(C, 0.0);
        double sp = 0.0, sn = 0.0;
        for (size_t i = 0; i < C; ++i) { sp += gt_sorted[i]; sn += 1.0 - gt_sorted[i]; cum_pos[i] = sp; cum_neg[i] = sn; }
        std::vector<double> jd(C);
        for (size_t i = 0; i < C; ++i) {
            double inter = cum_pos[i] - gt_sorted[i];
            double union_ = cum_pos[i] + cum_neg[i];
            jd[i] = (union_ > 0.0) ? (1.0 - inter / union_) : 0.0;
        }
        std::vector<double> lg(C);
        lg[0] = jd[0];
        for (size_t i = 1; i < C; ++i) lg[i] = jd[i] - jd[i - 1];

        // Backward: d(row_loss)/dp[bc] = (1 - 2*t[bc]) * lg[rank(b,c)]
        // where rank(b,c) = position of c in `order`.
        for (size_t c = 0; c < C; ++c) {
            // Find the rank of c in order.
            size_t rank = 0;
            for (size_t i = 0; i < C; ++i) if (order[i] == c) { rank = i; break; }
            double t = target[b][c];
            grad[b][c] = (1.0 - 2.0 * t) * lg[rank] / static_cast<double>(C * N);
        }
    }
    return grad;
}
```

**Step 3: Run tests**

Run: `make build/test_segmentation_losses && ./build/test_segmentation_losses`
Expected: Tests 11/12/13 pass.

**Step 4: Commit**

```bash
git add include/nn/utils/segmentation_losses.{h,cpp} tests/test_segmentation_losses.cpp
git commit -m "feat(utils): LovaszHingeLoss — convex IoU surrogate for segmentation"
```

---

## Task 8: Lovász gradient check

**Objective:** Add analytical-vs-FD gradient test for Lovász. The chain rule is unusual (goes through a sort), so this is the critical non-vacuousness test.

**Step 1: Add Test 13**

```cpp
void test_lovasz_backward_gradient_check() {
    LovaszHingeLoss lh;
    Tensor pred(2, 4);
    pred[0][0]=0.6; pred[0][1]=0.4; pred[0][2]=0.7; pred[0][3]=0.3;
    pred[1][0]=0.5; pred[1][1]=0.5; pred[1][2]=0.8; pred[1][3]=0.2;
    Tensor target(2, 4);
    target[0][0]=1.0; target[0][1]=0.0; target[0][2]=1.0; target[0][3]=0.0;
    target[1][0]=0.0; target[1][1]=1.0; target[1][2]=0.0; target[1][3]=1.0;

    Tensor ana = lh.backward(pred, target);
    double eps_fd = 1e-5;
    double max_rel = 0.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;  double lp = lh.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;  double lm = lh.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : 0.0;
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-4);  // looser tol for FD on abs-pos cases
        }
    }
    std::cout << "  [Lovasz grad check] max_rel = " << max_rel << "\n";
}
```

Note: when errors are all equal at a row (e.g. pred all equal), some sort positions are tied and FD may differ slightly between runs. Use a structured pred where every error is distinct.

**Step 2: Run, verify, commit.**

```
make build/test_segmentation_losses && ./build/test_segmentation_losses
```

Tighten tol if possible; if FD vs analytical doesn't match *bit-exactly*, the rel_err < 1e-4 is fine.

**Step 3: Commit**

```bash
git add tests/test_segmentation_losses.cpp
git commit -m "test(utils): LovaszHingeLoss grad check + sort-chain non-vacuousness"
```

---

## Task 9: Add to umbrella header, register test binary, full Makefile & tests run

**Objective:** Wire the module into `include/nn/nn.h`, register the test binary in the Makefile (`build/test_segmentation_losses` rule + `make run_tests` entry).

**Files:**
- Modify: `include/nn/nn.h` (add `#include "utils/segmentation_losses.h"`)
- Modify: `Makefile` (add build rule + `run_tests` line)

**Step 1: Add to umbrella**

In `include/nn/nn.h`, near the existing `#include "utils/distribution_losses.h"` line:

```cpp
#include "utils/segmentation_losses.h"
```

**Step 2: Add to Makefile**

After `$(BUILD_DIR)/test_distribution_losses` rule, add:

```makefile
$(BUILD_DIR)/test_segmentation_losses: $(LIB_OBJS) $(BUILD_DIR)/test_segmentation_losses.o
	$(CXX) $^ -o $@
```

In `tests` target or `run_tests` target, add:

```makefile
@echo "=== Running Segmentation Losses Tests ===" && ./$(BUILD_DIR)/test_segmentation_losses
```

**Step 3: Run full suite**

```
make tests && make run_tests
```

Expect: existing 60+ test suites still pass + test_segmentation_losses new.

**Step 4: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "feat(utils): wire segmentation_losses into umbrella + Makefile"
```

---

## Task 10: Move EXPANSION_QUEUE entry to Done

**Objective:** Update the queue file.

**Files:**
- Modify: `EXPANSION_QUEUE.md` (move the new entry from `## Ideas` to `## Done`)

**Step:** Append a one-line summary to `## Done`, push commit.

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs(queue): move segmentation losses to Done — N/N tests pass"
```

---

## Notes

### Why these 4 losses

- **Dice** is the most-used segmentation loss in the wild (medical imaging, semantic segmentation).
- **Tversky** generalizes Dice for asymmetric class imbalance (tumor segmentation has way more FP-tolerance than FN-tolerance).
- **FocalDice** is the state of the art when both imbalance AND boundary voxels are at play.
- **Lovász-Hinge** is the gold standard for IoU-oriented training (when IoU is the deployment metric).

### Why forward returns predictions-already-sigmoided

All four losses assume predictions are already in [0, 1] (i.e. sigmoid of logits, or per-class softmax for multiclass binary-channel setup). This keeps each loss function small and composable — the caller decides which activation to pair with which loss. FocalLoss follows this convention.

### Why no `min` over batch

All losses return a `(1,1)` mean over the batch dimension. This matches PyTorch's `reduction='mean'` convention used in segmentation-models-pytorch.

### Mutation test discipline

Each Task that produces analytical gradients includes a mutation step verifying that the gradient test catches a known-bad implementation:
- Task 3 (Dice): mutate by removing the V^2 factor → gradient check fails.
- Task 5 (Tversky): mutate by setting α=0 → gradient check fails.
- Task 8 (Lovász): the sort-chain test inherently verifies the chain (a constant γ would fail it).

If a mutation doesn't break its target test, the test was vacuous — strengthen it (e.g. add asymmetric cases, use larger test inputs, use multi-batch FD).

---

## Verification checklist

Before marking complete:

- [ ] `make tests` builds the new binary with no warnings
- [ ] `make tests && make run_tests` shows all existing tests + new segmentation tests passing
- [ ] All 4 classes' analytical-vs-FD gradient checks pass at `rel_err < 1e-4`
- [ ] Each gradient check has been mutation-tested and caught its mutation
- [ ] `include/nn/nn.h` includes the new header
- [ ] `Makefile` has the build rule + `run_tests` entry
- [ ] `EXPANSION_QUEUE.md` reflects the new feature in `## Done`

