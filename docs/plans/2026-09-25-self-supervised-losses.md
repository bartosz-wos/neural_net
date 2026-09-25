# Self-Supervised Learning Losses — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add five canonical self-supervised learning (SSL) losses to the repo in a single file, matching the existing `distribution_losses.{h,cpp}` pattern. These losses power SimCLR/CLIP (InfoNCE is already shipped), DINO, BYOL, VICReg, BarlowTwins, and W-MSE — none of which are currently in the repo.

**Architecture:** One new pair of files (`include/nn/utils/self_supervised_losses.{h,cpp}` + `tests/test_self_supervised_losses.cpp`) with five loss classes, each with `forward()` + `backward()` returning a 1×1 loss and (N, D) input gradient respectively. All losses take L2-normalised projection vectors z ∈ R^{N×D} as the standard SSL input (after the projector head, before the loss).

**Tech Stack:** C++17, plain Tensor (the existing `Tensor` class), `nn/nn.h` umbrella. No new dependencies.

---

## Background — what these losses are

Self-supervised contrastive / non-contrastive representation learning on images, audio, video etc. takes two augmented "views" of each sample in a batch (so a batch of N becomes 2N views). The losses below learn representations such that:
- **DINO** — student matches teacher's sharpened softmax of similarities (Caron et al. 2021).
- **BYOL** — student predicts teacher's projection via MSE; teacher is EMA of student (Grill et al. 2020).
- **VICReg** — variance + invariance + covariance terms (Bardes & Weston 2022).
- **BarlowTwins** — cross-correlation matrix should be identity (Zbontar et al. 2021).
- **W-MSE** — whitening-based MSE (Ermolov et al. 2021).

The first three (DINO, BYOL, VICReg) are the most-cited in the family. W-MSE and BarlowTwins are well-established, smaller, and useful for cross-validation. We'll implement all five in one file.

## Conventions

- **Input**: `z` shape (2N, D) where rows 2i and 2i+1 are the two views of sample i. Loss is computed over all 2N anchors (the InfoNCE convention).
- **Output**: 1×1 Tensor of scalar loss (mean over anchors).
- **Backward**: shape (2N, D) gradient w.r.t. z.
- All losses assume L2-normalised z for the cosine-similarity losses (DINO, BYOL, W-MSE). For VICReg and BarlowTwins, normalisation is applied internally as required by the paper.

---

### Task 1: Header file — class skeletons + paper refs

**Files:**
- Create: `include/nn/utils/self_supervised_losses.h`

**Step 1:** Write the header with all 5 classes declared, with full paper citations in comments (Caron 2021 for DINO, Grill 2020 for BYOL, Bardes 2022 for VICReg, Zbontar 2021 for BarlowTwins, Ermolov 2021 for W-MSE). For each class: constructor with sensible defaults, `forward(const Tensor& z) → Tensor(1,1)`, `backward(const Tensor& z) → Tensor(2N, D)`, and accessors for hyper-parameters.

**Step 2:** Add `#include "self_supervised_losses.h"` to `include/nn/nn.h` (right after `distribution_losses.h`).

**Step 3:** `make -j` to verify header inclusion compiles. Expected: build succeeds (the .cpp may not exist yet, that's OK — header-only is enough for the umbrella check; or we make the header compile to a stub .cpp).

### Task 2: BYOL (loss + gradient + TDD)

**Why first:** simplest non-trivial loss; MSE in projection space; deterministic gradient.

**Files:**
- Modify: `include/nn/utils/self_supervised_losses.cpp` — implement BYOL
- Test: `tests/test_self_supervised_losses.cpp`

BYOL forward:
- For each anchor a ∈ {0, ..., 2N-1}, the positive is b = a XOR 1.
- target_a = z_b (the target network's projection — passed in by the caller as `z_target` of shape (2N, D), since BYOL uses EMA-target)
- predicted_a = z_a
- loss_a = MSE(predicted_a, target_a) = (1/D) * Σ_d (z_a_d - target_a_d)²
- loss = (1/(2N)) * Σ_a loss_a

Backward w.r.t. z_a (the student):
- dL/dz_a = (2/(2N * D)) * (z_a - target_a)
- dL/dz_b (target): zero (the EMA network doesn't get gradients — caller does the EMA update)

API: `BYOL(double projection_dim = 0)` — projection_dim defaults to 0 = same as input dim D (the loss doesn't care, just for shape docs).

### Task 3: VICReg (loss + gradient + TDD)

**Files:**
- Modify: `include/nn/utils/self_supervised_losses.cpp`
- Test: `tests/test_self_supervised_losses.cpp`

VICReg forward (Bardes & Weston 2022):
- Invariance: (1/2N) * Σ_a ||z_a - z_{a⊕1}||²  (MSE between paired views)
- Variance: (1/D) * Σ_d max(0, γ − std_d(z))  where std_d is the per-dim std across the 2N batch
- Covariance: (1/D) * Σ_{i≠j} C_{ij}²  where C is the (D × D) cross-correlation matrix of z

Three losses are weighted by `λ`, `μ`, `ν` (default 1, 1, 1).

Backward: hand-derive for invariance (trivial) and for variance / covariance (chain rule through mean and std, plus the off-diagonal terms of the correlation matrix).

### Task 4: BarlowTwins (loss + gradient + TDD)

**Files:**
- Modify: `include/nn/utils/self_supervised_losses.cpp`
- Test: `tests/test_self_supervised_losses.cpp`

BarlowTwins forward (Zbontar et al. 2021):
- Center z by subtracting per-column mean (across the 2N batch).
- Cross-correlation matrix: C_{ij} = Σ_a z_a_i · z_a_j / 2N  (the centred-z outer product divided by batch)
- Loss = Σ_i (1 − C_{ii})² + λ * Σ_{i≠j} C_{ij}²

Backward: chain rule through the mean-subtraction + outer-product. The on-diagonal and off-diagonal gradients are different (C_{ii} term goes through `1 - C_{ii}` squared).

### Task 5: DINO (loss + gradient + TDD)

**Files:**
- Modify: `include/nn/utils/self_supervised_losses.cpp`
- Test: `tests/test_self_supervised_losses.cpp`

DINO forward (Caron et al. 2021):
- Student and teacher projections of shape (2N, K) where K is the prototype count (e.g. 4096 or 65536).
- Student softmax with temperature t_s (default 0.1).
- Teacher softmax centered + sharpened with temperature t_t (default 0.07), then optionally centering across batch.
- Cross-entropy: L_a = -Σ_k t_a_k * log(s_a_k)  where s is student softmax, t is teacher softmax
- Final: -(1/2N) * Σ_a L_a

Backward: hand-derived CE gradient `(s - t) / (2N)`.

### Task 6: W-MSE (loss + gradient + TDD)

**Files:**
- Modify: `include/nn/utils/self_supervised_losses.cpp`
- Test: `tests/test_self_supervised_losses.cpp`

W-MSE forward (Ermolov et al. 2021):
- Whitening transform W applied to L2-normalised z (Cholesky-based whitening of the centred z).
- Loss = (1/(2N·D)) * Σ_a ||W·z_a − W·z_{a⊕1}||²

Backward: chain rule through whitening (matrix-inverse-based gradient).

### Task 7: Integration tests — end-to-end training reduces loss

For each loss, build a tiny 2-layer MLP encoder, generate two random views, compute loss + backward, do one SGD step on a simple regression target, and verify the loss decreases over 30 steps.

### Task 8: Register in Makefile + nn.h + cleanup

**Files:**
- Modify: `Makefile` — add `$(BUILD_DIR)/test_self_supervised_losses: $(LIB_OBJS) ...` rule, add to `tests:` target deps, add `=== Running SSL Losses Tests ===` to `run_tests`.

---

## Bite-sized checklist

- [ ] Task 1: Header skeletons + paper citations
- [ ] Task 2: BYOL
- [ ] Task 3: VICReg
- [ ] Task 4: BarlowTwins
- [ ] Task 5: DINO
- [ ] Task 6: W-MSE
- [ ] Task 7: End-to-end training smoke tests
- [ ] Task 8: Register + umbrella + Makefile

## Verification

- All 5 losses' forward outputs match hand-computed values within 1e-9.
- All 5 losses' backward gradients match finite-difference reference within 1e-4 (or rel_err < 1e-3 for gradient magnitudes near zero).
- Mutation test: stub a critical term in each loss and confirm at least one test catches it.
- `make tests` builds and runs `test_self_supervised_losses` cleanly.
- `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` passes (umbrella compiles).
- Standalone test binary `./build/test_self_supervised_losses` reports a single integer PASS/FAIL count.
