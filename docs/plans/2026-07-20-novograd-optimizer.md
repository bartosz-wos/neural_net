# NovoGrad Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a professional NovoGrad optimizer with tensor-wise adaptive second moments, normalized-gradient momentum, paper-compatible weight decay, optional gradient averaging/AMSGrad, scheduler compatibility, and focused regression coverage.

**Architecture:** `NovoGrad` will implement `Optimizer` in `include/nn/optimizers/novograd.{h,cpp}`. Each parameter tensor owns one full-size momentum tensor plus scalar `v`/`v_max` norm statistics: initialize `v_1 = ||g_1||²`, then update `v_t = β2·v_{t-1} + (1-β2)·||g_t||²`; form `u_t = g_t/(sqrt(v_t or v_max_t)+eps) + wd·w_t`; update `m_t = β1·m_{t-1} + (grad_averaging ? 1-β1 : 1)·u_t`; finally apply `w_{t+1} = w_t - lr·m_t`. This preserves NovoGrad's one-tensor-plus-scalars memory advantage over Adam's two full tensors.

**Tech Stack:** C++17, repository `Tensor`/`Layer`/`Model`/`Optimizer` APIs, Make, deterministic hand-derived test fixtures.

---

## Reference behavior

Primary source: Ginsburg et al., "Training Deep Networks with Stochastic Gradient Normalized by Layerwise Adaptive Second Moments" (2020), https://arxiv.org/abs/1905.11286.

Reference implementation: NVIDIA OpenSeq2Seq `open_seq2seq/optimizers/novograd.py`, https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/optimizers/novograd.py.

```text
s_t = ||g_t||²
v_1 = s_1
v_t = β2·v_{t-1} + (1-β2)·s_t                 (t > 1)
v_used = amsgrad ? max(v_max, v_t) : v_t
u_t = g_t/(sqrt(v_used)+eps) + weight_decay·w_t
m_t = β1·m_{t-1} + (grad_averaging ? 1-β1 : 1)·u_t
w_{t+1} = w_t - lr·m_t
```

Defaults follow NVIDIA OpenSeq2Seq's stable moment choices (`β1=0.95`, `β2=0.98`, `eps=1e-8`, no gradient averaging), while using this library's conservative optimizer learning-rate default `1e-3`.

### Task 1: Record the selected queue item and plan

**Objective:** Make the autonomously selected feature explicit before production code.

**Files:**
- Modify: `EXPANSION_QUEUE.md:6-9`
- Create: `docs/plans/2026-07-20-novograd-optimizer.md`

**Steps:**
1. Add NovoGrad as the sole and last item under `## Ideas`.
2. Save this plan.
3. Read both regions back and inspect `git diff --check`.
4. Commit only these files as `docs: add NovoGrad optimizer implementation plan`.

### Task 2: Establish RED with public behavior tests

**Objective:** Define the complete optimizer contract before implementation.

**Files:**
- Create: `tests/test_novograd.cpp`
- Modify: `Makefile` (focused `build/test_novograd` link target only)

**Test behaviors:**
1. Default/custom accessors, inherited `Optimizer::lr`, and `handles_weight_decay()`.
2. Constructor and setter validation: `lr >= 0`, `β1/β2 in [0,1)`, `eps > 0`, `weight_decay >= 0`.
3. Lazy state with momentum shape matching Dense weights/bias; scalar norm state absent before the first step.
4. First-step soul test: `g=[3,4]`, `||g||²=25`, `m=g/(5+eps)`, exact parameter update.
5. Tensor-wise normalization: asymmetric `g=[3,4,0,0]` must share one denominator, not element-wise Adam denominators.
6. Second-step scalar EMA recurrence with non-collinear gradients and exact stored `v`.
7. Momentum recurrence without `(1-β1)` when `grad_averaging=false`.
8. `grad_averaging=true` multiplies the whole normalized-plus-decay direction by `(1-β1)`.
9. Weight decay enters the momentum direction after normalization and changes zero-gradient parameters.
10. AMSGrad keeps `v_max` monotone when the current scalar norm EMA falls.
11. Zero-gradient/zero-norm updates remain finite.
12. Independent state across parameters and layers.
13. Scheduler writes inherited LR consumed by `step()`.
14. Deterministic trajectory, gradient clearing, and end-to-end linear-regression loss reduction.
15. Parameter/gradient count and shape mismatch guards using a tiny test layer.

**RED command:**

```bash
make build/test_novograd
```

Expected: compile failure because `nn/optimizers/novograd.h` does not exist.

### Task 3: Implement minimal NovoGrad

**Objective:** Make the focused suite pass with one paper-faithful implementation.

**Files:**
- Create: `include/nn/optimizers/novograd.h`
- Create: `include/nn/optimizers/novograd.cpp`

**Public API:** constructor, validated setters, hyperparameter accessors, `num_steps()`, `has_state()`, `get_momentum()`, `get_second_moment()`, and `get_max_second_moment()`.

**Implementation constraints:**
- Do not redeclare `lr`; read/write `Optimizer::lr` for scheduler compatibility.
- Initialize the scalar second moment from the first gradient norm exactly, rather than zero-biased EMA initialization.
- Use one scalar norm statistic per parameter tensor, not one scalar per `Layer` object and not an element-wise Tensor.
- Add epsilon outside the square root, matching the paper/OpenSeq2Seq formula.
- Add weight decay after gradient normalization and before momentum.
- Validate parameter/gradient count and shapes before state/update access.
- Clear each processed layer's gradients after `step()`.

**GREEN command:**

```bash
make -B build/test_novograd && ./build/test_novograd
```

Expected: all focused assertions pass without compiler warnings from new files.

### Task 4: Mutation-test the algorithmic soul

**Objective:** Prove the suite rejects plausible but incorrect NovoGrad variants.

**Temporary mutations (restore after each):**
1. Replace tensor-wise `||g||²` with the first coordinate's `g²`; exact soul/EMA/trajectory tests must fail.
2. Replace first-step `v=||g||²` with zero-biased `(1-β2)||g||²`; first-step magnitude tests must fail.
3. Drop weight decay from `u`; decay closed-form tests must fail.
4. Force `v_used=v` under AMSGrad; max-state/update tests must fail.

Run `make -B build/test_novograd && ./build/test_novograd` for every mutation, then restore and rerun to GREEN.

### Task 5: Public and aggregate integration

**Objective:** Expose NovoGrad and register its test exactly once.

**Files:**
- Modify: `include/nn/nn.h:10-26`
- Modify: `Makefile` test target, `tests:` dependency list, and `run_tests` recipe

**Steps:**
1. Include `optimizers/novograd.h` beside modern optimizers.
2. Add `build/test_novograd` to aggregate build dependencies.
3. Add one `NovoGrad Tests` execution to `run_tests`.
4. Inspect `git diff Makefile` immediately for target-list drift.

### Task 6: Queue completion and verification

**Objective:** Record results and prove both focused and repository-wide health.

**Files:**
- Modify: `EXPANSION_QUEUE.md`
- Inspect: root/tests cleanup candidates (`debug_*`, `*.bak`, `ref_*.cpp`)

**Steps:**
1. Move NovoGrad from `## Ideas` to `## Done` with exact test/mutation results.
2. Run focused test: `make -B build/test_novograd && ./build/test_novograd`.
3. Run aggregate compile in foreground: `make tests`.
4. Run aggregate suite in foreground: `make run_tests`; investigate failures against recorded baseline before attribution.
5. Run `git diff --check`, inspect all changed files, and search the changed paths for TODO/debug artifacts.
6. Commit specific feature files as `feat(optimizers): add NovoGrad layer-wise moments`.
7. Push `master`, then verify clean synchronized status and pushed SHA.
