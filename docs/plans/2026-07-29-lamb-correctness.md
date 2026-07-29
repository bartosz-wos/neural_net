# LAMB Optimizer Correctness Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Correct the existing LAMB optimizer to match You et al. 2019 Algorithm 2, add professional behavioral coverage, and register that coverage in the aggregate build and run targets.

**Architecture:** Preserve the public `LAMB` class and its existing five-argument constructor prefix, but make `Optimizer::lr` the single scheduler-visible learning-rate source. For each parameter tensor, compute bias-corrected Adam moments, form `adam_direction = m_hat / (sqrt(v_hat) + epsilon)`, add coupled weight decay, and compute the tensor-local trust ratio from the norm of that complete update. Preserve the repository's existing `trust_ratio_gamma` extension as a validated symmetric clamp on the ratio; add state/diagnostic accessors and fail-fast count/shape guards matching neighboring optimizers.

**Tech Stack:** C++17, repository `Tensor`/`Layer`/`Model`/`Optimizer` APIs, GNU Make, self-contained behavioral test executable.

---

## Decisions and references

1. **Primary source:** You et al., “Large Batch Optimization for Deep Learning: Training BERT in 76 minutes,” ICLR 2020, Algorithm 2, <https://arxiv.org/abs/1904.00962>.
2. **Maintained-reference cross-check:** the CybertronAI/PyTorch LAMB implementation forms `adam_step`, adds `weight_decay * parameter`, computes `adam_norm = ||adam_step||`, then applies `trust_ratio = ||parameter|| / adam_norm`.
3. **Correct recurrence:**
   - `m_t = beta1*m_{t-1} + (1-beta1)*g_t`
   - `v_t = beta2*v_{t-1} + (1-beta2)*g_t^2`
   - `m_hat = m_t/(1-beta1^t)`, `v_hat = v_t/(1-beta2^t)`
   - `r_t = m_hat/(sqrt(v_hat)+epsilon)`
   - `u_t = r_t + weight_decay*w_t`
   - `q_t = ||w_t||/||u_t||`, with `q_t=1` if either norm is zero, followed by the existing `[1/gamma, gamma]` clamp
   - `w_{t+1} = w_t - lr*q_t*u_t`
4. **Root-cause correction:** the old implementation used `||raw_gradient||` as the denominator of `q_t` and never added weight decay. That is not Algorithm 2 and can produce a materially different direction and scale.
5. **Compatibility:** keep `LAMB(lr, beta1, beta2, epsilon, trust_ratio_gamma)` valid; append `weight_decay=0.0`. Keep the existing public diagnostic fields where practical, but remove the shadowing `LAMB::lr` member so source expressions `optimizer.lr` resolve to `Optimizer::lr` and scheduler updates are consumed by `step()`.
6. **No new base classes or dependencies.** The change is limited to LAMB, its test, Makefile wiring, and the queue ledger.

## Task 1: Establish the public contract (RED -> GREEN)

**Objective:** Add the smallest test proving the wished-for scheduler-compatible API and defaults.

**Files:**
- Create: `tests/test_lamb.cpp`
- Modify: `include/nn/optimizers/lamb.h`
- Modify: `include/nn/optimizers/lamb.cpp`

**Step 1: Write failing test**

Add assertions for defaults: `lr=1e-3`, betas `(0.9,0.999)`, epsilon `1e-6`, gamma `10`, weight decay `0`, step counter `1`, and `handles_weight_decay()==true`.

**Step 2: Run test to verify failure**

Run: `make setup build/test_lamb`
Expected: FAIL at compile time because the accessors and focused target do not exist yet. During RED, compile directly if the Makefile target has not been added: `g++ ... tests/test_lamb.cpp ...`.

**Step 3: Write minimal implementation**

Add validated accessors/setters, append `weight_decay` to the constructor, assign `Optimizer::lr`, and expose `handles_weight_decay()`.

**Step 4: Run test to verify pass**

Run the focused binary and expect all Task-1 checks to pass.

**Step 5: Commit**

Defer commit until the single coherent LAMB correction is complete; do not commit a header-only intermediate state.

## Task 2: Reproduce and correct the algorithmic defect (RED -> GREEN)

**Objective:** Prove the trust ratio uses the complete Adam direction rather than the raw gradient.

**Files:**
- Modify: `tests/test_lamb.cpp`
- Modify: `include/nn/optimizers/lamb.cpp`

**Step 1: Write failing signature test**

Use a non-square/asymmetric tensor with `w=[3,4,0,0]`, `g=[6,8,0,0]`, `beta1=beta2=0`, `epsilon=1`, no decay, and a large gamma. Hand-compute `r=[6/7,8/9,0,0]`, `q=5/||r||`, and the exact updated weights. Assert both the cached ratio and coordinates.

**Step 2: Run test to verify failure**

Expected: the old kernel reports `q=||w||/||g||=0.5`, not `5/||r||`.

**Step 3: Write minimal implementation**

Update moments first, compute bias-corrected `r`, form a temporary complete update tensor, compute its norm, then apply `lr*q*u`.

**Step 4: Run test to verify pass**

Expected: the hand-derived ratio and parameter coordinates pass to `1e-12`.

## Task 3: Add coupled weight decay and edge cases (RED -> GREEN)

**Objective:** Cover the full Algorithm-2 update and numerical boundaries.

**Files:**
- Modify: `tests/test_lamb.cpp`
- Modify: `include/nn/optimizers/lamb.cpp`

**Steps:**
1. Add a zero-gradient, nonzero-decay test with `w=[3,4]`, `wd=0.5`; expect `u=0.5w`, `q=2`, and `w_new=0.9w` at `lr=0.1`.
2. Run and observe RED against the old no-decay kernel.
3. Add `weight_decay*w` before the update norm and trust-ratio scaling.
4. Add zero-weight and zero-update norm tests requiring `q=1` and finite output.
5. Add lower/upper gamma-clamp boundary tests using asymmetric fixtures.
6. Run focused tests to GREEN.

## Task 4: Harden state and malformed-input behavior (RED -> GREEN)

**Objective:** Match neighboring optimizer safety and introspection conventions.

**Files:**
- Modify: `tests/test_lamb.cpp`
- Modify: `include/nn/optimizers/lamb.h`
- Modify: `include/nn/optimizers/lamb.cpp`

**Steps:**
1. Add lazy-state tests for Dense weight/bias shapes and cloned `m`/`v` accessors.
2. Add two-layer state-isolation checks.
3. Add a custom malformed `Layer` for parameter/gradient count and shape mismatch; require `std::runtime_error`.
4. Run RED against the existing unchecked indexing path.
5. Add exact count/shape guards before state update and expose safe diagnostic accessors.
6. Run GREEN.

## Task 5: Verify integration behavior and non-vacuous coverage

**Objective:** Prove scheduler compatibility, determinism, optimization usefulness, gradient clearing, and mutation sensitivity.

**Files:**
- Modify: `tests/test_lamb.cpp`

**Steps:**
1. Add a `PolynomialLR` test proving scheduler writes to the learning rate consumed by LAMB.
2. Add a two-step varying-gradient test verifying stored `m`/`v` and bias correction.
3. Add deterministic repeated-trajectory checks.
4. Add an asymmetric two-output regression smoke test requiring substantial finite loss reduction.
5. Add gradient-clearing assertions.
6. Mutation-test at least three defects: denominator `||g||` instead of `||u||`; omit `weight_decay*w`; force `q=1`. Each mutation must fail a dedicated assertion, then be restored.

## Task 6: Wire build, umbrella coexistence, and aggregate gates

**Objective:** Make the focused suite part of normal repository verification without disturbing deferred failures.

**Files:**
- Modify: `Makefile`
- Test: `tests/test_lamb.cpp`
- Verify: `include/nn/nn.h`

**Steps:**
1. Add `build/test_lamb` link rule.
2. Add it once to the `tests:` prerequisites.
3. Add a `=== Running LAMB Tests ===` command to `run_tests`.
4. Verify `git diff -- Makefile` touches only the intended sections.
5. Compile the public umbrella header alone with `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only -`.
6. Run the focused suite from a clean object/binary.
7. Run `make tests` and mechanically compare compiled vs executed binaries.
8. Run `make run_tests`; the only compiled-but-not-executed binaries must remain the four entries in `NOT_FIXED.md`.

## Task 7: Update the expansion ledger and ship

**Objective:** Record the completed correctness/coverage expansion and push an atomic verified change.

**Files:**
- Modify: `EXPANSION_QUEUE.md`
- Keep: `NOT_FIXED.md` unchanged unless aggregate evidence changes a deferred entry
- Include: `docs/plans/2026-07-29-lamb-correctness.md`

**Steps:**
1. Add a concise `## Done` entry with the corrected recurrence, files, focused result, mutation result, and aggregate result.
2. Confirm no cleanup artifacts (`debug_*`, `*.bak`, `ref_*.cpp`) exist.
3. Review the exact diff for secrets, debug output, commented-out implementation, and scope drift.
4. Obtain independent review; fix blocking findings and re-run gates.
5. Stage explicit paths only.
6. Commit with conventional messages, keeping the implementation/test change atomic and the ledger update separate if useful.
7. Push `master` to `origin` and verify local/remote SHA equality.
