# Signum Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the paper-correct Signum optimizer—signSGD with exponential-moving-average momentum—with professional validation, state introspection, and non-vacuous behavioral tests.

**Architecture:** `Signum` derives from `Optimizer`, uses inherited `Optimizer::lr` for scheduler compatibility, and stores one EMA momentum tensor per parameter. Each step computes `m_t = beta*m_{t-1} + (1-beta)*g_t`, applies an element-wise `sign(m_t)` update, and then applies decoupled weight decay in the same combined update expression used by Lion. This intentionally differs from the queue draft: a second moment is not part of Bernstein et al.'s Signum, and `sign(m_hat/sqrt(v_hat)) == sign(m_hat)` for nonnegative `v_hat`, so such state would be algebraically inert.

**Tech Stack:** C++17, repository `Tensor`/`Layer`/`Model` abstractions, Make, custom assertion harness, strict RED→GREEN→REFACTOR TDD with mutation testing.

---

## References

- Bernstein, Wang, Azizzadenesheli, Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018, Algorithm 2 (Signum): https://proceedings.mlr.press/v80/bernstein18a.html
- Apache MXNet Signum production API: `state = momentum*state + (1-momentum)*grad`; `weight = (1-lr*wd)*weight - lr*sign(state)`.
- Comparative implementation: `include/nn/optimizers/lion.{h,cpp}`.

## Algorithm

```
m_t = beta * m_{t-1} + (1-beta) * grad_t
s_t = (m_t > 0) - (m_t < 0)
param_t = param_{t-1} - lr * (s_t + weight_decay * param_{t-1})
```

`beta=0` reduces exactly to signSGD. Bias correction is unnecessary because multiplication by the positive scalar `1/(1-beta^t)` cannot change `sign(m_t)`. `sign(0)=0` gives exact zero-gradient invariance when weight decay is disabled.

## Files

- Create: `include/nn/optimizers/signum.h`
- Create: `include/nn/optimizers/signum.cpp`
- Create: `tests/test_signum.cpp`
- Modify: `include/nn/nn.h`
- Modify: `Makefile`
- Modify: `EXPANSION_QUEUE.md`

## TDD Tasks

### Task 1: RED — tests and focused build rule

1. Create `tests/test_signum.cpp` covering defaults/setters/validation, lazy one-tensor state, first-step and multi-step closed forms, signSGD reduction, exact constant-magnitude updates, zero-gradient behavior, decoupled weight decay, a sequence distinguishing Signum from Lion, scheduler compatibility, state isolation, malformed-layer guards, deterministic trajectories, gradient clearing, and training convergence.
2. Add only the focused `build/test_signum` rule to the Makefile.
3. Run `make build/test_signum` and observe the expected missing-header failure.

### Task 2: GREEN — header and implementation

1. Create `include/nn/optimizers/signum.h` with constructor, validated setters/accessors, `step`, `has_state`, and `get_momentum`.
2. Create `include/nn/optimizers/signum.cpp` implementing the exact recurrence above.
3. Use `Optimizer::lr` as the sole learning-rate field.
4. Validate `lr >= 0`, `beta in [0,1)`, and `weight_decay >= 0`.
5. Reject parameter/gradient count and shape mismatches before indexing.
6. Run the focused test until all assertions pass.

### Task 3: Integration and queue bookkeeping

1. Add `#include "optimizers/signum.h"` to `include/nn/nn.h`.
2. Add `test_signum` to `tests:` and `run_tests:`.
3. Replace the queue draft under `Ideas` with a concise paper-correct entry under `Done`, including focused and aggregate results.

### Task 4: Mutation testing and full verification

1. Mutate `sign(momentum)` to `momentum`; confirm constant-magnitude and closed-form tests fail.
2. Restore and rerun the focused suite.
3. Run `make tests`.
4. Mechanically compare compiled test targets against `run_tests` invocations.
5. Run `make run_tests` in the foreground and separate any pre-existing deferred flaky failures from regressions.

### Task 5: Commit and push

```bash
git add docs/plans/2026-07-22-signum-optimizer.md
git add include/nn/optimizers/signum.h include/nn/optimizers/signum.cpp
git add tests/test_signum.cpp include/nn/nn.h Makefile EXPANSION_QUEUE.md
git commit -m "feat(optimizers): add Signum sign-momentum optimizer"
git push origin master
```

## Verification checklist

- [ ] Focused test observed RED before production code existed.
- [ ] One state tensor per parameter; no inert second-moment state.
- [ ] `beta=0` reduces exactly to signSGD.
- [ ] Persistent nonzero gradient updates by exactly `lr` per coordinate.
- [ ] Momentum sequence test distinguishes Signum from Lion.
- [ ] Weight decay is decoupled and not double-applied by wrappers.
- [ ] Critical mutation is caught.
- [ ] Focused suite, aggregate compile, and stable run suite pass.
- [ ] Commit pushed to `origin/master`.
