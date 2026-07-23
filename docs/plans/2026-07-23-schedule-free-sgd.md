# Schedule-Free SGD Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. (Already executed by the cron job that authored this file — see git history for the implementation commit.)

**Goal:** Implement `ScheduleFreeSGD` — the SGD companion to the already-implemented `ScheduleFreeAdamW` — in `include/nn/optimizers/schedule_free_sgd.{h,cpp}`.

**Architecture:** Mirror `ScheduleFreeAdamW` exactly except remove the second-moment path (no `β2`, no `exp_avg_sq`, no bias correction). The per-step update is the bare (coupled-decay) gradient step. State per parameter is one tensor (z). Same three-sequence coupling `y = (1-β1)·z + β1·x`, same `train(model)` / `eval(model)` mode-swap API.

**Tech Stack:** C++17, g++ `-std=c++17 -O2 -Wall -Wextra -march=native`, single-threaded.

---

## Algorithm (per parameter)

Per-step update (in train mode, parameter holding `y = p`):

```
u          = g + weight_decay * y           # coupled decay (no Adam denom)
z_{k+1}    = z_k - lr * u                   # iterate step (bare)
y_{k+1}    = ckp1 * z_{k+1} + (1-ckp1) * y_k
              + lr * (beta1*(1-ckp1) - 1) * u
```

where `ckp1 = weight / weight_sum` and `weight = ((k+1)^r) * (lr_max^weight_lr_power)`.

Mode swaps (canonical formulas, identical to Schedule-Free AdamW):
```
eval (y -> x):  p = p + (1 - 1/beta1) * (z - p)
train (x -> y): p = p + (1 - beta1) * (z - p)
```

State init at first step: `z = clone(param)`. No second moment.

---

## Files

- Create: `include/nn/optimizers/schedule_free_sgd.h`
- Create: `include/nn/optimizers/schedule_free_sgd.cpp`
- Create: `tests/test_schedule_free_sgd.cpp`
- Modify: `include/nn/nn.h` (umbrella include)
- Modify: `Makefile` (build rule + test deps + run_tests entry)

---

## Test Coverage (98/98 focused checks)

1. **T1 — defaults**: lr=1.0, β1=0.9, wd=0, warmup=0, r=0, wlp=2.0, k=0, lr_max=-1, weight_sum=0, train_mode=false, handles_weight_decay=true.
2. **T2 — constructor + validation**: non-default ctor + mutators throw on invalid inputs.
3. **T3 — state init shape + values**: post-step z[0][0]=-0.5 (z0=0.5, u=1); k=1; lr_max=1.0; weight_sum=1.0.
4. **T4 — closed-form first step (β1=0)**: z_1=-0.5, y_1=-1.5 bit-exact.
5. **T5 — z recurrence under constant gradient**: z_1=-2, z_2=-6, z_3=-6 for g=2,4,0.
6. **T6 — y and z are distinct sequences**.
7. **T7 — eval/train mode swap**: closed-form swap correctness + idempotence.
8. **T8 — coupled weight decay**: z shrinks by lr·wd·y; y shrinks accordingly.
9. **T9 — malformed layer guards**: param/grad count and shape mismatch throw.
10. **T10 — determinism over 30 random-grad steps**: two fresh instances produce bit-exact params + z.
11. **T11 — end-to-end linear regression**: Dense(1,1), loss 16.875 → 0.0004.
12. **T12 — signature test vs vanilla SGD**: SF-SGD produces a different trajectory than vanilla SGD on `[1,1,1,1,1,1]`.
13. **T12b — signature test vs Schedule-Free AdamW**: SF-SGD produces a different trajectory than SF-AdamW under high-variance grads `[10,0,10,0,10,0]`.
14. **T13 — ckp1 progression**: weight_sum 1→2→3→4 for default r=0, wlp=2.
15. **T14 — warmup schedule (linear ramp)**: lr_max ramps 2/3 → 4/3 → 2 → 2.
16. **T15 — independent state across layers**.
17. **T16 — independent state across parameters**: weight z and bias z differ in sign under opposite grad.
18. **T17 — gradient clearing after step**.
19. **T18 — Dense(2,2) integration**: loss 7.7 → 0.0009.
20. **T19 — r=1, wlp=0 progression**: weight_sum grows as triangular numbers 1→3→6.
21. **T20 — state accessors before step**: get_z returns empty (0,0).
22. **T21 — layer with zero params is skipped**.
23. **T22 — pure SGD reduction (β1=0)**: loss 16.875 → 0.00009.

---

## Mutation Tests (4 mutations, all caught)

1. **Drop `weight_decay·y` term in `u`** → T8 fails (z stays at 1.0, y stays at 1.0).
2. **Skip `z_{k+1}` update entirely** → 14 failures across T3/T4/T5/T6/T8/T15/T16.
3. **Drop `ckp1·z_{k+1}` term in y update** → 3 failures (T4 y, T6 distinctness, T8 weight decay).
4. **Wrong lerp weight in eval (use 1-β1 instead of 1-1/β1)** → 2 failures (T7 mode swap, T7 idempotence).

All 4 mutations caught, confirming test coverage exercises the full Schedule-Free SGD signature.

---

## Reference

- Defazio, Yang, Khaled, Mahdavi, Lacoste-Julien 2024 — "The Road Less Scheduled" (https://arxiv.org/abs/2405.15682), NeurIPS 2024 best-paper nominee.
- Canonical reference: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/sgd_schedulefree.py

---

## Verification

```bash
make build/test_schedule_free_sgd && ./build/test_schedule_free_sgd
# Expected: === Summary: 98 passed, 0 failed ===

make tests  # all 80+ suites compile, schedule_free_sgd linked into every binary
```

The companion Schedule-Free AdamW suite (`test_schedule_free_adamw`) remains at 110/110 — verified after this change.