# GaLore (Gradient Low-Rank Projection) Optimizer

> **For Hermes:** Single-task delivery: implement GaLore (Zhao et al. 2024,
> https://arxiv.org/abs/2403.03508) at
> `include/nn/optimizers/galore.{h,cpp}` plus
> `tests/test_galore.cpp`. Register in Makefile and add to
> `include/nn/nn.h` umbrella.

**Goal:** Memory-efficient optimizer that projects per-parameter gradients
onto a low-rank subspace (via top-r right singular vectors of the EMA
gradient covariance), applies Adam in that subspace, then projects back.
Reduces optimizer-state memory by `O(r/d + r/k)` vs Adam at rank `r ≪ min(d,k)`.

**Why this addition:** The library has 32+ optimizers covering the Adam
family, Shampoo/SOAP eigendecompositions, and D-Adaptation-style scaling,
but no low-rank projection variant. GaLore is one of the most-cited 2024
optimizer papers and is the standard reference for memory-efficient LLM
fine-tuning. Filling this gap rounds out the optimizer family.

**Architecture (single-pass per step, projecting per-PARAMETER not per-layer):**

Per parameter (shape `(m, n)`, target rank `r ≤ min(m, n)`):
1. **Gradient EMA** (rank projection): `G_EMA_t = β2 · G_EMA_{t-1} + (1-β2) · G_t`
   (full matrix, same shape as G).
2. **Rank-r projection update** (every `proj_update_interval` steps, default 200):
   - QR on `G_EMA_t` to get an orthonormal `P_t ∈ ℝ^{n×r}` (column basis of the row space).
   - The projection matrix: `P_t` is used to project gradient onto row-space.
3. **Adam-in-subspace** (every step):
   - `g_low = G_t @ P_t`                       (shape `(m, r)` — low-rank grad)
   - `m_low = β1 · m_low + (1-β1) · g_low`
   - `v_low = β2 · v_low + (1-β2) · g_low²`
   - `m̂ = m_low / (1-β1^t)`
   - `v̂ = v_low / (1-β2^t)`
   - `update_low = m̂ / (sqrt(v̂) + ε)`
   - `update = update_low @ P_t^T`             (project back to original shape)
   - `θ -= lr · update` (with decoupled weight decay if `weight_decay > 0`)

**Mathematical note:** This is the **projection-only** variant of GaLore
(Zhao et al. §3, Eq. 4) — the simpler formulation that achieves most of
the memory savings without the dual low-rank trick. The full GaLore
"memory-efficient" variant (separate LoRA-style A/B factors) is
strictly an optimization; for the API contract that matches the
PyTorch reference, we ship the projection-only variant with full
optimizer state in the projected space.

**Defaults (Zhao et al. 2024 §5, recommended for LLM fine-tuning):**
- `lr = 0.001`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`
- `rank = 4` (paper: `r ∈ {2, 4, 8}` works well)
- `proj_update_interval = 200` (paper: update projection every T steps)
- `weight_decay = 0.0` (decoupled, AdamW-style)
- `scale = 1.0` (no extra scaling on the projection)

**Validation rules:**
- `lr > 0`, `β1 ∈ [0, 1)`, `β2 ∈ [0, 1)`, `ε > 0`, `rank ≥ 1`,
  `proj_update_interval ≥ 1`, `weight_decay ≥ 0`, `scale > 0`.

**State per parameter (lazy on first step):**
- `P` (n × r) — projection matrix
- `m_low` (m × r) — first moment in projected space
- `v_low` (m × r) — second moment in projected space
- `G_EMA` (m × n) — gradient EMA used for projection refresh
- `t_proj` — last step when `P` was refreshed
- `step` — Adam step counter (separate from `t_proj`)

**Global state:** step counter (Adam-style, starts at 1).

**Public API (parity with Prodigy/SOAP):**
- Constructor with paper defaults
- All hyperparameter setters with `std::invalid_argument` on invalid input
- `get_*` accessors for every hyperparameter
- `step(Model&)` (per-parameter update)
- `handles_weight_decay() → true` (decoupled)
- `has_state(layer_ptr) / num_params_with_state(layer_ptr)`
- `get_P / get_m_low / get_v_low / get_G_EMA / get_step / get_step_proj`
  for testing

**Files:**
- `include/nn/optimizers/galore.{h,cpp}` (NEW)
- `tests/test_galore.cpp` (NEW)
- `Makefile` (build rule + tests deps + run_tests echo)
- `include/nn/nn.h` (umbrella include)
- `EXPANSION_QUEUE.md` (move to Done after commit)

**Test coverage (target ≥ 40 focused checks):**
1. **Defaults** — every constructor default matches paper
2. **Constructor with non-default args** — round-trip
3. **Setter validation** — invalid input throws on each setter
4. **Step counter starts at 1**
5. **First step** — bias correction applied (`1-β^t`)
6. **Closed-form first step** — β1=β2=0.5, lr=1, eps=1, g=1, init=0 → expected value
7. **Bias correction scales m and v properly** — verified by hand-computed formula
8. **Projection shape** — `P` is `(n, r)` where `g` is `(m, n)`
9. **First step initializes projection** — `G_EMA` and `P` populated
10. **Projection refresh interval** — `P` unchanged for `proj_update_interval-1` steps, refreshed on the next
11. **Projection matrix is orthonormal** — `P^T P ≈ I` (r × r)
12. **Projection-only variant matches GaLore §3 math** — `update = update_low @ P^T`, applied to param
13. **Grad EMA updates every step** — even between projection refreshes
14. **m_low / v_low stay in projected space** — shape (m, r)
15. **Projection-on-same-grads is stable** — gradient zero → no change (within FP noise)
16. **Decoupled weight decay** — `param *= (1 - lr·wd)` per step
17. **Multi-step determinism** — same grads → same params
18. **Multi-layer independence** — each layer's projection separate
19. **`handles_weight_decay() → true`**
20. **End-to-end training** — y=2x regression reduces loss > 50% over 60 steps
21. **Skinny (m > n) tensor** — Dense(3, 5) → weights (5, 3), rank=2 → P (3, 2)
22. **Square tensor** — (4, 4) → rank=2 → P (4, 2)
23. **Param shape preservation** — output shape unchanged
24. **Rank ≥ 1** — throws on rank=0
25. **Update interval ≥ 1** — throws on interval=0
26. **Projection refresh at step 1** — first step does refresh (since `t_proj == 0`)
27. **state accessors before step** — return empty (0, 0) tensors
28. **step() doesn't crash on empty model**
29. **Single-step closed form with grad passed through rank-1 projection** — exact arithmetic verification

**Mutation-tested** (per `test-driven-development` skill hygiene):
- Drop the projection (`update_low = m̂ / (sqrt(v̂) + ε)` directly to original shape) → step produces wrong param update.
- Use wrong EMA decay (1-β instead of β) → multiple gradient checks fail.
- Skip bias correction → step counter test fails.
- Apply weight decay in the wrong direction → decoupled WD test fails.

**Pitfalls to watch:**
- `Dense` weights are `(out, in)` — projection `P` is `(in, r)`. Mixed orientation is the #1 gradient bug in projection methods.
- `proj_update_interval` is per-parameter, not global. Resetting on each new param is correct.
- `G_EMA` is full matrix, not projection-rank. That's the memory cost of the projection-only variant (vs full GaLore).
- `Tensor::operator*` does row-major `(m, k) @ (k, n) → (m, n)`. Our `g @ P` is `(m, n) @ (n, r) → (m, r)` ✓. `update_low @ P^T` is `(m, r) @ (r, n) → (m, n)` ✓.
- `Tensor::transpose()` is non-trivial — keep `P^T` explicit (don't reuse `P` after transposing in place).

**Verify-before-final:**
- `git add -p` for each new file (no `git add .`).
- `make tests` runs from clean rebuilt `build/`.
- 100% of focused checks pass; 0 mutation tests pass after stubs.
- Commit message: `feat(optimizers): add GaLore (Zhao 2024) — N/N tests pass at machine precision`.
- Push to `origin/master`.
