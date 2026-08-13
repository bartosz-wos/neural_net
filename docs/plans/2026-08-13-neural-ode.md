# Neural ODE / ODE-RNN Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add continuous-depth / ODE-based sequence models to the library — explicit ODE solvers (Euler, RK4, Midpoint, Dormand-Prince), a parameterized dynamics network, a `NeuralODE` block that integrates between two fixed time points, an `ODERNN` block that handles variable-length sequences (De Brouwer et al. 2019), and the **adjoint method** for O(1)-memory training through the solver.

**Architecture:** Composes existing primitives (`Tensor`, `Dense`, `Activation`). Files go in `include/nn/layers/architectures/neural_ode.{h,cpp}` (the architectures dir already houses composite model definitions like TimesNet/Hyena/Conformer that wrap a *block* with auxiliary logic). The forward path is a black-box ODE solver stepping `dh/dt = f(h, t, x)`. Two backward paths are implemented: (a) **direct backprop-through-solver** (the easy path; O(depth) memory), and (b) **the adjoint method** (the elegant path; O(1) memory w.r.t. depth — Chen et al. 2018 §4). Both must agree numerically.

**Tech Stack:** Existing `Tensor`, `Layer`, `Dense`, `Activation`. Standard `<vector>`, `<cmath>`, `<memory>`, `<functional>`. Numerical integration uses single-precision doubles.

---

## Background

### Paper context

The "Neural Ordinary Differential Equations" paper (Chen, Rubanova, Du, Chen; NeurIPS 2018, https://arxiv.org/abs/1806.07366) reframes a residual network as the Euler discretization of a continuous transformation:

```
h_{t+1} = h_t + f(h_t, θ_t)             (ResNet step)
dh/dt   = f(h(t), t, θ)                 (Neural ODE — continuous analogue)
```

The model is the ODE itself. Training is gradient computation through a black-box ODE solver. The paper's killer feature is the **adjoint method**: instead of storing all intermediate `h_t` values during the forward solve and backpropping through them (which would require O(depth) memory), solve a second ODE backward in time for the adjoint state `a(t) = dL/dh(t)` and integrate the parameter gradients `dL/dθ = -∫ a(t)^T ∂f/∂θ dt`. Memory cost becomes O(1) in the depth of the solver.

### De Brouwer et al. 2019 (ODE-RNN) — for sequence data

For variable-length sequence data with irregular observation times, "GRU-ODE-Bayes" (De Brouwer et al. 2019, https://arxiv.org/abs/1905.04374) couples a Neural ODE for the between-observation evolution with a GRU-style update at each observation. We implement a simpler variant: `ODERNN(input_dim, hidden_dim, output_dim, solver_type)` evolves the hidden state via the ODE between observations, then injects input via a standard gated update.

### Math

**ODE solver interface** (per-step signature): given `f(h, t, x)` and current state `(h, t)`, advance to `t + dt` and return the new `(h_new, t_new)`.

**Euler**: `h_new = h + dt · f(h, t, x)`. O(1) stage.

**Midpoint**: `k1 = f(h, t, x); h_mid = h + dt/2 · k1; k2 = f(h_mid, t + dt/2, x); h_new = h + dt · k2`. 2 stages, 2nd-order accurate.

**RK4**: classical 4-stage. 4 stages, 4th-order accurate.

**Dormand-Prince (RK45)**: 7-stage adaptive solver with embedded error estimate. Default `rtol=1e-3, atol=1e-4`. We use a fixed-step DOPRI5 for simplicity (no adaptive step control — that's a v2 feature).

**Direct backprop**: at each solver step, also compute `dh_new/dh = I + dt · ∂f/∂h` and chain. For Euler: `dh_new/dh = I + dt · J_f` where `J_f = ∂f/∂h`. This is the easy path — requires storing all intermediate `h_t` and `f_t` values.

**Adjoint method**: solve the augmented ODE backward in time:
```
da/dt = -a^T · ∂f/∂h                  (adjoint dynamics — same shape as h)
dL/dθ = -∫ a^T · ∂f/∂θ dt             (integral form for parameters)
```
We implement this with a second ODE solve running backward from `t=T` to `t=0`, accumulating parameter gradients at each step. To keep memory O(1), we DON'T store the forward trajectory — we recompute `f` at each adjoint step using the adjoint's own time-march.

For a non-stiff ODE with smooth `f`, both paths must produce numerically identical parameter gradients (the adjoint is mathematically equivalent to backprop, just computed differently).

### Why this addition

- **Coverage gap.** The repo has Mamba/H3/RWKV/RetNet/GLA/DeltaNet (discrete-time SSMs), Hawk/LTC (gated recurrences), and 35+ optimizers — but no continuous-depth family. Neural ODE is one of the most-cited 2018 papers and fills a real conceptual gap.
- **Educational value.** The adjoint method is a beautiful algorithm that deserves a clean implementation in this repo. It's also a great reference for future ODE-related work (Neural SDEs, Graph Neural ODEs, FFJORD).
- **Composition with existing code.** The `Dense` + `Activation` primitives already cover what `f(h, t, x)` needs. No new building blocks.

---

## Layout

New files:
- `include/nn/layers/architectures/neural_ode.h`
- `include/nn/layers/architectures/neural_ode.cpp`
- `tests/test_neural_ode.cpp`

Edits:
- `include/nn/nn.h` — add `#include "layers/architectures/neural_ode.h"`
- `Makefile` — add `build/test_neural_ode` rule, `tests:` deps entry, `=== Running Neural ODE Tests ===` echo line

---

## Phase 0 — Design decisions

### D1: Where to put the files

`include/nn/layers/architectures/neural_ode.{h,cpp}`. The architectures dir already houses composite models (TimesNet, Hyena, Conformer) that wrap a *block* with auxiliary logic. Neural ODE and ODE-RNN are model-level definitions that compose multiple sublayers.

### D2: Solver API surface

Expose the solvers as a `namespace odesolver` with free functions:
```cpp
Tensor euler_step(const ODEFunc& f, const Tensor& h, double t, double dt, const Tensor& x);
Tensor rk4_step   (const ODEFunc& f, const Tensor& h, double t, double dt, const Tensor& x);
Tensor midpoint_step(...);
Tensor dopri5_step(...);
```
This keeps the API testable in isolation (we can write a "solver agrees with analytic solution" test without constructing a `NeuralODE`).

### D3: ODEFunc interface

`ODEFunc` is a `Layer` subclass with `forward(h, t, x) -> dh/dt` (returns a tensor of the same shape as h). For tractability of gradient checks, we keep `t` as a `double` (not part of the state) — most practical Neural ODEs don't depend explicitly on `t`. The `x` is the constant input conditioning (held fixed during integration). Default implementation: 2-layer MLP with tanh (the canonical Neural ODE `f`).

### D4: Backward modes

Two separate `backward` paths, switchable via a `use_adjoint_` bool:
- **Direct** (default): forward trajectory is stored, then backprop chains `dh_new/dh = I + dt · ∂f/∂h` at each step.
- **Adjoint**: no trajectory storage; a second backward-in-time ODE solve computes `a(t)`, and parameter gradients accumulate via `a^T · ∂f/∂θ dt`.

Both must agree at ~1e-5 on a non-stiff problem. The direct path is the *reference* — the adjoint path is verified against it.

### D5: Sequence handling (ODERNN)

`ODERNN` uses fixed-step solver (Euler/RK4) between observations, with input injection at each observation time:
```
h_{t_k} = RNN_step(h_{t_{k-1}}, x_k)   # input injection at observation
h(t)    = ODESolve(h_{t_k}, t_k, t_{k+1})  # continuous evolution to next obs
```
For simplicity, we treat each observation as "h gets updated by gated input, then evolves via ODE to the next observation time." `Δt` is configurable per-sequence via a `last_dt_` field.

### D6: Numerical gradient check strategy

- **Direct backward** vs **central finite difference**: should match at ~1e-7 (machine precision).
- **Adjoint backward** vs **direct backward**: should match at ~1e-5 (the adjoint integrates numerically — second-order accuracy of the solver compounds).
- **Solver convergence**: RK4 should match RK4-with-half-step at ~1e-15 on a smooth test problem.

---

## Phase 1 — TDD red/green per task

### Task 1: ODESolver free functions (Euler + Midpoint + RK4 + DOPRI5)

**Files:**
- Create: `include/nn/layers/architectures/neural_ode.h` (the namespace declarations only)
- Create: `include/nn/layers/architectures/neural_ode.cpp` (the namespace implementations)
- Create: `tests/test_neural_ode.cpp` (test scaffolding)

**Step 1:** Write the failing test. Construct a trivial `f(h, t, x) = h` (exponential growth). Integrate from `h(0)=1` to `t=1` with `dt=0.1` using Euler, Midpoint, RK4. Assert:
- Euler at `t=1`: ~`2.59` (close to `e^1 ≈ 2.718`).
- RK4 at `t=1`: ~`2.72` (closer).
- Hand-derived single-step Euler: `1 + 0.1 · 1 = 1.1` matches first step.

Run with `make build/test_neural_ode && ./build/test_neural_ode`. Expected: FAIL — `ODESolver` functions don't exist yet.

**Step 2:** Implement the four solver functions as free functions taking a callable `f`. Use `std::function<Tensor(const Tensor&, double, const Tensor&)>` for the dynamics. Use `Tensor` arithmetic (already supported).

**Step 3:** Run test. Expected: PASS.

**Step 4:** Commit.

```bash
git add include/nn/layers/architectures/neural_ode.h include/nn/layers/architectures/neural_ode.cpp tests/test_neural_ode.cpp
git commit -m "feat(architectures): Neural ODE solvers (Euler, Midpoint, RK4, DOPRI5) - red/green for first test"
```

### Task 2: ODEFunc — parameterized dynamics network

**Step 1:** Add failing test: `ODEFunc(2, 4).forward(h, 0.0, x)` returns a `(1, 4)` tensor when `h` is `(1, 2)`. `parameters()` returns `4` tensors (W1, b1, W2, b2). Constructor with `input_dim=0` throws.

**Step 2:** Implement `ODEFunc : public Layer` with two `Dense` layers (`(hidden + input) → hidden → hidden`) and a tanh. Cache `last_h_`, `last_t_`, `last_x_` for backward.

**Step 3:** Run test. PASS.

**Step 4:** Commit.

### Task 3: NeuralODE forward (no backward yet)

**Step 1:** Add failing test: `NeuralODE(2, 4, 3, EULER, 0.1).forward(x)` returns a `(1, 3)` tensor. Zero input gives zero output (because `f(0) = 0` for the linear-with-zero-bias MLP).

**Step 2:** Implement `NeuralODE : public Layer`. Constructor takes `(input_dim, hidden_dim, output_dim, solver_type, dt)`. Forward: `h_0 = 0`, then loop `t ∈ [0, dt, 2*dt, ..., T]` calling the solver step, finally project `h_T → output_dim` via a Dense. Cache the trajectory for direct backward.

**Step 3:** Run test. PASS.

**Step 4:** Commit.

### Task 4: NeuralODE direct backward — input + parameter gradients

**Step 1:** Add failing tests:
- `NeuralODE.backward(grad_output, 0.0)` returns a `(1, input_dim)` tensor.
- `parameters()` returns `4 + 2 = 6` tensors (4 from ODEFunc, 2 from output projection).
- Hand-derived Euler backward for `dh/dt = h`: starting from `h(0)=1`, integrating to `t=1` with `dt=0.1`, the gradient of `h(1)` w.r.t. `h(0)` is `(1 + dt)^10 = 2.5937`. Test `rel_err < 1e-7`.

**Step 2:** Implement `backward` using the chain rule on the stored trajectory:
```cpp
for step t from T-1 down to 0:
    grad_h_t = grad_h_{t+1} · (I + dt · ∂f/∂h at step t) + grad_proj_term
    grad_ODEFunc_params += grad_h_t · ∂f/∂θ at step t
```
For `∂f/∂h`, use the analytic Jacobian of `f` (Dense + tanh → easy to derive). For `∂f/∂θ`, use the parameter gradients accumulated from the `Dense` backward passes during the forward step.

**Step 3:** Run numerical gradient check test. PASS at `rel_err < 1e-7`.

**Step 4:** Commit.

### Task 5: NeuralODE adjoint backward

**Step 1:** Add failing test: `NeuralODE.backward(grad_output, 0.0)` with `use_adjoint_=true` matches the direct backward at `rel_err < 1e-5`.

**Step 2:** Implement adjoint method:
```cpp
// augmented state: a_h (gradient w.r.t. h) + a_theta (gradient w.r.t. params, integrated)
Tensor a_h = grad_output_replicated_to_hidden_dim;
Tensor a_theta = zeros;
for step t from T-1 down to 0:
    Tensor J_h = odefunc_.jacobian_h(last_h_t_, t, last_x_);   // (hidden, hidden)
    Tensor J_theta = odefunc_.jacobian_theta(last_h_t_, t, last_x_);  // (hidden, n_params)
    a_h = a_h · (I + dt · J_h)
    a_theta -= dt · a_h · J_theta
    grad_odefunc_params += a_theta
```
Note: this is the **continuous adjoint** approximation. For RK4, the adjoint of an RK4 step is the "adjoint RK4" — but for simplicity, we use Euler for the adjoint step (since the test uses Euler forward, they match exactly).

**Step 3:** Run test. PASS at `rel_err < 1e-5`.

**Step 4:** Commit.

### Task 6: ODERNN forward + backward

**Step 1:** Add failing test: `ODERNN(2, 4, 3, EULER, 0.1).forward(X_seq, T_seq)` where `X_seq` is `(T, 2)` and `T_seq` is `(T,)` observation times. Returns `(T, 3)` predictions. `backward` returns `(T, 2)` grad.

**Step 2:** Implement: at each observation `t_k`:
- `h_{k-1/2} = RNN_step(h_{k-1}, x_k)` — gated input injection (Dense(x) + Dense(h) + tanh, standard).
- `h_k = ODESolve(h_{k-1/2}, t_{k-1}, t_k)` — evolve via ODE.
- `o_k = output_proj(h_k)`.
Backward chains through both the RNN update and the ODE solve at each step.

**Step 3:** Run numerical gradient check. PASS.

**Step 4:** Commit.

### Task 7: Training reduces loss (end-to-end)

**Step 1:** Add failing test: synthesize a target trajectory `(y_t = sin(t))`, train `ODERNN` for 50 SGD steps, assert `loss` decreases by >50%.

**Step 2:** Implement training loop in the test (already common pattern in other test files).

**Step 3:** Run test. PASS.

**Step 4:** Commit.

### Task 8: Register in Makefile + nn.h

**Step 1:** Add `#include "layers/architectures/neural_ode.h"` to `nn.h`.
**Step 2:** Add `build/test_neural_ode` rule to Makefile.
**Step 3:** Add `$(BUILD_DIR)/test_neural_ode` to the `tests:` deps.
**Step 4:** Add `echo "=== Running Neural ODE Tests ==="` to `run_tests`.
**Step 5:** Run `make tests`. Verify no regressions.
**Step 6:** Commit.

---

## Phase 2 — Audit

- [ ] Every test passed at expected precision
- [ ] Numerical gradient checks pass for both direct and adjoint paths
- [ ] No regression in existing 72-suite integration
- [ ] `make tests` clean
- [ ] Mutation-tested: stub out a step of the adjoint integration and confirm test catches it

---

## Verification

```bash
make build/test_neural_ode && ./build/test_neural_ode
make tests | grep -i "neural.ode"
```

Expected: all focused checks pass; existing 72 suites still pass.

---

## Bug-fix history (filled in during implementation)

- (placeholder — populated as we encounter issues)

---

## Phase 3 — Update EXPANSION_QUEUE

After passing all tests, move the entry from `## Ideas` to `## Done` with a one-line summary of test results and any bug fixes encountered.

```bash
# Move Neural ODE entry to Done
```
