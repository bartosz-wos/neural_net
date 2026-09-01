# TTT-Linear (Test-Time Training, Linear Variant) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a TTT-Linear layer (Sun et al., NeurIPS 2024, "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", https://arxiv.org/abs/2407.04620) to `include/nn/layers/recurrent/ttt_linear.{h,cpp}`. The "linear" variant has closed-form updates and is the natural starting point.

**Architecture:** TTT-Linear is a sequence mixer where the *recurrent state IS the weight matrix of a small linear MLP*. At each time step, the layer updates this weight matrix via one (or more) steps of gradient descent on a self-supervised reconstruction loss applied to the current input. The closed-form update avoids explicit gradient descent in the inner loop.

**Tech Stack:** C++17, existing Tensor (2D), in `include/nn/layers/recurrent/`.

---

## Mathematical formulation

A TTT-Linear layer is a map `R^{T × d_model} → R^{T × d_model}` defined by:

```
Inputs:  x ∈ R^{T × d_model},  T = sequence length
State:   W ∈ R^{d_inner × d_inner}    (the linear-MLP weight, "fast weights")
Bias:    b ∈ R^{d_inner}              (per-channel bias, learnable, slow weights)

Per-token forward (t = 0..T-1):
  z_t   = LN(x_t)                                  ∈ R^{d_inner}    (pre-norm)
  y_hat_t = W_{t-1} · z_t                          ∈ R^{d_inner}    (linear-MLP output)
  # Self-supervised target: reconstruction of z_t (the "denoising" objective)
  target_t = z_t
  # Compute closed-form update via Woodbury identity for least-squares:
  #   minimize ||W z - z||² + λ · ||W - W_{t-1}||²
  # Closed-form: W_t = W_{t-1} - η_t (W_{t-1} z_t z_t^T - z_t z_t^T) / (||z_t||² + λ)
  # which simplifies to:
  #   W_t = (I - η_t z_t z_t^T / (||z_t||² + λ)) W_{t-1} + η_t z_t z_t^T / (||z_t||² + λ)
  # But the simplified version above (single-step GD on quadratic) is:
  #   W_t = W_{t-1} - η_t · (W_{t-1} z_t - target_t) ⊗ z_t / (||z_t||² + λ)
  o_t = W_t · z_t + b
  cache z_t, W_{t-1}, W_t for backward
```

### Shape conventions
- Input `x`: `(T, d_model)` 
- Hidden dim `d_inner`: typically `d_model` (must be > 0)
- Pre-LN projects `d_model → d_inner` (a Dense layer)
- Post-projection: `d_inner → d_model` (a Dense layer) — so the layer composes cleanly
- State `W`: `(d_inner, d_inner)` persistent matrix (the "fast weights")
- Bias `b`: `(d_inner,)` (slow, learnable)
- η_t: per-token learning rate (scalar, derived from input)

### Closed-form update derivation
For quadratic loss L(W) = 0.5 · ||W z - z||² + 0.5 · λ · ||W - W_{t-1}||²_F:
  ∂L/∂W = (W z - z) z^T + λ (W - W_{t-1})

Setting ∂L/∂W = 0:
  (W z) z^T + λ W = z z^T + λ W_{t-1}
  (W z - z) z^T + λ (W - W_{t-1}) = 0
  ⇒ This is implicit in W; not directly solvable.

For the SIMPLIFIED (TTT-Linear paper §3.2) update rule (single GD step):
  W_t = W_{t-1} - η · (W_{t-1} z_t - z_t) z_t^T / (||z_t||² + λ)

This is exactly one gradient descent step on the unregularized quadratic loss. It's the canonical TTT-Linear update — tractable, closed-form per token, and matches the paper.

### Backward (per-token, in reverse order)
For each token t = T-1..0, given the upstream gradient grad_o_t ∈ R^{d_inner}:
  d/dW_{t-1}: backprop through the per-step update + the output = sum of contributions from all later tokens
  d/dz_t: similar
  d/db: standard Dense-like bias gradient

### Key insight for implementation
The full BPTT backward through TTT is O(T · d_inner² · T) in the naive form. The paper exploits the mini-batch ("chunk") form to make it O(d_inner² · T) total via the Woodbury identity. We implement the simple T-by-T form with explicit unrolling (T small in tests, e.g. T=3-5), which makes the gradient check tractable and the code readable.

### Learnable parameters
- Pre-projection `W_in`: `(d_inner, d_model)` Dense
- Post-projection `W_out`: `(d_model, d_inner)` Dense
- State initialization: zero or small random
- Per-token learning rate η_t: derived from input (e.g., sigmoid(W_eta · z_t + b_eta)), in (0, 1)
- Bias `b`: `(d_inner,)`, init zero
- LayerNorm before the inner MLP

### Why this is a good choice
- Novel (NeurIPS 2024 spotlight)
- Distinct from existing recurrent layers (state is a weight matrix, not a vector)
- Closed-form forward update
- Tractable gradient check (T small)
- Test cases:
  - State shape (d_inner, d_inner)
  - Forward shape (T, d_model) → (T, d_model)
  - Input gradient FD vs analytical rel_err < 1e-4 (T small)
  - W_state gradient FD vs analytical rel_err < 1e-4
  - Training reduces loss in end-to-end test

## Plan structure

Tasks 1-4: TDD each step.

### Task 1: Skeleton + forward shape
- Write failing test (forward shape + finiteness + state init)
- Implement forward (TT-by-TT loop, closed-form W update)
- Verify shape and finite output
- Commit

### Task 2: Input gradient FD
- Write failing test (input grad FD vs analytical rel_err < 1e-4)
- Implement backward (full BPTT through T steps)
- Verify FD matches analytical
- Commit

### Task 3: State gradient FD
- Write failing test (state W init grad FD vs analytical)
- This is the critical path — the state gradient must accumulate through ALL time steps
- Verify FD matches analytical
- Commit

### Task 4: End-to-end training
- Write failing test (TTT-LinearModel with input projection → TTT-Linear → output projection → MSE loss)
- Verify training reduces loss over 50 SGD steps
- Commit

---

## Risks

- The backward through T time steps is O(T) sequential — for T > 10, gradient check may be slow. Keep test T small (T=3-4).
- The state gradient accumulates from ALL future tokens via the chain rule (the state W_{t-1} influences W_t which influences o_t which influences loss) — must be careful about gradient accumulation order.
- Per-token η_t may be tricky to derive — start with a CONSTANT η for the v1 (just an init param), then add the per-token variant in a v2.

---

## Validation gates

- Forward shape (T, d_model) → (T, d_model) verified
- Input gradient FD vs analytical rel_err < 1e-4 (T=3)
- W_state gradient FD vs analytical rel_err < 1e-4 (T=3)
- W_in / W_out gradient FD vs analytical rel_err < 1e-4
- End-to-end training reduces loss

---

## Cleanup pass

After each task, run `git status` to ensure no debug artifacts.