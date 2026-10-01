# Megalodon Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the Megalodon long-context sequence architecture (complex EMA +
timestep normalization + normalized gated attention + two-hop residual).

**Architecture:** Three new components in
`include/nn/layers/{normalization,recurrent,architectures}/`:
`TimestepNorm` (§3.2, causal cumulative normalization along the timestep
axis), `ComplexEMA` (§3.1, multi-dimensional EMA over the complex domain),
and `MegalodonBlock` / `MegalodonModel` (the full block, §3.3–3.4). All
backward passes are hand-derived and finite-difference verified.

**Tech Stack:** C++17, `Tensor`/`Layer`/`Dense` from `nn/core`, `RMSNorm`,
`LayerNorm`, `SwiGLU`, `std::complex`. Mirrors `mega.{h,cpp}` conventions.

**Reference:** Ma, Yang, Xiong, Chen, Yu, Zhang, Ma, Zhou — "MEGALODON:
Efficient LLM Pretraining and Inference with Unlimited Context Length",
https://arxiv.org/abs/2404.08801. Official implementation
`XuezheMax/megalodon` (`megalodon/modules/complex_exponential_moving_average.py`,
`moving_average_gated_attention.py`, `normalized_feedforward_network.py`,
`model/mega.py`).

---

## Paper specification

### §3.1 CEMA — Complex Exponential Moving Average (Eq. 2–3)

Mega's multi-dimensional damped EMA (Eq. 1) is extended to the complex domain.
For each channel `j` and each of the `h` (paper `ndim`) complex hidden states:

```
h_{t,j,k} = p_{j,k}·x_{t,j} + q_{j,k}·h_{t-1,j,k}        (complex)
y_{t,j}   = Re( Σ_k η_{j,k}·h_{t,j,k} ) + ω_j·x_{t,j}
```

Parameterization (from the official implementation, `_calc_coeffs`):

```
θ_{j,k} = (k+1) · sigmoid(θ_j) · (2π/h)        k = 0..h-1   (Eq. 3)
p_{j,k} = sigmoid(α_{j,k})                                   ∈ (0,1)
q_{j,k} = polar(1 - α_{j,k}·δ_{j,k}, θ_{j,k}) = (1-α·δ)·(cos θ + i sin θ)
η_{j,k} ∈ C   (stored as two real params, scaled by 1/√h)
ω_j     ∈ R   (residual path coefficient)
```

`|q| = 1 - α·δ < 1` so the kernel decays — the paper's stated key property.

**Correspondence to the paper.** The paper writes
`h_t = α_j(cosθ + i sinθ) ⊙ u_t + (1 - α_j δ_j)(cosθ + i sinθ) ⊙ h_{t-1}`
with an explicit expansion matrix `β` mapping the scalar channel to `h` dims.
The official implementation absorbs `β` into `α` (so `p = α`) and absorbs the
expansion into `η`; the residual path `ω_j·x_{t,j}` is an addition in the
reference implementation not present in Eq. 2. We follow the reference
implementation, since it is what the released 7B model was trained with.

### §3.2 Timestep Normalization (Eq. 4)

Group Normalization along the timestep axis, made causal by using *cumulative*
statistics. For `d` features split into `k` groups of `d_g = d/k`:

```
μ_{t,g}  = 1/(t·d_g) · Σ_{i=1..t} Σ_{j∈g} x_{i,j}
σ²_{t,g} = 1/(t·d_g) · Σ_{i=1..t} Σ_{j∈g} (x_{i,j} - μ_{t,g})²
y_{t,j}  = (γ_j + 1)·(x_{t,j} - μ_{t,g(j)})/√(σ²_{t,g(j)} + eps) + β_j
```

The `+1` on the scale is the paper's **plus-1 reparameterization** (§Appendix
B.2) so weight decay keeps γ centered at zero. `k = 0` (or `k >= d`) means one
group over all features.

No future information is used at step `t` (only `x_{1..t}`), so this is
causal-safe for autoregressive modeling — the paper's stated motivation.

### §3.3 Normalized attention (Eq. 5–9)

```
X'  = CEMA(X)                                          ∈ R^{n×d}
Z   = X'·W_z + b_z                                      ∈ R^{n×z}
Z'  = Z / ‖Z‖                                           (per-token L2 norm)
Q   = κ_q ⊙ Z' + μ_q      K = κ_k ⊙ Z' + μ_k            ∈ R^{n×z}
O   = softmax(Q·Kᵀ)·V                                   ∈ R^{n×v}
```

The `‖Z‖` norm removes the need for the `τ(X)` scaling term, and the SiLU on
Z is removed because the norm supplies the nonlinearity. In the reference
implementation the normalization is an RMSNorm over the per-head slice of `z`,
which is algebraically the same operation up to the constant factor (RMSNorm
divides by `rms = ‖z‖/√d_h`; the paper divides by `‖z‖`) — we implement the
reference's per-head RMSNorm since it is numerically better conditioned and
combines with `γ/√d_h` as the paper's `κ`.

### §3.4 Pre-norm with two-hop residual (Eq. 11)

```
Ŷ = Attention(Norm(X)) + X
Y = FFN(Norm(Ŷ)) + X              ← note: + X, not + Ŷ
```

The FFN residual reuses the block input `X`, not the attention output `Ŷ`.
This removes the update gate `φ` of Mega entirely and is the paper's fix for
the variance growth of deep pre-norm stacks. Confirmed in the reference:
`MegaBlock.forward` does `y, cache = self.mega(x, ...)` then
`out = self.nffn(y, x)` — the second argument is the block input `x`.

TimestepNorm is applied only before the attention sublayer; LayerNorm before
the FFN (paper §3.4: LayerNorm is cheaper and the attention output is already
a mixture over timesteps).

---

## Design decisions

### Data layout

- `ComplexEMA`: parameters `alpha_ (d, h)`, `delta_ (d, h)`, `theta_ (d, 1)`,
  `eta_re_ (d, h)`, `eta_im_ (d, h)`, `omega_ (d, 1)`. Forward cache:
  `last_h_` as a `(d, h, 2)` real tensor holding the complex hidden state
  (`[..., 0]` = re, `[..., 1]` = im) — the repo's `Tensor` is real-only.
  Cache `p_ (d,h)`, `q_re_ (d,h)`, `q_im_ (d,h)`, `eta_ (d,h,2)`.
- `TimestepNorm`: `gamma_ (1, d)`, `beta_ (1, d)`. Cache `x_ (n, d)`,
  `mu_ (n, k)`, `var_ (n, k)`, `inv_std_ (n, k)`, `count_ (n)` (= t+1).
- `MegalodonBlock`: z-dim and v-dim split across `num_heads` heads; Q/K share
  the normalized `z` with separate `(γ, β)` per Q and K (2 × `z_dim` params,
  matching the reference's `gamma`/`beta` of shape `(2, z_dim)`), V comes from
  the timestep-normalized input (not the CEMA output — reference line 331:
  `v = F.silu(self.wv(out_tsn))`).

### Causal attention and Megalodon-chunk

Full causal masking gives O(n²). The paper's efficiency claim comes from
chunking (c = 128 for LRA, 4096 for the 7B model). We support both via a
`chunk_size` parameter: `0` = full causal; `chunk_size > 0` = attention
restricted to `(same chunk) ∧ (s ≤ t)`, which is exactly `Megalodon-chunk`.
Same masked softmax code either way; only the mask predicate changes.

### v1 simplifications (documented, not silently dropped)

- No RoPE. The reference applies rotary embeddings inside the inner
  attention; the paper lists RoPE as a Llama-2-derived hyperparameter choice
  rather than a Megalodon contribution, and adding it would need a separate
  FD-checked backward chain. Documented as a known omission.
- No chunk-parallel / distributed plumbing (§3.5) — inference-only concern.
- No dropout anywhere (repo-wide convention for sequence layers).
- No KV cache / incremental decoding.
- `η` is a free complex parameter rather than constrained to unit modulus.

---

## Tasks

Each task is a TDD cycle: failing test → run it → minimal impl → run it.

### Task 1: `TimestepNorm` header + constructor + forward

- Create `include/nn/layers/normalization/timestep_norm.h`
- Test file `tests/test_timestep_norm.cpp`:
  - throws on `features == 0`, `eps <= 0`
  - accessors: `features()`, `num_groups()`, `eps()`, `name()`
  - forward shape `(n, d) -> (n, d)`, finite
  - hand-derived `d=2, k=1`: at `t=0`, `μ=0, σ²=0` so `y = γ·x/√eps + β`
  - causal property: perturbing `x[t]` for `t > s` leaves `y[s]` bit-exact
- Register in `include/nn/nn.h` + Makefile (`build/test_timestep_norm` rule,
  `=== Running TimestepNorm Tests ===` echo, add to `tests:` target)

### Task 2: `TimestepNorm` backward + FD checks

- Hand-derived backward for `y = (γ+1)·x̂ + β` where `x̂ = (x-μ)/σ`, with
  `μ` and `σ²` themselves functions of `x_{1..t}`:
  - `dβ_j = Σ_t g_{t,j}`, `dγ_j = Σ_t g_{t,j}·x̂_{t,j}`
  - per row: `dx̂_{t,j} = g_{t,j}·(γ_j+1)`
  - `dμ_t = -Σ_j dx̂_{t,j}/σ_t`, `dσ_t = -Σ_j dx̂_{t,j}·(x_{t,j}-μ_t)/σ_t³`
  - then the cumulative-statistics chain:
    `∂μ_t/∂x_{i,j} = [1_{i≤t}]/(t·d_g)`,
    `∂σ²_t/∂x_{i,j} = (1_{i≤t})/(t·d_g)·[2(x_{t,j}-μ_t) + (σ²_{t-1} - σ²_t)·(t-1)/t]`
  - i.e. `dx_{i,j} += dμ_t·[1_{i≤t}]/(t·d_g) + dσ²_t·(∂σ²_t/∂x_{i,j})`
- FD tests: input gradient, `γ` gradient, `β` gradient at `n=4, d=4, k=2`
- FD with `k=1` (single group, the more common setting) at `n=5`

### Task 3: `ComplexEMA` header + constructor + forward

- Create `include/nn/layers/recurrent/complex_ema.h`
- Test file `tests/test_complex_ema.cpp`:
  - throws on `d_model == 0`, `ndim == 0`
  - accessors `d_model()`, `ndim()`, `name()`
  - parameter/gradient contract: 6 tensors
  - forward shape `(n, d) -> (n, d)`, finite
  - **decay property** `|q_{j,k}| < 1` for all `j,k` (the paper's key claim)
  - hand-derived `n=1, d=1, h=1`: `h_0 = p·x_0`, `y_0 = Re(η·p·x_0) + ω·x_0`
  - **zero state at t=0**: with all `ω=0` and only `η_re` non-zero, `y_0` is
    exactly `p·x_0·η_re` — proves the recurrence starts from `h_{-1}=0`
  - determinism: two consecutive forwards bit-exact

### Task 4: `ComplexEMA` backward + FD checks

- Hand-derived complex-linear backward for
  `h_t = p·x_t + q·h_{t-1}`, `y = Re(Σ η_k h_k) + ω·x`
  (treat each real/imag component as a real scalar with the 2×2 rotation
  matrix implied by `q`; or use the Wirtinger-free chain directly)
- FP grad: input, `alpha`, `delta`, `theta`, `eta_re`, `eta_im`, `omega`
- All 7 FD checks at `n=3, d=3, h=4`

### Task 5: `MegalodonBlock` header + constructor + forward

- Create `include/nn/layers/architectures/megalodon.h`
- Test file `tests/test_megalodon.cpp`:
  - throws on `d_model == 0`, `num_heads == 0`, non-divisible `z_dim`/`h_dim`,
    `num_groups < 0`, `cema_ndim == 0`
  - accessors
  - forward shape `(n, d) -> (n, d)`, finite, nonzero
  - **two-hop residual signature** (the paper's §3.4 claim): zeroing the
    FFN changes the output by *less* than zeroing the attention branch,
    because the FFN residual is `X` not `Ŷ`. Better: assert directly that
    `Y - FFN_output == X` by zeroing the FFN weights → `Y == X` exactly.
  - **Megalodon-chunk signature** with `chunk_size = 2, n = 4`: changing
    `x[3]` does not change `y[0..1]`
  - determinism

### Task 6: `MegalodonBlock` backward + FD checks

- Block backward composed from: SwiGLU FFN → LayerNorm → residual `+X`
  (two-hop — `dX` gets contributions from BOTH the attention residual and the
  FFN residual) → attention → V/Wr → Wz → RMSNorm → CEMA → TimestepNorm.
- **Critical trap:** the two-hop residual means `grad_x` accumulates from two
  paths (`Ŷ = Attn(Norm(X)) + X` and `Y = FFN(Norm(Ŷ)) + X`). Forgetting the
  second gives a clean ~2× error on a gradient that FD will catch.
- FD: block input gradient, and representative parameter gradients
  (`W_z.weights`, `W_v.weights`, `gamma_qk`, CEMA `alpha`, `gamma`/`beta`
  of `TimestepNorm`)

### Task 7: `MegalodonModel` + training smoke test

- Stack of `num_layers` blocks + input projection + final `TimestepNorm` +
  classifier.
- End-to-end: train on a toy regression for N SGD steps, assert loss drops
  by > 10% (the repo's standard smoke test).

### Task 8: mutation testing + final verification

- Mutate the CEMA `|q| = 1 - α·δ` term (drop the `α·δ`) → FD checks on
  `alpha`/`delta` must fail.
- Mutate the two-hop residual to a one-hop (`Y = FFN(Norm(Ŷ)) + Ŷ`) → FD on
  block input gradient must fail.
- Run `make tests` and confirm no regressions; run the focused suite 3× for
  determinism.

---

## References

- `include/nn/layers/architectures/mega.{h,cpp}` — the predecessor
  architecture; same file layout and test conventions apply.
- `include/nn/layers/normalization/{layer_norm,rms_norm}.h` — the
  normalization classes `TimestepNorm` sits beside.
- `include/nn/layers/utility/swiglu.h` — the FFN used in the block.
- Official implementation: `megalodon/modules/complex_exponential_moving_average.py`,
  `megalodon/modules/moving_average_gated_attention.py` (lines 275–348),
  `megalodon/modules/normalized_feedforward_network.py` (line 75),
  `megalodon/model/mega.py` (line 79).
