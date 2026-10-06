# Spiking LIF + Surrogate Gradient Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the spiking-neural-network paradigm for this repo — leaky integrate-and-fire (LIF) neurons with a **hard** binary spike forward pass and a **surrogate gradient** backward pass, plus rate-encoding helpers and a feedforward SNN model.

**Architecture:** New file `include/nn/layers/recurrent/spiking_lif.{h,cpp}` with three classes: `SurrogateSpike` (a stateless polynomial helper: value + surrogate derivative), `LIFNeuron` (one layer of LIF neurons over a `(T, d)` sequence, with optional recurrent `V` connections), and `SpikingNet` (input encoder → N LIF layers → leaky rate readout). Registration in `include/nn/nn.h` + a `tests/test_spiking_lif.cpp` + Makefile wiring.

**Tech Stack:** C++17, hand-rolled `Tensor` (2-D real, row-major), `Layer` interface, `Dense`, naive O(T²) BPTT (repo convention).

---

## Source of truth (verified from the PDF, not from memory)

Neftci, Gutierrez, Bohte, "Surrogate Gradient Learning in Spiking Neural Networks", **arXiv:1901.09948v2** — Box 1 and Eq. 4–5, read directly out of the downloaded PDF text.

**Discrete-time LIF dynamics (Eq. 4–5), with the survey's stated normalizations `U_rest = 0`, `R = 1`, `ϑ = 1`:**

```
I_t = α · I_{t-1} + X_t · W^T                       (Eq. 4, synaptic current)
U_t = β · U_{t-1} + I_t − S_{t-1} · ϑ               (Eq. 5, membrane; hard reset to U_rest=0)
S_t = Θ(U_t − ϑ)                                    (hard Heaviside, output spikes in {0,1})
α  = exp(−Δt / τ_syn)     with 0 < α < 1
β  = exp(−Δt / τ_mem)     with 0 < β < 1
```

Two details the plan pins because each is a natural mistake:

1. **The reset term uses `S_{t-1}`, not `S_t`.** Eq. 5 literally carries `−S_i[n] · ϑ` while the spike is emitted as `S_i[n] = Θ(U_i[n] − ϑ)` — so in a single-array implementation the emitted spike of step `t` is *subtracted from the membrane of step `t+1`*. A "reset immediately after firing" implementation would use `−S_t·ϑ` and produce a **shifted** membrane trace. Both are self-consistent; only this one matches the cited equations.
2. **`Θ(0) = 1` is used**, i.e. `S = (U >= ϑ)`. The Heaviside at exactly 0 is a measure-zero detail of the forward pass, but it matters for the FD harness (see Task 5) — a strict `>` vs `>=` flips the sign of the whole spike train for degenerate fixtures.

**Surrogate gradient (§V-B, Fig. 4):** the true `dΘ/dU` is zero everywhere except at `U = ϑ`. Training replaces it by the derivative of a smooth function. The survey names four families: piece-wise linear / ReLU-like (Bohte 2018), **fast sigmoid** (Zenke & Ganguli 2018 — its ref [2], the "three-factor online learning rule"), exponential (Shrestha & Orchard 2018), and SLAYER. This plan implements the **fast-sigmoid (SuperSpike)** family, which the survey attributes to Zenke & Ganguli:

```
forward:  S_t = Θ(U_t − ϑ)          (HARD, unchanged — the SG never touches the forward pass)
backward: ∂S/∂U ≈ 1 / (γ · |U_t − ϑ| + 1)
```

The survey's Fig. 3 plots it and explicitly labels it "Derivative of a fast sigmoid". The paper's stated virtue is that the surrogate gradient is **non-zero, continuous and finite**, so `BPTT` keeps working where the true gradient is a plateau. `γ` is a learnable sharpness (default `10.0`; larger = narrower, closer to a true derivative).

**Why not Steer / arctan / Sigmoid?** YAGNI. The fast sigmoid is the paper's own named choice and gives a clean closed form with no `exp` overflow concerns at `|U−ϑ| → ∞`.

### v1 scope decisions

- **`U_rest = 0`, `R = 1` fixed** (the survey's "without loss of generality"). A non-zero rest potential would only add a constant shift that the bias absorbs.
- **`ϑ` is a constructor scalar (default 1.0), not learnable.** Keeps the spike set meaningful and the FD harness deterministic. The `γ` sharpness IS learnable.
- **Recurrent connections are supported** via an optional `V` matrix (`use_recurrence` flag), matching Eq. 2's `V` term. Default off (feedforward) so the plain FD path is unambiguous.
- **Multi-layer depth and the rate readout** live in `SpikingNet`, not in `LIFNeuron`. One layer = one neuron population = one `Layer` (repo convention; mirrors `LTC`).
- **No adaptive thresholds, no STDP, no conversion-coding-to-ANN.** Out of scope for v1.

## Conventions

Sequences are `(T, d)` row-major `Tensor`s (repo-wide). Within a layer:
- `X: (T, input_dim)`, `I, U, S: (T, hidden_size)`
- `W: (hidden_size, input_dim)`, `b: (1, hidden_size)`, `V: (hidden_size, hidden_size)`

Because `Repo Tensor` is 2-D real and there is no complex type, all per-time-step per-neuron quantities are `(T, hidden)` rows. `S` is stored as a float tensor holding **exactly 0.0 or 1.0** (the binary spike train), not as indices — the `(T, hidden)` layout is what lets the backward loops be dense and matches every other recurrent layer here.

### Backward derivation (hand-derived, written out so the implementer does not re-derive)

Per time step `t`, with `g_t = ∂L/∂S_t` (surrogate-gradient carriers) *premultiplied* by the surrogate derivative:

```
dU_t/∂U_{t-1} = β                                        (through the leak)
dU_t/∂I_t     = 1
dI_t/∂I_{t-1} = α
```

With **surrogate substitution** `∂S_t/∂U_t ≈ 1/(γ|U_t − ϑ| + 1)`, the incoming gradient at the spike
`g_t = ∂L/∂S_t` first becomes `∂L/∂U_t = g_t · σ'(U_t − ϑ)`.

Then (all with `ϑ = U_rest = 0` for reset, so reset is `U_{t+1} = β·U_t + I_{t+1} − ϑ·S_t`):

```
∂L/∂I_t       = ∂L/∂U_t                                 (direct)
∂L/∂I_{t-1}   += α · ∂L/∂I_t                            (Eq. 4 carry)
∂L/∂U_{t-1}   += β · ∂L/∂U_t  +  α · ∂L/∂I_t           (Eq. 5 carry + current carry)

grad_W[i,k]  += (∂L/∂I_t[i]) · X_t[k]
grad_b[i]    += (∂L/∂I_t[i])
grad_V[i,j]  += (∂L/∂I_t[i]) · S_{t-1}[j]               (if use_recurrence)
∂L/∂X_t[k]   += Σ_i (∂L/∂I_t[i]) · W[i,k]
grad_γ       += Σ_{t,i} (∂L/∂S_t[i]) · d/dγ [1/(γ·|U_t−ϑ|+1)]
                       = Σ_{t,i} (∂L/∂S_t[i]) · (−|U_t−ϑ|) / (γ·|U_t−ϑ|+1)²
```

**Critical:** the `-S_{t-1}·ϑ` reset term is a **constant** with respect to every parameter and to `U`, so it contributes **nothing** to any gradient. It changes the forward trace only. This is easy to get wrong by writing a `+ϑ·S_{t-1}` term into `∂L/∂U_{t-1}`; the tests pin that this term is absent.

## Tasks

### Task 1: Write the RED test file with hand-derived references

Create `tests/test_spiking_lif.cpp` with the harness + tests 1–5. Before writing, hand-derive (on paper, then confirm with the debug print) the single-neuron T=3 trace used in Test 4, with `d=1`, `input_dim=1`, `W = [[1.0]]`, `b = [[0.0]]`, `α = 0.5`, `β = 0.5`, `ϑ = 1.0`, no recurrence, and `X = [[1.0],[0.0],[1.0]]`.

Hand computation (verify each line against the impl's debug output before asserting):
```
t=0: I_0 = 0.5·0 + 1·1 + 0 = 1.0 ; U_0 = 0.5·0 + 1.0 − 0 = 1.0 ; S_0 = 1 (1.0 >= 1.0)
t=1: I_1 = 0.5·1.0 + 0 + 0 = 0.5 ; U_1 = 0.5·1.0 + 0.5 − 1·1.0 = 0.0 ; S_1 = 0
t=2: I_2 = 0.5·0.5 + 1 + 0 = 1.25 ; U_2 = 0.5·0.0 + 1.25 − 1·0 = 1.25 ; S_2 = 1
```
If the impl disagrees with this, the impl is wrong — do not "fix" the test.

Tests in this task:
1. **Constructor validation** — `input_dim == 0` throws; `hidden_size == 0` throws; `beta <= 0` throws; `beta >= 1` throws; `alpha <= 0` throws; `alpha >= 1` throws; `threshold <= 0` throws; `gamma <= 0` throws; valid constructs; `name()` returns `"LIFNeuron"` / `"SpikingNet"`.
2. **Forward shape** — `(T=5, input_dim=3)` → `(5, hidden)`; SpikingNet `(6,3)` → `(6, out)`.
3. **Forward binary spike contract** — every entry of the cached `S` is exactly `0.0` or `1.0` (max deviation from the set {0,1} < 1e-12). This is THE SNN property; it must not be soft.
4. **Hand-derived single-neuron trace** — the T=3 trace above, `|diff| < 1e-12` on `I`, `U`, and `S`.
5. **`Θ(0) = 1` boundary** — set `U` so the pre-spike value lands exactly on `ϑ`; assert the spike fires.

Run: `make build/test_spiking_lif 2>&1 | tail -20` → expect **compile failure** (classes don't exist). That is the RED for the whole file's skeleton.

Commit: `test(recurrent): SpikingLIF RED test file — constructor, shape, binary spikes, hand-derived trace`.

### Task 2: `SurrogateSpike` + `LIFNeuron` forward

Files: create `include/nn/layers/recurrent/spiking_lif.h`, `include/nn/layers/recurrent/spiking_lif.cpp`.

`SurrogateSpike` — stateless helper:
```cpp
struct SurrogateSpike {
    // Forward: hard Heaviside. THETA(0) = 1.
    static double spike(double u, double threshold);
    // Surrogate derivative of the fast-sigmoid SG family:
    //   1 / (gamma * |u - threshold| + 1)
    static double surrogate_derivative(double u, double threshold, double gamma);
    // d/dgamma of the above (needed for the learnable sharpness).
    static double surrogate_derivative_dgamma(double u, double threshold, double gamma);
};
```

`LIFNeuron(size_t input_dim, size_t hidden_size, double alpha=0.5, double beta=0.5, double threshold=1.0, double gamma=10.0, bool use_recurrence=false)`.

Parameters (4 tensors when `use_recurrence`, else 3 + the scalar):
- `W_ (hidden, input_dim)`, `b_ (1, hidden)`, `gamma_ (1,1)` (learnable, init `gamma`)
- `V_ (hidden, hidden)` only when `use_recurrence`

Forward, per `t` from `0..T-1`, following Eq. 4–5 exactly (note `S_{t-1}` on the reset):
```
I_t[i] = alpha * I_{t-1}[i] + Σ_k W[i,k]·X[t,k] + b[i]   (+ Σ_j V[i,j]·S_{t-1}[j] if recurrent)
U_t[i] = beta  * U_{t-1}[i] + I_t[i] − threshold * S_{t-1}[i]
S_t[i] = (U_t[i] >= threshold) ? 1.0 : 0.0
```
Cache `I`, `U`, `S`, and `X`. `I_{-1} = U_{-1} = S_{-1} = 0`.

Run: `make build/test_spiking_lif` → tests 1–5 pass. Commit: `feat(recurrent): LIFNeuron forward — Eq.4-5 synaptic current + membrane, hard spike`.

### Task 3: `LIFNeuron` backward

Hand-derived per the Backward derivation block above. Walk `t` from `T-1` down to `0` carrying `dI_next` and `dU_next`:

```
dU_t[i] = g_t[i] * surrogate_derivative(U_t[i], threshold, gamma)      // SG substitution
dI_t[i] = dU_t[i]
dI_prev[i] = alpha * dI_t[i]
dU_prev[i] = beta * dU_t[i] + alpha * dI_t[i]                          // NO reset term — it is constant
grad_W[i,k] += dI_t[i]·X[t,k];  grad_b[i] += dI_t[i]
grad_V[i,j] += dI_t[i]·S[t-1][j]        (if recurrent)
dX[t,k]    += Σ_i dI_t[i]·W[i,k]
grad_gamma  += Σ_i g_t[i] · d/dγ surrogate_derivative(...)
```
where `g_t = grad_output[t]`.

`update_weights(lr)` does a plain in-place SGD step on all params. `zero_grad()`, `parameters()`, `gradients()` per convention, and `last_spikes() / last_membrane() / last_current() / last_alpha() / last_beta() / last_gamma()` accessors for the tests.

Add Test 6 to the test file: **FD input gradient**, `rel_err < 1e-4`, **random non-uniform init** (`Tensor::random(0.3)`, NOT uniform — uniform init makes `Σ_k W[i,k] == Σ_k W[k,i]` and would mask a transposed-gradient bug). Add Test 7: FD on `grad_W`, `grad_b`, `grad_gamma`.

Run: `make build/test_spiking_lif`. If a rel_err is a *clean* constant factor (0.5 or 2.0), suspect the test's loss scaling before the impl. Commit: `feat(recurrent): LIFNeuron backward — BPTT with fast-sigmoid surrogate`.

### Task 4: `SpikingNet` (encoder → stacked LIF → leaky rate readout)

`SpikingNet(size_t input_dim, size_t hidden_size, size_t output_dim, size_t num_layers=2, ...)`.

- Input encoding: **rate coding** — `EncodeRate(input, T, dt, gain)` maps a real `(B, input_dim)` into `(T, input_dim)` by thresholding at `gain·(i-th quantile of |x|)`, emitting spikes. Expose `last_encoded_spikes()`. This is a documented v1 simplification: a Bernoulli/rate encoder is standard and the tests only need it to be binary and deterministic.
- Stack `num_layers` `LIFNeuron`s, feeding `S` of layer `l` as the `X` of layer `l+1` (spike trains as inputs — the native SNN wiring, Fig. 2's `W^(2)` edge).
- Readout: **mean of spikes over time** (rate decode) → a `Dense`. Non-spiking because the loss is defined on it — the standard "rate decoder" of SNN classifiers.

Tests (8–11): forward shape; binary spikes at every layer (`SpikingNet` must expose the per-layer spike tensors); encoder output is binary; training reduces loss over N SGD steps.

Commit: `feat(recurrent): SpikingNet — rate encoder, stacked LIF layers, rate readout`.

### Task 5: Recurrence + edge cases

Tests (12–16):
- `use_recurrence=true` forward shape + FD input gradient (this exercises the `V` path and the `S_{t-1}` shift — the two things most likely to be wrong).
- `T=1` (single time step, so `S_{t-1}` is always 0) — finite, and FD-checkable.
- Sub-threshold input (never spikes) → `S` all zero, output finite.
- **Surrogate gradient is non-zero where the true gradient is zero** — the whole point of the SG. Hand-computed, not FD.
- **FD under the `S_{t-1}` convention**: `γ` gradient. Note the FD harness perturbs a parameter and re-runs forward; because the forward is **deterministic and hard**, no noise enters and central differences are valid. If a rel_err is large, check whether the fixture accidentally landed on a threshold tie (a tie makes the finite difference measure a discontinuity — use a fixture with membrane values far from `ϑ`).

Commit: `test(recurrent): LIFNeuron recurrence + edge cases`.

### Task 6: Mutation testing — prove the tests are non-vacuous

Apply each mutation in turn, rebuild, run, **confirm ≥1 test fails**, then revert:

1. **Drop `β` from the membrane** (`U_t = I_t − ...`) → the FD input-grad and `W` tests must fail.
2. **Drop `α` from the synaptic current** → FD tests must fail.
3. **Drop the surrogate derivative** (use `1.0` instead) → `grad_gamma` and FD input-grad must fail. This is the mutation that proves the SG is actually *tested* rather than present-but-inert.
4. **Reset on the current step** (`−threshold·S_t` instead of `S_{t-1}`) → the hand-derived T=3 trace (Test 4) must fail. This is the mutation that pins the Eq. 5 convention.
5. **Soft spike** (`S = sigmoid((U−ϑ)·γ)` instead of hard `Θ`) → the binary-spike contract (Test 3) must fail.

Record the exact failure counts in the `## Done` entry. Commit: `test(recurrent): SpikingLIF mutation testing`.

### Task 7: Integration — umbrella header, Makefile, full suite

1. `#include "layers/recurrent/spiking_lif.h"` in `include/nn/nn.h` (recurrent block, after `ltc.h`).
2. **Compile the umbrella header standalone** (catches cross-header collisions that focused tests cannot):
   ```bash
   g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
   ```
3. Makefile: `$(BUILD_DIR)/test_spiking_lif: $(LIB_OBJS) $(BUILD_DIR)/test_spiking_lif.o` rule + add to the `tests:` prerequisite list **and** a `run_tests` line. (The 2026-10-04 audit found `test_mixture_of_softmaxes` had a rule and a `run_tests` line but was missing from `tests:` — it silently never built. Check all three.)
4. Build warning-clean: `make build/test_spiking_lif 2>&1 | grep -i 'warning' ` must be empty.
5. Run the focused suite 3× to confirm determinism, then the aggregate: `make run_tests`.

Commit: `chore(build): register SpikingLIF in nn.h + Makefile`.

### Task 8: Documentation

Move the item from `## Ideas` to `## Done` in `EXPANSION_QUEUE.md` with a one-paragraph summary + exact test counts + the mutation results + any bugs found. Append the source citation and the two pinned conventions (`S_{t-1}` reset, `Θ(0)=1`).

Commit: `docs: mark SpikingLIF Done in EXPANSION_QUEUE`.

---

## Risks / things to watch

- **The reset-term convention is the #1 risk.** If the hand-derived Test 4 trace fails, re-read Eq. 5 before touching anything — do not adjust the test to match a wrong impl.
- **`Θ(0)` boundary.** Avoid fixtures where `U_t` lands exactly on `ϑ` unless the test is specifically about that, or the FD will be measuring a discontinuity.
- **Uniform init is forbidden in FD tests** (masks transposed-gradient bugs).
- **Do not "improve" the surrogate to an arctan/steer form** to match other papers — the fast sigmoid is the cited choice and the tests encode it.