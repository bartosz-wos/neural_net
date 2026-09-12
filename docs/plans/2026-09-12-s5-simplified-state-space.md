# S5 (Simplified State Space Layers) Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).
> **Skills loaded:** systematic-debugging (numerical-gradient trap awareness), test-driven-development (mutation-tested non-vacuous checks), writing-plans (this template).

**Goal:** Add `S5Layer` and `S5Block` — Smith et al., ICLR 2022, "Simplified State Space Layers for Sequence Modeling" (https://arxiv.org/abs/2208.04933). A diagonal complex-valued MIMO SSM with HiPPO-LegS initialization and bilinear (Tustin) discretization. The fundamental building block later used by H3, Mamba, and many other modern SSMs.

**Paper:** Smith, Warrington, Linderman, Gu — "Simplified State Space Layers for Sequence Modeling" (ICLR 2022).

**Architecture:** Two classes in `include/nn/layers/recurrent/s5.{h,cpp}`:

1. `S5Layer(d_model, d_state, bidirectional=true)` — the multi-channel diagonal-complex SSM with HiPPO-LegS init.
2. `S5Block(d_model, d_state, bidirectional=true)` — input Dense → S5Layer → output Dense + residual gating.

**Tech Stack:** C++17, repo's `Tensor`, `Layer`, `Dense`. Uses `<complex>` for SSM eigenvalues.

---

## The math (paper §2, paraphrased)

**Continuous-time SSM:**
```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```
with `h ∈ ℂ^N`, `A ∈ ℂ^{N×N}`, `B ∈ ℂ^{N}`, `C ∈ ℂ^{N}`.

**S5 simplification (§2.2):** A is the diagonalization of the HiPPO-LegS matrix, giving `A = diag(λ_1, ..., λ_N) ∈ ℂ^{N×N}`. With diagonal A, the SSM is a bank of N independent scalar SSMs sharing the same input — a "multi-input" / diagonal system.

**HiPPO-LegS eigenvalues** (closed-form, paper §2.2.1 / Appendix):
```
λ_n = − (2n + 1)^0.5 · (2n + 2)^0.5,  for n = 0, 1, ..., N-1
   = − sqrt((2n+1)(2n+2))   ∈ ℝ_{<0}
```
These are REAL and negative. S5 uses them directly (no need to diagonalize a real matrix into complex pairs, because LegS is already diagonal in the orthogonal Legendre basis).

**B (HiPPO-LegS):** paper §2.2.1:
```
B_n = (2n + 1)^0.5  ∈ ℝ_{>0}
```

**C:** random Gaussian init (`std::complex<double>` with real and imag parts).

**D:** learnable scalar skip (paper uses 1.0 default).

**Bilinear (Tustin) discretization** with step size Δ:
```
Ā  = (I + Δ/2 · A) · (I − Δ/2 · A)^{-1}
B̄  = (I − Δ/2 · A)^{-1} · B
```

For diagonal A, this is a per-channel scalar operation — no matrix inverse.

**Discrete recurrence (paper §2.3, eq. 5):**
```
h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t       (element-wise on the diagonal)
y_t = Re( C^H h_t ) + D x_t        (paper eq. 6 — real part only)
```
`⊙` denotes element-wise multiply. `Re(C^H h_t)` is a complex inner product, taken real.

**Multi-channel wrap (our contribution):** the SSM is single-input single-output. To process `(T, d_model)`, we run `d_model` independent SSMs in parallel — each with its OWN `(λ, B, C, D)` — and let the input/output Dense projections handle cross-channel mixing. This is the standard "depthwise" SSM pattern (used by S4D and Mamba).

**Bidirectional extension (paper §3.1):** two SSMs, one forward, one backward. Forward processes `x_0, x_1, ..., x_{T-1}`. Backward processes `x_{T-1}, ..., x_0`. Outputs are mixed via a learnable per-channel scalar `Λ_gate[g] ∈ (0, 1)`:
```
y_t = Λ_gate · y_t^{fwd} + (1 − Λ_gate) · y_t^{bwd}
```

---

## Backward derivation (hand-derived, plain form)

For each SSM channel, given `d_y_t = ∂L/∂y_t` (real scalar):

1.  **Discrete state backward sweep** (reverse recurrence):
    ```
    dh_t = conj(C) · d_y_t
    dB̄_t = dh_t · x_t                (per-channel contribution to B̄)
    dĀ_t = dh_t · conj(h_{t-1})
    For s = t-1, t-2, ..., 0:
        dh_s += Ā · dh_{s+1}
    ```
    We accumulate into `d_h_{t-1}` via the propagated `Ā · dh_{s+1}`.
    After the sweep, we have `dh_t = d_y_t · conj(C)` for every t, plus propagated gradients into earlier `dh_s`.

2.  **dB̄** = Σ_t dh_t (over t) — diagonal accumulator.
    **dĀ** = Σ_t dh_t · conj(h_{t-1}) — diagonal accumulator.

3.  **Tustin discretization inverse** (per channel, since A is diagonal):
    For each channel `n`:
    ```
    Let A_n = λ_n (complex), B_n (complex).
    Let I_Δ = 1 − Δ/2 · A_n
    Let plus_Δ = 1 + Δ/2 · A_n

    dI_Δ  += dĀ_n · plus_Δ · I_Δ^{-1}    (chain through Ā)
    dplus_Δ += dĀ_n · I_Δ
    dA_n += dI_Δ · (−Δ/2) + dplus_Δ · (+Δ/2)
    dB_n += dB̄_n · I_Δ
    dA_n += dB̄_n · B_n · I_Δ · (−Δ/2) · I_Δ^{-1}      (chain through B̄)
    ```
    Simplified per channel (since A_n is scalar):
    ```
    Ā_n = (1 + Δ/2 · A_n) / (1 − Δ/2 · A_n)
    B̄_n = B_n / (1 − Δ/2 · A_n)

    dB̄_n/dA_n = B_n · (Δ/2) · (1 − Δ/2 · A_n)^{-2}
    dĀ_n/dA_n = (Δ/2) · I_Δ^{-1} + (Δ/2) · plus_Δ · I_Δ^{-2}
              = (Δ/2) · I_Δ^{-2} · (I_Δ + plus_Δ)
    ```
    so
    ```
    dA_n = dĀ_n · (Δ/2) · I_Δ^{-2} · (I_Δ + plus_Δ) + dB̄_n · B_n · (Δ/2) · I_Δ^{-2}
    ```

4.  **dC**: `dC_n = Σ_t conj(h_t) · d_y_t`.

5.  **dD**: `dD = Σ_t d_y_t · x_t` (scalar).

6.  **d_input (SSM-level)**: per channel, `d_x_t = Re(conj(B̄_n) · dh_t)`. Summed across channels.

7.  **Bidirectional**: gradients flow separately through each direction; the gate's gradient is `dΛ_gate = Σ_t d_y_t · (y_t^{fwd} − y_t^{bwd}) · σ'(Λ_gate_log)`.

8.  **Dense projections** wrap this with standard Dense backward.

---

## File layout

- Header: `include/nn/layers/recurrent/s5.h`
- Implementation: `include/nn/layers/recurrent/s5.cpp`
- Test: `tests/test_s5.cpp`
- Plan: this file

## Test plan

8 focused checks (minimum viable):

1. **Constructor validation**: `d_model=0` throws, `d_state=0` throws, `d_state=2` valid (smallest), `bidirectional=true` and `=false` both construct.
2. **Forward shape & finite**: input `(T=8, d_model=4)` → output `(T=8, d_model=4)`, finite, nonzero.
3. **HiPPO-LegS eigenvalues**: first channel λ_0 = `-sqrt(1·2) = −√2 ≈ −1.4142`, λ_1 = `-sqrt(3·4) = −√12 ≈ −3.4641`. Verify with `rel_err < 1e-12`.
4. **Hand-derived forward reference** (T=1, d_model=1, no bidirectional):
   - `Δ = 0.1, A = -√2, B = 1, C = 1, D = 0, x_0 = 1`
   - `I_Δ = 1 - 0.05·(-√2) = 1 + 0.05√2`
   - `Ā = (1 - 0.05·(-√2)) / (1 + 0.05√2) ... wait, careful with signs`
   - `Ā_0 = (1 + 0.05√2) / (1 - 0.05√2)`, `B̄_0 = 1 / (1 - 0.05√2)`
   - `h_0 = B̄_0 · 1 = 1/(1 - 0.05√2)`
   - `y_0 = Re(C · h_0) = 1/(1 - 0.05√2)`
5. **FD gradient check on input** with random init, d_model=4, d_state=4, T=4, bidirectional=false: `rel_err < 1e-5`.
6. **FD parameter gradients** (Ā, B̄, C, Λ_gate if bidirectional) — all at `rel_err < 1e-5`.
7. **Bidirectional equivalence at Λ_gate = 0.5**: unidirectional-forward output == bidirectional output · 0.5 (modulo the backward SSM contribution, which differs); with Λ_gate_log → +∞ (pure forward), bidirectional output ≈ unidirectional forward output.
8. **End-to-end training** on synthetic regression (y = sin(x) signal): S5Block with 5 channels, MSE loss decreases over 30 SGD steps at lr=0.01.

**Non-vacuous mutation tests:**
- Stub out the `Ā ⊙ h_{t-1}` term (drop the recurrence) — gradient check must fail.
- Stub out `Re(...)` (return C·h instead of Re(C·h)) — gradient check must fail.
- Skip the backward SSM contribution (use forward only) when bidirectional=true — Test 7 fails.

---

## Step-by-step tasks

### Task 1: Header skeleton + constructor + eigenvalue init

**Files:**
- Create: `include/nn/layers/recurrent/s5.h` (skeleton with class declarations + HiPPO-LegS init in constructor)
- Create: `include/nn/layers/recurrent/s5.cpp` (constructor only, forward stub that throws)
- Create: `tests/test_s5.cpp` (Test 1 — constructor validation; Test 3 — HiPPO eigenvalues)

**Step 1: Write the failing tests.**

```cpp
// tests/test_s5.cpp
#include "nn/layers/recurrent/s5.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nn;

int main() {
    // Test 1: constructor validation
    bool threw_dm0 = false, threw_ds0 = false;
    try { S5Layer l(0, 4); } catch (std::invalid_argument&) { threw_dm0 = true; }
    try { S5Layer l(4, 0); } catch (std::invalid_argument&) { threw_ds0 = true; }
    assert(threw_dm0 && threw_ds0);

    // Test 3: HiPPO eigenvalues
    S5Layer l(4, 4);
    auto A = l.A_values(); // (d_state,) complex
    double expect0 = -std::sqrt(1.0 * 2.0);
    double expect1 = -std::sqrt(3.0 * 4.0);
    // ... verify with rel_err < 1e-12

    std::cout << "PASS" << std::endl;
    return 0;
}
```

**Step 2: Run tests to verify failure.**

`g++ -std=c++17 -Iinclude tests/test_s5.cpp -o build/test_s5` — should fail to compile because `S5Layer` doesn't exist.

**Step 3: Write header + minimal class.**

Header defines `S5Layer` with:
- Constructor `S5Layer(size_t d_model, size_t d_state, bool bidirectional = true)`
- Member tensors: `A_diag` (complex), `B_diag` (complex), `C_diag` (complex), `D_skip` (double)
- HiPPO-LegS init in constructor body
- `Tensor A_values()`, `Tensor B_values()`, `Tensor C_values()` accessors returning real/imag pairs as a 2×P real tensor for tests
- `forward`, `backward`, `update_weights`, `zero_grad`, `parameters`, `gradients` declarations (stubs throwing)

**Step 4: Run tests to verify pass.** Both constructor and eigenvalue tests pass.

**Step 5: Commit.**

```bash
git add include/nn/layers/recurrent/s5.h include/nn/layers/recurrent/s5.cpp tests/test_s5.cpp
git commit -m "feat(recurrent): S5 layer skeleton + HiPPO-LegS init"
```

### Task 2: Forward pass (unidirectional) + discretization

**Files:**
- Modify: `include/nn/layers/recurrent/s5.cpp` — implement forward() for the unidirectional case.

**Step 1: Add forward tests.**

```cpp
// Tests 2, 4
S5Layer l(4, 4, /*bidirectional=*/false);
Tensor x(8, 4); /* fill with random or seed-fixed values */
Tensor y = l.forward(x);
assert(y.rows == 8 && y.cols == 4);
// finite + nonzero

// Test 4: hand-derived T=1 reference
S5Layer tiny(1, 1, /*bidir=*/false);
tiny.set_D_skip(0.0); tiny.reset_state();
// Set A = -sqrt(2), B = 1, C = 1 (overriding init for testability)
tiny.override_params(/*A=*/std::complex<double>(-std::sqrt(2.0), 0),
                      /*B=*/std::complex<double>(1.0, 0),
                      /*C=*/std::complex<double>(1.0, 0),
                      /*dt=*/0.1);
Tensor one(1, 1); one(0, 0) = 1.0;
Tensor y_one = tiny.forward(one);
double expect = 1.0 / (1.0 - 0.05 * std::sqrt(2.0)); // h_0 = B̄ · x_0, y_0 = Re(C · h_0)
assert(std::abs(y_one(0, 0) - expect) < 1e-9);
```

**Step 2: Run tests to verify failure.** `forward` throws "not implemented".

**Step 3: Implement forward (unidirectional).** Per-channel discretization, recurrence, complex inner product, real-part output.

**Step 4: Run tests to verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(recurrent): S5 forward (unidirectional) + Tustin discretization"
```

### Task 3: Backward pass + FD verification

**Files:**
- Modify: `include/nn/layers/recurrent/s5.cpp` — implement backward() for unidirectional.
- Modify: `tests/test_s5.cpp` — add FD gradient checks (Tests 5, 6).

**Step 1: Add FD tests.** See `references/ml-layer-backward-bugs.md` (systematic-debugging skill) for non-degenerate init rules.

**Step 2: Run tests to verify failure.**

**Step 3: Implement backward.** Reverse sweep, complex conjugates, Tustin chain rule.

**Step 4: Run tests to verify pass.** All FD checks at `rel_err < 1e-5`.

**Step 5: Commit.**

```bash
git commit -am "feat(recurrent): S5 backward + FD verification (unidirectional)"
```

### Task 4: Bidirectional extension

**Files:**
- Modify: `include/nn/layers/recurrent/s5.cpp` — add bidirectional path + gate.

**Step 1: Add bidirectional tests (Test 7).**

**Step 2: Run tests to verify failure.**

**Step 3: Implement bidirectional.** Reverse the input, run unidirectional SSM, reverse the output back. Mix with gate.

**Step 4: Run tests to verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(recurrent): S5 bidirectional + learnable per-channel gate"
```

### Task 5: S5Block wrapper + end-to-end training

**Files:**
- Modify: `include/nn/layers/recurrent/s5.{h,cpp}` — add `S5Block` class.
- Modify: `tests/test_s5.cpp` — add Test 8 (end-to-end training).

**Step 1: Add S5Block test.**

**Step 2: Run tests to verify failure.**

**Step 3: Implement S5Block:** input Dense → S5Layer → output Dense + residual gating.

**Step 4: Run tests to verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(recurrent): S5Block wrapper + end-to-end training"
```

### Task 6: Register in umbrella + Makefile

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/recurrent/s5.h"`.
- Modify: `Makefile` — add `$(BUILD_DIR)/test_s5` build rule, add to `tests:` deps, add `=== Running S5 Tests ===` echo in `run_tests`.

**Step 1: Register.**

**Step 2: Verify umbrella compiles standalone.**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

**Step 3: Run full S5 test.**

```bash
make tests && ./build/test_s5
```

**Step 4: Commit + push.**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register S5 in umbrella + Makefile"
git push origin master
```

---

## Decisions / gotchas

- **HiPPO-LegS eigenvalues are REAL** (not complex). S5 uses them directly. We don't need complex pairs unless we want to diagonalize a non-diagonal A. Keeping them real avoids unnecessary complexity, but C is still complex (initialized random).
- **Bidirectional SSM** shares parameters between fwd/bwd (no separate B̄_fwd vs B̄_bwd). The gate mixes the two output streams. This matches paper §3.1.
- **No D_skip in S5Layer** — D is exposed as `set_D_skip(double)` for testability but defaults to 1.0 (paper convention).
- **Δ (step size)** is a fixed hyperparameter (paper sets Δ = sqrt(d_state / (d_model * T)) typically; we use 0.1 as a simple default). Learnable Δ is a future extension.
- **Complex in Tensor**: Tensor is real-valued, so we store complex A/B/C as TWO Tensors (`*_re`, `*_im`) of shape (P,) and do manual complex arithmetic in the SSM. This keeps the interface consistent with the rest of the repo.
- **Multi-channel:** S5Layer has `d_model` independent SSMs sharing the same parameters (B/C/D are broadcast across channels via Dense projections before/after the SSM).