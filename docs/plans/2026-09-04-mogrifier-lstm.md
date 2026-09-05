# Mogrifier LSTM Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).
> Always watch the failing test before writing the next chunk.

**Goal:** Add `MogrifierLSTM` (Melis et al. 2020, "Mogrifier LSTM", https://arxiv.org/abs/1909.09592) — a vanilla LSTM whose input `x_t` and previous hidden state `h_{t-1}` are passed through `r` rounds of multiplicative gating *before* the standard gate projections. The paper's central finding is that this small architectural change closes most of the quality gap between LSTM and Transformer language models at small-to-moderate scale.

**Architecture:** For each round `i = 1, …, r`:

```
x_t ← 2 · σ( (h_{t-1} @ Q_i) ) ⊙ x_t     if i is odd
h_{t-1} ← 2 · σ( (x_t @ R_i) ) ⊙ h_{t-1}  if i is even
```

Then the *modified* `x_t` and `h_{t-1}` are concatenated and fed through the standard 4-gate LSTM (`i`, `f`, `g`, `o`) using a `Dense`-style combined projection `W ∈ R^{4·hidden × (input + hidden)}` and bias `b ∈ R^{4·hidden}`. Each round adds two `Dense`-sized projections `(hidden, input)` or `(input, hidden)`.

**Tech stack:** C++17, hand-rolled `Tensor`, `Layer` base class. Pattern follows `include/nn/layers/recurrent/min_lstm.{h,cpp}` (stateful BPTT with caches, separate per-timestep forward and full-sequence `forward_sequence`). The class is `MogrifierLSTM : public Layer`.

---

## Math

Per timestep `t`, with `round ∈ {1, …, r}` (default `r = 2` from the paper; the paper studies `r ∈ {1, 2, 3, 4, 5}` and finds diminishing returns past 2):

```
For round = 1: x_t = 2 σ(h_{t-1} Q_1) ⊙ x_t
For round = 2: h_{t-1} = 2 σ(x_t R_1) ⊙ h_{t-1}
For round = 3: x_t = 2 σ(h_{t-1} Q_2) ⊙ x_t
For round = 4: h_{t-1} = 2 σ(x_t R_2) ⊙ h_{t-1}
…
```

Q and R are round-indexed projections of shape `(hidden, input)` for Q and `(input, hidden)` for R (the convention matches the paper's Eq. 1: `x ← 2σ(Qh) ⊙ x`, so `Q ∈ R^{input × hidden}`; we store Q in standard Dense layout `(hidden, input)` and transpose in the matmul).

After all rounds, the standard LSTM gates on the *mogrified* `[x_t; h_{t-1}]`:

```
[z_i, z_f, z_g, z_o] = [x_t; h_{t-1}] @ W^T + b     ∈ R^{4·hidden}
i = σ(z_i), f = σ(z_f), g = tanh(z_g), o = σ(z_o)
c_t = f ⊙ c_{t-1} + i ⊙ g
h_t = o ⊙ tanh(c_t)
```

(`z_g` is the candidate pre-activation; `g = tanh(z_g)`; no tanh on `i, f, o` since those are sigmoid.)

For `r = 0` the layer reduces to vanilla LSTM (sanity). For `r = 2` (default) the layer matches the paper's "two-round mogrifier" variant.

---

## Backward

The Mogrifier LSTM is a normal LSTM whose *inputs* `x_t` and `h_{t-1}` are mogrified before the gate computation. Backward therefore has two stages:

1. **Standard LSTM backward** through the gates — `grad_x_mog`, `grad_h_mog_prev`, plus parameter grads `dW`, `db`.
2. **Mogrifier backward** — for each round (in reverse), propagate `grad_x` and `grad_h_prev` through the multiplicative gate.

For round `i` (forward `x ← 2σ(Q h_{prev}) ⊙ x`):

```
q_pre = h_prev @ Q^T                (input,)            # raw matmul
g     = 2 σ(q_pre)                  (input,)            # gate value (the "2 ×" is a scale)
x_new = g ⊙ x                       (input,)

# Backward
grad_g = grad_x_new ⊙ x                             (input,)
grad_q_pre = grad_g ⊙ 2 σ'(q_pre) = grad_g ⊙ 2 g(1 − g/2)   ← careful: 2σ, σ'(2σ) = 2 σ (1 − σ)
        # If we treat g = 2σ(p) then dg/dp = 2σ(p)(1−σ(p)) = g(1 − g/2)
        # Easier: dg/dp = 2 σ'(p) = 2 σ(p)(1 − σ(p)) = g − g²/2
grad_h_prev += (grad_q_pre) @ Q                     (hidden,)
grad_x     = grad_x_new ⊙ g                         (input,)  ← x itself unchanged by the gate

grad_Q      += h_prev^T @ grad_q_pre   shape (hidden, input) — outer product

For round `i` (forward `h_{prev} ← 2σ(R x) ⊙ h_prev`):
symmetric, swap roles of input and hidden.
```

In code we use the existing `Dense`-style matmul + bias; we don't reuse `Layer::Dense` itself because the cache structure differs (mogrifier is a pre-pass before the LSTM matmul, and we need separate caches for each round). We inline the matmul + add in the implementation, but keep the interface clean.

---

## Constructor & validation

`MogrifierLSTM(input_dim, hidden_size, num_rounds=2)`:

- Throws `std::invalid_argument` if `input_dim == 0` or `hidden_size == 0` (via placeholder-then-validate pattern).
- Throws if `num_rounds < 0` or `num_rounds > MAX_ROUNDS` (we cap at 5 — paper studies up to 5).
- For `num_rounds == 0` the layer is a vanilla LSTM.

---

## Parameter count

For `r` rounds (`r` even or odd both work — paper uses `r=2`):

| Tensor       | Shape                | r=0 | r=1 | r=2 | r=3 | r=4 |
| ------------ | ------------------- | --- | --- | --- | --- | --- |
| Q_i          | (hidden, input)     | 0   | 1   | 2   | 3   | 4   |
| R_i          | (input, hidden)     | 0   | 0   | 1   | 1   | 2   |
| W (lstm)     | (4·hidden, input+hidden) | 1 | 1 | 1 | 1 | 1 |
| b (lstm)     | (4·hidden,)         | 1   | 1   | 1   | 1   | 1   |

For r=2 (default): 2 Q + 1 R + 1 W + 1 b = **5 parameter tensors**.
For r=0 (vanilla): 1 W + 1 b = **2 parameter tensors**.

---

## Files

- Create: `include/nn/layers/recurrent/mogrifier_lstm.h`
- Create: `include/nn/layers/recurrent/mogrifier_lstm.cpp`
- Create: `tests/test_mogrifier_lstm.cpp`
- Modify: `include/nn/nn.h` (umbrella include after `min_lstm.h`)
- Modify: `Makefile` (`build/test_mogrifier_lstm` rule, `tests:` deps line, `=== Running Mogrifier LSTM Tests ===` echo in `run_tests`)
- Plan: this file `docs/plans/2026-09-04-mogrifier-lstm.md`

`LIB_SRCS` is a wildcard, so the new `.cpp` is picked up automatically.

---

## TDD approach

Each task ends with the test red→green→verified. Bite-sized steps, frequent commits.

---

### Task 1: Write the failing tests file (skeleton)

**Files:**
- Create: `tests/test_mogrifier_lstm.cpp`

Write 18 test stubs with clear pass/fail reporting. Tests:

1. Constructor validation: `input_dim=0`, `hidden_size=0`, `num_rounds=-1`, `num_rounds=6` throw
2. Forward shape `(T=4, input_dim=3) → (T=4, hidden=2)` is finite and nonzero
3. Round-0 equivalence: `num_rounds=0` reduces to vanilla LSTM (compared to a hand-derived T=1 reference)
4. Mogrifier signature test: with random init, the post-mogrifier `x_t` differs from the pre-mogrifier `x_t` (sanity — confirms the gating fires)
5. T=1 hand-derived reference: with hand-picked small weights, the mogrifier modifies `x` and `h_prev` before the gate computation; output matches a closed-form computation at rel_err 1e-9
6. Input gradient FD check (T=3, num_rounds=2) — rel_err < 1e-5
7. Q_1 gradient FD check (T=3) — rel_err < 1e-5
8. R_1 gradient FD check (T=3) — rel_err < 1e-5
9. W (lstm) gradient FD check (T=3) — rel_err < 1e-5
10. b (lstm) gradient FD check (T=3) — rel_err < 1e-5
11. zero_grad clears all 5 gradients
12. update_weights moves all 5 parameters
13. training reduces loss over 80 SGD steps
14. Longer sequence (T=6) input gradient FD check (rel_err < 1e-4 — accounts for BPTT accumulation)
15. `num_rounds=4` extra-round gradient check — confirms higher-round chain works
16. parameters()/gradients() contract — exactly 5 tensors (r=2) with shape-matched grads
17. Determinism: two fresh layers with copied params produce bit-exact forward
18. `num_rounds=0` FD check (T=3) — confirms vanilla LSTM backward path

Use the `l2_loss_value` / `l2_loss_grad` helpers and `relative_error` pattern from `tests/test_min_lstm.cpp` lines ~38–50.

**Verification:** the file compiles but every test reports "FAIL" because no implementation exists. (`g++` will fail because `MogrifierLSTM` doesn't exist yet — that's the red.)

```bash
g++ -std=c++17 -Iinclude -c tests/test_mogrifier_lstm.cpp -o /tmp/x.o 2>&1 | head -20
# expected: error: 'MogrifierLSTM' was not declared in this scope
```

**Commit:** none yet (file compiles fail — wait for impl stub first).

---

### Task 2: Write the header (public API + skeleton implementation that throws)

**Files:**
- Create: `include/nn/layers/recurrent/mogrifier_lstm.h` (full public API)
- Create: `include/nn/layers/recurrent/mogrifier_lstm.cpp` (skeleton that throws on every method)

The header declares the class with:
- `size_t input_dim_, hidden_size_, num_rounds_`
- Parameters: `std::vector<Tensor> Q_list_, R_list_, W_, b_`  (Q_list_ has `ceil(num_rounds/2)` entries; R_list_ has `floor(num_rounds/2)` entries)
- Gradients: same names prefixed `grad_`
- Cache tensors: `inputs_`, `h_states_`, `c_states_`, plus per-round caches `mog_x_list_` and `mog_h_list_` (shape `(T, input)` / `(T, hidden)`)
- Public methods matching `Layer`: `forward`, `backward`, `update_weights`, `get_weights`, `get_gradients`, `parameters`, `gradients`, `zero_grad`, `name`
- Plus `forward_sequence(const Tensor& seq)` returning the full `(T, hidden)` trajectory
- Plus `reset_state()` for clean FD re-evaluations

The cpp skeleton defines every method to throw `std::logic_error("not implemented")` except the constructor, which sets dimensions and zero-fills everything.

**Verification:**

```bash
g++ -std=c++17 -Iinclude -c tests/test_mogrifier_lstm.cpp -o /tmp/x.o 2>&1 | head -10
g++ -std=c++17 -Iinclude -c include/nn/layers/recurrent/mogrifier_lstm.cpp -o /tmp/y.o 2>&1 | head -10
# expected: both compile clean
```

```bash
g++ build/test_mogrifier_lstm.o include/nn/layers/recurrent/mogrifier_lstm.o ... -o build/test_mogrifier_lstm
./build/test_mogrifier_lstm
# expected: 18/18 FAIL — every method throws
```

**Commit:**

```bash
git add tests/test_mogrifier_lstm.cpp include/nn/layers/recurrent/mogrifier_lstm.h include/nn/layers/recurrent/mogrifier_lstm.cpp
git commit -m "test: mogrifier lstm test scaffold (18 tests, 0/18 pass)"
```

---

### Task 3: Implement constructor + forward + parameter layout (no backward yet)

**Files:**
- Modify: `include/nn/layers/recurrent/mogrifier_lstm.cpp`

Implement:
- `MogrifierLSTM(input_dim, hidden_size, num_rounds=2)` — validate after init-list, call `init_weights()`.
- `init_weights()` — Xavier-uniform per-projection with independent RNG draws (per tensor), bias-init for forget gate (b_forget = 1.0 in slice of b), zero for input/output/candidate bias. Mogrifier Q/R init: smaller scale (0.1 × Xavier) so initial gates are ~0.5 sigmoid → small mogrification at init, train grows them.
- `forward_sequence(Tensor seq)` — run mogrifier rounds for each timestep, then standard LSTM gates. Cache everything.
- `forward(Tensor input)` — single-step API matching `MinLSTM::forward` (uses cached state from last `forward_sequence` or zero-init).
- `parameters()`, `gradients()`, `zero_grad()`, `get_weights()`, `get_gradients()`, `name()`.

`backward()` and `update_weights()` still throw `not implemented` for now.

**Verification:**

```bash
make build/test_mogrifier_lstm 2>&1 | tail -5
./build/test_mogrifier_lstm
# expected: tests 1-5 pass (constructor validation, forward shape, finiteness,
# round-0 equivalence, signature sanity). Tests 6-18 still fail with "not implemented".
```

Particularly:
- Test 1: all 4 invalid inputs throw `std::invalid_argument` — verify the throw message contains `"input_dim"`, `"hidden_size"`, `"num_rounds"`.
- Test 2: `(4, 3) → (4, 2)` finite, nonzero.
- Test 3: with `num_rounds=0`, the layer matches vanilla LSTM (no mogrification). Build a tiny vanilla LSTM forward reference in the test and compare at rel_err < 1e-10.
- Test 4: post-mogrifier `x_t` ≠ pre-mogrifier `x_t` for at least one (round, channel) pair (i.e. the gate is not the identity).
- Test 5: hand-derived T=1 reference matches at rel_err < 1e-9.

**Commit:**

```bash
git add include/nn/layers/recurrent/mogrifier_lstm.cpp
git commit -m "feat(recurrent): MogrifierLSTM forward + constructor (5/18 tests pass)"
```

---

### Task 4: Implement backward (the hard part)

**Files:**
- Modify: `include/nn/layers/recurrent/mogrifier_lstm.cpp`

Implement `backward(grad_output, learning_rate)`:

1. Standard LSTM backward first:
   - For each `t` in reverse, compute `grad_i`, `grad_f`, `grad_g`, `grad_o` from `grad_h[t]`.
   - Standard grad formulas:
     - `grad_o = grad_h ⊙ tanh(c_t) ⊙ σ'(z_o) = grad_h ⊙ tanh(c_t) ⊙ o(1-o)`
     - `grad_c[t] += grad_h ⊙ o ⊙ (1 - tanh²(c_t))`
     - `grad_c[t-1] = grad_c[t] ⊙ f`
     - `grad_z_g = grad_c[t] ⊙ i ⊙ (1 - g²)` where `g = tanh(z_g)`
     - `grad_z_i = grad_c[t] ⊙ g ⊙ i(1-i)`
     - `grad_z_f = grad_c[t] ⊙ c_{t-1} ⊙ f(1-f)`
   - Accumulate `grad_W` via outer products `[x_mog_t; h_mog_{t-1}] ⊗ [dz_i; dz_f; dz_g; dz_o]`.
   - Accumulate `grad_b` via sum.
   - Hold `grad_x_mog[t]` and `grad_h_mog_prev[t]` separately for the mogrifier backward.

2. Mogrifier backward (reverse order through rounds):
   - Start with `grad_x = grad_x_mog[T-1]`, `grad_h_prev = grad_h_mog_prev[T]` (the carrier for next timestep, plus the direct grad from timestep T).
   - For each `t` in reverse:
     - Combine `grad_x_mog[t]` into `grad_x` (accumulate), `grad_h_mog_prev[t]` into `grad_h_prev` (accumulate).
     - For round index `r_idx` in reverse (largest round first):
       - If round is x-modifying: `grad_h_prev += grad_x ⊙ 2 σ(Q h_prev) → propagate through Q matmul`. Use the cached `q_pre` and `g_val` from the forward pass.
       - Symmetric for h-modifying rounds.
       - Accumulate `grad_Q[r_idx]` or `grad_R[r_idx]` via outer product.
     - Write `grad_x` into `grad_inputs[t]`.

3. Return `grad_inputs`.

**Verification (this is the critical task):**

```bash
make build/test_mogrifier_lstm 2>&1 | tail -5
./build/test_mogrifier_lstm
# expected: tests 6-10 pass (input grad, Q_1 grad, R_1 grad, W grad, b grad all match FD within rel_err 1e-5)
```

For each gradient test, FD must compare the analytical grad against central-difference `(f(x+ε) - f(x-ε)) / (2ε)` at `ε = 1e-4`. Test uses `l2_loss = sum((out - target)²) / 2` so the grad-out is just `(out - target)`.

**Mutation test before declaring GREEN**: stub out the mogrifier contribution by zeroing `grad_Q_list` and `grad_R_list` accumulators. Test 7/8 (Q_1 / R_1 grad) should fail. Restore. Then stub out the LSTM gates (zero `grad_W`/`grad_b`) — test 9/10 should fail.

**Commit:**

```bash
git add include/nn/layers/recurrent/mogrifier_lstm.cpp
git commit -m "feat(recurrent): MogrifierLSTM analytical backward (10/18 tests pass, mutation-tested)"
```

---

### Task 5: Implement update_weights + training + contract + determinism

**Files:**
- Modify: `include/nn/layers/recurrent/mogrifier_lstm.cpp`

Implement `update_weights(lr)`:
- For each parameter, do `param -= lr * grad_param`, then `grad_param.fill(0)` is NOT done here (caller's job; same convention as `MinLSTM::update_weights`).

Verify:
- Test 11 (zero_grad): after `zero_grad`, every `grad_*` entry is `0.0`.
- Test 12 (update_weights): after `update_weights(lr)` with non-zero grads, every param changed.
- Test 13 (training): over 80 SGD steps with hand-crafted synthetic regression, loss decreases by > 50%.
- Test 14 (long seq): T=6 input grad FD check at rel_err < 1e-4.
- Test 15 (r=4): an extra mogrifier round is exercised — input grad FD rel_err < 1e-4.
- Test 16 (contract): `parameters()` returns 5 tensors; `gradients()` returns 5 tensors; shapes match element-wise.
- Test 17 (determinism): construct two layers, copy params, run forward, max abs diff = 0.
- Test 18 (vanilla): `num_rounds=0` — input grad FD matches what vanilla LSTM would produce.

**Verification:**

```bash
make build/test_mogrifier_lstm 2>&1 | tail -5
./build/test_mogrifier_lstm
# expected: 18/18 pass
```

**Commit:**

```bash
git add include/nn/layers/recurrent/mogrifier_lstm.cpp
git commit -m "feat(recurrent): MogrifierLSTM training + contract + determinism (18/18 tests pass)"
```

---

### Task 6: Wire into umbrella + Makefile

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/recurrent/mogrifier_lstm.h"` after `min_lstm.h`.
- Modify: `Makefile`:
  - Add `$(BUILD_DIR)/test_mogrifier_lstm: $(LIB_OBJS) $(BUILD_DIR)/test_mogrifier_lstm.o` + `$(CXX) $^ -o $@` (alphabetically near `test_min_lstm`).
  - Add `$(BUILD_DIR)/test_mogrifier_lstm` to the `tests:` deps line.
  - Add `	@echo "=== Running Mogrifier LSTM Tests ===" && ./$(BUILD_DIR)/test_mogrifier_lstm` in `run_tests` (alphabetically near the other recurrent echoes).

**Verification:**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
# expected: clean (no errors)
```

```bash
make tests 2>&1 | tail -10
# expected: includes "=== Running Mogrifier LSTM Tests ===" with 18 passes
```

No regressions in any of the 130+ existing test binaries.

**Commit:**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register MogrifierLSTM in umbrella + Makefile"
```

---

### Task 7: Update EXPANSION_QUEUE.md

**Files:**
- Modify: `EXPANSION_QUEUE.md`

Move the MogrifierLSTM item from `## Ideas` to `## Done` with a one-line summary of what was implemented and the final test count.

**Verification:**

```bash
grep -A 2 "## Ideas" EXPANSION_QUEUE.md | head -5
# expected: ## Ideas (empty line) followed by ## Done
```

```bash
grep "Mogrifier" EXPANSION_QUEUE.md | head -3
# expected: "## Done" section now contains the Mogrifier entry with summary
```

**Commit:**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark MogrifierLSTM as Done in EXPANSION_QUEUE"
```

---

## Acceptance criteria

- [ ] All 18 focused checks pass
- [ ] Mutation-tested non-vacuous (at least 2 mutations, each caught)
- [ ] Compiles cleanly under `-Wall -Wextra`
- [ ] Umbrella compiles standalone
- [ ] No regressions in existing test suites
- [ ] Conventional-commit messages with descriptive scope
- [ ] Plan file persisted at `docs/plans/2026-09-04-mogrifier-lstm.md`