# Adaptive Computation Time (ACT) — Implementation Plan

> **For Hermes:** Use TDD to implement this plan task-by-task.

**Goal:** Implement Graves 2016's Adaptive Computation Time as a recurrent layer that learns per-sequence how many recurrent steps to take.

**Architecture:** `ACTRecurrentLayer` wraps a simple tanh RNN cell with a learnable halting head. Forward maintains a running "ponder sum" N_t and halts the moment N_t crosses 1. Output is a weighted average of per-step outputs, where the final-step weight is the remainder budget (1 − N_{t-1}). Backward is hand-derived BPTT through the halting head and the underlying RNN.

**Tech Stack:** C++17, `Tensor` / `Layer` / `Model` abstractions, Make, custom assertion harness, strict RED→GREEN→REFACTOR TDD.

---

## Algorithm summary (Graves 2016, §3)

Per token t (within a fixed sequence of max_steps tokens):
1. `h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)` and `y_t = h_t` (the output equals the hidden state — the simplest case, mirroring the paper's equation 1).
2. `p_t = σ(W_halt · y_t + b_halt)` — halting logit, in (0, 1).
3. `N_t = N_{t-1} + p_t`.
4. **Halt condition**: stop after the first step where `N_{t-1} < 1 AND N_{t-1} + p_t ≥ 1`. If we never cross 1 by max_steps, halt at the last step.
5. Per-step output weight `w_t`:
   - For `t < halt_step`: `w_t = p_t`.
   - For `t = halt_step`: `w_t = 1 − N_{t-1}` (the "remainder" — paper eq. 4).
   - For `t > halt_step`: `w_t = 0`.
6. **Output**: `ŷ = Σ_t w_t · y_t` (a weighted average; the weights sum to 1).
7. **Ponder cost**: `ρ = Σ_t p_t` (paper eq. 5; Σ is over ALL steps up to halt, NOT just to halt_step; in our impl with a fixed max sequence we sum over the steps we ran).

For the first implementation we treat N=1 (single sample per call), matching the repo's `SimpleRNN` convention. Batch-size>1 with per-sample variable halting is a strict extension (see "Future extensions" at the bottom).

---

## Files to touch

- **Create**: `include/nn/layers/recurrent/act.h` — public API.
- **Create**: `include/nn/layers/recurrent/act.cpp` — implementation.
- **Create**: `tests/test_act.cpp` — focused test suite.
- **Modify**: `include/nn/nn.h` — add `#include "layers/recurrent/act.h"` in the recurrent section (verified by standalone `-fsyntax-only` compile).
- **Modify**: `Makefile` — add `build/test_act` rule, register `tests:` dep, add `=== Running ACT Tests ===` echo in `run_tests`.
- **Modify**: `EXPANSION_QUEUE.md` — move ACT entry from `## Ideas` to `## Done` with a one-line summary.
- **Create**: this plan file.

---

## Task 1: Failing test for constructor + forward shape

**Step 1: Write failing test**

```cpp
// in tests/test_act.cpp
#include "nn/layers/recurrent/act.h"

TEST_START("ACT constructor validates inputs")
    // hidden=0 throws
    CHECK_THROWS(ACTRecurrentLayer(3, 0, 5));
    // max_steps=0 throws
    CHECK_THROWS(ACTRecurrentLayer(3, 4, 0));
    // valid constructs
    ACTRecurrentLayer ok(3, 4, 5);
    CHECK(ok.input_dim() == 3);
    CHECK(ok.hidden_size() == 4);
    CHECK(ok.max_steps() == 5);

TEST_START("ACT forward produces a (1, hidden) output from a (1, max_steps*input) input")
    ACTRecurrentLayer layer(3, 4, 5);
    Tensor input(1, 5 * 3);
    input.fill(0.5);
    Tensor out = layer.forward(input);
    CHECK(out.rows == 1);
    CHECK(out.cols == 4);
    CHECK(is_finite(out));   // helper
```

**Step 2: Run test — expect FAIL (header not found).**

**Step 3: Implement minimal header + skeleton cpp.**

```cpp
// include/nn/layers/recurrent/act.h
#ifndef ACT_H
#define ACT_H

#include "../../core/layer.h"

class ACTRecurrentLayer : public Layer {
public:
    ACTRecurrentLayer(int input_dim, int hidden_size, int max_steps, double ponder_penalty = 0.01);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "ACTRecurrentLayer"; }

    int input_dim() const { return input_dim_; }
    int hidden_size() const { return hidden_size_; }
    int max_steps() const { return max_steps_; }
    double ponder_penalty() const { return ponder_penalty_; }

    int    actual_steps() const { return halt_step_ + 1; }
    double halt_probability(int t) const { return halt_probs_[0][t]; }

private:
    int input_dim_, hidden_size_, max_steps_;
    double ponder_penalty_;

    // RNN cell weights
    Tensor W_xh_, W_hh_, b_h_;
    Tensor grad_W_xh_, grad_W_hh_, grad_b_h_;

    // Halting head: W_halt : (hidden, 1), b_halt : (1, 1)
    Tensor W_halt_, b_halt_;
    Tensor grad_W_halt_, grad_b_halt_;

    // Caches
    Tensor inputs_;
    Tensor hidden_states_;   // (max_steps+1, hidden)  — h_0 .. h_T
    Tensor outputs_;         // (max_steps, hidden)    — y_t
    Tensor halt_probs_;      // (1, max_steps)
    Tensor weights_;         // (1, max_steps)
    int    halt_step_;
};

#endif
```

cpp: minimal init weights (uniform small), empty forward that just returns `Tensor(1, hidden_size_)`.

**Step 4: Run test — expect PASS.**

**Step 5: Commit.**

```
feat(recurrent): ACTRecurrentLayer skeleton + forward-shape tests
```

---

## Task 2: Forward halt logic correctness

**Step 1: Test cases**

```cpp
TEST_START("ACT weights sum to 1 and respect p_t bounds")
    ACTRecurrentLayer layer(2, 3, 4);   // 4 steps
    Tensor input(1, 4 * 2);
    // Fill input with random small values so p_t vary
    std::mt19937 gen(7);
    std::normal_distribution<> nd(0.0, 0.3);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = nd(gen);
    Tensor out = layer.forward(input);

    // Sum of per-step weights should equal 1 exactly (to FP noise)
    double wsum = 0.0;
    for (int t = 0; t < 4; ++t) wsum += layer.weight_at(t);
    CHECK(std::abs(wsum - 1.0) < 1e-12);

    // For t < halt: weight = p_t
    int hs = layer.actual_steps() - 1;
    for (int t = 0; t < hs; ++t) {
        double w = layer.weight_at(t);
        double p = layer.halt_probability(t);
        CHECK(std::abs(w - p) < 1e-12);
    }
    // For t = halt: weight = 1 - N_{t-1}
    // For t > halt: weight = 0
    for (int t = hs + 1; t < 4; ++t) {
        CHECK(std::abs(layer.weight_at(t)) < 1e-12);
    }

TEST_START("ACT forward reaches the halt step, not all max_steps")
    // Force large p_t so we halt early
    ACTRecurrentLayer layer(2, 3, 10);
    // Init the halting head bias to a large positive value to make p_t near 1
    // (need an accessor — add set_halt_bias in the API)
    layer.set_halt_bias(5.0);
    Tensor input(1, 10 * 2);
    input.fill(0.1);
    layer.forward(input);
    // With p_t ≈ 1, should halt at step 0 (or 1)
    CHECK(layer.actual_steps() <= 2);
```

**Step 2: Add `weight_at(t)` accessor + `set_halt_bias(double)` to header. Implement forward.**

The forward pass must:
1. Reshape input (1, max_steps·input_dim) into per-step x_t's.
2. Initialize `N_0 = 0`, `h_0 = 0`.
3. For t = 0, 1, …, max_steps-1:
   - `pre_h = x_t · W_xh^T + h_{t-1} · W_hh^T + b` (broadcasted)
   - `h_t = tanh(pre_h)`, `y_t = h_t`
   - `z_t = y_t · W_halt + b_halt` (scalar)
   - `p_t = σ(z_t)`
   - If `t > 0` and `N_{t-1} >= 1`: stop (early; already halted last step).
   - Else if `N_{t-1} + p_t >= 1`: record halt at t, set weight t = 1 − N_{t-1}, halt_step = t, set all later weights = 0, STOP.
   - Else: weight t = p_t, N_t = N_{t-1} + p_t.
4. If we reach t = max_steps without halting: halt_step = max_steps-1, last weight = 1 − N_{max_steps-2}, all earlier weights = their p_t.

**Step 3: Run, expect PASS.**

**Step 4: Commit.**

```
feat(recurrent): ACT forward with halt logic + weights invariants
```

---

## Task 3: Hand-derived reference (sanity)

**Step 1: Test**

```cpp
TEST_START("ACT hand-derived single-step reference")
    // hidden=1, input=1, max_steps=3. Trivial config where we can compute by hand.
    ACTRecurrentLayer layer(1, 1, 3);
    // Zero out everything except input_dim → first step is just tanh(x_0 * W_xh[0][0] + b[0])
    layer.zero_all_weights();
    layer.set_W_xh(0, 0, 1.0);   // x*W_xh = x_0
    layer.set_b(0, 0.0);          // h_1 = tanh(x_0)  (h_0 = 0, so W_hh * 0 = 0)
    layer.set_W_halt(0, 0, 0.0);   // b_halt = 0 → p_0 = 0.5
    layer.set_b_halt(0.0);
    Tensor input(1, 3);  // (1, max_steps * input_dim = 3*1 = 3)
    input[0][0] = 0.0;   // x_0 = 0 → h_1 = tanh(0) = 0
    input[0][1] = 0.0;
    input[0][2] = 0.0;
    Tensor out = layer.forward(input);
    // Expected: at step 0, N_prev = 0, p_0 = 0.5 → N_new = 0.5 < 1 → continue, weight_0 = p_0 = 0.5
    // h_1 = tanh(0) = 0 → y_0 = 0 → step-0 contributes 0.5*0 = 0
    // Step 1: h_2 = tanh(0*W_xh + 0*W_hh + 0) = 0; p_1 = 0.5; weight_1 = 0.5
    // Step 2: h_3 = tanh(0) = 0; p_2 = 0.5; weight_2 = 0.5; (final if no halt) but N_1=1 → halt at step 2? wait N_prev + p_t >= 1 → 1.0 + 0.5 = 1.5 >= 1 → halt. weight_2 = 1 - 1.0 = 0
    // Σ weights = 0.5 + 0.5 + 0 = 1.0, Σ y*w = 0+0+0 = 0
    CHECK(std::abs(out[0][0] - 0.0) < 1e-12);
```

Then a non-trivial case:
```cpp
TEST_START("ACT hand-derived non-trivial output sum")
    ACTRecurrentLayer layer(1, 2, 3);
    layer.zero_all_weights();
    layer.set_W_xh(0, 0, 0.5);  // W_xh: 2 rows × 1 col
    layer.set_W_xh(1, 0, 0.5);
    layer.set_b(0, 0.0); layer.set_b(1, 0.0);
    layer.set_W_halt(0, 0, -100.0);  // very negative → p_t ≈ 0 → never halts before max_steps
    layer.set_W_halt(1, 0, -100.0);
    layer.set_b_halt(-100.0);
    Tensor input(1, 3);
    input[0][0] = 1.0; input[0][1] = 1.0; input[0][2] = 1.0;
    // With p_t ~ 0, weights are all ~0 except the last step weight = 1 - N_prev ≈ 1 - 0 = 1
    // So output ≈ y_{max_steps-1} = h_3
    // h_0 = 0
    // h_1 = tanh(1*0.5 + 0*W_hh + 0) = tanh(0.5) ≈ 0.4621
    // h_2 = tanh(1*0.5 + tanh(0.5)*W_hh[0][0] + tanh(0.5)*W_hh[1][0] + 0)
    //   but W_hh is 2x2 = 0, so h_2 = tanh(0.5) ≈ 0.4621
    // h_3 = tanh(1*0.5 + 0) = tanh(0.5) ≈ 0.4621
    // Final step weight ≈ 1, output ≈ [0.4621, 0.4621]
    Tensor out = layer.forward(input);
    double expected = std::tanh(0.5);
    CHECK(std::abs(out[0][0] - expected) < 1e-6);
    CHECK(std::abs(out[0][1] - expected) < 1e-6);
```

**Step 2: Implement. Add setters.**

**Step 3: Run, expect PASS.**

**Step 4: Commit.**

```
feat(recurrent): ACT hand-derived single-step and constant-halt references
```

---

## Task 4: Backward — hand-derived for halting head + ponder penalty

**Step 1: Test**

```cpp
TEST_START("ACT FD gradient of W_halt at machine precision")
    ACTRecurrentLayer layer(2, 3, 4);
    // Random non-uniform init
    std::mt19937 gen(11);
    std::normal_distribution<> nd(0.0, 0.3);
    // populate W_xh, W_hh, b, W_halt, b_halt randomly
    layer.random_init(gen);
    Tensor input(1, 4 * 2);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = nd(gen);
    // Forward + backward + FD
    Tensor out = layer.forward(input);
    Tensor grad_out(1, 3);
    for (size_t i = 0; i < 3; ++i) grad_out[0][i] = nd(gen);
    layer.zero_grad();
    layer.backward(grad_out, 0.0);   // learning_rate ignored
    double analytical = layer.grad_W_halt_at(0, 0);
    double numerical = fd_W_halt(layer, input, grad_out, 0, 0, 1e-5);
    double rel_err = std::abs(analytical - numerical) / std::max(std::abs(analytical), std::abs(numerical));
    CHECK(rel_err < 1e-5);
```

**Step 2: Implement backward through the weighted output → halting head → RNN.**

The chain:
- `ŷ = Σ_t w_t y_t`
- `d_ŷ = grad_output`
- For each step t: `d_y[t] = w_t · d_ŷ`, `d_w[t] = ⟨y_t, d_ŷ⟩` (dot product — returns scalar per step).
- For `t < halt_step`: `w_t = p_t` so `d_p[t] = d_w[t]`.
- For `t = halt_step`: `w_t = 1 − N_{t-1}` so `d_N_{t-1} = -d_w[t]`, `d_p[t] = 0`.
- For `t > halt_step`: `w_t = 0` so `d_p[t] = 0`.
- Recursively propagate `d_N` backward through the running sum: `d_N_{t-1} += d_N_t` (since `N_t = N_{t-1} + p_t`), then `d_p[t-1] += d_N_t` for each step. But this only matters if halt happens EARLIER than max_steps — if we halt at step h, then `N_h = N_{h-1} + p_h ≥ 1`, but for steps AFTER h the weights are 0 so d_w[t] = 0; AND N_t is no longer computed (loop exits). So the propagation is bounded: for t > halt_step, no d_p accumulates from the running sum.

Actually, simpler: we don't need to propagate `d_N_t` forward at all, because `N_{t-1}` only enters the formula for `w_{halt_step}`. So the FULL derivation is:
- For each step t ≤ halt_step:
  - Compute `d_y[t] = w_t · d_ŷ`
  - Compute `d_w[t] = ⟨y_t, d_ŷ⟩`
  - If `t < halt_step`: `d_p[t] = d_w[t]`. (w_t = p_t, d/dp = 1.)
  - If `t = halt_step`: `d_p[t] = 0`, `d_N_{t-1} = -d_w[t]`.
- For each step t ≤ halt_step, propagate `d_N_{t-1}` further: if `t-1 < halt_step`, then `d_p[t-1] += d_N_{t-1}` (because w_{t-1} = p_{t-1} and N_{t-1} appears in no other w). Actually wait — `N_{t-1}` is used ONLY in `w_t = 1 - N_{t-1}` when t = halt_step. So the only propagation that matters is: from `d_N_{t-1}` (received from the halt-step derivative), add it to `d_p[t-1]` ONLY if `t-1 < halt_step` (i.e. t-1 is a regular non-final step). But for the FIRST halting, `t-1 = halt_step - 1 < halt_step` so `d_p[halt_step-1] += d_N_{t-1}`.

Hmm wait, but actually for steps BEFORE halt_step, `w_t = p_t` — `N_{t-1}` doesn't appear there at all! So `d_N_{t-1}` from the halt step doesn't affect `d_p[t-1]`; it affects only the previous step's BPTT through the RNN (because the next-step p_t depends on h_t, which depends on the previous step's state). But the ACT paper §3.3 says that when computing the gradient, `d N_{t-1} / dp_{t-1} = 1` (chain through the running sum), which feeds into the upstream gradient for h_{t-1} → RNN backward.

The cleanest derivation:
1. Compute the "intermediate" gradient `d_w[t] = ⟨y_t, d_ŷ⟩` for t ≤ halt_step, and the corresponding `d_y[t] = w_t · d_ŷ`.
2. For each step t ≤ halt_step:
   - If `t < halt_step`: add `d_w[t]` to `d_p[t]` (since w_t = p_t).
   - If `t = halt_step`: store `d_N_{t-1} = -d_w[t]` (NOT contributing to d_p[t] since w_t doesn't depend on p_t).
3. The "upstream through N" propagation: for each step t, the contribution to `d_N_t` from the running sum is 1 (since N_t = N_{t-1} + p_t, dN_t/dN_{t-1} = 1). So we propagate `d_N` from step t to step t-1 with `d_N_{t-1} += d_N_t` for t = halt_step+1 backward.
   - But for steps > halt_step, no gradient flows (loop stopped). So the propagation is bounded: starting from `d_N_{halt_step-1} = -d_w[halt_step]`, we propagate BACKWARDS through the chain `d_N_{s-1} += d_N_s` for s = halt_step, halt_step-1, ..., 1. Each step adds to `d_N_{s-1}`.
   - Each `d_N_{s-1}` for s-1 < halt_step adds to `d_p[s-1]` (since p_{s-1} contributed to N_{s-1}). So: `d_p[s-1] += d_N_s` for s = 1, ..., halt_step.

So the full p-gradient is:
- `d_p[t] = d_w[t]` for t < halt_step (direct contribution from w_t)
- `d_p[t] = d_w[t] + d_N_{t+1}` for t < halt_step (chain contribution from the next step)
- `d_p[t] = 0` for t = halt_step
- `d_p[t] = 0` for t > halt_step (loop stopped, no gradient)

Where `d_N_{t+1}` for t < halt_step is the propagated gradient from the running sum. We can compute it iteratively from `d_N_{halt_step} = -d_w[halt_step]` and propagate backward:
```
d_N[halt_step] = -d_w[halt_step]
for s = halt_step, halt_step-1, ..., 1:
    d_p[s-1] += d_N[s]      # if s-1 < halt_step
    d_N[s-1] = d_N[s]       # propagate to previous step
```
Then add the direct contributions:
```
for t = 0, ..., halt_step - 1:
    d_p[t] += d_w[t]
```
And add the ponder contribution:
```
for t = 0, ..., halt_step:
    d_p[t] += ponder_penalty   # always, because the loss adds ponder_penalty * Σ p_t
```
(ponder is on ALL p_t's up to halt, including the halt step; the paper's §4 specifies this.)

Then `d_z[t] = d_p[t] · σ'(z[t]) = d_p[t] · p_t · (1 − p_t)`.
- `d_W_halt += y_t ⊗ d_z[t]` (outer product, shape (hidden, 1)).
- `d_b_halt += d_z[t]`.
- `d_y[t] += d_z[t] · W_halt^T` (broadcast over hidden dims, scalar broadcasted).

The total `d_y[t]` (sum of direct weighted-output contribution and halt-head contribution) is then fed into the RNN backward.

**Step 3: Run, expect PASS.**

**Step 4: Commit.**

```
feat(recurrent): ACT hand-derived backward (halt head + ponder)
```

---

## Task 5: Backward — RNN BPTT through y_t

**Step 1: Test**

```cpp
TEST_START("ACT FD gradient of W_xh at machine precision")
    ACTRecurrentLayer layer(2, 3, 4);
    layer.random_init(gen);
    Tensor input(1, 4 * 2);
    // ... fill with random
    Tensor out = layer.forward(input);
    Tensor grad_out(1, 3); grad_out.fill(0.5);
    layer.zero_grad();
    layer.backward(grad_out, 0.0);
    double analytical = layer.grad_W_xh_at(0, 0);
    double numerical = fd_W_xh(layer, input, grad_out, 0, 0, 1e-5);
    double rel_err = std::abs(analytical - numerical) / std::max(std::abs(analytical), std::abs(numerical));
    CHECK(rel_err < 1e-5);

TEST_START("ACT FD gradient of W_xh passes through halting head chain")
    // Same as above but with high ponder_penalty so the d_p[t] → d_y[t] chain is exercised
    ACTRecurrentLayer layer(2, 3, 4, /*ponder_penalty=*/1.0);
    // ... etc
```

**Step 2: Implement RNN BPTT.**

This is standard BPTT for the tanh RNN, identical to `SimpleRNN::backward`. The difference is the per-step gradient we feed into it: `d_y[t]` for t ≤ halt_step (d_y[t] = 0 for t > halt_step). Run BPTT from t = halt_step down to t = 0, accumulating grad_W_xh, grad_W_hh, grad_b.

Then the input gradient: `grad_input[i, t*input_dim + d] = d_x[t][i, d]` (since input is just a reshape of per-step tokens; no projection involved).

**Step 3: Run, expect PASS.**

**Step 4: Commit.**

```
feat(recurrent): ACT RNN-cell BPTT backward
```

---

## Task 6: Backward — input gradient

**Step 1: Test**

```cpp
TEST_START("ACT FD gradient of input at machine precision")
    ACTRecurrentLayer layer(2, 3, 4);
    layer.random_init(gen);
    Tensor input(1, 4 * 2);
    // ... fill with random
    Tensor out = layer.forward(input);
    Tensor grad_out(1, 3); grad_out.fill(0.5);
    Tensor grad_in = layer.backward(grad_out, 0.0);
    double analytical = grad_in[0][0];
    double numerical = fd_input(layer, input, grad_out, 0, 0, 1e-5);
    double rel_err = std::abs(analytical - numerical) / std::max(std::abs(analytical), std::abs(numerical));
    CHECK(rel_err < 1e-5);
```

**Step 2: Verify the BPTT chain returns grad_input with shape (1, max_steps·input_dim).**

**Step 3: Commit.**

```
feat(recurrent): ACT input gradient FD check
```

---

## Task 7: update_weights + training reduces loss

**Step 1: Test**

```cpp
TEST_START("ACT training reduces MSE loss over 50 SGD steps")
    ACTRecurrentLayer layer(2, 3, 4);
    layer.random_init(gen);
    Tensor input(1, 4 * 2);  // fill with random
    Tensor target(1, 3);     // fill with random (e.g. sin of input)
    double initial_loss = mse(layer.forward(input), target);
    for (int step = 0; step < 50; ++step) {
        Tensor out = layer.forward(input);
        Tensor grad_out = (out - target) * (2.0 / 3.0);   // d/dy of MSE
        layer.zero_grad();
        layer.backward(grad_out, 0.05);
        layer.update_weights(0.05);
    }
    Tensor final_out = layer.forward(input);
    double final_loss = mse(final_out, target);
    CHECK(final_loss < initial_loss);
```

**Step 2: Implement update_weights (in-place SGD on W_xh, W_hh, b, W_halt, b_halt) — standard.**

**Step 3: Run, expect PASS.**

**Step 4: Commit.**

```
feat(recurrent): ACT update_weights + end-to-end training
```

---

## Task 8: Mutation testing (non-vacuousness)

**Step 1: Mutation tests**

Each of these mutations must break at least one existing test:

1. Drop the `d_p[t] += d_w[t]` direct contribution → FD W_halt fails.
2. Drop the ponder contribution `d_p[t] += ponder_penalty` → FD W_halt differs (different value, not magnitude).
3. Drop the propagated `d_N` from the halt step → FD W_halt fails.
4. Skip the d_y[t] += d_z[t] · W_halt^T (forgetting the backflow into RNN) → FD W_xh fails.
5. Zero W_halt (so p_t = σ(b_halt)) — should still pass forward, but W_halt FD fails (it's zero).
6. In forward, forget the halt early-exit and run all max_steps — the weights sum > 1, breaks invariant test.

**Step 2: Verify each mutation breaks at least one assertion.**

**Step 3: Commit.**

```
test(recurrent): ACT mutation tests confirm non-vacuous coverage
```

---

## Task 9: Register in umbrella header + Makefile

**Step 1: `include/nn/nn.h`**

Find the line `#include "layers/recurrent/s5.h"` (or wherever s5.h is included) and add `#include "layers/recurrent/act.h"` right after it (alphabetical/insertion order).

Verify: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'` compiles cleanly (only pre-existing warnings).

**Step 2: `Makefile`**

- Add `$(BUILD_DIR)/test_act: $(LIB_OBJS) $(BUILD_DIR)/test_act.o` after the existing test_act-like rules.
- Add `$(BUILD_DIR)/test_act.o` in the deps chain.
- Add `$(BUILD_DIR)/test_act` to the `tests:` deps list (alphabetical/insertion).
- Add `run_tests:` echo: `@echo "=== Running ACT Tests ===" && ./$(BUILD_DIR)/test_act`.

**Step 3: Run `make tests` and verify ACT passes alongside all other suites.**

**Step 4: Commit.**

```
chore: register ACTRecurrentLayer in umbrella + Makefile
```

---

## Task 10: Update EXPANSION_QUEUE.md

Move the ACT line from `## Ideas` to `## Done` with a brief summary (≥5 lines, matching the format used for other entries).

```
docs: mark ACT as Done in EXPANSION_QUEUE
```

---

## Test plan summary (final focused suite)

After all tasks:
- Constructor validation (3-4 cases)
- Forward shape, finiteness
- Halt invariants: weights sum to 1; weights[t] = p[t] for t < halt; weights[t] = 1 − N_{t-1} at halt; weights[t] = 0 for t > halt
- Hand-derived single-step reference
- Hand-derived constant-halt reference (p_t ≈ 0 ⇒ weight ≈ 0 except last)
- FD gradient: W_halt, b_halt, W_xh, W_hh, b, input (all at machine precision)
- update_weights moves all 5 parameter groups
- end-to-end training reduces MSE loss
- Ponder penalty contributes to gradients (verifiable: doubling ponder_penalty doubles d_b_halt contribution)
- 5-6 mutation tests confirm non-vacuous coverage

Target: ~30 focused checks.

---

## Future extensions (NOT in this PR)

- **Batched ACT (B > 1 with per-sample variable halt)**: each sample maintains its own N_t; halting logic runs per-row. The output is computed per-row, and the gradient is summed across rows but only for t ≤ sample_halt_step[row]. This is a strict extension; we keep this PR at B=1.
- **Slower-than-sigmoid halting head** (e.g. with temperature or hard sigmoid) — paper's "Improved ACT" variant.
- **Ponder-loss-only training** (detach p_t from output gradient, only use it as a regularizer) — paper §4.