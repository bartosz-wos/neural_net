# GAU (Gated Attention Unit) Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → verify).

**Goal:** Add `GatedAttentionUnit` (Hua et al. 2022, "Transformer Quality in Linear Time", https://arxiv.org/abs/2206.03637) — a single-head, softmax-free, gated attention unit that achieves quality comparable to softmax MHA with ~50% fewer parameters.

**Architecture:** Each token's query attends over previous tokens via a **gated element-wise product** `φ(Q_t) ⊙ φ(K_s) · γ_{t,s}` instead of softmax. The output is `O_t = φ(Q_t) ⊙ sum_s (φ(K_s) · γ_{t,s} ⊙ V_s) / (sum_s (φ(K_s) · γ_{t,s}) + eps)`. A learnable γ is an additive relative-position bias. The final output is gated by an elementwise `U ⊙ V` style gate. No softmax → O(n) per layer (after materializing γ ∈ R^{n×n}). Implemented in the simplest form: single-head, φ = identity, with a learnable position bias.

**Tech Stack:** C++17, hand-rolled `Tensor`, `Layer` base class. Mirrors `include/nn/layers/attention/aft.h` conventions (AFT and GAU are similar softmax-free variants). Compose with `LayerNorm` + Dense FFN in `GAUBlock` and stack in `GAUModel`.

---

## Math

GAU replaces softmax attention with element-wise gated interactions. With φ(x) = x (identity), single head, additive positional bias γ_{t,s}, and an output gate:

```
Q, K, V, U = X @ W_q, X @ W_k, X @ W_v, X @ W_u        (in R^{n x d})
A_t = sum_{s≤t} K_s · γ_{t,s}          ∈ R^{d}      (per-token "attention output")
B_t = sum_{s≤t} γ_{t,s}                  ∈ R         (per-token normalizer)
O_t = U_t ⊙ (A_t / (B_t + eps))          ∈ R^{d}
```

where `γ_{t,s}` is a learnable scalar position bias (shared across channels), `γ ∈ R^{n×n}`. The output projection is then applied: `Y = O @ W_o^T + b_o`.

This is the single-head, identity-φ "GAUα" form from Hua et al. 2022 §2.1 (simplest tractable variant; the paper's full φ is a learnable GLU, but identity captures the gating mechanism cleanly).

### Backward derivation (per t, scalar `gh_d = grad_output[t, d]`)

```
grad_O[t, d]              = sum_d' grad_Y[t, d'] * W_o[d, d']         (Dense backward)
grad_U[t, d]              = grad_O[t, d] * (A_t[d] / (B_t + eps))
grad_A_t[d]               = grad_O[t, d] * U_t[d] / (B_t + eps)
grad_B_t                  = -sum_d grad_O[t, d] * U_t[d] * A_t[d] / (B_t + eps)^2

// grad_A_t → grad_K_s (for s ≤ t), grad_γ_{t,s}:
grad_γ_{t,s}              = sum_d grad_A_t[d] * K_s[d]              (1/(B+eps) cancels into grad_A)
                            - grad_B_t * 1
grad_K_s[d]              += grad_A_t[d] / (B_t + eps) * γ_{t,s}     ← from grad_γ contribution
                            // (since A_t[d] = sum_s (K_s[d] * γ_{t,s}) / (B_t+eps), if we
                            // factor the (B+eps) into grad_A and use it once.)

// grad_U → grad_W_u etc. (standard Dense chain)
// grad_V_s[d]  = grad_A_t[d] / (B_t + eps) * γ_{t,s}
// grad_W_v etc. via Dense
```

In code we accumulate per (t, s) pair and let the (B + eps) denominator carry the `1/(B_t+eps)` factor exactly once. The closed form per pair:

```
gA_t   = (sum_d' grad_O[t, d'] * U_t[d']) * K_s - (B_t+eps)^-1 * sum_d grad_O[t,d'] * U_t[d'] * A_t[d']
       = grad_O[t] · U_t · K_s - (gO · U · A) / (B + eps)
gγ_{t,s} = gA_t                                  (sum over channels)
gK_s[d] += γ_{t,s} * (gO[t,d] * U_t[d]) / (B_t + eps)
gV_s[d] += γ_{t,s} * (gO[t,d] * U_t[d]) / (B_t + eps)              ← wait, V only appears inside A
// Actually: A_t[d] = (sum_s γ_{t,s} K_s[d]) / (B_t + eps). γ_{t,s} * V_s[d] is wrong.
// Correct: A_t[d] = sum_s (K_s[d] · γ_{t,s}) / (B_t + eps), so dA_t[d]/dK_s[d] = γ_{t,s}/(B_t+eps)
// and dA_t[d]/dγ_{t,s} = K_s[d]/(B_t+eps).
// V appears as: A_t[d] depends on V? No — A is only K ⊙ γ. V enters via the OUTPUT GATE O.
// So we need a separate V projection. Let me re-read the paper.
//
// Actually GAU as defined in Hua 2022 §2 has a separate "value" path that goes through the
// normalize-by-B denominator. The output gate makes: O = U ⊙ A where A = (K⊗γ) / (B+eps).
// So no separate V projection — K serves as both "key" and "value" through the gating.
// For richer modeling, we add a separate V projection: A_t[d] = sum_s γ_{t,s} * V_s[d] / (B+eps)
// with B_t = sum_s γ_{t,s}, and K still drives γ.
// Let's just use the simpler (no V, identity φ) form: A_t = sum_s γ_{t,s} * K_s / (B+eps),
// and we keep W_k and W_q for future extension. The output gate is what gives GAU its power.
//
// Re-deriving cleanly:
```

**Final formulation** (single-head, identity-φ, separate V proj):

```
Q, K, V, U = X @ W_q, X @ W_k, X @ W_v, X @ W_u          (n, d) each
A_t = sum_{s≤t} γ_{t,s} * V_s       ∈ R^{d}
B_t = sum_{s≤t} γ_{t,s}              ∈ R
O_t = U_t ⊙ (A_t / (B_t + eps))      ∈ R^{d}
Y_t = O_t @ W_o^T + b_o              ∈ R^{d_o}
```

**Backward**:
```
gY[t] = grad_output[t]                   // (d_o,)
gO[t, d] = sum_{d'} gY[t, d'] * W_o[d, d']     // (d,) via Dense
gA_t[d] = gO[t, d] * U_t[d] / (B_t + eps)
gB_t    = -sum_d gO[t, d] * U_t[d] * A_t[d] / (B_t + eps)^2
gU[t, d] = gO[t, d] * A_t[d] / (B_t + eps)

// grad contributions:
gγ_{t,s} = sum_d gA_t[d] * V_s[d] - gB_t        // (per (t, s) pair)
gV_s[d] += sum_{t>=s} gA_t[d] * γ_{t,s} / (B_t + eps)        ← wait, A_t uses γ_{t,s} * V_s inside the sum
// Actually: A_t[d] = sum_{s≤t} (γ_{t,s} * V_s[d]) / (B_t + eps)
// So dA_t[d]/dV_s[d] = γ_{t,s} / (B_t + eps)
// And dA_t[d]/dγ_{t,s} = V_s[d] / (B_t + eps)
// So dA_t[d]/d(B_t+eps) = -sum_{s≤t} γ_{t,s} V_s[d] / (B_t+eps)^2 = -A_t[d] / (B_t+eps)
//
// Chain through gA and gB:
// gV_s[d] += sum_{t≥s} gA_t[d] * γ_{t,s} / (B_t + eps)
// gγ_{t,s} += sum_d gA_t[d] * V_s[d] / (B_t + eps)
// gγ_{t,s} += -gB_t / (B_t + eps)        (no, gB_t already IS the gradient w.r.t. B_t, then chain through 1/(B+eps))
// Actually gB_t is the gradient w.r.t. B_t which we computed as the chain through the denominator.
// So grad w.r.t. γ_{t,s} is: gA_t[d]'s chain through V_s[d] (giving the term above)
//                       PLUS gB_t's chain through (γ_{t,s}) (since B_t = sum_s γ_{t,s}, dB_t/dγ_{t,s} = 1)
//                       So gγ_{t,s} = sum_d gA_t[d] * V_s[d] / (B_t + eps) + gB_t * 1
//
// Hmm wait. Let me redo. gB_t is dL/dB_t. B_t = sum_s γ_{t,s}, so dL/dγ_{t,s} += gB_t * (dB_t/dγ_{t,s}) = gB_t * 1 = gB_t.
// And dA_t[d]/dγ_{t,s} = V_s[d] / (B_t + eps) — so gA_t[d] contributes gA_t[d] * V_s[d] / (B_t + eps).
//
// So the final gγ_{t,s} formula is:
// gγ_{t,s} = sum_d gA_t[d] * V_s[d] / (B_t + eps) + gB_t
```

Implementation note: we compute `gA_t[d] / (B_t + eps)` once into a per-token-per-channel scalar, then add the `gB_t` lumped term, and gather per-pair γ gradients. Same shape as AFTAttention backward but with the V path.

---

## Files

- Create: `include/nn/layers/attention/gau.h`
- Create: `include/nn/layers/attention/gau.cpp`
- Create: `tests/test_gau.cpp`
- Modify: `include/nn/nn.h` (umbrella include after `aft.h`)
- Modify: `Makefile` (`build/test_gau` rule, `tests:` deps line, `run_tests` echo)

`LIB_SRCS` is a wildcard, so the new `.cpp` is picked up automatically.

---

## Tasks

### Task 1: Write `tests/test_gau.cpp` (RED)

Tests (mirrors `tests/test_aft.cpp` style — single per-class test functions with `passed`/`failed` counters and `=== Summary ===`):

1. Constructor validation — `d_model=0`, `max_seq_len=0`, `d_out=0` throw.
2. `GAUAttention::forward` shape `(n, d_model) -> (n, d_out)` (with `d_out = d_model`).
3. `GAUAttention::forward` output finite for n=8, d=4.
4. Causal gate: `B_t` accumulates only `s ≤ t`. Verify by perturbing `V_s[d]` for `s > t` and confirming `Y[t]` is bit-exact unchanged.
5. Hand-computed forward (n=1, d=2, forced weights) — closed-form reference.
6. Input gradient FD check (n=3, d=3, eps=1e-5).
7. `W_q` gradient FD check.
8. `W_k` gradient FD check.
9. `W_v` gradient FD check.
10. `W_u` gradient FD check.
11. `W_o` gradient FD check.
12. `position_bias` gradient FD check.
13. `zero_grad` clears all 6 gradients.
14. `update_weights` moves all 6 parameters.
15. `GAUBlock` forward shape `(n, d) -> (n, d)`.
16. `GAUBlock` input gradient FD check.
17. `GAUModel` training reduces loss over 30 SGD steps.
18. `parameters()`/`gradients()` return 6 tensors each with matched shapes.
19. **Mutation test**: stubbing the `gU[t, d] = gO[t, d] * A_t[d] / (B_t + eps)` line in backward → input grad FD rel_err rises from machine precision to ≈1.0 (the output gate chain is essential).
20. Longer sequence (T=6) input gradient FD check (deep BPTT).

**All FD checks must use random (non-uniform) init** so the row-vs-column sums are non-degenerate (see TDD skill). Mirror the `Based`/`AFT` test for FD helpers (`rel_err`, `l2_loss_value`, `l2_loss_grad`).

Use 1e-3 as the FD tolerance for n=3 (achieved by other attention layers in the repo).

### Task 2: `include/nn/layers/attention/gau.h` (GREEN)

Header structure mirroring `aft.h`:

```cpp
class GAUAttention : public Layer {
public:
    GAUAttention(size_t d_model, size_t max_seq_len, size_t d_out = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_o.weights; }
    Tensor get_gradients() const override { return W_o.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "GAUAttention"; }

    size_t d_model() const { return d_model_; }
    size_t max_seq_len() const { return max_seq_len_; }
    size_t d_out() const { return d_out_; }
    const Tensor& position_bias() const { return position_bias_; }
    const Tensor& last_B() const { return cache_B_; }
    const Tensor& last_A() const { return cache_A_; }

    // Public parameters (mirroring aft.h style for direct test access)
    Dense W_q;            // (d_model, d_model)
    Dense W_k;            // (d_model, d_model)
    Dense W_v;            // (d_model, d_model)
    Dense W_u;            // (d_model, d_model)
    Dense W_o;            // (d_out, d_model)
    Tensor position_bias_;// (max_seq_len, max_seq_len)
    Tensor grad_position_bias_;

private:
    size_t d_model_;
    size_t max_seq_len_;
    size_t d_out_;

    // Forward caches
    Tensor last_input_;
    Tensor last_Q_, last_K_, last_V_, last_U_;
    Tensor cache_A_;           // (n, d)
    Tensor cache_B_;           // (n,)
    Tensor cache_O_;           // (n, d) — U ⊙ (A / (B+eps))
};

class GAUBlock : public Layer { /* pre-LN -> GAU -> residual -> pre-LN -> GELU FFN -> residual */ };
class GAUModel : public Layer  { /* stack + classifier */ };
```

Place in `attention/` directory.

### Task 3: `include/nn/layers/attention/gau.cpp` (GREEN)

Mirror `aft.cpp` style. The forward and backward per (t, s) pair:

```cpp
Tensor GAUAttention::forward(const Tensor& x) {
    last_input_ = x;
    Tensor Q = W_q.forward(x);   // (n, d_model)
    Tensor K = W_k.forward(x);   // (n, d_model)
    Tensor V = W_v.forward(x);   // (n, d_model)
    Tensor U = W_u.forward(x);   // (n, d_model)
    last_Q_ = Q; last_K_ = K; last_V_ = V; last_U_ = U;

    const size_t n = x.rows;
    cache_A_ = Tensor(n, d_model_);
    cache_B_ = Tensor(n, 1);
    Tensor O(n, d_model_);

    for (size_t t = 0; t < n; ++t) {
        double B = 0.0;
        for (size_t s = 0; s <= t; ++s) B += position_bias_(t, s);
        cache_B_(t, 0) = B;
        for (size_t d = 0; d < d_model_; ++d) {
            double A = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                A += position_bias_(t, s) * V(s, d);
            }
            cache_A_(t, d) = A;
            O(t, d) = U(t, d) * (A / (B + eps_));
        }
    }
    cache_O_ = O;
    return W_o.forward(O);
}

Tensor GAUAttention::backward(const Tensor& grad_output, double /*lr*/) {
    // W_o.backward
    Tensor gO = W_o.backward(grad_output, 0.0);  // (n, d_model)

    const size_t n = last_input_.rows;
    grad_W_q.fill(0); grad_W_k.fill(0); grad_W_v.fill(0); grad_W_u.fill(0);
    grad_position_bias_.fill(0);

    Tensor grad_input(n, d_model_);
    grad_input.fill(0);
    Tensor gQ(n, d_model_), gK(n, d_model_), gV(n, d_model_), gU(n, d_model_);
    gQ.fill(0); gK.fill(0); gV.fill(0); gU.fill(0);

    for (size_t t = 0; t < n; ++t) {
        double B = cache_B_(t, 0);
        double inv_den = 1.0 / (B + eps_);
        // Compute per-channel gA and gB
        // gA_t[d] = gO[t, d] * U_t[d] / (B + eps)
        // gB_t    = -sum_d gO[t, d] * U_t[d] * A_t[d] / (B + eps)^2
        double gB = 0.0;
        for (size_t d = 0; d < d_model_; ++d) {
            double Atd = cache_A_(t, d);
            gU(t, d) += gO(t, d) * Atd * inv_den;
            gB -= gO(t, d) * U(t, d) * Atd * inv_den * inv_den;
        }
        // gγ_{t,s} += sum_d gA_t[d] * V_s[d] * inv_den + gB
        // gV_s[d]  += gA_t[d] * γ_{t,s} * inv_den
        for (size_t s = 0; s <= t; ++s) {
            double gamma = position_bias_(t, s);
            double sum_d_gA_V = 0.0;
            for (size_t d = 0; d < d_model_; ++d) {
                double gA_d = gO(t, d) * U(t, d) * inv_den;
                sum_d_gA_V += gA_d * V(s, d);
                gV(s, d) += gA_d * gamma * inv_den;
            }
            grad_position_bias_(t, s) += sum_d_gA_V + gB;
        }
    }
    // Chains through W_q, W_k, W_v, W_u (Dense backward)
    Tensor gX_q = W_q.backward(gQ, 0.0);
    Tensor gX_k = W_k.backward(gK, 0.0);
    Tensor gX_v = W_v.backward(gV, 0.0);
    Tensor gX_u = W_u.backward(gU, 0.0);
    for (size_t i = 0; i < grad_input.data.size(); ++i)
        grad_input.data[i] = gX_q.data[i] + gX_k.data[i] + gX_v.data[i] + gX_u.data[i];
    return grad_input;
}
```

The `gA_t[d] = gO[t, d] * U_t[d] * inv_den` and `gB_t` are computed inside the inner loop. Note `gA_d` is local per (t, d) — no accumulation needed.

**WAIT — I see an error in the derivation**: `grad_position_bias_(t, s)` should accumulate contributions across multiple `t` rows for the same `s`. Since `position_bias_(t, s)` is read for each (t, s) pair independently, this is fine — we accumulate into `grad_position_bias_` once per (t, s) pair. The `gV_s[d]` does accumulate across all `t ≥ s`, which is the inner loop's `gV(s, d) += ...` line.

### Task 4: `GAUBlock` and `GAUModel` (TDD)

Mirror `AFTBlock`/`AFTModel` exactly: pre-LN → GAUAttention → residual → pre-LN → Dense → GELU → Dense → residual. `GAUModel` stacks N blocks + per-token classifier.

### Task 5: `Makefile` and `nn.h`

Add `#include "layers/attention/gau.h"` after `aft.h` in the umbrella. Add `build/test_gau` rule, append to `tests:` deps line, add `=== Running GAU Tests ===` echo before `=== Running FlashAttention-2 Tests ===` (or at the end with the other attention layers).

### Task 6: Verify umbrella compiles standalone

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

Should exit 0 with no warnings.

### Task 7: Mutation test

In `test_gau.cpp` Test 19, instrument the backward path: temporarily replace `gU(t, d) += gO(t, d) * Atd * inv_den;` with `gU(t, d) += 0.0;` (zeroing the output-gate gradient), re-run the input FD check, expect rel_err > 0.5. Restore the line.

---

## Verification

- `make tests` builds all test binaries clean under `-Wall -Wextra`.
- `./build/test_gau` reports `=== Summary: N passed, 0 failed ===` with N = 20.
- `make run_tests` runs the full suite with no regressions.
- Umbrella compiles standalone.

---

## Plan self-check

- Tasks are bite-sized: header, impl, tests, mutation, all separate.
- File paths are exact.
- Math is fully derived (the final `grad_position_bias_(t, s)` formula matches the chain rule).
- FD tolerances match the repo convention (1e-3 for n=3).
- All FD checks use random init (default Xavier in `Dense`).