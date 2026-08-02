# StableAdamW Optimizer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a StableAdamW (Wortsman et al. 2024) optimizer to the repo — AdamW with **update clipping** instead of gradient clipping, designed for stable low-precision (bf16) training of large-scale vision-language models.

**Architecture:** A standard Adam-class optimizer (per-parameter m and v state) that adds a **post-update-direction** clip to ±1 BEFORE applying the learning rate. This means the *per-coordinate* update is bounded by `lr * (1 + |wd*param|)`, regardless of how large the gradient gets. Crucially: it removes the need for gradient-norm clipping (`clip_grad_norm_`) entirely.

**Tech Stack:** C++17, Tensor class, follows the Adam/AdamW pattern (similar to `adam_mini.{h,cpp}`).

**Location:** `include/nn/optimizers/stableadamw.{h,cpp}` + `tests/test_stableadamw.cpp` + Makefile + umbrella header.

---

## Algorithm reference (Wortsman et al. 2024, "Stable and Low-Precision Training for Large-Scale Vision-Language Models")

### Per-step, per-parameter (Algorithm 1, "StableAdamW")

```
g_t         = grad_t                                           (current gradient)
m_t         = β1 * m_{t-1} + (1-β1) * g_t                     (first-moment EMA)
v_t         = β2 * v_{t-1} + (1-β2) * g_t²                    (second-moment EMA)

m_hat_t     = m_t / (1 - β1^t)                                 (bias correction)
v_hat_t     = v_t / (1 - β2^t)                                 (bias correction)

update_t    = m_hat_t / (sqrt(v_hat_t) + ε)                   (Adam direction)
update_t    = clip(update_t, -1, +1)                          ← KEY: update clipping (not grad clipping)

if (decoupled_weight_decay):  θ *= (1 - lr * wd)              (AdamW-style decoupled WD)
θ          -= lr * update_t                                    (apply clipped update)
```

### Defaults (paper + optimi reference implementation)

- `lr = 1e-3` — same as Adam
- `β1 = 0.9, β2 = 0.999`
- `ε = 1e-8`
- `weight_decay = 0.01` — AdamW-equivalent decoupled WD (paper recommends 0.01-0.1)
- `decoupled = true` — AdamW-style decoupled WD (paper default)

### Why update clipping instead of gradient clipping?

In low-precision (bf16) training, large gradients lose precision when stored in bf16 (only 8 bits of mantissa). The standard practice of `clip_grad_norm_(g, max_norm)` then DIVIDES the gradient by `||g||` which is itself an imprecise value, propagating large error into every parameter's update. StableAdamW instead lets the Adam direction be computed in float32, and **caps the per-coordinate update at ±1**, so the learning rate `lr` is the only place where scale matters. The bf16 quantization (or any subsequent low-precision cast) only sees values bounded by `[-1, 1]`, which it can represent well.

### Validation (mirrors optimi reference)

- `lr > 0`
- `0 ≤ β1 < 1`, `0 ≤ β2 < 1`
- `ε > 0`
- `weight_decay ≥ 0`

### State per parameter (lazy on first step)

- `m ∈ R^{shape}` — first-moment EMA
- `v ∈ R^{shape}` — second-moment EMA
- `t` — global step counter (shared across all params; starts at 1)

---

## Public API

```cpp
class StableAdamW : public Optimizer {
public:
    double lr;            // base learning rate (default 1e-3)
    double beta1;         // first-moment EMA (default 0.9)
    double beta2;         // second-moment EMA (default 0.999)
    double epsilon;       // numerical stabilizer in denom (default 1e-8)
    double weight_decay;  // decoupled WD (default 0.01)
    int    t;             // global step counter (default 1)

    explicit StableAdamW(double lr_ = 1e-3,
                         double b1 = 0.9,
                         double b2 = 0.999,
                         double eps = 1e-8,
                         double wd = 0.01);

    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);

    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_epsilon() const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t() const { return t; }

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // State accessors
    bool has_state(void* layer_ptr) const;
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_v(void* layer_ptr, size_t param_idx) const;
private:
    struct ParamState { Tensor m, v; };
    std::map<void*, std::vector<ParamState>> state_;
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};
```

---

## Test plan (mirrors the test patterns in `test_adam_mini.cpp`, `test_adamp.cpp`)

### Coverage targets (≥80 focused checks, machine precision where applicable)

1. **Defaults round-trip** (12 checks): `lr=1e-3, β1=0.9, β2=0.999, ε=1e-8, wd=0.01, t=1, handles_weight_decay=true`
2. **Non-default constructor** (5 attrs)
3. **Validated setters throw** on: `lr<0` (and lr=0 OK for shutdown? — paper says lr>0), `β1<0`, `β1≥1`, `β2<0`, `β2≥1`, `ε≤0`, `wd<0` (≥12 invalid inputs)
4. **Constructor-time validation** (3 invalid inputs throw)
5. **Round-trip setters** (5 attrs)
6. **Lazy state init** (no state before step, state after step, t increments 1→2→3→4)
7. **State shape correctness**: Dense(3,4) weights (4,3) → m (4,3), v (4,3); bias (1,4) → m (1,4), v (1,4)
8. **Closed-form first step** on a hand-derived Dense(2,2) example — verify the **clip** is applied
9. **Update clipping IS active** for a hand-constructed large-update case (e.g. grad=1000, v=1e-6, β2=0 → update = 1000/1e-3 = 1e6 → clipped to 1.0, step = lr * 1.0)
10. **Update clipping NOT triggered** for small updates (rel_err 0 vs unclipped Adam)
11. **Decoupled weight decay**: param *= (1-lr*wd) BEFORE update, then param -= lr * clipped_update
12. **Zero weight decay**: identical to AdamW math (modulo the clip)
13. **End-to-end y=2x regression** reduces loss >50% over 100 steps
14. **Determinism** (bit-identical across 5 steps with same seed/init)
15. **Signature vs Adam**: when both updates are well within [-1, 1], produces identical params (clipped = unclipped); when large, StableAdamW diverges (trajectory is bounded)
16. **Empty model doesn't crash**
17. **Multi-layer model**: each layer tracked independently
18. **Stability under large gradients**: grad=1e10 → update still bounded at ±1, param never explodes
19. **Step counter increments**: t=1→2→3→4

### Non-vacuousness (mutation tests)

After all 80 checks pass, run 2 mutations and confirm failures:
- **Mutation 1**: Stub out the `update = clip(update, -1, 1)` line → expect: (a) closed-form test 8 fails (large update was 1e6, should have been 1.0), (b) stability test 18 fails (param explodes without clip)
- **Mutation 2**: Stub out the `m_hat = m / bc1` bias correction → expect: closed-form test 8 fails (bc1 ≠ 1 at t=1, β1=0.5)

---

## Build integration

1. Create `include/nn/optimizers/stableadamw.{h,cpp}`
2. Add `#include "optimizers/stableadamw.h"` to `include/nn/nn.h` umbrella
3. Add `$(BUILD_DIR)/test_stableadamw: $(LIB_OBJS) $(BUILD_DIR)/test_stableadamw.o` to Makefile
4. Add `$(BUILD_DIR)/test_stableadamw \` to the `tests:` list
5. Add `@echo "=== Running StableAdamW Tests ===" && ./$(BUILD_DIR)/test_stableadamw` to `run_tests:`

## Commit strategy

Single commit: `feat(optimizers): add StableAdamW (Wortsman 2024) — AdamW with update clipping`
