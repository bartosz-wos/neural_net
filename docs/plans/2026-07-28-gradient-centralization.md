# Gradient Centralization (GC) Implementation Plan

> **For Hermes:** Implement using TDD per task.

**Goal:** Implement Gradient Centralization (Yu et al. 2020) — a per-parameter optimizer wrapper that centers gradients by subtracting their mean before passing them to the inner optimizer. Improves generalization and convergence without hyperparameter overhead.

**Architecture:** `GradientCentralization` is an `Optimizer` wrapper (similar to `WeightDecay`, `Lookahead`, `SAM`). It modifies gradients in-place on each parameter before delegating to the inner optimizer. GC is multiplicative-free, state-free, and works with any inner optimizer.

**Tech Stack:** C++17, existing `Optimizer`/`Model`/`Tensor` interfaces. No new dependencies.

**Paper:** Yu et al. 2020 "Gradient Centralization: A New Optimization Strategy for Deep Neural Networks" (https://arxiv.org/abs/2004.01461).

**Key math (per-parameter, applied to grad tensor before inner step):**
- For 2-D weight matrix of shape (R, C): `grad[i][j] -= mean(grad[i][*])` over each row, OR over each column. The paper uses row-wise mean for Conv (per filter) and column-wise mean for Dense (per output). We implement BOTH and let caller pick via `CenterMode` enum.
- For 1-D bias (shape (1, C)): column-wise mean is the only natural choice (each output feature is one column).
- After centering, the L2 norm of the gradient shrinks; the inner optimizer takes a smaller step. Counter-intuitively, this works better in practice.

**Reference implementation:**
```cpp
// Yu et al. 2020, Algorithm 1
// For each weight tensor W with gradient G:
//   if 2-D:  G -= mean(G, axis=mode)  broadcast back
//   else:    G -= mean(G)            scalar broadcast
// Then call inner.step(model).
```

**Wrapper pattern (mirrors `WeightDecay`):**
```cpp
class GradientCentralization : public Optimizer {
public:
    enum class CenterMode { ROW, COLUMN };
    GradientCentralization(Optimizer* inner, CenterMode mode = CenterMode::COLUMN);
    void step(Model& model) override;
    bool handles_weight_decay() const override { return inner_->handles_weight_decay(); }
    Optimizer* inner() const;
    CenterMode mode() const;
private:
    std::unique_ptr<Optimizer> inner_;
    CenterMode mode_;
};
```

**Why COLUMN default:** Dense `(out_features, in_features)` — each output column is one neuron's incoming weights. Centering per column keeps the per-output feature's weight vector centered. This is the natural choice for the project's existing Dense layer convention.

**Validation:** non-null inner; throws `std::invalid_argument` on null.

**Tests:** 35+ checks covering:
- Defaults (lr=0.001, mode=COLUMN, inner=non-null)
- Constructor validation (null inner throws)
- Zero gradient → no-op (params unchanged, centered grad still zero)
- Constant gradient → after centering becomes all-zero (mean exactly = g → grad -= g → 0)
- Asymmetric gradient for ROW mode: row-wise mean subtraction (test 2-row matrix, each row centered to sum 0)
- Asymmetric gradient for COLUMN mode: col-wise mean subtraction (test 2-col matrix, each col centered to sum 0)
- Inner optimizer is called exactly once per step
- Inner optimizer sees the CENTERED gradient (not the raw gradient)
- After step, gradients are zeroed (delegated to inner)
- `handles_weight_decay()` delegates to inner (true for AdamW, false for SGD)
- `inner()` and `mode()` accessors return correct values
- `set_mode()` toggle works (column → row)
- Multi-layer model: every layer's gradient is centered before inner.step
- GC(Adam) reduces loss on linear regression (y=2x) by meaningful amount
- Sign: GC of constant positive gradient leaves zero gradient (so inner.step applies ZERO update → params unchanged)
- Determinism: two fresh instances produce bit-identical result
- GC of all-zero gradient is still all-zero (no NaN)
- Backward-compat: GC(GC(Adam)) nested (not strictly required but allowed)
- GC(AdamW) with weight decay chain still functions: inner applies its own wd AND the centering

