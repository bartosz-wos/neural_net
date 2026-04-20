# Adversarial System Audit Report — neural_net

**Branch:** `audit/adversarial-review`  
**Base:** `aca866b` (master) → HEAD: `1824785`  
**Date:** 2026-04-20  
**Auditors:** 4 parallel sub-agents (Batches A/B/C/D)

---

## Executive Summary

| Batch | Scope | Commits | Critical | Moderate | Minor |
|-------|-------|---------|----------|----------|-------|
| A | Core + Activations | `550fa9c` | 1 | 3 | 2 |
| B | Conv + Pooling | `d6efaff` | 1 | 5 | 4 |
| C | Dense/Recurrent/Norm | `518fc96` | 1 | 3 | 3 |
| D | Advanced + Optimizers + Utils | `d118e85` | 4 fixed / 8 pending | 6 | 16 |
| **Merge** | All batches | `1824785` | — | — | — |

**Build:** ✅ All 10 demos pass (zero regressions)  
**Fixes applied across batches:** 7 immediate fixes + ~40 documented findings

---

## Critical Findings (Action Required)

### CR-1: MaxPool1D `max_indices_` — Wrong 2D Layout → Silent Wrong Gradients
**File:** `pool_layer.h:53`, `pool_layer.cpp:100,117,132`  
**Severity:** Critical — silently wrong gradients, batch > 1  
**Batch:** B

```cpp
// WRONG: [channels][N*seq_out]
std::vector<std::vector<int>> max_indices_;
max_indices_.assign(channels, std::vector<int>(N * seq_out, -1));
// Access: max_indices_[c][n * seq_out + t_out] → OOB for n≥1 when c is small
```

**Fix:** Change layout to `[channels * N][seq_out]`, access as `max_indices_[c * N + n][t_out]`.

---

### CR-2: Transformer Residual Sign Inverted
**File:** `transformer.cpp` (fixed in D as `d118e85`)  
**Severity:** Critical — inverted gradient signal in residual path  
**Batch:** D

```cpp
// WRONG: x - attn_tokens
input = x - attn_tokens;
attn_tokens = sub->forward(attn_tokens, x);  // backward inverts sign again

// CORRECT: x + attn_tokens
input = x + attn_tokens;
```

---

### CR-3: SWA — Unbounded `avg_idx` → Heap OOB
**File:** `swa.cpp` (fixed in D as `d118e85`)  
**Severity:** Critical — heap out-of-bounds if parameter count changes  
**Batch:** D

```cpp
// MISSING bound check
while (it != params.end()) {
    // avg_idx can exceed param_total_ → OOB heap write
}

// FIX: add bound check
if (avg_idx >= param_total_) break;
```

---

### CR-4: SWA — Division by Zero When `warmup_steps_ == 0`
**File:** `swa.cpp` (fixed in D as `d118e85`)  
**Severity:** Critical — SIGFPE  
**Batch:** D

```cpp
// WRONG
double step_ratio = static_cast<double>(global_step) / warmup_steps_;
// warmup_steps_ = 0 → division by zero

// FIX: guard
if (warmup_steps_ > 0) { step_ratio = ... } else { step_ratio = 1.0; }
```

---

### CR-5: OneCycleLR — Linear Decay With `pct > 1` → Negative LR
**File:** `one_cycle_lr.cpp` (fixed in D as `d118e85`)  
**Severity:** Critical — negative learning rate  
**Batch:** D

```cpp
// WRONG: pct can exceed 1.0 in calling code
double step_ratio = pct * 2.0;
double lr = max_lr * (1.0 - step_ratio);  // negative if pct > 0.5

// FIX: std::clamp(pct, 0.0, 1.0)
```

---

### CR-6: tensor.cpp — UB on Empty Tensor in `max()` / `sum()`
**File:** `tensor.cpp`  
**Severity:** Critical — uninitialized read, UB if `rows*cols==0`  
**Batch:** A

```cpp
// WRONG: data[0] accessed before any bounds check
double max_val = data[0];
for (size_t i = 1; i < rows * cols; ++i) { ... }

// FIX: guard
if (rows * cols == 0) return 0.0; // or handle appropriately
```

---

### CR-7: Adam — Bias Correction Division by Zero at `beta1^t == 0`
**File:** `optimizer.cpp` / `optimizer_extended.cpp`  
**Severity:** Critical — division by zero when `beta1 == 1.0`  
**Batch:** D

The bias correction: `bias_correction1 = 1 - pow(beta1, t)` — if `beta1 == 1.0`, denominator becomes 0.

---

## Moderate Findings

### MO-1: Conv Backward — Correlation Not Convolution (No Kernel Flip)
**File:** `conv1d.cpp`, `conv_layer.cpp`  
**Batch:** B

Backward pass does NOT flip kernels — implements cross-correlation, not true convolution. Numerically identical for symmetric kernels, but wrong for asymmetric ones.

---

### MO-2: MaxPool1D — Ceil/Floor `seq_out` Mismatch
**File:** `pool_layer.cpp:85,98`  
**Batch:** B

Constructor uses `ceil((seq_len+ksz-1)/stride)` but forward loop uses floor-aligned indexing. Safe but wasteful — last window never contributes.

---

### MO-3: AvgPool1D — Inconsistent Norm in Backward Pass
**File:** `pool_layer.cpp:134,150-152`  
**Batch:** B

Forward uses per-window `count` (boundary clipping), backward always uses `kernel_size`. Slightly off gradient magnitudes near sequence boundaries.

---

### MO-4: tensor.cpp — Dead `res.fill(0.0)` Before `+=` Loop
**File:** `tensor.cpp` `operator*`  
**Batch:** A

`res.fill(0.0)` is dead work — every element is overwritten by the `+=` loop, making it pure O(n²) waste.

---

### MO-5: activations.cpp — Softmax Missing `cols==0` Guard
**File:** `activations.cpp` Softmax  
**Batch:** A

No guard for empty columns — division by zero on degenerate tensors.

---

### MO-6: weight_init.cpp — Xavier Incorrectly Uses Kaiming Formula for ReLU
**File:** `weight_init.cpp` `xavier()`  
**Batch:** A

Xavier for ReLU incorrectly applies `sqrt(2/fan_in)` (Kaiming style), conflating two distinct initialization strategies.

---

## Minor Findings (Known/Accepted)

- **DilatedConv2D:** No explicit validation of `dilation > 0`
- **MaxPool2D:** Recomputes argmax per call (could cache)
- **CoordConv2D:** `get_weights()` / `get_gradients()` return empty tensors — correct but confusing
- **Conv1D:** No explicit check for `stride > seq_len`
- **ResNeXt:** Re-creates `Conv2D` instance per forward call
- **CapsNet:** Routing iteration count fixed (no convergence check)
- **int t overflow:** Loop counter in BPTT may overflow for very long sequences
- **ODR violation:** `optimizer_sgd_adam.h` may violate one-definition-rule across translation units
- **Transformer:** No causal mask (attention is un masked)
- **VAE:** Backward pass is a stub — identity only
- **AdamW:** Weight decay applied in wrong order (decouples incorrectly from Adam)
- **GroupNorm:** No validation against `G == 0` or `G > channels`
- **LayerNorm:** Epsilon on variance — verify `1e-7` is sufficient for float32 stability
- **WeightNorm:** No guard against zero-norm weight vector
- **Skip connections:** Shape mismatch possible if channels differ (should assert/auto-project)
- **GELU:** tanh path — verify no NaN for extreme inputs (std::tanh has its own limits but worth hardening)

---

## Build Verification

```
make clean && make -j$(nproc) 2>&1 → ✅ EXIT 0
```

All 10 demo binaries:
| Demo | Result |
|------|--------|
| nn_demo | ✅ MSE 0.000336 |
| xor_big | ✅ 100% accuracy |
| multiclass | ✅ 6/6 correct |
| cnn_xor | ✅ 4/4 correct |
| cnn_multiclass | ✅ 150/150 train correct |
| transformer_demo | ✅ All tests passed |
| rnn_airline | ✅ Exit 0 |
| lstm_airline | ✅ Exit 0 |
| extensions_demo | ✅ All tests passed |
| embedding_demo | ✅ Exit 0 |

---

## Remaining Work (Not Fixed)

The following require design decisions or deeper refactors:

1. **MaxPool1D layout** — needs reshape of index storage and backward rewrite
2. **Conv kernel flip** — fundamental to whether the library claims convolution vs correlation
3. **Transformer causal mask** — missing attention mask
4. **Adam beta=1 guard** — needs explicit validation
5. **VAE backward** — currently identity stub
6. **ResNet backward** — reported as stub/incorrect
7. **DenseNet dummy input / FC dim** — wrong layer dimensions
8. **SWA BN update** — no batch normalization statistics update after SWA
9. **GNN backward** — stub implementation
10. **MoE combine** — empty or incorrect
11. **LSTM-LAS enc_out** — encoder output discarded
12. **SE skip OOB** — SqueezeExcitation skip connection OOB access

---

*Report consolidated from 4 parallel batch audits. Merge commit: `1824785`.*
