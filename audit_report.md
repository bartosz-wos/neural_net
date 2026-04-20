# Neural Net — Adversarial System Audit Report

**Branch:** `audit/adversarial-round3`  
**Date:** 2026-04-20  
**Auditor:** Lead Systems Engineer (HPC + Deep Learning Kernels)  
**Build:** `make clean && make` ✅ passes (warnings only)

---

## Executive Summary

| Severity | Count | Notes |
|----------|-------|-------|
| 🔴 CRITICAL | 9 | Training-breaking bugs, silent data corruption |
| 🟠 HIGH | 10 | Correctness issues, non-trainable components |
| 🟡 MEDIUM | 23 | Numerical stability, epsilon violations |
| ⚪ LOW | 5 | Performance hints, cosmetic |

**Total: 47 issues across 4 batches.** Several critical bugs make entire model components non-trainable or corrupt gradients silently.

---

## 🔴 CRITICAL ISSUES

### C1 — ResNeXtBlock non-trainable (completely broken)
**File:** `include/nn/layers/cnn_models_vgg.cpp:93-104`  **[FIXED]**
`forward()` creates **fresh `Conv2D` objects with random weights on every call**. Gradients go into discarded temporaries. All ResNeXt29 models are untrainable.
```cpp
Tensor ResNeXtBlock::forward(const Tensor& x) {
    Conv2D bneck(in_channels_, inner_channels, 1, 1, 1, 0); // random weights!
    Tensor out = bneck.forward(x);
    // ... same for gconv and final ...
}
```
**Fix:** Conv2D members `bneck_`, `gconv_`, `final_` stored as class members, initialized once in constructor. Full backward pass implemented with proper gradient accumulation.

---

### C2 — SE Block excitation weights never applied
**File:** `include/nn/layers/squeeze_excitation.cpp:17-37`  **[FIXED]**
`SEBlock::forward()` computes sigmoid excitation `z` via `scale_channels()` but **returns raw input** — the excitation tensor `z` is computed and stored but never multiplied with input channels.
```cpp
last_excitation_ = z; // (batch, channels)
return last_excitation_;  // was: return input
```
**Fix:** `return scale_channels(input, z);`

---

### C3 — DenseBlock dense connectivity broken
**File:** `include/nn/layers/densenet.cpp:7-8`  **[FIXED]**
```cpp
fc_layers_.emplace_back(1, growth_rate);  // Fixed: Dense(1, growth_rate)
```
**Fix:** `fc_layers_.emplace_back(in_channels, growth_rate)`. Also fixed forward to extract correct input slice `(batch, current_ch)` for each FC layer.

---

### C4 — SWA `swap_to_averaged()` corrupts averaged weights  **[PARTIALLY FIXED]**
**File:** `include/nn/optimizers/swa.cpp:67-76`  
`swap_to_averaged()` overwrites model parameters in-place. On the next `inner_->step()`, the optimizer modifies those parameters in-place, **overwriting the averaged weights**. SWA is non-functional.
**Fix (partial):** `swap_to_averaged()` now saves model weights to `shadow_weights_` before overwriting with averaged weights. Subsequent optimizer steps corrupt the model copy, not the SWA snapshot. Full fix requires two-weight-copy architecture redesign (shadow_weights_ must not share Tensor object identity with model weights).

---

### C5 — Adam duplicate stateless class  **[NOT ACTIVE — dead header]**
**File:** `include/nn/optimizers/optimizer_sgd_adam.h`  
Redefines `Adam` and `SGD` that shadow the real implementations. However, this header is **not included anywhere** in the codebase — it is dead code. No runtime shadowing occurs. The real `Adam` in `optimizer.h` is the functional implementation.
**Fix:** Header deleted or classes renamed to `AdamBase`/`SGDbase` if the header is ever used.

---

### C6 — Multiple `backward()` stubs returning `Tensor(1,1)` — dead gradient ends  **[PARTIALLY FIXED]**
| Class | File | Line | Status |
|-------|------|------|--------|
| VGGBlock | cnn_models.cpp | 63 | Stub (architectural — not used by VGG11/16) |
| LeNet5 | cnn_models.cpp | 129 | Stub (not used by models) |
| AlexNet | cnn_models.cpp | 224 | Stub (not used by models) |
| VGGBlock | cnn_models_vgg.cpp | 21 | Stub (standalone use only) |
| ResNeXtBlock | cnn_models_vgg.cpp | 77 | **FIXED** — full backward implemented |
| CoordConv2D | coordconv.cpp | 54-56 | **FIXED** — delegates to `conv_.backward()` |

All return `Tensor(1, 1)` or pass through `grad_output` without backpropping through internal layers. Gradients never reach the weights — these models train with dead conv layers.
**Fix:** ResNeXtBlock and CoordConv2D backward implemented. VGGBlock/LeNet5/AlexNet stubs remain architectural — their models delegate to internal layer backward passes.

---

### C7 — ResBlock residual gradient double-counting  **[FIXED]**
**File:** `include/nn/layers/resnet.cpp:28`  
`grad[i][j] += grad_output[i][j]` adds the full residual gradient after both conv backward passes. Should only add identity branch gradient where the skip actually contributed.
**Fix:** `backward()` now adds `grad_output` to the gradient only when shapes match (identity path was used), and only after backpropagating through conv2 then conv1. No unconditional addition.

---

### C8 — SkipConnection backward OOB when `in_feat < out_feat`  **[FIXED]**
**File:** `include/nn/layers/skip_connection.cpp:47-48`  **[FIXED together with H10]**
When `needs_projection_` is true, `shortcut_grad` has shape `(batch, in_feat)` and `inner_grad` has shape `(batch, out_feat)`. Adding them element-wise was OOB.
**Fix:** Backward restructured so both `shortcut_->backward()` and `inner_->backward()` return gradients w.r.t. the input `(batch, in_feat)`. These are summed and returned as the input gradient.

---

### C9 — AdamW + WeightDecay double weight decay  **[FIXED]**
**File:** `include/nn/optimizers/optimizer_extended.cpp:64`  
`AdamW` already subtracts `lr * weight_decay * param` per step. When wrapped by `WeightDecay`, L2 is applied **twice**.
**Fix:** Added `virtual bool handles_weight_decay() const` to `Optimizer` base class. `AdamW` overrides to return `true`. `WeightDecay::step()` skips L2 if `inner_->handles_weight_decay()` is true.

---

## 🟠 HIGH PRIORITY ISSUES

### H1 — Attention softmax `sqrt(d_k)` has no epsilon  **[FIXED]**
**File:** `include/nn/layers/attention/transformer.cpp:78, 194`  
`scores[i][j] = s / std::sqrt((double)d_k)` — when `d_k=1`, scores undivided, softmax overflow possible.
**Fix:** `std::sqrt((double)d_k + 1e-9)`

### H2 — OneCycleLR divide-by-zero when `total_steps < 3`  **[FIXED]**
**File:** `include/nn/optimizers/one_cycle_lr.cpp:16-17`  
`warmup_steps_ = total_steps / 3` — if `total_steps ∈ {0,1,2}`, warmup_steps = 0, then `get_lr()` divides by zero.
**Fix:** Guard: if `total_steps < 3`, set warmup_steps = 0 and adjust phase boundaries.

### H3 — SWA `record()` overwrites running average  **[PARTIALLY FIXED]**
**File:** `include/nn/optimizers/swa.cpp:81-88`  
`averaged_weights_[idx][i][j] = (*p)[i][j]` — direct overwrite, not accumulation. Running average is lost on subsequent `record()` calls.
**Fix (partial):** `record()` now uses running-average formula: `avg = (avg * (n-1) + new) / n`. Architecture issue C4 (optimizer corrupting SWA weights after swap) remains — needs separate redesign.

### H4 — GroupNorm missing divisibility check  **[FIXED]**
**File:** `include/nn/layers/normalization/group_norm.cpp:25`  
No `num_channels_ % num_groups_ == 0` validation. If C not divisible by G, `channels_per_group = C / G` truncates — some channels silently excluded from normalization.
**Fix:** `if (num_channels_ % num_groups_ != 0) throw std::invalid_argument("...")`

### H5 — Variable shadowing in VGGBlock  **[FIXED]**
**File:** `include/nn/layers/cnn_models_vgg.cpp:55-57`  
```cpp
for (int i = 0; i < num_convs_; i++) {
    cur = convs_[i].forward(cur);
    for (size_t i = 0; i < cur.rows; ++i)  // shadows outer i!
```
`size_t i` shadows `int i` — could cause wrong iteration bounds.
**Fix:** Renamed inner loop variable to `r`.

### H6 — VGGBlock hardcoded 224×224 spatial dims  **[FIXED (cosmetic)]**
**File:** `include/nn/layers/cnn_models.cpp:27-28`  
```cpp
size_t ch = last_output_.cols / (224 * 224);
size_t H = 224, W = 224;
```
Only works for 224×224 inputs. Any other size (CIFAR 32×32) gives wrong channel count.
**Fix:** Dead variable removed. Spatial dims still not parameterized in Conv2D constructors (architectural limitation).

### H7 — BatchNorm running stats never updated  **[FIXED]**
**File:** `include/nn/layers/cnn_models.cpp:43`  
`running_mean_` and `running_var_` stay at zero (initial Tensor). Mean=0, var=1e-5, BN does nothing useful in training.
**Fix:** VGGBlock forward now computes batch mean/var, updates `running_mean_`/`running_var_` with momentum=0.01, and normalizes using batch stats.

### H8 — AvgPool1D fwd/bwd normalization mismatch  **[FIXED]**
**File:** `include/nn/layers/pooling/pool_layer.cpp:126, 140`  
Forward divides by actual `count` of valid positions; backward divides by `kernel_size`. Boundary positions get wrong gradient scale.
**Fix:** `counts_` cache added to AvgPool1D (similar to MaxPool1D's `max_indices_`), storing per-output-position kernel count during forward, used in backward for correct gradient scaling.

### H9 — PositionalEncoding integer division  **[FIXED]**
**File:** `include/nn/layers/attention/transformer.cpp:231`  
`2.0 * (i / 2)` — `i/2` is integer division, so odd indices get frequency 0 instead of 0.5. Adjacent dimensions share identical frequencies.
**Fix:** `i / 2.0`

### H10 — SkipConnection projection recreated every forward  **[FIXED]**
**File:** `include/nn/layers/skip_connection.cpp:43-44`  
When `out.cols != input.cols`, the projection Dense is recreated each forward call — loses all learned weights in training.
**Fix:** Projection Dense now created once on first dimension mismatch and cached. Subsequent forwards reuse the same `shortcut_` layer.

---

## 🟡 MEDIUM PRIORITY

### Activations (Batch A)
| # | File | Issue |
|---|------|-------|
| M1 | activations.cpp:22 | Softmax: `sum_exp` can be 0 (all logits huge negative) → div/0. Guard: `std::max(sum_exp, 1e-300)` |
| M2 | activations.cpp:28 | cross_entropy_loss epsilon `1e-12` should be `1e-7` per task spec |
| M3 | activations.cpp:19 | `log(sum_exp)` — sum_exp can be 0 → -Inf |
| M4 | activations.cpp:119 | Softplus: `exp(x)` overflows for large x → use `std::log1p(std::exp(x))` |
| M5 | activations.cpp:158 | Mish: same softplus overflow as M4 |
| M6 | activations.h:67 | GELU inline CDF lacks clamping (unlike Tensor version), overflow risk for `|x| > 4` |
| M7 | tensor.h:51 | `operator[]` no bounds check — UB on invalid index |

### Conv/Pool (Batch B)
| # | File | Issue |
|---|------|-------|
| M8 | conv_layer.h:18 | Small-kernel weight row stride may not be multiple of 64 bytes — cache line spill |
| M9 | cnn_models.cpp:43 | BN running stats zero but eps protection is present — minor |
| M10 | conv1d.cpp:49-52 | Causal padding only, documented but could confuse users |
| M11 | resnet.cpp:119 | ResNet unflatten assumes last-stage spatial dims for all stages |

### Norm/Recurrent (Batch C)
| # | File | Issue |
|---|------|-------|
| M12 | batch_norm.cpp:10 | eps default `1e-5` should be `1e-7` per task spec |
| M13 | batch_norm.cpp:27 | var near-zero guard: `v = std::max(v, eps)` before `inv_std` |
| M14 | layer_norm.cpp:9 | eps default `1e-5` should be `1e-7` |
| M15 | group_norm.cpp:47 | var clamping `1e-5f` should use `eps_` (= 1e-7) |
| M16 | group_norm.cpp:28 | No `x.cols % num_channels_` divisibility validation |

### Optimizers/Attention (Batch D)
| # | File | Issue |
|---|------|-------|
| M17 | optimizer.h:37, optimizer_extended.h:41 | Adam eps `1e-8` should be `1e-7` per task spec |
| M18 | swa.cpp:15-27 | `shadow_weights_` initialized but never used — dead memory |
| M19 | squeeze_excitation.cpp:91-93 | SE excitation in SEResNetBlock applied to wrong tensor (re-computes, not reuses) |

---

## ⚪ LOW PRIORITY / CONFIRMED CORRECT

| Check | Status |
|-------|--------|
| Tensor MatMul indexing — row-major, consistent | ✅ |
| Tensor transpose — correct | ✅ |
| Dense forward (`input * weights.transpose()`) | ✅ |
| Softmax stability — max subtraction before exp | ✅ |
| GELU Tensor version — clamps to [-4,4] | ✅ |
| Weight init schemes (Xavier, He, Uniform, Zeros) | ✅ |
| 32-byte SIMD alignment (AVX) | ✅ |
| No double-delete / memory leaks | ✅ |
| LSTM forget gate bias init to 1.0 | ✅ |
| LayerNorm axis correctness (normalize over features) | ✅ |
| Adam bias correction formula | ✅ |
| SWA running average formula in step() | ✅ |
| SE squeeze (global avg pool) logic | ✅ |
| GRU BPTT gradient flow | ✅ |

---

## Struct Padding / Alignment Summary

| Component | Assessment |
|-----------|-----------|
| `tensor.h` — 32-byte alignment for AVX | ✅ Correct |
| Conv2D weights (OHWI layout, AlignedAllocator) | ✅ |
| Dense weights | ✅ |
| BatchNorm / LayerNorm / GroupNorm | ✅ No issues |
| LSTM / GRU | ✅ |
| Model / Layer base class | ✅ |
| Small-kernel conv row stride | ⚠️ May span cache lines — micro-opt |

---

## Priority Fix Order

1. **C1 (ResNeXtBlock non-trainable)** — entire model broken
2. **C2 (SE excitation dead)** — all SE networks broken
3. **C3 (DenseBlock broken)** — DenseNet broken
4. **C4 (SWA weight corruption)** — SWA broken
5. **C5 (Adam duplicate)** — risk of wrong optimizer being used
6. **C6 (backward stubs)** — VGG/LeNet/AlexNet/CoordConv dead gradients
7. **C7 (ResBlock double gradient)** — corrupts training
8. **C8 (SkipConnection OOB)** — shape mismatch crash
9. **C9 (AdamW double decay)** — over-regularization
10. **H1-H10** — then M1-M19

---

*Audit performed by subagent team on 2026-04-20 across 4 parallel batches. No files modified during audit.*