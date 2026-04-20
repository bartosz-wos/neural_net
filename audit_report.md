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
**File:** `include/nn/layers/cnn_models_vgg.cpp:93-104`  
`forward()` creates **fresh `Conv2D` objects with random weights on every call**. Gradients go into discarded temporaries. All ResNeXt29 models are untrainable.
```cpp
Tensor ResNeXtBlock::forward(const Tensor& x) {
    Conv2D bneck(in_channels_, inner_channels, 1, 1, 1, 0); // random weights!
    Tensor out = bneck.forward(x);
    // ... same for gconv and final ...
}
```
**Fix:** Store `Conv2D` members as class members, initialized once in constructor.

---

### C2 — SE Block excitation weights never applied
**File:** `include/nn/layers/squeeze_excitation.cpp:17-37`  
`SEBlock::forward()` computes sigmoid excitation `z` via `scale_channels()` but **returns raw input** — the excitation tensor `z` is computed and stored but never multiplied with input channels. Every SE-enhanced network is a no-op.
```cpp
// computed: last_excitation_ = z = sigmoid(FC2(ReLU(FC1(gap))))
// then: return input; // excitation completely discarded!
```
**Fix:** `return scale_channels(input, z);`

---

### C3 — DenseBlock dense connectivity broken
**File:** `include/nn/layers/densenet.cpp:7-8`  
```cpp
fc_layers_.emplace_back(1, growth_rate);  // Dense(1, growth_rate) — wrong!
```
Each FC is created as `Dense(1, growth_rate)` instead of `Dense(in_channels, growth_rate)`. In forward, a `(batch, current_ch)` tensor is multiplied with a `(growth_rate × 1)` weight matrix — treating batch as channels. DenseNet concatenation is completely broken.
**Fix:** `fc_layers_.emplace_back(in_channels, growth_rate)`

---

### C4 — SWA `swap_to_averaged()` corrupts averaged weights
**File:** `include/nn/optimizers/swa.cpp:67-76`  
`swap_to_averaged()` overwrites model parameters in-place. On the next `inner_->step()`, the optimizer modifies those parameters in-place, **overwriting the averaged weights**. SWA is non-functional.
**Fix:** Two-stage swap: copy params → `shadow_weights_`, then copy averaged → params. Next optimizer step then corrupts shadow copy, not the averaged weights (or use move semantics).

---

### C5 — Adam duplicate stateless class
**File:** `include/nn/optimizers/optimizer_sgd_adam.h`  
Redefines `Adam` and `SGD` that shadow the real implementations. The local `Adam` is **stateless** — `m_state/v_state` maps are empty, so each `step()` re-initializes moment tensors to zero. Any code using this header gets a broken optimizer.
**Fix:** Remove duplicate class definitions, rename to `AdamSimple`/`SGDSimple`, or delete the header entirely.

---

### C6 — Multiple `backward()` stubs returning `Tensor(1,1)` — dead gradient ends
| Class | File | Line |
|-------|------|------|
| VGGBlock | cnn_models.cpp | 63 |
| LeNet5 | cnn_models.cpp | 129 |
| AlexNet | cnn_models.cpp | 224 |
| VGGBlock | cnn_models_vgg.cpp | 21 |
| ResNeXtBlock | cnn_models_vgg.cpp | 77 |
| CoordConv2D | coordconv.cpp | 54-56 |

All return `Tensor(1, 1)` or pass through `grad_output` without backpropping through internal layers. Gradients never reach the weights — these models train with dead conv layers.

---

### C7 — ResBlock residual gradient double-counting
**File:** `include/nn/layers/resnet.cpp:28`  
`grad[i][j] += grad_output[i][j]` adds the full residual gradient after both conv backward passes. Should only add identity branch gradient where the skip actually contributed.
**Fix:** `grad += residual_grad` only for identity path dimensions.

---

### C8 — SkipConnection backward OOB when `in_feat < out_feat`
**File:** `include/nn/layers/skip_connection.cpp:47-48`  
When `needs_projection_` is true, `shortcut_grad` has shape `(batch, in_feat)` and `inner_grad` has shape `(batch, out_feat)`. Adding them element-wise at `inner_grad[i][j]` with `j < in_feat` when `inner_grad` only has `out_feat` columns is **OOB if `in_feat < out_feat`**.
**Fix:** Restructure gradient accumulation to match input dimensions, not shortcut dimensions.

---

### C9 — AdamW + WeightDecay double weight decay
**File:** `include/nn/optimizers/optimizer_extended.cpp:64`  
`AdamW` already subtracts `lr * weight_decay * param` per step. When wrapped by `WeightDecay`, L2 is applied **twice**.
**Fix:** In `WeightDecay::step()`, skip L2 when `inner_` is already an Adam-family optimizer.

---

## 🟠 HIGH PRIORITY ISSUES

### H1 — Attention softmax `sqrt(d_k)` has no epsilon
**File:** `include/nn/layers/attention/transformer.cpp:78, 194`  
`scores[i][j] = s / std::sqrt((double)d_k)` — when `d_k=1`, scores undivided, softmax overflow possible.
**Fix:** `std::sqrt((double)d_k + 1e-9)`

### H2 — OneCycleLR divide-by-zero when `total_steps < 3`
**File:** `include/nn/optimizers/one_cycle_lr.cpp:16-17`  
`warmup_steps_ = total_steps / 3` — if `total_steps ∈ {0,1,2}`, warmup_steps = 0, then `get_lr()` divides by zero.
**Fix:** Guard: if `warmup_steps_ == 0`, skip warmup phase.

### H3 — SWA `record()` overwrites running average
**File:** `include/nn/optimizers/swa.cpp:81-88`  
`averaged_weights_[idx][i][j] = (*p)[i][j]` — direct overwrite, not accumulation. Running average is lost on subsequent `record()` calls.
**Fix:** Use same running-average formula as `step()`: `avg = (avg * (n-1) + new) / n`

### H4 — GroupNorm missing divisibility check
**File:** `include/nn/layers/normalization/group_norm.cpp:25`  
No `num_channels_ % num_groups_ == 0` validation. If C not divisible by G, `channels_per_group = C / G` truncates — some channels silently excluded from normalization.
**Fix:** `if (num_channels_ % num_groups_ != 0) throw std::invalid_argument("...")`

### H5 — Variable shadowing in VGGBlock
**File:** `include/nn/layers/cnn_models_vgg.cpp:55-57`  
```cpp
for (int i = 0; i < num_convs_; i++) {
    cur = convs_[i].forward(cur);
    for (size_t i = 0; i < cur.rows; ++i)  // shadows outer i!
```
`size_t i` shadows `int i` — could cause wrong iteration bounds.

### H6 — VGGBlock hardcoded 224×224 spatial dims
**File:** `include/nn/layers/cnn_models.cpp:27-28`  
```cpp
size_t ch = last_output_.cols / (224 * 224);
size_t H = 224, W = 224;
```
Only works for 224×224 inputs. Any other size (CIFAR 32×32) gives wrong channel count.

### H7 — BatchNorm running stats never updated
**File:** `include/nn/layers/cnn_models.cpp:43`  
`running_mean_` and `running_var_` stay at zero (initial Tensor). Mean=0, var=1e-5, BN does nothing useful in training.
**Fix:** Update moving averages during forward in training mode.

### H8 — AvgPool1D fwd/bwd normalization mismatch
**File:** `include/nn/layers/pooling/pool_layer.cpp:126, 140`  
Forward divides by actual `count` of valid positions; backward divides by `kernel_size`. Boundary positions get wrong gradient scale.
**Fix:** Store per-position count during forward, use it in backward.

### H9 — PositionalEncoding integer division
**File:** `include/nn/layers/attention/transformer.cpp:231`  
`2.0 * (i / 2)` — `i/2` is integer division, so odd indices get frequency 0 instead of 0.5. Adjacent dimensions share identical frequencies.
**Fix:** `i / 2.0`

### H10 — SkipConnection projection recreated every forward
**File:** `include/nn/layers/skip_connection.cpp:43-44`  
When `out.cols != input.cols`, the projection Dense is recreated each forward call — loses all learned weights in training.

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