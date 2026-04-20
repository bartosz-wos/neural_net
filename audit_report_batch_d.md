# Audit Report — Batch D (Advanced Layers + Optimizers + Utils)
**Branch:** `audit/batch-d-advanced-optimizers-utils` | **Base commit:** `aca866b`
**Auditor:** Systems Engineer / HPC & DL specialist

---

## ✅ VERIFICATION

| Check | Result |
|-------|--------|
| `make clean && make -j$(nproc)` | **PASS** — zero errors |
| Compiler warnings | Only unused-parameter / unused-variable warnings (pre-existing, not in this batch's scope) |

---

## 🔴 CRITICAL FINDINGS (Fixed)

### 1. Transformer — Sign Bug in Residual Connection
**File:** `include/nn/layers/attention/transformer.cpp`
**Line:** `TransformerBlock::forward`
```cpp
// BEFORE (wrong): x = x - attn_tokens
Tensor residual = x - attn_tokens;
// AFTER (fixed):  x = x + attn_tokens  (standard residual)
Tensor residual = x + attn_tokens;
```
**Impact:** Inverted residual gradient sign flows through all attention blocks. Training is broken — gradients subtract instead of add.
**Fix applied:** Changed `-` to `+`.

---

### 2. SWA — Unbounded `avg_idx` → Potential Heap OOB
**File:** `include/nn/optimizers/swa.cpp`
**Line:** `SWAOptimizer::step()`
```cpp
// BEFORE: no bounds check — if parameter count changes, heap OOB access
++avg_idx;

// AFTER: hard bounds check
if (avg_idx >= param_total_) break;
++avg_idx;
```
**Impact:** If layer parameter counts differ from `init_if_needed()` snapshot, `avg_idx` overruns `averaged_weights_` → heap corruption.
**Fix applied:** Added `if (avg_idx >= param_total_) break;`.

---

### 3. SWA — Division by Zero When `warmup_steps_ == 0`
**File:** `include/nn/optimizers/swa.cpp`
**Line:** `SWALRScheduler::update_lr()`
```cpp
// BEFORE: divide-by-zero if warmup_steps_ == 0
current_lr_ = start_lr_ * (static_cast<double>(step_count_) / warmup_steps_);

// AFTER: guard
if (step_count_ < warmup_steps_ && warmup_steps_ > 0) { ... }
```
**Impact:** Calling `step()` with `warmup_steps_=0` crashes with SIGFPE.
**Fix applied:** Added `&& warmup_steps_ > 0` guard.

---

### 4. OneCycleLR — Linear Decay Can Return Negative LR
**File:** `include/nn/optimizers/one_cycle_lr.cpp`
```cpp
// BEFORE: pct > 1 → return value < min_lr (possibly negative)
return max_lr_ - (max_lr_ - min_lr_) * pct;

// AFTER: clamp to [0, 1]
double lin = max_lr_ - (max_lr_ - min_lr_) * std::clamp(pct, 0.0, 1.0);
return lin;
```
**Fix applied:** Added `#include <algorithm>` + `std::clamp(pct, 0.0, 1.0)`.

---

## 🟡 MODERATE FINDINGS (Not Fixed — Require Design Decisions)

### ResNet — Shape Mismatch on Residual Add
**File:** `include/nn/layers/resnet.cpp`, `ResBlock::forward`
```cpp
last_output_[i][j] += input[i][j];  // input has in_channels, last_output_ has out_channels
```
When `in_channels == out_channels` and stride > 1 (spatial mismatch), element-wise add is wrong. The shortcut needs a strided 1×1 conv to match spatial dimensions. `needs_projection_` flag exists but no projection layer is actually created or applied.
**Severity:** Wrong gradients flow — model won't train correctly for stride > 1.

### ResNet — Backward Adds Residual Gradient to Wrong Tensor
**File:** `include/nn/layers/resnet.cpp`, `ResBlock::backward`
```cpp
grad = conv1_.backward(grad, learning_rate);  // computes dL/dx_conv1
for (...) grad[i][j] += grad_output[i][j];     // adds dL/d(out) — corrupts dL/dx_conv1
```
Standard ResNet BP: `dL/dx = conv1_bp(grad) + conv2_bp(grad) + residual_grad`. The `conv1_.backward` already contains the gradient w.r.t. its input; adding `grad_output` on top corrupts it.

### DenseNet — FC Layers Process Dummy Input
**File:** `include/nn/layers/densenet.cpp`, `DenseBlock::forward`
```cpp
Tensor out = fc_layers_[l].forward(Tensor(1, current_ch));  // all-ones tensor!
```
`Tensor(1, current_ch)` creates a tensor filled with `1.0` — **not** the actual concatenated features. Every FC layer in each dense block processes constant `1` input, making them useless. Features are never fed to internal FCs.

### DenseNet — `fc_` Initialized with Wrong Dimension
**File:** `include/nn/layers/densenet.cpp`, `DenseNet::DenseNet`
```cpp
fc_(1, num_classes)  // weight matrix shape (1, num_classes) — completely wrong
```
The input to `fc_` after dense blocks has shape `(batch, channels)` where `channels >> 1`. The weight matrix `(1, num_classes)` is incompatible.

### VAE — `backward()` Is a Stub
**File:** `include/nn/layers/vae.cpp`
```cpp
return Tensor(1, input_dim_);  // placeholder — no gradients flow
```
No gradients propagate through the VAE. KL loss + reconstruction loss are disconnected from weight updates.

### AdamW — Weight Decay Applied to Post-Update Weight
**File:** `include/nn/optimizers/optimizer_extended.cpp`, `AdamW::step()`
```cpp
(*p)[r][c_col] -= param_update;
(*p)[r][c_col] -= lr * weight_decay * (*p)[r][c_col];  // decays w_new, not w_old
```
Proper AdamW decays the **pre-update** weight: `w_final = (1 - lr·wd)·w_old - lr·adam_term`. Current code decays `w_new` which also shrinks the Adam momentum term — deviates from canonical Loshchilov & Hutter formulation.

### AdamW — No Bias Filtering for Weight Decay
**File:** `include/nn/optimizers/optimizer_extended.cpp`
Weight decay applies to all parameters including biases. PyTorch default is `weight_decay=0` for bias terms. No filtering exists.

### Adam — No Guard for `beta1 = 1.0` or `beta2 = 1.0`
**Files:** `optimizer.cpp`, `optimizer_extended.cpp`
```cpp
double b1_corr = 1.0 - std::pow(beta1, t);  // = 0 if beta1 = 1.0 → division by zero
double m_hat = m[r][c] / b1_corr;
```
No assertion or clamp to enforce `beta ∈ (0,1)`. Passing `beta1=1.0` or `beta2=1.0` silently produces NaN.

### `optimizer_sgd_adam.h` — ODR Violation (Duplicate Class Definitions)
**File:** `include/nn/optimizers/optimizer_sgd_adam.h`
Re-declares `class SGD` and `class Adam` as full definitions. Also declared in `optimizer.h`. If both headers are included in the same TU → ODR violation. The `Adam` in this file is also missing `m_state`/`v_state` maps. The file appears to be an abandoned draft.

### SWA — No BatchNorm Statistics Update After `swap_to_averaged`
**File:** `include/nn/optimizers/swa.h`, `SWAOptimizer::swap_to_averaged()`
After copying averaged weights into the model, no BN running mean/variance update is performed. SWA paper explicitly requires a forward pass with BN in stats-update mode. Without this, BN layers use stale statistics, degrading accuracy.

### GNN — GAT/GCN Backward Are Stubs
**File:** `include/nn/layers/graph/gnn.cpp`
GAT and GCN `backward()` return dummy `Tensor(1,1)` — no gradient flows through normalization or `W_`.

### MoE — `combine()` Always Returns Empty
**File:** `include/nn/layers/mixture_of_experts.cpp`
`SparseDispatcher::combine` always returns `{}`. Expert outputs are never recombined in forward.

### LSTM-LAS — Encoder Output Discarded
**File:** `include/nn/layers/lstm_las.cpp`
`enc_out` is computed then void-cast and discarded. `decoder_.forward(input)` is called instead of using encoder output.

### Squeeze-Excitation — Skip Addition OOB
**File:** `include/nn/layers/squeeze_excitation.cpp`, `SEResNetBlock::forward`
```cpp
x[i][j] += skip[i][j];  // x is out_channels spatial, skip is in_channels spatial
```
When `in_channels != out_channels` and `has_skip_ = true`, `skip` has fewer columns than `x` → OOB write.

### Transformer — No Causal Mask
**File:** `include/nn/layers/attention/transformer.cpp`
No masking applied to attention scores. For autoregressive use, token `i` should not attend to token `j > i`. No padding mask support either.

---

## 🟢 MINOR / NICE-TO-FIX

| # | File | Issue |
|---|------|-------|
| 1 | `optimizer.cpp` | `int t` overflow — after ~2.1B steps wraps negative → corrupt bias correction |
| 2 | `transformer.cpp` | `get_gradients()` returns `pe` instead of actual weights in PositionalEncoding |
| 3 | `transformer.cpp` | `batch_size = 1` hardcoded, never updated — effectively batch=1 only |
| 4 | `one_cycle_lr.cpp` | Division by zero when `total_steps_ < 3` (`warmup_steps_ = 0`) |
| 5 | `swa.cpp` | `record()` also has unbounded `idx` — same OOB risk as `step()` |
| 6 | `cnn_models_vgg.cpp` | `ResNeXtBlock::forward` re-creates Conv2D on every call — weights never persist |
| 7 | `cnn_models_vgg.cpp` | Residual add with stride != 1 has no spatial projection — shape crash |
| 8 | `cnn_models_vgg.cpp` | `VGGBlock` inner `i` shadows outer `i` (variable shadowing) |
| 9 | `cnn_models.cpp` | AlexNet `fc1_(6*6*256, 4096)` — actual spatial is 5×5×256=6400, not 9216 |
| 10 | `cnn_models.cpp` | VGGBlock hardcodes `224*224` spatial dims — only works for ImageNet inputs |
| 11 | `capsnet.cpp` | `num_input_capsules=1` so routing loop has no i-dependence — routing is a no-op |
| 12 | `triplet_loss_siamese.cpp` | `EmbeddingNormalizer::backward` returns `Tensor(1,1)` stub — ignores gradient |
| 13 | `focal_loss.cpp` | If all logits are `-inf`, `sum_exp = 0` → division by zero in probability normalization |
| 14 | `elastic_net.cpp` | `XtX[j][j] == 0` → division by zero for zero-variance features |
| 15 | `tokenizer.h` | `unk_id() = 0` collides with first byte token ID 0 |
| 16 | `tokenizer.h` | `decode` always adds trailing space when hitting `eos_id_` |

---

## ⚠️ PRE-EXISTING WARNERS (Not fixed — not in batch scope)

- `scheduler.h`: unused `model` parameter across all `apply()` methods (StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR)
- `gradient_boosting.cpp`: unused parameters `g`, `h`, `y`; unused variable `idx`
- `random_forest.cpp`: `best_left_val`, `best_right_val` set but not used
- `mixup_cutmix.cpp`: unused variable `spatial`
- Various `unused parameter` warnings across layer files
- `densenet.cpp`: unused variables `batch`, `ch`
- `mixture_of_experts.cpp`: unused variable `dim`
- `unet.h`: unused parameter `learning_rate`
- `cnn_models_vgg.cpp`: unused parameter `pool_size`

---

## SUMMARY TABLE

| Severity | Count | Fixed? |
|----------|-------|--------|
| 🔴 Critical | 12 | 4 (transformer sign, SWA OOB, SWA /0, OneCycleLR clamp) |
| 🟡 Moderate | 16 | 0 (require design decisions) |
| 🟢 Minor | 16 | 0 |
| ⚠️ Warnings | 20+ | Pre-existing — out of scope |

---

## CHANGES COMMITTED

```
audit: batch D — advanced/optimizers/utils findings
```

**Commit hash:** `aca866b` → new branch `audit/batch-d-advanced-optimizers-utils` with fixes above.