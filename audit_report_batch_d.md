# Neural Net Code Audit Report — Batch D
**Files:** optimizers + advanced layers
**Build status:** ✅ `make clean && make` succeeds (with warnings)

---

## CRITICAL Issues

### 1. Adam duplicate class + stateless optimizer in `optimizer_sgd_adam.h`
- **File:** `include/nn/optimizers/optimizer_sgd_adam.h`
- **Severity:** CRITICAL
- **Description:** This header redefines `Adam` (and `SGD`) classes that shadow the implementations in `optimizer.h`. More critically, the `Adam` defined here is **stateless** — it has no `m_state`/`v_state` maps, so each call to `step()` re-initializes moment tensors to zero. Any code that uses this header (instead of `optimizer.cpp`'s `Adam`) would get a completely broken optimizer that resets its state every step.
- **Suggested fix:** Remove the duplicate class definitions from `optimizer_sgd_adam.h` or rename them to `AdamSimple`/`SGDSimple` to prevent shadowing. The stateful `Adam` in `optimizer.cpp` is correct and should be the only one used.

### 2. DenseBlock — Dense FC layers created with wrong input size `1` instead of `current_ch`
- **File:** `include/nn/layers/densenet.cpp:7-8`
- **Severity:** CRITICAL
- **Description:** `DenseBlock` creates each FC sub-layer as `Dense(1, growth_rate)` regardless of `in_channels`. In `forward()`, each layer is called with `Tensor(1, current_ch)` — a `(1, current_ch)` tensor fed to a `(growth_rate × 1)` weight matrix. The Dense matmul (`input * weights.transpose()`) multiplies `(1, current_ch) @ (growth_rate, 1) → (1, growth_rate)`, treating the entire batch as a single sample. This breaks dense connectivity — the block is not correctly DenseNet.
- **Suggested fix:** Replace `fc_layers_.emplace_back(1, growth_rate)` with `fc_layers_.emplace_back(in_channels, growth_rate)`. Then in `forward`, each layer should operate on the full `(batch, current_ch)` input and output `(batch, growth_rate)`.

### 3. SEBlock — excitation weights computed but never applied to input
- **File:** `include/nn/layers/squeeze_excitation.cpp:17-37`
- **Severity:** CRITICAL
- **Description:** `SEBlock::forward()` computes global average pooling, FC→ReLU→FC→Sigmoid, stores `last_excitation_`, and returns it. But **the excitation is never multiplied with the input channels**. `SEResNetBlock::forward()` calls `se_.forward(x)` and passes the excitation to `scale_channels()`, but `scale_channels()` is never invoked in `SEBlock::forward()` itself. The SE attention is completely non-functional — all blocks return excitation weights that are discarded.
- **Suggested fix:** In `SEBlock::forward()`, after computing the sigmoid excitation `z`, apply channel scaling and return the result:
  ```cpp
  Tensor output = scale_channels(input, z);
  last_excitation_ = z;
  return output;
  ```

### 4. SWA — `swap_to_averaged()` overwrites model params in-place, corrupting averaged weights on next optimizer step
- **File:** `include/nn/optimizers/swa.cpp:67-76`
- **Severity:** CRITICAL
- **Description:** `swap_to_averaged()` directly overwrites `(*p)[i][j] = averaged_weights_[idx][i][j]`. The optimizer holds `Tensor*` references to these same parameter tensors. On the next `inner_->step()` call, the inner optimizer modifies the parameters **in place**, overwriting the averaged weights that were just swapped in. After one training step post-swap, the SWA averaged weights are lost.
- **Suggested fix:** Use a two-stage swap: copy current params to `shadow_weights_`, then copy `averaged_weights_` into params. After the inner optimizer's next step, optionally restore from `shadow_weights_` if you want to keep SWA params stable across optimizer updates.

---

## HIGH Issues

### 5. Attention softmax — `sqrt(d_k)` without epsilon in forward pass
- **File:** `include/nn/layers/attention/transformer.cpp:78`
- **Severity:** HIGH
- **Description:** `scores[i][j] = s / std::sqrt((double)d_k)` — raw `sqrt(d_k)` used in forward pass attention scores with no epsilon. When `d_k = 1`, the score magnitude is undivided, which can cause large exponentials in softmax and potential overflow. Backward pass has the same issue at line 194.
- **Suggested fix:** Add epsilon: `std::sqrt((double)d_k + 1e-9)`.

### 6. OneCycleLR — division by zero when `total_steps < 3`
- **File:** `include/nn/optimizers/one_cycle_lr.cpp:16-17`
- **Severity:** HIGH
- **Description:** `warmup_steps_ = total_steps / 3`. If `total_steps = 0, 1,` or `2`, then `warmup_steps_ = 0`. Then `get_lr()` divides by `warmup_steps_` (line 23: `s / warmup_steps_`), causing division by zero.
- **Suggested fix:** Guard: if `warmup_steps_ == 0`, skip warmup phase and go directly to anneal.

### 7. AdamW + WeightDecay double weight decay
- **File:** `include/nn/optimizers/optimizer_extended.cpp:64`
- **Severity:** HIGH
- **Description:** `AdamW` already implements AdamW-style weight decay (line 64: `(*p)[r][c_col] -= lr * weight_decay * (*p)[r][c_col]`). If `WeightDecay` wraps an `AdamW`, the L2 penalty is applied twice — once explicitly in `AdamW::step()` and once in `WeightDecay::step()`.
- **Suggested fix:** In `WeightDecay::step()`, skip the L2 subtraction when `inner_` is an `AdamW` (or any optimizer that already handles weight decay), or document this constraint clearly.

### 8. SWA — `record()` overwrites running average instead of accumulating
- **File:** `include/nn/optimizers/swa.cpp:81-88`
- **Severity:** HIGH
- **Description:** `record()` does `averaged_weights_[idx][i][j] = (*p)[i][j]` — a direct copy that **overwrites** the running average. If called after `step()` multiple times, subsequent calls lose all prior averaging. The running average accumulation only happens in `step()`.
- **Suggested fix:** Either remove `record()` (it conflicts with SWA semantics), or have it call the same running-average update formula used in `step()`:
  ```cpp
  double n = static_cast<double>(step_count_ - start_after_ + 1);
  averaged_weights_[idx][i][j] = (averaged_weights_[idx][i][j] * (n - 1) + (*p)[i][j]) / n;
  ```

---

## MEDIUM Issues

### 9. Adam epsilon should be 1e-7 (per task spec) vs 1e-8
- **File:** `include/nn/optimizers/optimizer.h:37`, `include/nn/optimizers/optimizer_extended.h:41`
- **Severity:** MEDIUM
- **Description:** Default `Adam(0.001, 0.9, 0.999, 1e-8)` and `AdamW(0.001, 0.9, 0.999, 1e-8, 0.01)` use epsilon=1e-8. The task specifies epsilon should be 1e-7. While 1e-8 is the PyTorch default and widely used, 1e-7 is safer for fp16 training and is the recommendation here.
- **Suggested fix:** Change default eps to `1e-7` in both constructors.

### 10. SWA shadow_weights_ initialized but never meaningfully used
- **File:** `include/nn/optimizers/swa.h:29-30`, `swa.cpp:15-27`
- **Severity:** MEDIUM
- **Description:** `shadow_weights_` is initialized to current parameter values in `init_if_needed()`, but `swap_to_averaged()` and `record()` never use it. It's dead weight consuming memory.
- **Suggested fix:** Either use `shadow_weights_` for a proper save/restore mechanism (save original params to shadow, swap in averaged, then optionally restore), or remove it entirely.

### 11. SE-ResNet — SE excitation weights not actually used in SEResNetBlock
- **File:** `include/nn/layers/squeeze_excitation.cpp:91-93`
- **Severity:** HIGH (same root cause as #3, elevated)
- **Description:** In `SEResNetBlock::forward()`, `se_.forward(x)` computes and returns excitation weights, then `scale_channels(x, se_.forward(x))` multiplies `x` by those weights. But since `SEBlock::forward()` returned raw input `x` instead of scaled output, the excitation weights computed by the SE block are applied to the original unscaled input — not the correct feature map. The SE recalibration is misapplied.

---

## LOW / Minor Issues

### 12. PositionalEncoding integer division in angle denominator
- **File:** `include/nn/layers/attention/transformer.cpp:231`
- **Severity:** LOW
- **Description:** `double angle = pos / std::pow(10000.0, 2.0 * (i / 2) / d_model);` — `i / 2` is integer division. For odd `i`, this gives 0 instead of 0.5, causing adjacent dimensions to use identical frequencies. Should be `i / 2.0`.
- **Suggested fix:** Change `(i / 2)` to `(i / 2.0)`.

### 13. DenseBlock `concat_buffers_[0] = input` — index 0 is correct but concat_buffers_ is otherwise unused
- **File:** `include/nn/layers/densenet.cpp:24`
- **Severity:** LOW
- **Description:** `concat_buffers_[0]` stores the initial input (for potential backward pass use), but all other `concat_buffers_` entries are set to layer outputs (`concat_buffers_[l] = out`) and never read back. Dead code.
- **Suggested fix:** Clean up unused `concat_buffers_` or wire them for proper DenseNet backward pass (dense connections require routing gradients to all prior layers).

### 14. No cache-line padding on Model/Layer structs
- **File:** `include/nn/core/layer.h`
- **Severity:** INFO
- **Description:** `struct Layer` has no `alignas` directives. With virtual function table pointer and multiple Tensor members (each containing `std::vector<double>` + `size_t` fields), cache-line misalignment is possible but unlikely to cause correctness bugs in this codebase. No critical alignment issues found.
- **Suggested fix:** For production, add `alignas(64)` to Tensor member declarations in hot paths (e.g., `last_input` in attention layers).

---

## Issues Not Found (Confirmed Correct)

| Check | Status |
|-------|--------|
| Tensor indexing in matmul/attention (row-major, consistent) | ✅ Correct — `Q_h_t[i][dk]` dot `K_h_t[j][dk]` over d_k gives `(tokens, tokens)` scores |
| Attention backward pass QK^T consistency with forward | ✅ Correct — backward recalculates scores with same formula |
| Adam bias correction formula `m_hat = m / (1-beta1^t)` | ✅ Correct — standard formula applied |
| AdamW bias correction | ✅ Correct |
| SWA running average formula | ✅ Correct — `(old_avg * (n-1) + new) / n` |
| DenseNet concatenation (batch dim preserved) | ✅ Correct — `Tensor(batch, current_ch + growth_rate_)` |
| SE-Net squeeze (global avg pool) | ✅ Correct — `sum / spatial` per channel |
| SE-Net excitation (FC→ReLU→FC→Sigmoid) | ✅ The logic is correct; bug is excitation not applied (issue #3) |
| Variable shadowing in optimizers | ✅ None found — loop variables scoped correctly |

---

## Summary by Severity

| Count | Severity |
|-------|----------|
| 4 | CRITICAL |
| 4 | HIGH |
| 3 | MEDIUM |
| 2 | LOW |

**Worst offenders:** DenseBlock non-functional dense connections (#2), SE attention completely dead (#3), SWA weight corruption (#4), stateless Adam shadowing (#1).
