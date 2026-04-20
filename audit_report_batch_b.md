# Audit Report — Batch B: Convolution & Pooling

**Branch:** `audit/batch-b-conv-pooling`  
**Base commit:** `aca866b`  
**Files audited:**
- `include/nn/layers/conv1d.h` / `conv1d.cpp`
- `include/nn/layers/convolutions/conv_layer.h` / `conv_layer.cpp`
- `include/nn/layers/coordconv.h`
- `include/nn/layers/pooling/pool_layer.h` / `pool_layer.cpp`
- `include/nn/layers/dilated_conv2d.h`

**Build status:** ✅ Clean — `make clean && make -j$(nproc)` exits 0, no warnings.

---

## Critical Findings (Must Fix)

### 1. MaxPool1D `max_indices_` — Wrong 2D Layout → Silent Wrong Gradients
**File:** `pool_layer.h` line 53, `pool_layer.cpp` lines 100, 117, 132

**Bug:** `max_indices_` is declared as `vector<vector<int>>` with shape `[channels][N * seq_out]`. But the data it stores is per `(batch, channel, time)` — the outer dimension should be `channels * N`, not `channels`. The index `max_indices_[c][col_idx]` where `col_idx = n * seq_out + t_out` silently indexes out-of-bounds for `n > 0`.

```cpp
// pool_layer.h line 53 — WRONG layout
std::vector<std::vector<int>> max_indices_; // [batch*channels][seq_out]

// pool_layer.cpp line 100 — allocates only channels rows
max_indices_.assign(channels, std::vector<int>(N * seq_out, -1));

// pool_layer.cpp line 117 — col_idx spans full batch, but only channels rows exist
int col_idx = n * seq_out + t_out;
max_indices_[c][col_idx] = max_idx; // OOB for n>=1 when c is small
```

**Fix:** Change to `vector<vector<int>> max_indices_(channels * N, vector<int>(seq_out, -1))` and access as `max_indices_[c * N + n][t_out]`.

**Impact:** Silent wrong gradients for batch size > 1. Training appears to work but gradients go to wrong positions.

---

## Moderate Findings (Should Fix)

### 2. Conv1D / Conv2D Backward Pass — Filters NOT Flipped (Correlation Not Convolution)
**File:** `conv1d.cpp`, `conv_layer.cpp`

**Bug:** The backward pass does NOT flip the kernels. This implements **cross-correlation** forward/backward, not true convolution. For symmetric kernels this is numerically identical, but for asymmetric kernels it produces wrong gradient signals.

```cpp
// conv1d.cpp backward: same kernel index k as forward
for (int k = 0; k < kernel_size; ++k) {
    int t = t_out * stride + k - pad;  // k not flipped
    ...
}
```

**Fix:** Use `kernel_size - 1 - k` in backward pass when computing the gradient w.r.t. input (col2im step).

**Impact:** Incorrect gradient signals for asymmetric kernels. Training converges to wrong optimum.

---

### 3. MaxPool1D `seq_out` Formula — Inconsistent with Forward/Backward
**File:** `pool_layer.cpp` lines 85, 98

**Constructor:** `seq_out = (seq_len + ksz - 1) / stride` — **ceil** division  
**Forward/Backward loop:** `for (int t_out = 0; t_out < seq_out; ++t_out)` with `t = t_out * stride + k` — **floor**-aligned

```cpp
// Constructor (line 85)
seq_out = (seq_len + ksz - 1) / stride;  // ceil

// Forward (line 98)
for (int k = 0; k < kernel_size; ++k) {
    int t = t_out * stride + k;
    if (t < seq_len) { ... }  // ceil vs floor mismatch
}
```

**Impact:** Minor inefficiency; conceptually misleading since actual coverage is floor-based.

---

### 4. Conv2D Backward Pass — Stride Not Applied in `col2im` (Matches Forward, But Worth Noting)
**File:** `conv_layer.cpp` lines 76–78

```cpp
int h = i_out * stride_h + i * dilation_h - pad_h;
int w = j_out * stride_w + j * dilation_w - pad_w;
```

Same arithmetic as forward `im2col`. This is **correct** for cross-correlation implementation. But if filters were flipped (fix #2), the backward would need `kernel_h - 1 - i` offsets.

**Note:** This is consistent with the current (cross-correlation) design, but if #2 is fixed, this offset must be updated accordingly.

---

### 5. CoordConv2D `get_weights()` / `get_gradients()` — Return Empty Tensors
**File:** `coordconv.h` lines 20–21

```cpp
Tensor get_weights() const override { return Tensor(); }
Tensor get_gradients() const override { return Tensor(); }
```

Returns default-constructed empty tensors. Callers (e.g., `Optimizer::step()` over `parameters()`) may silently skip CoordConv's weights since it delegates to the internal `Conv2D conv_`. This is actually **correct behavior** (CoordConv has no standalone weights), but the empty return may confuse debugging.

**Fix:** Document that parameters must be obtained via the internal `conv_` member, not via the `Layer` interface on `CoordConv2D`.

**Impact:** Debugging confusion; optimizer may silently skip CoordConv params if called on it directly (though internal Conv2D handles it).

---

### 6. AvgPool1D Backward Pass — Inconsistent `norm` vs Forward
**File:** `pool_layer.cpp` lines 150–152 vs 134

```cpp
// Forward: uses per-window count
output = sum / (count > 0 ? count : 1);

// Backward: always uses kernel_size
norm = 1.0 / kernel_size;
```

When the pooling window extends beyond `seq_len` at the end, `count < kernel_size` in forward, but the backward divides by `kernel_size`. The gradient magnitude is slightly off near sequence boundaries.

**Impact:** Gradient magnitudes slightly wrong near sequence boundaries when `count < kernel_size`.

---

## Minor Findings (Nice to Fix)

### 7. Conv1D `seq_out` Computation — No Stride > Input Size Validation
**File:** `conv1d.cpp` line 15

```cpp
seq_out = (seq_len + 2 * pad - ksz) / stride + 1;
```

If `stride > seq_len`, `seq_out` can still be positive (e.g., `seq_len=5, ksz=3, stride=6` → `(5+0-3)/6+1 = 1`). The code handles this correctly via bounds checks, but conceptually the pool region may be larger than the input.

---

### 8. MaxPool2D Backward — No Index Tracking, Recomputes Max Each Time
**File:** `pool_layer.cpp` lines 51–60

`MaxPool2D::backward` recomputes the max position from scratch instead of caching it during forward. This is not incorrect but is O(H*W*kH*kW) extra work per backward call vs. storing argmax indices.

**Contrast:** `MaxPool1D` DOES store `max_indices_` (though with the critical bug above). `MaxPool2D` should similarly cache indices to be efficient.

---

### 9. DilatedConv2D — No Dilation Rate Validation
**File:** `dilated_conv2d.h`

```cpp
DilatedConv2D(int in_ch, int out_ch, int kH, int kW, int H_in, int W_in,
              int dilation_rate, int stride = 1, int pad = 0)
    : Conv2D(in_ch, out_ch, kH, kW, H_in, W_in,
             stride, stride, pad, pad,
             dilation_rate, dilation_rate)
```

If `dilation_rate` causes effective kernel size to exceed input size + padding, `H_out`/`W_out` computed in `Conv2D` constructor will be non-positive and throw. That's handled. But a warning or explicit check would be better UX.

---

### 10. im2col / col2im — Heap Allocation Is Fine; No Stack Overflow Risk
**Files:** `conv_layer.cpp`

The `im2col` helper returns a `Tensor` (heap-allocated via `Tensor` class), not a stack array. No risk of stack overflow. ✅

**Minor:** `col_T = col.transpose()` creates a full copy. For large tensors this could be avoided. Not a bug, just a memory efficiency note.

---

## Summary Table

| # | Severity | File(s) | Issue |
|---|----------|---------|-------|
| 1 | **CRITICAL** | pool_layer.h, pool_layer.cpp | `MaxPool1D::max_indices_` wrong layout → OOB + wrong gradients for N>1 |
| 2 | **MODERATE** | conv1d.cpp, conv_layer.cpp | Backward pass doesn't flip kernels (cross-correlation, not convolution) |
| 3 | **MODERATE** | pool_layer.cpp | `MaxPool1D::seq_out` ceil formula inconsistent with floor-based forward access |
| 4 | **MODERATE** | conv_layer.cpp | Backward stride arithmetic note (only correct given #2 unfixed) |
| 5 | **MODERATE** | coordconv.h | `get_weights()/get_gradients()` return empty tensors — confusing for debugging |
| 6 | **MODERATE** | pool_layer.cpp | `AvgPool1D` forward uses per-window `count`, backward uses `kernel_size` divisor |
| 7 | **MINOR** | conv1d.cpp | `seq_out` formula fine but worth noting stride > input_size case |
| 8 | **MINOR** | pool_layer.cpp | `MaxPool2D::backward` recomputes max instead of caching argmax |
| 9 | **MINOR** | dilated_conv2d.h | No explicit dilation rate validation beyond H_out/W_out check |
| 10 | **MINOR** | conv_layer.cpp | `col.transpose()` creates copy — memory efficiency note |

---

## Verification Status

- **Build:** ✅ Clean — `make clean && make -j$(nproc)` exits 0, no warnings
- **Compiler:** g++ with default flags
- **Demos:** Not run (code audit only)
- **Findings:** 1 critical, 5 moderate, 4 minor
