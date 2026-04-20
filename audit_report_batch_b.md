# Neural Net Convolution/Pooling Layer Audit Report

**Date:** 2026-04-20  
**Files Audited:** conv_layer.cpp/h, pool_layer.cpp/h, conv1d.cpp/h, coordconv.cpp/h, cnn_models.cpp/h, cnn_models_vgg.cpp/h, resnet.cpp/h  
**Build Status:** ✅ `make clean && make` succeeds (warnings only)

---

## 1. Variable Shadowing

**Severity: HIGH**

**File:** `include/nn/layers/cnn_models_vgg.cpp:55-57`
```cpp
for (int i = 0; i < num_convs_; i++) {
    cur = convs_[i].forward(cur);
    for (size_t i = 0; i < cur.rows; ++i)   // 'i' shadows outer loop variable
```
`size_t i` shadows `int i`. Same pattern repeats for `j` on the inner loop. Could cause incorrect iteration bounds with large tensors.

**Suggested fix:** Rename inner loop variables to `b`, `j`.

---

## 2. CoordConv2D::backward Is a Stub — Returns Wrong Shape

**Severity: CRITICAL**

**File:** `include/nn/layers/coordconv.cpp:54-56`
```cpp
Tensor CoordConv2D::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);  // ❌ WRONG SHAPE — returns 1×1 instead of input shape
}
```
The backward pass returns `Tensor(1, 1)` (1 row, 1 col) instead of a properly shaped gradient matching the input. When this layer sits in a model being backpropped, the gradient shape mismatch will corrupt all upstream gradients or crash.

**Suggested fix:** Implement proper chain-rule backprop through the coordinate concatenation and Conv2D.

---

## 3. All Model forward() Layers Return Stub Tensors on backward()

**Severity: CRITICAL**

These components return `Tensor(1, 1)` from `backward()` instead of propagating gradients, making them dead ends during backpropagation:

| File | Class | Line | Return |
|------|-------|------|--------|
| cnn_models.cpp | VGGBlock | 63 | `Tensor(1, 1)` |
| cnn_models.cpp | LeNet5 | 129 | `Tensor(1, 1)` |
| cnn_models.cpp | AlexNet | 224 | `Tensor(1, 1)` |
| cnn_models_vgg.cpp | VGGBlock | 21 | `return grad_output` (partial, no conv backprop) |
| cnn_models_vgg.cpp | ResNeXtBlock | 77 | `return grad_output` |
| cnn_models_vgg.cpp | VGG11 | — | delegates to `model_.backward()` ✅ |
| cnn_models_vgg.cpp | VGG16 | — | delegates to `model_.backward()` ✅ |
| cnn_models_vgg.cpp | ResNeXt29 | — | delegates to `model_.backward()` ✅ |
| resnet.cpp | ResBlock | 28 | partial (doesn't add residual gradient correctly) |
| resnet.cpp | ResNet | 119 | partial (doesn't unflatten correctly) |

**Specific issues:**

**`cnn_models_vgg.cpp:21` VGGBlock::backward** — returns `grad_output` directly without backproping through convolutions or ReLU. Gradients never reach the conv weights.

**`resnet.cpp:28` ResBlock::backward** — residual gradient is `grad[i][j] += grad_output[i][j]`, but this adds the full residual gradient after both conv backward passes, doubling the residual contribution. Should be: `grad[i][j] += grad_output[i][j]` only where the skip connection actually touched the output.

**`resnet.cpp:119` ResNet::backward** — Unflatten: `stage_cols = stages_.back().out_channels() * stages_.back().H_out() * stages_.back().W_out()`. This computes the gradient shape from the last stage output, but in general each stage produces different spatial dims. The flat gradient from `fc_.backward` is assumed to match the last stage's output, which may not hold.

**Suggested fix:** Implement actual backward passes through each layer's computation graph.

---

## 4. im2col Buffer Overflow Risk (Stride Boundary Check)

**Severity: MEDIUM**

**File:** `include/nn/layers/convolutions/conv_layer.cpp:62`
```cpp
H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
Tensor col(C * kH * kW, N * H_out * W_out);  // allocation
```
`col` is correctly sized. No overflow here. However:

**File:** `include/nn/layers/convolutions/conv_layer.cpp:45`
```cpp
int col_idx = n * H_out * W_out + i_out * W_out + j_out;
```
When `n = N-1` and `i_out = H_out-1` and `j_out = W_out-1`, `col_idx = (N-1)*H_out*W_out + (H_out-1)*W_out + (W_out-1) = N*H_out*W_out - 1`. This is valid (0-indexed, last column). ✅ Consistent.

**File:** `include/nn/layers/convolutions/conv_layer.cpp:47`
```cpp
int row_idx = c * kH * kW + i * kW + j;
// col[row_idx][col_idx] — row_idx max = C*kH*kW-1 ✅
```
No overflow detected. This is correct.

---

## 5. Conv2D Padding/Stride Math

**Severity: LOW (math is correct)**

**File:** `include/nn/layers/convolutions/conv_layer.cpp:22-23`
```cpp
H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
```
This is the standard floor division formula. For **SAME** padding, `pad = floor(k/2)` for odd kernels. For **VALID** padding, `pad = 0`. The math is correct for both modes.

`MaxPool2D` (pool_layer.cpp:9): `H_out = (H - kH) / stride_h + 1` — no padding, correct.

---

## 6. Conv1D Padding Boundary

**Severity: MEDIUM**

**File:** `include/nn/layers/conv1d.cpp:49-52`
```cpp
int t = t_out * stride + k - pad;
double val = 0.0;
if (t >= 0 && t < seq_len) {
    val = input[n][c * seq_len + t];
}
```
When `pad > 0`, `t` can be negative for early output positions. The check `t >= 0` handles this. ✅ However, the formula is correct for causal padding (left-padding).

**Backward (conv1d.cpp:112):**
```cpp
int t = t_out * stride + k - pad;
if (t >= 0 && t < seq_len) {
    grad_input[n][c * seq_len + t] += dX_col[row_idx][col_idx];
}
```
Same bounds check. ✅ Consistent.

---

## 7. Weight Layout (OHWI)

**Severity: LOW (no inconsistency found)**

**conv_layer.h comment:** `// weight shape: (out_channels, in_channels * kernel_h * kernel_w)`  
**Constructor (conv_layer.cpp:38):** `weights = Tensor(out_channels, in_channels * kH * kW);` ✅

**Im2col row layout:** `row_idx = c * kH * kW + i * kW + j`  
This indexes within the kernel as `[channel][i][j]`. The weight matrix is `(out_channels, in_channels*kH*kW)`. In `weights * col` (matmul), the computation is `Z[o][n] = sum_i weights[o][i] * col[i][n]`. Since `col` row i corresponds to input channel c at offset (i, j) within the kernel, `weights[o][c*kH*kW + i*kW + j]` holds the filter for `o` at position `(c, i, j)`. This is **HWIO-like but transposed**: `W[o][c*kH*kW + offset]` = `W[o][c][i][j]` = filter for output o at kernel position (i, j) of input channel c.

No inconsistency with backprop: `dW = grad_out_mat * col^T` produces `(out_channels, in_channels*kH*kW)` gradient with the same layout. ✅

---

## 8. Cache-Line Alignment

**Severity: MEDIUM**

**File:** `include/nn/layers/convolutions/conv_layer.h:18-20`
```cpp
Tensor weights;   // (out_channels, in_channels * kH * kW)
Tensor bias;      // (out_channels, 1)
Tensor grad_weights;
Tensor grad_bias;
```
`weights.data` is `std::vector<double, AlignedAllocator<double>>` (from tensor.h), which ensures 64-byte alignment via `AlignedAllocator`. ✅ However, stride between rows of the weights matrix is `in_channels * kH * kW * sizeof(double)` bytes — not guaranteed to be a multiple of cache line size (64 bytes). For large kernels this is fine, but for small kernels (e.g., 3×3) where stride < 64 bytes, adjacent row data may share cache lines causing potential minor thrash. This is a micro-optimization concern, not a correctness bug.

**Suggested fix:** For performance-critical small-kernel convolutions, consider padding row stride to a multiple of 64 bytes.

---

## 9. Struct Padding — No Explicit Issue

**Severity: LOW**

**conv_layer.h** members are all `Tensor` (heap-allocated) or `int`. No padding issues since `Tensor` is a class with only data members. `Conv2D` members include primitive ints which may not be 8-byte aligned depending on declaration order, but no performance-critical hot-path accesses `Conv2D` objects across socket boundaries — they're always accessed via pointers. No action needed.

---

## 10. NaN/Inf — BatchNorm in VGGBlock (cnn_models.cpp)

**Severity: MEDIUM**

**File:** `include/nn/layers/cnn_models.cpp:43`
```cpp
double var = running_var_[i][c][0] + 1e-5;  // eps added
double gamma = bn_gamma_[i][c][0];
double beta = bn_beta_[i][c][0];
normalized[b][c * S + s] = gamma * (x - mean) / std::sqrt(var) + beta;
```
`eps = 1e-5` added to variance before sqrt. ✅ NaN protection present. However:

1. No momentum EMA update of running_mean/running_var during training forward pass — these stay at initial values (zero-initialized Tensor), so `mean = 0` and `var = 0 + 1e-5 = 1e-5`. BatchNorm with zero running stats effectively does nothing useful in training.

2. No check for `var <= 0` — though eps prevents sqrt(0).

**Suggested fix:** Add running mean/var updates during forward in training mode.

---

## 11. AvgPool1D Normalization Mismatch

**Severity: MEDIUM**

**Forward (pool_layer.cpp:126):**
```cpp
double norm = 1.0 / kernel_size;
// ... count valid positions ...
output[n][c * seq_out + t_out] = sum / (count > 0 ? count : 1);
```
Uses actual `count` of valid positions (handles boundary cases where kernel extends beyond seq_len).

**Backward (pool_layer.cpp:140):**
```cpp
double norm = 1.0 / kernel_size;
grad_input[n][c * seq_len + t] += grad_val * norm;
```
Uses `kernel_size` as divisor, not `count`. For boundary positions where the kernel is partially out of bounds, the forward divides by `count < kernel_size`, but backward divides by `kernel_size`. Gradient will be scaled incorrectly.

**Suggested fix:** Store the per-position count during forward pass, use it in backward.

---

## 12. VGGBlock Hardcoded Spatial Dimensions

**Severity: HIGH**

**File:** `include/nn/layers/cnn_models.cpp:27-28`
```cpp
size_t ch = last_output_.cols / (224 * 224); // infer H,W assuming 224
size_t H = 224, W = 224;
```
Only works with 224×224 inputs. For any other spatial size (e.g., 32×32 CIFAR), this silently computes wrong channel count and corrupts the computation.

**Suggested fix:** Pass H, W as constructor parameters or compute from input shape at runtime.

---

## 13. ResNeXtBlock Forward Creates Temporary Conv2D Layers

**Severity: HIGH**

**File:** `include/nn/layers/cnn_models_vgg.cpp:93-104`
```cpp
Tensor ResNeXtBlock::forward(const Tensor& x) {
    Conv2D bneck(in_channels_, inner_channels, 1, 1, 1, 0);        // fresh weights each call!
    Tensor out = bneck.forward(x);
    Conv2D gconv(inner_channels, inner_channels, 3, 1, 1, 0);     // fresh weights each call!
    out = gconv.forward(out);
    Conv2D final(in_channels_, out_ch, 1, 1, 1, 0);               // fresh weights each call!
    out = final.forward(out);
```
Creates new `Conv2D` objects on every forward call with freshly initialized (random) weights. Gradients from backprop go into these temporary objects and are discarded. The block is non-trainable — all weights are random each time.

**Suggested fix:** Store Conv2D members as class members, initialize in constructor.

---

## 14. CoordConv Coordinate Tensor Layout Mismatch

**Severity: MEDIUM**

**File:** `include/nn/layers/coordconv.cpp:34-35`
```cpp
x_coord_(H_in, W_in), y_coord_(H_in, W_in),
```
`x_coord_` and `y_coord_` are 2D tensors (rows=H_in, cols=W_in). But the main input augmentation uses col-major indexing: `aug_input[b][s * (in_channels_ + 2)]` where spatial position `s` maps to flat offset. The coordinate lookup:
```cpp
size_t h = s / W_in_;
size_t w = s % W_in_;
aug_input[b][out_off] = x_coord_[h][w];  // x_coord_[h][w] = x_coord_.data[h*W_in_ + w]
aug_input[b][out_off + 1] = y_coord_[h][w];
```
The coordinate access is correct — `x_coord_[h][w]` accesses row `h`, col `w` of a row-major matrix. `h = s / W_in_` gives the row index (Y), `w = s % W_in_` gives the column index (X). ✅

**Note:** No actual bug, but the naming could be clearer (x_coord indexed by w, y_coord indexed by h).

---

## Summary Table

| # | File | Line(s) | Severity | Issue |
|---|------|---------|----------|-------|
| 1 | cnn_models_vgg.cpp | 55-57 | HIGH | Variable shadowing (`i`, `j` reused) |
| 2 | coordconv.cpp | 54-56 | CRITICAL | backward() returns `Tensor(1,1)` wrong shape |
| 3 | cnn_models.cpp | 63, 129, 224 | CRITICAL | VGGBlock/LeNet5/AlexNet backward() stubs |
| 3b | cnn_models_vgg.cpp | 21 | CRITICAL | VGGBlock backward no conv gradient |
| 3c | cnn_models_vgg.cpp | 77 | CRITICAL | ResNeXtBlock backward no conv gradient |
| 3d | resnet.cpp | 28 | CRITICAL | ResBlock residual gradient doubled |
| 3e | resnet.cpp | 119 | MEDIUM | ResNet unflatten assumes last-stage spatial dims |
| 4 | conv_layer.cpp | 45, 62 | LOW | im2col buffer — no overflow (verified safe) |
| 5 | conv_layer.cpp | 22-23 | LOW | Padding/stride math correct |
| 6 | conv1d.cpp | 49-52 | MEDIUM | Negative t bounded correctly but causal-only |
| 7 | conv_layer.cpp | — | LOW | Weight layout OHWI consistent with backprop |
| 8 | conv_layer.h | 18 | MEDIUM | Small-kernel weight row stride may span cache lines |
| 9 | conv_layer.h | — | LOW | No struct padding correctness issues |
| 10 | cnn_models.cpp | 43 | MEDIUM | BatchNorm running stats never updated, zero at init |
| 11 | pool_layer.cpp | 126, 140 | MEDIUM | AvgPool1D fwd/bwd normalization mismatch (count vs ksize) |
| 12 | cnn_models.cpp | 27-28 | HIGH | VGGBlock hardcoded 224×224 spatial dims |
| 13 | cnn_models_vgg.cpp | 93-104 | CRITICAL | ResNeXtBlock creates fresh Conv2D each forward — non-trainable |

**CRITICAL: 5 issues**  
**HIGH: 4 issues**  
**MEDIUM: 5 issues**  
**LOW: 3 issues**

---

## Priority Fix Order

1. **ResNeXtBlock forward** — makes all ResNeXt29 models untrainable
2. **CoordConv2D backward** — corrupts gradient flow  
3. **Model backward stubs** — VGGBlock, LeNet5, AlexNet, ResBlock are dead ends in backprop
4. **AvgPool1D normalization** — gives wrong gradient magnitudes
5. **BatchNorm running stats** — BN does nothing in training with zero running mean/var
6. **VGGBlock hardcoded 224** — silently wrong for non-224 inputs
7. **Variable shadowing** — could cause subtle bugs with different tensor sizes