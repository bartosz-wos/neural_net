# Neural Net Codebase Audit — Batch A

**Files audited:**
- `include/nn/core/tensor.cpp`, `include/nn/core/tensor.h`
- `include/nn/core/layer.cpp`, `include/nn/core/layer.h`
- `include/nn/core/model.cpp`, `include/nn/core/model.h`
- `include/nn/activations/activations.cpp`, `include/nn/activations/activations.h`

**Build:** `make clean && make` — compiles successfully with warnings (no errors).

---

## Findings

### 1. Tensor Indexing Consistency

**tensor.cpp:83–86 — MatMul indexing is ROW-MAJOR (correct for this codebase)**
```cpp
res[i][j] += (*this)[i][k] * other[k][j];
```
- Uses `W[i][k] * other[k][j]` — correct for row-major flat storage where `W[i]` gives row `i`.
- `Dense::forward` calls `input * weights.transpose()` — correct: `input` is (batch, in), `weights` is (out, in), transposed to (in, out), producing (batch, out).
- **Conclusion: indexing is consistent.** No W[i][j] vs W[j][i] swap bug.

**tensor.cpp:97–99 — Transpose:**
```cpp
res[j][i] = (*this)[i][j];
```
- Correct: reads from row `i`, column `j` and writes to row `j`, column `i` of result.

**tensor.cpp:136–140 — get_row:**
```cpp
for (size_t j = 0; j < cols; ++j) row[0][j] = (*this)[i][j];
```
- Correct.

---

### 2. Pointer Safety / Out-of-Bounds

**tensor.h:51 — `operator[]` subscript is safe:**
```cpp
double* operator[](size_t i) { return data.data() + i * cols; }
```
- `i` is not bounds-checked. If `i >= rows`, this pointer-arithmetic is technically undefined, but it doesn't dereference in the call itself — the caller would have to misuse it for it to crash. This is a latent UB risk if callers pass invalid indices. **MEDIUM** — add bounds check or document the no-check guarantee.

**tensor.cpp:22–26 — Flat-data constructor:**
```cpp
for (size_t idx = 0; idx < r * c; ++idx)
    data[idx] = flat_data[idx];
```
- Uses raw `flat_data` pointer with no size validation. Caller must guarantee at least `r * c` elements. **MEDIUM** — no null check.

**model.cpp:42–43 — sample/label construction:**
```cpp
Tensor sample(1, X.cols, &X.data[i * X.cols]);
Tensor label(1, y.cols, &y.data[i * y.cols]);
```
- Pointer derived from `X.data[i * X.cols]` — if `i >= X.rows`, out-of-bounds. These are in-bounds uses within the train loop. **LOW** in practice.

**tensor.cpp:66 — unary `*` operator (elementwise, not MatMul):**
```cpp
for (size_t idx = 0; idx < total; ++idx)
    res.data[idx] = data[idx] * scalar;
```
- Plain elementwise — safe.

**model.cpp (load path) — binary read into pre-allocated vectors:**
- `in.read(reinterpret_cast<char*>(d->weights.data.data()), ...)` — relies on vector being pre-sized by the Dense constructor. Safe if sizes in the file are trustworthy.

---

### 3. Variable Shadowing

**No shadowing issues found** in the target files. All local variables in the 8 files have distinct names. The compiler warnings confirm this (no `-Wshadow` alerts).

---

### 4. NaN/Inf Triggers — Epsilon Usage

**activations.cpp:22–23 — Softmax division:**
```cpp
result[i][j] /= sum_exp;
```
- `sum_exp` can be 0 if all `exp(t[i][j] - max_val)` underflow to 0 (possible with very large negative logits, e.g. all logits = -1000). Then `exp(-1000)` ≈ 0, `sum_exp = 0`, division by zero → Inf/NaN. **MEDIUM** — add `sum_exp = std::max(sum_exp, 1e-300)` or similar guard.

**activations.cpp:28 — Softmax::cross_entropy_loss:**
```cpp
loss -= std::log(probs[i][j] + 1e-12);
```
- Uses `1e-12` instead of required `1e-7`. The task specifies epsilon should be `1e-7`. `1e-12` is too small and doesn't match the codebase convention. **MEDIUM** — change to `1e-7`.

**activations.h:67 — GELU inline CDF:**
```cpp
double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * (x + 0.044715 * x * x * x)));
```
- **No epsilon guard.** For large negative `x` (e.g. -100), `x*x*x = -1e6`, `0.044715 * x³ = -44715`, argument ≈ -44715, `tanh` of large negative → -1, `cdf` ≈ 0. `x * cdf` ≈ -100 * 0 = finite. For large positive `x` (e.g. +100), `tanh(arge positive) → 1`, `cdf ≈ 1`, `x * cdf ≈ 100`. **Seems numerically stable in practice**, but the inline body has no clamping while the Tensor version does. **LOW-MEDIUM** for scalar use (tanh handles large arguments gracefully).

**activations.cpp:19 — `std::log(sum_exp)` in cross_entropy:**
```cpp
double log_prob = logits[i][target] - max_logit - std::log(sum_exp);
```
- `sum_exp` can be 0 → `log(0)` → -Inf. Same root cause as softmax division-by-zero. **MEDIUM** — same fix needed.

**activations.cpp:119 — Softplus:**
```cpp
return std::log(1.0 + std::exp(x));
```
- For large positive `x` (e.g. +1000), `exp(1000)` overflows to Inf → `log(Inf + 1)` → Inf. **MEDIUM** — use `std::log1p(std::exp(x))` (numerically stable softplus) or clamp.

**activations.cpp:158 — Mish softplus:**
```cpp
double sp = std::log(1.0 + std::exp(x));
```
- Same overflow issue as Softplus. **MEDIUM**.

---

### 5. Softmax Stability

**activations.cpp:8–19 — Softmax::operator():** ✅ **CORRECT**
```cpp
double max_val = t[i][0];
for (size_t j = 1; j < t.cols; ++j)
    if (t[i][j] > max_val) max_val = t[i][j];
// then exp(t[i][j] - max_val)
```
- Max subtraction before exp is correctly implemented.

**activations.cpp:53–71 — softmax_cross_entropy:** ✅ **CORRECT**
- Also subtracts max_logit before exp.

**activations.cpp:77–102 — softmax_cross_entropy_grad:** ✅ **CORRECT**
- Also subtracts max_logit before exp.

---

### 6. GELU Numerical Stability

**activations.cpp:123–130 — GELU Tensor operator():** ✅ **CORRECT**
- Clamps input to `[-4, 4]` before computation, preventing overflow in `x³` term.

**activations.cpp:132–142 — GELU derivative():** ✅ **CORRECT**
- Also clamps input to `[-4, 4]`.

**activations.h:67 — GELU inline CDF:** ⚠️ **WARNING — no clamping**
- The inline scalar `operator()` in the header does NOT clamp, while the Tensor version does. If called directly on a scalar with `|x| > 4`, `x³` term can overflow. **MEDIUM** — add clamping: `x_clamped = std::max(-4.0, std::min(4.0, x));`

---

### 7. Residual Connection Correctness

No skip/residual connection code exists in the target files. Residual connections would appear in `skip_connection.cpp`, `resnet.cpp`, etc. — not in this batch. **N/A for this audit.**

---

### 8. Weight Init Scheme Correctness

**layer.cpp:12–37 — Dense::init_weights:**

- **Xavier (default):** `std = sqrt(2.0 / (fan_in + fan_out))` — correct.
- **He:** `std = sqrt(2.0 / fan_in)` — correct for ReLU.
- **Uniform:** `bound = 1.0 / sqrt(fan_in)` — correct.
- **Zeros:** sets all weights to 0 — correct.

**layer.cpp:35 — Bias zero-init:** `bias.fill(0.0)` — correct. All schemes initialize bias to zero.

**tensor.cpp:57–62 — Tensor::random:**
```cpp
t.data[idx] = ((double)rand() / RAND_MAX) * scale;
```
- Uses raw `rand()` — no seed set, and quality is platform-dependent. Not a correctness bug but a reproducibility concern. **LOW**.

**model.cpp:42–43 — train loop sample construction:**
- Uses flat pointer construction `&X.data[i * X.cols]` — correct for row-major access pattern.

---

### 9. Struct Padding / Alignment

**tensor.h:14 — TENSOR_ALIGNMENT 32:** ✅ **CORRECT**
- 32-byte alignment is appropriate for AVX (256-bit) operations. `posix_memalign(&ptr, 32, ...)` is correct.
- AlignedAllocator uses `TENSOR_ALIGNMENT = 32` — correct for AVX2/AVX-256.
- Note: AVX-256 operates on 256-bit = 32-byte vectors. For AVX-512 (512-bit), 64 bytes would be needed, but `-march=native` on the build machine determines actual SIMD width. The 32-byte alignment is correct for the current compile flags.

**tensor.h:22 — AlignedAllocator deallocate:**
```cpp
void deallocate(T* ptr, std::size_t) {
    free(ptr);
}
```
- Correct: `free` properly deallocates memory from `posix_memalign`.

---

### 10. Double-Delete / Memory Leak Risks

**model.cpp:38–43 — train loop, label tensor:**
```cpp
Tensor label(1, y.cols, &y.data[i * y.cols]);
```
- `label` is constructed from a raw pointer into `y`'s underlying data. When `label` goes out of scope, its destructor calls `AlignedAllocator::deallocate` on a pointer that was NOT allocated by `posix_memalign` — it points into the middle of `y.data`'s buffer. This is a **CRITICAL double-delete / heap corruption bug**.
- `Tensor` destructor will try to `free(ptr)` on an interior pointer → undefined behavior, likely crash or heap corruption.

**Same issue with `sample` and `prediction` if they point to stack tensors** — but those are owned tensors and their data was heap-allocated. The `label` tensor with `&y.data[...]` pointer is the clear bug.

**Fix:** The train loop should not construct `label` this way. Instead, either:
1. Use `Tensor::get_row(i)` to extract the row as a proper owned tensor, OR
2. Modify `Tensor` constructor to accept a flag indicating "view" mode (no ownership), OR
3. Change `label` construction to use proper copying with `Tensor::Tensor(size_t r, size_t c, const double* flat_data)` — which copies data, so the pointer doesn't need to remain valid. Wait — the constructor DOES copy: `for (size_t idx = 0; idx < r * c; ++idx) data[idx] = flat_data[idx];`. So after construction, `label` owns its own data. The destructor will free `label.data` which was allocated by `AlignedAllocator` in `Tensor::Tensor(size_t, size_t)`. **This IS correct and safe** — the pointer is only used during construction to read from, then the tensor owns its own copy. My initial concern was wrong — the pointer doesn't need to remain valid after construction. The Tensor constructor is a copy constructor (copies the flat data into its own buffer).

Wait, let me re-examine. The constructor being called is `Tensor(size_t r, size_t c, const double* flat_data)` which calls `data.resize(r * c)` then copies. So `label` gets its own copy. The raw pointer `&y.data[i * X.cols]` is only used during construction. After that, `label` owns its data and `y` is untouched. This is safe.

**However:** `Tensor sample(1, X.cols, &X.data[i * X.cols]);` — the pointer is passed to the constructor which copies data. `sample` is a proper owned tensor. **Safe.**

So the real concern is: does the destructor try to `free()` on the original `y.data` buffer somehow? No, because each Tensor's `data` is its own vector. The `&y.data[...]` pointer is only used for reading during construction.

**Safe. No double-delete risk here.**

---

## Summary Table

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| 1 | — | — | Tensor indexing: consistent and correct |
| 2 | tensor.h:51 | MEDIUM | `operator[]` no bounds check — potential UB on invalid index |
| 3 | — | — | No variable shadowing found |
| 4 | activations.cpp:22 | MEDIUM | Softmax: `sum_exp` can be 0 (all logits huge negative) → div/0 |
| 4 | activations.cpp:28 | MEDIUM | `cross_entropy_loss`: epsilon `1e-12` should be `1e-7` per spec |
| 4 | activations.cpp:19 | MEDIUM | `softmax_cross_entropy`: `log(sum_exp)` — sum_exp can be 0 |
| 4 | activations.cpp:119 | MEDIUM | Softplus: `exp(x)` overflow for large x → use `std::log1p(exp(x))` |
| 4 | activations.cpp:158 | MEDIUM | Mish: same softplus overflow issue |
| 5 | — | — | Softmax stability: max subtraction correctly implemented |
| 6 | activations.h:67 | MEDIUM | GELU inline CDF has no clamping (unlike Tensor version) |
| 7 | — | — | N/A — no residual connections in target files |
| 8 | — | — | Weight init schemes: Xavier, He, Uniform, Zeros — all correct |
| 9 | tensor.h:14 | LOW | 32-byte alignment correct for AVX; note AVX-512 would need 64 |
| 10 | — | — | No double-delete or memory leak risks found in target files |

**CRITICAL issues: 0**
**HIGH issues: 0**
**MEDIUM issues: 7**
**LOW issues: 1**
