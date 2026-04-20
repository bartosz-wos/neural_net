# Batch C Audit Report — Dense / Recurrent / Normalization

**Branch:** `audit/batch-c-dense-recurrent-norm` (from `aca866b`)  
**Auditor:** Lead Systems Engineer / HPC & DL specialist  
**Files audited:** embedding, flatten, rnn, lstm, lstm_bidirectional, gru, batch_norm, layer_norm, group_norm, weight_norm, seq2seq_attention, skip_connection, spatial_dropout

---

## Critical Findings (Must Fix)

### 1. `skip_connection.cpp` — Shape mismatch in backward (addition → confusion)
**File:** `include/nn/layers/skip_connection.cpp`  
**Location:** `SkipConnection::backward()`  
**Severity:** CRITICAL — logic bug, gradient shape mismatch

**Problem:** In the no-projection path (`needs_projection_ == false`), the code computes `grad_input = inner_grad` then adds `grad_output` elementwise. However `inner_grad` comes from `inner_->backward(grad_output, ...)` which returns `dL/d(input)`. The skip path's identity contribution should add `grad_output` directly (since `out = input + F(input)`). The current code adds `grad_output` to the inner path's gradient of the input — but then returns a tensor with `shortcut_grad.cols` columns, which is wrong when no projection is needed.

```cpp
// needs_projection_ == false path:
Tensor grad_input = inner_grad;           // shape: (batch, input_cols)
for (...) grad_input[i][j] += grad_output[i][j];  // same shape, OK
```

Actually re-examining: `inner_->backward(grad_output)` returns gradient w.r.t. the **input** of the inner layer, which is the **original input** `x`. Adding `grad_output` (dL/d(out)) is correct for the identity skip path. **This path is correct.**

However, in the **projection path** (`needs_projection_ == true`):
```cpp
Tensor grad_input(grad_output.rows, shortcut_grad.cols);
// ...
grad_input[i][j] = shortcut_grad[i][j] + inner_grad[i][j];
```
`shortcut_grad` = gradient from projecting `input → output` dimensions.  
`inner_grad` = gradient from inner layer backprop.  
Both are gradients w.r.t. the **original input** — this is correct addition.

**Re-evaluating:** After re-reading carefully, both paths appear correct. Closing this as **NOT a bug**.

---

### 2. `AttentionLayer::backward()` — Stub / placeholder returning garbage gradient
**File:** `include/nn/layers/seq2seq_attention.cpp`  
**Severity:** CRITICAL — gradients not computed, backprop terminates here

```cpp
Tensor AttentionLayer::backward(const Tensor& grad_context, double learning_rate) {
    (void)grad_context; (void)learning_rate;
    return Tensor(1, hidden_size_ + seq_len_ * encoder_dim_); // ← zero gradient!
}
```

The gradient is completely missing. Every upstream layer that relies on AttentionLayer (including Seq2SeqEncoder via LSTM backward) will receive a zero-gradient signal. This means no learning occurs through attention.

---

## Moderate Findings (Should Fix)

### 3. `lstm.cpp` — Variable shadowing in BPTT loop
**File:** `include/nn/layers/recurrent/lstm.cpp`  
**Severity:** MODERATE — compiler may warn (-Wshadow), hard to read

Inside the backward BPTT loop, the outer loop variable `i` shadows the class field `LSTM::input_dim`. The loop variable `i` is reused for batch indexing inside nested loops:

```cpp
for (int t = seq_len - 1; t >= 0; --t) {
    // ...
    for (int i = 0; i < N; ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            grad_h_prev[i][h] = grad_h_x[i][h];
```

While this compiles and runs correctly (the inner `i` shadows the outer scope), it risks confusion and will trigger `-Wshadow` warnings. Recommend renaming the batch loop variable to `b`.

---

### 4. `gru.cpp` — Variable shadowing (same pattern)
**File:** `include/nn/layers/recurrent/gru.cpp`  
**Severity:** MODERATE — same issue as LSTM

In `GRU::backward()` the BPTT loop uses:
```cpp
for (size_t t = seq_len; t-- > 0; ) {
    // ...
    for (size_t i = 0; i < input_dim_; ++i) {
        for (size_t j = 0; j < h; ++j) {
            grad_W_zr_[i][j] += x_t[0][i] * grad_zr[0][j];
```
The outer loop counter `i` shadows nothing critical here (GRU doesn't have an `input_dim_` in the backward scope), but still violates naming clarity.

---

### 5. `seq2seq_attention.cpp` — `Seq2Seq::forward()` is a stub
**File:** `include/nn/layers/seq2seq_attention.cpp`  
**Severity:** MODERATE — incomplete implementation

```cpp
Tensor Seq2Seq::forward(const Tensor& source) {
    (void)source;
    last_output_ = Tensor(1, 1); // placeholder
    return last_output_;
}
```

This always returns a `(1,1)` tensor regardless of input. Any model using this layer will train on garbage. Either implement it properly or remove the class.

---

### 6. `spatial_dropout.cpp` — `Dropout1D` mask semantics
**File:** `include/nn/layers/spatial_dropout.cpp`  
**Severity:** MODERATE — Dropout1D treats every element as independent channel

`Dropout1D::apply_mask()` generates a mask per element, not per channel/time-step:

```cpp
mask_[i].resize(total);
for (size_t j = 0; j < total; ++j) {
    mask_[i][j] = (rng_() / (double)RAND_MAX < p_);
```

For sequential data of shape `(batch, seq_len * channels)`, each individual scalar is dropped independently, rather than entire time-steps. This is standard dropout, not spatial dropout. The comment says "treat features as channels" but the implementation drops individual elements. Either fix the comment or implement true channel-wise dropout.

---

### 7. `WeightNorm::forward()` — Redundant normalize_weights() on every call
**File:** `include/nn/layers/normalization/weight_norm.cpp`  
**Severity:** MODERATE — performance: re-normalizes weights every forward even when not needed

```cpp
Tensor WeightNorm::forward(const Tensor& input) {
    normalize_weights();  // recomputes norms every single forward pass
    return wrapped_->forward(input);
}
```

For inference (`training=false`), this still renormalizes on every call, wasting computation. Add a guard to skip normalization when not training.

---

## Minor Findings (Nice to Fix)

### 8. `flatten.cpp` — No-op flatten is misleading
**File:** `include/nn/layers/dense/flatten.cpp`  
**Severity:** MINOR — Flatten.forward() returns input unchanged

The comment says "Just return as-is, the data format already flattens spatial dims". This is semantically a no-op layer, which is confusing. A Flatten that does nothing should either be documented as a no-op or the layer system should bypass it.

---

### 9. `embedding.cpp` — No gradient averaging over token occurrences
**File:** `include/nn/layers/dense/embedding.cpp`  
**Severity:** MINOR — gradient accumulation may overflow with repeated tokens

In `Embedding::backward()`, if the same token ID appears multiple times in the same sequence, the gradient for that token is **accumulated** (not averaged):

```cpp
grad_table[token_id][d] += grad_output[b][t * dim + d];
```

The comment says "Each gradient w.r.t. embedding is averaged over occurrences" but the code does not divide by the number of occurrences. This makes the effective learning rate depend on token frequency. Should divide by occurrence count, or the comment is wrong.

---

### 10. `lstm_bidirectional.cpp` — Backward pass gradient routing issue
**File:** `include/nn/layers/recurrent/lstm_bidirectional.cpp`  
**Severity:** MINOR — subtle but correct in practice

The backward pass reverses `grad_input_rev` correctly to combine with `grad_input_fwd`. The logic is:
```cpp
grad_input[b][t * input_dim + j] =
    grad_input_fwd[b][t * input_dim + j]
  + grad_input_rev[b][rev_t * input_dim + j];
```
Where `rev_t = seq_len_ - 1 - t`. This is correct.

However, `BiLSTM::backward()` calls `backward_lstm_.backward(grad_bwd, 0.0)` which receives `grad_bwd` (shape `(N, h)`) for the **last backward hidden state** only. For the reversed-sequence LSTM, the "last" hidden state corresponds to position `t=0` of the original sequence. This is the correct semantics for reverse-direction backprop. **No issue.**

---

### 11. `layer_norm.h` — `get_gradients()` returns wrong tensor
**File:** `include/nn/layers/normalization/layer_norm.h`  
**Severity:** MINOR — API inconsistency

```cpp
Tensor get_gradients() const override { return gamma; } // placeholder
```

This returns `gamma` instead of `grad_gamma_`, which is the actual gradient tensor. This is labeled as a placeholder in the comment. Should return `grad_gamma_`.

---

### 12. `batch_norm.h` — Same `get_gradients()` issue
**File:** `include/nn/layers/normalization/batch_norm.h`  
**Severity:** MINOR — same pattern as LayerNorm

```cpp
Tensor get_gradients() const override { return gamma; }
```

Should return `grad_gamma_`.

---

### 13. `layer_norm.cpp` — `Dropout` rand() without seed guard
**File:** `include/nn/layers/normalization/layer_norm.cpp`  
**Severity:** MINOR — `Dropout::forward()` uses `rand()` which is seeded by default

Uses `rand()` instead of a controlled RNG (like `std::mt19937`). This makes dropout behavior non-deterministic across platforms and during testing. The rest of the codebase uses `std::mt19937` (e.g., `Embedding::Embedding`).

---

### 14. `skip_connection.cpp` — Includes conv_layer.h unnecessarily
**File:** `include/nn/layers/skip_connection.cpp`  
**Severity:** MINOR — unnecessary coupling

```cpp
#include "convolutions/conv_layer.h"
```

The skip_connection layer uses `Dense` for the projection (as seen in `forward()`), but includes the conv_layer header which is heavier. Just include what's needed.

---

## Warnings Fixed During Build

None in Batch C files. Build completed cleanly with `g++ -std=c++17 -O2 -Wall -Wextra -march=native`.

---

## Verification Status

| Check | Result |
|---|---|
| Build (clean + full compile) | ✅ PASS |
| Compiler warnings (-Wall -Wextra) | ✅ None in Batch C files |
| Demos executable | ✅ Binaries produced |
| Critical bugs found | 1 (AttentionLayer backward stub) |
| Moderate bugs found | 4 |
| Minor issues found | 8+ |

---

## Summary

**Most impactful fix:** `AttentionLayer::backward()` is a stub that returns zero gradient. This silently breaks backpropagation through the attention mechanism, making any seq2seq model that uses it untrainable via the attention pathway.

**Second priority:** Fix the LSTM/GRU variable shadowing (rename batch index `i` → `b`) for code clarity and to eliminate `-Wshadow` warnings.

**Lowest priority:** The `Seq2Seq::forward()` stub and `get_gradients()` returning `gamma` instead of `grad_gamma_` are cleaner to fix but lower impact than the attention backward issue.
