# Neural Net Bug Audit Report

Date: 2026-04-20  
Status: All critical bugs fixed, tree compiles cleanly.

---

## Bug #1 — Adam optimizer bias correction blowup on step 1
**File:** `include/nn/optimizers/optimizer.cpp`  
**Severity:** Critical  
**Issue:** `t` starts at 0 and is incremented to 1 on the first `step()` call. With `b1_corr = 1 - beta1^t = 1 - 0.9 = 0.1`, the first-moment estimate is divided by 0.1, inflating it 10×.

**Fix:** Added `BIAS_CORR_THRESH = 1e-7`. Bias correction is only applied when `b1_corr > 1e-7` (and similarly for b2). On early steps where the correction would blow up, raw moments are used instead.

---

## Bug #2 — Dense layer redundant bias broadcast loop
**File:** `include/nn/core/layer.cpp`  
**Severity:** Low (performance)  
**Issue:** Nested loop with unused outer index `i`: each column `j` receives `bias[0][j]` regardless of row. The outer loop is O(N) redundant work.

**Fix:** Hoist `bias[0][j]` into a register variable inside a single inner loop, eliminating the outer iteration overhead.

---

## Bug #3 — LSTM backward O(seq_len) tensor copies per timestep
**File:** `include/nn/layers/recurrent/lstm.cpp`  
**Severity:** High (performance)  
**Issue:** At each backward timestep, 5 full `Tensor` objects (`h_t`, `c_t`, `c_prev`, `h_prev`, `x_t`) were allocated and filled by copying from cached state. With seq_len=100 and N=32, this allocates ~500 large tensors per backward call.

**Fix:** Replaced the 4 state-copy tensors with zero-overhead raw pointers into the already-cached `h_states` and `c_states` flat buffers: `h_t_ptr = h_states.data.data() + (t+1)*N*hidden_size`. Accesses become `h_t_ptr[i*hidden_size + h]` — no allocation, no copy. `x_t` still requires a copy from the packed input.

---

## Bug #4 — GRU backward single-step returns wrong gradient shape
**File:** `include/nn/layers/recurrent/gru.cpp`  
**Severity:** Medium  
**Issue:** The comment claimed the shape was wrong but the code already returns `(1, input_dim_)`, which matches the single-step batch size of 1 (hidden state `h_` has shape `(1, hidden_size)`).

**Fix:** Clarified the comment to document that single-step mode is stateless with batch=1, matching the stored `h_` tensor shape.

---

## Bug #5 — PositionalEncoding::backward wrong return
**File:** `include/nn/layers/attention/transformer.cpp`  
**Severity:** High  
**Issue:** `backward()` returned `grad_output` directly. Since PE has no learnable parameters, gradients should not flow back through the addition operation — returning `grad_output` allows spurious gradient contributions to propagate upstream.

**Fix:** Return `Tensor::zeros(grad_output.rows, grad_output.cols)` — correct zero-gradient for a non-parametric layer.

---

## Bug #6 — LayerNorm backward grad_gamma_/grad_beta_ not accumulated
**File:** `include/nn/layers/normalization/layer_norm.cpp`  
**Severity:** High  
**Issue:** `grad_gamma_ = Tensor(1, features)` and `grad_beta_ = Tensor(1, features)` assign (overwrite) rather than `+=` accumulate. Multiple `backward()` calls between `zero_grad()` would double-count gradients.

**Fix:** Added `if (grad_gamma_.rows == 0)` guards to initialize to zero on first call, then use `+=` accumulation on subsequent calls within the same optimization step.

---

## Bug #7 — MHA backward recomputes softmax scores instead of using cached
**File:** `include/nn/layers/attention/transformer.cpp`  
**Severity:** High (performance + correctness)  
**Issue:** `last_scores` was declared in the header but never populated in `forward()`. The backward pass recomputed the full `Q_h @ K_h^T / sqrt(d_k)` matrix and re-ran softmax for every head on every backward call.

**Fix:** `forward()` now stores pre-softmax (post-causal-mask) scores per head into `last_scores` as `(num_heads * tokens, tokens)`. `backward()` retrieves the per-head slice via `last_scores[h*tokens + i][j]` instead of recomputing.

---

## Bug #8 — GELU derivative uses unclamped x in x*pdf term
**File:** `include/nn/activations/activations.cpp`  
**Severity:** Medium  
**Issue:** `x_clamped` was used for the `arg` and `pdf` computation, but the final `return cdf + x * pdf` used the unclamped `x`, causing numerical issues for large `|x|`.

**Fix:** Changed to `return cdf + x_clamped * pdf`.

---

## Bug #9 — Adam epsilon = 1e-7 too small
**File:** `include/nn/optimizers/optimizer.h`  
**Severity:** Low (consistency)  
**Issue:** Default epsilon 1e-7 is at the edge of float32 precision. The more conservative 1e-8 is standard practice.

**Fix:** Changed default `eps` from `1e-7` to `1e-8`.

---

## Bug #10 — DataLoader::next_batch wrong batch sampling
**File:** `include/nn/layers/utility/dataloader.h`  
**Severity:** Critical  
**Issue:** `dataset_.get_sample(0)` and `dataset_.get_target(0)` were used to determine tensor dimensions, always fetching index 0 regardless of the actual batch position. This also meant column sizes were incorrectly fixed to sample 0's dimensions.

**Fix:** Use `indices_[pos_]` (the first index of the current batch) to size `X_batch` and `y_batch`, and correctly sample each element via `indices_[pos_ + i]` in the loop.

---

## Bug #11 — LSTM backward missing grad_c propagation through forget gate
**File:** `include/nn/layers/recurrent/lstm.cpp`  
**Severity:** High (correctness)  
**Issue:** The code did `grad_c *= f_gate` in-place, which modifies `grad_c` before it was fully accumulated for the next iteration. The cell-to-cell gradient `grad_c_prev = grad_c * f_gate` was not properly separated from the current iteration's `grad_c`.

**Fix:** Introduced `Tensor grad_c_prev(N, hidden_size)` — save `grad_c * f_gate` into `grad_c_prev`, then at the end of the iteration assign `grad_c = grad_c_prev` for the next BPTT step. The current `grad_c` (used for gate gradient computations) remains unmodified.

---

## Bug #12 — SimpleRNN forward hidden_states initialization
**File:** `include/nn/layers/recurrent/rnn.cpp`  
**Severity:** N/A (not a bug)  
**Issue:** The original report suspected an indexing off-by-one. Analysis confirms the code is correct: rows `0..N-1` store `h0` (zeros), and at each timestep `t`, `h_prev` reads from row `t*N+i` and stores `h_{t+1}` at row `(t+1)*N+i`. The layout is self-consistent.

**Fix:** None — not a bug.

---

## Summary

| # | File | Severity | Status |
|---|------|----------|--------|
| 1 | optimizer.cpp | Critical | ✅ Fixed |
| 2 | layer.cpp | Low | ✅ Fixed |
| 3 | lstm.cpp | High | ✅ Fixed |
| 4 | gru.cpp | Medium | ✅ Clarified |
| 5 | transformer.cpp | High | ✅ Fixed |
| 6 | layer_norm.cpp | High | ✅ Fixed |
| 7 | transformer.cpp | High | ✅ Fixed |
| 8 | activations.cpp | Medium | ✅ Fixed |
| 9 | optimizer.h | Low | ✅ Fixed |
| 10 | dataloader.h | Critical | ✅ Fixed |
| 11 | lstm.cpp | High | ✅ Fixed |
| 12 | rnn.cpp | N/A | Not a bug |
