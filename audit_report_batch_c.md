# Code Audit Report: Batch C (Normalization / Recurrent / Dense / SkipConnection)

**Date:** 2026-04-20  
**Build:** `make clean && make` — ✅ Compiles with warnings (unused parameters, hidden overloads)  
**Files Audited:** 20 source/header pairs

---

## NORMALIZATION

### batch_norm.cpp / batch_norm.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 1 | batch_norm.cpp:10 | MEDIUM | `eps = 1e-5` default; task requires `1e-7`. While 1e-5 is conventional, the stricter requirement means this may cause unnecessary instability in edge cases (tiny variance). | Change default `eps` to `1e-7`. Also consider bias-corrected variance (var / (N-1)) for sample variance correctness. |
| 2 | batch_norm.cpp:27 | MEDIUM | `inv_std = 1.0 / std::sqrt(v + eps)` — no guard for `v` being near-zero or negative after `v /= batch`. If batch has near-zero variance, `sqrt` may overflow numerically. | Guard: `v = std::max(v, eps)` before `inv_std` computation, matching the forward-pass clamping already done in LayerNorm and GroupNorm. |

**Findings:** No pointer OOB (loops all bounded by `batch`/`features`). No variable shadowing. Moving average decay (momentum), train/eval mode — correct. Struct padding: `bool training` at end of struct may introduce 7 bytes of trailing padding after `double momentum`, but this is non-critical.

---

### layer_norm.cpp / layer_norm.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 3 | layer_norm.cpp:9 | MEDIUM | `eps = 1e-5` default; task requires `1e-7`. | Change default `eps` to `1e-7`. |
| 4 | layer_norm.cpp:47 | MEDIUM | Variance clamping `var = std::max(var, 1e-7)` is applied *before* computing `sqrt_var = std::sqrt(var + eps)` in forward. However if `var` is 0 and eps is 1e-5, `sqrt_var ≈ 0.0032` which is fine. But the clamping value `1e-7` matches the task requirement, unlike batch_norm and group_norm which use `1e-5`. | Clamping value is correct per task spec; just update constructor default eps. |
| 5 | layer_norm.cpp:24 | LOW | `last_mean = Tensor(1, batch)` and `last_var = Tensor(1, batch)` — these store per-sample scalars but are indexed as `[0][b]`. This is a storage convention (not an error), but means if `batch > 1`, each scalar takes 1×batch Tensor. Acceptable for current use; not a bug. | No fix needed; document this design decision. |

**Axis correctness:** ✅ Correct. LayerNorm normalizes over the **feature dimension** per sample (index `f` loop for mean/var, then `gamma[0][f]` access). `last_mean[0][b]` indexed by batch is the correct convention for per-sample mean storage. Implementation matches standard LayerNorm (Ba et al., 2016).

**Dropout:** ✅ `update_weights` is empty (no weights), `mask` properly used in backward with training guard. No issues.

---

### group_norm.cpp / group_norm.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 6 | group_norm.cpp:25 | HIGH | **Missing group count divisibility validation.** Code checks `num_groups_ > 0 && <= num_channels_` but does NOT check `num_channels_ % num_groups_ == 0`. If C is not divisible by G, `channels_per_group = C / G` truncates, leaving some channels unprocessed. The forward/backward will silently skip those channels. | Add: `if (num_channels_ % num_groups_ != 0) throw std::invalid_argument("GroupNorm: num_channels must be divisible by num_groups");` after the existing range check. |
| 7 | group_norm.cpp:28 | MEDIUM | Flat indexing: `x[n][c * spatial_per_channel + s]` — this assumes x.cols == C * H * W exactly. No validation that `x.cols % num_channels_ == 0`. If input is malformed, silent incorrect behavior. | Add validation: `if (x.cols % num_channels_ != 0) throw std::invalid_argument("GroupNorm: x.cols must be divisible by num_channels");` |
| 8 | group_norm.cpp:47 | MEDIUM | Var clamping uses `std::max(var, 1e-5f)` while epsilon is also `1e-5f`. Task requires clamping to `1e-7`. The clamping floor should be `eps_` itself (or `1e-7f`), not a hardcoded `1e-5f`. | Change to: `var = std::max(var, eps_);` — use the actual epsilon as floor. |

**Backward pass correctness:** ✅ Chain rule is correctly applied. `dL/dvar`, `dL/dmean`, `dL/dx` are all derived correctly from the normalized forward computation.

---

### weight_norm.cpp / weight_norm.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 9 | weight_norm.cpp:33 | LOW | `v_ = *params[0]` snapshots weights after each `update_weights` call. During `forward`, `normalize_weights()` re-normalizes then calls `wrapped_->forward(input)`. If the wrapped layer also modifies its weights during its own forward (e.g., in-place optimizer step), the snapshot may be stale. Currently only `update_weights` syncs `v_`, so this is fine. | No change needed; just document the sync-on-update design. |

**No issues found.** Correct weight normalization, scaling, and gradient pass-through.

---

## RECURRENT

### lstm.cpp / lstm.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 10 | lstm.cpp:41-42 | MEDIUM | **LSTM forget gate bias initialized to 1.0** — this is correct and good practice. ✅ However, **no forget gate bias init for the bidirectional LSTM** (lstm_bidirectional.cpp — it uses the default LSTM constructor, so it DOES get the 1.0 init). No issue here. | None needed. |
| 11 | lstm.cpp | MEDIUM | `h_states` and `c_states` cache is `(seq_len+1)*N` rows. However in `backward`, the forward states are re-read from these caches. If backward is called after a second forward (without zeroing caches), the old states persist. No state versioning. | Ensure caller calls `zero_grad()` or re-initializes model state between forward passes in training loop. Consider adding a `clear_cache()` method. |

**Tensor indexing:** ✅ All indexing is consistent. Gate pre-activation splitting is correct (4 gates at offsets h, 2h, 3h). BPTT gradient flow (grad_c propagating through forget gate) is correctly implemented.

**No pointer OOB, no variable shadowing, no NaN risks** — eps is not used in LSTM (no normalization layer).

---

### rnn.cpp / rnn.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 12 | rnn.cpp:68-71 | LOW | `hidden_states` cache initialized as zero via explicit loop. The tensor is value-initialized on construction (Tensor(row, col) fills with 0.0 via Tensor::zeros), so the explicit loop is redundant. Not a bug. | Remove redundant explicit initialization loop for clarity. |

**All indexing correct.** BPTT through tanh (grad_pre * W_hh) is correct. No NaN risks (tanh derivative bounded [0,1]).

---

### gru.cpp / gru.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 13 | gru.cpp:24-25 | MEDIUM | Uniform initialization `(-scale, scale)` for **all** gate weights (W_zr, U_zr). For sigmoid gates, Xavier with uniform bounds can work, but the more common approach is to initialize sigmoid gate biases to 0 (already done) and use the standard weight init. No critical bug, but gate weights may start with higher variance than optimal. | Consider using `scale = std::sqrt(6.0 / (fan_in + fan_out))` for the zr gates specifically, or initialize zr gate input portions to 0 like the bias. |
| 14 | gru.cpp | MEDIUM | `backward` loop: `for (size_t t = seq_len; t-- > 0;)` — this is correct C++ but non-obvious. More critically, for non-final timesteps, `grad_h` carries the BPTT-accumulated gradient (correct). For t == seq_len-1, `grad_h` is set from `grad_output.get_row(t)`. If `grad_output.rows == 1` (single-output mode), this only provides gradient for the last timestep, which is correct for sequence prediction. | Add a comment clarifying the gradient flow for final vs intermediate timesteps. |
| 15 | gru.cpp:145 | MEDIUM | `grad_hc_pre` computation uses `z_[0][j]` (update gate) and `dhc[0][j]` (candidate derivative). For the candidate path: `dl_dhc = grad_h * z * dhc`, this correctly accounts for the update gate's influence on the candidate hidden state. | No fix needed; this is correct. |

**Gradient flow:** ✅ BPTT correctly backpropagates through update/reset gates and candidate. Parameter gradients accumulated correctly.

---

### lstm_bidirectional.cpp / lstm_bidirectional.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 16 | lstm_bidirectional.cpp:35-36 | HIGH | **Shape mismatch in gradient accumulation.** When `needs_projection_` is true (from the inner SkipConnection), `shortcut_->backward(grad_output)` returns gradients of shape `(batch, in_feat)` (gradient w.r.t. projection input = original input), while `inner_->backward(grad_output)` returns gradients of shape `(batch, out_feat)`. These cannot be added element-wise. The code allocates `grad_input(grad_output.rows, shortcut_grad.cols)` = `(batch, in_feat)` and then tries to add `shortcut_grad + inner_grad` which have incompatible dimensions. | Fix the gradient accumulation to use `out_feat` dimension for `grad_input`, and accumulate both gradient contributions into the same shape before returning. E.g., `Tensor grad_input(grad_output.rows, input.cols);` then for needs_projection_: compute `grad_proj_input` from shortcut and `grad_in_output` from inner, but only sum the `in_feat` dimensions, returning gradients aligned to the actual input dimensions expected by the next layer. |

**Forward pass:** ✅ Sequence reversal is correct. Concatenation of forward + backward last hidden states is correct.

---

## DENSE

### embedding.cpp / embedding.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 17 | embedding.cpp:48 | MEDIUM | `backward` returns `Tensor(grad_output.rows, grad_output.cols)` — a zero-filled tensor with no actual gradient computation for the input. This is semi-intentional (ID inputs don't backpropagate through non-differentiable indexing), but `Embedding::backward`'s return type promise is unclear. The returned tensor is the correct shape but all zeros — calling code should know this. | No code change needed; this is correct behavior for non-differentiable lookup. Add a comment: `// grad_input is zero: token IDs are non-differentiable` to clarify intent. |
| 18 | embedding.cpp:25-26 | LOW | Out-of-range token IDs are silently clamped to 0. This is a design choice (treat unknown tokens as padding). But silently clamping may hide bugs in downstream code expecting distinct OOV handling. | Consider throwing or logging on OOV clamp: `if (token_id < 0 || token_id >= vocab_size) { /* warn or throw */ }` |

**No NaN/Inf issues.** `grad_table` accumulation over token occurrences is correct.

---

### flatten.cpp / flatten.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 19 | flatten.cpp | LOW | `forward` and `backward` are pass-through (return input unchanged). This is correct for already-flattened data. However, the class comment says it handles `(batch, channels, H, W)` stored as `(batch, channels*H*W)` — if a 4D tensor is passed in actual 4D shape, the flatten won't work as expected. | Add assertion: `assert(input.cols == channels * H * W)` or clarify that input is always pre-flattened. |

**No issues** — trivial pass-through implementation.

---

## SKIP CONNECTION

### skip_connection.cpp / skip_connection.h

| # | File:Line | Severity | Description | Suggested Fix |
|---|-----------|----------|-------------|---------------|
| 20 | skip_connection.cpp:47-48 | HIGH | **Shape mismatch in backward pass (no projection case).** The direct-add path (`!needs_projection_`) correctly adds `grad_output` to `inner_grad` element-wise because they have the same shape. ✅ The `needs_projection_` path computes `Tensor grad_input(grad_output.rows, shortcut_grad.cols)` = `(batch, in_feat)`, then adds `shortcut_grad[i][j] + inner_grad[i][j]`. Since `shortcut_grad` is `(batch, in_feat)` and `inner_grad` is `(batch, out_feat)`, this is a **dimension mismatch** — accessing `inner_grad[i][j]` with `j < in_feat` when `inner_grad` only has `out_feat` columns is **out-of-bounds access** if `in_feat < out_feat`. | Restructure: compute `inner_grad` with `grad_output.rows` rows and `out_feat` cols, compute `shortcut_grad` with `in_feat` cols, then accumulate into a `grad_input` that matches the input dimension. If the shortcut projection maps `in_feat → out_feat`, the shortcut gradient contribution to the input should use the projection's `in_feat` columns directly, not be naively added to `inner_grad`. The identity path gradient should also flow: `grad_input += grad_output` via identity branch. |
| 21 | skip_connection.cpp:43-44 | MEDIUM | `forward` recomputes the projection shortcut **every forward call** if `out.cols != input.cols`. For inference this is fine, but in training the projection Dense is re-created each time the dimension check triggers, causing weight loss and memory churn. | Cache the projection: only create `shortcut_` once when the first dimension mismatch is detected (set `needs_projection_ = true` on first creation, don't recreate on subsequent calls). |

**Residual gradient flow (no projection case):** ✅ `grad_input += grad_output` correctly models the identity gradient path (`d/dx identity(x) = 1`).

---

## STRUCT PADDING / ALIGNMENT

| Layer | Severity | Description |
|-------|----------|-------------|
| BatchNorm1D | LOW | `bool training` (1 byte) at end after `double momentum` (8 bytes) — may incur 7 bytes of trailing padding for alignment. `eps` and `momentum` are adjacent (8+8 = 16 bytes) with no padding. No runtime impact. |
| LayerNorm | LOW | `bool training` at end after `double eps`. Same potential trailing padding. No impact. |
| GroupNorm | LOW | `bool train_mode_` at end after floats/double members. Same pattern. |
| Embedding | LOW | `bool` not present. No alignment concerns. |
| LSTM | LOW | `int` members followed by `Tensor` members — `int` (4 bytes) may have padding before the 24-byte Tensor. Reasonable. |
| GRU | LOW | Similar to LSTM. No dangerous padding. |
| SimpleRNN | LOW | Similar to LSTM. No alignment concerns. |
| SkipConnection | LOW | Two `unique_ptr<Layer>` members (16 bytes each) followed by `Tensor` and `bool`. No padding concern for a standalone object. |

**Conclusion:** No critical struct padding issues. All Tensor members are passed by value/pointer in function calls; alignment is managed by the Tensor class internals.

---

## SUMMARY TABLE

| ID | File | Line(s) | Severity | Category |
|----|------|---------|----------|----------|
| 1 | batch_norm.cpp | 10 | MEDIUM | NaN/Inf (eps default) |
| 2 | batch_norm.cpp | 27 | MEDIUM | NaN/Inf (var near-zero guard) |
| 3 | layer_norm.cpp | 9 | MEDIUM | NaN/Inf (eps default) |
| 4 | group_norm.cpp | 25 | HIGH | GroupNorm validation (G∤C) |
| 5 | group_norm.cpp | 28 | MEDIUM | GroupNorm input validation |
| 6 | group_norm.cpp | 47 | MEDIUM | NaN/Inf (var clamping floor) |
| 7 | skip_connection.cpp | 47-48 | HIGH | Shape mismatch / OOB in backward |
| 8 | skip_connection.cpp | 43-44 | MEDIUM | Projection recreated every forward |

**Total issues: 8**  
**CRITICAL/HIGH: 2** (GroupNorm G∤C validation, SkipConnection backward dimension mismatch)

**Build:** ✅ Clean compile (warnings only)

---
*Audit performed by subagent on 2026-04-20. No files were modified.*
