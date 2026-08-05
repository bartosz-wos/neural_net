# Conformer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the **Conformer** block and model (Gulati et al., 2020, "Conformer: Convolution-augmented Transformer for Speech Recognition", https://arxiv.org/abs/2010.05656) — a transformer architecture that sandwiches each block with macaron-style half-step feed-forward modules and inserts a 1D depthwise convolution module between attention and the second feed-forward. Used in Whisper's encoder, USM, wav2vec 2.0, and modern speech recognition systems.

**Architecture:** Three classes:

1. **`FeedForward`** (no activation wrapper, dim, expansion=4, dropout=0.0) — used twice in each block. Implements `FFN(x) = SiLU(W1·x + b1) · dropout(W2·x + b2)`. Half-step residual is applied at the model level (the *half* is the canonical Conformer convention — see x → x + 0.5·FFN(x)).
2. **`ConvModule`** (dim, kernel_size=15) — LayerNorm → pointwise conv (1×1, expand to 2×dim) → GLU → 1D depthwise conv (kernel=15, padding=causal/symmetric) → BatchNorm → SiLU → pointwise conv (1×1, back to dim) → residual. Uses standard `Conv1D` (with `in_ch == out_ch == dim` for the depthwise stage, which gives a per-channel filter — strict depthwise would set `groups == dim`; we use the equivalent in_ch == out_ch with each filter operating on one channel via dense sampling, which is mathematically equivalent when the kernel mask is identity-like; we opt for the simpler per-channel projection).
3. **`ConformerBlock`** (dim, num_heads, ffn_expansion=4, conv_kernel_size=15) — `x' = x + 0.5·FFN1(x)`; `x'' = x' + MHSA(LN(x'))`; `x''' = x'' + ConvModule(LN(x''))`; `out = x''' + 0.5·FFN2(LN(x'''))`; then final `LN(out)`.
4. **`ConformerModel`** (input_dim, num_classes, dim=128, depth=4, num_heads=4, ffn_expansion=4, conv_kernel_size=15) — input projection (Linear from input_dim → dim) → stack of `ConformerBlock`s → mean-pool over time → LN → linear classifier (dim → num_classes).

**Why this addition:** The repo already has a rich set of transformers (vanilla, GQA, Flash, MLA, Performer, Linformer, BigBird, RoPE, ALiBi) and a wide SSM family (Mamba, Mamba-2, RWKV, RetNet, mLSTM, sLSTM, xLSTM, S4). However, **no conv-augmented transformer** — and Conformer is the canonical example used in production ASR (Whisper's encoder is essentially Conformer). Adding it gives the repo a distinct third path: (a) pure-attention transformers, (b) state-space / recurrence sequence models, and now (c) hybrid convolution + attention.

**Tech Stack:** Existing — `Tensor`, `Dense`, `LayerNorm`, `MultiHeadAttention`, GELU activation from `activations.h`, `Conv1D` for the depthwise convolution module.

**Reference:**
- Paper: https://arxiv.org/abs/2010.05656 (Gulati et al., 2020)
- Reference impl: https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/conformer_encoder.py

**File layout (matches repo convention):**
- Header: `include/nn/layers/architectures/conformer.{h,cpp}`
- Tests: `tests/test_conformer.cpp`
- Plan: `docs/plans/2026-08-05-conformer.md`
- Update `include/nn/nn.h` (umbrella include)
- Update `Makefile` (build target + `run_tests:` block)

---

## Mathematical Specification

### Conformer Block (the canonical sandwich)

```
def conformer_block(x):
    x = x + 0.5 * ffn1(layer_norm(x))                  # macaron FFN ½
    x = x + mhsa(layer_norm(x))                        # multi-head self-attention
    x = x + conv_module(layer_norm(x))                 # 1D conv module
    x = x + 0.5 * ffn2(layer_norm(x))                  # macaron FFN ½
    return layer_norm(x)
```

The "macaron" half-step FFNs at both ends is the distinctive Conformer choice (vs vanilla Transformer's single full-step FFN).

### FFN Module

```
ffn(x) = linear2(silu(linear1(x)))
linear1: dim → dim*ffn_expansion
linear2: dim*ffn_expansion → dim
```

Default `ffn_expansion = 4` (vs Transformer's `ffn_dim = 4*dim` too — same width).

### Conv Module

```
def conv_module(x):                       # x: (batch, T, dim) — channels-last
    x = layer_norm(x)                     # per-token LN over dim
    x = pointwise_conv1(x, dim -> 2*dim)  # expand with Linear
    x = glu(x, axis=-1)                   # GLU: split last dim in 2 halves, gate: a * sigmoid(b)
    x = depthwise_conv1d(x, kernel=15)    # per-channel filter
    x = batch_norm(x)
    x = silu(x)
    x = pointwise_conv2(x, dim -> dim)    # project back
    return x
```

We implement depthwise via per-channel Conv1D (in_ch == out_ch == dim with an identity-like filter pattern).

### Final LayerNorm

Following the canonical Conformer, there's a final LayerNorm after the last block of the model (sometimes also after every block — espnet uses both; we use a final-only LN for the model output).

### ConformerModel

```
def conformer_model(x):                   # x: (batch, T, input_dim)
    x = input_projection(x)               # Linear: input_dim -> dim
    for _ in range(depth):
        x = conformer_block(x)
    x = mean(x, dim=time)                 # average pool over time
    x = layer_norm(x)
    x = classifier(x)                     # Linear: dim -> num_classes
    return x
```

### Backward

Each sub-component (FFN, MHSA, ConvModule) handles its own backward. The block backward is:
1. Final LN backward with `grad_out` → `grad_pre_final_ln`
2. ffn2 backward: `grad_in = grad_out` (residual) + `0.5 · ffn2.backward(LN.backward(grad_out))`
3. conv_module backward: same pattern
4. mhsa backward: same pattern
5. ffn1 backward: `grad_in = grad_in` (residual) + `0.5 · ffn1.backward(LN.backward(grad_in))`
6. Returns `grad_in`

### Mathematical correctness

For the FD gradient check to be tractable:
- Use `Conv1D` with `kernel_size = conv_kernel_size` and `pad = conv_kernel_size / 2` (symmetric padding, valid for offline ASR — Whisper uses the same convention)
- BatchNorm in eval mode (constant scale=1, shift=0) during FD check to keep the loss surface smooth
- Small `dim` and `depth` for FD tests (e.g., `dim=8, depth=2, num_heads=2, conv_kernel_size=5`)

---

## Step-by-step Implementation

### Task 1: Header skeleton + FFN class
- Create `include/nn/layers/architectures/conformer.h` with `class FeedForward`.
- Declare all classes, all methods.

### Task 2: FFN forward + backward
- Two Dense layers, SiLU on first.
- Cache input for backward.
- Backward: standard Dense-backward chain.

### Task 3: ConvModule
- LayerNorm → Dense(expand) → GLU → Conv1D(depthwise) → BatchNorm(eval-mode=identity) → SiLU → Dense(project back).
- Cache all intermediates for backward.
- Backward: reverse the chain (start with `grad_out` and walk back through Conv1D, BN, SiLU, GLU, expand Dense, LN).

### Task 4: ConformerBlock
- 4 sub-modules: ffn1, mhsa, conv, ffn2, 5 LayerNorms (one before each sub-module + one final).
- Sandwich residual structure with 0.5·FFN weights.
- Backward walks back through: final LN → ffn2 → LN → conv → LN → mhsa → LN → ffn1 → LN.

### Task 5: ConformerModel
- Input projection Dense → stack of blocks → mean-pool over time → LayerNorm → classifier Dense.

### Task 6: Tests
- 16+ focused checks: forward shapes, gradient FD checks for each parameter (W1, W2 in FFN; LN gamma/beta; ConvModule W; classifier W), training reduces loss, determinism, etc.

### Task 7: Register
- Add to `include/nn/nn.h`.
- Add `$(BUILD_DIR)/test_conformer` to Makefile (build target + tests/run_tests).
- Commit + push.

---

## Verification

Run `./build/test_conformer` — all 16+ checks must pass at machine precision (rel_err < 1e-6) for the FD gradient checks. Run `./build/test_gradient_check` and a small subset of the existing test suite (`make tests`) to ensure no regressions.
