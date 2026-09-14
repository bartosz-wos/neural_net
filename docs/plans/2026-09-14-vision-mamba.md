# Vision Mamba (Vim) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a faithful C++ implementation of Zhu et al., ICLR 2024, "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Models" (https://arxiv.org/abs/2401.09417) as `PatchEmbed` + `VimBlock` + `VimModel`. Vim replaces the quadratic-cost self-attention in ViT with a bidirectional Mamba (state-space) mixer; the rest of the ViT recipe (patch embedding, class token, position embedding, FFN) is unchanged. The distinctive ingredient is **bidirectional SSM scanning** — a forward Mamba and a backward Mamba, both on the same patch sequence, summed.

**Architecture:**
- `PatchEmbed(C_in, D, H, W, patch_size)` — wraps a Conv2D with `kernel=patch_size, stride=patch_size` so it does non-overlapping patch tokenisation + linear projection in one step. Forwards `(1, C_in·H·W)` → `(N_p, D)` where `N_p = (H/p)·(W/p)`. Backward runs `Conv2D::backward` on the reshaped `(1, D·N_p)` representation and reshapes `grad_input` back to `(1, C_in·H·W)`.
- `VimBlock(D, d_state, d_inner, ffn_mult)` — pre-norm residual structure: `mixer = ln1 → [fwd_mamba + bwd_mamba]`, `ffn = ln2 → GELU(W1·x)·W2`. Both branches use the existing `MambaBlock` (forward and a separate one for backward). Output is `x + mixer + ffn`. Backward propagates through both SSM branches and the FFN.
- `VimModel(C_in, H, W, patch_size, D, num_classes, d_state, d_inner, num_layers, ffn_mult)` — `PatchEmbed → add_class_token_and_pos_embed → N · VimBlock → final LN → take_class_token → classifier`. Class token is learnable `(1, D)`, position embedding is learnable `(1 + N_p, D)`.

**Tech Stack:** Existing `Conv2D` (from `include/nn/layers/convolutions/conv_layer.h`), `MambaBlock` (from `include/nn/layers/recurrent/mamba.h`), `Dense`, `LayerNorm`. New `vision_mamba.{h,cpp}` under `include/nn/layers/architectures/`.

**Design choices:**
- **Single-image forward (N=1) for v1.** The repo's flat-tensor convention makes per-sample patch sequences easy (one (T, D) tensor) but per-batch (N · T, D) flattening is a separate problem. The paper itself does batch training; for v1 we focus on the per-image case so the gradient check is tractable. Adding a batch wrapper is straightforward later.
- **Bidirectional Mamba via two `MambaBlock`s.** The forward Mamba scans the sequence left-to-right; the backward Mamba scans the sequence right-to-left (input reversed at forward, output reversed back at the end of the block). Both share the same `(T, D)` shape; the residual sum makes the math identical to two parallel streams. We don't tie the weights (paper §3.1 uses independent forward/backward).
- **GELU FFN** (matches the paper's §3.2 — `out = x + W_2 · GELU(W_1 · LN(x))`).
- **Position embedding is learnable**, no sinusoidal / 2D-aware interpolation. Default ViT recipe.
- **Class token prepended, not pooled.** Standard ViT.

---

## Distinctive vs existing

The repo already has many SSM/attention-based vision and sequence architectures:
- `MambaBlock` — unidirectional SSM (used inside Vim for each direction).
- `S5Block` — bidirectional S5 (different SSM, different paper).
- `vit` — standard ViT (attention-based, our Vim is the SSM-based sibling).

What's NEW:
1. **PatchEmbed wrapper** — non-overlapping Conv2D-based tokeniser that flattens a 3-channel image into a `(N_p, D)` sequence. Not present elsewhere.
2. **Bidirectional Mamba** — two MambaBlocks (forward + reversed), summed. Not present elsewhere. Closest existing is S5Block which has its own bidirectional mechanism.
3. **Vim-specific block structure** — pre-norm residual around bidirectional SSM + FFN. Different from `vit_block` (which uses post-LN or pre-LN attention + FFN).

---

## Task 1: PatchEmbed — RED 1 (constructor + forward shape)

**Files:**
- Create: `include/nn/layers/architectures/vision_mamba.h`
- Create: `include/nn/layers/architectures/vision_mamba.cpp`
- Create: `tests/test_vision_mamba.cpp`

**Step 1:** In `vision_mamba.h`, declare `PatchEmbed` and the supporting public types. Stub `forward` to throw.

```cpp
#ifndef VISION_MAMBA_H
#define VISION_MAMBA_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include "../convolutions/conv_layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/mamba.h"
#include "../core/dense.h"
#include <vector>
#include <memory>

// PatchEmbed: Conv2D with kernel=patch_size, stride=patch_size.
// Wraps the existing Conv2D to flatten image → patch-sequence.
class PatchEmbed : public Layer {
public:
    PatchEmbed(size_t in_channels, size_t embed_dim, size_t H, size_t W, size_t patch_size);

    // Forward: (1, in_channels*H*W) -> (N_p, embed_dim) where N_p = (H/p)*(W/p)
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return conv_.get_weights(); }
    Tensor get_gradients() const override { return conv_.get_gradients(); }
    std::string name() const override { return "PatchEmbed"; }

    size_t in_channels() const { return in_channels_; }
    size_t embed_dim() const { return embed_dim_; }
    size_t H() const { return H_; }
    size_t W() const { return W_; }
    size_t patch_size() const { return patch_size_; }
    size_t num_patches() const { return num_patches_; }

    // Internal — public for tests
    Conv2D conv_;                   // kernel=patch_size, stride=patch_size
    size_t in_channels_;
    size_t embed_dim_;
    size_t H_;
    size_t W_;
    size_t patch_size_;
    size_t num_patches_;            // (H/p) * (W/p)

    // Forward cache
    Tensor last_image_;             // (1, C*H*W)
    Tensor last_patches_;           // (N_p, D) — the forward output (for backward reshape)
};

#endif // VISION_MAMBA_H
```

**Step 2:** In `vision_mamba.cpp`, implement the constructor (validates `H % patch_size == 0` and `W % patch_size == 0`, computes `num_patches = (H/patch_size) * (W/patch_size)`, instantiates the underlying `Conv2D` with `in_ch=in_channels, out_ch=embed_dim, kernel=patch_size, stride=patch_size, pad=0`). Implement `forward` to call `conv_.forward(image)` which returns `(1, embed_dim * num_patches)`, then reshape to `(num_patches, embed_dim)`. Implement `backward(grad_output)` to reshape `grad_output` (N_p, D) → `(1, D * N_p)` (the *permutation* — element `(p, d)` becomes column `d * N_p + p`), then call `conv_.backward` and return the image-shaped gradient.

**Step 3:** In `tests/test_vision_mamba.cpp`, write a tiny test that:
- Constructs `PatchEmbed(3, 8, 8, 8, 4)` (8×8 RGB image, patch=4 → N_p=4 patches).
- Builds a `(1, 3*8*8)` input with `seed=42, scale=0.1`.
- Runs forward, asserts output shape `(4, 8)`.
- Asserts finite, nonzero.
- Asserts each row of the output equals the conv2d output reshaped (compare to a reference manual computation).
- Verifies backward: perturb `(1, 3*8*8)[0, k]` by ±ε, re-forward, check `(out[p][d] - out[p][d]_no_perturb) / (2ε) ≈ backward_input_grad[0][k]` to rel_err < 1e-4.

**Step 4:** Verify RED (test fails because class doesn't exist yet — `g++ -c -Iinclude` errors on the missing header).

**Step 5:** Implement minimal code in `vision_mamba.cpp`. The forward reshape is:
```
for p in 0..N_p-1:
    for d in 0..D-1:
        out[p][d] = conv_out[0][d * N_p + p]
```
The backward is the inverse permutation.

**Step 6:** Verify GREEN.

**Step 7:** Commit: `feat(architectures): PatchEmbed wrapper for Vision Mamba`

---

## Task 2: VimBlock — RED 1 (constructor + forward shape)

**Files:**
- Modify: `include/nn/layers/architectures/vision_mamba.h`
- Modify: `include/nn/layers/architectures/vision_mamba.cpp`
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** In `vision_mamba.h`, declare `VimBlock(D, d_state, d_inner=0, ffn_mult=4)` — pre-norm residual block: `mixer = ln1 + fwd_mamba + bwd_mamba`, residual; `ffn = ln2 + GELU(W1) + W2`, residual. Two `MambaBlock` instances (`mamba_fwd`, `mamba_bwd`). Two `Dense` layers for FFN.

**Step 2:** In `vision_mamba.cpp`, implement the constructor (validates d_model > 0, ffn_mult > 0; computes `ffn_hidden = d_model * ffn_mult`). Implement `forward`:
1. Cache input
2. `ln1_out = ln1_.forward(x)`
3. `fwd_out = mamba_fwd_.forward(ln1_out)` (shape (T, D))
4. Reverse ln1_out along the time axis (flip rows), `bwd_out = mamba_bwd_.forward(reversed)` (shape (T, D)), then flip bwd_out back along rows.
5. `mixer = fwd_out + bwd_out`
6. `res1 = x + mixer`
7. `ln2_out = ln2_.forward(res1)`
8. `ffn_hidden = ffn1_.forward(ln2_out)` (shape (T, ffn_hidden))
9. `ffn_act = gelu(ffn_hidden)`
10. `ffn_out = ffn2_.forward(ffn_act)` (shape (T, D))
11. `out = res1 + ffn_out`
12. Cache all intermediates (`last_ln1_out`, `last_fwd_out`, `last_bwd_out`, `last_res1`, `last_ln2_out`, `last_ffn_hidden`, `last_ffn_act`, `last_ffn_out`, `last_input`).

**Step 3:** Add Test 2 in `tests/test_vision_mamba.cpp`:
- Construct `VimBlock(D=4, d_state=2, ffn_mult=2)` — small.
- Input `(6, 4)`.
- Forward shape `(6, 4)`.
- Finite, nonzero.
- Identity-init sanity: set both `mamba_fwd_` and `mamba_bwd_` to output `ln1_out` directly (skipping the SSM — too tricky to set up; instead just verify the block's output is not constant and depends on ln1). Simpler sanity: assert the residual contributes to the gradient (perturb input, check output moves).

**Step 4:** Verify RED (constructor / forward not yet wired).

**Step 5:** Implement minimal code.

**Step 6:** Verify GREEN.

**Step 7:** Commit: `feat(architectures): VimBlock forward (bidirectional Mamba + FFN, pre-norm residual)`

---

## Task 3: VimBlock backward — RED 1 (input gradient)

**Files:**
- Modify: `include/nn/layers/architectures/vision_mamba.cpp`
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Implement `VimBlock::backward(grad_output, lr)`:
1. `grad_ffn_out = grad_output` (residual at the end).
2. `grad_res1 = grad_ffn_out` (residual sum means `d(out)/d(res1) = I + d(ffn)/d(res1)`; we'll add the path through FFN later). Compute `grad_ln2_out = ffn2_.backward(grad_ffn_out, lr)` → `grad_ffn_act = gelu_derivative(ffn_hidden) ⊙ grad_ln2_out` → `grad_ln2_out = ffn1_.backward(grad_ffn_act, lr)`.
3. `grad_res1_from_ffn = ln2_.backward(grad_ln2_out, lr)` (post-LN backward through `ln2_`).
4. `grad_mixer = grad_res1 + grad_res1_from_ffn` (split residual at `out = res1 + mixer`).
5. `grad_ln1_fwd = mamba_fwd_.backward(grad_mixer, lr)`.
6. Reverse `grad_mixer` along rows to align with the reversed-input bwd pass; `grad_ln1_bwd = mamba_bwd_.backward(reversed_grad_mixer, lr)`; reverse `grad_ln1_bwd` back.
7. `grad_ln1 = grad_ln1_fwd + grad_ln1_bwd` (the input to both Mambas is `ln1_out`).
8. `grad_input_from_mixer = ln1_.backward(grad_ln1, lr)` (post-LN backward through `ln1_`).
9. `grad_input = grad_res1 + grad_input_from_mixer` (residual at `res1 = x + mixer`).

**Step 2:** Add Test 3: input gradient via centered FD on `VimBlock`:
- Construct `VimBlock(D=4, d_state=2, ffn_mult=2)` with seeded random init.
- Input `(6, 4)` random.
- Forward → output.
- Backward with `grad_output = ones(6, 4)` (uniform gradient).
- Compare analytical `grad_input` to centered FD with ε=1e-4 on every input element → rel_err < 1e-3 (relaxed because of the LN+SSM chain).

**Step 3:** Verify RED.

**Step 4:** Implement.

**Step 5:** Verify GREEN. **Likely bugs:** (a) forgetting to reverse the bwd-mamba gradient back, (b) double-counting or missing one of the residual splits, (c) GELU-derivative element-wise multiply, (d) `MambaBlock::backward` requires the upstream gradient in `(T, d_model)` shape — make sure the reshape is right.

**Step 6:** Commit: `feat(architectures): VimBlock backward (FD input gradient passes at rel_err < 1e-3)`

---

## Task 4: VimBlock parameter gradients — RED 1 (W_mamba and W_ffn via FD)

**Files:**
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Add Test 4:
- Construct VimBlock(D=4, d_state=2, ffn_mult=2).
- Forward with `(6, 4)` random input.
- Backward with `grad_output = ones(6, 4)`.
- Compare each public parameter gradient (`mamba_fwd_.in_proj.grad_weights`, `mamba_fwd_.out_proj.grad_weights`, `mamba_bwd_.in_proj.grad_weights`, `mamba_bwd_.out_proj.grad_weights`, `ffn_proj1_.grad_weights`, `ffn_proj2_.grad_weights`, `ln1_.grad_gamma_`, `ln2_.grad_gamma_`) to centered FD with ε=1e-4.
- Accept rel_err < 1e-3 for FD-gradient mismatch (LN+SSM chain has tight precision; 1e-3 is the standard tolerance for this depth of chain).

**Step 2:** Implement — should already pass if Task 3 is correct.

**Step 3:** Verify GREEN.

**Step 4:** Commit: `test(architectures): VimBlock FD parameter gradients (8 param groups)`

---

## Task 5: VimModel — RED 1 (constructor + forward shape)

**Files:**
- Modify: `include/nn/layers/architectures/vision_mamba.h`
- Modify: `include/nn/layers/architectures/vision_mamba.cpp`
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Declare `VimModel(C_in, H, W, patch_size, D, num_classes, d_state, d_inner, num_layers, ffn_mult)`:
- `PatchEmbed patch_embed_`
- `Tensor class_token_` (1, D) — learnable
- `Tensor pos_embed_` (1+N_p, D) — learnable
- `std::vector<std::unique_ptr<VimBlock>> blocks_`
- `LayerNorm final_ln_`
- `Dense classifier_` (D → num_classes)

**Step 2:** Implement forward:
1. `patches = patch_embed_.forward(image)` → (N_p, D)
2. Build the full sequence by concatenating `class_token_` (shape (1, D)) with `patches` → (1+N_p, D).
3. Add `pos_embed_`.
4. Loop through blocks: `x = blocks_[i]->forward(x)` for each i.
5. `x = final_ln_.forward(x)`.
6. Take the first row (class token): `cls = x[0]` (or use `x.get_row(0)`).
7. `logits = classifier_.forward(cls)` → (1, num_classes).

**Step 3:** Add Test 5:
- Construct `VimModel(C_in=3, H=8, W=8, patch_size=4, D=8, num_classes=3, d_state=2, num_layers=2)`.
- Input `(1, 3*8*8)` random.
- Forward → `(1, 3)` logits.
- Finite, nonzero.
- Per-block count: 2 blocks (`num_patches = 4`, each block has its own SSM, LN, FFN).

**Step 4:** Verify RED.

**Step 5:** Implement.

**Step 6:** Verify GREEN.

**Step 7:** Commit: `feat(architectures): VimModel forward (PatchEmbed + class token + pos embed + N VimBlocks + classifier)`

---

## Task 6: VimModel backward — RED 1 (input gradient via FD)

**Files:**
- Modify: `include/nn/layers/architectures/vision_mamba.cpp`
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Implement `VimModel::backward(grad_output, lr)`:
1. `grad_cls = classifier_.backward(grad_output, lr)` → (1, D).
2. Build a `(1+N_p, D)` gradient with row 0 = `grad_cls`, rows 1..N_p = zeros.
3. `grad_after_blocks = final_ln_.backward(grad_seq, lr)`.
4. Loop backwards: `grad_after_blocks = blocks_[i]->backward(grad_after_blocks, lr)`.
5. `grad_patches_with_cls = grad_after_blocks`. Subtract `pos_embed_`'s gradient from the add-step (the gradient is `grad_patches_with_cls` itself for the pos_embed, and `grad_patches_with_cls` minus that for the patches). Actually since pos_embed is just added, `grad_seq = grad_after_blocks` (the add distributes uniformly).
6. Take rows 1..N_p to get `grad_patches` (skip the class token row).
7. Reshape `(N_p, D) → (1, D * N_p)` (inverse of patch_embed forward), then `grad_image = patch_embed_.backward(grad_patches_reshaped, lr)`.

**Step 2:** Add Test 6: input gradient via centered FD on `VimModel` (rel_err < 5e-3 — the multi-block + pos-embed chain is loose).

**Step 3:** Verify RED.

**Step 4:** Implement.

**Step 5:** Verify GREEN. **Likely bugs:** (a) forgetting to add pos_embed_grad as the same gradient as patches (add → d_patches = grad_seq, d_pos_embed = grad_seq), (b) slicing off the class token row before patch_embed.backward, (c) the inverse patch-embed reshape is the same permutation as forward — make sure the directions match.

**Step 6:** Commit: `feat(architectures): VimModel backward (input gradient via FD passes)`

---

## Task 7: Training reduces loss (end-to-end)

**Files:**
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Add Test 7:
- Build a `VimModel(C_in=3, H=8, W=8, patch_size=4, D=8, num_classes=3, d_state=2, num_layers=2)`.
- Construct a small dataset: 8 random images (3*8*8 each), 8 labels (one-hot 3-class).
- For 60 SGD steps at lr=0.01:
  - Forward on each image, get logits.
  - Compute MSE loss vs one-hot labels.
  - Backward, zero_grad, update_weights.
- Assert loss decreased by > 30%.

**Step 2:** Implement (no new code needed; just runs forward/backward/update in a loop).

**Step 3:** Verify GREEN.

**Step 4:** Commit: `test(architectures): VimModel end-to-end training reduces MSE loss > 30% over 60 SGD steps`

---

## Task 8: Mutation tests (non-vacuousness)

**Files:**
- Modify: `tests/test_vision_mamba.cpp`

**Step 1:** Add a mutation test that proves the bidirectional Mamba is wired in:
- Construct `VimBlock(D=4, d_state=2, ffn_mult=2)`.
- Set `mamba_bwd_.in_proj.weights` to all zeros (kills the backward SSM).
- Forward on `(6, 4)` random input.
- Assert: the output of the full VimBlock changes by more than 1e-3 compared to the un-mutated version. If the bidirectional Mamba is wired correctly (and the gradient flows through both branches), zeroing the backward branch must change the output.

**Step 2:** Add a mutation test for class-token + pos-embed wiring:
- Construct `VimModel`.
- Set `pos_embed_` to all zeros. Set `class_token_` to all zeros.
- Forward → assert logits change vs the un-mutated (because class token + pos embed are now non-trivial inputs; if they aren't wired, output is unchanged).

**Step 3:** Verify both fail under mutation.

**Step 4:** Commit: `test(architectures): VimBlock + VimModel mutation tests (non-vacuousness)`

---

## Task 9: Registration (umbrella + Makefile + plan summary)

**Files:**
- Modify: `include/nn/nn.h`
- Modify: `Makefile`

**Step 1:** Add `#include "layers/architectures/vision_mamba.h"` to `include/nn/nn.h` (next to `xlstm_block.h`).

**Step 2:** Verify the umbrella compiles standalone:
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

**Step 3:** Add `build/test_vision_mamba` rule to Makefile:
- `$(BUILD_DIR)/test_vision_mamba: $(LIB_OBJS) $(BUILD_DIR)/test_vision_mamba.o` linking with the test object.
- Add `$(BUILD_DIR)/test_vision_mamba` to the `tests:` aggregate.
- Add `echo "=== Running Vision Mamba Tests ===" && ./$(BUILD_DIR)/test_vision_mamba` to `run_tests`.

**Step 4:** Run full suite `make run_tests`. Confirm:
- `test_vision_mamba` passes all checks.
- No regressions in adjacent tests (mamba, vit, attention, etc.).

**Step 5:** Move the queue entry from Ideas to Done in `EXPANSION_QUEUE.md` with a one-line summary.

**Step 6:** Commit:
- `chore: register Vision Mamba in umbrella nn.h and Makefile`
- `docs: mark Vision Mamba as Done in EXPANSION_QUEUE`

---

## Verification checklist

- [ ] All 8 tasks RED → GREEN with real `terminal` output
- [ ] `tests/test_vision_mamba.cpp` ≥ 7 tests pass
- [ ] No new warnings under `-Wall -Wextra`
- [ ] `make run_tests` passes with no regressions in adjacent suites
- [ ] `include/nn/nn.h` umbrella compiles standalone
- [ ] Mutation tests confirm non-vacuous coverage
- [ ] Clean working tree, all commits pushed