# Monarch Mixer Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Implement **Monarch Mixer** (Dao, Gu, Ermon, Rudra, Ma 2022, https://arxiv.org/abs/2204.00945, "Monarch: Expressive Structured Matrices for Efficient Accurate Deep Learning") — a sequence+channel mixer that replaces both self-attention AND MLP sublayers with **Monarch matrices** (block-diagonal × permutation structured matrices). The result is a drop-in Transformer alternative with O(N · log N) sequence mixing and O(d · log d) channel mixing, both strictly cheaper than the O(N²) attention or O(d²) MLP they replace.

**Architecture:** Four classes — (1) `MonarchMatrix(n, b)` — a single Monarch matrix as a `Layer`: applies an `(n, n)` structured matrix `M = P · B · Q` to a `(batch, n)` input, where `B` is block-diagonal with `n/b²` blocks of size `(b, b)` and `P, Q` are stored permutations. (2) `MonarchSequenceMix(seq_len, d_model, block_size, num_blocks)` — applies N Monarch matrices over the sequence axis, each on a `d_model`-channel stream (per-token Dense-then-Monarch pattern). (3) `MonarchChannelMix(seq_len, d_model, block_size)` — applies Monarch matrices over the channel axis. (4) `MonarchMixerBlock(seq_len, d_model, block_size_seq, block_size_ch, ffn_mult)` — pre-LN + MonarchSequenceMix + residual + pre-LN + MonarchChannelMix + residual. (5) `MonarchMixerModel(input_dim, d_model, output_dim, seq_len, num_blocks, block_size_seq, block_size_ch, ffn_mult)` — full embed + stack + final LN + classifier.

**Math:**
- **Monarch matrix** `M = P · B · Q` applied to `x ∈ R^{B × n}` gives `Mx^T = P · (B · (Q · x^T))`. With `n = m²·b²` (block-diagonal has `m²` blocks of `b²`), the parameter count is `m²·b² + n` (block params + permutation indices, stored as 2 permutation vectors of size n).
- **Monarch matrix is closed under multiplication and inversion** (paper §2.2) — but v1 only needs the forward + backward path.
- **Backward** (hand-derived): given `dM·x = dL/dy`:
  - `dx = Q^T · B^T · P^T · dL/dy` (linear chain, transpose of forward ops)
  - `dB = P^T · (dL/dy) · x^T · Q^T` accumulated per block (B, b, b)
  - Permutations are NOT learned (paper convention; we initialize P, Q randomly and freeze them) so no permutation gradient.
- **Block-diagonal forward** `B · z` where `z ∈ R^{n}` reshapes to `(m, m·b, b)` and applies per-block matmul `(b, b) · (b,)` (or similar).
- **Channel mixing** uses Monarch matrices applied per-token on the channel dim: `(d_model,)` → `(d_model,)` via `MonarchMatrix(d_model, b_ch)`.
- **Sequence mixing** uses Monarch matrices applied per-channel on the sequence dim: each of the `d_model` channels gets its own Monarch over the sequence: `(seq_len,)` → `(seq_len,)` via `MonarchMatrix(seq_len, b_seq)`. Per-channel weights are stored as a `(d_model, n_blocks, b, b)` block-diagonal tensor (NOT shared across channels).

**Tech Stack:** C++17, `Tensor`, hand-derived backward (permutation × block-diagonal chain), no Dense dependency.

## Design Decisions

- **Block size constraints**: `b` must divide `√n` evenly (so the matrix can be partitioned into `n/b²` square blocks of `(b, b)`). Constructor throws otherwise.
- **Permutations are frozen**: the matrix `M` itself is learnable in its block-diagonal entries but the permutation is fixed at init (this is the paper convention — Monarch is structured but not "fully learnable" in the permutation sense; learnable permutations are out of scope).
- **`num_blocks=1` (default)**: a single Monarch matrix per layer. Increasing `num_blocks > 1` stacks multiple Monarch matrices back-to-back (paper §2.4 — "concatenating Monarch matrices gives strictly more expressivity"; `M = M_k · … · M_2 · M_1` is still a Monarch matrix).
- **Permutation initialization**: a random permutation sampled uniformly from `std::mt19937`. Tests verify that two different seeds produce different permutations (signature of "permutations are actually random, not a fixed pattern") and that the same seed is deterministic.
- **Forward cache**: caches `Q · x`, `B · (Q · x)`, `P · (B · (Q · x))` so backward can walk the chain cheaply.
- **Layer contract**: each class implements `forward`, `backward`, `update_weights`, `zero_grad`, `parameters`, `gradients`, `get_weights`, `get_gradients`, `name`.
- **No LayerNorm inheritance on MonarchMatrix itself** — it is a primitive. But `MonarchMixerBlock` and `MonarchMixerModel` use LayerNorm for the residual structure.
- **`num_blocks` validation**: `num_blocks >= 1`.
- **For tests**: a hand-derived 4×4 Monarch with `b=2` should give bit-exact identity when `B = I` and `P = Q = identity`.

## Conventions (file layout)

- Header: `include/nn/layers/architectures/monarch_mixer.h`
- Implementation: `include/nn/layers/architectures/monarch_mixer.cpp`
- Test: `tests/test_monarch_mixer.cpp`
- Plan: `docs/plans/2026-09-19-monarch-mixer.md`
- Umbrella registration: `include/nn/nn.h` (architectures section)
- Makefile: `build/test_monarch_mixer` rule, `tests:` deps, `=== Running Monarch Mixer Tests ===` echo in `run_tests`

## Phase 0 — Setup files

### Task 0.1: Create header `include/nn/layers/architectures/monarch_mixer.h`

```cpp
#ifndef MONARCH_MIXER_H
#define MONARCH_MIXER_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <cstdint>

// ============================================================================
// Monarch Matrix (Dao et al. 2022)
//   "Monarch: Expressive Structured Matrices for Efficient Accurate Deep Learning"
//   https://arxiv.org/abs/2204.00945
//
// A Monarch matrix of dim n with block-size b is M = P · B · Q where:
//   - P, Q are fixed permutations of size n (random init, frozen during training)
//   - B is block-diagonal with n/b² blocks of size (b, b) (the only learnable part)
// Total params: n/b² · b² = n (in B) + 2n (two permutation vectors) — wait, that's 3n.
// Actually the perms are index maps so they cost n + n = 2n scalars (int). Block storage
// is n floats (the (b,b) blocks hold b² values each, there are n/b² of them). So total
// scalar count is n (block values) + 2n (perm indices as size_t) = 3n.
//
// Constraints:
//   - n must be a perfect square (n = m²) so the block-diagonal is square
//   - b² must divide n (block count = n/b² must be a perfect square itself, i.e. (m/b)²)
//   - b must divide m (= √n)
//
// Forward (applies M to a (B, n) input, returns (B, n)):
//   y = M @ x^T   →   y^T = P · B · Q · x^T
//     = P · (B · (Q · x^T))
// Implementation: permute cols via Q, apply per-block matmul, permute via P.
//
// Backward:
//   dx = M^T · dy = Q^T · B^T · P^T · dy
//   dB_blocks[b_idx][i, j] += sum_t P^T(dy)[t, b_idx·b² + j·b + i] · x[t, ...]
//   (more precisely: dB = P^T · dy · x^T · Q^T summed across the block index)
//
// Permutations are FROZEN: no gradient flows through them. They are stored as
// std::vector<size_t>.
// ============================================================================

class MonarchMatrix : public Layer {
public:
    // n: matrix dimension (must be a perfect square)
    // b: block side (must divide sqrt(n) evenly)
    // perm_seed: RNG seed for the random permutations P, Q
    MonarchMatrix(size_t n, size_t b, uint32_t perm_seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMatrix"; }

    size_t n() const { return n_; }
    size_t b() const { return b_; }
    size_t num_blocks() const { return n_ / (b_ * b_); }
    size_t m() const { return m_; }  // sqrt(n)

    const std::vector<size_t>& P() const { return perm_P_; }
    const std::vector<size_t>& Q() const { return perm_Q_; }
    // block_values_: stored as a flat (num_blocks * b * b) tensor for simplicity
    // The (b, b) blocks are accessed as block_values_[block_idx * b * b + i * b + j].
    const Tensor& block_values() const { return block_values_; }
    Tensor& block_values() { return block_values_; }

private:
    size_t n_;
    size_t b_;
    size_t m_;                  // sqrt(n)
    size_t num_blocks_;         // n / b²

    std::vector<size_t> perm_P_;  // forward output perm: y[t, P[i]] = intermediate[t, i]
    std::vector<size_t> perm_Q_;  // forward input perm: intermediate[t, i] = input[t, Q[i]]

    Tensor block_values_;       // (num_blocks, b, b) — stored flat as (num_blocks * b * b,)
    Tensor grad_block_values_;  // same shape

    // Caches
    Tensor last_input_;          // (B, n)
    Tensor last_after_Q_;        // (B, n)  = input with cols permuted by Q
    Tensor last_after_BQ_;       // (B, n)  = block-diag applied to last_after_Q_
    // last_output_ is the forward return value — no separate cache needed since
    // grad_output arrives in backward().
};

// ============================================================================
// MonarchSequenceMix — applies a stack of MonarchMatrix(seq_len, b_seq) over
// the sequence axis. Each channel of d_model gets its OWN block_values tensor
// (per-channel weights, NOT shared across channels — this is what makes it
// a "sequence mixer": each channel can mix the seq_len positions differently).
// num_blocks controls how many Monarch matrices are composed per channel.
// ============================================================================

class MonarchSequenceMix : public Layer {
public:
    MonarchSequenceMix(size_t seq_len, size_t d_model,
                       size_t block_size, size_t num_blocks = 1,
                       uint32_t perm_seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchSequenceMix"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t n() const { return seq_len_; }

private:
    size_t seq_len_, d_model_, block_size_, num_blocks_;
    // Per-channel block_values: stacked as (d_model * num_blocks * num_mb * b * b)
    // where num_mb = seq_len / block_size² (blocks per Monarch per channel)
    Tensor blocks_;       // (d_model * num_blocks * num_mb, b, b) flattened
    Tensor grad_blocks_;  // same shape
    std::vector<std::vector<size_t>> perms_P_;  // [d_model][num_blocks][seq_len]
    std::vector<std::vector<size_t>> perms_Q_;  // [d_model][num_blocks][seq_len]
    size_t m_;            // sqrt(seq_len)
    size_t num_mb_;       // seq_len / block_size²
    Tensor last_input_;   // (B, seq_len * d_model)  flattened channel-major
};

// ============================================================================
// MonarchChannelMix — applies a stack of MonarchMatrix(d_model, b_ch) over the
// channel axis. Weights are SHARED across all sequence positions (paper §3):
// each token gets the same Monarch matrix applied to its (d_model,) vector.
// ============================================================================

class MonarchChannelMix : public Layer {
public:
    MonarchChannelMix(size_t seq_len, size_t d_model, size_t block_size,
                      uint32_t perm_seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchChannelMix"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }
    size_t block_size() const { return block_size_; }
    size_t n() const { return d_model_; }

private:
    size_t seq_len_, d_model_, block_size_;
    Tensor blocks_;       // (num_mb, b, b)
    Tensor grad_blocks_;
    std::vector<size_t> perm_P_;  // (d_model,)
    std::vector<size_t> perm_Q_;  // (d_model,)
    size_t m_;            // sqrt(d_model)
    size_t num_mb_;       // d_model / block_size²
    Tensor last_input_;   // (B, seq_len * d_model)
};

// ============================================================================
// MonarchMixerBlock — pre-LN + MonarchSequenceMix + residual + pre-LN +
// MonarchChannelMix + residual. The "Monarch Transformer block" replacement.
// ============================================================================

class MonarchMixerBlock : public Layer {
public:
    MonarchMixerBlock(size_t seq_len, size_t d_model,
                      size_t block_size_seq, size_t block_size_ch,
                      size_t ffn_mult = 0,
                      uint32_t perm_seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMixerBlock"; }

    size_t seq_len() const { return seq_len_; }
    size_t d_model() const { return d_model_; }

private:
    size_t seq_len_, d_model_;
    LayerNorm ln1_;
    MonarchSequenceMix seq_mix_;
    LayerNorm ln2_;
    MonarchChannelMix ch_mix_;
};

// ============================================================================
// MonarchMixerModel — embed + stack of MonarchMixerBlocks + final LN + classifier.
// ============================================================================

class MonarchMixerModel : public Layer {
public:
    MonarchMixerModel(size_t input_dim, size_t d_model, size_t output_dim,
                      size_t seq_len, size_t num_blocks = 2,
                      size_t block_size_seq = 0,  // 0 → sqrt(seq_len) (auto)
                      size_t block_size_ch = 0,  // 0 → sqrt(d_model) (auto)
                      uint32_t perm_seed = 42);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "MonarchMixerModel"; }

    size_t input_dim() const { return input_dim_; }
    size_t d_model() const { return d_model_; }
    size_t output_dim() const { return output_dim_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_blocks() const { return num_blocks_; }

private:
    size_t input_dim_, d_model_, output_dim_, seq_len_, num_blocks_;
    size_t block_size_seq_, block_size_ch_;
    Dense embed_;
    std::vector<std::unique_ptr<MonarchMixerBlock>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;
};

#endif // MONARCH_MIXER_H
```

### Task 0.2: Create stub `include/nn/layers/architectures/monarch_mixer.cpp`

Bare-bones stubs that throw "not implemented" — verified to compile by `g++ -c`.

### Task 0.3: Create umbrella registration

Add `#include "layers/architectures/monarch_mixer.h"` to `include/nn/nn.h` (architectures section, near `mixture_of_depths`).

### Task 0.4: Makefile entries

Append the standard test build rule for `test_monarch_mixer`, add to `tests:` deps, add to `run_tests` echo line.

## Phase 1 — MonarchMatrix primitive

### Task 1.1: TDD — constructor validation

Test cases:
- `MonarchMatrix(16, 2)` valid: n=16=4², b=2, m=4, num_blocks=4
- `MonarchMatrix(16, 4)` valid: num_blocks=1
- `MonarchMatrix(16, 1)` valid: num_blocks=16
- `MonarchMatrix(0, 1)` throws
- `MonarchMatrix(16, 0)` throws
- `MonarchMatrix(15, 1)` throws (not a perfect square)
- `MonarchMatrix(16, 3)` throws (b doesn't divide sqrt(n)=4)
- `MonarchMatrix(16, 5)` throws

Test name and n/b/num_blocks accessors. Test that two different perm_seeds produce different P/Q. Test that same perm_seed is bit-exact deterministic. Test blocks_ is init small (||block|| ≈ sqrt(num_blocks · b²) · 0.1 = sqrt(16) · 0.1 ≈ 0.4).

### Task 1.2: TDD — forward shape

`forward((B=3, n=16)) → (B=3, n=16)`, finite, nonzero for non-trivial blocks. Hand-derived reference with `b=1, m=4, n=16`: M = P · B · Q where B is diagonal. For `b=1`, each block is a 1×1 scalar, so B is just a diagonal of `n` values `b_0, ..., b_{n-1}`. Verify the forward matches the manual computation:
`y[t, P[i]] = b_i · x[t, Q[i]]` for each i.
(This is the hand-derived n=16, b=1 reference test.)

### Task 1.3: TDD — identity-Monarch test: `M = I`

Set `block_values_ = I` (identity blocks) and `P = Q = identity`. Then forward = input bit-exact (max_diff < 1e-12).

### Task 1.4: TDD — backward hand-derived reference

Hand-derive the `M = I` case: backward should be the identity for grad_output too. Plus a non-trivial hand-derived case with `b=1, m=2, n=4, P=identity, Q=identity, B=diag(2,3,5,7)`:
- forward: `y[t, i] = b_i · x[t, i]`
- dy/dx = diag(b_i²) (elementwise Jacobian — since output gradient flows back proportional to b_i)
Wait, no — `dL/dx[i] = b_i · dL/dy[i]` (chain rule, B is on x).
- test: with known input/grad_output, verify analytical matches hand-derived exactly.

### Task 1.5: TDD — FD gradient check on MonarchMatrix

Centered FD vs analytical: for a small `(n=4, b=2)` config with random non-uniform init, verify `rel_err < 1e-5` on:
- d_block_values (perturb one entry, verify the analytical block gradient matches)
- d_input (input gradient matches FD)

Test must use random init (NOT identity blocks) so the block asymmetry is exercised — a common vacuity trap is that uniform block init gives identical row and column sums, hiding index bugs.

### Task 1.6: TDD — multiple Monarch compositions (num_blocks > 1)

When `num_blocks > 1`, the effective matrix is `M = M_k · … · M_2 · M_1`. Forward caches intermediate products so backward can walk the chain. Test that for `n=4, b=2, num_blocks=3` with random init, FD vs analytical on the LEFTMOST Monarch's block gradient matches (the rightmost blocks see the gradient propagated through the product chain).

### Task 1.7: TDD — parameter update

`update_weights` should be plain SGD: `blocks -= lr · grad_blocks`. Test: `grad_blocks` moves blocks in the correct direction; verify the change equals the analytical gradient magnitude at lr=1.

### Task 1.8: TDD — training reduces loss

Build a tiny regression task: target `y = x @ W_target` for a fixed random `W_target`. Train for 30 SGD steps; verify loss reduces by > 50%.

## Phase 2 — MonarchSequenceMix

### Task 2.1: TDD — constructor + accessors

`MonarchSequenceMix(seq_len=16, d_model=4, block_size=4, num_blocks=1, perm_seed=42)`.
- `seq_len()=16, d_model()=4, block_size()=4, num_blocks()=1, n()=16`
- For `seq_len=16, block_size=4`: `m = sqrt(16) = 4`, `num_mb = 16 / 4² = 1`. Each channel has 1 Monarch with 1 block of (4, 4).
- Total params: `d_model * num_blocks * num_mb * b² = 4 * 1 * 1 * 16 = 64` (just block values; perms are 4*1*16 = 64 indices on top).

### Task 2.2: TDD — forward shape

Input shape `(B, seq_len * d_model)` (channel-major flatten: row t holds [token_t_token_0_channel, token_t_token_1_channel, ..., token_t_token_{d_model-1}_channel]). Wait — actually let's use the simpler convention: input `(B, seq_len, d_model)` per the rest of the repo's sequence-mixing layers (FNet, GatedMLP, etc.).

Actually for simplicity and consistency with the existing repo (look at how Based, FNet handle the dim), let me use `(B, seq_len * d_model)` flattened channel-major `[t, c] = flat[t * d_model + c]`. This makes per-channel Monarch application a slice on `flat[t*d_model:(t+1)*d_model]` → no wait, that's wrong. Let me think again.

OK, for **per-channel** Monarch over the sequence axis: we want `out[t, c] = sum_s W[c, t, s] · x[s, c]`. So for each channel c, we have a (seq_len, seq_len) Monarch matrix. The per-channel blocks are different. The forward is: for each c, `out[:, c] = M_c · x[:, c]`.

For `(B, seq_len, d_model)` input, the implementation walks `c` from 0 to `d_model-1` and applies the c-th Monarch to column `c`.

### Task 2.3: TDD — identity-Monarch: each channel is identity

Set all blocks to identity, perms = identity. Forward output == input bit-exact (max_diff < 1e-12).

### Task 2.4: TDD — input gradient FD check

Random non-uniform init, FD vs analytical `rel_err < 1e-5` for `d_input` and `d_blocks`.

### Task 2.5: TDD — training reduces loss

Tiny regression: input `(B, seq_len, d_model)`, target `(B, seq_len, d_model)`. Loss = MSE. 30 SGD steps; verify loss reduces.

## Phase 3 — MonarchChannelMix

### Task 3.1: TDD — constructor + forward shape

`MonarchChannelMix(seq_len=16, d_model=16, block_size=4)`. Block diag: 4 blocks of (4, 4). Forward `(B, seq_len, d_model) → (B, seq_len, d_model)`: per token, apply the shared Monarch matrix to the (d_model,) vector.

### Task 3.2: TDD — identity-Monarch

Identity blocks + identity perms → forward == input bit-exact.

### Task 3.3: TDD — FD gradient checks

`d_input` and `d_blocks` FD vs analytical `rel_err < 1e-5`.

## Phase 4 — MonarchMixerBlock

### Task 4.1: TDD — constructor + forward shape

`MonarchMixerBlock(seq_len=16, d_model=16, block_size_seq=4, block_size_ch=4)`.
Forward `(B, seq_len, d_model) → (B, seq_len, d_model)`, finite, nonzero.

### Task 4.2: TDD — residual identity at init

With small init on Monarch blocks AND no LN effect (gamma=1, beta=0), the output should be approximately equal to the input. For LayerNorm, with gamma=1, beta=0, the LN output is centered and normalized — NOT identity. So at init, the output is NOT exactly the input. The test should instead verify:
- `update_weights(0)` (lr=0) followed by `forward(input)` and `backward` and `update_weights(0)` again produces bit-exact forward (no learning).
- The forward IS NOT input-equal at init (because LN normalizes), but the SHAPE is `(B, seq_len, d_model)` and finite/nonzero.

### Task 4.3: TDD — block input gradient FD check

FD `rel_err < 1e-3` on input gradient (looser than layer-level because LN amplifies noise near saturation).

### Task 4.4: TDD — training reduces loss

Tiny regression over 30 SGD steps; verify loss reduces.

## Phase 5 — MonarchMixerModel

### Task 5.1: TDD — constructor + forward shape

`MonarchMixerModel(input_dim=8, d_model=16, output_dim=3, seq_len=16, num_blocks=2)`. Forward `(B, seq_len * input_dim) → (B, output_dim)` (mean-pool over seq, classifier).

### Task 5.2: TDD — input gradient FD check

FD vs analytical on d_input.

### Task 5.3: TDD — training reduces loss

Tiny classification: target is argmax over a fixed random projection of input. 30 SGD steps; verify loss reduces.

## Phase 6 — Integration & registration

### Task 6.1: Register in umbrella header

Add `#include "layers/architectures/monarch_mixer.h"` to `include/nn/nn.h` (architectures section, near `mixture_of_depths`).

### Task 6.2: Verify umbrella header compiles standalone

```bash
g++ -std=c++17 -Iinclude -fsyntax-only -x c++ - <<< '#include "nn/nn.h"'
```

This catches header collisions and missing transitive includes.

### Task 6.3: Run aggregate build

```bash
make tests
```

Verify the full suite still builds. The new test (`test_monarch_mixer`) should appear in the build output.

### Task 6.4: Run the new test

```bash
make build/test_monarch_mixer && ./build/test_monarch_mixer
```

Verify all Monarch Mixer checks pass.

### Task 6.5: Run aggregate `make run_tests`

Verify no regressions in other test suites (especially the SSM, attention, and normalization tests, which share the `Layer` interface).

### Task 6.6: Document in EXPANSION_QUEUE.md

Move from `## Ideas` to `## Done` with a one-line summary of what was built and test counts.

### Task 6.7: Commit

```bash
git add include/nn/layers/architectures/monarch_mixer.{h,cpp} \
        tests/test_monarch_mixer.cpp \
        docs/plans/2026-09-19-monarch-mixer.md \
        include/nn/nn.h \
        EXPANSION_QUEUE.md \
        Makefile
git commit -m "feat(architectures): Monarch Mixer — block-diagonal × permutation structured matrices"
git push origin master
```

## Verification checklist

- [ ] All Monarch-specific focused tests pass (target: ≥30 checks)
- [ ] No regressions in aggregate `make run_tests`
- [ ] Plan saved at `docs/plans/2026-09-19-monarch-mixer.md`
- [ ] Umbrella header compiles standalone
- [ ] Commit pushed to `master`
- [ ] EXPANSION_QUEUE.md updated (move entry from Ideas → Done)