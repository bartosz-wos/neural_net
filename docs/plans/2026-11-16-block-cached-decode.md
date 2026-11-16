# Block-Level Cached Decode Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship an inference-time **block-level** cached decode — `TransformerBlock::forward_cached` — whose output equals the ordinary full-sequence `TransformerBlock::forward` **token for token, to the last bit**, and make the positional encoding absolute so a decode does not silently mis-position its tokens.

**Architecture:** Two pieces.
1. `TransformerBlock::forward_cached` (`include/nn/layers/attention/transformer.{h,cpp}`) — the consumer. It delegates the attention to the **already shipped** `MultiHeadAttention::forward_cached` and computes the block's post-attention tail on the new tokens only. The tail arithmetic is factored into a **shared private helper** so the ordinary and the cached path cannot drift apart.
2. `PositionalEncoding::forward_at(input, offset)` (`include/nn/layers/attention/transformer.{h,cpp}`) — the positional encoding at **absolute** positions. `forward()` becomes `forward_at(input, 0)`.

**Tech Stack:** C++17 (repo's `-std=c++17 -O2 -Wall -Wextra -march=native`), float64 numpy oracle, bash mutation harnesses, ASan+UBSan.

---

## THE GAP, MEASURED (2026-11-16)

The `## Ideas` list was drained a twenty-fourth time, so this session fell back to the queue's own documented method: a verified-absent concept sweep. The last `## Done` entry named the item itself:

> **A RESIDUAL-STREAM CACHE, for a BLOCK-LEVEL cached decode.**
> `grep -rinE 'residual.?stream|hidden.?state.?cache|x.?cache' include/` -> the attention cache exists now (`utils/kv_cache.h`), but nothing caches the block's RESIDUAL, which carries the prefix just as the attention does. `TransformerBlock::forward_cached` therefore needs a SECOND cache … **A block-level decode is NOT shipped**, and shipping a half-working one would be worse than shipping none.

Re-measured, unchanged:

```
grep -rinE 'residual.?stream|hidden.?state.?cache|x.?cache' include/
  -> 3 hits, ALL COMMENTS (griffin.h:16, xlstm_block.h:35, ngpt.h:18)
  -> ZERO implementation.
```

## THE QUEUE'S PREMISE WAS **REFUTED BY MEASUREMENT** — AND THAT IS THE FINDING

The queue asserts the block needs a SECOND cache because "the FFN's position-wise nature does not remove the requirement". **That is false in this repo, and the reason is a repo-specific convention worth recording.**

`include/nn/layers/normalization/layer_norm.cpp` normalises over the **feature** axis of each ROW:

```cpp
size_t batch = input.rows;      // tokens
size_t features = input.cols;   // d_model
for (size_t b = 0; b < batch; ++b) {
    double mean = 0.0;
    for (size_t f = 0; f < features; ++f) mean += input[b][f];
    ...
}
```

So a `LayerNorm` over a `(tokens, d_model)` tensor is **per token**, not per sequence. The block's tail — `residual → ln1 → FFN → + → ln2` — is therefore position-wise, and the ONLY cross-token coupling in the whole block is the **attention**, which the shipped KV cache already covers exactly.

Measured (float64 numpy, `d_model=4`, `T=6`):

| tail computed on | vs. the full sequence's rows |
|---|---|
| the FULL sequence | — |
| the LAST TWO TOKENS ONLY, feature-axis LN | `max_abs_diff = 0.0` (**exactly zero**) |
| the LAST TWO TOKENS ONLY, **token-axis** LN (counterfactual) | `max_abs_diff = 2.170093293154385` |

The counterfactual matters: it proves the zero is a property of **this repo's LayerNorm convention**, not a tautology. A repository whose LayerNorm normalised over the sequence axis would genuinely need a residual-stream cache. **This one does not.** Shipping a second cache would be a second source of truth that must be kept in sync with the first, for zero behavioural difference — so it is NOT shipped.

**THE CONSEQUENCE: the feature is the BLOCK-LEVEL DECODE itself, and the queue's residual-stream cache is recorded as REFUTED.**

## THE SECOND, INDEPENDENT GAP: THE POSITIONAL ENCODING IS **LOCAL**, NOT ABSOLUTE

`PositionalEncoding::forward` uses the tensor's **column index** as the position:

```cpp
for (size_t s = 0; s < seq_len && s < max_len; ++s)
    output[f][s] = input[f][s] + pe[s][f];      // `s` is the LOCAL index
```

During a decode the caller hands it **only the new tokens**, so column 0 is the new token — and it receives `pe[0]` while its absolute position is `past`. Measured: `pe[0][0] = 0.0` against `pe[3][0] = 0.1411200080598672`. A decode built on the shipped block would therefore be **silently mis-positioned**: wrong output, no error. This is the "shipping a half-working one would be worse than shipping none" failure the queue warns about, and it must be fixed for the block-level decode to mean anything.

Two further latent defects in the same six lines, both fixed by the same change:

- **`input.rows` is never checked against `d_model`.** `pe[s][f]` reads `pe`'s columns with `f` running to `input.rows-1`; `pe` has `d_model` columns. A caller with more rows than `d_model` reads **out of bounds**.
- **Positions past `max_len` are silently zeroed.** `Tensor output(rows, cols)` value-initialises to `0.0`, and the loop stops at `max_len`, so the tail of an over-long sequence is replaced by zeros — the token's content is destroyed with no error.

Both become explicit, distinct errors.

## THE INVARIANT

For a causal block, decoding token by token with a cache must equal the full-sequence forward **token for token**. It is pinned **exactly** (`max_abs_diff == 0.0`), not within a tolerance: the cached path performs the same arithmetic in the same order, and the tail is row-wise so it produces bit-identical values. A tolerance would hide a decode that attends to the wrong tokens.

## THE NON-VACUITY REQUIREMENT

A cached path that silently ignores the cache produces the right answer at step 0 and the wrong one everywhere after. The suite therefore also proves the cache is **read** (perturbing token 0 must change a later step) and that a deliberately broken incremental path **must differ** from the full forward.

---

### Task 1: The float64 oracle — `tests/block_cached_decode_oracle.py`

**Objective:** Specify the block-level cached decode in numpy, independently of the C++, and prove the equivalence before any C++ exists.

**Files:**
- Create: `tests/block_cached_decode_oracle.py`

**Step 1: Write the references**

- `ref_layernorm_rows(x, gamma, beta, eps)` — the repo's **feature-axis** LayerNorm, including its `var = max(var, 1e-7)` clamp.
- `ref_mha_full(...)` — the repo's causal MHA, in the repo's `(d_model, seq_len)` layout and the repo's projection convention (`X @ Wq`, NOT `X @ Wq.T`).
- `ref_block_full(x, P, heads)` — the whole block: attention → `x + attn` → ln1 → FFN → `+` → ln2.
- `ref_block_cached(x, P, heads, chunk)` — the same, with the attention going through an explicit `(T, d_model)` K/V store and the tail computed on the chunk only. Written the **other way round** from `ref_block_full` so a shared bug is unlikely to survive both.

**Step 2: Sections**

1. **equivalence** — cached == full over several `(d_model, seq_len, heads, chunk)` combinations;
2. **the tail is position-wise** — the measurement that refutes the residual-stream cache, with the token-axis counterfactual that proves the check has teeth;
3. **non-vacuity** — an ignored cache MUST differ; perturbing token 0 MUST change a later step;
4. **the positional encoding is absolute** — `pe[offset + s]`, not `pe[s]`;
5. **pinned fixtures** emitted as C++ literals (`--emit-cpp`).

**Step 3: Run**

Run: `/usr/bin/python3 tests/block_cached_decode_oracle.py`
Expected: `=== SUMMARY: N passed, 0 failed ===`, exit 0.

---

### Task 2: The oracle's own mutation harness — `tests/block_cached_decode_mutate_oracle.py`

**Objective:** MEASURE the oracle's discriminating power. An oracle that cannot fail certifies nothing.

**Files:**
- Create: `tests/block_cached_decode_mutate_oracle.py`

Break the oracle's model code in specific ways (the layer-wise normalisation axis, the tail's row/column mix-up, the K/V store's append offset, the causal mask's boundary, the FFN's weight order, the PE's offset, the equivalence's own comparison) and require that the oracle FAIL each time.

**Run:** `/usr/bin/python3 tests/block_cached_decode_mutate_oracle.py`
Expected: `N caught / 0 survived / 0 broken / 0 noop`.

---

### Task 3: The C++ implementation

**Objective:** `TransformerBlock::forward_cached` and `PositionalEncoding::forward_at`.

**Files:**
- Modify: `include/nn/layers/attention/transformer.h`
- Modify: `include/nn/layers/attention/transformer.cpp`

**Step 1: Write the failing test** — see Task 4 (the test is written before the implementation exists).

**Step 2: Factor the block's tail into a shared helper**

Both `forward` and `forward_cached` call `compute_block_tail(x_tokens, attn_tokens, &ffn_pregelu, &out_tokens)`. The loop bodies move **verbatim** so the ordinary path's arithmetic is bit-for-bit what it was.

**Step 3: Implement `forward_cached`**

- throws if no cache is attached (distinct message);
- throws if `input.rows != d_model` (distinct message);
- throws if the input holds no tokens (distinct message);
- delegates the attention to `attn.forward_cached(input)`;
- computes the tail on the new tokens only;
- records `last_forward_was_cached_` so `backward()` throws **early**, before it can corrupt any gradient buffer.

**Step 4: Implement `PositionalEncoding::forward_at`**

- `forward(input)` becomes `forward_at(input, 0)`;
- the position is `offset + s`;
- throws if `input.rows != d_model` (the out-of-bounds read);
- throws if `offset + input.cols > max_len` (the silent zeroing).

**Step 5: Build**

Run: `make -j$(nproc) build/layers/attention/transformer.o`
Expected: compiles clean, no warnings.

---

### Task 4: The C++ suite — `tests/test_block_cached_decode.cpp`

**Objective:** Pin the invariant in the language that ships.

**Files:**
- Create: `tests/test_block_cached_decode.cpp`

The suite is **exception-safe** (an escaping exception is reported as a FAILURE, not an abort) and every throwing assertion goes through an `expect_throw(fn, needle, name)` helper that calls, captures, and **then** reports — C++ evaluates function-call arguments in unspecified order, so a message read before the call can print the wrong call's message.

Checks:

1. the cached decode == the full forward, **exactly** (`0.0`), for chunk sizes 1..6;
2. both == the pinned numpy output to `1e-12`;
3. a whole prompt handed in one chunk == the full forward (the prefill case);
4. two blocks sharing ONE cache at different layer indices;
5. non-vacuity: perturbing token 0 changes a later step;
6. the error messages, each **distinct** and asserted by message;
7. `backward()` after a cached forward throws, and an ordinary forward re-enables it;
8. `forward_at` == `forward` at offset 0, and uses `pe[offset + s]` beyond it;
9. the PE's two latent defects now throw instead of corrupting memory.

**Run:** `make build/test_block_cached_decode && ./build/test_block_cached_decode`
Expected: `N passed, 0 failed`.

---

### Task 5: The C++ mutation harness — `tests/mutate_block_cached_decode.sh`

**Objective:** prove the suite CATCHES the defects it was written for.

The harness obeys the nine rules this repo has already learned: `restore()` never deletes its backup and cleanup happens once on EXIT; the link set comes from `make print-lib-objs`; the mutated object is compiled separately and swapped in; `MUT-BROKEN` and `rc >= 128` are **separate verdict classes**; the intermediates are removed before every build; never run a build while a harness runs; the anchor must be **unique**; the edits are made with **Python**, not perl/sed.

**Run:** `bash tests/mutate_block_cached_decode.sh`
Expected: `N caught / 0 survived / 0 broken / 0 crashed / 0 noop / 0 anchor`.

---

### Task 6: The build wiring

**Files:**
- Modify: `Makefile`

- add the `$(BUILD_DIR)/test_block_cached_decode` link rule;
- add it to the `tests:` prerequisite list;
- add its line to `run_tests`.

---

### Task 7: The verification gate

Run: `make -j$(nproc) tests && make run_tests`
Expected: **zero new failures**. The baseline on this commit is exactly one failing suite — `test_neural_ode`, the long-standing entry in `NOT_FIXED.md` (`ODERNN dense1.W grad vs FD`, line 561). It must be re-measured **identically**.

Run: `make BUILD_DIR=build_asan CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer" build/test_block_cached_decode && ./build_asan/test_block_cached_decode`
Expected: **CLEAN** under ASan+UBSan.

---

### Task 8: The record

**Files:**
- Modify: `EXPANSION_QUEUE.md` — move the item to `## Done` with the measurement, the refutation, and the results.
- Modify: `NOT_FIXED.md` — only if a new defect is found and deferred.

Commit and push to `master` without asking for review.
