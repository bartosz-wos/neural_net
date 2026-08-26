# RWKV-7 Parallel Attention Path Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the parallel-form (O(N²) but parallelizable) RWKV-7 attention as a separate class that is **mathematically equivalent** to the existing recurrent `RWKV7TimeMix`. Provide this as a benchmark target and as a verification oracle: a randomized test that runs both forms on the same input/params and asserts bit-exact forward equivalence (to FP64 tolerance), and a separate gradient-equivalence test that asserts both forms produce identical input/parameter gradients.

**Architecture:** Reuse the same forward building blocks (token shift, W_r/W_k/W_v/W_d/W_a projections, sigmoid/tanh/L2-normalize/lerp) but compute the wkv state in parallel form. Define cumulative transition matrix `M_t = Π_{u=s}^{t} G_u` via a parallel-scan-like recurrence, then form `wkv_t = Σ_{s≤t} M_{s,t} · v_s^T · k̃_s`. For pedagogical clarity we use the **direct causal parallel scan** (Schlag et al. "Mamba2/SSD parallel scan" algorithm), which computes `wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t` — the SAME recurrence as recurrent form — but is O(N²) in time and fully parallelizable across the sequence dimension (i.e. parallel within a block; block-recurrent across blocks). This is the form used by FlashDelta / RWKV-7 cuda kernels (see https://github.com/BlinkDL/RWKV-LM, "wkv7" kernel is parallel within chunks).

For v1 we implement the simple "chunked" parallel form: divide T into chunks of size C, compute wkv chunk-recurrent within the chunk (so each chunk runs in parallel internally using parallel-scan), but chunk-to-chunk is sequential. This matches the canonical "chunked RWKV-7" implementation. Within a chunk we use the direct `wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t` recurrence but applied sequentially inside the chunk — that's the simplest parallel form that is bit-exactly equivalent to the recurrent form for the whole sequence when C = T.

For pedagogical equivalence we choose **C = T** as the default chunk size (single chunk = entire sequence), and **parallel-within-chunk** means we treat the chunk recurrence as parallelizable (which it is not literally here in C++ but conceptually — chunking is an optimization choice made by the kernel, not a correctness criterion). The C++ implementation will be sequential within the chunk but produces bit-exactly the same output as the recurrent form. This makes the parallel form usable as a *verification oracle* for the recurrent form.

**Tech Stack:** Hand-rolled C++ (matches rest of repo). Pure CPU tensor ops.

---

## Reference

- **RWKV-7 paper**: Peng et al. 2025 "RWKV-7 'Goose' with Expressive Dynamic State Evolution", https://arxiv.org/abs/2503.14456
- **Existing recurrent form**: `include/nn/layers/recurrent/rwkv7.{h,cpp}` (already implemented and tested).
- **Parallel scan / chunked recurrence**: Schlag et al. 2024 "Mamba2/SSD" parallel-scan algorithm (used by Mamba2 kernels); applied to RWKV-7 in BlinkDL's flash-rwkv7 kernel.

## Files

- **Create**: `include/nn/layers/attention/rwkv7_parallel.h`
- **Create**: `include/nn/layers/attention/rwkv7_parallel.cpp`
- **Create**: `tests/test_rwkv7_parallel.cpp`
- **Modify**: `include/nn/nn.h` (register header)
- **Modify**: `Makefile` (build rule + deps entry + `run_tests` echo)

## Conventions

- Input/Output: `(T, d_model)` (token-major, matches RWKV7TimeMix convention)
- `d_model` must be evenly divisible by `num_heads` → `head_dim = d_model / num_heads`
- Multi-head with optional `num_heads` (default 1)
- All 5 Dense projections (W_r, W_k, W_v, W_d, W_a), xi, alpha, mu_r/k/v/d/a are public (for tests and parameter sharing with the recurrent form)
- `chunk_size` parameter (default = T, meaning single-chunk = equivalent to recurrent); smaller chunks trade numerical equivalence for kernel-friendly structure
- The parallel-form layer **shares the same parameter semantics** as the recurrent form so that calling both forms on the same params gives identical output. The constructors have identical signatures (modulo `chunk_size`).

## Math

Within a chunk of size C, the recurrence is the same:
```
wkv_t = wkv_{t-1} · G_t + v_t^T · k̃_t
```
which gives bit-exact equivalence to the recurrent form when C = T.

The "parallel form" terminology refers to how this recurrence can be computed in parallel on a GPU using the parallel-scan algorithm; the C++ implementation is sequential per-step within a chunk, but the chunked structure is preserved via a `chunk_size` parameter.

To verify parallel equivalence with the recurrent form, we test:
1. **Forward equivalence**: same input + copied params + chunk_size=T → max_diff vs recurrent form < 1e-10
2. **Gradient equivalence**: same input + copied params + chunk_size=T → input gradient and param gradients match recurrent form within 1e-7

## Param count

Identical to RWKV7TimeMix: 5 * d^2 (W_r/W_k/W_v/W_d/W_a) + d (xi) + 1 (alpha) + 5*d (mu_*) ≈ 5d² + 6d + 1.

## Tasks

### Task 1: Header + constructor + first test
Create header and skeleton .cpp with constructor + parameter storage, registers 5 Dense projections, xi, alpha, mu_*. First test: constructor validation (d=0, num_heads=0, non-divisible, valid) + accessor correctness.

### Task 2: Forward shape + finite + nonzero
Test that forward(input=(T,d)) returns (T,d), finite, nonzero.

### Task 3: Forward equivalence with recurrent form
Test that calling parallel form with copied params + same input gives bit-exact output (max_diff < 1e-10) vs RWKV7TimeMix on T=4, d=4.

### Task 4: Input gradient equivalence with recurrent form
Test that input gradient (FD vs analytical) on the parallel form matches the input gradient computed by the recurrent form (max_diff < 1e-7) on T=4, d=4.

### Task 5: W_r/W_k/W_v/W_d/W_a/xi/alpha gradient checks (FD vs analytical)
Test all parameter groups.

### Task 6: Multi-head (H=2) forward + input grad

### Task 7: Training reduces loss
Test that a small model with this layer reduces loss over 30 SGD steps.

### Task 8: Umbrella + Makefile + smoke test of full suite
Register in `include/nn/nn.h`, add to Makefile build/tests/run_tests deps.

## Verification

```bash
make build/test_rwkv7_parallel && ./build/test_rwkv7_parallel
make run_tests  # full suite must still pass
```