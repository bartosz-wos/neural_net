# MultiHeadHyena Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement `MultiHeadHyena` (MultiHyena) — the multi-head long-convolution operator described by Massaroli et al., "Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions", arXiv:2310.18780 §4 — as the layer MAD (arXiv:2403.17844) evaluates as its "Multi-Head Hyena" baseline (its ref [24]).

**Architecture:** Three classes in `include/nn/layers/architectures/multi_head_hyena.{h,cpp}`: `MultiHeadHyenaOperator` (the head-split outer-product operator), `MultiHeadHyenaBlock` (pre-RMSNorm → operator → residual → pre-RMSNorm → SwiGLU → residual), `MultiHeadHyenaModel` (input proj → stack → final RMSNorm → mean-pool → classifier). The operator's filter generator reuses the already-shipped `HyenaFilter`; the head structure, outer-product featurization and the per-head convolution are new.

**Tech Stack:** C++17, hand-rolled `Tensor` (2-D real, row-major), `Layer` interface, `Dense`, `RMSNorm`, `SwiGLU<>`, naive O(L²) causal convolution (repo-wide convention for gradient tractability).

---

## Source of truth (verified, not from memory)

Matharoli/Massaroli et al., *Laughing Hyena Distillery* (arXiv:2310.18780v1) §4, read from the arXiv HTML full text:

> 1. "Given the projections `q,k,v ∈ R^{L×D}`, we split them into `M` chunks of size `N = D/M`, `q^m, k^m, v^m ∈ R^{L×N}`."
> 2. "Each chunk is processed by a modified Hyena operator: first, we perform the outer product of `k^m` and `v^m` along the spatial dimension, `z^m ≜ k^m ⊗ v^m ∈ R^{L×N×N}`, apply a long convolution with filter `h^m` to **all `N×N` elements independently**, then compute `y^m_t = (h^m * z^m)_t q^m_t`, `y^m ∈ R^{L×N}` as shown in Figure 4.1."
> 3. "Finally, we compose `y^1, …, y^m` into a single output `y ∈ R^{L×D}` via concatenation."

MAD arXiv:2403.17844 (its "Multi-Head Hyena" baseline, ref [24]) confirms the hyperparameters: **heads = 16, state dimension of heads = 2, filter order = 2, short filter order = 3** (App. B.3.2 / C.2), with head dimension swept to **8**.

### The single most important detail: the filter is SHARED across the N×N outer-product plane

"`z^m ∈ R^{L×N×N}`… apply a long convolution with filter `h^m` to **all `N×N` elements independently**."

`h^m` is **one** filter applied to all `N²` channels. This is precisely the point of the layer — §4 is motivated by the finding that "at initialization, filters correspond to high-dimensional SSMs, and gradually converge to lower-dimensional representations during training", so the same low-rank structure is tied across the outer-product plane. It is also what makes `filter order: 2` sensible at `N = 8` (a per-element filter would need 64 independent filter channels). **A per-`(i,j)` filter is the natural mistake and the tests must pin it.**

### Second detail: the paper writes `y^m_t = (h^m * z^m)_t q^m_t` — element-wise, not inner-product

`q^m_t ∈ R^N` and `(h^m * z^m)_t ∈ R^{N×N}`; the product is the **Hadamard** product, reducing `R^{N×N} → R^N`. This is a dimension-`N` output per head (concatenating `M` heads gives `M·N = D`). Reading it as an inner product would give `R^{1×N}`-shaped… the same thing numerically, but reading it as a **matmul over N** would produce a scalar per head and break the concat. Element-wise is also the only reading consistent with `y^m ∈ R^{L×N}`.

### v1 scope decisions

- **Filter generator: reuse `HyenaFilter`.** It already implements the canonical filter parametrization (positional-embedding MLP + exponential modulation) that MAD App. B.3.2 attributes to this same ref [24]. Writing a second, different filter generator would duplicate a proven chain and import its bugs. A **separate implementation is not** warranted here (contrast with StripedHyena/HyenaDNA, where the generator itself differs).
- **Short depthwise conv (order 3): omitted.** MAD lists "short filter order: 3" as a shared hyperparameter of both Hyena and Multi-Head Hyena, but §4's `MultiHyena` definition has **no** short conv stage — it is applied directly to the `q,k,v` projections. The shipped `HyenaOperator` already covers the short-conv variant. Adding one here would import a stage the defining section does not have.
- **No exponential-decay/gating stage beyond the filter's own modulation.** §4 defines exactly one long convolution per head.
- **Single-head degenerate case (`M = 1`) is tested** as the recovery identity.

## Conventions

Sequences are `(L, d_model)` row-major `Tensor`s (repo-wide). Inside the operator, the `L×N×N` outer-product plane is flattened to `(L, N*N)` and the filter to `(L, N*N)` so everything is a 2-D `Tensor`; index `c = i*N + j` means outer-product element `(i,j)`. Backward is hand-derived; the outer-product Jacobian is

```
z[l][i*N+j] = k[l][i] * v[l][j]
dz/dk[l][i] = Σ_j dz[l][i*N+j] * v[l][j]
dz/dv[l][j] = Σ_i dz[l][i*N+j] * k[l][i]
```

and the conv backward is the same `causal_conv_1d_backward_v/_k` pair `hyena.cpp` already uses. The Hadamard step backprops `dq[l][i] = Σ_{i,j} dy[l][i] * dz_conv[l][i*N+j]`… precisely: with `y[l][i] = Σ_j (h*z)[l][i*N+j] · q[l][i]`, so `dy[l][i]` multiplies `q[l][i]` in **every** `j` slot:

```
dy[l][i] = q[l][i] * Σ_j dW[l][i*N+j]          (dW = grad of the conv output)
dq[l][i] = Σ_j dW[l][i*N+j]
```

## Tasks

### Task 1: Write the RED test file with the hand-derived reference

Create `tests/test_multi_head_hyena.cpp` with an **independent** hand-rolled reference implementation of the §4 three-step definition (written straight from the quoted text, not calling the impl) and the harness (`check`, `max_abs_diff`, `max_all_finite`, `random_tensor`) matching `tests/test_striped_hyena.cpp`.

Critical: the reference must pin (a) the shared-`h^m`-across-`N×N` contract, (b) Hadamard-not-matmul, (c) causality, (d) `M=1` recovery. Without (a) the "per-element filter" bug passes.

**Step 1:** write `tests/test_multi_head_hyena.cpp` — constructor validation, accessors, forward shape/finite/non-zero, hand-derived reference match, shared-filter contract, Hadamard-vs-matmul discriminator, determinism, parameter/gradient contract, FD gradient checks (input + every param group), zero_grad/update_weights, block forward + FD, model forward + training-reduces-loss.

**Step 2:** verify RED — `make build/test_multi_head_hyena` fails to compile (class does not exist yet). That is the expected failure.

### Task 2: Implement `multi_head_hyena.h`

Three classes, public config with trailing-underscore members + accessors (repo convention), `std::unique_ptr` for sub-layers per `striped_hyena.h`'s documented double-ownership rule.

### Task 3: Implement the operator forward — minimal, to pass forward tests

`in_proj: Dense(d_model, 3*d_model)` → split into `q,k,v ∈ (L, D)` → per head `m`: outer product `z (L, N*N)` → causal conv with shared `h^m (L, N*N)` → Hadamard with `q^m` → concat heads → `out_proj: Dense(d_model, d_model)`.

Filter reuse: one `HyenaFilter` per head is *not* what the paper does (it says filter `h^m` per head — that IS per-head). Per-head `HyenaFilter(d_model=N, l_max, filter_order)`. Run `filter(L)` once per head per forward, and route `grad_h` back into it in backward.

### Task 4: Implement the operator backward

`out_proj` → concat-split of `dy` into per-head `dy^m` → Hadamard backward → conv backward (`_backward_v` and `_backward_k`) → outer-product backward → concat-split `dq/dk/dv` → `in_proj` backward → per-head `HyenaFilter::backward(grad_h^m, 0)`.

### Task 5: Block + Model

`MultiHeadHyenaBlock`: `u = x + op(RMSNorm(x))`, `y = u + SwiGLU_down(RMSNorm(u))` (down-projection required — SwiGLU has no output projection, same reason as `MegalodonBlock`).
`MultiHeadHyenaModel`: `in_proj` → blocks → final `RMSNorm` → mean-pool → `classifier`.

### Task 6: Register in `nn.h` + `Makefile`

- `nn.h`: `#include "layers/architectures/multi_head_hyena.h"` after the `hyena_dna.h` line.
- `Makefile`: `$(BUILD_DIR)/test_multi_head_hyena: $(LIB_OBJS) $(BUILD_DIR)/test_multi_head_hyena.o` link rule; add to the `tests:` prerequisite list; add the `@echo "=== Running test_multi_head_hyena ==="; …` `run_tests` line. (`LIB_SRCS` is a wildcard — the new `.cpp` auto-builds, no explicit source registration needed.)

### Task 7: Verify

```bash
make build/test_multi_head_hyena && ./build/test_multi_head_hyena   # all checks pass
make build/test_multi_head_hyena 2>&1 | grep -i warning             # clean under -Wall -Wextra
# rerun 3x to confirm determinism of the suite itself
# mutation test: break the shared-filter contract, confirm tests fail
```

### Task 8: Mutation verification (required before commit)

Three mutations, each must be caught:
1. **Per-element filter** — give each of the `N×N` outer-product channels its own filter row (i.e. ignore the shared `h^m`) → the shared-filter contract test must fail.
2. **Drop the Hadamard** — remove the `·q^m_t` multiply → the reference-match and FD tests must fail.
3. **Drop the `out_proj`** → forward/FD tests must fail.

### Task 9: Commit

```bash
git add docs/plans/2026-10-06-multi-head-hyena.md
git commit -m "docs: add MultiHeadHyena implementation plan (outer-product featurization + shared-filter conv)"
# … implement, then:
git commit -m "feat(architectures): MultiHeadHyena — outer-product long conv with head-shared filter (N/N, M mutations caught)"
git push origin master
```

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Reuse shipped `HyenaFilter` for `h^m` | MAD App. B.3.2 attributes the filter featurization to this same ref [24]; a second generator would duplicate a proven chain. |
| D2 | **One** filter shared across all `N×N` outer-product channels | Verbatim: "apply a long convolution with filter `h^m` to all `N×N` elements independently". This *is* the layer's thesis. |
| D3 | Hadamard `·q^m_t`, not matmul | `y^m ∈ R^{L×N}` per the paper; matmul would give a scalar per head. |
| D4 | No short depthwise conv | §4's definition has none; the shipped `HyenaOperator` covers that variant. |
| D5 | Naive O(L²) conv | Repo-wide convention (`hyena.h` documents this explicitly) — keeps FD tractable at small L. |