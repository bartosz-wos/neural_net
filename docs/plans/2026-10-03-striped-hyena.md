# StripedHyena Implementation Plan

> **For Hermes:** Already executed. This plan documents the design decisions
> for StripedHyena.

**Goal:** Add the `StripedHyena` architecture — a *striped* (interleaved)
hybrid that alternates the already-shipped Hyena operator with
multi-head attention along model depth.

**Architecture:** `StripedHyenaOperator` (reuses `HyenaOperator` verbatim),
`StripedHyenaBlock` (a discriminated block that runs either Hyena or MHA as
its sequence mixer, with a pre-norm residual around each), and
`StripedHyenaModel` (input projection → a striping schedule of blocks → final
RMSNorm → mean-pool → classifier).

**Tech Stack:** C++17, mirrors `megalodon.{h,cpp}` and composes existing
`HyenaOperator` / `MultiHeadAttention` / `RMSNorm` / `SwiGLU`.

---

## Paper specification (verified against source)

### Sources

- **Poli, Massaro, Muntoni, Singhal, Beigi, et al. — "Mechanistic Design and
  Scaling of Hybrid Architectures"**, arXiv:2403.17844v2 (the StripedHyena
  architecture, the MAD framework, the striping schedules).
- **Poli, Massaro, Beigi, et al. — "Hyena Hierarchy: Towards Larger
  Convolutional Language Models"**, arXiv:2302.10866v3 (the underlying
  `Hyena_N` operator, already shipped as `hyena.{h,cpp}`).
- **Reference implementation**: `athms/mad-lab`, `mad/model/layers/hyena.py`
  (`HyenaOperator.forward`, lines 222–243) and
  `mad/model/layers/featurization/hyena_filter.py` (`HyenaFilter`).

### Striping (arXiv:2403.17844 §3.3 "Topology", Appendix C.2)

> "We use 3 striping schedule ratios: `1A:1H`, `1A:3H`, `1A:11H`, where
> A=Attention and H=Hyena along model depth. In instances where the number of
> layers is not a multiple of the schedule, the ratio is repeated until the
> target depth is reached."

So the schedule is a **model-depth** concept, not an operator concept: a
StripedHyena model of depth `n_layers` lays out a repeating run of
`1 attention block : hyena_ratio Hyena blocks`. `hyena_ratio ∈ {1, 3, 11}` are
the paper's three swept schedules. The paper's Finding 7 is that a 25%
attention allocation is compute-optimal, which is what the `1:11` schedule
approximates (1/(1+11) ≈ 8%) and the `1:3` schedule brackets (25%).

Layout rule, implemented as `build_stripe_schedule(n_layers, hyena_ratio)`:

```
pattern  = [ATTENTION, H, H, ... H]      (1 + hyena_ratio entries)
schedule = pattern repeated, truncated to n_layers
```

Because the pattern begins with an attention block, a depth that is not a
multiple of `1 + hyena_ratio` still starts with attention — the truncation is
deterministic and the schedule always ends mid-run rather than dropping the
tail entirely.

### What is NOT in scope (deliberate)

The "RePU" short filter that is sometimes attributed to StripedHyena is **not**
part of either paper. Verified: arXiv:2302.10866v3 describes only a learnable
depthwise `Conv1d` short filter (Algorithm 1, line 2: "h is a short
convolution filter"), and the `athms/mad-lab` reference implements exactly that
— `nn.Conv1d(in_channels=total_width, out_channels=total_width,
kernel_size=short_filter_order, groups=total_width,
padding=short_filter_order - 1)` (hyena.py:197–203), which is a causal
depthwise conv with left padding. The existing `HyenaOperator` already
implements this. No RePU layer is added; adding one would encode a formula no
source supports.

Also out of scope for v1: MoE variants (`StripedHyena-MoE`,
`StripedHyena Experts + MoE`), multi-head Hyena heads (`MultiHeadHyena`), and
the `mSISO` state-expansion configurations. Those are separate paper sections
and separate layers.

---

## Design decisions

### Data layout

The repo's `Tensor` is real-only, 2-D, row-major `(rows, cols)`. Sequences are
`(L, d_model)`. This matches the shipped `HyenaOperator` and `MegalodonBlock`,
so `StripedHyenaOperator` is a thin composition and inherits their conventions
untouched.

### Why a discriminated block rather than two block classes

`StripedHyenaBlock` holds a `bool is_attention_` and an
`std::unique_ptr<HyenaOperator>` / `std::unique_ptr<MultiHeadAttention>`.
Two separate block classes would duplicate the pre-norm residual wiring, the
parameter list, and the backward loop for the FFN path — and the striping
schedule needs both kinds to live in one `std::vector`. A single class with a
discriminator keeps `StripedHyenaModel.blocks` a flat vector, which is what
makes the schedule trivially checkable (`schedule[i] == block[i].is_attention`).

**Ownership**: the mixers are `new`ed and held by `unique_ptr`. Passing a
stack-local to a container that takes ownership is the double-free trap called
out in the systematic-debugging skill (5d) and the Spatial Transformer
incident; `unique_ptr` here is the correct Model-owns-it form.

### Block structure

Each block is a pre-norm residual around ONE sequence mixer plus a channel
mixer, following the same shape as the shipped `MegalodonBlock` and the
paper's §B.2 ("blocks combine a sequence mixing layer with a subsequent
channel mixing layer"):

```
u = x + Mixer(RMSNorm(x))          // Mixer = HyenaOperator or MultiHeadAttention
y = u + SwiGLU(RMSNorm(u))         // channel mixer
```

The channel mixer is a `SwiGLU` with inner width `d_model * ffn_mult`. The
paper uses SwiGLU as the channel mixer for every architecture except
Mamba/StripedMamba (Appendix C.2), so v1 is uniform here.

`ffn_mult = 0` skips the channel mixer entirely, which is the minimal
sequence-mixer-only configuration and is what the FD checks use to isolate the
Hyena backward from the SwiGLU backward.

### Parameters

- `in_proj`: `Dense(input_dim, d_model)`
- per block: `RMSNorm` ×2 (gamma `(1, d_model)`), one mixer, one `SwiGLU`
- `final_norm`: `RMSNorm(d_model)`
- `classifier`: `Dense(d_model, output_dim)`

`StripedHyenaModel::striping_ratio()` and `is_attention(i)` expose the
schedule so tests can assert the layout directly rather than inferring it from
output values.

---

## Task list (executed)

1. **RED** — write `tests/test_striped_hyena.cpp` with constructor-validation
   and schedule-layout assertions; confirm it fails to compile (class absent).
2. **GREEN (part 1)** — header + `build_stripe_schedule` + constructors +
   accessors. Re-run: schedule tests pass.
3. **GREEN (part 2)** — forward through both mixer paths + the model.
4. **RED (FD)** — add finite-difference checks for the input gradient and the
   mixer parameters. Confirm they FAIL (proving the tests are not vacuous).
5. **GREEN (FD)** — implement `backward`.
6. **MUTATE** — break the backward in a representative way, confirm the FD
   tests catch it, restore.
7. **Commit**, update `EXPANSION_QUEUE.md` → `## Done`, register in `nn.h`
   and the `Makefile`.

---

## Test plan

Constructor validation (throw cases), schedule layout for all three paper
ratios at several depths, forward shape/finiteness, block kind dispatch,
bit-exact determinism, FD input gradient, FD `in_proj` gradient, FD Hyena
filter gradient, FD attention `W_q` gradient, `zero_grad` / `update_weights`,
end-to-end training reduces loss, and two non-vacuous mutations.

FD uses random non-uniform init (a uniform init makes
`Σ_k W[k,j] == Σ_k W[j,k]` by construction, so a row-vs-column confusion in
the matmul backward would pass vacuously).
