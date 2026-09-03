# FT-Transformer Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement FT-Transformer (Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data") — a pure-transformer architecture for tabular data with per-feature numerical embeddings and CLS-token aggregation.

**Architecture:** Each numerical feature becomes its own learnable linear embedding (`t_i = W_i * x_i + b_i`); optional categorical features are one-hot/dense-vector embeddings. The stack of feature tokens is processed by N pre-norm transformer blocks (using the existing `GQABlock` — pre-LN → GQA attention → residual → pre-LN → GELU FFN → residual). A CLS token is prepended and its final hidden state is fed to a classifier MLP.

**Tech Stack:** Existing C++ ML infra (Tensor, Layer, LayerNorm, GQABlock, Dense).

---

## Architecture overview

```
       ┌── numerical input: (B, n_num_features) ──► NumericalFeatureTokenizer ──► (B, n_num, d_model)
x ─────┤
       └── categorical input: (B, n_cat_features) one-hot or sparse ─► CategoricalFeatureTokenizer ──► (B, n_cat, d_model)
                                                                                                            │
concat + prepend CLS token                                                                                    ▼
tokens = [CLS; num_emb; cat_emb] in R^{(B, 1 + n_num + n_cat, d_model)}                                       │
                                                                                                            ▼
                                                                                       stack of N × GQABlock
                                                                                                            │
                                                                                                            ▼
                                                                                                  CLS hidden (B, d_model)
                                                                                                            │
                                                                                                            ▼
                                                                                       classifier MLP → logits (B, n_classes)
```

**NumericalFeatureTokenizer**: per-feature `Linear(1 → d_model)` with bias. Each feature i has its own `W_i ∈ R^{d_model × 1}` and `b_i ∈ R^{1 × d_model}`. Implementation: weight matrix `W ∈ R^{d_model × n_num}` (column j = W_j), bias `b ∈ R^{1 × d_model}`. Forward: `tokens = x @ W^T + b` (broadcasting).

**CategoricalFeatureTokenizer**: per-feature embedding lookup. Each feature i has `E_i ∈ R^{d_model × vocab_i}` (column = embedding). For simplicity in v1: each categorical feature is already pre-tokenized into a one-hot row, and the embedding is a per-feature linear `Linear(vocab_i → d_model)`. (The cleaner formulation: feature i's lookup is `E_i[one_hot]`.)

**Stack of GQABlocks**: each block applies attention over all feature tokens (no causal mask — features are permutation-equivariant; the paper relies on the CLS token to aggregate).

**CLS token**: learnable parameter `cls_token ∈ R^{1 × d_model}` prepended to the sequence at every forward.

**Final LayerNorm → classifier MLP → logits**.

**Backward**: All components are differentiable; GQABlock handles its own backward. The custom backward path:
- numerical: dW = x^T @ dL/dtokens ; db = sum_B dL/dtokens ; d_input_x = dL/dtokens @ W
- categorical: dE_i = dL/dtokens_i^T @ one_hot_input_i ; d_input_x = dL/dtokens_i @ E_i
- CLS: d_cls += sum_B dL/dCLStokens[0]

---

## Task granularity

The work breaks into bite-sized tasks:

1. Write failing tests for `NumericalFeatureTokenizer` (constructor, forward shape, finiteness, input gradient, weight gradient, bias gradient, determinism).
2. Implement `NumericalFeatureTokenizer`.
4. Write failing tests for `CategoricalFeatureTokenizer` (constructor, forward shape, finiteness, input gradient, embedding gradient, determinism).
5. Implement `CategoricalFeatureTokenizer`.
6. Write failing tests for `FTBlock` (forward shape, input grad FD vs analytical).
7. Implement `FTBlock` as wrapper around GQABlock with an explicit CLS-token prepend and CLS-pool.
8. Write failing tests for `FTTransformer` (forward shape, output shape, training reduces loss).
9. Implement `FTTransformer` end-to-end.
10. Umbrella registration, Makefile wiring, run full suite.

Each task = ~5 min of focused work with the `test-driven-development` cycle.