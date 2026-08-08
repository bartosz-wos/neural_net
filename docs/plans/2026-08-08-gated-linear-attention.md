# Gated Linear Attention (GLA) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a `GatedLinearAttention` layer to the library, following Yang et al. 2023/2024 "Gated Linear Attention Transformers with Hardware-Efficient Training" (https://arxiv.org/abs/2312.06635, NeurIPS 2024).

**Architecture:** Per-head single-step recurrence with a (head_dim × head_dim) matrix state. The update rule replaces standard linear attention's `S_t = S_{t-1} + outer(k_t, v_t)` with a gated update `S_t = α_t · S_{t-1} + outer(k_t, v_t)` where α_t ∈ (0, 1) is a per-head, input-dependent scalar gate produced by a small Dense projection followed by sigmoid. Multi-head with standard projections (W_Q, W_K, W_V, W_O, W_gate). Output projection back to d_model.

**Tech Stack:** Existing `Layer`, `Tensor`, `Dense` primitives. Files go in `include/nn/layers/recurrent/`. Tested with numerical gradient checks at machine precision.

---

## Background

**Paper context.** Linear attention (Katharopoulos et al. 2020) replaces softmax attention with the linear-time recurrence:

```
S_t = S_{t-1} + outer(k_t, v_t)
o_t = S_t · q_t
```

This is O(d²) per step and O(T·d²) total, vs softmax attention's O(T²·d). However, plain linear attention has a known issue: it cannot selectively **forget** past information. The state S_t grows unboundedly as more keys/values are added, with no mechanism to decay stale content.

The **Gated Linear Attention** (Yang et al. 2023, "Gated Linear Attention Transformers with Hardware-Efficient Training", arXiv:2312.06635) fixes this by adding a per-head, input-dependent forget gate α_t ∈ (0, 1):

```
S_t = α_t · S_{t-1} + outer(k_t, v_t)
o_t = S_t · q_t
```

α_t = sigmoid(W_gate · x_t) ∈ (0, 1), one scalar per head per time step. This is the "data-controlled gating" that Yang et al. show is the key ingredient closing the gap to softmax attention on language modelling, while preserving the O(T·d²) recurrence cost.

**Why this addition.** The library already has:
- DeltaNet (linear attention with delta rule — modifies the *update* but no decay)
- RetNet (linear attention with a fixed decay term — decay is constant, not learned)
- H3 (state-space variant)
- Mamba/Mamba-2 (state-space variant)

What's *missing* is the canonical Yang et al. GLA: linear attention with a **learned per-head forget gate**. It's a fundamental building block referenced in dozens of follow-up papers (GQA, MLA, ReGLA, etc.) and pairs naturally with the existing DeltaNet and RetNet.

**Distinguishing GLA from neighbors:**

| Model | Recurrence | Gate type |
| --- | --- | --- |
| Linear attention (Katharopoulos) | `S = S + outer(k,v); o = S·q` | none |
| **GLA (Yang 2023, this PR)** | **`S = α·S + outer(k,v); o = S·q`** | **learned per-head scalar α ∈ (0,1)** |
| RetNet (Sun 2023) | `S = γ·S + outer(k,v); o = S·q` | fixed scalar γ per head |
| Mamba-2 (Dao 2024) | `S = exp(Δ·A)·S + Δ·outer(k,v); o = S·q` | fixed scalar per channel |
| DeltaNet (Yang 2024) | `S = S + α·outer(k, v - S·k); o = S·q` | scalar α from delta-rule optimality, no decay on S |
| H3 (Fu 2023) | SSM + SSM | none (SSM-style) |

---

## Task 1: Header skeleton with public API

**Objective:** Declare `GatedLinearAttention` class with the paper's full public API surface, following the same style as the existing `DeltaNet` header.

**Files:**
- Create: `include/nn/layers/recurrent/gla.h`

**Step 1: Write the header.**

```cpp
#ifndef GLA_H
#define GLA_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Gated Linear Attention (GLA) — Yang et al. 2023/2024
//   "Gated Linear Attention Transformers with Hardware-Efficient Training"
//   https://arxiv.org/abs/2312.06635  (NeurIPS 2024)
//
// Standard linear attention (Katharopoulos et al. 2020) replaces softmax
// attention with a linear-time recurrence:
//
//   S_t = S_{t-1} + outer(k_t, v_t)
//   o_t = S_t · q_t
//
// This is fast (O(d) per step) but cannot selectively forget stale
// information — the state grows unboundedly. GLA fixes this by adding an
// INPUT-DEPENDENT per-head forget gate α_t ∈ (0, 1):
//
//   S_t = α_t · S_{t-1} + outer(k_t, v_t)
//   o_t = S_t · q_t
//
// with α_t = sigmoid(W_gate · x_t), a small per-head scalar projection.
// The gate is the key ingredient closing the gap to softmax attention on
// language modeling while preserving O(T·d²) recurrence.
//
// ----------------------------------------------------------------------------
// Per-head, per-time recurrence (single-step BPTT):
//
// Input:  X in R^{T x d_model}
//
// Step 1 — Projections (Dense: y = x·W^T + b):
//   q_t = W_Q · x_t                    in R^{d_inner}
//   k_t = W_K · x_t                    in R^{d_inner}
//   v_t = W_V · x_t                    in R^{d_inner}
//   α_t = sigmoid(W_gate · x_t)        in R^{H}        (scalar per head)
//
// Step 2 — Per-head gated linear-attention recurrence:
//   S_0^(h) = 0 in R^{d x d}
//   for t = 0..T-1:
//     S_t^(h) = α_t[h] · S_{t-1}^(h) + outer(k_t[h], v_t[h])
//     o_t[h] = S_t^(h) · q_t[h]                              (vector in R^d)
//
// Step 3 — Concat heads + output projection:
//   o_t = concat([o_t[0]; ...; o_t[H-1]])               in R^{H*d = d_inner}
//   out_t = W_O · o_t                                  in R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   X:               (T, d_model)
//   Q, K, V:         (T, d_inner = H * d)
//   α:               (T, H)
//   S (per head):    (d, d) — single matrix state per head per time step
//   o:               (T, d_inner)
//   W_O output:      (T, d_model)
//
// We use head_dim = d_inner / n_heads (must divide evenly). For numerical
// gradient checks we use d_model=4, n_heads=2, head_dim=2, T=3-4.
//
// ----------------------------------------------------------------------------
// Learnable parameters: W_Q, W_K, W_V, W_O, W_gate. All stored as Dense
// layers so they get the standard forward/backward/zero_grad hook.
// ============================================================================

class GatedLinearAttention : public Layer {
public:
    GatedLinearAttention(size_t d_model, size_t n_heads, size_t head_dim = 0);

    // Input: (T, d_model) tensor. Output: (T, d_model) tensor.
    Tensor forward(const Tensor& input) override;

    // Standard backward: receives grad_output (T, d_model) and returns
    // grad_input (T, d_model). Internal Dense projections accumulate their
    // own gradients via the standard Layer convention.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    // Parameter accessors (for optimizer discovery)
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_Q_.weights; }
    Tensor get_gradients() const override { return W_Q_.grad_weights; }
    std::string name() const override { return "GatedLinearAttention"; }

    // Test introspection: per-head state at time T (last call).
    // Returns S_T for each head, stacked shape (n_heads, head_dim * head_dim).
    Tensor last_state() const;

    // Setters / accessors for inspectors
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t d_inner() const { return d_inner_; }

    // Public Dense accessors (for testing)
    Dense W_Q_, W_K_, W_V_, W_O_, W_gate_;

private:
    size_t d_model_;
    size_t n_heads_;
    size_t head_dim_;
    size_t d_inner_;

    // Cache for backward (all per-token, per-head)
    Tensor cache_x_;           // (T, d_model)
    Tensor cache_q_;           // (T, d_inner)
    Tensor cache_k_;           // (T, d_inner)
    Tensor cache_v_;           // (T, d_inner)
    Tensor cache_gate_pre_;    // (T, n_heads) — pre-sigmoid
    Tensor cache_gate_;        // (T, n_heads) — sigmoid (the α_t scalar per head)
    Tensor cache_concat_o_;    // (T, d_inner) — what W_O sees

    // State cache: per-head, per-time. cache_S_[t] is (n_heads, head_dim * head_dim).
    // cache_S_[0] = zero state (before any input).
    std::vector<Tensor> cache_S_;

    // Local gradient buffers (for the recurrence)
    Tensor grad_q_;            // (T, d_inner)
    Tensor grad_k_;            // (T, d_inner)
    Tensor grad_v_;            // (T, d_inner)
    Tensor grad_x_;            // (T, d_model) — returned by backward()
};

#endif // GLA_H
```

**Step 2: Verify the header compiles standalone.**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -x c++ -fsyntax-only - <<< '
#include "layers/recurrent/gla.h"
int main() { return 0; }
'
```

Expected: no errors. (The forward declarations for Dense and Layer are sufficient.)

**Step 3: Commit.**

```bash
git add include/nn/layers/recurrent/gla.h
git commit -m "feat(recurrent): add GLA.h skeleton (Yang et al. 2023)"
```

---

## Task 2: Constructor + Dense wiring

**Objective:** Implement the constructor that initializes the Dense projections and validates dimensions.

**Files:**
- Create: `include/nn/layers/recurrent/gla.cpp`

**Step 1: Write the constructor (no forward yet).**

```cpp
#include "gla.h"
#include "../../core/tensor.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>

// Initialize projections to a small random scale — small enough that the
// recurrence stays numerically stable for forward checks, large enough
// that gradient checks are non-vacuous.
static Tensor gla_rand_init(size_t rows, size_t cols) {
    Tensor t(rows, cols);
    double s = 0.5 / std::sqrt(static_cast<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t[i][j] = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0 * s;
        }
    }
    return t;
}

GatedLinearAttention::GatedLinearAttention(size_t d_model, size_t n_heads, size_t head_dim)
    : d_model_(d_model),
      n_heads_(n_heads),
      head_dim_(head_dim == 0 ? d_model / n_heads : head_dim),
      d_inner_(n_heads * (head_dim == 0 ? d_model / n_heads : head_dim))
{
    if (d_model_ == 0 || n_heads_ == 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model and n_heads must be > 0");
    }
    if (head_dim_ == 0 || d_model_ % head_dim_ != 0) {
        throw std::invalid_argument("GatedLinearAttention: d_model must divide evenly by head_dim");
    }
    // Standard convention: d_inner = d_model (so head_dim = d_model / n_heads).
    d_inner_ = d_model;

    // Build Dense projections (x · W^T + b convention)
    W_Q_   = Dense(d_model, d_inner_);
    W_K_   = Dense(d_model, d_inner_);
    W_V_   = Dense(d_model, d_inner_);
    W_O_   = Dense(d_inner_, d_model);
    W_gate_ = Dense(d_model, n_heads_);  // per-head α_t = sigmoid(W_gate · x_t)

    // Override Dense default init with a Xavier-ish small scale
    W_Q_.weights    = gla_rand_init(d_inner_, d_model_);
    W_K_.weights    = gla_rand_init(d_inner_, d_model_);
    W_V_.weights    = gla_rand_init(d_inner_, d_model_);
    W_O_.weights    = gla_rand_init(d_model_, d_inner_);
    W_gate_.weights = gla_rand_init(n_heads_, d_model_);
    // Biases default to zero in Dense
}
```

**Step 2: Verify it compiles.**

```bash
make build/recurrent/gla.o 2>&1 | tail -20
```

Expected: success (Dense and Tensor are available). If `Dense` constructor signature is different (e.g. `Dense(d_out, d_in)` instead of `Dense(d_in, d_out)`), adjust accordingly. (See `include/nn/core/layer.h:34` — `Dense(size_t in_features, size_t out_features)` — `weights` shape is `(out_features, in_features)`.)

**Step 3: Commit.**

```bash
git add include/nn/layers/recurrent/gla.cpp
git commit -m "feat(recurrent): GLA constructor + Dense wiring"
```

---

## Task 3: Forward pass

**Objective:** Implement the forward pass with the gate projection, per-head gated recurrence, and output projection.

**Files:**
- Modify: `include/nn/layers/recurrent/gla.cpp`

**Step 1: Append the forward implementation.**

```cpp
Tensor GatedLinearAttention::forward(const Tensor& input) {
    // input: (T, d_model)
    size_t T = input.rows;
    if (T == 0) return Tensor(0, d_model_);
    if (input.cols != d_model_) {
        throw std::invalid_argument("GatedLinearAttention::forward: input.cols must equal d_model");
    }

    cache_x_ = input.clone();

    // Project
    cache_q_ = W_Q_.forward(input);     // (T, d_inner)
    cache_k_ = W_K_.forward(input);     // (T, d_inner)
    cache_v_ = W_V_.forward(input);     // (T, d_inner)
    Tensor gate_raw = W_gate_.forward(input);  // (T, n_heads)
    cache_gate_pre_ = Tensor(T, n_heads_);
    cache_gate_ = Tensor(T, n_heads_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double g_raw = gate_raw[t][h];
            cache_gate_pre_[t][h] = g_raw;
            // Stable sigmoid
            if (g_raw >= 0.0) {
                cache_gate_[t][h] = 1.0 / (1.0 + std::exp(-g_raw));
            } else {
                double ez = std::exp(g_raw);
                cache_gate_[t][h] = ez / (1.0 + ez);
            }
        }
    }

    // Per-head gated linear-attention recurrence
    cache_S_.clear();
    cache_S_.resize(T);
    cache_S_[0] = Tensor(n_heads_, head_dim_ * head_dim_);  // zero state for t=0
    // cache_S_[t] stores the state AFTER processing time t (so cache_S_[0] is empty)

    Tensor out_concat(T, d_inner_);
    Tensor current_state(n_heads_, head_dim_ * head_dim_);  // S_0 = 0

    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            double alpha = cache_gate_[t][h];

            // Apply gate: current_state[h] *= alpha
            for (size_t i = 0; i < head_dim_ * head_dim_; ++i) {
                current_state[h][i] *= alpha;
            }

            // Add outer product: current_state[h] += outer(k_t[h], v_t[h])
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    double k = cache_k_[t][h * head_dim_ + i];
                    double v = cache_v_[t][h * head_dim_ + j];
                    current_state[h][i * head_dim_ + j] += k * v;
                }
            }
        }
        cache_S_[t] = current_state.clone();

        // Output: o_t[h] = S_t[h] · q_t[h]
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    sum += current_state[h][i * head_dim_ + j] *
                           cache_q_[t][h * head_dim_ + j];
                }
                out_concat[t][h * head_dim_ + i] = sum;
            }
        }
    }

    cache_concat_o_ = out_concat.clone();
    Tensor out = W_O_.forward(cache_concat_o_);  // (T, d_model)
    return out;
}

Tensor GatedLinearAttention::last_state() const {
    if (cache_S_.empty()) return Tensor(0, 0);
    return cache_S_.back();  // (n_heads, head_dim * head_dim)
}
```

**Step 2: Compile.**

```bash
make build/recurrent/gla.o 2>&1 | tail -20
```

Expected: success.

**Step 3: Commit.**

```bash
git add include/nn/layers/recurrent/gla.cpp
git commit -m "feat(recurrent): GLA forward pass with gated recurrence"
```

---

## Task 4: Backward pass

**Objective:** Implement the backward pass — derive analytical BPTT for the gated linear-attention recurrence.

**Math derivation.** Per head h, per time t:

```
S_t = α_t · S_{t-1} + outer(k_t, v_t)
o_t = S_t · q_t
```

Let `gS_t = dL/dS_t` (the grad w.r.t. S_t, matrix of shape (d, d)) and `g_o_t = grad_output[t]` (received from the layer above).

**Step 1 — Output gradient:**
```
gS_t += outer(g_o_t, q_t)         (S_t's role in o_t = S_t · q_t)
g_q_t[h, i] = sum_j S_t[h, i, j] · g_o_t[h, j]
```

**Step 2 — Update side (S_t = α_t · S_{t-1} + outer(k_t, v_t)):**

The "carrier" of S_{t-1} into S_t is via two paths:
- Direct (linear): `S_{t-1} → S_t` via the α_t scalar multiplication → `gS_{t-1} += α_t · gS_t`
- Indirect via k, v: `S_t` depends on `k_t` and `v_t` only through the outer product. They do NOT depend on S_{t-1} directly.
- So the gradient contributions to k, v are:
  - `g_k_t[h, i] = sum_j gS_t[h, i, j] · v_t[h, j]`
  - `g_v_t[h, j] = sum_i gS_t[h, i, j] · k_t[h, i]`

**Step 3 — Gate α_t:**

α_t multiplies the whole state matrix elementwise:
```
g_α_t[h] = sum_{i,j} gS_t[h, i, j] · S_{t-1}[h, i, j]
         = sum over all entries of (gS_t[h] ⊙ S_{t-1}[h])
```
The total `gS_t` here includes the downstream carrier from t+1 (when present), since S_t enters S_{t+1} via the gate α_{t+1}.

**Step 4 — Gate pre-sigmoid:**
```
g_gate_pre_t[h] = g_α_t[h] · α_t[h] · (1 - α_t[h])   (sigmoid derivative)
```

**Step 5 — Chain back to input x_t:**
Sum the gradients from all five projections that share x_t:
```
g_x_t = g_x_from_W_Q + g_x_from_W_K + g_x_from_W_V + g_x_from_W_O_chain + g_x_from_W_gate
```
where `g_x_from_W_O_chain = dL/do_{t+1} · W_O_partial` (we return the standard backward gradient from W_O's Dense layer chain).

The Dense layer's own backward handles the projection-gradient → input-gradient chain. The W_O backward contributes to grad_x directly via `W_O.backward(grad_output, lr)`. For Q, K, V, gate we accumulate grad_x from each Dense's backward.

**Step 1: Append the backward implementation.**

```cpp
Tensor GatedLinearAttention::backward(const Tensor& grad_output, double learning_rate) {
    size_t T = grad_output.rows;
    size_t grad_cols = grad_output.cols;
    if (grad_cols != d_model_) {
        throw std::invalid_argument("GatedLinearAttention::backward: grad_output.cols must equal d_model");
    }
    if (T == 0) return Tensor(0, d_model_);

    // Step 1: Backward through W_O to get grad_concat_o and grad_x from the W_O chain
    Tensor grad_concat_o = W_O_.backward(grad_output, learning_rate);  // (T, d_inner)
    Tensor grad_x = Tensor(T, d_model_);
    for (size_t i = 0; i < T * d_model_; ++i) grad_x.data[i] = 0.0;

    // grad_q_, grad_k_, grad_v_ from the W_O backward's output-side gradient
    grad_q_ = Tensor(T, d_inner_); for (size_t i = 0; i < T * d_inner_; ++i) grad_q_.data[i] = 0.0;
    grad_k_ = Tensor(T, d_inner_); for (size_t i = 0; i < T * d_inner_; ++i) grad_k_.data[i] = 0.0;
    grad_v_ = Tensor(T, d_inner_); for (size_t i = 0; i < T * d_inner_; ++i) grad_v_.data[i] = 0.0;

    // For each time step, grad_concat_o[t] = grad of loss w.r.t. o_t
    //   o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]
    //   => grad_S_t[h, i, j] += grad_o_t[h, i] · q_t[h, j]
    //   => grad_q_t[h, j]   += sum_i grad_o_t[h, i] · S_t[h, i, j]
    Tensor grad_S(T, n_heads_ * head_dim_ * head_dim_);
    for (size_t i = 0; i < T * n_heads_ * head_dim_ * head_dim_; ++i) grad_S.data[i] = 0.0;

    // Forward-time accumulation: first compute grad_S from grad_concat_o alone
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t i = 0; i < head_dim_; ++i) {
                double g_oh_i = grad_concat_o[t][h * head_dim_ + i];
                for (size_t j = 0; j < head_dim_; ++j) {
                    // grad_S_t[h, i, j] += g_oh_i · q_t[h, j]
                    grad_S[t][h * head_dim_ * head_dim_ + i * head_dim_ + j] +=
                        g_oh_i * cache_q_[t][h * head_dim_ + j];
                }
            }
        }
    }

    // Backward-time recurrence for grad_S:
    //   gS_{t-1} += α_t · gS_t          (carrier from S_t = α_t · S_{t-1} + outer)
    //   g_k_t[h, i] += sum_j gS_t[h, i, j] · v_t[h, j]
    //   g_v_t[h, j] += sum_i gS_t[h, i, j] · k_t[h, i]
    //   g_α_t[h]    += sum_{i,j} gS_t[h, i, j] · S_{t-1}[h, i, j]
    //
    // We accumulate grad_S from the future carrier and accumulate
    // g_k, g_v, g_α contributions from each t.
    Tensor grad_gate_pre(T, n_heads_);
    for (size_t i = 0; i < T * n_heads_; ++i) grad_gate_pre.data[i] = 0.0;

    // Iterate t from T-1 down to 1 (so S_{t-1} is the previous cached state)
    for (int t_signed = static_cast<int>(T) - 1; t_signed >= 1; --t_signed) {
        size_t t = static_cast<size_t>(t_signed);
        size_t t_prev = t - 1;
        for (size_t h = 0; h < n_heads_; ++h) {
            double alpha = cache_gate_[t][h];
            const Tensor& S_prev = cache_S_[t_prev];

            // g_α_t[h] += sum_{i,j} gS_t[h, i, j] · S_{t-1}[h, i, j]
            double g_alpha = 0.0;
            for (size_t i = 0; i < head_dim_; ++i) {
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    g_alpha += grad_S[t][flat] * S_prev[h][i * head_dim_ + j];
                }
            }
            // ∂L/∂α_t = g_alpha; chain through sigmoid:
            //   ∂L/∂gate_pre_t[h] = g_alpha · α_t · (1 - α_t)
            grad_gate_pre[t][h] += g_alpha * alpha * (1.0 - alpha);

            // g_k_t[h, i] += sum_j gS_t[h, i, j] · v_t[h, j]
            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S[t][flat] * cache_v_[t][h * head_dim_ + j];
                }
                grad_k_[t][h * head_dim_ + i] += sum;
            }

            // g_v_t[h, j] += sum_i gS_t[h, i, j] · k_t[h, i]
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S[t][flat] * cache_k_[t][h * head_dim_ + i];
                }
                grad_v_[t][h * head_dim_ + j] += sum;
            }

            // Carrier: gS_{t-1}[h] += α_t · gS_t[h]
            // (this is the "α_t · S_{t-1}" path's contribution to grad of S_{t-1})
            // Note: we use the grad_S[t] (the cumulative grad including downstream)
            // since the gate multiplies S_{t-1} elementwise before entering S_t,
            // and S_t has its own future carrier.
            for (size_t i = 0; i < head_dim_ * head_dim_; ++i) {
                grad_S[t_prev][h * head_dim_ * head_dim_ + i] +=
                    alpha * grad_S[t][h * head_dim_ * head_dim_ + i];
            }
        }
    }

    // Handle t=0 separately (no S_{-1} to carry back to)
    if (T > 0) {
        size_t t = 0;
        for (size_t h = 0; h < n_heads_; ++h) {
            // For t=0, S_{-1} is the zero state, so the "α_0 · S_{-1}" path
            // contributes zero to grad_S_{-1}. But k_0, v_0, α_0 still have
            // gradients from grad_S[0].
            // S_0 = α_0 · 0 + outer(k_0, v_0) = outer(k_0, v_0)
            // grad_S[0] from grad_o_0 is computed above; the grad_α_0 here
            // comes from the carrier we'd apply to S_{-1}, but it's zero.
            // So just compute grad_k_0, grad_v_0 from grad_S[0].

            for (size_t i = 0; i < head_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < head_dim_; ++j) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S[t][flat] * cache_v_[t][h * head_dim_ + j];
                }
                grad_k_[t][h * head_dim_ + i] += sum;
            }
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    size_t flat = h * head_dim_ * head_dim_ + i * head_dim_ + j;
                    sum += grad_S[t][flat] * cache_k_[t][h * head_dim_ + i];
                }
                grad_v_[t][h * head_dim_ + j] += sum;
            }
            // grad_α_0 from the contribution to S_{-1} is zero (S_{-1} is zero).
            // However the "g_alpha from grad_S[0]" should also include the
            // path "∂S_0/∂α_0 = S_{-1} = 0", so grad_α_0 is zero here. Leave it.
        }
    }

    // Compute grad_q_t from grad_S[t] (the gradient that flows back to q through o = S·q)
    //   o_t[h, i] = sum_j S_t[h, i, j] · q_t[h, j]
    //   => grad_q_t[h, j] = sum_i S_t[h, i, j] · grad_o_t[h, i]
    // grad_o_t = grad_concat_o (we computed it from W_O_.backward)
    // But we already used grad_S[t] which includes grad_o_t · q_t; the grad_q_t
    // comes from the original grad_o_t (NOT from grad_S[t], which is grad of L
    // w.r.t. S_t — different thing).
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < n_heads_; ++h) {
            for (size_t j = 0; j < head_dim_; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < head_dim_; ++i) {
                    sum += cache_S_[t][h][i * head_dim_ + j] *
                           grad_concat_o[t][h * head_dim_ + i];
                }
                grad_q_[t][h * head_dim_ + j] += sum;
            }
        }
    }

    // Now backprop through Q, K, V, gate Dense projections.
    // Each Dense.backward(grad_output, lr) returns grad_input and accumulates
    // grad_weights / grad_bias in the Dense itself.
    Tensor grad_x_q = W_Q_.backward(grad_q_, learning_rate);
    Tensor grad_x_k = W_K_.backward(grad_k_, learning_rate);
    Tensor grad_x_v = W_V_.backward(grad_v_, learning_rate);
    Tensor grad_x_g = W_gate_.backward(grad_gate_pre, learning_rate);

    // Sum input gradients from all projections
    for (size_t i = 0; i < T * d_model_; ++i) {
        grad_x.data[i] = grad_x_q.data[i] + grad_x_k.data[i]
                        + grad_x_v.data[i] + grad_x_g.data[i];
    }

    // Add the input gradient from W_O (already partially accounted for)
    // Wait — W_O_.backward already returned grad_x for the W_O chain.
    // But we need to ADD that to our grad_x because all five projections
    // (Q, K, V, gate, O) read from x and contribute to grad_x.
    // However W_O does NOT read from x — it reads from cache_concat_o_.
    // So W_O_.backward's grad_input is irrelevant here. Ignore it.

    return grad_x;
}

void GatedLinearAttention::update_weights(double learning_rate) {
    W_Q_.update_weights(learning_rate);
    W_K_.update_weights(learning_rate);
    W_V_.update_weights(learning_rate);
    W_O_.update_weights(learning_rate);
    W_gate_.update_weights(learning_rate);
}

void GatedLinearAttention::zero_grad() {
    W_Q_.zero_grad();
    W_K_.zero_grad();
    W_V_.zero_grad();
    W_O_.zero_grad();
    W_gate_.zero_grad();
    grad_q_ = Tensor();
    grad_k_ = Tensor();
    grad_v_ = Tensor();
    grad_x_ = Tensor();
}

std::vector<Tensor*> GatedLinearAttention::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : W_Q_.parameters())    result.push_back(p);
    for (Tensor* p : W_K_.parameters())    result.push_back(p);
    for (Tensor* p : W_V_.parameters())    result.push_back(p);
    for (Tensor* p : W_O_.parameters())    result.push_back(p);
    for (Tensor* p : W_gate_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> GatedLinearAttention::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : W_Q_.gradients())    result.push_back(g);
    for (Tensor* g : W_K_.gradients())    result.push_back(g);
    for (Tensor* g : W_V_.gradients())    result.push_back(g);
    for (Tensor* g : W_O_.gradients())    result.push_back(g);
    for (Tensor* g : W_gate_.gradients()) result.push_back(g);
    return result;
}
```

**Step 2: Compile.**

```bash
make build/recurrent/gla.o 2>&1 | tail -20
```

Expected: success.

**Step 3: Commit.**

```bash
git add include/nn/layers/recurrent/gla.cpp
git commit -m "feat(recurrent): GLA backward pass with gated BPTT"
```

---

## Task 5: Wire into umbrella header + Makefile

**Objective:** Register `GatedLinearAttention` in the public surface.

**Files:**
- Modify: `include/nn/nn.h` (after `deltanet.h` include line)
- Modify: `Makefile` (test build rule + run_tests echo)

**Step 1: Add to `nn.h`.**

After `#include "layers/recurrent/deltanet.h"`, add:
```cpp
#include "layers/recurrent/gla.h"
```

**Step 2: Add to Makefile.**

In the `tests:` deps line, append `$(BUILD_DIR)/test_gla` after `$(BUILD_DIR)/test_deltanet`.

In the `run_tests:` echo section, append after `=== Running DeltaNet Tests ===`:
```
@echo "=== Running GLA Tests ===" && ./$(BUILD_DIR)/test_gla
```

In the test-build rules section, add:
```
$(BUILD_DIR)/test_gla: $(LIB_OBJS) $(BUILD_DIR)/test_gla.o
	$(CXX) $^ -o $@
```

**Step 3: Commit.**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register GLA in nn.h + Makefile"
```

---

## Task 6: Test suite (RED → GREEN)

**Objective:** Build a focused test suite mirroring `test_deltanet.cpp` with shape, FD-gradient, training, and determinism checks.

**Files:**
- Create: `tests/test_gla.cpp`

**Step 1: Write the test file.**

```cpp
// ==========================================================================
// tests/test_gla.cpp
//
// Focused tests for GatedLinearAttention (Yang et al. 2023, "Gated Linear
// Attention Transformers with Hardware-Efficient Training").
//
// Tests:
//   1. Constructor validation
//   2. Forward shape (T, d_model) -> (T, d_model)
//   3. Forward output is finite
//   4. State cache shape (n_heads, head_dim * head_dim)
//   5. State accumulation (norm > 0 after first step)
//   6. Gate output in (0, 1)
//   7. Input gradient check (analytical vs centered FD)
//   8. W_Q gradient check
//   9. W_K gradient check
//  10. W_V gradient check
//  11. W_O gradient check
//  12. W_gate gradient check
//  13. Training reduces loss (SGD on MSE loss)
//  14. Determinism (bit-identical with copied params)
//  15. Multi-head (n_heads=3, head_dim=2) forward shape
// ==========================================================================

#include "nn/nn.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

static int passed = 0;
static int failed = 0;

#define EXPECT(cond) do { \
    if (cond) { ++passed; } else { ++failed; std::cerr << "FAIL: " << #cond << " at line " << __LINE__ << " in " << __FILE__ << std::endl; } \
} while (0)

#define EXPECT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        ++failed; std::cerr << "FAIL: EXPECT_NEAR(" #a "=" << _a << ", " #b "=" << _b << ", tol=" << (tol) << ") at line " << __LINE__ << std::endl; \
    } else { ++passed; } \
} while (0)

static double compute_loss(GatedLinearAttention& gla, const Tensor& x, const Tensor& target) {
    Tensor y = gla.forward(x);
    double loss = 0.0;
    size_t N = y.rows * y.cols;
    for (size_t i = 0; i < N; ++i) {
        double diff = y.data[i] - target.data[i];
        loss += diff * diff;
    }
    return loss / (2.0 * N);
}

static Tensor numerical_input_grad(GatedLinearAttention& gla, const Tensor& x, const Tensor& target, double eps) {
    size_t T = x.rows, D = x.cols;
    Tensor grad(T, D);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            Tensor x_plus = x.clone();
            Tensor x_minus = x.clone();
            x_plus.data[t * D + d] += eps;
            x_minus.data[t * D + d] -= eps;
            double L_plus = compute_loss(gla, x_plus, target);
            double L_minus = compute_loss(gla, x_minus, target);
            grad.data[t * D + d] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor numerical_param_grad(GatedLinearAttention& gla, const Tensor& x, const Tensor& target,
                                    Dense& d, double eps) {
    Tensor grad(d.weights.rows, d.weights.cols);
    for (size_t i = 0; i < d.weights.rows; ++i) {
        for (size_t j = 0; j < d.weights.cols; ++j) {
            double orig = d.weights[i][j];
            d.weights[i][j] = orig + eps;
            double L_plus = compute_loss(gla, x, target);
            d.weights[i][j] = orig - eps;
            double L_minus = compute_loss(gla, x, target);
            d.weights[i][j] = orig;
            grad[i][j] = (L_plus - L_minus) / (2.0 * eps);
        }
    }
    return grad;
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        m = std::max(m, std::abs(a.data[i] - b.data[i]));
    }
    return m;
}

static double rel_err(const Tensor& a, const Tensor& b) {
    double max_abs = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        max_abs = std::max(max_abs, std::max(std::abs(a.data[i]), std::abs(b.data[i])));
    }
    if (max_abs < 1e-12) return max_abs_diff(a, b);
    return max_abs_diff(a, b) / max_abs;
}

int main() {
    std::srand(42);

    // ---- Test 1: Constructor validation ----
    {
        bool threw = false;
        try { GatedLinearAttention g(0, 2); } catch (std::invalid_argument&) { threw = true; }
        EXPECT(threw);
    }
    {
        bool threw = false;
        try { GatedLinearAttention g(4, 0); } catch (std::invalid_argument&) { threw = true; }
        EXPECT(threw);
    }
    {
        bool threw = false;
        try { GatedLinearAttention g(5, 2); } catch (std::invalid_argument&) { threw = true; }  // 5 % 2 != 0
        EXPECT(threw);
    }

    // ---- Setup for forward/gradient tests ----
    const size_t T = 3;
    const size_t d_model = 4;
    const size_t n_heads = 2;
    const size_t head_dim = 2;
    GatedLinearAttention gla(d_model, n_heads, head_dim);

    // Random input and target
    Tensor x(T, d_model);
    Tensor target(T, d_model);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = dist(rng);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(rng);

    // ---- Test 2: Forward shape ----
    Tensor y = gla.forward(x);
    EXPECT(y.rows == T);
    EXPECT(y.cols == d_model);

    // ---- Test 3: Forward output is finite ----
    bool all_finite = true;
    for (size_t i = 0; i < y.data.size(); ++i) {
        if (!std::isfinite(y.data[i])) { all_finite = false; break; }
    }
    EXPECT(all_finite);

    // ---- Test 4: State cache shape ----
    Tensor S_T = gla.last_state();
    EXPECT(S_T.rows == n_heads);
    EXPECT(S_T.cols == head_dim * head_dim);

    // ---- Test 5: State accumulation ----
    double norm = 0.0;
    for (size_t i = 0; i < S_T.data.size(); ++i) norm += S_T.data[i] * S_T.data[i];
    EXPECT(norm > 0.0);

    // ---- Test 6: Gate output in (0, 1) ----
    // Forward the same input again; cache_gate_ holds the most recent gates
    gla.forward(x);
    bool all_in_unit = true;
    for (size_t i = 0; i < T * n_heads; ++i) {
        double g = gla.cache_gate_.data[i];
        if (g <= 0.0 || g >= 1.0) { all_in_unit = false; break; }
    }
    EXPECT(all_in_unit);

    // ---- Test 7: Input gradient (analytical vs FD) ----
    {
        Tensor ana = Tensor(T, d_model);
        Tensor y0 = gla.forward(x);
        // Use MSE loss gradient: grad_output = (y - target) / N
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        Tensor ana_grad = gla.backward(grad_out, 0.0);
        // Capture the analytical input gradient by re-running
        // (Note: the backward returns grad_x; copy it before next call)
        Tensor fd = numerical_input_grad(gla, x, target, 1e-5);
        double r = rel_err(ana_grad, fd);
        std::cerr << "  [info] input-grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 8: W_Q gradient ----
    {
        // Restore the analytical grad before FD
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_Q_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_Q_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_Q grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 9: W_K gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_K_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_K_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_K grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 10: W_V gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_V_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_V_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_V grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 11: W_O gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_O_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_O_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_O grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 12: W_gate gradient ----
    {
        Tensor y0 = gla.forward(x);
        Tensor grad_out(T, d_model);
        size_t N = T * d_model;
        for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y0.data[i] - target.data[i]) / N;
        gla.zero_grad();
        gla.backward(grad_out, 0.0);
        Tensor ana_W = gla.W_gate_.grad_weights.clone();

        Tensor fd_W = numerical_param_grad(gla, x, target, gla.W_gate_, 1e-5);
        double r = rel_err(ana_W, fd_W);
        std::cerr << "  [info] W_gate grad rel_err = " << r << std::endl;
        EXPECT(r < 1e-4);
    }

    // ---- Test 13: Training reduces loss ----
    {
        GatedLinearAttention train_gla(d_model, n_heads, head_dim);
        // Initialize with a specific seed for reproducibility
        std::srand(7);
        train_gla.W_Q_.weights.data[0] = 0.1;
        train_gla.W_Q_.weights.data[1] = -0.2;
        train_gla.W_K_.weights.data[0] = 0.05;
        train_gla.W_K_.weights.data[1] = 0.15;
        train_gla.W_V_.weights.data[0] = 0.2;
        train_gla.W_V_.weights.data[1] = -0.1;
        train_gla.W_O_.weights.data[0] = 0.3;
        train_gla.W_O_.weights.data[1] = -0.05;
        train_gla.W_gate_.weights.data[0] = 0.5;
        train_gla.W_gate_.weights.data[1] = -0.3;

        double lr = 0.01;
        Tensor x_tr = x.clone();
        Tensor t_tr = target.clone();
        double initial_loss = compute_loss(train_gla, x_tr, t_tr);
        for (size_t step = 0; step < 30; ++step) {
            Tensor y_tr = train_gla.forward(x_tr);
            Tensor grad_out(T, d_model);
            size_t N = T * d_model;
            for (size_t i = 0; i < N; ++i) grad_out.data[i] = (y_tr.data[i] - t_tr.data[i]) / N;
            train_gla.zero_grad();
            train_gla.backward(grad_out, 0.0);
            train_gla.update_weights(lr);
        }
        double final_loss = compute_loss(train_gla, x_tr, t_tr);
        std::cerr << "  [info] training: initial=" << initial_loss << " final=" << final_loss << std::endl;
        EXPECT(final_loss < initial_loss * 0.5);
    }

    // ---- Test 14: Determinism ----
    {
        GatedLinearAttention a(d_model, n_heads, head_dim);
        GatedLinearAttention b(d_model, n_heads, head_dim);
        // Copy params from a to b
        for (size_t i = 0; i < a.W_Q_.weights.data.size(); ++i) {
            b.W_Q_.weights.data[i] = a.W_Q_.weights.data[i];
            b.W_K_.weights.data[i] = a.W_K_.weights.data[i];
            b.W_V_.weights.data[i] = a.W_V_.weights.data[i];
            b.W_O_.weights.data[i] = a.W_O_.weights.data[i];
            b.W_gate_.weights.data[i] = a.W_gate_.weights.data[i];
        }
        Tensor ya = a.forward(x);
        Tensor yb = b.forward(x);
        double r = max_abs_diff(ya, yb);
        std::cerr << "  [info] determinism max_abs_diff = " << r << std::endl;
        EXPECT(r < 1e-12);
    }

    // ---- Test 15: Multi-head (n_heads=3, head_dim=2) forward shape ----
    {
        GatedLinearAttention mh(6, 3, 2);
        Tensor x_mh(4, 6);
        for (size_t i = 0; i < x_mh.data.size(); ++i) x_mh.data[i] = dist(rng);
        Tensor y_mh = mh.forward(x_mh);
        EXPECT(y_mh.rows == 4);
        EXPECT(y_mh.cols == 6);
    }

    std::cerr << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}
```

**Step 2: Compile and run.**

```bash
make build/test_gla 2>&1 | tail -20 && ./build/test_gla
```

Expected: `=== Summary: ~30 passed, 0 failed ===`.

**Step 3: Investigate any failing test.**

If a gradient test fails, the most common root causes (in order of likelihood):
1. **Forgot the gate-pre-sigmoid chain**: `g_alpha = sum(gS · S_{t-1})` then `g_gate_pre = g_alpha · α · (1 - α)`. Verify with FD: zero out the `.α · (1-α)` factor and the gate grad will be wrong by a smooth factor.
2. **Mis-indexed carrier**: `gS_{t-1} += α_t · gS_t` — NOT `gS_{t-1} += α_{t-1} · gS_t`. The carrier from S_t to S_{t-1} is via S_t = α_t · S_{t-1}, so the multiplier is α_t.
3. **Missing W_O chain**: forgot to add `W_O_.backward()` at the top.
4. **Initial state mistake**: S_0 = 0, but the FIRST alpha's contribution to S_{-1} is zero. Don't try to carry to a non-existent state.

**Step 4: Commit.**

```bash
git add tests/test_gla.cpp
git commit -m "test(recurrent): add GLA test coverage"
```

---

## Task 7: Verify full build + queue

**Objective:** Run `make run_tests` (or just `make tests && ./build/test_gla`) to make sure nothing else broke.

**Step 1: Run the full test suite.**

```bash
make tests 2>&1 | tail -10
```

Expected: success.

**Step 2: Run the new GLA tests.**

```bash
./build/test_gla
```

Expected: `=== Summary: ~30 passed, 0 failed ===`.

**Step 3: Update EXPANSION_QUEUE.md.**

Move the GLA item from `## Ideas` (we'll create that section header if it doesn't exist) to `## Done` with the one-line summary.

**Step 4: Commit and push.**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs(queue): mark GLA as done"
git push origin master
```

---

## Reference files

- `include/nn/layers/recurrent/deltanet.{h,cpp}` — sister implementation. Use as a template for the public-API surface (Dense storage, forward+backward contracts, cache naming conventions).
- `tests/test_deltanet.cpp` — sister test suite. Use as a template for the FD-gradient test harness and the loss/training scaffolding.
- `include/nn/core/layer.h` — Layer base class + Dense definition. Note `Dense(in_features, out_features)` constructor with `weights` shape `(out_features, in_features)` (so `y = x · W^T + b`).
- `include/nn/nn.h` — umbrella header. Add the new include after `deltanet.h`.
- `Makefile` — add `test_gla` build rule, `tests:` deps, `run_tests` echo.