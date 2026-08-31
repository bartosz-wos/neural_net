# Titans MAL (Memory as a Layer) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Complete the Titans trio (MAC, MAG, MAL) by adding `TitansMAL` — the third Behrouz et al. 2025 variant where the persistent test-time neural memory `M` is used as a downstream layer that processes a per-token-learned-gated version of the input.

**Architecture:** A recurrent layer (same per-segment per-token surprise-weighted momentum update of `M` as MAC/MAG), but the *use* of `M` in the output is the simplest of the three: `y_t = M_t · x̃_t` where `x̃_t = p_t ⊙ x_t` is the input modulated by a learned per-token 1-D gate `p_t = sigmoid(W_p · x_t + b_p)` (paper Eq. 29-30, `p_1 ... p_Np` are learnable per-channel multiplicative gates on `x`).

**Tech Stack:** Existing `Tensor`, `Layer`, `Dense` from `include/nn/core/`. Standard `<random>`, `<cmath>`, `<algorithm>`. New file `include/nn/layers/recurrent/titans_mal.{h,cpp}` + test `tests/test_titans_mal.cpp`. Mirror MAC/MAG structure.

**Reference:** Behrouz et al. 2025, "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663), §4.3 (Memory as a Layer, Eqs. 29-30) and §3.2 (neural memory M, Eqs. 9-10).

---

## Why this completes the trilogy

- **MAC** (`y_t = M_t · q_t`): memory output *is* the token's contextual representation. q is a separate projection.
- **MAG** (`y_t = (M_t · x_t) ⊙ x_t`): memory output *gates* the raw input. q is unused.
- **MAL** (`y_t = M_t · (p_t ⊙ x_t)`): memory processes the input **after** it has been modulated by a learned per-channel gate. The paper writes the gate as a 1-vector `p_t ∈ (0,1)^d` from `p_t = sigmoid(W_p · x_t + b_p)`. The output is the cleanest of the three — pure linear projection through M with input-side preconditioning.

The M-update rule is identical to MAC/MAG (same surprise-weighted momentum), so the TitansMAL backward reuses the entire M-update chain with only the **output-side** derivative being different. This is structurally identical to MAC's `y_t = M_t · q_t` chain with `q_t ← p_t ⊙ x_t` (a per-channel-modulated `q_t`), plus the additional `W_p` gradient chain through the gate.

---

## Task Structure (bite-sized, TDD each)

### Task 1: Header + constructor

**Files:**
- Create: `include/nn/layers/recurrent/titans_mal.h`
- Test: `tests/test_titans_mal.cpp` (skeleton)

**Step 1: Write failing tests for constructor (TDD RED)**

```cpp
// test_titans_mal.cpp (skeleton — Task 1)
#include "nn/nn.h"
int main() {
    // Test 1: d_model=0 throws
    bool threw = false;
    try { TitansMAL bad(0, 4); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    // Test 2: d_inner=0 defaults to d_model (no throw)
    TitansMAL ok(4, 0);
    assert(ok.d_inner() == 4);

    // Test 3: d_inner!=d_model throws
    threw = false;
    try { TitansMAL bad(4, 7); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    // Test 4: name() == "TitansMAL"
    TitansMAL layer(4, 4);
    assert(layer.name() == "TitansMAL");

    // Test 5: 6-tensor parameters() contract (one more than MAC/MAG — W_p added)
    auto p = layer.parameters();
    assert(p.size() == 6);
    auto g = layer.gradients();
    assert(g.size() == 6);
    // p[0]: W_qkv.weights  (3*d, d)
    assert(p[0]->rows == 12 && p[0]->cols == 4);
    // p[1]: W_qkv.bias     (1, 3*d)
    assert(p[1]->rows == 1 && p[1]->cols == 12);
    // p[2]: W_alpha.weights (1, d+1)
    assert(p[2]->rows == 1 && p[2]->cols == 5);
    // p[3]: W_alpha.bias    (1, 1)
    assert(p[3]->rows == 1 && p[3]->cols == 1);
    // p[4]: W_p.weights    (d, d)   — NEW (MAL-specific input-gate projection)
    assert(p[4]->rows == 4 && p[4]->cols == 4);
    // p[5]: W_p.bias       (1, d)
    assert(p[5]->rows == 1 && p[5]->cols == 4);
    // Note: M is NOT in parameters() — same as MAC/MAG (M is state, get_weights() returns it)
    return 0;
}
```

**Step 2: Run test — it fails to compile (header doesn't exist). Expected.**

**Step 3: Write the header**

```cpp
// include/nn/layers/recurrent/titans_mal.h
#ifndef TITANS_MAL_H
#define TITANS_MAL_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Titans MAL (Memory as a Layer) — Behrouz et al. 2025 §4.3
//   "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663)
//
// Third of the three Titans variants (MAC < MAG < MAL):
//   y_t = M_t · x̃_t   where   x̃_t = p_t ⊙ x_t,   p_t = sigmoid(W_p · x_t + b_p)
//
// M is updated by the same surprise-weighted momentum rule as MAC/MAG (§3.2,
// Eqs. 9-10):
//   q_t, k_t, v_t = W_qkv · x_t
//   η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
//   α_t = sigmoid(W_α · [x_t; ||v_t||]) · η_t
//   M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
//
// The MAL-specific part is the per-token learnable INPUT gate p_t and the
// cleanest output y_t = M_t · x̃_t (no ⊙, no separate q_t). The downstream
// SW-Attn from paper Eq. 31 is intentionally NOT in this layer — it composes
// cleanly with the existing SlidingWindowAttention layer.
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   * x:               (T, d_model)
//   * W_qkv.weights:   (3*d_model, d_model)
//   * W_qkv.bias:      (1, 3*d_model)
//   * W_alpha.weights: (1, d_model + 1)
//   * W_alpha.bias:    (1, 1)
//   * W_p.weights:     (d_model, d_model)        — MAL-specific input gate
//   * W_p.bias:        (1, d_model)              — input gate bias
//   * M (persistent):  (d_model, d_model)
//
// Caching for backward (T tokens):
//   last_input_       (T, d_model)
//   last_k_t_         (T, d_model)
//   last_v_t_         (T, d_model)
//   last_v_norm_      (T, 1)
//   last_eta_         (T, 1)
//   last_alpha_       (T, 1)
//   last_M_t_         (T+1)*d_model rows × d_model cols (M after each step)
//   last_p_t_         (T, d_model)               — input gate (post-sigmoid)
//   last_x_tilde_     (T, d_model)               — p_t ⊙ x_t (the q-equivalent)
//   last_y_t_         (T, d_model)
//
// Backward chain (per-token, last to first):
//   dL/dy_t → split:
//     direct: dx_t via W_p → x̃_t → p_t (input-side gate chain)
//     M-chain: dx_tilde → dM_post → (1-α)·dM_carrier + α·outer(v,k)
//       AND dx_tilde feeds into M-update via the MAC-equivalent output path.
//   Same MAC chain: dα ← dM_post·(v⊗k - M_prev); dα → dz → W_qkv.weights/bias
//   + M-specific: dα → dx (surprise chain) identical to MAC
// ============================================================================

class TitansMAL : public Layer {
public:
    size_t d_model_;
    size_t d_inner_;
    size_t seg_len_;

    Dense W_qkv_;            // (d_model → 3*d_model)
    Dense W_alpha_;          // (d_model+1 → 1)
    Dense W_p_;              // (d_model → d_model)        — MAL-specific
    Tensor M_;               // (d_model, d_model)         — persistent memory

    Tensor grad_W_qkv_w_;
    Tensor grad_W_qkv_b_;
    Tensor grad_W_alpha_w_;
    Tensor grad_W_alpha_b_;
    Tensor grad_W_p_w_;
    Tensor grad_W_p_b_;
    Tensor grad_M_;

    TitansMAL(size_t d_model, size_t d_inner = 0, size_t seg_len = 0);
    ~TitansMAL() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return M_; }
    Tensor get_gradients() const override { return grad_M_; }
    std::string name() const override { return "TitansMAL"; }

    void copy_params_from(const TitansMAL& other);

    size_t d_model() const { return d_model_; }
    size_t d_inner() const { return d_inner_; }
    size_t seg_len() const { return seg_len_; }

    Tensor last_input_;
    Tensor last_k_t_;
    Tensor last_v_t_;
    Tensor last_v_norm_;
    Tensor last_eta_;
    Tensor last_alpha_;
    Tensor last_M_t_;
    Tensor last_p_t_;
    Tensor last_x_tilde_;
    Tensor last_y_t_;
    Tensor last_qkv_;
};

#endif
```

**Step 4: Run test — it still fails to compile (no .cpp). Expected.**

**Step 5: Write minimal .cpp stub to make RED tests pass (only Task 1's tests)**

```cpp
// include/nn/layers/recurrent/titans_mal.cpp
#include "titans_mal.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

TitansMAL::TitansMAL(size_t d_model, size_t d_inner, size_t seg_len)
    : d_model_(d_model), seg_len_(seg_len),
      W_qkv_(d_model, 3 * d_model),
      W_alpha_(d_model + 1, 1),
      W_p_(d_model, d_model) {
    if (d_model == 0)
        throw std::invalid_argument("TitansMAL: d_model must be > 0");
    if (d_inner == 0) {
        d_inner_ = d_model;
    } else if (d_inner != d_model) {
        throw std::invalid_argument("TitansMAL: d_inner must equal d_model (v1 constraint)");
    } else {
        d_inner_ = d_inner;
    }
    if (seg_len != 0) seg_len_ = seg_len;

    // Initialize W_qkv (Xavier-like)
    std::mt19937 rng(0xCAFE0003u);  // distinct from MAC (0xCAFE0001) and MAG (0xCAFE0002)
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt((double)d_model));
    auto& W = W_qkv_.weights;
    for (size_t i = 0; i < W.rows; ++i)
        for (size_t j = 0; j < W.cols; ++j) W(i, j) = nd(rng);
    std::fill(W_qkv_.bias.data.begin(), W_qkv_.bias.data.end(), 0.0);

    // Initialize W_alpha
    std::normal_distribution<double> nd_a(0.0, 1.0 / std::sqrt((double)(d_model + 1)));
    auto& Wa = W_alpha_.weights;
    for (size_t i = 0; i < Wa.rows; ++i)
        for (size_t j = 0; j < Wa.cols; ++j) Wa(i, j) = 0.1 * nd_a(rng);
    std::fill(W_alpha_.bias.data.begin(), W_alpha_.bias.data.end(), 0.0);

    // Initialize W_p (MAL-specific) — Xavier init
    auto& Wp = W_p_.weights;
    for (size_t i = 0; i < Wp.rows; ++i)
        for (size_t j = 0; j < Wp.cols; ++j) Wp(i, j) = nd(rng);
    std::fill(W_p_.bias.data.begin(), W_p_.bias.data.end(), 0.0);

    // M_ initialized to zero
    M_ = Tensor(d_model, d_model);
    std::fill(M_.data.begin(), M_.data.end(), 0.0);

    grad_W_qkv_w_ = Tensor(3 * d_model, d_model);
    grad_W_qkv_b_ = Tensor(1, 3 * d_model);
    grad_W_alpha_w_ = Tensor(1, d_model + 1);
    grad_W_alpha_b_ = Tensor(1, 1);
    grad_W_p_w_ = Tensor(d_model, d_model);
    grad_W_p_b_ = Tensor(1, d_model);
    grad_M_ = Tensor(d_model, d_model);
}

// Forward + backward: throw "not implemented" for now (covered in Tasks 2-4)
Tensor TitansMAL::forward(const Tensor&) {
    throw std::runtime_error("TitansMAL::forward not yet implemented");
}
Tensor TitansMAL::backward(const Tensor&, double) {
    throw std::runtime_error("TitansMAL::backward not yet implemented");
}
void TitansMAL::update_weights(double) {
    throw std::runtime_error("TitansMAL::update_weights not yet implemented");
}
void TitansMAL::zero_grad() {
    std::fill(W_qkv_.grad_weights.data.begin(), W_qkv_.grad_weights.data.end(), 0.0);
    std::fill(W_qkv_.grad_bias.data.begin(), W_qkv_.grad_bias.data.end(), 0.0);
    std::fill(W_alpha_.grad_weights.data.begin(), W_alpha_.grad_weights.data.end(), 0.0);
    std::fill(W_alpha_.grad_bias.data.begin(), W_alpha_.grad_bias.data.end(), 0.0);
    std::fill(W_p_.grad_weights.data.begin(), W_p_.grad_weights.data.end(), 0.0);
    std::fill(W_p_.grad_bias.data.begin(), W_p_.grad_bias.data.end(), 0.0);
    std::fill(grad_W_qkv_w_.data.begin(), grad_W_qkv_w_.data.end(), 0.0);
    std::fill(grad_W_qkv_b_.data.begin(), grad_W_qkv_b_.data.end(), 0.0);
    std::fill(grad_W_alpha_w_.data.begin(), grad_W_alpha_w_.data.end(), 0.0);
    std::fill(grad_W_alpha_b_.data.begin(), grad_W_alpha_b_.data.end(), 0.0);
    std::fill(grad_W_p_w_.data.begin(), grad_W_p_w_.data.end(), 0.0);
    std::fill(grad_W_p_b_.data.begin(), grad_W_p_b_.data.end(), 0.0);
    std::fill(grad_M_.data.begin(), grad_M_.data.end(), 0.0);
}
std::vector<Tensor*> TitansMAL::parameters() {
    return { &W_qkv_.weights, &W_qkv_.bias,
             &W_alpha_.weights, &W_alpha_.bias,
             &W_p_.weights, &W_p_.bias };
}
std::vector<Tensor*> TitansMAL::gradients() {
    return { &grad_W_qkv_w_, &grad_W_qkv_b_,
             &grad_W_alpha_w_, &grad_W_alpha_b_,
             &grad_W_p_w_, &grad_W_p_b_ };
}
void TitansMAL::copy_params_from(const TitansMAL& other) {
    W_qkv_.weights = other.W_qkv_.weights.clone();
    W_qkv_.bias = other.W_qkv_.bias.clone();
    W_alpha_.weights = other.W_alpha_.weights.clone();
    W_alpha_.bias = other.W_alpha_.bias.clone();
    W_p_.weights = other.W_p_.weights.clone();
    W_p_.bias = other.W_p_.bias.clone();
    M_ = other.M_.clone();
}
```

**Step 6: Run test — constructor tests pass. GREEN for Task 1.**

**Step 7: Commit**

```bash
git add include/nn/layers/recurrent/titans_mal.h include/nn/layers/recurrent/titans_mal.cpp tests/test_titans_mal.cpp
git commit -m "feat(recurrent): Titans MAL (Memory as a Layer) — header, constructor, contracts"
```

---

### Task 2: Forward — shape, finiteness, MAL-specific gating

**Step 1: Add failing forward tests (RED)**

```cpp
// Append to test_titans_mal.cpp main()
// Test 6: forward shape (T=3, d=4)
{
    TitansMAL layer(4, 4);
    Tensor input = rand_tensor(3, 4, 0xC0FFEE);
    Tensor output = layer.forward(input);
    assert(output.rows == 3 && output.cols == 4);
}
// Test 7: forward finite + nonzero
{
    TitansMAL layer(4, 4);
    Tensor input = rand_tensor(3, 4, 0xBEEF);
    Tensor output = layer.forward(input);
    bool finite = true, nonzero = false;
    for (auto v : output.data) {
        if (!std::isfinite(v)) finite = false;
        if (std::abs(v) > 1e-9) nonzero = true;
    }
    assert(finite && nonzero);
}
// Test 8: forward shape preserved for T=6
{
    TitansMAL layer(4, 4);
    Tensor input = rand_tensor(6, 4, 0xDEAD);
    Tensor output = layer.forward(input);
    assert(output.rows == 6 && output.cols == 4);
}
// Test 9: forward with copied params → bit-exact
{
    TitansMAL a(4, 4);
    TitansMAL b(4, 4);
    Tensor input = rand_tensor(3, 4, 0xABCD);
    Tensor y_a = a.forward(input);
    b.copy_params_from(a);
    Tensor y_b = b.forward(input);
    double max_diff = 0.0;
    for (size_t i = 0; i < y_a.data.size(); ++i)
        max_diff = std::max(max_diff, std::abs(y_a.data[i] - y_b.data[i]));
    assert(max_diff < 1e-12);
}
// Test 10: MAL-specific zero-input → zero-output
{
    TitansMAL layer(4, 4);
    Tensor input(3, 4);
    std::fill(input.data.begin(), input.data.end(), 0.0);
    Tensor output = layer.forward(input);
    // y_t = M_t · (p_t ⊙ x_t) = M_t · (p_t ⊙ 0) = 0 for ALL t — gating zero-extinguishes
    double max_abs = 0.0;
    for (auto v : output.data) max_abs = std::max(max_abs, std::abs(v));
    assert(max_abs < 1e-12);
}
```

**Step 2: Run test — it fails (forward throws). Expected.**

**Step 3: Implement forward**

```cpp
Tensor TitansMAL::forward(const Tensor& input) {
    const size_t T = input.rows;
    if (input.cols != d_model_)
        throw std::invalid_argument("TitansMAL: input.cols must equal d_model");

    // 1) Joint projection: qkv = x · W_qkv^T + b_qkv   (k,v only used; q unused)
    Tensor qkv = W_qkv_.forward(input);   // (T, 3*d_model)
    Tensor k_t(T, d_model_);
    Tensor v_t(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            k_t(t, j) = qkv(t, d_model_ + j);
            v_t(t, j) = qkv(t, 2 * d_model_ + j);
        }
    }

    // 2) MAL-specific input gate: p_t = sigmoid(W_p · x_t + b_p)   (T, d_model)
    Tensor p_t_pre = W_p_.forward(input);   // (T, d_model)
    Tensor p_t(T, d_model_);
    for (size_t i = 0; i < p_t_pre.data.size(); ++i) {
        p_t.data[i] = 1.0 / (1.0 + std::exp(-p_t_pre.data[i]));
    }

    // 3) x̃_t = p_t ⊙ x_t                                            (T, d_model)
    Tensor x_tilde(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            x_tilde(t, j) = p_t(t, j) * input(t, j);
        }
    }

    // 4-5) M-update + output (same structure as MAC, but with q_t ← x̃_t)
    Tensor v_norm(T, 1);
    Tensor eta(T, 1);
    Tensor alpha(T, 1);
    Tensor M_t((T + 1) * d_model_, d_model_);
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            M_t(i, j) = M_(i, j);

    Tensor y(T, d_model_);
    for (size_t t = 0; t < T; ++t) {
        // v_norm[t] = ||v_t||_2
        double vnorm = 0.0;
        for (size_t j = 0; j < d_model_; ++j) vnorm += v_t(t, j) * v_t(t, j);
        vnorm = std::sqrt(vnorm);
        v_norm(t, 0) = vnorm;

        // alpha_input = concat[x_t, v_norm[t]] → (1, d_model+1)
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = input(t, j);
        alpha_in(0, d_model_) = vnorm;
        Tensor alpha_pre = W_alpha_.forward(alpha_in);
        double sigma_alpha = 1.0 / (1.0 + std::exp(-alpha_pre(0, 0)));

        // surprise η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
        Tensor mk_pred(d_model_, 1);
        double xnorm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t(t * d_model_ + i, j) * k_t(t, j);
            mk_pred(i, 0) = s;
            xnorm += input(t, i) * input(t, i);
        }
        xnorm = std::sqrt(xnorm);
        double diff_norm = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double d_i = input(t, i) - mk_pred(i, 0);
            diff_norm += d_i * d_i;
        }
        diff_norm = std::sqrt(diff_norm);
        const double eps = 1e-8;
        double eta_t = diff_norm / (xnorm + eps);
        eta(t, 0) = eta_t;
        alpha(t, 0) = sigma_alpha * eta_t;

        // M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
        double a = alpha(t, 0);
        double om_a = 1.0 - a;
        for (size_t i = 0; i < d_model_; ++i) {
            for (size_t j = 0; j < d_model_; ++j) {
                double prev = M_t(t * d_model_ + i, j);
                double outer_vk = v_t(t, i) * k_t(t, j);
                M_t((t + 1) * d_model_ + i, j) = om_a * prev + a * outer_vk;
            }
        }

        // 6) MAL-specific output: y_t = M_t · x̃_t
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += M_t((t + 1) * d_model_ + i, j) * x_tilde(t, j);
            y(t, i) = s;
        }
    }

    last_input_ = input.clone();
    last_k_t_ = k_t;
    last_v_t_ = v_t;
    last_v_norm_ = v_norm;
    last_eta_ = eta;
    last_alpha_ = alpha;
    last_M_t_ = M_t;
    last_p_t_ = p_t;
    last_x_tilde_ = x_tilde;
    last_y_t_ = y.clone();
    last_qkv_ = qkv;
    return y;
}
```

**Step 4: Run test — all forward tests pass. GREEN for Task 2.**

**Step 5: Commit**

```bash
git add include/nn/layers/recurrent/titans_mal.cpp tests/test_titans_mal.cpp
git commit -m "feat(recurrent): Titans MAL — forward pass + 5-tensor state caching"
```

---

### Task 3: Backward — input gradient + parameter gradients (FD check)

**Step 1: Add failing backward tests (RED)**

The backward must:
- Return `dx_t` correctly via two paths:
  1. The M-chain: `dy_t → dx̃_t (via M_t · x̃_t) → dx̃_t → dx_t (via x̃_t = p_t ⊙ x_t)`
  2. The W_p chain: `dx̃_t → dp_t (sigmoid) → dx_t (via p_t = σ(W_p · x_t + b_p))`
- Compute all 7 parameter gradients: W_qkv.weights/bias (k,v slices only — q unused), W_alpha.weights/bias, W_p.weights/bias, M
- M gradient via the surprise-weighted-momentum chain (identical to MAC)

```cpp
// Tests 11-15: input grad + parameter grad FD checks
// (Full code analogous to Tests 7-9 of test_titans_mag.cpp — same centered-FD pattern,
// loss = 0.5 * sum((y - target)^2), rel_err = |ana - num| / max(1e-12, max(|ana|, |num|)))
// Param IDs checked: 0 (W_qkv.weights), 2 (W_alpha.weights), 4 (W_p.weights), 5 (W_p.bias)
// Bias IDs: 1 (W_qkv.bias), 3 (W_alpha.bias)
```

**Step 2: Run test — fails (backward throws). Expected.**

**Step 3: Implement backward**

```cpp
Tensor TitansMAL::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    const size_t T = last_input_.rows;

    Tensor dk_t(T, d_model_);
    Tensor dv_t(T, d_model_);
    Tensor dx_t(T, d_model_);
    std::fill(dk_t.data.begin(), dk_t.data.end(), 0.0);
    std::fill(dv_t.data.begin(), dv_t.data.end(), 0.0);
    std::fill(dx_t.data.begin(), dx_t.data.end(), 0.0);

    // dL/dx̃_t (the q-equivalent in MAC's formulation): accumulated via M_t · x̃_t chain
    // AND via the p_t ⊙ x_t chain (the gate chain feeds dL/dx̃ into dL/dx)
    Tensor dx_tilde(T, d_model_);
    std::fill(dx_tilde.data.begin(), dx_tilde.data.end(), 0.0);

    Tensor dM_carrier(d_model_, d_model_);
    std::fill(dM_carrier.data.begin(), dM_carrier.data.end(), 0.0);

    for (int t_signed = (int)T - 1; t_signed >= 0; --t_signed) {
        size_t t = (size_t)t_signed;
        double a = last_alpha_(t, 0);
        double om_a = 1.0 - a;
        double eta_t = last_eta_(t, 0);

        // ----- Output chain: y_t = M_t · x̃_t → dy_t -----
        // dL/dx̃_t[i] = sum_j dL/dy_t[j] · M_t[j, i]
        // dL/dM_t[j, i] += dL/dy_t[j] · x̃_t[i]
        Tensor dM_post(d_model_, d_model_);
        std::fill(dM_post.data.begin(), dM_post.data.end(), 0.0);
        for (size_t j = 0; j < d_model_; ++j) {
            double dy_j = grad_output(t, j);
            for (size_t i = 0; i < d_model_; ++i) {
                dx_tilde(t, i) += dy_j * last_M_t_((t + 1) * d_model_ + j, i);
                dM_post(j, i) += dy_j * last_x_tilde_(t, i);
            }
        }

        // Add the M-update carrier from later tokens
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_post(i, j) += dM_carrier(i, j);

        // dα from M_t update: α_t contributes v�k - M_{t-1} to M_t
        double dalpha = 0.0;
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dalpha += dM_post(i, j) * (last_v_t_(t, i) * last_k_t_(t, j)
                                            - last_M_t_(t * d_model_ + i, j));

        // dL/dv_t and dL/dk_t from the M_t update
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                double c = dM_post(i, j) * a;
                dv_t(t, i) += c * last_k_t_(t, j);
                dk_t(t, j) += c * last_v_t_(t, i);
            }

        // dM_carrier → next iteration: (1-α) · dM_post
        Tensor dM_new_carrier(d_model_, d_model_);
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_new_carrier(i, j) = om_a * dM_post(i, j);

        // ----- Surprise chain (identical to MAC) -----
        Tensor mk(d_model_, 1);
        Tensor diff(d_model_, 1);
        double N_d = 0.0, N_x = 0.0;
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += last_M_t_(t * d_model_ + i, j) * last_k_t_(t, j);
            mk(i, 0) = s;
            diff(i, 0) = last_input_(t, i) - s;
            N_d += diff(i, 0) * diff(i, 0);
            N_x += last_input_(t, i) * last_input_(t, i);
        }
        N_d = std::sqrt(N_d);
        N_x = std::sqrt(N_x);
        const double eps = 1e-8;

        // W_alpha gradient: dα = dz · σ(1-σ) · α_input
        Tensor alpha_in(1, d_model_ + 1);
        for (size_t j = 0; j < d_model_; ++j) alpha_in(0, j) = last_input_(t, j);
        alpha_in(0, d_model_) = last_v_norm_(t, 0);
        double z_pre = W_alpha_.bias(0, 0);
        for (size_t j = 0; j < d_model_ + 1; ++j) z_pre += W_alpha_.weights(0, j) * alpha_in(0, j);
        double sigma_pre = 1.0 / (1.0 + std::exp(-z_pre));
        double sigma_pre_deriv = sigma_pre * (1.0 - sigma_pre);
        double deta = dalpha * sigma_pre;
        double dz_pre = dalpha * eta_t * sigma_pre_deriv;

        for (size_t j = 0; j < d_model_ + 1; ++j) {
            grad_W_alpha_w_(0, j) += dz_pre * alpha_in(0, j);
        }
        grad_W_alpha_b_(0, 0) += dz_pre;

        // dα_in → dx, dvnorm (the input-side contributions to alpha_in)
        for (size_t j = 0; j < d_model_; ++j) {
            dx_t(t, j) += dz_pre * W_alpha_.weights(0, j);
        }
        if (last_v_norm_(t, 0) > eps) {
            double dvnorm = dz_pre * W_alpha_.weights(0, d_model_);
            for (size_t i = 0; i < d_model_; ++i) {
                dv_t(t, i) += dvnorm * last_v_t_(t, i) / last_v_norm_(t, 0);
            }
        }

        // dL/dη_t = deta. Routes to dx, dk, dM_{t-1}.
        if (N_x > eps && N_d > eps) {
            double inv_Nx = 1.0 / N_x;
            double inv_Nd = 1.0 / N_d;
            double inv_Nxpe = 1.0 / (N_x + eps);
            for (size_t i = 0; i < d_model_; ++i) {
                double deta_dx = deta * (diff(i, 0) * inv_Nd - eta_t * last_input_(t, i) * inv_Nx) * inv_Nxpe;
                dx_t(t, i) += deta_dx;
                for (size_t j = 0; j < d_model_; ++j) {
                    dM_new_carrier(i, j) += deta * (-inv_Nxpe * inv_Nd * diff(i, 0) * last_k_t_(t, j));
                    dk_t(t, j) += deta * (-inv_Nxpe * inv_Nd * last_M_t_(t * d_model_ + i, j) * diff(i, 0));
                }
            }
        }

        // dL/dW_qkv (k,v slices only — q unused) and dL/dW_qkv.bias
        for (size_t i = 0; i < d_model_; ++i) {
            grad_W_qkv_b_(0, d_model_ + i) += dk_t(t, i);
            grad_W_qkv_b_(0, 2 * d_model_ + i) += dv_t(t, i);
        }
        for (size_t j = 0; j < 2 * d_model_; ++j) {
            double dqkv_tj = (j < d_model_) ? dk_t(t, j) : dv_t(t, j - d_model_);
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_qkv_w_(d_model_ + j, k) += dqkv_tj * last_input_(t, k);
            }
        }

        // ----- MAL-specific: input gate chain -----
        // x̃_t = p_t ⊙ x_t. We have dx_tilde accumulated above.
        //   dx[i] += dx_tilde[i] · p_t[i]                        (the ⊙ path)
        //   dp_t[i] += dx_tilde[i] · x_t[i]                     (the gate derivative)
        Tensor dp_t(d_model_);
        for (size_t i = 0; i < d_model_; ++i) {
            dx_t(t, i) += dx_tilde(t, i) * last_p_t_(t, i);
            dp_t(i) = dx_tilde(t, i) * last_input_(t, i);
        }
        // p_t = sigmoid(W_p · x_t + b_p). sigmoid'(z) = p(1-p).
        //   dW_p.weights[i, k] += dp_t[i] · p_t(1-p_t) · x_t[k]
        //   dW_p.bias[i]        += dp_t[i] · p_t(1-p_t)
        //   dx_t[k]             += dp_t[i] · p_t(1-p_t) · W_p.weights[i, k]
        for (size_t i = 0; i < d_model_; ++i) {
            double p_i = last_p_t_(t, i);
            double p_deriv = p_i * (1.0 - p_i);
            double dp_s = dp_t(i) * p_deriv;
            grad_W_p_b_(0, i) += dp_s;
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_p_w_(i, k) += dp_s * last_input_(t, k);
                dx_t(t, k) += dp_s * W_p_.weights(i, k);
            }
        }

        // Update dM_carrier for the NEXT iteration (token t-1)
        for (size_t i = 0; i < d_model_; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                dM_carrier(i, j) = dM_new_carrier(i, j);
    }

    // After the loop: grad_M_ = dM_carrier (dL/dM_initial)
    for (size_t i = 0; i < d_model_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_M_(i, j) = dM_carrier(i, j);

    // dL/dx_t from W_qkv (k,v slices only — q unused) — same as MAG
    for (size_t t = 0; t < T; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < d_model_; ++j) s += dk_t(t, j) * W_qkv_.weights(d_model_ + j, k);
            for (size_t j = 0; j < d_model_; ++j) s += dv_t(t, j) * W_qkv_.weights(2 * d_model_ + j, k);
            dx_t(t, k) += s;
        }
    }

    return dx_t;
}
```

**Step 4: Run test — input gradient FD rel_err < 1e-2, parameter grads < 5e-2. GREEN for Task 3.**

**Step 5: Commit**

```bash
git add include/nn/layers/recurrent/titans_mal.cpp tests/test_titans_mal.cpp
git commit -m "feat(recurrent): Titans MAL — full analytical backward + FD gradient verification"
```

---

### Task 4: Update weights + contracts + training

**Step 1: Add failing tests (RED)**

```cpp
// Test 16: zero_grad clears all 7 gradients (6 param grads + grad_M_)
// Test 17: update_weights moves all 7 parameters (lr=0.01)
// Test 18: training reduces loss over 50 SGD steps (lr=0.01)
// Test 19: longer sequence (T=6) input grad FD rel_err < 1e-2
```

**Step 2: Implement update_weights (already partially stubbed)**

```cpp
void TitansMAL::update_weights(double learning_rate) {
    W_qkv_.weights   = W_qkv_.weights   - grad_W_qkv_w_   * learning_rate;
    W_qkv_.bias      = W_qkv_.bias      - grad_W_qkv_b_   * learning_rate;
    W_alpha_.weights = W_alpha_.weights - grad_W_alpha_w_ * learning_rate;
    W_alpha_.bias    = W_alpha_.bias    - grad_W_alpha_b_ * learning_rate;
    W_p_.weights     = W_p_.weights     - grad_W_p_w_     * learning_rate;
    W_p_.bias        = W_p_.bias        - grad_W_p_b_     * learning_rate;
    M_ = M_ - grad_M_ * learning_rate;
}
```

**Step 3: Run test — all pass. GREEN for Task 4.**

**Step 4: Commit**

```bash
git add include/nn/layers/recurrent/titans_mal.cpp tests/test_titans_mal.cpp
git commit -m "feat(recurrent): Titans MAL — update_weights, training, T=6 verification"
```

---

### Task 5: MAL-specific property tests + mutation tests

**Step 1: Add tests (RED for mutation tests, GREEN for properties)**

```cpp
// Test 20: W_p random init → measurably different forward output
// Test 21: mutation — perturb W_p.weights[0][0] → forward changes
// Test 22: mutation — perturbing k-slice of W_qkv changes forward (q-slice perturbation has NO effect)
```

**Step 2: Implement — these tests just exercise existing forward/backward. GREEN immediately.**

**Step 3: Commit**

```bash
git add tests/test_titans_mal.cpp
git commit -m "test(recurrent): Titans MAL — mutation tests for W_p and q-slice no-op"
```

---

### Task 6: Wire into umbrella + Makefile

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/recurrent/titans_mal.h"` after `titans_mag.h`
- Modify: `Makefile` — add `build/test_titans_mal` rule, add to `tests:` deps line, add `=== Running Titans MAL Tests ===` to `run_tests:`

**Step 1: Make edits**

In `include/nn/nn.h` after line 115 (after `titans_mag.h`):
```cpp
#include "layers/recurrent/titans_mal.h"
```

In `Makefile` after the `test_titans_mag` rule:
```makefile
$(BUILD_DIR)/test_titans_mal: $(LIB_OBJS) $(BUILD_DIR)/test_titans_mal.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
```

Add `$(BUILD_DIR)/test_titans_mal` to the `tests:` deps line.

Add to `run_tests:` target:
```makefile
@echo "=== Running Titans MAL Tests ===" && ./$(BUILD_DIR)/test_titans_mal
```

**Step 2: Build the umbrella standalone + run the new test**

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
make build/test_titans_mal && ./build/test_titans_mal
```

Expected: umbrella compiles clean, new test passes.

**Step 3: Run full test suite — confirm no regressions**

```bash
make tests 2>&1 | tail -30
```

**Step 4: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore(recurrent): register Titans MAL in umbrella + Makefile"
```

---

### Task 7: Update EXPANSION_QUEUE.md

**File:** `EXPANSION_QUEUE.md` — move TitansMAL from `## Ideas` to `## Done` with a one-line summary of test results.

**Step 1: Edit file**

**Step 2: Commit**

```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark Titans MAL as Done in EXPANSION_QUEUE"
```

---

## Key Pitfalls

1. **The `W_p` input-gate chain is the MAL-specific contribution.** Without it (i.e., if you skip the `dp_t → dW_p → dx` block), the input grad FD reports rel_err > 1.0 because the gradient from `y_t = M_t · (p_t ⊙ x_t)` back to `x_t` has two paths: (a) through the `⊙` (`dx[i] += dx_tilde[i] · p_t[i]`), (b) through `p_t = σ(W_p · x_t + b_p)` (sigmoid chain back to `x_t`). If only path (a) is implemented, FD will catch it.

2. **The q-slice of `W_qkv` is unused in MAL's output.** Just like MAG, the query projection does NOT enter `y_t = M_t · (p_t ⊙ x_t)`. The q-slice of W_qkv's gradient is exactly zero. **Mutation test:** perturbing `W_qkv.weights[0..d-1, ...]` should NOT change forward output.

3. **`last_M_t_` is indexed `[t*d_model + i, j]` for `M_t` (after token t).** Off-by-one error here is the classic M-update chain bug — make sure `M_t[t * d_model + i, j]` is `M_{t-1}` (the prior state used to compute `M_t`).

4. **`grad_M_` is set after the per-token loop, not inside it.** After the loop, `dM_carrier` holds `dL/dM_0 = dL/dM_` (the persistent memory initial state). Copy it into `grad_M_` exactly once at the end.

5. **W_p gradient direction.** `W_p.weights[i, k]` is row-major: `(d_model, d_model)` shape, where `i` is the output index (matches `p_t[i]`) and `k` is the input index (matches `x_t[k]`). The gradient accumulates as `grad_W_p_w_(i, k) += dp_t(i) · p(1-p) · x_t(k)`. Off-by-one on these indices is the standard "wrong orientation" bug for sigmoid-projection layers.

6. **FD vacuity trap (single-pass loops).** The `for (int t = T-1; t >= 0; --t)` loop runs T times — fine. But the W_p sigmoid chain is **per-token** and runs in the same outer loop, so mutation testing (drop a term in the sigmoid chain) must produce a measurable FD vs analytical diff. Verify with a mutation test that zeros out `p_deriv` and confirms input-grad rel_err explodes.

7. **`get_weights()` returns `M_` (the persistent memory), NOT any of the 6 learnable params.** This matches MAC/MAG convention and lets the optimizer iterate all learnables via `parameters()` while `M_` is a separate state surface. Confirmed via the existing `get_weights/get_gradients` test contract.

8. **The dα_in → dx contribution (from W_alpha gradient).** It's `dx_t[j] += dz_pre · W_alpha.weights[0, j]` for j in [0, d_model). Easy to forget — without it, input grad FD reports ~0.4 rel_err.

## Verification Checklist

- [ ] Constructor: 3 invalid inputs throw, valid constructs
- [ ] parameters()/gradients() return 6 tensors each, all correct shapes (no M in either — same as MAC/MAG)
- [ ] Forward shape (T=3, d=4) and (T=6, d=4); forward finite; forward nonzero
- [ ] Forward bit-exact with copied params (max_diff=0)
- [ ] **Zero-input → zero-output bit-exact** (the gating property — `p ⊙ 0 = 0`)
- [ ] Persistent M init=0 + nonzero input → forward uses the post-update M, not 0 (nonzero output)
- [ ] Random W_p → forward measurably different
- [ ] **Input gradient FD rel_err < 1e-2**
- [ ] **All 6 parameter gradient FD rel_err < 5e-2** (W_qkv.weights, W_alpha.weights, W_p.weights, M, plus biases)
- [ ] **T=6 input gradient FD rel_err < 1e-2**
- [ ] zero_grad clears all 7 gradient buffers
- [ ] update_weights moves all 7 parameters
- [ ] Training reduces loss over 50 SGD steps
- [ ] Mutation: perturbing k-slice of W_qkv → forward changes
- [ ] Mutation: perturbing q-slice of W_qkv → forward unchanged (q unused in MAL output)
- [ ] Mutation: perturbing W_p.weights[0,0] → forward changes measurably
- [ ] Umbrella header compiles standalone: `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
- [ ] Full test suite passes (`make tests`)

## Reference

- Paper: Behrouz, Zhong, Mirrokni — "Titans: Learning to Memorize at Test Time", https://arxiv.org/abs/2501.00663
  - §3.2 (Eqs. 9-10): the neural memory M update rule (shared with MAC/MAG/MAL)
  - §4.3 (Eqs. 29-31): the MAL architecture (input gate `p_t`, `y = M(x̃)`, downstream attention)
- Existing repo: `include/nn/layers/recurrent/titans_mac.{h,cpp}`, `titans_mag.{h,cpp}` — mirror style exactly.
