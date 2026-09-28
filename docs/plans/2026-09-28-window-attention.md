# Window Attention (Swin-style) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement multi-head self-attention restricted to local M×M windows with a learnable relative position bias — the core sequence-mixing primitive of Swin Transformer (Liu et al. ICCV 2021 best paper).

**Architecture:** A `WindowAttention(d_model, num_heads, window_size, bias_type="relative")` class that (a) partitions a flat `(H*W, d_model)` token grid into `⌈H/M⌉·⌈W/M⌉` non-overlapping M×M windows, (b) runs standard multi-head QKV attention **independently within each window** so attention complexity is O(N·M²) instead of O(N²), (c) adds a learnable bias `B ∈ R^{(2M-1)²}` indexed by relative-position pair (paper §3.2, relative position bias).

**Tech Stack:** Plain C++17 in the existing `nn` umbrella, Tensor ops from `core/tensor.h`, registered next to the other attention variants in `include/nn/layers/attention/window_attention.{h,cpp}`.

---

## Background and conventions

The repo uses a `Layer` base class with `forward(Tensor) → Tensor`, `backward(grad_output, lr) → grad_input`, `parameters()`, `gradients()`, `update_weights(lr)`, `zero_grad()`, `get_weights()`, `get_gradients()`. Each attention variant follows the same shape: 4 projection matrices `(d_model, d_model)` for W_q / W_k / W_v / W_o plus optional per-head output bias `(1, d_model)`, Xavier-uniform init (factor 1/sqrt(d_model)), `Tensor::random(d_model, d_model, 0.02)` style.

For Swin-style window attention specifically:

- Input: a flat token grid of shape `(N, d_model)` where `N = H·W` for some pre-batched `H, W`. Layer does NOT include patch embedding — caller provides `(N, d_model)`.
- Output: `(N, d_model)` (same shape, attention is a per-window mixer).
- Window size M (default M=7). Pad H, W up to the next multiple of M with `reflect`-style zero-token padding if H/M or W/M is non-integer (paper §4.2: "Padding" is the standard implementation; for v1 we use **zero-pad** because reflected/cyclic padding adds API surface without changing the gradient check — see "deliberate deviations" below).
- For each of `num_windows = (H_padded/M)·(W_padded/M)` windows:
  - slice a `(M·M, d_model)` window from the input
  - project QKV with the layer's shared `W_q, W_k, W_v`
  - reshape to `(num_heads, M·M, head_dim)`
  - compute `scores[h, i, j] = Q[h,i,:]·K[h,j,:] / sqrt(head_dim) + B[(i-j)]` (B is **per-head**, paper §3.2, gives `B ∈ R^{num_heads, (2M-1)²}`). The `(i-j)` index maps a 2-D relative offset `(-(M-1)..(M-1)) × (-(M-1)..(M-1))` to a flat offset `(M-1+dr)·(2M-1) + (M-1+dc)`.
  - softmax row-wise within the window
  - weighted sum of V → `(num_heads, M·M, head_dim)`
  - reshape back to `(M·M, d_model)`
  - apply output projection `W_o` and optional output bias
- Aggregate window outputs back into the (H_padded, W_padded, d_model) tensor, slice off padding.

**Deliberate deviations from the paper** (recorded in the header's deviation table):

1. **No shifted-window partitioning in v1.** Swin's §3.2 "Shifted window" uses a `⌊M/2⌋` pixel cyclic shift before the second block and a masked attention to keep cross-window information local. The shift trick is **a separate layer** (`ShiftedWindowAttention`) or an option flag on `WindowAttention`. v1 ships ONLY the non-shifted window attention + relative position bias. The shifted variant is a future-TODO (one-line forward call with a wrap-around shift and a precomputed attention mask); the relative position bias table is the SAME for shifted and non-shifted windows because relative coords are invariant under cyclic shift.
2. **Zero-padding instead of reflection padding** for H/W not divisible by M. Padding is bookkeeping only — it does NOT participate in the test fixture (tests use H=W=8, M=4 → 4 windows exactly; or H=W=4, M=2). The relative position bias for padded tokens is computed normally; the padding token's contribution is masked by `attention_mask = 0` for the padded rows during softmax (paper convention).
3. **`qkv_bias=False` (default)** — the canonical Swin uses a QKV bias vector of shape `(3, d_model)`. v1 omits it for symmetry with the existing attention variants in this repo (all of which use bias-free QKV projections and a `b_o` output bias). This is a 1-line addition later if needed.
4. **No `attn_drop` / `proj_drop`** — the repo has no dropout primitive for the attention variants (the closest is `SpatialDropout` in the `utility` folder but no attention-drop convention).

---

## Files

- Create: `include/nn/layers/attention/window_attention.h`
- Create: `include/nn/layers/attention/window_attention.cpp`
- Create: `tests/test_window_attention.cpp`
- Modify: `include/nn/nn.h` (add `#include "nn/layers/attention/window_attention.h"` in alphabetical position, after `wire.h` if present, else near the other attention includes)
- Modify: `Makefile` (add `$(BUILD_DIR)/test_window_attention` rule, append dep to `tests:` target, add `=== Running Window Attention Tests ===` echo in `run_tests`)

---

## Task 1: header skeleton + Layer subclass declarations

**Files:** Create `include/nn/layers/attention/window_attention.h`

**Step 1: Write the header**

```cpp
#ifndef WINDOW_ATTENTION_H
#define WINDOW_ATTENTION_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Window Attention — Liu et al. 2021, "Swin Transformer: Hierarchical Vision
// Transformer using Shifted Windows" (https://arxiv.org/abs/2103.14030,
// ICCV 2021 Best Paper)
//
// The Swin block's local self-attention primitive. For a (H, W) grid of
// tokens, partition into ⌈H/M⌉·⌈W/M⌉ non-overlapping M×M windows, run
// standard multi-head attention WITHIN each window, and add a learnable
// relative position bias B ∈ R^{num_heads, (2M-1)²} indexed by the
// pair-wise relative offset between tokens in the same window.
//
// Complexity per window is O(M²·d_model); the per-image cost is
// O(N·M·d_model) instead of O(N²·d_model) for vanilla attention, where
// N = H·W. With M=7 (paper default) and H=W=224, that's a ~32× reduction.
//
// Conventions:
//   * Input:  (N, d_model) — flat token grid for one example, with
//             H_padded * W_padded = N and H_padded % M == 0, W_padded % M == 0.
//             The caller is responsible for H_padded, W_padded being
//             pre-multiples of M; the layer validates and throws otherwise.
//             (We expose H_pad_, W_pad_ accessors for tests.)
//   * Output: (N, d_model)
//   * d_model must be evenly divisible by num_heads
//   * bias_type controls the relative-position-bias table:
//       "relative"  (default) — relative position bias B, the paper's choice
//       "none"               — no bias (vanilla window attention, no RPB)
//   * QKV projections are bias-free; output projection has a per-head
//     output bias b_o ∈ R^{1, d_model}. Matches the other attention
//     variants in this repo.
//
// Classes:
//   WindowAttention       — the local-window multi-head self-attention layer
// ============================================================================

class WindowAttention : public Layer {
public:
    WindowAttention(size_t d_model, size_t num_heads, size_t window_size,
                    size_t H_pad, size_t W_pad,
                    const std::string& bias_type = "relative");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "WindowAttention"; }

    // Accessors
    size_t d_model()     const { return d_model_; }
    size_t num_heads()   const { return num_heads_; }
    size_t head_dim()    const { return head_dim_; }
    size_t window_size() const { return window_size_; }
    size_t H_pad()       const { return H_pad_; }
    size_t W_pad()       const { return W_pad_; }
    size_t num_windows() const { return num_windows_; }
    const std::string& bias_type() const { return bias_type_; }

    // Public parameter tensors (test access + mutation)
    Tensor W_q, W_k, W_v, W_o;                  // (d_model, d_model) each
    Tensor b_o;                                  // (1, d_model)
    Tensor relative_position_bias_;              // (num_heads, (2M-1)²)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_o;
    Tensor grad_relative_position_bias_;

private:
    size_t d_model_, num_heads_, head_dim_;
    size_t window_size_, H_pad_, W_pad_, num_windows_;
    size_t M_, M2_;                              // window_size_, M_*M_
    size_t rel_bias_size_;                       // (2M-1)²
    std::string bias_type_;
    double scale_;

    // BPTT cache
    Tensor last_input_;                          // (N, d_model)
    Tensor last_q_, last_k_, last_v_;            // per-window flattened (num_windows*M², d_model)
    Tensor last_attn_;                           // (num_windows, num_heads, M², M²)
    Tensor last_head_out_;                       // (num_windows*M², d_model)
    Tensor last_out_pre_;                        // (num_windows*M², d_model) — pre-clamp/pad-mask

    // For the backward pass: the per-window index buffer is rebuilt each
    // forward (deterministic); cache it so backward doesn't recompute.
    std::vector<size_t> window_indices_;         // (num_windows, M²) flat indices into N
};

#endif
```

**Step 2: Verify the header compiles standalone**

Run:
```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -x c++ -fsyntax-only - <<'EOF'
#include "nn/layers/attention/window_attention.h"
int main() { return 0; }
EOF
```

Expected: clean exit, no errors. (The class body uses only types declared above — Tensor, Layer, vector, cmath.)

**Step 3: Commit**

```bash
git add include/nn/layers/attention/window_attention.h
git commit -m "feat(attention): WindowAttention header — Swin-style local-window MHA"
```

---

## Task 2: Constructor + accessors + parameter wiring (failing test first)

**Files:**
- Modify: `include/nn/layers/attention/window_attention.cpp` (add constructor + parameter accessors + zero_grad + update_weights)
- Modify: `tests/test_window_attention.cpp` (add Tests 1–6)

**Step 1: Write failing tests for constructor validation and parameter shape**

```cpp
// In tests/test_window_attention.cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include "nn/layers/attention/window_attention.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static void check(const string& name, bool cond, double err = 0.0) {
    if (cond) { ++passed; cout << "  [PASS] " << name << "\n"; }
    else      { ++failed; cout << "  [FAIL] " << name << " (err=" << err << ")\n"; }
}

static double tensor_max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        m = max(m, fabs(a.data[i] - b.data[i]));
    return m;
}

// ---- Test 1: constructor validation -------------------------------------
static void test_constructor_validation() {
    cout << "Test 1: WindowAttention constructor validation\n";
    // (a) d_model=0 throws
    try { WindowAttention wa(0, 2, 4, 4, 4); check("d_model=0 throws", false); }
    catch (const invalid_argument&) { check("d_model=0 throws", true); }
    // (b) num_heads=0 throws
    try { WindowAttention wa(8, 0, 4, 4, 4); check("num_heads=0 throws", false); }
    catch (const invalid_argument&) { check("num_heads=0 throws", true); }
    // (c) d_model not divisible by num_heads throws
    try { WindowAttention wa(8, 3, 4, 4, 4); check("d_model%num_heads!=0 throws", false); }
    catch (const invalid_argument&) { check("d_model%num_heads!=0 throws", true); }
    // (d) window_size=0 throws
    try { WindowAttention wa(8, 2, 0, 4, 4); check("window_size=0 throws", false); }
    catch (const invalid_argument&) { check("window_size=0 throws", true); }
    // (e) H_pad not divisible by window_size throws
    try { WindowAttention wa(8, 2, 4, 5, 4); check("H_pad%M!=0 throws", false); }
    catch (const invalid_argument&) { check("H_pad%M!=0 throws", true); }
    // (f) W_pad not divisible by window_size throws
    try { WindowAttention wa(8, 2, 4, 4, 7); check("W_pad%M!=0 throws", false); }
    catch (const invalid_argument&) { check("W_pad%M!=0 throws", true); }
    // (g) valid construct
    try { WindowAttention wa(8, 2, 4, 8, 8); check("valid constructs", true); }
    catch (...) { check("valid constructs", false); }
}

int main() {
    test_constructor_validation();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
```

**Step 2: Run to verify failure (linker error — class not yet defined)**

Run:
```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa 2>&1 | head -20
```

Expected: link error (undefined reference to `WindowAttention::...`) or compile error on `WindowAttention` constructor since the .cpp doesn't yet define one. We are verifying the test fails *before* writing the impl.

**Step 3: Implement the constructor + parameter wiring in `window_attention.cpp`**

```cpp
#include "window_attention.h"
#include <random>

WindowAttention::WindowAttention(size_t d_model, size_t num_heads,
                                  size_t window_size, size_t H_pad, size_t W_pad,
                                  const std::string& bias_type)
    : d_model_(d_model), num_heads_(num_heads),
      window_size_(window_size), H_pad_(H_pad), W_pad_(W_pad),
      bias_type_(bias_type)
{
    // Validation
    if (d_model_ == 0)
        throw std::invalid_argument("WindowAttention: d_model must be > 0");
    if (num_heads_ == 0)
        throw std::invalid_argument("WindowAttention: num_heads must be > 0");
    if (d_model_ % num_heads_ != 0)
        throw std::invalid_argument("WindowAttention: d_model must be divisible by num_heads");
    if (window_size_ == 0)
        throw std::invalid_argument("WindowAttention: window_size must be > 0");
    if (H_pad_ % window_size_ != 0)
        throw std::invalid_argument("WindowAttention: H_pad must be divisible by window_size");
    if (W_pad_ % window_size_ != 0)
        throw std::invalid_argument("WindowAttention: W_pad must be divisible by window_size");
    if (bias_type_ != "relative" && bias_type_ != "none")
        throw std::invalid_argument("WindowAttention: bias_type must be 'relative' or 'none'");

    head_dim_ = d_model_ / num_heads_;
    M_ = window_size_;
    M2_ = M_ * M_;
    num_windows_ = (H_pad_ / M_) * (W_pad_ / M_);
    rel_bias_size_ = (2 * M_ - 1) * (2 * M_ - 1);
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Xavier-uniform init for Q, K, V, O: scale = sqrt(1/d_model)
    const double proj_scale = std::sqrt(1.0 / static_cast<double>(d_model_));
    W_q = Tensor::random(d_model_, d_model_, proj_scale);
    W_k = Tensor::random(d_model_, d_model_, proj_scale);
    W_v = Tensor::random(d_model_, d_model_, proj_scale);
    W_o = Tensor::random(d_model_, d_model_, proj_scale);
    b_o = Tensor::zeros(1, d_model_);

    // Relative position bias: small init (paper §4.2 "initialize with truncated normal";
    // we use a small uniform scale so FD checks are well-conditioned)
    if (bias_type_ == "relative") {
        relative_position_bias_ = Tensor::zeros(num_heads_, rel_bias_size_);
        // Truncated-normal-style: random with small scale.
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.02);
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t i = 0; i < rel_bias_size_; ++i) {
                double v;
                do { v = dist(rng); } while (std::fabs(v) > 0.04);  // truncated at 2σ
                relative_position_bias_[h][i] = v;
            }
        }
    } else {
        // "none" — still allocate so parameters()/gradients() contract is uniform
        relative_position_bias_ = Tensor::zeros(num_heads_, rel_bias_size_);
    }

    // Allocate gradient tensors (zeros)
    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
    grad_b_o = Tensor::zeros(1, d_model_);
    grad_relative_position_bias_ = Tensor::zeros(num_heads_, rel_bias_size_);
}

std::vector<Tensor*> WindowAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &b_o, &relative_position_bias_};
}

std::vector<Tensor*> WindowAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_b_o, &grad_relative_position_bias_};
}

void WindowAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
    grad_relative_position_bias_.fill(0.0);
}

void WindowAttention::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_k -= grad_W_k * learning_rate;
    W_v -= grad_W_v * learning_rate;
    W_o -= grad_W_o * learning_rate;
    b_o -= grad_b_o * learning_rate;
    if (bias_type_ == "relative") {
        relative_position_bias_ -= grad_relative_position_bias_ * learning_rate;
    }
}
```

**Step 4: Run tests to verify green**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa && /tmp/test_wa
```

Expected: `=== Summary: 6 passed, 0 failed ===`. (Test 1 has 7 sub-checks but they roll up into 1 logical test in the summary count via the helper; or count sub-checks separately — adjust as the helper supports.)

**Step 5: Commit**

```bash
git add include/nn/layers/attention/window_attention.cpp tests/test_window_attention.cpp
git commit -m "feat(attention): WindowAttention constructor + parameter wiring (7-case validation)"
```

---

## Task 3: Forward pass + window partitioning (failing test first)

**Files:**
- Modify: `include/nn/layers/attention/window_attention.cpp` (add `forward`)
- Modify: `tests/test_window_attention.cpp` (add Tests 7–10: forward shape, finiteness, mask property, determinism)

**Step 1: Write failing tests for forward**

Add to `tests/test_window_attention.cpp`:

```cpp
static Tensor make_input(size_t N, size_t d, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Tensor t(N, d);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(rng);
    return t;
}

static double tensor_l2_norm(const Tensor& t) {
    double s = 0;
    for (double v : t.data) s += v * v;
    return std::sqrt(s);
}

// ---- Test 7: forward shape + finiteness --------------------------------
static void test_forward_shape() {
    cout << "Test 7: forward shape and finiteness\n";
    // 4 windows of (2×2) on an (4,4) grid, d=8, H=2
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 1);
    Tensor out = wa.forward(input);
    check("forward shape (16, 8)", out.rows == 16 && out.cols == 8);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
    check("forward finite", finite);
    check("forward nonzero (L2 > 0)", tensor_l2_norm(out) > 1e-6);
}

// ---- Test 8: forward on larger grid (H=8, W=8, M=4, 4 windows, d=16, H=4)
static void test_forward_larger() {
    cout << "Test 8: forward larger (H=W=8, M=4)\n";
    WindowAttention wa(16, 4, 4, 8, 8);
    Tensor input = make_input(64, 16, 2);
    Tensor out = wa.forward(input);
    check("forward shape (64, 16)", out.rows == 64 && out.cols == 16);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) { finite = false; break; }
    check("forward finite", finite);
}

// ---- Test 9: determinism — two consecutive forwards bit-exact ----------
static void test_forward_determinism() {
    cout << "Test 9: forward determinism\n";
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 3);
    Tensor out1 = wa.forward(input);
    Tensor out2 = wa.forward(input);
    check("two consecutive forwards bit-exact", tensor_max_abs_diff(out1, out2) < 1e-12);
}

// ---- Test 10: window isolation — perturbing one window must not change
//       output at unrelated window positions (different spatial neighborhoods)
//   Specifically: with W_q fixed, perturbing window 0's input must change
//   window 0's output, but the magnitude of the change in window 0 must
//   dominate the change in window 1.
static void test_window_isolation() {
    cout << "Test 10: window isolation\n";
    // (H=4, W=4, M=2) → 4 windows of 2x2. Window 0 covers (0,0)-(1,1).
    // Perturbing only window 0's input rows must NOT meaningfully touch
    // window 3's output rows.
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor x0 = make_input(16, 8, 4);
    Tensor out0 = wa.forward(x0);
    // Perturb window 0 (rows 0, 1, 4, 5 in flat (H,W)=(4,4) → (i=0,j=0)=0; (0,1)=1; (1,0)=4; (1,1)=5)
    Tensor x1 = x0.clone();
    for (size_t r : {0u, 1u, 4u, 5u}) {
        for (size_t c = 0; c < 8; ++c) x1[r][c] += 1.0;
    }
    Tensor out1 = wa.forward(x1);
    // Change in window 0 outputs (rows 0..3)
    double d_window0 = 0.0;
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 8; ++c)
            d_window0 += std::fabs(out1[r][c] - out0[r][c]);
    // Change in window 3 outputs (rows 12..15 in flat = (2,0)..(3,1))
    double d_window3 = 0.0;
    for (size_t r = 12; r < 16; ++r)
        for (size_t c = 0; c < 8; ++c)
            d_window3 += std::fabs(out1[r][c] - out0[r][c]);
    check("perturbing window 0 changes window 0 output more than window 3",
          d_window0 > 1e-6 && d_window0 > 10.0 * d_window3);
}
```

Update `main()` to call tests 7–10 and increment passed/failed counts.

**Step 2: Run to verify failure**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa 2>&1 | head -10
```

Expected: link error (undefined reference to `WindowAttention::forward`) OR, if the constructor is not yet validated by tests, the tests fail at compile time on `wa.forward(input)`.

**Step 3: Implement `forward`**

```cpp
Tensor WindowAttention::forward(const Tensor& input) {
    const size_t N = input.rows;
    const size_t expected_N = H_pad_ * W_pad_;
    if (N != expected_N) {
        throw std::invalid_argument(
            "WindowAttention.forward: input.rows must equal H_pad*W_pad");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument(
            "WindowAttention.forward: input.cols must equal d_model");
    }

    last_input_ = input.clone();

    // ---- 1. Partition input into windows ----
    // window_indices_[w * M2 + i] = flat index into input rows for window w, in-window pos i.
    // Token order within a window: row-major within the MxM window (i.e. (r,c) → r*M+c).
    const size_t nH = H_pad_ / M_;
    const size_t nW = W_pad_ / M_;
    if (num_windows_ != nH * nW) {
        throw std::logic_error("WindowAttention: internal window count mismatch");
    }
    window_indices_.assign(num_windows_ * M2_, 0);
    for (size_t wh = 0; wh < nH; ++wh) {
        for (size_t ww = 0; ww < nW; ++ww) {
            const size_t w = wh * nW + ww;
            for (size_t r = 0; r < M_; ++r) {
                for (size_t c = 0; c < M_; ++c) {
                    const size_t in_pos = r * M_ + c;
                    const size_t gi = (wh * M_ + r) * W_pad_ + (ww * M_ + c);
                    window_indices_[w * M2_ + in_pos] = gi;
                }
            }
        }
    }

    // ---- 2. QKV projections (no bias) ----
    // Each window token is independently projected with the same W_q, W_k, W_v.
    Tensor Q(N, d_model_), K(N, d_model_), V(N, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double q = 0.0, k = 0.0, vv = 0.0;
            for (size_t f = 0; f < d_model_; ++f) {
                double x = input[t][f];
                q += x * W_q[j][f];
                k += x * W_k[j][f];
                vv += x * W_v[j][f];
            }
            Q[t][j] = q;
            K[t][j] = k;
            V[t][j] = vv;
        }
    }
    last_q_ = Q;
    last_k_ = K;
    last_v_ = V;

    // ---- 3. Per-window attention ----
    // We accumulate per-window head output into a (num_windows_ * M2_, d_model) tensor,
    // then we will scatter back into the (N, d_model) output shape via the same indices.
    Tensor head_out_flat(num_windows_ * M2_, d_model_);
    Tensor attn_flat(num_windows_, num_heads_, M2_, M2_);
    for (size_t w = 0; w < num_windows_; ++w) {
        for (size_t h = 0; h < num_heads_; ++h) {
            // Slice Q, K, V for this window/head
            const size_t h_off = h * head_dim_;
            // scores: (M2_, M2_)
            Tensor scores(M2_, M2_);
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t ki = 0; ki < M2_; ++ki) {
                    const size_t g_k = window_indices_[w * M2_ + ki];
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        s += Q[g_q][h_off + d] * K[g_k][h_off + d];
                    }
                    s *= scale_;
                    // Add relative position bias
                    if (bias_type_ == "relative") {
                        // Compute (qr, qc) and (kr, kc) within the window
                        const size_t qr = qi / M_;
                        const size_t qc = qi % M_;
                        const size_t kr = ki / M_;
                        const size_t kc = ki % M_;
                        const long dr = (long)kr - (long)qr;     // in [-(M-1), M-1]
                        const long dc = (long)kc - (long)qc;
                        const size_t bias_idx = ((size_t)(dr + (long)(M_ - 1))) * (2 * M_ - 1)
                                              + ((size_t)(dc + (long)(M_ - 1)));
                        s += relative_position_bias_[h][bias_idx];
                    }
                    scores[qi][ki] = s;
                }
            }
            // Row-wise softmax
            Tensor A = scores;  // copy
            for (size_t qi = 0; qi < M2_; ++qi) {
                double maxv = -1e30;
                for (size_t ki = 0; ki < M2_; ++ki)
                    maxv = std::max(maxv, A[qi][ki]);
                double sum = 0.0;
                for (size_t ki = 0; ki < M2_; ++ki) {
                    A[qi][ki] = std::exp(A[qi][ki] - maxv);
                    sum += A[qi][ki];
                }
                if (sum < 1e-30) sum = 1e-30;
                for (size_t ki = 0; ki < M2_; ++ki) A[qi][ki] /= sum;
            }
            // Cache attention for backward
            for (size_t qi = 0; qi < M2_; ++qi)
                for (size_t ki = 0; ki < M2_; ++ki)
                    attn_flat[w][h][qi][ki] = A[qi][ki];
            // head output for this window/head:  A @ V_slice  ∈ R^{M2_ x head_dim_}
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t ki = 0; ki < M2_; ++ki) {
                        const size_t g_k = window_indices_[w * M2_ + ki];
                        s += A[qi][ki] * V[g_k][h_off + d];
                    }
                    head_out_flat[g_q][h_off + d] = s;
                }
            }
        }
    }
    last_attn_ = attn_flat;
    last_head_out_ = head_out_flat;

    // ---- 4. Output projection ----
    Tensor out_flat(num_windows_ * M2_, d_model_);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = b_o[0][j];
            for (size_t k = 0; k < d_model_; ++k) {
                s += head_out_flat[t][k] * W_o[k][j];
            }
            out_flat[t][j] = s;
        }
    }
    last_out_pre_ = out_flat;
    return out_flat;
}
```

NOTE on the 4-D `attn_flat` indexing — `Tensor::operator[]` is 2-D only. We need a helper to access `[w][h][qi][ki]`. Add a helper in the impl:

```cpp
// Helper: 4-D access on a Tensor-shaped (num_windows_, num_heads_, M2_, M2_) tensor.
// We store it as a (num_windows_ * num_heads_ * M2_, M2_) tensor with row-major flat indexing.
Tensor& a4(Tensor& t, size_t w, size_t h, size_t qi) {
    return t[(w * num_heads_ + h) * M2_ + qi];
}
```

Update forward to use `a4(attn_flat, w, h, qi)[ki]` for the softmax output and cache writes. (The cache write becomes:
```cpp
for (size_t qi = 0; qi < M2_; ++qi)
    for (size_t ki = 0; ki < M2_; ++ki)
        a4(attn_flat, w, h, qi)[ki] = A[qi][ki];
```

And `last_attn_` is sized as `(num_windows_ * num_heads_ * M2_, M2_)` with this access pattern — adjust the declaration in the header to match.

**Header adjustment** — modify `last_attn_` from `Tensor (num_windows, num_heads, M², M²)` to `Tensor (num_windows*num_heads*M², M²)` and add the `a4` helper.

**Step 4: Run tests to verify green**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa && /tmp/test_wa
```

Expected: `=== Summary: 14 passed, 0 failed ===` (Test 1's 7 sub-checks + Tests 7, 8, 9, 10 = 11 logical assertions; the helper counts each `check()` call).

**Step 5: Commit**

```bash
git add include/nn/layers/attention/window_attention.{h,cpp} tests/test_window_attention.cpp
git commit -m "feat(attention): WindowAttention forward + window partitioning + RPB"
```

---

## Task 4: Backward pass (failing test first — input + parameter FD checks)

**Files:**
- Modify: `include/nn/layers/attention/window_attention.cpp` (add `backward`)
- Modify: `tests/test_window_attention.cpp` (add Tests 11–16: FD input + FD params)

**Step 1: Write failing FD tests**

Add the standard FD helper + 6 failing FD tests:

```cpp
static double l2_loss_value(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - tgt.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i)
        g.data[i] = out.data[i] - tgt.data[i];
    return g;
}
static Tensor fd_input_grad(WindowAttention& wa, Tensor input, const Tensor& tgt,
                             double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            double lp = l2_loss_value(wa.forward(input), tgt);
            input[i][j] = orig - eps;
            double lm = l2_loss_value(wa.forward(input), tgt);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}
static Tensor fd_param_grad(WindowAttention& wa, size_t pidx, Tensor input,
                             const Tensor& tgt, double eps = 1e-5) {
    Tensor* p = wa.parameters()[pidx];
    Tensor grad(p->rows, p->cols);
    for (size_t i = 0; i < p->rows; ++i) {
        for (size_t j = 0; j < p->cols; ++j) {
            double orig = (*p)[i][j];
            (*p)[i][j] = orig + eps;
            double lp = l2_loss_value(wa.forward(input), tgt);
            (*p)[i][j] = orig - eps;
            double lm = l2_loss_value(wa.forward(input), tgt);
            (*p)[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// ---- Test 11: FD input gradient (H=4, W=4, M=2, d=8, Hq=2) -----------
static void test_fd_input_grad() {
    cout << "Test 11: FD input gradient\n";
    srand(7);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 7);
    Tensor target = make_input(16, 8, 8);
    Tensor out = wa.forward(input);
    Tensor ana = wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor fd = fd_input_grad(wa, input.clone(), target);
    double rel = tensor_rel_err(ana, fd);
    check("FD input grad rel_err < 1e-4", rel < 1e-4, rel);
}

// ---- Test 12: FD W_q gradient ----------------------------------------
static void test_fd_W_q() {
    cout << "Test 12: FD W_q gradient\n";
    srand(7);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 9);
    Tensor target = make_input(16, 8, 10);
    wa.forward(input);
    wa.backward(l2_loss_grad(wa.get_weights(), target), 0.0);  // forward again to get grad
    Tensor* p = wa.parameters()[0];  // W_q
    Tensor* g = wa.gradients()[0];   // grad_W_q
    Tensor fd = fd_param_grad(wa, 0, input.clone(), target);
    double rel = tensor_rel_err(*g, fd);
    check("FD W_q rel_err < 1e-5", rel < 1e-5, rel);
}
```

(Plus Tests 13–16 for W_k, W_v, W_o, b_o, relative_position_bias_.)

**Step 2: Run to verify failure**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa 2>&1 | head -10
```

Expected: link error (undefined reference to `WindowAttention::backward`).

**Step 3: Implement `backward`**

```cpp
Tensor WindowAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (N, d_model)
    if (grad_output.rows != last_out_pre_.rows || grad_output.cols != d_model_)
        throw std::invalid_argument("WindowAttention.backward: grad_output shape mismatch");

    const size_t N = last_out_pre_.rows;

    // ---- 1. Output projection backward ----
    // out_pre = head_out @ W_o + b_o
    // d_head_out: (N, d_model)
    Tensor d_head_out(N, d_model_);
    d_head_out.fill(0.0);
    grad_W_o.fill(0.0);
    grad_b_o.fill(0.0);
    for (size_t t = 0; t < N; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double g = grad_output[t][j];
            grad_b_o[0][j] += g;
            for (size_t k = 0; k < d_model_; ++k) {
                grad_W_o[k][j] += last_head_out_[t][k] * g;
                d_head_out[t][k] += W_o[k][j] * g;
            }
        }
    }

    // ---- 2. Per-window attention backward ----
    // For each (w, h), the local computation is identical to standard MHA
    // for that window. Use the cached window_indices_ to know which rows of
    // last_q_, last_k_, last_v_ correspond to each window.
    Tensor d_Q(N, d_model_), d_K(N, d_model_), d_V(N, d_model_);
    d_Q.fill(0.0); d_K.fill(0.0); d_V.fill(0.0);
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0);
    grad_relative_position_bias_.fill(0.0);

    for (size_t w = 0; w < num_windows_; ++w) {
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t h_off = h * head_dim_;
            // Pull A, Q, K, V slices for this (w, h)
            Tensor A(M2_, M2_);
            for (size_t qi = 0; qi < M2_; ++qi)
                for (size_t ki = 0; ki < M2_; ++ki)
                    A[qi][ki] = a4(last_attn_, w, h, qi)[ki];

            // d_head_out_slice (M2_, head_dim_)
            Tensor d_ho(M2_, head_dim_);
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_ho[qi][d] = d_head_out[g_q][h_off + d];
                }
            }

            // dV_slice (M2_, head_dim_)
            Tensor dVs(M2_, head_dim_);
            dVs.fill(0.0);
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    double s = 0.0;
                    for (size_t ki = 0; ki < M2_; ++ki) {
                        s += A[qi][ki] * d_ho[ki][d];
                    }
                    dVs[qi][d] = s;
                }
            }
            // dA = d_ho @ V_slice^T  → (M2_, M2_)
            Tensor dA(M2_, M2_);
            for (size_t qi = 0; qi < M2_; ++qi) {
                for (size_t ki = 0; ki < M2_; ++ki) {
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim_; ++d) {
                        const size_t g_k = window_indices_[w * M2_ + ki];
                        s += d_ho[qi][d] * last_v_[g_k][h_off + d];
                    }
                    dA[qi][ki] = s;
                }
            }
            // Softmax backward:
            //   dscores = A ⊙ (dA − Σ_k A ⊙ dA)        (row-wise)
            Tensor dscores(M2_, M2_);
            for (size_t qi = 0; qi < M2_; ++qi) {
                double dot = 0.0;
                for (size_t ki = 0; ki < M2_; ++ki) dot += A[qi][ki] * dA[qi][ki];
                for (size_t ki = 0; ki < M2_; ++ki) {
                    dscores[qi][ki] = A[qi][ki] * (dA[qi][ki] - dot);
                }
            }
            // Apply scale
            for (size_t qi = 0; qi < M2_; ++qi)
                for (size_t ki = 0; ki < M2_; ++ki)
                    dscores[qi][ki] *= scale_;
            // Accumulate relative position bias gradient (if applicable)
            if (bias_type_ == "relative") {
                for (size_t qi = 0; qi < M2_; ++qi) {
                    const size_t qr = qi / M_;
                    const size_t qc = qi % M_;
                    for (size_t ki = 0; ki < M2_; ++ki) {
                        const size_t kr = ki / M_;
                        const size_t kc = ki % M_;
                        const long dr = (long)kr - (long)qr;
                        const long dc = (long)kc - (long)qc;
                        const size_t bias_idx = ((size_t)(dr + (long)(M_ - 1))) * (2 * M_ - 1)
                                              + ((size_t)(dc + (long)(M_ - 1)));
                        grad_relative_position_bias_[h][bias_idx] += dscores[qi][ki];
                    }
                }
            }
            // dQ_slice = dscores @ K_slice   (M2_, head_dim_)
            // dK_slice = dscores^T @ Q_slice (M2_, head_dim_)
            Tensor dQs(M2_, head_dim_);
            Tensor dKs(M2_, head_dim_);
            dQs.fill(0.0); dKs.fill(0.0);
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g_q = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    double sQ = 0.0, sK = 0.0;
                    for (size_t ki = 0; ki < M2_; ++ki) {
                        const size_t g_k = window_indices_[w * M2_ + ki];
                        sQ += dscores[qi][ki] * last_k_[g_k][h_off + d];
                        sK += dscores[ki][qi] * last_q_[g_q][h_off + d];
                    }
                    dQs[qi][d] = sQ;
                    dKs[qi][d] = sK;
                }
            }

            // Scatter into d_Q, d_K, d_V (per global token row)
            for (size_t qi = 0; qi < M2_; ++qi) {
                const size_t g = window_indices_[w * M2_ + qi];
                for (size_t d = 0; d < head_dim_; ++d) {
                    d_Q[g][h_off + d] += dQs[qi][d];
                    d_K[g][h_off + d] += dKs[qi][d];
                    d_V[g][h_off + d] += dVs[qi][d];
                }
            }
        }
    }

    // ---- 3. QKV projection backward ----
    // Each of Q = X @ W_q^T  (W_q has shape (d_model, d_model), so W_q^T is the transpose)
    // We use the convention X[t][f] * W_q[j][f] in forward → so:
    //   grad_W_q[j][f] += Σ_t X[t][f] * d_Q[t][j]
    //   grad_input[t][f] += Σ_j d_Q[t][j] * W_q[j][f]
    Tensor grad_input(N, d_model_);
    grad_input.fill(0.0);
    for (size_t t = 0; t < N; ++t) {
        for (size_t f = 0; f < d_model_; ++f) {
            double x = last_input_[t][f];
            for (size_t j = 0; j < d_model_; ++j) {
                grad_W_q[j][f] += x * d_Q[t][j];
                grad_W_k[j][f] += x * d_K[t][j];
                grad_W_v[j][f] += x * d_V[t][j];
                grad_input[t][f] += d_Q[t][j] * W_q[j][f];
                grad_input[t][f] += d_K[t][j] * W_k[j][f];
                grad_input[t][f] += d_V[t][j] * W_v[j][f];
            }
        }
    }

    // Caller is expected to call update_weights(learning_rate) after we return.
    return grad_input;
}
```

**Step 4: Run tests to verify green**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa && /tmp/test_wa
```

Expected: all 6 FD tests pass at rel_err < 1e-5 (input) and < 1e-4 (params). Common failure modes:

- If grad_input is consistently off by a constant factor → W_o shape or bias convention is wrong.
- If grad_W_q is wrong → forward and backward disagree on the QK^T @ W convention; the paper's "Q @ K^T / sqrt(d)" is `Q[h,i,:] · K[h,j,:] / sqrt(d_head)`, NOT `K · Q`.
- If `grad_relative_position_bias_` is exactly 0 everywhere → the `if (bias_type_ == "relative")` guard accidentally zeros the gradient; check it after the loop.

**Step 5: Commit**

```bash
git add include/nn/layers/attention/window_attention.cpp tests/test_window_attention.cpp
git commit -m "feat(attention): WindowAttention backward — full FD-checked analytical chain"
```

---

## Task 5: Mutation tests + end-to-end training + final wiring

**Files:**
- Modify: `tests/test_window_attention.cpp` (add Tests 17–22: mutation tests, training, parameters/gradients contract)

**Step 1: Write mutation tests + training test**

```cpp
// ---- Test 17: mutation — zeroing the relative position bias output →
//       input gradient should still be finite (RPB contributes additively)
//       AND grad_relative_position_bias_ should be nonzero with RPB on.
static void test_rpb_gradient_nonzero() {
    cout << "Test 17: grad_relative_position_bias_ non-zero\n";
    srand(11);
    WindowAttention wa(8, 2, 2, 4, 4, "relative");
    Tensor input = make_input(16, 8, 11);
    Tensor target = make_input(16, 8, 12);
    wa.forward(input);
    wa.backward(l2_loss_grad(wa.get_weights(), target), 0.0);  // placeholder; recompute properly
    // Re-do the forward+backward properly: forward the layer with `input`, then
    // backward the loss grad w.r.t. layer output.
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor& g = wa.grad_relative_position_bias_;
    double gsum = 0;
    for (double v : g.data) gsum += std::fabs(v);
    check("grad_relative_position_bias_ non-zero", gsum > 1e-6);
}

// ---- Test 18: mutation — setting bias_type="none" makes the RPB contribution zero
static void test_no_bias_type() {
    cout << "Test 18: bias_type='none' works\n";
    srand(13);
    WindowAttention wa(8, 2, 2, 4, 4, "none");
    Tensor input = make_input(16, 8, 13);
    Tensor target = make_input(16, 8, 14);
    Tensor out = wa.forward(input);
    Tensor ana = wa.backward(l2_loss_grad(out, target), 0.0);
    Tensor fd = fd_input_grad(wa, input.clone(), target);
    double rel = tensor_rel_err(ana, fd);
    check("FD input grad (no bias_type) rel_err < 1e-4", rel < 1e-4, rel);
}

// ---- Test 19: parameters()/gradients() contract ----------------------
static void test_parameters_gradients_contract() {
    cout << "Test 19: parameters()/gradients() contract\n";
    WindowAttention wa(8, 2, 2, 4, 4);
    auto params = wa.parameters();
    auto grads = wa.gradients();
    check("6 parameters", params.size() == 6);
    check("6 gradients", grads.size() == 6);
    for (size_t i = 0; i < params.size(); ++i) {
        check("param/grad shape match",
              params[i]->rows == grads[i]->rows && params[i]->cols == grads[i]->cols);
    }
    check("W_q is (8,8)", params[0]->rows == 8 && params[0]->cols == 8);
    check("relative_position_bias_ is (2, 9)",
          params[5]->rows == 2 && params[5]->cols == 9);  // (2M-1)² = 3² = 9
}

// ---- Test 20: zero_grad clears all 6 gradients -----------------------
static void test_zero_grad_clears() {
    cout << "Test 20: zero_grad clears all gradients\n";
    srand(15);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 15);
    Tensor target = make_input(16, 8, 16);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    wa.zero_grad();
    for (auto* g : wa.gradients()) {
        for (double v : g->data) {
            if (std::fabs(v) > 0) { check("zero_grad clears", false); return; }
        }
    }
    check("zero_grad clears", true);
}

// ---- Test 21: update_weights moves parameters ------------------------
static void test_update_weights_moves() {
    cout << "Test 21: update_weights moves parameters\n";
    srand(17);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 17);
    Tensor target = make_input(16, 8, 18);
    Tensor out = wa.forward(input);
    wa.backward(l2_loss_grad(out, target), 0.0);
    auto params_before = wa.parameters();
    std::vector<Tensor> saved;
    for (auto* p : params_before) saved.push_back(p->clone());
    wa.update_weights(0.05);
    bool moved = false;
    for (size_t i = 0; i < params_before.size(); ++i) {
        if (tensor_max_abs_diff(*params_before[i], saved[i]) > 1e-9) { moved = true; break; }
    }
    check("update_weights(0.05) moves parameters", moved);
}

// ---- Test 22: end-to-end training reduces loss -----------------------
static void test_training_reduces_loss() {
    cout << "Test 22: training reduces loss\n";
    srand(19);
    WindowAttention wa(8, 2, 2, 4, 4);
    Tensor input = make_input(16, 8, 19);
    Tensor target = make_input(16, 8, 20);
    double L0 = l2_loss_value(wa.forward(input), target);
    for (int step = 0; step < 30; ++step) {
        Tensor out = wa.forward(input);
        Tensor grad = l2_loss_grad(out, target);
        wa.backward(grad, 0.0);
        wa.update_weights(0.05);
    }
    double Lf = l2_loss_value(wa.forward(input), target);
    check("loss reduces by > 10%", Lf < 0.9 * L0, Lf / L0);
}
```

Update `main()` to call all 22 tests.

**Step 2: Verify all tests pass**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude tests/test_window_attention.cpp include/nn/layers/attention/window_attention.cpp include/nn/core/tensor.cpp include/nn/core/layer.cpp -o /tmp/test_wa && /tmp/test_wa
```

Expected: `=== Summary: 35+ passed, 0 failed ===`. (Exact count depends on per-`check()` granularity.)

**Step 3: Mutation test (out-of-band; not in the test file)**

Verify the test catches a broken implementation:

1. Zero out the RPB contribution: change `s += relative_position_bias_[h][bias_idx];` → `// s += 0.0;`. Recompile, run. **Expected failures**: Test 11 (FD input grad rel_err jumps from <1e-4 to O(1)), Test 17 (grad_rpb still nonzero, since we still accumulate it). At minimum, Test 11 should fail. Revert.

2. Drop the scale factor: change `s *= scale_;` → `// s *= 1.0;`. Recompile, run. **Expected failures**: Test 11 rel_err jumps to ~0.95 (the classic half/scale fingerprint). Revert.

3. Drop the softmax-normalization: change `for (ki) A[qi][ki] /= sum;` → `// skip`. Recompile, run. **Expected**: many of Test 11–22 fail (the softmax is what makes attention a softmax, not a sum). Revert.

If 2+ of these do NOT fail under mutation, the test suite has vacuous coverage — strengthen it before shipping.

**Step 4: Wire into the umbrella and Makefile**

Add `#include "nn/layers/attention/window_attention.h"` to `include/nn/nn.h` (alphabetical order, after `wire.h` or near the other attention includes).

Add to `Makefile`:
- New rule: `$(BUILD_DIR)/test_window_attention: $(LIB_OBJS) $(BUILD_DIR)/test_window_attention.o` + `$(CXX) $^ -o $@`
- Append `$(BUILD_DIR)/test_window_attention` to the `tests:` target dependency list
- Add `@echo "=== Running Window Attention Tests ===" && ./$(BUILD_DIR)/test_window_attention` to the `run_tests` target (place it after the related attention tests)

Verify umbrella compiles standalone:
```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<'EOF'
#include "nn/nn.h"
int main() { return 0; }
EOF
```

**Step 5: Run the full repo build + the focused suite three times**

```bash
make build/test_window_attention && ./build/test_window_attention
./build/test_window_attention
./build/test_window_attention
```

Expected: 35+/35+ passed, deterministic (identical numerics across runs).

**Step 6: Commit + push**

```bash
git add include/nn/layers/attention/window_attention.{h,cpp} tests/test_window_attention.cpp include/nn/nn.h Makefile docs/plans/2026-09-28-window-attention.md
git commit -m "feat(attention): Window Attention (Swin-style local-window MHA + RPB) — 35+ tests pass"
git push origin master
```

---

## Self-review checklist (run before pushing)

- [ ] All 7 constructor-validation sub-checks throw or construct correctly.
- [ ] Forward produces (N, d_model) for at least one M=2 and one M=4 config.
- [ ] Forward is bit-exact across two consecutive calls.
- [ ] Window isolation test passes (perturbing one window changes that window's output more than others).
- [ ] FD input gradient rel_err < 1e-4 with random init.
- [ ] FD W_q, W_k, W_v, W_o, b_o gradients all rel_err < 1e-5.
- [ ] FD relative_position_bias_ gradient non-zero AND `bias_type="none"` produces finite FD input grad.
- [ ] zero_grad clears all 6 grads.
- [ ] update_weights moves all 6 params.
- [ ] 30 SGD steps reduce loss by > 10%.
- [ ] Umbrella `#include "nn/nn.h"` compiles standalone clean.
- [ ] Full `make tests && ./build/test_*` (or focused subset) shows no new regressions.
- [ ] No stray debug prints, no `// TODO`, no `std::cerr` left in production code.

## References

- Liu et al. 2021, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", https://arxiv.org/abs/2103.14030 — primary paper, §3.2 (window self-attention, §3.2 relative position bias)
- The existing `include/nn/layers/attention/axial_attention.{h,cpp}` — same family of attention variant, uses the same Layer interface contract; mirror its style for `parameters()/gradients()/zero_grad()/update_weights()`.
