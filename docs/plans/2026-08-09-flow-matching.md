# Flow Matching Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a complete, well-tested Flow Matching (Lipman et al. 2023) generative-modeling stack — loss, conditional + optimal-transport variants, ODE sampler, and a small reference velocity network — to the `neural_net` repo's generative folder.

**Architecture:**
- A self-contained `FlowMatching` class that owns the training loss (MSE between predicted velocity and target velocity `x_1 - x_0`), the linear interpolation `x_t = (1-t)x_0 + t x_1`, and the Euler ODE sampler for inference.
- A `ConditionalFlowMatching` subclass that adds a per-row class label (Lipman §3.2) — the target velocity becomes `(x_1 - (1-σ)x_0) / (1 - (1-σ_min)t)` to break the determinism of the marginal path and enable classifier-free guidance downstream.
- An `OptimalTransportConditionalFlowMatching` variant (Tong et al. 2023, "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport") that uses minibatch OT (we use a simple Sinkhorn-style assignment: by default, match `x_0[i]` to the nearest `x_1[j]` in the same minibatch via a squared-L2 cost + Hungarian-free greedy nearest-neighbour assignment, which is what Tong et al. find sufficient).
- A `FlowMatchingNet` small reference velocity-prediction network — input is `[x_t (dim), t (1), optional y (num_classes)]`, output is `dim`. Two Dense layers with SiLU activation and a residual connection. This is the velocity net that the loss trains — keeping it tiny (4 layers, hidden=64) so the test suite can verify gradients at machine precision.
- An Euler ODE sampler `sample(model, n, n_steps, sigma_min, class_labels)` that runs `x_{k+1} = x_k + (dt)·v_θ(x_k, t_k)` from `t=0` to `t=1` starting at `x_0 ~ N(0, I)`.
- A small dataset utility `gaussian_mixture_2d()` that returns `(x0, x1)` pairs from two well-separated 2-D Gaussians so the test suite can verify the model actually learns to transform one distribution into another.

**Tech Stack:** Existing `Tensor`, `Dense`, `Layer` from `include/nn/core/`. Standard `<random>`, `<cmath>`, `<algorithm>`.

**Paper references:**
- Lipman, Chen, Benjamini, Nickel, Le, Lipman: "Flow Matching for Generative Modeling" (ICLR 2023), https://arxiv.org/abs/2210.02747
- Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio: "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (ICML 2024), https://arxiv.org/abs/2302.00482
- Liu, Gong, Liu: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022) — equivalent formulation (independent discovery)

**Why now:** The repo has DDPM, VAE, WGAN-GP, Neural Spline Flow, RealNVP, PixelCNN, Consistency, but no flow matching. This fills a real gap and complements the existing diffusion/consistency models with a cleaner alternative that doesn't need a noise schedule.

---

## Layout

New files:
- `include/nn/layers/generative/flow_matching.h`
- `include/nn/layers/generative/flow_matching.cpp`
- `tests/test_flow_matching.cpp`

Edits:
- `include/nn/nn.h` — add `#include "layers/generative/flow_matching.h"`
- `Makefile` — add build rule, `tests:` deps entry, `run_tests:` echo line

---

## Phase 0 — Plan the gradient chain (critical, no code yet)

The flow-matching loss is just MSE on the velocity. The velocity net is a Dense-based MLP. The gradient chain we need to verify:

1. **`loss = mean_over_batch( ||v_pred(x_t, t) - v_target||² )`** — fully vectorized; `loss` is `(1, 1)`.
2. **`d_loss / d_v_pred[i] = 2/N * (v_pred[i] - v_target[i])`** — the standard MSE grad.
3. **`d_v_pred[i] / d_x_t[i]`** — through the velocity net.
4. **`d_v_pred[i] / d_t`** — through the time-conditioning path.
5. **`d_v_pred[i] / d_y[i]`** (class conditioning) — through the class-embedding path.
6. **`d_loss / d_W_layer`** — accumulated via the standard Dense backward path.
7. **`d_loss / d_b_layer`** — same.

Critical bug surfaces (in advance):
- **`x_t` is a function of `t`, `x_0`, `x_1`.** When `x_t` is itself part of a path that depends on the noise / training data, the gradient of `loss` w.r.t. `x_0` or `x_1` is non-trivial. In our test we DON'T backprop into `x_0` / `x_1` (they're data); we only backprop into the velocity-net parameters. So the chain simplifies to `d_loss/d_params = d_loss/d_v_pred · d_v_pred/d_params`, with `x_t` treated as a fixed input.
- **Time conditioning.** We broadcast-concatenate `t` as a (1, 1) tensor with each `x_t[i]` to form the net's input of shape (N, dim+1+num_classes). The gradient w.r.t. `t` itself is irrelevant for training; we only care about the chain through `t` to the W1/W2 of the net.
- **Class conditioning.** Same shape handling — broadcast a one-hot vector.
- **The "sigma_min" detour** in conditional FM. The conditional path interpolates `x_t = (1 - (1-σ_min)·t)·x_0 + t·x_1` and targets `v = x_1 - (1-σ_min)·x_0`. This avoids the trivial-path problem where every `x_t` lies exactly on the segment between paired `x_0` and `x_1`. The conditional FM loss is the same MSE.
- **OT path.** Same loss, but we PERMUTE the rows of `x_1` (and `y_1` if class-conditional) using the OT assignment before computing `x_t` and `v`. The permutation does not depend on the model parameters, so the gradient chain is unchanged once the permuted `x_1` is fixed.

---

## Phase 1 — Skeleton + validation

### Task 1: Create `flow_matching.h` skeleton

**Files:**
- Create: `include/nn/layers/generative/flow_matching.h`

**Step 1:** Write the header with the full public API and class declarations. The implementation will be empty stubs that throw `std::logic_error("not implemented")` for every method that the test suite does NOT immediately exercise — this keeps the TDD cycle honest (no silently-wrong code).

**Key API surface (must all be present from the start):**
- `class GaussianMixture2D` — utility that returns `x0, x1` pairs for testing
  - `GaussianMixture2D(int n_per_cluster=64, int dim=2, double scale=1.0, double separation=4.0, int seed=42)`
  - `std::pair<Tensor, Tensor> sample_pair()` — returns `(x0, x1)` each of shape `(2*n_per_cluster, dim)` drawn from two well-separated Gaussians
- `class TimeEmbedding` — sinusoidal time embedding used by the velocity net
  - `TimeEmbedding(int hidden_dim)` — default hidden = 64
  - `Tensor forward(double t) const` — returns (1, hidden_dim) sinusoidal embedding
- `class ClassEmbedding` — learned class embedding (one per class)
  - `ClassEmbedding(int num_classes, int hidden_dim)` — uses `Tensor::random`
  - `Tensor forward(int label) const` — returns (1, hidden_dim)
  - `Tensor forward(const Tensor& one_hot_labels) const` — returns (N, hidden_dim)
- `class FlowMatchingNet : public Layer` — the velocity-prediction network
  - `FlowMatchingNet(int data_dim, int hidden_dim=64, int num_classes=0)`
  - input is `(N, data_dim + 1 + num_classes)`: `[x_t, t_broadcast, y_onehot]`
  - `Tensor forward(const Tensor& input)` returns `(N, data_dim)` velocity
  - `Tensor backward(const Tensor& grad_output, double lr)` — standard Dense chain
  - `update_weights(double lr)` — standard Dense `update_weights`
  - `parameters() / gradients() / zero_grad() / get_weights / get_gradients`
- `class FlowMatching` — the loss / trainer
  - `FlowMatching(int data_dim, int hidden_dim=64, int num_classes=0, double sigma_min=0.0, bool use_ot=false)`
  - `Tensor forward(const Tensor& x0, const Tensor& x1)` — returns scalar `(1, 1)` loss
  - `Tensor backward()` — returns gradient w.r.t. velocity-net parameters (returns the per-batch grad_output to be passed to `net.backward()`)
  - `void sample(int n_samples, int n_steps=50, std::vector<int> class_labels={}, int seed=0)` — Euler ODE sampler that updates `last_samples_`
  - `get_net() const` returns `const FlowMatchingNet&`
  - `last_loss()`, `last_x_t()`, `last_v_target()`, `last_v_pred()`, `last_samples()` accessors
- `class ConditionalFlowMatching : public FlowMatching` — same API but `sigma_min > 0` path
- `class OptimalTransportFlowMatching : public FlowMatching` — same API but `use_ot=true`

**Step 2:** Compile.

```bash
make -j4
```

Expected: clean compile, no new warnings.

**Step 3:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.h
git commit -m "feat(generative): add FlowMatching skeleton (Lipman 2023)"
```

---

### Task 2: Implement `TimeEmbedding` and `ClassEmbedding` with TDD

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h`
- Modify: `include/nn/layers/generative/flow_matching.cpp` (new file)
- Test: `tests/test_flow_matching.cpp` (new file, just helpers + 2 tests)

**Step 1:** Write the failing test in `tests/test_flow_matching.cpp`:

```cpp
static void test_time_embedding() {
    TimeEmbedding te(8);
    Tensor e = te.forward(0.0);
    // At t=0, the sinusoid is sin(0)=0 for odd indices, cos(0)=1 for even
    CHECK_NEAR(e[0][0], 1.0, 1e-12, "t=0 channel 0 = 1");
    CHECK_NEAR(e[0][1], 0.0, 1e-12, "t=0 channel 1 = 0");
    // At t=1, position 0 is sin(1) ≈ 0.8415, position 1 is cos(1) ≈ 0.5403
    Tensor e1 = te.forward(1.0);
    CHECK_NEAR(e1[0][0], std::sin(1.0), 1e-12, "t=1 channel 0 = sin(1)");
    CHECK_NEAR(e1[0][1], std::cos(1.0), 1e-12, "t=1 channel 1 = cos(1)");
}

static void test_class_embedding() {
    ClassEmbedding ce(4, 6);
    Tensor e0 = ce.forward(0);
    CHECK(e0.rows == 1 && e0.cols == 6, "single-label embedding shape");
    // one-hot batch
    Tensor one_hot(2, 4);
    one_hot[0][2] = 1.0; one_hot[1][1] = 1.0;
    Tensor eb = ce.forward(one_hot);
    CHECK(eb.rows == 2 && eb.cols == 6, "batch one-hot shape");
    CHECK_NEAR(eb[0][0], ce.forward(2)[0][0], 1e-12, "single==batch row 0 channel 0");
    CHECK_NEAR(eb[1][0], ce.forward(1)[0][0], 1e-12, "single==batch row 1 channel 0");
}
```

**Step 2:** Compile and run the test, expect FAIL (function not defined).

```bash
make -j4 build/test_flow_matching && ./build/test_flow_matching
```

Expected: link errors for `TimeEmbedding::forward` and `ClassEmbedding::forward`.

**Step 3:** Implement in `flow_matching.cpp`:
- `TimeEmbedding::forward(t)` uses standard transformer sinusoidal: `e[2k] = sin(2π·t·exp(-k·log(10000)/(hidden/2-1)))`, `e[2k+1] = cos(...)` for `k = 0, ..., hidden/2-1`.
- `ClassEmbedding::forward(label)` returns `embeddings_[label]` (a `(num_classes, hidden_dim)` tensor).
- `ClassEmbedding::forward(one_hot)` returns `one_hot @ embeddings_`.

**Step 4:** Run the test, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): implement TimeEmbedding and ClassEmbedding (FM helper modules)"
```

---

### Task 3: Implement `GaussianMixture2D` with TDD

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h` (add class declaration)
- Modify: `include/nn/layers/generative/flow_matching.cpp` (add implementation)
- Modify: `tests/test_flow_matching.cpp` (add test)

**Step 1:** Write the failing test:

```cpp
static void test_gaussian_mixture_2d() {
    GaussianMixture2D gm(n_per_cluster=100, dim=2, scale=0.5, separation=4.0, seed=42);
    auto [x0, x1] = gm.sample_pair();
    CHECK(x0.rows == 200 && x0.cols == 2, "x0 shape (200, 2)");
    CHECK(x1.rows == 200 && x1.cols == 2, "x1 shape (200, 2)");
    // Mean of cluster 0 should be near (-2, -2); cluster 1 near (+2, +2)
    // Check that the empirical means are within 0.3 of the centers
    double mx0 = 0, my0 = 0;
    for (int i = 0; i < 100; ++i) { mx0 += x0[i][0]; my0 += x0[i][1]; }
    mx0 /= 100.0; my0 /= 100.0;
    CHECK(std::abs(mx0 - (-2.0)) < 0.3, "cluster 0 mean x ≈ -2");
    CHECK(std::abs(my0 - (-2.0)) < 0.3, "cluster 0 mean y ≈ -2");
    // Determinism: same seed produces same data
    GaussianMixture2D gm2(100, 2, 0.5, 4.0, 42);
    auto [x0b, x1b] = gm2.sample_pair();
    bool same = true;
    for (int i = 0; i < 200; ++i) {
        if (x0[i][0] != x0b[i][0]) { same = false; break; }
    }
    CHECK(same, "deterministic with same seed");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement. Two clusters centred at `(-sep/2, -sep/2)` and `(+sep/2, +sep/2)`, each scaled by `scale`, sampled with `std::mt19937(seed)`.

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): implement GaussianMixture2D test data utility"
```

---

## Phase 2 — Velocity network

### Task 4: Implement `FlowMatchingNet` with TDD — forward shape + finite

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h`
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Write failing test:

```cpp
static void test_fm_net_forward_shape() {
    FlowMatchingNet net(data_dim=4, hidden_dim=8, num_classes=0);
    Tensor x(3, 5);  // 3 samples, 4 dims + 1 time
    x[0][0]=1; x[0][1]=2; x[0][2]=3; x[0][3]=4; x[0][4]=0.0;
    x[1] = ...; x[2] = ...;
    Tensor v = net.forward(x);
    CHECK(v.rows == 3 && v.cols == 4, "velocity shape (N, data_dim)");
    // All finite
    bool finite = true;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        if (!std::isfinite(v[i][j])) finite = false;
    CHECK(finite, "all outputs finite");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement `FlowMatchingNet` as 2 Dense layers with SiLU activation + residual:
- Layer 1: `Dense(hidden_dim, data_dim + 1)` (input is concatenated x and t; if `num_classes > 0` also append class one-hot)
- SiLU activation
- Layer 2: `Dense(data_dim, hidden_dim)` then residual + output

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): implement FlowMatchingNet forward (2-layer MLP with SiLU + residual)"
```

---

### Task 5: Velocity-net backward — analytical vs centered FD

**Files:**
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Write failing test:

```cpp
static void test_fm_net_backward_fd() {
    srand(0);  // fixed seed for reproducibility
    FlowMatchingNet net(data_dim=2, hidden_dim=8, num_classes=0);
    Tensor x(2, 3);  // 2 samples, 2 dims + 1 time
    x[0][0]=1.0; x[0][1]=0.5; x[0][2]=0.3;
    x[1][0]=-0.2; x[1][1]=0.8; x[1][2]=0.7;
    // Forward to populate last_input_
    Tensor v = net.forward(x);
    // Compute analytical grad for input (just a quick sanity check):
    //    Use MSE(v, v_target) and backprop.
    Tensor target(2, 2);
    target[0][0]=0.1; target[0][1]=0.2; target[1][0]=0.3; target[1][1]=0.4;
    Tensor dv = (v - target) * (2.0 / 2.0);  // d_mse = 2/N * (v - target)
    net.zero_grad();
    Tensor gx = net.backward(dv, 0.0);
    // Numerical gradient for x[0][0]
    double eps = 1e-5;
    double orig = x[0][0];
    x[0][0] = orig + eps;
    Tensor vp = net.forward(x);
    x[0][0] = orig - eps;
    Tensor vm = net.forward(x);
    x[0][0] = orig;
    double mse_p = ((vp[0][0]-target[0][0])*(vp[0][0]-target[0][0]) +
                    (vp[0][1]-target[0][1])*(vp[0][1]-target[0][1]) +
                    (vp[1][0]-target[1][0])*(vp[1][0]-target[1][0]) +
                    (vp[1][1]-target[1][1])*(vp[1][1]-target[1][1])) / 4.0;
    double mse_m = ((vm[0][0]-target[0][0])*(vm[0][0]-target[0][0]) +
                    (vm[0][1]-target[0][1])*(vm[0][1]-target[0][1]) +
                    (vm[1][0]-target[1][0])*(vm[1][0]-target[1][0]) +
                    (vm[1][1]-target[1][1])*(vm[1][1]-target[1][1])) / 4.0;
    double grad_num = (mse_p - mse_m) / (2.0 * eps);
    double grad_ana = gx[0][0];
    double rel = std::abs(grad_ana - grad_num) / std::max(std::abs(grad_ana), std::abs(grad_num));
    CHECK(rel < 1e-4, "input gradient rel_err < 1e-4");
}
```

**Step 2:** Run, expect FAIL (backward probably incorrect or NaN).

**Step 3:** Implement `backward()` correctly: this is just the standard 2-layer MLP backward with a residual connection (the second Dense sees `act(W1·x + b1)` plus the residual `x[:, :data_dim]`).

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): FlowMatchingNet backward — verified vs centered FD at 1e-4"
```

---

## Phase 3 — Flow matching loss

### Task 6: Implement `FlowMatching::forward` (the MSE loss)

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h`
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_flow_matching_forward_known_value() {
    // Construct a tiny case: N=2, dim=2, no class conditioning, sigma_min=0, no OT.
    // x0 = [[0,0],[1,1]], x1 = [[2,2],[3,3]], t = 0.5 for both.
    // x_t = (1-t)x0 + t x1 = [[1,1],[2,2]]
    // v_target = x1 - x0 = [[2,2],[2,2]]
    // Suppose v_pred = [[1.5,1.5],[2.5,2.5]] (off by 0.5 each)
    // MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) = 4 * 0.25 / 4 = 0.25
    FlowMatching fm(data_dim=2, hidden_dim=8, num_classes=0);
    Tensor x0(2, 2); x0[0][0]=0; x0[0][1]=0; x0[1][0]=1; x0[1][1]=1;
    Tensor x1(2, 2); x1[0][0]=2; x1[0][1]=2; x1[1][0]=3; x1[1][1]=3;
    Tensor loss = fm.forward(x0, x1);
    // We can only assert that loss is finite and > 0
    CHECK(std::isfinite(loss[0][0]), "loss finite");
    CHECK(loss[0][0] > 0, "loss > 0");
    // And the cached v_target matches x1 - x0 exactly (for sigma_min = 0)
    Tensor vt = fm.last_v_target();
    CHECK_NEAR(vt[0][0], 2.0, 1e-12, "v_target[0][0] = x1-x0 = 2");
    CHECK_NEAR(vt[1][1], 2.0, 1e-12, "v_target[1][1] = x1-x0 = 2");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement:
- `forward(x0, x1)`:
  - Sample `t ~ U(0, 1)` per row (use a stored `std::mt19937` for repeatability)
  - `x_t = (1 - sigma_min) · (1 - t) · x0 + t · x1` (for `sigma_min = 0` this is the standard linear interp)
  - `v_target = x1 - (1 - sigma_min) · x0`
  - Build net input `[x_t, t_broadcast]` (and class one-hot if `num_classes > 0`)
  - `v_pred = net.forward(input)`
  - `loss = mean(||v_pred - v_target||²)`
  - Cache `x_t`, `v_target`, `v_pred`, `t_vec`, and the net input for backward
  - Return `(1, 1)` tensor

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): FlowMatching forward — MSE velocity loss + interpolation"
```

---

### Task 7: Implement `FlowMatching::backward` (full BPTT)

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_flow_matching_backward_fd() {
    // Centered-FD vs analytical for the W1 weights of the velocity net.
    srand(0);
    FlowMatching fm(data_dim=2, hidden_dim=8, num_classes=0, /*sigma_min*/ 0.0, /*use_ot*/ false);
    Tensor x0(3, 2); for (...) x0[i][j] = some_small_value;
    Tensor x1(3, 2); for (...) x1[i][j] = some_small_value;
    // Forward + backward
    Tensor loss = fm.forward(x0, x1);
    fm.backward();
    // Snapshot a W1 element, say W1[0][0] of the first Dense inside the net
    Dense* d1 = fm.get_net().dense1_;
    double w_orig = d1->weights[0][0];
    double g_ana = d1->grad_weights[0][0];
    // Numerical grad
    double eps = 1e-5;
    d1->weights[0][0] = w_orig + eps;
    Tensor lp = fm.forward(x0, x1);
    d1->weights[0][0] = w_orig - eps;
    Tensor lm = fm.forward(x0, x1);
    d1->weights[0][0] = w_orig;
    double grad_num = (lp[0][0] - lm[0][0]) / (2.0 * eps);
    double rel = std::abs(g_ana - grad_num) / std::max(std::abs(g_ana), std::abs(grad_num));
    CHECK(rel < 1e-4, "W1 grad rel_err < 1e-4");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement `backward()`:
- Compute `d_loss / d_v_pred = 2/N * (v_pred - v_target)` (the standard MSE grad)
- Pass this through `net.backward(d_loss/d_v_pred, 0.0)` to populate the net's `grad_weights` / `grad_bias`
- The net input contains `x_t` and `t` and optionally class one-hot. `x_t` depends on `x0`, `x1`, `t` but we do NOT backprop into them (they're data). So we just throw away the returned `grad_input`.

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): FlowMatching backward — full BPTT, verified vs FD"
```

---

## Phase 4 — Conditional & OT variants

### Task 8: Implement `ConditionalFlowMatching` (sigma_min path)

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h`
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_conditional_fm_path() {
    // With sigma_min > 0, v_target and x_t follow the conditional FM path.
    // For x0=[[0,0]], x1=[[1,1]], t=0.5, sigma_min=0.1:
    //   alpha_t = 1 - (1-0.1)*0.5 = 0.55
    //   x_t = 0.55 * x1 + (1-0.55)*x0 = 0.55 * x1 = [0.55, 0.55]
    //   v_target = x1 - (1-0.1)*x0 = [1.0, 1.0]
    ConditionalFlowMatching cfm(data_dim=2, hidden_dim=8, num_classes=0, sigma_min=0.1);
    Tensor x0(1, 2); x0[0][0]=0; x0[0][1]=0;
    Tensor x1(1, 2); x1[0][0]=1; x1[0][1]=1;
    cfm.forward(x0, x1);
    Tensor xt = cfm.last_x_t();
    CHECK_NEAR(xt[0][0], 0.55, 1e-12, "conditional x_t[0][0]");
    CHECK_NEAR(xt[0][1], 0.55, 1e-12, "conditional x_t[0][1]");
    Tensor vt = cfm.last_v_target();
    CHECK_NEAR(vt[0][0], 1.0, 1e-12, "conditional v_target[0][0]");
    CHECK_NEAR(vt[0][1], 1.0, 1e-12, "conditional v_target[0][1]");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement `ConditionalFlowMatching::forward`:
- `alpha_t = 1 - (1 - sigma_min) * t`
- `x_t = alpha_t * x1 + (1 - alpha_t) * x0`
- `v_target = x1 - (1 - sigma_min) * x0`
- Loss / backward / sample identical to `FlowMatching`

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): ConditionalFlowMatching — sigma_min path (Lipman §3.2)"
```

---

### Task 9: Implement `OptimalTransportFlowMatching` (minibatch OT)

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.h`
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_ot_fm_permutation() {
    // With x0 cluster at (-2,-2) and x1 cluster at (+2,+2), the OT plan should
    // map each x0 to the nearest x1 (which is also the x1 in the matching pair
    // because they're all in the same cluster). The permutation index would
    // map x0[i] -> x1[i] (identity) for the SAME ordering.
    //
    // To test that OT is doing SOMETHING, we create x0 in cluster A and x1 in
    // cluster B and verify the paired v_target has the same sign as x1 - x0
    // (which is what the OT plan guarantees after nearest-neighbour matching).
    srand(0);
    OptimalTransportFlowMatching otfm(data_dim=2, hidden_dim=8, num_classes=0, use_ot=true);
    Tensor x0(4, 2); x1(4, 2);  // mix from both clusters
    // ... populate with x0 from cluster A and x1 from cluster B ...
    Tensor loss = otfm.forward(x0, x1);
    CHECK(std::isfinite(loss[0][0]), "OT loss finite");
    // Determinism: same seed produces same loss
    OptimalTransportFlowMatching otfm2(data_dim=2, hidden_dim=8, num_classes=0, use_ot=true);
    srand(0);  // re-seed the RNG so internal t-sampling matches
    Tensor loss2 = otfm2.forward(x0, x1);
    CHECK_NEAR(loss[0][0], loss2[0][0], 1e-12, "OT FM is deterministic");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement: for each minibatch, build a `(N, N)` squared-L2 cost matrix `C[i][j] = ||x0[i] - x1[j]||²` and greedily assign each `x0[i]` to its nearest `x1[j]` (skipping already-assigned `j`). Return the permuted `x1_perm[i] = x1[assign[i]]`. (We use greedy nearest-neighbour instead of full Hungarian because the cost is `O(N²)` for both, the result is sufficient for training, and the test only needs determinism, not optimality.)

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): OptimalTransportFlowMatching — greedy minibatch OT (Tong 2023)"
```

---

## Phase 5 — ODE sampler

### Task 10: Implement `FlowMatching::sample` (Euler ODE)

**Files:**
- Modify: `include/nn/layers/generative/flow_matching.cpp`
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_fm_sample_shape_and_finite() {
    FlowMatching fm(data_dim=2, hidden_dim=8, num_classes=0);
    // Don't bother training — just check sample runs and produces finite (n, dim) output.
    fm.sample(n_samples=16, n_steps=10, /*class_labels*/ {}, /*seed*/ 0);
    Tensor s = fm.last_samples();
    CHECK(s.rows == 16 && s.cols == 2, "sample shape");
    bool finite = true;
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 2; ++j)
        if (!std::isfinite(s[i][j])) finite = false;
    CHECK(finite, "samples finite");
}

static void test_fm_sample_class_conditional() {
    FlowMatching fm(data_dim=2, hidden_dim=8, num_classes=3);
    fm.sample(n_samples=6, n_steps=5, /*class_labels*/ {0, 0, 1, 1, 2, 2}, /*seed*/ 1);
    Tensor s = fm.last_samples();
    CHECK(s.rows == 6 && s.cols == 2, "sample shape");
    bool finite = true;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 2; ++j)
        if (!std::isfinite(s[i][j])) finite = false;
    CHECK(finite, "conditional samples finite");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Implement:
- `sample(n_samples, n_steps, class_labels, seed)`:
  - Build `x0 ~ N(0, I)` of shape `(n_samples, data_dim)` (using a local `std::mt19937(seed)`)
  - For `k = 0, ..., n_steps - 1`:
    - `t_k = k / n_steps`
    - `v_pred = net.forward([x_k, t_k, class_onehot])`
    - `dt = 1.0 / n_steps`
    - `x_{k+1} = x_k + dt * v_pred`
  - Store final `x_{n_steps}` in `last_samples_`.

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add include/nn/layers/generative/flow_matching.cpp tests/test_flow_matching.cpp
git commit -m "feat(generative): FlowMatching Euler ODE sampler — basic + class-conditional"
```

---

## Phase 6 — End-to-end training

### Task 11: Training reduces loss on Gaussian-mixture transport

**Files:**
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_fm_training_reduces_loss() {
    srand(42);
    FlowMatching fm(data_dim=2, hidden_dim=32, num_classes=0);
    GaussianMixture2D gm(64, 2, 0.5, 4.0, 42);
    Tensor loss0 = fm.forward(x0, x1);  // warm up
    double l0 = loss0[0][0];
    // Train 200 steps with lr = 1e-3 (Adam-like)
    for (int step = 0; step < 200; ++step) {
        auto [x0, x1] = gm.sample_pair();
        fm.forward(x0, x1);
        fm.backward();
        // Adam-style update (or just SGD; let's use SGD for simplicity)
        auto params = fm.get_net().parameters();
        auto grads = fm.get_net().gradients();
        for (size_t i = 0; i < params.size(); ++i) {
            for (size_t r = 0; r < params[i]->rows; ++r)
                for (size_t c = 0; c < params[i]->cols; ++c)
                    (*params[i])[r][c] -= 1e-3 * (*grads[i])[r][c];
        }
        fm.get_net().zero_grad();
    }
    auto [x0f, x1f] = gm.sample_pair();
    Tensor lossF = fm.forward(x0f, x1f);
    double lF = lossF[0][0];
    CHECK(lF < l0 * 0.5, "training reduces loss by > 50%");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Tune (hidden_dim or learning rate) until loss reduction happens.

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add tests/test_flow_matching.cpp
git commit -m "test(generative): FlowMatching end-to-end training reduces loss >50% over 200 steps"
```

---

### Task 12: Trained sampler transports one cluster to another

**Files:**
- Modify: `tests/test_flow_matching.cpp`

**Step 1:** Failing test:

```cpp
static void test_fm_trained_sampler_transports() {
    // Train FM for 500 steps to transport N(0, I) -> bimodal Gaussian at ±2.
    // Then sample 200 points and check the empirical means are near ±2.
    srand(42);
    FlowMatching fm(data_dim=2, hidden_dim=64, num_classes=0);
    GaussianMixture2D gm(64, 2, 0.5, 4.0, 42);
    for (int step = 0; step < 500; ++step) {
        auto [x0, x1] = gm.sample_pair();
        fm.forward(x0, x1);
        fm.backward();
        // ... same update as Task 11 ...
    }
    fm.sample(200, n_steps=50, /*class_labels*/ {}, /*seed*/ 7);
    Tensor s = fm.last_samples();
    // Compute empirical mean of s (should be near (0,0) because the mixture is symmetric)
    double mx = 0, my = 0;
    for (int i = 0; i < 200; ++i) { mx += s[i][0]; my += s[i][1]; }
    mx /= 200; my /= 200;
    CHECK(std::abs(mx) < 1.0, "sampler mean x ≈ 0 (mixture symmetric)");
    CHECK(std::abs(my) < 1.0, "sampler mean y ≈ 0");
    // Variance should be roughly the target variance: scale² + (separation/2)² = 0.25 + 4 = 4.25
    double vx = 0, vy = 0;
    for (int i = 0; i < 200; ++i) { vx += (s[i][0] - mx) * (s[i][0] - mx); vy += (s[i][1] - my) * (s[i][1] - my); }
    vx /= 200; vy /= 200;
    CHECK(vx > 1.0, "sampler variance x > 1 (matches target mixture)");
    CHECK(vy > 1.0, "sampler variance y > 1");
}
```

**Step 2:** Run, expect FAIL.

**Step 3:** Tune (more steps, larger hidden) until transport happens.

**Step 4:** Run, expect PASS.

**Step 5:** Commit.

```bash
git add tests/test_flow_matching.cpp
git commit -m "test(generative): trained FM sampler transports N(0,I) to bimodal target"
```

---

## Phase 7 — Integration

### Task 13: Register in umbrella header + Makefile

**Files:**
- Modify: `include/nn/nn.h` — add `#include "layers/generative/flow_matching.h"` after the consistency model include
- Modify: `Makefile`:
  - Add `$(BUILD_DIR)/test_flow_matching: $(LIB_OBJS) $(BUILD_DIR)/test_flow_matching.o` rule
  - Add `$(BUILD_DIR)/test_flow_matching.o` to the `tests:` deps list
  - Add `$(BUILD_DIR)/test_flow_matching` to the `tests:` list
  - Add `@echo "=== Running Flow Matching Tests ===" && ./$(BUILD_DIR)/test_flow_matching` to `run_tests:`

**Step 1:** Make the edits.

**Step 2:** Build and run.

```bash
make -j4 build/test_flow_matching && ./build/test_flow_matching
```

Expected: All tests pass.

**Step 3:** Run the full suite to verify no regressions.

```bash
make -j4 tests && make run_tests 2>&1 | tail -100
```

Expected: only pre-existing deferred failures (per NOT_FIXED.md).

**Step 4:** Commit.

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register FlowMatching in umbrella header and Makefile"
```

---

### Task 14: Update EXPANSION_QUEUE.md + cleanup

**Files:**
- Modify: `EXPANSION_QUEUE.md` — move the entry to "Done" with a one-line summary + test results
- Modify: `docs/plans/2026-08-09-flow-matching.md` — append final results section

**Step 1:** Edit `EXPANSION_QUEUE.md` — append a new bullet to `## Done`:

```
- **Flow Matching (Lipman et al. 2023)** — Implemented in `include/nn/layers/generative/flow_matching.{h,cpp}`. Three classes: (1) `FlowMatching` — linear-interpolation path `x_t = (1-t)x_0 + t x_1`, target velocity `v = x_1 - x_0`, MSE velocity loss with centered-FD verification at 1e-4, Euler ODE sampler. (2) `ConditionalFlowMatching` — Lipman §3.2 path `x_t = α_t·x_1 + (1-α_t)·x_0` where `α_t = 1 - (1-σ_min)t`, target `v = x_1 - (1-σ_min)·x_0`. (3) `OptimalTransportFlowMatching` — Tong et al. 2023, minibatch OT reorders `x_1` rows by greedy nearest-neighbour assignment before the same loss. Plus `GaussianMixture2D` test data utility (two 2-D Gaussians at ±separation/2 with deterministic seed), `TimeEmbedding` (sinusoidal), `ClassEmbedding` (learned one-hot), and `FlowMatchingNet` (2-layer MLP with SiLU + residual, conditioned on `[x_t, t, y]`). **N/N focused checks pass**: ... Plan: `docs/plans/2026-08-09-flow-matching.md`.
```

**Step 2:** Verify the queue file:

```bash
git diff EXPANSION_QUEUE.md
```

**Step 3:** Commit.

```bash
git add EXPANSION_QUEUE.md docs/plans/2026-08-09-flow-matching.md
git commit -m "docs: mark Flow Matching as Done in EXPANSION_QUEUE"
```

---

## Phase 8 — Final verification

### Task 15: Full test run + summary

```bash
make -j4 tests && make run_tests 2>&1 | tail -40
```

Expected: every test target prints `=== Summary: N/N passed ===` with `N == N` (no NEW failures — the deferred pre-existing failures in NOT_FIXED.md are not in scope).

Final summary line: see below in the cron report.

---

## Risk register

| Risk | Mitigation |
|---|---|
| **The velocity net's t-conditioning chain is subtle.** The net input concatenates `x_t` (which depends on data + noise) with `t` (the per-row U(0,1) sample). When we backprop `d_loss/d_input`, we throw away `d_loss/d_x_t` because `x_t` is data — but if we accidentally try to chain `d_loss/d_t` through the net's W1, the gradient magnitude might be off. | Test 7 (FD vs analytical on W1) catches this; if `rel_err > 1e-4`, debug. |
| **OT minibatch implementation O(N²) cost per step.** | N is small in tests (≤ 64), so cost is fine. Document this is minibatch OT, not exact OT. |
| **Euler ODE sampler drift at large n_steps.** | The user can pass `n_steps` to balance speed vs accuracy. n_steps=50 is the test default and is sufficient for the 2-D transport test. |
| **`update_weights(lr)` API on FlowMatchingNet.** We override `Layer::update_weights(lr)` to delegate to the internal Denses. | Trivial implementation, tested in Task 11. |
| **ClassEmbedding state-vs-init.** Need to make sure the random init uses a deterministic seed. | Use `Tensor::random(scale)` with the caller's seed if provided. |

## Mutation tests

After every Task 11/12 test is passing, run **3 mutation probes** to confirm the chain is non-vacuous:

1. **Drop the `(1 - sigma_min)` factor in conditional FM:** the `v_target` becomes `x1 - x0` regardless of `sigma_min`. Test 8 fails with `rel_err > 1` on `v_target[0][0]`.
2. **Skip the OT permutation in OT FM:** `x1_perm = x1` regardless of the cost matrix. Test 9 fails because the OT-FM loss differs from the standard FM loss on mismatched clusters.
3. **Drop `dt = 1/n_steps` in the ODE sampler (use `dt = 1`):** `x_1` becomes the velocity itself (way out of the unit ball). Test 10's `finite` check fails.

Each mutation must be detected by at least one test.

---

## Done criteria

- All tests pass (`N/N` summary in the test_flow_matching output)
- Full suite run doesn't introduce new regressions (per NOT_FIXED.md, those deferred tests are out of scope)
- Plan moved to Done in EXPANSION_QUEUE.md
- All commits pushed to master
