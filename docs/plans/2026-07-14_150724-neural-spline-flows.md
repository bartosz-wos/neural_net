# Neural Spline Flow Coupling Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a professional, batch-aware neural rational-quadratic spline coupling layer with monotone transforms, exact inversion, log-Jacobian tracking, and analytical gradients suitable for density estimation.

**Architecture:** `RationalQuadraticSpline` implements Durkan et al.'s monotone rational-quadratic transform on `[-tail_bound, tail_bound]` with identity linear tails, normalized positive widths/heights, and positive internal derivatives. `NSFCouplingLayer` splits each input row into conditioning and transformed halves, predicts one spline parameter vector per transformed scalar through `Dense -> tanh -> Dense`, applies the spline independently, and backpropagates both output and log-determinant objectives. The spline's local Jacobian is evaluated by a small private forward-mode dual-number helper so production gradients are exact analytical derivatives of the implemented formula rather than numerical finite differences.

**Tech Stack:** C++17, repository `Tensor`/`Layer`/`Dense` APIs, Make, deterministic centered finite differences, no external runtime dependencies.

**References:** Durkan et al., “Neural Spline Flows,” NeurIPS 2019, arXiv:1906.04032; Bayes Labs `nflows` rational-quadratic reference implementation for the standard constrained parameterization and stable inverse root.

---

### Task 1: Establish the public spline and coupling contracts with a failing test

**Objective:** Lock validation, dimensions, linear-tail behavior, and the intended public API before production code exists.

**Files:**
- Create: `tests/test_neural_spline_flow.cpp`
- Modify: `Makefile:60-325`
- Test: `tests/test_neural_spline_flow.cpp`

**Step 1: Write the failing API tests**

Create a check-based test executable that includes the missing header:

```cpp
#include "nn/layers/generative/neural_spline_flow.h"

RationalQuadraticSpline spline(/*num_bins=*/8, /*tail_bound=*/3.0);
CHECK(spline.num_bins() == 8, "stores bin count");
CHECK_NEAR(spline.tail_bound(), 3.0, 0.0, "stores tail bound");
CHECK(spline.num_derivative_params() == 7,
      "linear tails use K-1 learned internal derivatives");

NSFCouplingLayer layer(/*input_dim=*/4, /*hidden_size=*/16,
                       /*num_bins=*/8, /*tail_bound=*/3.0,
                       /*swap_halves=*/false);
CHECK(layer.input_dim() == 4, "stores input dimension");
CHECK(layer.half_dim() == 2, "derives half dimension");
CHECK(layer.params_per_scalar() == 23, "K+K+(K-1) outputs");
```

The standard linear-tail NSF parameterization learns `K-1` internal knot derivatives and fixes both boundary derivatives to `1`; this gives `K+1` total knot derivatives while remaining continuously differentiable with the identity tails. The queue shorthand “K derivatives” is therefore refined to the paper-compatible `K-1` learned internal values.

Add constructor rejection checks for:

- `num_bins < 2`;
- non-positive or non-finite `tail_bound`;
- `num_bins * min_bin_width >= 1`;
- `num_bins * min_bin_height >= 1`;
- odd or zero `input_dim`;
- zero `hidden_size`.

Add forward rejection for `input.cols != input_dim` and an empty batch.

**Step 2: Add only the focused Make target**

```make
$(BUILD_DIR)/test_neural_spline_flow: $(LIB_OBJS) $(BUILD_DIR)/test_neural_spline_flow.o
	$(CXX) $^ -o $@
```

Do not add it to aggregate `tests` or `run_tests` before GREEN.

**Step 3: Run RED**

Run:

```bash
rm -f build/test_neural_spline_flow.o build/test_neural_spline_flow
make build/test_neural_spline_flow
```

Expected: compilation fails because `nn/layers/generative/neural_spline_flow.h` does not exist. This proves the new test target reaches the intended API.

**Step 4: Keep RED uncommitted**

The test and production code land together after the complete cycle is GREEN.

---

### Task 2: Implement the monotone rational-quadratic spline primitive

**Objective:** Make validation, identity-tail, monotonicity, derivative, and forward/inverse tests pass independently of the neural conditioner.

**Files:**
- Create: `include/nn/layers/generative/neural_spline_flow.h`
- Create: `include/nn/layers/generative/neural_spline_flow.cpp`
- Modify: `tests/test_neural_spline_flow.cpp`

**Step 1: Define the exact public primitive**

```cpp
struct SplineTransformResult {
    double value;
    double log_abs_det;
};

class RationalQuadraticSpline {
public:
    RationalQuadraticSpline(size_t num_bins = 8,
                            double tail_bound = 3.0,
                            double min_bin_width = 1e-3,
                            double min_bin_height = 1e-3,
                            double min_derivative = 1e-3);

    SplineTransformResult forward(
        double input,
        const std::vector<double>& unnormalized_widths,
        const std::vector<double>& unnormalized_heights,
        const std::vector<double>& unnormalized_internal_derivatives) const;

    SplineTransformResult inverse(
        double input,
        const std::vector<double>& unnormalized_widths,
        const std::vector<double>& unnormalized_heights,
        const std::vector<double>& unnormalized_internal_derivatives) const;

    size_t num_bins() const;
    size_t num_derivative_params() const;
    double tail_bound() const;
    double min_bin_width() const;
    double min_bin_height() const;
    double min_derivative() const;
};
```

Validate vector sizes exactly (`K`, `K`, and `K-1`) and reject non-finite raw parameters.

**Step 2: Implement constrained parameters**

For `K` bins over `[-B, B]`:

```cpp
soft_w[k] = softmax(raw_w)[k];
width[k] = 2*B * (min_width + (1 - K*min_width) * soft_w[k]);

soft_h[k] = softmax(raw_h)[k];
height[k] = 2*B * (min_height + (1 - K*min_height) * soft_h[k]);

derivative[0] = 1;
derivative[K] = 1;
derivative[k] = min_derivative
              + softplus(raw_d[k-1] + inverse_softplus(1-min_derivative));
```

The inverse-softplus shift makes raw derivative `0` map exactly to derivative `1`, so all-zero raw parameters produce the identity spline: uniform widths, uniform heights, and unit derivatives.

Use max-subtracted softmax and branch-stable softplus.

**Step 3: Implement the forward formula**

For the selected bin `k`, with:

```text
x_k = cumulative width at bin start
y_k = cumulative height at bin start
w_k = bin width
h_k = bin height
delta_k = h_k / w_k
theta = (x - x_k) / w_k
u = theta * (1 - theta)
D = delta_k + (d_k + d_{k+1} - 2*delta_k) * u
```

the transform is:

```text
y = y_k + h_k * (delta_k*theta^2 + d_k*u) / D
```

and:

```text
dy/dx numerator = delta_k^2 *
  (d_{k+1}*theta^2 + 2*delta_k*u + d_k*(1-theta)^2)
log|dy/dx| = log(derivative_numerator) - 2*log(D)
```

Outside `[-B, B]`, return identity with `log_abs_det = 0`.

**Step 4: Implement the stable inverse**

Search cumulative heights for the output bin. Let `y_rel = y - y_k` and:

```text
a = y_rel * (d_k + d_{k+1} - 2*delta_k)
  + h_k * (delta_k - d_k)
b = h_k*d_k
  - y_rel * (d_k + d_{k+1} - 2*delta_k)
c = -delta_k * y_rel
```

Use the stable root:

```text
theta = (2*c) / (-b - sqrt(max(0, b^2 - 4*a*c)))
x = x_k + theta*w_k
```

Clamp only roundoff excursions of `theta` to `[0,1]`; do not mask materially invalid roots. Return the negative forward log determinant evaluated at that `theta`.

**Step 5: Add non-degenerate primitive tests**

Use asymmetric raw values, for example:

```cpp
raw_w = {-0.7, 0.2, 1.1, -0.3};
raw_h = { 0.4, 1.0,-0.5,  0.1};
raw_d = {-0.2, 0.8, 0.3};
```

Add tests for:

1. all-zero raw parameters produce identity inside the interval;
2. inputs outside either tail are exact identity with zero log determinant;
3. 401 sorted inputs across `[-B, B]` produce strictly increasing outputs;
4. forward then inverse recovers each input within `1e-10`;
5. inverse then forward recovers each output within `1e-10`;
6. `exp(log_abs_det)` matches centered finite difference `dy/dx` within `1e-6` relative error;
7. boundary values map exactly to `-B` and `B` and have derivative `1`;
8. extreme finite raw logits remain finite and monotone.

**Step 6: Run the primitive GREEN suite**

Run a clean affected-object rebuild. Expected: primitive checks pass; coupling tests remain RED until Task 3.

---

### Task 3: Implement batch-aware neural spline coupling forward and inverse

**Objective:** Add the conditioner, per-row split/scatter logic, exact round trips, and log-determinant accumulation.

**Files:**
- Modify: `include/nn/layers/generative/neural_spline_flow.h`
- Modify: `include/nn/layers/generative/neural_spline_flow.cpp`
- Modify: `tests/test_neural_spline_flow.cpp`

**Step 1: Define the exact layer API**

```cpp
class NSFCouplingLayer : public Layer {
public:
    NSFCouplingLayer(size_t input_dim,
                     size_t hidden_size,
                     size_t num_bins = 8,
                     double tail_bound = 3.0,
                     bool swap_halves = false,
                     double min_bin_width = 1e-3,
                     double min_bin_height = 1e-3,
                     double min_derivative = 1e-3);

    Tensor forward(const Tensor& input) override;
    Tensor inverse(const Tensor& output);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void set_log_det_gradient(double gradient);
    double log_det_jacobian() const;

    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "NSFCouplingLayer"; }

    size_t input_dim() const;
    size_t half_dim() const;
    size_t hidden_size() const;
    size_t num_bins() const;
    size_t params_per_scalar() const;
    bool swap_halves() const;
};
```

`params_per_scalar = K + K + (K-1) = 3K-1`.

**Step 2: Build the conditioner**

Use:

```text
conditioned half: (N, half_dim)
conditioner_hidden: Dense(half_dim, hidden_size)
hidden activation: tanh
conditioner_out: Dense(hidden_size, half_dim * (3K-1))
```

Initialize `conditioner_out` weights and bias to zero so the initial coupling is exactly identity, while keeping `conditioner_hidden` Xavier-initialized to provide non-uniform features for first-step output-weight gradients.

**Step 3: Implement batch-aware split/scatter**

For every input row, gather the conditioning and transformed halves independently. `swap_halves=false` conditions on the left half and transforms the right; `swap_halves=true` conditions on the right and transforms the left. Unlike the legacy affine coupling, never flatten multiple batch rows into one conditioner input.

Apply one spline parameter slice to each transformed scalar. Accumulate:

```cpp
log_det_jacobian_ += result.log_abs_det;
```

over all rows and transformed dimensions.

Cache explicit clones of the last forward input, conditioning half, transformed half, tanh hidden output, and raw conditioner output.

**Step 4: Implement inverse without stale-forward ambiguity**

The unchanged half is available directly in the output, so recompute conditioner parameters from it and invert each transformed scalar. Mark the cache as not containing a backward-compatible forward; `backward()` after `inverse()` must throw rather than silently use overwritten `Dense` caches.

**Step 5: Add coupling forward/inverse tests**

Add tests for:

1. output shape preserved for `(N=3, D=4)`;
2. default zero-initialized conditioner is exact identity;
3. manually perturbing conditioner output parameters produces a non-trivial transform;
4. forward/inverse round trip for `N=3`, including values inside and outside tails;
5. `swap_halves=false` leaves left half exact; `swap_halves=true` leaves right half exact;
6. different batch rows receive independent conditioner outputs;
7. log determinant equals the sum of each transformed scalar's primitive result;
8. inverse does not permit a subsequent backward on stale caches.

**Step 6: Run focused forward/inverse GREEN**

Rebuild the layer/test objects explicitly because the Makefile does not track header dependencies.

---

### Task 4: Implement exact analytical backward and prove it with finite differences

**Objective:** Backpropagate output and log-determinant objectives through the spline and conditioner with strict, non-vacuous gradient tests.

**Files:**
- Modify: `include/nn/layers/generative/neural_spline_flow.cpp`
- Modify: `tests/test_neural_spline_flow.cpp`

**Step 1: Add failing gradient tests**

Use a non-square, batch-aware fixture:

```cpp
NSFCouplingLayer layer(/*input_dim=*/4, /*hidden_size=*/5,
                       /*num_bins=*/4, /*tail_bound=*/2.5);
Tensor input(/*rows=*/2, /*cols=*/4);
```

Overwrite both Dense layers with deterministic asymmetric values at scale around `0.2`, including a non-zero output bias so the spline is not the identity. Use inputs away from knot boundaries and tails.

Define one consistent objective:

```text
L = 0.5 * sum((output - target)^2) + lambda * log_det

grad_output = output - target
set_log_det_gradient(lambda)
```

Check every input entry and representative entries from all four parameter tensors with centered finite differences at `eps=1e-5` and:

```cpp
abs(analytical - numerical)
    <= max(2e-7, 5e-4 * max(abs(analytical), abs(numerical), 1e-12));
```

Also run a log-det-only check (`grad_output=0`, `lambda=1`) so the determinant path cannot hide behind the output path.

**Step 2: Run RED**

Expected: input and conditioner gradient checks fail because backward is absent.

**Step 3: Implement a private forward-mode dual evaluator**

For one transformed scalar, create `Dual` values containing:

```cpp
double value;
std::vector<double> derivative; // length 1 + (3K-1)
```

Seed coordinate `0` with the transformed input and coordinates `1..P` with raw spline parameters. Implement dual `+`, `-`, `*`, `/`, `exp`, `log`, stable `softplus`, and max-subtracted `softmax`. Bin selection uses primal values and is piecewise constant; this is the correct almost-everywhere derivative of the continuous spline.

Evaluate the same constrained-parameter and forward formulas as the public primitive. Return derivatives of both transformed output and log determinant with respect to input and every raw parameter.

Outside the tail interval, return `dy/dx=1`, zero raw-parameter derivatives, and zero log-det derivatives.

**Step 4: Accumulate local spline gradients**

For every transformed scalar:

```text
grad_x2 = grad_y * dy/dx + lambda * d(logdet)/dx
grad_raw[p] = grad_y * dy/draw[p]
            + lambda * d(logdet)/draw[p]
```

Scatter `grad_x2` to the transformed half and assemble `grad_raw` in the conditioner output layout.

**Step 5: Backpropagate through the conditioner**

```cpp
grad_hidden = conditioner_out_.backward(grad_raw, 0.0);
grad_pre_hidden = grad_hidden * (1 - tanh_hidden^2);
grad_x1_from_conditioner = conditioner_hidden_.backward(grad_pre_hidden, 0.0);
```

The unchanged half receives direct `grad_output` plus the conditioner contribution. The transformed half receives `grad_x2`. Scatter according to `swap_halves`.

**Step 6: Run strict gradient checks**

Expected: input, hidden/output weights and biases, and log-det-only checks pass below `5e-4`, ideally near machine precision.

**Step 7: Mutation-test non-vacuousness**

Temporarily apply and restore these mutations one at a time:

1. drop `lambda * d(logdet)/draw[p]` — log-det-only parameter test must fail;
2. return `grad_y * dy/dx` without `lambda * d(logdet)/dx` — log-det-only input test must fail;
3. zero the conditioner contribution to the unchanged half — conditioned-half input check must fail;
4. replace one `grad_raw +=` accumulation across batch rows with assignment — use `N=2`; parameter check must fail;
5. drop the derivative softplus shift or force all knot derivatives to one — primitive non-identity/gradient tests must fail.

After each mutation, rebuild affected objects and record the failing checks. Restore and rerun GREEN.

---

### Task 5: Add density-estimation learning, publish the feature, and verify the repository

**Objective:** Prove the layer learns a density objective, wire it into the public library, move the queue item to Done, clean artifacts, and ship atomic commits.

**Files:**
- Modify: `tests/test_neural_spline_flow.cpp`
- Modify: `include/nn/nn.h:160-169`
- Modify: `Makefile:60-390`
- Modify: `EXPANSION_QUEUE.md:6-16`
- Inspect: repository root and `tests/` for debug artifacts

**Step 1: Add a deterministic density-estimation smoke test**

Create a fixed batch with non-degenerate conditioning values and a shifted/scaled transformed coordinate. Train with:

```text
z = layer.forward(x)
L = mean(0.5 * sum(z^2) - log_det)
grad_z = z / N
set_log_det_gradient(-1.0 / N)
backward(grad_z)
update_weights(lr)
```

Use a small learning rate, 100–300 steps, and require:

- finite loss and parameters throughout;
- final NLL at least 20% below initial NLL;
- at least one conditioner parameter changes.

This tests the actual density objective rather than only supervised output fitting.

**Step 2: Add state and lifecycle tests**

Assert:

```cpp
parameters().size() == 4;
parameters().size() == gradients().size();
params[i]->rows == grads[i]->rows;
params[i]->cols == grads[i]->cols;
zero_grad() clears every entry exactly;
update_weights(lr) moves parameters after a real backward;
backward before forward throws;
grad_output shape mismatch throws;
```

**Step 3: Publish the umbrella include**

```cpp
#include "layers/generative/affine_coupling.h"
#include "layers/generative/coupling_layer.h"
#include "layers/generative/neural_spline_flow.h"
```

**Step 4: Register aggregate Make targets**

Add `$(BUILD_DIR)/test_neural_spline_flow` to `tests` and:

```make
	@echo "=== Running Neural Spline Flow Tests ===" && ./$(BUILD_DIR)/test_neural_spline_flow
```

to `run_tests`. Anchor patches on unique target headers and inspect `git diff Makefile` immediately.

**Step 5: Update the expansion queue**

Remove only the final Neural Spline Flows bullet from `## Ideas` and add the first `## Done` entry documenting:

- exact source/test files;
- batch-aware `Dense -> tanh -> Dense` coupling conditioner;
- constrained RQS formula and identity tails;
- stable analytical inverse;
- forward-mode local analytical Jacobian and full determinant chain;
- focused check count, density loss change, maximum FD errors, and mutation evidence.

**Step 6: Run cleanup pass**

Inspect root and `tests/` for `debug_*`, `*.bak`, and `ref_*.cpp`; read candidates and check Makefile/history before deletion. Modify `.gitignore` only if a new artifact class appears.

**Step 7: Run focused and full verification**

Because public headers change and the Makefile lacks generated dependency files:

```bash
make clean
make build/test_neural_spline_flow
./build/test_neural_spline_flow
make tests
make run_tests
```

Expected: focused suite passes, all aggregate binaries build, and every registered test executable exits zero.

**Step 8: Review exact state**

```bash
git status --short
git diff --check
git diff -- include/nn/layers/generative/neural_spline_flow.h \
  include/nn/layers/generative/neural_spline_flow.cpp \
  tests/test_neural_spline_flow.cpp include/nn/nn.h Makefile \
  EXPANSION_QUEUE.md
git status --ignored --short
```

Verify only intended files changed and no mutation/debug code remains.

**Step 9: Commit atomically**

Commit the plan separately:

```bash
git add docs/plans/2026-07-14_150724-neural-spline-flows.md
git commit -m "docs: add neural spline flow implementation plan"
```

Then commit the feature:

```bash
git add include/nn/layers/generative/neural_spline_flow.h \
  include/nn/layers/generative/neural_spline_flow.cpp \
  tests/test_neural_spline_flow.cpp include/nn/nn.h Makefile \
  EXPANSION_QUEUE.md
git commit -m "feat(generative): add neural spline flow coupling"
```

**Step 10: Push and verify the remote**

```bash
git push origin master
git rev-parse HEAD
git rev-parse origin/master
```

Expected: local `HEAD` and `origin/master` match. Never force-push a rejection.
