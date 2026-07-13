# ConvBERT-Style Convolution-Augmented Attention Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a production-quality `ConvBertLayer` that learns a gated local convolution branch alongside global multi-head self-attention and combines both with a learnable scalar.

**Architecture:** The layer consumes a dynamic-length `(tokens, d_model)` tensor. The global branch uses standard multi-head scaled dot-product self-attention; the local branch computes `Dense(d_model, 2*d_model) -> GLU -> depthwise Conv1D(kernel_size)`. The output is `alpha * attention + (1-alpha) * convolution`, with full analytical BPTT through every projection, softmax, GLU gate, depthwise kernel, and `alpha`.

**Tech Stack:** C++17, the repository's `Tensor`, `Layer`, and `Dense` abstractions, Make, centered finite-difference gradient tests, no external dependencies.

**Reference:** Ren et al., “ConvBERT: Improving BERT with Span-based Dynamic Convolution,” arXiv:2008.02496. This implementation intentionally follows the repository queue's simplified ConvBERT-style contract (parallel local/global branches plus learnable `alpha`), rather than reproducing the paper's full span-dynamic convolution generator.

---

### Task 1: Establish the public contract with a failing test

**Objective:** Define the user-visible constructor, shape rules, branch mixing semantics, and defensive validation before production code exists.

**Files:**
- Create: `tests/test_conv_bert.cpp`
- Modify: `Makefile:60-380`
- Test: `tests/test_conv_bert.cpp`

**Step 1: Write failing test**

Create a small check-based harness that includes the not-yet-existing header and starts with these behaviors:

```cpp
#include "nn/layers/attention/conv_bert.h"

ConvBertLayer layer(/*d_model=*/4, /*kernel_size=*/3, /*num_heads=*/2);
CHECK(layer.d_model() == 4, "stores d_model");
CHECK(layer.kernel_size() == 3, "stores kernel_size");
CHECK(layer.num_heads() == 2, "stores num_heads");
CHECK_NEAR(layer.alpha(), 0.5, 0.0, "starts with balanced branches");

Tensor x(5, 4);
Tensor y = layer.forward(x);
CHECK(y.rows == 5 && y.cols == 4, "preserves (tokens, d_model)");
```

Add constructor rejection tests for zero dimensions, `d_model % num_heads != 0`, and even kernels. Add a `set_alpha()` check because branch-isolation tests need an explicit, stable API.

**Step 2: Run test to verify failure**

Run:

```bash
make build/test_conv_bert
```

Expected: FAIL at compilation because `nn/layers/attention/conv_bert.h` does not exist.

**Step 3: Add only build registration needed for the focused test**

Add:

```make
$(BUILD_DIR)/test_conv_bert: $(LIB_OBJS) $(BUILD_DIR)/test_conv_bert.o
	$(CXX) $^ -o $@
```

Do not add the test to `run_tests` until the implementation is green.

**Step 4: Re-run RED**

Run:

```bash
make build/test_conv_bert
```

Expected: the same missing-header failure, proving the test target reaches the intended source file.

**Step 5: Commit**

Do not commit while RED. The test and implementation form one feature commit after GREEN.

---

### Task 2: Implement constructor, forward branches, and branch mixing

**Objective:** Make constructor/shape/forward tests pass with the smallest complete forward path.

**Files:**
- Create: `include/nn/layers/attention/conv_bert.h`
- Create: `include/nn/layers/attention/conv_bert.cpp`
- Test: `tests/test_conv_bert.cpp`

**Step 1: Define the exact API**

```cpp
class ConvBertLayer : public Layer {
public:
    ConvBertLayer(size_t d_model, size_t kernel_size = 7,
                  size_t num_heads = 1, double alpha = 0.5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "ConvBertLayer"; }

    size_t d_model() const;
    size_t kernel_size() const;
    size_t num_heads() const;
    size_t head_dim() const;
    double alpha() const;
    void set_alpha(double value);

    const Tensor& last_attention_output() const;
    const Tensor& last_convolution_output() const;
};
```

Store five `Dense` layers (`q`, `k`, `v`, attention output, and `2*d_model` GLU projection), depthwise weights `(d_model, kernel_size)`, depthwise bias `(1, d_model)`, matching gradient tensors, and `alpha` / `grad_alpha` as `(1,1)` tensors so they participate in the repository's parameter API.

**Step 2: Implement standard multi-head attention forward**

For each head and token pair:

```cpp
score[h,t,s] = dot(q[t,h_slice], k[s,h_slice]) / sqrt(head_dim);
prob[h,t,:] = stable_softmax(score[h,t,:]);
context[t,h_slice] = sum_s prob[h,t,s] * v[s,h_slice];
attention_out = out_projection.forward(context);
```

Cache projected Q/K/V, softmax probabilities, and pre-output-projection context. Use a `(num_heads * tokens, tokens)` cache so dynamic sequence lengths require no constructor-time `seq_len`.

**Step 3: Implement GLU plus depthwise convolution forward**

```cpp
pre = conv_projection.forward(input);       // (N, 2D)
a = pre[:, 0:D];
gate = sigmoid(pre[:, D:2D]);
glu = a * gate;
conv[t,c] = bias[c] + sum_j depthwise_weight[c,j] * glu[t+j-pad,c];
```

Use zero padding and require odd `kernel_size`, preserving `N` exactly. Cache `pre`, `gate`, and `glu` before any mutation.

**Step 4: Mix branches**

```cpp
output = alpha * attention_out + (1.0 - alpha) * conv_out;
```

Expose branch-output cache accessors so tests can establish:

```cpp
mixed == alpha * last_attention_output()
       + (1-alpha) * last_convolution_output();
```

**Step 5: Run focused tests**

Run:

```bash
make build/test_conv_bert && ./build/test_conv_bert
```

Expected: constructor, validation, shape, finite-output, branch-difference, and convex-combination checks pass. Gradient tests should still fail or be absent until Task 3.

---

### Task 3: Add strict analytical backward with non-degenerate finite-difference tests

**Objective:** Verify input, attention, convolution, and branch-mixing gradients before training tests.

**Files:**
- Modify: `tests/test_conv_bert.cpp`
- Modify: `include/nn/layers/attention/conv_bert.cpp`
- Test: `tests/test_conv_bert.cpp`

**Step 1: Add failing gradient tests**

Use `d_model=4`, `num_heads=2`, `kernel_size=3`, `N=4`, asymmetric input/target values, and loss:

```cpp
L = 0.5 * sum((output - target)^2);
grad_output = output - target;
```

Check centered finite differences for:

1. every input cell;
2. representative Q/K/V/output projection weights;
3. representative GLU-projection weights from both the linear and gate halves;
4. every depthwise-kernel element;
5. depthwise bias;
6. the learnable `alpha` scalar.

Use a combined tolerance:

```cpp
bool close_grad(double analytical, double numerical) {
    const double scale = std::max({1e-12, std::fabs(analytical), std::fabs(numerical)});
    return std::fabs(analytical - numerical) <= std::max(1e-7, 1e-4 * scale);
}
```

**Step 2: Run test to verify RED**

Run:

```bash
make build/test_conv_bert && ./build/test_conv_bert
```

Expected: gradient assertions fail because backward is not implemented.

**Step 3: Implement branch split and alpha derivative**

```cpp
grad_attention = alpha * grad_output;
grad_convolution = (1.0 - alpha) * grad_output;
grad_alpha = sum(grad_output * (attention_out - convolution_out));
```

**Step 4: Implement local branch backward**

Depthwise convolution:

```cpp
grad_dw[c,j] += grad_conv[t,c] * glu[src,c];
grad_glu[src,c] += grad_conv[t,c] * dw[c,j];
grad_dw_bias[c] += grad_conv[t,c];
```

GLU:

```cpp
grad_a = grad_glu * gate;
grad_gate_pre = grad_glu * a * gate * (1.0 - gate);
grad_pre = concat(grad_a, grad_gate_pre);
grad_input_conv = conv_projection.backward(grad_pre, 0.0);
```

**Step 5: Implement global branch backward**

Apply, in order:

```cpp
grad_context = out_projection.backward(grad_attention, 0.0);
grad_prob[t,s] = dot(grad_context[t], v[s]);
grad_v[s] += prob[t,s] * grad_context[t];
grad_score[t,s] = prob[t,s] * (grad_prob[t,s] - sum_u prob[t,u]*grad_prob[t,u]);
grad_q[t] += scale * grad_score[t,s] * k[s];
grad_k[s] += scale * grad_score[t,s] * q[t];
```

Then call Q/K/V `Dense::backward()` and sum all four input contributions:

```cpp
grad_input = grad_input_conv + grad_input_q + grad_input_k + grad_input_v;
```

**Step 6: Run focused gradient suite**

Run:

```bash
make build/test_conv_bert && ./build/test_conv_bert
```

Expected: all analytical-vs-FD checks pass with relative error below `1e-4` outside the absolute-noise regime.

---

### Task 4: Verify training behavior and test non-vacuousness

**Objective:** Prove the layer learns, manages parameter state correctly, and that tests fail under known-bad mutations.

**Files:**
- Modify: `tests/test_conv_bert.cpp`
- Modify: `include/nn/layers/attention/conv_bert.cpp` only if a test reveals a root-cause defect
- Test: `tests/test_conv_bert.cpp`

**Step 1: Add state-management tests**

Assert:

```cpp
parameters().size() == gradients().size();
parameters()[i]->shape == gradients()[i]->shape;
zero_grad() makes every gradient entry exactly zero;
update_weights(lr) moves alpha and at least one tensor parameter when gradients are nonzero;
```

**Step 2: Add end-to-end learning test**

Train on a fixed non-square `(N=5,D=4)` regression target for 100–200 updates using:

```cpp
layer.zero_grad();
out = layer.forward(input);
grad = out - target;
layer.backward(grad, 0.0);
layer.update_weights(lr);
```

Expected: final `0.5*L2` loss is lower than initial loss and all parameters remain finite.

**Step 3: Verify multi-head and default-kernel paths**

Run `D=4,H=2,K=3` and a separate `D=8,H=2,K=7,N=8` forward/backward smoke case. Confirm dynamic `N` does not require reconstructing the layer.

**Step 4: Mutation-test alpha gradient**

Temporarily replace:

```cpp
grad_alpha += grad_output * (attention_out - convolution_out);
```

with zero. Rebuild and run the focused suite.

Expected: the alpha finite-difference assertion fails. Restore the correct line immediately.

**Step 5: Mutation-test depthwise gradient**

Temporarily skip one depthwise-kernel accumulation. Re-run the focused suite.

Expected: the depthwise finite-difference assertion fails. Restore immediately.

**Step 6: Re-run GREEN after restoring mutations**

Run:

```bash
make build/test_conv_bert && ./build/test_conv_bert
```

Expected: all focused checks pass with pristine output.

---

### Task 5: Publish the layer through the library and queue

**Objective:** Wire the tested layer into the public build, move the queue item to Done, and verify no regressions.

**Files:**
- Modify: `include/nn/nn.h:87-108`
- Modify: `Makefile:60-380`
- Modify: `EXPANSION_QUEUE.md:6-16`
- Test: full repository suite

**Step 1: Add public include**

```cpp
#include "layers/attention/conv_bert.h"
```

Place it adjacent to other attention modules.

**Step 2: Register full test targets**

Add `$(BUILD_DIR)/test_conv_bert` to `tests` and:

```make
	@echo "=== Running ConvBERT Tests ===" && ./$(BUILD_DIR)/test_conv_bert
```

to `run_tests`.

**Step 3: Update queue atomically**

Remove the ConvBERT bullet from `## Ideas` and append a concise `## Done` entry documenting:

- exact files;
- global self-attention + local GLU/depthwise-convolution architecture;
- full backward paths;
- focused pass count and max gradient errors;
- mutation-test evidence.

Do not modify unrelated queue entries.

**Step 4: Run focused and full verification**

Run:

```bash
make build/test_conv_bert
./build/test_conv_bert
make tests
make run_tests
```

Expected: focused suite passes, all repository test binaries build, and every test command exits zero.

**Step 5: Inspect exact diff and artifacts**

Run:

```bash
git status --short
git diff --check
git diff -- include/nn/layers/attention/conv_bert.h \
  include/nn/layers/attention/conv_bert.cpp tests/test_conv_bert.cpp \
  include/nn/nn.h Makefile EXPANSION_QUEUE.md
git status --short | grep -E '(^|/)(debug_|ref_.*\.cpp)|\.bak$' && exit 1 || true
```

Expected: only intended files changed; no whitespace errors or debug artifacts.

**Step 6: Commit**

```bash
git add include/nn/layers/attention/conv_bert.h \
  include/nn/layers/attention/conv_bert.cpp \
  tests/test_conv_bert.cpp include/nn/nn.h Makefile EXPANSION_QUEUE.md
git commit -m "feat(attention): add ConvBERT-style mixed attention"
```

**Step 7: Push and verify**

```bash
git push origin master
git rev-parse HEAD
git rev-parse origin/master
```

Expected: local and remote SHAs match.
