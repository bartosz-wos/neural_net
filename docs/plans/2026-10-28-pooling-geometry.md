# Fix: MaxPool2D / AvgPool2D call sites pass the stride into the H_in/W_in slots

> **For Hermes:** implement this plan task-by-task with strict TDD (RED -> GREEN).

**Goal:** Make `MaxPool2D` and `AvgPool2D` constructible with the spatial
dimensions they actually pool over, so that `AlexNet` and `SqueezeNet` — both
shipped public layers — can be instantiated.

**Architecture:** `include/nn/layers/pooling/pool_layer.h:13` declares

```cpp
MaxPool2D(int kH, int kW, int H_in, int W_in, int stride_h = 1, int stride_w = 1);
```

which matches `Conv2D`'s convention exactly
(`conv_layer.h:26`: `Conv2D(int in_ch, int out_ch, int kH, int kW, H_in, W_in, stride_h, stride_w, ...)`).
Like `Conv2D`, the pooling layers are **shape-static**: `H`/`W` are baked in at
construction and `forward` asserts `input.cols % (H * W) == 0`.

**Every call site in the repo passes the STRIDE where `H_in`/`W_in` belong.**

**Tech Stack:** C++17, `include/nn/layers/pooling/pool_layer.{h,cpp}`,
`include/nn/layers/architectures/{cnn_models,squeezenet}.{h,cpp}`.

---

## The defect, measured

The ctor is `(kH, kW, H_in, W_in, stride_h, stride_w)`. The call sites are:

| Site | Written | Means | Should mean |
|---|---|---|---|
| `cnn_models.cpp:8` | `pool_(2, 2, pool_stride, pool_stride)` | k=2×2, **H_in=pool_stride, W_in=pool_stride**, stride=1 | k=2×2, H_in/W_in = feature map, **stride=pool_stride** |
| `cnn_models.cpp:119` | `pool1_(2, 2, 2, 2), pool2_(2, 2, 2, 2)` | k=2×2, H_in=2, W_in=2, stride=1 | k=2×2, **stride=2**, H_in/W_in = 28×28 / 12×12 |
| `cnn_models.cpp:198` | `pool1_(3, 3, 2, 2), pool2_(3, 3, 2, 2), pool3_(3, 3, 2, 2)` | k=3×3, H_in=2, W_in=2, stride=1 | k=3×3, **stride=2**, H_in/W_in = 56/27/13 |
| `squeezenet.cpp:110,112` | `pool1_(3, 3, 2, 2)`, `pool_final_(3, 3, 2, 2)` | k=3×3, H_in=2, W_in=2, stride=1 | k=3×3, **stride=2**, H_in/W_in = 55 / 13 |

### WHY IT NEVER FAILED BEFORE

Two independent reasons, and both are worth recording:

1. **`MaxPool2D` was never constructed on a working path.** The one layer that
   uses 2-D pooling for real — `SpatialTransformer` — does NOT use the class at
   all: it hand-rolls `maxpool2d_forward` / `maxpool2d_backward`
   (`spatial_transformer.cpp:239,260`) precisely because the class could not
   follow a changing spatial size.

2. **The classes that DO construct it could not be instantiated anyway**, because
   they were abstract (the defect fixed in
   `docs/plans/2026-10-28-abstract-layer-accessors.md`). `AlexNet` and
   `SqueezeNet` were therefore never constructed, so the throw was never
   reached.

The two defects **mask each other**: fixing the abstractness alone surfaced the
pool bug immediately (the new suite aborted with
`MaxPool2D: output dimensions non-positive`). That is the fingerprint of a
latent bug that was hidden behind an earlier one, and it is why this is a
separate commit rather than a silent extra change.

### MEASURED CONSEQUENCE

With `H_in = W_in = 2` and `kH = kW = 3` (the `AlexNet`/`SqueezeNet` case):

```
H_out = (H_in - kH) / stride_h + 1 = (2 - 3) / 1 + 1 = 0
```

so the ctor throws `std::invalid_argument("MaxPool2D: output dimensions non-positive")`.
With `H_in = W_in = 2` and `kH = kW = 2` (the `LeNet5` case) it does not
throw, but it computes `H_out = 1` — a pool that collapses the whole feature
map to a single spatial position. `LeNet5` constructs without error and would
then fail later, at `forward`, on the shape assertion
`input.cols % (H * W) != 0`.

---

## Task 1: RED — the pooling-geometry test

**Objective:** A test that pins the spatial geometry of the pooling layers and
fails against the current call sites.

**Files:**
- Create: `tests/test_pool_geometry.cpp`
- Modify: `Makefile`

**Step 1: Write the failing test**

```cpp
// tests/test_pool_geometry.cpp
#include "nn/nn.h"
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace nn;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}
```

The test has two halves.

**Half A — the layer's own contract.** A `MaxPool2D` constructed with
`(kH, kW, H_in, W_in, stride_h, stride_w)` must:

- expose `H == H_in` and `W == W_in`,
- compute `H_out == (H_in - kH) / stride_h + 1` and likewise for `W`,
- actually pool: a `(1, C*H*W)` input must come back as `(1, C*H_out*W_out)`,
- and the pooling must be CORRECT: with a 4×4 input, a 2×2 kernel and
  stride 2, the output must be the 2×2 max of each 2×2 block.

**Half B — the call sites.** Each architecture's pools must be built with the
spatial dims of the stage they pool, so that the whole network's spatial chain
is consistent. The chain is the spec:

```
LeNet5    : 32x32 -> conv(5x5,pad2) -> 32x32 -> pool(2x2,s2) -> 16x16
                      -> conv(5x5,pad0) -> 12x12 -> pool(2x2,s2) ->  6x6
AlexNet   : 227x227 -> conv(11,s4,pad3) -> 56x56 -> pool(3x3,s2) -> 27x27
                      -> conv(5,s1,pad2)  -> 27x27 -> pool(3x3,s2) -> 13x13
                      -> conv(3,s1,pad1)x3 -> 13x13 -> pool(3x3,s2) ->  6x6
SqueezeNet: 224x224 -> conv(7,s2,pad3)  -> 111x111 -> pool(3x3,s2) -> 55x55
```

**Step 2: Run to verify failure**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -fsyntax-only tests/test_pool_geometry.cpp
```

Expected: the file compiles (the layers are concrete now), but the RUNTIME
assertions fail:

- `AlexNet` / `SqueezeNet` construction throws
  `MaxPool2D: output dimensions non-positive` — the test must catch that
  (wrap each construction in a try/catch and report a FAIL, never abort).
- `LeNet5`'s pool reports `H == 2` where the spec says `28`.

**A CONSTRUCTION THAT THROWS MUST NOT ABORT THE SUITE.** The first draft of the
instantiability test aborted with exit 134 and no summary line, which is the
"a probe ran OUTSIDE a try/catch" defect the repo's own notes record. Every
construction in this file goes through a helper that catches and reports.

**Step 3: Commit the RED test**

```bash
git add tests/test_pool_geometry.cpp Makefile
git commit -m "test(pooling): RED test for MaxPool2D/AvgPool2D spatial geometry"
```

---

## Task 2: GREEN — fix the call sites

**Objective:** Build each pool with the spatial dims of the stage it pools,
and pass the stride in the stride slot.

**Files:**
- Modify: `include/nn/layers/architectures/cnn_models.cpp`
  (`VGGBlock`, `LeNet5`, `AlexNet` constructors)
- Modify: `include/nn/layers/architectures/squeezenet.cpp`
  (`SqueezeNet` constructor)

**Step 1: Fix `VGGBlock`**

`VGGBlock`'s conv stack is built at a fixed `224, 224` input
(`cnn_models.cpp:11`: `convs_.emplace_back(ch, filters[i], 3, 3, 224, 224, 1, 1, 1, 1)`).
The pool must therefore be built for a 224×224 input with the requested stride:

```cpp
: use_bn_(use_bn), pool_(2, 2, 224, 224, pool_stride, pool_stride), n_layers_(n_layers) {
```

**Step 2: Fix `LeNet5`**

Chain: conv1 (5×5, pad 2) on 32×32 -> 32×32 -> pool -> 16×16;
conv2 (5×5, pad 0) on 16×16 -> 12×12 -> pool -> 6×6.

```cpp
pool1_(2, 2, 32, 32, 2, 2), pool2_(2, 2, 12, 12, 2, 2),
```

**Step 3: Fix `AlexNet`**

Chain: conv1 (11×11, s4, pad3) on 227×227 -> 56×56 -> pool -> 27×27;
conv2 (5×5, s1, pad2) on 27×27 -> 27×27 -> pool -> 13×13;
conv3-5 (3×3, s1, pad1) on 13×13 -> 13×13 -> pool -> 6×6.

```cpp
pool1_(3, 3, 56, 56, 2, 2), pool2_(3, 3, 27, 27, 2, 2), pool3_(3, 3, 13, 13, 2, 2),
```

**Step 4: Fix `SqueezeNet`**

Chain: conv1 (7×7, s2, pad3) on H_in×W_in -> ceil((H_in-7)/2)+1; pool -> ...
The fire modules do NOT change the spatial size, and a pool follows every
second fire module. The pool1 input is the conv1 output size and the
pool_final input is the size after the last fire module.

**Step 5: Verify**

```bash
make build/test_pool_geometry && ./build/test_pool_geometry
```
Expected: all PASS.

**Step 6: Commit**

```bash
git add include/nn/layers/architectures/cnn_models.cpp include/nn/layers/architectures/squeezenet.cpp
git commit -m "fix(pooling): build MaxPool2D with the spatial dims it pools over"
```

---

## Task 3: Verification

```bash
make -j$(nproc) run_tests          # expect only the pre-existing test_neural_ode failure
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
    tests/test_pool_geometry.cpp $(make -s print-lib-objs | tr '\n' ' ') \
    -o /tmp/test_pool_geom_asan && /tmp/test_pool_geom_asan
```

---

## Out of scope (recorded, not done)

- **`MaxPool2D` cannot follow a CHANGING spatial size** — the layer is
  shape-static, exactly like `Conv2D`. `SpatialTransformer` works around this
  by hand-rolling its own pool. Making the pooling layers shape-dynamic is a
  different design decision and is not needed to fix the call sites.
- **The `VAE`'s KL is a SUM while its reconstruction loss is a MEAN**
  (`vae.cpp:48` divides by `input.rows`; `vae.cpp:53` does not divide at all).
  A separate defect, a separate commit.
- **`SqueezeNet::backward`'s comment block** says the flatten output "wasn't
  stored or used" — a separate correctness question, not a construction
  question.
