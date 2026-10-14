# Adversarial Robustness (FGSM / L-inf PGD) — Implementation Plan

**Goal:** Implement `include/nn/utils/adversarial.{h,cpp}` — ℓ∞-bounded adversarial
example generation (FGSM, multi-step PGD, randomised-start PGD) and an
adversarial-training loop, verified against real PyTorch.

**Architecture:** Model-free *attack core* operating on an arbitrary
"gradient oracle" callback, plus a thin `Model`-driven adapter. This mirrors the
established convention in this repo (knowledge distillation, temperature scaling,
Grad-CAM are all model-free) and it is what makes the attack testable without
constructing a network.

**Tech Stack:** C++17, `-O2 -march=native -Wall -Wextra`, hand-rolled `Tensor`,
`Model`/`Layer`, `CrossEntropyLoss`. Oracle: PyTorch 2.14.1+cpu (float64).

---

## Where the math came from (read, not remembered)

1. **Paper** — Madry, Makelov, Schmidt, Tsipras, Vladu, "Towards Deep Learning
   Models Resistant to Adversarial Attacks", arXiv:1706.06083, §2.1, verbatim:
   - `x + ε sgn(∇_x L(θ, x, y))` for FGSM
   - `x_{t+1} = Π_{x+S} ( x_t + α sgn(∇_x L(θ, x, y)) )` for PGD
   - the attack is a **maximisation** — "a simple one-step scheme for maximizing
     the inner part of the saddle point formulation" — so the gradient is
     **ASCENT**: `+ ε sgn(∇)`, never `-`.
2. **Reference implementation** — `MadryLab/mnist_challenge/pgd_attack.py`,
   `LinfPGDAttack.perturb`, read in full. It settles four things the paper does
   not spell out:
   - the projection is centred on **`x_nat`, the ORIGINAL input**, not on `x_t`;
     `x = np.clip(x, x_nat - eps, x_nat + eps)`
   - there is a **second** clip to the valid data range `[0, 1]` when
     `random_start` is on, and it is applied *inside* the loop too
   - with `random_start`, `x_0 = x_nat + U(-ε, ε)` **then clipped to [0,1]**
   - the step is applied BEFORE the projection every iteration
3. **Authors' own challenge config** — `mnist_challenge/config.json`: `epsilon:
   0.3, k: 40, a: 0.01, random_start: true`. MNIST is in [0,1] and ε = 0.3 is
   10/255-ish in spirit, so the [0,1] clamp is the image convention and is
   **opt-in**, not universal.
4. **PyTorch semantics** — `torch.sign(0.0) == 0.0` and `torch.sign(-0.0) ==
   0.0`, measured. This is why a coordinate with an exactly-zero gradient must
   NOT move at all.

---

## Semantics pinned (each gets a named test)

- **D1 — ASCENT.** The perturbation is `+ε·sgn(∇)`. A descent implementation
  produces an attack that *reduces* the loss. Tested by asserting the attack
  loss strictly increases.
- **D2 — `sign(0) = 0`, not `+1`.** `x >= 0 ? 1 : -1` moves a zero-gradient
  coordinate by a full ε. Oracle measured: the two zero coordinates on the
  mixed-sign fixture have delta **exactly 0.0**. This is the single most likely
  bug in the whole feature.
- **D3 — THE PROJECTION IS CENTRED ON `x_nat`.** Not on `x_t`, and not on the
  random start. Oracle discriminator: at ε=0.3, α=0.1, k=7 the correct answer
  gives `max|δ| = 0.3`; the wrong-centre variant gives `0.7`.
- **D4 — THE STEP IS APPLIED BEFORE THE PROJECTION.** Reordering changes every
  intermediate iterate.
- **D5 — THE DATA-RANGE CLAMP IS OPT-IN AND ORDERED LAST.** The paper's formula
  has no clamp; the reference impl clamps to [0,1] because MNIST pixels live
  there. Both are available; the clamp is never applied implicitly.
- **D6 — `ε` AND `α` ARE INDEPENDENT.** ε bounds the ball radius, α is the step.
  The paper's own config uses ε=0.3, α=0.01 — a 30× ratio. Defaulting α=ε is
  the classic error.
- **D7 — `k` IS THE NUMBER OF GRADIENT EVALUATIONS.** `k=0` returns the input
  unchanged (identity attack), not the perturbation.
- **D8 — GRADIENT W.r.t. INPUT, NOT W.r.t. PARAMETERS.** The repo already returns
  `dL/d(input)` from `Layer::backward`; the attack must not update any weights.
  Verified by snapshotting all parameters before and after an attack.

---

## Files

- Create: `include/nn/utils/adversarial.h`
- Create: `include/nn/utils/adversarial.cpp`
- Create: `tests/test_adversarial.cpp`
- Modify: `include/nn/nn.h` (umbrella `#include`)
- Modify: `Makefile` (build rule, `tests:` prereq, `run_tests` line)

---

## Tasks

### Task 1 — RED: oracle-derived test suite
Write `tests/test_adversarial.cpp` with every expectation transcribed from
`adv_oracle2.out` (real torch float64). Run `make build/test_adversarial`;
expect a COMPILE failure (`adversarial.h` does not exist). That is the RED.

### Task 2 — GREEN: the attack core
Header + `linf_fgsm`, `linf_pgd`, `sign_tensor`, `l2_fgsm`. Run the suite.

### Task 3 — Mutation harness
Prove the suite is non-vacuous: flip the ascent sign, flip `sign(0)` to
`x >= 0`, re-centre the projection on `x_t`, drop the step-before-project
order, default α to ε. Every mutation must be caught.

### Task 4 — Adversarial training
`adversarial_train` loop; assert PGD training raises adversarial accuracy
relative to standard training on a separable-ish task.

### Task 5 — Wire up, full suite, commit
`nn.h` umbrella standalone compile; `make run_tests`; commit + push.