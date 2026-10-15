# Virtual Adversarial Training Implementation Plan

> **For Hermes:** use subagent-driven-development to execute task-by-task.
> This plan follows Template A (there is code to write, with a clear first task).

**Goal:** ship `include/nn/utils/vat.{h,cpp}` — Virtual Adversarial Training
(Miyato et al., ICLR 2018) — the label-free robustness regulariser that closes
the third and last corner of the repo's robustness triangle: `adversarial.{h,cpp}`
SEARCHES for a worst-case delta, `randomized_smoothing.{h,cpp}` BOUNDS one, and
VAT regularises a model to be smooth where neither an attack nor a certificate
has to be computed at test time.

**Architecture:** model-free in exactly the same sense as `adversarial.h`. The
only interface to a network is two callbacks — a `LogitsFn` mapping an input to
logits, and the repo's existing `GradientFn` returning dL/d(input) — so the
entire power-iteration core is testable with a synthetic oracle and no network,
and the VAT loss composes with the already-shipped PGD attacker for the CRWD-
style hybrid without any new machinery.

**Tech stack:** C++17, the repo's hand-rolled `Tensor` (double precision),
`-O2 -Wall -Wextra -march=native`, an `std::function`-injected direction source
for determinism.

---

## 0. Citation correction (do this first; it is not optional)

The queue proposed `VAT, Miyato et al. arXiv:1704.03941`. **That ID is wrong**,
and it is the fifth wrong ID this repo has caught:

| Queue ID | What arXiv actually returns |
|---|---|
| `1704.03941` | *"Analysis of pacemaker activity in a two-component model of some brainstem neurons"* |
| `1704.03976` | *"Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning"* ✅ |

The correct ID is **arXiv:1704.03976**, verified against `arxiv.org/abs/`.

**The queue's other citation, `CRWD, Stutz et al. arXiv:2007.01174`, is ALSO
wrong AND THE PAPER COULD NOT BE FOUND AT ALL.** `2007.01174` is *"Robust Inverse
Reinforcement Learning under Transition Dynamics Mismatch"*. An exhaustive search
(arXiv title query → 0 results; full NeurIPS 2022 proceedings index, 2834 entries
→ 0 hits; NeurIPS virtual-site JSON, 3490 records → 0; Semantic Scholar
title-match → "not found"; OpenReview, OpenAlex, Crossref, CORE → 0; the TSRML
2022 workshop accepted list → 0) found no paper by that title anywhere. The
author list in the queue includes "Sebastian EWSCLN", which is not a human name
and looks generated. **CRWD is therefore DROPPED from this plan and must not be
implemented from that citation.** See `EXPANSION_QUEUE.md` for the record.

---

## 1. Pinned semantics (from the paper + the official `takerum/vat_tf` impl)

Source of truth: `takerum/vat_tf` `vat.py:33-59` and `layers.py:131-148`, plus
paper Eqs. (5), (6), (12)–(17) and Algorithm 1.

| # | Rule | Why it matters |
|---|------|----------------|
| D1 | KL is `KL(clean ‖ perturbed)` — **clean is the FIRST argument** | Not symmetric; measured 0.32342102 vs 0.32363019 on the fixture. The reference's parameter names (`q_logit`) are misleading — the ORDER is what matters. |
| D2 | The clean prediction is **detached** (`tf.stop_gradient`, `vat.py:55`) | The outer gradient flows only through the perturbed branch. Getting this wrong silently accumulates a second gradient path. |
| D3 | `r_probe = xi · normalise(d)` is formed **before** the gradient step; `r_vadv = eps · normalise(g)` **after** | The chain rule picks up a `xi` factor the final normalise then discards. |
| D4 | `∇ KL` w.r.t. `r` is **exactly 0 at r = 0** | KL is quadratic in `r`. Consequence: a finite-difference gradient check taken at `r = 0` is vacuous — see §5c-zeroth. The oracle's first FD probe failed for exactly this reason. |
| D5 | The clean `p_bar` is **cached**, never recomputed at the probe | Recomputing gives `p ≡ q`, so `∇KL(p‖p) = 0` exactly: `‖g‖ = 0.0` measured. The perturbation would be pure noise. |
| D6 | `‖r_vadv‖ = eps · 0.9999997`, **not** `eps` | The `1e-6` inside `sqrt(1e-6 + Σd²)` (a second stabiliser, separate from the `1e-12` one) biases the norm below 1. Measured `7.9999983496010874`. |
| D7 | The VAT loss is **monotone non-decreasing in eps** | A larger virtual-adversarial radius can only increase the inner max. Measured across eps ∈ {0.1 … 8.0}. |
| D8 | Entropy minimisation is **NEGATIVE entropy, added** (`layers.py:145-148`) | `entropy_y_x` returns `-mean Σ p log p`, so adding it under gradient *descent* sharpens predictions. Entropy minimisation is an **optional extra term**, never a replacement for the VAT term. |
| D9 | The output-layer weight factor is **load-bearing** in the backward | `dlogit_k/dx = Σ_h W2[k,h]·sech²(h)·W1[h,c]`. Dropping `W2` collapses the gradient to ~1e-17 (measured `-9.305e-18` vs FD `-3.099e-09`) because `dl[k=1] = -dl[k=0]` for `K=2` and the hidden terms then cancel. Found by the oracle's FD self-check. |
| D10 | Normalisation is **per-sample over every non-batch axis** | Not a global norm over the batch. |
| D11 | The `1e-12 + max|d|` step is an **overflow guard that cancels** | Mathematically a no-op between the two divides; `1e-6` in the second one is not. |

---

## 2. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Direction is **injectable** (`DirectionFn`), defaulting to a seeded Gaussian | The paper initialises `d` iid-Gaussian, but `std::normal_distribution` is not bit-reproducible across libstdc++ versions, so a test asserting exact digits on a Gaussian is a flaky test. Injecting lets the suite pin exact numbers and lets a caller supply its own prior. |
| D2 | Follow **TF's** normalise-at-top-of-loop form, not Chainer's normalise-at-bottom | TF is the official implementation and what `num_power_iterations > 1` should match. At `K = 1` the two are provably identical (cosine = 1.0). |
| D3 | `xi` validated `> 0` but `xi = 1` **not** forbidden | `xi` is a finite-difference step for a second-order approximation requiring `xi ≪ eps`; `xi = 1` voids the linearisation. The header documents it; a hard throw would be surprising. |
| D4 | Expose `kl_divergence`, `normalise_direction`, `virtual_adversarial_perturbation`, `vat_loss`, `entropy_negative`, and a `VATTrainer` | Each is independently testable; the training loop is the only piece needing a `Model`. |
| D5 | Default `eps = 8.0`, `xi = 1e-6`, `power_iterations = 1` | The reference's own flag defaults (`vat.py:10-12`) and the paper's CIFAR-10 setting. |
| D6 | No `alpha` weight in `vat_loss` | The reference **hard-codes** `nll_loss + vat_loss` (`vat.py:68`); there is no scale flag. `alpha` lives in the trainer config only. |

---

## 3. File layout

- Create: `include/nn/utils/vat.h` — declarations + the D1..D11 semantics block.
- Create: `include/nn/utils/vat.cpp` — implementation.
- Create: `tests/test_vat.cpp` — the suite (written FIRST, per TDD).
- Modify: `include/nn/nn.h` — umbrella include.
- Modify: `Makefile` — `build/test_vat` rule, `tests:` prerequisite, `run_tests` line.

---

## 4. Tasks

### Task 1: RED — write the failing test suite

**Files:** Create `tests/test_vat.cpp`.

**Step 1:** Write the suite with **every numeric expectation transcribed from
`~/.hermes/cache/scratch/vat_np_oracle.out`** (numpy float64, self-checked
against finite differences). Do not hand-compute anything.

**Step 2: Run it, confirm RED.**
```
make build/test_vat
```
Expected: compile error — `nn/utils/vat.h: No such file or directory`.

**Step 3: Commit the RED test.**
```bash
git add tests/test_vat.cpp
git commit -m "test(utils): RED suite for virtual adversarial training (oracle-derived)"
```

### Task 2: GREEN — minimal implementation

**Files:** Create `include/nn/utils/vat.h`, `include/nn/utils/vat.cpp`.

**Step 1:** Implement, in order: `log_softmax_rows`, `softmax_rows`,
`kl_divergence` (D1), `normalise_direction` (D6/D10/D11),
`dkl_dlogits` (D2), `virtual_adversarial_perturbation` (D3/D5),
`entropy_negative` (D8), `vat_loss`, then `VATTrainer::train`.

**Step 2: Run the suite, confirm GREEN.**
```
make build/test_vat && ./build/test_vat
```
Expected: all checks pass.

**Step 3: Commit.**
```bash
git add include/nn/utils/vat.h include/nn/utils/vat.cpp
git commit -m "feat(utils): virtual adversarial training (Miyato arXiv:1704.03976)"
```

### Task 3: Mutation testing (non-vacuousness)

**Files:** edit `vat.cpp` temporarily; restore after each mutation.

Mutations to apply and confirm are CAUGHT:

| # | Mutation | Must be caught by |
|---|---|---|
| M1 | `dkl_dlogits`: `(p - q)/N` → `(q - p)/N` | the KL-gradient test (sign flip) |
| M2 | drop the `W2` factor in the backward (via the `GradientFn` fixture) | the input-gradient test |
| M3 | `r_vadv = eps * normalise(g)` → `normalise(g)` | the `‖r_vadv‖` test |
| M4 | cache `p_bar` → recompute at the probe | the "recomputed p_bar gives 0" test |
| M5 | `xi * normalise(d)` → `normalise(d)` (probe not scaled) | the probe-norm test |
| M6 | remove the batch mean from `kl_divergence` | the KL-value test |
| M7 | `entropy_negative` → `+entropy` | the entropy-sign test |

**Guard (learned from the adversarial session):** checksum the tree before and
after each mutation, confirm the mutated string is actually present with `grep`,
and assert the mutated object file is NOT in the link set — a link error is not
a test failure.

### Task 4: Integration

**Files:** `include/nn/nn.h`, `Makefile`.

```bash
# umbrella compiles standalone
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
Expected: no output, exit 0.

Add the `build/test_vat` rule, the `tests:` entry, and a
`=== Running VAT Tests ===` line to `run_tests`.

```bash
git add include/nn/nn.h Makefile
git commit -m "build: register test_vat (Makefile rule, tests: prereq, run_tests line) + nn.h umbrella include"
```

### Task 5: Docs

**Files:** `EXPANSION_QUEUE.md` — move VAT to `## Done` with the citation
correction and the D9 bug; record the dropped CRWD citation.
```bash
git add EXPANSION_QUEUE.md
git commit -m "docs: mark VAT Done, record the wrong-arXiv-ID correction and the dropped CRWD citation"
```

---

## 5. Verification

- [ ] `./build/test_vat` → 0 failed, deterministic across 3 reruns.
- [ ] All 7 mutations caught; no harness self-deception.
- [ ] Clean under `g++ -O1 -g -fsanitize=address,undefined`.
- [ ] Zero warnings at `-O2 -Wall -Wextra -march=native`.
- [ ] `nn.h` compiles standalone.
- [ ] `make run_tests` shows no new failures vs the recorded baseline
      (`test_neural_ode` is the only known real failure).

---

## 6. References

- Miyato, Maeda, Koyama, Ishii — *Virtual Adversarial Training: A Regularization
  Method for Supervised and Semi-Supervised Learning*, ICLR 2018,
  **arXiv:1704.03976**.
- Official implementation: `github.com/takerum/vat_tf` (`vat.py`, `layers.py`).
- Oracle: `~/.hermes/cache/scratch/vat_np_oracle.py` → `vat_np_oracle.out`.