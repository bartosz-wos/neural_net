# muP (Maximal Update Parametrization) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship parameter-width scaling (muP) + the optimizer-side scaling it requires, so
hyperparameters tuned at one width transfer to another width unchanged.

**Architecture:** A model-free header `include/nn/utils/mup.{h,cpp}`. Every core is a free
function over plain `size_t`/`double` — no `Tensor`, no `Model`, no optimizer. That is the
shape `utils/ptq.h`, `utils/tta.h` and `utils/ortho_penalty.h` were built with, and it makes
the whole scaling law testable with no network.

**Tech Stack:** C++17, the repo's `-O2 -march=native -Wall -Wextra`, the repo's `check()`/`check_close()` test harness.

---

## Sources (verified, not from memory)

**CITATION.** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu,
David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao — "Tensor Programs V:
Tuning Large Neural Networks via **Zero-Shot** Hyperparameter Transfer", arXiv:2203.03466.

Verified 2026-10-18 by scraping the abs page's `citation_*` meta tags:

```
citation_title  -> ['Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer']
citation_author -> ['Yang, Greg','Hu, Edward J.','Babuschkin, Igor','Sidor, Szymon',
                    'Liu, Xiaodong','Farhi, David','Ryder, Nick','Pachocki, Jakub',
                    'Chen, Weizhu','Gao, Jianfeng']
citation_date   -> ['2022/03/07']
```

**The title says "Zero-Shot", NOT "Zero-Sharpness".** An arXiv API full-text query for
`"zero-sharpness hyperparameter transfer"` returns `totalResults=0` — the phrase does not
occur in the paper. The commonly-repeated "muP = zero-sharpness" gloss is a mangling of this
title. Do not reintroduce it.

**Table 3 transcribed from the HTML table structure** (`ar5iv` render of 2203.03466), not
from the flattened text where the μP and SP columns interleave into unreadable noise:

|              | Input weights & all biases | Output weights | Hidden weights |
|--------------|---------------------------|----------------|----------------|
| Init. Var.   | `1/fan_in`                | `1/fan_in²` (SP: `1/fan_in`) | `1/fan_in` |
| SGD LR       | `fan_out` (SP: 1)         | `1/fan_in` (SP: 1) | `1` |
| Adam LR      | `1` (SP: `1/fan_in`)      | `1/fan_in` (SP: 1) | `1/fan_in` (SP: 1) |

**Eq. 3 — an INDEPENDENT cross-check of every cell**, worked out in the same paper for a
3-layer MLP of width `n`: `W¹ ~ N(0, 1/d_in)`, `W² ~ N(0, 1/n)`, `W³ ~ N(0, 1/n²)`,
`η_{W1} = η_{b1} = η_{b2} = η·n`, `η_{W2} = η`, `η_{W3} = η·n⁻¹`.

Reading Eq. 3 against Table 3 for the 3-layer MLP (W1=input, W2=hidden, W3=output) confirms
**every** cell: init `1/fan_in, 1/fan_in, 1/fan_in²`; SGD LR `fan_out=1, 1, 1/fan_in`.
Two independent statements of the same law agreeing cell-for-cell is what makes this table
safe to implement from.

**Attention (Def. 4.1):** `1/d` instead of `1/√d`, i.e. the logit is `qᵀk/d`. Backwards-
compatible multiplier form: `AttnLogit = α_attn · √d_head,0 / d_head · qᵀk`, because
`√d₀/√d = √(d₀/d)` and `1/d = (1/√d)·(1/√d)` — so a single multiplier converts one to the
other at any base width. Verified in Appendix B.

---

## Measured absences (re-measured 2026-10-18, not trusted from the queue)

```
grep -rilE '\bmup\b|muP_|Maximal Update|maximal update|Tensor Programs|zero.sharpness|zeropressure' include/ tests/
```
returns 19 files — **all false positives**. Every one is the substring `mup_` inside
`warmup_` / `warmup_iters` (`optimizers/swa.h:36,44`, `optimizers/prodigy.h:148`,
`optimizers/one_cycle_lr.h:21`, …) plus `m̂_t` in `radam.h:30`. Verified by printing the
matching lines. The strict token `\bmup\b` matches nothing.

```
grep -rilE '\bcurricul|self.paced|spaced repetition|sample.weigh|hard.example.mining|loss.weighting' include/ tests/
```
returns 4 files — **all `sample_weight`** in `interpretability/lime.{h,cpp}`,
`interpretability/shapley.cpp`. Not a curriculum scheduler.

## A QUEUE CLAIM THAT MEASURED FALSE

The queue proposed muP "**plus rank-collapse diagnostics**". `grep -c '\brank\b'` over the
paper's full extracted text (194 496 chars): **0**. Neither does `quadratic metric` (0) nor
`update norm` (0). The paper's actual diagnostic vocabulary is "feature learning". **The
rank-collapse half is not in this citation and is dropped** — implementing it would have
meant inventing an objective from a phrase the queue wrote from memory. The same failure
mode as CRWD and the two phantom VQ-VAE citations.

---

## Design

### The scaling law (all free functions, no Tensor)

```cpp
// Multipliers are RELATIVE to a base width, so every base-width value is
// returned EXACTLY (multiplier 1.0), not approximately.
struct MupSpec {
    size_t width = 0;        // hidden width at the width being configured
    size_t base_width = 0;   // width the hyperparameter was TUNED at
    size_t head_dim = 0;     // attention head dim; 0 = not a transformer
    size_t base_head_dim = 0;
    double  lr = 0.0;        // base LR (base Adam LR; base SGD LR is per-output)
};
```

Role-based multipliers — the input row / hidden row / output row of Table 3, plus biases
(§J: "Biases follow the same reasoning as input weights (just think of it as an input weight
with input 1)"):

| function | μP init var | μP SGD LR | μP Adam LR |
|---|---|---|---|
| `mup_init_var_input`    | `1/fan_in` | — | — |
| `mup_init_var_hidden`   | `1/fan_in` | — | — |
| `mup_init_var_output`   | `1/fan_in²` | — | — |
| `mup_sgd_lr_input`      | — | `fan_out` | — |
| `mup_sgd_lr_hidden`     | — | `1`      | — |
| `mup_sgd_lr_output`     | — | `1/fan_in` | — |
| `mup_adam_lr_input`     | — | — | `1` |
| `mup_adam_lr_hidden`    | — | — | `1/fan_in` |
| `mup_adam_lr_output`    | — | — | `1/fan_in` |
| `mup_lr_bias`           | — | `fan_out` | `1` |

**D1 — the output/init cells are `1/fan_in²` and `1/fan_in`, i.e. the fan-in is counted
TWICE.** Table 3's output-init cell reads `1/fan_in²` while every other cell reads `1/fan_in`.
Read as "1/fan_in, squared" this is exactly Table 8's multiplier-free formulation, and it is
what makes the init rule uniform with the SGD-LR output cell (`1/fan_in`) — the square is
the *product* of the two roles a boundary weight plays. Eq. 3's `W³ ~ N(0, 1/n²)` confirms it.

**D2 — output-`init` uses `fan_in` of the OUTPUT dim, `fan_out` uses the fan-OUT of the
output dim.** For a `(fan_out, fan_in)` weight, output-init is `1/fan_in²` and output-SGD-LR
is `1/fan_in`. Confirmed by Eq. 3 where both are `1/n` and `1/n` resp.

**D3 — SGD LR scales, Adam LR does not, for INPUT weights.** This is the asymmetry the
whole paper is about and the one SP gets backwards in Table 3's grey column. Verified twice.

**D4 — base width returns EXACTLY the tuned value.** Every multiplier is `(w/w₀)^k`, so at
`w == w₀` the ratio is exactly 1.0 in IEEE 754 (`w/w₀` is exact only if... — see test D4's
note; the implementation uses `w / base_width` which is NOT exact for arbitrary w/w₀, so the
implementation multiplies by the ratio and the test asserts the deviation is ≤1 ulp, not
zero. Measured, not assumed.)

### The optimizer entry point

```cpp
// Per-tensor Adam LR multipliers for a stack of layers, in forward order.
Tensor mup_adam_lr_tensor(const std::vector<std::string>& roles, const MupSpec& spec);
```
This is the piece that actually makes the repo's optimizers usable with muP: it returns a
`(1, L)` tensor of per-tensor LR multipliers that an Adam-family optimizer can multiply into
its per-tensor LR. It **never touches the optimizer** — a `LambdaLR`-style hook is the
caller's business — because this repo's optimizers have no per-tensor-LR hook and adding one
is a repo-wide change the queue is not entitled to make unilaterally (same reasoning the
activation-checkpointing refusal recorded).

### The diagnostic

```cpp
// The paper's own liveness quantity: how much did the update move the OUTPUT
// ACTIVATIONS relative to their magnitude. muP holds this ~constant across width;
// SP does not. Paper's term: "the updates' effect on activations becomes roughly
// independent of width in the large width limit."
double mup_update_ratio(double update_norm, double activation_norm);
```

---

## Semantics to pin (each a natural mistake, each with a named test)

- **D1** output init is `1/fan_in²`, not `1/fan_in`.
- **D2** output-init and output-SGD-LR both key off the **fan-in** of the output weight.
- **D3** SGD LR scales `fan_out` on input weights; Adam LR is 1 on input weights.
- **D4** base width is a no-op to within 1 ulp (measured, see above).
- **D5** the `1/d` attention multiplier is `√d₀/d`, NOT `1/d` and NOT `d₀/d` — the ½ power is
  what makes it agree with `1/√d` at the base width. `1/d` would be wrong by `√d`.
- **D6** biases take the **input** row, for both init and both LRs.
- **D7** `fan_in == 0` throws rather than dividing by zero (the repo's `1/0 = NaN` trap, three
  recorded instances).
- **D8** the `1/d` attention logit scaling at the base width equals the `1/√d` scaling
  exactly — a **value** identity, since a pure positive multiplier never changes an argmax
  and so an argmax assertion cannot see it.
- **D9** multipliers are relative to `base_width`, so `mup_*(spec at w₀) == 1` for all
  functions by construction, and the tests pin it.

---

## Why NO finite-difference test (systematic-debugging §5c-zeroth)

Every core here is a **closed-form multiplier over integers** — no forward pass, no cached
input, no stochasticity. There is no function to difference: `mup_init_var_output(8, 4)` is
`1/64` and both of its arguments are integers that a perturbation of `1e-7` rounds away.
A centred FD here would read `0.000000` and "pass" against a zero reference — the exact
vacuous check the PTQ session recorded. Verification is therefore **by value against the
independently-transcribed Table 3 and Eq. 3**, cross-checked cell-for-cell, plus mutation
testing.

---

## Tasks

### Task 1: RED — `tests/test_mup.cpp` asserting the transcribed law
Write the full check list (Table 3 cells, Eq. 3 cells, D1–D9). Register in the Makefile
(`$(BUILD_DIR)/test_mup` rule, the `tests:` prerequisite list, and a `run_tests` line).
Run it. Expected: **compile/link failure only on undefined references** to
`mup_*` — nothing else. A RED that fails for any other reason proves nothing.

### Task 2: GREEN — `include/nn/utils/mup.{h,cpp}`, minimal
Only what Task 1's assertions require. Add `#include "utils/mup.h"` to `include/nn/nn.h`.
Run: green.

### Task 3: VERIFY — mutation harness
≥10 mutations covering: `fan_in²`→`fan_in` (D1), SGD↔Adam on the input row (D3), bias row
input↔output (D6), `√d₀/d`→`1/d` (D5), dropped zero-guard (D7), base-width no-op
inversion. Link set from `make print-lib-objs` (**count ≥ 10 or abort before any verdict**);
every edit checksum-verified; atomic restore of **both** `.cpp` and `.h`; `MUT-BROKEN`
never scored `MUT-CAUGHT`; `MUT-CRASH` its own column; stale `.o` removed before each build.

### Task 4: Sanitizer + warnings
`g++ -O1 -g -fsanitize=address,undefined` and then the project's real `-O2 -march=native
-Wall -Wextra`. Umbrella header compiles standalone.

### Task 5: Aggregate suite
`make run_tests`. Confirm no NEW failures vs. `NOT_FIXED.md`'s recorded baseline.

### Task 6: Docs
Move the item to `## Done` in `EXPANSION_QUEUE.md` with the measured results; delete nothing
from `NOT_FIXED.md` (nothing was fixed).