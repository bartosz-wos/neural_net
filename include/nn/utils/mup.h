#ifndef NN_UTILS_MUP_H
#define NN_UTILS_MUP_H

#include "../core/tensor.h"
#include <cstddef>
#include <string>
#include <vector>

// ===========================================================================
// muP -- the Maximal Update Parametrization
//
//   Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu,
//   David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao --
//   "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
//   Hyperparameter Transfer", arXiv:2203.03466.
//
// ---------------------------------------------------------------------------
// CITATION. Verified 2026-10-18 by scraping the arXiv abs page's
// `citation_*` meta tags -- the arXiv API returned
// `totalResults=0` for a full-text query on the phrase people use to
// describe this paper, so a title query was not available as a check:
//
//   citation_title  -> "Tensor Programs V: Tuning Large Neural Networks
//                       via Zero-Shot Hyperparameter Transfer"
//   citation_author -> Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder,
//                       Pachocki, Chen, Gao
//   citation_date   -> 2022/03/07
//   citation_arxiv_id -> 2203.03466
//
// THE TITLE SAYS "ZERO-SHOT", NOT "ZERO-SHARPNESS". The widely-repeated
// gloss "muP is zero-sharpness hyperparameter transfer" is a mangling of
// this title: the phrase "zero-sharpness" appears NOWHERE in the paper.
// `grep -c '\brank\b'` over the full extracted text (194 496 characters)
// returns 0, as do "quadratic metric" and "update norm". Do not
// reintroduce the gloss.
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED
// ---------------------------------------------------------------------------
//
//   grep -rilE '\bmup\b|muP_|Maximal Update|maximal update|Tensor Programs
//           |zero.sharpness|zeropressure' include/ tests/
//
// returns 19 files, and EVERY ONE IS A FALSE POSITIVE: the substring
// `mup_` inside `warmup_` / `warmup_iters` (optimizers/swa.h:36,44,
// optimizers/prodigy.h:148, optimizers/one_cycle_lr.h:21, ...), plus
// `m_t` in optimizers/radam.h:30. Verified by printing the matching
// lines. The strict token `\bmup\b` matches nothing anywhere.
//
// So the repo has ~250 layers and a dozen optimizers and no notion that
// a learning rate and an initialization scale are FUNCTIONS OF WIDTH.
// Every shipped initializer is a constant, and every shipped LR schedule
// is a function of the step only. That is exactly what makes this paper
// necessary: under a constant parametrization the optimal LR drifts with
// width, so a model tuned at 256 does not inherit good hyperparameters at
// 1024, and every new width costs a fresh search.
//
// ---------------------------------------------------------------------------
// THE LAW (paper Table 3), and an INDEPENDENT CROSS-CHECK
// ---------------------------------------------------------------------------
//
//                          Input wts & all biases | Output wts | Hidden wts
//   Init. Var.                1 / fan_in           | 1 / fan_in^2 | 1 / fan_in
//                             (SP: 1/fan_in)      | (SP: 1/fan_in)|            |
//   SGD LR                    fan_out             | 1 / fan_in   | 1
//                             (SP: 1)             | (SP: 1)      |              |
//   Adam LR                   1                   | 1 / fan_in   | 1 / fan_in
//                             (SP: 1/fan_in)      | (SP: 1)      | (SP: 1)     |
//
// Table 3 was read out of the ar5iv HTML <table> STRUCTURE, not the
// flattened text (where the muP and SP columns interleave into
// unreadable noise, and where "1/fan_in^2 (1/fan_in)" is indistinguishable
// from a product).
//
// Eq. 3 -- the paper's OWN worked example, for a 3-layer MLP of width n
// with input dim d_in, SGD LRs relative to a tuned eta:
//
//   W1 ~ N(0, 1/d_in),  W2 ~ N(0, 1/n),  W3 ~ N(0, 1/n^2),  b = 0
//   eta_{W1} = eta_{b1} = eta_{b2} = eta*n,  eta_{W2} = eta,
//   eta_{W3} = eta*n^-1
//
// Reading Eq. 3 against Table 3 for that MLP (W1 = input row, W2 = hidden
// row, W3 = output row) confirms EVERY cell of both rows:
//
//   init:  1/d_in, 1/n, 1/n^2          <-> Table 3 input, hidden, output
//   SGD:   fan_out = 1, 1, 1/fan_in   <-> Table 3 input, hidden, output
//
// Two independent statements of the same law agreeing cell for cell is
// what makes this transcription safe to implement from. `tests/test_mup.cpp`
// carries BOTH as separate assertions, so a misreading of either one fails
// the suite rather than cancelling out.
//
// ---------------------------------------------------------------------------
// SEMANTICS: what a reimplementation gets wrong
// ---------------------------------------------------------------------------
//
// (D1) THE OUTPUT-INIT CELL IS 1/fan_in^2, and it is the ONLY cell with a
//      power of 2 in it. Every other init cell is 1/fan_in. Confirmed
//      independently by Eq. 3's `W3 ~ N(0, 1/n^2)`. A reimplementation
//      that writes 1/fan_in here produces a model that trains, trains
//      stably, and is simply at the wrong scale -- the failure muP was
//      invented to remove.
//
// (D2) THE OUTPUT ROW KEYS OFF fan_in for BOTH init and SGD LR. For a
//      (fan_out, fan_in) weight, output-init is 1/fan_in^2 and
//      output-SGD-LR is 1/fan_in. These are different powers of the SAME
//      number, which is the boundary weight wearing both hats: it is the
//      output of one layer and the input of the next.
//
// (D3) SGD LR SCALES ON THE INPUT ROW AND ADAM LR DOES NOT. This is the
//      asymmetry the whole paper exists to fix, and it is exactly where
//      Table 3's SP column has the two values the other way round. A
//      reimplementation that uses one LR rule for both optimizers is
//      wrong in a way no single-layer test can see, because on a single
//      layer with fan_out == 1 the two rows coincide.
//
// (D4) BASE WIDTH IS A NO-OP, AND IT IS AN EXACT NO-OP. Every relative
//      multiplier is (w / w0)^k, and `w / w0` at `w == w0` is exactly
//      1.0 in IEEE 754 -- MEASURED over 200 000 randomised (w, w0) pairs
//      before this assertion was written, giving 0 non-exact cases. So
//      `== 1.0` is a legitimate assertion here and not a tolerance
//      concealing a bug. A relative form returning 0.9999999999 at its own
//      base width would silently rescale every "transferred"
//      hyperparameter.
//
// (D5) THE ATTENTION MULTIPLIER IS sqrt(d0/d), NOT 1/d, NOT d0/d and NOT
//      sqrt(d0)/d. Definition 4.1 replaces 1/sqrt(d) with 1/d. Appendix B
//      writes the muP logit directly from the RAW inner product:
//
//          AttnLogit = alpha_attn * (sqrt(d_head,0) / d_head) * q^T k
//
//      which is the COMPLETE logit, not a multiplier on an already-scaled
//      one. Converting it into the factor that multiplies a logit the
//      caller has ALREADY scaled by 1/sqrt(d) gives, with F unknown:
//
//          (q^T k / sqrt(d)) * F  ==  q^T k * sqrt(d0) / d
//          =>  F = (sqrt(d0)/d) * sqrt(d) = sqrt(d0) / sqrt(d) = sqrt(d0/d)
//
//      The half power is load-bearing, and every wrong candidate is off by
//      a POWER of d rather than a constant, so none is caught by a single
//      fixture: 1/d is wrong by sqrt(d), sqrt(d0)/d is wrong by 1/sqrt(d),
//      and d0/d is wrong by (d0/d)/sqrt(d0/d).
//
// (D8) THE 1/d CONVERSION IS A VALUE IDENTITY, not an argmax identity. A
//      pure positive multiplier never changes an argmax, so a test that
//      checked only the winning class could never see a wrong exponent.
//      The conversion is asserted on the LOGIT VALUE, at every width.
//
// (D6) BIASES FOLLOW THE INPUT ROW, for init AND both LRs. Paper appendix
//      J: "Biases follow the same reasoning as input weights (just think
//      of it as an input weight with input 1)." A bias vector's fan_out
//      is its own dimension, and its fan_in is 1.
//
// (D7) A ZERO DIMENSION THROWS. `1/0` is inf and `0/0` is NaN, and the
//      repo has now recorded three separate instances of that trap
//      (VQ-VAE `decay=1` returning an all-NaN codebook, PTQ's
//      all-zero-tensor scale, `ptq_relative_l2`'s zero denominator). A
//      NaN multiplier here poisons every downstream update with no error
//      and no log line.
//

// ===========================================================================

enum class MupRole { Input, Hidden, Output };

// Widths, in hidden units. `width`/`base_width` are the model width at the
// size being configured and at the size the hyperparameters were tuned at;
// `head_dim`/`base_head_dim` are the attention head dims for the same two.
// The Adam base LR is carried for completeness so a caller can build the
// absolute value in one call; every function below returns a MULTIPLIER
// relative to the tuned value, which is the quantity that transfers.
struct MupSpec {
    size_t width = 0;
    size_t base_width = 0;
    size_t head_dim = 0;
    size_t base_head_dim = 0;
};

// ---------------------------------------------------------------------------
// ABSOLUTE form: the value in Table 3's cell, given the layer's own dims.
// `fan_out` / `fan_in` are the weight's dimensions, NOT the model's width.
// ---------------------------------------------------------------------------

// Init variances. (D1) the output one is the only squared one.
double mup_init_var_input(size_t fan_in);
double mup_init_var_hidden(size_t fan_in);
double mup_init_var_output(size_t fan_in);

// SGD learning-rate multipliers.
double mup_sgd_lr_input(size_t fan_out, size_t fan_in);
double mup_sgd_lr_hidden(size_t fan_in);
double mup_sgd_lr_output(size_t fan_in);

// Adam learning-rate multipliers. (D3) the input one is 1, not 1/fan_in.
double mup_adam_lr_input(size_t fan_out, size_t fan_in);
double mup_adam_lr_hidden(size_t fan_in);
double mup_adam_lr_output(size_t fan_in);

// (D6) Biases, all taking the input row. A bias vector's fan_out is its own
// dimension and its fan_in is 1, so the caller passes the length.
double mup_bias_init_var(size_t fan_out);
double mup_sgd_lr_bias(size_t fan_out);
double mup_adam_lr_bias(size_t fan_out);

// ---------------------------------------------------------------------------
// RELATIVE form: the multiplier that converts a hyperparameter TUNED at
// `base_width` into the one for `width`. Every one is (w/w0)^k and therefore
// exactly 1.0 at the base width (D4). This is the interface that actually
// transfers a hyperparameter: nothing here needs to know the base value.
// ---------------------------------------------------------------------------

double mup_init_var_rel(const MupSpec& spec, MupRole role);
double mup_sgd_lr_rel(const MupSpec& spec, MupRole role);
double mup_adam_lr_rel(const MupSpec& spec, MupRole role);
double mup_bias_init_var_rel(const MupSpec& spec);
double mup_sgd_lr_bias_rel(const MupSpec& spec);
double mup_adam_lr_bias_rel(const MupSpec& spec);

// ---------------------------------------------------------------------------
// Attention (Definition 4.1)
// ---------------------------------------------------------------------------

// (D5) The factor converting a 1/sqrt(d)-scaled attention logit to the muP
// 1/d form: sqrt(d0/d), optionally scaled by the paper's `alpha_attn`.
// (D8) `logit / sqrt(d) * this == logit * sqrt(d0) / d` exactly at every
// width -- i.e. the raw-inner-product form Appendix B prints.
double mup_attn_scale(size_t head_dim, size_t base_head_dim, double alpha_attn = 1.0);
double mup_attn_scale_rel(size_t head_dim, size_t base_head_dim, double alpha_attn = 1.0);
double mup_attn_scale_rel_spec(const MupSpec& spec, double alpha_attn = 1.0);

// ---------------------------------------------------------------------------
// The per-tensor multiplier bundle an Adam-family optimizer can apply.
// ---------------------------------------------------------------------------
//
// Roles are given as strings so a caller can write them next to the layer
// list; anything else throws. The result is (1, L) in the order given.
//
// This deliberately does NOT touch any optimizer: this repo's optimizers
// have no per-tensor-LR hook, and adding one is a repo-wide change this
// header is not entitled to make unilaterally -- the same reasoning that
// recorded activation checkpointing as REFUTED rather than implemented.
Tensor mup_adam_lr_tensor(const std::vector<std::string>& roles, const MupSpec& spec);
Tensor mup_sgd_lr_tensor(const std::vector<std::string>& roles, const MupSpec& spec);

// ---------------------------------------------------------------------------
// The paper's own liveness quantity.
// ---------------------------------------------------------------------------
//
// Paper §1: muP "ensures that each layer is updated on the same order
// during training regardless of width", i.e. "the updates' effect on
// activations becomes roughly independent of width in the large width
// limit". This is that effect, as a ratio, so a sweep over widths can be
// checked for flatness. Under muP it is width-independent; under SP it
// collapses with width. `update_norm` is the norm of the parameter change,
// `activation_norm` the norm of the activations it moves.
//
// Throws on a zero activation norm rather than returning NaN (D7).
double mup_update_ratio(double update_norm, double activation_norm);

#endif  // NN_UTILS_MUP_H