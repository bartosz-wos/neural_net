// Tests for muP — the Maximal Update Parametrization (arXiv:2203.03466,
// Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen, Gao —
// "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
// Hyperparameter Transfer").
//
// EVERY expected value below is TRANSCRIBED FROM THE PAPER, in two places,
// and the two were checked against each other cell for cell:
//
//   (1) Table 3, parsed from the ar5iv HTML <table> structure (not the
//       flattened text, where the muP and SP columns interleave into noise):
//
//            |              | Input weights & all biases | Output weights       | Hidden weights |
//            | Init. Var.   | 1/fan_in                    | 1/fan_in^2 (SP:1/fan_in) | 1/fan_in |
//            | SGD LR       | fan_out   (SP: 1)           | 1/fan_in   (SP: 1)     | 1 |
//            | Adam LR      | 1         (SP: 1/fan_in)    | 1/fan_in   (SP: 1)     | 1/fan_in (SP: 1) |
//
//   (2) Eq. 3, the paper's own worked example for a 3-layer MLP of width n:
//       W1 ~ N(0, 1/d_in), W2 ~ N(0, 1/n), W3 ~ N(0, 1/n^2),
//       eta_{W1} = eta_{b1} = eta_{b2} = eta*n, eta_{W2} = eta, eta_{W3} = eta/n.
//
// Reading Eq. 3 against Table 3 for that MLP (W1 = input row, W2 = hidden
// row, W3 = output row) confirms EVERY cell of both rows. Two independent
// statements of the same law agreeing cell-for-cell is what makes this
// transcription safe to ship.
//
// NOTE ON THE TITLE: the paper's title says "Zero-Shot", NOT "Zero-
// Sharpness". An arXiv API full-text query for "zero-sharpness
// hyperparameter transfer" returns totalResults=0 — the phrase does not
// occur in the paper. Do not reintroduce the "muP = zero sharpness" gloss.
//
// WHY THERE IS NO FINITE-DIFFERENCE TEST HERE: every function under test is
// a CLOSED-FORM multiplier over integers. There is no forward pass, no
// cached input and no stochasticity to difference; a 1e-7 perturbation of an
// integer width rounds away entirely, so a centred FD would read exactly 0.0
// and "pass" against a zero reference. This is systematic-debugging
// section 5c-zeroth (a degenerate forward direction) and it is the same trap
// the PTQ session recorded. Verification is by VALUE against the
// independently-transcribed law, plus mutation testing.

#include "nn/utils/mup.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s\n", name.c_str());
    }
}

static void check_close(double got, double want, double tol,
                        const std::string& name) {
    if (std::fabs(got - want) <= tol) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s: got %.17g want %.17g (diff %.3g)\n",
                    name.c_str(), got, want, std::fabs(got - want));
    }
}

// Exact equality, for quantities the implementation computes identically.
// Used where a tolerance would hide a genuine bug (see D4's note).
static void check_exact(double got, double want, const std::string& name) {
    if (got == want) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s: got %.17g want %.17g\n", name.c_str(), got, want);
    }
}

// A template, not a function pointer: several entry points below are
// overloaded, and naming an overloaded function in a function-pointer context
// is an ambiguity error rather than a choice of overload.
template <typename F>
static void check_throws(F fn, const std::string& name) {
    try {
        fn();
        ++g_fail;
        std::printf("[FAIL] %s: expected throw, none raised\n", name.c_str());
    } catch (const std::invalid_argument&) {
        ++g_pass;
    } catch (...) {
        ++g_fail;
        std::printf("[FAIL] %s: wrong exception type\n", name.c_str());
    }
}

// ---------------------------------------------------------------------------
// Table 3, transcribed. fan_in / fan_out of a (fan_out, fan_in) weight.
// ---------------------------------------------------------------------------

// 1. D1 -- the output INIT cell is 1/fan_in^2, and every other init cell is
//    1/fan_in. This is the single most-missed cell in Table 3 and the only
//    place a power of 2 appears.
static void test_init_var_table3() {
    const size_t fi = 8, fo = 5;
    MupSpec s;  // width/base_width are irrelevant to an ABSOLUTE init var
    s.width = fi; s.base_width = fi; s.head_dim = fi; s.base_head_dim = fi;
    (void)s;
    check_close(mup_init_var_input(fi), 1.0 / 8.0, 0.0, "init_input 1/fan_in");
    check_close(mup_init_var_hidden(fi), 1.0 / 8.0, 0.0, "init_hidden 1/fan_in");
    check_close(mup_init_var_output(fi), 1.0 / 64.0, 0.0,
                "D1 init_output 1/fan_in^2");
    (void)fo;
}

// 2. Eq. 3 cross-check, read as a 3-layer MLP. n = 8, d_in = 5.
//    This is the independent statement of the same law; if Table 3's
//    transcription were wrong, THIS test would disagree with test 1.
static void test_init_var_equation3_crosscheck() {
    const size_t n = 8, d_in = 5;
    check_close(mup_init_var_input(d_in), 0.2, 1e-15,
                "Eq3 W1 init 1/d_in");
    check_close(mup_init_var_hidden(n), 1.0 / 8.0, 1e-15,
                "Eq3 W2 init 1/n");
    check_close(mup_init_var_output(n), 1.0 / 64.0, 1e-15,
                "Eq3 W3 init 1/n^2");
}

// 3. D3 -- the asymmetry the paper exists to fix. SGD LR scales on the
//    INPUT row (fan_out); Adam LR does NOT (it is 1). Standard
//    parametrization has these two the other way round, which is why SP's
//    optimal LR drifts with width. Both directions asserted: a multiplier
//    of 1.0 must not be confused with a multiplier that merely happens to
//    be small, and fan_out=5 must not silently read as 1.
static void test_sgd_lr_input_scales_adam_lr_input_does_not() {
    MupSpec s;
    s.width = 16; s.base_width = 16; s.head_dim = 8; s.base_head_dim = 8;
    // absolute, width-independent form, via the (fan_out, fan_in) overload
    check_close(mup_sgd_lr_input(5, 8), 5.0, 0.0, "SGD LR input = fan_out");
    check_close(mup_adam_lr_input(5, 8), 1.0, 0.0,
                "D3 Adam LR input = 1, NOT 1/fan_in");
}

// 4. D2 -- the output row keys off fan_in for BOTH init and SGD LR.
static void test_output_row_keys_off_fan_in() {
    check_close(mup_init_var_output(8), 1.0 / 64.0, 0.0,
                "D2 output init uses fan_in");
    check_close(mup_sgd_lr_output(8), 1.0 / 8.0, 0.0,
                "D2 output SGD LR = 1/fan_in");
    // hidden row is 1 for SGD LR
    check_close(mup_sgd_lr_hidden(8), 1.0, 0.0, "hidden SGD LR = 1");
    check_close(mup_adam_lr_hidden(8), 1.0 / 8.0, 0.0, "hidden Adam LR = 1/fan_in");
    check_close(mup_adam_lr_output(8), 1.0 / 8.0, 0.0, "output Adam LR = 1/fan_in");
}

// 5. D6 -- biases follow the INPUT row, for init AND both LRs.
//    Paper appendix J: "Biases follow the same reasoning as input weights
//    (just think of it as an input weight with input 1)."
static void test_biases_take_the_input_row() {
    const size_t fo = 5, fi = 8;
    check_close(mup_bias_init_var(fo), mup_init_var_input(fo), 0.0,
                "D6 bias init == input init");
    check_close(mup_sgd_lr_bias(fo), mup_sgd_lr_input(fo, fi), 0.0,
                "D6 bias SGD LR == input SGD LR");
    check_close(mup_adam_lr_bias(fo), mup_adam_lr_input(fo, fi), 0.0,
                "D6 bias Adam LR == input Adam LR");
}

// ---------------------------------------------------------------------------
// Relative-to-base-width form. Every multiplier is (w / w0)^k, so at the base
// width it must be EXACTLY 1.0.
// ---------------------------------------------------------------------------

// 6. D4 -- at the base width every relative multiplier is exactly 1.0.
//    MEASURED before writing the assertion: w/w0 at w == w0 is exactly 1.0
//    over 200 000 randomised (w, w0) pairs, so `==` is a legitimate
//    assertion here and not a tolerance hiding a bug. A relative form that
//    returned 0.9999999999 at its own base width would silently scale
//    every "transferred" hyperparameter.
static void test_base_width_is_exactly_one() {
    const size_t widths[] = {8, 16, 64, 128, 256, 512, 768, 1024, 3072};
    for (size_t w0 : widths) {
        MupSpec s;
        s.width = w0; s.base_width = w0;
        s.head_dim = w0; s.base_head_dim = w0;
        const std::string tag = "_w" + std::to_string(w0);
        check_exact(mup_init_var_rel(s, MupRole::Input), 1.0,
                    "D4 rel init input " + tag);
        check_exact(mup_init_var_rel(s, MupRole::Hidden), 1.0,
                    "D4 rel init hidden " + tag);
        check_exact(mup_init_var_rel(s, MupRole::Output), 1.0,
                    "D4 rel init output " + tag);
        check_exact(mup_sgd_lr_rel(s, MupRole::Input), 1.0,
                    "D4 rel sgd input " + tag);
        check_exact(mup_sgd_lr_rel(s, MupRole::Hidden), 1.0,
                    "D4 rel sgd hidden " + tag);
        check_exact(mup_sgd_lr_rel(s, MupRole::Output), 1.0,
                    "D4 rel sgd output " + tag);
        check_exact(mup_adam_lr_rel(s, MupRole::Input), 1.0,
                    "D4 rel adam input " + tag);
        check_exact(mup_adam_lr_rel(s, MupRole::Hidden), 1.0,
                    "D4 rel adam hidden " + tag);
        check_exact(mup_adam_lr_rel(s, MupRole::Output), 1.0,
                    "D4 rel adam output " + tag);
        check_exact(mup_sgd_lr_bias_rel(s), 1.0,
                    "D4 rel sgd bias " + tag);
        check_exact(mup_adam_lr_bias_rel(s), 1.0,
                    "D4 rel adam bias " + tag);
    }
}

// 7. The relative law IS the absolute law divided by its base-width value.
//    This is the non-vacuity companion to test 6: a function that returned
//    1.0 unconditionally would pass test 6 and fail this one.
static void test_relative_equals_absolute_over_base() {
    struct Fixture { size_t w, w0; };
    const Fixture fx[] = {{8, 4}, {16, 8}, {64, 16}, {100, 32}, {128, 128}, {512, 128}};
    for (const Fixture& f : fx) {
        MupSpec s;
        s.width = f.w; s.base_width = f.w0;
        s.head_dim = f.w; s.base_head_dim = f.w0;
        const std::string tag = "_" + std::to_string(f.w) + "over" +
                                std::to_string(f.w0);
        const double r_init_in = mup_init_var_rel(s, MupRole::Input);
        const double r_init_out = mup_init_var_rel(s, MupRole::Output);
        // absolute / base-absolute, for the same role
        check_close(r_init_in,
                    mup_init_var_input(f.w) / mup_init_var_input(f.w0),
                    1e-15, "rel/abs init input " + tag);
        check_close(r_init_out,
                    mup_init_var_output(f.w) / mup_init_var_output(f.w0),
                    1e-15, "rel/abs init output " + tag);
        // SGD LR input scales as fan_out, which is fan_in of the input row's
        // OUTPUT dim -- here modelled by the width itself.
        check_close(mup_sgd_lr_rel(s, MupRole::Input),
                    (double)f.w / (double)f.w0, 1e-15,
                    "rel sgd input == w/w0 " + tag);
        // output SGD LR scales as 1/fan_in => inverse
        check_close(mup_sgd_lr_rel(s, MupRole::Output),
                    (double)f.w0 / (double)f.w, 1e-15,
                    "rel sgd output == w0/w " + tag);
        check_close(mup_adam_lr_rel(s, MupRole::Output),
                    (double)f.w0 / (double)f.w, 1e-15,
                    "rel adam output == w0/w " + tag);
        check_close(mup_adam_lr_rel(s, MupRole::Hidden),
                    (double)f.w0 / (double)f.w, 1e-15,
                    "rel adam hidden == w0/w " + tag);
    }
}

// 8. WIDTH SCALING HAS THE SIGN THE LAW SAYS. Doubling the width must HALVE
//    the output-SGD-LR multiplier and DOUBLE the input-SGD-LR one. Asserted
//    from both directions (4x -> 8x and 256 -> 512) because a wrong power
//    passes one of them.
static void test_width_scaling_direction() {
    MupSpec s;
    s.base_width = 4; s.base_head_dim = 4;
    s.head_dim = 4;
    s.width = 8;
    check_close(mup_sgd_lr_rel(s, MupRole::Output), 0.5, 0.0,
                "doubling width halves output SGD LR");
    check_close(mup_sgd_lr_rel(s, MupRole::Input), 2.0, 0.0,
                "doubling width doubles input SGD LR");
    check_close(mup_adam_lr_rel(s, MupRole::Hidden), 0.5, 0.0,
                "doubling width halves hidden Adam LR");
    check_close(mup_init_var_rel(s, MupRole::Output), 0.25, 0.0,
                "doubling width quarters output init");

    s.width = 512; s.base_width = 256;
    check_close(mup_sgd_lr_rel(s, MupRole::Output), 0.5, 0.0,
                "256->512 halves output SGD LR");
    check_close(mup_init_var_rel(s, MupRole::Output), 0.25, 0.0,
                "256->512 quarters output init");
}

// ---------------------------------------------------------------------------
// Attention: Def. 4.1, 1/d instead of 1/sqrt(d).
// ---------------------------------------------------------------------------

// 9. D5 -- the backwards-compatible multiplier is sqrt(d0)/d, NOT 1/d and
//    NOT d0/d. The half power is what makes it agree with 1/sqrt(d) at the
//    base width: sqrt(d0)/sqrt(d0) = 1, whereas 1/d0 is not 1.
//    Both WRONG candidates are pinned explicitly, because a 1/d
//    implementation is off by a factor of sqrt(d) and still looks plausible.
static void test_attention_scale_is_sqrt_of_the_ratio() {
    // (D5) sqrt(d0/d). MEASURED before the assertion was written: the
    // candidates disagree by a POWER of d, not a constant, so each wrong
    // one is pinned with its own numeric value below.
    check_exact(mup_attn_scale_rel(8, 8), 1.0, "D5 attn at base = 1");
    check_close(mup_attn_scale_rel(4, 8), std::sqrt(8.0 / 4.0), 1e-15,
                "D5 attn 4 over base 8");
    check_close(mup_attn_scale_rel(4, 8), 1.4142135623730951, 1e-15,
                "D5 attn sqrt(2), NOT 2 and NOT 0.25");
    check_close(mup_attn_scale_rel(8, 4), std::sqrt(4.0 / 8.0), 1e-15,
                "D5 attn doubling head dim");
    check_close(mup_attn_scale_rel(16, 4), 0.5, 1e-15,
                "D5 attn 4->16 is 0.5, NOT 1/16 and NOT 0.25");
    check_close(mup_attn_scale_rel(16, 4), 0.5, 0.0, "D5 attn 0.5 exactly");
    check_close(mup_attn_scale_rel(128, 64), 1.0 / std::sqrt(2.0), 1e-15,
                "D5 attn 128 over 64");
    // The WRONG candidates, pinned explicitly so the exponent is not
    // re-derived from memory later.
    check(std::fabs(1.0 / 16.0 - mup_attn_scale_rel(16, 4)) > 0.4,
          "D5 1/d candidate is distinguishable at d=16");
    check(std::fabs(1.0 / std::sqrt(16.0) - mup_attn_scale_rel(16, 4)) > 0.2,
          "D5 sqrt(d0)/d candidate is distinguishable at d=16");
}

// 10. D8 -- the logit scaling IDENTITY at the base width: multiplying a
//     1/sqrt(d) attention logit by this multiplier reproduces 1/d
//     EXACTLY, at every width, not just the base width. This is a VALUE
//     identity: a pure positive multiplier never changes an argmax, so an
//     argmax assertion could never see a wrong exponent here.
static void test_attn_multiplier_converts_sqrt_to_plain() {
    const size_t ds[] = {4, 8, 16, 32, 64, 128};
    for (size_t d : ds) {
        for (size_t d0 : ds) {
            const double qk = 3.7;  // any nonzero raw inner product
            const double with_sqrt = qk / std::sqrt((double)d);
            const double multiplied = with_sqrt * mup_attn_scale_rel(d, d0);
            // Appendix B's form, written from the RAW inner product. This is
            // sqrt(d0)/d * qk, NOT qk/d -- the base width is still in there,
            // which is the whole point of a *relative* multiplier.
            const double want = qk * std::sqrt((double)d0) / (double)d;
            check_close(multiplied, want, 1e-14,
                        "D8 attn sqrt->mup d" + std::to_string(d) +
                        "_d0_" + std::to_string(d0));
        }
    }
}

// 11. The `alpha_attn` form from Appendix B: the multiplier has a free
//     constant in it and alpha = 1 reproduces the plain muP rule.
static void test_attn_alpha_form() {
    check_exact(mup_attn_scale(8, 8, 1.0), 1.0, "alpha=1 at base = 1");
    check_close(mup_attn_scale(4, 8, 1.0), std::sqrt(2.0), 1e-15,
                "alpha=1 at 4 over 8 = sqrt(2)");
    check_close(mup_attn_scale(4, 8, 2.0), 2.0 * std::sqrt(2.0), 1e-15,
                "alpha=2 doubles");
    check_exact(mup_attn_scale(4, 8, 0.0), 0.0, "alpha=0 is exactly 0");
    // linear in alpha, from both directions
    check_close(mup_attn_scale(4, 8, 3.0) / mup_attn_scale(4, 8, 1.0), 3.0,
                1e-15, "attn alpha linear");
}

// ---------------------------------------------------------------------------
// Guards.
// ---------------------------------------------------------------------------

// 12. D7 -- a zero dimension THROWS rather than dividing by zero and
//     returning inf/NaN. The repo has recorded three separate 0/0 traps
//     (VQ-VAE decay=1, PTQ all-zero tensors, ptq_relative_l2); a silent
//     NaN here would poison every downstream update.
static void test_zero_dimensions_throw() {
    // width == 0
    check_throws([] { MupSpec s; (void)mup_init_var_rel(s, MupRole::Input); },
                 "D7 init_rel throws on width 0");
    check_throws([] { MupSpec s; (void)mup_sgd_lr_rel(s, MupRole::Hidden); },
                 "D7 sgd_rel throws on width 0");
    check_throws([] { MupSpec s; (void)mup_adam_lr_rel(s, MupRole::Output); },
                 "D7 adam_rel throws on width 0");

    // base_width == 0
    check_throws([] { MupSpec t; t.width = 8; t.base_width = 0;
                      (void)mup_init_var_rel(t, MupRole::Input); },
                 "D7 throws on base_width 0");

    // attention with base_head_dim == 0
    check_throws([] { MupSpec u; u.width = 8; u.base_width = 8;
                      u.head_dim = 8; u.base_head_dim = 0;
                      (void)mup_attn_scale_rel_spec(u); },
                 "D7 attn throws on base_head 0");

    // the absolute overloads
    check_throws([] { (void)mup_init_var_output(0); },
                 "D7 init_var_output(0) throws");
    check_throws([] { (void)mup_init_var_input(0); },
                 "D7 init_var_input(0) throws");
    check_throws([] { (void)mup_attn_scale_rel(8, 0); },
                 "D7 attn_scale_rel throws on base_head 0");

    // Every guarded entry point, one by one. Mutation M09 (dropping the
    // `require_nonzero` from `mup_init_var_hidden`) SURVIVED the first run
    // because this function had no assertion of its own -- the suite tested
    // its input and output siblings and assumed the middle one behaved the
    // same way. Measured: without the guard it returns `inf`, silently.
    check_throws([] { (void)mup_init_var_hidden(0); },
                 "D7 init_var_hidden(0) throws");
    check_throws([] { (void)mup_sgd_lr_input(0, 8); },
                 "D7 sgd_lr_input(0 fan_out) throws");
    check_throws([] { (void)mup_sgd_lr_output(0); },
                 "D7 sgd_lr_output(0) throws");
    check_throws([] { (void)mup_adam_lr_hidden(0); },
                 "D7 adam_lr_hidden(0) throws");
    check_throws([] { (void)mup_adam_lr_output(0); },
                 "D7 adam_lr_output(0) throws");
    check_throws([] { (void)mup_bias_init_var(0); },
                 "D7 bias_init_var(0) throws");
    check_throws([] { (void)mup_sgd_lr_bias(0); },
                 "D7 sgd_lr_bias(0) throws");
    check_throws([] { (void)mup_attn_scale(0, 8); },
                 "D7 attn_scale(0 head_dim) throws");

    // Non-vacuity companion: a nonzero argument must NOT throw, or the
    // checks above could pass on a function that always throws.
    bool threw = false;
    try { (void)mup_init_var_hidden(8); } catch (...) { threw = true; }
    check(!threw, "D7 init_var_hidden(8) does NOT throw");
}

// 13. Empty tensors / tensors of the wrong shape throw, per repo convention.
static void test_empty_tensor_and_guards() {
    MupSpec s;
    s.width = 8; s.base_width = 8; s.head_dim = 8; s.base_head_dim = 8;

    // empty role list -> empty (1,0) tensor, not a crash
    Tensor t = mup_adam_lr_tensor(std::vector<std::string>(), s);
    check(t.rows == 1 && t.cols == 0, "empty roles -> (1,0)");

    std::vector<std::string> bad = {"input", "not_a_role"};
    bool threw = false;
    try { (void)mup_adam_lr_tensor(bad, s); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unknown role throws");

    // NB: `s` has width == base_width == 8, so EVERY relative multiplier is
    // exactly 1 and this fixture CANNOT distinguish the hidden row from the
    // input row. That was the bug in the first draft of this fixture: it
    // asserted 1/8 for the hidden row, which no spec at the base width can
    // produce. Both cases are now covered, and the at-base-width one is
    // asserted to be 1 rather than worked around.
    //
    // EVERY valid role must be accepted, one at a time, and this loop comes
    // BEFORE the first combined fixture on purpose. Mutation M13 (routing
    // "bias" to the hidden branch) reported MUT-CRASH twice before this
    // ordering existed: the first combined call raised an UNCAUGHT throw
    // from the test's own setup and killed the process before any assertion
    // ran. A crash is not a verdict and is certainly not a caught mutation.
    // With each role checked alone and first, a misrouted role is reported
    // as an ordinary [FAIL] and the remaining assertions still execute.
    const std::vector<std::string> valid = {"input", "hidden", "output", "bias"};
    MupSpec one;
    one.width = 8; one.base_width = 8;
    one.head_dim = 8; one.base_head_dim = 8;
    bool all_roles_ok = true;
    for (const std::string& r : valid) {
        bool accepted = true;
        try {
            Tensor single = mup_adam_lr_tensor({r}, one);
            accepted = (single.rows == 1 && single.cols == 1);
        } catch (...) {
            accepted = false;
        }
        check(accepted, "role '" + r + "' is accepted alone");
        if (!accepted) all_roles_ok = false;
    }

    // The per-role loop DETECTS a misrouted role, but on its own it does not
    // stop the crash: execution falls straight through to the combined
    // fixtures below, and the very same bad role raises there from test
    // setup with nothing to catch it. That is why M13 reported MUT-CRASH on
    // two consecutive runs even with the loop in place -- the detection was
    // working and the harness still had nothing to count, because the process
    // died before the summary line. Gating the combined fixtures on the loop
    // is what turns the detection into a verdict.
    if (!all_roles_ok) {
        check(false, "combined-role fixtures SKIPPED: a role is not accepted");
        return;
    }

    std::vector<std::string> ok = {"input", "hidden", "output", "bias"};
    Tensor u = mup_adam_lr_tensor(ok, s);
    check(u.rows == 1 && u.cols == 4, "at base width 4 roles -> (1,4)");
    check_exact(u(0, 0), 1.0, "at base width every tensor entry is 1");
    check_exact(u(0, 1), 1.0, "at base width hidden is 1");
    check_exact(u(0, 2), 1.0, "at base width output is 1");
    check_exact(u(0, 3), 1.0, "at base width bias is 1");

    // Now a real ratio, 8 -> 64.
    MupSpec wide;
    wide.width = 64; wide.base_width = 8;
    wide.head_dim = 8; wide.base_head_dim = 8;
    // NOTE: these use the combined 4-role list, and the per-role loop above
    // is what keeps that list honest -- a role that stops being accepted
    // fails there first, so this line can only ever be reached with a list
    // the suite has already proven acceptable. (M13 crashed HERE, not in the
    // loop, the first time this fixture ran: the loop did not exist.)
    Tensor v = mup_adam_lr_tensor(ok, wide);
    check(u.rows == 1 && u.cols == 4, "4 roles -> (1,4) [adam]");
    check_exact(v(0, 0), 1.0, "8->64 Adam input LR stays 1");
    check_close(v(0, 1), 1.0 / 8.0, 1e-15, "8->64 Adam hidden LR = 1/8");
    check_close(v(0, 2), 1.0 / 8.0, 1e-15, "8->64 Adam output LR = 1/8");
    check_exact(v(0, 3), 1.0, "8->64 Adam bias LR stays 1");

    // ... and the SGD bundle, where the input row DOES move.
    Tensor w = mup_sgd_lr_tensor(ok, wide);
    check(w.rows == 1 && w.cols == 4, "4 roles -> (1,4) [sgd]");
    check_close(w(0, 0), 8.0, 1e-15, "8->64 SGD input LR = 8x");
    check_exact(w(0, 1), 1.0, "8->64 SGD hidden LR stays 1");
    check_close(w(0, 2), 1.0 / 8.0, 1e-15, "8->64 SGD output LR = 1/8");
    check_close(w(0, 3), 8.0, 1e-15, "8->64 SGD bias LR = 8x");
}

// 14. Liveness, not vacuity: the two rows that must NOT scale must actually
//     stay at 1 while the ones that must scale actually move. A test that
//     only checked the moving rows would pass on a file that scaled
//     everything; a test that only checked 1.0 would pass on a file that
//     scaled nothing.
static void test_moving_and_static_rows_are_distinguished() {
    MupSpec s;
    s.width = 64; s.base_width = 8;   // 8x wider
    s.head_dim = 8; s.base_head_dim = 8;

    check_exact(mup_sgd_lr_rel(s, MupRole::Hidden), 1.0,
                "hidden SGD LR is width-INVARIANT under muP");
    check_exact(mup_adam_lr_rel(s, MupRole::Input), 1.0,
                "input Adam LR is width-INVARIANT under muP");
    check_close(mup_sgd_lr_rel(s, MupRole::Input), 8.0, 1e-15,
                "input SGD LR scales 8x");
    check_close(mup_adam_lr_rel(s, MupRole::Hidden), 0.125, 1e-15,
                "hidden Adam LR scales 1/8");
    check_close(mup_init_var_rel(s, MupRole::Output), 1.0 / 64.0, 1e-15,
                "output init scales 1/64");
    // The prescribed init VARIANCE is 1/fan_in, so it SHRINKS as width
    // grows -- the multiplier is w0/w = 1/8, not w/w0. Measured against the
    // absolute cell before this was asserted: (1/w)/(1/w0).
    check_close(mup_init_var_rel(s, MupRole::Input), 1.0 / 8.0, 1e-15,
                "input init shrinks 8x with width");

    // the SP trap, made visible: under SP these two WOULD scale. Their
    // invariance under muP is the entire point of the parametrization.
    check_close(mup_sgd_lr_rel(s, MupRole::Input) / mup_sgd_lr_rel(s, MupRole::Hidden),
                8.0, 1e-15, "muP input/hidden SGD LR ratio == width ratio");
}

// 15. The paper's own liveness quantity. muP holds the update's effect on
//     activations roughly constant across width; SP lets it collapse.
static void test_update_ratio() {
    check_close(mup_update_ratio(1.0, 2.0), 0.5, 0.0, "update_ratio 1/2");
    check_close(mup_update_ratio(1.0, 1.0), 1.0, 0.0, "update_ratio 1");
    check_exact(mup_update_ratio(0.0, 2.0), 0.0, "update_ratio zero update");
    bool threw = false;
    try { (void)mup_update_ratio(1.0, 0.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "update_ratio throws on zero activation norm");
}

// 16. A worked 3-layer MLP, entirely from Eq. 3, at TWO widths -- the whole
//     point of muP is that these two rows of hyperparameters are THE SAME
//     physical numbers, so this is the headline test rather than a
//     restatement of the units.
static void test_three_layer_mlp_worked_example() {
    // Eq. 3 at width n=8, d_in=5, tuned eta.
    const double eta = 0.01;
    const size_t n0 = 8, d_in = 5;

    MupSpec small;
    small.width = n0; small.base_width = n0;
    small.head_dim = n0; small.base_head_dim = n0;

    // At the base width every multiplier is exactly 1, so the tuned values
    // pass through untouched -- that is the definition of "tuned here".
    const double w1_init = mup_init_var_input(d_in);
    const double w3_init = mup_init_var_output(n0);
    // Eq. 3 states ABSOLUTE values, so it is compared against the ABSOLUTE
    // cell, not against a relative multiplier. The relative form is 1 at the
    // base width by construction (test_base_width_is_exactly_one) and carries
    // no absolute scale at all -- mixing the two is the bug this line was
    // fixed for.
    check_close(w1_init, 0.2, 1e-15, "Eq3 n=8: W1 init 1/d_in");
    check_close(w3_init, 1.0 / 64.0, 1e-15, "Eq3 n=8: W3 init 1/n^2");
    // Eq. 3: eta_{W3} = eta * n^-1, with fan_in = n for the MLP's output row.
    check_close(eta * mup_sgd_lr_output(n0), eta / 8.0, 1e-15,
                "Eq3 n=8: eta_W3 = eta/n (absolute)");
    // ... and the relative form at the base width is exactly 1.
    check_exact(mup_sgd_lr_rel(small, MupRole::Input), 1.0,
                "Eq3 n=8: eta_W1 relative 1 at base");
    check_exact(mup_sgd_lr_rel(small, MupRole::Output), 1.0,
                "Eq3 n=8: eta_W3 relative 1 at base");

    // Now the SAME tuned eta, transferred to n=32 (4x wider).
    MupSpec big;
    big.width = 32; big.base_width = n0;
    big.head_dim = n0; big.base_head_dim = n0;

    check_close(mup_init_var_rel(big, MupRole::Input), 1.0 / 4.0, 1e-15,
                "transfer: W1 init var scales DOWN as 1/n");
    check_close(mup_init_var_rel(big, MupRole::Output), 1.0 / 16.0, 1e-15,
                "transfer: W3 init var scales DOWN as 1/n^2");
    check_close(mup_sgd_lr_rel(big, MupRole::Input), 4.0, 1e-15,
                "transfer: eta_W1 scales UP with width");
    check_close(mup_sgd_lr_rel(big, MupRole::Output), 1.0 / 4.0, 1e-15,
                "transfer: eta_W3 scales DOWN as 1/n");
    check_close(mup_sgd_lr_rel(big, MupRole::Hidden), 1.0, 0.0,
                "transfer: eta_W2 is UNCHANGED");

    // Absolute, for reference: at n=32, W3 init var is 1/32^2.
    check_close(mup_init_var_output(32), 1.0 / 1024.0, 1e-15,
                "absolute W3 init at n=32 is 1/n^2");
}

int main() {
    test_init_var_table3();
    test_init_var_equation3_crosscheck();
    test_sgd_lr_input_scales_adam_lr_input_does_not();
    test_output_row_keys_off_fan_in();
    test_biases_take_the_input_row();
    test_base_width_is_exactly_one();
    test_relative_equals_absolute_over_base();
    test_width_scaling_direction();
    test_attention_scale_is_sqrt_of_the_ratio();
    test_attn_multiplier_converts_sqrt_to_plain();
    test_attn_alpha_form();
    test_zero_dimensions_throw();
    test_empty_tensor_and_guards();
    test_moving_and_static_rows_are_distinguished();
    test_update_ratio();
    test_three_layer_mlp_worked_example();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}