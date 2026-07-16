#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <vector>
#include <utility>
#include "nn/utils/segmentation_losses.h"
#include "nn/core/tensor.h"

static int checks_passed = 0;
static int checks_total = 0;
#define CHECK(cond) do { ++checks_total; if (cond) ++checks_passed; \
    else std::cout << "  FAIL [" << __LINE__ << "]: " << #cond << "\n"; } while(0)


// =============================================================================
// DiceLoss forward tests
// =============================================================================

void test_dice_perfect_prediction() {
    DiceLoss loss;
    Tensor pred(2, 1);   pred[0][0] = 1.0;  pred[1][0] = 1.0;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 1.0;
    Tensor out = loss.forward(pred, target);
    CHECK(std::abs(out[0][0]) < 1e-9);
}

void test_dice_zero_intersection() {
    DiceLoss loss;
    Tensor pred(2, 1);   pred[0][0] = 1.0;  pred[1][0] = 1.0;
    Tensor target(2, 1); target[0][0] = 0.0; target[1][0] = 0.0;
    Tensor out = loss.forward(pred, target);
    // For each row (single cell): per-row Dice = (0+1)/(1+0+1) = 1/2, row loss = 1/2.
    // Mean over 2 rows → 0.5.
    CHECK(std::abs(out[0][0] - 0.5) < 1e-6);
}

void test_dice_partial() {
    DiceLoss loss;
    Tensor pred(2, 1);   pred[0][0] = 1.0;  pred[1][0] = 0.0;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 1.0;
    // Per row: inter = 2 * 1*1 = 2; |P|=1, |T|=2; dice = (2+1)/(1+2+1) = 3/4; loss = 1/4
    Tensor out = loss.forward(pred, target);
    CHECK(std::abs(out[0][0] - 0.25) < 1e-6);
}

void test_dice_shape_mismatch_throws() {
    DiceLoss loss;
    Tensor pred(2, 1);
    Tensor target(3, 1);
    bool threw = false;
    try { loss.forward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}

void test_dice_default_eps_accessor() {
    DiceLoss loss;
    CHECK(std::abs(loss.get_eps() - 1.0) < 1e-12);
    DiceLoss custom(0.5);
    CHECK(std::abs(custom.get_eps() - 0.5) < 1e-12);
}


// =============================================================================
// DiceLoss backward (FD check)
// =============================================================================

void test_dice_backward_gradient_check() {
    DiceLoss loss(0.1);  // smaller eps so FD signal is less zero-padded
    Tensor pred(3, 2);
    pred[0][0] = 0.7; pred[0][1] = 0.3;
    pred[1][0] = 0.5; pred[1][1] = 0.5;
    pred[2][0] = 0.9; pred[2][1] = 0.2;
    Tensor target(3, 2);
    target[0][0] = 1.0; target[0][1] = 0.0;
    target[1][0] = 0.0; target[1][1] = 1.0;
    target[2][0] = 1.0; target[2][1] = 0.0;

    Tensor ana;
    try { ana = loss.backward(pred, target); }
    catch (const std::exception& e) {
        ++checks_total;
        std::cout << "  FAIL [" << __LINE__ << "]: backward threw: " << e.what() << "\n";
        return;
    }

    double eps_fd = 1e-5;
    double max_rel = 0.0;
    int n = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;
            double lp = loss.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;
            double lm = loss.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : std::abs(ana[i][j] - num);
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-5); ++n;
        }
    }
    std::cout << "  [Dice grad check] max rel_err = " << max_rel << " (over " << n << " entries)\n";
}


// =============================================================================
// TverskyLoss tests
// =============================================================================

void test_tversky_alpha_beta_balanced() {
    // At α=β=0.5, Tversky gives a "balanced" loss that is NOT identical to Dice
    // (Dice uses 2*inter / (|P|+|T|); Tversky uses inter / (0.5*FP + 0.5*FN + inter)).
    // Verify the relationship: Tversky numerator = inter; Dice numerator = 2*inter.
    // So Tversky loss > Dice loss at the same eps, but both must lie in [0, 1].
    DiceLoss dl(0.5);
    TverskyLoss tl(0.5, 0.5, 0.5);
    Tensor pred(2, 1);   pred[0][0] = 0.6; pred[1][0] = 0.4;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 0.0;
    double dl_v = dl.forward(pred, target)[0][0];
    double tl_v = tl.forward(pred, target)[0][0];
    CHECK(dl_v >= 0.0 && dl_v <= 1.0);
    CHECK(tl_v >= 0.0 && tl_v <= 1.0);
    // With balanced α=β and identical input, both losses should be REASONABLY close
    // (within 50% relative — they saturate the same interval but at different rates)
    double rel_diff = std::abs(dl_v - tl_v) / std::max(1e-12, std::max(std::abs(dl_v), std::abs(tl_v)));
    CHECK(rel_diff < 0.5);
}

void test_tversky_asymmetric_weighting() {
    // With alpha (FP weight) big: predicting 1 over target=0 is penalized more
    // → over-prediction (pred > target) gets higher loss than under-prediction.
    TverskyLoss tl_low_alpha(0.2, 0.8, 1.0);   // low FP weight  → forgives over-prediction
    TverskyLoss tl_high_alpha(0.8, 0.2, 1.0);  // high FP weight → penalizes over-prediction
    Tensor pred(3, 1);   pred[0][0] = 1.0; pred[1][0] = 1.0; pred[2][0] = 1.0;
    Tensor target(3, 1); target[0][0] = 1.0; target[1][0] = 0.0; target[2][0] = 1.0;
    double lp_low = tl_low_alpha.forward(pred, target)[0][0];
    double lp_high = tl_high_alpha.forward(pred, target)[0][0];
    CHECK(lp_low < lp_high);
}

void test_tversky_perfect_prediction() {
    TverskyLoss tl(0.3, 0.7);
    Tensor pred(2, 1);   pred[0][0] = 1.0; pred[1][0] = 1.0;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 1.0;
    Tensor out = tl.forward(pred, target);
    CHECK(std::abs(out[0][0]) < 1e-12);
}

void test_tversky_shape_mismatch_throws() {
    TverskyLoss tl;
    Tensor pred(2, 1);
    Tensor target(2, 2);
    bool threw = false;
    try { tl.forward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}

void test_tversky_backward_gradient_check() {
    TverskyLoss tl(0.3, 0.7, 0.1);  // asymmetric, smaller eps for FD sensitivity
    Tensor pred(3, 2);
    pred[0][0] = 0.6; pred[0][1] = 0.4;
    pred[1][0] = 0.5; pred[1][1] = 0.5;
    pred[2][0] = 0.8; pred[2][1] = 0.2;
    Tensor target(3, 2);
    target[0][0] = 1.0; target[0][1] = 0.0;
    target[1][0] = 0.0; target[1][1] = 1.0;
    target[2][0] = 1.0; target[2][1] = 0.0;

    Tensor ana;
    try { ana = tl.backward(pred, target); }
    catch (const std::exception& e) {
        ++checks_total;
        std::cout << "  FAIL [" << __LINE__ << "]: backward threw: " << e.what() << "\n";
        return;
    }
    double eps_fd = 1e-5;
    double max_rel = 0.0;
    int n = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;
            double lp = tl.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;
            double lm = tl.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : 0.0;
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-5); ++n;
        }
    }
    std::cout << "  [Tversky grad check] max_rel = " << max_rel << " (over " << n << " entries)\n";
}


// =============================================================================
// FocalDiceLoss tests
// =============================================================================

void test_focal_dice_gamma_zero_reduces_to_tversky() {
    TverskyLoss tl(0.5, 0.5, 0.5);
    FocalDiceLoss fdl_zero(0.0, 0.5, 0.5, 0.5);   // γ=0 → modulator=1 → plain Tversky
    FocalDiceLoss fdl_higher(2.0, 0.5, 0.5, 0.5);
    Tensor pred(2, 1);   pred[0][0] = 0.6; pred[1][0] = 0.4;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 0.0;
    CHECK(std::abs(tl.forward(pred, target)[0][0] - fdl_zero.forward(pred, target)[0][0]) < 1e-9);
    // γ > 0 should give a different loss value (focusing matters for hard pixels)
    CHECK(std::abs(fdl_zero.forward(pred, target)[0][0] - fdl_higher.forward(pred, target)[0][0]) > 1e-6);
}

void test_focal_dice_perfect_prediction_zero_loss() {
    FocalDiceLoss fdl(2.0);
    Tensor pred(2, 1);   pred[0][0] = 1.0; pred[1][0] = 1.0;
    Tensor target(2, 1); target[0][0] = 1.0; target[1][0] = 1.0;
    Tensor out = fdl.forward(pred, target);
    // p_t = 1 → (1 - 1)^γ = 0, so per-row contribution is 0 regardless of Tversky.
    CHECK(std::abs(out[0][0]) < 1e-12);
}

void test_focal_dice_shape_mismatch_throws() {
    FocalDiceLoss fdl;
    Tensor pred(2, 1);
    Tensor target(2, 2);
    bool threw = false;
    try { fdl.forward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}

void test_focal_dice_accessors() {
    FocalDiceLoss fdl(1.5, 0.3, 0.7, 0.25);
    CHECK(std::abs(fdl.get_gamma() - 1.5) < 1e-12);
    CHECK(std::abs(fdl.get_alpha() - 0.3) < 1e-12);
    CHECK(std::abs(fdl.get_beta() - 0.7) < 1e-12);
    CHECK(std::abs(fdl.get_eps() - 0.25) < 1e-12);
}

void test_focal_dice_backward_gradient_check() {
    FocalDiceLoss fdl(1.5, 0.4, 0.6, 0.1);
    Tensor pred(3, 2);
    pred[0][0] = 0.6; pred[0][1] = 0.4;
    pred[1][0] = 0.5; pred[1][1] = 0.5;
    pred[2][0] = 0.8; pred[2][1] = 0.2;
    Tensor target(3, 2);
    target[0][0] = 1.0; target[0][1] = 0.0;
    target[1][0] = 0.0; target[1][1] = 1.0;
    target[2][0] = 1.0; target[2][1] = 0.0;

    Tensor ana;
    try { ana = fdl.backward(pred, target); }
    catch (const std::exception& e) {
        ++checks_total;
        std::cout << "  FAIL [" << __LINE__ << "]: backward threw: " << e.what() << "\n";
        return;
    }
    double eps_fd = 1e-5;
    double max_rel = 0.0;
    int n = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;
            double lp = fdl.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;
            double lm = fdl.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : 0.0;
            if (rel > max_rel) max_rel = rel;
            CHECK(rel < 1e-4); ++n;  // slightly looser — focal has a near-zero terms
        }
    }
    std::cout << "  [FocalDice grad check] max_rel = " << max_rel << " (over " << n << " entries)\n";
}


// =============================================================================
// Lovász-Hinge tests
// =============================================================================

void test_lovasz_perfect_or_near_perfect() {
    LovaszHingeLoss lh;
    // pred and target match → errors: pos rows have error=0, neg rows have error=1.
    // hinge surrogate: t=1,p=1 → err=0; t=0,p=1 → err=1 (high).
    // Best-case for "all positives predicted correctly" is the (t=1,p=1) error=0 with t=0,p=0 err=0.
    Tensor pred(1, 3);   pred[0][0] = 1.0; pred[0][1] = 0.0; pred[0][2] = 0.0;
    Tensor target(1, 3); target[0][0] = 1.0; target[0][1] = 0.0; target[0][2] = 0.0;
    Tensor out = lh.forward(pred, target);
    // All errors are zero → loss is zero.
    CHECK(out[0][0] < 1e-9);
}

void test_lovasz_perfect_iorow() {
    LovaszHingeLoss lh;
    // pred all 1, target all 1 — but we need at least one false positive to exercise the sort.
    // Make a 1x2 row that's perfect: t=1,p=1, t=0,p=0.
    Tensor pred(1, 2);   pred[0][0] = 1.0;  pred[0][1] = 0.0;
    Tensor target(1, 2); target[0][0] = 1.0; target[0][1] = 0.0;
    Tensor out = lh.forward(pred, target);
    CHECK(out[0][0] < 1e-9);
}

void test_lovasz_better_than_worst_case() {
    LovaszHingeLoss lh;
    // Perfect prediction (pred matches target perfectly)
    Tensor pred_perfect(1, 3);   pred_perfect[0][0] = 1.0; pred_perfect[0][1] = 1.0; pred_perfect[0][2] = 0.0;
    Tensor tgt_perfect(1, 3);    tgt_perfect[0][0] = 1.0; tgt_perfect[0][1] = 1.0; tgt_perfect[0][2] = 0.0;
    // Worst case: every positive predicted as 0, every negative predicted as 1
    Tensor pred_worst(1, 3);     pred_worst[0][0] = 0.0; pred_worst[0][1] = 0.0; pred_worst[0][2] = 1.0;
    // Same target as above
    double lp = lh.forward(pred_perfect, tgt_perfect)[0][0];
    double lw = lh.forward(pred_worst, tgt_perfect)[0][0];
    CHECK(lp < lw);
    CHECK(lp < 1e-9);
}

void test_lovasz_shape_mismatch_throws() {
    LovaszHingeLoss lh;
    Tensor pred(2, 1);
    Tensor target(2, 2);
    bool threw = false;
    try { lh.forward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}

void test_lovasz_backward_gradient_check() {
    LovaszHingeLoss lh;
    // Use distinct errors so the sort is unambiguous; pick values so no row has
    // tied errors (which would break FD for the slope discontinuity).
    Tensor pred(2, 4);
    pred[0][0] = 0.62; pred[0][1] = 0.41; pred[0][2] = 0.73; pred[0][3] = 0.29;
    pred[1][0] = 0.51; pred[1][1] = 0.55; pred[1][2] = 0.83; pred[1][3] = 0.18;
    Tensor target(2, 4);
    target[0][0] = 1.0; target[0][1] = 0.0; target[0][2] = 1.0; target[0][3] = 0.0;
    target[1][0] = 0.0; target[1][1] = 1.0; target[1][2] = 0.0; target[1][3] = 1.0;

    Tensor ana;
    try { ana = lh.backward(pred, target); }
    catch (const std::exception& e) {
        ++checks_total;
        std::cout << "  FAIL [" << __LINE__ << "]: backward threw: " << e.what() << "\n";
        return;
    }
    double eps_fd = 1e-5;
    double max_rel = 0.0;
    int n = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double orig = pred[i][j];
            pred[i][j] = orig + eps_fd;
            double lp = lh.forward(pred, target)[0][0];
            pred[i][j] = orig - eps_fd;
            double lm = lh.forward(pred, target)[0][0];
            pred[i][j] = orig;
            double num = (lp - lm) / (2.0 * eps_fd);
            double den = std::max(std::abs(ana[i][j]), std::abs(num));
            double rel = (den > 1e-12) ? std::abs(ana[i][j] - num) / den : 0.0;
            if (rel > max_rel) max_rel = rel;
            // Sort-chain has small linear segments; FD is fine but not bit-exact.
            CHECK(rel < 1e-3); ++n;
        }
    }
    std::cout << "  [Lovasz grad check] max_rel = " << max_rel << " (over " << n << " entries)\n";
}


// =============================================================================
// Cross-cutting: each loss has a backward and shape validation
// =============================================================================

void test_loss_backward_shape_mismatch_throws() {
    DiceLoss dl;
    TverskyLoss tl;
    FocalDiceLoss fdl;
    LovaszHingeLoss lh;
    Tensor pred(2, 1);
    Tensor target(3, 1);
    bool threw = false;
    try { dl.backward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
    threw = false;
    try { tl.backward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
    threw = false;
    try { fdl.backward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
    threw = false;
    try { lh.backward(pred, target); } catch (...) { threw = true; }
    CHECK(threw);
}


// =============================================================================
// End-to-end training smoke tests
// =============================================================================

void test_dice_training_reduces_loss() {
    // Toy 2-pixel binary segmentation, 1 sample, 50 SGD steps with lr=0.1.
    // target = [1, 0]; start pred = [0.3, 0.7] (both wrong).
    // After ~50 steps with Adam-like behavior, pred should approach [1, 0].
    DiceLoss loss;
    Tensor pred(1, 2);
    pred[0][0] = 0.3; pred[0][1] = 0.7;
    Tensor target(1, 2);
    target[0][0] = 1.0; target[0][1] = 0.0;
    double lr = 0.1;
    double L0 = loss.forward(pred, target)[0][0];
    for (int step = 0; step < 50; ++step) {
        Tensor g = loss.backward(pred, target);
        for (size_t c = 0; c < 2; ++c)
            pred[0][c] -= lr * g[0][c];
        // Clamp pred to [0, 1] for stability.
        for (size_t c = 0; c < 2; ++c)
            pred[0][c] = std::max(0.001, std::min(0.999, pred[0][c]));
    }
    double L_final = loss.forward(pred, target)[0][0];
    std::cout << "  [Dice smoke] L0=" << L0 << " -> Lf=" << L_final << "\n";
    CHECK(L_final < L0);  // training reduces loss
    CHECK(pred[0][0] > 0.5);  // pos prediction grows toward 1
    CHECK(pred[0][1] < 0.5);  // neg prediction falls toward 0
}

void test_tversky_asymmetric_training_balances() {
    // Asymmetric Tversky where we want FN < FP (e.g. tumor segmentation).
    // Set beta=0.9 (high FN weight) so the loss heavily penalizes missing positives.
    // Train to predict 2 positive pixels from an initial over-prediction.
    TverskyLoss tl(0.5, 0.9, 1.0);
    Tensor pred(1, 4);
    pred[0][0] = 0.4; pred[0][1] = 0.8; pred[0][2] = 0.6; pred[0][3] = 0.5;
    Tensor target(1, 4);
    target[0][0] = 1.0; target[0][1] = 1.0; target[0][2] = 0.0; target[0][3] = 0.0;
    double L0 = tl.forward(pred, target)[0][0];
    for (int step = 0; step < 50; ++step) {
        Tensor g = tl.backward(pred, target);
        for (size_t c = 0; c < 4; ++c)
            pred[0][c] = std::max(0.001, std::min(0.999, pred[0][c] - 0.05 * g[0][c]));
    }
    double Lf = tl.forward(pred, target)[0][0];
    std::cout << "  [Tversky smoke] L0=" << L0 << " -> Lf=" << Lf << "\n";
    CHECK(Lf < L0);
}


int main() {
    std::cout << std::setprecision(12);
    std::cout << "=== Segmentation Losses Tests ===\n";

    std::cout << "\n--- DiceLoss forward ---\n";
    test_dice_perfect_prediction();
    test_dice_zero_intersection();
    test_dice_partial();
    test_dice_shape_mismatch_throws();
    test_dice_default_eps_accessor();

    std::cout << "\n--- DiceLoss backward (FD) ---\n";
    test_dice_backward_gradient_check();

    std::cout << "\n--- TverskyLoss ---\n";
    test_tversky_alpha_beta_balanced();
    test_tversky_asymmetric_weighting();
    test_tversky_perfect_prediction();
    test_tversky_shape_mismatch_throws();
    test_tversky_backward_gradient_check();

    std::cout << "\n--- FocalDiceLoss ---\n";
    test_focal_dice_gamma_zero_reduces_to_tversky();
    test_focal_dice_perfect_prediction_zero_loss();
    test_focal_dice_shape_mismatch_throws();
    test_focal_dice_accessors();
    test_focal_dice_backward_gradient_check();

    std::cout << "\n--- LovaszHingeLoss ---\n";
    test_lovasz_perfect_or_near_perfect();
    test_lovasz_perfect_iorow();
    test_lovasz_better_than_worst_case();
    test_lovasz_shape_mismatch_throws();
    test_lovasz_backward_gradient_check();

    std::cout << "\n--- Cross-cutting ---\n";
    test_loss_backward_shape_mismatch_throws();

    std::cout << "\n--- End-to-end training ---\n";
    test_dice_training_reduces_loss();
    test_tversky_asymmetric_training_balances();

    std::cout << "\n=== Summary: " << checks_passed << "/" << checks_total
              << " checks passed ===\n";
    return (checks_passed == checks_total) ? 0 : 1;
}
