#!/usr/bin/env Rscript
# ===========================================================================
# tests/_r_contingency.R -- the R half of the contingency-statistics
# cross-check.
#
# R is the reference for McNemar's continuity-correction rule, and that rule
# is NOT the textbook one: R SKIPS the correction on a symmetric table, so
# the statistic is 0 there rather than 1/(2*n01). The oracle's fixture
# F11 exists precisely to pin that, and R is what settles it.
#
# The output format is one record per line:
#
#     <TAG> <statistic> <p_value> <label>
#
# parsed by tests/contingency_oracle.py's r_cross_check().
#
# RUN BY tests/contingency_oracle.py -- NOT a standalone test.
# ===========================================================================

options(digits = 17)

emit_mcnemar <- function(n00, n01, n10, n11, label) {
  x <- matrix(c(n00, n01, n10, n11), 2, 2, byrow = TRUE)
  r <- mcnemar.test(x)
  cat(sprintf("MCNEMAR %.17g %.17g %s\n", r$statistic, r$p.value, label))
}

emit_chisq <- function(m, label) {
  r <- suppressWarnings(chisq.test(m, correct = TRUE))
  cat(sprintf("CHISQ %.17g %.17g %s\n", r$statistic, r$p.value, label))
}

# --- McNemar ---------------------------------------------------------------
# F10: n01 = 5, n10 = 3 -> ASYMMETRIC, so R APPLIES the correction.
emit_mcnemar(12, 5, 3, 20, "F10")

# F11: n01 = n10 = 5 -> SYMMETRIC off-diagonal, so R SKIPS the correction
# and the statistic is 0. A textbook implementation that always subtracts
# 1 would report (0-1)^2/10 = 0.1 here.
emit_mcnemar(30, 5, 5, 0, "F11")

# F12: n01 = 3 < n10 = 5 -> R's numerator is (|n01-n10|-1)^2 = 1, so the
# statistic is 0.125. The SIGNED form would give (3-5-1)^2/8 = 0.5625.
emit_mcnemar(10, 3, 5, 0, "F12")

# F13: no discordant pairs -> 0/0, so R reports NaN for both.
emit_mcnemar(30, 0, 0, 30, "F13")

# --- chi-square ------------------------------------------------------------
# F6C: the 2x2 with the Yates correction applied (the CLAMPED form).
emit_chisq(matrix(c(10, 20, 30, 15), 2, 2, byrow = TRUE), "F6C")

# F8C: the CLAMP fixture. Every cell has |O - E| = 0.2 < 0.5, so R's
# elementwise clamp drives the statistic to 0. An unclamped
# implementation would report 0.3125.
emit_chisq(matrix(c(1, 1, 1, 2), 2, 2, byrow = TRUE), "F8C")
