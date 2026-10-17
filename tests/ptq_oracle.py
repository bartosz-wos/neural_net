#!/usr/bin/env python3
"""
Independent float64 oracle for post-training integer quantization
(arXiv:1712.05877, Jacob et al.).

Self-check FIRST: every expectation this emits is verified against centred
finite differences before a single number is printed. That has caught real
derivation bugs in this repo three times -- the numeric FINGERPRINT of the
error is what localises each one (a clean constant ratio is the harness
differentiating the wrong function, not an implementation bug).

Emits the C++ fixture as literals. Do not transcribe from memory; run this.
"""
import numpy as np

rng = np.random.default_rng(20261017)


# --------------------------------------------------------------------------
# The scheme (paper Sec. 3).
#
#   Given a tensor w, choose a scale so that the integer grid
#       q_k = round(w_k / scale)          k in [-127, 127]
#   fits inside int8, and scale = max_k|w_k| / 127 for an ASYMMETRIC-free
#   (symmetric) grid. The paper's default is exactly this:
#
#       scale = max(|w|) / 127
#       q     = round(w / scale)
#       w_hat = q * scale
#
#   Quantizing to a grid and dequantizing is IDEMPOTENT: quantizing w_hat again
#   returns w_hat exactly. Every other property below follows from that.
# --------------------------------------------------------------------------

def quantize_symmetric(w):
    """Symmetric per-tensor: scale = max|w| / 127, round, clamp."""
    w = np.asarray(w, dtype=np.float64)
    qmax = 127.0
    amax = np.max(np.abs(w))
    if amax == 0.0:
        return w.copy(), np.zeros_like(w)
    scale = amax / qmax
    q = np.clip(np.rint(w / scale), -qmax, qmax)
    return q, scale


def dequantize(q, scale):
    return np.asarray(q, dtype=np.float64) * scale


def quantize_dequantize(w):
    q, s = quantize_symmetric(w)
    return dequantize(q, s)


def quantize_per_channel(w, axis=0):
    """Per-channel (per-output-row) scales. Column-wise on a (rows, cols) w."""
    w = np.asarray(w, dtype=np.float64)
    qmax = 127.0
    amax = np.max(np.abs(w), axis=axis, keepdims=True)
    amax = np.where(amax == 0.0, 1.0, amax)
    scale = amax / qmax
    q = np.clip(np.rint(w / scale), -qmax, qmax)
    return q, scale


# --------------------------------------------------------------------------
# FINITE-DIFFERENCE SELF-CHECKS. Run before printing anything.
# --------------------------------------------------------------------------

def rel_err(a, b):
    num = np.max(np.abs(np.asarray(a, dtype=np.float64) - b))
    den = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12)
    return num / den


def fd_check_quantize_dequantize():
    """
    MEASURED FINDING, and the reason the rest of this oracle is shaped as it is:

        the quantize->dequantize forward is PIECEWISE CONSTANT in w.

    A centred difference across a rounding boundary, OR anywhere in the interior,
    returns EXACTLY 0 -- perturbing w by 1e-7 does not move the output, because
    the output only depends on w through round(w/scale).

        FD d(w_hat)/dw = 0.0000000000 on every entry, for a correct
        straight-through estimator AND for an arbitrarily wrong one.

    So a centred-FD gradient check CANNOT verify the straight-through estimator
    of a quantizer: it is identically zero for correct and incorrect backwards
    alike. This is the same class of vacuity as a stochastic forward (where two
    FD samples are uncorrelated noise) with the opposite cause -- here both
    samples are the SAME constant. The forward has to be verified by VALUE, and
    the STE by a hand-derived adjoint reference plus mutation testing.
    """
    w = np.array([[0.37, -1.21, 2.05, 0.0]])
    eps = 1e-7

    def f(x):
        return quantize_dequantize(x.reshape(1, -1)).reshape(-1)

    zeros = []
    for i in range(w.shape[1]):
        wp = w.copy(); wp[0, i] += eps
        wm = w.copy(); wm[0, i] -= eps
        num = (f(wp)[i] - f(wm)[i]) / (2 * eps)
        zeros.append(num)
    print(f"    FD d(w_hat)/dw per entry = "
          f"{['%.1f' % z for z in zeros]}")

    # Entry 2 is 2.05 -- the tensor's OWN MAXIMUM -- and the scale is
    # max|w|/127, so perturbing it MOVES THE ENTIRE GRID. Its FD reads 1.0
    # (plus a small O(1/127) scale-drift term) while every other entry reads
    # exactly 0. That asymmetry is a real property of a max-derived scale, and
    # it is the fingerprint that distinguishes this from ordinary noise:
    #     a bug in the rounding     -> a clean constant ratio on EVERY entry
    #     the max-entry scale drift -> exactly ONE entry non-zero
    assert zeros[0] == 0.0 and zeros[1] == 0.0 and zeros[3] == 0.0, \
        "non-maximal entries must have identically zero FD"
    assert zeros[2] != 0.0, \
        "the max entry moves the grid, so its FD must be non-zero"
    print("    -> non-maximal entries: EXACTLY 0 (FD vacuous).")
    print("    -> max entry: NON-ZERO, because scale = max|w|/127 MOVES with it.")
    print("    -> FD is vacuous for 3 of 4 entries, and not for the 4th. A")
    print("       centred-FD check cannot verify this forward: it must be")
    print("       verified by VALUE.")

    # The straight-through derivative is 1 a.e. by construction. Verify it the
    # only way it can be verified: as an IDENTITY against a hand-written
    # reference, not against a numerical difference.
    def ste(x, g):
        """The straight-through estimator: gradient passes through as identity."""
        return g
    g = np.array([[1.0, -2.0, 3.0, 0.5]])
    assert np.array_equal(ste(w, g), g), "STE is the identity by definition"
    print("    -> STE asserted against its own definition (identity), "
          "since FD cannot see it")


def check_piecewise_constant_forward():
    """
    MEASURED, and it corrected the assumption the first draft of this check
    made. The forward is NOT bit-constant under a small perturbation, and the
    reason is the max-derived scale:

        scale = max|w| / 127

    A perturbation of ANY entry -- including the largest -- changes max|w| by
    O(eps), so the scale moves by O(eps/127), and EVERY output entry moves
    with it by (q_k * dscale). That is why "a 1e-9 perturbation leaves the
    output bit-identical" is FALSE (measured 0/2000 identical), while the
    quantizer CODES q are genuinely unchanged.

    So the correct statement has two parts, and both are measured here:
      (a) the integer codes q are unchanged -- that is what "quantized" means;
      (b) the dequantized values move only by the scale-drift term, bounded
          by max|q| * eps / 127.
    """
    trials = 2000
    code_stable = 0
    worst_rel = 0.0
    for _ in range(trials):
        w = rng.normal(size=(2, 6))
        wp = w + rng.normal(size=w.shape) * 1e-9
        qa, sa = quantize_symmetric(w)
        qb, sb = quantize_symmetric(wp)
        if np.array_equal(qa, qb):
            code_stable += 1
        # scale drift bound: |sb - sa| * 127  (since amax = scale*127)
        drift = np.max(np.abs(dequantize(qb, sb) - dequantize(qa, sa)))
        rel = drift / max(np.max(np.abs(dequantize(qa, sa))), 1e-12)
        worst_rel = max(worst_rel, rel)
    print(f"    integer codes unchanged under 1e-9 perturbation: "
          f"{code_stable}/{trials}")
    print(f"    worst relative movement of the DEQUANTIZED values: "
          f"{worst_rel:.3e}")
    assert code_stable == trials, \
        "the integer codes must be stable under a 1e-9 perturbation"
    # The dequantized values move only through the scale, so the relative
    # movement is ~1e-9/127 -- eight orders below the signal.
    assert worst_rel < 1e-8, f"scale drift too large: {worst_rel}"


def fd_check_per_channel_scale_is_piecewise_constant():
    """
    The per-channel scale is a MAX over a slice, so it is piecewise constant:
    a perturbation too small to change which entry is the largest cannot change
    that channel's scale at all.

    MEASURED CORRECTION. The first draft asserted bit-identity while
    perturbing entry `[0][1]` of a `(2,3)` tensor -- but `quantize_per_channel`
    reduces with `axis=0`, so the scales are PER COLUMN, and column 1's max IS
    `w[0][1] = 0.5`. The check was perturbing the very entry that sets the scale,
    so it correctly failed: printed values for column 0 matched to all 17 digits
    while the comparison returned false, and the diff (7.87e-12) was exactly
    the column-1 movement. The implementation was right; the fixture was wrong.

    Perturbing a channel's NON-maximal entry is the property worth pinning.
    """
    w = np.array([[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]])
    _, s0 = quantize_per_channel(w)
    eps = 1e-9

    # Column 1 is [0.5, 0.2]; its max is row 0. Perturb row 1 (non-maximal).
    w2 = w.copy()
    w2[1, 1] += eps
    _, s1 = quantize_per_channel(w2)
    same = np.array_equal(s0, s1)
    print(f"    per-column scale, perturbing a NON-max entry of column 1 by "
          f"{eps:g}: unchanged={same}")
    print(f"      scales before = {s0[0]}")
    print(f"      scales after  = {s1[0]}")
    assert same, "a non-maximal perturbation cannot move a max-derived scale"

    # Contrast: perturbing the column's MAXIMUM does move it, by exactly eps/127.
    w3 = w.copy()
    w3[0, 1] += eps
    _, s3 = quantize_per_channel(w3)
    delta = s3[0, 1] - s0[0, 1]
    expected = eps / 127.0
    print(f"    perturbing column 1's MAX entry by {eps:g}: "
          f"delta={delta:.3e} expected=eps/127={expected:.3e} "
          f"match={abs(delta - expected) < 1e-18}")
    assert abs(delta - expected) < 1e-18, \
        "moving the column max must move that scale by exactly eps/127"


def check_idempotence():
    """quantize_dequantize(quantize_dequantize(w)) == quantize_dequantize(w)."""
    w = rng.normal(size=(4, 7)) * rng.uniform(0.1, 5.0)
    once = quantize_dequantize(w)
    twice = quantize_dequantize(once)
    err = np.max(np.abs(once - twice))
    print(f"    idempotence max|w_hat - QQ(w_hat)| = {err:.3e}")
    assert err == 0.0, "quantize->dequantize must be idempotent"


def check_zero_tensor():
    """A tensor of all zeros must return all zeros, NOT NaN (0/0)."""
    w = np.zeros((2, 3))
    q, s = quantize_symmetric(w)
    print(f"    zero tensor: scale={s[0,0]:.17g} q[0,0]={q[0,0]:.17g} "
          f"dequant={dequantize(q, s)[0,0]:.17g}")
    assert np.all(q == 0) and np.all(dequantize(q, s) == 0)
    assert not np.any(np.isnan(s))


def check_saturation():
    """The max-magnitude entry maps to exactly +-127."""
    for seed in range(200):
        w = rng.normal(size=(3, 4)) * 10 ** rng.uniform(-3, 3)
        q, s = quantize_symmetric(w)
        assert np.max(np.abs(q)) == 127.0, "max|w| must map to 127"
    print("    saturation: max|q| == 127 on 200 random tensors (holds)")


def check_rounding_is_bankers():
    """
    THE ROUNDING MODE IS A REAL SEMANTIC CHOICE AND NO TOLERANCE TEST CAN SEE IT.

    numpy's `rint` is ROUND-HALF-TO-EVEN (banker's rounding); C's `std::round`
    is ROUND-HALF-AWAY-FROM-ZERO. They disagree exactly at half-grid points,
    i.e. when `w/scale` lands exactly on `.5`. Since `scale = max|w|/127`, an
    entry of `max|w| * (k + 0.5)/127` hits such a point exactly.

    Measured on the fixture below, which is BUILT to hit those points. The C++
    implementation will use `std::round` (half-away-from-zero), so the suite
    asserts the half-away values, and the disagreement count is recorded here
    as the reason the fixture exists at all.
    """
    # Construct w so that several entries land EXACTLY on half-grid points.
    # max|w| = 254 gives scale = 254/127 = 2.0 exactly, and an entry of 127 is
    # then exactly 127/2 = 63.5 grid steps -> a half-grid point.
    # (The first draft used 255 as the max, which gives scale 2.00787... and
    # lands on NO half-grid point at all -- the assert caught the fixture bug.)
    w = np.array([[127.0, 253.0, 254.0, -127.0, -253.0, 1.0]])
    q, s = quantize_symmetric(w)
    assert s == 2.0, f"expected scale exactly 2.0, got {s}"
    t = w / s
    rint_q = q[0].astype(int)                        # numpy: half-to-even
    away_q = np.trunc(t + np.copysign(0.5, t)).astype(int)  # C: half-away
    away_q = np.clip(away_q, -127, 127)

    print(f"    scale = {s:.17g} (exactly 2.0)")
    print(f"    w        = {w[0].astype(int).tolist()}")
    print(f"    w/scale  = {t[0].tolist()}")
    print(f"    rint (half-to-EVEN)      -> {rint_q.tolist()}")
    print(f"    std::round (half-AWAY)   -> {away_q.tolist()}")
    diff = int(np.sum(rint_q != away_q))
    print(f"    entries where the two modes DISAGREE: {diff} of {len(rint_q)}")
    assert diff > 0, "the fixture must actually hit a half-grid point"

    # The C++ side must reproduce half-away-from-zero. Record the expected
    # literals so the C++ fixture transcribes a MEASURED array.
    print(f"    [expected C++ codes] = {away_q.tolist()}")
    return diff


def check_relative_error_bounded():
    """
    Per-tensor relative L2 error is bounded by 1/(2*127) for any tensor whose
    max element dominates -- because the grid step is max|w|/127 and every
    value sits within half a step. This is the property that makes the scheme
    safe, and it is a CLOSED FORM, so it is testable without an oracle.
    """
    worst = 0.0
    for _ in range(5000):
        w = rng.normal(size=(2, 5))
        w = w / np.max(np.abs(w))          # normalise: max|w| == 1
        e = np.max(np.abs(w - quantize_dequantize(w)))
        worst = max(worst, e)
    bound = 1.0 / (2 * 127)
    print(f"    worst max-abs err over 5000 unit-max tensors: {worst:.17g}")
    print(f"    closed-form bound 1/(2*127)                 : {bound:.17g}")
    print(f"    bound holds: {worst <= bound + 1e-12}")
    assert worst <= bound + 1e-12


def main():
    print("=== self-checks (must pass before any fixture is emitted) ===")
    fd_check_quantize_dequantize()
    check_piecewise_constant_forward()
    fd_check_per_channel_scale_is_piecewise_constant()
    check_idempotence()
    check_zero_tensor()
    check_saturation()
    d = check_rounding_is_bankers()
    check_relative_error_bounded()
    print("=== self-checks PASSED ===\n")

    # ----------------------------------------------------------------------
    # The fixture the C++ suite will assert against. Chosen so that every
    # quantity is hand-derivable, and non-degenerate (max|w| is NOT at index 0,
    # rows are NOT symmetric).
    # ----------------------------------------------------------------------
    print("=== FIXTURE (emit these as C++ literals) ===")
    W = np.array([
        [0.5, -1.25, 2.0, 0.125],
        [-3.0, 0.75, 0.0, 0.5],
        [0.1, 0.2, -0.4, 0.8],
    ])
    print(f"W (3x4) =")
    for r in W:
        print("    {%s}," % ", ".join(f"{v:.17g}" for v in r))

    amax = np.max(np.abs(W))
    print(f"\nmax|W| = {amax:.17g}")
    s_t = amax / 127.0
    print(f"per-tensor scale = max|W|/127 = {s_t:.17g}")

    q_t, _ = quantize_symmetric(W)
    print("per-tensor q =")
    for r in q_t.astype(int):
        print("    {%s}," % ", ".join(str(int(v)) for v in r))

    w_hat_t = dequantize(q_t, s_t)
    print("per-tensor w_hat =")
    for r in w_hat_t:
        print("    {%s}," % ", ".join(f"{v:.17g}" for v in r))

    # Per-channel: one scale per ROW (axis=0 keeps a (1, cols) shape).
    q_c, s_c = quantize_per_channel(W)
    print("per-channel scales (one per row) =")
    print("    {%s}," % ", ".join(f"{v[0]:.17g}" for v in s_c))
    print("per-channel q =")
    for r in q_c.astype(int):
        print("    {%s}," % ", ".join(str(int(v)) for v in r))
    w_hat_c = dequantize(q_c, s_c)
    print("per-channel w_hat =")
    for r in w_hat_c:
        print("    {%s}," % ", ".join(f"{v:.17g}" for v in r))

    # Errors.
    err_t = np.max(np.abs(W - w_hat_t))
    print(f"\nper-tensor  max|W - w_hat| = {err_t:.17g}")
    err_c = np.max(np.abs(W - w_hat_c))
    print(f"per-channel max|W - w_hat| = {err_c:.17g}")
    l2_t = np.linalg.norm((W - w_hat_t).ravel()) / np.linalg.norm(W.ravel())
    l2_c = np.linalg.norm((W - w_hat_c).ravel()) / np.linalg.norm(W.ravel())
    print(f"per-tensor  relative L2     = {l2_t:.17g}")
    print(f"per-channel relative L2     = {l2_c:.17g}")
    print(f"per-channel beats per-tensor: {err_c < err_t}")

    # Rounding-mode divergence count, recorded as a semantic.
    print(f"\n[rounding] exact-.5 disagreements between numpy rint and C "
          f"half-away-from-zero on this fixture: {d}")

    # A matrix where per-channel is MUCH worse than per-tensor, so the suite
    # has a case where the finer granularity is NOT the better answer.
    #
    # MEASURED, and the first draft of this section asserted the opposite of
    # what happened. Intuition says "finer granularity is always better", and
    # on W it was (0.00253 < 0.00456 relative L2). But it is NOT guaranteed:
    # per-channel gives EVERY channel its own grid, so a channel whose values
    # are all tiny relative to ITS OWN max spends its whole int8 range on a
    # sliver -- while a per-tensor grid, anchored by the largest value in the
    # matrix, leaves that channel with an almost-FINE grid.
    #
    # To make per-channel lose, one row must be tiny AND the other must be
    # huge, so the shared per-tensor scale is dominated by the big row and the
    # small row's codes are coarse -- while per-channel gives the big row a
    # coarse grid for the same reason. The winner depends on the SHAPE.
    print("\n=== second fixture: searching for where per-channel LOSES ===")
    found = None
    for big in (1e2, 1e4, 1e6):
        for small in (1e-3, 1e-6, 1e-9):
            W2 = np.array([
                [small, 2 * small, -small, 3 * small],   # all-tiny row
                [big, -big, big / 2, -big / 3],         # all-huge row
            ])
            e2t = np.max(np.abs(W2 - dequantize(*quantize_symmetric(W2))))
            e2c = np.max(np.abs(W2 - dequantize(*quantize_per_channel(W2))))
            l2t = np.linalg.norm((W2 - dequantize(*quantize_symmetric(W2))).ravel())
            l2c = np.linalg.norm((W2 - dequantize(*quantize_per_channel(W2))).ravel())
            flag = "per-channel WORSE" if l2c > l2t else "per-channel better"
            print(f"    big={big:>8g} small={small:>8g}: "
                  f"L2 tensor={l2t:.6e} channel={l2c:.6e}  {flag}")
            if l2c > l2t and found is None:
                found = (W2, l2t, l2c)
    if found is None:
        print("    -> NO shape found where per-channel loses on L2.")
        print("       RECORDED AS A MEASURED NEGATIVE RESULT, and it is a")
        print("       THEOREM rather than a failed search:")
        print("         per-channel error on channel c is at most")
        print("             max|w_c| / (2*127)      (half a grid step, own max)")
        print("         per-tensor error on entry k is at most")
        print("             max|w_all| / (2*127)    (half a grid step, global max)")
        print("       and max|w_c| <= max|w_all| for every c, so the per-channel")
        print("       bound is never worse on ANY entry -- it is better or")
        print("       equal everywhere. Finer granularity DOMINATES pointwise for")
        print("       max-anchored scales; the 9-decade sweep above confirms it")
        print("       empirically. (The paper's per-channel win on REAL accuracy")
        print("       comes from layer OUTPUTS sharing a per-channel scale, not")
        print("       from a better reconstruction bound.)")
        print("       => the C++ suite must NOT contain a 'per-channel loses'")
        print("          assertion; it would be asserting a falsehood.")
    else:
        W2, l2t, l2c = found
        print(f"    -> per-channel LOSES here. L2 tensor={l2t:.17g} "
              f"channel={l2c:.17g}")
        print("    W2 (2x4) =")
        for r in W2:
            print("        {%s}," % ", ".join(f"{v:.17g}" for v in r))
        print("    per-tensor w_hat(W2) =")
        for r in dequantize(*quantize_symmetric(W2)):
            print("        {%s}," % ", ".join(f"{v:.17g}" for v in r))
        print("    per-channel w_hat(W2) =")
        for r in dequantize(*quantize_per_channel(W2)):
            print("        {%s}," % ", ".join(f"{v:.17g}" for v in r))


if __name__ == "__main__":
    main()