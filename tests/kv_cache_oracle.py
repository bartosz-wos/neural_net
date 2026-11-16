#!/usr/bin/env python3
"""
tests/kv_cache_oracle.py -- the float64 numpy specification of the
inference-time KEY/VALUE CACHE and of the CACHED INCREMENTAL FORWARD.

This file was written and run BEFORE the C++ existed. Its discriminating
power was MEASURED, not narrated: tests/kv_cache_mutate_oracle.py breaks the
model code below in specific ways and requires that this oracle FAIL each
time. An oracle that cannot fail certifies nothing.

    python3 tests/kv_cache_oracle.py            # run the specification
    python3 tests/kv_cache_oracle.py --emit-cpp # emit the pinned C++ fixtures

--------------------------------------------------------------------------
THE GAP THIS SPECIFIES (measured 2026-11-16, not assumed)
--------------------------------------------------------------------------

    grep -rinE '\\bkv.?cache|paged.attn|prefix.cache|past_key' include/
    -> 3 hits, ALL COMMENTS (mla.h:33, sliding_window.h:17, megalodon.h:67)
    -> ZERO implementation across ~250 layers.

Many attention layers hold `last_k_` / `last_v_` -- but those are caches for
the BACKWARD pass, not an inference-time cache. The repo has no incremental
decoding path at all.

--------------------------------------------------------------------------
THE INVARIANT
--------------------------------------------------------------------------

For a causal attention, decoding one token at a time with a cache must equal
the full-sequence forward, token for token. This is the property the whole
feature rests on, and it is a property of the OUTPUT -- not of a theorem.

Two independent references are implemented here, deliberately written the two
different ways round:

  * `ref_mha_full`         -- the full-sequence forward, in the repo's own
                              conventions and layout.
  * `ref_mha_incremental`  -- the cached path: it keeps an explicit (T, d)
                              K/V store, appends to it, and masks only among
                              the NEW chunk (every cached key is in the past,
                              so it is admissible).

They are not the same code with a flag flipped: the full path materialises
one (T, T) score matrix, the incremental path materialises one (n, T+n) score
matrix per chunk. A shared bug is much less likely to survive both.

--------------------------------------------------------------------------
NO CITATION IS ASSERTED
--------------------------------------------------------------------------

The key/value cache is an implementation technique rather than a paper, and
this repo's expansion queue has produced fifteen wrong arXiv identifiers --
every one a plausible-looking neighbour of the right paper. The one reference
that IS asserted is the standard "Fast Transformer Decoding: One Write-Head
is All You Need" (Shazeer, 2019), which is a TITLE and a venue, not an
identifier. Nothing else is cited.
"""

import math
import sys

import numpy as np

PASSED = 0
FAILED = 0


def check(cond, msg):
    """Record one check. `cond` must already be a bool."""
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print("  [FAIL] %s" % msg)


def check_close(got, want, tol, msg):
    """Record one closeness check, printing the real values on failure.

    A boolean-only check costs a full bisect loop; printing the numbers
    usually ends the investigation on the spot.
    """
    global PASSED, FAILED
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        FAILED += 1
        print("  [FAIL] %s -- shape got %s want %s" % (msg, got.shape, want.shape))
        return
    diff = np.max(np.abs(got - want)) if got.size else 0.0
    if diff <= tol:
        PASSED += 1
    else:
        FAILED += 1
        print("  [FAIL] %s -- max_abs_diff = %.17g (tol %.3g)" % (msg, diff, tol))
        print("         got  = %s" % np.array2string(got.ravel()[:6], precision=17))
        print("         want = %s" % np.array2string(want.ravel()[:6], precision=17))


def check_throws(fn, needle, msg):
    """Record that `fn` throws, AND that the message says WHY.

    A check that only asserts "it threw" is blind to a guard that was
    removed while a DIFFERENT guard still fires: the throw happens either
    way, and the suite goes green. THE DISCRIMINATOR IS THE MESSAGE, NOT
    THE THROW -- the rule this repo has learned the hard way, and the
    reason the mutation harness must be able to see it.
    """
    global PASSED, FAILED
    try:
        fn()
    except Exception as exc:                    # noqa: BLE001 - the message is the point
        text = str(exc)
        if needle in text:
            PASSED += 1
        else:
            FAILED += 1
            print("  [FAIL] %s -- threw %r, expected a message containing %r"
                  % (msg, text, needle))
        return
    FAILED += 1
    print("  [FAIL] %s -- did not throw" % msg)


def rel_err(got, want):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    denom = max(np.max(np.abs(want)) if want.size else 0.0, 1e-12)
    return float(np.max(np.abs(got - want)) / denom) if got.size else 0.0


# ===========================================================================
# SECTION 1 -- the references
# ===========================================================================

def ref_mha_full(x, Wq, Wk, Wv, Wo, heads):
    """The repo's plain causal multi-head attention, full sequence.

    Follows include/nn/layers/attention/transformer.cpp exactly:
      * input layout (d_model, seq_len); tokens are COLUMNS of the input;
      * internal layout (tokens, d_model);
      * Q = X @ Wq^T, K = X @ Wk^T, V = X @ Wv^T;
      * per-head slice is the column block [h*d_k, (h+1)*d_k);
      * scores = Q K^T / sqrt(d_k + 1e-9);
      * causal mask adds -1e9 to j > i;
      * softmax is max-subtracted;
      * output = acc @ Wo^T, then back to (d_model, seq_len).
    """
    d_model, seq_len = x.shape
    X = x.T                                    # (tokens, d_model)
    # THE PROJECTION CONVENTION IS THE REPO'S, NOT THE TEXTBOOK'S.
    #
    # The C++ layer computes Q[i][j] = sum_k x[i][k] * W_q[k][j], which is
    # the STANDARD matmul X @ W_q -- and Tensor::operator* in this repo is
    # exactly that (res[i][j] += a_ik * other[k][j]). The first draft of
    # this oracle wrote X @ Wq.T, the textbook convention for a projection
    # stored as (out, in); the repo stores W_q as (d_model, d_model) with
    # the INPUT feature on the row axis, so the transpose is wrong.
    #
    # The defect was invisible to the oracle's own checks -- it agreed with
    # itself -- and was found by the CROSS-LANGUAGE comparison against the
    # C++ (max_abs_diff = 1.546). That is what the pinned-fixture transfer
    # is FOR: two implementations of the same maths can agree with each
    # other and both be wrong about the convention.
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_k = d_model // heads
    scale = math.sqrt(d_k + 1e-9)

    acc = np.zeros((seq_len, d_model), dtype=float)
    idx = np.arange(seq_len)
    for h in range(heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        Qh = Q[:, sl]                          # (tokens, d_k)
        Kh = K[:, sl]
        Vh = V[:, sl]
        scores = (Qh @ Kh.T) / scale           # (tokens, tokens)
        # causal: j > i is inadmissible
        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)
        m = scores.max(axis=1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=1, keepdims=True)
        acc[:, sl] = p @ Vh
    out = acc @ Wo
    return out.T                               # (d_model, seq_len)


def ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, chunk,
                        use_cache=True, mask_new_chunk=True):
    """The cached incremental forward.

    `use_cache=False` is the DELIBERATELY BROKEN variant used to prove that
    the equivalence check is not a tautology: it attends to the new chunk
    only, exactly as a cache that is silently ignored would.

    `mask_new_chunk=False` drops the causal mask among the new tokens.
    """
    d_model, seq_len = x.shape
    d_k = d_model // heads
    scale = math.sqrt(d_k + 1e-9)

    Kc = np.zeros((0, d_model), dtype=float)
    Vc = np.zeros((0, d_model), dtype=float)
    blocks = []

    for start in range(0, seq_len, chunk):
        n = min(chunk, seq_len - start)
        Xn = x[:, start:start + n].T            # (n, d_model)
        Q = Xn @ Wq                             # the repo's convention
        K = Xn @ Wk
        V = Xn @ Wv
        past = Kc.shape[0]
        if use_cache:
            Kc = np.vstack([Kc, K])
            Vc = np.vstack([Vc, V])
        else:
            Kc = K.copy()
            Vc = V.copy()
        total = Kc.shape[0]

        acc = np.zeros((n, d_model), dtype=float)
        idx_new = np.arange(n)
        idx_all = np.arange(total)
        for h in range(heads):
            sl = slice(h * d_k, (h + 1) * d_k)
            Qh = Q[:, sl]
            Kh = Kc[:, sl]
            Vh = Vc[:, sl]
            scores = (Qh @ Kh.T) / scale
            if mask_new_chunk:
                # key j is in the past iff j < past; among the new keys only
                # j <= past + i is admissible.
                admissible = idx_all[None, :] <= (past + idx_new[:, None])
                scores = np.where(admissible, scores, scores - 1e9)
            m = scores.max(axis=1, keepdims=True)
            e = np.exp(scores - m)
            p = e / e.sum(axis=1, keepdims=True)
            acc[:, sl] = p @ Vh
        out = acc @ Wo
        blocks.append(out.T)                    # (d_model, n)

    return np.hstack(blocks)


class RefKVCache(object):
    """The BOOKKEEPING contract of nn::utils::KVCache.

    It owns no attention maths. It owns length, capacity, per-layer storage
    and the error cases. The C++ suite mirrors these rules exactly.
    """

    def __init__(self, capacity=128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.layers = {}                        # layer -> [k_buf, v_buf, len, d_kv]

    def reserve(self, layer, d_kv):
        if d_kv <= 0:
            raise ValueError("d_kv must be positive")
        if layer in self.layers:
            if self.layers[layer][3] != d_kv:
                raise ValueError("layer %d already reserved with d_kv=%d"
                                 % (layer, self.layers[layer][3]))
            return
        self.layers[layer] = [np.zeros((self.capacity, d_kv)),
                              np.zeros((self.capacity, d_kv)),
                              0, d_kv]

    def append(self, layer, k, v):
        if layer not in self.layers:
            raise ValueError("layer %d is not reserved" % layer)
        k = np.asarray(k, dtype=float)
        v = np.asarray(v, dtype=float)
        if k.ndim != 2 or v.ndim != 2:
            raise ValueError("k and v must be 2-D")
        if k.shape[0] != v.shape[0]:
            raise ValueError("k and v must have the same number of rows")
        if k.shape[1] != v.shape[1]:
            raise ValueError("k and v must have the same number of columns")
        if k.shape[1] != self.layers[layer][3]:
            raise ValueError("k has %d columns, the layer was reserved with %d"
                             % (k.shape[1], self.layers[layer][3]))
        if k.shape[0] == 0:
            raise ValueError("cannot append zero rows")
        n = self.layers[layer][2]
        if n + k.shape[0] > self.capacity:
            raise ValueError("cache capacity exceeded")
        self.layers[layer][0][n:n + k.shape[0]] = k
        self.layers[layer][1][n:n + k.shape[0]] = v
        self.layers[layer][2] = n + k.shape[0]

    def size(self, layer):
        if layer not in self.layers:
            raise ValueError("layer %d is not reserved" % layer)
        return self.layers[layer][2]

    def keys(self, layer):
        if layer not in self.layers:
            raise ValueError("layer %d is not reserved" % layer)
        return self.layers[layer][0][:self.layers[layer][2]].copy()

    def values(self, layer):
        if layer not in self.layers:
            raise ValueError("layer %d is not reserved" % layer)
        return self.layers[layer][1][:self.layers[layer][2]].copy()

    def reset(self, layer):
        if layer not in self.layers:
            raise ValueError("layer %d is not reserved" % layer)
        self.layers[layer][2] = 0

    def reset_all(self):
        for layer in self.layers:
            self.layers[layer][2] = 0

    def total_tokens(self):
        return sum(l[2] for l in self.layers.values())

    def memory_bytes(self):
        # two buffers of (capacity, d_kv) doubles per layer
        return sum(2 * self.capacity * l[3] * 8 for l in self.layers.values())


# ===========================================================================
# SECTION 2 -- the equivalence
# ===========================================================================

def make_fixture(seed, d_model, seq_len, heads, scale=0.35):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(d_model, seq_len)) * 0.7
    Wq = rng.normal(size=(d_model, d_model)) * scale
    Wk = rng.normal(size=(d_model, d_model)) * scale
    Wv = rng.normal(size=(d_model, d_model)) * scale
    Wo = rng.normal(size=(d_model, d_model)) * scale
    return x, Wq, Wk, Wv, Wo


def section_equivalence():
    print("\n[SECTION] the cached incremental forward == the full forward")
    cases = [
        # (seed, d_model, seq_len, heads, chunk)
        (20261116, 8, 1, 2, 1),
        (20261116, 8, 2, 2, 1),
        (20261116, 8, 5, 2, 1),
        (20261116, 8, 5, 2, 2),
        (20261116, 8, 7, 4, 3),
        (20261116, 12, 6, 3, 1),
        (20261116, 16, 9, 4, 1),
        (20261116, 16, 9, 4, 4),
        (20261116, 16, 9, 4, 9),
        (20261117, 6, 4, 1, 2),
    ]
    worst = 0.0
    for seed, d_model, seq_len, heads, chunk in cases:
        x, Wq, Wk, Wv, Wo = make_fixture(seed, d_model, seq_len, heads)
        full = ref_mha_full(x, Wq, Wk, Wv, Wo, heads)
        inc = ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, chunk)
        check(full.shape == (d_model, seq_len),
              "full shape d_model=%d seq_len=%d" % (d_model, seq_len))
        check(inc.shape == (d_model, seq_len),
              "incremental shape d_model=%d seq_len=%d" % (d_model, seq_len))
        d = float(np.max(np.abs(full - inc)))
        worst = max(worst, d)
        check(d <= 1e-12,
              "equivalence d_model=%d seq_len=%d heads=%d chunk=%d -- "
              "max_abs_diff = %.17g" % (d_model, seq_len, heads, chunk, d))
    print("  worst max_abs_diff over %d cases: %.17g" % (len(cases), worst))


def section_non_vacuity():
    print("\n[SECTION] the equivalence check is NOT a tautology")
    # A cache that is silently ignored must DIFFER from the full forward.
    x, Wq, Wk, Wv, Wo = make_fixture(20261116, 8, 6, 2)
    full = ref_mha_full(x, Wq, Wk, Wv, Wo, 2)
    broken = ref_mha_incremental(x, Wq, Wk, Wv, Wo, 2, 1, use_cache=False)
    d = float(np.max(np.abs(full - broken)))
    check(d > 1e-6,
          "a cache that is ignored must differ from the full forward "
          "(max_abs_diff = %.17g)" % d)
    print("  ignored-cache max_abs_diff = %.17g (must be > 1e-6)" % d)

    # A missing causal mask among the new tokens must also DIFFER.
    unmasked = ref_mha_incremental(x, Wq, Wk, Wv, Wo, 2, 2, mask_new_chunk=False)
    d2 = float(np.max(np.abs(full - unmasked)))
    check(d2 > 1e-6,
          "dropping the causal mask must differ from the full forward "
          "(max_abs_diff = %.17g)" % d2)
    print("  unmasked max_abs_diff = %.17g (must be > 1e-6)" % d2)

    # The past must MATTER: perturbing token 0 must change a later step.
    x2 = x.copy()
    x2[:, 0] += 3.0
    full2 = ref_mha_full(x2, Wq, Wk, Wv, Wo, 2)
    later = float(np.max(np.abs(full[:, 1:] - full2[:, 1:])))
    check(later > 1e-6,
          "perturbing token 0 must change the later steps (max_abs_diff = %.17g)"
          % later)
    print("  token-0 perturbation effect on later steps = %.17g" % later)

    # And the causality must hold in the other direction too: perturbing the
    # LAST token must leave the FIRST step untouched (it cannot see the
    # future) and must change ONLY the last step.
    #
    # THE FIRST DRAFT OF THIS CHECK PERTURBED TOKEN 0 AND THEN ASSERTED THAT
    # STEP 0 WAS UNCHANGED -- WHICH IS FALSE, BECAUSE TOKEN 0 *IS* STEP 0. It
    # failed by 4.387, and the failure was in the CHECK, not in the model:
    # the property being tested is "a step cannot see a LATER token", so the
    # token that has to be perturbed is the LAST one.
    x3 = x.copy()
    x3[:, -1] += 3.0
    full3 = ref_mha_full(x3, Wq, Wk, Wv, Wo, 2)
    first = float(np.max(np.abs(full[:, 0] - full3[:, 0])))
    check(first < 1e-15,
          "the first step cannot see a later token (max_abs_diff = %.17g)" % first)
    print("  last-token perturbation effect on step 0 = %.17g (must be ~0)" % first)

    last = float(np.max(np.abs(full[:, -1] - full3[:, -1])))
    check(last > 1e-6,
          "perturbing the last token must change the last step "
          "(max_abs_diff = %.17g)" % last)
    print("  last-token perturbation effect on step -1 = %.17g (must be > 1e-6)"
          % last)

    # Only the last step may change: every earlier step is causally before it.
    earlier = float(np.max(np.abs(full[:, :-1] - full3[:, :-1])))
    check(earlier < 1e-15,
          "perturbing the last token must leave every earlier step unchanged "
          "(max_abs_diff = %.17g)" % earlier)
    print("  last-token perturbation effect on steps < -1 = %.17g (must be ~0)"
          % earlier)


def section_cache_semantics():
    print("\n[SECTION] the KV cache bookkeeping contract")
    d = 4
    c = RefKVCache(capacity=8)
    check(c.capacity == 8, "capacity is reported")

    # unreserved layer
    check_throws(lambda: c.size(0), "not reserved",
                 "size() on an unreserved layer throws")

    c.reserve(0, d)
    check(c.size(0) == 0, "a reserved layer starts empty")

    k1 = np.arange(3 * d, dtype=float).reshape(3, d)
    v1 = -np.arange(3 * d, dtype=float).reshape(3, d)
    c.append(0, k1, v1)
    check(c.size(0) == 3, "append grows the layer by the number of rows")
    check_close(c.keys(0), k1, 0.0, "the cached keys are the appended rows")
    check_close(c.values(0), v1, 0.0, "the cached values are the appended rows")

    k2 = np.arange(2 * d, dtype=float).reshape(2, d) + 100.0
    v2 = np.arange(2 * d, dtype=float).reshape(2, d) - 100.0
    c.append(0, k2, v2)
    check(c.size(0) == 5, "a second append grows the layer again")
    check_close(c.keys(0)[:3], k1, 0.0, "the FIRST append is not disturbed")
    check_close(c.keys(0)[3:], k2, 0.0, "the second append lands after the first")
    check_close(c.values(0)[3:], v2, 0.0, "the values land after the first")

    # layer independence
    c.reserve(1, d)
    c.append(1, k1, v1)
    check(c.size(0) == 5 and c.size(1) == 3, "layers are independent")
    check(c.total_tokens() == 8, "total_tokens sums the layers")

    # re-reserving: the SAME width is allowed and must not disturb the layer;
    # a DIFFERENT width is an error.
    c.reserve(0, d)
    check(c.size(0) == 5, "re-reserving a layer with the same width keeps it")
    check_throws(lambda: c.reserve(0, d + 1), "already reserved",
                 "re-reserving a layer with a different width throws")

    # reset
    c.reset(0)
    check(c.size(0) == 0, "reset empties one layer")
    check(c.size(1) == 3, "reset leaves the other layers alone")
    c.reset_all()
    check(c.total_tokens() == 0, "reset_all empties every layer")
    c.append(0, k1, v1)
    check(c.size(0) == 3, "a reset layer is reusable")
    check_close(c.keys(0), k1, 0.0, "a reset layer really starts empty")

    # capacity overflow
    small = RefKVCache(capacity=4)
    small.reserve(0, d)
    small.append(0, np.zeros((4, d)), np.zeros((4, d)))
    check_throws(lambda: small.append(0, np.zeros((1, d)), np.zeros((1, d))),
                 "capacity",
                 "appending past the capacity throws")

    # dimension mismatch
    bad = RefKVCache(capacity=4)
    bad.reserve(0, d)
    check_throws(lambda: bad.append(0, np.zeros((2, d + 1)), np.zeros((2, d + 1))),
                 "columns",
                 "appending the wrong width throws")
    check_throws(lambda: bad.append(0, np.zeros((2, d)), np.zeros((3, d))),
                 "same number of rows",
                 "appending mismatched row counts throws")
    check_throws(lambda: bad.append(0, np.zeros((0, d)), np.zeros((0, d))),
                 "zero rows",
                 "appending zero rows throws")
    check_throws(lambda: bad.append(9, np.zeros((1, d)), np.zeros((1, d))),
                 "not reserved",
                 "appending to an unreserved layer throws")

    # memory accounting: two (capacity, d_kv) buffers per layer
    mem = RefKVCache(capacity=8)
    mem.reserve(0, 4)
    check(mem.memory_bytes() == 2 * 8 * 4 * 8,
          "memory_bytes is two buffers of capacity x d_kv doubles")
    mem.reserve(1, 4)
    check(mem.memory_bytes() == 2 * 2 * 8 * 4 * 8,
          "memory_bytes sums the layers")


def section_pinned_fixture():
    print("\n[SECTION] the pinned fixture")
    seed = 20261116
    d_model, seq_len, heads = 8, 6, 2
    x, Wq, Wk, Wv, Wo = make_fixture(seed, d_model, seq_len, heads)

    full = ref_mha_full(x, Wq, Wk, Wv, Wo, heads)
    inc = ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, 1)
    check_close(full, inc, 1e-12,
                "the pinned fixture's incremental output matches its full output")

    # The output must not be degenerate: a cache that returned zeros would
    # also "match" a full forward that returned zeros.
    check(float(np.max(np.abs(full))) > 1e-3,
          "the pinned output is not all-zero (max_abs = %.17g)"
          % float(np.max(np.abs(full))))
    check(float(np.std(full)) > 1e-3,
          "the pinned output is not constant (std = %.17g)" % float(np.std(full)))

    # The per-step outputs must DIFFER from one another, otherwise the
    # decode would be producing the same answer at every step.
    steps = [float(np.max(np.abs(full[:, t] - full[:, 0]))) for t in range(seq_len)]
    check(min(steps[1:]) > 1e-6,
          "the decode produces different outputs at different steps "
          "(min step-to-step difference = %.17g)" % min(steps[1:]))
    print("  per-step max_abs_diff vs step 0: %s" %
          ", ".join("%.6g" % s for s in steps))

    print("\n  pinned fixture (seed %d, d_model %d, seq_len %d, heads %d):"
          % (seed, d_model, seq_len, heads))
    print("    max_abs(full)            = %.17g" % float(np.max(np.abs(full))))
    print("    max_abs(full - incr)     = %.17g"
          % float(np.max(np.abs(full - inc))))
    print("    rel_err(full, incr)      = %.17g" % rel_err(inc, full))


def emit_cpp():
    """Emit the pinned fixture as C++ literals.

    The repo has been burned by hand-transcribed numbers before (see
    tests/_dump_fixture.py): both numbers were CORRECT for the fixture they
    were given, and the fixture was simply not the oracle's. Pinned numbers
    transfer only across an IDENTICAL fixture, so emit the real one.
    """
    seed = 20261116
    d_model, seq_len, heads = 8, 6, 2
    x, Wq, Wk, Wv, Wo = make_fixture(seed, d_model, seq_len, heads)
    full = ref_mha_full(x, Wq, Wk, Wv, Wo, heads)
    inc = ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, 1)

    def emit(name, arr):
        flat = arr.reshape(-1)
        body = ", ".join(repr(float(v)) for v in flat)
        print("const std::vector<double> %s = {%s};" % (name, body))

    print("// ==== generated by tests/kv_cache_oracle.py --emit-cpp; "
          "do not hand-edit ====")
    print("// seed %d, d_model %d, seq_len %d, heads %d"
          % (seed, d_model, seq_len, heads))
    print("const int KV_D_MODEL = %d;" % d_model)
    print("const int KV_SEQ_LEN = %d;" % seq_len)
    print("const int KV_HEADS = %d;" % heads)
    emit("KV_X_FIX", x)
    emit("KV_WQ_FIX", Wq)
    emit("KV_WK_FIX", Wk)
    emit("KV_WV_FIX", Wv)
    emit("KV_WO_FIX", Wo)
    emit("KV_OUT_FULL_FIX", full)
    emit("KV_OUT_INC_FIX", inc)
    print("// max_abs(full - incremental) in float64 = %.17g"
          % float(np.max(np.abs(full - inc))))
    print("// the C++ test asserts: the incremental decode == the C++ full "
          "forward to the LAST BIT (0.0), and both match the pinned numpy "
          "output to 1e-12 (float64 BLAS vs the C++ explicit loops).")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 72)
    print("tests/kv_cache_oracle.py -- the float64 numpy specification")
    print("=" * 72)
    print("numpy %s | python %s" % (np.__version__, sys.version.split()[0]))

    if "--emit-cpp" in sys.argv:
        emit_cpp()
        return 0

    section_equivalence()
    section_non_vacuity()
    section_cache_semantics()
    section_pinned_fixture()

    print("")
    print("=== SUMMARY: %d passed, %d failed ===" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
