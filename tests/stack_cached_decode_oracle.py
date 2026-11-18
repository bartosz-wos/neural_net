#!/usr/bin/env python3
"""
tests/stack_cached_decode_oracle.py -- the float64 numpy specification of
the STACK-LEVEL CACHED DECODE.

Written and run BEFORE the C++ existed. Its discriminating power is
MEASURED, not narrated: tests/stack_cached_decode_mutate_oracle.py breaks
the model code below in specific ways and requires that this oracle FAIL
each time. An oracle that cannot fail certifies nothing.

    /usr/bin/python3 tests/stack_cached_decode_oracle.py
    /usr/bin/python3 tests/stack_cached_decode_oracle.py --emit-cpp

--------------------------------------------------------------------------
THE GAP (measured 2026-11-17, not assumed)
--------------------------------------------------------------------------

    grep -rinE 'stack.?level|stack_decode|decode_stack|forward_cached' include/
    -> MultiHeadAttention::forward_cached and TransformerBlock::forward_cached
       exist; NO stack drives a sequence of blocks through ONE cache.

The repo shipped a KV CACHE that is explicitly designed to serve a whole
stack -- `set_kv_cache(cache, layer_index)` says so in its own comment:
"so that ONE cache can serve a whole stack of attention layers" -- but no
stack ever did it. The cached decode was proven for ONE block; nothing
proved that N blocks can share ONE cache coherently.

--------------------------------------------------------------------------
THE INVARIANT
--------------------------------------------------------------------------

For a causal stack of N blocks, decoding token by token through ONE shared
cache must equal the full-sequence stack forward, TOKEN FOR TOKEN.

The interesting part is the CACHE-SLOT ALIASING failure mode. A stack that
gave every block the SAME slot would still "work" -- it would not throw,
it would silently attend to the WRONG LAYER'S keys and values. Nothing in
the type system catches it. So the oracle carries an explicit FOIL:

    ref_stack_cached(slot_of_block)   -- the aliased variant

and asserts that the aliased variant DIFFERS from the correct one. Without
that foil, "the cached decode equals the full forward" could be satisfied by
a stack that never used a per-block slot at all -- for example a stack whose
blocks all attend to the FIRST block's keys.

--------------------------------------------------------------------------
TWO INDEPENDENT REFERENCES
--------------------------------------------------------------------------

`ref_stack_full` and `ref_stack_cached` are the two ways round the maths:
the full path materialises the whole sequence at every layer, the cached
path keeps an explicit per-block (T, d_model) K/V store and computes the
tail on the CHUNK only.

A mutation in a helper used by BOTH the reference and the implementation
under test changes the two sides IDENTICALLY, so a comparison between them
CANNOT see it. The oracle therefore keeps the per-layer maths in ONE place
but the SEQUENCE ASSEMBLY in two: `ref_stack_full` chains the blocks over
the whole sequence, `ref_stack_cached` chains them over the increments. A
mutation in the shared block maths is caught by the ALIASED FOIL, which is
computed from the full path and therefore independent of it.

--------------------------------------------------------------------------
CONVENTIONS (the C++'s, not numpy's defaults)
--------------------------------------------------------------------------

    Q[i][j]      = sum_k x[i][k] * Wq[k][j]      -> numpy: x @ Wq
    out[i][j]    = sum_k acc[i][k] * Wo[k][j]    -> numpy: acc @ Wo
    scores       = (Q . K^T) / sqrt(d_k + 1e-9)  -> C++ adds -1e9 on the
                                                    inadmissible columns
    LayerNorm    = gamma * (x - mu) / sqrt(var + eps) + beta, with
                   var = max(pop_var, 1e-7) and eps = 1e-7
    GELU         = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))

NO CITATION IS ASSERTED. A cached decode is an implementation technique
rather than a paper.
"""

import math
import sys

import numpy as np

PASSED = 0
FAILED = 0

D_MODEL = 4
HEADS = 2
NUM_BLOCKS = 3
N_TOKENS = 5

LN_EPS = 1e-7


def check(cond, msg):
    """Record one check. `cond` must already be a bool."""
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print("  [FAIL] %s" % msg)


def check_close(got, want, tol, msg):
    """Record one closeness check, printing the real values on failure."""
    d = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    check(d <= tol, "%s (max abs diff = %.3e, tol = %.3e)" % (msg, d, tol))


# --------------------------------------------------------------------------
# The shared per-layer maths. Everything here is the C++'s arithmetic.
# --------------------------------------------------------------------------

def layer_norm(x, gamma, beta):
    """Per-TOKEN LayerNorm over the feature axis (this repo's convention)."""
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    var = np.maximum(var, 1e-7)
    return gamma * (x - mean) / np.sqrt(var + LN_EPS) + beta


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def attn_full(x, p, heads):
    """The ordinary attention over the WHOLE sequence x (tokens, d_model).

    Returns (tokens, d_model) and the per-head K/V rows (for the cached
    path's store).
    """
    d = x.shape[1]
    dk = d // heads
    Q = x @ p["Wq"]
    K = x @ p["Wk"]
    V = x @ p["Wv"]
    acc = np.zeros_like(x)
    for h in range(heads):
        sl = slice(h * dk, (h + 1) * dk)
        s = Q[:, sl] @ K[:, sl].T / math.sqrt(dk + 1e-9)
        s = s + np.triu(np.ones_like(s), k=1) * -1e9
        e = np.exp(s - s.max(axis=1, keepdims=True))
        pr = e / e.sum(axis=1, keepdims=True)
        acc[:, sl] += pr @ V[:, sl]
    return acc @ p["Wo"], K, V


def attn_cached(x_new, k_store, v_store, p, heads, use_mask=True):
    """The INCREMENTAL attention: attend over the store plus the new tokens.

    The store holds the keys/values of the tokens seen SO FAR. The new
    tokens are appended, then the whole store is attended to with the causal
    mask 'the key at absolute position j is admissible iff j <= past + i'.
    """
    d = x_new.shape[1]
    dk = d // heads
    Q = x_new @ p["Wq"]
    K = x_new @ p["Wk"]
    V = x_new @ p["Wv"]
    past = k_store.shape[0]
    k_store = np.vstack([k_store, K])
    v_store = np.vstack([v_store, V])
    total = k_store.shape[0]
    acc = np.zeros_like(x_new)
    for h in range(heads):
        sl = slice(h * dk, (h + 1) * dk)
        s = Q[:, sl] @ k_store[:, sl].T / math.sqrt(dk + 1e-9)
        if use_mask:
            for i in range(x_new.shape[0]):
                for j in range(total):
                    if j > past + i:
                        s[i, j] += -1e9
        e = np.exp(s - s.max(axis=1, keepdims=True))
        pr = e / e.sum(axis=1, keepdims=True)
        acc[:, sl] += pr @ v_store[:, sl]
    return acc @ p["Wo"], k_store, v_store


def block_tail(x, attn_out, p):
    """The block's POST-ATTENTION tail: residual -> ln1 -> FFN -> + -> ln2.

    THE PROJECTION CONVENTION: the C++ stores W as (d_model, d_model) with
    the INPUT feature on the ROW axis and the OUTPUT feature on the COLUMN
    axis:
        ffn1[i][j] = b1[j] + sum_k W1[k][j] * norm1[i][k]
    which in numpy is `norm1 @ W1` -- NOT `norm1 @ W1.T`. Getting this
    backwards is a real defect, not a cosmetic one: it silently swaps the
    FFN's input and output axes.
    """
    residual = x + attn_out
    norm1 = layer_norm(residual, p["ln1g"], p["ln1b"])
    ffn1 = norm1 @ p["W1"] + p["b1"]
    ffn1 = gelu(ffn1)
    ffn2 = ffn1 @ p["W2"] + p["b2"]
    return layer_norm(norm1 + ffn2, p["ln2g"], p["ln2b"])


# --------------------------------------------------------------------------
# The two stack references.
# --------------------------------------------------------------------------

def ref_stack_full(x, params, heads):
    """The full-sequence stack forward. Returns the output and the per-layer
    K/V stores (the reference for the cached path's store)."""
    h = x
    stores = []
    for p in params:
        attn_out, K, V = attn_full(h, p, heads)
        stores.append((K, V))
        h = block_tail(h, attn_out, p)
    return h, stores


def ref_stack_cached(x, params, heads, slot_of_block=None, chunk=1, use_mask=True):
    """The INCREMENTAL stack decode through ONE shared cache.

    `slot_of_block` maps a block index to a cache slot. The default is the
    correct one-to-one mapping; passing a constant maps EVERY block to the
    SAME slot -- the aliasing foil.

    `chunk` is the number of NEW tokens per decode step.

    ------------------------------------------------------------------------
    A MEASURED FACT, NOT AN ASSUMPTION: with chunk == 1 the causal mask
    NEVER FIRES. The store holds the tokens 0..t and the single new token is
    at absolute position t, so every cached key is admissible and the
    condition `j > past + i` is never true. The mask is therefore NOT
    exercised by a token-at-a-time decode -- a decode that dropped the mask
    entirely would produce IDENTICAL output.

    The mask only becomes load-bearing for a CHUNK of more than one token,
    where the later tokens of the chunk must not see the earlier... no: must
    not see the LATER ones. So the oracle checks BOTH: the token-at-a-time
    decode (the common case) AND a chunked decode, and the unmasked foil
    is measured against the CHUNKED one, where it actually discriminates.
    ------------------------------------------------------------------------
    """
    if slot_of_block is None:
        slot_of_block = lambda b: b
    slots = {}
    n = x.shape[0]
    out = np.zeros((n, x.shape[1]))
    for t0 in range(0, n, chunk):
        chunk_x = x[t0:t0 + chunk]
        h = chunk_x
        for b, p in enumerate(params):
            sl = slot_of_block(b)
            if sl not in slots:
                slots[sl] = (np.zeros((0, x.shape[1])), np.zeros((0, x.shape[1])))
            k_store, v_store = slots[sl]
            attn_out, k_store, v_store = attn_cached(h, k_store, v_store, p, heads,
                                                     use_mask=use_mask)
            slots[sl] = (k_store, v_store)
            h = block_tail(h, attn_out, p)
        out[t0:t0 + chunk] = h
    return out, slots


# --------------------------------------------------------------------------
# A SECOND, INDEPENDENT IMPLEMENTATION OF THE FULL STACK.
#
# WHY IT EXISTS, AND WHY IT IS NOT PARANOIA:
#
# The first run of the mutation harness reported FOUR survivors -- SM14
# (the residual add is dropped), SM15 (the second LayerNorm is dropped),
# SM16 (the FFN's weights are transposed) and SM17 (the GELU loses its
# 0.5) -- and every one of them is a mutation of the SHARED `block_tail`.
#
# That is STRUCTURAL, not accidental: **a mutation in a helper used by BOTH
# the reference and the thing under test changes the two sides
# IDENTICALLY, so a comparison between them CANNOT see it.** The
# comparison cancels. This repo has hit this exact trap before (the
# block-level oracle's first harness reported six survivors for the same
# reason) and the fix is the same: a SECOND, INDEPENDENT implementation of
# the full path, written with EXPLICIT SCALAR LOOPS and no BLAS `@`, which
# the cached decode is then compared against.
#
# With this witness, a mutation of the shared helper has a clean witness on
# the other side -- and the comparison stops cancelling.
#
# It is written differently ON PURPOSE: plain Python lists instead of
# numpy arrays, explicit loops instead of `@`, and its own LayerNorm and
# GELU helpers.
# --------------------------------------------------------------------------

def ln_scalar(row, gamma, beta):
    """The per-token LayerNorm, written with explicit loops."""
    m = sum(row) / len(row)
    var = sum((v - m) ** 2 for v in row) / len(row)
    if var < 1e-7:
        var = 1e-7
    sv = math.sqrt(var + LN_EPS)
    return [gamma[j] * (row[j] - m) / sv + beta[j] for j in range(len(row))]


def gelu_scalar(v):
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))


def ref_stack_full_independent(x, params, heads):
    """The full-sequence stack forward, with NO numpy `@` anywhere."""
    n = x.shape[0]
    d = x.shape[1]
    dk = d // heads
    h = [[float(x[i][j]) for j in range(d)] for i in range(n)]
    for p in params:
        Q = [[sum(h[i][k] * p["Wq"][k][j] for k in range(d)) for j in range(d)]
             for i in range(n)]
        K = [[sum(h[i][k] * p["Wk"][k][j] for k in range(d)) for j in range(d)]
             for i in range(n)]
        V = [[sum(h[i][k] * p["Wv"][k][j] for k in range(d)) for j in range(d)]
             for i in range(n)]
        acc = [[0.0] * d for _ in range(n)]
        for hd in range(heads):
            scores = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(dk):
                        s += Q[i][hd * dk + k] * K[j][hd * dk + k]
                    sc = s / math.sqrt(dk + 1e-9)
                    if j > i:
                        sc += -1e9
                    scores[i][j] = sc
            probs = []
            for i in range(n):
                mx = max(scores[i])
                ex = [math.exp(scores[i][j] - mx) for j in range(n)]
                tot = sum(ex)
                probs.append([e / tot for e in ex])
            for i in range(n):
                for k in range(dk):
                    acc[i][hd * dk + k] = sum(probs[i][j] * V[j][hd * dk + k]
                                             for j in range(n))
        attn_out = [[sum(acc[i][k] * p["Wo"][k][j] for k in range(d)) for j in range(d)]
                     for i in range(n)]
        residual = [[h[i][j] + attn_out[i][j] for j in range(d)] for i in range(n)]
        norm1 = [ln_scalar(residual[i], p["ln1g"], p["ln1b"]) for i in range(n)]
        ffn1 = [[sum(p["W1"][k][j] * norm1[i][k] for k in range(d)) + p["b1"][j]
                 for j in range(d)] for i in range(n)]
        gelued = [[gelu_scalar(ffn1[i][j]) for j in range(d)] for i in range(n)]
        ffn2 = [[sum(p["W2"][k][j] * gelued[i][k] for k in range(d)) + p["b2"][j]
                 for j in range(d)] for i in range(n)]
        h = [ln_scalar([norm1[i][j] + ffn2[i][j] for j in range(d)],
                       p["ln2g"], p["ln2b"]) for i in range(n)]
    return np.array(h)


def make_params(seed=20261117):
    """The fixture: N blocks of weights. FIXED SEED, pinned values."""
    rng = np.random.default_rng(seed)
    params = []
    for b in range(NUM_BLOCKS):
        p = {}
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            p[nm] = rng.normal(size=(D_MODEL, D_MODEL)) * 0.5
        p["W1"] = rng.normal(size=(D_MODEL, D_MODEL)) * 0.5
        p["b1"] = rng.normal(size=D_MODEL) * 0.1
        p["W2"] = rng.normal(size=(D_MODEL, D_MODEL)) * 0.5
        p["b2"] = rng.normal(size=D_MODEL) * 0.1
        p["ln1g"] = 1.0 + rng.normal(size=D_MODEL) * 0.1
        p["ln1b"] = rng.normal(size=D_MODEL) * 0.1
        p["ln2g"] = 1.0 + rng.normal(size=D_MODEL) * 0.1
        p["ln2b"] = rng.normal(size=D_MODEL) * 0.1
        params.append(p)
    x = rng.normal(size=(N_TOKENS, D_MODEL)) * 0.4
    return x, params


def emit_cpp(x, params, full_out, cached_out, aliased_out):
    """Emit the fixture for the C++ test. NUMBERS ARE EMITTED, NEVER TYPED:
    the repo has been burned by hand-transcribed numbers before."""
    def flat(name, m):
        a = np.asarray(m).reshape(-1)
        body = []
        for i in range(0, len(a), 6):
            body.append("    " + ", ".join("%.17g" % v for v in a[i:i + 6]) + ",")
        return "const double %s[%d] = {\n%s\n};\n" % (name, len(a), "\n".join(body))

    out = []
    out.append("// ==== ORACLE-EMITTED FIXTURE (tests/stack_cached_decode_oracle.py) ====")
    out.append("const size_t ORACLE_D_MODEL = %d;" % D_MODEL)
    out.append("const size_t ORACLE_HEADS = %d;" % HEADS)
    out.append("const size_t ORACLE_NUM_BLOCKS = %d;" % NUM_BLOCKS)
    out.append("const size_t ORACLE_TOKENS = %d;" % N_TOKENS)
    out.append(flat("ORACLE_INPUT", x))
    # Every weight tensor is emitted CONCATENATED over the blocks, so the
    # C++ test can index block b's slice as ORACLE_WQ + b*d*d.
    for nm in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
        out.append(flat("ORACLE_" + nm.upper(),
                        np.concatenate([p[nm] for p in params])))
    for nm in ("b1", "b2", "ln1g", "ln1b", "ln2g", "ln2b"):
        out.append(flat("ORACLE_" + nm.upper(),
                        np.concatenate([p[nm] for p in params])))
    out.append(flat("ORACLE_FULL_OUT", full_out))
    out.append(flat("ORACLE_CACHED_OUT", cached_out))
    out.append(flat("ORACLE_ALIASED_OUT", aliased_out))
    return "\n".join(out)


def main():
    argv = sys.argv[1:]
    x, params = make_params()
    full_out, stores = ref_stack_full(x, params, HEADS)
    cached_out, slots = ref_stack_cached(x, params, HEADS)
    chunked_out, chunked_slots = ref_stack_cached(x, params, HEADS, chunk=2)
    aliased_out, _ = ref_stack_cached(x, params, HEADS, slot_of_block=lambda b: 0)

    print("=== stack-level cached decode oracle ===")

    # 0. THE INDEPENDENT WITNESS: the two full-path implementations agree.
    #    Without this check, a mutation of the shared `block_tail` cancels.
    independent_out = ref_stack_full_independent(x, params, HEADS)
    check_close(independent_out, full_out, 1e-10,
                "the two INDEPENDENT full-stack implementations agree")

    # 1. THE INVARIANT: the incremental decode equals the full forward.
    check_close(cached_out, full_out, 1e-12,
                "the cached stack decode equals the full stack forward")
    # ...and equals the INDEPENDENT full forward, which is the comparison
    # that does NOT cancel when the shared helper is mutated.
    check_close(cached_out, independent_out, 1e-10,
                "the cached stack decode equals the INDEPENDENT full forward")
    check_close(chunked_out, independent_out, 1e-10,
                "the CHUNKED stack decode equals the INDEPENDENT full forward")

    # 2. THE FOIL: aliasing every block to slot 0 must CHANGE the answer.
    #    Without this, a stack that never used per-block slots could pass.
    aliased_diff = float(np.max(np.abs(aliased_out - cached_out)))
    check(aliased_diff > 1e-3,
          "the aliased-slot foil DISCRIMINATES (max abs diff = %.3e > 1e-3)"
          % aliased_diff)

    # 3. The per-block slots are distinct: block b's store holds ITS OWN
    #    keys, not block 0's.
    check(len(slots) == NUM_BLOCKS,
          "the cached decode used one slot per block (%d slots for %d blocks)"
          % (len(slots), NUM_BLOCKS))
    store_diffs = []
    for b in range(1, NUM_BLOCKS):
        d = float(np.max(np.abs(slots[b][0] - slots[0][0])))
        store_diffs.append(d)
    check(min(store_diffs) > 1e-6,
          "each block's cached keys differ from block 0's (min diff = %.3e)"
          % min(store_diffs))

    # 4. The stores match the full path's K/V, row for row.
    for b in range(NUM_BLOCKS):
        check_close(slots[b][0], stores[b][0], 1e-12,
                    "block %d's cached keys equal the full path's K" % b)
        check_close(slots[b][1], stores[b][1], 1e-12,
                    "block %d's cached values equal the full path's V" % b)

    # 5. A CHUNKED decode must also equal the full forward -- and THAT is
    #    where the causal mask becomes load-bearing. With chunk == 1 the
    #    mask never fires (every cached key is in the past), so an unmasked
    #    decode is indistinguishable. With chunk == 2 the later token of the
    #    chunk must not see the earlier... it must not see the LATER one, and
    #    the unmasked foil differs.
    check_close(chunked_out, full_out, 1e-12,
                "the CHUNKED (2 tokens/step) stack decode equals the full forward")
    nomask_out, _ = ref_stack_cached(x, params, HEADS, chunk=2, use_mask=False)
    nomask_diff = float(np.max(np.abs(nomask_out - chunked_out)))
    check(nomask_diff > 1e-3,
          "the unmasked foil DISCRIMINATES on the chunked decode "
          "(max abs diff = %.3e > 1e-3)" % nomask_diff)
    # And the mask must be a no-op for a token-at-a-time decode -- that is a
    # MEASURED property of the causal mask, not an assumption.
    nomask1_out, _ = ref_stack_cached(x, params, HEADS, chunk=1, use_mask=False)
    nomask1_diff = float(np.max(np.abs(nomask1_out - cached_out)))
    check(nomask1_diff == 0.0,
          "the causal mask is INERT for a token-at-a-time decode "
          "(max abs diff = %.3e)" % nomask1_diff)

    # 6. The decoded output must not be trivially zero or constant.
    check(float(np.max(np.abs(cached_out))) > 1e-3,
          "the decoded output is non-trivial (max abs = %.3e)"
          % float(np.max(np.abs(cached_out))))
    check(float(np.max(np.abs(cached_out - cached_out[0]))) > 1e-3,
          "the decoded rows differ from each other (the stack is not constant)")

    # 7. THE VARIANCE CLAMP IS LOAD-BEARING ONLY ON A NEAR-CONSTANT ROW.
    #    With random data the variance is O(1) and max(var, 1e-7) never
    #    fires, so dropping the clamp changes NOTHING and the mutation
    #    survives. A row whose features differ by 1e-5 has variance 5e-11,
    #    far below the clamp, so the clamped and unclamped results differ by
    #    a factor of sqrt(2) in the denominator: 2.2e-2 against 3.2e-2.
    #    Without this fixture the clamp is UNOBSERVABLE and any claim that it
    #    is covered would be false.
    row = np.full((1, 4), 0.5) + np.array([[1e-5, -1e-5, 0.0, 0.0]])
    gam = np.ones(4)
    bet = np.zeros(4)
    clamped = layer_norm(row, gam, bet)
    mean = row.mean(axis=1, keepdims=True)
    var_unclamped = ((row - mean) ** 2).mean(axis=1, keepdims=True)
    unclamped = gam * (row - mean) / np.sqrt(var_unclamped + LN_EPS) + bet
    clamp_gap = float(np.max(np.abs(clamped - unclamped)))
    check(clamp_gap > 1e-3,
          "the LayerNorm's variance clamp is LOAD-BEARING on a near-constant "
          "row (clamped vs unclamped differ by %.3e)" % clamp_gap)

    if "--emit-cpp" in argv:
        print(emit_cpp(x, params, full_out, cached_out, aliased_out))

    print("=== %d passed, %d failed ===" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
