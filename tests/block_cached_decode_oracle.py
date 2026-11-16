#!/usr/bin/env python3
"""
tests/block_cached_decode_oracle.py -- the float64 numpy specification of
the BLOCK-LEVEL CACHED DECODE and of the ABSOLUTE positional encoding.

This file was written and run BEFORE the C++ existed. Its discriminating
power was MEASURED, not narrated: tests/block_cached_decode_mutate_oracle.py
breaks the model code below in specific ways and requires that this oracle
FAIL each time. An oracle that cannot fail certifies nothing.

    /usr/bin/python3 tests/block_cached_decode_oracle.py
    /usr/bin/python3 tests/block_cached_decode_oracle.py --emit-cpp

--------------------------------------------------------------------------
THE GAP (measured 2026-11-16, not assumed)
--------------------------------------------------------------------------

    grep -rinE 'residual.?stream|hidden.?state.?cache|x.?cache' include/
    -> 3 hits, ALL COMMENTS -> ZERO implementation.

The repo shipped an attention-level KV cache and a cached incremental path
for MultiHeadAttention, but `TransformerBlock` -- the block every decoder
stack is built from -- has NO cached forward at all. A block-level decode is
NOT shipped.

--------------------------------------------------------------------------
THE QUEUE'S PREMISE, AND WHY IT IS REFUTED HERE
--------------------------------------------------------------------------

The expansion queue asserted that a block-level decode "needs a SECOND cache
[for the residual stream], and the FFN's position-wise nature does not remove
the requirement". THAT IS FALSE IN THIS REPO, and the reason is a repo
convention worth pinning:

    include/nn/layers/normalization/layer_norm.cpp normalises over the
    FEATURE axis of each ROW:

        size_t batch = input.rows;      // tokens
        size_t features = input.cols;   // d_model
        for (size_t b = 0; b < batch; ++b) { ...mean over features... }

So a LayerNorm over a (tokens, d_model) tensor is PER TOKEN, and the block's
post-attention tail -- residual -> ln1 -> FFN -> + -> ln2 -- is therefore
POSITION-WISE. The ONLY cross-token coupling in the whole block is the
ATTENTION, which the shipped KV cache already covers exactly.

This file MEASURES that claim rather than asserting it, and it measures the
COUNTERFACTUAL too: a tail whose LayerNorm normalises over the TOKEN axis
does NOT have the property. Without the counterfactual the zero would be a
tautology rather than evidence.

--------------------------------------------------------------------------
THE INVARIANT
--------------------------------------------------------------------------

For a causal block, decoding token by token with a cache must equal the
full-sequence forward, TOKEN FOR TOKEN. It is pinned EXACTLY (0.0), because
the cached path performs the same arithmetic in the same order and the tail
is row-wise, so it produces bit-identical values. A tolerance would hide a
decode that attends to the wrong tokens.

TWO INDEPENDENT REFERENCES, and the second one exists for a measured reason
--------------------------------------------------------------------------
`ref_block_full` and `ref_block_cached` are the two ways round the maths:
the full path materialises the whole sequence, the cached path keeps an
explicit (T, d_model) K/V store and computes the tail on the CHUNK only.

**THE FIRST DRAFT OF THIS ORACLE HAD ONLY ONE REFERENCE, AND ITS OWN
MUTATION HARNESS FOUND SIX SURVIVORS** -- every mutation of the SHARED
`ref_block_tail` (dropping the second LayerNorm, dropping the residual add,
moving the GELU, transposing the FFN's weights, dropping the GELU's 0.5,
dropping the LayerNorm's variance clamp). The reason is structural, and it
is the trap this repo has hit before: **a mutation in a helper used by BOTH
the reference and the implementation under test changes the two sides
IDENTICALLY, so a comparison between them CANNOT SEE IT.** The comparison
cancels.

So there is now a SECOND, INDEPENDENT implementation of the full block
(`ref_block_full_independent`), written from the C++ source with EXPLICIT
SCALAR LOOPS and no BLAS `@`, and the oracle asserts the two agree. A
mutation in the shared helper now has a clean witness on the other side.

The variance clamp needs a DEGENERATE fixture to be observable at all: with
random data the variance is O(1) and `max(var, 1e-7)` never fires, so
dropping the clamp changes nothing. `section_degenerate_layer_norm` pins it
with a CONSTANT row, and asserts that the clamp is LOAD-BEARING there.

--------------------------------------------------------------------------
NO CITATION IS ASSERTED
--------------------------------------------------------------------------

A cached decode is an implementation technique rather than a paper, and this
repo's expansion queue has produced fifteen wrong arXiv identifiers -- every
one a plausible-looking neighbour of the right paper. The one reference
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
    THE THROW.
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


# ==========================================================================
# SECTION 1 -- the references
# ==========================================================================

def ref_layernorm_rows(x, gamma, beta, eps=1e-7):
    """The repo's LayerNorm, over the FEATURE axis of each ROW.

    Mirrors include/nn/layers/normalization/layer_norm.cpp exactly:

        batch = input.rows;  features = input.cols;
        mean  = sum_f x[b][f] / features
        var   = sum_f (x[b][f] - mean)^2 / features
        var   = max(var, 1e-7)
        sqrt_var = sqrt(var + eps)
        out[b][f] = gamma[f] * (x[b][f] - mean) / sqrt_var + beta[f]

    THE AXIS IS THE POINT. Normalising over the FEATURES of each row makes
    the operation PER TOKEN; normalising over the ROWS would make it
    sequence-coupled, and the block's tail would then NOT be position-wise.
    """
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    var = np.maximum(var, 1e-7)
    inv = 1.0 / np.sqrt(var + eps)
    return gamma * (x - mean) * inv + beta


def ref_layernorm_tokens(x, gamma, beta, eps=1e-7):
    """The COUNTERFACTUAL: the same LayerNorm over the TOKEN axis.

    Used only to prove that the "the tail is position-wise" check is not a
    tautology -- this variant does NOT have the property.
    """
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    var = np.maximum(var, 1e-7)
    inv = 1.0 / np.sqrt(var + eps)
    return gamma * (x - mean) * inv + beta


def ref_layernorm_independent(x, gamma, beta, eps=1e-7):
    """A SECOND implementation of the repo's LayerNorm, with EXPLICIT LOOPS.

    Deliberately not written in terms of ref_layernorm_rows: this is the
    witness that makes a mutation of the shared helper OBSERVABLE. Its
    summation order also differs from numpy's, so the agreement between the
    two is a real cross-check rather than an identity.
    """
    tokens, features = x.shape
    out = np.zeros((tokens, features), dtype=float)
    for b in range(tokens):
        mean = 0.0
        for f in range(features):
            mean += x[b][f]
        mean /= features
        var = 0.0
        for f in range(features):
            diff = x[b][f] - mean
            var += diff * diff
        var /= features
        var = max(var, 1e-7)
        sqrt_var = math.sqrt(var + eps)
        for f in range(features):
            nrm = (x[b][f] - mean) / sqrt_var
            out[b][f] = gamma[f] * nrm + beta[f]
    return out


def ref_mha_full(x, Wq, Wk, Wv, Wo, heads):
    """The repo's plain causal multi-head attention, full sequence.

    Follows include/nn/layers/attention/transformer.cpp exactly:
      * input layout (d_model, seq_len); tokens are COLUMNS of the input;
      * internal layout (tokens, d_model);
      * Q = X @ Wq, K = X @ Wk, V = X @ Wv (THE REPO'S CONVENTION -- the
        textbook convention for a (out, in) projection is X @ W.T, and it
        is WRONG here; see the note in the C++ implementation);
      * per-head slice is the column block [h*d_k, (h+1)*d_k);
      * scores = Q K^T / sqrt(d_k + 1e-9);
      * causal mask adds -1e9 to j > i;
      * softmax is max-subtracted;
      * output = acc @ Wo, then back to (d_model, seq_len).
    """
    d_model, seq_len = x.shape
    X = x.T                                    # (tokens, d_model)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_k = d_model // heads
    scale = math.sqrt(d_k + 1e-9)

    acc = np.zeros((seq_len, d_model), dtype=float)
    idx = np.arange(seq_len)
    for h in range(heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        scores = (Q[:, sl] @ K[:, sl].T) / scale
        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)
        m = scores.max(axis=1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=1, keepdims=True)
        acc[:, sl] = p @ V[:, sl]
    return (acc @ Wo).T


def ref_gelu(x):
    """The repo's tanh-approximation GELU, from transformer.cpp."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (x + 0.044715 * x ** 3)))


class BlockParams(object):
    """The block's parameters, in the repo's layouts.

    Wq/Wk/Wv/Wo are (d_model, d_model); W1/W2 are (d_model, d_model);
    b1/b2 are (1, d_model); ln1/ln2 are (d_model,) gamma/beta pairs.
    """

    def __init__(self, Wq, Wk, Wv, Wo, W1, b1, W2, b2,
                 ln1_g, ln1_b, ln2_g, ln2_b):
        self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.ln1_g, self.ln1_b = ln1_g, ln1_b
        self.ln2_g, self.ln2_b = ln2_g, ln2_b


def ref_block_tail(x_tokens, attn_tokens, P, ln=None):
    """The block's POST-ATTENTION tail.

        residual = x + attn
        norm1    = LayerNorm(residual)
        ffn1     = W1 @ norm1 + b1 ; GELU
        ffn2     = W2 @ gelu(ffn1) + b2
        ffn_res  = norm1 + ffn2
        out      = LayerNorm(ffn_res)

    `ln` selects the LayerNorm axis: 'rows' is the repo's convention,
    'tokens' is the counterfactual used only by the refutation check.

    The tail is written on TOKEN ROWS: x_tokens and attn_tokens are
    (tokens, d_model), which is the block's internal layout.
    """
    norm = ref_layernorm_tokens if ln == "tokens" else ref_layernorm_rows
    residual = x_tokens + attn_tokens
    norm1 = norm(residual, P.ln1_g, P.ln1_b)
    ffn1 = norm1 @ P.W1 + P.b1
    ffn1 = ref_gelu(ffn1)
    ffn2 = ffn1 @ P.W2 + P.b2
    ffn_res = norm1 + ffn2
    return norm(ffn_res, P.ln2_g, P.ln2_b)


def ref_block_tail_independent(x_tokens, attn_tokens, P):
    """A SECOND implementation of the tail, with EXPLICIT SCALAR LOOPS.

    Transcribed from TransformerBlock::forward in
    include/nn/layers/attention/transformer.cpp, term by term. It is NOT
    written in terms of ref_block_tail, and that is the whole point: it is
    the witness that makes a mutation of the shared tail observable.
    """
    residual = x_tokens + attn_tokens
    norm1 = ref_layernorm_independent(residual, P.ln1_g, P.ln1_b)
    tokens, d_model = norm1.shape

    ffn1 = np.zeros((tokens, d_model), dtype=float)
    for i in range(tokens):
        for j in range(d_model):
            val = P.b1[0][j]
            for k in range(d_model):
                val += P.W1[k][j] * norm1[i][k]
            ffn1[i][j] = val

    for i in range(tokens):
        for j in range(d_model):
            x_g = ffn1[i][j]
            ffn1[i][j] = 0.5 * x_g * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                                                       (x_g + 0.044715 * x_g ** 3)))

    ffn2 = np.zeros((tokens, d_model), dtype=float)
    for i in range(tokens):
        for j in range(d_model):
            val = P.b2[0][j]
            for k in range(d_model):
                val += P.W2[k][j] * ffn1[i][k]
            ffn2[i][j] = val

    ffn_res = norm1 + ffn2
    return ref_layernorm_independent(ffn_res, P.ln2_g, P.ln2_b)


def ref_block_full(x, P, heads):
    """The full-sequence block. x is (d_model, seq_len) -- the repo layout.

    TransformerBlock::forward takes (d_model, seq_len) and works on
    (tokens, d_model) internally.
    """
    d_model, seq_len = x.shape
    attn_out = ref_mha_full(x, P.Wq, P.Wk, P.Wv, P.Wo, heads)   # (d_model, seq_len)
    x_tokens = x.T                                               # (tokens, d_model)
    attn_tokens = attn_out.T                                     # (tokens, d_model)
    out = ref_block_tail(x_tokens, attn_tokens, P)               # (tokens, d_model)
    return out.T                                                 # (d_model, seq_len)


def ref_block_full_independent(x, P, heads):
    """A SECOND, fully independent implementation of the full block.

    Transcribed from include/nn/layers/attention/transformer.cpp with
    EXPLICIT SCALAR LOOPS and no BLAS `@` anywhere. It shares NO helper
    with ref_block_full, so it is a clean witness for every mutation of
    the shared code.
    """
    d_model, seq_len = x.shape
    d_k = d_model // heads
    scale = math.sqrt(d_k + 1e-9)

    # The input's COLUMNS are the tokens: X[i][k] = input[k][i].
    X = [[x[f][s] for f in range(d_model)] for s in range(seq_len)]

    Q = [[0.0] * d_model for _ in range(seq_len)]
    K = [[0.0] * d_model for _ in range(seq_len)]
    V = [[0.0] * d_model for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(d_model):
            qv = kv = vv = 0.0
            for k in range(d_model):
                qv += X[i][k] * P.Wq[k][j]
                kv += X[i][k] * P.Wk[k][j]
                vv += X[i][k] * P.Wv[k][j]
            Q[i][j] = qv
            K[i][j] = kv
            V[i][j] = vv

    acc = [[0.0] * d_model for _ in range(seq_len)]
    for h in range(heads):
        for i in range(seq_len):
            scores = [0.0] * seq_len
            for j in range(seq_len):
                s = 0.0
                for dk in range(d_k):
                    s += Q[i][h * d_k + dk] * K[j][h * d_k + dk]
                scores[j] = s / scale
            for j in range(seq_len):
                if j > i:
                    scores[j] += -1e9
            mx = scores[0]
            for j in range(1, seq_len):
                if scores[j] > mx:
                    mx = scores[j]
            se = 0.0
            for j in range(seq_len):
                scores[j] = math.exp(scores[j] - mx)
                se += scores[j]
            for j in range(seq_len):
                scores[j] /= se
            # The weighted sum runs INSIDE the per-row loop, exactly as the
            # C++ does (it forms `attn` per (i, dk) and then accumulates it
            # into output_acc).
            for dk in range(d_k):
                val = 0.0
                for j in range(seq_len):
                    val += scores[j] * V[j][h * d_k + dk]
                acc[i][h * d_k + dk] += val

    # The output projection, back into the (d_model, seq_len) layout.
    attn_out = [[0.0] * seq_len for _ in range(d_model)]
    for i in range(seq_len):
        for j in range(d_model):
            val = 0.0
            for k in range(d_model):
                val += acc[i][k] * P.Wo[k][j]
            attn_out[j][i] = val

    x_tokens = np.array([[x[f][s] for f in range(d_model)] for s in range(seq_len)],
                        dtype=float)
    attn_tokens = np.array([[attn_out[f][s] for f in range(d_model)]
                             for s in range(seq_len)], dtype=float)
    out = ref_block_tail_independent(x_tokens, attn_tokens, P)
    return out.T


def ref_block_cached(x, P, heads, chunk, use_cache=True, mask_new_chunk=True,
                     ln=None):
    """The cached block: the attention over a K/V store, the tail per chunk.

    Written the OTHER WAY ROUND from ref_block_full: the K/V store is
    explicit and the tail is computed on the chunk only, rather than the
    whole sequence being projected at once.

    `use_cache=False` is the DELIBERATELY BROKEN variant used to prove the
    equivalence check is not a tautology: it attends to the new chunk only,
    exactly as a cache that is silently ignored would.

    `mask_new_chunk=False` drops the causal mask among the new tokens.
    """
    d_model, seq_len = x.shape
    d_k = d_model // heads
    scale = math.sqrt(d_k + 1e-9)

    Kc = np.zeros((0, d_model), dtype=float)
    Vc = np.zeros((0, d_model), dtype=float)
    outs = []

    for start in range(0, seq_len, chunk):
        n = min(chunk, seq_len - start)
        Xn = x[:, start:start + n].T                 # (n, d_model)
        Q = Xn @ P.Wq
        K = Xn @ P.Wk
        V = Xn @ P.Wv
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
            scores = (Q[:, sl] @ Kc[:, sl].T) / scale
            if mask_new_chunk:
                # key j is in the past iff j < past; among the new keys only
                # j <= past + i is admissible.
                admissible = idx_all[None, :] <= (past + idx_new[:, None])
                scores = np.where(admissible, scores, scores - 1e9)
            m = scores.max(axis=1, keepdims=True)
            e = np.exp(scores - m)
            p = e / e.sum(axis=1, keepdims=True)
            acc[:, sl] = p @ Vc[:, sl]
        attn_tokens = (acc @ P.Wo)                    # (n, d_model)
        out = ref_block_tail(Xn, attn_tokens, P, ln=ln)
        outs.append(out.T)                            # (d_model, n)

    return np.hstack(outs)


def ref_pe_table(max_len, d_model):
    """The repo's PositionalEncoding table: pe[pos][feature]."""
    pe = np.zeros((max_len, d_model), dtype=float)
    for pos in range(max_len):
        for i in range(d_model):
            angle = pos / (10000.0 ** (2.0 * i / d_model))
            pe[pos][i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return pe


def ref_pe_forward(x, pe):
    """The repo's PositionalEncoding::forward -- the LOCAL column index.

    output[f][s] = input[f][s] + pe[s][f]
    """
    d_model, seq_len = x.shape
    out = np.zeros_like(x)
    for s in range(seq_len):
        out[:, s] = x[:, s] + pe[s]
    return out


def ref_pe_forward_at(x, pe, offset):
    """The ABSOLUTE-position form: output[f][s] = input[f][s] + pe[offset+s][f].

    This is what a decode needs: the new token's absolute position is
    `offset + s`, not `s`.
    """
    d_model, seq_len = x.shape
    out = np.zeros_like(x)
    for s in range(seq_len):
        out[:, s] = x[:, s] + pe[offset + s]
    return out


# ==========================================================================
# SECTION 2 -- the equivalence
# ==========================================================================

def make_fixture(seed, d_model, seq_len, heads, scale=0.35):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(d_model, seq_len)) * 0.7
    Wq = rng.normal(size=(d_model, d_model)) * scale
    Wk = rng.normal(size=(d_model, d_model)) * scale
    Wv = rng.normal(size=(d_model, d_model)) * scale
    Wo = rng.normal(size=(d_model, d_model)) * scale
    W1 = rng.normal(size=(d_model, d_model)) * scale
    b1 = rng.normal(size=(1, d_model)) * 0.1
    W2 = rng.normal(size=(d_model, d_model)) * scale
    b2 = rng.normal(size=(1, d_model)) * 0.1
    ln1_g = rng.normal(size=d_model) * 0.3 + 1.0
    ln1_b = rng.normal(size=d_model) * 0.1
    ln2_g = rng.normal(size=d_model) * 0.3 + 1.0
    ln2_b = rng.normal(size=d_model) * 0.1
    P = BlockParams(Wq, Wk, Wv, Wo, W1, b1, W2, b2,
                    ln1_g, ln1_b, ln2_g, ln2_b)
    return x, P


def section_independent_agreement():
    """THE TWO INDEPENDENT IMPLEMENTATIONS MUST AGREE.

    This section is what gives the oracle its teeth against a mutation of
    the SHARED code: `ref_block_full` (BLAS) and `ref_block_full_independent`
    (explicit scalar loops, transcribed from the C++) share no helper, so a
    defect in either one is visible.

    The first draft of this oracle had ONE reference, and its own mutation
    harness reported SIX survivors -- every mutation of the shared tail.
    A comparison between a reference and an implementation that both use
    the same helper CANNOT see a defect in that helper.
    """
    print("\n[SECTION] the two independent implementations agree")
    for seed, d_model, seq_len, heads in [
            (20261116, 8, 6, 2),
            (20261116, 4, 5, 1),
            (20261117, 6, 4, 2),
            (20261118, 3, 3, 3)]:
        x, P = make_fixture(seed, d_model, seq_len, heads)
        a = ref_block_full(x, P, heads)
        b = ref_block_full_independent(x, P, heads)
        d = float(np.max(np.abs(a - b)))
        check(d <= 1e-12,
              "the BLAS reference and the explicit-loop reference agree -- "
              "d_model=%d seq_len=%d heads=%d, max_abs_diff = %.17g"
              % (d_model, seq_len, heads, d))
        print("  d_model=%d seq_len=%d heads=%d: max_abs_diff = %.17g"
              % (d_model, seq_len, heads, d))

    # The independent implementation is only a witness if it is not the
    # SAME code, so it must not be a bit-for-bit copy. Its summation order
    # differs, which is why the agreement above is a real cross-check.
    x, P = make_fixture(20261116, 8, 6, 2)
    a = ref_block_full(x, P, 2)
    b = ref_block_full_independent(x, P, 2)
    check(not np.array_equal(a, b),
          "the two references are not bit-identical (they sum in different "
          "orders), so the agreement above is a genuine cross-check")


def section_equivalence():
    print("\n[SECTION] the cached block decode == the full block forward")
    cases = [
        # (seed, d_model, seq_len, heads, chunk)
        (20261116, 8, 1, 2, 1),
        (20261116, 8, 2, 2, 1),
        (20261116, 8, 5, 2, 1),
        (20261116, 8, 5, 2, 2),
        (20261116, 8, 6, 2, 3),
        (20261116, 8, 6, 2, 6),
        (20261116, 12, 6, 3, 1),
        (20261116, 16, 9, 4, 1),
        (20261116, 16, 9, 4, 4),
        (20261116, 16, 9, 4, 9),
        (20261117, 6, 4, 1, 2),
    ]
    worst = 0.0
    for seed, d_model, seq_len, heads, chunk in cases:
        x, P = make_fixture(seed, d_model, seq_len, heads)
        full = ref_block_full(x, P, heads)
        inc = ref_block_cached(x, P, heads, chunk)
        check(full.shape == (d_model, seq_len),
              "full shape d_model=%d seq_len=%d" % (d_model, seq_len))
        check(inc.shape == (d_model, seq_len),
              "cached shape d_model=%d seq_len=%d" % (d_model, seq_len))
        d = float(np.max(np.abs(full - inc)))
        worst = max(worst, d)
        check(d <= 1e-12,
              "block equivalence d_model=%d seq_len=%d heads=%d chunk=%d -- "
              "max_abs_diff = %.17g" % (d_model, seq_len, heads, chunk, d))
    print("  worst max_abs_diff over %d cases: %.17g" % (len(cases), worst))


def section_tail_is_position_wise():
    """THE MEASUREMENT THAT REFUTES THE QUEUE'S RESIDUAL-STREAM CACHE.

    The queue asserted that a block-level decode needs a SECOND cache for
    the residual, because "the FFN's position-wise nature does not remove
    the requirement". This section measures whether the tail is position-
    wise in THIS repo, and measures the counterfactual so that a zero
    cannot be a tautology.
    """
    print("\n[SECTION] the block's tail is position-wise (no residual cache)")

    x, P = make_fixture(20261116, 4, 6, 2)
    x_tokens = x.T                                          # (tokens, d_model)
    attn_tokens = (np.random.default_rng(3).normal(size=x_tokens.shape))

    full_tail = ref_block_tail(x_tokens, attn_tokens, P, ln="rows")
    chunk_tail = ref_block_tail(x_tokens[4:], attn_tokens[4:], P, ln="rows")
    d = float(np.max(np.abs(full_tail[4:] - chunk_tail)))
    check(d <= 1e-15,
          "the tail computed on the last two tokens alone equals the full "
          "sequence's rows -- max_abs_diff = %.17g" % d)
    print("  rows-axis (the repo's convention): max_abs_diff = %.17g" % d)

    # THE COUNTERFACTUAL. A tail whose LayerNorm normalises over the TOKEN
    # axis does NOT have the property. Without this check, the zero above
    # would be a tautology rather than evidence.
    cf_full = ref_block_tail(x_tokens, attn_tokens, P, ln="tokens")
    cf_chunk = ref_block_tail(x_tokens[4:], attn_tokens[4:], P, ln="tokens")
    d_cf = float(np.max(np.abs(cf_full[4:] - cf_chunk)))
    check(d_cf > 1e-6,
          "the counterfactual (a token-axis LayerNorm) is NOT position-wise "
          "-- max_abs_diff = %.17g" % d_cf)
    print("  tokens-axis (the counterfactual): max_abs_diff = %.17g" % d_cf)

    # And the whole cached block with the counterfactual's tail must also
    # FAIL to match the full block -- i.e. the property is what makes the
    # equivalence hold, not something incidental.
    x2, P2 = make_fixture(20261116, 4, 6, 2)
    full2 = ref_block_full(x2, P2, 2)
    cf_cached = ref_block_cached(x2, P2, 2, 1, ln="tokens")
    d2 = float(np.max(np.abs(full2 - cf_cached)))
    check(d2 > 1e-6,
          "a token-axis tail breaks the cached/full equivalence too -- "
          "max_abs_diff = %.17g" % d2)
    print("  token-axis tail breaks the equivalence: max_abs_diff = %.17g" % d2)


def section_degenerate_layer_norm():
    """THE VARIANCE CLAMP IS LOAD-BEARING, AND ONLY A DEGENERATE ROW SHOWS IT.

    `var = max(var, 1e-7)` never fires on random data -- the variance is
    O(1) -- so dropping the clamp changes nothing on the ordinary fixtures
    and the mutation SURVIVED the first version of this oracle.

    **AND AN EXACTLY CONSTANT ROW DOES NOT SHOW IT EITHER**, which the
    first draft of this section got wrong. For a constant row the deviation
    `x[b][f] - mean` is exactly zero, so the NUMERATOR is zero and the
    output is `0 / anything == 0` -- the denominator is irrelevant. The
    clamp is observable only where the variance is TINY BUT NON-ZERO: the
    deviations must be small enough that `var < 1e-7`, yet not so small
    that they vanish.
    """
    print("\n[SECTION] the LayerNorm's variance clamp is load-bearing")
    tokens, features = 3, 5
    gamma = np.ones(features)
    beta = np.zeros(features)

    # A row whose deviations are ~1e-4: the variance is ~1e-8, BELOW the
    # 1e-7 clamp, but the deviations themselves are not zero.
    rng = np.random.default_rng(17)
    near_const = 1.0 + rng.normal(size=(tokens, features)) * 1e-4

    got = ref_layernorm_rows(near_const, gamma, beta)
    want = ref_layernorm_independent(near_const, gamma, beta)
    check_close(got, want, 1e-12,
                "the two LayerNorm implementations agree on a near-constant row")

    # Without the clamp the denominator is sqrt(var + eps); with it, it is
    # sqrt(1e-7 + eps). The two differ by a factor of about 1.35 here, so
    # the clamp is observable.
    mean = near_const.mean(axis=1, keepdims=True)
    var = ((near_const - mean) ** 2).mean(axis=1, keepdims=True)
    check(float(np.max(var)) < 1e-7,
          "the fixture's variance is BELOW the 1e-7 clamp (max var = %.17g)"
          % float(np.max(var)))
    without = gamma * (near_const - mean) / np.sqrt(var + 1e-7) + beta
    d = float(np.max(np.abs(got - without)))
    check(d > 1e-3,
          "dropping the variance clamp MUST change a near-constant row's "
          "output -- max_abs_diff = %.17g" % d)
    print("  clamped vs unclamped on a near-constant row: max_abs_diff = %.17g"
          % d)

    # The independent implementation must carry the clamp too, or the
    # agreement above would be vacuous.
    check(float(np.max(np.abs(want - without))) > 1e-3,
          "the independent implementation also carries the clamp")

    # AND the exactly-constant row is a COUNTERFACTUAL for this check: it
    # does NOT expose the clamp, because its numerator is zero. Without
    # this, a reader could reasonably think the constant row was the
    # fixture -- and it is not.
    const = np.ones((tokens, features))
    got_const = ref_layernorm_rows(const, gamma, beta)
    mean_c = const.mean(axis=1, keepdims=True)
    var_c = ((const - mean_c) ** 2).mean(axis=1, keepdims=True)
    without_c = gamma * (const - mean_c) / np.sqrt(var_c + 1e-7) + beta
    d_const = float(np.max(np.abs(got_const - without_c)))
    check(d_const == 0.0,
          "an EXACTLY constant row does NOT expose the clamp (its numerator "
          "is zero) -- max_abs_diff = %.17g" % d_const)
    print("  an exactly constant row does not expose it: max_abs_diff = %.17g"
          % d_const)


def section_non_vacuity():
    print("\n[SECTION] the equivalence check is NOT a tautology")
    x, P = make_fixture(20261116, 8, 6, 2)
    full = ref_block_full(x, P, 2)

    # A cache that is silently ignored must DIFFER from the full forward.
    broken = ref_block_cached(x, P, 2, 1, use_cache=False)
    d = float(np.max(np.abs(full - broken)))
    check(d > 1e-6,
          "a cache that is ignored must differ from the full forward "
          "(max_abs_diff = %.17g)" % d)
    print("  ignored-cache max_abs_diff = %.17g (must be > 1e-6)" % d)

    # Dropping the causal mask among the new tokens must also DIFFER.
    unmasked = ref_block_cached(x, P, 2, 2, mask_new_chunk=False)
    d2 = float(np.max(np.abs(full - unmasked)))
    check(d2 > 1e-6,
          "dropping the causal mask must differ from the full forward "
          "(max_abs_diff = %.17g)" % d2)
    print("  unmasked max_abs_diff = %.17g (must be > 1e-6)" % d2)

    # The past must MATTER: perturbing token 0 must change a later step.
    x2 = x.copy()
    x2[:, 0] += 3.0
    full2 = ref_block_full(x2, P, 2)
    later = float(np.max(np.abs(full[:, 1:] - full2[:, 1:])))
    check(later > 1e-6,
          "perturbing token 0 must change the later steps (max_abs_diff = %.17g)"
          % later)
    print("  token-0 perturbation effect on later steps = %.17g" % later)

    # Causality: perturbing the LAST token must leave the EARLIER steps
    # untouched (a token cannot see the future) and change only the last.
    x3 = x.copy()
    x3[:, -1] += 3.0
    full3 = ref_block_full(x3, P, 2)
    earlier = float(np.max(np.abs(full[:, :-1] - full3[:, :-1])))
    last = float(np.max(np.abs(full[:, -1] - full3[:, -1])))
    check(earlier == 0.0,
          "perturbing the LAST token must leave the earlier steps untouched "
          "(max_abs_diff = %.17g)" % earlier)
    check(last > 1e-6,
          "perturbing the LAST token must change the last step "
          "(max_abs_diff = %.17g)" % last)
    print("  last-token perturbation: earlier = %.17g, last = %.17g"
          % (earlier, last))

    # The block's tail must NOT be a constant: a block that returned a
    # constant would also "match" a full forward that returned a constant.
    check(float(np.std(full)) > 1e-6,
          "the block's output is not constant (std = %.17g)"
          % float(np.std(full)))
    check(float(np.max(np.abs(full))) > 1e-3,
          "the block's output is not all-zero (max_abs = %.17g)"
          % float(np.max(np.abs(full))))

    # THE FFN'S WEIGHTS MUST NOT BE SYMMETRIC, or the "X @ W vs X @ W.T"
    # distinction is invisible -- a symmetric fixture lets a transposed
    # implementation pass.
    _, P2 = make_fixture(20261116, 8, 6, 2)
    check(not np.allclose(P2.W2, P2.W2.T),
          "the fixture's W2 is not symmetric (so a transposed projection "
          "would differ)")
    check(not np.allclose(P2.W1, P2.W1.T),
          "the fixture's W1 is not symmetric (so a transposed projection "
          "would differ)")


def section_positional_encoding():
    print("\n[SECTION] the positional encoding is ABSOLUTE, not local")
    max_len, d_model = 8, 4
    pe = ref_pe_table(max_len, d_model)
    rng = np.random.default_rng(5)
    # THE RNG IS A STREAM: two draws are two DIFFERENT fixtures. The first
    # draft of this section drew `x` and then drew a separate `whole`, and
    # compared the encoding of one against the encoding of the other --
    # they differed because the INPUTS differed, and the failure pointed at
    # the reference when the fault was in the fixture. `x` is now DEFINED as
    # the tail of `whole`, so the two are the same numbers by construction.
    whole = rng.normal(size=(d_model, max_len))
    x = whole[:, 5:].copy()          # 3 columns at absolute positions 5,6,7

    local = ref_pe_forward(x, pe)
    absolute = ref_pe_forward_at(x, pe, 3)
    d = float(np.max(np.abs(local - absolute)))
    check(d > 1e-6,
          "the LOCAL and the ABSOLUTE positional encodings must differ "
          "(max_abs_diff = %.17g)" % d)
    print("  local vs absolute at offset 3: max_abs_diff = %.17g" % d)

    # The absolute form at offset 0 IS the local form.
    check_close(ref_pe_forward_at(x, pe, 0), local, 0.0,
                "the absolute encoding at offset 0 equals the local one")

    # And the absolute form must equal the local form applied to the WHOLE
    # sequence and then sliced -- i.e. it is a genuine absolute position,
    # not a shifted approximation.
    whole_abs = ref_pe_forward_at(whole, pe, 0)
    tail = whole_abs[:, 5:]
    check_close(ref_pe_forward_at(x, pe, 5), tail, 0.0,
                "the absolute encoding of a tail equals the tail of the "
                "whole sequence's encoding")


def section_pinned_fixture():
    print("\n[SECTION] the pinned fixture")
    seed = 20261116
    d_model, seq_len, heads = 8, 6, 2
    x, P = make_fixture(seed, d_model, seq_len, heads)

    full = ref_block_full(x, P, heads)
    inc = ref_block_cached(x, P, heads, 1)
    check_close(full, inc, 1e-12,
                "the pinned fixture's cached output matches its full output")

    check(float(np.max(np.abs(full))) > 1e-3,
          "the pinned output is not all-zero (max_abs = %.17g)"
          % float(np.max(np.abs(full))))
    check(float(np.std(full)) > 1e-6,
          "the pinned output is not constant (std = %.17g)" % float(np.std(full)))

    steps = [float(np.max(np.abs(full[:, t] - full[:, 0]))) for t in range(seq_len)]
    check(min(steps[1:]) > 1e-6,
          "the decode produces different outputs at different steps "
          "(min step-to-step difference = %.17g)" % min(steps[1:]))
    print("  per-step max_abs_diff vs step 0: %s" %
          ", ".join("%.6g" % s for s in steps))

    print("\n  pinned fixture (seed %d, d_model %d, seq_len %d, heads %d):"
          % (seed, d_model, seq_len, heads))
    print("    max_abs(full)            = %.17g" % float(np.max(np.abs(full))))
    print("    max_abs(full - cached)   = %.17g"
          % float(np.max(np.abs(full - inc))))


def emit_cpp():
    """Emit the pinned fixture as C++ literals.

    The repo has been burned by hand-transcribed numbers before (see
    tests/_dump_fixture.py): both numbers were CORRECT for the fixture they
    were given, and the fixture was simply not the oracle's. Pinned numbers
    transfer only across an IDENTICAL fixture, so emit the real one.
    """
    seed = 20261116
    d_model, seq_len, heads = 8, 6, 2
    x, P = make_fixture(seed, d_model, seq_len, heads)
    full = ref_block_full(x, P, heads)
    inc = ref_block_cached(x, P, heads, 1)

    def emit(name, arr):
        flat = np.asarray(arr, dtype=float).reshape(-1)
        body = ", ".join(repr(float(v)) for v in flat)
        print("const std::vector<double> %s = {%s};" % (name, body))

    print("// ==== generated by tests/block_cached_decode_oracle.py --emit-cpp; "
          "do not hand-edit ====")
    print("// seed %d, d_model %d, seq_len %d, heads %d"
          % (seed, d_model, seq_len, heads))
    print("const int BC_D_MODEL = %d;" % d_model)
    print("const int BC_SEQ_LEN = %d;" % seq_len)
    print("const int BC_HEADS = %d;" % heads)
    emit("BC_X_FIX", x)
    emit("BC_WQ_FIX", P.Wq)
    emit("BC_WK_FIX", P.Wk)
    emit("BC_WV_FIX", P.Wv)
    emit("BC_WO_FIX", P.Wo)
    emit("BC_W1_FIX", P.W1)
    emit("BC_B1_FIX", P.b1)
    emit("BC_W2_FIX", P.W2)
    emit("BC_B2_FIX", P.b2)
    emit("BC_LN1_G_FIX", P.ln1_g)
    emit("BC_LN1_B_FIX", P.ln1_b)
    emit("BC_LN2_G_FIX", P.ln2_g)
    emit("BC_LN2_B_FIX", P.ln2_b)
    emit("BC_OUT_FULL_FIX", full)
    emit("BC_OUT_CACHED_FIX", inc)
    print("// max_abs(full - cached) in float64 = %.17g"
          % float(np.max(np.abs(full - inc))))
    print("// The C++ test asserts: the cached decode == the C++ full forward")
    print("// to the LAST BIT (0.0), and both match the pinned numpy output to")
    print("// 1e-12 (float64 BLAS vs the C++ explicit loops).")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("=" * 72)
    print("tests/block_cached_decode_oracle.py -- the float64 numpy specification")
    print("=" * 72)
    print("numpy %s | python %s" % (np.__version__, sys.version.split()[0]))

    if "--emit-cpp" in sys.argv:
        emit_cpp()
        return 0

    section_independent_agreement()
    section_equivalence()
    section_tail_is_position_wise()
    section_degenerate_layer_norm()
    section_non_vacuity()
    section_positional_encoding()
    section_pinned_fixture()

    print("")
    print("=== SUMMARY: %d passed, %d failed ===" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
