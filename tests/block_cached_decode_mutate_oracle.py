#!/usr/bin/env python3
"""
tests/block_cached_decode_mutate_oracle.py -- the mutation harness for
tests/block_cached_decode_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** This repo has been bitten by that several times: the
off-policy-evaluation oracle's first draft CLAIMED it had caught two defects
in itself and had caught NEITHER, and the rank-correlation and the
quantile-regression oracles were green on their first execution. The claim
is deleted rather than kept as flavour, and the oracle's discriminating
power is MEASURED here, not narrated.

**THE FIRST RUN OF THIS HARNESS REPORTED SIX SURVIVORS AND SEVEN AMBIGUOUS
ANCHORS**, and BOTH halves were informative:

  * The six survivors were every mutation of the SHARED `ref_block_tail`.
    The reason is structural and it is the trap this repo has hit before:
    **a mutation in a helper used by BOTH the reference and the thing under
    test changes the two sides IDENTICALLY, so a comparison between them
    CANNOT SEE IT.** The fix was to give the oracle a SECOND, INDEPENDENT
    implementation (`ref_block_full_independent`, explicit scalar loops, no
    BLAS) and a DEGENERATE fixture for the LayerNorm's variance clamp --
    the clamp never fires on random data, so without a constant row it is
    unobservable and its mutation survived too.
  * The seven ambiguous anchors were the harness's own defect: an anchor
    that occurs twice makes the edit a coin flip, and a coin flip is not a
    verdict. Every one was re-anchored on LONGER CONTEXT.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the cached block IGNORES the cached keys (the past is masked out)
       -> the cache is actually read.
  OM2  the causal mask among the NEW chunk is dropped
       -> the causality of the cached path.
  OM3  the FULL path's causal mask is dropped
       -> the reference itself.
  OM4  the FULL path's mask direction is flipped
       -> the reference itself, from the other side.
  OM5  the cached path's scale loses the + 1e-9
       -> the attention scale convention.
  OM6  the cached path's softmax normalises over the WRONG AXIS
       -> the softmax.
  OM7  the cached path's per-head isolation is broken (every head reads head 0)
       -> the head split.
  OM8  the cached path skips the output projection
       -> the W_o projection.
  OM9  the K/V store's LENGTH is not tracked (the mask's `past` is always 0)
       -> the cache bookkeeping.
  OM10 the new keys are appended in REVERSE order
       -> the append order.
  OM11 the cached VALUES are the cached KEYS
       -> the key/value separation.
  OM12 the tail's LayerNorm normalises over the TOKEN axis
       -> THE REFUTATION MEASUREMENT. This is the mutation that makes the
          "the tail is position-wise" check meaningful: with the repo's
          feature-axis convention replaced by the token-axis one, the tail
          is no longer position-wise and the cached block stops matching
          the full block. It is the heart of this oracle's central claim.
  OM13 the tail's second LayerNorm is dropped (the output is the FFN's)
       -> the tail's structure. **A SHARED-HELPER MUTATION**: it is caught
          only by the second, independent implementation.
  OM14 the tail's residual add is DROPPED
       -> the tail's structure, from the other side.
  OM15 the FFN's second linear is applied to the PRE-GELU activation
       -> the FFN's ordering.
  OM16 the FFN's weight order is transposed (X @ W.T instead of X @ W)
       -> THE PROJECTION CONVENTION. The repo stores W as (d_model,
          d_model) with the INPUT feature on the row axis, so X @ W is
          right and X @ W.T is wrong.
  OM17 the GELU loses its 0.5 factor
       -> the activation.
  OM18 the positional encoding's offset is IGNORED (it uses the local index)
       -> THE ABSOLUTE-POSITION CLAIM.
  OM19 the positional encoding's offset is applied with the WRONG sign
       -> the absolute-position claim, from the other side.
  OM20 the LayerNorm's variance clamp (max(var, 1e-7)) is dropped
       -> the numerical-stability convention. **It needs a DEGENERATE
          fixture**: with random data the clamp never fires, so this
          mutation survived until the constant-row fixture was added.
  OM21 the non-vacuity check's "the ignored cache must differ" is inverted
       -> THE NON-VACUITY CLAIM ITSELF.
  OM22 the independent implementation's LayerNorm loses the clamp
       -> the independent witness itself.
  OM23 the independent implementation transposes the FFN's weights
       -> the independent witness, from the other side.

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
================================================

  1. The anchor must occur EXACTLY ONCE. A twice-occurring anchor makes
     the edit a coin flip, and a coin flip is not a verdict. The harness
     asserts the count and reports MUT-ANCHOR otherwise.
  2. The mutated source is compared to the pristine source -- a mutation
     that did not apply is a HARNESS DEFECT, not a verdict.
  3. A non-zero exit with ZERO reported failures is a CRASH, not a
     verdict -> MUT-BROKEN.
  4. A SURVIVOR is a hole in the SPECIFICATION, never a pass.

RUN
===

    /usr/bin/python3 tests/block_cached_decode_mutate_oracle.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "block_cached_decode_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# id | description | search | replacement
MUTATIONS = [
    ("OM1",
     "the cached block IGNORES the cached keys",
     "                admissible = idx_all[None, :] <= (past + idx_new[:, None])\n"
     "                scores = np.where(admissible, scores, scores - 1e9)\n"
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)",
     "                admissible = idx_all[None, :] <= (idx_new[:, None])\n"
     "                scores = np.where(admissible, scores, scores - 1e9)\n"
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)"),

    ("OM2",
     "the causal mask among the NEW chunk is dropped",
     "                admissible = idx_all[None, :] <= (past + idx_new[:, None])\n"
     "                scores = np.where(admissible, scores, scores - 1e9)\n"
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)",
     "                admissible = idx_all[None, :] <= (past + idx_new[:, None])\n"
     "                scores = scores\n"
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)"),

    ("OM3",
     "the FULL path's causal mask is dropped",
     "        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)",
     "        scores = scores"),

    ("OM4",
     "the FULL path's mask direction is flipped",
     "        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)",
     "        scores = np.where(idx[None, :] < idx[:, None], scores - 1e9, scores)"),

    ("OM5",
     "the cached path's scale loses the + 1e-9",
     "    d_model, seq_len = x.shape\n"
     "    d_k = d_model // heads\n"
     "    scale = math.sqrt(d_k + 1e-9)\n"
     "\n"
     "    Kc = np.zeros((0, d_model), dtype=float)\n"
     "    Vc = np.zeros((0, d_model), dtype=float)\n"
     "    outs = []",
     "    d_model, seq_len = x.shape\n"
     "    d_k = d_model // heads\n"
     "    scale = math.sqrt(d_k)\n"
     "\n"
     "    Kc = np.zeros((0, d_model), dtype=float)\n"
     "    Vc = np.zeros((0, d_model), dtype=float)\n"
     "    outs = []"),

    ("OM6",
     "the cached path's softmax normalises over the WRONG AXIS",
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)",
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=0, keepdims=True)\n"
     "            acc[:, sl] = p @ Vc[:, sl]\n"
     "        attn_tokens = (acc @ P.Wo)"),

    ("OM7",
     "the cached path's per-head isolation is broken (every head reads head 0)",
     "        for h in range(heads):\n"
     "            sl = slice(h * d_k, (h + 1) * d_k)\n"
     "            scores = (Q[:, sl] @ Kc[:, sl].T) / scale\n"
     "            if mask_new_chunk:",
     "        for h in range(heads):\n"
     "            sl = slice(0, d_k)\n"
     "            scores = (Q[:, sl] @ Kc[:, sl].T) / scale\n"
     "            if mask_new_chunk:"),

    ("OM8",
     "the cached path skips the output projection",
     "        attn_tokens = (acc @ P.Wo)                    # (n, d_model)",
     "        attn_tokens = acc                             # (n, d_model)"),

    ("OM9",
     "the K/V store's LENGTH is not tracked",
     "        past = Kc.shape[0]\n        if use_cache:",
     "        past = 0\n        if use_cache:"),

    ("OM10",
     "the new keys are appended in REVERSE order",
     "            Kc = np.vstack([Kc, K])",
     "            Kc = np.vstack([Kc, K[::-1]])"),

    ("OM11",
     "the cached VALUES are the cached KEYS",
     "            Vc = np.vstack([Vc, V])",
     "            Vc = np.vstack([Vc, K])"),

    ("OM12",
     "the tail's LayerNorm normalises over the TOKEN axis",
     "    norm = ref_layernorm_tokens if ln == \"tokens\" else ref_layernorm_rows",
     "    norm = ref_layernorm_tokens"),

    ("OM13",
     "the tail's second LayerNorm is dropped (the output is the FFN's)",
     "    ffn_res = norm1 + ffn2\n"
     "    return norm(ffn_res, P.ln2_g, P.ln2_b)",
     "    ffn_res = norm1 + ffn2\n"
     "    return ffn_res"),

    ("OM14",
     "the tail's residual add is DROPPED",
     "    residual = x_tokens + attn_tokens\n"
     "    norm1 = norm(residual, P.ln1_g, P.ln1_b)",
     "    residual = attn_tokens\n"
     "    norm1 = norm(residual, P.ln1_g, P.ln1_b)"),

    ("OM15",
     "the FFN's second linear is applied to the PRE-GELU activation",
     "    ffn1 = norm1 @ P.W1 + P.b1\n"
     "    ffn1 = ref_gelu(ffn1)\n"
     "    ffn2 = ffn1 @ P.W2 + P.b2",
     "    ffn1 = norm1 @ P.W1 + P.b1\n"
     "    ffn2 = ffn1 @ P.W2 + P.b2\n"
     "    ffn1 = ref_gelu(ffn1)"),

    ("OM16",
     "the FFN's weight order is transposed (X @ W.T instead of X @ W)",
     "    ffn2 = ffn1 @ P.W2 + P.b2",
     "    ffn2 = ffn1 @ P.W2.T + P.b2"),

    ("OM17",
     "the GELU loses its 0.5 factor",
     "    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *\n"
     "                                    (x + 0.044715 * x ** 3)))",
     "    return x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *\n"
     "                             (x + 0.044715 * x ** 3)))"),

    ("OM18",
     "the positional encoding's offset is IGNORED (it uses the local index)",
     "        out[:, s] = x[:, s] + pe[offset + s]",
     "        out[:, s] = x[:, s] + pe[s]"),

    ("OM19",
     "the positional encoding's offset is applied with the WRONG sign",
     "        out[:, s] = x[:, s] + pe[offset + s]",
     "        out[:, s] = x[:, s] + pe[max(offset - s, 0)]"),

    ("OM20",
     "the LayerNorm's variance clamp (max(var, 1e-7)) is dropped",
     "    var = np.maximum(var, 1e-7)\n"
     "    inv = 1.0 / np.sqrt(var + eps)\n"
     "    return gamma * (x - mean) * inv + beta\n"
     "\n"
     "\n"
     "def ref_layernorm_tokens",
     "    inv = 1.0 / np.sqrt(var + eps)\n"
     "    return gamma * (x - mean) * inv + beta\n"
     "\n"
     "\n"
     "def ref_layernorm_tokens"),

    ("OM21",
     "the non-vacuity check's \"the ignored cache must differ\" is inverted",
     "    check(d > 1e-6,\n"
     "          \"a cache that is ignored must differ from the full forward \"\n"
     "          \"(max_abs_diff = %.17g)\" % d)",
     "    check(d <= 1e-6,\n"
     "          \"a cache that is ignored must differ from the full forward \"\n"
     "          \"(max_abs_diff = %.17g)\" % d)"),

    ("OM22",
     "the INDEPENDENT implementation's LayerNorm loses the variance clamp",
     "        var /= features\n"
     "        var = max(var, 1e-7)\n"
     "        sqrt_var = math.sqrt(var + eps)",
     "        var /= features\n"
     "        sqrt_var = math.sqrt(var + eps)"),

    ("OM23",
     "the INDEPENDENT implementation transposes the FFN's weights",
     "            for k in range(d_model):\n"
     "                val += P.W2[k][j] * ffn1[i][k]",
     "            for k in range(d_model):\n"
     "                val += P.W2[j][k] * ffn1[i][k]"),
]


def run_oracle(path):
    """Run the (possibly mutated) oracle. Returns (rc, failures_reported, stdout)."""
    proc = subprocess.run([PYTHON, path], capture_output=True, text=True)
    failures = proc.stdout.count("[FAIL]")
    return proc.returncode, failures, proc.stdout


def main():
    if not os.path.exists(ORACLE):
        print("MUT-HARNESS-BROKEN: %s does not exist" % ORACLE)
        return 2

    with open(ORACLE, "r") as f:
        pristine = f.read()

    workdir = tempfile.mkdtemp(prefix="block_cached_decode_oracle_mut_")
    target = os.path.join(workdir, "block_cached_decode_oracle.py")
    shutil.copy(ORACLE, target)

    # The oracle imports nothing from the repo, so a copy in a scratch
    # directory runs standalone.
    baseline_rc, baseline_failures, _ = run_oracle(target)
    if baseline_rc != 0 or baseline_failures != 0:
        print("MUT-HARNESS-BROKEN: the pristine oracle is not green "
              "(rc=%d, %d failures)" % (baseline_rc, baseline_failures))
        shutil.rmtree(workdir, ignore_errors=True)
        return 2

    global CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD

    try:
        for mid, desc, search, replacement in MUTATIONS:
            # --- restore FIRST: every mutation starts from the pristine file.
            with open(target, "w") as f:
                f.write(pristine)

            count = pristine.count(search)
            if count != 1:
                print("%s MUT-ANCHOR  the anchor occurs %d times -- harness "
                      "defect, not a verdict" % (mid, count))
                ANCHOR_BAD += 1
                continue

            with open(target, "w") as f:
                f.write(pristine.replace(search, replacement))

            # --- checksum: a mutation that did not apply is a harness defect.
            with open(target, "r") as f:
                mutated = f.read()
            if mutated == pristine:
                print("%s MUT-NOOP  the source is unchanged -- harness defect"
                      % mid)
                NOOP += 1
                continue

            rc, failures, out = run_oracle(target)

            if rc == 0 and failures == 0:
                # The oracle PASSED with the mutation in place. That is a
                # hole in the ORACLE, and it is reported as such -- never as
                # a pass.
                print("%s SURVIVED  %s" % (mid, desc))
                SURVIVED += 1
            elif rc != 0 and failures == 0:
                # A non-zero exit with zero reported failures is a CRASH.
                print("%s MUT-BROKEN  the oracle crashed without reporting a "
                      "failure (rc=%d)" % (mid, rc))
                BROKEN += 1
            else:
                first = ""
                for line in out.split("\n"):
                    if "[FAIL]" in line:
                        first = line.strip()
                        break
                print("%s CAUGHT  %s" % (mid, desc))
                print("      %s" % first)
                CAUGHT += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("")
    print("=== SUMMARY: %d caught, %d survived, %d broken, %d noop, "
          "%d anchor ===" % (CAUGHT, SURVIVED, BROKEN, NOOP, ANCHOR_BAD))
    if SURVIVED > 0:
        print("SURVIVORS ARE HOLES IN THE ORACLE (the SPECIFICATION):")
        return 1
    if BROKEN > 0 or NOOP > 0 or ANCHOR_BAD > 0:
        print("HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
