#!/usr/bin/env python3
"""
tests/stack_cached_decode_mutate_oracle.py -- the mutation harness for
tests/stack_cached_decode_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

    /usr/bin/python3 tests/stack_cached_decode_mutate_oracle.py

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** The stack oracle's first execution printed "15 passed, 0
failed" -- and that established nothing, because it was the first
execution. The question that matters is not "does the oracle agree with
itself" but "what would this oracle NOT notice?". This harness answers
that with a measured mutation list rather than with reasoning.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  SM1  the cached stack IGNORES the cached keys (the past is masked out)
       -> the cache is actually read.
  SM2  the causal mask among the NEW chunk is dropped
       -> the causality of the cached path. **This one needs the CHUNKED
          decode**: with one token per step the mask never fires, so a
          token-at-a-time decode cannot see it at all. That was measured
          while writing the oracle, not assumed.
  SM3  the cached mask's absolute position is wrong (`past` instead of
       `past + i`)
       -> the mask's position arithmetic.
  SM4  the cached path's scale loses the + 1e-9
       -> the attention scale convention.
  SM5  the cached path's softmax normalises over the WRONG AXIS
       -> the softmax.
  SM6  the cached path's per-head isolation is broken (every head reads
       head 0)
       -> the head split.
  SM7  the cached path skips the output projection
       -> the W_o projection.
  SM8  the cached VALUES are the cached KEYS
       -> the key/value separation.
  SM9  the new keys are appended in REVERSE order
       -> the append order.
  SM10 the K/V store's LENGTH is not tracked (the mask's `past` is always 0)
       -> the cache bookkeeping.
  SM11 EVERY BLOCK IS ALIASED TO SLOT 0
       -> THE CACHE-SLOT ALIASING CLAIM. This is the mutation the whole
          feature exists to prevent, and it is the one a naive
          implementation is most likely to ship.
  SM12 the stack's blocks are chained in the WRONG ORDER (reversed)
       -> the stack's ordering.
  SM13 the tail's LayerNorm normalises over the TOKEN axis
       -> the per-token LayerNorm convention the whole equivalence rests on.
  SM14 the tail's residual add is DROPPED
       -> the tail's structure.
  SM15 the tail's second LayerNorm is dropped
       -> the tail's structure, from the other side.
  SM16 the FFN's weight order is transposed (X @ W.T instead of X @ W)
       -> THE PROJECTION CONVENTION.
  SM17 the GELU loses its 0.5 factor
       -> the activation.
  SM18 the LayerNorm's variance clamp (max(var, 1e-7)) is dropped
       -> the numerical-stability convention. **It needs the NEAR-CONSTANT
          fixture**: with random data the clamp never fires, so this
          mutation would survive.
  SM19 the non-vacuity check's "the aliased foil must differ" is inverted
       -> THE NON-VACUITY CLAIM ITSELF.
  SM20 the reference's own full-path causal mask is dropped
       -> the reference itself.

THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
=================================================

  * AN AMBIGUOUS ANCHOR IS NOT A VERDICT. An anchor that occurs twice makes
    the edit a coin flip, and a coin flip is not a verdict. The harness
    REFUSES to edit and reports the mutation as AMBIGUOUS -- a harness
    defect, not a survivor.
  * A MUTATION THAT CANNOT CHANGE ANYTHING IS NOT A SURVIVOR. Each
    mutation is first CHECKED for ambiguity, and the harness reports
    MUT-NOOP when the edit did not apply at all.
  * THE HARNESS'S OWN EXIT CODE IS ASSERTED. A harness that cannot fail is
    not a harness.
"""

import os
import subprocess
import sys
import tempfile

ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "stack_cached_decode_oracle.py")

# (id, description, anchor, replacement)
MUTATIONS = [
    ("SM1", "the cached stack ignores the cached keys (the past is masked out)",
     "        if use_mask:\n            for i in range(x_new.shape[0]):\n"
     "                for j in range(total):\n                    if j > past + i:\n"
     "                        s[i, j] += -1e9\n",
     "        if False:\n            for i in range(x_new.shape[0]):\n"
     "                for j in range(total):\n                    if j > past + i:\n"
     "                        s[i, j] += -1e9\n"),

    ("SM2", "the causal mask among the NEW chunk is dropped",
     "                    if j > past + i:\n                        s[i, j] += -1e9\n",
     "                    if j > past + i + 1:\n                        s[i, j] += -1e9\n"),

    ("SM3", "the cached mask's absolute position is wrong",
     "                    if j > past + i:\n",
     "                    if j > past:\n"),

    ("SM4", "the cached path's scale loses the + 1e-9",
     "        s = Q[:, sl] @ k_store[:, sl].T / math.sqrt(dk + 1e-9)\n",
     "        s = Q[:, sl] @ k_store[:, sl].T / math.sqrt(dk)\n"),

    ("SM5", "the cached softmax normalises over the wrong axis",
     "        e = np.exp(s - s.max(axis=1, keepdims=True))\n"
     "        pr = e / e.sum(axis=1, keepdims=True)\n"
     "        acc[:, sl] += pr @ v_store[:, sl]\n"
     "    return acc @ p[\"Wo\"], k_store, v_store\n",
     "        e = np.exp(s - s.max(axis=0, keepdims=True))\n"
     "        pr = e / e.sum(axis=0, keepdims=True)\n"
     "        acc[:, sl] += pr @ v_store[:, sl]\n"
     "    return acc @ p[\"Wo\"], k_store, v_store\n"),

    ("SM6", "the cached path's per-head isolation is broken",
     "        s = Q[:, sl] @ k_store[:, sl].T / math.sqrt(dk + 1e-9)\n"
     "        if use_mask:\n",
     "        s = Q[:, :dk] @ k_store[:, :dk].T / math.sqrt(dk + 1e-9)\n"
     "        if use_mask:\n"),

    ("SM7", "the cached path skips the output projection",
     "    return acc @ p[\"Wo\"], k_store, v_store\n",
     "    return acc, k_store, v_store\n"),

    ("SM8", "the cached VALUES are the cached KEYS",
     "    k_store = np.vstack([k_store, K])\n    v_store = np.vstack([v_store, V])\n",
     "    k_store = np.vstack([k_store, K])\n    v_store = np.vstack([v_store, K])\n"),

    ("SM9", "the new keys are appended in reverse order",
     "    k_store = np.vstack([k_store, K])\n    v_store = np.vstack([v_store, V])\n",
     "    k_store = np.vstack([k_store, K[::-1]])\n    v_store = np.vstack([v_store, V[::-1]])\n"),

    ("SM10", "the K/V store's length is not tracked",
     "    past = k_store.shape[0]\n",
     "    past = 0\n"),

    ("SM11", "EVERY BLOCK IS ALIASED TO SLOT 0",
     "    if slot_of_block is None:\n        slot_of_block = lambda b: b\n",
     "    if slot_of_block is None:\n        slot_of_block = lambda b: 0\n"),

    ("SM12", "the stack's blocks are chained in the wrong order",
     "    for b, p in enumerate(params):\n            sl = slot_of_block(b)\n",
     "    for b, p in enumerate(reversed(params)):\n            sl = slot_of_block(b)\n"),

    ("SM13", "the tail's LayerNorm normalises over the TOKEN axis",
     "    mean = x.mean(axis=1, keepdims=True)\n"
     "    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)\n",
     "    mean = x.mean(axis=0, keepdims=True)\n"
     "    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)\n"),

    ("SM14", "the tail's residual add is dropped",
     "    residual = x + attn_out\n",
     "    residual = attn_out\n"),

    ("SM15", "the tail's second LayerNorm is dropped",
     "    return layer_norm(norm1 + ffn2, p[\"ln2g\"], p[\"ln2b\"])\n",
     "    return norm1 + ffn2\n"),

    ("SM16", "the FFN's weight order is transposed",
     "    ffn1 = norm1 @ p[\"W1\"] + p[\"b1\"]\n",
     "    ffn1 = norm1 @ p[\"W1\"].T + p[\"b1\"]\n"),

    ("SM17", "the GELU loses its 0.5 factor",
     "    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))\n",
     "    return x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))\n"),

    ("SM18", "the LayerNorm's variance clamp is dropped",
     "    var = np.maximum(var, 1e-7)\n",
     "    var = var\n"),

    ("SM19", "the non-vacuity check's 'the aliased foil must differ' is inverted",
     "    check(aliased_diff > 1e-3,\n",
     "    check(aliased_diff < 1e-3,\n"),

    ("SM20", "the reference's own full-path causal mask is dropped",
     "        s = s + np.triu(np.ones_like(s), k=1) * -1e9\n",
     "        s = s + np.triu(np.ones_like(s), k=1) * 0.0\n"),
]


def run_oracle(source, tmpdir):
    path = os.path.join(tmpdir, "mutated_oracle.py")
    with open(path, "w") as f:
        f.write(source)
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
    return proc


def main():
    src = open(ORACLE).read()

    # The pristine oracle must PASS, or every verdict below is meaningless.
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = run_oracle(src, tmpdir)
        if proc.returncode != 0:
            print("THE PRISTINE ORACLE FAILS -- every verdict below is void:")
            print(proc.stdout)
            return 1
    print("pristine oracle: PASS")

    caught = 0
    survived = []
    broken = []
    ambiguous = []

    for mid, desc, anchor, repl in MUTATIONS:
        n = src.count(anchor)
        if n != 1:
            ambiguous.append((mid, desc, n))
            print("  [AMBIGUOUS] %s: the anchor occurs %d times -- harness "
                  "defect, not a verdict (%s)" % (mid, n, desc))
            continue
        mutated = src.replace(anchor, repl, 1)
        if mutated == src:
            broken.append((mid, desc))
            print("  [MUT-NOOP] %s: the edit did not change the source (%s)"
                  % (mid, desc))
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = run_oracle(mutated, tmpdir)
        if proc.returncode != 0:
            caught += 1
            print("  [CAUGHT] %s: %s" % (mid, desc))
        else:
            survived.append((mid, desc))
            print("  [SURVIVED] %s: %s" % (mid, desc))

    print("=== %d caught, %d survived, %d ambiguous, %d no-op ==="
          % (caught, len(survived), len(ambiguous), len(broken)))
    for mid, desc in survived:
        print("  survivor: %s -- %s" % (mid, desc))
    return 0 if (not survived and not broken and not ambiguous) else 1


if __name__ == "__main__":
    sys.exit(main())
