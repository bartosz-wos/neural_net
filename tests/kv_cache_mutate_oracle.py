#!/usr/bin/env python3
"""
tests/kv_cache_mutate_oracle.py -- the mutation harness for
tests/kv_cache_oracle.py.

PURPOSE: prove that the oracle actually CATCHES the defects it was written
for. A green oracle proves nothing on its own; the mutation has to fail.

WHY THIS HARNESS EXISTS AT ALL
==============================

**A GREEN ORACLE ON ITS FIRST RUN PROVES NOTHING ABOUT ITS OWN
SENSITIVITY.** This repo has been bitten by that three times: the
off-policy-evaluation oracle's first draft CLAIMED it had caught two
defects in itself and had caught NEITHER, and both the rank-correlation and
the quantile-regression oracles were green on their first execution. The
claim is deleted rather than kept as flavour, and the oracle's
discriminating power is MEASURED here, not narrated.

THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
==================================================

  OM1  the incremental path IGNORES the cached keys (the past is masked out)
       -> the cache is actually read.
  OM2  the causal mask among the NEW chunk is dropped
       -> the causality of the cached path.
  OM3  the FULL path's causal mask is dropped
       -> the reference itself.
  OM4  the FULL path's mask direction is flipped
       -> the reference itself, from the other side.
  OM5  the incremental path's scale loses the + 1e-9
       -> the attention scale convention.
  OM6  the incremental path's softmax normalises over the WRONG AXIS
       -> the softmax.
  OM7  the incremental path's per-head isolation is broken (heads are summed)
       -> the head split.
  OM8  the incremental path skips the output projection
       -> the W_o projection.
  OM9  the cache's LENGTH is not tracked (the mask's `past` is always 0)
       -> the cache bookkeeping.
  OM10 the new keys are appended in REVERSE order
       -> the append order.
  OM11 the cached VALUES are the cached KEYS
       -> the key/value separation.
  OM12 the FULL path skips the output projection
       -> the reference itself.
  OM13 the cache's length grows by 1 regardless of the rows appended
       -> the bookkeeping arithmetic.
  OM14 the appended keys land in the VALUE buffer
       -> the key/value separation, in the cache.
  OM15 the second append OVERWRITES the first
       -> the append offset.
  OM16 the cache's WIDTH guard is dropped
       -> the dimension guard. **This one is the reason the oracle's
          error checks assert the MESSAGE and not merely "it threw": with
          the guard removed, numpy's own broadcast error still throws, so a
          throw-only check would stay GREEN.**
  OM17 the cache's CAPACITY guard is dropped
       -> the capacity guard, for the same reason.
  OM18 the cache's ROW-COUNT guard is dropped
       -> the row-count guard.
  OM19 the cache's ZERO-ROWS guard is dropped
       -> the zero-row guard.
  OM20 reset does not clear the layer
       -> the reset.
  OM21 total_tokens counts the LAYERS, not the tokens
       -> the accounting.
  OM22 memory_bytes counts ONE buffer per layer
       -> the accounting, from the other side.

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

    /usr/bin/python3 tests/kv_cache_mutate_oracle.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "kv_cache_oracle.py")
PYTHON = "/usr/bin/python3"

CAUGHT = 0
SURVIVED = 0
BROKEN = 0
NOOP = 0
ANCHOR_BAD = 0

# id | description | search | replacement
MUTATIONS = [
    ("OM1",
     "the incremental path IGNORES the cached keys",
     "                admissible = idx_all[None, :] <= (past + idx_new[:, None])",
     "                admissible = idx_all[None, :] <= (idx_new[:, None])"),

    ("OM2",
     "the causal mask among the NEW chunk is dropped",
     "def ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, chunk,\n"
     "                        use_cache=True, mask_new_chunk=True):",
     "def ref_mha_incremental(x, Wq, Wk, Wv, Wo, heads, chunk,\n"
     "                        use_cache=True, mask_new_chunk=False):"),

    ("OM3",
     "the FULL path's causal mask is dropped",
     "        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)",
     "        scores = scores"),

    ("OM4",
     "the FULL path's mask direction is flipped",
     "        scores = np.where(idx[None, :] > idx[:, None], scores - 1e9, scores)",
     "        scores = np.where(idx[None, :] < idx[:, None], scores - 1e9, scores)"),

    ("OM5",
     "the incremental path's scale loses the + 1e-9",
     "    d_model, seq_len = x.shape\n"
     "    d_k = d_model // heads\n"
     "    scale = math.sqrt(d_k + 1e-9)\n"
     "\n"
     "    Kc = np.zeros((0, d_model), dtype=float)",
     "    d_model, seq_len = x.shape\n"
     "    d_k = d_model // heads\n"
     "    scale = math.sqrt(d_k)\n"
     "\n"
     "    Kc = np.zeros((0, d_model), dtype=float)"),

    ("OM6",
     "the incremental path's softmax normalises over the WRONG AXIS",
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=1, keepdims=True)\n"
     "            acc[:, sl] = p @ Vh",
     "            m = scores.max(axis=1, keepdims=True)\n"
     "            e = np.exp(scores - m)\n"
     "            p = e / e.sum(axis=0, keepdims=True)\n"
     "            acc[:, sl] = p @ Vh"),

    ("OM7",
     "the incremental path's per-head isolation is broken (every head reads head 0)",
     "        for h in range(heads):\n"
     "            sl = slice(h * d_k, (h + 1) * d_k)",
     "        for h in range(heads):\n"
     "            sl = slice(0, d_k)"),

    ("OM8",
     "the incremental path skips the output projection",
     "        out = acc @ Wo.T\n        blocks.append(out.T)",
     "        out = acc\n        blocks.append(out.T)"),

    ("OM9",
     "the cache's LENGTH is not tracked",
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
     "the FULL path skips the output projection",
     "    out = acc @ Wo.T\n    return out.T",
     "    out = acc\n    return out.T"),

    ("OM13",
     "the cache's length grows by 1 regardless of the rows appended",
     "        self.layers[layer][2] = n + k.shape[0]",
     "        self.layers[layer][2] = n + 1"),

    ("OM14",
     "the appended keys land in the VALUE buffer",
     "        self.layers[layer][0][n:n + k.shape[0]] = k",
     "        self.layers[layer][1][n:n + k.shape[0]] = k"),

    ("OM15",
     "the second append OVERWRITES the first",
     "        self.layers[layer][0][n:n + k.shape[0]] = k",
     "        self.layers[layer][0][0:k.shape[0]] = k"),

    ("OM16",
     "the cache's WIDTH guard is dropped",
     "        if k.shape[1] != self.layers[layer][3]:\n"
     "            raise ValueError(\"k has %d columns, the layer was reserved with %d\"\n"
     "                             % (k.shape[1], self.layers[layer][3]))",
     "        pass"),

    ("OM17",
     "the cache's CAPACITY guard is dropped",
     "        if n + k.shape[0] > self.capacity:\n"
     "            raise ValueError(\"cache capacity exceeded\")",
     "        pass"),

    ("OM18",
     "the cache's ROW-COUNT guard is dropped",
     "        if k.shape[0] != v.shape[0]:\n"
     "            raise ValueError(\"k and v must have the same number of rows\")",
     "        pass"),

    ("OM19",
     "the cache's ZERO-ROWS guard is dropped",
     "        if k.shape[0] == 0:\n"
     "            raise ValueError(\"cannot append zero rows\")",
     "        pass"),

    ("OM20",
     "reset does not clear the layer",
     "        self.layers[layer][2] = 0\n\n"
     "    def reset_all(self):",
     "        pass\n\n"
     "    def reset_all(self):"),

    ("OM21",
     "total_tokens counts the LAYERS, not the tokens",
     "        return sum(l[2] for l in self.layers.values())",
     "        return len(self.layers)"),

    ("OM22",
     "memory_bytes counts ONE buffer per layer",
     "        return sum(2 * self.capacity * l[3] * 8 for l in self.layers.values())",
     "        return sum(self.capacity * l[3] * 8 for l in self.layers.values())"),
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

    workdir = tempfile.mkdtemp(prefix="kv_cache_oracle_mut_")
    target = os.path.join(workdir, "kv_cache_oracle.py")
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
