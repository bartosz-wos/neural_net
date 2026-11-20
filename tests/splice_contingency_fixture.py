#!/usr/bin/env python3
"""Splice the oracle's emitted fixture into the C++ test file.

THE REPO HAS BEEN BURNED BY HAND-TRANSCRIBED NUMBERS BEFORE: the numbers
were CORRECT for the fixture they were given, and the fixture was simply
not the oracle's. Pinned numbers transfer only across an IDENTICAL fixture,
so they are EMITTED, never typed.

This script replaces the region between the two marker comments in
tests/test_contingency.cpp with the oracle's --emit-cpp output.

IT FAILS LOUDLY. A splice that silently did nothing -- because the oracle
crashed, or because a marker was missing, or because the output did not
contain the fixture -- would leave the suite pinning STALE numbers while
looking exactly as if it had been refreshed. That is the failure this
script exists to make impossible:

  * the oracle exits non-zero        -> error, exit 1
  * the oracle's output has no marker -> error, exit 1
  * the test file is missing          -> error, exit 1
  * the test file has no marker       -> error, exit 1
  * the test file has TWO of a marker -> error, exit 1 (ambiguous)

Every one of those paths prints a message to stderr and exits non-zero.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ORACLE = os.path.join(HERE, "contingency_oracle.py")
TEST = os.path.join(HERE, "test_contingency.cpp")

BEGIN = "// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<"
END = "// >>> ORACLE-EMITTED FIXTURE END <<<"

# THE REPO'S CONVENTION: the oracles run under /usr/bin/python3, which is
# the interpreter that has numpy and scipy. The sibling scripts hardcode
# it for the same reason (`tests/survival_oracle.py`,
# `tests/dtw_mutate_oracle.py`, `tests/block_cached_decode_mutate_oracle.py`,
# `tests/edit_distance_mutate_oracle.py`, and
# `tests/splice_sample_quality_fixture.py` all name it explicitly).
# Using `sys.executable` here would run the oracle under whatever
# interpreter happens to be first on PATH -- which, inside the Hermes
# agent's own venv, is NOT the one with numpy.
PYTHON = "/usr/bin/python3"


def fail(message):
    sys.stderr.write("splice_contingency_fixture: %s\n" % message)
    sys.exit(1)


if not os.path.isfile(ORACLE):
    fail("the oracle %s does not exist" % ORACLE)
if not os.path.isfile(TEST):
    fail("the test file %s does not exist" % TEST)

out = subprocess.run([PYTHON, ORACLE, "--emit-cpp"],
                     capture_output=True, text=True, cwd=REPO)
if out.returncode != 0:
    sys.stderr.write(out.stdout)
    sys.stderr.write(out.stderr)
    fail("the oracle failed with exit status %d" % out.returncode)

lines = out.stdout.split("\n")
# Keep only the emitted C++ declarations and their bodies. The oracle's
# emitter opens with the BEGIN marker and ends with the END marker; the
# region BETWEEN the two markers is the fixture, and everything outside it
# is the oracle's own commentary.
start = -1
end = -1
for i, l in enumerate(lines):
    if start < 0 and l.startswith(BEGIN):
        start = i
    elif start >= 0 and l.startswith(END):
        end = i
        break
if start < 0 or end < 0:
    sys.stderr.write(out.stdout[:2000])
    fail("the oracle's output has no fixture markers")
if end <= start:
    fail("the oracle's output has its markers in the wrong order")
kept = [l for l in lines[start + 1:end] if l.strip()]
if not kept:
    fail("the oracle's output has an EMPTY fixture region")

with open(TEST) as f:
    src = f.read()

if src.count(BEGIN) != 1:
    fail("the test file has %d BEGIN markers (expected exactly 1)"
         % src.count(BEGIN))
if src.count(END) != 1:
    fail("the test file has %d END markers (expected exactly 1)"
         % src.count(END))

b = src.index(BEGIN)
e = src.index(END)
if e <= b:
    fail("the test file has its markers in the wrong order")

new = src[:b] + BEGIN + "\n" + "\n".join(kept) + "\n" + END + src[e + len(END):]
with open(TEST, "w") as f:
    f.write(new)
print("spliced %d fixture lines into %s" % (len(kept), TEST))
