#!/usr/bin/env python3
"""Splice the oracle's emitted fixture into the C++ test file.

THE REPO HAS BEEN BURNED BY HAND-TRANSCRIBED NUMBERS BEFORE: the numbers
were CORRECT for the fixture they were given, and the fixture was simply
not the oracle's. Pinned numbers transfer only across an IDENTICAL fixture,
so they are EMITTED, never typed.

This script replaces the region between the two marker comments in
tests/test_survival.cpp with the oracle's --emit-cpp output.
"""
import subprocess
import sys

ORACLE = "tests/survival_oracle.py"
TEST = "tests/test_survival.cpp"
BEGIN = "// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<"
END = "// >>> ORACLE-EMITTED FIXTURE END <<<"

# THE REPO'S CONVENTION: the oracles run under /usr/bin/python3, which is
# the interpreter that has numpy and scipy. The sibling scripts hardcode
# it for the same reason (`tests/dtw_mutate_oracle.py`, `tests/
# block_cached_decode_mutate_oracle.py`, and
# `tests/splice_sample_quality_fixture.py` all name it explicitly).
# Using `sys.executable` here would run the oracle under whatever
# interpreter happens to be first on PATH -- which, inside the Hermes
# agent's own venv, is NOT the one with numpy.
PYTHON = "/usr/bin/python3"

out = subprocess.run([PYTHON, ORACLE, "--emit-cpp"],
                     capture_output=True, text=True)
if out.returncode != 0:
    print("the oracle failed:\n%s" % out.stdout)
    sys.exit(1)

lines = out.stdout.split("\n")
# Keep only the emitted C++ declarations and their bodies. The oracle's
# emitter opens with a "// ----" banner and ends with a "// >>> ORACLE-
# EMITTED FIXTURE END <<<" marker; the region BETWEEN the two markers is
# the fixture, and everything outside it is the oracle's own commentary.
try:
    start = next(i for i, l in enumerate(lines) if l.startswith("// >>> ORACLE-EMITTED FIXTURE BEGIN"))
    end = next(i for i, l in enumerate(lines) if l.startswith("// >>> ORACLE-EMITTED FIXTURE END"))
except StopIteration:
    print("the oracle's output has no fixture markers:\n%s" % out.stdout[:2000])
    sys.exit(1)
kept = [l for l in lines[start + 1:end] if l.strip()]

src = open(TEST).read()
b = src.index(BEGIN)
e = src.index(END)
new = src[:b] + BEGIN + "\n" + "\n".join(kept) + "\n" + END + src[e + len(END):]
open(TEST, "w").write(new)
print("spliced %d fixture lines into %s" % (len(kept), TEST))
