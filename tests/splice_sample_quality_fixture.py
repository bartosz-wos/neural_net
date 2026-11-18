#!/usr/bin/env python3
"""Splice the oracle's emitted fixture into the C++ test file.

THE REPO HAS BEEN BURNED BY HAND-TRANSCRIBED NUMBERS BEFORE: the numbers
were CORRECT for the fixture they were given, and the fixture was simply
not the oracle's. Pinned numbers transfer only across an IDENTICAL fixture,
so they are EMITTED, never typed.

This script replaces the region between the two marker comments in
tests/test_sample_quality.cpp with the oracle's --emit-cpp output.
"""
import subprocess
import sys

ORACLE = "tests/sample_quality_oracle.py"
TEST = "tests/test_sample_quality.cpp"
BEGIN = "// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<"
END = "// >>> ORACLE-EMITTED FIXTURE END <<<"

out = subprocess.run(["/usr/bin/python3", ORACLE, "--emit-cpp"],
                     capture_output=True, text=True)
if out.returncode != 0:
    print("the oracle failed:\n%s" % out.stdout)
    sys.exit(1)

lines = out.stdout.split("\n")
# Drop the oracle's banner (the "=== ... ===" header) and its trailing
# summary line, keeping the emitted C++ declarations AND their bodies.
start = next(i for i, l in enumerate(lines) if l.startswith("// ===="))
kept = []
for l in lines[start:]:
    if not l.strip():
        continue
    # The oracle's own progress/summary lines start with "=== " (the emitted
    # banner starts with "// ====", which is kept).
    if l.startswith("=== "):
        continue
    kept.append(l)

src = open(TEST).read()
b = src.index(BEGIN)
e = src.index(END)
new = src[:b] + BEGIN + "\n" + "\n".join(kept) + "\n" + END + src[e + len(END):]
open(TEST, "w").write(new)
print("spliced %d fixture lines into %s" % (len(kept), TEST))
