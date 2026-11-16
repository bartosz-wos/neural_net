#!/usr/bin/env python3
"""Splice the oracle's emitted fixture into the C++ test file.

THE REPO HAS BEEN BURNED BY HAND-TRANSCRIBED NUMBERS BEFORE: both numbers
were CORRECT for the fixture they were given, and the fixture was simply not
the oracle's. Pinned numbers transfer only across an IDENTICAL fixture, so
they are EMITTED, never typed.

This script replaces the region between the two marker comments in
tests/test_block_cached_decode.cpp with the oracle's --emit-cpp output.
"""
import re
import subprocess
import sys

ORACLE = "tests/block_cached_decode_oracle.py"
TEST = "tests/test_block_cached_decode.cpp"
BEGIN = "// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<"
END = "// >>> ORACLE-EMITTED FIXTURE END <<<"

out = subprocess.run(["/usr/bin/python3", ORACLE, "--emit-cpp"],
                     capture_output=True, text=True)
if out.returncode != 0:
    print("the oracle failed:\n%s" % out.stderr)
    sys.exit(1)

lines = out.stdout.split("\n")
# Drop the oracle's banner (the first three lines) and the trailing comment
# block, keeping only the C++ declarations.
start = next(i for i, l in enumerate(lines) if l.startswith("// ===="))
body = [l for l in lines[start:] if l.strip()]
# Keep only the `const` declarations and the leading comment.
kept = [l for l in body if l.startswith("//") or l.startswith("const ")]
fixture = "\n".join(kept)

src = open(TEST).read()
b = src.index(BEGIN)
e = src.index(END)
new = src[:b] + BEGIN + "\n" + fixture + "\n" + END + src[e + len(END):]
open(TEST, "w").write(new)
print("spliced %d fixture lines into %s" % (len(kept), TEST))
