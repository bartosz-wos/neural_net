#!/usr/bin/env python3
"""Sweep the repo for the FATAL form of the bug this session found.

THE BUG: `TransformerBlock` declared `Tensor grad_W1, grad_b1, grad_W2,
grad_b2;` and its constructor initialised NONE of them -- so all four were
0x0 EMPTY tensors, and `backward()` wrote `grad_W2[k][j] += ...` PAST THE
END of an empty buffer on every training step. `MultiHeadAttention`, next
to it, does it correctly.

THE FATAL FORM IS PRECISE, AND THE LOOSE FORM IS NOT:

    a member named like a gradient (grad*, *_grad) that is WRITTEN in the
    class's backward() and is NEVER INITIALISED anywhere in the class's
    .cpp.

The loose form -- "a Tensor member the constructor does not initialise" --
is dominated by legitimate patterns and was measured to be useless:

  * `last_*` members are set by `forward()`, not by the constructor;
  * optimiser state (`m`, `v`, `exp_avg_sq`) is lazily initialised on the
    first `step()`, which is the correct pattern;
  * `KVCache::keys`/`values` live in a NESTED struct and are assigned by
    `reserve()`.

The first version of this sweep looked only at the constructor's
INITIALISER LIST and reported ~190 false positives -- `MLAAttention` and
`QKNorm` both initialise their gradients in the constructor's BODY. A
sweep that reports a defect that is not there is worse than no sweep.

WHAT THIS CANNOT DO: it is STATIC. It cannot prove a member is initialised
with the RIGHT SHAPE, and it cannot see a member initialised in a base
class or in a helper. It finds the whole class of "never initialised at
all" defects, which is the one that corrupts memory.
"""
import os
import re
import sys

ROOT = "include/nn"

TENSOR_MEMBER = re.compile(
    r"^\s*Tensor\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*(=[^;]*)?;\s*$", re.M)
GRAD_NAME = re.compile(r"^(grad|.*_grad)|^(last_grad)", re.I)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def class_bodies(htext):
    """Yield (class_name, body_text) for every class in the header."""
    for cm in re.finditer(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*[^{]*\{", htext):
        cname = cm.group(1)
        depth = 0
        i = cm.end() - 1
        end = len(htext)
        while i < len(htext):
            if htext[i] == "{":
                depth += 1
            elif htext[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        yield cname, htext[cm.end():end]


def method_bodies(cpp_text, cname):
    """Yield (method_name, body_text) for the class's methods."""
    for mm in re.finditer(
            r"^\s*(?:[A-Za-z_][A-Za-z0-9_:<>, \*&]*\s+)?"
            + re.escape(cname) + r"::([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*"
            r"(?:const\s*)?\{", cpp_text, re.M):
        depth = 0
        i = mm.end() - 1
        while i < len(cpp_text):
            if cpp_text[i] == "{":
                depth += 1
            elif cpp_text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield mm.group(1), cpp_text[mm.end():i]


def main():
    findings = []
    checked = 0

    for dirpath, _dirs, files in os.walk(ROOT):
        for fn in files:
            if not fn.endswith(".h"):
                continue
            hpath = os.path.join(dirpath, fn)
            cpp_path = hpath[:-2] + ".cpp"
            if not os.path.exists(cpp_path):
                continue
            htext = strip_comments(open(hpath).read())
            ctext = strip_comments(open(cpp_path).read())

            for cname, body in class_bodies(htext):
                members = []
                for m in TENSOR_MEMBER.finditer(body):
                    if m.group(2):
                        continue          # initialised in the header
                    # A declaration may name SEVERAL tensors:
                    #   `Tensor grad_W1, grad_b1, grad_W2, grad_b2;`
                    # The first version of this regex matched only the
                    # FIRST name -- and `TransformerBlock`'s declaration is
                    # exactly a multi-name one, so the sweep MISSED the very
                    # bug it was written to find and reported "NO
                    # UNINITIALISED GRADIENT MEMBERS FOUND" on a tree that
                    # had the defect. A detector that cannot find the bug it
                    # was written for is worse than no detector.
                    for name in m.group(1).split(","):
                        name = name.strip()
                        if GRAD_NAME.match(name):
                            members.append(name)
                if not members:
                    continue
                checked += 1

                methods = dict(method_bodies(ctext, cname))
                backward = methods.get("backward", "")
                if not backward:
                    continue

                # Written in backward()?
                written = []
                for m in members:
                    if re.search(r"(^|[\s;{(,])" + re.escape(m) + r"\s*(\[[^\]]*\])*\s*(=[^=]|\+=|-=|\*=|/=)",
                                 backward):
                        written.append(m)
                if not written:
                    continue

                # Initialised ANYWHERE in the class's .cpp?
                for m in written:
                    if re.search(r"(^|[\s;{(,])" + re.escape(m) + r"\s*(=[^=]|\()", ctext):
                        continue
                    findings.append((hpath, cname, m))

    print("checked %d classes with gradient-shaped Tensor members" % checked)
    if not findings:
        print("\nNO UNINITIALISED GRADIENT MEMBERS FOUND")
        return 0
    print("\nGRADIENT MEMBERS WRITTEN IN backward() BUT NEVER INITIALISED:")
    seen = set()
    for path, cname, m in findings:
        key = (path, cname, m)
        if key in seen:
            continue
        seen.add(key)
        print("  %-56s %-26s %s" % (path, cname, m))
    print("\n%d distinct findings" % len(seen))
    return 1


if __name__ == "__main__":
    sys.exit(main())
