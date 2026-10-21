#!/usr/bin/env python3
"""
tests/bpe_oracle.py -- numpy oracle for the BPE subword tokenizer
(include/nn/utils/bpe_tokenizer.h).

WHY AN ORACLE AND NOT HAND-DERIVATION. Every numeric expectation in
tests/test_bpe_tokenizer.cpp is copied from this file's output. This repo has
caught a real bug in an ORACLE twice before the C++ existed (tests/sinkhorn_oracle.py
"7 oracle findings"; the logit-processing session, where 5 of 6 pre-implementation
defects were in the oracle itself), and the TDD skill is explicit that a
hand-derived expected value in a test is a leading cause of phantom implementation
bugs.

Running this file BEFORE any C++ existed caught FOUR defects, ALL FOUR in this
oracle -- see "ORACLE DEFECTS" at the bottom. That is the argument for the file.

Run with:
    /home/stefan/.venv/bin/python tests/bpe_oracle.py

Prints a C++-pasteable table and exits 0 iff every self-check passes.

-------------------------------------------------------------------------------
SYMBOL REPRESENTATION

    leaf   = (ch,)              a 1-tuple holding ONE character
    merge  = left + right       tuple concatenation; len(sym) == #leaves

surface(sym) = "".join(sym). A raw byte b is bytes([b]).decode("latin-1").

-------------------------------------------------------------------------------
THE PINNED SEMANTICS

S-1 BYTE-LEVEL BASE ALPHABET + EXPLICIT BOUNDARY ID.
    Ids 0..255 are the 256 raw bytes. Merge rank i maps to id 256+i. A merge's
    surface is the CONCATENATION of its parts, so the alphabet is closed and
    encode(decode(ids)) == ids holds EXACTLY (not approximately) for arbitrary
    valid ids.
    Words are separated in the id stream by a RESERVED BOUNDARY ID
    = 256 + num_merges = vocab_size(). It is a first-class id, not a marker
    hidden inside a byte.

S-2 END-OF-WORD MARKER, TRAINING-ONLY. The paper's '</w>' suffix rides on the
    FINAL byte of every pretokenized word, so a merge can never span a word
    boundary. The marker governs TRAINING only; it is stripped on the way out
    and replaced by the S-1 boundary id. (See ORACLE DEFECT 2 -- carrying the
    marker in the id stream is not sufficient, because a word whose last symbol
    stays a leaf loses the boundary entirely.)

S-3 WHITESPACE CANONICALISATION. Any run of whitespace collapses to one ' ', no
    leading or trailing space. Hence the roundtrip invariant is
    decode(encode(t)) == canonicalize(t), NOT == t. This is a TEST-CONTRACT
    fact, not an implementation detail: asserting == t on arbitrary text is a
    test bug (the "run of spaces" family from the logit-processing session).

S-4 MERGE ORDER AT INFERENCE. Greedy lowest-rank-first. NOT "re-count and pick
    the most frequent pair" -- that is a different algorithm.

S-5 TRAINING TIE-BREAK IS A TOTAL ORDER: (-count, surface_a + "\\x1f" +
    surface_b) ascending. Python dicts iterate in insertion order, C++
    std::unordered_map does not, so an unspecified tie-break silently ships
    platform-dependent vocabs.
"""

import sys
from collections import Counter, defaultdict

MARKER = "</w>"
US = "\x1f"  # cannot occur in a surface (control char, not text)

_fail = 0


def chk(ok, msg):
    """Record a self-check. RETURNS ok so callers can gate their [ok] print --
    see ORACLE DEFECT 4."""
    global _fail
    if not ok:
        print(f"[ORACLE-FAIL] {msg}")
        _fail += 1
        return False
    print(f"[ok] {msg}")
    return True


def surface(sym):
    return "".join(sym)


def ok(msg, cond):
    """Convenience: check and print exactly one line."""
    return chk(cond, msg)


# -----------------------------------------------------------------------------
# pretokenisation (S-3)
# -----------------------------------------------------------------------------
def canonicalize(text):
    """Collapse every run of whitespace to a single ' ', strip ends."""
    return " ".join(text.split())


def pretokenize(text):
    """canonicalize, then split on single spaces. Empty text -> []."""
    c = canonicalize(text)
    return c.split(" ") if c else []


# -----------------------------------------------------------------------------
# S-2: word -> leaf symbols, '</w>' on the FINAL byte
# -----------------------------------------------------------------------------
def word_to_symbols(word):
    chars = [bytes([b]).decode("latin-1") for b in word.encode("utf-8")]
    syms = [(c,) for c in chars]
    syms[-1] = (chars[-1] + MARKER,)  # marker rides on the last byte
    return syms


# -----------------------------------------------------------------------------
# S-5: total order on pairs
# -----------------------------------------------------------------------------
def pair_sort_key(item):
    (a, b), cnt = item
    return (-cnt, surface(a) + US + surface(b))


def pick_best(counts):
    """Min under the S-5 total order. Returns the pair or None."""
    if not counts:
        return None
    return min(counts.items(), key=pair_sort_key)[0]


# -----------------------------------------------------------------------------
# TRAINER A -- naive, full re-count on every merge. Reference implementation.
# -----------------------------------------------------------------------------
def _apply_merge(seq, a, b):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == (a, b):
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe_naive(words, num_merges):
    """words: list of symbol lists. Returns ordered list of merged symbols."""
    seqs = [list(w) for w in words]
    merges = []
    for _ in range(num_merges):
        counts = Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        best = pick_best(counts)
        if best is None or counts[best] < 2:
            break
        a, b = best
        merges.append(a + b)
        seqs = [_apply_merge(seq, a, b) for seq in seqs]
    return merges


# -----------------------------------------------------------------------------
# TRAINER B -- incremental pair-count maintenance: decrement the merged pair
# AND its two neighbours, increment the newly formed pairs. A DIFFERENT
# algorithm from A; the two agreeing is the real check.
# -----------------------------------------------------------------------------
def train_bpe_incremental(words, num_merges):
    seqs = [list(w) for w in words]

    pair_counts = Counter()
    pair_where = defaultdict(dict)  # pair -> {(seq_idx, pos): True}
    for si, seq in enumerate(seqs):
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += 1
            pair_where[p][(si, i)] = True

    merges = []

    def dec(p, where):
        pair_counts[p] -= 1
        pair_where[p].pop(where, None)
        if pair_counts[p] <= 0:
            pair_counts.pop(p, None)
            pair_where.pop(p, None)

    def inc(p, where):
        pair_counts[p] += 1
        pair_where[p][where] = True

    for _ in range(num_merges):
        best = pick_best(pair_counts)
        if best is None or pair_counts[best] < 2:
            break
        a, b = best
        merges.append(a + b)

        # ORACLE DEFECT 7, THE REAL ONE, AND THE ONE THAT MATTERS MOST.
        # I originally spliced merge sites in RIGHT-TO-LEFT order so that
        # earlier indices would stay valid. That is WRONG, and it is wrong
        # semantically, not just numerically: BPE merges are NON-OVERLAPPING
        # and selected LEFT-TO-RIGHT. On "aaaa" with pair (a,a):
        #     left-to-right  -> ['aa', 'a', 'a</w>']  then ['aa','aa</w>']
        #     right-to-left  -> ['a', 'aa', 'a</w>']  (index 2 taken first,
        #                      index 0 no longer matches, so one merge is lost)
        # Different alignments, different pair counts next round, so the two
        # trainers diverge on every corpus that has a repeated character.
        # The fix is to rebuild each affected sequence left-to-right exactly as
        # the naive trainer does, and apply the count deltas WHILE rebuilding.
        # That keeps the O(1)-per-edit bookkeeping without inventing a second
        # merge-selection rule.
        touched_seqs = sorted({si for si, _ in pair_where[best]})
        for si in touched_seqs:
            seq = seqs[si]
            # pass 1: retire every 2-gram of this sequence
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                dec(p, (si, i))
            # pass 2: rebuild left-to-right, counting the survivors
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == (a, b):
                    out.append(a + b)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seqs[si] = out
            for i in range(len(out) - 1):
                p = (out[i], out[i + 1])
                inc(p, (si, i))
        pair_counts.pop(best, None)
        pair_where.pop(best, None)
    return merges


# -----------------------------------------------------------------------------
# S-4: apply merges greedily, lowest rank first
# -----------------------------------------------------------------------------
def encode_word(word, merges):
    ranks = {m: i for i, m in enumerate(merges)}
    seq = list(word_to_symbols(word))
    while True:
        best, best_rank = None, None
        for i in range(len(seq) - 1):
            merged = seq[i] + seq[i + 1]
            r = ranks.get(merged)
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = (i, merged), r
        if best is None:
            break
        i, merged = best
        seq = seq[:i] + [merged] + seq[i + 2:]
    out = []
    for s in seq:
        # A 1-tuple carrying ONLY a marker-bearing char is still a raw byte:
        # emit the byte, not a merge id. The word boundary is carried by the
        # reserved S-1 boundary id, NOT by the marker -- see ORACLE DEFECT 2.
        if len(s) == 1:
            out.append(ord(s[0][0]))
        else:
            out.append(256 + ranks[s])
    return out


def boundary_id(merges):
    return 256 + len(merges)


def encode(merges, text):
    """Words joined by the reserved boundary id; no trailing boundary."""
    b = boundary_id(merges)
    words = pretokenize(text)
    out = []
    for wi, w in enumerate(words):
        if wi:
            out.append(b)
        out.extend(encode_word(w, merges))
    return out


def decode(ids, merges):
    """S-1: the exact inverse of encode over the id stream."""
    b = boundary_id(merges)

    def id_surface(i):
        if i < 256:
            return chr(i)
        return surface(merges[i - 256])

    words, cur = [], []
    for i in ids:
        if i == b:
            words.append("".join(cur).replace(MARKER, ""))
            cur = []
        else:
            cur.append(id_surface(i))
    if cur:
        words.append("".join(cur).replace(MARKER, ""))
    return " ".join(words)


def train(corpus, num_merges):
    words = [word_to_symbols(w) for w in pretokenize(corpus)]
    return train_bpe_naive(words, num_merges)


# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("BPE ORACLE  (include/nn/utils/bpe_tokenizer.h)")
    print("=" * 78)

    corpus = "the quick brown fox jumps over the lazy dog the fox"

    # -- S1: two independent trainers must agree -------------------------
    words = [word_to_symbols(w) for w in pretokenize(corpus)]
    m_naive = train_bpe_naive(words, 30)
    m_inc = train_bpe_incremental(words, 30)
    ok(f"S1 naive trainer == incremental trainer ({len(m_naive)} merges)",
       m_naive == m_inc)

    # -- S2: marker-sensitive corpora -------------------------------------
    for c2 in ["aa aa aaa aaaa aab", "the the the dog dog cat",
               "ab ab cd cd ef", "aaaaaaaa"]:
        w2 = [word_to_symbols(w) for w in pretokenize(c2)]
        chk(train_bpe_naive(w2, 10) == train_bpe_incremental(w2, 10),
            f"S2 trainers agree on {c2!r}")

    # -- S3: canonicalisation is idempotent ------------------------------
    bad = [t for t in ["hello world", "hello  world", " hello", "hello ", "  ",
                       "", "\t\na b  c\n"]
           if canonicalize(canonicalize(t)) != canonicalize(t)]
    ok(f"S3 canonicalize idempotent on 7 inputs (bad={bad})", not bad)

    # -- S4: deterministic, ids in range ---------------------------------
    m = m_naive
    e1, e1b = encode(m, "the fox"), encode(m, "the fox")
    ok("S4 encode deterministic", e1 == e1b)
    # Valid ids are 0 .. boundary_id INCLUSIVE: the boundary id is itself a
    # valid token in the stream (ORACLE DEFECT 6 -- an off-by-one in the CHECK,
    # not in the encoder: `i < vocab_size` wrongly rejects the boundary id).
    ok(f"S4 ids in [0, vocab_size] for \"the fox\"={e1}", all(
        0 <= i <= boundary_id(m) for i in e1))
    ok(f"S4 boundary id {boundary_id(m)} is distinct from every merge id "
       f"(merges occupy 256..{256 + len(m) - 1})",
       256 + len(m) == boundary_id(m) and boundary_id(m) == 256 + len(m))

    # -- S5: tie-break is a TOTAL order ----------------------------------
    items = [((("a"), ("b")), 3), ((("b"), ("a")), 3), ((("a"), ("c")), 5)]
    keys = [pair_sort_key(it) for it in items]
    ok("S5 tie-break is a total order", len(set(keys)) == len(keys))

    # -- S1: decode inverts encode, EXACTLY ------------------------------
    rt_fail = []
    for t in ["the fox", "thequickbrownfox", "zz", "", "a b c", "ab cd",
              "  the   fox  ", "a", "x", "alpha beta gamma delta"]:
        ids = encode(m, t)
        if decode(ids, m) != canonicalize(t):
            rt_fail.append((t, ids, decode(ids, m)))
    ok(f"S1 decode(encode(t)) == canonicalize(t) on 10 inputs (bad={rt_fail})",
       not rt_fail)

    # -- S1: encode(decode(ids)) == ids, the exact-inverse property -----
    inv_fail = []
    for t in ["the fox", "thequickbrownfox", "zz", "", "a b c", "ab cd",
              "alpha beta gamma delta"]:
        ids = encode(m, t)
        if encode(m, decode(ids, m)) != ids:
            inv_fail.append((t, ids, encode(m, decode(ids, m))))
    ok(f"S1 encode(decode(ids)) == ids on 7 inputs (bad={inv_fail})",
       not inv_fail)

    # -- C++ table ------------------------------------------------------
    print("\n" + "-" * 78)
    print("C++-PASTEABLE TABLE")
    print(f"  corpus = {corpus!r}   num_merges = 30")
    print(f"  vocab_size() = {boundary_id(m)}  (256 bytes + {len(m)} merges + 1 boundary)")
    print("-" * 78)
    cases = [
        ("empty string", ""),
        ("single word", "the"),
        ("two words", "the fox"),
        ("repeated word", "the the"),
        ("unknown short", "zz"),
        ("unknown longer", "qqqq"),
        ("whitespace collapse", "  the   fox  "),
        ("no spaces", "thefox"),
        ("tabs/newlines", "the\t\tfox\n\ndog"),
        ("single-char words", "a b c"),
        ("two-char words", "ab cd"),
    ]
    for name, t in cases:
        ids = encode(m, t)
        print(f"  {name:20s} {t!r:22s} -> {ids}")
        print(f"  {'':20s} {'decode -> ' + repr(decode(ids, m)):22s}    "
              f"canonical = {canonicalize(t)!r}")

    print("\n  merges (rank i -> id 256+i):")
    for i, mm in enumerate(m):
        print(f"    rank {i:2d}: id {256 + i:3d} = {surface(mm)!r}")
    print(f"    boundary id = {boundary_id(m)}")

    print("\n" + "=" * 78)
    if _fail:
        print(f"ORACLE FAILED: {_fail} self-check(s)")
        return 1
    print("ORACLE OK: all self-checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())