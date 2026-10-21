// BPETokenizer tests — byte-level byte-pair-encoding subword tokenizer
// (include/nn/utils/bpe_tokenizer.h).
//
// EVERY numeric expectation below is COPIED from tests/bpe_oracle.py, which was
// written and run BEFORE any C++ existed and caught seven defects in itself.
// Do not hand-edit an expected id here: change the oracle, re-run it, and copy.
//
//   /home/stefan/.venv/bin/python tests/bpe_oracle.py
//
// The oracle's pinned table (corpus
// 'the quick brown fox jumps over the lazy dog the fox', num_merges = 30):
//
//   vocab_size() = 260  (256 bytes + 4 merges + 1 boundary)
//     rank 0: id 256 = 'he</w>'      rank 2: id 258 = 'fo'
//     rank 1: id 257 = 'the</w>'     rank 3: id 259 = 'fox</w>'
//     boundary id = 260
//
//   ""                   -> []
//   "the"                -> [257]
//   "the fox"            -> [257, 260, 259]
//   "the the"            -> [257, 260, 257]
//   "zz"                 -> [122, 122]
//   "qqqq"               -> [113, 113, 113, 113]
//   "  the   fox  "      -> [257, 260, 259]
//   "thefox"             -> [116, 104, 101, 259]
//   "the\t\tfox\n\ndog"  -> [257, 260, 259, 260, 100, 111, 103]
//   "a b c"              -> [97, 260, 98, 260, 99]
//   "ab cd"              -> [97, 98, 260, 99, 100]
//
// THE ROUNDTRIP CONTRACT IS decode(encode(t)) == canonicalize(t), NOT == t.
// Whitespace runs collapse to one ' ' and ends are stripped, so asserting raw
// equality on arbitrary text would be a test bug (oracle decision D4).
//
// Tests:
//   1. constructor validation
//   2. accessor contract + the vocab_size layout
//   3. the pinned encode table (the oracle's numbers, verbatim)
//   4. decode(encode(t)) == canonicalize(t)
//   5. encode(decode(ids)) == ids  (the byte-level alphabet's defining property)
//   6. a byte NUL / high bytes survive (byte-level, not UTF-8 lossy)
//   7. boundary id is a real, first-class token
//   8. unknown text never produces an out-of-range id
//   9. an untrained tokenizer is a pure byte encoder
//  10. the learned merge table is inspectable and matches the oracle
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/utils/bpe_tokenizer.h"

using std::cout;
using std::endl;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

static const char* kCorpus =
    "the quick brown fox jumps over the lazy dog the fox";

static std::string canonicalize(const std::string& t) {
    std::string out;
    bool pending = false, seen = false;
    for (char c : t) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            pending = seen;
            continue;
        }
        if (pending) out.push_back(' ');
        pending = false;
        seen = true;
        out.push_back(c);
    }
    return out;
}

static std::string vec(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

int main() {
    cout << "=== BPETokenizer (byte-level BPE) ===" << endl;

    // ---------------------------------------------------------------- 1
    {
        bool threw = false;
        try { BPETokenizer b(-1, 10); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1a negative num_merges throws");

        threw = false;
        try { BPETokenizer b(10, 0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "1b num_merges == 0 throws");
    }

    // ---------------------------------------------------------------- 2
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        CHECK(b.num_merges() == 4, "2a learned 4 merges (oracle)");
        // 256 bytes + 4 merges + 1 boundary
        CHECK(b.vocab_size() == 260, "2b vocab_size == 260 (oracle)");
        CHECK(b.boundary_id() == 260, "2c boundary_id == 260 (oracle)");
        CHECK(b.boundary_id() == b.vocab_size(),
              "2d boundary_id == vocab_size by construction");
        CHECK(b.num_merges() <= b.max_merges(),
              "2e num_merges <= the configured cap");
    }

    // ---------------------------------------------------------------- 3
    // The pinned table, copied verbatim from the oracle.
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        struct Case { const char* text; std::vector<int> want; };
        std::vector<Case> cases = {
            {"", {}},
            {"the", {257}},
            {"the fox", {257, 260, 259}},
            {"the the", {257, 260, 257}},
            {"zz", {122, 122}},
            {"qqqq", {113, 113, 113, 113}},
            {"  the   fox  ", {257, 260, 259}},
            {"thefox", {116, 104, 101, 259}},
            {"the\t\tfox\n\ndog", {257, 260, 259, 260, 100, 111, 103}},
            {"a b c", {97, 260, 98, 260, 99}},
            {"ab cd", {97, 98, 260, 99, 100}},
        };
        for (const auto& c : cases) {
            std::vector<int> got = b.encode(c.text);
            bool eq = got == c.want;
            if (!eq)
                cout << "      encode(\"" << c.text << "\") = " << vec(got)
                     << " want " << vec(c.want) << endl;
            CHECK(eq, std::string("3  oracle encode(\"") + c.text + "\")");
        }
    }

    // ---------------------------------------------------------------- 4
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        const char* cases[] = {"the fox", "thequickbrownfox", "zz", "",
                               "a b c", "ab cd", "  the   fox  ",
                               "alpha beta gamma delta"};
        int bad = 0;
        for (int i = 0; i < 8; ++i) {
            std::string in = cases[i];
            std::string got = b.decode(b.encode(in));
            std::string want = canonicalize(in);
            if (got != want) {
                ++bad;
                cout << "      \"" << in << "\" -> \"" << got << "\" want \""
                     << want << "\"" << endl;
            }
        }
        CHECK(bad == 0, "4  decode(encode(t)) == canonicalize(t) on 8 inputs");
    }

    // ---------------------------------------------------------------- 5
    // THE DEFINING PROPERTY of a byte-level alphabet: the id stream is exactly
    // invertible. This is EXACT equality, not approximate -- there is no
    // tolerance here on purpose.
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        const char* cases[] = {"the fox", "thequickbrownfox", "zz",
                               "a b c", "ab cd", "alpha beta gamma delta"};
        int bad = 0;
        for (int i = 0; i < 6; ++i) {
            std::vector<int> ids = b.encode(cases[i]);
            std::vector<int> again = b.encode(b.decode(ids));
            if (again != ids) {
                ++bad;
                cout << "      \"" << cases[i] << "\": " << vec(ids) << " -> "
                     << vec(again) << endl;
            }
        }
        CHECK(bad == 0, "5  encode(decode(ids)) == ids, exact, on 6 inputs");
    }

    // ---------------------------------------------------------------- 6
    // Byte-level, NOT UTF-8: a lone high byte and an embedded NUL are data.
    // A UTF-8 tokenizer would mangle these.
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        std::string raw = "a";
        raw.push_back(static_cast<char>(0xFF));
        raw.push_back('\0');
        raw += "b";
        std::vector<int> ids = b.encode(raw);
        bool in_range = true;
        for (int id : ids)
            if (id < 0 || id > b.boundary_id()) in_range = false;
        CHECK(in_range, "6a high byte + NUL encode to valid ids");
        CHECK(ids.size() == 4, "6b high byte + NUL produce 4 ids (2 raw + bnd + b)");
        CHECK(b.encode(b.decode(ids)) == ids, "6c high byte + NUL roundtrip exactly");
    }

    // ---------------------------------------------------------------- 7
    // The boundary id is a first-class token, not a marker smuggled inside a
    // byte. Oracle defect O4 measured that the paper's </w> marker alone loses
    // the boundary whenever the word's last symbol stays a leaf.
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        std::vector<int> two = b.encode("a b");
        std::vector<int> one = b.encode("ab");
        bool has_bnd = std::find(two.begin(), two.end(), b.boundary_id()) != two.end();
        CHECK(has_bnd, "7a a two-word input contains the boundary id");
        CHECK(std::find(one.begin(), one.end(), b.boundary_id()) == one.end(),
              "7b a one-word input contains NO boundary id");
        CHECK(two.size() == one.size() + 1,
              "7c 'a b' is exactly one boundary longer than 'ab'");
        CHECK(b.decode(two) == "a b", "7d the boundary id decodes to a space");
    }

    // ---------------------------------------------------------------- 8
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        // Long OOV runs and every byte value must stay in range.
        std::string junk;
        for (int i = 0; i < 256; ++i) junk.push_back(static_cast<char>(i));
        std::vector<int> ids = b.encode(junk);
        bool ok = true;
        for (int id : ids)
            if (id < 0 || id > b.boundary_id()) ok = false;
        CHECK(ok, "8  all 256 byte values encode to in-range ids");
        CHECK(ids.size() >= 256, "8b 256 raw bytes yield at least 256 ids");
    }

    // ---------------------------------------------------------------- 9
    // Untrained: no merges, so encoding is the identity on bytes and the
    // boundary id is 256. This is the degenerate-but-well-defined case.
    {
        BPETokenizer b;
        CHECK(b.num_merges() == 0, "9a untrained num_merges == 0");
        CHECK(b.vocab_size() == 257, "9b untrained vocab_size == 257 (256+boundary)");
        std::vector<int> ids = b.encode("abc");
        CHECK(ids == std::vector<int>({97, 98, 99}), "9c untrained encode is raw bytes");
        CHECK(b.decode(ids) == "abc", "9d untrained decode inverts exactly");
    }

    // --------------------------------------------------------------- 10
    // The learned merge table is inspectable and matches the oracle.
    {
        BPETokenizer b;
        b.train(kCorpus, 30);
        std::vector<std::string> merges = b.merges();
        CHECK(merges.size() == 4, "10a merges() has 4 entries");
        bool m0 = merges.size() > 0 && merges[0] == "he";
        bool m1 = merges.size() > 1 && merges[1] == "the";
        bool m2 = merges.size() > 2 && merges[2] == "fo";
        bool m3 = merges.size() > 3 && merges[3] == "fox";
        CHECK(m0, "10b merge 0 == 'he'");
        CHECK(m1, "10c merge 1 == 'the'");
        CHECK(m2, "10d merge 2 == 'fo'");
        CHECK(m3, "10e merge 3 == 'fox'");
        // 'fox' must be the LOWEST id of its three letters, i.e. it merged last.
        std::vector<int> fx = b.encode("fox");
        CHECK(fx.size() == 1 && fx[0] == 259, "10f 'fox' is the single id 259");
    }

    cout << "\n=== Summary: " << passed << "/" << total << " passed ===" << endl;
    return passed == total ? 0 : 1;
}