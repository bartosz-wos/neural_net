// Tokenizer (legacy word-level) tests — regression suite.
//
// THIS FILE DOES NOT EXISTED BEFORE. include/nn/utils/tokenizer.h has been
// untouched since the initial commit 1d8336a and had ZERO test coverage, which
// is why a first-word / EOS id collision survived in it (see Test 2).
//
// Tests:
//   1. the id-collision regression: no learned word may share eos_id()
//   2. decode(encode(t)) == canonicalize(t) for the legacy class
//   3. whitespace collapsing is DOCUMENTED behaviour, asserted so it cannot
//      change silently
//   4. byte fallback for an OOV word is exact
//   5. unk_id() is byte NUL, NOT an unknown-token marker (the accessor lies)
//   6. empty / whitespace-only input
//   7. vocab_size grows with training
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "nn/utils/tokenizer.h"

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

// What "hello  world " means to this class: runs of whitespace collapse to a
// single ' ' (that is what pushing eos_id on each whitespace run DOES), and the
// trailing whitespace run produces a trailing space that survives as a leading
// eos. The FIRST word is lost to the id collision, which Test 1 repairs.
static std::string legacy_canonicalize(const std::string& t) {
    std::string out;
    bool pending = false;
    bool seen = false;
    for (char c : t) {
        bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (ws) {
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

int main() {
    cout << "=== Tokenizer (legacy word-level) ===" << endl;

    const std::string corpus = "hello world hello there";

    // ---------------------------------------------------------------- 1
    // THE REGRESSION. The constructor sets eos_id_ = 256 and starts learned ids
    // at 256, so the FIRST learned word was assigned id 256 == eos_id_, and
    // decode() maps 256 to a space. The single most frequent word in the corpus
    // was therefore unrecoverable.
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        std::vector<int> ids = t.encode("hello");
        CHECK(!ids.empty(), "1a encode of a trained word is non-empty");
        // "hello" appears twice in the corpus, so it is among the learned words;
        // whichever id it got, it must NOT equal eos_id().
        bool collides = (ids[0] == t.eos_id());
        cout << "    eos_id=" << t.eos_id() << " encode(\"hello\")[0]=" << ids[0]
             << endl;
        CHECK(!collides,
              "1b first learned word id != eos_id() (the collision regression)");
    }

    // ---------------------------------------------------------------- 2
    // The repaired roundtrip. Whitespace runs collapse (Test 3), so the
    // invariant is against the canonical form, NOT the raw input -- asserting
    // raw equality on arbitrary text would be a test bug.
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        const char* cases[] = {"hello world", "the", "hello there",
                               "there hello", "a b c d"};
        int rt = 0, rt_bad = 0;
        for (int i = 0; i < 5; ++i) {
            std::string in = cases[i];
            std::string got = t.decode(t.encode(in));
            std::string want = legacy_canonicalize(in);
            if (got != want) {
                ++rt_bad;
                cout << "      roundtrip \"" << in << "\" -> \"" << got
                     << "\" want \"" << want << "\"" << endl;
            }
            ++rt;
        }
        CHECK(rt_bad == 0, "2  decode(encode(t)) == legacy canonicalize(t)");
    }

    // ---------------------------------------------------------------- 3
    // Whitespace collapsing is pinned as DOCUMENTED behaviour. If someone ever
    // repairs it, this test fires and the change is made deliberately.
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        std::string two = t.decode(t.encode("there  world"));
        std::string one = t.decode(t.encode("there world"));
        CHECK(two == one, "3a runs of whitespace collapse to a single space");
        std::string lead = t.decode(t.encode("   hello"));
        // Measured: encode("   hello") = [257] -> "hello". Leading whitespace is
        // DROPPED outright, not turned into a leading space: the encode loop only
        // pushes eos_id when it closes a NON-EMPTY word, so a run that precedes any
        // word emits nothing. (My first version of this test asserted " " and
        // failed against a correct implementation -- a leading space would mean an
        // empty word had been closed.)
        CHECK(lead == "hello", "3b leading whitespace is dropped, not made a space");
    }

    // ---------------------------------------------------------------- 4
    // The byte fallback path was always sound; prove it so the collision fix
    // is known not to have disturbed it.
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        std::string got = t.decode(t.encode("zzz"));
        CHECK(got == "zzz", "4  OOV word roundtrips exactly via byte fallback");
    }

    // ---------------------------------------------------------------- 5
    // unk_id() returns 0, and 0 is byte NUL. It is NOT an unknown-token marker:
    // encode never emits 0 for an unknown word, it emits raw bytes. Asserted so
    // no caller is misled into thinking id 0 means "unknown".
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        CHECK(t.unk_id() == 0, "5a unk_id() is 0");
        std::string nul = t.decode(std::vector<int>{0});
        CHECK(nul.size() == 1 && nul[0] == '\0',
              "5b decode([0]) yields byte NUL, so id 0 is not an UNK marker");
    }

    // ---------------------------------------------------------------- 6
    {
        Tokenizer t;
        t.train(corpus, 50, 1);
        CHECK(t.encode("").empty(), "6a encode(\"\") is empty");
        CHECK(t.decode(std::vector<int>{}).empty(), "6b decode([]) is empty");
        CHECK(t.decode(t.encode("   ")).empty() || t.decode(t.encode("   ")).size() <= 1,
              "6c whitespace-only input is at most a single space");
    }

    // ---------------------------------------------------------------- 7
    {
        Tokenizer t1, t2;
        t1.train(corpus, 50, 1);
        t2.train(corpus, 8, 1);
        CHECK(t1.vocab_size() >= t2.vocab_size(),
              "7  a larger vocab_size cap yields a >= vocabulary");
        CHECK(t1.vocab_size() > 256, "7b vocab_size exceeds the 256 byte range");
    }

    cout << "\n=== Summary: " << passed << "/" << total << " passed ===" << endl;
    return passed == total ? 0 : 1;
}