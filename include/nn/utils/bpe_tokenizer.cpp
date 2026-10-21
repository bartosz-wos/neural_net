#include "bpe_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>

// ===========================================================================
// Implementation notes, kept here rather than in the header because they are
// about HOW, not WHAT:
//
// * A training symbol is (id, marked). `marked` says whether the '</w>' marker
//   is attached, and it is NOT derivable from the id alone. This is the whole
//   reason the oracle models a symbol as a TUPLE and not a bare int: the same
//   id pair (h, e) means the merge "he" in the middle of a word and the merge
//   "he</w>" at the end of one. Keying a pair on (id_a, id_b) alone collapses
//   those two DISTINCT merges into one key. That bug is silent -- training
//   produces a plausible merge table and every assertion on merges() still
//   passes -- while encode() looks up marker-bearing keys and finds nothing.
//   Measured: train() learned keys he/the/fo/fox and encode("the") returned
//   [116, 104, 101] with not one merge applied.
//
// * Each round counts all adjacent pairs, picks the best under the total order
//   (-count, left + '\x1f' + right), and applies the merge NON-OVERLAPPING and
//   LEFT-TO-RIGHT. Recounting is O(total symbols) per round; the incremental
//   variant saves that but introduces a second merge-selection rule, which is
//   exactly the class of bug the oracle exists to catch. Correctness over speed
//   for a training-time routine. See the scan-order note in the header.
//
// * encode() applies merges greedily in LEARNED-RANK order (lowest rank first),
//   NOT by re-counting frequencies at encode time. Those are different
//   algorithms; only the first is "BPE".
// ===========================================================================

namespace {

const std::string kMarker = BPETokenizer::kWordEndMarker();
const char kSep = '\x1f';

inline bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

// Canonicalise (runs of whitespace -> one ' ', ends stripped) and split on the
// resulting single spaces. Returns the canonical form; fills `words`.
std::string pretokenize_into(const std::string& text, std::vector<std::string>* words) {
    words->clear();
    std::string out;
    bool pending = false, seen = false;
    for (char c : text) {
        if (is_ws(c)) { pending = seen; continue; }
        if (pending) out.push_back(' ');
        pending = false;
        seen = true;
        out.push_back(c);
    }
    if (!out.empty()) {
        std::string cur;
        for (char c : out) {
            if (c == ' ') { words->push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        words->push_back(cur);
    }
    return out;
}

inline std::string strip_marker(const std::string& s) {
    std::string out = s;
    size_t p;
    while ((p = out.find(kMarker)) != std::string::npos)
        out.erase(p, kMarker.size());
    return out;
}

} // namespace

std::string bpe_canonicalize(const std::string& text) {
    std::vector<std::string> unused;
    return pretokenize_into(text, &unused);
}

// ---------------------------------------------------------------------------
BPETokenizer::BPETokenizer(int max_merges)
    : max_merges_(max_merges < 0 ? 0 : static_cast<size_t>(max_merges)),
      min_frequency_(2) {
    if (max_merges < 0)
        throw std::invalid_argument("BPETokenizer: max_merges must be >= 0");
}

BPETokenizer::BPETokenizer(int min_frequency, int max_merges)
    : max_merges_(max_merges < 0 ? 0 : static_cast<size_t>(max_merges)),
      min_frequency_(min_frequency < 1 ? 1 : static_cast<size_t>(min_frequency)) {
    if (max_merges < 0)
        throw std::invalid_argument("BPETokenizer: max_merges must be >= 0");
    if (min_frequency < 1)
        throw std::invalid_argument("BPETokenizer: min_frequency must be >= 1");
}

// ---------------------------------------------------------------------------
// train: learn merges from `corpus`. Resets first so repeated calls are
// idempotent rather than cumulative.
// ---------------------------------------------------------------------------
void BPETokenizer::train(const std::string& corpus) {
    merges_.clear();
    merge_keys_.clear();
    rank_of_.clear();

    std::vector<std::string> words;
    pretokenize_into(corpus, &words);
    if (words.empty()) return;

    // A training symbol. `marked` is NOT derivable from `id` -- see the file
    // header. The FINAL symbol of every word is marked.
    struct Sym {
        int id;
        bool marked;
    };
    struct PairKey {
        int a; bool am;
        int b; bool bm;
        bool operator<(const PairKey& o) const {
            if (a != o.a) return a < o.a;
            if (am != o.am) return am < o.am;
            if (b != o.b) return b < o.b;
            return bm < o.bm;
        }
    };

    // Surface of a symbol.
    //
    // THE MARKER LIVES IN EXACTLY ONE PLACE, and this is the third attempt at
    // getting it right. A merge's stored key (merge_keys_[i]) ALREADY ends in
    // '</w>' when it absorbed one, so appending the marker again here produced
    // "the</w></w>" and encode() could never match it:
    //     [TRAIN] key=he</w>  ->  [TRAIN] key=the</w></w>   <-- doubled
    //     [ENC] miss cand=the</w>
    // The rule is therefore: a MERGE id's surface is its stored key verbatim;
    // only a LEAF (id < 256) is a byte plus a possibly-appended marker. The
    // oracle gets this for free because it stores the marker inside the symbol
    // tuple -- appending to a stored string double-counts.
    auto surface = [&](const Sym& s) -> std::string {
        if (s.id >= kByteBase)
            return merge_keys_[static_cast<size_t>(s.id - kByteBase)];
        std::string base(1, static_cast<char>(s.id));
        if (s.marked) base += kMarker;
        return base;
    };

    std::vector<std::vector<Sym>> seqs;
    seqs.reserve(words.size());
    for (const std::string& w : words) {
        std::vector<Sym> s;
        s.reserve(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            bool last = (i + 1 == w.size());
            s.push_back(Sym{static_cast<unsigned char>(w[i]), last});
        }
        seqs.push_back(s);
    }

    for (size_t round = 0; round < max_merges_; ++round) {
        // 1. count adjacent pairs
        std::map<PairKey, int> counts;
        std::map<PairKey, std::string> lsurf, rsurf;
        for (const std::vector<Sym>& s : seqs) {
            for (size_t i = 0; i + 1 < s.size(); ++i) {
                PairKey k{s[i].id, s[i].marked, s[i + 1].id, s[i + 1].marked};
                counts[k]++;
                if (counts[k] == 1) {
                    lsurf[k] = surface(s[i]);
                    rsurf[k] = surface(s[i + 1]);
                }
            }
        }
        if (counts.empty()) break;

        // 2. best pair under the TOTAL order (-count, left + SEP + right).
        //    The order must be TOTAL: candidates come from a std::map keyed by
        //    ids, which is NOT the oracle's insertion order, so an unspecified
        //    tie-break would ship a platform-dependent vocabulary.
        PairKey best{0, false, 0, false};
        int best_count = -1;
        std::string best_tie;
        bool have = false;
        for (const auto& kv : counts) {
            if (kv.second < static_cast<int>(min_frequency_)) continue;
            std::string tie = lsurf[kv.first] + kSep + rsurf[kv.first];
            if (!have || kv.second > best_count ||
                (kv.second == best_count && tie < best_tie)) {
                have = true;
                best = kv.first;
                best_count = kv.second;
                best_tie = tie;
            }
        }
        if (!have) break;

        // 3. record the merge and apply it NON-OVERLAPPING, LEFT-TO-RIGHT.
        //    The merged symbol inherits the RIGHT operand's marker flag: its
        //    surface is the concatenation and the marker sits at the end.
        Sym merged{kByteBase + static_cast<int>(merges_.size()), best.bm};
        std::string key = surface(Sym{best.a, best.am}) + surface(Sym{best.b, best.bm});

        merge_keys_.push_back(key);
        merges_.push_back(strip_marker(key));  // public form: what it decodes to
        rank_of_[key] = static_cast<int>(merges_.size()) - 1;

        for (std::vector<Sym>& s : seqs) {
            std::vector<Sym> out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                if (i + 1 < s.size() &&
                    s[i].id == best.a && s[i].marked == best.am &&
                    s[i + 1].id == best.b && s[i + 1].marked == best.bm) {
                    out.push_back(merged);
                    i += 2;
                } else {
                    out.push_back(s[i]);
                    i += 1;
                }
            }
            s.swap(out);
        }
    }
}

// ---------------------------------------------------------------------------
// encode: canonicalise -> pre-tokenize -> per-word greedy merge by rank ->
// join words with boundary_id().
// ---------------------------------------------------------------------------
std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<std::string> words;
    pretokenize_into(text, &words);

    std::vector<int> out;
    const int bnd = boundary_id();

    for (size_t wi = 0; wi < words.size(); ++wi) {
        if (wi) out.push_back(bnd);

        const std::string& w = words[wi];
        // Symbol surfaces with the marker on the FINAL byte, so a merge can
        // never span a word boundary. These strings live in the SAME key space
        // as merge_keys_, which is what rank_of_ is indexed by.
        std::vector<std::string> syms;
        syms.reserve(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            std::string s(1, w[i]);
            if (i + 1 == w.size()) s += kMarker;
            syms.push_back(s);
        }

        // Greedy lowest-rank-first until no learned merge applies.
        for (;;) {
            int best_rank = -1;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                std::string cand = syms[i] + syms[i + 1];
                auto it = rank_of_.find(cand);
                if (it == rank_of_.end()) continue;
                if (best_rank < 0 || it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_rank < 0) break;
            std::vector<std::string> next;
            next.reserve(syms.size());
            size_t i = 0;
            while (i < syms.size()) {
                if (i + 1 < syms.size() && i == best_i) {
                    next.push_back(syms[i] + syms[i + 1]);
                    i += 2;
                } else {
                    next.push_back(syms[i]);
                    i += 1;
                }
            }
            syms.swap(next);
        }

        // Emit ids: a 1-byte symbol is a raw byte; anything else is a learned
        // merge looked up by its (marker-bearing) key.
        for (const std::string& s : syms) {
            if (s.size() == 1) {
                out.push_back(static_cast<unsigned char>(s[0]));
                continue;
            }
            auto it = rank_of_.find(s);
            if (it != rank_of_.end()) {
                out.push_back(kByteBase + it->second);
            } else {
                // A leaf carrying only the trailing marker: strip it, emit the
                // raw byte. (A multi-byte leaf is impossible -- a leaf is one
                // byte plus at most the marker.)
                std::string bare = strip_marker(s);
                if (bare.size() == 1) {
                    out.push_back(static_cast<unsigned char>(bare[0]));
                } else {
                    for (char c : bare)
                        out.push_back(static_cast<unsigned char>(c));
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// decode: exact inverse over the id stream. Words separated by boundary_id()
// are joined with a single space. Unknown ids are skipped, not fatal.
// ---------------------------------------------------------------------------
std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    const int bnd = boundary_id();
    auto surface_of = [&](int id) -> std::string {
        if (id >= kByteBase && id < bnd)
            return merges_[static_cast<size_t>(id - kByteBase)];
        if (id >= 0 && id < kByteBase)
            return std::string(1, static_cast<char>(id));
        return std::string();  // unknown id -> skipped
    };

    std::vector<std::string> words_out;
    std::string cur;
    for (int id : ids) {
        if (id == bnd) {
            words_out.push_back(cur);
            cur.clear();
        } else {
            cur += surface_of(id);
        }
    }
    if (!cur.empty()) words_out.push_back(cur);

    std::string out;
    for (size_t i = 0; i < words_out.size(); ++i) {
        if (i) out.push_back(' ');
        out += strip_marker(words_out[i]);
    }
    return out;
}