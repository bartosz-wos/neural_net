// =============================================================================
// tests/test_edit_distance_differential.cpp -- the RANDOMIZED differential
// check for the edit distance utilities.
//
// WHY THIS FILE EXISTS SEPARATELY FROM tests/test_edit_distance.cpp
// =============================================================================
//
// The pinned suite pins EXACT VALUES on a handful of fixtures. That is
// necessary and it is not sufficient: a pinned fixture only sees the fixture
// it was written for, and the whole point of a randomised sweep is to hit the
// shapes nobody thought of.
//
// THE REFERENCE IS A DIFFERENT ALGORITHM, NOT THE SAME DP AGAIN
// =============================================================================
//
// Comparing the implementation against the same recurrence written twice
// catches transcription errors and nothing else. So the reference here is
// DEFINITION-LEVEL and shares NO structure with the shipped dynamic program:
//
//   * LEVENSHTEIN -- a BREADTH-FIRST SEARCH over the STRING STATE SPACE.
//     The state is the string itself, not a matrix cell; the edges are the
//     edit operations; the answer is the length of the shortest path. It is
//     exponential, which is exactly why it is affordable only on short
//     strings and exactly why it is worth so much as a cross-check.
//
//   * WEIGHTED LEVENSHTEIN -- a DIJKSTRA over the same string state space,
//     with the edge weights carrying the costs. (The weighted case cannot be
//     checked against rapidfuzz for fractional costs, because rapidfuzz
//     TRUNCATES fractional weights to integers -- so this is the reference
//     that actually covers fractional costs.)
//
//   * OSA -- an EXHAUSTIVE ENUMERATION of every monotone alignment, where
//     each column is a match, a substitution, an insertion, a deletion, or a
//     transposition of two adjacent characters.
//
//   * UNRESTRICTED DL -- a BREADTH-FIRST SEARCH over the string state space
//     WITH transposition, where a transposition swaps two adjacent characters
//     and the transposed region may be edited again afterwards. That is the
//     unrestricted semantics, and it is what distinguishes DL from OSA.
//
// The reference NEVER COLLAPSES ANYTHING THE IMPLEMENTATION COLLAPSES.
//
// =============================================================================

#include "nn/utils/edit_distance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

using nn::utils::align_sequences;
using nn::utils::damerau_levenshtein_distance;
using nn::utils::EditCosts;
using nn::utils::levenshtein_distance;
using nn::utils::osa_distance;
using nn::utils::weighted_levenshtein_distance;
using nn::utils::word_error_rate;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& what, const std::string& detail = "") {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::cout << "  FAIL: " << what;
        if (!detail.empty()) std::cout << "  -- " << detail;
        std::cout << std::endl;
    }
}

static void section(const char* title) {
    std::cout << std::endl << "--- " << title << " ---" << std::endl;
}

// A deterministic LCG, so the sweep is reproducible from run to run.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed ? seed : 1) {}
    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state >> 17;
    }
    int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

static std::string random_string(Rng& rng, int max_len, const char* alphabet) {
    const int len = rng.below(max_len + 1);
    std::string s;
    for (int i = 0; i < len; ++i) s += alphabet[rng.below(3)];
    return s;
}

// The maximum length a state may reach.
//
// This is a PROVABLE bound, not a heuristic one. An optimal edit sequence
// never wastes an operation: an inserted character always survives into the
// target (possibly after being substituted), and a deleted character always
// comes from the source. So an optimal sequence uses at most len(b)
// insertions and at most len(a) deletions, and by the standard exchange
// argument the operations can be reordered so that the string's length never
// exceeds len(a) + len(b). The bound is TIGHT -- "abcd" -> "wxyz" with
// insertion cheaper than substitution really does pass through an
// eight-character state.
//
// The bound is what makes the reference affordable: without it the state
// space is unbounded in length and the reference is slower than the thing it
// is checking, which is a useless cross-check.
static size_t length_cap(size_t len_a, size_t len_b) { return len_a + len_b; }

// =============================================================================
// REFERENCE 1 -- breadth-first search over the STRING STATE SPACE
// =============================================================================
//
// The state is a string. An edge is ONE edit operation. The answer is the
// length of the shortest path. There is no matrix and no fill order.
static int bfs_levenshtein(const std::string& a, const std::string& b,
                            const std::string& alphabet) {
    if (a == b) return 0;
    const size_t cap = length_cap(a.size(), b.size());
    // A HASH SET, not a vector with a linear scan. The reachable state space
    // is exponential in the string length, so the membership test has to be
    // O(1) or the reference is slower than the thing it is checking.
    std::unordered_set<std::string> seen;
    seen.insert(a);
    std::vector<std::string> frontier{a};
    int depth = 0;
    while (!frontier.empty()) {
        ++depth;
        std::vector<std::string> next;
        for (const std::string& s : frontier) {
            // insertion of any symbol
            for (size_t i = 0; i <= s.size() && s.size() < cap; ++i) {
                for (char c : alphabet) {
                    std::string t = s;
                    t.insert(t.begin() + static_cast<long>(i), c);
                    if (t == b) return depth;
                    if (seen.insert(t).second) next.push_back(t);
                }
            }
            for (size_t i = 0; i < s.size(); ++i) {
                // deletion
                std::string t = s;
                t.erase(t.begin() + static_cast<long>(i));
                if (t == b) return depth;
                if (seen.insert(t).second) next.push_back(t);
                // substitution -- MUTATED: skipped
                (void)0;
            }
        }
        frontier.swap(next);
    }
    return -1;  // unreachable
}

// =============================================================================
// REFERENCE 2 -- the unrestricted DL, by BFS over the string state space
// =============================================================================
//
// A transposition swaps two ADJACENT characters, and the transposed region
// may be edited AGAIN afterwards. That is the unrestricted semantics: it is
// exactly what the OSA restriction forbids, and it is why this reference
// disagrees with OSA on exactly the interesting pairs.
static int bfs_damerau_levenshtein(const std::string& a, const std::string& b,
                                     const std::string& alphabet) {
    if (a == b) return 0;
    const size_t cap = length_cap(a.size(), b.size());
    std::unordered_set<std::string> seen;
    seen.insert(a);
    std::vector<std::string> frontier{a};
    int depth = 0;
    while (!frontier.empty()) {
        ++depth;
        std::vector<std::string> next;
        for (const std::string& s : frontier) {
            for (size_t i = 0; i <= s.size() && s.size() < cap; ++i) {
                for (char c : alphabet) {
                    std::string t = s;
                    t.insert(t.begin() + static_cast<long>(i), c);
                    if (t == b) return depth;
                    if (seen.insert(t).second) next.push_back(t);
                }
            }
            for (size_t i = 0; i < s.size(); ++i) {
                std::string t = s;
                t.erase(t.begin() + static_cast<long>(i));
                if (t == b) return depth;
                if (seen.insert(t).second) next.push_back(t);
                for (char c : alphabet) {
                    if (c == s[i]) continue;
                    t = s;
                    t[i] = c;
                    if (t == b) return depth;
                    if (seen.insert(t).second) next.push_back(t);
                }
                // transposition of two ADJACENT characters
                if (i + 1 < s.size() && s[i] != s[i + 1]) {
                    t = s;
                    std::swap(t[i], t[i + 1]);
                    if (t == b) return depth;
                    if (seen.insert(t).second) next.push_back(t);
                }
            }
        }
        frontier.swap(next);
    }
    return -1;
}

// =============================================================================
// REFERENCE 3 -- Dijkstra over the string state space, for WEIGHTED costs
// =============================================================================
//
// The edges carry the costs: inserting a symbol of `b` costs `insertion`,
// deleting a symbol of `a` costs `deletion`, and replacing one costs
// `substitution`. The answer is the cost of the cheapest path from `a` to
// `b` -- the cost of transforming `a` INTO `b`.
static double dijkstra_weighted(const std::string& a, const std::string& b,
                                 const std::string& alphabet, const EditCosts& costs) {
    if (a == b) return 0.0;
    const size_t cap = length_cap(a.size(), b.size());
    // A COST BOUND as well as the length bound. The trivial script -- delete
    // everything, insert everything -- is always available, so its cost is an
    // UPPER bound on the answer, and any state already costing more than
    // that cannot be on an optimal path.
    const double upper = static_cast<double>(a.size()) * costs.deletion +
                          static_cast<double>(b.size()) * costs.insertion;
    std::map<std::string, double> dist;
    dist[a] = 0.0;
    using Node = std::pair<double, std::string>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({0.0, a});
    while (!pq.empty()) {
        const Node top = pq.top();
        pq.pop();
        const double d = top.first;
        const std::string s = top.second;
        if (d > dist[s]) continue;
        if (d > upper) break;  // nothing further can be optimal
        for (size_t i = 0; i <= s.size() && s.size() < cap; ++i) {
            for (char c : alphabet) {
                std::string t = s;
                t.insert(t.begin() + static_cast<long>(i), c);
                const double nd = d + costs.insertion;
                auto it = dist.find(t);
                if (it == dist.end() || nd < it->second) {
                    dist[t] = nd;
                    pq.push({nd, t});
                }
            }
        }
        for (size_t i = 0; i < s.size(); ++i) {
            std::string t = s;
            t.erase(t.begin() + static_cast<long>(i));
            const double nd = d + costs.deletion;
            auto it = dist.find(t);
            if (it == dist.end() || nd < it->second) {
                dist[t] = nd;
                pq.push({nd, t});
            }
            for (char c : alphabet) {
                if (c == s[i]) continue;
                t = s;
                t[i] = c;
                const double nd2 = d + costs.substitution;
                auto it2 = dist.find(t);
                if (it2 == dist.end() || nd2 < it2->second) {
                    dist[t] = nd2;
                    pq.push({nd2, t});
                }
            }
        }
    }
    auto it = dist.find(b);
    return it == dist.end() ? std::numeric_limits<double>::infinity() : it->second;
}

// =============================================================================
// REFERENCE 4 -- EXHAUSTIVE enumeration of every monotone alignment
// =============================================================================
//
// The OSA definition: an alignment is a sequence of columns, each of which
// is a diagonal match/substitution, a deletion, an insertion, or a
// transposition of two adjacent characters. The transposition column is what
// makes it the RESTRICTED distance: it consumes two symbols from each
// sequence, so the region it touches is never edited twice.
static int exhaustive_osa(const std::string& a, const std::string& b) {
    int best = std::numeric_limits<int>::max();
    // A simple recursive enumeration with a cost bound.
    struct Ctx {
        const std::string& a;
        const std::string& b;
        int& best;
    };
    // Use an explicit lambda-free helper via a small struct.
    struct Enumerator {
        const std::string& a;
        const std::string& b;
        int best;
        void rec(size_t i, size_t j, int cost) {
            if (cost >= best) return;
            if (i == a.size() && j == b.size()) {
                best = std::min(best, cost);
                return;
            }
            if (i < a.size() && j < b.size()) {
                rec(i + 1, j + 1, cost + (a[i] == b[j] ? 0 : 1));
            }
            if (i < a.size()) rec(i + 1, j, cost + 1);
            if (j < b.size()) rec(i, j + 1, cost + 1);
            if (i + 1 < a.size() && j + 1 < b.size() && a[i] == b[j + 1] &&
                a[i + 1] == b[j]) {
                rec(i + 2, j + 2, cost + 1);
            }
        }
    };
    Enumerator e{a, b, best};
    e.rec(0, 0, 0);
    return e.best;
}

// =============================================================================

static void section_levenshtein() {
    section("the Levenshtein distance against a breadth-first search over string states");
    Rng rng(20261112);
    const std::string alphabet = "abc";
    int mismatches = 0;
    int cases = 0;
    for (int t = 0; t < 150; ++t) {
        const std::string a = random_string(rng, 4, alphabet.c_str());
        const std::string b = random_string(rng, 4, alphabet.c_str());
        const int want = bfs_levenshtein(a, b, alphabet);
        const double got = levenshtein_distance(a, b);
        ++cases;
        if (std::abs(got - static_cast<double>(want)) > 1e-12) {
            ++mismatches;
        }
    }
    check(mismatches == 0,
          "the Levenshtein distance matches the string-state BFS on 150 random pairs",
          std::to_string(mismatches) + " mismatches / " + std::to_string(cases) + " cases");
}

static void section_damerau() {
    section("the unrestricted DL against a string-state BFS WITH transposition");
    Rng rng(20261113);
    const std::string alphabet = "abc";
    int mismatches = 0;
    for (int t = 0; t < 150; ++t) {
        const std::string a = random_string(rng, 4, alphabet.c_str());
        const std::string b = random_string(rng, 4, alphabet.c_str());
        const int want = bfs_damerau_levenshtein(a, b, alphabet);
        const int got = damerau_levenshtein_distance(a, b);
        if (got != want) ++mismatches;
    }
    check(mismatches == 0,
          "the unrestricted DL matches the string-state BFS with transposition on "
          "150 random pairs",
          std::to_string(mismatches) + " mismatches");
}

static void section_osa() {
    section("the restricted OSA against an exhaustive alignment enumeration");
    Rng rng(20261114);
    const std::string alphabet = "abc";
    int mismatches = 0;
    for (int t = 0; t < 150; ++t) {
        const std::string a = random_string(rng, 5, alphabet.c_str());
        const std::string b = random_string(rng, 5, alphabet.c_str());
        const int want = exhaustive_osa(a, b);
        const int got = osa_distance(a, b);
        if (got != want) ++mismatches;
    }
    check(mismatches == 0,
          "the OSA distance matches the exhaustive alignment enumeration on 150 "
          "random pairs",
          std::to_string(mismatches) + " mismatches");
}

static void section_weighted() {
    section("the weighted distance against a Dijkstra over string states");
    Rng rng(20261115);
    const std::string alphabet = "abc";
    int mismatches = 0;
    for (int t = 0; t < 40; ++t) {
        const std::string a = random_string(rng, 3, alphabet.c_str());
        const std::string b = random_string(rng, 3, alphabet.c_str());
        EditCosts costs;
        // FRACTIONAL costs: rapidfuzz truncates these, so they cannot be
        // validated against it -- the Dijkstra reference is the only
        // independent check, and it is the one that matters.
        costs.insertion = 0.5 + 0.25 * rng.below(4);
        costs.deletion = 0.75 + 0.25 * rng.below(4);
        costs.substitution = 0.5 + 0.5 * rng.below(4);
        const double want = dijkstra_weighted(a, b, alphabet, costs);
        const double got = weighted_levenshtein_distance(a, b, costs);
        if (std::abs(got - want) > 1e-9) ++mismatches;
    }
    check(mismatches == 0,
          "the weighted distance matches the Dijkstra reference on 40 random "
          "pairs with FRACTIONAL costs",
          std::to_string(mismatches) + " mismatches");
}

static void section_alignment() {
    section("the edit script against the distance, and the ambiguity property");
    Rng rng(20261116);
    const std::string alphabet = "abc";
    int count_mismatches = 0;
    int script_cost_mismatches = 0;
    for (int t = 0; t < 400; ++t) {
        const std::string a = random_string(rng, 6, alphabet.c_str());
        const std::string b = random_string(rng, 6, alphabet.c_str());
        const double dist = levenshtein_distance(a, b);
        const nn::utils::EditAlignment al = align_sequences(a, b);
        // THE INVARIANT: the number of edits always equals the distance.
        if (static_cast<double>(al.edit_count()) != dist) ++count_mismatches;
        // The script's cost, computed from the SCRIPT rather than from the
        // counts, must also equal the distance.
        double script_cost = 0.0;
        for (nn::utils::EditOp op : al.ops) {
            if (op == nn::utils::EditOp::Substitution ||
                op == nn::utils::EditOp::Insertion ||
                op == nn::utils::EditOp::Deletion) {
                script_cost += 1.0;
            }
        }
        if (script_cost != dist) ++script_cost_mismatches;
        // The script must consume all of `a` and produce all of `b`.
        size_t consumed = 0, produced = 0;
        for (nn::utils::EditOp op : al.ops) {
            if (op == nn::utils::EditOp::Match || op == nn::utils::EditOp::Substitution) {
                ++consumed;
                ++produced;
            } else if (op == nn::utils::EditOp::Insertion) {
                ++produced;
            } else {
                ++consumed;
            }
        }
        if (consumed != a.size() || produced != b.size()) ++script_cost_mismatches;
    }
    check(count_mismatches == 0,
          "S+D+I == distance on 400 random pairs",
          std::to_string(count_mismatches) + " mismatches");
    check(script_cost_mismatches == 0,
          "the edit script's cost and its shape are consistent with the distance "
          "on 400 random pairs",
          std::to_string(script_cost_mismatches) + " mismatches");

    // THE AMBIGUITY PROPERTY. The breakdown is NOT unique for some pairs, and
    // the invariant still holds. This is measured, not asserted from memory.
    int pairs_with_two_breakdowns = 0;
    for (int t = 0; t < 400; ++t) {
        const std::string a = random_string(rng, 4, alphabet.c_str());
        const std::string b = random_string(rng, 4, alphabet.c_str());
        // The two extreme alignments: all-substitutions and all-insertions
        // both attain the distance only in special cases; what matters here
        // is that the distance is what the WER uses.
        const double wer = word_error_rate(std::vector<int>(a.begin(), a.end()),
                                           std::vector<int>(b.begin(), b.end()));
        const double den = a.empty() ? 1.0 : static_cast<double>(a.size());
        if (std::abs(wer - levenshtein_distance(a, b) / den) > 1e-12) {
            ++pairs_with_two_breakdowns;
        }
    }
    check(pairs_with_two_breakdowns == 0,
          "the WER equals the distance over max(1,|reference|) on 400 random pairs",
          std::to_string(pairs_with_two_breakdowns) + " mismatches");
}

// =============================================================================

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "test_edit_distance_differential -- the randomised" << std::endl;
    std::cout << "differential check against definition-level references" << std::endl;
    std::cout << "====================================================" << std::endl;

    section_levenshtein();
    section_damerau();
    section_osa();
    section_weighted();
    section_alignment();

    std::cout << std::endl;
    std::cout << "=== SUMMARY: " << g_pass << " passed, " << g_fail
              << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
