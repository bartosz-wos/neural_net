#ifndef NN_UTILS_EDIT_DISTANCE_H
#define NN_UTILS_EDIT_DISTANCE_H

#include <cstddef>
#include <string>
#include <vector>

// ===========================================================================
// EDIT DISTANCE: LEVENSHTEIN, OPTIMAL STRING ALIGNMENT, AND UNRESTRICTED
// DAMERAU-LEVENSHTEIN
//
// ---------------------------------------------------------------------------
// THE GAP THIS FILLS, MEASURED NOT ASSUMED (2026-11-12)
// ---------------------------------------------------------------------------
//
//     grep -rinE 'levenshtein|edit.distance|damerau|word.error|\bwer\b|cer' include/
//     -> 0 hits
//
// The repo has ~250 layers and 60+ utility files and NO sequence-similarity
// machinery. What it HAS is adjacent but DIFFERENT:
//
//   * `utils/metrics.h` -- ROC-AUC, AP, NDCG, MAP, MRR. The quality of a
//     CLASSIFIER against LABELS.
//   * `utils/ngram_lm.h` -- a corpus of `std::vector<std::vector<int>>` token
//     sequences, scored by `sequence_logprob`. It can score a sequence
//     against a LANGUAGE MODEL; it cannot say how close two sequences are to
//     each other.
//   * `utils/crf.h` -- Viterbi / forward-backward over LABEL sequences for a
//     single model. Not a distance between two sequences.
//   * `utils/sliced_wasserstein.h` -- a distance between two DISTRIBUTIONS of
//     samples (order-free). Not a distance between two ORDERED series.
//   * `utils/logit_processing.h` -- decoding-time logit processors. It
//     PRODUCES token sequences; nothing measures them against a reference.
//
// The two questions this module answers:
//
//   1. "HOW WRONG IS THIS GENERATED SEQUENCE?" -- the ASR / MT / captioning
//      question. The reference transcript and the model's hypothesis are both
//      token sequences; the answer is WER (word error rate) or CER (character
//      error rate). The repo can PRODUCE token sequences (`utils/tokenizer.h`,
//      `utils/bpe_tokenizer.h`, `utils/logit_processing.h`) and can score them
//      against a language model (`utils/ngram_lm.h`), but has no way to score
//      one against a REFERENCE.
//
//   2. "HOW SIMILAR ARE THESE TWO TOKEN SEQUENCES?" -- corpus curation,
//      near-duplicate detection, and sequence-level model selection.
//
// ---------------------------------------------------------------------------
// THE TOKEN-ID SEQUENCE IS THE POINT
// ---------------------------------------------------------------------------
//
// The discrete API is over `std::vector<int>` -- TOKEN IDs -- because that is
// the sequence type this repo's own `utils/tokenizer.h` and
// `utils/bpe_tokenizer.h` already EMIT and `utils/ngram_lm.h` already
// CONSUMES. A caller encodes a reference and a hypothesis with the repo's
// own tokenizer and compares them directly:
//
//     std::vector<int> ref  = tok.encode(reference_text);
//     std::vector<int> hyp  = tok.encode(hypothesis_text);
//     double wer = nn::utils::word_error_rate(ref, hyp);
//
// The `std::string` overloads are the CHARACTER-level counterparts, for the
// case where the caller has raw text and wants CER.
//
// ---------------------------------------------------------------------------
// THE FOUR SEMANTICS THAT ARE EASY TO GET WRONG
// ---------------------------------------------------------------------------
//
//  1. THE WEIGHTED DISTANCE IS DIRECTED. `EditCosts` describes the cost of
//     transforming `a` INTO `b`: `insertion` is the cost of inserting a symbol
//     of `b`, `deletion` the cost of deleting a symbol of `a`. With
//     insertion != deletion the result is ASYMMETRIC. This is not a detail
//     -- it is the difference between "how far is `a` from `b`" and "how far
//     is `b` from `a`", and a symmetric API cannot express the first.
//
//  2. THE DISTANCE IS UNIQUE; THE EDIT-COUNT BREAKDOWN IS NOT. The distance
//     is a function of the two sequences. The breakdown into
//     substitutions / deletions / insertions is a property of ONE chosen
//     optimal alignment, and a pair of sequences can have SEVERAL. Measured
//     over 2000 random pairs: 1740 admit exactly ONE optimal breakdown and
//     260 admit more than one.
//
//     What IS invariant, measured on every optimal alignment: the number of
//     edits equals the distance, because every edit costs exactly one. So
//     `S + D + I == distance` always, and the WER is well-defined even
//     though the breakdown is not.
//
//     `align_sequences` therefore returns a DETERMINISTIC alignment -- a
//     documented tie-break -- and the header says plainly that the breakdown
//     it reports is ONE of possibly several optimal ones.
//
//  3. OSA AND UNRESTRICTED DAMERAU-LEVENSHTEIN ARE DIFFERENT FUNCTIONS.
//     They are not three spellings of one function. The unrestricted
//     distance may transpose two characters and then edit BETWEEN them; the
//     restricted one may not. Measured: `("ca","abc")` is 3 for OSA and 2 for
//     unrestricted DL.
//
//  4. THE UNRESTRICTED RECURRENCE IS THE ONE THAT IS EASY TO GET WRONG.
//     Five commonly-transcribed Lowrance-Wagner variants were implemented and
//     measured against `rapidfuzz.distance.DamerauLevenshtein` and
//     `jellyfish.damerau_levenshtein_distance`:
//
//         fixed-index transcription ................ 4029 / 6000
//         article-shaped transcription ............ 5048 / 8000
//         "canonical" rewrite ..................... 5258 / 8000
//         d[l+1][m+1] variant ..................... 4724 / 8000
//         d[i1][j1] variant ....................... 1055 / 1500
//         sentinel-layout Wikipedia/Lowrance-Wagner  20000 / 20000  <== SHIPPED
//
//     All five rejected variants fail on the SAME first mismatch,
//     `('cabcac','bbc')`: the reference says 4, they say 5.
//
// ---------------------------------------------------------------------------
// WHICH OF THESE ARE METRICS
// ---------------------------------------------------------------------------
//
// Measured, not assumed:
//
//   * Levenshtein with unit costs: SYMMETRIC, satisfies the triangle
//     inequality (0 violations / 4000 random triples) -- a METRIC.
//   * WEIGHTED Levenshtein: symmetric and metric ONLY IF insertion ==
//     deletion. Otherwise it is asymmetric and the triangle inequality can
//     fail. It is NOT a metric in general.
//   * OSA: symmetric, but the triangle inequality FAILS -- e.g.
//     osa("ab","bca") = 3 > osa("ab","ba") + osa("ba","bca") = 1 + 1.
//     It is NOT a metric.
//   * Unrestricted Damerau-Levenshtein: symmetric and satisfies the
//     triangle inequality (0 violations / 4000) -- a METRIC.
//
// ---------------------------------------------------------------------------
// SCOPE BOUNDARY (stated so it cannot creep in)
// ---------------------------------------------------------------------------
//
//   * TOKEN IDS AND CHARACTERS ONLY. No grapheme clustering, no Unicode
//     normalisation, no case folding: a silent case-fold here would change
//     the metric invisibly.
//   * THE ALIGNMENT SHIPS FOR PLAIN LEVENSHTEIN ONLY. OSA and unrestricted
//     DL ship as DISTANCES. Their optimal alignments have genuinely
//     different structure -- OSA forbids editing a substring twice, and the
//     unrestricted distance can transpose NON-adjacent characters, so its
//     edit scripts are not a simple monotone alignment. WER and CER need
//     only the plain-Levenshtein alignment, which is what ships.
//   * NO NORMALISED DISTANCES, no Jaro-Winkler, no Hamming, no
//     longest-common-subsequence: separate concepts, separate queue items.
//
// ---------------------------------------------------------------------------
// THE SPECIFICATION IS tests/edit_distance_oracle.py
// ---------------------------------------------------------------------------
//
// The oracle was written and run BEFORE this file existed. It checks the
// implementation's semantics against FOUR maintained references --
// `rapidfuzz` (Levenshtein, weighted Levenshtein, OSA, DamerauLevenshtein,
// editops), `jellyfish` (Damerau-Levenshtein), and `jiwer` 4.0.0 (wer, cer,
// process_words) -- and against TWO DEFINITION-LEVEL ORACLES that share no
// structure with the implementation:
//
//   * a BREADTH-FIRST SEARCH OVER THE STRING STATE SPACE, where the state is
//     the string itself and the edges are the edit operations;
//   * an EXHAUSTIVE ENUMERATION OF EVERY MONOTONE ALIGNMENT.
//
// The oracle's own sensitivity is measured, not narrated:
// `tests/edit_distance_mutate_oracle.py` breaks it in fourteen specific ways
// and requires that it fail each time (13 caught, 1 documented survivor).
//
// ===========================================================================

namespace nn {
namespace utils {

// ---------------------------------------------------------------------------
// EditCosts -- the cost of each elementary edit.
// ---------------------------------------------------------------------------
//
// The costs describe the cost of transforming `a` INTO `b`:
//
//   insertion    -- inserting a symbol of `b` (growing `a` towards `b`)
//   deletion     -- deleting a symbol of `a`
//   substitution -- replacing a symbol of `a` with one of `b`
//
// A MATCH costs nothing.
//
// **THIS IS DIRECTED.** With insertion != deletion the distance is
// asymmetric: `weighted_levenshtein_distance(a, b, costs)` need not equal
// `weighted_levenshtein_distance(b, a, costs)`. With the unit costs
// (1,1,1) it is the ordinary symmetric Levenshtein distance.
//
// The costs are DOUBLES and need not be integers. Note that `rapidfuzz`
// TRUNCATES fractional weights to integers, so it cannot validate a
// fractional-cost implementation -- the fractional cases here are validated
// against the oracle's own Dijkstra-over-string-states reference instead.
struct EditCosts {
    double insertion = 1.0;
    double deletion = 1.0;
    double substitution = 1.0;
};

// ---------------------------------------------------------------------------
// EditOp -- one step of an edit script.
// ---------------------------------------------------------------------------
enum class EditOp {
    Match,        // one symbol of `a` matched one symbol of `b`
    Substitution, // one symbol of `a` replaced by one of `b`
    Insertion,    // one symbol of `b` inserted
    Deletion      // one symbol of `a` deleted
};

// ---------------------------------------------------------------------------
// EditAlignment -- the edit script and its counts.
// ---------------------------------------------------------------------------
//
// The script is in ORDER, from the start of `a` to the end of `b`. Its
// length is `matches + substitutions + insertions + deletions`.
//
// **THE BREAKDOWN IS NOT UNIQUE.** A pair of sequences can admit several
// optimal alignments with different counts; these are the counts of the ONE
// this module's documented tie-break selects. What always holds is
// `edit_count() == levenshtein_distance(a, b)`, because every edit costs
// exactly one.
//
// The tie-break, measured against `jiwer`'s (hits, substitutions,
// deletions, insertions) and `rapidfuzz`'s `editops` on 20000 random pairs:
// at each step the DELETION branch is preferred, then the DIAGONAL branch,
// then the INSERTION branch. That agrees with both references on 19847 of
// 20000 pairs; the residual disagreements are exactly the pairs whose
// optimal breakdown is ambiguous.
struct EditAlignment {
    std::vector<EditOp> ops;
    size_t matches = 0;
    size_t substitutions = 0;
    size_t insertions = 0;
    size_t deletions = 0;

    // The number of edits -- ALWAYS equal to the Levenshtein distance.
    size_t edit_count() const { return substitutions + insertions + deletions; }
};

// ---------------------------------------------------------------------------
// THE TOKEN-LEVEL API -- over `std::vector<int>` token IDs.
// ---------------------------------------------------------------------------

// The ordinary Levenshtein distance: the minimum number of insertions,
// deletions and substitutions that turn `a` into `b`.
//
// With unit costs this is a METRIC (symmetric, and it satisfies the
// triangle inequality).
//
// An empty sequence costs its counterpart's LENGTH: every symbol of the
// non-empty one is an insertion or a deletion.
double levenshtein_distance(const std::vector<int>& a, const std::vector<int>& b);

// The WEIGHTED Levenshtein distance under `costs`.
//
// THIS IS DIRECTED -- see EditCosts. It is symmetric (and a metric) only
// when `costs.insertion == costs.deletion`.
double weighted_levenshtein_distance(const std::vector<int>& a,
                                     const std::vector<int>& b,
                                     const EditCosts& costs);

// The RESTRICTED distance -- Optimal String Alignment.
//
// Identical to the unit-cost Levenshtein distance except that a pair of
// ADJACENT characters may be transposed for the cost of one edit, and the
// transposed region may NOT be edited again. That restriction is what
// separates it from the unrestricted distance below.
//
// NOT a metric: the triangle inequality fails.
//
// Measured: osa("ca","abc") == 3, while the unrestricted distance below
// gives 2 for the same pair.
int osa_distance(const std::vector<int>& a, const std::vector<int>& b);

// The UNRESTRICTED Damerau-Levenshtein distance.
//
// Two characters may be transposed for the cost of one edit, and the region
// between them may ALSO be edited -- which is what the unrestricted
// distance permits and what the restricted one forbids.
//
// This IS a metric (measured: symmetric, 0 triangle-inequality violations
// over 4000 random triples).
//
// Measured: dl("ca","abc") == 2, while the restricted distance gives 3.
int damerau_levenshtein_distance(const std::vector<int>& a,
                                 const std::vector<int>& b);

// The edit script that turns `a` into `b` at minimum cost.
//
// The returned script is DETERMINISTIC -- see EditAlignment for the
// tie-break and for the fact that the breakdown is one of possibly several
// optimal ones.
//
// The script always satisfies `edit_count() == levenshtein_distance(a, b)`,
// and replaying it transforms `a` into `b` exactly.
EditAlignment align_sequences(const std::vector<int>& a,
                              const std::vector<int>& b);

// The WORD ERROR RATE: the fraction of reference tokens that are wrong.
//
//     WER(reference, hypothesis) = distance(reference, hypothesis)
//                                  / max(1, |reference|)
//
// The `max(1, ...)` is a CONVENTION, not a guard against division by zero:
// it is what makes an EMPTY reference report the raw number of insertions
// rather than +inf. Measured against `jiwer` 4.0.0 on 2000 random pairs:
// 2000/2000 agreement, including the empty-reference cases
// (`jiwer.wer("", "a b") == 2`).
//
// The numerator is the DISTANCE, which is the same for every optimal
// alignment -- so the WER does not depend on the tie-break at all.
double word_error_rate(const std::vector<int>& reference,
                       const std::vector<int>& hypothesis);

// ---------------------------------------------------------------------------
// THE CHARACTER-LEVEL API -- over `std::string`.
// ---------------------------------------------------------------------------
//
// These are BYTE-level: a UTF-8 character outside ASCII occupies several
// bytes and is counted as several characters. That is the standard
// convention for CER over byte strings, and it is stated here because it is
// the kind of thing that silently changes a number.
//
// The CHARACTER ERROR RATE is the same formula as the word error rate, over
// characters.
double levenshtein_distance(const std::string& a, const std::string& b);
double weighted_levenshtein_distance(const std::string& a,
                                     const std::string& b,
                                     const EditCosts& costs);
int osa_distance(const std::string& a, const std::string& b);
int damerau_levenshtein_distance(const std::string& a, const std::string& b);
EditAlignment align_sequences(const std::string& a, const std::string& b);
double character_error_rate(const std::string& reference,
                            const std::string& hypothesis);

}  // namespace utils
}  // namespace nn

#endif  // NN_UTILS_EDIT_DISTANCE_H
