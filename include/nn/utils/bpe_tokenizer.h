#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// BPETokenizer -- byte-level byte-pair-encoding (BPE) subword tokenizer.
//
// Reference: Sennrich, Haddow & Birch, "Neural Machine Translation of Rare
// Words with Subword Units", ACL 2016 (Volume 1: Long Papers), pp. 1715-1725.
// ACL Anthology P16-1162, doi 10.18653/v1/P16-1162; arXiv:1508.07909.
//
// Every numeric behaviour here is pinned by tests/bpe_oracle.py, a numpy oracle
// written and run BEFORE this file existed. Every expected id in
// tests/test_bpe_tokenizer.cpp is copied from that oracle's output. If you
// change a semantic below, change the oracle first and re-copy -- do not
// hand-derive an expected value in the test.
//
// ---------------------------------------------------------------------------
// WHY A BYTE-LEVEL ALPHABET (and why that matters for this repo)
//
// Ids 0..255 are the 256 raw bytes, NOT a unicode alphabet. That makes the id
// stream EXACTLY invertible for arbitrary text:
//     encode(decode(ids)) == ids
// with no tolerance, no replacement characters, no lossy paths. It also lines
// up with the byte-native generative layers this repo already ships
// (layers/generative/pixelcnn.h, layers/recurrent/mambabyte.h), which index
// byte values directly.
//
// Consequence: there is NO unknown/unk token. With a complete byte alphabet
// every input string is representable, so OOV is impossible by construction
// and a <unk> id would be dead weight. Rare words decompose into subwords
// instead of collapsing to a single unknown symbol -- which is the paper's
// whole point.
//
// ---------------------------------------------------------------------------
// ID LAYOUT
//
//   ids 0..255   the 256 raw bytes
//   ids 256..    learned merges, in LEARNED RANK ORDER: merge rank i -> id 256+i
//   boundary_id  = vocab_size() = 256 + num_merges. A reserved, first-class
//                 token separating words in the id stream.
//
// The BOUNDARY ID IS NOT OPTIONAL, and this was measured rather than assumed.
// The paper's '</w>' marker alone is insufficient: it rides on the FINAL byte
// of a word, so if that final symbol never gets merged it stays a leaf and
// encode() emits the bare byte -- the boundary is gone. Measured on the oracle:
//     encode("a b c") with the marker alone -> [97, 98, 99] -> decodes to "abc"
// It is not only single-character words: ANY word whose last symbol survives as
// a leaf loses its boundary. So the marker is a TRAINING-TIME construct that
// stops a merge spanning a word boundary, and the emitted stream carries an
// explicit boundary id instead.
//
// ---------------------------------------------------------------------------
// THE ROUNDTRIP CONTRACT
//
//     decode(encode(t)) == canonicalize(t)
//
// NOT decode(encode(t)) == t. encode canonicalises whitespace: any run of
// whitespace collapses to a single ' ' and leading/trailing whitespace is
// stripped. (This matches GPT-2 BPE's documented behaviour.) A caller wanting
// exact preservation must not expect it; a test asserting == t on arbitrary
// text is a test bug, not an implementation bug.
//
// Note decode() emits single-space-separated words -- it does NOT restore the
// original whitespace. That asymmetry is the price of canonicalisation.
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT HERE
//
//   * No regex pre-tokenizer (GPT-2's `pat`). Plain whitespace splitting, which
//     is the paper's own pre-tokenization. Punctuation is NOT split off; it is
//     just part of the word.
//   * No dropout. That is a training-time BPE regulariser, not a property of
//     the tokenizer.
//   * No serialization / load-from-disk.
//   * No unicode normalisation. The alphabet is bytes, so this is automatic.
//
// ---------------------------------------------------------------------------
// ALGORITHMIC NOTE -- SCAN ORDER IS PART OF THE SEMANTICS
//
// Merges within a word are applied NON-OVERLAPPING and LEFT-TO-RIGHT. On
// "aaaa" with pair (a,a):
//     left-to-right  -> ['aa','a','a</w>']  then ['aa','aa</w>']
//     right-to-left  -> index 2 taken first -> ['a','aa','a</w>'], and index 0
//                        no longer matches, so one merge is LOST
// The two scan orders pick different alignments, which changes the pair counts
// and therefore every subsequent merge. Any implementation of this class must
// scan left-to-right; "scan the way that keeps my indices valid" is a different
// algorithm, not an optimisation. (Cost me a full debugging round in the
// oracle -- see tests/bpe_oracle.py defect O7.)
// ===========================================================================
class BPETokenizer {
public:
    // kByteBase is the size of the raw-byte prefix of the vocabulary.
    static const int kByteBase = 256;

    // The '</w>' end-of-word suffix, TRAINING-TIME ONLY (see the note above).
    // It stops a merge spanning two words during BPE training; it is stripped
    // from the emitted id stream and replaced by boundary_id().
    static const char* kWordEndMarker() { return "</w>"; }

    explicit BPETokenizer(int max_merges = 512);
    BPETokenizer(int min_frequency, int max_merges);

    // Learns at most max_merges merges from `corpus`. A pair is only merged if
    // it occurs at least `min_frequency_` times (default 2). Resets any
    // previously learned merges first, so train() is idempotent.
    //
    // TIE-BREAK IS A TOTAL ORDER, and that is not a detail: the candidate pairs
    // are held in a std::unordered_map, whose iteration order is unspecified,
    // while the oracle iterates an insertion-ordered dict. Without an explicit
    // total order the two disagree and the shipped vocabulary becomes
    // platform-dependent. The key is (-count, left_surface + '\x1f' +
    // right_surface) ascending; '\x1f' is a control character that cannot occur
    // in a surface.
    void train(const std::string& corpus);

    // Canonicalises whitespace (runs collapse to one ' ', ends stripped),
    // pre-tokenizes on single spaces, encodes each word with the learned merges
    // in greedy lowest-rank order, and joins the words with boundary_id().
    // No trailing boundary id.
    std::vector<int> encode(const std::string& text) const;

    // Exact inverse of encode over the id stream. Unknown ids (>= vocab_size()
    // and != boundary_id()) are skipped rather than throwing, so a partially
    // decoded stream still yields whatever was recoverable.
    std::string decode(const std::vector<int>& ids) const;

    // The learned merges in learned-rank order: merges()[i] is the surface of
    // the symbol with id 256+i. Inspectable for tests and for debugging.
    const std::vector<std::string>& merges() const { return merges_; }

    size_t num_merges() const { return merges_.size(); }
    size_t max_merges() const { return max_merges_; }
    size_t min_frequency() const { return min_frequency_; }

    // 256 raw bytes + num_merges() merges. Note the boundary id is a VALID id
    // (ids are 0 .. vocab_size() INCLUSIVE), it is simply not a merge.
    int vocab_size() const { return kByteBase + static_cast<int>(merges_.size()); }
    int boundary_id() const { return vocab_size(); }

private:
    // Surfaces are stored in TWO forms and conflating them is a real bug:
    //
    //   merge_keys_[i]  the TRAINING surface INCLUDING the '</w>' marker
    //                   (e.g. "he</w>"). This is the key encode() looks up,
    //                   because during encoding the final byte of a word still
    //                   carries the marker and a merge must match it exactly.
    //   merges_[i]      the PUBLIC surface with the marker STRIPPED
    //                   (e.g. "he"). This is what the symbol DECODES to, so it
    //                   is what merges() reports and what a caller printing the
    //                   vocabulary expects to see.
    //
    // I first stored only the marker-bearing form and encode() silently failed
    // to apply a single merge while merges() still looked correct -- training
    // proved right, inference proved wrong. The two forms are now distinct
    // members and both are exercised by tests.
    std::vector<std::string> merges_;               // public, marker-stripped
    std::vector<std::string> merge_keys_;           // training, marker-bearing
    std::unordered_map<std::string, int> rank_of_;  // merge key -> rank
    size_t max_merges_;
    size_t min_frequency_;
};

// The canonicalisation encode() applies, exposed so a caller (or a test) can
// state the roundtrip contract explicitly:
//     decode(encode(t)) == bpe_canonicalize(t)
std::string bpe_canonicalize(const std::string& text);

#endif // BPE_TOKENIZER_H