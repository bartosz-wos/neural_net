#!/usr/bin/env bash
# =============================================================================
# tools/ngram_mutation.sh -- mutation harness for utils/ngram_lm.
#
# This repo has recorded TWENTY-ONE instances of a mutation harness producing
# a WRONG verdict, several of which reported one harness defect as N mutation
# verdicts. Every defence below was learned from one of those; the comments
# say which, so a future session does not have to rediscover them.
#
# THE DEFENCES, AND WHERE EACH CAME FROM:
#  1. Link set from `make print-lib-objs`, NEVER a regex over the Makefile.
#     `LIB_OBJS` is a RECURSIVE make variable (`:=` over a `$(LIB_SRCS:...)`
#     substitution reference), so its literal text contains no `.o` filenames.
#     A regex yields ZERO objects and every mutation becomes a false verdict.
#  2. The mutated object is rebuilt IN PLACE, never compiled separately and
#     linked alongside the original -- that double-defines every symbol and
#     reports "CAUGHT (compile error)" for all of them. A LINK ERROR IS NOT A
#     MUTATION VERDICT.
#  3. `rm -f` the test object and the mutated object before every build.
#  4. Verify each edit APPLIED by checksum, else MUT-NOOP. A sed that does
#     not match is a silent no-op, and "the suite does not cover this" is a
#     lie. The edit is applied with a PYTHON str.replace, never perl s///:
#     the anchors are full of `/` and perl read the first one as the
#     delimiter and died (six of fourteen mutations never applied).
#  5. Snapshot and restore BOTH the .cpp and the header.
#  6. Parse the summary as `N passed, M failed` and GUARD against a
#     malformed count reaching a `[ ... -gt 0 ]` test.
#  7. Never run `make` concurrently with this harness: it races on build/*.o
#     and produces phantom MUT-BROKEN verdicts.
#  8. Rebuild and re-verify the REAL binary on exit. The harness restores the
#     source but leaves build/ holding the last mutant.
#  9. Mutations must compile WARNING-FREE. A warning is a BUILD verdict, not
#     a mutation verdict.
# 10. Spec delimiter 0x1f, not `|`: `|` collides with the `||` in every C++
#     guard, splitting the anchor mid-expression and CORRUPTING the file.
# 11. `restore()` must NEVER delete the backup -- cleanup happens exactly
#     once, on EXIT. A restore that cleans up after itself works ONCE, and
#     the tree is then left holding the last mutant.
# =============================================================================
set -u

SRC=include/nn/utils/ngram_lm.cpp
HDR=include/nn/utils/ngram_lm.h
TEST=tests/test_ngram_lm.cpp
OBJ=build/utils/ngram_lm.o
TOBJ=build/test_ngram_lm.o
BIN=build/test_ngram_lm
DELIM=$'\x1f'

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

[ -f "$SRC" ] && [ -f "$TEST" ] || { red "missing sources"; exit 1; }

# --- defence 7: refuse to run concurrently with a build ---------------------
if pgrep -f "make.*test_ngram_lm" >/dev/null 2>&1; then
    red "ABORT: a make is running; it races on build/*.o (defence 7)."; exit 1
fi

# --- defence 1+2: the link set ---------------------------------------------
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep -v '^$')
LIB_COUNT=$(printf '%s\n' "$LIB_LINK" | wc -l)
if [ "$LIB_COUNT" -lt 10 ]; then
    red "ABORT: link set has $LIB_COUNT objects (need >= 10). Defence 1."
    exit 1
fi
echo "link set: $LIB_COUNT objects"

if [ ! -f "$OBJ" ]; then
    red "ABORT: $OBJ absent; the harness would link a stale object. Defence 2."
    exit 1
fi
if ! make -s print-lib-objs | grep -qx "$OBJ"; then
    red "ABORT: $OBJ is not in the link set at all (defence 1/2)."
    exit 1
fi

# --- snapshot BOTH files (defence 5), restored ATOMICALLY ------------------
# The restore must NEVER delete the backup: it is called once PER MUTATION
# and on EXIT, so a self-cleaning restore works exactly once and leaks the
# last mutant into the tree.
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
restore() {
    cp "$BAK_SRC" "$SRC.tmp" && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK_SRC" "$BAK_HDR"; }
trap cleanup EXIT INT TERM

# --- pristine check: refuse to run unless the suite is green ---------------
rm -f "$TOBJ" "$OBJ"
make -s build/test_ngram_lm >/dev/null 2>&1
PRISTINE=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/\1 \2/p')
read -r PP PF <<<"$PRISTINE"
if [ "${PF:-1}" != "0" ]; then
    red "ABORT: pristine suite is not green ($PRISTINE). Refusing to score mutations."
    restore; exit 1
fi
green "pristine suite green: $PP passed, 0 failed"

# --- apply a spec: <file><DELIM><anchor><DELIM><replacement> --------------
apply_spec() {
    local file="$1" anchor="$2" repl="$3"
    python3 - "$file" "$anchor" "$repl" <<'PY'
import sys
path, anchor, repl = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
if src.count(anchor) != 1:
    print(f"ANCHOR-COUNT:{src.count(anchor)}")
    sys.exit(3)
open(path, "w").write(src.replace(anchor, repl))
PY
}

verify_changed() {  # $1 = checksum before, $2 = file
    [ "$(md5sum "$2" | cut -d' ' -f1)" != "$1" ]
}

run_one() {   # id  anchor  replacement  [file]
    local id="$1" anchor="$2" repl="$3" file="${4:-$SRC}"
    local before after out nf

    restore
    before=$(md5sum "$file" | cut -d' ' -f1)
    if ! apply_spec "$file" "$anchor" "$repl"; then
        printf '  %-6s MUT-NOOP   (anchor not unique in %s)\n' "$id" "$file"
        return
    fi
    after=$(md5sum "$file" | cut -d' ' -f1)
    if [ "$before" = "$after" ]; then
        printf '  %-6s MUT-NOOP   (edit did not apply)\n' "$id"
        return
    fi

    rm -f "$TOBJ" "$OBJ"            # defence 3
    if ! make -s build/test_ngram_lm >/tmp/ngram_build.log 2>&1; then
        if grep -qE 'warning:' /tmp/ngram_build.log; then
            printf '  %-6s MUT-BROKEN (warning -- a BUILD verdict, not a mutation one)\n' "$id"
        else
            printf '  %-6s MUT-BROKEN (build failed)\n' "$id"
        fi
        return
    fi
    if [ ! -x "$BIN" ]; then
        printf '  %-6s UNAVAILABLE (no binary produced)\n' "$id"   # never a verdict
        return
    fi

    out=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/x \1 \2/p')
    if [ -z "$out" ]; then
        printf '  %-6s MUT-CRASH  (no summary line)\n' "$id"
        return
    fi
    set -- $out
    local np=$2 nf=$3
    if ! printf '%s' "$nf" | grep -qE '^[0-9]+$'; then
        red "ABORT: unparsable failure count '$nf' (defence 6)"; exit 1
    fi
    if [ "$nf" -gt 0 ]; then
        printf '  %-6s CAUGHT     (%s failed)\n' "$id" "$nf"
    else
        printf '  %-6s SURVIVED   <-- investigate\n' "$id"
    fi
}

echo
echo "=== mutation run ==="

# ---- M01: the KN unigram uses the RAW count, not the continuation count ----
# The defining property of the Kneser-Ney unigram. A suite that only checks
# a value would catch this, but a suite that checks a PROPERTY (the
# continuation count is not the raw count) is what makes it non-vacuous.
run_one M01 \
  '    return static_cast<double>(continuation_count(w)) /
           static_cast<double>(continuation_total_);' \
  '    return static_cast<double>(unigram_count(w)) /
           static_cast<double>(num_tokens_);'

# ---- M02: the backoff weight uses c(h) instead of V_h ----------------------
# The single most confusable quantity in the whole file. c(the) = 6 and
# V_the = 4 on the fixture, so the two give visibly different numbers.
run_one M02 \
  '    const double v_h = static_cast<double>(ctx_types_of(context));
    const double d = discount_;
    const double ch = static_cast<double>(c_h);

    // (ii) The discount is ABSOLUTE and the term is CLAMPED AT ZERO.
    const double first = std::max(c_hw - d, 0.0) / ch;
    // (iii) The backoff weight uses V_h, the number of DISTINCT FOLLOWERS.
    const double lambda = d * v_h / ch;' \
  '    const double v_h = static_cast<double>(ctx_count_of(context));
    const double d = discount_;
    const double ch = static_cast<double>(c_h);

    // (ii) The discount is ABSOLUTE and the term is CLAMPED AT ZERO.
    const double first = std::max(c_hw - d, 0.0) / ch;
    // (iii) The backoff weight uses V_h, the number of DISTINCT FOLLOWERS.
    const double lambda = d * v_h / ch;'

# ---- M03: the discount term is not clamped at zero -------------------------
# Without the clamp, an unseen n-gram gets a NEGATIVE first term.
run_one M03 \
  '    const double first = std::max(c_hw - d, 0.0) / ch;' \
  '    const double first = (c_hw - d) / ch;'

# ---- M04: the discount is RELATIVE, not absolute ---------------------------
run_one M04 \
  '    const double first = std::max(c_hw - d, 0.0) / ch;' \
  '    const double first = c_hw * (1.0 - d) / ch;'

# ---- M05: the backoff takes the PREFIX of the context, not the suffix -----
# h = (h1, h2) -> h' = (h1) instead of (h2). A language model's context is
# a SUFFIX of the history.
run_one M05 \
  '    std::vector<int> shorter(context.begin() + 1, context.end());
    const double lower = shorter.empty() ? kn_unigram(w)
                                         : kn_order(order - 1, w, shorter);' \
  '    std::vector<int> shorter(context.begin(), context.end() - 1);
    const double lower = shorter.empty() ? kn_unigram(w)
                                         : kn_order(order - 1, w, shorter);'

# ---- M06: the Witten-Bell reserved mass is not renormalized by Z_h --------
# Without the Z_h division the distribution does not sum to one, and the
# normalization test at EVERY context catches it.
run_one M06 \
  '    return (t_h / (n_h + t_h)) * lower / z;' \
  '    return (t_h / (n_h + t_h)) * lower;'

# ---- M07: the Witten-Bell seen term drops the T_h ------------------------
# p(w|h) = c(h,w)/(N_h + T_h) becomes c(h,w)/N_h.
run_one M07 \
  '        return c_hw / (n_h + t_h);' \
  '        return c_hw / n_h;'

# ---- M08: the absolute-discounting interpolation weight is inverted -------
run_one M08 \
  '    return (1.0 - lambda) * c_hw / ch + lambda * lower;' \
  '    return lambda * c_hw / ch + (1.0 - lambda) * lower;'

# ---- M09: the AD unigram uses the continuation count ----------------------
# The AD base case is the RAW MLE. Swapping it for the KN base case makes
# the two schemes agree, and the "the three schemes disagree" test fires.
run_one M09 \
  '    if (num_tokens_ == 0) return 0.0;
    return static_cast<double>(unigram_count(w)) /
           static_cast<double>(num_tokens_);' \
  '    if (num_tokens_ == 0) return 0.0;
    return static_cast<double>(continuation_count(w)) /
           static_cast<double>(continuation_total_);'

# ---- M10: the Witten-Bell unigram reserves no OOV mass -------------------
# The reserved mass T/(N+T) for the unseen class is what makes the WB
# distribution sum to one over (vocabulary + OOV).
run_one M10 \
  '    const double c = static_cast<double>(unigram_count(w));
    if (c > 0.0) return c / (n + t);
    return t / (n + t);' \
  '    const double c = static_cast<double>(unigram_count(w));
    if (c > 0.0) return c / (n + t);
    return 0.0;'

# ---- M11: the OOV class is dropped from the unseen set --------------------
# The reserved mass is then spread over fewer tokens, and the distribution
# over (vocabulary + OOV) no longer sums to one.
run_one M11 \
  '    unseen.push_back(kOovClass);' \
  '    // the OOV class is deliberately not added here'

# ---- M12: the growth curve records only the NEW types ---------------------
# The curve then has no plateaus and the Heaps fit is a different object.
run_one M12 \
  '            growth_n_.push_back(static_cast<double>(num_tokens_));
            growth_v_.push_back(static_cast<double>(unigram_.size()));' \
  '            if (unigram_[w] == 1) {
                growth_n_.push_back(static_cast<double>(num_tokens_));
                growth_v_.push_back(static_cast<double>(unigram_.size()));
            }'

# ---- M13: the Heaps fit drops the beta*mx term of the intercept -----------
# THE FIRST DRAFT OF THIS MUTATION WAS A HARNESS DEFECT, AND THE KIND THAT
# PRODUCES A FALSE VERDICT: it wrote `if (false) log_k = 0.0;`, but `log_k`
# is declared `const`, so the build FAILED with "assignment of read-only
# variable" and the harness reported MUT-BROKEN. A BUILD ERROR IS NOT A
# MUTATION VERDICT -- it says nothing about whether the suite covers the
# behaviour. The mutation now changes the FITTED INTERCEPT instead of
# trying to write to a const.
run_one M13 \
  '    const double beta = num / den;
    const double log_k = my - beta * mx;' \
  '    const double beta = num / den;
    const double log_k = my;'

# ---- M13b: the Heaps fit uses a hard-coded K = 1 --------------------------
# A second, INDEPENDENT probe of the same behaviour that does not touch a
# const: the fitted K is multiplied by 2. The synthetic power law must then
# recover K = 2 rather than 1.
run_one M13b \
  '    fit.beta = beta;
    fit.k = std::exp(log_k);' \
  '    fit.beta = beta;
    fit.k = std::exp(log_k) * 2.0;'

# ---- M14: the Good-Turing discounted count loses the (r+1) factor --------
run_one M14 \
  '    return static_cast<double>(r + 1) * static_cast<double>(nr1) /
           static_cast<double>(nr);' \
  '    return static_cast<double>(nr1) / static_cast<double>(nr);'

# ---- M15: the Good-Turing unseen mass P_0 is inverted --------------------
run_one M15 \
  '    return static_cast<double>(n1) / static_cast<double>(num_tokens_);' \
  '    return static_cast<double>(num_tokens_ - n1) / static_cast<double>(num_tokens_);'

# ---- M16: the discount bound is not enforced -----------------------------
# D > 1 breaks the Kneser-Ney normalization, and the constructor is the
# only thing standing between a caller and a silently broken model.
run_one M16 \
  '    if (!std::isfinite(discount) || discount <= 0.0 || discount > 1.0) {
        throw std::invalid_argument("ngram: discount must be finite and in (0, 1]");
    }' \
  '    if (!std::isfinite(discount)) {
        throw std::invalid_argument("ngram: discount must be finite and in (0, 1]");
    }'

# ---- M17: retraining ACCUMULATES instead of replacing --------------------
run_one M17 \
  '    unigram_.clear();
    bigram_.clear();
    trigram_.clear();' \
  '    // the count tables are deliberately NOT cleared'

# ---- defence 8: restore, rebuild, and re-verify the REAL binary ----------
restore
rm -f "$TOBJ" "$OBJ"
make -s build/test_ngram_lm >/dev/null 2>&1
FINAL=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/\1 \2/p')
read -r FP FF <<<"$FINAL"
if [ "${FF:-1}" = "0" ]; then
    green "post-harness rebuild verified: $FP passed, 0 failed"
else
    red "POST-HARNESS REBUILD FAILED: $FINAL -- the tree is not what it was"
fi
