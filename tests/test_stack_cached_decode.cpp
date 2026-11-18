// Focused regression test for the STACK-LEVEL CACHED DECODE.
//
// THE GAP (measured 2026-11-17, not assumed):
//
//     grep -rinE 'stack.?level|stack_decode|decode_stack|forward_cached' include/
//     -> MultiHeadAttention::forward_cached and TransformerBlock::forward_cached
//        exist; NO stack drives a sequence of blocks through ONE cache.
//
// The repo shipped a KV cache whose own header says it exists "so that ONE
// cache can serve a whole stack of attention layers", and a block-level
// cached decode proven for exactly ONE block. What was never shipped -- and
// never proven -- is N blocks sharing ONE cache.
//
// ---------------------------------------------------------------------------
// WHAT THIS TEST PINS, AND WHY EACH CHECK IS THERE
// ---------------------------------------------------------------------------
//
//  1. The stack's ORDINARY forward matches the independent numpy oracle.
//  2. The stack's CACHED decode matches the oracle.
//
//  3. THE CACHE-SLOT ALIASING FOIL. A stack that gave every block the same
//     cache slot would not throw and would not crash: it would silently
//     attend to the FIRST block's keys and values from EVERY block. The
//     oracle emits the answer such a stack produces (ORACLE_ALIASED_OUT),
//     and this test asserts the real stack's answer DIFFERS from it. This
//     is the check the whole feature exists for, and the block-level
//     cached-decode test could not make it: that path only ever had one
//     block, so "one slot" and "one slot per block" were the same thing.
//
//  4. The per-slot bookkeeping: every block's slot holds every token, and
//     the slots hold DIFFERENT keys (block 1's cached keys are not block
//     0's).
//
//  5. The decode is an INFERENCE path: backward() after a cached forward
//     throws.
//
//  6. The ordinary path is still trainable: the stack's backward chains
//     the blocks and returns a gradient.
//
//  7. reset_cache() empties the slots and the stack is reusable.
//
//  8. The constructor rejects a degenerate configuration, and every error
//     message is DISTINCT -- a substring that appears in two messages
//     discriminates nothing.
//
// The fixture is EMITTED by tests/stack_cached_decode_oracle.py, never
// hand-transcribed: see tests/splice_stack_cached_decode_fixture.py.

#include "nn/layers/architectures/transformer_stack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static int passed = 0, failed = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { ++passed; std::printf("  [PASS] %s\n", name.c_str()); }
    else      { ++failed; std::printf("  [FAIL] %s\n", name.c_str()); }
}

static bool threw_with(const std::function<void()>& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        const std::string what = e.what();
        return what.find(needle) != std::string::npos;
    }
    return false;
}

// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
// ==== ORACLE-EMITTED FIXTURE (tests/stack_cached_decode_oracle.py) ====
const size_t ORACLE_D_MODEL = 4;
const size_t ORACLE_HEADS = 2;
const size_t ORACLE_NUM_BLOCKS = 3;
const size_t ORACLE_TOKENS = 5;
const double ORACLE_INPUT[20] = {
    0.69492758811179056, -0.26085045408347313, 0.305439334735784, -0.012149959254014447, -0.19468008383615051, -0.17194800707249813,
    0.22129945791142444, 0.17367464141198993, 0.062159707572185655, -0.066413371109117089, 0.14052504603449872, 0.090243400449818448,
    -0.18196322073808505, 0.05852193656863533, 0.72600066624526582, -0.05162975559931824, -0.18892921935245649, -0.52917164336938904,
    -0.26134307502990672, -0.37850709833197088,
};
const double ORACLE_WQ[48] = {
    0.5703048433342468, -0.45773148729083446, 0.39662665971203104, -0.19776777349533173, -0.2299862191196162, 0.39325579833798424,
    0.12911507264501451, -0.10565907743073256, 0.85588281569753644, -1.2459234602488169, -0.19568048074124347, -0.24174411458284301,
    0.80476295839608181, -0.25308108787458461, 0.17422732571478497, -0.39218353478534851, 0.25890061199093495, 0.7262041184209137,
    -1.0679702003207698, -0.014097341619443517, 0.11384883858827854, 0.16942939728973722, -0.78493459808475208, -0.093196894102341069,
    -0.13729289915677739, 0.12423860808966099, -0.66169774977785945, -0.4890517230359907, 1.2843773968348589, 0.46252092113674126,
    0.14671971919411703, 0.39173962127040141, -0.23055966784432042, -0.51857959297425837, 0.28635778094100378, -0.061101019078852674,
    0.44418914978576729, -0.045754376266314158, 0.43800258777861456, -0.033725312793147375, -0.12520910168828078, 0.53019803037152291,
    -0.67617548877637945, -0.098647590950104738, -0.085259043303487866, 0.29848609993054531, 0.33961168672805087, 0.040058787505963575,
};
const double ORACLE_WK[48] = {
    0.88869555706155845, 0.16146208330887701, -1.3315036225269741, 0.10501901180113964, -0.80682685037481661, 0.80051951388243048,
    0.035208278167916815, -0.28854047088638018, -0.21593627406606561, -0.48610750210247078, 0.47989911775915856, -0.61561787283354197,
    -0.056610003840051486, -0.80876409302166519, -0.047918139782620328, -0.013214812098632022, -0.40792030693605835, 0.28157915469419259,
    0.52180090420998704, 0.21123654651015064, 0.48275274421156961, 0.12647670515038351, -0.0016907937080943613, 0.17960507095450085,
    -0.50655257579758284, -0.9883571367609465, -0.70443623833102542, -0.54555270434531467, 0.19158484070877688, 0.064398098604675297,
    -0.61508800898468385, 0.39555805875246697, 0.1346469679460241, 0.65480521707116979, 0.41699205560148189, 0.64632252224907016,
    0.53320124532206148, -0.012305630064687772, -0.53248422880233259, -0.11224920197607968, -0.42709602218766224, -0.14328136126251892,
    0.38967330953245971, -0.67303153992502152, -0.71966546168532042, -0.22880140042718439, -0.64451096188628398, 0.072718779012242066,
};
const double ORACLE_WV[48] = {
    -0.0062981686947601485, -0.0454205229025232, 0.66148772431349145, 0.11214754328145136, 0.14386737011278089, 0.40014371296232398,
    0.76122990665485979, -0.18375509566243692, -0.35692941378267207, 0.46288743304752028, -0.68209453462690928, 0.4669146825935454,
    0.68435572051020621, 0.41948448021130358, 0.64105957912570743, -1.0198438564116339, -0.32971523478468212, 1.0919628201743798,
    -0.032896148218718793, 0.15315482874639544, -0.33238380913955107, 0.080652505825542828, -0.33178713291577611, 0.59089868009615798,
    -0.033089386090011594, 0.42097591368602388, -0.16509679095157812, -0.30019791293247805, 0.022742358297021828, 1.0159797245646269,
    -0.63637930440224288, -0.2115079808039117, -0.081659332519642608, -0.66606864318777936, -0.41749602723788076, 0.047775953222254579,
    -0.94539920650481291, -0.01410762736467264, 0.31749795641096551, 0.77378789182404639, -0.24097238241654093, 0.46460001390244399,
    -0.13285271309794425, -0.27885542441292999, -0.50898165520991523, -0.10371606979960202, -0.13422031165682016, -0.796046866298732,
};
const double ORACLE_WO[48] = {
    -0.23130147733675041, -0.38657842999165221, 0.41135947113265858, -0.12752638375068412, 0.3812649850731254, -0.19643593030593207,
    0.74334174209099035, -0.67614460953937028, 0.69279027255274794, 0.19913499937266774, -0.13494463643132032, 0.20524167817234035,
    0.21501039886221238, 0.23141806993115013, 0.438734773812121, -0.21362412258792174, -0.075434659889206307, -0.015397102954672526,
    -0.20663521562568987, -0.62077800600940247, 1.3053687519206678, -0.13643558124301489, -0.56981785187989697, 0.26760783681355677,
    -0.32592685651392955, -0.32861868440906344, -0.16126769675907812, 0.049715392160188589, -0.52116847207996864, 0.21884499100732493,
    -0.93823383447550235, 0.4233766326345017, 1.0183306245059123, 0.41632622633493971, -0.37701127775136828, -0.94531931881809828,
    -0.15209941413782557, -0.27540977313132753, 0.52900636261091238, 0.018676255878711929, -0.30133441327494065, -0.5580569155184737,
    0.018835750858781107, -0.40273356146565287, 0.29069713235992328, 0.040584478608488649, -0.40438712604073707, -0.15974842097947761,
};
const double ORACLE_W1[48] = {
    -0.18611652192968539, -0.11968394736187214, 0.52445522021467061, -0.32916788200166741, 0.11165930643905833, -0.91962499834979117,
    -0.47188400472904973, -0.40786906953324042, 0.17414165545001506, 0.26183188295305532, -0.7616898991726998, 0.49459723601133837,
    1.1102171998762127, -0.42855770600266846, -0.76160550444127728, 0.9542020688669387, -0.24044536621442328, -0.15682628410506394,
    -0.69515420594120525, 0.022260212378954439, -0.42377685682825306, 0.91623718742058868, 0.44440085230767129, 0.52218968356272555,
    -0.99629186600642949, -0.21106293835956916, -0.48996034379284858, 0.99418206909692042, 0.20697529577436913, -0.31449993266124143,
    -0.13308838030790213, 0.098133221132688561, -0.28485759689150147, -0.1415053709996589, -1.0554078908991584, -0.3336858699047216,
    0.73730764518620107, 0.49887531259449175, 1.0220238987355816, -0.11530236888206118, -0.31807009047845536, 0.55317560707622204,
    0.62905525236493154, 0.12819306004853284, 1.2185590348240061, -0.88776502559638226, 0.4311212328322126, 0.40439206871627276,
};
const double ORACLE_W2[48] = {
    -0.57676685375067749, -0.36977581491495787, -0.44150152729401348, -0.25829816835993308, 0.31537586300103676, -0.17442382914539645,
    -0.5312105494658842, 0.48951738794881322, 1.1893224282918742, 0.34190656514128703, -1.1030541413417645, -0.35899594662041301,
    0.19336867358695828, 0.042154641139812446, -0.047317994618361517, 0.94381606941589857, -0.43848516587389752, -0.11420217881710791,
    -0.32813812984325225, -0.11776778655341201, -0.64262111647600795, -0.024899336861919773, 0.50988261827425152, -0.64050678591822119,
    -0.0104755251305564, -0.62396670547487654, -0.82537601840753716, 0.12752690991840862, 0.80924768913015266, -0.2255964882156965,
    -0.34135703198760298, 0.60479704159829328, -0.29689053143563199, 0.029002635358192693, -0.81084775853680924, 0.583356887277652,
    -0.24497874864169689, 0.57743278539276433, -1.0931977494669107, -0.30257589923099781, 0.6787551059525212, 0.17441361840133185,
    -0.33682711812459493, 0.83613398795013461, -0.43619200983608986, -0.062071788826957103, 0.16098694127235313, 0.30499967021111229,
};
const double ORACLE_B1[12] = {
    -0.012404961535222612, -0.0016358515188073922, -0.089474416369337423, 0.078026861828398617, -0.025524419821207619, -0.10979871338091934,
    0.048951353218239291, 0.10658984241830811, -0.13505658193890605, 0.12871035220892221, 0.043218515973955082, 0.21337682339827846,
};
const double ORACLE_B2[12] = {
    0.098126048681481226, 0.076314676129505099, -0.19736251531190074, 0.052409979341316748, 0.036993973063162711, 0.1546011611572162,
    -0.068435086489318717, -0.0038406171926481242, -0.061049106752535789, 0.17647873873155767, -0.062734606529197809, -0.1408432699462655,
};
const double ORACLE_LN1G[12] = {
    0.8736875310998049, 1.1299551440294826, 1.0379474210935491, 0.8888861998589459, 0.97758934960156896, 0.96065168031320836,
    0.80633677218861632, 1.100793327179111, 1.0147981327897253, 0.7838701754663786, 1.0962230412530507, 1.0582739391855287,
};
const double ORACLE_LN1B[12] = {
    -0.017434416746042358, 0.056272162857110612, 0.15471798263864145, -0.028658721787321102, 0.049896459658500258, -0.035722858421739744,
    0.032381912669966505, 0.11190702889025797, 0.071893168494296361, 0.080603833944565259, 0.088650999597478256, 0.034275190467160203,
};
const double ORACLE_LN2G[12] = {
    0.87559098869872987, 0.99887381642890372, 0.97229771697629508, 1.038159891278021, 0.83426809848655847, 0.9557885208224044,
    0.95409163068591241, 1.0712693937911391, 0.86881165307664765, 0.99355484042120412, 1.1515032226962574, 0.97559534479745891,
};
const double ORACLE_LN2B[12] = {
    -0.12275873488552476, -0.078601443114573954, -0.073048131773929575, -0.036790046681476084, 0.090818297739140838, 0.012610402332894436,
    0.070783264309607449, -0.027361650041585101, 0.12939771833893177, -0.11949453761828942, 0.043591129237200127, 0.19105387539658858,
};
const double ORACLE_FULL_OUT[20] = {
    1.3098327685960109, 0.21190827276161733, -1.5671286017338295, -0.095217927027171412, -0.27773647143967006, -0.097031075194970243,
    -1.2616901108162539, 1.7320522676542005, 0.070851083446686558, -0.85921698586151873, -0.92522961405434068, 1.8039676868873302,
    0.44592851095927477, -0.50332202166911788, 1.6262255210557603, -1.1283569148801433, 1.5054148180829747, -0.79585600795419709,
    0.16501946669467948, -0.79282933534903899,
};
const double ORACLE_CACHED_OUT[20] = {
    1.3098327685960105, 0.21190827276161733, -1.5671286017338295, -0.09521792702717119, -0.27773647143967012, -0.097031075194970312,
    -1.2616901108162539, 1.7320522676542005, 0.070851083446685573, -0.8592169858615174, -0.92522961405434101, 1.8039676868873302,
    0.44592851095927483, -0.50332202166911866, 1.6262255210557608, -1.1283569148801429, 1.5054148180829752, -0.79585600795419775,
    0.16501946669467954, -0.79282933534903854,
};
const double ORACLE_ALIASED_OUT[20] = {
    1.4900768188821907, -0.26027151377462215, -1.3595029682235686, -0.0098786799940682735, -0.36227285573847817, -0.15125919905187543,
    -1.1354706389855285, 1.773289005789576, -0.40482309129645311, -0.27416029707290135, -0.99207273952827169, 1.8202566257216861,
    -0.56647745123889992, -0.70931249209168989, -0.31460936884634427, 1.8550943767974841, -0.44436530746324898, -0.54513454206666068,
    -0.68898690948880559, 1.8739496808267329,
};
// >>> ORACLE-EMITTED FIXTURE END <<<

// The oracle works in the (tokens, d_model) layout; the C++'s stack takes
// (d_model, seq_len). The transpose happens here, once.
static Tensor oracle_input() {
    Tensor x(ORACLE_D_MODEL, ORACLE_TOKENS);
    for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
        for (size_t s = 0; s < ORACLE_TOKENS; ++s)
            x(f, s) = ORACLE_INPUT[s * ORACLE_D_MODEL + f];
    return x;
}

// The oracle's output is (tokens, d_model); the C++'s is (d_model, tokens).
static std::vector<double> stack_output_as_tokens(const Tensor& y) {
    std::vector<double> v;
    v.reserve(y.cols * y.rows);
    for (size_t s = 0; s < y.cols; ++s)
        for (size_t f = 0; f < y.rows; ++f)
            v.push_back(y(f, s));
    return v;
}

static double max_abs_diff(const std::vector<double>& a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n && i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Decode the whole sequence one token at a time.
//
// A decode that THROWS is a FAILURE, not a CRASH. The mutation harness's
// verdict classes treat a crash with nothing reported as MUT-CRASH -- an
// undecided verdict -- so a defect that makes the implementation throw must
// be REPORTED by the suite. (Three of the twelve source mutations make the
// stack throw: the aliased slots overflow the cache's capacity, the
// constructor that does not attach a cache leaves the blocks without one.)
static bool decode_all(TransformerStack& s, const Tensor& x,
                       std::vector<double>& out, std::string& err) {
    out.clear();
    try {
        for (size_t t = 0; t < ORACLE_TOKENS; ++t) {
            Tensor step(ORACLE_D_MODEL, 1);
            for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
                step(f, 0) = x(f, t);
            Tensor y = s.forward_cached(step);
            std::vector<double> v = stack_output_as_tokens(y);
            for (size_t f = 0; f < ORACLE_D_MODEL; ++f) out.push_back(v[f]);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

// Fill a stack with the oracle's weights.
static void load_weights(TransformerStack& s) {
    const size_t D = ORACLE_D_MODEL;
    for (size_t b = 0; b < ORACLE_NUM_BLOCKS; ++b) {
        TransformerBlock& blk = s.block(b);
        // The oracle's arrays are concatenated over the blocks, so block b's
        // slice starts at b*d*d (or b*d for a vector).
        const double* wq  = ORACLE_WQ  + b * D * D;
        const double* wk  = ORACLE_WK  + b * D * D;
        const double* wv  = ORACLE_WV  + b * D * D;
        const double* wo  = ORACLE_WO  + b * D * D;
        const double* w1  = ORACLE_W1  + b * D * D;
        const double* b1  = ORACLE_B1  + b * D;
        const double* w2  = ORACLE_W2  + b * D * D;
        const double* b2  = ORACLE_B2  + b * D;
        const double* g1  = ORACLE_LN1G + b * D;
        const double* bt1 = ORACLE_LN1B + b * D;
        const double* g2  = ORACLE_LN2G + b * D;
        const double* bt2 = ORACLE_LN2B + b * D;
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j < D; ++j) {
                blk.attn.W_q(i, j) = wq[i * D + j];
                blk.attn.W_k(i, j) = wk[i * D + j];
                blk.attn.W_v(i, j) = wv[i * D + j];
                blk.attn.W_o(i, j) = wo[i * D + j];
                blk.W1(i, j) = w1[i * D + j];
                blk.W2(i, j) = w2[i * D + j];
            }
        for (size_t j = 0; j < D; ++j) {
            blk.b1(0, j) = b1[j];
            blk.b2(0, j) = b2[j];
            blk.ln1.gamma(0, j) = g1[j];
            blk.ln1.beta(0, j) = bt1[j];
            blk.ln2.gamma(0, j) = g2[j];
            blk.ln2.beta(0, j) = bt2[j];
        }
    }
}

int main() {
    std::printf("=== TransformerStack cached decode regression ===\n");

    const Tensor x = oracle_input();

    // -----------------------------------------------------------------------
    // 0. Constructor validation. EVERY MESSAGE IS DISTINCT.
    // -----------------------------------------------------------------------
    check(threw_with([] { TransformerStack s(0, 2, 2); }, "d_model must be positive"),
          "the constructor rejects d_model = 0");
    check(threw_with([] { TransformerStack s(4, 0, 2); }, "num_heads must be positive"),
          "the constructor rejects num_heads = 0");
    check(threw_with([] { TransformerStack s(4, 2, 0); }, "num_layers must be positive"),
          "the constructor rejects num_layers = 0");
    check(threw_with([] { TransformerStack s(4, 2, 2, 0); }, "max_seq_len must be positive"),
          "the constructor rejects max_seq_len = 0");

    // -----------------------------------------------------------------------
    // 1. The ORDINARY forward matches the oracle.
    // -----------------------------------------------------------------------
    TransformerStack stack(ORACLE_D_MODEL, ORACLE_HEADS, ORACLE_NUM_BLOCKS,
                           ORACLE_TOKENS);
    load_weights(stack);

    {
        Tensor y = stack.forward(x);
        std::vector<double> got = stack_output_as_tokens(y);
        double d = max_abs_diff(got, ORACLE_FULL_OUT, ORACLE_TOKENS * ORACLE_D_MODEL);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "the ordinary stack forward matches the oracle (max abs diff = %.3e < 1e-9)", d);
        check(d < 1e-9, buf);
    }

    // -----------------------------------------------------------------------
    // 2. The CACHED decode matches the oracle, token by token.
    // -----------------------------------------------------------------------
    {
        stack.reset_cache();
        std::vector<double> got;
        std::string err;
        if (!decode_all(stack, x, got, err)) {
            check(false, "the token-at-a-time stack decode threw: " + err);
        } else {
            double d = max_abs_diff(got, ORACLE_CACHED_OUT, ORACLE_TOKENS * ORACLE_D_MODEL);
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "the token-at-a-time stack decode matches the oracle (max abs diff = %.3e < 1e-9)", d);
            check(d < 1e-9, buf);
        }
    }

    // -----------------------------------------------------------------------
    // 3. THE INVARIANT: the cached decode equals the full forward.
    // -----------------------------------------------------------------------
    {
        Tensor full = stack.forward(x);
        std::vector<double> full_tok = stack_output_as_tokens(full);

        stack.reset_cache();
        std::vector<double> got;
        std::string err;
        if (!decode_all(stack, x, got, err)) {
            check(false, "the cached stack decode threw: " + err);
        } else {
            double d = max_abs_diff(got, full_tok.data(), full_tok.size());
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "the cached stack decode equals the full stack forward (max abs diff = %.3e)", d);
            // The cached path performs the same arithmetic in the same order
            // and the tail is row-wise, so this is expected to be EXACTLY
            // zero.
            check(d == 0.0, buf);
        }
    }

    // -----------------------------------------------------------------------
    // 4. THE ALIASING FOIL: a stack that shared ONE slot would produce a
    //    different answer. The stack must NOT produce it.
    // -----------------------------------------------------------------------
    {
        stack.reset_cache();
        std::vector<double> got;
        std::string err;
        if (!decode_all(stack, x, got, err)) {
            // The aliased variant overflows the cache's capacity: three
            // blocks appending to one slot hold three times the tokens.
            // That is a DETECTION, not an accident -- report it.
            check(false, "the stack's decode threw (the aliased-slot foil "
                         "overflows the cache): " + err);
        } else {
            double d = max_abs_diff(got, ORACLE_ALIASED_OUT, ORACLE_TOKENS * ORACLE_D_MODEL);
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "the stack's answer DIFFERS from the aliased-slot foil "
                          "(max abs diff = %.3e > 1e-3)", d);
            check(d > 1e-3, buf);
        }
    }

    // -----------------------------------------------------------------------
    // 5. THE PER-SLOT BOOKKEEPING: one slot per block, each holding every
    //    token, and each holding DIFFERENT keys.
    // -----------------------------------------------------------------------
    {
        stack.reset_cache();
        {
            std::vector<double> got;
            std::string err;
            if (!decode_all(stack, x, got, err)) {
                check(false, "the per-slot bookkeeping decode threw: " + err);
                std::shared_ptr<nn::utils::KVCache> c0 = stack.kv_cache();
                (void)c0;
            }
        }
        std::shared_ptr<nn::utils::KVCache> c = stack.kv_cache();
        check(c != nullptr && c->layers() == (int)ORACLE_NUM_BLOCKS,
              "the stack reserved one cache slot per block");
        // `KVCache::size()` THROWS for a layer that was never reserved, so
        // the reservation is checked first -- otherwise a stack that failed
        // to reserve its slots would abort the suite instead of reporting a
        // failure.
        bool all_full = true;
        for (size_t b = 0; b < ORACLE_NUM_BLOCKS; ++b) {
            if (!c->has_layer((int)b) || c->size((int)b) != (int)ORACLE_TOKENS)
                all_full = false;
        }
        check(all_full, "every block's slot holds every decoded token");

        // The slots must hold DIFFERENT keys. If they did not, the blocks
        // would be sharing.
        double min_diff = 1e30;
        for (size_t b = 1; b < ORACLE_NUM_BLOCKS && c->has_layer(0) &&
                           c->has_layer((int)b); ++b) {
            const Tensor& k0 = c->key_buffer(0);
            const Tensor& kb = c->key_buffer((int)b);
            double d = 0.0;
            for (size_t i = 0; i < (size_t)c->size((int)b); ++i)
                for (size_t j = 0; j < ORACLE_D_MODEL; ++j)
                    d = std::max(d, std::fabs(k0(i, j) - kb(i, j)));
            min_diff = std::min(min_diff, d);
        }
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "each block's cached keys differ from block 0's (min diff = %.3e > 1e-6)",
                      min_diff);
        check(min_diff > 1e-6, buf);
    }

    // -----------------------------------------------------------------------
    // 6. The decode is an INFERENCE path: backward() after a cached forward
    //    throws rather than producing a wrong gradient.
    // -----------------------------------------------------------------------
    {
        stack.reset_cache();
        Tensor step(ORACLE_D_MODEL, 1);
        for (size_t f = 0; f < ORACLE_D_MODEL; ++f) step(f, 0) = x(f, 0);
        bool decoded = true;
        try {
            stack.forward_cached(step);
        } catch (const std::exception& e) {
            decoded = false;
            check(false, std::string("the single-step decode threw: ") + e.what());
        }
        if (decoded) {
            Tensor g(ORACLE_D_MODEL, 1);
            g.fill(0.0);
            check(threw_with([&] { stack.backward(g, 0.0); }, "CACHED"),
                  "backward() after a cached forward throws");
        }
    }

    // -----------------------------------------------------------------------
    // 7. The ORDINARY path is still trainable.
    // -----------------------------------------------------------------------
    {
        TransformerStack tr(ORACLE_D_MODEL, ORACLE_HEADS, ORACLE_NUM_BLOCKS,
                            ORACLE_TOKENS);
        load_weights(tr);
        Tensor y = tr.forward(x);
        Tensor g(ORACLE_D_MODEL, ORACLE_TOKENS);
        for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
            for (size_t s = 0; s < ORACLE_TOKENS; ++s)
                g(f, s) = 0.7 * y(f, s) - 0.3;
        tr.zero_grad();
        Tensor dx = tr.backward(g, 0.0);
        double gmax = 0.0;
        for (size_t i = 0; i < dx.rows; ++i)
            for (size_t j = 0; j < dx.cols; ++j)
                gmax = std::max(gmax, std::fabs(dx(i, j)));
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "the ordinary stack path is trainable (input grad max abs = %.3e > 0)", gmax);
        check(gmax > 0.0, buf);

        std::vector<Tensor*> p = tr.parameters();
        std::vector<Tensor*> gg = tr.gradients();
        char buf2[160];
        std::snprintf(buf2, sizeof(buf2),
                       "parameters() and gradients() have the same length (%zu vs %zu)",
                       p.size(), gg.size());
        check(p.size() == gg.size() && !p.empty(), buf2);

        bool all_shapes_match = true;
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i]->rows != gg[i]->rows || p[i]->cols != gg[i]->cols)
                all_shapes_match = false;
        check(all_shapes_match, "every parameter has a shape-matched gradient");

        // The stack's INPUT GRADIENT must match centered finite differences
        // of the stack's OWN forward.
        //
        // Without this, the backward's block ORDER is unconstrained: a
        // backward that chains the blocks in the FORWARD order still returns
        // a non-zero gradient, so the "the ordinary stack path is trainable"
        // check above passes and the defect ships. The finite differences
        // are the only thing that pins the order.
        {
            Tensor y = tr.forward(x);
            Tensor g(ORACLE_D_MODEL, ORACLE_TOKENS);
            for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
                for (size_t s = 0; s < ORACLE_TOKENS; ++s)
                    g(f, s) = 0.7 * y(f, s) - 0.3;
            tr.zero_grad();
            Tensor dx = tr.backward(g, 0.0);

            auto loss_of = [&](const Tensor& xx) {
                Tensor yy = tr.forward(xx);
                double L = 0.0;
                for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
                    for (size_t s = 0; s < ORACLE_TOKENS; ++s)
                        L += g(f, s) * yy(f, s);
                return L;
            };
            const double eps = 1e-5;
            double worst = 0.0;
            double fd_max = 0.0;
            for (size_t f = 0; f < ORACLE_D_MODEL; ++f)
                for (size_t s = 0; s < ORACLE_TOKENS; ++s) {
                    Tensor xp = x.clone(), xm = x.clone();
                    xp(f, s) += eps; xm(f, s) -= eps;
                    const double num = (loss_of(xp) - loss_of(xm)) / (2.0 * eps);
                    const double ana = dx(f, s);
                    fd_max = std::max(fd_max, std::fabs(num));
                    double den = std::max(std::fabs(ana), std::fabs(num));
                    if (den < 1e-12) den = 1.0;
                    worst = std::max(worst, std::fabs(ana - num) / den);
                }
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "the stack's input gradient matches FD (worst rel_err = %.3e < 1e-4, "
                          "|FD|max = %.3e > 1e-9)", worst, fd_max);
            check(worst < 1e-4 && fd_max > 1e-9, buf);
        }
    }

    // -----------------------------------------------------------------------
    // 7b. THE UPFRONT SLOT RESERVATION IS LOAD-BEARING.
    //
    // The stack reserves every slot when the cache is attached, so a cache
    // that is too narrow for this stack fails IMMEDIATELY with a message
    // about the slot -- not at the first decode with a message about a row
    // count. Without the upfront reservation the blocks would reserve their
    // slots lazily and the failure would arrive later, at the first decode,
    // with a less useful message.
    // -----------------------------------------------------------------------
    {
        TransformerStack s4(ORACLE_D_MODEL, ORACLE_HEADS, ORACLE_NUM_BLOCKS,
                            ORACLE_TOKENS);
        auto narrow = std::make_shared<nn::utils::KVCache>(ORACLE_TOKENS);
        narrow->reserve(0, ORACLE_D_MODEL + 1);   // the wrong width for slot 0
        check(threw_with([&] { s4.set_kv_cache(narrow); }, "already reserved"),
              "set_kv_cache rejects a cache whose slot has the wrong width");
    }

    // -----------------------------------------------------------------------
    // 8. reset_cache() empties the slots and the stack is reusable: the
    //    second decode reproduces the first one EXACTLY.
    // -----------------------------------------------------------------------
    {
        stack.reset_cache();
        std::vector<double> first, second;
        std::string err;
        bool ok1 = decode_all(stack, x, first, err);
        stack.reset_cache();
        bool ok2 = decode_all(stack, x, second, err);
        if (!ok1 || !ok2) {
            check(false, "the decode after reset_cache() threw: " + err);
        } else {
            double d = max_abs_diff(second, first.data(), first.size());
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "after reset_cache() the stack reproduces the decode exactly "
                          "(max abs diff = %.3e)", d);
            check(d == 0.0, buf);
        }
    }

    // -----------------------------------------------------------------------
    // 9. forward_cached() without a cache throws; the input shape is checked.
    // -----------------------------------------------------------------------
    {
        TransformerStack s2(ORACLE_D_MODEL, ORACLE_HEADS, ORACLE_NUM_BLOCKS,
                            ORACLE_TOKENS);
        s2.clear_kv_cache();
        Tensor step(ORACLE_D_MODEL, 1);
        step.fill(0.0);
        check(threw_with([&] { s2.forward_cached(step); },
                          "TransformerStack::forward_cached: no KV cache is attached"),
              "forward_cached() without a cache throws");

        TransformerStack s3(ORACLE_D_MODEL, ORACLE_HEADS, ORACLE_NUM_BLOCKS,
                            ORACLE_TOKENS);
        Tensor bad(ORACLE_D_MODEL + 1, 1);
        bad.fill(0.0);
        check(threw_with([&] { s3.forward_cached(bad); }, "d_model"),
              "forward_cached() rejects an input with the wrong d_model");
    }

    std::printf("=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
