// =============================================================================
// tests/test_kv_cache.cpp -- the inference-time key/value cache and the
// cached incremental forward path of the multi-head attention.
//
// THE SPECIFICATION IS tests/kv_cache_oracle.py (float64 numpy), which was
// written and run BEFORE this file existed. Its discriminating power was
// MEASURED, not narrated: tests/kv_cache_mutate_oracle.py breaks the
// oracle's model code in 22 specific ways and requires that it fail each
// time (22 caught / 0 survived).
//
// THE GAP, MEASURED NOT ASSUMED (2026-11-16):
//
//   grep -rinE '\bkv.?cache|paged.attn|prefix.cache|past_key' include/
//   -> 3 hits, ALL COMMENTS (mla.h:33, sliding_window.h:17, megalodon.h:67)
//   -> ZERO implementation across ~250 layers.
//
// Many attention layers hold last_k_ / last_v_ -- but those are caches for
// the BACKWARD pass, not an inference-time cache. The repo had no
// incremental decoding path at all.
//
// THE INVARIANT THIS WHOLE FEATURE RESTS ON
//
//   For a causal attention, decoding one token at a time with a cache must
//   equal the full-sequence forward, TOKEN FOR TOKEN. That is a property of
//   the OUTPUT, not a theorem, and the suite pins it.
//
// AND IT IS PINNED EXACTLY, NOT WITHIN A TOLERANCE. The cached path performs
// the same arithmetic in the same order as the full path, so the two agree
// to the LAST BIT -- and a tolerance would hide a cache that attends to the
// wrong tokens by a small amount.
//
// THE NON-VACUITY REQUIREMENT
//
//   A cached path that silently ignores the cache produces the right answer
//   at step 0 and the wrong one everywhere after. The suite therefore also
//   proves that the cache is READ: perturbing token 0 must change the
//   output of a later step.
//
// THE ERROR CHECKS ASSERT THE MESSAGE, NOT MERELY "IT THREW". A check that
// only asserts "it threw" is blind to a guard that was removed while a
// DIFFERENT guard still fires: the throw happens either way. This is not
// hypothetical -- the oracle's own mutation harness showed that removing
// the width guard leaves numpy's broadcast error throwing in its place, so
// a throw-only check stays GREEN.
//
//   C++ evaluates a function call's arguments in UNSPECIFIED order, so
//   every throwing assertion goes through expect_throw(), which calls,
//   captures, and THEN reports. A message printed from the wrong call is
//   worse than no message at all.
// =============================================================================

#include "nn/layers/attention/transformer.h"
#include "nn/utils/kv_cache.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::string;
using std::vector;

// =============================================================================
// THE PINNED FIXTURES ARE TRANSFERRED FROM THE ORACLE, NOT HAND-TRANSCRIBED.
//
// The repo has been burned by hand-transcribed numbers before (see
// tests/_dump_fixture.py): both numbers were CORRECT for the fixture they
// were given, and the fixture was simply not the oracle's. Pinned numbers
// transfer only across an IDENTICAL fixture, so these were EMITTED by
//     /usr/bin/python3 tests/kv_cache_oracle.py --emit-cpp
// and not typed by hand.
//
// seed 20261116, d_model 8, seq_len 6, heads 2
// =============================================================================

static const int KV_D_MODEL = 8;
static const int KV_SEQ_LEN = 6;
static const int KV_HEADS = 2;

static const std::vector<double> KV_X_FIX = {
    -0.35599974504217835, 0.43427471015988584, -0.514620988542703, -0.11772034489646466, 0.10368991150207786, -0.05987813340886757,
    0.6387447276298824, -0.6023808299584948, -1.3885997085208888, -0.09858042910958195, -1.3181005172874007, 0.556037312367726,
    -0.24049340722557727, -0.04285328413102079, 0.6041565965774298, -1.43992761586889, -0.35324297584907377, -0.7689358549211109,
    -0.5717137761028847, 0.46389095296338156, 0.007385089772057514, 0.028047965477155837, 0.20290445398480078, 0.21969026311498915,
    1.0777351126115138, -0.05415018145632807, 0.06497797607834745, 0.028307903316832066, 0.7438779793677471, -0.7617294480798199,
    -0.011220357049991263, -0.6829420607075286, 0.17416759568039356, 1.0629309201826975, 1.0097275087790958, 0.08387773830802611,
    0.38973983772789556, 0.44443989521182703, 0.33753395429088834, -0.7775290583278163, -1.130808610975766, -0.53044866060624,
    0.600033396979507, 0.11655764098634233, 0.6562169200607526, 0.7532724248919819, 0.29474833802158024, 0.4040661697208531};

static const std::vector<double> KV_WQ_FIX = {
    -0.4763354847199334, 0.2438675859175467, 0.03451708725101159, 0.46873169636813566, -0.36789906626359903, 0.06332975852858257,
    -0.3287281230375521, 0.26038787149021214, -0.05448529923338356, -0.30206465718096975, -0.09836012175400383, -0.06784556424969598,
    0.017534409122697207, 0.5210499295249775, 0.07957665109194392, 0.7442909080554723, -0.02727797169244726, -0.2595840169420761,
    0.28114975861504765, 0.014279068234050162, 0.34686995799558906, 0.364735313766668, -0.17839701718994677, 0.24759569766311781,
    -0.05967456886363218, -0.5896564900284109, -0.05330241193140768, 0.2924510871949626, 0.23782500458819866, -0.3198158851236363,
    0.4304035139763722, 0.005247608827290236, -0.18606363179313293, 0.23358940853779847, 0.4960234927661522, 0.1460031836833668,
    0.24905563226063557, 0.333550750507117, 0.20891097990864052, -0.14515830368284355, 0.2504334496004304, -0.043221211759016435,
    0.1690751448650314, -0.13400776367697104, -0.2092632118804653, -0.248267485466457, 0.13270932490621937, -0.6063473629367654,
    0.0690924419164847, -0.34677289788279986, -0.07799162220683671, 0.15057264011074814, 0.019619600290170992, -0.15980419782238328,
    0.030700424577188534, -0.10791890697622214, -0.13056977207263157, 0.487682727827473, 0.44977767535878505, 0.24092200103161474,
    -0.024931117319487626, -0.1743459269994268, -0.0018574770130816197, 0.030131629540430447};

static const std::vector<double> KV_WK_FIX = {
    -0.6155458084168441, -0.18375516007692183, -0.23591552171434907, -0.6229919078382474, 0.3108802878229946, 0.1017258633411431,
    0.023709796371865106, -0.11249004475382751, -0.22225562796313636, -0.1060577690012803, -0.051103514426867475, 0.21907339155703864,
    0.00920332676056683, 0.18505905329105754, 0.017830411398159673, -0.10577966862163544, -0.3957685346617383, 0.12287869696875754,
    -0.004021486635071403, 0.5704942144534151, 0.09660860916203483, 0.17401827364499528, -0.17972038296846113, 0.19681232259911477,
    0.03469293266121161, 0.20080621012966807, 0.20528904508696566, -0.1944533404431947, 0.40693084710280497, -0.3658952469587743,
    0.4136016062344613, -0.32213840019890394, 0.13149467470812812, 0.6984629363995409, -0.4201590199120475, 0.10082621659660958,
    0.8211716458715888, -0.21267495630860406, -0.1857937534545276, 0.34243693572901857, 0.24620717633989145, 0.09672534460824056,
    0.4202865077805471, -0.17667474698907024, -0.23932868502351193, -0.0015901412158791102, 0.5546419989426451, 0.42646549664129607,
    0.2174817389850116, -0.24925766340450797, 0.4198398284127045, -0.14651755129267968, -0.16357389958983792, -0.268656299422639,
    0.020318857296666513, -0.38558177798953397, 0.3273166236109777, 0.5106576930821638, 0.13974919543173797, 0.2755453638453848,
    0.13306694448898157, 0.5667796765443268, -0.5433406133058957, -0.00029891338649228456};

static const std::vector<double> KV_WV_FIX = {
    -1.0977307374778114, 0.4876618903206665, 0.7870047029789009, 0.21126686442739875, -0.020146825899662525, 0.04633346109028671,
    0.3927291843502208, -0.009921460366637252, 0.42363604621208917, -0.15482701472756558, 0.23890796632883843, -0.5691593210861626,
    -0.3819665168322729, -0.8184606673891109, -0.029670423141472287, -0.2215949699779129, -0.32651955991562104, -0.07652797830387852,
    0.721121182275189, -0.5828029981606442, 0.3705143917062357, 0.0996437173536573, -0.3403746791246567, 0.48538109853390526,
    0.3257465950412768, -0.020728363452084803, -0.6621974864635346, -0.010742755647388073, -0.35582875687529125, 0.14876253997878724,
    -0.12036285364828932, -0.013618044088707178, 0.24013275016471405, 0.19738370388973397, -0.21700355247613148, -0.08226941894107698,
    -0.20925299539118547, 0.021636530775107417, -0.154195576638721, 0.24026754672929346, 0.5269518765545661, 0.388444374923616,
    -0.1537566736853388, 0.008894283476998641, 0.2290351388071024, -0.34661899416083586, 0.3732486209285677, 0.6432931583977195,
    -0.22641102212701894, -0.04529085320588318, -0.0692695963948272, -0.45309744898753995, -0.12540756241814216, -0.1952458645524567,
    -0.13530452553016425, 0.28255318360316006, -0.5334299041735787, 1.0982815045414738, -0.6655325274203986, -0.033378872974364704,
    -0.1035561091416432, 0.13680573095655857, -0.21361393832586903, -0.09556178318745927};

static const std::vector<double> KV_WO_FIX = {
    0.03409218099774176, -0.30553458480762524, 0.318964520646644, 0.1918102377932118, -0.2245479627395704, -0.9104315015085709,
    0.40100441842276, 0.028198751436159394, 0.3135901448757111, 0.49029320022377115, 0.19148961651770854, -0.5193762793229317,
    -0.2620954314959867, -0.09058695758107516, -0.03056605202207582, -0.3771115029186018, 0.1608720957777542, -0.09657762415402835,
    -0.26552655169407985, 0.40225543655758667, -0.5549776909914623, 0.7050336226905547, 0.20538986007968807, -0.07337673595259159,
    -0.4485527670217885, -0.5902443474152387, -0.1589542443586756, -0.6564894199687412, -0.4488470452005488, -0.7106466190168292,
    0.5757518771813402, 0.24562991049600508, 0.03210142318235187, -0.06271112820536848, -0.024632193692253807, 0.38420013708135,
    1.0014113868788381, 0.14186339128839937, 0.10159760861369539, -0.3476940516347779, -0.09941027198677496, 0.13051736826741944,
    0.3876921962005029, -0.07246427582612838, 0.0760716701927897, 0.07152284784869561, -0.4276219935463477, -0.1688985465157054,
    -0.6757788535099317, 0.20502595891880146, 0.7735558255807509, -0.40355280452471387, -0.360753425909125, 0.024893151765607045,
    -0.8219942442514314, 0.06483125525807164, -0.23853369039268393, 0.04784523670511233, 0.04813638115769198, 0.3276144243194392,
    -0.15604018861119826, -0.02158142775654399, -0.31023336475271895, -0.0403503062864845};

static const std::vector<double> KV_OUT_FULL_FIX = {
    0.6452735994323197, 0.3232540923622666, 0.2238648316266104, -0.03567318758211593, -0.15102623011988256, -0.25287282960622576,
    0.4510693095202559, 0.40214470948137493, 0.4706122867989308, 0.3362449069526231, 0.23422468152493175, 0.18966148179455,
    -0.012875299325647304, -0.06386380010148358, 0.09846173751448122, 0.46181945612513514, 0.5091707036028388, 0.5318940170560769,
    -0.0623307405786346, -0.3288063933798089, -0.42530628512628055, -0.9396082197584074, -1.1567221732925361, -1.093400834186421,
    -0.055259215532171245, 0.06526915867514738, 0.3835889005110236, 0.18704685205443125, 0.0766492114209953, 0.018786393224579572,
    -0.5361821998650775, 0.07106995680387293, -0.024503188150233727, -0.9006612955172907, -1.1444312422136484, -1.065974185524498,
    0.18280790802069002, -0.13841894868913368, -0.31169773493606423, -0.17979023815821268, -0.07935151832967907, -0.12452907926718,
    -0.07840860176976364, -0.09378704846846221, -0.2767538274614065, -0.2136250818766077, -0.20846873089203757, -0.13399722601036979};

static const std::vector<double> KV_OUT_INC_FIX = {
    0.6452735994323198, 0.32325409236226665, 0.22386483162661036, -0.035673187582115955, -0.15102623011988262, -0.2528728296062257,
    0.451069309520256, 0.40214470948137493, 0.47061228679893086, 0.33624490695262316, 0.23422468152493164, 0.18966148179455,
    -0.012875299325647127, -0.06386380010148347, 0.09846173751448131, 0.4618194561251351, 0.5091707036028388, 0.5318940170560769,
    -0.06233074057863461, -0.32880639337980866, -0.42530628512628077, -0.9396082197584076, -1.1567221732925357, -1.0934008341864212,
    -0.055259215532171224, 0.0652691586751474, 0.3835889005110237, 0.18704685205443125, 0.07664921142099534, 0.01878639322457962,
    -0.5361821998650773, 0.07106995680387289, -0.02450318815023387, -0.9006612955172908, -1.1444312422136484, -1.0659741855244982,
    0.1828079080206898, -0.13841894868913374, -0.31169773493606445, -0.1797902381582127, -0.0793515183296791, -0.12452907926718007,
    -0.07840860176976372, -0.09378704846846228, -0.2767538274614065, -0.2136250818766077, -0.20846873089203755, -0.13399722601036984};

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const string& msg) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

static bool same(const Tensor& a, const Tensor& b, double tol = 0.0) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (std::fabs(a(i, j) - b(i, j)) > tol) return false;
    return true;
}

// A tensor of n rows x w cols with distinct values, so that a mix-up
// between the key buffer and the value buffer cannot pass.
static Tensor rows(int n, int w, double base) {
    Tensor t(n, w);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < w; ++j)
            t(i, j) = base + i * 100.0 + j;
    return t;
}

// THE DISCRIMINATOR IS THE MESSAGE, NOT THE THROW.
static void expect_throw(const std::function<void()>& fn,
                         const string& needle,
                         const string& name) {
    string what;
    bool threw = false;
    try {
        fn();
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    if (!threw) {
        ++g_fail;
        std::cout << "  [FAIL] " << name << " -- did not throw\n";
        return;
    }
    if (what.find(needle) == string::npos) {
        ++g_fail;
        std::cout << "  [FAIL] " << name << " -- threw \"" << what
                  << "\", expected a message containing \"" << needle << "\"\n";
        return;
    }
    ++g_pass;
}

// =============================================================================
// TEST 1: a fresh cache
// =============================================================================

static void test_construction() {
    nn::utils::KVCache cache(16);
    check(cache.capacity() == 16, "the capacity is reported");
    check(cache.layers() == 0, "a fresh cache has no layers");
    check(cache.total_tokens() == 0, "a fresh cache holds no tokens");
    check(cache.memory_bytes() == 0, "a fresh cache holds no bytes");
}

// =============================================================================
// TEST 2: reserve, append, and the buffer contents
// =============================================================================

static void test_append_and_size() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    check(cache.layers() == 1, "reserve creates a layer");
    check(cache.size(0) == 0, "a reserved layer starts empty");
    check(cache.d_kv(0) == 4, "the layer reports its width");

    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    check(cache.size(0) == 3, "append grows the layer by the row count");

    const Tensor& kb = cache.key_buffer(0);
    const Tensor& vb = cache.value_buffer(0);
    check(kb.rows == 8 && kb.cols == 4,
          "the key buffer is capacity x d_kv");
    check(vb.rows == 8 && vb.cols == 4,
          "the value buffer is capacity x d_kv");

    bool keys_ok = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (kb(i, j) != k(i, j)) keys_ok = false;
        }
    }
    check(keys_ok, "the cached keys are the appended rows");

    // The WHOLE buffer is compared, not just the valid rows: the tail must
    // still be zero, because a cache that scribbled past the length would
    // be read by a consumer that trusts size().
    Tensor expect_kb(8, 4), expect_vb(8, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) {
            expect_kb(i, j) = k(i, j);
            expect_vb(i, j) = v(i, j);
        }
    check(same(kb, expect_kb),
          "the key buffer holds the appended rows and nothing else");
    check(same(vb, expect_vb),
          "the value buffer holds the appended rows and nothing else");
}

// =============================================================================
// TEST 3: a second append lands AFTER the first, and does not disturb it
// =============================================================================

static void test_second_append() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    Tensor k1 = rows(3, 4, 1.0);
    Tensor v1 = rows(3, 4, -1.0);
    Tensor k2 = rows(2, 4, 500.0);
    Tensor v2 = rows(2, 4, -500.0);
    cache.append(0, k1, v1);
    cache.append(0, k2, v2);
    check(cache.size(0) == 5, "two appends grow the layer by five rows");

    const Tensor& kb = cache.key_buffer(0);
    const Tensor& vb = cache.value_buffer(0);
    bool first_intact = true, second_placed = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(i, j) != k1(i, j) || vb(i, j) != v1(i, j)) first_intact = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(3 + i, j) != k2(i, j) || vb(3 + i, j) != v2(i, j))
                second_placed = false;
    check(first_intact, "the FIRST append is not disturbed by the second");
    check(second_placed, "the second append lands after the first");
}

// =============================================================================
// TEST 4: the layers are independent
// =============================================================================

static void test_layer_independence() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    cache.reserve(1, 4);
    check(cache.layers() == 2, "two layers are reserved");
    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    cache.append(1, k, v);
    cache.append(1, k, v);
    check(cache.size(0) == 3, "layer 0 holds three tokens");
    check(cache.size(1) == 6, "layer 1 holds six tokens");
    check(cache.total_tokens() == 9, "total_tokens sums the layers");

    // A layer with a DIFFERENT width is a different layer.
    cache.reserve(2, 7);
    check(cache.d_kv(2) == 7, "each layer keeps its own width");
    check(cache.size(2) == 0, "a newly reserved layer is empty");
}

// =============================================================================
// TEST 5: reset
// =============================================================================

static void test_reset() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    cache.reserve(1, 4);
    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    cache.append(1, k, v);

    cache.reset(0);
    check(cache.size(0) == 0, "reset empties one layer");
    check(cache.size(1) == 3, "reset leaves the other layers alone");

    cache.reset_all();
    check(cache.total_tokens() == 0, "reset_all empties every layer");

    // A reset layer is REUSABLE, and it really starts empty.
    cache.append(0, k, v);
    check(cache.size(0) == 3, "a reset layer is reusable");
    const Tensor& kb = cache.key_buffer(0);
    bool ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(i, j) != k(i, j)) ok = false;
    check(ok, "a reset layer really starts empty");
}

// =============================================================================
// TEST 6: the memory accounting
// =============================================================================

static void test_memory_accounting() {
    nn::utils::KVCache cache(8);
    check(cache.memory_bytes() == 0, "an empty cache holds no bytes");
    cache.reserve(0, 4);
    check(cache.memory_bytes() == 2 * 8 * 4 * 8,
          "memory_bytes is two buffers of capacity x d_kv doubles");
    cache.reserve(1, 4);
    check(cache.memory_bytes() == 2 * 2 * 8 * 4 * 8,
          "memory_bytes sums the layers");
    cache.reserve(2, 7);
    check(cache.memory_bytes() == 2 * 8 * 4 * 8 + 2 * 8 * 4 * 8 + 2 * 8 * 7 * 8,
          "memory_bytes uses each layer's own width");
}

// =============================================================================
// TEST 7: the error cases -- each with its OWN message
// =============================================================================

static void test_errors() {
    nn::utils::KVCache cache(4);

    // An unreserved layer.
    expect_throw([&] { cache.size(0); }, "not reserved",
                 "size() on an unreserved layer throws");
    expect_throw([&] { cache.key_buffer(0); }, "not reserved",
                 "key_buffer() on an unreserved layer throws");
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 3, 0.0)); },
                 "not reserved",
                 "appending to an unreserved layer throws");

    cache.reserve(0, 3);

    // The width guard: the layer was reserved with 3 columns. The needle is
    // "reserved with" and NOT "columns", because the key/value width
    // mismatch below says "the same number of columns" -- a needle shared by
    // two messages discriminates nothing.
    expect_throw([&] { cache.append(0, rows(1, 4, 0.0), rows(1, 4, 0.0)); },
                 "reserved with",
                 "appending the wrong width throws");

    // The row-count guard.
    expect_throw([&] { cache.append(0, rows(2, 3, 0.0), rows(3, 3, 0.0)); },
                 "same number of rows",
                 "appending mismatched row counts throws");

    // The key/value width guard.
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 4, 0.0)); },
                 "same number of columns",
                 "appending keys and values of different widths throws");

    // The zero-rows guard.
    expect_throw([&] { cache.append(0, rows(0, 3, 0.0), rows(0, 3, 0.0)); },
                 "zero rows",
                 "appending zero rows throws");

    // The capacity guard.
    cache.append(0, rows(4, 3, 0.0), rows(4, 3, 0.0));
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 3, 0.0)); },
                 "capacity",
                 "appending past the capacity throws");

    // Re-reserving with a different width.
    expect_throw([&] { cache.reserve(0, 4); }, "already reserved",
                 "re-reserving a layer with a different width throws");

    // Re-reserving with the SAME width is allowed and keeps the content.
    nn::utils::KVCache keep(8);
    keep.reserve(0, 3);
    keep.append(0, rows(2, 3, 7.0), rows(2, 3, -7.0));
    keep.reserve(0, 3);
    check(keep.size(0) == 2,
          "re-reserving a layer with the same width keeps its content");
}

// =============================================================================
// TEST 8: the C++ full forward matches the ORACLE's pinned output
//
// The oracle ran in float64 numpy with BLAS; the C++ uses explicit loops, so
// the two agree to within the floating-point noise of the SUMMATION ORDER,
// not to the last bit. The tolerance is 1e-12 on values of order 1.
// =============================================================================

static void make_mha(MultiHeadAttention& mha) {
    mha.W_q = Tensor(KV_D_MODEL, KV_D_MODEL, KV_WQ_FIX.data());
    mha.W_k = Tensor(KV_D_MODEL, KV_D_MODEL, KV_WK_FIX.data());
    mha.W_v = Tensor(KV_D_MODEL, KV_D_MODEL, KV_WV_FIX.data());
    mha.W_o = Tensor(KV_D_MODEL, KV_D_MODEL, KV_WO_FIX.data());
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double worst = 0.0;
    const size_t rows = std::min(a.rows, b.rows);
    const size_t cols = std::min(a.cols, b.cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            const double d = std::fabs(a(i, j) - b(i, j));
            if (d > worst) worst = d;
        }
    return worst;
}

static void test_full_forward_matches_the_oracle() {
    MultiHeadAttention mha(KV_D_MODEL, KV_HEADS);
    make_mha(mha);
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());
    Tensor got = mha.forward(x);
    Tensor want(KV_D_MODEL, KV_SEQ_LEN, KV_OUT_FULL_FIX.data());
    const double d = max_abs_diff(got, want);
    check(d <= 1e-12,
          "the full forward matches the oracle's pinned output (max_abs_diff = " +
              std::to_string(d) + ")");
}

// =============================================================================
// TEST 9: the token-by-token decode EQUALS the full forward, TO THE LAST BIT
//
// This is the invariant the whole feature rests on. It is asserted EXACTLY
// (0.0), not within a tolerance: the cached path performs the same
// arithmetic in the same order, so the two agree bit for bit -- and a
// tolerance would hide a cache that attends to the WRONG tokens by a small
// amount.
//
// The oracle's own incremental output is pinned as well, so the C++ is
// checked against BOTH the numpy incremental path and its own full path.
// =============================================================================

static Tensor decode(MultiHeadAttention& mha, const Tensor& x, int chunk) {
    const size_t T = x.cols;
    Tensor out(KV_D_MODEL, T);
    for (size_t start = 0; start < T; start += static_cast<size_t>(chunk)) {
        const size_t n = std::min(static_cast<size_t>(chunk), T - start);
        Tensor part(KV_D_MODEL, n);
        for (size_t f = 0; f < KV_D_MODEL; ++f)
            for (size_t s = 0; s < n; ++s)
                part(f, s) = x(f, start + s);
        Tensor got = mha.forward_cached(part);
        check(got.rows == KV_D_MODEL && got.cols == n,
              "the cached forward returns the same shape as its input");
        for (size_t f = 0; f < KV_D_MODEL; ++f)
            for (size_t s = 0; s < n; ++s)
                out(f, start + s) = got(f, s);
    }
    return out;
}

static void test_decode_matches_full_forward() {
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());

    MultiHeadAttention full_mha(KV_D_MODEL, KV_HEADS);
    make_mha(full_mha);
    Tensor full = full_mha.forward(x);

    for (int chunk = 1; chunk <= KV_SEQ_LEN; ++chunk) {
        MultiHeadAttention mha(KV_D_MODEL, KV_HEADS);
        make_mha(mha);
        std::shared_ptr<nn::utils::KVCache> cache =
            std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
        mha.set_kv_cache(cache, 0);
        Tensor decoded = decode(mha, x, chunk);
        const double d = max_abs_diff(decoded, full);
        check(d == 0.0,
              "the decode (chunk " + std::to_string(chunk) +
                  ") equals the full forward TO THE LAST BIT (max_abs_diff = " +
                  std::to_string(d) + ")");
        check(cache->size(0) == KV_SEQ_LEN,
              "the cache holds every token after the decode");
    }

    // And against the ORACLE's own incremental output.
    MultiHeadAttention mha(KV_D_MODEL, KV_HEADS);
    make_mha(mha);
    std::shared_ptr<nn::utils::KVCache> cache =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    mha.set_kv_cache(cache, 0);
    Tensor decoded = decode(mha, x, 1);
    Tensor want(KV_D_MODEL, KV_SEQ_LEN, KV_OUT_INC_FIX.data());
    const double d = max_abs_diff(decoded, want);
    check(d <= 1e-12,
          "the decode matches the oracle's pinned incremental output "
          "(max_abs_diff = " + std::to_string(d) + ")");
}

// =============================================================================
// TEST 10: TWO ATTENTION LAYERS SHARE ONE CACHE
//
// This is what a transformer stack looks like: one cache, one layer index
// per attention layer. If the layer index were ignored, the two layers would
// read each other's keys and the decode would be wrong.
// =============================================================================

static void test_two_layers_share_one_cache() {
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());

    MultiHeadAttention full_a(KV_D_MODEL, KV_HEADS);
    MultiHeadAttention full_b(KV_D_MODEL, KV_HEADS);
    make_mha(full_a);
    make_mha(full_b);
    Tensor want_a = full_a.forward(x);
    Tensor want_b = full_b.forward(x);

    MultiHeadAttention a(KV_D_MODEL, KV_HEADS);
    MultiHeadAttention b(KV_D_MODEL, KV_HEADS);
    make_mha(a);
    make_mha(b);
    std::shared_ptr<nn::utils::KVCache> cache =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    a.set_kv_cache(cache, 0);
    b.set_kv_cache(cache, 1);

    // A real stack: layer 0 then layer 1, one token at a time.
    Tensor got_a(KV_D_MODEL, KV_SEQ_LEN), got_b(KV_D_MODEL, KV_SEQ_LEN);
    for (size_t t = 0; t < KV_SEQ_LEN; ++t) {
        Tensor one(KV_D_MODEL, 1);
        for (size_t f = 0; f < KV_D_MODEL; ++f)
            one(f, 0) = x(f, t);
        Tensor oa = a.forward_cached(one);
        Tensor ob = b.forward_cached(one);
        for (size_t f = 0; f < KV_D_MODEL; ++f) {
            got_a(f, t) = oa(f, 0);
            got_b(f, t) = ob(f, 0);
        }
    }
    check(max_abs_diff(got_a, want_a) == 0.0,
          "the first layer of the stack decodes exactly");
    check(max_abs_diff(got_b, want_b) == 0.0,
          "the second layer of the stack decodes exactly");
    check(cache->size(0) == KV_SEQ_LEN && cache->size(1) == KV_SEQ_LEN,
          "the shared cache holds one sequence per layer");
}

// =============================================================================
// TEST 11: THE CACHE IS ACTUALLY READ
//
// A cached path that silently ignored the cache would produce the right
// answer at step 0 and the wrong one at every later step. This test pins the
// NON-VACUITY: perturbing token 0 must change the output of a later step.
//
// Note what makes this discriminating: if the cache were ignored, the later
// steps would depend only on the LATER tokens, so they would be IDENTICAL
// for the two inputs -- and this test would fail.
// =============================================================================

static void test_the_cache_is_read() {
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());
    Tensor x2 = x.clone();
    x2(0, 0) += 3.0;

    MultiHeadAttention a(KV_D_MODEL, KV_HEADS);
    MultiHeadAttention b(KV_D_MODEL, KV_HEADS);
    make_mha(a);
    make_mha(b);
    std::shared_ptr<nn::utils::KVCache> ca =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    std::shared_ptr<nn::utils::KVCache> cb =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    a.set_kv_cache(ca, 0);
    b.set_kv_cache(cb, 0);

    Tensor da = decode(a, x, 1);
    Tensor db = decode(b, x2, 1);

    // The last step must differ: it attends to the perturbed token 0.
    double last = 0.0;
    for (size_t f = 0; f < KV_D_MODEL; ++f)
        last = std::max(last, std::fabs(da(f, KV_SEQ_LEN - 1) - db(f, KV_SEQ_LEN - 1)));
    check(last > 1e-6,
          "perturbing token 0 changes the LAST step -- the cache is read "
          "(max_abs_diff = " + std::to_string(last) + ")");

    // And the FIRST step must be identical in the one case where it must be:
    // the two inputs differ only in token 0, so step 0 differs -- but a
    // step that CANNOT see the future is unaffected by a LATER token.
    Tensor x3 = x.clone();
    x3(0, KV_SEQ_LEN - 1) += 3.0;
    MultiHeadAttention c(KV_D_MODEL, KV_HEADS);
    make_mha(c);
    std::shared_ptr<nn::utils::KVCache> cc =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    c.set_kv_cache(cc, 0);
    Tensor dc = decode(c, x3, 1);
    double first = 0.0;
    for (size_t f = 0; f < KV_D_MODEL; ++f)
        first = std::max(first, std::fabs(da(f, 0) - dc(f, 0)));
    check(first == 0.0,
          "the first step cannot see a later token (max_abs_diff = " +
              std::to_string(first) + ")");
}

// =============================================================================
// TEST 12: attaching a cache does NOT change the ordinary forward
//
// The cached path is a SEPARATE entry point. `forward()` must behave
// identically whether or not a cache is attached -- otherwise attaching a
// cache would silently change every model that uses this layer.
// =============================================================================

static void test_attaching_a_cache_does_not_change_forward() {
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());

    MultiHeadAttention plain(KV_D_MODEL, KV_HEADS);
    make_mha(plain);
    Tensor want = plain.forward(x);

    MultiHeadAttention cached(KV_D_MODEL, KV_HEADS);
    make_mha(cached);
    cached.set_kv_cache(std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN), 0);
    Tensor got = cached.forward(x);
    check(max_abs_diff(got, want) == 0.0,
          "forward() is unchanged by the attachment of a cache");

    // And the cache holds no tokens after an ordinary forward: the ordinary
    // path does not touch the cache.
    //
    // THE SLOT IS RESERVED EXPLICITLY, because the ordinary forward does NOT
    // reserve it -- and `size()` on an unreserved layer THROWS by design.
    // The first draft of this check called size(0) on a layer that had never
    // been reserved and took the whole suite down with an uncaught
    // std::invalid_argument. The failure was in the CHECK: the property
    // under test is "the ordinary forward writes nothing", and the way to
    // say that is to reserve the slot first and then assert it is empty.
    MultiHeadAttention c2(KV_D_MODEL, KV_HEADS);
    make_mha(c2);
    std::shared_ptr<nn::utils::KVCache> cache =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    c2.set_kv_cache(cache, 0);
    cache->reserve(0, KV_D_MODEL);
    c2.forward(x);
    check(cache->size(0) == 0,
          "the ordinary forward does not write to the cache");
}

// =============================================================================
// TEST 13: the cached path's error cases
// =============================================================================

static void test_cached_forward_errors() {
    Tensor x(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());

    // No cache attached.
    MultiHeadAttention mha(KV_D_MODEL, KV_HEADS);
    make_mha(mha);
    expect_throw([&] { mha.forward_cached(x); }, "no KV cache is attached",
                 "the cached forward without a cache throws");

    // The cache's capacity is exceeded: the cache refuses to grow.
    MultiHeadAttention small(KV_D_MODEL, KV_HEADS);
    make_mha(small);
    std::shared_ptr<nn::utils::KVCache> cache =
        std::make_shared<nn::utils::KVCache>(2);
    small.set_kv_cache(cache, 0);
    bool threw = false;
    try {
        decode(small, x, 1);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "the cached forward propagates the cache's capacity error");

    // A cached forward is an INFERENCE path: there is no adjoint for it.
    MultiHeadAttention inf(KV_D_MODEL, KV_HEADS);
    make_mha(inf);
    std::shared_ptr<nn::utils::KVCache> c2 =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    inf.set_kv_cache(c2, 0);
    Tensor one(KV_D_MODEL, 1);
    inf.forward_cached(one);
    Tensor grad(KV_D_MODEL, 1, KV_X_FIX.data());
    expect_throw([&] { inf.backward(grad, 0.01); }, "inference path",
                 "backward() after a cached forward throws");

    // After an ORDINARY forward the backward works again: the cached
    // forward must not leave the layer permanently unusable.
    MultiHeadAttention back(KV_D_MODEL, KV_HEADS);
    make_mha(back);
    std::shared_ptr<nn::utils::KVCache> c3 =
        std::make_shared<nn::utils::KVCache>(KV_SEQ_LEN);
    back.set_kv_cache(c3, 0);
    back.forward_cached(one);
    back.forward(x);
    Tensor grad_full(KV_D_MODEL, KV_SEQ_LEN, KV_X_FIX.data());
    bool backward_ok = true;
    try {
        back.backward(grad_full, 0.01);
    } catch (const std::exception&) {
        backward_ok = false;
    }
    check(backward_ok,
          "an ordinary forward re-enables the backward pass");
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "=== test_kv_cache ===\n";
    test_construction();
    test_append_and_size();
    test_second_append();
    test_layer_independence();
    test_reset();
    test_memory_accounting();
    test_errors();
    test_full_forward_matches_the_oracle();
    test_decode_matches_full_forward();
    test_two_layers_share_one_cache();
    test_the_cache_is_read();
    test_attaching_a_cache_does_not_change_forward();
    test_cached_forward_errors();
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    if (g_fail > 0) {
        std::cout << "TESTS FAILED\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
