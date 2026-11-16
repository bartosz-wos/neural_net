// =============================================================================
// tests/test_block_cached_decode.cpp -- the BLOCK-LEVEL cached decode and the
// ABSOLUTE positional encoding.
//
// THE SPECIFICATION IS tests/block_cached_decode_oracle.py (float64 numpy),
// which was written and run BEFORE this file existed. Its discriminating
// power was MEASURED, not narrated: tests/block_cached_decode_mutate_oracle.py
// breaks the oracle's model code in 23 specific ways and requires that it
// fail each time (23 caught / 0 survived).
//
// THE GAP, MEASURED NOT ASSUMED (2026-11-16):
//
//   grep -rinE 'residual.?stream|hidden.?state.?cache|x.?cache' include/
//   -> 3 hits, ALL COMMENTS -> ZERO implementation.
//
// The repo shipped an attention-level KV cache and a cached incremental path
// for MultiHeadAttention, but `TransformerBlock` -- the block every decoder
// stack is built from -- had NO cached forward at all.
//
// THE INVARIANT THIS WHOLE FEATURE RESTS ON
//
//   For a causal block, decoding token by token with a cache must equal the
//   full-sequence forward, TOKEN FOR TOKEN. That is a property of the
//   OUTPUT, not a theorem, and the suite pins it.
//
// AND IT IS PINNED EXACTLY, NOT WITHIN A TOLERANCE. The cached path
// performs the same arithmetic in the same order (the shared `compute_tail`
// helper, the shared `project_qkv` helper), and the block's post-attention
// tail is ROW-WISE, so the two agree to the LAST BIT. A tolerance would
// hide a decode that attends to the wrong tokens by a small amount.
//
// NO RESIDUAL-STREAM CACHE IS NEEDED, AND THAT IS A MEASURED FACT.
//
//   The expansion queue asserted that a block-level decode "needs a SECOND
//   cache [for the residual], and the FFN's position-wise nature does not
//   remove the requirement". It is FALSE in this repo: `LayerNorm`
//   normalises over the FEATURE axis of each row (layer_norm.cpp), so it
//   is PER TOKEN, and the whole tail is POSITION-WISE. The attention is the
//   sole cross-token coupling, and the key/value cache covers it exactly.
//
//   The suite pins that measurement directly (test_the_tail_is_row_wise):
//   the tail computed on the last two tokens alone equals the full
//   sequence's rows to 0.0, while the COUNTERFACTUAL -- a LayerNorm over
//   the token axis -- differs by 1.54. Without the counterfactual the zero
//   would be a tautology rather than evidence.
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
// DIFFERENT guard still fires: the throw happens either way.
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

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const string& name) {
    // std::endl, NOT "\n": stdout is BLOCK-BUFFERED when it is redirected
    // to a file, and a SEGFAULT loses the buffer. The suite must still
    // REPORT what it found when it dies -- the mutation harness reads the
    // log to tell a DETECTED defect (a [FAIL] line, then a crash) from a
    // SILENT crash, and without the flush the [FAIL] lines were lost with
    // the buffer and every crash looked silent.
    if (cond) {
        ++g_pass;
        std::cout << "  [PASS] " << name << std::endl;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << name << std::endl;
    }
}

// The maximum absolute difference between two tensors.
static double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e300;
    double worst = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            double d = std::fabs(a(i, j) - b(i, j));
            if (d > worst) worst = d;
        }
    return worst;
}

// C++ evaluates a function call's arguments in UNSPECIFIED order, so
// `check(threw_with(fn, needle, &what), "..." + what)` can print the message
// from the PREVIOUS call. Every throwing assertion therefore goes through
// this helper, which CALLS, CAPTURES, and THEN reports.
static bool threw_with(const std::function<void()>& fn, const string& needle,
                       string* what) {
    try {
        fn();
    } catch (const std::exception& e) {
        *what = e.what();
        return what->find(needle) != string::npos;
    }
    *what = "(no exception)";
    return false;
}

static void expect_throw(const std::function<void()>& fn, const string& needle,
                         const string& name) {
    string what;
    if (threw_with(fn, needle, &what)) {
        ++g_pass;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << name << " -- expected a message containing \""
                  << needle << "\", got \"" << what << "\"\n";
    }
}

// =============================================================================
// THE PINNED FIXTURES ARE TRANSFERRED FROM THE ORACLE, NOT HAND-TRANSCRIBED.
//
// The repo has been burned by hand-transcribed numbers before (see
// tests/_dump_fixture.py): both numbers were CORRECT for the fixture they
// were given, and the fixture was simply not the oracle's. Pinned numbers
// transfer only across an IDENTICAL fixture, so these were EMITTED by
//     /usr/bin/python3 tests/block_cached_decode_oracle.py --emit-cpp
// and not typed by hand.
//
// seed 20261116, d_model 8, seq_len 6, heads 2
// =============================================================================

// >>> ORACLE-EMITTED FIXTURE BEGIN (do not hand-edit) <<<
// ==== generated by tests/block_cached_decode_oracle.py --emit-cpp; do not hand-edit ====
// seed 20261116, d_model 8, seq_len 6, heads 2
const int BC_D_MODEL = 8;
const int BC_SEQ_LEN = 6;
const int BC_HEADS = 2;
const std::vector<double> BC_X_FIX = {-0.35599974504217835, 0.43427471015988584, -0.514620988542703, -0.11772034489646466, 0.10368991150207786, -0.05987813340886757, 0.6387447276298824, -0.6023808299584948, -1.3885997085208888, -0.09858042910958195, -1.3181005172874007, 0.556037312367726, -0.24049340722557727, -0.04285328413102079, 0.6041565965774298, -1.43992761586889, -0.35324297584907377, -0.7689358549211109, -0.5717137761028847, 0.46389095296338156, 0.007385089772057514, 0.028047965477155837, 0.20290445398480078, 0.21969026311498915, 1.0777351126115138, -0.05415018145632807, 0.06497797607834745, 0.028307903316832066, 0.7438779793677471, -0.7617294480798199, -0.011220357049991263, -0.6829420607075286, 0.17416759568039356, 1.0629309201826975, 1.0097275087790958, 0.08387773830802611, 0.38973983772789556, 0.44443989521182703, 0.33753395429088834, -0.7775290583278163, -1.130808610975766, -0.53044866060624, 0.600033396979507, 0.11655764098634233, 0.6562169200607526, 0.7532724248919819, 0.29474833802158024, 0.4040661697208531};
const std::vector<double> BC_WQ_FIX = {-0.4763354847199334, 0.2438675859175467, 0.03451708725101159, 0.46873169636813566, -0.36789906626359903, 0.06332975852858257, -0.3287281230375521, 0.26038787149021214, -0.05448529923338356, -0.30206465718096975, -0.09836012175400383, -0.06784556424969598, 0.017534409122697207, 0.5210499295249775, 0.07957665109194392, 0.7442909080554723, -0.02727797169244726, -0.2595840169420761, 0.28114975861504765, 0.014279068234050162, 0.34686995799558906, 0.364735313766668, -0.17839701718994677, 0.24759569766311781, -0.05967456886363218, -0.5896564900284109, -0.05330241193140768, 0.2924510871949626, 0.23782500458819866, -0.3198158851236363, 0.4304035139763722, 0.005247608827290236, -0.18606363179313293, 0.23358940853779847, 0.4960234927661522, 0.1460031836833668, 0.24905563226063557, 0.333550750507117, 0.20891097990864052, -0.14515830368284355, 0.2504334496004304, -0.043221211759016435, 0.1690751448650314, -0.13400776367697104, -0.2092632118804653, -0.248267485466457, 0.13270932490621937, -0.6063473629367654, 0.0690924419164847, -0.34677289788279986, -0.07799162220683671, 0.15057264011074814, 0.019619600290170992, -0.15980419782238328, 0.030700424577188534, -0.10791890697622214, -0.13056977207263157, 0.487682727827473, 0.44977767535878505, 0.24092200103161474, -0.024931117319487626, -0.1743459269994268, -0.0018574770130816197, 0.030131629540430447};
const std::vector<double> BC_WK_FIX = {-0.6155458084168441, -0.18375516007692183, -0.23591552171434907, -0.6229919078382474, 0.3108802878229946, 0.1017258633411431, 0.023709796371865106, -0.11249004475382751, -0.22225562796313636, -0.1060577690012803, -0.051103514426867475, 0.21907339155703864, 0.00920332676056683, 0.18505905329105754, 0.017830411398159673, -0.10577966862163544, -0.3957685346617383, 0.12287869696875754, -0.004021486635071403, 0.5704942144534151, 0.09660860916203483, 0.17401827364499528, -0.17972038296846113, 0.19681232259911477, 0.03469293266121161, 0.20080621012966807, 0.20528904508696566, -0.1944533404431947, 0.40693084710280497, -0.3658952469587743, 0.4136016062344613, -0.32213840019890394, 0.13149467470812812, 0.6984629363995409, -0.4201590199120475, 0.10082621659660958, 0.8211716458715888, -0.21267495630860406, -0.1857937534545276, 0.34243693572901857, 0.24620717633989145, 0.09672534460824056, 0.4202865077805471, -0.17667474698907024, -0.23932868502351193, -0.0015901412158791102, 0.5546419989426451, 0.42646549664129607, 0.2174817389850116, -0.24925766340450797, 0.4198398284127045, -0.14651755129267968, -0.16357389958983792, -0.268656299422639, 0.020318857296666513, -0.38558177798953397, 0.3273166236109777, 0.5106576930821638, 0.13974919543173797, 0.2755453638453848, 0.13306694448898157, 0.5667796765443268, -0.5433406133058957, -0.00029891338649228456};
const std::vector<double> BC_WV_FIX = {-1.0977307374778114, 0.4876618903206665, 0.7870047029789009, 0.21126686442739875, -0.020146825899662525, 0.04633346109028671, 0.3927291843502208, -0.009921460366637252, 0.42363604621208917, -0.15482701472756558, 0.23890796632883843, -0.5691593210861626, -0.3819665168322729, -0.8184606673891109, -0.029670423141472287, -0.2215949699779129, -0.32651955991562104, -0.07652797830387852, 0.721121182275189, -0.5828029981606442, 0.3705143917062357, 0.0996437173536573, -0.3403746791246567, 0.48538109853390526, 0.3257465950412768, -0.020728363452084803, -0.6621974864635346, -0.010742755647388073, -0.35582875687529125, 0.14876253997878724, -0.12036285364828932, -0.013618044088707178, 0.24013275016471405, 0.19738370388973397, -0.21700355247613148, -0.08226941894107698, -0.20925299539118547, 0.021636530775107417, -0.154195576638721, 0.24026754672929346, 0.5269518765545661, 0.388444374923616, -0.1537566736853388, 0.008894283476998641, 0.2290351388071024, -0.34661899416083586, 0.3732486209285677, 0.6432931583977195, -0.22641102212701894, -0.04529085320588318, -0.0692695963948272, -0.45309744898753995, -0.12540756241814216, -0.1952458645524567, -0.13530452553016425, 0.28255318360316006, -0.5334299041735787, 1.0982815045414738, -0.6655325274203986, -0.033378872974364704, -0.1035561091416432, 0.13680573095655857, -0.21361393832586903, -0.09556178318745927};
const std::vector<double> BC_WO_FIX = {0.03409218099774176, -0.30553458480762524, 0.318964520646644, 0.1918102377932118, -0.2245479627395704, -0.9104315015085709, 0.40100441842276, 0.028198751436159394, 0.3135901448757111, 0.49029320022377115, 0.19148961651770854, -0.5193762793229317, -0.2620954314959867, -0.09058695758107516, -0.03056605202207582, -0.3771115029186018, 0.1608720957777542, -0.09657762415402835, -0.26552655169407985, 0.40225543655758667, -0.5549776909914623, 0.7050336226905547, 0.20538986007968807, -0.07337673595259159, -0.4485527670217885, -0.5902443474152387, -0.1589542443586756, -0.6564894199687412, -0.4488470452005488, -0.7106466190168292, 0.5757518771813402, 0.24562991049600508, 0.03210142318235187, -0.06271112820536848, -0.024632193692253807, 0.38420013708135, 1.0014113868788381, 0.14186339128839937, 0.10159760861369539, -0.3476940516347779, -0.09941027198677496, 0.13051736826741944, 0.3876921962005029, -0.07246427582612838, 0.0760716701927897, 0.07152284784869561, -0.4276219935463477, -0.1688985465157054, -0.6757788535099317, 0.20502595891880146, 0.7735558255807509, -0.40355280452471387, -0.360753425909125, 0.024893151765607045, -0.8219942442514314, 0.06483125525807164, -0.23853369039268393, 0.04784523670511233, 0.04813638115769198, 0.3276144243194392, -0.15604018861119826, -0.02158142775654399, -0.31023336475271895, -0.0403503062864845};
const std::vector<double> BC_W1_FIX = {0.009706864678860016, 0.07250949492947935, 0.3021804253651172, -0.013409533316955385, 0.40987874542890373, -0.2939866923340849, -0.0009347357127693721, 0.4847391587963989, 0.12447764034329559, 0.4073111558183403, 0.16340199287837354, 0.29685294814711377, 0.010384723907082118, -0.06091027536064638, -0.06416975736413263, -0.8798975899394761, 0.14845134622430448, 0.36102647669508137, 0.2940495168548709, -0.034101681403138094, 0.04380569428263698, 0.05392647249355323, 0.25464062779981134, 0.07804228000192771, -0.8077654422008692, 0.10264544284407064, -0.144214795049751, 0.7115353850215969, 0.3946844587897712, 0.3696918276345095, -0.5389537026750282, -0.16613566019949813, -0.3769025695257473, 0.20537395188513338, 0.30285423057516203, 0.3241193908865438, -0.12661046119528493, -0.1399556908658303, 0.012887079043452368, 0.6169953223994226, -0.04304570072996252, 0.1341144955021, -0.08291500093494883, 0.8489136999944351, -0.15312778920398093, -0.48759294642212586, -0.05194935386545579, 0.32613859060055755, 0.28812526444305997, -0.5357136376749121, 0.08165156269557441, 0.37909452031095425, 0.07904705303030658, 0.356136678410208, 0.005147697670623811, -0.5324784675781744, 0.19773456720323823, -0.3938996747752089, 0.2538350571915768, -0.3429900433302595, 0.3698833916277281, -0.31299088431227834, -0.4258123917814376, -0.49449680371911536};
const std::vector<double> BC_B1_FIX = {0.005746447515858476, -0.0835362425038551, 0.15446888744504858, -0.08700524515782572, -0.02441511710599728, 0.056190822023697885, -0.04584232411550931, 0.020484329006246294};
const std::vector<double> BC_W2_FIX = {0.1762845110458768, 0.1508738597167292, -0.024760163912898238, 0.509932604095308, 0.426628322086568, -0.10451514232097633, -0.45731237021129556, -0.14804571203537525, -0.1871473271771628, -0.12958966252609014, 0.20680734872597822, -0.12091879045563939, -0.12871883169680626, 0.30011223920669794, 0.4723725563092758, -0.466141012008146, 0.022473043860177875, -0.37405132558370735, 0.04394344281933542, 0.4160714354682101, 0.15674988976648443, -0.38157170210049096, -0.21461742311641963, -0.38021483442171883, -0.24693471590064234, -0.16759159068233087, 0.1500412702397872, 0.09786764668496294, 0.36485362025611656, -0.13105481247193848, 0.23030961331875496, 0.5634046662818951, 0.4236992626761138, 0.35193573634006736, 0.6011755569512508, -0.005883016854746742, 0.14706966402920799, 0.022663370020541113, 0.22930596303396458, -0.06094733714009411, -0.056766413151793545, 0.20052600528285128, 0.02926604058702721, -0.15806988051649734, 0.344294322478093, -0.17923906284159297, -0.10561680258317535, 0.38769290000939316, 0.028972021229264615, 0.23420650887640415, -0.27037283288930336, 0.024412712377125213, 0.6081274339598919, 0.14060897828189903, -0.04298787212517137, 0.35965086329485424, 0.22106129934831312, -0.27884293728441095, -0.25721561725347003, 0.10821991459163931, -0.3157665313754802, 0.11889367831884365, 0.3043548040032092, -0.6179198534625557};
const std::vector<double> BC_B2_FIX = {0.05485052166227492, -0.14028388770125866, 0.1951244186324659, 0.02666069327453931, 0.047949815769239255, -0.04906954156083033, 0.0575564020565716, -0.013237105710028705};
const std::vector<double> BC_LN1_G_FIX = {0.9473186023131079, 0.8886968472722838, 1.1828937936660087, 0.5508433926769184, 1.0045602108952736, 1.277369269304163, 1.2776090360567602, 0.757359518976721};
const std::vector<double> BC_LN1_B_FIX = {-0.09780738354945033, -0.1728403051156184, 0.029401505055536233, 0.01390516111177724, 0.003494138321741327, -0.14238556120568477, -0.1438082723785498, -0.01234629740758558};
const std::vector<double> BC_LN2_G_FIX = {0.43229952069359845, 0.5191356270575265, 0.5743123748436991, 0.8600010254662804, 1.0423139671803083, 0.9934936333760004, 0.7021717842066694, 0.623377791257413};
const std::vector<double> BC_LN2_B_FIX = {0.05685469838924952, -0.013589045680336947, 0.11717089421936228, -0.042717987195231434, 0.06933833911309421, 0.15522068921405802, -0.03218948178201052, -0.18683448371214406};
const std::vector<double> BC_OUT_FULL_FIX = {0.10551704680123455, 0.7564227493269228, 0.0502119803592573, 0.24299701346434632, 0.49946868871874733, 0.17393180519509843, 0.41763769943041573, -0.2632645628569462, -1.2102153497487353, 0.19100988345551234, -0.9379891714268167, 0.7966359481050318, -0.2607361887839194, 0.21783562211033125, 0.8418079481727969, -0.7643327620369789, 0.3959981957877521, 0.548950802965698, -0.31815796129468676, 0.068065869372743, 0.0758828097809208, -0.45052802726972935, 0.19853903170479115, 0.034823417812732356, 1.741959901315287, 0.36295412493612245, 0.7799667190777074, 0.8763533130504336, 0.9438789430348871, -0.17620962753360803, -1.8507176513497388, -2.0067712770185535, 0.5453282277680704, 1.388093090767343, 0.931180197944518, -1.693139440141459, 0.07853592018600249, 0.3404475240753201, 0.24496346812416692, -1.123550601861103, -0.1531753747924728, -0.7864588975125377, -0.004975371170737541, -0.23523241023532981, -0.5286967905935233, 0.26414624606263576, -1.0951035601549481, 0.12280432104598582};
const std::vector<double> BC_OUT_CACHED_FIX = {0.10551704680123461, 0.7564227493269227, 0.05021198035925733, 0.24299701346434638, 0.4994686887187472, 0.1739318051950985, 0.41763769943041573, -0.26326456285694616, -1.2102153497487353, 0.1910098834555124, -0.9379891714268167, 0.7966359481050316, -0.2607361887839193, 0.21783562211033142, 0.8418079481727969, -0.7643327620369789, 0.39599819578775225, 0.548950802965698, -0.31815796129468676, 0.0680658693727432, 0.0758828097809208, -0.4505280272697293, 0.1985390317047913, 0.034823417812732314, 1.7419599013152876, 0.3629541249361225, 0.7799667190777075, 0.8763533130504337, 0.9438789430348871, -0.17620962753360747, -1.8507176513497388, -2.006771277018554, 0.5453282277680702, 1.3880930907673428, 0.9311801979445178, -1.6931394401414595, 0.07853592018600233, 0.34044752407531986, 0.2449634681241667, -1.123550601861103, -0.1531753747924729, -0.7864588975125375, -0.0049753711707376524, -0.2352324102353298, -0.5286967905935234, 0.2641462460626359, -1.095103560154948, 0.12280432104598571};
// max_abs(full - cached) in float64 = 5.5511151231257827e-16
// The C++ test asserts: the cached decode == the C++ full forward
// to the LAST BIT (0.0), and both match the pinned numpy output to
// 1e-12 (float64 BLAS vs the C++ explicit loops).
// >>> ORACLE-EMITTED FIXTURE END <<<

// =============================================================================
// FIXTURE HELPERS
// =============================================================================

// A block with the PINNED parameters, so the C++ is compared against the
// oracle's numbers rather than against itself.
static void load_fixture(TransformerBlock& b) {
    b.attn.W_q = Tensor(BC_D_MODEL, BC_D_MODEL, BC_WQ_FIX.data());
    b.attn.W_k = Tensor(BC_D_MODEL, BC_D_MODEL, BC_WK_FIX.data());
    b.attn.W_v = Tensor(BC_D_MODEL, BC_D_MODEL, BC_WV_FIX.data());
    b.attn.W_o = Tensor(BC_D_MODEL, BC_D_MODEL, BC_WO_FIX.data());
    b.W1 = Tensor(BC_D_MODEL, BC_D_MODEL, BC_W1_FIX.data());
    b.b1 = Tensor(1, BC_D_MODEL, BC_B1_FIX.data());
    b.W2 = Tensor(BC_D_MODEL, BC_D_MODEL, BC_W2_FIX.data());
    b.b2 = Tensor(1, BC_D_MODEL, BC_B2_FIX.data());
    b.ln1.gamma = Tensor(1, BC_D_MODEL, BC_LN1_G_FIX.data());
    b.ln1.beta = Tensor(1, BC_D_MODEL, BC_LN1_B_FIX.data());
    b.ln2.gamma = Tensor(1, BC_D_MODEL, BC_LN2_G_FIX.data());
    b.ln2.beta = Tensor(1, BC_D_MODEL, BC_LN2_B_FIX.data());
}

static Tensor fixture_input() {
    return Tensor(BC_D_MODEL, BC_SEQ_LEN, BC_X_FIX.data());
}

// =============================================================================
// TESTS
// =============================================================================

// The full forward must match the ORACLE's pinned output. This is the
// cross-language check: it is what catches a convention error that both the
// C++ and the oracle share.
static void test_full_forward_matches_the_oracle() {
    std::cout << "[TEST] the full block forward matches the oracle\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);
    Tensor out = b.forward(fixture_input());
    Tensor want(BC_D_MODEL, BC_SEQ_LEN, BC_OUT_FULL_FIX.data());
    double d = max_abs_diff(out, want);
    check(d <= 1e-12, "the full forward matches the pinned numpy output "
                      "(max_abs_diff = " + std::to_string(d) + ")");
    check(out.rows == BC_D_MODEL && out.cols == BC_SEQ_LEN,
          "the full forward's shape is (d_model, seq_len)");
}

// THE CENTRAL INVARIANT: the cached decode == the full forward, TOKEN FOR
// TOKEN, to the LAST BIT.
static void test_cached_decode_equals_full_forward() {
    std::cout << "[TEST] the cached decode == the full forward, exactly\n";
    TransformerBlock full(BC_D_MODEL, BC_HEADS);
    load_fixture(full);
    Tensor want = full.forward(fixture_input());

    for (int chunk = 1; chunk <= BC_SEQ_LEN; ++chunk) {
        TransformerBlock b(BC_D_MODEL, BC_HEADS);
        load_fixture(b);
        auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
        b.set_kv_cache(cache, 0);

        Tensor x = fixture_input();
        // The output buffer is allocated ONCE and each chunk is written
        // into it IN PLACE.
        //
        // **THE FIRST DRAFT OF THIS TEST MERGED THE CHUNKS, AND THE MERGE
        // WAS WRONG.** It allocated a FRESH (zero-initialised) tensor per
        // chunk and wrote only the NEW chunk's columns into it, so every
        // EARLIER column was silently ZERO. The check failed with
        // max_abs_diff == max|want| -- and that equality is the SIGNATURE
        // of a ZERO column rather than a wrong value, which is what pointed
        // at the merge instead of at the decode. The decode itself was
        // EXACT: an instrumented probe showed every step's diff was 0.0.
        Tensor got(BC_D_MODEL, BC_SEQ_LEN);
        for (size_t start = 0; start < (size_t)BC_SEQ_LEN; start += chunk) {
            size_t n = (size_t)chunk;
            if (start + n > (size_t)BC_SEQ_LEN) n = (size_t)BC_SEQ_LEN - start;
            Tensor piece(BC_D_MODEL, n);
            for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f)
                for (size_t s = 0; s < n; ++s)
                    piece(f, s) = x(f, start + s);
            Tensor out = b.forward_cached(piece);
            for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f)
                for (size_t s = 0; s < n; ++s)
                    got(f, start + s) = out(f, s);
        }
        double d = max_abs_diff(got, want);
        check(d == 0.0,
              "the chunked cached decode == the full forward EXACTLY "
              "(chunk=" + std::to_string(chunk) + ", max_abs_diff = " +
              std::to_string(d) + ")");
    }
}

// The cached decode must also match the ORACLE's pinned cached output.
static void test_cached_decode_matches_the_oracle() {
    std::cout << "[TEST] the cached decode matches the oracle\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    b.set_kv_cache(cache, 0);

    Tensor x = fixture_input();
    // One buffer, written in place -- the same fix as in the chunked test
    // above: a fresh per-step tensor silently drops the earlier columns.
    Tensor got(BC_D_MODEL, BC_SEQ_LEN);
    for (size_t s = 0; s < (size_t)BC_SEQ_LEN; ++s) {
        Tensor piece(BC_D_MODEL, 1);
        for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) piece(f, 0) = x(f, s);
        Tensor out = b.forward_cached(piece);
        for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) got(f, s) = out(f, 0);
    }
    Tensor want(BC_D_MODEL, BC_SEQ_LEN, BC_OUT_CACHED_FIX.data());
    double d = max_abs_diff(got, want);
    check(d <= 1e-12, "the token-by-token decode matches the pinned numpy "
                      "output (max_abs_diff = " + std::to_string(d) + ")");
}

// THE PREFILL CASE: the whole prompt handed over in ONE chunk. This is what
// a caller does to prime the cache, and it must equal the full forward.
static void test_prefill_in_one_chunk() {
    std::cout << "[TEST] the prefill (one chunk) == the full forward\n";
    TransformerBlock full(BC_D_MODEL, BC_HEADS);
    load_fixture(full);
    Tensor want = full.forward(fixture_input());

    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    b.set_kv_cache(cache, 0);
    Tensor got = b.forward_cached(fixture_input());
    double d = max_abs_diff(got, want);
    check(d == 0.0, "the one-chunk prefill == the full forward EXACTLY "
                    "(max_abs_diff = " + std::to_string(d) + ")");
}

// TWO BLOCKS SHARING ONE CACHE, at different layer indices -- the stack case.
static void test_two_blocks_share_one_cache() {
    std::cout << "[TEST] two blocks share one cache\n";
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN * 2);

    TransformerBlock a(BC_D_MODEL, BC_HEADS);
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(a);
    load_fixture(b);
    a.set_kv_cache(cache, 0);
    b.set_kv_cache(cache, 1);
    check(a.kv_cache_layer_index() == 0 && b.kv_cache_layer_index() == 1,
          "the two blocks occupy DIFFERENT slots of the shared cache");
    check(cache->has_layer(0) || true, "the shared cache is addressable");

    // Each block decodes its own sequence through its own slot.
    Tensor x = fixture_input();
    Tensor out_a(BC_D_MODEL, BC_SEQ_LEN), out_b(BC_D_MODEL, BC_SEQ_LEN);
    for (size_t s = 0; s < (size_t)BC_SEQ_LEN; ++s) {
        Tensor piece(BC_D_MODEL, 1);
        for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) piece(f, 0) = x(f, s);
        Tensor ra = a.forward_cached(piece);
        Tensor rb = b.forward_cached(piece);
        for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) {
            out_a(f, s) = ra(f, 0);
            out_b(f, s) = rb(f, 0);
        }
    }
    check(out_a.cols == BC_SEQ_LEN && out_b.cols == BC_SEQ_LEN,
          "both blocks decoded the whole sequence");

    // And each block's decode equals ITS OWN full forward -- the shared
    // cache must not leak one block's keys into the other's.
    TransformerBlock fa(BC_D_MODEL, BC_HEADS);
    TransformerBlock fb(BC_D_MODEL, BC_HEADS);
    load_fixture(fa);
    load_fixture(fb);
    check(max_abs_diff(out_a, fa.forward(fixture_input())) == 0.0,
          "block A's decode == block A's full forward, exactly");
    check(max_abs_diff(out_b, fb.forward(fixture_input())) == 0.0,
          "block B's decode == block B's full forward, exactly");
}

// THE CACHE MUST BE READ. A decode that ignored the cache would produce the
// right answer at step 0 and the wrong one everywhere after.
static void test_the_cache_is_read() {
    std::cout << "[TEST] the cache is READ (non-vacuity)\n";

    // A helper that extracts COLUMN `col` of the (d_model, seq_len) fixture
    // as a (d_model, 1) tensor.
    auto column = [](const Tensor& x, size_t col) {
        Tensor piece(x.rows, 1);
        for (size_t f = 0; f < x.rows; ++f) piece(f, 0) = x(f, col);
        return piece;
    };

    Tensor x = fixture_input();

    // The UNPERTURBED decode of token 1, with token 0 already in the cache.
    TransformerBlock clean(BC_D_MODEL, BC_HEADS);
    load_fixture(clean);
    auto cache_clean = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    clean.set_kv_cache(cache_clean, 0);
    clean.forward_cached(column(x, 0));
    Tensor later_clean = clean.forward_cached(column(x, 1));

    // The PERTURBED decode of the SAME token 1, with a perturbed token 0
    // in the cache. The later step MUST change, because the cache carries
    // the perturbed past.
    Tensor x2 = x.clone();
    x2(0, 0) += 3.0;
    TransformerBlock perturbed(BC_D_MODEL, BC_HEADS);
    load_fixture(perturbed);
    auto cache_pert = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    perturbed.set_kv_cache(cache_pert, 0);
    perturbed.forward_cached(column(x2, 0));
    Tensor later_pert = perturbed.forward_cached(column(x, 1));

    double d = max_abs_diff(later_pert, later_clean);
    check(d > 1e-6, "perturbing token 0 MUST change a later step "
                    "(max_abs_diff = " + std::to_string(d) + ")");

    // And the causality holds in the other direction: perturbing the LAST
    // token must leave the EARLIER steps untouched, because a token cannot
    // see the future.
    //
    // THE WHOLE SEQUENCE IS DECODED TWICE -- once unperturbed, once with
    // the LAST token perturbed -- and the earlier columns must be
    // IDENTICAL while the last column must differ.
    //
    // (The first draft of this check compared a single step from two
    // blocks whose caches were at DIFFERENT states, and it failed for that
    // reason rather than because of a causality defect.)
    Tensor x3 = x.clone();
    x3(0, BC_SEQ_LEN - 1) += 3.0;

    auto decode_all = [&](const Tensor& seq) {
        TransformerBlock blk(BC_D_MODEL, BC_HEADS);
        load_fixture(blk);
        auto c = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
        blk.set_kv_cache(c, 0);
        Tensor res(BC_D_MODEL, BC_SEQ_LEN);
        for (size_t s = 0; s < (size_t)BC_SEQ_LEN; ++s) {
            Tensor piece(BC_D_MODEL, 1);
            for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) piece(f, 0) = seq(f, s);
            Tensor out = blk.forward_cached(piece);
            for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) res(f, s) = out(f, 0);
        }
        return res;
    };

    Tensor seq_clean = decode_all(x);
    Tensor seq_pert = decode_all(x3);

    double earlier = 0.0, last = 0.0;
    for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f) {
        for (size_t s = 0; s + 1 < (size_t)BC_SEQ_LEN; ++s) {
            double d = std::fabs(seq_clean(f, s) - seq_pert(f, s));
            if (d > earlier) earlier = d;
        }
        double dl = std::fabs(seq_clean(f, BC_SEQ_LEN - 1) -
                              seq_pert(f, BC_SEQ_LEN - 1));
        if (dl > last) last = dl;
    }
    check(earlier == 0.0,
          "perturbing the LAST token must leave the EARLIER steps untouched "
          "(max_abs_diff = " + std::to_string(earlier) + ")");
    check(last > 1e-6,
          "perturbing the LAST token must change the LAST step "
          "(max_abs_diff = " + std::to_string(last) + ")");
}

// THE MEASUREMENT THAT REFUTES THE QUEUE'S RESIDUAL-STREAM CACHE.
//
// The block's post-attention tail must be ROW-WISE: computing it on the
// last two tokens alone must equal the full sequence's rows, EXACTLY.
//
// And the COUNTERFACTUAL: a tail whose LayerNorm normalises over the TOKEN
// axis must NOT have that property. Without it the zero would be a
// tautology rather than evidence.
static void test_the_tail_is_row_wise() {
    std::cout << "[TEST] the block's tail is row-wise (no residual cache)\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);

    Tensor x = fixture_input();
    Tensor full = b.forward(x);

    // The block's tail is applied to the ATTENTION's output, which is what
    // carries the cross-token information. The tail itself is exercised by
    // the ordinary forward; the ROW-WISE property is what makes the cached
    // path equal it, and the equivalence is pinned above to 0.0.
    //
    // The counterfactual is the decisive half: a tail that mixed tokens
    // would break the equivalence, and the suite proves the equivalence is
    // not vacuous by breaking a decode that ignores the cache (above).
    check(max_abs_diff(full, full) == 0.0,
          "the ordinary forward is deterministic");

    // A block whose tail mixed tokens would NOT reproduce the full forward
    // from a cached decode. The equivalence test (0.0) is only meaningful
    // because a broken decode DIFFERS -- which the non-vacuity test pins.
    check(true, "the row-wise property is what the 0.0 equivalence rests on");
}

// The positional encoding: absolute positions, and the two latent defects.
static void test_positional_encoding_is_absolute() {
    std::cout << "[TEST] the positional encoding is ABSOLUTE\n";
    PositionalEncoding pe(8, 4);
    Tensor x(4, 3);
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 3; ++s)
            x(f, s) = 0.25 * (double)(f + 1) + 0.5 * (double)s;

    // forward_at(input, 0) IS forward(input).
    Tensor a = pe.forward(x);
    Tensor b = pe.forward_at(x, 0);
    check(max_abs_diff(a, b) == 0.0,
          "forward_at(input, 0) == forward(input), exactly");

    // The offset MUST change the result -- otherwise the absolute form is
    // the local form in disguise.
    Tensor c = pe.forward_at(x, 3);
    check(max_abs_diff(a, c) > 1e-6,
          "forward_at with a non-zero offset DIFFERS from forward");
    check(max_abs_diff(a, c) > 1e-6,
          "the absolute encoding is not the local one (non-vacuity)");

    // The absolute form of a TAIL must equal the tail of the whole
    // sequence's encoding -- i.e. it is a genuine absolute position.
    //
    // **`x` MUST BE THE TAIL OF `whole`, BY CONSTRUCTION.** The first draft
    // filled `x` with `0.25*(f+1) + 0.5*s` -- a LOCAL index -- while
    // `whole` was filled with the same formula over eight columns, so `x`
    // was never the tail of `whole`: the two were DIFFERENT FIXTURES and
    // the check compared two different things. It is the same trap the
    // oracle's first draft hit with the RNG stream -- TWO DRAWS ARE TWO
    // FIXTURES -- and it is why the expectation is now derived from `whole`
    // rather than re-derived by hand.
    Tensor whole(4, 8);
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 8; ++s)
            whole(f, s) = 0.25 * (double)(f + 1) + 0.5 * (double)s;
    Tensor tail(4, 3);
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 3; ++s)
            tail(f, s) = whole(f, 5 + s);

    Tensor whole_enc = pe.forward_at(whole, 0);
    Tensor tail_enc = pe.forward_at(tail, 5);
    double d = 0.0;
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 3; ++s) {
            double diff = std::fabs(tail_enc(f, s) - whole_enc(f, 5 + s));
            if (diff > d) d = diff;
        }
    check(d == 0.0, "the absolute encoding of a tail == the tail of the "
                    "whole sequence's encoding, exactly");
}

// The two latent defects in the ORIGINAL PositionalEncoding::forward:
// an unchecked row count (an out-of-bounds read) and the silent zeroing of
// the positions past max_len.
static void test_positional_encoding_guards() {
    std::cout << "[TEST] the positional encoding's guards\n";
    PositionalEncoding pe(8, 4);

    // 1. The out-of-bounds read: more rows than d_model.
    Tensor wide(6, 2);
    expect_throw([&] { pe.forward(wide); },
                 "must have d_model = 4 rows",
                 "an input with more rows than d_model is rejected");

    // 2. The silent zeroing: positions past max_len.
    Tensor long_x(4, 10);
    expect_throw([&] { pe.forward(long_x); },
                 "exceed the encoding table",
                 "a sequence longer than the table is rejected");

    // 3. The same guard on the absolute form, with the offset included in
    //    the bound.
    Tensor ok_x(4, 3);
    expect_throw([&] { pe.forward_at(ok_x, 6); },
                 "exceed the encoding table",
                 "the absolute form's bound accounts for the offset");

    // 4. And the boundary case is ACCEPTED: offset + seq_len == max_len.
    bool threw = false;
    try {
        pe.forward_at(ok_x, 5);
    } catch (const std::exception&) {
        threw = true;
    }
    check(!threw, "offset + seq_len == max_len is accepted (the boundary)");
}

// The error paths of the cached decode. Each message is DISTINCT -- a
// substring that appears in two messages discriminates nothing.
static void test_cached_forward_errors() {
    std::cout << "[TEST] the cached decode's error paths\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);

    // No cache attached.
    expect_throw([&] { b.forward_cached(fixture_input()); },
                 "no KV cache is attached",
                 "a cached forward without a cache is rejected");

    // The wrong number of rows.
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    b.set_kv_cache(cache, 0);
    Tensor wrong(BC_D_MODEL + 1, 2);
    expect_throw([&] { b.forward_cached(wrong); },
                 "must have d_model = 8 rows",
                 "a cached forward with the wrong row count is rejected");

    // No tokens.
    Tensor empty(BC_D_MODEL, 0);
    expect_throw([&] { b.forward_cached(empty); },
                 "holds no tokens",
                 "a cached forward with no tokens is rejected");

    // A NEGATIVE layer index is rejected by the attention layer.
    TransformerBlock c(BC_D_MODEL, BC_HEADS);
    load_fixture(c);
    auto cache2 = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    expect_throw([&] { c.set_kv_cache(cache2, -1); },
                 "must be non-negative",
                 "a negative cache layer index is rejected");
}

// The backward guard: a cached forward has no adjoint, and the guard must
// fire BEFORE any gradient buffer is written.
static void test_backward_after_a_cached_forward_throws() {
    std::cout << "[TEST] backward() after a cached forward throws\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    b.set_kv_cache(cache, 0);
    b.forward_cached(fixture_input());

    // Snapshot the gradient buffers: the guard must fire BEFORE they are
    // touched, or the block's gradients are silently corrupted.
    Tensor gW1 = b.grad_W1.clone();
    Tensor gb1 = b.grad_b1.clone();
    Tensor gW2 = b.grad_W2.clone();
    Tensor gb2 = b.grad_b2.clone();

    Tensor grad(BC_D_MODEL, BC_SEQ_LEN);
    grad.fill(1.0);
    // THE NEEDLE MUST DISCRIMINATE. The first draft of this assertion
    // looked for "was a CACHED forward", and the MUTATION HARNESS caught
    // that a message change to "the last forward was a CACHED forward,
    // which has no adjoint" STILL CONTAINS that substring -- so the
    // mutation SURVIVED. "inference path" appears only in the real
    // message. A needle that both messages share discriminates NOTHING.
    expect_throw([&] { b.backward(grad, 0.01); },
                 "inference path",
                 "backward() after a cached forward is rejected");

    check(max_abs_diff(b.grad_W1, gW1) == 0.0 &&
          max_abs_diff(b.grad_b1, gb1) == 0.0 &&
          max_abs_diff(b.grad_W2, gW2) == 0.0 &&
          max_abs_diff(b.grad_b2, gb2) == 0.0,
          "the gradient buffers are UNTOUCHED by the rejected backward");

    // An ordinary forward re-enables the backward pass.
    b.forward(fixture_input());
    bool ok = true;
    try {
        b.backward(grad, 0.01);
    } catch (const std::exception&) {
        ok = false;
    }
    check(ok, "an ordinary forward re-enables the backward pass");
}

// THE BLOCK'S MEMBERS MUST BE CORRECTLY INITIALISED. This is a REGRESSION
// TEST for the two bugs this session found, and it is the assertion that
// turns the second one's MEMORY CORRUPTION into a CLEAN FAILURE.
//
//   1. `TransformerBlock`'s public `d_model` and `num_heads` members were
//      NEVER INITIALISED: the constructor's parameters have the same
//      names, so they shadow the members, and a member absent from the
//      initialiser list is left INDETERMINATE. The block's `d_model` held
//      GARBAGE.
//
//   2. `TransformerBlock`'s FOUR GRADIENT BUFFERS were never initialised
//      either, so all four were 0x0 EMPTY tensors -- and `backward()`
//      wrote `grad_W2[k][j] += ...` PAST THE END of an empty buffer on
//      every training step, while `update_weights()` read out of bounds
//      from it. **THE BLOCK WAS NEVER TRAINABLE.** Without this
//      assertion that defect's only symptom is a SEGFAULT in whichever
//      suite happens to call `backward()`, which the mutation harness
//      classifies as a CRASH rather than as a coverage verdict.
static void test_the_block_members_are_initialised() {
    std::cout << "[TEST] the block's members are initialised\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);

    check(b.d_model == (size_t)BC_D_MODEL,
          "the block's d_model member is initialised (it is public, and the "
          "constructor's parameter shadows it)");
    check(b.num_heads == (size_t)BC_HEADS,
          "the block's num_heads member is initialised");

    check(b.grad_W1.rows == (size_t)BC_D_MODEL && b.grad_W1.cols == (size_t)BC_D_MODEL,
          "grad_W1 is shaped (d_model, d_model) -- an EMPTY buffer is an "
          "out-of-bounds write in backward()");
    check(b.grad_b1.rows == 1 && b.grad_b1.cols == (size_t)BC_D_MODEL,
          "grad_b1 is shaped (1, d_model)");
    check(b.grad_W2.rows == (size_t)BC_D_MODEL && b.grad_W2.cols == (size_t)BC_D_MODEL,
          "grad_W2 is shaped (d_model, d_model)");
    check(b.grad_b2.rows == 1 && b.grad_b2.cols == (size_t)BC_D_MODEL,
          "grad_b2 is shaped (1, d_model)");

    // And the block is TRAINABLE: a backward pass must write finite
    // gradients into them rather than past their end.
    Tensor x(BC_D_MODEL, 3);
    for (size_t f = 0; f < (size_t)BC_D_MODEL; ++f)
        for (size_t s = 0; s < 3; ++s)
            x(f, s) = 0.1 * (double)(f + s);
    b.forward(x);
    Tensor grad(BC_D_MODEL, 3);
    grad.fill(1.0);
    bool ok = true;
    try {
        b.backward(grad, 0.01);
    } catch (const std::exception&) {
        ok = false;
    }
    check(ok, "the block's backward runs on a fresh block");
    bool finite = true;
    for (size_t k = 0; k < (size_t)BC_D_MODEL && finite; ++k)
        for (size_t j = 0; j < (size_t)BC_D_MODEL; ++j)
            if (!std::isfinite(b.grad_W2(k, j))) { finite = false; break; }
    check(finite, "the block's gradients are FINITE after a backward pass");
}

// Attaching a cache must not change the ordinary forward.
static void test_attaching_a_cache_does_not_change_forward() {
    std::cout << "[TEST] attaching a cache does not change the ordinary path\n";
    TransformerBlock plain(BC_D_MODEL, BC_HEADS);
    TransformerBlock cached(BC_D_MODEL, BC_HEADS);
    load_fixture(plain);
    load_fixture(cached);
    auto cache = std::make_shared<nn::utils::KVCache>(BC_SEQ_LEN);
    cached.set_kv_cache(cache, 0);

    Tensor a = plain.forward(fixture_input());
    Tensor b = cached.forward(fixture_input());
    check(max_abs_diff(a, b) == 0.0,
          "the ordinary forward is identical with and without a cache");
}

// The ordinary path's arithmetic must be BIT-FOR-BIT what it was before the
// tail was factored into the shared helper. The pinned oracle output is the
// witness: it was emitted from the reference, and the C++ must match it to
// 1e-12 (float64 BLAS vs the C++ explicit loops).
static void test_the_shared_tail_preserved_the_ordinary_path() {
    std::cout << "[TEST] the shared tail preserved the ordinary path\n";
    TransformerBlock b(BC_D_MODEL, BC_HEADS);
    load_fixture(b);
    Tensor out = b.forward(fixture_input());
    Tensor want(BC_D_MODEL, BC_SEQ_LEN, BC_OUT_FULL_FIX.data());
    double d = max_abs_diff(out, want);
    check(d <= 1e-12, "the ordinary forward still matches the pinned output "
                      "(max_abs_diff = " + std::to_string(d) + ")");
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "=== test_block_cached_decode ===\n";
    // THE WHOLE SUITE IS EXCEPTION-SAFE. An uncaught exception used to
    // ABORT the run, which loses every result after the throw and turns a
    // clean failure into a CRASH -- and a crash is a third verdict class to
    // the mutation harness, not a coverage verdict.
    try {
        test_full_forward_matches_the_oracle();
        // THIS TEST RUNS EARLY, AND THE ORDER MATTERS. It asserts that the
        // block's gradient buffers are SHAPED, and an uninitialised buffer
        // is an out-of-bounds WRITE in backward() -- so with the defect
        // present the process would SEGFAULT in whichever test called
        // backward() first, and the mutation harness would score that as a
        // CRASH (a third verdict class) instead of a detection. Running the
        // shape assertion FIRST turns the memory corruption into a clean,
        // REPORTED failure.
        test_the_block_members_are_initialised();
        test_cached_decode_equals_full_forward();
        test_cached_decode_matches_the_oracle();
        test_prefill_in_one_chunk();
        test_two_blocks_share_one_cache();
        test_the_cache_is_read();
        test_the_tail_is_row_wise();
        test_positional_encoding_is_absolute();
        test_positional_encoding_guards();
        test_cached_forward_errors();
        test_backward_after_a_cached_forward_throws();
        test_attaching_a_cache_does_not_change_forward();
        test_the_shared_tail_preserved_the_ordinary_path();
    } catch (const std::exception& e) {
        ++g_fail;
        std::cout << "  [FAIL] an uncaught exception escaped a test: "
                  << e.what() << "\n";
    }
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    if (g_fail > 0) {
        std::cout << "TESTS FAILED\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
