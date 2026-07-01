// test_transformer_decoder.cpp — Gradient correctness tests for the
// Encoder-Decoder Transformer (Vaswani et al. 2017 §3.2).
//
// Tests cover:
//  1. Masked self-attention forward shape + causal-mask invariant
//  2. Masked self-attention numerical gradient (input)
//  3. Masked self-attention numerical gradient (W_q, W_k, W_v, W_o)
//  4. Cross-attention forward shape (Q from one source, K/V from another)
//  5. Cross-attention numerical gradient on Q source
//  6. Cross-attention numerical gradient on K/V source
//  7. Cross-attention numerical gradient on W_o
//  8. Decoder block forward shape + finite output
//  9. Decoder block numerical gradient on decoder input
// 10. Decoder block numerical gradient on encoder input (via cross-attn)
// 11. Full TransformerDecoder end-to-end training step reduces loss
// 12. Cross-attention with mismatched seq lengths (n_q != n_kv)
// 13. Cross-attention output differs from masked self-attention on same input
//     (sanity: confirms Q and K/V are actually coming from different sources)
// 14. Decoder block parameter / gradient count consistency
// 15. Causal mask zeroes gradient at forbidden positions (future keys cannot
//     leak info back through backward)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include "nn/layers/attention/transformer_decoder.h"
#include "nn/core/tensor.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8);
}

static Tensor make_random_input(size_t rows, size_t cols, double scale = 0.5, int seed = 7) {
    srand(seed);
    Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            t[i][j] = ((rand() % 1000) / 500.0 - 1.0) * scale;
    return t;
}

// =====================================================================
// Test 1: Masked self-attention forward shape + causal mask invariant
// =====================================================================
static void test_masked_self_attn_forward() {
    cout << endl << "-- Test 1: MaskedMultiHeadSelfAttention forward --" << endl;
    size_t d_model = 8, num_heads = 2, n = 4;
    MaskedMultiHeadSelfAttention sa(d_model, num_heads);

    Tensor input = make_random_input(n, d_model, 0.5, 11);
    Tensor out = sa.forward(input);

    check("MaskedSelfAttn output shape (n, d_model)", out.rows == n && out.cols == d_model);
    check("MaskedSelfAttn output finite",
          std::isfinite(out[0][0]) && std::isfinite(out[n-1][d_model-1]));

    // Inspect cached attn weights: row i must be all zero for j > i
    bool causal_ok = true;
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double a = sa.last_attn_(h * n + i, j);  // (num_heads, n_q, n_kv)
                if (j > i && std::abs(a) > 1e-7) causal_ok = false;
            }
        }
    }
    check("Causal mask: future positions have zero attention weight", causal_ok);
}

// =====================================================================
// Test 2: Masked self-attention numerical gradient (input)
// =====================================================================
static void test_masked_self_attn_input_gradient() {
    cout << endl << "-- Test 2: MaskedMultiHeadSelfAttention input gradient --" << endl;
    size_t d_model = 8, num_heads = 2, n = 3;
    MaskedMultiHeadSelfAttention sa(d_model, num_heads);

    Tensor input = make_random_input(n, d_model, 0.5, 21);

    sa.zero_grad();
    Tensor out = sa.forward(input);
    Tensor grad_out(n, d_model);
    grad_out.fill(1.0);
    Tensor grad_in = sa.backward(grad_out, 0.0);

    double eps = 1e-4;
    int ri = 1, ci = 2;
    double orig = input[ri][ci];
    double ana = grad_in[ri][ci];

    input[ri][ci] = orig + eps;
    sa.zero_grad();
    Tensor out_plus = sa.forward(input);
    double loss_plus = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model; ++j)
            loss_plus += out_plus[i][j];

    input[ri][ci] = orig - eps;
    sa.zero_grad();
    Tensor out_minus = sa.forward(input);
    double loss_minus = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model; ++j)
            loss_minus += out_minus[i][j];

    input[ri][ci] = orig;  // restore
    double num = (loss_plus - loss_minus) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num
         << " eps=" << eps << " loss+=" << loss_plus << " loss-=" << loss_minus
         << " diff=" << (loss_plus - loss_minus) << endl;
    check("MaskedSelfAttn input grad rel_err < 1e-4", rel < 1e-4);
}
// Test 3: Masked self-attention numerical gradient on W_q/W_k/W_v/W_o
// =====================================================================
static void test_masked_self_attn_param_gradients() {
    cout << endl << "-- Test 3: MaskedMultiHeadSelfAttention parameter gradients --" << endl;
    size_t d_model = 6, num_heads = 2, n = 3;
    MaskedMultiHeadSelfAttention sa(d_model, num_heads);

    Tensor input = make_random_input(n, d_model, 0.5, 31);

    sa.zero_grad();
    Tensor out = sa.forward(input);
    Tensor grad_out(n, d_model);
    grad_out.fill(1.0);
    sa.backward(grad_out, 0.0);

    // Check each of W_q, W_k, W_v, W_o on one entry
    struct Entry { Tensor* W; Tensor* gradW; const char* name; int r, c; };
    std::vector<Entry> entries = {
        {&sa.W_q, &sa.grad_W_q, "W_q", 1, 2},
        {&sa.W_k, &sa.grad_W_k, "W_k", 0, 3},
        {&sa.W_v, &sa.grad_W_v, "W_v", 2, 1},
        {&sa.W_o, &sa.grad_W_o, "W_o", 3, 4},
    };

    double eps = 1e-5;
    bool all_ok = true;
    for (auto& e : entries) {
        double ana = (*e.gradW)[e.r][e.c];
        double orig = (*e.W)[e.r][e.c];
        (*e.W)[e.r][e.c] = orig + eps;
        Tensor out_plus = sa.forward(input);
        double lp = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                lp += out_plus[i][j];
        (*e.W)[e.r][e.c] = orig - eps;
        Tensor out_minus = sa.forward(input);
        double lm = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model; ++j)
                lm += out_minus[i][j];
        (*e.W)[e.r][e.c] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = rel_error(num, ana);
        cout << "    [DEBUG] " << e.name << "[" << e.r << "," << e.c
             << "] ana=" << ana << " num=" << num << " rel=" << rel << endl;
        if (rel > 1e-3) all_ok = false;
    }
    check("MaskedSelfAttn all 4 W grads rel_err < 1e-3", all_ok);
}

// =====================================================================
// Test 4: Cross-attention forward shape
// =====================================================================
static void test_cross_attn_forward_shape() {
    cout << endl << "-- Test 4: MultiHeadCrossAttention forward --" << endl;
    size_t d_model = 8, num_heads = 2, n_q = 3, n_kv = 5;
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.4, 41);
    Tensor enc = make_random_input(n_kv, d_model, 0.4, 42);

    Tensor out;
    ca.forward_two(dec, enc, out);

    check("CrossAttn output shape (n_q, d_model)", out.rows == n_q && out.cols == d_model);
    check("CrossAttn output finite",
          std::isfinite(out[0][0]) && std::isfinite(out[n_q-1][d_model-1]));

    // Attention rows must sum to 1 (softmax invariant)
    bool rows_sum_one = true;
    for (size_t h = 0; h < num_heads; ++h)
        for (size_t i = 0; i < n_q; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < n_kv; ++j)
                s += ca.last_attn_(h * n_q + i, j);
            if (std::abs(s - 1.0) > 1e-5) rows_sum_one = false;
        }
    check("CrossAttn attention rows sum to 1", rows_sum_one);
}

// =====================================================================
// Test 5: Cross-attention numerical gradient on decoder (Q source)
// =====================================================================
static void test_cross_attn_decoder_gradient() {
    cout << endl << "-- Test 5: CrossAttn gradient on Q source (decoder) --" << endl;
    size_t d_model = 6, num_heads = 2, n_q = 3, n_kv = 4;
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.3, 51);
    Tensor enc = make_random_input(n_kv, d_model, 0.3, 52);

    Tensor out;
    ca.forward_two(dec, enc, out);
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);

    Tensor gd, ge;
    ca.backward_two(grad_out, gd, ge);

    double eps = 1e-4;
    int ri = 1, ci = 2;
    double ana = gd[ri][ci];
    double orig = dec[ri][ci];

    dec[ri][ci] = orig + eps;
    Tensor out_p;
    ca.forward_two(dec, enc, out_p);
    double lp = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j)
            lp += out_p[i][j];

    dec[ri][ci] = orig - eps;
    Tensor out_m;
    ca.forward_two(dec, enc, out_m);
    double lm = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j)
            lm += out_m[i][j];
    dec[ri][ci] = orig;
    double num = (lp - lm) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num << endl;
    check("CrossAttn Q-source grad rel_err < 1e-4", rel < 1e-4);
}

// =====================================================================
// Test 6: Cross-attention numerical gradient on encoder (K/V source)
// =====================================================================
static void test_cross_attn_encoder_gradient() {
    cout << endl << "-- Test 6: CrossAttn gradient on KV source (encoder) --" << endl;
    size_t d_model = 6, num_heads = 2, n_q = 3, n_kv = 4;
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.3, 61);
    Tensor enc = make_random_input(n_kv, d_model, 0.3, 62);

    Tensor out;
    ca.forward_two(dec, enc, out);
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);

    Tensor gd, ge;
    ca.backward_two(grad_out, gd, ge);

    double eps = 1e-4;
    int ri = 2, ci = 1;
    double ana = ge[ri][ci];
    double orig = enc[ri][ci];

    enc[ri][ci] = orig + eps;
    Tensor out_p;
    ca.forward_two(dec, enc, out_p);
    double lp = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j)
            lp += out_p[i][j];

    enc[ri][ci] = orig - eps;
    Tensor out_m;
    ca.forward_two(dec, enc, out_m);
    double lm = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j)
            lm += out_m[i][j];
    enc[ri][ci] = orig;
    double num = (lp - lm) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num << endl;
    check("CrossAttn KV-source grad rel_err < 1e-4", rel < 1e-4);
}

// =====================================================================
// Test 7: Cross-attention numerical gradient on W_o
// =====================================================================
static void test_cross_attn_W_o_gradient() {
    cout << endl << "-- Test 7: CrossAttn gradient on W_o --" << endl;
    size_t d_model = 6, num_heads = 2, n_q = 3, n_kv = 4;
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.3, 71);
    Tensor enc = make_random_input(n_kv, d_model, 0.3, 72);

    Tensor out;
    ca.forward_two(dec, enc, out);
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);
    Tensor gd, ge;
    ca.backward_two(grad_out, gd, ge);

    double eps = 1e-5;
    int r = 1, c = 3;
    double ana = ca.grad_W_o[r][c];
    double orig = ca.W_o[r][c];
    ca.W_o[r][c] = orig + eps;
    Tensor out_p;
    ca.forward_two(dec, enc, out_p);
    double lp = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lp += out_p[i][j];
    ca.W_o[r][c] = orig - eps;
    Tensor out_m;
    ca.forward_two(dec, enc, out_m);
    double lm = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lm += out_m[i][j];
    ca.W_o[r][c] = orig;
    double num = (lp - lm) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num << endl;
    check("CrossAttn W_o grad rel_err < 1e-4", rel < 1e-4);
}

// =====================================================================
// Test 8: Decoder block forward shape + finite output
// =====================================================================
static void test_decoder_block_forward() {
    cout << endl << "-- Test 8: TransformerDecoderBlock forward --" << endl;
    size_t d_model = 8, num_heads = 2, n_q = 3, n_kv = 5;
    TransformerDecoderBlock block(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.3, 81);
    Tensor enc = make_random_input(n_kv, d_model, 0.3, 82);

    Tensor out;
    block.forward_two(dec, enc, out);

    check("DecoderBlock output shape (n_q, d_model)", out.rows == n_q && out.cols == d_model);
    check("DecoderBlock output finite",
          std::isfinite(out[0][0]) && std::isfinite(out[n_q-1][d_model-1]));
}

// =====================================================================
// Test 9: Decoder block numerical gradient on decoder input
// =====================================================================
static void test_decoder_block_decoder_gradient() {
    cout << endl << "-- Test 9: DecoderBlock gradient on decoder input --" << endl;
    size_t d_model = 6, num_heads = 2, n_q = 3, n_kv = 4;
    TransformerDecoderBlock block(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.2, 91);
    Tensor enc = make_random_input(n_kv, d_model, 0.2, 92);

    Tensor out;
    block.forward_two(dec, enc, out);
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);

    Tensor gd, ge;
    block.backward_two(grad_out, gd, ge);

    double eps = 1e-4;
    int ri = 1, ci = 2;
    double ana = gd[ri][ci];
    double orig = dec[ri][ci];

    dec[ri][ci] = orig + eps;
    Tensor out_p;
    block.forward_two(dec, enc, out_p);
    double lp = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lp += out_p[i][j];

    dec[ri][ci] = orig - eps;
    Tensor out_m;
    block.forward_two(dec, enc, out_m);
    double lm = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lm += out_m[i][j];
    dec[ri][ci] = orig;
    double num = (lp - lm) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num << endl;
    check("DecoderBlock decoder-input grad rel_err < 1e-3", rel < 1e-3);
}

// =====================================================================
// Test 10: Decoder block numerical gradient on encoder input
// =====================================================================
static void test_decoder_block_encoder_gradient() {
    cout << endl << "-- Test 10: DecoderBlock gradient on encoder input --" << endl;
    size_t d_model = 6, num_heads = 2, n_q = 3, n_kv = 4;
    TransformerDecoderBlock block(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.2, 101);
    Tensor enc = make_random_input(n_kv, d_model, 0.2, 102);

    Tensor out;
    block.forward_two(dec, enc, out);
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);

    Tensor gd, ge;
    block.backward_two(grad_out, gd, ge);

    double eps = 1e-4;
    int ri = 2, ci = 1;
    double ana = ge[ri][ci];
    double orig = enc[ri][ci];

    enc[ri][ci] = orig + eps;
    Tensor out_p;
    block.forward_two(dec, enc, out_p);
    double lp = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lp += out_p[i][j];

    enc[ri][ci] = orig - eps;
    Tensor out_m;
    block.forward_two(dec, enc, out_m);
    double lm = 0.0;
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < d_model; ++j) lm += out_m[i][j];
    enc[ri][ci] = orig;
    double num = (lp - lm) / (2.0 * eps);
    double rel = rel_error(num, ana);
    cout << "    [DEBUG] rel_err=" << rel << " ana=" << ana << " num=" << num << endl;
    check("DecoderBlock encoder-input grad rel_err < 1e-3", rel < 1e-3);
}

// =====================================================================
// Test 11: Full TransformerDecoder end-to-end training reduces loss
// =====================================================================
static void test_decoder_training_reduces_loss() {
    cout << endl << "-- Test 11: TransformerDecoder training reduces loss --" << endl;
    size_t d_model = 8, num_heads = 2, n_q = 3, n_kv = 4, out_features = 5;
    TransformerDecoder decoder(d_model, num_heads, out_features, /*num_blocks=*/1);

    Tensor dec = make_random_input(n_q, d_model, 0.3, 111);
    Tensor enc = make_random_input(n_kv, d_model, 0.3, 112);
    Tensor target(n_q, out_features);
    srand(113);
    for (size_t i = 0; i < n_q; ++i)
        for (size_t j = 0; j < out_features; ++j)
            target[i][j] = (rand() % 1000) / 500.0 - 1.0;

    auto compute_loss = [&]() {
        Tensor out;
        decoder.forward_two(dec, enc, out);
        double loss = 0.0;
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < out_features; ++j) {
                double d = out[i][j] - target[i][j];
                loss += d * d;
            }
        return loss;
    };

    double loss0 = compute_loss();
    double lr = 0.01;
    for (int step = 0; step < 30; ++step) {
        Tensor out;
        decoder.forward_two(dec, enc, out);
        Tensor grad_out(n_q, out_features);
        for (size_t i = 0; i < n_q; ++i)
            for (size_t j = 0; j < out_features; ++j)
                grad_out[i][j] = 2.0 * (out[i][j] - target[i][j]);

        Tensor gd, ge;
        decoder.backward_two(grad_out, gd, ge);
        decoder.update_weights(lr);
        decoder.zero_grad();
    }
    double lossN = compute_loss();
    cout << "    [DEBUG] loss0=" << loss0 << " lossN=" << lossN
         << " reduction=" << (loss0 - lossN) / loss0 * 100.0 << "%" << endl;
    check("Decoder training reduces loss > 30%", (loss0 - lossN) / loss0 > 0.30);
}

// =====================================================================
// Test 12: Cross-attention with mismatched seq lengths
// =====================================================================
static void test_cross_attn_mismatched_lengths() {
    cout << endl << "-- Test 12: CrossAttn with n_q != n_kv --" << endl;
    size_t d_model = 8, num_heads = 2, n_q = 2, n_kv = 6;
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor dec = make_random_input(n_q, d_model, 0.4, 121);
    Tensor enc = make_random_input(n_kv, d_model, 0.4, 122);

    Tensor out;
    ca.forward_two(dec, enc, out);

    check("CrossAttn output rows = n_q", out.rows == n_q);
    check("CrossAttn output cols = d_model", out.cols == d_model);

    // Verify the encoder gradient flows back through the full KV length.
    Tensor grad_out(n_q, d_model);
    grad_out.fill(1.0);
    Tensor gd, ge;
    ca.backward_two(grad_out, gd, ge);
    check("CrossAttn backward: gd shape (n_q, d_model)",
          gd.rows == n_q && gd.cols == d_model);
    check("CrossAttn backward: ge shape (n_kv, d_model)",
          ge.rows == n_kv && ge.cols == d_model);
}

// =====================================================================
// Test 13: Cross-attn output differs from masked self-attn on same input
//   — confirms Q and K/V are actually coming from different sources.
// =====================================================================
static void test_cross_attn_differs_from_self() {
    cout << endl << "-- Test 13: CrossAttn differs from self-attn on same input --" << endl;
    size_t d_model = 8, num_heads = 2, n = 4;
    MaskedMultiHeadSelfAttention sa(d_model, num_heads);
    MultiHeadCrossAttention ca(d_model, num_heads);

    Tensor x = make_random_input(n, d_model, 0.5, 131);
    Tensor out_sa = sa.forward(x);

    Tensor out_ca;
    ca.forward_two(x, x, out_ca);  // pass same tensor as both Q source and K/V source

    bool differs = false;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model; ++j)
            if (std::abs(out_sa[i][j] - out_ca[i][j]) > 1e-6) differs = true;
    check("CrossAttn output differs from self-attn output", differs);
}

// =====================================================================
// Test 14: Decoder block parameter / gradient count consistency
// =====================================================================
static void test_decoder_block_param_count() {
    cout << endl << "-- Test 14: DecoderBlock param count consistency --" << endl;
    size_t d_model = 6, num_heads = 2;
    TransformerDecoderBlock block(d_model, num_heads);

    size_t np = block.parameters().size();
    size_t ng = block.gradients().size();
    check("DecoderBlock params().size() == gradients().size()", np == ng);

    // Expected: 4 W's for self-attn + 4 W's for cross-attn + 2 ln1 + 2 ln2 +
    // 2 ln3 + 4 dense (W,b of ffn_fc1 and ffn_fc2) = 18
    check("DecoderBlock param count = 18", np == 18);
}

// =====================================================================
// Test 15: Causal mask zeroes gradient at forbidden positions
// =====================================================================
static void test_causal_mask_blocks_future_gradient() {
    cout << endl << "-- Test 15: causal mask zeros grad at future positions --" << endl;
    size_t d_model = 6, num_heads = 2, n = 4;
    MaskedMultiHeadSelfAttention sa(d_model, num_heads);

    // Single-row perturbation at position 0: only affects row 0 of attention.
    // Future positions (j > 0) must have zero attention weight from query i=0.
    Tensor x = make_random_input(n, d_model, 0.4, 141);
    Tensor out = sa.forward(x);
    Tensor grad_out(n, d_model);
    grad_out.fill(0.0);
    grad_out[0][0] = 1.0;  // only row 0 receives gradient
    sa.zero_grad();
    Tensor grad_in = sa.backward(grad_out, 0.0);

    // After causal masking, perturbing a future K (j > 0) should not change
    // row 0's output. Verify by numerical check.
    double eps = 1e-4;
    int rj = 2, cj = 1;  // perturb K-source position 2, feature 1
    // Note: K-source = same as input in self-attention.
    double orig = x[rj][cj];
    x[rj][cj] = orig + eps;
    Tensor out_p = sa.forward(x);
    double diff = out_p[0][0] - out[0][0];
    x[rj][cj] = orig;
    cout << "    [DEBUG] |Δout[0][0]| after perturbing future K = " << std::abs(diff) << endl;
    check("Causal mask: perturbing future K[2] does not change row 0 output",
          std::abs(diff) < 1e-6);
}

int main() {
    cout << "==========================================" << endl;
    cout << " Encoder-Decoder Transformer" << endl;
    cout << "==========================================" << endl;

    test_masked_self_attn_forward();
    test_masked_self_attn_input_gradient();
    test_masked_self_attn_param_gradients();
    test_cross_attn_forward_shape();
    test_cross_attn_decoder_gradient();
    test_cross_attn_encoder_gradient();
    test_cross_attn_W_o_gradient();
    test_decoder_block_forward();
    test_decoder_block_decoder_gradient();
    test_decoder_block_encoder_gradient();
    test_decoder_training_reduces_loss();
    test_cross_attn_mismatched_lengths();
    test_cross_attn_differs_from_self();
    test_decoder_block_param_count();
    test_causal_mask_blocks_future_gradient();

    cout << endl << "==========================================" << endl;
    cout << " Total: " << (passed + failed) << "  Passed: " << passed
         << "  Failed: " << failed << endl;
    cout << "==========================================" << endl;

    return failed == 0 ? 0 : 1;
}