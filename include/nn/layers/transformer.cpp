#include "transformer.h"
#include "layer_norm.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Fix dimension mismatch: standard convention is (tokens, d_model) internally
// Reshape at entry/exit: (d_model, seq_len) <-> (tokens, d_model)

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads)
{
    W_q = Tensor::random(d_model, d_model, 0.01);
    W_k = Tensor::random(d_model, d_model, 0.01);
    W_v = Tensor::random(d_model, d_model, 0.01);
    W_o = Tensor::random(d_model, d_model, 0.01);
    grad_W_q = Tensor::zeros(d_model, d_model);
    grad_W_k = Tensor::zeros(d_model, d_model);
    grad_W_v = Tensor::zeros(d_model, d_model);
    grad_W_o = Tensor::zeros(d_model, d_model);
    batch_size = 1;
}

std::vector<Tensor*> MultiHeadAttention::parameters() { return {&W_q, &W_k, &W_v, &W_o}; }
std::vector<Tensor*> MultiHeadAttention::gradients() { return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o}; }
void MultiHeadAttention::zero_grad() {
    grad_W_q.fill(0.0); grad_W_k.fill(0.0); grad_W_v.fill(0.0); grad_W_o.fill(0.0);
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    // input: (d_model, seq_len) -> internal (tokens, d_model) where tokens = seq_len (batch=1)
    size_t seq_len = input.cols;
    size_t tokens = seq_len; // batch = 1

    // Reshape input to (tokens, d_model): each row is one token embedding
    Tensor x(tokens, d_model);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x[s][f] = input[f][s];  // flip indices

    last_x = x;

    // Compute Q, K, V: each (tokens, d_model)
    // Q = x @ W_q^T: (tokens, d_model) @ (d_model, d_model) = (tokens, d_model)
    Tensor Q(tokens, d_model), K(tokens, d_model), V(tokens, d_model);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (size_t k = 0; k < d_model; ++k) {
                qv += x[i][k] * W_q[k][j];
                kv += x[i][k] * W_k[k][j];
                vv += x[i][k] * W_v[k][j];
            }
            Q[i][j] = qv; K[i][j] = kv; V[i][j] = vv;
        }
    }
    last_q = Q; last_k = K; last_v = V;

    // Multi-head attention
    // Reshape Q,K,V to (num_heads, tokens, d_k)
    std::vector<Tensor> Q_h(num_heads, Tensor(d_k, seq_len));
    std::vector<Tensor> K_h(num_heads, Tensor(d_k, seq_len));
    std::vector<Tensor> V_h(num_heads, Tensor(d_k, seq_len));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t dk = 0; dk < d_k; ++dk) {
            for (size_t t = 0; t < tokens; ++t) {
                Q_h[h][dk][t] = Q[t][h * d_k + dk];
                K_h[h][dk][t] = K[t][h * d_k + dk];
                V_h[h][dk][t] = V[t][h * d_k + dk];
            }
        }
    }

    // Attention per head, results stored in output_acc
    Tensor output_acc(tokens, d_model);
    output_acc.fill(0.0);

    for (size_t h = 0; h < num_heads; ++h) {
        // Q_h[h]: (d_k, tokens), K_h[h]: (d_k, tokens), V_h[h]: (d_k, tokens)
        // scores = Q_h[h]^T @ K_h[h]: (tokens, d_k) @ (d_k, tokens) = (tokens, tokens)
        Tensor Q_h_t = Q_h[h].transpose(); // (tokens, d_k)
        Tensor K_h_t = K_h[h].transpose(); // (tokens, d_k)

        Tensor scores(tokens, tokens);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t j = 0; j < tokens; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < d_k; ++dk)
                    s += Q_h_t[i][dk] * K_h_t[j][dk];
                scores[i][j] = s / std::sqrt((double)d_k);
            }
        }

        // Softmax over target (columns)
        for (size_t i = 0; i < tokens; ++i) {
            double max_s = scores[i][0];
            for (size_t j = 1; j < tokens; ++j)
                if (scores[i][j] > max_s) max_s = scores[i][j];
            double sum_exp = 0.0;
            for (size_t j = 0; j < tokens; ++j) {
                scores[i][j] = std::exp(scores[i][j] - max_s);
                sum_exp += scores[i][j];
            }
            for (size_t j = 0; j < tokens; ++j)
                scores[i][j] /= sum_exp;
        }

        // attn = scores @ V_h[h]^T: (tokens, tokens) @ (tokens, d_k) = (tokens, d_k)
        Tensor V_h_t = V_h[h].transpose(); // (tokens, d_k)
        Tensor attn(tokens, d_k);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double val = 0.0;
                for (size_t j = 0; j < tokens; ++j)
                    val += scores[i][j] * V_h_t[j][dk];
                attn[i][dk] = val;
            }
        }

        // Add to output_acc at positions [h*d_k ... (h+1)*d_k]
        for (size_t t = 0; t < tokens; ++t)
            for (size_t dk = 0; dk < d_k; ++dk)
                output_acc[t][h * d_k + dk] += attn[t][dk];
    }

    // Final projection: output_acc @ W_o^T: (tokens, d_model) @ (d_model, d_model) = (tokens, d_model)
    Tensor output(tokens, d_model);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            double val = 0.0;
            for (size_t k = 0; k < d_model; ++k)
                val += output_acc[i][k] * W_o[k][j];
            output[i][j] = val;
        }
    }
    last_attn_out = output;

    // Reshape output back to (d_model, seq_len)
    Tensor out_back(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            out_back[f][s] = output[s][f];
    return out_back;
}

Tensor MultiHeadAttention::backward(const Tensor&, double) { return Tensor(); }
void MultiHeadAttention::update_weights(double) {}

// ============== TransformerBlock ==============

TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads)
    : attn(d_model, num_heads)
    , ln1(d_model)
    , ln2(d_model)
    , W1(Tensor::random(d_model, d_model, 0.01))
    , b1(Tensor::zeros(1, d_model))
    , W2(Tensor::random(d_model, d_model, 0.01))
    , b2(Tensor::zeros(1, d_model))
{}

Tensor TransformerBlock::forward(const Tensor& input) {
    // input: (d_model, seq_len)
    size_t d_model_local = input.rows;
    size_t seq_len = input.cols;

    // Reshape to (tokens, d_model) where tokens = seq_len (batch=1)
    size_t tokens = seq_len;
    Tensor x(tokens, d_model_local);
    for (size_t f = 0; f < d_model_local; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x[s][f] = input[f][s];

    last_x = x;

    // Self-attention
    Tensor attn_out = attn.forward(input); // returns (d_model, seq_len)
    
    // Reshape attn_out to (tokens, d_model)
    Tensor attn_tokens(tokens, d_model_local);
    for (size_t f = 0; f < d_model_local; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            attn_tokens[s][f] = attn_out[f][s];

    // Residual in token space
    Tensor residual = x - attn_tokens;

    // LayerNorm
    Tensor norm1_out = ln1.forward(residual); // (tokens, d_model)

    // FFN: W2 @ GELU(W1 @ x + b1) + b2
    // W1: (d_model, d_model), norm1_out: (tokens, d_model)
    // W1 @ norm1_out^T: each row i = dot(W1[i], norm1_out) per column
    // Actually: W1 @ norm1_out^T where W1 is (d_model, d_model), norm1_out^T is (d_model, tokens)
    // Result: (d_model, tokens), then transpose to (tokens, d_model)

    // First linear + GELU
    // ffn1[i][j] = sum over k of W1[k][i] * norm1_out[j][k] + b1[0][i]
    Tensor ffn1(tokens, d_model_local);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_local; ++j) {
            double val = b1[0][j];
            for (size_t k = 0; k < d_model_local; ++k)
                val += W1[k][j] * norm1_out[i][k];
            ffn1[i][j] = val;
        }
    }

    // GELU activation
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_local; ++j) {
            double x_g = ffn1[i][j];
            ffn1[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));
        }
    }

    // Second linear
    // ffn2[i][j] = sum over k of W2[k][j] * ffn1[i][k] + b2[0][j]
    Tensor ffn2(tokens, d_model_local);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_local; ++j) {
            double val = b2[0][j];
            for (size_t k = 0; k < d_model_local; ++k)
                val += W2[k][j] * ffn1[i][k];
            ffn2[i][j] = val;
        }
    }

    // Final residual + LayerNorm
    Tensor ffn_res = norm1_out + ffn2;
    Tensor norm2_out = ln2.forward(ffn_res);

    // Reshape back to (d_model, seq_len)
    last_ffn_out = Tensor(d_model_local, seq_len);
    for (size_t f = 0; f < d_model_local; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            last_ffn_out[f][s] = norm2_out[s][f];

    return last_ffn_out;
}

Tensor TransformerBlock::backward(const Tensor&, double) { return Tensor(); }
void TransformerBlock::update_weights(double) {}

std::vector<Tensor*> TransformerBlock::parameters() {
    auto p = attn.parameters();
    p.push_back(&W1); p.push_back(&b1);
    p.push_back(&W2); p.push_back(&b2);
    return p;
}
std::vector<Tensor*> TransformerBlock::gradients() {
    auto g = attn.gradients();
    g.push_back(&grad_W1); g.push_back(&grad_b1);
    g.push_back(&grad_W2); g.push_back(&grad_b2);
    return g;
}
void TransformerBlock::zero_grad() {
    attn.zero_grad();
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

// ============== PositionalEncoding ==============

PositionalEncoding::PositionalEncoding(size_t max_len, size_t d_model) {
    pe = Tensor(max_len, d_model);
    for (size_t pos = 0; pos < max_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            double angle = pos / std::pow(10000.0, 2.0 * (i / 2) / d_model);
            pe[pos][i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& input) {
    // input: (d_model, seq_len) = (d_model, seq_len)
    size_t d_model_local = input.rows;
    size_t seq_len = input.cols;
    size_t max_len = pe.rows;

    Tensor output(input.rows, input.cols);
    for (size_t f = 0; f < d_model_local; ++f) {
        for (size_t s = 0; s < seq_len && s < max_len; ++s) {
            output[f][s] = input[f][s] + pe[s][f];
        }
    }
    return output;
}

Tensor PositionalEncoding::backward(const Tensor& grad_output, double) {
    return grad_output;
}