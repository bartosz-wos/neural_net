#include "transformer.h"
#include "../normalization/layer_norm.h"
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

Tensor MultiHeadAttention::backward(const Tensor& grad_output, double) {
    // grad_output: (d_model, seq_len)
    // 1. Reshape grad_output (d_model, seq_len) to internal (tokens, d_model)
    size_t seq_len = grad_output.cols;
    size_t tokens = seq_len;

    Tensor grad_out(tokens, d_model); // internal gradient (tokens, d_model)
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_out[s][f] = grad_output[f][s];

    // 2. Propagate through W_o: grad_proj = grad_out @ W_o^T
    // grad_proj: (tokens, d_model), W_o: (d_model, d_model)
    // grad_proj[i][j] = sum_k grad_out[i][k] * W_o[k][j]
    Tensor grad_proj(tokens, d_model);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model; ++k)
                v += grad_out[i][k] * W_o[k][j];
            grad_proj[i][j] = v;
        }
    }

    // Accumulate grad_W_o += grad_out^T @ last_attn_out
    // grad_out: (tokens, d_model), last_attn_out: (tokens, d_model)
    // grad_W_o += grad_out^T @ last_attn_out = (d_model, tokens) @ (tokens, d_model) = (d_model, d_model)
    for (size_t i = 0; i < d_model; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            double v = 0.0;
            for (size_t t = 0; t < tokens; ++t)
                v += grad_out[t][i] * last_attn_out[t][j];
            grad_W_o[i][j] += v;
        }
    }

    // 3. Split grad_proj per head: grad_proj[:, h*d_k:(h+1)*d_k] -> grad_attn_h (tokens, d_k)
    // 4. Backprop through attention for each head
    Tensor grad_q(tokens, d_model), grad_k(tokens, d_model), grad_v(tokens, d_model);
    grad_q.fill(0.0); grad_k.fill(0.0); grad_v.fill(0.0);
    for (size_t h = 0; h < num_heads; ++h) {
        // Reconstruct per-head Q_h, K_h, V_h from last_q, last_k, last_v
        Tensor Q_h(d_k, tokens), K_h(d_k, tokens), V_h(d_k, tokens);
        for (size_t dk = 0; dk < d_k; ++dk) {
            for (size_t t = 0; t < tokens; ++t) {
                Q_h[dk][t] = last_q[t][h * d_k + dk];
                K_h[dk][t] = last_k[t][h * d_k + dk];
                V_h[dk][t] = last_v[t][h * d_k + dk];
            }
        }

        // Extract grad_proj slice for this head: grad_attn_h (tokens, d_k)
        Tensor grad_attn_h(tokens, d_k);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t dk = 0; dk < d_k; ++dk)
                grad_attn_h[t][dk] = grad_proj[t][h * d_k + dk];

        // Compute attention scores: attn_scores = Q_h @ K_h^T / sqrt(d_k)
        Tensor attn_scores(tokens, tokens);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t j = 0; j < tokens; ++j) {
                double s = 0.0;
                for (size_t dk = 0; dk < d_k; ++dk)
                    s += Q_h[dk][i] * K_h[dk][j];
                attn_scores[i][j] = s / std::sqrt((double)d_k);
            }
        }

        // Recompute softmax of attn_scores
        Tensor attn_probs(tokens, tokens);
        for (size_t i = 0; i < tokens; ++i) {
            double max_s = attn_scores[i][0];
            for (size_t j = 1; j < tokens; ++j)
                if (attn_scores[i][j] > max_s) max_s = attn_scores[i][j];
            double sum_exp = 0.0;
            for (size_t j = 0; j < tokens; ++j) {
                attn_scores[i][j] = std::exp(attn_scores[i][j] - max_s);
                sum_exp += attn_scores[i][j];
            }
            for (size_t j = 0; j < tokens; ++j)
                attn_probs[i][j] = attn_scores[i][j] / sum_exp;
        }

        // Compute dL/dV_h_t = grad_attn_h^T @ attn_probs
        // grad_attn_h: (tokens, d_k), attn_probs: (tokens, tokens)
        // grad_V_h_t[i][dk] = sum_t grad_attn_h[t][dk] * attn_probs[t][i]
        Tensor grad_V_h_t(tokens, d_k);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < tokens; ++t)
                    v += grad_attn_h[t][dk] * attn_probs[t][i];
                grad_V_h_t[i][dk] = v;
            }
        }

        // Compute dL/dK_h_t = grad_attn_h @ Q_h hadamard attn_probs / sqrt(d_k)
        // First compute M = grad_attn_h @ Q_h: (tokens, d_k) @ (d_k, tokens) = (tokens, tokens)
        // Then grad_K_h_t[i][dk] = sum_j attn_probs[j][i] * M[j][dk] / sqrt(d_k)
        Tensor M(tokens, d_k);
        for (size_t j = 0; j < tokens; ++j) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < tokens; ++t)
                    v += grad_attn_h[t][dk] * Q_h[dk][t];
                M[j][dk] = v;
            }
        }
        Tensor grad_K_h_t(tokens, d_k);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < tokens; ++j)
                    v += attn_probs[j][i] * M[j][dk];
                grad_K_h_t[i][dk] = v / std::sqrt((double)d_k);
            }
        }

        // Compute dL/dQ_h_t = grad_attn_h @ K_h hadamard attn_probs / sqrt(d_k)
        // N = grad_attn_h @ K_h: (tokens, d_k) @ (d_k, tokens) = (tokens, tokens)
        Tensor N(tokens, d_k);
        for (size_t j = 0; j < tokens; ++j) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double v = 0.0;
                for (size_t t = 0; t < tokens; ++t)
                    v += grad_attn_h[t][dk] * K_h[dk][t];
                N[j][dk] = v;
            }
        }
        Tensor grad_Q_h_t(tokens, d_k);
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                double v = 0.0;
                for (size_t j = 0; j < tokens; ++j)
                    v += attn_probs[j][i] * N[j][dk];
                grad_Q_h_t[i][dk] = v / std::sqrt((double)d_k);
            }
        }

        // Reshape grad_Q_h_t (tokens, d_k) -> full Q gradient (tokens, d_model)
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t dk = 0; dk < d_k; ++dk) {
                grad_q[t][h * d_k + dk] = grad_Q_h_t[t][dk];
                grad_k[t][h * d_k + dk] = grad_K_h_t[t][dk];
                grad_v[t][h * d_k + dk] = grad_V_h_t[t][dk];
            }
        }

        // grad_W_v += last_x^T @ grad_v: (d_model, tokens) @ (tokens, d_model) = (d_model, d_model)
        // grad_W_k += last_x^T @ grad_k
        // grad_W_q += last_x^T @ grad_q
        for (size_t i = 0; i < d_model; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
                double gq = 0.0, gk = 0.0, gv = 0.0;
                for (size_t t = 0; t < tokens; ++t) {
                    gq += last_x[t][i] * grad_q[t][j];
                    gk += last_x[t][i] * grad_k[t][j];
                    gv += last_x[t][i] * grad_v[t][j];
                }
                grad_W_q[i][j] += gq;
                grad_W_k[i][j] += gk;
                grad_W_v[i][j] += gv;
            }
        }
    }

    // 5. Backprop through Q=x@W_q^T: grad_x = grad_q @ W_q
    // grad_q: (tokens, d_model), W_q: (d_model, d_model)
    // grad_x[i][j] = sum_k grad_q[i][k] * W_q[k][j]
    Tensor grad_x(tokens, d_model);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model; ++k)
                v += grad_q[i][k] * W_q[k][j];
            grad_x[i][j] = v;
        }
    }

    // 6. Reshape grad_x (tokens, d_model) back to (d_model, seq_len)
    Tensor grad_input(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_input[f][s] = grad_x[s][f];

    return grad_input;
}
void MultiHeadAttention::update_weights(double learning_rate) {
    // Gradient descent update for W_q, W_k, W_v, W_o
    for (size_t i = 0; i < d_model; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            W_q[i][j] -= learning_rate * grad_W_q[i][j];
            W_k[i][j] -= learning_rate * grad_W_k[i][j];
            W_v[i][j] -= learning_rate * grad_W_v[i][j];
            W_o[i][j] -= learning_rate * grad_W_o[i][j];
        }
    }
}

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

    last_attn_out = attn_out;

    // Residual in token space
    Tensor residual = x - attn_tokens;

    // LayerNorm
    Tensor norm1_out = ln1.forward(residual); // (tokens, d_model)

    // FFN: W2 @ GELU(W1 @ x + b1) + b2
    // First linear + GELU
    Tensor ffn1(tokens, d_model_local);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_local; ++j) {
            double val = b1[0][j];
            for (size_t k = 0; k < d_model_local; ++k)
                val += W1[k][j] * norm1_out[i][k];
            ffn1[i][j] = val;
        }
    }

    // Store pre-GELU activation for backward
    last_ffn_pregelu = ffn1;

    // GELU activation
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d_model_local; ++j) {
            double x_g = ffn1[i][j];
            ffn1[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));
        }
    }

    // Second linear
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

Tensor TransformerBlock::backward(const Tensor& grad_output, double) {
    size_t tokens = last_x.rows;
    size_t d = last_x.cols;
    size_t seq_len = grad_output.cols;

    // Reshape grad_output to token space: (d_model, seq_len) -> (tokens, d_model)
    Tensor grad_norm2(tokens, d);
    for (size_t f = 0; f < d; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_norm2[s][f] = grad_output[f][s];

    // 1. Backprop through ln2
    Tensor grad_residual2 = ln2.backward(grad_norm2, 0.0);

    // 2. Split residual2: dL/dnorm1 += dL/dffn2 (identity), dL/dffn2 = grad_residual2
    Tensor grad_ffn2 = grad_residual2;
    Tensor grad_norm1 = grad_residual2;

    // 3. Backprop through FFN2 (linear): W2[k][j], ffn2[i][j] = b2[0][j] + sum_k(W2[k][j] * ffn1[i][k])
    // Recompute post-GELU ffn1 from last_ffn_pregelu
    Tensor ffn1_postgelu(tokens, d);
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double x_g = last_ffn_pregelu[i][j];
            ffn1_postgelu[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));
        }
    }

    Tensor grad_ffn1(tokens, d);
    grad_ffn1.fill(0.0);
    grad_W2.fill(0.0);
    grad_b2.fill(0.0);

    for (size_t i = 0; i < tokens; ++i) {
        for (size_t k = 0; k < d; ++k) {
            double grad_ffn1_ik = 0.0;
            for (size_t j = 0; j < d; ++j) {
                grad_W2[k][j] += grad_ffn2[i][j] * ffn1_postgelu[i][k];
                grad_ffn1_ik += grad_ffn2[i][j] * W2[k][j];
            }
            grad_ffn1[i][k] = grad_ffn1_ik;
            grad_b2[0][k] += grad_ffn2[i][k];
        }
    }

    // 4. Backprop through GELU: multiply by GELU derivative
    // GELU'(x) = 0.5 * (1 - tanh^2(z)) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double x_g = last_ffn_pregelu[i][j];
            double z = std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g);
            double tanh_z = std::tanh(z);
            double gelu_deriv = 0.5 * (1.0 - tanh_z * tanh_z) * std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x_g * x_g);
            grad_ffn1[i][j] *= gelu_deriv;
        }
    }

    // 5. Backprop through FFN1 (linear): W1[k][j], pregelu = b1 + W1 @ norm1_out^T
    // Recompute norm1_out via ln1.forward on residual
    Tensor attn_tokens(tokens, d);
    for (size_t f = 0; f < d; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            attn_tokens[s][f] = last_attn_out[f][s];
    Tensor residual1 = last_x - attn_tokens;
    Tensor norm1_out = ln1.forward(residual1);

    grad_W1.fill(0.0);
    grad_b1.fill(0.0);
    Tensor grad_norm1_ffn(tokens, d);
    grad_norm1_ffn.fill(0.0);

    for (size_t i = 0; i < tokens; ++i) {
        for (size_t k = 0; k < d; ++k) {
            double grad_norm1_k = 0.0;
            for (size_t j = 0; j < d; ++j) {
                grad_W1[k][j] += grad_ffn1[i][j] * norm1_out[i][k];
                grad_norm1_k += grad_ffn1[i][j] * W1[k][j];
            }
            grad_norm1_ffn[i][k] = grad_norm1_k;
            grad_b1[0][k] += grad_ffn1[i][k];
        }
    }

    // Accumulate FFN contribution into grad_norm1
    grad_norm1 = grad_norm1 + grad_norm1_ffn;

    // 6. Backprop through residual1: residual1 = x - attn_tokens
    // dL/dx += dL/dnorm1 (identity), dL/dattn_tokens = -dL/dnorm1
    Tensor grad_attn_tokens = grad_norm1 * (-1.0);

    // 7. Backprop through ln1: dL/dresidual1 = ln1.backward(-grad_norm1)
    Tensor grad_residual1 = ln1.backward(grad_norm1 * (-1.0), 0.0);
    // dL/dx = grad_residual1 + grad_attn_tokens
    Tensor grad_x = grad_residual1 + grad_attn_tokens;

    // 8. Backprop through attention
    // Reshape grad_attn_tokens to standard space for attn.backward
    Tensor grad_attn_standard(d, seq_len);
    for (size_t f = 0; f < d; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_attn_standard[f][s] = grad_attn_tokens[s][f];

    Tensor grad_input(d, seq_len);
    for (size_t f = 0; f < d; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            grad_input[f][s] = grad_x[s][f];

    Tensor grad_from_attn = attn.backward(grad_attn_standard, 0.0);
    grad_input = grad_input + grad_from_attn;
    return grad_input;
}

void TransformerBlock::update_weights(double learning_rate) {
    for (size_t i = 0; i < d_model; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            W1[i][j] -= learning_rate * grad_W1[i][j];
            b1[0][j] -= learning_rate * grad_b1[0][j];
            W2[i][j] -= learning_rate * grad_W2[i][j];
            b2[0][j] -= learning_rate * grad_b2[0][j];
        }
    }
}

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
