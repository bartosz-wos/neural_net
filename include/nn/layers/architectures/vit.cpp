#include "vit.h"
#include <cmath>

// ============== ViTPatchEmbedding ==============

ViTPatchEmbedding::ViTPatchEmbedding(size_t patch_size, size_t C_in, size_t H_in,
                                      size_t W_in, size_t d_model)
    : patch_size(patch_size),
      C_in(C_in),
      H_in(H_in),
      W_in(W_in),
      d_model(d_model),
      num_patches((H_in / patch_size) * (W_in / patch_size)),
      H_patch(H_in / patch_size),
      W_patch(W_in / patch_size),
      conv(static_cast<int>(C_in), static_cast<int>(d_model),
           static_cast<int>(patch_size), static_cast<int>(patch_size),
           static_cast<int>(H_in), static_cast<int>(W_in),
           static_cast<int>(patch_size), static_cast<int>(patch_size),
           0, 0) {}

Tensor ViTPatchEmbedding::forward(const Tensor& input) {
    // input: (B, C_in * H_in * W_in)
    size_t B = input.rows;

    // Conv2D forward: output (B, d_model * H_patch * W_patch)
    Tensor conv_out = conv.forward(input);

    // Reshape to (B, num_patches, d_model) stored as flat (B, num_patches * d_model)
    // For patch n=(h,w), its d_model channels are at:
    // conv_out[b][d * H_patch*W_patch + h*W_patch + w]
    Tensor out(B, num_patches * d_model);
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < num_patches; ++n) {
            size_t h = n / W_patch;
            size_t w = n % W_patch;
            for (size_t d = 0; d < d_model; ++d) {
                size_t src_idx = d * H_patch * W_patch + h * W_patch + w;
                size_t dst_idx = n * d_model + d;
                out[b][dst_idx] = conv_out[b][src_idx];
            }
        }
    }
    return out;
}

Tensor ViTPatchEmbedding::backward(const Tensor& grad_output, double learning_rate) {
    size_t B = grad_output.rows;
    // grad_output: (B, N*d_model) -> reshape to (B, d_model, H_patch, W_patch) stored flat
    Tensor grad_reshaped(B, d_model * H_patch * W_patch);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < d_model; ++d) {
            for (size_t h = 0; h < H_patch; ++h) {
                for (size_t w = 0; w < W_patch; ++w) {
                    size_t n = h * W_patch + w;
                    size_t src_idx = n * d_model + d;
                    size_t dst_idx = d * H_patch * W_patch + h * W_patch + w;
                    grad_reshaped[b][dst_idx] = grad_output[b][src_idx];
                }
            }
        }
    }
    return conv.backward(grad_reshaped, learning_rate);
}

void ViTPatchEmbedding::update_weights(double learning_rate) {
    conv.update_weights(learning_rate);
}

std::vector<Tensor*> ViTPatchEmbedding::parameters() {
    return conv.parameters();
}

std::vector<Tensor*> ViTPatchEmbedding::gradients() {
    return conv.gradients();
}

void ViTPatchEmbedding::zero_grad() {
    conv.zero_grad();
}

// ============== ViTBlock ==============

ViTBlock::ViTBlock(size_t d_model, size_t num_heads, size_t mlp_hidden, size_t seq_len)
    : d_model(d_model),
      num_heads(num_heads),
      mlp_hidden(mlp_hidden),
      seq_len(seq_len),
      attn(d_model, num_heads),
      ln1(d_model, 1e-7),
      ln2(d_model, 1e-7),
      W1(Tensor::random(d_model, mlp_hidden, 0.01)),
      b1(Tensor::zeros(1, mlp_hidden)),
      grad_W1(Tensor::zeros(d_model, mlp_hidden)),
      grad_b1(Tensor::zeros(1, mlp_hidden)),
      W2(Tensor::random(mlp_hidden, d_model, 0.01)),
      b2(Tensor::zeros(1, d_model)),
      grad_W2(Tensor::zeros(mlp_hidden, d_model)),
      grad_b2(Tensor::zeros(1, d_model)) {}

Tensor ViTBlock::forward(const Tensor& input) {
    // input: (B, seq_len * d_model) -- flat representation of (B, seq_len, d_model)
    last_x = input;
    size_t B = input.rows;
    size_t flat_size = seq_len * d_model;

    // ---- Pre-norm + MultiHeadAttention ----
    Tensor ln1_out = ln1.forward(input);  // (B, flat_size)

    // Convert to (d_model, seq_len) per batch element for MultiHeadAttention
    // Process per batch since attn.forward expects (d_model, seq_len)
    Tensor attn_out_flat(B, flat_size);

    for (size_t b = 0; b < B; ++b) {
        // ln1_out[b] has flat_size elements: s*d_model + d
        // Convert to (d_model, seq_len)
        Tensor attn_in(d_model, seq_len);
        for (size_t d = 0; d < d_model; ++d)
            for (size_t s = 0; s < seq_len; ++s)
                attn_in[d][s] = ln1_out[b][s * d_model + d];

        Tensor sample_out = attn.forward(attn_in);  // (d_model, seq_len)

        // Convert back to flat and store
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < d_model; ++d)
                attn_out_flat[b][s * d_model + d] = sample_out[d][s];
    }
    last_ln1_out = ln1_out;
    last_attn_out = attn_out_flat;

    // Residual: input + attn_out
    Tensor after_attn(B, flat_size);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < flat_size; ++i)
            after_attn[b][i] = input[b][i] + attn_out_flat[b][i];

    // ---- Pre-norm 2 + MLP ----
    Tensor ln2_out = ln2.forward(after_attn);  // (B, flat_size)
    last_ln2_out = ln2_out;

    // FFN: first linear (d_model -> mlp_hidden)
    Tensor ffn_pregelu(B, seq_len * mlp_hidden);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < mlp_hidden; ++h) {
                double val = b1[0][h];
                for (size_t d = 0; d < d_model; ++d)
                    val += ln2_out[b][s * d_model + d] * W1[d][h];
                ffn_pregelu[b][s * mlp_hidden + h] = val;
            }
        }
    }
    last_ffn_pregelu = ffn_pregelu;

    // Apply GeLU in-place
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < seq_len * mlp_hidden; ++i) {
            double x = ffn_pregelu[b][i];
            double xc = std::max(-4.0, std::min(4.0, x));
            double sqrt_2_pi = std::sqrt(2.0 / std::acos(-1.0));
            double cdf = 0.5 * (1.0 + std::tanh(sqrt_2_pi * (xc + 0.044715 * xc * xc * xc)));
            ffn_pregelu[b][i] = x * cdf;
        }
    }

    // Second linear (mlp_hidden -> d_model)
    Tensor ffn_out(B, flat_size);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < d_model; ++d) {
                double val = b2[0][d];
                for (size_t h = 0; h < mlp_hidden; ++h)
                    val += ffn_pregelu[b][s * mlp_hidden + h] * W2[h][d];
                ffn_out[b][s * d_model + d] = val;
            }
        }
    }

    // Final residual: after_attn + ffn_out
    Tensor output(B, flat_size);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < flat_size; ++i)
            output[b][i] = after_attn[b][i] + ffn_out[b][i];

    return output;
}

Tensor ViTBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t B = grad_output.rows;
    size_t flat_size = seq_len * d_model;

    // ---- Residual split: grad flows to both after_attn and ffn_out ----
    Tensor grad_after_attn = grad_output.clone();
    Tensor grad_ffn_out = grad_output.clone();

    // ---- Backprop FFN2: W2, b2 ----
    // ffn_out[b][s*d_model + d] = b2[d] + sum_h ffn_pregelu[b][s*mlp_hidden + h] * W2[h][d]
    grad_W2.fill(0.0);
    grad_b2.fill(0.0);
    for (size_t h = 0; h < mlp_hidden; ++h) {
        for (size_t d = 0; d < d_model; ++d) {
            double g = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    g += grad_ffn_out[b][s * d_model + d] * last_ffn_pregelu[b][s * mlp_hidden + h];
                }
            }
            grad_W2[h][d] += g;
        }
    }
    for (size_t d = 0; d < d_model; ++d) {
        double g = 0.0;
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < seq_len; ++s)
                g += grad_ffn_out[b][s * d_model + d];
        grad_b2[0][d] += g;
    }

    // dL/d(ffn_pregelu) = grad_ffn_out @ W2^T
    Tensor grad_ffn_mid(B, seq_len * mlp_hidden);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < mlp_hidden; ++h) {
                double g = 0.0;
                for (size_t d = 0; d < d_model; ++d)
                    g += grad_ffn_out[b][s * d_model + d] * W2[h][d];
                grad_ffn_mid[b][s * mlp_hidden + h] = g;
            }
        }
    }

    // ---- Backprop GeLU ----
    // Recompute pregelu z = W1 * ln2_out + b1 for GeLU derivative
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < mlp_hidden; ++h) {
                double z = b1[0][h];
                for (size_t d = 0; d < d_model; ++d)
                    z += last_ln2_out[b][s * d_model + d] * W1[d][h];
                double xc = std::max(-4.0, std::min(4.0, z));
                double sqrt_2_pi = std::sqrt(2.0 / std::acos(-1.0));
                double cdf = 0.5 * (1.0 + std::tanh(sqrt_2_pi * (xc + 0.044715 * xc * xc * xc)));
                double pdf = std::exp(-0.5 * xc * xc) / std::sqrt(2.0 * std::acos(-1.0));
                double gelu_deriv = cdf + xc * pdf * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * xc * xc);
                grad_ffn_mid[b][s * mlp_hidden + h] *= gelu_deriv;
            }
        }
    }

    // ---- Backprop FFN1: W1, b1 ----
    // dL/d(ln2_out) = grad_ffn_mid @ W1^T
    grad_W1.fill(0.0);
    grad_b1.fill(0.0);
    Tensor grad_ln2_out(B, flat_size);
    for (size_t b = 0; b < B; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < d_model; ++d) {
                double g = 0.0;
                for (size_t h = 0; h < mlp_hidden; ++h)
                    g += grad_ffn_mid[b][s * mlp_hidden + h] * W1[d][h];
                grad_ln2_out[b][s * d_model + d] = g;
            }
        }
    }
    for (size_t d = 0; d < d_model; ++d) {
        for (size_t h = 0; h < mlp_hidden; ++h) {
            double g = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t s = 0; s < seq_len; ++s)
                    g += grad_ffn_mid[b][s * mlp_hidden + h] * last_ln2_out[b][s * d_model + d];
            }
            grad_W1[d][h] += g;
        }
    }
    for (size_t h = 0; h < mlp_hidden; ++h) {
        double g = 0.0;
        for (size_t b = 0; b < B; ++b)
            for (size_t s = 0; s < seq_len; ++s)
                g += grad_ffn_mid[b][s * mlp_hidden + h];
        grad_b1[0][h] += g;
    }

    // ---- Backprop through ln2 ----
    Tensor grad_ln2_residual = ln2.backward(grad_ln2_out, 0.0);

    // grad_after_attn += grad from ln2
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < flat_size; ++i)
            grad_after_attn[b][i] += grad_ln2_residual[b][i];

    // ---- Backprop through MultiHeadAttention ----
    Tensor grad_ln1_out(B, flat_size);
    for (size_t b = 0; b < B; ++b) {
        // Convert grad_after_attn to (d_model, seq_len) for attn backward
        Tensor grad_attn_standard(d_model, seq_len);
        for (size_t d = 0; d < d_model; ++d)
            for (size_t s = 0; s < seq_len; ++s)
                grad_attn_standard[d][s] = grad_after_attn[b][s * d_model + d];

        Tensor grad_ln1 = attn.backward(grad_attn_standard, 0.0);  // (d_model, seq_len)

        // Convert back to flat
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < d_model; ++d)
                grad_ln1_out[b][s * d_model + d] = grad_ln1[d][s];
    }

    // Backprop through ln1: dL/dinput += grad_ln1_out (from ln1 backward) + grad_after_attn (residual)
    Tensor grad_ln1_from_ln1 = ln1.backward(grad_ln1_out, 0.0);
    Tensor grad_input(B, flat_size);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < flat_size; ++i)
            grad_input[b][i] = grad_ln1_from_ln1[b][i] + grad_after_attn[b][i];

    return grad_input;
}

void ViTBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    for (size_t i = 0; i < W1.rows; ++i)
        for (size_t j = 0; j < W1.cols; ++j)
            W1[i][j] -= learning_rate * grad_W1[i][j];
    for (size_t j = 0; j < b1.cols; ++j)
        b1[0][j] -= learning_rate * grad_b1[0][j];
    for (size_t i = 0; i < W2.rows; ++i)
        for (size_t j = 0; j < W2.cols; ++j)
            W2[i][j] -= learning_rate * grad_W2[i][j];
    for (size_t j = 0; j < b2.cols; ++j)
        b2[0][j] -= learning_rate * grad_b2[0][j];
}

std::vector<Tensor*> ViTBlock::parameters() {
    std::vector<Tensor*> result;
    result.push_back(&W1); result.push_back(&b1);
    result.push_back(&W2); result.push_back(&b2);
    for (Tensor* p : attn.parameters()) result.push_back(p);
    for (Tensor* p : ln1.parameters()) result.push_back(p);
    for (Tensor* p : ln2.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> ViTBlock::gradients() {
    std::vector<Tensor*> result;
    result.push_back(&grad_W1); result.push_back(&grad_b1);
    result.push_back(&grad_W2); result.push_back(&grad_b2);
    for (Tensor* p : attn.gradients()) result.push_back(p);
    for (Tensor* p : ln1.gradients()) result.push_back(p);
    for (Tensor* p : ln2.gradients()) result.push_back(p);
    return result;
}

void ViTBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    grad_W1.fill(0.0); grad_b1.fill(0.0);
    grad_W2.fill(0.0); grad_b2.fill(0.0);
}

// ============== ViT ==============

ViT::ViT(size_t patch_size, size_t d_model, size_t num_heads, size_t num_layers,
         size_t C_in, size_t H_in, size_t W_in, size_t num_classes,
         double mlp_ratio)
    : patch_size(patch_size),
      d_model(d_model),
      num_heads(num_heads),
      num_layers(num_layers),
      num_classes(num_classes),
      C_in(C_in),
      H_in(H_in),
      W_in(W_in),
      num_patches((H_in / patch_size) * (W_in / patch_size)),
      N_plus_1(num_patches + 1),
      mlp_hidden(static_cast<size_t>(d_model * mlp_ratio)),
      patch_embed(patch_size, C_in, H_in, W_in, d_model),
      class_token(Tensor::random(1, d_model, 0.01)),
      pos_embedding(Tensor::random(N_plus_1, d_model, 0.01)),
      transformer_blocks(),
      ln(d_model, 1e-7),
      head(d_model, num_classes) {
    for (size_t i = 0; i < num_layers; ++i)
        transformer_blocks.emplace_back(d_model, num_heads, mlp_hidden, N_plus_1);
}

Tensor ViT::forward(const Tensor& input) {
    // input: (B, C_in * H_in * W_in)
    size_t B = input.rows;

    // 1. Patch embedding: (B, C*H*W) -> (B, N*d_model)
    Tensor patches = patch_embed.forward(input);

    // 2. Prepend class token: build (B, (N+1)*d_model) flat
    size_t flat_n1 = N_plus_1 * d_model;
    Tensor x(B, flat_n1);
    for (size_t b = 0; b < B; ++b) {
        // CLS token at position 0
        for (size_t d = 0; d < d_model; ++d)
            x[b][d] = class_token[0][d];
        // Patch tokens at positions 1..N
        for (size_t n = 0; n < num_patches; ++n)
            for (size_t d = 0; d < d_model; ++d)
                x[b][(n + 1) * d_model + d] = patches[b][n * d_model + d];
    }

    // 3. Add positional embedding (broadcast across batch)
    for (size_t b = 0; b < B; ++b)
        for (size_t p = 0; p < N_plus_1; ++p)
            for (size_t d = 0; d < d_model; ++d)
                x[b][p * d_model + d] += pos_embedding[p][d];

    last_patch_tokens = x;

    // 4. Transformer blocks
    for (auto& block : transformer_blocks)
        x = block.forward(x);

    // 5. Final LayerNorm on flattened sequence
    x = ln.forward(x);

    // 6. Extract class token at position 0: x[b][0*d_model + d]
    last_cls = Tensor(B, d_model);
    for (size_t b = 0; b < B; ++b)
        for (size_t d = 0; d < d_model; ++d)
            last_cls[b][d] = x[b][d];

    // 7. Classification head: (B, d_model) -> (B, num_classes)
    Tensor output = head.forward(last_cls);
    return output;
}

Tensor ViT::backward(const Tensor& grad_output, double learning_rate) {
    size_t B = grad_output.rows;

    // Backprop through head
    Tensor grad_cls = head.backward(grad_output, learning_rate);

    // Expand to (B, (N+1)*d_model), only CLS position (0) has non-zero gradient
    size_t flat_n1 = N_plus_1 * d_model;
    Tensor grad_x(B, flat_n1);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < d_model; ++d) {
            grad_x[b][d] = grad_cls[b][d];
            for (size_t p = 1; p < N_plus_1; ++p)
                grad_x[b][p * d_model + d] = 0.0;
        }
    }

    // Backprop through final LayerNorm
    grad_x = ln.backward(grad_x, learning_rate);

    // Backprop through transformer blocks in reverse order
    for (auto it = transformer_blocks.rbegin(); it != transformer_blocks.rend(); ++it)
        grad_x = it->backward(grad_x, learning_rate);

    // Positional embedding is additive (no weights) -> gradient passes through

    // Remove class token -> grad_patches (gradients only at patch positions 1..N)
    size_t flat_n = num_patches * d_model;
    Tensor grad_patches(B, flat_n);
    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < num_patches; ++n) {
            for (size_t d = 0; d < d_model; ++d) {
                grad_patches[b][n * d_model + d] = grad_x[b][(n + 1) * d_model + d];
            }
        }
    }

    // Backprop through patch embedding
    Tensor grad_input = patch_embed.backward(grad_patches, learning_rate);
    return grad_input;
}

void ViT::update_weights(double learning_rate) {
    patch_embed.update_weights(learning_rate);
    for (auto& block : transformer_blocks)
        block.update_weights(learning_rate);
    ln.update_weights(learning_rate);
    head.update_weights(learning_rate);
}

std::vector<Tensor*> ViT::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : patch_embed.parameters()) result.push_back(p);
    for (auto& block : transformer_blocks)
        for (Tensor* p : block.parameters()) result.push_back(p);
    for (Tensor* p : ln.parameters()) result.push_back(p);
    for (Tensor* p : head.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> ViT::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* p : patch_embed.gradients()) result.push_back(p);
    for (auto& block : transformer_blocks)
        for (Tensor* p : block.gradients()) result.push_back(p);
    for (Tensor* p : ln.gradients()) result.push_back(p);
    for (Tensor* p : head.gradients()) result.push_back(p);
    return result;
}

void ViT::zero_grad() {
    patch_embed.zero_grad();
    for (auto& block : transformer_blocks)
        block.zero_grad();
    ln.zero_grad();
    head.zero_grad();
}

void ViT::set_training(bool t) {
    ln.set_training(t);
    for (auto& block : transformer_blocks) {
        block.ln1.set_training(t);
        block.ln2.set_training(t);
    }
}
