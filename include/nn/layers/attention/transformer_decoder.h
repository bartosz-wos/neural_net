#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>

// ============================================================================
// Encoder-Decoder Transformer (Vaswani et al. 2017 §3.2)
//
// The original Transformer paper has two halves: an encoder (self-attention
// only) and a decoder that combines (1) masked self-attention over the target
// sequence so far, and (2) cross-attention over the encoder's output.
// Cross-attention is the bridge that lets the decoder "look at" the source.
//
// Conventions:
//   * All layers use the (tokens, d_model) layout convention (matching GQA,
//     Linformer, Hopfield, ALiBi, MLA, etc.) so that they compose with the
//     rest of the repo without reshaping. The legacy `TransformerBlock` in
//     transformer.h uses the older (d_model, seq_len) layout; here we use
//     the cleaner modern convention.
//   * d_model must be evenly divisible by num_heads.
//   * Multi-head: split d_model into num_heads * head_dim blocks.
//   * Q, K, V, O all stored as flat (d_model, d_model) tensors; the head
//     blocks are stacked along the row axis (block h occupies rows
//     [h*head_dim : (h+1)*head_dim]).
//   * No biases on Q/K/V/O projections (matches paper; Llama-style).
// ============================================================================

// ============================================================================
// MaskedMultiHeadSelfAttention
//
//   Multi-head self-attention with a CAUSAL mask (the "masked" in
//   "masked multi-head self-attention" of Vaswani §3.2.3).
//
//   forward(input, q_len = input.rows)
//     * input : (n_q, d_model) — same tensor as both Q-source and K/V-source
//     * q_len : number of query positions (default = input.rows). For
//               autoregressive decoding this is < n_kv; the layer still
//               works (n_q != n_kv).
//     * returns: (n_q, d_model)
//
//   The causal mask is applied as an additive −1e9 to the (i, j) score when
//   j > i (key position is in the future relative to query position).
//
//   Backward takes a single grad_output (n_q, d_model); the K/V source
//   gradient flows back to input rows in [0, n_q) — if n_kv > n_q the extra
//   K/V source rows accumulate gradient from Q at later positions only.
// ============================================================================
class MaskedMultiHeadSelfAttention : public Layer {
public:
    MaskedMultiHeadSelfAttention(size_t d_model, size_t num_heads);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "MaskedMultiHeadSelfAttention"; }

    size_t d_model()     const { return d_model_; }
    size_t num_heads()   const { return num_heads_; }
    size_t head_dim()    const { return head_dim_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double scale_;

public:
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // BPTT cache
    Tensor last_input_;
    Tensor last_q_;
    Tensor last_k_;
    Tensor last_v_;
    Tensor last_attn_;        // (num_heads, n_q, n_kv) softmax probs (post-mask)
    Tensor last_head_out_;    // (n_q, d_model)
    size_t last_n_q_;
    size_t last_n_kv_;
};

// ============================================================================
// MultiHeadCrossAttention
//
//   The encoder-decoder bridge: Q comes from the decoder hidden state,
//   K and V come from the encoder output. This is what lets the decoder
//   "attend to" the source sequence.
//
//   forward(decoder_hidden, encoder_output)
//     * decoder_hidden : (n_q, d_model)   — Q source
//     * encoder_output : (n_kv, d_model)  — K and V source
//     * returns: (n_q, d_model)
//
//   Q is projected from decoder_hidden; K and V are projected from
//   encoder_output. There is NO causal mask in cross-attention (each
//   decoder position can attend to every encoder position).
//
//   Backward takes grad_output (n_q, d_model) and returns a Tensor of
//   shape (n_q + n_kv, d_model): the first n_q rows are dL/ddecoder_hidden,
//   the last n_kv rows are dL/dencoder_output.
// ============================================================================
class MultiHeadCrossAttention : public Layer {
public:
    MultiHeadCrossAttention(size_t d_model, size_t num_heads);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q; }
    Tensor get_gradients() const override { return grad_W_q; }
    std::string name() const override { return "MultiHeadCrossAttention"; }

    // Two-source forward + backward — distinct from the Layer single-input
    // interface; use these when wiring the decoder block.
    void forward_two(const Tensor& decoder_hidden,
                     const Tensor& encoder_output,
                     Tensor& out);
    void backward_two(const Tensor& grad_output,
                      Tensor& grad_decoder_hidden,
                      Tensor& grad_encoder_output);

    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    double scale_;

public:
    Tensor W_q, W_k, W_v, W_o;
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;

    // BPTT cache (two-source form)
    Tensor last_decoder_;
    Tensor last_encoder_;
    Tensor last_encoder_cached_;  // for Layer::forward single-input fallback
    Tensor last_q_;
    Tensor last_k_;
    Tensor last_v_;
    Tensor last_attn_;
    Tensor last_head_out_;
    size_t last_n_q_;
    size_t last_n_kv_;
};

// ============================================================================
// TransformerDecoderBlock — full decoder block (Vaswani §3.1, decoder side):
//
//   z1  = LayerNorm(x)
//   sa  = MaskedSelfAttention(z1) + x             (sub-layer 1, residual + LN pre)
//   z2  = LayerNorm(sa)
//   ca  = CrossAttention(z2, encoder_output) + sa (sub-layer 2, residual + LN pre)
//   z3  = LayerNorm(ca)
//   ffn = GELU(Dense1(z3)) → Dense2 + ca          (sub-layer 3, residual + LN pre)
//
//   Output: ffn, shape (n_q, d_model).
// ============================================================================
class TransformerDecoderBlock : public Layer {
public:
    TransformerDecoderBlock(size_t d_model, size_t num_heads, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "TransformerDecoderBlock"; }

    // Two-source form: forward decoder hidden + encoder output, return decoder hidden.
    void forward_two(const Tensor& decoder_in,
                     const Tensor& encoder_out,
                     Tensor& decoder_out);
    // Backward returns grad_decoder and grad_encoder as two output Tensors.
    void backward_two(const Tensor& grad_decoder_out,
                      Tensor& grad_decoder_in,
                      Tensor& grad_encoder_in);

    void set_encoder(Tensor enc) { last_encoder_cached_ = enc; }

private:
    size_t d_model_;
    size_t ffn_dim_;

    LayerNorm ln1_;            // pre-masked-self-attn
    MaskedMultiHeadSelfAttention self_attn_;
    LayerNorm ln2_;            // pre-cross-attn
    MultiHeadCrossAttention cross_attn_;
    LayerNorm ln3_;            // pre-FFN
    Dense ffn_fc1_;
    Dense ffn_fc2_;

    // Cache for the single-input Layer interface (used when no separate
    // encoder pass is needed). The two-source form overrides this.
    Tensor last_encoder_cached_;

    // BPTT cache
    Tensor last_decoder_in_;
    Tensor last_encoder_in_;
    Tensor last_z1_;
    Tensor last_sa_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_ca_out_;
    Tensor last_res2_;
    Tensor last_z3_;
    Tensor last_ffn_pregelu_;  // pre-GELU activations (for GELU backward)
    Tensor last_ffn_hidden_;   // post-GELU activations (for ffn_fc2 backward input)
    Tensor last_ffn_out_;
};

// ============================================================================
// TransformerDecoder — stack of decoder blocks + output projection.
//
//   Output projection: Dense(d_model, out_features) — produces per-token
//   logits over the target vocabulary (typical NMT / seq2seq setup).
// ============================================================================
class TransformerDecoder : public Layer {
public:
    TransformerDecoder(size_t d_model, size_t num_heads, size_t out_features,
                       size_t num_blocks = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "TransformerDecoder"; }

    void forward_two(const Tensor& decoder_in,
                     const Tensor& encoder_out,
                     Tensor& decoder_out);
    void backward_two(const Tensor& grad_decoder_out,
                      Tensor& grad_decoder_in,
                      Tensor& grad_encoder_in);

    void set_encoder(Tensor enc);

private:
    size_t d_model_;
    size_t out_features_;
    std::vector<TransformerDecoderBlock> blocks_;
    Dense classifier_;

    Tensor last_decoder_in_;
    Tensor last_encoder_in_;
};

#endif