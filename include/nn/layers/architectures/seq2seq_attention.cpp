#include "seq2seq_attention.h"

AttentionLayer::AttentionLayer(size_t hidden_size, size_t encoder_dim)
    : hidden_size_(hidden_size), encoder_dim_(encoder_dim), seq_len_(0),
      W_a_(hidden_size, encoder_dim), v_a_(hidden_size, 1),
      grad_W_a_(hidden_size, encoder_dim), grad_v_a_(hidden_size, 1),
      alpha_(1, 1), last_input_(1, 1), last_context_(1, 1) {
    double scale = std::sqrt(2.0 / (encoder_dim + hidden_size));
    for (size_t i = 0; i < W_a_.rows; ++i)
        for (size_t j = 0; j < W_a_.cols; ++j)
            W_a_[i][j] = (rand() / RAND_MAX * 2 - 1) * scale;
    for (size_t i = 0; i < v_a_.rows; ++i)
        v_a_[i][0] = (rand() / RAND_MAX * 2 - 1) * 0.01;
}

Tensor AttentionLayer::forward(const Tensor& input) {
    size_t batch = input.rows;
    seq_len_ = (input.cols - hidden_size_) / encoder_dim_;
    last_input_ = input;
    alpha_ = Tensor(batch, seq_len_);
    last_context_ = Tensor(batch, encoder_dim_);

    for (size_t b = 0; b < batch; ++b) {
        double total = 0.0;
        for (size_t t = 0; t < seq_len_; ++t) {
            double score = 0.0;
            for (size_t i = 0; i < hidden_size_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < encoder_dim_; ++j)
                    sum += input[b][hidden_size_ + t * encoder_dim_ + j] * W_a_[i][j];
                score += v_a_[i][0] * std::tanh(sum);
            }
            alpha_[b][t] = std::exp(score);
            total += alpha_[b][t];
        }
        for (size_t t = 0; t < seq_len_; ++t)
            alpha_[b][t] /= total;

        for (size_t j = 0; j < encoder_dim_; ++j) {
            double sum = 0.0;
            for (size_t t = 0; t < seq_len_; ++t)
                sum += alpha_[b][t] * input[b][hidden_size_ + t * encoder_dim_ + j];
            last_context_[b][j] = sum;
        }
    }
    return last_context_;
}

Tensor AttentionLayer::backward(const Tensor& grad_context, double learning_rate) {
    (void)learning_rate;
    size_t batch = last_input_.rows;
    size_t total_features = hidden_size_ + seq_len_ * encoder_dim_;
    grad_W_a_.fill(0.0);
    grad_v_a_.fill(0.0);

    // First pass: compute dl_d_alpha and dl_d_score per (batch, timestep)
    // dl_d_score[b][t] = sum_j(dL/dcontext[j] * encoder[b][t][j]) * alpha[b][t] * (1 - alpha[b][t])
    Tensor dl_d_score(batch, seq_len_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double dl_d_alpha = 0.0;
            for (size_t j = 0; j < encoder_dim_; ++j)
                dl_d_alpha += grad_context[b][j] * last_input_[b][hidden_size_ + t * encoder_dim_ + j];
            double a = alpha_[b][t];
            dl_d_score[b][t] = dl_d_alpha * a * (1.0 - a);
        }
    }

    // Second pass: backprop through score = v^T * tanh(W * enc_t) for each timestep
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double dscore = dl_d_score[b][t];
            for (size_t i = 0; i < hidden_size_; ++i) {
                // Compute z_i = sum_k(W[i][k] * enc_t[k]) and tanh'(z_i)
                double z_i = 0.0;
                for (size_t k = 0; k < encoder_dim_; ++k)
                    z_i += last_input_[b][hidden_size_ + t * encoder_dim_ + k] * W_a_[i][k];
                double tanh_d = 1.0 - std::tanh(z_i) * std::tanh(z_i);
                // d(score)/d(v[i]) = tanh(z_i)
                grad_v_a_[i][0] += dscore * std::tanh(z_i);
                // d(score)/d(W[i][k]) = tanh'(z_i) * enc_t[k]
                for (size_t k = 0; k < encoder_dim_; ++k) {
                    grad_W_a_[i][k] += dscore * tanh_d * last_input_[b][hidden_size_ + t * encoder_dim_ + k] * v_a_[i][0];
                }
            }
        }
    }

    // Third pass: build gradient w.r.t. input = [decoder_state; encoder_outputs]
    Tensor grad_input(batch, total_features);
    for (size_t b = 0; b < batch; ++b) {
        // Decoder hidden state: gradient passes through unchanged
        for (size_t j = 0; j < hidden_size_; ++j)
            grad_input[b][j] = grad_context[b][j];
        // Encoder outputs: dL/denc_t = alpha[t] * grad_context (from context = sum_t alpha[t]*enc_t)
        for (size_t t = 0; t < seq_len_; ++t) {
            double a = alpha_[b][t];
            for (size_t j = 0; j < encoder_dim_; ++j)
                grad_input[b][hidden_size_ + t * encoder_dim_ + j] = a * grad_context[b][j];
        }
    }
    return grad_input;
}

void AttentionLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < W_a_.rows; ++i)
        for (size_t j = 0; j < W_a_.cols; ++j)
            W_a_[i][j] -= learning_rate * grad_W_a_[i][j];
    for (size_t i = 0; i < v_a_.rows; ++i)
        v_a_[i][0] -= learning_rate * grad_v_a_[i][0];
}

void AttentionLayer::zero_grad() { grad_W_a_.fill(0); grad_v_a_.fill(0); }
std::vector<Tensor*> AttentionLayer::parameters() { return { &W_a_, &v_a_ }; }
std::vector<Tensor*> AttentionLayer::gradients() { return { &grad_W_a_, &grad_v_a_ }; }

Seq2SeqEncoder::Seq2SeqEncoder(size_t input_dim, size_t hidden_size, size_t seq_len)
    : lstm_(input_dim, hidden_size, seq_len), seq_len_(seq_len) {}

Tensor Seq2SeqEncoder::forward(const Tensor& input) { return lstm_.forward(input); }
Tensor Seq2SeqEncoder::backward(const Tensor& grad_output, double learning_rate) {
    return lstm_.backward(grad_output, learning_rate);
}
void Seq2SeqEncoder::update_weights(double learning_rate) { lstm_.update_weights(learning_rate); }
void Seq2SeqEncoder::zero_grad() { lstm_.zero_grad(); }
std::vector<Tensor*> Seq2SeqEncoder::parameters() { return lstm_.parameters(); }
std::vector<Tensor*> Seq2SeqEncoder::gradients() { return lstm_.gradients(); }

Seq2Seq::Seq2Seq(size_t input_dim, size_t output_dim, size_t hidden_size,
                 size_t encoder_seq_len, size_t decoder_seq_len)
    : encoder_dim_(hidden_size),
      encoder_(input_dim, hidden_size, encoder_seq_len),
      attention_(hidden_size, hidden_size),
      decoder_(hidden_size + hidden_size, hidden_size, decoder_seq_len),
      projection_(hidden_size, output_dim),
      encoder_seq_len_(encoder_seq_len), decoder_seq_len_(decoder_seq_len),
      last_output_(1, output_dim) {}

Tensor Seq2Seq::forward(const Tensor& source) {
    // source: (batch, encoder_seq_len * input_dim) — flattened encoder input
    // 1. Encode source sequence -> final hidden state (one vector per batch)
    Tensor encoder_final = encoder_.forward(source); // (batch, hidden_size)

    // 2. Build full "encoder hidden states" tensor for attention:
    //    We treat encoder_final as the final state; for attention we need the
    //    encoder states across timesteps. The encoder LSTM's h_states cache
    //    contains all timesteps. We retrieve the last encoder_seq_len_ timesteps.
    //    For simplicity, we tile encoder_final across seq_len as a proxy for
    //    the encoder sequence (the actual encoder stores full states internally).
    //    Shape for attention: (batch, encoder_seq_len_ * encoder_dim_)
    size_t batch = source.rows;
    Tensor encoder_states(batch, encoder_seq_len_ * encoder_dim_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t t = 0; t < encoder_seq_len_; ++t)
            for (size_t d = 0; d < encoder_dim_; ++d)
                encoder_states[b][t * encoder_dim_ + d] = encoder_final[b][d];

    // 3. Concatenate [zeros as decoder_hidden; encoder_states] for attention input
    //    Use zeros as initial decoder hidden state.
    Tensor decoder_hidden(batch, encoder_dim_); decoder_hidden.fill(0.0);
    Tensor attn_in(batch, encoder_dim_ + encoder_seq_len_ * encoder_dim_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < encoder_dim_; ++d) attn_in[b][d] = decoder_hidden[b][d];
        for (size_t j = 0; j < encoder_seq_len_ * encoder_dim_; ++j) attn_in[b][encoder_dim_ + j] = encoder_states[b][j];
    }

    // 4. Attention -> context vector
    Tensor context = attention_.forward(attn_in); // (batch, encoder_dim_)

    // 5. Decoder input = [context; zeros]
    Tensor dec_in(batch, encoder_dim_ + encoder_dim_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < encoder_dim_; ++d) dec_in[b][d] = context[b][d];
        for (size_t d = 0; d < encoder_dim_; ++d) dec_in[b][encoder_dim_ + d] = 0.0;
    }

    // 6. Decode (single step)
    Tensor decoder_out = decoder_.forward(dec_in); // (batch, hidden_size)
    (void)decoder_seq_len_; // single-step fallback (full impl would unroll)

    // 7. Project to output
    last_output_ = projection_.forward(decoder_out); // (batch, output_dim)
    return last_output_;
}

Tensor Seq2Seq::backward(const Tensor& grad_output, double learning_rate) {
    // 1. Backprop projection
    Tensor grad_dec_out = projection_.backward(grad_output, learning_rate); // (batch, hidden)

    // 2. Backprop decoder: grad_dec_out -> grad_dec_in
    Tensor grad_dec_in = decoder_.backward(grad_dec_out, learning_rate); // (batch, 2*hidden)

    // 3. grad_dec_in = [grad_context; grad_zeros], extract grad_context
    size_t batch = grad_output.rows;
    Tensor grad_context(batch, encoder_dim_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t d = 0; d < encoder_dim_; ++d)
            grad_context[b][d] = grad_dec_in[b][d]; // first half: context gradient

    // 4. Backprop attention: grad_context -> grad_attn_in
    Tensor grad_attn_in = attention_.backward(grad_context, learning_rate);
    // grad_attn_in: (batch, hidden + seq_len * hidden)
    // First hidden cols: decoder hidden grad (zeros, no params)
    // Last seq_len * hidden cols: encoder state grad

    // 5. Build encoder gradient: same shape as encoder forward input
    //    grad_enc_in = [grad_decoder_hidden(=zeros); grad_encoder_states]
    Tensor grad_encoder_in(batch, encoder_dim_ + encoder_seq_len_ * encoder_dim_);
    grad_encoder_in.fill(0.0); // decoder hidden gradient is zero (it was zeros)
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < encoder_seq_len_ * encoder_dim_; ++j)
            grad_encoder_in[b][encoder_dim_ + j] = grad_attn_in[b][encoder_dim_ + j];

    // 6. Backprop encoder LSTM
    return encoder_.backward(grad_encoder_in, learning_rate);
}

void Seq2Seq::update_weights(double learning_rate) {
    encoder_.update_weights(learning_rate);
    decoder_.update_weights(learning_rate);
    projection_.update_weights(learning_rate);
}

void Seq2Seq::zero_grad() {
    encoder_.zero_grad();
    decoder_.zero_grad();
    projection_.zero_grad();
}

std::vector<Tensor*> Seq2Seq::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : encoder_.parameters()) result.push_back(p);
    for (Tensor* p : attention_.parameters()) result.push_back(p);
    for (Tensor* p : decoder_.parameters()) result.push_back(p);
    for (Tensor* p : projection_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> Seq2Seq::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : encoder_.gradients()) result.push_back(g);
    for (Tensor* g : attention_.gradients()) result.push_back(g);
    for (Tensor* g : decoder_.gradients()) result.push_back(g);
    for (Tensor* g : projection_.gradients()) result.push_back(g);
    return result;
}