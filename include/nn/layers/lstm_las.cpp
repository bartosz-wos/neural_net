#include "lstm_las.h"

LASEncoder::LASEncoder(size_t input_dim, size_t hidden_size, size_t num_layers)
    : input_dim_(input_dim), hidden_size_(hidden_size), num_layers_(num_layers),
      last_output_(1, 1) {

    for (size_t i = 0; i < num_layers_; ++i)
        lstm_layers_.emplace_back(input_dim, hidden_size, 1);
}

Tensor LASEncoder::forward(const Tensor& input) {
    seq_len_ = input.rows;
    h_states_ = Tensor(seq_len_, hidden_size_);

    for (size_t t = 0; t < seq_len_; ++t) {
        Tensor timestep(1, input_dim_);
        for (size_t j = 0; j < input_dim_; ++j)
            timestep[0][j] = input[t][j];

        // Pass through all LSTM layers
        Tensor layer_output = timestep;
        for (size_t l = 0; l < num_layers_; ++l)
            layer_output = lstm_layers_[l].forward(layer_output);
        for (size_t h = 0; h < hidden_size_; ++h)
            h_states_[t][h] = layer_output[0][h];
    }
    return h_states_;
}

Tensor LASEncoder::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void LASEncoder::update_weights(double learning_rate) {
    for (auto& l : lstm_layers_) l.update_weights(learning_rate);
}

void LASEncoder::zero_grad() {
    for (auto& l : lstm_layers_) l.zero_grad();
}

std::vector<Tensor*> LASEncoder::parameters() {
    std::vector<Tensor*> result;
    for (auto& l : lstm_layers_)
        for (Tensor* p : l.parameters())
            result.push_back(p);
    return result;
}

std::vector<Tensor*> LASEncoder::gradients() {
    std::vector<Tensor*> result;
    for (auto& l : lstm_layers_)
        for (Tensor* g : l.gradients())
            result.push_back(g);
    return result;
}

// === LASSelfAttention ===

LASSelfAttention::LASSelfAttention(size_t hidden_size)
    : hidden_size_(hidden_size),
      query_proj_(1, hidden_size),
      key_proj_(1, hidden_size),
      value_proj_(1, hidden_size) {}

Tensor LASSelfAttention::forward(const Tensor& query, const Tensor& keys,
                                   const Tensor& values) {
    Tensor q = query;
    size_t seq_len = keys.rows;

    // Attention scores: q · k^T / sqrt(hidden)
    Tensor scores(1, seq_len);
    for (size_t t = 0; t < seq_len; ++t) {
        double dot = 0.0;
        for (size_t j = 0; j < hidden_size_; ++j)
            dot += q[0][j] * keys[t][j];
        scores[0][t] = dot / std::sqrt(static_cast<double>(hidden_size_));
    }

    // Softmax
    double max_s = scores[0][0];
    for (size_t t = 1; t < seq_len; ++t)
        max_s = std::max(max_s, scores[0][t]);

    double sum_exp = 0.0;
    for (size_t t = 0; t < seq_len; ++t) {
        scores[0][t] = std::exp(scores[0][t] - max_s);
        sum_exp += scores[0][t];
    }
    for (size_t t = 0; t < seq_len; ++t)
        scores[0][t] /= sum_exp;

    // Weighted sum
    Tensor context(1, hidden_size_);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t j = 0; j < hidden_size_; ++j)
            context[0][j] += scores[0][t] * values[t][j];

    return context;
}

void LASSelfAttention::update_weights(double learning_rate) {
    query_proj_.update_weights(learning_rate);
    key_proj_.update_weights(learning_rate);
    value_proj_.update_weights(learning_rate);
}

void LASSelfAttention::zero_grad() {
    query_proj_.zero_grad();
    key_proj_.zero_grad();
    value_proj_.zero_grad();
}

std::vector<Tensor*> LASSelfAttention::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : query_proj_.parameters()) result.push_back(p);
    for (Tensor* p : key_proj_.parameters()) result.push_back(p);
    for (Tensor* p : value_proj_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> LASSelfAttention::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : query_proj_.gradients()) result.push_back(g);
    for (Tensor* g : key_proj_.gradients()) result.push_back(g);
    for (Tensor* g : value_proj_.gradients()) result.push_back(g);
    return result;
}

// === LASDecoder ===

LASDecoder::LASDecoder(size_t input_dim, size_t hidden_size,
                       const Tensor& encoder_hidden, size_t encoder_seq_len)
    : controller_(input_dim, hidden_size, 1),
      attention_(hidden_size),
      context_proj_(hidden_size + input_dim, input_dim),
      output_proj_(hidden_size, input_dim),
      encoder_hidden_(encoder_hidden),
      encoder_seq_len_(encoder_seq_len),
      hidden_size_(hidden_size),
      last_output_(1, input_dim) {}

Tensor LASDecoder::forward(const Tensor& input) {
    // Attend
    Tensor context = attention_.forward(input, encoder_hidden_, encoder_hidden_);

    // Concat [input; context]
    Tensor concat(1, hidden_size_ + input.cols);
    for (size_t j = 0; j < (size_t)input.cols; ++j)
        concat[0][j] = input[0][j];
    for (size_t j = 0; j < hidden_size_; ++j)
        concat[0][input.cols + j] = context[0][j];

    last_context_ = context;
    last_output_ = context_proj_.forward(concat);
    return last_output_;
}

Tensor LASDecoder::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void LASDecoder::update_weights(double learning_rate) {
    controller_.update_weights(learning_rate);
    attention_.update_weights(learning_rate);
    context_proj_.update_weights(learning_rate);
    output_proj_.update_weights(learning_rate);
}

void LASDecoder::zero_grad() {
    controller_.zero_grad();
    attention_.zero_grad();
    context_proj_.zero_grad();
    output_proj_.zero_grad();
}

std::vector<Tensor*> LASDecoder::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : controller_.parameters()) result.push_back(p);
    for (Tensor* p : attention_.parameters()) result.push_back(p);
    for (Tensor* p : context_proj_.parameters()) result.push_back(p);
    for (Tensor* p : output_proj_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> LASDecoder::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : controller_.gradients()) result.push_back(g);
    for (Tensor* g : attention_.gradients()) result.push_back(g);
    for (Tensor* g : context_proj_.gradients()) result.push_back(g);
    for (Tensor* g : output_proj_.gradients()) result.push_back(g);
    return result;
}

// === ListenAttendSpell ===

ListenAttendSpell::ListenAttendSpell(size_t vocab_size, size_t embedding_dim,
                                     size_t encoder_hidden, size_t decoder_hidden,
                                     size_t num_layers)
    : encoder_embedding_(vocab_size, embedding_dim),
      encoder_(embedding_dim, encoder_hidden, num_layers),
      decoder_embedding_(vocab_size, embedding_dim),
      decoder_(embedding_dim, decoder_hidden, Tensor(), 1),
      output_layer_(decoder_hidden, vocab_size),
      vocab_size_(vocab_size),
      last_output_(1, vocab_size) {}

Tensor ListenAttendSpell::forward(const Tensor& input, const Tensor& target) {
    (void)target;
    Tensor enc_out = encoder_.forward(input);
    last_output_ = output_layer_.forward(decoder_.forward(input));
    (void)enc_out;
    return last_output_;
}

Tensor ListenAttendSpell::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void ListenAttendSpell::update_weights(double learning_rate) {
    encoder_embedding_.update_weights(learning_rate);
    encoder_.update_weights(learning_rate);
    decoder_embedding_.update_weights(learning_rate);
    decoder_.update_weights(learning_rate);
    output_layer_.update_weights(learning_rate);
}

void ListenAttendSpell::zero_grad() {
    encoder_embedding_.zero_grad();
    encoder_.zero_grad();
    decoder_embedding_.zero_grad();
    decoder_.zero_grad();
    output_layer_.zero_grad();
}

std::vector<Tensor*> ListenAttendSpell::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : encoder_embedding_.parameters()) result.push_back(p);
    for (Tensor* p : encoder_.parameters()) result.push_back(p);
    for (Tensor* p : decoder_embedding_.parameters()) result.push_back(p);
    for (Tensor* p : decoder_.parameters()) result.push_back(p);
    for (Tensor* p : output_layer_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> ListenAttendSpell::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : encoder_embedding_.gradients()) result.push_back(g);
    for (Tensor* g : encoder_.gradients()) result.push_back(g);
    for (Tensor* g : decoder_embedding_.gradients()) result.push_back(g);
    for (Tensor* g : decoder_.gradients()) result.push_back(g);
    for (Tensor* g : output_layer_.gradients()) result.push_back(g);
    return result;
}

Tensor ListenAttendSpell::decode(const Tensor& source, size_t max_len) {
    (void)source; (void)max_len;
    return Tensor(1, vocab_size_);
}