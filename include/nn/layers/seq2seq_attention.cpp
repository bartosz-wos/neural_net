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
    (void)grad_context; (void)learning_rate;
    return Tensor(1, hidden_size_ + seq_len_ * encoder_dim_);
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
    (void)source;
    last_output_ = Tensor(1, 1); // placeholder
    return last_output_;
}

Tensor Seq2Seq::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
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