// ============================================================================
// Expire-Span Attention — Sukhbaatar et al. 2021 implementation
// ============================================================================

#include "expire_span.h"
#include "../../activations/activations.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

inline Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j)
            if (x[i][j] > row_max) row_max = x[i][j];
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j)
            result[i][j] *= inv;
    }
    return result;
}

inline double gelu_deriv(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    double th = std::tanh(u);
    double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
}

}  // namespace

// ============================================================================
// ExpireSpanAttention
// ============================================================================

ExpireSpanAttention::ExpireSpanAttention(size_t d_model,
                                         size_t num_query_heads,
                                         size_t num_kv_heads,
                                         size_t S_max,
                                         double s_min)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      W_span(1, d_model),
      b_span(1, 1),
      grad_W_q(d_model, d_model),
      grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model),
      grad_W_o(d_model, d_model),
      grad_W_span(1, d_model),
      grad_b_span(1, 1),
      d_model_(d_model),
      num_query_heads_(num_query_heads),
      num_kv_heads_(num_kv_heads),
      S_max_(S_max),
      s_min_(s_min),
      soft_threshold_(1e9),
      temperature_(0.5)
{
    // Validate first — must happen BEFORE we compute head_dim_ or scale_,
    // both of which would div-by-zero on invalid config.
    if (d_model_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: d_model must be > 0");
    }
    if (num_query_heads_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: num_query_heads must be > 0");
    }
    if (num_kv_heads_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: num_kv_heads must be > 0");
    }
    if (d_model_ % num_query_heads_ != 0) {
        throw std::invalid_argument(
            "ExpireSpanAttention: d_model must be evenly divisible by num_query_heads");
    }
    if (num_query_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "ExpireSpanAttention: num_query_heads must be evenly divisible by num_kv_heads");
    }
    if (s_min_ < 0.0) {
        throw std::invalid_argument("ExpireSpanAttention: s_min must be >= 0");
    }

    head_dim_ = d_model_ / num_query_heads_;
    group_size_ = num_query_heads_ / num_kv_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Initialize the active K/V head blocks with small random values.
    // (The remaining tail of W_k/W_v beyond num_kv_heads * head_dim stays 0.)
    {
        Tensor W_q_init = Tensor::random(d_model_, d_model_, 0.02);
        Tensor W_o_init = Tensor::random(d_model_, d_model_, 0.02);
        W_q = W_q_init;
        W_o = W_o_init;
        size_t kv_dim = num_kv_heads_ * head_dim_;
        for (size_t i = 0; i < kv_dim; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                W_k[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 7 + j * 13) % 1000) / 1000.0) - 1.0);
                W_v[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 11 + j * 17) % 1000) / 1000.0) - 1.0);
            }
    }
}

std::vector<Tensor*> ExpireSpanAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &W_span, &b_span};
}

std::vector<Tensor*> ExpireSpanAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_W_span, &grad_b_span};
}

void ExpireSpanAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_W_span.fill(0.0);
    grad_b_span.fill(0.0);
}

void ExpireSpanAttention::update_weights(double learning_rate) {
    W_q    -= grad_W_q    * learning_rate;
    W_k    -= grad_W_k    * learning_rate;
    W_v    -= grad_W_v    * learning_rate;
    W_o    -= grad_W_o    * learning_rate;
    W_span -= grad_W_span * learning_rate;
    b_span -= grad_b_span * learning_rate;
}

Tensor ExpireSpanAttention::forward(const Tensor& input) {
    (void)input;
    throw std::logic_error("ExpireSpanAttention::forward not yet implemented");
}

Tensor ExpireSpanAttention::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output;
    (void)learning_rate;
    throw std::logic_error("ExpireSpanAttention::backward not yet implemented");
}

// ============================================================================
// ExpireSpanBlock — stub (Tasks 6+)
// ============================================================================

ExpireSpanBlock::ExpireSpanBlock(size_t d_model,
                                 size_t num_query_heads,
                                 size_t num_kv_heads,
                                 size_t S_max,
                                 double s_min,
                                 size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      attn_(d_model, num_query_heads, num_kv_heads, S_max, s_min),
      ln1_(d_model),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{
}

std::vector<Tensor*> ExpireSpanBlock::parameters() {
    auto p = attn_.parameters();
    auto p1 = ln1_.parameters();
    auto p2 = ln2_.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), pf1.begin(), pf1.end());
    p.insert(p.end(), pf2.begin(), pf2.end());
    return p;
}

std::vector<Tensor*> ExpireSpanBlock::gradients() {
    auto g = attn_.gradients();
    auto g1 = ln1_.gradients();
    auto g2 = ln2_.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}

void ExpireSpanBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void ExpireSpanBlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

Tensor ExpireSpanBlock::forward(const Tensor& input) {
    (void)input;
    throw std::logic_error("ExpireSpanBlock::forward not yet implemented");
}

Tensor ExpireSpanBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output;
    (void)learning_rate;
    throw std::logic_error("ExpireSpanBlock::backward not yet implemented");
}

// ============================================================================
// ExpireSpanModel — stub (Tasks 6+)
// ============================================================================

ExpireSpanModel::ExpireSpanModel(size_t d_input,
                                 size_t d_model,
                                 size_t d_output,
                                 size_t num_blocks,
                                 size_t num_query_heads,
                                 size_t num_kv_heads,
                                 size_t S_max,
                                 double s_min,
                                 size_t ffn_dim)
    : d_input_(d_input),
      d_model_(d_model),
      d_output_(d_output),
      W_in_(d_input, d_model),
      W_out_(d_model, d_output),
      blocks_()
{
    W_in_.init_weights("xavier");
    W_out_.init_weights("xavier");
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, num_query_heads, num_kv_heads, S_max, s_min, ffn_dim);
    }
}

std::vector<Tensor*> ExpireSpanModel::parameters() {
    std::vector<Tensor*> p;
    auto wi = W_in_.parameters();
    auto wo = W_out_.parameters();
    p.insert(p.end(), wi.begin(), wi.end());
    p.insert(p.end(), wo.begin(), wo.end());
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}

std::vector<Tensor*> ExpireSpanModel::gradients() {
    std::vector<Tensor*> g;
    auto wi = W_in_.gradients();
    auto wo = W_out_.gradients();
    g.insert(g.end(), wi.begin(), wi.end());
    g.insert(g.end(), wo.begin(), wo.end());
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    return g;
}

void ExpireSpanModel::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
}

void ExpireSpanModel::update_weights(double learning_rate) {
    W_in_.update_weights(learning_rate);
    W_out_.update_weights(learning_rate);
    for (auto& b : blocks_) b.update_weights(learning_rate);
}

Tensor ExpireSpanModel::forward(const Tensor& input) {
    (void)input;
    throw std::logic_error("ExpireSpanModel::forward not yet implemented");
}

Tensor ExpireSpanModel::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output;
    (void)learning_rate;
    throw std::logic_error("ExpireSpanModel::backward not yet implemented");
}
