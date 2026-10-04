#include "hyena_dna.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// HyenaDNA — arXiv:2306.15794 §3.1/§3.2/§3.3. See hyena_dna.h for the
// operator derivation and the reference-implementation provenance.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: naive causal depthwise 1-D convolution over a (L, C) tensor.
//   y[t][c] = b[c] + sum_{j=0..S-1, t-j >= 0} W[c][j] * x[t-j][c]
// Causal left-padding only (pad_left = S-1, pad_right = 0), matching the
// reference `nn.Conv1d(..., padding=short_filter_order - 1)` followed by the
// `[..., :l]` truncation, which is a causal conv.
// ---------------------------------------------------------------------------
static Tensor causal_depthwise_conv1d(const Tensor& x, const Tensor& W,
                                      const Tensor& b) {
    size_t L = x.rows;
    size_t C = x.cols;
    size_t S = W.cols;
    Tensor y(L, C);
    for (size_t t = 0; t < L; ++t) {
        for (size_t c = 0; c < C; ++c) {
            double acc = b[c][0];
            for (size_t j = 0; j < S; ++j) {
                if (j > t) break;            // causal: no future taps
                acc += W[c][j] * x[t - j][c];
            }
            y[t][c] = acc;
        }
    }
    return y;
}

// ---------------------------------------------------------------------------
// Helper: causal long convolution of a single (L,) signal with a (L,) filter,
//   y[t] = sum_{s=0..t} k[s] * v[t-s]
// Naive O(L^2) — see the header for why (FFT is a drop-in upgrade).
// ---------------------------------------------------------------------------
static void causal_long_conv_1d(const std::vector<double>& v,
                                const std::vector<double>& k, size_t L,
                                std::vector<double>& y) {
    y.assign(L, 0.0);
    for (size_t t = 0; t < L; ++t) {
        double acc = 0.0;
        for (size_t s = 0; s <= t; ++s) acc += k[s] * v[t - s];
        y[t] = acc;
    }
}

// ===========================================================================
// HyenaDNAFilter
// ===========================================================================

HyenaDNAFilter::HyenaDNAFilter(size_t d_model, size_t l_max, size_t filter_order,
                               size_t emb_dim, size_t num_inner)
    : d_model_(d_model), l_max_(l_max), filter_order_(filter_order),
      emb_dim_(emb_dim), num_inner_(num_inner),
      mlp_in_W(Tensor::zeros(filter_order, emb_dim)),
      mlp_in_b(Tensor::zeros(1, filter_order)),
      sin_freq(Tensor::zeros(1, filter_order)),
      mlp_out_W(Tensor::zeros(d_model, filter_order)),
      deltas(Tensor::zeros(1, d_model)),
      bias(Tensor::zeros(1, d_model)) {
    // Validate before allocating: Tensor(0, N) is a legal but useless object,
    // and the shipped repo convention is to throw on a degenerate config.
    if (d_model == 0) throw std::invalid_argument("HyenaDNAFilter: d_model must be > 0");
    if (l_max == 0) throw std::invalid_argument("HyenaDNAFilter: l_max must be > 0");
    if (filter_order == 0)
        throw std::invalid_argument("HyenaDNAFilter: filter_order must be > 0");
    // The reference implementation asserts emb_dim is odd and >= 3 because the
    // positional embedding is [time, cos bands, sin bands].
    if (emb_dim < 3 || emb_dim % 2 == 0)
        throw std::invalid_argument("HyenaDNAFilter: emb_dim must be odd and >= 3");

    mlp_in_W = Tensor::random(filter_order, emb_dim, 0.3);
    for (size_t j = 0; j < filter_order; ++j) sin_freq[0][j] = 1.0;

    mlp_W.resize(num_inner);
    mlp_b.resize(num_inner);
    for (size_t i = 0; i < num_inner; ++i) {
        mlp_W[i] = Tensor::random(filter_order, filter_order, 0.3);
        mlp_b[i] = Tensor::zeros(1, filter_order);
    }
    // Final projection has NO bias (reference: nn.Linear(order, d_model, bias=False)).
    mlp_out_W = Tensor::random(d_model, filter_order, 0.3);

    grad_mlp_in_W = Tensor::zeros(filter_order, emb_dim);
    grad_mlp_in_b = Tensor::zeros(1, filter_order);
    grad_sin_freq = Tensor::zeros(1, filter_order);
    grad_mlp_W.resize(num_inner);
    grad_mlp_b.resize(num_inner);
    for (size_t i = 0; i < num_inner; ++i) {
        grad_mlp_W[i] = Tensor::zeros(filter_order, filter_order);
        grad_mlp_b[i] = Tensor::zeros(1, filter_order);
    }
    grad_mlp_out_W = Tensor::zeros(d_model, filter_order);
    grad_deltas = Tensor::zeros(1, d_model);
    grad_bias = Tensor::zeros(1, d_model);
}

Tensor HyenaDNAFilter::positional_embedding(size_t L) const {
    if (L > l_max_)
        throw std::invalid_argument("HyenaDNAFilter: L > l_max");
    Tensor z(L, emb_dim_);
    size_t bands = (emb_dim_ - 1) / 2;
    const double TWO_PI = 2.0 * 3.14159265358979323846;
    for (size_t l = 0; l < L; ++l) {
        // t normalized so t_{L-1} = 1 — the reference comment calls this
        // "the time embedding fed to the filters is normalized so that t_f = 1".
        z[l][0] = (L == 1) ? 1.0 : (double)l / (double)(L - 1);
        if (bands == 0) continue;
        // t_rescaled = linspace(0, L-1, L); w = 2*pi*t_rescaled/L
        double w = TWO_PI * (double)l / (double)L;
        for (size_t b = 0; b < bands; ++b) {
            // f = linspace(1e-4, bands-1, bands)
            double f;
            if (bands == 1) {
                f = 1e-4;
            } else {
                f = 1e-4 + (double)b * (double)(bands - 1 - 1e-4) / (double)(bands - 1);
            }
            // z = exp(-i*f*w) => [cos(f*w), -sin(f*w)]
            z[l][1 + b] = std::cos(f * w);
            z[l][1 + bands + b] = -std::sin(f * w);
        }
    }
    return z;
}

Tensor HyenaDNAFilter::filter(size_t L) {
    throw std::runtime_error("HyenaDNAFilter::filter: not implemented");
}

void HyenaDNAFilter::backward(const Tensor&) {
    throw std::runtime_error("HyenaDNAFilter::backward: not implemented");
}

void HyenaDNAFilter::zero_grad() {
    grad_mlp_in_W.fill(0.0);
    grad_mlp_in_b.fill(0.0);
    grad_sin_freq.fill(0.0);
    for (size_t i = 0; i < num_inner_; ++i) {
        grad_mlp_W[i].fill(0.0);
        grad_mlp_b[i].fill(0.0);
    }
    grad_mlp_out_W.fill(0.0);
    grad_deltas.fill(0.0);
    grad_bias.fill(0.0);
}

void HyenaDNAFilter::update_weights(double lr) {
    // parameters() and gradients() are index-aligned by construction (see the
    // two methods below), so a positional zip is the whole step.
    std::vector<Tensor*> ps = parameters();
    std::vector<Tensor*> gs = gradients();
    for (size_t i = 0; i < ps.size(); ++i)
        for (size_t r = 0; r < ps[i]->rows; ++r)
            for (size_t c = 0; c < ps[i]->cols; ++c)
                (*ps[i])[r][c] -= lr * (*gs[i])[r][c];
}

std::vector<Tensor*> HyenaDNAFilter::parameters() {
    std::vector<Tensor*> p = {&mlp_in_W, &mlp_in_b, &sin_freq};
    for (size_t i = 0; i < num_inner_; ++i) {
        p.push_back(&mlp_W[i]);
        p.push_back(&mlp_b[i]);
    }
    p.push_back(&mlp_out_W);
    p.push_back(&deltas);
    p.push_back(&bias);
    return p;
}

std::vector<Tensor*> HyenaDNAFilter::gradients() {
    std::vector<Tensor*> g = {&grad_mlp_in_W, &grad_mlp_in_b, &grad_sin_freq};
    for (size_t i = 0; i < num_inner_; ++i) {
        g.push_back(&grad_mlp_W[i]);
        g.push_back(&grad_mlp_b[i]);
    }
    g.push_back(&grad_mlp_out_W);
    g.push_back(&grad_deltas);
    g.push_back(&grad_bias);
    return g;
}

// ===========================================================================
// HyenaDNAOperator
// ===========================================================================

HyenaDNAOperator::HyenaDNAOperator(size_t d_model, size_t l_max, size_t num_heads,
                                   size_t order, size_t filter_order,
                                   size_t short_filter_order)
    : d_model_(d_model), l_max_(l_max), num_heads_(num_heads), order_(order),
      filter_order_(filter_order), short_filter_order_(short_filter_order),
      // The `? : 1` guards keep every Tensor/Dense constructor well-formed
      // (no zero-size allocation) AND keep the filter's width expression
      // division-free, so the validation in the body can still throw before
      // any real allocation. Same placeholder-then-validate idiom as MinLSTM.
      in_proj(d_model ? d_model : 1,
              (order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1),
      out_proj(d_model ? d_model : 1, d_model ? d_model : 1),
      short_W(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1,
                            short_filter_order ? short_filter_order : 1)),
      short_b(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1, 1)),
      grad_short_W(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1,
                                 short_filter_order ? short_filter_order : 1)),
      grad_short_b(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1, 1)),
      filter(num_heads && d_model && order >= 2
                 ? (d_model / num_heads) * (order - 1)
                 : 1,
             l_max ? l_max : 1, filter_order ? filter_order : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNAOperator: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNAOperator: l_max must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("HyenaDNAOperator: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("HyenaDNAOperator: d_model must be divisible by num_heads");
    // The reference implementation asserts order >= 2.
    if (order < 2)
        throw std::invalid_argument("HyenaDNAOperator: order must be >= 2");
    if (short_filter_order == 0)
        throw std::invalid_argument("HyenaDNAOperator: short_filter_order must be > 0");
    if (filter_order == 0)
        throw std::invalid_argument("HyenaDNAOperator: filter_order must be > 0");

    head_dim_ = d_model / num_heads;

    short_W = Tensor::random((order + 1) * d_model, short_filter_order, 0.2);
}

Tensor HyenaDNAOperator::forward(const Tensor&) {
    throw std::runtime_error("HyenaDNAOperator::forward: not implemented");
}

Tensor HyenaDNAOperator::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNAOperator::backward: not implemented");
}

void HyenaDNAOperator::update_weights(double lr) {
    in_proj.update_weights(lr);
    out_proj.update_weights(lr);
    for (size_t i = 0; i < short_W.rows; ++i) {
        for (size_t j = 0; j < short_W.cols; ++j)
            short_W[i][j] -= lr * grad_short_W[i][j];
        short_b[i][0] -= lr * grad_short_b[i][0];
    }
    // The filter holds raw tensors, not Dense layers, so its SGD step is its
    // own method rather than a Dense::update_weights call.
    filter.update_weights(lr);
}

Tensor HyenaDNAOperator::get_weights() const { return in_proj.weights; }
Tensor HyenaDNAOperator::get_gradients() const { return in_proj.grad_weights; }

std::vector<Tensor*> HyenaDNAOperator::parameters() {
    std::vector<Tensor*> p = in_proj.parameters();
    for (auto* q : filter.parameters()) p.push_back(q);
    p.push_back(&out_proj.weights);
    p.push_back(&out_proj.bias);
    p.push_back(&short_W);
    p.push_back(&short_b);
    return p;
}

std::vector<Tensor*> HyenaDNAOperator::gradients() {
    std::vector<Tensor*> g = in_proj.gradients();
    for (auto* q : filter.gradients()) g.push_back(q);
    g.push_back(&out_proj.grad_weights);
    g.push_back(&out_proj.grad_bias);
    g.push_back(&grad_short_W);
    g.push_back(&grad_short_b);
    return g;
}

void HyenaDNAOperator::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    filter.zero_grad();
    grad_short_W.fill(0.0);
    grad_short_b.fill(0.0);
}

// ===========================================================================
// HyenaDNABlock
// ===========================================================================

HyenaDNABlock::HyenaDNABlock(size_t d_model, size_t l_max, size_t num_heads,
                             size_t order, size_t filter_order, size_t ffn_mult)
    : d_model_(d_model), l_max_(l_max), ffn_mult_(ffn_mult),
      ln1(d_model ? d_model : 1), ln2(d_model ? d_model : 1),
      op(d_model, l_max, num_heads, order, filter_order),
      ffn1(d_model ? d_model : 1,
           (ffn_mult > 0) ? ffn_mult * (d_model ? d_model : 1) : 1),
      ffn2((ffn_mult > 0) ? ffn_mult * (d_model ? d_model : 1) : 1,
           d_model ? d_model : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNABlock: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNABlock: l_max must be > 0");
}

Tensor HyenaDNABlock::forward(const Tensor&) {
    throw std::runtime_error("HyenaDNABlock::forward: not implemented");
}

Tensor HyenaDNABlock::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNABlock::backward: not implemented");
}

void HyenaDNABlock::update_weights(double lr) {
    op.update_weights(lr);
    if (ffn_mult_ > 0) {
        ffn1.update_weights(lr);
        ffn2.update_weights(lr);
    }
    ln1.update_weights(lr);
    ln2.update_weights(lr);
}

Tensor HyenaDNABlock::get_weights() const { return op.in_proj.weights; }
Tensor HyenaDNABlock::get_gradients() const { return op.in_proj.grad_weights; }

std::vector<Tensor*> HyenaDNABlock::parameters() {
    std::vector<Tensor*> p = ln1.parameters();
    for (auto* q : op.parameters()) p.push_back(q);
    if (ffn_mult_ > 0) {
        for (auto* q : ffn1.parameters()) p.push_back(q);
        for (auto* q : ffn2.parameters()) p.push_back(q);
    }
    for (auto* q : ln2.parameters()) p.push_back(q);
    return p;
}

std::vector<Tensor*> HyenaDNABlock::gradients() {
    std::vector<Tensor*> g = ln1.gradients();
    for (auto* q : op.gradients()) g.push_back(q);
    if (ffn_mult_ > 0) {
        for (auto* q : ffn1.gradients()) g.push_back(q);
        for (auto* q : ffn2.gradients()) g.push_back(q);
    }
    for (auto* q : ln2.gradients()) g.push_back(q);
    return g;
}

void HyenaDNABlock::zero_grad() {
    ln1.zero_grad();
    op.zero_grad();
    if (ffn_mult_ > 0) {
        ffn1.zero_grad();
        ffn2.zero_grad();
    }
    ln2.zero_grad();
}

// ===========================================================================
// HyenaDNAModel
// ===========================================================================

HyenaDNAModel::HyenaDNAModel(size_t d_model, size_t l_max, size_t depth,
                             size_t num_classes, size_t num_heads, size_t order,
                             size_t filter_order, size_t ffn_mult)
    : d_model_(d_model), l_max_(l_max), depth_(depth), num_classes_(num_classes),
      classifier(d_model ? d_model : 1, num_classes ? num_classes : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNAModel: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNAModel: l_max must be > 0");
    if (depth == 0)
        throw std::invalid_argument("HyenaDNAModel: depth must be > 0");
    if (num_classes == 0)
        throw std::invalid_argument("HyenaDNAModel: num_classes must be > 0");

    blocks.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks.emplace_back(d_model, l_max, num_heads, order, filter_order,
                           ffn_mult);
    }
}

Tensor HyenaDNAModel::forward(const Tensor&) {
    throw std::runtime_error("HyenaDNAModel::forward: not implemented");
}

Tensor HyenaDNAModel::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNAModel::backward: not implemented");
}

void HyenaDNAModel::update_weights(double lr) {
    for (size_t i = 0; i < depth_; ++i) blocks[i].update_weights(lr);
    classifier.update_weights(lr);
}

Tensor HyenaDNAModel::get_weights() const {
    return blocks.empty() ? Tensor(0, 0) : blocks[0].op.in_proj.weights;
}
Tensor HyenaDNAModel::get_gradients() const {
    return blocks.empty() ? Tensor(0, 0) : blocks[0].op.in_proj.grad_weights;
}

std::vector<Tensor*> HyenaDNAModel::parameters() {
    std::vector<Tensor*> p;
    for (size_t i = 0; i < depth_; ++i)
        for (auto* q : blocks[i].parameters()) p.push_back(q);
    for (auto* q : classifier.parameters()) p.push_back(q);
    return p;
}

std::vector<Tensor*> HyenaDNAModel::gradients() {
    std::vector<Tensor*> g;
    for (size_t i = 0; i < depth_; ++i)
        for (auto* q : blocks[i].gradients()) g.push_back(q);
    for (auto* q : classifier.gradients()) g.push_back(q);
    return g;
}

void HyenaDNAModel::zero_grad() {
    for (size_t i = 0; i < depth_; ++i) blocks[i].zero_grad();
    classifier.zero_grad();
}

// ===========================================================================
// SequenceLengthWarmup (arXiv:2306.15794 §3.2)
// ===========================================================================

SequenceLengthWarmup::SequenceLengthWarmup(size_t start_len, size_t num_stages,
                                           size_t epoch_scale)
    : start_len_(start_len), num_stages_(num_stages), epoch_scale_(epoch_scale) {
    if (start_len == 0)
        throw std::invalid_argument("SequenceLengthWarmup: start_len must be > 0");
    if (num_stages == 0)
        throw std::invalid_argument("SequenceLengthWarmup: num_stages must be > 0");
    if (epoch_scale == 0)
        throw std::invalid_argument("SequenceLengthWarmup: epoch_scale must be > 0");
}

size_t SequenceLengthWarmup::stage_for_epoch(size_t epoch) const {
    size_t stage = 0;
    size_t stage_start = 0;
    for (size_t s = 0; s < num_stages_; ++s) {
        size_t stage_len = epoch_scale_ << s;   // epoch_scale * 2^s
        if (epoch < stage_start + stage_len) return s;
        stage_start += stage_len;
        ++stage;
    }
    return num_stages_ - 1;   // saturate at the final stage
}

size_t SequenceLengthWarmup::seq_len_for_epoch(size_t epoch) const {
    return start_len_ << stage_for_epoch(epoch);
}

size_t SequenceLengthWarmup::total_epochs() const {
    return epoch_scale_ * ((size_t(1) << num_stages_) - 1);
}

size_t SequenceLengthWarmup::max_seq_len() const {
    return start_len_ << (num_stages_ - 1);
}

// ===========================================================================
// SoftPrompting (arXiv:2306.15794 §3.3, Eq. 3.2)
// ===========================================================================

SoftPrompting::SoftPrompting(size_t prompt_len, size_t d_model, bool at_front)
    : prompt_len_(prompt_len), d_model_(d_model), at_front_(at_front) {
    if (prompt_len == 0)
        throw std::invalid_argument("SoftPrompting: prompt_len must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("SoftPrompting: d_model must be > 0");
    prompt_ = Tensor::random(prompt_len, d_model, 0.02);
    grad_prompt_ = Tensor::zeros(prompt_len, d_model);
}

Tensor SoftPrompting::forward(const Tensor& embedded) {
    if (embedded.cols != d_model_)
        throw std::invalid_argument("SoftPrompting: embedded d_model mismatch");
    last_embedded = embedded;
    size_t T = embedded.rows;
    Tensor out(T + prompt_len_, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t c = 0; c < d_model_; ++c)
            out[at_front_ ? (t + prompt_len_) : t][c] = embedded[t][c];
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            out[at_front_ ? n : (T + n)][c] = prompt_[n][c];
    return out;
}

Tensor SoftPrompting::backward(const Tensor& grad_output, double) {
    if (grad_output.rows != last_embedded.rows + prompt_len_)
        throw std::invalid_argument("SoftPrompting: grad_output shape mismatch");
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            grad_prompt_[n][c] += grad_output[at_front_ ? n : (last_embedded.rows + n)][c];
    // The embedded input is fixed (a frozen embedding table in the paper's
    // setting), so the gradient w.r.t. it is zero.
    return Tensor::zeros(last_embedded.rows, last_embedded.cols);
}

void SoftPrompting::update_weights(double lr) {
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            prompt_[n][c] -= lr * grad_prompt_[n][c];
}

void SoftPrompting::zero_grad() { grad_prompt_.fill(0.0); }
