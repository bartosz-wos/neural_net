#include "xlstm_block.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Kaiming-He-style initialization for Dense weights (used for slstm_proj,
// mlstm_proj, ffn_proj1, ffn_proj2).
void init_dense(Dense& d, size_t in_f, size_t out_f, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / static_cast<double>(in_f)));
    std::vector<double> w(in_f * out_f);
    for (auto& v : w) v = nd(rng);
    d.weights = Tensor(out_f, in_f, w.data());
    d.bias = Tensor(1, out_f);
    std::fill(d.bias.data.begin(), d.bias.data.end(), 0.0);
    d.grad_weights = Tensor(out_f, in_f);
    std::fill(d.grad_weights.data.begin(), d.grad_weights.data.end(), 0.0);
    d.grad_bias = Tensor(1, out_f);
    std::fill(d.grad_bias.data.begin(), d.grad_bias.data.end(), 0.0);
}

// GELU element-wise (closed-form using erf).
double gelu(double x) {
    return x * 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// GELU derivative.
double gelu_deriv(double x) {
    double s = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    return s + x * pdf;
}

// Element-wise addition: c = a + b (same shape).
void tensor_add(const Tensor& a, const Tensor& b, Tensor& c) {
    for (size_t i = 0; i < a.data.size(); ++i) {
        c.data[i] = a.data[i] + b.data[i];
    }
}

}  // namespace

// ============================================================================
// XLSTMBlock
// ============================================================================

XLSTMBlock::XLSTMBlock(size_t d_model, size_t slstm_hidden, size_t mlstm_hidden,
                       size_t ffn_mult, XLSTMCellType cell_type)
    : d_model_(d_model),
      slstm_hidden_(slstm_hidden),
      mlstm_hidden_(cell_type == XLSTMCellType::SLSTM_ONLY ? 0 : mlstm_hidden),
      ffn_mult_(ffn_mult),
      ffn_hidden_(ffn_mult * d_model),
      cell_type_(cell_type),
      ln1_(d_model),
      ln2_(d_model),
      slstm_(d_model, slstm_hidden),
      slstm_proj_(slstm_hidden, d_model),
      mlstm_(d_model, cell_type == XLSTMCellType::SLSTM_ONLY ? 1 : mlstm_hidden),
      mlstm_proj_(cell_type == XLSTMCellType::SLSTM_ONLY ? 1 : mlstm_hidden, d_model),
      ffn_proj1_(d_model, ffn_hidden_),
      ffn_proj2_(ffn_hidden_, d_model) {
    if (d_model == 0) throw std::invalid_argument("XLSTMBlock: d_model must be > 0");
    if (slstm_hidden == 0) throw std::invalid_argument("XLSTMBlock: slstm_hidden must be > 0");
    if (ffn_mult == 0) throw std::invalid_argument("XLSTMBlock: ffn_mult must be > 0");
    if (cell_type != XLSTMCellType::SLSTM_ONLY) {
        if (mlstm_hidden == 0) throw std::invalid_argument("XLSTMBlock: mlstm_hidden must be > 0 when cell_type != SLSTM_ONLY");
    }

    // Initialize dense weights with Kaiming-He.
    init_dense(slstm_proj_, slstm_hidden, d_model, 0xC0FFEE01);
    init_dense(ffn_proj1_, d_model, ffn_hidden_, 0xC0FFEE02);
    init_dense(ffn_proj2_, ffn_hidden_, d_model, 0xC0FFEE03);
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        init_dense(mlstm_proj_, mlstm_hidden_, d_model, 0xC0FFEE04);
    }
}

Tensor XLSTMBlock::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("XLSTMBlock::forward: input feature dim mismatch");
    }
    const size_t T = input.rows;
    last_input = input;

    // --- Mixer sublayer ---
    last_ln1_out = ln1_.forward(input);                              // (T, d)
    last_slstm_h = slstm_.forward(last_ln1_out);                     // (T, slstm_hidden)
    last_slstm_proj = slstm_proj_.forward(last_slstm_h);             // (T, d)

    if (cell_type_ == XLSTMCellType::SLSTM_ONLY) {
        last_mixer = last_slstm_proj;
    } else {
        last_mlstm_h = mlstm_.forward(last_ln1_out);                 // (T, mlstm_hidden)
        last_mlstm_proj = mlstm_proj_.forward(last_mlstm_h);         // (T, d)
        last_mixer = Tensor(T, d_model_);
        tensor_add(last_slstm_proj, last_mlstm_proj, last_mixer);
    }

    // --- Residual: x + mixer ---
    last_residual1 = Tensor(T, d_model_);
    tensor_add(input, last_mixer, last_residual1);

    // --- FFN sublayer ---
    last_ln2_out = ln2_.forward(last_residual1);                     // (T, d)
    last_ffn_hidden = ffn_proj1_.forward(last_ln2_out);              // (T, ffn_hidden)
    last_ffn_act = Tensor(T, ffn_hidden_);
    for (size_t i = 0; i < last_ffn_hidden.rows; ++i)
        for (size_t j = 0; j < last_ffn_hidden.cols; ++j)
            last_ffn_act[i][j] = gelu(last_ffn_hidden[i][j]);
    last_ffn_out = ffn_proj2_.forward(last_ffn_act);                  // (T, d)

    // --- Output: x + mixer + ffn_out ---
    Tensor out(T, d_model_);
    Tensor partial(T, d_model_);
    tensor_add(last_residual1, last_ffn_out, partial);
    for (size_t i = 0; i < partial.data.size(); ++i)
        out.data[i] = partial.data[i];
    return out;
}

Tensor XLSTMBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // Verify caches are populated (forward was called)
    if (last_input.data.empty()) {
        throw std::logic_error("XLSTMBlock::backward: forward was not called");
    }
    const size_t T = last_input.rows;

    // grad_output = gradient w.r.t. out = x + mixer + ffn_out.
    // The "x" residual term contributes grad_output as a passthrough to
    // grad_x. We'll accumulate three terms:
    //   - grad_via_x_residual = grad_output  (always)
    //   - grad_via_mixer      = grad_output  (always)
    //   - grad_via_ffn        = grad_output  (always)
    // and combine at the end.

    // --- FFN sublayer backward ---
    // ffn_out = ffn_proj2(GELU(ffn_proj1(ln2(x + mixer))))
    Tensor grad_ffn_act = ffn_proj2_.backward(grad_output, 0.0);     // (T, ffn_hidden)
    // GELU backward (apply in-place)
    for (size_t i = 0; i < grad_ffn_act.rows; ++i)
        for (size_t j = 0; j < grad_ffn_act.cols; ++j)
            grad_ffn_act[i][j] *= gelu_deriv(last_ffn_hidden[i][j]);
    Tensor grad_ln2_out = ffn_proj1_.backward(grad_ffn_act, 0.0);   // (T, d)

    // ln2 backward returns gradient w.r.t. (x + mixer) input to ln2.
    Tensor grad_residual1_from_ffn = ln2_.backward(grad_ln2_out, 0.0);  // (T, d)

    // --- Mixer sublayer backward ---
    // grad_residual1_from_ffn goes to (x + mixer), so we split into:
    //   - +grad_residual1_from_ffn to grad_x (the residual passthrough)
    //   - +grad_residual1_from_ffn to grad_mixer (which is split between slstm and mlstm)

    // First, grad_mixer = grad_residual1_from_ffn + grad_output (residual)
    Tensor grad_mixer = Tensor(T, d_model_);
    for (size_t i = 0; i < grad_mixer.data.size(); ++i)
        grad_mixer.data[i] = grad_residual1_from_ffn.data[i] + grad_output.data[i];

    // Mixer's contributions to grad_ln1_out (= LN_1(x)):
    Tensor grad_ln1_out = Tensor(T, d_model_);
    std::fill(grad_ln1_out.data.begin(), grad_ln1_out.data.end(), 0.0);

    // sLSTM path: grad_mixer -> slstm_proj backward -> grad_slstm_h -> slstm backward -> grad_ln1_out
    Tensor grad_slstm_h = slstm_proj_.backward(grad_mixer, 0.0);   // (T, slstm_hidden)
    Tensor grad_ln1_via_slstm = slstm_.backward(grad_slstm_h, 0.0); // (T, d)
    for (size_t i = 0; i < grad_ln1_out.data.size(); ++i)
        grad_ln1_out.data[i] += grad_ln1_via_slstm.data[i];

    // mLSTM path (if present)
    Tensor grad_ln1_via_mlstm;
    bool have_mlstm = (cell_type_ != XLSTMCellType::SLSTM_ONLY);
    if (have_mlstm) {
        Tensor grad_mlstm_h = mlstm_proj_.backward(grad_mixer, 0.0);   // (T, mlstm_hidden)
        grad_ln1_via_mlstm = mlstm_.backward(grad_mlstm_h, 0.0);       // (T, d)
        for (size_t i = 0; i < grad_ln1_out.data.size(); ++i)
            grad_ln1_out.data[i] += grad_ln1_via_mlstm.data[i];
    }

    // LN_1 backward returns gradient w.r.t. x.
    Tensor grad_x_from_mixer = ln1_.backward(grad_ln1_out, 0.0);    // (T, d)

    // --- Aggregate grad_input ---
    // grad_input = grad_output (from x-residual) + grad_residual1_from_ffn (from FFN's x-residual) + grad_x_from_mixer (from mixer)
    Tensor grad_input(T, d_model_);
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        grad_input.data[i] = grad_output.data[i]              // x-residual
                           + grad_residual1_from_ffn.data[i]  // FFN sublayer's x-residual contribution
                           + grad_x_from_mixer.data[i];       // mixer's x via LN_1
    }
    return grad_input;
}

void XLSTMBlock::update_weights(double learning_rate) {
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    slstm_.update_weights(learning_rate);
    slstm_proj_.update_weights(learning_rate);
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        mlstm_.update_weights(learning_rate);
        mlstm_proj_.update_weights(learning_rate);
    }
    ffn_proj1_.update_weights(learning_rate);
    ffn_proj2_.update_weights(learning_rate);
}

void XLSTMBlock::zero_grad() {
    ln1_.zero_grad();
    ln2_.zero_grad();
    slstm_.zero_grad();
    slstm_proj_.zero_grad();
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        mlstm_.zero_grad();
        mlstm_proj_.zero_grad();
    }
    ffn_proj1_.zero_grad();
    ffn_proj2_.zero_grad();
}

std::vector<Tensor*> XLSTMBlock::parameters() {
    std::vector<Tensor*> p;
    auto ln1p = ln1_.parameters();
    p.insert(p.end(), ln1p.begin(), ln1p.end());
    auto ln2p = ln2_.parameters();
    p.insert(p.end(), ln2p.begin(), ln2p.end());
    auto slstmp = slstm_.parameters();
    p.insert(p.end(), slstmp.begin(), slstmp.end());
    auto slstm_projp = slstm_proj_.parameters();
    p.insert(p.end(), slstm_projp.begin(), slstm_projp.end());
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        auto mlstmp = mlstm_.parameters();
        p.insert(p.end(), mlstmp.begin(), mlstmp.end());
        auto mlstm_projp = mlstm_proj_.parameters();
        p.insert(p.end(), mlstm_projp.begin(), mlstm_projp.end());
    }
    auto ffn1p = ffn_proj1_.parameters();
    p.insert(p.end(), ffn1p.begin(), ffn1p.end());
    auto ffn2p = ffn_proj2_.parameters();
    p.insert(p.end(), ffn2p.begin(), ffn2p.end());
    return p;
}

std::vector<Tensor*> XLSTMBlock::gradients() {
    std::vector<Tensor*> g;
    auto ln1g = ln1_.gradients();
    g.insert(g.end(), ln1g.begin(), ln1g.end());
    auto ln2g = ln2_.gradients();
    g.insert(g.end(), ln2g.begin(), ln2g.end());
    auto slstmg = slstm_.gradients();
    g.insert(g.end(), slstmg.begin(), slstmg.end());
    auto slstm_projg = slstm_proj_.gradients();
    g.insert(g.end(), slstm_projg.begin(), slstm_projg.end());
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        auto mlstmg = mlstm_.gradients();
        g.insert(g.end(), mlstmg.begin(), mlstmg.end());
        auto mlstm_projg = mlstm_proj_.gradients();
        g.insert(g.end(), mlstm_projg.begin(), mlstm_projg.end());
    }
    auto ffn1g = ffn_proj1_.gradients();
    g.insert(g.end(), ffn1g.begin(), ffn1g.end());
    auto ffn2g = ffn_proj2_.gradients();
    g.insert(g.end(), ffn2g.begin(), ffn2g.end());
    return g;
}

void XLSTMBlock::copy_params_from(const XLSTMBlock& other) {
    if (d_model_ != other.d_model_ ||
        slstm_hidden_ != other.slstm_hidden_ ||
        mlstm_hidden_ != other.mlstm_hidden_ ||
        ffn_mult_ != other.ffn_mult_ ||
        cell_type_ != other.cell_type_) {
        throw std::invalid_argument("XLSTMBlock::copy_params_from: shape mismatch");
    }
    ln1_.gamma = other.ln1_.gamma;
    ln1_.beta = other.ln1_.beta;
    ln2_.gamma = other.ln2_.gamma;
    ln2_.beta = other.ln2_.beta;
    slstm_.W = other.slstm_.W;
    slstm_.b = other.slstm_.b;
    slstm_proj_.weights = other.slstm_proj_.weights;
    slstm_proj_.bias = other.slstm_proj_.bias;
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        mlstm_.W = other.mlstm_.W;
        mlstm_.b = other.mlstm_.b;
        mlstm_proj_.weights = other.mlstm_proj_.weights;
        mlstm_proj_.bias = other.mlstm_proj_.bias;
    }
    ffn_proj1_.weights = other.ffn_proj1_.weights;
    ffn_proj1_.bias = other.ffn_proj1_.bias;
    ffn_proj2_.weights = other.ffn_proj2_.weights;
    ffn_proj2_.bias = other.ffn_proj2_.bias;
}

size_t XLSTMBlock::count_parameters() const {
    size_t n = 0;
    // Iterate member tensors directly (this is const, so we can't call non-const parameters()).
    n += ln1_.gamma.data.size() + ln1_.beta.data.size();
    n += ln2_.gamma.data.size() + ln2_.beta.data.size();
    n += slstm_.W.data.size() + slstm_.b.data.size();
    n += slstm_proj_.weights.data.size() + slstm_proj_.bias.data.size();
    if (cell_type_ != XLSTMCellType::SLSTM_ONLY) {
        n += mlstm_.W.data.size() + mlstm_.b.data.size();
        n += mlstm_proj_.weights.data.size() + mlstm_proj_.bias.data.size();
    }
    n += ffn_proj1_.weights.data.size() + ffn_proj1_.bias.data.size();
    n += ffn_proj2_.weights.data.size() + ffn_proj2_.bias.data.size();
    return n;
}

// ============================================================================
// XLSTMModel
// ============================================================================

XLSTMModel::XLSTMModel(size_t input_dim, size_t d_model, size_t output_dim,
                       size_t num_layers, size_t slstm_hidden,
                       size_t mlstm_hidden, size_t ffn_mult,
                       XLSTMCellType cell_type)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      embed_(input_dim, d_model),
      final_ln_(d_model),
      classifier_(d_model, output_dim) {
    if (input_dim == 0) throw std::invalid_argument("XLSTMModel: input_dim must be > 0");
    if (d_model == 0) throw std::invalid_argument("XLSTMModel: d_model must be > 0");
    if (output_dim == 0) throw std::invalid_argument("XLSTMModel: output_dim must be > 0");
    if (num_layers == 0) throw std::invalid_argument("XLSTMModel: num_layers must be > 0");
    if (slstm_hidden == 0) throw std::invalid_argument("XLSTMModel: slstm_hidden must be > 0");
    if (ffn_mult == 0) throw std::invalid_argument("XLSTMModel: ffn_mult must be > 0");

    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(std::make_unique<XLSTMBlock>(
            d_model, slstm_hidden, mlstm_hidden, ffn_mult, cell_type));
    }
}

Tensor XLSTMModel::forward(const Tensor& input) {
    if (input.cols != input_dim_) {
        throw std::invalid_argument("XLSTMModel::forward: input feature dim mismatch");
    }
    Tensor x = embed_.forward(input);
    for (auto& blk : blocks_) {
        x = blk->forward(x);
    }
    x = final_ln_.forward(x);
    return classifier_.forward(x);
}

Tensor XLSTMModel::backward(const Tensor& grad_output, double lr) {
    Tensor g = classifier_.backward(grad_output, lr);
    g = final_ln_.backward(g, lr);
    // Iterate blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        g = (*it)->backward(g, lr);
    }
    return embed_.backward(g, lr);
}

void XLSTMModel::update_weights(double learning_rate) {
    embed_.update_weights(learning_rate);
    for (auto& blk : blocks_) blk->update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void XLSTMModel::zero_grad() {
    embed_.zero_grad();
    for (auto& blk : blocks_) blk->zero_grad();
    final_ln_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> XLSTMModel::parameters() {
    std::vector<Tensor*> p;
    auto ep = embed_.parameters();
    p.insert(p.end(), ep.begin(), ep.end());
    for (auto& blk : blocks_) {
        auto bp = blk->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fp = final_ln_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> XLSTMModel::gradients() {
    std::vector<Tensor*> g;
    auto eg = embed_.gradients();
    g.insert(g.end(), eg.begin(), eg.end());
    for (auto& blk : blocks_) {
        auto bg = blk->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fg = final_ln_.gradients();
    g.insert(g.end(), fg.begin(), fg.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}