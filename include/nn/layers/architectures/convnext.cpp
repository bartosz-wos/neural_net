#include "convnext.h"
#include "../../activations/activations.h"
#include <cmath>
#include <stdexcept>

// ===========================================================================
// ChannelLayerNorm2D
// ===========================================================================

ChannelLayerNorm2D::ChannelLayerNorm2D(size_t channels, size_t H, size_t W, double eps)
    : gamma(1, channels), beta(1, channels),
      grad_gamma(1, channels), grad_beta(1, channels),
      eps(eps), last_xhat(0, 0), last_inv_std(0, 0),
      channels_(channels), H_(H), W_(W)
{
    if (channels == 0) {
        throw std::invalid_argument("ChannelLayerNorm2D: channels must be > 0");
    }
    if (H == 0 || W == 0) {
        throw std::invalid_argument("ChannelLayerNorm2D: H and W must be > 0");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("ChannelLayerNorm2D: eps must be >= 0");
    }
    gamma.fill(1.0);
    beta.fill(0.0);
    grad_gamma.fill(0.0);
    grad_beta.fill(0.0);
}

Tensor ChannelLayerNorm2D::forward(const Tensor& input) {
    const size_t N = input.rows;
    const size_t S = H_ * W_;
    if (input.cols != channels_ * S) {
        throw std::invalid_argument(
            "ChannelLayerNorm2D::forward: expected cols = C*H*W = " +
            std::to_string(channels_ * S) + ", got " + std::to_string(input.cols));
    }

    last_xhat    = Tensor(N, channels_ * S);
    last_inv_std = Tensor(N, S);
    Tensor out(N, channels_ * S);

    for (size_t n = 0; n < N; ++n) {
        for (size_t s = 0; s < S; ++s) {
            double mean = 0.0;
            for (size_t c = 0; c < channels_; ++c) {
                mean += input[n][c * S + s];
            }
            mean /= static_cast<double>(channels_);

            double var = 0.0;
            for (size_t c = 0; c < channels_; ++c) {
                const double d = input[n][c * S + s] - mean;
                var += d * d;
            }
            var /= static_cast<double>(channels_);

            const double inv = 1.0 / std::sqrt(var + eps);
            last_inv_std[n][s] = inv;
            for (size_t c = 0; c < channels_; ++c) {
                const double xhat = (input[n][c * S + s] - mean) * inv;
                last_xhat[n][c * S + s] = xhat;
                out[n][c * S + s] = gamma[0][c] * xhat + beta[0][c];
            }
        }
    }
    return out;
}

Tensor ChannelLayerNorm2D::backward(const Tensor& grad_output, double /*lr*/) {
    const size_t N = grad_output.rows;
    const size_t S = H_ * W_;
    if (grad_output.cols != channels_ * S) {
        throw std::invalid_argument(
            "ChannelLayerNorm2D::backward: expected cols = C*H*W");
    }
    if (last_xhat.rows != N) {
        throw std::runtime_error(
            "ChannelLayerNorm2D::backward: forward() must be called first");
    }

    // Accumulate (not overwrite) so multiple backward() calls before one
    // update_weights() sum correctly.
    for (size_t c = 0; c < channels_; ++c) {
        double gg = 0.0, gb = 0.0;
        for (size_t n = 0; n < N; ++n) {
            for (size_t s = 0; s < S; ++s) {
                const double go = grad_output[n][c * S + s];
                gg += go * last_xhat[n][c * S + s];
                gb += go;
            }
        }
        grad_gamma[0][c] += gg;
        grad_beta[0][c]  += gb;
    }

    // dxhat = grad_out * gamma, then the channel-axis LayerNorm Jacobian:
    //   dx = rstd * (dxhat - mean_c(dxhat) - xhat * mean_c(dxhat * xhat))
    // Both means reduce over the CHANNEL axis only (not the row) — the same
    // distinction that makes this class different from the row-wise LayerNorm.
    Tensor dx(N, channels_ * S);
    for (size_t n = 0; n < N; ++n) {
        for (size_t s = 0; s < S; ++s) {
            const double inv = last_inv_std[n][s];
            double d_mean = 0.0, d_cross = 0.0;
            for (size_t c = 0; c < channels_; ++c) {
                const double dh = grad_output[n][c * S + s] * gamma[0][c];
                d_mean  += dh;
                d_cross += dh * last_xhat[n][c * S + s];
            }
            d_mean  /= static_cast<double>(channels_);
            d_cross /= static_cast<double>(channels_);
            for (size_t c = 0; c < channels_; ++c) {
                const double dh = grad_output[n][c * S + s] * gamma[0][c];
                dx[n][c * S + s] = inv * (dh - d_mean - last_xhat[n][c * S + s] * d_cross);
            }
        }
    }
    return dx;
}

void ChannelLayerNorm2D::update_weights(double learning_rate) {
    for (size_t c = 0; c < channels_; ++c) {
        gamma[0][c] -= learning_rate * grad_gamma[0][c];
        beta[0][c]  -= learning_rate * grad_beta[0][c];
    }
}

std::vector<Tensor*> ChannelLayerNorm2D::parameters() { return {&gamma, &beta}; }
std::vector<Tensor*> ChannelLayerNorm2D::gradients()  { return {&grad_gamma, &grad_beta}; }

void ChannelLayerNorm2D::zero_grad() {
    grad_gamma.fill(0.0);
    grad_beta.fill(0.0);
}

// ===========================================================================
// ConvNeXtBlock
// ===========================================================================

ConvNeXtBlock::ConvNeXtBlock(size_t dim, size_t H, size_t W,
                             double layer_scale_init, size_t kernel)
    : dim_(dim), H_(H), W_(W), kernel_(kernel), layer_scale_init_(layer_scale_init),
      dw_(), ln_(dim, H, W, 1e-5),
      pwconv1_(dim, 4 * dim), pwconv2_(4 * dim, dim),
      gamma_(dim, layer_scale_init),
      last_input_(0, 0), last_pre_gelu_(0, 0)
{
    if (dim == 0) {
        throw std::invalid_argument("ConvNeXtBlock: dim must be > 0");
    }
    if (H == 0 || W == 0) {
        throw std::invalid_argument("ConvNeXtBlock: H and W must be > 0");
    }
    if (kernel == 0 || kernel % 2 == 0) {
        throw std::invalid_argument("ConvNeXtBlock: kernel must be a positive odd number");
    }
    if (layer_scale_init < 0.0) {
        throw std::invalid_argument("ConvNeXtBlock: layer_scale_init must be >= 0");
    }

    const int pad = static_cast<int>(kernel / 2);
    dw_.reserve(dim);
    for (size_t c = 0; c < dim; ++c) {
        Conv2D d(1, 1, static_cast<int>(kernel), static_cast<int>(kernel),
                 static_cast<int>(H), static_cast<int>(W),
                 1, 1, pad, pad, 1, 1);
        // §2.1: the depthwise conv has NO bias.
        d.bias.fill(0.0);
        dw_.push_back(d);
    }
}

Tensor ConvNeXtBlock::depthwise_forward(const Tensor& input) {
    const size_t N = input.rows;
    const size_t S = H_ * W_;
    Tensor out(N, dim_ * S);
    Tensor ch_in(N, S), ch_out(N, S);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t n = 0; n < N; ++n)
            for (size_t s = 0; s < S; ++s)
                ch_in[n][s] = input[n][c * S + s];
        ch_out = dw_[c].forward(ch_in);
        for (size_t n = 0; n < N; ++n)
            for (size_t s = 0; s < S; ++s)
                out[n][c * S + s] = ch_out[n][s];
    }
    return out;
}

Tensor ConvNeXtBlock::depthwise_backward(const Tensor& grad, double lr) {
    const size_t N = grad.rows;
    const size_t S = H_ * W_;
    Tensor out(N, dim_ * S);
    Tensor g_ch(N, S), gx_ch(N, S);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t n = 0; n < N; ++n)
            for (size_t s = 0; s < S; ++s)
                g_ch[n][s] = grad[n][c * S + s];
        gx_ch = dw_[c].backward(g_ch, lr);
        for (size_t n = 0; n < N; ++n)
            for (size_t s = 0; s < S; ++s)
                out[n][c * S + s] = gx_ch[n][s];
    }
    return out;
}

// Reshape (N, C*S) <-> (N*S, C): a 1x1 convolution is a Dense over the
// CHANNEL axis at each spatial position, NOT over the flattened row.
// `N` is passed explicitly rather than inferred from the row count, because in
// the backward pass the gradient is already in positions-space (N*S rows) and
// N cannot be recovered from it without a second argument.
static Tensor to_positions_n(const Tensor& x, size_t N, size_t channels, size_t spatial) {
    Tensor out(N * spatial, channels);
    for (size_t n = 0; n < N; ++n)
        for (size_t s = 0; s < spatial; ++s)
            for (size_t c = 0; c < channels; ++c)
                out[n * spatial + s][c] = x[n][c * spatial + s];
    return out;
}

static Tensor from_positions_n(const Tensor& p, size_t N, size_t channels, size_t spatial) {
    Tensor out(N, channels * spatial);
    for (size_t n = 0; n < N; ++n)
        for (size_t s = 0; s < spatial; ++s)
            for (size_t c = 0; c < channels; ++c)
                out[n][c * spatial + s] = p[n * spatial + s][c];
    return out;
}

// Forward-facing conveniences: N is the batch size of the (N, C*S) tensor.
static Tensor to_positions(const Tensor& x, size_t channels, size_t spatial) {
    return to_positions_n(x, x.rows, channels, spatial);
}

static Tensor from_positions(const Tensor& p, size_t channels, size_t spatial) {
    return from_positions_n(p, p.rows / spatial, channels, spatial);
}

Tensor ConvNeXtBlock::forward(const Tensor& input) {
    const size_t S = H_ * W_;
    if (input.cols != dim_ * S) {
        throw std::invalid_argument(
            "ConvNeXtBlock::forward: expected cols = C*H*W = " +
            std::to_string(dim_ * S) + ", got " + std::to_string(input.cols));
    }
    last_input_ = input;

    // (1) dwconv 7x7 over the C channels, no bias, stride 1
    Tensor u = depthwise_forward(input);
    // (2) ChannelLayerNorm2D — INSIDE the residual branch (post-norm)
    u = ln_.forward(u);
    // (3) pwconv1: C -> 4C. A 1x1 conv acts on the CHANNEL AXIS AT EACH
    // SPATIAL POSITION, so the (N, C*S) tensor must be reshaped to (N*S, C)
    // first. Applying Dense to the flattened row instead would contract across
    // all C*S features and mix spatial positions — the classic "1x1 conv is not
    // a Dense" bug.
    Tensor pos = to_positions(u, dim_, S);
    Tensor h = pwconv1_.forward(pos);
    // (4) GELU
    last_pre_gelu_ = h;
    {
        GELU g;
        h = g(h);
    }
    // (5) pwconv2: 4C -> C
    h = pwconv2_.forward(h);
    // (6) per-channel LayerScale gamma — applied in positions-space (N*S, C),
    // because LayerScale gates the CHANNEL axis (one lambda per channel), not
    // the C*S flattened features.
    h = gamma_.forward(h);
    u = from_positions(h, dim_, S);
    // (7) residual add with the BLOCK INPUT (not the LN output)
    Tensor out = input;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            out[i][j] += u[i][j];
    return out;
}

Tensor ConvNeXtBlock::backward(const Tensor& grad_output, double /*lr*/) {
    const size_t S = H_ * W_;
    const size_t N = grad_output.rows;
    Tensor g = grad_output;                       // (7) residual add: copy
    g = to_positions_n(g, N, dim_, S);            // back through the reshape
    g = gamma_.backward(g, 0.0);                  // (6)
    g = pwconv2_.backward(g, 0.0);                // (5)
    {                                              // (4) GELU
        GELU gelu;
        Tensor dg(g.rows, g.cols);
        for (size_t i = 0; i < g.rows; ++i)
            for (size_t j = 0; j < g.cols; ++j)
                dg[i][j] = g[i][j] * gelu.derivative(last_pre_gelu_[i][j]);
        g = dg;
    }
    g = pwconv1_.backward(g, 0.0);                // (3)
    g = from_positions_n(g, N, dim_, S);          // (3) back through the reshape
    g = ln_.backward(g, 0.0);                     // (2)
    Tensor g_dw = depthwise_backward(g, 0.0);     // (1)
    // The dwconv also receives gradient through the residual path.
    for (size_t i = 0; i < g_dw.rows; ++i)
        for (size_t j = 0; j < g_dw.cols; ++j)
            g_dw[i][j] += grad_output[i][j];
    return g_dw;
}

void ConvNeXtBlock::update_weights(double learning_rate) {
    for (auto& d : dw_)     d.update_weights(learning_rate);
    ln_.update_weights(learning_rate);
    pwconv1_.update_weights(learning_rate);
    pwconv2_.update_weights(learning_rate);
    gamma_.update_weights(learning_rate);
}

std::vector<Tensor*> ConvNeXtBlock::parameters() {
    std::vector<Tensor*> out;
    for (auto& d : dw_) {
        auto p = d.parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    auto a = ln_.parameters();
    out.insert(out.end(), a.begin(), a.end());
    auto c = pwconv1_.parameters();
    out.insert(out.end(), c.begin(), c.end());
    auto e = pwconv2_.parameters();
    out.insert(out.end(), e.begin(), e.end());
    auto f = gamma_.parameters();
    out.insert(out.end(), f.begin(), f.end());
    return out;
}

std::vector<Tensor*> ConvNeXtBlock::gradients() {
    std::vector<Tensor*> out;
    for (auto& d : dw_) {
        auto g = d.gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    auto a = ln_.gradients();
    out.insert(out.end(), a.begin(), a.end());
    auto c = pwconv1_.gradients();
    out.insert(out.end(), c.begin(), c.end());
    auto e = pwconv2_.gradients();
    out.insert(out.end(), e.begin(), e.end());
    auto f = gamma_.gradients();
    out.insert(out.end(), f.begin(), f.end());
    return out;
}

void ConvNeXtBlock::zero_grad() {
    for (auto& d : dw_) d.zero_grad();
    ln_.zero_grad();
    pwconv1_.zero_grad();
    pwconv2_.zero_grad();
    gamma_.zero_grad();
}

// ===========================================================================
// ConvNeXt
// ===========================================================================

ConvNeXt::ConvNeXt(size_t in_channels, size_t H, size_t W, size_t num_classes,
                   const std::vector<size_t>& depths,
                   const std::vector<size_t>& dims,
                   double layer_scale_init)
    : in_channels_(in_channels), H_in_(H), W_in_(W), num_classes_(num_classes),
      depths_(depths), dims_(dims),
      stem_(), down_ln_(), down_conv_(), blocks_(),
      final_ln_(dims.empty() ? 1 : dims.back(),
                std::max<size_t>(1, H >> (2 * dims.size())),
                std::max<size_t>(1, W >> (2 * dims.size())), 1e-5),
      head_(dims.empty() ? 1 : dims.back(), num_classes),
      last_input_(0, 0)
{
    if (in_channels == 0) {
        throw std::invalid_argument("ConvNeXt: in_channels must be > 0");
    }
    if (H == 0 || W == 0) {
        throw std::invalid_argument("ConvNeXt: H and W must be > 0");
    }
    if (num_classes == 0) {
        throw std::invalid_argument("ConvNeXt: num_classes must be > 0");
    }
    if (layer_scale_init < 0.0) {
        throw std::invalid_argument("ConvNeXt: layer_scale_init must be >= 0");
    }
    if (depths.empty() || dims.empty()) {
        throw std::invalid_argument("ConvNeXt: depths and dims must be non-empty");
    }
    if (depths.size() != dims.size()) {
        throw std::invalid_argument("ConvNeXt: depths and dims must have equal length");
    }
    for (size_t s = 0; s < depths.size(); ++s) {
        if (depths[s] == 0) {
            throw std::invalid_argument("ConvNeXt: every stage depth must be > 0");
        }
        if (dims[s] == 0) {
            throw std::invalid_argument("ConvNeXt: every stage dim must be > 0");
        }
    }

    // §3 stem: 4x4 conv, stride 4, no padding. Requires H,W >= 4.
    if (H < 4 || W < 4) {
        throw std::invalid_argument(
            "ConvNeXt: the 4x4 stride-4 stem needs H,W >= 4");
    }
    stem_ = Conv2D(static_cast<int>(in_channels), static_cast<int>(dims[0]),
                   4, 4, static_cast<int>(H), static_cast<int>(W),
                   4, 4, 0, 0, 1, 1);

    size_t h = H / 4, w = W / 4;
    // Precondition: each stage boundary downsamples with a 2x2 stride-2 conv, so
    // the spatial extent entering a downsample must be >= 2 (a 2x2 kernel on a
    // 1x1 input has no valid output). Check this on LOCAL COPIES — decrementing
    // h/w here would double-count, because the stage-building loop below halves
    // them again.
    {
        size_t ch = h, cw = w;
        for (size_t k = 0; k + 1 < depths.size(); ++k) {
            if (ch < 2 || cw < 2) {
                throw std::invalid_argument(
                    "ConvNeXt: " + std::to_string(depths.size()) +
                    " stages need a larger image; the stride-4 stem leaves " +
                    std::to_string(h) + "x" + std::to_string(w) +
                    ", which cannot support " + std::to_string(depths.size() - 1) +
                    " 2x2 stride-2 downsamples");
            }
            ch /= 2;
            cw /= 2;
        }
    }

    size_t in_dim = dims[0];
    blocks_.resize(depths.size());
    for (size_t s = 0; s < depths.size(); ++s) {
        blocks_[s].reserve(depths[s]);
        for (size_t b = 0; b < depths[s]; ++b) {
            blocks_[s].emplace_back(dims[s], h, w, layer_scale_init);
        }
        if (s + 1 < depths.size()) {
            // §3 downsample: LayerNorm -> 2x2 conv stride 2
            down_ln_.emplace_back(dims[s], h, w, 1e-5);
            down_conv_.push_back(Conv2D(static_cast<int>(dims[s]),
                                        static_cast<int>(dims[s + 1]),
                                        2, 2, static_cast<int>(h), static_cast<int>(w),
                                        2, 2, 0, 0, 1, 1));
            h = h / 2;
            w = w / 2;
            in_dim = dims[s + 1];
        }
    }
    (void)in_dim;

    // The final LN and the head see the LAST stage's spatial extent.
    final_ln_ = ChannelLayerNorm2D(dims.back(), h, w, 1e-5);
    head_ = Dense(dims.back(), num_classes);
}

size_t ConvNeXt::stage_spatial(size_t s) const {
    // Stage 0 begins after the stride-4 stem; each boundary halves it.
    size_t h = H_in_ / 4;
    for (size_t k = 0; k < s && k + 1 < depths_.size(); ++k) h = h / 2;
    return h;
}

void ConvNeXt::build_downsamplers() {
    // Built in the constructor; kept as a named seam for future v2 changes.
}

Tensor ConvNeXt::forward(const Tensor& input) {
    if (input.cols != in_channels_ * H_in_ * W_in_) {
        throw std::invalid_argument("ConvNeXt::forward: input cols must be in_channels*H*W");
    }
    last_input_ = input;

    Tensor x = stem_.forward(input);
    for (size_t s = 0; s < blocks_.size(); ++s) {
        for (auto& b : blocks_[s]) x = b.forward(x);
        if (s + 1 < blocks_.size()) {
            x = down_ln_[s].forward(x);
            x = down_conv_[s].forward(x);
        }
    }
    x = final_ln_.forward(x);

    // Global average pool over the last stage's H*W, per (n, c).
    const size_t N = x.rows;
    const size_t S = x.cols / dims_.back();
    Tensor pooled(N, dims_.back());
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < dims_.back(); ++c) {
            double acc = 0.0;
            for (size_t s = 0; s < S; ++s) acc += x[n][c * S + s];
            pooled[n][c] = acc / static_cast<double>(S);
        }
    }
    return head_.forward(pooled);
}

Tensor ConvNeXt::backward(const Tensor& grad_output, double /*lr*/) {
    // head -> un-pool -> final LN -> stages in reverse -> stem
    Tensor g = head_.backward(grad_output, 0.0);

    const size_t N = g.rows;
    const size_t S = last_input_.cols / in_channels_;   // NOT used directly; recomputed below
    (void)S;

    // Un-pool: the head sees one value per (n, c), and the pool is a mean over
    // S positions, so dL/dx[n][c*S+s] = dL/dpooled[n][c] / S for every s.
    // S is the LAST stage's spatial extent: derive it from final_ln_.
    const size_t last_S = final_ln_.spatial();
    Tensor gp(N, dims_.back() * last_S);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < dims_.back(); ++c)
            for (size_t s = 0; s < last_S; ++s)
                gp[n][c * last_S + s] = g[n][c] / static_cast<double>(last_S);

    g = final_ln_.backward(gp, 0.0);

    for (size_t s = blocks_.size(); s-- > 0;) {
        for (size_t b = blocks_[s].size(); b-- > 0;) {
            g = blocks_[s][b].backward(g, 0.0);
        }
        if (s > 0) {
            g = down_conv_[s - 1].backward(g, 0.0);
            g = down_ln_[s - 1].backward(g, 0.0);
        }
    }
    return stem_.backward(g, 0.0);
}

void ConvNeXt::update_weights(double learning_rate) {
    stem_.update_weights(learning_rate);
    for (auto& d : down_conv_) d.update_weights(learning_rate);
    for (auto& l : down_ln_)  l.update_weights(learning_rate);
    for (auto& stage : blocks_)
        for (auto& b : stage) b.update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    head_.update_weights(learning_rate);
}

std::vector<Tensor*> ConvNeXt::parameters() {
    std::vector<Tensor*> out;
    auto add = [&out](const std::vector<Tensor*>& v) { out.insert(out.end(), v.begin(), v.end()); };
    add(stem_.parameters());
    for (auto& l : down_ln_)  add(l.parameters());
    for (auto& d : down_conv_) add(d.parameters());
    for (auto& stage : blocks_)
        for (auto& b : stage) add(b.parameters());
    add(final_ln_.parameters());
    add(head_.parameters());
    return out;
}

std::vector<Tensor*> ConvNeXt::gradients() {
    std::vector<Tensor*> out;
    auto add = [&out](const std::vector<Tensor*>& v) { out.insert(out.end(), v.begin(), v.end()); };
    add(stem_.gradients());
    for (auto& l : down_ln_)  add(l.gradients());
    for (auto& d : down_conv_) add(d.gradients());
    for (auto& stage : blocks_)
        for (auto& b : stage) add(b.gradients());
    add(final_ln_.gradients());
    add(head_.gradients());
    return out;
}

void ConvNeXt::zero_grad() {
    stem_.zero_grad();
    for (auto& l : down_ln_)   l.zero_grad();
    for (auto& d : down_conv_) d.zero_grad();
    for (auto& stage : blocks_)
        for (auto& b : stage) b.zero_grad();
    final_ln_.zero_grad();
    head_.zero_grad();
}
