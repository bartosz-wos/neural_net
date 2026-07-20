#include "ddpm.h"
#include "../../activations/activations.h"
#include <stdexcept>
#include <iostream>

// =============================================================================
// NoiseScheduler
// =============================================================================
NoiseScheduler::NoiseScheduler()
    : T_(1000), beta_start_(1e-4f), beta_end_(0.02f) {}

NoiseScheduler::NoiseScheduler(int T, float beta_start, float beta_end)
    : T_(T), beta_start_(beta_start), beta_end_(beta_end) {
    compute_schedule();
}

void NoiseScheduler::initialize(float beta_start, float beta_end, int T) {
    beta_start_ = beta_start;
    beta_end_   = beta_end;
    T_          = T;
    compute_schedule();
}

void NoiseScheduler::compute_schedule() {
    betas.resize(T_);
    alphas.resize(T_);
    alphas_cumprod.resize(T_);
    alphas_cumprod_prev.assign(T_ + 1, 1.0f);

    for (int t = 0; t < T_; ++t) {
        float t_norm = static_cast<float>(t) / static_cast<float>(T_ - 1);
        betas[t] = beta_start_ + t_norm * (beta_end_ - beta_start_);
    }

    for (int t = 0; t < T_; ++t) {
        alphas[t] = 1.0f - betas[t];
    }

    alphas_cumprod[0] = alphas[0];
    alphas_cumprod_prev[0] = 1.0f;
    for (int t = 1; t < T_; ++t) {
        alphas_cumprod[t] = alphas_cumprod[t - 1] * alphas[t];
        alphas_cumprod_prev[t] = alphas_cumprod[t - 1];
    }
    alphas_cumprod_prev[T_] = alphas_cumprod[T_ - 1];
}

float NoiseScheduler::sqrt_alphas_cumprod(int t) const {
    if (t < 0)   t = 0;
    if (t >= T_) t = T_ - 1;
    return std::sqrt(std::max(alphas_cumprod[t], 1e-8f));
}

float NoiseScheduler::sqrt_one_minus_alphas_cumprod(int t) const {
    if (t < 0)   t = 0;
    if (t >= T_) t = T_ - 1;
    return std::sqrt(std::max(1.0f - alphas_cumprod[t], 1e-8f));
}

float NoiseScheduler::sqrt_recip_alphas(int t) const {
    if (t < 0)   t = 0;
    if (t >= T_) t = T_ - 1;
    return 1.0f / std::sqrt(std::max(alphas[t], 1e-8f));
}

float NoiseScheduler::extract(std::vector<float>& vec, int t) const {
    if (t < 0)   t = 0;
    if (t >= T_) t = T_ - 1;
    return vec[t];
}

float NoiseScheduler::posterior_variance(int t) const {
    if (t <= 0) return 0.0f;
    float alpha_bar_t   = alphas_cumprod[t];
    float alpha_bar_tm1 = alphas_cumprod[t - 1];
    float beta_t        = betas[t];
    return (beta_t * (1.0f - alpha_bar_tm1)) / std::max(1.0f - alpha_bar_t, 1e-8f);
}

Tensor NoiseScheduler::q_sample(const Tensor& x0, int t, const Tensor& noise) const {
    float sqrt_alpha_bar = sqrt_alphas_cumprod(t);
    float sqrt_one_minus = sqrt_one_minus_alphas_cumprod(t);

    size_t batch = x0.rows;
    size_t dim   = x0.cols;
    Tensor x_t(batch, dim);

    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            x_t[i][j] = sqrt_alpha_bar * x0[i][j] + sqrt_one_minus * noise[i][j];
        }
    }
    return x_t;
}

// =============================================================================
// TimeEmbedding
// =============================================================================
TimeEmbedding::TimeEmbedding(int hidden_dim) : hidden_dim_(hidden_dim) {}

Tensor TimeEmbedding::forward(int t) const {
    size_t half_dim = hidden_dim_ / 2;
    Tensor emb(1, hidden_dim_);
    double log_step = std::log(10000.0) / static_cast<double>(half_dim - 1);
    for (size_t i = 0; i < half_dim; ++i) {
        double freq = std::exp(static_cast<double>(i) * log_step);
        double arg  = freq * static_cast<double>(t);
        emb[0][2 * i]     = std::sin(arg);
        emb[0][2 * i + 1] = std::cos(arg);
    }
    return emb;
}

// =============================================================================
// SkipTransform
// =============================================================================
SkipTransform::SkipTransform(int in_channels, int out_channels, int seq_len)
    : conv_(in_channels, out_channels, 1, seq_len, 1, 0) {}

SkipTransform::~SkipTransform() {}

Tensor SkipTransform::forward(const Tensor& x) {
    last_x_ = x;
    return conv_.forward(x);
}

Tensor SkipTransform::backward(const Tensor& grad_output, double learning_rate) {
    return conv_.backward(grad_output, learning_rate);
}

void SkipTransform::update_weights(double learning_rate) {
    conv_.update_weights(learning_rate);
}

std::vector<Tensor*> SkipTransform::parameters() {
    return conv_.parameters();
}

std::vector<Tensor*> SkipTransform::gradients() {
    return conv_.gradients();
}

void SkipTransform::zero_grad() {
    conv_.zero_grad();
}

// =============================================================================
// DDPMResBlock
// =============================================================================
// static
bool DDPMResBlock::initialized_ = false;


DDPMResBlock::DDPMResBlock(int in_channels, int out_channels, int time_emb_dim, int seq_len)
    : in_channels_(in_channels), out_channels_(out_channels), seq_len_(seq_len),
      conv1_(in_channels, out_channels, 3, seq_len, 1, 1),
      conv2_(out_channels, out_channels, 3, seq_len, 1, 1),
      time_mlp_(time_emb_dim, out_channels) {}

DDPMResBlock::~DDPMResBlock() {
    // conv1_, conv2_, time_mlp_ are value members (not pointers) — auto-destroyed
}

// Required by Layer interface — single-arg version (time embedding unavailable)
Tensor DDPMResBlock::forward(const Tensor& x) {
    return x;
}

// Two-arg forward with time embedding
Tensor DDPMResBlock::forward(const Tensor& x, const Tensor& time_emb) {
    last_x_ = x;
    last_time_emb_ = time_emb;

    // Conv1 -> activation
    last_h_ = conv1_.forward(x);
    last_h_ = act_(last_h_);

    // Project time embedding: (1, time_emb_dim) -> (1, out_channels)
    Tensor t_proj = time_mlp_.forward(time_emb); // (1, out_channels)

    // Broadcast-add time projection to each spatial position
    // last_h_ shape: (batch, out_channels * seq_len)
    size_t batch   = last_h_.rows;
    size_t spatial = static_cast<size_t>(seq_len_);
    size_t ch      = static_cast<size_t>(out_channels_);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < spatial; ++s) {
            for (size_t c = 0; c < ch; ++c) {
                last_h_[b][c * spatial + s] += t_proj[0][c];
            }
        }
    }

    // Conv2
    Tensor h2 = conv2_.forward(last_h_);

    // Zero-initialize conv2 weights once at start of training (starts as identity)
    // Guard with static initialized_ so this only happens on the very first forward pass
    if (!initialized_) {
        for (auto& w : conv2_.weights.data)    w = 0.0f;
        for (auto& b : conv2_.bias.data)        b = 0.0f;
        for (auto& g : conv2_.grad_weights.data) g = 0.0f;
        for (auto& g : conv2_.grad_bias.data)   g = 0.0f;
        initialized_ = true;
    }

    // Residual connection with channel padding if needed
    if (in_channels_ != out_channels_) {
        Tensor x_padded(batch, out_channels_ * seq_len_);
        x_padded.fill(0.0);
        size_t spatial2 = static_cast<size_t>(seq_len_);
        size_t in_ch_sz = static_cast<size_t>(in_channels_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < spatial2; ++s) {
                for (size_t c = 0; c < ch; ++c) {
                    if (c < in_ch_sz) {
                        x_padded[b][c * spatial2 + s] = x[b][c * spatial2 + s];
                    }
                }
            }
        }
        return h2 + x_padded;
    }
    return h2 + x;
}

Tensor DDPMResBlock::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output;
    (void)learning_rate;
    return grad_output;
}

void DDPMResBlock::update_weights(double learning_rate) {
    conv1_.update_weights(learning_rate);
    conv2_.update_weights(learning_rate);
    time_mlp_.update_weights(learning_rate);
}

std::vector<Tensor*> DDPMResBlock::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : conv1_.parameters()) result.push_back(p);
    for (Tensor* p : conv2_.parameters()) result.push_back(p);
    for (Tensor* p : time_mlp_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> DDPMResBlock::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : conv1_.gradients()) result.push_back(g);
    for (Tensor* g : conv2_.gradients()) result.push_back(g);
    for (Tensor* g : time_mlp_.gradients()) result.push_back(g);
    return result;
}

void DDPMResBlock::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
    time_mlp_.zero_grad();
}

// =============================================================================
// DenoisingUNet
// =============================================================================
DenoisingUNet::DenoisingUNet(int in_channels, const std::vector<int>& channels,
                             int time_emb_dim, int seq_len)
    : in_channels_(in_channels), channels_(channels),
      time_emb_dim_(time_emb_dim), seq_len_(seq_len),
      time_emb_(time_emb_dim),
      conv_out_(channels.back(), in_channels, 3, seq_len, 1, 1) {

    enc_levels_ = static_cast<int>(channels.size());

    // Track per-level spatial dims (seq_len) as we go
    std::vector<int> level_seq_len(enc_levels_);
    level_seq_len[0] = seq_len;
    for (int l = 0; l < enc_levels_ - 1; ++l) {
        level_seq_len[l + 1] = seq_len / (1 << (l + 1));
    }

    // ---- Build encoder ----
    for (int level = 0; level < enc_levels_; ++level) {
        int out_ch = channels[level];
        int cur_seq = level_seq_len[level];
        encoder_channels_.push_back(out_ch);

        // First ResBlock: channel change
        int in_ch = (level == 0) ? in_channels : channels[level - 1];
        DDPMResBlock* block1 = new DDPMResBlock(in_ch, out_ch, time_emb_dim, cur_seq);
        down_blocks_.push_back(block1);

        // Second ResBlock: same channels
        DDPMResBlock* block2 = new DDPMResBlock(out_ch, out_ch, time_emb_dim, cur_seq);
        down_blocks_.push_back(block2);

        // Downsample if not last level
        if (level < enc_levels_ - 1) {
            Conv1D* down = new Conv1D(out_ch, out_ch, 3, cur_seq, 2, 1);
            down_sample_.push_back(down);
        }
    }

    // ---- Middle ----
    int mid_ch = channels.back();
    int mid_seq = level_seq_len[enc_levels_ - 1];
    middle_block1_ = new DDPMResBlock(mid_ch, mid_ch, time_emb_dim, mid_seq);
    middle_block2_ = new DDPMResBlock(mid_ch, mid_ch, time_emb_dim, mid_seq);

    // ---- Build decoder ----
    // Decoder: at each level, upsample 2x then add skip from encoder at same spatial level.
    // up_sample_ is built in forward order (level 0 first = bottom), indexed in reverse for
    // bottom-up decoding: level 0 uses up_sample_[enc_levels_-2], level 1 uses up_sample_[enc_levels_-3].
    for (int level = 0; level < enc_levels_ - 1; ++level) {
        // out_ch at decoder level 0 must match bottleneck (channels.back())
        // to ensure the final output has channels.back() channels before conv_out.
        // Decoder level 0 -> encoder level (enc_levels-1) -> channels[enc_levels-1]
        // Decoder level 1 -> encoder level (enc_levels-2) -> channels[enc_levels-2]
        int out_ch = channels[enc_levels_ - 1 - level];
        int cur_seq = level_seq_len[level];
        int prev_seq = level_seq_len[level + 1];
        decoder_channels_.push_back(out_ch);

        // Upsample: 2x nearest neighbor from prev_seq -> cur_seq
        // up_ch: number of channels at the level we're upsampling from
        // level=0: upsample from bottleneck -> mid_ch = channels.back()
        // level>0: upsample from level below -> channels[level-1]
        int up_ch = (level == 0) ? channels.back() : channels[level - 1];
        Upsample1D* up = new Upsample1D(up_ch, prev_seq);
        up_sample_.push_back(up);

        // SkipTransform: 1x1 conv projects skip from encoder level (skip_in_ch) to out_ch
        // After upsampling: h (out_ch) + concatenated tskip (out_ch) = 2*out_ch channels total
        // ResBlock1 expects 2*out_ch input (concat of h + tskip), reduces to out_ch
        // ResBlock2: out_ch -> out_ch
        int skip_in_ch = channels[enc_levels_ - 1 - level];
        SkipTransform* st1 = new SkipTransform(skip_in_ch, out_ch, cur_seq);
        skip_transforms_.push_back(st1);
        SkipTransform* st2 = new SkipTransform(skip_in_ch, out_ch, cur_seq);
        skip_transforms_.push_back(st2);

        // ResBlock 1: skip (out_ch) + h (out_ch) = 2*out_ch -> out_ch
        DDPMResBlock* block1 = new DDPMResBlock(2 * out_ch, out_ch, time_emb_dim, cur_seq);
        up_blocks_.push_back(block1);

        // ResBlock 2: out_ch -> out_ch
        DDPMResBlock* block2 = new DDPMResBlock(out_ch, out_ch, time_emb_dim, cur_seq);
        up_blocks_.push_back(block2);
    }
}

DenoisingUNet::~DenoisingUNet() {
    for (DDPMResBlock*   b : down_blocks_)    delete b;
    for (Conv1D*        c : down_sample_)    delete c;
    delete middle_block1_;
    delete middle_block2_;
    for (DDPMResBlock*   b : up_blocks_)   delete b;
    for (Upsample1D* u : up_sample_)   delete u;
    for (SkipTransform* s : skip_transforms_) delete s;
}

Tensor DenoisingUNet::forward(const Tensor& x, int t) {
    return forward(x, time_emb_.forward(t));
}

Tensor DenoisingUNet::forward(const Tensor& x, const Tensor& time_emb) {
    last_x_ = x;
    last_time_emb_ = time_emb;
    skip_connections_.clear();

    // ---- Encoder ----
    std::vector<int> level_seq_len(enc_levels_);
    level_seq_len[0] = seq_len_;
    for (int l = 0; l < enc_levels_ - 1; ++l) {
        level_seq_len[l + 1] = seq_len_ / (1 << (l + 1));
    }

    Tensor h = x;
    int block_idx = 0;
    for (int level = 0; level < enc_levels_; ++level) {
        // First ResBlock
        h = down_blocks_[block_idx++]->forward(h, time_emb);
        skip_connections_.push_back(h);

        // Second ResBlock
        h = down_blocks_[block_idx++]->forward(h, time_emb);
        skip_connections_.push_back(h);

        // Downsample if not last level
        if (level < enc_levels_ - 1) {
            h = down_sample_[level]->forward(h);
        }
    }

    // ---- Middle ----
    h = middle_block1_->forward(h, time_emb);
    h = middle_block2_->forward(h, time_emb);

    // ---- Decoder (bottom-up) ----
    // For each level:
    //   1. Upsample h from prev_seq to cur_seq using stored up_sample_[idx]
    //   2. Get skip from encoder, upsample to cur_seq, project to out_ch via 1x1 conv
    //   3. h = h + skip (element-wise addition)
    //   4. Apply ResBlocks to refine
    //
    // skip_connections_ layout for enc_levels_=3, blocks per level=2:
    //   [L0_b1, L0_b2, L1_b1, L1_b2, L2_b1, L2_b2]
    // Decoder level 0 (bottom): uses L2_b1, L2_b2
    // Decoder level 1: uses L1_b1, L1_b2
    for (int level = 0; level < enc_levels_ - 1; ++level) {
        // up_sample_ is indexed in reverse: level 0 uses up_sample_[enc_levels_-2]
        int up_sample_idx = enc_levels_ - 2 - level;
        int up_block_idx = level * 2;

        std::cerr << "[Decoder L" << level << "] up_sample_idx=" << up_sample_idx
                  << ", before h:" << h.rows << "x" << h.cols << std::endl;


        // Step 1: Upsample h from prev_seq to cur_seq
        h = up_sample_[up_sample_idx]->forward(h);
        std::cerr << "[Decoder L" << level << "] after up h:" << h.rows << "x" << h.cols << std::endl;


        // Decoder level 0: up_sample_[0], skip_idx=0 (st[0], st[1])
        // Decoder level 1: up_sample_[1], skip_idx=2 (st[2], st[3])
        int skip_idx = level * 2;
        Tensor skip1 = skip_connections_[skip_idx];
        Tensor skip2 = skip_connections_[skip_idx + 1];
        std::cerr << "[Decoder L" << level << "] skip1:" << skip1.rows << "x" << skip1.cols
                  << " skip2:" << skip2.rows << "x" << skip2.cols << std::endl;

        std::cerr << "[Decoder L" << level << "] st[" << skip_idx << "]->in_ch=" << skip_transforms_[skip_idx]->in_ch() << " st[" << skip_idx+1 << "]->in_ch=" << skip_transforms_[skip_idx+1]->in_ch() << std::endl;


        // Step 2a: Upsample each skip from encoder seq to current decoder seq (same channels)
        // up_sample_[up_sample_idx]: (batch, skip_in_ch * prev_seq) -> (batch, skip_in_ch * cur_seq)
        Tensor uskip1 = up_sample_[up_sample_idx]->forward(skip1);
        Tensor uskip2 = up_sample_[up_sample_idx]->forward(skip2);

        // Step 2b: Project upsampled skip channels from skip_in_ch -> out_ch (at cur_seq resolution)
        // Each SkipTransform: (batch, skip_in_ch * cur_seq) -> (batch, out_ch * cur_seq)
        std::cerr << "[Decoder L" << level << "] BEFORE st[" << skip_idx << "] uskip1:" << uskip1.rows << "x" << uskip1.cols << std::endl;
        std::cerr << "[Decoder L" << level << "] st[" << skip_idx << "] Conv1D: in_ch=" << skip_transforms_[skip_idx]->in_ch() << " out_ch=" << skip_transforms_[skip_idx]->out_ch() << std::endl;
        Tensor tskip1 = skip_transforms_[skip_idx]->forward(uskip1);
        std::cerr << "[Decoder L" << level << "] AFTER st[" << skip_idx << "] tskip1:" << tskip1.rows << "x" << tskip1.cols << std::endl;
        Tensor tskip2 = skip_transforms_[skip_idx + 1]->forward(uskip2);
        std::cerr << "[Decoder L" << level << "] AFTER st[" << skip_idx+1 << "] tskip2:" << tskip2.rows << "x" << tskip2.cols << std::endl;
        Tensor tskip = tskip1 + tskip2;
        std::cerr << "[Decoder L" << level << "] tskip:" << tskip.rows << "x" << tskip.cols << std::endl;

        std::cerr << "[Decoder L" << level << "] h:" << h.rows << "x" << h.cols << " tskip:" << tskip.rows << "x" << tskip.cols << std::endl;


        // Step 3: Concatenate h and transformed skip along channels (column dimension)
        // h: (batch, out_ch * cur_seq), tskip: (batch, out_ch * cur_seq)
        // concat: (batch, 2 * out_ch * cur_seq) -> ResBlock reduces to (batch, out_ch * cur_seq)
        h = h.concatenate(tskip, true);

        // Step 5: Apply ResBlocks to refine
        std::cerr << "[Decoder L" << level << "] up_block[" << up_block_idx << "].in_ch=" << up_blocks_[up_block_idx]->in_ch() << " .out_ch=" << up_blocks_[up_block_idx]->out_ch() << " .seq_len_=" << up_blocks_[up_block_idx]->seq_len() << std::endl;
        h = up_blocks_[up_block_idx++]->forward(h, time_emb);
        std::cerr << "[Decoder L" << level << "] after block1 h:" << h.rows << "x" << h.cols << std::endl;
        std::cerr << "[Decoder L" << level << "] up_block[" << up_block_idx << "].in_ch=" << up_blocks_[up_block_idx]->in_ch() << " .out_ch=" << up_blocks_[up_block_idx]->out_ch() << " .seq_len_=" << up_blocks_[up_block_idx]->seq_len() << std::endl;
        h = up_blocks_[up_block_idx++]->forward(h, time_emb);
        std::cerr << "[Decoder L" << level << "] after block2 h:" << h.rows << "x" << h.cols << std::endl;
    }

    // ---- Output conv ----
    std::cerr << "[Output conv] h:" << h.rows << "x" << h.cols << " conv_out:in_ch=" << channels_.back() << " out_ch=" << in_channels_ << " seq_len=" << seq_len_ << std::endl;
    h = conv_out_.forward(h);

    return h;
}

Tensor DenoisingUNet::backward(const Tensor& grad_output, double learning_rate) {
    // ---- Decoder (bottom-up) ----
    // Backward in reverse order:
    //   conv_out backward, up_blocks backward, skip_transforms backward, up_sample backward
    Tensor grad = conv_out_.backward(grad_output, learning_rate);

    for (int level = enc_levels_ - 2; level >= 0; --level) {
        int up_sample_idx = enc_levels_ - 2 - level;
        int up_block_idx = level * 2;

        // Backward through up_blocks_ (reverse order)
        grad = up_blocks_[up_block_idx + 1]->backward(grad, learning_rate);
        grad = up_blocks_[up_block_idx]->backward(grad, learning_rate);

        // Backward through skip_transforms_ (reverse construction order)
        // skip_idx for backward: same formula as forward (level * 2)
        int skip_idx = level * 2;
        for (int s = 0; s < 2; ++s) {
            skip_transforms_[skip_idx + s]->backward(grad, learning_rate);
        }

        // Backward through up_sample_ (nearest neighbor — gradient is downsample)
        grad = up_sample_[up_sample_idx]->backward(grad, learning_rate);
    }

    // ---- Middle backward ----
    grad = middle_block2_->backward(grad, learning_rate);
    grad = middle_block1_->backward(grad, learning_rate);

    // ---- Encoder backward ----
    for (int level = enc_levels_ - 1; level >= 0; --level) {
        int block_base = level * 2;
        if (level < enc_levels_ - 1) {
            grad = down_sample_[level]->backward(grad, learning_rate);
        }
        grad = down_blocks_[block_base + 1]->backward(grad, learning_rate);
        grad = down_blocks_[block_base]->backward(grad, learning_rate);
    }

    return grad;
}

void DenoisingUNet::update_weights(double learning_rate) {
    for (DDPMResBlock*   b : down_blocks_)   b->update_weights(learning_rate);
    for (Conv1D*     c : down_sample_)   c->update_weights(learning_rate);
    middle_block1_->update_weights(learning_rate);
    middle_block2_->update_weights(learning_rate);
    for (DDPMResBlock*   b : up_blocks_)     b->update_weights(learning_rate);
    for (Upsample1D* u : up_sample_)    u->update_weights(learning_rate);
    for (SkipTransform* s : skip_transforms_) s->update_weights(learning_rate);
    conv_out_.update_weights(learning_rate);
}

std::vector<Tensor*> DenoisingUNet::parameters() {
    std::vector<Tensor*> result;
    for (DDPMResBlock* b : down_blocks_)
        for (Tensor* p : b->parameters()) result.push_back(p);
    for (Conv1D* c : down_sample_)
        for (Tensor* p : c->parameters()) result.push_back(p);
    for (Tensor* p : middle_block1_->parameters()) result.push_back(p);
    for (Tensor* p : middle_block2_->parameters()) result.push_back(p);
    for (DDPMResBlock* b : up_blocks_)
        for (Tensor* p : b->parameters()) result.push_back(p);
    for (SkipTransform* s : skip_transforms_)
        for (Tensor* p : s->parameters()) result.push_back(p);
    for (Tensor* p : conv_out_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> DenoisingUNet::gradients() {
    std::vector<Tensor*> result;
    for (DDPMResBlock* b : down_blocks_)
        for (Tensor* g : b->gradients()) result.push_back(g);
    for (Conv1D* c : down_sample_)
        for (Tensor* g : c->gradients()) result.push_back(g);
    for (Tensor* g : middle_block1_->gradients()) result.push_back(g);
    for (Tensor* g : middle_block2_->gradients()) result.push_back(g);
    for (DDPMResBlock* b : up_blocks_)
        for (Tensor* g : b->gradients()) result.push_back(g);
    for (SkipTransform* s : skip_transforms_)
        for (Tensor* g : s->gradients()) result.push_back(g);
    for (Tensor* g : conv_out_.gradients()) result.push_back(g);
    return result;
}

void DenoisingUNet::zero_grad() {
    for (DDPMResBlock*   b : down_blocks_)   b->zero_grad();
    for (Conv1D*     c : down_sample_)   c->zero_grad();
    middle_block1_->zero_grad();
    middle_block2_->zero_grad();
    for (DDPMResBlock*   b : up_blocks_)    b->zero_grad();
    for (Upsample1D* u : up_sample_)    u->zero_grad();
    for (SkipTransform* s : skip_transforms_) s->zero_grad();
    conv_out_.zero_grad();
}

// =============================================================================
// DDPMModel
// =============================================================================
DDPMModel::DDPMModel(int in_channels, const std::vector<int>& channels,
                     int time_emb_dim, int seq_len, int diffusion_steps,
                     float beta_start, float beta_end)
    : in_channels_(in_channels), channels_(channels),
      time_emb_dim_(time_emb_dim), seq_len_(seq_len),
      diffusion_steps_(diffusion_steps),
      beta_start_(beta_start), beta_end_(beta_end),
      scheduler_(diffusion_steps, beta_start, beta_end),
      rng_(42), dist_t_(0, diffusion_steps - 1), normal_(0.0f, 1.0f) {
    setup();
}

DDPMModel::~DDPMModel() = default;

void DDPMModel::setup() {
    unet_ = std::make_unique<DenoisingUNet>(in_channels_, channels_,
                                             time_emb_dim_, seq_len_);
}

Tensor DDPMModel::training_forward(const Tensor& x0) {
    // Sample random timestep
    int t = dist_t_(rng_);

    // Generate noise
    size_t batch = x0.rows;
    size_t dim   = x0.cols;
    Tensor noise(batch, dim);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            noise[i][j] = normal_(rng_);
        }
    }

    // Forward noising: x_t = sqrt(alphabar_t) * x0 + sqrt(1-alphabar_t) * noise
    Tensor x_t = scheduler_.q_sample(x0, t, noise);

    // Predict noise epsilon_theta(x_t, t)
    Tensor eps_theta = unet_->forward(x_t, t);

    // MSE loss = ||epsilon - epsilon_theta||^2
    double mse = 0.0;
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            double diff = eps_theta[i][j] - noise[i][j];
            mse += diff * diff;
        }
    }
    mse /= static_cast<double>(batch * dim);

    last_x0_    = x0;
    last_xt_    = x_t;
    last_noise_ = noise;
    last_t_     = t;
    last_loss_  = static_cast<float>(mse);

    return eps_theta;
}

float DDPMModel::loss(const Tensor& x0) {
    (void)x0;
    return last_loss_;
}

Tensor DDPMModel::denoise(const Tensor& x_t, int t) {
    // Predict noise epsilon_theta(x_t, t)
    Tensor eps_theta = unet_->forward(x_t, t);

    size_t batch = x_t.rows;
    size_t dim   = x_t.cols;

    float sqrt_recip_alpha   = scheduler_.sqrt_recip_alphas(t);
    float sqrt_one_minus_bar = scheduler_.sqrt_one_minus_alphas_cumprod(t);
    float beta_t             = scheduler_.betas[t];
    float sigma_t            = std::sqrt(beta_t);
    float alpha_t            = 1.0f - scheduler_.betas[t];

    // DDPM mean formula (reparameterized from epsilon prediction):
    // x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alphabar_t) * eps_theta) + sigma_t * z
    float coef1 = sqrt_recip_alpha;
    float coef2 = (1.0f - alpha_t) / (sqrt_recip_alpha * sqrt_one_minus_bar);

    Tensor x_prev(batch, dim);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            x_prev[i][j] = coef1 * x_t[i][j] - coef2 * eps_theta[i][j];
        }
    }

    // Add noise for t > 0 (standard DDPM sampler)
    if (t > 0) {
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                x_prev[i][j] += sigma_t * normal_(rng_);
            }
        }
    }

    return x_prev;
}

Tensor DDPMModel::sample(int num_samples) {
    int T = diffusion_steps_;
    size_t dim = static_cast<size_t>(in_channels_) * static_cast<size_t>(seq_len_);

    // Start from pure Gaussian noise
    Tensor x(num_samples, dim);
    for (size_t i = 0; i < static_cast<size_t>(num_samples); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            x[i][j] = normal_(rng_);
        }
    }

    // Reverse diffusion loop
    for (int t = T - 1; t >= 0; --t) {
        x = denoise(x, t);
    }

    return x;
}

Tensor DDPMModel::forward(const Tensor& input) {
    return training_forward(input);
}

Tensor DDPMModel::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output;
    (void)learning_rate;
    return grad_output;
}

void DDPMModel::update_weights(double learning_rate) {
    unet_->update_weights(learning_rate);
}

Tensor DDPMModel::get_weights() const {
    auto params = unet_->parameters();
    return params.empty() ? Tensor(0, 0) : *params[0];
}

Tensor DDPMModel::get_gradients() const {
    auto grads = unet_->gradients();
    return grads.empty() ? Tensor(0, 0) : *grads[0];
}

std::vector<Tensor*> DDPMModel::parameters() {
    return unet_->parameters();
}

std::vector<Tensor*> DDPMModel::gradients() {
    return unet_->gradients();
}

void DDPMModel::zero_grad() {
    unet_->zero_grad();
}