#include "consistency.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// =====================================================================
// Local im2col helper (mirrors Conv2D::im2col, no protection needed)
// =====================================================================
static Tensor local_im2col(const Tensor& input, int N, int C, int H, int W,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dilation_h, int dilation_w,
                           int& H_out, int& W_out) {
    H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    Tensor col(C * kH * kW, N * H_out * W_out);
    col.fill(0.0);

    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            for (int j_out = 0; j_out < W_out; ++j_out) {
                int col_idx = n * H_out * W_out + i_out * W_out + j_out;
                for (int c = 0; c < C; ++c) {
                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h = i_out * stride_h + i * dilation_h - pad_h;
                            int w = j_out * stride_w + j * dilation_w - pad_w;
                            double val = 0.0;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                val = input[n][c * H * W + h * W + w];
                            }
                            int row_idx = c * kH * kW + i * kW + j;
                            col[row_idx][col_idx] = val;
                        }
                    }
                }
            }
        }
    }
    return col;
}

// =====================================================================
// Local col2im helper
// =====================================================================
static void local_col2im(Tensor& grad_input, const Tensor& grad_col,
                         int N, int C, int H, int W,
                         int kH, int kW, int stride_h, int stride_w,
                         int pad_h, int pad_w, int dilation_h, int dilation_w) {
    int H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    for (int n = 0; n < N; ++n) {
        for (int i_out = 0; i_out < H_out; ++i_out) {
            for (int j_out = 0; j_out < W_out; ++j_out) {
                int col_idx = n * H_out * W_out + i_out * W_out + j_out;
                for (int c = 0; c < C; ++c) {
                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h = i_out * stride_h + i * dilation_h - pad_h;
                            int w = j_out * stride_w + j * dilation_w - pad_w;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int row_idx = c * kH * kW + i * kW + j;
                                grad_input[n][c * H * W + h * W + w] += grad_col[row_idx][col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// =====================================================================
// ConvTranspose2D
// =====================================================================
ConvTranspose2D::ConvTranspose2D(int in_ch, int out_ch, int kH, int kW,
                                 int H_in, int W_in,
                                 int stride_h, int stride_w,
                                 int pad_h, int pad_w,
                                 int output_pad_h, int output_pad_w)
    : in_channels(in_ch), out_channels(out_ch),
      kernel_h(kH), kernel_w(kW),
      stride_h(stride_h), stride_w(stride_w),
      pad_h(pad_h), pad_w(pad_w),
      output_pad_h(output_pad_h), output_pad_w(output_pad_w),
      H(H_in), W(W_in)
{
    // Compute output spatial dimensions
    // H_out = (H_in - 1) * stride - 2 * pad + kernel_h + output_pad_h
    int H_out_tmp = (H - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int W_out_tmp = (W - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    (void)H_out_tmp; (void)W_out_tmp;

    // Initialize weights: same convention as Conv2D
    // weight shape: (out_channels, in_channels * kernel_h * kernel_w)
    int fan_in = in_channels * kernel_h * kernel_w;
    int fan_out = out_channels * kernel_h * kernel_w;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));

    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, scale);

    weights = Tensor(out_channels, in_channels * kernel_h * kernel_w);
    bias = Tensor(out_channels, 1);
    grad_weights = Tensor(out_channels, in_channels * kernel_h * kernel_w);
    grad_bias = Tensor(out_channels, 1);

    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kernel_h * kernel_w; ++i) {
            weights[o][i] = dis(gen);
        }
        bias[o][0] = 0.0;
    }
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

Tensor ConvTranspose2D::forward(const Tensor& input) {
    int N = input.rows;
    if (input.cols != static_cast<size_t>(in_channels * H * W)) {
        throw std::invalid_argument("ConvTranspose2D: input dimension mismatch");
    }

    last_input = input;

    // Compute output spatial dimensions
    int H_out = (H - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int W_out = (W - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    // Reshape input to 4D and expand (insert zeros between spatial positions)
    // Input: (N, in_channels * H * W) row-major (N, C, H, W)
    int H_exp = H * stride_h;
    int W_exp = W * stride_w;
    Tensor expanded(N, in_channels * H_exp * W_exp);
    expanded.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int src_idx = n * in_channels * H * W + c * H * W + h * W + w;
                    int dst_h = h * stride_h;
                    int dst_w = w * stride_w;
                    int dst_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + dst_h * W_exp + dst_w;
                    expanded.data[dst_idx] = input.data[src_idx];
                }
            }
        }
    }

    // Pad for convolution
    int H_padded = H_exp + 2 * pad_h;
    int W_padded = W_exp + 2 * pad_w;
    Tensor padded(N, in_channels * H_padded * W_padded);
    padded.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H_exp; ++h) {
                for (int w = 0; w < W_exp; ++w) {
                    int src_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + h * W_exp + w;
                    int dst_h = h + pad_h;
                    int dst_w = w + pad_w;
                    int dst_idx = n * in_channels * H_padded * W_padded + c * H_padded * W_padded + dst_h * W_padded + dst_w;
                    padded.data[dst_idx] = expanded.data[src_idx];
                }
            }
        }
    }

    // im2col on padded expanded input (stride=1, pad=0 after expansion)
    int H_col, W_col;
    Tensor col = local_im2col(padded, N, in_channels, H_padded, W_padded,
                               kernel_h, kernel_w, 1, 1, 0, 0, 1, 1,
                               H_col, W_col);

    // Weights: (out_channels, in_channels * kH * kW)
    // Z = weights * col -> (out_ch, N * H_out * W_out)
    Tensor Z = weights * col;

    // Add bias
    int out_spatial = H_out * W_out;
    for (int o = 0; o < out_channels; ++o) {
        for (int idx = 0; idx < N * out_spatial; ++idx) {
            Z[o][idx] += bias[o][0];
        }
    }

    // Reshape to (N, out_channels * H_out * W_out)
    Tensor output(N, out_channels * out_spatial);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int s = 0; s < out_spatial; ++s) {
                output[n][o * out_spatial + s] = Z[o][n * out_spatial + s];
            }
        }
    }
    return output;
}

Tensor ConvTranspose2D::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    int N = grad_output.rows;

    int H_out = (H - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int W_out = (W - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    int out_spatial = H_out * W_out;

    // Reshape grad_output: (N, out_ch * out_spatial) -> (out_ch, N * out_spatial)
    Tensor grad_out_mat(out_channels, N * out_spatial);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int s = 0; s < out_spatial; ++s) {
                grad_out_mat[o][n * out_spatial + s] = grad_output[n][o * out_spatial + s];
            }
        }
    }

    // Reconstruct padded expanded input from last_input (2D)
    // last_input: (N, in_channels * H * W)
    int H_exp = H * stride_h;
    int W_exp = W * stride_w;
    int H_padded = H_exp + 2 * pad_h;
    int W_padded = W_exp + 2 * pad_w;

    // Expand last_input
    Tensor expanded(N, in_channels * H_exp * W_exp);
    expanded.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int src_idx = n * in_channels * H * W + c * H * W + h * W + w;
                    int dst_h = h * stride_h;
                    int dst_w = w * stride_w;
                    int dst_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + dst_h * W_exp + dst_w;
                    expanded.data[dst_idx] = last_input.data[src_idx];
                }
            }
        }
    }

    // Pad
    Tensor padded(N, in_channels * H_padded * W_padded);
    padded.fill(0.0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H_exp; ++h) {
                for (int w = 0; w < W_exp; ++w) {
                    int src_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + h * W_exp + w;
                    int dst_h = h + pad_h;
                    int dst_w = w + pad_w;
                    int dst_idx = n * in_channels * H_padded * W_padded + c * H_padded * W_padded + dst_h * W_padded + dst_w;
                    padded.data[dst_idx] = expanded.data[src_idx];
                }
            }
        }
    }

    // im2col on padded
    int H_col, W_col;
    Tensor col = local_im2col(padded, N, in_channels, H_padded, W_padded,
                              kernel_h, kernel_w, 1, 1, 0, 0, 1, 1,
                              H_col, W_col);

    // dW = grad_out_mat * col^T
    Tensor col_T = col.transpose();
    Tensor dW = grad_out_mat * col_T;
    grad_weights = grad_weights + dW;

    // d_bias
    Tensor db(out_channels, 1);
    for (int o = 0; o < out_channels; ++o) {
        double sum = 0.0;
        for (int i = 0; i < N * out_spatial; ++i) {
            sum += grad_out_mat[o][i];
        }
        db[o][0] = sum;
    }
    grad_bias = grad_bias + db;

    // dX_col = weights^T * grad_out_mat
    Tensor weights_T = weights.transpose();
    Tensor dX_col = weights_T * grad_out_mat;

    // col2im to get grad_padded
    Tensor grad_padded(N, in_channels * H_padded * W_padded);
    grad_padded.fill(0.0);
    local_col2im(grad_padded, dX_col, N, in_channels, H_padded, W_padded,
                 kernel_h, kernel_w, 1, 1, 0, 0, 1, 1);

    // Remove padding: (N, in_ch, H_padded, W_padded) -> (N, in_ch, H_exp, W_exp)
    Tensor grad_exp(N, in_channels * H_exp * W_exp);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H_exp; ++h) {
                for (int w = 0; w < W_exp; ++w) {
                    int src_h = h + pad_h;
                    int src_w = w + pad_w;
                    int src_idx = n * in_channels * H_padded * W_padded + c * H_padded * W_padded + src_h * W_padded + src_w;
                    int dst_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + h * W_exp + w;
                    grad_exp.data[dst_idx] = grad_padded.data[src_idx];
                }
            }
        }
    }

    // Remove zeros between spatial positions (stride > 1 downsampling)
    // (N, in_ch, H_exp, W_exp) -> (N, in_ch, H, W)
    Tensor grad_input(N, in_channels * H * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int src_h = h * stride_h;
                    int src_w = w * stride_w;
                    int src_idx = n * in_channels * H_exp * W_exp + c * H_exp * W_exp + src_h * W_exp + src_w;
                    int dst_idx = n * in_channels * H * W + c * H * W + h * W + w;
                    grad_input.data[dst_idx] = grad_exp.data[src_idx];
                }
            }
        }
    }

    return grad_input;
}

void ConvTranspose2D::update_weights(double learning_rate) {
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels * kernel_h * kernel_w; ++i) {
            weights[o][i] -= learning_rate * grad_weights[o][i];
        }
        bias[o][0] -= learning_rate * grad_bias[o][0];
    }
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

std::vector<Tensor*> ConvTranspose2D::parameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> ConvTranspose2D::gradients() {
    return {&grad_weights, &grad_bias};
}

void ConvTranspose2D::zero_grad() {
    grad_weights.fill(0.0);
    grad_bias.fill(0.0);
}

// =====================================================================
// TimeMLP
// =====================================================================
TimeMLP::TimeMLP(int in_dim, int hidden_dim, int out_dim) {
    fc1 = new Dense(in_dim, hidden_dim);
    fc2 = new Dense(hidden_dim, out_dim);
}

// =====================================================================
// ConsistencyTimeEmbedding
// =====================================================================
ConsistencyTimeEmbedding::ConsistencyTimeEmbedding(int dim, int max_len)
    : dim_(dim), half_(dim / 2), max_len_(max_len)
{
    freq_.resize(half_);
    for (int i = 0; i < half_; ++i) {
        freq_[i] = std::exp(std::log(10000.0) * (-2.0 * i / dim_));
    }
}

Tensor ConsistencyTimeEmbedding::embed(double t) const {
    Tensor result(1, dim_);
    double* out = result.data.data();

    // Clamp t to [0, 1] range
    t = std::max(0.0, std::min(1.0, t));

    for (int i = 0; i < half_; ++i) {
        double freq = freq_[i];
        out[i]         = std::sin(t * freq);
        out[i + half_] = std::cos(t * freq);
    }
    return result;
}

// =====================================================================
// Helper functions
// =====================================================================
static double sigmoid_fn(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Tensor ConsistencyStudent::silu(const Tensor& x) {
    return x.apply([](double v) { return v * sigmoid_fn(v); });
}

Tensor ConsistencyStudent::upsample2x(const Tensor& x, int N, int C, int H, int W) {
    // x: (N, C*H*W) row-major where spatial is (H, W)
    // Upsample by 2: (N, C, H, W) -> (N, C, 2H, 2W) -> (N, C*4H*W)
    Tensor result(N, C * 2 * H * 2 * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    double val = x(n, c * H * W + h * W + w);
                    int base = n * C * 2 * H * 2 * W + c * 2 * H * 2 * W;
                    int r1 = base + (2*h)   * 2 * W + (2*w);
                    int r2 = base + (2*h)   * 2 * W + (2*w+1);
                    int r3 = base + (2*h+1) * 2 * W + (2*w);
                    int r4 = base + (2*h+1) * 2 * W + (2*w+1);
                    result.data[r1] = val;
                    result.data[r2] = val;
                    result.data[r3] = val;
                    result.data[r4] = val;
                }
            }
        }
    }
    return result;
}

Tensor ConsistencyStudent::reshape_4d(const Tensor& flat, int N, int C, int H, int W) const {
    (void)flat; (void)N; (void)C; (void)H; (void)W;
    // Already flat row-major (N, C*H*W), no change needed
    return Tensor(flat.rows, flat.cols);
}

Tensor ConsistencyStudent::flatten_4d(const Tensor& tensor, int N, int C, int H, int W) const {
    (void)tensor; (void)N; (void)C; (void)H; (void)W;
    return tensor;
}

Tensor ConsistencyStudent::add_time(const Tensor& feat, const Tensor& te, int out_ch) {
    // Placeholder — addition done explicitly in forward where shapes are known
    (void)feat; (void)te; (void)out_ch;
    return feat;
}

// =====================================================================
// ConsistencyStudent
// =====================================================================
ConsistencyStudent::ConsistencyStudent(int channels, int time_dim, int num_classes,
                                       int depth, int base_ch, int H, int W)
    : channels_(channels), time_dim_(time_dim), num_classes_(num_classes),
      depth_(depth), base_ch_(base_ch), H_(H), W_(W),
      time_embed_(time_dim),
      class_embed_(num_classes, Tensor(1, class_embed_dim_)),
      class_proj_(std::make_unique<Dense>(class_embed_dim_, time_dim))
{
    // Initialize class embeddings with small random values
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.02);
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < class_embed_dim_; ++j) {
            class_embed_[i].data[j] = dis(gen);
        }
    }

    // Build encoder blocks
    int ch = channels;
    int H_cur = H;
    int W_cur = W;
    for (int i = 0; i < depth; ++i) {
        int out_ch = base_ch * (1 << i);
        EncBlock blk;
        blk.out_channels = out_ch;
        blk.conv1 = Conv2D(ch, out_ch, 3, 3, H_cur, W_cur, 1, 1, 1, 1);
        blk.conv2 = Conv2D(out_ch, out_ch, 3, 3, H_cur, W_cur, 1, 1, 1, 1);
        // downsample: Conv2D with stride=2, no padding
        int H_ds = (H_cur + 1) / 2; // ceil(H_cur / 2)
        int W_ds = (W_cur + 1) / 2;
        blk.downsample = Conv2D(out_ch, out_ch, 3, 3, H_cur, W_cur, 2, 2, 0, 0);
        blk.downsample.H = H_cur;
        blk.downsample.W = W_cur;
        enc_blocks_.push_back(blk);
        ch = out_ch;
        H_cur = H_ds;
        W_cur = W_ds;
    }

    // Build time MLPs for each resolution level
    for (int i = 0; i < depth; ++i) {
        int ch_i = base_ch * (1 << i);
        time_mlps_.push_back(TimeMLP(time_dim, ch_i, ch_i));
    }

    // Middle convs (bottleneck)
    int mid_ch = base_ch * (1 << (depth - 1));
    int H_mid = H / (1 << depth);
    int W_mid = W / (1 << depth);
    mid_conv1_ = Conv2D(mid_ch, mid_ch, 3, 3, H_mid, W_mid, 1, 1, 1, 1);
    mid_conv2_ = Conv2D(mid_ch, mid_ch, 3, 3, H_mid, W_mid, 1, 1, 1, 1);

    // Build decoder blocks (reverse order of encoder)
    ch = mid_ch;
    int H_dec = H_mid;
    int W_dec = W_mid;
    for (int i = depth - 1; i >= 0; --i) {
        int enc_ch = base_ch * (1 << i);
        int H_enc = H / (1 << i);
        int W_enc = W / (1 << i);
        DecBlock blk;
        blk.out_channels = enc_ch;
        blk.upsample = ConvTranspose2D(ch, enc_ch, 4, 4, H_dec, W_dec, 2, 2, 1, 1, 1, 1);
        blk.upsample.H = H_dec;
        blk.upsample.W = W_dec;
        blk.conv1 = Conv2D(enc_ch * 2, enc_ch, 3, 3, H_enc, W_enc, 1, 1, 1, 1);
        blk.conv2 = Conv2D(enc_ch, enc_ch, 3, 3, H_enc, W_enc, 1, 1, 1, 1);
        dec_blocks_.push_back(blk);

        // Skip transform: 1x1 conv to match encoder channels
        Conv2D skip_trans(enc_ch, enc_ch, 1, 1, H_enc, W_enc, 1, 1, 0, 0);
        skip_transforms_.push_back(skip_trans);

        ch = enc_ch;
        H_dec = H_enc;
        W_dec = W_enc;
    }

    // Final output conv: base_ch -> channels
    final_conv_ = Conv2D(base_ch, channels, 3, 3, H, W, 1, 1, 1, 1);
    final_conv_.H = H;
    final_conv_.W = W;

    // Gradient buffers for decoder
    grad_bufs_.resize(dec_blocks_.size());
    for (size_t i = 0; i < dec_blocks_.size(); ++i) {
        int ch_i = dec_blocks_[i].out_channels;
        int H_i = H / (1 << (depth - 1 - i));
        grad_bufs_[i] = Tensor(ch_i, H_i * W);
    }
}

Tensor ConsistencyStudent::forward(const Tensor& input) {
    last_input_ = input;

    // Get time embedding
    Tensor te = time_embed_.embed(last_t_);

    // Get class embedding and project
    Tensor ce;
    if (last_class_label_ >= 0 && last_class_label_ < num_classes_) {
        ce = class_embed_[last_class_label_];
    } else {
        ce = Tensor(1, class_embed_dim_);
        ce.fill(0.0);
    }
    Tensor ce_proj = class_proj_->forward(ce);
    Tensor t_enc = te + ce_proj;

    int N = input.rows;
    int H_cur = H_;
    int W_cur = W_;
    int ch_cur = channels_;

    std::vector<Tensor> enc_outputs;
    std::vector<Tensor> enc_time_outs;

    Tensor x = input;
    for (int i = 0; i < depth_; ++i) {
        EncBlock& blk = enc_blocks_[i];

        // conv1 -> SiLU -> conv2 -> SiLU
        Tensor h = blk.conv1.forward(x);
        h = silu(h);
        h = blk.conv2.forward(h);
        h = silu(h);
        enc_outputs.push_back(h);

        // Time MLP at this resolution
        Tensor te_i = time_mlps_[i].fc1->forward(t_enc);
        te_i = silu(te_i);
        te_i = time_mlps_[i].fc2->forward(te_i);
        enc_time_outs.push_back(te_i);

        // Add time embedding to feature maps (broadcast across spatial dims)
        int spatial = H_cur * W_cur;
        int blk_ch = blk.out_channels;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                for (int c = 0; c < blk_ch; ++c) {
                    h[n][c * spatial + s] += te_i(0, c);
                }
            }
        }

        // Downsample for next block (unless last encoder block)
        if (i < depth_ - 1) {
            x = blk.downsample.forward(h);
            H_cur /= 2;
            W_cur /= 2;
            ch_cur = blk.out_channels;
        } else {
            x = h; // Bottleneck
        }
    }

    // Middle convs
    x = mid_conv1_.forward(x);
    x = silu(x);
    x = mid_conv2_.forward(x);
    x = silu(x);

    // Decoder with skip connections
    for (int i = depth_ - 1; i >= 0; --i) {
        DecBlock& blk = dec_blocks_[i];
        EncBlock& enc_blk = enc_blocks_[i];
        int H_enc = H_ / (1 << i);
        int W_enc = W_ / (1 << i);
        int spatial = H_enc * W_enc;
        int enc_ch = enc_blk.out_channels;

        // Upsample
        x = blk.upsample.forward(x);
        H_cur = H_enc;
        W_cur = W_enc;

        // Skip connection: transform encoder output
        Tensor skip = skip_transforms_[i].forward(enc_outputs[i]);

        // Concatenate along channel dimension
        // x: (N, enc_ch * spatial), skip: (N, enc_ch * spatial)
        Tensor concat(N, enc_ch * 2 * spatial);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                for (int c = 0; c < enc_ch; ++c) {
                    concat[n][c * spatial + s]            = x[n][c * spatial + s];
                    concat[n][enc_ch * spatial + c * spatial + s] = skip[n][c * spatial + s];
                }
            }
        }

        // conv1 -> SiLU -> conv2 -> SiLU
        x = blk.conv1.forward(concat);
        x = silu(x);
        x = blk.conv2.forward(x);
        x = silu(x);

        // Add time embedding
        Tensor te_i = enc_time_outs[i];
        int x_spatial = H_cur * W_cur;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < x_spatial; ++s) {
                for (int c = 0; c < enc_ch; ++c) {
                    x[n][c * x_spatial + s] += te_i(0, c);
                }
            }
        }
    }

    // Final output conv
    x = final_conv_.forward(x);

    return x;
}

Tensor ConsistencyStudent::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;
    int N = grad_output.rows;

    // Gradient through final_conv_
    Tensor g = final_conv_.backward(grad_output, 0.0);

    // Backprop through decoder blocks (reverse order)
    for (int i = depth_ - 1; i >= 0; --i) {
        EncBlock& enc_blk = enc_blocks_[i];
        int enc_ch = enc_blk.out_channels;
        int H_enc = H_ / (1 << i);
        int W_enc = W_ / (1 << i);
        int spatial = H_enc * W_enc;

        // conv2 backward
        g = dec_blocks_[i].conv2.backward(g, 0.0);

        // conv1 backward
        g = dec_blocks_[i].conv1.backward(g, 0.0);

        // Split gradient into upsample and skip parts
        // g: gradient w.r.t. concat output (enc_ch * 2 * spatial)
        Tensor grad_upsample(N, enc_ch * spatial);
        Tensor grad_skip(N, enc_ch * spatial);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < enc_ch * spatial; ++s) {
                grad_upsample[n][s] = g[n][s];
                grad_skip[n][s]     = g[n][enc_ch * spatial + s];
            }
        }

        // Backprop through skip_transforms_
        grad_skip = skip_transforms_[i].backward(grad_skip, 0.0);

        // Backprop through upsample and accumulate with skip gradient
        Tensor grad_up = dec_blocks_[i].upsample.backward(grad_upsample, 0.0);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < enc_ch * spatial; ++s) {
                grad_up[n][s] += grad_skip[n][s];
            }
        }
        g = grad_up;
    }

    // Backprop through middle convs
    g = mid_conv2_.backward(g, 0.0);
    g = mid_conv1_.backward(g, 0.0);

    // Backprop through encoder (reverse order)
    for (int i = depth_ - 1; i >= 0; --i) {
        if (i < depth_ - 1) {
            g = enc_blocks_[i].downsample.backward(g, 0.0);
        }
        g = enc_blocks_[i].conv2.backward(g, 0.0);
        g = enc_blocks_[i].conv1.backward(g, 0.0);
    }

    return g;
}

void ConsistencyStudent::update_weights(double learning_rate) {
    for (auto& blk : enc_blocks_) {
        blk.conv1.update_weights(learning_rate);
        blk.conv2.update_weights(learning_rate);
        blk.downsample.update_weights(learning_rate);
    }
    for (auto& blk : dec_blocks_) {
        blk.upsample.update_weights(learning_rate);
        blk.conv1.update_weights(learning_rate);
        blk.conv2.update_weights(learning_rate);
    }
    for (auto& mlp : time_mlps_) {
        mlp.fc1->update_weights(learning_rate);
        mlp.fc2->update_weights(learning_rate);
    }
    mid_conv1_.update_weights(learning_rate);
    mid_conv2_.update_weights(learning_rate);
    final_conv_.update_weights(learning_rate);
    class_proj_->update_weights(learning_rate);
}

std::vector<Tensor*> ConsistencyStudent::parameters() {
    std::vector<Tensor*> params;
    for (auto& blk : enc_blocks_) {
        for (Tensor* p : blk.conv1.parameters()) params.push_back(p);
        for (Tensor* p : blk.conv2.parameters()) params.push_back(p);
        for (Tensor* p : blk.downsample.parameters()) params.push_back(p);
    }
    for (auto& blk : dec_blocks_) {
        for (Tensor* p : blk.upsample.parameters()) params.push_back(p);
        for (Tensor* p : blk.conv1.parameters()) params.push_back(p);
        for (Tensor* p : blk.conv2.parameters()) params.push_back(p);
    }
    for (auto& mlp : time_mlps_) {
        for (Tensor* p : mlp.fc1->parameters()) params.push_back(p);
        for (Tensor* p : mlp.fc2->parameters()) params.push_back(p);
    }
    for (Tensor* p : mid_conv1_.parameters()) params.push_back(p);
    for (Tensor* p : mid_conv2_.parameters()) params.push_back(p);
    for (Tensor* p : final_conv_.parameters()) params.push_back(p);
    for (Tensor* p : class_proj_->parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> ConsistencyStudent::gradients() {
    std::vector<Tensor*> grads;
    for (auto& blk : enc_blocks_) {
        for (Tensor* g : blk.conv1.gradients()) grads.push_back(g);
        for (Tensor* g : blk.conv2.gradients()) grads.push_back(g);
        for (Tensor* g : blk.downsample.gradients()) grads.push_back(g);
    }
    for (auto& blk : dec_blocks_) {
        for (Tensor* g : blk.upsample.gradients()) grads.push_back(g);
        for (Tensor* g : blk.conv1.gradients()) grads.push_back(g);
        for (Tensor* g : blk.conv2.gradients()) grads.push_back(g);
    }
    for (auto& mlp : time_mlps_) {
        for (Tensor* g : mlp.fc1->gradients()) grads.push_back(g);
        for (Tensor* g : mlp.fc2->gradients()) grads.push_back(g);
    }
    for (Tensor* g : mid_conv1_.gradients()) grads.push_back(g);
    for (Tensor* g : mid_conv2_.gradients()) grads.push_back(g);
    for (Tensor* g : final_conv_.gradients()) grads.push_back(g);
    for (Tensor* g : class_proj_->gradients()) grads.push_back(g);
    return grads;
}

void ConsistencyStudent::zero_grad() {
    for (auto& blk : enc_blocks_) {
        blk.conv1.zero_grad();
        blk.conv2.zero_grad();
        blk.downsample.zero_grad();
    }
    for (auto& blk : dec_blocks_) {
        blk.upsample.zero_grad();
        blk.conv1.zero_grad();
        blk.conv2.zero_grad();
    }
    for (auto& mlp : time_mlps_) {
        mlp.fc1->zero_grad();
        mlp.fc2->zero_grad();
    }
    mid_conv1_.zero_grad();
    mid_conv2_.zero_grad();
    final_conv_.zero_grad();
    class_proj_->zero_grad();
}

// =====================================================================
// UNetDenoiser
// =====================================================================
UNetDenoiser::UNetDenoiser()
    : last_t_(0.0), last_class_label_(-1), time_embed_(256) {}

void UNetDenoiser::set_condition(double t, int class_label) {
    last_t_ = t;
    last_class_label_ = class_label;
}

Tensor UNetDenoiser::forward(const Tensor& x) const {
    (void)x;
    // Placeholder teacher — returns zeros (teacher not trained in this setup)
    return Tensor(x.rows, x.cols);
}

// =====================================================================
// ConsistencyModel
// =====================================================================
ConsistencyModel::ConsistencyModel(UNetDenoiser* teacher, ConsistencyStudent* student,
                                     int T, int distillage_steps,
                                     int min_step, int max_step)
    : teacher_(teacher), student_(student),
      T_(T), distillage_steps_(distillage_steps),
      min_step_(min_step), max_step_(max_step == -1 ? T - 1 : max_step),
      step_size_(std::max(1, (max_step_ - min_step_ + 1) / distillage_steps_)),
      current_distill_step_(0),
      rng_(42)
{
    build_distillation_schedule();
}

void ConsistencyModel::build_distillation_schedule() {
    distillation_schedule_.clear();
    for (int i = 0; i < distillage_steps_; ++i) {
        int step = min_step_ + i * step_size_;
        distillation_schedule_.push_back(std::min(step, max_step_));
    }
}

double ConsistencyModel::distill(const Tensor& x_t, int t) {
    double t_norm = static_cast<double>(t) / static_cast<double>(T_);
    teacher_->set_condition(t_norm, last_class_label_);
    student_->set_condition(t_norm, last_class_label_);

    // Teacher prediction (no gradient)
    teacher_pred_ = teacher_->forward(x_t);

    // Student prediction
    student_->zero_grad();
    student_pred_ = student_->forward(x_t);

    // L2 consistency loss
    double loss = 0.0;
    for (size_t i = 0; i < student_pred_.data.size(); ++i) {
        double diff = student_pred_.data[i] - teacher_pred_.data[i];
        loss += diff * diff;
    }
    loss /= student_pred_.rows;

    // Backward: compute gradient w.r.t. student output and backprop
    Tensor grad_student(student_pred_.rows, student_pred_.cols);
    for (size_t i = 0; i < student_pred_.data.size(); ++i) {
        grad_student.data[i] = 2.0 * (student_pred_.data[i] - teacher_pred_.data[i]) / student_pred_.rows;
    }
    student_->backward(grad_student, 0.0);

    return loss;
}

Tensor ConsistencyModel::sample(const Tensor& x_t, int t) const {
    double t_norm = static_cast<double>(t) / static_cast<double>(T_);
    student_->set_condition(t_norm, last_class_label_);
    return student_->forward(x_t);
}

Tensor ConsistencyModel::sample_multistep(const Tensor& x_start, int num_steps) {
    // Progressive sampling: denoise from t=T to t=0
    sample_buf_ = x_start;

    for (int step = 0; step < distillage_steps_; ++step) {
        int t = distillation_schedule_[step];
        double t_norm = static_cast<double>(t) / static_cast<double>(T_);
        student_->set_condition(t_norm, last_class_label_);
        sample_buf_ = student_->forward(sample_buf_);

        // Apply consistency correction
        if (step < distillage_steps_ - 1) {
            int t_next = distillation_schedule_[step + 1];
            double t_next_norm = static_cast<double>(t_next) / static_cast<double>(T_);
            student_->set_condition(t_next_norm, last_class_label_);
            sample_buf_ = student_->forward(sample_buf_);
        }
    }
    return sample_buf_;
}

Tensor ConsistencyModel::consistency_correction(const Tensor& x, int t, int t_next) const {
    (void)t; (void)t_next;
    // Placeholder: maps x to x at next timestep
    return x;
}

Tensor ConsistencyModel::sample_with_cfg(const Tensor& x_t, int t, int class_label,
                                          double guidance_scale) {
    (void)guidance_scale;
    double t_norm = static_cast<double>(t) / static_cast<double>(T_);
    student_->set_condition(t_norm, class_label);
    return student_->forward(x_t);
}

Tensor ConsistencyModel::forward(const Tensor& input) {
    return student_->forward(input);
}

Tensor ConsistencyModel::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor();
}

void ConsistencyModel::update_weights(double learning_rate) {
    student_->update_weights(learning_rate);
}

std::vector<Tensor*> ConsistencyModel::parameters() {
    return student_->parameters();
}

std::vector<Tensor*> ConsistencyModel::gradients() {
    return student_->gradients();
}

void ConsistencyModel::zero_grad() {
    student_->zero_grad();
}