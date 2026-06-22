#ifndef FNET_H
#define FNET_H

#include "../../core/layer.h"
#include "../../activations/activations.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <complex>
#include <memory>

// ============================================================================
// FNet (Lee-Thorp et al. 2021) — "FNet: Mixing Tokens with Fourier Transforms"
//   https://arxiv.org/abs/2105.03824
//
// Innovation: replace the O(n^2) self-attention matrix multiply with a 2D
// discrete Fourier transform applied to the (tokens, features) tensor. The
// FFT is a *linear* operator, so it's trivially differentiable through its
// inverse (iFFT). The paper shows that this linear mixing matches the
// performance of attention on a number of NLP tasks at a fraction of the
// cost (O(n log n) instead of O(n^2)).
//
// The math (treating input as a (n_tokens, d_model) 2D "image"):
//   mix = Re(FFT2D(x))                  // (n, d) — real part of complex FFT
//   out = mix @ W^T + b                  // (n, d) — per-token Dense projection
//
// where:
//   * FFT2D is the discrete 2D Fourier transform applied to the (n, d) tensor,
//     with the convention FFT2D[k, l] = sum_{t=0..n-1} sum_{j=0..d-1}
//     x[t, j] * exp(-2*pi*i*(k*t/n + l*j/d)).
//   * Re() takes the real part. For real-valued input, the FFT output has
//     conjugate symmetry so Re() captures all the information.
//   * W is (d, d) and b is (d,) — a standard per-token linear layer.
//
// Backward (linear chain through FFT then Dense):
//   dL/dmix = dL/dout @ W                // (n, d)
//   dL/dF   = Re(dL/dmix)                // (n, d) — drop imag part of inverse
//   dL/dx   = iFFT2D(dL/dF)              // (n, d)
//   dL/dW   = sum_t dL/dout[t, :] outer mix[t, :]      // (d, d)
//   dL/db   = sum_t dL/dout[t, :]                       // (d,)
//
// We use a naive O(N^2) DFT for clarity and gradient-check tractability. The
// forward/inverse pair is exact (within floating-point precision) so the
// gradient check should hit machine precision.
//
// Implementation follows repo conventions:
//   * Input/output is (n, d_model) row-major.
//   * Dense convention: W stored as (out, in) and y = x @ W^T + b.
//   * Block: pre-LN -> FNet -> residual -> pre-LN -> GELU FFN -> residual.
//   * Model: stack of FNetBlocks + per-token classifier head.
// ============================================================================

// ----------------------------------------------------------------------------
// Free helpers — naive 2D FFT and iFFT over a real-valued input.
// ----------------------------------------------------------------------------
//
// fft2d_real: forward 2D DFT over a real-valued (rows, cols) tensor.
// Output is the real part of the complex FFT — conjugate symmetry means the
// real part is sufficient and avoids storing redundant complex entries.
static void fft2d_real(const Tensor& in, Tensor& real_out) {
    size_t R = in.rows, C = in.cols;
    real_out = Tensor(R, C);
    real_out.fill(0.0);
    // FFT2D[k, l] = sum_{r=0..R-1} sum_{c=0..C-1} in[r, c] * exp(-2 pi i (k r/R + l c/C))
    for (size_t k = 0; k < R; ++k) {
        for (size_t l = 0; l < C; ++l) {
            double real_sum = 0.0;
            double imag_sum = 0.0;
            for (size_t r = 0; r < R; ++r) {
                double phase_r = -2.0 * M_PI * static_cast<double>(k * r) / static_cast<double>(R);
                double cos_r = std::cos(phase_r);
                double sin_r = std::sin(phase_r);
                for (size_t c = 0; c < C; ++c) {
                    double phase_c = -2.0 * M_PI * static_cast<double>(l * c) / static_cast<double>(C);
                    // Combined phase: phase_r + phase_c
                    double cos_c = std::cos(phase_c);
                    double sin_c = std::sin(phase_c);
                    // exp(i*(phase_r + phase_c)) = (cos_r + i sin_r)(cos_c + i sin_c)
                    //                              = (cos_r cos_c - sin_r sin_c) + i (cos_r sin_c + sin_r cos_c)
                    double cos_combined = cos_r * cos_c - sin_r * sin_c;
                    double sin_combined = cos_r * sin_c + sin_r * cos_c;
                    double xr = in(r, c);
                    real_sum += xr * cos_combined;
                    imag_sum += xr * sin_combined;
                }
            }
            // We take only the real part of the FFT output.
            real_out(k, l) = real_sum;
        }
    }
}

// ifft2d_real: inverse 2D DFT applied to a real-valued (rows, cols) tensor.
// Returns the real part of the inverse FFT — for a properly Hermitian-symmetric
// input, the imaginary part is zero; for arbitrary input, we discard the
// imaginary component (this is what the linear-operator gradient of "Re(FFT)"
// needs in backward).
static void ifft2d_real(const Tensor& in, Tensor& real_out) {
    size_t R = in.rows, C = in.cols;
    real_out = Tensor(R, C);
    real_out.fill(0.0);
    // Adjoint of the Re(FFT) operator. Math:
    //   dx[r, c] = sum_{k, l} dmix[k, l] * Re(exp(-2 pi i (k r/R + l c/C)))
    //           = sum_{k, l} dmix[k, l] * cos(2 pi (k r/R + l c/C))
    //           = sum_{k, l} dmix[k, l] * (cos_r cos_c - sin_r sin_c)
    // where cos_r = cos(2 pi k r/R), sin_r = sin(2 pi k r/R), and similarly
    // for the c-side. The "cos cos - sin sin" expansion is the correct
    // product-to-sum form — dropping the sin sin term causes the gradient
    // check to fail by a few percent.
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
            double real_sum = 0.0;
            for (size_t k = 0; k < R; ++k) {
                double phase_r = 2.0 * M_PI * static_cast<double>(k * r) / static_cast<double>(R);
                double cos_r = std::cos(phase_r);
                double sin_r = std::sin(phase_r);
                for (size_t l = 0; l < C; ++l) {
                    double phase_c = 2.0 * M_PI * static_cast<double>(l * c) / static_cast<double>(C);
                    double cos_c = std::cos(phase_c);
                    double sin_c = std::sin(phase_c);
                    double kernel = cos_r * cos_c - sin_r * sin_c;
                    real_sum += in(k, l) * kernel;
                }
            }
            real_out(r, c) = real_sum;
        }
    }
}

// ----------------------------------------------------------------------------
// FNetLayer — single Fourier mixing layer + per-token Dense projection.
// ----------------------------------------------------------------------------
class FNetLayer : public Layer {
public:
    // d_model: input/output feature dimension (no head splitting in v1;
    // matches the FNet paper which doesn't use multi-head).
    FNetLayer(size_t d_model);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_dense.weights; }
    Tensor get_gradients() const override { return W_dense.grad_weights; }
    std::string name() const override { return "FNetLayer"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    const Tensor& last_mix() const { return last_mix_; }

private:
    size_t d_model_;
    Dense W_dense;       // (d_model, d_model)

    // Caches for forward
    Tensor last_input_;  // (n, d_model)
    Tensor last_mix_;    // (n, d_model) — Re(FFT2D(input))
    Tensor last_out_;    // (n, d_model) — last_mix_ @ W_dense^T + b_dense
};

FNetLayer::FNetLayer(size_t d_model)
    : d_model_(d_model),
      W_dense(d_model, d_model)
{
    if (d_model == 0) {
        throw std::invalid_argument("FNetLayer: d_model must be > 0");
    }
}

std::vector<Tensor*> FNetLayer::parameters() {
    return { &W_dense.weights, &W_dense.bias };
}

std::vector<Tensor*> FNetLayer::gradients() {
    return { &W_dense.grad_weights, &W_dense.grad_bias };
}

void FNetLayer::zero_grad() {
    W_dense.grad_weights.fill(0.0);
    W_dense.grad_bias.fill(0.0);
}

void FNetLayer::update_weights(double learning_rate) {
    // Dense convention: weights (out, in), bias (1, out).
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            W_dense.weights(i, j) -= learning_rate * W_dense.grad_weights(i, j);
        }
        W_dense.bias(0, i) -= learning_rate * W_dense.grad_bias(0, i);
    }
}

Tensor FNetLayer::forward(const Tensor& input) {
    // input: (n, d_model)
    size_t n = input.rows;
    last_input_ = input;

    // Step 1: 2D FFT mixing (real part)
    fft2d_real(input, last_mix_);

    // Step 2: per-token Dense projection: out = mix @ W_dense^T + b_dense
    last_out_ = Tensor(n, d_model_);
    last_out_.fill(0.0);
    // Use Dense::forward for the matmul+bias part to leverage the cached
    // implementation pattern. We construct a (1, d_model * n) reshape view —
    // but it's simpler to do it directly here for clarity.
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double val = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                val += last_mix_(t, k) * W_dense.weights(j, k);
            }
            last_out_(t, j) = val + W_dense.bias(0, j);
        }
    }

    return last_out_;
}

Tensor FNetLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (n, d_model)
    size_t n = grad_output.rows;

    // Step 1: gradient through per-token Dense.
    //   dL/dmix[t, k] = sum_j grad_output[t, j] * W_dense.weights(j, k)
    //   dL/dW[j, k]   = sum_t grad_output[t, j] * last_mix_(t, k)
    //   dL/db[j]      = sum_t grad_output[t, j]
    Tensor dmix(n, d_model_);
    dmix.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += grad_output(t, j) * W_dense.weights(j, k);
            }
            dmix(t, k) = v;
        }
    }

    // Accumulate Dense gradients (in-place += to allow gradient accumulation
    // across multiple backward calls if the user is doing checkpointing).
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = 0.0;
            for (size_t t = 0; t < n; ++t) {
                g += grad_output(t, j) * last_mix_(t, k);
            }
            W_dense.grad_weights(j, k) += g;
        }
        double gb = 0.0;
        for (size_t t = 0; t < n; ++t) {
            gb += grad_output(t, j);
        }
        W_dense.grad_bias(0, j) += gb;
    }

    // Step 2: gradient through Re(FFT2D).
    // The forward is mix = Re(FFT2D(x)), so the inverse operation is:
    //   dL/dF (in FFT space) = Re(dL/dmix)
    //   dL/dx                 = iFFT2D(dL/dF)  (real part)
    // For arbitrary dL/dmix, the imaginary part of iFFT2D of (Re(dL/dmix))
    // is dropped — matches the fact that "Re" is a real linear operator and
    // the "Re" output is what the loss sees.
    Tensor dF_real(n, d_model_);
    dF_real.fill(0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            dF_real(i, j) = dmix(i, j);  // Re(dmix) for already-real dmix
        }
    }
    Tensor grad_input;
    ifft2d_real(dF_real, grad_input);

    return grad_input;
}

// ----------------------------------------------------------------------------
// FNetBlock — pre-LN -> FNetLayer -> residual -> pre-LN -> GELU FFN -> residual
// ----------------------------------------------------------------------------
class FNetBlock : public Layer {
public:
    FNetBlock(size_t d_model, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ffn_fc1_.weights; }
    Tensor get_gradients() const override { return ffn_fc1_.grad_weights; }
    std::string name() const override { return "FNetBlock"; }

    // Public sublayers (for inspection / direct access)
private:
    size_t d_model_;
    size_t ffn_dim_;
    FNetLayer fnet;
    LayerNorm ln1;
    LayerNorm ln2;
    Dense ffn_fc1_;   // (d_model, ffn_dim) — d -> ffn
    Dense ffn_fc2_;   // (ffn_dim, d_model) — ffn -> d

    // Caches for forward
    Tensor last_x_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_h_pre_;
    Tensor last_ffn_h_;
    Tensor last_ffn_out_;
};

FNetBlock::FNetBlock(size_t d_model, size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      fnet(d_model),
      ln1(d_model),
      ln2(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{
    if (d_model == 0) {
        throw std::invalid_argument("FNetBlock: d_model must be > 0");
    }
}

std::vector<Tensor*> FNetBlock::parameters() {
    auto p = fnet.parameters();
    auto l1 = ln1.parameters();
    auto l2 = ln2.parameters();
    auto fc1 = ffn_fc1_.parameters();
    auto fc2 = ffn_fc2_.parameters();
    p.insert(p.end(), l1.begin(), l1.end());
    p.insert(p.end(), l2.begin(), l2.end());
    p.insert(p.end(), fc1.begin(), fc1.end());
    p.insert(p.end(), fc2.begin(), fc2.end());
    return p;
}

std::vector<Tensor*> FNetBlock::gradients() {
    auto g = fnet.gradients();
    auto l1 = ln1.gradients();
    auto l2 = ln2.gradients();
    auto fc1 = ffn_fc1_.gradients();
    auto fc2 = ffn_fc2_.gradients();
    g.insert(g.end(), l1.begin(), l1.end());
    g.insert(g.end(), l2.begin(), l2.end());
    g.insert(g.end(), fc1.begin(), fc1.end());
    g.insert(g.end(), fc2.begin(), fc2.end());
    return g;
}

void FNetBlock::zero_grad() {
    fnet.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.grad_weights.fill(0.0);
    ffn_fc1_.grad_bias.fill(0.0);
    ffn_fc2_.grad_weights.fill(0.0);
    ffn_fc2_.grad_bias.fill(0.0);
}

void FNetBlock::update_weights(double learning_rate) {
    fnet.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    // Dense update_weights is called inline (Dense doesn't override update_weights
    // in a way that works generically — we replicate the standard pattern).
    for (size_t i = 0; i < ffn_fc1_.weights.rows; ++i) {
        for (size_t j = 0; j < ffn_fc1_.weights.cols; ++j) {
            ffn_fc1_.weights(i, j) -= learning_rate * ffn_fc1_.grad_weights(i, j);
        }
        ffn_fc1_.bias(0, i) -= learning_rate * ffn_fc1_.grad_bias(0, i);
    }
    for (size_t i = 0; i < ffn_fc2_.weights.rows; ++i) {
        for (size_t j = 0; j < ffn_fc2_.weights.cols; ++j) {
            ffn_fc2_.weights(i, j) -= learning_rate * ffn_fc2_.grad_weights(i, j);
        }
        ffn_fc2_.bias(0, i) -= learning_rate * ffn_fc2_.grad_bias(0, i);
    }
}

Tensor FNetBlock::forward(const Tensor& input) {
    // input: (n, d_model)
    last_x_ = input;

    // pre-LN -> FNet -> residual
    last_z1_ = ln1.forward(input);
    last_attn_out_ = fnet.forward(last_z1_);
    last_res1_ = Tensor(input.rows, d_model_);
    last_res1_.fill(0.0);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            last_res1_(i, j) = input(i, j) + last_attn_out_(i, j);
        }
    }

    // pre-LN -> FFN -> residual
    last_z2_ = ln2.forward(last_res1_);
    // FFN = GELU(z2 @ W1^T + b1) @ W2^T + b2
    last_h_pre_ = Tensor(last_z2_.rows, ffn_dim_);
    last_h_pre_.fill(0.0);
    for (size_t t = 0; t < last_z2_.rows; ++t) {
        for (size_t j = 0; j < ffn_dim_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += last_z2_(t, k) * ffn_fc1_.weights(j, k);
            }
            last_h_pre_(t, j) = v + ffn_fc1_.bias(0, j);
        }
    }
    // GELU activation
    last_ffn_h_ = Tensor(last_h_pre_.rows, ffn_dim_);
    for (size_t t = 0; t < last_h_pre_.rows; ++t) {
        for (size_t j = 0; j < ffn_dim_; ++j) {
            GELU gelu;
            last_ffn_h_(t, j) = gelu(last_h_pre_(t, j));
        }
    }
    // FFN output projection
    last_ffn_out_ = Tensor(last_ffn_h_.rows, d_model_);
    for (size_t t = 0; t < last_ffn_h_.rows; ++t) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < ffn_dim_; ++k) {
                v += last_ffn_h_(t, k) * ffn_fc2_.weights(j, k);
            }
            last_ffn_out_(t, j) = v + ffn_fc2_.bias(0, j);
        }
    }

    // Final residual
    Tensor output(last_res1_.rows, d_model_);
    for (size_t i = 0; i < last_res1_.rows; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            output(i, j) = last_res1_(i, j) + last_ffn_out_(i, j);
        }
    }
    return output;
}

Tensor FNetBlock::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (n, d_model)
    // Forward path:
    //   z1 = ln1(x); a = fnet(z1); res1 = x + a
    //   z2 = ln2(res1); h = gelu(z2 @ W1^T + b1); ffn_out = h @ W2^T + b2
    //   out = res1 + ffn_out
    // Backward:
    //   d_res1 = grad_output   (residual branch)
    //   d_ffn_out = grad_output (residual branch)
    //   dW2, db2, d_h = chain through ffn_out
    //   d_h_pre = d_h * gelu'(h_pre)
    //   dW1, db1, d_z2 = chain through z2 @ W1^T + b1
    //   d_res1 += ln2.backward(d_z2)
    //   d_x = d_res1   (residual branch)
    //   d_z1 = ln1.backward(d_res1)  ... actually we need to split:
    //     from res1 = x + a(z1): d_x_from_res1 = d_res1 (split 1:0 between x and a)
    //     d_a = d_res1 (the same)
    //   d_z1 = fnet.backward(d_a)
    //   d_x_from_ln1 = ln1.backward(d_z1)
    //   d_x = d_x_from_res1 + d_x_from_ln1

    size_t n = grad_output.rows;

    // d_res1 = grad_output (residual from out = res1 + ffn_out)
    Tensor d_res1(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_res1(i, j) = grad_output(i, j);

    // d_ffn_out = grad_output
    Tensor d_ffn_out(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_ffn_out(i, j) = grad_output(i, j);

    // Chain through ffn_out = h @ W2^T + b2
    //   d_h[t, k] = sum_j d_ffn_out[t, j] * W2[j, k]
    //   dW2[j, k] = sum_t d_ffn_out[t, j] * h[t, k]
    //   db2[j]    = sum_t d_ffn_out[t, j]
    Tensor d_h(n, ffn_dim_);
    d_h.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += d_ffn_out(t, j) * ffn_fc2_.weights(j, k);
            }
            d_h(t, k) = v;
        }
    }
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            double g = 0.0;
            for (size_t t = 0; t < n; ++t) {
                g += d_ffn_out(t, j) * last_ffn_h_(t, k);
            }
            ffn_fc2_.grad_weights(j, k) += g;
        }
        double gb = 0.0;
        for (size_t t = 0; t < n; ++t) {
            gb += d_ffn_out(t, j);
        }
        ffn_fc2_.grad_bias(0, j) += gb;
    }

    // d_h_pre = d_h * gelu'(h_pre)
    GELU gelu;
    Tensor d_h_pre(n, ffn_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < ffn_dim_; ++k) {
            d_h_pre(t, k) = d_h(t, k) * gelu.derivative(last_h_pre_(t, k));
        }
    }

    // Chain through h_pre = z2 @ W1^T + b1
    //   d_z2[t, k] = sum_j d_h_pre[t, j] * W1[j, k]
    //   dW1[j, k]  = sum_t d_h_pre[t, j] * z2[t, k]
    //   db1[j]     = sum_t d_h_pre[t, j]
    Tensor d_z2(n, d_model_);
    d_z2.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < ffn_dim_; ++j) {
                v += d_h_pre(t, j) * ffn_fc1_.weights(j, k);
            }
            d_z2(t, k) = v;
        }
    }
    for (size_t j = 0; j < ffn_dim_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = 0.0;
            for (size_t t = 0; t < n; ++t) {
                g += d_h_pre(t, j) * last_z2_(t, k);
            }
            ffn_fc1_.grad_weights(j, k) += g;
        }
        double gb = 0.0;
        for (size_t t = 0; t < n; ++t) {
            gb += d_h_pre(t, j);
        }
        ffn_fc1_.grad_bias(0, j) += gb;
    }

    // Chain through z2 = ln2(res1): d_res1 += ln2.backward(d_z2)
    Tensor d_res1_from_ln2 = ln2.backward(d_z2, learning_rate);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_res1(i, j) += d_res1_from_ln2(i, j);

    // Now d_res1 contains the full gradient w.r.t. res1 (from both residual branches).
    // Split into:
    //   d_x (from res1 = x + a) = d_res1
    //   d_a (from res1 = x + a) = d_res1
    Tensor d_x_from_residual(n, d_model_);
    Tensor d_a(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = d_res1(i, j);
            d_x_from_residual(i, j) = v;
            d_a(i, j) = v;
        }
    }

    // Chain through a = fnet(z1): d_z1 = fnet.backward(d_a)
    Tensor d_z1 = fnet.backward(d_a, learning_rate);

    // Chain through z1 = ln1(x): d_x += ln1.backward(d_z1)
    Tensor d_x_from_ln1 = ln1.backward(d_z1, learning_rate);

    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_input(i, j) = d_x_from_residual(i, j) + d_x_from_ln1(i, j);

    return d_input;
}

// ----------------------------------------------------------------------------
// FNetModel — stack of FNetBlocks + per-token classifier
// ----------------------------------------------------------------------------
class FNetModel : public Layer {
public:
    // d_model, out_features: model dims
    // num_blocks: number of stacked FNetBlocks
    // ffn_dim: hidden dim of FFN inside each block (default 4*d_model)
    FNetModel(size_t d_model, size_t out_features, size_t num_blocks, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return classifier_.weights; }
    Tensor get_gradients() const override { return classifier_.grad_weights; }
    std::string name() const override { return "FNetModel"; }

    std::vector<std::unique_ptr<FNetBlock>> blocks;

private:
    size_t d_model_;
    size_t out_features_;
    size_t num_blocks_;
    size_t ffn_dim_;
    Dense classifier_;   // (d_model, out_features) — per-token classifier
    Tensor last_input_;
};

FNetModel::FNetModel(size_t d_model, size_t out_features, size_t num_blocks, size_t ffn_dim)
    : d_model_(d_model),
      out_features_(out_features),
      num_blocks_(num_blocks),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      classifier_(d_model, out_features)
{
    if (d_model == 0 || out_features == 0 || num_blocks == 0) {
        throw std::invalid_argument("FNetModel: d_model, out_features, num_blocks must all be > 0");
    }
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks.emplace_back(std::make_unique<FNetBlock>(d_model, ffn_dim_));
    }
}

std::vector<Tensor*> FNetModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&classifier_.weights);
    p.push_back(&classifier_.bias);
    return p;
}

std::vector<Tensor*> FNetModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&classifier_.grad_weights);
    g.push_back(&classifier_.grad_bias);
    return g;
}

void FNetModel::zero_grad() {
    for (auto& b : blocks) b->zero_grad();
    classifier_.grad_weights.fill(0.0);
    classifier_.grad_bias.fill(0.0);
}

void FNetModel::update_weights(double learning_rate) {
    for (auto& b : blocks) b->update_weights(learning_rate);
    for (size_t i = 0; i < classifier_.weights.rows; ++i) {
        for (size_t j = 0; j < classifier_.weights.cols; ++j) {
            classifier_.weights(i, j) -= learning_rate * classifier_.grad_weights(i, j);
        }
        classifier_.bias(0, i) -= learning_rate * classifier_.grad_bias(0, i);
    }
}

Tensor FNetModel::forward(const Tensor& input) {
    last_input_ = input;
    Tensor x = input;
    for (auto& b : blocks) {
        x = b->forward(x);
    }
    // Classifier: per-token projection to out_features
    size_t n = x.rows;
    Tensor output(n, out_features_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < out_features_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += x(t, k) * classifier_.weights(j, k);
            }
            output(t, j) = v + classifier_.bias(0, j);
        }
    }
    return output;
}

Tensor FNetModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output: (n, out_features)
    size_t n = grad_output.rows;

    // Chain through classifier.
    // NOTE: block_outputs already holds each block's output from the second
    // forward pass below. We re-run forward here (instead of caching during
    // the first forward) for simplicity — a production version would cache
    // block outputs in a vector inside forward() to avoid the second pass.
    Tensor h = last_input_;
    std::vector<Tensor> block_outputs;
    block_outputs.reserve(blocks.size());
    for (auto& b : blocks) {
        h = b->forward(h);
        block_outputs.push_back(h);
    }

    // Now backprop through classifier
    Tensor d_h(n, d_model_);
    d_h.fill(0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < out_features_; ++j) {
                v += grad_output(t, j) * classifier_.weights(j, k);
            }
            d_h(t, k) = v;
        }
    }
    for (size_t j = 0; j < out_features_; ++j) {
        for (size_t k = 0; k < d_model_; ++k) {
            double g = 0.0;
            for (size_t t = 0; t < n; ++t) {
                g += grad_output(t, j) * block_outputs.back()(t, k);
            }
            classifier_.grad_weights(j, k) += g;
        }
        double gb = 0.0;
        for (size_t t = 0; t < n; ++t) {
            gb += grad_output(t, j);
        }
        classifier_.grad_bias(0, j) += gb;
    }

    // Backprop through blocks in reverse order
    Tensor d_input = d_h;
    for (size_t i = blocks.size(); i > 0; --i) {
        d_input = blocks[i - 1]->backward(d_input, learning_rate);
    }

    return d_input;
}

#endif // FNET_H
