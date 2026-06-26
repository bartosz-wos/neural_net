#include "capsnet.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Local helpers
// ============================================================================
namespace {

inline double stable_norm(double ss, double eps) {
    return std::sqrt(ss + eps);
}

// Softmax over the j (output-capsule) axis of a (B, I*J) tensor.
//   c[b, i, j] = exp(b[b, i, j]) / sum_{j'} exp(b[b, i, j'])
// We treat the input as B*I independent softmax problems of width J.
inline void softmax_over_j(const Tensor& b, Tensor& c, size_t B, size_t I, size_t J) {
    for (size_t bb = 0; bb < B; ++bb) {
        for (size_t ii = 0; ii < I; ++ii) {
            // Find max for numerical stability.
            double m = -1e30;
            for (size_t j = 0; j < J; ++j) {
                double v = b(bb, ii * J + j);
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (size_t j = 0; j < J; ++j) {
                double v = std::exp(b(bb, ii * J + j) - m);
                c(bb, ii * J + j) = v;
                sum += v;
            }
            double inv = 1.0 / (sum + 1e-30);
            for (size_t j = 0; j < J; ++j) {
                c(bb, ii * J + j) *= inv;
            }
        }
    }
}

// Apply squash to each (b, j) slice of an (B, J*D) tensor. Output written in-place to v.
//   squash(s)[k] = s[k] * ||s|| / (1 + ||s||²)
// where ||s||² = sum_k s[k]² (just the k-axis — i.e. the dim_capsule axis).
inline void squash_inplace(const Tensor& s, Tensor& v, size_t B, size_t J, size_t D, double eps) {
    for (size_t bb = 0; bb < B; ++bb) {
        for (size_t j = 0; j < J; ++j) {
            double ss = 0.0;
            for (size_t k = 0; k < D; ++k) {
                double x = s(bb, j * D + k);
                ss += x * x;
            }
            double n = stable_norm(ss, eps);
            double scale = n / (1.0 + ss);
            for (size_t k = 0; k < D; ++k) {
                v(bb, j * D + k) = s(bb, j * D + k) * scale;
            }
        }
    }
}

}  // namespace

// ============================================================================
// CapsuleLayer
// ============================================================================

CapsuleLayer::CapsuleLayer(size_t num_input_capsules, size_t input_capsule_dim,
                           size_t num_capsules, size_t dim_capsule,
                           size_t num_routing)
    : num_input_capsules_(num_input_capsules),
      input_capsule_dim_(input_capsule_dim),
      num_capsules_(num_capsules),
      dim_capsule_(dim_capsule),
      num_routing_(num_routing == 0 ? 1 : num_routing) {

    if (num_input_capsules == 0 || input_capsule_dim == 0 ||
        num_capsules == 0 || dim_capsule == 0) {
        throw std::invalid_argument("CapsuleLayer: dims must be > 0");
    }

    // Xavier-uniform init on each W[j] of shape (D_in, D).
    std::mt19937 gen(42);
    double bound = std::sqrt(6.0 / static_cast<double>(input_capsule_dim_ + dim_capsule_));
    std::uniform_real_distribution<double> dis(-bound, bound);

    W_.resize(num_capsules_);
    grad_W_.resize(num_capsules_);
    for (size_t j = 0; j < num_capsules_; ++j) {
        W_[j] = Tensor(input_capsule_dim_, dim_capsule_);
        grad_W_[j] = Tensor(input_capsule_dim_, dim_capsule_);
        for (size_t i = 0; i < input_capsule_dim_; ++i)
            for (size_t k = 0; k < dim_capsule_; ++k)
                W_[j](i, k) = dis(gen);
    }

    // Allocate caches
    iter_s_.resize(num_routing_);
    iter_v_.resize(num_routing_);
    iter_c_.resize(num_routing_);
    for (size_t r = 0; r < num_routing_; ++r) {
        iter_s_[r] = Tensor(0, 0);
        iter_v_[r] = Tensor(0, 0);
        iter_c_[r] = Tensor(0, 0);
    }
    last_u_hat_ = Tensor(0, 0);
    last_c_ = Tensor(0, 0);
    last_input_ = Tensor(0, 0);
}

Tensor CapsuleLayer::forward(const Tensor& input) {
    const size_t B = input.rows;
    const size_t I = num_input_capsules_;
    const size_t D_in = input_capsule_dim_;
    const size_t J = num_capsules_;
    const size_t D = dim_capsule_;
    const size_t R = num_routing_;

    if (input.cols != I * D_in) {
        throw std::invalid_argument(
            "CapsuleLayer::forward: input.cols must equal num_input_capsules * input_capsule_dim");
    }

    last_input_ = input;

    // 1) Predictions: u_hat[b, j, i, k] = sum_{d_in} W_[j][d_in, k] · u[b, i, d_in]
    // Flatten to (B, J * I * D), indexed as [b * (J*I*D) + j*(I*D) + i*D + k].
    Tensor u_hat(B, J * I * D);
    for (size_t bb = 0; bb < B; ++bb) {
        for (size_t j = 0; j < J; ++j) {
            const Tensor& Wj = W_[j];
            for (size_t i = 0; i < I; ++i) {
                for (size_t k = 0; k < D; ++k) {
                    double sum = 0.0;
                    for (size_t d = 0; d < D_in; ++d) {
                        sum += input(bb, i * D_in + d) * Wj(d, k);
                    }
                    u_hat(bb, j * I * D + i * D + k) = sum;
                }
            }
        }
    }
    last_u_hat_ = u_hat;

    // 2) Routing loop
    Tensor b(B, I * J);          // routing logits
    Tensor c(B, I * J);          // couplings
    Tensor s(B, J * D);          // pre-squash
    Tensor v(B, J * D);          // post-squash (output of each iter)

    for (size_t r = 0; r < R; ++r) {
        // c_r = softmax_j(b_r)
        softmax_over_j(b, c, B, I, J);
        iter_c_[r] = c.clone();

        // s_r[b, j, k] = sum_i c_r[b, i, j] · u_hat[b, j, i, k]
        for (size_t bb = 0; bb < B; ++bb) {
            for (size_t j = 0; j < J; ++j) {
                for (size_t k = 0; k < D; ++k) {
                    double sum = 0.0;
                    for (size_t i = 0; i < I; ++i) {
                        sum += c(bb, i * J + j) * u_hat(bb, j * I * D + i * D + k);
                    }
                    s(bb, j * D + k) = sum;
                }
            }
        }
        iter_s_[r] = s.clone();

        // v_r = squash(s_r)
        squash_inplace(s, v, B, J, D, epsilon_);
        iter_v_[r] = v.clone();

        // Update b_{r+1} = b_r + sum_k u_hat · v_r   (only if r < R-1)
        if (r + 1 < R) {
            for (size_t bb = 0; bb < B; ++bb) {
                for (size_t i = 0; i < I; ++i) {
                    for (size_t j = 0; j < J; ++j) {
                        double agree = 0.0;
                        for (size_t k = 0; k < D; ++k) {
                            agree += u_hat(bb, j * I * D + i * D + k)
                                   * v(bb, j * D + k);
                        }
                        b(bb, i * J + j) += agree;
                    }
                }
            }
        }
    }

    last_c_ = c;     // final couplings (post last softmax)
    return v;        // (B, J*D)
}

Tensor CapsuleLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t B = last_input_.rows;
    const size_t I = num_input_capsules_;
    const size_t D_in = input_capsule_dim_;
    const size_t J = num_capsules_;
    const size_t D = dim_capsule_;
    const size_t R = num_routing_;

    if (grad_output.rows != B || grad_output.cols != J * D) {
        throw std::invalid_argument(
            "CapsuleLayer::backward: grad_output shape mismatch");
    }
    if (last_u_hat_.rows == 0) {
        throw std::logic_error("CapsuleLayer::backward called before forward");
    }

    // Reset parameter gradients (backward is a fresh accumulation).
    for (size_t j = 0; j < J; ++j) grad_W_[j].fill(0.0);

    // d u_hat accumulator, shape (B, J*I*D). Accumulates contributions from
    // (a) s_r = sum_i c_r · u_hat (the coupling-s-weighted path), and
    // (b) agreement_{r-1} = sum_k u_hat · v_{r-1} (the agreement update path).
    Tensor d_u_hat(B, J * I * D);
    d_u_hat.fill(0.0);

    Tensor d_v(B, J * D);
    Tensor d_s(B, J * D);
    Tensor d_c(B, I * J);
    Tensor d_b(B, I * J);   // carries d_loss/d_b_r across iterations
    Tensor d_input(B, I * D_in);

    // Seed d_v from grad_output for the last iteration (r = R-1).
    for (size_t i = 0; i < B * J * D; ++i) d_v.data[i] = grad_output.data[i];

    // d_b starts at 0 (d_loss/d_b_R = 0 since b_R is a leaf in the forward graph).
    d_b.fill(0.0);

    for (int r = static_cast<int>(R) - 1; r >= 0; --r) {
        const Tensor& s_r = iter_s_[r];
        const Tensor& c_r = iter_c_[r];

        // ---- 1. squash backward: d_s[b, j, k] from d_v[b, j, :] via Jacobian ----
        //   d v_k / d s_m = δ_{km} * norm/(1+ss) + s_k * s_m * (1-ss) / (norm * (1+ss)²)
        for (size_t bb = 0; bb < B; ++bb) {
            for (size_t j = 0; j < J; ++j) {
                double ss = 0.0;
                for (size_t k = 0; k < D; ++k) {
                    double x = s_r(bb, j * D + k);
                    ss += x * x;
                }
                double n = stable_norm(ss, epsilon_);
                double a = n / (1.0 + ss);                                  // δ-coefficient
                double b_coef = (1.0 - ss) / (n * (1.0 + ss) * (1.0 + ss)); // s_k s_m coefficient
                for (size_t k = 0; k < D; ++k) {
                    double ds = a * d_v(bb, j * D + k);
                    for (size_t m = 0; m < D; ++m) {
                        ds += s_r(bb, j * D + k) * s_r(bb, j * D + m)
                            * b_coef * d_v(bb, j * D + m);
                    }
                    d_s(bb, j * D + k) = ds;
                }
            }
        }

        // ---- 2. d_c[b, i, j] from d_s[b, j, k] via u_hat ----
        //   s_r[b, j, k] = sum_i c_r[b, i, j] · u_hat[b, j, i, k]
        d_c.fill(0.0);
        for (size_t bb = 0; bb < B; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                for (size_t j = 0; j < J; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < D; ++k) {
                        sum += last_u_hat_(bb, j * I * D + i * D + k)
                             * d_s(bb, j * D + k);
                    }
                    d_c(bb, i * J + j) = sum;
                }
            }
        }

        // ---- 3. accumulate d_u_hat from s_r's u_hat-dependence (c_r path) ----
        //   d u_hat[b, j, i, k] += c_r[b, i, j] · d_s[b, j, k]
        for (size_t bb = 0; bb < B; ++bb) {
            for (size_t j = 0; j < J; ++j) {
                for (size_t i = 0; i < I; ++i) {
                    double cij = c_r(bb, i * J + j);
                    for (size_t k = 0; k < D; ++k) {
                        d_u_hat(bb, j * I * D + i * D + k) += cij * d_s(bb, j * D + k);
                    }
                }
            }
        }

        // ---- 4. softmax inverse: d_loss/d_b_r (additive onto d_b carry) ----
        //   c_r[b, i, j] = softmax_j(b_r[b, i, :])
        //   d_b[j] += c_r[j] · (d_c[j] - dot),  dot = sum_j c_r[j]·d_c[j]
        // We ADD onto d_b (which already holds the d_loss/d_b_{r+1} pass-through).
        for (size_t bb = 0; bb < B; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                double dot = 0.0;
                for (size_t j = 0; j < J; ++j) {
                    dot += c_r(bb, i * J + j) * d_c(bb, i * J + j);
                }
                for (size_t j = 0; j < J; ++j) {
                    d_b(bb, i * J + j) += c_r(bb, i * J + j) * (d_c(bb, i * J + j) - dot);
                }
            }
        }

        // ---- 5. chain backward to iter r-1 via agreement update ----
        //   b_r = b_{r-1} + agreement_{r-1},  agreement_{r-1}[b, i, j] = sum_k u_hat · v_{r-1}
        //   d v_{r-1}[b, j, k] += sum_i u_hat[b, j, i, k] · d_b_r[b, i, j]   (v-side)
        //   d u_hat[b, j, i, k] += v_{r-1}[b, j, k] · d_b_r[b, i, j]        (u_hat-side)
        // Only do this if r > 0 — there's no agreement at "iter -1".
        if (r > 0) {
            const Tensor& v_prev = iter_v_[r - 1];
            // (a) accumulate d_u_hat from agreement's u_hat-dependence.
            for (size_t bb = 0; bb < B; ++bb) {
                for (size_t i = 0; i < I; ++i) {
                    for (size_t j = 0; j < J; ++j) {
                        double dbi = d_b(bb, i * J + j);
                        for (size_t k = 0; k < D; ++k) {
                            d_u_hat(bb, j * I * D + i * D + k) += v_prev(bb, j * D + k) * dbi;
                        }
                    }
                }
            }
            // (b) compute d_v_{r-1} from agreement's v-dependence (and carry forward
            //     d_b to the next backward iteration, since d_b_r feeds d_b_{r-1}
            //     via the additive identity — but we'll compute the next iteration's
            //     softmax inverse on top of the EXISTING d_b values, which already
            //     include this pass-through thanks to step 4 being additive).
            d_v.fill(0.0);
            for (size_t bb = 0; bb < B; ++bb) {
                for (size_t j = 0; j < J; ++j) {
                    for (size_t k = 0; k < D; ++k) {
                        double sum = 0.0;
                        for (size_t i = 0; i < I; ++i) {
                            sum += last_u_hat_(bb, j * I * D + i * D + k)
                                 * d_b(bb, i * J + j);
                        }
                        d_v(bb, j * D + k) = sum;
                    }
                }
            }
        }
        // If r == 0, the d_b accumulated at this iter will NOT propagate further
        // (there's no b_{-1}). It's still correct because d_b_0 carries the full
        // contribution to b_0 — and b_0 = 0 is a leaf constant, so d_loss/d_b_0
        // is not used to update any parameter.
    }

    // 3) Accumulate d_W_[j][d_in, k] += sum_{b, i} u[b, i, d_in] · d_u_hat[b, j, i, k]
    //    d_input[b, i, d_in] += sum_{j, k} W_[j][d_in, k] · d_u_hat[b, j, i, k]
    d_input.fill(0.0);
    for (size_t bb = 0; bb < B; ++bb) {
        for (size_t j = 0; j < J; ++j) {
            Tensor& gW = grad_W_[j];
            for (size_t i = 0; i < I; ++i) {
                for (size_t k = 0; k < D; ++k) {
                    double duh = d_u_hat(bb, j * I * D + i * D + k);
                    for (size_t d = 0; d < D_in; ++d) {
                        gW(d, k) += last_input_(bb, i * D_in + d) * duh;
                        d_input(bb, i * D_in + d) += W_[j](d, k) * duh;
                    }
                }
            }
        }
    }

    return d_input;
}

void CapsuleLayer::update_weights(double learning_rate) {
    for (size_t j = 0; j < num_capsules_; ++j) {
        Tensor& Wj = W_[j];
        const Tensor& gWj = grad_W_[j];
        for (size_t i = 0; i < Wj.rows; ++i)
            for (size_t k = 0; k < Wj.cols; ++k)
                Wj(i, k) -= learning_rate * gWj(i, k);
    }
}

void CapsuleLayer::zero_grad() {
    for (auto& gW : grad_W_) gW.fill(0.0);
}

std::vector<Tensor*> CapsuleLayer::parameters() {
    std::vector<Tensor*> out;
    out.reserve(W_.size());
    for (auto& W : W_) out.push_back(&W);
    return out;
}

std::vector<Tensor*> CapsuleLayer::gradients() {
    std::vector<Tensor*> out;
    out.reserve(grad_W_.size());
    for (auto& gW : grad_W_) out.push_back(&gW);
    return out;
}

Tensor CapsuleLayer::get_weights() const {
    // Return the first W (most layers have just one if J=1); for J>1, return
    // a stack along rows. Caller should know what to do.
    if (W_.empty()) return Tensor();
    Tensor out(W_[0].rows * W_.size(), W_[0].cols);
    for (size_t j = 0; j < W_.size(); ++j)
        for (size_t i = 0; i < W_[j].rows; ++i)
            for (size_t k = 0; k < W_[j].cols; ++k)
                out(j * W_[0].rows + i, k) = W_[j](i, k);
    return out;
}

Tensor CapsuleLayer::get_gradients() const {
    if (grad_W_.empty()) return Tensor();
    Tensor out(grad_W_[0].rows * grad_W_.size(), grad_W_[0].cols);
    for (size_t j = 0; j < grad_W_.size(); ++j)
        for (size_t i = 0; i < grad_W_[j].rows; ++i)
            for (size_t k = 0; k < grad_W_[j].cols; ++k)
                out(j * grad_W_[0].rows + i, k) = grad_W_[j](i, k);
    return out;
}

// ============================================================================
// CapsNet — kept for API compatibility. Forward works (no BPTT through
// the routing + decoder). backward() returns Tensor(1, 1) for safety.
// ============================================================================

CapsNet::CapsNet(size_t input_channels, size_t H, size_t W,
                 size_t num_classes, size_t dim_capsule,
                 size_t primary_dim, size_t primary_channels,
                 size_t num_routing)
    : primary_caps_fc_(primary_channels, primary_dim * 32),
      digit_caps_(primary_dim * 32, num_classes, dim_capsule, num_routing),
      fc1_(num_classes * dim_capsule, 512),
      fc2_(512, 1024),
      fc3_(1024, H * W * input_channels),
      last_capsule_output_(1, 1), dim_capsule_(dim_capsule) {}

Tensor CapsNet::forward(const Tensor& input) {
    last_input_ = input;
    size_t batch = input.rows;

    Tensor x = primary_caps_fc_.forward(input);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            double v = x[i][j];
            double norm_sq = v * v;
            x[i][j] = (norm_sq / (1.0 + norm_sq)) * v / (std::sqrt(norm_sq) + 1e-8);
        }
    }

    last_capsule_output_ = digit_caps_.forward(x);

    size_t dim_caps = digit_caps_.dim_capsule();
    Tensor lengths(batch, last_capsule_output_.cols / dim_caps);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < lengths.cols; ++c) {
            double norm_sq = 0.0;
            for (size_t k = 0; k < dim_caps; ++k)
                norm_sq += last_capsule_output_[b][c * dim_caps + k]
                         * last_capsule_output_[b][c * dim_caps + k];
            lengths[b][c] = std::sqrt(norm_sq);
        }
    }

    return lengths;
}

Tensor CapsNet::backward(const Tensor& grad_output, double learning_rate) {
    (void)grad_output; (void)learning_rate;
    return Tensor(1, 1);
}

void CapsNet::update_weights(double learning_rate) {
    primary_caps_fc_.update_weights(learning_rate);
    digit_caps_.update_weights(learning_rate);
    fc1_.update_weights(learning_rate);
    fc2_.update_weights(learning_rate);
    fc3_.update_weights(learning_rate);
}

void CapsNet::zero_grad() {
    primary_caps_fc_.zero_grad();
    digit_caps_.zero_grad();
    fc1_.zero_grad();
    fc2_.zero_grad();
    fc3_.zero_grad();
}

std::vector<Tensor*> CapsNet::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* p : primary_caps_fc_.parameters()) result.push_back(p);
    for (Tensor* p : digit_caps_.parameters()) result.push_back(p);
    for (Tensor* p : fc1_.parameters()) result.push_back(p);
    for (Tensor* p : fc2_.parameters()) result.push_back(p);
    for (Tensor* p : fc3_.parameters()) result.push_back(p);
    return result;
}

std::vector<Tensor*> CapsNet::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* g : primary_caps_fc_.gradients()) result.push_back(g);
    for (Tensor* g : digit_caps_.gradients()) result.push_back(g);
    for (Tensor* g : fc1_.gradients()) result.push_back(g);
    for (Tensor* g : fc2_.gradients()) result.push_back(g);
    for (Tensor* g : fc3_.gradients()) result.push_back(g);
    return result;
}

double CapsNet::reconstruction_loss(const Tensor& input,
                                     const Tensor& digit_capsules,
                                     size_t correct_label) {
    (void)input; (void)digit_capsules; (void)correct_label;
    return 0.0;
}