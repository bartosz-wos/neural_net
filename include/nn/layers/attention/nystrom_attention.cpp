#include "nystrom_attention.h"
#include "../../activations/activations.h"
#include <algorithm>
#include <stdexcept>

NystromAttention::NystromAttention(int embed_dim, int num_heads, int num_landmarks, float dropout)
    : embed_dim_(embed_dim), num_heads_(num_heads),
      num_landmarks_(num_landmarks), head_dim_(embed_dim / num_heads),
      dropout_(dropout), scale_(1.0f / std::sqrt(static_cast<float>(head_dim_) + 1e-6f)),
      is_initialized_(false), batch_size_(0), seq_len_(0)
{
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }

    W_q = Tensor::random(embed_dim_, embed_dim_, 0.01f);
    W_k = Tensor::random(embed_dim_, embed_dim_, 0.01f);
    W_v = Tensor::random(embed_dim_, embed_dim_, 0.01f);
    W_o = Tensor::random(embed_dim_, embed_dim_, 0.01f);

    grad_W_q = Tensor::zeros(embed_dim_, embed_dim_);
    grad_W_k = Tensor::zeros(embed_dim_, embed_dim_);
    grad_W_v = Tensor::zeros(embed_dim_, embed_dim_);
    grad_W_o = Tensor::zeros(embed_dim_, embed_dim_);
}

std::vector<Tensor*> NystromAttention::parameters() {
    return { &W_q, &W_k, &W_v, &W_o };
}

std::vector<Tensor*> NystromAttention::gradients() {
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o };
}

void NystromAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    // Clear cached inputs
    last_query = Tensor();
    last_key = Tensor();
    last_value = Tensor();
}

void NystromAttention::update_weights(double learning_rate) {
    for (size_t i = 0; i < grad_W_q.rows; ++i) {
        for (size_t j = 0; j < grad_W_q.cols; ++j) {
            W_q[i][j] -= learning_rate * grad_W_q[i][j];
            W_k[i][j] -= learning_rate * grad_W_k[i][j];
            W_v[i][j] -= learning_rate * grad_W_v[i][j];
            W_o[i][j] -= learning_rate * grad_W_o[i][j];
        }
    }
}

void NystromAttention::print() const {
    printf("NystromAttention[embed_dim=%d, heads=%d, num_landmarks=%d, head_dim=%d]\n",
           embed_dim_, num_heads_, num_landmarks_, head_dim_);
}

// Softmax over last axis (row-wise)
Tensor NystromAttention::softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double max_val = x[i][0];
        for (size_t j = 1; j < x.cols; ++j)
            if (x[i][j] > max_val) max_val = x[i][j];
        double sum_exp = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            result[i][j] = std::exp(x[i][j] - max_val);
            sum_exp += result[i][j];
        }
        sum_exp = std::max(sum_exp, 1e-300);
        for (size_t j = 0; j < x.cols; ++j)
            result[i][j] /= sum_exp;
    }
    return result;
}

// LU decomposition with partial pivoting, in-place on A (n x n)
// Returns pivot array
static std::vector<size_t> lu_decompose(Tensor& A, size_t n) {
    std::vector<size_t> pivot(n);
    for (size_t i = 0; i < n; ++i) pivot[i] = i;

    for (size_t col = 0; col < n; ++col) {
        size_t max_row = col;
        double max_val = std::abs(A[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            double v = std::abs(A[row][col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }

        if (max_val < 1e-12) {
            // Singular matrix - add small regularization to diagonal
            A[col][col] += 1e-8;
            max_val = std::abs(A[col][col]);
        }

        if (max_row != col) {
            std::swap(pivot[col], pivot[max_row]);
            for (size_t k = 0; k < n; ++k)
                std::swap(A[col][k], A[max_row][k]);
        }

        for (size_t row = col + 1; row < n; ++row) {
            double factor = A[row][col] / A[col][col];
            A[row][col] = factor;
            for (size_t k = col + 1; k < n; ++k)
                A[row][k] -= factor * A[col][k];
        }
    }
    return pivot;
}

// Solve A @ X = B via LU for one head: A (m, m), B (m, n), returns X (m, n)
static Tensor lu_solve_head(const Tensor& A_tilde, const Tensor& A_bar) {
    size_t m = A_tilde.rows;
    size_t n = A_bar.cols;

    Tensor P(m, n);

    Tensor A_lu = A_tilde.clone();
    std::vector<size_t> pivot = lu_decompose(A_lu, m);

    std::vector<double> y(m), x(m);

    for (size_t col = 0; col < n; ++col) {
        std::vector<double> b_rhs(m);
        for (size_t i = 0; i < m; ++i) b_rhs[i] = A_bar[i][col];

        // Forward substitution: y = P @ b_rhs (apply permutation in-place)
        for (size_t i = 0; i < m; ++i)
            y[i] = b_rhs[pivot[i]];
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < i; ++j)
                y[i] -= A_lu[i][j] * y[j];

        // Backward substitution
        for (int i = (int)m - 1; i >= 0; --i) {
            x[i] = y[i];
            for (size_t j = (size_t)i + 1; j < m; ++j)
                x[i] -= A_lu[i][j] * x[j];
            x[i] /= A_lu[i][i];
        }

        for (size_t i = 0; i < m; ++i) P[i][col] = x[i];
    }

    return P;
}

Tensor NystromAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
    // Input shapes: (batch, seq, embed_dim) — stored as (batch, seq * embed_dim) flat
    size_t batch = query.rows;
    size_t seq = query.cols / embed_dim_;  // actual sequence length
    size_t E = embed_dim_;

    batch_size_ = batch;
    seq_len_ = seq;

    // Determine number of landmarks
    int num_lm = num_landmarks_;
    if (num_lm == 0) num_lm = static_cast<int>(std::max<size_t>(1, seq / 4));
    num_lm = std::min(num_lm, static_cast<int>(seq));

    // Landmark indices: linspace(0, seq-1, num_lm)
    landmark_indices_.resize(num_lm);
    for (int i = 0; i < num_lm; ++i) {
        if (num_lm == 1) {
            landmark_indices_[i] = 0;
        } else {
            landmark_indices_[i] = static_cast<size_t>(std::round(
                static_cast<double>(i) / (num_lm - 1) * (seq - 1)));
        }
    }

    // ============================================================
    // Fall back to regular softmax attention if num_lm >= seq
    // ============================================================
    if (num_lm >= static_cast<int>(seq)) {
        // Project Q, K, V: each (batch, seq * E)
        Tensor Q = Tensor::zeros(batch, seq * E);
        Tensor K = Tensor::zeros(batch, seq * E);
        Tensor V = Tensor::zeros(batch, seq * E);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t i = 0; i < E; ++i) {
                    double qv = 0.0, kv = 0.0, vv = 0.0;
                    for (size_t k = 0; k < E; ++k) {
                        qv += query[b][k] * W_q[k][i];
                        kv += key[b][k] * W_k[k][i];
                        vv += value[b][k] * W_v[k][i];
                    }
                    Q[b][s * E + i] = qv;
                    K[b][s * E + i] = kv;
                    V[b][s * E + i] = vv;
                }
            }
        }
        last_q = Q; last_k = K; last_v = V;

        // Per-head split: (batch, seq, num_heads, head_dim)
        Tensor output_accum = Tensor::zeros(batch, seq * E);
        last_output_accum = output_accum;

        // Cache shapes
        size_t m = num_lm;
        size_t H = num_heads_;
        size_t d = head_dim_;
        size_t N = seq;

        last_A_bar = Tensor(batch * H, m * N);
        last_A_tilde = Tensor(batch * H, m * m);
        last_P = Tensor(batch * H, m * N);
        last_landmark_v = Tensor(batch * H, m * d);

        for (size_t h = 0; h < H; ++h) {
            // Per-head Q_h, K_h, V_h: extract from Q,K,V
            // Q[b][s*E+h*d+dk] = Q_h[b][s*d+dk]
            for (size_t b = 0; b < batch; ++b) {
                // S = Q_h @ K_h^T: (seq, seq)
                Tensor S = Tensor::zeros(seq, seq);
                for (size_t i = 0; i < seq; ++i) {
                    for (size_t j = 0; j < seq; ++j) {
                        double s = 0.0;
                        for (size_t dk = 0; dk < d; ++dk)
                            s += Q[b][i * E + h * d + dk] * K[b][j * E + h * d + dk];
                        S[i][j] = s * scale_;
                    }
                }
                // softmax over last axis
                Tensor A = softmax(S);
                // output = A @ V_h: (seq, d)
                for (size_t i = 0; i < seq; ++i) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double v = 0.0;
                        for (size_t j = 0; j < seq; ++j)
                            v += A[i][j] * V[b][j * E + h * d + dk];
                        output_accum[b][i * E + h * d + dk] += v;
                    }
                }
            }
        }

        last_output_accum = output_accum;

        // Final output projection
        Tensor output = Tensor::zeros(batch, seq * E);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t j = 0; j < E; ++j) {
                    double v = 0.0;
                    for (size_t k = 0; k < E; ++k)
                        v += output_accum[b][s * E + k] * W_o[k][j];
                    output[b][s * E + j] = v;
                }
            }
        }
        last_query = query;
        last_key = key;
        last_value = value;
        return output;
    }

    // ============================================================
    // Nyström attention path
    // ============================================================

    // Cache original inputs for correct gradient computation
    last_query = query.clone();
    last_key = key.clone();
    last_value = value.clone();

    // Project Q, K, V: each (batch, seq * E)
    Tensor Q = Tensor::zeros(batch, seq * E);
    Tensor K = Tensor::zeros(batch, seq * E);
    Tensor V = Tensor::zeros(batch, seq * E);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t i = 0; i < E; ++i) {
                double qv = 0.0, kv = 0.0, vv = 0.0;
                for (size_t k = 0; k < E; ++k) {
                    qv += query[b][k] * W_q[k][i];
                    kv += key[b][k] * W_k[k][i];
                    vv += value[b][k] * W_v[k][i];
                }
                Q[b][s * E + i] = qv;
                K[b][s * E + i] = kv;
                V[b][s * E + i] = vv;
            }
        }
    }
    last_q = Q; last_k = K; last_v = V;

    size_t m = landmark_indices_.size();
    size_t H = num_heads_;
    size_t d = head_dim_;
    size_t N = seq;

    // Cache for backward: flat per-head layout
    // last_A_bar: (batch*H, m*N) — row h: A_bar[p][j] = flat[h*m*N + p*N + j]
    last_A_bar = Tensor(batch * H, m * N);
    last_A_tilde = Tensor(batch * H, m * m);
    last_P = Tensor(batch * H, m * N);
    // last_landmark_v: (batch*H, m*d)
    last_landmark_v = Tensor(batch * H, m * d);

    // Accumulator for attention output: (batch, seq * E)
    Tensor output_accum = Tensor::zeros(batch, seq * E);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < H; ++h) {
            // Per-head Q_h, K_h, V_h: each (seq, d)
            Tensor Q_h = Tensor::zeros(N, d);
            Tensor K_h = Tensor::zeros(N, d);
            Tensor V_h = Tensor::zeros(N, d);
            for (size_t s = 0; s < N; ++s) {
                for (size_t dk = 0; dk < d; ++dk) {
                    Q_h[s][dk] = Q[b][s * E + h * d + dk];
                    K_h[s][dk] = K[b][s * E + h * d + dk];
                    V_h[s][dk] = V[b][s * E + h * d + dk];
                }
            }

            // Landmark V values: V_L[p][dk] = V_h[landmark_indices_[p]][dk]
            Tensor V_L = Tensor::zeros(m, d);
            for (size_t p = 0; p < m; ++p) {
                size_t idx = landmark_indices_[p];
                for (size_t dk = 0; dk < d; ++dk)
                    V_L[p][dk] = V_h[idx][dk];
            }

            // Cache V_L for backward
            for (size_t p = 0; p < m; ++p)
                for (size_t dk = 0; dk < d; ++dk)
                    last_landmark_v[b * H + h][p * d + dk] = V_L[p][dk];

            // Landmark Q and K values
            Tensor Q_L = Tensor::zeros(m, d);
            Tensor K_L = Tensor::zeros(m, d);
            for (size_t p = 0; p < m; ++p) {
                size_t idx = landmark_indices_[p];
                for (size_t dk = 0; dk < d; ++dk) {
                    Q_L[p][dk] = Q_h[idx][dk];
                    K_L[p][dk] = K_h[idx][dk];
                }
            }

            // S_tilde = K_L @ Q_L^T * scale: (m, m)
            Tensor S_tilde = Tensor::zeros(m, m);
            for (size_t p = 0; p < m; ++p) {
                for (size_t q = 0; q < m; ++q) {
                    double s = 0.0;
                    for (size_t dk = 0; dk < d; ++dk)
                        s += K_L[p][dk] * Q_L[q][dk];
                    S_tilde[p][q] = s * scale_;
                }
            }

            // S_bar = K_L @ Q_h^T * scale: (m, N)
            Tensor S_bar = Tensor::zeros(m, N);
            for (size_t p = 0; p < m; ++p) {
                for (size_t j = 0; j < N; ++j) {
                    double s = 0.0;
                    for (size_t dk = 0; dk < d; ++dk)
                        s += K_L[p][dk] * Q_h[j][dk];
                    S_bar[p][j] = s * scale_;
                }
            }

            // A_tilde = softmax(S_tilde, dim=-1): (m, m)
            Tensor A_tilde = softmax(S_tilde);
            // A_bar = softmax(S_bar, dim=-1): (m, N)
            Tensor A_bar = softmax(S_bar);

            // Store A_bar for backward
            for (size_t p = 0; p < m; ++p)
                for (size_t j = 0; j < N; ++j)
                    last_A_bar[b * H + h][p * N + j] = A_bar[p][j];

            // Store A_tilde for backward
            for (size_t p = 0; p < m; ++p)
                for (size_t q = 0; q < m; ++q)
                    last_A_tilde[b * H + h][p * m + q] = A_tilde[p][q];

            // Solve A_tilde @ P = A_bar → P (m, N)
            Tensor P_mat = lu_solve_head(A_tilde, A_bar);

            // Store P for backward
            for (size_t p = 0; p < m; ++p)
                for (size_t j = 0; j < N; ++j)
                    last_P[b * H + h][p * N + j] = P_mat[p][j];

            // Normalize P rows to sum to 1 (A_bar rows sum to 1 after softmax)
            // output_h = P_mat^T @ V_L: (N, d)
            // output_h[j][dk] = sum_p P_mat[p][j] * V_L[p][dk]
            Tensor output_h = Tensor::zeros(N, d);
            for (size_t j = 0; j < N; ++j) {
                // Compute row sum of P_mat for normalization
                double p_row_sum = 0.0;
                for (size_t p = 0; p < m; ++p)
                    p_row_sum += P_mat[p][j];
                p_row_sum = std::max(p_row_sum, 1e-300);
                double inv_sum = 1.0 / p_row_sum;

                for (size_t dk = 0; dk < d; ++dk) {
                    double v = 0.0;
                    for (size_t p = 0; p < m; ++p)
                        v += P_mat[p][j] * V_L[p][dk];
                    output_h[j][dk] = v * inv_sum;
                }
            }

            // Accumulate into output_accum at head offset
            for (size_t s = 0; s < N; ++s)
                for (size_t dk = 0; dk < d; ++dk)
                    output_accum[b][s * E + h * d + dk] += output_h[s][dk];
        }
    }

    last_output_accum = output_accum;

    // Final output projection: output_accum @ W_o^T
    Tensor output = Tensor::zeros(batch, seq * E);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t j = 0; j < E; ++j) {
                double v = 0.0;
                for (size_t k = 0; k < E; ++k)
                    v += output_accum[b][s * E + k] * W_o[k][j];
                output[b][s * E + j] = v;
            }
        }
    }

    return output;
}

Tensor NystromAttention::backward(const Tensor& grad_output, double) {
    // grad_output: (batch, seq * E) flat
    size_t batch = grad_output.rows;
    size_t seq = grad_output.cols / embed_dim_;
    size_t E = embed_dim_;

    // grad_proj = dL/d(output_accum) = grad_out @ W_o
    // output = output_accum @ W_o^T, so dL/doutput_accum = grad_out @ W_o
    Tensor grad_proj = Tensor::zeros(batch, seq * E);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t i = 0; i < E; ++i) {
                double v = 0.0;
                for (size_t k = 0; k < E; ++k)
                    v += grad_output[b][s * E + k] * W_o[i][k];
                grad_proj[b][s * E + i] = v;
            }
        }
    }

    // grad_W_o = last_output_accum^T @ grad_out
    // (E, B*seq) @ (B*seq, E) = (E, E)
    // In flat form: grad_W_o[i][j] = sum_{b,s} last_output_accum[b][s*E+i] * grad_output[b][s*E+j]
    for (size_t i = 0; i < E; ++i) {
        for (size_t j = 0; j < E; ++j) {
            double v = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s)
                    v += last_output_accum[b][s * E + i] * grad_output[b][s * E + j];
            }
            grad_W_o[i][j] += v;
        }
    }

    // grad_W_q = last_query^T @ grad_proj (chain rule: Q = last_query @ W_q^T)
    // grad_W_k = last_key^T @ grad_proj
    // grad_W_v = last_value^T @ grad_proj
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < E; ++i) {
            for (size_t j = 0; j < E; ++j) {
                double gq = 0.0, gk = 0.0, gv = 0.0;
                for (size_t s = 0; s < seq; ++s) {
                    gq += last_query[b][s * E + i] * grad_proj[b][s * E + j];
                    gk += last_key[b][s * E + i] * grad_proj[b][s * E + j];
                    gv += last_value[b][s * E + i] * grad_proj[b][s * E + j];
                }
                grad_W_q[i][j] += gq;
                grad_W_k[i][j] += gk;
                grad_W_v[i][j] += gv;
            }
        }
    }

    // grad_input = grad_proj @ W_q^T (backprop through Q projection)
    Tensor grad_input = Tensor::zeros(batch, seq * E);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t i = 0; i < E; ++i) {
                double v = 0.0;
                for (size_t k = 0; k < E; ++k)
                    v += grad_proj[b][s * E + k] * W_q[i][k];
                grad_input[b][s * E + i] = v;
            }
        }
    }

    return grad_input;
}