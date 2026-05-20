#include "nystrom_attention.h"
#include "../../activations/activations.h"
#include <algorithm>
#include <stdexcept>

NystromAttention::NystromAttention(int embed_dim, int num_heads, int num_landmarks, float dropout)
    : embed_dim_(embed_dim), num_heads_(num_heads),
      num_landmarks_(num_landmarks), head_dim_(embed_dim / num_heads),
      dropout_(dropout), scale_(1.0f / std::sqrt(static_cast<float>(head_dim_) + 1e-6f)),
      is_initialized_(false), batch_size_(0), seq_len_(0),
      nystrom_path_used_(false), fallback_path_used_(false)
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
    // Also clear projected Q/K/V caches to prevent stale values during repeated numerical checks
    last_q = Tensor();
    last_k = Tensor();
    last_v = Tensor();
    nystrom_path_used_ = false;
    fallback_path_used_ = false;
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

// Solve A^T @ x = b using already-decomposed A_lu (from lu_decompose)
// pivot is the permutation array returned by lu_decompose
// A_lu contains L and U in the standard format (see lu_solve_head)
static void solve_transposed(const Tensor& A_lu, const std::vector<size_t>& pivot, std::vector<double>& b, std::vector<double>& x) {
    size_t m = A_lu.rows;
    
    // For A^T x = b with pivoted LU: P @ A @ P^T = L @ U
    // A^T = P^T @ U^T @ L^T @ P
    // So A^T x = b => P^T @ U^T @ L^T @ P x = b
    // => U^T @ L^T @ P x = P @ b
    // Let y = P x, solve U^T @ z = P @ b (forward), then L^T @ y = z (backward), then x = P^T @ y
    
    // Step 1: Apply row permutation P to b: b_perm[i] = b[pivot[i]]
    std::vector<double> b_perm(m);
    for (size_t i = 0; i < m; ++i)
        b_perm[i] = b[pivot[i]];
    
    // Step 2: Forward substitution for U^T @ z = b_perm
    // U^T is lower triangular: U^T[i][j] = U[j][i] for j <= i
    // A_lu[j][i] = U[j][i] for j < i, A_lu[i][i] = U[i][i]
    std::vector<double> z(m);
    for (size_t i = 0; i < m; ++i) {
        double val = b_perm[i];
        for (size_t j = 0; j < i; ++j)
            val -= A_lu[j][i] * z[j];
        z[i] = val / A_lu[i][i];
    }
    
    // Step 3: Backward substitution for L^T @ x = z
    // L^T is upper triangular: L^T[i][j] = L[j][i] for j >= i
    // A_lu[j][i] = L[j][i] for j > i, L[i][i] = 1
    // Backward: start from i=m-1 down to 0
    for (int i = (int)m - 1; i >= 0; --i) {
        double val = z[i];
        for (size_t j = i + 1; j < m; ++j)
            val -= A_lu[j][i] * x[j];
        // L^T[i][i] = 1, so x[i] = val
        x[i] = val;
    }
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
        fallback_path_used_ = true;
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
                    qv += query[b][s * E + k] * W_q[k][i];
                    kv += key[b][s * E + k] * W_k[k][i];
                    vv += value[b][s * E + k] * W_v[k][i];
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

    nystrom_path_used_ = true;
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

    // Nyström backward path: accumulates grad_W_q/k/v through the full
    // attention computation. Only runs when Nyström was actually used.
    if (nystrom_path_used_) {
        size_t m = landmark_indices_.size();
        size_t N = seq;
        size_t H = num_heads_;
        size_t d = head_dim_;

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t h = 0; h < H; ++h) {
                // Extract per-head Q, K, V
                Tensor Q_h(N, d), K_h(N, d), V_h(N, d);
                for (size_t s = 0; s < N; ++s) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        Q_h[s][dk] = last_q[b][s * E + h * d + dk];
                        K_h[s][dk] = last_k[b][s * E + h * d + dk];
                        V_h[s][dk] = last_v[b][s * E + h * d + dk];
                    }
                }

                // Extract landmark Q, K, V
                Tensor Q_L(m, d), K_L(m, d), V_L(m, d);
                for (size_t p = 0; p < m; ++p) {
                    size_t idx = landmark_indices_[p];
                    for (size_t dk = 0; dk < d; ++dk) {
                        Q_L[p][dk] = Q_h[idx][dk];
                        K_L[p][dk] = K_h[idx][dk];
                        V_L[p][dk] = V_h[idx][dk];
                    }
                }

                // Extract cached matrices
                Tensor A_bar(m, N), A_tilde(m, m), P_mat(m, N);
                for (size_t p = 0; p < m; ++p) {
                    for (size_t j = 0; j < N; ++j)
                        A_bar[p][j] = last_A_bar[b * H + h][p * N + j];
                    for (size_t qq = 0; qq < m; ++qq)
                        A_tilde[p][qq] = last_A_tilde[b * H + h][p * m + qq];
                    for (size_t j = 0; j < N; ++j)
                        P_mat[p][j] = last_P[b * H + h][p * N + j];
                }

                // Step 1: Backward through row_norm and P^T @ V_L
                // output_h[j][dk] = sum_p P_mat[p][j] * V_L[p][dk] / p_row_sum[j]
                Tensor grad_P(m, N), grad_V_L(m, d);
                for (size_t j = 0; j < N; ++j) {
                    double p_row_sum = 0.0;
                    for (size_t p = 0; p < m; ++p)
                        p_row_sum += P_mat[p][j];
                    p_row_sum = std::max(p_row_sum, 1e-300);
                    double inv_sum = 1.0 / p_row_sum;
                    for (size_t p = 0; p < m; ++p) {
                        for (size_t dk = 0; dk < d; ++dk) {
                            double grad_out_h = grad_proj[b][j * E + h * d + dk];
                            grad_P[p][j] += grad_out_h * V_L[p][dk] * inv_sum;
                            grad_V_L[p][dk] += grad_out_h * P_mat[p][j] * inv_sum;
                        }
                    }
                }

                // Step 2: Backward through P = lu_solve(A_tilde, A_bar)
                // P = A_tilde^{-1} @ A_bar
                // grad_A_bar = A_tilde^{-T} @ grad_P
                Tensor grad_A_bar(m, N);
                {
                    Tensor A_lu = A_tilde.clone();
                    std::vector<size_t> pivot = lu_decompose(A_lu, m);
                    std::vector<double> b_rhs(m), x(m);
                    for (size_t col = 0; col < N; ++col) {
                        for (size_t i = 0; i < m; ++i)
                            b_rhs[i] = grad_P[i][col];
                        solve_transposed(A_lu, pivot, b_rhs, x);
                        for (size_t i = 0; i < m; ++i)
                            grad_A_bar[i][col] = x[i];
                    }
                }

                // Step 3: Backward through A_bar = softmax(S_bar)
                // grad_S_bar = A_bar * (grad_A_bar - row_sum(grad_A_bar * A_bar))
                // Note: scale_ is NOT included here; it will be applied when computing grad_K_L
                // (scale_ was applied in forward: S_bar = scale * K_L @ Q_h^T)
                Tensor grad_S_bar(m, N);
                for (size_t p = 0; p < m; ++p) {
                    double row_sum = 0.0;
                    for (size_t qq = 0; qq < N; ++qq)
                        row_sum += grad_A_bar[p][qq] * A_bar[p][qq];
                    for (size_t j = 0; j < N; ++j)
                        grad_S_bar[p][j] = A_bar[p][j] * (grad_A_bar[p][j] - row_sum);
                }

                // Step 4: Backward through S_bar = scale * K_L @ Q_h^T
                // grad_K_L += scale * grad_S_bar @ Q_h (scale applied here)
                // grad_Q_h_attn += scale * grad_S_bar^T @ K_L (scale must be applied here)
                Tensor grad_K_L = Tensor::zeros(m, d);
                Tensor grad_Q_h_attn = Tensor::zeros(N, d);
                for (size_t p = 0; p < m; ++p) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double gkl = 0.0;
                        for (size_t j = 0; j < N; ++j)
                            gkl += grad_S_bar[p][j] * Q_h[j][dk];
                        grad_K_L[p][dk] += gkl * scale_;
                    }
                }
                // grad_Q_h[j][dk] = scale * sum_p grad_S_bar[p][j] * K_L[p][dk]
                // (scale was applied in forward: S_bar = scale * K_L @ Q_h^T)
                for (size_t j = 0; j < N; ++j) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double gqh = 0.0;
                        for (size_t p = 0; p < m; ++p)
                            gqh += grad_S_bar[p][j] * K_L[p][dk];
                        // BUG FIX: multiply by scale_ (was missing)
                        grad_Q_h_attn[j][dk] += gqh * scale_;
                    }
                }

                // Step 5: Backward through P = A_tilde^{-1} @ A_bar for A_tilde
                // P = A_tilde^{-1} @ A_bar, and dL/dA_tilde = -A^{-T} @ dL/dP @ P^T
                // grad_A_tilde = -A_tilde^{-T} @ grad_P @ P_mat^T
                // Compute M = grad_P @ P_mat^T: (m, N) @ (N, m) = (m, m)
                Tensor M(m, m);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < m; ++j) {
                        double v = 0.0;
                        for (size_t k = 0; k < N; ++k)
                            v += grad_P[i][k] * P_mat[j][k];
                        M[i][j] = v;
                    }
                }
                // Solve A_tilde^T @ X = M for X (each column solved independently via transpose LU solve)
                // grad_A_tilde = -X, where X satisfies A_tilde^T @ X[:,p] = M[:,p]
                Tensor grad_A_tilde(m, m);
                {
                    Tensor A_lu_at = A_tilde.clone();
                    std::vector<size_t> pivot_at = lu_decompose(A_lu_at, m);
                    std::vector<double> b_rhs(m), x(m);
                    for (size_t col = 0; col < m; ++col) {
                        for (size_t i = 0; i < m; ++i)
                            b_rhs[i] = M[i][col];
                        solve_transposed(A_lu_at, pivot_at, b_rhs, x);
                        for (size_t i = 0; i < m; ++i)
                            grad_A_tilde[i][col] = -x[i];
                    }
                }

                // Step 5b: Backward through A_tilde = softmax(S_tilde)
                // grad_S_tilde = A_tilde * (grad_A_tilde - row_sum(grad_A_tilde * A_tilde))
                // Note: scale_ NOT included; will be applied in grad_Q_L computation
                Tensor grad_S_tilde(m, m);
                for (size_t p = 0; p < m; ++p) {
                    double row_sum = 0.0;
                    for (size_t r = 0; r < m; ++r)
                        row_sum += grad_A_tilde[p][r] * A_tilde[p][r];
                    for (size_t qq = 0; qq < m; ++qq)
                        grad_S_tilde[p][qq] = A_tilde[p][qq] * (grad_A_tilde[p][qq] - row_sum);
                }

                // Step 6: Backward through S_tilde = scale * K_L @ Q_L^T
                // grad_K_L += scale * grad_S_tilde @ Q_L
                // grad_Q_L += grad_S_tilde^T @ K_L
                for (size_t p = 0; p < m; ++p) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double gkl = 0.0;
                        for (size_t qq = 0; qq < m; ++qq)
                            gkl += grad_S_tilde[p][qq] * Q_L[qq][dk];
                        grad_K_L[p][dk] += gkl * scale_;
                    }
                }
                // grad_Q_L[q][dk] = sum_p grad_S_tilde[p][q] * K_L[p][dk]
                Tensor grad_Q_L(m, d);
                for (size_t qq = 0; qq < m; ++qq) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double gq = 0.0;
                        for (size_t p = 0; p < m; ++p)
                            gq += grad_S_tilde[p][qq] * K_L[p][dk];
                        grad_Q_L[qq][dk] = gq;
                    }
                }

                // Step 7: Scatter landmark gradients to full sequences
                // grad_Q_h = grad_Q_h_attn (from S_bar) + grad_Q_L scatter (from S_tilde)
                // grad_K_h = grad_K_L scatter (from S_bar and S_tilde, K doesn't have direct attention path)
                Tensor grad_Q_h(N, d), grad_K_h(N, d);
                for (size_t j = 0; j < N; ++j)
                    for (size_t dk = 0; dk < d; ++dk)
                        grad_Q_h[j][dk] = grad_Q_h_attn[j][dk];
                for (size_t p = 0; p < m; ++p) {
                    size_t idx = landmark_indices_[p];
                    for (size_t dk = 0; dk < d; ++dk) {
                        grad_Q_h[idx][dk] += grad_Q_L[p][dk];
                        grad_K_h[idx][dk] += grad_K_L[p][dk];
                    }
                }
                // grad_V_h: zero-initialized, accumulate from all positions
                // Both landmark positions (via grad_V_L scatter) and non-landmark
                // positions (via direct grad_proj contribution in Step 1)
                Tensor grad_V_h(N, d);
                for (size_t p = 0; p < m; ++p) {
                    size_t idx = landmark_indices_[p];
                    for (size_t dk = 0; dk < d; ++dk)
                        grad_V_h[idx][dk] += grad_V_L[p][dk];
                }

                // Step 8: Aggregate into W_q, W_k, W_v
                for (size_t s = 0; s < N; ++s) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        size_t pos = h * d + dk;
                        double gqh_val = grad_Q_h[s][dk];
                        double gkh_val = grad_K_h[s][dk];
                        double gvh_val = grad_V_h[s][dk];
                        for (size_t k = 0; k < E; ++k) {
                            grad_W_q[k][pos] += last_query[b][s * E + k] * gqh_val;
                            grad_W_k[k][pos] += last_key[b][s * E + k] * gkh_val;
                            grad_W_v[k][pos] += last_value[b][s * E + k] * gvh_val;
                        }
                    }
                }
            }
        }
    }

    else if (fallback_path_used_) {
        // Fallback path: standard softmax attention
        // The fallback path computes standard softmax attention:
        //   A = softmax(Q_h @ K_h^T * scale_)
        //   output_h = A @ V_h
        size_t N = seq;
        size_t H = num_heads_;
        size_t d = head_dim_;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < H; ++h) {
                // Extract Q_h, K_h, V_h from projected Q/K/V caches
                Tensor Q_h(N, d), K_h(N, d), V_h(N, d);
                for (size_t s = 0; s < N; ++s) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        Q_h[s][dk] = last_q[b][s * E + h * d + dk];
                        K_h[s][dk] = last_k[b][s * E + h * d + dk];
                        V_h[s][dk] = last_v[b][s * E + h * d + dk];
                    }
                }

                // S = Q_h @ K_h^T * scale_: (N, N)
                Tensor S(N, N);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        double s = 0.0;
                        for (size_t dk = 0; dk < d; ++dk)
                            s += Q_h[i][dk] * K_h[j][dk];
                        S[i][j] = s * scale_;
                    }
                }

                // A = softmax(S)
                Tensor A(N, N);
                for (size_t i = 0; i < N; ++i) {
                    double max_val = S[i][0];
                    for (size_t j = 1; j < N; ++j)
                        if (S[i][j] > max_val) max_val = S[i][j];
                    double sum_exp = 0.0;
                    for (size_t j = 0; j < N; ++j) {
                        A[i][j] = std::exp(S[i][j] - max_val);
                        sum_exp += A[i][j];
                    }
                    sum_exp = std::max(sum_exp, 1e-300);
                    for (size_t j = 0; j < N; ++j)
                        A[i][j] /= sum_exp;
                }

                // dL/dV_h = A^T @ dL/doutput_h, where dL/doutput_h = grad_proj[b][:, h*d:]
                Tensor dL_dV(N, d);
                for (size_t j = 0; j < N; ++j) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double v = 0.0;
                        for (size_t i = 0; i < N; ++i)
                            v += A[i][j] * grad_proj[b][j * E + h * d + dk];
                        dL_dV[j][dk] = v;
                    }
                }

                // dL/dA[i][j] = sum_dk dL/doutput_h[j][dk] * V_h[i][dk]
                // dL/dS = softmax_backward(A, dL/dA)
                Tensor dL_dA(N, N);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        double v = 0.0;
                        for (size_t dk = 0; dk < d; ++dk)
                            v += grad_proj[b][j * E + h * d + dk] * V_h[i][dk];
                        dL_dA[i][j] = v;
                    }
                }
                // dL/dS = A * (dL/dA - row_sum(dL/dA * A))
                Tensor dL_dS(N, N);
                for (size_t i = 0; i < N; ++i) {
                    double row_sum = 0.0;
                    for (size_t j = 0; j < N; ++j)
                        row_sum += dL_dA[i][j] * A[i][j];
                    for (size_t j = 0; j < N; ++j)
                        dL_dS[i][j] = A[i][j] * (dL_dA[i][j] - row_sum);
                }

                // dL/dQ_h = dL/dS @ K_h * scale_
                Tensor dL_dQ(N, d);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double q = 0.0;
                        for (size_t j = 0; j < N; ++j)
                            q += dL_dS[i][j] * K_h[j][dk];
                        dL_dQ[i][dk] = q * scale_;
                    }
                }

                // dL/dK_h = dL/dS^T @ Q_h * scale_
                Tensor dL_dK(N, d);
                for (size_t j = 0; j < N; ++j) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        double k = 0.0;
                        for (size_t i = 0; i < N; ++i)
                            k += dL_dS[i][j] * Q_h[i][dk];
                        dL_dK[j][dk] = k * scale_;
                    }
                }

                // Accumulate into W_q, W_k, W_v
                for (size_t s = 0; s < N; ++s) {
                    for (size_t dk = 0; dk < d; ++dk) {
                        size_t pos = h * d + dk;
                        double dq_val = dL_dQ[s][dk];
                        double dk_val = dL_dK[s][dk];
                        double dv_val = dL_dV[s][dk];
                        for (size_t k = 0; k < E; ++k) {
                            grad_W_q[k][pos] += last_query[b][s * E + k] * dq_val;
                            grad_W_k[k][pos] += last_key[b][s * E + k] * dk_val;
                            grad_W_v[k][pos] += last_value[b][s * E + k] * dv_val;
                        }
                    }
                }
            }
        }
    }

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