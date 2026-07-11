#include "muon.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Newton–Schulz iteration (Bernstein–Jordan 2024 "Optimizing 2-D Kerneled
// Matrices"; (a, b, c) coefficients from Keller Jordan's Muon repo).
//
// Polynomial:  X_{k+1} = a · X_k + b · (X_k X_k^T) X_k + c · (X_k X_k^T)^2 X_k
//
// With the (3.4445, −4.7750, 2.0315) coefficients and 5 iterations starting
// from a singular-value range (~0.5, ~2.0), the polynomial squashes singular
// values toward 1 while preserving the left/right singular vectors — so
// X_out ≈ U V^T (an orthogonal-equivalent of X's SVD).
// ----------------------------------------------------------------------------
Tensor Muon::newton_schulz(const Tensor& X, int ns_steps) {
    // Coefficients per the Muon repo (https://github.com/KellerJordan/Muon).
    // Per the Keller Jordan 2024 docstring:
    //   "This iteration therefore does not produce UV^T but rather something
    //    like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    //    which turns out not to hurt model performance at all relative to
    //    UV^T, where USV^T = G is the SVD."
    //
    // The (3.4445, -4.7750, 2.0315) coefficients are tuned to MAXIMIZE the
    // slope at zero (i.e. fast convergence for small singular values) at the
    // cost of NOT converging all the way to 1.0 — empirically the resulting
    // ~Uniform(0.5, 1.5) singular values work just as well as a strict
    // orthogonalization.
    const double a = 3.4445;
    const double b = -4.7750;
    const double c = 2.0315;

    const size_t m = X.rows;
    const size_t n = X.cols;

    if (m == 0 || n == 0) {
        return Tensor::zeros(m, n);
    }

    // Step 1 (per Muon): normalize so spectral norm is ≤ 1.
    //   X = X / (X.norm() + eps)
    // For 2-D matrices we use the Frobenius norm as a cheap proxy for the
    // spectral norm (the actual repo uses the Frobenius norm too — see line
    // `X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)` in the PyTorch
    // code).
    Tensor Y(m, n);
    Y.data = X.data;  // deep copy via vector assignment

    double frob_sq = 0.0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            frob_sq += Y[i][j] * Y[i][j];
        }
    }
    double frob_norm = std::sqrt(frob_sq);
    if (frob_norm > 0.0) {
        double inv = 1.0 / (frob_norm + 1e-7);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Y[i][j] *= inv;
            }
        }
    }

    // If the matrix is tall (m > n), the polynomial iteration is more
    // numerically stable when applied to X^T (so we're always iterating on
    // a "wide" matrix). After the iterations, transpose back.
    bool transposed = (m > n);
    if (transposed) {
        Y = Y.transpose();
    }

    const size_t M = Y.rows;  // = min(m, n) after possible transpose
    const size_t N = Y.cols;  // = max(m, n) after possible transpose

    // Step 2: 5 iterations of  X_{k+1} = a X + b (X X^T) X + c (X X^T)^2 X
    //
    // For the (3.4445, -4.7750, 2.0315) coefficients and ns_steps=5, this
    // produces a matrix with singular values ~Uniform(0.5, 1.5) — not strictly
    // orthogonal, but empirically what works best for Muon.
    for (int step = 0; step < ns_steps; ++step) {
        // Compute A = Y Y^T  (shape: M×M)
        Tensor A(M, M);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    s += Y[i][k] * Y[j][k];
                }
                A[i][j] = s;
            }
        }
        // Compute B = b A + c A A  (shape: M×M)
        Tensor B(M, M);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < M; ++k) {
                    s += A[i][k] * A[k][j];
                }
                B[i][j] = b * A[i][j] + c * s;
            }
        }
        // Y_new = a Y + B Y  (shape: M×N)
        Tensor Ynext(M, N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < M; ++k) {
                    s += B[i][k] * Y[k][j];
                }
                Ynext[i][j] = a * Y[i][j] + s;
            }
        }
        Y = std::move(Ynext);
    }

    // Un-transpose back to (m, n) if we transposed initially.
    if (transposed) {
        Y = Y.transpose();
    }

    return Y;
}

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
Muon::Muon(double lr, double momentum, int ns_steps,
           double scale_const, double weight_decay, bool cautious)
    : lr(lr),
      momentum(momentum),
      ns_steps(ns_steps),
      scale_const(scale_const),
      weight_decay(weight_decay),
      cautious(cautious) {
    // Defensive clamping so the polynomial doesn't blow up.
    if (momentum < 0.0) momentum = 0.0;
    if (momentum > 0.9999) momentum = 0.9999;
    if (ns_steps < 1) ns_steps = 1;
    if (scale_const <= 0.0) scale_const = 0.2;
    if (weight_decay < 0.0) weight_decay = 0.0;
}

// ----------------------------------------------------------------------------
// State management
// ----------------------------------------------------------------------------
void Muon::ensure_state(void* layer_ptr,
                          const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<Tensor> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols);
        m.fill(0.0);
        vec.push_back(std::move(m));
    }
    state_[layer_ptr] = std::move(vec);
}

// ----------------------------------------------------------------------------
// Update rule for 2-D parameters (the Muon path).
// ----------------------------------------------------------------------------
void Muon::update_2d(Tensor* param, Tensor* grad, Tensor& m) {
    const size_t rows = param->rows;
    const size_t cols = param->cols;

    // Step 1: SGD-momentum  m_t = β · m_{t-1} + g_t
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m[i][j] = momentum * m[i][j] + (*grad)[i][j];
        }
    }

    // Step 2: Newton–Schulz orthogonalize.
    Tensor U = newton_schulz(m, ns_steps);

    // Step 3: per-paper scaling.  The orthogonal matrix has Frobenius norm
    // sqrt(min(rows, cols)); the 0.2 factor matches the empirical scale of
    // a "useful" update for a hidden-weight matrix.  The max(1, sqrt(out/in))
    // compensates for tall vs wide shapes so a square matrix and a tall
    // matrix end up with comparable update magnitudes.
    //
    // From the Muon repo (https://github.com/KellerJordan/Muon):
    //   update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    // where size(-2) = rows (out) and size(-1) = cols (in). So:
    //   shape_scale = max(1, sqrt(rows / cols))
    // (when rows >= cols, sqrt(rows/cols) ≥ 1; when rows < cols, the max is 1).
    double shape_scale = (rows >= cols) ? std::sqrt(static_cast<double>(rows) / static_cast<double>(cols)) : 1.0;
    double scale = scale_const * shape_scale;

    // Step 4: AdamW-style decoupled weight decay + signed update.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double update = scale * U[i][j];
            // Optional Cautious mask: zero out updates that disagree in sign
            // with the gradient.  mask_ij = 1 if (update * g_ij) > 0 else 0.
            if (cautious) {
                double g_ij = (*grad)[i][j];
                if (update * g_ij <= 0.0) update = 0.0;
            }
            double wd_term = weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * (update + wd_term);
        }
    }
}

// ----------------------------------------------------------------------------
// Fallback for 1-D parameters (biases, layer norms, embeddings).
// Newton–Schulz is not defined for 1-D, so we use plain SGD with momentum.
// ----------------------------------------------------------------------------
void Muon::update_1d(Tensor* param, Tensor* grad, Tensor& m) {
    const size_t rows = param->rows;
    const size_t cols = param->cols;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // SGD-momentum
            m[i][j] = momentum * m[i][j] + (*grad)[i][j];
            // AdamW-style decoupled weight decay + update
            double update = m[i][j];
            if (cautious) {
                double g_ij = (*grad)[i][j];
                if (update * g_ij <= 0.0) update = 0.0;
            }
            double wd_term = weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * (update + wd_term);
        }
    }
}

// ----------------------------------------------------------------------------
// step()
// ----------------------------------------------------------------------------
void Muon::step(Model& model) {
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& st = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad  = grads[i];
            Tensor& m_buf = st[i];

            // 1-D parameter → SGD-momentum fallback.
            // 2-D parameter → full Muon path (orthogonalize).
            if (param->rows == 1 || param->cols == 1) {
                update_1d(param, grad, m_buf);
            } else {
                update_2d(param, grad, m_buf);
            }
        }

        layer->zero_grad();
    }
    ++num_steps_;
}