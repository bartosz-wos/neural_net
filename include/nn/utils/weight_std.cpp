#include "weight_std.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace nn {
namespace utils {
namespace {

void require_matrix(const Tensor& w, const char* who) {
    if (w.rows == 0 || w.cols == 0)
        throw std::invalid_argument(std::string(who) +
                                    ": the weight matrix must have at least one "
                                    "row and one column");
    // A variance needs a divisor, and the population divisor is K. At K == 1 the
    // only row is constant, so its variance is identically 0 and WS is 0/0
    // whatever eps is. Refuse rather than return a plausible-looking zero row.
    if (w.cols < 2)
        throw std::invalid_argument(std::string(who) +
                                    ": each channel needs at least 2 weights for a "
                                    "variance, so the weight matrix must have at "
                                    "least 2 columns");
}

void require_eps(double eps, const char* who) {
    if (!std::isfinite(eps))
        throw std::invalid_argument(std::string(who) +
                                    ": eps must be finite");
    if (eps < 0.0)
        throw std::invalid_argument(std::string(who) +
                                    ": eps must be non-negative");
}

// Mean and standard deviation of one row. Returns the scale sqrt(var + eps) and
// writes the mean out. This is the ONE place the population divisor 1/K lives;
// computing it in three separate loops is how a 1/(K-1) slip gets in.
double row_mean_scale(const double* row, size_t n, double eps, double& mean_out) {
    double s = 0.0;
    for (size_t j = 0; j < n; ++j) s += row[j];
    mean_out = s / (double)n;
    double acc = 0.0;
    for (size_t j = 0; j < n; ++j) {
        const double d = row[j] - mean_out;
        acc += d * d;
    }
    // POPULATION variance: divide by n, never by (n - 1). See the header.
    const double var = acc / (double)n;
    const double denom = var + eps;
    if (!(denom > 0.0))
        throw std::invalid_argument(
            "ws_standardize: a channel has zero variance, so the standard "
            "deviation is 0/0 and the transform is undefined; pass eps > 0 to "
            "define it as 0");
    return std::sqrt(denom);
}

}  // namespace

Tensor ws_standardize(const Tensor& w, double eps) {
    require_matrix(w, "ws_standardize");
    require_eps(eps, "ws_standardize");

    Tensor out(w.rows, w.cols);
    for (size_t c = 0; c < w.rows; ++c) {
        double mean = 0.0;
        const double s = row_mean_scale(w[c], w.cols, eps, mean);
        for (size_t j = 0; j < w.cols; ++j) out[c][j] = (w[c][j] - mean) / s;
    }
    return out;
}

void ws_standardize_(Tensor& w, double eps) {
    require_matrix(w, "ws_standardize_");
    require_eps(eps, "ws_standardize_");

    // Read the statistics BEFORE writing, row by row: writing row c cannot
    // change row c's own statistics because each row is handled exactly once.
    for (size_t c = 0; c < w.rows; ++c) {
        double mean = 0.0;
        const double s = row_mean_scale(w[c], w.cols, eps, mean);
        for (size_t j = 0; j < w.cols; ++j) w[c][j] = (w[c][j] - mean) / s;
    }
}

Tensor ws_standardize_backward(const Tensor& w, const Tensor& g, double eps) {
    require_matrix(w, "ws_standardize_backward");
    require_eps(eps, "ws_standardize_backward");
    if (g.rows != w.rows || g.cols != w.cols)
        throw std::invalid_argument(
            "ws_standardize_backward: the gradient must have exactly the same "
            "shape as the weight matrix");

    Tensor out(w.rows, w.cols);
    for (size_t c = 0; c < w.rows; ++c) {
        double mean = 0.0;
        const double s = row_mean_scale(w[c], w.cols, eps, mean);
        const double inv_s = 1.0 / s;

        // x_hat is the STANDARDIZED row. It appears in the mean(g * x_hat) term
        // below, and using `u = W - mean(W)` there instead is a clean 1/s error.
        double sum_g = 0.0, sum_gx = 0.0;
        for (size_t j = 0; j < w.cols; ++j) {
            const double x = (w[c][j] - mean) * inv_s;
            sum_g += g[c][j];
            sum_gx += g[c][j] * x;
        }
        const double mean_g = sum_g / (double)w.cols;
        const double mean_gx = sum_gx / (double)w.cols;

        // The 1/s prefactor is OUTSIDE the parenthesis -- see the header.
        for (size_t j = 0; j < w.cols; ++j) {
            const double x = (w[c][j] - mean) * inv_s;
            out[c][j] = (g[c][j] - mean_g - x * mean_gx) * inv_s;
        }
    }
    return out;
}

double ws_channel_lipschitz(const Tensor& w_hat, size_t c) {
    require_matrix(w_hat, "ws_channel_lipschitz");
    if (c >= w_hat.rows)
        throw std::invalid_argument("ws_channel_lipschitz: channel index out of range");
    double s = 0.0;
    for (size_t j = 0; j < w_hat.cols; ++j) s += w_hat[c][j] * w_hat[c][j];
    return std::sqrt(s);
}

double ws_channel_lipschitz_of(const Tensor& w, size_t c, double eps) {
    require_matrix(w, "ws_channel_lipschitz_of");
    require_eps(eps, "ws_channel_lipschitz_of");
    if (c >= w.rows)
        throw std::invalid_argument(
            "ws_channel_lipschitz_of: channel index out of range");
    double mean = 0.0;
    const double s = row_mean_scale(w[c], w.cols, eps, mean);
    // ||w_hat||_2^2 == sum_j ((w_j - mean)/s)^2 == (sum_j u_j^2) / s^2
    //                 == (n * var) / (var + eps).
    const double var = (s * s - eps);
    const double n = (double)w.cols;
    return std::sqrt(n * var / (s * s));
}

double ws_mean_channel_lipschitz(const Tensor& w, double eps) {
    require_matrix(w, "ws_mean_channel_lipschitz");
    double total = 0.0;
    for (size_t c = 0; c < w.rows; ++c) total += ws_channel_lipschitz_of(w, c, eps);
    return total / (double)w.rows;
}

// ===========================================================================
// WeightStandardization
// ===========================================================================

WeightStandardization::WeightStandardization(Layer* inner, double eps)
    : inner_(inner), eps_(eps) {
    if (inner_ == nullptr)
        throw std::invalid_argument(
            "WeightStandardization: the wrapped layer must not be null");
    require_eps(eps_, "WeightStandardization");

    const std::vector<Tensor*> p = inner_->parameters();
    if (p.empty())
        throw std::invalid_argument(
            "WeightStandardization: the wrapped layer exposes no parameters, so "
            "there is nothing to standardize");
    if (p[0]->rows == 0 || p[0]->cols == 0)
        throw std::invalid_argument(
            "WeightStandardization: the wrapped layer's first parameter is empty");

    sync_from_inner();
}

WeightStandardization::~WeightStandardization() = default;

void WeightStandardization::sync_from_inner() {
    const std::vector<Tensor*> p = inner_->parameters();
    v_ = *p[0];
    v_grad_ = Tensor::zeros(v_.rows, v_.cols);
    raw_grad_ = Tensor::zeros(v_.rows, v_.cols);
}

void WeightStandardization::standardize_params_() {
    // The forward runs on the STANDARDIZED weights, so write them into the
    // wrapped layer and keep the RAW copy in v_. snapshot_ records exactly what
    // the forward saw, which is what the adjoint must be evaluated at.
    const std::vector<Tensor*> p = inner_->parameters();
    snapshot_ = *p[0];
    ws_standardize_(*p[0], eps_);
}

Tensor WeightStandardization::forward(const Tensor& input) {
    standardize_params_();
    return inner_->forward(input);
}

Tensor WeightStandardization::backward(const Tensor& grad_output, double lr) {
    // inner_->backward fills gradients()[0] with dL/d(W_hat), because that is the
    // tensor the forward computed with, and it ACCUMULATES into that buffer
    // (core/layer.cpp: `grad_weights += grad_w`). Converting the ACCUMULATED
    // buffer would double-count on the second call: the adjoint is linear in g,
    // so adjoint(2g) = 2*adjoint(g), and adding that to a buffer that already
    // holds adjoint(g) yields 3*adjoint(g), not 2*. This was caught by the S15
    // accumulation test, which is the only check that can see it -- a
    // single-backward test passes either way.
    //
    // So: snapshot the buffer BEFORE the call and convert the INCREMENT. The
    // buffer is kept in RAW space between calls (the tail of this function
    // writes it back), so the units match and the difference is exactly the new
    // contribution.
    const std::vector<Tensor*> p = inner_->parameters();
    const std::vector<Tensor*> gr = inner_->gradients();
    const bool convertible = !gr.empty() && gr[0] != nullptr && p[0] != nullptr &&
                             gr[0]->rows == snapshot_.rows && gr[0]->cols == snapshot_.cols;
    Tensor prev;
    if (convertible) prev = *gr[0];

    const Tensor dinput = inner_->backward(grad_output, lr);

    if (convertible) {
        const Tensor delta = *gr[0] - prev;   // the increment, in standardized space
        v_grad_ = v_grad_ + ws_standardize_backward(snapshot_, delta, eps_);
        raw_grad_ = v_grad_;
        // Keep the inner layer's buffer in RAW space, which is what prev reads
        // on the next call. A caller reading inner().gradients() therefore sees
        // the gradient with respect to the weights it can actually change.
        *gr[0] = v_grad_;
    }
    return dinput;
}

void WeightStandardization::update_weights(double lr) {
    // Apply the RAW update to the RAW tensor. The wrapped layer's own
    // update_weights is deliberately NOT called here: its gradient buffer holds
    // the RAW-space gradient only after the wrapper has written it back, and a
    // Dense would otherwise apply a SECOND update on the standardized tensor,
    // silently doubling the step for one call.
    v_ = v_ - v_grad_ * lr;
    raw_grad_ = v_grad_;
    // Put the raw weights back into the wrapped layer so the next forward
    // standardizes from the updated values.
    const std::vector<Tensor*> p = inner_->parameters();
    *p[0] = v_;
}

void WeightStandardization::zero_grad() {
    inner_->zero_grad();
    v_grad_.fill(0.0);
    raw_grad_.fill(0.0);
}

}  // namespace utils
}  // namespace nn