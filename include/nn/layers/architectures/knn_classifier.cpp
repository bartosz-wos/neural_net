#include "knn_classifier.h"
#include <stdexcept>
#include <numeric>

KNNClassifier::KNNClassifier(size_t feature_dim, size_t num_classes,
                             size_t k, double temperature)
    : feature_dim_(feature_dim),
      num_classes_(num_classes),
      k_(k),
      temperature_(temperature) {
    if (feature_dim == 0) throw std::invalid_argument("KNNClassifier: feature_dim must be > 0");
    if (num_classes == 0) throw std::invalid_argument("KNNClassifier: num_classes must be > 0");
    if (k == 0) throw std::invalid_argument("KNNClassifier: k must be > 0");
    if (temperature <= 0.0) throw std::invalid_argument("KNNClassifier: temperature must be > 0");
}

void KNNClassifier::fit(const Tensor& support_features,
                        const std::vector<int>& support_labels) {
    if (support_features.cols != feature_dim_) {
        throw std::invalid_argument(
            "KNNClassifier::fit: support feature dim mismatch "
            "(expected " + std::to_string(feature_dim_) +
            ", got " + std::to_string(support_features.cols) + ")");
    }
    if (support_features.rows != support_labels.size()) {
        throw std::invalid_argument(
            "KNNClassifier::fit: support_labels size must equal number of support rows");
    }
    for (size_t i = 0; i < support_labels.size(); ++i) {
        if (support_labels[i] < 0 ||
            static_cast<size_t>(support_labels[i]) >= num_classes_) {
            throw std::invalid_argument(
                "KNNClassifier::fit: support_labels[i] out of range");
        }
    }
    // Deep clone so the user can mutate their tensor afterwards without
    // affecting our stored support set.
    support_features_ = support_features.clone();
    support_labels_ = support_labels;
}

Tensor KNNClassifier::forward(const Tensor& input) {
    if (input.cols != feature_dim_) {
        throw std::invalid_argument(
            "KNNClassifier::forward: input feature dim mismatch "
            "(expected " + std::to_string(feature_dim_) +
            ", got " + std::to_string(input.cols) + ")");
    }
    last_input_ = input.clone();

    const size_t batch = input.rows;
    const size_t N_sup = support_features_.rows;

    // Allocate output and per-query caches.
    Tensor output(batch, num_classes_);
    last_topk_idx_.assign(batch, {});
    last_topk_w_.assign(batch, {});
    last_negd2_tau_ = Tensor::zeros(batch, std::max<size_t>(N_sup, 1));

    // Empty-support fast path: uniform probabilities, no weights.
    if (N_sup == 0) {
        const double u = 1.0 / static_cast<double>(num_classes_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t c = 0; c < num_classes_; ++c)
                output(b, c) = u;
        return output;
    }

    const size_t k_eff = std::min(k_, N_sup);
    const double inv_tau = 1.0 / temperature_;

    for (size_t b = 0; b < batch; ++b) {
        // ----------------------------------------------------------------
        // Step 1: squared-Euclidean distance from query b to every support.
        // ----------------------------------------------------------------
        std::vector<double> dist2(N_sup);
        for (size_t i = 0; i < N_sup; ++i) {
            double s = 0.0;
            for (size_t d = 0; d < feature_dim_; ++d) {
                double diff = input(b, d) - support_features_(i, d);
                s += diff * diff;
            }
            dist2[i] = s;
        }

        // ----------------------------------------------------------------
        // Step 2: pick the k smallest via partial sort.
        // We build a list of (dist, index), sort, and take the first k.
        // ----------------------------------------------------------------
        std::vector<std::pair<double, size_t>> order(N_sup);
        for (size_t i = 0; i < N_sup; ++i) {
            order[i] = {dist2[i], i};
        }
        // Use nth_element for O(N) average instead of full sort.
        std::nth_element(order.begin(), order.begin() + k_eff, order.end(),
                         [](const auto& a, const auto& b) {
                             return a.first < b.first;
                         });
        // nth_element leaves the first k in arbitrary order; sort that prefix.
        std::sort(order.begin(), order.begin() + k_eff,
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        // ----------------------------------------------------------------
        // Step 3: softmax over the top-k squared distances (negated, / tau).
        // ----------------------------------------------------------------
        std::vector<double> topk_d2(k_eff);
        std::vector<size_t> topk_idx(k_eff);
        double max_neg = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < k_eff; ++j) {
            topk_d2[j] = order[j].first;
            topk_idx[j] = order[j].second;
            double a = -topk_d2[j] * inv_tau;
            if (a > max_neg) max_neg = a;
        }
        std::vector<double> topk_w(k_eff);
        double sum_w = 0.0;
        for (size_t j = 0; j < k_eff; ++j) {
            double a = -topk_d2[j] * inv_tau;
            topk_w[j] = std::exp(a - max_neg);
            sum_w += topk_w[j];
        }
        if (sum_w > 0.0) {
            for (size_t j = 0; j < k_eff; ++j) topk_w[j] /= sum_w;
        } else {
            // Degenerate (all distances equal); fall back to uniform.
            for (size_t j = 0; j < k_eff; ++j) topk_w[j] = 1.0 / static_cast<double>(k_eff);
        }

        // Save caches
        last_topk_idx_[b] = topk_idx;
        last_topk_w_[b] = topk_w;
        for (size_t j = 0; j < k_eff; ++j) {
            // d L / d (-d^2 / tau) starts at zero; we accumulate into it in backward.
            // We pre-store the (negated-distance / tau) value at the selected
            // support slots so backward can recover the distances without
            // recomputing them.
            last_negd2_tau_(b, topk_idx[j]) = -topk_d2[j] * inv_tau;
        }

        // ----------------------------------------------------------------
        // Step 4: accumulate vote weights per class.
        // ----------------------------------------------------------------
        for (size_t j = 0; j < k_eff; ++j) {
            int cls = support_labels_[topk_idx[j]];
            output(b, static_cast<size_t>(cls)) += topk_w[j];
        }
    }

    return output;
}

Tensor KNNClassifier::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (batch, num_classes)
    // Returns grad_input: (batch, feature_dim)
    if (grad_output.rows != last_input_.rows || grad_output.cols != num_classes_) {
        throw std::invalid_argument(
            "KNNClassifier::backward: grad_output shape mismatch "
            "(expected (batch, " + std::to_string(num_classes_) + "))");
    }
    const size_t batch = last_input_.rows;
    const size_t N_sup = support_features_.rows;

    Tensor grad_input(batch, feature_dim_);

    // Empty-support fast path: forward was uniform, so gradient is zero.
    if (N_sup == 0) return grad_input;

    const size_t k_eff = std::min(k_, N_sup);
    const double neg_inv_tau = -1.0 / temperature_;   // d(-d^2/tau)/d(d^2) = -1/tau

    for (size_t b = 0; b < batch; ++b) {
        const std::vector<size_t>& topk_idx = last_topk_idx_[b];
        const std::vector<double>& topk_w   = last_topk_w_[b];

        // ----------------------------------------------------------------
        // 1) d L / d w_j = sum_c grad_output[b, c] * 1[label_{topk_idx[j]} == c]
        //    i.e. pick the column corresponding to the support's label.
        // ----------------------------------------------------------------
        std::vector<double> d_w(k_eff);
        for (size_t j = 0; j < k_eff; ++j) {
            int cls = support_labels_[topk_idx[j]];
            d_w[j] = grad_output(b, static_cast<size_t>(cls));
        }

        // ----------------------------------------------------------------
        // 2) softmax backward over the k logits:
        //    a_j = -d^2_j / tau;   w_j = exp(a_j - M) / Z
        //    d w_j / d a_j = w_j (1 - w_j);   d w_j / d a_m = -w_j w_m (m != j)
        //    d L / d a_j = w_j (d L / d w_j - sum_m w_m * d L / d w_m)
        // ----------------------------------------------------------------
        double sw_dw = 0.0;
        for (size_t j = 0; j < k_eff; ++j) sw_dw += topk_w[j] * d_w[j];

        std::vector<double> d_a(k_eff);
        for (size_t j = 0; j < k_eff; ++j) {
            d_a[j] = topk_w[j] * (d_w[j] - sw_dw);
        }

        // ----------------------------------------------------------------
        // 3) chain to d d^2:
        //    d L / d d^2_j = d L / d a_j * d a_j / d d^2_j = d_a[j] * (-1/tau)
        // ----------------------------------------------------------------
        std::vector<double> d_d2(k_eff);
        for (size_t j = 0; j < k_eff; ++j) {
            d_d2[j] = d_a[j] * neg_inv_tau;
        }

        // ----------------------------------------------------------------
        // 4) d d^2(q, s_i) / d q[d] = 2 * (q[d] - s_i[d])
        //    sum over the k selected support points.
        // ----------------------------------------------------------------
        for (size_t j = 0; j < k_eff; ++j) {
            size_t i = topk_idx[j];
            double g = 2.0 * d_d2[j];
            for (size_t d = 0; d < feature_dim_; ++d) {
                grad_input(b, d) += g * (last_input_(b, d) - support_features_(i, d));
            }
        }
    }

    return grad_input;
}

void KNNClassifier::update_weights(double /*learning_rate*/) {
    // No parameters; support set is fixed during training of upstream layers.
}

void KNNClassifier::zero_grad() {
    // No parameter gradients; zero the cache so accidental reuse is safe.
    last_input_ = Tensor();
    last_topk_idx_.clear();
    last_topk_w_.clear();
    last_negd2_tau_ = Tensor();
}

std::vector<Tensor*> KNNClassifier::parameters() {
    return {};   // no learnable parameters
}

std::vector<Tensor*> KNNClassifier::gradients() {
    return {};
}

Tensor KNNClassifier::get_weights() const {
    return support_features_;
}

Tensor KNNClassifier::get_gradients() const {
    // The "gradient" of a non-parametric layer is the gradient of its input
    // computed during the most recent backward. For convenience we expose
    // it here; it lives in last_input_ (cached) but we don't store the input
    // gradient long-term, so return zeros if not available.
    // (Caller can also just hold the grad_input returned by backward().)
    return Tensor();
}