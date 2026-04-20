#include "triplet_loss_siamese.h"
#include <cmath>
#include <algorithm>

// === EmbeddingNormalizer ===

Tensor EmbeddingNormalizer::forward(const Tensor& x) {
    int batch = (int)x.rows;
    int dim = (int)x.cols;
    Tensor out(batch, dim);

    for (int b = 0; b < batch; b++) {
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            norm += x[b][i] * x[b][i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8) norm = 1.0;
        for (int i = 0; i < dim; i++) {
            out[b][i] = x[b][i] / norm;
        }
    }
    return out;
}

Tensor EmbeddingNormalizer::backward(const Tensor& grad_output, double) {
    // Gradient through normalization is non-trivial (chain rule with norm).
    // Mark trainable = false to prevent gradient updates on this layer.
    (void)grad_output;
    return Tensor(1, 1); // zero-sized dummy; trainable_ = false prevents backprop
}

// === SiameseNetwork ===

Tensor SiameseNetwork::forward(const Tensor& x) {
    if (base_network_ == nullptr) return x;
    return base_network_->forward(x);
}

Tensor SiameseNetwork::backward(const Tensor& grad_output, double) {
    return grad_output;
}

void SiameseNetwork::set_base_network(Layer* base_network) {
    base_network_ = base_network;
}

std::pair<Tensor, Tensor> SiameseNetwork::forward_pair(const Tensor& x1, const Tensor& x2) {
    if (base_network_ == nullptr) return {x1, x2};
    Tensor e1 = base_network_->forward(x1);
    Tensor e2 = base_network_->forward(x2);
    return {e1, e2};
}

// === TripletLoss ===

Tensor TripletLoss::normalize_embeddings(const Tensor& embeddings) const {
    int batch = (int)embeddings.rows;
    int dim = (int)embeddings.cols;
    Tensor out(batch, dim);

    for (int b = 0; b < batch; b++) {
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            norm += embeddings[b][i] * embeddings[b][i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8) norm = 1.0;
        for (int i = 0; i < dim; i++) {
            out[b][i] = embeddings[b][i] / norm;
        }
    }
    return out;
}

float TripletLoss::forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative) {
    int batch = (int)anchor.rows;
    int embedding_dim = (int)anchor.cols;

    Tensor a = normalize_ ? normalize_embeddings(anchor) : anchor;
    Tensor p = normalize_ ? normalize_embeddings(positive) : positive;
    Tensor n = normalize_ ? normalize_embeddings(negative) : negative;

    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        float d_ap = 0.0f, d_an = 0.0f;
        for (int i = 0; i < embedding_dim; i++) {
            float diff_p = a[b][i] - p[b][i];
            float diff_n = a[b][i] - n[b][i];
            d_ap += diff_p * diff_p;
            d_an += diff_n * diff_n;
        }
        d_ap = std::sqrt(d_ap);
        d_an = std::sqrt(d_an);
        total_loss += std::max(0.0f, d_ap - d_an + margin_);
    }
    return total_loss / batch;
}

// === ContrastiveLoss ===

Tensor ContrastiveLoss::normalize_embeddings(const Tensor& embeddings) const {
    int batch = (int)embeddings.rows;
    int dim = (int)embeddings.cols;
    Tensor out(batch, dim);

    for (int b = 0; b < batch; b++) {
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            norm += embeddings[b][i] * embeddings[b][i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8) norm = 1.0;
        for (int i = 0; i < dim; i++) {
            out[b][i] = embeddings[b][i] / norm;
        }
    }
    return out;
}

float ContrastiveLoss::euclidean_distance(const Tensor& a, const Tensor& b) {
    int dim = (int)a.cols;
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[0][i] - b[0][i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

float ContrastiveLoss::forward(const Tensor& emb1, const Tensor& emb2, float label) {
    Tensor e1 = normalize_ ? normalize_embeddings(emb1) : emb1;
    Tensor e2 = normalize_ ? normalize_embeddings(emb2) : emb2;
    float d = euclidean_distance(e1, e2);
    if (label > 0.5f) {
        return d * d;
    } else {
        return std::max(0.0f, margin_ - d) * std::max(0.0f, margin_ - d);
    }
}

// === HardTripletMiner ===

TripletBatch HardTripletMiner::mine(const std::vector<Tensor>& embeddings,
                                     const std::vector<int>& labels) {
    TripletBatch batch;
    int n = (int)embeddings.size();
    int emb_dim = n > 0 ? (int)embeddings[0].cols : 0;

    std::vector<float> dists(n * n, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float d = 0.0f;
            for (int k = 0; k < emb_dim; k++) {
                float diff = embeddings[i][0][k] - embeddings[j][0][k];
                d += diff * diff;
            }
            dists[i * n + j] = std::sqrt(d);
        }
    }

    for (int a = 0; a < n; a++) {
        float max_d_pos = -1.0f;
        int hardest_pos = -1;
        for (int p = 0; p < n; p++) {
            if (labels[p] == labels[a] && p != a) {
                if (dists[a * n + p] > max_d_pos) {
                    max_d_pos = dists[a * n + p];
                    hardest_pos = p;
                }
            }
        }

        float min_d_neg = 1e9f;
        int hardest_neg = -1;
        for (int neg = 0; neg < n; neg++) {
            if (labels[neg] != labels[a]) {
                if (dists[a * n + neg] < min_d_neg) {
                    min_d_neg = dists[a * n + neg];
                    hardest_neg = neg;
                }
            }
        }

        if (hardest_pos >= 0 && hardest_neg >= 0) {
            float loss = max_d_pos - min_d_neg + margin_;
            if (loss > 0) {
                batch.anchors.push_back(a);
                batch.positives.push_back(hardest_pos);
                batch.negatives.push_back(hardest_neg);
            }
        }
    }

    return batch;
}
